"""Customer creation through the A7 metadata-ticket lifecycle.

Prepare dry-runs the same core as commit in a savepoint. The shared A1 layer
owns identity, permission rechecks, supersession, expiry, hashes and replay.
The legacy customer routes remain unchanged.
"""
import json
import unicodedata
from datetime import datetime
from typing import Optional

from fastapi import Depends, HTTPException, Request
from psycopg2.extras import Json
from pydantic import constr, root_validator, validator

import permissions
import write_tickets as wt

FIELDS = ('name', 'contact_name', 'email', 'phone', 'address', 'notes', 'notes_es')


class CreateCustomerRequest(wt.IdInput):
    # Same fields and required name as the direct CustomerCreate request.
    name: constr(strict=True, min_length=1)
    contact_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    notes: Optional[str] = None
    notes_es: Optional[str] = None
    client_source: wt.ClientSource = 'api'

    @validator('name')
    def real_name(cls, value):
        name = ' '.join(value.split())
        if not name:
            wt.fail(422, 'CUSTOMER_NAME_REQUIRED', 'Enter the customer name.')
        return name

    @root_validator(pre=True)
    def authenticated_identity_only(cls, values):
        if isinstance(values, dict) and any(k in values for k in ('by', 'changed_by', 'created_by')):
            wt.fail(422, 'SELF_REPORTED_IDENTITY_REJECTED',
                    'The person is the authenticated key; do not supply a creator.')
        return values


COMPANY_SUFFIXES = frozenset(('inc', 'llc', 'co', 'corp', 'ltd'))


def duplicate_key(name):
    # Normalize spelling variants only; Sunrise and Sunshine remain different.
    # Drop abbreviation dots/apostrophes and separate other punctuation so
    # Bakery,Inc., Bakery (Inc.) and dotted L.L.C. share the same identity.
    normalized = ''.join('' if char in ".'’" else ' ' if unicodedata.category(char).startswith('P')
                         else char for char in name.casefold())
    words = normalized.split()
    while len(words) > 1 and words[-1] in COMPANY_SUFFIXES:
        words.pop()
    return ' '.join(words)


def duplicate(cur, name):
    # Use one Python normalization for input, legacy names and both alias
    # catalogs. Database regex whitespace/case semantics differ by locale.
    # Include inactive customers: a second identity cannot reactivate one.
    normalized = duplicate_key(name)
    cur.execute("""SELECT id,name,active,name AS match_name FROM customers
                   UNION ALL
                   SELECT c.id,c.name,c.active,ca.alias FROM customers c
                   JOIN customer_aliases ca ON ca.customer_id=c.id
                   UNION ALL
                   SELECT c.id,c.name,c.active,sa.alias FROM customers c
                   JOIN search_aliases sa ON sa.customer_id=c.id
                   WHERE sa.kind='customer' AND sa.active
                   ORDER BY active DESC NULLS LAST,id""")
    row = next((row for row in cur.fetchall() if duplicate_key(row['match_name']) == normalized), None)
    if not row:
        return None
    existing = {field: row[field] for field in ('id', 'name', 'active')}
    advice = ('Use this customer instead.' if row['active'] else
              'Ask the owner to reactivate this customer instead.')
    return {'code': 'CUSTOMER_ALREADY_EXISTS',
            'message': f"Customer '{row['name']}' already exists (customer #{row['id']}). {advice}",
            'message_es': f"El cliente '{row['name']}' ya existe (cliente #{row['id']}). "
                          + ('Use este cliente.' if row['active'] else 'Pida al dueño que lo reactive.'),
            'existing_customer': existing}


def post(api, cur, payload, request, *, ticket_id):
    api.validate_bilingual(payload.get('notes'), payload.get('notes_es'), 'notes')
    # A normalized unique index would change the legacy direct routes. This
    # short transaction-scoped catalog lock instead serializes both ticket and
    # direct INSERT/UPDATE writes, including the first creation of a name.
    cur.execute('LOCK TABLE customers IN SHARE ROW EXCLUSIVE MODE')
    existing = duplicate(cur, payload['name'])
    if existing:
        wt.fail(409, existing['code'], existing['message'], blockers=[existing])
    cur.execute("""INSERT INTO customers(name,contact_name,email,phone,address,notes,notes_es,ticket_id)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                   RETURNING id,name,contact_name,email,phone,address,notes,notes_es,active,ticket_id""",
                tuple(payload.get(field) for field in FIELDS) + (ticket_id,))
    customer = dict(cur.fetchone())
    api._record_actor_write(cur, request, 'customers', customer['id'])
    return {'customer_id': customer['id'], 'name': customer['name'], 'customer': customer,
            'message': f"Customer '{customer['name']}' created"}


def commit(api, cur, row, body, actor, request, entry_timing):
    if body.lot_confirmations:
        wt.fail(422, 'UNUSED_LOT_CONFIRMATION', 'Creating a customer does not use ingredient lots.')
    if body.attachment_ref:
        wt.fail(422, 'UNUSED_ATTACHMENT', 'Creating a customer does not use photo evidence.')
    cur.execute('SAVEPOINT ticket_post')
    try:
        if actor['id'] is not None:
            cur.execute('SELECT active FROM actors WHERE id=%s FOR SHARE', (actor['id'],))
            current = cur.fetchone()
            if not (current and current['active']):
                wt.fail(403, 'ACTOR_INACTIVE', 'The preparing actor is no longer active.')
        unresolved = row['draft'].get('blockers', [])
        if unresolved:
            wt.fail(409, 'DRAFT_BLOCKED', 'Prepare again after resolving the draft blockers.', blockers=unresolved)
        response = post(api, cur, row['payload'], request, ticket_id=row['id'])
        day = datetime.fromisoformat(row['payload']['occurred_at']).astimezone(api.PLANT_TIMEZONE).date()
        receipt = wt.allocate_receipt(cur, 'create_customer', day)
        response.update(receipt_number=receipt, ticket_id=row['id'], replayed=False,
                        state_changed=False, success=True)
        cur.execute('SELECT clock_timestamp() AS at')
        response['entry_timing'] = {**entry_timing, 'entered_at': cur.fetchone()['at'],
                                    'entered_by': actor, 'late_entry_exception_id': None}
        response = wt.json_value(api, response)
    except HTTPException as exc:
        if exc.status_code >= 500:
            raise
        errors = exc.detail.get('blockers') if isinstance(exc.detail, dict) else None
        errors = errors or [wt.blocker(exc)]
        cur.execute('ROLLBACK TO SAVEPOINT ticket_post')
        cur.execute("UPDATE write_tickets SET status='rejected',reject_reason=%s WHERE id=%s",
                    (json.dumps(errors), row['id']))
        return api.JSONResponse(status_code=409, content={'detail': {
            'error_code': 'TICKET_STALE', 'message': 'Draft no longer valid; prepare again.',
            'blockers': errors}})
    cur.execute('RELEASE SAVEPOINT ticket_post')
    result_ref = {'customer_id': response['customer_id'], 'transaction_ids': [], 'lot_ids': []}
    cur.execute("""UPDATE write_tickets SET status='committed',committed_at=clock_timestamp(),
                   receipt_number=%s,result_ref=%s,response=%s,acknowledged=%s WHERE id=%s""",
                (receipt, Json(result_ref), Json(response), Json(sorted(set(body.acknowledged_warnings))), row['id']))
    return response


def receipt_customer(cur, ticket):
    if ticket['action'] != 'create_customer':
        return None
    customer_id = (ticket.get('result_ref') or {}).get('customer_id')
    cur.execute('SELECT id,name,active,ticket_id FROM customers WHERE id=%s', (customer_id,))
    row = cur.fetchone()
    return dict(row) if row else {'id': customer_id}


def register_routes(app, api):
    @app.post('/customers/prepare')
    def prepare_customer(req: CreateCustomerRequest, request: Request,
                         _: bool = Depends(api.verify_api_key)):
        actor = wt.identity(api, request)
        permissions.require('create_customer', actor)
        payload = json.loads(req.json(exclude={'client_source'}))
        now = api.get_plant_now()
        # A7 metadata entries carry the current plant day, never a backdate.
        payload['occurred_at'] = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        draft = {'customer': {field: payload.get(field) for field in FIELDS},
                 'summary': f"Create customer {payload['name']}",
                 'summary_es': f"Crear el cliente {payload['name']}",
                 'entry_timing': permissions.timing(now, now)}
        blockers = []
        with api.get_transaction() as cur:
            cur.execute('SAVEPOINT customer_draft')
            try:
                post(api, cur, payload, request, ticket_id=None)
            except HTTPException as exc:
                if exc.status_code >= 500:
                    raise
                blockers = exc.detail.get('blockers') if isinstance(exc.detail, dict) else None
                blockers = blockers or [wt.blocker(exc)]
            finally:
                cur.execute('ROLLBACK TO SAVEPOINT customer_draft')
            return wt.issue_ticket(api, cur, actor=actor, action='create_customer', payload=payload,
                                   draft=draft, state={}, warnings=[], blockers=blockers,
                                   client_source=req.client_source, event_time=now)
