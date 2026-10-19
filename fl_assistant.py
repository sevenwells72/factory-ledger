"""F1 transport: FL owns every business decision; only a human Record commits.

No network hop to another FL deployment, process-local ticket store, model prose
renderer, hosted MCP, browser OpenAI key, or model-accessible commit tool.
"""
import asyncio
import json
import logging
import threading
import os
import re
from urllib.parse import quote
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

import httpx
from fastapi import Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, StrictInt, StrictStr
from psycopg2.extras import Json

import assistant_runtime as runtime
import assistant_tools as catalog
import write_tickets
import resolution
import assistant_replies as replies
import assistant_service
from assistant_intents import office_handoff, is_receive_continuation, customer_creation, customer_only
from assistant_voice import normalize_voice_text, normalize_order_code
from assistant_receive import merge_receive_lot_facts, normalize_receive_arguments, package_facts, split_receive_items

ROUTES = ('session', 'resume', 'turn', 'record', 'cancel', 'confirm-sku', 'attachment', 'attachment/read', 'transcribe')
ACTOR_ROUTES = frozenset({('POST', '/assistant/' + p) for p in ROUTES} |
                         {('GET', '/correction-reasons')})
MAX_PHOTO = 5 * 1024 * 1024
MAX_AUDIO = 20 * 1024 * 1024
MAX_STEPS = 8
MODEL = 'gpt-6.1-sol'  # PoC choice; service may explicitly override it.
TRANSCRIBE_MODEL = 'gpt-transcribe'
logger = logging.getLogger(__name__)


def voice_lot_code(value):
    """Normalize a supplied spoken code without treating its label as identity."""
    normalized = normalize_voice_text(value)
    label = re.match(r'^\s*(?:lot|lote)(?:\s+(?:number|n[uú]mero))?\s+(.+)$', value, re.I)
    # Literal existing codes such as LOT 1234 stay literal; strip a dictation
    # label only when spoken words actually changed within the supplied code.
    if label and normalized != value:
        value = label.group(1)
    return normalize_voice_text('lot ' + value)[4:]


def order_number_in_text(text):
    """Recognize order identity before the language model selects any tool."""
    normalized = normalize_order_code(text)
    code = re.search(r'\bSO-\d{6}-\d{3,}\b', normalized, re.I)
    if code:
        return code
    # Malformed separators still denote an order, but ordinary wording such
    # as "received two cases, so 10 lb total" is not an order identifier.
    return next((match for match in re.finditer(r'\b(?:s\s*o|ese\s+o)[ -]*\d[\d -]*', normalized, re.I)
                 if sum(char.isdigit() for char in match[0]) >= 9), None)


def pack_without_batch(text):
    return (bool(re.search(r'\b(?:pack(?:ed)?|empacamos|empacado|empaque)\b', text, re.I))
            and not re.search(r'\b(?:lot|lote|batch|tanda)\b|\b(?:from|desde|del)\b', text, re.I))


def fail(status, code, message):
    raise HTTPException(status, {'error_code': code, 'message': message})


def argument_validation_error(status, result):
    """Only schema/transport mistakes are retried, never FL business refusals."""
    if status != 422 or not isinstance(result, dict):
        return False
    detail = result.get('detail')
    return (isinstance(detail, list)
            or (result.get('error_detail') or {}).get('code') == 'VALIDATION_ERROR'
            or (isinstance(detail, dict) and detail.get('error_code') in {
                'TOOL_ARGUMENTS_INVALID', 'INVALID_RECEIPT', 'RESOLVE_REQUIRED', 'ORDER_REQUIRED',
                'IDS_REQUIRED', 'AMBIGUOUS_EVENT_TIME'}))


def missing_prepare_card(name, status, result, arguments):
    """Keep user omissions out of the model's formatting-repair budget."""
    if name not in catalog.PREPARES or status != 422 or not isinstance(result, dict):
        return None
    errors = result.get('detail')
    correction = name in ('prepare_adjust', 'prepare_found')
    if correction and isinstance(errors, dict) and errors.get('error_code') == 'REASON_CODE_REQUIRED':
        return {'kind': 'refusal', 'status': status, 'result': catalog.safe_result(result), 'reason_required': True}
    if not isinstance(errors, list) or not errors:
        return None
    def missing_reason(error):
        return (correction and isinstance(error, dict) and error.get('loc') == ['body', 'reason_code']
                and error.get('type') in ('missing', 'string_too_short')
                and (arguments.get('reason_code') is None
                     or isinstance(arguments.get('reason_code'), str) and not arguments['reason_code'].strip()))
    reason_required = correction and (arguments.get('reason_code') is None
        or isinstance(arguments.get('reason_code'), str) and not arguments['reason_code'].strip())
    if any(not isinstance(error, dict) or error.get('type') != 'missing' and not missing_reason(error) for error in errors):
        return None
    locations = [error.get('loc') for error in errors]
    if any(not isinstance(loc, (list, tuple)) or len(loc) < 2 or loc[0] != 'body' for loc in locations):
        return None
    if reason_required and all(list(loc) == ['body', 'reason_code'] for loc in locations):
        return {'kind': 'refusal', 'status': status, 'result': catalog.safe_result(result), 'reason_required': True}
    fields = []
    for loc in locations:
        if reason_required and list(loc) == ['body', 'reason_code']:
            fields.append('reason_code')
            continue
        value = arguments
        try:
            for key in loc[1:]:
                value = value[key]
        except (KeyError, IndexError, TypeError):
            return None
        if value is not None or not isinstance(loc[-1], str):
            return None
        if loc[-1] not in fields:
            fields.append(loc[-1])
    card = {'kind': 'missing_facts', 'fields': fields}
    if reason_required:
        card['reason_required'] = True
        if 'reason_code' not in fields:
            fields.append('reason_code')
    if name == 'prepare_pack' and 'source_product_id' in fields:
        card['packing_batch_needed'] = True
    return card


class Body(BaseModel):
    class Config:
        extra = 'forbid'


class SessionBody(Body):
    session_id: UUID


class TurnBody(SessionBody):
    turn_id: UUID
    text: str = Field('', max_length=4000)
    language: Literal['en', 'es'] | None = None
    choice_id: UUID | None = None
    selected_id: StrictInt | StrictStr | None = None
    next_page: bool = False
    attachment_ids: list[write_tickets.PositiveId] = Field(default_factory=list, max_items=5)


class DraftBody(Body):
    draft_id: UUID


class RecordBody(DraftBody):
    acknowledged_warnings: list[str] = Field(default_factory=list, max_items=25)
    # Opaque FL evidence; validated only by CommitRequest/A5, never inferred here.
    lot_confirmations: list[dict] = Field(default_factory=list, max_items=100)


class AttachmentRead(Body):
    attachment_id: write_tickets.PositiveId


async def openai(path, *, body=None, file=None):
    key = os.getenv('OPENAI_API_KEY', '').strip()
    if not key:
        fail(503, 'ASSISTANT_NOT_CONFIGURED', 'OpenAI is unavailable.')
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
            kwargs = {'json': body} if file is None else {
                'files': {'file': file},
                'data': {'model': os.getenv('ASSISTANT_TRANSCRIBE_MODEL', TRANSCRIBE_MODEL), 'response_format': 'json'}}
            response = await client.post('https://api.openai.com/v1/' + path,
                                         headers={'Authorization': 'Bearer ' + key}, **kwargs)
        if response.status_code != 200:
            try:
                problem = response.json()
            except ValueError:
                problem = {}
            if assistant_service.quota_exhausted(response.status_code, problem):
                logger.error('ASSISTANT_QUOTA_EXHAUSTED: provider billing/credit limit; status=%s', response.status_code)
                raise HTTPException(503, {'error_code': assistant_service.CODE,
                    'message': assistant_service.MESSAGE, 'message_es': assistant_service.MESSAGE_ES})
            logger.warning('OpenAI request failed: status=%s', response.status_code)
            fail(502, 'OPENAI_UNAVAILABLE', 'OpenAI request failed.')
        return response.json()
    except (httpx.HTTPError, ValueError):
        fail(502, 'OPENAI_UNAVAILABLE', 'OpenAI connection failed.')


async def relay(app, request, method, path, *, body=None, params=None, file=None, form=None, binary=False):
    """Invoke actual FL routes including their auth, validation and error envelopes.

    A small ASGI client avoids logging the ticket in an HTTP client's URL log.
    Destination is the current application, never a configurable remote URL.
    """
    from urllib.parse import urlencode, unquote
    if file is not None:
        multipart = httpx.Request(method, 'http://fl-internal' + path,
                                  files={'file': file}, data=form or {})
        content = multipart.read()
        content_type = multipart.headers['content-type'].encode()
    else:
        content = json.dumps(body, default=str).encode() if body is not None else b''
        content_type = b'application/json'
    headers = [(b'x-api-key', request.headers.get('x-api-key', '').encode()),
               (b'content-type', content_type), (b'content-length', str(len(content)).encode())]
    scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
             'method': method, 'scheme': 'http', 'path': unquote(path), 'raw_path': path.encode(),
             'query_string': urlencode(params or {}).encode(), 'root_path': '',
             'headers': headers, 'server': ('fl-internal', 80), 'client': ('127.0.0.1', 0)}
    if '_fl_verified_identity' in request.scope:
        scope['_fl_verified_identity'] = request.scope['_fl_verified_identity']
    messages, sent = [], False
    finished = asyncio.Event()

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {'type': 'http.request', 'body': content, 'more_body': False}
        await finished.wait()
        return {'type': 'http.disconnect'}

    async def send(message):
        messages.append(message)
        if message['type'] == 'http.response.body' and not message.get('more_body', False):
            finished.set()

    await app(scope, receive, send)
    start = next(m for m in messages if m['type'] == 'http.response.start')
    status = start['status']
    raw = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
    if binary and status == 200:
        return status, (raw, dict(start['headers']))
    try:
        result = json.loads(raw)
    except ValueError:
        fail(502, 'FL_RESPONSE_INVALID', 'FL returned no usable response.')
    return status, result


def session_row(api, session_id, actor_id, *, lock=False):
    with api.get_transaction() as cur:
        cur.execute('SELECT * FROM assistant_sessions WHERE id=%s AND actor_id=%s' +
                    (' FOR UPDATE' if lock else ''), (str(session_id), actor_id))
        row = cur.fetchone()
    if not row:
        fail(404, 'SESSION_NOT_FOUND', 'Chat not found for this person / Chat no encontrado para esta persona.')
    return row


def draft_row(api, draft_id, actor_id, *, lock=False, cur=None):
    if cur is None:
        with api.get_transaction() as cursor:
            return draft_row(api, draft_id, actor_id, lock=lock, cur=cursor)
    cur.execute('SELECT * FROM assistant_drafts WHERE id=%s AND actor_id=%s' +
                (' FOR UPDATE' if lock else ''), (str(draft_id), actor_id))
    row = cur.fetchone()
    if not row:
        fail(404, 'DRAFT_NOT_FOUND', 'Draft not found for this person / Borrador no encontrado.')
    return row


def linked_attachments(api, session_id, actor_id, attachment_ids, *, cur=None):
    """Chat provenance only; A6 owns evidence storage and ticket-context checks."""
    if cur is None:
        with api.get_transaction() as cursor:
            return linked_attachments(api, session_id, actor_id, attachment_ids, cur=cursor)
    found = []
    for aid in attachment_ids:
        cur.execute('''SELECT a.id,a.kind FROM assistant_attachment_links l
            JOIN attachments a ON a.id=l.attachment_id
            JOIN assistant_sessions s ON s.id=l.session_id
            WHERE l.attachment_id=%s AND l.session_id=%s AND l.actor_id=%s
              AND s.actor_id=%s AND a.uploaded_by_actor_id=%s''',
            (aid, str(session_id), actor_id, actor_id, actor_id))
        row = cur.fetchone()
        if not row:
            fail(404, 'ATTACHMENT_NOT_FOUND', 'Attachment unavailable / Adjunto no disponible.')
        found.append(dict(row))
    return found


def evidence_fields(action, selected):
    # Evidence comes from the page's authenticated uploads, never model output.
    allowed = (('bol', 'lot_tag') if action == 'receive' else
               ('bol', 'truck_photo', 'signed_packing_slip') if action in write_tickets.SHIP_ACTIONS else
               ('correction_photo',) if action in ('adjust', 'found') else ())
    if any(photo['kind'] not in allowed for photo in selected):
        help_en, help_es = (
            ('Use a BOL or lot-tag photo for a receive.', 'Usa una foto del BOL o de la etiqueta del lote para una recepción.')
            if action == 'receive' else
            ('Use a BOL, truck or signed packing-slip photo for a shipment.', 'Usa una foto del BOL, del camión o de la nota de empaque firmada para un envío.')
            if action in write_tickets.SHIP_ACTIONS else
            ('Use a correction photo for this entry.', 'Usa una foto de corrección para esta entrada.')
            if action in ('adjust', 'found') else
            ('This entry does not accept photos.', 'Esta entrada no acepta fotos.'))
        raise HTTPException(422, {'error_code': 'ATTACHMENT_KIND_UNSUPPORTED',
            'message': 'This photo is not attached to an entry. Remove it or change its type before preparing the draft. ' + help_en,
            'message_es': 'Esta foto no está adjunta a una entrada. Quítala o cambia su tipo antes de preparar el borrador. ' + help_es})
    if action == 'receive':
        ids = [a['id'] for a in selected]
        return {'attachment_ids': ids} if ids else {}
    if action in write_tickets.SHIP_ACTIONS:
        return {'attachment_ids': [a['id'] for a in selected
                                  if a['kind'] in ('bol', 'truck_photo', 'signed_packing_slip')]}
    if action in ('adjust', 'found'):
        photos = [a for a in selected if a['kind'] == 'correction_photo']
        if len(photos) > 1:
            fail(422, 'CORRECTION_PHOTO_AMBIGUOUS', 'Select one correction photo / Selecciona una foto de corrección.')
        return {'attachment_id': photos[0]['id']} if photos else {}
    return {}


def record_attachment_ids(card):
    # Pre-A6 UUID photos cannot silently disappear from a new receipt.
    # Already committed receipt recovery skips evidence processing entirely.
    selected = []
    for aid in card.get('attachment_ids', []):
        if isinstance(aid, int) and not isinstance(aid, bool) and aid > 0:
            selected.append(aid)
        else:
            try:
                UUID(str(aid))
            except ValueError:
                fail(409, 'ATTACHMENT_REUPLOAD_REQUIRED', 'Upload this photo again / Vuelve a subir esta foto.')
            fail(409, 'ATTACHMENT_REUPLOAD_REQUIRED',
                 'This older photo cannot be attached to a new receipt. Upload it again before recording / '
                 'Esta foto anterior no se puede adjuntar a un recibo nuevo. Vuelve a subirla antes de registrar.')
    return selected


def approve_id(state, kind, candidate):
    values = state.setdefault('resolved', {}).setdefault(kind, [])
    if candidate.get('id') not in values:
        values.append(candidate['id'])


def untrusted_ids(args, state):
    """Transport provenance only. FL still validates identity and eligibility."""
    kinds = {'product_id': 'product', 'source_product_id': 'product', 'target_product_id': 'product',
             'ingredient_product_id': 'product', 'substitute_product_id': 'product',
             'supplier_id': 'supplier', 'customer_id': 'customer', 'order_id': 'order', 'lot_id': 'lot'}
    missing = []
    order_lines = state.get('resolved', {}).get('order_line', {}).get(str(args.get('order_id')), [])

    def walk(value):
        if isinstance(value, dict):
            for name, item in value.items():
                if name in kinds and item is not None and item not in state.get('resolved', {}).get(kinds[name], []):
                    missing.append(name)
                if name == 'line_id' and item is not None and item not in order_lines:
                    missing.append(name)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(args)
    return missing


def receipt_ok(status, result):
    return (status == 200 and isinstance(result, dict) and result.get('success') is not False
            and isinstance(result.get('receipt_number'), str)
            and re.fullmatch(r'(?:RCV|MK|PK|ADJ|FND|SHP|CUS|ORD)-\d{6}-\d{3,}', result['receipt_number']) is not None)


def register_routes(app, api):
    def authenticate(request: Request, _: bool = Depends(api.verify_api_key)):
        if os.getenv('ASSISTANT_ENABLED') != '1':
            fail(503, 'ASSISTANT_DISABLED', 'FL Assistant is not enabled / FL Assistant no está habilitado.')
        actor = api.request_actor(request)
        if not actor:
            fail(403, 'PERSONAL_IDENTITY_REQUIRED', 'Sign in with your personal key / Usa tu clave personal.')
        return actor

    reason_cache = {'source': None, 'data': None}
    reason_lock = threading.Lock()

    def cached_reasons():
        # The reason catalog is fixed reference data. Separate test/database
        # connections never share a cached catalog with another source.
        source = (api.get_db_connection, os.getenv('DATABASE_URL'))
        with reason_lock:
            if reason_cache['source'] != source or reason_cache['data'] is None:
                with api.get_transaction() as cur:
                    cur.execute('SELECT code,label_en,label_es,applies_to,adjust_sign,note_required FROM correction_reasons ORDER BY sort_order,code')
                    data = {'reasons': [dict(r) for r in cur.fetchall()]}
                reason_cache.update(source=source, data=data)
            return reason_cache['data']

    @app.get('/correction-reasons')
    def correction_reasons(_: bool = Depends(api.verify_api_key)):
        return cached_reasons()

    @app.get('/dash/fl-assistant', include_in_schema=False)
    def page():
        if os.getenv('ASSISTANT_ENABLED') != '1':
            raise HTTPException(404, 'Not found')
        return FileResponse(Path(__file__).parent / 'dashboard' / 'fl-assistant.html',
                            headers={'Cache-Control': 'no-store'})

    @app.post('/assistant/session')
    def create_session(actor=Depends(authenticate)):
        sid = str(uuid4())
        with api.get_transaction() as cur:
            cur.execute('INSERT INTO assistant_sessions(id,actor_id) VALUES (%s,%s)', (sid, actor['id']))
        return {'session_id': sid, 'actor': {'id': actor['id'], 'name': actor['name'], 'role': actor['role']}}

    @app.post('/assistant/resume')
    def resume(body: SessionBody, actor=Depends(authenticate)):
        row = session_row(api, body.session_id, actor['id'])
        with api.get_transaction() as cur:
            cur.execute('SELECT response,created_at FROM assistant_turns WHERE session_id=%s ORDER BY created_at,id', (str(body.session_id),))
            # The page dates its history so old replies never look current.
            turns = [{**r['response'], 'created_at': r['created_at'].isoformat()} for r in cur.fetchall()]
            cur.execute('SELECT id,status,result,card,record_started_at FROM assistant_drafts WHERE session_id=%s ORDER BY created_at,id', (str(body.session_id),))
            drafts = [dict(r) for r in cur.fetchall()]
        return {'session_id': str(body.session_id), 'turns': turns, 'drafts': drafts,
                'pending': row['state'].get('pending'),
                'actor': {'id': actor['id'], 'name': actor['name'], 'role': actor['role']}}

    async def approve_candidate(request, state, kind, candidate):
        if kind != 'order':
            approve_id(state, kind, candidate)
            return 200, None
        # Resolve names first, then obtain line identifiers from this actor's
        # real order read. A model-supplied line ID is never its own provenance.
        status, detail = await relay(app, request, 'GET', '/sales/orders/' + str(candidate['id']))
        if status != 200:
            return status, detail
        if not isinstance(detail, dict) or detail.get('order_id') != candidate['id'] or not isinstance(detail.get('lines'), list):
            fail(502, 'FL_RESPONSE_INVALID', 'FL returned no usable order detail / Detalle de pedido no disponible.')
        approve_id(state, kind, candidate)
        state.setdefault('order_summaries', {})[str(candidate['id'])] = {
            'order_id': candidate['id'], 'order_number': detail['order_number'],
            'remaining_lb': float(detail.get('remaining_effective_lb') or 0),
            'items': [{'remaining_lb': line.get('readiness', {}).get('remaining_lb', line.get('remaining_lb', 0)),
                       'case_size_lb': line.get('case_size_lb')}
                      for line in detail['lines']
                      if not line.get('is_service') and not line.get('is_non_weight')
                      and line.get('line_status') != 'cancelled']}
        state['resolved'].setdefault('order_line', {})[str(candidate['id'])] = [
            line['line_id'] for line in detail['lines']
            if line.get('line_status') not in ('cancelled', 'fulfilled')]
        return status, detail

    async def supplier_choices(request, state, query='', product_id=None):
        # Search/rank the display only. Supplier records and IDs remain intact.
        status, data = await relay(app, request, 'GET', '/suppliers', params={'q': query} if query else {})
        if status != 200:
            return []
        with api.get_transaction() as cur:
            cur.execute("""SELECT raw.supplier_id,count(*) AS uses,max(raw.timestamp) AS latest
                FROM ledger_current_transactions t JOIN transactions raw ON raw.id=t.id
                JOIN ledger_current_transaction_lines l ON l.transaction_id=t.id
                WHERE t.effective_status='posted' AND t.type='receive'
                  AND l.product_id=%s AND raw.supplier_id IS NOT NULL
                GROUP BY raw.supplier_id""", (product_id,))
            history = {r['supplier_id']: (int(r['uses']), str(r['latest'])) for r in cur.fetchall()}
            cur.execute('SELECT DISTINCT supplier_id FROM supplier_product_aliases WHERE product_id=%s', (product_id,))
            related = {r['supplier_id'] for r in cur.fetchall()}
        def display_key(name):
            name = re.sub(r'[^a-z0-9]+', ' ', name.lower().replace('&', ' and ')).strip()
            name = re.sub(r'(?:\s+(?:incorporated|corporation|company|limited|inc|corp|llc|ltd|co))+$', '', name)
            name = re.sub(r'\s+(?:supply|supplies)$', '', name)
            return re.sub(r'\bbakery\b', 'baker', name)
        candidates = [dict(id=r['id'], name=r['name'], label=r['name']) for r in data.get('suppliers', [])
                      if resolution.normalize(r['name']) not in resolution.SENTINEL_SUPPLIERS]
        if query:
            exact = [r for r in candidates if display_key(r['name']) == display_key(query)]
            literal = [r for r in candidates if resolution.normalize(query) in resolution.normalize(r['name'])]
            candidates = exact or literal or candidates
        candidates.sort(key=lambda r: (-(r['id'] in history), -(r['id'] in related),
            -history.get(r['id'], (0, ''))[0], -int(bool(query) and resolution.normalize(r['name']) == resolution.normalize(query)),
            r['name'].lower(), r['id']))
        unique = {}
        for row in candidates:
            unique.setdefault(display_key(row['name']), row)
        return list(unique.values())[:6]

    def supplier_choice_card(state, candidates, query=''):
        choice_id = str(uuid4())
        result = {'candidates': candidates, 'has_more': False, 'query': query}
        state['pending'] = {'id': choice_id, 'args': {'kind': 'supplier', 'query': query},
                            'result': result, 'supplier_search': True}
        return {'kind': 'choices', 'id': choice_id, 'choice_id': choice_id,
                'resolution_kind': 'supplier', 'supplier_search': True, 'result': result}

    def catalog_voice_text(value):
        normalized = normalize_voice_text(value)
        if normalized != value and re.search(r'\b(?:lot|lote)\b', value, re.I):
            # A literal tag can itself contain spoken words. Preserve catalog
            # identity before sending the normalized utterance to the model.
            with api.get_transaction() as cur:
                cur.execute('SELECT lot_code FROM lots WHERE position(lower(lot_code) in lower(%s)) > 0', (value,))
                codes = [row['lot_code'] for row in cur.fetchall()]
            return normalize_voice_text(value, protected_codes=codes)
        return normalized

    def catalog_lot_code(value):
        normalized = voice_lot_code(value)
        if normalized != value:
            with api.get_transaction() as cur:
                cur.execute('SELECT lot_code FROM lots WHERE lower(lot_code)=lower(%s) LIMIT 1', (value,))
                exact = cur.fetchone()
            if exact:
                return exact['lot_code']
        return normalized

    def packing_output_choices(state, context):
        with api.get_transaction() as cur:
            cur.execute("SELECT id,name,case_size_lb FROM products WHERE parent_batch_product_id=%s AND type='finished' AND active IS DISTINCT FROM false ORDER BY name,id", (context['source_product_id'],))
            outputs = [dict(row, case_size_lb=float(row['case_size_lb']) if row['case_size_lb'] is not None else None) for row in cur.fetchall()]
        if not outputs:
            return {'kind': 'missing_facts', 'fields': ['target_product_id']}
        choice_id = str(uuid4())
        result = {'candidates': outputs, 'has_more': False}
        state['pending'] = {'id': choice_id, 'args': {'kind': 'product', 'query': '', 'context': {'action': 'pack'}},
                            'result': result, 'packing_output_context': context}
        return {'kind': 'choices', 'id': choice_id, 'resolution_kind': 'product', 'packing_outputs': True, 'result': result}

    def packing_lot_choices(state, target, query=''):
        source_id = target['parent_batch_product_id']
        def core(name):
            return sorted(word for word in re.findall(r'[a-z0-9]+', resolution.catalog_query(name).lower())
                          if word not in {'batch', 'granola', 'and'})
        with api.get_transaction() as cur:
            if source_id:
                cur.execute('SELECT id,name FROM products WHERE id=%s AND active IS DISTINCT FROM false', (source_id,))
                sources = [dict(row) for row in cur.fetchall()]
            else:
                cur.execute("SELECT id,name FROM products WHERE type='batch' AND active IS DISTINCT FROM false ORDER BY name,id")
                sources = [dict(row) for row in cur.fetchall() if core(row['name']) == core(target['name'])]
            candidates = []
            for source in sources:
                for row in api.fifo_lot_balances(cur, source['id']):
                    candidates.append({'id': row['id'], 'name': row['lot_code'], 'lot_code': row['lot_code'],
                        'product_id': source['id'], 'source_product_name': source['name'],
                        'lot_date': str(row['lot_date']),
                        'label': f"{source['name']} · {row['lot_code']} · {replies.number(row['available'])} lb"})
        candidates.sort(key=lambda row: (row['lot_date'], row['id']))
        if not candidates and not sources:
            return None
        choice_id = str(uuid4())
        result = {'candidates': candidates, 'has_more': False, 'query': query}
        state['pending'] = {'id': choice_id, 'args': {'kind': 'lot', 'query': query}, 'result': result,
            'packing_context': {'target_product_id': target['id'],
                'case_size_lb': float(target['case_size_lb']) if target['case_size_lb'] is not None else None},
            'packing_target_invalid': target['type'] != 'finished'}
        return {'kind': 'choices', 'id': choice_id, 'resolution_kind': 'lot', 'query': query,
                'packing_lots': True, 'result': result}

    def resolved_packing_lots(state, product_id):
        with api.get_transaction() as cur:
            cur.execute('SELECT id,name,type,case_size_lb,parent_batch_product_id FROM products WHERE id=%s',
                        (product_id,))
            target = cur.fetchone()
        if target and target['type'] != 'batch':
            return packing_lot_choices(state, target)
        return None

    async def shipping_order(request, body, actor, state, args):
        query = normalize_order_code(args.get('query', ''))
        code = re.search(r'\bSO-\d{6}-\d{3,}\b', query, re.I)
        typed = re.search(r'\b(?:s\s*o|ese\s+o)[ -]*\d[\d -]*', query, re.I)
        identifier = code[0] if code else typed[0].strip() if typed else None
        status, data = await relay(app, request, 'GET', '/sales/orders', params={'state': 'open', 'limit': 200})
        if status != 200:
            return status, data, {'kind': 'lookup_failed', 'status': status, 'result': data}
        active = [row for row in data.get('orders', [])
                  if float(row.get('remaining_effective_lb', row.get('remaining_lb', 0)) or 0) > api.BALANCE_EPSILON]
        matches = [row for row in active if identifier and row['order_number'].upper() == identifier.upper()]
        if identifier and not matches:
            exact_status, exact = await relay(app, request, 'GET', '/sales/orders/' + quote(identifier, safe=''))
            if exact_status == 200 and isinstance(exact, dict):
                remaining = float(exact.get('remaining_effective_lb', exact.get('remaining_lb', 0)) or 0)
                if remaining <= api.BALANCE_EPSILON and exact.get('status') != 'cancelled':
                    return 409, exact, {'kind': 'refusal', 'status': 409, 'result': {'detail': {
                        'error_code': 'ORDER_ALREADY_FULFILLED', 'order_number': exact.get('order_number', identifier)}}}
        customer = None
        if not matches:
            context = args.get('context') or {}
            customer_ids = [context['customer_id']] if context.get('customer_id') else state.get('resolved', {}).get('customer', [])[-1:]
            party_query = query if not identifier else re.sub(r'.*\b(?:for|para)\s+', '', query, flags=re.I) if re.search(r'\b(?:for|para)\s+', query, re.I) else ''
            if identifier and not party_query:
                # The model may supply only the order code while the worker's
                # original transcript also names its customer.
                party_query = re.sub(re.escape(identifier), '', normalize_order_code(body.text), flags=re.I)
            party_query = re.sub(r'^(?:(?:please|por favor)\s+)?(?:ship|send|env[ií]a|enviar|enviamos)\s+', '', party_query, flags=re.I)
            party_query = re.sub(r'^(?:the|el|la)\s+|\b(?:open|abiertos?|orders?|pedidos?|all|remaining|todo|restante)\b|\b(?:for|para|de|of)\b', ' ', party_query, flags=re.I).strip(' .?!¿¡')
            if party_query:
                party_status, party = await relay(app, request, 'POST', '/resolve', body={'kind': 'customer', 'query': party_query})
                if party_status == 200 and party.get('outcome') == 'match':
                    customer = party['match']['name']
                    customer_ids = [party['match']['id']]
                elif not identifier:
                    return await execute_tool(request, body, actor, state, 'resolve', {'kind': 'customer', 'query': party_query, 'context': {'action': 'ship'}})
            if not customer and customer_ids:
                with api.get_transaction() as cur:
                    cur.execute('SELECT name FROM customers WHERE id=%s AND active IS DISTINCT FROM false', (customer_ids[-1],))
                    person = cur.fetchone()
                    customer = person['name'] if person else None
            if not customer and identifier:
                previous = next((row for row in data.get('orders', []) if row['order_number'].upper() == identifier.upper()), None)
                customer = previous['customer'] if previous else None
            choices = [row for row in active if customer and resolution.normalize(row['customer']) == resolution.normalize(customer)]
            if not identifier:
                matches = choices
        else:
            choices = matches
        if len(matches) == 1:
            candidate = {'id': matches[0]['order_id'], 'name': matches[0]['order_number'], 'order_number': matches[0]['order_number']}
            selected_status, detail = await approve_candidate(request, state, 'order', candidate)
            result = {'outcome': 'match', 'match': candidate, 'order_detail': detail}
            return selected_status, result, None if selected_status == 200 else {'kind': 'lookup_failed'}
        candidates = [{'id': row['order_id'], 'name': row['order_number'], 'order_number': row['order_number'],
                       'label': row['order_number'] + ' · ' + row['customer']} for row in (matches or choices)]
        if not candidates:
            return 200, {}, {'kind': 'clarify', 'resolution_kind': 'order', 'query': identifier or query}
        choice_id = str(uuid4())
        result = {'outcome': 'ambiguous' if len(matches) > 1 else 'none', 'candidates': candidates, 'has_more': False}
        state['pending'] = {'id': choice_id, 'args': {'kind': 'order', 'query': query, 'context': {'action': 'ship'}}, 'result': result}
        return 200, result, {'kind': 'choices', 'id': choice_id, 'resolution_kind': 'order',
            'query': identifier or query, 'order_not_found': bool(identifier), 'result': result}

    async def execute_tool(request, body, actor, state, name, args):
        args = dict(args)
        if name == 'prepare_receive':
            args = normalize_receive_arguments(args, state.get('receive_facts', {}))
        if name in catalog.PREPARES or name == 'packing_context':
            runtime.progress(request, 'lots', language=state.get('language', 'en'))
        elif name == 'today_entries':
            runtime.progress(request, 'today', language=state.get('language', 'en'))
        elif name in ('resolve', 'inventory_lookup', 'lot_lookup', 'open_orders', 'receipt_lookup'):
            runtime.progress(request, 'lookup', args.get('q') or args.get('query') or args.get('lot_code') or args.get('customer') or '', state.get('language', 'en'))
        if name == 'resolve' and args.get('kind') == 'product' and isinstance(args.get('query'), str):
            args['query'] = normalize_voice_text(args.get('query', ''))
        elif name == 'resolve' and args.get('kind') == 'lot' and isinstance(args.get('query'), str):
            args['query'] = catalog_lot_code(args['query'])
        elif name == 'inventory_lookup' and isinstance(args.get('q'), str):
            args['q'] = normalize_voice_text(args['q'])
        elif name in ('lot_lookup', 'packing_context') and isinstance(args.get('lot_code'), str):
            args['lot_code'] = catalog_lot_code(args['lot_code'])
        context = args.get('context')
        action = context.get('action') if isinstance(context, dict) else None
        if name not in ('resolve', 'prepare_receive') or (name == 'resolve' and (action not in (None, 'receive') or args.get('kind') in ('order', 'lot', 'customer'))):
            state.pop('receive_context', None)
            state.pop('receive_facts', None)
        if name == 'unsupported_request':
            return 200, {'outcome': 'unsupported_request'}, {'kind': 'unsupported_request'}
        if name == 'resolve':
            explicit_order = (re.search(r'\b(?:order|pedido|orden)\b', body.text, re.I)
                              and not re.search(r'\boff[ -]order\b|\bwithout\b.*\border\b|\bsin\b.*\b(?:pedido|orden)\b', body.text, re.I))
            if (action == 'ship' or re.match(r'\s*(?:(?:please|por favor)\s+)?(?:ship|send|env[ií]a|enviar|enviamos)\b', body.text, re.I)) and (args.get('kind') == 'order' or
                    (args.get('kind') == 'product' and (explicit_order or re.search(r'\b(?:s\s*o|ese\s+o)[ -]*\d', normalize_order_code(args.get('query', '')), re.I)))):
                order_args = {**args, 'query': body.text} if args.get('kind') == 'product' and explicit_order else args
                return await shipping_order(request, body, actor, state, order_args)
            if args.get('kind') == 'supplier':
                args = {**args, 'limit': 6}
            status, result = await relay(app, request, 'POST', '/resolve', body=args)
            # A similar customer can be a different business. Let office staff
            # choose the suggestion or add their exact wording before approving
            # that customer ID for an order. Exact names and aliases still resolve.
            similar_name = (status == 200 and args.get('kind') in ('product', 'supplier', 'customer')
                            and (result.get('match') or {}).get('tier') == 'trigram')
            similar_customer = (status == 200 and args.get('kind') == 'customer' and actor['role'] in ('office', 'owner')
                                and (result.get('match') or {}).get('tier') not in ('exact', 'alias'))
            # Creating an already-known customer is a complete FL answer. Sending
            # that exact/alias match back to the model leaves it trying to offer
            # an existing customer with no terminal tool, exhausting its budget.
            creating_customer = args.get('kind') == 'customer' and customer_creation(body.text)
            if (status == 200 and result.get('outcome') == 'match'
                    and not result.get('needs_clarification') and not similar_customer and not similar_name
                    and not creating_customer):
                selected_status, detail = await approve_candidate(request, state, args['kind'], result['match'])
                if selected_status != 200:
                    return selected_status, detail, {'kind': 'lookup_failed'}
                if detail is not None:
                    result = {**result, 'order_detail': detail}
                if args['kind'] == 'product' and (args.get('context') or {}).get('action') == 'make':
                    state.setdefault('make_products', []).append(result['match']['id'])
                if args['kind'] == 'product' and pack_without_batch(body.text):
                    # The model may resolve the raw product name or omit the
                    # source/target fields entirely. FL owns parent lookup and
                    # asks for the actual batch before another model decision.
                    choice = resolved_packing_lots(state, result['match']['id'])
                    if choice:
                        return 200, choice['result'], choice
                return status, result, None
            if status == 200:
                if args.get('kind') in ('product', 'supplier', 'customer'):
                    candidates = result.get('candidates') or result.get('near_misses', [])
                    if candidates:
                        result = {**result, 'candidates': candidates}
                if not result.get('candidates') and not similar_customer:
                    state.pop('pending', None)
                    with api.get_transaction() as cur:
                        cur.execute('INSERT INTO assistant_unmatched_words(session_id,actor_id,kind,query,context) VALUES (%s,%s,%s,%s,%s)',
                                    (str(body.session_id), actor['id'], args['kind'], args.get('query', ''), Json(args.get('context', {}))))
                    return status, result, {'kind': 'clarify', 'query': args.get('query', ''),
                        'resolution_kind': args['kind'], 'candidates': result.get('near_misses', [])}
                choice_id = str(uuid4())
                card = {'kind': 'choices', 'id': choice_id, 'resolution_kind': args['kind'], 'query': args.get('query', ''), 'result': result,
                        'supplier_search': args['kind'] == 'supplier'}
                known_customer = args['kind'] == 'customer' and any(row.get('tier') in ('exact', 'alias', 'keyword') for row in result.get('candidates', []))
                if known_customer:
                    card['existing_customer_only'] = True
                state['pending'] = {'id': choice_id, 'args': args, 'result': result, 'supplier_search': args['kind'] == 'supplier',
                                    'packing_batch_prompt': args['kind'] == 'product' and pack_without_batch(body.text)}
                if creating_customer and customer_only(body.text):
                    state['pending']['existing_customer_only'] = True
                if result.get('outcome') == 'none':
                    with api.get_transaction() as cur:
                        cur.execute('INSERT INTO assistant_unmatched_words(session_id,actor_id,kind,query,context) VALUES (%s,%s,%s,%s,%s)',
                                    (str(body.session_id), actor['id'], args['kind'], args.get('query', ''), Json(args.get('context', {}))))
                return status, result, card
            return status, result, {'kind': 'lookup_failed'}
        if name in catalog.PREPARES:
            if catalog.has_ui_fields(args):
                fail(502, 'MODEL_EVIDENCE_FORBIDDEN', 'Confirmation must come from the page / Confirma en la página.')
            if name == 'prepare_create_customer' and args.get('name'):
                status, existing = await relay(app, request, 'POST', '/resolve', body={'kind': 'customer', 'query': args['name']})
                candidates = existing.get('candidates', []) if status == 200 else []
                # A clear catalog name/alias/prefix must be offered, never silently
                # used or duplicated by the model's direct create-customer tool.
                known = [row for row in candidates if row.get('tier') in ('exact', 'alias', 'keyword')]
                if known:
                    choice_id = str(uuid4())
                    result = {**existing, 'candidates': known, 'has_more': False}
                    state['pending'] = {'id': choice_id, 'args': {'kind': 'customer', 'query': args['name']},
                        'result': result, 'existing_customer_only': customer_only(body.text)}
                    return 200, result, {'kind': 'choices', 'id': choice_id, 'resolution_kind': 'customer',
                        'query': args['name'], 'result': result, 'existing_customer_only': True}
            unknown = untrusted_ids(args, state)
            if unknown:
                return 422, {'detail': {'error_code': 'RESOLVE_REQUIRED', 'fields': unknown}}, None
            if name == 'prepare_pack' and args.get('target_product_id'):
                context = state.get('packing_context') or {}
                bound = context.get('target_product_id') == args['target_product_id'] and context.get('lot_id')
                if not bound:
                    # A model can skip packing_context. Its chosen IDs are not
                    # evidence of which physical batch the worker actually used.
                    allocations = args.get('lot_allocations') or []
                    selected_ids = [row['lot_id'] for row in allocations] if allocations else state.get('resolved', {}).get('lot', [])
                    human_text = catalog_voice_text(body.text) if body.text.strip() else next(
                        (row.get('content', '') for row in reversed(state.get('history', [])) if row.get('role') == 'user'), '')
                    explicit = []
                    if selected_ids:
                        with api.get_transaction() as cur:
                            cur.execute("SELECT l.id,l.lot_code,l.product_id FROM lots l JOIN products p ON p.id=l.product_id WHERE l.id=ANY(%s) AND p.type='batch'", (selected_ids,))
                            explicit = [dict(row) for row in cur.fetchall() if re.search(r'(?<!\w)' + re.escape(row['lot_code']) + r'(?!\w)', human_text, re.I)]
                    if len(explicit) == 1 and (not allocations or len(allocations) == 1):
                        status, result, card = await execute_tool(request, body, actor, state, 'packing_context',
                            {'target_product_id': args['target_product_id'], 'lot_code': explicit[0]['lot_code']})
                        if card or status != 200:
                            return status, result, card
                    elif (not allocations or len(explicit) != len(allocations)
                          or any(row['product_id'] != args.get('source_product_id') for row in explicit)):
                        return await execute_tool(request, body, actor, state, 'packing_context',
                            {'target_product_id': args['target_product_id']})
            path, _ = catalog.PREPARES[name]
            prepared_args = {**args, 'client_source': 'fl_assistant'}
            if name == 'prepare_make' and args.get('product_id') in state.get('make_products', []):
                # Only a human's product wording resolved with make context, or
                # an actual selected FL candidate, establishes output identity.
                prepared_args['confirmed_sku'] = True
            if name == 'prepare_pack' and state.get('packing_context'):
                context = state['packing_context']
                if args.get('target_product_id') == context['target_product_id']:
                    prepared_args['source_product_id'] = context['source_product_id']
                    # Catalog case weight is authoritative unless the worker supplied an override.
                    prepared_args['case_weight_lb'] = state.get('packing_case_weight_lb') or context.get('case_size_lb')
                    if context.get('lot_id') and args.get('cases'):
                        weight = state.get('packing_case_weight_lb') or context.get('case_size_lb')
                        if weight:
                            prepared_args['lot_allocations'] = [{'lot_id': context['lot_id'],
                                'quantity_lb': float(weight) * args['cases']}]
            if name == 'prepare_receive' and state.get('receive_facts', {}).get('quantity_lb') is not None:
                given = state['receive_facts']['quantity_lb']
                cases, size = args.get('cases'), args.get('case_size_lb')
                if isinstance(cases, int) and isinstance(size, (int, float)) and abs(cases * size - given) > api.BALANCE_EPSILON:
                    return 422, {'missing_fields': ['cases', 'case_size_lb']}, {
                        'kind': 'missing_facts', 'fields': ['cases', 'case_size_lb'],
                        'known_facts': {**args, 'quantity_lb': given}, 'quantity_mismatch_lb': cases * size}
            if name == 'prepare_ship_order' and 'order_id' in prepared_args:
                order = state.get('order_summaries', {}).get(str(prepared_args['order_id']))
                if (order and order['remaining_lb'] > api.BALANCE_EPSILON
                        and not prepared_args.get('ship_all') and not prepared_args.get('lines')):
                    return 200, {'missing_fields': ['lines'], 'order': order}, {
                        'kind': 'missing_facts', 'fields': ['lines'], 'shipping_quantity_needed': True, 'order': order}
                path = path.format(order_id=prepared_args.pop('order_id'))
            elif name == 'prepare_ship_order':
                return 422, {'detail': {'error_code': 'ORDER_REQUIRED', 'message': 'Resolve the order first.'}}, None
            prepare_attachments = list(body.attachment_ids)
            if name == 'prepare_receive':
                prepare_attachments = list(dict.fromkeys(state.get('receive_context', {}).get('attachment_ids', []) + prepare_attachments))
            if len(prepare_attachments) > 5:
                fail(422, 'ATTACHMENT_LIMIT', 'Use at most 5 attachments / Use como máximo 5 adjuntos.')
            action = name.removeprefix('prepare_')
            try:
                prepared_args.update(evidence_fields(action, linked_attachments(
                    api, body.session_id, actor['id'], prepare_attachments)))
            except HTTPException as exc:
                if not isinstance(exc.detail, dict) or exc.detail.get('error_code') != 'ATTACHMENT_KIND_UNSUPPORTED':
                    raise
                result = {'detail': exc.detail}
                return exc.status_code, result, {'kind': 'refusal', 'status': exc.status_code, 'result': result}
            status, result = await relay(app, request, 'POST', path, body=prepared_args)
            if status != 200 or not result.get('ticket') or not result.get('payload_hash') or not isinstance(result.get('draft'), dict):
                return status, result, {'kind': 'refusal', 'status': status, 'result': catalog.safe_result(result)}
            state.pop('packing_context', None)
            state.pop('packing_case_weight_lb', None)
            runtime.progress(request, 'draft', language=state.get('language', 'en'))
            card = store_draft(result, args, body.session_id, actor['id'], prepare_attachments, state.get('language', 'en'))
            if name == 'prepare_receive':
                state['receive_context'] = {'args': args, 'draft_id': card['id'], 'attachment_ids': prepare_attachments}
                if state.get('receive_queue_current'):
                    state['receive_queue_draft_id'] = card['id']
            return status, result, card
        if name == 'packing_context':
            target_id = args.get('target_product_id')
            if target_id not in state.get('resolved', {}).get('product', []):
                return 422, {'detail': {'error_code': 'RESOLVE_REQUIRED', 'fields': ['target_product_id']}}, None
            state.pop('packing_context', None)
            with api.get_transaction() as cur:
                cur.execute('SELECT id,name,type,case_size_lb,parent_batch_product_id FROM products WHERE id=%s AND active IS DISTINCT FROM false', (target_id,))
                target = cur.fetchone()
                if not target:
                    return 404, {}, {'kind': 'lookup_failed'}
                source_id = target['parent_batch_product_id']
            lot_code = args.get('lot_code')
            lot_data = None
            if lot_code:
                params = {'product_id': source_id} if source_id else {}
                status, lot_data = await relay(app, request, 'GET', '/lots/by-code/' + quote(lot_code, safe=''), params=params)
                if status != 200:
                    if status == 404:
                        choice = packing_lot_choices(state, target, lot_code)
                        if choice:
                            return 200, choice['result'], choice
                    return status, lot_data, {'kind': 'refusal', 'query': lot_code, 'status': status,
                        'result': {'detail': {'error_code': 'LOT_NOT_FOUND'}} if status == 404 else lot_data}
                if lot_data.get('product_type') != 'batch':
                    return 422, {}, {'kind': 'clarify', 'resolution_kind': 'lot', 'query': lot_code}
                source_id = lot_data['product_id']
            if not source_id:
                if target['type'] != 'finished':
                    def core(name):
                        return sorted(word for word in re.findall(r'[a-z0-9]+', resolution.catalog_query(name).lower())
                                      if word not in {'batch', 'granola', 'and'})
                    with api.get_transaction() as cur:
                        cur.execute("SELECT id,name FROM products WHERE type='batch' AND active IS DISTINCT FROM false ORDER BY name,id")
                        roots = [dict(row) for row in cur.fetchall() if core(row['name']) == core(target['name'])]
                    if len(roots) == 1:
                        choice = packing_output_choices(state, {'source_product_id': roots[0]['id'],
                            'source_product_name': roots[0]['name'], 'target_product_id': target_id, 'case_size_lb': None})
                        return 200, choice.get('result', {}), choice
                choice = packing_lot_choices(state, target)
                if choice:
                    return 200, choice['result'], choice
                return 200, {}, {'kind': 'missing_facts', 'fields': ['source_product_id'], 'packing_batch_needed': True}
            if not lot_data:
                choice = packing_lot_choices(state, target)
                if choice:
                    return 200, choice['result'], choice
                return 200, {}, {'kind': 'missing_facts', 'fields': ['source_product_id'], 'packing_batch_needed': True}
            status, source = await relay(app, request, 'POST', '/resolve', body={
                'kind': 'product', 'query': str(source_id), 'context': {'action': 'pack'}})
            if status != 200 or source.get('outcome') != 'match':
                return status, source, {'kind': 'lookup_failed'}
            approve_id(state, 'product', source['match'])
            result = {'target_product_id': target_id, 'source_product_id': source_id,
                      'source_product_name': source['match']['name'],
                      'case_size_lb': float(target['case_size_lb']) if target['case_size_lb'] is not None else None}
            if lot_data:
                approve_id(state, 'lot', {'id': lot_data['id']})
                result.update(lot_id=lot_data['id'], lot_code=lot_data['lot_code'])
            if target['type'] != 'finished':
                choice = packing_output_choices(state, result)
                return 200, choice.get('result', {}), choice
            state['packing_context'] = result
            return 200, result, None
        if name == 'open_orders':
            params = {'state': 'open', 'limit': 200}
            if args.get('customer'):
                # Party names use their literal wording, without product aliases.
                resolved_status, party = await relay(app, request, 'POST', '/resolve', body={'kind': 'customer', 'query': args['customer']})
                if resolved_status != 200 or party.get('outcome') != 'match':
                    return await execute_tool(request, body, actor, state, 'resolve', {'kind': 'customer', 'query': args['customer']})
                params['customer'] = party['match']['name']
            status, result = await relay(app, request, 'GET', '/sales/orders', params=params)
            if status != 200:
                return status, result, {'kind': 'refusal', 'status': status, 'result': result}
            all_orders = result.get('orders', [])
            exhausted = [row for row in all_orders if float(row.get('remaining_effective_lb', row.get('remaining_lb', 0)) or 0) <= api.BALANCE_EPSILON]
            orders = [row for row in all_orders if row not in exhausted]
            if args.get('customer'):
                canonical = resolution.normalize(params['customer'])
                orders = [r for r in orders if resolution.normalize(r['customer']) == canonical]
            if args.get('product'):
                query = resolution.catalog_query(args['product'])
                product_status, products = await relay(app, request, 'POST', '/resolve', body={
                    'kind': 'product', 'query': query, 'context': {'action': 'order'}, 'limit': 25})
                if product_status != 200:
                    return product_status, products, {'kind': 'lookup_failed', 'status': product_status, 'result': products}
                matches = [products['match']] if products.get('match') else products.get('candidates', [])
                if not matches or (not products.get('match') and any(r.get('tier') == 'trigram' for r in matches)):
                    return await execute_tool(request, body, actor, state, 'resolve', {'kind': 'product', 'query': args['product']})
                product_ids = [r['id'] for r in matches]
                with api.get_transaction() as cur:
                    cur.execute('SELECT name,odoo_code FROM products WHERE active IS DISTINCT FROM false AND (id=ANY(%s) OR parent_batch_product_id=ANY(%s))', (product_ids, product_ids))
                    identities = [dict(r) for r in cur.fetchall()]
                names = {resolution.normalize(r['name']) for r in identities}
                skus = {str(r['odoo_code']) for r in identities if r.get('odoo_code')}
                orders = [r for r in orders if any(resolution.normalize(i.get('product')) in names or str(i.get('sku')) in skus
                                                 for i in r.get('pallet_lines', []))]
            rows = [dict(order_id=r['order_id'], order_number=r['order_number'], customer=r['customer'],
                         items=r.get('pallet_lines', []), ship_date=r.get('requested_ship_date'),
                         status=r.get('fulfillment', r.get('status')), state=r.get('state'),
                         remaining_lb=r.get('remaining_effective_lb', r.get('remaining_lb'))) for r in orders]
            result = {'orders': rows, 'count': len(rows), 'customer': args.get('customer'),
                      'product': args.get('product'), 'truncated': len(result.get('orders', [])) == 200,
                      'nothing_left_to_ship': [{'order_number': row['order_number'], 'customer': row['customer']} for row in exhausted]}
            return 200, result, replies.read_card(name, result, state.get('language', 'en'))
        if name == 'lot_lookup':
            path, params = '/lots/by-code/' + quote(args.get('lot_code', ''), safe=''), {}
        elif name == 'today_entries':
            day = api.get_plant_now().date().isoformat()
            rows, offset = [], 0
            while True:
                status, page = await relay(app, request, 'GET', '/receipts',
                    params={'date': day, 'summary': 'true', 'limit': 500, 'offset': offset})
                if status != 200:
                    return status, page, {'kind': 'lookup_failed', 'status': status, 'result': page}
                rows.extend(page.get('receipts', []))
                if not page.get('has_more'):
                    break
                next_offset = page.get('next_offset')
                if not isinstance(next_offset, int) or next_offset <= offset:
                    fail(502, 'FL_RESPONSE_INVALID', 'Factory Ledger is temporarily unavailable — try again in a minute.')
                offset = next_offset
            own = sum(1 for row in rows if str((row.get('actor') or {}).get('id')) == str(actor['id']))
            result = {'date': day, 'receipts': runtime.day_preview(rows), 'own_count': own, 'team_count': len(rows), 'view_all': True, 'preview_limit': 20}
            return 200, result, replies.read_card(name, result, state.get('language', 'en'))
        elif name == 'inventory_lookup':
            if not isinstance(args.get('q'), str) or not args['q'].strip():
                return 422, {'detail': [{'loc': ['query', 'q'], 'type': 'missing', 'msg': 'Supply product wording.'}]}, None
            query = resolution.catalog_query(args.get('q', ''))
            resolved_status, products = await relay(app, request, 'POST', '/resolve', body={'kind': 'product', 'query': query})
            if resolved_status != 200:
                return resolved_status, products, {'kind': 'lookup_failed', 'status': resolved_status, 'result': products}
            with api.get_transaction() as cur:
                matches = resolution.inventory_products(cur, query, products)
            if not matches:
                # Several keyword matches are product choices, not one combined
                # stock answer. Keep the first choices short on a phone.
                candidates = (products.get('candidates') or products.get('near_misses', []))[:4]
                if not candidates:
                    return await execute_tool(request, body, actor, state, 'resolve',
                        {'kind': 'product', 'query': args['q']})
                result = {**products, 'candidates': candidates, 'has_more': False}
                choice_id = str(uuid4())
                state['pending'] = {'id': choice_id, 'args': {'kind': 'product', 'query': args['q']}, 'result': result}
                return 200, result, {'kind': 'choices', 'id': choice_id, 'resolution_kind': 'product',
                    'query': args['q'], 'result': result}
            rows = []
            for product in matches:
                lookup_status, inventory = await relay(app, request, 'GET', '/inventory/lookup', params={'q': product.get('odoo_code') or product['name'], 'limit': 50})
                if lookup_status != 200:
                    return lookup_status, inventory, {'kind': 'lookup_failed', 'status': lookup_status, 'result': inventory}
                rows.extend(r for r in inventory.get('results', []) if r['product'] == product['name'])
            # Different-lot trace debits are unresolved evidence, not negative physical stock.
            for row in rows:
                lots = row.get('lots', [])
                unresolved = [lot for lot in lots if str(lot.get('lot', '')).upper().startswith('UNVERIFIED-')]
                if unresolved:
                    row['lots'] = [lot for lot in lots if lot not in unresolved]
                    row['total_on_hand'] = round(sum(float(lot.get('qty_on_hand', 0)) for lot in row['lots']), 2)
                    row['unverified_lots'] = [dict(lot=lot['lot'], unverified=True, status='unverified, awaiting follow-up') for lot in unresolved]
                    # Counts supplied by the raw inventory endpoint include unresolved debits.
                    row.pop('unit_count', None)
                    row.pop('batch_count', None)
            # Hiding an empty physical-stock row must not hide its unresolved
            # tag evidence or turn that debit into available inventory.
            unverified_evidence = [dict(product=row['product'], **lot)
                for row in rows for lot in row.get('unverified_lots', [])]
            include_zero = replies.include_zero_stock(body.text + ' ' + args.get('q', ''))
            if not include_zero:
                rows = [row for row in rows if abs(float(row.get('total_on_hand') or 0)) > 0.0001]
            # A stock question gets one plain answer; lot evidence stays in
            # the result and is shown in full only when explicitly requested.
            show_lots = bool(re.search(r'\b(?:lots?|lotes?)\b', body.text, re.I))
            result = {'query': args.get('q', ''), 'results': rows, 'include_zero': include_zero,
                      'stock_summary_only': not show_lots}
            if unverified_evidence:
                result['unverified_evidence'] = unverified_evidence
            if len(matches) > 1 and matches[0].get('type') == 'batch':
                result['stock_family'] = {'bulk_product': matches[0]['name']}
            if not rows:
                result['product_names'] = [product['name'] for product in matches]
            result['summary_en'] = replies.read_card(name, result, 'en')['summary']
            result['summary_es'] = replies.read_card(name, result, 'es')['summary']
            return 200, result, replies.read_card(name, result, state.get('language', 'en'))
        elif name == 'receipt_lookup':
            number = args.get('receipt_number', '')
            if not isinstance(number, str) or not re.fullmatch(r'[A-Z]{2,3}-\d{6}-\d{3,}', number):
                return 422, {'detail': {'error_code': 'INVALID_RECEIPT', 'message': 'Supply an exact FL receipt number.'}}, None
            path, params = '/receipts/' + number, {}
        elif name == 'shift_summary' and any(getattr(r, 'path', '') == '/reports/shift-summary' for r in app.routes):
            path, params = '/reports/shift-summary', {'date': api.get_plant_now().date().isoformat(), 'actor': 'me'}
        else:
            fail(502, 'TOOL_NOT_ALLOWED', 'No permitted FL tool completed / Ninguna herramienta FL completó.')
        status, result = await relay(app, request, 'GET', path, params=params)
        if name == 'lot_lookup':
            if status == 404:
                return status, result, {'kind': 'lookup_failed', 'query': args.get('lot_code', ''),
                    'status': status, 'result': {'detail': {'error_code': 'LOT_NOT_FOUND'}}}
            if status == 200 and str(result.get('lot_code', '')).upper().startswith('UNVERIFIED-'):
                result = dict(result, quantity_on_hand=None, unverified=True)
                result.pop('unit_count', None)
                result.pop('batch_count', None)
        card = replies.read_card(name, result, state.get('language', 'en')) if status == 200 else {'kind': 'lookup_failed', 'status': status, 'result': result}
        return status, result, card

    def store_draft(result, args, sid, actor_id, attachments, language='en'):
        draft_id = str(uuid4())
        card = {'kind': 'draft', 'id': draft_id, 'prepared': catalog.safe_result(result),
                'attachment_ids': list(attachments), 'language': language}
        card = replies.decorate(card, language)
        original = {**args, 'occurred_at': result['draft']['happened_at']}
        with api.get_transaction() as cur:
            cur.execute('INSERT INTO assistant_drafts(id,session_id,actor_id,ticket,payload_hash,card,prepare_body) VALUES (%s,%s,%s,%s,%s,%s,%s)',
                        (draft_id, str(sid), actor_id, result['ticket'], result['payload_hash'], Json(card), Json(original)))
        return card

    @app.post('/assistant/confirm-sku')
    async def confirm_sku(body: DraftBody, request: Request, actor=Depends(authenticate)):
        row = draft_row(api, body.draft_id, actor['id'])
        prepared = row['card']['prepared']
        if (row['status'] != 'pending' or row['record_started_at'] or prepared.get('action') != 'make'
                or not any(b.get('code') == 'SKU_CONFIRMATION_REQUIRED' for b in prepared.get('blockers', []))):
            fail(409, 'SKU_CONFIRMATION_UNAVAILABLE', 'Use the current FL draft / Usa el borrador actual de FL.')
        args = {**row['prepare_body'], 'confirmed_sku': True, 'client_source': 'fl_assistant'}
        status, result = await relay(app, request, 'POST', '/make/prepare', body=args)
        if status != 200 or not result.get('ticket'):
            return JSONResponse(status_code=status, content=catalog.safe_result(result))
        card = store_draft(result, args, row['session_id'], actor['id'], row['card'].get('attachment_ids', []), row['card'].get('language', 'en'))
        with api.get_transaction() as cur:
            cur.execute("UPDATE assistant_drafts SET status='cancelled' WHERE id=%s AND status='pending' AND record_started_at IS NULL", (str(body.draft_id),))
        return {'card': card}

    def renew_lease(sid, lease):
        # A crashed worker holds the chat for at most two minutes. Active turns
        # renew between bounded OpenAI calls; a replaced worker cannot save.
        with api.get_transaction() as cur:
            cur.execute("""UPDATE assistant_sessions SET lease_until=clock_timestamp()+interval '2 minutes'
                WHERE id=%s AND lease_id=%s""", (sid, lease))
            if cur.rowcount != 1:
                fail(409, 'CHAT_BUSY', 'Chat changed; reload / El chat cambió; recarga.')

    @app.post('/assistant/turn')
    async def turn(body: TurnBody, request: Request, actor=Depends(authenticate)):
        return await runtime.run_turn(lambda: turn_impl(body, request, actor), request)

    async def turn_impl(body, request, actor):
        runtime.progress(request, 'read', language=replies.language(body.text, body.language or 'en'))
        sid, tid, lease = str(body.session_id), str(body.turn_id), str(uuid4())
        digest = write_tickets.canonical_hash(jsonable_encoder(body))
        session_row(api, sid, actor['id'])
        with api.get_transaction() as cur:
            cur.execute('SELECT request_hash,response FROM assistant_turns WHERE session_id=%s AND id=%s', (sid, tid))
            previous = cur.fetchone()
            if previous:
                if previous['request_hash'] != digest:
                    fail(409, 'TURN_CHANGED', 'Retry the original message / Reintenta el mensaje original.')
                return previous['response']
            cur.execute("""UPDATE assistant_sessions SET lease_id=%s,lease_until=clock_timestamp()+interval '2 minutes'
                WHERE id=%s AND actor_id=%s AND (lease_until IS NULL OR lease_until < clock_timestamp()) RETURNING state""",
                        (lease, sid, actor['id']))
            claimed = cur.fetchone()
            if not claimed:
                fail(409, 'CHAT_BUSY', 'Wait for the current message / Espera el mensaje actual.')
        try:
            state = claimed['state']
            state['_submitted_text'] = body.text
            state['language'] = replies.language(body.text, body.language or state.get('language', 'en'))
            handoff = office_handoff(body.text, state['language']) if not body.choice_id else None
            if handoff:
                return save_turn(state, [handoff], body, actor, digest, lease)
            if not body.choice_id and is_receive_continuation(body.text):
                queued = state.get('receive_queue', [])
                current_id = state.get('receive_queue_draft_id') or state.get('receive_context', {}).get('draft_id')
                if current_id and draft_row(api, current_id, actor['id'])['status'] == 'pending':
                    return save_turn(state, [{'kind': 'receive_queue', 'finish_current': True}], body, actor, digest, lease)
                if not current_id and state.get('receive_queue_current'):
                    return save_turn(state, [{'kind': 'receive_queue', 'finish_current': True, 'missing_current': True}], body, actor, digest, lease)
                if not queued:
                    return save_turn(state, [{'kind': 'receive_queue', 'queue_empty': True}], body, actor, digest, lease)
                body = body.model_copy(update={'text': queued.pop(0)})
                state['receive_queue_current'] = body.text
                state.pop('receive_queue_draft_id', None)
                state.pop('receive_facts', None)
                state.pop('receive_context', None)
            text = catalog_voice_text(body.text)
            items = split_receive_items(text) if not body.choice_id else []
            if items:
                state.setdefault('receive_original_messages', []).append(body.text)
                state.setdefault('receive_queue', []).extend(items[1:])
                state['receive_queue_current'] = items[0]
                state.pop('receive_queue_draft_id', None)
                body = body.model_copy(update={'text': items[0]})
                text = items[0]
            if re.search(r'\b(?:pack(?:ed)?|empacamos|empacado|empaque)\b', text, re.I):
                state.pop('packing_case_weight_lb', None)
                if not body.choice_id:
                    state.pop('packing_context', None)
                weight = re.search(r'(\d+(?:\.\d+)?)\s*(?:lb|pounds?|libras?)\s*(?:per|por|/)\s*(?:case|caja)', text, re.I)
                if weight:
                    state['packing_case_weight_lb'] = float(weight.group(1))
            # A correction belongs to the current receive, even without repeating "received".
            receive_start = bool(items or re.search(r'\b(?:receive(?:d)?|recibimos|recibi|recibí|recibir)\b', text, re.I))
            receive_correction = not receive_start and bool(state.get('receive_context') or state.get('receive_facts')) and bool(
                re.search(r'\b(?:total|actually|correction|correct|it was|were|era|eran|fue|fueron|corrijo|correccion|corrección)\b', text, re.I))
            if receive_start:
                state.pop('receive_facts', None)
                state.pop('receive_context', None)
            if receive_start or state.get('receive_context') or state.get('receive_facts'):
                state['receive_facts'] = merge_receive_lot_facts(state.get('receive_facts', {}), body.text)
                state['receive_facts'].update(package_facts(body.text))
            new_receive_total = None
            if (receive_start or receive_correction) and not package_facts(body.text):
                for qty in re.finditer(r'(?<![\w.])(\d+(?:\.\d+)?)\s*(?:lb\b|lbs\b|pounds?\b|libras?\b)', text, re.I):
                    before, after = text[:qty.start()], text[qty.end():]
                    if re.match(r'\s*(?:(?:per|por|/)\s*(?:case|caja|bag|bolsa|saco)|each|cada|bags?\b)', after, re.I):
                        continue
                    is_total = re.search(r'\btotal\s*[:=]?\s*$', before, re.I) or re.match(r'\s*(?:in\s+|en\s+)?total\b', after, re.I)
                    if not is_total and (re.search(r'\b\d+\s*(?:cases?|cajas?|bags?|bolsas?|sacos?)\b', before, re.I) or re.search(r'[x×@]\s*$', before)):
                        continue
                    new_receive_total = float(qty.group(1))
                    state.setdefault('receive_facts', {})['quantity_lb'] = new_receive_total
                    break
            # Names/bytes stay outside model context; A6 stores every photo.
            linked_attachments(api, sid, actor['id'], body.attachment_ids)
            history = state.setdefault('history', [])
            runtime.compact_history(history)
            # Only a customer actually recorded in this person's chat grants an
            # ID. Read the saved receipt at turn start so Record never races a
            # running turn's state/lease. Pending/cancelled drafts grant nothing.
            with api.get_transaction() as cur:
                cur.execute("""SELECT id,result FROM assistant_drafts
                    WHERE session_id=%s AND actor_id=%s AND status='committed'
                      AND card->'prepared'->>'action'='create_customer'
                    ORDER BY created_at,id""", (sid, actor['id']))
                customer_receipts = [dict(row) for row in cur.fetchall()]
            known_receipts = state.setdefault('customer_receipts', [])
            for saved in customer_receipts:
                result = saved['result']
                if not receipt_ok(200, result) or not result['receipt_number'].startswith('CUS-'):
                    continue
                customer = result.get('customer')
                if not isinstance(customer, dict):
                    continue
                customer_id = result.get('customer_id')
                if (not isinstance(customer_id, int) or isinstance(customer_id, bool)
                        or customer_id <= 0 or customer.get('id') != customer_id):
                    continue
                approve_id(state, 'customer', customer)
                receipt_number = result['receipt_number']
                if receipt_number not in known_receipts:
                    history.append({'role': 'user', 'content': 'FL confirmed the customer Record: ' +
                        json.dumps({'receipt_number': receipt_number, 'customer': {
                            'id': customer_id, 'name': customer.get('name')}}, ensure_ascii=False)})
                    known_receipts.append(receipt_number)
            pending = state.get('pending')
            search = re.match(r'^\s*(?:supplier|proveedor)\s*:\s*(.*)$', body.text, re.I)
            if search and (pending or {}).get('supplier_search'):
                candidates = await supplier_choices(request, state, search.group(1), (state.get('receive_context', {}).get('args') or {}).get('product_id'))
                card = supplier_choice_card(state, candidates, search.group(1))
                return save_turn(state, [card], body, actor, digest, lease)
            if receive_correction and new_receive_total is not None and state.get('receive_context'):
                context = state['receive_context']
                if context.get('draft_id'):
                    current = draft_row(api, context['draft_id'], actor['id'])
                    if current['status'] != 'pending' or current['record_started_at']:
                        state.pop('receive_context', None)
                        state.pop('receive_facts', None)
                        return save_turn(state, [{'kind': 'refusal', 'result': {'detail': {'error_code': 'RECEIVE_CORRECTION_UNAVAILABLE'}}}], body, actor, digest, lease)
                args = dict(context['args'])
                if args.get('cases'):
                    # Explicit total / previously supplied case count is arithmetic, not a guessed fact.
                    args['case_size_lb'] = state['receive_facts']['quantity_lb'] / args['cases']
                    if 'case_size_lb' in state['receive_facts']:
                        state['receive_facts']['case_size_lb'] = args['case_size_lb']
                    history.append({'role': 'user', 'content': text.strip()})
                    if context.get('draft_id'):
                        # Claim the replacement before preparing. A concurrent Record
                        # holds this row lock and wins by setting record_started_at.
                        with api.get_transaction() as cur:
                            cur.execute("UPDATE assistant_drafts SET status='cancelled' WHERE id=%s AND status='pending' AND record_started_at IS NULL RETURNING id", (context['draft_id'],))
                            replaced = cur.fetchone()
                        if not replaced:
                            state.pop('receive_context', None)
                            state.pop('receive_facts', None)
                            return save_turn(state, [{'kind': 'refusal', 'result': {'detail': {'error_code': 'RECEIVE_CORRECTION_UNAVAILABLE'}}}], body, actor, digest, lease)
                        state['receive_context'] = {'args': args, 'attachment_ids': context.get('attachment_ids', [])}
                    status, result, card = await execute_tool(request, body, actor, state, 'prepare_receive', args)
                    if card and context.get('draft_id'):
                        card['supersedes_draft_id'] = context['draft_id']
                    if card:
                        return save_turn(state, [card], body, actor, digest, lease)
            if body.choice_id:
                if not pending or str(body.choice_id) != pending['id']:
                    fail(409, 'CHOICE_EXPIRED', 'Choose from the current choices / Usa las opciones actuales.')
                if body.next_page:
                    args = {**pending['args'], 'offset': pending['result'].get('next_offset', 0)}
                    if not pending['result'].get('has_more'):
                        fail(422, 'NO_MORE_CHOICES', 'No more choices / No hay más opciones.')
                    status, result, card = await execute_tool(request, body, actor, state, 'resolve', args)
                    cards = [card] if card else [{'kind': 'read', 'tool': 'resolve', 'result': result}]
                    return save_turn(state, cards, body, actor, digest, lease)
                if body.selected_id == 'different_batch' and pending.get('packing_context'):
                    state.pop('pending', None)
                    return save_turn(state, [{'kind': 'missing_facts', 'fields': ['source_product_id'],
                        'packing_batch_needed': True}], body, actor, digest, lease)
                candidates = pending['result'].get('candidates', [])
                selected = next((c for c in candidates if c['id'] == body.selected_id), None)
                if not selected:
                    if body.selected_id is not None:
                        fail(422, 'CHOICE_INVALID', 'Choose a displayed option / Elige una opción mostrada.')
                    state.pop('pending', None)
                    cards = [{'kind': 'clarify', 'query': pending['args'].get('query', ''),
                              'resolution_kind': pending['args']['kind']}]
                    history.append({'role': 'user', 'content': 'None of these candidates; ask me for a fuller name.'})
                    return save_turn(state, cards, body, actor, digest, lease)
                selected_status, detail = await approve_candidate(request, state, pending['args']['kind'], selected)
                if selected_status != 200:
                    return save_turn(state, [{'kind': 'lookup_failed'}],
                                     body, actor, digest, lease)
                if pending.get('existing_customer_only'):
                    state.pop('pending', None)
                    es = state['language'] == 'es'
                    return save_turn(state, [{'kind': 'handoff', 'message':
                        (f"{selected['name']} ya existe. Use este cliente para el pedido." if es else
                         f"{selected['name']} already exists. Use this customer for the order."),
                        'actions': [{'href': '/dashboard/office.html?action=create_order',
                                     'label': 'Crear pedido' if es else 'Create order'}]}], body, actor, digest, lease)
                if pending.get('packing_batch_prompt'):
                    history.append({'role': 'user', 'content': 'I selected this FL candidate: ' + json.dumps(selected, ensure_ascii=False)})
                    card = resolved_packing_lots(state, selected['id'])
                    if card:
                        return save_turn(state, [card], body, actor, digest, lease)
                if pending.get('packing_context'):
                    context = {**pending['packing_context'], 'source_product_id': selected['product_id'],
                        'source_product_name': selected['source_product_name'], 'lot_id': selected['id'], 'lot_code': selected['lot_code']}
                    approve_id(state, 'product', {'id': selected['product_id']})
                    if pending.get('packing_target_invalid'):
                        history.append({'role': 'user', 'content': 'I selected this FL candidate: ' + json.dumps(selected, ensure_ascii=False)})
                        card = packing_output_choices(state, context)
                        return save_turn(state, [card], body, actor, digest, lease)
                    state['packing_context'] = context
                if pending.get('packing_output_context'):
                    state['packing_context'] = {**pending['packing_output_context'],
                        'target_product_id': selected['id'], 'case_size_lb': selected['case_size_lb']}
                    approve_id(state, 'product', {'id': state['packing_context']['source_product_id']})
                    if not state['packing_context'].get('lot_id'):
                        history.append({'role': 'user', 'content': 'I selected this FL candidate: ' + json.dumps(selected, ensure_ascii=False)})
                        status, result, card = await execute_tool(request, body, actor, state, 'packing_context',
                            {'target_product_id': selected['id']})
                        return save_turn(state, [card], body, actor, digest, lease)
                if pending['args']['kind'] == 'product' and (pending['args'].get('context') or {}).get('action') == 'make':
                    state.setdefault('make_products', []).append(selected['id'])
                state.pop('pending', None)
                selection = {'candidate': selected, 'order_detail': detail} if detail is not None else selected
                history.append({'role': 'user', 'content': 'I selected this FL candidate: ' + json.dumps(selection, ensure_ascii=False)})
            else:
                if not body.text.strip():
                    fail(422, 'TEXT_REQUIRED', 'Enter a message / Escribe un mensaje.')
                state.pop('pending', None)
                history.append({'role': 'user', 'content': text.strip()})
            if not body.choice_id and not order_number_in_text(body.text) and replies.is_today_query(body.text):
                _, _, card = await execute_tool(request, body, actor, state, 'today_entries', {})
                return save_turn(state, [card], body, actor, digest, lease)
            explicit_order = None
            if not body.choice_id and order_number_in_text(body.text):
                # This runs before OpenAI can choose a product tool. Preserve
                # the full transcript for customer recovery and supplied facts.
                _, resolved_order, card = await shipping_order(request, body, actor, state,
                    {'kind': 'order', 'query': body.text, 'context': {'action': 'ship'}})
                if card:
                    return save_turn(state, [card], body, actor, digest, lease)
                explicit_order = resolved_order['match']['id']
                history.append({'role': 'user', 'content': 'FL resolved this order number before tool selection. '
                    'Continue its shipping flow using only the supplied shipment facts: ' +
                    json.dumps(resolved_order, ensure_ascii=False)})
            reasons = cached_reasons()
            functions = catalog.tools(reasons['reasons'], shift_summary=any(getattr(r, 'path', '') == '/reports/shift-summary' for r in app.routes))
            if explicit_order:
                functions = [function for function in functions if function['name'] == 'prepare_ship_order']
            allowed = {f['name'] for f in functions}
            cards = []
            validation_retries = 0
            for step in range(MAX_STEPS + 1):
                if step >= MAX_STEPS + validation_retries:
                    break
                renew_lease(sid, lease)
                try:
                    response = await openai('responses', body={
                        'model': os.getenv('ASSISTANT_MODEL', MODEL), 'store': False,
                        'instructions': catalog.INSTRUCTIONS
                            + '\nReply language: ' + state['language'] + '. User Spanish always takes precedence.'
                            + '\nFL reason catalog: ' + json.dumps(reasons['reasons'], ensure_ascii=False)
                            + '\nCurrent FL plant time (America/New_York): ' + api.get_plant_now().isoformat(),
                        'input': history, 'tools': functions, 'tool_choice': 'required',
                        'parallel_tool_calls': False, 'max_output_tokens': 4000,
                        'reasoning': {'effort': 'low'},
                    })
                except HTTPException as exc:
                    if isinstance(exc.detail, dict) and exc.detail.get('error_code') == assistant_service.CODE:
                        notice_id = assistant_service.record_outage(api, actor)
                        offline = assistant_service.offline_card(state['language'])
                        offline['notice_id'] = notice_id
                        return save_turn(state, [offline], body, actor, digest, lease)
                    if not validation_retries:
                        raise
                    cards.append({'kind': 'misunderstood'})
                    break
                output = response.get('output', [])
                calls = [c for c in output if c.get('type') == 'function_call']
                if response.get('status') != 'completed' or len(calls) != 1 or calls[0].get('name') not in allowed:
                    if validation_retries:
                        cards.append({'kind': 'misunderstood'})
                        break
                    fail(502, 'TOOL_REQUIRED', 'No FL tool completed.')
                call = calls[0]
                # Keep the failed call/output pair for the one model repair attempt.
                # Model prose is never rendered or retained.
                history.extend(item for item in output if item.get('type') in ('reasoning', 'function_call'))
                missing_card = None
                try:
                    args = json.loads(call['arguments'])
                    if not isinstance(args, dict):
                        raise ValueError()
                except (KeyError, ValueError, TypeError):
                    status, result, card = 422, {'detail': {'error_code': 'TOOL_ARGUMENTS_INVALID', 'message': 'Arguments must be a JSON object.'}}, None
                else:
                    if call['name'] == 'prepare_receive':
                        args = normalize_receive_arguments(args, state.get('receive_facts', {}))
                    if explicit_order and call['name'] == 'prepare_ship_order':
                        # The person's explicit number outranks any earlier
                        # order ID still present in this chat's model history.
                        args['order_id'] = explicit_order
                    status, result, card = await execute_tool(request, body, actor, state, call['name'], catalog.clean_arguments(args))
                    missing_card = missing_prepare_card(call['name'], status, result, args)
                    if card and card.get('kind') == 'missing_facts':
                        missing_card = card
                history.append(catalog.output_item(call['call_id'], runtime.history_result(call['name'], result)))
                if call['name'] == 'prepare_receive':
                    detail = result.get('detail', {}) if isinstance(result, dict) else {}
                    if isinstance(detail, dict) and detail.get('error_code') == 'SUPPLIER_REQUIRED':
                        missing_card = {'kind': 'missing_facts', 'fields': ['supplier_id']}
                    if missing_card:
                        if args.get('supplier_id') is None and 'supplier_id' not in missing_card['fields']:
                            missing_card['fields'].insert(0, 'supplier_id')
                        missing_card['known_facts'] = {**catalog.clean_arguments(state.get('receive_facts', {})),
                                                       **catalog.clean_arguments(args)}
                        if 'supplier_id' in missing_card['fields']:
                            candidates = await supplier_choices(request, state, product_id=args.get('product_id'))
                            choice = supplier_choice_card(state, candidates)
                            missing_card.update(choice_id=choice['choice_id'], supplier_choices=candidates,
                                                supplier_search=True, resolution_kind='supplier')
                        state['receive_context'] = {'args': catalog.clean_arguments(args),
                            'attachment_ids': list(dict.fromkeys(state.get('receive_context', {}).get('attachment_ids', []) + body.attachment_ids))}
                if missing_card and missing_card.get('reason_required'):
                    missing_card['reasons'] = reasons['reasons']
                    missing_card['known_facts'] = {key: value for key, value in catalog.clean_arguments(args).items()
                                                   if key != 'reason_code'}
                if missing_card:
                    card = missing_card
                elif argument_validation_error(status, result):
                    if validation_retries == 0:
                        validation_retries = 1
                        continue
                    cards.append({'kind': 'misunderstood'})
                    break
                if card:
                    if card['kind'] == 'refusal':
                        card['reasons'] = reasons['reasons']
                    cards.append(card)
                    break
            if not cards and validation_retries:
                cards.append({'kind': 'misunderstood'})
            if not cards:
                fail(502, 'TOOL_LIMIT', 'Please give more precise details.')
            # Bounded transcript, trim only at user-message boundaries to keep call/output pairs.
            while len(json.dumps(history)) > 90000 and sum(x.get('role') == 'user' for x in history) > 1:
                next_user = next(i for i, x in enumerate(history[1:], 1) if x.get('role') == 'user')
                del history[:next_user]
            return save_turn(state, cards, body, actor, digest, lease)
        finally:
            with api.get_transaction() as cur:
                cur.execute('UPDATE assistant_sessions SET lease_id=NULL,lease_until=NULL WHERE id=%s AND lease_id=%s', (sid, lease))

    def save_turn(state, cards, body, actor, digest, lease):
        if state.get('receive_queue') and not any(card['kind'] == 'receive_queue' for card in cards):
            cards = [*cards, {'kind': 'receive_queue'}]
        cards = [dict(card, queued_text=list(state.get('receive_queue', []))) if card['kind'] == 'receive_queue' else card for card in cards]
        cards = [replies.decorate(card, state.get('language', 'en')) for card in cards]
        result = {'turn_id': str(body.turn_id), 'text': state.pop('_submitted_text', body.text), 'language': state.get('language', 'en'), 'cards': cards}
        with api.get_transaction() as cur:
            cur.execute('UPDATE assistant_sessions SET state=%s,updated_at=clock_timestamp() WHERE id=%s AND actor_id=%s AND lease_id=%s',
                        (Json(state), str(body.session_id), actor['id'], lease))
            if cur.rowcount != 1:
                fail(409, 'CHAT_BUSY', 'Chat changed; reload / El chat cambió; recarga.')
            cur.execute('INSERT INTO assistant_turns(session_id,id,request_hash,response) VALUES (%s,%s,%s,%s)',
                        (str(body.session_id), str(body.turn_id), digest, Json(result)))
        return result

    @app.post('/assistant/record')
    async def record(body: RecordBody, request: Request, actor=Depends(authenticate)):
        # Durably mark the attempt BEFORE FL commit. A lost response must never
        # permit Cancel to claim NOT recorded for a ticket that actually posted.
        with api.get_transaction() as cur:
            row = draft_row(api, body.draft_id, actor['id'], lock=True, cur=cur)
            if row['status'] == 'cancelled':
                fail(409, 'DRAFT_CANCELLED', 'Draft cancelled. NOT recorded / Borrador cancelado. NO registrado.')
            # A lost response can leave the chat pending after FL committed.
            # Historical cards may name photos the old code never bound. They
            # cannot change an existing receipt or prevent its authenticated
            # recovery; the commit route still checks the actor and payload hash.
            cur.execute("SELECT status='committed' AS recorded FROM write_tickets WHERE ticket_hash=%s",
                        (write_tickets.token_hash(row['ticket']),))
            saved = cur.fetchone()
            evidence = {} if saved and saved['recorded'] else evidence_fields(
                row['card'].get('prepared', {}).get('action'),
                # Same transaction: Record must not hold one pooled connection while waiting for another.
                linked_attachments(api, row['session_id'], actor['id'], record_attachment_ids(row['card']), cur=cur))
            prior_attempt = row['record_started_at']
            cur.execute('UPDATE assistant_drafts SET record_started_at=clock_timestamp() WHERE id=%s RETURNING record_started_at', (str(body.draft_id),))
            attempt = cur.fetchone()['record_started_at']
        payload = {'payload_hash': row['payload_hash'], 'acknowledged_warnings': body.acknowledged_warnings}
        if body.lot_confirmations:
            payload['lot_confirmations'] = body.lot_confirmations
        payload.update(evidence)
        # Always relay even after a saved result: FL re-authenticates and replays.
        status, result = await relay(app, request, 'POST', '/tickets/' + row['ticket'] + '/commit', body=payload)
        if status == 202 and result.get('held') is True and result.get('status') == 'awaiting_approval':
            # A3b accepted a hold, not a ledger post. Keep the attempt marker:
            # only FL's owner approval/rejection can settle this ticket.
            with api.get_transaction() as cur:
                cur.execute("""UPDATE assistant_drafts SET result=%s
                    WHERE id=%s AND status='pending' AND record_started_at=%s""",
                    (Json(result), str(body.draft_id), attempt))
            return JSONResponse(status_code=202, content=jsonable_encoder({
                'kind': 'awaiting_approval', 'draft_id': str(body.draft_id),
                'result': catalog.safe_result(result)}))
        if not receipt_ok(status, result):
            if 400 <= status < 500 and prior_attempt is None:
                # Clear only this definitive attempt. A concurrent retry changes
                # the marker; a prior lost response remains uncertain even when
                # a later role check denies replay of an already-posted ticket.
                with api.get_transaction() as cur:
                    cur.execute("""UPDATE assistant_drafts SET record_started_at=NULL
                        WHERE id=%s AND status='pending' AND record_started_at=%s
                        AND NOT EXISTS (SELECT 1 FROM write_tickets
                            WHERE ticket_hash=%s AND status='committed')""",
                        (str(body.draft_id), attempt, write_tickets.token_hash(row['ticket'])))
            return JSONResponse(status_code=status if status >= 400 else 502,
                                content=jsonable_encoder(replies.decorate({'kind': 'not_recorded', 'result': catalog.safe_result(result)}, row['card'].get('language', 'en'))))
        with api.get_transaction() as cur:
            cur.execute("UPDATE assistant_drafts SET status='committed',result=%s WHERE id=%s", (Json(result), str(body.draft_id)))
        return {'kind': 'receipt', 'draft_id': str(body.draft_id), 'result': catalog.safe_result(result)}

    @app.post('/assistant/cancel')
    def cancel(body: DraftBody, actor=Depends(authenticate)):
        with api.get_transaction() as cur:
            row = draft_row(api, body.draft_id, actor['id'], lock=True, cur=cur)
            if row['status'] == 'committed':
                fail(409, 'ALREADY_RECORDED', 'This draft already has an FL receipt / Este borrador ya tiene recibo FL.')
            if row['record_started_at']:
                fail(409, 'RECORD_OUTCOME_PENDING', 'Record was attempted. Retry the same ticket to verify the receipt / Reintenta Registrar para verificar el recibo.')
            cur.execute("UPDATE assistant_drafts SET status='cancelled' WHERE id=%s", (str(body.draft_id),))
        return {'kind': 'cancelled', 'draft_id': str(body.draft_id)}

    @app.post('/assistant/attachment')
    async def attachment(session_id: UUID, request: Request, file: UploadFile = File(...),
                         kind: Literal['bol','truck_photo','correction_photo','lot_tag','count_sheet','signed_packing_slip','other'] = Form(...),
                         actor=Depends(authenticate)):
        session_row(api, session_id, actor['id'])
        data = await file.read(MAX_PHOTO + 1)
        await file.close()
        mime = ('image/jpeg' if data.startswith(b'\xff\xd8\xff') else
                'image/png' if data.startswith(b'\x89PNG\r\n\x1a\n') else
                'image/webp' if data[:4] == b'RIFF' and data[8:12] == b'WEBP' else None)
        if not data or len(data) > MAX_PHOTO or not mime or file.content_type != mime:
            fail(422, 'PHOTO_INVALID', 'Use a JPEG, PNG or WebP photo up to 5 MB / Foto de hasta 5 MB.')
        filename = Path(file.filename or 'photo').name[:150]
        status, result = await relay(app, request, 'POST', '/attachments',
            file=(filename, data, mime), form={'kind': kind})
        if status != 200:
            return JSONResponse(status_code=status, content=result)
        aid = result['attachment_id']
        with api.get_transaction() as cur:
            cur.execute('INSERT INTO assistant_attachment_links(session_id,attachment_id,actor_id,filename) VALUES (%s,%s,%s,%s)',
                        (str(session_id), aid, actor['id'], filename))
        return {'id': aid, 'attachment_id': aid, 'filename': filename, 'kind': kind,
                'media_type': mime, 'size': len(data), 'attachment_only': True}

    @app.post('/assistant/attachment/read')
    async def read_attachment(body: AttachmentRead, request: Request, actor=Depends(authenticate)):
        with api.get_transaction() as cur:
            cur.execute('''SELECT l.session_id FROM assistant_attachment_links l
                JOIN assistant_sessions s ON s.id=l.session_id
                JOIN attachments a ON a.id=l.attachment_id
                WHERE l.attachment_id=%s AND l.actor_id=%s AND s.actor_id=%s
                  AND a.uploaded_by_actor_id=%s''',
                (body.attachment_id, actor['id'], actor['id'], actor['id']))
            if not cur.fetchone():
                fail(404, 'ATTACHMENT_NOT_FOUND', 'Attachment unavailable / Adjunto no disponible.')
        status, result = await relay(app, request, 'GET',
                                    '/attachments/' + str(body.attachment_id) + '/file', binary=True)
        if status != 200:
            return JSONResponse(status_code=status, content=result)
        content, headers = result
        return Response(content, media_type=headers.get(b'content-type', b'application/octet-stream').decode(),
                        headers={'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff'})

    @app.post('/assistant/transcribe')
    async def transcribe(file: UploadFile = File(...), actor=Depends(authenticate)):
        data = await file.read(MAX_AUDIO + 1)
        await file.close()
        mime = (file.content_type or '').split(';')[0]
        extensions = {'audio/webm': 'webm', 'video/webm': 'webm', 'audio/mp4': 'mp4',
                      'audio/mpeg': 'mp3', 'audio/wav': 'wav', 'audio/ogg': 'ogg'}
        if mime not in extensions or not data or len(data) > MAX_AUDIO:
            fail(422, 'AUDIO_INVALID', 'Audio must be at most 20 MB / Audio de hasta 20 MB.')
        try:
            result = await openai('audio/transcriptions', file=('dictation.' + extensions[mime], data, mime))
        except HTTPException as exc:
            if isinstance(exc.detail, dict) and exc.detail.get('error_code') == assistant_service.CODE:
                assistant_service.record_outage(api, actor)
            raise
        if not isinstance(result.get('text'), str) or len(result['text']) > 4000:
            fail(502, 'TRANSCRIPT_INVALID', 'Transcript unavailable; type instead / Escribe el mensaje.')
        return {'text': result['text'], 'editable': True, 'sent': False}
