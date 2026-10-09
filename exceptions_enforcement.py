"""A3b: exceptions ENFORCEMENT on top of the 061 tables (design rev 3.7 §5 R2/R3,
§5.1, §7.1, §11 item 7; Michael's A3b decisions 2026-10-08).

One module, called from four named hook points:

* `ticket_actions.validate`  — reason codes (R2), correction thresholds, the
  "never add stock to cover a shortage" refusal, and the shortfall that make/pack
  post instead of blocking (R3);
* `ticket_actions.post`      — shortage flags + SHORTAGE exceptions in the same
  transaction as the ledger post; the review facts on the correction receipt;
* `write_tickets.execute_commit` — the > 500 lb photo gate: post (highlighted) with a
  photo, HOLD the ticket (`awaiting_approval`) without one;
* `register_routes`          — `/exceptions` list/view/resolve/approve/reject.

Rules (owner decisions, verbatim where it matters):
  CORRECTIONS (adjust / found) — `reason_code` REQUIRED from `correction_reasons`
  (8 fixed reasons; legacy values are translated through the 061 map during the
  overlap); `unknown` needs a note. Highlight when |lb| > 500 OR |lb| > 10 % of the
  lot's BOOK balance just before the correction (recomputed under lock at commit);
  book balance ≤ 0 → always highlighted; found: 500 lb rule only, always on the
  weekly view. > 500 lb needs a photo: with one it posts (highlighted); without one
  the ticket is held (`awaiting_approval`) + an owner-approval exception, and the
  hold never expires. Approval IS the commit: one transaction, the preparer stays
  `entered_by`, replay returns the original receipt, never posts twice.
  SHORTAGES (make / pack) — insufficient stock is the ONLY non-blocking condition.
  Prepare warns; commit recomputes under lock; the shortfall posts against the
  confirmed/pinned lot (negative) with ONE shortage_flags row + ONE SHORTAGE
  exception per short lot, due in 2 business days. Never add stock to cover: a
  positive adjust on a lot with an open shortage, or a found for that product, is
  409 SHORTAGE_OPEN_RESOLVE_INSTEAD. Closing a shortage needs EVIDENCE: `counted`
  posts the counted correction in the same transaction as the close, `missing_movement`
  names the committed receipt that put the missing pounds on the SAME lot after the
  shortage opened, `voided` needs the short posting itself voided. Writing off or
  waiving is the owner's call. A positive adjust by the same actor within 30 min
  before a make/pack on an ingredient it consumes is tagged `pre_make_adjust` (§5).
"""
import json
from datetime import datetime, time, timedelta
from typing import Literal, Optional
from zoneinfo import ZoneInfo

from fastapi import Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from psycopg2.extras import Json

import permissions

PLANT_TIMEZONE = ZoneInfo('America/New_York')
LARGE_CORRECTION_LB = 500.0     # |Δ lb| above this → highlighted AND photo required
LARGE_CORRECTION_PCT = 0.10     # |Δ| above this share of the lot's book balance → highlighted
HELD = 'awaiting_approval'      # write_tickets.status (migration 069)
BUSINESS_DAYS = {'SHORTAGE': 2, 'UNIDENTIFIED_LOT': 7}   # Mon–Fri after entry day, due 23:59 plant time
PRE_MAKE_WINDOW = timedelta(minutes=30)   # §5 R3: a positive adjust this close before a make/pack is tagged
EPSILON = 0.0001                # = main.BALANCE_EPSILON
# Prepare refuses these outright (no ticket is issued), like A5's SUPPLIER_REQUIRED:
# a correction without a valid reason, or one that would cover an open shortage.
PREPARE_REFUSALS = frozenset({'REASON_CODE_REQUIRED', 'REASON_CODE_INVALID', 'NOTE_REQUIRED',
                              'REASON_SIGN_MISMATCH', 'SHORTAGE_OPEN_RESOLVE_INSTEAD'})

# Named-actor routes; the two shared keys never reach them (A2: nothing new).
ACTOR_ROUTES = frozenset({
    ('GET', '/exceptions'),
    ('GET', '/exceptions/{exception_id}'),
    ('POST', '/exceptions/{exception_id}/resolve'),
    ('POST', '/exceptions/{exception_id}/approve'),
    ('POST', '/exceptions/{exception_id}/reject'),
})
# §4.3: large-correction approval, late-entry acceptance, proof waiver and the
# pre-make-adjust tag are the owner's to close.
OWNER_KINDS = frozenset({'LARGE_CORRECTION', 'LATE_ENTRY', 'SHIPMENT_PROOF_MISSING', 'PRE_MAKE_ADJUST'})
RESOLUTION_KINDS = {
    'SHORTAGE': ('counted', 'missing_movement', 'voided'),
    'NEGATIVE_BALANCE': ('counted', 'missing_movement', 'voided'),
    'UNIDENTIFIED_LOT': ('identified', 'written_off'),
    'LATE_ENTRY': ('acknowledged',),
    'SHIPMENT_PROOF_MISSING': ('photo_attached', 'waived'),
    'PRE_MAKE_ADJUST': ('acknowledged',),
}
DEFAULT_RESOLUTIONS = ('resolved', 'waived')
WAIVE_KINDS = frozenset({'waived', 'written_off'})
OWNER_RESOLUTIONS = WAIVE_KINDS            # closing without evidence is the owner's call (any role may list)
EVIDENCE_KINDS = frozenset({'SHORTAGE', 'NEGATIVE_BALANCE'})   # closing needs a matching correction or receipt
MAX_COUNTED_LB = 100_000.0                 # a physical count above this is a typo, not a count (P2: finite, sane, 422)
UNIT_TO_LB = {'lb': 1.0, 'lbs': 1.0, 'pound': 1.0, 'pounds': 1.0,
              'kg': 2.20462262185, 'kgs': 2.20462262185, 'kilogram': 2.20462262185, 'kilograms': 2.20462262185,
              'g': 1 / 453.59237, 'gram': 1 / 453.59237, 'grams': 1 / 453.59237,
              'oz': 1 / 16, 'ounce': 1 / 16, 'ounces': 1 / 16}


def fail(http_status, code, message, **extra):
    raise HTTPException(http_status, {'error_code': code, 'message': message, **extra})


# ---------------------------------------------------------------------------
# Business-day deadlines (C). Same clock as A5's unidentified-lot rule: count
# weekdays AFTER the local entry day, due 23:59 America/New_York. Plant-closure
# days are not skipped yet (FOLLOWUPS P1.12).
# ---------------------------------------------------------------------------
def business_deadline(entered_at, business_days):
    if entered_at.tzinfo is None:
        raise ValueError('entered_at must have a timezone')
    day = entered_at.astimezone(PLANT_TIMEZONE).date()
    remaining = int(business_days)
    while remaining > 0:
        day += timedelta(days=1)
        if day.weekday() < 5:
            remaining -= 1
    return datetime.combine(day, time(23, 59), tzinfo=PLANT_TIMEZONE)


def floor_owner_id(cur):
    """R1: the floor owns shortages; Arturo by name while he is the only floor actor (A5 rule)."""
    cur.execute("""SELECT id FROM actors WHERE role='floor' AND active
                   ORDER BY CASE WHEN lower(name)='arturo' THEN 0 ELSE 1 END, id LIMIT 1""")
    row = cur.fetchone()
    return row['id'] if row else None


# ---------------------------------------------------------------------------
# Units. Ticket quantities are in the product's ledger unit (pounds for every
# weight-labelled product). The 500 lb rules need pounds: convert through the
# catalog; a count-based product without a case weight has no lb equivalent,
# so only the 10 % rule applies to it (reported as lb_rule='not_applicable').
# ---------------------------------------------------------------------------
def to_lb(product, quantity, unit):
    unit = (unit or product.get('uom') or 'lb').strip().lower().rstrip('.')
    if unit in UNIT_TO_LB:
        return float(quantity) * UNIT_TO_LB[unit]
    if unit in ('case', 'cases', 'cs'):
        weight = product.get('case_size_lb') or product.get('default_case_weight_lb')
        return float(quantity) * float(weight) if weight else None
    if unit in ('bag', 'bags'):
        weight = product.get('pack_size_lbs')
        return float(quantity) * float(weight) if weight else None
    return None


# ---------------------------------------------------------------------------
# A. Reason codes (R2, §5.1)
# ---------------------------------------------------------------------------
_REASON_COLUMNS = 'code,label_en,label_es,applies_to,adjust_sign,note_required,active'
_COUNT_LIKE = r'(physical.*count|physical inventory|inventory count|cycle count|count correction|recon)'


def normalise(value):
    return ' '.join(str(value or '').split()).lower()


def _valid_reasons(cur, source):
    cur.execute(f'SELECT code,label_en,label_es,note_required,adjust_sign FROM correction_reasons '
                f'WHERE active AND %s=ANY(applies_to) ORDER BY sort_order,code', (source,))
    return [dict(r) for r in cur.fetchall()]


def resolve_reason(cur, source, raw, *, note=None, delta_lb=None):
    """The fixed-list reason for a ticket correction, or a 422.

    `source` is 'adjust' or 'found'. The 8 codes are accepted as written
    (case/space-insensitive); a legacy value a client still sends during the
    overlap is translated through `correction_reason_legacy_codes` (§5.1) and the
    translation is reported. `unknown` (and any future note_required reason)
    needs a note; `adjust_sign` ('positive'/'negative') is enforced on the delta.
    """
    if raw is None or not str(raw).strip():
        fail(422, 'REASON_CODE_REQUIRED', 'A correction needs a reason_code from the fixed list.',
             message_es='Una corrección necesita un reason_code de la lista fija.',
             valid_reasons=_valid_reasons(cur, source))
    norm = normalise(raw)
    cur.execute(f'SELECT {_REASON_COLUMNS} FROM correction_reasons WHERE code=%s', (norm.replace(' ', '_'),))
    row = cur.fetchone()
    legacy = None
    if row is None:
        cur.execute('SELECT reason_code, note_prefill FROM correction_reason_legacy_codes '
                    'WHERE source=%s AND legacy_code=%s', (source, norm))
        legacy = cur.fetchone()
        if legacy:
            cur.execute(f'SELECT {_REASON_COLUMNS} FROM correction_reasons WHERE code=%s', (legacy['reason_code'],))
            row = cur.fetchone()
    if row is None or not row['active'] or source not in row['applies_to']:
        label = row['label_en'] if row else str(raw)
        fail(422, 'REASON_CODE_INVALID',
             f'{label!r} is not a correction reason for {source}; use one of the fixed list.',
             message_es=f'{label!r} no es un motivo de corrección para {source}; use uno de la lista fija.',
             reason_code=str(raw), valid_reasons=_valid_reasons(cur, source))
    effective_note = (note or '').strip() or ((legacy or {}).get('note_prefill') or None)
    if row['note_required'] and not effective_note:
        fail(422, 'NOTE_REQUIRED', f'{row["label_en"]} needs a note saying what happened.',
             message_es=f'{row["label_es"]} necesita una nota que explique qué pasó.', reason_code=row['code'])
    if delta_lb is not None:
        if row['adjust_sign'] == 'positive' and float(delta_lb) < 0:
            fail(422, 'REASON_SIGN_MISMATCH', f'{row["label_en"]} only adds stock; this correction removes {abs(float(delta_lb))}.',
                 message_es=f'{row["label_es"]} solo agrega inventario.', reason_code=row['code'])
        if row['adjust_sign'] == 'negative' and float(delta_lb) > 0:
            fail(422, 'REASON_SIGN_MISMATCH', f'{row["label_en"]} only removes stock; this correction adds {float(delta_lb)}.',
                 message_es=f'{row["label_es"]} solo quita inventario.', reason_code=row['code'])
    return {'code': row['code'], 'label_en': row['label_en'], 'label_es': row['label_es'],
            'note_required': row['note_required'], 'input': str(raw),
            'legacy_code': norm if legacy else None, 'note': effective_note}


def legacy_reason_code(cur, source, raw):
    """Direct (legacy) routes: stamp `transactions.reason_code` with the 061 backfill
    tiers (legacy map → already a code → count-like text → unknown) so no new row
    is left NULL (P1.8). Never raises and never changes the route's behaviour; None
    only when the 061 seed is absent (then the column stays NULL, as today)."""
    norm = normalise(raw)
    cur.execute('''SELECT (SELECT code FROM correction_reasons WHERE code = COALESCE(
                       m.reason_code, a.code,
                       CASE WHEN %s='adjust' AND %s ~* %s THEN 'physical_count' END,
                       'unknown')) AS code
                   FROM (SELECT 1) one
                   LEFT JOIN correction_reason_legacy_codes m ON m.source=%s AND m.legacy_code=%s
                   LEFT JOIN correction_reasons a ON a.code=%s''',
                (source, norm, _COUNT_LIKE, source, norm, norm.replace(' ', '_')))
    row = cur.fetchone()
    return row['code'] if row else None


# ---------------------------------------------------------------------------
# A. Correction thresholds (highlight / photo) and B. cover-up refusal
# ---------------------------------------------------------------------------
def correction_review(api, action, product, quantity, *, book_balance_before=None, uom=None, reason_code=None):
    """The facts the weekly view highlights on, computed from the state the caller
    read (prepare: current; commit: under FOR UPDATE on the lot). `unknown` is
    highlighted whatever the size (§5.1)."""
    unit = uom or api.ledger_quantity_unit(product.get('uom'))
    lb = to_lb(product, quantity, unit)
    magnitude = abs(lb) if lb is not None else None
    review = {'action': action, 'quantity': float(quantity), 'unit': unit, 'lb_equivalent': lb,
              'book_balance_before': None if book_balance_before is None else float(book_balance_before),
              'pct_of_book': None, 'rules': [], 'lb_rule': 'applied' if lb is not None else 'not_applicable',
              'highlighted': False, 'photo_required': False, 'weekly_view': True}
    if action == 'adjust':
        book = float(book_balance_before or 0.0)
        if book <= EPSILON:
            review['rules'].append('BOOK_BALANCE_NOT_POSITIVE')
        else:
            review['pct_of_book'] = round(abs(float(quantity)) / book, 4)
            if abs(float(quantity)) > LARGE_CORRECTION_PCT * book + EPSILON:
                review['rules'].append('OVER_10_PERCENT_OF_BOOK')
    if magnitude is not None and magnitude > LARGE_CORRECTION_LB + EPSILON:
        review['rules'].append('OVER_500_LB')
        review['photo_required'] = True
    if reason_code == 'unknown':
        review['rules'].append('REASON_UNKNOWN')
    review['reason_code'] = reason_code
    review['highlighted'] = bool(review['rules'])
    review['message'] = None
    if review['photo_required']:
        review['message'] = (f'{product["name"]}: corrections over {int(LARGE_CORRECTION_LB)} lb need a photo; '
                             'without one the entry is held for the owner to approve.')
    elif review['highlighted']:
        review['message'] = f'{product["name"]}: this correction is highlighted on the owner\'s weekly view.'
    return review


def refuse_cover_up(cur, action, product_id, lot_id, delta):
    """R3 "never add stock to get around it": positive adjust on a lot with an open
    shortage, or found stock for a product with one → 409; resolve the shortage first."""
    if action == 'adjust':
        if float(delta) <= 0:
            return
        cur.execute("""SELECT e.id, e.due_at, f.short_lb FROM exceptions e
                       LEFT JOIN shortage_flags f ON f.exception_id=e.id
                       WHERE e.kind='SHORTAGE' AND e.lot_id=%s AND e.status IN ('open','escalated')
                       ORDER BY e.id LIMIT 1""", (lot_id,))
    elif action == 'found':
        cur.execute("""SELECT e.id, e.due_at, f.short_lb FROM exceptions e
                       LEFT JOIN shortage_flags f ON f.exception_id=e.id
                       WHERE e.kind='SHORTAGE' AND e.product_id=%s AND e.status IN ('open','escalated')
                       ORDER BY e.id LIMIT 1""", (product_id,))
    else:
        return
    row = cur.fetchone()
    if row:
        fail(409, 'SHORTAGE_OPEN_RESOLVE_INSTEAD',
             f'Shortage #{row["id"]} is open on this stock; resolve it (count, missing movement or void) '
             'instead of adding inventory.',
             message_es=f'El faltante #{row["id"]} está abierto; resuélvalo en vez de agregar inventario.',
             exception_id=row['id'], short_lb=float(row['short_lb']) if row['short_lb'] is not None else None)


# ---------------------------------------------------------------------------
# B. Shortfall assignment (prepare) and recomputation (commit)
# ---------------------------------------------------------------------------
def assign_shortfall(plan, product_id, shortfall, candidates, pinned_lot_id=None):
    """Put the uncovered pounds on ONE lot so the post balances and the flag names a lot.

    Pinned/confirmed lot → that lot. Otherwise the last FIFO lot already in the
    plan (the one that ran out), else the newest lot of the product. No lot at all
    → the caller keeps the INSUFFICIENT_STOCK blocker: there is nothing to go negative.
    """
    if shortfall <= EPSILON:
        return plan
    target = None
    if pinned_lot_id is not None:
        target = next((c for c in candidates if c['id'] == pinned_lot_id), None)
    if target is None:
        planned = [item for item in plan if item['product_id'] == product_id]
        if planned:
            target = next((c for c in candidates if c['id'] == planned[-1]['lot_id']), None)
    if target is None and candidates:
        target = candidates[-1]
    if target is None:
        return None
    for item in plan:
        if item['product_id'] == product_id and item['lot_id'] == target['id']:
            item['quantity_lb'] = float(item['quantity_lb']) + float(shortfall)
            item['short_lb'] = round(float(item.get('short_lb', 0.0)) + float(shortfall), 4)
            return plan
    plan.append({'product_id': product_id, 'lot_id': target['id'], 'quantity_lb': float(shortfall),
                 'short_lb': round(float(shortfall), 4)})
    return plan


def pack_shortfall(cur, source, draft, pinned, primary):
    """Pack prepare with insufficient batch stock: return the source plan with the
    shortfall pinned. Explicit allocations each carry their own uncovered pounds
    (requested − positive balance); FIFO puts the remainder on the lot that ran out."""
    cur.execute('''SELECT id, lot_code FROM lots WHERE product_id=%s AND status IS DISTINCT FROM 'merged'
                   ORDER BY COALESCE(received_at, created_at), id''', (source['id'],))
    candidates = [dict(r) for r in cur.fetchall()]
    total = float(draft['total_lb'])
    if pinned:
        by_code = {c['lot_code'].lower(): c for c in candidates}
        result = []
        for entry in draft['allocations']:
            lot = by_code.get((entry.get('lot_code') or '').lower())
            if lot is None:
                fail(409, 'LOT_NOT_FOUND', f'{source["name"]} lot {entry.get("lot_code")} is missing; resolve an existing lot.')
            requested = float(entry['allocated_lb'])
            available = max(float(entry.get('available_lb') or 0.0), 0.0)
            item = {'product_id': source['id'], 'lot_id': lot['id'], 'quantity_lb': requested}
            if requested - available > EPSILON:
                item['short_lb'] = round(requested - available, 4)
            result.append(item)
        return result
    shortfall = total - sum(float(a['quantity_lb']) for a in primary)
    if shortfall > EPSILON and assign_shortfall(primary, source['id'], shortfall, candidates) is None:
        fail(409, 'INSUFFICIENT_STOCK', f'{source["name"]} needs {total} lb and has no lot to record the shortage against; make it first.')
    return primary


def shortfalls(input_plan, states):
    """Per plan item: what the post will leave uncovered given the balances just read."""
    lots = {s['id']: s for s in states}
    result = []
    for item in sorted(input_plan or [], key=lambda i: i['lot_id']):
        lot = lots[item['lot_id']]
        on_hand = float(lot['on_hand_lb'])
        short = float(item['quantity_lb']) - max(on_hand, 0.0)
        if short > EPSILON:
            result.append({'product_id': item['product_id'], 'product_name': lot['product_name'],
                           'lot_id': lot['id'], 'lot_code': lot['lot_code'],
                           'needed_lb': float(item['quantity_lb']), 'on_hand_lb': on_hand,
                           'short_lb': round(short, 4), 'balance_after_lb': round(on_hand - float(item['quantity_lb']), 4)})
    return result


def shortage_warnings(draft):
    """Prepare: "will create shortage" — a warning, never a blocker (R3)."""
    return [{'code': 'WILL_CREATE_SHORTAGE', 'requires_ack': False,
             'message': (f'{s["product_name"]} lot {s["lot_code"]} has {s["on_hand_lb"]} lb; posting {s["needed_lb"]} lb '
                         f'leaves it short {s["short_lb"]} lb. A shortage flag opens for the floor to resolve '
                         f'within {BUSINESS_DAYS["SHORTAGE"]} business days.'),
             'message_es': (f'{s["product_name"]} lote {s["lot_code"]} tiene {s["on_hand_lb"]} lb; quedará con un '
                            f'faltante de {s["short_lb"]} lb que el piso debe resolver en {BUSINESS_DAYS["SHORTAGE"]} días hábiles.'),
             'refs': {'lot_id': s['lot_id'], 'short_lb': s['short_lb'], 'on_hand_lb': s['on_hand_lb']}}
            for s in draft.get('shortages') or []]


def record_shortages(cur, transaction_id, shortages, *, ticket_id, receipt_number, action, actor_id):
    """One shortage_flags row + one exceptions(SHORTAGE) per short lot, in the posting
    transaction. Replays never get here (the ticket returns its stored receipt); the
    069 unique indexes make a second row impossible anyway."""
    if not shortages:
        return []
    cur.execute('SELECT created_at FROM transactions WHERE id=%s', (transaction_id,))
    entered_at = cur.fetchone()['created_at']
    due_at = business_deadline(entered_at, BUSINESS_DAYS['SHORTAGE'])
    owner = floor_owner_id(cur)
    recorded = []
    for s in shortages:
        detail = {'short_lb': s['short_lb'], 'needed_lb': s['needed_lb'], 'on_hand_before_lb': s['on_hand_lb'],
                  'balance_after_lb': s['balance_after_lb'], 'action': action, 'entered_by_actor_id': actor_id,
                  'clock': f'{BUSINESS_DAYS["SHORTAGE"]} business days; Mon–Fri; entry date excluded; 23:59 America/New_York'}
        cur.execute('''INSERT INTO exceptions(kind,status,severity,product_id,lot_id,transaction_id,receipt_number,
                           ticket_id,detail,owner_actor_id,opened_at,due_at)
                       VALUES ('SHORTAGE','open','warn',%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id''',
                    (s['product_id'], s['lot_id'], transaction_id, receipt_number, ticket_id, Json(detail),
                     owner, entered_at, due_at))
        exception_id = cur.fetchone()['id']
        cur.execute('''INSERT INTO shortage_flags(transaction_id,product_id,lot_id,short_lb,exception_id,
                           opened_at,due_at,owner_actor_id)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id''',
                    (transaction_id, s['product_id'], s['lot_id'], s['short_lb'], exception_id, entered_at, due_at, owner))
        recorded.append(dict(s, exception_id=exception_id, shortage_flag_id=cur.fetchone()['id'],
                             due_at=due_at, owner_actor_id=owner))
    return recorded


def pinned_lot_rows(cur, product_id, lot_ids):
    """Pack commit: pinned lots at or below zero are not in the FIFO list; shape them
    like `available_lots_for_product` rows so the shared posting core consumes them."""
    rows = []
    for lot_id in sorted(lot_ids):
        cur.execute('SELECT id, lot_code FROM lots WHERE id=%s AND product_id=%s FOR UPDATE', (lot_id, product_id))
        lot = cur.fetchone()
        if not lot:
            continue
        cur.execute('''SELECT COALESCE(SUM(tl.quantity_lb), 0) AS balance
                       FROM ledger_current_transaction_lines tl
                       JOIN ledger_current_transactions ct ON ct.id = tl.transaction_id
                       WHERE ct.effective_status = 'posted' AND tl.lot_id = %s''', (lot_id,))
        on_hand = float(cur.fetchone()['balance'])
        rows.append({'id': lot_id, 'lot_id': lot_id, 'lot_code': lot['lot_code'], 'on_hand': on_hand,
                     'available': on_hand, 'takeable': 0.0, 'reserved_lot_lb': 0.0, 'reserved_others_lot': 0.0,
                     'reserved_this_line_lot': 0.0, 'takeable_unpinned': 0.0, 'foreign_sku_shadow_lb': 0.0})
    return rows


# ---------------------------------------------------------------------------
# A. The photo gate at commit: hold, replay the hold, or release it
# ---------------------------------------------------------------------------
def hold_response(cur, ticket_id):
    cur.execute("SELECT id, status, due_at, detail FROM exceptions WHERE kind='LARGE_CORRECTION' AND ticket_id=%s", (ticket_id,))
    exc = cur.fetchone()
    return {'held': True, 'status': HELD, 'ticket_id': ticket_id, 'exception_id': exc['id'] if exc else None,
            'exception_status': exc['status'] if exc else None,
            'error_code': 'PHOTO_REQUIRED',
            'message': (f'Corrections over {int(LARGE_CORRECTION_LB)} lb need a photo. Nothing was posted; the entry is '
                        'held for the owner to approve, or commit again with attachment_ref.'),
            'message_es': (f'Las correcciones de más de {int(LARGE_CORRECTION_LB)} lb necesitan foto. No se registró nada; '
                           'queda en espera de la aprobación del dueño, o vuelva a confirmar con attachment_ref.'),
            'correction_review': (exc['detail'] or {}).get('correction_review') if exc else None}


def hold_ticket(api, cur, row, actor, draft, acknowledged, payload_hash):
    """Nothing posts. Ticket → awaiting_approval (never expires), ONE
    exceptions(LARGE_CORRECTION, block) linked to ticket_id + payload_hash."""
    review = draft['correction_review']
    detail = {'payload_hash': payload_hash, 'action': row['action'], 'correction_review': review,
              'reason_code': draft.get('reason_code'), 'note': draft.get('note'),
              'product_id': draft.get('product_id'), 'lot_id': draft.get('lot_id'),
              'prepared_by_actor_id': row['actor_id'], 'prepared_by': row['operator_id'],
              'client_source': row['client_source'], 'happened_at': row['payload'].get('occurred_at')}
    cur.execute('''INSERT INTO exceptions(kind,status,severity,product_id,lot_id,ticket_id,detail,owner_actor_id)
                   VALUES ('LARGE_CORRECTION','open','block',%s,%s,%s,%s,%s)
                   ON CONFLICT (ticket_id) WHERE kind='LARGE_CORRECTION' DO NOTHING''',
                (draft.get('product_id'), draft.get('lot_id'), row['id'], Json(detail), permissions.owner_actor_id(cur)))
    cur.execute('''UPDATE write_tickets SET status=%s, acknowledged=%s,
                       draft = draft || jsonb_build_object('correction_review', %s::jsonb, 'held_at', clock_timestamp())
                   WHERE id=%s AND status IN ('prepared', %s)''',
                (HELD, Json(sorted(set(acknowledged))), json.dumps(review), row['id'], HELD))
    return hold_response(cur, row['id'])


def photo_required(draft):
    """A shared key (no named preparer) gets the plain refusal instead of a hold: there
    is nobody to post *for*, and the owner approves people, not keys. The ticket
    stays `prepared`; committing again with attachment_ref posts it."""
    review = draft['correction_review']
    message = (f'Corrections over {int(LARGE_CORRECTION_LB)} lb need a photo. Nothing was posted; commit again '
               'with attachment_ref (a shared key cannot hold an entry for owner approval — use your own key).')
    return {'detail': {'error_code': 'PHOTO_REQUIRED', 'message': message,
                       'message_es': (f'Las correcciones de más de {int(LARGE_CORRECTION_LB)} lb necesitan foto. No se '
                                      'registró nada; vuelva a confirmar con attachment_ref.'),
                       'held': False, 'blockers': [{'code': 'PHOTO_REQUIRED', 'message': message}],
                       'correction_review': review}}


def release_hold(cur, ticket_id, *, resolution_kind, actor_id, note, response, attachment_ref=None):
    """Close the LARGE_CORRECTION exception when the held post goes through (photo or approval)."""
    cur.execute("""UPDATE exceptions SET status='resolved', resolved_at=clock_timestamp(), resolved_by_actor_id=%s,
                       resolution_kind=%s, resolution_note=%s, resolution_ticket_id=%s,
                       transaction_id=%s, receipt_number=%s,
                       detail = detail || %s::jsonb
                   WHERE kind='LARGE_CORRECTION' AND ticket_id=%s AND status IN ('open','escalated') RETURNING id""",
                (actor_id, resolution_kind, note, ticket_id, response.get('transaction_id'), response.get('receipt_number'),
                 json.dumps({'attachment_ref': attachment_ref, 'released_by': resolution_kind}), ticket_id))
    row = cur.fetchone()
    return row['id'] if row else None


def preparer_request(actor_row):
    """A Request whose identity is the PREPARER, so an owner's approval posts with
    entered_by = the person who prepared it (operator_id, entered_by_actor_id, trace)."""
    request = Request({'type': 'http', 'method': 'POST', 'path': '/exceptions/approve', 'headers': [],
                       'query_string': b'', 'route': None})
    request.state.actor = dict(actor_row)
    request.state.key_kind = 'actor'
    return request


# ---------------------------------------------------------------------------
# B. The pre-make-adjust tag (§5 R3, weekly view item 3)
# ---------------------------------------------------------------------------
def record_pre_make_adjusts(cur, transaction_id, input_plan, *, ticket_id, receipt_number, action, actor_id, operator_id):
    """A POSITIVE adjust by the same actor on an ingredient this make/pack consumes,
    entered within 30 min before it, is `pre_make_adjust`: one exceptions(PRE_MAKE_ADJUST,
    info) per adjust posting (070 unique index — a second make never re-tags it), owner
    acknowledges on the weekly view. Same-actor = same entered_by_actor_id, or the same
    operator_id for the shared key. Clock = entry time (created_at), not occurred_at."""
    products = sorted({item['product_id'] for item in input_plan or []})
    if not products:
        return []
    cur.execute('SELECT created_at FROM transactions WHERE id=%s', (transaction_id,))
    entered_at = cur.fetchone()['created_at']
    consumed = {item['lot_id'] for item in input_plan or []}
    cur.execute('''SELECT t.id, t.created_at, t.receipt_number, t.ticket_id, tl.product_id, tl.lot_id,
                          l.lot_code, p.name AS product_name, tl.quantity_lb
                   FROM transactions t
                   JOIN ledger_current_transactions ct ON ct.id = t.id AND ct.effective_status = 'posted'
                   JOIN ledger_current_transaction_lines tl ON tl.transaction_id = t.id
                   JOIN lots l ON l.id = tl.lot_id
                   JOIN products p ON p.id = tl.product_id
                   WHERE t.type = 'adjust' AND tl.quantity_lb > 0 AND tl.product_id = ANY(%s)
                     AND t.created_at >= %s - %s::interval AND t.created_at < %s
                     AND ((%s::int IS NOT NULL AND t.entered_by_actor_id = %s)
                          OR (%s::int IS NULL AND t.entered_by_actor_id IS NULL AND t.operator_id = %s))
                   ORDER BY t.id''',
                (products, entered_at, PRE_MAKE_WINDOW, entered_at, actor_id, actor_id, actor_id, operator_id))
    tagged = []
    for adj in cur.fetchall():
        minutes = round((entered_at - adj['created_at']).total_seconds() / 60, 1)
        detail = {'pre_make_adjust': True, 'adjust_lb': float(adj['quantity_lb']), 'minutes_before': minutes,
                  'same_lot': adj['lot_id'] in consumed, 'actor_id': actor_id, 'operator_id': operator_id,
                  'followed_by': {'action': action, 'transaction_id': transaction_id, 'receipt_number': receipt_number,
                                  'ticket_id': ticket_id},
                  'rule': f'positive adjust within {int(PRE_MAKE_WINDOW.total_seconds() // 60)} min before a {action} '
                          'on the same ingredient by the same actor (§5 R3)'}
        cur.execute('''INSERT INTO exceptions(kind,status,severity,product_id,lot_id,transaction_id,receipt_number,
                           ticket_id,detail,owner_actor_id,opened_at)
                       VALUES ('PRE_MAKE_ADJUST','open','info',%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (transaction_id) WHERE kind='PRE_MAKE_ADJUST' DO NOTHING RETURNING id''',
                    (adj['product_id'], adj['lot_id'], adj['id'], adj['receipt_number'], adj['ticket_id'], Json(detail),
                     permissions.owner_actor_id(cur), entered_at))
        inserted = cur.fetchone()
        if inserted is None:
            cur.execute("SELECT id FROM exceptions WHERE kind='PRE_MAKE_ADJUST' AND transaction_id=%s", (adj['id'],))
            inserted = cur.fetchone()
        tagged.append({'exception_id': inserted['id'], 'adjust_transaction_id': adj['id'],
                       'adjust_receipt_number': adj['receipt_number'], 'product_id': adj['product_id'],
                       'product_name': adj['product_name'], 'lot_id': adj['lot_id'], 'lot_code': adj['lot_code'],
                       'adjust_lb': float(adj['quantity_lb']), 'minutes_before': minutes, 'same_lot': adj['lot_id'] in consumed})
    return tagged


# ---------------------------------------------------------------------------
# E. /exceptions routes
# ---------------------------------------------------------------------------
def _non_blank(value):
    if value is not None and not str(value).strip():
        raise ValueError('must not be blank')
    return value.strip() if isinstance(value, str) else value


class ResolveRequest(BaseModel):
    resolution_kind: str = Field(min_length=1, max_length=40)
    note: str = Field(min_length=1, max_length=2000)
    receipt_number: Optional[str] = Field(None, max_length=40)
    # 'counted': what is physically on the lot now — finite and sane, or 422 (NaN/±inf/1e400/absurd never reach SQL).
    counted_lb: Optional[float] = Field(None, ge=0, le=MAX_COUNTED_LB, allow_inf_nan=False)
    attachment_ref: Optional[str] = Field(None, min_length=1, max_length=500)   # photo for a > 500 lb counted correction

    _strip = field_validator('note', 'receipt_number', 'attachment_ref')(_non_blank)

    class Config:
        extra = 'forbid'


class DecisionRequest(BaseModel):
    note: Optional[str] = Field(None, max_length=2000)

    _strip = field_validator('note')(_non_blank)

    class Config:
        extra = 'forbid'


# ---------------------------------------------------------------------------
# E. Resolution evidence (Codex review of PR #94, item 1)
# ---------------------------------------------------------------------------
def _receipt_ticket(cur, receipt_number):
    cur.execute('SELECT id, action, status, committed_at FROM write_tickets WHERE receipt_number=%s', (receipt_number,))
    ticket = cur.fetchone()
    if not ticket:
        fail(404, 'RECEIPT_NOT_FOUND', 'No recorded receipt has this number.')
    return ticket


def shortage_evidence(api, cur, request, actor, row, body):
    """What closes a shortage. Returns (resolution_ticket_id, detail patch) or fails.

    counted          — `counted_lb` (the physical count now) is required; the counted
                       correction (reason physical_count, entered by the resolver) posts
                       HERE, in the same transaction as the close — the only way a
                       positive adjust ever reaches a lot with an open shortage. Over
                       500 lb it needs `attachment_ref` like any correction.
    missing_movement — `receipt_number` of a COMMITTED receipt whose still-effective
                       posting put stock on the SAME lot after the shortage opened. Its
                       pounds are CONSUMED: `shortage_evidence_claims` (071) records how
                       much of each receipt every resolution used, under FOR UPDATE on the
                       receipt's ticket, so one 20 lb receipt can never close 26 lb of
                       shortages. A receipt that covers less than what is still open is
                       claimed in part and the shortage STAYS OPEN for the remainder
                       (returns closes=False). Anything else is RECEIPT_NOT_MATCHING.
    voided           — the short posting itself is no longer effective (voided).
    Returns (resolution_ticket_id, detail patch, closes).
    """
    cur.execute('SELECT id, short_lb FROM shortage_flags WHERE exception_id=%s', (row['id'],))
    flag = cur.fetchone()
    short_lb = float(flag['short_lb']) if flag else None
    kind = body.resolution_kind
    if kind == 'counted':
        if body.counted_lb is None:
            fail(422, 'COUNT_REQUIRED', "Resolving as 'counted' needs counted_lb: what is physically on the lot now.",
                 message_es="Para resolver como 'contado' indique counted_lb: lo que hay físicamente en el lote.")
        if row['lot_id'] is None:
            fail(409, 'LOT_REQUIRED', 'This exception names no lot to count.')
        permissions.require('adjust', actor)
        cur.execute('SELECT id, lot_code, product_id, status FROM lots WHERE id=%s FOR UPDATE', (row['lot_id'],))
        lot = cur.fetchone()
        if not lot or lot['status'] == 'merged':
            fail(409, 'LOT_NOT_FOUND', 'The short lot is missing or merged; resolve it on the surviving lot.')
        cur.execute('SELECT * FROM products WHERE id=%s', (lot['product_id'],))
        product = dict(cur.fetchone())
        book = float(api.lot_on_hand(cur, lot['id']))
        delta = round(float(body.counted_lb) - book, 4)
        review = correction_review(api, 'adjust', product, delta, book_balance_before=book, reason_code='physical_count')
        resolution = {'kind': 'counted', 'counted_lb': float(body.counted_lb), 'book_before_lb': book, 'delta_lb': delta,
                      'short_lb': short_lb, 'correction_review': review, 'attachment_ref': body.attachment_ref,
                      'transaction_id': None, 'new_balance_lb': book}
        if abs(delta) > EPSILON:
            if review['photo_required'] and not body.attachment_ref:
                fail(422, 'PHOTO_REQUIRED',
                     f'The counted correction is {abs(delta)} lb, over {int(LARGE_CORRECTION_LB)} lb: attach a photo (attachment_ref).',
                     message_es=f'La corrección contada es de {abs(delta)} lb, más de {int(LARGE_CORRECTION_LB)} lb: adjunte foto.',
                     correction_review=review)
            cur.execute("SELECT label_en, label_es FROM correction_reasons WHERE code='physical_count'")
            labels = cur.fetchone() or {'label_en': 'Physical count', 'label_es': 'Conteo físico'}
            req = api.AdjustRequest(mode='commit', product_name=product['name'], lot_code=lot['lot_code'], adjustment_lb=delta,
                                    reason=labels['label_en'], reason_es=labels['label_es'])
            posted = api._adjust_commit_core(cur, req, request, api.get_plant_now(), 'database', product=product,
                                             lot_id=lot['id'], reason_code='physical_count',
                                             note=f'Shortage #{row["id"]} counted {float(body.counted_lb)} lb: {body.note}')
            if hasattr(posted, 'status_code'):
                fail(409, 'CORRECTION_BLOCKED', 'The counted correction is blocked by the product rules.',
                     blocked=json.loads(bytes(posted.body).decode()))
            resolution.update(transaction_id=posted['transaction_id'], new_balance_lb=float(posted['new_balance_lb']))
        return None, {'resolution': resolution}, True
    if kind == 'missing_movement':
        if not body.receipt_number:
            fail(422, 'RECEIPT_REQUIRED', "Resolving as 'missing_movement' needs receipt_number: the receive/make that was never entered.",
                 message_es="Para resolver como 'movimiento faltante' indique receipt_number del movimiento que faltaba.")
        ticket = _receipt_ticket(cur, body.receipt_number)
        # Every claim on this receipt is serialized on its ticket row (lock held to COMMIT),
        # so two resolutions cannot both spend the same pounds.
        cur.execute('SELECT id FROM write_tickets WHERE id=%s FOR UPDATE', (ticket['id'],))
        problems = []
        if ticket['status'] != 'committed':
            problems.append(f'receipt {body.receipt_number} is {ticket["status"]}, not committed')
        cur.execute('''SELECT t.id, t.created_at, COALESCE(SUM(tl.quantity_lb), 0) AS added_lb
                       FROM transactions t
                       JOIN ledger_current_transactions ct ON ct.id = t.id AND ct.effective_status = 'posted'
                       JOIN ledger_current_transaction_lines tl ON tl.transaction_id = t.id
                       WHERE t.ticket_id = %s AND tl.lot_id = %s AND tl.quantity_lb > 0
                       GROUP BY t.id, t.created_at ORDER BY t.id''', (ticket['id'], row['lot_id']))
        lines = cur.fetchall()
        added = sum(float(line['added_lb']) for line in lines)
        cur.execute('SELECT COALESCE(SUM(claimed_lb), 0) AS used FROM shortage_evidence_claims WHERE evidence_ticket_id=%s AND lot_id=%s',
                    (ticket['id'], row['lot_id']))
        used = float(cur.fetchone()['used'])
        available = round(added - used, 4)
        cur.execute('SELECT COALESCE(SUM(claimed_lb), 0) AS claimed FROM shortage_evidence_claims WHERE exception_id=%s', (row['id'],))
        already = float(cur.fetchone()['claimed'])
        remaining = None if short_lb is None else round(short_lb - already, 4)
        if not lines:
            problems.append('it puts no stock on the short lot (same lot required)')
        else:
            if any(line['created_at'] < row['opened_at'] for line in lines):
                problems.append('it was entered before the shortage opened, so it was already counted')
            if available <= EPSILON:
                problems.append(f'its {added} lb on this lot are already used by other shortage resolutions ({used} lb claimed)')
        if problems:
            fail(409, 'RECEIPT_NOT_MATCHING',
                 f'Receipt {body.receipt_number} does not explain shortage #{row["id"]}: ' + '; '.join(problems) + '.',
                 message_es=f'El recibo {body.receipt_number} no explica el faltante #{row["id"]}.',
                 problems=problems, receipt_number=body.receipt_number, short_lb=short_lb,
                 receipt_added_lb=added, receipt_used_lb=used, remaining_lb=remaining)
        claim = available if remaining is None else min(available, remaining)
        claim_id = None
        if claim > EPSILON:
            cur.execute('''INSERT INTO shortage_evidence_claims(exception_id, shortage_flag_id, evidence_ticket_id, evidence_receipt_number,
                               lot_id, claimed_lb, claimed_by_actor_id)
                           VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING id''',
                        (row['id'], flag['id'] if flag else None, ticket['id'], body.receipt_number, row['lot_id'], claim, actor['id']))
            claim_id = cur.fetchone()['id']
        left = None if remaining is None else round(remaining - claim, 4)
        closes = left is None or left <= EPSILON
        resolution = {'kind': 'missing_movement', 'receipt_number': body.receipt_number,
                      'transaction_ids': [line['id'] for line in lines], 'added_lb': added, 'receipt_used_before_lb': used,
                      'claimed_lb': claim, 'claim_id': claim_id, 'claimed_total_lb': round(already + claim, 4),
                      'short_lb': short_lb, 'remaining_lb': left, 'closes': closes,
                      'balance_now_lb': float(api.lot_on_hand(cur, row['lot_id']))}
        return ticket['id'], {'resolution' if closes else 'last_claim': resolution}, closes
    if kind == 'voided':
        status = None
        if row['transaction_id'] is not None:
            cur.execute('SELECT effective_status FROM ledger_current_transactions WHERE id=%s', (row['transaction_id'],))
            found = cur.fetchone()
            status = found['effective_status'] if found else None
        if status != 'voided':
            fail(409, 'POSTING_STILL_EFFECTIVE',
                 f'The short posting (transaction {row["transaction_id"]}) is {status or "unknown"}, not voided; void it first.',
                 message_es='El registro con faltante no está anulado; anúlelo primero.',
                 transaction_id=row['transaction_id'], effective_status=status)
        ticket_id = _receipt_ticket(cur, body.receipt_number)['id'] if body.receipt_number else None
        return ticket_id, {'resolution': {'kind': 'voided', 'transaction_id': row['transaction_id'], 'effective_status': status}}, True
    return None, {}, True


def identified_evidence(cur, row):
    """UNIDENTIFIED_LOT closes as `identified` only once the lot carries its identity (A5)."""
    cur.execute('SELECT identity_status FROM lots WHERE id=%s', (row['lot_id'],))
    lot = cur.fetchone()
    status = lot['identity_status'] if lot else None
    if status != 'identified':
        fail(409, 'LOT_NOT_IDENTIFIED', 'Identify the lot first (supplier lot); it is still ' + (status or 'unknown') + '.',
             message_es='Identifique el lote primero.', identity_status=status)
    return {'resolution': {'kind': 'identified', 'identity_status': status}}


def _nest(item, id_key, name_key, out_key, name_out='name'):
    id_, name = item.pop(id_key, None), item.pop(name_key, None)
    item[out_key] = {'id': id_, name_out: name} if id_ is not None else None


def _view(cur, row):
    item = dict(row)
    item['overdue'] = bool(item['due_at'] and item['status'] in ('open', 'escalated')
                           and item['due_at'] < datetime.now(PLANT_TIMEZONE))
    _nest(item, 'product_id', 'product_name', 'product')
    _nest(item, 'lot_id', 'lot_code', 'lot', 'lot_code')
    _nest(item, 'owner_actor_id', 'owner_name', 'owner')
    _nest(item, 'resolved_by_actor_id', 'resolved_by_name', 'resolved_by')
    if item['kind'] in EVIDENCE_KINDS:
        cur.execute('SELECT id,short_lb,status,resolution_kind,resolved_at,due_at FROM shortage_flags WHERE exception_id=%s', (item['id'],))
        flag = cur.fetchone()
        item['shortage_flag'] = dict(flag) if flag else None
        cur.execute('''SELECT id, evidence_ticket_id, evidence_receipt_number, lot_id, claimed_lb, claimed_at, claimed_by_actor_id
                       FROM shortage_evidence_claims WHERE exception_id=%s ORDER BY id''', (item['id'],))
        claims = [dict(c) for c in cur.fetchall()]
        claimed = round(sum(float(c['claimed_lb']) for c in claims), 4)
        short = float(flag['short_lb']) if flag else None
        item['evidence'] = {'claimed_lb': claimed, 'remaining_lb': None if short is None else round(short - claimed, 4), 'claims': claims}
    return item


_SELECT = '''SELECT e.*, p.name AS product_name, l.lot_code, o.name AS owner_name, r.name AS resolved_by_name,
                    wt.status AS ticket_status, wt.receipt_number AS ticket_receipt_number
             FROM exceptions e
             LEFT JOIN products p ON p.id=e.product_id
             LEFT JOIN lots l ON l.id=e.lot_id
             LEFT JOIN actors o ON o.id=e.owner_actor_id
             LEFT JOIN actors r ON r.id=e.resolved_by_actor_id
             LEFT JOIN write_tickets wt ON wt.id=e.ticket_id'''


def _one(cur, exception_id, lock=False):
    if lock:
        cur.execute('SELECT id FROM exceptions WHERE id=%s FOR UPDATE', (exception_id,))
        if not cur.fetchone():
            fail(404, 'EXCEPTION_NOT_FOUND', 'No exception has this id.')
    cur.execute(_SELECT + ' WHERE e.id=%s', (exception_id,))
    row = cur.fetchone()
    if not row:
        fail(404, 'EXCEPTION_NOT_FOUND', 'No exception has this id.')
    return row


def register_routes(app, api):
    import write_tickets

    def who(request):
        return write_tickets.identity(api, request)

    @app.get('/exceptions')
    def list_exceptions(request: Request, kind: Optional[str] = None,
                        status: Literal['open', 'escalated', 'resolved', 'waived', 'all'] = 'open',
                        overdue: Optional[bool] = None, owner_actor_id: Optional[int] = None,
                        product_id: Optional[int] = None, lot_id: Optional[int] = None,
                        limit: int = Query(200, ge=1, le=1000), _: bool = Depends(api.verify_api_key)):
        permissions.require('list_exceptions', who(request))
        kinds = [k.strip().upper() for k in kind.split(',')] if kind else None
        # 'open' is the queue: open AND escalated (escalated is still open, just overdue).
        statuses = {'open': ['open', 'escalated'], 'escalated': ['escalated'], 'resolved': ['resolved'],
                    'waived': ['waived'], 'all': None}[status]
        with api.get_transaction() as cur:
            cur.execute(_SELECT + '''
                WHERE (%s::text[] IS NULL OR e.kind = ANY(%s)) AND (%s::text[] IS NULL OR e.status = ANY(%s))
                  AND (%s::bool IS NULL OR (%s AND e.status IN ('open','escalated') AND e.due_at < clock_timestamp())
                       OR (NOT %s AND NOT (e.status IN ('open','escalated') AND e.due_at < clock_timestamp())))
                  AND (%s::int IS NULL OR e.owner_actor_id=%s) AND (%s::int IS NULL OR e.product_id=%s)
                  AND (%s::int IS NULL OR e.lot_id=%s)
                ORDER BY (e.status IN ('open','escalated') AND e.due_at < clock_timestamp()) DESC,
                         e.due_at NULLS LAST, e.opened_at, e.id LIMIT %s''',
                (kinds, kinds, statuses, statuses, overdue, overdue, overdue, owner_actor_id, owner_actor_id,
                 product_id, product_id, lot_id, lot_id, limit))
            rows = cur.fetchall()
            return {'count': len(rows), 'status': status, 'exceptions': [_view(cur, r) for r in rows]}

    @app.get('/exceptions/{exception_id}')
    def get_exception(exception_id: int, request: Request, _: bool = Depends(api.verify_api_key)):
        permissions.require('list_exceptions', who(request))
        with api.get_transaction() as cur:
            return _view(cur, _one(cur, exception_id))

    @app.post('/exceptions/{exception_id}/resolve')
    def resolve_exception(exception_id: int, body: ResolveRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        actor = who(request)
        permissions.require('resolve_exception', actor)
        with api.get_transaction() as cur:
            row = _one(cur, exception_id, lock=True)
            if row['kind'] in OWNER_KINDS:
                permissions.require('approve_exception', actor)
            if row['kind'] == 'LARGE_CORRECTION':
                fail(409, 'USE_APPROVE_OR_REJECT', 'A held correction is approved or rejected, not resolved.',
                     approve=f'/exceptions/{exception_id}/approve', reject=f'/exceptions/{exception_id}/reject')
            if row['status'] not in ('open', 'escalated'):
                fail(409, 'EXCEPTION_CLOSED', f'Exception #{exception_id} is already {row["status"]}.',
                     status=row['status'], resolution_kind=row['resolution_kind'],
                     resolved_at=row['resolved_at'].isoformat() if row['resolved_at'] else None)
            allowed = RESOLUTION_KINDS.get(row['kind'], DEFAULT_RESOLUTIONS)
            if body.resolution_kind not in allowed:
                fail(422, 'RESOLUTION_KIND_INVALID', f'{row["kind"]} resolves as one of {", ".join(allowed)}.',
                     allowed=list(allowed))
            if body.resolution_kind in OWNER_RESOLUTIONS:
                permissions.require('approve_exception', actor)     # written_off / waived: owner only
            # Evidence: a shortage closes on a matching correction or receipt, never on a note alone.
            resolution_ticket_id, patch, closes = None, {}, True
            if row['kind'] in EVIDENCE_KINDS:
                resolution_ticket_id, patch, closes = shortage_evidence(api, cur, request, actor, row, body)
            elif row['kind'] == 'UNIDENTIFIED_LOT' and body.resolution_kind == 'identified':
                patch = identified_evidence(cur, row)
            if body.receipt_number and resolution_ticket_id is None:
                resolution_ticket_id = _receipt_ticket(cur, body.receipt_number)['id']
            if not closes:
                # Partial coverage: the claim is recorded, the shortage stays open for the remainder.
                cur.execute('UPDATE exceptions SET detail = detail || %s::jsonb WHERE id=%s',
                            (json.dumps(patch, default=str), exception_id))
                api._record_actor_write(cur, request, 'exceptions', exception_id)
                return _view(cur, _one(cur, exception_id))
            new_status = 'waived' if body.resolution_kind in WAIVE_KINDS else 'resolved'
            cur.execute('''UPDATE exceptions SET status=%s, resolved_at=clock_timestamp(), resolved_by_actor_id=%s,
                               resolution_kind=%s, resolution_note=%s, resolution_ticket_id=%s,
                               detail = detail || %s::jsonb WHERE id=%s''',
                        (new_status, actor['id'], body.resolution_kind, body.note, resolution_ticket_id,
                         json.dumps(patch, default=str), exception_id))
            if row['kind'] == 'SHORTAGE':
                cur.execute('''UPDATE shortage_flags SET status='resolved', resolution_kind=%s, resolved_at=clock_timestamp(),
                                   resolved_by_actor_id=%s, resolution_ticket_id=%s
                               WHERE exception_id=%s AND status IN ('open','escalated')''',
                            (body.resolution_kind, actor['id'], resolution_ticket_id, exception_id))
            api._record_actor_write(cur, request, 'exceptions', exception_id)
            return _view(cur, _one(cur, exception_id))

    def _held(cur, exception_id):
        """Lock the TICKET first, then the exception — the order the photo-release commit
        already uses (write_tickets FOR UPDATE, then UPDATE exceptions) — so approve,
        reject and a concurrent commit on the same hold queue up instead of deadlocking.
        `ticket_id` never changes on an exception row, so the unlocked peek is safe."""
        cur.execute('SELECT kind, ticket_id FROM exceptions WHERE id=%s', (exception_id,))
        head = cur.fetchone()
        if not head:
            fail(404, 'EXCEPTION_NOT_FOUND', 'No exception has this id.')
        if head['kind'] != 'LARGE_CORRECTION':
            fail(409, 'APPROVAL_NOT_APPLICABLE', f'{head["kind"]} is resolved, not approved.',
                 resolve=f'/exceptions/{exception_id}/resolve')
        ticket = None
        if head['ticket_id'] is not None:
            cur.execute('SELECT * FROM write_tickets WHERE id=%s FOR UPDATE', (head['ticket_id'],))
            ticket = cur.fetchone()
        if not ticket:
            fail(409, 'TICKET_NOT_FOUND', 'The held ticket is missing.')
        return _one(cur, exception_id, lock=True), ticket

    def _current_approver(cur, owner):
        """A2 at the decision: the approver's role and active flag as they are NOW, read
        FOR SHARE inside the transaction (the auth cache may still admit a key)."""
        if owner['id'] is not None:
            cur.execute('SELECT role, active FROM actors WHERE id=%s FOR SHARE', (owner['id'],))
            current = cur.fetchone()
            if not current or not current['active']:
                fail(403, 'ACTOR_INACTIVE', 'This actor is no longer active.')
            owner = {**owner, 'role': current['role']}
        permissions.require('approve_exception', owner)
        return owner

    def _stale(code, message):
        return api.JSONResponse(status_code=409, content={'detail': {
            'error_code': 'TICKET_STALE', 'message': 'The preparer can no longer post this; the hold stays open.',
            'blockers': [{'code': code, 'message': message}]}})

    def _current_preparer(cur, ticket):
        """The preparer as they are NOW (FOR SHARE): a named, active actor still allowed
        the ticket's action. Otherwise (preparer, row) is None and the 409 is returned
        — nothing posts and the hold stays the owner's to reject."""
        preparer_row = None
        if ticket['actor_id'] is not None:
            cur.execute('SELECT * FROM actors WHERE id=%s FOR SHARE', (ticket['actor_id'],))
            preparer_row = cur.fetchone()
        if not preparer_row:
            return None, _stale('PREPARER_UNKNOWN', 'The held ticket has no named preparer; reject it and prepare again with an actor key.')
        if not preparer_row['active']:
            return None, _stale('ACTOR_INACTIVE', 'The preparing actor is no longer active.')
        preparer = {'id': preparer_row['id'], 'name': preparer_row['name'], 'role': preparer_row['role'], 'key_kind': 'actor'}
        if not permissions.allowed(ticket['action'], permissions.role_of(preparer)):
            return None, _stale('ROLE_NOT_ALLOWED',
                                f'{preparer_row["name"]} ({preparer_row["role"]}) is no longer allowed to {permissions.label(ticket["action"])[0]}.')
        return (preparer, preparer_row), None

    @app.post('/exceptions/{exception_id}/approve')
    def approve_exception(exception_id: int, body: DecisionRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        owner = who(request)
        permissions.require('approve_exception', owner)
        with api.get_transaction() as cur:
            row, ticket = _held(cur, exception_id)
            owner = _current_approver(cur, owner)
            if ticket['status'] == 'committed' and ticket['response']:
                # Approved already, or a photo released the hold first: same receipt, nothing posts twice.
                return {**ticket['response'], 'replayed': True}
            if row['status'] not in ('open', 'escalated'):
                fail(409, 'EXCEPTION_CLOSED', f'Exception #{exception_id} is already {row["status"]} ({row["resolution_kind"]}).',
                     status=row['status'], resolution_kind=row['resolution_kind'])
            if ticket['status'] != HELD:
                fail(409, 'TICKET_NOT_HELD', 'The ticket is no longer awaiting approval.', ticket_status=ticket['status'])
            if ((row['detail'] or {}).get('payload_hash') != ticket['payload_hash']
                    or write_tickets.canonical_hash(ticket['payload']) != ticket['payload_hash']):
                fail(409, 'TICKET_PAYLOAD_MISMATCH', 'The held ticket no longer matches the approval request.')
            current, stale = _current_preparer(cur, ticket)
            if stale:
                return stale
            preparer, preparer_row = current
            approval = {'exception_id': exception_id, 'approved_by': {'id': owner['id'], 'name': owner['name'], 'role': owner['role']},
                        'note': body.note or None}
            result = write_tickets.execute_commit(
                api, cur, ticket, preparer, preparer_request(preparer_row),
                effective_payload=ticket['payload'], acknowledged=list(ticket['acknowledged'] or []),
                attachment_ref=None, approval=approval)
            if hasattr(result, 'status_code'):
                return result   # 409 TICKET_STALE: nothing posted, hold and exception untouched
            release_hold(cur, ticket['id'], resolution_kind='approved', actor_id=owner['id'], note=approval['note'],
                         response=result)
            api._record_actor_write(cur, request, 'exceptions', exception_id)
            return result

    @app.post('/exceptions/{exception_id}/reject')
    def reject_exception(exception_id: int, body: ResolveRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        owner = who(request)
        permissions.require('approve_exception', owner)
        if body.resolution_kind != 'declined':
            fail(422, 'RESOLUTION_KIND_INVALID', "Rejecting a held correction uses resolution_kind 'declined'.", allowed=['declined'])
        with api.get_transaction() as cur:
            row, ticket = _held(cur, exception_id)
            owner = _current_approver(cur, owner)
            if row['status'] == 'resolved' and row['resolution_kind'] == 'declined' and ticket['status'] == 'rejected':
                return {'rejected': True, 'replayed': True, 'exception_id': exception_id, 'ticket_id': ticket['id'],
                        'ticket_status': ticket['status']}
            if row['status'] not in ('open', 'escalated'):
                fail(409, 'ALREADY_APPROVED' if row['resolution_kind'] in ('approved', 'photo_attached') else 'EXCEPTION_CLOSED',
                     f'Exception #{exception_id} is already {row["status"]} ({row["resolution_kind"]}).',
                     status=row['status'], resolution_kind=row['resolution_kind'])
            if ticket['status'] != HELD:
                fail(409, 'TICKET_NOT_HELD', 'The ticket is no longer awaiting approval.', ticket_status=ticket['status'])
            note = body.note
            cur.execute("""UPDATE exceptions SET status='resolved', resolved_at=clock_timestamp(), resolved_by_actor_id=%s,
                               resolution_kind='declined', resolution_note=%s WHERE id=%s""", (owner['id'], note, exception_id))
            cur.execute("UPDATE write_tickets SET status='rejected', reject_reason=%s WHERE id=%s",
                        (json.dumps([{'code': 'DECLINED_BY_OWNER', 'message': note, 'actor': owner['name']}]), ticket['id']))
            api._record_actor_write(cur, request, 'exceptions', exception_id)
            return {'rejected': True, 'replayed': False, 'exception_id': exception_id, 'ticket_id': ticket['id'],
                    'ticket_status': 'rejected', 'posted': False, 'declined_by': owner['name'], 'note': note}
