"""Trial recall (migration 097; docs/deployments/trial-recall.md).

A scheduled drill, Monday / Wednesday / Friday: could we run a real recall
today from the records we have? Two tests, each on one randomly picked lot.

  Test 1, forward   a random receiving Form row (on/after the cutoff) that has a
                    BOL photo. The photo is read by the extraction model, then
                    Sheet vs BOL vs FL are compared, and the received lot is
                    traced forward: consumption -> batches -> finished lots ->
                    packing -> shipments -> customers.
  Test 2, backward  a random SHIPPED finished lot (made on/after the cutoff),
                    back to every ingredient lot, each ingredient lot's Form row
                    and BOL photo, then forward again from those ingredient
                    lots to every other customer they reached.

Rules this module keeps:

  * STRICTLY READ-ONLY on the ledger. It reads posted current-ledger data only
    (ledger_current_transactions + ledger_current_transaction_lines with
    effective_status = 'posted'), in transactions that are READ ONLY. Its only
    write is one INSERT per run into trial_recall_runs (migration 097).
  * The Google Sheet and the Drive photos are independent evidence; they are
    read, never changed (google_evidence.py can only read).
  * Never assume FL is right: where the three sources differ, all three values
    are shown and none is called the correct one.
  * A BOL photo that cannot be opened or read is UNVERIFIED, never PASS.
  * Product wording never fails anything: a product agrees with confidence on all three
    records or the line is UNVERIFIED.
  * "Full recall possible" is NO only when an affected customer or shipment cannot be
    identified (FLAGS level BLOCKS). Receiving mismatches fail the receiving section;
    a missing BOL and shared-login entries are warnings.
  * Nothing is assumed about which stock was used first (no FIFO): the traces
    follow recorded lot lines only (lot_trace.py, lot_trace_backward.py).
  * Before the cutoff nothing is tested: an older ingredient in a chain is
    shown as "before cutoff - not tested". Nothing eligible is "No eligible
    lot", not a failure.

Shape (as reconciliation.py): `load` reads the ledger into plain data,
`gather_evidence` reads Drive and the photo, `confirm_names` checks the BOL's
wording against the product and supplier lists, `analyse` is pure, and
`render_email` writes the owner email. Never imports main.
"""
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
import html as html_lib
import os
import random
import re
from zoneinfo import ZoneInfo

from fastapi import Depends, HTTPException, Request
from psycopg2.extras import Json

import extraction
import google_evidence
import lot_trace
from lot_trace_backward import BackwardTrace
import permissions
import receiving_form
import reconciliation
import resolution

PLANT_TIMEZONE = ZoneInfo('America/New_York')   # = main.PLANT_TIMEZONE
MARKER = receiving_form.MARKER
CUTOFF = date(2026, 11, 11)
RECALL_WEEKDAYS = frozenset({0, 2, 4})          # Monday, Wednesday, Friday (plant time)
LB_SLACK = 0.5                                  # pounds: a rounding allowance when two records are compared
MAX_PHOTOS_READ = 6                             # photos of one Form row read by the model; more is UNVERIFIED
MAX_EMAIL_ROWS = 40                             # longer lists are cut in the email and saved whole with the run
RUN_LOCK_NAME = 'factory_ledger.trial_recall'
PASS, FAIL, UNVERIFIED, NO_LOT = 'pass', 'fail', 'unverified', 'no_eligible_lot'
AGREE, DISAGREE, NOT_COMPARABLE = 'agree', 'disagree', 'cannot compare'
_WORST = {PASS: 0, UNVERIFIED: 1, FAIL: 2}
ROUTES = frozenset({
    ('GET', '/verification/trial-recall/runs'),
    ('GET', '/verification/trial-recall/runs/{run_id}'),
})
WEIGHT_UNITS = {'lb': 1.0, 'lbs': 1.0, 'pound': 1.0, 'pounds': 1.0, '#': 1.0,
                'kg': receiving_form.KG_TO_LB, 'kgs': receiving_form.KG_TO_LB, 'kilos': receiving_form.KG_TO_LB}
# Package units a BOL counts in, by kind. The Form's "number of cases" is a count of whatever package the
# product came in (staff type bags and pails there too), so one kind of package on the BOL is compared with
# it. Different kinds are never added together, and pieces / each are not packages (they may be inner units).
COUNT_UNITS = {**dict.fromkeys(('case', 'cases', 'cs', 'box', 'boxes', 'ctn', 'ctns', 'carton', 'cartons'), 'case'),
               **dict.fromkeys(('bag', 'bags', 'sack', 'sacks'), 'bag'), **dict.fromkeys(('pail', 'pails'), 'pail'),
               **dict.fromkeys(('drum', 'drums'), 'drum'), **dict.fromkeys(('tote', 'totes'), 'tote')}

# The flags, and what each one means for the answer (owner, 2026-11-18):
#   BLOCKS     an affected customer or shipment cannot be identified -> "Full recall possible: NO"
#   RECEIVING  a receiving-record gap: it fails the receiving check, and does not by itself make the answer NO
#   WARNING    listed only
BLOCKS, RECEIVING, WARNING = 'blocks', 'receiving', 'warning'
FLAGS = {
    'MADE_REMOVED_BY_ADJUSTMENT': (BLOCKS, 'Made product removed by adjustment, never packed or shipped'),
    'ORDER_SHIPPED_NOT_RECORDED': (BLOCKS, 'Order closed "shipped, not recorded"'),
    'UNKNOWN_ADJUSTMENT_INTO_PRODUCT': (BLOCKS, 'Stock added with reason "unknown" went into product'),
    'NEGATIVE_BALANCE': (BLOCKS, 'Lot below zero'),
    'LOT_ATTRIBUTION_UNRESOLVED': (BLOCKS, 'Lot attribution not resolved'),
    'TRACE_GAP': (BLOCKS, 'A consumed lot has no identified output lot'),
    'PRODUCTION_INPUTS_MISSING': (BLOCKS, 'Make or pack entry with no ingredient lots recorded'),
    'SHIPMENT_CUSTOMER_UNKNOWN': (BLOCKS, 'Shipment with no identifiable customer'),
    'MISSING_SUPPLIER_LOT': (RECEIVING, 'Ingredient lot with no supplier lot'),
    'RECEIVING_EVIDENCE_MISSING': (RECEIVING, 'No receiving Form row for an ingredient lot'),
    'SHIPMENT_NO_BOL': (WARNING, 'Shipment with no BOL'),
    'SHARED_KEY_POSTING': (WARNING, 'Entry made with the shared login (no named person)'),
}


def _levelled(flags):
    """Every flag carries its level; `blocks` is true only for the ones that make the answer NO."""
    for flag in flags:
        flag['level'] = FLAGS[flag['kind']][0]
        flag['blocks'] = flag['level'] == BLOCKS
    return flags


class TraceApi:
    """The few `main` helpers lot_trace.py and lot_trace_backward.py call, for the
    scheduled command (which never imports main). A test pins each one to main's."""

    def __init__(self):
        self._backward = BackwardTrace(self)

    @staticmethod
    def format_timestamp(dt):
        if dt is None:
            return None, None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        local = dt.astimezone(PLANT_TIMEZONE)
        return local.strftime('%Y-%m-%d'), local.strftime('%I:%M %p') + ' ET'

    @staticmethod
    def ledger_quantity_unit(uom):
        unit = (uom or 'lb').strip() or 'lb'
        return 'lb' if re.search(r'(?<![a-z])(?:lbs?|pounds?)(?![a-z])', unit, re.I) else unit

    def _trace_shipper_kind(self, name):
        return self._backward._trace_shipper_kind(name)

    def _trace_stamp(self, prefix, value):
        return self._backward._trace_stamp(prefix, value)

    def _trace_receipt_origin(self, row, lot):
        return self._backward._trace_receipt_origin(row, lot)


# ---------------------------------------------------------------------------
# Small helpers.
# ---------------------------------------------------------------------------

def _plain(value):
    """JSON-able copy: Decimal -> float, dates -> ISO text."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _dicts(cur):
    names = [column[0] for column in cur.description]
    return [dict(row) if isinstance(row, dict) else dict(zip(names, row)) for row in cur.fetchall()]


def _code(text):
    """A number or code for comparing: letters and digits only, no leading zeros."""
    return re.sub(r'[^A-Z0-9]', '', (text or '').upper()).lstrip('0')


def _words(text):
    return frozenset(re.findall(r'[a-z0-9]+', (text or '').lower()))


def _blank_supplier_lot(code):
    return (code or '').strip().upper() in reconciliation.BLANK_SUPPLIER_CODES


def lb(value):
    return f'{value:,.2f}'.rstrip('0').rstrip('.') + ' lb'


def worst(*results):
    present = [r for r in results if r in _WORST]
    return max(present, key=_WORST.get) if present else PASS


def is_recall_day(now):
    return now.astimezone(PLANT_TIMEZONE).weekday() in RECALL_WEEKDAYS


def migration_applied(cur):
    return receiving_form.migration_applied(cur)


# ---------------------------------------------------------------------------
# Reading the ledger. Everything below this banner only SELECTs.
# ---------------------------------------------------------------------------

_POSTED_LINES = """FROM ledger_current_transactions t
                   JOIN ledger_current_transaction_lines i ON i.transaction_id = t.id
                   JOIN transactions tx ON tx.id = t.id
                   WHERE t.effective_status = 'posted'"""


def _order_customers(cur, order_ids):
    if not order_ids:
        return {}
    cur.execute("""SELECT so.id, so.order_number, c.name AS customer_name
                   FROM sales_orders so JOIN customers c ON c.id = so.customer_id
                   WHERE so.id = ANY(%s)""", (sorted(order_ids),))
    return {row['id']: row for row in _dicts(cur)}


def summarize_forward(cur, graph):
    """lot_trace.forward's graph as compact facts: lots reached, each step, shipments, customers."""
    by_id = {node['lot_id']: node for node in graph['lots']}
    order_ids = {order['sales_order_id'] for node in graph['lots'] for shipment in node['shipments']
                 for order in (shipment.get('sales_orders') or [])}
    orders = _order_customers(cur, order_ids)
    shipments = []
    for node in graph['lots']:
        for shipment in node['shipments']:
            numbers = [order['order_number'] for order in (shipment.get('sales_orders') or [])]
            customer = (shipment.get('customer_name') or '').strip() or next(
                (orders[order['sales_order_id']]['customer_name'] for order in (shipment.get('sales_orders') or [])
                 if order['sales_order_id'] in orders), None)
            shipments.append({
                'transaction_id': shipment['transaction_id'], 'shipped_date': shipment.get('shipped_date'),
                'customer': customer, 'order_numbers': numbers or ([shipment['order_reference']]
                                                                   if shipment.get('order_reference') else []),
                'lot_id': node['lot_id'], 'lot_code': node['lot_code'], 'product_id': node['product_id'],
                'product_name': node['product_name'], 'quantity_lb': float(shipment['quantity_lb']),
                'shipment_ids': list(shipment.get('shipment_ids') or [])})
    steps = [{'transaction_id': edge['transaction_id'], 'type': edge['type'], 'date': edge.get('produced_date'),
              'source_lot_id': edge['source_lot_id'],
              'source_lot_code': by_id.get(edge['source_lot_id'], {}).get('lot_code'),
              'source_product': by_id.get(edge['source_lot_id'], {}).get('product_name'),
              'used': edge['quantity_used'], 'unit': edge['unit'], 'output_lot_id': edge['lot_id'],
              'output_lot_code': edge.get('batch_lot_code'), 'output_product': edge.get('batch_product'),
              'output_quantity': edge['output_quantity']} for edge in graph['consumption']]
    return _plain({
        'start_lot_id': graph['start_lot_id'],
        'lots': [{'lot_id': n['lot_id'], 'lot_code': n['lot_code'], 'product_id': n['product_id'],
                  'product_name': n['product_name'], 'product_type': n['product_type'], 'unit': n['unit'],
                  'on_hand': n['on_hand']} for n in graph['lots']],
        'steps': steps, 'shipments': shipments,
        'customers': sorted({s['customer'] or 'customer not named on the shipment' for s in shipments}),
        'gaps': graph['gaps'], 'quantity_basis': graph['quantity_basis']})


def chain_facts(cur, lot_ids, cutoff):
    """What the books say about the lots of one chain, and the flags they raise. Reads only."""
    lot_ids = sorted(set(lot_ids))
    if not lot_ids:
        return {'lots': [], 'flags': [], 'shipments': []}
    cur.execute("""
        SELECT l.id AS lot_id, l.lot_code, l.product_id, p.name AS product_name, p.type AS product_type, p.uom,
               l.supplier_lot_code, l.identity_status, l.entry_source, l.entry_source_notes,
               ARRAY(SELECT e.supplier_lot_code FROM lot_supplier_codes e
                     WHERE e.lot_id = l.id AND e.supplier_lot_code IS NOT NULL ORDER BY e.id) AS supplier_codes
        FROM lots l JOIN products p ON p.id = l.product_id WHERE l.id = ANY(%s) ORDER BY l.id""", (lot_ids,))
    lots = {row['lot_id']: dict(row, moves={}) for row in _dicts(cur)}
    cur.execute("""
        SELECT i.lot_id, t.type,
               SUM(CASE WHEN i.quantity_lb > 0 THEN i.quantity_lb ELSE 0 END) AS qty_in,
               SUM(CASE WHEN i.quantity_lb < 0 THEN -i.quantity_lb ELSE 0 END) AS qty_out
        """ + _POSTED_LINES + """ AND i.lot_id = ANY(%s)
        GROUP BY i.lot_id, t.type""", (lot_ids,))
    for row in _dicts(cur):
        lots[row['lot_id']]['moves'][row['type']] = (float(row['qty_in']), float(row['qty_out']))
    flags = []

    def flag(kind, message, **refs):
        flags.append({'kind': kind, 'blocks': FLAGS[kind][0] == BLOCKS, 'title': FLAGS[kind][1], 'message': message,
                      'refs': _plain(refs)})

    def moved(lot, kind, side):
        return lot['moves'].get(kind, (0.0, 0.0))[side]

    for lot in lots.values():
        lot['unit'] = 'lb' if lot['product_type'] in ('batch', 'finished') else TraceApi.ledger_quantity_unit(lot['uom'])
        lot['quantity_in'] = sum(pair[0] for pair in lot['moves'].values())
        lot['quantity_out'] = sum(pair[1] for pair in lot['moves'].values())
        lot['on_hand'] = round(lot['quantity_in'] - lot['quantity_out'], 4)
        lot['made'] = moved(lot, 'make', 0) + moved(lot, 'pack', 0)
        lot['used'] = moved(lot, 'make', 1) + moved(lot, 'pack', 1)
        lot['shipped'] = moved(lot, 'ship', 1)
        lot['received'] = moved(lot, 'receive', 0)
        lot['adjusted_in'], lot['adjusted_out'] = moved(lot, 'adjust', 0), moved(lot, 'adjust', 1)
        name = f'{lot["product_name"]} lot {lot["lot_code"]}'
        if lot['on_hand'] < -0.00005:
            flag('NEGATIVE_BALANCE', f'{name} is at {lb(lot["on_hand"])}: more was taken out than the books '
                                     f'ever put in.', lot_id=lot['lot_id'])
        if lot['made'] > 0 and lot['adjusted_out'] > 0 and lot['used'] == 0 and lot['shipped'] == 0:
            flag('MADE_REMOVED_BY_ADJUSTMENT',
                 f'{name}: {lb(lot["made"])} was made and {lb(lot["adjusted_out"])} was removed by adjustment. '
                 f'None of it was ever packed or shipped in the books.', lot_id=lot['lot_id'])
        codes = [c for c in [lot['supplier_lot_code'], *lot['supplier_codes']] if not _blank_supplier_lot(c)]
        lot['supplier_lots'] = sorted(set(codes))
        if lot['product_type'] not in ('batch', 'finished') and lot['received'] > 0 and not codes:
            said = (' Staff chose "No lot on label".' if lot['identity_status'] == 'unidentified' else '')
            flag('MISSING_SUPPLIER_LOT', f'{name} has no supplier lot recorded.{said} A supplier recall by lot '
                                         f'number could not find it.', lot_id=lot['lot_id'])
        if lot['identity_status'] == 'unidentified' and codes:
            flag('LOT_ATTRIBUTION_UNRESOLVED', f'{name} is still marked unidentified.', lot_id=lot['lot_id'])
        if 'POOLED' in (lot['entry_source_notes'] or '').upper():
            flag('LOT_ATTRIBUTION_UNRESOLVED', f'{name} is a pooled lot: it holds more than one supplier lot, so '
                                               f'it cannot be tied to one.', lot_id=lot['lot_id'])
    cur.execute("""
        SELECT t.id AS transaction_id, tx.receipt_number, tx.reason_code, tx.business_date, i.lot_id,
               SUM(i.quantity_lb) AS quantity
        """ + _POSTED_LINES + """ AND t.type = 'adjust' AND i.lot_id = ANY(%s)
        GROUP BY t.id, tx.receipt_number, tx.reason_code, tx.business_date, i.lot_id
        ORDER BY t.id""", (lot_ids,))
    for row in _dicts(cur):
        lot = lots[row['lot_id']]
        went_on = lot['used'] > 0 or lot['shipped'] > 0 or moved(lot, 'pack', 0) > 0
        if float(row['quantity']) > 0 and (row['reason_code'] or 'unknown').strip().lower() == 'unknown' and went_on:
            flag('UNKNOWN_ADJUSTMENT_INTO_PRODUCT',
                 f'{lb(float(row["quantity"]))} was added to {lot["product_name"]} lot {lot["lot_code"]} with '
                 f'reason "unknown" ({row["receipt_number"] or "entry #" + str(row["transaction_id"])}, '
                 f'{row["business_date"].isoformat()}), and that lot went into packed or shipped product.',
                 lot_id=row['lot_id'], transaction_id=row['transaction_id'])
    cur.execute("""
        SELECT DISTINCT t.id AS transaction_id, t.type, tx.receipt_number, tx.business_date
        """ + _POSTED_LINES + """ AND i.lot_id = ANY(%s) AND tx.entered_by_actor_id IS NULL
              AND tx.business_date >= %s
        ORDER BY t.id""", (lot_ids, cutoff))
    shared = _dicts(cur)
    if shared:
        flag('SHARED_KEY_POSTING',
             f'{len(shared)} entr{"y" if len(shared) == 1 else "ies"} in this chain {"was" if len(shared) == 1 else "were"} '
             'made with the shared login, so no person is named: '
             + ', '.join(f'{row["receipt_number"] or "#" + str(row["transaction_id"])} ({row["type"]}, '
                         f'{row["business_date"].isoformat()})' for row in shared[:12])
             + (' and more.' if len(shared) > 12 else '.'),
             transaction_ids=[row['transaction_id'] for row in shared])
    cur.execute("""SELECT id, kind, status, lot_id, receipt_number FROM exceptions
                   WHERE status IN ('open', 'escalated') AND lot_id = ANY(%s)
                     AND kind IN ('SHORTAGE', 'UNIDENTIFIED_LOT', 'NEGATIVE_BALANCE')
                   ORDER BY id""", (lot_ids,))
    for row in _dicts(cur):
        lot = lots[row['lot_id']]
        flag('LOT_ATTRIBUTION_UNRESOLVED',
             f'Open exception #{row["id"]} ({row["kind"].replace("_", " ").lower()}) on {lot["product_name"]} lot '
             f'{lot["lot_code"]}: what was really used from this lot is not settled.',
             lot_id=row['lot_id'], exception_id=row['id'])
    cur.execute("""
        SELECT t.id AS transaction_id, tx.receipt_number, tx.business_date, t.customer_name,
               t.bol_reference AS entry_bol, s.id AS shipment_id, s.bol_reference AS shipment_bol, s.proof_status,
               EXISTS (SELECT 1 FROM attachments a WHERE a.shipment_id = s.id
                       AND a.kind IN ('bol', 'signed_packing_slip')) AS has_photo
        FROM ledger_current_transactions t
        JOIN transactions tx ON tx.id = t.id
        LEFT JOIN shipments s ON s.transaction_id = t.id
        WHERE t.effective_status = 'posted' AND t.type = 'ship'
          AND EXISTS (SELECT 1 FROM ledger_current_transaction_lines i
                      WHERE i.transaction_id = t.id AND i.lot_id = ANY(%s) AND i.quantity_lb < 0)
        ORDER BY t.id""", (lot_ids,))
    shipments = _dicts(cur)
    for row in shipments:
        row['bol'] = (row['shipment_bol'] or row['entry_bol'] or '').strip() or None
        if row['bol'] is None:
            flag('SHIPMENT_NO_BOL',
                 f'Shipment {row["receipt_number"] or "entry #" + str(row["transaction_id"])} on '
                 f'{row["business_date"].isoformat()}'
                 + (f' to {row["customer_name"]}' if row['customer_name'] else '')
                 + ' has no BOL number' + ('' if row['has_photo'] else ' and no BOL photo') + '.',
                 transaction_id=row['transaction_id'], shipment_id=row['shipment_id'])
    finished = sorted({lot['product_id'] for lot in lots.values() if lot['product_type'] == 'finished'})
    if finished:
        cur.execute("""
            SELECT so.id, so.order_number, c.name AS customer_name, so.state_changed_at, so.state_changed_by,
                   ARRAY_AGG(DISTINCT p.name) AS products
            FROM sales_orders so
            JOIN customers c ON c.id = so.customer_id
            JOIN sales_order_lines sol ON sol.sales_order_id = so.id
            JOIN products p ON p.id = sol.product_id
            WHERE so.state = 'closed' AND so.state_reason = 'shipped_not_recorded'
              AND sol.product_id = ANY(%s)
              AND (so.state_changed_at AT TIME ZONE 'America/New_York')::date >= %s
            GROUP BY so.id, so.order_number, c.name, so.state_changed_at, so.state_changed_by
            ORDER BY so.id""", (finished, cutoff))
        for row in _dicts(cur):
            flag('ORDER_SHIPPED_NOT_RECORDED',
                 f'Order {row["order_number"]} ({row["customer_name"]}) was closed "shipped, not recorded" for '
                 f'{", ".join(row["products"])}. Product may have gone to that customer with no shipment in the '
                 f'books, so the lots it took are unknown.', sales_order_id=row['id'])
    return _plain({'lots': [{key: lot[key] for key in (
        'lot_id', 'lot_code', 'product_id', 'product_name', 'product_type', 'unit', 'received', 'made', 'used',
        'shipped', 'adjusted_in', 'adjusted_out', 'on_hand', 'supplier_lots', 'identity_status')}
        for lot in lots.values()], 'flags': _levelled(flags), 'shipments': shipments})


def _fl_receipt(line):
    """One FL receipt line as the facts the three-way comparison needs."""
    return {'transaction_id': line['transaction_id'], 'receipt_number': line['receipt_number'],
            'business_date': line['business_date'], 'bol_reference': line['bol_reference'],
            'supplier_id': line['supplier_id'], 'supplier_name': line['supplier_name'] or line['shipper_name'],
            'lot_id': line['lot_id'], 'lot_code': line['lot_code'], 'supplier_lot_code': line['supplier_lot_code'],
            'product_id': line['product_id'], 'product_name': line['product_name'],
            'quantity': line['quantity'],
            'unit': 'lb' if line['product_type'] in ('batch', 'finished') else TraceApi.ledger_quantity_unit(line['uom']),
            'named_person': line['entered_by_actor_id'] is not None}


def _row_facts(row, sheet):
    facts = receiving_form.describe_row(row, sheet['sheet_id'], sheet.get('gid'))
    facts['product_entries'] = row['products']
    return facts


def load(cur, sheet, *, api=None, cutoff=CUTOFF, rng=None):
    """Pick the two lots and read everything the books say about their chains.

    `sheet` is {'sheet_id', 'tab', 'gid', 'time_zone', 'values'} read by the
    caller before this (read-only) transaction. Returns plain data.
    """
    api = api or TraceApi()
    rng = rng or random.SystemRandom()
    rows = receiving_form.current_rows(receiving_form.parse_rows(sheet['values'], sheet.get('time_zone')), cutoff)
    receipts = receiving_form.load_receipts(cur, cutoff - timedelta(days=receiving_form.DATE_TOLERANCE_DAYS))
    lines = {(line['transaction_id'], line['lot_id']): line for line in receipts}
    index = receiving_form.receipt_index(receipts)
    matcher = receiving_form.ProductMatcher(cur)
    matches = {row['row_number']: receiving_form.match_row(row, index, matcher) for row in rows}
    rows_by_lot = {}
    for row in rows:
        for link in matches[row['row_number']]['links']:
            rows_by_lot.setdefault((link['transaction_id'], link['lot_id']), []).append(row)
    data = {'cutoff': cutoff, 'sheet': {key: sheet.get(key) for key in ('sheet_id', 'tab', 'gid')},
            'form_rows_on_or_after_cutoff': len(rows),
            'unmatched_rows': [dict(_row_facts(row, sheet), status=matches[row['row_number']]['status'],
                                    reasons=matches[row['row_number']]['reasons'])
                               for row in rows if matches[row['row_number']]['status'] != 'matched']}

    # ---- Test 1, forward: a random Form row with a BOL photo -----------------
    def eligible(row):
        links = matches[row['row_number']]['links']
        # A row whose only FL receipts are dated before the cutoff is not tested.
        return bool(row['photo_file_ids']) and not (
            links and all(date.fromisoformat(link['business_date']) < cutoff for link in links))

    pool = [row for row in rows if eligible(row)]
    if not pool:
        data['forward'] = {'eligible': False, 'pool_size': 0,
                           'reason': f'No receiving Form row on or after {cutoff.isoformat()} has a BOL photo.'}
    else:
        row = rng.choice(pool)
        outcome = matches[row['row_number']]
        links = [link for link in outcome['links'] if date.fromisoformat(link['business_date']) >= cutoff]
        receipts_of_row = [dict(_fl_receipt(lines[(link['transaction_id'], link['lot_id'])]),
                                form_product=link['form_product'], basis=link['basis']) for link in links]
        if receipts_of_row:
            # How many deliveries ever went into each of these internal lots (FL keeps supplier lot codes
            # per lot, not per delivery, so with more than one the codes cannot be pinned to this BOL).
            cur.execute("""SELECT i.lot_id, COUNT(DISTINCT t.id) AS deliveries
                           """ + _POSTED_LINES + """ AND t.type = 'receive' AND i.quantity_lb > 0
                             AND i.lot_id = ANY(%s) GROUP BY i.lot_id""",
                        (sorted({receipt['lot_id'] for receipt in receipts_of_row}),))
            deliveries = {row['lot_id']: row['deliveries'] for row in _dicts(cur)}
            for receipt in receipts_of_row:
                receipt['receipts_into_lot'] = deliveries.get(receipt['lot_id'], 1)
        traces, chain = [], set()
        for link in links:
            graph = lot_trace.forward(cur, link['lot_id'], api)
            chain.update(node['lot_id'] for node in graph['lots'])
            traces.append(summarize_forward(cur, graph))
        data['forward'] = {'eligible': True, 'pool_size': len(pool), 'row': _row_facts(row, sheet),
                           'match_status': outcome['status'], 'match_reasons': outcome['reasons'],
                           'receipts': receipts_of_row, 'traces': traces,
                           'chain': chain_facts(cur, chain, cutoff)}

    # ---- Test 2, backward: a random shipped finished lot ---------------------
    cur.execute("""
        WITH made AS (
            SELECT i.lot_id, MIN(tx.business_date) AS first_made
            """ + _POSTED_LINES + """ AND t.type IN ('make', 'pack') AND i.quantity_lb > 0
            GROUP BY i.lot_id
        ), shipped AS (
            SELECT DISTINCT i.lot_id
            """ + _POSTED_LINES + """ AND t.type = 'ship' AND i.quantity_lb < 0
        )
        SELECT l.id, l.id AS lot_id, l.lot_code, l.product_id, l.entry_source, l.supplier_lot_code,
               p.name AS product_name, m.first_made
        FROM made m JOIN shipped s ON s.lot_id = m.lot_id
        JOIN lots l ON l.id = m.lot_id JOIN products p ON p.id = l.product_id
        WHERE p.type = 'finished' AND m.first_made >= %s
        ORDER BY l.id""", (cutoff,))
    finished = _dicts(cur)
    if not finished:
        data['backward'] = {'eligible': False, 'pool_size': 0,
                            'reason': f'No shipped finished lot was made on or after {cutoff.isoformat()}.'}
    else:
        lot = rng.choice(finished)
        origin = BackwardTrace(api)._trace_lot_origin(cur, lot, 0, frozenset())
        made_lots, ingredients, unexplained = {}, {}, []

        def walk(node_origin, made_lot):
            # A production step with no ingredient lots recorded (or a trail cut short) leaves the
            # ingredients of that lot unknown: remembered per lot, so it can never look complete.
            for step in node_origin.get('transactions', []):
                if not step.get('inputs') or node_origin.get('truncated'):
                    unexplained.append({'lot_id': made_lot['lot_id'], 'lot_code': made_lot['lot_code'],
                                        'product_name': made_lot['product_name'],
                                        'transaction_id': step['transaction_id'], 'type': step.get('type'),
                                        'date': step.get('produced_date'),
                                        'reason': 'cycle' if node_origin.get('truncated') else 'no_inputs'})
                for item in step.get('inputs', []):
                    source = item.get('origin') or {}
                    if source.get('kind') == 'made':
                        # A made lot can also hold stock that was received or found straight into it:
                        # kept, so the verdict never looks complete while part of a batch has no known inputs.
                        made_lots.setdefault(item['lot_id'], dict(
                            {key: item.get(key) for key in ('lot_id', 'lot_code', 'product_id', 'product_name')},
                            also_received=source.get('also_received') or []))
                        walk(source, item)
                    else:
                        entry = ingredients.setdefault(item['lot_id'], dict(
                            {key: item.get(key) for key in ('lot_id', 'lot_code', 'product_id', 'product_name',
                                                            'product_type', 'supplier_lot_code')},
                            origin_kind=source.get('kind'), used=0.0, unit=item.get('unit'),
                            # EVERY way stock came into the lot (supplier receipt, found, unnamed), each
                            # checked on its own: one documented receipt never covers another.
                            receipts=[{'transaction_id': r['transaction_id'], 'kind': r.get('kind'),
                                       'receipt_number': r.get('receipt_number'),
                                       'supplier_name': r.get('supplier_name') or r.get('recorded_shipper'),
                                       'quantity_lb': r.get('quantity_lb')}
                                      for r in (source.get('receipts') or []) if r.get('transaction_id') is not None]))
                        entry['used'] += float(item.get('quantity') or 0)

        walk(origin, lot)
        dated, receipt_dates = {}, {}
        if ingredients or made_lots:
            cur.execute("""SELECT i.lot_id, MIN(tx.business_date) AS first_in, MAX(tx.business_date) AS last_in
                           """ + _POSTED_LINES + """ AND i.quantity_lb > 0 AND i.lot_id = ANY(%s)
                           GROUP BY i.lot_id""", (sorted({*ingredients, *made_lots}),))
            dated = {row['lot_id']: row for row in _dicts(cur)}
        receipt_ids = sorted({r['transaction_id'] for item in ingredients.values() for r in item['receipts']})
        if receipt_ids:
            cur.execute("""SELECT t.id, tx.business_date FROM ledger_current_transactions t
                           JOIN transactions tx ON tx.id = t.id
                           WHERE t.effective_status = 'posted' AND t.id = ANY(%s)""", (receipt_ids,))
            receipt_dates = {row['id']: row['business_date'] for row in _dicts(cur)}

        def older(found):
            # "Before the cutoff" needs a KNOWN date, and every entry that put stock in before it.
            # A lot with no dated entry at all is not old: it is unresolved, and it is tested.
            return found is not None and found['last_in'] is not None and found['last_in'] < cutoff

        for made in made_lots.values():
            # A batch made only before the cutoff is, like an older ingredient, shown and not tested.
            made['first_in'] = (dated.get(made['lot_id']) or {}).get('first_in')
            made['before_cutoff'] = older(dated.get(made['lot_id']))
        own = lot_trace.forward(cur, lot['lot_id'], api)
        own_summary = summarize_forward(cur, own)
        # By shipment AND lot: one truck can carry the picked lot and another lot made from the same ingredient.
        own_shipments = {(s['transaction_id'], s['lot_id']) for s in own_summary['shipments']}
        tested, others, reached = [], {}, set()
        for item in ingredients.values():
            item['first_in'] = (dated.get(item['lot_id']) or {}).get('first_in')
            item['before_cutoff'] = older(dated.get(item['lot_id']))
            for receipt in item['receipts']:
                receipt['business_date'] = receipt_dates.get(receipt['transaction_id'])
                receipt['before_cutoff'] = (receipt['business_date'] is not None
                                            and receipt['business_date'] < cutoff)
                receipt['form_rows'] = [] if receipt['before_cutoff'] or receipt['kind'] != 'received' else [
                    _row_facts(row, sheet) for row in rows_by_lot.get((receipt['transaction_id'], item['lot_id']), [])]
            if item['before_cutoff']:
                continue
            tested.append(item['lot_id'])
            graph = lot_trace.forward(cur, item['lot_id'], api)
            summary = summarize_forward(cur, graph)
            item['forward'] = {'lots_reached': len(summary['lots']), 'customers': summary['customers'],
                               'gaps': summary['gaps']}
            # A recall of this ingredient lot covers everything made from it, not only the picked
            # finished lot: those lots are checked for flags too.
            reached.update(node['lot_id'] for node in summary['lots'])
            for shipment in summary['shipments']:
                if (shipment['transaction_id'], shipment['lot_id']) not in own_shipments:
                    others.setdefault((shipment['transaction_id'], shipment['lot_id']),
                                      dict(shipment, via_ingredient_lots=[]))['via_ingredient_lots'].append(
                                          item['lot_code'])
        chain = {lot['lot_id'], *(m['lot_id'] for m in made_lots.values() if not m['before_cutoff']), *tested,
                 *reached}
        data['backward'] = {
            'eligible': True, 'pool_size': len(finished),
            'lot': {key: lot[key] for key in ('lot_id', 'lot_code', 'product_id', 'product_name', 'first_made')},
            'origin_kind': origin.get('kind'), 'inputs_mismatch': bool(origin.get('inputs_mismatch')),
            'also_received': origin.get('also_received') or [],
            # Only steps of lots that are tested (the picked lot, and batches not made before the cutoff).
            'unexplained_steps': [step for step in unexplained if step['lot_id'] == lot['lot_id']
                                  or not made_lots.get(step['lot_id'], {}).get('before_cutoff', True)],
            'made_lots': list(made_lots.values()), 'ingredients': list(ingredients.values()),
            'own': own_summary, 'other_shipments': list(others.values()),
            'chain': chain_facts(cur, chain, cutoff)}
    return _plain(data)


# ---------------------------------------------------------------------------
# Reading the evidence: Drive and the photo. No database here.
# ---------------------------------------------------------------------------

def _photo_state(client, file_id, probe=False):
    """Can this Drive file be opened as a photo? {'file_id', 'url', 'accessible', 'mime_type', 'problem'}.

    probe=True also starts reading the file itself and stops after the first piece: Drive can
    show a file's details and still refuse its contents, and that is not "on file and openable".
    """
    state = {'file_id': file_id, 'url': google_evidence.drive_file_url(file_id), 'accessible': False,
             'mime_type': None, 'problem': None}
    try:
        meta = client.file_metadata(file_id)
    except google_evidence.GoogleError as exc:
        state['problem'] = str(exc)
        return state
    state['mime_type'] = meta['mime_type']
    if meta['trashed']:
        state['problem'] = 'The photo is in the Drive trash.'
    elif meta['mime_type'] not in receiving_form.PHOTO_TYPES:
        state['problem'] = f'The linked Drive file is not a photo or PDF ({meta["mime_type"]}).'
    elif meta['size'] is not None and meta['size'] > google_evidence.MAX_PHOTO_BYTES:
        state['problem'] = 'The photo is larger than 15 MB.'
    else:
        state['accessible'] = True
    if state['accessible'] and probe:
        try:
            chunks = client.open_media(file_id)
            try:
                if not next(iter(chunks), None):
                    state.update(accessible=False, problem='The photo file is empty.')
            finally:
                close = getattr(chunks, 'close', None)
                if close:
                    close()                     # nothing is kept; the rest is never downloaded
        except google_evidence.GoogleError as exc:
            state.update(accessible=False, problem=str(exc))
    return state


def gather_evidence(data, client, extractor=None):
    """Open the picked Form row's BOL photo(s) and read them with the extraction
    model; check that each backward ingredient lot's photo can be opened.
    Adds data['forward']['bol'] and each ingredient's 'photos'. Never raises
    for a missing or unreadable photo: that is a finding (UNVERIFIED)."""
    extractor = extractor or extraction.extract_bol
    forward = data['forward']
    if forward.get('eligible'):
        documents = []
        for file_id in forward['row']['photo_file_ids'][:MAX_PHOTOS_READ]:
            document = _photo_state(client, file_id)
            document.update(readable=False, extraction=None, model=None)
            readable_type = document['mime_type']
            if document['accessible'] and readable_type in receiving_form.HEIC_TYPES:
                readable_type = 'image/jpeg'                # a phone HEIC photo is converted in memory first
                document['converted_from'] = document['mime_type']
            if document['accessible'] and readable_type not in extraction.ALLOWED_MIME_TYPES:
                document['problem'] = (f'The photo format ({document["mime_type"]}) cannot be read by the '
                                       f'extraction model.')
            elif document['accessible']:
                try:
                    content = client.download(file_id)
                    if document.get('converted_from'):
                        content = extraction.heic_to_jpeg(content)
                    result = extractor(content, readable_type)
                    document.update(extraction=result['extraction'], model=result.get('extraction_model'))
                    if result['extraction'].get('readable') and result['extraction'].get('lines'):
                        document['readable'] = True
                    else:
                        document['problem'] = 'The extraction model could not read product lines on this photo.'
                except google_evidence.GoogleError as exc:
                    document.update(accessible=False, problem=str(exc))
                except extraction.ExtractionError as exc:
                    document['problem'] = f'The extraction model could not read this photo ({str(exc)[:160]}).'
            documents.append(document)
        forward['bol'] = {'documents': documents,
                          'photos_not_read': max(0, len(forward['row']['photo_file_ids']) - MAX_PHOTOS_READ)}
    backward = data['backward']
    if backward.get('eligible'):
        for item in backward['ingredients']:
            if item.get('before_cutoff'):
                continue
            for receipt in item['receipts']:
                receipt['photos'] = [_photo_state(client, file_id, probe=True)
                                     for row in receipt['form_rows'] for file_id in row['photo_file_ids']]
    return data


def confirm_names(cur, data):
    """Check the BOL's wording against FL's product and supplier lists (the shared
    resolver, read-only), so `analyse` can stay pure. Adds 'fits' to each BOL line
    and 'bol_supplier' to the forward test."""
    forward = data['forward']
    if not forward.get('eligible') or not forward.get('bol'):
        return data
    matcher = receiving_form.ProductMatcher(cur)
    for document in forward['bol']['documents']:
        for line in (document.get('extraction') or {}).get('lines', []):
            found = matcher.lookup((line.get('description') or '').strip()) if line.get('description') else None
            line['fits'] = [{'product_id': receipt['product_id'],
                             'basis': matcher.basis(line['description'], receipt['product_id'], receipt['product_name'])}
                            for receipt in forward['receipts'] if line.get('description')]
            line['resolves_to'] = found['match_id'] if found else None
            line['received_elsewhere'] = []
            if line['resolves_to'] is not None and not any(f['basis'] for f in line['fits']) \
                    and forward['row'].get('submitted_date'):
                # A product on the BOL that this Form row's receipts do not cover: is it in FL at all,
                # on another receipt of about the same date?
                on = date.fromisoformat(forward['row']['submitted_date'])
                span = timedelta(days=receiving_form.DATE_TOLERANCE_DAYS)
                cur.execute("""SELECT DISTINCT t.id, tx.receipt_number
                               """ + _POSTED_LINES + """ AND t.type = 'receive' AND i.quantity_lb > 0
                                 AND i.product_id = %s AND tx.business_date BETWEEN %s AND %s
                               ORDER BY t.id""", (line['resolves_to'], on - span, on + span))
                line['received_elsewhere'] = [row['receipt_number'] or f'entry #{row["id"]}' for row in _dicts(cur)]
    names = [d['extraction'].get('supplier_name') for d in forward['bol']['documents'] if d.get('extraction')]
    name = next((n for n in names if n), None)
    forward['bol_supplier'] = {'name': name, 'supplier_id': None}
    if name:
        cur.execute('SAVEPOINT trial_recall_supplier')
        try:
            result = resolution.resolve(cur, resolution.ResolveRequest(kind='supplier', query=name[:500]))
            forward['bol_supplier']['supplier_id'] = (result.get('match') or {}).get('id')
        except Exception:  # an unreadable name is simply not confirmed
            cur.execute('ROLLBACK TO SAVEPOINT trial_recall_supplier')
        cur.execute('RELEASE SAVEPOINT trial_recall_supplier')
    return data


# ---------------------------------------------------------------------------
# The comparison and the verdicts. Pure: no database, no network.
# ---------------------------------------------------------------------------

def _bol_quantity(line):
    """A BOL line's quantity as ('lb', pounds), ('count:<package kind>', number) or (None, None)."""
    quantity, unit = line.get('quantity'), (line.get('unit') or '').strip().lower().rstrip('.')
    if quantity is None:
        return None, None
    if unit in WEIGHT_UNITS:
        return 'lb', quantity * WEIGHT_UNITS[unit]
    if unit in COUNT_UNITS:
        return 'count:' + COUNT_UNITS[unit], quantity
    return None, None


def _status(pairs):
    """Pairs of (comparable?, equal?): any difference -> disagree; else agree if anything was compared."""
    compared = [equal for comparable, equal in pairs if comparable]
    if not compared:
        return NOT_COMPARABLE
    return AGREE if all(compared) else DISAGREE


def _reference_pieces(text):
    """The whole numbers a reference can be read as: the reference itself, and the reference with
    a leading word label taken off ("PO 4412", "PO-4412", "PO4412" all also read as 4412). Parts
    of a number are never numbers of their own: BOL-2026-001 is not 2026. Only readings that hold
    a digit count, so two "PO ..." never agree on the label alone."""
    text = (text or '').strip()
    pieces = {_code(text)}
    label = re.match(r'[A-Za-z]+\s*[#:.\-]?\s*(.+)$', text)
    if label:
        pieces.add(_code(label.group(1)))
    return {piece for piece in pieces if re.search(r'\d', piece)}


def _same_reference(one, other):
    """Do two references name the same number? Whole numbers only, on both sides:
    "4412" = "PO 4412"; "1234" is never found inside "912345"."""
    return bool(_reference_pieces(one) & _reference_pieces(other))


def _numbers_agree(reference, printed):
    """Is the number written on the Form / in FL among the numbers printed on the BOL?"""
    return any(_same_reference(reference, text) for text in printed)


def compare_receipt(row, receipt, bol_lines, bol_readable):
    """Sheet vs BOL vs FL for one received product. Every value is shown; none is called correct."""
    entry = next((e for e in row['product_entries'] if e['text'] == receipt['form_product']), None) or {}
    fitting = [line for line in bol_lines
               if any(f['product_id'] == receipt['product_id'] and f['basis'] for f in line.get('fits', []))]
    # One BOL line and one Form product, and the BOL wording names no FL product at
    # all: the line is shown beside the others for its quantity, never called a match.
    candidate_line = (bol_lines[0] if not fitting and len(bol_lines) == 1 and len(row['product_entries']) == 1
                      and bol_lines[0].get('resolves_to') is None else None)
    out = []
    # Product. Wording alone never fails anything (owner, 2026-11-18): the product either agrees with
    # confidence on all three records, or the line is "cannot compare" (UNVERIFIED).
    form_unconfirmed = (receipt.get('basis') or {}).get('product') == receiving_form.PRODUCT_NOT_CONFIRMED
    shown = fitting or bol_lines
    bol_product = ('BOL not readable' if not bol_readable
                   else '; '.join(line['description'] for line in shown) or 'no product line read')
    product_status = AGREE if bol_readable and fitting and not form_unconfirmed else NOT_COMPARABLE
    if product_status == AGREE or not bol_readable:
        product_note = None
    elif form_unconfirmed:
        product_note = 'The product wording on the Form does not confirm the FL product; linked by lot and date.'
    else:
        product_note = 'The wording on the BOL could not be tied to this product with confidence.'
    out.append({'field': 'Product', 'sheet': entry.get('name') or receipt['form_product'], 'bol': bol_product,
                'fl': receipt['product_name'], 'status': product_status, 'form_unconfirmed': form_unconfirmed,
                'note': product_note})
    # Quantity
    lines = fitting or ([candidate_line] if candidate_line else [])
    sheet_lb, sheet_cases = entry.get('total_lb'), entry.get('cases')
    fl_lb = receipt['quantity'] if receipt['unit'] == 'lb' else None
    bol_lb = bol_count = None
    kinds = {_bol_quantity(line)[0] for line in lines}
    # The BOL's quantity for this product is used only when EVERY one of its lines can be added up in
    # one unit. A line with no quantity, or in pallets, or pounds on one line and cases on another,
    # leaves part of the delivery uncounted: then nothing is compared and the quantity is not confirmed.
    bol_complete = len(kinds) == 1 and None not in kinds
    for line in lines if bol_complete else []:
        kind, amount = _bol_quantity(line)
        if kind == 'lb':
            bol_lb = (bol_lb or 0) + amount
        else:                                   # one kind of package only (bol_complete guarantees it)
            bol_count = (bol_count or 0) + amount
    pairs = [(sheet_lb is not None and fl_lb is not None,
              sheet_lb is not None and fl_lb is not None and abs(sheet_lb - fl_lb) <= LB_SLACK),
             (bol_lb is not None and fl_lb is not None,
              bol_lb is not None and fl_lb is not None and abs(bol_lb - fl_lb) <= LB_SLACK),
             (bol_lb is not None and sheet_lb is not None,
              bol_lb is not None and sheet_lb is not None and abs(bol_lb - sheet_lb) <= LB_SLACK),
             (bol_count is not None and sheet_cases is not None,
              bol_count is not None and sheet_cases is not None and abs(bol_count - sheet_cases) < 0.0001)]
    if entry.get('cases') is not None:
        size = f'{entry["case_size"]:g} {entry["case_unit"] or "(no unit written)"}'
        sheet_quantity = f'{entry["cases"]:g} x {size}' + (f' = {lb(sheet_lb)}' if sheet_lb is not None else '')
    else:
        sheet_quantity = 'not written as size and cases'
    bol_quantity = ('; '.join(f'{line["quantity"]:g} {line.get("unit") or "(no unit printed)"}' for line in lines
                              if line.get('quantity') is not None) or ('BOL not readable' if not bol_readable
                                                                       else 'no quantity read for this product'))
    out.append({'field': 'Quantity', 'sheet': sheet_quantity, 'bol': bol_quantity,
                'fl': f'{receipt["quantity"]:,.2f} {receipt["unit"]}'.replace('.00 ', ' '),
                'status': _status(pairs), 'bol_compared': bool(pairs[1][0] or pairs[2][0] or pairs[3][0]),
                # The BOL agreeing with the Form says nothing about FL: FL's own quantity must be compared too.
                'fl_compared': bool(pairs[0][0] or pairs[1][0]),
                'note': None})
    # Internal lot (the Form and FL carry it; a supplier's BOL does not)
    short_lot = (receipt.get('basis') or {}).get('lot') == receiving_form.LOT_WITHOUT_SEQUENCE
    out.append({'field': 'Internal lot', 'sheet': row['lot_text'], 'bol': 'not printed on a supplier BOL',
                'fl': receipt['lot_code'], 'status': AGREE,
                'note': 'The Form gives the lot without its sequence number; it is the only FL receipt that fits.'
                if short_lot else None})
    # Supplier lot numbers
    printed = sorted({lot for line in (fitting or bol_lines) for lot in line.get('lot_numbers', []) if lot.strip()})
    recorded = [] if _blank_supplier_lot(receipt['supplier_lot_code']) else [receipt['supplier_lot_code']]
    recorded = sorted(set(recorded) | set(receipt.get('supplier_lots') or []))
    if not bol_readable or not printed:
        lot_status = NOT_COMPARABLE
    elif not recorded:
        lot_status = DISAGREE
    else:
        on_bol, in_fl = {_code(p) for p in printed}, {_code(code) for code in recorded}
        single = (receipt.get('receipts_into_lot') or 1) <= 1
        if fitting and on_bol - in_fl:
            # A lot printed on this product's own BOL line and missing from FL: a supplier recall could not find it.
            lot_status = DISAGREE
        elif single and in_fl - on_bol:
            # FL holds a supplier lot this delivery's BOL does not show, and this is the lot's only delivery.
            lot_status = DISAGREE
        elif fitting or single:
            lot_status = AGREE
        else:
            # Several deliveries went into this internal lot and the BOL line could not be tied to the
            # product: FL's codes may belong to another delivery, so nothing can be said from this BOL.
            lot_status = NOT_COMPARABLE
    out.append({'field': 'Supplier lot', 'sheet': 'not asked on the Form',
                'bol': ', '.join(printed) or ('BOL not readable' if not bol_readable else 'none printed'),
                'fl': ', '.join(recorded) or 'none recorded', 'status': lot_status, 'note': None})
    return out


def compare_document(row, receipts, documents, bol_supplier, extra_lines=None):
    """The receipt-level facts: BOL number, supplier, manufacturer, carrier."""
    read = [d['extraction'] for d in documents if d.get('readable')]
    printed = [n for e in read for n in [e.get('bol_number'), *e.get('reference_numbers', [])] if n]
    fl_numbers = sorted({r['bol_reference'] for r in receipts if (r['bol_reference'] or '').strip()})
    sheet_number = (row['bol_number'] or '').strip()
    pairs = [(bool(sheet_number and fl_numbers), all(_same_reference(n, sheet_number) for n in fl_numbers))]
    if printed:
        pairs.append((bool(sheet_number), _numbers_agree(sheet_number, printed)))
        pairs.extend((True, _numbers_agree(n, printed)) for n in fl_numbers)
    out = [{'field': 'BOL / reference number', 'sheet': sheet_number or 'blank',
            'bol': ', '.join(printed) or ('BOL not readable' if not read else 'none printed'),
            'fl': ', '.join(fl_numbers) or 'none recorded', 'status': _status(pairs), 'note': None}]
    fl_suppliers = sorted({r['supplier_name'] for r in receipts if r['supplier_name']})
    name = (bol_supplier or {}).get('name')

    def one(receipt):
        """This receipt's supplier against the BOL's: (comparable?, same?)."""
        theirs = receipt['supplier_name']
        if not name or not theirs:
            return (False, False)
        if (bol_supplier.get('supplier_id') is not None and bol_supplier['supplier_id'] == receipt['supplier_id']) \
                or _words(name) <= _words(theirs) or _words(theirs) <= _words(name):
            return (True, True)
        # Different names: a disagreement only when FL knows who the BOL's supplier is; otherwise not judged.
        return (bol_supplier.get('supplier_id') is not None, False)

    # Every matched receipt is checked on its own: one receipt agreeing never hides another that does not.
    checked = [one(receipt) for receipt in receipts]
    supplier_status = _status(checked)
    if supplier_status == AGREE and not all(comparable for comparable, _ in checked):
        supplier_status = NOT_COMPARABLE
    out.append({'field': 'Supplier', 'sheet': 'not asked on the Form',
                'bol': name or ('BOL not readable' if not read else 'not printed'),
                'fl': ', '.join(fl_suppliers) or 'none recorded', 'status': supplier_status,
                'note': None if supplier_status != NOT_COMPARABLE or not name or not fl_suppliers
                else 'The two names could not be tied to the same supplier with certainty.'})
    if extra_lines is not None and read:
        # A product the BOL shows that this Form row's matched receipts do not cover.
        elsewhere = [line for line in extra_lines if line.get('received_elsewhere')]
        missing = [line for line in extra_lines if line.get('resolves_to') is not None
                   and not line.get('received_elsewhere')]
        unnamed = [line for line in extra_lines if line.get('resolves_to') is None]
        # Recognising the product rests on the BOL's wording, so this is never a FAIL: unconfirmed at most.
        status = NOT_COMPARABLE if missing or unnamed else AGREE
        fl_words = '; '.join(
            [f'{line["description"]}: on FL receipt {", ".join(line["received_elsewhere"])}' for line in elsewhere]
            + [f'{line["description"]}: no FL receipt for this product within a day of the Form date' for line in missing]
            + [f'{line["description"]}: not recognised as an FL product' for line in unnamed]) or 'none'
        out.append({'field': 'Other products on the BOL', 'sheet': 'not on this Form row' if extra_lines else 'none',
                    'bol': '; '.join(line['description'] for line in extra_lines) or 'none',
                    'fl': fl_words, 'status': status,
                    'note': None if status != NOT_COMPARABLE else
                    'The BOL has a line that is not accounted for by the receipts matched to this Form row.'})
    makers = sorted({e['manufacturer_name'] for e in read if e.get('manufacturer_name')})
    out.append({'field': 'Manufacturer', 'sheet': 'not asked on the Form',
                'bol': ', '.join(makers) or ('BOL not readable' if not read else 'not printed'),
                'fl': 'FL does not record a manufacturer', 'status': NOT_COMPARABLE, 'note': None})
    carriers = sorted({e['carrier_name'] for e in read if e.get('carrier_name')})
    sheet_carrier = (row['carrier'] or '').strip()
    carrier_status = NOT_COMPARABLE
    if sheet_carrier and carriers and any(_words(sheet_carrier) & _words(c) for c in carriers):
        carrier_status = AGREE
    out.append({'field': 'Carrier', 'sheet': sheet_carrier or 'blank',
                'bol': ', '.join(carriers) or ('BOL not readable' if not read else 'not printed'),
                'fl': 'FL does not record the carrier', 'status': carrier_status, 'note': None})
    return out


def _unknown_customers(shipments):
    """A shipment whose customer the books cannot name: the one thing a recall must be able to say."""
    flags, seen = [], set()
    for shipment in shipments:
        key = (shipment['transaction_id'], shipment['lot_id'])
        if (shipment.get('customer') or '').strip() or key in seen:
            continue
        seen.add(key)
        flags.append({'kind': 'SHIPMENT_CUSTOMER_UNKNOWN', 'blocks': True,
                      'title': FLAGS['SHIPMENT_CUSTOMER_UNKNOWN'][1],
                      'message': f'Shipment entry #{shipment["transaction_id"]} on {shipment.get("shipped_date") or "no date"} '
                                 f'took {lb(shipment["quantity_lb"])} of {shipment["product_name"]} lot '
                                 f'{shipment["lot_code"]}, and the books name no customer and no order for it.',
                      'refs': {'transaction_id': shipment['transaction_id'], 'lot_id': shipment['lot_id']}})
    return flags


def analyse(data):
    """Verdicts for the two tests and the answer to "Full recall possible?". Pure."""
    result = {'cutoff': data['cutoff'], 'sheet': data['sheet'], 'unmatched_rows': data['unmatched_rows'],
              'form_rows_on_or_after_cutoff': data['form_rows_on_or_after_cutoff']}
    reasons = []            # why the answer is NO: only what stops an affected customer or shipment being identified
    receiving_gaps = []     # receiving checks that did not pass; shown, and never by themselves a NO

    # ---- Test 1 ----------------------------------------------------------
    forward = data['forward']
    if not forward.get('eligible'):
        result['forward'] = {'result': NO_LOT, 'reason': forward['reason']}
    else:
        row, receipts = forward['row'], forward['receipts']
        documents = (forward.get('bol') or {}).get('documents', [])
        readable = [d for d in documents if d.get('readable')]
        # Lines from every readable photo. A photo whose WHOLE reading equals an earlier photo's (same
        # BOL number, same lines) is the same page taken twice and is counted once. A single line that
        # repeats across two different pages is kept, and marked: it cannot be told from the photos
        # whether that is one delivery shown twice or two, so the row cannot pass on it.
        bol_lines, pages, earlier, repeated = [], [], set(), []
        for document in readable:
            reading = document['extraction']
            keys = [(line.get('description'), line.get('quantity'), line.get('unit'),
                     tuple(line.get('lot_numbers') or [])) for line in reading['lines']]
            page = (reading.get('bol_number'), tuple(keys))
            if page in pages:
                continue
            pages.append(page)
            repeated.extend(line['description'] for line, key in zip(reading['lines'], keys) if key in earlier)
            bol_lines.extend(reading['lines'])
            earlier.update(keys)
        chain_lots = {lot['lot_id']: lot for lot in forward['chain']['lots']}
        for receipt in receipts:
            receipt['supplier_lots'] = chain_lots.get(receipt['lot_id'], {}).get('supplier_lots', [])
        comparisons = [{'receipt': receipt, 'fields': compare_receipt(row, receipt, bol_lines, bool(readable))}
                       for receipt in receipts]
        # BOL lines that fit none of the matched receipts (and are not the single line shown beside a
        # single product): a delivery the BOL documents that the Form row and FL do not account for.
        covered = {id(line) for line in bol_lines
                   if any(f['basis'] for f in line.get('fits', []))}
        lone = (len(bol_lines) == 1 and len(row['product_entries']) == 1 and len(receipts) == 1
                and bol_lines[0].get('resolves_to') is None)
        extra_lines = [] if lone else [line for line in bol_lines if id(line) not in covered]
        document_fields = (compare_document(row, receipts, documents, forward.get('bol_supplier'), extra_lines)
                           if receipts else [])
        problems, notes = [], []
        if not receipts:
            receiving = FAIL
            problems.append('No FL receipt could be matched to this Form row: '
                            + ' '.join(forward['match_reasons'] or ['no reason recorded.']))
        else:
            receiving = PASS
            if forward['match_reasons']:
                receiving = FAIL
                problems.extend(forward['match_reasons'])
            every = [f for c in comparisons for f in c['fields']] + document_fields
            for field in every:
                if field['status'] == DISAGREE:
                    receiving = FAIL
                    problems.append(f'{field["field"]}: the records disagree (Sheet: {field["sheet"]} | BOL: '
                                    f'{field["bol"]} | FL: {field["fl"]}).')
            unread = [d for d in documents if not d.get('readable')]
            skipped = (forward.get('bol') or {}).get('photos_not_read', 0)
            if not readable:
                receiving = worst(receiving, UNVERIFIED)
                notes.append('The BOL photo could not be opened or read: '
                             + ' '.join(d['problem'] or 'no reason given.' for d in documents) if documents
                             else 'The Form row has no readable BOL photo.')
            else:
                # Every photo on the row is evidence: one readable page never covers an unreadable one.
                if unread:
                    receiving = worst(receiving, UNVERIFIED)
                    notes.append(f'{len(unread)} of the {len(documents)} BOL photos on this Form row could not be '
                                 'opened or read: ' + ' '.join(d['problem'] or 'no reason given.' for d in unread))
                if repeated:
                    receiving = worst(receiving, UNVERIFIED)
                    notes.append('Two different BOL photos on this Form row show the same line ('
                                 + '; '.join(dict.fromkeys(repeated)) + '). It cannot be told whether that is one '
                                 'delivery photographed twice or two deliveries, so the quantity is not confirmed.')
                if skipped:
                    receiving = worst(receiving, UNVERIFIED)
                    notes.append(f'{skipped} more BOL photo(s) on this Form row were not read (only the first '
                                 f'{MAX_PHOTOS_READ} are).')
                for field in document_fields:
                    if field['field'] == 'Other products on the BOL' and field['status'] == NOT_COMPARABLE:
                        receiving = worst(receiving, UNVERIFIED)
                        notes.append(f'The BOL also lists "{field["bol"]}", which the receipts matched to this Form '
                                     f'row do not account for ({field["fl"]}).')
                for comparison in comparisons:
                    by_field = {f['field']: f for f in comparison['fields']}
                    if by_field['Product']['status'] != AGREE:
                        receiving = worst(receiving, UNVERIFIED)
                        notes.append(f'{comparison["receipt"]["product_name"]}: '
                                     + ('the product wording on the Form does not confirm it.'
                                        if by_field['Product']['form_unconfirmed'] else
                                        'the BOL could not confirm the product.'))
                    quantity = by_field['Quantity']
                    if quantity['status'] != DISAGREE and not (
                            quantity['status'] == AGREE and quantity['bol_compared'] and quantity['fl_compared']):
                        receiving = worst(receiving, UNVERIFIED)
                        notes.append(f'{comparison["receipt"]["product_name"]}: '
                                     + ('the BOL could not confirm the quantity.' if not quantity['bol_compared'] else
                                        "FL's quantity could not be compared with the Form or the BOL (no weight "
                                        'written with a unit).'))
        flags = list(forward['chain']['flags'])
        for trace in forward['traces']:
            for gap in trace['gaps']:
                flags.append({'kind': 'TRACE_GAP', 'blocks': True, 'title': FLAGS['TRACE_GAP'][1],
                              'message': f'Entry #{gap["transaction_id"]} used a lot and recorded no output lot, '
                                         f'so the trail stops there.', 'refs': gap})
        flags.extend(_unknown_customers(shipment for trace in forward['traces'] for shipment in trace['shipments']))
        _levelled(flags)
        for flag in flags:
            if flag['level'] == RECEIVING:              # e.g. the received lot has no supplier lot
                receiving = worst(receiving, FAIL)
                problems.append(flag['message'])
        # The trace is judged only on what stops an affected customer or shipment being identified.
        # With no FL receipt there was nothing to trace from: not run, rather than passed or failed.
        trace_result = None if not receipts else (FAIL if any(f['blocks'] for f in flags) else PASS)
        overall = worst(receiving, trace_result)
        result['forward'] = {
            'result': overall, 'receiving_result': receiving, 'trace_result': trace_result,
            'pool_size': forward['pool_size'], 'row': row, 'receipts': receipts, 'comparisons': comparisons,
            'document_fields': document_fields, 'documents': documents, 'problems': problems, 'notes': notes,
            'traces': forward['traces'], 'chain': forward['chain'], 'flags': flags,
            'customers': sorted({c for trace in forward['traces'] for c in trace['customers']})}
        reasons.extend(f'Forward test: {f["message"]}' for f in flags if f['blocks'])
        if receiving != PASS:
            receiving_gaps.append(f'Forward test receiving verification: {_RESULT_WORDS[receiving]}')

    # ---- Test 2 ----------------------------------------------------------
    backward = data['backward']
    if not backward.get('eligible'):
        result['backward'] = {'result': NO_LOT, 'reason': backward['reason']}
    else:
        flags, notes, evidence = list(backward['chain']['flags']), [], PASS
        tested = [item for item in backward['ingredients'] if not item['before_cutoff']]
        def flag(kind, message, **refs):
            flags.append({'kind': kind, 'blocks': FLAGS[kind][0] == BLOCKS, 'title': FLAGS[kind][1], 'message': message,
                          'refs': refs})

        for item in tested:
            name = f'{item["product_name"]} lot {item["lot_code"]}'
            if not item['receipts']:
                flag('LOT_ATTRIBUTION_UNRESOLVED',
                     f'{name} did not come in on a supplier receipt (the books show '
                     f'{(item["origin_kind"] or "no origin").replace("_", " ")}), so it cannot be traced back to a '
                     f'supplier.', lot_id=item['lot_id'])
                item['evidence'] = 'no supplier receipt in FL'
            # Each receipt into the lot is checked on its own.
            for receipt in item['receipts']:
                label = receipt['receipt_number'] or f'entry #{receipt["transaction_id"]}'
                if receipt['before_cutoff']:
                    receipt['evidence'] = 'before cutoff - not tested'
                elif receipt['kind'] != 'received':
                    flag('LOT_ATTRIBUTION_UNRESOLVED',
                         f'{name}: part of this lot came in on {label} as '
                         f'{"found stock" if receipt["kind"] == "found_inventory" else "a receipt with no supplier named"}, '
                         f'so that part cannot be traced back to a supplier.',
                         lot_id=item['lot_id'], transaction_id=receipt['transaction_id'])
                    receipt['evidence'] = 'not a supplier receipt'
                elif not receipt['form_rows']:
                    flag('RECEIVING_EVIDENCE_MISSING',
                         f'{name}: no receiving Form row could be matched to FL receipt {label}, so there is no '
                         f'independent record or BOL photo for it.',
                         lot_id=item['lot_id'], transaction_id=receipt['transaction_id'])
                    receipt['evidence'] = 'no Form row'
                elif not receipt.get('photos'):
                    evidence = worst(evidence, UNVERIFIED)
                    notes.append(f'{name}, receipt {label}: its Form row has no BOL photo.')
                    receipt['evidence'] = 'Form row, no BOL photo'
                elif not all(photo['accessible'] for photo in receipt['photos']):
                    evidence = worst(evidence, UNVERIFIED)
                    notes.append(f'{name}, receipt {label}: its BOL photo could not be opened ('
                                 + ' '.join(p['problem'] or '' for p in receipt['photos'] if not p['accessible']).strip()
                                 + ').')
                    receipt['evidence'] = 'Form row, BOL photo could not be opened'
                else:
                    receipt['evidence'] = 'Form row and BOL photo on file'
            if item['receipts']:
                item['evidence'] = '; '.join(dict.fromkeys(receipt['evidence'] for receipt in item['receipts']))
            for gap in (item.get('forward') or {}).get('gaps', []):
                flags.append({'kind': 'TRACE_GAP', 'blocks': True, 'title': FLAGS['TRACE_GAP'][1],
                              'message': f'{name}: entry #{gap["transaction_id"]} used a lot and recorded no '
                                         f'output lot, so the trail stops there.', 'refs': gap})
        for step in backward['unexplained_steps']:
            flags.append({'kind': 'PRODUCTION_INPUTS_MISSING', 'blocks': True,
                          'title': FLAGS['PRODUCTION_INPUTS_MISSING'][1],
                          'message': f'{step["product_name"]} lot {step["lot_code"]}: {step["type"]} entry '
                                     f'#{step["transaction_id"]} ({step["date"] or "no date"}) '
                                     + ('records no ingredient lots at all' if step['reason'] == 'no_inputs'
                                        else 'leads back to itself, so the trail was cut short')
                                     + ', so what went into this lot is not known.',
                          'refs': {'lot_id': step['lot_id'], 'transaction_id': step['transaction_id']}})
        for made in backward['made_lots']:
            if not made['before_cutoff'] and made['also_received']:
                flags.append({'kind': 'LOT_ATTRIBUTION_UNRESOLVED', 'blocks': True,
                              'title': FLAGS['LOT_ATTRIBUTION_UNRESOLVED'][1],
                              'message': f'{made["product_name"]} lot {made["lot_code"]} (a batch inside this '
                                         f'finished lot) was not only made: stock was also received or found '
                                         f'straight into it, so its ingredients are not all known.',
                              'refs': {'lot_id': made['lot_id']}})
        if backward['origin_kind'] != 'made':
            flags.append({'kind': 'LOT_ATTRIBUTION_UNRESOLVED', 'blocks': True,
                          'title': FLAGS['LOT_ATTRIBUTION_UNRESOLVED'][1],
                          'message': 'The books hold no make or pack entry for this finished lot.', 'refs': {}})
        if backward['also_received']:
            flags.append({'kind': 'LOT_ATTRIBUTION_UNRESOLVED', 'blocks': True,
                          'title': FLAGS['LOT_ATTRIBUTION_UNRESOLVED'][1],
                          'message': 'Part of this finished lot was not made: stock was also received or found '
                                     'straight into it, so its ingredients are not all known.', 'refs': {}})
        flags.extend(_unknown_customers([*backward['own']['shipments'], *backward['other_shipments']]))
        _levelled(flags)
        if any(flag['level'] == RECEIVING for flag in flags):       # no Form row, or no supplier lot
            evidence = worst(evidence, FAIL)
        overall = worst(FAIL if any(f['blocks'] for f in flags) else PASS, evidence)
        result['backward'] = {
            'result': overall, 'pool_size': backward['pool_size'], 'lot': backward['lot'],
            'made_lots': backward['made_lots'], 'ingredients': backward['ingredients'], 'own': backward['own'],
            'other_shipments': backward['other_shipments'], 'chain': backward['chain'], 'flags': flags,
            'evidence_result': evidence,
            'notes': notes, 'not_tested_count': len(backward['ingredients']) - len(tested),
            'made_not_tested_count': sum(1 for made in backward['made_lots'] if made['before_cutoff']),
            'inputs_mismatch': backward['inputs_mismatch'],
            'other_customers': sorted({s['customer'] or 'customer not named on the shipment'
                                       for s in backward['other_shipments']})}
        reasons.extend(f'Backward test: {f["message"]}' for f in flags if f['blocks'])
        if evidence != PASS:
            receiving_gaps.append(f'Backward test receiving evidence: {_RESULT_WORDS[evidence]}')

    # Full recall possible (owner, 2026-11-18): NO when any affected customer or shipment cannot be
    # identified (the BLOCKS flags). Receiving mismatches and unreadable evidence fail or leave
    # unverified their own section and never by themselves make it NO. With nothing traced at all
    # (no eligible lot, or a Form row with no FL receipt to trace from) it is NOT TESTED.
    traced = bool(result['forward'].get('receipts')) or result['backward']['result'] != NO_LOT
    if reasons:
        answer = 'no'
    else:
        answer = 'yes' if traced else 'not_tested'
    seen, unique = set(), []
    for text in reasons:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    all_flags = (result['forward'].get('flags') or []) + (result['backward'].get('flags') or [])
    result.update(full_recall_possible=answer, reasons=unique, receiving_gaps=receiving_gaps,
                  flag_count=len(all_flags))
    return result


# ---------------------------------------------------------------------------
# The email. Plain words; every disagreement shows all three values.
# ---------------------------------------------------------------------------

_RESULT_WORDS = {PASS: 'PASS', FAIL: 'FAIL', UNVERIFIED: 'UNVERIFIED', NO_LOT: 'No eligible lot'}
_ANSWER_WORDS = {'yes': 'YES', 'no': 'NO', 'not_tested': 'NOT TESTED (no eligible lot)'}
_FLAG_SUFFIX = {BLOCKS: '', WARNING: ' (warning only)',
                RECEIVING: ' (a receiving-record gap: it fails the receiving check; by itself it does not make the answer NO)'}


class _Email:
    """Builds the text and the HTML together from the same lines."""

    def __init__(self):
        self.text, self.html = [], []

    def heading(self, words):
        self.text += ['', words.upper(), '-' * min(len(words), 60)]
        self.html.append(f'<h3 style="font:bold 16px sans-serif;margin:22px 0 6px">{html_lib.escape(words)}</h3>')

    def line(self, words='', bold=False):
        self.text.append(words)
        style = 'font:14px sans-serif;margin:4px 0' + (';font-weight:bold' if bold else '')
        self.html.append(f'<p style="{style}">{html_lib.escape(words)}</p>')

    def link(self, label, url):
        if not url:
            return
        self.text.append(f'{label}: {url}')
        self.html.append(f'<p style="font:14px sans-serif;margin:4px 0"><a href="{html_lib.escape(url, quote=True)}">'
                         f'{html_lib.escape(label)}</a></p>')

    def bullets(self, items):
        if not items:
            return
        self.text += [f'  - {item}' for item in items]
        self.html.append('<ul style="font:14px sans-serif;margin:4px 0">'
                         + ''.join(f'<li>{html_lib.escape(item)}</li>' for item in items) + '</ul>')

    def table(self, header, rows):
        for row in rows:
            self.text.append('  ' + ' | '.join(f'{name}: {value}' for name, value in zip(header, row)))
        cell = 'style="border:1px solid #ccc;padding:4px 6px;font:13px sans-serif;text-align:left"'
        self.html.append('<table style="border-collapse:collapse;margin:6px 0"><tr>'
                         + ''.join(f'<th {cell}>{html_lib.escape(str(h))}</th>' for h in header) + '</tr>'
                         + ''.join('<tr>' + ''.join(f'<td {cell}>{html_lib.escape(str(v))}</td>' for v in row) + '</tr>'
                                   for row in rows) + '</table>')


def _amount(value, unit=''):
    return (f'{value:,.2f}'.rstrip('0').rstrip('.') + (f' {unit}' if unit else '')) if value else '-'


def _lot_rows(chain):
    rows = []
    for lot in chain['lots']:
        unit = lot['unit']
        adjusted = ' '.join(part for part in (
            '+' + _amount(lot['adjusted_in']) if lot['adjusted_in'] else '',
            '-' + _amount(lot['adjusted_out']) if lot['adjusted_out'] else '') if part)
        rows.append([f'{lot["product_name"]} lot {lot["lot_code"]}', _amount(lot['received'], unit),
                     _amount(lot['made'], unit), _amount(lot['used'], unit), _amount(lot['shipped'], unit),
                     adjusted or '-', _amount(lot['on_hand'], unit) if lot['on_hand'] else f'0 {unit}'])
    return rows


def render_email(analysis, run_at):
    day = run_at.astimezone(PLANT_TIMEZONE)
    forward, backward = analysis['forward'], analysis['backward']
    answer = _ANSWER_WORDS[analysis['full_recall_possible']]
    email = _Email()
    email.line(f'Trial recall, {day.strftime("%a %b")} {day.day}, {day.year}. Full recall possible: {answer}.', bold=True)
    email.line(f'Forward test: {_RESULT_WORDS[forward["result"]]}. Backward test: {_RESULT_WORDS[backward["result"]]}. '
               f'Only receipts and finished lots dated on or after {analysis["cutoff"]} are tested.')
    email.line('This is a drill. It only read the books, the receiving Form and the BOL photos. Nothing was changed.')

    # 1 ------------------------------------------------------------------
    email.heading('1. Receiving verification (Sheet / BOL vs FL)')
    if forward['result'] == NO_LOT:
        email.line(f'Forward test: No eligible lot. {forward["reason"]}')
    else:
        row = forward['row']
        email.line(f'Forward test picked Form row {row["row_number"]} at random (out of {forward["pool_size"]} with a '
                   f'BOL photo): {row["timestamp"]}, entered by {row["submitted_by"] or "no name"}, '
                   f'lot {row["lot_text"] or "blank"}, {"; ".join(row["products"]) or "no product"}.')
        email.line(f'Result: {_RESULT_WORDS[forward["receiving_result"]]}', bold=True)
        email.link(f'Open Form row {row["row_number"]} in the sheet', row['sheet_row_url'])
        for document in forward['documents']:
            email.link('Open the BOL photo', document['url'])
            if document['problem']:
                email.line(f'BOL photo: {document["problem"]}')
        header = ['Field', 'Sheet', 'BOL', 'FL', 'Result']
        for comparison in forward['comparisons']:
            receipt = comparison['receipt']
            email.line(f'FL receipt {receipt["receipt_number"] or "#" + str(receipt["transaction_id"])}, dated '
                       f'{receipt["business_date"]}: {receipt["product_name"]}, lot {receipt["lot_code"]}.')
            email.table(header, [[f['field'], f['sheet'], f['bol'], f['fl'], f['status']] for f in comparison['fields']])
            email.bullets([f'{f["field"]}: {f["note"]}' for f in comparison['fields']
                           if f['status'] == AGREE and f.get('note')])
        if forward['document_fields']:
            email.table(header, [[f['field'], f['sheet'], f['bol'], f['fl'], f['status']]
                                 for f in forward['document_fields']])
        email.bullets(forward['problems'] + forward['notes'])
        email.line('Where the records differ, all three are shown. None of them is assumed to be the right one.')
    if backward['result'] != NO_LOT:
        lot = backward['lot']
        email.line(f'Backward test picked finished lot {lot["lot_code"]} ({lot["product_name"]}, first made '
                   f'{lot["first_made"]}) at random out of {backward["pool_size"]} shipped lots. Receiving evidence '
                   f'for each ingredient lot ({_RESULT_WORDS[backward["evidence_result"]]}):', bold=True)
        rows = []
        for item in backward['ingredients']:
            if item['before_cutoff']:
                rows.append([item['product_name'], item['lot_code'], '', item.get('first_in') or 'no date',
                             'before cutoff - not tested', '', ''])
                continue
            if not item['receipts']:
                rows.append([item['product_name'], item['lot_code'], '', item.get('first_in') or 'no date',
                             item.get('evidence', ''), '', ''])
            for receipt in item['receipts']:
                rows.append([item['product_name'], item['lot_code'],
                             receipt['receipt_number'] or f'#{receipt["transaction_id"]}',
                             receipt.get('business_date') or 'no date', receipt.get('evidence', ''),
                             ' '.join(r['sheet_row_url'] or f'row {r["row_number"]}' for r in receipt['form_rows']),
                             ' '.join(p['url'] for p in receipt.get('photos', []))])
        if rows:
            email.table(['Ingredient', 'Lot', 'FL receipt', 'Received', 'Evidence', 'Form row', 'BOL photo'], rows)
        else:
            email.line('The books show no ingredient lots behind this finished lot.')
        email.line('In the backward test each BOL photo is opened to check it is on file; it is not read by the model.')

    # 2 ------------------------------------------------------------------
    email.heading('2. Inventory and consumption reconciliation')
    header = ['Lot', 'Received', 'Made', 'Used in production', 'Shipped', 'Adjusted', 'On hand now']
    for label, test in (('Forward test', forward), ('Backward test', backward)):
        if test['result'] == NO_LOT:
            continue
        email.line(f'{label}: every lot in the chain, from posted entries only.', bold=True)
        rows = _lot_rows(test['chain'])
        email.table(header, rows[:MAX_EMAIL_ROWS])
        if len(rows) > MAX_EMAIL_ROWS:
            email.line(f'... and {len(rows) - MAX_EMAIL_ROWS} more lots (all saved with this run).')
    if backward['result'] != NO_LOT and backward['inputs_mismatch']:
        email.line('Backward test: for at least one make in this chain, the pounds of ingredients recorded are '
                   'outside 80-120% of the pounds made.')
    email.line('On hand = everything put in minus everything taken out. Quantities follow the recorded lot lines '
               'only; nothing is assumed about which stock was used first.')

    # 3 ------------------------------------------------------------------
    email.heading('3. Forward traceability')
    if forward['result'] == NO_LOT:
        email.line('Forward test: No eligible lot.')
    else:
        for trace in forward['traces']:
            start = trace['lots'][0]
            email.line(f'From {start["product_name"]} lot {start["lot_code"]}: {len(trace["lots"]) - 1} lot(s) made '
                       f'from it, {len(trace["shipments"])} shipment(s), {len(trace["customers"])} customer(s).', bold=True)
            if trace['steps']:
                email.table(['Date', 'Entry', 'Used', 'From lot', 'Into', 'Output lot'],
                            [[s['date'] or '', f'{s["type"]} #{s["transaction_id"]}', _amount(s['used'], s['unit']),
                              s['source_lot_code'] or '', s['output_product'] or 'no output recorded',
                              s['output_lot_code'] or ''] for s in trace['steps']])
            else:
                email.line('Not used in any make or pack yet.')
            if trace['shipments']:
                email.table(['Shipped', 'Customer', 'Order', 'Product', 'Lot', 'Quantity'],
                            [[s['shipped_date'] or '', s['customer'] or 'not named', ', '.join(s['order_numbers']),
                              s['product_name'], s['lot_code'], lb(s['quantity_lb'])] for s in trace['shipments']])
            else:
                email.line('Nothing made from this lot has shipped yet.')
        if not forward['traces']:
            email.line('There is no FL receipt to trace forward from.')
    if backward['result'] != NO_LOT:
        own = backward['own']
        email.line(f'Backward test, finished lot {backward["lot"]["lot_code"]}: shipped to '
                   f'{", ".join(own["customers"]) or "no one yet"}.', bold=True)
        if own['shipments']:
            email.table(['Shipped', 'Customer', 'Order', 'Quantity'],
                        [[s['shipped_date'] or '', s['customer'] or 'not named', ', '.join(s['order_numbers']),
                          lb(s['quantity_lb'])] for s in own['shipments']])
        others = backward['other_shipments']
        email.line(f'Other customers reached by the same ingredient lots: '
                   f'{", ".join(backward["other_customers"]) or "none"} ({len(others)} other shipment(s)).')
        if others:
            email.table(['Shipped', 'Customer', 'Order', 'Product', 'Lot', 'Quantity', 'Shared ingredient lot'],
                        [[s['shipped_date'] or '', s['customer'] or 'not named', ', '.join(s['order_numbers']),
                          s['product_name'], s['lot_code'], lb(s['quantity_lb']),
                          ', '.join(sorted(set(s['via_ingredient_lots'])))] for s in others[:60]])
            if len(others) > 60:
                email.line(f'... and {len(others) - 60} more shipments (all saved with this run).')
        if backward['not_tested_count']:
            email.line(f'{backward["not_tested_count"]} ingredient lot(s) in this chain are before the cutoff and '
                       f'were not tested or traced forward.')
        if backward['made_not_tested_count']:
            email.line(f'{backward["made_not_tested_count"]} batch lot(s) in this chain were first made before the '
                       f'cutoff and were not tested.')

    # 4 ------------------------------------------------------------------
    email.heading('4. Missing or conflicting records')
    any_flag = False
    for label, test in (('Forward test', forward), ('Backward test', backward)):
        flags = test.get('flags') or []
        if flags:
            any_flag = True
            email.line(f'{label}:', bold=True)
            email.bullets([f'{f["title"]}: {f["message"]}' + _FLAG_SUFFIX[f['level']] for f in flags[:MAX_EMAIL_ROWS]])
            if len(flags) > MAX_EMAIL_ROWS:
                email.line(f'... and {len(flags) - MAX_EMAIL_ROWS} more (all saved with this run).')
    if not any_flag:
        email.line('No flags on the lots in these two chains.')
    unmatched = analysis['unmatched_rows']
    email.line(f'Receiving form row with no matching FL receipt: {len(unmatched)} of '
               f'{analysis["form_rows_on_or_after_cutoff"]} row(s) on or after the cutoff.', bold=True)
    for item in unmatched[:25]:
        email.bullets([f'Row {item["row_number"]} ({item["timestamp"]}, lot {item["lot_text"] or "blank"}, '
                       f'{"; ".join(item["products"]) or "no product"}): {" ".join(item["reasons"])}'])
        email.link(f'Open Form row {item["row_number"]}', item['sheet_row_url'])
    if len(unmatched) > 25:
        email.line(f'... and {len(unmatched) - 25} more (all saved with this run).')

    # 5 ------------------------------------------------------------------
    email.heading(f'5. Full recall possible: {answer}')
    if analysis['full_recall_possible'] == 'yes':
        email.line('Every affected customer and shipment in the chains tested can be identified from the records.')
    elif analysis['full_recall_possible'] == 'not_tested':
        email.line('Nothing could be traced this time (no eligible lot, or no FL receipt to trace from), so nothing '
                   'was proven either way.')
    else:
        email.line('Why not (each of these stops an affected customer or shipment being identified):')
        email.bullets(analysis['reasons'][:MAX_EMAIL_ROWS])
    if analysis['receiving_gaps']:
        email.line('Receiving checks that did not pass (section 1): ' + '; '.join(analysis['receiving_gaps'])
                   + '. By themselves these do not change the answer above.')
    email.line('NO means only this: somewhere in these chains a customer or shipment could not be identified '
               '(unresolved lot attribution, made product removed with no destination, an order closed "shipped, '
               'not recorded", an ingredient lot with no receipt, or a broken lot link). Receiving mismatches show '
               'as FAIL in section 1. A missing BOL photo or number and entries under the shared login are '
               'warnings. The flags cover the lots in these two chains (in the backward test: everything the '
               'shared ingredient lots went into), not the whole plant. UNVERIFIED means the evidence could not '
               'be read; it is never counted as a pass.')
    subject = (f'Factory Ledger trial recall, {day.strftime("%a %b")} {day.day}: Full recall possible: {answer}')
    return {'subject': subject, 'text': '\n'.join(email.text).strip() + '\n', 'html': '\n'.join(email.html)}


# ---------------------------------------------------------------------------
# Saving a run and sending the email. The ONLY write is the INSERT below.
# ---------------------------------------------------------------------------

def email_enabled(env=None):
    env = os.environ if env is None else env
    return env.get('TRIAL_RECALL_EMAIL_ENABLED', '').strip().lower() in ('1', 'true', 'yes', 'on')


def deliver(email, env=None, sender=None):
    """Send through the daily reconciliation's Resend setup (ALERT_EMAIL_TO, RESEND_API_KEY,
    RECONCILIATION_EMAIL_FROM). Off unless TRIAL_RECALL_EMAIL_ENABLED=1. Never raises."""
    env = os.environ if env is None else env
    if not email_enabled(env):
        return {'status': 'disabled'}
    try:
        return (sender or reconciliation.send_email)(email, env)
    except Exception as exc:  # the run is saved whatever the email step does
        return {'status': 'failed', 'error': f'Email step failed ({type(exc).__name__})'}


def save_run(cur, *, started_at, analysis, email, delivery):
    cur.execute("""INSERT INTO trial_recall_runs
                       (started_at, plant_date, cutoff_date, job_status, forward_result, backward_result,
                        full_recall_possible, flag_count, email_status, email_to, email_sent_at, email_error,
                        email_provider_id, result)
                   VALUES (%s, %s, %s, 'completed', %s, %s, %s, %s, %s, %s,
                           CASE WHEN %s = 'sent' THEN clock_timestamp() END, %s, %s, %s) RETURNING id""",
                (started_at, started_at.astimezone(PLANT_TIMEZONE).date(), analysis['cutoff'],
                 analysis['forward']['result'], analysis['backward']['result'], analysis['full_recall_possible'],
                 analysis['flag_count'], delivery['status'], delivery.get('to'), delivery['status'],
                 delivery.get('error'), delivery.get('provider_id'),
                 Json(dict(analysis, email_subject=email['subject'], email_text=email['text']))))
    row = cur.fetchone()
    return row['id'] if isinstance(row, dict) else row[0]


def record_failed_run(cur, error_name, started_at, cutoff=CUTOFF):
    """Best-effort note that today's drill did not run, so a missing email is never mistaken for a pass."""
    cur.execute("""INSERT INTO trial_recall_runs (started_at, plant_date, cutoff_date, job_status, job_error, email_status)
                   VALUES (%s, %s, %s, 'failed', %s, 'not_attempted')""",
                (started_at, started_at.astimezone(PLANT_TIMEZONE).date(), cutoff, error_name[:200]))


def ran_today(cur, now):
    cur.execute("""SELECT 1 FROM trial_recall_runs WHERE job_status = 'completed' AND plant_date = %s LIMIT 1""",
                (now.astimezone(PLANT_TIMEZONE).date(),))
    return cur.fetchone() is not None


# ---------------------------------------------------------------------------
# Two read-only endpoints. Owner and office (and the master key); never in
# either GPT schema.
# ---------------------------------------------------------------------------

def _fail(status, code, message):
    raise HTTPException(status_code=status, detail={'error_code': code, 'message': message})


def _require(api, request):
    role = permissions.role_of(api._actor_identity(request))
    if role not in {permissions.OWNER, permissions.SELECTED_OWNER, permissions.LEGACY_LEDGER, permissions.OFFICE}:
        _fail(403, 'ROLE_NOT_ALLOWED', 'The trial recall results are for the owner and the office.')


_RUN_COLUMNS = """id, started_at, finished_at, plant_date, cutoff_date, job_status, job_error, forward_result,
                  backward_result, full_recall_possible, flag_count, email_status, email_to, email_sent_at,
                  email_error"""


def register_routes(app, api):
    @app.get('/verification/trial-recall/runs', include_in_schema=False)
    def trial_recall_runs(request: Request, limit: int = 30, _: bool = Depends(api.verify_api_key)):
        _require(api, request)
        with api.get_transaction() as cur:
            if not migration_applied(cur):
                _fail(503, 'TRIAL_RECALL_UNAVAILABLE', 'The trial recall is not switched on yet.')
            cur.execute(f'SELECT {_RUN_COLUMNS} FROM trial_recall_runs ORDER BY id DESC LIMIT %s',
                        (max(1, min(limit, 200)),))
            return {'runs': [dict(row) for row in cur.fetchall()]}

    @app.get('/verification/trial-recall/runs/{run_id}', include_in_schema=False)
    def trial_recall_run(run_id: int, request: Request, _: bool = Depends(api.verify_api_key)):
        _require(api, request)
        with api.get_transaction() as cur:
            if not migration_applied(cur):
                _fail(503, 'TRIAL_RECALL_UNAVAILABLE', 'The trial recall is not switched on yet.')
            cur.execute(f'SELECT {_RUN_COLUMNS}, result FROM trial_recall_runs WHERE id = %s', (run_id,))
            run = cur.fetchone()
            if run is None:
                _fail(404, 'RUN_NOT_FOUND', f'No trial recall run {run_id}.')
            return {'run': dict(run)}
