"""Daily inventory reconciliation (migration 094; docs/verification/daily-reconciliation-scoping.md).

A nightly books-integrity check. It snapshots every lot's posted balance, then
checks the business day (America/New_York):

  * make vs recipe      each make's ingredient pounds against batch_formulas x batches
  * balance roll-forward previous snapshot + what was entered = tonight's balance, per lot
  * pack provenance     packed more than was made / packed with no make, per batch lot
  * always listed       lots below zero; adjustments with reason "unknown" or no named
                        person; voids and late entries (information only)

It proves nothing moved a balance except entries it saw. It cannot see the
shelf: physical loss, miscounts and unrecorded use need a count.

READ-ONLY toward the ledger. The only statements that write are the three
INSERTs in `save_run`, and they target the three tables of migration 094
(inventory_snapshots, reconciliation_runs, reconciliation_flags), all
append-only. Nothing here posts, voids, adjusts, updates or deletes; a test
(tests/test_daily_reconciliation.py) pins that.

Shape: `load_window` reads the ledger into plain JSON-able data, `analyse` is a
pure function of that data (so a saved export replays without a database),
`render_email` turns the result into the owner email, `run_job` is the nightly
sequence. Feature-module rule: never imports main; routes reach it via `api`.
"""
import html as html_lib
import os
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import httpx
from fastapi import Depends, HTTPException, Request
from psycopg2.extras import Json, execute_values

import permissions

PLANT_TIMEZONE = ZoneInfo('America/New_York')   # = main.PLANT_TIMEZONE
MARKER = '094_daily_reconciliation'
TOLERANCE_FLOOR_LB = 5.0
TOLERANCE_PCT = 0.02
COCONUT_YIELD_PCT = 0.02
EPS = 0.00005                                   # quantities are numeric(14,4)
RESEND_URL = 'https://api.resend.com/emails'
# Resend's shared test sender delivers only to the Resend account's own address.
# Set RECONCILIATION_EMAIL_FROM once sevenwellsgranola.com is verified there.
DEFAULT_EMAIL_FROM = 'Factory Ledger <onboarding@resend.dev>'

ALARM, REVIEW, YIELD, INFO = 'alarm', 'review', 'yield', 'info'
ROUTES = frozenset({
    ('GET', '/verification/reconciliation/runs'),
    ('GET', '/verification/reconciliation/runs/{run_id}'),
    ('GET', '/verification/reconciliation/preview'),
})


# ---------------------------------------------------------------------------
# Reading the ledger. Everything below this banner only SELECTs.
# ---------------------------------------------------------------------------

def _rows(cur, sql, params=None):
    cur.execute(sql, params or {})
    names = [column[0] for column in cur.description]
    return [dict(row) if isinstance(row, dict) else dict(zip(names, row)) for row in cur.fetchall()]


# A transaction's status as it stood at a moment: the latest correction entered
# by then, on top of the row's own status (rows are append-only since 039, so
# the row itself never changes). Same rule as view ledger_current_transactions,
# with a time limit.
_STATUS_AT = """COALESCE((SELECT CASE c.event_type WHEN 'void' THEN 'voided' WHEN 'restore' THEN 'posted'
                                 ELSE COALESCE(c.replacement_values->>'status', t.status, 'posted') END
                          FROM ledger_corrections c
                          WHERE c.target_table = 'transactions' AND c.target_id = t.id
                            AND c.created_at <= %({at})s
                          ORDER BY c.created_at DESC, c.id DESC LIMIT 1), t.status, 'posted')"""

# Posted lines as they stood at %(at)s (line amendments entered by then applied).
_LINES_AT = """
    SELECT tl.id AS line_id, t.id AS transaction_id, t.type,
           COALESCE((lc.replacement_values->>'product_id')::integer, tl.product_id) AS product_id,
           COALESCE((lc.replacement_values->>'lot_id')::integer, tl.lot_id) AS lot_id,
           COALESCE((lc.replacement_values->>'quantity_lb')::numeric, tl.quantity_lb) AS quantity_lb
    FROM transactions t
    JOIN transaction_lines tl ON tl.transaction_id = t.id
    LEFT JOIN LATERAL (SELECT c.replacement_values FROM ledger_corrections c
                       WHERE c.target_table = 'transaction_lines' AND c.target_id = tl.id
                         AND c.created_at <= %(at)s
                       ORDER BY c.created_at DESC, c.id DESC LIMIT 1) lc ON true
    WHERE t.created_at <= %(at)s AND """ + _STATUS_AT.format(at='at') + """ = 'posted'
      AND (%(tx_ids)s::integer[] IS NULL OR t.id = ANY(%(tx_ids)s::integer[]))
"""


def _iso(value):
    return value.isoformat() if isinstance(value, (datetime, date)) else value


def _lines_at(cur, at, tx_ids=None):
    rows = _rows(cur, _LINES_AT + ' ORDER BY tl.id', {'at': at, 'tx_ids': tx_ids})
    for row in rows:
        row['quantity_lb'] = float(row['quantity_lb'])
    return rows


def balances_at(cur, at):
    """Posted balance per lot as it stood at a moment, rebuilt from entry and
    correction times. At 'now' this equals view v_lot_quantities."""
    rows = _rows(cur, 'SELECT x.lot_id, SUM(x.quantity_lb) AS quantity_lb FROM (' + _LINES_AT
                 + ') x GROUP BY x.lot_id HAVING SUM(x.quantity_lb) <> 0 ORDER BY x.lot_id',
                 {'at': at, 'tx_ids': None})
    return [[row['lot_id'], float(row['quantity_lb'])] for row in rows]


def current_balances(cur):
    """Tonight's posted, void-aware balance per lot, straight from the view."""
    rows = _rows(cur, 'SELECT lot_id, quantity_on_hand FROM v_lot_quantities '
                      'WHERE quantity_on_hand <> 0 ORDER BY lot_id')
    return [[row['lot_id'], float(row['quantity_on_hand'])] for row in rows]


def migration_applied(cur):
    cur.execute('SELECT 1 FROM migration_markers WHERE name = %s', (MARKER,))
    return cur.fetchone() is not None


def load_window(cur, start, end, *, opening_run_id=None, reconstruct=False):
    """Read everything `analyse` needs for entries made in (start, end].

    The window is by ENTRY time (transactions.created_at) and correction time
    (ledger_corrections.created_at), not by business date, so nothing entered
    late or voided later can fall between two nights.

    opening_run_id: the earlier run whose stored snapshot is the opening balance
    (the nightly path). reconstruct=True rebuilds opening and ending balances
    from timestamps instead (replaying a past day; no snapshot needed).
    """
    window = {'start': start, 'end': end}
    corrections = _rows(cur, """
        SELECT c.target_table, c.event_type, c.reason, c.operator_id, c.created_at, a.name AS actor_name,
               CASE WHEN c.target_table = 'transactions' THEN c.target_id::integer
                    ELSE (SELECT tl.transaction_id FROM transaction_lines tl WHERE tl.id = c.target_id) END
                   AS transaction_id
        FROM ledger_corrections c LEFT JOIN actors a ON a.id = c.entered_by_actor_id
        WHERE c.created_at > %(start)s AND c.created_at <= %(end)s
        ORDER BY c.created_at, c.id""", window)
    created = _rows(cur, 'SELECT id FROM transactions WHERE created_at > %(start)s AND created_at <= %(end)s', window)
    tx_ids = sorted({row['id'] for row in created}
                    | {row['transaction_id'] for row in corrections if row['transaction_id'] is not None})
    transactions = _rows(cur, """
        SELECT t.id, t.type, t.created_at, t.business_date, t.operator_id, t.entered_by_actor_id AS actor_id,
               a.name AS actor_name, t.reason_code, t.receipt_number, left(t.notes, 200) AS notes,
               (t.created_at AT TIME ZONE 'America/New_York')::date AS entered_plant_date,
               t.created_at > %(start)s AS in_window,
               """ + _STATUS_AT.format(at='end') + """ = 'posted' AS posted_after
        FROM transactions t LEFT JOIN actors a ON a.id = t.entered_by_actor_id
        WHERE t.id = ANY(%(ids)s::integer[]) ORDER BY t.id""", {**window, 'ids': tx_ids})
    before = _lines_at(cur, start, tx_ids)
    after = _lines_at(cur, end, tx_ids)
    by_id = {t['id']: t for t in transactions}
    # Entered and voided inside the window: no balance effect, shown for information.
    voided_lines = _rows(cur, 'SELECT transaction_id, product_id, lot_id, quantity_lb FROM transaction_lines '
                              'WHERE transaction_id = ANY(%(ids)s::integer[]) ORDER BY id',
                         {'ids': [t['id'] for t in transactions if t['in_window'] and not t['posted_after']]})

    make_ids = [t['id'] for t in transactions if t['type'] == 'make' and t['in_window'] and t['posted_after']]
    output_products = sorted({line['product_id'] for line in after
                              if line['transaction_id'] in make_ids and line['quantity_lb'] > 0})
    recipes = _rows(cur, """
        SELECT product_id, ingredient_product_id, quantity_lb,
               COALESCE(exclude_from_inventory, false) AS exclude_from_inventory
        FROM batch_formulas WHERE product_id = ANY(%(ids)s::integer[])
        ORDER BY product_id, ingredient_product_id, id""", {'ids': output_products})
    substitutions = _rows(cur, """
        SELECT transaction_id, ingredient_product_id, substitute_product_id, reason_code
        FROM transaction_substitutions WHERE transaction_id = ANY(%(ids)s::integer[])
        ORDER BY transaction_id, ingredient_product_id""", {'ids': make_ids})

    # Whole-life history of every lot a pack drew from in the window.
    pack_lots = sorted({line['lot_id'] for line in after if line['quantity_lb'] < 0
                        and by_id[line['transaction_id']]['type'] == 'pack'
                        and by_id[line['transaction_id']]['in_window']})
    history_rows = _rows(cur, """
        SELECT x.lot_id, x.transaction_id, x.type, x.quantity_lb, t.reason_code, t.operator_id,
               a.name AS actor_name
        FROM (""" + _LINES_AT + """) x
        JOIN transactions t ON t.id = x.transaction_id
        LEFT JOIN actors a ON a.id = t.entered_by_actor_id
        WHERE x.lot_id = ANY(%(lots)s::integer[]) ORDER BY x.transaction_id, x.line_id""",
                         {'at': end, 'tx_ids': None, 'lots': pack_lots}) if pack_lots else []
    histories = {lot: {'lot_id': lot, 'made_lb': 0.0, 'packed_lb': 0.0, 'used_in_makes_lb': 0.0,
                       'other_in_lb': 0.0, 'other_out_lb': 0.0, 'adjustments': []} for lot in pack_lots}
    for row in history_rows:
        history, quantity = histories[row['lot_id']], float(row['quantity_lb'])
        if row['type'] == 'make':
            history['made_lb' if quantity > 0 else 'used_in_makes_lb'] += abs(quantity)
        elif row['type'] == 'pack' and quantity < 0:
            history['packed_lb'] += -quantity
        else:
            history['other_in_lb' if quantity > 0 else 'other_out_lb'] += abs(quantity)
            if row['type'] == 'adjust':
                history['adjustments'].append({
                    'transaction_id': row['transaction_id'], 'quantity_lb': quantity,
                    'reason_code': row['reason_code'], 'actor_name': row['actor_name'],
                    'operator_id': row['operator_id']})

    if reconstruct:
        opening, ending = balances_at(cur, start), balances_at(cur, end)
    else:
        ending = current_balances(cur)
        opening = None
        if opening_run_id is not None:
            opening = [[row['lot_id'], float(row['quantity_lb'])] for row in _rows(
                cur, 'SELECT lot_id, quantity_lb FROM inventory_snapshots WHERE job_run_id = %(run)s ORDER BY lot_id',
                {'run': opening_run_id})]

    lot_ids = sorted({line['lot_id'] for line in before + after + voided_lines} | {lot for lot, _ in ending}
                     | {lot for lot, _ in (opening or [])})
    lots = _rows(cur, 'SELECT id, lot_code, product_id FROM lots WHERE id = ANY(%(ids)s::integer[]) ORDER BY id',
                 {'ids': lot_ids})
    product_ids = sorted({lot['product_id'] for lot in lots} | {line['product_id'] for line in before + after}
                         | {r['ingredient_product_id'] for r in recipes} | set(output_products)
                         | {s[key] for s in substitutions
                            for key in ('ingredient_product_id', 'substitute_product_id') if s[key] is not None})
    products = _rows(cur, """SELECT id, name, type, default_batch_lb, COALESCE(yield_multiplier, 1.0) AS yield_multiplier
                             FROM products WHERE id = ANY(%(ids)s::integer[]) ORDER BY id""", {'ids': product_ids})

    def plain(rows):
        return [{key: (float(value) if key.endswith('_lb') and value is not None else _iso(value))
                 for key, value in row.items()} for row in rows]

    plant_day = start.astimezone(PLANT_TIMEZONE).date() if reconstruct else \
        end.astimezone(PLANT_TIMEZONE).date() - timedelta(days=1)
    return {
        'plant_date': plant_day.isoformat(), 'window_start': _iso(start), 'window_end': _iso(end),
        'mode': 'replay' if reconstruct else 'nightly', 'opening_run_id': opening_run_id,
        'transactions': plain(transactions), 'lines_before': before, 'lines_after': after,
        'voided_lines': plain(voided_lines),
        'corrections': plain(corrections), 'recipes': plain(recipes), 'substitutions': plain(substitutions),
        'lot_histories': [histories[lot] for lot in pack_lots],
        'opening': opening, 'ending': ending, 'lots': lots,
        'products': [dict(p, default_batch_lb=float(p['default_batch_lb']) if p['default_batch_lb'] is not None else None,
                          yield_multiplier=float(p['yield_multiplier'])) for p in products],
    }


# ---------------------------------------------------------------------------
# The checks. Pure functions of the loaded data: no database, no clock.
# ---------------------------------------------------------------------------

def tolerance(movement_lb):
    """The one tolerance rule: the greater of 5 lb or 2% of the pounds that moved."""
    return max(TOLERANCE_FLOOR_LB, TOLERANCE_PCT * abs(movement_lb))


def expected_usage(output_lb, batch_lb, recipe, actual, substitutions=()):
    """Recipe pounds a make should have used, per ingredient product.

    Batch count is not stored: it is output / batch size. Water and any other
    `exclude_from_inventory` line is skipped. A recorded substitution is
    correct by definition: whatever the original ingredient fell short by moves
    to the substitute (or to nothing, when "not used" was recorded).
    Returns (batches, {ingredient_product_id: expected_lb}).
    """
    batches = output_lb / batch_lb
    expected = defaultdict(float)
    for row in recipe:
        if not row['exclude_from_inventory']:
            expected[row['ingredient_product_id']] += row['quantity_lb'] * batches
    for sub in substitutions:
        original, substitute = sub['ingredient_product_id'], sub['substitute_product_id']
        shortfall = max(expected.get(original, 0.0) - actual.get(original, 0.0), 0.0)
        expected[original] = expected.get(original, 0.0) - shortfall
        if substitute is not None:
            expected[substitute] += shortfall
    return batches, dict(expected)


def lb(value, signed=False):
    text = f'{value:+,.1f}' if signed else f'{value:,.1f}'
    return (text[:-2] if text.endswith('.0') else text).replace('-', '−') + ' lb'


def _when(iso, with_date=True):
    moment = datetime.fromisoformat(iso).astimezone(PLANT_TIMEZONE)
    clock = f'{moment.hour % 12 or 12}:{moment:%M} {"AM" if moment.hour < 12 else "PM"}'
    return f'{moment:%b} {moment.day} {clock}' if with_date else clock


def _person(actor_name, operator_id=None):
    if actor_name:
        return actor_name
    if operator_id and operator_id != 'legacy-shared-key':
        return f'{operator_id} (no personal sign-in)'
    return 'the shared key (no named person)'


def analyse(data):
    """Run every check over one window of loaded ledger data."""
    products = {p['id']: p for p in data['products']}
    lots = {lot['id']: lot for lot in data['lots']}
    transactions = {t['id']: t for t in data['transactions']}
    before, after = defaultdict(list), defaultdict(list)
    for line in data['lines_before']:
        before[line['transaction_id']].append(line)
    for line in data['lines_after']:
        after[line['transaction_id']].append(line)
    flags, makes = [], []

    def product_name(product_id):
        return products[product_id]['name'] if product_id in products else f'product {product_id}'

    def lot_label(lot_id):
        lot = lots.get(lot_id)
        return f'{product_name(lot["product_id"])} lot {lot["lot_code"]}' if lot else f'lot {lot_id}'

    def flag(kind, severity, message, *, product_id=None, lot_id=None, transaction_id=None, expected=None,
             actual=None, tolerance_lb=None, **detail):
        if lot_id is not None and product_id is None and lot_id in lots:
            product_id = lots[lot_id]['product_id']
        flags.append({
            'kind': kind, 'severity': severity, 'product_id': product_id, 'lot_id': lot_id,
            'transaction_id': transaction_id,
            'expected_lb': None if expected is None else round(expected, 4),
            'actual_lb': None if actual is None else round(actual, 4),
            'difference_lb': None if expected is None or actual is None else round(actual - expected, 4),
            'tolerance_lb': None if tolerance_lb is None else round(tolerance_lb, 4),
            'detail': {'message': message, **detail}})

    # --- make vs recipe ----------------------------------------------------
    expected_ratio = {}        # (make id, ingredient product) -> expected / actual
    for tx in data['transactions']:
        if tx['type'] != 'make' or not tx['in_window'] or not tx['posted_after']:
            continue
        lines = after[tx['id']]
        output = defaultdict(float)
        actual = defaultdict(float)
        for line in lines:
            if line['quantity_lb'] > 0:
                output[line['product_id']] += line['quantity_lb']
            else:
                actual[line['product_id']] += -line['quantity_lb']
        out_product = max(output, key=output.get) if output else None
        recipe = [r for r in data['recipes'] if r['product_id'] == out_product]
        batch_lb = None
        if out_product in products and products[out_product]['default_batch_lb']:
            batch_lb = products[out_product]['default_batch_lb'] * products[out_product]['yield_multiplier']
        label = f'Make TX{tx["id"]} ({product_name(out_product) if out_product else "no output line"}'
        record = {'transaction_id': tx['id'], 'product_id': out_product,
                  'product': product_name(out_product) if out_product else None,
                  'output_lb': round(sum(output.values()), 4), 'batch_lb': batch_lb, 'batches': None,
                  'ingredients': [], 'substitutions': [], 'checked': False}
        makes.append(record)
        if not recipe or not batch_lb:
            flag('MAKE_NO_RECIPE', ALARM,
                 f'{label}) has no recipe or batch size on file, so its ingredient use could not be checked.',
                 product_id=out_product, transaction_id=tx['id'])
            continue
        subs = [s for s in data['substitutions'] if s['transaction_id'] == tx['id']]
        batches, expected = expected_usage(output[out_product], batch_lb, recipe, actual, subs)
        skipped = {r['ingredient_product_id'] for r in recipe if r['exclude_from_inventory']}
        label += f', {batches:g} batch{"" if batches == 1 else "es"})'
        record.update(batches=round(batches, 4), checked=True, substitutions=[
            {'ingredient': product_name(s['ingredient_product_id']),
             'substitute': product_name(s['substitute_product_id']) if s['substitute_product_id'] else None,
             'reason_code': s['reason_code']} for s in subs])
        for ingredient in sorted(set(expected) | set(actual)):
            if ingredient in skipped and ingredient not in expected:
                continue
            want, used = expected.get(ingredient, 0.0), actual.get(ingredient, 0.0)
            allowed = tolerance(want)
            record['ingredients'].append({'product_id': ingredient, 'product': product_name(ingredient),
                                          'expected_lb': round(want, 4), 'actual_lb': round(used, 4)})
            if used > EPS:
                expected_ratio[(tx['id'], ingredient)] = want / used
            common = dict(product_id=ingredient, transaction_id=tx['id'], expected=want, actual=used,
                          tolerance_lb=allowed, batches=round(batches, 4), make_product=product_name(out_product))
            name = product_name(ingredient)
            if want <= EPS and used > EPS:
                flag('MAKE_UNLISTED_INGREDIENT', ALARM,
                     f'{label} used {lb(used)} of {name}, which is not in the recipe.', **common)
            elif abs(used - want) <= allowed + EPS:
                continue
            elif used <= EPS:
                flag('MAKE_INGREDIENT_MISSING', ALARM,
                     f'{label} used no {name}. The recipe calls for {lb(want)} and no substitution was recorded.',
                     **common)
            else:
                flag('MAKE_OFF_RECIPE', ALARM,
                     f'{label} used {lb(used)} of {name}. The recipe calls for {lb(want)} '
                     f'({lb(used - want, signed=True)}, {(used - want) / want * 100:+.1f}%).', **common)

    # --- movement per lot --------------------------------------------------
    book_move, expected_move, gross = defaultdict(float), defaultdict(float), defaultdict(float)
    moved = set()
    for tx_id, tx in transactions.items():
        if tx['in_window']:
            for line in after[tx_id]:           # empty when voided inside the window
                quantity = line['quantity_lb']
                ratio = expected_ratio.get((tx_id, line['product_id'])) if quantity < 0 else None
                want = quantity * ratio if ratio is not None else quantity
                book_move[line['lot_id']] += quantity
                expected_move[line['lot_id']] += want
                gross[line['lot_id']] += abs(want)
                moved.add(line['lot_id'])
        else:                                   # an earlier entry voided, restored or amended in the window
            delta = defaultdict(float)
            for line in after[tx_id]:
                delta[line['lot_id']] += line['quantity_lb']
            for line in before[tx_id]:
                delta[line['lot_id']] -= line['quantity_lb']
            for lot_id, change in delta.items():
                if abs(change) > EPS:
                    book_move[lot_id] += change
                    expected_move[lot_id] += change
                    gross[lot_id] += abs(change)
                    moved.add(lot_id)

    # --- balance roll-forward ----------------------------------------------
    ending = dict(map(tuple, data['ending']))
    opening = None if data['opening'] is None else dict(map(tuple, data['opening']))
    checked = set(moved)
    if opening is not None:
        checked |= {lot for lot in set(opening) | set(ending)
                    if abs(ending.get(lot, 0.0) - opening.get(lot, 0.0)) > EPS}
        for lot_id in sorted(checked):
            start_lb, end_lb = opening.get(lot_id, 0.0), ending.get(lot_id, 0.0)
            want = start_lb + expected_move.get(lot_id, 0.0)
            allowed = tolerance(gross.get(lot_id, 0.0))
            if abs(end_lb - want) <= allowed + EPS:
                continue
            unexplained = end_lb - (start_lb + book_move.get(lot_id, 0.0))
            recipe_gap = book_move.get(lot_id, 0.0) - expected_move.get(lot_id, 0.0)
            if abs(unexplained) > EPS:
                why = (f'{lb(unexplained, signed=True)} of the change is not explained by any entry or void '
                       'made in this window. Something changed the books outside the normal entries.')
            else:
                why = 'The entries add up; the gap is a make that used a different amount than its recipe.'
            flag('BALANCE_DIFFERENCE', ALARM,
                 f'{lot_label(lot_id)}: expected to end at {lb(want)}, the books show {lb(end_lb)} '
                 f'({lb(end_lb - want, signed=True)}; tolerance {lb(allowed)}). {why}',
                 lot_id=lot_id, expected=want, actual=end_lb, tolerance_lb=allowed,
                 opening_lb=round(start_lb, 4), entered_movement_lb=round(book_move.get(lot_id, 0.0), 4),
                 expected_movement_lb=round(expected_move.get(lot_id, 0.0), 4),
                 unexplained_lb=round(unexplained, 4), recipe_gap_lb=round(recipe_gap, 4))

    # --- pack provenance (per batch lot packed from in the window) ---------
    last_pack, packs_today, packed_today = {}, defaultdict(list), defaultdict(float)
    for tx_id in sorted(transactions):
        tx = transactions[tx_id]
        if tx['type'] == 'pack' and tx['in_window'] and tx['posted_after']:
            for line in after[tx_id]:
                if line['quantity_lb'] < 0:
                    last_pack[line['lot_id']] = tx_id      # ids ascend: the pack that took the lot furthest
                    if tx_id not in packs_today[line['lot_id']]:
                        packs_today[line['lot_id']].append(tx_id)
                    packed_today[line['lot_id']] += -line['quantity_lb']
    for history in data['lot_histories']:
        lot_id = history['lot_id']
        lot = lots.get(lot_id)
        if lot_id not in last_pack or not lot or products.get(lot['product_id'], {}).get('type') != 'batch':
            continue
        made, packed = history['made_lb'], history['packed_lb']
        common = dict(lot_id=lot_id, transaction_id=last_pack[lot_id], expected=made, actual=packed,
                      packed_in_window_lb=round(packed_today[lot_id], 4),
                      used_in_makes_lb=round(history['used_in_makes_lb'], 4),
                      adjustments=history['adjustments'])
        opener = (' and '.join(f'TX{tx_id}' for tx_id in packs_today[lot_id])
                  + f' packed {lb(packed_today[lot_id])} from {lot_label(lot_id)}')
        if made <= EPS:
            flag('PACK_NO_MAKE', ALARM, f'{opener}, but no make was ever recorded for that lot.', **common)
            continue
        over = packed - made
        if over <= EPS:
            continue
        allowed = tolerance(made)
        totals = f'{lb(made)} made, {lb(packed)} packed ({lb(over, signed=True)}, {over / made * 100:.1f}%)'
        if 'coconut' in product_name(lot['product_id']).lower() and over <= COCONUT_YIELD_PCT * made + EPS:
            flag('COCONUT_YIELD', YIELD, f'{lot_label(lot_id)}: {totals}.', tolerance_lb=allowed, **common)
        elif over > allowed + EPS:
            added = [a for a in history['adjustments'] if a['quantity_lb'] > 0]
            extra = ''
            if added:
                extra = ' Stock was added to this lot by ' + '; '.join(
                    f'adjustment TX{a["transaction_id"]} ({lb(a["quantity_lb"], signed=True)}, reason '
                    f'"{a["reason_code"] or "none"}", entered by {_person(a["actor_name"], a["operator_id"])})'
                    for a in added) + '.'
            flag('PACK_EXCEEDS_MAKE', ALARM,
                 f'{opener}. Only {lb(made)} of that lot was ever made; {lb(packed)} has now been packed out '
                 f'({lb(over, signed=True)}, {over / made * 100:.1f}%).{extra}', tolerance_lb=allowed, **common)

    # --- always listed -----------------------------------------------------
    for lot_id in sorted(ending):
        if ending[lot_id] < -EPS:
            flag('NEGATIVE_LOT', ALARM, f'{lot_label(lot_id)} is at {lb(ending[lot_id])}.',
                 lot_id=lot_id, actual=ending[lot_id], moved_in_window=lot_id in moved)

    def movement_text(lines):
        return '; '.join(f'{lb(line["quantity_lb"], signed=True)} {lot_label(line["lot_id"])}' for line in lines)

    for tx in data['transactions']:
        if not tx['in_window']:
            continue
        if not tx['posted_after']:
            voids = [c for c in data['corrections'] if c['transaction_id'] == tx['id']]
            last = voids[-1] if voids else None
            tail = (f' and voided {_when(last["created_at"], with_date=False)} by '
                    f'{_person(last["actor_name"], last["operator_id"])}: "{last["reason"]}"') if last else \
                ' and is not posted'
            gone = [line for line in data['voided_lines'] if line['transaction_id'] == tx['id']]
            flag('VOID_SAME_DAY', INFO,
                 f'TX{tx["id"]} ({tx["type"]}: {movement_text(gone) or "no lines"}) was entered '
                 f'{_when(tx["created_at"])}{tail}. No effect on balances.',
                 transaction_id=tx['id'], lines=gone)
            continue
        if tx['business_date'] != tx['entered_plant_date']:
            flag('LATE_ENTRY', INFO,
                 f'TX{tx["id"]} ({tx["type"]}) was entered {_when(tx["created_at"])} for business date '
                 f'{date.fromisoformat(tx["business_date"]):%b} {date.fromisoformat(tx["business_date"]).day} '
                 f'by {_person(tx["actor_name"], tx["operator_id"])}.',
                 transaction_id=tx['id'], business_date=tx['business_date'])
        if tx['type'] == 'adjust':
            reason = (tx['reason_code'] or '').strip().lower()
            no_reason, no_person = reason in ('', 'unknown'), tx['actor_id'] is None
            if no_reason or no_person:
                problems = ' and '.join(filter(None, [
                    f'reason "{reason or "none"}"' if no_reason else '',
                    'no named person (shared key)' if no_person else '']))
                reason_text = '' if no_reason else f' Reason given: "{reason}".'
                person_text = '' if no_person else f' Entered by {tx["actor_name"]}.'
                flag('UNEXPLAINED_ADJUSTMENT', REVIEW,
                     f'TX{tx["id"]} at {_when(tx["created_at"], with_date=False)}: {movement_text(after[tx["id"]])} '
                     f'with {problems}.{reason_text}{person_text}',
                     transaction_id=tx['id'], lot_id=after[tx['id']][0]['lot_id'] if after[tx['id']] else None,
                     actual=sum(line['quantity_lb'] for line in after[tx['id']]),
                     reason_code=tx['reason_code'], operator_id=tx['operator_id'], notes=tx['notes'],
                     no_reason=no_reason, no_named_person=no_person)
    for correction in data['corrections']:
        tx = transactions.get(correction['transaction_id'])
        if tx is None or tx['in_window']:
            continue
        verb = {'void': 'voided', 'restore': 'restored', 'amend': 'amended'}[correction['event_type']]
        flag('VOID_OF_EARLIER_ENTRY', INFO,
             f'TX{tx["id"]} ({tx["type"]}, business date {date.fromisoformat(tx["business_date"]):%b} '
             f'{date.fromisoformat(tx["business_date"]).day}) was {verb} {_when(correction["created_at"])} by '
             f'{_person(correction["actor_name"], correction["operator_id"])}: "{correction["reason"]}". '
             'The balance change is included in the roll-forward.',
             transaction_id=tx['id'], event_type=correction['event_type'], business_date=tx['business_date'])

    order = {ALARM: 0, REVIEW: 1, YIELD: 2, INFO: 3}
    flags.sort(key=lambda f: (order[f['severity']], f['kind'], f['transaction_id'] or 0, f['lot_id'] or 0))
    attention = sum(1 for f in flags if f['severity'] in (ALARM, REVIEW))
    return {
        'plant_date': data['plant_date'], 'window_start': data['window_start'], 'window_end': data['window_end'],
        'mode': data['mode'], 'balance_checked': opening is not None,
        'entries_checked': sum(1 for t in data['transactions'] if t['in_window'] and t['posted_after']),
        'makes_checked': sum(1 for m in makes if m['checked']), 'lots_checked': len(checked),
        'result': 'attention' if attention else 'all_clear', 'attention_count': attention,
        'flags': flags, 'makes': makes,
    }


# ---------------------------------------------------------------------------
# The owner email: plain language, flags first.
# ---------------------------------------------------------------------------

_SECTIONS = (
    ('OVER TOLERANCE', lambda f: f['severity'] == ALARM and f['kind'] != 'NEGATIVE_LOT',
     'Nothing over tolerance.'),
    ('LOTS BELOW ZERO', lambda f: f['kind'] == 'NEGATIVE_LOT', 'No lot is below zero.'),
    ('ADJUSTMENTS WITH NO REASON OR NO NAMED PERSON', lambda f: f['severity'] == REVIEW, 'None.'),
    ('COCONUT YIELD (packed slightly more than made, within 2%; not an alarm)',
     lambda f: f['severity'] == YIELD, 'None.'),
    ('FOR INFORMATION: VOIDS AND LATE ENTRIES', lambda f: f['severity'] == INFO, 'None.'),
)


def _plural(count, word):
    return f'{count} {word}{"" if count == 1 else "s"}'


def render_email(analysis):
    """{'subject', 'text', 'html'} for one analysis. Sent every night, all-clear included."""
    day = date.fromisoformat(analysis['plant_date'])
    flags = analysis['flags']
    over = sum(1 for f in flags if f['severity'] == ALARM and f['kind'] != 'NEGATIVE_LOT')
    negative = sum(1 for f in flags if f['kind'] == 'NEGATIVE_LOT')
    review = sum(1 for f in flags if f['severity'] == REVIEW)
    if analysis['result'] == 'all_clear':
        headline = 'ALL CLEAR. Nothing over tolerance, no lot below zero, no unexplained adjustment.'
        subject_tail = 'all clear'
    else:
        parts = [_plural(over, 'item') + ' over tolerance' if over else '',
                 _plural(negative, 'lot') + ' below zero' if negative else '',
                 _plural(review, 'adjustment') + ' to review' if review else '']
        subject_tail = ', '.join(p for p in parts if p)
        headline = f'NEEDS ATTENTION: {subject_tail}.'
    subject = f'Factory Ledger daily check, {day:%a %b} {day.day}: {subject_tail}'
    checked = (f'Checked {_plural(analysis["entries_checked"], "entry").replace("entrys", "entries")}, '
               f'{_plural(analysis["makes_checked"], "make")} and {_plural(analysis["lots_checked"], "lot")}. '
               f'Entries made {_when(analysis["window_start"])} to {_when(analysis["window_end"])} Eastern.')
    notes = []
    if not analysis['balance_checked']:
        notes.append('First run: a starting snapshot of every lot was saved. The balance roll-forward '
                     'begins with the next run.')
    lines = [f'Factory Ledger daily check for {day:%A, %B} {day.day}, {day.year}', '', headline, checked, *notes]
    blocks = []
    for title, belongs, empty in _SECTIONS:
        items = [f['detail']['message'] for f in flags if belongs(f)]
        blocks.append((f'{title} ({len(items)})', items, empty))
        lines += ['', f'{title} ({len(items)})'] + ([f'  {n}. {m}' for n, m in enumerate(items, 1)] or [f'  {empty}'])
    footer = [
        'How to read this',
        '  Tolerance is the greater of 5 lb or 2% of the pounds that moved in that lot that day.',
        '  This check compares the books with themselves. It cannot see the shelf; only a count can.',
        '  It is sent every night, including when all is clear. No email means the job did not run.',
    ]
    lines += [''] + footer
    esc = html_lib.escape
    html = [f'<h2 style="font:600 18px sans-serif">{esc(lines[0])}</h2>',
            f'<p style="font:600 15px sans-serif">{esc(headline)}</p>',
            f'<p style="font:14px sans-serif">{esc(checked)}</p>'] + [
        f'<p style="font:14px sans-serif">{esc(n)}</p>' for n in notes]
    for title, items, empty in blocks:
        html.append(f'<h3 style="font:600 14px sans-serif;margin-top:20px">{esc(title)}</h3>')
        html.append('<ol style="font:14px sans-serif">' + ''.join(f'<li>{esc(m)}</li>' for m in items) + '</ol>'
                    if items else f'<p style="font:14px sans-serif">{esc(empty)}</p>')
    html.append('<p style="font:12px sans-serif;color:#555;margin-top:24px">'
                + '<br>'.join(esc(line.strip()) for line in footer[1:]) + '</p>')
    return {'subject': subject, 'text': '\n'.join(lines) + '\n', 'html': '\n'.join(html)}


def email_enabled(env=None):
    env = os.environ if env is None else env
    return env.get('RECONCILIATION_EMAIL_ENABLED', '').strip().lower() in ('1', 'true', 'yes', 'on')


def send_email(email, env=None):
    """One HTTPS POST to Resend. Never raises and never returns the API key.

    Returns {'status': 'sent' | 'failed', 'to', 'provider_id', 'error'}.
    """
    env = os.environ if env is None else env
    recipients = [a.strip() for a in env.get('ALERT_EMAIL_TO', '').split(',') if a.strip()]
    outcome = {'status': 'failed', 'to': ', '.join(recipients) or None, 'provider_id': None, 'error': None}
    key = env.get('RESEND_API_KEY', '').strip()
    if not recipients or not key:
        outcome['error'] = 'ALERT_EMAIL_TO or RESEND_API_KEY is not set'
        return outcome
    try:
        response = httpx.post(RESEND_URL, timeout=20.0, headers={'Authorization': f'Bearer {key}'}, json={
            'from': env.get('RECONCILIATION_EMAIL_FROM', '').strip() or DEFAULT_EMAIL_FROM,
            'to': recipients, 'subject': email['subject'], 'text': email['text'], 'html': email['html']})
    except httpx.HTTPError as exc:
        outcome['error'] = f'Could not reach the email service ({type(exc).__name__})'
        return outcome
    try:
        body = response.json()
    except ValueError:
        body = {}
    if 200 <= response.status_code < 300:
        outcome.update(status='sent', provider_id=str(body.get('id') or '') or None)
    else:
        outcome['error'] = f'Email service answered {response.status_code}: {str(body.get("message") or "")[:200]}'
    return outcome


# ---------------------------------------------------------------------------
# The nightly sequence.
# ---------------------------------------------------------------------------

def _clock(cur):
    cur.execute('SELECT clock_timestamp() AS now')
    row = cur.fetchone()
    return row['now'] if isinstance(row, dict) else row[0]


def _start_of_plant_day(day):
    return datetime.combine(day, time.min, tzinfo=PLANT_TIMEZONE)


def replay(cur, day):
    """Check one past plant day from timestamps alone. Reads only; saves nothing."""
    analysis = analyse(load_window(cur, _start_of_plant_day(day), _start_of_plant_day(day + timedelta(days=1)),
                                   reconstruct=True))
    return {'analysis': analysis, 'email': render_email(analysis), 'run_id': None, 'email_status': 'not_attempted'}


def save_run(cur, *, started_at, baseline_id, data, analysis, email, delivery):
    """The ONLY writes in this module: three INSERTs into the 094 tables."""
    cur.execute("SELECT nextval(pg_get_serial_sequence('reconciliation_runs', 'id')) AS id")
    row = cur.fetchone()
    run_id = row['id'] if isinstance(row, dict) else row[0]
    lot_product = {lot['id']: lot['product_id'] for lot in data['lots']}
    execute_values(cur, """INSERT INTO inventory_snapshots (taken_at, plant_date, job_run_id, lot_id, product_id, quantity_lb)
                           VALUES %s""",
                   [(data['window_end'], data['plant_date'], run_id, lot, lot_product[lot], quantity)
                    for lot, quantity in data['ending']])
    execute_values(cur, """INSERT INTO reconciliation_flags (run_id, kind, severity, product_id, lot_id, transaction_id,
                                                             expected_lb, actual_lb, difference_lb, tolerance_lb, detail)
                           VALUES %s""",
                   [(run_id, f['kind'], f['severity'], f['product_id'], f['lot_id'], f['transaction_id'],
                     f['expected_lb'], f['actual_lb'], f['difference_lb'], f['tolerance_lb'], Json(f['detail']))
                    for f in analysis['flags']])
    summary = {key: analysis[key] for key in ('mode', 'makes')}
    summary.update(email_subject=email['subject'], email_text=email['text'],
                   snapshot_lots=len(data['ending']),
                   tolerance={'floor_lb': TOLERANCE_FLOOR_LB, 'pct_of_day_movement': TOLERANCE_PCT})
    cur.execute("""INSERT INTO reconciliation_runs
                       (id, started_at, plant_date, previous_run_id, window_start, window_end, job_status, result,
                        balance_checked, entries_checked, makes_checked, lots_checked, flag_count, attention_count,
                        email_status, email_to, email_sent_at, email_error, email_provider_id, summary)
                   VALUES (%s, %s, %s, %s, %s, %s, 'completed', %s, %s, %s, %s, %s, %s, %s, %s, %s,
                           CASE WHEN %s = 'sent' THEN clock_timestamp() END, %s, %s, %s)""",
                (run_id, started_at, data['plant_date'], baseline_id, data['window_start'], data['window_end'],
                 analysis['result'], analysis['balance_checked'], analysis['entries_checked'],
                 analysis['makes_checked'], analysis['lots_checked'], len(analysis['flags']),
                 analysis['attention_count'], delivery['status'], delivery.get('to'), delivery['status'],
                 delivery.get('error'), delivery.get('provider_id'), Json(summary)))
    return run_id


def run_job(cur, *, now=None, dry_run=False, sender=None, env=None, force_email=False):
    """Snapshot, check the business day, save the flags, email, record the send.

    The opening balance is the most recent snapshot from an EARLIER plant day,
    whatever its date, so a missed night or a weekend needs no special handling
    and running twice in one night never moves the baseline. A second run for
    the same plant day saves its own snapshot and flags but does not email again.

    dry_run=True reads the same data and returns the same email, and writes
    nothing. The caller owns the transaction (commit / rollback).
    """
    env = os.environ if env is None else env
    started_at = _clock(cur)
    end = now or started_at
    plant_day = end.astimezone(PLANT_TIMEZONE).date() - timedelta(days=1)
    baseline = _rows(cur, """SELECT id, window_end FROM reconciliation_runs
                             WHERE job_status = 'completed' AND plant_date < %(day)s
                             ORDER BY window_end DESC, id DESC LIMIT 1""", {'day': plant_day})
    baseline_id = baseline[0]['id'] if baseline else None
    start = baseline[0]['window_end'] if baseline else _start_of_plant_day(plant_day)
    data = load_window(cur, start, end, opening_run_id=baseline_id)
    analysis = analyse(data)
    email = render_email(analysis)
    outcome = {'analysis': analysis, 'email': email, 'run_id': None, 'email_status': 'not_attempted'}
    if dry_run:
        return outcome
    if not email_enabled(env):
        delivery = {'status': 'disabled'}
    else:
        cur.execute("SELECT 1 FROM reconciliation_runs WHERE plant_date = %s AND email_status = 'sent' LIMIT 1",
                    (plant_day,))
        if cur.fetchone() is not None and not force_email:
            delivery = {'status': 'already_sent'}
        else:
            delivery = (sender or send_email)(email, env)
    outcome['run_id'] = save_run(cur, started_at=started_at, baseline_id=baseline_id, data=data,
                                 analysis=analysis, email=email, delivery=delivery)
    outcome['email_status'], outcome['email_error'] = delivery['status'], delivery.get('error')
    return outcome


def record_failed_run(cur, error_name):
    """Best-effort note that tonight's job failed (job_status is separate from the result)."""
    cur.execute("""INSERT INTO reconciliation_runs (started_at, plant_date, job_status, job_error, email_status)
                   VALUES (clock_timestamp(), ((clock_timestamp() AT TIME ZONE 'America/New_York')::date - 1),
                           'failed', %s, 'not_attempted')""", (error_name[:200],))


# ---------------------------------------------------------------------------
# Three read-only endpoints. Owner and office (and the master key); never in
# either GPT schema (include_in_schema=False; the office schema is at 30/30).
# ---------------------------------------------------------------------------

def _fail(status, code, message):
    raise HTTPException(status_code=status, detail={'error_code': code, 'message': message})


def _require(api, request, *, owner_only=False):
    role = permissions.role_of(api._actor_identity(request))
    allowed = {permissions.OWNER, permissions.SELECTED_OWNER, permissions.LEGACY_LEDGER}
    if not owner_only:
        allowed.add(permissions.OFFICE)
    if role not in allowed:
        _fail(403, 'ROLE_NOT_ALLOWED',
              'Only Michael can run the preview.' if owner_only
              else 'The reconciliation results are for the owner and the office.')


def _ready(cur):
    if not migration_applied(cur):
        _fail(503, 'RECONCILIATION_UNAVAILABLE', 'The daily reconciliation is not switched on yet.')


_RUN_COLUMNS = """id, started_at, finished_at, plant_date, previous_run_id, window_start, window_end, job_status,
                  job_error, result, balance_checked, entries_checked, makes_checked, lots_checked, flag_count,
                  attention_count, email_status, email_to, email_sent_at, email_error"""


def register_routes(app, api):
    @app.get('/verification/reconciliation/runs', include_in_schema=False)
    def reconciliation_runs(request: Request, limit: int = 30, _: bool = Depends(api.verify_api_key)):
        _require(api, request)
        with api.get_transaction() as cur:
            _ready(cur)
            cur.execute(f'SELECT {_RUN_COLUMNS} FROM reconciliation_runs ORDER BY id DESC LIMIT %s',
                        (max(1, min(limit, 200)),))
            return {'runs': [dict(row) for row in cur.fetchall()]}

    @app.get('/verification/reconciliation/runs/{run_id}', include_in_schema=False)
    def reconciliation_run(run_id: int, request: Request, _: bool = Depends(api.verify_api_key)):
        _require(api, request)
        with api.get_transaction() as cur:
            _ready(cur)
            cur.execute(f'SELECT {_RUN_COLUMNS}, summary FROM reconciliation_runs WHERE id = %s', (run_id,))
            run = cur.fetchone()
            if run is None:
                _fail(404, 'RUN_NOT_FOUND', f'No reconciliation run {run_id}.')
            cur.execute("""SELECT id, kind, severity, product_id, lot_id, transaction_id, expected_lb, actual_lb,
                                  difference_lb, tolerance_lb, detail, created_at
                           FROM reconciliation_flags WHERE run_id = %s ORDER BY id""", (run_id,))
            return {'run': dict(run), 'flags': [dict(row) for row in cur.fetchall()]}

    @app.get('/verification/reconciliation/preview', include_in_schema=False)
    def reconciliation_preview(request: Request, _: bool = Depends(api.verify_api_key)):
        """Tonight's check against the latest snapshot, as of now. Saves nothing, sends nothing."""
        _require(api, request, owner_only=True)
        with api.get_transaction() as cur:
            cur.execute('SET TRANSACTION READ ONLY')    # this transaction only; never a session setting
            _ready(cur)
            outcome = run_job(cur, dry_run=True)
            return {'saved': False, 'analysis': outcome['analysis'], 'email': outcome['email']}
