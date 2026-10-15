"""A9: end-of-shift summary and confirmation (design §7.3; frozen contract
docs/contracts/shift-summary.md).

`GET /reports/shift-summary` renders what the FL receipts of one plant day say
happened, for one person (or, for the two shared keys with no `actor`, for the
whole plant, read-only). `POST /reports/shift-summary/confirm` records that the
person looked at exactly that document (`summary_hash`, §1.3) and whether it
matched the floor. Nothing here computes from raw ledger lines: receipts come
from `shift_receipts.load` (the same effective tables/evidence as `GET /receipts`),
lot balances from the posted-only effective lines in that one-query snapshot, reason labels from
`correction_reasons`.

Interface neutrality (§1.8): the D3-lite page, the FL Assistant and the MCP
`endOfShift` tool all render this JSON; every `*_en`/`*_es` pair carries an
un-suffixed `lang` copy so a client can print the document in one language
without composing text.

Boundaries kept deliberately (named here so the review can check them):
  * "linked MK/PK" for planned production runs: `production_runs` has no
    ticket/transaction column, so the link FL can state is product + plant day
    + action — a run still `planned` / `in_progress` is "entered" when a
    committed MK (bake/coconut/other) or PK (pack) receipt for its product
    happened on its planned date (any actor; one receipt covers one run, lowest
    run id first). Runs without such a receipt are listed (contract §2.3).
  * `VOIDED` rows need `void` tickets (`VD-` receipts); until that action has
    tickets the section is present and empty, and a receipt whose transaction
    was voided through the legacy route stays in its own section with its
    ORIGINAL quantities/detail, `effective_status: voided` and flag `VOIDED`
    (`voided_by_receipt_number: null` — no receipt names the void). Voided
    lines never enter LOT BALANCES TOUCHED TODAY.
  * Receipts of the metadata ticket actions (orders `ORD-`, expected receipts
    `ER-`, lot moves `LOT-`) are not ledger postings and no §2 section lists
    them; they are reported under the additive top-level `other_receipts`
    block so the day's `GET /receipts` set is fully accounted for. They are
    never called adjustments and never enter `summary_hash` (§1.3).
  * `INVOICE_PENDING` waits for A8; A12 kosher evidence and holds follow §8.
"""
import json
from datetime import date, datetime, time, timedelta
from typing import Literal, Optional

from fastapi import Depends, Request
from psycopg2.extras import Json
from pydantic import BaseModel, Field

import permissions
import write_tickets as wt
import kosher_tiers
import shift_receipts

ROUTES = frozenset({('GET', '/reports/shift-summary'), ('POST', '/reports/shift-summary/confirm')})

# Fixed order, keyed on `key` (contract §2). Clients never key on position.
SECTIONS = (
    ('received', 'RECEIVED', 'RECIBIDO', 'none', 'ninguno'),
    ('made', 'MADE', 'PRODUCIDO', 'none', 'ninguno'),
    ('packed', 'PACKED', 'EMPACADO', 'none', 'ninguno'),
    ('shipped', 'SHIPPED', 'ENVIADO', 'none', 'ninguno'),
    ('adjusted', 'ADJUSTED', 'AJUSTADO', 'none', 'ninguno'),
    ('voided', 'VOIDED', 'ANULADO', 'none', 'ninguno'),
    ('not_entered_yet', 'NOT ENTERED YET?', '¿FALTA REGISTRAR?', 'nothing expected', 'nada pendiente'),
    ('lot_balances_touched', 'LOT BALANCES TOUCHED TODAY', 'LOTES MOVIDOS HOY', 'none', 'ninguno'),
    ('open_exceptions', 'OPEN EXCEPTIONS (yours)', 'EXCEPCIONES ABIERTAS (tuyas)', 'none', 'ninguna'),
    ('late_entries', 'LATE ENTRIES TODAY', 'ENTRADAS TARDÍAS HOY', 'none', 'ninguna'),
)
RECEIPT_SECTIONS = ('received', 'made', 'packed', 'shipped', 'adjusted', 'voided', 'late_entries')
ACTION_SECTION = {'receive': 'received', 'make': 'made', 'pack': 'packed', 'adjust': 'adjusted', 'found': 'adjusted',
                  'rename_lot': 'adjusted', 'update_supplier_lot': 'adjusted', 'void': 'voided',
                  **{a: 'shipped' for a in wt.SHIP_ACTIONS}}
HASH_FIELDS = ('receipt_number', 'action', 'status', 'effective_status', 'happened_at', 'entered_at', 'late_entry',
               'detail', 'transaction_ids', 'kosher_tier', 'kosher_attested_by_actor_id',
               'kosher_attested_at', 'transaction_kosher_downgrade')
# Metadata tickets (A7 orders, A7 expected receipts, A5 lot moves): receipts, not ledger postings.
OTHER_KIND = {**{a: 'order' for a in wt.ORDER_ACTIONS}, **{a: 'expected_receipt' for a in wt.EXPECTED_RECEIPT_ACTIONS},
              'move_lot': 'lot_move'}
_OTHER_LABELS = {
    'create_order': ('order created', 'pedido creado'), 'add_order_lines': ('order lines added', 'líneas añadidas al pedido'),
    'update_order_line': ('order line updated', 'línea del pedido actualizada'),
    'cancel_order_line': ('order line cancelled', 'línea del pedido cancelada'),
    'update_order_header': ('order header updated', 'encabezado del pedido actualizado'),
    'update_order_status': ('order status updated', 'estado del pedido actualizado'),
    'mark_order_ready': ('order marked ready', 'pedido marcado listo'), 'cancel_order': ('order cancelled', 'pedido cancelado'),
    'close_order': ('order closed', 'pedido cerrado'), 'reopen_order': ('order reopened', 'pedido reabierto'),
    'create_expected_receipt': ('expected receipt created', 'recepción esperada creada'),
    'update_expected_receipt': ('expected receipt updated', 'recepción esperada actualizada'),
    'cancel_expected_receipt': ('expected receipt cancelled', 'recepción esperada cancelada'),
    'move_lot': ('lot moved', 'lote movido'),
}

_DAYS_ES = ('lun', 'mar', 'mié', 'jue', 'vie', 'sáb', 'dom')
_MONTHS_ES = ('ene', 'feb', 'mar', 'abr', 'may', 'jun', 'jul', 'ago', 'sep', 'oct', 'nov', 'dic')
_ACTION_ES = {'make': 'Lote de producción', 'pack': 'Empaque'}

# Exception kinds → (singular_en, plural_en, singular_es, plural_es, line label en, line label es)
_KINDS = {
    'SHORTAGE': ('shortage', 'shortages', 'faltante', 'faltantes', 'SHORT', 'FALTAN'),
    'UNIDENTIFIED_LOT': ('unidentified lot', 'unidentified lots', 'lote sin identificar', 'lotes sin identificar',
                         'UNIDENTIFIED', 'SIN IDENTIFICAR'),
    'LARGE_CORRECTION': ('large correction', 'large corrections', 'corrección grande', 'correcciones grandes',
                         'LARGE CORRECTION', 'CORRECCIÓN GRANDE'),
    'LATE_ENTRY': ('late entry', 'late entries', 'entrada tardía', 'entradas tardías', 'LATE ENTRY', 'ENTRADA TARDÍA'),
    'BOL_PHOTO_MISSING': ('missing BOL photo', 'missing BOL photos', 'foto de BOL faltante', 'fotos de BOL faltantes',
                          'BOL PHOTO MISSING', 'FALTA FOTO DEL BOL'),
    'SHIPMENT_PROOF_MISSING': ('missing shipment proof', 'missing shipment proofs', 'comprobante de envío faltante',
                               'comprobantes de envío faltantes', 'PROOF MISSING', 'FALTA COMPROBANTE'),
    'NEGATIVE_BALANCE': ('negative balance', 'negative balances', 'saldo negativo', 'saldos negativos',
                         'NEGATIVE BALANCE', 'SALDO NEGATIVO'),
    'POSSIBLE_DUPLICATE_ACK': ('acknowledged duplicate', 'acknowledged duplicates', 'duplicado confirmado',
                               'duplicados confirmados', 'POSSIBLE DUPLICATE', 'POSIBLE DUPLICADO'),
    'UNSHIPPED_PAST_DUE': ('order past due', 'orders past due', 'pedido vencido', 'pedidos vencidos',
                           'PAST DUE, UNSHIPPED', 'VENCIDO SIN ENVIAR'),
    'SUNSHINE_INVOICE_PENDING': ('invoice pending', 'invoices pending', 'factura pendiente', 'facturas pendientes',
                                 'INVOICE PENDING', 'FACTURA PENDIENTE'),
    'SHIFT_DISCREPANCY': ('shift discrepancy', 'shift discrepancies', 'discrepancia de turno',
                          'discrepancias de turno', 'SHIFT DISCREPANCY', 'DISCREPANCIA DE TURNO'),
    'KOSHER_ATTESTATION': ('kosher attestation pending', 'kosher attestations pending', 'atestación kosher pendiente',
                           'atestaciones kosher pendientes', 'KOSHER ATTESTATION', 'ATESTACIÓN KOSHER'),
    'PRE_MAKE_ADJUST': ('pre-make adjust', 'pre-make adjusts', 'ajuste previo a producción',
                        'ajustes previos a producción', 'PRE-MAKE ADJUST', 'AJUSTE PREVIO'),
}


def fail(http_status, code, message, message_es=None, **extra):
    wt.fail(http_status, code, message, **({'message_es': message_es} if message_es else {}), **extra)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------
def fmt_lb(value, signed=False):
    if value is None:
        return '?'
    value = float(value)
    whole = abs(value - round(value)) < 1e-9
    text = f'{abs(value):,.0f}' if whole else f'{abs(value):,.2f}'.rstrip('0').rstrip('.')
    if value < 0:
        return '−' + text
    return ('+' + text) if signed and value > 0 else text


def fmt_cases(value):
    value = float(value)
    return f'{value:,.0f}' if abs(value - round(value)) < 1e-9 else f'{value:,.2f}'.rstrip('0').rstrip('.')


def short_date(d, lang):
    """`Oct 9` / `9 oct` (contract mock)."""
    if isinstance(d, datetime):
        d = d.astimezone(permissions.PLANT_TIMEZONE).date()
    return f'{d:%b} {d.day}' if lang == 'en' else f'{d.day} {_MONTHS_ES[d.month - 1]}'


def title_date(d, lang):
    return f'{d:%a %b} {d.day}' if lang == 'en' else f'{_DAYS_ES[d.weekday()]} {d.day} {_MONTHS_ES[d.month - 1]}'


def plant(dt, api):
    return dt.astimezone(api.PLANT_TIMEZONE) if dt is not None else None


def iso(dt, api):
    return plant(dt, api).isoformat() if dt is not None else None


def pair(en, es, key):
    return {f'{key}_en': en, f'{key}_es': es}


def localise(node, lang):
    """Add the un-suffixed `lang` copy next to every `*_en`/`*_es` pair, recursively (§1.1)."""
    if isinstance(node, dict):
        for key in list(node):
            if key.endswith('_en') and key[:-3] + '_es' in node:
                node[key[:-3]] = node[f'{key[:-3]}_{lang}']
        for value in node.values():
            localise(value, lang)
    elif isinstance(node, list):
        for value in node:
            localise(value, lang)
    return node


# ---------------------------------------------------------------------------
# Request-local catalog, preloaded with the receipt snapshot; fallback for other references
# ---------------------------------------------------------------------------
class Catalog:
    def __init__(self, cur):
        self.cur = cur
        self._products, self._lots, self._reasons, self._actors = {}, {}, {}, {}

    def product(self, product_id):
        if product_id is None:
            return None
        if product_id not in self._products:
            self.cur.execute('SELECT id,name,odoo_code,type FROM products WHERE id=%s', (product_id,))
            row = self.cur.fetchone()
            self._products[product_id] = dict(row) if row else {'id': product_id, 'name': f'product {product_id}',
                                                                   'odoo_code': None, 'type': None}
        return dict(self._products[product_id])

    def product_ref(self, product_id):
        p = self.product(product_id)
        return {'id': p['id'], 'name': p['name']} if p else None

    def lot(self, lot_id):
        if lot_id is None:
            return None
        if lot_id not in self._lots:
            self.cur.execute('SELECT id,product_id,lot_code,supplier_lot_code,identity_status,identify_by FROM lots WHERE id=%s',
                             (lot_id,))
            row = self.cur.fetchone()
            self._lots[lot_id] = dict(row) if row else {'id': lot_id, 'product_id': None, 'lot_code': f'lot {lot_id}',
                                                          'supplier_lot_code': None, 'identity_status': None, 'identify_by': None}
        return dict(self._lots[lot_id])

    def lot_ref(self, lot_id):
        lot = self.lot(lot_id)
        return {'id': lot['id'], 'lot_code': lot['lot_code']} if lot else None

    def reason(self, code):
        """Labels from `correction_reasons` (061) — never hard-coded in a client (contract §6.4)."""
        if code is None:
            return None
        if code not in self._reasons:
            self.cur.execute('SELECT code,label_en,label_es,note_required FROM correction_reasons WHERE code=%s', (code,))
            row = self.cur.fetchone()
            self._reasons[code] = dict(row) if row else {'code': code, 'label_en': code, 'label_es': code, 'note_required': False}
        return dict(self._reasons[code])

    def actor(self, actor_id):
        if actor_id is None:
            return None
        if actor_id not in self._actors:
            self.cur.execute('SELECT id,name,role FROM actors WHERE id=%s', (actor_id,))
            row = self.cur.fetchone()
            self._actors[actor_id] = dict(row) if row else None
        return dict(self._actors[actor_id]) if self._actors[actor_id] else None


# ---------------------------------------------------------------------------
# Receipt rows (contract §2.2)
# ---------------------------------------------------------------------------
def _lines(receipt, sign=None, posted_only=False):
    """Effective lines of the receipt's transactions — the amounts the receipt recorded.
    A voided receipt keeps its original quantities (contract §2.2: it stays in its
    section with flag VOIDED); only LOT BALANCES reads posted lines (`posted_only`)."""
    out = []
    for txn in receipt['transactions']:
        if posted_only and txn['effective_status'] != 'posted':
            continue
        for line in txn['lines']:
            q = float(line['quantity_lb'])
            if sign is None or (sign > 0 and q > 0) or (sign < 0 and q < 0):
                out.append(line)
    return out


def _flag(code, severity, en, es, refs, due_at=None):
    flag = {'code': code, 'severity': severity, **pair(en, es, 'message'), 'refs': refs}
    if due_at is not None:
        flag['due_at'] = due_at
    return flag


def _open_exception(cur, catalog=None, **where):
    if catalog is not None and hasattr(catalog, "_exceptions"):
        return next((r for r in catalog._exceptions if r["status"] in ("open", "escalated")
                     and all(r.get(k) == v for k, v in where.items())), None)
    clauses = ' AND '.join(f'{k}=%s' for k in where)
    cur.execute(f"SELECT id,due_at,status FROM exceptions WHERE {clauses} AND status IN ('open','escalated') ORDER BY id LIMIT 1",
                tuple(where.values()))
    return cur.fetchone()


def receipt_row(api, cur, catalog, number, day, *, receipt=None, ticket=None):
    receipt = receipt if receipt is not None else wt.receipt_detail(api, cur, number)
    if ticket is None:
        cur.execute('SELECT id,action,actor_id,operator_id,key_kind,payload,response,draft,warnings,acknowledged,result_ref,committed_at '
                'FROM write_tickets WHERE receipt_number=%s', (number,))
        ticket = cur.fetchone()
    payload, response, draft = ticket['payload'], ticket['response'] or {}, ticket['draft'] or {}
    action = ticket['action']
    happened, entered = plant(receipt['happened_at'], api), plant(receipt['entered_at'], api)
    posted = [t for t in receipt['transactions'] if t['effective_status'] == 'posted']
    effective = 'posted' if posted or not receipt['transactions'] else 'voided'
    timing = response.get('entry_timing') or {}
    hours_late = timing.get('hours_late')
    if hours_late is None:
        # UTC instants (permissions.elapsed): same-zone local subtraction is off by an hour across a DST edge.
        hours_late = max(round(permissions.elapsed(receipt['happened_at'], receipt['entered_at']).total_seconds() / 3600, 2), 0.0)
    days_late = timing.get('days_late', int(hours_late // 24))
    actor = receipt['actor'] or {}
    row = {
        'receipt_number': number, 'action': action, 'status': receipt['status'], 'effective_status': effective,
        'actor': {'id': actor.get('id'), 'name': actor.get('name') or receipt['operator_id'], 'role': actor.get('role')},
        'client_source': receipt['client_source'],
        'happened_at': happened.isoformat(), 'entered_at': entered.isoformat(),
        'late_entry': bool(receipt['late_entry']), 'days_late': days_late, 'hours_late': hours_late,
        'entered_on_later_day': entered.date() > day,
        'product': None, 'lot': None, 'quantity': None, 'detail': {}, 'flags': [],
        'transaction_ids': [t['id'] for t in receipt['transactions']],
        'transactions': [{'id': t['id'], 'type': t['type'], 'effective_status': t['effective_status']} for t in receipt['transactions']],
        'exceptions_opened': [],
    }
    lot_meta = {lot['id']: lot for lot in receipt['lots']}
    clock = happened.strftime('%H:%M')

    def quantity(lb, cases=None, unit='lb', display_en=None, display_es=None):
        display_en = display_en or f'{fmt_lb(lb)} lb'
        return {'lb': lb, 'cases': cases, 'unit': unit, **pair(display_en, display_es or display_en, 'display')}

    def product_full(product_id):
        p = catalog.product(product_id)
        return {'id': p['id'], 'name': p['name'], 'odoo_code': p.get('odoo_code'), 'type': p.get('type')} if p else None

    def lot_full(lot_id):
        if lot_id is None:
            return None
        lot = catalog.lot(lot_id)
        meta = lot_meta.get(lot_id, {})
        return {'id': lot['id'], 'lot_code': lot['lot_code'], 'supplier_lot_code': lot.get('supplier_lot_code'),
                'identity_status': meta.get('identity_status', lot.get('identity_status'))}

    txn0 = receipt['transactions'][0] if receipt['transactions'] else {}
    if action == 'receive':
        lines = _lines(receipt, +1) or (txn0.get('lines') or [])
        lb = sum(float(l['quantity_lb']) for l in lines)
        lot_id = response.get('lot_id') or (lines[0]['lot_id'] if lines else None)
        row['product'], row['lot'] = product_full(payload['product_id']), lot_full(lot_id)
        cases = payload.get('cases')
        row['quantity'] = quantity(lb, cases, 'cases' if cases else 'lb', f'{fmt_lb(lb)} lb')
        supplier = {'id': txn0.get('supplier_id'), 'name': txn0.get('supplier_name')} if txn0.get('supplier_id') else None
        er = draft.get('expected_receipt_match') or None
        row['detail'] = {'supplier': supplier, 'supplier_lot_code': payload.get('supplier_lot_code'),
                         'cases': cases, 'case_size_lb': payload.get('case_size_lb'), 'bol_reference': payload.get('bol_reference'),
                         'expected_receipt': {'id': er['id'], 'reference_number': er.get('reference_number')} if er else None}
        sup = f' (supplier lot {row["lot"]["supplier_lot_code"]})' if row['lot'] and row['lot'].get('supplier_lot_code') else ''
        sup_es = sup.replace('supplier lot', 'lote del proveedor')
        row.update(pair(f'{number}  {row["product"]["name"]}  {fmt_lb(lb)} lb  lot {row["lot"]["lot_code"] if row["lot"] else "?"}{sup}  {clock}',
                        f'{number}  {row["product"]["name"]}  {fmt_lb(lb)} lb  lote {row["lot"]["lot_code"] if row["lot"] else "?"}{sup_es}  {clock}', 'line'))
        if row['lot'] and row['lot']['identity_status'] == 'unidentified':
            exc = _open_exception(cur, catalog, kind='UNIDENTIFIED_LOT', lot_id=lot_id)
            due = exc['due_at'] if exc else None
            row['flags'].append(_flag('UNIDENTIFIED_LOT', 'warn',
                                      f'UNIDENTIFIED — resolve by {short_date(due, "en")}' if due else 'UNIDENTIFIED',
                                      f'SIN IDENTIFICAR — resolver antes del {short_date(due, "es")}' if due else 'SIN IDENTIFICAR',
                                      {'lot_id': lot_id, 'exception_id': exc['id'] if exc else None, 'due_at': iso(due, api)}, iso(due, api)))
            if exc:
                row['exceptions_opened'].append(exc['id'])
        if not txn0.get('supplier_id'):
            row['flags'].append(_flag('SUPPLIER_MISSING', 'warn', 'SUPPLIER MISSING', 'FALTA PROVEEDOR',
                                      {'transaction_id': txn0.get('id')}))
    elif action in ('make', 'pack'):
        out_lines = _lines(receipt, +1)
        lb = sum(float(l['quantity_lb']) for l in out_lines)
        if action == 'make':
            out_product, out_lot = payload['product_id'], response.get('lot_id')
        else:
            out_product, out_lot = payload['target_product_id'], response.get('output_lot_id') or response.get('lot_id')
        row['product'], row['lot'] = product_full(out_product), lot_full(out_lot)
        confirmations = {c['lot_id']: c for c in txn0.get('lot_confirmations') or []}
        shortages = {s['lot_id']: s for s in response.get('shortages') or []}
        for s in shortages.values():
            lot = catalog.lot(s['lot_id'])
            p = catalog.product(s.get('product_id') or lot['product_id'])
            due = s.get('due_at')
            due_dt = datetime.fromisoformat(due) if isinstance(due, str) else due
            row['flags'].append(_flag('SHORT', 'warn',
                                      f'SHORT {fmt_lb(s["short_lb"])} lb {p["name"]} lot {lot["lot_code"]} — resolve by {short_date(due_dt, "en")}',
                                      f'FALTAN {fmt_lb(s["short_lb"])} lb {p["name"]} lote {lot["lot_code"]} — resolver antes del {short_date(due_dt, "es")}',
                                      {'shortage_flag_id': s.get('shortage_flag_id'), 'exception_id': s.get('exception_id'),
                                       'lot_id': s['lot_id'], 'short_lb': s['short_lb']}, iso(due_dt, api)))
            if s.get('exception_id'):
                row['exceptions_opened'].append(s['exception_id'])
        if action == 'make':
            ingredients = []
            for item in draft.get('input_plan') or []:
                c = confirmations.get(item['lot_id'])
                ingredients.append({'product': catalog.product_ref(item['product_id']), 'lot': catalog.lot_ref(item['lot_id']),
                                    'quantity_lb': item['quantity_lb'],
                                    'confirmation': {'method': c['method'], 'value': c['value']} if c else None,
                                    'short_lb': shortages.get(item['lot_id'], {}).get('short_lb', 0)})
            substitutions = []
            for s in txn0.get('substitutions') or []:
                reason = catalog.reason(s.get('reason_code'))
                substitutions.append({'ingredient_product': catalog.product_ref(s['ingredient_product_id']),
                                      'substitute_product': catalog.product_ref(s['substitute_product_id']),
                                      'lot': catalog.lot_ref(s.get('lot_id')),
                                      'note': s.get('note'),
                                      'reason_code': {k: reason[k] for k in ('code', 'label_en', 'label_es')} if reason else None})
            row['quantity'] = quantity(lb)
            row['detail'] = {'batch_product': catalog.product_ref(out_product), 'output_lb': lb, 'ingredients': ingredients,
                             'substitutions': substitutions, 'production_warning': draft.get('production_warning')}
            ing_en = ', '.join(f'{i["product"]["name"]} lot {i["lot"]["lot_code"]}'
                               + (f' ({i["confirmation"]["method"].replace("_", "-")} typed)' if i['confirmation'] else '') for i in ingredients)
            ing_es = ing_en.replace(' lot ', ' lote ').replace(' typed)', ' escrito)')
            code = row['product'].get('odoo_code') or ''
            row.update(pair(f'{number}  {row["product"]["name"]} Batch {code}  {fmt_lb(lb)} lb  lot {row["lot"]["lot_code"] if row["lot"] else "?"}'
                            + (f'  ingredients: {ing_en}' if ing_en else ''),
                            f'{number}  {row["product"]["name"]} {_ACTION_ES["make"]} {code}  {fmt_lb(lb)} lb  lote {row["lot"]["lot_code"] if row["lot"] else "?"}'
                            + (f'  ingredientes: {ing_es}' if ing_es else ''), 'line'))
            if response.get('auto_substitutions'):
                row['detail']['auto_substitutions'] = response['auto_substitutions']
                for substitution in response['auto_substitutions']:
                    row['line_en'] += '  ' + substitution['message_en']
                    row['line_es'] += '  ' + substitution['message_es']
        else:
            source = payload['source_product_id']
            src = next((i for i in draft.get('input_plan') or [] if i['product_id'] == source), None)
            cases = payload.get('cases')
            row['quantity'] = quantity(lb, cases, 'cases', f'{fmt_cases(cases)} cases ({fmt_lb(lb)} lb)', f'{fmt_cases(cases)} cajas ({fmt_lb(lb)} lb)')
            row['detail'] = {'source_product': catalog.product_ref(source), 'source_lot': catalog.lot_ref(src['lot_id']) if src else None,
                             'target_product': catalog.product_ref(out_product), 'cases': cases, 'lb': lb,
                             'ownership': 'cns', 'invoice_status': None}
            src_code = row['detail']['source_lot']['lot_code'] if row['detail']['source_lot'] else '?'
            row.update(pair(f'{number}  {row["product"]["name"]}  {fmt_cases(cases)} cases ({fmt_lb(lb)} lb) from lot {src_code}',
                            f'{number}  {row["product"]["name"]}  {fmt_cases(cases)} cajas ({fmt_lb(lb)} lb) del lote {src_code}', 'line'))
    elif action in wt.SHIP_ACTIONS:
        shipment = receipt['shipment'] or {}
        lines = []
        for line in draft.get('lines') or []:
            lines.append({'product': catalog.product_ref(line['product_id']), 'cases': line.get('cases'), 'lb': line.get('quantity_lb'),
                          'lot': catalog.lot_ref(line['lots'][0]['lot_id']) if line.get('lots') else None, 'is_service': line.get('is_service', False)})
        goods = [l for l in lines if not l['is_service']]
        lb = sum(float(l['lb'] or 0) for l in goods)
        cases = sum(float(l['cases']) for l in goods) if goods and all(l['cases'] is not None for l in goods) else None
        products = {l['product']['id'] for l in goods if l['product']}
        lots = {l['lot']['id'] for l in goods if l['lot']}
        row['product'] = product_full(next(iter(products))) if len(products) == 1 else None
        row['lot'] = lot_full(next(iter(lots))) if len(lots) == 1 else None
        row['quantity'] = quantity(lb, cases, 'cases' if cases is not None else 'lb',
                                   f'{fmt_cases(cases)} cases' if cases is not None else f'{fmt_lb(lb)} lb',
                                   f'{fmt_cases(cases)} cajas' if cases is not None else f'{fmt_lb(lb)} lb')
        order = None
        if shipment.get('sales_order_id'):
            if hasattr(catalog, '_orders'):
                o = catalog._orders.get(shipment['sales_order_id'])
            else:
                cur.execute('SELECT so.id,so.order_number,c.name AS customer FROM sales_orders so JOIN customers c ON c.id=so.customer_id WHERE so.id=%s',
                            (shipment['sales_order_id'],))
                o = cur.fetchone()
            order = {'id': o['id'], 'so_number': o['order_number'], 'customer_name': o['customer']} if o else None
        row['detail'] = {'sales_order': order, 'customer': {'id': draft.get('customer_id'), 'name': draft.get('customer')},
                         'lines': [{k: v for k, v in l.items() if k != 'is_service'} for l in lines],
                         'bol_reference': shipment.get('bol_reference') or payload.get('bol_reference'),
                         'proof_status': shipment.get('proof_status'), 'bins': [] if action == 'ship_bulk' else None}
        who = f'{order["so_number"]} {order["customer_name"]}' if order else (draft.get('customer') or '')
        photo_en = 'photo ✓' if shipment.get('proof_status') == 'complete' else 'photo pending'
        photo_es = 'foto ✓' if shipment.get('proof_status') == 'complete' else 'foto pendiente'
        row.update(pair(f'{number}  {who}  {row["quantity"]["display_en"]}  BOL {row["detail"]["bol_reference"] or "?"}  {photo_en}',
                        f'{number}  {who}  {row["quantity"]["display_es"]}  BOL {row["detail"]["bol_reference"] or "?"}  {photo_es}', 'line'))
        if shipment.get('proof_status') == 'photo_pending':
            exc = _open_exception(cur, catalog, kind='BOL_PHOTO_MISSING', shipment_id=shipment['id'])
            due = shipment.get('proof_due_at')
            row['flags'].append(_flag('PHOTO_PENDING', 'warn',
                                      f'BOL photo pending — due by {plant(due, api).strftime("%H:%M")}' if due else 'BOL photo pending',
                                      f'foto del BOL pendiente — vence a las {plant(due, api).strftime("%H:%M")}' if due else 'foto del BOL pendiente',
                                      {'shipment_id': shipment['id'], 'exception_id': exc['id'] if exc else None}, iso(due, api)))
        if any(a.get('code') == 'SHIP_NOT_SAME_DAY' for a in response.get('policy_acknowledgements') or []):
            row['flags'].append(_flag('NOT_SAME_DAY', 'info', 'shipped on a different plant day than entered (acknowledged)',
                                      'enviado en un día de planta distinto al registro (confirmado)', {'shipment_id': shipment.get('id')}))
    elif action in ('adjust', 'found'):
        lines = _lines(receipt) or (txn0.get('lines') or [])
        delta = sum(float(l['quantity_lb']) for l in lines)
        lot_id = response.get('lot_id') or payload.get('lot_id') or (lines[0]['lot_id'] if lines else None)
        product_id = payload.get('product_id') or (lines[0]['product_id'] if lines else None) or catalog.lot(lot_id)['product_id']
        row['product'], row['lot'] = product_full(product_id), lot_full(lot_id)
        row['quantity'] = quantity(delta, None, 'lb', f'{fmt_lb(delta, signed=True)} lb')
        reason = catalog.reason(response.get('reason_code') or draft.get('reason_code'))
        review = response.get('correction_review') or draft.get('correction_review') or {}
        note = draft.get('note') if action == 'adjust' else draft.get('notes')
        # Commit-time balances: `new_balance_lb` is lot_on_hand() read right after the post
        # (adjust: main._adjust_commit_core; found: ticket_actions.post), so before = after − delta
        # is the balance the correction actually moved from — never the prepare-time draft figure.
        after = response.get('new_balance_lb')
        if after is not None:
            after = float(after)
            before = round(after - delta, 4)
        elif action == 'found' and not draft.get('lot_exists'):
            before, after = 0.0, delta          # a found lot that did not exist starts at zero
        else:
            before = after = None               # legacy receipt without a stored commit balance
        row['detail'] = {'lot': catalog.lot_ref(lot_id), 'delta_lb': delta, 'lot_on_hand_before': before, 'lot_on_hand_after': after,
                         'reason_code': {k: reason[k] for k in ('code', 'label_en', 'label_es', 'note_required')} if reason else None,
                         'note': note, 'photo_attached': bool(review.get('attachment_ref'))}
        label_en = reason['label_en'] if reason else '?'
        label_es = reason['label_es'] if reason else '?'
        row.update(pair(f'{number}  {row["product"]["name"]} lot {row["lot"]["lot_code"] if row["lot"] else "?"}  {fmt_lb(delta, signed=True)} lb  reason: {label_en}',
                        f'{number}  {row["product"]["name"]} lote {row["lot"]["lot_code"] if row["lot"] else "?"}  {fmt_lb(delta, signed=True)} lb  razón: {label_es}', 'line'))
        if review.get('photo_required') or 'OVER_10_PERCENT_OF_BOOK' in (review.get('rules') or []):
            # The hold's exception (A3b) is resolved by the approval or the photo that let this
            # receipt post, so it is read in ANY status: the reference must survive resolution.
            if hasattr(catalog, '_exceptions'):
                exc = next((e for e in catalog._exceptions if e['kind']=='LARGE_CORRECTION' and e['ticket_id']==ticket['id']), None)
            else:
                cur.execute("SELECT id,status,resolution_kind FROM exceptions WHERE kind='LARGE_CORRECTION' AND ticket_id=%s ORDER BY id LIMIT 1",
                            (ticket['id'],))
                exc = cur.fetchone()
            approved_by = (response.get('approval') or {}).get('approved_by') or None
            pct = review.get('pct_of_book')
            pct_text = f' ({pct * 100:.0f}% of lot)' if pct is not None else ''
            if approved_by:
                tail_en, tail_es = f' — approved by {approved_by.get("name")}', f' — aprobada por {approved_by.get("name")}'
            elif review.get('attachment_ref'):
                tail_en, tail_es = ' — photo attached', ' — con foto'
            elif review.get('photo_required'):
                tail_en, tail_es = ' — photo required', ' — requiere foto'
            else:
                tail_en = tail_es = ''
            row['flags'].append(_flag('LARGE_CORRECTION', 'warn',
                                      f'LARGE CORRECTION {fmt_lb(delta, signed=True)} lb{pct_text}{tail_en}',
                                      f'CORRECCIÓN GRANDE {fmt_lb(delta, signed=True)} lb{pct_text.replace(" of lot", " del lote")}{tail_es}',
                                      {'exception_id': exc['id'] if exc else None, 'delta_lb': delta, 'pct': pct,
                                       'resolution': exc['resolution_kind'] if exc else None,
                                       'approved_by_actor_id': approved_by.get('id') if approved_by else None}))
            if exc:
                row['exceptions_opened'].append(exc['id'])
        if reason and reason['code'] == 'unknown':
            row['flags'].append(_flag('UNKNOWN_REASON', 'warn', 'reason: unknown — highlighted on the weekly view',
                                      'razón: desconocida — resaltado en la vista semanal', {'transaction_id': txn0.get('id')}))
    else:
        # rename_lot / update_supplier_lot / void: no tickets exist yet (contract §2); render generically.
        row['quantity'] = quantity(0)
        row['detail'] = {}
        row.update(pair(f'{number}  {action}', f'{number}  {action}', 'line'))

    row.update(kosher_tiers.summary_fields(receipt))
    if row['kosher_attested_at'] is not None:
        value = row['kosher_attested_at']
        row['kosher_attested_at'] = iso(datetime.fromisoformat(value) if isinstance(value, str) else value, api)
    if action in kosher_tiers.LOT_RECEIPT_ACTIONS:
        row['detail']['kosher_lots'] = kosher_tiers.summary_lots(receipt)
        for evidence in row['detail']['kosher_lots']:
            value = evidence.get('kosher_attested_at')
            if value is not None:
                evidence['kosher_attested_at'] = iso(datetime.fromisoformat(value) if isinstance(value, str) else value, api)
    if row['kosher_tier'] == kosher_tiers.EXTRA and action in ('receive', 'make', 'pack', 'found'):
        row['flags'].append(_flag('EXTRA_KOSHER', 'info', 'EXTRA-KOSHER (SS)',
                                  'EXTRA-KOSHER (SS)', {'kosher_tier': row['kosher_tier']}))
        row['line_en'] += '  EXTRA-KOSHER (SS)'
        row['line_es'] += '  EXTRA-KOSHER (SS)'
    if action == 'pack' and row['transaction_kosher_downgrade']:
        row['flags'].append(_flag('KOSHER_DOWNGRADE', 'warn',
                                  'KOSHER DOWNGRADE — extra-kosher source packed as Regular',
                                  'DESCENSO KOSHER — origen extra-kosher empacado como Regular',
                                  {'transaction_id': response.get('transaction_id') or txn0.get('id'),
                                   'lot_id': row['lot']['id'] if row['lot'] else None,
                                   'transaction_kosher_downgrade': True}))

    if timing.get('late_entry_exception_id'):
        row['flags'].append(_flag('LATE_ENTRY', 'warn',
                                  f'LATE ENTRY — {fmt_lb(hours_late)} h after it happened; owner to acknowledge',
                                  f'ENTRADA TARDÍA — {fmt_lb(hours_late)} h después de ocurrir; el dueño debe revisarla',
                                  {'exception_id': timing['late_entry_exception_id'], 'hours_late': hours_late}))
        row['exceptions_opened'].append(timing['late_entry_exception_id'])
    if 'POSSIBLE_DUPLICATE' in (ticket['acknowledged'] or []):
        earlier = next((w.get('refs', {}).get('receipt_number') for w in ticket['warnings'] or [] if w.get('code') == 'POSSIBLE_DUPLICATE'), None)
        row['flags'].append(_flag('POSSIBLE_DUPLICATE_ACKED', 'info',
                                  f'possible duplicate of {earlier or "an earlier entry"} acknowledged',
                                  f'posible duplicado de {earlier or "una entrada anterior"} confirmado', {'receipt_number': earlier}))
    if effective == 'voided':
        row['flags'].append(_flag('VOIDED', 'warn', 'VOIDED', 'ANULADO', {'voided_by_receipt_number': None}))
    if row['quantity'] is None:
        row['quantity'] = quantity(0)
    row['_receipt'] = receipt   # posted lines for LOT BALANCES; popped before the document is returned
    return row


def other_receipt_row(api, cur, catalog, number, *, receipt=None):
    """A metadata receipt (order / expected receipt / lot move): listed under `other_receipts`,
    outside the ten sections and outside the hash — never rendered as an adjustment."""
    receipt = receipt if receipt is not None else wt.receipt_detail(api, cur, number)
    action = receipt['action']
    kind = OTHER_KIND.get(action, 'other')
    happened, entered = plant(receipt['happened_at'], api), plant(receipt['entered_at'], api)
    actor = receipt['actor'] or {}
    label_en, label_es = _OTHER_LABELS.get(action, (action.replace('_', ' '), action.replace('_', ' ')))
    # A later ready/un-ready action must not rewrite this receipt's meaning.
    if action == 'mark_order_ready' and (receipt.get('response') or {}).get('ready') is False:
        label_en, label_es = 'order marked not ready', 'pedido marcado no listo'
    refs = {'order_id': None, 'order_number': None, 'expected_receipt_ids': [], 'lot_id': None}
    subject_en = subject_es = ''
    if kind == 'order' and receipt.get('order'):
        order = receipt['order']
        refs.update(order_id=order.get('id'), order_number=order.get('order_number'))
        subject_en = subject_es = ' '.join(str(x) for x in (order.get('order_number'), order.get('customer')) if x)
    elif kind == 'expected_receipt':
        refs['expected_receipt_ids'] = [e['id'] for e in receipt.get('expected_receipts') or [] if e.get('id') is not None]
        subject_en = ', '.join(f'ER #{i}' for i in refs['expected_receipt_ids'])
        subject_es = ', '.join(f'RE #{i}' for i in refs['expected_receipt_ids'])
    elif kind == 'lot_move':
        payload = receipt.get('draft') or {}
        lot_id = (receipt.get('response') or {}).get('lot_id') or payload.get('lot_id')
        if lot_id is None and receipt.get('lots'):
            lot_id = receipt['lots'][0]['id']
        refs['lot_id'] = lot_id
        lot = catalog.lot(lot_id) if lot_id is not None else None
        to = (receipt.get('response') or {}).get('to_location') or payload.get('to_location') or ''
        subject_en = f'lot {lot["lot_code"]}' + (f' → {to}' if to else '') if lot else ''
        subject_es = f'lote {lot["lot_code"]}' + (f' → {to}' if to else '') if lot else ''
    clock = happened.strftime('%H:%M')
    row = {'receipt_number': number, 'action': action, 'kind': kind, 'status': receipt['status'],
           'actor': {'id': actor.get('id'), 'name': actor.get('name') or receipt['operator_id'], 'role': actor.get('role')},
           'client_source': receipt['client_source'], 'happened_at': happened.isoformat(), 'entered_at': entered.isoformat(),
           'late_entry': bool(receipt['late_entry']), 'refs': refs,
           **pair(f'{number}  {label_en}' + (f'  {subject_en}' if subject_en else '') + f'  {clock}',
                  f'{number}  {label_es}' + (f'  {subject_es}' if subject_es else '') + f'  {clock}', 'line')}
    return row


def _without_lang_copies(node):
    """Drop the un-suffixed `lang` copies (§1.1) so the hash is the same in `en` and `es`."""
    if isinstance(node, dict):
        return {k: _without_lang_copies(v) for k, v in node.items() if not (f'{k}_en' in node and f'{k}_es' in node)}
    if isinstance(node, list):
        return [_without_lang_copies(v) for v in node]
    return node


def _stable_detail(node):
    """Display names/labels cannot invalidate confirmed immutable receipt evidence."""
    if isinstance(node, dict):
        return {key: _stable_detail(value) for key, value in node.items()
                if key not in ('name', 'customer_name', 'label', 'label_en', 'label_es', 'kosher_label', 'kosher_attested_by_name')}
    if isinstance(node, list):
        return [_stable_detail(value) for value in node]
    return node


def reduced(row):
    """The hash view of a receipt row — stable fields and flag codes/refs only (§1.3).
    Computed on the serialised document, so numbers hash as the response prints them."""
    out = {k: _without_lang_copies(row[k]) for k in HASH_FIELDS}
    out['detail'] = _stable_detail(out['detail'])
    out['product'] = {'id': row['product']['id']} if row['product'] else None
    out['lot'] = {'id': row['lot']['id']} if row['lot'] else None
    out['quantity'] = {'lb': row['quantity']['lb'], 'cases': row['quantity']['cases']}
    out['flags'] = [{'code': f['code'], 'refs': f['refs']} for f in row['flags']]
    return out


def summary_hash(api, sections, direct_makes=None):
    reduced_sections = [{'key': s['key'], 'rows': [reduced(r) for r in s['rows']]} for s in sections if s['key'] in RECEIPT_SECTIONS]
    if direct_makes:
        # Direct GPT makes have no MK receipt. Include their immutable substitution
        # evidence without inventing receipt numbers or changing receipt counts.
        reduced_sections.append({'key': 'direct_make_substitutions', 'rows': [
            dict(_without_lang_copies({k: r[k] for k in ('transaction_id', 'happened_at', 'entered_at',
                                                       'quantity_lb', 'auto_substitutions')}),
                 consumed_lots=[{k: lot[k] for k in ('lot_id', 'quantity_lb')} for lot in r['consumed_lots']])
            for r in direct_makes]})
    return wt.canonical_hash(wt.json_value(api, reduced_sections))


def direct_make_substitutions(api, cur, day, actor):
    """Expose the direct GPT route's persisted swaps, without manufacturing tickets."""
    cur.execute(f'''SELECT t.id AS transaction_id,t.occurred_at,t.created_at,
                          p.id AS product_id,p.name AS product_name,l.id AS lot_id,l.lot_code,
                          output.quantity_lb,s.note,
                          COALESCE((SELECT jsonb_agg(jsonb_build_object(
                              'lot_id',used.lot_id,'lot_code',used_lot.lot_code,
                              'quantity_lb',-used.quantity_lb) ORDER BY used.id)
                              FROM {api.POSTED_LINES} used JOIN lots used_lot ON used_lot.id=used.lot_id
                              WHERE used.transaction_id=t.id AND used.product_id=s.substitute_product_id
                                AND used.quantity_lb<0),'[]'::jsonb) AS consumed_lots
                   FROM transactions t JOIN transaction_substitutions s ON s.transaction_id=t.id
                   JOIN {api.POSTED_LINES} output ON output.transaction_id=t.id AND output.quantity_lb>0
                   JOIN products p ON p.id=output.product_id JOIN lots l ON l.id=output.lot_id
                   WHERE t.type='make' AND t.ticket_id IS NULL AND t.business_date=%s
                     AND s.reason_code='oat_bran_auto_substitute'
                     AND (%s::integer IS NULL OR t.entered_by_actor_id=%s)
                   ORDER BY t.occurred_at,t.id''', (day, actor['id'] if actor else None, actor['id'] if actor else None))
    rows = []
    for result in cur.fetchall():
        try:
            evidence = json.loads(result['note'] or '{}')
        except (TypeError, ValueError):
            continue
        if not isinstance(evidence, dict) or not evidence.get('automatic'):
            continue
        row = {k: result[k] for k in ('transaction_id', 'quantity_lb', 'consumed_lots')}
        row.update(product={'id': result['product_id'], 'name': result['product_name']},
                   lot={'id': result['lot_id'], 'lot_code': result['lot_code']},
                   happened_at=iso(result['occurred_at'], api), entered_at=iso(result['created_at'], api),
                   auto_substitutions=[evidence])
        en = f'Transaction {result["transaction_id"]}  {result["product_name"]}  {fmt_lb(result["quantity_lb"])} lb  lot {result["lot_code"]}'
        es = f'Transacción {result["transaction_id"]}  {result["product_name"]}  {fmt_lb(result["quantity_lb"])} lb  lote {result["lot_code"]}'
        row.update(pair(en + '  ' + evidence['message_en'], es + '  ' + evidence['message_es'], 'line'))
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Sections 7–9
# ---------------------------------------------------------------------------
def not_entered_yet(cur, catalog, day):
    rows = []
    cur.execute('''SELECT er.id,er.reference_number,er.product_id,er.expected_qty,er.expected_date,er.status,
                          s.id AS supplier_id,s.name AS supplier_name
                   FROM expected_receipts er LEFT JOIN suppliers s ON s.id=er.supplier_id
                   WHERE er.status='open' AND er.expected_date IS NOT NULL AND er.expected_date<=%s
                     AND NOT EXISTS (SELECT 1 FROM transactions t WHERE t.expected_receipt_id=er.id AND t.receipt_number IS NOT NULL)
                   ORDER BY er.expected_date,er.id''', (day,))
    for er in cur.fetchall():
        p = catalog.product(er['product_id'])
        ref = er['reference_number'] or f'#{er["id"]}'
        overdue = er['expected_date'] < day
        when_en = 'due today' if not overdue else f'due {short_date(er["expected_date"], "en")}, overdue'
        when_es = 'vence hoy' if not overdue else f'venció el {short_date(er["expected_date"], "es")}'
        rows.append({'kind': 'expected_receipt', 'overdue': overdue,
                     'ref': {'expected_receipt_id': er['id'], 'reference_number': er['reference_number'], 'production_run_id': None},
                     'product': {'id': p['id'], 'name': p['name'], 'odoo_code': p.get('odoo_code')},
                     'expected': {'quantity': er['expected_qty'], 'unit': 'lb', 'date': er['expected_date'].isoformat(),
                                  'supplier': {'id': er['supplier_id'], 'name': er['supplier_name']} if er['supplier_id'] else None},
                     'status': er['status'], '_sort': (0 if overdue else 1, er['expected_date'], p['name']),
                     **pair(f'expected receipt {ref} ({p["name"]}, {when_en}) — no RCV receipt',
                            f'recepción esperada {ref} ({p["name"]}, {when_es}) — sin recibo RCV', 'line')})
    # Runs are "entered" when a committed MK/PK receipt for the run's product happened on the
    # planned date (any actor — expectations are plant-wide, §2.3). One receipt covers one run,
    # lowest run id first; what is left is listed.
    cur.execute('''SELECT action, (payload->>'product_id')::int AS product_id, (payload->>'target_product_id')::int AS target_product_id
                   FROM write_tickets WHERE status='committed' AND action IN ('make','pack')
                     AND (((payload->>'occurred_at')::timestamptz AT TIME ZONE 'America/New_York')::date)=%s
                   ORDER BY committed_at,id''', (day,))
    covered = {}
    for t in cur.fetchall():
        product = t['target_product_id'] if t['action'] == 'pack' else t['product_id']
        covered[(t['action'], product)] = covered.get((t['action'], product), 0) + 1
    cur.execute('''SELECT id,product_id,run_type,status,planned_qty,planned_unit,planned_qty_lb,planned_date
                   FROM production_runs WHERE planned_date=%s AND status IN ('planned','in_progress') ORDER BY id''', (day,))
    for run in cur.fetchall():
        wants = ('pack',) if run['run_type'] == 'pack' else (('make',) if run['run_type'] in ('bake', 'coconut') else ('make', 'pack'))
        matched = next((w for w in wants if covered.get((w, run['product_id']), 0) > 0), None)
        if matched:
            covered[(matched, run['product_id'])] -= 1
            continue
        p = catalog.product(run['product_id'])
        prefix = 'PK' if run['run_type'] == 'pack' else ('MK' if run['run_type'] in ('bake', 'coconut') else 'MK/PK')
        status_es = {'planned': 'planeada', 'in_progress': 'en curso'}[run['status']]
        type_es = {'pack': 'empaque', 'bake': 'horneado', 'coconut': 'coco', 'other': 'otro'}[run['run_type']]
        rows.append({'kind': 'production_run', 'overdue': False,
                     'ref': {'expected_receipt_id': None, 'reference_number': None, 'production_run_id': run['id']},
                     'product': {'id': p['id'], 'name': p['name'], 'odoo_code': p.get('odoo_code')},
                     'expected': {'run_type': run['run_type'], 'planned_qty': run['planned_qty'] or run['planned_qty_lb'],
                                  'planned_unit': run['planned_unit'] or 'lb', 'quantity': run['planned_qty_lb'], 'unit': 'lb',
                                  'date': run['planned_date'].isoformat()},
                     'status': run['status'], '_sort': (1, run['planned_date'], p['name']),
                     **pair(f'production run #{run["id"]} {p["name"]} {run["run_type"]}, planned today — status {run["status"].replace("_", " ")}, no {prefix} receipt',
                            f'corrida #{run["id"]} {p["name"]} {type_es}, planeada hoy — estado {status_es}, sin recibo {prefix}', 'line')})
    rows.sort(key=lambda r: r.pop('_sort'))
    return rows


def kosher_holds(api, cur, catalog, day, actor_id=None):
    """Unposted expectations remain visible through expiry until owner rejection.

    Holds are excluded from receipt count, confirmed hash and touched lot balances.
    """
    cur.execute('''SELECT t.id,t.action,t.receipt_number,t.actor_id,t.operator_id,t.payload,t.draft,t.response,
                          t.expires_at,e.id AS exception_id,e.owner_actor_id
                   FROM write_tickets t JOIN exceptions e ON e.ticket_id=t.id
                   WHERE t.status='awaiting_approval' AND t.action IN ('make','found')
                     AND e.kind='KOSHER_ATTESTATION' AND e.status IN ('open','escalated')
                     AND t.response->>'held'='true' AND t.response->>'posted'='false'
                     AND t.response->>'kosher_tier'='extra_kosher'
                     AND (((t.payload->>'occurred_at')::timestamptz AT TIME ZONE 'America/New_York')::date)<=%s
                     AND (%s::integer IS NULL OR t.actor_id=%s)
                     AND NOT EXISTS (SELECT 1 FROM transactions posted WHERE posted.ticket_id=t.id)
                   ORDER BY t.expires_at,t.id''', (day, actor_id, actor_id))
    rows = []
    now = api.get_plant_now()
    for ticket in cur.fetchall():
        draft, response = ticket['draft'] or {}, ticket['response'] or {}
        preparer = draft.get('actor') or {}
        product_id = ticket['payload'].get('product_id')
        product = catalog.product(product_id)
        expires_at = iso(ticket['expires_at'], api)
        expired = ticket['expires_at'] <= now
        approval_state = 'expired' if expired else 'awaiting_approval'
        receipt = ticket['receipt_number']
        state_en = 'approval expired — owner must reject and prepare again' if expired else 'awaiting owner approval'
        state_es = 'aprobación vencida — el dueño debe rechazar y preparar de nuevo' if expired else 'pendiente de aprobación del dueño'
        product_name = draft.get('product_name') or product['name']
        row = {'kind': 'kosher_hold', 'receipt_number': receipt, 'action': ticket['action'],
               'ticket_id': ticket['id'], 'status': 'awaiting_approval', 'held': True, 'posted': False,
               'actor': {'id': preparer.get('id', ticket['actor_id']),
                         'name': preparer.get('name') or ticket['operator_id'], 'role': preparer.get('role')},
               'product': {'id': product_id, 'name': product_name, 'odoo_code': product.get('odoo_code')},
               'ref': {'ticket_id': ticket['id'], 'exception_id': ticket['exception_id'], 'receipt_number': receipt},
               'approval': {'status': approval_state, 'owner_actor_id': ticket['owner_actor_id'], 'expires_at': expires_at},
               'expires_at': expires_at, 'overdue': expired,
               **pair(f'{receipt}  {product_name}  EXTRA-KOSHER (SS) — {state_en}; expires {expires_at}',
                      f'{receipt}  {product_name}  EXTRA-KOSHER (SS) — {state_es}; vence {expires_at}', 'line')}
        row.update(kosher_tiers.summary_fields({'action': ticket['action'], 'draft': draft, 'response': response}))
        rows.append(row)
    return rows


def lot_balances(api, cur, catalog, receipt_rows):
    touched = {}
    seen = set()
    for row in receipt_rows:
        if row['receipt_number'] in seen:
            continue
        seen.add(row['receipt_number'])
        for txn in row['_receipt']['transactions']:
            if txn['effective_status'] != 'posted':
                continue
            for line in txn['lines']:
                if line['lot_id'] is None:
                    continue
                entry = touched.setdefault(line['lot_id'], {'delta': 0.0, 'receipts': []})
                entry['delta'] += float(line['quantity_lb'])
                if row['receipt_number'] not in entry['receipts']:
                    entry['receipts'].append(row['receipt_number'])
    rows = []
    for lot_id, entry in touched.items():
        lot = catalog.lot(lot_id)
        p = catalog.product(lot['product_id'])
        if 'on_hand_lb' in lot:
            after, short = float(lot['on_hand_lb']), lot['has_shortage']
        else:
            after = api.lot_on_hand(cur, lot_id)
            cur.execute("SELECT 1 FROM shortage_flags WHERE lot_id=%s AND status IN ('open','escalated') LIMIT 1", (lot_id,))
            short = cur.fetchone() is not None
        before = round(after - entry['delta'], 4)
        negative = after < 0
        tail_en, tail_es = (' (short)', ' (faltante)') if negative or short else ('', '')
        rows.append({'lot': {'id': lot_id, 'lot_code': lot['lot_code'], 'product': {'id': p['id'], 'name': p['name']}},
                     'before_lb': before, 'after_lb': after, 'delta_lb': round(entry['delta'], 4),
                     'short': short, 'negative': negative, 'receipts': entry['receipts'],
                     **pair(f'{p["name"]} lot {lot["lot_code"]} {fmt_lb(before)} → {fmt_lb(after)} lb{tail_en}',
                            f'{p["name"]} lote {lot["lot_code"]} {fmt_lb(before)} → {fmt_lb(after)} lb{tail_es}', 'line')})
    rows.sort(key=lambda r: (r['lot']['product']['name'], r['lot']['lot_code']))
    return rows


def exception_summary(exc, catalog):
    kind = exc['kind']
    labels = _KINDS.get(kind, (kind.lower().replace('_', ' '),) * 4 + (kind.replace('_', ' '), kind.replace('_', ' ')))
    detail = exc.get('detail') or {}
    head_en, head_es = labels[4], labels[5]
    if kind == 'SHORTAGE' and detail.get('short_lb') is not None:
        head_en, head_es = f'SHORT {fmt_lb(detail["short_lb"])} lb', f'FALTAN {fmt_lb(detail["short_lb"])} lb'
    subject_en = subject_es = ''
    if exc.get('lot_id'):
        lot = catalog.lot(exc['lot_id'])
        p = catalog.product(exc.get('product_id') or lot['product_id'])
        subject_en, subject_es = f' {p["name"]} lot {lot["lot_code"]}', f' {p["name"]} lote {lot["lot_code"]}'
    elif exc.get('product_id'):
        p = catalog.product(exc['product_id'])
        subject_en = subject_es = f' {p["name"]}'
    elif exc.get('shipment_id'):
        subject_en, subject_es = f' shipment {exc["shipment_id"]}', f' envío {exc["shipment_id"]}'
    src_en = f' (from {exc["receipt_number"]})' if exc.get('receipt_number') else ''
    src_es = f' (de {exc["receipt_number"]})' if exc.get('receipt_number') else ''
    due = exc.get('due_at')
    due_en = f' — resolve by {short_date(due, "en")}' if due else ''
    due_es = f' — resolver antes del {short_date(due, "es")}' if due else ''
    esc_en = ' — ESCALATED' if exc['status'] == 'escalated' else ''
    esc_es = ' — ESCALADA' if exc['status'] == 'escalated' else ''
    return f'{head_en}{subject_en}{src_en}{due_en}{esc_en}', f'{head_es}{subject_es}{src_es}{due_es}{esc_es}'


def open_exceptions(api, cur, catalog, actor_id):
    cur.execute('''SELECT e.*, o.name AS owner_name FROM exceptions e LEFT JOIN actors o ON o.id=e.owner_actor_id
                   WHERE e.status IN ('open','escalated') AND (%s::int IS NULL OR e.owner_actor_id=%s)
                   ORDER BY (e.due_at IS NOT NULL AND e.due_at < clock_timestamp()) DESC, e.due_at NULLS LAST, e.opened_at, e.id''',
                (actor_id, actor_id))
    rows, by_kind = [], {}
    now = datetime.now(api.PLANT_TIMEZONE)
    for exc in cur.fetchall():
        en, es = exception_summary(exc, catalog)
        refs = {'receipt_number': exc.get('receipt_number'), 'lot_id': exc.get('lot_id'), 'product_id': exc.get('product_id'),
                'transaction_id': exc.get('transaction_id'), 'shipment_id': exc.get('shipment_id'), 'ticket_id': exc.get('ticket_id')}
        if exc['kind'] == 'SHORTAGE':
            cur.execute('SELECT id FROM shortage_flags WHERE exception_id=%s', (exc['id'],))
            flag = cur.fetchone()
            refs['shortage_flag_id'] = flag['id'] if flag else None
        rows.append({'exception_id': exc['id'], 'kind': exc['kind'], 'severity': exc['severity'], 'status': exc['status'],
                     'opened_at': iso(exc['opened_at'], api), 'due_at': iso(exc['due_at'], api),
                     'overdue': bool(exc['due_at'] and exc['due_at'] < now), 'escalated': exc['status'] == 'escalated',
                     'owner': {'id': exc['owner_actor_id'], 'name': exc['owner_name']} if exc['owner_actor_id'] else None,
                     'refs': refs, **pair(en, es, 'summary')})
        by_kind.setdefault(exc['kind'], []).append(exc['due_at'])
    parts_en, parts_es = [], []
    for kind, dues in by_kind.items():
        labels = _KINDS.get(kind, (kind.lower().replace('_', ' '),) * 4)
        n = len(dues)
        noun_en = labels[0] if n == 1 else labels[1]
        noun_es = labels[2] if n == 1 else labels[3]
        dated = sorted(d for d in dues if d)
        tail_en = tail_es = ''
        if dated:
            earliest = dated[0]
            m = sum(1 for d in dated if plant(d, api).date() == plant(earliest, api).date())
            tail_en = f' (due {short_date(earliest, "en")})' if n == 1 else f' ({m} due {short_date(earliest, "en")})'
            tail_es = f' (vence {short_date(earliest, "es")})' if n == 1 else f' ({m} vence{"n" if m > 1 else ""} {short_date(earliest, "es")})'
        parts_en.append(f'{n} {noun_en}{tail_en}')
        parts_es.append(f'{n} {noun_es}{tail_es}')
    return rows, ', '.join(parts_en), ', '.join(parts_es)


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------
def resolve_actor(cur, raw):
    """`actor` query param: id or exact name (the `GET /receipts?actor=` rule). Unknown → 422."""
    if raw is None or not str(raw).strip():
        return None
    raw = str(raw).strip()
    if raw.isdigit():
        cur.execute('SELECT id,name,role,active FROM actors WHERE id=%s', (int(raw),))
    else:
        cur.execute('SELECT id,name,role,active FROM actors WHERE name=%s ORDER BY id LIMIT 1', (raw,))
    row = cur.fetchone()
    if not row:
        fail(422, 'ACTOR_UNKNOWN', f'No actor matches {raw!r}.', f'Ningún actor coincide con {raw!r}.')
    return dict(row)


def confirmation_view(api, cur, catalog, row):
    actor = catalog.actor(row['actor_id'])
    confirmed = plant(row['confirmed_at'], api)
    return {'confirmation_id': row['id'], 'business_date': row['business_date'].isoformat(),
            'actor': {'id': actor['id'], 'name': actor['name'], 'role': actor['role']} if actor else {'id': row['actor_id'], 'name': None, 'role': None},
            'outcome': row['outcome'], 'notes': row['notes'], 'confirmed_at': confirmed.isoformat(),
            'confirmed_late': confirmed.date() != row['business_date'], 'receipts_seen': row['receipts_seen'],
            'summary_hash': row['summary_hash'],
            'exception': {'id': row['exception_id'], 'kind': 'SHIFT_DISCREPANCY'} if row['exception_id'] else None}


def build(api, cur, *, day, actor, caller, lang='en'):
    """The §1.2 document for (day, actor); `actor` None = all actors; `caller` = write_tickets.identity()."""
    catalog = Catalog(cur)
    terms = [actor['name'], str(actor['id'])] if actor else None
    # Committed tickets only (contract §2.1/§2.2): a rejected or held ticket may already hold a receipt number.
    numbers = wt.receipt_numbers(cur, day, actor_terms=terms, status='committed')
    loaded = shift_receipts.load(api, cur, numbers, catalog)
    action_of = {n: ticket['action'] for n, (_, ticket) in loaded.items()}
    ledger_numbers = [n for n in numbers if action_of.get(n) in ACTION_SECTION]
    other_numbers = [n for n in numbers if action_of.get(n) not in ACTION_SECTION]
    receipt_rows = [receipt_row(api, cur, catalog, number, day, receipt=loaded[number][0], ticket=loaded[number][1]) for number in ledger_numbers]
    other_rows = [other_receipt_row(api, cur, catalog, number, receipt=loaded[number][0]) for number in other_numbers]
    direct_makes = direct_make_substitutions(api, cur, day, actor)
    sections = []
    by_key = {}
    for key, label_en, label_es, empty_en, empty_es in SECTIONS:
        section = {'key': key, **pair(label_en, label_es, 'label'), **pair(empty_en, empty_es, 'empty'), 'count': 0, 'rows': []}
        sections.append(section)
        by_key[key] = section
    def instant(row):
        # Aware datetimes compare as instants; the local ISO text would mis-order the fall-back hour.
        return (datetime.fromisoformat(row['happened_at']), row['receipt_number'])
    for row in sorted(receipt_rows, key=instant):
        by_key[ACTION_SECTION[row['action']]]['rows'].append(row)
    other_rows.sort(key=instant)
    # §2.6: receipts ENTERED on `day` that happened on an earlier plant day. A receipt back-filled
    # on a later day shows on this day's summary (entered_on_later_day) but is not a late entry OF this day.
    late = sorted((r for r in receipt_rows if r['late_entry'] and r['entered_at'][:10] == day.isoformat()),
                  key=lambda r: (-r['hours_late'], r['receipt_number']))
    by_key['late_entries']['rows'] = late
    by_key['not_entered_yet']['rows'] = not_entered_yet(cur, catalog, day)
    by_key['not_entered_yet']['rows'].extend(kosher_holds(api, cur, catalog, day, actor['id'] if actor else None))
    by_key['lot_balances_touched']['rows'] = lot_balances(api, cur, catalog, receipt_rows)
    exc_rows, rollup_en, rollup_es = open_exceptions(api, cur, catalog, actor['id'] if actor else None)
    by_key['open_exceptions']['rows'] = exc_rows
    by_key['open_exceptions'].update(pair(rollup_en, rollup_es, 'rollup'))
    for section in sections:
        section['count'] = len(section['rows'])
    for row in receipt_rows:
        row.pop('_receipt', None)
    confirmation = {'status': 'unconfirmed', 'latest': None, 'can_confirm': True, **pair(None, None, 'reason')}
    if actor is None:
        confirmation.update(can_confirm=False, **pair('Sign in as a person to confirm a shift',
                                                      'Inicia sesión como persona para confirmar el turno', 'reason'))
    else:
        cur.execute('SELECT * FROM shift_confirmations WHERE business_date=%s AND actor_id=%s ORDER BY confirmed_at DESC,id DESC LIMIT 1',
                    (day, actor['id']))
        latest = cur.fetchone()
        if latest:
            confirmation.update(status=latest['outcome'], latest=confirmation_view(api, cur, catalog, latest))
        if caller.get('id') is None:
            confirmation.update(can_confirm=False, **pair('Sign in as a person to confirm a shift',
                                                          'Inicia sesión como persona para confirmar el turno', 'reason'))
    who_en = actor['name'] if actor else 'all'
    who_es = actor['name'] if actor else 'todos'
    document = {
        'date': day.isoformat(), 'plant_timezone': 'America/New_York', 'generated_at': api.get_plant_now().isoformat(),
        'actor': {'id': actor['id'], 'name': actor['name'], 'role': actor['role']} if actor else None, 'lang': lang,
        **pair(f'SHIFT SUMMARY — {title_date(day, "en")} — {who_en}', f'RESUMEN DEL TURNO — {title_date(day, "es")} — {who_es}', 'title'),
        'receipts_today': len(receipt_rows), 'summary_hash': None, 'confirmation': confirmation,
        # Additive (fix round 2026-10-13): the day's metadata receipts, so the GET /receipts set is fully
        # accounted for. Outside `receipts_today` and outside the hash (§1.3 covers the seven receipt sections).
        'other_receipts': {**pair('OTHER ENTRIES (orders, expected receipts, lot moves)',
                                  'OTRAS ENTRADAS (pedidos, recepciones esperadas, movimientos de lote)', 'label'),
                           **pair('none', 'ninguno', 'empty'), 'count': len(other_rows), 'rows': other_rows},
        'direct_make_substitutions': {
            **pair('DIRECT MAKES — OAT SUBSTITUTIONS', 'PRODUCCIÓN DIRECTA — SUSTITUCIONES CON AVENA', 'label'),
            'count': len(direct_makes), 'rows': direct_makes},
        'sections': sections,
    }
    document = localise(wt.json_value(api, document), lang)
    document['summary_hash'] = summary_hash(api, document['sections'], document['direct_make_substitutions']['rows'])
    return document



# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
class ConfirmRequest(BaseModel):
    date: date
    outcome: str
    notes: str = Field('', max_length=2000)
    summary_hash: str = Field(..., min_length=1)
    receipts_seen: int = Field(..., ge=0)
    lang: Literal['en', 'es'] = 'en'

    class Config:
        extra = 'forbid'


def register_routes(app, api):
    def target_actor(cur, caller, raw):
        if raw is not None and str(raw).strip().lower() == 'me':
            raw = None      # the F1 Assistant asks for `actor=me`: the caller's own summary
        requested = resolve_actor(cur, raw)
        if requested is None:
            return catalog_actor(cur, caller['id']) if caller.get('id') is not None else None
        if caller.get('id') is not None and caller.get('role') == 'floor' and requested['id'] != caller['id']:
            fail(403, 'ACTOR_NOT_ALLOWED', f'{caller.get("name")} may only view their own shift summary.',
                 f'{caller.get("name")} solo puede ver su propio resumen del turno.', actor=requested['name'])
        return requested

    def catalog_actor(cur, actor_id):
        cur.execute('SELECT id,name,role,active FROM actors WHERE id=%s', (actor_id,))
        return dict(cur.fetchone())

    def check_day(day):
        today = api.get_plant_now().date()
        day = day or today
        if day > today:
            fail(422, 'DATE_IN_FUTURE', f'{day.isoformat()} is after today\'s plant day ({today.isoformat()}).',
                 f'{day.isoformat()} es posterior al día de planta de hoy ({today.isoformat()}).')
        return day

    @app.get('/reports/shift-summary')
    def shift_summary(request: Request, date: Optional[date] = None, actor: Optional[str] = None,
                      lang: Literal['en', 'es'] = 'en', _: bool = Depends(api.verify_api_key)):
        caller = wt.identity(api, request)
        day = check_day(date)
        with api.get_transaction() as cur:
            target = target_actor(cur, caller, actor)
            return build(api, cur, day=day, actor=target, caller=caller, lang=lang)

    @app.post('/reports/shift-summary/confirm', status_code=201)
    def confirm_shift(body: ConfirmRequest, request: Request, _: bool = Depends(api.verify_api_key)):
        caller = wt.identity(api, request)
        if caller.get('id') is None:
            fail(403, 'ACTOR_REQUIRED', 'A shift is confirmed by a person; sign in with a personal key.',
                 'El turno lo confirma una persona; inicia sesión con tu clave personal.')
        if body.outcome not in ('match', 'discrepancy'):
            fail(422, 'OUTCOME_INVALID', "outcome must be 'match' or 'discrepancy'.", "outcome debe ser 'match' o 'discrepancy'.")
        notes = body.notes      # stored verbatim (§3.1); stripping is only for the blank check
        if body.outcome == 'discrepancy' and not notes.strip():
            fail(422, 'NOTES_REQUIRED', 'Say what is missing so the owner can act on it.',
                 'Indica qué falta para que el dueño pueda actuar.')
        day = check_day(body.date)
        with api.get_transaction() as cur:
            actor = catalog_actor(cur, caller['id'])
            # One person confirms serially: a second confirm for the same day waits for the first.
            cur.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (f'shift-confirm:{actor["id"]}:{day.isoformat()}',))
            fresh = build(api, cur, day=day, actor=actor, caller=caller, lang=body.lang)
            if fresh['summary_hash'] != body.summary_hash:
                fail(409, 'SHIFT_SUMMARY_STALE', 'The shift summary changed since it was shown; review the fresh one and confirm again.',
                     'El resumen del turno cambió desde que se mostró; revisa el nuevo y confirma otra vez.', summary=fresh)
            exception_id = None
            owner_id = permissions.owner_actor_id(cur)
            owner = Catalog(cur).actor(owner_id) if owner_id else None
            if body.outcome == 'discrepancy':
                # One open SHIFT_DISCREPANCY per (actor, day): a repeated or retried "something is
                # missing" appends its confirmation row but re-uses the open exception (its detail
                # keeps the latest notes/hash and counts the confirmations). Once the owner has
                # resolved it, a new discrepancy opens a new one. The advisory lock above makes
                # the select-then-insert atomic per (actor, day).
                cur.execute('''SELECT id, detail FROM exceptions WHERE kind='SHIFT_DISCREPANCY' AND status IN ('open','escalated')
                               AND detail->>'actor_id'=%s AND detail->>'business_date'=%s ORDER BY id LIMIT 1 FOR UPDATE''',
                            (str(actor['id']), day.isoformat()))
                existing = cur.fetchone()
                if existing:
                    exception_id = existing['id']
                    cur.execute('UPDATE exceptions SET detail = detail || %s::jsonb WHERE id=%s',
                                (Json({'notes': notes, 'receipts_seen': body.receipts_seen, 'summary_hash': body.summary_hash,
                                       'reconfirmed_at': api.get_plant_now().isoformat(),
                                       'confirmations': int((existing['detail'] or {}).get('confirmations', 1)) + 1}), exception_id))
                else:
                    cur.execute('''INSERT INTO exceptions(kind,status,severity,detail,owner_actor_id,opened_at,due_at)
                                   VALUES ('SHIFT_DISCREPANCY','open','warn',%s,%s,clock_timestamp(),NULL) RETURNING id''',
                                (Json({'business_date': day.isoformat(), 'actor_id': actor['id'], 'actor_name': actor['name'],
                                       'notes': notes, 'receipts_seen': body.receipts_seen, 'summary_hash': body.summary_hash}), owner_id))
                    exception_id = cur.fetchone()['id']
            cur.execute('''INSERT INTO shift_confirmations(business_date,actor_id,outcome,notes,receipts_seen,summary_hash,summary_snapshot,exception_id)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *''',
                        (day, actor['id'], body.outcome, notes, body.receipts_seen, body.summary_hash, Json(fresh['sections']), exception_id))
            row = cur.fetchone()
            view = confirmation_view(api, cur, Catalog(cur), row)
            owner_en = owner['name'] if owner else 'The owner'
            owner_es = owner['name'] if owner else 'El dueño'
            if body.outcome == 'discrepancy':
                msg = (f'Shift recorded as: something is missing. {owner_en} will see this on the weekly view.',
                       f'Turno registrado como: falta algo. {owner_es} lo verá en la vista semanal.')
            elif notes.strip():
                msg = ('Shift recorded as: matches the floor, with notes.', 'Turno registrado como: coincide con el piso, con notas.')
            else:
                msg = ('Shift recorded as: matches the floor.', 'Turno registrado como: coincide con el piso.')
            return localise(wt.json_value(api, {**view, **pair(msg[0], msg[1], 'message')}), body.lang)
