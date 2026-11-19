"""Migration 098 guards. Call while holding the posting lot locks.

Balances and receipt pack sizes come exclusively from the effective ledger.
Allowance usage is deliberately lifetime usage, including subsequently voided counts.
"""
import math
from decimal import Decimal

from fastapi import HTTPException
import lot_confirmation

MARKER = '098_no_stock_by_hand'


def live_at(cur):
    cur.execute('SELECT applied_at FROM migration_markers WHERE name=%s', (MARKER,))
    row = cur.fetchone()
    return row['applied_at'] if row else None


def fail(code, en, es, status=422):
    raise HTTPException(status, dict(error_code=code, message=en, message_es=es))


def receive(cur, req, product_id, lot_id=None):
    if not live_at(cur):
        return
    if not (req.bol_reference or '').strip():
        fail('BOL_REQUIRED', 'Enter the BOL or delivery reference.', 'Ingresa el BOL o la referencia de entrega.')
    if not math.isfinite(req.cases) or not math.isfinite(req.case_size_lb) or req.cases <= 0 or req.case_size_lb <= 0:
        fail('INVALID_QUANTITY', 'Cases and pounds per case must be above zero.', 'Los cajones y las libras por cajón deben ser mayores que cero.')
    if lot_id and lot_confirmation.supplier_lot_identity_applies(cur, product_id):
        cur.execute("""SELECT 1 FROM ledger_current_transaction_lines l
            JOIN ledger_current_transactions t ON t.id=l.transaction_id
            WHERE l.lot_id=%s AND l.quantity_lb>0 AND t.effective_status='posted' LIMIT 1""", (lot_id,))
        if cur.fetchone():
            fail('LOT_ALREADY_HAS_STOCK', 'Every delivery gets its own lot. Leave the FL lot code blank and FL makes a new one.',
                 'Cada entrega tiene su propio lote. Deja vacío el código de lote FL y FL crea uno nuevo.', 409)


def require_paired_move(cur, product_id, delta, reason, *, paired=False):
    """The wrong-lot rule also covers a new Found lot, unlike the give-back cap."""
    reason = '_'.join(str(reason or '').lower().split())
    if reason != 'wrong_lot' or paired or Decimal(str(delta)) <= 0:
        return
    if live_at(cur) and lot_confirmation.supplier_lot_identity_applies(cur, product_id):
        fail('PAIRED_MOVE_REQUIRED', 'Choose the lot that was really used on the Adjust screen. FL moves the pounds between the two lots.',
             'Elige el lote que se usó realmente en Ajustar. FL mueve las libras entre los dos lotes.')


def addition(cur, product_id, lot_id, delta, *, count_to=None, reason=None, paired=False):
    """Return the allowance pounds to record atomically with the posting, or zero."""
    at = live_at(cur)
    delta = Decimal(str(delta))
    if not at or delta <= 0 or not lot_confirmation.supplier_lot_identity_applies(cur, product_id):
        return Decimal(0)
    if reason == 'missing_production':
        fail('NO_STOCK_BY_HAND', 'Purchased ingredients must be received, not made or packed.',
             'Los ingredientes comprados se reciben; no se producen ni se empacan.')
    if reason == 'wrong_lot':
        if paired:
            return Decimal(0)
        require_paired_move(cur, product_id, delta, reason)
    cur.execute("""SELECT COALESCE(sum(l.quantity_lb),0) AS lb
        FROM ledger_current_transaction_lines l JOIN ledger_current_transactions t ON t.id=l.transaction_id
        WHERE l.lot_id=%s AND t.effective_status='posted' AND t.type='adjust' AND t.created_at >= %s
          AND NOT EXISTS (SELECT 1 FROM paired_lot_moves m WHERE m.transaction_id=t.id)""", (lot_id, at))
    if Decimal(str(cur.fetchone()['lb'])) + delta <= 0:
        return Decimal(0)
    if count_to is not None and Decimal(str(count_to)) == 0:
        cur.execute("""SELECT COALESCE(sum(l.quantity_lb),0) AS lb FROM ledger_current_transaction_lines l
            JOIN ledger_current_transactions t ON t.id=l.transaction_id
            WHERE l.lot_id=%s AND t.effective_status='posted'""", (lot_id,))
        book = Decimal(str(cur.fetchone()['lb']))
        cur.execute("""SELECT min((t.effective_record->>'case_size_lb')::numeric) AS pack,
                bool_and((t.effective_record->>'case_size_lb') IS NOT NULL AND (t.effective_record->>'case_size_lb')::numeric>0) AS valid
            FROM ledger_current_transactions t JOIN ledger_current_transaction_lines l ON l.transaction_id=t.id
            WHERE l.lot_id=%s AND t.type='receive' AND t.effective_status='posted' AND l.quantity_lb>0""", (lot_id,))
        receipt = cur.fetchone()
        cur.execute('SELECT COALESCE(sum(lb),0) AS lb FROM under_pack_counts WHERE lot_id=%s AND entered_at >= %s', (lot_id, at))
        used = Decimal(str(cur.fetchone()['lb']))
        if receipt['valid'] and receipt['pack'] is not None:
            limit = min(Decimal(str(receipt['pack'])), Decimal(100))
            if book < 0 and delta == -book and -book < limit and used + delta < limit:
                return delta
    en = 'FL cannot add new stock to this lot by hand. For a delivery use Receive; without paperwork use Found, which makes a new lot.'
    es = 'FL no puede agregar inventario nuevo a este lote a mano. Para una entrega usa Recibir; sin documentos usa Encontrado, que crea un lote nuevo.'
    if count_to is not None:
        en += ' A lot short by less than one bag can be counted at zero, within its lifetime allowance. Otherwise: record a paired wrong-lot move from the lot really used (Receive it first if needed), then close as missing_movement naming its receipt; or void the wrong entry and enter it again (voided); or close as unverified with a note, without changing stock or the book balance.'
        es += ' Un lote con faltante menor de una bolsa puede contarse en cero, dentro de su límite acumulado. Si no: registra el movimiento pareado desde el lote que se usó (Recíbelo primero si hace falta) y cierra como missing_movement indicando su recibo; o anula la entrada equivocada y regístrala de nuevo (voided); o cierra como unverified con una nota, sin cambiar inventario ni saldo.'
    if reason == 'data_entry_error':
        en += ' To fix a typing mistake, void the wrong entry and enter it again.'
        es += ' Para corregir un error de captura, anula la entrada equivocada y regístrala de nuevo.'
    fail('NO_STOCK_BY_HAND', en, es)


def validate_pair(api, cur, product, restored_id, used_id, delta, reason, note):
    if restored_id == used_id or delta <= 0 or reason != 'wrong_lot' or not (note or '').strip():
        fail('INVALID_PAIRED_MOVE', 'Choose two different lots, positive pounds and a note for a wrong-lot move.',
             'Elige dos lotes distintos, libras positivas y una nota para mover entre lotes.')
    cur.execute('SELECT id,product_id,lot_code,status FROM lots WHERE id=ANY(%s) ORDER BY id', ([restored_id, used_id],))
    rows = {r['id']: r for r in cur.fetchall()}
    if len(rows) != 2 or any(r['status'] == 'merged' for r in rows.values()):
        fail('INVALID_PAIRED_MOVE', 'Both lots must exist and neither may be merged.', 'Ambos lotes deben existir y ninguno puede estar fusionado.')
    if any(r['product_id'] != product['id'] for r in rows.values()):
        fail('WRONG_LOT_SAME_PRODUCT', 'A wrong-lot move is only between two lots of the same product.',
             'El movimiento es solo entre dos lotes del mismo producto.')
    if not lot_confirmation.supplier_lot_identity_applies(cur, product['id']):
        fail('INVALID_PAIRED_MOVE', 'Paired moves are for purchased ingredients.', 'Los movimientos pareados son para ingredientes comprados.')
    import kosher_tiers
    for lot_id in (restored_id, used_id):
        kosher_tiers.output_matches(cur, lot_id, product)
    warning = api.check_private_label_merge(product['name'], product.get('label_type') or 'house', reason, -delta)
    if warning:
        fail('PRIVATE_LABEL_MERGE', warning, warning)
    return dict(rows[used_id], on_hand_lb=api.lot_on_hand(cur, used_id))
