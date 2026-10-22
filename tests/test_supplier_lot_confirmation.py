"""Supplier labels, explicit physical choices and exact splits through real tickets."""
from pathlib import Path

import pytest

from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_write_tickets_part2 import items, body  # noqa: F401
from tests.test_write_tickets import headers, commit, error, posted_count  # noqa: F401
from tests.test_lot_confirmation import prepare, evidence, substitute, sub_payload  # noqa: F401

pytestmark = pytest.mark.db


def stocked_lot(cur, product_id, code, *, supplier_code=None, pounds=100, notes=None):
    cur.execute('''INSERT INTO lots(product_id,lot_code,supplier_lot_code,entry_source_notes)
        VALUES (%s,%s,%s,%s) RETURNING id''', (product_id, code, supplier_code, notes))
    lid = cur.fetchone()['id']
    cur.execute("INSERT INTO transactions(type) VALUES ('receive') RETURNING id")
    tid = cur.fetchone()['id']
    cur.execute('''INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb)
        VALUES (%s,%s,%s,%s)''', (tid, product_id, lid, pounds))
    return lid


def supplier_evidence(lid, code, *, selected=False):
    return {'lot_id': lid, 'method': 'supplier_lot', 'value': code, 'selected': selected}


def allocation(pid, lid, pounds):
    return {'ingredient_product_id': pid, 'lot_id': lid, 'quantity_lb': pounds}


@pytest.mark.parametrize('alias', [False, True])
def test_supplier_code_case_and_all_spaces_match_and_persist(client, db_cursor, items, alias):
    item = items['ingredient']
    if alias:
        db_cursor.execute('INSERT INTO lot_supplier_codes(lot_id,supplier_lot_code,supplier_name) VALUES (%s,%s,%s)',
                          (item['lot_id'], 'SUP A 123', 'The supplier'))
    else:
        db_cursor.execute('UPDATE lots SET supplier_lot_code=%s WHERE id=%s', ('SUP A 123', item['lot_id']))
    # An out-of-stock twin and a different product never create ambiguity.
    stocked_lot(db_cursor, item['id'], 'SUP-ZERO', supplier_code='SUPA123', pounds=0)
    stocked_lot(db_cursor, items['batch']['id'], 'SUP-OTHER-PRODUCT', supplier_code='SUPA123')
    prepared = prepare(client, 'make', body('make', items) | {
        'lot_confirmations': [supplier_evidence(item['lot_id'], ' sUp\t a 123 ')]})
    assert prepared['can_commit']
    result = commit(client, prepared)
    assert result.status_code == 200, result.text
    db_cursor.execute('SELECT method,value FROM transaction_lot_confirmations WHERE transaction_id=%s',
                      (result.json()['transaction_id'],))
    assert db_cursor.fetchone()['method'] == 'supplier_lot'


def test_ambiguous_supplier_code_needs_active_choice(client, db_cursor, items):
    item = items['ingredient']
    db_cursor.execute("UPDATE lots SET supplier_lot_code='SHARED-1' WHERE id=%s", (item['lot_id'],))
    second = stocked_lot(db_cursor, item['id'], 'SUP-SECOND', supplier_code='shared -1')
    prepared = prepare(client, 'make', body('make', items))
    response = commit(client, prepared, lot_confirmations=[supplier_evidence(item['lot_id'], 'SHARED-1')])
    error(response, 422, 'AMBIGUOUS_SUPPLIER_LOT')
    assert {row['lot_id'] for row in response.json()['detail']['candidates']} == {item['lot_id'], second}
    assert posted_count(db_cursor, prepared) == 0
    response = commit(client, prepared, lot_confirmations=[supplier_evidence(item['lot_id'], 'SHARED-1', selected=True)])
    assert response.status_code == 200, response.text


def test_selected_cannot_bypass_supplier_mismatch(client, db_cursor, items):
    item = items['ingredient']
    stocked_lot(db_cursor, item['id'], 'CORRECT-CODE-OTHER-LOT', supplier_code='CORRECT')
    prepared = prepare(client, 'make', body('make', items))
    error(commit(client, prepared, lot_confirmations=[supplier_evidence(item['lot_id'], 'CORRECT', selected=True)]),
          422, 'LOT_CONFIRMATION_MISMATCH')
    assert posted_count(db_cursor, prepared) == 0


def test_missing_supplier_lot_bilingual_guidance_lists_available_lots(client, db_cursor, items):
    prepared = prepare(client, 'make', body('make', items))
    response = commit(client, prepared, lot_confirmations=[supplier_evidence(items['ingredient']['lot_id'], 'MISSING')])
    error(response, 422, 'SUPPLIER_LOT_NOT_IN_STOCK')
    detail = response.json()['detail']
    assert detail['message'] == "That supplier lot isn't in stock in FL — if it's old stock already here on Oct 21, choose the OPENING STOCK lot"
    assert '21 de octubre' in detail['message_es']
    assert detail['candidates'][0]['lot_id'] == items['ingredient']['lot_id']
    assert posted_count(db_cursor, prepared) == 0


@pytest.mark.parametrize('ambiguous', [False, True])
def test_invalid_supplier_prepare_preserves_choices_without_confirming(client, db_cursor, items, ambiguous):
    item = items['ingredient']
    if ambiguous:
        db_cursor.execute("UPDATE lots SET supplier_lot_code='DUP-BAG' WHERE id=%s", (item['lot_id'],))
        stocked_lot(db_cursor, item['id'], 'DUP-BAG-SECOND', supplier_code='dup -bag')
    prepared = prepare(client, 'make', body('make', items) | {
        'lot_confirmations': [supplier_evidence(item['lot_id'], 'DUP-BAG' if ambiguous else 'NO-SUCH-BAG')]})
    assert not prepared['can_commit']
    code = 'AMBIGUOUS_SUPPLIER_LOT' if ambiguous else 'SUPPLIER_LOT_NOT_IN_STOCK'
    assert prepared['blockers'][0]['code'] == code
    choices = prepared['draft']['lot_confirmation_group']['ingredients'][0]['candidates']
    assert choices and choices[0]['lot_id'] == item['lot_id']
    assert len(choices) == (2 if ambiguous else 1)
    assert not prepared['draft']['input_plan'][0]['confirmed']
    error(commit(client, prepared), 422, code)
    assert posted_count(db_cursor, prepared) == 0


def test_opening_lot_first_labeled_in_both_languages_never_auto_confirmed(client, db_cursor, items):
    item = items['ingredient']
    opening = stocked_lot(db_cursor, item['id'], 'OPEN-20261021', notes='Opening stock — counted 2026-10-21')
    prepared = prepare(client, 'make', body('make', items))
    group = prepared['draft']['lot_confirmation_group']['ingredients'][0]
    assert group['candidates'][0]['lot_id'] == opening
    assert group['candidates'][0]['label_en'] == 'OPENING STOCK — use for any bag already here on Oct 21'
    assert 'EXISTENCIAS INICIALES' in group['candidates'][0]['label_es']
    assert group['candidates'][1]['fifo_suggested']
    assert not prepared['can_commit']
    assert not prepared['draft']['input_plan'][0]['confirmed']
    chosen = body('make', items) | {'ingredient_lot_allocations': [allocation(item['id'], opening, 10)],
        'lot_confirmations': [supplier_evidence(opening, 'OPENING STOCK', selected=True)]}
    prepared = prepare(client, 'make', chosen)
    assert prepared['can_commit']
    assert commit(client, prepared).status_code == 200


def test_opening_alias_needs_explicit_row_choice(client, db_cursor, items):
    item = items['ingredient']
    db_cursor.execute("UPDATE lots SET entry_source_notes='Opening stock reset' WHERE id=%s", (item['lot_id'],))
    prepared = prepare(client, 'make', body('make', items))
    error(commit(client, prepared, lot_confirmations=[supplier_evidence(item['lot_id'], 'OPENING STOCK')]),
          422, 'SUPPLIER_LOT_NOT_IN_STOCK')


def test_two_selected_make_lots_debit_exact_split(client, db_cursor, items):
    item = items['ingredient']
    second = stocked_lot(db_cursor, item['id'], 'SPLIT-MAKE-2', supplier_code='BAG-2')
    db_cursor.execute("UPDATE lots SET supplier_lot_code='BAG-1' WHERE id=%s", (item['lot_id'],))
    prepared = prepare(client, 'make', body('make', items) | {
        'ingredient_lot_allocations': [allocation(item['id'], item['lot_id'], 4), allocation(item['id'], second, 6)],
        'lot_confirmations': [supplier_evidence(item['lot_id'], 'BAG-1', selected=True),
                              supplier_evidence(second, 'BAG-2', selected=True)]})
    assert prepared['can_commit']
    result = commit(client, prepared)
    assert result.status_code == 200, result.text
    db_cursor.execute('SELECT lot_id,quantity_lb FROM transaction_lines WHERE transaction_id=%s AND quantity_lb<0 ORDER BY lot_id',
                      (result.json()['transaction_id'],))
    assert [(row['lot_id'], float(row['quantity_lb'])) for row in db_cursor.fetchall()] == [(item['lot_id'], -4), (second, -6)]


def test_lot_card_retains_ledger_precision_for_exact_split(client, db_cursor, items):
    item = items['ingredient']
    db_cursor.execute('UPDATE batch_formulas SET quantity_lb=10.0003 WHERE product_id=%s AND ingredient_product_id=%s',
                      (items['batch']['id'], item['id']))
    second = stocked_lot(db_cursor, item['id'], 'PRECISE-SPLIT-2')
    prepared = prepare(client, 'make', body('make', items))
    assert prepared['draft']['lot_confirmation_group']['ingredients'][0]['needed_lb'] == 10.0003
    prepared = prepare(client, 'make', body('make', items) | {
        'ingredient_lot_allocations': [allocation(item['id'], item['lot_id'], 4.0001),
                                       allocation(item['id'], second, 6.0002)]})
    confirmations = [evidence(row, 'full_code') for row in prepared['draft']['input_plan']]
    response = commit(client, prepared, lot_confirmations=confirmations)
    assert response.status_code == 200, response.text
    db_cursor.execute('SELECT sum(quantity_lb) AS qty FROM transaction_lines WHERE transaction_id=%s AND product_id=%s',
                      (response.json()['transaction_id'], item['id']))
    assert float(db_cursor.fetchone()['qty']) == -10.0003


@pytest.mark.parametrize('action', ['make', 'pack'])
def test_selected_split_must_add_up_server_side(client, db_cursor, items, action):
    item = items['ingredient' if action == 'make' else 'batch']
    second = stocked_lot(db_cursor, item['id'], 'SPLIT-WRONG-TOTAL')
    response = client.post('/' + action + '/prepare', headers=headers(), json=body(action, items) | {
        'ingredient_lot_allocations': [allocation(item['id'], item['lot_id'], 4), allocation(item['id'], second, 5)]})
    assert response.status_code == 200, response.text
    assert not response.json()['can_commit']
    assert response.json()['blockers'][0]['code'] in ('LOT_ALLOCATION_TOTAL_MISMATCH', 'INPUT_PLAN_CHANGED')


def test_pack_base_and_addin_can_each_split_and_need_every_confirmation(client, db_cursor, items):
    source, ingredient = items['batch'], items['ingredient']
    db_cursor.execute("INSERT INTO products(name,type,uom) VALUES ('Supplier split intermediate','batch','lb') RETURNING id")
    intermediate = db_cursor.fetchone()['id']
    db_cursor.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s', (intermediate, items['finished']['id']))
    for pid, qty in [(source['id'], 10), (ingredient['id'], 2)]:
        db_cursor.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,%s)',
                          (intermediate, pid, qty))
    source2 = stocked_lot(db_cursor, source['id'], 'SPLIT-PACK-SOURCE')
    ingredient2 = stocked_lot(db_cursor, ingredient['id'], 'SPLIT-PACK-ADDIN')
    payload = body('pack', items) | {'ingredient_lot_allocations': [
        allocation(source['id'], source['lot_id'], 4), allocation(source['id'], source2, 6),
        allocation(ingredient['id'], ingredient['lot_id'], .75), allocation(ingredient['id'], ingredient2, 1.25)]}
    prepared = prepare(client, 'pack', payload)
    groups = prepared['draft']['lot_confirmation_group']['ingredients']
    assert {row['product_id'] for row in groups} == {source['id'], ingredient['id']}
    assert len(prepared['draft']['input_plan']) == 4
    confirmations = [evidence(row, 'full_code') for row in prepared['draft']['input_plan']]
    error(commit(client, prepared, lot_confirmations=confirmations[:3]), 422, 'LOT_NOT_CONFIRMED')
    result = commit(client, prepared, lot_confirmations=confirmations)
    assert result.status_code == 200, result.text
    db_cursor.execute('SELECT lot_id,quantity_lb FROM transaction_lines WHERE transaction_id=%s AND quantity_lb<0 ORDER BY lot_id',
                      (result.json()['transaction_id'],))
    actual = {row['lot_id']: float(row['quantity_lb']) for row in db_cursor.fetchall()}
    assert actual == {source['lot_id']: -4, source2: -6, ingredient['lot_id']: -.75, ingredient2: -1.25}


@pytest.mark.parametrize('case', ['wrong_product', 'empty', 'duplicate'])
def test_invalid_selected_lots_cannot_prepare(client, db_cursor, items, case):
    item = items['ingredient']
    lid = (items['batch']['lot_id'] if case == 'wrong_product' else
           stocked_lot(db_cursor, item['id'], 'NO-STOCK', pounds=0) if case == 'empty' else item['lot_id'])
    rows = [allocation(item['id'], lid, 10)]
    if case == 'duplicate':
        rows = [allocation(item['id'], lid, 4), allocation(item['id'], lid, 6)]
    response = client.post('/make/prepare', headers=headers(), json=body('make', items) | {'ingredient_lot_allocations': rows})
    assert response.status_code == 200, response.text
    assert not response.json()['can_commit']
    assert response.json()['blockers'][0]['code'] == {
        'wrong_product': 'LOT_NOT_FOUND', 'empty': 'LOT_NOT_IN_STOCK', 'duplicate': 'DUPLICATE_LOT_SELECTION'}[case]


@pytest.mark.parametrize('split', [False, True])
def test_substitution_audit_uses_actual_selected_lots(client, db_cursor, items, substitute, split):
    selected = stocked_lot(db_cursor, substitute['id'], 'ACTUAL-SUB-B')
    second = stocked_lot(db_cursor, substitute['id'], 'ACTUAL-SUB-C') if split else None
    quantities = [(selected, 4), (second, 6)] if split else [(selected, 10)]
    sub = sub_payload(items) | {'substitute_product_id': substitute['id'], 'lot_id': substitute['lot_id']}
    prepared = prepare(client, 'make', body('make', items) | {'substitutions': [sub],
        'ingredient_lot_allocations': [allocation(substitute['id'], lid, pounds) for lid, pounds in quantities]})
    actual = prepared['draft']['substitutions'][0]
    assert actual['lot_id'] == selected and actual['lot_id'] != substitute['lot_id']
    assert [(row['lot_id'], row['quantity_lb']) for row in actual['actual_lot_allocations']] == quantities
    result = commit(client, prepared, lot_confirmations=[evidence(row, 'full_code') for row in prepared['draft']['input_plan']])
    assert result.status_code == 200, result.text
    receipt = result.json()
    assert receipt['substitutions'][0]['lot_id'] == selected
    db_cursor.execute('SELECT lot_id,actual_lot_allocations FROM transaction_substitutions WHERE transaction_id=%s',
                      (receipt['transaction_id'],))
    audit = db_cursor.fetchone()
    assert audit['lot_id'] == selected
    assert [(row['lot_id'], row['quantity_lb']) for row in audit['actual_lot_allocations']] == quantities
    db_cursor.execute('SELECT lot_id,quantity_lb FROM transaction_lines WHERE transaction_id=%s AND quantity_lb<0 ORDER BY lot_id',
                      (receipt['transaction_id'],))
    assert [(row['lot_id'], float(-row['quantity_lb'])) for row in db_cursor.fetchall()] == quantities


def test_shared_substitution_product_does_not_double_count_audit_pounds(client, db_cursor, items, substitute):
    db_cursor.execute("INSERT INTO products(name,type,uom) VALUES ('Second manual replacement original','ingredient','lb') RETURNING id")
    original2 = db_cursor.fetchone()['id']
    for pid, pounds in [(substitute['id'], 2), (original2, 3)]:
        db_cursor.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,%s)',
                          (items['batch']['id'], pid, pounds))
    selected = stocked_lot(db_cursor, substitute['id'], 'SHARED-ACTUAL-B')
    second = stocked_lot(db_cursor, substitute['id'], 'SHARED-ACTUAL-C')
    sub = sub_payload(items) | {'substitute_product_id': substitute['id'], 'lot_id': substitute['lot_id']}
    prepared = prepare(client, 'make', body('make', items) | {'substitutions': [sub, dict(sub, ingredient_product_id=original2)],
        'ingredient_lot_allocations': [allocation(substitute['id'], selected, 7), allocation(substitute['id'], second, 8)]})
    result = commit(client, prepared, lot_confirmations=[evidence(row, 'full_code') for row in prepared['draft']['input_plan']])
    assert result.status_code == 200, result.text
    db_cursor.execute('SELECT ingredient_product_id,actual_lot_allocations FROM transaction_substitutions WHERE transaction_id=%s',
                      (result.json()['transaction_id'],))
    audit = {row['ingredient_product_id']: row['actual_lot_allocations'] for row in db_cursor.fetchall()}
    assert [(row['lot_id'], row['quantity_lb']) for row in audit[items['ingredient']['id']]] == [(selected, 5), (second, 5)]
    assert [(row['lot_id'], row['quantity_lb']) for row in audit[original2]] == [(second, 3)]
    assert sum(row['quantity_lb'] for rows in audit.values() for row in rows) == 13


def test_supplier_evidence_migration_is_idempotent(db_cursor):
    sql = (Path(__file__).parents[1] / 'migrations/087_supplier_lot_confirmation.sql').read_text()
    db_cursor.execute(sql)
    db_cursor.execute(sql)
    db_cursor.execute("SELECT count(*) AS n FROM migration_markers WHERE name='087_supplier_lot_confirmation'")
    assert db_cursor.fetchone()['n'] == 1
    db_cursor.execute('''SELECT count(*) AS n FROM information_schema.columns
        WHERE table_schema='public' AND table_name='transaction_substitutions' AND column_name='actual_lot_allocations' ''')
    assert db_cursor.fetchone()['n'] == 1


def test_supplier_migration_preserves_confirmation_methods_from_other_branches(db_cursor):
    db_cursor.execute('''ALTER TABLE transaction_lot_confirmations
        DROP CONSTRAINT transaction_lot_confirmations_method_check;
        ALTER TABLE transaction_lot_confirmations ADD CONSTRAINT transaction_lot_confirmations_method_check
        CHECK (method IN ('last4','full_code','scan','pallet','tap'))''')
    sql = (Path(__file__).parents[1] / 'migrations/087_supplier_lot_confirmation.sql').read_text()
    db_cursor.execute(sql)
    db_cursor.execute(sql)
    db_cursor.execute('''SELECT pg_get_constraintdef(oid) AS definition FROM pg_constraint
        WHERE conrelid='transaction_lot_confirmations'::regclass
          AND conname='transaction_lot_confirmations_method_check' ''')
    definition = db_cursor.fetchone()['definition']
    assert "'tap'" in definition and "'supplier_lot'" in definition
    for method in ('last4', 'full_code', 'scan', 'pallet'):
        assert "'" + method + "'" in definition
