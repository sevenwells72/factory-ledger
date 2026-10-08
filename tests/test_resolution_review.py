"""Regression coverage for both independent PR #81 reviews."""
from decimal import Decimal

import pytest

import resolution as r
from tests.test_resolution import MIGRATION, alias, assert_no_pick, catalog, client, resolve


def lot(cur, product, code, quantity=0, supplier_code=None, days_ago=0):
    cur.execute('INSERT INTO lots(product_id,lot_code,supplier_lot_code) VALUES (%s,%s,%s) RETURNING id',
                (product, code, supplier_code))
    lot_id = cur.fetchone()['id']
    transaction_id = None
    if quantity:
        cur.execute("INSERT INTO transactions(type,occurred_at) VALUES ('receive', CURRENT_TIMESTAMP - %s * interval '1 day') RETURNING id", (days_ago,))
        transaction_id = cur.fetchone()['id']
        cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,%s)',
                    (transaction_id, product, lot_id, quantity))
    return lot_id, transaction_id


@pytest.mark.parametrize('balance', [0, -5])
def test_exact_ineligible_lot_never_substitutes_similar_stocked_lot(catalog, balance):
    cur, product = catalog['cur'], catalog['classic'][0]
    empty, _ = lot(cur, product, 'REVIEW-20261008-1001', balance)
    lot(cur, product, 'REVIEW-20261008-1002', 10)
    result = resolve(catalog, 'lot', 'REVIEW-20261008-1001', {'product_id': product, 'action': 'ship'})
    assert_no_pick(result, 'none')
    assert 'Exact lot exists' in result['note'] and 'balance' in result['note']
    assert resolve(catalog, 'lot', 'REVIEW-20261008-1001', {'product_id': product, 'action': 'adjust'})['match']['id'] == empty


@pytest.mark.parametrize('spelling,expansion,wrong', [
    ('CLS', 'Classic', 'CLSX Specialty Mix'),
    ('choc', 'chocolate', 'Chock Specialty Mix'),
    ('house cereal', 'Classic', 'House CerealX Specialty Mix'),
])
def test_aliases_of_every_length_cannot_match_substrings(catalog, spelling, expansion, wrong):
    alias(catalog, text=spelling, expansion=expansion)
    catalog['product'](wrong)
    result = resolve(catalog, 'product', spelling + ' Specialty')
    assert_no_pick(result, 'none')
    good = catalog['product'](expansion + ' Specialty Mix')
    assert resolve(catalog, 'product', spelling + ' Specialty')['match']['id'] == good


@pytest.mark.parametrize('sentinel', r.SENTINEL_SUPPLIERS)
@pytest.mark.parametrize('padding', ['\t{}\t', '\n{}\n', ' \t{} \n '])
def test_all_pseudo_suppliers_excluded_after_whitespace_normalization(catalog, sentinel, padding):
    cur = catalog['cur']
    name = padding.format(sentinel.upper().replace(' ', ' \t\n '))
    cur.execute('INSERT INTO suppliers(name,active) VALUES (%s,true) RETURNING id', (name,))
    pseudo = cur.fetchone()['id']
    alias(catalog, kind='supplier', text='Fake Vendor', expansion=None, supplier_id=pseudo)
    assert_no_pick(resolve(catalog, 'supplier', sentinel), 'none')
    assert_no_pick(resolve(catalog, 'supplier', 'Fake Vendor'), 'none')


@pytest.mark.parametrize('uom,weight,expected', [
    ('50 lb bag', 50, True), ('50 LB BAGS', 50, True), ('bags', 25, True),
    ('50 lb bag', 25, False), ('bags', None, False), ('bags', 0, False), ('lb', 50, False),
])
def test_ingredient_bags_require_catalog_uom_and_consistent_positive_weight(catalog, uom, weight, expected):
    product = catalog['product']('Review Bulk Ingredient', uom=uom, case_size_lb=weight)
    result = resolve(catalog, 'unit', 'bags', {'product_id': product}, quantity=2)
    assert (result['match'] is not None) == expected
    if expected:
        assert result['draft'] == {'product_id': product, 'quantity': Decimal(2), 'unit': 'bags'}


@pytest.mark.parametrize('multiple_codes', [False, True])
def test_literal_supplier_lot_suffix_resolves_exactly(catalog, multiple_codes):
    cur, product = catalog['cur'], catalog['classic'][0]
    wanted, _ = lot(cur, product, 'REVIEW-INTERNAL-2001', 10,
                    supplier_code=None if multiple_codes else 'VENDOR-9988 LOT')
    if multiple_codes:
        cur.execute('INSERT INTO lot_supplier_codes(lot_id,supplier_lot_code) VALUES (%s,%s)',
                    (wanted, 'VENDOR-9988 LOT'))
    result = resolve(catalog, 'lot', ' vendor-9988   LOT ', {'product_id': product})
    assert result['match']['id'] == wanted and result['match']['tier'] == 'exact'


def test_bulk_name_limits_validated_before_resolution(client, monkeypatch):
    monkeypatch.setattr(r, 'resolve_bulk_product', lambda *_: pytest.fail('invalid body reached resolver'))
    response = client.post('/products/resolve', json={'names': ['Classic', 'x' * 501]}, headers={'X-API-Key': 'a4-master'})
    assert response.status_code == 422


def test_seeded_aliases_and_both_classic_nine_tiers(catalog):
    cur = catalog['cur']
    cur.execute(MIGRATION.read_text())
    rows, _ = r.read_aliases(cur)
    assert {row['alias']: row['expansion'] for row in rows} == {
        'SS': 'Sunshine', 'BS': 'Blue Stripes', 'CLS': 'Classic', 'choc': 'chocolate', '#9': '9'}
    before = rows
    cur.execute(MIGRATION.read_text())
    assert r.read_aliases(cur)[0] == before
    regular = [catalog['product']('Batch Classic Granola #9', 'batch'),
               catalog['product']('Batch Classic Chocolate Chip Granola #9', 'batch')]
    regular.append(catalog['product']('Regular Classic #9 Finished', 'finished', parent_batch_product_id=regular[0]))
    all_ids = set(regular + catalog['batches'] + catalog['finished'])
    for query in ('#9', '9', 'Classic 9'):
        result = resolve(catalog, 'product', query, limit=25)
        assert_no_pick(result)
        assert {candidate['id'] for candidate in result['candidates']} == all_ids
    for query in ('SS 9', 'Sunshine 9'):
        result = resolve(catalog, 'product', query, limit=25)
        assert_no_pick(result)
        assert {candidate['id'] for candidate in result['candidates']} == set(catalog['batches'] + catalog['finished'])
    cur.execute("UPDATE search_aliases SET active=false WHERE alias='SS'")
    cur.execute(MIGRATION.read_text())
    assert not next(row for row in r.read_aliases(cur)[0] if row['alias'] == 'SS')['active']


def test_pagination_counts_deduplicated_alias_union_before_slicing(catalog):
    alias(catalog)
    for i in range(30):
        catalog['product'](f'SS Sunshine #9 Review {i:02}')
    seen = []
    for offset in range(0, 40, 5):
        result = resolve(catalog, 'product', 'Sunshine 9', limit=5, offset=offset)
        assert_no_pick(result)
        assert result['candidate_count'] == 38
        assert result['has_more'] == (offset + 5 < 38)
        seen.extend(candidate['id'] for candidate in result['candidates'])
    assert len(seen) == len(set(seen)) == 38
    empty = resolve(catalog, 'product', 'Sunshine 9', limit=1, offset=1000)
    assert_no_pick(empty)
    assert empty['candidates'] == [] and empty['candidate_count'] == 38
    one = resolve(catalog, 'product', 'A4-70050', limit=1, offset=1000)
    assert one['match']['id'] == catalog['classic'][0] and one['candidates'] == []


@pytest.mark.parametrize('fields', [{'limit': 0}, {'limit': 26}, {'limit': True}, {'offset': -1}, {'offset': 1.5}])
def test_http_pagination_rejects_invalid_bounds(client, fields):
    response = client.post('/resolve', json={'kind': 'product', 'query': 'Classic', **fields}, headers={'X-API-Key': 'a4-master'})
    assert response.status_code == 422


def test_http_pagination_defaults_and_one_item_page_never_choose(client):
    body = {'kind': 'product', 'query': 'Classic'}
    default = client.post('/resolve', json=body, headers={'X-API-Key': 'a4-master'}).json()
    assert default['limit'] == 8 and default['offset'] == 0 and len(default['candidates']) == 8
    one = client.post('/resolve', json={**body, 'limit': 1, 'offset': 1}, headers={'X-API-Key': 'a4-master'}).json()
    assert_no_pick(one)
    assert len(one['candidates']) == 1 and one['candidate_count'] == default['candidate_count']


def test_lot_and_order_pages_retain_global_ambiguity(catalog):
    cur, product = catalog['cur'], catalog['classic'][0]
    for i in range(12):
        lot(cur, product, f'REVIEW-{i:02}-1234', 5)
        cur.execute('INSERT INTO sales_orders(order_number,customer_id) VALUES (%s,%s)',
                    (f'REVIEW-ORDER-{i:02}', catalog['customers'][0]))
    for kind, query, context in [('lot', '1234', {'product_id': product}),
                                  ('order', 'open orders', {'customer_id': catalog['customers'][0]})]:
        result = resolve(catalog, kind, query, context, limit=1, offset=11)
        assert_no_pick(result)
        assert result['candidate_count'] == 12 and len(result['candidates']) == 1 and not result['has_more']


def test_product_recency_uses_posted_history_and_open_lines_in_180_days(catalog):
    cur = catalog['cur']
    names = ['A Stale', 'B Unused', 'C Voided', 'D Recent', 'E Open Order', 'F Closed Order']
    products = [catalog['product']('RecencyProbe ' + name) for name in names]
    lot(cur, products[0], 'RECENCY-STALE-0001', 1, days_ago=181)
    _, voided = lot(cur, products[2], 'RECENCY-VOID-0001', 1)
    cur.execute("""INSERT INTO ledger_corrections(target_table,target_id,event_type,previous_values,replacement_values,reason)
                   VALUES ('transactions',%s,'void','{}','{}','review fixture')""", (voided,))
    lot(cur, products[3], 'RECENCY-RECENT-0001', 1, days_ago=1)
    for index, state in ((4, 'open'), (5, 'closed')):
        cur.execute('INSERT INTO sales_orders(order_number,customer_id,state,state_reason) VALUES (%s,%s,%s,%s) RETURNING id',
                    (f'RECENCY-SO-{index}', catalog['customers'][0], state, 'short_closed' if state == 'closed' else None))
        order = cur.fetchone()['id']
        cur.execute('INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES (%s,%s,10)', (order, products[index]))
    result = resolve(catalog, 'product', 'RecencyProbe', limit=25)
    assert_no_pick(result)
    assert [row['id'] for row in result['candidates']] == [products[i] for i in (4, 3, 0, 1, 2, 5)]
    exact = resolve(catalog, 'product', 'RecencyProbe A Stale')
    assert exact['match']['id'] == products[0]


def test_recency_never_changes_scores_or_breaks_ambiguity():
    rows = [dict(id=1, name='Old but stronger', score=.9, tier='trigram', recent_activity=0),
            dict(id=2, name='Recent but weaker', score=.6, tier='trigram', recent_activity=100)]
    result = r.decide('query', rows, limit=1)
    assert_no_pick(result)
    assert result['candidates'][0]['id'] == 1


def test_exact_inactive_lot_cannot_fall_through_to_active_product(catalog):
    cur = catalog['cur']
    old, current = catalog['classic']
    lot(cur, old, 'REVIEW-INACTIVE-1001', 10)
    lot(cur, current, 'REVIEW-INACTIVE-1002', 10)
    cur.execute('UPDATE products SET active=false WHERE id=%s', (old,))
    result = resolve(catalog, 'lot', 'REVIEW-INACTIVE-1001', {'action': 'ship'})
    assert_no_pick(result, 'none')
    assert 'inactive' in result['note']


def test_pagination_preserves_sql_numeric_id_tie_order():
    rows = [dict(id=2, name='Duplicate', score=1., tier='exact', _position=1),
            dict(id=10, name='Duplicate', score=1., tier='exact', _position=2)]
    result = r.decide('Duplicate', rows)
    assert_no_pick(result)
    assert [row['id'] for row in result['candidates']] == [2, 10]
