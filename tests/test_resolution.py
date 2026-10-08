"""A4 safety contract against real, disposable LOCAL PostgreSQL only."""
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path

import psycopg2
import pytest
from fastapi.testclient import TestClient

import main
import resolution as r


MIGRATION = Path(__file__).resolve().parents[1] / 'migrations/060_search_aliases.sql'


@pytest.fixture
def catalog(db_cursor):
    cur = db_cursor
    cur.execute(MIGRATION.read_text())
    def product(name, type='ingredient', **fields):
        columns = ['name', 'type', 'active', *fields]
        cur.execute(f"INSERT INTO products ({','.join(columns)}) VALUES ({','.join(['%s'] * len(columns))}) RETURNING id",
                    (name, type, True, *fields.values()))
        return cur.fetchone()['id']
    classic = [product('A4 Classic 25 LB', 'finished', odoo_code='A4-70050'),
               product('A4 Classic 10 LB', 'finished')]
    chips = [product('White Chocolate Chips'), product('Chocolate Chips Sugar Free'),
             product('Chocolate Chips Real 1000 CT'), product('Chocolate Chips Real 4000 CT'),
             product('Granola Chocolate Chip 25 LB', 'finished')]
    batches = [product('Batch SS Classic #9', 'batch', odoo_code='A4-90025'),
               product('Batch SS Classic Chocolate Chip #9', 'batch', odoo_code='A4-90026')]
    finished = [product(f'Granola SS Classic {flavor}#9 {size} LB', 'finished',
                        parent_batch_product_id=batches[i])
                for i, flavor in enumerate(('', 'Chocolate Chip ')) for size in (5, 10, 25)]
    pouch = product('A4 Pouches', 'finished', pack_format='bagged', case_size_lb=Decimal('7.5'),
                    bags_per_case=12, units_per_case=12, retail_bag_oz=10)
    service = product('A4 Service', 'finished', is_service=True)
    cur.execute("INSERT INTO customers(name,address,active) VALUES ('A4 Setton North','North Road',true), ('A4 Setton South','South Road',true) RETURNING id")
    customers = [row['id'] for row in cur.fetchall()]
    cur.execute("INSERT INTO suppliers(name) VALUES ('A4 Supplier One'), ('A4 Supplier Two') RETURNING id")
    suppliers = [row['id'] for row in cur.fetchall()]
    return dict(cur=cur, product=product, classic=classic, chips=chips, batches=batches,
                finished=finished, pouch=pouch, service=service, customers=customers, suppliers=suppliers)


def resolve(catalog, kind, query, context=None, **kwargs):
    return r.resolve(catalog['cur'], r.ResolveRequest(kind=kind, query=query, context=context or {}, **kwargs))


def alias(catalog, kind='token', text='SS', expansion='Sunshine', **target):
    cur = catalog['cur']
    columns = ['kind', 'alias', 'expansion', *target]
    cur.execute(f"INSERT INTO search_aliases ({','.join(columns)}) VALUES ({','.join(['%s'] * len(columns))}) RETURNING id",
                (kind, text, expansion, *target.values()))
    return cur.fetchone()['id']


def assert_no_pick(result, outcome='ambiguous'):
    assert result['outcome'] == outcome, result
    assert result['match'] is None
    assert result['needs_clarification']
    assert result['ask']


def test_classic_is_ambiguous_and_display_cap_never_selects(catalog):
    result = resolve(catalog, 'product', 'Classic')
    assert_no_pick(result)
    assert len(result['candidates']) == 8
    assert result['has_more'] and result['candidate_count'] >= 10


def test_chocolate_chip_never_picks_white_and_context_only_ranks(catalog):
    for context in ({}, {'group': 'floor', 'action': 'make'}, {'action': 'order'}):
        result = resolve(catalog, 'product', 'chocolate chip', context)
        assert_no_pick(result)
        assert result['confidence'] == 'medium'
        if context.get('action') == 'order':
            assert result['candidates'][0]['type'] == 'finished'
        if context.get('group') == 'floor':
            assert result['candidates'][0]['type'] == 'ingredient'


def test_alias_family_bidirectional_context_and_normalization(catalog):
    aid = alias(catalog)
    for query in ('Sunshine 9', 'SS 9', '  SUNSHINE   9  '):
        result = resolve(catalog, 'product', query)
        assert_no_pick(result)
        assert {c['id'] for c in result['candidates']} == set(catalog['batches'] + catalog['finished'])
        assert result['candidate_count'] == 8
        if 'sunshine' in query.lower():
            assert result['query_normalized'] == 'ss 9'
            assert result['expansions_applied'] == [{'from': 'sunshine', 'to': 'ss', 'alias_id': aid}]
    made = resolve(catalog, 'product', 'Sunshine 9', {'action': 'make'})
    assert_no_pick(made)
    assert {c['id'] for c in made['candidates']} == set(catalog['batches'])
    packed = resolve(catalog, 'product', 'Sunshine 9', {'action': 'pack', 'product_id': catalog['batches'][0]})
    assert_no_pick(packed)
    assert len(packed['candidates']) == 3
    assert all(c['parent_batch_product_id'] == catalog['batches'][0] for c in packed['candidates'])


@pytest.mark.parametrize('query', ['glass jar', 'mass', 'SSX', 'herbs'])
def test_aliases_never_expand_inside_words(catalog, query):
    alias(catalog)
    alias(catalog, text='BS', expansion='Blue Stripes')
    result = resolve(catalog, 'product', query)
    assert result['expansions_applied'] == []
    assert_no_pick(result, 'none')


def test_short_ss_cannot_match_classic_substring_even_without_seed(catalog):
    result = resolve(catalog, 'product', 'SS')
    assert_no_pick(result)
    assert not set(catalog['classic']) & {c['id'] for c in result['candidates']}
    assert all('ss' in c['name'].lower().split() for c in result['candidates'])


def test_multiword_alias_reverse_direction_and_no_partial_alias(catalog):
    expected = catalog['product']('Blue Stripes Whole Cacao Beans')
    alias(catalog, text='BS', expansion='Blue Stripes')
    result = resolve(catalog, 'product', 'bs cacao')
    assert result['match']['id'] == expected
    assert result['expansions_applied'][0]['from'] == 'bs'
    assert_no_pick(resolve(catalog, 'product', 'bsx cacao'), 'none')


def test_exact_names_codes_duplicate_names_and_inactive(catalog):
    result = resolve(catalog, 'product', 'A4-70050')
    assert result['match']['id'] == catalog['classic'][0]
    assert result['confidence'] == 'high'
    assert resolve(catalog, 'product', ' A4   Classic 25 LB ')['match']['id'] == catalog['classic'][0]
    catalog['product']('a4 classic 25 lb', 'finished')
    assert_no_pick(resolve(catalog, 'product', 'A4 Classic 25 LB'))
    catalog['cur'].execute('UPDATE products SET active=false WHERE id=%s', (catalog['classic'][0],))
    assert_no_pick(resolve(catalog, 'product', 'A4-70050'), 'none')


def test_global_product_aliases_conflict_without_auto_pick(catalog):
    for product_id in catalog['classic']:
        alias(catalog, kind='product', text='house cereal', expansion=None, product_id=product_id)
    result = resolve(catalog, 'product', 'house cereal')
    assert_no_pick(result)
    assert result['confidence'] == 'high'
    catalog['cur'].execute("UPDATE search_aliases SET active=false WHERE product_id=%s", (catalog['classic'][1],))
    assert resolve(catalog, 'product', 'house cereal')['match']['id'] == catalog['classic'][0]
    assert_no_pick(resolve(catalog, 'product', 'house cere'), 'none')


def test_scoped_product_aliases_require_exact_customer_or_supplier(catalog):
    cur = catalog['cur']
    cur.execute('INSERT INTO customer_product_aliases(customer_id,customer_item_code,customer_description,product_id) VALUES (%s,%s,%s,%s)',
                (catalog['customers'][0], 'CUSTOM-1', 'My Exact Item', catalog['classic'][0]))
    cur.execute('INSERT INTO supplier_product_aliases(supplier_id,vendor_description,product_id) VALUES (%s,%s,%s)',
                (catalog['suppliers'][0], 'VENDOR-1', catalog['classic'][1]))
    assert_no_pick(resolve(catalog, 'product', 'CUSTOM-1'), 'none')
    assert resolve(catalog, 'product', 'CUSTOM-1', {'customer_id': catalog['customers'][0]})['match']['id'] == catalog['classic'][0]
    assert resolve(catalog, 'product', 'My Exact Item', {'customer_id': catalog['customers'][0]})['match']['id'] == catalog['classic'][0]
    assert resolve(catalog, 'product', 'VENDOR-1', {'supplier_id': catalog['suppliers'][0]})['match']['id'] == catalog['classic'][1]
    assert_no_pick(resolve(catalog, 'product', 'VENDOR', {'supplier_id': catalog['suppliers'][0]}), 'none')


def test_customer_alias_address_boost_and_supplier_support(catalog):
    cur = catalog['cur']
    cur.execute('INSERT INTO customer_aliases(customer_id,alias) VALUES (%s,%s)', (catalog['customers'][0], 'SN'))
    assert resolve(catalog, 'customer', 'SN')['match']['id'] == catalog['customers'][0]
    result = resolve(catalog, 'customer', 'A4 Setton', {'customer_address': 'South Road'})
    assert_no_pick(result)
    assert result['candidates'][0]['id'] == catalog['customers'][1]
    alias(catalog, kind='supplier', text='vendor shorthand', expansion=None, supplier_id=catalog['suppliers'][1])
    assert resolve(catalog, 'supplier', 'vendor shorthand')['match']['id'] == catalog['suppliers'][1]
    assert_no_pick(resolve(catalog, 'supplier', 'A4 Supplier'))
    cur.execute('UPDATE customers SET active=false WHERE id=%s', (catalog['customers'][0],))
    assert_no_pick(resolve(catalog, 'customer', 'SN'), 'none')


def test_supplier_pseudo_vendors_never_resolve_even_if_active(catalog):
    cur = catalog['cur']
    for name in r.SENTINEL_SUPPLIERS:
        cur.execute('INSERT INTO suppliers(name,active) VALUES (%s,true) RETURNING id', (name.upper(),))
        supplier_id = cur.fetchone()['id']
        alias(catalog, kind='supplier', text='Vendor ' + name, expansion=None, supplier_id=supplier_id)
        assert_no_pick(resolve(catalog, 'supplier', name), 'none')
        assert_no_pick(resolve(catalog, 'supplier', 'Vendor ' + name), 'none')


def test_no_confident_match_and_trigram_near_misses(catalog):
    result = resolve(catalog, 'product', 'qzxv uncommon zz')
    assert_no_pick(result, 'none')
    assert result['candidates'] == []
    assert len(result['near_misses']) <= 3
    assert all(c['score'] < .25 for c in result['near_misses'])
    catalog['product']('Almonds')
    assert resolve(catalog, 'product', 'Almond')['match']['name'] == 'Almonds'
    typo = resolve(catalog, 'product', 'Almnds')
    assert typo['confidence'] == 'low'
    assert typo['candidates'][0]['tier'] == 'trigram'


def test_rank_gap_cannot_disambiguate_and_low_confidence_requires_choice():
    rows = [dict(id=1, name='First', score=.96, tier='trigram'),
            dict(id=2, name='Second', score=.51, tier='trigram')]
    assert_no_pick(r.decide('query', rows))
    assert_no_pick(r.decide('query', [dict(id=1, name='Near', score=.3, tier='trigram')]))


def test_alias_expansion_overflow_is_never_a_match():
    aliases = [dict(id=i, kind='token', alias=f'term{i}', expansion=f'word{i}') for i in range(6)]
    _, truncated = r.query_variants(' '.join(f'term{i}' for i in range(6)), aliases)
    assert truncated
    assert_no_pick(r.decide('query', [dict(id=1, name='Candidate', score=1., tier='exact')], truncated=True))


def test_lots_full_code_supplier_code_suffix_balance_and_merged(catalog):
    cur, product = catalog['cur'], catalog['classic'][0]
    ids = []
    for code in ('A4-FIRST-1234', 'A4-SECOND-1234', 'A4-EMPTY-9876'):
        cur.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id', (product, code))
        ids.append(cur.fetchone()['id'])
    for lot in ids[:2]:
        cur.execute("INSERT INTO transactions(type) VALUES ('receive') RETURNING id")
        txn = cur.fetchone()['id']
        cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,10)', (txn, product, lot))
    context = {'product_id': product}
    assert resolve(catalog, 'lot', ' a4-first-1234   LOT ', context)['match']['id'] == ids[0]
    assert_no_pick(resolve(catalog, 'lot', '1234', context))
    assert_no_pick(resolve(catalog, 'lot', '1234'), 'none')
    assert_no_pick(resolve(catalog, 'lot', '9876', context), 'none')
    assert resolve(catalog, 'lot', '9876', dict(context, action='adjust'))['match']['id'] == ids[2]
    cur.execute('INSERT INTO lot_supplier_codes(lot_id,supplier_lot_code) VALUES (%s,%s)', (ids[0], 'SUPPLIER-42'))
    assert resolve(catalog, 'lot', 'SUPPLIER-42', context)['match']['id'] == ids[0]
    cur.execute("UPDATE lots SET status='merged', merged_into_lot_id=%s WHERE id=%s", (ids[1], ids[0]))
    result = resolve(catalog, 'lot', 'A4-FIRST-1234', context)
    assert_no_pick(result, 'none')
    assert str(ids[1]) in result['note']


def test_orders_exact_po_customer_browse_unknown_and_state(catalog):
    cur = catalog['cur']
    orders = []
    for n, customer in enumerate(catalog['customers']):
        cur.execute('INSERT INTO sales_orders(order_number,customer_id,customer_po) VALUES (%s,%s,%s) RETURNING id', (f'SO-A4-{n}', customer, 'SAME-PO'))
        orders.append(cur.fetchone()['id'])
    assert resolve(catalog, 'order', 'SO-A4-0')['match']['id'] == orders[0]
    assert_no_pick(resolve(catalog, 'order', 'SAME-PO'))
    assert resolve(catalog, 'order', 'SAME-PO', {'customer_id': catalog['customers'][1]})['match']['id'] == orders[1]
    assert_no_pick(resolve(catalog, 'order', 'the A4 Setton order'))
    assert resolve(catalog, 'order', 'open orders', {'customer_id': catalog['customers'][0]})['match']['id'] == orders[0]
    assert_no_pick(resolve(catalog, 'order', 'SO-UNKNOWN', {'customer_id': catalog['customers'][0]}), 'none')
    cur.execute("UPDATE sales_orders SET state='closed',state_reason='short_closed' WHERE id=%s", (orders[0],))
    assert_no_pick(resolve(catalog, 'order', 'open orders', {'customer_id': catalog['customers'][0]}), 'none')
    assert resolve(catalog, 'order', 'SO-A4-0')['match']['id'] == orders[0]


@pytest.mark.parametrize('quantity,expected', [('24', '2'), ('120', '10')])
def test_pouches_convert_to_cases_in_draft(catalog, quantity, expected):
    result = resolve(catalog, 'unit', 'pouches', {'product_id': catalog['pouch']}, quantity=quantity)
    assert result['outcome'] == 'match'
    assert result['draft'] == {'product_id': catalog['pouch'], 'unit': 'cases', 'quantity': Decimal(expected)}
    assert result['conversion']['pouches_per_case'] == 12


@pytest.mark.parametrize('quantity', ['13', '23.5', None])
def test_partial_pouches_need_clarification_without_a_draft(catalog, quantity):
    result = resolve(catalog, 'unit', 'pouches', {'product_id': catalog['pouch']}, quantity=quantity)
    assert_no_pick(result)
    assert result['code'] == 'NEEDS_CLARIFICATION'
    assert 'draft' not in result


def test_pouches_refuse_missing_inconsistent_and_nonpouch_factors(catalog):
    cur = catalog['cur']
    for updates in ('bags_per_case=10', 'bags_per_case=NULL,units_per_case=NULL,retail_bag_oz=NULL'):
        cur.execute(f'UPDATE products SET {updates} WHERE id=%s', (catalog['pouch'],))
        assert_no_pick(resolve(catalog, 'unit', 'pouches', {'product_id': catalog['pouch']}, quantity=24))
    assert_no_pick(resolve(catalog, 'unit', 'pouches', {'product_id': catalog['classic'][0]}, quantity=24))
    assert_no_pick(resolve(catalog, 'unit', 'pouches', quantity=24))


def test_pouches_explicit_catalog_pack_label_must_agree_with_case_weight(catalog):
    product = catalog['product']('A4 Retail 12x10 OZ Case', 'finished', pack_format='bagged', case_size_lb=Decimal('7.5'))
    result = resolve(catalog, 'unit', 'pouches', {'product_id': product}, quantity=24)
    assert result['draft']['quantity'] == 2
    catalog['cur'].execute('UPDATE products SET case_size_lb=8 WHERE id=%s', (product,))
    assert_no_pick(resolve(catalog, 'unit', 'pouches', {'product_id': product}, quantity=24))


def test_units_enum_missing_and_product_restrictions(catalog):
    assert resolve(catalog, 'unit', 'lb')['match']['unit'] == 'lb'
    assert_no_pick(resolve(catalog, 'unit', 'lbs'), 'none')
    assert_no_pick(resolve(catalog, 'unit', 'lb', {'product_id': catalog['service']}), 'none')
    assert_no_pick(resolve(catalog, 'unit', '9', {'product_id': catalog['service']}))
    assert_no_pick(resolve(catalog, 'unit', '', {'product_id': catalog['pouch']}))
    assert resolve(catalog, 'unit', 'each', {'product_id': catalog['service']})['match']['unit'] == 'each'


def test_migration_idempotent_seed_format_constraints_and_readback(catalog):
    cur = catalog['cur']
    cur.execute(MIGRATION.read_text())
    rows = r.validate_alias_seed({'version': 1, 'aliases': [
        {'kind': 'token', 'alias': '  SS  ', 'expansion': 'Sunshine'},
        {'kind': 'product', 'alias': 'my cereal', 'product_id': catalog['classic'][0], 'language': 'es'},
    ]})
    for row in rows:
        cur.execute(f"INSERT INTO search_aliases ({','.join(row)}) VALUES ({','.join(['%s']*len(row))})", tuple(row.values()))
    stored, available = r.read_aliases(cur)
    assert available and len(stored) == 2
    assert stored[0]['alias_norm'] == 'ss'
    for sql, params in [
        ("INSERT INTO search_aliases(kind,alias) VALUES ('token','bad')", ()),
        ("INSERT INTO search_aliases(kind,alias,expansion,product_id) VALUES ('token','bad','x',%s)", (catalog['classic'][0],)),
        ("INSERT INTO search_aliases(kind,alias,expansion) VALUES ('token','SS','Different')", ()),
        ("INSERT INTO search_aliases(kind,alias,expansion) VALUES ('token',E'\\t','x')", ()),
    ]:
        cur.execute('SAVEPOINT bad_alias')
        with pytest.raises(psycopg2.IntegrityError):
            cur.execute(sql, params)
        cur.execute('ROLLBACK TO SAVEPOINT bad_alias')


@pytest.mark.parametrize('row', [
    {'kind': 'token', 'alias': 'x'},
    {'kind': 'token', 'alias': ' ', 'expansion': 'y'},
    {'kind': 'product', 'alias': 'x', 'product_id': 1, 'expansion': 'y'},
    {'kind': 'supplier', 'alias': 'x', 'supplier_id': 1, 'created_by': 1},
])
def test_invalid_seed_rejected(row):
    with pytest.raises(ValueError):
        r.validate_alias_seed({'version': 1, 'aliases': [row]})


@pytest.fixture
def client(catalog, monkeypatch):
    @contextmanager
    def transaction():
        yield catalog['cur']
    monkeypatch.setattr(main, 'get_transaction', transaction)
    monkeypatch.setattr(main, 'API_KEY', 'a4-master')
    monkeypatch.setattr(main, 'DASHBOARD_API_KEY', 'a4-dashboard')
    monkeypatch.setattr(main, '_resolve_actor', lambda key: None)
    # No lifespan: avoid startup sweeps; all request SQL uses the test savepoint.
    return TestClient(main.app)


def test_http_auth_bulk_compatibility_and_alias_writes_absent(client, catalog, monkeypatch):
    payload = {'kind': 'product', 'query': 'Classic'}
    assert client.post('/resolve', json=payload).status_code == 401
    assert client.post('/resolve', json=payload, headers={'X-API-Key': 'bad'}).status_code == 403
    for key in ('a4-master', 'a4-dashboard'):
        response = client.post('/resolve', json=payload, headers={'X-API-Key': key})
        assert response.status_code == 200, response.text
        assert_no_pick(response.json())
        assert client.get('/aliases', headers={'X-API-Key': key}).status_code == 200
    bulk = client.post('/products/resolve', json={'names': ['Classic', 'A4-70050', 'zzqvx']}, headers={'X-API-Key': 'a4-master'})
    assert bulk.status_code == 200, bulk.text
    assert bulk.json()['summary'] == {'total': 3, 'resolved': 1, 'unresolved': 2}
    assert bulk.json()['resolved'][0]['match'] is None
    assert bulk.json()['resolved'][1]['match']['id'] == catalog['classic'][0]
    routes = {(method, route.path) for route in main.app.routes for method in getattr(route, 'methods', [])}
    assert not any(path.startswith('/aliases') and method != 'GET' for method, path in routes)


def test_http_unknown_inputs_and_untrusted_alias_injection_rejected(client):
    for payload in ({'kind': 'other', 'query': 'Classic'},
                    {'kind': 'product', 'query': 'Classic', 'aliases': []},
                    {'kind': 'product', 'query': 'Classic', 'context': {'product_id': -1}},
                    {'kind': 'unit', 'query': 'pouches', 'quantity': 'NaN'},
                    {'kind': 'unit', 'query': 'pouches', 'quantity': 'not a number'},
                    {'kind': 'unit', 'query': 'pouches', 'quantity': '1e10000'},
                    {'kind': 'unit', 'query': 'pouches', 'quantity': 0},
                    {'kind': 'unit', 'query': 'pouches', 'quantity': True}):
        assert client.post('/resolve', json=payload, headers={'X-API-Key': 'a4-master'}).status_code == 422


@pytest.mark.parametrize('role', ['owner', 'office', 'floor'])
def test_new_reads_accept_all_actor_roles_via_dashboard_scope(client, monkeypatch, role):
    actor = {'id': 1, 'name': 'A4 Actor', 'role': role, 'key_hash': 'test-a4-actor'}
    monkeypatch.setattr(main, '_resolve_actor', lambda key: actor if key == 'a4-actor' else None)
    monkeypatch.setattr(main, '_touch_actor_last_used', lambda value: False)
    headers = {'X-API-Key': 'a4-actor'}
    result = client.post('/resolve', json={'kind': 'product', 'query': 'Classic'}, headers=headers)
    assert result.status_code == 200
    assert_no_pick(result.json())
    assert client.get('/aliases', headers=headers).status_code == 200


def test_empty_exact_lot_does_not_make_a_positive_exact_hit_ambiguous(catalog):
    cur = catalog['cur']
    for product in catalog['classic']:
        cur.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id', (product, 'A4-SHARED-CODE'))
        lot = cur.fetchone()['id']
    cur.execute("INSERT INTO transactions(type) VALUES ('receive') RETURNING id")
    txn = cur.fetchone()['id']
    cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,5)',
                (txn, catalog['classic'][1], lot))
    result = resolve(catalog, 'lot', 'A4-SHARED-CODE')
    assert result['match']['id'] == lot
    assert result['candidate_count'] == 1


def test_resolver_runs_in_real_read_only_transaction(catalog):
    # Fixtures were written on the test connection. A separate connection runs
    # the public core with DB-enforced READ ONLY and an in-memory token fixture.
    import os
    from psycopg2.extras import RealDictCursor
    with psycopg2.connect(os.environ['TEST_DATABASE_URL']) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SET TRANSACTION READ ONLY')
            for kind, query in [('product', 'Classic'), ('customer', 'Setton'),
                                ('supplier', 'A4'), ('order', 'UNKNOWN'), ('lot', '1234'), ('unit', 'lb')]:
                r.resolve(cur, r.ResolveRequest(kind=kind, query=query), aliases=[])
            cur.execute('SHOW transaction_read_only')
            assert cur.fetchone()['transaction_read_only'] == 'on'
