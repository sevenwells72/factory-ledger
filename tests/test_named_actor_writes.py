"""FR-15 owner decision: all 14 writes and product resolution are reachable by actor keys.

Since A2 (design §4.3) a named actor's ROLE also has to allow the write:
`permissions.ROUTE_ACTIONS` maps each direct route to its matrix action, and a
denied role gets 403 ROLE_NOT_ALLOWED before the handler. The attribution
assertions below therefore run as a role the matrix allows (`_actor_for`), and
the all-roles parametrizations assert the denial where it applies.

Real PostgreSQL writes through HTTP, with persisted attribution assertions.
All fixtures use the local TEST_DATABASE_URL and roll back after each test.
"""
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

import pytest

import main
import permissions
from tests.test_actor_attribution import client as actor_client, _ConnProxy, _seed, _allocate, _allocation  # noqa: F401
from tests.test_trace_emission import _seed_make_setup, _seed_pack_setup
from tests.test_sales_order_extract import _insert_document, _approve_payload, _approve_line

MIGRATION = Path(__file__).resolve().parents[1] / 'migrations/056_actor_write_audit.sql'
NAMES = ('Blubber', 'Arturo', 'Luz', 'Miriam')
ROLES = {'Blubber': 'owner', 'Arturo': 'floor', 'Luz': 'floor', 'Miriam': 'office', 'Retired': 'floor'}


def _denied(identity, method, route):
    """True when §4.3 denies this named actor the route's action (shared key: never)."""
    if identity == 'shared':
        return False
    action = permissions.ROUTE_ACTIONS.get((method, route))
    return action is not None and not permissions.allowed(action, ROLES[identity])


def _actor_for(preferred, method, route):
    """`preferred` if the matrix allows it on the route, else the owner (allowed everywhere)."""
    return preferred if not _denied(preferred, method, route) else 'Blubber'


def _assert_role_denied(response, identity, before, cur):
    assert response.status_code == 403, response.text
    detail = response.json()['detail']
    assert (detail['error_code'], detail['actor'], detail['role']) == ('ROLE_NOT_ALLOWED', identity, ROLES[identity])
    assert _snapshot(cur) == before
# Intentionally independent of the implementation's allowlist.
CASES = [
    ('POST', '/receive'), ('POST', '/ship'), ('POST', '/make'),
    ('POST', '/pack'), ('POST', '/adjust'), ('POST', '/void/{transaction_id}'),
    ('POST', '/customers'), ('PATCH', '/customers/{customer_id}'),
    ('PATCH', '/lots/{lot_code}/supplier-lot'), ('PATCH', '/lots/{lot_id}/rename'),
    ('POST', '/sales/orders'), ('POST', '/sales/orders/{order_id}/lines'),
    ('PATCH', '/sales/orders/{order_id}/lines/{line_id}/cancel'),
    ('POST', '/sales/orders/{order_id}/ship'),
]
LEDGER = {'/receive', '/ship', '/make', '/pack', '/adjust',
          '/sales/orders/{order_id}/ship'}
METADATA = [case for case in CASES if case[1] not in LEDGER
            and not case[1].startswith('/void/')]
TABLES = ('transactions', 'transaction_lines', 'ledger_corrections', 'trace_events',
          'trace_event_lots', 'customers', 'customer_aliases', 'lots',
          'lot_supplier_codes', 'sales_orders', 'sales_order_lines',
          'sales_order_allocations', 'shipments', 'shipment_lines', 'actor_write_audit')


@pytest.fixture
def client(actor_client, db_cursor, monkeypatch):
    # Customer PATCH uses the retry helper's pool directly. Keep that REAL
    # commit/rollback path on the same savepoint-isolated test connection too.
    monkeypatch.setattr(main.db_pool, 'getconn',
                        lambda: _ConnProxy(db_cursor.connection, 'named_actor_pool'))
    monkeypatch.setattr(main.db_pool, 'putconn', lambda *args, **kwargs: None)
    yield actor_client


@pytest.fixture
def named_actors(db_cursor):
    db_cursor.execute((MIGRATION.parent / '052_actors.sql').read_text())
    db_cursor.execute(MIGRATION.read_text())
    result = {}
    for name in NAMES + ('Retired',):
        role = ROLES[name]
        key = 'test-only-' + uuid4().hex
        db_cursor.execute(
            'INSERT INTO actors (name, role, key_hash, active) VALUES (%s,%s,%s,%s) RETURNING id',
            (name, role, sha256(key.encode()).hexdigest(), name != 'Retired'),
        )
        result[name] = {'id': db_cursor.fetchone()['id'], 'key': key}
    main._reset_actor_cache()
    yield result
    main._reset_actor_cache()


@pytest.fixture
def prepared(db_cursor, named_actors):
    seed = _seed(db_cursor)
    db_cursor.execute('SELECT name FROM products WHERE id=%s', (seed['product_id'],))
    seed['product_name'] = db_cursor.fetchone()['name']
    db_cursor.execute('SELECT name FROM customers WHERE id=%s', (seed['customer_id'],))
    seed['customer_name'] = db_cursor.fetchone()['name']
    db_cursor.execute('SELECT lot_code FROM lots WHERE id=%s', (seed['lot_id'],))
    seed['lot_code'] = db_cursor.fetchone()['lot_code']
    db_cursor.execute('SELECT transaction_id FROM transaction_lines WHERE lot_id=%s',
                      (seed['lot_id'],))
    seed['transaction_id'] = db_cursor.fetchone()['transaction_id']
    return seed


def _payload(cur, route, seed):
    base = {'mode': 'commit'}
    if route == '/receive':
        base.update(product_name=seed['product_name'], cases=2, case_size_lb=25,
                    shipper_name='Actor Test Supplier', bol_reference='ACTOR-TEST')
    elif route == '/ship':
        base.update(product_name=seed['product_name'], quantity_lb=10,
                    customer_name=seed['customer_name'], order_reference='ACTOR-TEST',
                    lot_code=seed['lot_code'], force_standalone=True)
    elif route == '/make':
        batch, _ = _seed_make_setup(cur.connection)
        base.update(product_name=batch['name'], batches=1)
    elif route == '/pack':
        batch, fg, _ = _seed_pack_setup(cur.connection)
        base.update(source_product=batch['name'], target_product=fg['name'], cases=2)
    elif route == '/adjust':
        base.update(product_name=seed['product_name'], lot_code=seed['lot_code'],
                    adjustment_lb=10, reason='Actor test', reason_es='Prueba de actor')
    elif route.startswith('/void/'):
        return {'reason': 'Actor test void'}
    elif route == '/customers':
        return {'name': 'Actor Test Customer ' + uuid4().hex}
    elif route == '/customers/{customer_id}':
        return {'contact_name': 'Updated Contact', 'aliases': ['ALIAS-' + uuid4().hex]}
    elif route.endswith('/supplier-lot'):
        return {'supplier_lot_code': 'SUPPLIER-ACTOR-TEST'}
    elif route.endswith('/rename'):
        return {'new_lot_code': 'RENAMED-' + uuid4().hex[:8].upper()}
    elif route in ('/sales/orders', '/sales/orders/{order_id}/lines'):
        body = {'lines': [{'product_name': seed['product_name'], 'quantity_lb': 20}]}
        if route == '/sales/orders':
            body['customer_name'] = seed['customer_name']
        return body
    elif route.endswith('/cancel'):
        return None
    elif route.endswith('/ship'):
        base.update(ship_all=True)
    return base


def _snapshot(cur):
    result = {}
    for table in TABLES:
        cur.execute(f'SELECT row_to_json(t) AS data FROM {table} t ORDER BY id')
        result[table] = [r['data'] for r in cur.fetchall()]
    return result


def _request(client, method, route, seed, body, key):
    return client.request(method, route.format(**seed), json=body,
                          headers={'X-API-Key': key} if key is not None else {})


def test_actor_only_scope_is_approved_writes_plus_product_resolution():
    assert main.ACTOR_WRITE_ALLOWLIST == set(CASES) | {('POST', '/products/resolve')} | {
        ('POST', '/receive/prepare'), ('POST', '/tickets/{ticket}/commit'),
        ('POST', '/make/prepare'), ('POST', '/pack/prepare'),
        ('POST', '/lots/{lot_id}/move/prepare'),
        ('POST', '/adjust/prepare'), ('POST', '/inventory/found/prepare'),
        ('GET', '/receipts'), ('GET', '/receipts/{receipt_number}'),
        ('GET', '/receipts/by-transaction/{transaction_id}'),
    }
    assert ('POST', '/products/resolve') not in main.DASHBOARD_KEY_ALLOWLIST


@pytest.mark.db
@pytest.mark.parametrize('identity', NAMES + ('shared',))
def test_product_resolution_accepts_actor_and_shared_keys_without_business_changes(
        client, db_cursor, named_actors, prepared, identity, monkeypatch):
    if identity == 'shared':
        key = main.API_KEY

        def unexpected_lookup(*args):
            raise AssertionError('Shared key must never resolve actors')

        monkeypatch.setattr(main, '_resolve_actor', unexpected_lookup)
    else:
        key = named_actors[identity]['key']
    db_cursor.execute('SELECT row_to_json(p) AS data FROM products p ORDER BY id')
    products_before = db_cursor.fetchall()
    db_cursor.execute('SELECT odoo_code FROM products WHERE id=%s', (prepared['product_id'],))
    odoo_code = db_cursor.fetchone()['odoo_code']
    before = _snapshot(db_cursor)
    response = client.post('/products/resolve',
                           json={'names': [prepared['product_name']]},
                           headers={'X-API-Key': key})
    assert response.status_code == 200, response.text
    assert response.json() == {
        'resolved': [{
            'input': prepared['product_name'],
            'match': {'id': prepared['product_id'], 'name': prepared['product_name'],
                      'odoo_code': odoo_code},
            'match_tier': 'exact',
            'confidence': 'high',
        }],
        'summary': {'total': 1, 'resolved': 1, 'unresolved': 0},
        'success': True,
    }
    assert _snapshot(db_cursor) == before
    db_cursor.execute('SELECT row_to_json(p) AS data FROM products p ORDER BY id')
    assert db_cursor.fetchall() == products_before


@pytest.mark.db
@pytest.mark.parametrize('identity,status,detail', [
    ('missing', 401, 'API key required'), ('invalid', 403, 'Invalid API key'),
    ('Retired', 403, 'Invalid API key'),
    ('dashboard', 403, 'API key not authorized for this endpoint'),
])
def test_product_resolution_rejects_unauthorized_keys(
        client, db_cursor, named_actors, prepared, identity, status, detail):
    key = {'missing': None, 'invalid': 'not-a-valid-key',
           'Retired': named_actors['Retired']['key'], 'dashboard': main.DASHBOARD_API_KEY}[identity]
    before = _snapshot(db_cursor)
    response = _request(client, 'POST', '/products/resolve', prepared,
                        {'names': [prepared['product_name']]}, key)
    assert response.status_code == status, response.text
    assert response.json()['detail'] == detail
    assert response.json()['success'] is False
    assert _snapshot(db_cursor) == before


@pytest.mark.db
@pytest.mark.parametrize('method,route', CASES)
@pytest.mark.parametrize('identity', NAMES + ('shared',))
def test_each_write_persists_the_authenticated_operator(
        client, db_cursor, named_actors, prepared, method, route, identity, monkeypatch):
    body = _payload(db_cursor, route, prepared)
    allocation_id = None
    if route.endswith('/cancel'):
        allocation_id = _allocate(db_cursor, prepared)
    if identity == 'shared':
        key = main.API_KEY
        def unexpected_lookup(*args):
            raise AssertionError('Shared key must never resolve actors')
        monkeypatch.setattr(main, '_resolve_actor', unexpected_lookup)
    else:
        key = named_actors[identity]['key']
        # Self-reported identities must never override authenticated identity.
        if body is not None:
            body.update(operator_id='Somebody Else', created_by='Somebody Else')
    before = _snapshot(db_cursor)
    response = _request(client, method, route, prepared, body, key)
    if _denied(identity, method, route):
        _assert_role_denied(response, identity, before, db_cursor)
        return
    assert response.status_code == 200, response.text
    after = _snapshot(db_cursor)
    expected = 'legacy-shared-key' if identity == 'shared' else identity
    if route in LEDGER:
        previous = {r['id'] for r in before['transactions']}
        transactions = [r for r in after['transactions'] if r['id'] not in previous]
        assert transactions and {r['operator_id'] for r in transactions} == {expected}
        tids = {r['id'] for r in transactions}
        events = [r for r in after['trace_events'] if r['transaction_id'] in tids]
        assert len(events) == len(transactions)
        assert {r['operator_id'] for r in events} == {None if identity == 'shared' else identity}
    elif route.startswith('/void/'):
        corrections = after['ledger_corrections'][len(before['ledger_corrections']):]
        assert len(corrections) == 1
        assert corrections[0]['operator_id'] == expected
        assert corrections[0]['target_id'] == prepared['transaction_id']
        # Voiding attributes the correction, never rewrites its original.
        assert after['transactions'] == before['transactions']
        events = [r for r in after['trace_events'] if r['correction_id'] == corrections[0]['id']]
        assert len(events) == 1
        assert events[0]['operator_id'] == (None if identity == 'shared' else identity)
    else:
        audit = after['actor_write_audit'][len(before['actor_write_audit']):]
        if identity == 'shared':
            assert audit == []
        else:
            assert audit
            for row in audit:
                assert (row['actor_id'], row['operator_id'], row['method'], row['route']) == (
                    named_actors[identity]['id'], identity, method, route)
                assert any(r['id'] == row['target_id'] for r in after[row['target_table']])
        changed = {table for table in TABLES if before[table] != after[table]}
        assert changed - {'actor_write_audit'}, 'The business write must actually persist'
    if allocation_id:
        allocation = _allocation(db_cursor, allocation_id)
        assert allocation['status'] == 'released'
        assert allocation['released_by'] == (None if identity == 'shared' else identity)


@pytest.mark.db
@pytest.mark.parametrize('method,route', CASES)
@pytest.mark.parametrize('identity,status,detail', [
    ('missing', 401, 'API key required'), ('invalid', 403, 'Invalid API key'),
    ('Retired', 403, 'Invalid API key'),
    ('dashboard', 403, 'API key not authorized for this endpoint'),
])
def test_each_write_rejects_unauthorized_keys_without_business_changes(
        client, db_cursor, named_actors, prepared, method, route, identity, status, detail):
    body = _payload(db_cursor, route, prepared)
    key = {'missing': None, 'invalid': 'not-a-valid-key',
           'Retired': named_actors['Retired']['key'], 'dashboard': main.DASHBOARD_API_KEY}[identity]
    before = _snapshot(db_cursor)
    response = _request(client, method, route, prepared, body, key)
    assert response.status_code == status, response.text
    assert response.json()['detail'] == detail
    assert response.json()['success'] is False
    assert _snapshot(db_cursor) == before


@pytest.mark.db
@pytest.mark.parametrize('method,route', METADATA)
def test_metadata_audit_failure_rolls_back_the_write(
        client, db_cursor, named_actors, prepared, method, route, monkeypatch):
    body = _payload(db_cursor, route, prepared)
    before = _snapshot(db_cursor)
    def fail_audit(cur, *args):
        cur.execute('SELECT 1 / 0')
    monkeypatch.setattr(main, '_record_actor_write', fail_audit)
    response = _request(client, method, route, prepared, body, named_actors[_actor_for('Arturo', method, route)]['key'])
    assert response.status_code == 500, response.text
    assert _snapshot(db_cursor) == before


@pytest.mark.db
@pytest.mark.parametrize('route', sorted(LEDGER))
def test_actor_preview_has_no_business_writes(client, db_cursor, named_actors, prepared, route):
    body = _payload(db_cursor, route, prepared)
    body['mode'] = 'preview'
    before = _snapshot(db_cursor)
    response = _request(client, 'POST', route, prepared, body, named_actors[_actor_for('Luz', 'POST', route)]['key'])
    assert response.status_code == 200, response.text
    assert _snapshot(db_cursor) == before


@pytest.mark.db
def test_migration_is_idempotent_and_audit_is_append_only(db_cursor, named_actors):
    import psycopg2
    db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute("INSERT INTO actor_write_audit (actor_id, operator_id, method, route, target_table, target_id) "
                      "VALUES (%s, 'Arturo', 'POST', '/customers', 'customers', 1) RETURNING id",
                      (named_actors['Arturo']['id'],))
    row_id = db_cursor.fetchone()['id']
    for sql in ('UPDATE actor_write_audit SET operator_id=\'Other\' WHERE id=%s',
                'DELETE FROM actor_write_audit WHERE id=%s'):
        db_cursor.execute('SAVEPOINT append_only_check')
        with pytest.raises(psycopg2.Error, match='append-only'):
            db_cursor.execute(sql, (row_id,))
        db_cursor.execute('ROLLBACK TO SAVEPOINT append_only_check')
        db_cursor.execute('RELEASE SAVEPOINT append_only_check')


@pytest.mark.db
@pytest.mark.parametrize('route', sorted(LEDGER))
def test_named_ledger_write_rolls_back_if_trace_fails(
        client, db_cursor, named_actors, prepared, route, monkeypatch):
    body = _payload(db_cursor, route, prepared)
    before = _snapshot(db_cursor)
    def fail_trace(cur, *args, **kwargs):
        cur.execute('SELECT 1 / 0')
    monkeypatch.setattr(main, 'emit_trace_event', fail_trace)
    response = _request(client, 'POST', route, prepared, body, named_actors[_actor_for('Miriam', 'POST', route)]['key'])
    assert response.status_code == 500, response.text
    assert _snapshot(db_cursor) == before


@pytest.mark.db
@pytest.mark.parametrize('method,route', METADATA)
def test_shared_metadata_writes_do_not_require_the_new_audit_table(
        client, db_cursor, named_actors, prepared, method, route):
    body = _payload(db_cursor, route, prepared)
    db_cursor.execute('DROP TABLE actor_write_audit')  # rolled back by fixture
    response = _request(client, method, route, prepared, body, main.API_KEY)
    assert response.status_code == 200, response.text


@pytest.mark.db
def test_actor_identity_does_not_leak_between_requests(client, db_cursor, named_actors, prepared):
    body = _payload(db_cursor, '/adjust', prepared)
    for name in ('Arturo', 'shared', 'Luz', 'shared', 'Miriam', 'Blubber'):
        key = main.API_KEY if name == 'shared' else named_actors[name]['key']
        response = _request(client, 'POST', '/adjust', prepared, body, key)
        if name == 'Miriam':   # office: denied, and the denial names HER, not the previous caller
            assert response.status_code == 403 and response.json()['detail']['actor'] == 'Miriam', response.text
            continue
        assert response.status_code == 200, response.text
        db_cursor.execute('SELECT operator_id FROM transactions WHERE id=%s',
                          (response.json()['transaction_id'],))
        assert db_cursor.fetchone()['operator_id'] == ('legacy-shared-key' if name == 'shared' else name)


@pytest.mark.db
@pytest.mark.parametrize('route', ['/ship', '/pack', '/sales/orders/{order_id}/ship'])
@pytest.mark.parametrize('identity', ('Arturo', 'shared'))
def test_expired_allocation_release_keeps_the_correct_identity(
        client, db_cursor, named_actors, prepared, route, identity):
    body = _payload(db_cursor, route, prepared)
    allocation_seed = dict(prepared)
    if route == '/pack':
        db_cursor.execute('SELECT id FROM products WHERE name=%s', (body['source_product'],))
        allocation_seed['product_id'] = db_cursor.fetchone()['id']
        db_cursor.execute('UPDATE sales_order_lines SET product_id=%s WHERE id=%s',
                          (allocation_seed['product_id'], prepared['line_id']))
    allocation_id = _allocate(db_cursor, allocation_seed, source='auto_fifo', expired=True)
    if identity != 'shared':
        identity = _actor_for(identity, 'POST', route)   # floor may not ship standalone
    key = main.API_KEY if identity == 'shared' else named_actors[identity]['key']
    response = _request(client, 'POST', route, prepared, body, key)
    assert response.status_code == 200, response.text
    allocation = _allocation(db_cursor, allocation_id)
    assert allocation['status'] == 'released'
    # Historical standalone and SO paths intentionally use different legacy tags.
    expected = identity if identity != 'shared' else (
        None if route.endswith('/ship') and route != '/ship' else 'legacy-shared-key')
    assert allocation['released_by'] == expected


# ── PR #66 review fixes ─────────────────────────────────────────────────────
# F1: the /x/preview + /x/commit shortcut wrappers must hand the request to the
# handler, or an actor-keyed call is stored as legacy-shared-key. HTTP scope is
# unchanged: only /receive/preview and /receive/commit are actor-reachable
# (dashboard allowlist); the other shortcuts stay master-key only.
SHORTCUTS = {'receive': main.ReceiveRequest, 'ship': main.ShipRequest,
             'make': main.MakeRequest, 'pack': main.PackRequest,
             'adjust': main.AdjustRequest}


def _new_transactions(before, after):
    previous = {r['id'] for r in before['transactions']}
    transactions = [r for r in after['transactions'] if r['id'] not in previous]
    tids = {r['id'] for r in transactions}
    events = [r for r in after['trace_events'] if r['transaction_id'] in tids]
    return transactions, events


@pytest.mark.db
@pytest.mark.parametrize('identity', NAMES + ('shared',))
def test_receive_shortcut_commit_persists_the_authenticated_operator(
        client, db_cursor, named_actors, prepared, identity):
    body = _payload(db_cursor, '/receive', prepared)
    body.pop('mode')
    key = main.API_KEY if identity == 'shared' else named_actors[identity]['key']
    before = _snapshot(db_cursor)
    response = _request(client, 'POST', '/receive/commit', prepared, body, key)
    assert response.status_code == 200, response.text
    transactions, events = _new_transactions(before, _snapshot(db_cursor))
    assert len(transactions) == 1 and len(events) == 1
    assert transactions[0]['operator_id'] == ('legacy-shared-key' if identity == 'shared' else identity)
    assert events[0]['operator_id'] == (None if identity == 'shared' else identity)


@pytest.mark.db
def test_receive_shortcut_preview_has_no_business_writes(client, db_cursor, named_actors, prepared):
    body = _payload(db_cursor, '/receive', prepared)
    body.pop('mode')
    before = _snapshot(db_cursor)
    response = _request(client, 'POST', '/receive/preview', prepared, body,
                        named_actors['Luz']['key'])
    assert response.status_code == 200, response.text
    assert _snapshot(db_cursor) == before


@pytest.mark.db
@pytest.mark.parametrize('name', ('ship', 'make', 'pack', 'adjust'))
def test_other_shortcut_commits_keep_actor_scope_and_shared_attribution(
        client, db_cursor, named_actors, prepared, name):
    body = _payload(db_cursor, '/' + name, prepared)
    body.pop('mode')
    before = _snapshot(db_cursor)
    denied = _request(client, 'POST', f'/{name}/commit', prepared, body,
                      named_actors['Arturo']['key'])
    assert denied.status_code == 403, denied.text
    assert denied.json()['detail'] == 'API key not authorized for this endpoint'
    assert _snapshot(db_cursor) == before
    response = _request(client, 'POST', f'/{name}/commit', prepared, body, main.API_KEY)
    assert response.status_code == 200, response.text
    transactions, events = _new_transactions(before, _snapshot(db_cursor))
    assert transactions and {r['operator_id'] for r in transactions} == {'legacy-shared-key'}
    assert events and {r['operator_id'] for r in events} == {None}


@pytest.mark.db
@pytest.mark.parametrize('name', sorted(SHORTCUTS))
def test_every_shortcut_commit_wrapper_passes_the_request_through(
        client, db_cursor, named_actors, prepared, name):
    """Direct call with an actor-bearing Request: proves attribution survives
    each commit wrapper without widening any route's HTTP scope."""
    from starlette.requests import Request
    body = _payload(db_cursor, '/' + name, prepared)
    body.pop('mode')
    actor = {'id': named_actors['Luz']['id'], 'name': 'Luz', 'role': 'floor'}
    request = Request({'type': 'http', 'method': 'POST', 'path': f'/{name}/commit',
                       'headers': [], 'query_string': b'',
                       'state': {'actor': actor, 'key_kind': 'actor'}})
    before = _snapshot(db_cursor)
    result = getattr(main, f'{name}_commit')(SHORTCUTS[name](**body), True, request=request)
    assert isinstance(result, dict) and result.get('success') is not False, result
    transactions, events = _new_transactions(before, _snapshot(db_cursor))
    assert transactions and {r['operator_id'] for r in transactions} == {'Luz'}
    assert events and {r['operator_id'] for r in events} == {'Luz'}


# F2: a customer auto-created inside POST /sales/orders or POST /ship is a
# metadata write with no ledger row of its own; it gets its own audit record
# in the same transaction as the order/shipment.
def _auto_create_body(cur, route, seed, new_name):
    if route == '/sales/orders':
        return {'customer_name': new_name,
                'lines': [{'product_name': seed['product_name'], 'quantity_lb': 20}]}
    body = _payload(cur, '/ship', seed)
    body['customer_name'] = new_name
    return body


@pytest.mark.db
@pytest.mark.parametrize('route', ['/sales/orders', '/ship'])
@pytest.mark.parametrize('identity', ('Miriam', 'Blubber', 'shared'))   # office + owner: floor may not
def test_auto_created_customer_is_audited_with_the_business_write(
        client, db_cursor, named_actors, prepared, route, identity):
    # Unique first word: no exact, fuzzy or prefix match, so it must auto-create.
    new_name = f'Zq{uuid4().hex[:10]} Autocreate'
    body = _auto_create_body(db_cursor, route, prepared, new_name)
    key = main.API_KEY if identity == 'shared' else named_actors[identity]['key']
    before = _snapshot(db_cursor)
    response = _request(client, 'POST', route, prepared, body, key)
    assert response.status_code == 200, response.text
    after = _snapshot(db_cursor)
    created = [r for r in after['customers'] if r['name'] == new_name]
    assert len(created) == 1
    audit = after['actor_write_audit'][len(before['actor_write_audit']):]
    customer_audit = [r for r in audit if r['target_table'] == 'customers']
    if identity == 'shared':
        assert audit == []
    else:
        assert [(r['actor_id'], r['operator_id'], r['method'], r['route'], r['target_id'])
                for r in customer_audit] == [
            (named_actors[identity]['id'], identity, 'POST', route, created[0]['id'])]
    if route == '/ship':
        transactions, _ = _new_transactions(before, after)
        assert {r['operator_id'] for r in transactions} == {
            'legacy-shared-key' if identity == 'shared' else identity}


@pytest.mark.db
@pytest.mark.parametrize('route', ['/sales/orders', '/ship'])
def test_auto_created_customer_audit_failure_rolls_back_everything(
        client, db_cursor, named_actors, prepared, route, monkeypatch):
    new_name = f'Zq{uuid4().hex[:10]} Autocreate'
    body = _auto_create_body(db_cursor, route, prepared, new_name)
    before = _snapshot(db_cursor)
    def fail_audit(cur, *args):
        cur.execute('SELECT 1 / 0')
    monkeypatch.setattr(main, '_record_actor_write', fail_audit)
    response = _request(client, 'POST', route, prepared, body, named_actors['Miriam']['key'])
    assert response.status_code == 500, response.text
    assert _snapshot(db_cursor) == before


# F7: RLS is enabled (not forced) on actor_write_audit. The backend role owns
# the table, so it bypasses RLS; any other non-superuser role, even one with
# table grants (as Supabase anon/authenticated may have), sees and writes nothing.
@pytest.mark.db
def test_audit_table_rls_blocks_other_roles_but_not_the_owning_backend_role(
        client, db_cursor, named_actors, prepared):
    import psycopg2
    db_cursor.execute("SELECT relrowsecurity, relforcerowsecurity FROM pg_class "
                      "WHERE oid = 'public.actor_write_audit'::regclass")
    assert tuple(db_cursor.fetchone().values()) == (True, False)
    token = uuid4().hex[:8]
    owner, outsider = f'pr66_backend_{token}', f'pr66_outsider_{token}'
    for role in (owner, outsider):  # rolled back with the test transaction
        db_cursor.execute(f'CREATE ROLE {role} NOLOGIN NOSUPERUSER NOBYPASSRLS')
        db_cursor.execute(f'GRANT USAGE ON SCHEMA public TO {role}')
        db_cursor.execute(f'GRANT ALL ON ALL TABLES IN SCHEMA public TO {role}')
        db_cursor.execute(f'GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO {role}')
    db_cursor.execute(f'ALTER TABLE public.actor_write_audit OWNER TO {owner}')
    try:
        # The real app write path, running as the non-superuser owning role.
        db_cursor.execute(f'SET LOCAL ROLE {owner}')
        response = _request(client, 'POST', '/customers', prepared,
                            _payload(db_cursor, '/customers', prepared),
                            named_actors['Miriam']['key'])
        assert response.status_code == 200, response.text
        customer_id = response.json()['customer_id']
        db_cursor.execute("SELECT operator_id FROM actor_write_audit "
                          "WHERE target_table = 'customers' AND target_id = %s", (customer_id,))
        assert [r['operator_id'] for r in db_cursor.fetchall()] == ['Miriam']

        db_cursor.execute(f'SET LOCAL ROLE {outsider}')
        db_cursor.execute('SELECT count(*) AS n FROM actor_write_audit')
        assert db_cursor.fetchone()['n'] == 0
        db_cursor.execute('SAVEPOINT rls_insert_check')
        with pytest.raises(psycopg2.Error, match='row-level security'):
            db_cursor.execute(
                "INSERT INTO actor_write_audit (actor_id, operator_id, method, route, "
                "target_table, target_id) VALUES (%s, 'Forged', 'POST', '/customers', "
                "'customers', %s)", (named_actors['Arturo']['id'], customer_id))
        db_cursor.execute('ROLLBACK TO SAVEPOINT rls_insert_check')
        db_cursor.execute('RELEASE SAVEPOINT rls_insert_check')
    finally:
        db_cursor.execute('RESET ROLE')


# Order creation must audit both callers of the shared core, including intake.
def _intake_body(cur, seed):
    token = uuid4().hex
    doc = _insert_document(cur, path=f'review/{token}.png', kind='sales',
                           status='extracted', sha=token)
    return _approve_payload(doc['id'], seed['customer_id'], [
        _approve_line(seed['product_id'], 20, code='AUDIT-1', save_alias=True),
        _approve_line(seed['product_id'], 30, code='AUDIT-2', save_alias=True),
    ], po=f'AUDIT-{token}')


def _intake_snapshot(cur):
    result = _snapshot(cur)
    for table in ('purchase_documents', 'customer_product_aliases'):
        cur.execute(f'SELECT row_to_json(t) AS data FROM {table} t ORDER BY id')
        result[table] = [r['data'] for r in cur.fetchall()]
    return result


@pytest.mark.db
@pytest.mark.parametrize('identity', NAMES + ('shared',))
def test_intake_creation_audits_order_and_each_line(
        client, db_cursor, named_actors, prepared, identity):
    body = _intake_body(db_cursor, prepared)
    key = main.API_KEY if identity == 'shared' else named_actors[identity]['key']
    before = _intake_snapshot(db_cursor)
    response = client.post('/sales/orders/extract/approve', json=body,
                           headers={'X-API-Key': key})
    if _denied(identity, 'POST', '/sales/orders/extract/approve'):
        assert response.status_code == 403, response.text
        assert response.json()['detail']['error_code'] == 'ROLE_NOT_ALLOWED'
        assert _intake_snapshot(db_cursor) == before
        return
    assert response.status_code == 201, response.text
    result = response.json()
    assert len(result['lines']) == 2 and result['aliases_saved'] == 2
    after = _intake_snapshot(db_cursor)
    assert after['customers'] == before['customers']
    doc = next(r for r in after['purchase_documents'] if r['id'] == body['document_id'])
    assert doc['status'] == 'approved'
    audit = after['actor_write_audit'][len(before['actor_write_audit']):]
    if identity == 'shared':
        assert audit == []
    else:
        assert len(audit) == 3  # No duplicate audit from either caller.
        assert {(r['target_table'], r['target_id']) for r in audit} == {
            ('sales_orders', result['order_id']),
            *(('sales_order_lines', line['line_id']) for line in result['lines']),
        }
        assert {(r['actor_id'], r['operator_id'], r['method'], r['route'])
                for r in audit} == {
            (named_actors[identity]['id'], identity, 'POST', '/sales/orders/extract/approve')}


@pytest.mark.db
@pytest.mark.parametrize('route', ['/sales/orders', '/sales/orders/extract/approve'])
@pytest.mark.parametrize('fail_at', [1, 3])
def test_core_audit_failure_rolls_back_both_creation_paths(
        client, db_cursor, named_actors, prepared, route, fail_at, monkeypatch):
    if route.endswith('/approve'):
        body = _intake_body(db_cursor, prepared)
    else:
        body = {'customer_name': 'Zq' + uuid4().hex,
                'lines': [{'product_name': prepared['product_name'], 'quantity_lb': qty}
                          for qty in (20, 30)]}
    before = _intake_snapshot(db_cursor)
    original = main._record_actor_write
    calls = []

    def fail_core_audit(cur, request, table, target_id):
        # Let manual customer creation succeed before testing core rollback.
        if table in ('sales_orders', 'sales_order_lines'):
            calls.append(table)
            if len(calls) == fail_at:
                cur.execute('SELECT 1 / 0')
        original(cur, request, table, target_id)

    monkeypatch.setattr(main, '_record_actor_write', fail_core_audit)
    response = client.post(route, json=body,
                           headers={'X-API-Key': named_actors['Miriam']['key']})
    assert response.status_code == 500, response.text
    assert calls == ['sales_orders', 'sales_order_lines', 'sales_order_lines'][:fail_at]
    assert _intake_snapshot(db_cursor) == before


@pytest.mark.db
def test_shared_intake_creation_still_works_without_audit_table(
        client, db_cursor, named_actors, prepared):
    body = _intake_body(db_cursor, prepared)
    db_cursor.execute('DROP TABLE actor_write_audit')  # Rolled back by fixture.
    response = client.post('/sales/orders/extract/approve', json=body,
                           headers={'X-API-Key': main.API_KEY})
    assert response.status_code == 201, response.text
    assert len(response.json()['lines']) == 2


def _audit_down_sql():
    # Run the unchanged body within the fixture's rollback-only transaction.
    sql = (MIGRATION.parent / 'down/056_actor_write_audit_down.sql').read_text()
    return sql.split('\nBEGIN;', 1)[1].rsplit('\nCOMMIT;', 1)[0]


@pytest.mark.db
@pytest.mark.parametrize('state', ['empty', 'nonempty', 'exported', 'absent'])
def test_audit_down_requires_explicit_export_for_nonempty_table(db_cursor, named_actors, state):
    import psycopg2
    if state in ('nonempty', 'exported'):
        db_cursor.execute(
            "INSERT INTO actor_write_audit (actor_id, operator_id, method, route, "
            "target_table, target_id) VALUES (%s, 'Luz', 'POST', '/customers', 'customers', 1)",
            (named_actors['Luz']['id'],))
    if state == 'absent':
        db_cursor.execute('DROP TABLE actor_write_audit')
    db_cursor.execute('SELECT row_to_json(m) AS data FROM migration_markers m ORDER BY name')
    markers_before = [r['data'] for r in db_cursor.fetchall()]
    if state == 'nonempty':
        before = _snapshot(db_cursor)
        db_cursor.execute('SAVEPOINT audit_down_guard')
        with pytest.raises(psycopg2.errors.RaiseException, match='056 rollback refused'):
            db_cursor.execute(_audit_down_sql())
        db_cursor.execute('ROLLBACK TO SAVEPOINT audit_down_guard')
        assert _snapshot(db_cursor) == before
        db_cursor.execute('SELECT row_to_json(m) AS data FROM migration_markers m ORDER BY name')
        assert [r['data'] for r in db_cursor.fetchall()] == markers_before
    else:
        if state == 'exported':
            db_cursor.execute("SET LOCAL factory_ledger.confirm_audit_export = 'yes'")
        db_cursor.execute(_audit_down_sql())
        db_cursor.execute("SELECT to_regclass('public.actor_write_audit') AS audit_table")
        assert db_cursor.fetchone()['audit_table'] is None
        db_cursor.execute('SELECT row_to_json(m) AS data FROM migration_markers m ORDER BY name')
        assert [r['data'] for r in db_cursor.fetchall()] == [
            r for r in markers_before if r['name'] != '056_actor_write_audit']
