"""Offline cleanup safety tests: no staging connections, credentials or CLI apply."""
from copy import deepcopy
from pathlib import Path
import httpx
import pytest

from scripts import staging_pilot_cleanup as cleanup

URI = 'postgresql://postgres.jygmyvxnxdjiiilhxseq:dummy@aws-0-us-east-1.pooler.supabase.com:6543/postgres'
ROUTES = ('/adjust/prepare', '/sales/orders/{order_id}/cancel/prepare',
          '/exceptions/{exception_id}/resolve/prepare', '/tickets/{ticket}/commit', '/receipts/{receipt_number}')


class ReadAPI:
    def __init__(self):
        self.calls = []
        self.schema = {'paths': {r: {} for r in ROUTES}, 'components': {'schemas': {'Version': 1}}}
        self.schema['paths'].update({'/admin/products/{product_id}': {}, '/customers/{customer_id}': {}})

    def call(self, method, path, payload=None):
        self.calls.append((method, path, payload))
        assert method == 'GET', 'dry-run attempted a mutation'
        if path == '/openapi.json':
            return self.schema
        if path == '/auth/whoami':
            return {'key_kind': 'actor', 'actor': {'id': 1, 'role': 'owner', 'name': 'Staging owner'}}
        if path.startswith('/sales/orders/'):
            return {'fulfillment': 'partial' if path.endswith('/51') else 'unshipped'}
        if path.startswith('/exceptions/'):
            return {'allowed_actions': ['counted', 'missing_movement', 'voided'] if path.endswith('/31')
                    else ['identified', 'written_off'] if path.endswith('/32') else ['resolved', 'waived']}
        raise AssertionError(path)


@pytest.fixture
def snap(monkeypatch):
    monkeypatch.setattr(cleanup, 'PROTECTED_EXCEPTION_IDS', frozenset({900}))
    def lot(id, code, balance, product=3, uom='lb', **catalog):
        return dict(id=id, product_id=product, lot_code=code, status='active', kosher_tier='extra',
                    product_name='Real oats', product_active=True, balance_lb=balance,
                    catalog=dict(name='Real oats', uom=uom, **catalog))
    def exception(id, kind, code, name='STG test ingredient'):
        return dict(id=id, kind=kind, status='open', lot_code=code, product_name=name, detail={}, customer_name=None)
    return dict(lots=[lot(10, 'STG-positive', '25'), lot(11, 'STG-negative', '-20'),
                      lot(12, 'STG-zero', '0'), lot(13, 'OPEN261014-real', '200'),
                      lot(14, 'STAGING-SHARED-LOT', '100'), lot(15, 'STG-bags', '30', uom='bag', pack_size_lbs=25)],
                products=[dict(id=1000000001, name='STG test', active=True), dict(id=283, name='Real extra kosher', active=True)],
                customers=[dict(id=1000000001, name='STAGING Test Customer', active=True), dict(id=2, name='Real account', active=True)],
                orders=[dict(id=id, order_number=f'SO-{id}', customer_id=1000000001,
                             customer_name='STAGING Test Customer', state='open', status='confirmed') for id in (50,51)],
                shortages=[dict(id=30, lot_id=11, status='open', exception_id=31)],
                exceptions=[exception(31, 'SHORTAGE', 'STG-negative'), exception(32, 'UNIDENTIFIED_LOT', 'STG-test'),
                            exception(33, 'SHIFT_DISCREPANCY', None), exception(900, 'UNIDENTIFIED_LOT', 'renamed-test-label'),
                            exception(901, 'UNIDENTIFIED_LOT', 'OPEN261014-protected')],
                write_counts={'tickets': 0, 'transactions': 0, 'actor_audits': 0}, order_lines=[])


@pytest.mark.parametrize('base', ['https://fastapi-production-b73a.up.railway.app',
    cleanup.BASE+'/', cleanup.BASE+'/other', cleanup.BASE+'.evil.example',
    cleanup.BASE.replace('https:', 'http:'), 'https://user:pass@'+cleanup.BASE[8:]])
def test_refuses_non_exact_api_origin(base):
    with pytest.raises(cleanup.CleanupError):
        cleanup.guard(base, URI)


@pytest.mark.parametrize('uri', [URI.replace('aws-0-', 'aws-1-'),
    URI.replace(cleanup.DB_REF, 'vrafvwcdpcijvxdvefpr'), URI+'?host=production.example',
    URI.replace('/postgres', '/other'), URI.replace(':6543/', ':9999/')])
def test_refuses_production_or_rerouted_database(uri):
    with pytest.raises((cleanup.CleanupError, RuntimeError)):
        cleanup.guard(cleanup.BASE, uri)


@pytest.mark.parametrize('name', ['PGHOST', 'PGHOSTADDR', 'PGSERVICE', 'PGSERVICEFILE', 'PGOPTIONS', 'PGPORT', 'PGDATABASE', 'PGUSER'])
def test_refuses_libpq_overrides(monkeypatch, name):
    monkeypatch.setenv(name, 'unsafe')
    with pytest.raises(cleanup.CleanupError):
        cleanup.guard(cleanup.BASE, URI)


def test_refuses_production_environment_even_with_staging_file(monkeypatch):
    monkeypatch.setenv('DATABASE_URL', URI.replace(cleanup.DB_REF, 'vrafvwcdpcijvxdvefpr'))
    with pytest.raises(RuntimeError):
        cleanup.guard(cleanup.BASE, URI)


def test_dry_run_cannot_prepare_or_commit():
    api = cleanup.API(cleanup.BASE, 'offline-dummy')
    try:
        for path in ('/adjust/prepare', '/tickets/fake/commit', '/customers/1'):
            with pytest.raises(cleanup.CleanupError, match='Dry run refuses'):
                api.call('POST', path, {})
    finally:
        api.close()


def test_http_errors_redact_capability_tokens():
    api = cleanup.API(cleanup.BASE, 'offline-dummy', apply=True)
    api.http.close()
    api.http = httpx.Client(base_url=cleanup.BASE, transport=httpx.MockTransport(
        lambda request: httpx.Response(500, json={})))
    try:
        with pytest.raises(cleanup.CleanupError) as error:
            api.call('POST', '/tickets/secret-capability/commit', {})
        assert 'secret-capability' not in str(error.value)
        assert '[redacted]' in str(error.value)
    finally:
        api.close()


def test_plan_preserves_open_stock_and_fixed_exception_ids(snap):
    api = ReadAPI()
    plan = cleanup.build_plan(snap, api)
    lots = [r for r in plan['items'] if r['category'] == 'lots']
    assert {r['id'] for r in lots} == {10,11,12,15}
    assert next(r for r in lots if r['id'] == 11)['status'] == 'blocked'
    assert next(r for r in lots if r['id'] == 12)['status'] == 'noop'
    assert next(r for r in lots if r['id'] == 15)['owner_approval']  # 30 bags × 25 lb
    assert {r['id'] for r in plan['protected_exceptions']} == {900,901}
    assert not ({900,901} & {r['id'] for r in plan['items'] if r['category'] == 'exceptions'})
    assert all(method == 'GET' for method,_,_ in api.calls)


def test_supported_exception_semantics_and_unsupported_shortage(snap):
    plan = cleanup.build_plan(snap, ReadAPI())
    shortage = next(r for r in plan['items'] if r['category'] == 'shortages')
    assert shortage['status'] == 'blocked' and shortage['payload']['resolution_kind'] == 'waived'
    unidentified = next(r for r in plan['items'] if r['id'] == 32 and r['category'] == 'exceptions')
    assert unidentified['payload']['resolution_kind'] == 'written_off'
    assert unidentified['after']['status'] == 'waived'
    assert all(not r.get('payload', {}).get('resolves_shortage_ids') for r in plan['items'] if r.get('payload'))


def test_no_direct_deactivations_or_shipped_order_cancellations(snap):
    plan = cleanup.build_plan(snap, ReadAPI())
    for category in ('products', 'customers'):
        selected = [r for r in plan['items'] if r['category'] == category]
        assert len(selected) == 1 and selected[0]['status'] == 'blocked' and selected[0]['path'] is None
    assert [r['id'] for r in plan['items'] if r['category'] == 'orders'] == [50]
    assert [r['id'] for r in plan['preserved_test_orders']] == [51]


def test_component_schema_changes_invalidate_review_hash(snap):
    api = ReadAPI()
    original = cleanup.plan_hash(cleanup.build_plan(snap, api))
    api.schema['components']['schemas']['Version'] = 2
    assert cleanup.plan_hash(cleanup.build_plan(snap, api)) != original


def test_markdown_command_quotes_paths_and_records_every_item(snap):
    plan = cleanup.build_plan(snap, ReadAPI())
    report = cleanup.markdown(plan, Path('/tmp/space here/plan.md'), python='/tmp/python path',
                              options=['--key-file', '/tmp/key path'])
    assert "'/tmp/python path'" in report and "'/tmp/space here/plan.md'" in report
    assert "--key-file '/tmp/key path'" in report
    assert 'STAGING PILOT CLEANUP' in report and 'written_off' in report
    for item in plan['items']:
        assert f"| {item['id']} |" in report


def test_reader_is_transaction_scoped_and_rolls_back(monkeypatch):
    calls = []
    class Cursor:
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def execute(self, sql): calls.append(sql)
    class Connection:
        def cursor(self, **kwargs): return Cursor()
        def rollback(self): calls.append('rollback')
        def close(self): calls.append('close')
    def connect(uri, **kwargs):
        assert kwargs['port'] == 5432 and kwargs['sslmode'] == 'require'
        return Connection()
    monkeypatch.setattr(cleanup.psycopg2, 'connect', connect)
    with cleanup.reader('offline'):
        pass
    assert calls[0].endswith('REPEATABLE READ READ ONLY')
    assert calls[-2:] == ['rollback', 'close']
    assert not any('SESSION' in c or 'default_transaction_read_only' in c for c in calls)


def test_journal_exclusivity_prevents_second_executor(tmp_path):
    journal = tmp_path/'journal.jsonl'
    plan = dict(items=[])
    cleanup.apply_plan(plan, None, journal)
    with pytest.raises(cleanup.CleanupError, match='journal already exists'):
        cleanup.apply_plan(plan, None, journal)


def test_repurposed_order_stops_before_commit(snap, tmp_path):
    plan = cleanup.build_plan(snap, ReadAPI())
    plan['items'] = [r for r in plan['items'] if r['category'] == 'orders']
    class PreparedAPI:
        def call(self, method, path, body=None):
            assert path.endswith('/cancel/prepare'), 'must not commit a repurposed order'
            return dict(can_commit=True, ticket_id=1, payload_hash='offline', draft=dict(order=dict(
                id=50, order_number='SO-50', customer_id=2, customer='Real account', state='open')))
    with pytest.raises(cleanup.CleanupError, match='reviewed test customer'):
        cleanup.apply_plan(plan, PreparedAPI(), tmp_path/'journal')


def test_stale_lot_draft_stops_before_commit(snap, tmp_path):
    plan = cleanup.build_plan(snap, ReadAPI())
    plan['items'] = [r for r in plan['items'] if r['category'] == 'lots' and r['id'] == 10]
    class StaleAPI:
        def call(self, method, path, body=None):
            if method == 'GET':
                return dict(lot_code='STG-positive', product_id=3, quantity_on_hand=25)
            assert path == '/adjust/prepare', 'must not commit a stale count'
            return dict(can_commit=True, ticket_id=1, payload_hash='offline', draft=dict(
                lot_id=10, current_quantity_lb=26, new_balance_lb=1, note=cleanup.REASON))
    with pytest.raises(cleanup.CleanupError, match='differs from reviewed plan'):
        cleanup.apply_plan(plan, StaleAPI(), tmp_path/'journal')


def test_post_commit_state_change_stops_before_next_item():
    with pytest.raises(cleanup.CleanupError, match='after commit'):
        cleanup.verify_receipt(None, {}, {'state_changed': True})
