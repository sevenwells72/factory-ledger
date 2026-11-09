"""Product archive (owner request, Michael, 2026-11-09; product_archive.py, migration 093).

Archive = hide from active lists, never delete. Real HTTP routes against the
LOCAL test database (never staging/prod).
"""
from pathlib import Path
from uuid import uuid4

import psycopg2
import pytest

import main
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_trusted_browsers import migrated, people  # noqa: F401  (091 + the four names)

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[1]
MIGRATION = ROOT / 'migrations/093_product_archive.sql'
DOWN = ROOT / 'migrations/down/093_product_archive_down.sql'


def named(name):
    return {'X-FL-Person': name}


@pytest.fixture
def archive_ready(db_cursor):
    db_cursor.execute(MIGRATION.read_text())


@pytest.fixture
def tag():
    return 'ARCHT ' + uuid4().hex[:6].upper()


def product(cur, name, type_='ingredient', on_hand=0, active=None):
    cur.execute("INSERT INTO products (name, type, uom, active) VALUES (%s, %s, 'lb', %s) RETURNING id",
                (name, type_, active))
    pid = cur.fetchone()['id']
    lot_code = None
    if on_hand:
        lot_code = f'ARCH-LOT-{pid}'
        cur.execute('INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id', (pid, lot_code))
        lot_id = cur.fetchone()['id']
        cur.execute("""INSERT INTO transactions (type, timestamp, status, occurred_at, business_date, operator_id)
                       VALUES ('make', clock_timestamp() AT TIME ZONE 'UTC', 'posted', clock_timestamp(),
                               timezone('America/New_York', clock_timestamp())::date, 'archive-test')
                       RETURNING id""")
        txn = cur.fetchone()['id']
        cur.execute('INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) '
                    'VALUES (%s, %s, %s, %s)', (txn, pid, lot_id, on_hand))
    return pid, lot_code


def recipe(cur, batch_id, ingredient_id, lb=10):
    cur.execute('INSERT INTO batch_formulas (product_id, ingredient_product_id, quantity_lb) VALUES (%s, %s, %s)',
                (batch_id, ingredient_id, lb))


def ledger(cur):
    cur.execute('SELECT count(*) AS t FROM transactions')
    t = cur.fetchone()['t']
    cur.execute('SELECT count(*) AS l, COALESCE(sum(quantity_lb), 0) AS q FROM transaction_lines')
    row = cur.fetchone()
    cur.execute('SELECT count(*) AS n FROM lots')
    return t, row['l'], float(row['q']), cur.fetchone()['n']


def state(cur, pid):
    cur.execute("SELECT active, archived_at IS NOT NULL AS stamped, archived_by FROM products WHERE id=%s", (pid,))
    return dict(cur.fetchone())


def archive(client_, ids, who='Michael', restore=False):
    path = '/admin/products/unarchive' if restore else '/admin/products/archive'
    return client_.post(path, json={'product_ids': ids}, headers=named(who))


def config(monkeypatch, **cfg):
    monkeypatch.setattr(main, '_load_dashboard_config', lambda: cfg)


def test_michael_archives_without_touching_history_and_unarchive_restores(client, db_cursor, people, archive_ready, tag):
    pid, _ = product(db_cursor, f'{tag} BS Cacao', on_hand=1400)
    before = ledger(db_cursor)
    r = archive(client, [pid])
    assert r.status_code == 200, r.text
    assert [p['product_id'] for p in r.json()['archived']] == [pid] and r.json()['archived_by'] == 'Michael'
    assert state(db_cursor, pid) == {'active': False, 'stamped': True, 'archived_by': 'Michael'}
    assert ledger(db_cursor) == before                 # stock, lots, transactions untouched
    again = archive(client, [pid])                     # idempotent
    assert again.status_code == 200 and again.json()['already_archived'] == [pid]
    r = archive(client, [pid], restore=True)
    assert r.status_code == 200, r.text
    assert [p['product_id'] for p in r.json()['restored']] == [pid]
    assert state(db_cursor, pid) == {'active': True, 'stamped': False, 'archived_by': None}
    assert ledger(db_cursor) == before


@pytest.mark.parametrize('who', ['Luz', 'Arturo', 'Miriam'])
def test_only_michael_may_archive(client, db_cursor, people, archive_ready, tag, who):
    pid, _ = product(db_cursor, f'{tag} Item')
    r = archive(client, [pid], who=who)
    assert r.status_code == 403, r.text
    assert state(db_cursor, pid)['active'] is None


def test_shared_keys_cannot_archive(client, db_cursor, archive_ready, tag):
    pid, _ = product(db_cursor, f'{tag} Item')
    master = client.post('/admin/products/archive', json={'product_ids': [pid]}, headers={'X-API-Key': main.API_KEY})
    assert master.status_code == 403 and master.json()['detail']['error_code'] == 'OWNER_REQUIRED'
    assert state(db_cursor, pid)['active'] is None


def test_without_migration_093_nothing_changes(client, db_cursor, people, tag):
    pid, _ = product(db_cursor, f'{tag} Item')
    db_cursor.execute("SELECT 1 FROM migration_markers WHERE name='093_product_archive'")
    if db_cursor.fetchone():
        pytest.skip('test database already carries 093')
    r = archive(client, [pid])
    assert r.status_code == 503 and r.json()['detail']['error_code'] == 'ARCHIVE_UNAVAILABLE'


def test_ingredient_used_by_active_recipe_is_blocked_unless_recipe_archived_too(client, db_cursor, people, archive_ready, tag):
    ing, _ = product(db_cursor, f'{tag} BS Oats')
    batch, _ = product(db_cursor, f'Batch {tag} BS Granola 350', 'batch')
    recipe(db_cursor, batch, ing)
    r = archive(client, [ing])
    assert r.status_code == 409, r.text
    detail = r.json()['detail']
    assert detail['error_code'] == 'USED_BY_ACTIVE_RECIPE'
    assert f'Batch {tag} BS Granola 350' in detail['message']
    assert detail['blocked'] == [{'product_id': ing, 'product_name': f'{tag} BS Oats',
                                  'recipes': [f'Batch {tag} BS Granola 350']}]
    assert state(db_cursor, ing)['active'] is None
    # Recipe and ingredient together (bulk) is allowed; so is the ingredient once the recipe is archived.
    r = archive(client, [ing, batch])
    assert r.status_code == 200, r.text
    assert {p['product_id'] for p in r.json()['archived']} == {ing, batch}


def test_bulk_is_all_or_nothing(client, db_cursor, people, archive_ready, tag):
    free, _ = product(db_cursor, f'{tag} Free')
    used, _ = product(db_cursor, f'{tag} Used')
    batch, _ = product(db_cursor, f'Batch {tag} Live', 'batch')
    recipe(db_cursor, batch, used)
    assert archive(client, [free, used]).status_code == 409
    assert state(db_cursor, free)['active'] is None
    assert archive(client, [free, 2147483647]).status_code == 404
    assert state(db_cursor, free)['active'] is None


def test_ingredient_lists_hide_archived_and_drop_all_archived_sections(client, db_cursor, people, archive_ready, tag, monkeypatch):
    a, _ = product(db_cursor, f'{tag} BS Nibs', on_hand=1400)
    b, _ = product(db_cursor, f'{tag} BS Salt')
    sun, _ = product(db_cursor, f'{tag} Sunflower Seeds', on_hand=2114)
    config(monkeypatch, ingredient_categories=[
        {'id': 'bs', 'title': 'BS Ingredients', 'unit': 'lb', 'items': [f'{tag} BS Nibs', f'{tag} BS Salt']},
        {'id': 'gen', 'title': 'Granola General', 'unit': 'lb', 'items': [f'{tag} Sunflower Seeds', f'{tag} BS Nibs']}])
    assert archive(client, [a, b]).status_code == 200
    cats = {c['id']: c for c in client.get('/dashboard/api/inventory/ingredients').json()['categories']}
    assert 'bs' not in cats                                         # whole section hidden
    assert [i['name'] for i in cats['gen']['items']] == [f'{tag} Sunflower Seeds']
    assert cats['gen']['items'][0]['on_hand'] == 2114.0 and cats['gen']['missing_skus'] == []
    shown = {c['id']: c for c in client.get('/dashboard/api/inventory/ingredients?include_archived=true').json()['categories']}
    assert shown['bs']['all_archived'] is True
    nibs = shown['bs']['items'][0]
    assert nibs['archived'] is True and nibs['archived_by'] == 'Michael' and nibs['on_hand'] == 1400.0
    assert sun


def test_batches_and_finished_goods_hide_archived(client, db_cursor, people, archive_ready, tag, monkeypatch):
    batch, _ = product(db_cursor, f'Batch {tag} BS Dark Chocolate Granola 350', 'batch', on_hand=1078)
    fg, _ = product(db_cursor, f'{tag} BS Granola – Dark Chocolate – 6x7 OZ Case', 'finished')
    config(monkeypatch, batch_skus=[{'name': f'Batch {tag} BS Dark Chocolate Granola 350'}],
           finished_goods_panels=[{'id': 'retail_bs', 'title': '6x7 OZ Retail Cases (BS Line)',
                                   'skus': [f'{tag} BS Granola – Dark Chocolate – 6x7 OZ Case']}])
    monkeypatch.setattr(main, '_production_family_from_name', lambda name: 'granola')
    assert archive(client, [batch, fg]).status_code == 200
    batches = client.get('/dashboard/api/inventory/batches').json()
    assert batch not in [b['product_id'] for b in batches['batches']] and batches['missing_skus'] == []
    shown = client.get('/dashboard/api/inventory/batches?include_archived=true').json()['batches']
    assert [b['archived'] for b in shown if b['product_id'] == batch] == [True]
    assert client.get('/dashboard/api/inventory/finished-goods').json()['panels'] == []
    panel = client.get('/dashboard/api/inventory/finished-goods?include_archived=true').json()['panels'][0]
    assert panel['all_archived'] is True and panel['products'][0]['archived'] is True


def test_history_lookup_and_search_still_show_archived(client, db_cursor, people, archive_ready, tag):
    pid, lot_code = product(db_cursor, f'Batch {tag} BS Peanut', 'batch', on_hand=350)
    assert archive(client, [pid]).status_code == 200
    lot = client.get('/dashboard/api/lot/' + lot_code)
    assert lot.status_code == 200 and tag in lot.text
    assert client.get(f'/dashboard/api/product/{pid}/lots').status_code == 200
    found = client.get('/dashboard/api/search', params={'q': tag}).json()
    assert [(p['product_id'], p['archived']) for p in found['products']] == [(pid, True)]
    by_lot = client.get('/dashboard/api/search', params={'q': lot_code}).json()
    assert [(lt['product_id'], lt['product_archived']) for lt in by_lot['lots']] == [(pid, True)]


def test_supplies_never_flag_archived_as_low(client, db_cursor, people, archive_ready, tag):
    pid, _ = product(db_cursor, f'{tag} Bag – Printed – BS 7 oz', 'packaging')
    db_cursor.execute('UPDATE products SET low_stock_threshold=100 WHERE id=%s', (pid,))
    row = lambda: next(i for i in client.get('/supplies/inventory', headers={'X-API-Key': main.API_KEY}).json()['items']
                       if i['product_id'] == pid)
    assert row()['is_low'] is True
    assert archive(client, [pid]).status_code == 200
    assert row()['is_low'] is False and row()['active'] is False


def test_migration_093_rerunnable_and_down_refuses_while_attributed(client, db_cursor, people, archive_ready, tag):
    db_cursor.execute(MIGRATION.read_text())          # idempotent
    pid, _ = product(db_cursor, f'{tag} Item')
    assert archive(client, [pid]).status_code == 200
    db_cursor.execute('SAVEPOINT down093')
    with pytest.raises(psycopg2.Error, match='keep migration 093'):
        db_cursor.execute(DOWN.read_text())
    db_cursor.execute('ROLLBACK TO SAVEPOINT down093')
    assert archive(client, [pid], restore=True).status_code == 200
    db_cursor.execute(DOWN.read_text())
    db_cursor.execute("SELECT count(*) AS n FROM information_schema.columns WHERE table_name='products' AND column_name='archived_by'")
    assert db_cursor.fetchone()['n'] == 0
