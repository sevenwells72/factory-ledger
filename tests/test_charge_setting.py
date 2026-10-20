"""Charge or Product (PR #143, migration 085): explicit choice, one check, human answer."""
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import HTTPException

import charge_setting
import exceptions_enforcement
import main
import permissions
from scripts.exceptions_sweep import sweep
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import headers, error

ROOT = Path(__file__).resolve().parent.parent
UP = (ROOT / 'migrations/085_charge_setting_check.sql').read_text()
DOWN = (ROOT / 'migrations/down/085_charge_setting_check_down.sql').read_text()


def tag():
    return uuid4().hex[:8].upper()


# ── Rules without a database ──────────────────────────────────────

@pytest.mark.parametrize('name,words', [
    ('Pallet Charge', True), ('Delivery Fee', True), ('FREIGHT', True), ('Fuel surcharge', True),
    ('Batch Setton Coffee Granola #12', True),   # "fee" inside Coffee: asked once, never decided
    ('Classic Granola 25 LB', False), ('Shrink wrap', False), ('', False), (None, False)])
def test_charge_words(name, words):
    assert charge_setting.has_charge_words(name) is words


@pytest.mark.parametrize('value', [None, '', '   '])
def test_no_choice_is_refused_in_both_languages(value):
    with pytest.raises(HTTPException) as problem:
        charge_setting.require_item_kind(value)
    detail = problem.value.detail
    assert problem.value.status_code == 422 and detail['error_code'] == 'ITEM_KIND_REQUIRED'
    assert detail['message'] == 'Choose Product or Charge (pallet, freight, delivery…).'
    assert detail['message_es'] == 'Elige Producto o Cargo (palé, flete, entrega…).'


def test_choice_values():
    assert charge_setting.require_item_kind('product') is False
    assert charge_setting.require_item_kind(' Charge ') is True
    with pytest.raises(HTTPException) as problem:
        charge_setting.require_item_kind('service')
    assert problem.value.detail['error_code'] == 'ITEM_KIND_INVALID'
    with pytest.raises(HTTPException) as problem:
        charge_setting.require_item_kind('charge', charge_allowed=False, what='batch')
    assert problem.value.detail['error_code'] == 'ITEM_KIND_NOT_ALLOWED'
    assert 'Producto' in problem.value.detail['message_es']


def test_check_text_both_languages():
    text = charge_setting.check_text({'name': 'Setton Coffee Granola', 'marked_as': 'product',
                                      'reason': 'charge_words_marked_product'})
    assert text['question'] == 'Confirm: Charge or Product?' and text['question_es'] == 'Confirmar: ¿Cargo o Producto?'
    assert text['message'] == ('Confirm: Charge or Product? Now marked Product, but its name contains "fee". '
                               'Nothing changes until someone confirms.')
    assert text['message_es'].startswith('Confirmar: ¿Cargo o Producto? Ahora está marcado como Producto, pero su nombre contiene "fee".')
    other = charge_setting.check_text({'name': 'Shrink wrap', 'marked_as': 'charge', 'reason': 'charge_without_words'})
    assert 'Now marked Charge (pallet, freight, delivery…), but its name has no charge word' in other['message']
    assert 'ninguna palabra de cargo' in other['message_es']


@pytest.mark.parametrize('role,allowed', [(role, role in ('owner', 'office')) for role in permissions.ROLES])
def test_owner_and_office_manage_and_confirm(role, allowed):
    assert permissions.allowed('manage_catalog', role) is allowed
    assert permissions.allowed('confirm_item_kind', role) is allowed


def test_python_and_sql_word_lists_match():
    assert '(' + '|'.join(charge_setting.CHARGE_WORDS) + ')' in UP


# ── Database ──────────────────────────────────────────────────────



def checks(cur, product_id=None, status=('open', 'escalated')):
    cur.execute('''SELECT * FROM exceptions WHERE kind='CHARGE_SETTING_CHECK'
                   AND (%s::int IS NULL OR product_id=%s) AND status = ANY(%s) ORDER BY id''',
                (product_id, product_id, list(status)))
    return cur.fetchall()


def item(cur, product_id):
    cur.execute('SELECT * FROM products WHERE id=%s', (product_id,))
    return cur.fetchone()


def create(client, actors, role='office', **body):
    return client.post('/catalog/items', json=body, headers=headers(actors[role]['key']))


@pytest.mark.db
@pytest.mark.parametrize('body', [{}, {'item_kind': None}, {'item_kind': ''}])
def test_office_create_without_choice_is_refused_and_writes_nothing(client, actors, db_cursor, body):
    name = 'Granola ' + tag()
    response = create(client, actors, name=name, type='finished', **body)
    error(response, 422, 'ITEM_KIND_REQUIRED')
    assert response.json()['detail']['message_es'] == 'Elige Producto o Cargo (palé, flete, entrega…).'
    db_cursor.execute('SELECT count(*) AS n FROM products WHERE name=%s', (name,))
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.db
@pytest.mark.parametrize('role', ['office', 'owner'])
def test_office_creates_product_and_charge_exactly_as_chosen(client, actors, db_cursor, role):
    product = create(client, actors, role, name='Classic Granola ' + tag(), item_kind='product', type='finished', case_size_lb=25)
    assert product.status_code == 200, product.text
    made = product.json()['item']
    assert made['item_kind'] == 'product' and made['item_kind_label_es'] == 'Producto'
    assert made['is_service'] is False and made['case_size_lb'] == 25 and made['check_exception_id'] is None
    charge = create(client, actors, role, name='Delivery Fee ' + tag(), item_kind='charge', type='finished', case_size_lb=25)
    made = charge.json()['item']
    # A charge is per unit: no pounds, no case size, whatever else was sent.
    assert made['is_service'] is True and made['uom'] == 'each' and made['case_size_lb'] is None
    assert made['item_kind_label'] == 'Charge (pallet, freight, delivery…)' and made['check_exception_id'] is None
    db_cursor.execute("SELECT count(*) AS n FROM actor_write_audit WHERE target_table='products' AND target_id=%s", (made['id'],))
    assert db_cursor.fetchone()['n'] == 1


@pytest.mark.db
def test_product_needs_a_type_and_floor_or_shared_keys_cannot_create(client, actors):
    error(create(client, actors, name='X ' + tag(), item_kind='product'), 422, 'ITEM_TYPE_REQUIRED')
    error(create(client, actors, 'floor', name='X ' + tag(), item_kind='charge'), 403, 'ROLE_NOT_ALLOWED')
    for key in (main.API_KEY, main.DASHBOARD_API_KEY):
        response = client.post('/catalog/items', json={'name': 'X ' + tag(), 'item_kind': 'charge'}, headers=headers(key))
        assert response.status_code == 403, response.text


@pytest.mark.db
def test_duplicate_name_is_refused(client, actors):
    name = 'Pallet Charge ' + tag()
    assert create(client, actors, name=name, item_kind='charge').status_code == 200
    error(create(client, actors, name=name.lower(), item_kind='charge'), 409, 'ITEM_EXISTS')


@pytest.mark.db
@pytest.mark.parametrize('name,kind,reason', [
    ('Setton Coffee Granola {}', 'product', 'charge_words_marked_product'),
    ('Pallet wrap {}', 'product', 'charge_words_marked_product'),
    ('Shrink wrap {}', 'charge', 'charge_without_words'),
])
def test_disagreeing_name_and_choice_raise_one_check_and_change_nothing(client, actors, db_cursor, name, kind, reason):
    response = create(client, actors, name=name.format(tag()), item_kind=kind, type='finished')
    made = response.json()['item']
    assert made['item_kind'] == kind                       # never changed automatically
    rows = checks(db_cursor, made['id'])
    assert len(rows) == 1 and made['check_exception_id'] == rows[0]['id']
    assert rows[0]['detail']['reason'] == reason and rows[0]['detail']['marked_as'] == kind
    assert rows[0]['owner_actor_id'] == actors['owner']['id'] and rows[0]['severity'] == 'warn'
    # Edits, re-checks and the sweep never open a second one.
    flip = 'charge' if kind == 'product' else 'product'
    for choice in (flip, kind):
        assert client.post(f"/catalog/items/{made['id']}/item-kind", json={'item_kind': choice},
                           headers=headers(actors['office']['key'])).status_code == 200
    charge_setting.check(db_cursor, made['id'])
    charge_setting.sweep(db_cursor)
    assert len(checks(db_cursor, made['id'])) == 1


@pytest.mark.db
@pytest.mark.parametrize('name,kind', [('Delivery Fee {}', 'charge'), ('Classic Granola {}', 'product')])
def test_agreeing_name_and_choice_raise_nothing(client, actors, db_cursor, name, kind):
    made = create(client, actors, name=name.format(tag()), item_kind=kind, type='finished').json()['item']
    assert checks(db_cursor, made['id']) == []


@pytest.mark.db
@pytest.mark.parametrize('spanish', [False, True])
@pytest.mark.parametrize('role,answer', [('office', 'product'), ('owner', 'charge')])
def test_confirming_answers_once_records_the_choice_and_is_never_asked_again(client, actors, db_cursor, role, answer, spanish):
    made = create(client, actors, name='Setton Coffee Granola ' + tag(), item_kind='product', type='finished').json()['item']
    check_id = made['check_exception_id']
    listed = client.get('/exceptions?kind=CHARGE_SETTING_CHECK', headers=headers(actors[role]['key'])).json()['exceptions']
    row = next(r for r in listed if r['id'] == check_id)
    assert row['allowed_actions'] == ['confirmed_product', 'confirmed_charge']
    assert row['confirm_path'] == f"/catalog/items/{made['id']}/confirm-kind"
    assert row['detail']['question' + ('_es' if spanish else '')] == ('Confirmar: ¿Cargo o Producto?' if spanish else 'Confirm: Charge or Product?')
    assert 'contains "fee"' in row['detail']['message'] and 'contiene "fee"' in row['detail']['message_es']
    response = client.post(row['confirm_path'], json={'item_kind': answer}, headers=headers(actors[role]['key']))
    assert response.status_code == 200, response.text
    result = response.json()
    assert result['closed_exception_ids'] == [check_id] and result['changed'] is (answer == 'charge')
    assert result['item']['item_kind'] == answer and result['item']['confirmed'] is True
    saved = item(db_cursor, made['id'])
    assert saved['is_service'] is (answer == 'charge') and saved['charge_setting_confirmed_by_actor_id'] == actors[role]['id']
    db_cursor.execute('SELECT * FROM exceptions WHERE id=%s', (check_id,))
    closed = db_cursor.fetchone()
    assert closed['status'] == 'resolved' and closed['resolution_kind'] == 'confirmed_' + answer
    assert closed['resolved_by_actor_id'] == actors[role]['id'] and closed['detail']['confirmed_as'] == answer
    # Permanent: later edits, renames and the sweep never ask about this item again.
    flip = 'product' if answer == 'charge' else 'charge'
    client.post(f"/catalog/items/{made['id']}/item-kind", json={'item_kind': flip}, headers=headers(actors['office']['key']))
    db_cursor.execute("UPDATE products SET name=name || ' Freight' WHERE id=%s", (made['id'],))
    charge_setting.sweep(db_cursor)
    assert checks(db_cursor, made['id']) == []


@pytest.mark.db
def test_floor_cannot_confirm_and_generic_resolve_points_to_the_item(client, actors, db_cursor):
    made = create(client, actors, name='Freight Granola ' + tag(), item_kind='product', type='finished').json()['item']
    floor = client.get(f"/exceptions/{made['check_exception_id']}", headers=headers(actors['floor']['key'])).json()
    assert floor['allowed_actions'] == []
    error(client.post(f"/catalog/items/{made['id']}/confirm-kind", json={'item_kind': 'product'},
                      headers=headers(actors['floor']['key'])), 403, 'ROLE_NOT_ALLOWED')
    error(client.post(f"/exceptions/{made['check_exception_id']}/resolve", json={'resolution_kind': 'confirmed_product', 'note': 'checked'},
                      headers=headers(actors['owner']['key'])), 409, 'USE_CONFIRM_KIND')
    error(client.post(f"/catalog/items/{made['id']}/confirm-kind", json={}, headers=headers(actors['owner']['key'])),
          422, 'ITEM_KIND_REQUIRED')
    assert len(checks(db_cursor, made['id'])) == 1 and item(db_cursor, made['id'])['is_service'] is False


@pytest.mark.db
def test_catalog_search_shows_setting_words_and_open_check(client, actors):
    marker = tag()
    create(client, actors, name=f'Coffee Crunch {marker}', item_kind='product', type='finished')
    create(client, actors, name=f'Pallet Charge {marker}', item_kind='charge')
    found = client.get(f'/catalog/items?q={marker}', headers=headers(actors['office']['key'])).json()['items']
    by_kind = {row['item_kind']: row for row in found}
    assert by_kind['product']['charge_words'] is True and by_kind['product']['check_exception_id']
    assert by_kind['charge']['check_exception_id'] is None and by_kind['charge']['item_kind_label_es'] == 'Cargo (palé, flete, entrega…)'
    assert client.get(f'/catalog/items?q={marker}', headers=headers(actors['floor']['key'])).status_code == 403


# ── Every other create/edit route ────────────────────────────────

@pytest.mark.db
@pytest.mark.parametrize('path,body', [
    ('/products/quick-create', {'product_type': 'ingredient'}),
    ('/products/quick-create-batch', {'category': 'granola', 'production_context': 'standard'}),
    ('/inventory/found-with-new-product', {'product_type': 'ingredient', 'quantity': 5, 'reason_code': 'physical_count'}),
])
def test_api_create_routes_refuse_a_missing_choice(client, db_cursor, path, body):
    name = 'Quick ' + tag()
    response = client.post(path, json={'product_name': name, **body}, headers=headers())
    error(response, 422, 'ITEM_KIND_REQUIRED')
    db_cursor.execute('SELECT count(*) AS n FROM products WHERE name=%s', (name,))
    assert db_cursor.fetchone()['n'] == 0


@pytest.mark.db
@pytest.mark.parametrize('path,body', [
    ('/products/quick-create-batch', {'category': 'granola', 'production_context': 'standard'}),
    ('/inventory/found-with-new-product', {'product_type': 'ingredient', 'quantity': 5, 'reason_code': 'physical_count'}),
])
def test_stock_creating_routes_refuse_charge(client, path, body):
    error(client.post(path, json={'product_name': 'Quick ' + tag(), 'item_kind': 'charge', **body}, headers=headers()),
          422, 'ITEM_KIND_NOT_ALLOWED')


@pytest.mark.db
@pytest.mark.parametrize('kind,name,checked', [('charge', 'Delivery Fee {}', False), ('product', 'Delivery Day Oats {}', True),
                                              ('product', 'Rolled Oats {}', False), ('charge', 'Shrink wrap {}', True)])
def test_quick_create_saves_the_choice_and_runs_the_check(client, db_cursor, kind, name, checked):
    response = client.post('/products/quick-create', json={'product_name': name.format(tag()), 'product_type': 'ingredient',
                                                          'item_kind': kind}, headers=headers())
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['item_kind'] == kind and item(db_cursor, body['product_id'])['is_service'] is (kind == 'charge')
    assert bool(body['charge_check_exception_id']) is checked
    assert len(checks(db_cursor, body['product_id'])) == int(checked)


@pytest.mark.db
def test_admin_edit_keeps_setting_unless_chosen_and_validates(client, db_cursor):
    db_cursor.execute("INSERT INTO products(name,type,uom,is_service,active) VALUES (%s,'finished','each',false,true) RETURNING id",
                      ('Freight Out ' + tag(),))
    product_id = db_cursor.fetchone()['id']
    assert client.put(f'/admin/products/{product_id}', json={'case_size_lb': 5}, headers=headers()).status_code == 200
    assert item(db_cursor, product_id)['is_service'] is False
    assert len(checks(db_cursor, product_id)) == 1        # the edit ran the safety net, setting untouched
    error(client.put(f'/admin/products/{product_id}', json={'item_kind': 'fee'}, headers=headers()), 422, 'ITEM_KIND_INVALID')
    response = client.put(f'/admin/products/{product_id}', json={'item_kind': 'charge'}, headers=headers())
    assert response.status_code == 200 and response.json()['changes'] == {'item_kind': 'charge'}
    assert item(db_cursor, product_id)['is_service'] is True


@pytest.mark.db
def test_verify_rename_runs_the_check(client, db_cursor):
    db_cursor.execute("INSERT INTO products(name,type,uom,is_service,active,verification_status) VALUES (%s,'ingredient','lb',false,true,'unverified') RETURNING id",
                      ('Oats ' + tag(),))
    product_id = db_cursor.fetchone()['id']
    response = client.post(f'/products/{product_id}/verify', json={'action': 'verify', 'verified_name': 'Oats delivery ' + tag()}, headers=headers())
    assert response.status_code == 200, response.text
    assert len(checks(db_cursor, product_id)) == 1 and item(db_cursor, product_id)['is_service'] is False


# ── Sweep and migration ─────────────────────────────────────────

@pytest.mark.db
def test_sweep_catches_items_added_outside_the_app_once(db_cursor):
    db_cursor.execute("""INSERT INTO products(name,type,uom,active) VALUES (%s,'finished','lb',true),
                         (%s,'finished','each',false) RETURNING id""", ('Imported Freight ' + tag(), 'Inactive Fee ' + tag()))
    active, inactive = [r['id'] for r in db_cursor.fetchall()]
    sweep(db_cursor)
    sweep(db_cursor)
    assert len(checks(db_cursor, active)) == 1 and checks(db_cursor, inactive) == []
    assert item(db_cursor, active)['is_service'] is False


@pytest.mark.db
def test_migration_backfill_rollback_and_reapply(db_cursor):
    db_cursor.execute('SET LOCAL search_path TO public')
    db_cursor.execute(DOWN)
    db_cursor.execute("""INSERT INTO products(name,type,uom,is_service,active) VALUES
        (%s,'batch','lb',false,true), (%s,'packaging','each',true,true), (%s,'finished','each',true,true),
        (%s,'finished','lb',false,true), (%s,'ingredient','lb',false,false) RETURNING id, name""",
        ('Batch Setton Coffee Granola ' + tag(), 'Pallet Charge ' + tag(), 'Shrink wrap ' + tag(),
         'Classic Granola ' + tag(), 'Old Coffee ' + tag()))
    coffee, pallet, wrap, granola, inactive = [r['id'] for r in db_cursor.fetchall()]
    db_cursor.execute(UP)
    db_cursor.execute(UP)                                  # idempotent
    opened = {r['product_id']: r for r in checks(db_cursor)}
    assert {coffee, wrap} <= set(opened) and not {pallet, granola, inactive} & set(opened)
    assert opened[coffee]['detail']['reason'] == 'charge_words_marked_product'
    assert opened[wrap]['detail']['reason'] == 'charge_without_words'
    for product_id in (coffee, pallet, wrap, granola):
        assert item(db_cursor, product_id)['charge_setting_confirmed_at'] is None
    db_cursor.execute('SELECT is_service FROM products WHERE id = ANY(%s) ORDER BY id', ([coffee, pallet, wrap, granola],))
    assert [r['is_service'] for r in db_cursor.fetchall()] == [False, True, True, False]   # nothing changed
    db_cursor.execute(DOWN)                                # only open checks: removable
    db_cursor.execute("SELECT to_regclass('exceptions_one_charge_check_per_product_idx') IS NULL AS gone")
    assert db_cursor.fetchone()['gone']
    db_cursor.execute(UP)
    db_cursor.execute("UPDATE products SET charge_setting_confirmed_at=now() WHERE id=%s", (coffee,))
    db_cursor.execute('SAVEPOINT refuse')
    with pytest.raises(Exception, match='history'):
        db_cursor.execute(DOWN)
    db_cursor.execute('ROLLBACK TO SAVEPOINT refuse')


@pytest.mark.db
def test_exception_kind_is_registered():
    assert exceptions_enforcement.RESOLUTION_KINDS['CHARGE_SETTING_CHECK'] == ('confirmed_product', 'confirmed_charge')
