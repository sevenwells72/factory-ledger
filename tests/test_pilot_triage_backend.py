"""October 14 pilot failures: real routes, transactions, and query budgets."""
from datetime import date, datetime, timezone
from pathlib import Path
from time import perf_counter

import pytest
import main
import permissions
import shift_summary
from tests.test_actor_attribution import client, actors
from tests.test_write_tickets import headers, commit
from tests.test_write_tickets_part2 import items, body, prepare
from tests.test_ship_tickets import seed_shipping, prepared as ship_prepare, upload, storage
from tests.test_order_tickets import catalog, create_order
from tests.test_shift_summary import receive, summary

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[1]


def counts(cur):
    cur.execute('SELECT (SELECT count(*) FROM transactions) AS tx, (SELECT count(*) FROM lots) AS lots')
    return dict(cur.fetchone())


@pytest.mark.parametrize('mode', ['empty', 'zero', 'excluded', 'manual'])
def test_make_prepare_refuses_no_deductible_recipe(client, db_cursor, actors, items, mode):
    pid = items['batch']['id']
    if mode == 'empty':
        db_cursor.execute('DELETE FROM batch_formulas WHERE product_id=%s', (pid,))
    elif mode == 'zero':
        db_cursor.execute('UPDATE batch_formulas SET quantity_lb=0 WHERE product_id=%s', (pid,))
    elif mode == 'excluded':
        db_cursor.execute('UPDATE batch_formulas SET exclude_from_inventory=true WHERE product_id=%s', (pid,))
    payload = body('make', items)
    if mode == 'manual': payload['excluded_ingredients'] = [items['ingredient']['id']]
    if mode == 'empty': payload['substitutions'] = [{'ingredient_product_id': items['ingredient']['id'], 'substitute_product_id': items['finished']['id'], 'lot_id': items['batch']['lot_id'], 'reason_code': 'ingredient_substitution'}]
    before = counts(db_cursor)
    r = client.post('/make/prepare', json=payload, headers=headers(actors['floor']['key']))
    assert r.status_code == 200, r.text
    assert not r.json()['can_commit']
    assert r.json()['blockers'][0]['code'] == 'RECIPE_MISSING'
    assert 'recipe' in r.json()['blockers'][0]['message'].lower()
    assert counts(db_cursor) == before


@pytest.mark.parametrize('path,mode', [('/make','preview'),('/make','commit'),('/make/preview','preview'),('/make/commit','commit')])
def test_legacy_make_missing_recipe_posts_nothing(client, db_cursor, items, path, mode):
    db_cursor.execute('DELETE FROM batch_formulas WHERE product_id=%s', (items['batch']['id'],))
    before = counts(db_cursor)
    r = client.post(path, json={'product_name': items['batch']['name'], 'batches': 1, 'mode': mode, 'confirmed_sku': True}, headers=headers())
    assert r.status_code == 409, r.text
    assert r.json()['detail']['error_code'] == 'RECIPE_MISSING'
    assert counts(db_cursor) == before


def test_recipe_removed_after_prepare_cannot_commit(client, db_cursor, items, actors):
    key = actors['floor']['key']
    p = prepare(client, 'make', body('make', items), key)
    assert p['can_commit']
    db_cursor.execute('DELETE FROM batch_formulas WHERE product_id=%s', (items['batch']['id'],))
    before = counts(db_cursor)
    r = commit(client, p, key)
    assert r.status_code == 409, r.text
    assert 'RECIPE_MISSING' in r.text
    assert counts(db_cursor) == before


@pytest.mark.parametrize('path,mode', [('/pack/prepare',None),('/pack','preview'),('/pack','commit'),('/pack/preview','preview'),('/pack/commit','commit')])
def test_pack_missing_intermediate_recipe_refused(client, db_cursor, items, actors, path, mode):
    db_cursor.execute("INSERT INTO products(name,type,default_batch_lb,is_service) VALUES ('Missing pilot intermediate','batch',10,false) RETURNING id")
    intermediate = db_cursor.fetchone()['id']
    db_cursor.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s', (intermediate, items['finished']['id']))
    payload = body('pack', items) if mode is None else {'source_product':items['batch']['name'],'target_product':items['finished']['name'],'cases':2,'mode':mode}
    before = counts(db_cursor)
    r = client.post(path, json=payload, headers=headers(actors['floor']['key'] if mode is None else None))
    if mode is None:
        assert r.status_code == 200, r.text
        assert not r.json()['can_commit']
        assert r.json()['blockers'][0]['code'] == 'RECIPE_MISSING'
    else:
        assert r.status_code == 409, r.text
        assert r.json()['detail']['error_code'] == 'RECIPE_MISSING'
    assert counts(db_cursor) == before


def test_order_detail_header_id_can_receive_photo(client, db_cursor, actors):
    key = actors['office']['key']
    db_cursor.execute("SELECT nextval('shipments_id_seq') FROM generate_series(1,7)")
    p = ship_prepare(client, seed_shipping(db_cursor), key)
    r = commit(client, p, key)
    assert r.status_code == 200, r.text
    result = r.json()
    order = client.get(f'/sales/orders/{p["draft"]["order_id"]}', headers=headers(key)).json()
    rows = order['shipments']
    assert rows and all(s['shipment_id'] == result['shipment_id'] for s in rows)
    assert all(s['legacy_shipment_row_id'] != s['shipment_id'] for s in rows)
    photo = upload(client, key)
    linked = client.post(f'/shipments/{rows[0]["shipment_id"]}/attachments', json={'attachment_id':photo['id']}, headers=headers(key))
    assert linked.status_code == 200, linked.text


def test_held_correction_never_reports_success(client, db_cursor, items, actors):
    p = prepare(client, 'adjust', body('adjust', items) | {'delta_lb':600}, actors['floor']['key'])
    r = commit(client, p, actors['floor']['key'])
    assert r.status_code == 202, r.text
    data = r.json()
    assert data['success'] is False and data['posted'] is False
    assert data['held'] is True and data['status'] == 'awaiting_approval'


def test_owner_exception_refusal_does_not_claim_held_correction():
    with pytest.raises(Exception) as e:
        permissions.require('approve_exception', {'id':12,'name':'Arturo','role':'floor','key_kind':'actor'})
    assert 'held correction' not in e.value.detail['message']
    assert 'owner' in e.value.detail['message'].lower()


def test_order_default_date_uses_plant_clock(client, db_cursor, actors, catalog, monkeypatch):
    # 00:30 UTC is still October 13 in New York; DB session deliberately remains UTC.
    clock = datetime(2026,10,14,0,30,tzinfo=timezone.utc)
    monkeypatch.setattr(main, 'get_plant_now', lambda: clock.astimezone(main.PLANT_TIMEZONE))
    monkeypatch.setattr(main, '_factory_today', lambda: clock.astimezone(main.PLANT_TIMEZONE).date())
    db_cursor.execute("SET LOCAL TIME ZONE 'UTC'")
    result = create_order(client, catalog, actors['office']['key'])
    db_cursor.execute('SELECT order_date FROM sales_orders WHERE id=%s',(result['order_id'],))
    assert db_cursor.fetchone()['order_date'] == date(2026,10,13)


def test_order_number_plant_day_independent_of_database_timezone(db_cursor):
    # Transaction now() stays fixed. Pick a DB timezone whose calendar day differs.
    db_cursor.execute("SELECT (now() AT TIME ZONE 'America/New_York')::date AS day, now() AS instant")
    today = db_cursor.fetchone()['day']
    for zone in ['Pacific/Kiritimati','Etc/GMT+12']:
        db_cursor.execute('SET LOCAL TIME ZONE %s',(zone,))
        db_cursor.execute('SELECT CURRENT_DATE AS day')
        if db_cursor.fetchone()['day'] != today: break
    else: pytest.fail('No opposite database day found')
    migration = ROOT/'migrations/082_order_plant_date.sql'
    if migration.exists(): db_cursor.execute(migration.read_text())
    db_cursor.execute("INSERT INTO customers(name) VALUES ('pilot date') RETURNING id")
    cid = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO sales_orders(customer_id,order_number) VALUES (%s,'') RETURNING order_number",(cid,))
    assert db_cursor.fetchone()['order_number'].startswith(f'SO-{today:%y%m%d}-')


class CountCursor:
    def __init__(self, cur): self.cur, self.calls = cur, 0
    def execute(self, *args, **kwargs): self.calls += 1; return self.cur.execute(*args, **kwargs)
    def __getattr__(self, name): return getattr(self.cur, name)


def test_thirty_receipt_summary_has_bounded_queries(client, db_cursor, actors):
    key = actors['floor']['key']
    for _ in range(30): receive(client, db_cursor, key)
    cur = CountCursor(db_cursor)
    actor = {k:actors['floor'][k] for k in ['id','name','role']}
    started = perf_counter()
    doc = shift_summary.build(main,cur,day=main.get_plant_now().date(),actor=actor,caller=actor|{'key_kind':'actor'})
    elapsed = perf_counter()-started
    assert doc['receipts_today'] == 30
    assert cur.calls <= 20, f'{cur.calls} queries for 30 receipts'
    assert elapsed < 3, f'{elapsed:.3f}s for 30 receipts'


@pytest.mark.parametrize('action',['receive','make','pack','adjust','found','ship_order'])
def test_batch_summary_rows_match_single_receipt_reader(client,db_cursor,actors,items,action):
    import shift_receipts
    key=actors['office' if action=='ship_order' else 'floor']['key']
    if action=='receive':
        posted,_,_=receive(client,db_cursor,key)
    else:
        p=ship_prepare(client,seed_shipping(db_cursor),key) if action=='ship_order' else prepare(client,action,body(action,items),key)
        r=commit(client,p,key,acknowledged_warnings=[w['code'] for w in p['warnings'] if w.get('requires_ack')])
        assert r.status_code==200,r.text
        posted=r.json()
    number=posted['receipt_number'];day=main.get_plant_now().date()
    old_catalog=shift_summary.Catalog(db_cursor)
    expected=shift_summary.receipt_row(main,db_cursor,old_catalog,number,day)
    new_catalog=shift_summary.Catalog(db_cursor)
    counted=CountCursor(db_cursor)
    loaded=shift_receipts.load(main,counted,[number],new_catalog)
    assert counted.calls==1
    receipt,ticket=loaded[number]
    actual=shift_summary.receipt_row(main,db_cursor,new_catalog,number,day,receipt=receipt,ticket=ticket)
    expected.pop('_receipt');actual.pop('_receipt')
    assert actual==expected
