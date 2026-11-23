"""Standalone shared-key shutdown and named-owner maintenance HTTP proofs."""
import pytest
import main
from tests.test_actor_attribution import client, actors
from tests.test_write_tickets import headers, error

def test_shared_key_default_and_owner_scope(client,db_cursor,actors,monkeypatch):
    error(client.post('/adjust',json={},headers=headers(main.API_KEY)),403,'SHARED_KEY_READ_ONLY')
    assert client.get('/health',headers=headers(main.API_KEY)).status_code==200
    assert client.post('/resolve',json={'kind':'product','query':'none'},headers=headers(main.API_KEY)).status_code==200
    monkeypatch.setenv('SHARED_KEY_WRITES','allow')
    assert client.post('/adjust',json={},headers=headers(main.API_KEY)).status_code==422

def test_named_owner_reaches_maintenance_but_floor_does_not(client,actors):
    assert client.get('/admin/lots/duplicates',headers=headers(actors['owner']['key'])).status_code==200
    assert client.get('/admin/lots/duplicates',headers=headers(actors['floor']['key'])).status_code==403

def test_every_authenticated_write_refuses_shared_key_without_rows(client,db_cursor,actors,monkeypatch):
    monkeypatch.setenv("KEY_SESSIONS_ENABLED", "1")
    monkeypatch.setenv("PIN_SIGNIN_ENABLED", "1")
    monkeypatch.setenv("PERSON_SELECT_ENABLED", "1")
    from fastapi.routing import APIRoute
    import re
    tables=['transactions','transaction_lines','write_tickets','lots','exceptions','actor_write_audit','customers','sales_orders']
    def snapshot():
        result=[]
        for table in tables:
            db_cursor.execute('SELECT count(*) AS n FROM '+table);result.append(db_cursor.fetchone()['n'])
        return result
    before=snapshot();checked=[]
    def key_dependency(dep):
        return getattr(dep.call,'__name__','') in ('verify_api_key','verify_api_key_flexible') or any(key_dependency(c) for c in dep.dependencies)
    for route in main.app.routes:
        if not isinstance(route,APIRoute) or not key_dependency(route.dependant):continue
        path=re.sub(r'\{[^}]+\}','1',route.path)
        for method in route.methods-{'GET','HEAD','OPTIONS'}:
            if (method,route.path) in {('POST','/resolve'),('POST','/products/resolve')}:continue
            response=client.request(method,path,json={},headers=headers(main.API_KEY))
            assert response.status_code==403,(method,path,response.text)
            assert response.json()['detail']['error_code']=='SHARED_KEY_READ_ONLY',(method,path,response.text)
            checked.append((method,route.path))
    assert len(checked)>80
    assert snapshot()==before

def test_disabled_session_route_keeps_404_before_auth(client,monkeypatch):
    monkeypatch.setenv('KEY_SESSIONS_ENABLED','0')
    assert client.post('/auth/session/logout',headers=headers(main.API_KEY)).status_code==404


# ── Review finding 1 (PR #234): the named owner is stored on maintenance writes ──
# The owner key now reaches these routes, so each must record the person, not the
# 'legacy-shared-key' placeholder. The emergency switch keeps the old stamp.
from uuid import uuid4

def _who(identity, actors, monkeypatch):
    if identity == 'owner':
        return actors['owner']['key'], (actors['owner']['name'], actors['owner']['id'])
    monkeypatch.setenv('SHARED_KEY_WRITES', 'allow')
    return main.API_KEY, ('legacy-shared-key', None)

def _product(cur):
    token = uuid4().hex[:10].upper()
    cur.execute("INSERT INTO products(name,odoo_code,type,uom) VALUES (%s,%s,'ingredient','lb') RETURNING id",
                ('Owner stamp ' + token, 'OS-' + token))
    return cur.fetchone()['id']

def _lot_with_line(cur, product_id):
    cur.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id',
                (product_id, 'OS-' + uuid4().hex[:10].upper()))
    lot_id = cur.fetchone()['id']
    cur.execute("INSERT INTO transactions(type,timestamp,notes) VALUES ('adjust',now(),'owner stamp seed') RETURNING id")
    cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,10) RETURNING id',
                (cur.fetchone()['id'], product_id, lot_id))
    return lot_id, cur.fetchone()['id']

def _line_correction_stamp(cur, line_id):
    cur.execute("""SELECT operator_id, entered_by_actor_id FROM ledger_corrections
                   WHERE target_table='transaction_lines' AND target_id=%s""", (line_id,))
    rows = cur.fetchall()
    assert len(rows) == 1, rows
    return rows[0]['operator_id'], rows[0]['entered_by_actor_id']

def _lot_transaction_stamp(cur, lot_id):
    cur.execute("""SELECT t.operator_id, t.entered_by_actor_id FROM transactions t
                   JOIN transaction_lines l ON l.transaction_id=t.id WHERE l.lot_id=%s""", (lot_id,))
    rows = cur.fetchall()
    assert len(rows) == 1, rows
    return rows[0]['operator_id'], rows[0]['entered_by_actor_id']

IDENTITIES = pytest.mark.parametrize('identity', ['owner', 'shared_allow'])

@IDENTITIES
def test_lot_reassign_stores_the_person(client, db_cursor, actors, monkeypatch, identity):
    key, expected = _who(identity, actors, monkeypatch)
    lot_id, line_id = _lot_with_line(db_cursor, _product(db_cursor))
    response = client.post(f'/lots/{lot_id}/reassign', headers=headers(key),
                           json={'to_product_id': _product(db_cursor), 'reason_code': 'data_entry_error'})
    assert response.status_code == 200, response.text
    assert _line_correction_stamp(db_cursor, line_id) == expected

@IDENTITIES
def test_lot_merge_stores_the_person(client, db_cursor, actors, monkeypatch, identity):
    key, expected = _who(identity, actors, monkeypatch)
    product_id = _product(db_cursor)
    source, line_id = _lot_with_line(db_cursor, product_id)
    target, _ = _lot_with_line(db_cursor, product_id)
    response = client.post('/admin/lots/merge', headers=headers(key),
                           json={'source_lot_id': source, 'target_lot_id': target, 'reason': 'Same bag, two labels'})
    assert response.status_code == 200, response.text
    assert _line_correction_stamp(db_cursor, line_id) == expected

@IDENTITIES
def test_certification_and_its_correction_store_the_person(client, db_cursor, actors, monkeypatch, identity):
    key, expected = _who(identity, actors, monkeypatch)
    created = client.post('/records/certifications', headers=headers(key),
                          json={'business_date': '2097-03-04', 'certified_at': '2097-03-04T20:00:00-05:00'})
    assert created.status_code == 200, created.text
    corrected = client.post(f"/records/certifications/{created.json()['certification_id']}/corrections",
                            headers=headers(key),
                            json={'certified_at': '2097-03-04T20:15:00-05:00', 'reason': 'wrong message time'})
    assert corrected.status_code == 200, corrected.text
    db_cursor.execute("SELECT operator_id FROM certifications WHERE business_date='2097-03-04' ORDER BY created_at")
    assert [row['operator_id'] for row in db_cursor.fetchall()] == [expected[0], expected[0]]

@IDENTITIES
def test_direct_found_stores_the_person(client, db_cursor, actors, monkeypatch, identity):
    key, expected = _who(identity, actors, monkeypatch)
    response = client.post('/inventory/found', headers=headers(key),
                           json={'product_id': _product(db_cursor), 'quantity': 5.0, 'reason_code': 'found_count'})
    assert response.status_code == 200, response.text
    assert _lot_transaction_stamp(db_cursor, response.json()['lot_id']) == expected

@IDENTITIES
def test_found_with_new_product_stores_the_person(client, db_cursor, actors, monkeypatch, identity):
    key, expected = _who(identity, actors, monkeypatch)
    response = client.post('/inventory/found-with-new-product', headers=headers(key),
                           json={'product_name': 'Owner stamp new ' + uuid4().hex[:10].upper(),
                                 'product_type': 'ingredient', 'quantity': 5.0, 'reason_code': 'found_count'})
    assert response.status_code == 200, response.text
    assert _lot_transaction_stamp(db_cursor, response.json()['lot_id']) == expected
