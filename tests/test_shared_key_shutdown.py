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
