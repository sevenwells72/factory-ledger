"""Receipt UI choices reuse live void policy; commit still rechecks it."""
from datetime import timedelta
import pytest
import main
from tests.test_write_tickets import client, actors, payload, prepare, commit, headers

pytestmark=pytest.mark.db

def source(client,payload,actors,role='floor'):
    key=actors[role]['key']
    return commit(client,prepare(client,payload,key),key).json()

def read(client,number,key):
    result=client.get('/receipts/'+number,headers=headers(key))
    assert result.status_code==200,result.text
    return result.json()

@pytest.mark.parametrize('role,expected',[('floor',['void']),('owner',['void']),('office',[]),(None,[])])
def test_named_receipt_choices_match_fl_own_today_policy(client,payload,actors,role,expected):
    original=source(client,payload,actors)
    key=actors[role]['key'] if role else None
    assert read(client,original['receipt_number'],key)['transactions'][0]['allowed_actions']==expected
    other=client.get('/receipts/by-transaction/'+str(original['transaction_id']),headers=headers(key)).json()
    assert other['transactions'][0]['allowed_actions']==expected
    all_=client.get('/receipts',headers=headers(key)).json()['receipts']
    assert next(r for r in all_ if r['receipt_number']==original['receipt_number'])['transactions'][0]['allowed_actions']==expected

@pytest.mark.parametrize('post_role,previous_day',[('owner',False),('floor',True)])
def test_other_person_and_previous_plant_day_hide_void(client,payload,actors,monkeypatch,post_role,previous_day):
    original=source(client,payload,actors,post_role);floor=actors['floor']['key']
    if previous_day:
        tomorrow=main.get_plant_now()+timedelta(days=1)
        monkeypatch.setattr(main,'get_plant_now',lambda:tomorrow)
    assert read(client,original['receipt_number'],floor)['transactions'][0]['allowed_actions']==[]
    assert read(client,original['receipt_number'],actors['owner']['key'])['transactions'][0]['allowed_actions']==['void']

@pytest.mark.parametrize('role',['floor','owner'])
def test_void_fixed_reason_receipt_and_replay_remove_button(client,payload,actors,role):
    original=source(client,payload,actors);key=actors[role]['key']
    prepared=client.post(f"/void/{original['transaction_id']}/prepare",headers=headers(key),
        json={'reason':'data_entry_error','client_source':'dashboard'}).json()
    assert prepared['draft']['reason']=='data_entry_error'
    result=commit(client,prepared,key)
    assert result.status_code==200,result.text
    saved=result.json()
    assert saved['receipt_number'].startswith('VD-')
    assert saved['source_receipt_number']==original['receipt_number']
    assert commit(client,prepared,key).json()==saved|{'replayed':True}
    old=read(client,original['receipt_number'],key)
    assert old['effective_status']=='voided'
    assert old['void_receipt_number']==saved['receipt_number']
    assert old['transactions'][0]['allowed_actions']==[]
    assert read(client,saved['receipt_number'],key)['response']['voided_transaction_id']==original['transaction_id']

def test_role_change_after_draft_cannot_record(client,db_cursor,payload,actors):
    original=source(client,payload,actors);key=actors['floor']['key']
    prepared=client.post(f"/void/{original['transaction_id']}/prepare",headers=headers(key),
        json={'reason':'data_entry_error'}).json()
    db_cursor.execute("UPDATE actors SET role='office' WHERE id=%s",(actors['floor']['id'],))
    assert read(client,original['receipt_number'],key)['transactions'][0]['allowed_actions']==[]
    assert commit(client,prepared,key).status_code==403
    assert read(client,original['receipt_number'],actors['owner']['key'])['effective_status']=='posted'
