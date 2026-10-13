"""A12 rule coverage against isolated LOCAL Postgres; never staging/prod."""
from datetime import timedelta
from pathlib import Path
from uuid import uuid4
import psycopg2
import pytest
import main
from tests.test_actor_attribution import client, actors
from tests.test_write_tickets import headers,commit,ticket_row,posted_count,error
from tests.test_write_tickets_part2 import seed,body,prepare

pytestmark = pytest.mark.db
MIGRATION = Path(__file__).parents[1]/'migrations/075_kosher_tiers.sql'

@pytest.fixture
def items(db_cursor):
    items = seed(db_cursor)
    db_cursor.execute("UPDATE products SET kosher_tier='extra_kosher' WHERE id IN (%s,%s)",
        (items['batch']['id'],items['finished']['id']))
    # The pre-existing source lot deliberately stays regular after catalog edit.
    db_cursor.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id,lot_code',
        (items['batch']['id'],'WT2-XK-'+uuid4().hex[:12].upper()))
    row = db_cursor.fetchone()
    items['extra_lot'] = {'lot_id':row['id'],'lot_code':row['lot_code']}
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('receive','A12 fixture') RETURNING id")
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,100)',
        (db_cursor.fetchone()['id'],items['batch']['id'],row['id']))
    return items

def made(client,items,actors,**extra):
    key=actors['floor']['key']
    prepared=prepare(client,'make',body('make',items)|{'confirmed_sku':True}|extra,key)
    assert prepared['can_commit'],prepared
    held=commit(client,prepared,key)
    assert held.status_code==202,held.text
    return prepared,held.json()

def approve(client,held,actors):
    return client.post(f"/exceptions/{held['exception_id']}/approve",json={},headers=headers(actors['owner']['key']))

def test_held_make_posts_nothing_has_receipt_seven_days_and_replays(client,db_cursor,items,actors):
    before=main.lot_on_hand(db_cursor,items['ingredient']['lot_id'])
    prepared,held=made(client,items,actors)
    assert held['posted'] is False and held['kosher_tier']=='extra_kosher'
    assert held['receipt_number'].startswith('MK-')
    assert 'HELD' in held['message'] and 'Michael' in held['message']
    assert posted_count(db_cursor,prepared)==0
    assert main.lot_on_hand(db_cursor,items['ingredient']['lot_id'])==before
    row=ticket_row(db_cursor,prepared)
    assert timedelta(days=6)<row['expires_at']-row['prepared_at']<timedelta(days=8)
    assert row['committed_at'] is None
    assert commit(client,prepared,actors['floor']['key']).json()==held
    db_cursor.execute("SELECT count(*) AS n FROM exceptions WHERE kind='KOSHER_ATTESTATION' AND ticket_id=%s",(prepared['ticket_id'],))
    assert db_cursor.fetchone()['n']==1
    detail=client.get('/receipts/'+held['receipt_number'],headers=headers()).json()
    assert detail['transactions']==[] and detail['status']=='awaiting_approval'
    assert detail['response']==held and detail['kosher_tier']=='extra_kosher'

def test_approve_posts_once_keeps_preparer_and_attests_owner_on_all_evidence(client,db_cursor,items,actors):
    prepared,held=made(client,items,actors)
    approved=approve(client,held,actors)
    assert approved.status_code==200,approved.text
    response=approved.json()
    assert response['receipt_number']==held['receipt_number']
    assert response['kosher_attested_by_actor_id']==actors['owner']['id']
    db_cursor.execute('SELECT * FROM transactions WHERE id=%s',(response['transaction_id'],))
    txn=db_cursor.fetchone()
    db_cursor.execute('SELECT * FROM lots WHERE id=%s',(response['lot_id'],))
    lot=db_cursor.fetchone()
    db_cursor.execute('SELECT * FROM ticket_attestations WHERE ticket_id=%s',(prepared['ticket_id'],))
    att=db_cursor.fetchone()
    assert txn['entered_by_actor_id']==actors['floor']['id']
    assert att['kind']=='kosher_procedure'
    assert txn['kosher_attested_by_actor_id']==lot['kosher_attested_by_actor_id']==att['actor_id']==actors['owner']['id']
    assert txn['kosher_attested_at']==lot['kosher_attested_at']==att['attested_at']
    assert txn['kosher_tier']==lot['kosher_tier']=='extra_kosher'
    db_cursor.execute('SELECT status,resolved_by_actor_id,resolution_kind FROM exceptions WHERE id=%s',(held['exception_id'],))
    assert dict(db_cursor.fetchone())=={'status':'resolved','resolved_by_actor_id':actors['owner']['id'],'resolution_kind':'approved'}
    assert approve(client,held,actors).json()==response|{'replayed':True}
    assert commit(client,prepared,actors['floor']['key']).json()==response|{'replayed':True}
    assert posted_count(db_cursor,prepared)==1
    detail=client.get('/receipts/'+response['receipt_number'],headers=headers()).json()
    assert detail['transactions'][0]['kosher_attested_by_actor_id']==actors['owner']['id']
    assert any(l['kosher_tier']=='extra_kosher' for l in detail['lots'])

def test_owner_self_make_attested_without_hold(client,db_cursor,items,actors):
    prepared=prepare(client,'make',body('make',items)|{'confirmed_sku':True},actors['owner']['key'])
    response=commit(client,prepared,actors['owner']['key'])
    assert response.status_code==200,response.text
    assert response.json()['kosher_attested_by_actor_id']==actors['owner']['id']
    db_cursor.execute("SELECT count(*) AS n FROM exceptions WHERE kind='KOSHER_ATTESTATION' AND ticket_id=%s",(prepared['ticket_id'],))
    assert db_cursor.fetchone()['n']==0

def test_reject_posts_nothing_and_cannot_be_approved(client,db_cursor,items,actors):
    prepared,held=made(client,items,actors)
    path=f"/exceptions/{held['exception_id']}/reject"
    payload={'resolution_kind':'declined','note':'Kosher procedure was not followed'}
    rejected=client.post(path,json=payload,headers=headers(actors['owner']['key']))
    assert rejected.status_code==200,rejected.text
    assert rejected.json()['posted'] is False
    assert posted_count(db_cursor,prepared)==0
    assert ticket_row(db_cursor,prepared)['status']=='rejected'
    assert client.post(path,json=payload,headers=headers(actors['owner']['key'])).json()['replayed']
    error(approve(client,held,actors),409,'EXCEPTION_CLOSED')
    error(commit(client,prepared,actors['floor']['key']),409,'TICKET_NOT_COMMITTABLE')

@pytest.mark.parametrize('role',['floor','office'])
@pytest.mark.parametrize('decision',['approve','reject'])
def test_only_own_signed_in_owner_can_decide(client,db_cursor,items,actors,role,decision):
    prepared,held=made(client,items,actors)
    response=client.post(f"/exceptions/{held['exception_id']}/{decision}",json={} if decision=='approve' else {'resolution_kind':'declined','note':'x'},headers=headers(actors[role]['key']))
    assert response.status_code==403,response.text
    assert posted_count(db_cursor,prepared)==0

def test_generic_resolve_or_photo_cannot_release_kosher_hold(client,db_cursor,items,actors):
    prepared,held=made(client,items,actors)
    error(client.post(f"/exceptions/{held['exception_id']}/resolve",json={'resolution_kind':'waived','note':'x'},headers=headers(actors['owner']['key'])),409,'USE_APPROVE_OR_REJECT')
    assert commit(client,prepared,actors['floor']['key'],attachment_ref='not-a-kosher-attestation').status_code==202
    assert posted_count(db_cursor,prepared)==0

def test_late_lot_confirmation_survives_hold_then_owner_approval(client,items,actors):
    response=client.post('/make/prepare',json=body('make',items)|{'confirmed_sku':True},headers=headers(actors['floor']['key']))
    prepared=response.json()
    confirmations=[{'lot_id':i['lot_id'],'method':'full_code','value':i['lot_code']} for i in prepared['draft']['input_plan']]
    held=commit(client,prepared,actors['floor']['key'],lot_confirmations=confirmations)
    assert held.status_code==202,held.text
    result=approve(client,held.json(),actors)
    assert result.status_code==200,result.text

def test_seven_day_expiry_keeps_make_unposted(client,db_cursor,items,actors):
    prepared,held=made(client,items,actors)
    db_cursor.execute("UPDATE write_tickets SET expires_at=clock_timestamp()-interval '1 second' WHERE id=%s",(prepared['ticket_id'],))
    error(approve(client,held,actors),409,'TICKET_EXPIRED')
    error(commit(client,prepared,actors['floor']['key']),409,'TICKET_EXPIRED')
    assert posted_count(db_cursor,prepared)==0

@pytest.mark.parametrize('change',["UPDATE products SET active=false WHERE id=%s", "UPDATE actors SET active=false WHERE id=%s"])
def test_stale_approval_keeps_hold_unposted(client,db_cursor,items,actors,change):
    prepared,held=made(client,items,actors)
    db_cursor.execute(change,(items['batch']['id'] if 'products' in change else actors['floor']['id'],))
    error(approve(client,held,actors),409,'TICKET_STALE')
    assert ticket_row(db_cursor,prepared)['status']=='awaiting_approval'
    assert posted_count(db_cursor,prepared)==0

def test_shared_key_cannot_attest_or_create_unattributable_hold(client,db_cursor,items):
    prepared=prepare(client,'make',body('make',items)|{'confirmed_sku':True})
    result=commit(client,prepared)
    assert result.status_code in (403,409),result.text
    assert posted_count(db_cursor,prepared)==0

def test_client_cannot_supply_attesting_actor_or_pin(client,items,actors):
    for extra in ({'kosher_attested_by_actor_id':actors['owner']['id']},{'pin':'1234'},{'kosher_attestation':{'actor_id':actors['owner']['id']}}):
        assert client.post('/make/prepare',json=body('make',items)|extra,headers=headers(actors['floor']['key'])).status_code==422

def test_pack_blocks_wrong_source_lot_and_suggests_only_extra(client,db_cursor,items,actors):
    wrong=body('pack',items)|{'lot_allocations':[{'lot_id':items['batch']['lot_id'],'quantity_lb':10}]}
    blocked=prepare(client,'pack',wrong,actors['floor']['key'])
    assert blocked['can_commit'] is False
    assert blocked['blockers'][0]['code']=='KOSHER_SOURCE_REQUIRED'
    error(commit(client,blocked,actors['floor']['key']),409,'TICKET_STALE')
    assert posted_count(db_cursor,blocked)==0
    good=prepare(client,'pack',body('pack',items),actors['floor']['key'])
    assert [i['lot_id'] for i in good['draft']['input_plan']]==[items['extra_lot']['lot_id']]
    assert commit(client,good,actors['floor']['key']).status_code==200

def test_pack_regular_source_product_is_allowed_only_with_extra_lot(client,db_cursor,items,actors):
    db_cursor.execute("UPDATE products SET kosher_tier='regular' WHERE id=%s",(items['batch']['id'],))
    prepared=prepare(client,'pack',body('pack',items),actors['floor']['key'])
    assert prepared['can_commit']
    response=commit(client,prepared,actors['floor']['key'])
    assert response.status_code==200,response.text
    assert response.json()['kosher_tier']=='extra_kosher'

def test_downgrade_records_pack_flag_and_output_lot_event(client,db_cursor,items,actors):
    db_cursor.execute("UPDATE products SET kosher_tier='regular' WHERE id=%s",(items['finished']['id'],))
    payload=body('pack',items)|{'lot_allocations':[{'lot_id':items['extra_lot']['lot_id'],'quantity_lb':10}]}
    prepared=prepare(client,'pack',payload,actors['floor']['key'])
    assert prepared['can_commit'] and prepared['draft']['transaction_kosher_downgrade']
    response=commit(client,prepared,actors['floor']['key'])
    assert response.status_code==200,response.text
    result=response.json()
    assert result['transaction_kosher_downgrade'] and result['kosher_tier']=='regular'
    db_cursor.execute('SELECT transaction_kosher_downgrade FROM transactions WHERE id=%s',(result['transaction_id'],))
    assert db_cursor.fetchone()['transaction_kosher_downgrade']
    db_cursor.execute('SELECT kosher_tier FROM lots WHERE id=%s',(result['output_lot_id'],))
    assert db_cursor.fetchone()['kosher_tier']=='regular'
    db_cursor.execute("SELECT * FROM lot_events WHERE transaction_id=%s AND event='kosher_downgrade'",(result['transaction_id'],))
    event=db_cursor.fetchone()
    assert event['lot_id']==result['output_lot_id'] and event['entered_by_actor_id']==actors['floor']['id']
    assert event['old']['source_lot_ids']==[items['extra_lot']['lot_id']]
    assert event['ticket_id']==prepared['ticket_id']
    assert commit(client,prepared,actors['floor']['key']).json()['replayed']
    db_cursor.execute('SELECT count(*) AS n FROM lot_events WHERE transaction_id=%s',(result['transaction_id'],))
    assert db_cursor.fetchone()['n']==1

def test_regular_pack_is_not_downgrade(client,db_cursor,items,actors):
    db_cursor.execute("UPDATE products SET kosher_tier='regular' WHERE id=%s",(items['finished']['id'],))
    payload=body('pack',items)|{'lot_allocations':[{'lot_id':items['batch']['lot_id'],'quantity_lb':10}]}
    prepared=prepare(client,'pack',payload,actors['floor']['key'])
    result=commit(client,prepared,actors['floor']['key'])
    assert result.status_code==200,result.text
    assert result.json()['transaction_kosher_downgrade'] is False

def test_lot_tier_copied_and_immutable_even_after_product_edit(db_cursor,items):
    db_cursor.execute('SELECT kosher_tier FROM lots WHERE id IN (%s,%s) ORDER BY id',(items['batch']['lot_id'],items['extra_lot']['lot_id']))
    assert [r['kosher_tier'] for r in db_cursor.fetchall()]==['regular','extra_kosher']
    db_cursor.execute('SAVEPOINT immutable_tier')
    with pytest.raises(psycopg2.Error,match='LOT_KOSHER_TIER_IMMUTABLE'):
        db_cursor.execute("UPDATE lots SET kosher_tier='extra_kosher' WHERE id=%s",(items['batch']['lot_id'],))
    db_cursor.execute('ROLLBACK TO SAVEPOINT immutable_tier')

def test_migration_is_idempotent_and_never_changes_formula_or_historical_lots(db_cursor,items):
    db_cursor.execute('SELECT * FROM batch_formulas ORDER BY product_id,ingredient_product_id')
    formulas=db_cursor.fetchall()
    db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute('SELECT * FROM batch_formulas ORDER BY product_id,ingredient_product_id')
    assert db_cursor.fetchall()==formulas
    db_cursor.execute('SELECT kosher_tier FROM lots WHERE id=%s',(items['batch']['lot_id'],))
    assert db_cursor.fetchone()['kosher_tier']=='regular'

@pytest.mark.parametrize('direct',['make','pack'])
def test_legacy_route_cannot_bypass_rules(client,db_cursor,items,direct):
    if direct=='make':
        payload={'product_name':items['batch']['name'],'batches':1,'confirmed_sku':True,'mode':'commit'}
    else:
        payload={'source_product':items['batch']['name'],'target_product':items['finished']['name'],'cases':2,'mode':'commit',
            'lot_allocations':[{'lot_code':items['batch']['lot_code'],'quantity_lb':10}]}
    response=client.post('/'+direct,json=payload,headers=headers())
    assert response.status_code in (403,409),response.text
    assert response.json()['detail']['error_code']==('KOSHER_ATTESTATION_REQUIRED' if direct=='make' else 'KOSHER_SOURCE_REQUIRED')
