"""A5 rules through real tickets/PostgreSQL; no implicit test confirmations."""
from pathlib import Path
from datetime import timedelta
import pytest
import psycopg2
import main
from tests.test_actor_attribution import client, actors  # noqa
from tests.test_write_tickets_part2 import items, body  # noqa
from tests.test_write_tickets import headers, commit, posted_count, ticket_row, error  # noqa

pytestmark = pytest.mark.db


def prepare(client, action, payload, key=None):
    r = client.post('/' + action + '/prepare', json=payload, headers=headers(key))
    assert r.status_code == 200, r.text
    return r.json()


def evidence(item, method='last4', value=None):
    return {'lot_id': item['lot_id'], 'method': method,
            'value': value if value is not None else item['lot_code'][-4:] if method == 'last4' else item['lot_code']}


@pytest.mark.parametrize('action', ['make', 'pack'])
def test_no_preconfirmation_and_late_evidence(client, db_cursor, items, actors, action):
    key = actors['floor']['key']
    draft = prepare(client, action, body(action, items), key)
    assert not draft['can_commit']
    assert draft['blockers'][0]['code'] == 'LOT_NOT_CONFIRMED'
    assert all(i['confirmed'] is False and i['suggested_lot']['confirmed'] is False for i in draft['draft']['input_plan'])
    error(commit(client, draft, key), 422, 'LOT_NOT_CONFIRMED')
    assert posted_count(db_cursor, draft) == 0
    assert ticket_row(db_cursor, draft)['status'] == 'prepared'
    confirmations = [evidence(i) for i in draft['draft']['input_plan']]
    result = commit(client, draft, key, lot_confirmations=confirmations)
    assert result.status_code == 200, result.text
    result = result.json()
    assert all(i['confirmed'] for i in result['input_plan'])
    db_cursor.execute('SELECT * FROM transaction_lot_confirmations WHERE transaction_id=%s', (result['transaction_id'],))
    rows = db_cursor.fetchall()
    assert len(rows) == len(confirmations)
    assert rows[0]['actor_id'] == actors['floor']['id']
    assert rows[0]['value'] == confirmations[0]['value']
    assert commit(client, draft, key).json() == result | {'replayed': True}


@pytest.mark.parametrize('method,value', [('last4','NOPE'), ('last4',''), ('full_code','bogus'), ('scan','bogus'), ('pallet','bogus')])
def test_mismatched_evidence_cannot_post(client, db_cursor, items, method, value):
    draft = prepare(client, 'make', body('make',items))
    r = commit(client, draft, lot_confirmations=[evidence(items['ingredient'], method, value)])
    assert r.status_code == 422
    assert posted_count(db_cursor, draft) == 0


@pytest.mark.parametrize('method', ['last4', 'full_code', 'scan'])
def test_matching_evidence_at_prepare(client, items, method):
    draft = prepare(client, 'make', body('make',items) | {'lot_confirmations':[evidence(items['ingredient'], method)]})
    assert draft['can_commit'] and draft['draft']['input_plan'][0]['confirmed']
    assert commit(client,draft).status_code == 200


def test_suffix_ambiguity_same_product_active_only(client, db_cursor, items):
    item = items['ingredient']
    db_cursor.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id', (item['id'],'OTHER-'+item['lot_code'][-4:]))
    twin = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO transactions(type) VALUES ('receive') RETURNING id")
    tx = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,20)', (tx,item['id'],twin))
    draft=prepare(client,'make',body('make',items))
    error(commit(client,draft,lot_confirmations=[evidence(item)]),422,'AMBIGUOUS_SUFFIX')
    assert commit(client,draft,lot_confirmations=[evidence(item,'full_code')]).status_code==200


def test_every_fifo_lot_and_pack_addin_needs_evidence(client, db_cursor, items):
    item = items['ingredient']
    db_cursor.execute("INSERT INTO transactions(type) VALUES ('adjust') RETURNING id")
    tx = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,-95)', (tx,item['id'],item['lot_id']))
    db_cursor.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id', (item['id'],'SECOND-UNIQ'))
    lid=db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,100)', (tx,item['id'],lid))
    draft=prepare(client,'make',body('make',items))
    assert len(draft['draft']['input_plan'])==2
    error(commit(client,draft,lot_confirmations=[evidence(item)]),422,'LOT_NOT_CONFIRMED')
    assert commit(client,draft,lot_confirmations=[evidence(i,'full_code') for i in draft['draft']['input_plan']]).status_code==200
    db_cursor.execute("INSERT INTO products(name,type,uom) VALUES ('A5 intermediate','batch','lb') RETURNING id")
    intermediate=db_cursor.fetchone()['id']
    db_cursor.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s',(intermediate,items['finished']['id']))
    for pid,qty in [(items['batch']['id'],10),(item['id'],2)]:
        db_cursor.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,%s)',(intermediate,pid,qty))
    pack=prepare(client,'pack',body('pack',items))
    assert len(pack['draft']['input_plan'])==2
    error(commit(client,pack,lot_confirmations=[evidence(items['batch'],'full_code')]),422,'LOT_NOT_CONFIRMED')


def move(client, item, **changes):
    r=client.post(f'/lots/{item["lot_id"]}/move/prepare',json={
        'to_location':'production','method':'full_code','value':item['lot_code'], **changes},headers=headers())
    assert r.status_code==200,r.text
    draft=r.json()
    result=commit(client,draft)
    assert result.status_code==200,result.text
    return draft,result.json()


def test_pallet_move_receipt_and_replay(client,db_cursor,items):
    draft=prepare(client,'make',body('make',items))
    error(commit(client,draft,lot_confirmations=[evidence(items['ingredient'],'pallet')]),422,'PALLET_MOVE_REQUIRED')
    md,mr=move(client,items['ingredient'])
    assert commit(client,md).json()==mr | {'replayed':True}
    detail=client.get('/receipts/'+mr['receipt_number'],headers=headers()).json()
    assert detail['transactions']==[] and detail['lots'][0]['id']==items['ingredient']['lot_id']
    result=commit(client,draft,lot_confirmations=[evidence(items['ingredient'],'pallet')])
    assert result.status_code==200,result.text
    db_cursor.execute('SELECT move_id FROM transaction_lot_confirmations WHERE transaction_id=%s',(result.json()['transaction_id'],))
    assert db_cursor.fetchone()['move_id']==mr['move_id']


@pytest.mark.parametrize('kind',['old','moved_back'])
def test_pallet_evidence_must_be_recent_and_in_production(client,items,kind):
    if kind=='old':
        move(client,items['ingredient'],occurred_at=(main.get_plant_now()-timedelta(hours=25)).isoformat())
    else:
        move(client,items['ingredient'])
        move(client,items['ingredient'],to_location='storage')
    draft=prepare(client,'make',body('make',items))
    error(commit(client,draft,lot_confirmations=[evidence(items['ingredient'],'pallet')]),422,'PALLET_MOVE_REQUIRED')


def test_confirmations_atomic_with_post(client,db_cursor,items,monkeypatch):
    import lot_confirmation as a5
    draft=prepare(client,'make',body('make',items))
    original=a5.record_confirmations
    def crash(*args,**kwargs):
        original(*args,**kwargs)
        raise RuntimeError('evidence failure')
    monkeypatch.setattr(a5,'record_confirmations',crash)
    with pytest.raises(RuntimeError,match='evidence failure'):
        commit(client,draft,lot_confirmations=[evidence(items['ingredient'])])
    assert posted_count(db_cursor,draft)==0
    assert ticket_row(db_cursor,draft)['status']=='prepared'


def test_migration_rerun_and_append_only(client,db_cursor,items):
    sql=(Path(__file__).parents[1]/'migrations/062_lot_confirmation.sql').read_text()
    db_cursor.execute(sql); db_cursor.execute(sql)
    draft=prepare(client,'make',body('make',items))
    result=commit(client,draft,lot_confirmations=[evidence(items['ingredient'])]).json()
    db_cursor.execute('SAVEPOINT a5_immutable')
    with pytest.raises(psycopg2.IntegrityError):
        db_cursor.execute('DELETE FROM transaction_lot_confirmations WHERE transaction_id=%s',(result['transaction_id'],))
    db_cursor.execute('ROLLBACK TO SAVEPOINT a5_immutable')


def sub_payload(items):
    # The existing batch stock is a distinct valid intermediate ingredient.
    return {'ingredient_product_id':items['ingredient']['id'],
            'substitute_product_id':items['batch']['id'],'lot_id':items['batch']['lot_id'],
            'reason_code':'ingredient_unavailable','note':'Operator chose the alternative.'}


@pytest.fixture
def substitute(db_cursor,items):
    db_cursor.execute("INSERT INTO products(name,type,uom) VALUES ('A5 replacement','ingredient','lb') RETURNING id")
    pid=db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO lots(product_id,lot_code) VALUES (%s,'A5-SUBSTITUTE-4312') RETURNING id,lot_code",(pid,))
    row=dict(db_cursor.fetchone()); row['lot_id']=row.pop('id'); row['id']=pid
    db_cursor.execute("INSERT INTO transactions(type) VALUES ('receive') RETURNING id")
    tx=db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,100)',(tx,pid,row['lot_id']))
    return row


def test_substitution_consumes_replacement_and_records_reason(client,db_cursor,items,substitute,actors):
    sub=sub_payload(items) | {'substitute_product_id':substitute['id'],'lot_id':substitute['lot_id']}
    payload=body('make',items) | {'substitutions':[sub],'lot_confirmations':[evidence(substitute)]}
    d=prepare(client,'make',payload,actors['floor']['key'])
    assert d['can_commit'],d
    assert d['draft']['substitutions'][0]['reason_code']=='ingredient_unavailable'
    r=commit(client,d,actors['floor']['key']); assert r.status_code==200,r.text
    tx=r.json()['transaction_id']
    db_cursor.execute('SELECT product_id,lot_id,quantity_lb FROM transaction_lines WHERE transaction_id=%s AND quantity_lb<0',(tx,))
    rows=db_cursor.fetchall()
    assert len(rows)==1 and rows[0]['product_id']==substitute['id'] and rows[0]['lot_id']==substitute['lot_id'] and rows[0]['quantity_lb']==-10
    db_cursor.execute('SELECT * FROM transaction_substitutions WHERE transaction_id=%s',(tx,))
    saved=db_cursor.fetchone()
    assert saved['ingredient_product_id']==items['ingredient']['id'] and saved['reason_code']==sub['reason_code'] and saved['note']==sub['note']
    assert saved['actor_id']==actors['floor']['id']
    db_cursor.execute('SELECT ingredient_product_id FROM ingredient_lot_consumption WHERE transaction_id=%s',(tx,))
    assert db_cursor.fetchone()['ingredient_product_id']==substitute['id']
    detail=client.get('/receipts/'+r.json()['receipt_number'],headers=headers()).json()
    assert detail['response']['substitutions'][0]['lot_id']==substitute['lot_id']


@pytest.mark.parametrize('reason',[None,'','   '])
def test_substitution_requires_nonblank_reason(client,items,reason):
    sub=sub_payload(items)
    if reason is None: sub.pop('reason_code')
    else: sub['reason_code']=reason
    r=client.post('/make/prepare',json=body('make',items)|{'substitutions':[sub]},headers=headers())
    assert r.status_code==422,r.text


def test_manual_exclusion_needs_reason_and_is_recorded(client,db_cursor,items):
    payload=body('make',items)|{'excluded_ingredients':[items['ingredient']['id']]}
    d=prepare(client,'make',payload)
    assert d['blockers'][0]['code']=='SUBSTITUTION_REASON_REQUIRED'
    d=prepare(client,'make',payload|{'reason_code':'trial_without_ingredient','note':'Approved trial'})
    assert d['can_commit'],d
    r=commit(client,d); assert r.status_code==200,r.text
    db_cursor.execute('SELECT substitute_product_id,lot_id,reason_code FROM transaction_substitutions WHERE transaction_id=%s',(r.json()['transaction_id'],))
    assert dict(db_cursor.fetchone())=={'substitute_product_id':None,'lot_id':None,'reason_code':'trial_without_ingredient'}


@pytest.mark.parametrize('change',['wrong_lot','duplicate','self','excluded','inactive'])
def test_invalid_substitutions_block(client,db_cursor,items,substitute,change):
    sub=sub_payload(items)|{'substitute_product_id':substitute['id'],'lot_id':substitute['lot_id']}
    payload=body('make',items)|{'substitutions':[sub]}
    if change=='wrong_lot': sub['lot_id']=items['ingredient']['lot_id']
    if change=='duplicate': payload['substitutions']=[sub,sub]
    if change=='self': sub['substitute_product_id']=sub['ingredient_product_id']
    if change=='excluded': payload|={'excluded_ingredients':[items['ingredient']['id']],'reason_code':'trial'}
    if change=='inactive': db_cursor.execute('UPDATE products SET active=false WHERE id=%s',(substitute['id'],))
    d=prepare(client,'make',payload)
    assert not d['can_commit']
    assert d['blockers'][0]['code']!='LOT_NOT_CONFIRMED'


def test_substitution_reason_rows_rollback_with_ledger(client,db_cursor,items,substitute,monkeypatch):
    import lot_confirmation as a5
    sub=sub_payload(items)|{'substitute_product_id':substitute['id'],'lot_id':substitute['lot_id']}
    d=prepare(client,'make',body('make',items)|{'substitutions':[sub],'lot_confirmations':[evidence(substitute)]})
    original=a5.record_substitutions
    def crash(*args):
        original(*args)
        raise RuntimeError('substitution failure')
    monkeypatch.setattr(a5,'record_substitutions',crash)
    with pytest.raises(RuntimeError,match='substitution failure'): commit(client,d)
    assert posted_count(db_cursor,d)==0
