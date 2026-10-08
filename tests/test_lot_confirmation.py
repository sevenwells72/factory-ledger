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


@pytest.mark.parametrize('entered,expected',[
    ('2026-10-09T12:00:00-04:00','2026-10-20T23:59:00-04:00'),
    ('2026-10-10T12:00:00-04:00','2026-10-20T23:59:00-04:00'),
    ('2026-10-12T12:00:00-04:00','2026-10-21T23:59:00-04:00'),
    ('2026-10-30T12:00:00-04:00','2026-11-10T23:59:00-05:00'),
    ('2026-03-06T12:00:00-05:00','2026-03-17T23:59:00-04:00'),
    ('2026-10-10T01:00:00+00:00','2026-10-20T23:59:00-04:00'),
])
def test_seven_business_day_deadline_across_weekends_and_dst(entered,expected):
    from datetime import datetime
    from lot_confirmation import identification_deadline
    assert identification_deadline(datetime.fromisoformat(entered)).isoformat()==expected


@pytest.mark.parametrize('code',[None,'','   ','N/A','unknown'])
def test_unidentified_receive_exception_uses_entry_not_happened(client,db_cursor,actors,code):
    from tests.test_write_tickets import seed
    from lot_confirmation import identification_deadline
    payload=seed(db_cursor)|{'supplier_lot_code':code,'occurred_at':(main.get_plant_now()-timedelta(days=3)).isoformat()}
    d=prepare(client,'receive',payload,actors['floor']['key'])
    assert d['can_commit'],d
    assert d['draft']['identity_status']=='unidentified'
    assert d['draft']['identity_notice'].startswith('UNIDENTIFIED')
    r=commit(client,d,actors['floor']['key']); assert r.status_code==200,r.text
    result=r.json()
    db_cursor.execute('SELECT * FROM exceptions WHERE ticket_id=%s',(d['ticket_id'],))
    exc=db_cursor.fetchone()
    db_cursor.execute('SELECT created_at FROM transactions WHERE id=%s',(result['transaction_id'],))
    entered=db_cursor.fetchone()['created_at']
    assert exc['kind']=='UNIDENTIFIED_LOT' and exc['owner_actor_id']==actors['floor']['id']
    assert exc['opened_at']==entered and exc['due_at']==identification_deadline(entered)
    assert result['identify_by']==identification_deadline(entered).date().isoformat()
    db_cursor.execute('SELECT identity_status,identify_by FROM lots WHERE id=%s',(result['lot_id'],))
    assert dict(db_cursor.fetchone())=={'identity_status':'unidentified','identify_by':identification_deadline(entered).date()}
    assert commit(client,d,actors['floor']['key']).json()==result|{'replayed':True}
    db_cursor.execute('SELECT count(*) AS n FROM exceptions WHERE ticket_id=%s',(d['ticket_id'],))
    assert db_cursor.fetchone()['n']==1


def test_identified_receive_no_exception(client,db_cursor):
    from tests.test_write_tickets import seed
    d=prepare(client,'receive',seed(db_cursor))
    r=commit(client,d); assert r.status_code==200,r.text
    assert r.json()['identity_status']=='identified' and r.json()['exception_id'] is None
    db_cursor.execute('SELECT count(*) AS n FROM exceptions WHERE ticket_id=%s',(d['ticket_id'],))
    assert db_cursor.fetchone()['n']==0


def test_unidentified_topup_preserves_original_deadline(client,db_cursor):
    from tests.test_write_tickets import seed
    payload=seed(db_cursor)|{'supplier_lot_code':'N/A'}
    d=prepare(client,'receive',payload); first=commit(client,d).json()
    second=prepare(client,'receive',payload|{'cases':1})
    r=commit(client,second,acknowledged_warnings=['POSSIBLE_DUPLICATE']);assert r.status_code==200,r.text
    assert r.json()['exception_id']==first['exception_id'] and r.json()['identification_due_at']==first['identification_due_at']
    correction=prepare(client,'receive',payload|{'supplier_lot_code':'REAL-CODE'})
    assert correction['blockers'][0]['code']=='LOT_IDENTITY_REQUIRES_CORRECTION'


def test_found_unidentified_and_exception_failure_rolls_back(client,db_cursor,items,monkeypatch):
    import lot_confirmation as a5
    d=prepare(client,'inventory/found',body('found',items))
    assert d['draft']['identity_status']=='unidentified'
    original=a5.record_identity
    def crash(*args):
        original(*args)
        raise RuntimeError('exception failure')
    monkeypatch.setattr(a5,'record_identity',crash)
    with pytest.raises(RuntimeError,match='exception failure'): commit(client,d)
    assert posted_count(db_cursor,d)==0
    db_cursor.execute('SELECT count(*) AS n FROM exceptions WHERE ticket_id=%s',(d['ticket_id'],))
    assert db_cursor.fetchone()['n']==0


@pytest.mark.parametrize('supplier_name',[None,'FOUND','initial inventory','PHYSICAL COUNT','UNKNOWN','found inventory','Inventory Intake'])
def test_receipt_without_real_supplier_blocked(client,db_cursor,supplier_name):
    from tests.test_write_tickets import seed
    payload=seed(db_cursor)
    if supplier_name is None: payload['supplier_id']=None
    else:
        db_cursor.execute('INSERT INTO suppliers(name,active) VALUES (%s,true) RETURNING id',(supplier_name,))
        payload['supplier_id']=db_cursor.fetchone()['id']
    d=prepare(client,'receive',payload)
    assert not d['can_commit'] and d['blockers'][0]['code']=='SUPPLIER_REQUIRED'
    assert commit(client,d).status_code==409
    assert posted_count(db_cursor,d)==0
    if supplier_name:
        r=client.post('/resolve',json={'kind':'supplier','query':supplier_name},headers=headers())
        assert r.status_code==200,r.text
        assert payload['supplier_id'] not in [c['id'] for c in r.json().get('candidates',[])]


def test_supplier_id_is_saved_at_insert_never_inferred_from_prefix(client,db_cursor):
    from tests.test_write_tickets import seed
    payload=seed(db_cursor)|{'lot_code':'26-10-08-DUTC-001'}
    db_cursor.execute('SELECT name FROM suppliers WHERE id=%s',(payload['supplier_id'],))
    name=db_cursor.fetchone()['name']
    resolved=client.post('/resolve',json={'kind':'supplier','query':name},headers=headers()).json()
    assert resolved['outcome']=='match'
    assert resolved['match']['id']==payload['supplier_id']
    d=prepare(client,'receive',payload)
    assert d['draft']['supplier_id']==payload['supplier_id']
    result=commit(client,d); assert result.status_code==200,result.text
    result=result.json()
    db_cursor.execute('SELECT supplier_id FROM transactions WHERE id=%s',(result['transaction_id'],))
    assert db_cursor.fetchone()['supplier_id']==payload['supplier_id']
    db_cursor.execute('SELECT supplier_id FROM lots WHERE id=%s',(result['lot_id'],))
    assert db_cursor.fetchone()['supplier_id']==payload['supplier_id']
    detail=client.get('/receipts/'+result['receipt_number'],headers=headers()).json()
    assert detail['lots'][0]['supplier_id']==payload['supplier_id']
    assert detail['transactions'][0]['supplier_name']==name
    # The committed lot supplier is immutable, even when no stock is changed.
    db_cursor.execute('SAVEPOINT supplier_immutable')
    with pytest.raises(psycopg2.IntegrityError):
        db_cursor.execute('UPDATE lots SET supplier_id=NULL WHERE id=%s',(result['lot_id'],))
    db_cursor.execute('ROLLBACK TO SAVEPOINT supplier_immutable')


def test_supplier_change_or_deactivation_between_prepare_and_commit(client,db_cursor):
    from tests.test_write_tickets import seed
    payload=seed(db_cursor)
    d=prepare(client,'receive',payload)
    db_cursor.execute('UPDATE suppliers SET active=false WHERE id=%s',(payload['supplier_id'],))
    r=commit(client,d)
    assert r.status_code==409 and r.json()['detail']['blockers'][0]['code']=='SUPPLIER_REQUIRED'
    assert posted_count(db_cursor,d)==0


def test_same_prefix_suppliers_get_unique_labels_and_cannot_share_lot(client,db_cursor):
    from tests.test_write_tickets import seed
    payload=seed(db_cursor)
    db_cursor.execute("INSERT INTO suppliers(name) VALUES ('Dutch Valley A5'),('Dutch Gold A5') RETURNING id,short_code")
    first,second=db_cursor.fetchall()
    assert first['short_code']!=second['short_code']
    payload|={'supplier_id':first['id'],'lot_code':None}
    d=prepare(client,'receive',payload)
    assert '-'+first['short_code']+'-' in d['draft']['lot_code']
    result=commit(client,d);assert result.status_code==200,result.text
    wrong=prepare(client,'receive',payload|{'supplier_id':second['id'],'lot_code':result.json()['lot_code']})
    r=commit(client,wrong,acknowledged_warnings=['POSSIBLE_DUPLICATE'])
    assert r.status_code==409 and r.json()['detail']['blockers'][0]['code']=='LOT_SUPPLIER_MISMATCH'


def test_expected_receipt_carries_real_supplier_identity(client,db_cursor):
    from tests.test_write_tickets import seed
    payload=seed(db_cursor)
    db_cursor.execute("INSERT INTO expected_receipts(product_id,supplier_id,expected_qty,status) VALUES (%s,%s,50,'open') RETURNING id",(payload['product_id'],payload['supplier_id']))
    er=db_cursor.fetchone()['id']
    d=prepare(client,'receive',payload);r=commit(client,d)
    assert r.status_code==200,r.text
    db_cursor.execute('SELECT expected_receipt_id,supplier_id FROM transactions WHERE id=%s',(r.json()['transaction_id'],))
    assert dict(db_cursor.fetchone())=={'expected_receipt_id':er,'supplier_id':payload['supplier_id']}


def test_commingled_entry_supplier_ids_and_missing_identity(client,db_cursor):
    from tests.test_write_tickets import seed
    payload=seed(db_cursor)
    payload|={'supplier_lot_entries':[{'supplier_id':payload['supplier_id'],'supplier_lot_code':'REAL-1','quantity_lb':25},
        {'supplier_lot_code':'UNKNOWN','quantity_lb':25}]}
    d=prepare(client,'receive',payload)
    assert d['draft']['identity_status']=='unidentified'
    r=commit(client,d);assert r.status_code==200,r.text
    db_cursor.execute('SELECT supplier_id FROM lot_supplier_codes WHERE lot_id=%s',(r.json()['lot_id'],))
    assert [v['supplier_id'] for v in db_cursor.fetchall()]==[payload['supplier_id']]*2
