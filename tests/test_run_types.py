"""S1 amendment: run-type contract, routing, yields and reversible schema."""
from pathlib import Path
from uuid import uuid4
from decimal import Decimal
import pytest
import psycopg2
import main
from tests.test_production_runs import (
    schema, client, _seed, _product, _create, _cover, _line, _post_output,
    _production_line, _assign, DASH, PLANNED, ROOT,
)

pytestmark = pytest.mark.db


def batch(cur, weight=323):
    pid = _product(cur, uuid4().hex[:8], label="recipe", ptype="batch", case_size_lb=None)
    cur.execute("UPDATE products SET default_batch_lb=%s, yield_multiplier=1.11 WHERE id=%s", (weight, pid))
    return {"product_id": pid}


@pytest.mark.parametrize('kind,weight', [('bake',323),('coconut',360),('bake',384.52)])
def test_pan_yield_snapshot_and_reconversion(client, schema, kind, weight):
    b=batch(schema, weight)
    r=_create(client,b,qty=12,unit='pans',run_type=kind)
    assert r['run_type']==kind and r['planned_qty']==12
    assert r['pan_yield_lb_used']==weight and r['case_size_lb_used'] is None
    assert r['expected_lb']==r['planned_qty_lb']==float(Decimal(str(weight))*12)
    assert r['uncovered_lb']==r['expected_lb']
    schema.execute('UPDATE products SET default_batch_lb=400 WHERE id=%s',(b['product_id'],))
    url=f"/production/runs/{r['id']}"
    out=client.patch(url,json={'notes':'keep original yield'},headers=DASH).json()['run']
    assert out['pan_yield_lb_used']==weight
    out=client.patch(url,json={'planned_qty':13},headers=DASH).json()['run']
    assert out['pan_yield_lb_used']==400 and out['expected_lb']==5200
    out=client.patch(url,json={'planned_unit':'lb'},headers=DASH).json()['run']
    assert out['pan_yield_lb_used'] is None and out['expected_lb']==13
    assert client.patch(url,json={'run_type':'pack'},headers=DASH).status_code==422
    assert client.patch(url,json={'product_id':1},headers=DASH).status_code==422
    rows=client.get('/production/runs?run_type='+kind,headers=DASH).json()['runs']
    assert r['id'] in [x['id'] for x in rows] and all(x['run_type']==kind for x in rows)


@pytest.mark.parametrize('kind,unit,qty,code',[
    ('bake','pans',1.5,'INVALID_QUANTITY'),('coconut','pans',0.99999,'INVALID_QUANTITY'),
    ('bake','cases',1,'INVALID_UNIT'),('coconut','cases',1,'INVALID_UNIT'),
    ('pack','pans',1,'INVALID_UNIT'),('other','pans',1,'INVALID_UNIT'),
])
def test_native_unit_rules(client,schema,kind,unit,qty,code):
    b=batch(schema) if kind in ('bake','coconut') else _seed(schema)
    payload=dict(run_type=kind,product_id=b['product_id'],planned_qty=qty,planned_unit=unit,planned_date=str(PLANNED))
    out=client.post('/production/runs',json=payload,headers=DASH)
    assert out.status_code==422 and out.json()['detail']['error_code']==code
    r=_create(client,b,qty=1,unit='lb',run_type=kind)
    out=client.patch(f"/production/runs/{r['id']}",json={'planned_qty':qty,'planned_unit':unit},headers=DASH)
    assert out.status_code==422 and out.json()['detail']['error_code']==code


def test_missing_type_and_yield(client,schema):
    b=batch(schema,None)
    payload=dict(product_id=b['product_id'],planned_qty=1,planned_unit='pans',planned_date=str(PLANNED))
    assert client.post('/production/runs',json=payload,headers=DASH).status_code==422
    payload['run_type']='bake'
    out=client.post('/production/runs',json=payload,headers=DASH)
    assert out.status_code==400 and out.json()['detail']['error_code']=='PAN_YIELD_REQUIRED'
    _create(client,b,qty=50,unit='lb',run_type='bake')
    payload['run_type']='typo'
    assert client.post('/production/runs',json=payload,headers=DASH).status_code==422
    assert client.get('/production/runs?run_type=typo',headers=DASH).status_code==422


@pytest.mark.parametrize('kind,ptype,active,service,no_production,ok',[
    ('bake','finished',True,False,False,False),('coconut','finished',True,False,False,False),
    ('bake','batch',False,False,False,False),('coconut','batch',True,True,False,False),
    ('other','ingredient',True,False,True,True),('other','finished',True,False,True,True),
    ('other','ingredient',False,False,False,False),('other','finished',True,True,False,False),
])
def test_product_rules(client,schema,kind,ptype,active,service,no_production,ok):
    pid=_product(schema,uuid4().hex[:8],label='typed',ptype=ptype,active=active,is_service=service,no_production=no_production)
    out=client.post('/production/runs',json=dict(product_id=pid,run_type=kind,planned_qty=1,planned_unit='lb',planned_date=str(PLANNED)),headers=DASH)
    assert out.status_code==(201 if ok else 400),out.text
    if not ok: assert kind in out.json()['detail']['message']


@pytest.mark.parametrize('route',['parent','bom'])
def test_batch_routes_multiple_skus_and_mixed_coverage(client,schema,route):
    b=batch(schema); a=_seed(schema,qty=1000)
    second=_product(schema,uuid4().hex[:8],label='second')
    line2=_line(schema,a['order_id'],second,1000)
    for pid in (a['product_id'],second):
        if route=='parent': schema.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s',(b['product_id'],pid))
        else: schema.execute('INSERT INTO product_bom (finished_product_id,component_product_id,quantity,uom) VALUES (%s,%s,1,\'lb\')',(pid,b['product_id']))
    r=_create(client,b,qty=2,unit='pans',run_type='bake')
    assert set(r['coverage_product_ids'])=={a['product_id'],second}
    rows=[{'sales_order_line_id':a['line_id'],'qty_lb':100},{'sales_order_line_id':line2,'qty_lb':200}]
    out=_cover(client,r['id'],rows)
    assert out.status_code==200,out.text
    assert out.json()['run']['uncovered_lb']==346
    bad=_seed(schema)
    out=_cover(client,r['id'],[{'sales_order_line_id':bad['line_id'],'qty_lb':1}])
    assert out.status_code==409 and out.json()['detail']['error_code']=='LINE_PRODUCT_MISMATCH'
    pack=_create(client,a,qty=100,unit='lb')
    out=_cover(client,pack['id'],[rows[0]])
    assert out.status_code==200 and out.json()['run']['coverage'][0]['mixed_bake_pack'] is True
    # Cross-date warning is derived from all active runs, not the current week.
    client.patch(f"/production/runs/{pack['id']}",json={'planned_date':'2026-10-01'},headers=DASH)
    listed=client.get('/production/runs?run_type=bake',headers=DASH).json()['runs']
    assert next(x for x in listed if x['id']==r['id'])['coverage'][0]['mixed_bake_pack'] is True
    client.post(f"/production/runs/{pack['id']}/cancel",json={},headers=DASH)
    listed=client.get('/production/runs?run_type=bake',headers=DASH).json()['runs']
    assert not next(x for x in listed if x['id']==r['id'])['coverage'][0]['mixed_bake_pack']
    # Coverage limits still apply at the batch level.
    assert _cover(client,r['id'],[{'sales_order_line_id':a['line_id'],'qty_lb':647}]).json()['detail']['error_code']=='RUN_OVERCOVERED'
    assert _cover(client,r['id'],[{'sales_order_line_id':a['line_id'],'qty_lb':1001}]).json()['detail']['error_code']=='COVERAGE_EXCEEDS_REMAINING'


@pytest.mark.parametrize('kind,fmt,code',[('bake',None,'granola'),('coconut',None,'coconut'),('pack','bagged','pouch'),('pack','10lb','bulk_pack'),('pack','25lb','bulk_pack'),('pack',None,None),('other','bagged',None)])
def test_line_fallback_and_assignment_priority(client,schema,kind,fmt,code):
    b=batch(schema) if kind in ('bake','coconut') else _seed(schema)
    schema.execute('UPDATE products SET pack_format=%s WHERE id=%s',(fmt,b['product_id']))
    expected=None
    if code:
        schema.execute('INSERT INTO production_lines (name,line_code) VALUES (%s,%s) ON CONFLICT (line_code) DO UPDATE SET line_code=EXCLUDED.line_code RETURNING id',(code,code))
        expected=schema.fetchone()['id']
    assert _create(client,b,qty=1,unit='lb',run_type=kind)['line_id']==expected
    first=_production_line(schema,uuid4().hex[:8],'first');second=_production_line(schema,uuid4().hex[:8],'second')
    _assign(schema,b['product_id'],first)
    assert _create(client,b,qty=1,unit='lb',run_type=kind)['line_id']==first
    _assign(schema,b['product_id'],second)
    assert _create(client,b,qty=1,unit='lb',run_type=kind)['line_id']==expected
    assert _create(client,b,qty=1,unit='lb',run_type=kind,line_id=second)['line_id']==second


@pytest.mark.parametrize('kind,expected',[('bake',323),('coconut',323),('pack',100),('other',423)])
def test_evidence_filters_by_type(client,schema,kind,expected):
    b=batch(schema) if kind in ('bake','coconut') else _seed(schema)
    r=_create(client,b,qty=2,unit='pans' if kind in ('bake','coconut') else 'lb',run_type=kind)
    _post_output(schema,b['product_id'],323,PLANNED,ttype='make')
    _post_output(schema,b['product_id'],100,PLANNED,ttype='pack')
    ev=client.get(f"/production/runs/{r['id']}/evidence",headers=DASH).json()
    assert ev['recorded_lb']==expected
    if kind in ('bake','coconut'): assert ev['recorded_qty']==1


def test_migration_reapply_and_old_insert_default(schema):
    sql=(ROOT/'migrations/054_run_type.sql').read_text()
    schema.execute(sql);schema.execute(sql)
    b=_seed(schema)
    schema.execute("INSERT INTO production_runs(product_id,planned_qty_lb,planned_date) VALUES (%s,1,%s) RETURNING run_type,pan_yield_lb_used",(b['product_id'],PLANNED))
    assert dict(schema.fetchone())=={'run_type':'pack','pan_yield_lb_used':None}
    for column,value in [('run_type','wrong'),('planned_unit','trays'),('pan_yield_lb_used',0)]:
        schema.execute('SAVEPOINT constraint_check')
        with pytest.raises(psycopg2.errors.CheckViolation):
            schema.execute(f'UPDATE production_runs SET {column}=%s',(value,))
        schema.execute('ROLLBACK TO SAVEPOINT constraint_check')


@pytest.mark.parametrize('kind,unit',[('bake','lb'),('pack','pans')])
def test_down_refuses_without_deleting(schema,kind,unit):
    b=_seed(schema)
    schema.execute('INSERT INTO production_runs(product_id,planned_qty_lb,planned_date,run_type,planned_unit) VALUES (%s,1,%s,%s,%s)',(b['product_id'],PLANNED,kind,unit))
    sql=(ROOT/'migrations/down/054_run_type_down.sql').read_text()
    # Execute the exact guard inside a savepoint; the manual transaction wrapper is tested separately.
    guard=sql[sql.index('DO $$'):sql.index('ALTER TABLE')]
    schema.execute('SAVEPOINT down_guard')
    with pytest.raises(psycopg2.errors.RaiseException,match='rollback refused'): schema.execute(guard)
    schema.execute('ROLLBACK TO SAVEPOINT down_guard')
    schema.execute('SELECT count(*) AS n FROM production_runs WHERE run_type=%s AND planned_unit=%s',(kind,unit))
    assert schema.fetchone()['n']>=1


def test_down_preserves_pack_coverage(schema,client):
    b=_seed(schema);r=_create(client,b,qty=10,unit='lb')
    _cover(client,r['id'],[{'sales_order_line_id':b['line_id'],'qty_lb':5}])
    sql=(ROOT/'migrations/down/054_run_type_down.sql').read_text().replace('BEGIN;','',1).rsplit('COMMIT;',1)[0]
    schema.execute(sql)
    schema.execute('SELECT qty_lb FROM run_coverage WHERE run_id=%s',(r['id'],))
    assert schema.fetchone()['qty_lb']==5
    schema.execute("SELECT 1 FROM migration_markers WHERE name='054_run_type'")
    assert schema.fetchone() is None
    schema.execute((ROOT/'migrations/054_run_type.sql').read_text())


def test_q8_assignments_are_separate_and_idempotent(schema):
    schema.execute("INSERT INTO production_lines(name,line_code) VALUES ('Granola','granola') ON CONFLICT (line_code) DO NOTHING")
    for sku in ('90008','90025','90026'):
        schema.execute("INSERT INTO products(name,type,odoo_code,uom) SELECT %s,'batch',%s,'lb' WHERE NOT EXISTS (SELECT 1 FROM products WHERE odoo_code=%s)",('Batch '+sku,sku,sku))
    sql=(ROOT/'scripts/s1_batch_line_assignments.sql').read_text()
    schema.execute(sql);schema.execute(sql)
    schema.execute("SELECT p.odoo_code,count(*) AS n FROM product_line_assignments a JOIN products p ON p.id=a.product_id JOIN production_lines l ON l.id=a.line_id WHERE p.odoo_code IN ('90008','90025','90026') AND l.line_code='granola' GROUP BY p.odoo_code")
    assert {r['odoo_code']:r['n'] for r in schema.fetchall()}=={'90008':1,'90025':1,'90026':1}
    assert 'INSERT INTO public.product_line_assignments' not in (ROOT/'migrations/054_run_type.sql').read_text()
