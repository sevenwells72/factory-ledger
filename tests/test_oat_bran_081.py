"""Michael's October 14 oat bran correction: future recipes only, strict rollback."""
import json
from decimal import Decimal
from pathlib import Path
import psycopg2
import pytest

ROOT = Path(__file__).resolve().parents[1]
UP = ROOT/'migrations/081_restore_ss_oat_bran.sql'
DOWN = ROOT/'migrations/down/081_restore_ss_oat_bran_down.sql'
TARGET = {114: Decimal(29),116:Decimal(28),118:Decimal(28)}
pytestmark = pytest.mark.db


def seed(cur):
    snapshot = json.loads((ROOT/'docs/deployments/pilot-triage-production-recipe-read.json').read_text())
    products = {p['id']:p for p in snapshot['products']}
    for f in snapshot['formulas']:
        products.setdefault(f['ingredient_product_id'],{'id':f['ingredient_product_id'],'odoo_code':f['ingredient_code'], 'name':f['ingredient_name'],'type':'ingredient','default_batch_lb':None})
    for p in products.values():
        cur.execute('INSERT INTO products(id,odoo_code,name,type,default_batch_lb,active,is_service) VALUES (%s,%s,%s,%s,%s,true,false)',tuple(p[k] for k in ['id','odoo_code','name','type','default_batch_lb']))
    for f in snapshot['formulas']:
        cur.execute('INSERT INTO batch_formulas(id,product_id,ingredient_product_id,quantity_lb,exclude_from_inventory) VALUES (%s,%s,%s,%s,%s)',tuple(f[k] for k in ['id','product_id','ingredient_product_id','quantity_lb','exclude_from_inventory']))


def quantities(cur):
    cur.execute('SELECT product_id,quantity_lb FROM batch_formulas WHERE ingredient_product_id=31 AND product_id IN (114,116,118)')
    return {r['product_id']:r['quantity_lb'] for r in cur.fetchall()}


def apply(cur, path):
    cur.execute(path.read_text())


def test_restore_exact_original_oat_bran_without_changing_batches_or_history(db_cursor):
    seed(db_cursor)
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('make','immutable pilot history') RETURNING id")
    txn=db_cursor.fetchone()['id']
    for pid, qty in [(116,350),(31,0)]:
        db_cursor.execute("INSERT INTO lots(product_id,lot_code) VALUES (%s,'PILOT-HISTORY') RETURNING id",(pid,))
        lid=db_cursor.fetchone()['id']
        db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,%s)',(txn,pid,lid,qty))
    db_cursor.execute('SELECT row_to_json(t) AS row FROM transactions t WHERE id=%s',(txn,));before=db_cursor.fetchone()['row']
    if UP.exists(): apply(db_cursor,UP)
    assert quantities(db_cursor)==TARGET
    db_cursor.execute('SELECT id,default_batch_lb FROM products WHERE id IN (114,116,118)')
    assert {r['id']:r['default_batch_lb'] for r in db_cursor.fetchall()}=={114:393,116:350,118:379}
    db_cursor.execute('SELECT row_to_json(t) AS row FROM transactions t WHERE id=%s',(txn,));assert db_cursor.fetchone()['row']==before
    db_cursor.execute('SELECT quantity_lb FROM transaction_lines WHERE transaction_id=%s AND product_id=31',(txn,));assert db_cursor.fetchone()['quantity_lb']==0


def test_migration_rerunnable_and_down_restores_only_the_three_values(db_cursor):
    seed(db_cursor)
    db_cursor.execute('SELECT row_to_json(b) AS row FROM batch_formulas b ORDER BY id');before=db_cursor.fetchall()
    apply(db_cursor,UP);apply(db_cursor,UP)
    assert quantities(db_cursor)==TARGET
    apply(db_cursor,DOWN)
    db_cursor.execute('SELECT row_to_json(b) AS row FROM batch_formulas b ORDER BY id');assert db_cursor.fetchall()==before
    apply(db_cursor,UP);assert quantities(db_cursor)==TARGET


@pytest.mark.parametrize('drift',[
    'UPDATE batch_formulas SET quantity_lb=1 WHERE product_id=116 AND ingredient_product_id=31',
    'UPDATE batch_formulas SET exclude_from_inventory=true WHERE product_id=116 AND ingredient_product_id=31',
    'DELETE FROM batch_formulas WHERE product_id=116 AND ingredient_product_id=31',
    "UPDATE products SET odoo_code='wrong' WHERE id=116",
    'UPDATE products SET default_batch_lb=322 WHERE id=116',
    'UPDATE products SET active=false WHERE id=31',
    'UPDATE batch_formulas SET quantity_lb=184 WHERE product_id=116 AND ingredient_product_id=32',
])
def test_migration_refuses_drift_without_partial_correction(db_cursor,drift):
    seed(db_cursor);db_cursor.execute(drift);before=quantities(db_cursor)
    db_cursor.execute('SAVEPOINT drift')
    with pytest.raises(psycopg2.Error,match='081 refused'): apply(db_cursor,UP)
    db_cursor.execute('ROLLBACK TO SAVEPOINT drift');assert quantities(db_cursor)==before


@pytest.mark.parametrize('drift',[
    'UPDATE batch_formulas SET quantity_lb=0 WHERE product_id=116 AND ingredient_product_id=31',
    'UPDATE batch_formulas SET exclude_from_inventory=true WHERE product_id=114 AND ingredient_product_id=31',
    'UPDATE products SET default_batch_lb=350 WHERE id=118',
    "UPDATE products SET odoo_code='wrong' WHERE id=31",
])
def test_strict_down_refuses_any_non_after_state(db_cursor,drift):
    seed(db_cursor);apply(db_cursor,UP);db_cursor.execute(drift)
    db_cursor.execute('SAVEPOINT drift')
    with pytest.raises(psycopg2.Error,match='081 down refused'): apply(db_cursor,DOWN)
    db_cursor.execute('ROLLBACK TO SAVEPOINT drift')
