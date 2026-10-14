"""Seed and serve a dedicated LOCAL browser DB. Never reads credential files."""
import os,sys,json,hashlib
from pathlib import Path
from urllib.parse import urlsplit
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
url=os.environ['TEST_DATABASE_URL']; parsed=urlsplit(url)
assert parsed.hostname in ('localhost','127.0.0.1') and parsed.path.startswith('/fl_d12_browser'), 'Dedicated local DB required'
os.environ.update(DATABASE_URL=url,API_KEY='d12-local-master',DASHBOARD_API_KEY='d12-local-read',KEY_SESSIONS_ENABLED=os.environ.get('KEY_SESSIONS_ENABLED','1'),EVIDENCE_STORAGE_DIR=str(ROOT/'.venv-test/d12-evidence'),PACKING_SLIP_SIGNING_KEY='d12-local-signing')
# O-7: pages read DASHBOARD_API_KEY from GET /auth/dashboard-key, so any value works.
import psycopg2
from psycopg2.extras import RealDictCursor
with psycopg2.connect(url) as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
 s=(ROOT/'migrations/061_exceptions_tables.sql').read_text();cur.execute(s[s.index('CREATE TABLE IF NOT EXISTS correction_reasons'):s.index('-- 3. transactions.reason_code')])
 actors={}
 for role,name in [('floor','Arturo D12'),('owner','Owner D12'),('office','Office D12')]:
  key='d12-local-'+role;cur.execute('INSERT INTO actors(name,role,key_hash,active) VALUES(%s,%s,%s,true) RETURNING id',(name,role,hashlib.sha256(key.encode()).hexdigest()));actors[role]=cur.fetchone()['id']
 products={}
 for name,kind,code in [('Oat Bran','ingredient','D1201'),('Rolled Oats','ingredient','D1202'),('Chips 4000 CT','ingredient','D1203'),('Chips 1000 CT','ingredient','D1204'),('Test Batch','batch','D1290'),('Test Cases','finished','D1270')]:
  cur.execute("INSERT INTO products(name,odoo_code,type,uom,default_batch_lb,case_size_lb,active) VALUES(%s,%s,%s,'lb',10,5,true) RETURNING id",('D12 '+name,code,kind));products[name]=cur.fetchone()['id']
 cur.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s',(products['Test Batch'],products['Test Cases']))
 cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES(%s,%s,10)',(products['Test Batch'],products['Oat Bran']))
 cur.execute("INSERT INTO suppliers(name,short_code) VALUES('D12 Test Supplier','DTST') RETURNING id");supplier=cur.fetchone()['id']
 cur.execute("INSERT INTO customers(name) VALUES('D12 Test Customer') RETURNING id");customer=cur.fetchone()['id']
 lots={}
 for index,(name,pid) in enumerate(products.items()):
  code=f'D12-{index}-001';cur.execute('INSERT INTO lots(product_id,lot_code) VALUES(%s,%s) RETURNING id',(pid,code));lid=cur.fetchone()['id'];lots[name]={'id':lid,'code':code}
  cur.execute("INSERT INTO transactions(type,notes) VALUES('receive','D12 local browser fixture') RETURNING id");tx=cur.fetchone()['id'];cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,100)',(tx,pid,lid))
 cur.execute("INSERT INTO sales_orders(customer_id,order_number,order_date,status) VALUES(%s,'SO-D12-BROWSER',CURRENT_DATE,'confirmed') RETURNING id",(customer,));order=cur.fetchone()['id']
 cur.execute('INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb,unit_price) VALUES(%s,%s,5,1)',(order,products['Test Cases']))
 data={'products':products,'lots':lots,'supplier':supplier,'customer':customer,'actors':actors,'order':order}
 if os.environ.get('D2_OFFICE_FIXTURE')=='1':
  from tests.visual.office_seed import seed_office
  seed_office(cur,data)
 (ROOT/'.venv-test/d12-fixture.json').write_text(json.dumps(data))
import uvicorn
uvicorn.run('main:app',host='127.0.0.1',port=int(os.environ.get('D12_PORT','8122')),log_level='error',access_log=False)
