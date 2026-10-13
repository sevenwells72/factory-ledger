#!/usr/bin/env python3
"""D2 browser against hosted staging API; local branch assets, no deploy or migrations."""
from pathlib import Path
import os,json,secrets,subprocess
from uuid import uuid4
from hashlib import sha256
from psycopg2.extras import RealDictCursor
from check_assistant_staging import connect,BASE,ROOT

def check():
 marker='STG-D12-'+uuid4().hex[:8].upper();key=secrets.token_urlsafe(40);actor_id=None
 fixture={'base':BASE,'marker':marker,'actor_name':marker+' Floor','products':{}}
 try:
  with connect() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
   cur.execute("SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='30s'")
   cur.execute('INSERT INTO actors(name,role,key_hash,active) VALUES(%s,\'floor\',%s,true) RETURNING id',(fixture['actor_name'],sha256(key.encode()).hexdigest()));actor_id=cur.fetchone()['id']
   for name,kind,weight in [('Oat Bran','ingredient',None),('Rolled Oats','ingredient',None),('Batch','batch',10)]:
    cur.execute("INSERT INTO products(name,odoo_code,type,uom,default_batch_lb,active) VALUES(%s,%s,%s,'lb',%s,true) RETURNING id,name,odoo_code",(marker+' '+name,marker+'-'+name.replace(' ','-'),kind,weight));fixture['products'][name]=dict(cur.fetchone())
   cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES(%s,%s,10)',(fixture['products']['Batch']['id'],fixture['products']['Oat Bran']['id']))
   cur.execute('INSERT INTO suppliers(name) VALUES(%s) RETURNING id,name',(marker+' Supplier',));fixture['supplier']=dict(cur.fetchone())
  output=ROOT/'.venv-test/d12-staging-fixture.json';output.write_text(json.dumps(fixture))
  env=dict(os.environ,D12_STAGING_PERSONAL_KEY=key)
  subprocess.run(['node','tests/visual/run-d12-staging.mjs'],cwd=ROOT,env=env,check=True)
 finally:
  if actor_id:
   with connect() as conn,conn.cursor() as cur:cur.execute('UPDATE actors SET active=false WHERE id=%s',(actor_id,))
   print('Synthetic D12 staging actor deactivated. No production or deployment changes.')
if __name__=='__main__':check()
