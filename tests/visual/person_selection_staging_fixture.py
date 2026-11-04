"""Create uniquely labeled synthetic staging inventory for hosted browser proof.

Explicit opt-in, exact staging project guard, no migrations, no actor changes,
no edits/deletes of prior records. Credentials stay in memory and are not logged.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit

import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.seed_staging import secret_file
from staging_safety import assert_staging_database, PRODUCTION_DATABASE_HOST

assert sys.argv[1:] == ['--seed-staging'], 'Explicit staging fixture opt-in required'
url = secret_file(Path.home() / 'Documents/fl-secrets/staging-db-url.txt')
parsed = urlsplit(url)
assert unquote(parsed.username or '') == 'postgres.jygmyvxnxdjiiilhxseq'
assert_staging_database(url, environment='staging', production_host=PRODUCTION_DATABASE_HOST)
marker = 'STG-NAME-' + datetime.now(timezone.utc).strftime('%y%m%d%H%M%S')
out = ROOT / '.venv-test/person-selection'
out.mkdir(parents=True, exist_ok=True)
assert not (out / 'staging-fixture.json').exists(), 'Inspect existing fixture before creating another'
with psycopg2.connect(url, connect_timeout=15) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute('SET LOCAL lock_timeout = \'5s\'')
    cur.execute('SET LOCAL statement_timeout = \'30s\'')
    cur.execute('SELECT display_name,actor_id FROM person_choices ORDER BY sort_order')
    rows = cur.fetchall()
    assert [row['display_name'] for row in rows] == ['Michael', 'Luz', 'Arturo', 'Miriam']
    people = {row['display_name']: row['actor_id'] for row in rows}
    products = {}
    for name, kind in (('Ingredient', 'ingredient'), ('Batch', 'batch'), ('Cases', 'finished')):
        code = marker + '-' + name[:3].upper()
        cur.execute("INSERT INTO products(name,odoo_code,type,uom,default_batch_lb,case_size_lb,active) "
                    "VALUES(%s,%s,%s,'lb',10,5,true) RETURNING id", (marker + ' ' + name, code, kind))
        products[name] = {'id': cur.fetchone()['id'], 'code': code}
    cur.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s',
                (products['Batch']['id'], products['Cases']['id']))
    cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES(%s,%s,10)',
                (products['Batch']['id'], products['Ingredient']['id']))
    supplier_name, customer_name = marker + ' Supplier', marker + ' Customer'
    cur.execute('INSERT INTO suppliers(name,short_code) VALUES(%s,%s) RETURNING id', (supplier_name, 'SNAM'))
    supplier = cur.fetchone()['id']
    cur.execute('INSERT INTO customers(name) VALUES(%s) RETURNING id', (customer_name,))
    customer = cur.fetchone()['id']
    lots, seed_transactions = {}, []
    for name, product in products.items():
        code = marker + '-' + name.upper() + '-001'
        cur.execute('INSERT INTO lots(product_id,lot_code,supplier_lot_code) VALUES(%s,%s,%s) RETURNING id',
                    (product['id'], code, marker + '-SYNTHETIC'))
        lot = cur.fetchone()['id']
        lots[name] = {'id': lot, 'code': code}
        cur.execute("INSERT INTO transactions(type,notes,operator_id,entered_by_actor_id) "
                    "VALUES('receive',%s,'Michael',%s) RETURNING id",
                    (marker + ': synthetic staging-only acceptance stock; not physically verified', people['Michael']))
        tx = cur.fetchone()['id']; seed_transactions.append(tx)
        cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,100)',
                    (tx, product['id'], lot))
    fixture = {'marker': marker, 'people': people, 'products': products, 'lots': lots,
               'supplier': supplier, 'supplier_name': supplier_name, 'customer': customer,
               'customer_name': customer_name, 'seed_transactions': seed_transactions}
    # Write fixture evidence before commit; if the write fails, the database transaction rolls back.
    (out / 'staging-fixture.json').write_text(json.dumps(fixture, indent=2))
print('Created synthetic staging-only fixture ' + marker)
