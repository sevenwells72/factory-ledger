"""Create synthetic data and serve the real app on a dedicated local browser DB.

The database must already contain tests/schema/schema.sql and pg_trgm. This
runner refuses hosted databases and existing fixture data. It never reads
production credentials. Run with TEST_DATABASE_URL=postgresql://localhost:5432/
fl_person_browser_<unique> and the project's pinned test interpreter.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
url = os.environ['TEST_DATABASE_URL']
parsed = urlsplit(url)
assert parsed.hostname in ('localhost', '127.0.0.1')
assert parsed.path.startswith('/fl_person_browser_'), 'Dedicated browser database required'
out = ROOT / '.venv-test/person-selection'
out.mkdir(parents=True, exist_ok=True)
os.environ.update(
    DATABASE_URL=url, API_KEY='person-browser-machine', DASHBOARD_API_KEY='person-browser-read',
    PERSON_SELECT_ENABLED='0', KEY_SESSIONS_ENABLED='0', PIN_SIGNIN_ENABLED='0',
    EVIDENCE_STORAGE_DIR=str(out / 'evidence'), PACKING_SLIP_SIGNING_KEY='person-browser-signing',
)
with psycopg2.connect(url) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute('SELECT count(*) AS n FROM actors')
    assert cur.fetchone()['n'] == 0, 'Refusing to seed a used database'
    for name in ('083_customer_tickets', '084_assistant_service_alert', '088_ship_order_idempotency',
                 '089_device_pins', '090_make_lot_selection', '091_trusted_browsers'):
        cur.execute((ROOT / 'migrations' / (name + '.sql')).read_text())
    reasons = (ROOT / 'migrations/061_exceptions_tables.sql').read_text()
    cur.execute(reasons[reasons.index('CREATE TABLE IF NOT EXISTS correction_reasons'):
                        reasons.index('-- 3. transactions.reason_code')])
    people = {}
    for order, (name, role, label) in enumerate((('Michael', 'owner', 'Owner'), ('Luz', 'office', 'Office'),
                                               ('Arturo', 'floor', 'Floor Lead'), ('Miriam', 'office', 'Office')), 1):
        # Distinct historical actor names prove the new transaction snapshots use the tapped name.
        cur.execute('INSERT INTO actors(name,role,key_hash,active) VALUES(%s,%s,%s,true) RETURNING id',
                    ('Historical ' + name, role, hashlib.sha256(('unused-' + name).encode()).hexdigest()))
        people[name] = cur.fetchone()['id']
        cur.execute('INSERT INTO person_choices(actor_id,display_name,role_label,sort_order) VALUES(%s,%s,%s,%s)',
                    (people[name], name, label, order))
    products = {}
    for name, kind, code in (('Ingredient', 'ingredient', 'PS-01'), ('Batch', 'batch', 'PS-90'),
                             ('Cases', 'finished', 'PS-70')):
        cur.execute("INSERT INTO products(name,odoo_code,type,uom,default_batch_lb,case_size_lb,active) "
                    "VALUES(%s,%s,%s,'lb',10,5,true) RETURNING id", ('Person Browser ' + name, code, kind))
        products[name] = {'id': cur.fetchone()['id'], 'code': code}
    cur.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s',
                (products['Batch']['id'], products['Cases']['id']))
    cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES(%s,%s,10)',
                (products['Batch']['id'], products['Ingredient']['id']))
    cur.execute("INSERT INTO suppliers(name,short_code) VALUES('Person Browser Supplier','PBST') RETURNING id")
    supplier = cur.fetchone()['id']
    cur.execute("INSERT INTO customers(name) VALUES('Person Browser Customer') RETURNING id")
    customer = cur.fetchone()['id']
    lots = {}
    for name, product in products.items():
        code = 'PS-' + name.upper() + '-001'
        cur.execute('INSERT INTO lots(product_id,lot_code) VALUES(%s,%s) RETURNING id', (product['id'], code))
        lot = cur.fetchone()['id']
        lots[name] = {'id': lot, 'code': code}
        cur.execute("INSERT INTO transactions(type,notes) VALUES('receive','Synthetic local browser fixture') RETURNING id")
        cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,1000)',
                    (cur.fetchone()['id'], product['id'], lot))
    fixture = {'people': people, 'products': products, 'lots': lots, 'supplier': supplier, 'customer': customer}
    if os.environ.get('COCONUT_BROWSER') == '1':
        # Weighed batch output (run-weighed-make.mjs): sweetened coconut, 360 lb pan, recipe as on
        # fl-staging product 125 on 2026-11-26 (water is excluded from inventory).
        coco, coco_lots = {}, {}
        for name, kind, code, pan in (('Coconut Sweetened Fancy', 'batch', 'CB-90003', 360),
                                      ('Coconut Fancy Desiccated', 'ingredient', 'CB-10', None),
                                      ('Corn Starch', 'ingredient', 'CB-16', None), ('Glycol', 'ingredient', 'CB-29', None),
                                      ('Salt', 'ingredient', 'CB-39', None), ('Sugar 6X', 'ingredient', 'CB-40', None),
                                      ('Water', 'ingredient', 'CB-50', None)):
            cur.execute("INSERT INTO products(name,odoo_code,type,uom,default_batch_lb,yield_multiplier,active) "
                        "VALUES(%s,%s,%s,'lb',%s,1.0,true) RETURNING id", ('Coconut Browser ' + name, code, kind, pan))
            coco[name] = cur.fetchone()['id']
        for ingredient, lb, excluded in (('Coconut Fancy Desiccated', 200, False), ('Corn Starch', 5, False),
                                         ('Glycol', 7, False), ('Salt', 2.5, False), ('Sugar 6X', 110, False),
                                         ('Water', 65, True)):
            cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb,exclude_from_inventory) '
                        'VALUES(%s,%s,%s,%s)', (coco['Coconut Sweetened Fancy'], coco[ingredient], lb, excluded))
        for name, qty in (('Coconut Fancy Desiccated', 5000), ('Corn Starch', 200), ('Glycol', 200),
                          ('Salt', 100), ('Sugar 6X', 3000)):
            code = 'CB-' + name.split()[0].upper() + '-001'
            cur.execute("INSERT INTO lots(product_id,lot_code,received_at) VALUES(%s,%s,now()-interval '3 day') RETURNING id",
                        (coco[name], code))
            coco_lots[name] = {'id': cur.fetchone()['id'], 'code': code}
            cur.execute("INSERT INTO transactions(type,notes) VALUES('receive','Coconut weighed-output browser fixture') "
                        "RETURNING id")
            cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,%s)',
                        (cur.fetchone()['id'], coco[name], coco_lots[name]['id'], qty))
        fixture['coconut'] = {'products': coco, 'lots': coco_lots}
    (out / 'fixture.json').write_text(json.dumps(fixture))

import uvicorn
uvicorn.run('main:app', host='127.0.0.1', port=int(os.environ.get('PERSON_BROWSER_PORT', '8149')),
            log_level='warning', access_log=False)
