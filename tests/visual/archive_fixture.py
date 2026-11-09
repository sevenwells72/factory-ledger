"""Seed synthetic BS / CNS catalog data and serve the real app for the archive screenshots.

Local only: refuses any host but localhost and any database not named
fl_archive_browser_*. Never reads production credentials. Product names come
from dashboard/dashboard_config.json so the real on-hand panels fill in; the
BS quantities mirror the production read-only dry run of 2026-11-09, and
Sunflower Seeds carries 2,114 lb. The four 6x8 OZ cases start archived, as in
production (row 152). One old Made record (2026-07-29) consumes BS items so the
history view can be checked after archiving.

Run with TEST_DATABASE_URL=postgresql://localhost:5432/fl_archive_browser_<x>
and the pinned test interpreter; then tests/visual/run-archive-screenshots.mjs.
"""
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
assert parsed.path.startswith('/fl_archive_browser_'), 'Dedicated browser database required'
os.environ.update(DATABASE_URL=url, API_KEY='archive-browser-machine', DASHBOARD_API_KEY='archive-browser-read',
                  PERSON_SELECT_ENABLED='0', KEY_SESSIONS_ENABLED='0', PIN_SIGNIN_ENABLED='0',
                  PACKING_SLIP_SIGNING_KEY='archive-browser-signing')

BS_ON_HAND = {
    'Batch BS Dark Chocolate Granola 350': 1078, 'BS Banana Bites – Small': 484, 'BS Cacao Nibs': 1400,
    'BS Cocoa Liquor – Chips': 4400, 'BS Hazelnuts – Diced': 1725, 'BS Oats GF': 2000,
    'BS Oil – Almond': 3200, 'BS Pumpkin Seeds': 110, 'Sunflower Seeds': 2114,
}
EXTRA_BS_INGREDIENTS = ['BS Coconut – Chips', 'BS Hazelnuts']
SIX_BY_EIGHT = 'retail_bs_8oz'

config = json.loads((ROOT / 'dashboard/dashboard_config.json').read_text())
catalog = {}   # name -> (type, active)
for panel in config['finished_goods_panels'] + [config['coconut_panel']]:
    for sku in panel['skus']:
        catalog[sku] = ('finished', panel['id'] != SIX_BY_EIGHT)
for batch in config['batch_skus']:
    catalog[batch['name']] = ('batch', True)
for cat in config['ingredient_categories']:
    for item in cat['items']:
        catalog.setdefault(item, ('packaging' if cat['id'] == 'bs_packaging' else 'ingredient', True))
for item in EXTRA_BS_INGREDIENTS:
    catalog[item] = ('ingredient', True)

with psycopg2.connect(url) as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
    cur.execute('SELECT count(*) AS n FROM products')
    assert cur.fetchone()['n'] == 0, 'Refusing to seed a used database'
    for name in ('083_customer_tickets', '084_assistant_service_alert', '088_ship_order_idempotency',
                 '089_device_pins', '090_make_lot_selection', '091_trusted_browsers', '093_product_archive'):
        cur.execute((ROOT / 'migrations' / (name + '.sql')).read_text())
    for order, (name, role, label) in enumerate((('Michael', 'owner', 'Owner'), ('Luz', 'office', 'Office'),
                                               ('Arturo', 'floor', 'Floor Lead'), ('Miriam', 'office', 'Office')), 1):
        cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES(%s,%s,md5(%s)||md5(%s),true) RETURNING id",
                    (name, role, name, name[::-1]))
        cur.execute('INSERT INTO person_choices(actor_id,display_name,role_label,sort_order) VALUES(%s,%s,%s,%s)',
                    (cur.fetchone()['id'], name, label, order))
    ids, lots = {}, {}
    for i, (name, (kind, active)) in enumerate(sorted(catalog.items())):
        cur.execute("""INSERT INTO products(name,type,uom,active,case_size_lb,default_batch_lb)
                       VALUES(%s,%s,%s,%s,%s,%s) RETURNING id""",
                    (name, kind, 'units' if kind == 'packaging' else 'lb', active,
                     2.625 if '6x7' in name or '12x10' in name else (25 if '25 LB' in name else 10)
                     if kind == 'finished' else None, 350 if kind == 'batch' else None))
        ids[name] = cur.fetchone()['id']
        qty = BS_ON_HAND.get(name, 0 if name.startswith(('BS ', 'Batch BS', 'Bag')) or not active
                             else 250 + (i * 37) % 900)
        if qty:
            code = f'ARCH-{ids[name]:04d}'
            cur.execute('INSERT INTO lots(product_id,lot_code) VALUES(%s,%s) RETURNING id', (ids[name], code))
            lots[name] = cur.fetchone()['id']
            cur.execute("INSERT INTO transactions(type,notes,operator_id) VALUES('receive','Synthetic archive fixture','fixture') RETURNING id")
            cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,%s)',
                        (cur.fetchone()['id'], ids[name], lots[name], qty))
    # BS recipe (also uses the CNS Sunflower Seeds) and a CNS recipe.
    for batch, items in (('Batch BS Dark Chocolate Granola 350', ['BS Oats GF', 'BS Cacao Nibs', 'Sunflower Seeds']),
                         ('Batch Classic Granola #9', ['Oats', 'Sunflower Seeds', 'Honey'])):
        for item in items:
            cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES(%s,%s,20)',
                        (ids[batch], ids[item]))
    # Old Made record (like production TX 1785, 2026-07-29) that used BS items.
    cur.execute("""INSERT INTO lots(product_id,lot_code) VALUES(%s,'26-07-29-BSDC-001') RETURNING id""",
                (ids['Batch BS Dark Chocolate Granola 350'],))
    made_lot = cur.fetchone()['id']
    cur.execute("""INSERT INTO transactions(type,timestamp,occurred_at,business_date,notes,operator_id)
                   VALUES('make','2026-07-29 14:00','2026-07-29 10:00-04','2026-07-29','Synthetic old BS make','Arturo')
                   RETURNING id""")
    made = cur.fetchone()['id']
    cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,350),'
                '(%s,%s,%s,-200),(%s,%s,%s,-150)',
                (made, ids['Batch BS Dark Chocolate Granola 350'], made_lot,
                 made, ids['BS Oats GF'], lots['BS Oats GF'], made, ids['BS Cacao Nibs'], lots['BS Cacao Nibs']))
    out = ROOT / '.venv-test/archive-browser'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'fixture.json').write_text(json.dumps({'ids': ids, 'made_transaction': made,
                                                'made_lot': '26-07-29-BSDC-001'}, ensure_ascii=False))

import uvicorn
uvicorn.run('main:app', host='127.0.0.1', port=int(os.environ.get('ARCHIVE_BROWSER_PORT', '8157')),
            log_level='warning', access_log=False)
