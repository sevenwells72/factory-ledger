"""PR #143 actual-page bridge: real local FL on a throwaway database.
Seeds a Product, a Coffee-named Product (opens one Charge/Product check) and a
Pallet Charge on one order. OpenAI is hard-refused; nothing leaves 127.0.0.1.
"""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import hashlib
import json
import logging
import mimetypes
import os
import signal
import sys
import tempfile

import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
URL = os.environ['TEST_DATABASE_URL']
parsed = urlsplit(URL)
assert parsed.hostname == '127.0.0.1' and parsed.path.startswith('/fl_charge_')
os.environ.update(DATABASE_URL=URL, API_KEY='charge-master', DASHBOARD_API_KEY='charge-read', ASSISTANT_ENABLED='0',
                  ENVIRONMENT='test', EVIDENCE_STORAGE_DIR=tempfile.mkdtemp(prefix='fl_charge_evidence_'))
for name in ('OPENAI_API_KEY', 'SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY'):
    os.environ.pop(name, None)
logging.disable(logging.CRITICAL)


@contextmanager
def connection():
    conn = psycopg2.connect(URL)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
    import charge_setting
    actors = {}
    for role, name in [('floor', 'Arturo Charge Check'), ('office', 'Luz Charge Check'), ('owner', 'Michael Charge Check')]:
        key = 'charge-' + role
        cur.execute('INSERT INTO actors(name,role,key_hash,active) VALUES(%s,%s,%s,true) RETURNING id',
                    (name, role, hashlib.sha256(key.encode()).hexdigest()))
        actors[role] = {'id': cur.fetchone()['id'], 'key': key, 'name': name}
    products = {}
    for name, sku, kind, uom, case, service in [('Classic Granola 25 LB', '70001', 'finished', '25 lb case', 25, False),
                                                ('Setton Coffee Granola 25 LB', '70012', 'finished', '25 lb case', 25, False),
                                                ('Pallet Charge', 'PALLET', 'finished', 'each', None, True)]:
        cur.execute('''INSERT INTO products(name,odoo_code,type,uom,case_size_lb,is_service,active)
                       VALUES(%s,%s,%s,%s,%s,%s,true) RETURNING id''', (name, sku, kind, uom, case, service))
        products[name] = cur.fetchone()['id']
    for name, code in [('Classic Granola 25 LB', '26-10-19-CLAS-001'), ('Setton Coffee Granola 25 LB', '26-10-19-SETT-002')]:
        cur.execute('INSERT INTO lots(product_id,lot_code) VALUES(%s,%s) RETURNING id', (products[name], code))
        lot = cur.fetchone()['id']
        cur.execute("INSERT INTO transactions(type,notes) VALUES('receive','PR #143 page fixture') RETURNING id")
        cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,500)',
                    (cur.fetchone()['id'], products[name], lot))
    cur.execute("INSERT INTO customers(name,active) VALUES('Whole Foods',true) RETURNING id")
    customer = cur.fetchone()['id']
    cur.execute("""INSERT INTO sales_orders(customer_id,order_number,state,status,order_date,requested_ship_date)
                   VALUES(%s,'SO-261020-143','open','confirmed',current_date,current_date+2) RETURNING id""", (customer,))
    order = cur.fetchone()['id']
    for name, lb, quantity, unit, weight, price in [('Classic Granola 25 LB', 100, 4, 'cases', 25, 45.0),
                                                    ('Setton Coffee Granola 25 LB', 50, 2, 'cases', 25, 48.0),
                                                    ('Pallet Charge', 0, 1, 'each', None, 25.0)]:
        cur.execute('''INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb,ordered_quantity,ordered_unit,
                       ordered_case_weight_lb,unit_price,amount) VALUES(%s,%s,%s,%s,%s,%s,%s,%s)''',
                    (order, products[name], lb, quantity, unit, weight, price, quantity * price))
    charge_setting.sweep(cur)        # what migration 085 / the nightly sweep does for existing items
    fixture = {'actors': actors, 'products': products, 'order': order, 'order_number': 'SO-261020-143'}

import main
import fl_assistant
from fastapi.testclient import TestClient


async def refuse_openai(*args, **kwargs):
    raise AssertionError('OpenAI is forbidden in this runner')
fl_assistant.openai = refuse_openai
main.get_db_connection = connection
main._reset_actor_cache()
client = TestClient(main.app)  # no lifespan: no startup sweeps


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.handle_request()

    def do_POST(self):
        self.handle_request()

    def handle_request(self):
        try:
            pathname = urlsplit(self.path).path
            raw = self.rfile.read(int(self.headers.get('Content-Length', '0')))
            if pathname.startswith('/dashboard/'):
                target = (ROOT / pathname.lstrip('/')).resolve()
                assert target.is_relative_to(ROOT / 'dashboard')
                status, data, kind = 200, target.read_bytes(), mimetypes.guess_type(target)[0] or 'application/octet-stream'
            elif pathname == '/favicon.ico':
                status, data, kind = 204, b'', 'image/x-icon'
            elif pathname == '/test/fixture':
                status, data, kind = 200, json.dumps(fixture).encode(), 'application/json'
            else:
                assert not pathname.startswith('/assistant/turn'), 'Assistant turns forbidden'
                headers = {k: v for k, v in self.headers.items() if k.lower() in ('x-api-key', 'content-type', 'x-fl-background')}
                response = client.request(self.command, self.path, content=raw, headers=headers)
                status, data, kind = response.status_code, response.content, response.headers.get('content-type', 'application/json')
                if status >= 500:
                    print(pathname, data.decode(errors='replace'), file=sys.stderr, flush=True)
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stderr)
            status, data, kind = 500, b'{"fixture_error":"local bridge failed"}', 'application/json'
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)


server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
print(json.dumps({'port': server.server_port, 'local_only': True, 'openai_forbidden': True}), flush=True)
signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(SystemExit(0)))
try:
    server.serve_forever()
finally:
    client.close()
    server.server_close()
    import shutil
    shutil.rmtree(os.environ['EVIDENCE_STORAGE_DIR'], ignore_errors=True)
