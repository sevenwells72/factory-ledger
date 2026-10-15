"""Local real-page acceptance server: real handlers/DB; selectable real model.

The runner owns and drops a fresh fl_fixB3_ database. This process never reads
hosted database credentials. Only deterministic mode replaces model selection.
"""
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
import os
from pathlib import Path
import sys
import time
from urllib.parse import urlsplit
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
URL = os.environ['TEST_DATABASE_URL']
parsed = urlsplit(URL)
assert parsed.hostname == '127.0.0.1' and parsed.path.startswith('/' + os.environ.get('TEST_DB_PREFIX', 'fl_fixB3_'))
REAL_MODEL = os.environ.get('FIXB_REAL_MODEL') == '1'
os.environ.update(DATABASE_URL=URL, API_KEY='fixb-page-master',
                  DASHBOARD_API_KEY='fixb-page-read', ASSISTANT_ENABLED='1')
if REAL_MODEL:
    os.environ['OPENAI_API_KEY'] = (Path.home() / 'Documents/fl-secrets/openai-api-key.txt').read_text().strip()
else:
    os.environ.pop('OPENAI_API_KEY', None)
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
    text = (ROOT / 'migrations/061_exceptions_tables.sql').read_text()
    cur.execute(text[text.index('CREATE TABLE IF NOT EXISTS correction_reasons'):text.index('-- 3. transactions.reason_code')])
    cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES('Round B Worker','floor',%s,true) RETURNING id",
                (hashlib.sha256(b'fixb-page-floor').hexdigest(),))
    actor = cur.fetchone()['id']
    products = {}
    for pid, name, kind, sku in [
        (283, 'Sunflower Seeds', 'ingredient', '11010'),
        (284, 'Rolled Oats', 'ingredient', '11020'),
        (285, 'Batch Classic Granola #9', 'batch', '90002'),
        (286, 'Granola Classic 25 LB', 'finished', '70013'),
        (287, 'Granola Fruit Nut Batch', 'batch', '90008'),
        (288, 'Batch Classic Chocolate Chip Granola #9', 'batch', '90001'),
        (289, 'Granola Kookies & Kreme 25 LB', 'finished', '70028'),
    ]:
        cur.execute("INSERT INTO products(id,name,odoo_code,type,uom,default_batch_lb,case_size_lb,active) VALUES(%s,%s,%s,%s,'lb',10,5,true)",
                    (pid, name, sku, kind))
        products[name] = pid
    cur.execute('UPDATE products SET parent_batch_product_id=285 WHERE id=286')
    cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES(285,283,4),(285,284,6)')
    lots = {}
    for pid, code in [(283, 'SUNFLOWER-ABCD'), (284, 'OATS-EFGH'), (285, 'BATCH-B001'), (286, 'FINISHED-F001'), (289, 'KOOKIES-K001')]:
        cur.execute('INSERT INTO lots(product_id,lot_code) VALUES(%s,%s) RETURNING id', (pid, code))
        lid = cur.fetchone()['id']
        lots[str(pid)] = {'id': lid, 'code': code}
        cur.execute("INSERT INTO transactions(type,notes) VALUES('receive','Round B local page fixture') RETURNING id")
        tid = cur.fetchone()['id']
        cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,100)', (tid, pid, lid))
    suppliers = []
    names = ['Grain Supply', 'Grain Supply LLC', 'Grain-Supply', 'Farm Direct', 'Valley Ingredients', 'Seed Source', 'Organic Growers', 'Premium Foods']
    names += ['Supplier %02d' % i for i in range(1, 59)]
    for index, name in enumerate(names):
        short_code = 'SB' + chr(65 + index // 26) + chr(65 + index % 26)
        cur.execute('INSERT INTO suppliers(name,short_code) VALUES(%s,%s) RETURNING id', (name, short_code))
        suppliers.append({'id': cur.fetchone()['id'], 'name': name})
    orders = []
    for index, name in enumerate(['Sunshine Granola', 'Setton Farms', 'UNIPRO']):
        cur.execute('INSERT INTO customers(name,active) VALUES(%s,true) RETURNING id', (name,))
        cid = cur.fetchone()['id']
        cur.execute("INSERT INTO sales_orders(customer_id,order_number,status,requested_ship_date) VALUES(%s,'','confirmed','2026-10-16') RETURNING id,order_number", (cid,))
        order = dict(cur.fetchone())
        order.update(customer=name)
        orders.append(order)
        cur.execute('INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES(%s,%s,10)', (order['id'], 289 if index == 1 else 286))
    fixture = {'products': products, 'lots': lots, 'supplier': suppliers[0]['id'], 'supplier_count': len(suppliers), 'orders': orders, 'actor_id': actor}

import main
import fl_assistant
from fastapi.testclient import TestClient

main.get_db_connection = connection
main._reset_actor_cache()
client = TestClient(main.app)  # No startup/lifespan migration or sweep.
queue = []
model_requests = []


if not REAL_MODEL:
    async def fake(path, *, body=None, file=None):
        assert path == 'responses' and body['tool_choice'] == 'required'
        assert body['parallel_tool_calls'] is False and body['store'] is False
        assert not any('commit' in t['name'] for t in body['tools'])
        assert queue, 'Model fixture exhausted'
        name, args = queue.pop(0)
        model_requests.append({'tool': name, 'validated_request': True})
        schema = next(t['parameters'] for t in body['tools'] if t['name'] == name)
        args = {field: args.get(field) for field in schema['properties']}
        return {'status': 'completed', 'output': [{'type': 'function_call', 'call_id': 'call_' + uuid4().hex,
                                                   'name': name, 'arguments': json.dumps(args)}]}
    fl_assistant.openai = fake
else:
    actual_openai = fl_assistant.openai

    async def actual(path, *, body=None, file=None):
        started = time.monotonic()
        response = await actual_openai(path, body=body, file=file)
        model_requests.append({'model': body.get('model') if body else None,
                               'tools': [item.get('name') for item in response.get('output', []) if item.get('type') == 'function_call'],
                               'status': response.get('status'),
                               'elapsed_ms': round((time.monotonic() - started) * 1000)})
        return response

    fl_assistant.openai = actual


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.handle_request()

    def do_POST(self):
        self.handle_request()

    def handle_request(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))) or 'null')
            pathname = urlsplit(self.path).path
            if pathname in ('/dash/fl-assistant', '/dashboard/fl-assistant.html'):
                status, data, kind = 200, (ROOT / 'dashboard/fl-assistant.html').read_bytes(), 'text/html'
            elif pathname.startswith('/dashboard/'):
                target = (ROOT / pathname.lstrip('/')).resolve()
                assert target.is_relative_to(ROOT / 'dashboard')
                status, data, kind = 200, target.read_bytes(), mimetypes.guess_type(target)[0] or 'application/octet-stream'
            elif pathname == '/favicon.ico':
                status, data, kind = 204, b'', 'image/x-icon'
            elif pathname == '/test/fixture':
                status, data, kind = 200, json.dumps(fixture).encode(), 'application/json'
            elif pathname == '/test/model':
                assert not queue, 'Unconsumed tool calls: ' + repr(queue)
                if not REAL_MODEL:
                    queue.extend(body)
                status, data, kind = 200, b'{}', 'application/json'
            elif pathname == '/test/verify':
                with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute('SELECT count(*) AS n FROM transactions')
                    transactions = cur.fetchone()['n']
                    cur.execute("SELECT count(*) AS n FROM assistant_drafts WHERE status='committed'")
                    committed = cur.fetchone()['n']
                    cur.execute('SELECT count(*) AS n FROM suppliers')
                    supplier_count = cur.fetchone()['n']
                    cur.execute('SELECT COALESCE(sum(quantity_lb),0) AS n FROM ledger_current_transaction_lines WHERE lot_id=%s', (lots['283']['id'],))
                    sunflower = float(cur.fetchone()['n'])
                    cur.execute("SELECT kind,detail FROM exceptions WHERE kind='SHIFT_DISCREPANCY'")
                    exceptions = [dict(row) for row in cur.fetchall()]
                status, data, kind = 200, json.dumps({'transactions': transactions, 'committed': committed, 'supplier_count': supplier_count, 'sunflower_original_stock': sunflower, 'exceptions': exceptions, 'model_requests': model_requests}, default=str).encode(), 'application/json'
            else:
                response = client.request(self.command, self.path, json=body, headers={'X-API-Key': self.headers.get('X-API-Key', '')})
                status, data, kind = response.status_code, response.content, response.headers.get('content-type', 'application/json')
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            status, data, kind = 500, json.dumps({'fixture_error': type(exc).__name__}).encode(), 'application/json'
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)


server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
print(json.dumps({'port': server.server_port, 'real_model': REAL_MODEL}), flush=True)
try:
    server.serve_forever()
finally:
    client.close()
    server.server_close()
