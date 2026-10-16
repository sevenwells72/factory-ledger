"""Real local dashboard/API/ledger bridge for the round C phone transcripts.

Only model selection is queued by default; FIXC_REAL_MODEL=1 uses the model
transport with a caller-supplied OPENAI_API_KEY. No hosted DB is accessed. Actual-model mode reads its key inside this process only.
"""
from contextlib import contextmanager
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import mimetypes
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT))
URL = os.environ['TEST_DATABASE_URL']
parsed = urlsplit(URL)
assert parsed.hostname == '127.0.0.1' and parsed.path.startswith('/' + os.environ.get('TEST_DB_PREFIX', 'fl_fixC_'))
REAL_MODEL = os.environ.get('FIXC_REAL_MODEL') == '1'
os.environ.update(DATABASE_URL=URL, API_KEY='fixc-page-master',
                  DASHBOARD_API_KEY='fixc-page-read', ASSISTANT_ENABLED='1')
if not REAL_MODEL:
    os.environ.pop('OPENAI_API_KEY', None)
else:
    if not os.environ.get('OPENAI_API_KEY'):
        os.environ['OPENAI_API_KEY'] = (Path.home() / 'Documents/fl-secrets/openai-api-key.txt').read_text().strip()
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
    cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES('Round C Worker','owner',%s,true) RETURNING id",
                (sha256(b'fixc-page-worker').hexdigest(),))
    actor = cur.fetchone()['id']
    products = {}
    for key, name, kind in [
        ('seeds', 'Sunflower Seeds', 'ingredient'),
        ('classic', 'Batch Classic Granola #9', 'batch'),
        ('ss1', 'Batch SS #1', 'batch'), ('ss2', 'Batch SS #2', 'batch'), ('ss3', 'Batch SS #3', 'batch'),
        ('fruit_batch', 'Fruit and Nut Granola Batch', 'batch'),
        ('fruit', 'Fruit and Nut Granola', 'finished'), ('sprinkles', 'Rainbow Sprinkles', 'finished'),
    ]:
        cur.execute("""INSERT INTO products(name,type,uom,default_batch_lb,case_size_lb,active)
            VALUES(%s,%s,'lb',10,5,true) RETURNING id,name""", (name, kind))
        products[key] = dict(cur.fetchone())
    cur.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s', (products['fruit_batch']['id'], products['fruit']['id']))
    for key in ('classic', 'ss1', 'ss2', 'ss3'):
        cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES(%s,%s,10)', (products[key]['id'], products['seeds']['id']))
    for key, code in [('seeds', 'SUNFLOWER-ABCD'), ('classic', 'CLASSIC-9012'), ('fruit_batch', '1234'), ('ss1', 'P1585')]:
        item = products[key]
        cur.execute('INSERT INTO lots(product_id,lot_code) VALUES(%s,%s) RETURNING id', (item['id'], code))
        item.update(lot_id=cur.fetchone()['id'], lot_code=code)
        cur.execute("INSERT INTO transactions(type,notes) VALUES('receive','Round C browser fixture') RETURNING id")
        tid = cur.fetchone()['id']
        cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,1000)', (tid, item['id'], item['lot_id']))
    suppliers = []
    for name in ('Grain Supply', 'A-1 Bakery Supply', 'Bakers Depot'):
        cur.execute('INSERT INTO suppliers(name) VALUES(%s) RETURNING id,name', (name,))
        suppliers.append(dict(cur.fetchone()))
    customers = []
    for name in ('Lake Bakery', 'Rainbow Bakery'):
        cur.execute('INSERT INTO customers(name,active) VALUES(%s,true) RETURNING id,name', (name,))
        customers.append(dict(cur.fetchone()))
    fixture = {'products': products, 'suppliers': suppliers, 'customers': customers, 'actor_id': actor}

import main
import fl_assistant
from fastapi.testclient import TestClient

main.get_db_connection = connection
main._reset_actor_cache()
client = TestClient(main.app)
queue = []
requests = []
if not REAL_MODEL:
    async def fake(path, *, body=None, file=None):
        assert path == 'responses' and body['tool_choice'] == 'required'
        assert body['parallel_tool_calls'] is False and body['store'] is False
        assert not any('commit' in t['name'] for t in body['tools'])
        assert queue, 'Model fixture exhausted'
        name, args = queue.pop(0)
        requests.append({'tool': name, 'normalized_user_text': [r.get('content') for r in body['input'] if r.get('role') == 'user']})
        schema = next(t['parameters'] for t in body['tools'] if t['name'] == name)
        args = {field: args.get(field) for field in schema['properties']}
        return {'status': 'completed', 'output': [{'type': 'function_call', 'call_id': 'call_' + uuid4().hex,
                                                   'name': name, 'arguments': json.dumps(args)}]}
    fl_assistant.openai = fake
else:
    original_openai = fl_assistant.openai

    async def actual(path, *, body=None, file=None):
        result = await original_openai(path, body=body, file=file)
        requests.append({'model': body.get('model'), 'tools': [r.get('name') for r in result.get('output', []) if r.get('type') == 'function_call'], 'status': result.get('status')})
        return result

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
                    cur.execute('SELECT count(*) AS n FROM sales_orders')
                    orders = cur.fetchone()['n']
                status, data, kind = 200, json.dumps({'transactions': transactions, 'committed': committed, 'orders': orders, 'model_requests': requests}).encode(), 'application/json'
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
