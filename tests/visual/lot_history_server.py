"""Real HTTP/FastAPI/local-PostgreSQL browser harness. Never use production.

Run via run-lot-history.mjs. Application lifespan is OFF, fixture writes remain
in one rolled-back local transaction, and API responses are not route stubs.
"""
import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
url = os.environ.get('TEST_DATABASE_URL', 'postgresql://localhost:5432/factory_ledger_test')
parsed = urlparse(url)
assert parsed.hostname in ('localhost', '127.0.0.1') and parsed.path == '/factory_ledger_test'
os.environ['DATABASE_URL'] = url
os.environ['DASHBOARD_API_KEY'] = 'lot-history-browser-key'
os.environ['API_KEY'] = 'lot-history-browser-owner'
import psycopg2
from psycopg2.extras import RealDictCursor
import uvicorn
import main
from tests.test_trace_batch_input_tree import item, post, receive

conn = psycopg2.connect(url)
lock = threading.RLock()
with conn.cursor(cursor_factory=RealDictCursor) as cur:
    # Only this rolled-back connection sees the name-selection fixture.
    cur.execute('CREATE TEMP TABLE person_choices(actor_id integer,display_name text,sort_order integer)')
    cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES ('LH browser owner','owner',%s,true) RETURNING id",('b'*64,))
    actor=cur.fetchone()['id']
    cur.execute("INSERT INTO person_choices VALUES (%s,'Michael',1)",(actor,))
    oats = item(cur, 'Oats', 'ingredient', 'LH-BROWSER-OATS', 'received', supplier_lot='SUP-001')
    bulk = item(cur, 'LH browser bulk', 'batch', 'LH-BROWSER-BULK', 'production_output')
    finished = item(cur, 'Granola SS Chocolate Chip 12x10 OZ Case', 'finished', 'LH-BROWSER-FINISHED', 'pack_output', uom='12x10 oz case')
    cur.execute('UPDATE products SET case_size_lb=7.5 WHERE id=%s', (finished[0],))
    receive(cur, oats, 100, 'Supplier <script>bad</script>', bol='BOL-123')
    makes = [post(cur, 'make', '2026-11-01 14:00', [(bulk,50),(oats,-50)])[0] for _ in range(2)]
    packed = post(cur, 'pack', '2026-11-02 14:00', [(finished,75),(bulk,-75)])[0]
    shipped = post(cur, 'ship', '2026-11-03 14:00', [(finished,-15)], customer_name='Audit Customer', order_reference='SO-AUDIT')[0]
    cur.execute("INSERT INTO lots(product_id,lot_code,entry_source) VALUES (%s,'LH-BROWSER-FOUND','found_inventory') RETURNING id", (oats[0],))
    found = (oats[0],cur.fetchone()['id'])
    post(cur,'adjust','2026-11-04 14:00',[(found,10)])


@contextmanager
def transaction():
    with lock, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute('SAVEPOINT browser_request')
        try:
            yield cur
        except Exception:
            cur.execute('ROLLBACK TO SAVEPOINT browser_request')
            raise
        finally:
            cur.execute('RELEASE SAVEPOINT browser_request')


class ConnectionProxy:
    def cursor(self, *args, **kwargs):
        return conn.cursor(*args, **kwargs)
    def commit(self):
        pass
    def rollback(self):
        with conn.cursor() as cur:
            cur.execute('ROLLBACK TO SAVEPOINT browser_connection')


@contextmanager
def connection():
    with lock, conn.cursor() as cur:
        cur.execute('SAVEPOINT browser_connection')
        try:
            yield ConnectionProxy()
        except Exception:
            cur.execute('ROLLBACK TO SAVEPOINT browser_connection')
            raise
        finally:
            cur.execute('RELEASE SAVEPOINT browser_connection')


main.get_db_connection = connection
main.get_transaction = transaction
requests = []


@main.app.middleware('http')
async def observe_real_requests(request, call_next):
    path = request.url.path
    response=await call_next(request)
    if path.startswith('/dashboard/api/') or not path.startswith(('/dashboard/','/auth/','/__','/favicon')):
        requests.append({'path':path, 'key_ok':request.headers.get('x-api-key') == main.DASHBOARD_API_KEY,'status':response.status_code})
    return response


@main.app.get('/__lot_history_test')
def observations():
    return {'oats':oats, 'finished':finished, 'makes':makes, 'packed':packed, 'shipped':shipped, 'requests':requests}


try:
    uvicorn.run(main.app, host='127.0.0.1', port=int(sys.argv[1]), lifespan='off', log_level='error')
finally:
    conn.rollback()
    conn.close()
