// Real browser -> HTTP -> TestClient -> real branch handlers/PostgreSQL.
// Only OpenAI's tool selection is stubbed. Creates and drops its own local DBs.
// This validates the B data contract; Session A owns the dashboard renderer.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline';
import { spawn, execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const python = process.env.FIXB_PYTHON || path.join(root, '.venv-test/bin/python');
const pg = process.env.FIXB_PG_BIN || '/opt/homebrew/opt/postgresql@17/bin';
const output = process.env.FIXB_BROWSER_OUT || '/private/tmp/fl_fixB2_browser_contract.json';
const localEnv = { ...process.env };
for (const name of ['PGHOSTADDR', 'PGSERVICE', 'PGSERVICEFILE', 'PGOPTIONS']) delete localEnv[name];
const bridgeSource = String.raw`
import json,os,sys,hashlib,logging
from contextlib import contextmanager
from pathlib import Path
from http.server import BaseHTTPRequestHandler,HTTPServer
from urllib.parse import urlsplit
from uuid import uuid4
import psycopg2
from psycopg2.extras import RealDictCursor
ROOT=Path.cwd();sys.path.insert(0,str(ROOT))
url=os.environ['TEST_DATABASE_URL'];parsed=urlsplit(url)
assert parsed.hostname=='127.0.0.1' and parsed.path.startswith('/fl_fixB2_browser_')
os.environ.update(DATABASE_URL=url,API_KEY='fixb-local-master',DASHBOARD_API_KEY='fixb-local-read',ASSISTANT_ENABLED='1')
os.environ.pop('OPENAI_API_KEY',None)
logging.disable(logging.CRITICAL)
@contextmanager
def connection():
 conn=psycopg2.connect(url)
 try:
  with conn:yield conn
 finally:conn.close()
with connection() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
 text=(ROOT/'migrations/061_exceptions_tables.sql').read_text();cur.execute(text[text.index('CREATE TABLE IF NOT EXISTS correction_reasons'):text.index('-- 3. transactions.reason_code')])
 key='fixb-local-floor';cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES('Round B Worker','floor',%s,true) RETURNING id",(hashlib.sha256(key.encode()).hexdigest(),));actor=cur.fetchone()['id']
 products={}
 for name,kind,code,pid in [('Sunflower Seeds','ingredient','11010',283),('Rolled Oats','ingredient','11020',284),('Batch Classic Granola #9','batch','90002',285),('Granola Classic 25 LB','finished','70013',286),('Granola Fruit Nut Batch','batch','90008',287),('Batch Classic Chocolate Chip Granola #9','batch','90001',288)]:
  cur.execute("INSERT INTO products(id,name,odoo_code,type,uom,default_batch_lb,case_size_lb,active) VALUES(%s,%s,%s,%s,'lb',10,5,true)",(pid,name,code,kind));products[name]=pid
 cur.execute('UPDATE products SET parent_batch_product_id=285 WHERE id=286')
 cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES(285,283,4),(285,284,6)')
 lots={}
 for pid,code in [(283,'SUNFLOWER-ABCD'),(284,'OATS-EFGH'),(285,'BATCH-B001'),(286,'FINISHED-F001')]:
  cur.execute('INSERT INTO lots(product_id,lot_code) VALUES(%s,%s) RETURNING id',(pid,code));lid=cur.fetchone()['id'];lots[str(pid)]={'id':lid,'code':code}
  cur.execute("INSERT INTO transactions(type,notes) VALUES('receive','Round B local fixture') RETURNING id");tid=cur.fetchone()['id'];cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,100)',(tid,pid,lid))
 cur.execute("INSERT INTO suppliers(name,short_code) VALUES('Grain Supply','GRAN') RETURNING id");supplier=cur.fetchone()['id']
 customers={};orders=[]
 for name in ['Sunshine Granola','Setton Farms','UNIPRO']:
  cur.execute('INSERT INTO customers(name,active) VALUES(%s,true) RETURNING id',(name,));cid=cur.fetchone()['id'];customers[name]=cid
  cur.execute("INSERT INTO sales_orders(customer_id,order_number,status,requested_ship_date) VALUES(%s,'','confirmed','2026-10-16') RETURNING id,order_number",(cid,));order=dict(cur.fetchone());orders.append(order)
  cur.execute('INSERT INTO sales_order_lines(sales_order_id,product_id,quantity_lb) VALUES(%s,286,10)',(order['id'],))
 fixture={'products':products,'lots':lots,'supplier':supplier,'customers':customers,'orders':orders}
import main,fl_assistant
from fastapi.testclient import TestClient
main.get_db_connection=connection;main._reset_actor_cache();client=TestClient(main.app)
queue=[]
async def fake(path,*,body=None,file=None):
 assert path=='responses' and body['tool_choice']=='required' and body['parallel_tool_calls'] is False
 assert body['store'] is False and not any('commit' in t['name'] for t in body['tools'])
 assert queue,'Model fixture exhausted'
 name,args=queue.pop(0)
 schema=next(t['parameters'] for t in body['tools'] if t['name']==name)
 args={field:args.get(field) for field in schema['properties']}
 return {'status':'completed','output':[{'type':'function_call','call_id':'call_'+uuid4().hex,'name':name,'arguments':json.dumps(args)}]}
fl_assistant.openai=fake
class Handler(BaseHTTPRequestHandler):
 def log_message(self,*args):pass
 def do_GET(self):self.handle_request()
 def do_POST(self):self.handle_request()
 def handle_request(self):
  try:
   body=json.loads(self.rfile.read(int(self.headers.get('Content-Length','0'))) or 'null')
   if self.path=='/':status,data,kind=200,b'<html><body>Round B backend contract browser fixture</body></html>','text/html'
   elif self.path=='/test/fixture':status,data,kind=200,json.dumps(fixture).encode(),'application/json'
   elif self.path=='/test/model':
    assert not queue,'Unconsumed tool calls';queue.extend(body);status,data,kind=200,b'{}','application/json'
   elif self.path=='/test/verify':
    with connection() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
     cur.execute("SELECT COALESCE(sum(quantity_lb),0) AS quantity FROM ledger_current_transaction_lines WHERE lot_id=%s",(lots['283']['id'],));stock=cur.fetchone()['quantity']
     cur.execute("SELECT kind,detail FROM exceptions WHERE kind='SHIFT_DISCREPANCY'");exceptions=[dict(r) for r in cur.fetchall()]
    status,data,kind=200,json.dumps({'sunflower_original_stock':float(stock),'exceptions':exceptions},default=str).encode(),'application/json'
   else:
    response=client.request(self.command,self.path,json=body,headers={'X-API-Key':self.headers.get('X-API-Key','')});status,data,kind=response.status_code,response.content,'application/json'
  except Exception as exc:
   import traceback;traceback.print_exc(file=sys.stderr)
   status,data,kind=500,json.dumps({'fixture_error':type(exc).__name__}).encode(),'application/json'
  self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
server=HTTPServer(('127.0.0.1',0),Handler);print(json.dumps({'port':server.server_port}),flush=True)
try:server.serve_forever()
finally:client.close();server.server_close()
`;
const evidence = { runner: 'tests/visual/run-assistant-round-b.mjs', local_only: true, model_tool_selection_stubbed: true, renderer_test: false, modes: [] };
const browser = await chromium.launch({ headless: true, ...(process.env.PLAYWRIGHT_CHANNEL ? { channel: process.env.PLAYWRIGHT_CHANNEL } : {}) });
try {
  for (const mode of [0, 1]) {
    const db = `fl_fixB2_browser_${process.pid}_${mode}`;
    let child;
    const run = (command, args) => execFileSync(path.join(pg, command), args, { cwd: root, env: localEnv, stdio: 'pipe' });
    run('createdb', ['-h', '127.0.0.1', db]);
    try {
      run('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-c', 'CREATE EXTENSION pg_trgm']);
      run('psql', ['-h', '127.0.0.1', '-d', db, '-v', 'ON_ERROR_STOP=1', '-q', '-f', path.join(root, 'tests/schema/schema.sql')]);
      child = spawn(python, ['-u', '-c', bridgeSource], { cwd: root, env: { ...localEnv, TEST_DATABASE_URL: `postgresql://127.0.0.1:5432/${db}`, KEY_SESSIONS_ENABLED: String(mode) }, stdio: ['ignore', 'pipe', 'pipe'] });
      let diagnostic = '';
      child.stderr.on('data', chunk => { diagnostic += chunk; process.stderr.write(chunk); });
      const lines = readline.createInterface({ input: child.stdout });
      const ready = await Promise.race([
        new Promise((resolve, reject) => { lines.once('line', line => { try { resolve(JSON.parse(line)); } catch { reject(new Error('Invalid fixture startup output')); } }); child.once('exit', code => reject(new Error(`Fixture exited ${code}: ${diagnostic}`))); }),
        new Promise((_, reject) => { const timer = setTimeout(() => reject(new Error('Fixture startup timed out')), 30000); timer.unref(); }),
      ]);
      const context = await browser.newContext();
      const page = await context.newPage();
      await page.goto(`http://127.0.0.1:${ready.port}/`);
      let key = 'fixb-local-floor';
      const checks = [];
      const api = async (url, body = undefined, expected = 200) => {
        const response = await page.evaluate(async ({ url, body, key }) => { const r = await fetch(url, { method: body === undefined ? 'GET' : 'POST', headers: { 'Content-Type': 'application/json', 'X-API-Key': key }, body: body === undefined ? undefined : JSON.stringify(body) }); return { status: r.status, data: await r.json() }; }, { url, body, key });
        assert.equal(response.status, expected, `${url}: ${JSON.stringify(response.data)}`);
        return response.data;
      };
      const fixture = await api('/test/fixture');
      assert.equal((await api('/auth/session/config')).key_sessions_enabled, Boolean(mode));
      if (mode) { key = (await api('/auth/session', { actor_key: key })).session_token; assert.ok(key.startsWith('fls_')); }
      async function turn(text, calls, extra = {}, session = null) {
        await api('/test/model', calls);
        const sid = session || (await api('/assistant/session', {})).session_id;
        const response = await api('/assistant/turn', { session_id: sid, turn_id: crypto.randomUUID(), text, ...extra });
        assert.ok(!/payload_hash|wt_/.test(JSON.stringify(response)), 'Ticket secrets must stay server-side');
        return { ...response, sid };
      }
      const resolve = (kind, query, context = {}) => ['resolve', { kind, query, context }];
      for (const [text, lang] of [['How much Sunflower Seeds do we have?', 'en'], ['¿Cuánto tenemos de semillas de girasol?', 'es']]) {
        const response = await turn(text, [['inventory_lookup', { q: lang === 'es' ? 'semillas de girasol' : 'Sunflower Seeds' }]]);
        const card = response.cards[0]; assert.equal(card.kind, 'read'); assert.equal(card.language, lang); assert.equal(card.rows.length, 1); assert.match(card.summary, /Sunflower Seeds: 100 lb/); assert.equal(card.message, card.summary);
      }
      checks.push('EN/ES inventory answers start with source-backed summary and structured rows');
      for (const [text, filter, lang, count] of [['What orders are open?', null, 'en', 3], ['Show open orders for Sunshine', 'Sunshine', 'en', 1], ['¿Qué pedidos abiertos hay?', null, 'es', 3], ['Muestra pedidos abiertos para Sunshine', 'Sunshine', 'es', 1]]) {
        const card = (await turn(text, [['open_orders', { customer: filter, product: null }]])).cards[0];
        assert.equal(card.kind, 'read'); assert.equal(card.language, lang); assert.equal(card.rows.length, count);
        for (const row of card.rows) for (const field of ['customer', 'order_number', 'items', 'ship_date', 'status']) assert.ok(Object.hasOwn(row, field));
        assert.match(card.summary, lang === 'es' ? /^Hay / : /^There (?:is|are) /);
      }
      const productFiltered = (await turn('Show open orders for product 70013', [['open_orders', { customer: null, product: '70013' }]])).cards[0];
      assert.equal(productFiltered.rows.length, 3);
      checks.push('EN/ES open orders with customer and SKU filters use real read routes');
      for (const query of ['Sunshine', 'Setton', 'UNIPRO']) assert.equal((await api('/resolve', { kind: 'customer', query })).outcome, 'match');
      for (const [query, pid] of [['semillas de girasol', 283], ['283', 283], ['70013', 286]]) assert.equal((await api('/resolve', { kind: 'product', query })).match.id, pid);
      checks.push('Real customers, Spanish ingredient alias, product number and SKU resolve');
      const unmatchedTurn = await turn('Recibí 10 lb de ingrediente desconocido', [resolve('product', 'ingrediente desconocido')]);
      const unmatched = unmatchedTurn.cards[0];
      assert.notEqual(unmatched.kind, 'choices'); assert.equal(unmatched.language, 'es'); assert.match(unmatched.message, /No reconocí/);
      const recovered = (await turn('283', [resolve('product', '283'), ['prepare_receive', { product_id: 283 }]], {}, unmatchedTurn.sid)).cards[0];
      assert.equal(recovered.kind, 'missing_facts');
      checks.push('Unknown wording never produces an empty choice card; a numeric follow-up can resolve');
      for (const [text, lang] of [['Received 10 lb of Sunflower Seeds', 'en'], ['Recibí 10 lb de semillas de girasol', 'es']]) {
        const response = await turn(text, [resolve('product', lang === 'es' ? 'semillas de girasol' : 'Sunflower Seeds'), ['prepare_receive', { product_id: 283 }]]);
        const card = response.cards[0]; assert.equal(card.kind, 'missing_facts'); assert.ok(card.fields.includes('supplier_id')); assert.equal(card.known_facts.quantity_lb, 10); assert.ok(card.supplier_choices.some(s => s.id === fixture.supplier)); assert.ok(card.choice_id); assert.ok(card.missing_details.some(d => d.label === (lang === 'es' ? 'proveedor' : 'supplier'))); assert.ok(!/SUPPLIER_REQUIRED|\/resolve/.test(card.message));
        const completed = await turn('', [['prepare_receive', { product_id: 283, supplier_id: fixture.supplier, cases: 1, case_size_lb: 10, bol_reference: 'FIXB-RECEIVE-' + lang, supplier_lot_code: 'SUP-' + lang }]], { choice_id: card.choice_id, selected_id: fixture.supplier }, response.sid);
        assert.equal(completed.cards[0].kind, 'draft'); assert.equal(completed.cards[0].prepared.draft.total_lb, 10); assert.equal(completed.cards[0].language, lang);
      }
      checks.push('Receive keeps 10 lb and offers real supplier choices with EN/ES labels');
      for (const [text, lang] of [['How much is in lot SUNFLOWER-ABCD?', 'en'], ['¿Cuánto hay en el lote SUNFLOWER-ABCD?', 'es']]) {
        const card = (await turn(text, [['lot_lookup', { lot_code: 'SUNFLOWER-ABCD' }]])).cards[0]; assert.equal(card.kind, 'read'); assert.equal(card.language, lang); assert.match(card.summary, /100 lb/); assert.equal(card.rows[0].lot_code, 'SUNFLOWER-ABCD');
      }
      checks.push('EN/ES lot questions return plain quantities from lot lookup');
      for (const explicit of [true, false]) {
        const args = { source_product_id: 285, target_product_id: 286, cases: 10, ...(explicit ? { lot_allocations: [{ lot_id: fixture.lots['285'].id, quantity_lb: 50 }] } : {}) };
        const response = await turn(`Packed 10 cases of Granola Classic 25 LB${explicit ? ' from batch BATCH-B001' : ''}`, [resolve('product', 'Granola Classic 25 LB', { action: 'pack' }), ['packing_context', { target_product_id: 286, lot_code: explicit ? 'BATCH-B001' : null }], ['prepare_pack', args]]);
        assert.equal(response.cards[0].kind, 'draft'); assert.equal(response.cards[0].prepared.action, 'pack'); assert.equal(response.cards[0].prepared.draft.source_product_id, 285);
      }
      checks.push('Packing reaches filled drafts from explicit batch lot and finished product parent');
      const matching = await turn('Made 1 batch of Classic #9', [resolve('product', 'Classic #9', { action: 'make' })]);
      const choices = matching.cards[0]; assert.equal(choices.kind, 'choices');
      assert.ok(choices.result.candidates.length > 0); assert.ok(choices.result.candidates.every(c => c.id === 285));
      const make = await turn('', [['prepare_make', { product_id: 285, batches: 1 }]], { choice_id: choices.id, selected_id: 285 }, matching.sid);
      assert.equal(make.cards.length, 1); const card = make.cards[0]; assert.equal(card.kind, 'draft');
      assert.ok(!card.prepared.blockers.some(b => b.code === 'SKU_CONFIRMATION_REQUIRED'));
      const group = card.prepared.draft.lot_confirmation_group;
      assert.equal(group.kind, 'lot_confirmation'); assert.equal(group.ingredients.length, 2);
      assert.equal(group.instruction, 'Type the last 4 characters from the tag on each bag you used.');
      for (const ingredient of group.ingredients) { assert.equal(ingredient.candidates.length, 1); assert.equal(ingredient.suggested_lots[0].confirmed, false); assert.equal(ingredient.different_lot.label_es, 'Otro lote'); }
      assert.equal(card.prepared.can_commit, false);
      const empty = await api('/assistant/record', { draft_id: card.id }, 422); assert.match(JSON.stringify(empty), /LOT_NOT_CONFIRMED/);
      const bad = group.ingredients.map(i => ({ lot_id: i.suggested_lots[0].lot_id, method: 'last4', value: 'NOPE' }));
      const refused = await api('/assistant/record', { draft_id: card.id, lot_confirmations: bad }, 422); assert.match(JSON.stringify(refused), /LOT_CONFIRMATION_MISMATCH/);
      const confirmations = group.ingredients.map(i => ({ lot_id: i.suggested_lots[0].lot_id, method: i.product_id === 283 ? 'different_lot' : 'last4', value: i.product_id === 283 ? 'PHYSICAL-TAG-7654' : i.suggested_lots[0].lot_code.slice(-4) }));
      const receipt = await api('/assistant/record', { draft_id: card.id, lot_confirmations: confirmations }); assert.equal(receipt.kind, 'receipt'); assert.equal(receipt.result.lot_exceptions.length, 1);
      const proof = await api('/test/verify'); assert.equal(proof.sunflower_original_stock, 100); assert.equal(proof.exceptions.length, 1); assert.equal(proof.exceptions[0].detail.physical_tag, 'PHYSICAL-TAG-7654');
      checks.push('One filled make draft groups all ingredients; single-lot tags required; wrong tag rejected; Different lot posts traceable exception without substituting FIFO');
      const spanishMake = (await turn('Hicimos 1 tanda del producto 90002', [resolve('product', '90002', { action: 'make' }), ['prepare_make', { product_id: 285, batches: 1 }]])).cards[0];
      assert.equal(spanishMake.kind, 'draft'); assert.equal(spanishMake.language, 'es');
      assert.equal(spanishMake.prepared.draft.lot_confirmation_group.instruction, 'Escriba los últimos 4 caracteres de la etiqueta de cada bolsa que usó.');
      for (const [text, lang] of [['Tell me a joke', 'en'], ['¿Cuánto es siete más cinco?', 'es']]) { const card = (await turn(text, [['unsupported_request', {}]])).cards[0]; assert.equal(card.language, lang); assert.match(card.message, /Factory Ledger/); assert.ok(!/ChatGPT|try again/i.test(card.message)); }
      checks.push('EN/ES out-of-scope answers use Factory Ledger wording');
      if (mode) await api('/auth/session/logout', {});
      evidence.modes.push({ sessions_enabled: Boolean(mode), checks_passed: checks.length, checks });
      await context.close();
      console.log(`Round B real backend browser: sessions ${mode ? 'on' : 'off'}, ${checks.length} contract groups passed.`);
    } finally {
      if (child && child.exitCode === null) { child.kill('SIGTERM'); await new Promise(resolve => child.once('exit', resolve)); }
      run('dropdb', ['-h', '127.0.0.1', '--if-exists', '--force', db]);
    }
  }
} finally { await browser.close(); }
evidence.databases_dropped = true;
await fs.writeFile(output, JSON.stringify(evidence, null, 2) + '\n');
