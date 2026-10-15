// Layout-only acceptance: actual dashboard assets over local HTTP, explicit API
// fixtures, no model, hosted API, database, or Record requests. The round C page
// runner separately covers actual handlers and PostgreSQL.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
const playwright = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const out = process.env.FIXC_PHONE_OUT || path.join(root, 'docs/validation/fl-assistant-round-c/phone');
await fs.mkdir(out, {recursive: true});
let mode = 0, reply = [], records = 0, errorReply = false;
const actor = {id: 7, name: 'Michael', role: 'floor'};
const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost');
    const json = value => {response.setHeader('Content-Type', 'application/json'); response.end(JSON.stringify(value));};
    if (url.pathname === '/auth/session/config') return json({key_sessions_enabled: !!mode});
    if (url.pathname === '/auth/whoami') return json({key_kind: mode ? 'session' : 'actor', actor});
    if (url.pathname === '/auth/session') return json({key_kind: 'session', actor, session_token: 'fl-phone-fixture'});
    if (url.pathname === '/assistant/session') return json({session_id: 'phone-' + Date.now(), turns: [], drafts: []});
    if (url.pathname === '/assistant/resume') return json({turns: [], drafts: []});
    if (url.pathname === '/assistant/record') {records++; throw Error('Phone runner must stop before Record');}
    if (url.pathname === '/assistant/turn') {
      if (errorReply) { response.statusCode = 500; response.setHeader('Content-Type', 'text/plain'); return response.end('Internal server error'); }
      return json({cards: reply});
    }
    if (url.pathname === '/receipts') return json({receipts: [
      {receipt_number: 'RCV-261015-001', actor, entered_at: '2026-10-15T15:00:00Z', action: 'receive', draft: {product_name: 'Sunflower Seeds', total_lb: 200}},
      {receipt_number: 'MK-261015-002', actor: {id: 8, name: 'Arturo'}, entered_at: '2026-10-15T16:00:00Z', action: 'make', draft: {product_name: 'Classic #9', total_lb: 700}}
    ], has_more: false});
    const local = url.pathname === '/dash/fl-assistant' ? 'dashboard/fl-assistant.html' : url.pathname.replace(/^\//, '');
    assert.ok(local.startsWith('dashboard/'));
    const filename = path.join(root, local), ext = path.extname(filename);
    response.setHeader('Content-Type', {'.html':'text/html', '.js':'text/javascript', '.css':'text/css'}[ext] || 'text/plain');
    response.end(await fs.readFile(filename));
  } catch (error) {response.statusCode = 500; response.end(String(error));}
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const results = [], engines = (process.env.FIXC_PHONE_ENGINES || 'chromium,webkit').split(',');
try {
  for (const engine of engines) {
    const browser = await playwright[engine].launch({headless: true});
    try {
      for (mode of [0, 1]) for (const language of ['en','es']) for (const width of [375,390,393,430]) {
        const context = await browser.newContext({viewport: {width, height: 844}, deviceScaleFactor: 1, isMobile: true, hasTouch: true, timezoneId: 'America/New_York', colorScheme: 'light'});
        await context.addInitScript(language => localStorage.setItem('fl-assistant-language', language), language);
        const page = await context.newPage(), errors = [];
        page.on('pageerror', error => errors.push(error.message));
        const check = async (scenario, screenshot = false) => {
          const overflow = await page.evaluate(() => ({screen: innerWidth, document: document.documentElement.scrollWidth, body: document.body.scrollWidth, extra: [...document.querySelectorAll('body *')].filter(node => {const r = node.getBoundingClientRect(); return r.width && (r.right > innerWidth + 1 || r.left < -1);}).map(node => ({tag: node.tagName, class: node.className, right: node.getBoundingClientRect().right})).slice(0,8)}));
          assert.equal(overflow.document, width, JSON.stringify({scenario, engine, width, overflow}));
          assert.equal(overflow.body, width, JSON.stringify({scenario, engine, width, overflow}));
          assert.deepEqual(errors, []);
          const row = {engine, sessions_enabled: !!mode, language, width, scenario, status: 'passed', scroll_width: overflow.document};
          if (screenshot) {row.screenshot = `${engine}-${mode}-${language}-${width}-${scenario}.png`; if (scenario === 'trace-draft') await page.locator('.assistant-message').last().scrollIntoViewIfNeeded(); await page.screenshot({path: path.join(out,row.screenshot)});}
          results.push(row);
        };
        const focusPhoneField = async locator => {
          assert.ok(await locator.evaluate(node => parseFloat(getComputedStyle(node).fontSize)) >= 16, 'Phone fields need at least 16px text to avoid Safari focus zoom');
          await locator.focus();
          assert.equal(await page.evaluate(() => document.documentElement.scrollWidth), width, 'Focusing a phone field must not widen the page');
          assert.equal(await page.evaluate(() => document.body.scrollWidth), width, 'Focusing a phone field must keep body within the viewport');
          await locator.evaluate(node => node.blur());
        };
        const send = async (text, cards) => {
          reply = cards; const before = await page.locator('.assistant-message').count();
          await page.locator('#message').fill(text); await page.locator('#send').click();
          await page.waitForFunction(() => !document.getElementById('send').disabled);
          assert.ok(await page.locator('.assistant-message').count() > before);
          return page.locator('.assistant-message .card').last();
        };
        await page.goto(origin + '/dash/fl-assistant');
        await page.locator('#sign-in-panel').waitFor({state:'visible'}); await check('sign-in');
        await page.locator('#actor-key').fill('phone-fixture'); await page.locator('#sign-in-form button').click();
        await page.locator('#chat-panel').waitFor({state:'visible'});
        // Like the existing round A header test, exercise every shared tool
        // regardless of the session-mode fixture's default visibility switch.
        await page.evaluate(() => { document.documentElement.dataset.flWriteUi = 'on'; });
        assert.equal(await page.locator('#header-menu').getAttribute('open'), null);
        assert.ok((await page.locator('.topbar').boundingBox()).height <= 100, 'Closed phone header must leave room for chat');
        await check('welcome', true);
        await page.locator('#header-menu summary').click();
        await page.locator('.receipt-lookup').waitFor({state:'visible'});
        await focusPhoneField(page.locator('.receipt-lookup input'));
        const menuLinks = page.locator('#header-menu .receipt-tools>a');
        assert.equal(await menuLinks.count(), 3);
        for (const link of await menuLinks.all()) {
          assert.equal(await link.isVisible(), true, 'All three shared tools must be visible inside the open Menu');
          const box = await link.boundingBox();
          assert.ok(box.width > 0 && box.x >= 0 && box.x + box.width <= width, JSON.stringify(box));
        }
        await check('menu', true); await page.locator('#header-menu summary').click();
        await send('How much classic number nine do we have?', [{kind:'read', language, tool:'inventory_lookup', result:{results:[{product:'Batch Classic Granola #9', total_on_hand:12345, lots:[{lot:'LONG-SUPPLIER-LOT-1234567890-ABCD',qty_on_hand:12345}]}]}}]);
        await check('inventory', true);
        const choices = await send('from Green Supply', [{kind:'choices', language, id:'supplier', resolution_kind:'supplier', result:{ask:'Please clarify: 1) Grain Supply 2) Farm Direct', candidates:[{id:1,name:'Grain Supply'},{id:2,name:'Farm Direct'}]}}]);
        assert.doesNotMatch(await choices.locator('p').first().innerText(), /Grain Supply|Please clarify/);
        await check('choices');
        await send('Made two batches of classic number nine.', [{kind:'draft', id:'make', language, prepared:{action:'make', actor, can_commit:false, blockers:[{code:'LOT_NOT_CONFIRMED'}], draft:{product_name:'Batch Classic Granola #9',batches:2,total_output_lb:700,input_plan:[{product_id:283,product_name:'Sunflower Seeds',lot_id:1,lot_code:'SUNFLOWER-ABCD',quantity_lb:200,confirmed:false}],lot_confirmation_group:{instruction_en:'Type the last 4 characters from the tag on each bag you used.',instruction_es:'Escriba los últimos 4 caracteres de la etiqueta de cada bolsa que usó.',ingredients:[{product_id:283,product_name:'Sunflower Seeds',needed_lb:200,lots:[{lot_id:1,lot_code:'SUNFLOWER-ABCD',quantity_lb:200,product_id:283}]}]}}}}]);
        assert.equal(await page.locator('.assistant-message').last().locator('input[id^=lot-]').count(), 1);
        await focusPhoneField(page.locator('.assistant-message').last().locator('.lot-controls input'));
        await focusPhoneField(page.locator('.assistant-message').last().locator('.lot-controls select'));
        assert.equal(await page.locator('.assistant-message').last().getByRole('button', {name: language === 'en' ? 'Record' : 'Registrar', exact:true}).isDisabled(), true);
        await check('trace-draft', true);
        const order = await send('New order from fake bakery. Five cases of rainbow sprinkles.', [{kind:'draft', id:'order', language, prepared:{action:'create_order', actor,can_commit:true,blockers:[],draft:{product_name:'Rainbow Sprinkles',cases:5,note:'order_id, line ids and the order number are assigned at commit.',happened_at:'2026-10-15T04:00:00Z'}}}]);
        assert.doesNotMatch(await order.innerText(), /order_id|assigned at commit|12:00 AM/); await check('order-draft');
        const today = await send(language === 'en' ? "Open today's entries. Is everything there correct?" : '¿Qué registré hoy?', []);
        assert.match(await today.innerText(), language === 'en' ? /You recorded 1 entry today; the team recorded 2/ : /Registraste 1 entrada hoy; el equipo registró 2/);
        assert.match(await today.locator('.today-entry').first().innerText(), /MK-261015-002/); await check('today');
        errorReply = true;
        const error = await send('Check stock', []); errorReply = false;
        assert.match(await error.innerText(), language === 'en' ? /Factory Ledger is temporarily unavailable/ : /Factory Ledger no está disponible temporalmente/);
        assert.doesNotMatch(await error.innerText(), /unreadable|Internal server error/); await check('service-unavailable');
        await context.close();
        console.log(`${engine}, sessions ${mode}, ${language}, ${width}px: 9 checks passed`);
      }
    } finally {await browser.close();}
  }
  assert.equal(records,0);
} finally {
  server.close();
  await fs.writeFile(path.join(out,'results.json'), JSON.stringify({runner:'tests/visual/run-fl-assistant-round-c-phone.mjs',layout_only:true,api_fixtures:true,local_only:true,records,results},null,2)+'\n');
}
console.log(`${results.length} phone checks passed; ${records} Record requests.`);
