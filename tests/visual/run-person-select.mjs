// Real-browser checks for owner-approved browsers + "who is entering?" (091).
// The REAL dashboard files are served from disk; every FL API call is answered by
// an in-memory synthetic FL below (no network, no database, no staging/production).
// Grants come from tests/visual/fixtures/person-select-whoami.json, generated
// from the real permissions.py matrix.
//
//   PLAYWRIGHT_MODULE=/path/to/node_modules/playwright/index.mjs node tests/visual/run-person-select.mjs
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startStaticServer } from './lib/server.mjs';

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '../../dashboard');
const out = process.env.PERSON_SELECT_OUT || path.resolve(here, '../../.venv-test/person-select-visual');
await fs.mkdir(out, { recursive: true });
const WHOAMI = JSON.parse(await fs.readFile(path.join(here, 'fixtures/person-select-whoami.json'), 'utf8'));
const PEOPLE = [
  { id: 1, display_name: 'Michael', role_label: 'Owner', available: true },
  { id: 3, display_name: 'Luz', role_label: 'Office', available: true },
  { id: 2, display_name: 'Arturo', role_label: 'Floor Lead', available: true },
  { id: 4, display_name: 'Miriam', role_label: 'Office', available: true },
];
const TOKEN = 'flb_' + 'T'.repeat(43);
const server = await startStaticServer(root);
const base = server.origin;
const browser = await chromium.launch({ channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome', headless: true });
const results = [];

function fakeFL({ mode = true, available = true, problem = null, approved = true, statusDelay = 0, configFail = null, legacyStatus = 200, configDelay = 0 } = {}) {
  const state = { approved, calls: [], mode, available, configCalls: 0, foreign: [] };
  const json = (route, status, body) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
  async function handle(route) {
    const request = route.request(), url = new URL(request.url()), p = url.pathname;
    if (url.origin !== base) {
      const fh = request.headers();
      state.foreign.push({ origin: url.origin, path: url.pathname, key: fh['x-api-key'] || null, person: fh['x-fl-person'] || null });
      return route.abort();
    }
    if (/\.(html|js|css|png|svg|webp|ico|json|woff2?)$/.test(p) || p === '/') return route.continue();
    const h = request.headers(), key = h['x-api-key'] || null, person = h['x-fl-person'] || null;
    state.calls.push({ path: p, method: request.method(), key, person });
    if (p === '/auth/session/config' && configFail && (configFail === 'always' || ++state.configCalls === 1)) {
      if (configFail !== 'always') state.configFailed = true;
      return json(route, 503, { detail: 'Service unavailable' });
    }
    // An OLD legacy key/PIN session answers as its person, so a restore would show.
    if (key && key.startsWith('fls_')) return legacyStatus === 200
      ? json(route, 200, { actor: { id: 3, name: 'Luz', role: 'office' }, key_kind: 'session', permissions: WHOAMI['3'].permissions })
      : json(route, legacyStatus, { detail: 'Synthetic old-session failure' });
    if (p === '/auth/session/config' && configDelay) await new Promise(r => setTimeout(r, configDelay));
    if (p === '/auth/session/config') return json(route, 200, { key_sessions_enabled: true, pin_sign_in: true, idle_minutes: 180, pin_session_hours: 12,
      person_select_enabled: state.mode, person_select_available: state.available, person_select_problem: state.available ? null : (problem || 'PERSON_SELECT_UNAVAILABLE') });
    if (p === '/auth/dashboard-key') return json(route, 200, { dashboard_key: 'fixture-dashboard-key' });
    if (p === '/auth/browsers/request') return json(route, 201, { browser_token: TOKEN, pairing_code: 'K7QM-2XPA', pairing_expires_at: new Date(Date.now() + 600e3).toISOString(), pairing_minutes: 10 });
    if (p === '/auth/browsers/status') {
      if (statusDelay) await new Promise(r => setTimeout(r, statusDelay));
      if (key !== TOKEN) return json(route, 401, { detail: { error_code: 'BROWSER_NOT_APPROVED' } });
      return json(route, 200, state.approved ? { status: 'approved', label: 'Tablet', people: PEOPLE } : { status: 'pending', pairing_expires_at: null });
    }
    // Every other FL route: a browser token must be approved; writes need a person.
    if (key === TOKEN && !state.approved) return json(route, 401, { detail: { error_code: 'BROWSER_NOT_APPROVED' } });
    const write = request.method() !== 'GET';
    if (write && key !== TOKEN) return json(route, 403, { detail: { error_code: 'PERSON_REQUIRED' } });
    if (write && !person) return json(route, 403, { detail: { error_code: 'PERSON_REQUIRED' } });
    if (p === '/auth/whoami') return person ? json(route, 200, WHOAMI[person]) : json(route, 200, { actor: null, key_kind: 'browser', permissions: {} });
    if (p.startsWith('/receipts/')) return json(route, 200, { receipt_number: 'RCV-261103-001', action: 'receive', status: 'committed', effective_status: 'posted',
      draft: { product_name: 'Oats' }, response: { lot_code: 'L-1' }, transactions: [{ id: 5, effective_status: 'posted', allowed_actions: person === '2' ? ['void'] : [] }] });
    if (p === '/receipts') return json(route, 200, { receipts: [], has_more: false });
    if (p === '/assistant/resume') return json(route, 404, { detail: { error_code: 'SESSION_NOT_FOUND' } });
    if (p === '/assistant/session') return json(route, 200, { session_id: 'chat-' + person, actor: WHOAMI[person]?.actor });
    if (p === '/reports/shift-summary') return json(route, 200, { title: 'Shift', receipts_today: 0, sections: [], date: url.searchParams.get('date'), confirmation: { can_confirm: false, reason: 'Nothing to confirm' } });
    if (p === '/correction-reasons') return json(route, 200, { reasons: [] });
    return json(route, 200, request.method() === 'GET' ? {} : { success: true });
  }
  return { state, handle };
}

async function scenario(name, fn, { viewport = { width: 390, height: 844 } } = {}) {
  const context = await browser.newContext({ viewport });
  const errors = [];
  try {
    await fn({ context, errors, newPage: async fl => {
      const page = await context.newPage();
      page.on('pageerror', e => errors.push(e.message));
      await page.route('**/*', fl.handle);
      return page;
    } });
    results.push({ name, ok: true });
    console.log('ok -', name);
  } catch (error) {
    results.push({ name, ok: false, error: String(error.stack || error) });
    console.log('not ok -', name, '\n', error.stack || error);
  } finally { await context.close(); }
}
const seed = (page, values) => page.addInitScript(v => { if (!sessionStorage.getItem('__seeded')) { for (const [k, x] of Object.entries(v)) localStorage.setItem(k, x); sessionStorage.setItem('__seeded', '1'); } }, values);
const savedPerson = id => { const p = PEOPLE.find(x => x.id === id); return JSON.stringify({ id, display_name: p.display_name, role_label: p.role_label, browser: TOKEN.slice(4, 16) }); };
const visible = (page, selector) => page.locator(selector).filter({ visible: true }).count();

await scenario('selector OFF: no chip, legacy personal sign-in unchanged (compatibility)', async ({ newPage }) => {
  const fl = fakeFL({ mode: false });
  const page = await newPage(fl);
  await page.goto(base + '/office.html');
  await page.evaluate(() => FLDashboard.ready);
  await page.waitForTimeout(300);
  assert.equal(await visible(page, '#fl-person-chip'), 0);
  assert.equal(await visible(page, '.fl-person-overlay'), 0);
  assert.equal(await visible(page, '#sign-in-form'), 1);
  assert.ok(!fl.state.calls.some(c => c.path.startsWith('/auth/browsers')));
});

await scenario('untrusted Dashboard: approval code, then the four names appear by themselves', async ({ newPage }) => {
  const fl = fakeFL({ approved: false });
  const page = await newPage(fl);
  await page.goto(base + '/index.html');
  const ask = page.getByRole('button', { name: 'Ask Michael to approve this browser' });
  await ask.waitFor();
  await page.screenshot({ path: out + '/untrusted-390.png' });
  await ask.click();
  await page.locator('.fl-person-code', { hasText: 'K7QM-2XPA' }).waitFor();
  await page.screenshot({ path: out + '/pairing-code-390.png' });
  fl.state.approved = true;
  await page.getByRole('heading', { name: 'Who is entering?' }).waitFor({ timeout: 8000 });
  assert.deepEqual(await page.locator('.fl-person-option-name').allTextContents(), ['Michael', 'Luz', 'Arturo', 'Miriam']);
  await page.screenshot({ path: out + '/names-390.png' });
});

await scenario('Dashboard opens with the names shown (no extra tap); choice persists across reload and a new tab', async ({ context, newPage }) => {
  const fl = fakeFL();
  const page = await newPage(fl);
  await seed(page, { 'fl-browser-token': TOKEN });
  let navigations = 0; page.on('framenavigated', f => { if (f === page.mainFrame()) navigations += 1; });
  await page.goto(base + '/index.html');
  await page.getByRole('heading', { name: 'Who is entering?' }).waitFor();
  await page.getByRole('button', { name: /Arturo/ }).click();
  await page.locator('#fl-person-chip', { hasText: 'Entering as: Arturo (Floor Lead)' }).waitFor();
  assert.equal(await visible(page, '.fl-person-overlay'), 0);
  await page.screenshot({ path: out + '/chip-390.png' });
  assert.equal(navigations, 1);
  await page.reload();
  await page.locator('#fl-person-chip', { hasText: 'Entering as: Arturo (Floor Lead)' }).waitFor();
  assert.equal(await visible(page, '.fl-person-overlay'), 0);
  // A new tab in the same running browser keeps the name (this is NOT a restart;
  // the persistent-profile scenario below closes and relaunches the browser).
  const again = await newPage(fl);
  await again.goto(base + '/index.html');
  await again.locator('#fl-person-chip', { hasText: 'Arturo' }).waitFor();
});

await scenario('Office: selected Luz gets FL grants via /auth/whoami and sees the office actions, no key/PIN offers', async ({ newPage }) => {
  const fl = fakeFL();
  const page = await newPage(fl);
  await seed(page, { 'fl-browser-token': TOKEN, 'fl-person': savedPerson(3) });
  await page.goto(base + '/office.html');
  await page.locator('#office-action').waitFor();
  assert.equal(await page.getByText('Sign in with permission for this action.').count(), 0);
  const who = fl.state.calls.filter(c => c.path === '/auth/whoami');
  assert.ok(who.length >= 1 && who.every(c => c.key === TOKEN && c.person === '3'), JSON.stringify(who));
  assert.equal(await visible(page, '#sign-in-form'), 0);
  assert.equal(await visible(page, '.fl-pin-offer, .fl-pin-toggle, #actor-key'), 0);
  const options = await page.locator('#office-action option').allTextContents();
  assert.ok(options.length >= 3, options.join(','));
  await page.screenshot({ path: out + '/office-luz-390.png', fullPage: true });
  // Change user -> Miriam: grants are fetched again for Miriam; nothing as Luz afterwards.
  await page.getByRole('button', { name: 'Change user' }).click();
  await page.getByRole('button', { name: /Miriam/ }).click();
  await page.locator('#fl-person-chip', { hasText: 'Miriam' }).waitFor();
  await page.waitForFunction(() => true);
  await page.waitForTimeout(300);
  const after = fl.state.calls.filter(c => c.path === '/auth/whoami').at(-1);
  assert.equal(after.person, '4');
  await page.locator('#office-action').waitFor();
});

await scenario('Receipt: selected Arturo is offered the FL-granted void; no legacy sign-in', async ({ newPage }) => {
  const fl = fakeFL();
  const page = await newPage(fl);
  await seed(page, { 'fl-browser-token': TOKEN, 'fl-person': savedPerson(2) });
  await page.goto(base + '/receipt.html?number=RCV-261103-001');
  await page.getByRole('button', { name: 'Prepare void for entry #5' }).waitFor();
  assert.ok(fl.state.calls.some(c => c.path === '/receipts/RCV-261103-001' && c.person === '2'));
  assert.equal(await visible(page, '#sign-in-form'), 0);
  await page.screenshot({ path: out + '/receipt-arturo-390.png', fullPage: true });
});

await scenario('R10 Assistant: saved Luz + slow status opens ONCE, chat starts as Luz, no PIN/key panel', async ({ newPage }) => {
  const fl = fakeFL({ statusDelay: 200 });
  const page = await newPage(fl);
  await seed(page, { 'fl-browser-token': TOKEN, 'fl-person': savedPerson(3), 'fl-pin-session-token': 'fls_' + 'x'.repeat(43) });
  let navigations = 0; page.on('framenavigated', f => { if (f === page.mainFrame()) navigations += 1; });
  await page.goto(base + '/fl-assistant.html');
  await page.waitForTimeout(1500);
  assert.equal(navigations, 1, 'navigations=' + navigations);
  const sessions = fl.state.calls.filter(c => c.path === '/assistant/session');
  assert.equal(sessions.length, 1);
  assert.equal(sessions[0].person, '3');
  assert.equal(await visible(page, '#sign-in-panel'), 0);
  assert.equal(await visible(page, '.fl-pin-offer, .fl-pin-toggle'), 0);
  assert.ok(!fl.state.calls.some(c => c.key && c.key.startsWith('fls_')), 'old PIN session token never sent');
  // A real Change user drops Luz's conversation by reloading ONCE (the click
  // itself reloads); choosing Arturo on the fresh page starts his chat in place.
  const reloaded = page.waitForEvent('framenavigated');
  await page.getByRole('button', { name: 'Change user' }).click();
  await reloaded;
  await page.getByRole('button', { name: /Arturo/ }).click();
  await page.waitForTimeout(1200);
  assert.equal(navigations, 2, 'navigations=' + navigations);
  assert.equal(fl.state.calls.filter(c => c.path === '/assistant/session').at(-1).person, '2');
  assert.match(await page.locator('#identity').innerText(), /Arturo/);
});

await scenario('R6+R10 Assistant: Change user whose storage clear FAILS stays blocked, no reload, never back to Luz', async ({ newPage }) => {
  const fl = fakeFL();
  const page = await newPage(fl);
  await seed(page, { 'fl-browser-token': TOKEN, 'fl-person': savedPerson(3) });
  let navigations = 0; page.on('framenavigated', f => { if (f === page.mainFrame()) navigations += 1; });
  await page.goto(base + '/fl-assistant.html');
  await page.locator('#chat-panel').waitFor();
  assert.match(await page.locator('#identity').innerText(), /Luz/);
  await page.evaluate(() => {
    const original = Storage.prototype.removeItem;
    Storage.prototype.removeItem = function (key) {
      if (key === 'fl-person') throw new DOMException('Synthetic storage clear failure', 'SecurityError');
      return original.call(this, key);
    };
  });
  const before = fl.state.calls.length;
  await page.getByRole('button', { name: 'Change user', exact: true }).click();
  await page.waitForTimeout(1500);
  assert.equal(navigations, 1, 'no automatic reload that would restore the uncleared name');
  assert.doesNotMatch(await page.locator('.fl-person-name').innerText(), /Entering as/);
  assert.doesNotMatch(await page.locator('#identity').innerText(), /Luz/);
  assert.equal(await page.locator('#chat-panel').isVisible(), false);
  const sent = await page.evaluate(() => FLDashboard.person().headers({ write: true }) && 'sent').catch(e => e.message);
  assert.match(String(sent), /not available|Nothing/);
  assert.ok(!fl.state.calls.slice(before).some(c => c.person), 'nothing sent as anyone after the failure');
  await page.screenshot({ path: out + '/assistant-storage-failure-390.png' });
});

await scenario('Assistant: first name chosen on the page starts the chat in place (no reload)', async ({ newPage }) => {
  const fl = fakeFL();
  const page = await newPage(fl);
  await seed(page, { 'fl-browser-token': TOKEN });
  let navigations = 0; page.on('framenavigated', f => { if (f === page.mainFrame()) navigations += 1; });
  await page.goto(base + '/fl-assistant.html');
  await page.getByRole('button', { name: /Miriam/ }).click();
  await page.waitForTimeout(800);
  assert.equal(navigations, 1);
  assert.equal(fl.state.calls.filter(c => c.path === '/assistant/session').at(-1)?.person, '4');
});

await scenario('cross-tab: Change user in one tab updates the other tab', async ({ newPage }) => {
  const fl = fakeFL();
  const a = await newPage(fl);
  await seed(a, { 'fl-browser-token': TOKEN, 'fl-person': savedPerson(2) });
  await a.goto(base + '/index.html');
  await a.locator('#fl-person-chip', { hasText: 'Arturo' }).waitFor();
  const b = await newPage(fl);
  await b.goto(base + '/index.html');
  await b.locator('#fl-person-chip', { hasText: 'Arturo' }).waitFor();
  await a.getByRole('button', { name: 'Change user' }).click();
  await a.getByRole('button', { name: /Luz/ }).click();
  await b.locator('#fl-person-chip', { hasText: 'Luz' }).waitFor();
});

await scenario('R6 browser: storage that refuses writes leaves nobody selected and blocks recording', async ({ newPage }) => {
  const fl = fakeFL();
  const page = await newPage(fl);
  await seed(page, { 'fl-browser-token': TOKEN });
  await page.addInitScript(() => { const real = Storage.prototype.setItem; Storage.prototype.setItem = function (k, v) { if (k === 'fl-person') throw new Error('QuotaExceeded'); return real.call(this, k, v); }; });
  await page.goto(base + '/entry.html');
  await page.getByRole('button', { name: /Arturo/ }).click();
  await page.getByRole('heading', { name: 'Recording is not available' }).waitFor();
  const sent = await page.evaluate(() => FLLedger.request('/receive/prepare', {}).then(() => 'sent', e => e.message));
  assert.match(sent, /not available|Nothing/);
  assert.ok(!fl.state.calls.some(c => c.path === '/receive/prepare'));
  assert.ok(!fl.state.calls.some(c => c.person));
});

await scenario('R8 browser: selector mode with person-select.js missing never falls back to legacy credentials', async ({ newPage }) => {
  const fl = fakeFL();
  const page = await newPage(fl);
  await seed(page, { 'fl-browser-token': TOKEN, 'fl-person': savedPerson(2), 'fl-pin-session-token': 'fls_' + 'y'.repeat(43) });
  await page.route('**/person-select.js*', route => route.abort());
  await page.goto(base + '/entry.html');
  await page.evaluate(() => FLDashboard.ready);
  const sent = await page.evaluate(() => FLLedger.request('/receive/prepare', {}).then(() => 'sent', e => e.message));
  assert.match(sent, /did not load|Nothing was recorded/);
  assert.ok(!fl.state.calls.some(c => c.path === '/receive/prepare'));
  assert.ok(!fl.state.calls.some(c => c.key && c.key.startsWith('fls_')));
  assert.equal(await visible(page, '#sign-in-form'), 0);
});

await scenario('mode on but unavailable: Record entry shows "not available", no key/PIN prompt', async ({ newPage }) => {
  const fl = fakeFL({ available: false, problem: 'PERSON_CHOICES_MISCONFIGURED' });
  const page = await newPage(fl);
  await page.goto(base + '/entry.html');
  await page.getByRole('heading', { name: 'Recording is not available' }).waitFor();
  assert.equal(await visible(page, '#sign-in-form'), 0);
  assert.equal(await visible(page, '#actor-key'), 0);
  await page.screenshot({ path: out + '/unavailable-390.png' });
});

for (const [outage, leftovers] of [
  ['first config 503 then 200 (transient)', { 'fl-browser-token': TOKEN, 'fl-person': savedPerson(2) }],
  ['config always 503 (permanent outage)', { 'fl-browser-token': TOKEN, 'fl-person': savedPerson(4), 'fl-pin-session-token': 'fls_' + 'M'.repeat(43) }],
]) {
  for (const page_ of ['fl-assistant.html', 'office.html', 'shift-summary.html', 'entry.html', 'index.html']) {
    await scenario(`R11 ${outage}: ${page_} with leftover identities never restores or prompts legacy credentials`, async ({ newPage }) => {
      const fl = fakeFL({ configFail: outage.startsWith('first') ? 'first' : 'always' });
      const page = await newPage(fl);
      await page.addInitScript(() => { if (!sessionStorage.getItem('__s')) { sessionStorage.setItem('fl-key-session-token', 'fls_' + 'L'.repeat(43)); sessionStorage.setItem('__s', '1'); } });
      await seed(page, leftovers);
      await page.goto(base + '/' + page_);
      await page.waitForTimeout(1500);
      const legacy = fl.state.calls.filter(c => c.key && c.key.startsWith('fls_'));
      assert.equal(legacy.length, 0, 'legacy session requests: ' + JSON.stringify(legacy));
      assert.equal(await page.getByText(/Signed in as|Sesión de/).count(), 0);
      assert.equal(await visible(page, '#sign-in-form, #sign-in-panel, #actor-key, .fl-pin-offer, .fl-pin-toggle'), 0);
      assert.ok(!fl.state.calls.some(c => c.method !== 'GET' && c.path !== '/auth/browsers/request'), 'no write was sent');
      assert.ok(!fl.state.calls.some(c => c.person), 'no request as a person while settings are unknown');
      assert.equal(await page.evaluate(() => FLDashboard.configUnknown), true);
      if (await page.locator('#identity').count()) assert.doesNotMatch(await page.locator('#identity').innerText(), /personal key|clave personal|Sign in to/);
      if (page_ === 'entry.html') {
        const sent = await page.evaluate(() => FLLedger.request('/receive/prepare', {}).then(() => 'sent', e => e.message));
        assert.match(sent, /could not be loaded|Nothing was recorded/);
        assert.ok(!fl.state.calls.some(c => c.path === '/receive/prepare'));
      }
    });
  }
}

// R12: selector mode KNOWN on, approved browser, saved Arturo, plus different
// leftover credentials (old Luz key session, an enrolled phone with its PIN
// session) whose legacy whoami fails. Normal pages must never touch them.
for (const page_ of ['index.html', 'runs.html', 'office.html', 'entry.html', 'receipt.html?number=RCV-261103-001', 'shift-summary.html', 'fl-assistant.html']) {
  await scenario(`R12 ${page_}: leftover key/PIN identities are never initialised, restored or offered; Arturo's requests work`, async ({ newPage }) => {
    const fl = fakeFL({ legacyStatus: 503 });
    const page = await newPage(fl);
    await page.addInitScript(() => { if (!sessionStorage.getItem('__s')) { sessionStorage.setItem('fl-key-session-token', 'fls_' + 'L'.repeat(43)); sessionStorage.setItem('__s', '1'); } });
    await seed(page, { 'fl-browser-token': TOKEN, 'fl-person': savedPerson(2), 'fl-pin-session-token': 'fls_' + 'P'.repeat(43),
      'fl-device': JSON.stringify({ secret: 'D'.repeat(43), actor: { id: 4, name: 'Miriam', role: 'office' }, label: "Miriam's phone" }) });
    await page.goto(base + '/' + page_);
    await page.locator('#fl-person-chip', { hasText: 'Entering as: Arturo (Floor Lead)' }).waitFor();
    await page.waitForTimeout(800);
    if (await page.evaluate(() => !!window.FL?.fetchWithTimeout)) {
      const status = await page.evaluate(() => FL.fetchWithTimeout('/inventory/current').then(r => r.status, e => 'failed: ' + e.message));
      assert.equal(status, 200);
      assert.ok(fl.state.calls.some(c => c.path === '/inventory/current' && c.key === TOKEN && c.person === '2'));
    }
    const legacy = fl.state.calls.filter(c => c.key && c.key.startsWith('fls_'));
    assert.equal(legacy.length, 0, 'legacy requests: ' + JSON.stringify(legacy));
    assert.equal(await visible(page, '.fl-pin-offer, .fl-pin-toggle, #sign-in-form, #sign-in-panel, #actor-key'), 0);
    assert.equal(await page.getByText(/Set up a 4-digit PIN|PIN for |Signed in as|Sesión de/).filter({ visible: true }).count(), 0);
    assert.ok(fl.state.calls.filter(c => c.person).every(c => c.person === '2' && c.key === TOKEN), 'only Arturo is ever named');
  });
}

await scenario('owner-proof page keeps the legacy owner sign-in (Staff & phones still offers the key/PIN field)', async ({ newPage }) => {
  const fl = fakeFL();
  const page = await newPage(fl);
  await seed(page, { 'fl-browser-token': TOKEN, 'fl-person': savedPerson(1) });
  await page.goto(base + '/staff-access.html');
  await page.evaluate(() => FLDashboard.ready);
  await page.locator('#sign-in-form').waitFor();
  assert.equal(await visible(page, '#sign-in-form'), 1);
  assert.equal(await page.evaluate(() => FLLedger.selecting()), false);
});

// R13: while the page's one config is still on its way, the mode is unknown:
// no credential field, no personal-key copy, and a queued sign-in does nothing.
for (const [label, mode] of [['known selector-off', false], ['known selector-on', true]]) {
  for (const page_ of ['shift-summary.html', 'fl-assistant.html', 'entry.html']) {
    await scenario(`R13 slow config then ${label}: ${page_} shows no credential prompt before settings are known`, async ({ newPage }) => {
      const fl = fakeFL({ mode, configDelay: 1000 });
      const page = await newPage(fl);
      await seed(page, mode ? { 'fl-browser-token': TOKEN, 'fl-person': savedPerson(2) } : {});
      await page.goto(base + '/' + page_, { waitUntil: 'domcontentloaded' });
      await page.waitForTimeout(100);
      assert.equal(await visible(page, '#actor-key, #sign-in-form, #sign-in-panel, #read-only'), 0, 'credential UI visible before config');
      assert.doesNotMatch(await page.locator('#identity').innerText().catch(() => ''), /personal key|clave personal|Sign in to/);
      // A queued submit / read-only while pending must not start a legacy sign-in.
      await page.evaluate(() => {
        const key = document.getElementById('actor-key'); if (key) key.value = 'fls-should-never-be-exchanged';
        document.getElementById('sign-in-form')?.requestSubmit?.();
        document.getElementById('read-only')?.click?.();
      });
      await page.waitForTimeout(1500);
      assert.ok(!fl.state.calls.some(c => c.path === '/auth/session'), 'no key exchange');
      assert.ok(!fl.state.calls.some(c => c.path === '/reports/shift-summary' && !c.person), 'no read-only legacy view started while pending');
      if (mode) {
        await page.locator('#fl-person-chip', { hasText: 'Arturo' }).waitFor();
        assert.equal(await visible(page, '#actor-key, #sign-in-form, #sign-in-panel'), 0);
      } else if (page_ !== 'entry.html') {
        // Known selector-off keeps the existing legacy sign-in exactly where it was.
        await page.locator('#sign-in-panel').waitFor();
        assert.equal(await visible(page, '#actor-key'), 1);
      }
    });
  }
}

await scenario('R14 Shift Summary ?api= override: the browser token and name are never sent to another origin', async ({ newPage }) => {
  const fl = fakeFL();
  const page = await newPage(fl);
  await seed(page, { 'fl-browser-token': TOKEN, 'fl-person': savedPerson(2) });
  await page.goto(base + '/shift-summary.html?api=https://other.invalid');
  await page.waitForTimeout(1500);
  const leaked = fl.state.foreign.filter(c => c.key === TOKEN || c.person);
  assert.equal(leaked.length, 0, 'credentialed foreign requests: ' + JSON.stringify(leaked));
  assert.ok(!fl.state.foreign.some(c => c.key && c.key.startsWith('flb_')));
  assert.equal(await page.getByText(/different Factory Ledger address/).count() > 0, true);
});

await scenario('Shift Summary: selected Arturo loads his own summary with his name', async ({ newPage }) => {
  const fl = fakeFL();
  const page = await newPage(fl);
  await seed(page, { 'fl-browser-token': TOKEN, 'fl-person': savedPerson(2) });
  await page.goto(base + '/shift-summary.html');
  await page.waitForTimeout(1200);
  const loads = fl.state.calls.filter(c => c.path === '/reports/shift-summary');
  assert.ok(loads.length >= 1 && loads.every(c => c.person === '2' && c.key === TOKEN), JSON.stringify(loads));
  assert.equal(await visible(page, '#sign-in-panel'), 0);
});

// A REAL browser restart: persistent profile on disk, browser closed and
// relaunched, storage NOT re-seeded after the restart.
{
  const name = 'browser close + relaunch (persistent profile, no re-seed): the chosen name survives';
  const profile = await fs.mkdtemp(path.join(out, 'profile-'));
  try {
    const fl = fakeFL();
    let ctx = await chromium.launchPersistentContext(profile, { channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome', headless: true, viewport: { width: 390, height: 844 } });
    let page = ctx.pages()[0] || await ctx.newPage();
    await page.route('**/*', fl.handle);
    await page.goto(base + '/index.html');
    await page.getByRole('button', { name: 'Ask Michael to approve this browser' }).click();
    fl.state.approved = true;
    await page.getByRole('button', { name: /Miriam/ }).click({ timeout: 8000 });
    await page.locator('#fl-person-chip', { hasText: 'Entering as: Miriam (Office)' }).waitFor();
    await ctx.close();                                       // browser process ends
    ctx = await chromium.launchPersistentContext(profile, { channel: process.env.PLAYWRIGHT_CHANNEL || 'chrome', headless: true, viewport: { width: 390, height: 844 } });
    page = ctx.pages()[0] || await ctx.newPage();
    await page.route('**/*', fl.handle);
    await page.goto(base + '/office.html');
    await page.locator('#fl-person-chip', { hasText: 'Entering as: Miriam (Office)' }).waitFor();
    await page.locator('#office-action').waitFor();
    assert.equal(fl.state.calls.filter(c => c.path === '/auth/whoami').at(-1).person, '4');
    assert.equal(fl.state.calls.filter(c => c.path === '/auth/browsers/request').length, 1, 'no second approval needed');
    await ctx.close();
    results.push({ name, ok: true }); console.log('ok -', name);
  } catch (error) {
    results.push({ name, ok: false, error: String(error.stack || error) }); console.log('not ok -', name, '\n', error.stack || error);
  } finally { await fs.rm(profile, { recursive: true, force: true }); }
}

await browser.close();
await server.close();
await fs.writeFile(out + '/results.json', JSON.stringify(results, null, 1));
const failed = results.filter(r => !r.ok);
console.log(`# pass ${results.length - failed.length}\n# fail ${failed.length}`);
process.exit(failed.length ? 1 : 0);
