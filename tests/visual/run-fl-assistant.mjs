// UI interaction/viewport checks. All API/media calls are deterministic fixtures.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startStaticServer } from './lib/server.mjs';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const out = process.env.F1_VISUAL_OUT || '/tmp/fl-f1-visual';
await fs.mkdir(out, { recursive: true });
const server = await startStaticServer(root);
const browser = await chromium.launch({ headless: true });
const sid = '0c101ab1-8c56-42a2-8f29-c048330030c3', did = '616eac35-0b6e-4288-b40f-920639a83525';
const receipt = { success: true, receipt_number: 'MK-261008-001', transaction_id: 17, product_name: 'Classic Test Batch', lot_code: 'B26-10-08-001' };
let records = [], turns = [], uploaded = 0, transcribed = 0;
const draft = { kind: 'draft', id: did, attachment_ids: [], prepared: {
  action: 'make', can_commit: false, expires_at: '2026-10-08T18:30:00-04:00', actor: { name: 'Arturo', role: 'floor' },
  blockers: [{ code: 'LOT_NOT_CONFIRMED', message: 'Confirm the physical lot / Confirma el lote físico.' }],
  warnings: [{ code: 'POSSIBLE_DUPLICATE', message: 'A similar entry exists.', message_es: 'Existe una entrada similar.', requires_ack: true }],
  draft: { product_name: 'Classic Test Batch', batches: 2, total_output_lb: 800, happened_at: '2026-10-08T18:20:00-04:00',
    input_plan: [{ lot_id: 19, lot_code: '26-10-08-OATS-1234', quantity_lb: 800, product_name: 'Rolled Oats', confirmed: false }] }
} };
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  await context.addInitScript(() => {
    Object.defineProperty(navigator, 'mediaDevices', { value: { getUserMedia: async () => ({ getTracks: () => [{ stop() {} }] }) } });
    window.MediaRecorder = class { static isTypeSupported() { return true; } constructor() { this.mimeType = 'audio/webm'; this.state = 'inactive'; } start() { this.state = 'recording'; } stop() { this.state = 'inactive'; this.ondataavailable({ data: new Blob(['audio']) }); setTimeout(() => this.onstop(), 0); } };
  });
  const page = await context.newPage(), errors = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.route('**/auth/whoami', route => route.fulfill({ json: { actor: { id: 4, name: 'Arturo', role: 'floor' } } }));
  await page.route('**/assistant/**', async route => {
    const url = new URL(route.request().url()), data = route.request().postDataJSON?.bind(route.request());
    if (url.pathname.endsWith('/session')) return route.fulfill({ json: { session_id: sid } });
    if (url.pathname.endsWith('/resume')) return route.fulfill({ json: { turns: [{ text: 'Made two batches', cards: [draft] }], drafts: [{ id: did, card: draft, status: 'committed', result: receipt }] } });
    if (url.pathname.endsWith('/turn')) {
      const body = data(); turns.push(body);
      const cards = body.text === 'ambiguous' ? [{ kind: 'choices', id: '811eac35-0b6e-4288-b40f-920639a83525', result: { ask: 'Which product?', candidates: [{ id: 1, label: '<img src=x onerror=alert(1)> Classic' }, { id: 2, label: 'Classic Chocolate Chip' }] } }] : [draft];
      return route.fulfill({ json: { cards } });
    }
    if (url.pathname.endsWith('/record')) {
      records.push(data());
      if (records.length === 1) return route.abort('failed');
      return route.fulfill({ json: { kind: 'receipt', result: { ...receipt, replayed: true } } });
    }
    if (url.pathname.endsWith('/transcribe')) { transcribed++; return route.fulfill({ json: { text: 'Recibí dos cajas', editable: true, sent: false } }); }
    if (url.pathname.endsWith('/attachment')) { uploaded++; return route.fulfill({ json: { id: '7e3381ea-acfb-4cd7-92c0-4ceda5d7c2e3', filename: 'tag.png', attachment_only: true } }); }
    return route.fulfill({ status: 500, json: { detail: 'Unexpected fixture request' } });
  });
  await page.goto(server.origin + '/dashboard/fl-assistant.html');
  await page.screenshot({ path: path.join(out, 'signin-desktop.png'), fullPage: true });
  await page.locator('#actor-key').fill('fixture-key'); await page.locator('#sign-in-form button').click();
  await page.locator('#chat-panel').waitFor({ state: 'visible' });
  await page.screenshot({ path: path.join(out, 'chat-desktop.png'), fullPage: true });
  assert.equal(await page.evaluate(() => JSON.stringify(localStorage).includes('fixture-key')), false);
  await page.locator('#message').fill('made two batches'); await page.locator('#send').click();
  const record = page.getByRole('button', { name: 'Record', exact: true });
  await record.waitFor(); assert.equal(await record.isDisabled(), true);
  const lotInput = page.locator('.lot-controls input'); assert.equal(await lotInput.inputValue(), '');
  await lotInput.fill('1234'); await page.locator('.warning input').check();
  await page.screenshot({ path: path.join(out, 'draft-desktop.png'), fullPage: true });
  await record.click(); await page.getByText('NOT recorded — no receipt confirmed.', { exact: false }).waitFor();
  assert.equal(await page.locator('.receipt').count(), 0);
  await record.click(); await page.locator('.receipt').waitFor();
  assert.deepEqual(records[0], records[1]); assert.equal(records[0].draft_id, did);
  assert.deepEqual(records[0].lot_confirmations, [{ lot_id: 19, method: 'last4', value: '1234' }]);
  assert.equal(JSON.stringify(records).includes('payload_hash'), false);
  await page.screenshot({ path: path.join(out, 'receipt-desktop.png'), fullPage: true });
  // Dictation never submits a turn and the resulting text is editable.
  const priorTurns = turns.length;
  await page.locator('#dictate').dispatchEvent('pointerdown', { pointerId: 1 });
  await page.waitForFunction(() => document.querySelector('#dictate').getAttribute('aria-pressed') === 'true');
  await page.locator('#dictate').dispatchEvent('pointerup', { pointerId: 1 });
  await page.waitForFunction(() => document.querySelector('#message').value.includes('Recibí dos cajas'));
  assert.equal(turns.length, priorTurns); assert.equal(transcribed, 1);
  await page.locator('#message').fill('Edited transcript');
  await page.locator('#photo').setInputFiles({ name: 'tag.png', mimeType: 'image/png', buffer: Buffer.from('photo fixture') });
  await page.locator('.attachment-chip').waitFor(); assert.equal(uploaded, 1);
  await page.locator('#message').fill('ambiguous'); await page.locator('#send').click();
  await page.getByRole('button', { name: 'Classic Chocolate Chip', exact: true }).waitFor();
  assert.equal(await page.locator('.card img').count(), 0);
  await page.getByRole('button', { name: 'Classic Chocolate Chip', exact: true }).click();
  await page.waitForFunction(() => document.querySelectorAll('[data-draft-id]').length === 2);
  assert.equal(turns.at(-1).selected_id, 2);
  // Recovered receipt comes from saved commit JSON, with no duplicate Record.
  await page.reload(); await page.locator('#actor-key').fill('fixture-key'); await page.locator('#sign-in-form button').click();
  await page.locator('.receipt').waitFor(); assert.equal(records.length, 2);
  await page.locator('#language').click(); assert.equal(await page.locator('#send').textContent(), 'Enviar ↑');
  for (const scheme of ['light', 'dark']) {
    await page.emulateMedia({ colorScheme: scheme }); await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await page.screenshot({ path: path.join(out, 'receipt-mobile-' + scheme + '.png'), fullPage: true });
  }
  assert.deepEqual(errors, []);
  console.log('F1 browser: draft/receipt, lost-response retry, lot inputs, warnings, editable dictation, attachments, choices/XSS, reload recovery, ES, 390px light/dark passed.');
} finally { await browser.close(); await server.close(); }
