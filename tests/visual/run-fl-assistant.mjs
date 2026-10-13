// UI interaction/viewport checks. All API/media calls are deterministic fixtures.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { startStaticServer } from './lib/server.mjs';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const out = process.env.F1_VISUAL_OUT || path.join(root, '.venv-test', 'fl-f1-visual');
await fs.mkdir(out, { recursive: true });
const server = await startStaticServer(root);
const browser = await chromium.launch({ headless: true, ...(process.env.PLAYWRIGHT_CHANNEL ? { channel: process.env.PLAYWRIGHT_CHANNEL } : {}) });
const sid = '0c101ab1-8c56-42a2-8f29-c048330030c3', did = '616eac35-0b6e-4288-b40f-920639a83525';
const receipt = { success: true, receipt_number: 'MK-261008-001', transaction_id: 17, product_name: 'Classic Test Batch', lot_code: 'B26-10-08-001',
  shortages: [{ product_name: 'Rolled Oats', lot_code: '26-10-08-OATS-004', short_lb: 10, exception_id: 26, due_at: '2026-10-12T23:59:00-04:00' }] };
let records = [], turns = [], uploaded = 0, uploadedKinds = [], reads = [], shipAttempts = 0, transcribed = 0, heldRequested = false, approved = false;
const photoBytes = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+iDbcAAAAASUVORK5CYII=', 'base64');
const shipId = '8dba4c89-a6e7-4da2-a9a8-4d697ea58dbc', blockedShipId = '0b01d8be-76b7-409b-9ef1-4c90f9f718ee';
const shipReceipt = { success: true, receipt_number: 'SHP-261013-001', shipment_id: 24, bol_reference: 'BOL-261013-001', proof_status: 'complete' };
const shipDraft = { kind: 'draft', id: shipId, attachment_ids: [], prepared: { action: 'ship_order', can_commit: false, actor: { name: 'Arturo' },
  blockers: [{ code: 'LOT_NOT_CONFIRMED', message: 'Confirm the shipping lot.' }], warnings: [],
  draft: { customer: 'Staging Customer', order: 'SO-261013-001', total_lb: 50, total_cases: 2, bol_reference: 'BOL-261013-001', photo: 'attached',
    lines: [{ product: 'Packed Test Product', quantity_lb: 50, cases: 2, is_service: false, lots: [{ lot_id: 21, lot_code: '26-10-13-FIN-001' }] }],
    input_plan: [{ lot_id: 21, lot_code: '26-10-13-FIN-001', quantity_lb: 50, confirmed: false }] } } };
const blockedShipDraft = { ...shipDraft, id: blockedShipId, prepared: { ...shipDraft.prepared, can_commit: true,
  blockers: [{ code: 'BOL_REQUIRED', message: 'Enter the BOL number before recording.' }],
  draft: { ...shipDraft.prepared.draft, bol_reference: '', photo: 'pending', input_plan: [] } } };
const heldId = 'c9246549-acd4-44c2-8a21-3d3858848286';
const heldResult = { held: true, status: 'awaiting_approval', exception_id: 27 };
const heldDraft = { kind: 'draft', id: heldId, prepared: { action: 'found', can_commit: true,
  actor: { name: 'Arturo' }, blockers: [], warnings: [], draft: { product_name: 'Counted ingredient', quantity: 600 } } };
const heldReceipt = { success: true, receipt_number: 'FND-261009-001', transaction_id: 18 };
const draft = { kind: 'draft', id: did, attachment_ids: [], prepared: {
  action: 'make', can_commit: false, expires_at: '2026-10-08T18:30:00-04:00', actor: { name: 'Arturo', role: 'floor' },
  blockers: [{ code: 'LOT_NOT_CONFIRMED', message: 'Confirm the physical lot / Confirma el lote físico.' }],
  warnings: [{ code: 'POSSIBLE_DUPLICATE', message: 'A similar entry exists.', message_es: 'Existe una entrada similar.', requires_ack: true },
    { code: 'WILL_CREATE_SHORTAGE', message: 'Will create a shortage of 10 lb.', message_es: 'Creará un faltante de 10 lb.', requires_ack: false }],
  draft: { product_name: 'Classic Test Batch', batches: 2, total_output_lb: 800, happened_at: '2026-10-08T18:20:00-04:00',
    input_plan: [{ lot_id: 19, lot_code: '26-10-08-OATS-004', quantity_lb: 800, product_name: 'Rolled Oats', confirmed: false }] }
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
    if (url.pathname.endsWith('/resume')) return route.fulfill({ json: { turns: [{ text: 'Made two batches', cards: [draft] }], drafts: [{ id: did, card: draft, status: 'committed', result: receipt },
      ...(heldRequested ? [{ id: heldId, card: heldDraft, status: 'pending', result: heldResult }] : [])] } });
    if (url.pathname.endsWith('/turn')) {
      const body = data(); turns.push(body);
      const cards = body.text === 'ambiguous' ? [{ kind: 'choices', id: '811eac35-0b6e-4288-b40f-920639a83525', result: { ask: 'Which product?', candidates: [{ id: 1, label: '<img src=x onerror=alert(1)> Classic' }, { id: 2, label: 'Classic Chocolate Chip' }] } }] : [draft];
      return route.fulfill({ json: { cards: body.text === 'held' ? [{ ...heldDraft, attachment_ids: body.attachment_ids }] :
        body.text === 'ship without BOL' ? [blockedShipDraft] : body.text === 'ship' ? [{ ...shipDraft, attachment_ids: body.attachment_ids }] : cards } });
    }
    if (url.pathname.endsWith('/record')) {
      records.push(data());
      if (data().draft_id === shipId) {
        shipAttempts++;
        if (shipAttempts === 1) return route.abort('failed');
        return route.fulfill({ json: { kind: 'receipt', draft_id: shipId, result: { ...shipReceipt, replayed: true } } });
      }
      if (data().draft_id === heldId) {
        heldRequested = true;
        return route.fulfill({ status: approved ? 200 : 202, json: approved
          ? { kind: 'receipt', draft_id: heldId, result: heldReceipt }
          : { kind: 'awaiting_approval', draft_id: heldId, result: heldResult } });
      }
      if (records.length === 1) return route.abort('failed');
      return route.fulfill({ json: { kind: 'receipt', result: { ...receipt, replayed: true } } });
    }
    if (url.pathname.endsWith('/transcribe')) { transcribed++; return route.fulfill({ json: { text: 'Recibí dos cajas', editable: true, sent: false } }); }
    if (url.pathname.endsWith('/attachment/read')) { reads.push(data().attachment_id); return route.fulfill({ body: photoBytes, contentType: 'image/png' }); }
    if (url.pathname.endsWith('/attachment')) {
      assert.equal(url.searchParams.get('session_id'), sid);
      const form = route.request().postData(), kind = form.match(/name="kind"\r\n\r\n([^\r]+)/)?.[1];
      assert.ok(form.includes('name="file"')); assert.ok(['bol', 'truck_photo', 'correction_photo', 'lot_tag'].includes(kind));
      uploaded++; uploadedKinds.push(kind); return route.fulfill({ json: { id: 100 + uploaded, filename: 'tag.png', kind, attachment_only: true } });
    }
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
  await page.getByText('Will create a shortage of 10 lb.', { exact: true }).waitFor();
  await page.getByText('including the hyphen (for example, "-004").', { exact: false }).waitFor();
  await page.locator('.lot-controls select').selectOption('pallet');
  await page.getByText('Enter the full lot code from the pallet tag.', { exact: false }).waitFor();
  assert.equal(await lotInput.inputValue(), '');
  await page.locator('.lot-controls select').selectOption('last4');
  await lotInput.fill('-004'); await page.locator('.warning input').check();
  await page.screenshot({ path: path.join(out, 'draft-desktop.png'), fullPage: true });
  await record.click(); await page.getByText('NOT recorded — no receipt confirmed.', { exact: false }).waitFor();
  assert.equal(await page.locator('.receipt').count(), 0);
  await record.click(); await page.locator('.receipt').waitFor();
  assert.deepEqual(records[0], records[1]); assert.equal(records[0].draft_id, did);
  assert.deepEqual(records[0].lot_confirmations, [{ lot_id: 19, method: 'last4', value: '-004' }]);
  assert.equal(JSON.stringify(records).includes('payload_hash'), false);
  await page.getByText('Shortage recorded', { exact: true }).waitFor();
  await page.getByText('short 10 lb. The floor must resolve this shortage.', { exact: false }).waitFor();
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
  await page.locator('.attachment-chip').waitFor(); assert.equal(uploaded, 1); assert.deepEqual(uploadedKinds, ['lot_tag']);
  await page.locator('#message').fill('ambiguous'); await page.locator('#send').click();
  await page.getByRole('button', { name: 'Classic Chocolate Chip', exact: true }).waitFor();
  assert.equal(await page.locator('.card img').count(), 0);
  await page.getByRole('button', { name: 'Classic Chocolate Chip', exact: true }).click();
  await page.waitForFunction(() => document.querySelectorAll('[data-draft-id]').length === 2);
  assert.equal(turns.at(-1).selected_id, 2); assert.deepEqual(turns.at(-1).attachment_ids, [101]);
  // Recovered receipt comes from saved commit JSON, with no duplicate Record.
  await page.reload(); await page.locator('#actor-key').fill('fixture-key'); await page.locator('#sign-in-form button').click();
  await page.locator('.receipt').waitFor(); assert.equal(records.length, 2);
  await page.locator('#photo-kind').selectOption('correction_photo');
  await page.locator('#photo').setInputFiles({ name: 'correction.png', mimeType: 'image/png', buffer: photoBytes });
  await page.locator('.attachment-chip').waitFor();
  await page.locator('#photo').setInputFiles({ name: 'duplicate.png', mimeType: 'image/png', buffer: photoBytes });
  await page.getByText('Send one correction photo with a message.', { exact: false }).waitFor();
  assert.equal(uploaded, 2);
  await page.locator('#message').fill('held'); await page.locator('#send').click();
  await page.getByRole('button', { name: 'View attached photo', exact: true }).click();
  await page.locator('dialog img').waitFor(); assert.deepEqual(reads, [102]);
  assert.equal(await page.locator('dialog img').evaluate(image => image.src.startsWith('blob:')), true);
  await page.getByRole('button', { name: 'Close', exact: true }).click();
  assert.deepEqual(turns.at(-1).attachment_ids, [102]);
  await page.getByRole('button', { name: 'Record', exact: true }).click();
  await page.getByRole('heading', { name: 'Waiting for owner approval', exact: true }).waitFor();
  assert.equal(await page.getByRole('button', { name: 'Cancel', exact: true }).count(), 0);
  assert.equal(await page.locator('.receipt').count(), 1);
  await page.reload(); await page.locator('#actor-key').fill('fixture-key'); await page.locator('#sign-in-form button').click();
  await page.getByRole('heading', { name: 'Waiting for owner approval', exact: true }).waitFor();
  await page.locator('#language').click();
  await page.getByRole('heading', { name: 'En espera de la aprobación del dueño', exact: true }).waitFor();
  await page.locator('#language').click();
  approved = true;
  await page.getByRole('button', { name: 'Check approval', exact: true }).click();
  await page.getByRole('heading', { name: 'FND-261009-001', exact: true }).waitFor();
  assert.equal(records.at(-1).draft_id, heldId);
  // A6 BOL is a hard blocker, and shipment retries keep their ticket and lot evidence.
  const beforeShip = records.length;
  await page.locator('#message').fill('ship without BOL'); await page.locator('#send').click();
  const blockedShip = page.locator(`[data-draft-id="${blockedShipId}"]`);
  await blockedShip.getByText('Enter the BOL number before recording.', { exact: true }).waitFor();
  assert.equal(await blockedShip.getByRole('button', { name: 'Record', exact: true }).isDisabled(), true);
  assert.equal(records.length, beforeShip);
  await page.locator('#photo-kind').selectOption('bol');
  await page.locator('#photo').setInputFiles({ name: 'bol.png', mimeType: 'image/png', buffer: photoBytes });
  await page.locator('.attachment-chip').waitFor();
  await page.locator('#photo-kind').selectOption('truck_photo');
  await page.locator('#photo').setInputFiles({ name: 'truck.png', mimeType: 'image/png', buffer: photoBytes });
  await page.waitForFunction(() => document.querySelectorAll('.attachment-chip').length === 2);
  await page.locator('#message').fill('ship'); await page.locator('#send').click();
  const shipment = page.locator(`[data-draft-id="${shipId}"]`);
  await shipment.getByRole('heading', { name: 'Ship order', exact: true }).waitFor();
  await shipment.getByText('Staging Customer', { exact: true }).waitFor();
  await shipment.getByText('Packed Test Product', { exact: true }).waitFor();
  await shipment.getByText('BOL-261013-001', { exact: true }).waitFor();
  await shipment.getByText('Attached', { exact: true }).waitFor();
  assert.deepEqual(turns.at(-1).attachment_ids, [103, 104]);
  assert.deepEqual(uploadedKinds, ['lot_tag', 'correction_photo', 'bol', 'truck_photo']);
  const shipRecord = shipment.getByRole('button', { name: 'Record', exact: true });
  assert.equal(await shipRecord.isDisabled(), true);
  await shipment.locator('.lot-controls input').fill('-001');
  await page.screenshot({ path: path.join(out, 'ship-draft-desktop.png'), fullPage: true });
  await shipRecord.click(); await shipment.getByText('NOT recorded — no receipt confirmed.', { exact: false }).waitFor();
  assert.equal(await shipment.locator('.receipt-number').count(), 0);
  await shipRecord.click(); await shipment.getByRole('heading', { name: 'SHP-261013-001', exact: true }).waitFor();
  assert.deepEqual(records.at(-1), records.at(-2));
  assert.deepEqual(records.at(-1).lot_confirmations, [{ lot_id: 21, method: 'last4', value: '-001' }]);
  await shipment.getByText('BOL-261013-001', { exact: true }).waitFor();
  await shipment.getByText('Attached', { exact: true }).waitFor();
  await shipment.scrollIntoViewIfNeeded();
  await page.screenshot({ path: path.join(out, 'ship-receipt-desktop-viewport.png') });
  await page.locator('#language').click(); assert.equal(await page.locator('#send').textContent(), 'Enviar ↑');
  for (const scheme of ['light', 'dark']) {
    await page.emulateMedia({ colorScheme: scheme }); await page.setViewportSize({ width: 390, height: 844 });
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    await shipment.scrollIntoViewIfNeeded();
    await page.screenshot({ path: path.join(out, 'receipt-mobile-' + scheme + '.png'), fullPage: true });
    await page.screenshot({ path: path.join(out, 'ship-receipt-mobile-' + scheme + '-viewport.png') });
  }
  assert.deepEqual(errors, []);
  console.log('F1 browser: draft/receipt, lost-response retry, lot inputs, warnings, editable dictation, A6 typed private integer attachments, BOL blocking, SHP retry, choices/XSS, reload recovery, ES, 390px light/dark passed.');
} finally { await browser.close(); await server.close(); }
