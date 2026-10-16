// Mobile UX round: open at latest, Today button, photo states. Actual dashboard
// assets over local HTTP with API fixtures; no model, database or Record request.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
const playwright = await import(process.env.PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const out = process.env.MOBILE_UX_OUT || path.join(root, 'docs/validation/fl-assistant-mobile-ux/browser');
await fs.mkdir(out, { recursive: true });
const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==', 'base64');
const actor = { id: 7, name: 'Michael', role: 'floor' };
const SAVED = 'saved-chat-1';
let mode = 0, records = 0, sessionsCreated = 0, uploadFailure = null, uploads = [], resumed = [], nextId = 500;
const hour = 36e5, now = Date.now();
// 30 turns yesterday (the old failures) and 6 this morning.
const history = Array.from({ length: 36 }, (_, i) => ({
  turn_id: 'old-' + i, text: i < 30 ? `Old message ${i + 1} from yesterday` : `Morning message ${i - 29}`,
  created_at: new Date(i < 30 ? now - 26 * hour + i * 60e3 : now - 2 * hour + (i - 30) * 60e3).toISOString(),
  cards: [{ kind: 'misunderstood' }]
}));
const server = http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, 'http://localhost');
    const json = (value, status = 200) => { response.statusCode = status; response.setHeader('Content-Type', 'application/json'); response.end(JSON.stringify(value)); };
    const body = async () => { const chunks = []; for await (const chunk of request) chunks.push(chunk); return Buffer.concat(chunks); };
    if (url.pathname === '/auth/session/config') return json({ key_sessions_enabled: !!mode });
    if (url.pathname === '/auth/whoami') return json({ key_kind: mode ? 'session' : 'actor', actor });
    if (url.pathname === '/auth/session') return json({ key_kind: 'session', actor, session_token: 'fl-mobile-fixture' });
    if (url.pathname === '/assistant/session') { sessionsCreated++; return json({ session_id: 'new-' + sessionsCreated, turns: [], drafts: [] }); }
    if (url.pathname === '/assistant/resume') { const sid = JSON.parse(await body()).session_id; resumed.push(sid); return sid === SAVED ? json({ session_id: SAVED, turns: history, drafts: [] }) : json({ detail: { error_code: 'SESSION_NOT_FOUND' } }, 404); }
    if (url.pathname === '/assistant/record') { records++; throw Error('Mobile UX runner must stop before Record'); }
    if (url.pathname === '/assistant/attachment') {
      const raw = (await body()).toString('latin1'), kind = raw.match(/name="kind"\r\n\r\n([a-z_]+)/)?.[1];
      uploads.push(kind);
      if (uploadFailure === 'abort') { uploadFailure = null; request.socket.destroy(); return; }
      if (uploadFailure === 'invalid') { uploadFailure = null; return json({ detail: { error_code: 'PHOTO_INVALID', message: 'Use a JPEG, PNG or WebP photo up to 5 MB' } }, 422); }
      const id = nextId++; return json({ id, attachment_id: id, filename: 'photo.png', kind, media_type: 'image/png', size: 70, attachment_only: true });
    }
    if (url.pathname === '/assistant/attachment/read') { response.setHeader('Content-Type', 'image/png'); return response.end(png); }
    if (url.pathname === '/assistant/turn') {
      const turn = JSON.parse(await body());
      if (/received/i.test(turn.text)) return json({ cards: [{ kind: 'draft', id: 'draft-' + turn.turn_id, attachment_ids: turn.attachment_ids,
        prepared: { action: 'receive', actor, can_commit: true, blockers: [], draft: { product_name: 'Sunflower Seeds', cases: 6, case_size_lb: 50, total_lb: 300 } } }] });
      return json({ cards: [{ kind: 'unsupported_request' }] });
    }
    if (url.pathname === '/receipts') return json({ receipts: Array.from({ length: 12 }, (_, i) => ({ receipt_number: `RCV-261016-${String(i + 1).padStart(3, '0')}`, actor, entered_at: new Date(now - i * 60e3).toISOString(), action: 'receive', draft: { product_name: 'Sunflower Seeds', total_lb: 300 } })), has_more: false });
    const local = url.pathname === '/dash/fl-assistant' ? 'dashboard/fl-assistant.html' : url.pathname.replace(/^\//, '');
    assert.ok(local.startsWith('dashboard/'));
    const filename = path.join(root, local), ext = path.extname(filename);
    response.setHeader('Content-Type', { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css' }[ext] || 'text/plain');
    response.end(await fs.readFile(filename));
  } catch (error) { response.statusCode = 500; response.end(String(error)); }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
const origin = `http://127.0.0.1:${server.address().port}`;
const words = {
  en: { yesterday: 'Yesterday', today: 'Today', jump: 'Jump to latest ↓', lotTag: 'Lot tag', bol: 'BOL (bill of lading)', choose: 'What does this photo show?', attached: 'Attached ✓ · goes with your message',
    failed: 'Upload failed — tap to retry', wrong: 'Not a photo. Use a JPEG, PNG or WebP photo.', remove: 'Remove photo', view: 'View attached photo', hint: 'Your photo is still attached', todayButton: 'Today', entries: 'You recorded 12 entries today' },
  es: { yesterday: 'Ayer', today: 'Hoy', jump: 'Ir a lo más reciente ↓', lotTag: 'Etiqueta de lote', bol: 'BOL (conocimiento de embarque)', choose: '¿Qué muestra esta foto?', attached: 'Adjunta ✓ · se enviará con tu mensaje',
    failed: 'Falló la carga — toca para reintentar', wrong: 'No es una foto. Usa una foto JPEG, PNG o WebP.', remove: 'Quitar foto', view: 'Ver foto adjunta', hint: 'Tu foto sigue adjunta', todayButton: 'Hoy', entries: 'Registraste 12 entradas hoy' }
};
const results = [], engines = (process.env.MOBILE_UX_ENGINES || 'webkit').split(',');
const position = page => page.evaluate(() => {
  const scroller = document.scrollingElement, last = [...document.querySelectorAll('#messages > .message')].at(-1).getBoundingClientRect();
  return { fromBottom: scroller.scrollHeight - scroller.scrollTop - innerHeight, lastVisible: last.top < innerHeight && last.bottom > 0, jumpHidden: document.getElementById('jump-latest').hidden,
    headerTop: document.querySelector('.topbar').getBoundingClientRect().top, width: scroller.scrollWidth };
});
try {
  for (const engine of engines) {
    const browser = await playwright[engine].launch({ headless: true });
    try {
      for (mode of [0, 1]) for (const language of ['en', 'es']) for (const width of [375, 390, 393, 430]) {
        const w = words[language], shot = (name) => language === 'en' ? page.screenshot({ path: path.join(out, `${engine}-${mode}-${width}-${name}.png`) }) : null;
        const context = await browser.newContext({ viewport: { width, height: 844 }, deviceScaleFactor: 1, isMobile: true, hasTouch: true, timezoneId: 'America/New_York' });
        await context.addInitScript(({ language, saved }) => { if (!sessionStorage.getItem('seeded')) { localStorage.setItem('fl-assistant-language', language); localStorage.setItem('fl-assistant-chat:7', saved); sessionStorage.setItem('seeded', '1'); } }, { language, saved: SAVED });
        const page = await context.newPage(), errors = [], row = (scenario) => results.push({ engine, sessions_enabled: !!mode, language, width, scenario, status: 'passed' });
        page.on('pageerror', error => errors.push(error.message));
        sessionsCreated = 0; resumed = []; uploads = [];
        const signIn = async () => {
          if (!(await page.locator('#chat-panel').isVisible())) {
            await page.locator('#actor-key').fill('fixture-key'); await page.locator('#sign-in-form button').click();
          }
          await page.locator('#chat-panel').waitFor({ state: 'visible' }); await page.waitForFunction(() => !document.getElementById('send').disabled);
        };
        await page.goto(origin + '/dash/fl-assistant'); await signIn();
        // 1. Open at the latest message, history dated, no new chat.
        let at = await position(page);
        assert.ok(at.fromBottom < 4 && at.lastVisible && at.jumpHidden, JSON.stringify(at)); assert.equal(at.width, width);
        const dividers = await page.locator('.day-divider').allInnerTexts();
        assert.equal(dividers[0], w.yesterday); assert.match(dividers.at(-1), new RegExp('^' + w.today + ' · '));
        await shot('1-open'); row('opens-at-latest');
        await page.evaluate(() => scrollTo(0, 400)); await page.waitForFunction(() => !document.getElementById('jump-latest').hidden);
        at = await position(page); assert.equal(at.headerTop, 0, 'header stays on screen');
        assert.equal(await page.locator('#jump-latest').innerText(), w.jump);
        await shot('1-jump'); await page.locator('#jump-latest').tap();
        await page.waitForFunction(() => document.getElementById('jump-latest').hidden);
        at = await position(page); assert.ok(at.fromBottom < 4 && at.lastVisible, JSON.stringify(at)); row('jump-to-latest');
        await page.reload(); await signIn();
        at = await position(page); assert.ok(at.fromBottom < 4 && at.lastVisible, JSON.stringify(at));
        assert.deepEqual(resumed, [SAVED, SAVED]); assert.equal(sessionsCreated, 0, 'reload never starts a new chat'); row('reload-resumes-same-chat');
        // 2. Today from the header, starting at the top of a long history.
        await page.evaluate(() => scrollTo(0, 0));
        await page.getByRole('button', { name: w.todayButton, exact: true }).tap();
        await page.waitForFunction(() => !document.getElementById('send').disabled);
        const todayCard = page.locator('#messages > .message').last();
        assert.match(await todayCard.innerText(), new RegExp(w.entries));
        const box = await todayCard.boundingBox(); assert.ok(box.y >= 0 && box.y < 844 / 2, JSON.stringify(box));
        await shot('2-today'); row('today-opens-on-screen');
        // 3. Photo: pick -> type -> attached; nothing uploads before the type is chosen.
        await page.locator('#photo').setInputFiles({ name: 'tag.png', mimeType: 'image/png', buffer: png });
        await page.locator('.attachment-chip.photo-choose').waitFor();
        assert.match(await page.locator('.attachment-chip').innerText(), new RegExp(w.choose.replace(/[?¿]/g, '.')));
        assert.equal(await page.locator('.attachment-chip img.photo-thumb').evaluate(image => image.complete && image.naturalWidth > 0), true);
        assert.deepEqual(uploads, []);
        await page.locator('#message').fill('test'); await page.locator('#send').tap();
        assert.notEqual(await page.locator('#notice').innerText(), ''); assert.equal(await page.locator('.user-message', { hasText: /^test$/ }).count(), 0);
        await shot('3-choose');
        await page.getByRole('button', { name: w.lotTag, exact: true }).tap();
        await page.locator('.attachment-chip.photo-attached').waitFor();
        assert.match(await page.locator('.attachment-chip').innerText(), new RegExp(w.attached));
        assert.deepEqual(uploads, ['lot_tag']); await shot('3-attached'); row('photo-attached');
        // Words-free photo turn: plain reminder, photo stays attached.
        await page.locator('#message').fill('Photo'); await page.locator('#send').tap();
        await page.locator('.photo-hint').waitFor(); assert.match(await page.locator('.photo-hint').innerText(), new RegExp(w.hint));
        assert.equal(await page.locator('.attachment-chip.photo-attached').count(), 1); row('photo-only-hint');
        // Draft shows the photo before Record.
        await page.locator('#message').fill('we received six cases of sunflower seeds'); await page.locator('#send').tap();
        const figure = page.locator('[data-draft-id] figure.draft-photo').last(); await figure.waitFor();
        assert.equal(await figure.locator('img').evaluate(image => image.complete && image.naturalWidth > 0), true);
        await figure.getByRole('button', { name: w.view }).waitFor();
        assert.equal(await page.locator('.attachment-chip').count(), 0); await shot('3-draft'); row('photo-on-draft');
        // Network drop -> retry; FL rejection -> plain reason; remove.
        uploadFailure = 'abort';
        await page.locator('#photo').setInputFiles({ name: 'bol.png', mimeType: 'image/png', buffer: png });
        await page.getByRole('button', { name: w.bol, exact: true }).tap();
        await page.getByRole('button', { name: w.failed, exact: true }).waitFor(); await shot('3-failed');
        await page.getByRole('button', { name: w.failed, exact: true }).tap();
        await page.locator('.attachment-chip.photo-attached').waitFor(); row('photo-retry');
        await page.getByRole('button', { name: w.remove }).tap(); assert.equal(await page.locator('.attachment-chip').count(), 0);
        uploadFailure = 'invalid';
        await page.locator('#photo').setInputFiles({ name: 'tag.png', mimeType: 'image/png', buffer: png });
        await page.getByRole('button', { name: w.lotTag, exact: true }).tap();
        await page.locator('.attachment-chip.photo-wrong_type').waitFor(); assert.match(await page.locator('.attachment-chip').innerText(), new RegExp(w.wrong.replace(/\./g, '\\.')));
        await page.getByRole('button', { name: w.remove }).tap(); row('photo-rejected-and-removed');
        at = await position(page); assert.equal(at.width, width);
        assert.deepEqual(errors, []);
        await context.close();
        console.log(`${engine}, sessions ${mode}, ${language}, ${width}px: 9 checks passed`);
      }
    } finally { await browser.close(); }
  }
  assert.equal(records, 0);
} finally {
  server.close();
  await fs.writeFile(path.join(out, 'results.json'), JSON.stringify({ runner: 'tests/visual/run-fl-assistant-mobile-ux.mjs', api_fixtures: true, local_only: true, records, results }, null, 2) + '\n');
}
console.log(`${results.length} mobile UX checks passed; ${records} Record requests.`);
