// O-7 page check: every key-reading dashboard page loads against a hosted site.
// Opens each page in headless Chromium, read-only (no clicks that write), and
// fails if GET /auth/dashboard-key is not 200, if any request carrying
// X-API-Key answers 401/403, if a page throws, or if one of the five pages the
// runbook names makes no keyed read at all. Prints statuses and sha256
// prefixes only — never a key value.
//
//   node scripts/check_dashboard_pages.mjs https://<site>/dashboard/ [out.json]
//   (Netlify: https://<deploy>--cns-factory-ledger.netlify.app/ — pages at the root)
import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import { chromium } from 'playwright';

const site = process.argv[2];
if (!site) throw new Error('usage: check_dashboard_pages.mjs <site base url> [out.json]');
const base = site.endsWith('/') ? site : site + '/';
if (/^https:\/\/cns-factory-ledger\.netlify\.app\//.test(base) && process.env.FL_ALLOW_PRODUCTION !== '1')
  throw new Error('Refusing production without FL_ALLOW_PRODUCTION=1');
const sha = value => crypto.createHash('sha256').update(value).digest('hex').slice(0, 12);
// [page, must make a keyed read, optional read-only step that triggers one].
// entry.html reads only after a personal sign-in while key sessions are on.
const PAGES = [
  ['index.html', true], ['process-flow.html', true], ['sankey.html', true], ['traceability.html', true],
  ['history.html', false], ['runs.html', false], ['entry.html', false],
  ['shift-summary.html', true, async page => {
    const readOnly = page.getByRole('button', { name: /read-only|solo lectura/i }).first();
    if (await readOnly.count()) await readOnly.click();
  }],
];
const browser = await chromium.launch();
const report = { site: base, checked_at: new Date().toISOString(), pages: [] };
let failed = false;
try {
  for (const [name, needsKeyedRead, step] of PAGES) {
    const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
    const page = await context.newPage();
    const row = { page: name, key_endpoint: [], keyed: 0, refused: [], errors: [], served_sha256: [] };
    page.on('pageerror', error => row.errors.push(error.message.slice(0, 160)));
    page.on('response', async response => {
      const url = new URL(response.url());
      if (url.pathname === '/auth/dashboard-key') {
        row.key_endpoint.push(response.status());
        const value = await response.json().then(body => body.dashboard_key).catch(() => '');
        if (value && !row.served_sha256.includes(sha(value))) row.served_sha256.push(sha(value));
        return;
      }
      const key = await response.request().headerValue('x-api-key');
      if (!key || key.startsWith('fls_')) return;
      row.keyed += 1;
      if ([401, 403].includes(response.status())) row.refused.push(`${response.status()} ${url.pathname}`);
    });
    const loaded = await page.goto(base + name, { waitUntil: 'load' });
    row.status = loaded?.status();
    const settle = () => page.waitForLoadState('networkidle', { timeout: 20000 }).catch(() => {});
    await settle();
    if (step) { await step(page); await page.waitForTimeout(1000); await settle(); }
    row.ok = row.status === 200 && row.key_endpoint.every(s => s === 200) && !row.refused.length && !row.errors.length
      && (!needsKeyedRead || (row.keyed > 0 && row.key_endpoint.length > 0));
    failed ||= !row.ok;
    report.pages.push(row);
    console.log(`${row.ok ? 'PASS' : 'FAIL'}  ${name}: page ${row.status}, /auth/dashboard-key ${row.key_endpoint.join(',') || 'none'} (sha256 ${row.served_sha256.join(',') || '-'}), keyed reads ${row.keyed}, refused ${row.refused.length ? row.refused.join('; ') : 0}, page errors ${row.errors.length}`);
    await context.close();
  }
} finally {
  await browser.close();
}
if (process.argv[3]) await fs.writeFile(process.argv[3], JSON.stringify(report, null, 2) + '\n');
console.log(failed ? 'FAILED' : 'ALL PASS');
process.exit(failed ? 1 : 0);
