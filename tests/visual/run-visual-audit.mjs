#!/usr/bin/env node
// Visual / mechanical design audit of dashboard/.
//
//   npm run test:visual                 # every screen, every variant
//   npm run test:visual -- --screens S-25,S-42
//   npm run test:visual -- --variants 1440-light
//   npm run test:visual -- --headed
//
// Captures every screen in docs/design/audit/00-screen-inventory.md and runs
// the mechanical clauses of TOUCH-003, ACCESS-008, LAYOUT-003/ACCESS-001,
// LAYOUT-011, LAYOUT-020 and the eight machine-checkable STATUS rules
// (STATUS-002, -004, -005, -006, -007, -008, -010, -011) against each
// capture. Writes the matrix to
// docs/design/audit/06-browser-check.md. It changes no application file and
// makes no network request: every API response comes from tests/visual/fixtures.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

import { startStaticServer } from './lib/server.mjs';
import { installApiStub, buildTokenTable } from './lib/stub.mjs';
import { CHECK_SOURCE, INTERVAL_CAPTURE_SOURCE } from './lib/checks.mjs';
import { SCREENS, PAGE_URLS } from './lib/screens.mjs';
import { writeReport } from './lib/report.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(HERE, '..', '..');
const DASHBOARD = path.join(REPO, 'dashboard');
const SHOT_ROOT = path.join(REPO, 'docs', 'design', 'audit', 'screenshots');
const REPORT = path.join(REPO, 'docs', 'design', 'audit', '06-browser-check.md');

// 1440×900 at 200% browser zoom exposes a 720 CSS-px viewport at DPR 2 — the
// layout consequence ACCESS-001 asks about, not a scaled bitmap.
const VARIANTS = [
  { key: '390-light',  label: '390px · light',  width: 390,  height: 844, dsf: 3, theme: 'light', tier: 'mobile' },
  { key: '390-dark',   label: '390px · dark',   width: 390,  height: 844, dsf: 3, theme: 'dark',  tier: 'mobile' },
  { key: '1440-light', label: '1440px · light', width: 1440, height: 900, dsf: 1, theme: 'light', tier: 'desktop' },
  { key: '1440-dark',  label: '1440px · dark',  width: 1440, height: 900, dsf: 1, theme: 'dark',  tier: 'desktop' },
  { key: 'zoom200-light', label: '1440px @ 200% · light', width: 720, height: 450, dsf: 2, theme: 'light', tier: 'zoom' },
  { key: 'zoom200-dark',  label: '1440px @ 200% · dark',  width: 720, height: 450, dsf: 2, theme: 'dark',  tier: 'zoom' },
];

// The four tabs the brief names for the 200% pass, expressed as the screens
// that live in them (00-screen-inventory groups C, G, H, I) plus the chrome
// those tabs are read through.
const ZOOM_SCREENS = new Set([
  'S-01', 'S-03', 'S-05',
  'S-06', 'S-07', 'S-08', 'S-09', 'S-10', 'S-11', 'S-12', 'S-13',
  'S-24', 'S-25', 'S-26', 'S-27', 'S-28', 'S-29', 'S-30', 'S-31', 'S-32',
  'S-33', 'S-34', 'S-36', 'S-37', 'S-39', 'S-40',
  'S-41', 'S-42', 'S-43', 'S-44', 'S-45',
  'S-46', 'S-47', 'S-48', 'S-49', 'S-50', 'S-51', 'S-52', 'S-53',
]);

const RULES = [
  'TOUCH-003', 'ACCESS-008', 'LAYOUT-003/ACCESS-001', 'LAYOUT-011', 'LAYOUT-020',
  // Category 17 — Status & Data Display. The other six STATUS rules are manual
  // review and are deliberately absent: a rule with no mechanical clause must
  // not appear in the matrix as a pass.
  'STATUS-002', 'STATUS-004', 'STATUS-005', 'STATUS-006', 'STATUS-007',
  'STATUS-008', 'STATUS-010', 'STATUS-011',
];

function parseArgs(argv) {
  const out = { screens: null, variants: null, headed: false, concurrency: 4, reportOnly: false };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--screens') out.screens = new Set(argv[++i].split(',').map(s => s.trim()));
    else if (a === '--variants') out.variants = new Set(argv[++i].split(',').map(s => s.trim()));
    else if (a === '--headed') out.headed = true;
    else if (a === '--concurrency') out.concurrency = Number(argv[++i]) || 1;
    // Re-render the matrix from the last run's raw results without re-capturing.
    else if (a === '--report-only') out.reportOnly = true;
  }
  return out;
}

function appliesTo(screen, variant) {
  if (screen.capturable === false) return false;
  if (variant.tier === 'mobile') return screen.mobile === 'yes' || screen.mobile === 'partial';
  if (variant.tier === 'zoom') return ZOOM_SCREENS.has(screen.id);
  return true;
}

async function newContext(browser, variant) {
  const context = await browser.newContext({
    viewport: { width: variant.width, height: variant.height },
    deviceScaleFactor: variant.dsf,
    colorScheme: variant.theme,
    isMobile: false,
    reducedMotion: 'reduce',
    timezoneId: 'America/New_York',
    locale: 'en-US',
  });
  await context.addInitScript(INTERVAL_CAPTURE_SOURCE);
  await context.addInitScript(
    ([theme]) => {
      try { localStorage.setItem('dashboard-theme', theme); } catch (e) {}
    },
    [variant.theme],
  );
  return context;
}

async function forceTheme(page, theme) {
  // dashboard.css defines paired `[data-theme="dark"]` / `[data-theme="light"]`
  // token blocks. sankey, traceability, process-flow and the scheduler put
  // their tokens on a bare `:root`, so the `data-theme` attribute two of them
  // carry in their markup is decorative and there is no light mode to honour.
  //
  // Toggling the attribute once cannot tell these apart — on the dashboard the
  // init script has already applied the right theme, so setting it again moves
  // nothing either. Compare the two states explicitly instead, then leave the
  // requested one in place.
  // Compare the resolved custom properties, not a painted colour: `body` has a
  // background transition, so reading backgroundColor straight after the
  // attribute flip returns a mid-animation value and two different themes look
  // the same. Custom properties are not transitioned.
  const info = await page.evaluate(t => {
    const root = document.documentElement;
    const tokens = () => {
      const cs = getComputedStyle(root);
      const out = [];
      for (const name of cs) {
        if (name.startsWith('--')) out.push(name + ':' + cs.getPropertyValue(name).trim());
      }
      return out.sort().join(';');
    };
    root.setAttribute('data-theme', 'dark');
    const dark = tokens();
    root.setAttribute('data-theme', 'light');
    const light = tokens();
    root.setAttribute('data-theme', t);
    return { tokenCount: light.split(';').filter(Boolean).length, responded: dark !== light };
  }, theme);
  // The transition has to finish before anything is measured or captured.
  await page.waitForTimeout(400);
  return info;
}

async function runCapture(context, origin, screen, variant, tokens, stubState) {
  const page = await context.newPage();
  const consoleErrors = [];
  page.on('console', msg => { if (msg.type() === 'error') consoleErrors.push(msg.text().slice(0, 200)); });
  page.on('pageerror', err => consoleErrors.push('pageerror: ' + String(err.message).slice(0, 200)));
  page.on('dialog', d => d.dismiss().catch(() => {}));

  stubState.fail = screen.fail || [];
  stubState.overrides = screen.overrides || {};
  stubState.status = screen.status || {};

  const result = {
    id: screen.id, name: screen.name, variant: variant.key, page: screen.page,
    rules: {}, notes: [], consoleErrors: [],
  };

  try {
    await page.goto(origin + PAGE_URLS[screen.page], { waitUntil: 'domcontentloaded', timeout: 30000 });
    const themeInfo = await forceTheme(page, variant.theme);
    result.themeResponded = themeInfo.responded;
    if (!themeInfo.responded && variant.theme === 'light') {
      result.notes.push('Page defines no light palette — the light capture is the dark surface.');
    }
    // Print emulation is applied after the setup steps: several print
    // stylesheets hide the very controls the setup has to drive.
    if (screen.setup) await screen.setup(page, { width: variant.width, theme: variant.theme });
    if (screen.print) {
      await page.emulateMedia({ media: 'print' });
      await page.waitForTimeout(300);
    }
    await page.waitForTimeout(400);
    await page.addScriptTag({ content: CHECK_SOURCE });

    // Element-scoped rules
    result.rules['TOUCH-003'] = await page.evaluate(sel => window.__FL_AUDIT.touchTargets(sel), screen.region || null);
    result.rules['ACCESS-008'] = await page.evaluate(sel => window.__FL_AUDIT.contrast(sel), screen.region || null);

    // Page-scoped rules
    result.rules['LAYOUT-003/ACCESS-001'] = await page.evaluate(() => window.__FL_AUDIT.horizontalOverflow());

    // Category 17 — Status & Data Display. All eight are region-scoped: they
    // ask about what this screen renders, and the region is what the screen is.
    // STATUS-008 needs the variant's CSS width because its clause is desktop-only
    // and window.innerWidth on the zoom variants is the zoomed width, not 1440.
    const sel = screen.region || null;
    result.rules['STATUS-002'] = await page.evaluate(s => window.__FL_AUDIT.nominalBadges(s), sel);
    result.rules['STATUS-004'] = await page.evaluate(s => window.__FL_AUDIT.alarmsPerRow(s), sel);
    result.rules['STATUS-005'] = await page.evaluate(s => window.__FL_AUDIT.explainHooks(s), sel);
    result.rules['STATUS-006'] = await page.evaluate(s => window.__FL_AUDIT.numberFormat(s), sel);
    result.rules['STATUS-007'] = await page.evaluate(s => window.__FL_AUDIT.orphanPlaceholders(s), sel);
    result.rules['STATUS-008'] = await page.evaluate(
      ([s, w]) => window.__FL_AUDIT.rowHeights(s, { width: w }), [sel, variant.width]);
    result.rules['STATUS-010'] = await page.evaluate(s => window.__FL_AUDIT.devVocabulary(s), sel);
    result.rules['STATUS-011'] = await page.evaluate(s => window.__FL_AUDIT.repeatedSentences(s), sel);

    // Screenshot before the occlusion probe scrolls the page to its end.
    // Viewport capture with the screen's region scrolled into view — what the
    // user actually sees at this width, sticky chrome included. An element-clip
    // would have captured content wider than the viewport and painted the
    // fixed bars across the middle of it.
    const shotDir = path.join(SHOT_ROOT, variant.key);
    await fs.mkdir(shotDir, { recursive: true });
    const shotPath = path.join(shotDir, `${screen.id}.png`);
    let framed = false;
    if (screen.region) {
      const loc = page.locator(screen.region).first();
      if (await loc.count() && await loc.isVisible().catch(() => false)) {
        await loc.scrollIntoViewIfNeeded({ timeout: 8000 }).catch(() => {});
        // Clear the sticky stack so the region is not captured behind it.
        await page.evaluate(() => {
          const nav = document.querySelector('.site-nav');
          const header = document.querySelector('.app-header');
          const stack = (nav ? nav.getBoundingClientRect().height : 0) +
                        (header ? header.getBoundingClientRect().height : 0);
          if (stack > 0 && window.scrollY > stack) window.scrollBy(0, -stack);
        }).catch(() => {});
        await page.waitForTimeout(200);
        framed = true;
      } else {
        result.notes.push(`Region \`${screen.region}\` was not visible; captured the page as it stood.`);
      }
    }
    await page.screenshot({ path: shotPath, fullPage: !screen.region, timeout: 25000 });
    result.regionFramed = framed;
    result.screenshot = path.relative(path.dirname(REPORT), shotPath);

    result.rules['LAYOUT-011'] = await page.evaluate(sel => window.__FL_AUDIT.occlusionAtBottom(sel), screen.region || null);

    // LAYOUT-020 — run the app's own refresh and measure what moved.
    result.rules['LAYOUT-020'] = await measureRefreshShift(page, screen);

    result.consoleErrors = consoleErrors.slice(0, 5);
  } catch (err) {
    result.error = String(err && err.message ? err.message : err).slice(0, 300);
  } finally {
    await page.close().catch(() => {});
  }
  return result;
}

async function measureRefreshShift(page, screen) {
  await page.evaluate(() => window.__FL_AUDIT.scrollTop());
  await page.waitForTimeout(200);
  await page.evaluate(() => window.__FL_AUDIT.beginShiftWatch());

  // Prefer the app's own background-refresh callback: that is the refresh the
  // rule is about. Fall back to the explicit Refresh control where a screen
  // has no timer of its own.
  const trigger = await page.evaluate(() => {
    const timers = (window.__FL_INTERVALS || []).filter(t => t.ms >= 20000);
    if (timers.length) {
      timers.forEach(t => { try { t.fn(); } catch (e) {} });
      return 'background interval callback';
    }
    const btn = document.getElementById('refresh-btn');
    if (btn) { btn.click(); return '#refresh-btn'; }
    return null;
  });

  if (!trigger) {
    return { applicable: false, reason: 'No background refresh and no refresh control on this surface.' };
  }
  await page.waitForTimeout(2200);
  const measured = await page.evaluate(() => window.__FL_AUDIT.endShiftWatch());
  return { applicable: true, trigger, ...measured };
}

// ── Verdicts ──────────────────────────────────────────────────────────────
export function verdict(rule, data) {
  if (!data) return { status: 'ERROR', detail: 'not measured' };
  switch (rule) {
    case 'TOUCH-003':
      if (!data.checked) return { status: 'N/A', detail: 'no interactive element in this region' };
      return data.failures === 0
        ? { status: 'PASS', detail: `${data.checked} targets ≥44pt` }
        : { status: 'FAIL', detail: `${data.failures}/${data.checked} under 44pt (${data.failuresNotNested} not inside a larger target)` };
    case 'ACCESS-008':
      if (!data.checked) return { status: 'N/A', detail: 'no rendered text in this region' };
      return data.failures === 0
        ? { status: 'PASS', detail: `${data.checked} text nodes ≥AA` }
        : { status: 'FAIL', detail: `${data.failures}/${data.checked} below AA (worst ${data.worst[0] ? data.worst[0].ratio : '?'}:1)` };
    case 'LAYOUT-003/ACCESS-001':
      return data.overflowPx <= 1
        ? { status: 'PASS', detail: `no document overflow at ${data.viewport}px` }
        : { status: 'FAIL', detail: `${data.overflowPx}px horizontal overflow (${data.offenderCount} element${data.offenderCount === 1 ? '' : 's'})` };
    case 'LAYOUT-011':
      return data.coveredCount === 0
        ? { status: 'PASS', detail: `${data.bars.length} fixed/sticky bar${data.bars.length === 1 ? '' : 's'}; last row of every scroll region reachable` }
        : { status: 'FAIL', detail: `${data.coveredCount} actionable element${data.coveredCount === 1 ? '' : 's'} behind a fixed bar` };
    // ── Category 17 — Status & Data Display ──────────────────────────────
    case 'STATUS-002':
      if (!data.checked) return { status: 'N/A', detail: 'no chip-shaped element in this region' };
      return data.failures === 0
        ? { status: 'PASS', detail: `${data.checked} chips, none a coloured nominal badge` }
        : { status: 'FAIL', detail: `${data.failures}/${data.checked} chips announce a nominal value in colour` };
    case 'STATUS-004':
      if (!data.applicable) return { status: 'N/A', detail: data.reason };
      if (!data.checked) return { status: 'N/A', detail: 'no list row in this region' };
      return data.failures === 0
        ? { status: 'PASS', detail: `${data.checked} rows, at most one alarm each (worst ${data.maxAlarms})` }
        : { status: 'FAIL', detail: `${data.failures}/${data.checked} rows carry more than one alarm (worst ${data.maxAlarms})` };
    case 'STATUS-005':
      if (!data.checked) return { status: 'N/A', detail: 'no chip-shaped element in this region' };
      return data.failures === 0
        ? { status: 'PASS', detail: `${data.checked} chips carry the data-explain hook` }
        : { status: 'FAIL', detail: `${data.failures}/${data.checked} chips have no explanation hook (${data.titleOnly} rely on title=)` };
    case 'STATUS-006': {
      if (!data.checked && !data.alignedCells) return { status: 'N/A', detail: 'no rendered number in this region' };
      if (data.failures === 0) return { status: 'PASS', detail: `${data.checked} numbers, all formatted` };
      const kinds = Object.entries(data.byKind || {}).map(([k, n]) => `${k} ${n}`).join(', ');
      return { status: 'FAIL', detail: `${data.failures} formatting failures (${kinds})` };
    }
    case 'STATUS-007':
      return data.failures === 0
        ? { status: 'PASS', detail: 'no dash-only element outside a table cell' }
        : { status: 'FAIL', detail: `${data.failures} orphan placeholder${data.failures === 1 ? '' : 's'}` };
    case 'STATUS-008':
      if (!data.applicable) return { status: 'N/A', detail: data.reason };
      if (!data.checked) return { status: 'N/A', detail: 'no list row in this region' };
      return data.failures === 0
        ? { status: 'PASS', detail: `${data.checked} rows, tallest ${data.tallest}px` }
        : { status: 'FAIL', detail: `${data.failures}/${data.checked} rows over 56px (tallest ${data.tallest}px)` };
    case 'STATUS-010':
      return data.failures === 0
        ? { status: 'PASS', detail: `${data.checked} text elements, no implementation vocabulary` }
        : { status: 'FAIL', detail: `${data.failures} element${data.failures === 1 ? '' : 's'} use developer vocabulary` };
    case 'STATUS-011':
      return data.failures === 0
        ? { status: 'PASS', detail: `${data.checked} sentences, none repeated` }
        : { status: 'FAIL', detail: `${data.failures} sentence${data.failures === 1 ? '' : 's'} rendered more than once (${data.repeatedInstances} extra renders)` };
    case 'LAYOUT-020': {
      if (!data.applicable) return { status: 'N/A', detail: data.reason };
      if (!data.clsSupported) return { status: 'ERROR', detail: 'layout-shift observer unavailable' };
      if (data.cls <= 0.1 && data.moved === 0) return { status: 'PASS', detail: `CLS ${data.cls}, nothing moved (${data.trigger})` };
      if (data.cls <= 0.25) return { status: 'WARN', detail: `CLS ${data.cls}, ${data.moved} element(s) moved, max ${data.maxDeltaPx}px` };
      return { status: 'FAIL', detail: `CLS ${data.cls}, ${data.moved} element(s) moved, max ${data.maxDeltaPx}px` };
    }
    default:
      return { status: 'ERROR', detail: 'unknown rule' };
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.reportOnly) {
    const raw = JSON.parse(await fs.readFile(path.join(SHOT_ROOT, 'results.json'), 'utf8'));
    const md = writeReport({
      results: raw.results, screens: SCREENS, variants: raw.variants || VARIANTS,
      rules: raw.rules || RULES, verdict, zoomScreens: ZOOM_SCREENS,
    });
    await fs.writeFile(REPORT, md);
    console.log(`Re-rendered ${path.relative(REPO, REPORT)} from ${raw.results.length} stored captures.`);
    return;
  }

  const tokens = buildTokenTable(new Date());

  const screens = SCREENS.filter(s => !args.screens || args.screens.has(s.id));
  const variants = VARIANTS.filter(v => !args.variants || args.variants.has(v.key));

  const jobs = [];
  for (const variant of variants) {
    for (const screen of screens) {
      if (appliesTo(screen, variant)) jobs.push({ screen, variant });
    }
  }

  console.log(`Factory Ledger visual audit — ${screens.length} screens, ${variants.length} variants, ${jobs.length} captures.`);
  await fs.rm(SHOT_ROOT, { recursive: true, force: true }).catch(() => {});

  const server = await startStaticServer(DASHBOARD);
  const browser = await chromium.launch({ headless: !args.headed });
  const results = [];
  const unmatched = new Set();

  let cursor = 0;
  let done = 0;
  const worker = async () => {
    const stubState = { fail: [], overrides: {}, status: {}, unmatched };
    const contexts = new Map();
    while (true) {
      const index = cursor++;
      if (index >= jobs.length) break;
      const { screen, variant } = jobs[index];
      if (!contexts.has(variant.key)) {
        const ctx = await newContext(browser, variant);
        await installApiStub(ctx, tokens, stubState);
        contexts.set(variant.key, ctx);
      }
      const res = await runCapture(contexts.get(variant.key), server.origin, screen, variant, tokens, stubState);
      results.push(res);
      done++;
      if (done % 10 === 0 || done === jobs.length) {
        process.stdout.write(`  ${done}/${jobs.length} captures\n`);
      }
    }
    for (const ctx of contexts.values()) await ctx.close().catch(() => {});
  };

  await Promise.all(Array.from({ length: Math.max(1, args.concurrency) }, worker));

  await browser.close();
  await server.close();

  results.sort((a, b) =>
    a.id.localeCompare(b.id) ||
    VARIANTS.findIndex(v => v.key === a.variant) - VARIANTS.findIndex(v => v.key === b.variant));

  await fs.mkdir(SHOT_ROOT, { recursive: true });
  await fs.writeFile(path.join(SHOT_ROOT, 'results.json'), JSON.stringify({
    generatedAt: new Date().toISOString(),
    variants, rules: RULES, results,
    unmatchedEndpoints: [...unmatched],
  }, null, 2));

  const md = writeReport({ results, screens: SCREENS, variants, rules: RULES, verdict, zoomScreens: ZOOM_SCREENS });
  await fs.writeFile(REPORT, md);

  const counts = { PASS: 0, FAIL: 0, WARN: 0, 'N/A': 0, ERROR: 0 };
  for (const r of results) {
    for (const rule of RULES) {
      const v = verdict(rule, r.rules[rule]);
      counts[v.status] = (counts[v.status] || 0) + 1;
    }
  }
  console.log('\nRule results across all captures:', counts);
  if (unmatched.size) console.log('Unmatched endpoints (no fixture):', [...unmatched].join(', '));
  console.log(`Matrix:      ${path.relative(REPO, REPORT)}`);
  console.log(`Screenshots: ${path.relative(REPO, SHOT_ROOT)}/<variant>/<S-id>.png`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
