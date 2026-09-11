// Sales Orders exit regressions in real native dialogs, using the shipped
// module/styles and representative API responses. No request leaves the page.
// Run: node tests/visual/run-so-exit-actions.mjs [dashboard-root]
import assert from 'node:assert/strict';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { startStaticServer } from './lib/server.mjs';
import { installApiStub, buildTokenTable, API_HOST } from './lib/stub.mjs';
import { SCREENS } from './lib/screens.mjs';

const dashboard = path.resolve(process.argv[2] || fileURLToPath(new URL('../../dashboard/', import.meta.url)));
const browser = await chromium.launch();
const results = [];

async function prepare(page, theme) {
  await page.setContent(`<html data-theme="${theme}"><body>
    <button id="refresh-btn">Refresh</button><button id="trigger">Close…</button>
  </body></html>`);
  await page.addStyleTag({ path: path.join(dashboard, 'dashboard.css') });
  await page.addStyleTag({ path: path.join(dashboard, 'so-list-actions.css') });
  await page.addScriptTag({ path: path.join(dashboard, 'so-list-actions.js') });
  await page.evaluate(() => {
    window.testState = { calls: [], commits: 0, refreshes: 0, scenario: '' };
    window.beginExit = (action, state = 'open', scenario = '') => {
      Object.assign(window.testState, { calls: [], commits: 0, refreshes: 0, scenario });
      return window.SOListActions.open({ id: 12, order_number: 'SO-26-012', state }, {
        action,
        format: number => new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(number),
        trigger: document.getElementById('trigger'),
        request: async (url, config) => {
          const body = config ? JSON.parse(config.body) : null;
          const test = window.testState;
          test.calls.push({ url, body });
          const apiError = status => Object.assign(new Error('HTTP RAW_DATABASE_SECRET'), {
            status, detail: { message: '{"error":"RAW_DATABASE_SECRET"}' },
          });
          if (/^error-/.test(test.scenario)) throw apiError(Number(test.scenario.slice(6)));
          if (test.scenario === 'network') throw new TypeError('RAW_DATABASE_SECRET');
          if (!config) {
            if (test.scenario === 'related-missing') throw apiError(404);
            return test.scenario === 'related-self'
              ? { order_id: 12, order_number: 'SO-26-012' }
              : { order_id: 44, order_number: 'SO-26-044' };
          }
          if (['cancel-409', 'legacy-cancel-409'].includes(test.scenario) && url.endsWith('/cancel')) {
            const detail = {
              error_code: 'ORDER_ALREADY_SHIPPED', suggested_action: 'close',
              suggested_reason: 'short_closed', message: 'RAW_DATABASE_SECRET',
            };
            throw test.scenario === 'legacy-cancel-409' ? { status: 409, payload: { detail } } : { status: 409, detail };
          }
          if (body.mode === 'commit') {
            test.commits++;
            if (test.scenario === 'commit-failure') throw apiError(500);
            if (test.scenario === 'delayed-commit') await new Promise(resolve => { window.resolveCommit = resolve; });
            return { mode: 'commit' };
          }
          if (test.scenario === 'incomplete-preview') return { mode: 'preview' };
          return {
            mode: 'preview',
            reservations_to_release: url.endsWith('/reopen') ? [] : [
              { id: 101, quantity_lb: 1234.5 }, { id: 102, quantity_lb: 765.5 },
            ],
          };
        },
        onCommitted: async () => {
          testState.refreshes++;
          if (testState.scenario === 'refresh-failure') throw new Error('RAW_DATABASE_SECRET');
          if (testState.scenario === 'rerender-without-tabs') document.getElementById('trigger').remove();
        },
      });
    };
  });
}

async function begin(page, action, state = 'open', scenario = '') {
  await page.evaluate(args => window.beginExit(...args), [action, state, scenario]);
  await page.locator('.so-exit-dialog[open]').waitFor();
}

async function dismiss(page) {
  await page.keyboard.press('Escape');
  await page.locator('.so-exit-dialog').waitFor({ state: 'detached' });
}

async function preview(page, action) {
  await page.locator('.so-exit-dialog').getByRole('button', { name: `Preview ${action}`, exact: true }).click();
  await page.waitForFunction(() => document.querySelector('.so-exit-dialog').getAttribute('aria-busy') === 'false');
}

async function commit(page, action) {
  await page.locator('.so-exit-dialog').getByRole('button', { name: `${action} order`, exact: true }).click();
  await page.locator('.so-exit-dialog').waitFor({ state: 'detached' });
}

async function runChecks(page, variant) {
  async function check(name, run) {
    await run();
    results.push({ variant, check: name, passed: true });
  }

  await check('Preview release totals; edits invalidate preview; explicit commit refreshes and restores focus', async () => {
    await begin(page, 'close');
    assert.equal(await page.getByRole('dialog', { name: 'Close SO-26-012', exact: true }).count(), 1);
    assert.equal(await page.evaluate(() => document.activeElement.id), 'so-exit-reason');
    assert.equal(await page.getByRole('button', { name: 'Close order', exact: true }).count(), 0);
    await preview(page, 'close');
    assert.equal(await page.locator('.so-exit-release-summary').textContent(), 'Will release 2 reservations (2,000 lb)');
    assert.equal(await page.evaluate(() => testState.commits), 0);
    await page.getByLabel('Note (optional)', { exact: true }).fill('Final inventory counted');
    assert.equal(await page.getByRole('button', { name: 'Close order', exact: true }).count(), 0);
    await preview(page, 'close');
    await commit(page, 'Close');
    assert.equal(await page.evaluate(() => document.activeElement.id), 'trigger');
    const test = await page.evaluate(() => testState);
    assert.deepEqual(test.calls.map(call => call.body.mode), ['preview', 'preview', 'commit']);
    assert.equal(test.calls[2].body.note, 'Final inventory counted');
    assert.equal(test.calls.some(call => call.body && 'changed_by' in call.body), false);
    assert.equal(test.refreshes, 1);
  });

  await check('Other requires a nonblank note before any request', async () => {
    await begin(page, 'cancel');
    await page.getByLabel('Reason', { exact: true }).selectOption('other');
    await page.getByLabel('Note (required)', { exact: true }).fill('   ');
    await preview(page, 'cancel');
    assert.equal(await page.evaluate(() => testState.calls.length), 0);
    await page.getByLabel('Note (required)', { exact: true }).fill('Requested by customer');
    await preview(page, 'cancel');
    assert.equal(await page.evaluate(() => testState.calls[0].body.note), 'Requested by customer');
    await dismiss(page);
  });

  for (const reason of ['duplicate', 'superseded']) {
    await check(`${reason} requires a related SO and sends its resolved ID`, async () => {
      await begin(page, 'cancel');
      await page.getByLabel('Reason', { exact: true }).selectOption(reason);
      await preview(page, 'cancel');
      assert.equal(await page.evaluate(() => testState.calls.length), 0);
      await page.getByLabel('Related SO', { exact: true }).fill('SO-26-044');
      await preview(page, 'cancel');
      const calls = await page.evaluate(() => testState.calls);
      assert.equal(calls[0].url, '/sales/orders/SO-26-044');
      assert.equal(calls[1].body.related_so_id, 44);
      // Enter on the form never acts as an implicit cancellation confirmation.
      await page.getByLabel('Reason', { exact: true }).focus();
      await page.keyboard.press('Enter');
      assert.equal(await page.evaluate(() => testState.commits), 0);
      await commit(page, 'Cancel');
      assert.equal(await page.evaluate(() => testState.calls.at(-1).body.related_so_id), 44);
    });
  }

  for (const scenario of ['related-self', 'related-missing']) {
    await check(`${scenario} is rejected in plain language before an exit request`, async () => {
      await begin(page, 'cancel', 'open', scenario);
      await page.getByLabel('Reason', { exact: true }).selectOption('duplicate');
      await page.getByLabel('Related SO', { exact: true }).fill('SO-26-012');
      await preview(page, 'cancel');
      assert.equal(await page.evaluate(() => testState.calls.length), 1);
      assert.equal(await page.evaluate(() => testState.calls[0].body), null);
      assert.match(await page.getByRole('alert').textContent(), scenario === 'related-self' ? /different sales order/ : /not found/);
      assert.equal(await page.getByRole('button', { name: 'Cancel order', exact: true }).count(), 0);
      await dismiss(page);
    });
  }

  for (const scenario of ['cancel-409', 'legacy-cancel-409']) {
    await check(`${scenario} offers the suggested close reason and previews without committing`, async () => {
      await begin(page, 'cancel', 'open', scenario);
      await preview(page, 'cancel');
      assert.match(await page.getByRole('alert').textContent(), /already shipped/);
      await page.getByRole('button', { name: 'Preview close instead · Short-closed', exact: true }).click();
      await page.getByRole('button', { name: 'Close order', exact: true }).waitFor();
      assert.equal(await page.getByRole('dialog', { name: 'Close SO-26-012', exact: true }).count(), 1);
      assert.equal(await page.getByLabel('Reason', { exact: true }).inputValue(), 'short_closed');
      const calls = await page.evaluate(() => testState.calls);
      assert.deepEqual(calls.map(call => call.body.mode), ['preview', 'preview']);
      assert.equal(calls[1].url, '/sales/orders/12/close');
      assert.equal(calls[1].body.reason, 'short_closed');
      assert.equal(await page.evaluate(() => testState.commits), 0);
      await dismiss(page);
    });
  }

  for (const state of ['closed', 'cancelled']) {
    await check(`Reopening ${state} order explains reservation loss and previews before commit`, async () => {
      await begin(page, 'reopen', state);
      assert.match(await page.locator('.so-exit-description').textContent(), /reservations are not restored/);
      await preview(page, 'reopen');
      assert.equal(await page.locator('.so-exit-release-summary').textContent(), 'Will release 0 reservations (0 lb)');
      assert.equal(await page.evaluate(() => 'reason' in testState.calls[0].body), false);
      await commit(page, 'Reopen');
      assert.deepEqual(await page.evaluate(() => testState.calls.map(call => call.body.mode)), ['preview', 'commit']);
    });
  }

  await check('API and connection errors never expose raw response or exception text', async () => {
    for (const scenario of ['error-400', 'error-401', 'error-403', 'error-404', 'error-409', 'error-422', 'error-500', 'error-503', 'network']) {
      await begin(page, 'close', 'open', scenario);
      await preview(page, 'close');
      const error = await page.getByRole('alert').textContent();
      assert(error.trim().length > 25, `${scenario} needs an actionable message`);
      assert.doesNotMatch(await page.locator('.so-exit-dialog').textContent(), /RAW_DATABASE_SECRET|HTTP|\{"error"/);
      if (scenario === 'error-503') assert.match(error, /temporarily unavailable/);
      assert.equal(await page.getByRole('button', { name: 'Close order', exact: true }).count(), 0);
      await dismiss(page);
    }
  });

  await check('Missing preview totals cannot be mistaken for zero releases', async () => {
    await begin(page, 'close', 'open', 'incomplete-preview');
    await preview(page, 'close');
    assert.equal(await page.getByRole('button', { name: 'Close order', exact: true }).count(), 0);
    assert.equal(await page.locator('.so-exit-preview').isVisible(), false);
    assert.equal(await page.getByRole('alert').isVisible(), true);
    await dismiss(page);
  });

  await check('A failed commit requires a new preview and does not refresh as if saved', async () => {
    await begin(page, 'close', 'open', 'commit-failure');
    await preview(page, 'close');
    await page.getByRole('button', { name: 'Close order', exact: true }).click();
    await page.getByRole('button', { name: 'Preview close', exact: true }).waitFor();
    assert.equal(await page.getByRole('button', { name: 'Close order', exact: true }).count(), 0);
    assert.match(await page.getByRole('alert').textContent(), /Refresh the list before trying again/);
    assert.equal(await page.evaluate(() => testState.refreshes), 0);
    await dismiss(page);
  });

  await check('An in-flight commit cannot repeat or be dismissed', async () => {
    await begin(page, 'close', 'open', 'delayed-commit');
    await preview(page, 'close');
    await page.getByRole('button', { name: 'Close order', exact: true }).click();
    assert.equal(await page.locator('.so-exit-dialog').getAttribute('aria-busy'), 'true');
    const committing = page.getByRole('button', { name: 'Close in progress…', exact: true });
    assert.equal(await committing.isDisabled(), true);
    // Even a dispatched repeat event cannot create another request.
    await committing.dispatchEvent('click');
    assert.equal(await page.evaluate(() => testState.commits), 1);
    await page.keyboard.press('Escape');
    assert.equal(await page.locator('.so-exit-dialog[open]').count(), 1);
    await page.evaluate(() => window.resolveCommit());
    await page.locator('.so-exit-dialog').waitFor({ state: 'detached' });
    assert.equal(await page.evaluate(() => testState.refreshes), 1);
  });

  await check('Refresh failure preserves the saved result and cannot commit twice', async () => {
    await begin(page, 'close', 'open', 'refresh-failure');
    await preview(page, 'close');
    await page.getByRole('button', { name: 'Close order', exact: true }).click();
    await page.getByRole('button', { name: 'Dismiss', exact: true }).waitFor();
    assert.match(await page.getByRole('alert').textContent(), /Order closed\. The list could not refresh/);
    assert.equal(await page.getByRole('button', { name: 'Close order', exact: true }).count(), 0);
    assert.equal(await page.getByRole('button', { name: 'Preview close', exact: true }).count(), 0);
    assert.equal(await page.evaluate(() => testState.commits), 1);
    await dismiss(page);
  });

  await check('Dialog fits the viewport with labeled 44px controls and native keyboard containment', async () => {
    await begin(page, 'cancel');
    await page.getByLabel('Reason', { exact: true }).selectOption('duplicate');
    const controls = page.locator('.so-exit-dialog :is(button,input,select,textarea):visible');
    const sizes = await controls.evaluateAll(nodes => nodes.map(node => {
      const rect = node.getBoundingClientRect();
      return { width: rect.width, height: rect.height };
    }));
    assert(sizes.length >= 5);
    assert(sizes.every(size => size.width >= 44 && size.height >= 44));
    assert(await page.locator('.so-exit-dialog').evaluate(node => {
      const rect = node.getBoundingClientRect();
      return node.scrollWidth <= node.clientWidth && rect.left >= 0 && rect.right <= innerWidth && rect.top >= 0 && rect.bottom <= innerHeight;
    }));
    await page.getByLabel('Reason', { exact: true }).focus();
    await page.keyboard.press('Tab');
    assert.equal(await page.evaluate(() => document.activeElement.id), 'so-exit-related');
    await page.keyboard.press('Shift+Tab');
    assert.equal(await page.evaluate(() => document.activeElement.id), 'so-exit-reason');
    // Native modal inertness also prevents focus from escaping to the list.
    await page.evaluate(() => document.getElementById('trigger').focus());
    assert(await page.locator('.so-exit-dialog').evaluate(node => node.contains(document.activeElement)));
    await dismiss(page);
    assert.equal(await page.evaluate(() => document.activeElement.id), 'trigger');
  });

  await check('Removed trigger falls back to global Refresh when no selected Sales Orders tab exists', async () => {
    await begin(page, 'close', 'open', 'rerender-without-tabs');
    await preview(page, 'close');
    await commit(page, 'Close');
    assert.equal(await page.evaluate(() => document.activeElement.id), 'refresh-btn');
  });
}

async function runIntegratedFocusChecks() {
  const server = await startStaticServer(dashboard);
  try {
    for (const width of [1440, 390]) {
      const context = await browser.newContext({ viewport: { width, height: 844 } });
      try {
        // Only the local dashboard is served. The API adapter and real
        // onCommitted refresh run unchanged against in-memory responses.
        await context.route('**/*', route => new URL(route.request().url()).origin === server.origin
          ? route.continue() : route.abort());
        await installApiStub(context, buildTokenTable(), { fail: [], overrides: {}, status: {} });
        const writes = [];
        await context.route(`https://${API_HOST}/sales/orders/*/close`, async route => {
          const body = route.request().postDataJSON();
          writes.push(body);
          await route.fulfill({ json: body.mode === 'preview'
            ? { mode: 'preview', reservations_to_release: [{ id: 401, quantity_lb: 1200 }] }
            : { mode: 'commit', state: 'closed' } });
        });
        const page = await context.newPage();
        await page.goto(server.origin);
        await SCREENS.find(screen => screen.id === 'S-25').setup(page);
        await page.locator('.order-expand-toggle').first().click();
        const trigger = page.locator('.so-exit-action[data-action="close"]').first();
        await trigger.waitFor();
        await trigger.evaluate(node => { window.exitTriggerBeforeCommit = node; });
        await trigger.click();
        await preview(page, 'close');
        assert.equal(await page.locator('.so-exit-release-summary').textContent(), 'Will release 1 reservations (1,200 lb)');
        await commit(page, 'Close');
        await page.waitForFunction(() => !window.exitTriggerBeforeCommit.isConnected);
        assert(await page.evaluate(() => document.activeElement === document.querySelector('[data-orders-tab][aria-selected="true"]')),
          `${width}px integrated refresh must restore focus to the selected Sales Orders tab`);
        assert.deepEqual(writes.map(body => body.mode), ['preview', 'commit']);
        assert.equal(writes.some(body => 'changed_by' in body), false);
        results.push({ variant: `${width}-integrated`, check: 'Real list commit rerenders and restores focus to selected tab', passed: true });
        console.log(`PASS ${width}-integrated: commit rerender restores selected-tab focus`);
      } finally {
        await context.close();
      }
    }
  } finally {
    await server.close();
  }
}

try {
  for (const width of [1440, 390]) {
    for (const theme of ['light', 'dark']) {
      const variant = `${width}-${theme}`;
      const context = await browser.newContext({ viewport: { width, height: 844 }, colorScheme: theme });
      // The module receives its request adapter directly. Abort any unexpected
      // network use so this test can never write to the real sales API.
      await context.route('**/*', route => route.abort());
      try {
        const page = await context.newPage();
        const pageErrors = [];
        page.on('pageerror', error => pageErrors.push(error.message));
        await prepare(page, theme);
        await runChecks(page, variant);
        assert.deepEqual(pageErrors, [], `${variant}: unexpected browser errors`);
        console.log(`PASS ${variant}: ${results.filter(result => result.variant === variant).length} exit-action checks`);
      } finally {
        await context.close();
      }
    }
  }
  await runIntegratedFocusChecks();
  console.log(`PASS: ${results.length} Sales Orders exit-action checks; all API calls stubbed in memory.`);
} finally {
  await browser.close();
}
