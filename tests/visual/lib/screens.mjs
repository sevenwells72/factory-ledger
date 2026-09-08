// The 91 screens of docs/design/audit/00-screen-inventory.md, each with the
// recipe that puts the app into that state. `mobile` mirrors the inventory's
// Mobile column verbatim and decides whether the screen is captured at 390px.
//
//   region      CSS selector the screenshot is clipped to and the element-scoped
//               rules (TOUCH-003, ACCESS-008) are measured within. null = page.
//   setup       async (page, ctx) — drives the app into the state.
//   capturable  false for browser-native dialogs, which render outside the page.

const T = { short: 250, med: 600, long: 1200 };

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function ready(page) {
  await page.waitForFunction(
    () => {
      const el = document.getElementById('last-refreshed');
      return el && el.textContent.trim().length > 0;
    },
    { timeout: 20000 },
  ).catch(() => {});
  await sleep(T.med);
}

async function tab(page, name) {
  await ready(page);
  await page.click(`.tab[data-tab="${name}"]`);
  await sleep(T.med);
}

async function clickIf(page, selector) {
  const el = page.locator(selector).first();
  if (await el.count() && await el.isVisible().catch(() => false)) {
    await el.click({ timeout: 4000 }).catch(() => {});
    await sleep(T.short);
    return true;
  }
  return false;
}

async function expandPanel(page, index = 0) {
  const headers = page.locator('#finished-goods-panels .collapsible-header');
  if (await headers.count() > index) {
    const h = headers.nth(index);
    if (!(await h.evaluate(el => el.classList.contains('expanded')))) {
      await h.click().catch(() => {});
      await sleep(T.short);
    }
  }
}

async function openFirstOrder(page) {
  await tab(page, 'orders');
  await page.locator('.order-row').first().click({ timeout: 6000 }).catch(() => {});
  await page.waitForSelector('.order-detail-header', { timeout: 8000 }).catch(() => {});
  await sleep(T.med);
}

async function openOrderById(page, orderId) {
  await tab(page, 'orders');
  // The default filter is "All Open Orders"; a closed order is only in the list
  // once the filter is widened.
  await page.selectOption('#orders-status-filter', 'all').catch(() => {});
  await page.waitForSelector(`.order-row[data-order-id="${orderId}"]`, { timeout: 8000 }).catch(() => {});
  await page.locator(`.order-row[data-order-id="${orderId}"]`).first().click({ timeout: 6000 }).catch(() => {});
  await page.waitForSelector('.order-detail-header', { timeout: 8000 }).catch(() => {});
  await sleep(T.med);
}

async function openSupplies(page) {
  await tab(page, 'supplies');
  await page.waitForSelector('.supply-item-row', { timeout: 8000 }).catch(() => {});
}

async function traceALot(page) {
  // An ingredient lot: the forward trace's first-class case (ingredient →
  // batches → shipments → customers), which exercises the most of the graph.
  await page.fill('#lotSearch', 'OAT-4471');
  await sleep(T.med);
  await page.click('#traceBtn').catch(() => {});
  await page.waitForFunction(() => {
    const svg = document.getElementById('graphSvg');
    return svg && svg.querySelectorAll('g, rect, circle').length > 2;
  }, { timeout: 12000 }).catch(() => {});
  await sleep(T.long);
}

async function schedulerSample(page) {
  await page.waitForSelector('#btn-sample', { timeout: 10000 }).catch(() => {});
  await clickIf(page, '#btn-sample');
  await sleep(T.long);
}

async function openDetails(page, summaryText) {
  await page.evaluate(text => {
    for (const d of document.querySelectorAll('details')) {
      const s = d.querySelector('summary');
      if (s && s.textContent.includes(text)) d.open = true;
    }
  }, summaryText);
  await sleep(T.short);
}

export const SCREENS = [
  // ── A. Shared chrome ────────────────────────────────────────────────────
  { id: 'S-01', name: 'Site navigation bar', page: 'index', mobile: 'yes', region: '.site-nav',
    setup: async (page, ctx) => { await ready(page); if (ctx.width < 769) await clickIf(page, '#navToggle'); } },
  { id: 'S-02', name: 'Mini-calendar strip (3-month)', page: 'index', mobile: 'partial', region: '[data-mini-calendar]',
    setup: ready },
  // IMP-069: below 520px the strip is a collapsed row; this is the expanded
  // month, the only state that adds width to the header. Above 520px there is
  // no toggle and the capture is S-02 again.
  { id: 'S-02b', name: 'Mini-calendar — expanded month (phone)', page: 'index', mobile: 'yes', region: '[data-mini-calendar]',
    setup: async page => { await ready(page); await clickIf(page, '.mini-calendar-toggle'); } },

  // ── B. Factory Dashboard — page chrome ──────────────────────────────────
  { id: 'S-03', name: 'App header', page: 'index', mobile: 'yes', region: '.app-header', setup: ready },
  { id: 'S-04', name: 'Global search field + results dropdown', page: 'index', mobile: 'yes', region: '.header-center',
    setup: async page => {
      await ready(page);
      await page.fill('#global-search', 'granola');
      await page.waitForSelector('#search-results:not(.hidden)', { timeout: 6000 }).catch(() => {});
      await sleep(T.med);
    } },
  { id: 'S-05', name: 'Tab bar (7 tabs)', page: 'index', mobile: 'partial', region: '.tab-bar', setup: ready },

  // ── C. Tab 1 — Operations ───────────────────────────────────────────────
  { id: 'S-06', name: 'Today So Far tile', page: 'index', mobile: 'yes', region: '.today-tile-section', setup: ready },
  { id: 'S-07', name: 'Today So Far — error/retry state', page: 'index', mobile: 'yes', region: '.today-tile-section',
    fail: ['/production/today-tile'],
    setup: async page => { await ready(page); await page.waitForSelector('.today-tile-error', { timeout: 8000 }).catch(() => {}); } },
  { id: 'S-08', name: 'Production Calendar (rolling 5-day)', page: 'index', mobile: 'partial', region: '#section-production',
    setup: ready },
  { id: 'S-09', name: 'Production Calendar — day detail panel', page: 'index', mobile: 'partial', region: '#section-production',
    setup: async page => {
      await ready(page);
      await clickIf(page, '.day-card-trigger');
      await sleep(T.med);
    } },
  { id: 'S-10', name: 'Finished Goods On-Hand — collapsible panels', page: 'index', mobile: 'partial', region: '#section-finished-goods',
    setup: async page => { await ready(page); await expandPanel(page, 0); await expandPanel(page, 1); } },
  { id: 'S-11', name: 'Finished Goods — per-product lot breakdown', page: 'index', mobile: 'partial', region: '#section-finished-goods',
    setup: async page => {
      await ready(page); await expandPanel(page, 0);
      await clickIf(page, '#section-finished-goods tr.expandable');
    } },
  { id: 'S-12', name: 'Batch Inventory On-Hand', page: 'index', mobile: 'partial', region: '#section-batches',
    setup: async page => { await ready(page); await clickIf(page, '#section-batches tr.expandable'); } },
  { id: 'S-13', name: 'On-Hand Ingredients', page: 'index', mobile: 'partial', region: '#section-ingredients',
    setup: async page => {
      await ready(page);
      await clickIf(page, '#ingredients-panels .collapsible-header');
      await clickIf(page, '#section-ingredients tr.expandable');
    } },

  // ── D. Tab 2 — Recent Entries ───────────────────────────────────────────
  { id: 'S-14', name: 'Recent Entries feed', page: 'index', mobile: 'yes', region: '#tab-recent',
    setup: async page => { await tab(page, 'recent'); await page.waitForSelector('.recent-entry-card', { timeout: 8000 }).catch(() => {}); } },
  { id: 'S-15', name: 'Recent Entries — loading / empty / error states', page: 'index', mobile: 'yes', region: '#tab-recent',
    fail: ['/ledger/recent'],
    setup: async page => { await tab(page, 'recent'); await page.waitForSelector('.recent-entries-error', { timeout: 8000 }).catch(() => {}); } },

  // ── E. Tab 3 — Activity ─────────────────────────────────────────────────
  { id: 'S-16', name: 'Daily Entries + day/mode toolbar', page: 'index', mobile: 'partial', region: '#section-daily-entries',
    setup: async page => { await tab(page, 'activity'); } },
  { id: 'S-17', name: 'Shipping log', page: 'index', mobile: 'partial', region: '#section-shipments',
    setup: async page => { await tab(page, 'activity'); await clickIf(page, '#section-shipments tr.expandable'); } },
  { id: 'S-18', name: 'Receiving log', page: 'index', mobile: 'partial', region: '#section-receipts',
    setup: async page => { await tab(page, 'activity'); await clickIf(page, '#section-receipts tr.expandable'); } },

  // ── F. Tab 4 — Notes / To-Dos / Reminders ───────────────────────────────
  { id: 'S-19', name: 'Notes toolbar', page: 'index', mobile: 'yes', region: '.notes-toolbar',
    setup: async page => { await tab(page, 'notes'); } },
  { id: 'S-20', name: 'Notes list (cards)', page: 'index', mobile: 'yes', region: '#notes-list',
    setup: async page => { await tab(page, 'notes'); await page.waitForSelector('.note-card', { timeout: 8000 }).catch(() => {}); } },
  { id: 'S-21', name: 'Notes — empty state', page: 'index', mobile: 'yes', region: '#notes-list',
    overrides: { '/dashboard/api/notes': 'notes-empty.json' },
    setup: async page => { await tab(page, 'notes'); await page.waitForSelector('.notes-empty', { timeout: 8000 }).catch(() => {}); } },
  { id: 'S-22', name: 'Dialog — Note create/edit modal', page: 'index', mobile: 'yes', region: '#note-modal-overlay',
    setup: async page => {
      await tab(page, 'notes');
      await clickIf(page, '#notes-add-btn');
      await page.waitForSelector('#note-modal-overlay:not(.hidden)', { timeout: 6000 }).catch(() => {});
    } },
  { id: 'S-23', name: 'Dialog — Delete note confirmation', page: 'index', mobile: 'yes', capturable: false,
    note: 'Browser-native window.confirm — rendered by the browser, not the page. Nothing in the document can be measured.' },

  // ── G. Tab 5 — Sales Orders ─────────────────────────────────────────────
  { id: 'S-24', name: 'Orders toolbar', page: 'index', mobile: 'yes', region: '#tab-orders .orders-toolbar',
    setup: async page => { await tab(page, 'orders'); } },
  { id: 'S-25', name: 'Orders list table', page: 'index', mobile: 'partial', region: '#section-orders',
    setup: async page => { await tab(page, 'orders'); await page.waitForSelector('.order-row', { timeout: 8000 }).catch(() => {}); } },
  { id: 'S-26', name: 'Orders list — Dispatch Queue mode', page: 'index', mobile: 'partial', region: '#orders-list-view',
    setup: async page => {
      await tab(page, 'orders');
      await page.selectOption('#orders-status-filter', 'dispatch_queue');
      await page.waitForSelector('.order-row', { timeout: 8000 }).catch(() => {});
      await sleep(T.med);
    } },
  { id: 'S-27', name: 'Orders list — expandable line rows', page: 'index', mobile: 'partial', region: '#section-orders',
    setup: async page => {
      await tab(page, 'orders');
      await page.waitForSelector('.order-expand-toggle', { timeout: 8000 }).catch(() => {});
      await clickIf(page, '.order-expand-toggle');
      await sleep(T.med);
    } },
  { id: 'S-28', name: 'Orders list — Factory Ready toggle + note', page: 'index', mobile: 'partial', region: '#section-orders',
    setup: async page => {
      await tab(page, 'orders');
      await page.waitForSelector('.order-expand-toggle', { timeout: 8000 }).catch(() => {});
      await clickIf(page, '.order-expand-toggle');
      await page.waitForSelector('.order-ready-drawer', { timeout: 6000 }).catch(() => {});
    } },
  { id: 'S-29', name: 'Orders list — empty state', page: 'index', mobile: 'yes', region: '#section-orders',
    setup: async page => {
      await tab(page, 'orders');
      await page.fill('#orders-customer-search', 'zzzz-no-such-customer');
      await page.locator('#orders-customer-search').press('Enter').catch(() => {});
      await sleep(T.med);
    } },
  { id: 'S-30', name: 'Order Detail — header, status, dates, KPI row', page: 'index', mobile: 'yes', region: '#order-detail-view',
    setup: openFirstOrder },
  { id: 'S-31', name: 'Order Detail — line readiness table', page: 'index', mobile: 'yes', region: '.order-detail-table-wrap',
    setup: openFirstOrder },
  { id: 'S-32', name: 'Order Detail — per-line inventory expander', page: 'index', mobile: 'yes', region: '#order-detail-container',
    setup: async page => { await openFirstOrder(page); await clickIf(page, '.order-inventory-toggle'); await sleep(T.med); } },
  { id: 'S-33', name: 'Form — Order Detail edit mode', page: 'index', mobile: 'yes', region: '#order-detail-view',
    setup: async page => {
      await openFirstOrder(page);
      await clickIf(page, '.order-edit-toggle-btn');
      await sleep(T.med);
    } },
  { id: 'S-34', name: 'Order Detail — edit-locked notice', page: 'index', mobile: 'yes', region: '#order-detail-view',
    setup: async page => { await openOrderById(page, 109); } },
  { id: 'S-35', name: 'Dialog — Order status change confirmation', page: 'index', mobile: 'yes', capturable: false,
    note: 'Browser-native window.confirm.' },
  { id: 'S-36', name: 'Form — Reservations / allocation section', page: 'index', mobile: 'yes', region: '.order-allocation-card',
    setup: openFirstOrder },
  { id: 'S-37', name: 'Reservations — allocation history table', page: 'index', mobile: 'yes', region: '.allocation-table-wrap',
    setup: openFirstOrder },
  { id: 'S-38', name: 'Dialog — Release reservation confirmation', page: 'index', mobile: 'yes', capturable: false,
    note: 'Browser-native window.confirm.' },
  { id: 'S-39', name: 'Shipping capacity preview', page: 'index', mobile: 'yes', region: '.order-preview-card',
    setup: async page => { await openFirstOrder(page); await clickIf(page, '.order-preview-btn'); await sleep(T.long); } },
  { id: 'S-40', name: 'Order Detail — notes card', page: 'index', mobile: 'yes', region: '.order-notes-card',
    setup: openFirstOrder },

  // ── H. Tab 6 — Expected Receipts ────────────────────────────────────────
  { id: 'S-41', name: 'Expected Receipts toolbar', page: 'index', mobile: 'yes', region: '#tab-expected .orders-toolbar',
    setup: async page => { await tab(page, 'expected'); } },
  { id: 'S-42', name: 'Expected Receipts table', page: 'index', mobile: 'partial', region: '#section-expected',
    setup: async page => { await tab(page, 'expected'); await page.waitForSelector('.er-row', { timeout: 8000 }).catch(() => {}); } },
  { id: 'S-43', name: 'Expected Receipts — row actions', page: 'index', mobile: 'partial', region: '#section-expected',
    setup: async page => {
      await tab(page, 'expected');
      await page.waitForSelector('.er-close-btn', { timeout: 8000 }).catch(() => {});
    } },
  { id: 'S-44', name: 'Expected Receipts — empty state', page: 'index', mobile: 'yes', region: '#section-expected',
    setup: async page => {
      await tab(page, 'expected');
      await page.fill('#er-text-filter', 'zzzz-no-match');
      await page.locator('#er-text-filter').press('Enter').catch(() => {});
      await sleep(T.med);
    } },
  { id: 'S-45', name: 'Dialog / Form — Expected Receipt create/edit modal', page: 'index', mobile: 'yes', region: '#er-modal-overlay',
    setup: async page => {
      await tab(page, 'expected');
      await clickIf(page, '#er-new-btn');
      await page.waitForSelector('#er-modal-overlay:not(.hidden)', { timeout: 6000 }).catch(() => {});
      await sleep(T.med);
    } },

  // ── I. Tab 7 — Supplies ─────────────────────────────────────────────────
  { id: 'S-46', name: 'Supplies page header + Request Supply', page: 'index', mobile: 'yes', region: '.supplies-page-header',
    setup: openSupplies },
  { id: 'S-47', name: 'Supplies inventory sub-tabs', page: 'index', mobile: 'yes', region: '.supplies-inventory-toolbar',
    setup: openSupplies },
  { id: 'S-48', name: 'Supplies search field', page: 'index', mobile: 'yes', region: '.supplies-search-label',
    setup: openSupplies },
  { id: 'S-49', name: 'Supplies inventory table', page: 'index', mobile: 'yes', region: '#supplies-inventory-table',
    setup: openSupplies },
  { id: 'S-50', name: 'Supplies — expanded lot / incoming detail row', page: 'index', mobile: 'yes', region: '#supplies-inventory-table',
    setup: async page => {
      await openSupplies(page);
      await clickIf(page, '.supply-item-row');
      await page.waitForSelector('.supply-lot-detail-row', { timeout: 8000 }).catch(() => {});
      await sleep(T.med);
    } },
  { id: 'S-51', name: 'Supply Requests list', page: 'index', mobile: 'yes', region: '#section-supply-requests',
    setup: async page => { await openSupplies(page); await page.waitForSelector('.supply-requests-table', { timeout: 8000 }).catch(() => {}); } },
  { id: 'S-52', name: 'Supply Requests — feedback / empty states', page: 'index', mobile: 'yes', region: '#section-supply-requests',
    overrides: { '/supply-requests': 'supply-requests-empty.json' },
    setup: async page => { await openSupplies(page); await sleep(T.med); } },
  { id: 'S-53', name: 'Dialog / Form — Request Supply modal', page: 'index', mobile: 'yes', region: '#supply-request-modal-overlay',
    setup: async page => {
      await openSupplies(page);
      await clickIf(page, '#supply-request-btn');
      await page.waitForSelector('#supply-request-modal-overlay:not(.hidden)', { timeout: 6000 }).catch(() => {});
      await sleep(T.med);
    } },

  // ── J. Overlays shared across tabs ──────────────────────────────────────
  { id: 'S-54', name: 'Lot Detail side panel', page: 'index', mobile: 'yes', region: '#lot-panel-overlay',
    setup: async page => {
      await ready(page);
      await expandPanel(page, 0);
      await clickIf(page, '#section-finished-goods tr.expandable');
      await clickIf(page, '#section-finished-goods .lot-link');
      await page.waitForSelector('#lot-panel-overlay:not(.hidden)', { timeout: 6000 }).catch(() => {});
      await sleep(T.med);
    } },
  { id: 'S-55', name: 'Lot disambiguation view', page: 'index', mobile: 'yes', region: '#lot-panel-overlay',
    status: { '/dashboard/api/lot/': 409 },
    overrides: { '/dashboard/api/lot/': 'lot-ambiguous.json' },
    setup: async page => {
      await ready(page);
      await expandPanel(page, 0);
      await clickIf(page, '#section-finished-goods tr.expandable');
      await clickIf(page, '#section-finished-goods .lot-link');
      await page.waitForSelector('.disambig-wrap', { timeout: 6000 }).catch(() => {});
    } },
  { id: 'S-56', name: 'Product Detail panel (active + depleted lots)', page: 'index', mobile: 'yes', region: '#lot-panel-overlay',
    setup: async page => {
      await ready(page);
      await page.fill('#global-search', 'granola');
      await page.waitForSelector('[data-search-product-id]', { timeout: 8000 }).catch(() => {});
      await clickIf(page, '[data-search-product-id]');
      await page.waitForSelector('#lot-panel-overlay:not(.hidden)', { timeout: 6000 }).catch(() => {});
      await sleep(T.med);
    } },

  // ── K. Material Flow — Sankey ───────────────────────────────────────────
  { id: 'S-57', name: 'Sankey controls bar', page: 'sankey', mobile: 'partial', region: '.controls',
    setup: async page => { await sleep(T.long); } },
  { id: 'S-58', name: 'Material-flow Sankey chart + legend', page: 'sankey', mobile: 'partial', region: '#chart-container',
    setup: async page => {
      await page.waitForSelector('#sankey-chart svg', { timeout: 15000 }).catch(() => {});
      await sleep(T.long);
    } },
  { id: 'S-59', name: 'Sankey banner + loading overlay', page: 'sankey', mobile: 'partial', region: null,
    fail: ['/transactions/history'],
    setup: async page => { await sleep(2500); } },

  // ── L. Production Lines — Process Flow ──────────────────────────────────
  { id: 'S-60', name: 'Summary strip', page: 'process-flow', mobile: 'yes', region: '#summary-strip',
    setup: async page => { await page.waitForSelector('#lines-grid .line-card, #lines-grid *', { timeout: 12000 }).catch(() => {}); await sleep(T.long); } },
  { id: 'S-61', name: 'Production lines grid', page: 'process-flow', mobile: 'yes', region: '#lines-grid',
    setup: async page => { await page.waitForSelector('#lines-grid *', { timeout: 12000 }).catch(() => {}); await sleep(T.long); } },
  { id: 'S-62', name: 'Error / stale banners + auto-refresh footer', page: 'process-flow', mobile: 'yes', region: null,
    fail: ['/transactions/history'],
    setup: async page => { await sleep(2500); } },

  // ── M. Traceability ─────────────────────────────────────────────────────
  { id: 'S-63', name: 'Lot search field + type-ahead dropdown', page: 'traceability', mobile: 'partial', region: '.search-area',
    setup: async page => {
      await sleep(T.long);
      await page.fill('#lotSearch', 'CQ');
      await sleep(T.med);
    } },
  { id: 'S-64', name: 'Trace direction switch + Trace button', page: 'traceability', mobile: 'partial', region: '.search-row',
    setup: async page => { await sleep(T.long); await page.fill('#lotSearch', 'OAT-4471'); await sleep(T.med); } },
  { id: 'S-65', name: 'Recent lots strip', page: 'traceability', mobile: 'partial', region: '#recentLots',
    setup: async page => { await page.waitForSelector('#recentLots .recent-lot, #recentLots', { timeout: 12000 }).catch(() => {}); await sleep(T.long); } },
  { id: 'S-66', name: 'Trace legend', page: 'traceability', mobile: 'partial', region: '.legend', setup: async () => sleep(T.long) },
  { id: 'S-67', name: 'Status bar', page: 'traceability', mobile: 'partial', region: '#statusBar', setup: async () => sleep(T.long) },
  { id: 'S-68', name: 'Trace graph (D3) + zoom controls', page: 'traceability', mobile: 'no', region: '#graphContainer',
    setup: traceALot },
  { id: 'S-69', name: 'Node tooltip', page: 'traceability', mobile: 'no', region: '#graphContainer',
    setup: async page => {
      await traceALot(page);
      const node = page.locator('#graphSvg g.node, #graphSvg rect').first();
      if (await node.count()) await node.hover().catch(() => {});
      await sleep(T.med);
    } },
  { id: 'S-70', name: 'Trace detail panel + export buttons', page: 'traceability', mobile: 'partial', region: '#detailPanel',
    setup: async page => {
      await traceALot(page);
      const node = page.locator('#graphSvg g.node, #graphSvg rect').first();
      if (await node.count()) await node.click().catch(() => {});
      await sleep(T.med);
    } },
  { id: 'S-71', name: 'Print audit report view', page: 'traceability', mobile: 'na', region: null, print: true,
    setup: traceALot },

  // ── N. Production Board / Scheduler ─────────────────────────────────────
  { id: 'S-72', name: 'Topbar KPI strip', page: 'scheduler', mobile: 'no', region: '#topbar', setup: schedulerSample },
  { id: 'S-73', name: 'Topbar actions', page: 'scheduler', mobile: 'no', region: '#topbar', setup: schedulerSample,
    note: 'Set as baseline / Print / Copy week are direct children of #topbar with no container of their own, so this capture is the whole KPI bar — the same frame as S-72.' },
  { id: 'S-74', name: 'Legend bar', page: 'scheduler', mobile: 'no', region: '#legendbar', setup: schedulerSample },
  { id: 'S-75', name: 'Delta panel (vs baseline)', page: 'scheduler', mobile: 'no', region: '#deltapanel',
    setup: async page => { await schedulerSample(page); await clickIf(page, '#btn-baseline'); await sleep(T.med); } },
  { id: 'S-76', name: 'Form — Left settings panel', page: 'scheduler', mobile: 'no', region: '#leftpanel', setup: schedulerSample },
  { id: 'S-77', name: 'Left panel — Finished goods on hand', page: 'scheduler', mobile: 'no', region: '#leftpanel',
    setup: async page => { await schedulerSample(page); await openDetails(page, 'Finished goods on hand'); } },
  { id: 'S-78', name: 'Left panel — Bulk-bin WIP', page: 'scheduler', mobile: 'no', region: '#leftpanel',
    setup: async page => { await schedulerSample(page); await openDetails(page, 'Bulk-bin WIP'); } },
  { id: 'S-79', name: 'Form — Left panel: Product catalog', page: 'scheduler', mobile: 'no', region: '#leftpanel',
    setup: async page => { await schedulerSample(page); await openDetails(page, 'Product catalog'); } },
  { id: 'S-80', name: 'Left panel — Orders in / plans out', page: 'scheduler', mobile: 'no', region: '#leftpanel', setup: schedulerSample },
  { id: 'S-81', name: 'Left panel — CSV import result message', page: 'scheduler', mobile: 'no', region: '#leftpanel',
    setup: async page => {
      await schedulerSample(page);
      await page.evaluate(() => {
        const el = document.getElementById('csvmsg');
        if (el) el.innerHTML = "<div class='ok'>Imported 34 order lines.</div>"
          + "<div class='warn'>3 unknown SKUs skipped: GR-40, CQ-05, SS-50.</div>"
          + "<div class='warn'>2 rows skipped: missing due_date.</div>";
      });
      await sleep(T.short);
    },
    note: 'The import-result block is populated directly; the harness uploads no file.' },
  { id: 'S-82', name: 'Left panel — "How this works"', page: 'scheduler', mobile: 'no', region: '#leftpanel',
    setup: async page => { await schedulerSample(page); await openDetails(page, 'How this works'); } },
  { id: 'S-83', name: 'Production board table (days × stations)', page: 'scheduler', mobile: 'no', region: '#center', setup: schedulerSample },
  { id: 'S-84', name: 'Board cell — day copy control + "more" toggle', page: 'scheduler', mobile: 'no', region: '#board',
    setup: async page => { await schedulerSample(page); await clickIf(page, '.more-toggle'); } },
  { id: 'S-85', name: 'Schedule panel (side)', page: 'scheduler', mobile: 'no', region: '#schedule-panel',
    setup: async page => {
      await schedulerSample(page);
      // `showSchedulePanel` is reached from a day header's copy control; the
      // topbar's Copy week only copies text.
      await clickIf(page, '.copyday');
      await page.waitForSelector('#schedule-panel.show', { timeout: 6000 }).catch(() => {});
      await sleep(T.med);
    } },
  { id: 'S-86', name: 'Dialog / Form — Pin modal', page: 'scheduler', mobile: 'partial', region: '#modal',
    setup: async page => {
      await schedulerSample(page);
      // `openPin` returns early for the `packonly` station, so pick a cell that
      // actually opens the modal.
      if (!(await clickIf(page, 'td.cell[data-st="bake"]:not(.off)'))) {
        await clickIf(page, 'td.cell:not(.off)');
      }
      // At 390px the board is far wider than the viewport and the sticky
      // station column covers the cell, so a real click cannot reach it —
      // itself a finding about the scheduler at phone width (IMP-027). Open the
      // modal directly so the dialog can still be measured.
      if (!(await page.locator('#modalbg.show').count())) {
        await page.evaluate(() => {
          const cell = document.querySelector('td.cell[data-st="bake"]:not(.off)')
            || document.querySelector('td.cell:not(.off)');
          if (cell && cell.onclick) cell.onclick();
        });
      }
      await page.waitForSelector('#modalbg.show', { timeout: 6000 }).catch(() => {});
      await sleep(T.med);
    } },
  { id: 'S-87', name: 'Right panel — Order book', page: 'scheduler', mobile: 'no', region: '#rightpanel', setup: schedulerSample },
  { id: 'S-88', name: 'Form — Add order line', page: 'scheduler', mobile: 'no', region: '#rightpanel',
    setup: async page => { await schedulerSample(page); await clickIf(page, '#btn-addorder'); await sleep(T.med); } },
  { id: 'S-89', name: 'Dialog — Add-order validation alert', page: 'scheduler', mobile: 'partial', capturable: false,
    note: 'Browser-native window.alert.' },
  { id: 'S-90', name: 'Order book — empty state', page: 'scheduler', mobile: 'no', region: '#rightpanel',
    setup: async page => { await page.waitForSelector('#rightpanel', { timeout: 10000 }).catch(() => {}); await sleep(T.med); },
    note: 'Captured before sample orders are loaded — the state a first-time user meets.' },
  { id: 'S-91', name: 'Print view', page: 'scheduler', mobile: 'na', region: null, print: true, setup: schedulerSample },
];

export const PAGE_URLS = {
  index: '/index.html',
  sankey: '/sankey.html',
  'process-flow': '/process-flow.html',
  traceability: '/traceability.html',
  scheduler: '/scheduler/seven-wells-production-board.html',
};
