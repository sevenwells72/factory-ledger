const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const dashboard = fs.readFileSync(path.join(__dirname, '../dashboard/dashboard.js'), 'utf8');

function fn(source, name) {
  const start = source.search(new RegExp('  (?:async )?function ' + name + '\\('));
  assert.ok(start >= 0, name);
  const end = source.indexOf('\n  }', start) + 4;
  return source.slice(start, end);
}

function render(data) {
  const container = { innerHTML: '' };
  const ctx = vm.createContext({
    isPanelExpanded: () => false,
    CASES_PER_PALLET: {},
    escHtml: s => String(s),
    fmt: v => String(v), fmtInt: v => String(v), fmtWt: v => String(v),
    inventoryUnitCount: () => null, caseBadgeClass: () => '', fmtPallets: () => '',
    bindCollapsibles: () => {}, bindExpandableRows: () => {}, bindLotLinks: () => {},
  });
  vm.runInContext(fn(dashboard, 'renderFinishedGoodsPanels'), ctx);
  ctx.renderFinishedGoodsPanels(data, container);
  return container.innerHTML;
}

test('a panel whose SKUs are all archived is not rendered', () => {
  const html = render({ panels: [
    { id: 'retail_bs_8oz', title: '6x8 OZ Retail Cases (BS Line)', products: [], missing_skus: [],
      archived_skus: ['BS Granola – Peanut Butter Banana – 6x8 OZ Case'] },
    { id: 'retail_bs', title: '6x7 OZ Retail Cases (BS Line)', case_weight_lb: null,
      products: [{ product_name: 'BS Granola – Dark Chocolate – 6x7 OZ Case', on_hand_lbs: 30, case_weight_lb: 2.625, lots: [] }],
      missing_skus: [], archived_skus: [] },
  ]});
  assert.ok(!html.includes('6x8 OZ Retail Cases'), html);
  assert.ok(html.includes('6x7 OZ Retail Cases'), html);
});

test('a panel with a missing SKU still renders even if the rest are archived', () => {
  const html = render({ panels: [
    { id: 'typo', title: 'Typo Panel', products: [], missing_skus: ['Nope'], archived_skus: ['Old'] },
  ]});
  assert.ok(html.includes('Typo Panel'), html);
  assert.ok(html.includes('Missing SKUs'), html);
});

test('an empty panel with nothing archived keeps its legacy empty rendering', () => {
  const html = render({ panels: [
    { id: 'legacy', title: 'Legacy Panel', products: [], missing_skus: [] },
  ]});
  assert.ok(html.includes('Legacy Panel'), html);
  assert.ok(html.includes('No inventory on hand.'), html);
});
