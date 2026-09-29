const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const source = fs.readFileSync(path.join(__dirname, '../dashboard/dashboard.js'), 'utf8');

function functionSource(name, async = false) {
  const start = source.indexOf(`  ${async ? 'async ' : ''}function ${name}(`);
  assert.ok(start >= 0);
  return source.slice(start, source.indexOf('\n  }', start) + 4);
}

test('edit mode renders service counts read-only and physical pounds editable', () => {
  const noop = () => '';
  const context = vm.createContext({
    state: { orderDetailEditMode: true, orderReadyWrites: new Set() },
    SOList: { closeExplanation: noop, number: String, paragraph: String,
      explanation: () => ({ attrs: '', content: '' }), trigger: (id, label) => label,
      healthContent: noop, bindExplanations: noop },
    canEditOrderHeader: () => true, canEditOrderLines: () => true,
    orderDetailFlag: () => ({ ready: false, metadataKnown: true }),
    orderDetailShipDate: noop, escHtml: String, escAttr: String,
    PalletCalculations: { calculateLinePallets: () => ({ calculatedPallets: null }) },
    renderOrderEditActions: noop, renderAllocationSection: noop, renderShippingPreviewSection: noop,
    bindOrderInventoryToggles: noop, bindOrderDetailEditControls: noop,
    bindOrderDetailActions: noop, bindAllocationControls: noop,
  });
  vm.runInContext(functionSource('orderLineQuantities') + functionSource('renderOrderDetail'), context);
  const container = { innerHTML: '' };
  context.renderOrderDetail({ order_id: 1, state: 'open', lines: [
    { line_id: 11, product: 'Pallet', unit: 'each', quantity: 3, unit_quantity: 3,
      quantity_lb: 0, is_service: true, case_price: 12.5, line_status: 'pending' },
    { line_id: 12, product: 'Food', quantity_lb: 50, case_price: 30, line_status: 'pending' },
  ] }, container, true);
  const service = container.innerHTML.match(/<tr[^>]*data-line-id="11">([\s\S]*?)<\/tr>/)[1];
  assert.match(service, /data-label="Ordered units">3 units<\/td>/);
  assert.doesNotMatch(service, /order-line-qty-input/);
  assert.match(service, /order-line-price-input/);
  const physical = container.innerHTML.match(/<tr[^>]*data-line-id="12">([\s\S]*?)<\/tr>/)[1];
  assert.match(physical, /order-line-qty-input[^>]*value="50"/);
});

test('service price can still be saved without a quantity input', async () => {
  const requests = [];
  const messages = [];
  const row = { dataset: { lineId: '11' }, querySelector: selector =>
    selector === '.order-line-price-input' ? { value: '0' } : null };
  const context = vm.createContext({
    URLSearchParams,
    state: { currentOrderDetail: { order_id: 1, lines: [
      { line_id: 11, unit: 'each', quantity: 3, quantity_lb: 0, case_price: 12.5 },
    ] } },
    setOrderDetailMessage: (container, text) => messages.push(text), hideError() {},
    fetchSalesAPI: async (url, options) => requests.push({ url, options }),
    refreshOrderDetail: async () => {},
  });
  vm.runInContext(functionSource('saveOrderLines', true), context);
  await context.saveOrderLines({ querySelectorAll: () => [row], querySelector: () => null });
  assert.equal(requests.length, 1, messages.join('; '));
  assert.equal(requests[0].url, '/sales/orders/1/lines/11/update?unit_price=0');
  assert.equal(requests[0].options.method, 'PATCH');
});
