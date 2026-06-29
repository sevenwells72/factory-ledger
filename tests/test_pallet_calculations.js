const test = require('node:test');
const assert = require('node:assert/strict');

const {
  inferCasesPerPallet,
  calculateLinePallets,
  calculateOrderPallets
} = require('../dashboard/pallet-calculations.js');

test('3,640 cases of a 10-lb product is 26 pallets', () => {
  const result = calculateLinePallets({ product: 'CQ Granola 10 LB' }, 3640);
  assert.equal(result.casesPerPallet, 140);
  assert.equal(result.calculatedPallets, 26);
  assert.equal(result.physicalPalletsRoundedUp, 26);
  assert.equal(result.display, '26 pallets');
});

test('24 cases of a 25-lb UOM is 0.4 pallet', () => {
  const result = calculateLinePallets({ uom: '25-lb case' }, 24);
  assert.equal(result.casesPerPallet, 60);
  assert.equal(result.calculatedPallets, 0.4);
  assert.equal(result.physicalPalletsRoundedUp, 1);
  assert.equal(result.display, '0.4 pallet');
});

test('seven 25-lb lines total 2.8 calculated and 3 physical mixed pallets', () => {
  const lines = Array.from({ length: 7 }, (_, index) => ({
    line_id: index + 1,
    product: `Granola Flavor ${index + 1} 25 LB`,
    unit_count: 24
  }));
  const result = calculateOrderPallets(lines, 'unit_count');
  assert.ok(Math.abs(result.calculatedPallets - 2.8) < 1e-9);
  assert.equal(result.physicalPalletsRoundedUp, 3);
  assert.equal(result.display, '2.8 pallets / 3 physical mixed pallets');
});

test('unknown product and UOM do not crash or invent a pallet value', () => {
  assert.equal(inferCasesPerPallet({ product: 'Retail pouch', uom: 'each' }), null);
  assert.deepEqual(calculateLinePallets({ product: 'Retail pouch' }, 12), {
    casesPerPallet: null,
    calculatedPallets: null,
    physicalPalletsRoundedUp: null,
    display: '\u2014'
  });
});
