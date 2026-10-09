const test = require('node:test');
const assert = require('node:assert/strict');
const ui = require('../dashboard/fl-assistant.js');
test('only a complete real receipt shape may become green', () => {
  for (const value of [null, {}, { receipt_number: 'recorded' }, { receipt_number: 'MK-261008-001', success: false }, { text: 'RCV-261008-001' }]) assert.equal(ui.validReceipt(value), false);
  assert.equal(ui.validReceipt({ receipt_number: 'RCV-261008-001', success: true }), true);
});
test('A5 recoverable lot confirmation blockers allow manual evidence; other blockers stay blocked', () => {
  assert.equal(ui.canRecord({ can_commit: true }), true);
  assert.equal(ui.canRecord({ can_commit: false, blockers: [{ code: 'LOT_NOT_CONFIRMED' }] }), true);
  assert.equal(ui.canRecord({ can_commit: false, blockers: [{ code: 'LOT_NOT_CONFIRMED' }, { code: 'INSUFFICIENT_STOCK' }] }), false);
  assert.equal(ui.canRecord({ can_commit: false, blockers: [] }), false);
});
test('permission refusals retain FL error code, action and role', () => {
  assert.equal(ui.errorText({ result: { detail: { error_code: 'ROLE_NOT_ALLOWED', action: 'make', role: 'office', message: 'Not allowed' } } }), 'ROLE_NOT_ALLOWED · Not allowed · make · office');
  assert.equal(ui.errorText({ detail: [{ loc: ['body', 'cases'], msg: 'Field required' }] }), 'cases: Field required');
});

test('A3b hold is an explicit pending outcome and cannot become a receipt', () => {
  const held = { held: true, status: 'awaiting_approval', exception_id: 17 };
  assert.equal(ui.awaitingApproval(held), true);
  assert.equal(ui.validReceipt(held), false);
  for (const value of [null, {}, { held: true }, { status: 'awaiting_approval' }, { held: false, status: 'awaiting_approval' }]) assert.equal(ui.awaitingApproval(value), false);
});
