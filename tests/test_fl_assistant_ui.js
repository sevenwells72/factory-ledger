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
  const validation = ui.errorText({ detail: [{ loc: ['body', 'cases'], msg: 'Field required' }] });
  assert.equal(validation, ui.neutralMessages.misunderstood.join(' / '));
  assert.ok(!validation.includes('Field required') && !validation.includes('cases:'));
});

test('neutral outcomes have fixed English and Spanish text with no receipt assertion', () => {
  for (const [kind, messages] of Object.entries(ui.neutralMessages)) {
    assert.equal(messages.length, 2, kind);
    assert.notEqual(messages[0], messages[1], kind);
    assert.ok(messages.every(text => !/NOT recorded|NO registrado|Field required|VALIDATION_ERROR/.test(text)), kind);
    assert.equal(ui.validReceipt({ kind }), false);
  }
  assert.ok(ui.neutralMessages.unsupported_request.every(text => text.includes('ChatGPT')));
});

test('A3b hold is an explicit pending outcome and cannot become a receipt', () => {
  const held = { held: true, status: 'awaiting_approval', exception_id: 17 };
  assert.equal(ui.awaitingApproval(held), true);
  assert.equal(ui.validReceipt(held), false);
  for (const value of [null, {}, { held: true }, { status: 'awaiting_approval' }, { held: false, status: 'awaiting_approval' }]) assert.equal(ui.awaitingApproval(value), false);
});

test('A6 SHP receipts are real receipts while BOL blockers always prevent Record', () => {
  assert.equal(ui.validReceipt({ success: true, receipt_number: 'SHP-261013-001', shipment_id: 4 }), true);
  assert.equal(ui.validReceipt({ success: false, receipt_number: 'SHP-261013-001' }), false);
  for (const can_commit of [false, true]) {
    assert.equal(ui.canRecord({ can_commit, blockers: [{ code: 'BOL_REQUIRED' }] }), false);
    assert.equal(ui.canRecord({ can_commit, blockers: [{ code: 'LOT_NOT_CONFIRMED' }, { code: 'BOL_REQUIRED' }] }), false);
  }
});
test('A6 private evidence needs a positive integer ID and an explicit evidence kind', () => {
  for (const kind of ui.evidenceKinds) assert.equal(ui.validAttachment({ id: 7, kind }), true);
  for (const value of [null, {}, { id: '7', kind: 'bol' }, { id: '7e3381ea-acfb-4cd7-92c0-4ceda5d7c2e3', kind: 'bol' },
    { id: 0, kind: 'bol' }, { id: 7.5, kind: 'bol' }, { id: 7 }, { id: 7, kind: 'public_photo' }]) assert.equal(ui.validAttachment(value), false);
});
