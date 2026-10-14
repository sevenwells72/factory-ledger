const { test } = require('node:test');
const assert = require('node:assert/strict');
const ui = require('../dashboard/so-list-actions.js');

const messages = [
  ['SHIPMENT_REQUIRED', 'close',
    'This order has no recorded shipment, so it cannot be closed as shipped. Record the shipment first.',
    'Este pedido no tiene un envío registrado, así que no puede cerrarse como enviado. Registra el envío primero.'],
  ['SHIPMENT_REQUIRED', 'reopen',
    'This order has no recorded shipment, so it cannot be reopened with its saved shipped or invoiced status. Record the shipment first.',
    'Este pedido no tiene un envío registrado, así que no puede reabrirse con su estado guardado de enviado o facturado. Registra el envío primero.'],
  ['LEGACY_CLOSE_CUTOFF', 'close',
    'This order was created on or after the cutover date. It cannot be closed as “Shipped, not recorded”; record the shipment first.',
    'Este pedido se creó en la fecha de corte o después. No puede cerrarse como “Enviado, no registrado”; registra el envío primero.'],
  ['LEGACY_CLOSE_CUTOFF', 'reopen',
    'This order was created on or after the cutover date and cannot be reopened from “Shipped, not recorded”.',
    'Este pedido se creó en la fecha de corte o después y no puede reabrirse desde “Enviado, no registrado”.'],
  ['ROLE_NOT_ALLOWED', 'close',
    'Only the owner can close an order as “Shipped, not recorded”.',
    'Solo el propietario puede cerrar un pedido como “Enviado, no registrado”.'],
  ['ROLE_NOT_ALLOWED', 'reopen',
    'Only the owner can reopen an order closed as “Shipped, not recorded”.',
    'Solo el propietario puede reabrir un pedido cerrado como “Enviado, no registrado”.'],
  ['NOTE_REQUIRED', 'close',
    'Add a note explaining why this order is being closed as “Shipped, not recorded”.',
    'Agrega una nota que explique por qué se cierra este pedido como “Enviado, no registrado”.'],
  ['NOTE_REQUIRED', 'reopen',
    'Add a note explaining why this order closed as “Shipped, not recorded” is being reopened.',
    'Agrega una nota que explique por qué se reabre este pedido cerrado como “Enviado, no registrado”.'],
];
for (const [code, action, en, es] of messages) {
  for (const [lang, expected] of [['en', en], ['es', es]]) {
    test(`${action} ${code} is specific and safe in ${lang} across current and legacy error envelopes`, () => {
      const detail = { error_code: code, message: 'INTERNAL_RESPONSE_MUST_NOT_RENDER' };
      for (const envelope of [{ detail }, { payload: { detail } }, { payload: detail }]) {
        for (const context of ['preview', 'commit']) {
          assert.equal(ui.errorMessage({ status: code === 'LEGACY_CLOSE_CUTOFF' ? 403 : 409, ...envelope }, context, lang, action, 'shipped_not_recorded'), expected);
        }
      }
    });
  }
}

test('unrecorded shipped close and exceptional reopen require notes; ordinary exits preserve their note policy', () => {
  const legacy = { state: 'closed', state_reason: 'shipped_not_recorded' };
  assert.equal(ui.noteRequired({ state: 'open' }, 'close', 'shipped_not_recorded'), true);
  assert.equal(ui.noteRequired(legacy, 'reopen', ''), true);
  assert.equal(ui.noteRequired({ state: 'open' }, 'cancel', 'other'), true);
  for (const reason of ['shipped_recorded', 'short_closed']) assert.equal(ui.noteRequired({ state: 'open' }, 'close', reason), false);
  for (const reason of ['shipped_recorded', 'short_closed', 'customer_cancelled', undefined]) {
    assert.equal(ui.noteRequired({ state: 'closed', state_reason: reason }, 'reopen', ''), false);
  }
  assert.equal(ui.noteRequired({ state: 'cancelled', state_reason: 'shipped_not_recorded' }, 'reopen', ''), false);
});

test('unknown backend detail stays private while status fallback remains actionable in both languages', () => {
  for (const lang of ['en', 'es']) {
    for (const status of [400, 401, 403, 404, 409, 422, 500, 503]) {
      const text = ui.errorMessage({ status, detail: { message: 'INTERNAL_RESPONSE_MUST_NOT_RENDER' } }, 'preview', lang);
      assert.ok(text.length > 25);
      assert.doesNotMatch(text, /INTERNAL_RESPONSE_MUST_NOT_RENDER/);
    }
  }
});

for (const lang of ['en', 'es']) {
  test(`ordinary role refusals identify the selected action in ${lang}`, () => {
    for (const action of ['close', 'cancel', 'reopen']) {
      const message = ui.errorMessage({ status: 403, detail: { error_code: 'ROLE_NOT_ALLOWED' } }, 'preview', lang, action, 'shipped_recorded');
      assert.doesNotMatch(message, /Shipped, not recorded|Enviado, no registrado/);
      assert.match(message, lang === 'es' ? /Tu rol no permite/ : /Your role does not allow/);
      const verb = lang === 'en' ? action : { close: 'cerrar', cancel: 'cancelar', reopen: 'reabrir' }[action];
      assert.ok(message.includes(verb));
    }
  });
}
