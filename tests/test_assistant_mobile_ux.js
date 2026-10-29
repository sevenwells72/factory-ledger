const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ui = require('../dashboard/fl-assistant.js');

// 2026-10-16 08:00 in the plant (EDT, UTC-4).
const now = new Date('2026-10-16T12:00:00Z');

test('history dividers name the plant day in English and Spanish', () => {
  assert.equal(ui.dayLabel('2026-10-16T11:51:00Z', now), 'Today');
  assert.equal(ui.dayLabel('2026-10-15T17:59:00Z', now), 'Yesterday');
  assert.equal(ui.dayLabel('2026-10-15T17:59:00Z', now, 'es'), 'Ayer');
  assert.equal(ui.dayLabel('2026-10-16T11:51:00Z', now, 'es'), 'Hoy');
  assert.equal(ui.dayLabel('2026-10-13T15:00:00Z', now), 'Tue, Oct 13');
  // 03:30 UTC is still the previous evening at the plant.
  assert.equal(ui.dayLabel('2026-10-16T03:30:00Z', now), 'Yesterday');
});

test('a divider starts each day and marks an hour-long gap, never every message', () => {
  assert.equal(ui.dividerLabel('2026-10-15T17:59:00Z', null, now), 'Yesterday');
  assert.equal(ui.dividerLabel('2026-10-15T18:01:00Z', '2026-10-15T17:59:00Z', now), null);
  assert.equal(ui.dividerLabel('2026-10-15T23:03:00Z', '2026-10-15T18:37:00Z', now), 'Yesterday · 7:03 PM');
  assert.equal(ui.dividerLabel('2026-10-16T11:51:00Z', '2026-10-15T23:05:00Z', now), 'Today · 7:51 AM');
  assert.equal(ui.dividerLabel('2026-10-16T11:51:00Z', '2026-10-15T23:05:00Z', now, 'es'), 'Hoy · 7:51 a.m.');
  assert.equal(ui.dividerLabel(undefined, '2026-10-15T23:05:00Z', now), null);
  assert.equal(ui.dividerLabel('not a date', null, now), null);
});

test('jump to latest appears only after scrolling up', () => {
  assert.equal(ui.showJump({ scrollTop: 31000, scrollHeight: 31844, clientHeight: 844 }), false);
  assert.equal(ui.showJump({ scrollTop: 30800, scrollHeight: 31844, clientHeight: 844 }), false);
  assert.equal(ui.showJump({ scrollTop: 30000, scrollHeight: 31844, clientHeight: 844 }), true);
  assert.equal(ui.showJump({ scrollTop: 0, scrollHeight: 31844, clientHeight: 844 }), true);
});

test('every photo state has plain English and Spanish words', () => {
  const states = ['choose', 'uploading', 'attached', 'failed', 'too_large', 'wrong_type'];
  for (const state of states) {
    assert.ok(ui.photoStatus({ state }, 'en'), state);
    assert.ok(ui.photoStatus({ state }, 'es'), state);
    assert.notEqual(ui.photoStatus({ state }, 'en'), ui.photoStatus({ state }, 'es'));
  }
  assert.equal(ui.photoStatus({ state: 'uploading', progress: 45 }), 'Uploading 45%…');
  assert.equal(ui.photoStatus({ state: 'uploading', progress: 45 }, 'es'), 'Subiendo 45%…');
  assert.equal(ui.photoStatus({ state: 'attached' }), 'Attached ✓ · goes with your message');
  assert.equal(ui.photoStatus({ state: 'failed' }), 'Upload failed — tap to retry');
  assert.deepEqual(Object.keys(ui.photoKinds).sort(), [...ui.evidenceKinds].sort());
});

test('only a definite FL rejection is final; drops and server errors can be retried', () => {
  const invalid = { detail: { error_code: 'PHOTO_INVALID' }, httpStatus: 422 };
  assert.equal(ui.photoProblem(invalid, 2_000_000), 'wrong_type');
  assert.equal(ui.photoProblem(invalid, ui.MAX_PHOTO + 1), 'too_large');
  assert.equal(ui.photoProblem({ httpStatus: 413 }, 1000), 'too_large');
  assert.equal(ui.photoProblem({ photo: 'too_large' }, 0), 'too_large');
  assert.equal(ui.photoProblem({ photo: 'wrong_type' }, 0), 'wrong_type');
  assert.equal(ui.photoProblem({ network: true }, 1000), 'failed');
  assert.equal(ui.photoProblem({ httpStatus: 500 }, 1000), 'failed');
  assert.equal(ui.photoProblem({ detail: { error_code: 'STORAGE_FAILED' }, httpStatus: 502 }, 1000), 'failed');
});

test('page: photo type is chosen after picking, jump button exists, assets re-fetched', () => {
  const html = fs.readFileSync(path.join(__dirname, '../dashboard/fl-assistant.html'), 'utf8');
  assert.equal(html.includes('id="photo-kind"'), false, 'no hidden default evidence type');
  assert.match(html, /<button id="jump-latest"[^>]* hidden /);
  assert.match(html, /fl-assistant\.js\?v=19/);
  assert.match(html, /fl-assistant\.css\?v=10/);
  const css = fs.readFileSync(path.join(__dirname, '../dashboard/fl-assistant.css'), 'utf8');
  assert.match(css, /\.topbar\{position:sticky;top:0/);
});
