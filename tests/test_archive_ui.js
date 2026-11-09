// dashboard/archive.js: wording, query flag, row controls (no browser needed).
const test = require('node:test');
const assert = require('node:assert/strict');
const archive = require('../dashboard/archive.js');

test('confirm wording keeps history promise in both languages', () => {
  assert.deepEqual(archive.confirmText(['BS Salt']), {
    en: 'Archive BS Salt? It will be hidden but history is kept.',
    es: '¿Archivar BS Salt? Se ocultará, pero se conserva el historial.' });
  assert.equal(archive.confirmText(['A', 'B', 'C']).en, 'Archive 3 items? It will be hidden but history is kept.');
  assert.match(archive.confirmText(['BS Salt'], true).en, /^Restore BS Salt\?/);
});

test('include_archived is added only when Show archived is on', () => {
  assert.equal(archive.withArchivedQuery('/inventory/batches', false), '/inventory/batches');
  assert.equal(archive.withArchivedQuery('/inventory/batches', true), '/inventory/batches?include_archived=true');
  assert.equal(archive.withArchivedQuery('/x?a=1', true), '/x?a=1&include_archived=true');
});

test('row controls: owner gets archive or restore, others only the tag', () => {
  const live = { product_id: 7, name: 'BS <Oats>', archived: false };
  const gone = { product_id: 8, name: 'BS Salt', archived: true };
  assert.match(archive.rowControls('ingredients', live, true), /data-archive-one="7"/);
  assert.match(archive.rowControls('ingredients', live, true), /BS &lt;Oats&gt;/);
  assert.doesNotMatch(archive.rowControls('ingredients', live, true), /archived-tag/);
  assert.match(archive.rowControls('ingredients', gone, true), /data-archive-restore="8"/);
  assert.match(archive.rowControls('ingredients', gone, true), /Archived/);
  assert.equal(archive.rowControls('ingredients', live, false), '');
  assert.equal(archive.rowControls('ingredients', gone, false), archive.tag());
  assert.match(archive.rowCheckbox('ingredients', live, true), /class="archive-select"/);
  assert.equal(archive.rowCheckbox('ingredients', gone, true), '');
  assert.equal(archive.rowClass(gone), ' archived-row');
});

test('server refusals read in both languages', () => {
  const body = { detail: { error_code: 'USED_BY_ACTIVE_RECIPE', message: 'BS Oats is used by active recipe Batch X.', message_es: 'BS Oats se usa en la receta activa Batch X.' } };
  assert.equal(archive.errorText(409, body), 'BS Oats is used by active recipe Batch X. / BS Oats se usa en la receta activa Batch X.');
  assert.equal(archive.errorText(403, { detail: 'This action is available under Michael.' }), 'HTTP 403: This action is available under Michael.');
});
