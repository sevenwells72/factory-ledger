const test = require('node:test');
const assert = require('node:assert/strict');
const display = require('../dashboard/assistant-display.js');
const ledger = require('../dashboard/ledger-ui.js');
const today = '2026-10-15T19:00:00Z';

const inventoryCard = {
  kind: 'read', tool: 'inventory_lookup', result: { query: 'Classic', results: [{
    product: 'Batch Classic Granola #9', total_on_hand: 17678,
    lots: [5000, 4000, 3000, 2000, 2000, 1678].map((qty_on_hand, i) => ({ lot: 'LOT-' + (i + 1), qty_on_hand, unit: '10 lb case' }))
  }] }
};
test('inventory answer begins with a plain accurate sentence and a lot/quantity/unit table', () => {
  const [section] = display.readSections(inventoryCard);
  assert.equal(section.sentence, 'Batch Classic Granola #9: 17,678 lb in 6 lots.');
  assert.deepEqual(section.columns, ['Lot', 'Quantity', 'Unit']);
  assert.deepEqual(section.rows[0], ['LOT-1', '5,000', 'lb']);
  assert.equal(section.rows.length, 6);
  assert.doesNotMatch(JSON.stringify(section), /qty_on_hand|10 lb case|match_tier/);
});
test('inventory packaging descriptions never turn pounds into cases', () => {
  for (const unit of ['lb', '10 lb case', '25 LB BAG', 'case']) {
    assert.equal(display.inventoryUnit({ qty_on_hand: 10, unit }), 'lb');
    assert.equal(display.inventoryUnit({ quantity_lb: 10, unit }), 'lb');
  }
});
test('an explicitly counted case result stays in cases in the sentence and rows', () => {
  const [section] = display.readSections({ kind: 'read', tool: 'inventory_lookup', result: { results: [{
    product: 'Classic cases', quantity: 12, lots: [{ lot: 'CASE-1', quantity: 12, unit: 'cases' }]
  }] } });
  assert.equal(section.sentence, 'Classic cases: 12 cases in 1 lot.');
  assert.deepEqual(section.rows, [['CASE-1', '12', 'cases']]);
});
test('empty inventory answers stay plain and useful', () => {
  assert.equal(display.readSections({ tool: 'inventory_lookup', result: { results: [] } })[0].sentence, 'No inventory matches.');
});
test('identical result cards in one reply collapse despite property order', () => {
  const same = { ...inventoryCard, id: 'other', result: { results: inventoryCard.result.results, query: 'Classic' } };
  const different = { ...inventoryCard, result: { query: 'oats', results: [] } };
  const draft = { kind: 'draft', id: 'draft-1', prepared: {} };
  assert.deepEqual(display.dedupeCards([inventoryCard, same, different, draft, draft]), [inventoryCard, different, draft, draft]);
  assert.equal(display.dedupeCards([inventoryCard]).length, 1, 'a later reply can repeat the same information');
});
test('receipt answer summarizes actual output and retains lot weights with no internal fields', () => {
  const [section] = display.readSections({ tool: 'receipt_lookup', result: {
    receipt_number: 'RCV-261015-001', action: 'receive', actor: { name: 'Michael' }, happened_at: '2026-10-15T15:19:00Z',
    draft: { product_name: 'Sunflower Seeds', total_lb: 50 },
    transactions: [{ lines: [{ product_name: 'Sunflower Seeds', lot_code: 'GRAI-1', quantity_lb: 50 }] }],
    ticket: 'private', payload_hash: 'private'
  } }, { now: today });
  assert.equal(section.sentence, 'Received 50 lb Sunflower Seeds.');
  assert.deepEqual(section.rows, [['Sunflower Seeds', 'GRAI-1', '50', 'lb']]);
  assert.ok(section.notes.includes('Today 11:19 AM'));
  assert.doesNotMatch(JSON.stringify(section), /private|payload_hash|transactions/);
});
test('voided receipt clearly says its stock no longer counts', () => {
  const [section] = display.readSections({ result: { receipt_number: 'ADJ-261015-001', action: 'adjust', effective_status: 'voided',
    draft: { product_name: 'Oats', current_quantity_lb: 2000, new_balance_lb: 1995, delta_lb: -5 } } });
  assert.equal(section.sentence, 'Adjusted Oats 2,000 → 1,995 lb.');
  assert.ok(section.notes.includes('Voided — no longer counts in stock'));
});
test('all Today rows use newest recording first and plain time/what/receipt/who', () => {
  const receipt = (id, entered_at, happened_at) => ({ receipt_number: 'RCV-261015-' + id, action: 'receive', entered_at, happened_at,
    actor: { name: 'Michael' }, draft: { product_name: 'Sunflower Seeds', total_lb: 50 } });
  const [section] = display.readSections({ tool: 'today_entries_all', result: { receipts: [
    receipt('001', '2026-10-15T15:19:00Z', '2026-10-15T15:19:00Z'),
    receipt('002', '2026-10-15T15:20:00Z', '2026-10-14T11:00:00Z')
  ] } }, { now: today });
  assert.equal(section.sentence, '2 entries today.');
  assert.deepEqual(section.columns, ['Time', 'What', 'Receipt', 'Who']);
  assert.deepEqual(section.rows[0], ['11:20 AM', 'Received 50 lb Sunflower Seeds', 'RCV-261015-002', 'Michael']);
});
test('300 Today entries are never truncated by the display helper', () => {
  const receipts = Array.from({ length: 300 }, (_, i) => ({ receipt_number: `RCV-261015-${String(i).padStart(3, '0')}`,
    entered_at: new Date(Date.parse(today) - i * 1000).toISOString(), action: 'receive',
    draft: { product_name: 'Oats', total_lb: 1 }, operator_id: 'Michael' }));
  const [section] = display.readSections({ result: { receipts } });
  assert.equal(section.rows.length, 300);
  assert.equal(section.sentence, '300 entries today.');
});
test('draft shows product, quantity, plain reason, stock before and after, and readable plant times', () => {
  const fields = Object.fromEntries(display.draftFields({ action: 'adjust', expires_at: '2026-10-15T15:29:00Z', draft: {
    product_name: 'Oats', delta_lb: -5, current_quantity_lb: 2000, new_balance_lb: 1995,
    reason_code: 'physical_count', happened_at: '2026-10-15T15:19:00Z', identity_status: 'UNIDENTIFIED'
  } }, { now: today }));
  assert.equal(fields.Product, 'Oats');
  assert.equal(fields.Quantity, '-5 lb');
  assert.equal(fields.Reason, 'Physical count');
  assert.equal(fields['Stock before → after'], '2,000 → 1,995 lb');
  assert.equal(fields['Happened at'], 'Today 11:19 AM');
  assert.equal(fields.Expires, 'Today 11:29 AM');
  assert.equal(fields['Lot note'], 'Lot identity needs to be checked');
  assert.doesNotMatch(Object.values(fields).join(' '), /physical_count|UNIDENTIFIED|T15:|2026-/);
});
test('draft reason objects use FL catalog labels in the selected language', () => {
  const prepared = { draft: { product_name: 'Oats', quantity: 20, reason: { code: 'physical_count', label_en: 'Physical count', label_es: 'Conteo físico' } } };
  const fields = Object.fromEntries(display.draftFields(prepared, { language: 'es' }));
  assert.equal(fields.Motivo, 'Conteo físico');
  assert.equal(fields.Cantidad, '20 lb');
});
test('an incomplete SKU draft shows exact blocker product plus requested batches', () => {
  const prepared = { blockers: [{ code: 'SKU_CONFIRMATION_REQUIRED', message: 'Confirm Batch Classic Granola #9 as the output SKU and prepare again.' }], draft: {} };
  const fields = Object.fromEntries(display.draftFields(prepared, { fallback: { text: 'Made 1 batch of Classic #9' } }));
  assert.equal(fields.Product, 'Batch Classic Granola #9');
  assert.equal(fields.Quantity, '1 batch requested — confirm the draft');
});
test('unknown draft fields stay explicit and never imply a made-up weight', () => {
  const fields = Object.fromEntries(display.draftFields({ draft: {} }));
  assert.equal(fields.Product, 'Product not supplied');
  assert.equal(fields.Quantity, 'Quantity not supplied — check the draft');
  assert.equal(Object.fromEntries(display.draftFields({ request: { product_id: 283, cases: 2 }, draft: {} })).Quantity, '2 cases');
});
test('pack draft preserves both pounds and cases', () => {
  const fields = Object.fromEntries(display.draftFields({ draft: { target_product_name: 'Classic 10 lb case', total_lb: 500, cases: 50 } }));
  assert.equal(fields.Product, 'Classic 10 lb case');
  assert.equal(fields.Quantity, '500 lb');
  assert.equal(fields.Cases, '50');
});
test('plant dates handle UTC midnight and Today uses the New York day', () => {
  assert.equal(display.formatTime('2026-10-15T00:19:00Z', { now: '2026-10-15T03:00:00Z' }), 'Today 8:19 PM');
  assert.match(display.formatTime('2026-10-15T00:19:00Z', { now: today }), /Oct 14, 2026 8:19 PM/);
  assert.equal(display.formatTime('bad date'), 'Time unavailable');
});
test('read answers localize plain tables and include direct-make substitutions', () => {
  const [section] = display.readSections(inventoryCard, { language: 'es' });
  assert.equal(section.sentence, 'Batch Classic Granola #9: 17,678 lb en 6 lotes.');
  assert.deepEqual(section.columns, ['Lote', 'Cantidad', 'Unidad']);
  const sections = display.readSections({ result: { sections: [{ label_en: 'RECEIVED', label_es: 'RECIBIDO', rows: [{ line_en: 'Received oats', line_es: 'Avena recibida' }] }],
    direct_make_substitutions: { label_en: 'Oat substitutions', label_es: 'Sustituciones de avena', rows: [{ line_en: 'Used extra oats', line_es: 'Se usó más avena' }] } } }, { language: 'es' });
  assert.deepEqual(sections.map(s => s.rows), [[['Avena recibida']], [['Se usó más avena']]]);
});
test('unknown result fields and error objects never become raw JSON', () => {
  const [section] = display.readSections({ result: { internal: { access_token: 'secret' } } });
  assert.equal(section.sentence, 'FL returned this information.');
  assert.doesNotMatch(JSON.stringify(section), /secret|access_token/);
  assert.equal(display.errorMessage({ detail: { error_code: 'ROLE_NOT_ALLOWED', action: 'make', role: 'office' } }), 'This action is not allowed for your role');
  assert.equal(display.errorMessage({ detail: { unknown: { secret: 'no' } } }), 'FL could not complete this request.');
});
test('void navigation accepts posting receipt numbers without arbitrary URLs', () => {
  assert.equal(display.voidReceipt('Please void ADJ-261015-242.'), 'ADJ-261015-242');
  assert.equal(display.voidReceipt('javascript:alert(1)'), null);
  assert.equal(display.voidReceipt('VD-261015-001'), null);
});
test('receipt renderer translates status and reason fields without changing product or lot names', () => {
  ledger.lang = 'en';
  const html = ledger.draftHtml({ product_name: 'UNIDENTIFIED', lot_code: 'physical_count', reason_code: 'physical_count', identity_status: 'UNIDENTIFIED', proof_status: 'photo_pending' });
  assert.match(html, /<dd>UNIDENTIFIED<\/dd>/);
  assert.match(html, /<dd>physical_count<\/dd>/);
  assert.match(html, /<dd>Physical count<\/dd>/);
  assert.match(html, /Lot identity needs to be checked/);
  assert.match(html, /Photo pending/);
});
test('receipt reasons render catalog labels and safe escaped notes', () => {
  ledger.lang = 'es';
  const html = ledger.draftHtml({ reason: { code: 'missing_receipt', label_en: 'Missing receipt', label_es: 'Falta recibo', note: '<script>' } });
  assert.match(html, /Falta recibo/);
  assert.match(html, /&lt;script&gt;/);
  assert.doesNotMatch(html, /missing_receipt|<script>/);
  ledger.lang = 'en';
});

test('shipment drafts show line products and sums while excluding service charges', () => {
  const fields = Object.fromEntries(display.draftFields({ action: 'ship_bulk', draft: { lines: [
    { product: 'Classic', quantity_lb: 50, cases: 2 },
    { product_name: 'Chocolate', quantity_lb: 25, cases: 1 },
    { product: 'Freight', quantity_lb: 999, is_service: true }
  ] } }));
  assert.equal(fields.Product, 'Classic, Chocolate');
  assert.equal(fields.Quantity, '75 lb');
  assert.equal(fields.Cases, '3');
});
test('receipt sentences use effective amended line weights before the stored draft', () => {
  const [section] = display.readSections({ tool: 'receipt_lookup', result: {
    receipt_number: 'RCV-261015-009', action: 'receive', draft: { product_name: 'Oats', total_lb: 50 },
    transactions: [{ lines: [{ product_name: 'Oats', lot_code: 'OAT-1', quantity_lb: 42 }] }]
  } });
  assert.equal(section.sentence, 'Received 42 lb Oats.');
  assert.equal(section.rows[0][2], '42');
});
test('shipment receipt totals are positive and describe all shipped products', () => {
  const summary = display.receiptSummary({ action: 'ship_bulk', transactions: [{ lines: [
    { product_name: 'Oats', quantity_lb: -50 }, { product_name: 'Seeds', quantity_lb: -25 }
  ] }] });
  assert.equal(summary, 'Shipped 75 lb across 2 products');
});

test('receipt status codes are plain and translated without changing product names', () => {
  const ledger = require('../dashboard/ledger-ui.js');
  ledger.lang='en'; const html=ledger.draftHtml({status:'committed',effective_status:'posted',product_name:'active'});
  assert.match(html,/>Recorded</); assert.match(html,/>Counts in stock</); assert.match(html,/>active</);
  ledger.lang='es'; assert.match(ledger.draftHtml({status:'committed'}),/>Registrado</); ledger.lang='en';
});


test('legacy Today cards explicitly count only the signed-in person in both languages', () => {
  const card = { tool: 'today_entries', result: { receipts: [] } };
  assert.equal(display.readSections(card)[0].sentence, "You've recorded 0 entries today. Team total: see Today's entries.");
  assert.equal(display.readSections(card, { language: 'es' })[0].sentence, 'Has registrado 0 entradas hoy. Total del equipo: consulta las entradas de hoy.');
  card.result.receipts.push({ receipt_number: 'RCV-261015-001' });
  assert.equal(display.readSections(card)[0].sentence, "You've recorded 1 entry today. Team total: see Today's entries.");
  assert.equal(display.readSections(card, { language: 'es' })[0].sentence, 'Has registrado 1 entrada hoy. Total del equipo: consulta las entradas de hoy.');
});
test('direct all-staff Today cards keep the team count clear in both languages', () => {
  const card = { tool: 'today_entries_all', result: { receipts: Array.from({ length: 6 }, () => ({})) } };
  assert.equal(display.readSections(card)[0].sentence, '6 entries today.');
  assert.equal(display.readSections(card, { language: 'es' })[0].sentence, '6 entradas hoy.');
});
test('finished inventory displays supplied pounds and cases together in both languages', () => {
  const card = { tool: 'inventory_lookup', result: { results: [{ product: 'Classic 20 lb case', total_on_hand: 1600, unit_count: 80,
    lots: [{ lot: 'FIN-1', qty_on_hand: 1600, unit: '20 lb case' }] }] } };
  assert.equal(display.readSections(card)[0].sentence, 'Classic 20 lb case: 1,600 lb (80 cases) in 1 lot.');
  assert.equal(display.readSections(card, { language: 'es' })[0].sentence, 'Classic 20 lb case: 1,600 lb (80 cajas) en 1 lote.');
  assert.deepEqual(display.readSections(card)[0].rows, [['FIN-1', '1,600', 'lb']], 'never derive per-lot counts from a product total');
  card.result.results[0].unit_count = 0;
  assert.match(display.readSections(card)[0].sentence, /1,600 lb \(0 cases\)/, 'zero is a supplied count');
  delete card.result.results[0].unit_count;
  assert.equal(display.readSections(card)[0].sentence, 'Classic 20 lb case: 1,600 lb in 1 lot.', 'a packaging name does not establish a case count');
});
test('inventory case totals require complete counts covering the whole pound total', () => {
  const item = { product: 'Finished', total_on_hand: 1600, lots: [
    { lot: 'FIN-1', qty_on_hand: 1000, unit_count: 50 }, { lot: 'FIN-2', qty_on_hand: 600, unit_count: 30 }
  ] };
  const card = { tool: 'inventory_lookup', result: { results: [item] } };
  assert.equal(display.readSections(card)[0].sentence, 'Finished: 1,600 lb (80 cases) in 2 lots.');
  assert.deepEqual(display.readSections(card)[0].columns, ['Lot', 'Quantity', 'Unit', 'Cases']);
  assert.deepEqual(display.readSections(card)[0].rows, [['FIN-1', '1,000', 'lb', '50'], ['FIN-2', '600', 'lb', '30']]);
  delete item.lots[1].unit_count;
  assert.equal(display.readSections(card)[0].sentence, 'Finished: 1,600 lb in 2 lots.');
  assert.deepEqual(display.readSections(card)[0].rows[1], ['FIN-2', '600', 'lb', '—']);
  item.lots[1].unit_count = 30;
  item.total_on_hand = 1800;
  assert.equal(display.readSections(card)[0].sentence, 'Finished: 1,800 lb in 2 lots.', 'partial lot counts cannot become the result total');
});
test('mixed inventory units remain separate instead of being added as pounds', () => {
  const card = { tool: 'inventory_lookup', result: { results: [{ product: 'Mixed', lots: [
    { lot: 'BULK', quantity_lb: 100, unit_count: 5 }, { lot: 'CASES', quantity: 12, unit: 'cases' }
  ] }] } };
  assert.equal(display.readSections(card)[0].sentence, 'Mixed: 100 lb and 12 cases in 2 lots.');
  assert.equal(display.readSections(card, { language: 'es' })[0].sentence, 'Mixed: 100 lb y 12 cajas en 2 lotes.');
  assert.deepEqual(display.readSections(card)[0].rows, [['BULK', '100', 'lb', '5'], ['CASES', '12', 'cases', '—']]);
});


for (const language of ['en', 'es']) test(`customer draft displays contact details without stock placeholders in ${language}`, () => {
  const labels = language === 'en'
    ? ['Customer', 'Contact', 'Email', 'Phone', 'Address', 'Note']
    : ['Cliente', 'Contacto', 'Correo', 'Teléfono', 'Dirección', 'Nota'];
  const values = ['Sunrise Bakery', 'A Person', 'orders@example.test', '555-1234', '10 Main St', language === 'en' ? 'Call before delivery' : 'Llamar antes de entregar'];
  const pairs = display.draftFields({ action: 'create_customer', draft: { customer: {
    name: values[0], contact_name: values[1], email: values[2], phone: values[3], address: values[4],
    notes: 'Call before delivery', notes_es: 'Llamar antes de entregar'
  } } }, { language });
  assert.deepEqual(pairs, labels.map((label, i) => [label, values[i]]));
  assert.doesNotMatch(JSON.stringify(pairs), /Product|Producto|Quantity|Cantidad|not supplied|no indicad/);
});

for (const language of ['en', 'es']) test(`order draft preserves PO, requested date, physical quantities and service prices in ${language}`, () => {
  const stock = { product: 'Classic Granola', quantity: 2, unit: 'cases', quantity_lb: 50, unit_price: 30, amount: 60 };
  const service = { product: 'Pallet Charge', quantity: 3, unit: 'each', quantity_lb: 999, unit_price: 12.5, amount: 37.5, is_service: true };
  const fields = Object.fromEntries(display.draftFields({ action: 'create_order', draft: {
    customer: 'Sunrise Bakery', customer_po: 'SUNRISE-PO-1', requested_ship_date: '2026-10-20', lines: [stock, service]
  } }, { language }));
  assert.equal(fields[language === 'en' ? 'Product' : 'Producto'], 'Classic Granola');
  assert.equal(fields[language === 'en' ? 'Quantity' : 'Cantidad'], '50 lb', 'service charges never inflate the order weight');
  assert.equal(fields[language === 'en' ? 'Customer' : 'Cliente'], 'Sunrise Bakery');
  assert.equal(fields[language === 'en' ? 'Customer PO' : 'Orden de compra del cliente'], 'SUNRISE-PO-1');
  assert.equal(fields[language === 'en' ? 'Requested ship date' : 'Fecha solicitada de envío'], '2026-10-20');
  const stockFields = Object.fromEntries(display.lineFields(stock, language));
  assert.equal(stockFields[language === 'en' ? 'Quantity' : 'Cantidad'], language === 'en' ? '2 cases' : '2 cajas');
  assert.equal(stockFields[language === 'en' ? 'Pounds' : 'Libras'], 50);
  const serviceFields = Object.fromEntries(display.lineFields(service, language));
  assert.equal(serviceFields[language === 'en' ? 'Type' : 'Tipo'], language === 'en' ? 'Service charge' : 'Cargo por servicio');
  assert.equal(serviceFields[language === 'en' ? 'Quantity' : 'Cantidad'], language === 'en' ? '3 each' : '3 cada uno');
  assert.equal(serviceFields[language === 'en' ? 'Pounds' : 'Libras'], undefined);
  assert.equal(serviceFields[language === 'en' ? 'Unit price' : 'Precio unitario'], 12.5);
  assert.equal(serviceFields[language === 'en' ? 'Amount' : 'Importe'], 37.5);
  assert.equal(Object.fromEntries(display.lineFields({ ...service, is_service: false, line_type: 'service' }, language))[language === 'en' ? 'Pounds' : 'Libras'], undefined);
});

test('line fields retain legacy physical quantities and zero service prices', () => {
  const physical = Object.fromEntries(display.lineFields({ original_quantity: 2, original_unit: 'bags', quantity_lb: 50 }));
  assert.equal(physical.Quantity, '2 bags');
  assert.equal(physical.Pounds, 50);
  const service = Object.fromEntries(display.lineFields({ quantity: 1, unit: 'each', is_service: true, unit_price: 0, amount: 0 }));
  assert.equal(service['Unit price'], 0);
  assert.equal(service.Amount, 0);
});

for (const language of ['en', 'es']) test(`customer duplicate offers only a usable existing customer in ${language}`, () => {
  const blocker = { code: 'CUSTOMER_ALREADY_EXISTS', existing_customer: { id: 42, name: 'Sunshine Bakery', active: true } };
  assert.deepEqual(display.existingCustomerChoice(blocker, language), language === 'en' ? {
    label: 'Use existing customer: Sunshine Bakery', text: 'Use the existing customer Sunshine Bakery for the order.'
  } : { label: 'Usar cliente existente: Sunshine Bakery', text: 'Usa el cliente existente Sunshine Bakery para el pedido.' });
  for (const unusable of [{}, { ...blocker, code: 'OTHER_BLOCKER' }, { ...blocker, existing_customer: { id: 42 } },
    { ...blocker, existing_customer: { ...blocker.existing_customer, active: false } }]) {
    assert.equal(display.existingCustomerChoice(unusable, language), null);
  }
});

for (const language of ['en', 'es']) test(`Add customer stays available for office and owner only on the first choices page in ${language}`, () => {
  const expected = language === 'en' ? {
    label: "Add 'Sunrise Bakery' as a new customer", text: 'Create a new customer named Sunrise Bakery first, then continue the order.'
  } : { label: "Agregar 'Sunrise Bakery' como cliente nuevo", text: 'Crea primero un cliente nuevo llamado Sunrise Bakery y luego continúa el pedido.' };
  for (const outcome of ['none', 'ambiguous', 'match']) for (const offset of [undefined, null, 0]) {
    const card = { kind: 'choices', resolution_kind: 'customer', query: 'Sunrise Bakery', result: { outcome, offset } };
    for (const role of ['office', 'owner']) assert.deepEqual(display.newCustomerChoice(card, role, language), expected);
    for (const role of ['floor', 'readonly', undefined]) assert.equal(display.newCustomerChoice(card, role, language), null);
    assert.equal(display.newCustomerChoice({ ...card, result: { ...card.result, offset: 10 } }, 'office', language), null);
    assert.equal(display.newCustomerChoice({ ...card, resolution_kind: 'product' }, 'office', language), null);
    assert.equal(display.newCustomerChoice({ ...card, query: '' }, 'office', language), null);
  }
  assert.deepEqual(display.newCustomerChoice({ resolution_kind: 'customer', query: 'Sunrise Bakery', result: {} }, 'office', language), expected,
    'legacy first-page cards without offset retain the Add choice');
});

for (const language of ['en', 'es']) {
  test(`Round B open orders are readable rows with correct singular in ${language}`, () => {
    const [section] = display.readSections({tool: 'open_orders', result: {product: 'Kookies & Kreme', orders: [{order_number: 'SO-261015-002', customer: 'Sunshine', items: [{product: 'Kookies & Kreme'}, {product: 'Freight', is_service: true}], ship_date: '2026-10-16', remaining_lb: 15}]}}, {language});
    assert.match(section.sentence, language === 'en' ? /There is 1 open order containing Kookies & Kreme/ : /Hay 1 pedido abierto con Kookies & Kreme/);
    assert.equal(section.layout, 'orders');
    assert.deepEqual(section.rows[0], ['SO-261015-002', 'Sunshine', 'Kookies & Kreme', '2026-10-16', '15 lb']);
  });
  test(`Round B lot questions answer with product and quantity in ${language}`, () => {
    const [section] = display.readSections({tool: 'lot_lookup', result: {lot_code: 'SUN-ABCD', product_name: 'Sunflower Seeds', quantity_on_hand: 50}}, {language});
    assert.equal(section.sentence, language === 'en' ? 'Lot SUN-ABCD: 50 lb of Sunflower Seeds.' : 'Lote SUN-ABCD: 50 lb de Sunflower Seeds.');
    assert.doesNotMatch(section.sentence, /FL returned|FL devolvió/);
  });
  test(`Round B unresolved Different lot inputs cannot become negative available stock in ${language}`, () => {
    const [section] = display.readSections({tool: 'inventory_lookup', result: {results: [{product: 'Seeds', total_on_hand: 96, lots: [{lot: 'SEED-ABCD', qty_on_hand: 100}, {lot: 'UNVERIFIED-999', qty_on_hand: -4}]}]}}, {language});
    assert.match(section.sentence, /Seeds: 100 lb/);
    assert.deepEqual(section.rows, [['SEED-ABCD', '100', 'lb']]);
    assert.match(section.notes[0], language === 'en' ? /unverified, awaiting follow-up/ : /sin verificar, pendiente de seguimiento/);
    assert.doesNotMatch(JSON.stringify(section), /-4/);
  });
  test(`Round B receipt effective void status is visible in ${language}`, () => {
    const [section] = display.readSections({tool: 'receipt_lookup', result: {receipt_number: 'ADJ-261015-240', effective_status: 'voided', action: 'adjust', draft: {product_name: 'Seeds', delta_lb: -5}}}, {language});
    assert.ok(section.notes.includes(language === 'en' ? 'VOIDED' : 'ANULADO'));
  });
}

for (const language of ['en', 'es']) test(`round C errors and server Today counts are readable in ${language}`, () => {
  for (const value of ['Internal server error', {detail: {message: 'Internal server error'}}, {detail: 'FL returned an unreadable response. Please try again.'}]) {
    const shown = display.errorMessage(value, language);
    assert.equal(shown, display.unavailable(language));
    assert.doesNotMatch(shown, /Internal server error|unreadable response/);
  }
  const [section] = display.readSections({tool: 'today_entries', result: {receipts: [], own_count: 5, team_count: 23}}, {language});
  assert.equal(section.sentence, language === 'en' ? 'You recorded 5 entries today; the team recorded 23.' : 'Registraste 5 entradas hoy; el equipo registró 23.');
});


test('stock questions render one plain bulk and finished answer without lot tables', () => {
  const card = {tool: 'inventory_lookup', result: {stock_summary_only: true, summary_en: 'Classic #9: 100 lb in bulk and 50 lb in finished products.', summary_es: 'Classic #9: 100 lb a granel y 50 lb en productos terminados.', results: [{product: 'Batch Classic Granola #9', total_on_hand: 100, lots: [{lot: 'B-1', qty_on_hand: 100}]}, {product: 'Granola Classic 25 LB', total_on_hand: 50, lots: [{lot: 'F-1', qty_on_hand: 50}]}]}};
  for (const language of ['en', 'es']) {
    const sections = display.readSections(card, {language});
    assert.equal(sections.length, 1);
    assert.equal(sections[0].sentence, card.result['summary_' + language]);
    assert.deepEqual(sections[0].rows, []);
  }
});


test('empty physical stock still displays the unverified follow-up notice in both languages', () => {
  const result = {stock_summary_only: true, results: [], product_names: ['Oats'], summary_en: 'There is no stock of “Oats”. Some lots are unverified, awaiting follow-up.', summary_es: 'No hay existencias de «Oats». Hay lotes sin verificar, pendientes de seguimiento.'};
  for (const language of ['en', 'es']) {
    const [section] = display.readSections({tool: 'inventory_lookup', result}, {language});
    assert.equal(section.sentence, result['summary_' + language]);
    assert.deepEqual(section.rows, []);
  }
});
