const test = require('node:test');
const assert = require('node:assert/strict');
const display = require('../dashboard/assistant-display.js');
const ledger = require('../dashboard/ledger-ui.js');
const assistant = require('../dashboard/fl-assistant.js');
for (const language of ['en','es']) {
  test(`receive evidence survives commit and lookup shapes in ${language}`, () => {
    const receipt = {receipt_number:'RCV-261016-001',action:'receive',lots:[{supplier_lot_code:'SUP-123'}],transactions:[{bol_reference:'BOL-987',lines:[]}],draft:{supplier_lot_code:'SUP-123'}};
    assert.deepEqual(display.receiptEvidence(receipt,language),[[language==='en'?'Supplier lot':'Lote del proveedor','SUP-123'],['BOL','BOL-987']]);
    const notes=display.readSections({result:receipt},{language})[0].notes.join(' ');assert.match(notes,/SUP-123/);assert.match(notes,/BOL-987/);
  });
  test(`quantity noise is rounded only in presentation in ${language}`, () => {
    const value=88.80000000000001;ledger.lang=language;
    assert.equal(display.quantityLabel(value,'lb',language),'88.8 lb');
    assert.equal(display.quantityLabel(1.236,'lb',language),'1.24 lb');
    assert.match(ledger.draftHtml({needed_lb:value,quantity_lb:'1.236',lot_code:'88.80000000000001'}),/<dd>88.8<\/dd>/);
    assert.match(ledger.draftHtml({lot_code:'88.80000000000001'}),/88.80000000000001/);
    assert.equal(value,88.80000000000001);
  });
  test(`Spanish refusal uses supplied plain reason in ${language}`, () => {
    assert.equal(display.errorMessage({code:'AMBIGUOUS_SUFFIX',message:'More than one lot matches. Enter the full code.',message_es:'Coinciden varios lotes. Escribe el código completo.'},language),language==='en'?'More than one lot matches. Enter the full code.':'Coinciden varios lotes. Escribe el código completo.');
  });
}
test('only quota errors suppress retry',()=>{
  assert.equal(assistant.assistantOffline({detail:{error_code:'ASSISTANT_QUOTA_EXHAUSTED'}}),true);
  assert.equal(assistant.assistantOffline({result:{detail:{code:'ASSISTANT_QUOTA_EXHAUSTED'}}}),true);
  assert.equal(assistant.assistantOffline({detail:{error_code:'ASSISTANT_UNAVAILABLE'}}),false);
});
test('Office handoffs allow only the intended local screens',()=>{
  assert.equal(assistant.safeOfficeLink('/dashboard/office.html?action=cancel_order&order=SO-261016-001'),'/dashboard/office.html?action=cancel_order&order=SO-261016-001');
  for(const href of ['https://example.com','javascript:alert(1)','//example.com','/dashboard/office.html.evil'])assert.equal(assistant.safeOfficeLink(href),null);
});
test('new unresolved exceptions sort before history; opening requires explicit marker',()=>{
  assert.deepEqual(ledger.exceptionQueue([{id:1,status:'open',opened_at:'2026-10-14'},{id:2,status:'resolved',opened_at:'2026-10-17'},{id:3,status:'escalated',opened_at:'2026-10-16'}]).map(row=>row.id),[3,1,2]);
  assert.equal(ledger.openingStock({lot_entry_source:'found_inventory'}),false);
  assert.equal(ledger.openingStock({is_opening_stock:true}),true);
  assert.equal(ledger.exceptionLabel('PRE_MAKE_ADJUST'),'Adjustment before a make');
  assert.equal(ledger.exceptionLabel('UNIDENTIFIED_LOT','es'),'Lote sin identificar');
});

test('known customer suggestions never offer duplicate creation',()=>{assert.equal(display.newCustomerChoice({resolution_kind:'customer',query:'Setton',existing_customer_only:true,result:{offset:0}},'owner'),null);});
