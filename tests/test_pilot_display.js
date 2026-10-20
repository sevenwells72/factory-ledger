const {test}=require('node:test');
const assert=require('node:assert/strict');
const ui=require('../dashboard/ledger-ui.js');
for(const lang of ['en','es'])test(`completed ship receipt uses current proof everywhere (${lang})`,()=>{
 ui.lang=lang;
 const receipt={shipment:{proof_status:'complete',proof_due_at:'2026-10-22T03:59:00Z'},draft:{photo:'pending'},response:{proof_status:'photo_pending',proof_due_at:'2026-10-22T03:59:00Z',lots_used:[{lot_code:'JUST-PACKED',quantity_lb:15}]}};
 const html=ui.draftHtml(ui.receiptData(receipt));
 assert.ok(html.includes(lang==='en'?'Photo attached ✓':'Foto adjunta ✓'));
 assert.doesNotMatch(html,/Photo pending|Photo due|Foto pendiente|Fecha límite de foto|11:59/);
 assert.match(html,/JUST-PACKED/);assert.match(html,/15/);
 assert.equal(receipt.draft.photo,'pending');assert.equal(receipt.response.proof_status,'photo_pending');
});
test('pending ship proof keeps the pending state and deadline',()=>{
 ui.lang='en';const html=ui.draftHtml(ui.receiptData({shipment:{proof_status:'photo_pending',proof_due_at:'2026-10-22T03:59:00Z'}}));
 assert.match(html,/Photo pending/);assert.match(html,/Photo due/);assert.doesNotMatch(html,/Photo attached ✓/);
});
test('make shows chosen plan first and preserves every other lot behind Show more lots',()=>{
 ui.lang='en';const lots=Array.from({length:65},(_,i)=>({lot_code:`LOT-${i}`,available_lb:20}));
 const draft={ingredients:[{ingredient_id:1,ingredient_name:'Oats',needed_lb:25,lots}],input_plan:[{product_id:1,lot_id:60,lot_code:'LOT-59',quantity_lb:20,confirmed:false},{product_id:1,lot_id:61,lot_code:'LOT-60',quantity_lb:5,confirmed:false}]};
 const before=JSON.stringify(draft),phone=ui.makeDraftHtml(draft,true),desktop=ui.makeDraftHtml(draft,false);
 assert.match(phone,/<details class="more-lots">/);assert.match(desktop,/<details class="more-lots" open>/);
 assert.ok(phone.indexOf('LOT-59')<phone.indexOf('<details'));assert.ok(phone.indexOf('LOT-60')<phone.indexOf('<details'));
 assert.ok(phone.indexOf('LOT-0')>phone.indexOf('<details'));
 for(const lot of lots)assert.ok(phone.includes(lot.lot_code));
 assert.equal(JSON.stringify(draft),before);assert.equal(draft.input_plan.every(lot=>!lot.confirmed),true);
});
test('pilot details and notes render escaped on draft and receipt',()=>{
 ui.lang='en';const html=ui.draftHtml({found_location:'Dry storage rack 3',target_product:'Classic Granola 12 oz',note:'<b>private note</b>',lines:[{product:'Classic',lots:[{lot_code:'PACKED-TODAY',quantity_lb:12}]}]});
 for(const label of ['Found location','Dry storage rack 3','Packed product','Classic Granola 12 oz','Note','PACKED-TODAY'])assert.ok(html.includes(label));
 assert.doesNotMatch(html,/<b>/);assert.match(html,/&lt;b&gt;/);
});
test('void note remains optional and passes the exact trimmed note to prepare',()=>{
 assert.deepEqual(ui.voidRequest(12,'   ').body,{reason:'data_entry_error',client_source:'dashboard'});
 assert.equal(ui.voidRequest(12,' Wrong entry after recount ').body.note,'Wrong entry after recount');
});

for(const action of ['create_customer','create_order'])test(`${action} receipt replaces historical midnight snapshot with actual entry time`,()=>{
 ui.lang='en';const original={action,happened_at:'2026-10-19T19:37:00-04:00',draft:{happened_at:'2026-10-19T00:00:00-04:00'},response:{entry_timing:{happened_at:'2026-10-19T00:00:00-04:00'}}};
 const html=ui.draftHtml(ui.receiptData(original));assert.match(html,/7:37 PM/);assert.doesNotMatch(html,/12:00 AM/);
 assert.equal(original.draft.happened_at,'2026-10-19T00:00:00-04:00');
});

for(const [code,en,es] of [['floor','Floor','Piso'],['office','Office','Oficina'],['owner','Owner','Propietario']])test(`actor role ${code} has bilingual draft and Assistant display labels`,()=>{
 const display=require('../dashboard/assistant-display.js');for(const [lang,label] of [['en',en],['es',es]]){ui.lang=lang;const html=ui.draftHtml({actor:{role:code}});assert.ok(html.includes('>'+label+'</span>'));assert.equal(display.plainLabel(code,lang),label);assert.match(html,/data-en=/);assert.match(html,/data-es=/);}
});
