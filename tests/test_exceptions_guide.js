// Exceptions guidance text (2026-11-05): every resolution has use/example/effect in EN and ES,
// explanations and confirmations come from the item's real values.
const {test}=require('node:test');const assert=require('node:assert/strict');
const fs=require('node:fs');const path=require('node:path');
const g=require('../dashboard/exceptions-guide.js');

const shortage={id:73,kind:'SHORTAGE',product:{name:'Canola oil'},lot:{lot_code:'26-09-17-CBSF-001'},
  detail:{needed_lb:44,on_hand_before_lb:30,balance_after_lb:-14,short_lb:14,action:'make'}};

test('every resolution the server can offer has use, example and effect in both languages',()=>{
  const server=fs.readFileSync(path.join(__dirname,'..','exceptions_enforcement.py'),'utf8');
  const block=server.match(/RESOLUTION_KINDS = \{([\s\S]*?)\n\}/)[1];
  const kinds=new Set([...block.matchAll(/'([a-z_]+)'/g)].map(m=>m[1]).filter(k=>k===k.toLowerCase()));
  for(const k of ['approved','declined','waived'])kinds.add(k);
  assert.ok(kinds.has('unverified')&&kinds.has('counted')&&kinds.has('identified'));
  for(const kind of kinds)for(const lang of ['en','es']){
    const o=g.option(kind,lang);assert.ok(o,`${kind} ${lang}`);
    for(const key of ['use','example','effect'])assert.ok(o[key]&&o[key].length>10,`${kind}.${key} ${lang}`);
  }
});

test('effects state the real stock behaviour',()=>{
  assert.match(g.option('counted','en').effect,/Sets this lot to the pounds you enter/);
  for(const kind of ['missing_movement','voided','unverified','written_off','waived','acknowledged'])
    assert.match(g.option(kind,'en').effect,/Changes no stock/,kind);
  assert.match(g.option('unverified','en').effect,/Unverified/);
  assert.match(g.option('unverified','es').effect,/Sin verificar/);
  assert.match(g.option('missing_movement','en').effect,/stays open for the rest/);
  assert.match(g.option('voided','en').effect,/refuses if that entry is not voided/);
});

test('shortage explanation is built from the item data',()=>{
  assert.equal(g.explain(shortage,'en'),'The make used 44 lb of Canola oil from lot 26-09-17-CBSF-001, but the system only had 30 lb. '
    +'After that entry the lot was at -14 lb on the books. Likely causes: a delivery never entered, the wrong lot picked, or an earlier entry overstated.');
  assert.match(g.explain(shortage,'es'),/^La producción usó 44 lb de Canola oil del lote 26-09-17-CBSF-001, pero el sistema solo tenía 30 lb\./);
  const empty={...shortage,detail:{...shortage.detail,action:'pack',on_hand_before_lb:-25,needed_lb:25,balance_after_lb:-50}};
  assert.match(g.explain(empty,'en'),/^The pack used 25 lb .* already showed -25 lb \(none available\)\./);
  assert.match(g.explain(empty,'es'),/^El empaque usó 25 lb/);
  const partial={...shortage,evidence:{claimed_lb:4,remaining_lb:10}};
  assert.match(g.explain(partial,'en'),/Receipts already explain 4 lb; 10 lb is still unexplained\./);
  assert.match(g.explain({id:1,kind:'SHORTAGE',detail:{}},'en'),/More of this lot was used than the system had\./);
});

test('unidentified-lot and other kinds explain in plain words, never empty',()=>{
  const lot={id:5,kind:'UNIDENTIFIED_LOT',product:{name:'Oats'},lot:{lot_code:'26-11-01-OATS-002'},detail:{},is_opening_stock:true};
  assert.match(g.explain(lot,'en'),/without the supplier's lot code.*opening stock count.*bag, box, label or BOL/);
  for(const kind of ['LARGE_CORRECTION','KOSHER_ATTESTATION','LATE_ENTRY','BOL_PHOTO_MISSING','SHIPMENT_PROOF_MISSING','PRE_MAKE_ADJUST','ASSISTANT_OFFLINE','SOMETHING_NEW'])
    for(const lang of ['en','es'])assert.ok(g.explain({id:1,kind,detail:{}},lang).length>20,kind+lang);
});

test('confirmation line uses the values about to be sent',()=>{
  assert.equal(g.confirmLine(shortage,'counted',{counted_lb:10},'en'),'You are about to: set Canola oil lot 26-09-17-CBSF-001 to 10 lb and close exception #73.');
  assert.equal(g.confirmLine(shortage,'counted',{counted_lb:0},'es'),'Estás a punto de: poner el lote 26-09-17-CBSF-001 de Canola oil en 0 lb y cerrar la excepción #73.');
  assert.match(g.confirmLine(shortage,'missing_movement',{receipt_number:'RCV-261105-002'},'en'),/use receipt RCV-261105-002 .* Stock does not change\./);
  assert.match(g.confirmLine(shortage,'unverified',{},'en'),/close exception #73 without a count\. Canola oil lot 26-09-17-CBSF-001 keeps its book balance\. It will be marked Unverified\./);
  assert.match(g.confirmLine({...shortage,kind:'UNIDENTIFIED_LOT'},'identified',{supplier_lot_code:'2409-117'},'en'),/save supplier lot 2409-117 on Canola oil lot 26-09-17-CBSF-001 and close exception #73\./);
  for(const kind of Object.keys(g.OPTIONS))for(const lang of ['en','es'])assert.match(g.confirmLine(shortage,kind,{},lang),/#73/);
});

test('field helpers, tips and the help panel exist in both languages',()=>{
  for(const name of ['resolution','counted_lb','note','note_unverified','photo','photo_bol','receipt_number','supplier_lot_code','reason'])
    for(const lang of ['en','es'])assert.ok(g.field(name,lang).length>10,name+lang);
  assert.match(g.field('counted_lb','en'),/THIS lot left right now\. Enter 0 if it is empty\./);
  for(const name of ['refresh','help','explain','resolve','review','confirm'])for(const lang of ['en','es'])assert.ok(g.tip(name,lang).length>10);
  const en=g.help('en'),es=g.help('es');assert.equal(en.length,4);assert.equal(es.length,4);
  assert.deepEqual(en.map(s=>s.lines.length),es.map(s=>s.lines.length));
  assert.match(en[2].lines.join(' '),/Leave it open\. Tell the owner\. Do not guess\./);
});

test('exceptions page loads the guide before the page script and offers the no-count close',()=>{
  const html=fs.readFileSync(path.join(__dirname,'..','dashboard','exceptions.html'),'utf8');
  assert.ok(html.indexOf('exceptions-guide.js')>0&&html.indexOf('exceptions-guide.js')<html.indexOf('exceptions.js?'));
  const page=fs.readFileSync(path.join(__dirname,'..','dashboard','exceptions.js'),'utf8');
  assert.match(page,/unverified:\['Resolve without count','Resolver sin conteo'\]/);
  assert.match(page,/status=resolved&resolution_kind=unverified/);
});
