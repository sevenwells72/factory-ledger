const {test}=require('node:test');
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('dashboard/make-formula.js','utf8');
class Element{
  constructor(tag,text=''){this.tag=tag;this.textContent=String(text??'');this.children=[];this.parent=null;this.dataset={};this.value='';this.hidden=false;this.className='';}
  append(child){child.parent=this;this.children.push(child);}
  replaceChildren(...children){this.children=[];children.forEach(child=>this.append(child));}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(child=>child!==this);this.parent=null;}
  setAttribute(name,value){this[name]=value;}
}
const descendants=root=>[root,...root.children.flatMap(descendants)];
const plain=value=>JSON.parse(JSON.stringify(value));
function load(){
  const ui={lang:'en',t:(en,es)=>Object.assign(new String(ui.lang==='es'?es:en),{en,es}),formatNumber:n=>String(n),
    el(tag,text,parent){const node=new Element(tag,text);parent?.append(node);return node;},setText(node,value){node.textContent=String(value);}};
  vm.runInNewContext(source,{window:{FLLedger:ui}});return ui;
}
const candidates=[{lot_id:1,lot_code:'OATS-OLD-0001',received_on:'2026-10-01',available_lb:50},{lot_id:2,lot_code:'OATS-NEW-0002',received_on:'2026-10-20',available_lb:500},{lot_id:3,lot_code:'OATS-SPR-1002',received_on:'2026-10-30',available_lb:300}];
// Make-entry simplification PR-1: one suggested lot per ingredient, confirmed by a tap.
const one=()=>({pans:2,pan_lb:100,output_lb:200,tolerance_pct:10,ingredients:[
  {product_id:10,product_name:'Plain Oats',standard_lb:160,bags:[{lot_id:1,lot_code:'OATS-OLD-0001',quantity_lb:160,confirmed:false}],candidates:[{...candidates[0],supplier_lot:{kind:'supplier',code:'SUP-77'}},{...candidates[1],supplier_lot:{kind:'pooled',code:null}},{...candidates[2],supplier_lot:{kind:'no_lot',code:null}}]},
  {product_id:20,product_name:'Honey',standard_lb:40,bags:[{lot_id:9,lot_code:'HONEY-0009',quantity_lb:40,confirmed:true,confirmation_method:'selected'}],candidates:[{lot_id:9,lot_code:'HONEY-0009',available_lb:100}]}]});
function harness(data=one(),options={}){
  const ui=load(),card=new Element('section'),calls=[];let dirty=0,changed=0;
  const api=ui.formulaEntry(card,data,{dirty:()=>dirty++,changed:()=>changed++,update:(lines,next)=>calls.push({lines,next}),...options});
  const all=fn=>descendants(card).filter(fn),find=fn=>all(fn)[0],box=id=>find(n=>n.dataset.productId===String(id)),inBox=(id,fn)=>descendants(box(id)).filter(fn);
  return {ui,card,api,calls,all,find,box,inBox,get dirty(){return dirty;},get changed(){return changed;},
    tap:id=>inBox(id,n=>n.dataset.lotConfirm==='')[0],state:id=>inBox(id,n=>n.dataset.bagState==='')[0],
    choice:(id,lot)=>inBox(id,n=>n.dataset.lotId===String(lot))[0],lb:id=>inBox(id,n=>n.name==='bag_lb'),
    button:(id,text)=>inBox(id,n=>n.tag==='button'&&n.textContent===text)[0],
    check(){find(n=>n.tag==='button'&&n.textContent==='Check lots and pounds → new draft').onclick();}};
}
test('one suggested lot per ingredient, shown with supplier lot, received date and pounds — not confirmed until tapped',()=>{
  const h=harness();
  assert.equal(h.find(n=>n.dataset.formulaSummary==='').textContent.startsWith('2 pans × 100 lb = 200 lb'),true);
  assert.equal(h.find(n=>n.dataset.standard==='160').textContent,'Recipe: 160 lb');
  assert.equal(h.tap(10).textContent,'Tap to confirm: OATS-OLD-0001 · supplier lot SUP-77 · received 2026-10-01 · 50 lb');
  assert.equal(h.state(10).textContent,'Not confirmed — tap the lot you used');
  assert.equal(h.tap(20).textContent,'✓ HONEY-0009 · 100 lb','a lot FL already accepted stays confirmed');
  assert.equal(h.all(n=>n.name==='bag_code'||n.name==='bag_method').length,0,'no typed codes');
  assert.equal(h.api.pending(),1);assert.deepEqual(plain(h.api.selections()),[{lot_id:9,method:'selected',value:'HONEY-0009'}]);
  // The list: every in-stock lot, oldest marked; pooled stock never shows a supplier lot.
  assert.deepEqual(h.inBox(10,n=>n.dataset.lotId!==undefined).map(n=>n.textContent),['OATS-OLD-0001 · supplier lot SUP-77 · received 2026-10-01 · 50 lb','OATS-NEW-0002 · pooled — no single supplier lot · received 2026-10-20 · 500 lb','OATS-SPR-1002 · no lot on label · received 2026-10-30 · 300 lb']);
  assert.equal(h.choice(10,1)['aria-checked'],'true');assert.equal(h.box(10).children.find(n=>n.className==='formula-edit').open,undefined,'details start folded');
});
test('tapping the suggested lot confirms it without changing the plan',()=>{
  const h=harness();h.tap(10).onclick();
  assert.equal(h.state(10).textContent,'Confirmed — 160 lb from this lot');assert.equal(h.dirty,0);assert.ok(h.changed>0);
  assert.equal(h.api.pending(),0);
  assert.deepEqual(plain(h.api.selections()),[{lot_id:1,method:'selected',value:'OATS-OLD-0001'},{lot_id:9,method:'selected',value:'HONEY-0009'}]);
  h.tap(10).onclick();assert.equal(h.api.pending(),1,'a second tap takes the confirmation back');
});
test('a different lot is an intentional choice from the list and needs a re-check',()=>{
  const h=harness();h.choice(10,2).onclick();
  assert.equal(h.dirty,1);assert.equal(h.choice(10,2)['aria-checked'],'true');assert.equal(h.choice(10,1)['aria-checked'],'false');
  assert.match(h.tap(10).textContent,/^✓ OATS-NEW-0002/);
  h.check();const {lines,next}=plain(h.calls[0]);
  assert.deepEqual(lines[0],{ingredient_product_id:10,bags:[{lot_id:2,quantity_lb:160}],reason_code:null,note:null});
  assert.deepEqual(next,[{lot_id:2,method:'selected',value:'OATS-NEW-0002'},{lot_id:9,method:'selected',value:'HONEY-0009'}]);
});
test('changing pounds keeps the tap (D7)',()=>{
  const h=harness();h.tap(10).onclick();const [lb]=h.lb(10);lb.value='150';lb.oninput();
  assert.equal(h.state(10).textContent,'Confirmed — 150 lb from this lot');assert.equal(h.dirty,1);
});
test('a second lot needs its own lot choice and pounds; both post exactly',()=>{
  const h=harness();h.tap(10).onclick();const [first]=h.lb(10);first.value='50';first.oninput();
  h.button(10,'Add another lot').onclick();const [,second]=h.lb(10);
  assert.equal(second.value,'');assert.equal(h.box(10).children.find(n=>n.className==='formula-edit').open,true);
  second.value='110';second.oninput();h.check();
  assert.equal(h.calls.length,0,'pounds without a lot are refused in the browser');
  assert.ok(h.find(n=>n.className==='bad'&&!n.hidden&&/Choose a lot for every pound/.test(n.textContent)));
  const rows=h.inBox(10,n=>n.className==='formula-bag');descendants(rows[1]).find(n=>n.dataset.lotId==='2').onclick();
  h.check();const {lines,next}=plain(h.calls[0]);
  assert.deepEqual(lines[0].bags,[{lot_id:1,quantity_lb:50},{lot_id:2,quantity_lb:110}]);
  assert.deepEqual(next.filter(c=>c.lot_id!==9).map(c=>c.lot_id),[1,2]);
});
test('a lot with fewer book pounds explains the second-lot and shortage choice',()=>{
  const h=harness();const short=h.inBox(10,n=>n.dataset.bagShort==='')[0];
  assert.equal(short.hidden,false);assert.match(short.textContent,/^This lot shows only 50 lb on the books\. If you also used another lot/);
  const [lb]=h.lb(10);lb.value='50';lb.oninput();assert.equal(short.hidden,true);
});
test('removing a lot moves its pounds to the lot above',()=>{
  const h=harness();h.button(10,'Add another lot').onclick();const [,second]=h.lb(10);second.value='20';second.oninput();
  const rows=h.inBox(10,n=>n.className==='formula-bag');descendants(rows[1]).find(n=>n.tag==='button'&&n.textContent==='Remove this lot').onclick();
  assert.deepEqual(h.lb(10).map(i=>i.value),['180']);
});
test('not used shows the fixed reason list and sends the chosen reason',()=>{
  const h=harness(),reasonArea=h.box(20).children.find(n=>n.className==='formula-reason');
  assert.equal(reasonArea.hidden,true);h.button(20,'Not used (0 lb)').onclick();assert.equal(reasonArea.hidden,false);
  const select=h.inBox(20,n=>n.name==='ingredient_reason')[0];
  assert.deepEqual(plain(select.children.slice(1).map(o=>[o.value,o.textContent])),[['out_of_stock','Out of stock'],['not_used','Not used'],['substituted','Substituted'],['quality_problem','Quality problem'],['owner_instruction','Owner instruction'],['other','Other']]);
  select.value='other';h.inBox(20,n=>n.name==='ingredient_note')[0].value=' Jar cracked ';h.check();
  assert.deepEqual(plain(h.calls[0].lines[1]),{ingredient_product_id:20,bags:[],reason_code:'other',note:'Jar cracked'});
  assert.equal(h.api.selections().some(c=>c.lot_id===9),false,'an unused lot sends no confirmation');
});
test('more than 10% off recipe is shown; exactly 10% is not',()=>{
  const h=harness(),[lb]=h.lb(10),total=h.box(10).children.find(n=>n.dataset.total==='');
  lb.value='176';lb.oninput();assert.equal(total.dataset.over,'false');assert.equal(total.textContent,'Using 176 lb (+10%)');
  lb.value='177';lb.oninput();assert.equal(total.dataset.over,'true');assert.match(total.textContent,/^Using 177 lb \(\+10.6%\) — more than 10% off the recipe; the owner will review it\.$/);
});
test('Spanish labels follow the language',()=>{
  const ui=load();ui.lang='es';const card=new Element('section');ui.formulaEntry(card,one());
  const text=descendants(card).map(n=>n.textContent);
  assert.ok(text.includes('Receta: 160 lb'));assert.ok(text.includes('Agregar otro lote'));assert.ok(text.includes('No se usó (0 lb)'));assert.ok(text.includes('Cambiar lote o libras'));
  assert.ok(text.some(t=>t.startsWith('Toca para confirmar: OATS-OLD-0001 · lote proveedor SUP-77')));
});

test('oat bran swap card: recipe bran is checked against bran + swapped oats, labelled as a swap, no reason or warning',()=>{
  const ui=load(),card=new Element('section'),calls=[];
  const swapBlock={pans:1,pan_lb:350,output_lb:350,tolerance_pct:10,swap:{from_product_id:31,to_product_id:32,from_name:'Oat Bran',to_name:'Rolled Oats',swap_lb:18},ingredients:[
    {product_id:31,product_name:'Oat Bran',standard_lb:28,swap_role:'from',bags:[{lot_id:5,lot_code:'BRAN-0005',quantity_lb:10}],candidates:[{lot_id:5,lot_code:'BRAN-0005',available_lb:10}]},
    {product_id:32,product_name:'Rolled Oats',standard_lb:185,swap_role:'to',bags:[{lot_id:6,lot_code:'OATS-0006',quantity_lb:203}],candidates:[{lot_id:6,lot_code:'OATS-0006',available_lb:5000}]}]};
  const api=ui.formulaEntry(card,swapBlock,{update:(lines)=>calls.push(lines)});
  const nodes=descendants(card),box=id=>nodes.find(n=>n.dataset.productId===String(id)),inBox=(id,fn)=>descendants(box(id)).filter(fn);
  const legend=id=>inBox(id,n=>n.tag==='legend')[0].textContent,total=id=>inBox(id,n=>n.dataset.total==='')[0];
  assert.equal(legend(31),'Oat Bran → Rolled Oats (swap)');
  assert.equal(total(31).textContent,'Using 10 lb + 18 lb Rolled Oats = 28 lb');assert.equal(total(31).dataset.over,'false');
  assert.equal(total(32).textContent,'Using 203 lb (18 lb replace oat bran)');assert.equal(total(32).dataset.over,'false');
  assert.equal(inBox(31,n=>n.className==='formula-reason')[0].hidden,true);
  // The worker used the real bran instead: no swap, plain label, still exactly the recipe.
  const lb=id=>inBox(id,n=>n.name==='bag_lb')[0];
  lb(31).value='28';lb(31).oninput();lb(32).value='185';lb(32).oninput();
  assert.equal(legend(31),'Oat Bran');assert.equal(total(31).textContent,'Using 28 lb');assert.equal(total(32).textContent,'Using 185 lb');
  // Bran emptied and fully covered by oats: still a swap, the 0 lb line carries no reason.
  inBox(31,n=>n.tag==='button'&&n.textContent==='Not used (0 lb)')[0].onclick();lb(32).value='213';lb(32).oninput();
  assert.equal(legend(31),'Oat Bran → Rolled Oats (swap)');assert.equal(inBox(31,n=>n.className==='formula-reason')[0].hidden,true);
  const {lines}=api.read();assert.deepEqual(plain(lines[0]),{ingredient_product_id:31,bags:[],reason_code:null,note:null});
});

test('re-review N1: a 0 lb card the recipe does not use (rolled oats for the swap) asks for no reason',()=>{
  const ui=load(),card=new Element('section');
  const swapBlock={pans:1,pan_lb:100,output_lb:100,tolerance_pct:10,swap:{from_product_id:31,to_product_id:32,from_name:'Oat Bran',to_name:'Rolled Oats',swap_lb:0},ingredients:[
    {product_id:31,product_name:'Oat Bran',standard_lb:30,swap_role:'from',bags:[{lot_id:5,lot_code:'BRAN-0005',quantity_lb:30}],candidates:[{lot_id:5,lot_code:'BRAN-0005',available_lb:10}]},
    {product_id:32,product_name:'Rolled Oats',standard_lb:0,swap_role:'to',bags:[],candidates:[{lot_id:6,lot_code:'OATS-0006',available_lb:5000}]}]};
  const api=ui.formulaEntry(card,swapBlock,{});
  const box=descendants(card).find(n=>n.dataset.productId==='32'),inBox=fn=>descendants(box).filter(fn);
  assert.equal(inBox(n=>n.className==='formula-reason')[0].hidden,true);
  assert.equal(inBox(n=>n.dataset.total==='')[0].textContent,'Using 0 lb');
  assert.deepEqual(plain(api.read().lines[1]),{ingredient_product_id:32,bags:[],reason_code:null,note:null});
});

// PR #176 staging fixes (ChatGPT Work acceptance, 2026-11-02).
test('a negative amount is kept as typed, shown as an error, and never becomes Not used',()=>{
  const h=harness();h.tap(10).onclick();const [lb]=h.lb(10);lb.value='-25';lb.oninput();
  const total=h.box(10).children.find(n=>n.dataset.total==='');
  assert.equal(lb.value,'-25','the typed value stays for Arturo to fix');
  assert.equal(total.dataset.invalid,'true');assert.match(total.textContent,/^Pounds cannot be negative/);
  assert.notEqual(total.textContent,'Not used — 0 lb. Choose a reason.');
  assert.equal(h.box(10).children.find(n=>n.className==='formula-reason').hidden,true,'no reason list: it was not marked Not used');
  assert.equal(h.inBox(10,n=>n.className==='formula-bag').length,1,'the lot row stays');
  assert.match(h.tap(10).textContent,/^✓ OATS-OLD-0001 · supplier lot SUP-77/,'the chosen lot and its supplier lot stay');
  const error=h.inBox(10,n=>n.dataset.bagError==='')[0];assert.equal(error.hidden,false);assert.match(error.textContent,/"-25" lb is not allowed/);
  assert.equal(lb['aria-invalid'],'true');assert.equal(h.api.invalid(),1);
  h.check();assert.equal(h.calls.length,0,'nothing is sent to FL');assert.ok(h.find(n=>n.className==='bad'&&!n.hidden&&/Pounds cannot be negative \(Plain Oats\)/.test(n.textContent)));
  lb.value='150';lb.oninput();assert.equal(h.api.invalid(),0);assert.equal(error.hidden,true);
  h.check();assert.deepEqual(plain(h.calls[0].lines[0].bags),[{lot_id:1,quantity_lb:150}]);
});
test('a pasted negative amount (change event) on a second lot is refused the same way',()=>{
  const h=harness();h.button(10,'Add another lot').onclick();const [,second]=h.lb(10);
  second.value='-0.5';second.onchange();assert.equal(h.api.invalid(),1);assert.equal(second.value,'-0.5');
  assert.equal(h.inBox(10,n=>n.className==='formula-bag').length,2);
});
test('removing another lot never pours pounds into an invalid amount',()=>{
  const h=harness();const [first]=h.lb(10);first.value='-3';first.oninput();
  h.button(10,'Add another lot').onclick();const [,second]=h.lb(10);second.value='20';second.oninput();
  const rows=h.inBox(10,n=>n.className==='formula-bag');descendants(rows[1]).find(n=>n.tag==='button'&&n.textContent==='Remove this lot').onclick();
  assert.equal(h.lb(10)[0].value,'-3');
});
test('0 lb typed is not Not used: FL asks for the pounds or the explicit Not used button',()=>{
  const h=harness();const [lb]=h.lb(10);lb.value='0';lb.oninput();
  const total=h.box(10).children.find(n=>n.dataset.total==='');
  assert.match(total.textContent,/^0 lb entered\. Enter the pounds you used, or tap "Not used \(0 lb\)"/);
  assert.equal(h.box(10).children.find(n=>n.className==='formula-reason').hidden,true);
  h.check();assert.equal(h.calls.length,0);
  h.button(10,'Not used (0 lb)').onclick();assert.equal(total.textContent,'Not used — 0 lb. Choose a reason.');
  h.inBox(10,n=>n.name==='ingredient_reason')[0].value='out_of_stock';h.check();
  assert.deepEqual(plain(h.calls[0].lines[0]),{ingredient_product_id:10,bags:[],reason_code:'out_of_stock',note:null});
});
test('a chosen lot with zero or negative book stock keeps its full identity and is never "available"',()=>{
  const data=one();data.ingredients[1].bags[0]={lot_id:8,lot_code:'HONEY-0008',quantity_lb:40,confirmed:false};
  data.ingredients[1].unstocked_lots=[{lot_id:8,lot_code:'HONEY-0008',supplier_lot:{kind:'supplier',code:'HSUP-1'},received_on:'2026-09-01',book_lb:-10,available_lb:0}];
  const h=harness(data);
  assert.equal(h.tap(20).textContent,'Tap to confirm: HONEY-0008 · supplier lot HSUP-1 · received 2026-09-01 · on books -10 lb (none available)');
  assert.ok(!/available_lb|\b-10 lb available/.test(h.tap(20).textContent));
  const short=h.inBox(20,n=>n.dataset.bagShort==='')[0];assert.equal(short.hidden,false);
  assert.match(short.textContent,/^This lot shows -10 lb on the books — none available\..*recorded as a shortage on this lot/);
  assert.equal(h.choice(20,8)['aria-checked'],'true','FL does not switch to the lot with stock');
  h.tap(20).onclick();assert.deepEqual(plain(h.api.selections()).find(c=>c.lot_id===8),{lot_id:8,method:'selected',value:'HONEY-0008'});
});
test('one pan reads "1 pan" / "1 bandeja"',()=>{
  const data={...one(),pans:1,output_lb:100};
  assert.match(harness(data).find(n=>n.dataset.formulaSummary==='').textContent,/^1 pan × 100 lb = 100 lb\./);
  const ui=load();ui.lang='es';const card=new Element('section');ui.formulaEntry(card,data);
  assert.match(descendants(card).find(n=>n.dataset.formulaSummary==='').textContent,/^1 bandeja × 100 lb/);
});
