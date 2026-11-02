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
const block=()=>({pans:2,pan_lb:100,output_lb:200,tolerance_pct:10,ingredients:[
  {product_id:10,product_name:'Plain Oats',standard_lb:160,bags:[{lot_id:1,lot_code:'OATS-OLD-0001',quantity_lb:50,confirmed:true},{lot_id:2,lot_code:'OATS-NEW-0002',quantity_lb:110,confirmed:false}],candidates},
  {product_id:20,product_name:'Honey',standard_lb:40,bags:[{lot_id:9,lot_code:'HONEY-0009',quantity_lb:40,confirmed:false}],candidates:[{lot_id:9,lot_code:'HONEY-0009',available_lb:100}]}]});
function render(options={}){
  const ui=load(),card=new Element('section'),calls=[];let dirty=0;
  const api=ui.formulaEntry(card,block(),{confirmations:[{lot_id:1,method:'full_code',value:'OATS-OLD-0001'}],dirty:()=>dirty++,update:(lines,next)=>calls.push({lines,next}),...options});
  const all=fn=>descendants(card).filter(fn),find=fn=>all(fn)[0];
  const ingredient=id=>find(n=>n.dataset.productId===String(id));
  const within=(scope,fn)=>descendants(scope).filter(fn);
  const bags=id=>within(ingredient(id),n=>n.className==='lot-confirmation-row formula-bag');
  const field=(scope,name)=>within(scope,n=>n.name===name)[0];
  const button=(scope,text)=>within(scope,n=>n.tag==='button'&&n.textContent===text)[0];
  return {ui,card,api,calls,get dirty(){return dirty;},ingredient,bags,field,button,find,submit(){button(card,'Check bags and pounds → new draft').onclick();}};
}

test('pans summary and recipe pounds come from FL, with every FIFO bag pre-filled',()=>{
  const h=render();
  assert.equal(h.find(n=>n.dataset.formulaSummary==='').textContent.startsWith('2 pans × 100 lb = 200 lb'),true);
  assert.deepEqual(h.bags(10).map(b=>[h.field(b,'bag_lot').value,h.field(b,'bag_lb').value]),[['1','50'],['2','110']]);
  assert.equal(h.find(n=>n.dataset.standard==='160').textContent,'Recipe: 160 lb');
  assert.equal(h.bags(10)[0].children.find(n=>n.dataset.bagState==='').textContent,'✓ Confirmed');
  assert.equal(h.bags(10)[1].children.find(n=>n.dataset.bagState==='').textContent,'Not confirmed yet — type the code on the bag.');
  assert.equal(h.field(h.bags(10)[0],'bag_code').value,'OATS-OLD-0001');
  assert.deepEqual(h.field(h.bags(10)[0],'bag_lot').children.slice(1).map(o=>o.textContent),['OATS-OLD-0001 · 2026-10-01 · 50 lb','OATS-NEW-0002 · 2026-10-20 · 500 lb','OATS-SPR-1002 · 2026-10-30 · 300 lb']);
});

test('bag split sends exact pounds per bag and the typed codes',()=>{
  const h=render();const oats=h.ingredient(10);h.button(oats,'Add another bag').onclick();
  const [first,second,third]=h.bags(10);assert.equal(h.field(third,'bag_lot').value,'3');
  h.field(second,'bag_lb').value='70';h.field(third,'bag_lb').value='40';h.field(third,'bag_lb').oninput();
  h.field(second,'bag_code').value='0002';h.field(second,'bag_method').value='last4';h.field(third,'bag_code').value='OATS-SPR-1002';h.field(third,'bag_method').value='full_code';
  h.submit();
  const {lines,next}=plain(h.calls[0]);
  assert.deepEqual(lines[0],{ingredient_product_id:10,bags:[{lot_id:1,quantity_lb:50},{lot_id:2,quantity_lb:70},{lot_id:3,quantity_lb:40}],reason_code:null,note:null});
  assert.deepEqual(next,[{lot_id:1,method:'full_code',value:'OATS-OLD-0001'},{lot_id:2,method:'last4',value:'0002'},{lot_id:3,method:'full_code',value:'OATS-SPR-1002'}]);
  assert.ok(h.dirty>0);
});

test('changing pounds keeps the confirmation; changing the bag clears it',()=>{
  const h=render();const [first]=h.bags(10);
  h.field(first,'bag_lb').value='45';h.field(first,'bag_lb').oninput();
  assert.equal(h.field(first,'bag_code').value,'OATS-OLD-0001');
  h.field(first,'bag_lot').value='3';h.field(first,'bag_lot').onchange();
  assert.equal(h.field(first,'bag_code').value,'');
  assert.equal(first.children.find(n=>n.dataset.bagState==='').textContent,'Not confirmed yet — type the code on the bag.');
  h.submit();assert.ok(!h.calls[0].next.some(c=>c.lot_id===3));
});

test('removing a bag moves its pounds to the bag above',()=>{
  const h=render();const second=h.bags(10)[1];h.button(second,'Remove bag').onclick();
  assert.deepEqual(h.bags(10).map(b=>h.field(b,'bag_lb').value),['160']);
});

test('not used shows the fixed reason list and sends the chosen reason',()=>{
  const h=render();const honey=h.ingredient(20),reasonArea=honey.children.find(n=>n.className==='formula-reason');
  assert.equal(reasonArea.hidden,true);h.button(honey,'Not used (0 lb)').onclick();assert.equal(reasonArea.hidden,false);
  const select=h.field(honey,'ingredient_reason');
  assert.deepEqual(plain(select.children.slice(1).map(o=>[o.value,o.textContent])),[['out_of_stock','Out of stock'],['not_used','Not used'],['substituted','Substituted'],['quality_problem','Quality problem'],['owner_instruction','Owner instruction'],['other','Other']]);
  select.value='other';h.field(honey,'ingredient_note').value=' Jar cracked ';h.submit();
  assert.deepEqual(plain(h.calls[0].lines[1]),{ingredient_product_id:20,bags:[],reason_code:'other',note:'Jar cracked'});
});

test('more than 10% off recipe is shown; exactly 10% is not',()=>{
  const h=render();const [,second]=h.bags(10),total=h.ingredient(10).children.find(n=>n.dataset.total==='');
  h.field(second,'bag_lb').value='126';h.field(second,'bag_lb').oninput();
  assert.equal(total.dataset.over,'false');assert.equal(total.textContent,'Using 176 lb (+10%)');
  h.field(second,'bag_lb').value='127';h.field(second,'bag_lb').oninput();
  assert.equal(total.dataset.over,'true');assert.match(total.textContent,/^Using 177 lb \(\+10.6%\) — more than 10% off the recipe; the owner will review it\.$/);
});

test('an ambiguous last-4 code asks for the full code instead of sending',()=>{
  const c2=[{lot_id:4,lot_code:'AAA-1002',available_lb:1},{lot_id:5,lot_code:'BBB-1002',available_lb:1}];
  const ui=load();const card=new Element('section'),calls=[];ui.formulaEntry(card,{pans:1,pan_lb:10,output_lb:10,tolerance_pct:10,ingredients:[{product_id:1,product_name:'X',standard_lb:5,bags:[{lot_id:4,lot_code:'AAA-1002',quantity_lb:5}],candidates:c2}]},{update:(l,n)=>calls.push(n)});
  const nodes=descendants(card);nodes.find(n=>n.name==='bag_method').value='last4';nodes.find(n=>n.name==='bag_code').value='1002';nodes.find(n=>n.tag==='button'&&n.textContent==='Check bags and pounds → new draft').onclick();
  assert.equal(calls.length,0);assert.ok(nodes.find(n=>n.className==='bad'&&!n.hidden));
});

test('Spanish labels follow the language',()=>{
  const ui=load();ui.lang='es';const card=new Element('section');ui.formulaEntry(card,block());
  const text=descendants(card).map(n=>n.textContent);
  assert.ok(text.includes('Receta: 160 lb'));assert.ok(text.includes('Agregar otra bolsa'));assert.ok(text.includes('No se usó (0 lb)'));
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
