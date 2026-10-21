// PR #143: explicit Product/Charge choice, the "Confirm: Charge or Product?" card and visible charge lines (EN/ES).
const {test}=require('node:test');
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const ledger=require('../dashboard/ledger-ui.js');
const office=require('../dashboard/office-logic.js');
const flush=()=>new Promise(resolve=>setImmediate(resolve));
const plain=value=>JSON.parse(JSON.stringify(value));   // objects made inside vm compare by value

class Element{
 constructor(tag,text=''){this.tag=tag;this.textContent=String(text??'');this.children=[];this.parent=null;this.dataset={};this.value='';this.hidden=false;this.disabled=false;this.type='';this.required=false;this.checked=false;this.className='';}
 append(child){child.parent=this;this.children.push(child);}
 replaceChildren(...children){for(const child of this.children)child.parent=null;this.children=[];for(const child of children)this.append(child);}
 remove(){if(this.parent)this.parent.children=this.parent.children.filter(child=>child!==this);this.parent=null;}
 get isConnected(){return this.root||!!this.parent?.isConnected;}
 setAttribute(name,value){this[name]=value;}
 addEventListener(){}
 scrollIntoView(){}
 get text(){return [this.textContent,...this.children.map(child=>child.text)].join('');}
}
const descendants=root=>[root,...root.children.flatMap(descendants)];
function fakeUi(language,respond){
 const requests=[],failures=[],labelled=new Set(),notices=[];
 const ui={lang:language,session:{token:'session',actor:{id:1}},ready:Promise.resolve(true),bind(){},relabel(){},setText(node,value){node.textContent=String(value??'');return node;},
  t:(en,es)=>Object.assign(new String(ui.lang==='es'?es:en),{en,es}),problem:value=>Error(String(value)),
  el(tag,text,parent){const element=new Element(tag,text);if(text?.en!==undefined)labelled.add(element);parent?.append(element);return element;},
  notice(message,tone){notices.push({message:String(message),tone});},failure(error){failures.push(error);},
  async request(path,body){requests.push({path,body});return respond(path,body);},
  exceptionLabel:ledger.exceptionLabel,exceptionQueue:ledger.exceptionQueue,openingStock:ledger.openingStock,resolutionChoices:ledger.resolutionChoices,draftHtml:ledger.draftHtml,codeText:ledger.codeText};
 return {ui,requests,failures,notices};
}
function page(ids){const root=new Element('document');root.root=true;for(const id of ids){const node=new Element('div');node.id=id;root.append(node);}return root;}

// ── Office: catalog item form ──
async function officeHarness(language,respond=()=>({})){
 const root=page(['content','resolution','notice']);const byId=id=>descendants(root).find(n=>n.id===id),events={};
 const fake=fakeUi(language,(p,b)=>p==='/auth/whoami'?{permissions:{create_order:true,manage_catalog:true}}:respond(p,b));
 vm.runInNewContext(fs.readFileSync('dashboard/office.js','utf8'),{setTimeout,clearTimeout,FLLedger:fake.ui,FLOffice:office,FLDashboard:{writesEnabled:true},URLSearchParams,location:{search:''},sessionStorage:{getItem(){return null;},removeItem(){}},document:{getElementById:byId},window:{addEventListener(name,callback){events[name]=callback;}}});
 await flush();await flush();
 const find=(t,r=root)=>descendants(r).find(t),all=(t,r=root)=>descendants(r).filter(t);
 find(n=>n.dataset.officeStart==='catalog_item').onclick();
 return {...fake,root,find,all,byId};
}
const EN={choice:'This item is (required)',product:'Product',charge:'Charge (pallet, freight, delivery…)',create:'Create item',missing:'Choose Product or Charge (pallet, freight, delivery…).'};
const ES={choice:'Este artículo es (obligatorio)',product:'Producto',charge:'Cargo (palé, flete, entrega…)',create:'Crear artículo',missing:'Elige Producto o Cargo (palé, flete, entrega…).'};

test('office lists the catalog action only with manage_catalog',()=>{
 assert.deepEqual(office.permitted({manage_catalog:true},true).map(([a])=>a),['catalog_item']);
 assert.ok(!office.permitted({create_order:true},true).some(([a])=>a==='catalog_item'));
});

for(const [language,words] of [['en',EN],['es',ES]]){
 test(`catalog form starts with no Product/Charge choice and refuses to create without one (${language})`,async()=>{
  const h=await officeHarness(language);const form=h.byId('catalog-create');
  const box=h.find(n=>n.dataset.itemKindChoice!==undefined,form);
  assert.equal(h.find(n=>n.tag==='legend',box).textContent,words.choice);
  const radios=h.all(n=>n.tag==='input'&&n.type==='radio',box);
  assert.deepEqual(radios.map(r=>r.value),['product','charge']);
  assert.ok(radios.every(r=>!r.checked&&r.required),'nothing is pre-selected');
  assert.deepEqual(h.all(n=>n.tag==='strong',box).map(n=>n.textContent),[words.product,words.charge]);
  h.find(n=>n.tag==='input'&&n.type==='text',form).value='Coffee Crunch';
  await form.onsubmit({preventDefault(){}});
  assert.equal(h.requests.filter(r=>r.path==='/catalog/items').length,0,'nothing is sent');
  assert.equal(String(h.failures[0].message),words.missing);
 });

 test(`choosing Charge sends item_kind charge with no pounds; disagreement points to Exceptions (${language})`,async()=>{
  const h=await officeHarness(language,(p,b)=>({created:true,item:{id:9,name:b.name,item_kind:b.item_kind,check_exception_id:p==='/catalog/items'&&b.name.includes('wrap')?31:null}}));
  const form=h.byId('catalog-create'),[product,charge]=h.all(n=>n.type==='radio',form);
  const [name,sku]=h.all(n=>n.tag==='input'&&n.type==='text',form);name.value='Shrink wrap';sku.value='SW-1';
  charge.checked=true;charge.onchange();
  assert.equal(h.find(n=>n.tag==='select',form).parent.parent.hidden,true,'product type hidden for a charge');
  await form.onsubmit({preventDefault(){}});
  assert.deepEqual(plain(h.requests.at(-1)),{path:'/catalog/items',body:{name:'Shrink wrap',item_kind:'charge',sku:'SW-1'}});
  assert.match(form.text,language==='es'?/Creado Shrink wrap — Cargo/:/Created Shrink wrap — Charge/);
  assert.match(form.text,language==='es'?/Confirmar: ¿Cargo o Producto\?” \(#31\)/:/Confirm: Charge or Product\?” \(#31\)/);
  assert.equal(product.checked,false);assert.equal(charge.checked,false,'the choice resets after create');
 });

 test(`a product needs its type and is sent as item_kind product (${language})`,async()=>{
  const h=await officeHarness(language,(p,b)=>({created:true,item:{id:5,name:b.name,item_kind:b.item_kind,check_exception_id:null}}));
  const form=h.byId('catalog-create'),[product]=h.all(n=>n.type==='radio',form);
  h.find(n=>n.tag==='input'&&n.type==='text',form).value='Classic Granola';product.checked=true;product.onchange();
  await form.onsubmit({preventDefault(){}});
  assert.equal(h.requests.filter(r=>r.path==='/catalog/items').length,0);
  h.find(n=>n.tag==='select',form).value='finished';h.find(n=>n.type==='number',form).value='25';
  await form.onsubmit({preventDefault(){}});
  assert.deepEqual(plain(h.requests.at(-1).body),{name:'Classic Granola',item_kind:'product',type:'finished',case_size_lb:25});
 });

 test(`changing an existing item shows its setting and still requires a choice (${language})`,async()=>{
  const h=await officeHarness(language,(p,b)=>p.startsWith('/catalog/items?')?{items:[{id:7,name:'Pallet Charge',sku:'PAL',item_kind:'charge'}]}:{changed:true,item:{id:7,name:'Pallet Charge',item_kind:b.item_kind,check_exception_id:null}});
  const edit=h.byId('catalog-edit');h.find(n=>n.tag==='input',edit).value='pallet';
  await h.find(n=>n.tag==='button'&&n.textContent===(language==='es'?'Buscar':'Search'),edit).onclick();
  assert.equal(h.requests.at(-1).path,'/catalog/items?q=pallet');
  const found=h.find(n=>n.tag==='button'&&n.textContent==='Pallet Charge · PAL',edit);
  assert.equal(h.find(n=>n.className==='charge-tag',found).textContent,language==='es'?'Cargo':'Charge');
  found.onclick();
  assert.match(edit.text,language==='es'?/Ahora: Cargo \(palé, flete, entrega…\)/:/Now: Charge \(pallet, freight, delivery…\)/);
  const save=h.find(n=>n.tag==='button'&&n.textContent===(language==='es'?'Guardar elección':'Save choice'),edit);
  await save.onclick();assert.equal(String(h.failures.at(-1).message),(language==='es'?ES:EN).missing);
  const [product]=h.all(n=>n.type==='radio',edit);product.checked=true;await save.onclick();
  assert.deepEqual(plain(h.requests.at(-1)),{path:'/catalog/items/7/item-kind',body:{item_kind:'product'}});
  assert.match(edit.text,language==='es'?/Guardado: Pallet Charge — Producto/:/Saved: Pallet Charge — Product/);
 });
}

// ── Exceptions: "Confirm: Charge or Product?" card ──
async function exceptionsHarness(language,row){
 const root=page(['content','resolution','notice']);const byId=id=>descendants(root).find(n=>n.id===id);
 let rows=[row];const fake=fakeUi(language,(p,b)=>{if(p.startsWith('/exceptions?')){const out={exceptions:rows,count:rows.length};return out;}if(p===row.confirm_path){rows=[];return {confirmed:true,item:{id:row.product.id,name:row.product.name,item_kind:b.item_kind}};}return {};});
 ledger.lang=language;
 vm.runInNewContext(fs.readFileSync('dashboard/exceptions.js','utf8'),{FLLedger:fake.ui,FLReceipts:{link:n=>n},URLSearchParams,location:{search:''},document:{getElementById:byId},window:{addEventListener(){}},Date});
 await flush();await flush();
 return {...fake,root,byId,find:(t,r=root)=>descendants(r).find(t),all:(t,r=root)=>descendants(r).filter(t)};
}
const check=(actions)=>({id:31,kind:'CHARGE_SETTING_CHECK',status:'open',product:{id:7,name:'Batch Setton Coffee Granola #12'},
 detail:{marked_as:'product',message:'Confirm: Charge or Product? Now marked Product, but its name contains "fee". Nothing changes until someone confirms.',
  message_es:'Confirmar: ¿Cargo o Producto? Ahora está marcado como Producto, pero su nombre contiene "fee". Nada cambia hasta que alguien confirme.'},
 allowed_actions:actions,confirm_path:'/catalog/items/7/confirm-kind',owner:{name:'Owner'},due_at:null});

for(const language of ['en','es']){
 test(`exception card asks Charge or Product with two plain answers (${language})`,async()=>{
  const h=await exceptionsHarness(language,check(['confirmed_product','confirmed_charge']));
  const card=h.find(n=>n.dataset.exceptionId===31);
  assert.equal(h.find(n=>n.tag==='h3',card).textContent,language==='es'?'Confirmar: ¿Cargo o Producto?':'Confirm: Charge or Product?');
  assert.match(card.text,language==='es'?/Marcado ahora: Producto/:/Now marked: Product/);
  assert.match(card.text,language==='es'?/su nombre contiene "fee"/:/its name contains "fee"/);
  const buttons=h.all(n=>n.tag==='button'&&n.dataset.itemKind,card);
  assert.deepEqual(buttons.map(b=>[b.dataset.itemKind,b.textContent]),language==='es'?[['product','Es un Producto'],['charge','Es un Cargo (palé, flete, entrega…)']]:[['product','It is a Product'],['charge','It is a Charge (pallet, freight, delivery…)']]);
  assert.equal(h.find(n=>n.tag==='button'&&/^Resolv/.test(n.textContent),card),undefined,'no generic Resolve form');
  await buttons[0].onclick();
  assert.deepEqual(plain(h.requests.find(r=>r.path==='/catalog/items/7/confirm-kind')),{path:'/catalog/items/7/confirm-kind',body:{item_kind:'product'}});
  assert.match(h.notices.at(-1).message,language==='es'?/no volverá a preguntar/:/will not ask about this item again/);
  assert.equal(h.find(n=>n.dataset.exceptionId===31),undefined,'the answered check leaves the queue');
 });
 test(`floor sees the question but no answer buttons (${language})`,async()=>{
  const h=await exceptionsHarness(language,check([]));const card=h.find(n=>n.dataset.exceptionId===31);
  assert.equal(h.all(n=>n.tag==='button'&&n.dataset.itemKind,card).length,0);
  assert.match(card.text,language==='es'?/El dueño o la oficina confirma esta/:/The owner or office confirms this one/);
 });
 for(const [was,now] of [['product','charge'],['charge','product']]){
  test(`exception current setting follows ${was} to ${now} and preserves history (${language})`,async()=>{
   const row=check(['confirmed_product','confirmed_charge']);
   row.product.name='Current catalog name';
   row.detail={name:'Original catalog name',marked_as:was,current_name:row.product.name,current_marked_as:now,
    message:`Originally asked about Original catalog name: marked ${was}.`,
    message_es:`Se preguntó originalmente por Original catalog name: marcado ${was}.`};
   const h=await exceptionsHarness(language,row),card=h.find(n=>n.dataset.exceptionId===31);
   const current=h.find(n=>n.tag==='p'&&n.className==='neutral',card);
   assert.equal(current.textContent,(language==='es'?'Marcado ahora: ':'Now marked: ')+
    (now==='charge'?(language==='es'?'Cargo (palé, flete, entrega…)':'Charge (pallet, freight, delivery…)'):
     (language==='es'?'Producto':'Product')));
   assert.match(card.text,/Current catalog name/);
   assert.match(card.text,/Original catalog name/);
   assert.match(card.text,language==='es'?/Se preguntó originalmente/:/Originally asked about/);
   assert.equal(row.detail.marked_as,was);
  });
 }
}

// ── Draft cards (shared Office / Record entry) ──
for(const [language,tag] of [['en','Charge'],['es','Cargo']]){
 test(`draft card tags a charge line and drops its pounds and lot (${language})`,()=>{
  ledger.lang=language;
  const html=ledger.draftHtml({lines:[{product:'Pallet Charge',quantity:2,unit:'each',quantity_lb:0,lot_code:'L-1',is_service:true},
                                      {product:'Classic Granola',quantity_lb:50,lot_code:'L-2',is_service:false}]});
  const [charge,product]=html.split('<li>').slice(1);
  assert.match(charge,new RegExp(`<span class="charge-tag" data-en="Charge" data-es="Cargo">${tag}</span>`));
  assert.doesNotMatch(charge,/data-en="lb"|L-1|Service line/);
  assert.match(product,/data-en="lb"/);assert.match(product,/L-2/);assert.doesNotMatch(product,/charge-tag|Service line/);
  ledger.lang='en';
 });
 test(`exception kind title (${language})`,()=>{
  assert.equal(ledger.exceptionLabel('CHARGE_SETTING_CHECK',language),language==='es'?'Confirmar: ¿Cargo o Producto?':'Confirm: Charge or Product?');
 });
}
for(const flag of ['is_service','is_non_weight'])test(`isChargeLine follows only the service flag (${flag})`,()=>{
 assert.equal(ledger.isChargeLine({[flag]:true,product:'Classic Granola'}),true);
 assert.equal(ledger.isChargeLine({product:'Delivery Fee Coffee',is_service:false}),false,'never the name');
});

// ── Sales Orders board ──
const dashboard=fs.readFileSync(path.join(__dirname,'../dashboard/dashboard.js'),'utf8');
function functionSource(name){const start=dashboard.indexOf(`  function ${name}(`);assert.ok(start>=0);return dashboard.slice(start,dashboard.indexOf('\n  }',start)+4);}
test('order detail and order lines tag charge lines and keep pounds for products',()=>{
 const noop=()=>'';
 const context=vm.createContext({state:{orderDetailEditMode:false,orderReadyWrites:new Set(),ordersData:[]},SOList:{closeExplanation:noop,number:String,paragraph:String,explanation:()=>({attrs:'',content:''}),trigger:(id,label)=>label,healthContent:noop,bindExplanations:noop},
  canEditOrderHeader:()=>false,canEditOrderLines:()=>false,orderDetailFlag:()=>({ready:false,metadataKnown:true}),orderDetailShipDate:noop,escHtml:String,escAttr:String,
  PalletCalculations:{calculateLinePallets:()=>({calculatedPallets:null})},renderOrderEditActions:noop,renderAllocationSection:noop,renderShippingPreviewSection:noop,
  bindOrderInventoryToggles:noop,bindOrderDetailEditControls:noop,bindOrderDetailActions:noop,bindAllocationControls:noop});
 vm.runInContext(functionSource('orderLineQuantities')+functionSource('renderOrderDetail')+functionSource('renderOrderLinesContent'),context);
 const lines=[{line_id:11,product:'Pallet Charge',unit:'each',quantity:2,unit_quantity:2,quantity_lb:0,is_service:true,is_non_weight:true,line_status:'pending'},
              {line_id:12,product:'Setton Coffee Granola',quantity_lb:50,line_status:'pending'}];
 const container={innerHTML:''};context.renderOrderDetail({order_id:1,state:'open',lines},container,false);
 const row=id=>container.innerHTML.match(new RegExp(`<tr[^>]*data-line-id="${id}">([\\s\\S]*?)</tr>`))[1];
 assert.match(row(11),/class="charge-tag"[^>]*>Charge<\/span>/);assert.match(row(11),/2 units/);assert.doesNotMatch(row(11),/order-inventory-toggle/);
 assert.doesNotMatch(row(12),/charge-tag/);assert.match(row(12),/order-inventory-toggle/);
 const summary=context.renderOrderLinesContent({order_id:1,state:'closed',lines});
 assert.match(summary,/Pallet Charge <span class="charge-tag"/);assert.doesNotMatch(summary.split('Setton Coffee Granola')[1].split('</tr>')[0],/charge-tag/);
});
