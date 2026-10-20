const {test}=require('node:test');
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('dashboard/entry.js','utf8');
const flush=async()=>{for(let i=0;i<4;i++)await new Promise(resolve=>setImmediate(resolve));};
class Element{
  constructor(tag,text=''){this.tag=tag;this.textContent=String(text??'');this.children=[];this.parent=null;this.dataset={};this.value='';this.hidden=false;this.disabled=false;this.type='';this.required=false;this.checked=false;this.events={};}
  append(child){child.parent=this;this.children.push(child);}
  replaceChildren(...children){this.children.forEach(child=>child.parent=null);this.children=[];children.forEach(child=>this.append(child));}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(child=>child!==this);this.parent=null;}
  get parentElement(){return this.parent;}
  get isConnected(){return this.root||!!this.parent?.isConnected;}
  setAttribute(name,value){this[name]=value;}
  addEventListener(name,callback){this.events[name]=callback;}
  querySelectorAll(){return descendants(this).filter(node=>['input','select','textarea','button'].includes(node.tag));}
  scrollIntoView(){}
}
const descendants=root=>[root,...root.children.flatMap(descendants)];
const plain=value=>JSON.parse(JSON.stringify(value));
async function harness(action='ship',language='en',{role='floor'}={}){
  const root=new Element('document');root.root=true;
  for(const id of ['content','resolution','notice','language','sign-out']){const node=new Element('div');node.id=id;root.append(node);}
  const find=(predicate,scope=root)=>descendants(scope).find(predicate),all=(predicate,scope=root)=>descendants(scope).filter(predicate),byId=id=>find(node=>node.id===id);
  const tickets=[],requests=[],failures=[],uploads=[],timers=new Map();let nextTimer=0,resolver=null;
  const lots=[{lot_id:11,lot_code:'OPEN261014-P1561',available_lb:50},{lot_id:12,lot_code:'PACK261019-001',available_lb:80}];
  let stock=async()=>({product_id:1561,product:'Classic Granola',lots});
  const order={order_id:5,order_number:'SO-PILOT',lines:[{line_id:101,product_id:1561,product:'Classic Granola',remaining_lb:60,line_status:'open'},{line_id:102,product_id:9,product:'Pallet',remaining_lb:0,is_service:true,line_status:'open'},{line_id:103,product_id:1561,product:'Cancelled',remaining_lb:10,line_status:'cancelled'},{line_id:104,product_id:1561,product:'Fulfilled',remaining_lb:0,line_status:'fulfilled'}]};
  const labelled=new Set(),events={};const label=(node,value)=>{node.textContent=String(value??'');if(value?.en!==undefined){node.labels=[value.en,value.es];labelled.add(node);}else{delete node.labels;labelled.delete(node);}return node;};
  const ui={problem:value=>Error(String(value)),lang:language,correctionReasons:require('../dashboard/ledger-ui.js').correctionReasons,session:{actor:{role}},async upload(file,kind){const attached={attachment_id:88,kind};uploads.push({file,kind,attached});return attached;},formatNumber:require('../dashboard/ledger-ui.js').formatNumber,bind(){},ready:Promise.resolve(true),setText:label,relabel(){for(const node of labelled)if(node.isConnected)node.textContent=node.labels[ui.lang==='es'?1:0];},t:(en,es)=>Object.assign(new String(ui.lang==='es'?es:en),{en,es}),el(tag,text,parent){const node=label(new Element(tag),text);parent?.append(node);return node;},failure(error){failures.push(error.message);},
    async request(path,body,options){requests.push({path,body,options});if(path==='/entry/lots?'+new URLSearchParams({product_id:'1561'}))return stock(path);if(path.startsWith('/entry/lots?'))return stock(path);if(path==='/resolve'){if(resolver)return resolver(body);return {outcome:'match',match:{id:body.kind==='order'?5:body.kind==='customer'?7:body.kind==='supplier'?8:1561,label:body.kind,case_size_lb:25}};}if(path.startsWith('/sales/orders/'))return order;if(path.startsWith('/ship/lots?'))return stock(path);throw Error('Unexpected request '+path);},
    ticketCard(host,path,body,options){const ticket={host,path,body:plain(body),options};tickets.push(ticket);return {invalidate(){ticket.invalidated=true;}};}
  };
  vm.runInNewContext(source,{setTimeout(callback){const id=++nextTimer;timers.set(id,callback);return id;},clearTimeout(id){timers.delete(id);},FLLedger:ui,FLDashboard:{writesEnabled:true},URL,URLSearchParams,location:{search:'?action='+action,href:'https://example.test/entry.html?action='+action},history:{replaceState(){}},document:{getElementById:byId},window:{addEventListener(name,callback){events[name]=callback;}}});await flush();
  const shipMode=find(node=>node.name==='ship_mode');if(shipMode)shipMode.value='order';
  const field=(name,scope=root)=>find(node=>node.name===name,scope),button=(text,scope=root)=>find(node=>node.tag==='button'&&node.textContent===text,scope);
  async function pick(name){const scope=find(node=>node.dataset.picker===name);find(node=>node.tag==='input',scope).value=name;await find(node=>node.tag==='button',scope).onclick();find(node=>node.className==='picker-options',scope).children[0].onclick();await flush();}
  function submit(){byId('entry-form').onsubmit({preventDefault(){}});}
  async function standalone(){field('ship_mode').value='standalone';field('ship_mode').onchange();await pick('customer');await pick('ship_product');field('quantity_lb').value='40';field('order_reference').value='PO-100';field('bol_reference').value='BOL-1';}
  function manual(scope){const mode=field('shipment_lot_mode',scope);mode.value='manual';mode.onchange();return scope;}
  function selectLot(id,quantity,scope){const select=field('shipment_lot_id',scope);select.value=String(id);select.onchange();field('shipment_lot_quantity',scope).value=String(quantity);}
  return {root,find,all,byId,field,button,pick,switchLanguage(lang){ui.lang=lang;events['ledger-language']();},submit,standalone,manual,selectLot,tickets,requests,failures,uploads,timers,async runTimers(){const pending=[...timers.values()];timers.clear();pending.forEach(callback=>callback());await flush();},setResolver(fn){resolver=fn;},setUpload(fn){ui.upload=fn;},order,lots,setStock(fn){stock=fn;}};
}
for(const action of ['receive','make','pack'])test(`${action} optional note is sent verbatim with its draft`,async()=>{
  const h=await harness(action);if(action==='pack'){await h.pick('source');await h.pick('target');}else await h.pick('product');if(action==='receive')await h.pick('supplier');
  const note=h.field('note');assert.equal(note.tag,'textarea');assert.equal(note.required,false);assert.equal(note.maxLength,2000);note.value='Keep for Wednesday <sample>.';h.submit();
  assert.equal(h.tickets[0].body.note,note.value);assert.deepEqual(h.failures,[]);
});
test('blank optional note is omitted from receive request',async()=>{const h=await harness('receive');await h.pick('product');await h.pick('supplier');h.submit();assert.equal('note' in h.tickets[0].body,false);});
for(const language of ['en','es'])test(`standalone defaults to FIFO, exposes available lots and sends the actual chosen lot in ${language}`,async()=>{
  const h=await harness('ship',language);await h.standalone();const scope=h.find(node=>node.dataset.shipLots==='1561');
  assert.equal(h.field('shipment_lot_mode').value,'fifo');h.submit();assert.equal('lot_allocations' in h.tickets[0].body,false);h.tickets[0].options.onEdit();
  h.manual(scope);assert.equal(h.field('shipment_lot_id',scope).value,'11');assert.equal(h.field('shipment_lot_quantity',scope).value,'');
  const option=h.find(node=>node.tag==='option'&&node.value==='12',scope);assert.match(option.textContent,/PACK261019-001.*80 lb/);
  h.selectLot(12,40,scope);assert.equal(h.field('shipment_lot_quantity',scope).max,'80');assert.match(h.find(node=>node.dataset.selectedShipLot!==undefined,scope).children[0].textContent,/PACK261019-001.*80 lb/);h.submit();
  assert.deepEqual(h.tickets.at(-1).body.lot_allocations,[{lot_id:12,quantity_lb:40}]);assert.deepEqual(h.failures,[]);
});
test('manual shipping can split quantities across lots and restore default FIFO',async()=>{
  const h=await harness();await h.standalone();h.field('quantity_lb').value='70';const scope=h.manual(h.find(node=>node.dataset.shipLots));
  assert.equal(h.field('shipment_lot_quantity',scope).value,'');h.selectLot(11,50,scope);h.button('Add another shipped lot',scope).onclick();h.all(node=>node.name==='shipment_lot_quantity',scope)[1].value='20';h.submit();
  assert.deepEqual(h.tickets[0].body.lot_allocations,[{lot_id:11,quantity_lb:50},{lot_id:12,quantity_lb:20}]);h.tickets[0].options.onEdit();
  h.field('shipment_lot_mode').value='fifo';h.field('shipment_lot_mode').onchange();h.submit();assert.equal('lot_allocations' in h.tickets.at(-1).body,false);
});
for(const [name,mutate,message] of [
  ['total mismatch',h=>h.field('shipment_lot_quantity').value='20',/add up/],
  ['over available quantity',h=>{h.field('quantity_lb').value='51';h.field('shipment_lot_quantity').value='51';},/exceeds/],
  ['zero quantity',h=>h.field('shipment_lot_quantity').value='0',/positive/],
  ['duplicate lot',h=>{h.button('Add another shipped lot').onclick();const fields=h.all(node=>node.name==='shipment_lot_id');fields[1].value='11';fields[1].onchange();const qty=h.all(node=>node.name==='shipment_lot_quantity');qty[0].value='20';qty[1].value='20';},/once/],
  ['empty lot selection',h=>h.button('Remove shipped lot').onclick(),/positive/],
])test(`manual shipping refuses ${name} before preparing a draft`,async()=>{const h=await harness();await h.standalone();h.manual(h.find(node=>node.dataset.shipLots));mutate(h);h.submit();assert.equal(h.tickets.length,0);assert.match(h.failures.at(-1),message);});
test('ship-all preserves its default API path; manual choice expands remaining physical and service lines',async()=>{
  const h=await harness();await h.pick('order');h.field('ship_all').checked=true;h.submit();assert.equal(h.tickets[0].body.ship_all,true);assert.equal('lines' in h.tickets[0].body,false);h.tickets[0].options.onEdit();
  assert.equal(h.all(node=>node.dataset.shipLots).length,1,'cancelled, fulfilled and service lines do not get lot selectors');
  const stock=h.requests.find(request=>request.path.startsWith('/ship/lots?'));assert.equal(new URL('https://local'+stock.path).searchParams.get('line_id'),'101');
  const scope=h.manual(h.find(node=>node.dataset.shipLots));h.selectLot(12,60,scope);h.submit();
  assert.equal(h.tickets.at(-1).path,'/sales/orders/5/ship/prepare');assert.deepEqual(h.tickets.at(-1).body.lines,[{line_id:101,quantity_lb:60,lot_allocations:[{lot_id:12,quantity_lb:60}]},{line_id:102,quantity_lb:0}]);assert.equal(h.tickets.at(-1).body.ship_all,false);
});
test('partial order shipment supplies allocations only for the selected line',async()=>{
  const h=await harness();await h.pick('order');const scope=h.find(node=>node.dataset.shipLots),qty=h.find(node=>node.tag==='input'&&node.dataset.lineId===101);qty.value='30';h.manual(scope);h.selectLot(12,30,scope);h.submit();
  assert.deepEqual(h.tickets[0].body.lines,[{line_id:101,quantity_lb:30,lot_allocations:[{lot_id:12,quantity_lb:30}]}]);assert.deepEqual(h.failures,[]);
});
test('failed stock read blocks manual selection, allows FIFO and can be retried',async()=>{
  const h=await harness();h.setStock(async()=>{throw Error('offline');});await h.standalone();h.manual(h.find(node=>node.dataset.shipLots));h.submit();assert.equal(h.tickets.length,0);assert.match(h.failures.at(-1),/Load the available/);
  h.field('shipment_lot_mode').value='fifo';h.field('shipment_lot_mode').onchange();h.submit();assert.equal('lot_allocations' in h.tickets[0].body,false);h.tickets[0].options.onEdit();
  h.setStock(async()=>({lots:h.lots}));await h.button('Retry available lots').onclick();h.manual(h.find(node=>node.dataset.shipLots));h.selectLot(12,40,h.find(node=>node.dataset.shipLots));h.submit();assert.equal(h.tickets.at(-1).body.lot_allocations[0].lot_id,12);
});
test('changing the selected product discards previous manual lot choices',async()=>{
  const h=await harness();await h.standalone();h.manual(h.find(node=>node.dataset.shipLots));h.selectLot(12,40,h.find(node=>node.dataset.shipLots));
  const picker=h.find(node=>node.dataset.picker==='ship_product'),input=h.find(node=>node.tag==='input',picker);input.value='Different';input.oninput();assert.equal(h.find(node=>node.dataset.shipLots),undefined);await h.pick('ship_product');assert.equal(h.field('shipment_lot_mode').value,'fifo');h.submit();assert.equal('lot_allocations' in h.tickets[0].body,false);
});
test('late stock response for a removed product never restores its lot selector',async()=>{
  const h=await harness();let resolve;h.setStock(()=>new Promise(done=>resolve=done));await h.standalone();const picker=h.find(node=>node.dataset.picker==='ship_product'),input=h.find(node=>node.tag==='input',picker);input.value='Different';input.oninput();resolve({lots:h.lots});await flush();assert.equal(h.find(node=>node.dataset.shipLots),undefined);assert.deepEqual(h.failures,[]);
});

test('FIFO and other shipment modes keep hidden manual controls disabled, including after busy state',async()=>{
  const h=await harness();await h.standalone();const scope=h.manual(h.find(node=>node.dataset.shipLots));h.field('shipment_lot_quantity',scope).value='999';
  h.field('shipment_lot_mode',scope).value='fifo';h.field('shipment_lot_mode',scope).onchange();assert.equal(h.field('shipment_lot_quantity',scope).disabled,true);h.submit();h.tickets[0].options.onBusy(false);assert.equal(h.field('shipment_lot_quantity',scope).disabled,true);h.tickets[0].options.onEdit();
  h.manual(scope);h.field('ship_mode').value='order';h.field('ship_mode').onchange();assert.equal(h.field('shipment_lot_quantity',scope).disabled,true);assert.equal(h.field('shipment_lot_mode',scope).disabled,true);
  h.field('ship_mode').value='standalone';h.field('ship_mode').onchange();assert.equal(h.field('shipment_lot_quantity',scope).disabled,false);
});

test('manual lots require a deliberate quantity rather than guessing reserved-stock preference',async()=>{
  const h=await harness();await h.pick('order');h.field('ship_all').checked=true;const scope=h.manual(h.find(node=>node.dataset.shipLots));assert.equal(h.field('shipment_lot_quantity',scope).value,'');h.submit();assert.match(h.failures.at(-1),/positive/);
  assert.equal(h.all(node=>node.dataset.lineId!==undefined).every(node=>node.tag==='input'),true);
});

test('standalone product and total quantity precede actual lot choices',async()=>{
  const h=await harness();await h.standalone();const nodes=descendants(h.root),product=h.find(node=>node.dataset.picker==='ship_product'),quantity=h.field('quantity_lb'),lots=h.find(node=>node.dataset.shipLots);assert.ok(nodes.indexOf(product)<nodes.indexOf(quantity));assert.ok(nodes.indexOf(quantity)<nodes.indexOf(lots));
});

test('sales order and ship-all choice precede its line and lot controls',async()=>{
  const h=await harness();await h.pick('order');const nodes=descendants(h.root),order=h.find(node=>node.dataset.picker==='order'),all=h.field('ship_all'),lots=h.find(node=>node.dataset.shipLots);assert.ok(nodes.indexOf(order)<nodes.indexOf(all));assert.ok(nodes.indexOf(all)<nodes.indexOf(lots));
});

for(const language of ['en','es'])test(`live search waits for two characters, debounces, and keeps Enter working in ${language}`,async()=>{
  const h=await harness('receive',language),scope=h.find(node=>node.dataset.picker==='product'),input=h.find(node=>node.tag==='input',scope);
  input.value='O';input.oninput();await h.runTimers();assert.equal(h.requests.length,0);
  input.value='Oa';input.oninput();input.value='Oats';input.oninput();assert.equal(h.timers.size,1);await h.runTimers();assert.equal(h.requests.length,1);assert.equal(h.requests[0].body.query,'Oats');assert.equal(h.requests[0].options.background,true);
  input.value='Su';input.oninput();input.onkeydown({key:'Enter',preventDefault(){}});await flush();assert.equal(h.timers.size,0);assert.equal(h.requests.at(-1).body.query,'Su');
});
test('typing clears stale selection and leaves Search available after an abandoned response',async()=>{
  const h=await harness('receive'),scope=h.find(node=>node.dataset.picker==='product'),input=h.find(node=>node.tag==='input',scope),search=h.find(node=>node.tag==='button',scope);let finish;h.setResolver(()=>new Promise(resolve=>finish=resolve));
  input.value='Oats';const pending=search.onclick();assert.equal(search.disabled,true);input.value='O';input.oninput();finish({outcome:'match',match:{id:1,label:'old'}});await pending;assert.equal(search.disabled,false);assert.equal(h.find(node=>node.className==='picker-options',scope).children.length,0);
});
for(const language of ['en','es'])test(`Regular products are first and extra-kosher options carry a distinct tag in ${language}`,async()=>{
  const h=await harness('receive',language);h.setResolver(()=>({outcome:'ambiguous',candidates:[{id:2,label:'Granola SS Classic 25 EXTRA-KOSHER',kosher_tier:'extra_kosher'},{id:1,label:'Granola Classic 25 Regular',kosher_tier:'regular'}]}));
  const scope=h.find(node=>node.dataset.picker==='product');h.find(node=>node.tag==='input',scope).value='Granola Classic 25';await h.find(node=>node.tag==='button',scope).onclick();const options=h.find(node=>node.className==='picker-options',scope).children;assert.match(options[0].textContent,/Regular/);assert.equal(options[1].children[0].className,'tier-tag');assert.equal(options[1].children[0].textContent,'EXTRA-KOSHER');
});
for(const action of ['receive','ship'])test(`${action} form photo passes the existing evidence kind and attachment into the draft`,async()=>{
  const h=await harness(action);if(action==='ship')await h.standalone();else{await h.pick('product');await h.pick('supplier');}
  const photo=h.field('entry_photo');assert.equal(photo.accept,'image/jpeg,image/png,image/webp');photo.files=[{name:'tag.png'}];await photo.onchange();h.submit();assert.equal(h.uploads[0].kind,action==='ship'?'bol':'lot_tag');assert.deepEqual(plain(h.tickets[0].options.initialAttachment),{attachment_id:88,kind:action==='ship'?'bol':'lot_tag'});assert.equal(h.tickets[0].options.photoKind,action==='ship'?'bol':'lot_tag');assert.equal('entry_photo' in h.tickets[0].body,false);
});
test('Cancel discards the draft and returns an empty entry form',async()=>{
  const h=await harness('receive');await h.pick('product');await h.pick('supplier');h.field('bol_reference').value='OLD-BOL';h.submit();h.tickets[0].options.onCancel();assert.equal(h.tickets[0].invalidated,true);assert.equal(h.field('bol_reference').value,'');assert.equal(h.byId('entry-form').hidden,false);
});
for(const role of ['floor','office','owner'])test(`Owner backfill visibility respects ${role} role`,async()=>{
  const h=await harness('receive','en',{role}),backfill=h.field('backfill');assert.equal(backfill.parent.hidden,role!=='owner');assert.equal(backfill.disabled,role!=='owner');await h.pick('product');await h.pick('supplier');h.submit();assert.equal('backfill' in h.tickets[0].body,role==='owner');h.tickets[0].options.onBusy(false);assert.equal(backfill.disabled,role!=='owner');
});
test('Adjust product-name search shows exact full lots; Found known lot fills only its code',async()=>{
  for(const action of ['adjust','found']){const h=await harness(action);h.setStock(async()=>({lots:[{id:11,lot_id:11,lot_code:'OATS-001',product_id:1561,product_name:'Oats',available_lb:12}]}));if(action==='found')await h.pick('product');const scope=h.find(node=>node.dataset.picker===(action==='found'?'known_lot':'lot'));h.find(node=>node.tag==='input',scope).value='oats';await h.find(node=>node.tag==='button',scope).onclick();const choice=h.find(node=>node.className==='picker-options',scope).children[0];assert.match(choice.textContent,/OATS-001.*Oats.*12 lb/);choice.onclick();assert.equal(new URL('https://local'+h.requests.at(-1).path).searchParams.get('include_empty'),'true');if(action==='found')assert.equal(h.field('lot_code').value,'OATS-001');else{h.submit();assert.equal(h.tickets[0].body.lot_id,11);}}
});
test('Pack batch search and recent lot selection pin the chosen lot but add no tag confirmation',async()=>{
  const h=await harness('pack');h.setStock(async()=>({lots:[{id:21,lot_id:21,lot_code:'BATCH-TODAY',available_lb:323},{id:20,lot_id:20,lot_code:'BATCH-OLD',available_lb:300}]}));await h.pick('source');assert.equal(h.requests.find(r=>r.path==='/resolve').body.context.product_type,'batch');const recent=h.all(node=>node.dataset.recentLot);assert.equal(recent[0].dataset.recentLot,'21');recent[0].onclick();await h.pick('target');h.field('cases').value='4';h.submit();assert.deepEqual(h.tickets[0].body.lot_allocations,[{lot_id:21,quantity_lb:100}]);assert.equal('lot_confirmations' in h.tickets[0].body,false);h.tickets[0].options.onEdit();h.button('Use suggested FIFO (oldest first)').onclick();h.submit();assert.equal('lot_allocations' in h.tickets.at(-1).body,false);
});

test('a form photo upload finishing after changing entry type never leaks into the new form',async()=>{
  const h=await harness('receive');let finish;h.setUpload(()=>new Promise(resolve=>finish=resolve));const photo=h.field('entry_photo');photo.files=[{name:'old-tag.png'}];const pending=photo.onchange();h.byId('entry-action').value='ship';h.byId('entry-action').onchange();h.byId('entry-action').value='receive';h.byId('entry-action').onchange();finish({attachment_id:99});await pending;await h.pick('product');await h.pick('supplier');h.submit();assert.equal(h.tickets[0].options.initialAttachment,null);
});
test('Pack recent choice uses an explicit case-weight override and clears when the source changes',async()=>{
  const h=await harness('pack');h.setStock(async()=>({lots:[{lot_id:21,lot_code:'BATCH-TODAY',available_lb:323}]}));await h.pick('source');h.find(node=>node.dataset.recentLot==='21').onclick();await h.pick('target');h.field('cases').value='4';h.field('case_weight_lb').value='20';h.submit();assert.equal(h.tickets[0].body.case_weight_lb,20);assert.deepEqual(h.tickets[0].body.lot_allocations,[{lot_id:21,quantity_lb:80}]);h.tickets[0].options.onEdit();await h.pick('source');await h.pick('target');h.submit();assert.equal('lot_allocations' in h.tickets.at(-1).body,false);
});

for(const action of ['receive','ship'])test(`${action} draft replacement photo survives Edit/change/Prepare and Cancel clears it`,async()=>{
  const h=await harness(action);if(action==='ship')await h.standalone();else{await h.pick('product');await h.pick('supplier');}
  const photo=h.field('entry_photo');photo.files=[{name:'form-A.png'}];await photo.onchange();h.submit();const first=h.tickets[0],kind=action==='ship'?'bol':'lot_tag';assert.deepEqual(plain(first.options.initialAttachment),{attachment_id:88,kind});first.options.onAttachment({attachment_id:89,kind});assert.ok(h.find(node=>node.textContent==='Photo attached'));first.options.onEdit();if(action==='ship')h.field('bol_reference').value='EDITED-BOL';else h.field('note').value='Edited note';h.byId('entry-form').events.input();first.options.onAttachment({attachment_id:99});h.submit();assert.deepEqual(plain(h.tickets[1].options.initialAttachment),{attachment_id:89,kind});assert.equal(h.uploads.length,1);h.tickets[1].options.onCancel();first.options.onAttachment({attachment_id:100});if(action==='ship')await h.standalone();else{await h.pick('product');await h.pick('supplier');}h.submit();assert.equal(h.tickets[2].options.initialAttachment,null);assert.equal(h.uploads.length,1);
});

for(const language of ['en','es'])test(`Receive allows BOL or lot-tag form photos and preserves the uploaded kind in ${language}`,async()=>{
  const h=await harness('receive',language);await h.pick('product');await h.pick('supplier');const type=h.field('entry_photo_kind');assert.equal(type.value,'lot_tag');type.value='bol';type.onchange();const photo=h.field('entry_photo');photo.files=[{name:'delivery.png'}];await photo.onchange();h.submit();assert.equal(h.uploads[0].kind,'bol');assert.deepEqual(plain(h.tickets[0].options.initialAttachment),{attachment_id:88,kind:'bol'});assert.equal(h.tickets[0].options.photoKind,'bol');assert.equal('entry_photo_kind' in h.tickets[0].body,false);h.tickets[0].options.onEdit();h.tickets[0].options.onAttachment({attachment_id:100,kind:'lot_tag'});assert.equal(type.value,'bol','stale callback cannot change the edited form');
});
test('Receive reuploads the same attached image when its photo type changes and cannot prepare during upload',async()=>{
  const h=await harness('receive');await h.pick('product');await h.pick('supplier');let finish;const uploads=[];h.setUpload((file,kind)=>{uploads.push({file,kind});return new Promise(resolve=>finish=resolve);});const type=h.field('entry_photo_kind'),photo=h.field('entry_photo'),file={name:'label.png'};photo.files=[file];const first=photo.onchange();assert.equal(type.disabled,true);finish({attachment_id:89,kind:'lot_tag'});await first;
  assert.equal(type.disabled,false);type.value='bol';const changed=type.onchange();assert.equal(type.disabled,true);h.submit();assert.equal(h.tickets.length,0);assert.equal(uploads[1].file,file);assert.equal(uploads[1].kind,'bol');finish({attachment_id:90,kind:'bol'});await changed;h.submit();assert.deepEqual(plain(h.tickets[0].options.initialAttachment),{attachment_id:90,kind:'bol'});assert.equal(h.tickets[0].options.initialPhoto,file);
});
test('failed form photo type change preserves the original attachment and restores its selected type',async()=>{
  const h=await harness('receive');await h.pick('product');await h.pick('supplier');const photo=h.field('entry_photo'),type=h.field('entry_photo_kind');photo.files=[{name:'label.png'}];await photo.onchange();h.setUpload(async()=>{throw Error('offline');});type.value='bol';await type.onchange();assert.equal(type.value,'lot_tag');h.submit();assert.deepEqual(plain(h.tickets[0].options.initialAttachment),{attachment_id:88,kind:'lot_tag'});assert.equal(h.tickets[0].options.photoKind,'lot_tag');
});
test('changing languages refreshes selected records and attached photo status while keeping form values',async()=>{
  const h=await harness('receive');await h.pick('product');await h.pick('supplier');h.field('note').value='Keep this note';const photo=h.field('entry_photo');photo.files=[{name:'label.png'}];await photo.onchange();h.switchLanguage('es');assert.ok(h.find(node=>node.textContent==='Seleccionado: product'));assert.ok(h.find(node=>node.textContent==='Foto adjunta — incluida con el borrador y el recibo.'));assert.equal(h.field('note').value,'Keep this note');h.switchLanguage('en');assert.ok(h.find(node=>node.textContent==='Selected: product'));assert.ok(h.find(node=>node.textContent==='Photo attached — included with the draft and receipt.'));assert.equal(h.uploads.length,1);
});
test('changing languages refreshes selected batch status without changing the selected source lot',async()=>{
  const h=await harness('pack');await h.pick('source');h.find(node=>node.dataset.recentLot==='12').onclick();h.switchLanguage('es');assert.ok(h.find(node=>node.textContent==='Lote seleccionado: PACK261019-001'));h.switchLanguage('en');assert.ok(h.find(node=>node.textContent==='Selected batch: PACK261019-001'));await h.pick('target');h.field('cases').value='4';h.submit();assert.deepEqual(h.tickets[0].body.lot_allocations,[{lot_id:12,quantity_lb:100}]);
});

const reasons=[['physical_count','Physical count','Conteo físico'],['missing_receipt','Missing receipt','Recepción no registrada'],['missing_production','Missing production','Producción no registrada'],['wrong_lot','Wrong lot used','Lote equivocado'],['damage_disposal','Damage/disposal','Daño o desecho'],['unrecorded_usage','Unrecorded usage','Uso no registrado'],['data_entry_error','Data-entry error','Error de captura'],['unknown','Unknown','Desconocido']];
for(const language of ['en','es'])for(const [code,en,es] of reasons)test(`correction reason ${code} agrees on the form and draft in ${language}`,async()=>{
  const h=await harness('adjust',language),label=language==='es'?es:en;
  const option=h.find(node=>node.tag==='option'&&node.value===code,h.field('reason_code'));
  assert.equal(option.textContent,label);await h.pick('lot');h.field('reason_code').value=code;h.field('delta_lb').value='-5';h.submit();
  assert.equal(h.tickets[0].body.reason_code,code);
  const ledger=require('../dashboard/ledger-ui.js');ledger.lang=language;
  assert.equal(ledger.draftHtml(code,'reason_code').replace(/<[^>]+>/g,''),label);
  assert.equal(ledger.draftHtml({code,label_en:en,label_es:es},'reason').replace(/<[^>]+>/g,''),label);
  const display=require('../dashboard/assistant-display.js');assert.equal(display.plainLabel(code,language),label);assert.equal(display.plainMessage(code,language),label);
});
for(const language of ['en','es'])test(`Found offers exactly its four supported translated reasons in ${language}`,async()=>{
  const h=await harness('found',language),allowed=['physical_count','missing_receipt','missing_production','unknown'];
  assert.deepEqual(h.field('reason_code').children.slice(1).map(option=>[option.value,option.textContent]),reasons.filter(([code])=>allowed.includes(code)).map(([code,en,es])=>[code,language==='es'?es:en]));
});
for(const language of ['en','es'])for(const action of ['make','pack','receive','adjust','found'])test(`${action} form and shared draft photo permission in ${language}`,async()=>{
  const h=await harness(action,language);
  if(action==='pack'){await h.pick('source');await h.pick('target');}else if(action==='adjust')await h.pick('lot');else await h.pick('product');
  if(action==='receive')await h.pick('supplier');
  h.submit();assert.deepEqual(h.failures,[]);assert.equal(h.tickets[0].options.allowPhoto,!['make','pack'].includes(action));
  if(['make','pack'].includes(action))assert.equal(h.find(node=>node.type==='file'),undefined);
});

for(const action of ['receive','pack'])test(`Spanish ${action} case wording matches shared drafts through language switches`,async()=>{
  const h=await harness(action,'es'),ledger=require('../dashboard/ledger-ui.js');
  const weight=action==='receive'?'case_size_lb':'case_weight_lb';
  const labels=action==='receive'
    ?{en:['Cases / containers received','lb per case / container'],es:['Cajones / envases recibidos','lb por cajón / envase']}
    :{en:['Cases to pack','lb per case (optional override)'],es:['Cajones para empacar','lb por cajón (opcional)']};
  h.field('cases').value='4';h.field(weight).value='25';
  for(const language of ['es','en','es']){
    h.switchLanguage(language);ledger.lang=language;
    assert.equal(h.field('cases').parentElement.textContent,labels[language][0]);
    assert.equal(h.field(weight).parentElement.textContent,labels[language][1]);
    assert.equal(h.field('cases').value,'4');assert.equal(h.field(weight).value,'25');
    const headings=[...ledger.draftHtml({cases:4,[weight]:25}).matchAll(/<dt[^>]*>([^<]+)<\/dt>/g)].map(match=>match[1]);
    assert.deepEqual(headings,language==='es'?['Cajones','lb por cajón']:['Cases','lb per case']);
    assert.equal(ledger.codeText('boxes','es'),'Cajas','Office boxes retain their distinct Spanish unit');
  }
});
test('Spanish Pack missing case weight warning agrees with its form and prepares no draft',async()=>{
  const h=await harness('pack','es');
  h.setResolver(body=>({outcome:'match',match:{id:1561,label:body.kind}}));
  await h.pick('source');h.find(node=>node.dataset.recentLot==='12').onclick();await h.pick('target');
  h.field('cases').value='4';h.submit();
  assert.equal(h.tickets.length,0);
  assert.deepEqual(h.failures,['Escribe lb por cajón para usar el lote seleccionado.']);
});
