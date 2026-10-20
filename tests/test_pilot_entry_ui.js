const {test}=require('node:test');
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const source=fs.readFileSync('dashboard/entry.js','utf8');
const flush=async()=>{for(let i=0;i<4;i++)await new Promise(resolve=>setImmediate(resolve));};
class Element{
  constructor(tag,text=''){this.tag=tag;this.textContent=String(text??'');this.children=[];this.parent=null;this.dataset={};this.value='';this.hidden=false;this.disabled=false;this.type='';this.required=false;this.events={};}
  append(child){child.parent=this;this.children.push(child);}
  replaceChildren(...children){this.children.forEach(child=>child.parent=null);this.children=[];children.forEach(child=>this.append(child));}
  remove(){if(this.parent)this.parent.children=this.parent.children.filter(child=>child!==this);this.parent=null;}
  get isConnected(){return this.root||!!this.parent?.isConnected;}
  setAttribute(name,value){this[name]=value;}
  addEventListener(name,callback){this.events[name]=callback;}
  querySelectorAll(){return descendants(this).filter(node=>['input','select','textarea','button'].includes(node.tag));}
  scrollIntoView(){}
}
const descendants=root=>[root,...root.children.flatMap(descendants)];
const plain=value=>JSON.parse(JSON.stringify(value));
async function harness(action='ship',language='en'){
  const root=new Element('document');root.root=true;
  for(const id of ['content','resolution','notice','language','sign-out']){const node=new Element('div');node.id=id;root.append(node);}
  const find=(predicate,scope=root)=>descendants(scope).find(predicate),all=(predicate,scope=root)=>descendants(scope).filter(predicate),byId=id=>find(node=>node.id===id);
  const tickets=[],requests=[],failures=[];
  const lots=[{lot_id:11,lot_code:'OPEN261014-P1561',available_lb:50},{lot_id:12,lot_code:'PACK261019-001',available_lb:80}];
  let stock=async()=>({product_id:1561,product:'Classic Granola',lots});
  const order={order_id:5,order_number:'SO-PILOT',lines:[{line_id:101,product_id:1561,product:'Classic Granola',remaining_lb:60,line_status:'open'},{line_id:102,product_id:9,product:'Pallet',remaining_lb:0,is_service:true,line_status:'open'},{line_id:103,product_id:1561,product:'Cancelled',remaining_lb:10,line_status:'cancelled'},{line_id:104,product_id:1561,product:'Fulfilled',remaining_lb:0,line_status:'fulfilled'}]};
  const ui={formatNumber:require('../dashboard/ledger-ui.js').formatNumber,bind(){},ready:Promise.resolve(true),relabel(){},t:(en,es)=>language==='es'?es:en,el(tag,text,parent){const node=new Element(tag,text);parent?.append(node);return node;},failure(error){failures.push(error.message);},
    async request(path,body,options){requests.push({path,body,options});if(path==='/resolve')return {outcome:'match',match:{id:body.kind==='order'?5:body.kind==='customer'?7:body.kind==='supplier'?8:1561,label:body.kind}};if(path.startsWith('/sales/orders/'))return order;if(path.startsWith('/ship/lots?'))return stock(path);throw Error('Unexpected request '+path);},
    ticketCard(host,path,body,options){const ticket={host,path,body:plain(body),options};tickets.push(ticket);return {invalidate(){ticket.invalidated=true;}};}
  };
  vm.runInNewContext(source,{FLLedger:ui,FLDashboard:{writesEnabled:true},URL,URLSearchParams,location:{search:'?action='+action,href:'https://example.test/entry.html?action='+action},history:{replaceState(){}},document:{getElementById:byId},window:{addEventListener(){}}});await flush();
  const shipMode=find(node=>node.name==='ship_mode');if(shipMode)shipMode.value='order';
  const field=(name,scope=root)=>find(node=>node.name===name,scope),button=(text,scope=root)=>find(node=>node.tag==='button'&&node.textContent===text,scope);
  async function pick(name){const scope=find(node=>node.dataset.picker===name);find(node=>node.tag==='input',scope).value=name;await find(node=>node.tag==='button',scope).onclick();find(node=>node.className==='picker-options',scope).children[0].onclick();await flush();}
  function submit(){byId('entry-form').onsubmit({preventDefault(){}});}
  async function standalone(){field('ship_mode').value='standalone';field('ship_mode').onchange();await pick('customer');await pick('ship_product');field('quantity_lb').value='40';field('order_reference').value='PO-100';field('bol_reference').value='BOL-1';}
  function manual(scope){const mode=field('shipment_lot_mode',scope);mode.value='manual';mode.onchange();return scope;}
  function selectLot(id,quantity,scope){const select=field('shipment_lot_id',scope);select.value=String(id);select.onchange();field('shipment_lot_quantity',scope).value=String(quantity);}
  return {root,find,all,byId,field,button,pick,submit,standalone,manual,selectLot,tickets,requests,failures,order,lots,setStock(fn){stock=fn;}};
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
