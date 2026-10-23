const {test}=require('node:test');
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const office=require('../dashboard/office-logic.js');
const source=fs.readFileSync('dashboard/office.js','utf8');
const flush=()=>new Promise(resolve=>setImmediate(resolve));
class Element{
 constructor(tag,text=''){this.tag=tag;this.textContent=String(text??'');this.children=[];this.parent=null;this.dataset={};this.value='';this.hidden=false;this.disabled=false;this.type='';this.required=false;this.events={};}
 get textContent(){return this._textContent;}
 set textContent(value){this._textContent=String(value??'');}
 append(child){child.parent=this;this.children.push(child);}
 replaceChildren(...children){for(const child of this.children)child.parent=null;this.children=[];for(const child of children)this.append(child);}
 remove(){if(this.parent)this.parent.children=this.parent.children.filter(child=>child!==this);this.parent=null;}
 get isConnected(){return this.root||!!this.parent?.isConnected;}
 setAttribute(name,value){this[name]=value;}
 addEventListener(name,callback){this.events[name]=callback;}
 reportValidity(){return !this.required||!!this.value.trim();}
 scrollIntoView(){}
}
function descendants(root){return [root,...root.children.flatMap(descendants)];}
async function harness({on=true,create=true,language='en',permissions={}}={}){
 const documentRoot=new Element('document');documentRoot.root=true;
 for(const id of ['content','resolution','notice','language','sign-out']){const node=new Element('div');node.id=id;documentRoot.append(node);}
 let nextTimer=0;const timers=new Map();
 const nodes=()=>descendants(documentRoot),byId=id=>nodes().find(node=>node.id===id),tickets=[],requests=[],failures=[],events={};
 let resolveRequest=body=>body.kind==='customer'?{outcome:'none',candidates:[]}:
  body.kind==='product'?{outcome:'match',match:{id:44,name:'Test product'}}:
  {outcome:'match',draft:{product_id:body.context.product_id,quantity:body.quantity,unit:body.query}};
 const labelled=new Set();
 const ui={problem:value=>Error(String(value)),setText(node,value){node.textContent=String(value??'');if(value?.en!==undefined){node.labels=[value.en,value.es];labelled.add(node);}else{delete node.labels;labelled.delete(node);}return node;},bind(){},session:{token:'session'},ready:Promise.resolve(on),lang:language,codeText:require('../dashboard/ledger-ui.js').codeText,t:(en,es)=>Object.assign(new String(ui.lang==='es'?es:en),{en,es}),relabel(){for(const element of labelled){if(!element.isConnected){labelled.delete(element);continue;}element.textContent=element.labels[ui.lang==='es'?1:0];}},
  el(tag,text,parent){const element=new Element(tag,text);if(text?.en!==undefined){element.labels=[text.en,text.es];labelled.add(element);}parent?.append(element);return element;},
  notice(message){byId('notice').textContent=message;},failure(error){failures.push(error);},
  async request(path,body){requests.push({path,body});return path==='/auth/whoami'?{permissions:{create_order:true,create_expected_receipt:true,...(create?{create_customer:true}:{}),...permissions}}:resolveRequest(body);},
  ticketCard(host,path,body,options){const ticket={host,path,body,options,invalidated:false};tickets.push(ticket);return {invalidate(){ticket.invalidated=true;host.replaceChildren();}};}
 };
 const context={setTimeout(fn,ms){const key=++nextTimer;timers.set(key,{fn,ms});return key;},clearTimeout(key){timers.delete(key);},FLLedger:ui,FLOffice:office,FLDashboard:{writesEnabled:on},URLSearchParams,location:{search:''},sessionStorage:{getItem(){return null;},removeItem(){}},document:{getElementById:byId},window:{addEventListener(name,callback){events[name]=callback;}}};
 vm.runInNewContext(source,context);await flush();await flush();
 const find=(test,root=documentRoot)=>descendants(root).find(test),button=(text,root)=>find(node=>node.tag==='button'&&node.textContent===text,root);
 async function search(kind,query){const box=find(node=>node.dataset.picker===kind);find(node=>node.tag==='input',box).value=query;await button(language==='es'?'Buscar':'Search',box).onclick();return box;}
 async function open(){await search('customer','New Account');button("Add 'New Account' as a new customer").onclick();}
 return {ui,nodes,find,button,byId,tickets,requests,failures,events,search,open,timers,switchLanguage(next){language=ui.lang=next;events['ledger-language']();},async tick(){for(const [key,timer] of [...timers]){timers.delete(key);await timer.fn();}await flush();},setResolver(fn){resolveRequest=fn;}};
}
test('customer prepare sends exactly direct-route fields through its own CUS ticket',()=>{
 const fields={name:'New Account',contact_name:'Pat',email:'pat@example.test',phone:'555-0123',address:'1 Example Road',notes:'Call first',notes_es:'Llamar primero',customer_id:99,active:true};
 assert.deepEqual(office.build('create_customer',fields),{path:'/customers/prepare',prefix:'CUS',body:{client_source:'dashboard',name:fields.name,contact_name:fields.contact_name,email:fields.email,phone:fields.phone,address:fields.address,notes:fields.notes,notes_es:fields.notes_es}});
});
test('existing customer suggestions accept both prepare blockers and commit refusals',()=>{
 const existing={id:12,name:'Existing',active:true};
 assert.deepEqual(office.existingCustomer({blockers:[{code:'CUSTOMER_ALREADY_EXISTS',existing_customer:existing}]}),{...existing,label:'Existing'});
 assert.deepEqual(office.existingCustomer({detail:{error_code:'CUSTOMER_ALREADY_EXISTS',existing_customer:existing}}),{...existing,label:'Existing'});
 assert.equal(office.customerChoice({customer:{id:-1,name:'Wrong'}}),null);
});
test('new customer is hidden when sessions are off or create-customer permission is absent',async()=>{
 const off=await harness({on:false});assert.equal(off.byId('office-form'),undefined);assert.equal(off.requests.length,0);
 const denied=await harness({create:false});await denied.search('customer','Unknown');assert.equal(denied.find(n=>n.dataset.newCustomer!==undefined),undefined);
});
for(const outcome of ['match','ambiguous'])test('office offers a new customer beside similar '+outcome+' suggestions',async()=>{
 const h=await harness(),existing={id:88,name:'Sunshine Bakery'};
 h.setResolver(()=>outcome==='match'?{outcome,match:existing}:{outcome,candidates:[existing]});
 const picker=await h.search('customer','Sunrise Bakery');
 assert.ok(h.button('Sunshine Bakery',picker),'The existing customer remains an option');
 const add=h.button("Add 'Sunrise Bakery' as a new customer",picker);
 assert.ok(add,'The first search page offers the typed name as a new customer');
 add.onclick();assert.equal(h.find(n=>n.name==='name').value,'Sunrise Bakery');
 assert.equal(h.tickets.length,0,'Choosing Add only opens the customer form');
});
for(const [query,name] of [['Sun','Sunshine Granola'],['Green','Greenfield Bakery'],['Granola','Sunshine Granola'],['Wells','SevenWells Granola Co']]){
 for(const language of ['en','es'])test('Office '+query+' suggestions require an explicit customer choice in '+language,async()=>{
  const h=await harness({language}),spanish=language==='es',existing={id:88,name};
  h.setResolver(body=>body.kind==='customer'?{outcome:'ambiguous',match:null,candidates:[existing]}:
   body.kind==='product'?{outcome:'match',match:{id:44,name:'Test product'}}:
   {outcome:'match',draft:{product_id:44,quantity:2,unit:'cases'}});
  const picker=await h.search('customer',query),input=h.find(node=>node.tag==='input',picker);
  const addLabel=spanish?"Agregar '"+query+"' como cliente nuevo":"Add '"+query+"' as a new customer";
  assert.ok(h.button(name,picker));assert.ok(h.button(addLabel,picker));assert.equal(input.value,query);
  assert.equal(h.find(node=>/^(Selected:|Seleccionado:)/.test(node.textContent),picker),undefined);
  const request=h.requests.find(request=>request.body?.kind==='customer');
  assert.equal(request.path,'/resolve');assert.equal(request.body.context.group,'office');assert.equal(request.body.context.action,'order');
  const product=await h.search('product','Test');h.button('Test product',product).onclick();
  h.find(node=>node.name==='quantity').value='2';h.find(node=>node.name==='unit').value='cases';
  await h.byId('office-form').onsubmit({preventDefault(){}});
  assert.equal(h.tickets.length,0,'A suggestion cannot become an order customer on submit');
  assert.equal(h.failures.length,1);assert.equal(h.failures[0].message,'Choose a record / Elige un registro');
  assert.ok(h.button(addLabel,picker),'Add as new remains available after an unchosen suggestion');
  h.button(addLabel,picker).onclick();assert.equal(h.find(node=>node.name==='name').value,query);
  assert.equal(h.tickets.length,0,'Add as new opens the editor without recording');
  h.button(spanish?'Cancelar cliente nuevo':'Cancel new customer').onclick();
  h.button(name,picker).onclick();
  assert.equal(input.value,name);assert.ok(h.find(node=>node.textContent===(spanish?'Seleccionado: ':'Selected: ')+name,picker));
  await h.byId('office-form').onsubmit({preventDefault(){}});
  assert.equal(h.tickets.length,1);assert.equal(h.tickets[0].body.customer_id,88);
 });
}
test('inline customer prepares separately, keeps order fields and selects only after Record',async()=>{
 const h=await harness();h.find(n=>n.name==='customer_po').value='000NEW';h.find(n=>n.name==='quantity').value='2';h.find(n=>n.name==='unit').value='cases';
 await h.open();h.find(n=>n.name==='contact_name').value='Pat';h.button('Prepare customer').onclick();
 const customer=h.tickets[0];assert.equal(customer.path,'/customers/prepare');assert.equal(customer.body.contact_name,'Pat');assert.equal(customer.options.receiptPrefix,'CUS');assert.equal(h.button("Add 'New Account' as a new customer").disabled,true);
 assert.equal(h.requests.some(request=>request.path.endsWith('/commit')),false);assert.ok(!h.find(n=>n.dataset.picker==='customer').children.at(-1).textContent.startsWith('Selected:'));
 customer.options.onRecorded({customer_id:77,name:'New Account',customer:{id:77,name:'New Account'},receipt_number:'CUS-261015-001'});
 assert.equal(h.find(n=>n.dataset.picker==='customer').children.at(-1).textContent,'Selected: New Account');
 assert.equal(h.find(n=>n.name==='customer_po').value,'000NEW');assert.equal(h.find(n=>n.name==='quantity').value,'2');
 const product=await h.search('product','Test');h.find(n=>n.tag==='button'&&n.textContent==='Test product',product).onclick();
 await h.byId('office-form').onsubmit({preventDefault(){}});assert.equal(h.tickets[1].path,'/sales/orders/prepare');assert.equal(h.tickets[1].body.customer_id,77);assert.equal(h.tickets[1].body.customer_po,'000NEW');
 assert.equal(h.tickets[1].body.lines[0].quantity,2);assert.deepEqual(h.failures,[]);
});
test('duplicate suggests existing customer and inactive customers stay unavailable',async()=>{
 const h=await harness();await h.open();h.button('Prepare customer').onclick();const ticket=h.tickets[0];
 ticket.options.onPrepared({blockers:[{code:'CUSTOMER_ALREADY_EXISTS',existing_customer:{id:88,name:'Existing Account',active:false}}]});assert.equal(h.button('Use existing customer: Existing Account'),undefined);
 ticket.options.onFailure({result:{detail:{existing_customer:{id:88,name:'Existing Account',active:true}}}});h.button('Use existing customer: Existing Account').onclick();
 assert.equal(ticket.invalidated,true);assert.equal(h.find(n=>n.dataset.picker==='customer').children.at(-1).textContent,'Selected: Existing Account');assert.equal(h.button('Prepare draft').disabled,false);
});
test('changing customer search cancels a pending draft and ignores its late receipt',async()=>{
 const h=await harness();await h.open();h.button('Prepare customer').onclick();const ticket=h.tickets[0],input=h.find(n=>n.tag==='input',h.find(n=>n.dataset.picker==='customer'));
 input.value='Different';input.oninput();assert.equal(ticket.invalidated,true);ticket.options.onRecorded({customer:{id:99,name:'Old choice'}});
 assert.equal(input.value,'Different');assert.equal(h.button('Prepare draft').disabled,false);
});

test('signing out removes inline controls and ignores the previous draft receipt',async()=>{
 const h=await harness();await h.open();h.button('Prepare customer').onclick();const ticket=h.tickets[0];h.ui.session.token=null;await h.events['ledger-signout']();
 ticket.options.onRecorded({customer:{id:99,name:'Old choice'}});assert.equal(h.byId('office-form'),undefined);assert.equal(h.find(n=>n.dataset.newCustomer!==undefined),undefined);
});

test('customer review displays contact details before Record',()=>{
 const ledger=require('../dashboard/ledger-ui.js'),html=ledger.draftHtml({customer:{name:'New Account',contact_name:'Pat',email:'pat@example.test',phone:'555-0123',address:'1 Example Road'}});
 for(const detail of ['Contact name','Pat','Email','pat@example.test','Phone','555-0123','Address','1 Example Road'])assert.ok(html.includes(detail),detail);
});


test('prominent New customer opens an empty customer form without a search',async()=>{
 const h=await harness();h.button('New customer').onclick();
 assert.equal(h.byId('office-action').value,'create_customer');assert.equal(h.find(n=>n.name==='name').value,'');
 assert.equal(h.requests.some(request=>request.path==='/resolve'),false);assert.equal(h.find(n=>n.dataset.picker==='customer'),undefined);
 h.find(n=>n.name==='name').value='New Account';await h.byId('office-form').onsubmit({preventDefault(){}});
 assert.equal(h.tickets[0].path,'/customers/prepare');assert.equal(h.tickets[0].body.name,'New Account');
 h.tickets[0].options.onCancel();assert.equal(h.tickets[0].invalidated,true);assert.equal(h.find(n=>n.name==='name').value,'');
});
test('exact Setton Farms Add opens editor and duplicate refusal offers existing selection',async()=>{
 const h=await harness(),existing={id:5,name:'Setton Farms',active:true};h.setResolver(()=>({outcome:'match',match:existing}));
 const box=await h.search('customer','Setton Farms');const add=h.button("Add 'Setton Farms' as a new customer",box);
 assert.equal(add.className,'new-customer-option secondary');assert.ok(h.find(n=>n.tag==='hr',box));assert.ok(h.button('Setton Farms',box));
 add.onclick();assert.equal(h.find(n=>n.name==='name').value,'Setton Farms');assert.equal(h.tickets.length,0);
 h.button('Prepare customer').onclick();const ticket=h.tickets[0];assert.equal(ticket.body.name,'Setton Farms');
 ticket.options.onPrepared({can_commit:false,blockers:[{code:'CUSTOMER_ALREADY_EXISTS',message_en:"Customer 'Setton Farms' already exists",existing_customer:existing}]});
 h.button('Use existing customer: Setton Farms').onclick();assert.equal(ticket.invalidated,true);
 assert.equal(h.find(n=>n.dataset.picker==='customer').children.at(-1).textContent,'Selected: Setton Farms');assert.equal(h.tickets.length,1);
});
test('typing two characters debounces once; Enter cancels the queued search',async()=>{
 const h=await harness(),box=h.find(n=>n.dataset.picker==='customer'),input=h.find(n=>n.tag==='input',box);
 input.value='S';input.oninput();assert.equal(h.timers.size,0);input.value='Se';input.oninput();input.value='Set';input.oninput();
 assert.equal(h.timers.size,1);assert.equal([...h.timers.values()][0].ms,300);assert.equal(h.requests.filter(r=>r.path==='/resolve').length,0);
 await h.tick();assert.equal(h.requests.filter(r=>r.path==='/resolve').length,1);assert.equal(h.requests.at(-1).body.query,'Set');
 input.value='Setton';input.oninput();await input.onkeydown({key:'Enter',preventDefault(){}});await flush();
 assert.equal(h.timers.size,0);assert.equal(h.requests.filter(r=>r.path==='/resolve').length,2);
});
test('a pending search cannot replace the worker selected customer',async()=>{
 const h=await harness();let finish;h.setResolver(()=>new Promise(resolve=>{finish=resolve;}));
 const box=h.find(n=>n.dataset.picker==='customer'),input=h.find(n=>n.tag==='input',box);input.value='Setton';const pending=h.button('Search',box).onclick();
 input.value='Different';input.oninput();finish({outcome:'match',match:{id:5,name:'Setton Farms'}});await pending;
 assert.equal(h.button('Setton Farms',box),undefined);assert.equal(input.value,'Different');
});
test('expected delivery cases show calculated pounds and send only existing expected pounds contract',async()=>{
 const h=await harness();h.find(n=>n.dataset.officeStart==='create_expected_receipt').onclick();
 h.setResolver(body=>({outcome:'match',match:{id:body.kind==='supplier'?12:44,name:body.kind==='supplier'?'Supplier':'Sunflower Seeds'}}));
 for(const [kind,query] of [['supplier','Supplier'],['product','Sunflower']]){const box=await h.search(kind,query);h.find(n=>n.tag==='button'&&n.textContent===(kind==='supplier'?'Supplier':'Sunflower Seeds'),box).onclick();}
 const mode=h.find(n=>n.name==='expected_unit');mode.value='cases';mode.onchange();
 const cases=h.find(n=>n.name==='expected_cases'),size=h.find(n=>n.name==='expected_case_size');cases.value='10';size.value='25';size.oninput();
 assert.equal(h.find(n=>n.name==='expected_qty').value,250);await h.byId('office-form').onsubmit({preventDefault(){}});
 assert.equal(h.tickets[0].body.expected_qty,250);assert.equal(h.tickets[0].body.expected_cases,undefined);
});


for(const language of ['en','es'])test('direct new-customer start and duplicate choice are translated in '+language,async()=>{
 const h=await harness({language}),spanish=language==='es';h.button(spanish?'Cliente nuevo':'New customer').onclick();
 h.find(n=>n.name==='name').value='Setton Farms';await h.byId('office-form').onsubmit({preventDefault(){}});
 assert.ok(h.button(spanish?'Preparar cliente':'Prepare customer'));const ticket=h.tickets[0];
 ticket.options.onPrepared({blockers:[{code:'CUSTOMER_ALREADY_EXISTS',existing_customer:{id:5,name:'Setton Farms',active:true}}]});
 h.button((spanish?'Usar cliente existente: ':'Use existing customer: ')+'Setton Farms').onclick();
 assert.equal(h.byId('office-action').value,'create_order');assert.equal(h.find(n=>n.dataset.picker==='customer').children.at(-1).textContent,(spanish?'Seleccionado: ':'Selected: ')+'Setton Farms');
});

for(const language of ['en','es'])for(const action of ['create_order','add_order_lines'])test(action+' distinguishes cases and boxes on every order line in '+language,async()=>{
 const h=await harness({language,permissions:{add_order_lines:true}}),spanish=language==='es';
 const selector=h.byId('office-action');selector.value=action;selector.onchange();
 h.button(spanish?'Agregar línea':'Add line').onclick();
 const units=h.nodes().filter(node=>node.name==='unit');assert.equal(units.length,2);
 for(const unit of units){
  const label=value=>unit.children.find(option=>option.value===value).textContent;
  assert.equal(label('cases'),spanish?'Cajones':'Cases');assert.equal(label('boxes'),spanish?'Cajas':'Boxes');
  assert.notEqual(label('cases'),label('boxes'));assert.equal(label('pouches'),spanish?'Bolsitas → cajones':'Pouches → cases');
 }
});
for(const language of ['en','es'])test('expected delivery uses the same case word for mode, quantity, size and help in '+language,async()=>{
 const h=await harness({language}),spanish=language==='es';h.find(node=>node.dataset.officeStart==='create_expected_receipt').onclick();
 const mode=h.find(node=>node.name==='expected_unit'),cases=h.find(node=>node.name==='expected_cases'),size=h.find(node=>node.name==='expected_case_size');
 assert.equal(mode.children.find(option=>option.value==='cases').textContent,spanish?'Cajones':'Cases');
 assert.equal(cases.parent.textContent,spanish?'Cajones':'Cases');assert.equal(size.parent.textContent,spanish?'Tamaño de cajón (lb cada uno)':'Case size (lb each)');
 mode.value='cases';mode.onchange();
 const help=spanish?'Ingresa cajones y lb por cajón para calcular el total.':'Enter cases and lb per case to calculate the total.';
 assert.ok(h.find(node=>node.textContent===help));cases.value='10';size.value='25';size.oninput();
 assert.ok(h.find(node=>node.textContent===(spanish?'Total esperado: ':'Expected total: ')+'250 lb'));
});
for(const language of ['en','es'])test('pouch conversion uses the same case word on the form and Office draft in '+language,async()=>{
 const h=await harness({language}),spanish=language==='es';
 h.setResolver(body=>body.kind==='unit'?{outcome:'match',draft:{product_id:44,quantity:2,unit:'cases'},conversion:{from_quantity:24}}:{outcome:'match',match:{id:body.kind==='customer'?12:44,name:body.kind==='customer'?'Customer':'Product'}});
 for(const kind of ['customer','product']){const box=await h.search(kind,kind);h.button(kind==='customer'?'Customer':'Product',box).onclick();}
 h.find(node=>node.name==='quantity').value='24';h.find(node=>node.name==='unit').value='pouches';
 await h.byId('office-form').onsubmit({preventDefault(){}});
 assert.deepEqual(h.failures,[]);assert.equal(h.tickets[0].body.lines[0].quantity,2);assert.equal(h.tickets[0].body.lines[0].unit,'cases');
 const expected=spanish?'24 bolsitas → 2 cajones':'24 pouches → 2 cases';
 assert.equal(h.nodes().filter(node=>node.textContent===expected).length,2,'The line and draft both explain the same conversion');
});
test('switching Office languages relabels pouch conversion in place without changing the draft',async()=>{
 const h=await harness();
 h.setResolver(body=>body.kind==='unit'?{outcome:'match',draft:{product_id:44,quantity:2,unit:'cases'},conversion:{from_quantity:24}}:{outcome:'match',match:{id:body.kind==='customer'?12:44,name:body.kind==='customer'?'Customer':'Product'}});
 for(const kind of ['customer','product']){const box=await h.search(kind,kind);h.button(kind==='customer'?'Customer':'Product',box).onclick();}
 const quantity=h.find(node=>node.name==='quantity'),unit=h.find(node=>node.name==='unit');quantity.value='24';unit.value='pouches';
 await h.byId('office-form').onsubmit({preventDefault(){}});const ticket=h.tickets[0],body=JSON.stringify(ticket.body),form=h.byId('office-form');
 for(const [language,expected,cases,boxes] of [['es','24 bolsitas → 2 cajones','Cajones','Cajas'],['en','24 pouches → 2 cases','Cases','Boxes']]){
  h.switchLanguage(language);assert.equal(h.nodes().filter(node=>node.textContent===expected).length,2);
  assert.equal(unit.children.find(option=>option.value==='cases').textContent,cases);assert.equal(unit.children.find(option=>option.value==='boxes').textContent,boxes);
  assert.equal(h.byId('office-form'),form);assert.equal(form.hidden,true);assert.equal(quantity.value,'24');assert.equal(unit.value,'pouches');assert.equal(h.tickets.length,1);assert.equal(ticket.invalidated,false);assert.equal(JSON.stringify(ticket.body),body);
 }
});
test('switching Office languages relabels expected-delivery help and calculated total without clearing quantities',async()=>{
 const h=await harness();h.find(node=>node.dataset.officeStart==='create_expected_receipt').onclick();
 const mode=h.find(node=>node.name==='expected_unit'),cases=h.find(node=>node.name==='expected_cases'),size=h.find(node=>node.name==='expected_case_size'),pounds=h.find(node=>node.name==='expected_qty');mode.value='cases';mode.onchange();
 h.switchLanguage('es');assert.ok(h.find(node=>node.textContent==='Ingresa cajones y lb por cajón para calcular el total.'));assert.equal(cases.parent.textContent,'Cajones');assert.equal(size.parent.textContent,'Tamaño de cajón (lb cada uno)');
 h.switchLanguage('en');assert.ok(h.find(node=>node.textContent==='Enter cases and lb per case to calculate the total.'));
 cases.value='10';size.value='25';size.oninput();
 for(const [language,total] of [['es','Total esperado: 250 lb'],['en','Expected total: 250 lb']]){h.switchLanguage(language);assert.ok(h.find(node=>node.textContent===total));assert.equal(cases.value,'10');assert.equal(size.value,'25');assert.equal(pounds.value,250);assert.equal(mode.value,'cases');}
 assert.equal(h.tickets.length,0);
});
for(const action of ['approve_so','approve_er'])test(action+' distinguishes extracted case and box units and relabels them in place',async()=>{
 const h=await harness(),selector=h.byId('office-action');selector.value=action;selector.onchange();
 const lines=[{description:'Case product',quantity:3,unit:'cases',quantity_lb:75},{description:'Box product',quantity:4,unit:'boxes',quantity_lb:100}];
 h.setResolver(()=>({status:'extracted',document_kind:action==='approve_so'?'sales':'purchase',original_filename:'packing-list.pdf',extraction:{lines}}));
 h.find(node=>node.name==='document_id').value='41';await h.button('Load extracted document').onclick();
 for(const [language,caseText,boxText] of [['en','Extracted: 3 Cases','Extracted: 4 Boxes'],['es','Extraído: 3 Cajones','Extraído: 4 Cajas'],['en','Extracted: 3 Cases','Extracted: 4 Boxes']]){
  h.switchLanguage(language);assert.ok(h.find(node=>node.textContent===caseText));assert.ok(h.find(node=>node.textContent===boxText));
  assert.deepEqual(h.nodes().filter(node=>node.name==='quantity_lb').map(node=>node.value),[75,100]);
 }
 assert.deepEqual(lines.map(line=>line.unit),['cases','boxes']);assert.equal(h.tickets.length,0);assert.deepEqual(h.failures,[]);
});

test('Office selected-record status refreshes in both languages without changing the selected customer',async()=>{
 const h=await harness();h.setResolver(()=>({outcome:'match',match:{id:7,label:'Setton Farms'}}));const box=await h.search('customer','Setton');h.find(node=>node.tag==='button',h.find(node=>node.className==='picker-options',box)).onclick();assert.ok(h.find(node=>node.textContent==='Selected: Setton Farms'));h.ui.lang='es';h.events['ledger-language']();assert.ok(h.find(node=>node.textContent==='Seleccionado: Setton Farms'));h.ui.lang='en';h.events['ledger-language']();assert.ok(h.find(node=>node.textContent==='Selected: Setton Farms'));
});
