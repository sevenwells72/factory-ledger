const {test}=require('node:test');
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const office=require('../dashboard/office-logic.js');
const source=fs.readFileSync('dashboard/office.js','utf8');
const flush=()=>new Promise(resolve=>setImmediate(resolve));
class Element{
 constructor(tag,text=''){this.tag=tag;this.textContent=String(text??'');this.children=[];this.parent=null;this.dataset={};this.value='';this.hidden=false;this.disabled=false;this.type='';this.required=false;this.events={};}
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
async function harness({on=true,create=true,language='en'}={}){
 const documentRoot=new Element('document');documentRoot.root=true;
 for(const id of ['content','resolution','notice','language','sign-out']){const node=new Element('div');node.id=id;documentRoot.append(node);}
 let nextTimer=0;const timers=new Map();
 const nodes=()=>descendants(documentRoot),byId=id=>nodes().find(node=>node.id===id),tickets=[],requests=[],failures=[],events={};
 let resolveRequest=body=>body.kind==='customer'?{outcome:'none',candidates:[]}:
  body.kind==='product'?{outcome:'match',match:{id:44,name:'Test product'}}:
  {outcome:'match',draft:{product_id:body.context.product_id,quantity:body.quantity,unit:body.query}};
 const ui={bind(){},session:{token:'session'},ready:Promise.resolve(on),lang:language,t:(en,es)=>language==='es'?es:en,relabel(){},
  el(tag,text,parent){const element=new Element(tag,text);parent?.append(element);return element;},
  notice(message){byId('notice').textContent=message;},failure(error){failures.push(error);},
  async request(path,body){requests.push({path,body});return path==='/auth/whoami'?{permissions:{create_order:true,create_expected_receipt:true,...(create?{create_customer:true}:{})}}:resolveRequest(body);},
  ticketCard(host,path,body,options){const ticket={host,path,body,options,invalidated:false};tickets.push(ticket);return {invalidate(){ticket.invalidated=true;host.replaceChildren();}};}
 };
 const context={setTimeout(fn,ms){const key=++nextTimer;timers.set(key,{fn,ms});return key;},clearTimeout(key){timers.delete(key);},FLLedger:ui,FLOffice:office,FLDashboard:{writesEnabled:on},URLSearchParams,location:{search:''},sessionStorage:{getItem(){return null;},removeItem(){}},document:{getElementById:byId},window:{addEventListener(name,callback){events[name]=callback;}}};
 vm.runInNewContext(source,context);await flush();await flush();
 const find=(test,root=documentRoot)=>descendants(root).find(test),button=(text,root)=>find(node=>node.tag==='button'&&node.textContent===text,root);
 async function search(kind,query){const box=find(node=>node.dataset.picker===kind);find(node=>node.tag==='input',box).value=query;await button('Search',box).onclick();return box;}
 async function open(){await search('customer','New Account');button("Add 'New Account' as a new customer").onclick();}
 return {ui,nodes,find,button,byId,tickets,requests,failures,events,search,open,timers,async tick(){for(const [key,timer] of [...timers]){timers.delete(key);await timer.fn();}await flush();},setResolver(fn){resolveRequest=fn;}};
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
