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
async function harness({on=true,create=true}={}){
 const documentRoot=new Element('document');documentRoot.root=true;
 for(const id of ['content','resolution','notice','language','sign-out']){const node=new Element('div');node.id=id;documentRoot.append(node);}
 const nodes=()=>descendants(documentRoot),byId=id=>nodes().find(node=>node.id===id),tickets=[],requests=[],failures=[],events={};
 let resolveRequest=body=>body.kind==='customer'?{outcome:'none',candidates:[]}:
  body.kind==='product'?{outcome:'match',match:{id:44,name:'Test product'}}:
  {outcome:'match',draft:{product_id:body.context.product_id,quantity:body.quantity,unit:body.query}};
 const ui={bind(){},session:{token:'session'},ready:Promise.resolve(on),lang:'en',t:en=>en,relabel(){},
  el(tag,text,parent){const element=new Element(tag,text);parent?.append(element);return element;},
  notice(message){byId('notice').textContent=message;},failure(error){failures.push(error);},
  async request(path,body){requests.push({path,body});return path==='/auth/whoami'?{permissions:{create_order:true,...(create?{create_customer:true}:{})}}:resolveRequest(body);},
  ticketCard(host,path,body,options){const ticket={host,path,body,options,invalidated:false};tickets.push(ticket);return {invalidate(){ticket.invalidated=true;host.replaceChildren();}};}
 };
 const context={FLLedger:ui,FLOffice:office,FLDashboard:{writesEnabled:on},URLSearchParams,location:{search:''},sessionStorage:{getItem(){return null;},removeItem(){}},document:{getElementById:byId},window:{addEventListener(name,callback){events[name]=callback;}}};
 vm.runInNewContext(source,context);await flush();await flush();
 const find=(test,root=documentRoot)=>descendants(root).find(test),button=(text,root)=>find(node=>node.tag==='button'&&node.textContent===text,root);
 async function search(kind,query){const box=find(node=>node.dataset.picker===kind);find(node=>node.tag==='input',box).value=query;await button('Search',box).onclick();return box;}
 async function open(){await search('customer','New Account');button("Add 'New Account' as a new customer").onclick();}
 return {ui,nodes,find,button,byId,tickets,requests,failures,events,search,open,setResolver(fn){resolveRequest=fn;}};
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
