const {test}=require('node:test');const assert=require('node:assert/strict');
const office=require('../dashboard/office-logic.js'),ui=require('../dashboard/ledger-ui.js');
const fs=require('node:fs');
test('office visibility fails closed with flag off or missing FL permission',()=>{
  const all=Object.fromEntries(office.actions.map(([a])=>[office.permission(a),true]));
  assert.equal(office.permitted(all,false).length,0);assert.equal(office.permitted({},true).length,0);
  assert.deepEqual(office.permitted({create_order:true},true).map(([a])=>a),['create_order','approve_so']);
});
const common={order_id:10,line_id:2,customer_id:3,product_id:4,supplier_id:5,expected_receipt_id:6,document_id:7,customer_po:'000123',lines:[{product_id:4,quantity:2,unit:'each'}],reason:'short_closed',note:'reason',status:'in_production',quantity_lb:20,unit_price:0,expected_qty:50};
for(const [action] of office.actions)test('A7 route and no direct write: '+action,()=>{
  const built=office.build(action,common);assert.ok(built.path.endsWith('/prepare'));assert.equal(built.body.client_source,'dashboard');
  assert.ok(['ORD','ER','CUS'].includes(built.prefix));assert.ok(!('mode' in built.body));assert.ok(!('created_by' in built.body));assert.ok(!('customer_name' in built.body));
});
test('PO keeps leading zeroes and No PO is explicit null',()=>{
  assert.equal(office.build('create_order',common).body.customer_po,'000123');
  assert.equal(office.build('create_order',{...common,no_po:true}).body.customer_po,null);
  assert.ok(!office.build('create_order',common).body.allow_duplicate_po,'override requires FL warning acknowledgement');
});
test('header and ER updates preserve explicit clear intent; zero price survives',()=>{
 assert.equal(office.build('update_order_line',common).body.unit_price,0);
 assert.deepEqual(office.build('update_order_header',{...common,notes:'',notes_es:'',requested_ship_date:''}).body,{client_source:'dashboard',customer_po:'000123',notes:null,notes_es:null,requested_ship_date:null,customer_id:3});
 assert.equal(office.build('update_expected_receipt',{...common,reference_number:''}).body.reference_number,null);
});
test('pouches use FL returned cases and reject ambiguous or foreign product drafts',()=>{
 assert.deepEqual(office.resolvedLine({outcome:'match',draft:{product_id:4,quantity:2,unit:'cases'}},4,24,'pouches'),{product_id:4,quantity:2,unit:'cases'});
 for(const result of [{outcome:'ambiguous',ask:'Remainder pouches'}, {outcome:'match',draft:{product_id:5,quantity:2,unit:'cases'}}])assert.throws(()=>office.resolvedLine(result,4,24,'pouches'));
 assert.throws(()=>office.resolvedLine({outcome:'match',draft:{product_id:4,quantity:9,unit:'each'}},4,2,'each'));
});
test('missing receipt is neutral and ticket outcomes remain fail closed',()=>{
 const source=fs.readFileSync('dashboard/receipt.js','utf8');assert.match(source,/No receipt with this number/);assert.match(source,/'neutral'/);
 assert.equal(ui.recorded({success:true}),false);assert.equal(ui.recorded({receipt_number:'ORD-261014-001',success:false}),false);
 const ticket=fs.readFileSync('dashboard/ledger-ticket.js','utf8');assert.match(ticket,/receiptPrefix/);
});

const vm=require('node:vm'),source=fs.readFileSync('dashboard/dashboard.js','utf8');
for(const [name,args] of [['saveOrderHeader',[{}]],['saveOrderLines',[{}]],['erSetStatus',[4,'cancelled',{}]],['openErModal',[{id:4}]],['saveEr',[]],['erApprove',[]],['soApprove',[]]])test('flag-on legacy '+name+' opens the office ticket screen without a direct write',async()=>{
 const start=source.indexOf('  async function '+name+'('),code=source.slice(start,source.indexOf('\n  }',start)+4),visited=[];
 const context=vm.createContext({FLDashboard:{ready:Promise.resolve(true),writesEnabled:true},FLReceipts:{page:(page,params)=>({page,params})},location:{assign:value=>visited.push(value)},state:{currentOrderDetail:{order_id:3},erEditing:{id:4},erIntake:{documentId:5},soIntake:{documentId:6}},fetchSalesAPI:()=>{throw Error('Direct route called');}});
 vm.runInContext(code,context);await context[name](...args);assert.equal(visited.length,1);assert.equal(visited[0].page,'office.html');
});


test('expected cases calculate pounds without accepting missing or zero package sizes',()=>{
 assert.equal(office.expectedPounds(10,25),250);assert.equal(office.expectedPounds(1.5,2.2),3.3);
 for(const [count,size] of [[0,25],[10,0],[-1,25],[10,Infinity],[NaN,25]])assert.throws(()=>office.expectedPounds(count,size));
});
test('expected delivery validation uses the same distinct Spanish case word',()=>{
 assert.throws(()=>office.expectedPounds(0,25),/Enter cases and lb per case \/ Ingresa cajones y lb por cajón/);
});
for(const language of ['en','es'])test('shared Office draft and receipt keep case and box wording distinct in '+language,()=>{
 const previous=ui.lang;ui.lang=language;
 try{
  const draft={cases:2,case_size_lb:25,lines:[{quantity:2,unit:'cases'},{quantity:3,unit:'boxes'}]},receipt={action:'create_order',draft};
  for(const value of [draft,ui.receiptData(receipt)]){
   const html=ui.draftHtml(value),text=html.replace(/<[^>]+>/g,' ');
   assert.match(text,language==='es'?/Cajones/:/Cases/);assert.match(text,language==='es'?/Cajas/:/Boxes/);assert.match(text,language==='es'?/lb por cajón/:/lb per case/);
   assert.match(html,/data-en="Cases" data-es="Cajones"/);assert.match(html,/data-en="Boxes" data-es="Cajas"/);
  }
 }finally{ui.lang=previous;}
});
test('product choices put regular first and preserve every explicit extra-kosher identity',()=>{
 const rows=[{id:2,kosher_tier:'extra_kosher'},{id:3,kosher_tier:'extra_kosher'},{id:1,kosher_tier:'regular'}];
 assert.deepEqual(office.productChoices(rows).map(row=>row.id),[1,2,3]);assert.equal(rows[0].id,2);
});
