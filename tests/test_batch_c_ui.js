const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const read=name=>fs.readFileSync(path.join(__dirname,'../dashboard',name),'utf8');
function functionSource(source,name){const start=source.indexOf('  function '+name+'(');assert.ok(start>=0);return source.slice(start,source.indexOf('\n  }',start)+4);}
function helpers(){const window={addEventListener(){}};vm.runInNewContext(read('design-controls.js'),{window,URL,URLSearchParams,location:{href:'https://example.test/'}});return window.FLDesign;}
test('business dates do not shift zones; timestamp instants use the plant day across UTC midnight and DST',()=>{
 const f=helpers();assert.equal(f.date('2026-09-08'),'Sep 8, 2026');assert.equal(f.date('2026-02-30'),'2026-02-30');
 assert.match(f.time('2026-09-08T01:30:00Z'),/^Sep 7, 2026, 9:30 PM EDT$/);
 assert.match(f.time('2026-01-08T01:30:00Z'),/^Jan 7, 2026, 8:30 PM EST$/);
 assert.equal(f.time('2026-09-08 01:30 PM'),'Sep 8, 2026 01:30 PM ET');
 assert.equal(f.time('2026-09-08T01:30:00'),'Sep 8, 2026T01:30:00');
});
test('trace and product URLs preserve literal date-shaped codes and product identity',()=>{
 const f=helpers(),code='SEP 03 2026 / A&B';const url=new URL(f.traceURL(code,111),'https://example.test');
 assert.equal(url.searchParams.get('lot'),code);assert.equal(url.searchParams.get('product_id'),'111');
 const record={type:'product',id:111,name:'A&B / product'};assert.deepEqual(JSON.parse(new URL(f.recordURL(record),'https://example.test').searchParams.get('searchRecord')),record);
});
test('Other flow contributors reconcile to their displayed relationship values, without mixing stages',()=>{
 const ctx=vm.createContext({MIN_FLOW_LB:50,getProductionLine:()=> 'Baking'});vm.runInContext(functionSource(read('sankey.html'),'processData'),ctx);
 const make={transactions:[{lines:[{product_name:'A',quantity_lb:100},{product_name:'Oats',quantity_lb:-80},{product_name:'Honey',quantity_lb:-70}]},{lines:[{product_name:'B',quantity_lb:70},{product_name:'Oats',quantity_lb:-60}]}]};
 const ship={transactions:[{customer_name:'X',lines:[{product_name:'A',quantity_lb:-100}]},{customer_name:'Y',lines:[{product_name:'B',quantity_lb:-70}]}]};
 const links=ctx.processData(make,ship,{},1,1,1);
 assert.ok(links.some(l=>l.source.startsWith('Other')||l.target.startsWith('Other')));
 for(const link of links){assert.ok(link.members.length);assert.equal(link.members.reduce((n,m)=>n+m.value,0),link.value);}
 assert.equal(links.filter(l=>l.type==='fc').reduce((n,l)=>n+l.value,0),170);
});
test('unconfirmed or edited note references cannot pass the save gate; unchanged legacy references remain editable',()=>{
 const fields={'note-entity-type':{value:'product'},'note-entity-id':{value:'111',focus(){}},'note-reference-status':{textContent:''}};
 const ctx=vm.createContext({noteReference:null,document:{getElementById:id=>fields[id]}});
 vm.runInContext(functionSource(read('dashboard.js'),'validateNoteReference'),ctx);assert.equal(ctx.validateNoteReference(),false);
 ctx.noteReference={type:'product',value:'111'};assert.equal(ctx.validateNoteReference(),true);
 fields['note-entity-id'].value='112';assert.equal(ctx.validateNoteReference(),false);
 ctx.noteReference={type:'product',value:'Legacy product name'};fields['note-entity-id'].value='Legacy product name';assert.equal(ctx.validateNoteReference(),true);
});

test('production run day accepts explicit offsets and legacy UTC without appending a second timezone',()=>{
 const ctx=vm.createContext({});vm.runInContext(functionSource(read('process-flow.html'),'plantTransactionDay'),ctx);
 assert.equal(ctx.plantTransactionDay('2026-09-08T07:00:00-04:00'),'2026-09-08');
 assert.equal(ctx.plantTransactionDay('2026-09-08T01:00:00Z'),'2026-09-07');
 assert.equal(ctx.plantTransactionDay('2026-09-08T01:00:00'),'2026-09-07');
 assert.equal(ctx.plantTransactionDay('bad'),null);
});

test('activity sort keys compare noon and midnight in chronological order',()=>{
 const f=helpers();assert.equal(f.dateSortKey('2026-09-08','12:05 AM ET'),'2026-09-08T00:05:00');
 assert.equal(f.dateSortKey('2026-09-08','01:05 PM ET'),'2026-09-08T13:05:00');
 assert.ok(f.dateSortKey('2026-09-08','09:00 AM')<f.dateSortKey('2026-09-08','01:05 PM'));
});
