const {test}=require('node:test');const assert=require('node:assert/strict');
const fs=require('node:fs'),vm=require('node:vm'),office=require('../dashboard/office-logic.js');
const source=fs.readFileSync('dashboard/dashboard.js','utf8');
function functions(names,globals){const context=vm.createContext(globals);for(const name of names){const start=source.search(new RegExp('  (?:async )?function '+name+'\\('));assert.ok(start>=0,name);vm.runInContext(source.slice(start,source.indexOf('\n  }',start)+4),context);}return context;}
test('ready changes are offered from the server permission, including floor, and require an ORD ticket',()=>{
 assert.deepEqual(office.permitted({mark_order_ready:true},true).map(([action])=>action),['mark_order_ready']);
 assert.deepEqual(office.build('mark_order_ready',{order_id:7,ready:false,note:''}),{path:'/sales/orders/7/ready/prepare',body:{client_source:'dashboard',ready:false,note:null},prefix:'ORD'});
 assert.equal(office.build('mark_order_ready',{order_id:7,ready:true,note:'Pallet 3'}).body.note,'Pallet 3');
});
test('gated ready changes open a reviewed draft with the chosen mark and note, never the direct route',async()=>{
 const drafts=new Map(),visited=[];let calls=0;
 const ctx=functions(['postOrderReady'],{FLDashboard:{ready:Promise.resolve(true),writesEnabled:true},FLReceipts:{page:(page,params)=>({page,params})},sessionStorage:{setItem:(k,v)=>drafts.set(k,v)},location:{assign:v=>visited.push(v)},fetchSalesAPI:async()=>{calls++;return {};}});
 assert.equal(await ctx.postOrderReady({order_id:7,order_number:'SO-TEST'},false,'Pallet 3'),null);
 assert.equal(calls,0);assert.equal(visited[0].page,'office.html');assert.equal(visited[0].params.action,'mark_order_ready');assert.equal(visited[0].params.order,7);
 assert.deepEqual(JSON.parse(drafts.get('fl-order-ready-draft')),{order_id:7,ready:false,note:'Pallet 3'});
});
test('flag-off ready changes retain the legacy endpoint and payload',async()=>{
 const calls=[];const ctx=functions(['postOrderReady'],{FLDashboard:{ready:Promise.resolve(false),writesEnabled:false},fetchSalesAPI:async(...args)=>{calls.push(args);return {ready:true};}});
 assert.equal((await ctx.postOrderReady({order_id:7,order_number:'SO-TEST'},true,'Pallet 3')).ready,true);
 assert.equal(calls[0][0],'/sales-orders/SO-TEST/ready');assert.deepEqual(JSON.parse(calls[0][1].body),{ready:true,by:'floor',note:'Pallet 3'});
});
for(const mode of ['checkbox','note'])test('list '+mode+' opens a ticket without claiming or caching a recorded ready change',async()=>{
 const order={order_id:7,state:'open',ready:false,note:'Saved note'},events={},control={dataset:{orderId:'7'},checked:true,disabled:false,addEventListener:(name,fn)=>events[name]=fn},errors=[],posts=[];
 const container={querySelectorAll:()=>[control],querySelector:()=>({value:'New note'})};
 const ctx=functions([mode==='checkbox'?'bindOrderReadyToggles':'bindOrderReadyNoteControls'],{state:{ordersData:[order],orderReadyWrites:new Set()},document:{activeElement:null,querySelector:()=>control},postOrderReady:async(...args)=>{posts.push(args.slice(1));return null;},showError:(...args)=>errors.push(args),updateCachedOrderReady:()=>assert.fail('Unrecorded change cached'),refreshOrders:()=>assert.fail('Unrecorded change refreshed'),renderOrdersList:()=>assert.fail('Unrecorded change rendered')});
 if(mode==='checkbox'){ctx.bindOrderReadyToggles(container);await events.change();assert.equal(control.checked,false);}else{ctx.bindOrderReadyNoteControls(container);await events.click({stopPropagation(){}});assert.equal(order.note,'Saved note');}
 assert.equal(posts.length,1);assert.equal(order.ready,false);assert.deepEqual(errors,[]);assert.equal(control.disabled,false);
});
test('detail ready checkbox opens a ticket without reporting success or overwriting saved metadata',async()=>{
 const order={order_id:7,state:'open',ready:false,ready_note:'Saved note'},events={},control={checked:true,disabled:false,addEventListener:(name,fn)=>events[name]=fn},messages=[];
 const ctx=functions(['bindOrderDetailActions'],{state:{orderReadyWrites:new Set(),currentOrderDetail:order},document:{activeElement:null},orderDetailFlag:()=>({ready:false,note:'Saved note'}),ensureOrderDetailFlag:async()=>({ready:false,note:'Saved note'}),postOrderReady:async()=>null,updateCachedOrderReady:()=>assert.fail('Unrecorded change cached'),refreshOrders:()=>assert.fail('Unrecorded change refreshed'),refreshOrderDetail:()=>assert.fail('Unrecorded change refreshed'),setOrderDetailMessage:(...args)=>messages.push(args)});
 ctx.bindOrderDetailActions({querySelectorAll:()=>[],querySelector:()=>control},order);await events.change();assert.equal(control.checked,false);assert.equal(order.ready,false);assert.equal(order.ready_note,'Saved note');assert.deepEqual(messages,[]);
});
