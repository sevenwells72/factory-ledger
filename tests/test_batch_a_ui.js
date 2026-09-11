const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const dashboard = fs.readFileSync(path.join(__dirname, '../dashboard/dashboard.js'), 'utf8');
const production = fs.readFileSync(path.join(__dirname, '../dashboard/process-flow.html'), 'utf8');
// Run the actual presentation functions without booting the polling application.
function fn(source, name) {
  const start = source.search(new RegExp('  (?:async )?function ' + name + '\\('));
  assert.ok(start >= 0, name);
  const end = source.indexOf('\n  }', start) + 4;
  return source.slice(start, end);
}
function context(source, names, globals) {
  const ctx = vm.createContext(globals);
  vm.runInContext(names.map(name => fn(source, name)).join('\n'), ctx);
  return ctx;
}

test('receipt gate requires explicit product, supplier, finite positive pounds; edit and in-flight stay distinct', () => {
  const fields = Object.fromEntries(['er-product-id','er-supplier','er-qty','er-save-help','er-save-btn'].map(id => [id,{value:''}]));
  const state = {};
  const ctx = context(dashboard, ['readNumericInput','updateErSaveState'], {state,document:{getElementById:id=>fields[id]}});
  ctx.updateErSaveState();
  assert.equal(fields['er-save-btn'].disabled,true);
  fields['er-qty'].value='100'; fields['er-supplier'].value='Supplier';
  ctx.updateErSaveState(); assert.equal(fields['er-save-btn'].disabled,true);
  fields['er-product-id'].value='75';
  ctx.updateErSaveState(); assert.equal(fields['er-save-btn'].disabled,false);
  for(const value of ['0','-1','bad','1e999','']) {
    fields['er-qty'].value=value; ctx.updateErSaveState(); assert.equal(fields['er-save-btn'].disabled,true,value);
  }
  fields['er-qty'].value='100'; state.erSaving=true;
  ctx.updateErSaveState(); assert.equal(fields['er-save-btn'].disabled,true);
  state.erSaving=false; state.erEditing={id:1}; fields['er-product-id'].value='';fields['er-supplier'].value='';
  ctx.updateErSaveState(); assert.equal(fields['er-save-btn'].disabled,false);
});

test('dispatch attention counts the full response independently of order-list filters and fails honestly', async () => {
  const state={ordersData:[],attention:{failures:new Set()}};
  let result={orders:[{dispatch_ready:false},{dispatch_ready:true},{dispatch_ready:false}]};
  const ctx=context(dashboard,['refreshDispatchAttention','attnDispatchBlocked'],{state,fetchSalesAPI:async()=>result,renderAttentionStrip:()=>{}});
  await ctx.refreshDispatchAttention();assert.equal(ctx.attnDispatchBlocked(),2);
  result={orders:[{}]};await ctx.refreshDispatchAttention();assert.equal(ctx.attnDispatchBlocked(),null);
  assert.equal(state.attention.failures.has('dispatchAttention'),true);
  result={orders:[]};await ctx.refreshDispatchAttention();assert.equal(ctx.attnDispatchBlocked(),0);
  assert.equal(state.attention.failures.has('dispatchAttention'),false);
});

test('idle production is zero, missing run input is unavailable, known input computes yield', () => {
  const fields={};
  const ctx=context(production,['computeStages','fmtNum','renderDashboard'], {
    document:{getElementById:id=>fields[id] ||= {}},LINE_ORDER:['Line A'],LINE_DEFS:{},
    todayET:()=> '2026-09-08',escHtml:String,formatDateET:String,
  });
  ctx.renderDashboard({},{});
  assert.equal(fields['sum-produced'].textContent,'0 lb');
  assert.equal(fields['sum-yield'].textContent,'Not applicable');
  assert.match(fields['lines-grid'].innerHTML,/stage-value">0</);
  const tx={timestamp:'2026-09-08T10:00:00Z',outputProduct:'Product',outputLb:100,inputLb:0};
  ctx.renderDashboard({'Line A':{todayTx:[tx]}},{});
  assert.equal(fields['sum-produced'].textContent,'100 lb');
  assert.equal(fields['sum-yield'].textContent,'Unavailable');
  assert.match(fields['lines-grid'].innerHTML,/Unavailable \(input not recorded\)/);
  tx.inputLb=50;ctx.renderDashboard({'Line A':{todayTx:[tx]}},{});
  assert.equal(fields['sum-yield'].textContent,'200%');
});

test('rolling-deploy unit labels normalize weight packages and retain count-based supplies', () => {
  const ctx=context(dashboard,['ledgerUnit','operationalLabel'],{});
  assert.equal(ctx.ledgerUnit('25 lb case'),'lb');assert.equal(ctx.ledgerUnit('50 lb bag'),'lb');
  assert.equal(ctx.ledgerUnit('unit'),'unit');assert.equal(ctx.ledgerUnit('container'),'container');
  assert.equal(ctx.operationalLabel('adjusted_in'),'Stock added');
  assert.equal(ctx.operationalLabel('pack_output'),'Packed output');
});


test('sales order edit gates use administrative state and effective fulfillment independently', () => {
  const ctx=context(dashboard,['canEditOrderHeader','canEditOrderLines'],{});
  const makeOrder=(state,fulfillment,line_status='pending')=>({state,fulfillment,lines:[{line_status}],
    get status(){throw new Error('Legacy order status must never drive detail controls');}});
  assert.equal(ctx.canEditOrderHeader(makeOrder('open','unshipped')),true);
  for(const state of ['closed','cancelled']) for(const fulfillment of ['unshipped','partial','shipped']) {
    assert.equal(ctx.canEditOrderHeader(makeOrder(state,fulfillment)),false);
    assert.equal(ctx.canEditOrderLines(makeOrder(state,fulfillment)),false);
  }
  for(const fulfillment of ['partial','shipped'])assert.equal(ctx.canEditOrderHeader(makeOrder('open',fulfillment)),false);
  assert.equal(ctx.canEditOrderLines(makeOrder('open','partial')),true);
  assert.equal(ctx.canEditOrderLines(makeOrder('open','shipped','fulfilled')),false);
  assert.equal(ctx.canEditOrderLines(makeOrder('open','unshipped','cancelled')),false);
  assert.equal(ctx.canEditOrderHeader({}),false);
});

test('order line quantities use effective pounds and do not invent effective service shipments', () => {
  const ctx=context(dashboard,['orderLineQuantities'],{});
  const physical=ctx.orderLineQuantities({quantity_lb:1000,quantity_shipped_lb:999,
    readiness:{ordered_lb:1000,shipped_effective_lb:250,remaining_lb:750,allocated_lb:500,unallocated_need_lb:250}});
  assert.equal(physical.shipped,250);
  assert.equal(physical.remaining,750);
  const service=ctx.orderLineQuantities({is_non_weight:true,unit_quantity:3,quantity_lb:3,
    quantity_shipped_lb:1,remaining_lb:2,readiness:{ordered_lb:null,remaining_lb:null}});
  assert.equal(service.service,true);
  assert.equal(service.ordered,3);
  assert.equal(service.shipped,null,'No service effective shipment field means not tracked, not zero');
  assert.equal(service.remaining,2);
});

test('shipping preview formats structured warning quantities and does not repeat raw backend decimals', () => {
  const presentation=vm.createContext({document:{addEventListener(){}},window:{addEventListener(){}}});
  vm.runInContext(fs.readFileSync(path.join(__dirname,'../dashboard/so-list.js'),'utf8'),presentation);
  const ctx=context(dashboard,['renderShippingPreview'],{SOList:presentation.window.SOList,escHtml:String});
  for(const [warning_code,reserved_taken_lb,expected,meaning] of [
    ['STOCK_ALLOCATED',260.5,'261 lb',/blocked/i],
    ['STOCK_ALLOCATED',0,'1,501 lb',/blocked/i],
    ['RESERVED_STOCK_OBSERVE_ONLY',260.5,'261 lb',/advisory|does not prevent shipping/i],
  ]) {
    const html=ctx.renderShippingPreview({
      warnings:['Classic Granola: only 1240.0 lb currently takeable, need 1500.0 lb'],
      lines:[{line_id:1011,product:'Classic Granola',requested_ship_lb:1500,can_ship_lb:1240,short:260,
        reserved_others_lb:0,reserved_by_orders:[],allocation_warning:{warning_code,
          reserved_taken_lb,reserved_others_lb:1500.5,
          message:'Sales-order ship would use 260.5000 lb reserved for other orders. API commit RAW_DATABASE_SECRET'}}],
    });
    assert.match(html,/1,500 lb/);
    assert.match(html,/1,240 lb/);
    assert.match(html,/Short 260 lb/i);
    assert.ok(html.includes(expected),expected);
    assert.match(html,meaning);
    assert.doesNotMatch(html,/\b\d+\.\d+\s*lb|RAW_DATABASE_SECRET|API commit/);
  }
});
