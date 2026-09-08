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


test('factory preparation never masks advisory blockers or diverged closed shipments', () => {
  const ctx=context(dashboard,['renderDispatchState','renderOrderBlockers'],{
    SALES_ORDER_CLOSED_STATUSES:['shipped','invoiced','cancelled'],renderBlockerChips:()=> 'Shipment totals diverged',
  });
  assert.match(ctx.renderDispatchState({ready:true,status:'confirmed',dispatch_ready:false}),/Needs review/);
  assert.equal(ctx.renderDispatchState({status:'shipped',fulfillment_diverged:false}),'Not applicable');
  const diverged={status:'shipped',fulfillment_diverged:true,dispatch_ready:false};
  assert.match(ctx.renderDispatchState(diverged),/Needs review/);
  assert.equal(ctx.renderOrderBlockers(diverged),'Shipment totals diverged');
  assert.equal(ctx.renderDispatchState({status:'confirmed'}),'Not checked');
});
