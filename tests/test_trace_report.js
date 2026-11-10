const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const path=require('node:path');
const report=require('../dashboard/trace-report.js');
const fixture=JSON.parse(fs.readFileSync(path.join(__dirname,'visual/fixtures/trace-BB110927.json')));

test('BB110927 preserves every batch association, estimated share and actual gap count in screen and text',()=>{
  const data=report.build(fixture,'backward');
  const inputs=data.sections[0].rows;
  assert.equal(inputs.filter(r=>r[1]==='make #2947').length,7,'all seven recorded ingredients');
  const oats=inputs.filter(r=>r[3].includes('Oats'));
  assert.deepEqual(oats.map(r=>r[1]),['make #2926','make #2927','make #2936','make #2947','make #2965']);
  assert.deepEqual(oats.map(r=>r[6]),['87.1549 lb','217.8871 lb','458 lb','1,374 lb','109.2557 lb']);
  assert.equal(data.gaps.length,2);
  const html=report.html(data),text=report.text(data);
  for(const value of ['Trace gaps: 2','1,374 lb','109.2557 lb','Nation Harvest','National Harvest','not physically verified']) {
    assert(html.includes(value),`screen includes ${value}`);assert(text.includes(value),`text includes ${value}`);
  }
  assert(!text.includes('Legacy Gaps:     0'));
  assert(text.includes('Remaining quantity: 3,855 lb'));
});

test('shared upstream lots stay attached to each path and untrusted strings stay text',()=>{
  const changed=structuredClone(fixture);
  changed.forward_trace.lots[0].product_name='<img src=x onerror=alert(1)>';
  const r=report.build(changed,'backward');
  assert(report.html(r).includes('&lt;img'));
  assert(!report.html(r).includes('<img'));
  assert(report.text(r).includes('<img'));
  assert.equal(r.sections[0].rows.filter(row=>row[3].includes('(lot ID 616)')).length,5);
  assert.equal(r.gaps.filter(row=>row[0].includes('(lot ID 616)')).length,1,'unique origin gaps, not traversal count');
});

test('multiple outputs allocate a labelled proportional share without changing recorded use',()=>{
  const changed=structuredClone(fixture);
  const root=changed.input_tree;
  root.output_lb=1927.5;
  root.transactions[0].lot_output_lb=1927.5;
  const r=report.build(changed,'backward');
  const oats=r.sections[0].rows.find(row=>row[1]==='make #2947' && row[3].includes('Oats'));
  assert.equal(oats[5],'1,374 lb');assert.equal(oats[6],'687 lb');
});

test('pooled identity and historical notes never imply one verified supplier lot',()=>{
  const data=structuredClone(fixture);
  const input=data.input_tree.transactions[0].inputs[0].origin.transactions[0].inputs[0];
  input.entry_source_notes='POOLED opening stock; PROVISIONAL FIFO, not read from labels';
  const r=report.build(data,'backward');
  const html=report.html(r),text=report.text(r);
  assert(html.includes('Pooled inventory — no single supplier lot'));
  assert(text.includes('Pooled inventory — no single supplier lot'));
  assert(text.includes('PROVISIONAL FIFO, not read from labels'));
  assert(text.includes('Historical notes are recorded claims, not new physical verification.'));
});

test('commingled source entries are listed separately from receipts and never reduced to one lot',()=>{
  const data=structuredClone(fixture);
  const input=data.input_tree.transactions[0].inputs[0].origin.transactions[0].inputs[0];
  input.lot_type='commingled';
  input.supplier_lot_entries=[{supplier_lot_code:'SUP-A',supplier_name:'Source A',quantity_lb:20},{supplier_lot_code:'SUP-B',supplier_name:'Source B',quantity_lb:30}];
  const r=report.build(data,'backward');
  const entries=r.sections.find(s=>s.title==='Recorded supplier-lot entries').rows;
  assert.deepEqual(entries.map(e=>e[2]),['SUP-A','SUP-B']);
  for(const output of [report.html(r),report.text(r)])for(const value of ['Source A','Source B','SUP-A','SUP-B','Pooled inventory — no single supplier lot','not an allocation to individual receipts'])assert(output.includes(value));
});

test('multi-output makes show consumption once and label every repeated row',()=>{
  const data=structuredClone(fixture),n=data.forward_trace.lots[0];
  data.forward_trace.consumption=[1,2].map(id=>({source_lot_id:n.lot_id,lot_id:n.lot_id,type:'make',transaction_id:99,quantity_used:80,unit:'lb',output_quantity:40,produced_date:'2026-11-09'}));
  const r=report.build(data,'forward'),rows=r.sections.find(s=>s.title==='Forward — every consuming make and pack').rows;
  assert.equal(rows[0][3],'80 lb');assert.equal(rows[1][3],'Same consumption as above, do not add');
  assert(report.text(r).includes('Recorded quantity used: Same consumption as above, do not add'));
});
