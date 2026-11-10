/* One evidence model for screen, print/PDF and text; graph deduplication never
   determines which transactions an auditor sees. */
(function(root) {
  'use strict';
  const esc = x => String(x ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number = x => Number(x).toLocaleString('en-US', {maximumFractionDigits:4});
  const qty = (x, unit='lb') => x == null ? 'Not determinable' : `${number(x)} ${unit}`;
  const lot = n => `${n.lot_code || 'Unidentified lot'} · ${n.product_name || ''} (lot ID ${n.lot_id ?? 'missing'})`;
  const supplierLot = n => (n.lot_type === 'commingled' || (n.supplier_lot_entries || []).length > 1 || /pooled/i.test(n.entry_source_notes || '')) ? 'Pooled inventory — no single supplier lot' : n.supplier_lot_code || 'Not recorded';
  const date = x => root.FLDesign ? root.FLDesign.date(x) : (x || 'Not recorded');
  function link(n, direction='backward') {
    return 'traceability.html?' + new URLSearchParams({lot:n.lot_code, product_id:n.product_id, direction});
  }
  function build(data, direction) {
    const forward = data.forward_trace;
    const start = forward.lots.find(n => n.lot_id === forward.start_lot_id);
    const report = {title:`Lot trace — ${start.lot_code}`, product:start.product_name,
      productId:start.product_id, direction, sections:[], gaps:[],
      generated:new Date().toLocaleString('en-US', {timeZone:'America/New_York'}) + ' ET'};
    const inputs=[], receipts=[], checks=[], notes=new Map(), sources=new Map(), receiptKeys=new Set(), gapKeys=new Set();
    function gap(n, reason, txn) {
      const key = `${n.lot_id}:${reason}:${txn || ''}`;
      if (gapKeys.has(key)) return;
      gapKeys.add(key); report.gaps.push([lot(n), txn ? `Txn #${txn}` : 'No transaction', reason]);
    }
    function receipt(n, r) {
      const key = `${n.lot_id}:${r.transaction_id}`;
      if (receiptKeys.has(key)) return;
      receiptKeys.add(key);
      const name = r.kind === 'received' ? r.supplier_name : r.kind === 'found_inventory' ? 'Found inventory — no supplier receipt' : 'Supplier not named';
      receipts.push([lot(n), r.transaction_id ? `Txn #${r.transaction_id}` : 'No receipt',
        date(r.received_date), name, supplierLot(n),
        r.bol_reference || 'Not recorded', r.receipt_number || 'Not recorded', qty(r.quantity_lb, n.unit)]);
      if (r.kind !== 'received') gap(n, name, r.transaction_id);
    }
    function recordSources(n) {
      for (const [i,e] of (n.supplier_lot_entries || []).entries())
        sources.set(`${n.lot_id}:${i}`,[lot(n),e.supplier_name || 'Not recorded',e.supplier_lot_code || 'Not recorded',qty(e.quantity_lb,n.unit),e.notes || '']);
    }
    function walk(n, origin, scope, path) {
      recordSources(n);
      if (n.entry_source_notes) notes.set(n.lot_id,[lot(n),n.entry_source_notes]);
      if (!origin) { gap(n, 'No posted origin'); return; }
      if (origin.truncated) { gap(n, `Trace stops: ${origin.truncated}`); return; }
      if (origin.kind !== 'made') {
        const all = origin.receipts || (origin.transaction_id ? [origin] : []);
        all.forEach(r => receipt(n,r));
        if (!all.length) gap(n, origin.kind === 'found_inventory' ? 'Found inventory — no supplier receipt' : 'No posted origin');
        return;
      }
      const extra = origin.also_received || [];
      extra.forEach(r => receipt(n,r));
      const total = Number(origin.output_lb || 0) + extra.reduce((s,r) => s + Number(r.quantity_lb || 0),0);
      const fraction = total > 0 && scope != null ? scope / total : null;
      for (const t of origin.transactions || []) {
        const outputFraction = t.output_lb > 0 ? (t.lot_output_lb ?? t.output_lb) / t.output_lb : null;
        if (t.inputs_mismatch) checks.push([lot(n), `Txn #${t.transaction_id}`, qty(t.inputs_total_lb), qty(t.output_lb), 'Recorded ingredients outside 80–120% of output']);
        if (!(t.inputs || []).length) gap(n, 'No ingredient lines recorded', t.transaction_id);
        for (const i of t.inputs || []) {
          const share = fraction == null || outputFraction == null ? null : i.quantity * fraction * outputFraction;
          inputs.push([path, `${t.type} #${t.transaction_id}`, date(t.produced_date), lot(i),
            supplierLot(i), qty(i.quantity,i.unit), qty(share,i.unit)]);
          walk(i, i.origin, share, `${path} ← ${i.lot_code || 'Unidentified lot'}`);
        }
      }
    }
    if (direction === 'backward') {
      const origin = data.input_tree || data.origin;
      const total = Number(origin?.output_lb || 0) + (origin?.also_received || []).reduce((s,r)=>s+Number(r.quantity_lb || 0),0);
      walk(start, origin, total, start.lot_code);
      report.sections.push({title:'Backward — recorded inputs and estimated share',
        note:'Recorded use is the whole make/pack amount. Estimated share assumes proportional mixing across all recorded production and receipts into each lot; it is not physically verified. Multiple makes, later receipts, losses and missing records can prevent exact attribution. Every possible supplier origin remains in the trace.',
        headers:['Output path','Transaction','Date (ET)','Input lot / product','Supplier lot','Recorded use','Estimated share for selected lot'], rows:inputs});
    } else {
      start.receipts.forEach(r => receipt(start,r));
      if (!start.receipts.length && start.product_type==='ingredient') gap(start, start.entry_source==='found_inventory' ? 'Found inventory — no supplier receipt' : 'No posted supplier receipt');
    }
    report.sections.push({title:'Supplier receipts', note:'Names and supplier lots are shown as recorded. Different spellings are not merged.',
      headers:['Lot / product','Transaction','Received (ET)','Supplier / origin','Supplier lot','BOL','Receipt','Received quantity'], rows:receipts});
    for (const n of forward.lots) {
      recordSources(n);
      if(n.entry_source_notes) notes.set(n.lot_id,[lot(n),n.entry_source_notes]);
    }
    if(sources.size) report.sections.push({title:'Recorded supplier-lot entries',note:'Lot-level source metadata as recorded, not an allocation to individual receipts or physical verification. Pooled inventory is not one supplier lot.',headers:['Lot / product','Supplier','Supplier lot','Recorded quantity','Recorded note'],rows:[...sources.values()]});
    if(notes.size) report.sections.push({title:'Recorded lot notes and assumptions',note:'Historical notes are recorded claims, not new physical verification.',headers:['Lot / product','Recorded note'],rows:[...notes.values()]});
    const byId = new Map(forward.lots.map(n=>[n.lot_id,n]));
    const shownConsumption=new Set();
    report.sections.push({title:'Forward — every consuming make and pack', note:forward.quantity_basis + ' Each input amount is shown once per consuming transaction; other output rows explicitly refer to that same consumption.',
      headers:['Consumed lot / product','Transaction','Date (ET)','Recorded quantity used','Output lot / product','Output quantity'],
      rows:forward.consumption.map(e=>{
        const key=`${e.source_lot_id}:${e.transaction_id}`, repeated=shownConsumption.has(key);shownConsumption.add(key);
        return [lot(byId.get(e.source_lot_id)), `${e.type} #${e.transaction_id}`,date(e.produced_date),repeated ? 'Same consumption as above, do not add' : qty(e.quantity_used,e.unit),
        e.lot_id == null ? 'No identified output lot' : lot(byId.get(e.lot_id)), qty(e.output_quantity,byId.get(e.lot_id)?.unit)];})});
    const shipped=[];
    for (const n of forward.lots) for (const sh of n.shipments) {
      shipped.push([lot(n),`Txn #${sh.transaction_id}`, date(sh.shipped_date), sh.customer_name || 'Customer not named',
        sh.sales_orders.map(o=>o.order_number).join(', ') || sh.order_reference || 'Not recorded',
        sh.shipment_ids.join(', ') || 'Not recorded',qty(sh.quantity_lb,n.unit)]);
      if (!sh.customer_name) gap(n,'Shipment customer not named',sh.transaction_id);
    }
    for (const g of forward.gaps) gap(byId.get(g.lot_id),g.reason,g.transaction_id);
    report.sections.push({title:'Forward — shipments and customers',note:'Quantities are recorded shipments from potentially exposed lots, not the weight of the starting ingredient. References do not multiply shipped quantity.',
      headers:['Lot / product','Transaction','Shipped (ET)','Customer','Sales order / reference','Shipment ID','Shipped quantity'],rows:shipped});
    report.sections.push({title:'Remaining stock — every traced lot',note:'Current posted-ledger balance, including zero and negative balances.',
      headers:['Lot / product','Remaining quantity','Remaining cases'],rows:forward.lots.map(n=>[lot(n),qty(n.on_hand,n.unit),n.case_size_lb ? number(n.on_hand/n.case_size_lb) : 'Not applicable'])});
    if (checks.length) report.sections.push({title:'Input checks',note:'Display only; nothing was changed.',headers:['Lot / product','Transaction','Recorded inputs','Output','Check'],rows:checks});
    report.sections.push({title:`Trace gaps: ${report.gaps.length}`,headers:['Lot / product','Transaction','Gap'],rows:report.gaps});
    return report;
  }
  function html(r) {
    return `<h3>${esc(r.title)}</h3><p>${esc(r.product)} · ${esc(r.generated)}</p><p><strong>Trace gaps: ${r.gaps.length}</strong> · ${r.direction === 'backward' ? 'Backward to suppliers and forward to customers' : 'Forward to batches, packed lots and customers'}</p>` +
      r.sections.map(s=>`<section class="audit-section"><h4>${esc(s.title)}</h4>${s.note ? `<p>${esc(s.note)}</p>` : ''}${s.rows.length ?
        `<div class="audit-table-scroll" tabindex="0" role="region" aria-label="${esc(s.title)}"><table class="detail-table"><thead><tr>${s.headers.map(h=>`<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${s.rows.map(row=>`<tr>${row.map((v,i)=>`<td data-label="${esc(s.headers[i])}">${esc(v)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>` : '<p>No records in this section.</p>'}</section>`).join('');
  }
  function text(r) {
    return `LOT TRACEABILITY AUDIT REPORT\n${r.title}\n${r.product}\nGenerated: ${r.generated}\nTrace gaps: ${r.gaps.length}\n` +
      r.sections.map(s=>`\n${s.title.toUpperCase()}\n${s.note || ''}\n${s.rows.length ? s.rows.map(row=>row.map((v,i)=>`${s.headers[i]}: ${v}`).join('\n')).join('\n\n') : 'No records in this section.'}\n`).join('');
  }
  const api={build,html,text,link,esc,qty,supplierLot};
  if(typeof module!=='undefined')module.exports=api;
  root.FLTraceReport=api;
})(typeof window==='undefined'?globalThis:window);
