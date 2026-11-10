(function() {
  'use strict';
  const {esc,qty,link,supplierLot}=FLTraceReport;
  const params=new URLSearchParams(location.search), productId=Number(params.get('product_id'));
  const status=document.getElementById('historyStatus'), content=document.getElementById('historyContent'), retry=document.getElementById('historyRetry');
  const date=x=>x ? FLDesign.date(x) : 'Not recorded';
  async function load() {
    content.replaceChildren(); retry.hidden=true; status.textContent='Loading lot history…';
    try {
      if (!Number.isInteger(productId) || productId<1) throw new Error('Choose a product from the inventory list.');
      const response=await FL.fetchWithTimeout(`${FLDashboard.base}/dashboard/api/inventory/${productId}/lot-history`, {headers:{'X-API-Key':await FLDashboard.key()}});
      if(!response.ok)throw new Error(`Unable to load lot history (HTTP ${response.status}).`);
      const data=await response.json(), finished=data.product_type==='finished';
      document.getElementById('historyTitle').textContent=`Lot history — ${data.product_name}`;
      document.title=`${data.product_name} lot history — Factory Ledger`;
      document.getElementById('backInventory').href=`index.html?section=operations#section-${finished ? 'finished-goods' : 'ingredients'}`;
      status.textContent=`${data.lots.length} lots · includes zero balances`;
      if(!data.lots.length){content.textContent='No posted receipts or production recorded for this product.';return;}
      const headers=finished ? ['Lot','Pack date (ET)','Cases packed','Pounds packed','Remaining','Other receipts'] :
        ['CNS lot','Received date (ET)','Supplier','Supplier lot','BOL','Received quantity','Remaining quantity'];
      const rows=data.lots.map(l=>{
        const n={...l,product_id:productId};
        const name=`<a class="lot-history-action" href="${esc(link(n,finished?'backward':'forward'))}">${esc(l.lot_code)}</a>${l.found_inventory ? '<small><strong>Found inventory</strong> — supplier origin may be incomplete</small>' : ''}${supplierLot(l).startsWith('Pooled inventory') ? '<small><strong>Pooled inventory — no single supplier lot</strong></small>' : ''}${(l.supplier_lot_entries || []).length ? `<details><summary>Recorded supplier lots</summary>${l.supplier_lot_entries.map(e=>`<p>${esc(e.supplier_name || 'Supplier not recorded')} · ${esc(e.supplier_lot_code || 'Lot not recorded')} · ${esc(qty(e.quantity_lb,data.unit))}</p>`).join('')}</details>` : ''}${l.entry_source_notes ? `<details><summary>Recorded lot notes</summary><p>${esc(l.entry_source_notes)}</p></details>` : ''}`;
        const receipts=l.receipts || [];
        const values=finished ? [name,esc((l.production || []).map(p=>date(p.produced_date)).join(', ') || 'No pack recorded'),esc(l.packed_cases==null?'Not recorded':l.packed_cases),esc(qty(l.packed_quantity)),esc(qty(l.remaining,data.unit)),esc(qty(l.received_quantity,data.unit))] :
          [name, ...[
            receipts.map(r=>date(r.received_date)),
            receipts.map(r=>r.kind==='received'?r.supplier_name:r.kind==='found_inventory'?'Found inventory':'Supplier not named'),
            receipts.map(r=>supplierLot(l)),
            receipts.map(r=>r.bol_reference || 'Not recorded'),
            receipts.map(r=>qty(r.quantity_lb,data.unit))
          ].map(list=>list.map(v=>`<div>${esc(v)}</div>`).join('') || 'Not recorded'),esc(qty(l.remaining,data.unit))];
        return `<tr>${values.map((v,i)=>`<td data-label="${esc(headers[i])}">${v}</td>`).join('')}</tr>`;
      });
      content.innerHTML=`<div class="lot-history-scroll"><table><thead><tr>${headers.map(h=>`<th scope="col">${esc(h)}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table></div>`;
    } catch(e) { status.textContent=e.accessUnavailable ? FLDashboard.accessMessage() : e.message; retry.hidden=false; }
  }
  retry.addEventListener('click',load);
  load();
})();
