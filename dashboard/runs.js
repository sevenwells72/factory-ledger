/* S1 scheduling client. No sales-order writes; quantities stay in their entered unit. */
(() => {
  'use strict';
  const API = 'https://fastapi-production-b73a.up.railway.app';
  const $ = id => document.getElementById(id);
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const n = SOList.number, p = SOList.paragraph;
  const names = {planned:'Planned',in_progress:'In progress',done:'Done',cancelled:'Cancelled'};
  const active = run => ['planned','in_progress'].includes(run.status);
  const today = () => new Intl.DateTimeFormat('en-CA',{timeZone:'America/New_York',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());
  const addDays = (day, count) => { const d = new Date(day+'T12:00:00Z'); d.setUTCDate(d.getUTCDate()+count); return d.toISOString().slice(0,10); };
  const monday = day => addDays(day,-((new Date(day+'T12:00:00Z').getUTCDay()+6)%7));
  const date = (day, weekday=false) => new Intl.DateTimeFormat('en-US',{timeZone:'UTC',month:'short',day:'numeric',year:'numeric',...(weekday?{weekday:'long'}:{})}).format(new Date(day+'T12:00:00Z'));
  const explain = (key,text,definition,specifics,cls='') => SOList.trigger(key,esc(text),definition,p(specifics),cls);
  const errors = {
    CASE_WEIGHT_REQUIRED:'This product has no case weight. Choose pounds to plan this run.',
    PRODUCT_NOT_SCHEDULABLE:'This product cannot be scheduled. Choose an active finished product made in the factory.',
    PRODUCT_NOT_FOUND:'This product is no longer available. Search for another product.',
    PRODUCTION_LINE_NOT_FOUND:'That production line was not found. Check the line number or leave it blank.',
    INVALID_QUANTITY:'Enter a positive quantity that is at least one ten-thousandth of a pound.',
    INVALID_UNIT:'Choose cases or pounds.', INVALID_DATE:'Choose a valid planned date.',
    INVALID_STATUS:'Choose a valid run status.', INVALID_STATUS_TRANSITION:'Only planned and in-progress runs can be edited here.',
    RUN_NOT_FOUND:'This run could not be found. Close this panel and reload the week.',
    RUN_NOT_EDITABLE:'This run has already ended. Close this panel and reload the week.',
    RUN_NOT_ACTIVE:'This run has already ended. Close this panel and reload the week.',
    RUN_OVERCOVERED:'Coverage exceeds the planned pounds. Reduce coverage or increase the run quantity.',
    ORDER_NOT_OPEN:'This order is no longer open. Remove its coverage and reload the available lines.',
    LINE_NOT_OPEN:'This line is cancelled or fully shipped. Remove its coverage and reload the available lines.',
    LINE_NOT_FOUND:'This order line could not be found. Remove its coverage and reload the available lines.',
    LINE_PRODUCT_MISMATCH:'This line is for another product. Remove it from this run.',
    SERVICE_LINE_NOT_COVERABLE:'Service items cannot be covered by production.',
    COVERAGE_EXCEEDS_REMAINING:'Coverage is greater than this line still needs. Reduce its covered pounds.',
    DUPLICATE_LINE:'Each order line can appear only once.', NO_FIELDS:'Change a field before saving.'
  };
  async function request(route,method='GET',body) {
    try {
      const response = await FL.fetchWithTimeout(API+route,{method,headers:{'X-API-Key':'dashboard-key-2026',...(body===undefined?{}:{'Content-Type':'application/json'})},...(body===undefined?{}:{body:JSON.stringify(body)})});
      const data = await response.json().catch(()=>({}));
      if (!response.ok) { const detail=data.detail||{}; const e=new Error(errors[detail.error_code] || (response.status===401||response.status===403?'Access is unavailable. Contact the dashboard administrator.':'Could not reach the ledger. Please try again.')); e.detail=detail; throw e; }
      return data;
    } catch(e) { if(e.detail) throw e; throw new Error('Could not reach the ledger. Please try again.'); }
  }
  let start=monday(today()), selected='', runs=[], loadGeneration=0, dialogGeneration=0, busy=false, saveAction=null;
  function bind(root) { SOList.bindExplanations(root); }
  function explainInput(input,definition,specifics) {
    const info=SOList.explanation('input',definition,p(specifics));
    const holder=document.createElement('span');holder.innerHTML=info.content;input.after(holder);
    const id=holder.firstElementChild.id;input.dataset.explain=id;input.setAttribute('aria-describedby',id);
    SOList.bindExplanations(input.parentElement);
  }
  function render() {
    SOList.closeExplanation();
    $('week-range').textContent=date(start)+' – '+date(addDays(start,6));
    $('range-note').textContent='Counts cover the displayed Monday–Sunday range · Factory time (New York).';
    $('run-tabs').innerHTML=[['','This week'],...Object.entries(names)].map(([key,label])=>`<button type="button" role="tab" aria-controls="week-board" aria-selected="${selected===key}" tabindex="${selected===key?0:-1}" data-tab="${key}">${esc(label)} · ${n(key?runs.filter(r=>r.status===key).length:runs.length)}</button>`).join('');
    $('week-board').innerHTML=Array.from({length:7},(_,i)=>{
      const day=addDays(start,i), rows=runs.filter(r=>r.planned_date===day&&(!selected||r.status===selected));
      return `<section class="run-day"><h3>${date(day,true)}</h3><div role="table" aria-label="${date(day,true)} runs">${rows.length?rows.map(row).join(''):'<p class="empty-day">No runs</p>'}</div></section>`;
    }).join('');
    bind($('week-board'));
  }
  function row(r) {
    const native=r.planned_qty??r.planned_qty_lb, unit=r.planned_unit||'lb';
    const conversion=`${n(r.planned_qty_lb)} lb planned.`+(r.case_size_lb_used?` Saved case weight: ${n(r.case_size_lb_used)} lb per case.`:' Entered in pounds.');
    const coverage=(r.coverage||[]).map(c=>`${c.order_number}, line ${n(c.sales_order_line_id)}: ${n(c.qty_lb)} lb.`).join(' ')||'No sales order lines linked.';
    return `<div class="run-row" role="row" data-run-id="${r.id}">
      <div class="run-product-cell" role="cell">${explain('product',r.product_name,'The finished product planned for this run.',date(r.planned_date)+(r.notes?'. '+r.notes:''),'run-product')}</div>
      <div class="run-quantity-cell" role="cell">${explain('qty',`${n(native)} ${unit}`,'Planned quantity in the unit entered.',conversion,'run-quantity')}</div>
      <div class="run-line" role="cell">${esc(r.line_name||r.line_code||(r.line_id?`Line ${n(r.line_id)}`:''))}</div>
      <div class="run-status-cell" role="cell">${explain('status',names[r.status]||'Status unavailable','The run’s scheduling state.',`${names[r.status]||'State unavailable'} for ${date(r.planned_date)}.`+(r.status==='cancelled'&&r.notes?' '+r.notes:''),'status-chip')}</div>
      <div class="run-coverage-cell" role="cell">${explain('coverage',`covers ${n((r.coverage||[]).length)} lines · ${n(r.covered_lb||0)} lb of ${n(r.planned_qty_lb)} lb`,'Planned pounds linked to sales order lines; this is not inventory.',coverage,'run-coverage')}</div>
      <div class="run-action-cell" role="cell"><button type="button" data-open="${r.id}" aria-label="Open ${esc(r.product_name)} run">Open</button></div>
    </div>`;
  }
  async function load() {
    const generation=++loadGeneration;
    runs=[]; render(); $('week-board').hidden=true; $('retry-load').hidden=true; $('load-status').textContent='Loading runs…';
    try {
      const data=await request('/production/runs?'+new URLSearchParams({from:start,to:addDays(start,6)}));
      if(generation!==loadGeneration)return;
      runs=data.runs||[]; render(); $('week-board').hidden=false; $('load-status').textContent='';
    } catch(e) { if(generation!==loadGeneration)return; $('load-status').textContent=e.message; $('retry-load').hidden=false; }
  }
  const dialog=$('run-dialog');
  function closeDialog() { if(busy)return; dialogGeneration++; SOList.closeExplanation(); dialog.close(); }
  function panel(title,html,submit,action) {
    dialogGeneration++; SOList.closeExplanation();
    $('dialog-title').textContent=title; $('dialog-content').innerHTML=html; $('dialog-error').textContent='';
    $('save-run').hidden=!submit; $('save-run').textContent=submit||'Save'; $('save-run').disabled=false; saveAction=action;
    if(!dialog.open)dialog.showModal(); bind($('dialog-content'));
    $('close-dialog').focus();
    return dialogGeneration;
  }
  async function write(route,method,body) { const result=await request(route,method,body); dialog.close(); dialogGeneration++; await load(); return result; }
  function openRun(r) {
    panel(r.product_name,`${explain('plan',`${n(r.planned_qty??r.planned_qty_lb)} ${r.planned_unit||'lb'}`,'Planned production.',`${n(r.planned_qty_lb)} lb on ${date(r.planned_date)}.`)}
      ${p(date(r.planned_date,true)+(r.line_name?' · '+r.line_name:''))}${r.notes?p(r.notes):''}
      ${active(r)?'<div class="run-actions"><button type="button" id="edit-run">Edit run</button><button type="button" id="edit-coverage">Edit coverage</button><button type="button" id="complete-run">Complete</button><button type="button" id="cancel-run">Cancel run</button></div>':p('This run has ended and can no longer be changed.')}`);
    if(active(r)) {
      $('edit-run').onclick=()=>editRun(r); $('edit-coverage').onclick=()=>coverage(r);
      $('complete-run').onclick=()=>complete(r); $('cancel-run').onclick=()=>cancel(r);
    }
  }
  function editRun(r=null) {
    let product=r?{id:r.product_id,name:r.product_name,case_size_lb:r.case_size_lb_used}:null;
    let searchGeneration=0, timer;
    const gen=panel(r?'Edit run':'New run',`
      ${r?p(r.product_name):'<label>Finished product<input id="product-search" type="search" autocomplete="off" placeholder="Search product name or SKU"></label><div id="product-results" aria-label="Product matches"></div><p id="product-status" role="status"></p>'}
      <div class="form-pair"><label>Quantity<input id="planned-qty" type="number" min="0.0001" step="any" required value="${r?esc(r.planned_qty??r.planned_qty_lb):''}"></label>
      <label>Unit<select id="planned-unit"><option value="lb">lb</option><option value="cases" disabled>cases</option></select></label></div>
      <p id="case-reason" class="muted"></p>
      <label>Planned date<input id="planned-date" type="date" required value="${r?esc(r.planned_date):start===monday(today())?today():start}"></label>
      <label>Production line number (optional)<input id="planned-line" type="number" min="1" step="1" value="${r?.line_id??''}"></label>
      <p class="muted">Leave blank to use the product’s assigned line when creating a run.</p>
      ${r?`<label>Status<select id="planned-status"><option value="planned">Planned</option><option value="in_progress">In progress</option></select></label>`:''}
      <label>Notes<textarea id="planned-notes">${esc(r?.notes||'')}</textarea></label>`,r?'Save changes':'Create run',async()=>{
        if(!product){$('dialog-error').textContent='Choose a finished product from the search results.';return;}
        const body={planned_date:$('planned-date').value,line_id:$('planned-line').value?Number($('planned-line').value):null,notes:$('planned-notes').value};
        // Omit unchanged native fields on edit: avoid reconverting a historic case weight.
        if(!r||Number($('planned-qty').value)!==Number(r.planned_qty??r.planned_qty_lb)||$('planned-unit').value!==(r.planned_unit||'lb')) {
          body.planned_qty=Number($('planned-qty').value); body.planned_unit=$('planned-unit').value;
        }
        if(r)body.status=$('planned-status').value;else body.product_id=product.id;
        await write('/production/runs'+(r?'/'+r.id:''),r?'PATCH':'POST',body);
      });
    explainInput($('planned-qty'),'Production quantity in the chosen unit.','Enter cases or pounds; saved display quantities are rounded to whole numbers.');
    function applyProduct(prod,initial=false) {
      product=prod;
      const hasCase=Number(prod?.case_size_lb)>0;
      $('planned-unit').querySelector('[value="cases"]').disabled=!hasCase;
      if(!hasCase&&$('planned-unit').value==='cases')$('planned-unit').value='lb';
      if(initial&&r)$('planned-unit').value=r.planned_unit||'lb';
      $('case-reason').textContent=hasCase?'':'Cases unavailable: no case weight is recorded for this product. Use pounds.';
      if(!r) {
        $('product-search').value=prod.name; $('product-status').textContent='Selected: '+prod.name; $('product-results').innerHTML='';
        $('planned-line').value=prod.default_line_id??prod.line_id??'';
      }
    }
    if(r){
      applyProduct(product,true);$('planned-status').value=r.status;
      request('/products/search?'+new URLSearchParams({q:r.sku||r.product_name,limit:100})).then(data=>{
        if(gen!==dialogGeneration)return;
        const current=(data.products||[]).find(x=>Number(x.id)===Number(r.product_id));
        if(current)applyProduct(current);
      }).catch(()=>{});
    }
    else $('product-search').addEventListener('input',()=>{
      product=null; clearTimeout(timer); const search=++searchGeneration, q=$('product-search').value.trim();
      $('product-results').innerHTML=''; $('product-status').textContent=q?'Searching…':'';
      $('planned-unit').querySelector('[value="cases"]').disabled=true; $('planned-unit').value='lb';
      timer=setTimeout(async()=>{
        if(!q)return;
        try {
          const data=await request('/products/search?'+new URLSearchParams({q,limit:100}));
          if(gen!==dialogGeneration||search!==searchGeneration)return;
          const products=(data.products||[]).filter(x=>x.type==='finished'&&x.active!==false&&!x.is_service&&!x.no_production);
          $('product-results').innerHTML=products.map((x,i)=>`<button type="button" data-product="${i}">${esc(x.name)}</button>`).join('');
          $('product-status').textContent=products.length?'Choose a matching product.':'No finished products found. Try another name.';
          $('product-results').querySelectorAll('button').forEach(b=>b.onclick=()=>{searchGeneration++;applyProduct(products[Number(b.dataset.product)]);});
        }catch(e){if(gen===dialogGeneration&&search===searchGeneration)$('product-status').textContent=e.message;}
      },220);
    });
  }
  function cancel(r) {
    panel('Cancel run',p('Cancel the planned production of '+r.product_name+'? This run will no longer count as scheduled coverage.')+'<label>Reason note<textarea id="cancel-reason" required></textarea></label>','Confirm cancellation',()=>write(`/production/runs/${r.id}/cancel`,'POST',{reason:$('cancel-reason').value.trim()}));
  }
  async function complete(r) {
    const gen=panel('Complete run',p('Loading recorded production…'));
    try {
      const e=await request(`/production/runs/${r.id}/evidence`); if(gen!==dialogGeneration)return;
      const full=e.suggested_state==='looks_complete';
      const summary=e.suggested_state==='none'?'No production recorded':`${n(e.recorded_lb)} / ${n(e.planned_qty_lb)} lb recorded`;
      panel('Complete run',p(r.product_name)+`<div class="evidence-summary">${explain('evidence',summary,'Recorded production compared with this run’s plan.',`${n(e.recorded_lb)} lb recorded; ${n(e.planned_qty_lb)} lb planned.`)}</div>`+
        p(`Evidence window: ${date(e.window.from)} – ${date(e.window.to)}.`)+
        p('Evidence includes recorded production for this product in the window; it is not linked exclusively to this run.')+
        p('Completing this run creates no inventory, changes no sales order, and does not set Ready to Ship.')+
        (full?'':p('The recorded production does not meet the plan. Confirm anyway only if you intend to mark this run done.'))+
        '<label>Completion note (optional)<textarea id="completion-note"></textarea></label>',full?'Looks complete — Confirm':'Confirm anyway',()=>write(`/production/runs/${r.id}/complete`,'POST',{note:$('completion-note').value}));
    }catch(e){if(gen===dialogGeneration){$('dialog-content').innerHTML=p(e.message)+'<button type="button" id="retry-evidence">Retry evidence</button>';$('retry-evidence').onclick=()=>complete(r);}}
  }
  const matchesProduct=(line,r)=>line.product_id!=null?Number(line.product_id)===Number(r.product_id):r.sku?String(line.sku)===String(r.sku):line.product===r.product_name;
  async function coverage(r,customer='') {
    const gen=panel('Edit coverage',p('Loading open order lines…'));
    try {
      const data=await request('/sales/orders?'+new URLSearchParams({state:'open',limit:200,...(customer?{customer}:{})}));
      const candidates=(data.orders||[]).filter(o=>o.state==='open'&&(o.pallet_lines||[]).some(l=>matchesProduct(l,r)));
      const details=[];
      // Bound concurrency to avoid a burst of detail reads against the shared dashboard service.
      for(let i=0;i<candidates.length;i+=6) {
        details.push(...await Promise.all(candidates.slice(i,i+6).map(o=>request('/sales/orders/'+o.order_id))));
        if(gen!==dialogGeneration)return;
      }
      const lines=details.flatMap(o=>o.state==='open'?(o.lines||[]).filter(l=>matchesProduct(l,r)&&!['cancelled','fulfilled'].includes(l.line_status)&&!l.is_non_weight&&Number(l.readiness?.remaining_lb)>0).map(l=>({...l,order_number:o.order_number,remaining:Number(l.readiness.remaining_lb)})):[]);
      const saved=new Map((r.coverage||[]).map(c=>[c.sales_order_line_id,c]));
      const retained=(r.coverage||[]).filter(c=>!lines.some(l=>l.line_id===c.sales_order_line_id));
      let items=[...lines.map(l=>({id:l.line_id,order:l.order_number,remaining:l.remaining,qty:saved.get(l.line_id)?.qty_lb||0})),...retained.map(c=>({id:c.sales_order_line_id,order:c.order_number,remaining:null,qty:c.qty_lb}))];
      panel('Edit coverage',p(r.product_name)+
        '<label>Find customer (optional)<input id="coverage-customer" type="search" value="'+esc(customer)+'"></label><button type="button" id="coverage-reload">Reload available lines</button>'+
        ((data.orders||[]).length>=200?p('Showing the first '+n(200)+' open orders by ship date. Narrow by customer to find additional lines.'):'')+
        (retained.length?p('Existing links outside these available lines are preserved. Set their covered pounds to zero to remove them.'):'')+
        (items.length?items.map(item=>`<div class="coverage-item" data-line-id="${item.id}"><label><span>${esc(item.order)} · Line ${n(item.id)}</span><input aria-label="Covered pounds for ${esc(item.order)} line ${item.id}" data-cover="${item.id}" type="number" min="0" step="any" value="${item.qty}"></label>
          ${item.remaining===null?p('Existing link; remaining quantity unavailable.'):explain('remaining',`${n(item.remaining)} lb remaining`,'Pounds this open sales order line still needs.',`${item.order}, line ${n(item.id)}; effective shipments are deducted.`)}
          <p class="error line-error" role="alert"></p></div>`).join(''):p('No open lines for this product in the loaded orders.'))+
        '<div id="coverage-total"></div>','Save coverage',async()=>{
          const values=items.map(item=>({sales_order_line_id:item.id,qty_lb:Number(document.querySelector(`[data-cover="${item.id}"]`).value)}));
          document.querySelectorAll('.line-error').forEach(el=>el.textContent='');
          let invalid=false;
          items.forEach((item,i)=>{if(item.remaining!==null&&values[i].qty_lb>item.remaining){document.querySelector(`[data-line-id="${item.id}"] .line-error`).textContent=errors.COVERAGE_EXCEEDS_REMAINING;invalid=true;}});
          if(values.reduce((sum,v)=>sum+v.qty_lb,0)>Number(r.planned_qty_lb)){$('dialog-error').textContent=errors.RUN_OVERCOVERED;return;}
          if(invalid)return;
          try {await write(`/production/runs/${r.id}/coverage`,'PUT',{coverage:values.filter(v=>v.qty_lb>0)});}
          catch(e) {
            const id=e.detail?.sales_order_line_id??e.detail?.line_id;
            const field=items.some(x=>String(x.id)===String(id))?document.querySelector(`[data-line-id="${id}"] .line-error`):null;
            if(field)field.textContent=e.message;else throw e;
          }
        });
      function total() {
        SOList.closeExplanation(); const sum=items.reduce((v,item)=>v+Number(document.querySelector(`[data-cover="${item.id}"]`).value||0),0);
        $('coverage-total').innerHTML=explain('total',`${n(sum)} lb of ${n(r.planned_qty_lb)} lb planned`,'Total pounds linked to this run.',`Coverage across ${n(items.filter(item=>Number(document.querySelector(`[data-cover="${item.id}"]`).value)>0).length)} lines.`);
        bind($('coverage-total'));
      }
      document.querySelectorAll('[data-cover]').forEach(input=>{input.addEventListener('input',total);explainInput(input,'Pounds from this run assigned to this order line.','Enter zero to remove this link.');}); total();
      $('coverage-reload').onclick=()=>{
        // Preserve edits when narrowing: full replacement must retain off-page links.
        const coverageRows=items.map(item=>({sales_order_line_id:item.id,order_number:item.order,qty_lb:Number(document.querySelector(`[data-cover="${item.id}"]`).value)})).filter(c=>c.qty_lb>0);
        coverage({...r,coverage:coverageRows},$('coverage-customer').value.trim());
      };
    }catch(e){if(gen===dialogGeneration){$('dialog-content').innerHTML=p(e.message)+'<button type="button" id="retry-coverage">Retry order lines</button>';$('retry-coverage').onclick=()=>coverage(r,customer);}}
  }
  $('run-form').addEventListener('submit',async e=>{
    e.preventDefault(); if(busy||!saveAction)return;
    busy=true; $('dialog-error').textContent=''; $('save-run').disabled=true; $('dismiss-dialog').disabled=true; $('close-dialog').disabled=true;
    try { await saveAction(); }catch(error){$('dialog-error').textContent=error.message;}
    finally {busy=false; $('save-run').disabled=false; $('dismiss-dialog').disabled=false; $('close-dialog').disabled=false;}
  });
  $('close-dialog').onclick=closeDialog; $('dismiss-dialog').onclick=closeDialog;
  dialog.addEventListener('keydown',e=>{if(e.key==='Escape'&&dialog.querySelector('.so-explanation:not([hidden])'))e.preventDefault();});
  dialog.addEventListener('cancel',e=>{e.preventDefault();closeDialog();});
  $('new-run').onclick=()=>editRun(); $('retry-load').onclick=load;
  $('previous-week').onclick=()=>{start=addDays(start,-7);load();};
  $('next-week').onclick=()=>{start=addDays(start,7);load();};
  $('current-week').onclick=()=>{start=monday(today());selected='';load();};
  $('theme-toggle').onclick=()=>{const theme=document.documentElement.dataset.theme==='dark'?'light':'dark';document.documentElement.dataset.theme=theme;localStorage.setItem('dashboard-theme',theme);};
  $('run-tabs').addEventListener('click',e=>{const tab=e.target.closest('[data-tab]');if(tab){selected=tab.dataset.tab;render();$('run-tabs').querySelector('[aria-selected="true"]').focus();}});
  $('run-tabs').addEventListener('keydown',e=>{if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key))return;e.preventDefault();const tabs=[...$('run-tabs').children],index=tabs.indexOf(document.activeElement);tabs[e.key==='Home'?0:e.key==='End'?tabs.length-1:(index+(e.key==='ArrowRight'?1:-1)+tabs.length)%tabs.length].click();});
  $('week-board').addEventListener('click',e=>{const open=e.target.closest('[data-open]');if(open){const r=runs.find(r=>r.id===Number(open.dataset.open));if(r)openRun(r);}});
  load();
})();
