/* Shared record links, dates, search labels and table controls. */
(() => {
  'use strict';
  const F = window.FLDesign = {};
  F.icon = name => {
    const paths={menu:'M4 6h16 M4 12h16 M4 18h16',close:'M6 6l12 12 M18 6L6 18',sun:'M12 2v2 M12 20v2 M2 12h2 M20 12h2 M5 5l2 2 M17 17l2 2 M5 19l2-2 M17 7l2-2',moon:'M20 15a8 8 0 0 1-11-11 8 8 0 1 0 11 11',chevron:'M9 5l7 7-7 7',plus:'M5 12h14 M12 5v14',minus:'M5 12h14',sort:'M7 10l5-5 5 5 M7 14l5 5 5-5'};
    return '<svg class="fl-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="'+(paths[name]||paths.chevron)+'"/>'+(name==='sun'?'<circle cx="12" cy="12" r="4"/>':'')+'</svg>';
  };
  F.date = value => {
    const raw=String(value||''); if(!/^\d{4}-\d{2}-\d{2}$/.test(raw))return raw||'—';
    const d=new Date(raw+'T12:00:00Z');return Number.isNaN(+d)||d.toISOString().slice(0,10)!==raw?raw:new Intl.DateTimeFormat('en-US',{timeZone:'UTC',month:'short',day:'numeric',year:'numeric'}).format(d);
  };
  F.time = value => {
    const raw=String(value||'');
    // Legacy API values are already formatted in ET; do not invent an offset.
    if(!/(Z|[+-]\d{2}:\d{2})$/i.test(raw)){const text=raw.replace(/^(\d{4}-\d{2}-\d{2})/,m=>F.date(m));return (/^\d{4}-\d{2}-\d{2} \d{1,2}:\d{2} [AP]M$/.test(raw)?text+' ET':text)||'—';}
    const d=new Date(raw);return Number.isNaN(+d)?raw:new Intl.DateTimeFormat('en-US',{timeZone:'America/New_York',month:'short',day:'numeric',year:'numeric',hour:'numeric',minute:'2-digit',timeZoneName:'short'}).format(d);
  };
  F.dateSortKey = (day,time) => {
    const m=String(time||'').match(/^(\d{1,2}):(\d{2})(?::\d{2})?\s*([AP]M)?/i);
    if(!m)return day||'';let hour=Number(m[1]);if(m[3])hour=hour%12+(m[3].toUpperCase()==='PM'?12:0);
    return day+'T'+String(hour).padStart(2,'0')+':'+m[2]+':00';
  };
  F.recordURL = record => {const u=new URL('/',location.href);u.searchParams.set('searchRecord',JSON.stringify(record));return u.pathname+u.search;};
  F.setRecord = record => {const u=new URL(location.href);if(record){u.searchParams.set('searchRecord',JSON.stringify(record));u.searchParams.delete('query');}else u.searchParams.delete('searchRecord');history.replaceState({},'',u);};
  F.traceURL = (lot,product) => '/traceability.html?v=4&'+new URLSearchParams({lot,direction:'forward',...(product?{product_id:product}:{})});
  F.historyURL = (day,mode='event') => '/?'+new URLSearchParams({section:'activity',day,mode});
  F.filterCards = (input,selector) => {const q=input.value.trim().toLowerCase();document.querySelectorAll(selector).forEach(card=>card.hidden=!card.textContent.toLowerCase().includes(q));};
  function field(id,labelText) {
    const input=document.getElementById(id);if(!input || input.dataset.clearable)return;input.dataset.clearable='true';
    const wrapper=document.createElement('div');wrapper.className='labelled-search';input.before(wrapper);
    if(!document.querySelector('label[for="'+id+'"]')){const label=document.createElement('label');label.htmlFor=id;label.textContent=labelText;wrapper.append(label);}
    const row=document.createElement('div');row.className='clearable-field';wrapper.append(row);row.append(input);
    const clear=document.createElement('button');clear.type='button';clear.className='field-clear';clear.textContent='Clear';clear.setAttribute('aria-label','Clear '+labelText.toLowerCase());row.append(clear);
    const sync=()=>clear.hidden=!input.value;input.addEventListener('input',sync);input.addEventListener('change',sync);
    clear.addEventListener('click',()=>{input.value='';input.dispatchEvent(new Event('input',{bubbles:true}));input.dispatchEvent(new Event('change',{bubbles:true}));input.focus();sync();});sync();
  }
  function segments(id,name) {
    const select=document.getElementById(id);if(!select)return;
    const group=document.createElement('div');group.className='view-segments';group.setAttribute('role','group');group.setAttribute('aria-label',name);
    [...select.options].forEach(option=>{const b=document.createElement('button');b.type='button';b.textContent=option.textContent;b.dataset.value=option.value;b.addEventListener('click',()=>{select.value=option.value;select.dispatchEvent(new Event('change',{bubbles:true}));});group.append(b);});
    const sync=()=>group.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.value===select.value)));
    select.hidden=true;select.before(group);select.addEventListener('change',sync);sync();
  }
  function tableControls(table) {
    if(table.closest('#order-detail-container') || table.dataset.controlled || !table.tHead || !table.tBodies.length)return;
    table.dataset.controlled='true';
    const salesList=Boolean(table.closest('#orders-table-container'));
    const headers=[...table.tHead.rows[0].cells];
    const eligible=headers.map((h,i)=>({h,i,name:h.textContent.trim()})).filter(x=>x.name&&!x.h.classList.contains('order-ready-col'));
    if(!eligible.length)return;
    const owner=table.closest('[id]') || document.body;
    const preferenceKey='fl-table:'+location.pathname+':'+owner.id+':'+[...owner.querySelectorAll('table')].indexOf(table)+(salesList?':so-list-v3:'+(document.querySelector('[data-orders-tab][aria-selected="true"]')?.dataset.ordersTab || 'open'):'');
    let preferences={};try{preferences=JSON.parse(localStorage.getItem(preferenceKey)||'{}');}catch(_){}
    const persist=()=>{try{localStorage.setItem(preferenceKey,JSON.stringify(preferences));}catch(_){}};
    const bar=document.createElement('div');bar.className='table-tools';
    const title=salesList?'Sales Orders':table.closest('#order-detail-container')?'Order lines':table.closest('#tab-expected')?'Expected Receipts':table.closest('#tab-supplies')?'Supplies':table.closest('section')?.querySelector('h2,h3')?.textContent || 'Records';
    const label=document.createElement('label');label.textContent='Sort '+title+' ';const select=document.createElement('select');label.append(select);bar.append(label);
    const original=document.createElement('option');original.value='';original.textContent=salesList?'Default: order list priority':table.classList.contains('activity-table')?'Default: source date order':'Default: source order';select.append(original);
    const direction=document.createElement('button');direction.type='button';direction.textContent='Ascending';direction.disabled=true;bar.append(direction);
    const status=document.createElement('span');status.className='shell-sr';status.setAttribute('role','status');bar.append(status);
    const breakdownStates=new Map([...table.tBodies].filter(b=>b.id).map(b=>[b.id,b.className]));
    const groups=[];let span=0;
    // Keep expansion rows, lot breakdowns, and rowspan records attached to their parent.
    [...table.tBodies].flatMap(body=>[...body.rows]).forEach(row=>{
      const detail=row.matches('.lot-row,.activity-detail,.order-lines-row,.supply-lot-detail-row') || span>0;
      if(!detail || !groups.length)groups.push({row,rows:[],original:groups.length});
      groups.at(-1).rows.push(row);span=Math.max(span-1,...[...row.cells].map(c=>c.rowSpan-1),0);
    });
    groups.forEach(group=>{group.body=document.createElement('tbody');group.rows.forEach(row=>group.body.append(row));});
    [...table.tBodies].forEach(b=>b.remove());groups.forEach(g=>table.append(g.body));
    // Preserve separate lot-breakdown IDs/classes for existing expand handlers.
    groups.forEach(g=>{const id=g.row.dataset.expand;if(id&&g.rows.some(r=>r.classList.contains('lot-row'))){const details=document.createElement('tbody');details.id=id;details.className=breakdownStates.get(id)||'lot-breakdown';g.rows.slice(1).forEach(r=>details.append(r));g.body.after(details);g.extra=details;}});
    let descending=Boolean(preferences.descending);
    const value=(group,index)=>{const c=group.row.cells[index];return c?.dataset.sortValue ?? c?.textContent.trim() ?? '';};
    function sort(){
      const index=select.value===''?null:Number(select.value);direction.disabled=index===null;
      preferences.sort=select.value;preferences.descending=descending;persist();
      headers.forEach((h,i)=>h.setAttribute('aria-sort',i===index?(descending?'descending':'ascending'):'none'));
      const ordered=[...groups].sort((a,b)=>{if(index===null)return a.original-b.original;
        const av=value(a,index),bv=value(b,index);let cmp;
        const missing=v=>!v||v==='—';if(missing(av)!==missing(bv))return missing(av)?1:-1;
        if(headers[index].classList.contains('num'))cmp=(parseFloat(av.replaceAll(',',''))||0)-(parseFloat(bv.replaceAll(',',''))||0);
        else if(/date|ship by|occurred|entered/i.test(headers[index].textContent)){const ad=Date.parse(av),bd=Date.parse(bv);cmp=Number.isFinite(ad)&&Number.isFinite(bd)?ad-bd:av.localeCompare(bv);}
        else cmp=av.localeCompare(bv,undefined,{numeric:true,sensitivity:'base'});
        return (descending?-1:1)*cmp||a.original-b.original;});
      ordered.forEach(g=>{table.append(g.body);if(g.extra)table.append(g.extra);});
      if(index!==null){table.querySelectorAll('.overflow-row').forEach(r=>r.classList.remove('overflow-hidden'));if(table.tFoot)table.tFoot.hidden=true;}
      direction.textContent=descending?'Descending':'Ascending';status.textContent=index===null?'Default order restored':'Sorted by '+headers[index].textContent+', '+direction.textContent.toLowerCase()+'. All loaded records shown.';
    }
    eligible.forEach(({h,i,name})=>{const o=document.createElement('option');o.value=i;o.textContent=name;select.append(o);const b=document.createElement('button');b.type='button';b.className='table-sort';b.textContent=name;b.insertAdjacentHTML('beforeend',F.icon('sort'));b.title='Sort by '+name;h.replaceChildren(b);h.setAttribute('aria-sort','none');b.addEventListener('click',()=>{descending=select.value===String(i)?!descending:false;select.value=i;sort();});});
    select.addEventListener('change',()=>{descending=false;sort();});direction.addEventListener('click',()=>{descending=!descending;sort();});
    const resize=document.createElement('details');resize.className='table-resize';const summary=document.createElement('summary');summary.textContent='Resize columns';resize.append(summary);
    const widths=document.createElement('div');eligible.forEach(({h,i,name})=>{const l=document.createElement('label');l.textContent=name+' ';const range=document.createElement('input');range.type='range';range.min=80;range.max=640;range.step=16;range.value=preferences.widths?.[i] || Math.max(80,h.getBoundingClientRect().width||160);if(preferences.widths?.[i]){h.style.width=range.value+'px';h.style.minWidth=range.value+'px';}range.setAttribute('aria-label',name+' column width');range.addEventListener('input',()=>{h.style.width=range.value+'px';h.style.minWidth=range.value+'px';preferences.widths={...preferences.widths,[i]:Number(range.value)};persist();});l.append(range);widths.append(l);});resize.append(widths);bar.append(resize);table.before(bar);
    if(preferences.sort===undefined && salesList){select.value='3';descending=false;sort();}
    else if(preferences.sort!==undefined && [...select.options].some(o=>o.value===String(preferences.sort))){select.value=preferences.sort;sort();}
  }
  function init(){
    [['global-search','Search all records'],['orders-customer-search','Filter customers'],['supplies-search','Search supplies'],['er-text-filter','Filter expected receipts'],['er-supplier-filter','Find supplier by name or ID'],['lotSearch','Trace a lot'],['notes-search','Filter notes'],['recent-search','Filter recent entries']].forEach(x=>field(...x));
    document.querySelectorAll('.site-nav-toggle').forEach(b=>b.innerHTML=F.icon('menu'));
    document.querySelectorAll('.btn-close').forEach(b=>b.innerHTML=F.icon('close'));
    document.querySelectorAll('.graph-ctrl-btn').forEach(b=>{const label=b.getAttribute('aria-label')||'';if(label==='Zoom in'||label==='Zoom out')b.innerHTML=F.icon(label==='Zoom in'?'plus':'minus');});
    segments('er-status-filter','Expected receipt status');segments('daily-entries-mode','Activity date basis');
    const scan=()=>{document.querySelectorAll('table.orders-table,table.activity-table,table.inv-table,table.flow-relationships').forEach(tableControls);document.querySelectorAll('.order-expand-caret:not([data-vector])').forEach(el=>{el.dataset.vector='true';el.innerHTML=F.icon('chevron');});};
    let pending=false;new MutationObserver(()=>{if(!pending){pending=true;requestAnimationFrame(()=>{pending=false;scan();});}}).observe(document.body,{childList:true,subtree:true});scan();
    document.addEventListener('click',()=>{document.querySelectorAll('.clearable-field').forEach(row=>{row.querySelector('.field-clear').hidden=!row.querySelector('input').value;});});
    ['notes','recent'].forEach(kind=>{const input=document.getElementById(kind+'-search');if(input)input.addEventListener('input',()=>F.filterCards(input,kind==='notes'?'.note-card':'.recent-entry-card'));});
  }
  // shell.js creates global search on auxiliary pages during DOMContentLoaded.
  window.addEventListener('load',init);
})();
