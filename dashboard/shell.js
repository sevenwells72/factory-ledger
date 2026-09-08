/* Shared navigation, reference calendar and cross-record search. */
(() => {
  'use strict';
  function init() {
    const dashboard = !!document.getElementById('tab-operations');
    document.querySelectorAll('[data-mini-calendar]').forEach(calendar => {
      const details = document.createElement('details'); details.className = 'reference-calendar';
      const summary = document.createElement('summary'); summary.textContent = 'Reference calendar';
      const panel = document.createElement('div'); panel.className = 'reference-calendar-panel';
      const note = document.createElement('p'); note.textContent = 'Reference only — highlights order ship dates. Does not filter this page.';
      calendar.before(details); details.append(summary, panel); panel.append(note, calendar);
    });
    let input = document.getElementById('global-search');
    if (!input) {
      const wrap = document.createElement('div'); wrap.className = 'search-wrapper shared-search';
      wrap.innerHTML = '<input id="global-search" type="search" placeholder="Search SKU, lot, SO, customer…" autocomplete="off"><div id="search-results" class="search-dropdown hidden"></div>';
      document.querySelector('.header').append(wrap); input = wrap.querySelector('input');
    }
    const localTrace = document.getElementById('lotSearch');
    if (localTrace) {
      localTrace.setAttribute('aria-label', 'Search lots for trace');
      const label = document.createElement('label'); label.htmlFor = 'lotSearch'; label.textContent = 'Trace a lot'; label.className = 'trace-search-label';
      localTrace.closest('.search-row').before(label);
    }
    const results = document.getElementById('search-results');
    input.setAttribute('aria-label', 'Search all records'); input.setAttribute('role', 'combobox');
    input.setAttribute('aria-autocomplete','list'); input.setAttribute('aria-controls','search-results'); input.setAttribute('aria-expanded','false');
    results.setAttribute('role','listbox'); results.setAttribute('aria-label','Matching records');
    const announce = document.createElement('div'); announce.className='shell-sr'; announce.setAttribute('role','status'); input.after(announce);
    let generation=0, timer, active=-1, choices=[];
    function close() { generation++; results.classList.add('hidden'); input.setAttribute('aria-expanded','false'); input.removeAttribute('aria-activedescendant'); active=-1; }
    function select(record) {
      close();
      if (dashboard) window.dispatchEvent(new CustomEvent('fl-search-select',{detail:record}));
      else {
        const url=new URL('/',location.href); url.searchParams.set('searchRecord',JSON.stringify(record)); location.assign(url);
      }
    }
    function highlight(index) {
      const options=Array.from(results.querySelectorAll('[role="option"]')); if(!options.length)return;
      active=(index+options.length)%options.length;
      options.forEach((el,i)=>el.setAttribute('aria-selected',String(i===active)));
      input.setAttribute('aria-activedescendant',options[active].id); options[active].scrollIntoView({block:'nearest'});
    }
    input.addEventListener('input',()=>{
      close();clearTimeout(timer); const query=input.value.trim();if(query.length<2){announce.textContent='Enter at least two characters to search.';return;}
      const request= generation; announce.textContent='Searching…';
      timer=setTimeout(async()=>{
        try {
          const response=await FL.fetchWithTimeout('https://fastapi-production-b73a.up.railway.app/dashboard/api/search?q='+encodeURIComponent(query));
          if(!response.ok)throw Error('Search unavailable'); const data=await response.json(); if(request!==generation)return;
          choices=[...(data.products||[]).map(p=>({type:'product',id:p.product_id,name:p.name,label:'Product: '+p.name})),
            ...(data.lots||[]).map(l=>({type:'lot',id:l.product_id,name:l.lot_code,label:'Lot: '+l.lot_code+' — '+l.product_name})),
            ...(data.orders||[]).map(o=>({type:'order',id:o.order_id,label:'Order: '+o.order_number+' — '+o.customer})),
            ...(data.customers||[]).map(c=>({type:'customer',name:c.name,label:'Customer: '+c.name}))];
          results.replaceChildren(); active=-1;
          choices.forEach((record,i)=>{const button=document.createElement('button');button.type='button';button.className='shell-search-result';button.id='global-result-'+i;button.setAttribute('role','option');button.setAttribute('aria-selected','false');button.textContent=record.label;button.addEventListener('click',()=>select(record));results.append(button);});
          if(!choices.length) results.textContent='No matching records.';
          announce.textContent=choices.length+' matching records';results.classList.remove('hidden');input.setAttribute('aria-expanded','true');
        } catch(e) {if(request!==generation)return;results.textContent='Search unavailable. Edit the query to retry.';results.classList.remove('hidden');input.setAttribute('aria-expanded','true');announce.textContent=results.textContent;}
      },250);
    });
    input.addEventListener('keydown',event=>{
      if(event.key==='Escape'){close();announce.textContent='Search closed';return;}
      if(results.classList.contains('hidden'))return;
      if(event.key==='ArrowDown'||event.key==='ArrowUp'){event.preventDefault();highlight(active < 0 ? (event.key==='ArrowDown' ? 0 : choices.length-1) : active+(event.key==='ArrowDown'?1:-1));}
      if(event.key==='Enter'&&active>=0){event.preventDefault();select(choices[active]);}
    });
    const incomingQuery=new URLSearchParams(location.search).get('query');if(incomingQuery){input.value=incomingQuery;input.dispatchEvent(new Event('input'));}
    document.addEventListener('click',event=>{if(!event.target.closest('.search-wrapper'))close();});
    document.addEventListener('keydown',event=>{if(event.key==='Escape')document.querySelectorAll('.reference-calendar[open],.mobile-more[open]').forEach(el=>el.open=false);});
    const nav=document.createElement('nav');nav.className='mobile-nav';nav.setAttribute('aria-label','Primary');
    const sections=[['operations','Operations'],['activity','Activity'],['orders','Sales Orders'],['supplies','Supplies']];
    function destination(key,label){const link=document.createElement('a');link.href='/?section='+key;link.textContent=label;link.dataset.section=key;
      if(dashboard)link.addEventListener('click',e=>{e.preventDefault();document.querySelector('.tab[data-tab="'+key+'"]').click();});return link;}
    sections.forEach(([key,label])=>nav.append(destination(key,label)));
    const more=document.createElement('details');more.className='mobile-more';const summary=document.createElement('summary');summary.textContent='More';more.append(summary);
    const menu=document.createElement('div');menu.className='mobile-more-menu';
    [['recent','Recent Entries'],['notes','Notes'],['expected','Expected Receipts']].forEach(([key,label])=>menu.append(destination(key,label)));
    [['history.html?v=1','Ledger History'],['sankey.html?v=3','Material Flow'],['process-flow.html?v=4','Production Lines'],['traceability.html?v=4','Traceability']].forEach(([url,label])=>{const a=document.createElement('a');a.href='/'+url;a.textContent=label;menu.append(a);});
    more.append(menu);nav.append(more);document.body.append(nav);
    function sync(key){nav.querySelectorAll('[data-section]').forEach(a=>{if(a.dataset.section===key)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});summary.classList.toggle('active',!dashboard || ['recent','notes','expected'].includes(key));more.open=false;}
    window.addEventListener('fl-tab-change',e=>sync(e.detail));sync(new URLSearchParams(location.search).get('section')|| (dashboard?'operations':''));
    if(dashboard){
      const actions=document.createElement('div');actions.className='mobile-quick-actions';
      [['er-new-btn','Expected receipt'],['supply-request-btn','Supply request'],['notes-add-btn','New note']].forEach(([id,label])=>{const target=document.getElementById(id);if(!target)return;const b=document.createElement('button');b.type='button';b.textContent=label;b.addEventListener('click',()=>target.click());actions.append(b);});
      document.getElementById('today-tile').before(actions);
      document.getElementById('tab-operations').prepend(document.querySelector('.today-tile-section'));
    }
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
})();
