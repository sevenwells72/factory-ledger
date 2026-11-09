/* Shared navigation, reference calendar and cross-record search. */
(() => {
  'use strict';
  function init() {
    const dashboard = !!document.getElementById('tab-operations');
    // index.html opts into the sidebar + top-bar shell; other pages keep the
    // page header and the bottom mobile nav below.
    const appShell = document.documentElement.hasAttribute('data-fl-shell');
    document.querySelectorAll('[data-mini-calendar]').forEach(calendar => {
      const details = document.createElement('details'); details.className = 'reference-calendar';
      const summary = document.createElement('summary'); summary.textContent = 'Reference calendar';
      if (appShell) summary.innerHTML = '<svg class="fl-svg" aria-hidden="true" focusable="false"><use href="#i-calendar"></use></svg><span class="rc-label">Reference calendar</span>';
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
    // Receipt numbers (write_tickets.PREFIXES: RCV-261013-001, ADJ-…, MK-…)
    // open that receipt through the Receipt # lookup instead of searching.
    // SO numbers share the shape, so only the receipt prefixes qualify.
    const RECEIPT=/^(?:RCV|ADJ|MK|PK|SHP|FND|VD|XR|LOT|ORD|ER|CUS)-\d+-\d+$/i;
    const receiptNumber=()=>{const q=input.value.trim();return RECEIPT.test(q)&&window.FLReceipts?q.toUpperCase():'';};
    function close() { generation++; results.classList.add('hidden'); input.setAttribute('aria-expanded','false'); input.removeAttribute('aria-activedescendant'); active=-1; }
    function select(record) {
      close();
      if (record.type==='receipt') FLReceipts.open(record.number);
      else if (dashboard) window.dispatchEvent(new CustomEvent('fl-search-select',{detail:record}));
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
      const receipt=receiptNumber();
      if(receipt){choices=[{type:'receipt',number:receipt,label:'Open receipt / Abrir recibo '+receipt}];results.replaceChildren();active=-1;
        const button=document.createElement('button');button.type='button';button.className='shell-search-result';button.id='global-result-0';button.setAttribute('role','option');button.setAttribute('aria-selected','false');button.textContent=choices[0].label;button.addEventListener('click',()=>select(choices[0]));results.append(button);
        announce.textContent='Press Enter to open receipt '+receipt;results.classList.remove('hidden');input.setAttribute('aria-expanded','true');return;}
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
      if(event.key==='Enter'&&receiptNumber()){event.preventDefault();select({type:'receipt',number:receiptNumber()});return;}
      if(results.classList.contains('hidden'))return;
      if(event.key==='ArrowDown'||event.key==='ArrowUp'){event.preventDefault();highlight(active < 0 ? (event.key==='ArrowDown' ? 0 : choices.length-1) : active+(event.key==='ArrowDown'?1:-1));}
      if(event.key==='Enter'&&active>=0){event.preventDefault();select(choices[active]);}
    });
    const incomingQuery=new URLSearchParams(location.search).get('query');if(incomingQuery){input.value=incomingQuery;input.dispatchEvent(new Event('input'));}
    document.addEventListener('click',event=>{if(!event.target.closest('.search-wrapper'))close();});
    document.addEventListener('keydown',event=>{if(event.key==='Escape')document.querySelectorAll('.reference-calendar[open],.mobile-more[open]').forEach(el=>el.open=false);});
    if(appShell)initAppShell(input,close);
    else {
    const nav=document.createElement('nav');nav.className='mobile-nav';nav.setAttribute('aria-label','Primary');
    const sections=[['operations','Operations'],['activity','Activity'],['orders','Sales Orders'],['supplies','Supplies']];
    function destination(key,label){const link=document.createElement('a');link.href='/?section='+key;link.textContent=label;link.dataset.section=key;
      if(dashboard)link.addEventListener('click',e=>{e.preventDefault();document.querySelector('.tab[data-tab="'+key+'"]').click();});return link;}
    sections.forEach(([key,label])=>nav.append(destination(key,label)));
    const more=document.createElement('details');more.className='mobile-more';const summary=document.createElement('summary');summary.textContent='More';more.append(summary);
    const menu=document.createElement('div');menu.className='mobile-more-menu';
    [['recent','Recent Entries'],['notes','Notes'],['expected','Expected Receipts']].forEach(([key,label])=>menu.append(destination(key,label)));
    // Sibling pages resolve next to this page, so they also work where the dashboard is served under /dashboard/.
    [['shift-summary.html','Shift Summary',false],['fl-assistant.html','FL Assistant',true]].forEach(([url,label,signedIn])=>{const a=document.createElement('a');a.href=new URL(url,location.href).href;a.textContent=label;if(signedIn)a.dataset.flWriteUi='';menu.append(a);});
    [['history.html?v=1','Ledger History'],['sankey.html?v=3','Material Flow'],['process-flow.html?v=4','Production Lines'],['traceability.html?v=4','Traceability']].forEach(([url,label])=>{const a=document.createElement('a');a.href='/'+url;a.textContent=label;menu.append(a);});
    more.append(menu);nav.append(more);document.body.append(nav);
    function sync(key){nav.querySelectorAll('[data-section]').forEach(a=>{if(a.dataset.section===key)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});summary.classList.toggle('active',!dashboard || ['recent','notes','expected'].includes(key));more.open=false;}
    window.addEventListener('fl-tab-change',e=>sync(e.detail));sync(new URLSearchParams(location.search).get('section')|| (dashboard?'operations':''));
    }
    if(dashboard){
      const actions=document.createElement('div');actions.className='mobile-quick-actions';
      [['er-new-btn','Expected receipt'],['supply-request-btn','Supply request'],['notes-add-btn','New note']].forEach(([id,label])=>{const target=document.getElementById(id);if(!target)return;const b=document.createElement('button');b.type='button';b.textContent=label;b.addEventListener('click',()=>target.click());actions.append(b);});
      document.getElementById('today-tile').before(actions);
      document.getElementById('tab-operations').prepend(document.querySelector('.today-tile-section'));
    }
  }
  // Dashboard app shell (index.html): collapsible sidebar, overlay drawer
  // below 900px, phone search/tab bar below 600px, and the user menu. Layout
  // only: every control keeps the handler dashboard.js or person-select.js
  // already bound to it.
  function initAppShell(input,closeResults){
    const root=document.documentElement,sidebar=document.getElementById('fl-sidebar'),scrim=document.getElementById('fl-scrim');
    const topbar=document.getElementById('fl-topbar'),menuButton=document.getElementById('fl-menu-btn'),moreButton=document.getElementById('fl-tabbar-more');
    const collapse=document.getElementById('fl-sidebar-collapse'),drawerClose=document.getElementById('fl-drawer-close');
    const media=query=>typeof window.matchMedia==='function'?window.matchMedia(query):{matches:false,addEventListener(){}};
    const drawerQuery=media('(max-width: 899px)'),phoneQuery=media('(max-width: 599px)');
    // Storage can throw (private mode, blocked site data); the sidebar must still work.
    const remember=value=>{try{localStorage.setItem('fl-sidebar',value);}catch(e){}};
    function paintCollapse(){const collapsed=root.dataset.flSidebar==='collapsed';collapse.setAttribute('aria-expanded',String(!collapsed));collapse.querySelector('.shell-sr').textContent=collapsed?'Expand sidebar':'Collapse sidebar';}
    collapse.addEventListener('click',()=>{if(root.dataset.flSidebar==='collapsed'){delete root.dataset.flSidebar;remember('expanded');}else{root.dataset.flSidebar='collapsed';remember('collapsed');}paintCollapse();});
    paintCollapse();

    let opener=null;
    const openers=[menuButton,moreButton].filter(Boolean);
    function openDrawer(from){opener=from;root.dataset.flDrawer='open';scrim.hidden=false;openers.forEach(b=>b.setAttribute('aria-expanded','true'));drawerClose.focus();}
    function closeDrawer(restore){if(root.dataset.flDrawer!=='open')return;delete root.dataset.flDrawer;scrim.hidden=true;openers.forEach(b=>b.setAttribute('aria-expanded','false'));if(restore&&opener)opener.focus();opener=null;}
    openers.forEach(b=>b.addEventListener('click',()=>root.dataset.flDrawer==='open'?closeDrawer(true):openDrawer(b)));
    drawerClose.addEventListener('click',()=>closeDrawer(true));scrim.addEventListener('click',()=>closeDrawer(true));
    document.addEventListener('keydown',e=>{if(e.key==='Escape'&&root.dataset.flDrawer==='open'&&!sidebar.querySelector('.reference-calendar[open]'))closeDrawer(true);});
    drawerQuery.addEventListener('change',()=>{if(!drawerQuery.matches)closeDrawer(false);});

    // Phones: the top bar holds only menu, logo, search and user, so the
    // calendar and health chip move into the drawer. Moving the nodes keeps
    // their listeners and ids (setHealthScore, initHealthDetails).
    const wideSlot=document.querySelector('[data-fl-wide-slot]'),phoneSlot=document.querySelector('[data-fl-phone-slot]');
    function place(){const target=phoneQuery.matches?phoneSlot:wideSlot;[document.querySelector('.reference-calendar'),document.getElementById('health-badge')].forEach(node=>{if(!node||node.parentElement===target)return;if(node.open)node.open=false;target.append(node);});}
    phoneQuery.addEventListener('change',place);place();

    // Phone search: an icon that expands the field across the top bar.
    const searchOpen=document.getElementById('fl-search-open'),searchClose=document.getElementById('fl-search-close');
    function endSearch(restore){if(!topbar.classList.contains('is-searching'))return;topbar.classList.remove('is-searching');searchOpen.setAttribute('aria-expanded','false');closeResults();if(restore)searchOpen.focus();}
    searchOpen.addEventListener('click',()=>{topbar.classList.add('is-searching');searchOpen.setAttribute('aria-expanded','true');input.focus();});
    searchClose.addEventListener('click',()=>endSearch(true));
    input.addEventListener('keydown',e=>{if(e.key==='Escape')endSearch(true);});
    phoneQuery.addEventListener('change',()=>{if(!phoneQuery.matches)endSearch(false);});

    // User menu: avatar + name button; the existing Change user button
    // (person-select.js, its own click handler) moves into the dropdown.
    const host=document.querySelector('[data-fl-person-host]'),chip=document.getElementById('fl-person-chip');
    const people=window.FLPersonSelect,change=chip&&chip.querySelector('.fl-person-change'),who=chip&&chip.querySelector('.fl-person-name');
    if(host&&chip&&change&&who){
      const toggle=document.createElement('button');toggle.type='button';toggle.className='fl-user-btn';toggle.setAttribute('aria-haspopup','true');toggle.setAttribute('aria-expanded','false');toggle.setAttribute('aria-controls','fl-user-menu');
      const avatar=document.createElement('span');avatar.className='fl-avatar';avatar.setAttribute('aria-hidden','true');
      const name=document.createElement('span');name.className='fl-user-name';
      const caret=document.createElementNS('http://www.w3.org/2000/svg','svg');caret.setAttribute('class','fl-svg fl-user-caret');caret.setAttribute('aria-hidden','true');caret.innerHTML='<use href="#i-chevron-down"></use>';
      toggle.append(avatar,name,caret);
      const menu=document.createElement('div');menu.className='fl-user-menu';menu.id='fl-user-menu';menu.hidden=true;
      const signed=document.createElement('p');signed.className='fl-user-menu-who';menu.append(signed,change);
      who.classList.add('shell-sr');chip.prepend(toggle);chip.append(menu);
      const paint=()=>{const display=(people&&people.person&&people.person.display_name)||who.textContent||'';avatar.textContent=display.charAt(0).toUpperCase();name.textContent=display;toggle.setAttribute('aria-label','User / Usuario: '+display);signed.textContent=display;};
      if(people&&typeof people.on==='function')people.on(()=>{paint();setMenu(false);});
      paint();
      function setMenu(open){menu.hidden=!open;toggle.setAttribute('aria-expanded',String(open));}
      toggle.addEventListener('click',()=>setMenu(menu.hidden));
      change.addEventListener('click',()=>setMenu(false));
      document.addEventListener('click',e=>{if(!menu.hidden&&!e.target.closest('.fl-user'))setMenu(false);});
      document.addEventListener('keydown',e=>{if(e.key==='Escape'&&!menu.hidden){setMenu(false);toggle.focus();}});
    }

    // Collapsed rail: the "Updated" time is visually hidden, so it rides on
    // the Refresh button's tooltip as well.
    const stamp=document.getElementById('last-refreshed'),refresh=document.getElementById('refresh-btn');
    if(stamp&&refresh&&typeof MutationObserver==='function')new MutationObserver(()=>{refresh.title=stamp.textContent?'Refresh all data. '+stamp.textContent:'Refresh all data';}).observe(stamp,{childList:true,characterData:true,subtree:true});

    // Phone tab bar: Dashboard returns to Operations through the tab's own handler.
    const home=document.querySelector('[data-fl-tab="dashboard"]'),operations=document.querySelector('.tab[data-tab="operations"]');
    if(home&&operations)home.addEventListener('click',e=>{e.preventDefault();operations.click();window.scrollTo(0,0);});
    const badge=document.querySelector('[data-fl-exceptions-count]');
    if(badge)window.addEventListener('fl-exceptions-count',e=>{const n=Number(e.detail);badge.hidden=!(Number.isFinite(n)&&n>0);badge.textContent=n>99?'99+':String(n);});
  }
  if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
})();
