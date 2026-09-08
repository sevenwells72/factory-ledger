export function inspect(region) {
 const root=region?document.querySelector(region):document.body;
 if(!root)return {missing:true};
 const visible=e=>e.checkVisibility?.()!==false&&!!(e.getClientRects().length&&e.getBoundingClientRect().width&&getComputedStyle(e).visibility!=='hidden'&&!e.closest('.hidden'));
 const path=e=>e.id?'#'+e.id:e.tagName.toLowerCase()+'.'+[...e.classList].join('.');
 const elements=sel=>[...(root.matches(sel)?[root]:[]),...root.querySelectorAll(sel)].filter(visible);
 const inputs=elements('input:not([type=hidden]),select,textarea').map(e=>{
   const label=[...(e.labels||[])].map(l=>l.innerText).join(' ')||e.getAttribute('aria-label')||e.closest('.form-group,.ctl,.m-row')?.querySelector('label')?.innerText||'';
   const type=e.type||e.tagName.toLowerCase(),mode=e.inputMode||'',hint=[e.id,e.name,label,e.placeholder].join(' ');
   const numeric=/qty|quantity|weight|price|ttl|cases|pallets|workers|minutes|hours|capacity|yield|crew|rate|horizon/i.test(hint)&&!/note|search|title|customer/i.test(hint);
   const mismatch=numeric&&!['number','range','select-one','checkbox','radio'].includes(type)&&!['numeric','decimal'].includes(mode);
   return {path:path(e),id:e.id,label,type,inputmode:mode,placeholder:e.placeholder||'',expected:numeric?'numeric/decimal':type==='date'?'date':'text/selection',mismatch,disabled:e.disabled,tabindex:e.tabIndex};
 });
 const wrap=elements('.order-link,.lot-code,.lot-link,.att-label,.section-header h2,.section-header h3,.section-hint,.product-name,#health-badge').map(e=>{
   const range=document.createRange();range.selectNodeContents(e);const rects=[...range.getClientRects()];
   const lines=new Set(rects.filter(r=>r.width>0).map(r=>Math.round(r.top))).size;
   const r=e.getBoundingClientRect(),cs=getComputedStyle(e);
   // Per-word ranges distinguish ordinary wrapping from mid-word fragmentation.
   const brokenWords=[];const walk=document.createTreeWalker(e,NodeFilter.SHOW_TEXT);let node;
   while(node=walk.nextNode()){for(const m of node.textContent.matchAll(/[A-Za-z]{4,}/g)){const t=document.createRange();t.setStart(node,m.index);t.setEnd(node,m.index+m[0].length);if(new Set([...t.getClientRects()].map(r=>Math.round(r.top))).size>1)brokenWords.push(m[0]);}}
   return {path:path(e),text:e.innerText?.slice(0,180)||'',lines,brokenWords,w:r.width,h:r.height,x:r.x,y:r.y,font:cs.fontSize,whiteSpace:cs.whiteSpace,overflowWrap:cs.overflowWrap,clipped:e.scrollWidth>e.clientWidth+1&&['hidden','clip'].includes(cs.overflowX)};
 });
 const statuses=elements('.date-overdue,.supply-low-stock,.health-badge,.readiness-chip,.so-status,.er-status,.allocation-status,.status-badge,.day-dot,.dot,.legend-item,.line-status,.badge').map(e=>({path:path(e),text:e.innerText?.trim().slice(0,150)||'',title:e.title,aria:e.getAttribute('aria-label'),color:getComputedStyle(e).color,bg:getComputedStyle(e).backgroundColor}));
 const colorOnly=elements('.date-overdue').filter(e=>!/overdue|late|⚠/i.test(e.innerText)).map(e=>({path:path(e),text:e.innerText,reason:'Overdue state has only date/weekday text; no late/overdue label or warning icon'}));
 const clipping=elements('*').filter(e=>{
   const cs=getComputedStyle(e);return e.clientWidth>0&&e.scrollWidth>e.clientWidth+1&&['hidden','clip'].includes(cs.overflowX)&&e.innerText?.trim();
 }).map(e=>({path:path(e),width:e.clientWidth,scrollWidth:e.scrollWidth,text:e.innerText.slice(0,100)})).slice(0,100);
 const headers=elements('.section-header').map(e=>({path:path(e),direction:getComputedStyle(e).flexDirection,text:e.innerText.slice(0,180),width:e.getBoundingClientRect().width}));
 const inaccessible=elements('.lot-link,tr.expandable,.order-row,.search-item,.er-product-option,.product-lot-row,.collapsible-header').filter(e=>e.tabIndex<0).map(e=>({path:path(e),text:e.innerText?.slice(0,90)}));
 const disabled=elements('button:disabled,input:disabled,select:disabled').map(e=>({path:path(e),label:e.innerText||'',opacity:getComputedStyle(e).opacity}));
 const stripes=elements('.inv-table tbody').map(e=>({rows:e.rows.length,even:[...e.querySelectorAll('tr:nth-child(even)')].slice(0,3).map(r=>({bg:getComputedStyle(r).backgroundColor,cells:[...r.cells].slice(0,2).map(c=>getComputedStyle(c).backgroundColor)}))}));
 const d=document.documentElement;
 return {inputs,wrap,statuses,colorOnly,clipping,headers,inaccessible,disabled,stripes,overflow:Math.max(0,d.scrollWidth-d.clientWidth),regionText:root.innerText?.slice(0,250)||''};
}
export async function keyboard(p,region,{isForm=false}={}) {
 await p.evaluate(()=>document.querySelectorAll('[data-audit-key]').forEach(e=>e.removeAttribute('data-audit-key')));
 const loc=p.locator(region||'body').first();
 const meta=await loc.evaluate(root=>[...root.querySelectorAll('input:not([type=hidden]),textarea,select,button,a[href],[tabindex]')].filter(e=>e.checkVisibility?.()!==false&&e.getClientRects().length&&!e.disabled).map((e,i)=>{e.setAttribute('data-audit-key',String(i));return {key:String(i),id:e.id,tag:e.tagName,type:e.type,tab:e.tabIndex};}));
 if(!meta.length)return {status:'NA',sequence:[],enter:[]};
 const fields=meta.filter(x=>['INPUT','TEXTAREA','SELECT'].includes(x.tag));
 if(!fields.length&&!isForm)return {status:'NA',sequence:[],enter:[]};
 const first=meta.find(x=>x.tab>=0);if(!first)return {status:'FAIL',reason:'No tabbable field/control',sequence:[],enter:[]};
 await p.locator(`[data-audit-key="${first.key}"]`).focus();
 const sequence=[];
 for(let i=0;i<Math.min(meta.length+2,150);i++){
   sequence.push(await p.evaluate(sel=>{const e=document.activeElement;return {id:e.id,tag:e.tagName,key:e.getAttribute('data-audit-key'),inside:!!document.querySelector(sel||'body')?.contains(e)};},region));
   await p.keyboard.press('Tab');
 }
 const seen=new Set(sequence.filter(x=>x.inside).map(x=>x.key));
 const missing=fields.filter(x=>x.tab>=0&&!seen.has(x.key));
 const modal=await loc.evaluate(e=>/modal|overlay/.test(e.id));
 const escaped=modal&&sequence.some(x=>!x.inside);
 const enter=[];
 // No Enter on action buttons, selects, checkboxes, file inputs, or radio controls.
 // Capture-phase guards suppress any implicit submission before app handlers.
 for(const field of fields.filter(x=>['text','search','number','date','email','url','textarea'].includes(x.type))){
   const el=p.locator(`[data-audit-key="${field.key}"]`);if(!(await el.isVisible()))continue;
   await el.focus();const before=await el.inputValue();
   const result=await p.evaluate(()=>({guards:window.__auditGuard.length,text:document.body.innerText.length}));
   await el.press('Enter');await p.waitForTimeout(30);
   const after=await p.evaluate(()=>({guards:window.__auditGuard.length,text:document.body.innerText.length}));
   const value=await el.inputValue().catch(()=>before);
   enter.push({id:field.id,type:field.type,guarded:after.guards>result.guards,textChanged:after.text!==result.text,newline:value!==before});
   // Restore textarea's local draft after testing newline behavior, without a change event.
   if(value!==before)await el.evaluate((e,v)=>{e.value=v;},before);
 }
 return {status:missing.length||escaped?'FAIL':'PASS',missing,escaped,sequence,enter,scope:'Tab reachability in rendered state; business sequence and committing completion are not certified'};
}
