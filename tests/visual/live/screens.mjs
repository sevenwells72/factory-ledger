import { SCREENS as INVENTORY, PAGE_URLS } from '../lib/screens.mjs';
export { PAGE_URLS };
export const SCREENS = [...INVENTORY.map(({id,name,page,region,print})=>({id,name,page,region:({'S-69':'#tooltip','S-77':'#fgbody','S-78':'#wipbody','S-79':'#catbody','S-82':'#leftpanel details:last-of-type'})[id]||region,print})), {id:'S-92',name:'Needs Attention strip (post-inventory; IMP-068/071)',page:'index',region:'.attention-strip-section'}];
export const gaps = {
 'S-23':'Native delete confirmation requires Delete; excluded from allowed navigation.',
 'S-35':'Status confirmation requires saving a status change; prohibited.',
 'S-39':'Shipping preview requires POST; its discovery requests were blocked before transmission. Not invoked under the GET-only policy.',
 'S-38':'Reservation release confirmation requires Release; excluded from allowed navigation.',
 'S-55':'No naturally ambiguous lot identified; no synthetic HTTP 409 or fixture substitution.',
 'S-59':'No live failure/loading banner; no synthetic API failures.',
 'S-62':'Error/stale state requires live occurrence; footer alone does not establish that state.',
 'S-75':'Delta requires setting a baseline, a scheduler state write.',
 'S-81':'Import-result message requires importing a file; not read-only.',
 'S-85':'Opening schedule panel via Copy day also writes the clipboard; not read-only.',
 'S-89':'Validation alert requires submitting Add; prohibited.'
};
const pause = p=>p.waitForTimeout(450);
async function click(p,s){const l=p.locator(s).first();await l.waitFor({state:'visible',timeout:7000});await l.click({timeout:7000});await pause(p);}
async function optional(p,s){if(await p.locator(s).first().isVisible().catch(()=>false))await click(p,s);}
async function tab(p,t){await click(p,`.tab[data-tab="${t}"]`);if(t==='orders'){await optional(p,'#order-back-btn');await p.fill('#orders-customer-search','');await p.selectOption('#orders-status-filter','open');}if(t==='expected')await p.fill('#er-text-filter','');}
async function expand(p,s){const l=p.locator(s).first();if(await l.isVisible() && !(await l.evaluate(e=>e.classList.contains('expanded'))))await click(p,s);}
async function order(p,locked=false){
 await tab(p,'orders');
 if(locked)await p.selectOption('#orders-status-filter','all');
 await p.locator('.order-row').first().waitFor({state:'visible',timeout:12000});
 const rows=p.locator('.order-row'); let chosen=null;
 for(let i=0;i<await rows.count();i++){
   const badge=rows.nth(i).locator('.so-badge').first();const text=await badge.count()?await badge.innerText():await rows.nth(i).innerText();
   if(locked ? !/^(new|confirmed)$/i.test(text.trim()) : /^(new|confirmed)$/i.test(text.trim())){chosen=rows.nth(i);break;}
 }
 if(!chosen) throw Error(locked?'No live edit-locked order found':'No editable live order found');
 await chosen.click();await p.locator('.order-detail-header').waitFor({timeout:12000});await pause(p);
}
async function supplies(p){await tab(p,'supplies');await p.locator('#supplies-inventory-table').waitFor({state:'visible'});await pause(p);}
async function fg(p){await expand(p,'#finished-goods-panels .collapsible-header');}
async function trace(p){
 await p.locator('#recentLots .lot-pill').first().waitFor({timeout:15000});
 await click(p,'#recentLots .lot-pill');
 await click(p,'#traceBtn');
 await p.locator('#graphSvg g[cursor="pointer"]').first().waitFor({timeout:15000});await pause(p);
}
export async function setup(p,s) {
 const n=Number(s.id.slice(2));
 if(s.page==='index'){
   await p.waitForFunction(()=>document.querySelector('#last-refreshed')?.textContent.trim(),null,{timeout:25000});
 }
 if(n===6&&await p.locator('.today-tile-error').isVisible().catch(()=>false))throw Error('Live Today So Far returned an error; data tile unavailable');
 if(n===7)await p.locator('.today-tile-error').waitFor({state:'visible',timeout:1000});
 if(n===4){await p.fill('#global-search','granola');await p.locator('#search-results:not(.hidden)').waitFor();}
 if(n===9) await click(p,'.day-card-trigger');
 if(n===10||n===11||n===54){await fg(p);if(n!==10)await click(p,'#section-finished-goods tr.expandable');if(n===54){await click(p,'#section-finished-goods .lot-link');await p.locator('#lot-panel-overlay:not(.hidden)').waitFor();}}
 if(n===12)await optional(p,'#section-batches tr.expandable');
 if(n===13){await expand(p,'#ingredients-panels .collapsible-header');await optional(p,'#section-ingredients tr.expandable');}
 if(n===14||n===15){await tab(p,'recent');await p.locator(n===14?'.recent-entry-card':'.recent-entries-error,.recent-entries-empty').first().waitFor({state:'visible',timeout:4000});}
 if(n>=16&&n<=18){await tab(p,'activity');const sec={16:'#section-daily-entries',17:'#section-shipments',18:'#section-receipts'}[n];await expand(p,sec+' .collapsible-header');await optional(p,sec+' tr.expandable');}
 if(n>=19&&n<=23){await tab(p,'notes');if(n===22)await click(p,'#notes-add-btn');if(n===21){for(const category of ['all','note','todo','reminder']){await click(p,`.notes-filter-btn[data-cat="${category}"]`);if(await p.locator('.notes-empty').isVisible().catch(()=>false))break;}if(!(await p.locator('.notes-empty').isVisible().catch(()=>false)))throw Error('Every available notes category contains records; no natural empty state');}}
 if(n>=24&&n<=29){await tab(p,'orders');if(n===26)await p.selectOption('#orders-status-filter','dispatch_queue');if(n===27||n===28){if(!(await p.locator('.order-ready-drawer').first().isVisible().catch(()=>false)))await click(p,'.order-expand-toggle');}if(n===29){await p.fill('#orders-customer-search','zzzz-no-such-customer');await pause(p);}}
 if(n>=30&&n<=40){await order(p,n===34);if(n===32)await click(p,'.order-inventory-toggle');if(n===33)await click(p,'.order-edit-toggle-btn');if(n===39)await click(p,'.order-preview-btn');}
 if(n>=41&&n<=45){await tab(p,'expected');if(n===42)await p.selectOption('#er-status-filter','all');if(n===44){await p.fill('#er-text-filter','zzzz-no-match');await pause(p);}if(n===45)await click(p,'#er-new-btn');}
 if(n>=46&&n<=53){if(n===49||n===50)await supplies(p);else await tab(p,'supplies');if(n===50)await click(p,'.supply-item-row');if(n===53)await click(p,'#supply-request-btn');if(n===51)await p.locator('#supply-requests-show-done').check();if(n===52)await p.locator('#supply-requests-show-done').uncheck();if(n===52 && !(await p.locator('#section-supply-requests').innerText()).match(/no .*requests|no requests/i))throw Error('No live empty request state');}
 if(n===56){await p.fill('#global-search','granola');await click(p,'[data-search-product-id]');await p.locator('#lot-panel-overlay:not(.hidden)').waitFor();}
 if(n===58)await p.locator('#sankey-chart svg').waitFor({timeout:15000});
 if(n===60||n===61)await p.locator('#lines-grid .line-card').first().waitFor({timeout:15000});
 if(n===63||n===64){await p.locator('#recentLots .lot-pill').first().waitFor({timeout:15000});const code=await p.locator('#recentLots .lot-pill').first().innerText();await p.fill('#lotSearch',code.trim().split(/\s/)[0].slice(0,n===63?5:100));await pause(p);}
 if(n>=68&&n<=71){await trace(p);if(n===69)await p.locator('#graphSvg g[cursor="pointer"]').first().hover();if(n===70)await click(p,'#graphSvg g[cursor="pointer"]');}
 if(s.page==='scheduler'){
   if(n===90)await p.reload({waitUntil:'domcontentloaded'});
   await p.locator('#leftpanel').waitFor({timeout:12000});
   for(const summary of await p.locator('#leftpanel details[open] > summary').all())await summary.click();
   const labels={77:'Finished goods on hand',78:'Bulk-bin WIP',79:'Product catalog',82:'How this works'};
   if(labels[n])await p.locator('summary').filter({hasText:labels[n]}).click();
   if(n===84)await optional(p,'.more-toggle');
   if(n===86)await click(p,'td.cell[data-st="bake"]:not(.off)');
   if(n===88)await click(p,'#btn-addorder');
 }
 await pause(p);
 const readySelectors={20:'.note-card',25:'.order-row',26:'.order-row',27:'.order-lines-row:not(.hidden)',28:'.order-ready-drawer',30:'.order-detail-header',31:'.order-detail-table-wrap table',32:'.order-inventory-table',33:'.order-cancel-edit-btn',34:'.order-edit-locked',36:'.allocation-form',37:'.allocation-table-wrap table',39:'.order-preview-card',40:'.order-notes-card',42:'.er-row',43:'.er-edit-btn',49:'.supply-item-row',50:'.supply-lot-detail-row',51:'.supply-requests-table'};
 if(readySelectors[n])await p.locator(readySelectors[n]).first().waitFor({state:'visible',timeout:15000});
 // Assert the requested state instead of treating a visible parent as success.
 const expected={11:'.lot-detail-row:not(.hidden)',14:'.recent-entry-card',22:'#note-modal-overlay:not(.hidden)',27:'.order-lines-row',28:'.order-ready-drawer',33:'.order-edit-cancel-btn',34:'.order-edit-locked',37:'.allocation-table-wrap',43:'.er-edit-btn',45:'#er-modal-overlay:not(.hidden)',50:'.supply-lot-detail-row',53:'#supply-request-modal-overlay:not(.hidden)',54:'#lot-panel-overlay:not(.hidden)',56:'#lot-panel-overlay:not(.hidden)',69:'#tooltip',70:'#detailPanel',86:'#modalbg.show',88:'#ao-id'};
 // Assertions with selectors that changed since the inventory are resolved by the caller's region check.
 if([22,28,45,53,54,56,86,88].includes(n))await p.locator(expected[n]).first().waitFor({state:'visible',timeout:7000});
}
export async function cancel(p,s){
 const map={'S-22':'#note-cancel-btn','S-45':'#er-cancel-btn','S-53':'#supply-request-cancel-btn','S-86':'#pin-cancel'};
 if(map[s.id])await optional(p,map[s.id]);
}
