import fs from 'node:fs/promises';
import path from 'node:path';
import {chromium} from 'playwright';
import {protect,ORIGIN} from './safety.mjs';
import {CHECK_SOURCE} from '../lib/checks.mjs';
const out=path.resolve(import.meta.dirname,'../../../docs/design/audit/live-evidence');
const result={started:new Date().toISOString(),timezones:[],forms:[],drag:null,find:null};
const b=await chromium.launch();
for(const timezoneId of ['America/New_York','Pacific/Kiritimati']){
 const c=await b.newContext({viewport:{width:1280,height:800},timezoneId,serviceWorkers:'block'});const requests=[];await protect(c,requests);const p=await c.newPage();p.on('dialog',d=>d.dismiss());
 await p.goto(ORIGIN,{waitUntil:'domcontentloaded'});await p.waitForFunction(()=>document.querySelector('#last-refreshed')?.textContent.trim(),null,{timeout:40000});await p.waitForTimeout(4000);
 await p.locator('.tab[data-tab=orders]').click();await p.locator('.order-row').first().waitFor({timeout:15000});
 const dates=await p.locator('.order-row').evaluateAll(es=>es.map(e=>({id:e.dataset.orderId,date:e.querySelector('.ship-by-cell')?.innerText})));
 const shot=`supplement-${timezoneId.replace('/','-')}.png`;await p.screenshot({path:path.join(out,shot)});
 result.timezones.push({timezoneId,actual:await p.evaluate(()=>Intl.DateTimeFormat().resolvedOptions().timeZone),dates,screenshot:shot});
 if(timezoneId==='America/New_York'){
  for(const [tab,button,modal,cancel,field,value] of [['notes','#notes-add-btn','#note-modal-overlay','#note-cancel-btn','#note-title','Visual audit draft — not saved'],['expected','#er-new-btn','#er-modal-overlay','#er-cancel-btn','#er-qty','12.5'],['supplies','#supply-request-btn','#supply-request-modal-overlay','#supply-request-cancel-btn',null,null]]){
   await p.locator(`.tab[data-tab=${tab}]`).click();await p.locator(button).click();await p.locator(modal).waitFor({state:'visible'});await p.addScriptTag({content:CHECK_SOURCE});
   await p.evaluate(()=>document.documentElement.dataset.theme='dark');await p.waitForTimeout(400);
   const contrast=await p.evaluate(sel=>window.__FL_AUDIT.contrast(sel),modal);
   let enter=null,rotation=null;
   if(field){await p.fill(field,value);const before=await p.locator(modal).innerText();const guards=await p.evaluate(()=>window.__auditGuard.length);await p.locator(field).press('Enter');await p.waitForTimeout(200);enter={field,modalStillOpen:await p.locator(modal).isVisible(),textChanged:before!==await p.locator(modal).innerText(),guarded:await p.evaluate(n=>window.__auditGuard.length>n,guards)};
    await p.setViewportSize({width:390,height:844});const portrait=await p.inputValue(field);await p.setViewportSize({width:844,height:390});rotation={portrait,landscape:await p.inputValue(field),preserved:portrait===await p.inputValue(field)};await p.setViewportSize({width:1280,height:800});}
   const screenshot=`supplement-${tab}-dark.png`;await p.screenshot({path:path.join(out,screenshot)});result.forms.push({tab,modal,contrast,enter,rotation,screenshot});await p.locator(cancel).click();
  }
  await p.locator('.tab[data-tab=operations]').click();const h=p.locator('#finished-goods-panels .collapsible-header').first();if(!(await h.evaluate(e=>e.classList.contains('expanded'))))await h.click();await p.locator('#section-finished-goods tr.expandable').first().click();
  const link=p.locator('#section-finished-goods .lot-link').first();await link.scrollIntoViewIfNeeded();await p.evaluate(()=>{window.__dragClicks=0;document.addEventListener('click',e=>{if(e.target.closest('.lot-link'))window.__dragClicks++;},true);});
  const box=await link.boundingBox();if(box){await p.mouse.move(box.x+2,box.y+box.height/2);await p.mouse.down();await p.mouse.move(box.x+Math.max(15,box.width-2),box.y+box.height/2,{steps:12});await p.mouse.up();await p.waitForTimeout(400);result.drag=await p.evaluate(()=>({clicks:window.__dragClicks,selection:getSelection().toString(),panelOpen:!document.querySelector('#lot-panel-overlay').classList.contains('hidden')}));result.drag.screenshot='supplement-drag.png';await p.screenshot({path:path.join(out,result.drag.screenshot)});}
 }
 result.timezones.at(-1).requests=requests;
 result.timezones.at(-1).guards=await p.evaluate(()=>window.__auditGuard);
 await c.close();
}
const a=result.timezones[0].dates,bdates=result.timezones[1].dates;
result.dateDifferences=a.flatMap(x=>{const y=bdates.find(y=>x.id===y.id);return y&&x.date!==y.date?[{id:x.id,newYork:x.date,kiritimati:y.date}]:[];});
result.comparedDates=a.filter(x=>bdates.some(y=>y.id===x.id)).length;
result.finished=new Date().toISOString();await b.close();await fs.writeFile(path.join(out,'supplement.json'),JSON.stringify(result,null,2));console.log({dateDifferences:result.dateDifferences.length,compared:result.comparedDates,forms:result.forms.map(x=>({tab:x.tab,enter:x.enter,rotation:x.rotation})),drag:result.drag});
