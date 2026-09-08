// These interactions target only a synthetic data: document, never production.
import test from 'node:test';
import assert from 'node:assert/strict';
import {chromium} from 'playwright';
import {protect} from './safety.mjs';
test('read-only guard prevents every forbidden action before page handlers',async()=>{
 const b=await chromium.launch();const c=await b.newContext();const log=[];await protect(c,log);const p=await c.newPage();
 try{
  await p.goto('data:text/html,<body></body>');
  await p.evaluate(()=>{window.actions=[];document.body.innerHTML='<form id="f"><input><button>Save</button></form>'+['Post','Receive','Pack','Ship','Void','Delete','Release','Allocate','Cancel receipt'].map((x,i)=>`<button type="button" id="b${i}">${x}</button>`).join('')+'<button id="notes-add-btn">+ New</button><button id="note-cancel-btn">Cancel</button>';document.addEventListener('click',e=>window.actions.push(e.target.textContent));});
  for(const label of ['Save','Post','Receive','Pack','Ship','Void','Delete','Release','Allocate','Cancel receipt'])await p.getByRole('button',{name:label,exact:true}).click();
  assert.deepEqual(await p.evaluate(()=>window.actions),[]);
  await p.locator('#notes-add-btn').click();await p.locator('#note-cancel-btn').click();
  assert.deepEqual(await p.evaluate(()=>window.actions),['+ New','Cancel']);
  await p.evaluate(()=>{document.querySelector('form').submit();document.querySelector('form').requestSubmit();});
  assert.equal((await p.evaluate(()=>window.__auditGuard)).filter(x=>x.kind.includes('submit')).length,2);
  // Fetch is intercepted before a network request can be sent.
  await p.evaluate(()=>fetch('https://cns-factory-ledger.netlify.app/audit-guard-test',{method:'POST'}).catch(()=>{}));
  assert.equal(log.find(x=>x.method==='POST')?.blocked,true);
 }finally{await c.close();await b.close();}
});
