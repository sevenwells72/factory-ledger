import test from 'node:test';import assert from 'node:assert/strict';import {chromium} from 'playwright';import {CHECK_SOURCE} from './check-source.mjs';import {inspect} from './checks.mjs';
test('selects are measured, closed disclosure fields are not counted',async()=>{
 const b=await chromium.launch();const p=await b.newPage();try{
  await p.setContent('<main><button style="width:44px;height:44px">Open</button><select style="width:30px;height:30px"><option>x</option></select><details><summary>Closed</summary><div style="display:block"><input id="hidden-input" style="width:90px;height:18px"></div></details></main>');
  await p.addScriptTag({content:CHECK_SOURCE});const t=await p.evaluate(()=>window.__FL_AUDIT.touchTargets('main'));
  assert(t.worst.some(x=>x.tag==='select'));assert(!t.worst.some(x=>x.path.includes('hidden-input')));
  const d=await p.evaluate(inspect,'main');assert.equal(d.inputs.length,1);assert.equal(d.inputs[0].type,'select-one');
 }finally{await b.close();}
});
