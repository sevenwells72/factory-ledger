const test=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const path=require('node:path');const vm=require('node:vm');
// The tool bar on every ledger page: Assistant only with personal sign-in on; Shift Summary always (its read-only view needs no sign-in).
test('ledger tool bar links the FL Assistant behind sign-in and the Shift Summary always',()=>{
 const made=[];const element=tag=>{const n={tag,children:[],dataset:{},append(...c){n.children.push(...c);},setAttribute(){}};made.push(n);return n;};
 const host=element('header');const document={currentScript:{src:'https://cns-factory-ledger.netlify.app/receipt-shell.js?v=6'},readyState:'complete',createElement:element,querySelector:s=>s==='header, .header, .topbar'?host:null};
 vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../dashboard/receipt-shell.js'),'utf8'),{document,window:{},URL,location:{}});
 const links=host.children[0].children.filter(n=>n.tag==='a');
 const byText=Object.fromEntries(links.map(a=>[a.textContent,a]));
 assert.equal(byText['FL Assistant / Asistente FL'].href,'https://cns-factory-ledger.netlify.app/fl-assistant.html');
 assert.equal(byText['FL Assistant / Asistente FL'].dataset.flWriteUi,'');
 assert.equal(byText['Shift summary / Resumen del turno'].href,'https://cns-factory-ledger.netlify.app/shift-summary.html');
 assert.equal('flWriteUi' in byText['Shift summary / Resumen del turno'].dataset,false);
});
test('every ledger page tool bar starts with an always-visible Dashboard link home',()=>{
 const made=[];const element=tag=>{const n={tag,children:[],dataset:{},append(...c){n.children.push(...c);},setAttribute(){}};made.push(n);return n;};
 const host=element('header');const document={currentScript:{src:'https://cns-factory-ledger.netlify.app/receipt-shell.js?v=10'},readyState:'complete',createElement:element,querySelector:s=>s==='header, .header, .topbar'?host:null};
 vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../dashboard/receipt-shell.js'),'utf8'),{document,window:{},URL,location:{}});
 const first=host.children[0].children[0];
 assert.equal(first.tag,'a');assert.equal(first.textContent,'Dashboard / Panel');
 assert.equal(first.href,'https://cns-factory-ledger.netlify.app/index.html');
 assert.equal(first.dataset.flDashboard,'');assert.equal('flWriteUi' in first.dataset,false);
});
test('brand links go home and Today\'s entries uses a typographic apostrophe Netlify cannot mangle',()=>{
 for(const page of ['entry.html','office.html','exceptions.html','receipt.html']){
  const html=fs.readFileSync(path.join(__dirname,'../dashboard',page),'utf8');
  assert.match(html,/<a class="ledger-brand" href="index.html"/,page);
  assert.doesNotMatch(html,/data-en="[^"]*'[^"]*"/,page+': no straight apostrophe inside data-en');
 }
 for(const page of ['sankey.html','process-flow.html','traceability.html','runs.html'])assert.match(fs.readFileSync(path.join(__dirname,'../dashboard',page),'utf8'),/<a class="site-nav-brand" href="\/"/,page);
 const assistant=fs.readFileSync(path.join(__dirname,'../dashboard/fl-assistant.html'),'utf8');
 assert.match(assistant,/<a class="brand" href="index.html"/);assert.match(assistant,/<div class="header-actions"><a class="header-home nav-button" href="index.html" data-fl-dashboard/);
});
test('recorded screens add one Back to dashboard after the page\'s own buttons',()=>{
 const ticket=fs.readFileSync(path.join(__dirname,'../dashboard/ledger-ticket.js'),'utf8');
 assert.equal((ticket.match(/onRecorded\([^;]*\);ui\.dashboardLink\(host\)/g)||[]).length,2,'both recorded paths, after onRecorded');
 assert.match(ticket,/ui\.t\('Back to dashboard','Volver al panel'\)/);
 assert.match(fs.readFileSync(path.join(__dirname,'../dashboard/exceptions.js'),'utf8'),/function showOutcome[\s\S]{0,400}ui\.dashboardLink\(home\)/);
});
test('mobile More menu adds Shift Summary and a sign-in-gated FL Assistant next to the current page',()=>{
 const source=fs.readFileSync(path.join(__dirname,'../dashboard/shell.js'),'utf8');
 assert.match(source,/\['shift-summary\.html','Shift Summary',false\],\['fl-assistant\.html','FL Assistant',true\]/);
 assert.match(source,/a\.href=new URL\(url,location\.href\)\.href/);
 assert.match(source,/if\(signedIn\)a\.dataset\.flWriteUi=''/);
});
test('order detail Packing slip button is sign-in gated, hidden on cancelled orders, and never puts a key in a URL',()=>{
 const source=fs.readFileSync(path.join(__dirname,'../dashboard/dashboard.js'),'utf8');
 assert.match(source,/if \(data\.state !== 'cancelled'\) html \+= `<button type="button" data-fl-write-ui class="btn-secondary so-packing-slip-btn"/);
 const fn=source.slice(source.indexOf('async function openPackingSlip'),source.indexOf('function bindOrderDetailActions'));
 assert.match(fn,/headers: \{ 'X-API-Key': await FLDashboard\.key\(\) \}/);
 assert.doesNotMatch(fn,/[?&]key=/);
 assert.match(fn,/URL\.createObjectURL/);
});
