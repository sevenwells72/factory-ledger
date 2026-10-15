const {test}=require('node:test');const assert=require('node:assert/strict');const fs=require('node:fs');const vm=require('node:vm');const path=require('node:path');
const runtime=require('../dashboard/dashboard-runtime.js');
const loc=host=>({hostname:host,origin:'https://'+host,href:'https://'+host+'/dashboard/index.html',protocol:'https:'});
test('only the canonical Netlify site can select production; previews override unsafe configuration',()=>{
 assert.equal(runtime.apiBase(loc(runtime.PRODUCTION_SITE)),runtime.PRODUCTION);
 for(const host of ['deploy-preview-107--cns-factory-ledger.netlify.app','feat-dash--cns-factory-ledger.netlify.app','another-site.netlify.app'])for(const override of [undefined,runtime.PRODUCTION,'https://other.example'])assert.equal(runtime.apiBase(loc(host),override),runtime.STAGING);
 assert.equal(runtime.apiBase(loc('localhost:8125')),'https://localhost:8125');
 assert.equal(runtime.apiBase(loc('fastapi-staging-production-dd7b.up.railway.app')),runtime.STAGING);
 assert.equal(runtime.apiBase(loc('netlify.app.evil.example')),'https://netlify.app.evil.example');
 assert.equal(runtime.route(runtime.PRODUCTION+'/transactions/history?day=x',runtime.STAGING,loc('preview.netlify.app')),runtime.STAGING+'/transactions/history?day=x');
 assert.equal(runtime.route('https://example.org/file',runtime.STAGING,loc('preview.netlify.app')),'https://example.org/file');
});
test('write UI requires the exact existing boolean, never a local fallback',()=>{
 assert.equal(runtime.enabled({key_sessions_enabled:true}),true);
 for(const value of [undefined,null,false,'true',1,{}])assert.equal(runtime.enabled({key_sessions_enabled:value}),false);
 assert.equal(runtime.enabled({}),false);
});
async function boot(response){let release;const pending=new Promise(r=>release=r),calls=[];const root={location:loc('preview.netlify.app'),FL_SESSION_API_BASE:runtime.PRODUCTION,document:{documentElement:{dataset:{}}},fetch:async url=>{calls.push(url);await pending;if(response instanceof Error)throw response;return response;}};
 vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../dashboard/dashboard-runtime.js'),'utf8'),{window:root,document:root.document,AbortController,URL,setTimeout,clearTimeout});
 assert.equal(root.FLDashboard.writesEnabled,false);assert.equal(root.document.documentElement.dataset.flWriteUi,undefined);release();await root.FLDashboard.ready;return {root,calls};}
test('delayed, failed and malformed config cannot expose write UI; true enables it',async()=>{
 for(const response of [new Error('offline'),{ok:false},{ok:true,json:async()=>({})},{ok:true,json:async()=>({key_sessions_enabled:'true'})}]){const {root,calls}=await boot(response);assert.equal(root.FLDashboard.writesEnabled,false);assert.equal(root.document.documentElement.dataset.flWriteUi,'off');assert.deepEqual(calls,[runtime.STAGING+'/auth/session/config']);}
 const {root}=await boot({ok:true,json:async()=>({key_sessions_enabled:true})});assert.equal(root.FLDashboard.writesEnabled,true);assert.equal(root.document.documentElement.dataset.flWriteUi,'on');
});
test('every receipt-enabled page loads routing before other scripts and hides new UI initially',()=>{
 const directory=path.join(__dirname,'../dashboard');const files=[...fs.readdirSync(directory).filter(n=>n.endsWith('.html')).map(n=>path.join(directory,n)),path.join(directory,'scheduler/seven-wells-production-board.html')];
 for(const file of files){const source=fs.readFileSync(file,'utf8');if(!source.includes('receipt-shell.js'))continue;assert.match(source,/<head><script src="(?:\.\.\/)?dashboard-runtime\.js/,'routing first: '+file);}
 for(const name of ['entry','exceptions','receipt']){const source=fs.readFileSync(path.join(directory,name+'.html'),'utf8');assert.match(source,/<form hidden data-fl-write-ui id="sign-in-form"/);if(name!=='receipt')assert.match(source,/<main data-fl-write-ui>/);}
 assert.match(fs.readFileSync(path.join(directory,'ledger-forms.css'),'utf8'),/:root:not\(\[data-fl-write-ui="on"\]\) \[data-fl-write-ui\]\{display:none!important\}/);
});
test('shift-summary preview routing also refuses its production query override',()=>{
 const summary=require('../dashboard/shift-summary.js');assert.equal(summary.apiBase(loc('preview.netlify.app'),'?api='+runtime.PRODUCTION),runtime.STAGING);
});
test('O-7 key client: one no-store read per base, refreshed after the TTL, last good value survives a failed refresh',async()=>{
 let clock=0,answer={ok:true,json:async()=>({dashboard_key:'k1'})};const calls=[];
 const key=runtime.keyClient(async(url,options)=>{calls.push([url,options.cache]);if(answer instanceof Error)throw answer;return answer;},()=>clock);
 const [a,b]=await Promise.all([key(runtime.STAGING),key(runtime.STAGING)]);
 assert.deepEqual([a,b],['k1','k1']);assert.deepEqual(calls,[[runtime.STAGING+'/auth/dashboard-key','no-store']]);
 assert.equal(await key(runtime.PRODUCTION),'k1');assert.equal(calls.length,2,'each API base has its own key');
 clock=runtime.KEY_TTL_MS-1;await key(runtime.STAGING);assert.equal(calls.length,2);
 clock=runtime.KEY_TTL_MS;answer={ok:true,json:async()=>({dashboard_key:'k2'})};
 assert.equal(await key(runtime.STAGING),'k2','a rotated key reaches an open tab after the TTL');
 clock*=3;answer=new Error('offline');assert.equal(await key(runtime.STAGING),'k2','a failed refresh keeps the last good key');
});
test('O-7 key client: a failed or malformed first read rejects and is retried, never cached',async()=>{
 for(const bad of [new Error('offline'),{ok:false,status:503},{ok:true,json:async()=>({})},{ok:true,json:async()=>({dashboard_key:''})},{ok:true,json:async()=>({dashboard_key:42})}]){
  let answer=bad,calls=0;const key=runtime.keyClient(async()=>{calls++;if(answer instanceof Error)throw answer;return answer;});
  await assert.rejects(key(runtime.STAGING));answer={ok:true,json:async()=>({dashboard_key:'k'})};
  assert.equal(await key(runtime.STAGING),'k');assert.equal(calls,2);
 }
});
test('O-7 pages read the scoped key at runtime; no served file carries a key literal',()=>{
 const directory=path.join(__dirname,'../dashboard');
 for(const name of ['dashboard.js','mini-calendar.js','process-flow.html','sankey.html','traceability.html','history.js','runs.js','ledger-ui.js','shift-summary.js']){
  const source=fs.readFileSync(path.join(directory,name),'utf8');
  assert.match(source,/FLDashboard\.key\(/,name);assert.doesNotMatch(source,/SALES_API_KEY|readKey|DASHBOARD_KEY|const API_KEY/,name);
 }
});
