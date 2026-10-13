const test = require('node:test');
const assert = require('node:assert/strict');
const { create, STORE } = require('../dashboard/key-session.js');
function harness({ enabled = true, saved = '', failure } = {}) {
  const data = new Map(saved ? [[STORE,saved]] : []), calls = [];
  const storage = { getItem:k=>data.get(k), setItem:(k,v)=>data.set(k,v), removeItem:k=>data.delete(k) };
  let fail = failure;
  const fetch = async (url, options) => {
    calls.push({url,options});
    let result, status=200;
    if(url.endsWith('/config')) result={key_sessions_enabled:enabled};
    else if(url.endsWith('/whoami')) result={actor:{id:4,name:'Arturo',role:'floor'},key_kind:'session'};
    else if(url.endsWith('/logout')) result={signed_out:true};
    else result={session_token:'fls_fixture-token',key_kind:'session',actor:{id:4,name:'Arturo',role:'floor'}};
    if(fail && !url.endsWith('/config')) { result={detail:{error_code:fail,message:'Session unavailable'}}; status=fail==='SESSION_EXPIRED'?401:503; }
    return {ok:status===200,status,json:async()=>result};
  };
  const client=create({base:'https://staging.example',fetch,storage});
  return {client,data,calls,setFailure:code=>fail=code};
}
test('exchange sends personal key only once and persists only token',async()=>{
  const h=harness();await h.client.init();await h.client.signIn('personal-test-key');
  assert.equal(h.data.get(STORE),'fls_fixture-token');
  assert.equal(JSON.stringify([...h.data]).includes('personal-test-key'),false);
  assert.equal(h.calls[1].options.body,JSON.stringify({actor_key:'personal-test-key'}));
  const headers=await h.client.headers({'X-API-Key':'shared'});
  assert.equal(headers.get('X-API-Key'),'fls_fixture-token');
  assert.equal(h.client.actor.name,'Arturo');
});
test('reload restores the token and resolves the person',async()=>{
  const h=harness({saved:'fls_existing'});await h.client.init();
  assert.equal(h.calls[1].options.headers['X-API-Key'],'fls_existing');
  assert.equal(h.client.actor.name,'Arturo');
});
test('logout reaches FL before forgetting token',async()=>{
  const h=harness();await h.client.init();await h.client.signIn('personal');await h.client.signOut();
  assert.ok(h.calls.at(-1).url.endsWith('/auth/session/logout'));
  assert.equal(h.calls.at(-1).options.headers['X-API-Key'],'fls_fixture-token');
  assert.equal(h.data.size,0);assert.equal(h.client.actor,null);
});
test('failed logout keeps identity so hand-over cannot look complete',async()=>{
  const h=harness();await h.client.init();await h.client.signIn('personal');h.setFailure('UNAVAILABLE');
  await assert.rejects(h.client.signOut());assert.equal(h.client.actor.name,'Arturo');assert.equal(h.data.size,1);
});
test('expired token clears custody and requires fresh personal sign-in',async()=>{
  const h=harness({saved:'fls_existing',failure:'SESSION_EXPIRED'});await h.client.init();
  assert.equal(h.data.size,0);assert.equal(h.client.actor,null);await assert.rejects(h.client.headers());
});
test('flag off retains no token and performs no exchange',async()=>{
  const h=harness({enabled:false,saved:'fls_existing'});assert.equal(await h.client.init(),false);
  assert.equal(h.calls.length,1);assert.equal(h.data.size,0);await assert.rejects(h.client.signIn('personal'));
});
test('configuration failure is surfaced without raw-key fallback',async()=>{
  const h=harness(); const client=create({base:'https://staging.example',storage:{removeItem(){}},fetch:async()=>{throw Error('offline')}});
  await assert.rejects(client.init());await assert.rejects(client.signIn('personal'));
});
