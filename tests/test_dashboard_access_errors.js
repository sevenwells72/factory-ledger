const {test}=require('node:test');
const assert=require('node:assert/strict');
const path=require('node:path');
const runtime=require(path.join(process.env.FL_KEY_TEST_ROOT||path.resolve(__dirname,'..'),'dashboard/dashboard-runtime.js'));
const message='Access unavailable — refresh the page or contact the administrator';
test('O-7 access message: every failed first key read is plain and a successful retry is usable',async()=>{
 for(const failure of [Error('private network detail'),{ok:false,status:503},{ok:false,status:403},{ok:true,json:async()=>({})},{ok:true,json:async()=>{throw SyntaxError('private JSON detail');}}]){
  let answer=failure;const key=runtime.keyClient(async()=>{if(answer instanceof Error)throw answer;return answer;});
  await assert.rejects(key('https://fixture.invalid'),error=>error.message===message&&error.accessUnavailable===true);
  answer={ok:true,json:async()=>({dashboard_key:'fixture-recovered'})};
  assert.equal(await key('https://fixture.invalid'),'fixture-recovered');
 }
});
test('O-7 read refusals: keyed GET and HEAD 401/403 use one message without consuming server details; writes stay unchanged',()=>{
 for(const status of [401,403])for(const method of ['GET','get','HEAD'])for(const headers of [{'X-API-Key':'fixture'},new Headers({'x-api-key':'fixture'})]){
  const response={status,text(){throw Error('must not inspect private response');}};
  assert.throws(()=>runtime.checkReadAccess(response,{method,headers}),error=>error.message===message&&error.accessUnavailable===true);
  assert.throws(()=>runtime.checkReadAccess(response,{method,headers},'es'),{message:'Acceso no disponible — actualiza la página o contacta al administrador'});
  assert.equal(runtime.checkReadAccess(response,{method:'POST',headers}),response);
  assert.equal(runtime.checkReadAccess(response,{method}),response);
 }
 for(const status of [200,404,500]){const response={status};assert.equal(runtime.checkReadAccess(response,{headers:{'X-API-Key':'fixture'}}),response);}
});
test('O-7 access recovery: invalidate the refused cached key so Retry fetches the rotated value immediately',async()=>{
 let value='fixture-old',calls=0;const base='https://fixture.invalid';
 const key=runtime.keyClient(async()=>{calls++;return {ok:true,json:async()=>({dashboard_key:value})};});
 assert.equal(await key(base),'fixture-old');value='fixture-new';assert.equal(await key(base),'fixture-old');
 key.invalidate(base);assert.equal(await key(base),'fixture-new');assert.equal(calls,2);
});
test('O-7 custom API recovery: a refusal invalidates the matching base rather than the default environment',()=>{
 const bases=[],previous=runtime.invalidateKey;
 runtime.invalidateKey=base=>bases.push(base);
 try{assert.throws(()=>runtime.checkReadAccess({status:403},{headers:{'X-API-Key':'fixture'}},'en','https://custom-fixture.invalid'),{message});assert.deepEqual(bases,['https://custom-fixture.invalid']);}
 finally{if(previous)runtime.invalidateKey=previous;else delete runtime.invalidateKey;}
});
