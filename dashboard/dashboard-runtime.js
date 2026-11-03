/* One API destination and the server's existing cutover switch. No local feature flag. */
(function(root){
  'use strict';
  const PRODUCTION='https://fastapi-production-b73a.up.railway.app';
  const STAGING='https://fastapi-staging-production-dd7b.up.railway.app';
  const PRODUCTION_SITE='cns-factory-ledger.netlify.app';
  function apiBase(location,override){
    const host=(location?.hostname||'').toLowerCase();
    // Preview isolation wins even over an explicit production API override.
    if(host.endsWith('.netlify.app')&&host!==PRODUCTION_SITE)return STAGING;
    if(override&&/^https?:\/\//.test(override))return override.replace(/\/$/,'');
    if(host===PRODUCTION_SITE||!location||location.protocol==='file:')return PRODUCTION;
    return location.origin;
  }
  const enabled=config=>config?.key_sessions_enabled===true;
  const personSelect=config=>config?.person_select_enabled===true;
  function accessMessage(lang='en') {
    return lang==='es' ? 'Acceso no disponible — actualiza la página o contacta al administrador' : 'Access unavailable — refresh the page or contact the administrator';
  }
  function accessError(lang='en') {
    const error=new Error(accessMessage(lang));
    error.accessUnavailable=true;
    error.message_es=accessMessage('es');
    return error;
  }
  // Read refusals describe availability, never permission to make a change.
  // Do not consume the response body or put server details into this error.
  function checkReadAccess(response,options={},lang='en',base) {
    const read=['GET','HEAD'].includes((options.method||'GET').toUpperCase());
    const keyed=new Headers(options.headers||{}).has('X-API-Key');
    if(read&&keyed&&[401,403].includes(response.status)) {
      api.invalidateKey?.(base);
      throw accessError(lang);
    }
    return response;
  }
  function renderAccess(element,retry,lang='en') {
    element.replaceChildren();
    const message=root.document.createElement('span');message.textContent=accessMessage(lang);element.append(message);
    if(retry){const button=root.document.createElement('button');button.type='button';button.textContent=lang==='es'?'Reintentar':'Retry';
      button.onclick=async()=>{button.disabled=true;try{await retry();}finally{button.disabled=false;}};element.append(' ',button);}
  }
  // O-7: the scoped dashboard key comes from the API the page talks to, never a
  // literal, so rotation is a backend env change. Re-read every KEY_TTL_MS so an
  // open tab picks up a rotated key within a minute; a failed re-read keeps the
  // last good value, a failed first read is not cached.
  const KEY_TTL_MS=60000;
  function keyClient(transport,now=()=>Date.now()){
    const cache=new Map();
    function key(base){
      const hit=cache.get(base);
      if(hit&&(hit.pending||now()-hit.at<KEY_TTL_MS))return hit.promise;
      const entry={pending:true,at:0};
      const controller=typeof AbortController==='function'?new AbortController():null,timer=controller&&setTimeout(()=>controller.abort(),15000);
      entry.promise=transport(base+'/auth/dashboard-key',{cache:'no-store',...(controller?{signal:controller.signal}:{})}).then(async response=>{
        if(!response.ok)throw accessError();
        const value=(await response.json())?.dashboard_key;
        if(typeof value!=='string'||!value)throw accessError();
        Object.assign(entry,{pending:false,at:now(),value});return value;
      }).catch(error=>{
        if(hit?.value){Object.assign(entry,{pending:false,at:now(),value:hit.value});return hit.value;}
        cache.delete(base);throw accessError();
      }).finally(()=>timer&&clearTimeout(timer));
      cache.set(base,entry);
      return entry.promise;
    }
    key.invalidate=base=>cache.delete(base);
    return key;
  }
  function route(url,base,location){
    const target=new URL(url,location.href);
    if([PRODUCTION,STAGING].includes(target.origin))return new URL(target.pathname+target.search+target.hash,base).href;
    return target.href;
  }
  // Previews look identical to production, so any page whose API is not
  // production says so on every screen. The label follows the API, not the host.
  function environmentBanner(base){
    return base===PRODUCTION?null:'STAGING — test system. Entries here are not real inventory. / PRUEBAS — sistema de prueba. Lo que registres aquí no es inventario real.';
  }
  function showEnvironmentBanner(document,text){
    if(document.getElementById('fl-environment-banner'))return;
    const bar=document.createElement('div');bar.id='fl-environment-banner';bar.setAttribute('role','note');bar.textContent=text;
    // In normal flow, not sticky: pages already pin their own top bars there.
    bar.style.cssText='position:relative;z-index:1;background:#b3261e;color:#fff;font:600 14px/1.35 system-ui,sans-serif;text-align:center;padding:6px 12px';
    document.body.prepend(bar);
    if(!document.title.startsWith('[STAGING] '))document.title='[STAGING] '+document.title;
  }
  // R8: with selector mode configured, every transport asks this for the name
  // client. If person-select.js is missing or failed, it gets a stand-in that
  // refuses every write — never a personal key, PIN or the shared key instead.
  const SELECT_MISSING=['Name selection did not load. Refresh the page. Nothing was recorded.','La selección de nombre no cargó. Actualiza la página. No se registró nada.'];
  function missingClient(){
    const fail=()=>{const error=new Error(SELECT_MISSING[0]);error.code='PERSON_SELECT_UNAVAILABLE';error.labels=SELECT_MISSING.slice();error.personSelect=true;return error;};
    return {ready:Promise.resolve(),mode:true,person:null,revision:0,broken:true,state:{mode:true,status:'unavailable',problem:'SCRIPT_MISSING',people:[],person:null},
      headers({write=false}={}){if(write)throw fail();return null;},observe(){return null;},open(){},on(){return()=>{};},fail};
  }
  function personClient(root){return root.FLPersonSelect||missingClient();}
  const api={apiBase,enabled,personSelect,personClient,missingClient,route,environmentBanner,accessMessage,accessError,checkReadAccess,renderAccess,keyClient,KEY_TTL_MS,PRODUCTION,STAGING,PRODUCTION_SITE};
  if(typeof module!=='undefined')module.exports=api;
  root.FLDashboard=api;
  if(!root.document)return;
  api.base=apiBase(root.location,root.FL_SESSION_API_BASE);
  const banner=environmentBanner(api.base);
  if(banner&&typeof document.addEventListener==='function'){
    document.documentElement.dataset.flEnvironment='staging';
    if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',()=>showEnvironmentBanner(document,banner),{once:true});
    else showEnvironmentBanner(document,banner);
  }
  const keys=keyClient((url,options)=>root.fetch(url,options));
  api.key=(base,lang)=>keys(base||api.base).catch(()=>{throw accessError(lang);});
  api.invalidateKey=base=>keys.invalidate(base||api.base);
  api.writesEnabled=false;
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000);
  api.config=root.fetch(api.base+'/auth/session/config',{cache:'no-store',signal:controller.signal}).then(async response=>{
    if(!response.ok)throw Error('Session configuration unavailable');
    return response.json();
  }).finally(()=>clearTimeout(timer));
  // 091: name selection on owner-approved browsers. Exact true only; availability
  // problems are shown by person-select.js and block writes (never a key/PIN prompt).
  api.person=()=>personClient(root);
  // R11: this ONE config result is authoritative for every client on the page
  // (key sessions, PIN, name selection). If it cannot be read, the state is
  // UNKNOWN — not "selection off": writes stay blocked, no credential prompt is
  // offered and no old key/PIN session is restored until the page is reloaded.
  api.configUnknown=false;
  api.ready=api.config.then(config=>({on:enabled(config)||personSelect(config),pin:config?.pin_sign_in===true,select:personSelect(config)}),()=>{api.configUnknown=true;return {on:false,pin:false,select:false};}).then(({on,pin,select})=>{
    api.writesEnabled=on;api.personSelect=select;api.pinEnabled=on&&pin;
    document.documentElement.dataset.flConfig=api.configUnknown?'unknown':'ok';
    document.documentElement.dataset.flWriteUi=on?'on':'off';
    document.documentElement.dataset.flPersonSelect=select?'on':'off';
    // Phone PIN links ([data-fl-pin-ui]) show only while the server has PINs on
    // and staff are not choosing names instead.
    document.documentElement.dataset.flPin=api.pinEnabled&&!select?'on':'off';
    return on;
  });
})(typeof window==='undefined'?globalThis:window);
