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
  function route(url,base,location){
    const target=new URL(url,location.href);
    if([PRODUCTION,STAGING].includes(target.origin))return new URL(target.pathname+target.search+target.hash,base).href;
    return target.href;
  }
  const api={apiBase,enabled,route,PRODUCTION,STAGING,PRODUCTION_SITE};
  if(typeof module!=='undefined')module.exports=api;
  root.FLDashboard=api;
  if(!root.document)return;
  api.base=apiBase(root.location,root.FL_SESSION_API_BASE);
  api.writesEnabled=false;
  const controller=new AbortController(),timer=setTimeout(()=>controller.abort(),15000);
  api.config=root.fetch(api.base+'/auth/session/config',{cache:'no-store',signal:controller.signal}).then(async response=>{
    if(!response.ok)throw Error('Session configuration unavailable');
    return response.json();
  }).finally(()=>clearTimeout(timer));
  api.ready=api.config.then(config=>enabled(config),()=>false).then(on=>{
    api.writesEnabled=on;
    document.documentElement.dataset.flWriteUi=on?'on':'off';
    return on;
  });
})(typeof window==='undefined'?globalThis:window);
