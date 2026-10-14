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
  // O-7: the scoped dashboard key comes from the API the page talks to, never a
  // literal, so rotation is a backend env change. Re-read every KEY_TTL_MS so an
  // open tab picks up a rotated key within a minute; a failed re-read keeps the
  // last good value, a failed first read is not cached.
  const KEY_TTL_MS=60000;
  function keyClient(transport,now=()=>Date.now()){
    const cache=new Map();
    return function key(base){
      const hit=cache.get(base);
      if(hit&&(hit.pending||now()-hit.at<KEY_TTL_MS))return hit.promise;
      const entry={pending:true,at:0};
      const controller=typeof AbortController==='function'?new AbortController():null,timer=controller&&setTimeout(()=>controller.abort(),15000);
      entry.promise=transport(base+'/auth/dashboard-key',{cache:'no-store',...(controller?{signal:controller.signal}:{})}).then(async response=>{
        if(!response.ok)throw Error('Dashboard key unavailable (HTTP '+response.status+')');
        const value=(await response.json())?.dashboard_key;
        if(typeof value!=='string'||!value)throw Error('Dashboard key unavailable');
        Object.assign(entry,{pending:false,at:now(),value});return value;
      }).catch(error=>{
        if(hit?.value){Object.assign(entry,{pending:false,at:now(),value:hit.value});return hit.value;}
        cache.delete(base);throw error;
      }).finally(()=>timer&&clearTimeout(timer));
      cache.set(base,entry);
      return entry.promise;
    };
  }
  function route(url,base,location){
    const target=new URL(url,location.href);
    if([PRODUCTION,STAGING].includes(target.origin))return new URL(target.pathname+target.search+target.hash,base).href;
    return target.href;
  }
  const api={apiBase,enabled,route,keyClient,KEY_TTL_MS,PRODUCTION,STAGING,PRODUCTION_SITE};
  if(typeof module!=='undefined')module.exports=api;
  root.FLDashboard=api;
  if(!root.document)return;
  api.base=apiBase(root.location,root.FL_SESSION_API_BASE);
  const keys=keyClient((url,options)=>root.fetch(url,options));
  api.key=base=>keys(base||api.base);
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
