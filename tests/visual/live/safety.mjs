export const ORIGIN = 'https://cns-factory-ledger.netlify.app';
let apiQueue = Promise.resolve();
export async function protect(context, log) {
  await context.route('**/*', async route => {
    const req = route.request(), u = new URL(req.url());
    // Never record query strings, headers, bodies, or credentials.
    const item = {method:req.method(), host:u.hostname, path:u.pathname};
    if (!['GET','HEAD','OPTIONS'].includes(req.method()) || /\/(save|post|receive|pack|ship|void|delete|cancel|allocate|release)(\/|$)/i.test(u.pathname)) {
      log.push({...item, blocked:true}); return route.abort('blockedbyclient');
    }
    log.push({...item, blocked:false});
    if(u.hostname==='fastapi-production-b73a.up.railway.app'){
      // Serialize API reads across both engines. Keep the genuine response, including errors.
      const prev=apiQueue;let done;apiQueue=new Promise(r=>done=r);await prev;
      try {const response=await route.fetch({timeout:30000,maxRetries:0});await route.fulfill({response});}
      catch {await route.abort('failed').catch(()=>{});}
      finally {await new Promise(r=>setTimeout(r,100));done();}
    } else await route.continue();
  });
  await context.addInitScript(() => {
    window.__auditGuard = [];
    const add=EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener=function(type,handler,options){if(type==='click'&&this instanceof Element)this.setAttribute('data-audit-click','');return add.call(this,type,handler,options);};
    const record = (kind, target) => window.__auditGuard.push({kind, target});
    const forbidden = /^(save|post|receive|pack|ship|void|delete|release|allocate|pin &|clear pins|set as baseline|load sample|reset|add$|submit|close receipt|cancel receipt)\b/i;
    document.addEventListener('click', e => {
      const el = e.target.closest('button,a,input,[role="button"],[onclick]');
      if (!el) return;
      const label = (el.textContent || el.value || '').trim();
      const blockedId = /^(ao-save|pin-save|pin-clear|btn-baseline|btn-sample|btn-reset|.*save.*|.*submit.*)$/i.test(el.id);
      const rowMutation = el.matches('.order-ready-checkbox,.note-checkbox,.er-close-btn,.er-cancel-btn,[data-incl],[data-excl],[data-del]');
      if (blockedId || rowMutation || forbidden.test(label)) {
        e.preventDefault(); e.stopImmediatePropagation(); record('blocked-click', el.id || label.slice(0,60));
      }
    }, true);
    document.addEventListener('submit', e => {e.preventDefault();e.stopImmediatePropagation();record('blocked-submit',e.target.id);}, true);
    HTMLFormElement.prototype.submit = function(){record('blocked-programmatic-submit',this.id);};
    HTMLFormElement.prototype.requestSubmit = function(){record('blocked-request-submit',this.id);};
    navigator.sendBeacon = () => {record('blocked-beacon','');return false;};
    window.open = () => {record('blocked-window-open','');return null;};
  });
}
