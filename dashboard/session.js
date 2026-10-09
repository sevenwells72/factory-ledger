/* A11: one browser credential, an opaque FL session. No PIN/key persistence.
   Used by every dashboard request and exported for F1's api() adapter. */
(function (root) {
  'use strict';
  const IDLE_MS = 10 * 60 * 1000;
  const STORAGE = 'fl-session-v1';
  const production = 'https://fastapi-production-b73a.up.railway.app';
  const base = (location.hostname.endsWith('.up.railway.app') || ['localhost', '127.0.0.1'].includes(location.hostname)) ? location.origin : production;
  const transport = root.FL.fetchWithTimeout;
  const scriptURL = document.currentScript.src;
  let clientFetch = transport, clientSession = null;
  const ready = transport(base + '/auth/config', {cache:'no-store'}).then(async response => {
    if (!response.ok) throw new Error('Dashboard configuration is unavailable. Refresh to retry.');
    const config = await response.json();
    if (config.pin_login_enabled === true) {
      activate();
    } else if (config.pin_login_enabled === false && typeof config.dashboard_key === 'string') {
      clientFetch = (url, options = {}) => {
        const parsed = new URL(url, location.href);
        if (parsed.origin !== production && parsed.origin !== base) return transport(url, options);
        const headers = new Headers(options.headers || {});
        if (headers.get('X-FL-Client') === 'dashboard') {
          headers.delete('X-FL-Client');
          headers.set('X-API-Key', config.dashboard_key);
        }
        return transport(url, {...options, headers});
      };
    } else throw new Error('Dashboard configuration is unavailable. Refresh to retry.');
    return config.pin_login_enabled;
  });
  // A rejected config must not silently downgrade an enabled deployment.
  ready.catch(() => {});
  root.FL.fetchWithTimeout = async (url, options) => { await ready; return clientFetch(url, options); };
  root.FLSession = {
    base, ready, fetch:root.FL.fetchWithTimeout,
    requireSession:async () => { await ready; return clientSession ? clientSession.requireSession() : null; },
    signOut:async () => { await ready; if (clientSession) return clientSession.signOut(); },
    ownerPIN:async () => { await ready; if (clientSession) return clientSession.ownerPIN(); },
    actor:() => clientSession?.actor() || null, idle:() => clientSession?.idle() ?? true
  };
  function activate() {
  let state = null, lastActivity = Date.now(), lastSentActivity = 0, pendingLogin = null;
  try {
    state = JSON.parse(sessionStorage.getItem(STORAGE) || 'null');
    if (!state?.session_token?.startsWith('fls_') || !state.actor?.id || Date.now() - state.lastActivity >= IDLE_MS) state = null;
    if (state) lastActivity = state.lastActivity;
    else sessionStorage.removeItem(STORAGE);
    // Retire any previously persisted browser actor keys; keep preferences.
    ['fl-actor-key', 'fl-assistant-key', 'actor_key', 'actorKey'].forEach(k => localStorage.removeItem(k));
  } catch (_) { state = null; }
  function persist() {
    try {
      if (state) sessionStorage.setItem(STORAGE, JSON.stringify({...state, lastActivity}));
      else sessionStorage.removeItem(STORAGE);
    } catch (_) { /* memory-only sessions still work */ }
  }
  function changed() {
    root.dispatchEvent(new CustomEvent('fl-session-change', {detail: state?.actor || null}));
    const label = document.getElementById('fl-person');
    if (label) label.textContent = state ? 'Recording as ' + state.actor.name : 'Signed out';
  }
  function clear() { state = null; persist(); changed(); }
  async function signOut() {
    const token = state?.session_token;
    clear();
    if (token) {
      try { await transport(base + '/auth/session', {method:'DELETE', headers:{'X-API-Key':token}, cache:'no-store'}); }
      catch (_) { /* server expiry remains the backstop */ }
    }
  }
  function idle() { return !state || Date.now() - lastActivity >= IDLE_MS; }
  ['pointerdown', 'keydown', 'touchstart'].forEach(type => root.addEventListener(type, event => {
    if (!event.isTrusted) return;
    // An event after timeout must not resurrect a stale session.
    if (state && idle()) { void signOut(); return; }
    lastActivity = Date.now(); persist();
  }, {passive:true}));
  setInterval(() => { if (state && idle()) void signOut(); }, 1000);
  document.addEventListener('visibilitychange', () => { if (state && idle()) void signOut(); });

  function el(tag, text, parent) {
    const node = document.createElement(tag); if (text) node.textContent = text;
    if (parent) parent.append(node); return node;
  }
  function password(parent, label, pin = true) {
    const wrap = el('label', label, parent), input = el('input', '', wrap);
    input.type = 'password'; input.autocomplete = 'off'; input.required = true;
    if (pin) { input.inputMode = 'numeric'; input.pattern = '[0-9]{4}'; input.maxLength = 4; input.minLength = 4; }
    return input;
  }
  function dialog(title) {
    const d = el('dialog'); d.className = 'fl-login'; d.setAttribute('aria-label', title);
    const form = el('form', '', d); el('h2', title, form);
    const status = el('p', '', form); status.setAttribute('role', 'status');
    document.body.append(d); return {d, form, status};
  }
  function keypad(form, input) {
    const grid = el('div', '', form); grid.className = 'fl-keypad';
    ['1','2','3','4','5','6','7','8','9','Clear','0','⌫'].forEach(value => {
      const b = el('button', value, grid); b.type = 'button';
      b.onclick = () => { input.value = value === 'Clear' ? '' : value === '⌫' ? input.value.slice(0,-1) : (input.value + value).slice(0,4); input.focus(); };
    });
  }
  async function requireSession() {
    if (state && !idle()) return state.actor;
    if (state) await signOut();
    if (pendingLogin) return pendingLogin;
    pendingLogin = new Promise(resolve => {
      const {d, form, status} = dialog('Enter your PIN / Escribe tu PIN');
      el('p', 'Use your personal four-digit PIN on this device.', form);
      const pin = password(form, 'Personal PIN'); keypad(form, pin);
      const submit = el('button', 'Sign in', form); submit.type = 'submit';
      const details = el('details', '', form); el('summary', 'Office / owner sign-in', details);
      const key = password(details, 'Personal sign-in key', false); key.required = false;
      const keyButton = el('button', 'Use personal sign-in', details); keyButton.type = 'button';
      const send = async useKey => {
        if (!useKey && !pin.checkValidity()) { pin.reportValidity(); return; }
        submit.disabled = keyButton.disabled = true; status.textContent = 'Signing in…';
        const body = useKey ? {actor_key:key.value.trim()} : {pin:pin.value};
        pin.value = key.value = '';
        try {
          const response = await transport(base + '/auth/session' + (useKey ? '/key' : ''), {method:'POST',
            headers:{'Content-Type':'application/json'}, body:JSON.stringify(body), credentials:'include', cache:'no-store'});
          const data = await response.json();
          if (!response.ok) {
            const retry = response.headers.get('Retry-After');
            status.textContent = retry ? 'Too many attempts. Try again in ' + Math.ceil(Number(retry)/60) + ' minutes or contact Michael.' : data.detail?.message || 'Sign-in failed. Try again.';
            return;
          }
          state = data; lastActivity = Date.now(); lastSentActivity = 0; persist(); changed();
          d.close(); d.remove(); resolve(state.actor);
        } catch (_) { status.textContent = 'Cannot reach Factory Ledger. Try again.'; }
        finally { delete body.pin; delete body.actor_key; submit.disabled = keyButton.disabled = false; pin.focus(); }
      };
      form.onsubmit = event => { event.preventDefault(); void send(false); };
      keyButton.onclick = () => void send(true);
      d.addEventListener('cancel', event => event.preventDefault()); d.showModal(); pin.focus();
    }).finally(() => { pendingLogin = null; });
    return pendingLogin;
  }
  function ownerPIN() {
    return new Promise((resolve, reject) => {
      const {d, form} = dialog('Michael: confirm this action');
      el('p', 'Re-enter your PIN to approve this action only.', form);
      const input = password(form, 'Michael’s PIN'); keypad(form, input);
      el('button', 'Confirm action', form).type = 'submit';
      const cancel = el('button', 'Cancel', form); cancel.type = 'button';
      const close = () => { input.value = ''; d.close(); d.remove(); };
      form.onsubmit = event => { event.preventDefault(); const value = input.value; close(); resolve(value); };
      cancel.onclick = () => { close(); reject(new Error('Action cancelled.')); };
      d.addEventListener('cancel', event => { event.preventDefault(); cancel.click(); });
      d.showModal(); input.focus();
    });
  }
  function mappedURL(url) {
    const result = new URL(url, location.href);
    if (result.origin === production) return base + result.pathname + result.search;
    return result.href;
  }
  async function authenticatedFetch(url, options = {}) {
    const target = mappedURL(url), parsed = new URL(target);
    if (parsed.origin !== base) return transport(target, options);
    const person = await requireSession();
    const headers = new Headers(options.headers || {});
    headers.delete('X-API-Key'); headers.delete('X-FL-Owner-PIN');
    let reauthed = false, stepped = false;
    while (true) {
      headers.set('X-API-Key', state.session_token);
      if (lastActivity <= lastSentActivity && ['GET','HEAD'].includes((options.method || 'GET').toUpperCase())) headers.set('X-FL-Background','1');
      else { headers.delete('X-FL-Background'); lastSentActivity = lastActivity; }
      let response;
      try { response = await transport(target, {...options, headers, cache:'no-store', credentials:'include'}); }
      finally { headers.delete('X-FL-Owner-PIN'); }
      let code;
      if (response.status === 401 || response.status === 403) {
        try { const data = await response.clone().json(); code = (data.result?.detail || data.detail || data.error_detail)?.error_code || data.error_detail?.code; } catch (_) { /* return original response */ }
      }
      if (code === 'SESSION_EXPIRED' && !reauthed) {
        clear(); reauthed = true; const next = await requireSession();
        if (next.id !== person.id) throw new Error('This entry belongs to ' + person.name + '. Sign in as that person to continue.');
        continue; // exact same URL/body/ticket; never retry an uncertain write
      }
      if (code === 'OWNER_PIN_REQUIRED' && !stepped) {
        stepped = true; headers.set('X-FL-Owner-PIN', await ownerPIN()); continue;
      }
      return response;
    }
  }
  clientSession = {base, fetch:authenticatedFetch, requireSession, signOut, ownerPIN,
    actor:() => state?.actor || null, idle};
  clientFetch = authenticatedFetch;
  function header() {
    const bar = el('div'); bar.className = 'fl-person-bar';
    const label = el('span', '', bar); label.id = 'fl-person'; label.setAttribute('role','status');
    const pins = el('a', 'My PIN / PIN administration', bar);
    pins.href = new URL('pin-management.html', scriptURL).href;
    const signout = el('button', 'Sign out', bar); signout.type = 'button'; signout.onclick = async () => { await signOut(); await requireSession(); };
    document.body.prepend(bar); changed();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', header); else header();
  }
})(window);
