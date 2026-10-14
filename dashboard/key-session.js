/* Personal-key exchange. Persistence contains only the FL token. */
(function (root) {
  'use strict';
  const STORE = 'fl-key-session-token';
  function create({ base, fetch: transport, storage, config: sharedConfig, changed = () => {} }) {
    let enabled = false, token = '', actor = null, revision = 0;
    const publish = () => changed({ enabled, actor, signedIn: !!token });
    function clear() { revision += 1; token = ''; actor = null; storage.removeItem(STORE); publish(); }
    async function request(path, options = {}) {
      const requestToken = token;
      const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 15000);
      let response, result;
      try {
        response = await transport(base + path, { ...options, cache: 'no-store', signal: controller.signal,
          headers: { ...(options.headers || {}), ...(requestToken ? { 'X-API-Key': requestToken } : {}) } });
        result = await response.json();
      } finally { clearTimeout(timer); }
      if (!response.ok) {
        if (result.detail?.error_code === 'SESSION_EXPIRED' && requestToken === token) clear();
        const error = new Error(result.detail?.message || 'Sign-in request failed. Please try again.');
        error.status = response.status; error.detail = result.detail; throw error;
      }
      return result;
    }
    return {
      async init() {
        const operation = revision;
        const config = await (sharedConfig || request('/auth/session/config'));
        if (operation !== revision) return enabled;
        enabled = config.key_sessions_enabled === true;
        if (enabled) {
          token = storage.getItem(STORE) || '';
          if (token) {
            try { const who = await request('/auth/whoami', { headers: { 'X-FL-Background': '1' } });
              if (operation !== revision) return enabled;
              if (who.key_kind !== 'session' || !who.actor) clear(); else actor = who.actor;
            } catch (error) { if (operation !== revision) return enabled; if (error.status === 401 || error.status === 403) clear(); else throw error; }
          }
        } else storage.removeItem(STORE);
        publish(); return enabled;
      },
      async signIn(personalKey) {
        if (!enabled) throw new Error('Key sessions are unavailable.');
        const operation = ++revision;
        const result = await request('/auth/session', { method: 'POST',
          headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ actor_key: personalKey }) });
        if (operation !== revision) throw new Error('Sign-in changed. Please try again.');
        if (result.key_kind !== 'session' || !result.session_token || !result.actor) throw new Error('FL did not confirm a session.');
        token = result.session_token; actor = result.actor;
        storage.setItem(STORE, token); publish(); return actor;
      },
      async signOut() {
        const operation = ++revision;
        if (token) {
          try { await request('/auth/session/logout', { method: 'POST' }); }
          catch (error) { if (error.status !== 401) throw error; }
        }
        if (operation !== revision) return;
        clear();
      },
      clear,
      get enabled() { return enabled; }, get token() { return token; }, get actor() { return actor; },
      async headers(original) {
        if (!token || !actor) throw new Error('Sign in with your personal key before saving.');
        const headers = new Headers(original || {}); headers.set('X-API-Key', token); return headers;
      }
    };
  }
  function installDashboard(base) {
    let dialog, label, signIn, signOut, status, resolvePending;
    const runtime=root.FLDashboard;
    const client = create({ base, fetch: root.fetch.bind(root), storage: root.sessionStorage, config:runtime?.config,
      changed: state => {
        if (!label) return;
        label.textContent = state.actor ? 'Signed in as ' + state.actor.name : 'Sign in before recording an entry';
        signIn.hidden = !!state.actor; signOut.hidden = !state.actor;
      } });
    const initialized=runtime?runtime.ready.then(on=>{if(!on){client.clear();return false;}return client.init();}):client.init();
    const ready = initialized.then(enabled => {
      if (!enabled) return;
      const mount = () => {
        const bar = document.createElement('div'); bar.className = 'fl-session-bar';
        label = document.createElement('span'); label.setAttribute('role', 'status');
        signIn = document.createElement('button'); signIn.type = 'button'; signIn.textContent = 'Sign in';
        signOut = document.createElement('button'); signOut.type = 'button'; signOut.textContent = 'Sign out';
        status = document.createElement('span'); status.setAttribute('role', 'alert');
        bar.append(label, signIn, signOut, status);
        const header = document.querySelector('.app-header .header-right, .runs-header, .header-right');
        (header || document.body).append(bar);
        dialog = document.createElement('dialog'); dialog.className = 'fl-session-dialog';
        const form = document.createElement('form'), title = document.createElement('h2'); title.textContent = 'Personal sign-in';
        const fieldLabel = document.createElement('label'); fieldLabel.textContent = 'Personal actor key';
        const input = document.createElement('input'); input.type = 'password'; input.autocomplete = 'off'; input.required = true;
        fieldLabel.append(input);
        const submit = document.createElement('button'); submit.type = 'submit'; submit.textContent = 'Sign in';
        const cancel = document.createElement('button'); cancel.type = 'button'; cancel.textContent = 'Cancel';
        const errorBox = document.createElement('p'); errorBox.setAttribute('role', 'alert');
        const hint = document.createElement('p'); hint.textContent = 'Your sign-in ends after 180 minutes without activity. Sign out before handing over this device.';
        form.append(title, fieldLabel, hint, errorBox, submit, cancel); dialog.append(form); document.body.append(dialog);
        signIn.onclick = () => { errorBox.textContent = ''; dialog.showModal(); input.focus(); };
        cancel.onclick = () => dialog.close();
        dialog.addEventListener('close', () => { input.value = ''; if (resolvePending) { resolvePending(false); resolvePending = null; } });
        form.onsubmit = async event => {
          event.preventDefault(); submit.disabled = true; errorBox.textContent = '';
          const promise = client.signIn(input.value.trim()); input.value = '';
          try { await promise; if (resolvePending) { resolvePending(true); resolvePending = null; } dialog.close(); }
          catch (error) { errorBox.textContent = error.message; }
          finally { submit.disabled = false; }
        };
        signOut.onclick = async () => {
          signOut.disabled = true; status.textContent = '';
          try { await client.signOut(); }
          catch (error) { status.textContent = 'Sign out failed. Try again before handing over this device.'; }
          finally { signOut.disabled = false; }
        };
        label.textContent = client.actor ? 'Signed in as ' + client.actor.name : 'Sign in before recording an entry';
        signIn.hidden = !!client.actor; signOut.hidden = !client.actor;
      };
      if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', mount, { once: true }); else mount();
    });
    // Configuration errors are surfaced by saves and must not silently downgrade auth.
    ready.catch(() => {});
    return {
      async prepare(url, options = {}) {
        // Restoration must finish before any authenticated request. Signed-in
        // reads need the same person-bound scope as writes on the pilot API.
        if (new URL(url, location.href).origin !== new URL(base).origin) return options;
        await ready;
        if (!client.enabled) return options;
        if (['GET','HEAD','OPTIONS'].includes((options.method || 'GET').toUpperCase())) {
          return client.actor ? { ...options, headers: await client.headers(options.headers) } : options;
        }
        if (!client.actor) {
          if (!dialog) throw new Error('Sign-in is still loading. Please try again.');
          if (resolvePending) throw new Error('Complete the pending sign-in first.');
          const accepted = new Promise(resolve => { resolvePending = resolve; });
          dialog.showModal();
          if (!await accepted) throw new Error('Entry was not saved. Sign in to continue.');
        }
        return { ...options, headers: await client.headers(options.headers) };
      },
      async observe(response) {
        if (client.enabled && response.status === 401) {
          const data = await response.clone().json().catch(() => ({}));
          if (data.detail?.error_code === 'SESSION_EXPIRED') client.clear();
        }
        return response;
      }
    };
  }
  const api = { create, installDashboard, STORE };
  if (typeof module !== 'undefined') module.exports = api;
  root.FLKeySessions = api;
})(typeof window === 'undefined' ? globalThis : window);
