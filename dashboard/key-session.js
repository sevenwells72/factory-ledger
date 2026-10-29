/* Personal-key exchange and phone PINs. Persistence holds only FL tokens and,
   on an enrolled phone, its device secret (never a personal key or a PIN). */
(function (root) {
  'use strict';
  const STORE = 'fl-key-session-token';
  // A PIN session lasts the shift (12 h) and survives closing the browser, so it
  // lives in localStorage next to the phone's device secret.
  const PIN_STORE = 'fl-pin-session-token', DEVICE = 'fl-device';
  const PIN = /^[0-9]{4}$/, SECRET = /^[A-Za-z0-9_-]{43}$/;
  function create({ base, fetch: transport, storage, config: sharedConfig, changed = () => {},
                    deviceStorage = (typeof root.localStorage !== 'undefined' ? root.localStorage : null) }) {
    let enabled = false, pinEnabled = false, token = '', actor = null, revision = 0, device = null;
    const publish = () => changed({ enabled, actor, signedIn: !!token, pinEnabled, device: device && { actor: device.actor, label: device.label } });
    function clear() { revision += 1; token = ''; actor = null; storage.removeItem(STORE); deviceStorage?.removeItem(PIN_STORE); publish(); }
    function readDevice() {
      try { const saved = JSON.parse(deviceStorage?.getItem(DEVICE) || 'null');
        return saved && SECRET.test(saved.secret || '') && saved.actor?.name ? saved : null; } catch (_) { return null; }
    }
    function forgetLocally() { device = null; deviceStorage?.removeItem(DEVICE); }
    function adopt(result, { phone }) {
      if (result.key_kind !== 'session' || !result.session_token || !result.actor) throw new Error('FL did not confirm a session.');
      token = result.session_token; actor = result.actor;
      if (phone) { deviceStorage.setItem(PIN_STORE, token); storage.removeItem(STORE); }
      else { storage.setItem(STORE, token); deviceStorage?.removeItem(PIN_STORE); }
      publish(); return actor;
    }
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
        error.status = response.status; error.detail = result.detail; error.code = result.detail?.error_code; throw error;
      }
      return result;
    }
    return {
      async init() {
        const operation = revision;
        const config = await (sharedConfig || request('/auth/session/config'));
        if (operation !== revision) return enabled;
        enabled = config.key_sessions_enabled === true;
        pinEnabled = enabled && config.pin_sign_in === true && !!deviceStorage;
        device = pinEnabled ? readDevice() : null;
        // With PINs switched off, a phone session token is dropped (the device secret stays).
        if (!pinEnabled) deviceStorage?.removeItem(PIN_STORE);
        if (enabled) {
          token = storage.getItem(STORE) || (pinEnabled && deviceStorage.getItem(PIN_STORE)) || '';
          if (token) {
            try { const who = await request('/auth/whoami', { headers: { 'X-FL-Background': '1' } });
              if (operation !== revision) return enabled;
              if (who.key_kind !== 'session' || !who.actor) clear(); else actor = who.actor;
            } catch (error) { if (operation !== revision) return enabled; if (error.status === 401 || error.status === 403) clear(); else throw error; }
          }
        } else storage.removeItem(STORE);
        publish(); return enabled;
      },
      // One field, two credentials: six digits on an enrolled phone is a PIN,
      // anything else is a personal key.
      async signIn(value) {
        if (!enabled) throw new Error('Key sessions are unavailable.');
        if (pinEnabled && device && PIN.test(value)) return this.pinSignIn(value);
        const operation = ++revision;
        const result = await request('/auth/session', { method: 'POST',
          headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ actor_key: value }) });
        if (operation !== revision) throw new Error('Sign-in changed. Please try again.');
        return adopt(result, { phone: false });
      },
      async pinSignIn(pin) {
        if (!pinEnabled || !device) throw new Error('This phone is not set up for a PIN.');
        const operation = ++revision;
        try {
          const result = await request('/auth/pin', { method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ device_secret: device.secret, pin }) });
          if (operation !== revision) throw new Error('Sign-in changed. Please try again.');
          return adopt(result, { phone: true });
        } catch (error) { if (error.code === 'DEVICE_NOT_RECOGNIZED') { forgetLocally(); publish(); } throw error; }
      },
      async resetPin(newPin) {
        if (!pinEnabled || !device) throw new Error('This phone is not set up for a PIN.');
        const operation = ++revision;
        const result = await request('/auth/pin/reset', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ device_secret: device.secret, new_pin: newPin }) });
        if (operation !== revision) throw new Error('Sign-in changed. Please try again.');
        return adopt(result, { phone: true });
      },
      // Needs a personal-key session and the key typed again; the phone then signs in with the PIN.
      async enroll(pin, label, personalKey) {
        if (!pinEnabled || !token || !actor) throw new Error('Sign in with your personal key first.');
        const result = await request('/auth/device/enroll', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ actor_key: personalKey, pin, label, ...(device ? { replace_device_secret: device.secret } : {}) }) });
        device = { secret: result.device_secret, actor: result.actor, label: result.device.label };
        deviceStorage.setItem(DEVICE, JSON.stringify(device));
        return adopt(result, { phone: true });
      },
      async changePin(currentPin, newPin) {
        if (!device || !token) throw new Error('Sign in with your PIN on this phone first.');
        return request('/auth/pin/change', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ device_secret: device.secret, current_pin: currentPin, new_pin: newPin }) });
      },
      async forgetDevice() {
        if (!device) return;
        const phoneSession = !!token && deviceStorage?.getItem(PIN_STORE) === token;
        if (token) { try { await request('/auth/device/forget', { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ device_secret: device.secret }) }); } catch (error) { if (![401, 403, 404].includes(error.status)) throw error; } }
        forgetLocally();
        if (phoneSession) clear(); else publish();
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
      get pinEnabled() { return pinEnabled; }, get device() { return device && { actor: device.actor, label: device.label }; },
      get phoneSession() { return !!token && deviceStorage?.getItem(PIN_STORE) === token; },
      async headers(original) {
        if (!token || !actor) throw new Error('Sign in with your personal key before saving.');
        const headers = new Headers(original || {}); headers.set('X-API-Key', token); return headers;
      }
    };
  }
  // On an enrolled phone a page's personal-key field becomes "PIN (or key)",
  // and after a personal-key sign-in it offers PIN set-up. Pages keep their own
  // submit handler: client.signIn() tells a PIN from a key. Returns repaint().
  function pinField(client, input, { after } = {}) {
    const doc = root.document;
    // Pages without a real field (or test doubles) keep their own behaviour.
    if (!doc?.createElement || !input || typeof input.after !== 'function' || typeof input.closest !== 'function') return () => {};
    const label = input.closest('label')?.querySelector?.('[data-en]') || doc.querySelector?.('label[for="' + input.id + '"]');
    const original = { en: label?.dataset.en, es: label?.dataset.es, placeholder: input.placeholder, mode: input.inputMode || '' };
    const es = () => doc.documentElement.lang === 'es';
    const toggle = doc.createElement('button'); toggle.type = 'button'; toggle.className = 'fl-pin-toggle'; toggle.hidden = true;
    const offer = doc.createElement('p'); offer.className = 'fl-pin-offer'; offer.hidden = true;
    const link = doc.createElement('a'); link.href = 'pin.html'; offer.append(link);
    let useKey = false, swapped = false;
    function relabel(en, esText) { if (label) { label.dataset.en = en; label.dataset.es = esText; label.textContent = es() ? esText : en; } }
    function paint() {
      const enrolled = client.pinEnabled && !!client.device;
      if (enrolled && !useKey) {
        const name = client.device.actor.name;
        relabel('PIN for ' + name + ' (4 digits)', 'PIN de ' + name + ' (4 dígitos)');
        input.inputMode = 'numeric'; input.placeholder = es() ? 'PIN de 4 dígitos' : '4-digit PIN'; swapped = true;
      } else if (swapped) {
        // Hand the field back to the page as it was; untouched unless we changed it.
        if (original.en !== undefined) relabel(original.en, original.es);
        input.inputMode = original.mode; input.placeholder = original.placeholder; swapped = false;
      }
      toggle.hidden = !enrolled;
      toggle.textContent = useKey ? (es() ? 'Usar mi PIN' : 'Use my PIN') : (es() ? 'Usar clave personal' : 'Use personal key instead');
      offer.hidden = !(client.pinEnabled && client.actor && !client.phoneSession);
      link.textContent = es() ? 'Configura un PIN de 4 dígitos en este teléfono' : 'Set up a 4-digit PIN on this phone';
    }
    toggle.onclick = () => { useKey = !useKey; paint(); input.focus(); };
    input.after(toggle);
    (after || input.form || input).after(offer);
    paint();
    return paint;
  }
  function installDashboard(base) {
    let dialog, label, signIn, signOut, status, resolvePending, paintPin = () => {};
    const runtime=root.FLDashboard;
    const client = create({ base, fetch: root.fetch.bind(root), storage: root.sessionStorage, config:runtime?.config,
      changed: state => {
        if (!label) return;
        label.textContent = state.actor ? 'Signed in as ' + state.actor.name : 'Sign in before recording an entry';
        signIn.hidden = !!state.actor; signOut.hidden = !state.actor; paintPin();
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
        const fieldLabel = document.createElement('label'), fieldText = document.createElement('span');
        fieldText.dataset.en = fieldText.textContent = 'Personal actor key'; fieldText.dataset.es = 'Clave personal';
        const input = document.createElement('input'); input.type = 'password'; input.autocomplete = 'off'; input.required = true;
        fieldLabel.append(fieldText, input);
        const submit = document.createElement('button'); submit.type = 'submit'; submit.textContent = 'Sign in';
        const cancel = document.createElement('button'); cancel.type = 'button'; cancel.textContent = 'Cancel';
        const errorBox = document.createElement('p'); errorBox.setAttribute('role', 'alert');
        const hint = document.createElement('p'); hint.textContent = client.pinEnabled
          ? 'With a PIN, your sign-in lasts the shift (12 hours). With a personal key it ends after 180 minutes without activity. Sign out before handing over this device.'
          : 'Your sign-in ends after 180 minutes without activity. Sign out before handing over this device.';
        form.append(title, fieldLabel, hint, errorBox, submit, cancel); dialog.append(form); document.body.append(dialog);
        paintPin = pinField(client, input, { after: bar });
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
  const api = { create, installDashboard, pinField, STORE, PIN_STORE, DEVICE };
  if (typeof module !== 'undefined') module.exports = api;
  root.FLKeySessions = api;
})(typeof window === 'undefined' ? globalThis : window);
