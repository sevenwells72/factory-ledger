/* Owner-approved browser + "who is entering?" (migration 091).

   The browser token (localStorage) authenticates THIS BROWSER after Michael
   approves its pairing code once. The tapped name only IDENTIFIES the person;
   it is never a password, never owner proof, and never expires. FL checks both
   on every request and stays authoritative (tickets are bound to the person).

   Fail closed: if the mode is on but FL says it is unavailable (schema, people
   list, network), recording is blocked with a plain message. There is no
   fallback to a personal key, PIN or the shared dashboard key for writes. */
(function (root) {
  'use strict';
  const TOKEN = 'fl-browser-token', PERSON = 'fl-person';
  const TOKEN_RE = /^flb_[A-Za-z0-9_-]{43}$/;
  const POLL_MS = 4000;

  function problem(code, en, es) {
    const error = new Error(en); error.code = code; error.labels = [en, es]; error.personSelect = true; return error;
  }
  const PROBLEMS = {
    PERSON_REQUIRED: ['Choose your name before recording. Nothing was recorded.', 'Elige tu nombre antes de registrar. No se registró nada.'],
    PERSON_CHANGED: ['The name on this device was changed. Check who is entering, then try again. Nothing was recorded.',
      'Se cambió el nombre en este dispositivo. Revisa quién está registrando e intenta otra vez. No se registró nada.'],
    BROWSER_NOT_APPROVED: ['This browser is not approved yet. Ask Michael to approve it. Nothing was recorded.',
      'Este navegador aún no está aprobado. Pide a Michael que lo apruebe. No se registró nada.'],
    UNAVAILABLE: ['Name selection is not available right now. Nothing can be recorded. Try again, or ask Michael.',
      'La selección de nombre no está disponible ahora. No se puede registrar. Intenta otra vez o pregunta a Michael.'],
  };
  const fail = code => problem(code, ...PROBLEMS[code]);

  function create({ base, fetch: transport, storage, config, events = root, now = () => Date.now(), timers = root }) {
    let mode = false, status = 'off', problemCode = null, people = [], person = null, pairing = null, revision = 0, pollTimer = null;
    // R6: once storage fails (private mode, quota, blocked) the page cannot know
    // which name is saved, so selection stays unavailable until a reload. A
    // name is never restored from storage that could not be written or cleared.
    let broken = false, refreshSeq = 0, hydrated = false;
    const listeners = new Set();
    const snapshot = () => ({ mode, status, problem: problemCode, people: people.slice(), person, pairing, revision });
    function emit(previous) {
      const state = snapshot();
      listeners.forEach(fn => { try { fn(state, previous); } catch (_) { /* a listener must not break auth */ } });
      if (events?.dispatchEvent && typeof CustomEvent === 'function' && previous !== undefined)
        events.dispatchEvent(new CustomEvent('fl-person-change', { detail: { person, previous } }));
    }
    function read(key) { if (broken) return null; try { return storage.getItem(key); } catch (_) { breakStorage(); return null; } }
    function write(key, value) {
      if (broken) return false;
      try {
        if (value === null) storage.removeItem(key); else storage.setItem(key, value);
        if (storage.getItem(key) !== value) throw new Error('storage did not keep the value');
        return true;
      } catch (_) { breakStorage(); return false; }
    }
    function breakStorage() {
      if (broken) return;
      broken = true; stopPolling();
      const previous = person; person = null; revision += 1; status = 'unavailable'; problemCode = 'STORAGE';
      emit(previous);
    }
    const token = () => { const value = read(TOKEN) || ''; return TOKEN_RE.test(value) ? value : ''; };
    // The saved name is bound to the browser token it was chosen under, so a
    // re-approved browser never restores a name from an earlier approval.
    const tag = value => value ? value.slice(4, 16) : '';
    function savedPerson() {
      try { const saved = JSON.parse(read(PERSON) || 'null');
        return saved && Number.isInteger(saved.id) && saved.browser === tag(token()) ? saved : null; } catch (_) { return null; }
    }
    function setPerson(next, { notify = true } = {}) {
      const previous = person;
      const wanted = next ? { id: next.id, display_name: next.display_name, role_label: next.role_label } : null;
      // Storage first: memory follows only what storage really holds.
      if (!write(PERSON, wanted ? JSON.stringify({ ...wanted, browser: tag(token()) }) : null)) throw fail('UNAVAILABLE');
      person = wanted;
      if ((previous?.id ?? null) !== (person?.id ?? null)) { revision += 1; if (notify) emit(previous); }
    }
    function dropToken() { if (!write(TOKEN, null)) return false; try { setPerson(null, { notify: false }); } catch (_) { return false; } return true; }
    function adoptStored() {
      // Another tab may have changed the name: the stored one wins, as a CHANGE.
      if (broken) return false;
      const stored = savedPerson();
      const valid = stored && people.find(p => p.id === stored.id && p.available);
      if ((valid?.id ?? null) !== (person?.id ?? null)) {
        const previous = person; person = valid ? { id: valid.id, display_name: valid.display_name, role_label: valid.role_label } : null;
        revision += 1; emit(previous); return true;
      }
      return false;
    }
    async function call(path, { method = 'GET', key } = {}) {
      const controller = typeof AbortController === 'function' ? new AbortController() : null;
      const timer = controller && timers.setTimeout(() => controller.abort(), 15000);
      try {
        const response = await transport(base + path, { method, cache: 'no-store', ...(controller ? { signal: controller.signal } : {}),
          headers: key ? { 'X-API-Key': key } : {} });
        const data = await response.json().catch(() => ({}));
        return { response, data };
      } finally { if (timer) timers.clearTimeout(timer); }
    }
    function stopPolling() { if (pollTimer) { timers.clearTimeout(pollTimer); pollTimer = null; } }
    function poll() { stopPolling(); pollTimer = timers.setTimeout(() => { pollTimer = null; refresh().catch(() => {}); }, POLL_MS); }
    function setStatus(next, code = null) { const changed = next !== status || code !== problemCode; status = next; problemCode = code; return changed; }

    async function refresh() {
      if (!mode || broken) return snapshot();
      const seq = ++refreshSeq, key = token();
      if (broken) return snapshot();
      if (!key) {
        stopPolling(); people = [];
        try { setPerson(null, { notify: false }); } catch (_) { return snapshot(); }
        setStatus('untrusted'); emit(); return snapshot();
      }
      let result;
      try { result = await call('/auth/browsers/status', { key }); }
      catch (_) { if (seq === refreshSeq && token() === key) { setStatus('unavailable', 'NETWORK'); emit(); } return snapshot(); }
      // R7: an answer about an older token, or overtaken by a newer refresh, is ignored.
      if (seq !== refreshSeq || broken || token() !== key) return snapshot();
      const { response, data } = result, code = data?.detail?.error_code;
      if (response.status === 401) {
        people = []; if (!dropToken()) return snapshot();
        setStatus(code === 'BROWSER_REVOKED' ? 'revoked' : 'untrusted'); stopPolling(); emit(); return snapshot();
      }
      if (!response.ok) { setStatus('unavailable', code || 'PERSON_SELECT_UNAVAILABLE'); stopPolling(); emit(); return snapshot(); }
      if (data.status === 'approved') {
        pairing = null; stopPolling();
        people = (data.people || []).filter(p => Number.isInteger(p.id) && typeof p.display_name === 'string');
        setStatus('approved');
        const stored = savedPerson(), current = stored && people.find(p => p.id === stored.id && p.available);
        // R10: restoring the saved name when the page opens is HYDRATION, not a
        // change of user: no fl-person-change (listeners would reload or reset).
        // A refresh after a real change (another tab, revoked name) still notifies.
        const hydrating = !hydrated;
        try { if (!current) setPerson(null, { notify: !hydrating }); else if (current.id !== person?.id) setPerson(current, { notify: !hydrating }); }
        catch (_) { return snapshot(); }
        hydrated = true;
      } else if (data.status === 'pending') { setStatus('pending'); poll(); }
      else { if (!dropToken()) return snapshot(); setStatus(data.status === 'revoked' ? 'revoked' : 'expired'); stopPolling(); }
      emit(); return snapshot();
    }

    const api = {
      async init() {
        let cfg;
        try { cfg = await (config || call('/auth/session/config').then(r => r.data)); }
        catch (_) { cfg = null; }
        if (cfg === null || typeof cfg !== 'object') {
          // R11: settings UNKNOWN is not "selection off": say recording is not
          // available (writes are also blocked by FLDashboard.configUnknown).
          mode = true; setStatus('unavailable', 'CONFIG_UNKNOWN'); emit(); return snapshot();
        }
        mode = cfg.person_select_enabled === true;
        if (!mode) { setStatus('off'); return snapshot(); }
        if (cfg.person_select_available !== true) {
          setStatus('unavailable', cfg.person_select_problem || 'PERSON_SELECT_UNAVAILABLE'); emit(); return snapshot();
        }
        events?.addEventListener?.('storage', event => {
          if (broken || (event.key !== PERSON && event.key !== TOKEN && event.key !== null)) return;
          if (event.key === null) { refreshSeq += 1; refresh().catch(() => {}); return; }   // storage cleared elsewhere
          if (event.key === TOKEN) { refresh().catch(() => {}); return; }
          adoptStored();
        });
        return refresh();
      },
      refresh,
      on(fn) { listeners.add(fn); return () => listeners.delete(fn); },
      get state() { return snapshot(); },
      get mode() { return mode; },
      get person() { return person; },
      get revision() { return revision; },
      async requestApproval() {
        if (!mode) throw fail('UNAVAILABLE');
        let result;
        try { result = await call('/auth/browsers/request', { method: 'POST' }); } catch (_) { setStatus('unavailable', 'NETWORK'); emit(); throw fail('UNAVAILABLE'); }
        const { response, data } = result;
        if (!response.ok || !TOKEN_RE.test(data.browser_token || '')) {
          const error = new Error(data?.detail?.message || PROBLEMS.UNAVAILABLE[0]); error.code = data?.detail?.error_code; throw error;
        }
        refreshSeq += 1;                       // any refresh still running is about the old token
        if (!write(TOKEN, data.browser_token)) throw fail('UNAVAILABLE');
        setPerson(null, { notify: false });
        pairing = { code: data.pairing_code, expires_at: data.pairing_expires_at };
        setStatus('pending'); emit(); poll(); return pairing;
      },
      select(choice) {
        if (broken) throw fail('UNAVAILABLE');
        if (status !== 'approved') throw fail('BROWSER_NOT_APPROVED');
        const found = people.find(p => p.id === choice?.id);
        if (!found || !found.available) throw problem('PERSON_NOT_AVAILABLE', 'That name cannot record right now.', 'Ese nombre no puede registrar ahora.');
        setPerson(found); return person;
      },
      changeUser() { setPerson(null); },     // throws (and stays unavailable) if storage cannot be cleared
      /** Credentials for one request. `write` needs an approved browser AND a chosen name. */
      headers({ write = false } = {}) {
        if (!mode) return null;
        if (broken || status === 'unavailable') throw fail('UNAVAILABLE');
        const key = token();
        if (status !== 'approved' || !key) { if (write) throw fail('BROWSER_NOT_APPROVED'); return null; }
        if (adoptStored() && write) throw fail('PERSON_CHANGED');
        if (write && !person) throw fail('PERSON_REQUIRED');
        return { 'X-API-Key': key, ...(person ? { 'X-FL-Person': String(person.id) } : {}) };
      },
      /** Feed FL refusals back: revoked browsers, unavailable names, misconfiguration. */
      observe(httpStatus, data) {
        const code = data?.detail?.error_code;
        if (code === 'BROWSER_REVOKED' || code === 'BROWSER_NOT_APPROVED') {
          refreshSeq += 1;
          if (dropToken()) { setStatus(code === 'BROWSER_REVOKED' ? 'revoked' : 'untrusted'); emit(); }
        } else if (code === 'PERSON_NOT_SELECTABLE' || code === 'PERSON_NOT_AVAILABLE') {
          try { setPerson(null); } catch (_) { return code; }
          refresh().catch(() => {});
        }
        else if (code === 'PERSON_CHOICES_MISCONFIGURED' || code === 'PERSON_SELECT_UNAVAILABLE') { setStatus('unavailable', code); emit(); }
        return code;
      },
      fail,
      get broken() { return broken; },
    };
    return api;
  }

  // ------------------------------------------------------------------ UI
  function mount(client, { document: doc = root.document, required = false, lang = () => doc.documentElement.lang === 'es' ? 'es' : 'en' } = {}) {
    if (!doc?.createElement) return { open() {}, close() {} };
    const t = (en, es) => lang() === 'es' ? es : en;
    const chip = doc.createElement('div'); chip.className = 'fl-person-chip'; chip.id = 'fl-person-chip'; chip.hidden = true;
    const who = doc.createElement('span'); who.className = 'fl-person-name'; who.setAttribute('role', 'status'); who.setAttribute('aria-live', 'polite');
    const change = doc.createElement('button'); change.type = 'button'; change.className = 'fl-person-change';
    chip.append(who, change);
    const host = doc.querySelector('.app-header .header-right, .runs-header, .header-right, header .toolbar, header') || doc.body;
    host.append(chip);
    const overlay = doc.createElement('div'); overlay.className = 'fl-person-overlay'; overlay.hidden = true;
    overlay.setAttribute('role', 'dialog'); overlay.setAttribute('aria-modal', 'true'); overlay.setAttribute('aria-labelledby', 'fl-person-title');
    const panel = doc.createElement('div'); panel.className = 'fl-person-panel'; overlay.append(panel);
    doc.body.append(overlay);
    // Opening any page with selection on and no name chosen shows the names at
    // once (R9). Pages that only view may dismiss it; Record pages may not.
    let openRequested = true;
    const el = (tag, text, parent, cls) => { const e = doc.createElement(tag); if (text !== undefined) e.textContent = text; if (cls) e.className = cls; parent.append(e); return e; };
    function paint(state) {
      chip.hidden = !state.mode;
      if (!state.mode) { overlay.hidden = true; return; }
      if (state.status === 'approved' && state.person) {
        who.textContent = t('Entering as: ', 'Registrando como: ') + state.person.display_name + ' (' + state.person.role_label + ')';
        change.textContent = t('Change user', 'Cambiar usuario'); change.hidden = false;
      } else {
        who.textContent = state.status === 'approved' ? t('No name chosen', 'Sin nombre elegido')
          : state.status === 'unavailable' ? t('Recording unavailable', 'Registro no disponible') : t('Browser not approved', 'Navegador no aprobado');
        change.textContent = t('Choose your name', 'Elige tu nombre'); change.hidden = false;
      }
      const show = openRequested && !(state.status === 'approved' && state.person);
      overlay.hidden = !show;
      if (show) render(state);
    }
    function render(state) {
      panel.replaceChildren();
      const title = el('h2', '', panel); title.id = 'fl-person-title';
      if (state.status === 'unavailable') {
        title.textContent = t('Recording is not available', 'El registro no está disponible');
        el('p', PROBLEMS.UNAVAILABLE[lang() === 'es' ? 1 : 0], panel);
        const retry = el('button', t('Try again', 'Intentar otra vez'), panel, 'fl-person-primary'); retry.type = 'button';
        // Unknown settings need a fresh page (one authoritative config per page).
        retry.onclick = () => state.problem === 'CONFIG_UNKNOWN' ? root.location?.reload() : client.init().catch(() => {});
      } else if (state.status === 'approved') {
        title.textContent = t('Who is entering?', '¿Quién está registrando?');
        el('p', t('Tap your name. It stays chosen on this device until someone taps Change user.',
          'Toca tu nombre. Queda elegido en este dispositivo hasta que alguien toque Cambiar usuario.'), panel);
        const grid = el('div', undefined, panel, 'fl-person-grid');
        for (const p of state.people) {
          const b = el('button', undefined, grid, 'fl-person-option'); b.type = 'button'; b.disabled = !p.available;
          el('span', p.display_name, b, 'fl-person-option-name');
          el('span', p.available ? p.role_label : t('Not available', 'No disponible'), b, 'fl-person-option-role');
          b.onclick = () => { try { client.select(p); openRequested = required; } catch (error) { alertBox.textContent = error.message; } };
        }
      } else if (state.status === 'pending' && state.pairing) {
        title.textContent = t('Waiting for Michael', 'Esperando a Michael');
        el('p', t('Michael approves this browser once: on his own phone he opens Staff & browsers and enters this code.',
          'Michael aprueba este navegador una vez: en su propio teléfono abre Personal y navegadores y escribe este código.'), panel);
        el('p', state.pairing.code, panel, 'fl-person-code');
        el('p', t('The code works for 10 minutes. This page updates by itself.', 'El código sirve por 10 minutos. Esta página se actualiza sola.'), panel);
      } else {
        title.textContent = t('This browser is not approved', 'Este navegador no está aprobado');
        el('p', state.status === 'revoked' ? t('Michael removed this browser. Ask him to approve it again.', 'Michael quitó este navegador. Pídele que lo apruebe otra vez.')
          : state.status === 'expired' || state.status === 'pending' ? t('The approval code expired. Ask for a new one.', 'El código venció. Pide uno nuevo.')
            : t('Before anyone can record here, Michael approves this browser once.', 'Antes de registrar aquí, Michael aprueba este navegador una vez.'), panel);
        const ask = el('button', t('Ask Michael to approve this browser', 'Pedir a Michael que apruebe este navegador'), panel, 'fl-person-primary'); ask.type = 'button';
        ask.onclick = async () => { ask.disabled = true; try { await client.requestApproval(); } catch (error) { alertBox.textContent = error.message; } finally { ask.disabled = false; } };
      }
      if (!required || state.status !== 'approved') {
        const close = el('button', t('Not now (view only)', 'Ahora no (solo ver)'), panel, 'fl-person-secondary'); close.type = 'button';
        close.onclick = () => { openRequested = false; paint(client.state); };
      }
      const alertBox = el('p', '', panel, 'fl-person-alert'); alertBox.setAttribute('role', 'alert');
    }
    change.onclick = () => {
      // If storage refuses to forget the name, the client goes unavailable (R6); the panel says so.
      try { if (client.state.status === 'approved' && client.person) client.changeUser(); } catch (_) { /* shown as unavailable */ }
      openRequested = true; paint(client.state);
    };
    client.on(state => paint(state));
    paint(client.state);
    return { open() { openRequested = true; paint(client.state); }, close() { openRequested = false; paint(client.state); }, paint: () => paint(client.state) };
  }

  const exported = { create, mount, TOKEN, PERSON, PROBLEMS };
  if (typeof module !== 'undefined') module.exports = exported;
  root.FLPersonSelectLib = exported;
  if (!root.document || !root.FLDashboard) return;
  // One shared client per page, created against the API the page talks to.
  const client = create({ base: root.FLDashboard.base, fetch: root.fetch.bind(root), storage: root.localStorage, config: root.FLDashboard.config });
  let ui = { open() {}, close() {}, paint() {} };
  client.ready = client.init().catch(() => client.state);
  const start = () => { ui = mount(client, { required: root.document.body?.dataset.flPersonRequired === 'true' }); };
  if (root.document.readyState === 'loading') root.document.addEventListener('DOMContentLoaded', start, { once: true }); else start();
  client.open = () => ui.open();
  root.FLPersonSelect = client;
})(typeof window === 'undefined' ? globalThis : window);
