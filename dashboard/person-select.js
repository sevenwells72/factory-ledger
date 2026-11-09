/* The tapped name is the operator. No browser enrolment, secret or expiring session. */
(function (root) {
  'use strict';
  const PERSON = 'fl-person';
  const PEOPLE = [
    { id: 'Michael', display_name: 'Michael', role: 'owner', role_label: 'Owner', available: true },
    { id: 'Luz', display_name: 'Luz', role: 'office', role_label: 'Office', available: true },
    { id: 'Arturo', display_name: 'Arturo', role: 'floor', role_label: 'Floor Lead', available: true },
    { id: 'Miriam', display_name: 'Miriam', role: 'office', role_label: 'Office', available: true },
  ];
  const PROBLEMS = {
    PERSON_REQUIRED: ['Choose your name before recording.', 'Elige tu nombre antes de registrar.'],
    PERSON_CHANGED: ['The operator changed while this entry was being prepared. Open the entry again.', 'Cambió el operador mientras se preparaba esta entrada. Abre la entrada otra vez.'],
    UNAVAILABLE: ['The page could not load. Refresh the page.', 'No se pudo cargar la página. Actualiza la página.'],
  };
  function fail(code) {
    const labels = PROBLEMS[code] || PROBLEMS.UNAVAILABLE;
    const error = new Error(labels[0]); error.code = code; error.labels = labels; error.personSelect = true; return error;
  }
  function create({ storage, events = root } = {}) {
    let person = null, revision = 0, initialized = false;
    const listeners = new Set();
    const find = value => PEOPLE.find(p => p.display_name === value?.display_name || p.id === value?.id) || null;
    const saved = () => { try { return find(JSON.parse(storage?.getItem(PERSON) || 'null')); } catch (_) { return null; } };
    const snapshot = () => ({ mode: true, status: 'approved', problem: null, people: PEOPLE.slice(), person, revision });
    function emit(previous) {
      listeners.forEach(fn => { try { fn(snapshot(), previous); } catch (_) {} });
      if (previous !== undefined && events?.dispatchEvent && typeof CustomEvent === 'function')
        events.dispatchEvent(new CustomEvent('fl-person-change', { detail: { person, previous } }));
    }
    function choose(next, persist = true) {
      const previous = person;
      person = next;
      // Private/blocked storage must never prevent tapping a name. Keep the
      // choice in this page when the browser cannot persist it.
      if (persist) try {
        if (next) storage?.setItem(PERSON, JSON.stringify(next)); else storage?.removeItem(PERSON);
      } catch (_) {}
      if (previous?.id !== person?.id) { revision += 1; emit(previous); }
    }
    return {
      async init() {
        if (!initialized) {
          initialized = true; person = saved();
          events?.addEventListener?.('storage', event => { if (event.key === PERSON || event.key === null) choose(saved(), false); });
        }
        emit(); return snapshot();
      },
      async refresh() { return snapshot(); },
      on(fn) { listeners.add(fn); return () => listeners.delete(fn); },
      get state() { return snapshot(); }, get mode() { return true; }, get person() { return person; },
      get revision() { return revision; }, get broken() { return false; },
      select(choice) { const next = find(choice); if (!next) throw fail('PERSON_REQUIRED'); choose(next); return person; },
      changeUser() { choose(null); },
      headers({ write = false } = {}) {
        if (!person) { if (write) throw fail('PERSON_REQUIRED'); return null; }
        return { 'X-FL-Person': person.display_name };
      },
      observe() { return null; },
      fail,
    };
  }
  function mount(client, { document: doc = root.document, lang = () => doc.documentElement.lang === 'es' ? 'es' : 'en' } = {}) {
    if (!doc?.createElement) return { open() {}, close() {} };
    const chip = doc.createElement('div'); chip.className = 'fl-person-chip'; chip.id = 'fl-person-chip';
    const who = doc.createElement('span'); who.className = 'fl-person-name'; who.setAttribute('role', 'status');
    const change = doc.createElement('button'); change.type = 'button'; change.className = 'fl-person-change';
    chip.append(who, change);
    (doc.querySelector('[data-fl-person-host]') || doc.querySelector('.app-header .header-right, .runs-header, .header-right, header .toolbar, header') || doc.body).append(chip);
    const overlay = doc.createElement('div'); overlay.className = 'fl-person-overlay';
    overlay.setAttribute('role', 'dialog'); overlay.setAttribute('aria-modal', 'true'); overlay.setAttribute('aria-label', 'Choose your name');
    const panel = doc.createElement('div'); panel.className = 'fl-person-grid'; overlay.append(panel);
    for (const p of PEOPLE) {
      const button = doc.createElement('button'); button.type = 'button'; button.className = 'fl-person-option'; button.textContent = p.display_name;
      button.onclick = () => client.select(p); panel.append(button);
    }
    doc.body.append(overlay);
    function paint() {
      const person = client.person;
      overlay.hidden = !!person; chip.hidden = !person;
      doc.documentElement.dataset.flPersonPicker = person ? 'off' : 'on';
      who.textContent = person?.display_name || '';
      change.textContent = lang() === 'es' ? 'Cambiar usuario' : 'Change user';
    }
    change.onclick = () => client.changeUser(); client.on(paint); paint();
    return { open() { if (!client.person) paint(); }, close() { paint(); }, paint };
  }
  const exported = { create, mount, PERSON, PEOPLE, PROBLEMS };
  if (typeof module !== 'undefined') module.exports = exported;
  root.FLPersonSelectLib = exported;
  if (!root.document || !root.FLDashboard) return;
  let storage; try { storage = root.localStorage; } catch (_) {}
  const client = create({ storage });
  client.ready = client.init();
  root.document.documentElement.dataset.flPersonPicker = client.person ? 'off' : 'on';
  let ui = { open() {}, close() {} };
  const start = () => { ui = mount(client); };
  if (root.document.readyState === 'loading') root.document.addEventListener('DOMContentLoaded', start, { once: true }); else start();
  client.open = () => ui.open(); root.FLPersonSelect = client;
})(typeof window === 'undefined' ? globalThis : window);
