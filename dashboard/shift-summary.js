/* D3-lite: renders the frozen A9 JSON (docs/contracts/shift-summary.md). The page never
   computes anything from ledger rows — every line, label and flag text comes from FL. */
(function (root) {
  'use strict';
  const PRODUCTION_API = 'https://fastapi-production-b73a.up.railway.app';
  const esc = value => String(value ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /** Where the API lives: `?api=` override, same origin when FL serves the page, else production. */
  function apiBase(loc, search) {
    const override = new URLSearchParams(search || '').get('api');
    if(root.FLDashboard)return root.FLDashboard.apiBase(loc,override||root.FL_SESSION_API_BASE);
    if((loc?.hostname||'').endsWith('.netlify.app')&&loc.hostname!=='cns-factory-ledger.netlify.app')return 'https://fastapi-staging-production-dd7b.up.railway.app';
    if (override && /^https?:\/\//.test(override)) return override.replace(/\/$/, '');
    if (!loc || loc.protocol === 'file:' || /netlify\.app$/.test(loc.hostname || '')) return PRODUCTION_API;
    return loc.origin;
  }

  /** The error text a person reads: `error_detail.message_es` when Spanish and present, else the message. */
  function errorText(error, lang) {
    if(error?.accessUnavailable)return root.FLDashboard.accessMessage(lang);
    const data = error && (error.result || error);
    const detail = data && (data.error_detail || data.detail || data);
    if (Array.isArray(detail)) return detail.map(e => `${(e.loc || []).slice(1).join('.')}: ${e.msg}`).join('\n');
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail === 'object') {
      const message = (lang === 'es' && detail.message_es) || detail.message || detail.error_code || detail.code;
      return message || JSON.stringify(detail);
    }
    return lang === 'es' ? 'La solicitud falló' : 'Request failed';
  }

  /** A `409 SHIFT_SUMMARY_STALE` carries the fresh document at detail.summary (contract §3.1). */
  function staleSummary(error) {
    const data = error && (error.result || error);
    const code = data && ((data.error_detail && data.error_detail.code) || (data.detail && data.detail.error_code));
    return code === 'SHIFT_SUMMARY_STALE' && data.detail && data.detail.summary ? data.detail.summary : null;
  }

  /** The confirm body echoes exactly what was shown (§3.1): hash + receipt count from the document. */
  function confirmBody(doc, outcome, notes, lang) {
    return { date: doc.date, outcome, notes: notes || '', summary_hash: doc.summary_hash, receipts_seen: doc.receipts_today, lang: lang || 'en' };
  }

  function clock(iso) { const m = /T(\d{2}:\d{2})/.exec(iso || ''); return m ? m[1] : ''; }

  // The immutable values come from FL; these helpers only localise their presentation.
  const localized = (value, field, lang) => value[`${field}_${lang === 'es' ? 'es' : 'en'}`] ?? value[field] ?? value[`${field}_en`] ?? '';
  const tierText = (tier, lang) => tier === 'extra_kosher' ? 'Extra-kosher' :
    tier === 'regular' ? 'Regular' : (tier || (lang === 'es' ? 'sin registrar' : 'not recorded'));
  const stampText = iso => iso ? `${String(iso).slice(0, 10)} ${clock(iso)}`.trim() : '';
  const hasKosherEvidence = row => ['kosher_tier', 'kosher_label', 'transaction_kosher_downgrade', 'kosher_attested_by_actor_id', 'kosher_attested_at']
    .some(field => Object.prototype.hasOwnProperty.call(row, field));

  function attestationText(evidence, lang, pending) {
    const prefix = lang === 'es' ? 'Atestación del propietario: ' : 'Owner attestation: ';
    const actorId = evidence.kosher_attested_by_actor_id;
    const at = stampText(evidence.kosher_attested_at);
    if (actorId === null || actorId === undefined) return prefix + (pending ? (lang === 'es' ? 'pendiente' : 'pending') : (lang === 'es' ? 'sin registrar' : 'none recorded'));
    return prefix + `actor #${actorId}` + (at ? ` · ${at}` : (lang === 'es' ? ' · hora sin registrar' : ' · time not recorded'));
  }

  function kosherEvidenceHtml(row, lang) {
    let html = '';
    if (hasKosherEvidence(row)) {
      const tier = (lang === 'es' ? 'Nivel kosher: ' : 'Kosher tier: ') + tierText(row.kosher_tier, lang);
      const label = (lang === 'es' ? 'Etiqueta: ' : 'Label: ') + (row.kosher_label || (lang === 'es' ? 'sin registrar' : 'not recorded'));
      const downgrade = (lang === 'es' ? 'Cambio a regular: ' : 'Downgrade: ') +
        (row.transaction_kosher_downgrade === true ? (lang === 'es' ? 'Sí' : 'Yes') : row.transaction_kosher_downgrade === false ? 'No' : (lang === 'es' ? 'sin registrar' : 'not recorded'));
      html += `<div class="kosher-evidence"><span>${esc(tier)} · ${esc(label)}</span><span>${esc(downgrade)}</span><span>${esc(attestationText(row, lang, row.held === true && row.posted === false))}</span></div>`;
    }
    const lots = row.detail && row.detail.kosher_lots || [];
    if (lots.length) html += `<div class="kosher-lot-evidence">${lots.map(lot => `<span>${esc((lang === 'es' ? 'Lote #' : 'Lot #') + lot.lot_id)}: ${esc(tierText(lot.kosher_tier, lang))} · ${esc(attestationText(lot, lang, false))}</span>`).join('')}</div>`;
    return html;
  }

  function flagHtml(flag, lang) {
    const cls = flag.severity === 'info' ? ' info' : flag.severity === 'block' ? ' block' : '';
    return `<span class="flag${cls}">⚠ ${esc(localized(flag, 'message', lang) || flag.code)}</span>`;
  }

  function receiptRowHtml(row, lang) {
    const cls = ['row', row.late_entry ? 'late' : '', row.effective_status === 'voided' ? 'voided' : ''].filter(Boolean).join(' ');
    const line = localized(row, 'line', lang).replace(row.receipt_number, '').trim();
    const meta = [];
    if (row.actor && row.actor.name) meta.push((hasKosherEvidence(row) ? (lang === 'es' ? 'Preparado por ' : 'Prepared by ') : '') + row.actor.name);
    if (row.entered_at) meta.push((lang === 'es' ? 'registrado ' : 'entered ') + clock(row.entered_at));
    if (row.late_entry) meta.push(lang === 'es' ? `tardía ${row.hours_late} h` : `late ${row.hours_late} h`);
    if (row.entered_on_later_day) meta.push(lang === 'es' ? 'registrado otro día' : 'entered on a later day');
    return `<div class="${cls}" data-receipt="${esc(row.receipt_number)}"><span class="receipt-number">${esc(row.receipt_number)}</span><span class="line">${esc(line)}</span>` +
      `<div class="meta">${esc(meta.join(' · '))}</div>${kosherEvidenceHtml(row, lang)}${(row.flags || []).map(f => flagHtml(f, lang)).join('')}</div>`;
  }

  function expectationRowHtml(row, lang) {
    if (row.kind !== 'kosher_hold') return `<div class="row${row.overdue ? ' late' : ''}"><span class="line">${esc(localized(row, 'line', lang))}</span></div>`;
    const approval = row.approval || {};
    const expired = approval.status === 'expired';
    const line = localized(row, 'line', lang).replace(row.receipt_number, '').trim();
    const status = expired ? (lang === 'es' ? 'Vencido — pendiente de rechazo del propietario' : 'Expired — awaiting owner rejection') :
      (lang === 'es' ? 'Pendiente de aprobación del propietario' : 'Awaiting owner approval');
    const expiry = stampText(approval.expires_at || row.expires_at);
    const state = [status, lang === 'es' ? 'No registrado' : 'Not posted'];
    if (approval.owner_actor_id !== null && approval.owner_actor_id !== undefined) state.push((lang === 'es' ? 'Propietario para aprobación: actor #' : 'Owner for approval: actor #') + approval.owner_actor_id);
    if (expiry) state.push((expired ? (lang === 'es' ? 'Venció ' : 'Expired ') : (lang === 'es' ? 'Vence ' : 'Expires ')) + expiry);
    const preparer = row.actor && row.actor.name ? (lang === 'es' ? 'Preparado por ' : 'Prepared by ') + row.actor.name : '';
    return `<div class="row held${expired ? ' expired' : ''}" data-receipt="${esc(row.receipt_number)}"><span class="receipt-number">${esc(row.receipt_number)}</span><span class="line">${esc(line)}</span>` +
      `<div class="meta">${esc(preparer)}</div><div class="approval-state">${esc(state.join(' · '))}</div>${kosherEvidenceHtml(row, lang)}${(row.flags || []).map(f => flagHtml(f, lang)).join('')}</div>`;
  }

  function lotRowHtml(row, lang) {
    return `<div class="row lot-line${row.negative || row.short ? ' short' : ''}"><span class="line">${esc(localized(row, 'line', lang))}</span><div class="meta">${esc((row.receipts || []).join(', '))}</div></div>`;
  }

  function exceptionRowHtml(row, lang) {
    const due = row.due_at ? `${lang === 'es' ? 'vence' : 'due'} ${row.due_at.slice(0, 10)} ${clock(row.due_at)}` : (lang === 'es' ? 'sin fecha' : 'no due date');
    return `<div class="row exc${row.overdue ? ' overdue' : ''}"><span class="line">${esc(localized(row, 'summary', lang))}</span><span class="due">#${esc(row.exception_id)} · ${esc(due)}</span></div>`;
  }

  function sectionHtml(section, lang) {
    const rows = section.rows || [];
    let body;
    if (!rows.length) body = `<div class="section-empty">${esc(localized(section, 'empty', lang))}</div>`;
    else if (section.key === 'not_entered_yet') body = rows.map(r => expectationRowHtml(r, lang)).join('');
    else if (section.key === 'lot_balances_touched') body = rows.map(r => lotRowHtml(r, lang)).join('');
    else if (section.key === 'open_exceptions') body = rows.map(r => exceptionRowHtml(r, lang)).join('');
    else body = rows.map(r => receiptRowHtml(r, lang)).join('');
    const rollup = section.key === 'open_exceptions' && localized(section, 'rollup', lang) ? `<div class="section-rollup">${esc(localized(section, 'rollup', lang))}</div>` : '';
    return `<section class="section" data-key="${esc(section.key)}"><div class="section-head"><span>${esc(localized(section, 'label', lang))}</span><span class="count">${esc(section.count)}</span></div>${rollup}${body}</section>`;
  }

  /** The additive `other_receipts` block (orders, expected receipts, lot moves): shown only when FL lists any. */
  function otherReceiptsHtml(other, lang) {
    if (!other || !(other.rows || []).length) return '';
    return `<section class="section other" data-key="other_receipts"><div class="section-head"><span>${esc(localized(other, 'label', lang))}</span><span class="count">${esc(other.count)}</span></div>` +
      other.rows.map(r => receiptRowHtml(r, lang)).join('') + '</section>';
  }

  function directMakesHtml(block, lang) {
    if (!block || !(block.rows || []).length) return '';
    return `<section class="section" data-key="direct_make_substitutions"><div class="section-head"><span>${esc(localized(block, 'label', lang))}</span><span class="count">${esc(block.count)}</span></div>` +
      block.rows.map(row => `<div class="row"><div>${esc(localized(row, 'line', lang))}</div></div>`).join('') + '</section>';
  }

  function summaryHtml(doc, lang) { return (doc.sections || []).map(s => sectionHtml(s, lang)).join('') + otherReceiptsHtml(doc.other_receipts, lang) + directMakesHtml(doc.direct_make_substitutions, lang); }

  /** The confirmation banner text for the latest row, or null when unconfirmed. */
  function bannerText(doc, lang) {
    const c = doc.confirmation || {};
    if (!c.latest) return null;
    const when = `${c.latest.confirmed_at.slice(0, 10)} ${clock(c.latest.confirmed_at)}`;
    const who = c.latest.actor && c.latest.actor.name ? c.latest.actor.name : '';
    if (c.status === 'discrepancy') {
      return lang === 'es' ? `Marcado como "falta algo" por ${who} el ${when}${c.latest.notes ? ' — ' + c.latest.notes : ''}` : `Marked "something is missing" by ${who} at ${when}${c.latest.notes ? ' — ' + c.latest.notes : ''}`;
    }
    return lang === 'es' ? `Confirmado por ${who} el ${when}${c.latest.notes ? ' — ' + c.latest.notes : ''}` : `Confirmed by ${who} at ${when}${c.latest.notes ? ' — ' + c.latest.notes : ''}`;
  }

  const exports = { apiBase, errorText, staleSummary, confirmBody, receiptRowHtml, sectionHtml, summaryHtml, otherReceiptsHtml, directMakesHtml, bannerText, PRODUCTION_API };
  if (typeof module !== 'undefined') module.exports = exports;
  if (!root.document) return;

  // ------------------------------------------------------------------ DOM
  const $ = id => document.getElementById(id);
  const API = apiBase(location, location.search);
  let language = localStorage.getItem('fl-shift-language') || 'en';
  // `generation` stamps every request with the session/view it was made for: a response that
  // comes back after a sign-out, a new sign-in or a day change is dropped, never rendered.
  // Read-only holds this marker, never the key: every request re-reads the scoped
  // dashboard key (FLDashboard.key, 60 s cache) so a rotation reaches open views (O-7).
  const READ_ONLY = '\u0000read-only';
  let key = '', actor = null, doc = null, busy = false, generation = 0, pendingNotes = null;
  let session = null, sharedMode = false;
  // Same token store and exchange as the dashboard/office pages; personal keys
  // are never retained after a session-enabled sign-in.
  const sessionReady = root.FLKeySessions ? (async () => {
    session = root.FLKeySessions.create({ base: API, fetch: root.fetch.bind(root), storage: root.sessionStorage,
      config: root.FLDashboard?.config, changed: state => {
        if (sharedMode && !state.signedIn && key) resetView();
      } });
    sharedMode = await session.init();
    return sharedMode;
  })() : Promise.resolve(false);
  sessionReady.catch(() => {});

  function t(en, es) { return language === 'es' ? es : en; }
  function applyLanguage() {
    document.documentElement.lang = language;
    document.querySelectorAll('[data-en]').forEach(node => node.textContent = node.dataset[language]);
    $('language').textContent = language === 'en' ? 'ES' : 'EN';
    $('language').setAttribute('aria-label', language === 'en' ? 'Cambiar a español' : 'Switch to English');
    $('actor-key').placeholder = t('Enter your personal key', 'Escribe tu clave personal');
    if (actor) $('identity').textContent = `${t('Signed in as', 'Sesión de')} ${actor.name} · ${actor.role}`;
    else if (key) $('identity').textContent = t('Read-only view of all actors (shared key).', 'Vista de solo lectura de todos (clave compartida).');
    else $('identity').textContent = t('Sign in with your personal key, or continue read-only.', 'Inicia sesión con tu clave personal o continúa en modo lectura.');
  }
  function notice(message, ok) { $('notice').textContent = message || ''; $('notice').hidden = !message; $('notice').classList.toggle('ok', !!ok); }

  /** The fixed footer's real height drives the page's bottom padding (nothing hides under it). */
  function fitFooter() {
    const bar = $('confirm-bar');
    const height = bar.hidden ? 0 : (bar.offsetHeight || 0);
    document.documentElement.style.setProperty('--footer-h', `${height}px`);
  }
  if (typeof ResizeObserver === 'function') new ResizeObserver(fitFooter).observe($('confirm-bar'));

  async function api(path, { body, method = 'GET', params, withKey, background = false } = {}) {
    const url = new URL(path, API);
    const credential = withKey === undefined ? key : withKey;
    const requestKey = credential === READ_ONLY ? await root.FLDashboard.key(API,language) : credential;
    Object.entries(params || {}).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v); });
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 60000);
    try {
      const response = await fetch(url, { method, headers: { 'X-API-Key': requestKey, ...(background ? { 'X-FL-Background': '1' } : {}), ...(body ? { 'Content-Type': 'application/json' } : {}) },
        body: body ? JSON.stringify(body) : undefined, signal: controller.signal, cache: 'no-store' });
      const data = await Promise.resolve().then(()=>response.json()).catch(error=>{if([401,403].includes(response.status))return {};throw error;});
      if (response.status === 401 && data.detail?.error_code === 'SESSION_EXPIRED' && session?.token && requestKey === session.token) {
        session.clear();
        notice(errorText(data, language));
      }
      root.FLDashboard.checkReadAccess(response,{method,headers:{'X-API-Key':requestKey}},language,API);
      if (!response.ok) throw { ...data, httpStatus: response.status };
      return data;
    } catch (error) {
      if (error?.accessUnavailable) throw error;
      if (error instanceof Error) throw { detail: { error_code: 'NETWORK', message: 'Connection failed or timed out.', message_es: 'La conexión falló o tardó demasiado.' } };
      throw error;
    } finally { clearTimeout(timer); }
  }

  function plantToday() {
    const parts = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).formatToParts(new Date());
    const get = type => parts.find(p => p.type === type).value;
    return `${get('year')}-${get('month')}-${get('day')}`;
  }
  function shiftDate(days) {
    const d = new Date(plantToday() + 'T12:00:00Z'); d.setUTCDate(d.getUTCDate() + days); return d.toISOString().slice(0, 10);
  }

  /** A request made for (generation, key) is current only while both are unchanged. */
  function stamp() { return { ticket: ++generation, key }; }
  function current(s) { return s.ticket === generation && s.key === key; }

  /** Notes belong to the person and the day they were opened for; any change closes them. */
  function closeNotes() { pendingNotes = null; $('notes-form').hidden = true; $('notes').value = ''; fitFooter(); }

  function render() {
    if (!doc) return;
    $('title').textContent = doc.title || doc.title_en;
    $('receipts-count').textContent = `${t('FL receipts today', 'Recibos FL del día')}: ${doc.receipts_today}`;
    $('sections').innerHTML = summaryHtml(doc, language);
    const banner = bannerText(doc, language);
    $('status-banner').hidden = !banner; $('status-banner').textContent = banner || '';
    $('status-banner').className = 'banner ' + (doc.confirmation && doc.confirmation.status !== 'unconfirmed' ? doc.confirmation.status : '');
    const can = !!(doc.confirmation && doc.confirmation.can_confirm);
    $('confirm-bar').hidden = false;
    $('confirm-question').hidden = !can;
    $('confirm-reason').hidden = can; $('confirm-reason').textContent = can ? '' : ((doc.confirmation && (doc.confirmation.reason || doc.confirmation.reason_en)) || '');
    ['confirm-match', 'confirm-notes', 'confirm-missing'].forEach(id => { $(id).hidden = !can; $(id).disabled = busy; });
    if (!can || (pendingNotes && (pendingNotes.date !== doc.date || pendingNotes.key !== key))) closeNotes();
    fitFooter();
  }

  /** Nothing is shown and nothing can be confirmed until a summary for this person + the selected day loads. */
  function clearSummary() {
    doc = null; closeNotes();
    $('sections').innerHTML = ''; $('title').textContent = ''; $('receipts-count').textContent = ''; $('status-banner').hidden = true;
    $('confirm-bar').hidden = true; fitFooter();
  }
  function lockConfirm() { ['confirm-match', 'confirm-notes', 'confirm-missing'].forEach(id => { $(id).disabled = true; }); }

  async function load({ background = false } = {}) {
    const s = stamp(), date = $('date').value;
    if (pendingNotes && pendingNotes.date !== date) closeNotes();
    busy = true; lockConfirm(); $('load-status').textContent = t('Loading…', 'Cargando…'); notice('');
    try {
      const result = await api('/reports/shift-summary', { params: { date, lang: language }, withKey: s.key, background });
      if (!current(s)) return false;                 // superseded by a sign-out / new sign-in / newer load
      doc = result; $('load-status').textContent = '';
      busy = false; render();
      return true;
    } catch (error) {
      if (!current(s)) return false;
      // A failed load leaves nothing to confirm: the previous day's document is cleared so the
      // picker, the page and any confirm always agree (a confirm for the wrong day is impossible).
      busy = false; $('load-status').textContent = ''; clearSummary(); notice(errorText(error, language));
      return false;
    }
  }

  async function sendConfirm(outcome, notes) {
    if (!doc || busy || doc.date !== $('date').value) return;   // only the document shown for the selected day
    const s = stamp(), shown = doc;
    busy = true; render();
    try {
      const result = await api('/reports/shift-summary/confirm', { method: 'POST', body: confirmBody(shown, outcome, notes, language), withKey: s.key });
      if (!current(s)) return;
      busy = false; closeNotes();
      const message = result.message || result.message_en;
      if (await load() && key === s.key) notice(message, true);   // FL's message stays up on the reloaded page
    } catch (error) {
      if (!current(s)) return;
      busy = false;
      const fresh = staleSummary(error);
      if (fresh) {
        doc = fresh; render();
        notice(t('The summary changed since it was shown — review it and confirm again.', 'El resumen cambió desde que se mostró — revísalo y confirma otra vez.'));
      } else { render(); notice(errorText(error, language)); }
    }
  }

  function openNotes(outcome) {
    if (!doc) return;
    pendingNotes = { outcome, date: doc.date, key };
    $('notes-label').textContent = outcome === 'discrepancy' ? t('What is missing? (required)', '¿Qué falta? (obligatorio)') : t('Notes (optional)', 'Notas (opcional)');
    $('notes-form').hidden = false; $('notes').focus(); fitFooter();
  }

  function resetView() {
    generation += 1;                                 // every in-flight request for the old person is dropped
    key = ''; actor = null; busy = false; clearSummary();
    $('summary-panel').hidden = true; $('sign-in-panel').hidden = false; $('sign-out').hidden = true;
    applyLanguage(); notice(''); fitFooter();
  }

  async function signOut() {
    // Invalidate requests immediately, including while the server logout runs.
    const s = stamp(); busy = true; lockConfirm(); closeNotes();
    try {
      await sessionReady;
      if (!current(s)) return;
      if (sharedMode && session.token) await session.signOut();
      else { if (sharedMode) session.clear(); resetView(); }
    } catch (error) {
      if (!current(s)) return;
      busy = false; render(); notice(t('Sign out failed. Try again before handing over this device.', 'No se pudo cerrar la sesión. Inténtalo otra vez antes de entregar este dispositivo.'));
    }
  }

  function showIdentity(person, credential) {
    key = credential; actor = person;
    $('sign-in-panel').hidden = true; $('summary-panel').hidden = false; $('sign-out').hidden = false; $('actor-key').value = '';
    applyLanguage();
  }

  async function signIn(personalKey) {
    resetView();
    const s = stamp();
    try {
      await sessionReady;
      if (!current(s)) return;
      if (sharedMode && personalKey) {
        const person = await session.signIn(personalKey);
        if (!current(s)) return;
        showIdentity(person, session.token); await load(); return;
      }
      if (sharedMode) session.clear(); // read-only cancels any pending personal exchange
      key = personalKey || READ_ONLY;
      s.key = key;
      const who = await api('/auth/whoami', { withKey: s.key });
      if (!current(s)) return;
      showIdentity(who.actor || null, s.key); await load();
    } catch (error) { if (!current(s)) return; key = ''; actor = null; notice(errorText(error, language)); }
  }

  $('sign-in-form').addEventListener('submit', event => { event.preventDefault(); const v = $('actor-key').value.trim(); $('actor-key').value = ''; if (v) signIn(v); else notice(t('Enter your key or choose read-only.', 'Escribe tu clave o elige solo lectura.')); });
  $('read-only').addEventListener('click', () => signIn(''));
  $('sign-out').addEventListener('click', signOut);
  $('language').addEventListener('click', () => { language = language === 'en' ? 'es' : 'en'; localStorage.setItem('fl-shift-language', language); applyLanguage(); if (key) load(); });
  $('date').max = plantToday(); $('date').value = plantToday();
  $('date').addEventListener('change', () => { if ($('date').value > plantToday()) $('date').value = plantToday(); closeNotes(); load(); });
  $('today').addEventListener('click', () => { $('date').value = plantToday(); closeNotes(); load(); });
  $('yesterday').addEventListener('click', () => { $('date').value = shiftDate(-1); closeNotes(); load(); });
  $('refresh').addEventListener('click', load);
  $('confirm-match').addEventListener('click', () => sendConfirm('match', ''));
  $('confirm-notes').addEventListener('click', () => openNotes('match'));
  $('confirm-missing').addEventListener('click', () => openNotes('discrepancy'));
  $('notes-cancel').addEventListener('click', closeNotes);
  $('notes-form').addEventListener('submit', event => {
    event.preventDefault();
    const notes = $('notes').value;                  // sent verbatim (contract §3.1); only the blank check trims
    if (!pendingNotes || !doc || pendingNotes.date !== doc.date || pendingNotes.date !== $('date').value || pendingNotes.key !== key) {
      closeNotes(); notice(t('The day or the person changed — open the notes again.', 'Cambió el día o la persona — abre las notas otra vez.')); return;
    }
    if (pendingNotes.outcome === 'discrepancy' && !notes.trim()) { notice(t('Say what is missing so the owner can act on it.', 'Indica qué falta para que el dueño pueda actuar.')); return; }
    sendConfirm(pendingNotes.outcome, notes);
  });
  applyLanguage(); fitFooter();
  const initialGeneration = generation;
  sessionReady.then(enabled => {
    if (generation !== initialGeneration || !enabled || !session.actor) return;
    showIdentity(session.actor, session.token); load({ background: true });
  }).catch(error => notice(errorText(error, language)));
})(typeof window !== 'undefined' ? window : globalThis);
