/* F1: DOM cards come from FL responses. Model prose is never a UI input. */
(function (root) {
  'use strict';
  const validReceipt = value => !!(value && value.success !== false && typeof value.receipt_number === 'string' && /^(RCV|MK|PK|ADJ|FND)-\d{6}-\d{3,}$/.test(value.receipt_number));
  const canRecord = prepared => !!(prepared && (prepared.can_commit || (prepared.blockers?.length && prepared.blockers.every(b => b.code === 'LOT_NOT_CONFIRMED'))));
  function errorText(value) {
    const data = value?.result || value;
    const detail = data?.detail || data?.error_detail || data;
    if (Array.isArray(detail)) return detail.map(e => `${(e.loc || []).slice(1).join('.')}: ${e.msg}`).join('\n');
    if (typeof detail === 'string') return detail;
    if (detail && typeof detail === 'object') return [detail.error_code || detail.code, detail.message, detail.action, detail.role].filter(Boolean).join(' · ') || JSON.stringify(detail);
    return String(detail || 'Request failed / Solicitud fallida');
  }
  const exports = { validReceipt, canRecord, errorText };
  if (typeof module !== 'undefined') module.exports = exports;
  if (!root.document) return;
  const $ = id => document.getElementById(id);
  let language = localStorage.getItem('fl-assistant-language') || 'en';
  let key = '', session = '', actor = null, sessionStore = '', busy = false, attachments = [], retryTurn = null;
  let held = false, recorder = null, stream = null, recordingTimer = null, micStarting = false;
  const drafts = new Map();
  const labels = new Map();
  function t(en, es) { labels.set(en, [en, es]); labels.set(es, [en, es]); return language === 'es' ? es : en; }
  function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined) node.textContent = text;
    if (labels.has(text)) { const [en, es] = labels.get(text); node.dataset.en = en; node.dataset.es = es; }
    if (className) node.className = className;
    return node;
  }
  function button(label, handler, className) {
    const node = el('button', label, className); node.type = 'button';
    node.addEventListener('click', handler); return node;
  }
  function applyLanguage() {
    document.documentElement.lang = language;
    document.querySelectorAll('[data-en]').forEach(node => node.textContent = node.dataset[language]);
    $('language').textContent = language === 'en' ? 'ES' : 'EN';
    $('language').setAttribute('aria-label', language === 'en' ? 'Cambiar a español' : 'Switch to English');
    $('actor-key').placeholder = t('Enter your personal key', 'Escribe tu clave personal');
    if (actor) $('identity').textContent = `${t('Signed in as', 'Sesión de')} ${actor.name} · ${actor.role}`;
  }
  function notice(message) { $('notice').textContent = message; $('notice').hidden = !message; }
  function failed(node, error) {
    const box = el('div', t('NOT recorded — no receipt confirmed.\n', 'NO registrado — no se confirmó un recibo.\n') + errorText(error), 'error-box');
    box.setAttribute('role', 'alert'); node.append(box); return box;
  }
  async function api(path, { body, form, method = 'POST', binary = false, timeout = 95000 } = {}) {
    const controller = new AbortController(), timer = setTimeout(() => controller.abort(), timeout);
    try {
      const response = await fetch(new URL(path, location.origin), {
        method, headers: { 'X-API-Key': key, ...(form || !body ? {} : { 'Content-Type': 'application/json' }) },
        body: form || (body ? JSON.stringify(body) : undefined), signal: controller.signal, credentials: 'same-origin', cache: 'no-store'
      });
      if (binary && response.ok) return response.blob();
      const data = await response.json();
      if (!response.ok) throw { ...data, httpStatus: response.status };
      return data;
    } catch (error) {
      if (error instanceof Error) throw { detail: t('Connection failed or timed out. Retry the same draft.', 'La conexión falló. Reintenta el mismo borrador.') };
      throw error;
    } finally { clearTimeout(timer); }
  }
  function setBusy(value) {
    busy = value;
    ['send', 'photo-button', 'new-chat'].forEach(id => $(id).disabled = value || !session);
    $('dictate').disabled = value || !session || !navigator.mediaDevices?.getUserMedia || !root.MediaRecorder;
    $('sign-out').disabled = value;
  }
  function userMessage(text) { if (text) $('messages').append(el('div', text, 'message user-message')); }
  function shell() {
    $('welcome').hidden = true;
    const wrap = el('div', undefined, 'message assistant-message');
    const glyph = el('span', '✧', 'assistant-glyph'); glyph.setAttribute('aria-hidden', 'true');
    const card = el('article', undefined, 'card'); wrap.append(glyph, card); $('messages').append(wrap); return card;
  }
  function details(node, result) {
    const disclosure = el('details'); disclosure.append(el('summary', t('FL details', 'Detalles de FL')), el('pre', JSON.stringify(result, null, 2))); node.append(disclosure);
  }
  function fields(node, pairs) {
    const list = el('dl', undefined, 'field-list');
    pairs.filter(([, value]) => value !== undefined && value !== null && value !== '').forEach(([name, value]) => {
      list.append(el('dt', name), el('dd', typeof value === 'object' ? JSON.stringify(value) : String(value)));
    }); node.append(list);
  }
  function renderReceipt(node, result) {
    if (!validReceipt(result)) { failed(node, t('FL returned no receipt number.', 'FL no devolvió un número de recibo.')); return; }
    node.replaceChildren(); node.className = 'card receipt';
    node.append(el('span', t('✓ RECORDED IN FACTORY LEDGER', '✓ REGISTRADO EN FACTORY LEDGER'), 'eyebrow'), el('h3', result.receipt_number, 'receipt-number'));
    node.append(el('p', result.replayed ? t('Same entry, same receipt. Your retry did not create another entry.', 'Misma entrada, mismo recibo. El reintento no creó otra entrada.') : t('FL confirmed this entry.', 'FL confirmó esta entrada.')));
    fields(node, [[t('Transaction', 'Transacción'), result.transaction_id], [t('Product', 'Producto'), result.product_name || result.target_product_name], [t('Lot', 'Lote'), result.lot_code || result.output_lot_code]]);
    details(node, result);
  }
  function renderDraft(card, existing) {
    const node = existing || shell(), prepared = card.prepared, draft = prepared.draft;
    drafts.set(card.id, node); node.dataset.draftId = card.id;
    node.append(el('span', t('DRAFT · NOT RECORDED', 'BORRADOR · NO REGISTRADO'), 'draft-badge'));
    const actionNames = { receive: ['Receive', 'Recibir'], make: ['Make', 'Producir'], pack: ['Pack', 'Empacar'], adjust: ['Adjust', 'Ajustar'], found: ['Found inventory', 'Inventario encontrado'] };
    node.append(el('h3', t(...(actionNames[prepared.action] || [prepared.action, prepared.action]))));
    node.append(el('p', t(`Recording as ${prepared.actor?.name || actor.name}`, `Registrando como ${prepared.actor?.name || actor.name}`), 'muted'));
    fields(node, [[t('Product', 'Producto'), draft.product_name || draft.target_product_name], [t('Source', 'Origen'), draft.source_product_name],
      [t('Supplier', 'Proveedor'), draft.supplier_name], [t('Cases', 'Cajas'), draft.cases], [t('Batches', 'Lotes de producción'), draft.batches],
      [t('Pounds', 'Libras'), draft.total_lb ?? draft.total_output_lb ?? draft.total_weight_lb ?? draft.quantity],
      [t('Adjustment (lb)', 'Ajuste (lb)'), draft.delta_lb], [t('Lot', 'Lote'), draft.lot_code || draft.output_lot_code],
      [t('Supplier lot', 'Lote del proveedor'), draft.supplier_lot_code], [t('BOL', 'BOL'), draft.bol_reference],
      [t('Reason', 'Motivo'), language === 'es' ? draft.reason_es || draft.reason_code : draft.reason_code],
      [t('Happened at', 'Ocurrió el'), draft.happened_at], [t('Expires', 'Vence'), prepared.expires_at]]);
    const confirmations = [], warningChecks = [];
    for (const lot of draft.input_plan || []) {
      const row = el('div', undefined, 'lot-confirmation');
      const lotLabel = `${lot.product_name || t('Input', 'Insumo')} · ${lot.lot_code} · ${lot.quantity_lb} lb`;
      const title = el('label', lotLabel); row.append(title);
      if (lot.confirmed === false) {
        const controls = el('div', undefined, 'lot-controls'), method = el('select'), input = el('input');
        [['last4', t('Last 4 characters', 'Últimos 4 caracteres')], ['full_code', t('Full lot code', 'Código de lote completo')]].forEach(([value, label]) => {
          const option = el('option', label); option.value = value; method.append(option);
        });
        method.setAttribute('aria-label', t('Confirmation method', 'Método de confirmación'));
        input.type = 'text'; input.autocomplete = 'off'; input.maxLength = 200;
        input.id = 'lot-' + card.id + '-' + lot.lot_id; title.htmlFor = input.id;
        input.placeholder = t('Read the physical tag', 'Lee la etiqueta física');
        controls.append(method, input); row.append(controls);
        confirmations.push({ lot, method, input });
      } else row.append(el('small', lot.confirmed ? t('Confirmed by FL', 'Confirmado por FL') : t('Suggested by FL', 'Sugerido por FL'), 'muted'));
      node.append(row);
    }
    for (const warning of prepared.warnings || []) {
      const row = el('div', undefined, 'warning'), message = language === 'es' ? warning.message_es || warning.message : warning.message;
      if (warning.requires_ack) {
        const label = el('label'), checkbox = el('input'); checkbox.type = 'checkbox'; label.append(checkbox, el('span', message || warning.code)); row.append(label); warningChecks.push({ warning, checkbox });
      } else row.textContent = message || warning.code;
      node.append(row);
    }
    for (const blocker of prepared.blockers || []) {
      node.append(el('div', (language === 'es' ? blocker.message_es : null) || blocker.message || blocker.code, blocker.code === 'LOT_NOT_CONFIRMED' ? 'warning' : 'error-box'));
    }
    for (const aid of card.attachment_ids || []) node.append(button(t('View attached photo', 'Ver foto adjunta'), () => viewPhoto(aid)));
    details(node, draft);
    const errorArea = el('div'), actions = el('div', undefined, 'card-actions');
    if ((prepared.blockers || []).some(b => b.code === 'SKU_CONFIRMATION_REQUIRED')) {
      actions.append(button(t('Confirm this output SKU', 'Confirmar este producto de salida'), async () => {
        if (busy) return; setBusy(true); errorArea.replaceChildren();
        try {
          const response = await api('/assistant/confirm-sku', { body: { draft_id: card.id } });
          node.replaceChildren(el('p', t('Output SKU confirmed. Review the new draft below.', 'Producto confirmado. Revisa el nuevo borrador.'))); renderCard(response.card);
        } catch (error) { failed(errorArea, error); }
        finally { setBusy(false); }
      }));
    }
    const recordButton = button(t('Record', 'Registrar'), async () => {
      if (busy) return;
      errorArea.replaceChildren(); setBusy(true); recordButton.disabled = cancelButton.disabled = true;
      const recordBody = { draft_id: card.id, acknowledged_warnings: warningChecks.filter(w => w.checkbox.checked).map(w => w.warning.code),
        lot_confirmations: confirmations.map(c => ({ lot_id: c.lot.lot_id, method: c.method.value, value: c.input.value.trim() })) };
      try {
        const response = await api('/assistant/record', { body: recordBody });
        if (response.kind !== 'receipt' || !validReceipt(response.result)) throw response;
        renderReceipt(node, response.result);
      } catch (error) { failed(errorArea, error); errorArea.append(el('p', t('Try Record again to check the same ticket. You can also check today’s entries.', 'Pulsa Registrar otra vez para verificar el mismo ticket. También puedes consultar las entradas de hoy.'), 'muted')); }
      finally { setBusy(false); cancelButton.disabled = false; refreshRecord(); }
    }, 'primary');
    const cancelButton = button(t('Cancel', 'Cancelar'), async () => {
      if (busy) return; setBusy(true); recordButton.disabled = cancelButton.disabled = true; errorArea.replaceChildren();
      try { await api('/assistant/cancel', { body: { draft_id: card.id } }); node.replaceChildren(el('p', t('Cancelled · NOT recorded', 'Cancelado · NO registrado'))); }
      catch (error) { failed(errorArea, error); }
      finally { setBusy(false); cancelButton.disabled = false; refreshRecord(); }
    });
    function refreshRecord() { recordButton.disabled = busy || !canRecord(prepared) || warningChecks.some(w => !w.checkbox.checked) || confirmations.some(c => !c.input.value.trim()); }
    [...confirmations.map(c => c.input), ...warningChecks.map(w => w.checkbox)].forEach(input => input.addEventListener('input', refreshRecord));
    refreshRecord(); actions.append(recordButton, cancelButton); node.append(errorArea, actions);
  }
  function renderCard(card) {
    if (card.kind === 'draft') { renderDraft(card); return; }
    const node = shell();
    if (card.kind === 'choices') {
      node.append(el('span', t('CHOOSE FROM FL', 'ELIGE EN FL'), 'eyebrow'), el('h3', t('Which one?', '¿Cuál?')));
      node.append(el('p', card.result.ask || t('Choose the matching record.', 'Elige el registro correcto.')));
      const choices = [];
      function select(selectedId, nextPage = false) {
        if (busy) return;
        choices.forEach(b => b.disabled = true);
        sendTurn({ text: '', choice_id: card.id, selected_id: selectedId, next_page: nextPage }).finally(() => choices.forEach(b => b.disabled = false));
      }
      for (const candidate of card.result.candidates || []) {
        const b = button(candidate.label || candidate.name, () => select(candidate.id), 'choice-button'); choices.push(b); node.append(b);
      }
      const none = button(t('None of these', 'Ninguno de estos'), () => select(null), 'choice-button'); choices.push(none); node.append(none);
      if (card.result.has_more) { const more = button(t('More choices →', 'Más opciones →'), () => select(null, true), 'choice-button'); choices.push(more); node.append(more); }
    } else if (card.kind === 'read') {
      node.append(el('span', t('FROM FACTORY LEDGER', 'DE FACTORY LEDGER'), 'eyebrow'), el('h3', t('Here’s what FL shows', 'Esto muestra FL')));
      const receipts = card.result.receipts;
      if (receipts) {
        if (!receipts.length) node.append(el('p', t('No receipts for you today.', 'No hay recibos tuyos hoy.')));
        receipts.forEach(r => {
          const line = el('div'); line.append(el('p', `${r.receipt_number} · ${r.action || ''}`)); details(line, r); node.append(line);
        });
      } else if (card.tool === 'inventory_lookup') {
        const results = card.result.results || [];
        if (!results.length) node.append(el('p', t('No inventory matches.', 'No hay coincidencias de inventario.')));
        for (const item of results) { const section = el('div'); section.append(el('p', item.product_name || item.name || item.product?.name || t('Inventory', 'Inventario'))); details(section, item); node.append(section); }
      } else details(node, card.result);
    } else if (card.kind === 'refusal') {
      failed(node, card.result);
      node.append(el('p', t('Update the message with the requested details. FL has not recorded this request.', 'Completa el mensaje con los datos solicitados. FL no ha registrado esta solicitud.'), 'muted'));
      if (JSON.stringify(card.result).includes('reason_code')) {
        for (const reason of card.reasons || []) node.append(button(language === 'es' ? reason.label_es : reason.label_en, () => {
          $('message').value = `${t('The correction reason is', 'El motivo de corrección es')} ${reason.code}. `; $('message').focus();
        }, 'choice-button'));
      }
    } else node.append(el('p', card.message || t('Please give more details.', 'Agrega más detalles.')));
  }
  async function sendTurn(values, original) {
    if (busy) return;
    const body = original || { session_id: session, turn_id: crypto.randomUUID(), attachment_ids: attachments.map(a => a.id), ...values };
    setBusy(true); notice('');
    if (!original) userMessage(body.text);
    const waiting = el('p', t('Checking with FL…', 'Consultando FL…'), 'busy-note'); $('messages').append(waiting);
    try {
      const result = await api('/assistant/turn', { body, timeout: 570000 });
      result.cards.forEach(renderCard);
      retryTurn = null;
      if (result.cards.some(c => c.kind === 'draft')) { attachments = []; renderAttachments(); }
    } catch (error) {
      retryTurn = body; const card = shell(); failed(card, error);
      card.append(button(t('Retry this message', 'Reintentar este mensaje'), () => sendTurn({}, body)));
    } finally { waiting.remove(); setBusy(false); $('message').focus(); }
  }
  async function loadChat() {
    const saved = localStorage.getItem(sessionStore);
    let result;
    if (saved) {
      try { result = await api('/assistant/resume', { body: { session_id: saved } }); session = saved; }
      catch (error) { if (error.httpStatus !== 404) throw error; }
    }
    if (!session) { result = await api('/assistant/session'); session = result.session_id; localStorage.setItem(sessionStore, session); }
    for (const turn of result.turns || []) { userMessage(turn.text); turn.cards.forEach(renderCard); }
    // A saved receipt is a real commit result; no text/receipt lookup can call this renderer.
    for (const draft of result.drafts || []) {
      let node = drafts.get(draft.id);
      if (!node) { renderDraft(draft.card); node = drafts.get(draft.id); }
      if (draft.status === 'committed' && validReceipt(draft.result)) renderReceipt(node, draft.result);
      if (draft.status === 'cancelled') node.replaceChildren(el('p', t('Cancelled · NOT recorded', 'Cancelado · NO registrado')));
    }
  }
  $('sign-in-form').addEventListener('submit', async event => {
    event.preventDefault(); notice(''); key = $('actor-key').value.trim();
    const submit = event.target.querySelector('button'); submit.disabled = true;
    try {
      const who = await api('/auth/whoami', { method: 'GET' });
      if (!who.actor) throw { detail: t('Use your personal actor key.', 'Usa tu clave personal.') };
      actor = who.actor; sessionStore = 'fl-assistant-chat:' + (actor.id || actor.name);
      await loadChat(); $('actor-key').value = ''; $('sign-in-panel').hidden = true; $('chat-panel').hidden = false; $('sign-out').hidden = false;
      applyLanguage(); setBusy(false); $('message').focus();
    } catch (error) { key = ''; notice(errorText(error)); }
    finally { submit.disabled = false; }
  });
  $('sign-out').addEventListener('click', () => {
    if (busy) return; stopMic(); key = ''; session = ''; actor = null; location.reload();
  });
  $('new-chat').addEventListener('click', async () => {
    if (busy) return; setBusy(true);
    try { const result = await api('/assistant/session'); localStorage.setItem(sessionStore, result.session_id); location.reload(); }
    catch (error) { notice(errorText(error)); setBusy(false); }
  });
  $('language').addEventListener('click', () => { language = language === 'en' ? 'es' : 'en'; localStorage.setItem('fl-assistant-language', language); applyLanguage(); });
  $('today-entries').addEventListener('click', () => { if (session) sendTurn({ text: t('What did I enter today?', '¿Qué registré hoy?') }); else $('actor-key').focus(); });
  $('composer').addEventListener('submit', event => { event.preventDefault(); const text = $('message').value.trim(); if (text && !busy) { $('message').value = ''; sendTurn({ text }); } });
  $('message').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('composer').requestSubmit(); } });
  document.querySelectorAll('[data-prompt]').forEach(b => b.addEventListener('click', () => {
    $('message').value = language === 'es' ? (b.dataset.prompt.startsWith('What') ? '¿Qué registré hoy?' : 'Consulta el inventario de ') : b.dataset.prompt; $('message').focus();
  }));
  function renderAttachments() {
    $('attachments').replaceChildren();
    for (const a of attachments) { const chip = el('div', undefined, 'attachment-chip'); chip.append(el('span', a.filename), button('×', () => { attachments = attachments.filter(x => x.id !== a.id); renderAttachments(); })); $('attachments').append(chip); }
  }
  async function viewPhoto(id) {
    try {
      const blob = await api('/assistant/attachment/read', { body: { attachment_id: id }, binary: true });
      const dialog = el('dialog'), image = el('img'); const url = URL.createObjectURL(blob); image.src = url; image.alt = t('Attached photo', 'Foto adjunta'); image.style.maxWidth = 'min(85vw,800px)'; image.style.maxHeight = '75vh';
      dialog.append(image, button(t('Close', 'Cerrar'), () => dialog.close())); document.body.append(dialog); dialog.addEventListener('close', () => { URL.revokeObjectURL(url); dialog.remove(); }); dialog.showModal();
    } catch (error) { notice(errorText(error)); }
  }
  $('photo-button').addEventListener('click', () => $('photo').click());
  $('photo').addEventListener('change', async () => {
    const file = $('photo').files[0]; $('photo').value = ''; if (!file || busy) return;
    if (attachments.length >= 5) { notice(t('Send at most five photos with a message.', 'Envía hasta cinco fotos con un mensaje.')); return; }
    setBusy(true); notice(''); const form = new FormData(); form.append('file', file);
    try { const result = await api('/assistant/attachment?session_id=' + encodeURIComponent(session), { form }); attachments.push(result); renderAttachments(); }
    catch (error) { notice(t('NOT recorded — photo upload failed. ', 'NO registrado — falló la carga de foto. ') + errorText(error)); }
    finally { setBusy(false); }
  });
  async function startMic() {
    if (busy || held || micStarting || recorder) return;
    held = true; micStarting = true; setBusy(true); $('dictate').disabled = false; notice(''); $('dictate').setAttribute('aria-pressed', 'true');
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      if (!held) { stream.getTracks().forEach(track => track.stop()); stream = null; setBusy(false); return; }
      const mimeType = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/webm'].find(type => MediaRecorder.isTypeSupported(type));
      recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);
      const chunks = [], mime = recorder.mimeType;
      recorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      recorder.onstop = async () => {
        stream?.getTracks().forEach(track => track.stop()); stream = null; recorder = null; setBusy(true);
        const form = new FormData(); form.append('file', new Blob(chunks, { type: mime }), mime.includes('mp4') ? 'dictation.mp4' : 'dictation.webm');
        try {
          const result = await api('/assistant/transcribe', { form });
          $('message').value = [$('message').value.trim(), result.text].filter(Boolean).join(' ');
          $('composer-hint').textContent = t('Review and edit your transcript, then press Send. Nothing has been sent.', 'Revisa y edita el dictado. Después pulsa Enviar. No se ha enviado nada.');
        } catch (error) { notice(t('NOT recorded — dictation failed. ', 'NO registrado — falló el dictado. ') + errorText(error)); }
        finally { setBusy(false); $('message').focus(); }
      };
      recorder.start(); recordingTimer = setTimeout(stopMic, 90000);
    } catch (error) { stream?.getTracks().forEach(track => track.stop()); stream = null; setBusy(false); notice(t('Microphone unavailable. Type your message instead.', 'Micrófono no disponible. Escribe tu mensaje.')); held = false; $('dictate').setAttribute('aria-pressed', 'false'); }
    finally { micStarting = false; }
  }
  function stopMic() {
    held = false; clearTimeout(recordingTimer); $('dictate').setAttribute('aria-pressed', 'false');
    if (recorder?.state === 'recording') recorder.stop();
  }
  $('dictate').addEventListener('pointerdown', event => { event.preventDefault(); $('dictate').setPointerCapture(event.pointerId); startMic(); });
  ['pointerup', 'pointercancel', 'lostpointercapture'].forEach(name => $('dictate').addEventListener(name, stopMic));
  $('dictate').addEventListener('keydown', event => { if ([' ', 'Enter'].includes(event.key) && !event.repeat) { event.preventDefault(); startMic(); } });
  $('dictate').addEventListener('keyup', event => { if ([' ', 'Enter'].includes(event.key)) { event.preventDefault(); stopMic(); } });
  root.addEventListener('blur', stopMic); document.addEventListener('visibilitychange', () => { if (document.hidden) stopMic(); });
  applyLanguage(); setBusy(false);
})(typeof window === 'undefined' ? globalThis : window);
