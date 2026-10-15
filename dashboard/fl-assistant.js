/* F1: DOM cards come from FL responses. Model prose is never a UI input. */
(function (root) {
  'use strict';
  const validReceipt = value => !!(value && value.success !== false && typeof value.receipt_number === 'string' && /^(RCV|MK|PK|ADJ|FND|SHP|CUS|ORD)-\d{6}-\d{3,}$/.test(value.receipt_number));
  const canRecord = prepared => !!(prepared && (prepared.blockers?.length ? prepared.blockers.every(b => b.code === 'LOT_NOT_CONFIRMED') : prepared.can_commit === true));
  const evidenceKinds = ['bol', 'truck_photo', 'correction_photo', 'lot_tag', 'count_sheet', 'signed_packing_slip', 'other'];
  const validAttachment = value => !!(value && Number.isSafeInteger(value.id) && value.id > 0 && evidenceKinds.includes(value.kind));
  const neutralMessages = {
    unsupported_request: ['I can only record or look up Factory Ledger items.',
      'Solo puedo registrar o consultar elementos de Factory Ledger.'],
    misunderstood: ['I didn’t understand that — please rephrase (e.g. “we packed 60 cases of Classic”).',
      'No entendí eso — reformula el mensaje (por ejemplo, “empacamos 60 cajas de Classic”).'],
    lookup_failed: ['Factory Ledger is temporarily unavailable — try again in a minute.', 'Factory Ledger no está disponible temporalmente — inténtalo de nuevo en un minuto.'],
    request_failed: ['Couldn’t complete that request. Please try again.', 'No pude completar esa solicitud. Inténtalo de nuevo.']
  };
  function errorText(value, language = 'en') {
    if (value?.name === 'SyntaxError') return display.unavailable(language);
    const data = value?.result || value;
    const detail = data?.detail || data?.error_detail || data;
    if (Array.isArray(detail) || data?.error_detail?.code === 'VALIDATION_ERROR') return neutralMessages.misunderstood.join(' / ');
    if (typeof detail === 'string') return display ? display.plainMessage(detail, language) : detail;
    if (detail && typeof detail === 'object') return display ? display.errorMessage(detail, language) : detail.message || 'FL could not complete this request. Please check the details.';
    return String(detail || 'Request failed / Solicitud fallida');
  }
  const awaitingApproval = value => !!(value && value.held === true && value.status === 'awaiting_approval');
  const display = root.FLAssistantDisplay || (typeof require === 'function' ? require('./assistant-display.js') : null);
  const isTodayQuery = text => /^(?:(?:open|show|list)(?: me)?(?: all)?(?: of)? (?:today[’']?s entries|entries (?:for |from )?today)|what (?:did|have) (?:i|we) (?:enter(?:ed)?|record(?:ed)?) today|today[’']?s entries|is everything there correct|open today[’']?s entries[?.!]?\s*is everything there correct|[¿]?(?:est[aá] todo|todo est[aá]) (?:ah[ií] )?correcto|entradas de hoy|(?:abre|abrir|mostrar|muestra|ver|lista)(?: todas)?(?: las)? entradas de hoy|[¿]?qu[eé] registr(?:[eé]|amos) hoy)[?.!¿]*$/i.test(String(text || '').trim());
  const voidTarget = text => /\b(void|anular|anula)\b/i.test(text || '') ? (String(text).match(/\b(?:RCV|MK|PK|ADJ|FND|SHP)-\d{6}-\d{3,}\b/i)?.[0]?.toUpperCase() || null) : null;
  const refusalDraft = prepared => !!prepared?.blockers?.some(b => !['LOT_NOT_CONFIRMED', 'SKU_CONFIRMATION_REQUIRED', 'BOL_REQUIRED'].includes(b.code)
    && !(prepared.action === 'create_customer' && display.existingCustomerChoice(b)));
  function supplierChoices(candidates = []) {
    const seen = new Set();
    return candidates.filter(candidate => {
      const key = String(candidate.name || candidate.label || '').normalize('NFKD').replace(/[\u0300-\u036f]/g, '').toLowerCase()
        .replace(/&/g, ' and ').replace(/[^a-z0-9]+/g, ' ').trim()
        .replace(/(?:\s+(?:incorporated|corporation|company|limited|inc|corp|llc|ltd|co))+$/, '')
        .replace(/\s+(?:supply|supplies)$/, '').replace(/\bbakery\b/g, 'baker');
      if (!key || seen.has(key)) return false;
      seen.add(key); return true;
    }).slice(0, 6);
  }
  const exports = { supplierChoices, isTodayQuery, voidTarget, refusalDraft, validReceipt, canRecord, validAttachment, evidenceKinds, errorText, awaitingApproval, neutralMessages };
  if (typeof module !== 'undefined') module.exports = exports;
  if (!root.document) return;
  const $ = id => document.getElementById(id);
  let language = localStorage.getItem('fl-assistant-language') || 'en';
  let credential = '', session = '', actor = null, sessionStore = '', busy = false, attachments = [], retryTurn = null;
  let lastIntent = '', held = false, recorder = null, stream = null, recordingTimer = null, micStarting = false;
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
    node.addEventListener('click', event => {
      const result = handler(event);
      if (result?.then) {
        const previous = node.textContent;
        node.className += ' is-working'; node.setAttribute('aria-busy', 'true');
        node.textContent = t('Working…', 'Trabajando…');
        result.finally(() => { node.className = node.className.replace(/\bis-working\b/g, '').trim(); node.setAttribute('aria-busy', 'false'); node.textContent = node.dataset[language] || previous; });
      }
    }); return node;
  }
  function applyLanguage() {
    document.documentElement.lang = language;
    document.querySelectorAll('[data-en]').forEach(node => node.textContent = node.dataset[language]);
    $('language').textContent = language === 'en' ? 'ES' : 'EN';
    $('language').setAttribute('aria-label', language === 'en' ? 'Cambiar a español' : 'Switch to English');
    $('actor-key').placeholder = t('Enter your personal key', 'Escribe tu clave personal');
    if (actor) { delete $('identity').dataset.en; delete $('identity').dataset.es; }
    if (actor) $('identity').textContent = `${t('Signed in as', 'Sesión de')} ${actor.name} · ${actor.role}`;
    renderAttachments();
  }
  function notice(message) { $('notice').textContent = message; $('notice').hidden = !message; }
  function failed(node, error) {
    const box = el('div', undefined, 'error-box');
    box.append(el('p', t('NOT recorded — no receipt confirmed.', 'NO registrado — no se confirmó un recibo.')), el('p', errorText(error, language)));
    box.setAttribute('role', 'alert'); node.append(box); return box;
  }
  function neutral(node, kind) {
    node.append(el('p', t(...neutralMessages[kind])));
  }
  function missingFacts(node, missing) {
    const names = {
      product_id: ['product', 'producto'], supplier_id: ['supplier', 'proveedor'], supplier_lot_code: ['supplier lot number', 'número de lote del proveedor'],
      cases: ['number of cases', 'número de cajas'], case_size_lb: ['case size (lb)', 'peso por caja (lb)'],
      bol_reference: ['BOL number', 'número de BOL'], batches: ['number of batches', 'número de lotes de producción'],
      ingredient_product_id: ['ingredient', 'ingrediente'], substitute_product_id: ['substitute ingredient', 'ingrediente sustituto'],
      lot_id: ['lot', 'lote'],
      source_product_id: ['bulk batch', 'lote a granel'], target_product_id: ['finished product', 'producto terminado'],
      delta_lb: ['weight change (lb)', 'cambio de peso (lb)'], reason_code: ['correction reason', 'motivo de la corrección'],
      quantity: ['quantity (lb)', 'cantidad (lb)'], order_id: ['order', 'pedido'],
      line_id: ['order item', 'producto del pedido'], quantity_lb: ['quantity (lb)', 'cantidad (lb)'],
      name: ['customer name', 'nombre del cliente'], customer_id: ['customer', 'cliente'], order_reference: ['order reference', 'referencia del pedido'],
      lines: ['order items and quantities', 'productos y cantidades del pedido']
    };
    node.append(el('p', t('Please add the missing details:', 'Agrega los datos que faltan:')));
    const list = el('ul');
    for (const field of new Set(missing || [])) list.append(el('li', t(...(names[field] || ['additional details', 'datos adicionales']))));
    node.append(list);
  }
  async function api(path, { body, form, method = 'POST', binary = false, timeout = 95000, background = false, onProgress } = {}) {
    const controller = new AbortController();
    let timedOut = false;
    const timer = setTimeout(() => { timedOut = true; controller.abort(); }, timeout);
    try {
      const response = await fetch(new URL(path, root.FLDashboard?.base||location.origin), {
        method, headers: { 'X-API-Key': credential, ...(onProgress ? { Accept: 'application/x-ndjson' } : {}), ...(background ? { 'X-FL-Background': '1' } : {}), ...(form || !body ? {} : { 'Content-Type': 'application/json' }) },
        body: form || (body ? JSON.stringify(body) : undefined), signal: controller.signal, credentials: 'same-origin', cache: 'no-store'
      });
      if (binary && response.ok) return response.blob();
      let data;
      if (response.ok && onProgress && response.headers.get('content-type')?.includes('application/x-ndjson')) {
        const reader = response.body.getReader(), decoder = new TextDecoder();
        let pending = '', finalResult;
        const consume = line => {
          if (!line.trim()) return;
          const event = JSON.parse(line);
          if (event.type === 'progress' && typeof event.message === 'string') onProgress(event.message);
          else if (event.type === 'error') throw { detail: event.detail, httpStatus: event.status };
          else if (event.type === 'result') finalResult = event.result;
        };
        while (true) {
          const chunk = await reader.read();
          pending += decoder.decode(chunk.value || new Uint8Array(), { stream: !chunk.done });
          const lines = pending.split('\n'); pending = lines.pop();
          for (const line of lines) consume(line);
          if (chunk.done) break;
        }
        consume(pending);
        if (!finalResult) throw new SyntaxError('Missing FL result');
        data = finalResult;
      } else data = await response.json();
      if (!response.ok) {
        if (data.detail?.error_code === 'SESSION_EXPIRED') {
          pilot.clear(); credential = ''; actor = null; session = '';
          drafts.clear(); attachments = []; retryTurn = null;
          const welcome = $('welcome'); $('messages').replaceChildren(welcome); welcome.hidden = false;
          renderAttachments();
          $('sign-in-panel').hidden = false; $('chat-panel').hidden = true; $('sign-out').hidden = true;
          $('identity').textContent = t('Sign in to your workspace', 'Inicia sesión en tu espacio');
          notice(t('Your session ended. Sign in again; your saved drafts remain bound to you.', 'Tu sesión terminó. Inicia sesión otra vez; tus borradores siguen a tu nombre.'));
        }
        throw { ...data, httpStatus: response.status };
      }
      return data;
    } catch (error) {
      if (timedOut) throw { timedOut: true, detail: t('Connection timed out. Please try again.', 'La conexión tardó demasiado. Inténtalo de nuevo.') };
      if (error instanceof SyntaxError || error?.name === 'SyntaxError') throw { detail: display.unavailable(language) };
      if (error instanceof Error) throw { detail: t('Connection failed. Please try again.', 'La conexión falló. Inténtalo de nuevo.') };
      throw error;
    } finally { clearTimeout(timer); }
  }
  function setBusy(value) {
    busy = value;
    if ($('working-status')) $('working-status').hidden = !value;
    $('send').setAttribute('aria-busy', String(value));
    $('send').className = value ? 'primary is-working' : 'primary';
    $('send').textContent = value ? t('Working…', 'Trabajando…') : t('Send ↑', 'Enviar ↑');
    ['send', 'photo-button', 'photo-kind', 'new-chat', 'today-entries', 'today-entries-mobile'].forEach(id => $(id).disabled = value || !session);
    $('dictate').disabled = value || !session || !navigator.mediaDevices?.getUserMedia || !root.MediaRecorder;
    $('sign-out').disabled = value;
    drafts.forEach(node => node.refreshRecord?.());
  }
  function userMessage(text) {
    if (!text) return;
    const search = String(text).match(/^\s*(?:supplier|proveedor)\s*:\s*(.*)$/i);
    const shown = search ? t('Supplier search: ' + search[1], 'Buscar proveedor: ' + search[1]) : text;
    $('messages').append(el('div', display.plainMessage(shown, language), 'message user-message'));
  }
  function shell() {
    $('welcome').hidden = true;
    const wrap = el('div', undefined, 'message assistant-message');
    const glyph = el('span', '✧', 'assistant-glyph'); glyph.setAttribute('aria-hidden', 'true');
    const card = el('article', undefined, 'card'); wrap.append(glyph, card); $('messages').append(wrap); return card;
  }
  function readResult(node, card) {
    const english = display.readSections(card, { language: 'en' }), spanish = display.readSections(card, { language: 'es' });
    for (const [index, section] of english.entries()) {
      const translated = spanish[index] || section;
      node.append(el('p', t(section.sentence, translated.sentence), 'read-summary'));
      for (const [i, note] of (section.notes || []).entries()) node.append(el('p', t(note, translated.notes?.[i] || note), 'muted'));
      if (!section.rows?.length) continue;
      if (section.layout === 'orders') {
        const list = el('ul', undefined, 'order-list');
        for (const [i, row] of section.rows.entries()) {
          const translatedRow = translated.rows[i] || row, item = el('li', undefined, 'order-entry');
          item.append(el('strong', String(row[0])));
          for (let j = 1; j < row.length; j++) if (row[j]) item.append(el('p', t(`${section.columns[j]}: ${row[j]}`, `${translated.columns[j]}: ${translatedRow[j]}`)));
          list.append(item);
        }
        node.append(list); continue;
      }
      if (Array.isArray(card.result?.receipts)) {
        for (const [i, row] of section.rows.entries()) {
          const line = el('div', undefined, 'today-entry');
          line.append(el('p', t(row.join(' — '), (translated.rows[i] || row).join(' — ')))); node.append(line);
        }
        continue;
      }
      const scroll = el('div', undefined, 'read-table-scroll'), table = el('table'), head = el('thead'), tr = el('tr');
      for (const [i, label] of section.columns.entries()) { const th = el('th', t(label, translated.columns[i] || label)); th.setAttribute('scope', 'col'); tr.append(th); }
      head.append(tr); table.append(head);
      const body = el('tbody');
      for (const [i, values] of section.rows.entries()) { const row = el('tr'); for (const [j, value] of values.entries()) row.append(el('td', t(String(value ?? ''), String(translated.rows[i]?.[j] ?? value ?? '')))); body.append(row); }
      table.append(body); scroll.append(table); node.append(scroll);
    }
    if (card.result?.view_all) node.append(button(t('View all today’s entries', 'Ver todas las entradas de hoy'), () => todayEntries()));
  }
  function details() { /* Staff cards contain only reviewed, plain fields. */ }
  function fields(node, pairs) {
    const list = el('dl', undefined, 'field-list');
    pairs.filter(([, value]) => value !== undefined && value !== null && value !== '').forEach(([name, value]) => {
      list.append(el('dt', name), el('dd', typeof value === 'object' ? (value.name || value.label_en || t('See the entry details', 'Consulta los detalles')) : String(value)));
    }); node.append(list);
  }
  function autoSubstitutions(node, result) {
    for (const item of result.auto_substitutions || []) {
      if (!item.message_en || !item.message_es) continue;
      const message = el('p', t(item.message_en, item.message_es), 'auto-substitution');
      message.setAttribute('role', 'status'); node.append(message);
    }
  }
  function renderReceipt(node, result) {
    if (!validReceipt(result)) { failed(node, t('FL returned no receipt number.', 'FL no devolvió un número de recibo.')); return; }
    node.replaceChildren(); node.className = 'card receipt';
    node.append(el('span', t('✓ RECORDED IN FACTORY LEDGER', '✓ REGISTRADO EN FACTORY LEDGER'), 'eyebrow'), el('h3', result.receipt_number, 'receipt-number'));
    node.append(el('p', result.replayed ? t('Same entry, same receipt. Your retry did not create another entry.', 'Misma entrada, mismo recibo. El reintento no creó otra entrada.') : t('FL confirmed this entry.', 'FL confirmó esta entrada.')));
    autoSubstitutions(node, result);
    fields(node, [[t('Product', 'Producto'), result.product_name || result.target_product_name], [t('Lot', 'Lote'), result.lot_code || result.output_lot_code],
      [t('Customer', 'Cliente'), result.customer_name || result.customer], [t('Order', 'Pedido'), result.order_number], [t('Shipment', 'Envío'), result.shipment_id],
      [t('BOL', 'BOL'), result.bol_reference], [t('Photo status', 'Estado de foto'), result.proof_status === 'complete' ? t('Attached', 'Adjunta') : result.proof_status === 'photo_pending' ? t('Pending', 'Pendiente') : display.plainLabel(result.proof_status, language)]]);
    for (const shortage of result.shortages || []) {
      const warning = el('div', undefined, 'warning'); warning.setAttribute('role', 'status');
      warning.append(el('strong', t('Shortage recorded', 'Faltante registrado')),
        el('p', t(`${shortage.product_name} · ${shortage.lot_code}: short ${shortage.short_lb} lb. The floor must resolve this shortage.`,
          `${shortage.product_name} · ${shortage.lot_code}: faltan ${shortage.short_lb} lb. El piso debe resolver este faltante.`)));
      fields(warning, [[t('Due', 'Vence'), display.formatTime(shortage.due_at, { language })], [t('Exception', 'Excepción'), shortage.exception_id]]);
      node.append(warning);
    }
    if (result.receipt_number.startsWith('CUS-')) {
      node.append(el('p', t('Customer created. Continue with the order when you are ready.', 'Cliente creado. Continúa con el pedido cuando estés listo.')));
      node.append(button(t('Continue with an order', 'Continuar con un pedido'), () => {
        sendTurn({ text: t(`Continue the order using the customer from receipt ${result.receipt_number}.`, `Continúa el pedido usando el cliente del recibo ${result.receipt_number}.`) });
      }, 'primary'));
    }
    for (const exception of result.lot_exceptions || []) {
      const warning = el('div', undefined, 'warning lot-exception'); warning.setAttribute('role', 'status');
      warning.append(el('strong', t('Physical lot awaiting follow-up', 'Lote físico pendiente de seguimiento')),
        el('p', t(`${exception.product_name}: physical tag ${exception.physical_tag}. The suggested lot was not confirmed or used.`,
          `${exception.product_name}: etiqueta física ${exception.physical_tag}. El lote sugerido no se confirmó ni se usó.`)));
      fields(warning, [[t('Quantity', 'Cantidad'), display.quantityLabel(exception.quantity_lb, 'lb', language)], [t('Exception', 'Excepción'), exception.exception_id]]);
      node.append(warning);
    }
    details(node, result);
  }
  function renderRecordOutcome(node, response) {
    if (response.kind === 'receipt' && validReceipt(response.result)) renderReceipt(node, response.result);
    else if (response.kind === 'awaiting_approval' && awaitingApproval(response.result)) renderHeld(node, response.draft_id, response.result);
    else throw response;
  }
  function renderHeld(node, draftId, result) {
    node.replaceChildren(); node.className = 'card';
    const warning = el('div', undefined, 'warning'); warning.setAttribute('role', 'status');
    warning.append(el('h3', t('Waiting for owner approval', 'En espera de la aprobación del dueño')),
      el('p', t('NOT recorded. FL is holding this correction for the owner to review.',
        'NO registrado. FL retiene esta corrección para que el dueño la revise.')));
    node.append(warning);
    fields(node, [[t('Exception', 'Excepción'), result.exception_id]]);
    const errorArea = el('div');
    const check = button(t('Check approval', 'Consultar aprobación'), async () => {
      if (busy) return; setBusy(true); check.disabled = true; errorArea.replaceChildren();
      try { renderRecordOutcome(node, await api('/assistant/record', { body: { draft_id: draftId } })); }
      catch (error) { failed(errorArea, error); }
      finally { setBusy(false); check.disabled = false; }
    });
    node.append(check, errorArea); details(node, result);
  }
  function renderDraft(card, existing, fallback) {
    const prepared = card.prepared, draft = prepared.draft || {};
    if (refusalDraft(prepared)) {
      const node = existing || shell(); node.className = 'card refusal';
      node.append(el('span', t('CANNOT RECORD', 'NO SE PUEDE REGISTRAR'), 'draft-badge'), el('h3', t('This entry was refused', 'Esta entrada fue rechazada')));
      for (const blocker of prepared.blockers || []) node.append(el('p', t(display.errorMessage(blocker, 'en'), display.errorMessage(blocker, 'es'))));
      node.append(el('p', t('Nothing was recorded. Update the details or ask the office for help.', 'No se registró nada. Corrige los datos o pide ayuda a la oficina.')));
      return;
    }
    const node = existing || shell();
    drafts.set(card.id, node); node.dataset.draftId = card.id;
    node.append(el('span', t('DRAFT · NOT RECORDED', 'BORRADOR · NO REGISTRADO'), 'draft-badge'));
    const actionNames = { create_customer: ['Create customer', 'Crear cliente'], create_order: ['Create order', 'Crear pedido'], receive: ['Receive', 'Recibir'], make: ['Make', 'Producir'], pack: ['Pack', 'Empacar'], adjust: ['Adjust', 'Ajustar'], found: ['Found inventory', 'Inventario encontrado'],
      ship_order: ['Ship order', 'Enviar pedido'], ship_standalone: ['Ship', 'Enviar'], ship_bulk: ['Ship multiple products', 'Enviar varios productos'] };
    node.append(el('h3', t(...(actionNames[prepared.action] || [prepared.action, prepared.action]))));
    node.append(el('p', t(`Recording as ${prepared.actor?.name || actor.name}`, `Registrando como ${prepared.actor?.name || actor.name}`), 'muted'));
    const enFields = display.draftFields(prepared, { language: 'en', fallback }), esFields = display.draftFields(prepared, { language: 'es', fallback });
    // Ship line headings already identify every product, including service charges.
    const showLineProducts = /^ship_/.test(prepared.action || '') && draft.lines?.length;
    fields(node, enFields.flatMap(([label, value], i) => showLineProducts && label === 'Product' ? [] :
      [[t(label, esFields[i]?.[0] || label), t(String(value), String(esFields[i]?.[1] ?? value))]]));
    autoSubstitutions(node, draft);
    for (const line of draft.lines || []) {
      const row = el('div', undefined, 'ship-line');
      row.append(el('strong', line.product || line.product_name));
      const enLine = display.lineFields(line, 'en'), esLine = display.lineFields(line, 'es');
      fields(row, enLine.map(([label, value], i) => [t(label, esLine[i][0]),
        value == null ? value : t(String(value), String(esLine[i][1] ?? value))]));
      node.append(row);
    }
    const confirmations = [], warningChecks = [], lotGroups = new Map();
    const group = draft.lot_confirmation_group;
    if (group) {
      node.append(el('p', t(group.instruction_en || 'Type the last 4 characters from the tag on each bag you used.', group.instruction_es || 'Escriba los últimos 4 caracteres de la etiqueta de cada bolsa que usó.'), 'lot-instruction'));
      for (const ingredient of group.ingredients || []) {
        const section = el('section', undefined, 'ingredient-confirmation');
        section.append(el('strong', ingredient.product_name + ' · ' + display.quantityLabel(ingredient.needed_lb)));
        const different = button(t('Different lot', 'Otro lote'), () => {
          const entries = confirmations.filter(c => c.lot.product_id === ingredient.product_id);
          const useDifferent = !entries.every(c => c.different);
          for (const entry of entries) {
            entry.different = useDifferent; entry.input.value = '';
            entry.suggested.hidden = entry.method.hidden = entry.hint.hidden = useDifferent;
            entry.input.placeholder = useDifferent ? t('Code from the bag actually used', 'Código de la bolsa que usaste') : t('Read the physical tag', 'Lee la etiqueta física');
          }
          different.setAttribute('aria-pressed', String(useDifferent));
          status.textContent = useDifferent ? t('Enter the actual tag for every bag below. Its lot will remain unverified, awaiting follow-up.', 'Escribe la etiqueta real de cada bolsa abajo. Su lote quedará sin verificar, pendiente de seguimiento.') : '';
          node.refreshRecord?.();
        }, 'different-lot');
        different.setAttribute('aria-pressed', 'false');
        const status = el('p', '', 'muted'); status.setAttribute('role', 'status');
        section.append(different, status); node.append(section); lotGroups.set(ingredient.product_id, section);
      }
    }
    for (const lot of draft.input_plan || []) {
      const row = el('div', undefined, 'lot-confirmation');
      const title = el('label'), suggested = el('span', ` · ${lot.lot_code}`, 'suggested-lot-code');
      title.append(el('span', lot.product_name || t('Input', 'Insumo')), suggested, el('span', ` · ${lot.quantity_lb} lb`)); row.append(title);
      if (!lot.confirmed) {
        const controls = el('div', undefined, 'lot-controls'), method = el('select'), input = el('input');
        [['last4', t('Last 4, including hyphen', 'Últimos 4, incluido el guion')],
          ['full_code', t('Full lot code', 'Código de lote completo')],
          ['scan', t('Scan full lot code', 'Escanear código completo')],
          ['pallet', t('Pallet — full lot code', 'Palé — código de lote completo')]].forEach(([value, label]) => {
          const option = el('option', label); option.value = value; method.append(option);
        });
        method.value = 'last4';
        method.setAttribute('aria-label', t('Confirmation method', 'Método de confirmación'));
        input.type = 'text'; input.autocomplete = 'off'; input.maxLength = 200;
        input.id = 'lot-' + card.id + '-' + lot.lot_id; title.htmlFor = input.id;
        input.placeholder = t('Read the physical tag', 'Lee la etiqueta física');
        const hint = el('small', undefined, 'muted'); hint.id = input.id + '-hint';
        input.setAttribute('aria-describedby', hint.id);
        function updateHint() {
          if (group && (!method.value || method.value === 'last4')) { hint.replaceChildren(); return; }
          hint.replaceChildren(el('span', method.value === 'last4'
            ? t('Type the last 4 characters from the physical tag, including the hyphen (for example, "-004").',
              'Escribe los últimos 4 caracteres de la etiqueta física, incluido el guion (por ejemplo, "-004").')
            : method.value === 'pallet'
              ? t('Enter the full lot code from the pallet tag. FL checks its move to production within the last 24 hours.',
                'Escribe el código completo de la etiqueta del palé. FL verifica su traslado a producción en las últimas 24 horas.')
              : t('Enter or scan the full lot code from the physical tag.', 'Escribe o escanea el código completo de la etiqueta física.')));
        }
        method.addEventListener('change', updateHint); updateHint();
        controls.append(method, input); row.append(controls, hint);
        confirmations.push({ lot, method, input, suggested, hint });
      } else row.append(el('small', lot.confirmed ? t('Confirmed by FL', 'Confirmado por FL') : t('Suggested by FL', 'Sugerido por FL'), 'muted'));
      (lotGroups.get(lot.product_id) || node).append(row);
    }
    for (const warning of prepared.warnings || []) {
      const row = el('div', undefined, 'warning'), message = t(display.plainMessage(warning.message || display.plainLabel(warning.code, 'en'), 'en'), display.plainMessage(warning.message_es || warning.message || display.plainLabel(warning.code, 'es'), 'es'));
      if (warning.requires_ack) {
        const label = el('label'), checkbox = el('input'); checkbox.type = 'checkbox';
        const duplicate = /duplicate/i.test(warning.code || '');
        if (duplicate) row.append(el('p', message));
        label.append(checkbox, el('span', duplicate ? t('Yes, this is a separate entry, record it anyway.', 'Sí, esta es una entrada diferente; regístrala de todos modos.') : t(display.plainLabel(warning.message || warning.code, 'en'), display.plainLabel(warning.message_es || warning.message || warning.code, 'es')))); row.append(label); warningChecks.push({ warning, checkbox });
      } else row.append(el('span', message));
      node.append(row);
    }
    for (const blocker of prepared.blockers || []) {
      if (group && blocker.code === 'LOT_NOT_CONFIRMED') continue;
      node.append(el('div', t(display.errorMessage(blocker, 'en'), display.errorMessage(blocker, 'es')), blocker.code === 'LOT_NOT_CONFIRMED' ? 'warning' : 'error-box'));
      const enChoice = display.existingCustomerChoice(blocker, 'en'), esChoice = display.existingCustomerChoice(blocker, 'es');
      if (enChoice) node.append(button(t(enChoice.label, esChoice.label), () => {
        sendTurn({ text: t(enChoice.text, esChoice.text) });
      }, 'choice-button'));
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
        } catch (error) { neutral(errorArea, 'request_failed'); }
        finally { setBusy(false); }
      }));
    }
    const recordButton = button(t('Record', 'Registrar'), async () => {
      if (busy) return;
      errorArea.replaceChildren(); setBusy(true); recordButton.disabled = cancelButton.disabled = true;
      const recordBody = { draft_id: card.id, acknowledged_warnings: warningChecks.filter(w => w.checkbox.checked).map(w => w.warning.code),
        lot_confirmations: confirmations.map(c => ({ lot_id: c.lot.lot_id, method: c.different ? 'different_lot' : c.method.value, value: c.input.value.trim() })) };
      try {
        const response = await api('/assistant/record', { body: recordBody });
        renderRecordOutcome(node, response);
      } catch (error) { failed(errorArea, error); errorArea.append(el('p', t('Try Record again to check the same ticket. You can also check today’s entries.', 'Pulsa Registrar otra vez para verificar el mismo ticket. También puedes consultar las entradas de hoy.'), 'muted')); }
      finally { setBusy(false); cancelButton.disabled = false; refreshRecord(); }
    }, 'primary');
    const cancelButton = button(t('Cancel', 'Cancelar'), async () => {
      if (busy) return; setBusy(true); recordButton.disabled = cancelButton.disabled = true; errorArea.replaceChildren();
      try { await api('/assistant/cancel', { body: { draft_id: card.id } }); node.replaceChildren(el('p', t('Cancelled · NOT recorded', 'Cancelado · NO registrado'))); }
      catch (error) { neutral(errorArea, 'request_failed'); }
      finally { setBusy(false); cancelButton.disabled = false; refreshRecord(); }
    });
    function refreshRecord() { recordButton.disabled = busy || !canRecord(prepared) || warningChecks.some(w => !w.checkbox.checked) || confirmations.some(c => !c.input.value.trim()); }
    [...confirmations.map(c => c.input), ...warningChecks.map(w => w.checkbox)].forEach(input => input.addEventListener('input', refreshRecord));
    node.refreshRecord = refreshRecord;
    refreshRecord(); actions.append(recordButton, cancelButton); node.append(errorArea, actions);
  }
  function supplierPicker(node, card, candidates) {
    const choices = [], list = el('div', undefined, 'supplier-choices');
    function select(id) {
      if (busy) return;
      return sendTurn({ text: '', choice_id: card.choice_id || card.id, selected_id: id }).finally(() => choices.forEach(b => b.disabled = false));
    }
    for (const candidate of supplierChoices(candidates)) {
      const choice = button(candidate.label || candidate.name, () => { choices.forEach(b => b.disabled = true); return select(candidate.id); }, 'choice-button');
      choices.push(choice); list.append(choice);
    }
    const form = el('form', undefined, 'supplier-search'), input = el('input');
    const label = el('label', t('Search suppliers', 'Buscar proveedores')); input.id = 'supplier-search-' + (card.choice_id || card.id || crypto.randomUUID()); label.htmlFor = input.id;
    input.type = 'search'; input.placeholder = t('Supplier name', 'Nombre del proveedor'); input.autocomplete = 'off'; input.maxLength = 200;
    const search = el('button', t('Search', 'Buscar')); search.type = 'submit';
    form.addEventListener('submit', event => { event.preventDefault(); if (!busy && input.value.trim()) sendTurn({ text: 'supplier: ' + input.value.trim() }); });
    form.append(label, input, search); node.append(list, form);
  }
  function renderCard(card, sourceText) {
    if (card.language === 'en' || card.language === 'es') { language = card.language; localStorage.setItem('fl-assistant-language', language); applyLanguage(); }
    if (card.supersedes_draft_id) {
      const previous = drafts.get(card.supersedes_draft_id);
      if (previous) {
        previous.replaceChildren(el('p', card.kind === 'draft'
          ? t('Replaced by the corrected draft below · NOT recorded', 'Reemplazado por el borrador corregido abajo · NO registrado')
          : t('Previous draft cancelled after your correction · NOT recorded', 'Borrador anterior cancelado después de tu corrección · NO registrado')));
        previous.refreshRecord = null;
      }
    }
    if (card.kind === 'draft') { renderDraft(card, null, sourceText); return; }
    const node = shell();
    if (card.kind === 'choices') {
      const result = card.result || {}, candidates = result.candidates || [], query = card.query || result.query;
      const supplier = card.resolution_kind === 'supplier' || result.kind === 'supplier' || result.resolution_kind === 'supplier' || card.supplier_search;
      if (card.packing_lots) {
        node.append(el('p', query ? t(`I couldn't find lot ${query}. Which batch did these come from?`, `No encontré el lote ${query}. ¿De qué lote a granel salieron?`) : t('Which batch did these come from?', '¿De qué lote a granel salieron?')));
      } else if (card.packing_outputs) {
        node.append(el('p', t('Which finished product did you pack?', '¿Qué producto terminado empacaron?')));
      } else if (card.resolution_kind === 'order' && card.order_not_found) {
        node.append(el('p', t(`I couldn't find order ${query}. Choose that customer's open order.`, `No encontré el pedido ${query}. Elija un pedido abierto de ese cliente.`)));
      } else if (!candidates.length) {
        const followup = supplier ? ['Search for the supplier below.', 'Busca el proveedor abajo.'] : ['Check the name or add more details.', 'Revisa el nombre o agrega más detalles.'];
        node.append(el('p', query ? t(`I couldn’t find “${query}”. ${followup[0]}`, `No encontré “${query}”. ${followup[1]}`) : t(...followup)));
      } else {
        const lead = ['Did you mean one of these?', '¿Quiso decir una de estas opciones?'];
        node.append(el('p', result.outcome === 'none' && query ? t(`I couldn’t find “${query}”. ${lead[0]}`, `No encontré “${query}”. ${lead[1]}`) : t(...lead)));
      }
      if (supplier) { supplierPicker(node, card, candidates); return; }
      const choices = [];
      function select(selectedId, nextPage = false) {
        if (busy) return;
        choices.forEach(b => b.disabled = true);
        return sendTurn({ text: '', choice_id: card.id, selected_id: selectedId, next_page: nextPage }).finally(() => choices.forEach(b => b.disabled = false));
      }
      for (const candidate of candidates) {
        const b = button(candidate.label || candidate.name, () => select(candidate.id), 'choice-button'); choices.push(b); node.append(b);
      }
      const enChoice = display.newCustomerChoice(card, actor?.role, 'en'), esChoice = display.newCustomerChoice(card, actor?.role, 'es');
      if (enChoice) {
        const create = button(t(enChoice.label, esChoice.label), () => {
          sendTurn({ text: t(enChoice.text, esChoice.text) });
        }, 'choice-button'); choices.push(create); node.append(create);
      }
      if (card.packing_lots) { const different = button(t('Different batch', 'Otro lote a granel'), () => select('different_batch'), 'choice-button'); choices.push(different); node.append(different); }
      else if (candidates.length) { const none = button(t('None of these', 'Ninguno de estos'), () => select(null), 'choice-button'); choices.push(none); node.append(none); }
      if (result.has_more) { const more = button(t('More choices →', 'Más opciones →'), () => select(null, true), 'choice-button'); choices.push(more); node.append(more); }
    } else if (card.kind === 'read') {
      node.append(el('span', t('FROM FACTORY LEDGER', 'DE FACTORY LEDGER'), 'eyebrow'), el('h3', t('Here’s what FL shows', 'Esto muestra FL')));
      readResult(node, card);
    } else if (card.kind === 'missing_facts') {
      if (card.summary) node.append(el('p', card.summary, 'read-summary'));
      if (card.packing_batch_needed) node.append(el('p', t('Which batch did these come from? Give the code on its tag.', '¿De qué lote a granel salieron? Escriba el código de la etiqueta.')));
      else if (card.shipping_quantity_needed) {
        const order = card.order;
        node.append(el('p', t(`How much of order ${order.order_number} should we ship? Remaining: ${display.orderRemaining(order, 'en')}.`, `¿Cuánto del pedido ${order.order_number} vamos a enviar? Pendiente: ${display.orderRemaining(order, 'es')}.`)));
        node.append(button(t('Ship all remaining', 'Enviar todo lo pendiente'), () => sendTurn({text: t(`Ship all remaining on order ${order.order_number}.`, `Envía todo lo pendiente del pedido ${order.order_number}.`)}), 'choice-button'));
      } else missingFacts(node, card.fields);
      if (card.supplier_choices || card.supplier_search) supplierPicker(node, card, card.supplier_choices || []);
    } else if (neutralMessages[card.kind]) {
      if (card.message && card.kind !== 'unsupported_request') node.append(el('p', display.plainMessage(card.message, language)));
      else neutral(node, card.kind);
    } else if (card.kind === 'refusal') {
      const missingReason = card.status === 422 && card.reasons?.length && Array.isArray(card.result?.detail)
        && card.result.detail.length && card.result.detail.every(error => error.type === 'missing'
          && error.loc?.length === 2 && error.loc[0] === 'body' && error.loc[1] === 'reason_code');
      // Also sanitize saved pre-fix validation failures when resuming a chat.
      if (!missingReason && card.status === 422 && (Array.isArray(card.result?.detail) || card.result?.error_detail?.code === 'VALIDATION_ERROR')) {
        neutral(node, 'misunderstood'); return;
      }
      if (missingReason) node.append(el('p', t('Choose a correction reason.', 'Elige un motivo de la corrección.')));
      else {
        node.className = 'card refusal';
        node.append(el('span', t('CANNOT RECORD', 'NO SE PUEDE REGISTRAR'), 'draft-badge'), el('h3', t('This request was refused', 'Esta solicitud fue rechazada')));
        node.append(el('p', card.message ? display.plainMessage(card.message, language) : t(display.errorMessage(card.result, 'en'), display.errorMessage(card.result, 'es'))));
        node.append(el('p', t('Update the message with the requested details. FL has not recorded this request.', 'Completa el mensaje con los datos solicitados. FL no ha registrado esta solicitud.'), 'muted'));
      }
      if (JSON.stringify(card.result).includes('reason_code')) {
        for (const reason of card.reasons || []) node.append(button(t(reason.label_en, reason.label_es || reason.label_en), () => sendTurn({ text: `${t('The correction reason is', 'El motivo de corrección es')} ${reason.code}.` }), 'choice-button'));
      }
    } else node.append(el('p', card.message ? display.plainMessage(card.message, language) : t('Please give more details.', 'Agrega más detalles.')));
  }
  function renderCards(cards, sourceText) {
    if (sourceText && !/correction reason|motivo de correcci/i.test(sourceText)) lastIntent = sourceText;
    for (const card of display.dedupeCards(cards || [])) renderCard(card, sourceText || lastIntent);
    const receipt = voidTarget(sourceText);
    if (receipt) {
      const node = shell(); node.className = 'card void-handoff';
      node.append(el('p', t('To void this receipt, open its receipt screen and choose Prepare void.', 'Para anular este recibo, abre su pantalla y elige Preparar anulación.')));
      const link = el('a', t(`Open ${receipt} to void`, `Abrir ${receipt} para anular`), 'button-link');
      link.href = root.FLReceipts?.link(receipt) || '/dashboard/receipt.html?number=' + encodeURIComponent(receipt);
      node.append(link);
    }
  }
  function closePhoneMenu() {
    if (menu && !desktopMenu?.matches) menu.open = false;
  }
  async function todayEntries(requestText) {
    if (busy || !session) return;
    closePhoneMenu();
    if (typeof requestText === 'string') {
      language = /[¿]|\bhoy\b|entradas|registr[ée]/i.test(requestText) ? 'es' : 'en';
      localStorage.setItem('fl-assistant-language', language); applyLanguage();
    }
    setBusy(true); notice('');
    userMessage(typeof requestText === 'string' ? requestText : t('Show all today’s entries', 'Mostrar todas las entradas de hoy'));
    const node = shell(); node.append(el('p', t('Loading today’s entries…', 'Cargando las entradas de hoy…'), 'busy-note'));
    const deadline = Date.now() + 15000, receipts = [];
    try {
      const day = new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());
      let offset = 0;
      while (true) {
        const remaining = deadline - Date.now();
        if (remaining <= 0) throw { timedOut: true };
        const result = await api(`/receipts?date=${day}&summary=true&limit=500&offset=${offset}`, { method: 'GET', timeout: remaining });
        receipts.push(...(result.receipts || []));
        if (!result.has_more) break;
        const next = result.next_offset;
        if (!Number.isSafeInteger(next) || next <= offset) throw { detail: t('FL could not load the next page of today’s entries. Please try again.', 'FL no pudo cargar la siguiente página de las entradas de hoy. Inténtalo de nuevo.') };
        offset = next;
      }
      receipts.sort((a, b) => new Date(b.entered_at || b.happened_at) - new Date(a.entered_at || a.happened_at));
      node.replaceChildren();
      const own = receipts.filter(receipt => {
        const person = receipt.actor;
        const personId = person?.id || receipt.actor_id;
        return personId ? String(personId) === String(actor?.id) : (typeof person === 'string' ? person : person?.name || receipt.actor_name) === actor?.name;
      }).length;
      readResult(node, { kind: 'read', tool: 'today_entries_all', result: { receipts, entry_counts: { own, team: receipts.length } } });
    } catch (error) {
      node.replaceChildren(el('p', error.timedOut
        ? t('Today’s entries could not load within 15 seconds. Please try again.', 'Las entradas de hoy no se cargaron en 15 segundos. Inténtalo de nuevo.')
        : errorText(error, language)));
      node.append(button(t('Try again', 'Intentar de nuevo'), todayEntries));
    } finally { setBusy(false); }
  }
  async function sendTurn(values, original) {
    if (busy) return;
    closePhoneMenu();
    const body = original || { session_id: session, turn_id: crypto.randomUUID(), attachment_ids: attachments.map(a => a.id), ...values };
    setBusy(true); notice('');
    if (!original) userMessage(body.text);
    const waiting = el('p', t('Checking with FL…', 'Consultando FL…'), 'busy-note'); $('messages').append(waiting);
    try {
      const result = await api('/assistant/turn', { body, timeout: 570000, onProgress: message => { delete waiting.dataset.en; delete waiting.dataset.es; waiting.textContent = message; } });
      renderCards(result.cards, body.text);
      retryTurn = null;
      if (result.cards.some(c => c.kind === 'draft')) { attachments = []; renderAttachments(); }
    } catch (error) {
      retryTurn = body; const card = shell(); neutral(card, 'lookup_failed');
      card.append(button(t('Retry this message', 'Reintentar este mensaje'), () => sendTurn({}, body)));
    } finally { waiting.remove(); setBusy(false); $('message').focus(); }
  }
  async function loadChat({ background = false } = {}) {
    const saved = localStorage.getItem(sessionStore);
    let result;
    if (saved) {
      try { result = await api('/assistant/resume', { body: { session_id: saved }, background }); session = saved; }
      catch (error) { if (error.httpStatus !== 404) throw error; }
    }
    if (!session) { result = await api('/assistant/session', { background }); session = result.session_id; localStorage.setItem(sessionStore, session); }
    for (const turn of result.turns || []) { userMessage(turn.text); renderCards(turn.cards, turn.text); }
    // A saved receipt is a real commit result; no text/receipt lookup can call this renderer.
    for (const draft of result.drafts || []) {
      let node = drafts.get(draft.id);
      if (!node) { renderDraft(draft.card, null, lastIntent); node = drafts.get(draft.id); }
      if (!node) continue;
      if (draft.status === 'committed' && validReceipt(draft.result)) renderReceipt(node, draft.result);
      else if (awaitingApproval(draft.result)) renderHeld(node, draft.id, draft.result);
      if (draft.status === 'cancelled') node.replaceChildren(el('p', t('Cancelled · NOT recorded', 'Cancelado · NO registrado')));
    }
  }
  const pilot = root.FLKeySessions.create({ base: root.FLDashboard?.base||location.origin, fetch: root.fetch.bind(root), storage: sessionStorage });
  const authReady = pilot.init().then(enabled => {
    if (enabled) {
      const hint = $('signin-hint');
      hint.dataset.en = 'Sign in once per shift. Your key is exchanged for a session. Sign out before handing over this device.';
      hint.dataset.es = 'Inicia sesión una vez por turno. Tu clave se cambia por una sesión. Cierra sesión antes de entregar este dispositivo.';
      applyLanguage();
    }
    return enabled;
  });
  authReady.catch(() => {});
  async function enterChat(who, { background = false } = {}) {
    actor = who; sessionStore = 'fl-assistant-chat:' + (actor.id || actor.name);
    await loadChat({ background }); $('sign-in-panel').hidden = true; $('chat-panel').hidden = false; $('sign-out').hidden = false;
    applyLanguage(); setBusy(false); $('message').focus();
  }
  $('sign-in-form').addEventListener('submit', async event => {
    event.preventDefault(); notice(''); const submit = event.target.querySelector('button'); setBusy(true); submit.disabled = true; submit.setAttribute('aria-busy', 'true');
    try {
      const enabled = await authReady;
      if (enabled) {
        const exchange = pilot.signIn($('actor-key').value.trim()); $('actor-key').value = '';
        const person = await exchange; credential = pilot.token; await enterChat(person);
      } else {
        credential = $('actor-key').value.trim(); $('actor-key').value = '';
        const who = await api('/auth/whoami', { method: 'GET' });
        if (!who.actor) throw { detail: t('Use your personal actor key.', 'Usa tu clave personal.') };
        await enterChat(who.actor);
      }
    } catch (error) { credential = ''; notice(errorText(error, language)); }
    finally { $('actor-key').value = ''; submit.disabled = false; submit.setAttribute('aria-busy', 'false'); setBusy(false); }
  });
  $('sign-out').addEventListener('click', async () => {
    if (busy) return; setBusy(true); $('sign-out').disabled = true;
    try {
      if (pilot.enabled) await pilot.signOut();
      stopMic(); credential = ''; session = ''; actor = null; location.reload();
    } catch (error) { notice(t('Sign out failed. Try again before handing over this device.', 'No se cerró la sesión. Intenta otra vez antes de entregar este dispositivo.')); }
    finally { setBusy(false); }
  });
  authReady.then(async enabled => {
    if (enabled && pilot.actor) { credential = pilot.token; await enterChat(pilot.actor, { background: true }); }
    else $('sign-in-panel').hidden = false;
  }).catch(error => { $('sign-in-panel').hidden = false; notice(errorText(error, language)); })
    .finally(() => { if ($('auth-loading')) $('auth-loading').hidden = true; if (!actor) $('identity').textContent = t('Sign in to your workspace', 'Inicia sesión en tu espacio'); });
  $('new-chat').addEventListener('click', async () => {
    if (busy) return; setBusy(true);
    try { const result = await api('/assistant/session'); localStorage.setItem(sessionStore, result.session_id); location.reload(); }
    catch (error) { notice(errorText(error, language)); setBusy(false); }
  });
  $('language').addEventListener('click', () => { language = language === 'en' ? 'es' : 'en'; localStorage.setItem('fl-assistant-language', language); applyLanguage(); });
  const menu = $('header-menu'), desktopMenu = root.matchMedia?.('(min-width: 601px)');
  if (menu && desktopMenu) {
    const fitMenu = () => { menu.open = desktopMenu.matches; };
    fitMenu(); desktopMenu.addEventListener?.('change', fitMenu);
  }
  $('today-entries').addEventListener('click', todayEntries);
  $('today-entries-mobile').addEventListener('click', todayEntries);
  $('composer').addEventListener('submit', event => { event.preventDefault(); const text = $('message').value.trim(); if (text && !busy) { $('message').value = ''; if (isTodayQuery(text)) todayEntries(text); else sendTurn({ text }); } });
  $('message').addEventListener('keydown', event => { if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('composer').requestSubmit(); } });
  document.querySelectorAll('[data-prompt]').forEach(b => b.addEventListener('click', () => {
    if (b.dataset.prompt.startsWith('What')) { todayEntries(); return; }
    $('message').value = language === 'es' ? 'Consulta el inventario de ' : b.dataset.prompt; $('message').focus();
  }));
  function evidenceLabel(kind) {
    return ({ bol: t('BOL', 'BOL'), truck_photo: t('Truck photo', 'Foto de camión'), correction_photo: t('Correction photo', 'Foto de corrección'),
      lot_tag: t('Lot tag', 'Etiqueta de lote'), count_sheet: t('Count sheet', 'Hoja de conteo'),
      signed_packing_slip: t('Signed packing slip', 'Nota de empaque firmada'), other: t('Other evidence', 'Otra evidencia') })[kind] || kind;
  }
  function renderAttachments() {
    $('attachments').replaceChildren();
    for (const a of attachments) { const chip = el('div', undefined, 'attachment-chip'); chip.append(el('span', `${a.filename} · ${evidenceLabel(a.kind)}`), button('×', () => { attachments = attachments.filter(x => x.id !== a.id); renderAttachments(); })); $('attachments').append(chip); }
  }
  async function viewPhoto(id) {
    try {
      const blob = await api('/assistant/attachment/read', { body: { attachment_id: id }, binary: true });
      const dialog = el('dialog'), image = el('img'); const url = URL.createObjectURL(blob); image.src = url; image.alt = t('Attached photo', 'Foto adjunta'); image.style.maxWidth = 'min(85vw,800px)'; image.style.maxHeight = '75vh';
      dialog.append(image, button(t('Close', 'Cerrar'), () => dialog.close())); document.body.append(dialog); dialog.addEventListener('close', () => { URL.revokeObjectURL(url); dialog.remove(); }); dialog.showModal();
    } catch (error) { notice(errorText(error, language)); }
  }
  $('photo-button').addEventListener('click', () => $('photo').click());
  $('photo').addEventListener('change', async () => {
    const file = $('photo').files[0]; $('photo').value = ''; if (!file || busy) return;
    if (attachments.length >= 5) { notice(t('Send at most five photos with a message.', 'Envía hasta cinco fotos con un mensaje.')); return; }
    const kind = $('photo-kind').value;
    if (kind === 'correction_photo' && attachments.some(a => a.kind === 'correction_photo')) { notice(t('Send one correction photo with a message. Remove the first photo to replace it.', 'Envía una foto de corrección por mensaje. Quita la primera foto para reemplazarla.')); return; }
    setBusy(true); notice(''); const form = new FormData(); form.append('file', file); form.append('kind', kind);
    try {
      const result = await api('/assistant/attachment?session_id=' + encodeURIComponent(session), { form });
      if (!validAttachment(result)) throw { detail: t('FL did not confirm a private evidence attachment.', 'FL no confirmó un adjunto privado de evidencia.') };
      attachments.push(result); renderAttachments();
    }
    catch (error) { notice(t('Photo upload failed. Please try again.', 'Falló la carga de foto. Inténtalo de nuevo.')); }
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
        } catch (error) { notice(t('Dictation failed. Please try again or type your message.', 'Falló el dictado. Inténtalo de nuevo o escribe tu mensaje.')); }
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
