/* Staff-facing descriptions of FL response data. This module never changes ledger values. */
(function (root) {
  'use strict';
  const choose = (language, en, es) => language === 'es' ? es : en;
  const scalar = value => typeof value === 'string' || typeof value === 'number' ? String(value) : '';
  const name = value => typeof value === 'string' ? value : scalar(value?.name || value?.product_name || value?.label);
  const first = (...values) => values.find(value => value !== undefined && value !== null && value !== '');
  const number = value => value !== null && value !== '' && value !== undefined && Number.isFinite(Number(value)) ? Number(value) : null;
  const fmt = value => number(value) === null ? '' : new Intl.NumberFormat('en-US', { maximumFractionDigits: 3 }).format(Number(value));
  const codeNames = {
    physical_count: ['Physical count', 'Conteo físico'], missing_receipt: ['Missing receipt', 'Falta recibo'],
    missing_production: ['Missing production', 'Falta producción'], wrong_lot_used: ['Wrong lot used', 'Se usó el lote equivocado'],
    damage_disposal: ['Damage or disposal', 'Daño o desecho'], damaged: ['Damaged', 'Dañado'],
    unrecorded_usage: ['Unrecorded usage', 'Uso sin registrar'], data_entry_error: ['Data-entry error', 'Error de captura'],
    unknown: ['Unknown', 'Desconocido'], unidentified: ['Lot identity needs to be checked', 'Hay que verificar la identidad del lote'],
    unidentified_lot: ['Lot identity needs to be checked', 'Hay que verificar la identidad del lote'],
    photo_pending: ['Photo pending', 'Foto pendiente'], pending: ['Pending', 'Pendiente'], complete: ['Complete', 'Completo'],
    attached: ['Attached', 'Adjunta'], awaiting_approval: ['Waiting for owner approval', 'En espera de aprobación del dueño'],
    committed: ['Recorded', 'Registrado'], prepared: ['Draft', 'Borrador'], posted: ['Counts in stock', 'Cuenta en el inventario'],
    voided: ['Voided — no longer counts in stock', 'Anulado — ya no cuenta en el inventario'],
    partially_voided: ['Partly voided', 'Anulado en parte'],
    create_customer: ['Create customer', 'Crear cliente'], create_order: ['Create order', 'Crear pedido'],
    receive: ['Receive', 'Recibir'], make: ['Make', 'Producir'], pack: ['Pack', 'Empacar'], adjust: ['Adjust', 'Ajustar'],
    found: ['Found inventory', 'Inventario encontrado'], ship_order: ['Ship order', 'Enviar pedido'],
    ship_standalone: ['Ship', 'Enviar'], ship_bulk: ['Ship multiple products', 'Enviar varios productos'],
    role_not_allowed: ['This action is not allowed for your role', 'Tu rol no permite esta acción'],
    insufficient_stock: ['Not enough stock', 'No hay suficiente inventario'],
    extra_kosher_receive_blocked: ['Extra-kosher receiving is not allowed', 'No se permite recibir este producto extra kosher']
  };
  function plainLabel(value, language = 'en') {
    if (value && typeof value === 'object') return scalar(value['label_' + language] || value.label_en || value.label || value.message || value.name) || plainLabel(value.code, language);
    const raw = scalar(value), translated = codeNames[raw.toLowerCase()];
    if (translated) return translated[language === 'es' ? 1 : 0];
    if (/^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$/i.test(raw)) {
      const words = raw.replaceAll('_', ' ').toLowerCase(); return words.charAt(0).toUpperCase() + words.slice(1);
    }
    return raw;
  }
  function plainMessage(value, language = 'en') {
    return scalar(value).replace(/\bUNIDENTIFIED(?:_LOT)?\b/g, plainLabel('unidentified', language))
      .replace(/\b(?:physical_count|missing_receipt|missing_production|wrong_lot_used|damage_disposal|unrecorded_usage|data_entry_error)\b/g, code => plainLabel(code, language));
  }
  function formatTime(value, { language = 'en', now = new Date(), timeOnly = false } = {}) {
    if (!value) return '';
    const date = new Date(value), current = new Date(now);
    if (!Number.isFinite(date.getTime())) return choose(language, 'Time unavailable', 'Hora no disponible');
    const day = input => new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).format(input);
    const clock = new Intl.DateTimeFormat('en-US', { timeZone: 'America/New_York', hour: 'numeric', minute: '2-digit', hour12: true }).format(date);
    if (timeOnly) return clock;
    if (Number.isFinite(current.getTime()) && day(date) === day(current)) return choose(language, 'Today ', 'Hoy ') + clock;
    return new Intl.DateTimeFormat(language === 'es' ? 'es-US' : 'en-US', { timeZone: 'America/New_York', month: 'short', day: 'numeric', year: 'numeric' }).format(date) + ' ' + clock;
  }
  function errorMessage(value, language = 'en') {
    const data = value?.result || value, detail = data?.detail || data?.error_detail || data;
    if (Array.isArray(detail)) return choose(language, 'Please add the missing details and try again.', 'Agrega los datos que faltan e inténtalo otra vez.');
    if (typeof detail === 'string') return plainMessage(detail, language);
    if (detail && typeof detail === 'object') return plainMessage(detail['message_' + language] || detail.message || detail.action_message || plainLabel(detail.error_code || detail.code, language), language) || choose(language, 'FL could not complete this request.', 'FL no pudo completar esta solicitud.');
    return choose(language, 'FL could not complete this request.', 'FL no pudo completar esta solicitud.');
  }
  // The inventory endpoint sums quantity_lb. A product UOM such as "10 lb case"
  // describes packaging; it does not change that numeric field into cases.
  function inventoryUnit(row) {
    if (row?.quantity_lb !== undefined || row?.qty_on_hand !== undefined || row?.on_hand_lb !== undefined) return 'lb';
    return /^(case|cases|caja|cajas)$/i.test(row?.unit || row?.uom || '') ? 'cases' : 'lb';
  }
  function quantityLabel(value, unit = 'lb', language = 'en') {
    const quantity = fmt(value); return quantity ? quantity + ' ' + (unit === 'cases' ? choose(language, 'cases', 'cajas') : 'lb') : '';
  }
  function draftFields(prepared = {}, { language = 'en', now = new Date(), fallback = {} } = {}) {
    const draft = prepared.draft || {}, source = typeof fallback === 'object' && fallback ? fallback : {};
    const request = prepared.request || prepared.arguments || source;
    const stockLines = (draft.lines || []).filter(line => !line.is_service && line.line_type !== 'service');
    const lineProducts = [...new Set(stockLines.map(line => name(line.product || line.product_name)).filter(Boolean))];
    const lineTotal = stockLines.length && stockLines.every(line => number(line.quantity_lb) !== null) ? stockLines.reduce((sum, line) => sum + Number(line.quantity_lb), 0) : undefined;
    const lineCases = stockLines.length && stockLines.every(line => number(line.cases) !== null) ? stockLines.reduce((sum, line) => sum + Number(line.cases), 0) : undefined;
    const sourceText = typeof fallback === 'string' ? fallback : scalar(source.text);
    const requested = sourceText.match(/(?:^|\s)([\d,]+(?:\.\d+)?)\s*(batches|batch|cases|case|lb|lbs|pounds?)\b/i);
    const blockerProduct = (prepared.blockers || []).map(blocker => scalar(blocker.message).match(/^Confirm (.+) as the output SKU(?: and prepare again)?\.?$/i)?.[1]).find(Boolean);
    const productId = first(draft.target_product_id, draft.product_id, request.target_product_id, request.product_id);
    const product = name(first(draft.product_name, draft.target_product_name, draft.product, draft.target_product, lineProducts.join(', '), blockerProduct, request.product_name, request.target_product_name, request.product, source.product)) || (productId ? choose(language, 'Product #', 'Producto #') + productId : choose(language, 'Product not supplied', 'Producto no indicado'));
    const lb = first(draft.total_lb, draft.total_output_lb, draft.total_weight_lb, draft.quantity_lb, draft.quantity, lineTotal, request.quantity_lb, request.quantity);
    const cases = first(draft.cases, draft.total_cases, lineCases, request.cases), batches = first(draft.batches, request.batches);
    const delta = first(draft.delta_lb, request.delta_lb);
    const before = first(draft.current_quantity_lb, draft.current_balance_lb, draft.before_lb, draft.book_before_lb, draft.lot_on_hand_before);
    const after = first(draft.new_balance_lb, draft.after_lb, draft.lot_on_hand_after, number(before) !== null && number(delta) !== null ? Number(before) + Number(delta) : undefined);
    let quantity = quantityLabel(lb, 'lb', language);
    if (!quantity && number(cases) !== null) quantity = quantityLabel(cases, 'cases', language);
    if (!quantity && number(batches) !== null) quantity = fmt(batches) + choose(language, ' batches', ' lotes de producción');
    if (!quantity && number(delta) !== null) quantity = (Number(delta) > 0 ? '+' : '') + quantityLabel(delta, 'lb', language);
    if (!quantity && number(before) !== null && number(after) !== null) quantity = quantityLabel(after, 'lb', language);
    if (!quantity && requested) {
      const unit = /batch/i.test(requested[2]) ? (Number(requested[1].replaceAll(',', '')) === 1 ? choose(language, 'batch', 'lote de producción') : choose(language, 'batches', 'lotes de producción')) : /case/i.test(requested[2]) ? choose(language, 'cases', 'cajas') : 'lb';
      quantity = fmt(requested[1].replaceAll(',', '')) + ' ' + unit + choose(language, ' requested — confirm the draft', ' solicitados — confirma el borrador');
    }
    quantity ||= choose(language, 'Quantity not supplied — check the draft', 'Cantidad no indicada — revisa el borrador');
    const customer = draft.customer && typeof draft.customer === 'object' ? draft.customer : {};
    const pairs = prepared.action === 'create_customer' ? [] : [[choose(language, 'Product', 'Producto'), product], [choose(language, 'Quantity', 'Cantidad'), quantity]];
    const add = (en, es, value) => { if (value !== undefined && value !== null && value !== '') pairs.push([choose(language, en, es), scalar(value)]); };
    if (number(before) !== null && number(after) !== null) add('Stock before → after', 'Inventario antes → después', fmt(before) + ' → ' + fmt(after) + ' lb');
    if (number(cases) !== null && lb !== undefined) add('Cases', 'Cajas', fmt(cases));
    if (number(batches) !== null && lb !== undefined) add('Batches', 'Lotes de producción', fmt(batches));
    add('Source', 'Origen', name(draft.source_product_name || draft.source_product));
    add('Supplier', 'Proveedor', name(draft.supplier_name || draft.supplier));
    add('Customer', 'Cliente', name(draft.customer || draft.customer_name));
    add('Contact', 'Contacto', customer.contact_name);
    add('Email', 'Correo', customer.email);
    add('Phone', 'Teléfono', customer.phone);
    add('Address', 'Dirección', customer.address);
    add('Customer PO', 'Orden de compra del cliente', draft.customer_po);
    add('Requested ship date', 'Fecha solicitada de envío', draft.requested_ship_date);
    add('Order', 'Pedido', scalar(draft.order || draft.order_id));
    add('Lot', 'Lote', draft.lot_code || draft.output_lot_code);
    add('Supplier lot', 'Lote del proveedor', draft.supplier_lot_code);
    add('BOL', 'BOL', draft.bol_reference);
    add('Photo status', 'Estado de foto', plainLabel(draft.photo || draft.proof_status, language));
    add('Reason', 'Motivo', plainLabel(draft.reason || (language === 'es' && draft.reason_es) || draft.reason_code, language));
    add('Note', 'Nota', plainMessage((language === 'es' && (customer.notes_es || draft.notes_es || draft.note_es)) || customer.notes || draft.notes || draft.note, language));
    if (/unidentified/i.test(draft.identity_status || '')) add('Lot note', 'Nota del lote', plainLabel('unidentified', language));
    add('Happened at', 'Ocurrió el', formatTime(draft.happened_at, { language, now }));
    add('Expires', 'Vence', formatTime(prepared.expires_at, { language, now }));
    return pairs;
  }
  function lineFields(line = {}, language = 'en') {
    const quantity = line.quantity ?? line.original_quantity, unit = line.unit || line.original_unit;
    const units = { cases: ['cases', 'cajas'], bags: ['bags', 'bolsas'], boxes: ['boxes', 'cajas'], each: ['each', 'cada uno'], lb: ['lb', 'lb'] };
    const unitName = units[unit]?.[language === 'es' ? 1 : 0] || unit || '';
    const service = line.is_service || line.line_type === 'service';
    return [[choose(language, 'Type', 'Tipo'), service ? choose(language, 'Service charge', 'Cargo por servicio') : undefined],
      [choose(language, 'Quantity', 'Cantidad'), quantity == null ? undefined : `${quantity} ${unitName}`.trim()],
      [choose(language, 'Pounds', 'Libras'), service ? undefined : line.quantity_lb], [choose(language, 'Cases', 'Cajas'), line.cases],
      [choose(language, 'Unit price', 'Precio unitario'), line.unit_price], [choose(language, 'Amount', 'Importe'), line.amount],
      [choose(language, 'Lots', 'Lotes'), (line.lots || []).map(lot => lot.lot_code).join(', ')]];
  }
  function existingCustomerChoice(blocker = {}, language = 'en') {
    const existing = blocker.existing_customer;
    if (blocker.code !== 'CUSTOMER_ALREADY_EXISTS' || existing?.active === false || !existing?.name) return null;
    return { label: choose(language, `Use existing customer: ${existing.name}`, `Usar cliente existente: ${existing.name}`),
      text: choose(language, `Use the existing customer ${existing.name} for the order.`, `Usa el cliente existente ${existing.name} para el pedido.`) };
  }
  function newCustomerChoice(card = {}, role, language = 'en') {
    if (card.resolution_kind !== 'customer' || !['office', 'owner'].includes(role) || !card.query || (card.result?.offset || 0) !== 0) return null;
    return { label: choose(language, `Add '${card.query}' as a new customer`, `Agregar '${card.query}' como cliente nuevo`),
      text: choose(language, `Create a new customer named ${card.query} first, then continue the order.`,
        `Crea primero un cliente nuevo llamado ${card.query} y luego continúa el pedido.`) };
  }
  const actionVerbs = {
    receive: ['Received', 'Recibido'], make: ['Made', 'Producido'], pack: ['Packed', 'Empacado'],
    found: ['Found', 'Encontrado'], adjust: ['Adjusted', 'Ajustado'], void: ['Voided', 'Anulado'],
    ship_order: ['Shipped', 'Enviado'], ship_standalone: ['Shipped', 'Enviado'], ship_bulk: ['Shipped', 'Enviado']
  };
  function receiptSummary(receipt = {}, language = 'en') {
    const existing = receipt['summary_' + language] || receipt.summary || receipt['what_' + language] || receipt.what;
    if (typeof existing === 'string' && existing.trim()) return plainMessage(existing, language);
    const draft = receipt.draft || {}, detail = receipt.detail || {}, lines = (receipt.transactions || []).flatMap(txn => txn.lines || []);
    const outputs = lines.filter(line => number(line.quantity_lb) > 0), relevant = outputs.length ? outputs : lines;
    const products = [...new Set(relevant.map(line => name(line.product || line.product_name)).filter(Boolean))];
    const product = products.length > 1 ? choose(language, `across ${products.length} products`, `en ${products.length} productos`) : name(products[0] || receipt.product || receipt.product_name || draft.product_name || draft.target_product_name || detail.product);
    const lineWeight = relevant.length ? relevant.reduce((sum, line) => sum + Number(line.quantity_lb || 0), 0) : undefined;
    const lb = first(lineWeight === undefined ? undefined : receipt.action?.startsWith('ship') ? Math.abs(lineWeight) : lineWeight,
      draft.total_lb, draft.total_output_lb, draft.total_weight_lb, draft.quantity, receipt.quantity?.lb);
    const verb = (actionVerbs[receipt.action] || [plainLabel(receipt.action), plainLabel(receipt.action, 'es')])[language === 'es' ? 1 : 0];
    if (receipt.action === 'adjust') {
      const before = first(draft.current_quantity_lb, detail.lot_on_hand_before), after = first(draft.new_balance_lb, detail.lot_on_hand_after);
      if (number(before) !== null && number(after) !== null) return [verb, product, fmt(before) + ' → ' + fmt(after) + ' lb'].filter(Boolean).join(' ');
      return [verb, product, quantityLabel(first(draft.delta_lb, detail.delta_lb, lb), 'lb', language)].filter(Boolean).join(' ');
    }
    const cases = first(draft.cases, draft.total_cases, receipt.quantity?.cases);
    const quantity = receipt.action === 'pack' && number(cases) !== null ? quantityLabel(cases, 'cases', language) : quantityLabel(lb, 'lb', language);
    return [verb, quantity, product].filter(Boolean).join(' ') || choose(language, 'Entry recorded', 'Entrada registrada');
  }
  function entryRow(receipt, opts) {
    return [formatTime(receipt.entered_at || receipt.happened_at, { ...opts, timeOnly: true }), receiptSummary(receipt, opts.language), scalar(receipt.receipt_number), name(receipt.actor) || receipt.actor_name || receipt.operator_id || choose(opts.language, 'Unknown person', 'Persona desconocida')];
  }
  function readSections(card = {}, { language = 'en', now = new Date() } = {}) {
    const data = card.result || {}, opts = { language, now }, sections = [];
    const table = (sentence, columns = [], rows = [], notes = []) => sections.push({ sentence: plainMessage(sentence, language), columns, rows, notes: notes.filter(Boolean).map(note => plainMessage(note, language)) });
    if (card.tool === 'inventory_lookup' || Array.isArray(data.results)) {
      for (const item of data.results || []) {
        const lots = item.lots || [], product = name(item.product || item.product_name || item.name) || choose(language, 'Inventory', 'Inventario');
        const lotQuantity = lot => number(first(lot.qty_on_hand, lot.quantity_lb, lot.on_hand_lb, lot.quantity));
        const lotUnits = [...new Set(lots.map(inventoryUnit))];
        const completeWeights = lots.length && lots.every(lot => inventoryUnit(lot) === 'lb' && lotQuantity(lot) !== null);
        const lotWeight = completeWeights ? lots.reduce((sum, lot) => sum + lotQuantity(lot), 0) : null;
        const total = number(first(item.total_on_hand, item.total_lb, item.quantity));
        const totalUnit = item.total_on_hand !== undefined || item.total_lb !== undefined ? 'lb' : lotUnits.length === 1 && !item.unit && !item.uom ? lotUnits[0] : inventoryUnit(item);
        let quantity;
        if (total !== null) quantity = quantityLabel(total, totalUnit, language);
        else if (!lots.length) quantity = quantityLabel(0, totalUnit, language);
        else quantity = lotUnits.map(unit => {
          const sameUnit = lots.filter(lot => inventoryUnit(lot) === unit);
          return sameUnit.every(lot => lotQuantity(lot) !== null) ? quantityLabel(sameUnit.reduce((sum, lot) => sum + lotQuantity(lot), 0), unit, language) : choose(language, 'Quantity unavailable', 'Cantidad no disponible');
        }).join(choose(language, ' and ', ' y '));
        // A case count is an equivalent total only when supplied for the whole
        // result, or when every pound lot supplies a count and covers that total.
        const completeCases = completeWeights && lots.every(lot => number(lot.unit_count) !== null);
        const caseTotal = first(number(item.unit_count), completeCases && (total === null || Math.abs(total - lotWeight) < 0.000001) ? lots.reduce((sum, lot) => sum + Number(lot.unit_count), 0) : undefined);
        if ((total !== null ? totalUnit === 'lb' : lotUnits.length === 1 && lotUnits[0] === 'lb') && number(caseTotal) !== null) quantity += ' (' + quantityLabel(caseTotal, 'cases', language) + ')';
        const showLotCases = lots.some(lot => inventoryUnit(lot) === 'lb' && number(lot.unit_count) !== null);
        const columns = [choose(language, 'Lot', 'Lote'), choose(language, 'Quantity', 'Cantidad'), choose(language, 'Unit', 'Unidad')];
        if (showLotCases) columns.push(choose(language, 'Cases', 'Cajas'));
        const rows = lots.map(lot => {
          const row = [scalar(lot.lot || lot.lot_code), fmt(lotQuantity(lot)), inventoryUnit(lot) === 'cases' ? choose(language, 'cases', 'cajas') : 'lb'];
          if (showLotCases) row.push(inventoryUnit(lot) === 'lb' && number(lot.unit_count) !== null ? fmt(lot.unit_count) : '—');
          return row;
        });
        table(choose(language, `${product}: ${quantity} in ${lots.length} ${lots.length === 1 ? 'lot' : 'lots'}.`, `${product}: ${quantity} en ${lots.length} ${lots.length === 1 ? 'lote' : 'lotes'}.`), columns, rows);
      }
      if (!sections.length) table(choose(language, 'No inventory matches.', 'No hay coincidencias de inventario.'));
    } else if (Array.isArray(data.receipts)) {
      const receipts = [...data.receipts].sort((a, b) => new Date(b.entered_at || b.happened_at || 0) - new Date(a.entered_at || a.happened_at || 0) || scalar(b.receipt_number).localeCompare(scalar(a.receipt_number)));
      const sentence = card.tool === 'today_entries' ? choose(language,
        `You've recorded ${receipts.length} ${receipts.length === 1 ? 'entry' : 'entries'} today. Team total: see Today's entries.`,
        `Has registrado ${receipts.length} ${receipts.length === 1 ? 'entrada' : 'entradas'} hoy. Total del equipo: consulta las entradas de hoy.`) :
        choose(language, `${receipts.length} ${receipts.length === 1 ? 'entry' : 'entries'} today.`, `${receipts.length} ${receipts.length === 1 ? 'entrada' : 'entradas'} hoy.`);
      table(sentence,
        [choose(language, 'Time', 'Hora'), choose(language, 'What', 'Qué'), choose(language, 'Receipt', 'Recibo'), choose(language, 'Who', 'Quién')], receipts.map(receipt => entryRow(receipt, opts)));
    } else if (Array.isArray(data.sections)) {
      for (const section of data.sections) {
        const label = scalar(section['label_' + language] || section.label || plainLabel(section.key, language)), rows = section.rows || [];
        table(label + ': ' + (rows.length ? String(rows.length) : scalar(section['empty_' + language] || section.empty) || choose(language, 'none', 'ninguno')) + '.',
          rows.length ? [choose(language, 'Details', 'Detalles')] : [], rows.map(row => [plainMessage(row['line_' + language] || row.line || receiptSummary(row, language), language)]));
      }
    } else if (data.receipt_number) {
      const voided = data.effective_status === 'voided';
      const lines = (data.transactions || []).flatMap(txn => txn.lines || []);
      table(receiptSummary(data, language) + '.', [choose(language, 'Product', 'Producto'), choose(language, 'Lot', 'Lote'), choose(language, 'Quantity', 'Cantidad'), choose(language, 'Unit', 'Unidad')],
        lines.map(line => [name(line.product_name || line.product), scalar(line.lot_code), fmt(line.quantity_lb), 'lb']),
        [voided ? plainLabel('voided', language) : '', data.receipt_number, formatTime(data.happened_at, opts), name(data.actor) || data.operator_id]);
    } else {
      const message = data['message_' + language] || data.message || data['summary_' + language] || data.summary;
      table(typeof message === 'string' ? message : choose(language, 'FL returned this information.', 'FL devolvió esta información.'));
    }
    const direct = data.direct_make_substitutions;
    if (direct?.rows?.length) table(scalar(direct['label_' + language] || direct.label || choose(language, 'Ingredient substitutions', 'Sustituciones de ingredientes')),
      [choose(language, 'Details', 'Detalles')], direct.rows.map(row => [plainMessage(row['line_' + language] || row.line, language)]));
    return sections;
  }
  function stable(value) {
    if (Array.isArray(value)) return '[' + value.map(stable).join(',') + ']';
    if (value && typeof value === 'object') return '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + stable(value[key])).join(',') + '}';
    return JSON.stringify(value);
  }
  function dedupeCards(cards = []) {
    const seen = new Set(); return cards.filter(card => { if (card.kind !== 'read') return true; const key = stable({ tool: card.tool, result: card.result }); if (seen.has(key)) return false; seen.add(key); return true; });
  }
  function voidReceipt(value) {
    const text = typeof value === 'string' ? value : scalar(value?.receipt_number || value?.source_receipt_number || value?.result?.receipt_number);
    return text.match(/\b(?:RCV|MK|PK|ADJ|FND|SHP)-\d{6}-\d{3,}\b/i)?.[0]?.toUpperCase() || null;
  }
  const api = { plainLabel, plainMessage, formatTime, errorMessage, inventoryUnit, quantityLabel, draftFields, lineFields, existingCustomerChoice, newCustomerChoice, receiptSummary, readSections, dedupeCards, voidReceipt };
  if (typeof module !== 'undefined') module.exports = api;
  root.FLAssistantDisplay = api;
})(typeof window === 'undefined' ? globalThis : window);
