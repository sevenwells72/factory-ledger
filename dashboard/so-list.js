/* Sales Orders list presentation. State and Health come from the ledger. */
(() => {
  'use strict';
  const escape = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const formats = {
    number: new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }),
    pallets: new Intl.NumberFormat('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 1 }),
  };
  const number = (value, kind = 'number') => value == null || !Number.isFinite(Number(value)) ? '—' : (formats[kind] || formats.number).format(Number(value));
  const pounds = value => number(value) + ' lb';
  const date = value => value ? new Intl.DateTimeFormat('en-US', { month:'short', day:'numeric', timeZone:'UTC' }).format(new Date(value + 'T12:00:00Z')) : '—';
  const stamp = value => value ? new Intl.DateTimeFormat('en-US', { month:'short', day:'numeric', hour:'numeric', minute:'2-digit', timeZone:'America/New_York' }).format(new Date(value)) : '';
  const reasons = {shipped_recorded:'shipped, recorded',shipped_not_recorded:'shipped, not recorded',short_closed:'short-closed',customer_cancelled:'customer cancelled',cns_declined:'CNS declined',duplicate:'duplicate',superseded:'superseded',other:'other'};
  const paragraph = value => '<p>' + escape(value) + '</p>';
  let sequence = 0;
  function explanation(key, definition, specifics) {
    const id = 'so-explain-' + key + '-' + (++sequence);
    return { attrs: `data-explain="${id}" aria-describedby="${id}"`, content: `<div id="${id}" class="so-explanation" hidden>${paragraph(definition)}${specifics}</div>` };
  }
  function trigger(key, text, definition, specifics, className = '', label = '') {
    const info = explanation(key, definition, specifics);
    if (text === '—') return { attrs: info.attrs, content: '—' + info.content };
    return `<button type="button" class="so-explain-trigger ${className}" ${info.attrs}${label ? ` aria-label="${escape(label)}"` : ''}>${text}</button>${info.content}`;
  }
  function healthContent(order) {
    const health = order.health || {};
    let html = (health.reasons || []).map(paragraph).join('') + (health.info || []).map(paragraph).join('');
    if (health.info_detail?.length) html += '<details><summary>Show allocation details</summary><ul>' + health.info_detail.map(line => `<li>${escape(line.product_name || line.sku || 'Line')}${line.product_name && line.sku ? ' (' + escape(line.sku) + ')' : ''}: ${pounds(line.unallocated_lb)} not allocated</li>`).join('') + '</ul></details>';
    return html;
  }
  function palletContent(lines) {
    const eligible = (lines || []).filter(line => line.line_status !== 'cancelled');
    const totals = PalletCalculations.calculateOrderPallets(eligible, 'unit_count');
    const calculated = totals.calculatedPallets;
    const text = calculated == null ? '—' : totals.mappedLineCount > 1 && Math.abs(calculated - Math.round(calculated)) > 1e-9
      ? `${number(totals.physicalPalletsRoundedUp)} mixed (${number(calculated, 'pallets')})` : number(calculated, 'pallets');
    const specifics = eligible.map(line => {
      const result = PalletCalculations.calculateLinePallets(line, line.unit_count);
      return paragraph(`${line.product || line.product_name || line.sku || 'Line'}: ${result.calculatedPallets == null ? 'pallet estimate unavailable' : number(result.calculatedPallets, 'pallets') + ' pallets'}`);
    }).join('') || paragraph('No pallet quantities recorded.');
    return { text, calculated, specifics };
  }
  function row(order) {
    const id = order.order_id;
    const open = order.state === 'open';
    const ready = explanation('ready-' + id, 'Everything on this order is produced, packed, and staged for pickup.', paragraph(order.ready ? `Marked by ${order.ready_by || 'floor'}${order.ready_at ? ' · ' + stamp(order.ready_at) : ''}` : 'Not marked'));
    const readyCell = open ? `<label class="so-ready-target"><input class="order-ready-checkbox" type="checkbox" data-order-id="${id}" aria-label="Ready to ship: ${escape(order.order_number)}" ${ready.attrs} ${order.ready ? 'checked' : ''}${order.readyInFlight ? ' disabled' : ''}></label>${ready.content}` : '';
    const identityInfo = paragraph(order.order_date ? 'Order date · ' + FLDesign.date(order.order_date) : 'Order date not recorded.') + (order.customer_po ? paragraph('Customer PO · ' + order.customer_po) : '') + (order.health?.level === 'quiet' ? healthContent(order) : '');
    const expand = `<button type="button" class="order-expand-toggle" data-order-id="${id}" aria-expanded="false" aria-controls="order-lines-${id}" aria-label="Show line items and actions for ${escape(order.order_number)}"><span class="order-expand-caret">▸</span></button>`;
    let identity = expand + trigger('order-' + id, escape(order.order_number), 'Sales order identifier and the date the order was placed.', identityInfo, 'order-link');
    if (!open) identity += trigger('exit-' + id, `${order.state === 'closed' ? 'Closed' : 'Cancelled'} · ${escape(reasons[order.state_reason] || 'reason not recorded')}`, 'This order has been taken off the open orders board.', paragraph((order.state === 'closed' ? 'Closed' : 'Cancelled') + ' · ' + (reasons[order.state_reason] || 'reason not recorded')) + paragraph(order.state_note || 'No note recorded.') + (order.state_changed_at ? paragraph('Changed ' + stamp(order.state_changed_at)) : '') + (order.related_so_id ? paragraph('Related order reference: ' + order.related_so_id) : ''), 'so-exit-chip');
    const today = new Date().toLocaleDateString('en-CA', {timeZone:'America/New_York'});
    const days = order.requested_ship_date ? Math.round((Date.parse(order.requested_ship_date + 'T12:00:00Z') - Date.parse(today + 'T12:00:00Z')) / 86400000) : null;
    const relative = days == null ? '' : days < 0 ? `${number(-days)} ${days === -1 ? 'day' : 'days'} ${open && order.fulfillment !== 'shipped' ? 'overdue' : 'ago'}` : days === 0 ? 'today' : `in ${number(days)} ${days === 1 ? 'day' : 'days'}`;
    const ship = trigger('ship-' + id, escape(date(order.requested_ship_date)) + (relative ? ` <span class="so-relative">${relative}</span>` : ''), 'The requested date for this order to leave the factory.', paragraph(order.requested_ship_date ? FLDesign.date(order.requested_ship_date) + ' · ' + relative : 'No ship-by date recorded.'));
    const fulfillmentText = order.fulfillment === 'partial' ? `Partial · ${number(order.shipped_effective_lb)} / ${pounds(order.ordered_lb)}` : order.fulfillment === 'shipped' ? 'Shipped' : '—';
    const fulfillment = trigger('fulfillment-' + id, escape(fulfillmentText), 'Fulfillment measures shipments recorded in the ledger.', paragraph(`Ledger shows ${number(order.shipped_effective_lb)} of ${pounds(order.ordered_lb)} shipped; excludes voided shipments and cancelled lines.`), 'so-fulfillment');
    const level = order.health?.level;
    const health = open && ['critical','warning'].includes(level) ? trigger('health-' + id, '⚠', 'Health highlights work that needs attention based on stock and the ship-by date.', healthContent(order), 'so-health so-health-' + level, (level === 'critical' ? 'Critical' : 'Warning') + ' health for ' + order.order_number) : '';
    const remaining = trigger('remaining-' + id, pounds(order.remaining_effective_lb), 'The quantity still to ship on active order lines.', paragraph(`${pounds(order.remaining_effective_lb)} remains after effective shipments; voided shipments and cancelled lines are excluded.`));
    const pallet = palletContent(order.pallet_lines);
    const pallets = trigger('pallets-' + id, escape(pallet.text), 'Pallets estimate the space needed from case quantities. Mixed pallets combine products; parentheses show the pallet equivalent.', pallet.specifics);
    const cells = [readyCell,identity,trigger('customer-' + id,escape(order.customer || '—'),'The customer receiving this sales order.',paragraph(order.customer || 'Customer not recorded.'),'so-customer'),ship,fulfillment,health,remaining,pallets];
    const names = ['Ready to ship','SO #','Customer','Ship by','Fulfillment','Health','Left to ship','Pallets'];
    const classes = ['order-ready-cell','order-identity-cell','so-customer-cell','ship-by-cell','so-fulfillment-cell','so-health-cell','num','num order-pallet-total'];
    const sorts = [String(Boolean(order.ready)),order.order_number,order.customer,order.requested_ship_date,order.fulfillment,level,order.remaining_effective_lb,pallet.calculated];
    return `<tr class="order-row" data-order-id="${id}" data-state="${escape(order.state)}">` + cells.map((cell,i) => `<td class="${classes[i]}${typeof cell === 'object' ? ' so-empty-explained' : ''}" data-label="${names[i]}" data-sort-value="${escape(sorts[i])}"${typeof cell === 'object' ? ' tabindex="0" ' + cell.attrs + ' aria-label="' + names[i] + ' explanation"' : ''}>${typeof cell === 'object' ? cell.content : cell}</td>`).join('') + '</tr>';
  }

  // STATUS-005: one shared data-explain / aria-describedby / focus hook.
  let active = null;
  let dismissTimer;
  function closeExplanation() {
    clearTimeout(dismissTimer);
    if (active) active.panel.hidden = true;
    active = null;
  }
  function position() {
    if (!active) return;
    const { anchor, panel } = active;
    const rect = anchor.getBoundingClientRect();
    panel.style.left = Math.max(8, Math.min(rect.left, innerWidth - panel.offsetWidth - 8)) + 'px';
    const height = panel.offsetHeight;
    panel.style.top = Math.max(8, rect.bottom + height + 12 <= innerHeight ? rect.bottom + 6 : rect.top - height - 6) + 'px';
  }
  function openExplanation(anchor) {
    clearTimeout(dismissTimer);
    if (active?.anchor === anchor) return;
    closeExplanation();
    const panel = document.getElementById(anchor.dataset.explain);
    if (!panel) return;
    panel.hidden = false;
    active = {anchor,panel};
    position();
  }
  function scheduleDismiss() {
    clearTimeout(dismissTimer);
    dismissTimer = setTimeout(() => {
      if (active && !active.anchor.matches(':hover') && !active.panel.matches(':hover') && !active.panel.contains(document.activeElement) && active.anchor !== document.activeElement) closeExplanation();
    }, 180);
  }
  function bindExplanations(container) {
    container.querySelectorAll('[data-explain]').forEach(anchor => {
      anchor.addEventListener('pointerenter', event => { if (event.pointerType !== 'touch') openExplanation(anchor); });
      anchor.addEventListener('pointerleave', scheduleDismiss);
      anchor.addEventListener('focus', () => openExplanation(anchor));
      anchor.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { if (anchor.tagName === 'TD') { event.preventDefault(); openExplanation(anchor); } } });
      anchor.addEventListener('blur', scheduleDismiss);
      anchor.addEventListener('click', event => { event.stopPropagation(); openExplanation(anchor); });
      const panel = document.getElementById(anchor.dataset.explain);
      panel.addEventListener('pointerenter', () => clearTimeout(dismissTimer));
      panel.addEventListener('pointerleave', scheduleDismiss);
      panel.addEventListener('focusout', scheduleDismiss);
      panel.addEventListener('click', event => event.stopPropagation());
      panel.addEventListener('toggle', position, true);
    });
  }
  document.addEventListener('pointerdown', event => { if (active && !active.anchor.contains(event.target) && !active.panel.contains(event.target)) closeExplanation(); });
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && active) { const anchor = active.anchor; const focusInside = active.panel.contains(document.activeElement); closeExplanation(); if (focusInside) { anchor.focus(); closeExplanation(); } event.stopPropagation(); } });
  window.addEventListener('resize', closeExplanation);
  window.addEventListener('scroll', event => { if (active && !active.panel.contains(event.target)) closeExplanation(); }, true);
  window.SOList = {number,row,bindExplanations,closeExplanation};
})();
