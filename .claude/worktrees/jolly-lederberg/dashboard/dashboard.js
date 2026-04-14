/* === CNS Factory Dashboard === */
(function () {
  'use strict';

  const API_BASE = '/dashboard/api';

  // ── State ──
  const state = {
    config: null,
    currentTab: 'operations',
    lastRefresh: null,
    expandedPanels: new Set(JSON.parse(sessionStorage.getItem('expandedPanels') || '[]')),
    calendarMode: 'rolling', // 'rolling' | 'month'
    calendarOffset: 0,       // 0 = current period, -1 = previous, etc.
    searchTimeout: null,
  };

  // ── Theme ──
  function initTheme() {
    const saved = localStorage.getItem('dashboard-theme');
    const theme = saved || 'dark';
    document.documentElement.setAttribute('data-theme', theme);
    updateThemeIcon(theme);
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'dark';
    const next = current === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    localStorage.setItem('dashboard-theme', next);
    updateThemeIcon(next);
  }

  function updateThemeIcon(theme) {
    const btn = document.getElementById('theme-toggle');
    if (btn) btn.innerHTML = theme === 'dark' ? '&#9788;' : '&#9790;';
  }

  // ── Helpers ──
  function fmt(n) {
    if (n == null) return '—';
    return Number(n).toLocaleString('en-US', { maximumFractionDigits: 1 });
  }

  function fmtInt(n) {
    if (n == null) return '—';
    return Math.floor(Number(n)).toLocaleString('en-US');
  }

  function caseBadgeClass(cases) {
    if (cases >= 100) return 'stock-healthy';
    if (cases >= 20) return 'stock-low';
    return 'stock-critical';
  }

  function escHtml(s) {
    if (!s) return '';
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function saveExpandedPanels() {
    sessionStorage.setItem('expandedPanels', JSON.stringify([...state.expandedPanels]));
  }

  function isPanelExpanded(id) {
    return state.expandedPanels.has(id);
  }

  function togglePanel(id) {
    if (state.expandedPanels.has(id)) {
      state.expandedPanels.delete(id);
    } else {
      state.expandedPanels.add(id);
    }
    saveExpandedPanels();
    const header = document.querySelector(`.collapsible-header[data-panel="${id}"]`);
    const body = document.getElementById(id);
    if (header && body) {
      header.classList.toggle('expanded', isPanelExpanded(id));
      body.classList.toggle('expanded', isPanelExpanded(id));
    }
  }

  function showError(elementId, msg) {
    const el = document.getElementById(elementId);
    if (el) {
      el.textContent = msg || 'Failed to load data.';
      el.classList.remove('hidden');
    }
  }

  function hideError(elementId) {
    const el = document.getElementById(elementId);
    if (el) el.classList.add('hidden');
  }

  async function fetchAPI(path) {
    const res = await fetch(API_BASE + path);
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`HTTP ${res.status}: ${body}`);
    }
    return res.json();
  }

  // ── Tabs ──
  function initTabs() {
    document.querySelectorAll('.tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const target = tab.dataset.tab;
        state.currentTab = target;
        document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === target));
        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.toggle('active', tc.id === 'tab-' + target));
      });
    });
  }

  // ── Production Calendar ──
  function getCalendarParams() {
    if (state.calendarMode === 'month') {
      const now = new Date();
      const d = new Date(now.getFullYear(), now.getMonth() + state.calendarOffset, 1);
      const y = d.getFullYear();
      const m = String(d.getMonth() + 1).padStart(2, '0');
      return `month=${y}-${m}`;
    }
    const days = 5;
    const offset = state.calendarOffset * days;
    if (offset === 0) return `days=${days}`;
    // For past periods, we calculate the date range
    const now = new Date();
    const tz = 'America/New_York';
    const todayET = new Date(now.toLocaleString('en-US', { timeZone: tz }));
    const endDate = new Date(todayET);
    endDate.setDate(endDate.getDate() + offset);
    const startDate = new Date(endDate);
    startDate.setDate(startDate.getDate() - days + 1);
    const fmt2 = (dt) => `${dt.getFullYear()}-${String(dt.getMonth() + 1).padStart(2, '0')}`;
    // Use month view with a custom range — fallback to larger day window
    const totalDays = -offset + days;
    return `days=${totalDays}`;
  }

  function updateCalendarLabel() {
    const label = document.getElementById('cal-range-label');
    const toggleBtn = document.getElementById('cal-toggle');
    if (state.calendarMode === 'month') {
      const now = new Date();
      const d = new Date(now.getFullYear(), now.getMonth() + state.calendarOffset, 1);
      label.textContent = d.toLocaleString('en-US', { month: 'long', year: 'numeric' });
      toggleBtn.textContent = '5-Day View';
    } else {
      if (state.calendarOffset === 0) {
        label.textContent = 'Last 5 Days';
      } else {
        label.textContent = `${Math.abs(state.calendarOffset * 5)} days ago`;
      }
      toggleBtn.textContent = 'Month View';
    }
  }

  async function refreshProductionCalendar() {
    hideError('production-error');
    const container = document.getElementById('production-calendar');
    container.innerHTML = '<div class="loading-indicator">Loading production data...</div>';
    updateCalendarLabel();
    try {
      const params = getCalendarParams();
      const data = await fetchAPI('/production?' + params);
      renderProductionCalendar(data, container);
    } catch (e) {
      container.innerHTML = '';
      showError('production-error', 'Failed to load production calendar: ' + e.message);
    }
  }

  function renderProductionCalendar(data, container) {
    const days = data.days || [];
    if (state.calendarMode === 'month') {
      container.classList.add('month-view');
    } else {
      container.classList.remove('month-view');
    }

    if (days.length === 0) {
      container.innerHTML = '<div class="loading-indicator">No production data for this period.</div>';
      return;
    }

    // If rolling 5-day view, ensure we show exactly 5 days (fill empty ones)
    const todayStr = new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
    let displayDays = days;

    if (state.calendarMode === 'rolling' && state.calendarOffset === 0) {
      // Build 5 days: today and 4 days back
      const dayMap = {};
      days.forEach(d => { dayMap[d.date] = d; });
      displayDays = [];
      for (let i = 4; i >= 0; i--) {
        const dt = new Date(todayStr + 'T12:00:00');
        dt.setDate(dt.getDate() - i);
        const ds = dt.toLocaleDateString('en-CA');
        if (dayMap[ds]) {
          displayDays.push(dayMap[ds]);
        } else {
          displayDays.push({
            date: ds,
            day_name: dt.toLocaleDateString('en-US', { weekday: 'long' }),
            batches: [],
            finished_goods: []
          });
        }
      }
    }

    let html = '';
    for (const day of displayDays) {
      const isToday = day.date === todayStr;
      const hasProduction = day.batches.length > 0 || day.finished_goods.length > 0;
      const classes = ['day-card'];
      if (isToday) classes.push('today');
      if (hasProduction) classes.push('has-production');
      if (!hasProduction && state.calendarMode === 'month') classes.push('empty');
      html += `<div class="${classes.join(' ')}">`;
      html += `<div class="day-card-date"><span class="day-name">${escHtml(day.day_name)}</span> &mdash; ${escHtml(day.date)}</div>`;

      if (day.batches.length > 0) {
        html += '<div class="day-section-label">Batches Made</div>';
        for (const b of day.batches) {
          const batchCount = b.standard_batch_size_lbs
            ? (b.total_lbs / b.standard_batch_size_lbs).toFixed(1)
            : null;
          html += `<div class="day-item">`;
          html += `<div class="day-item-name">${escHtml(b.product_name)}</div>`;
          html += `<div class="day-item-stats"><span class="stat-lbs">${fmt(b.total_lbs)} lb</span>`;
          if (batchCount !== null) {
            html += ` &bull; <span class="stat-batches">${batchCount} batches</span>`;
          } else {
            html += ` &bull; <span class="badge unknown">batches: est.</span>`;
          }
          html += `</div></div>`;
        }
      }

      if (day.finished_goods.length > 0) {
        html += '<div class="day-section-label">Finished Goods Packed</div>';
        for (const fg of day.finished_goods) {
          const fgBatchCount = fg.standard_batch_size_lbs
            ? (fg.total_lbs / fg.standard_batch_size_lbs).toFixed(1)
            : null;
          html += `<div class="day-item">`;
          html += `<div class="day-item-name">${escHtml(fg.product_name)}</div>`;
          html += `<div class="day-item-stats"><span class="stat-lbs">${fmt(fg.total_lbs)} lb</span>`;
          if (fgBatchCount !== null) {
            html += ` &bull; <span class="stat-batches">${fgBatchCount} runs</span>`;
          }
          html += `</div></div>`;
        }
      }

      if (!hasProduction) {
        html += '<div class="no-production">No production</div>';
      }

      html += '</div>';
    }
    container.innerHTML = html;
  }

  // ── Finished Goods Inventory ──
  async function refreshFinishedGoods() {
    hideError('finished-goods-error');
    const container = document.getElementById('finished-goods-panels');
    container.innerHTML = '<div class="loading-indicator">Loading finished goods...</div>';
    try {
      const data = await fetchAPI('/inventory/finished-goods');
      renderFinishedGoodsPanels(data, container);
    } catch (e) {
      container.innerHTML = '';
      showError('finished-goods-error', 'Failed to load finished goods: ' + e.message);
    }
  }

  function renderFinishedGoodsPanels(data, container) {
    const panels = data.panels || [];
    let html = '';
    for (const panel of panels) {
      const panelId = 'fg-' + panel.id;
      const expanded = isPanelExpanded(panelId);
      html += `<div class="collapsible-header${expanded ? ' expanded' : ''}" data-panel="${panelId}">`;
      html += `<h3>${escHtml(panel.title)} <span class="panel-count">(${panel.products.length} SKUs)</span></h3>`;
      html += `<span class="chevron"></span></div>`;
      html += `<div id="${panelId}" class="collapsible-body${expanded ? ' expanded' : ''}">`;

      if (panel.products.length > 0) {
        html += '<table class="inv-table"><thead><tr><th>Product</th><th class="num">On Hand (lb)</th><th>Cases</th></tr></thead><tbody>';
        for (const p of panel.products) {
          const rowId = panelId + '-' + p.product_name.replace(/\W/g, '_');
          const caseWt = p.case_weight_lb || panel.case_weight_lb;
          const cases = caseWt ? Math.floor(p.on_hand_lbs / caseWt) : null;
          html += `<tr class="expandable" data-expand="${rowId}">`;
          html += `<td>${escHtml(p.product_name)}</td>`;
          html += `<td class="num">${fmt(p.on_hand_lbs)} lb${cases !== null ? ` (${fmtInt(cases)} × ${fmtInt(caseWt)} lb)` : ''}</td>`;
          html += `<td>${cases !== null ? `<span class="badge ${caseBadgeClass(cases)}">${fmtInt(cases)} cases</span>` : ''}</td>`;
          html += `</tr>`;
          // Lot breakdown
          html += `<tbody class="lot-breakdown" id="${rowId}">`;
          if (p.lots && p.lots.length > 0) {
            for (const lot of p.lots) {
              html += `<tr class="lot-row"><td><span class="lot-link" data-lot="${escHtml(lot.lot_code)}" data-product-id="${lot.product_id || ''}">${escHtml(lot.lot_code)}</span></td>`;
              html += `<td class="num">${fmt(lot.on_hand_lbs)}</td><td></td></tr>`;
            }
          } else {
            html += `<tr class="lot-row"><td colspan="3" style="color:var(--text-muted)">No lots on hand</td></tr>`;
          }
          html += `</tbody>`;
        }
        html += '</tbody></table>';
      } else {
        html += '<div class="loading-indicator">No inventory on hand.</div>';
      }

      if (panel.missing_skus && panel.missing_skus.length > 0) {
        html += '<div class="missing-list"><strong>Missing SKUs:</strong> ' + panel.missing_skus.map(escHtml).join(', ') + '</div>';
      }

      html += '</div>';
    }
    container.innerHTML = html;
    bindCollapsibles(container);
    bindExpandableRows(container);
    bindLotLinks(container);
  }

  // ── Batch Inventory ──
  async function refreshBatchInventory() {
    hideError('batches-error');
    const container = document.getElementById('batch-inventory');
    container.innerHTML = '<div class="loading-indicator">Loading batch inventory...</div>';
    try {
      const data = await fetchAPI('/inventory/batches');
      renderBatchInventory(data, container);
    } catch (e) {
      container.innerHTML = '';
      showError('batches-error', 'Failed to load batch inventory: ' + e.message);
    }
  }

  function renderBatchInventory(data, container) {
    const batches = data.batches || [];
    let html = '';
    if (batches.length > 0) {
      html += '<table class="inv-table"><thead><tr><th>Batch</th><th class="num">On Hand (lb)</th><th>Est. Batches</th></tr></thead><tbody>';
      for (const b of batches) {
        const rowId = 'batch-' + b.product_name.replace(/\W/g, '_');
        const estBatches = b.standard_batch_size_lbs
          ? (b.on_hand_lbs / b.standard_batch_size_lbs).toFixed(1)
          : null;
        html += `<tr class="expandable" data-expand="${rowId}">`;
        html += `<td>${escHtml(b.product_name)}</td>`;
        html += `<td class="num">${fmt(b.on_hand_lbs)}</td>`;
        html += `<td>`;
        if (estBatches !== null) {
          const batchClass = estBatches >= 5 ? 'stock-healthy' : estBatches >= 2 ? 'stock-low' : 'stock-critical';
          html += `<span class="badge ${batchClass}">${estBatches} batches</span>`;
        } else {
          html += `<span class="badge unknown">batches: unknown</span>`;
        }
        html += `</td></tr>`;
        // Lot breakdown
        html += `<tbody class="lot-breakdown" id="${rowId}">`;
        if (b.lots && b.lots.length > 0) {
          for (const lot of b.lots) {
            html += `<tr class="lot-row"><td><span class="lot-link" data-lot="${escHtml(lot.lot_code)}" data-product-id="${lot.product_id || ''}">${escHtml(lot.lot_code)}</span></td>`;
            html += `<td class="num">${fmt(lot.on_hand_lbs)}</td><td></td></tr>`;
          }
        } else {
          html += `<tr class="lot-row"><td colspan="3" style="color:var(--text-muted)">No lots on hand</td></tr>`;
        }
        html += `</tbody>`;
      }
      html += '</tbody></table>';
    } else {
      html += '<div class="loading-indicator">No batch inventory on hand.</div>';
    }

    if (data.missing_skus && data.missing_skus.length > 0) {
      html += '<div class="missing-list"><strong>Missing SKUs:</strong> ' + data.missing_skus.map(escHtml).join(', ') + '</div>';
    }

    container.innerHTML = html;
    bindExpandableRows(container);
    bindLotLinks(container);
  }

  // ── Ingredients ──
  async function refreshIngredients() {
    hideError('ingredients-error');
    const container = document.getElementById('ingredients-panels');
    container.innerHTML = '<div class="loading-indicator">Loading ingredients...</div>';
    try {
      const data = await fetchAPI('/inventory/ingredients');
      renderIngredients(data, container);
    } catch (e) {
      container.innerHTML = '';
      showError('ingredients-error', 'Failed to load ingredients: ' + e.message);
    }
  }

  function renderIngredients(data, container) {
    const categories = data.categories || [];
    let html = '';
    for (const cat of categories) {
      const panelId = 'ing-' + cat.id;
      const expanded = isPanelExpanded(panelId);
      html += `<div class="collapsible-header${expanded ? ' expanded' : ''}" data-panel="${panelId}">`;
      html += `<h3>${escHtml(cat.title)} <span class="ingredient-header-count">Total SKUs: ${cat.total_skus_expected}</span></h3>`;
      html += `<span class="chevron"></span></div>`;
      html += `<div id="${panelId}" class="collapsible-body${expanded ? ' expanded' : ''}">`;

      if (cat.items.length > 0) {
        html += `<table class="inv-table"><thead><tr><th>Ingredient</th><th class="num">On Hand (${escHtml(cat.unit)})</th></tr></thead><tbody>`;
        for (const item of cat.items) {
          const rowId = panelId + '-' + item.name.replace(/\W/g, '_');
          html += `<tr class="expandable" data-expand="${rowId}">`;
          html += `<td>${escHtml(item.name)}</td><td class="num">${fmt(item.on_hand)}</td>`;
          html += `</tr>`;
          // Lot breakdown
          html += `<tbody class="lot-breakdown" id="${rowId}">`;
          if (item.lots && item.lots.length > 0) {
            for (const lot of item.lots) {
              html += `<tr class="lot-row"><td><span class="lot-link" data-lot="${escHtml(lot.lot_code)}" data-product-id="${lot.product_id || ''}">${escHtml(lot.lot_code)}</span></td>`;
              html += `<td class="num">${fmt(lot.on_hand_lbs)}</td></tr>`;
            }
          } else {
            html += `<tr class="lot-row"><td colspan="2" style="color:var(--text-muted)">No lots on hand</td></tr>`;
          }
          html += `</tbody>`;
        }
        html += '</tbody></table>';
      } else {
        html += '<div class="loading-indicator">No inventory on hand.</div>';
      }

      if (cat.missing_skus && cat.missing_skus.length > 0) {
        html += '<div class="missing-list"><strong>Missing SKUs:</strong> ' + cat.missing_skus.map(escHtml).join(', ') + '</div>';
      }

      html += '</div>';
    }
    container.innerHTML = html;
    bindCollapsibles(container);
    bindExpandableRows(container);
    bindLotLinks(container);
  }

  // ── Activity: Shipments ──
  async function refreshShipments() {
    hideError('shipments-error');
    const container = document.getElementById('shipments-table');
    container.innerHTML = '<div class="loading-indicator">Loading shipments...</div>';
    try {
      const data = await fetchAPI('/activity/shipments?limit=100');
      renderShipments(data, container);
    } catch (e) {
      container.innerHTML = '';
      showError('shipments-error', 'Failed to load shipments: ' + e.message);
    }
  }

  function renderShipments(data, container) {
    const shipments = data.shipments || [];
    if (shipments.length === 0) {
      container.innerHTML = '<div class="loading-indicator">No shipments found.</div>';
      return;
    }
    let html = '<table class="activity-table"><thead><tr><th>Date/Time</th><th>Product(s)</th><th class="num">Qty (lb)</th><th>Customer</th><th>Ref</th></tr></thead><tbody>';
    for (const s of shipments) {
      const rowId = 'ship-' + s.transaction_id;
      const products = (s.lines || []).map(l => l.product_name).filter(Boolean);
      const uniqueProducts = [...new Set(products)];
      html += `<tr class="expandable" data-expand="${rowId}">`;
      html += `<td>${escHtml(s.date)} ${escHtml(s.time)}</td>`;
      html += `<td>${uniqueProducts.map(escHtml).join(', ')}</td>`;
      html += `<td class="num">${fmt(s.total_lbs)}</td>`;
      html += `<td>${escHtml(s.customer_name || '—')}</td>`;
      html += `<td>${escHtml(s.order_reference || '—')}</td>`;
      html += `</tr>`;
      // Detail row
      html += `<tr class="activity-detail" id="${rowId}"><td colspan="5">`;
      if (s.lines && s.lines.length > 0) {
        html += '<strong>Lots:</strong><br>';
        for (const l of s.lines) {
          html += `<span class="lot-link" data-lot="${escHtml(l.lot_code)}">${escHtml(l.lot_code)}</span> — ${escHtml(l.product_name)}: ${fmt(Math.abs(l.quantity_lb))} lb<br>`;
        }
      }
      if (s.notes) html += `<br><strong>Notes:</strong> ${escHtml(s.notes)}`;
      html += `</td></tr>`;
    }
    html += '</tbody></table>';
    container.innerHTML = html;
    bindExpandableRows(container);
    bindLotLinks(container);
  }

  // ── Activity: Receipts ──
  async function refreshReceipts() {
    hideError('receipts-error');
    const container = document.getElementById('receipts-table');
    container.innerHTML = '<div class="loading-indicator">Loading receipts...</div>';
    try {
      const data = await fetchAPI('/activity/receipts?limit=100');
      renderReceipts(data, container);
    } catch (e) {
      container.innerHTML = '';
      showError('receipts-error', 'Failed to load receipts: ' + e.message);
    }
  }

  function renderReceipts(data, container) {
    const receipts = data.receipts || [];
    if (receipts.length === 0) {
      container.innerHTML = '<div class="loading-indicator">No receipts found.</div>';
      return;
    }
    let html = '<table class="activity-table"><thead><tr><th>Date/Time</th><th>Product(s)</th><th class="num">Qty (lb)</th><th>Supplier</th><th>BOL</th></tr></thead><tbody>';
    for (const r of receipts) {
      const rowId = 'recv-' + r.transaction_id;
      const products = (r.lines || []).map(l => l.product_name).filter(Boolean);
      const uniqueProducts = [...new Set(products)];
      html += `<tr class="expandable" data-expand="${rowId}">`;
      html += `<td>${escHtml(r.date)} ${escHtml(r.time)}</td>`;
      html += `<td>${uniqueProducts.map(escHtml).join(', ')}</td>`;
      html += `<td class="num">${fmt(r.total_lbs)}</td>`;
      html += `<td>${escHtml(r.shipper_name || '—')}</td>`;
      html += `<td>${escHtml(r.bol_reference || '—')}</td>`;
      html += `</tr>`;
      // Detail row
      html += `<tr class="activity-detail" id="${rowId}"><td colspan="5">`;
      if (r.lines && r.lines.length > 0) {
        html += '<strong>Lots:</strong><br>';
        for (const l of r.lines) {
          html += `<span class="lot-link" data-lot="${escHtml(l.lot_code)}">${escHtml(l.lot_code)}</span> — ${escHtml(l.product_name)}: ${fmt(l.quantity_lb)} lb<br>`;
        }
      }
      if (r.cases_received) html += `<br><strong>Cases:</strong> ${r.cases_received} x ${r.case_size_lb} lb`;
      if (r.notes) html += `<br><strong>Notes:</strong> ${escHtml(r.notes)}`;
      html += `</td></tr>`;
    }
    html += '</tbody></table>';
    container.innerHTML = html;
    bindExpandableRows(container);
    bindLotLinks(container);
  }

  // ── Lot Detail Panel ──
  async function openLotPanel(lotCode, productId) {
    const overlay = document.getElementById('lot-panel-overlay');
    const body = document.getElementById('lot-panel-body');
    const title = document.getElementById('lot-panel-title');
    overlay.classList.remove('hidden');
    title.textContent = 'Lot: ' + lotCode;
    body.innerHTML = '<div class="loading-indicator">Loading lot detail...</div>';
    try {
      let url = '/lot/' + encodeURIComponent(lotCode);
      if (productId) url += '?product_id=' + encodeURIComponent(productId);
      const data = await fetchAPI(url);
      renderLotPanel(data, body);
    } catch (e) {
      body.innerHTML = `<div class="error-msg">Failed to load lot: ${escHtml(e.message)}</div>`;
    }
  }

  function fmtQtyCases(lbs, cases) {
    let s = fmt(lbs) + ' lb';
    if (cases != null) s += ` (${cases} cs)`;
    return s;
  }

  function renderLotPanel(data, body) {
    let html = '<dl class="lot-info-grid">';
    html += `<dt>Lot Code</dt><dd>${escHtml(data.lot_code)}</dd>`;
    html += `<dt>Product</dt><dd>${escHtml(data.product_name)}</dd>`;
    html += `<dt>Source</dt><dd>${escHtml(data.entry_source)}</dd>`;
    html += `<dt>Original Qty</dt><dd>${fmtQtyCases(data.original_quantity_lbs, data.original_cases)}</dd>`;
    html += `<dt>On Hand</dt><dd>${fmtQtyCases(data.on_hand_lbs, data.on_hand_cases)}</dd>`;
    html += '</dl>';

    html += '<h4 style="font-size:13px;margin-bottom:8px;">Transaction Timeline</h4>';
    if (data.timeline && data.timeline.length > 0) {
      html += '<ul class="timeline">';
      for (const t of data.timeline) {
        html += `<li class="txn-${t.type}">`;
        html += `<div class="tl-date">${escHtml(t.date)} ${escHtml(t.time)}</div>`;
        html += `<div><span class="tl-type">${escHtml(t.type)}</span> <span class="tl-qty">${fmtQtyCases(t.quantity_lb, t.cases)}</span></div>`;
        let ctx = '';
        if (t.customer_name) ctx += 'Customer: ' + t.customer_name;
        if (t.shipper_name) ctx += 'Supplier: ' + t.shipper_name;
        if (t.order_reference) ctx += (ctx ? ' | ' : '') + 'SO: ' + t.order_reference;
        if (t.bol_reference) ctx += (ctx ? ' | ' : '') + 'BOL: ' + t.bol_reference;
        if (t.adjust_reason) ctx += (ctx ? ' | ' : '') + 'Reason: ' + t.adjust_reason;
        if (t.notes) ctx += (ctx ? ' | ' : '') + t.notes;
        if (ctx) html += `<div class="tl-context">${escHtml(ctx)}</div>`;
        html += '</li>';
      }
      html += '</ul>';
    } else {
      html += '<div style="color:var(--text-muted);font-size:13px;">No transactions found.</div>';
    }
    body.innerHTML = html;
  }

  function closeLotPanel() {
    document.getElementById('lot-panel-overlay').classList.add('hidden');
  }

  async function openProductPanel(productId, productName) {
    const overlay = document.getElementById('lot-panel-overlay');
    const body = document.getElementById('lot-panel-body');
    const title = document.getElementById('lot-panel-title');
    overlay.classList.remove('hidden');
    title.textContent = productName;
    body.innerHTML = '<div class="loading-indicator">Loading product lots...</div>';
    try {
      const data = await fetchAPI('/product/' + productId + '/lots');
      let html = '<dl class="lot-info-grid">';
      html += `<dt>Product</dt><dd>${escHtml(data.product_name)}</dd>`;
      html += `<dt>Type</dt><dd>${escHtml(data.product_type)}</dd>`;
      if (data.odoo_code) html += `<dt>SKU</dt><dd>${escHtml(data.odoo_code)}</dd>`;
      const totalOnHand = data.lots.reduce((sum, l) => sum + l.on_hand_lbs, 0);
      html += `<dt>Total On Hand</dt><dd>${fmt(totalOnHand)} lb</dd>`;
      html += `<dt>Lot Count</dt><dd>${data.lots.length}</dd>`;
      html += '</dl>';

      if (data.lots.length > 0) {
        const activeLots = data.lots.filter(l => l.on_hand_lbs !== 0);
        const zeroLots = data.lots.filter(l => l.on_hand_lbs === 0);

        if (activeLots.length > 0) {
          html += '<h4 style="font-size:13px;margin:12px 0 8px;">Active Lots</h4>';
          html += '<table style="width:100%;font-size:13px;border-collapse:collapse;">';
          html += '<tr style="border-bottom:1px solid var(--border);"><th style="text-align:left;padding:4px 8px;">Lot Code</th><th style="text-align:left;padding:4px 8px;">Source</th><th style="text-align:right;padding:4px 8px;">On Hand</th></tr>';
          for (const l of activeLots) {
            html += `<tr class="product-lot-row" data-lot-code="${escHtml(l.lot_code)}" style="border-bottom:1px solid var(--border);cursor:pointer;">`;
            html += `<td style="padding:4px 8px;"><span class="lot-link">${escHtml(l.lot_code)}</span></td>`;
            html += `<td style="padding:4px 8px;">${escHtml(l.entry_source || '')}</td>`;
            html += `<td style="text-align:right;padding:4px 8px;">${fmt(l.on_hand_lbs)} lb</td>`;
            html += '</tr>';
          }
          html += '</table>';
        }

        if (zeroLots.length > 0) {
          html += `<h4 style="font-size:13px;margin:12px 0 8px;color:var(--text-muted);">Depleted Lots (${zeroLots.length})</h4>`;
          html += '<table style="width:100%;font-size:13px;border-collapse:collapse;opacity:0.6;">';
          for (const l of zeroLots.slice(0, 10)) {
            html += `<tr class="product-lot-row" data-lot-code="${escHtml(l.lot_code)}" style="border-bottom:1px solid var(--border);cursor:pointer;">`;
            html += `<td style="padding:4px 8px;"><span class="lot-link">${escHtml(l.lot_code)}</span></td>`;
            html += `<td style="padding:4px 8px;">${escHtml(l.entry_source || '')}</td>`;
            html += `<td style="text-align:right;padding:4px 8px;">0 lb</td>`;
            html += '</tr>';
          }
          html += '</table>';
          if (zeroLots.length > 10) {
            html += `<div style="font-size:12px;color:var(--text-muted);padding:4px 8px;">...and ${zeroLots.length - 10} more depleted lots</div>`;
          }
        }
      } else {
        html += '<div style="color:var(--text-muted);font-size:13px;margin-top:8px;">No lots found for this product.</div>';
      }

      body.innerHTML = html;

      // Bind lot clicks within product panel
      body.querySelectorAll('.product-lot-row').forEach(row => {
        row.addEventListener('click', () => {
          openLotPanel(row.dataset.lotCode, row.dataset.productId);
        });
      });
    } catch (e) {
      body.innerHTML = `<div class="error-msg">Failed to load product: ${escHtml(e.message)}</div>`;
    }
  }

  // ── Search ──
  async function performSearch(query) {
    const dropdown = document.getElementById('search-results');
    if (!query || query.length < 2) {
      dropdown.classList.add('hidden');
      return;
    }
    try {
      const data = await fetchAPI('/search?q=' + encodeURIComponent(query));
      renderSearchResults(data, dropdown);
    } catch (e) {
      dropdown.innerHTML = '<div class="search-item">Search failed</div>';
      dropdown.classList.remove('hidden');
    }
  }

  function renderSearchResults(data, dropdown) {
    let html = '';
    let hasResults = false;

    if (data.products && data.products.length > 0) {
      hasResults = true;
      html += '<div class="search-category">Products</div>';
      for (const p of data.products) {
        html += `<div class="search-item" data-search-product-id="${p.product_id}" data-search-product-name="${escHtml(p.name)}"><span class="lot-link">${escHtml(p.name)}</span> <span class="si-sub">${escHtml(p.type)} | ${fmt(p.on_hand_lbs)} lb</span></div>`;
      }
    }
    if (data.lots && data.lots.length > 0) {
      hasResults = true;
      html += '<div class="search-category">Lots</div>';
      for (const l of data.lots) {
        html += `<div class="search-item" data-search-lot="${escHtml(l.lot_code)}"><span class="lot-link">${escHtml(l.lot_code)}</span> <span class="si-sub">${escHtml(l.product_name)} | ${fmt(l.on_hand_lbs)} lb</span></div>`;
      }
    }
    if (data.orders && data.orders.length > 0) {
      hasResults = true;
      html += '<div class="search-category">Sales Orders</div>';
      for (const o of data.orders) {
        html += `<div class="search-item" data-search-order="${o.order_id}"><span class="lot-link">${escHtml(o.order_number)}</span> <span class="si-sub">${escHtml(o.customer)} | ${escHtml(o.status)}</span></div>`;
      }
    }
    if (data.customers && data.customers.length > 0) {
      hasResults = true;
      html += '<div class="search-category">Customers</div>';
      for (const c of data.customers) {
        html += `<div class="search-item" data-search-customer="${escHtml(c.name)}"><span class="lot-link">${escHtml(c.name)}</span> <span class="si-sub">${escHtml(c.contact_name || '')} ${escHtml(c.email || '')}</span></div>`;
      }
    }

    if (!hasResults) {
      html = '<div class="search-item">No results found</div>';
    }

    dropdown.innerHTML = html;
    dropdown.classList.remove('hidden');

    // Bind lot clicks in search results
    dropdown.querySelectorAll('[data-search-lot]').forEach(el => {
      el.addEventListener('click', () => {
        openLotPanel(el.dataset.searchLot);
        dropdown.classList.add('hidden');
      });
    });

    // Bind product clicks – open product detail panel
    dropdown.querySelectorAll('[data-search-product-id]').forEach(el => {
      el.addEventListener('click', () => {
        const productId = el.dataset.searchProductId;
        const productName = el.dataset.searchProductName;
        dropdown.classList.add('hidden');
        document.getElementById('global-search').value = '';
        openProductPanel(productId, productName);
      });
    });

    // Bind order clicks – switch to orders tab and open detail
    dropdown.querySelectorAll('[data-search-order]').forEach(el => {
      el.addEventListener('click', () => {
        const orderId = el.dataset.searchOrder;
        dropdown.classList.add('hidden');
        document.getElementById('global-search').value = '';
        // Switch to orders tab
        document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === 'orders'));
        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.toggle('active', tc.id === 'tab-orders'));
        state.currentTab = 'orders';
        openOrderDetail(parseInt(orderId));
      });
    });

    // Bind customer clicks – switch to orders tab and search by customer
    dropdown.querySelectorAll('[data-search-customer]').forEach(el => {
      el.addEventListener('click', () => {
        const name = el.dataset.searchCustomer;
        dropdown.classList.add('hidden');
        document.getElementById('global-search').value = '';
        // Switch to orders tab
        document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === 'orders'));
        document.querySelectorAll('.tab-content').forEach(tc => tc.classList.toggle('active', tc.id === 'tab-orders'));
        state.currentTab = 'orders';
        // If there's a customer filter on the orders tab, use it; otherwise just switch
        const custFilter = document.getElementById('orders-customer-filter');
        if (custFilter) {
          custFilter.value = name;
          custFilter.dispatchEvent(new Event('change'));
        }
      });
    });
  }

  // ── Binding Helpers ──
  function bindCollapsibles(container) {
    container.querySelectorAll('.collapsible-header').forEach(header => {
      header.addEventListener('click', () => {
        togglePanel(header.dataset.panel);
      });
    });
  }

  function bindExpandableRows(container) {
    container.querySelectorAll('tr.expandable').forEach(row => {
      row.addEventListener('click', () => {
        const targetId = row.dataset.expand;
        const tbody = document.getElementById(targetId);
        if (tbody) {
          tbody.classList.toggle('visible');
        } else {
          // activity detail rows
          const detailRow = container.querySelector(`#${targetId}`);
          if (detailRow) detailRow.classList.toggle('visible');
        }
      });
    });
  }

  function bindLotLinks(container) {
    container.querySelectorAll('.lot-link[data-lot]').forEach(link => {
      link.addEventListener('click', (e) => {
        e.stopPropagation();
        openLotPanel(link.dataset.lot, link.dataset.productId);
      });
    });
  }

  // ── Notes / To-Dos / Reminders ──

  // Notes sub-state
  state.notesFilter = 'all';   // 'all' | 'note' | 'todo' | 'reminder'
  state.notesShowDone = false;
  state.notesData = [];
  state.editingNoteId = null;

  async function refreshNotes() {
    hideError('notes-error');
    const container = document.getElementById('notes-list');
    container.innerHTML = '<div class="loading-indicator">Loading notes...</div>';
    try {
      let url = '/notes';
      const params = [];
      if (state.notesFilter !== 'all') params.push('category=' + state.notesFilter);
      if (!state.notesShowDone) params.push('status=open');
      if (params.length) url += '?' + params.join('&');
      const data = await fetchAPI(url);
      state.notesData = data.notes || [];
      renderNotes(container);
    } catch (e) {
      container.innerHTML = '';
      showError('notes-error', 'Failed to load notes: ' + e.message);
    }
  }

  function renderNotes(container) {
    const notes = state.notesData;
    if (notes.length === 0) {
      container.innerHTML = `<div class="notes-empty">
        <div class="notes-empty-icon">&#128221;</div>
        No ${state.notesFilter === 'all' ? 'items' : state.notesFilter + 's'} yet. Click <strong>+ New</strong> to create one.
      </div>`;
      return;
    }

    const todayStr = new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });

    let html = '';
    for (const n of notes) {
      const isDone = n.status === 'done' || n.status === 'dismissed';
      const classes = ['note-card'];
      if (isDone) classes.push('done');
      if (n.priority !== 'normal') classes.push('priority-' + n.priority);

      html += `<div class="${classes.join(' ')}" data-id="${n.id}">`;

      // Checkbox
      html += `<input type="checkbox" class="note-checkbox" data-id="${n.id}" ${isDone ? 'checked' : ''}>`;

      // Content
      html += '<div class="note-content">';
      html += '<div class="note-title-row">';
      html += `<span class="note-title">${escHtml(n.title)}</span>`;
      html += `<span class="note-cat-badge cat-${n.category}">${n.category}</span>`;
      if (n.priority === 'high') html += '<span class="note-priority-badge p-high">High</span>';
      if (n.priority === 'low') html += '<span class="note-priority-badge p-low">Low</span>';
      html += '</div>';

      if (n.body && n.body.trim()) {
        html += `<div class="note-body">${escHtml(n.body)}</div>`;
      }

      // Meta row
      const meta = [];
      if (n.due_date) {
        const overdue = !isDone && n.due_date < todayStr;
        meta.push(`<span class="note-due ${overdue ? 'overdue' : ''}">Due: ${n.due_date}</span>`);
      }
      if (n.entity_type && n.entity_id) {
        meta.push(`<span class="note-entity">${escHtml(n.entity_type)}: ${escHtml(n.entity_id)}</span>`);
      }
      if (n.created_at) {
        meta.push(`<span>Created: ${escHtml(n.created_at)}</span>`);
      }
      if (meta.length) {
        html += `<div class="note-meta">${meta.join('')}</div>`;
      }
      html += '</div>'; // .note-content

      // Actions
      html += '<div class="note-actions">';
      html += `<button class="note-action-btn edit" data-id="${n.id}" title="Edit">&#9998;</button>`;
      html += `<button class="note-action-btn delete" data-id="${n.id}" title="Delete">&#10005;</button>`;
      html += '</div>';

      html += '</div>'; // .note-card
    }
    container.innerHTML = html;

    // Bind checkbox toggles
    container.querySelectorAll('.note-checkbox').forEach(cb => {
      cb.addEventListener('change', async () => {
        const id = cb.dataset.id;
        try {
          await fetch(API_BASE + '/notes/' + id + '/toggle', { method: 'PUT' });
          refreshNotes();
        } catch (err) {
          showError('notes-error', 'Toggle failed: ' + err.message);
        }
      });
    });

    // Bind edit buttons
    container.querySelectorAll('.note-action-btn.edit').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = Number(btn.dataset.id);
        const note = state.notesData.find(n => n.id === id);
        if (note) openNoteModal(note);
      });
    });

    // Bind delete buttons
    container.querySelectorAll('.note-action-btn.delete').forEach(btn => {
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const id = btn.dataset.id;
        if (!confirm('Delete this item?')) return;
        try {
          await fetch(API_BASE + '/notes/' + id, { method: 'DELETE' });
          refreshNotes();
        } catch (err) {
          showError('notes-error', 'Delete failed: ' + err.message);
        }
      });
    });
  }

  function openNoteModal(note) {
    state.editingNoteId = note ? note.id : null;
    const title = document.getElementById('note-modal-title');
    title.textContent = note ? 'Edit Item' : 'New Item';

    // Populate fields
    const catRadios = document.querySelectorAll('input[name="note-cat"]');
    catRadios.forEach(r => { r.checked = r.value === (note ? note.category : 'note'); });

    document.getElementById('note-title').value = note ? (note.title || '') : '';
    document.getElementById('note-body').value = note ? (note.body || '') : '';
    document.getElementById('note-priority').value = note ? (note.priority || 'normal') : 'normal';
    document.getElementById('note-due').value = note ? (note.due_date || '') : '';
    document.getElementById('note-entity-type').value = note ? (note.entity_type || '') : '';
    document.getElementById('note-entity-id').value = note ? (note.entity_id || '') : '';

    document.getElementById('note-modal-overlay').classList.remove('hidden');
  }

  function closeNoteModal() {
    document.getElementById('note-modal-overlay').classList.add('hidden');
    state.editingNoteId = null;
  }

  async function saveNote() {
    const category = document.querySelector('input[name="note-cat"]:checked').value;
    const title = document.getElementById('note-title').value.trim();
    if (!title) {
      alert('Title is required');
      return;
    }

    const body = document.getElementById('note-body').value.trim();
    const priority = document.getElementById('note-priority').value;
    const due_date = document.getElementById('note-due').value || null;
    const entity_type = document.getElementById('note-entity-type').value || null;
    const entity_id = document.getElementById('note-entity-id').value.trim() || null;

    const payload = { title, body, priority, due_date, entity_type, entity_id };

    try {
      if (state.editingNoteId) {
        // Update
        await fetch(API_BASE + '/notes/' + state.editingNoteId, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      } else {
        // Create
        payload.category = category;
        await fetch(API_BASE + '/notes', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
      }
      closeNoteModal();
      refreshNotes();
    } catch (err) {
      alert('Save failed: ' + err.message);
    }
  }

  function initNotes() {
    // Filter buttons
    document.querySelectorAll('.notes-filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        state.notesFilter = btn.dataset.cat;
        document.querySelectorAll('.notes-filter-btn').forEach(b => b.classList.toggle('active', b.dataset.cat === state.notesFilter));
        refreshNotes();
      });
    });

    // Show-done toggle
    document.getElementById('notes-show-done').addEventListener('change', (e) => {
      state.notesShowDone = e.target.checked;
      refreshNotes();
    });

    // Add button
    document.getElementById('notes-add-btn').addEventListener('click', () => openNoteModal(null));

    // Modal close
    document.getElementById('note-modal-close').addEventListener('click', closeNoteModal);
    document.getElementById('note-cancel-btn').addEventListener('click', closeNoteModal);
    document.getElementById('note-modal-overlay').addEventListener('click', (e) => {
      if (e.target === e.currentTarget) closeNoteModal();
    });

    // Save
    document.getElementById('note-save-btn').addEventListener('click', saveNote);
  }

  // ── Sales Orders ──

  const SALES_API_BASE = 'https://fastapi-production-b73a.up.railway.app';
  const SALES_API_KEY = 'ledger-secret-2026-factory';

  // Orders sub-state
  state.ordersData = [];
  state.ordersLoaded = false;
  state.ordersScrollTop = 0;

  async function fetchSalesAPI(path) {
    const res = await fetch(SALES_API_BASE + path, {
      headers: { 'X-API-Key': SALES_API_KEY }
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`HTTP ${res.status}: ${body}`);
    }
    return res.json();
  }

  function formatDateShort(dateStr) {
    if (!dateStr) return '—';
    const parts = dateStr.split('-');
    if (parts.length !== 3) return dateStr;
    return parts[1] + '/' + parts[2] + '/' + parts[0].slice(2);
  }

  function fmtLbs(n) {
    if (n == null) return '—';
    return Number(n).toLocaleString('en-US', { maximumFractionDigits: 0 }) + ' lb';
  }

  function isOrderOverdue(order) {
    if (!order.requested_ship_date) return false;
    const closedStatuses = ['shipped', 'invoiced', 'cancelled'];
    if (closedStatuses.includes(order.status)) return false;
    const today = new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
    return order.requested_ship_date < today;
  }

  function soStatusLabel(status) {
    const labels = {
      'new': 'New',
      'confirmed': 'Confirmed',
      'in_production': 'In Production',
      'ready': 'Ready',
      'partial_ship': 'Partial Ship',
      'shipped': 'Shipped',
      'invoiced': 'Invoiced',
      'cancelled': 'Cancelled'
    };
    return labels[status] || status;
  }

  function getFilteredOrders() {
    const statusFilter = document.getElementById('orders-status-filter').value;
    const customerSearch = document.getElementById('orders-customer-search').value.trim().toLowerCase();
    const overdueOnly = document.getElementById('orders-overdue-only').checked;

    const openStatuses = ['new', 'confirmed', 'in_production', 'ready', 'partial_ship'];

    return state.ordersData.filter(order => {
      // Status filter
      if (statusFilter === 'open') {
        if (!openStatuses.includes(order.status)) return false;
      } else if (statusFilter !== 'all') {
        if (order.status !== statusFilter) return false;
      }

      // Customer search
      if (customerSearch && !(order.customer || '').toLowerCase().includes(customerSearch)) {
        return false;
      }

      // Overdue only
      if (overdueOnly && !isOrderOverdue(order)) {
        return false;
      }

      return true;
    });
  }

  async function refreshOrders() {
    hideError('orders-error');
    const container = document.getElementById('orders-table-container');
    container.innerHTML = '<div class="loading-indicator">Loading sales orders...</div>';
    try {
      const data = await fetchSalesAPI('/sales/orders?limit=200');
      state.ordersData = data.orders || [];
      state.ordersLoaded = true;
      renderOrdersList();
    } catch (e) {
      container.innerHTML = '';
      showError('orders-error', 'Failed to load sales orders: ' + e.message);
    }
  }

  function renderOrdersList() {
    const container = document.getElementById('orders-table-container');
    const orders = getFilteredOrders();

    if (orders.length === 0) {
      container.innerHTML = `<div class="orders-empty">
        <div class="orders-empty-icon">&#128230;</div>
        No orders match your filters.
      </div>`;
      return;
    }

    let html = '<table class="orders-table"><thead><tr>';
    html += '<th>SO #</th><th>Customer</th><th>Order Date</th><th>Ship By</th><th>Status</th><th class="num">Remaining</th>';
    html += '</tr></thead><tbody>';

    for (const o of orders) {
      const overdue = isOrderOverdue(o);
      html += `<tr class="order-row" data-order-id="${o.order_id}">`;
      html += `<td><span class="order-link">${escHtml(o.order_number)}</span></td>`;
      html += `<td>${escHtml(o.customer)}</td>`;
      html += `<td>${formatDateShort(o.order_date)}</td>`;
      html += `<td class="${overdue ? 'date-overdue' : ''}">${formatDateShort(o.requested_ship_date)}</td>`;
      html += `<td><span class="so-badge status-${o.status}">${soStatusLabel(o.status)}</span></td>`;
      html += `<td class="num">${fmtLbs(o.remaining_lb)}</td>`;
      html += `</tr>`;
    }

    html += '</tbody></table>';
    container.innerHTML = html;

    // Bind row clicks
    container.querySelectorAll('.order-row').forEach(row => {
      row.addEventListener('click', () => {
        const orderId = row.dataset.orderId;
        // Save scroll position
        state.ordersScrollTop = document.getElementById('tab-orders').scrollTop || window.scrollY;
        openOrderDetail(orderId);
      });
    });
  }

  async function openOrderDetail(orderId) {
    const listView = document.getElementById('orders-list-view');
    const detailView = document.getElementById('order-detail-view');
    const container = document.getElementById('order-detail-container');

    listView.style.display = 'none';
    detailView.classList.remove('hidden');
    hideError('order-detail-error');
    container.innerHTML = '<div class="loading-indicator">Loading order detail...</div>';

    try {
      const data = await fetchSalesAPI('/sales/orders/' + orderId);
      renderOrderDetail(data, container);
    } catch (e) {
      container.innerHTML = '';
      showError('order-detail-error', 'Failed to load order detail: ' + e.message);
    }
  }

  function renderOrderDetail(data, container) {
    let html = '';

    // Header
    html += '<div class="order-detail-header">';
    html += '<div class="order-detail-top">';
    html += `<span class="order-number">${escHtml(data.order_number)}</span>`;
    html += `<span class="so-badge status-${data.status}">${soStatusLabel(data.status)}</span>`;
    html += '</div>';
    html += `<div class="order-detail-top"><span class="order-customer">${escHtml(data.customer)}</span></div>`;
    html += '<div class="order-detail-dates">';
    html += `<span><strong>Order Date:</strong> ${formatDateShort(data.order_date)}</span>`;
    html += `<span><strong>Ship By:</strong> ${formatDateShort(data.requested_ship_date)}</span>`;
    html += '</div>';
    html += '</div>';

    // KPI row — totals may be nested under data.totals or at top level
    const totals = data.totals || {};
    const totalOrdered = totals.total_ordered_lb != null ? totals.total_ordered_lb : data.total_ordered_lb;
    const totalShipped = totals.total_shipped_lb != null ? totals.total_shipped_lb : data.total_shipped_lb;
    const totalRemaining = totals.remaining_lb != null ? totals.remaining_lb : (data.total_remaining_lb != null ? data.total_remaining_lb : data.remaining_lb);
    html += '<div class="order-kpi-row">';
    html += `<div class="order-kpi"><div class="kpi-label">Total Ordered</div><div class="kpi-value">${fmtLbs(totalOrdered)}</div></div>`;
    html += `<div class="order-kpi"><div class="kpi-label">Shipped</div><div class="kpi-value">${fmtLbs(totalShipped)}</div></div>`;
    html += `<div class="order-kpi"><div class="kpi-label">Remaining</div><div class="kpi-value">${fmtLbs(totalRemaining)}</div></div>`;
    html += '</div>';

    // Line items
    const lines = data.lines || [];
    if (lines.length > 0) {
      html += '<table class="orders-table"><thead><tr>';
      html += '<th>Product</th><th class="num">Ordered</th><th class="num">Shipped</th><th class="num">Remaining</th><th>Status</th>';
      html += '</tr></thead><tbody>';
      for (const l of lines) {
        const remaining = l.remaining_lb != null ? l.remaining_lb : ((l.quantity_lb || 0) - (l.quantity_shipped_lb || 0));
        const productName = l.product || l.name || '—';
        const lineStatusClass = l.line_status === 'fulfilled' ? 'status-shipped'
          : l.line_status === 'partial' ? 'status-partial_ship'
          : l.line_status === 'cancelled' ? 'status-cancelled'
          : 'status-new';
        html += '<tr>';
        html += `<td>${escHtml(productName)}</td>`;
        html += `<td class="num">${fmtLbs(l.quantity_lb)}</td>`;
        html += `<td class="num">${fmtLbs(l.quantity_shipped_lb)}</td>`;
        html += `<td class="num">${fmtLbs(remaining)}</td>`;
        html += `<td><span class="so-badge ${lineStatusClass}">${escHtml(l.line_status || 'pending')}</span></td>`;
        html += '</tr>';
      }
      html += '</tbody></table>';
    }

    // Notes
    if (data.notes && data.notes.trim()) {
      html += '<div class="order-notes-card">';
      html += '<h4>Notes</h4>';
      html += `<p>${escHtml(data.notes)}</p>`;
      html += '</div>';
    }

    container.innerHTML = html;
  }

  function closeOrderDetail() {
    const listView = document.getElementById('orders-list-view');
    const detailView = document.getElementById('order-detail-view');

    detailView.classList.add('hidden');
    listView.style.display = '';

    // Restore scroll position
    window.scrollTo(0, state.ordersScrollTop);
  }

  function initOrders() {
    // Status filter
    document.getElementById('orders-status-filter').addEventListener('change', () => {
      if (state.ordersLoaded) renderOrdersList();
    });

    // Customer search (debounced)
    let orderSearchTimeout;
    document.getElementById('orders-customer-search').addEventListener('input', () => {
      clearTimeout(orderSearchTimeout);
      orderSearchTimeout = setTimeout(() => {
        if (state.ordersLoaded) renderOrdersList();
      }, 200);
    });

    // Overdue toggle
    document.getElementById('orders-overdue-only').addEventListener('change', () => {
      if (state.ordersLoaded) renderOrdersList();
    });

    // Refresh button
    document.getElementById('orders-refresh-btn').addEventListener('click', refreshOrders);

    // Back button
    document.getElementById('order-back-btn').addEventListener('click', closeOrderDetail);
  }

  // ── Refresh All ──
  async function refreshAll() {
    const btn = document.getElementById('refresh-btn');
    btn.classList.add('loading');
    btn.textContent = 'Refreshing...';

    const ops = [
      refreshProductionCalendar(),
      refreshFinishedGoods(),
      refreshBatchInventory(),
      refreshIngredients(),
    ];

    // Always load activity data too (even if tab not visible) so it's ready
    ops.push(refreshShipments());
    ops.push(refreshReceipts());
    ops.push(refreshNotes());
    ops.push(refreshOrders());

    await Promise.allSettled(ops);

    state.lastRefresh = new Date();
    const ts = state.lastRefresh.toLocaleString('en-US', {
      timeZone: 'America/New_York',
      hour: 'numeric', minute: '2-digit', second: '2-digit',
      hour12: true
    }) + ' ET';
    document.getElementById('last-refreshed').textContent = 'Updated: ' + ts;
    btn.classList.remove('loading');
    btn.textContent = 'Refresh';
  }

  // ── Init ──
  function init() {
    initTheme();
    initTabs();
    initNotes();
    initOrders();

    // Theme toggle
    document.getElementById('theme-toggle').addEventListener('click', toggleTheme);

    // Refresh button
    document.getElementById('refresh-btn').addEventListener('click', refreshAll);

    // Calendar nav
    document.getElementById('cal-prev').addEventListener('click', () => {
      state.calendarOffset--;
      refreshProductionCalendar();
    });
    document.getElementById('cal-next').addEventListener('click', () => {
      if (state.calendarOffset < 0) {
        state.calendarOffset++;
        refreshProductionCalendar();
      }
    });
    document.getElementById('cal-toggle').addEventListener('click', () => {
      state.calendarMode = state.calendarMode === 'rolling' ? 'month' : 'rolling';
      state.calendarOffset = 0;
      refreshProductionCalendar();
    });

    // Search
    const searchInput = document.getElementById('global-search');
    searchInput.addEventListener('input', () => {
      clearTimeout(state.searchTimeout);
      state.searchTimeout = setTimeout(() => performSearch(searchInput.value.trim()), 300);
    });
    // Close search on outside click
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.search-wrapper')) {
        document.getElementById('search-results').classList.add('hidden');
      }
    });

    // Lot panel close
    document.getElementById('lot-panel-close').addEventListener('click', closeLotPanel);
    document.getElementById('lot-panel-overlay').addEventListener('click', (e) => {
      if (e.target === e.currentTarget) closeLotPanel();
    });

    // Activity tab collapsibles (pre-rendered in HTML)
    document.querySelectorAll('.collapsible-header[data-panel]').forEach(header => {
      header.addEventListener('click', () => {
        togglePanel(header.dataset.panel);
      });
    });

    // Initial data load
    refreshAll();
  }

  // Kick off
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
