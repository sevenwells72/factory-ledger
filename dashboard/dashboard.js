/* === CNS Factory Dashboard === */
(function () {
  'use strict';

  const API_BASE = 'https://fastapi-production-b73a.up.railway.app/dashboard/api';

  // ── State ──
  const state = {
    config: null,
    currentTab: 'operations',
    lastRefresh: null,
    expandedPanels: new Set(JSON.parse(sessionStorage.getItem('expandedPanels') || '[]')),
    calendarMode: 'rolling', // 'rolling' | 'month'
    calendarOffset: 0,       // 0 = current period, -1 = previous, etc.
    searchTimeout: null,
    salesOrderInventory: {
      data: null,
      promise: null,
      error: null
    },
    orderLinesCache: {}, // order_id -> full order detail (cached for inline expand)
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

  function fmtWt(n) {
    if (n == null) return '—';
    const v = Number(n);
    return Number.isInteger(v) ? v.toLocaleString('en-US') : v.toLocaleString('en-US', { minimumFractionDigits: 1, maximumFractionDigits: 2 });
  }

  /**
   * Universal dual-display formatter.
   * @param {number} lbs - weight in pounds
   * @param {number|null} caseSizeLb - case_size_lb (packaged/FG) or default_batch_lb (batch)
   * @param {string} productType - 'finished'|'batch'|'service'|'ingredient'
   * @returns {string} formatted display string
   */
  function fmtQty(lbs, caseSizeLb, productType) {
    if (lbs == null) return '\u2014';
    const v = Number(lbs);
    if (productType === 'service') {
      return Number.isInteger(v) ? v.toLocaleString('en-US') + ' units' : v.toLocaleString('en-US', { maximumFractionDigits: 1 }) + ' units';
    }
    if (productType === 'ingredient' || !caseSizeLb || Number(caseSizeLb) <= 0) {
      return fmtWt(v) + ' lb';
    }
    const cs = Number(caseSizeLb);
    const units = Math.round(v / cs);
    const lbStr = fmtWt(v);
    if (productType === 'batch') {
      return lbStr + ' lb \u00b7 ' + units.toLocaleString('en-US') + ' batches';
    }
    return lbStr + ' lb \u00b7 ' + units.toLocaleString('en-US') + ' units';
  }

  function inventoryUnitCount(lbs, caseWeightLb) {
    if (lbs == null || !caseWeightLb || Number(caseWeightLb) <= 0) return null;
    return Math.floor(Number(lbs) / Number(caseWeightLb));
  }

  const CASES_PER_PALLET_BY_CASE_SIZE_LB = {
    10: 140,
    25: 60
  };

  const {
    calculateLinePallets,
    calculateOrderPallets
  } = window.PalletCalculations;

  function normalizeCaseSizeLb(value) {
    if (value == null || value === '') return null;
    const n = Number(value);
    if (!Number.isFinite(n)) return null;
    return Number.isInteger(n) ? n : null;
  }

  function palletsForCases(caseSizeLb, cases, casesPerPalletOverride) {
    if (cases == null) return null;
    const casesPerPallet = casesPerPalletOverride
      || CASES_PER_PALLET_BY_CASE_SIZE_LB[normalizeCaseSizeLb(caseSizeLb)];
    if (!casesPerPallet) return null;
    const caseCount = Number(cases);
    if (!Number.isFinite(caseCount)) return null;
    if (caseCount <= 0) return 0;
    return Math.ceil(caseCount / casesPerPallet);
  }

  function parseCaseSizeLbFromText(text) {
    if (!text) return null;
    const match = String(text).match(/\b(10|25)\s*LB\b/i);
    return match ? Number(match[1]) : null;
  }

  function getProductCategory(product) {
    const explicitCategory = String(product?.category || product?.family || '').trim().toLowerCase();
    if (explicitCategory.includes('coconut')) return 'coconut';
    if (explicitCategory.includes('granola')) return 'granola';

    const productName = String(product?.product_name || product?.name || '').trim();
    const normalizedName = productName.replace(/^Batch\s+/i, '');
    if (/coconut/i.test(normalizedName)) return 'coconut';
    if (/granola/i.test(normalizedName) || /^(CQ|SS)\b/i.test(normalizedName)) return 'granola';
    console.warn('Unrendered production calendar product category:', product);
    return 'other';
  }

  function formatItemName(name, category) {
    let itemName = String(name || '').trim();
    itemName = itemName
      .replace(/^Batch\s+/i, '')
      .replace(/^SS\s+/i, '')
      .replace(/\bSweetened\b/gi, '')
      .replace(/\bChocolate Chip\b/gi, 'Choc Chip')
      .replace(/\bCase\b/gi, '')
      .replace(/\bCNS\b/gi, '');

    if (category === 'coconut') {
      itemName = itemName
        .replace(/\bCoconut\b/gi, '')
        .replace(/\b\d+\s*LB\b/gi, '');
    } else if (category === 'granola') {
      itemName = itemName
        .replace(/\bGranola\b/gi, '');
    }

    return itemName.replace(/\s+/g, ' ').trim();
  }

  function productionBatchCount(batch) {
    if (batch.batch_count != null) return Number(batch.batch_count);
    if (batch.standard_batch_size_lbs) return Number(batch.total_lbs) / Number(batch.standard_batch_size_lbs);
    return null;
  }

  function productionUnitCount(finishedGood) {
    if (finishedGood.unit_count != null) return Number(finishedGood.unit_count);
    if (finishedGood.case_size_lb) return Math.round(Number(finishedGood.total_lbs) / Number(finishedGood.case_size_lb));
    return null;
  }

  function formatBatchCount(count) {
    if (count == null || !Number.isFinite(count)) return '\u2014';
    const display = Number.isInteger(count) ? fmtInt(count) : count.toFixed(1);
    return `${display} ${Number(count) === 1 ? 'batch' : 'batches'}`;
  }

  function formatUnitCount(count) {
    if (count == null || !Number.isFinite(count)) return '\u2014';
    return `${fmtInt(count)} units`;
  }

  function buildProductionCategorySummary(day) {
    const summary = {
      coconut: { totalLbs: 0, batches: [], packed: [] },
      granola: { totalLbs: 0, batches: [], packed: [] }
    };

    for (const batch of (day.batches || [])) {
      const category = getProductCategory(batch);
      if (!summary[category]) continue;
      summary[category].totalLbs += Number(batch.total_lbs) || 0;
      summary[category].batches.push({
        name: formatItemName(batch.product_name, category),
        count: formatBatchCount(productionBatchCount(batch))
      });
    }

    for (const finishedGood of (day.finished_goods || [])) {
      const category = getProductCategory(finishedGood);
      if (!summary[category]) continue;
      summary[category].totalLbs += Number(finishedGood.total_lbs) || 0;
      summary[category].packed.push({
        name: formatItemName(finishedGood.product_name, category),
        count: formatUnitCount(productionUnitCount(finishedGood))
      });
    }

    return summary;
  }

  function getSalesOrderLineCaseSizeLb(line) {
    const fieldCaseSize = normalizeCaseSizeLb(line.case_weight_lb)
      || normalizeCaseSizeLb(line.case_size_lb)
      || normalizeCaseSizeLb(line.default_case_weight_lb);
    if (fieldCaseSize) return fieldCaseSize;
    return parseCaseSizeLbFromText([
      line.product,
      line.product_name,
      line.name,
      line.sku,
      line.odoo_code
    ].filter(Boolean).join(' '));
  }

  function salesOrderLinePallets(line, cases) {
    if (!line || line.is_non_weight || cases == null) return null;
    const caseSizeLb = getSalesOrderLineCaseSizeLb(line);
    return palletsForCases(caseSizeLb, cases);
  }

  function formatInventoryUnits(units, pallets) {
    if (units == null) return '\u2014';
    const unitText = fmtInt(units) + ' units';
    if (pallets == null) return unitText;
    const label = pallets === 1 ? 'pallet' : 'pallets';
    return unitText + ' \u00b7 ' + fmtInt(pallets) + ' ' + label;
  }

  function caseBadgeClass(cases) {
    if (cases >= 100) return 'stock-healthy';
    if (cases >= 20) return 'stock-low';
    return 'stock-critical';
  }

  // Cases per pallet, keyed by the finished-goods panel's stable id (the `id`
  // field in dashboard_config.json → finished_goods_panels). Keyed on the panel
  // id rather than case weight so fractional retail case weights (e.g. 7.5 lb,
  // 2.63 lb) never need fragile float matching. Extend as needed.
  const CASES_PER_PALLET = {
    cases_10lb: 140,  // 10 LB Cases
    bulk_25lb: 60,    // 25 LB Bulk Cases
    retail_ss: 115,   // 12x10 OZ Retail Cases (SS Line)
    retail_bs: 144,   // 6x7 OZ Retail Cases (BS Line)
    // retail_bs_8oz (6x8 OZ Retail Cases) intentionally omitted → renders "—"
  };

  function fmtPallets(cases, panelId, caseWeightLb) {
    if (cases == null) return '—';
    const perPallet = CASES_PER_PALLET[panelId]
      || CASES_PER_PALLET_BY_CASE_SIZE_LB[normalizeCaseSizeLb(caseWeightLb)];
    if (!perPallet) return '—';
    return (cases / perPallet).toFixed(1);
  }

  function escHtml(s) {
    if (!s) return '';
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  function escAttr(s) {
    return escHtml(s).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
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
      const dayBatches = day.batches || [];
      const dayFinishedGoods = day.finished_goods || [];
      const hasProduction = dayBatches.length > 0 || dayFinishedGoods.length > 0;
      const classes = ['day-card'];
      if (isToday) classes.push('today');
      if (hasProduction) classes.push('has-production');
      if (!hasProduction && state.calendarMode === 'month') classes.push('empty');
      html += `<div class="${classes.join(' ')}">`;
      html += `<div class="day-card-date"><span class="day-name">${escHtml(day.day_name)}</span> &mdash; ${escHtml(day.date)}</div>`;

      if (hasProduction) {
        const categorySummary = buildProductionCategorySummary(day);
        const categoryColumns = [
          { key: 'coconut', label: 'COCONUT' },
          { key: 'granola', label: 'GRANOLA' }
        ];
        html += '<div class="production-category-grid">';
        for (const column of categoryColumns) {
          const category = categorySummary[column.key];
          html += `<div class="production-category-column category-${column.key}">`;
          html += `<div class="production-category-header"><span class="category-dot"></span><span>${column.label}</span><span class="category-total">&middot; ${fmt(category.totalLbs)} lb</span></div>`;

          html += '<div class="day-section-label">Batches</div>';
          if (category.batches.length > 0) {
            for (const item of category.batches) {
              html += `<div class="day-item production-row"><div class="day-item-name">${escHtml(item.name)}</div><div class="day-item-stats">${escHtml(item.count)}</div></div>`;
            }
          } else {
            html += '<div class="production-empty">\u2014</div>';
          }

          html += '<div class="day-section-label">Packed</div>';
          if (category.packed.length > 0) {
            for (const item of category.packed) {
              html += `<div class="day-item production-row"><div class="day-item-name">${escHtml(item.name)}</div><div class="day-item-stats">${escHtml(item.count)}</div></div>`;
            }
          } else {
            html += '<div class="production-empty">\u2014</div>';
          }
          html += '</div>';
        }
        html += '</div>';
      } else {
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
      const palletRatio = CASES_PER_PALLET[panel.id];
      const ratioNote = palletRatio ? ` <span class="pallet-ratio">${palletRatio}/pallet</span>` : '';
      html += `<h3>${escHtml(panel.title)} <span class="panel-count">(${panel.products.length} SKUs)</span>${ratioNote}</h3>`;
      html += `<span class="chevron"></span></div>`;
      html += `<div id="${panelId}" class="collapsible-body${expanded ? ' expanded' : ''}">`;

      if (panel.products.length > 0) {
        html += '<table class="inv-table"><thead><tr><th>Product</th><th class="num">On Hand (lb)</th><th>Cases</th><th class="num">Pallets</th></tr></thead><tbody>';
        for (const p of panel.products) {
          const rowId = panelId + '-' + p.product_name.replace(/\W/g, '_');
          const caseWt = p.case_weight_lb || panel.case_weight_lb;
          const cases = inventoryUnitCount(p.on_hand_lbs, caseWt);
          html += `<tr class="expandable" data-expand="${rowId}">`;
          html += `<td>${escHtml(p.product_name)}</td>`;
          html += `<td class="num">${fmt(p.on_hand_lbs)} lb${cases !== null ? ` (${fmtInt(cases)} × ${fmtWt(caseWt)} lb)` : ''}</td>`;
          html += `<td>${cases !== null ? `<span class="badge ${caseBadgeClass(cases)}">${fmtInt(cases)} cases</span>` : ''}</td>`;
          html += `<td class="num">${fmtPallets(cases, panel.id, caseWt)}</td>`;
          html += `</tr>`;
          // Lot breakdown
          html += `<tbody class="lot-breakdown" id="${rowId}">`;
          if (p.lots && p.lots.length > 0) {
            for (const lot of p.lots) {
              const lotUc = lot.unit_count;
              const lotQty = lotUc != null ? fmt(lot.on_hand_lbs) + ' lb &middot; ' + fmtInt(lotUc) + ' units' : fmt(lot.on_hand_lbs) + ' lb';
              html += `<tr class="lot-row"><td><span class="lot-link" data-lot="${escHtml(lot.lot_code)}" data-product-id="${lot.product_id || ''}">${escHtml(lot.lot_code)}</span></td>`;
              html += `<td class="num">${lotQty}</td><td></td><td></td></tr>`;
            }
          } else {
            html += `<tr class="lot-row"><td colspan="4" style="color:var(--text-muted)">No lots on hand</td></tr>`;
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
            const bc = lot.batch_count;
            const lotQty = bc != null ? fmt(lot.on_hand_lbs) + ' lb &middot; ' + bc + ' batches' : fmt(lot.on_hand_lbs) + ' lb';
            html += `<tr class="lot-row"><td><span class="lot-link" data-lot="${escHtml(lot.lot_code)}" data-product-id="${lot.product_id || ''}">${escHtml(lot.lot_code)}</span></td>`;
            html += `<td class="num">${lotQty}</td><td></td></tr>`;
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
      html += `<td class="num">${s.total_units ? fmt(s.total_lbs) + ' lb &middot; ' + fmtInt(s.total_units) + ' units' : fmt(s.total_lbs) + ' lb'}</td>`;
      html += `<td>${escHtml(s.customer_name || '\u2014')}</td>`;
      html += `<td>${escHtml(s.order_reference || '\u2014')}</td>`;
      html += `</tr>`;
      // Detail row
      html += `<tr class="activity-detail" id="${rowId}"><td colspan="5">`;
      if (s.lines && s.lines.length > 0) {
        html += '<strong>Lots:</strong><br>';
        for (const l of s.lines) {
          const absQty = Math.abs(l.quantity_lb);
          const uc = l.unit_count;
          const qtyStr = uc ? fmt(absQty) + ' lb &middot; ' + fmtInt(uc) + ' units' : fmt(absQty) + ' lb';
          html += `<span class="lot-link" data-lot="${escHtml(l.lot_code)}" data-product-id="${l.product_id || ''}">${escHtml(l.lot_code)}</span> \u2014 ${escHtml(l.product_name)}: ${qtyStr}<br>`;
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
      const recvUnits = r.cases_received || null;
      html += `<td class="num">${recvUnits ? fmt(r.total_lbs) + ' lb &middot; ' + fmtInt(recvUnits) + ' units' : fmt(r.total_lbs) + ' lb'}</td>`;
      html += `<td>${escHtml(r.shipper_name || '\u2014')}</td>`;
      html += `<td>${escHtml(r.bol_reference || '\u2014')}</td>`;
      html += `</tr>`;
      // Detail row
      html += `<tr class="activity-detail" id="${rowId}"><td colspan="5">`;
      if (r.lines && r.lines.length > 0) {
        html += '<strong>Lots:</strong><br>';
        for (const l of r.lines) {
          const uc = l.unit_count;
          const qtyStr = uc ? fmt(l.quantity_lb) + ' lb &middot; ' + fmtInt(uc) + ' units' : fmt(l.quantity_lb) + ' lb';
          html += `<span class="lot-link" data-lot="${escHtml(l.lot_code)}" data-product-id="${l.product_id || ''}">${escHtml(l.lot_code)}</span> \u2014 ${escHtml(l.product_name)}: ${qtyStr}<br>`;
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
      let url = API_BASE + '/lot/' + encodeURIComponent(lotCode);
      if (productId) url += '?product_id=' + encodeURIComponent(productId);
      const res = await fetch(url);
      if (res.status === 409) {
        // Ambiguous lot code — show disambiguation picker
        const err = await res.json();
        renderLotDisambiguation(lotCode, err.matches || [], body);
        return;
      }
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text}`);
      }
      const data = await res.json();
      renderLotPanel(data, body);
    } catch (e) {
      body.innerHTML = `<div class="error-msg">Failed to load lot: ${escHtml(e.message)}</div>`;
    }
  }

  function renderLotDisambiguation(lotCode, matches, body) {
    let html = '<div style="padding:8px 0;">';
    html += `<p style="margin:0 0 12px;font-size:14px;">Lot code <strong>${escHtml(lotCode)}</strong> matches multiple products. Select the one you want:</p>`;
    html += '<div style="display:flex;flex-direction:column;gap:8px;">';
    for (const m of matches) {
      html += `<button class="disambig-btn" data-product-id="${m.product_id}" style="
        text-align:left;padding:10px 12px;border:1px solid var(--border);border-radius:6px;
        background:var(--bg-card,#fff);cursor:pointer;font-size:13px;
      ">`;
      html += `<strong>${escHtml(m.product_name)}</strong>`;
      if (m.source) html += ` <span style="color:var(--text-muted);font-size:12px;">(${escHtml(m.source)})</span>`;
      html += '</button>';
    }
    html += '</div></div>';
    body.innerHTML = html;
    body.querySelectorAll('.disambig-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        openLotPanel(lotCode, btn.dataset.productId);
      });
    });
  }

  function fmtQtyCases(lbs, cases) {
    let s = fmtWt(lbs) + ' lb';
    if (cases != null) s += ' \u00b7 ' + fmtInt(cases) + ' units';
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
            html += `<tr class="product-lot-row" data-lot-code="${escHtml(l.lot_code)}" data-product-id="${productId}" style="border-bottom:1px solid var(--border);cursor:pointer;">`;
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
            html += `<tr class="product-lot-row" data-lot-code="${escHtml(l.lot_code)}" data-product-id="${productId}" style="border-bottom:1px solid var(--border);cursor:pointer;">`;
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
        html += `<div class="search-item" data-search-lot="${escHtml(l.lot_code)}" data-search-lot-product-id="${l.product_id || ''}"><span class="lot-link">${escHtml(l.lot_code)}</span> <span class="si-sub">${escHtml(l.product_name)} | ${fmt(l.on_hand_lbs)} lb</span></div>`;
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
        openLotPanel(el.dataset.searchLot, el.dataset.searchLotProductId);
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
          await fetch(API_BASE + '/notes/' + id + '/toggle', {
            method: 'PUT',
            headers: { 'X-API-Key': SALES_API_KEY },
          });
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
          await fetch(API_BASE + '/notes/' + id, {
            method: 'DELETE',
            headers: { 'X-API-Key': SALES_API_KEY },
          });
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
          headers: { 'Content-Type': 'application/json', 'X-API-Key': SALES_API_KEY },
          body: JSON.stringify(payload),
        });
      } else {
        // Create
        payload.category = category;
        await fetch(API_BASE + '/notes', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'X-API-Key': SALES_API_KEY },
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
  const SALES_ORDER_OPEN_STATUSES = ['new', 'confirmed', 'in_production', 'ready', 'partial_ship'];

  // Orders sub-state
  state.ordersData = [];
  state.ordersLoaded = false;
  state.ordersScrollTop = 0;

  async function fetchSalesAPI(path, options = {}) {
    const headers = { 'X-API-Key': SALES_API_KEY, ...(options.headers || {}) };
    const res = await fetch(SALES_API_BASE + path, { ...options, headers });
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

  function getLocalDateFromISO(dateStr) {
    if (!dateStr) return null;
    const parts = dateStr.split('-').map(Number);
    if (parts.length !== 3 || parts.some(Number.isNaN)) return null;
    return new Date(parts[0], parts[1] - 1, parts[2]);
  }

  function formatShipByDate(dateStr) {
    const formattedDate = formatDateShort(dateStr);
    const localDate = getLocalDateFromISO(dateStr);
    if (!localDate) return escHtml(formattedDate);
    const weekday = localDate.toLocaleDateString(undefined, { weekday: 'short' });
    return `<span class="ship-by-date">${escHtml(formattedDate)}</span><span class="ship-by-weekday">${escHtml(weekday)}</span>`;
  }

  function formatReadyTime(value) {
    if (!value) return '';
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return String(value);
    return d.toLocaleTimeString('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      timeZone: 'America/New_York'
    });
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
      'ready': 'Ready to Ship',
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
    const hideReady = document.getElementById('orders-hide-ready').checked;

    return state.ordersData.filter(order => {
      // Status filter
      if (statusFilter === 'open') {
        if (!SALES_ORDER_OPEN_STATUSES.includes(order.status)) return false;
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

      if (hideReady && order.ready) {
        return false;
      }

      return true;
    });
  }

  function csvField(value) {
    const text = value == null ? '' : String(value);
    return /[",\r\n]/.test(text) ? '"' + text.replace(/"/g, '""') + '"' : text;
  }

  function csvDate(dateValue) {
    if (!dateValue) return '';
    const isoMatch = String(dateValue).match(/^(\d{4}-\d{2}-\d{2})/);
    if (isoMatch) return isoMatch[1];
    const date = new Date(dateValue);
    if (Number.isNaN(date.getTime())) return '';
    return date.toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
  }

  function getExportLineQuantity(line) {
    if (line.unit_count != null) return line.unit_count;
    return Number(line.quantity_lb) || 0;
  }

  async function loadOrderDetails(orders) {
    const details = new Array(orders.length);
    let nextIndex = 0;
    const workerCount = Math.min(6, orders.length);

    async function loadNext() {
      while (nextIndex < orders.length) {
        const index = nextIndex++;
        const orderId = orders[index].order_id;
        let detail = state.orderLinesCache[orderId];
        if (!detail) {
          detail = await fetchSalesAPI('/sales/orders/' + orderId);
          state.orderLinesCache[orderId] = detail;
        }
        details[index] = detail;
      }
    }

    await Promise.all(Array.from({ length: workerCount }, loadNext));
    return details;
  }

  async function exportOrdersCsv() {
    const button = document.getElementById('orders-export-btn');
    const orders = getFilteredOrders();
    const originalText = button.textContent;
    button.textContent = 'Exporting...';
    button.classList.add('loading');
    button.disabled = true;
    hideError('orders-error');

    try {
      const details = await loadOrderDetails(orders);
      const rows = [['order_id', 'customer', 'sku', 'product_name', 'qty', 'uom', 'due_date', 'notes']];

      details.forEach((detail, index) => {
        const summary = orders[index];
        for (const line of (detail.lines || [])) {
          const sku = line.sku || line.code || '';
          if (line.is_non_weight || line.is_service || line.no_production || !String(sku).trim()) continue;
          rows.push([
            detail.order_number || summary.order_number || '',
            detail.customer || summary.customer || '',
            sku,
            line.product || line.name || line.description || '',
            getExportLineQuantity(line),
            line.uom || (line.is_non_weight ? 'units' : 'lb'),
            csvDate(detail.requested_ship_date || summary.requested_ship_date),
            line.notes || line.note || ''
          ]);
        }
      });

      const csv = rows.map(row => row.map(csvField).join(',')).join('\r\n') + '\r\n';
      const blob = new Blob([csv], { type: 'text/csv;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'cns_open_orders_' + new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' }) + '.csv';
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (e) {
      showError('orders-error', 'Failed to export sales orders: ' + e.message);
    } finally {
      button.textContent = originalText;
      button.classList.remove('loading');
      button.disabled = false;
    }
  }

  async function exportOrdersMatrix() {
    const button = document.getElementById('orders-matrix-export-btn');
    const originalText = button.textContent;
    button.textContent = 'Exporting...';
    button.classList.add('loading');
    button.disabled = true;
    hideError('orders-error');

    try {
      const response = await fetch(SALES_API_BASE + '/export/orders-matrix.xlsx', {
        headers: { 'X-API-Key': SALES_API_KEY }
      });
      if (!response.ok) {
        const body = await response.text();
        throw new Error(`HTTP ${response.status}: ${body}`);
      }
      const blob = await response.blob();
      const disposition = response.headers.get('Content-Disposition') || '';
      const filenameMatch = disposition.match(/filename="?([^";]+)"?/i);
      const filename = filenameMatch
        ? filenameMatch[1]
        : 'CNS_Open_Orders_Matrix_' + new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' }) + '.xlsx';
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    } catch (e) {
      showError('orders-error', 'Failed to export orders matrix: ' + e.message);
    } finally {
      button.textContent = originalText;
      button.classList.remove('loading');
      button.disabled = false;
    }
  }

  function orderReadyPill(order) {
    if (!order.ready) return '';
    const parts = ['&#10003; READY'];
    if (order.ready_by) parts.push(escHtml(order.ready_by));
    if (order.ready_at) parts.push(escHtml(formatReadyTime(order.ready_at)));
    return `<span class="so-ready-pill">${parts.join(' &middot; ')}</span>`;
  }

  async function refreshOrders() {
    hideError('orders-error');
    const container = document.getElementById('orders-table-container');
    container.innerHTML = '<div class="loading-indicator">Loading sales orders...</div>';
    try {
      const data = await fetchSalesAPI('/sales/orders?limit=200');
      state.ordersData = data.orders || [];
      state.ordersLoaded = true;
      updateShipByCalendarIndicators();
      renderOrdersList();
    } catch (e) {
      container.innerHTML = '';
      showError('orders-error', 'Failed to load sales orders: ' + e.message);
    }
  }

  function updateShipByCalendarIndicators() {
    const counts = {};
    for (const order of state.ordersData) {
      if (!order.requested_ship_date || !SALES_ORDER_OPEN_STATUSES.includes(order.status)) continue;
      counts[order.requested_ship_date] = (counts[order.requested_ship_date] || 0) + 1;
    }
    window.dispatchEvent(new CustomEvent('factory-ledger:ship-dates', { detail: { counts } }));
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
    html += '<th class="order-expand-col" aria-label="Expand"></th><th class="order-ready-col" aria-label="Factory Ready"></th><th>SO #</th><th>Customer</th><th>Order Date</th><th>Ship By</th><th>Status</th><th class="num">Pallets</th><th class="num">Remaining</th>';
    html += '</tr></thead><tbody>';

    for (const o of orders) {
      const overdue = isOrderOverdue(o);
      html += `<tr class="order-row ${o.ready ? 'so-ready' : ''}" data-order-id="${o.order_id}">`;
      html += `<td class="order-expand-cell"><button type="button" class="order-expand-toggle" data-order-id="${o.order_id}" aria-expanded="false" aria-controls="order-lines-${o.order_id}" title="Show line items"><span class="order-expand-caret">&#9656;</span></button></td>`;
      html += `<td class="order-ready-cell"><input type="checkbox" class="order-ready-checkbox" data-order-id="${o.order_id}" ${o.ready ? 'checked' : ''} title="Factory Ready"></td>`;
      html += `<td><span class="order-link">${escHtml(o.order_number)}</span></td>`;
      html += `<td>${escHtml(o.customer)}</td>`;
      html += `<td>${formatDateShort(o.order_date)}</td>`;
      html += `<td class="ship-by-cell ${overdue ? 'date-overdue' : ''}">${formatShipByDate(o.requested_ship_date)}</td>`;
      html += `<td><span class="so-badge status-${o.status}">${soStatusLabel(o.status)}</span>${orderReadyPill(o)}</td>`;
      html += `<td class="num order-pallet-total">${escHtml(calculateOrderPallets(o.pallet_lines || [], 'unit_count').display)}</td>`;
      html += `<td class="num">${o.remaining_units ? fmtWt(o.remaining_lb) + ' lb &middot; ' + fmtInt(o.remaining_units) + ' units' : fmtLbs(o.remaining_lb)}</td>`;
      html += `</tr>`;
      // Hidden inline detail row — line items loaded on demand when expanded
      html += `<tr id="order-lines-${o.order_id}" class="order-lines-row hidden" data-order-id="${o.order_id}"><td colspan="9"><div class="order-lines-content"></div></td></tr>`;
    }

    html += '</tbody></table>';
    container.innerHTML = html;

    // Bind row clicks — clicking the row (incl. the SO number) opens the full detail page
    container.querySelectorAll('.order-row').forEach(row => {
      row.addEventListener('click', () => {
        const orderId = row.dataset.orderId;
        // Save scroll position
        state.ordersScrollTop = document.getElementById('tab-orders').scrollTop || window.scrollY;
        openOrderDetail(orderId);
      });
    });

    // Bind the separate inline expand/collapse controls
    bindOrderExpandToggles(container);
    bindOrderReadyToggles(container);
  }

  function renderOrderLinesContent(order) {
    const lines = (order && order.lines) || [];
    const listOrder = state.ordersData.find(o => String(o.order_id) === String(order.order_id)) || order;
    const readyNote = listOrder.note || '';
    let html = '<div class="order-ready-drawer">';
    html += '<label>Factory Ready note</label>';
    html += `<div class="order-ready-note-row"><input type="text" class="order-ready-note-input" data-order-id="${order.order_id}" value="${escAttr(readyNote)}" placeholder="Optional note for the floor">`;
    html += `<button type="button" class="btn-sm order-ready-note-save" data-order-id="${order.order_id}">Save</button></div>`;
    if (readyNote) html += `<div class="order-ready-note-text">${escHtml(readyNote)}</div>`;
    html += '</div>';

    if (lines.length === 0) {
      return html + '<div class="order-lines-empty">No line items on this order.</div>';
    }

    const totalPallets = calculateOrderPallets(lines, 'unit_count');
    html += `<div class="order-pallet-summary"><span>Order pallets</span><strong>${escHtml(totalPallets.display)}</strong></div>`;
    html += '<table class="order-lines-table"><thead><tr>';
    html += '<th>SKU</th><th>Product</th><th class="num">Ordered</th><th class="num">Pallets</th><th>UoM</th><th class="num">Remaining</th>';
    html += '</tr></thead><tbody>';
    for (const l of lines) {
      const nonWeight = l.is_non_weight;
      const uom = nonWeight ? 'units' : (l.uom || 'lb');
      const orderedQty = nonWeight
        ? fmtInt(l.unit_quantity != null ? l.unit_quantity : l.quantity_lb)
        : fmtWt(l.quantity_lb) + (l.unit_count != null ? ` <small>(${fmtInt(l.unit_count)} cs)</small>` : '');
      const remVal = l.remaining_lb != null ? l.remaining_lb : ((l.quantity_lb || 0) - (l.quantity_shipped_lb || 0));
      const remaining = remVal == null ? '&mdash;' : (nonWeight
        ? fmtInt(remVal)
        : fmtWt(remVal) + (l.remaining_units != null ? ` <small>(${fmtInt(l.remaining_units)} cs)</small>` : ''));
      const linePallets = calculateLinePallets(l, l.unit_count);
      html += '<tr>';
      html += `<td class="order-line-sku">${escHtml(l.sku || '—')}</td>`;
      html += `<td>${escHtml(l.product || l.name || '—')}</td>`;
      html += `<td class="num">${orderedQty}</td>`;
      html += `<td class="num order-line-pallets">${escHtml(linePallets.display)}</td>`;
      html += `<td>${escHtml(uom)}</td>`;
      html += `<td class="num">${remaining}</td>`;
      html += '</tr>';
    }
    html += '</tbody></table>';
    return html;
  }

  async function postOrderReady(order, ready, note) {
    return fetchSalesAPI('/sales-orders/' + encodeURIComponent(order.order_number) + '/ready', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ready, by: 'floor', note: note || null })
    });
  }

  function updateCachedOrderReady(orderId, flag) {
    const order = state.ordersData.find(o => String(o.order_id) === String(orderId));
    if (!order) return;
    order.ready = Boolean(flag.ready);
    order.ready_at = flag.ready_at || null;
    order.ready_by = flag.ready_by || 'floor';
    order.note = flag.note || null;
  }

  function bindOrderReadyNoteControls(container) {
    container.querySelectorAll('.order-ready-note-save').forEach(btn => {
      btn.addEventListener('click', async (ev) => {
        ev.stopPropagation();
        const orderId = btn.dataset.orderId;
        const order = state.ordersData.find(o => String(o.order_id) === String(orderId));
        const input = container.querySelector(`.order-ready-note-input[data-order-id="${orderId}"]`);
        if (!order || !input) return;
        const oldNote = order.note || null;
        order.note = input.value.trim() || null;
        btn.disabled = true;
        try {
          const saved = await postOrderReady(order, Boolean(order.ready), order.note);
          updateCachedOrderReady(orderId, saved);
          renderOrdersList();
        } catch (e) {
          order.note = oldNote;
          showError('orders-error', 'Factory Ready note save failed: ' + e.message);
        } finally {
          btn.disabled = false;
        }
      });
    });
  }

  function bindOrderReadyToggles(container) {
    container.querySelectorAll('.order-ready-checkbox').forEach(cb => {
      cb.addEventListener('click', ev => ev.stopPropagation());
      cb.addEventListener('change', async (ev) => {
        ev.stopPropagation();
        const orderId = cb.dataset.orderId;
        const order = state.ordersData.find(o => String(o.order_id) === String(orderId));
        if (!order) return;

        const oldFlag = {
          ready: Boolean(order.ready),
          ready_at: order.ready_at || null,
          ready_by: order.ready_by || 'floor',
          note: order.note || null
        };
        const nextReady = cb.checked;
        order.ready = nextReady;
        order.ready_at = nextReady ? (order.ready_at || new Date().toISOString()) : null;
        order.ready_by = 'floor';

        renderOrdersList();
        try {
          const saved = await postOrderReady(order, nextReady, order.note || null);
          updateCachedOrderReady(orderId, saved);
          renderOrdersList();
        } catch (e) {
          Object.assign(order, oldFlag);
          renderOrdersList();
          showError('orders-error', 'Factory Ready update failed: ' + e.message);
        }
      });
    });
  }

  function bindOrderExpandToggles(container) {
    container.querySelectorAll('.order-expand-toggle').forEach(btn => {
      btn.addEventListener('click', async (ev) => {
        // Keep this control independent of the row click (which opens the detail page)
        ev.stopPropagation();
        const orderId = btn.dataset.orderId;
        const detailRow = container.querySelector(`.order-lines-row[data-order-id="${orderId}"]`);
        if (!detailRow) return;
        const contentCell = detailRow.querySelector('.order-lines-content');

        const expanding = detailRow.classList.contains('hidden');
        detailRow.classList.toggle('hidden', !expanding);
        btn.classList.toggle('expanded', expanding);
        btn.setAttribute('aria-expanded', expanding ? 'true' : 'false');
        btn.setAttribute('title', expanding ? 'Hide line items' : 'Show line items');

        // Collapsing, or already rendered — nothing more to do (allows multiple open at once)
        if (!expanding || detailRow.dataset.loaded === 'true') return;

        contentCell.innerHTML = '<div class="loading-indicator">Loading line items…</div>';
        try {
          let data = state.orderLinesCache[orderId];
          if (!data) {
            data = await fetchSalesAPI('/sales/orders/' + orderId);
            state.orderLinesCache[orderId] = data;
          }
          contentCell.innerHTML = renderOrderLinesContent(data);
          bindOrderReadyNoteControls(contentCell);
          detailRow.dataset.loaded = 'true';
        } catch (e) {
          contentCell.innerHTML = `<div class="order-lines-error">Failed to load line items: ${escHtml(e.message)}</div>`;
        }
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

  function flattenFinishedGoodsInventory(data) {
    const inventory = {};
    for (const panel of (data.panels || [])) {
      for (const product of (panel.products || [])) {
        inventory[(product.product_name || '').toLowerCase()] = {
          productName: product.product_name,
          onHandLbs: Number(product.on_hand_lbs || 0),
          caseWeightLb: product.case_weight_lb || panel.case_weight_lb || null
        };
      }
    }
    return inventory;
  }

  async function getSalesOrderInventory() {
    if (state.salesOrderInventory.data) return state.salesOrderInventory.data;
    if (!state.salesOrderInventory.promise) {
      state.salesOrderInventory.error = null;
      state.salesOrderInventory.promise = fetchAPI('/inventory/finished-goods')
        .then(data => {
          state.salesOrderInventory.data = flattenFinishedGoodsInventory(data);
          return state.salesOrderInventory.data;
        })
        .catch(err => {
          state.salesOrderInventory.error = err;
          state.salesOrderInventory.promise = null;
          throw err;
        });
    }
    return state.salesOrderInventory.promise;
  }

  function renderOrderInventoryContent(line, inventoryByProduct) {
    if (!line) return '<div class="loading-indicator order-inventory-message">Unable to load inventory</div>';
    const productName = line.product || line.name || '';
    const inventory = inventoryByProduct[(productName || '').toLowerCase()];
    const caseWeight = inventory ? inventory.caseWeightLb : (line.case_size_lb || null);
    const onHandLbs = inventory ? inventory.onHandLbs : 0;
    const remainingLbs = line.remaining_lb != null ? line.remaining_lb : ((line.quantity_lb || 0) - (line.quantity_shipped_lb || 0));
    const onHandUnits = inventoryUnitCount(onHandLbs, caseWeight);
    const remainingUnits = line.remaining_units != null ? line.remaining_units : inventoryUnitCount(remainingLbs, caseWeight);
    const deltaUnits = onHandUnits != null && remainingUnits != null ? onHandUnits - remainingUnits : null;
    const onHandPallets = salesOrderLinePallets(line, onHandUnits);
    const remainingPallets = salesOrderLinePallets(line, remainingUnits);
    const deltaPallets = salesOrderLinePallets(line, deltaUnits == null ? null : Math.abs(deltaUnits));
    const deltaClass = deltaUnits < 0 ? 'inventory-delta-negative' : 'inventory-delta-positive';
    const deltaPrefix = deltaUnits > 0 ? '+' : (deltaUnits < 0 ? '\u2212' : '');
    const deltaValue = deltaUnits == null ? '\u2014' : deltaPrefix + formatInventoryUnits(Math.abs(deltaUnits), deltaPallets);

    return '<table class="order-inventory-table"><tbody>' +
      `<tr><th>On Hand</th><td>${formatInventoryUnits(onHandUnits, onHandPallets)}</td></tr>` +
      `<tr><th>Remaining</th><td>${formatInventoryUnits(remainingUnits, remainingPallets)}</td></tr>` +
      `<tr><th>Delta</th><td class="${deltaClass}">${deltaValue}</td></tr>` +
      '</tbody></table>';
  }

  function bindOrderInventoryToggles(container, lines) {
    const linesById = new Map(lines.map(line => [String(line.line_id), line]));
    container.querySelectorAll('.order-inventory-toggle').forEach(btn => {
      btn.addEventListener('click', async () => {
        const lineId = btn.dataset.lineId;
        const detailRow = document.getElementById(`order-inventory-${lineId}`);
        const detailCell = detailRow ? detailRow.querySelector('td') : null;
        if (!detailRow || !detailCell) return;

        const expanding = detailRow.classList.contains('hidden');
        detailRow.classList.toggle('hidden', !expanding);
        btn.setAttribute('aria-expanded', expanding ? 'true' : 'false');
        if (!expanding || detailRow.dataset.loaded === 'true') return;

        detailCell.innerHTML = '<div class="loading-indicator order-inventory-message">Loading\u2026</div>';
        try {
          const inventoryByProduct = await getSalesOrderInventory();
          detailCell.innerHTML = renderOrderInventoryContent(linesById.get(String(lineId)), inventoryByProduct);
          detailRow.dataset.loaded = 'true';
        } catch (e) {
          detailCell.innerHTML = '<div class="loading-indicator order-inventory-message">Unable to load inventory</div>';
        }
      });
    });
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
    const orderedUnits = totals.total_ordered_units;
    const shippedUnits = totals.total_shipped_units;
    const remainingUnits = totals.total_remaining_units;
    const lines = data.lines || [];
    const orderedPallets = calculateOrderPallets(lines, 'unit_count');
    const remainingPallets = calculateOrderPallets(lines, 'remaining_units');
    const kpiFmt = (lb, units) => units ? fmtWt(lb) + ' lb<br><small>' + fmtInt(units) + ' units</small>' : fmtLbs(lb);
    html += '<div class="order-kpi-row">';
    html += `<div class="order-kpi"><div class="kpi-label">Total Ordered</div><div class="kpi-value">${kpiFmt(totalOrdered, orderedUnits)}</div></div>`;
    html += `<div class="order-kpi"><div class="kpi-label">Shipped</div><div class="kpi-value">${kpiFmt(totalShipped, shippedUnits)}</div></div>`;
    html += `<div class="order-kpi"><div class="kpi-label">Remaining</div><div class="kpi-value">${kpiFmt(totalRemaining, remainingUnits)}</div></div>`;
    html += `<div class="order-kpi order-kpi-pallets"><div class="kpi-label">Pallets</div><div class="kpi-value">${escHtml(orderedPallets.display)}<br><small>Remaining: ${escHtml(remainingPallets.display)}</small></div></div>`;
    html += '</div>';

    // Line items
    if (lines.length > 0) {
      html += '<table class="orders-table"><thead><tr>';
      html += '<th>Product</th><th class="num">Ordered</th><th class="num">Shipped</th><th class="num">Remaining</th><th>Status</th>';
      html += '</tr></thead><tbody>';
      for (const l of lines) {
        const remaining = l.remaining_lb != null ? l.remaining_lb : ((l.quantity_lb || 0) - (l.quantity_shipped_lb || 0));
        const productName = l.product || l.name || '\u2014';
        const lineStatusClass = l.line_status === 'fulfilled' ? 'status-shipped'
          : l.line_status === 'partial' ? 'status-partial_ship'
          : l.line_status === 'cancelled' ? 'status-cancelled'
          : 'status-new';
        const isNw = l.is_non_weight;
        const lineFmt = (lb, units) => isNw ? (Number.isInteger(lb) ? lb : lb) + ' units' : (units != null ? fmtWt(lb) + ' lb &middot; ' + fmtInt(units) + ' units' : fmtLbs(lb));
        const orderedLinePallets = calculateLinePallets(l, l.unit_count);
        const remainingLinePallets = calculateLinePallets(l, l.remaining_units);
        html += '<tr>';
        html += `<td><div class="order-product-cell"><span>${escHtml(productName)}</span><button type="button" class="btn-sm order-inventory-toggle" data-line-id="${l.line_id}" aria-expanded="false" aria-controls="order-inventory-${l.line_id}">Inventory</button></div></td>`;
        html += `<td class="num">${lineFmt(l.quantity_lb, l.unit_count)}<small class="pallet-secondary">${escHtml(orderedLinePallets.display)}</small></td>`;
        html += `<td class="num">${lineFmt(l.quantity_shipped_lb, l.shipped_units)}</td>`;
        html += `<td class="num">${lineFmt(remaining, l.remaining_units)}<small class="pallet-secondary">${escHtml(remainingLinePallets.display)}</small></td>`;
        html += `<td><span class="so-badge ${lineStatusClass}">${escHtml(l.line_status || 'pending')}</span></td>`;
        html += '</tr>';
        html += `<tr id="order-inventory-${l.line_id}" class="order-inventory-row hidden"><td colspan="5"></td></tr>`;
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
    bindOrderInventoryToggles(container, lines);
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

    document.getElementById('orders-hide-ready').addEventListener('change', () => {
      if (state.ordersLoaded) renderOrdersList();
    });

    // Refresh button
    document.getElementById('orders-refresh-btn').addEventListener('click', refreshOrders);
    document.getElementById('orders-export-btn').addEventListener('click', exportOrdersCsv);
    document.getElementById('orders-matrix-export-btn').addEventListener('click', exportOrdersMatrix);

    // Back button
    document.getElementById('order-back-btn').addEventListener('click', closeOrderDetail);
  }

  // ── System Health Badge ──
  async function refreshHealthBadge() {
    const badge = document.getElementById('health-badge');
    try {
      const res = await fetch('https://fastapi-production-b73a.up.railway.app/audit/integrity');
      if (!res.ok) throw new Error('Failed');
      const data = await res.json();
      const score = data.score;
      badge.textContent = score;
      badge.className = 'health-badge';
      if (score >= 90) badge.classList.add('health-green');
      else if (score >= 70) badge.classList.add('health-yellow');
      else badge.classList.add('health-red');

      const failChecks = data.checks.filter(c => c.status === 'fail');
      if (failChecks.length) {
        badge.title = 'Health: ' + score + '/100 — ' + failChecks.map(c => c.name + ' (' + c.severity + ')').join(', ');
      } else {
        badge.title = 'System Health: ' + score + '/100 — All checks pass';
      }
    } catch {
      badge.textContent = '?';
      badge.className = 'health-badge';
      badge.title = 'Health check unavailable';
    }
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
    ops.push(refreshHealthBadge());

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
