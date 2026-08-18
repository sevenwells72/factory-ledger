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
    productionCalendarDays: [],
    selectedProductionDate: null,
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
    const units = Math.floor(v / cs);
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
    if (explicitCategory.includes('graham')) return 'graham';

    const productName = String(product?.product_name || product?.name || '').trim();
    const normalizedName = productName.replace(/^Batch\s+/i, '');
    if (/coconut/i.test(normalizedName)) return 'coconut';
    if (/granola/i.test(normalizedName) || /^(CQ|SS)\b/i.test(normalizedName)) return 'granola';
    if (/graham/i.test(normalizedName)) return 'graham';
    console.warn('Production calendar product categorized as OTHER:', product);
    return 'other';
  }

  function productionBatchCount(batch) {
    if (batch.batch_count != null) return Number(batch.batch_count);
    if (batch.made_unit_size_lbs) return Number(batch.total_lbs) / Number(batch.made_unit_size_lbs);
    if (batch.standard_batch_size_lbs) return Number(batch.total_lbs) / Number(batch.standard_batch_size_lbs);
    return null;
  }

  function productionUnitCount(finishedGood) {
    if (finishedGood.unit_count != null) return Number(finishedGood.unit_count);
    if (finishedGood.case_size_lb) return Math.floor(Number(finishedGood.total_lbs) / Number(finishedGood.case_size_lb));
    return null;
  }

  const MADE_CATEGORY_DEFS = [
    { key: 'coconut', family: 'coconut', label: 'Coconut pans', singular: 'pan', plural: 'pans' },
    { key: 'granola', family: 'granola', label: 'Granola batches', singular: 'batch', plural: 'batches' },
    { key: 'graham', family: 'graham', label: 'Graham batches', singular: 'batch', plural: 'batches' }
  ];

  const PACKED_CATEGORY_DEFS = [
    { key: 'granola-10lb', family: 'granola', label: 'Granola 10 lb' },
    { key: 'granola-25lb', family: 'granola', label: 'Granola 25 lb' },
    { key: 'granola-bagged', family: 'granola', label: 'Granola bagged' },
    { key: 'coconut', family: 'coconut', label: 'Coconut' },
    { key: 'graham', family: 'graham', label: 'Graham' }
  ];

  const PRODUCTION_FAMILY_DEFS = [
    {
      key: 'coconut', label: 'Coconut', madeKey: 'coconut', madeLabel: 'Made · pans',
      packed: [{ key: 'coconut', detailLabel: 'Packed · labels' }]
    },
    {
      key: 'granola', label: 'Granola', madeKey: 'granola', madeLabel: 'Made · batches',
      packed: [
        { key: 'granola-10lb', detailLabel: 'Packed · 10 lb' },
        { key: 'granola-25lb', detailLabel: 'Packed · 25 lb' },
        { key: 'granola-bagged', detailLabel: 'Packed · bagged' }
      ]
    },
    {
      key: 'graham', label: 'Graham', madeKey: 'graham', madeLabel: 'Made · batches',
      packed: [{ key: 'graham', detailLabel: 'Packed · labels' }]
    }
  ];

  function formatProductionCount(count) {
    if (count == null || !Number.isFinite(Number(count))) return '';
    const numeric = Number(count);
    return Number.isInteger(numeric) ? fmtInt(numeric) : numeric.toFixed(1);
  }

  function formatProductionUnit(count, singular, plural) {
    const formatted = formatProductionCount(count);
    if (!formatted) return '';
    return `${formatted} ${Number(count) === 1 ? singular : plural}`;
  }

  function buildProductionDaySummary(day) {
    const madeByKey = Object.fromEntries(MADE_CATEGORY_DEFS.map(def => [
      def.key,
      { ...def, count: 0, items: [] }
    ]));
    const packedByKey = Object.fromEntries(PACKED_CATEGORY_DEFS.map(def => [
      def.key,
      { ...def, count: 0, items: [] }
    ]));

    for (const batch of (day.batches || [])) {
      const category = getProductCategory(batch);
      const group = madeByKey[category];
      const count = productionBatchCount(batch);
      if (!group || count == null || !Number.isFinite(count) || count <= 0) {
        console.warn('Production calendar made row omitted from classified counts:', batch);
        continue;
      }
      group.count += count;
      group.items.push({
        name: String(batch.product_name || '').trim(),
        count
      });
    }

    for (const finishedGood of (day.finished_goods || [])) {
      const category = getProductCategory(finishedGood);
      let packedKey = null;
      if (category === 'granola' && ['10lb', '25lb', 'bagged'].includes(finishedGood.pack_format)) {
        packedKey = `granola-${finishedGood.pack_format}`;
      } else if (category === 'coconut' || category === 'graham') {
        packedKey = category;
      }

      const group = packedByKey[packedKey];
      const cases = productionUnitCount(finishedGood);
      if (!group || cases == null || !Number.isFinite(cases) || cases <= 0) {
        console.warn('Production calendar packed row omitted from classified case counts:', finishedGood);
        continue;
      }
      group.count += cases;
      group.items.push({
        sku: String(finishedGood.sku || '').trim(),
        name: String(finishedGood.product_name || '').trim(),
        cases
      });
    }

    const made = MADE_CATEGORY_DEFS.map(def => madeByKey[def.key]).filter(group => group.count > 0);
    const packed = PACKED_CATEGORY_DEFS.map(def => packedByKey[def.key]).filter(group => group.count > 0);
    const families = PRODUCTION_FAMILY_DEFS.map(def => {
      const madeGroup = madeByKey[def.madeKey];
      return {
        key: def.key,
        label: def.label,
        made: madeGroup.count > 0 ? { ...madeGroup, detailLabel: def.madeLabel } : null,
        packed: def.packed.map(packedDef => {
          const packedGroup = packedByKey[packedDef.key];
          return packedGroup.count > 0
            ? { ...packedGroup, detailLabel: packedDef.detailLabel }
            : null;
        }).filter(Boolean)
      };
    }).filter(family => family.made || family.packed.length > 0);
    return {
      made,
      packed,
      families,
      hasProduction: families.length > 0
    };
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

  function createdAtMeta(record) {
    if (!record || !record.created_date || !record.created_time) return '';
    const source = record.created_at_source || 'unknown';
    let provenance = '';
    if (source === 'migration_backfill_039') provenance = ' · backfilled';
    if (source === 'legacy_unverified') provenance = ' · legacy';
    const title = `Database created_at (${source})`;
    return `<div class="created-at-meta" title="${escAttr(title)}">Entered: ${escHtml(record.created_date)} ${escHtml(record.created_time)}${escHtml(provenance)}</div>`;
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
    const detail = document.getElementById('production-calendar-detail');
    container.innerHTML = '<div class="loading-indicator">Loading production data...</div>';
    detail.innerHTML = '';
    detail.classList.add('hidden');
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

    const dayModels = displayDays.map(day => ({ day, summary: buildProductionDaySummary(day) }));
    state.productionCalendarDays = dayModels;
    if (!dayModels.some(model => (
      model.day.date === state.selectedProductionDate && model.summary.hasProduction
    ))) {
      state.selectedProductionDate = null;
    }

    let html = '';
    for (const model of dayModels) {
      const { day, summary } = model;
      const isToday = day.date === todayStr;
      const isSelected = day.date === state.selectedProductionDate;
      const classes = ['day-card'];
      if (isToday) classes.push('today');
      if (summary.hasProduction) classes.push('has-production', 'day-card-trigger');
      if (isSelected) classes.push('selected');
      if (!summary.hasProduction && state.calendarMode === 'month') classes.push('empty');

      if (summary.hasProduction) {
        html += `<button type="button" class="${classes.join(' ')}" data-production-date="${escAttr(day.date)}" aria-expanded="${isSelected}" aria-controls="production-calendar-detail">`;
      } else {
        html += `<div class="${classes.join(' ')}">`;
      }
      html += `<span class="day-card-date"><span class="day-name">${escHtml(day.day_name)}</span> &mdash; ${escHtml(day.date)}</span>`;

      if (summary.hasProduction) {
        if (summary.made.length > 0) {
          html += '<span class="calendar-summary-section"><span class="day-section-label">Made</span>';
          for (const group of summary.made) {
            html += `<span class="calendar-summary-row"><span class="production-category-label category-${escAttr(group.family)}">${escHtml(group.label)}</span><strong>${escHtml(formatProductionCount(group.count))}</strong></span>`;
          }
          html += '</span>';
        }
        if (summary.packed.length > 0) {
          html += '<span class="calendar-summary-section packed"><span class="day-section-label">Packed &middot; cases</span>';
          for (const group of summary.packed) {
            html += `<span class="calendar-summary-row"><span class="production-category-label category-${escAttr(group.family)}">${escHtml(group.label)}</span><strong>${escHtml(formatProductionCount(group.count))}</strong></span>`;
          }
          html += '</span>';
        }
        html += '<span class="day-card-detail-hint">View details</span>';
      } else {
        html += '<div class="no-production">No production</div>';
      }

      html += summary.hasProduction ? '</button>' : '</div>';
    }
    container.innerHTML = html;

    container.querySelectorAll('.day-card-trigger').forEach(card => {
      card.addEventListener('click', () => {
        const selectedDate = card.dataset.productionDate;
        state.selectedProductionDate = state.selectedProductionDate === selectedDate ? null : selectedDate;
        updateProductionDaySelection(container, true);
      });
    });
    updateProductionDaySelection(container, false);
  }

  function productionDetailDate(day) {
    const parsed = new Date(`${day.date}T12:00:00`);
    if (Number.isNaN(parsed.getTime())) return day.date;
    return parsed.toLocaleDateString('en-US', {
      weekday: 'long', month: 'long', day: 'numeric', year: 'numeric'
    });
  }

  function renderMadeDetailRows(group) {
    let html = '';
    for (const item of group.items) {
      html += '<div class="production-detail-row">';
      html += `<div class="production-detail-name">${escHtml(item.name)}</div>`;
      html += `<div class="production-detail-count">${escHtml(formatProductionUnit(item.count, group.singular, group.plural))}</div>`;
      html += '</div>';
    }
    return html;
  }

  function renderPackedDetailRows(group) {
    let html = '';
    for (const item of group.items) {
      html += '<div class="production-detail-row">';
      html += '<div class="production-detail-product">';
      html += `<div class="production-detail-name">${escHtml(item.name)}</div>`;
      if (item.sku) html += `<div class="production-detail-sku">SKU ${escHtml(item.sku)}</div>`;
      html += '</div>';
      html += `<div class="production-detail-count">${escHtml(formatProductionUnit(item.cases, 'case', 'cases'))}</div>`;
      html += '</div>';
    }
    return html;
  }

  function renderProductionDetailFamily(family) {
    let html = `<section class="production-detail-family family-${escAttr(family.key)}">`;
    html += `<h4>${escHtml(family.label)}</h4>`;
    if (family.made) {
      html += `<section class="production-detail-subsection"><h5>${escHtml(family.made.detailLabel)}</h5>`;
      html += renderMadeDetailRows(family.made);
      html += '</section>';
    }
    for (const group of family.packed) {
      html += `<section class="production-detail-subsection"><h5>${escHtml(group.detailLabel)}</h5>`;
      html += renderPackedDetailRows(group);
      html += '</section>';
    }
    html += '</section>';
    return html;
  }

  function updateProductionDaySelection(container, shouldScroll) {
    container.querySelectorAll('.day-card-trigger').forEach(card => {
      const isSelected = card.dataset.productionDate === state.selectedProductionDate;
      card.classList.toggle('selected', isSelected);
      card.setAttribute('aria-expanded', String(isSelected));
    });

    const detail = document.getElementById('production-calendar-detail');
    const model = state.productionCalendarDays.find(item => (
      item.day.date === state.selectedProductionDate
    ));
    if (!model) {
      detail.innerHTML = '';
      detail.classList.add('hidden');
      return;
    }

    let html = '<div class="production-detail-header">';
    html += '<div><div class="production-detail-eyebrow">Production detail</div>';
    html += `<h3>${escHtml(productionDetailDate(model.day))}</h3></div>`;
    html += '<button type="button" class="btn-sm production-detail-close" aria-label="Close production detail">Close</button>';
    html += '</div>';

    html += '<div class="production-detail-grid">';
    for (const family of model.summary.families) html += renderProductionDetailFamily(family);
    html += '</div>';

    detail.innerHTML = html;
    detail.classList.remove('hidden');
    detail.querySelector('.production-detail-close').addEventListener('click', () => {
      state.selectedProductionDate = null;
      updateProductionDaySelection(container, false);
    });
    if (shouldScroll) detail.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
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

  const BATCH_FAMILY_ORDER = ['coconut', 'granola', 'graham', 'chips', 'sprinkles', 'other'];
  const BATCH_FAMILY_LABELS = {
    coconut: 'Coconut',
    granola: 'Granola',
    graham: 'Graham',
    chips: 'Chips',
    sprinkles: 'Sprinkles',
    other: 'Other'
  };

  function estimatedBatchesOnHand(batch) {
    if (batch == null) return null;
    if (batch.batch_count != null && batch.batch_count !== '') {
      const provided = Number(batch.batch_count);
      if (Number.isFinite(provided)) return provided;
    }
    const onHand = Number(batch.on_hand_lbs);
    if (!Number.isFinite(onHand)) return null;
    const made = Number(batch.made_unit_size_lbs);
    if (Number.isFinite(made) && made > 0) return onHand / made;
    const base = Number(batch.standard_batch_size_lbs);
    if (!Number.isFinite(base) || base <= 0) return null;
    const yieldMul = Number(batch.yield_multiplier);
    const multiplier = Number.isFinite(yieldMul) && yieldMul > 0 ? yieldMul : 1;
    return onHand / (base * multiplier);
  }

  function renderBatchFamilyTable(batches) {
    let html = '<table class="inv-table"><thead><tr><th>Batch</th><th class="num">On Hand (lb)</th><th>Est. Batches</th></tr></thead><tbody>';
    for (const b of batches) {
      const rowId = 'batch-' + b.product_name.replace(/\W/g, '_');
      const estRaw = estimatedBatchesOnHand(b);
      const estBatches = estRaw == null ? null : Number(estRaw).toFixed(1);
      html += `<tr class="expandable" data-expand="${rowId}">`;
      html += `<td>${escHtml(b.product_name)}</td>`;
      html += `<td class="num">${fmt(b.on_hand_lbs)}</td>`;
      html += `<td>`;
      if (estBatches !== null) {
        const n = Number(estBatches);
        const batchClass = n >= 5 ? 'stock-healthy' : n >= 2 ? 'stock-low' : 'stock-critical';
        html += `<span class="badge ${batchClass}">${estBatches} batches</span>`;
      } else {
        html += `<span class="badge unknown">batches: unknown</span>`;
      }
      html += `</td></tr>`;
      html += `<tbody class="lot-breakdown" id="${rowId}">`;
      if (b.lots && b.lots.length > 0) {
        for (const lot of b.lots) {
          const bc = lot.batch_count != null
            ? lot.batch_count
            : estimatedBatchesOnHand({
                on_hand_lbs: lot.on_hand_lbs,
                made_unit_size_lbs: b.made_unit_size_lbs,
                standard_batch_size_lbs: lot.default_batch_lb || b.standard_batch_size_lbs,
                yield_multiplier: b.yield_multiplier
              });
          const lotQty = bc != null
            ? fmt(lot.on_hand_lbs) + ' lb &middot; ' + Number(bc).toFixed(1) + ' batches'
            : fmt(lot.on_hand_lbs) + ' lb';
          html += `<tr class="lot-row"><td><span class="lot-link" data-lot="${escHtml(lot.lot_code)}" data-product-id="${lot.product_id || ''}">${escHtml(lot.lot_code)}</span></td>`;
          html += `<td class="num">${lotQty}</td><td></td></tr>`;
        }
      } else {
        html += `<tr class="lot-row"><td colspan="3" style="color:var(--text-muted)">No lots on hand</td></tr>`;
      }
      html += `</tbody>`;
    }
    html += '</tbody></table>';
    return html;
  }

  function renderBatchInventory(data, container) {
    const batches = data.batches || [];
    let html = '';
    if (batches.length > 0) {
      const grouped = {};
      for (const b of batches) {
        const family = BATCH_FAMILY_ORDER.includes(b.production_family) ? b.production_family : 'other';
        if (!grouped[family]) grouped[family] = [];
        grouped[family].push(b);
      }
      const familiesPresent = BATCH_FAMILY_ORDER.filter(key => grouped[key] && grouped[key].length);
      const showFamilyHeadings = familiesPresent.length > 1;
      for (const family of familiesPresent) {
        if (showFamilyHeadings) {
          html += `<h3 class="batch-family-heading family-${family}">${escHtml(BATCH_FAMILY_LABELS[family] || family)}</h3>`;
        }
        html += renderBatchFamilyTable(grouped[family]);
      }
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
        const itemUnits = cat.items.map(item => String(item.uom || cat.unit || 'lb').trim() || 'lb');
        const uniqueUnits = [...new Set(itemUnits)];
        const headerUnit = uniqueUnits.length === 1 ? uniqueUnits[0] : null;
        const qtyHeader = headerUnit ? `On Hand (${escHtml(headerUnit)})` : 'On Hand';
        html += `<table class="inv-table"><thead><tr><th>Ingredient</th><th class="num">${qtyHeader}</th></tr></thead><tbody>`;
        for (const item of cat.items) {
          const rowId = panelId + '-' + item.name.replace(/\W/g, '_');
          const uom = String(item.uom || cat.unit || 'lb').trim() || 'lb';
          const qtyLabel = headerUnit ? fmt(item.on_hand) : `${fmt(item.on_hand)} ${escHtml(uom)}`;
          html += `<tr class="expandable" data-expand="${rowId}">`;
          html += `<td>${escHtml(item.name)}</td><td class="num">${qtyLabel}</td>`;
          html += `</tr>`;
          // Lot breakdown
          html += `<tbody class="lot-breakdown" id="${rowId}">`;
          if (item.lots && item.lots.length > 0) {
            for (const lot of item.lots) {
              const lotUom = String(lot.uom || uom).trim() || uom;
              const lotQty = headerUnit ? fmt(lot.on_hand_lbs) : `${fmt(lot.on_hand_lbs)} ${escHtml(lotUom)}`;
              html += `<tr class="lot-row"><td><span class="lot-link" data-lot="${escHtml(lot.lot_code)}" data-product-id="${lot.product_id || ''}">${escHtml(lot.lot_code)}</span></td>`;
              html += `<td class="num">${lotQty}</td></tr>`;
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

  // ── Activity: show-more truncation (display-only; full data already fetched) ──
  const ACTIVITY_PREVIEW_ROWS = 4;

  function overflowClass(idx) {
    return idx >= ACTIVITY_PREVIEW_ROWS ? ' overflow-row overflow-hidden' : '';
  }

  function showMoreFooter(total, colspan) {
    const hidden = total - ACTIVITY_PREVIEW_ROWS;
    if (hidden <= 0) return '';
    const label = `Show all (${hidden} more)`;
    return `<tfoot><tr class="show-more-row"><td colspan="${colspan}"><button type="button" class="show-more-btn" data-more-label="${escAttr(label)}">${escHtml(label)}</button></td></tr></tfoot>`;
  }

  function bindShowMore(container) {
    container.querySelectorAll('.show-more-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const table = btn.closest('table');
        const expand = !btn.classList.contains('expanded');
        table.querySelectorAll('tr.overflow-row').forEach(tr => tr.classList.toggle('overflow-hidden', !expand));
        btn.classList.toggle('expanded', expand);
        btn.textContent = expand ? 'Show less' : btn.dataset.moreLabel;
      });
    });
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
    let html = '<table class="activity-table"><thead><tr><th>Occurred / Entered</th><th>Product(s)</th><th class="num">Qty (lb)</th><th>Customer</th><th>Ref</th></tr></thead><tbody>';
    for (const [idx, s] of shipments.entries()) {
      const rowId = 'ship-' + s.transaction_id;
      const products = (s.lines || []).map(l => l.product_name).filter(Boolean);
      const uniqueProducts = [...new Set(products)];
      html += `<tr class="expandable${overflowClass(idx)}" data-expand="${rowId}">`;
      html += `<td><div>${escHtml(s.date)} ${escHtml(s.time)}</div>${createdAtMeta(s)}</td>`;
      html += `<td>${uniqueProducts.map(escHtml).join(', ')}</td>`;
      html += `<td class="num">${s.total_units ? fmt(s.total_lbs) + ' lb &middot; ' + fmtInt(s.total_units) + ' units' : fmt(s.total_lbs) + ' lb'}</td>`;
      html += `<td>${escHtml(s.customer_name || '\u2014')}</td>`;
      html += `<td>${escHtml(s.order_reference || '\u2014')}</td>`;
      html += `</tr>`;
      // Detail row
      html += `<tr class="activity-detail${overflowClass(idx)}" id="${rowId}"><td colspan="5">`;
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
    html += '</tbody>' + showMoreFooter(shipments.length, 5) + '</table>';
    container.innerHTML = html;
    bindExpandableRows(container);
    bindLotLinks(container);
    bindShowMore(container);
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
    let html = '<table class="activity-table"><thead><tr><th>Occurred / Entered</th><th>Product(s)</th><th class="num">Qty (lb)</th><th>Supplier</th><th>BOL</th></tr></thead><tbody>';
    for (const [idx, r] of receipts.entries()) {
      const rowId = 'recv-' + r.transaction_id;
      const products = (r.lines || []).map(l => l.product_name).filter(Boolean);
      const uniqueProducts = [...new Set(products)];
      html += `<tr class="expandable${overflowClass(idx)}" data-expand="${rowId}">`;
      html += `<td><div>${escHtml(r.date)} ${escHtml(r.time)}</div>${createdAtMeta(r)}</td>`;
      html += `<td>${uniqueProducts.map(escHtml).join(', ')}</td>`;
      const recvUnits = r.cases_received || null;
      html += `<td class="num">${recvUnits ? fmt(r.total_lbs) + ' lb &middot; ' + fmtInt(recvUnits) + ' units' : fmt(r.total_lbs) + ' lb'}</td>`;
      html += `<td>${escHtml(r.shipper_name || '\u2014')}</td>`;
      html += `<td>${escHtml(r.bol_reference || '\u2014')}</td>`;
      html += `</tr>`;
      // Detail row
      html += `<tr class="activity-detail${overflowClass(idx)}" id="${rowId}"><td colspan="5">`;
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
    html += '</tbody>' + showMoreFooter(receipts.length, 5) + '</table>';
    container.innerHTML = html;
    bindExpandableRows(container);
    bindLotLinks(container);
    bindShowMore(container);
  }

  // ── Activity: Daily Entries ──
  function dailyEntriesDate() {
    const input = document.getElementById('daily-entries-date');
    if (!input.value) {
      // Default to today, plant time
      input.value = new Date().toLocaleDateString('en-CA', { timeZone: 'America/New_York' });
    }
    return input.value;
  }

  async function refreshDailyEntries() {
    hideError('daily-entries-error');
    const container = document.getElementById('daily-entries-table');
    container.innerHTML = '<div class="loading-indicator">Loading entries...</div>';
    const day = dailyEntriesDate();
    const mode = document.getElementById('daily-entries-mode').value;
    try {
      const data = await fetchAPI('/activity/daily-entries?date=' + encodeURIComponent(day) + '&date_mode=' + encodeURIComponent(mode));
      renderDailyEntries(data, container);
    } catch (e) {
      container.innerHTML = '';
      showError('daily-entries-error', 'Failed to load daily entries: ' + e.message);
    }
  }

  function lateLagText(entry) {
    if (!entry.late_entry) return '';
    const when = entry.days_late === 1 ? 'next day' : entry.days_late + ' days later';
    return 'entered ' + when + ' ' + (entry.created_time || '');
  }

  function renderDailyEntries(data, container) {
    const entries = data.entries || [];
    if (entries.length === 0) {
      container.innerHTML = '<div class="loading-indicator">No entries for ' + escHtml(data.date) + '.</div>';
      return;
    }
    let html = '<table class="activity-table"><thead><tr><th>Entered</th><th>Type</th><th>Product</th><th>SKU</th><th class="num">Qty (lb)</th></tr></thead><tbody>';
    for (const t of entries) {
      const rowClass = t.late_entry ? ' class="late-entry"' : '';
      const lines = (t.lines && t.lines.length > 0) ? t.lines : [{}];
      let provenance = '';
      if (t.created_at_source === 'migration_backfill_039') provenance = ' · backfilled';
      if (t.created_at_source === 'legacy_unverified') provenance = ' · legacy';
      lines.forEach((l, i) => {
        html += `<tr${rowClass}>`;
        if (i === 0) {
          html += `<td rowspan="${lines.length}"><div>${escHtml((t.created_date || '—') + ' ' + (t.created_time || ''))}${escHtml(provenance)}</div>`;
          if (t.late_entry) {
            html += `<div class="late-lag" title="Event date ${escAttr(t.event_date)}">${escHtml(lateLagText(t))}</div>`;
          }
          html += `</td>`;
          html += `<td rowspan="${lines.length}">${escHtml(t.type)}</td>`;
        }
        html += `<td>${escHtml(l.product_name || '—')}</td>`;
        html += `<td>${escHtml(l.sku || '—')}</td>`;
        const qty = (l.quantity_lb === null || l.quantity_lb === undefined) ? null : Number(l.quantity_lb);
        html += `<td class="num">${qty === null ? '—' : (qty > 0 ? '+' : '') + fmt(qty)}</td>`;
        html += `</tr>`;
      });
    }
    html += '</tbody></table>';
    container.innerHTML = html;
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
        html += `<div class="tl-date">Occurred: ${escHtml(t.date)} ${escHtml(t.time)}</div>`;
        html += createdAtMeta(t);
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
  const SALES_API_KEY = 'dashboard-key-2026';
  const SALES_ORDER_OPEN_STATUSES = ['new', 'confirmed', 'in_production', 'ready', 'partial_ship'];
  const SALES_ORDER_STATUS_VALUES = ['new', 'confirmed', 'in_production', 'ready', 'partial_ship', 'shipped', 'invoiced', 'cancelled'];
  const SALES_ORDER_HEADER_EDIT_STATUSES = ['new', 'confirmed'];

  // Orders sub-state
  state.ordersData = [];
  state.ordersLoaded = false;
  state.ordersScrollTop = 0;
  state.currentOrderDetail = null;
  state.orderDetailEditMode = false;

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
      const statusFilter = document.getElementById('orders-status-filter').value;
      const params = new URLSearchParams({ limit: '200' });
      if (statusFilter !== 'all') params.set('status', statusFilter);
      const data = await fetchSalesAPI('/sales/orders?' + params.toString());
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
      state.currentOrderDetail = data;
      state.orderDetailEditMode = false;
      state.orderLinesCache[orderId] = data;
      renderOrderDetail(data, container);
    } catch (e) {
      container.innerHTML = '';
      showError('order-detail-error', 'Failed to load order detail: ' + e.message);
    }
  }

  async function refreshOrderDetail(orderId, successMessage) {
    const container = document.getElementById('order-detail-container');
    const data = await fetchSalesAPI('/sales/orders/' + orderId);
    state.currentOrderDetail = data;
    state.orderDetailEditMode = false;
    state.orderLinesCache[orderId] = data;
    const existing = state.ordersData.find(order => String(order.order_id) === String(orderId));
    if (existing) {
      existing.status = data.status;
      existing.requested_ship_date = data.requested_ship_date;
      existing.customer = data.customer;
      existing.order_number = data.order_number;
    }
    renderOrderDetail(data, container, false, successMessage || '');
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

  function canEditOrderHeader(order) {
    return SALES_ORDER_HEADER_EDIT_STATUSES.includes(order.status);
  }

  function parseApiErrorMessage(error) {
    const raw = error && error.message ? error.message : String(error);
    const match = raw.match(/^HTTP\s+\d+:\s*([\s\S]*)$/);
    if (!match) return raw;
    const body = match[1];
    try {
      const payload = JSON.parse(body);
      if (payload && payload.detail) {
        if (typeof payload.detail === 'string') return payload.detail;
        if (payload.detail.message) return payload.detail.message;
      }
      if (payload && payload.error) return payload.error;
      if (payload && payload.error_detail && payload.error_detail.message) return payload.error_detail.message;
    } catch (_) {
      // Fall through to the raw response body.
    }
    return body;
  }

  function setOrderDetailMessage(container, message, kind) {
    const el = container.querySelector('.order-edit-message');
    if (!el) return;
    el.textContent = message || '';
    el.className = 'order-edit-message';
    if (message) el.classList.add(kind === 'success' ? 'success' : 'error');
  }

  function renderStatusOptions(currentStatus) {
    return SALES_ORDER_STATUS_VALUES.map(status => (
      `<option value="${escAttr(status)}"${status === currentStatus ? ' selected' : ''}>${escHtml(soStatusLabel(status))}</option>`
    )).join('');
  }

  function renderOrderEditActions(order, editMode) {
    if (!canEditOrderHeader(order)) {
      return `<div class="order-edit-locked">Editing opens only while the order is New or Confirmed. Current status: ${escHtml(soStatusLabel(order.status))}.</div>`;
    }
    if (editMode) {
      return '<div class="order-detail-actions">' +
        '<button type="button" class="btn-refresh order-save-header-btn">Save Header</button>' +
        '<button type="button" class="btn-refresh order-save-lines-btn">Save Lines</button>' +
        '<button type="button" class="btn-secondary order-cancel-edit-btn">Done</button>' +
        '</div>';
    }
    return '<div class="order-detail-actions"><button type="button" class="btn-refresh order-edit-toggle-btn">Edit Order</button></div>';
  }

  async function saveOrderHeader(container) {
    const order = state.currentOrderDetail;
    if (!order) return;
    const payload = {};
    const dateValue = container.querySelector('.order-edit-ship-date').value;
    const notesValue = container.querySelector('.order-edit-notes').value;
    if ((dateValue || '') !== (order.requested_ship_date || '')) payload.requested_ship_date = dateValue || '';
    if ((notesValue || '') !== (order.notes || '')) payload.notes = notesValue;

    if (Object.keys(payload).length === 0) {
      setOrderDetailMessage(container, 'No header changes to save.', 'error');
      return;
    }

    const button = container.querySelector('.order-save-header-btn');
    if (button) button.disabled = true;
    setOrderDetailMessage(container, '', 'success');
    hideError('order-detail-error');
    try {
      await fetchSalesAPI('/sales/orders/' + order.order_id, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      await refreshOrderDetail(order.order_id, 'Header saved.');
    } catch (e) {
      setOrderDetailMessage(container, parseApiErrorMessage(e), 'error');
    } finally {
      if (button) button.disabled = false;
    }
  }

  async function saveOrderLines(container) {
    const order = state.currentOrderDetail;
    if (!order) return;
    const changes = [];
    for (const row of container.querySelectorAll('tr.order-line-edit-row')) {
      const lineId = row.dataset.lineId;
      const original = (order.lines || []).find(line => String(line.line_id) === String(lineId));
      if (!original) continue;
      const qtyInput = row.querySelector('.order-line-qty-input');
      const priceInput = row.querySelector('.order-line-price-input');
      if (!qtyInput || !priceInput) continue;
      const params = new URLSearchParams();
      const nextQty = qtyInput.value === '' ? null : Number(qtyInput.value);
      const nextPrice = priceInput.value === '' ? null : Number(priceInput.value);
      if (nextQty != null && !Number.isFinite(nextQty)) {
        setOrderDetailMessage(container, 'Quantity must be a valid number.', 'error');
        return;
      }
      if (nextPrice != null && !Number.isFinite(nextPrice)) {
        setOrderDetailMessage(container, 'Price must be a valid number.', 'error');
        return;
      }
      if (nextQty != null && nextQty !== Number(original.quantity_lb || 0)) params.set('quantity_lb', String(nextQty));
      const originalPrice = original.case_price == null ? null : Number(original.case_price);
      if (nextPrice !== originalPrice) {
        if (nextPrice == null) {
          setOrderDetailMessage(container, 'Price cannot be blank for a line edit.', 'error');
          return;
        }
        params.set('unit_price', String(nextPrice));
      }
      if ([...params.keys()].length > 0) {
        changes.push({ lineId, params });
      }
    }

    if (changes.length === 0) {
      setOrderDetailMessage(container, 'No line changes to save.', 'error');
      return;
    }

    const button = container.querySelector('.order-save-lines-btn');
    if (button) button.disabled = true;
    setOrderDetailMessage(container, '', 'success');
    hideError('order-detail-error');
    let savedCount = 0;
    try {
      for (const change of changes) {
        await fetchSalesAPI('/sales/orders/' + order.order_id + '/lines/' + change.lineId + '/update?' + change.params.toString(), {
          method: 'PATCH'
        });
        savedCount += 1;
      }
      await refreshOrderDetail(order.order_id, changes.length === 1 ? 'Line saved.' : `${changes.length} lines saved.`);
    } catch (e) {
      if (savedCount > 0) {
        await refreshOrderDetail(order.order_id);
        setOrderDetailMessage(document.getElementById('order-detail-container'), parseApiErrorMessage(e), 'error');
      } else {
        setOrderDetailMessage(container, parseApiErrorMessage(e), 'error');
      }
    } finally {
      if (button) button.disabled = false;
    }
  }

  async function saveOrderStatus(container) {
    const order = state.currentOrderDetail;
    if (!order) return;
    const select = container.querySelector('.order-status-select');
    const nextStatus = select ? select.value : order.status;
    if (nextStatus === order.status) return;
    const ok = window.confirm(`Change ${order.order_number} status from ${soStatusLabel(order.status)} to ${soStatusLabel(nextStatus)}?`);
    if (!ok) {
      select.value = order.status;
      return;
    }
    select.disabled = true;
    setOrderDetailMessage(container, '', 'success');
    hideError('order-detail-error');
    try {
      await fetchSalesAPI('/sales/orders/' + order.order_id + '/status', {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: nextStatus })
      });
      await refreshOrderDetail(order.order_id, 'Status updated.');
    } catch (e) {
      select.value = order.status;
      setOrderDetailMessage(container, parseApiErrorMessage(e), 'error');
    } finally {
      select.disabled = false;
    }
  }

  function bindOrderDetailEditControls(container) {
    const editBtn = container.querySelector('.order-edit-toggle-btn');
    if (editBtn) {
      editBtn.addEventListener('click', () => {
        state.orderDetailEditMode = true;
        renderOrderDetail(state.currentOrderDetail, container, true);
      });
    }
    const cancelBtn = container.querySelector('.order-cancel-edit-btn');
    if (cancelBtn) {
      cancelBtn.addEventListener('click', () => {
        state.orderDetailEditMode = false;
        renderOrderDetail(state.currentOrderDetail, container, false);
      });
    }
    const headerBtn = container.querySelector('.order-save-header-btn');
    if (headerBtn) headerBtn.addEventListener('click', () => saveOrderHeader(container));
    const linesBtn = container.querySelector('.order-save-lines-btn');
    if (linesBtn) linesBtn.addEventListener('click', () => saveOrderLines(container));
    const statusSelect = container.querySelector('.order-status-select');
    if (statusSelect) statusSelect.addEventListener('change', () => saveOrderStatus(container));
  }

  function renderOrderDetail(data, container, editMode = state.orderDetailEditMode, successMessage = '') {
    let html = '';
    const editable = canEditOrderHeader(data);
    editMode = Boolean(editMode && editable);

    // Header
    html += '<div class="order-detail-header">';
    html += '<div class="order-detail-top">';
    html += `<span class="order-number">${escHtml(data.order_number)}</span>`;
    if (editMode) {
      html += `<select class="order-status-select" aria-label="Order status">${renderStatusOptions(data.status)}</select>`;
    } else {
      html += `<span class="so-badge status-${data.status}">${soStatusLabel(data.status)}</span>`;
    }
    html += '</div>';
    html += `<div class="order-detail-top"><span class="order-customer">${escHtml(data.customer)}</span></div>`;
    html += '<div class="order-detail-dates">';
    html += `<span><strong>Order Date:</strong> ${formatDateShort(data.order_date)}</span>`;
    if (editMode) {
      html += '<label class="order-edit-field"><strong>Ship By:</strong> ' +
        `<input type="date" class="order-edit-input order-edit-ship-date" value="${escAttr(data.requested_ship_date || '')}">` +
        '</label>';
    } else {
      html += `<span><strong>Ship By:</strong> ${formatDateShort(data.requested_ship_date)}</span>`;
    }
    html += '</div>';
    html += renderOrderEditActions(data, editMode);
    html += `<div class="order-edit-message${successMessage ? ' success' : ''}">${escHtml(successMessage)}</div>`;
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
    let summaryHtml = '<div class="order-kpi-row">';
    summaryHtml += `<div class="order-kpi"><div class="kpi-label">Total Ordered</div><div class="kpi-value">${kpiFmt(totalOrdered, orderedUnits)}</div></div>`;
    summaryHtml += `<div class="order-kpi"><div class="kpi-label">Shipped</div><div class="kpi-value">${kpiFmt(totalShipped, shippedUnits)}</div></div>`;
    summaryHtml += `<div class="order-kpi"><div class="kpi-label">Remaining</div><div class="kpi-value">${kpiFmt(totalRemaining, remainingUnits)}</div></div>`;
    summaryHtml += `<div class="order-kpi order-kpi-pallets"><div class="kpi-label">Pallets</div><div class="kpi-value">${escHtml(orderedPallets.display)}<br><small>Remaining: ${escHtml(remainingPallets.display)}</small></div></div>`;
    summaryHtml += '</div>';

    // Line items
    if (lines.length > 0) {
      html += '<table class="orders-table"><thead><tr>';
      html += '<th>Product</th><th class="num">Ordered</th>';
      if (editMode) html += '<th class="num">Price</th>';
      html += '<th class="num">Shipped</th><th class="num">Remaining</th><th>Status</th>';
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
        const lineEditable = editMode && !['fulfilled', 'cancelled'].includes(l.line_status);
        html += `<tr class="${lineEditable ? 'order-line-edit-row' : ''}" data-line-id="${escAttr(l.line_id)}">`;
        html += `<td><div class="order-product-cell"><span>${escHtml(productName)}</span><button type="button" class="btn-sm order-inventory-toggle" data-line-id="${l.line_id}" aria-expanded="false" aria-controls="order-inventory-${l.line_id}">Inventory</button></div></td>`;
        if (lineEditable) {
          html += '<td class="num order-edit-num-cell">' +
            `<input type="number" class="order-edit-input order-line-qty-input" min="0" step="0.01" value="${escAttr(String(l.quantity_lb == null ? '' : l.quantity_lb))}">` +
            `<small class="pallet-secondary">${escHtml(l.uom || 'lb')} · ${escHtml(orderedLinePallets.display)}</small>` +
            '</td>';
          html += '<td class="num order-edit-num-cell">' +
            `<input type="number" class="order-edit-input order-line-price-input" min="0" step="0.01" value="${escAttr(String(l.case_price == null ? '' : l.case_price))}">` +
            `<small class="pallet-secondary">${escHtml((l.price_basis || 'price').replace('_', ' '))}</small>` +
            '</td>';
        } else {
          html += `<td class="num">${lineFmt(l.quantity_lb, l.unit_count)}<small class="pallet-secondary">${escHtml(orderedLinePallets.display)}</small></td>`;
          if (editMode) {
            html += `<td class="num">${l.case_price == null ? '\u2014' : '$' + Number(l.case_price).toFixed(2)}<small class="pallet-secondary">${escHtml((l.price_basis || '').replace('_', ' '))}</small></td>`;
          }
        }
        html += `<td class="num">${lineFmt(l.quantity_shipped_lb, l.shipped_units)}</td>`;
        html += `<td class="num">${lineFmt(remaining, l.remaining_units)}<small class="pallet-secondary">${escHtml(remainingLinePallets.display)}</small></td>`;
        html += `<td><span class="so-badge ${lineStatusClass}">${escHtml(l.line_status || 'pending')}</span></td>`;
        html += '</tr>';
        html += `<tr id="order-inventory-${l.line_id}" class="order-inventory-row hidden"><td colspan="${editMode ? 6 : 5}"></td></tr>`;
      }
      html += '</tbody></table>';
    }

    html += summaryHtml;

    // Notes
    if (editMode) {
      html += '<div class="order-notes-card order-notes-edit">';
      html += '<h4>Notes</h4>';
      html += `<textarea class="order-edit-input order-edit-notes" rows="4">${escHtml(data.notes || '')}</textarea>`;
      html += '</div>';
    } else if (data.notes && data.notes.trim()) {
      html += '<div class="order-notes-card">';
      html += '<h4>Notes</h4>';
      html += `<p>${escHtml(data.notes)}</p>`;
      html += '</div>';
    }

    container.innerHTML = html;
    bindOrderInventoryToggles(container, lines);
    bindOrderDetailEditControls(container);
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
      refreshOrders();
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

  // ── Expected Receipts (FR-2) ──
  // Minimal view over GET /expected-receipts. `remaining` and `is_overdue`
  // come from the API (ledger SUM, floored at 0) — nothing is computed here.

  state.erData = [];
  state.erLoaded = false;
  state.erEditing = null;        // record being edited, or null for create
  state.erSuppliers = [];
  state.erProductTimer = null;

  async function refreshExpectedReceipts() {
    hideError('er-error');
    const container = document.getElementById('er-table-container');
    if (!state.erLoaded) container.innerHTML = '<div class="loading-indicator">Loading expected receipts...</div>';
    try {
      const status = document.getElementById('er-status-filter').value;
      const params = new URLSearchParams({ status, limit: '500' });
      const data = await fetchSalesAPI('/expected-receipts?' + params.toString());
      state.erData = data.expected_receipts || [];
      state.erLoaded = true;
      renderExpectedReceipts();
    } catch (e) {
      container.innerHTML = '';
      showError('er-error', 'Failed to load expected receipts: ' + e.message);
    }
  }

  function getFilteredExpectedReceipts() {
    const q = document.getElementById('er-text-filter').value.trim().toLowerCase();
    const overdueOnly = document.getElementById('er-overdue-only').checked;
    return state.erData.filter(r => {
      if (overdueOnly && !r.is_overdue) return false;
      if (!q) return true;
      const hay = [r.product_name, r.odoo_code, r.supplier_name, r.reference_number, r.notes]
        .filter(Boolean).join(' ').toLowerCase();
      return hay.includes(q);
    });
  }

  function erStatusBadge(r) {
    if (r.status === 'open') {
      return r.is_overdue
        ? `<span class="so-badge er-badge er-badge-overdue" title="${escAttr(r.days_overdue + ' day(s) past expected date')}">Overdue ${r.days_overdue}d</span>`
        : '<span class="so-badge er-badge er-badge-open">Open</span>';
    }
    if (r.status === 'closed') return '<span class="so-badge er-badge er-badge-closed">Closed</span>';
    return '<span class="so-badge er-badge er-badge-cancelled">Cancelled</span>';
  }

  function renderExpectedReceipts() {
    const container = document.getElementById('er-table-container');
    const rows = getFilteredExpectedReceipts();
    const openCount = state.erData.filter(r => r.status === 'open').length;
    const overdueCount = state.erData.filter(r => r.is_overdue).length;
    document.getElementById('er-summary').textContent =
      state.erLoaded ? `${rows.length} shown · ${openCount} open · ${overdueCount} overdue` : '';

    if (rows.length === 0) {
      container.innerHTML = `<div class="orders-empty">
        <div class="orders-empty-icon">&#128666;</div>
        No expected receipts match your filters.
      </div>`;
      return;
    }

    let html = '<table class="orders-table er-table"><thead><tr>';
    html += '<th>Product</th><th>Supplier</th><th class="num">Expected (lb)</th><th class="num">Received (lb)</th><th class="num">Remaining (lb)</th><th>Expected Date</th><th>Reference</th><th>Status</th><th class="er-actions-col"></th>';
    html += '</tr></thead><tbody>';
    for (const r of rows) {
      const cls = ['er-row', r.is_overdue ? 'er-overdue' : '', r.status !== 'open' ? 'er-inactive' : ''].join(' ');
      html += `<tr class="${cls}" data-er-id="${r.id}">`;
      html += `<td><div class="er-product">${escHtml(r.product_name)}</div>${r.odoo_code ? `<div class="er-sku">SKU ${escHtml(r.odoo_code)}</div>` : ''}${r.notes ? `<div class="er-notes" title="${escAttr(r.notes)}">${escHtml(r.notes)}</div>` : ''}</td>`;
      html += `<td>${escHtml(r.supplier_name)}</td>`;
      html += `<td class="num">${fmtWt(r.expected_qty)}</td>`;
      html += `<td class="num">${fmtWt(r.received_qty)}${r.over_receipt_qty > 0 ? ` <span class="er-over" title="Over-receipt">(+${fmtWt(r.over_receipt_qty)})</span>` : ''}</td>`;
      html += `<td class="num er-remaining">${fmtWt(r.remaining)}</td>`;
      html += `<td class="${r.is_overdue ? 'date-overdue' : ''}">${formatShipByDate(r.expected_date)}</td>`;
      html += `<td>${escHtml(r.reference_number || '—')}</td>`;
      html += `<td>${erStatusBadge(r)}</td>`;
      if (r.status === 'open') {
        html += `<td class="er-actions">
          <button type="button" class="btn-sm er-edit-btn" data-er-id="${r.id}" title="Edit qty / date / reference / notes">Edit</button>
          <button type="button" class="btn-sm er-close-btn" data-er-id="${r.id}" title="Mark as closed (nothing more expected)">Close</button>
          <button type="button" class="btn-sm er-cancel-btn" data-er-id="${r.id}" title="Cancel this expected receipt">Cancel</button>
        </td>`;
      } else {
        html += '<td class="er-actions"></td>';
      }
      html += '</tr>';
    }
    html += '</tbody></table>';
    container.innerHTML = html;

    container.querySelectorAll('.er-edit-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const rec = state.erData.find(r => r.id === Number(btn.dataset.erId));
        if (rec) openErModal(rec);
      });
    });
    container.querySelectorAll('.er-close-btn').forEach(btn => {
      btn.addEventListener('click', () => erSetStatus(Number(btn.dataset.erId), 'closed', btn));
    });
    container.querySelectorAll('.er-cancel-btn').forEach(btn => {
      btn.addEventListener('click', () => erSetStatus(Number(btn.dataset.erId), 'cancelled', btn));
    });
  }

  async function erSetStatus(id, status, btn) {
    const rec = state.erData.find(r => r.id === id);
    const label = rec ? `${fmtWt(rec.expected_qty)} lb ${rec.product_name} from ${rec.supplier_name}` : `#${id}`;
    const verb = status === 'closed' ? 'Close' : 'Cancel';
    // Inline two-step confirm (no window.confirm — keeps the page automation-safe).
    if (btn && btn.dataset.armed !== '1') {
      btn.dataset.armed = '1';
      btn.dataset.originalText = btn.textContent;
      btn.textContent = `${verb}? Confirm`;
      btn.classList.add('er-armed');
      setTimeout(() => {
        if (btn.isConnected) {
          btn.dataset.armed = '';
          btn.textContent = btn.dataset.originalText;
          btn.classList.remove('er-armed');
        }
      }, 4000);
      return;
    }
    hideError('er-error');
    try {
      await fetchSalesAPI(`/expected-receipts/${id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status }),
      });
      await refreshExpectedReceipts();
    } catch (e) {
      showError('er-error', `Failed to ${verb.toLowerCase()} ${label}: ${e.message}`);
    }
  }

  async function loadErSuppliers() {
    try {
      const data = await fetchSalesAPI('/suppliers');
      state.erSuppliers = data.suppliers || [];
    } catch (e) {
      state.erSuppliers = [];
    }
    const sel = document.getElementById('er-supplier');
    sel.innerHTML = '<option value="">— select supplier —</option>' +
      state.erSuppliers.map(s => `<option value="${escAttr(s.name)}">${escHtml(s.name)}</option>`).join('');
  }

  function setErProduct(id, name, sku) {
    document.getElementById('er-product-id').value = id || '';
    const chosen = document.getElementById('er-product-chosen');
    chosen.innerHTML = id ? `Selected: <strong>${escHtml(name)}</strong>${sku ? ` <span class="er-sku">SKU ${escHtml(sku)}</span>` : ''}` : '';
    document.getElementById('er-product-results').classList.add('hidden');
  }

  async function searchErProducts(q) {
    const box = document.getElementById('er-product-results');
    if (!q) { box.classList.add('hidden'); return; }
    try {
      const data = await fetchSalesAPI('/products/search?q=' + encodeURIComponent(q));
      const products = (data.products || []).slice(0, 12);
      if (!products.length) {
        box.innerHTML = '<div class="er-product-option er-product-none">No products found</div>';
      } else {
        box.innerHTML = products.map(p =>
          `<div class="er-product-option" data-id="${p.id}" data-name="${escAttr(p.name)}" data-sku="${escAttr(p.odoo_code || '')}">${escHtml(p.name)}${p.odoo_code ? ` <span class="er-sku">${escHtml(p.odoo_code)}</span>` : ''}</div>`
        ).join('');
        box.querySelectorAll('.er-product-option[data-id]').forEach(opt => {
          opt.addEventListener('click', () => {
            setErProduct(opt.dataset.id, opt.dataset.name, opt.dataset.sku);
            document.getElementById('er-product-search').value = opt.dataset.name;
          });
        });
      }
      box.classList.remove('hidden');
    } catch (e) {
      box.innerHTML = `<div class="er-product-option er-product-none">Search failed: ${escHtml(e.message)}</div>`;
      box.classList.remove('hidden');
    }
  }

  async function openErModal(record) {
    state.erEditing = record || null;
    hideError('er-modal-error');
    document.getElementById('er-modal-title').textContent = record ? `Edit Expected Receipt #${record.id}` : 'New Expected Receipt';
    await loadErSuppliers();
    const productGroup = document.getElementById('er-product-group');
    const supplierSel = document.getElementById('er-supplier');
    if (record) {
      // Product & supplier are fixed once created (auto-match key); edit the rest.
      productGroup.classList.add('hidden');
      supplierSel.value = record.supplier_name;
      supplierSel.disabled = true;
      document.getElementById('er-qty').value = record.expected_qty;
      document.getElementById('er-date').value = record.expected_date || '';
      document.getElementById('er-reference').value = record.reference_number || '';
      document.getElementById('er-notes').value = record.notes || '';
    } else {
      productGroup.classList.remove('hidden');
      document.getElementById('er-product-search').value = '';
      setErProduct(null);
      supplierSel.disabled = false;
      supplierSel.value = '';
      document.getElementById('er-qty').value = '';
      document.getElementById('er-date').value = '';
      document.getElementById('er-reference').value = '';
      document.getElementById('er-notes').value = '';
    }
    document.getElementById('er-modal-overlay').classList.remove('hidden');
    (record ? document.getElementById('er-qty') : document.getElementById('er-product-search')).focus();
  }

  function closeErModal() {
    document.getElementById('er-modal-overlay').classList.add('hidden');
    state.erEditing = null;
  }

  async function saveEr() {
    hideError('er-modal-error');
    const qty = parseFloat(document.getElementById('er-qty').value);
    if (!(qty > 0)) { showError('er-modal-error', 'Expected qty must be a positive number of pounds.'); return; }
    const expectedDate = document.getElementById('er-date').value || null;
    const reference = document.getElementById('er-reference').value.trim() || null;
    const notes = document.getElementById('er-notes').value.trim() || null;
    const btn = document.getElementById('er-save-btn');
    btn.disabled = true;
    try {
      if (state.erEditing) {
        await fetchSalesAPI(`/expected-receipts/${state.erEditing.id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ expected_qty: qty, expected_date: expectedDate, reference_number: reference, notes }),
        });
      } else {
        const productId = document.getElementById('er-product-id').value;
        const supplierName = document.getElementById('er-supplier').value;
        if (!productId) { showError('er-modal-error', 'Pick a product from the search results.'); return; }
        if (!supplierName) { showError('er-modal-error', 'Pick a supplier.'); return; }
        await fetchSalesAPI('/expected-receipts', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            product_id: Number(productId), supplier_name: supplierName, expected_qty: qty,
            expected_date: expectedDate, reference_number: reference, notes, created_by: 'dashboard',
          }),
        });
      }
      closeErModal();
      await refreshExpectedReceipts();
    } catch (e) {
      // Surface the API's structured message (e.g. supplier candidates) when present.
      let msg = e.message;
      const m = /HTTP \d+: (.*)$/s.exec(e.message);
      if (m) {
        try {
          const body = JSON.parse(m[1]);
          const d = body.detail || {};
          msg = (d.message || (body.error_detail && body.error_detail.message) || e.message);
          if (Array.isArray(d.suggestions) && d.suggestions.length) msg += ' Candidates: ' + d.suggestions.join(', ');
        } catch (_) { /* leave raw */ }
      }
      showError('er-modal-error', msg);
    } finally {
      btn.disabled = false;
    }
  }

  function initExpectedReceipts() {
    document.getElementById('er-status-filter').addEventListener('change', refreshExpectedReceipts);
    document.getElementById('er-refresh-btn').addEventListener('click', refreshExpectedReceipts);
    document.getElementById('er-overdue-only').addEventListener('change', () => { if (state.erLoaded) renderExpectedReceipts(); });
    let t;
    document.getElementById('er-text-filter').addEventListener('input', () => {
      clearTimeout(t);
      t = setTimeout(() => { if (state.erLoaded) renderExpectedReceipts(); }, 200);
    });
    document.getElementById('er-new-btn').addEventListener('click', () => openErModal(null));
    document.getElementById('er-modal-close').addEventListener('click', closeErModal);
    document.getElementById('er-cancel-btn').addEventListener('click', closeErModal);
    document.getElementById('er-modal-overlay').addEventListener('click', (e) => {
      if (e.target === e.currentTarget) closeErModal();
    });
    document.getElementById('er-save-btn').addEventListener('click', saveEr);
    document.getElementById('er-product-search').addEventListener('input', (e) => {
      setErProduct(null);
      clearTimeout(state.erProductTimer);
      const q = e.target.value.trim();
      state.erProductTimer = setTimeout(() => searchErProducts(q), 250);
    });
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
    ops.push(refreshDailyEntries());
    ops.push(refreshNotes());
    ops.push(refreshOrders());
    ops.push(refreshExpectedReceipts());
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
    initExpectedReceipts();

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

    // Daily Entries controls
    document.getElementById('daily-entries-date').addEventListener('change', refreshDailyEntries);
    document.getElementById('daily-entries-mode').addEventListener('change', refreshDailyEntries);

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
