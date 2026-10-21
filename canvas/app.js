// FL Canvas: zoomable, read-only view of Factory Ledger.
// World coordinates follow the ConceptA prototype (2600 x 1600 canvas).
(function () {
  const CFG = window.FL_CANVAS_CONFIG;
  const API = window.FLApi;
  const DASH = CFG.dashboardBase.replace(/\/$/, '');
  const $ = (s) => document.querySelector(s);

  // ---------- formatting ----------
  const ESC = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const esc = (v) => String(v == null ? '' : v).replace(/[&<>"']/g, (c) => ESC[c]);
  const num = (v) => { const n = Number(v); return Number.isFinite(n) ? n : 0; };
  const fmt = (v) => { const n = num(v); return n.toLocaleString('en-US', { maximumFractionDigits: Math.abs(n) < 10 && n % 1 ? 1 : 0 }); };
  const signed = (v) => (num(v) > 0 ? '+' : num(v) < 0 ? '−' : '') + fmt(Math.abs(num(v)));
  const ymdFmt = new Intl.DateTimeFormat('en-CA', { timeZone: CFG.timeZone, year: 'numeric', month: '2-digit', day: '2-digit' });
  const TODAY = ymdFmt.format(new Date());
  const addDays = (ymd, d) => { const t = new Date(ymd + 'T12:00:00Z'); t.setUTCDate(t.getUTCDate() + d); return t.toISOString().slice(0, 10); };
  const daysBetween = (a, b) => Math.round((new Date(b + 'T12:00:00Z') - new Date(a + 'T12:00:00Z')) / 86400000);
  const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const short = (ymd) => (ymd ? MONTHS[+ymd.slice(5, 7) - 1] + ' ' + +ymd.slice(8, 10) : '—');
  const longToday = () => { const t = new Date(TODAY + 'T12:00:00Z'); return DAYS[t.getUTCDay()] + ' ' + short(TODAY); };
  const nc = (what) => '<span class="nc" title="' + esc(what || 'No read endpoint for this yet') + '">not connected yet</span>';
  const TX_LABEL = { receive: 'Received', make: 'Made', pack: 'Packed', ship: 'Shipped', adjust: 'Adjusted' };
  const SOURCE_LABEL = { production_output: 'Made', pack_output: 'Packed', received: 'Received', receive: 'Received', found_inventory: 'Found', found: 'Found', adjustment: 'Adjusted' };
  const sourceLabel = (s) => SOURCE_LABEL[s] || (s ? String(s).replace(/_/g, ' ') : '—');

  // ---------- record timestamps ----------
  // One format everywhere: "Oct 20, 2026 · 10:30 AM", America/New_York. Only real API fields
  // are shown; a missing field reads "Not recorded". Naive timestamps from the API
  // (transactions.timestamp, last_received_at) are UTC, as the backend treats them.
  const NOT_RECORDED = 'Not recorded';
  const tsDate = new Intl.DateTimeFormat('en-US', { timeZone: CFG.timeZone, month: 'short', day: 'numeric', year: 'numeric' });
  const tsTime = new Intl.DateTimeFormat('en-US', { timeZone: CFG.timeZone, hour: 'numeric', minute: '2-digit' });
  function stamp(iso) {
    if (!iso) return null;
    const s = String(iso);
    const d = new Date(/[zZ]|[+-]\d\d:?\d\d$/.test(s) ? s : s + 'Z');
    return Number.isNaN(d.getTime()) ? null : tsDate.format(d) + ' · ' + tsTime.format(d);
  }
  // The API also returns some times pre-formatted in ET: created_date "2026-10-15" + created_time "11:54 AM ET".
  function stampET(ymd, hm) {
    if (!ymd) return null;
    const m = /^(\d{4})-(\d\d)-(\d\d)/.exec(ymd);
    if (!m) return null;
    const day = MONTHS[+m[2] - 1] + ' ' + +m[3] + ', ' + m[1];
    const t = hm ? String(hm).replace(/\s*ET$/, '').replace(/^0(\d)/, '$1') : '';
    return t ? day + ' · ' + t : day;
  }
  const tsPart = (label, v) => '<span>' + label + ' ' + (v ? esc(v) : '<span class="nr">' + NOT_RECORDED + '</span>') + '</span>';
  // Compact secondary line under a summary row: Created … · Updated … (+ optional extras).
  const tsLine = (created, updated, extra) => '<div class="ts">' + tsPart('Created', created) + '<span class="sep">·</span>' + tsPart('Updated', updated) + (extra ? '<span class="sep">·</span><span>' + extra + '</span>' : '') + '</div>';
  // Transactions are never edited in place; a correction is the closest thing to an update.
  const txCreated = (t) => stamp(t.created_at);
  const txUpdated = (t) => stamp(t.latest_correction_created_at);
  // "Record history" block for the side panel: [[label, value-or-null], ...].
  const history = (rows, note) => '<div class="sect"><h4>Record history</h4><div class="txlist hist">' + rows.map(([k, v]) => '<div><span>' + esc(k) + '</span><span>' + (v ? esc(v) : '<span class="nr">' + NOT_RECORDED + '</span>') + '</span></div>').join('') + '</div>' + (note ? '<div class="dim" style="font-size:12px">' + note + '</div>' : '') + '</div>';

  function dashUrl(page, params) {
    const u = new URL(DASH + '/' + page);
    for (const [k, v] of Object.entries(params || {})) if (v != null && v !== '') u.searchParams.set(k, v);
    return u.toString();
  }
  const openTab = (url) => window.open(url, '_blank', 'noopener');

  // ---------- world layout (prototype coordinates) ----------
  // x and w are fixed; y and h are set by layoutWorld() from each card's measured content.
  // `pair` cards share a height so matching cards keep matching proportions.
  const AREAS = {
    production: { x: 520, y: 120, w: 560, h: 340, name: 'Production', icon: 'factory', row: 0, pair: 'top' },
    packing: { x: 1520, y: 120, w: 560, h: 340, name: 'Packing', icon: 'package', row: 0, pair: 'top' },
    receiving: { x: 160, y: 640, w: 480, h: 380, name: 'Receiving', icon: 'truck', row: 1, pair: 'side' },
    inventory: { x: 900, y: 560, w: 800, h: 540, name: 'Inventory', hub: true, icon: 'warehouse', row: 1 },
    shipping: { x: 1960, y: 640, w: 480, h: 380, name: 'Orders & shipping', icon: 'clipboard-list', row: 1, pair: 'side' },
    quality: { x: 520, y: 1200, w: 560, h: 300, name: 'Quality & traceability', icon: 'shield-check', row: 2, pair: 'bottom' },
    reports: { x: 1520, y: 1200, w: 560, h: 300, name: 'Reports', icon: 'chart-column', row: 2, pair: 'bottom' }
  };
  const ROW_TOP = 120, ROW_GAP = 110;
  const ICONS = window.FL_ICONS || {};
  const icon = (name) => '<span class="ico" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round">' + (ICONS[name] || '') + '</svg></span>';
  const kicker = (id, label) => '<div class="kick-row">' + icon(AREAS[id].icon) + '<div class="kicker">' + (label || esc(AREAS[id].name)) + '</div></div>';
  const TILES = {
    ingredient: { dx: 24, dy: 64, name: 'Ingredients' },
    batch: { dx: 408, dy: 64, name: 'Bulk granola', accent: true },
    packaging: { dx: 24, dy: 252, name: 'Packaging' },
    finished: { dx: 408, dy: 252, name: 'Finished goods' }
  };
  const TILE_W = 368, TILE_H = 172;
  const GRID = { w: 1472, h: TILE_H / 0.25, k: 0.25, left: 40, top: 104, right: 40, bottom: 28, gap: 20 };
  // Product cards and their working screen share one aspect ratio (the screen is scaled to the card).
  const CARD = { w: 440, h: 272 };
  const WS = { w: 1100, h: 680 };
  const BOX = { x: 160, y: 120, w: 2280, h: 1380 };
  // A node "opens" (shows its next level) once it covers this share of the viewport.
  const OPEN_AT = { area: 0.45, tile: 0.6, product: 0.5 };

  const nodes = new Map();
  const data = {};
  const lotCache = new Map();
  let products = [];
  let cam = { cx: BOX.x + BOX.w / 2, cy: BOX.y + BOX.h / 2, s: 0.5 };
  let selection = null; // {kind:'lot'|'order'|'er', label, ...}
  let focus = { area: null, tile: null, product: null };

  const world = $('#world');
  const vp = $('#viewport');
  const vpSize = () => ({ w: vp.clientWidth, h: vp.clientHeight });

  function banner(msg, isError) {
    const b = $('#banner');
    if (!msg) { b.hidden = true; return; }
    b.hidden = false;
    b.className = isError ? 'error' : '';
    b.textContent = msg;
  }

  // ---------- data ----------
  async function loadAll() {
    const since7 = addDays(TODAY, -6);
    const jobs = {
      inv: API.get('/inventory/current', { limit: 500 }),
      supIng: API.get('/supplies/inventory', { category: 'ingredient' }),
      supPkg: API.get('/supplies/inventory', { category: 'packaging' }),
      er: API.get('/expected-receipts', { status: 'open' }),
      orders: API.get('/sales/orders', { state: 'open', limit: 200 }),
      counts: API.get('/sales/orders/counts'),
      sales: API.get('/sales/dashboard'),
      tx: API.get('/transactions/history', { since: since7, limit: 1000 }),
      runs: API.get('/production/runs', { from: TODAY, to: TODAY }),
      exceptions: API.get('/exceptions', { status: 'open' })
    };
    const keys = Object.keys(jobs);
    const settled = await Promise.allSettled(keys.map((k) => jobs[k]));
    settled.forEach((r, i) => { data[keys[i]] = r.status === 'fulfilled' ? r.value : { error: r.reason && r.reason.message, status: r.reason && r.reason.status }; });
    data.loadedAt = new Date();
  }

  const failed = (d) => !d || d.error;
  const postedTx = () => (failed(data.tx) ? [] : data.tx.transactions.filter((t) => t.status === 'posted'));

  function buildProducts() {
    const units = {};
    for (const src of [data.supIng, data.supPkg]) if (!failed(src)) for (const it of src.items) units[it.product_id] = it.unit;
    const byId = new Map();
    for (const r of failed(data.inv) ? [] : data.inv.inventory) {
      let p = byId.get(r.product_id);
      if (!p) {
        p = { id: r.product_id, name: r.product_name, type: r.product_type, sku: r.odoo_code, caseLb: num(r.case_size_lb) || null, batchLb: num(r.default_batch_lb) || null, unit: units[r.product_id] || 'lb', lots: [], qty: 0, units: 0 };
        byId.set(r.product_id, p);
      }
      p.lots.push({ id: r.lot_id, code: r.lot_code, qty: num(r.quantity_on_hand), units: num(r.unit_count) });
      p.qty += num(r.quantity_on_hand);
      p.units += num(r.unit_count) || (p.caseLb ? num(r.quantity_on_hand) / p.caseLb : 0);
    }
    products = [...byId.values()].sort((a, b) => b.qty - a.qty);
  }

  function qtyText(p, which) {
    const q = which === 'units' ? p.units : p.qty;
    if (p.type === 'finished' && p.caseLb) return fmt(p.units) + ' cs';
    if (p.type === 'packaging' || (p.unit && p.unit !== 'lb')) return fmt(q) + ' ' + (p.unit && p.unit !== 'unit' ? p.unit : 'ea');
    return fmt(q) + ' lb';
  }
  const ofType = (t) => products.filter((p) => p.type === t);
  const isEK = (name) => /EXTRA-KOSHER/i.test(name || '');

  // ---------- area content ----------
  function errLine(d) { return '<div class="sub t-red">Couldn’t load (' + esc(d && d.error) + ')</div>'; }

  function txOf(type) { return postedTx().filter((t) => t.type === type).sort((a, b) => (b.business_date + b.created_at).localeCompare(a.business_date + a.created_at)); }
  const outLine = (t) => (t.lines || []).find((l) => num(l.quantity_lb) > 0) || (t.lines || [])[0] || {};
  const inLine = (t) => (t.lines || []).find((l) => num(l.quantity_lb) < 0) || {};

  function productionArea() {
    if (failed(data.tx)) return { far: '<div class="far-body">' + kicker('production') + errLine(data.tx) + '</div>', near: '' };
    const makes = txOf('make');
    const today = makes.filter((t) => t.business_date === TODAY);
    const lbToday = today.reduce((s, t) => s + num(outLine(t).quantity_lb), 0);
    const last = makes[0];
    const runs = failed(data.runs) ? null : data.runs.runs;
    const runSignal = runs == null ? '<div class="signal"><span class="dot grey"></span><span class="muted">Planned runs: couldn’t load</span></div>'
      : runs.length ? '<div class="signal t-green"><span class="dot green"></span>' + runs.length + ' planned run' + (runs.length > 1 ? 's' : '') + ' today</div>'
        : '<div class="signal muted"><span class="dot grey"></span>No planned runs scheduled today</div>';
    const far = '<div class="far-body">' + kicker('production') + '<div class="big">' + today.length + ' make' + (today.length === 1 ? '' : 's') + '</div>'
      + '<div class="sub">' + (today.length ? 'today · ' + fmt(lbToday) + ' lb made' : 'today' + (last ? ' · last make ' + short(last.business_date) : ' · none in 7 days')) + '</div>' + runSignal + '</div>';
    const cols = 'grid-template-columns: 0.7fr 2fr 1.3fr 0.9fr 1fr';
    const rows = makes.slice(0, 5).map((t) => { const o = outLine(t); return '<button class="row" style="' + cols + '" data-lot="' + esc(o.lot_code) + '" data-pid="' + esc(o.product_id) + '" data-pname="' + esc(o.product_name) + '" data-tx="' + esc(t.id) + '"><div>' + short(t.business_date) + '</div><div style="font-weight:600">' + esc(o.product_name) + '</div><div class="mono" style="font-size:16px">' + esc(o.lot_code) + '</div><div>' + fmt(o.quantity_lb) + ' lb</div><div class="muted">' + esc(t.operator_id || '') + '</div>' + tsLine(txCreated(t), txUpdated(t)) + '</button>'; }).join('');
    const runRows = (runs || []).slice(0, 4).map((r) => '<div class="row" style="grid-template-columns: 1fr 2fr 1fr 1fr"><div>' + esc(r.run_type) + '</div><div style="font-weight:600">' + esc(r.product_name) + '</div><div>' + esc(r.planned_qty_lb != null ? fmt(r.planned_qty_lb) + ' lb' : (r.planned_batches != null ? fmt(r.planned_batches) + ' batches' : '')) + '</div><div><span class="chip grey">' + esc(r.status) + '</span></div></div>').join('');
    const near = '<div class="near-body"><div class="near-head"><div class="near-title">Production</div><div class="near-meta">Today · ' + longToday() + '</div></div>'
      + '<div class="cols" style="' + cols + '"><div>Date</div><div>Made</div><div>Lot</div><div>Amount</div><div>By</div></div>'
      + '<div class="rows">' + (rows || '<div class="row muted" style="grid-template-columns:1fr">No makes recorded in the last 7 days</div>') + '</div>'
      + '<div class="foot">' + (makes.length > 5 ? '+' + (makes.length - 5) + ' more makes in 7 days · ' : '') + (runs == null ? 'Planned runs couldn’t load' : runs.length ? 'Planned runs today' : 'Planner: no runs scheduled for today (GET /production/runs)') + '</div>'
      + (runRows ? '<div class="rows">' + runRows + '</div>' : '') + '</div>';
    return { far, near };
  }

  function packingArea() {
    if (failed(data.tx)) return { far: '<div class="far-body">' + kicker('packing') + errLine(data.tx) + '</div>', near: '' };
    const packs = txOf('pack');
    const today = packs.filter((t) => t.business_date === TODAY);
    const cases = (list) => list.reduce((s, t) => s + num(outLine(t).unit_count), 0);
    const last = packs[0];
    const far = '<div class="far-body">' + kicker('packing') + '<div class="big">' + today.length + ' pack' + (today.length === 1 ? '' : 's') + '</div>'
      + '<div class="sub">' + (today.length ? 'today · ' + fmt(cases(today)) + ' cases packed' : 'today' + (last ? ' · last pack ' + short(last.business_date) : ' · none in 7 days')) + '</div>'
      + '<div class="signal muted"><span class="dot grey"></span>' + fmt(cases(packs)) + ' cases packed in the last 7 days</div></div>';
    const cols = 'grid-template-columns: 0.7fr 2fr 1.6fr 0.7fr 1fr';
    const rows = packs.slice(0, 5).map((t) => { const o = outLine(t), i = inLine(t); return '<button class="row" style="' + cols + '" data-lot="' + esc(o.lot_code) + '" data-pid="' + esc(o.product_id) + '" data-pname="' + esc(o.product_name) + '" data-tx="' + esc(t.id) + '"><div>' + short(t.business_date) + '</div><div style="font-weight:600">' + esc(o.product_name) + '</div><div>' + esc(i.product_name || 'direct') + '</div><div>' + fmt(o.unit_count) + '</div><div class="muted">' + esc(t.operator_id || '') + '</div>' + tsLine(txCreated(t), txUpdated(t)) + '</button>'; }).join('');
    const near = '<div class="near-body"><div class="near-head"><div class="near-title">Packing</div><div class="near-meta">Last 7 days</div></div>'
      + '<div class="cols" style="' + cols + '"><div>Date</div><div>Finished SKU</div><div>From bulk</div><div>Cases</div><div>By</div></div>'
      + '<div class="rows">' + (rows || '<div class="row muted" style="grid-template-columns:1fr">No packs recorded in the last 7 days</div>') + '</div>'
      + '<div class="foot">' + (packs.length > 5 ? '+' + (packs.length - 5) + ' more packs · ' : '') + 'Planned packing ' + nc('Packing plan has no read endpoint the dashboard key can use') + '</div></div>';
    return { far, near };
  }

  function receivingArea() {
    const d = data.er;
    if (failed(d)) return { far: '<div class="far-body">' + kicker('receiving') + errLine(d) + '</div>', near: '' };
    const list = d.expected_receipts.slice().sort((a, b) => (b.is_overdue - a.is_overdue) || String(a.expected_date || '9').localeCompare(String(b.expected_date || '9')));
    const signal = d.overdue_count ? '<div class="signal t-red"><span class="dot red"></span>' + d.overdue_count + ' expected receipt' + (d.overdue_count > 1 ? 's' : '') + ' overdue</div>' : '<div class="signal t-green"><span class="dot green"></span>None overdue</div>';
    const far = '<div class="far-body">' + kicker('receiving') + '<div class="big">' + d.open_count + ' open</div><div class="sub">expected receipts</div>' + signal + '</div>';
    const cols = 'grid-template-columns: 1.6fr 1fr 1.1fr';
    const rows = list.slice(0, 5).map((e) => {
      // Expected date is a due date, kept apart from the record's created/updated times.
      const chip = e.is_overdue ? '<span class="chip red">' + e.days_overdue + ' day' + (e.days_overdue === 1 ? '' : 's') + ' late</span>'
        : e.expected_date === TODAY ? '<span class="chip blue">Due today</span>'
          : e.expected_date ? '<span class="chip blue">Due ' + short(e.expected_date) + '</span>' : '<span class="chip grey">No date</span>';
      return '<button class="row" style="' + cols + '" data-er="' + esc(e.id) + '"><div style="font-weight:600">' + esc(e.product_name) + '</div><div>' + fmt(e.remaining) + ' lb</div><div>' + chip + '</div>' + tsLine(stamp(e.created_at), stamp(e.updated_at)) + '</button>';
    }).join('');
    const near = '<div class="near-body"><div class="near-head"><div class="near-title">Expected receipts</div><div class="near-meta">' + d.open_count + ' open</div></div>'
      + '<div class="cols" style="' + cols + '"><div>Product</div><div>Remaining</div><div>Expected</div></div>'
      + '<div class="rows">' + (rows || '<div class="row muted" style="grid-template-columns:1fr">No open expected receipts</div>') + '</div>'
      + '<div class="foot">' + (list.length > 5 ? '+' + (list.length - 5) + ' more · ' : '') + 'remaining qty shown</div></div>';
    return { far, near };
  }

  const orderIndex = () => { const m = new Map(); if (!failed(data.orders)) for (const o of data.orders.orders) m.set(o.order_number, o); return m; };
  const shortOrders = () => (failed(data.orders) ? [] : data.orders.orders.filter((o) => num(o.shortage_lb) > 0));

  function shippingArea() {
    const sd = data.sales, c = data.counts;
    if (failed(sd)) return { far: '<div class="far-body">' + kicker('shipping', 'Orders &amp; shipping') + errLine(sd) + '</div>', near: '' };
    const idx = orderIndex();
    const short_ = shortOrders();
    // Order alerts live here (moved off the Inventory card): overdue, and orders short on stock.
    const sigs = [];
    if (sd.overdue_count) sigs.push('<div class="signal t-red"><span class="dot red"></span>' + sd.overdue_count + ' order' + (sd.overdue_count > 1 ? 's' : '') + ' overdue</div>');
    if (short_.length) sigs.push('<div class="signal t-amber"><span class="dot amber"></span>' + short_.length + ' short on stock</div>');
    if (!sigs.length) sigs.push('<div class="signal t-green"><span class="dot green"></span>Nothing overdue or short</div>');
    const far = '<div class="far-body">' + kicker('shipping', 'Orders &amp; shipping') + '<div class="big">' + sd.due_this_week_count + ' orders</div><div class="sub">due this week' + (!failed(c) ? ' · ' + c.open + ' open' : '') + '</div><div class="signals">' + sigs.join('') + '</div></div>';
    const list = [...(sd.overdue_orders || []).map((o) => Object.assign({ overdue: true }, o)), ...(sd.due_this_week || [])];
    const cols = 'grid-template-columns: 1.5fr 1.4fr 0.8fr 1.2fr';
    const rows = list.slice(0, 5).map((o) => {
      const full = idx.get(o.order_number) || {};
      const chip = o.overdue ? '<span class="chip red">Overdue</span>'
        : num(full.shortage_lb) > 0 ? '<span class="chip amber">Short ' + fmt(full.shortage_lb) + ' lb</span>'
          : full.inventory_ready ? '<span class="chip green">Stock ready</span>' : '<span class="chip grey">' + fmt(o.remaining_lb) + ' lb left</span>';
      // The open-orders list has no created time; it is filled from the order record once this area opens.
      return '<button class="row" style="' + cols + '" data-order="' + esc(full.order_id || o.order_number) + '"><div class="mono" style="font-size:16px">' + esc(o.order_number) + '</div><div style="font-weight:600">' + esc(o.customer) + '</div><div>' + short(o.requested_ship_date) + '</div><div>' + chip + '</div>'
        + '<div class="ts" data-order-ts="' + esc(full.order_id || o.order_number) + '"><span>Created …</span><span class="sep">·</span><span>Updated …</span></div></button>';
    }).join('');
    const near = '<div class="near-body"><div class="near-head"><div class="near-title">Orders due</div><div class="near-meta">Overdue + this week</div></div>'
      + '<div class="cols" style="' + cols + '"><div>Order</div><div>Customer</div><div>Ship by</div><div>Status</div></div>'
      + '<div class="rows">' + (rows || '<div class="row muted" style="grid-template-columns:1fr">Nothing due this week</div>') + '</div>'
      + '<div class="foot">' + (list.length > 5 ? '+' + (list.length - 5) + ' more · ' : '') + 'shipped only with a recorded BOL</div></div>';
    return { far, near };
  }

  function weekAdjusts() { return postedTx().filter((t) => t.type === 'adjust'); }

  function qualityArea() {
    const ex = data.exceptions;
    const exConnected = !failed(ex);
    const ekLots = products.filter((p) => isEK(p.name));
    const ekLb = ekLots.reduce((s, p) => s + p.qty, 0);
    const ekLotCount = ekLots.reduce((s, p) => s + p.lots.length, 0);
    const big = weekAdjusts().filter((t) => (t.lines || []).some((l) => Math.abs(num(l.quantity_lb)) >= 500));
    const far = '<div class="far-body">' + kicker('quality', 'Quality &amp; traceability')
      + (exConnected ? '<div class="big sm">' + ex.count + ' open</div><div class="sub">exceptions</div>'
        : '<div class="big sm" style="color:var(--ink-4)">— holds</div><div class="sub">' + nc('GET /exceptions needs a signed-in actor key; the dashboard key gets 403') + ' exceptions &amp; holds</div>')
      + '<div class="signal muted"><span class="dot grey"></span>' + ekLotCount + ' extra-kosher lot' + (ekLotCount === 1 ? '' : 's') + ' on hand</div></div>';
    const near = '<div class="near-body"><div class="near-title">Quality &amp; traceability</div><div class="boxes">'
      + (exConnected ? '<div class="box"><div class="lbl">Open exceptions</div><div class="val">' + ex.count + '</div></div>'
        : '<div class="box nc-block"><div class="lbl">Exceptions &amp; holds</div><div class="val">' + nc('GET /exceptions is actor-only (403 for the dashboard key)') + '</div></div>')
      + '<div class="box"><div class="lbl">Extra-kosher on hand</div><div class="val">' + ekLotCount + ' lots · ' + fmt(ekLb) + ' lb</div></div>'
      + '<div class="box"><div class="lbl">Large corrections (≥ 500 lb)</div><div class="val">' + big.length + ' in the last 7 days</div></div>'
      + '<a class="box" href="' + esc(dashUrl('traceability.html')) + '" target="_blank" rel="noopener" draggable="false"><div class="lbl">Trace a lot</div><div class="val">Open traceability ↗</div></a>'
      + '</div><div class="foot">Mock recall schedule ' + nc('No endpoint for mock recalls') + '</div></div>';
    return { far, near };
  }

  function reportsArea() {
    const adj = weekAdjusts();
    const latest = adj.slice().sort((a, b) => String(b.created_at).localeCompare(String(a.created_at)))[0];
    const far = '<div class="far-body">' + kicker('reports') + '<div class="big sm">' + (failed(data.tx) ? '—' : adj.length) + ' corrections</div><div class="sub">adjust entries, last 7 days</div>'
      + '<div class="signal muted"><span class="dot grey"></span>' + (latest ? 'Latest ' + esc(stamp(latest.created_at)) + (latest.operator_id ? ' · ' + esc(latest.operator_id) : '') : 'No corrections in 7 days') + '</div></div>';
    const link = (label, meta, url) => '<a class="row" href="' + esc(url) + '" target="_blank" rel="noopener" draggable="false" style="display:flex;justify-content:space-between;color:inherit;text-decoration:none"><div style="font-weight:600">' + label + ' ↗</div><div class="muted">' + meta + '</div></a>';
    const near = '<div class="near-body"><div class="near-title">Reports</div><div class="rows">'
      + link('Shift summary', 'who recorded what, by day', dashUrl('shift-summary.html'))
      + link('Entry history', 'every transaction by day', dashUrl('history.html'))
      + link('Orders board', 'open, ready, overdue', dashUrl('index.html', { section: 'orders' }))
      + link('Traceability', 'forward and back', dashUrl('traceability.html'))
      + '<div class="row" style="display:flex;justify-content:space-between"><div style="font-weight:600">Owner weekly view</div><div>' + nc('No endpoint for the owner weekly view') + '</div></div>'
      + '</div></div>';
    return { far, near };
  }

  // Inventory shows stock alerts only. Overdue orders sit on Orders & shipping and overdue
  // expected receipts on Receiving; an order short on stock is a stock problem, so it stays.
  function stockAlerts() {
    const out = [];
    const n = shortOrders().length;
    if (n) out.push(['amber', n + ' open order' + (n > 1 ? 's' : '') + ' short on stock']);
    for (const s of [data.supIng, data.supPkg]) if (!failed(s)) for (const it of s.items.filter((i) => i.is_low)) out.push(['red', it.name + ' below reorder']);
    return out;
  }

  function catSummary(t) {
    const list = ofType(t);
    const qty = list.reduce((s, p) => s + p.qty, 0);
    if (t === 'finished') return { big: fmt(list.reduce((s, p) => s + p.units, 0)) + ' cs', meta: list.length + ' SKUs · ' + fmt(qty) + ' lb' };
    if (t === 'packaging') return { big: list.length + ' items', meta: list.length + ' items in stock' };
    if (t === 'ingredient') return { big: list.length + ' items', meta: list.length + ' items · ' + fmt(qty) + ' lb' };
    return { big: fmt(qty) + ' lb', meta: list.length + ' products · ' + fmt(qty) + ' lb' };
  }

  function inventoryFar() {
    if (failed(data.inv)) return '<div class="far-body">' + kicker('inventory') + errLine(data.inv) + '</div>';
    const cell = (t) => { const c = catSummary(t); return '<div class="inv-cell"><div class="lbl">' + TILES[t].name + '</div><div class="val">' + c.big + '</div><div class="meta">' + esc(c.meta) + '</div></div>'; };
    const al = stockAlerts();
    const sig = al.length ? al.slice(0, 3).map(([c, m]) => '<div class="signal t-' + c + '"><span class="dot ' + c + '"></span>' + esc(m) + '</div>').join('')
        + (al.length > 3 ? '<div class="signal muted"><span class="dot grey"></span>+' + (al.length - 3) + ' more stock alerts</div>' : '')
      : '<div class="signal t-green"><span class="dot green"></span>No short or low-stock alerts</div>';
    return '<div class="far-body inv-far"><div class="near-head">' + kicker('inventory') + '<div class="inv-upd muted">updated ' + data.loadedAt.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', timeZone: CFG.timeZone }) + '</div></div>'
      + '<div class="inv-grid">' + cell('ingredient') + cell('batch') + cell('packaging') + cell('finished') + '</div>'
      + '<div class="signals">' + sig + '</div></div>';
  }

  // ---------- product grid layout ----------
  function gridLayout(n) {
    const W = GRID.w - GRID.left - GRID.right, H = GRID.h - GRID.top - GRID.bottom, g = GRID.gap, ar = CARD.w / CARD.h;
    let best = { cw: 0 };
    for (let c = 1; c <= Math.max(1, n); c++) {
      const r = Math.ceil(n / c);
      let cw = (W - (c - 1) * g) / c, ch = cw / ar;
      if (r * ch + (r - 1) * g > H) { ch = (H - (r - 1) * g) / r; cw = ch * ar; }
      if (cw > best.cw) best = { cols: c, rows: r, cw, ch };
    }
    return best;
  }

  // Ingredient view state: sort order and Grid / Relationship map. Kept for the session.
  const SORTS = {
    qty: { label: 'Quantity ↓', fn: (a, b) => b.qty - a.qty || a.name.localeCompare(b.name) },
    az: { label: 'Name A–Z', fn: (a, b) => a.name.localeCompare(b.name, 'en', { sensitivity: 'base' }) },
    za: { label: 'Name Z–A', fn: (a, b) => b.name.localeCompare(a.name, 'en', { sensitivity: 'base' }) }
  };
  const view = { sort: { ingredient: 'qty' }, mode: { ingredient: 'grid' }, basis: 'recipe' };
  const sortedOfType = (t) => { const k = view.sort[t] || 'qty'; return ofType(t).slice().sort(SORTS[k].fn); };

  function ingredientControls() {
    const seg = (attr, cur, opts) => '<div class="seg" role="group">' + opts.map(([v, l]) => '<button type="button" ' + attr + '="' + v + '" aria-pressed="' + (v === cur) + '"' + (v === cur ? ' class="on"' : '') + '>' + l + '</button>').join('') + '</div>';
    return '<div class="ing-ctl">' + seg('data-view', view.mode.ingredient, [['grid', 'Grid view'], ['map', 'Relationship map']])
      + '<div class="sortwrap"' + (view.mode.ingredient === 'map' ? ' hidden' : '') + '><span class="dim">Sort</span>' + seg('data-sort', view.sort.ingredient, Object.entries(SORTS).map(([k, v]) => [k, v.label])) + '</div></div>';
  }

  function tileHTML(t) {
    const tile = TILES[t], list = ofType(t), sum = catSummary(t);
    const top = list.slice(0, 5);
    const max = top.length ? top[0].qty : 1;
    const rows = top.map((p) => t === 'batch'
      ? '<div class="bar-row"><span>' + esc(p.name.replace(/^Batch\s+/i, '')) + '</span><span class="bar"><span style="width:' + Math.max(2, Math.round(p.qty / max * 100)) + '%"></span></span><span class="mono" style="text-align:right">' + qtyText(p) + '</span></div>'
      : '<div><span>' + esc(p.name) + '</span><span class="mono">' + qtyText(p) + '</span></div>').join('');
    const zeroNote = t === 'packaging' && !failed(data.supPkg) ? ' · ' + (data.supPkg.count - list.length) + ' at zero' : t === 'ingredient' && !failed(data.supIng) ? ' · ' + (data.supIng.count - list.length) + ' at zero' : '';
    const t1 = '<div class="layer fill tile-body t1"><div class="tile-head"><div class="tile-title">' + tile.name + '</div><div class="tile-meta">' + esc(sum.meta) + '</div></div>'
      + '<div class="tile-list">' + (rows || '<div class="muted">Nothing on hand</div>') + '</div>'
      + '<div class="tile-more">' + (list.length > top.length ? '+' + (list.length - top.length) + ' more' : '') + esc(zeroNote) + '</div></div>';
    // Cards are positioned by layoutCards(), so a re-sort only moves them.
    const cards = list.map((p) => '<div class="node pcard' + (isEK(p.name) ? ' warn' : '') + '" data-node="p' + p.id + '">'
      + '<div class="layer far" style="width:' + CARD.w + 'px;height:' + CARD.h + 'px">' + productCardHTML(p) + '</div>'
      + '<div class="layer near ws-slot" style="width:' + WS.w + 'px;height:' + WS.h + 'px"></div>'
      + '<button class="hit" data-fly="p' + p.id + '" aria-label="Zoom into ' + esc(p.name) + '"></button></div>').join('');
    const t2 = '<div class="layer t2' + (t === 'ingredient' && view.mode.ingredient === 'map' ? ' mapmode' : '') + '" style="width:' + GRID.w + 'px;height:' + GRID.h + 'px;transform:scale(' + GRID.k + ')">'
      + '<div class="grid-head"><div class="t">' + tile.name + '</div>' + (t === 'ingredient' ? ingredientControls() : '') + '<div class="m">' + esc(sum.meta) + ' on hand</div></div>' + cards
      + (t === 'ingredient' ? '<div class="relmap" data-relmap></div>' : '') + '</div>';
    return '<div class="node tile' + (tile.accent ? ' accent' : '') + '" data-node="t-' + t + '" style="left:' + tile.dx + 'px;top:' + tile.dy + 'px;width:' + TILE_W + 'px;height:' + TILE_H + 'px">'
      + t1 + t2 + '<button class="hit" data-fly="t-' + t + '" aria-label="Zoom into ' + tile.name + '"></button></div>';
  }

  // Place a category's cards in the current sort order and refresh their camera rects.
  function layoutCards(t) {
    const list = sortedOfType(t), L = gridLayout(list.length);
    const tileNode = nodes.get('t-' + t);
    const hidden = t === 'ingredient' && view.mode.ingredient === 'map';
    list.forEach((p, i) => {
      const c = i % L.cols, r = Math.floor(i / L.cols);
      const x = GRID.left + c * (L.cw + GRID.gap), y = GRID.top + r * (L.ch + GRID.gap);
      p.design = { x, y, w: L.cw, h: L.ch };
      const n = nodes.get('p' + p.id);
      if (!n || !n.el) return;
      const k = L.cw / CARD.w;
      Object.assign(n.el.style, { left: x + 'px', top: y + 'px', width: L.cw + 'px', height: L.ch + 'px' });
      n.el.style.setProperty('--k', k);
      n.el.querySelector(':scope > .far').style.transform = 'scale(' + k + ')';
      n.el.querySelector(':scope > .ws-slot').style.transform = 'scale(' + L.cw / WS.w + ')';
      n.hidden = hidden;
      if (tileNode) n.rect = { x: tileNode.rect.x + x * GRID.k, y: tileNode.rect.y + y * GRID.k, w: L.cw * GRID.k, h: L.ch * GRID.k };
    });
  }

  // Name first (it wraps), then quantity, then lots · SKU as secondary text.
  function productCardHTML(p) {
    const lots = p.lots.length;
    const meta = [lots + ' lot' + (lots === 1 ? '' : 's'), p.type === 'finished' ? fmt(p.qty) + ' lb' : '', p.sku ? 'SKU ' + p.sku : ''].filter(Boolean).join(' · ');
    return '<div class="pc-body"><div class="pc-name">' + esc(p.name) + '</div><div class="pc-qty">' + qtyText(p) + '</div><div class="pc-meta">' + esc(meta) + '</div>'
      + (isEK(p.name) ? '<div class="pc-foot t-amber">Extra-kosher (SS)</div>' : '') + '</div>';
  }

  function inventoryNear() {
    const al = stockAlerts();
    const reds = al.filter((a) => a[0] === 'red').length, ambers = al.length - reds;
    return '<div class="layer fill near" style="padding:0">'
      + '<div class="inv-near-head"><div style="display:flex;align-items:baseline;gap:10px"><div style="font-size:20px;font-weight:600">Inventory</div><div style="font-size:11px" class="muted">4 categories · ' + al.length + ' stock alert' + (al.length === 1 ? '' : 's') + '</div></div>'
      + '<div style="display:flex;gap:6px;font-size:10px">' + (ambers ? '<span class="chip amber" style="font-size:10px;padding:3px 8px">' + ambers + ' warning' + (ambers > 1 ? 's' : '') + '</span>' : '') + (reds ? '<span class="chip red" style="font-size:10px;padding:3px 8px">' + reds + ' urgent</span>' : '') + '</div></div>'
      + Object.keys(TILES).map(tileHTML).join('') + '</div>';
  }

  // ---------- build world ----------
  // Material flow between areas: [from, side, along, to, side, along, label]. Anchors are
  // computed from the real card bounds after layout, so they follow any card resize.
  const FLOWS = [
    ['receiving', 'r', 0.5, 'inventory', 'l', 0.4, 'receive'],
    ['inventory', 't', 0.26, 'production', 'b', 0.61, 'to bake'],
    ['production', 'r', 0.5, 'inventory', 't', 0.69, 'bulk'],
    ['inventory', 't', 0.9, 'packing', 'b', 0.36, 'to pack'],
    ['packing', 'b', 0.64, 'inventory', 'r', 0.32, 'cases'],
    ['inventory', 'r', 0.8, 'shipping', 'l', 0.6, 'to ship']
  ];
  const LINKS = [['quality', 't', 0.5, 'inventory', 'b', 0.125, 'trace'], ['reports', 't', 0.5, 'inventory', 'b', 0.875, 'reports']];
  const NORMAL = { l: [-1, 0], r: [1, 0], t: [0, -1], b: [0, 1] };
  function anchor(id, side, f) {
    const a = AREAS[id];
    if (side === 'l') return [a.x, a.y + a.h * f];
    if (side === 'r') return [a.x + a.w, a.y + a.h * f];
    if (side === 't') return [a.x + a.w * f, a.y];
    return [a.x + a.w * f, a.y + a.h];
  }
  function curve([from, fs, ff, to, ts, tf], gapEnd) {
    const p0 = anchor(from, fs, ff), p3 = anchor(to, ts, tf);
    const n0 = NORMAL[fs], n3 = NORMAL[ts];
    p3[0] += n3[0] * gapEnd; p3[1] += n3[1] * gapEnd;
    const d = Math.max(60, Math.hypot(p3[0] - p0[0], p3[1] - p0[1]) * 0.45);
    const c1 = [p0[0] + n0[0] * d, p0[1] + n0[1] * d], c2 = [p3[0] + n3[0] * d, p3[1] + n3[1] * d];
    const mid = [0, 1].map((i) => 0.125 * p0[i] + 0.375 * c1[i] + 0.375 * c2[i] + 0.125 * p3[i]);
    // Tangent at the midpoint, for placing the label beside the curve rather than on it.
    const tan = [0, 1].map((i) => 0.75 * (p3[i] + c2[i] - c1[i] - p0[i]));
    return { d: 'M' + p0.join(' ') + ' C ' + c1.join(' ') + ', ' + c2.join(' ') + ', ' + p3.join(' '), mid, tan };
  }
  function flowSVG() {
    const W = world.offsetWidth, H = world.offsetHeight;
    const label = (c, text, cls) => {
      const len = Math.hypot(c.tan[0], c.tan[1]) || 1;
      let nx = -c.tan[1] / len, ny = c.tan[0] / len;
      if (nx < -0.2 || (Math.abs(nx) <= 0.2 && ny > 0)) { nx = -nx; ny = -ny; }
      const anchor = nx > 0.35 ? 'start' : nx < -0.35 ? 'end' : 'middle';
      return '<text class="' + cls + '" text-anchor="' + anchor + '" dominant-baseline="middle" x="' + (c.mid[0] + nx * 16).toFixed(1) + '" y="' + (c.mid[1] + ny * 16).toFixed(1) + '">' + text + '</text>';
    };
    const flows = FLOWS.map((f) => curve(f, 4)), links = LINKS.map((f) => curve(f, 0));
    // Faint zones (shown by the Operational map theme) group the areas by stage.
    const zone = (ids, name) => {
      const rs = ids.map((i) => AREAS[i]);
      const x = Math.min(...rs.map((r) => r.x)) - 28, y = Math.min(...rs.map((r) => r.y)) - 44;
      const x2 = Math.max(...rs.map((r) => r.x + r.w)) + 28, y2 = Math.max(...rs.map((r) => r.y + r.h)) + 28;
      return '<rect x="' + x + '" y="' + y + '" width="' + (x2 - x) + '" height="' + (y2 - y) + '" rx="24"></rect><text x="' + (x + 22) + '" y="' + (y + 28) + '">' + name + '</text>';
    };
    return '<svg class="flow" width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H + '" aria-hidden="true"><defs><marker id="faArrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z"></path></marker></defs>'
      + '<g class="zones">' + zone(['receiving'], 'Inbound') + zone(['production', 'packing'], 'Make &amp; pack') + zone(['shipping'], 'Outbound') + zone(['quality', 'reports'], 'Oversight') + '</g>'
      + '<g class="flows">' + flows.map((c) => '<path d="' + c.d + '" marker-end="url(#faArrow)"></path>').join('') + '</g>'
      + '<g class="links">' + links.map((c) => '<path d="' + c.d + '"></path>').join('') + '</g>'
      + '<g id="flowtext">' + flows.map((c, i) => label(c, FLOWS[i][6], 'fl')).join('') + links.map((c, i) => label(c, LINKS[i][6], 'fl dim')).join('') + '</g></svg>';
  }

  function buildWorld() {
    const builders = { production: productionArea, packing: packingArea, receiving: receivingArea, shipping: shippingArea, quality: qualityArea, reports: reportsArea };
    let html = '<div class="dots" id="dots"></div><div id="flowwrap"></div>';
    for (const [id, a] of Object.entries(AREAS)) {
      let far, near;
      if (id === 'inventory') { far = '<div class="layer fill far">' + inventoryFar() + '</div>'; near = inventoryNear(); }
      else {
        const b = builders[id]();
        far = '<div class="layer fill far">' + b.far + '</div>';
        near = '<div class="layer near" style="width:' + a.w * 2 + 'px;transform:scale(0.5)">' + b.near + '</div>';
      }
      html += '<div class="node area' + (a.hub ? ' hub' : '') + '" data-node="' + id + '" style="left:' + a.x + 'px;width:' + a.w + 'px">' + far + near
        + '<button class="hit" data-fly="' + id + '" aria-label="Zoom into ' + esc(a.name) + '"></button></div>';
    }
    world.innerHTML = html;

    nodes.clear();
    for (const [id, a] of Object.entries(AREAS)) nodes.set(id, { id, kind: 'area', name: a.name, rect: a, parent: null });
    for (const [t, tile] of Object.entries(TILES)) {
      nodes.set('t-' + t, { id: 't-' + t, kind: 'tile', name: tile.name, rect: { x: 0, y: 0, w: TILE_W, h: TILE_H }, parent: 'inventory', type: t });
      for (const p of ofType(t)) nodes.set('p' + p.id, { id: 'p' + p.id, kind: 'product', name: p.name, product: p, parent: 't-' + t, rect: { x: 0, y: 0, w: 1, h: 1 } });
    }
    for (const n of nodes.values()) {
      n.el = world.querySelector('[data-node="' + n.id + '"]');
      n.hit = n.el && n.el.querySelector(':scope > .hit');
    }
    layoutWorld();
  }

  // Content-aware layout. Each area is as tall as its content needs at both zoom levels
  // (overview card, and the 2x near view scaled by 0.5); paired cards share the taller
  // height; rows stack with a fixed gap and the middle row is centred on Inventory.
  function layoutWorld() {
    const need = {};
    for (const [id, a] of Object.entries(AREAS)) {
      const el = nodes.get(id).el;
      const far = el.querySelector(':scope > .far > .far-body');
      far.style.height = 'auto';
      let h = far.offsetHeight;
      far.style.height = '';
      if (id === 'inventory') h = Math.max(h, TILES.packaging.dy + TILE_H + 24);
      else {
        const nb = el.querySelector(':scope > .near > .near-body');
        if (nb) h = Math.max(h, Math.ceil(nb.offsetHeight / 2));
      }
      need[id] = Math.ceil(h);
    }
    const pairH = {};
    for (const [id, a] of Object.entries(AREAS)) if (a.pair) pairH[a.pair] = Math.max(pairH[a.pair] || 0, need[id]);
    for (const [id, a] of Object.entries(AREAS)) a.h = a.pair ? pairH[a.pair] : need[id];
    const rowH = [0, 1, 2].map((r) => Math.max(...Object.values(AREAS).filter((a) => a.row === r).map((a) => a.h)));
    const rowY = [ROW_TOP, ROW_TOP + rowH[0] + ROW_GAP, ROW_TOP + rowH[0] + rowH[1] + 2 * ROW_GAP];
    for (const a of Object.values(AREAS)) a.y = rowY[a.row] + (rowH[a.row] - a.h) / 2;
    for (const [id, a] of Object.entries(AREAS)) {
      const el = nodes.get(id).el;
      Object.assign(el.style, { top: a.y + 'px', height: a.h + 'px' });
      const near = el.querySelector(':scope > .near');
      if (id !== 'inventory') near.style.height = a.h * 2 + 'px';
    }
    const all = Object.values(AREAS);
    BOX.x = Math.min(...all.map((a) => a.x)); BOX.y = Math.min(...all.map((a) => a.y));
    BOX.w = Math.max(...all.map((a) => a.x + a.w)) - BOX.x; BOX.h = Math.max(...all.map((a) => a.y + a.h)) - BOX.y;
    world.style.width = BOX.x * 2 + BOX.w + 'px';
    world.style.height = BOX.y * 2 + BOX.h + 'px';
    document.getElementById('flowwrap').innerHTML = flowSVG();
    const inv = AREAS.inventory;
    for (const [t, tile] of Object.entries(TILES)) {
      nodes.get('t-' + t).rect = { x: inv.x + tile.dx, y: inv.y + tile.dy, w: TILE_W, h: TILE_H };
      layoutCards(t);
    }
    buildMinimap();
  }

  // ---------- camera ----------
  // Fit a rect into the viewport, keeping clear of the floating bar at the bottom and,
  // while it is open, the record panel on the right.
  const BOTTOM_CHROME = 72;
  function fit(r, pad) {
    const v = vpSize();
    const side = selection && v.w > 900 ? 460 : 0;
    const w = v.w - side, h = v.h - BOTTOM_CHROME;
    const s = Math.min(w / r.w, h / r.h) * (pad || 0.92);
    return { cx: r.x + r.w / 2 + side / 2 / s, cy: r.y + r.h / 2 + BOTTOM_CHROME / 2 / s, s };
  }
  function limits() {
    let smallest = Infinity;
    for (const n of nodes.values()) if (n.kind === 'product') smallest = Math.min(smallest, n.rect.w);
    const deep = Number.isFinite(smallest) ? fit({ x: 0, y: 0, w: smallest, h: smallest * CARD.h / CARD.w }, 1).s * 1.6 : 12;
    return { min: fit(BOX, 0.95).s * 0.6, max: Math.max(12, deep) };
  }
  function setCam(next, animate) {
    const L = limits();
    cam = { cx: next.cx, cy: next.cy, s: Math.max(L.min, Math.min(L.max, next.s)) };
    world.classList.toggle('flying', !!animate);
    const v = vpSize();
    world.style.transform = 'translate(' + v.w / 2 + 'px,' + v.h / 2 + 'px) scale(' + cam.s + ') translate(' + -cam.cx + 'px,' + -cam.cy + 'px)';
    scheduleUpdate();
  }
  function flyTo(id) {
    const n = nodes.get(id);
    if (!n) return;
    setCam(fit(n.rect, n.kind === 'product' ? 0.86 : 0.92), true);
  }
  const goHome = () => { closeInspector(); setCam(fit(BOX, 0.95), true); };

  let raf = 0;
  function scheduleUpdate() { if (!raf) raf = requestAnimationFrame(() => { raf = 0; update(); }); }

  const inside = (r, x, y) => x >= r.x && x <= r.x + r.w && y >= r.y && y <= r.y + r.h;
  const overlaps = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;

  function update() {
    const v = vpSize();
    const frac = (r) => Math.max(r.w * cam.s / v.w, r.h * cam.s / v.h);
    const open = new Set();
    for (const n of nodes.values()) {
      if (n.kind === 'area') { if (frac(n.rect) >= OPEN_AT.area) open.add(n.id); }
    }
    // Deeper levels open only where the camera is: tiles that are on screen, and the one
    // product card under the viewport centre (each open product loads its own data).
    const view = { x: cam.cx - v.w / 2 / cam.s, y: cam.cy - v.h / 2 / cam.s, w: v.w / cam.s, h: v.h / cam.s };
    for (const n of nodes.values()) if (n.kind === 'tile' && open.has(n.parent) && overlaps(n.rect, view) && frac(n.rect) >= OPEN_AT.tile) open.add(n.id);
    for (const n of nodes.values()) if (n.kind === 'product' && !n.hidden && open.has(n.parent) && inside(n.rect, cam.cx, cam.cy) && frac(n.rect) >= OPEN_AT.product) open.add(n.id);
    if (open.has('shipping')) loadOrderStamps();
    for (const n of nodes.values()) {
      if (!n.el) continue;
      const isOpen = open.has(n.id);
      if (n.el.classList.contains('open') !== isOpen) n.el.classList.toggle('open', isOpen);
      const parentOpen = !n.parent || open.has(n.parent);
      if (n.hit) {
        const active = parentOpen && !isOpen;
        n.hit.classList.toggle('active', active);
        n.hit.tabIndex = active ? 0 : -1;
      }
      if (n.kind === 'product' && isOpen && !n.loaded) { n.loaded = true; buildWorkingScreen(n); }
    }
    // tiles: swap list (t1) for product grid (t2)
    for (const n of nodes.values()) if (n.kind === 'tile' && n.el) {
      const o = open.has(n.id);
      n.el.querySelector('.t1').style.opacity = o ? 0 : 1;
      const t2 = n.el.querySelector('.t2');
      t2.style.opacity = o ? 1 : 0;
      t2.style.pointerEvents = o ? 'auto' : 'none';
    }
    const anyArea = [...open].some((id) => nodes.get(id).kind === 'area');
    const anyTile = [...open].some((id) => nodes.get(id).kind === 'tile');
    $('#dots').style.opacity = anyTile ? 0 : 1;
    $('#flowtext').style.opacity = anyArea ? 0 : 1;

    focus = { area: null, tile: null, product: null };
    for (const n of nodes.values()) {
      if (!open.has(n.id) || n.hidden || !inside(n.rect, cam.cx, cam.cy)) continue;
      focus[n.kind] = n.id;
    }
    renderChrome();
  }

  // ---------- chrome: crumbs, level, minimap, assistant ----------
  function crumbs() {
    const list = [{ label: 'Entire factory', go: goHome }];
    for (const k of ['area', 'tile', 'product']) if (focus[k]) { const id = focus[k]; list.push({ label: nodes.get(id).name, go: () => { closeInspector(); flyTo(id); } }); }
    if (selection) list.push({ label: selection.label, go: () => {} });
    return list;
  }

  function renderChrome() {
    const nav = $('#crumbs');
    nav.innerHTML = '';
    crumbs().forEach((c, i, arr) => {
      if (i) { const s = document.createElement('span'); s.className = 'sep'; s.textContent = '/'; s.setAttribute('aria-hidden', 'true'); nav.appendChild(s); }
      const b = document.createElement('button');
      b.type = 'button'; b.textContent = c.label; b.title = c.label;
      if (i === arr.length - 1) { b.className = 'last'; b.setAttribute('aria-current', 'location'); }
      b.onclick = c.go;
      nav.appendChild(b);
    });
    const depth = focus.product ? 3 : focus.tile ? 2 : focus.area ? 1 : 0;
    const names = ['Entire factory', 'Area', 'Category', 'Working screen'];
    $('#level').innerHTML = '<b>' + names[depth] + '</b><span>' + Math.round(cam.s * 100) + '%</span>';
    const ctx = selection ? selection.label : focus.product ? nodes.get(focus.product).name : focus.tile ? nodes.get(focus.tile).name : focus.area ? nodes.get(focus.area).name : 'the whole factory';
    $('#ctx').textContent = ctx;
    $('#ask').placeholder = 'Ask about ' + ctx + '…';
    renderMinimap();
  }

  function buildMinimap() {
    const k = 200 / BOX.w, mm = $('#mm');
    mm.style.height = Math.round(BOX.h * k) + 'px';
    mm.innerHTML = Object.entries(AREAS).map(([id, r]) => '<button type="button" data-mm="' + id + '" class="' + (r.hub ? 'hub' : '') + '" aria-label="Go to ' + esc(r.name) + '" style="left:' + (r.x - BOX.x) * k + 'px;top:' + (r.y - BOX.y) * k + 'px;width:' + r.w * k + 'px;height:' + r.h * k + 'px"></button>').join('') + '<div class="vr"></div>';
    mm.querySelectorAll('[data-mm]').forEach((b) => { b.onclick = () => { closeInspector(); flyTo(b.dataset.mm); }; });
    $('#mm-src').textContent = CFG.env === 'production' ? 'Live production data' : 'Live ' + CFG.env + ' data';
  }
  function renderMinimap() {
    const k = 200 / BOX.w, v = vpSize();
    const vw = v.w / cam.s, vh = v.h / cam.s;
    const vr = $('#mm .vr');
    if (!vr) return;
    Object.assign(vr.style, { left: (cam.cx - vw / 2 - BOX.x) * k + 'px', top: (cam.cy - vh / 2 - BOX.y) * k + 'px', width: Math.max(4, vw * k) + 'px', height: Math.max(3, vh * k) + 'px' });
    $('#mm').querySelectorAll('[data-mm]').forEach((b) => b.classList.toggle('focus', b.dataset.mm === focus.area));
  }

  // ---------- product working screen ----------
  async function lotDetail(code, pid) {
    const key = pid + '|' + code;
    if (!lotCache.has(key)) lotCache.set(key, API.get('/dashboard/api/lot/' + encodeURIComponent(code), { product_id: pid }).catch((e) => { lotCache.delete(key); throw e; }));
    return lotCache.get(key);
  }
  const firstDate = (d) => (d && d.timeline && d.timeline.length ? d.timeline.map((t) => t.date).sort()[0] : null);

  function primaryTask(p) {
    if (p.type === 'batch') return ['Pack from this', 'pack'];
    if (p.type === 'finished') return ['Ship', 'ship'];
    if (p.type === 'ingredient' || p.type === 'packaging') return ['Receive', 'receive'];
    return ['Make', 'make'];
  }

  async function buildWorkingScreen(n) {
    const p = n.product;
    const slot = n.el.querySelector('.ws-slot');
    const [label, act] = primaryTask(p);
    const typeChip = { batch: 'Bulk · lb', finished: p.caseLb ? 'Finished · ' + fmt(p.caseLb) + ' lb case' : 'Finished', ingredient: 'Ingredient · ' + (p.unit || 'lb'), packaging: 'Packaging' }[p.type] || p.type;
    slot.innerHTML = '<div class="ws"><div class="ws-head"><div style="display:flex;flex-direction:column;gap:6px;min-width:0"><div class="ws-crumb">Inventory · ' + esc(TILES[p.type] ? TILES[p.type].name : p.type) + '</div>'
      + '<div class="ws-title"><h2>' + esc(p.name) + '</h2><span class="chip grey">' + esc(typeChip) + '</span>' + (isEK(p.name) ? '<span class="chip amber">Extra-kosher (SS)</span>' : '') + (p.sku ? '<span class="chip blue">SKU ' + esc(p.sku) + '</span>' : '') + '</div></div>'
      + '<div class="ws-actions"><button class="fa-btn" data-trace="1">Trace</button><button class="fa-btn" data-task="adjust">Adjust</button><button class="fa-btn" data-task="adjust" title="No separate count page: opens Adjust, choose reason Physical count">Record count</button><button class="fa-btn fa-primary" data-task="' + act + '">' + label + '</button></div></div>'
      + '<div class="kpis" data-kpis><div><div class="k">On hand</div><div class="v">' + qtyText(p) + '</div></div><div><div class="k">Lots</div><div class="v">' + p.lots.length + '</div></div><div><div class="k">Oldest lot</div><div class="v" data-k="oldest">…</div></div><div><div class="k">In · last 7 days</div><div class="v" data-k="in">…</div></div><div><div class="k">Out · last 7 days</div><div class="v" data-k="out">…</div></div></div>'
      + '<div class="ws-main"><div class="ws-col"><div class="panel flush" style="flex:1"><div class="panel-head"><h3>Lots on hand</h3><div class="dim">Click a lot for its full record</div></div>'
      + '<div class="lot-cols"><div>Lot</div><div>First entry</div><div>Source</div><div>On hand</div><div>Age</div></div><div class="lot-rows" data-lots><div class="loading" style="padding:14px 18px">Loading lots…</div></div></div>'
      + '<div class="panel"><h3>Planned production</h3><div class="mini-rows" data-runs><div class="loading">Loading…</div></div></div></div>'
      + '<div class="ws-col"><div class="panel"><h3>Where it comes from and goes · 30 days</h3><div data-flow class="loading">Loading…</div></div>'
      + '<div class="panel" style="flex:1"><h3>Recent activity</h3><div class="activity" data-activity><div class="act"><span></span><span class="loading">Loading…</span><span></span></div></div></div></div></div></div>';
    const ws = slot.firstChild;
    ws.addEventListener('click', (e) => {
      const t = e.target.closest('[data-task],[data-trace],[data-lot]');
      if (!t) return;
      if (t.dataset.task) openTab(taskUrl(t.dataset.task));
      else if (t.dataset.trace) { const l = (n.lotsSorted || p.lots)[0]; if (l) openLot(l.code, p.id, p.name); }
    });

    const since30 = addDays(TODAY, -29), since7 = addDays(TODAY, -6);
    const [lotsR, txR, runsR] = await Promise.allSettled([
      API.get('/dashboard/api/product/' + p.id + '/lots'),
      API.get('/transactions/history', { product_name: p.name, since: since30, limit: 500 }),
      API.get('/production/runs', { product: p.id, from: TODAY })
    ]);

    // lots
    const lotsEl = ws.querySelector('[data-lots]');
    let lots = lotsR.status === 'fulfilled' ? lotsR.value.lots.map((l) => ({ code: l.lot_code, qty: num(l.on_hand_lbs), source: l.entry_source })) : p.lots.map((l) => ({ code: l.code, qty: l.qty }));
    lots = lots.filter((l) => l.qty > 0);
    // First-entry dates: take them from the 30-day history first, then look up at most 20
    // older lots one by one (each lookup is a request against the API's small DB pool).
    if (txR.status === 'fulfilled') {
      const firstSeen = new Map();
      for (const t of txR.value.transactions) {
        if (t.status !== 'posted') continue;
        for (const l of t.lines || []) {
          if (l.product_id !== p.id || num(l.quantity_lb) <= 0) continue;
          if (!firstSeen.has(l.lot_code) || t.business_date < firstSeen.get(l.lot_code)) firstSeen.set(l.lot_code, t.business_date);
        }
      }
      for (const l of lots) if (firstSeen.has(l.code)) l.first = firstSeen.get(l.code);
    }
    const unknown = lots.filter((l) => !l.first);
    const details = await Promise.allSettled(unknown.slice(0, 20).map((l) => lotDetail(l.code, p.id)));
    details.forEach((d, i) => { if (d.status === 'fulfilled') { unknown[i].first = firstDate(d.value); unknown[i].source = unknown[i].source || d.value.entry_source; } });
    const undated = unknown.length - 20;
    lots.sort((a, b) => String(a.first || '9').localeCompare(String(b.first || '9')));
    n.lotsSorted = lots;
    const unitFmt = (lbs) => (p.type === 'finished' && p.caseLb ? fmt(lbs / p.caseLb) + ' cs' : p.unit && p.unit !== 'lb' ? fmt(lbs) + ' ' + (p.unit === 'unit' ? 'ea' : p.unit) : fmt(lbs) + ' lb');
    lotsEl.innerHTML = lots.length ? lots.map((l) => {
      const age = l.first ? daysBetween(l.first, TODAY) : null;
      return '<button class="lot-row" data-lot="' + esc(l.code) + '" data-pid="' + p.id + '" data-pname="' + esc(p.name) + '"><span class="mono" style="font-size:13px">' + esc(l.code) + '</span><span>' + short(l.first) + '</span><span><span class="chip grey">' + esc(sourceLabel(l.source)) + '</span></span><span style="font-weight:600">' + unitFmt(l.qty) + '</span><span class="muted">' + (age == null ? '—' : age + ' day' + (age === 1 ? '' : 's')) + '</span></button>';
    }).join('') + (undated > 0 ? '<div class="loading" style="padding:10px 18px">' + undated + ' older lots show no date (dates are looked up for at most 20 lots, to keep load on the API low)</div>' : '')
      : '<div class="loading" style="padding:14px 18px">' + (lotsR.status === 'fulfilled' ? 'No lots on hand' : 'Couldn’t load lots') + '</div>';
    const oldest = lots.find((l) => l.first);
    ws.querySelector('[data-k="oldest"]').textContent = oldest ? daysBetween(oldest.first, TODAY) + ' days' + (undated > 0 ? '+' : '') : '—';
    if (undated > 0) ws.querySelector('[data-k="oldest"]').title = 'At least this old: ' + undated + ' lots have no looked-up date';

    // runs
    const runsEl = ws.querySelector('[data-runs]');
    if (runsR.status === 'fulfilled') {
      const runs = runsR.value.runs;
      runsEl.innerHTML = runs.length ? runs.slice(0, 4).map((r) => '<div><span>' + esc(r.run_type) + ' · ' + esc(r.status) + '</span><span class="muted">' + short(String(r.planned_date || '').slice(0, 10)) + '</span></div>').join('') : '<div class="muted">No planned runs from today on (GET /production/runs)</div>';
    } else runsEl.innerHTML = '<div class="t-red">Couldn’t load planned runs</div>';

    // transactions → in/out, flow, activity
    const actEl = ws.querySelector('[data-activity]'), flowEl = ws.querySelector('[data-flow]');
    if (txR.status !== 'fulfilled') {
      actEl.innerHTML = '<div class="act"><span></span><span class="t-red">Couldn’t load activity</span><span></span></div>';
      flowEl.innerHTML = '<span class="t-red">Couldn’t load</span>';
      ws.querySelector('[data-k="in"]').textContent = ws.querySelector('[data-k="out"]').textContent = '—';
      return;
    }
    const txs = txR.value.transactions.filter((t) => t.status === 'posted' && (t.lines || []).some((l) => l.product_id === p.id))
      .sort((a, b) => (b.business_date + b.created_at).localeCompare(a.business_date + a.created_at));
    const mine = (t) => t.lines.filter((l) => l.product_id === p.id);
    const sumMine = (t) => mine(t).reduce((s, l) => s + num(l.quantity_lb), 0);
    let qin = 0, qout = 0;
    for (const t of txs) if (t.business_date >= since7) { const q = sumMine(t); if (q > 0) qin += q; else qout -= q; }
    ws.querySelector('[data-k="in"]').textContent = unitFmt(qin);
    ws.querySelector('[data-k="out"]').textContent = unitFmt(qout);

    const sources = new Map(), uses = new Map();
    for (const t of txs) {
      const q = sumMine(t);
      if (q > 0) sources.set(TX_LABEL[t.type] || t.type, (sources.get(TX_LABEL[t.type] || t.type) || 0) + 1);
      if (q < 0) {
        const outs = t.type === 'ship' ? [t.customer_name ? 'Shipped to ' + t.customer_name : 'Shipped'] : t.lines.filter((l) => l.product_id !== p.id && num(l.quantity_lb) > 0).map((l) => l.product_name);
        if (!outs.length) outs.push(TX_LABEL[t.type] || t.type);
        for (const o of outs) uses.set(o, (uses.get(o) || 0) + -q / outs.length);
      }
    }
    const useList = [...uses.entries()].sort((a, b) => b[1] - a[1]);
    flowEl.className = '';
    flowEl.innerHTML = '<div class="flowline">' + ([...sources.keys()].map((s) => '<span class="pill">' + esc(s) + '</span>').join('') || '<span class="pill">no inflow</span>') + '<span style="color:#8C95A4">→</span><span class="pill me">' + esc(p.name) + '</span><span style="color:#8C95A4">→</span><span class="pill">' + useList.length + ' use' + (useList.length === 1 ? '' : 's') + '</span></div>'
      + '<div style="display:flex;flex-direction:column;gap:6px;font-size:13px;margin-top:8px" class="muted">' + (useList.slice(0, 5).map(([name, q]) => '<div style="display:flex;justify-content:space-between;gap:10px"><span>' + esc(name) + '</span><span class="mono">' + unitFmt(q) + '</span></div>').join('') || '<div>Not used in the last 30 days</div>') + '</div>';

    actEl.innerHTML = txs.slice(0, 9).map((t) => {
      const q = sumMine(t), l = mine(t)[0] || {};
      const color = q > 0 ? 'color:#1A6B43' : '';
      return '<div class="act"><span class="dim">' + short(t.business_date) + '</span><span>' + esc(TX_LABEL[t.type] || t.type) + ' · ' + esc(l.lot_code || '') + (t.operator_id ? ' · ' + esc(t.operator_id) : '') + '</span><span class="mono" style="' + color + '">' + signed(p.type === 'finished' && p.caseLb ? q / p.caseLb : q) + '</span>' + tsLine(txCreated(t), txUpdated(t)) + '</div>';
    }).join('') || '<div class="act"><span></span><span class="muted">No activity in the last 30 days</span><span></span></div>';
  }

  // ---------- inspector ----------
  const ins = $('#inspector');
  // Opening the panel slides the camera left by half the panel width so the focused card
  // stays visible; closing slides it back.
  const panelShift = () => (vpSize().w > 900 ? 230 / cam.s : 0);
  function showInspector(html) {
    ins.innerHTML = html;
    if (!ins.classList.contains('show')) { ins.classList.add('show'); setCam({ cx: cam.cx + panelShift(), cy: cam.cy, s: cam.s }, true); }
    ins.setAttribute('aria-hidden', 'false');
  }
  function closeInspector() {
    if (!selection) return;
    selection = null;
    if (ins.classList.contains('show')) { ins.classList.remove('show'); setCam({ cx: cam.cx - panelShift(), cy: cam.cy, s: cam.s }, true); }
    ins.setAttribute('aria-hidden', 'true');
    renderChrome();
  }
  const insHead = (kicker, id, meta) => '<div class="ins-head"><div style="display:flex;flex-direction:column;gap:6px;min-width:0"><div class="k">' + esc(kicker) + '</div><div class="id">' + esc(id) + '</div><div class="m">' + meta + '</div></div><button class="xbtn" data-close aria-label="Close record">×</button></div>';
  const stat3 = (cells) => '<div class="stat3">' + cells.map(([k, v]) => '<div><div class="k">' + esc(k) + '</div><div class="v">' + v + '</div></div>').join('') + '</div>';
  const footBtns = (btns) => '<div class="ins-foot">' + btns.map(([label, url, primary]) => '<a class="fa-btn' + (primary ? ' fa-primary' : '') + '" href="' + esc(url) + '" target="_blank" rel="noopener">' + esc(label) + '</a>').join('') + '</div>';
  ins.addEventListener('click', (e) => { if (e.target.closest('[data-close]')) closeInspector(); });

  const findTx = (id) => (id && !failed(data.tx) ? data.tx.transactions.find((t) => String(t.id) === String(id)) : null);

  async function openLot(code, pid, pname, txId) {
    if (!code) return;
    selection = { kind: 'lot', label: code, code, pid };
    renderChrome();
    showInspector(insHead('Lot record · ' + (pname || ''), code, 'Loading…') + '<div class="ins-body"><div class="loading">Loading lot record…</div></div>');
    const p = products.find((x) => String(x.id) === String(pid));
    const traceType = p && (p.type === 'ingredient' || p.type === 'packaging') ? 'ingredient' : 'batch';
    const [dR, tR] = await Promise.allSettled([lotDetail(code, pid), API.get('/trace/' + traceType + '/' + encodeURIComponent(code), { product_id: pid })]);
    if (!selection || selection.code !== code) return;
    if (dR.status !== 'fulfilled') { showInspector(insHead('Lot record', code, '') + '<div class="ins-body"><div class="t-red">Couldn’t load this lot (' + esc(dR.reason && dR.reason.message) + ')</div></div>'); return; }
    const d = dR.value, tr = tR.status === 'fulfilled' ? tR.value : null;
    const tl = (d.timeline || []).slice().sort((a, b) => (a.created_at || a.date).localeCompare(b.created_at || b.date));
    const tin = tl.filter((t) => num(t.quantity_lb) > 0).reduce((s, t) => s + num(t.quantity_lb), 0);
    const tout = tl.filter((t) => num(t.quantity_lb) < 0).reduce((s, t) => s - num(t.quantity_lb), 0);
    const first = tl[0];
    const meta = (first ? 'First entry ' + short(first.date) + ' · ' : '') + esc(sourceLabel(d.entry_source)) + (tr && tr.kosher_label ? ' · ' + esc(tr.kosher_label) + ' kosher' : '');
    const cs = d.case_size_lb ? ' <span class="dim" style="font-size:12px">(' + fmt(num(d.on_hand_lbs) / d.case_size_lb) + ' cs)</span>' : '';
    let back = '', fwd = '';
    if (tr) {
      const ing = tr.ingredients || [];
      back = ing.length ? '<div class="trace">' + ing.map((i) => '<span>' + esc(i.ingredient_name || i.product_name) + '</span><span class="mono muted">' + esc(i.lot_code || '') + '</span><span class="mono">' + fmt(i.quantity_lb) + ' lb</span>').join('') + '</div>'
        : '<div class="muted" style="font-size:13px">' + (tr.supplier_lot_code ? 'Supplier lot ' + esc(tr.supplier_lot_code) + (tr.supplier ? ' · ' + esc(tr.supplier) : '') : 'No ingredient lots recorded for this lot.') + '</div>';
      const seen = new Set(), items = [];
      const push = (name, ref, q) => { const k = name + '|' + ref + '|' + q; if (seen.has(k)) return; seen.add(k); items.push([name, ref, q]); };
      for (const b of tr.downstream_batches || []) push(b.batch_product, b.batch_lot_code, b.quantity_consumed);
      for (const b of tr.used_in_batches || []) push(b.batch_product || b.product_name, b.batch_lot_code || b.lot_code, b.quantity_consumed || b.quantity_lb);
      for (const s of [...(tr.customer_shipments || []), ...(tr.direct_shipments || [])]) push('Shipped · ' + (s.customer_name || s.customer || ''), s.order_reference || s.order_number || '', s.quantity_lb || s.total_lb);
      fwd = items.length ? '<div class="trace">' + items.map(([a, b, q]) => '<span>' + esc(a) + '</span><span class="mono muted">' + esc(b || '') + '</span><span class="mono">' + fmt(q) + ' lb</span>').join('') + '</div>' : '<div class="muted" style="font-size:13px">Not used yet.</div>';
    } else {
      back = fwd = '<div class="t-red" style="font-size:13px">Trace couldn’t load</div>';
    }
    const txl = tl.map((t) => '<div class="txrow"><span>' + short(t.date) + ' · ' + esc(TX_LABEL[t.type] || t.type) + (t.customer_name ? ' · ' + esc(t.customer_name) : '') + (t.adjust_reason ? ' · ' + esc(t.adjust_reason) : '') + '</span><span class="mono">' + signed(t.quantity_lb) + ' lb</span>'
      + '<div class="ts">' + tsPart('Created', stampET(t.created_date, t.created_time) || stamp(t.created_at)) + '</div></div>').join('');
    // Lot timeline entries carry their recorded time but not who recorded them.
    const lastTl = tl[tl.length - 1];
    const entry = findTx(txId);
    const hist = history([
      ['Created (first entry)', first ? stampET(first.created_date, first.created_time) || stamp(first.created_at) : null],
      // Known only when the entry you opened is the lot's first entry.
      ['Created by', entry && first && String(first.transaction_id) === String(entry.id) ? entry.operator_id : null],
      ['Last updated (latest entry)', lastTl ? stampET(lastTl.created_date, lastTl.created_time) || stamp(lastTl.created_at) : null],
      ['Updated by', null]
    ], 'A lot has no created/updated fields of its own; these come from its ledger entries, which do not include who recorded them.');
    const entryHist = entry ? history([
      [(TX_LABEL[entry.type] || entry.type) + ' entry', (entry.receipt_number || '#' + entry.id)],
      ['Business date', stampET(entry.business_date)],
      ['Created', txCreated(entry)],
      ['Created by', entry.operator_id || null],
      ['Last updated (correction)', txUpdated(entry) ? txUpdated(entry) + (entry.latest_correction_type ? ' · ' + entry.latest_correction_type : '') : null],
      ['Updated by', null]
    ]).replace('<h4>Record history</h4>', '<h4>Record history · the entry you opened</h4>') : '';
    const adjNote = tl.some((t) => t.type === 'adjust') ? '<div class="warnbox">This lot has adjust entries. Check the reasons below before relying on the on-hand figure.</div>' : '';
    showInspector(insHead('Lot record · ' + (d.product_name || pname || ''), code, meta)
      + '<div class="ins-body">' + stat3([['In', fmt(tin) + ' lb'], ['Out', fmt(tout) + ' lb'], ['On hand', fmt(d.on_hand_lbs) + ' lb' + cs]]) + adjNote
      + '<div class="sect"><h4>Trace back · what went into it</h4>' + back + '</div>'
      + '<div class="sect"><h4>Trace forward · where it went</h4>' + fwd + '</div>'
      + '<div class="sect"><h4>Transactions</h4><div class="txlist">' + (txl || '<div class="muted">None</div>') + '</div></div>' + entryHist + hist + '</div>'
      + footBtns([['Full trace ↗', dashUrl('traceability.html', { v: 4, lot: code, direction: 'forward', product_id: pid })], ['Adjust ↗', taskUrl('adjust')], ['Record count ↗', taskUrl('adjust'), true]]));
  }

  // Order records are cached: the Orders area reads them for created times, the panel reuses them.
  const orderCache = new Map();
  function orderRecord(id) {
    const k = String(id);
    if (!orderCache.has(k)) orderCache.set(k, API.get('/sales/orders/' + encodeURIComponent(k)).catch((e) => { orderCache.delete(k); throw e; }));
    return orderCache.get(k);
  }
  const orderCreated = (o) => stampET(o.created_date, o.created_time);
  let orderStampsLoading = false;
  function loadOrderStamps() {
    if (orderStampsLoading) return;
    orderStampsLoading = true;
    world.querySelectorAll('[data-order-ts]').forEach((el) => {
      orderRecord(el.dataset.orderTs).then((o) => { el.outerHTML = tsLine(orderCreated(o), null); })
        .catch(() => { el.outerHTML = '<div class="ts"><span class="t-red">Record times couldn’t load</span></div>'; });
    });
  }

  async function openOrder(id) {
    selection = { kind: 'order', label: String(id), orderId: id };
    renderChrome();
    showInspector(insHead('Sales order', String(id), 'Loading…') + '<div class="ins-body"><div class="loading">Loading order…</div></div>');
    let o;
    try { o = await orderRecord(id); } catch (e) {
      showInspector(insHead('Sales order', String(id), '') + '<div class="ins-body"><div class="t-red">Couldn’t load (' + esc(e.message) + ')</div></div>'); return;
    }
    if (!selection || String(selection.orderId) !== String(id)) return;
    selection.label = o.order_number; selection.orderId = o.order_id;
    const listRow = failed(data.orders) ? null : data.orders.orders.find((x) => x.order_id === o.order_id);
    renderChrome();
    const reasons = (o.health && o.health.reasons) || [];
    const lines = (o.lines || []).map((l) => '<span>' + esc(l.product) + '</span><span class="mono muted">' + esc(l.unit_count != null ? fmt(l.unit_count) + ' ' + (l.unit || '') : '') + '</span><span class="mono">' + fmt(l.remaining_lb) + ' lb left</span>').join('');
    const ships = (o.shipments || []).map((s) => '<div class="txrow"><span>' + esc(s.product || s.bol_reference || 'Shipment') + '</span><span class="mono">' + fmt(s.total_lb || s.quantity_lb || 0) + ' lb</span>'
      + '<div class="ts">' + tsPart('Shipped', stampET(s.shipped_date, s.shipped_time)) + '</div></div>').join('');
    // Order dates (ordered / ship by) are business dates, kept apart from the record's own times.
    const dates = '<div class="sect"><h4>Dates</h4><div class="txlist hist"><div><span>Order date</span><span>' + esc(stampET(o.order_date) || NOT_RECORDED) + '</span></div><div><span>Requested ship date</span><span>' + esc(stampET(o.requested_ship_date) || NOT_RECORDED) + '</span></div></div></div>';
    const hist = history([
      ['Created', orderCreated(o)],
      ['Created by', null],
      ['Last updated', null],
      ['Updated by', null]
    ].concat(o.state_changed_at ? [['State changed', stamp(o.state_changed_at) + (o.state_changed_by ? ' · ' + o.state_changed_by : '')]] : [])
      .concat(listRow && listRow.ready_at ? [['Marked ready', stamp(listRow.ready_at) + (listRow.ready_by ? ' · ' + listRow.ready_by : '')]] : []),
    'GET /sales/orders/{id} returns a created time but no created-by, updated-at or updated-by.');
    showInspector(insHead('Sales order · ' + o.customer, o.order_number, 'Ship by ' + short(o.requested_ship_date) + ' · ' + esc(o.status) + (o.customer_po ? ' · PO ' + esc(o.customer_po) : ''))
      + '<div class="ins-body">' + stat3([['Ordered', fmt(o.ordered_lb) + ' lb'], ['Shipped', fmt(o.shipped_effective_lb) + ' lb'], ['Remaining', fmt(o.remaining_effective_lb) + ' lb']])
      + (reasons.length ? '<div class="warnbox">' + reasons.map(esc).join('<br>') + '</div>' : '')
      + (num(o.shortage_lb) > 0 ? '<div class="warnbox">Short ' + fmt(o.shortage_lb) + ' lb of stock.</div>' : '')
      + '<div class="sect"><h4>Lines</h4><div class="trace">' + (lines || '<span class="muted">No lines</span>') + '</div></div>'
      + '<div class="sect"><h4>Shipments</h4><div class="txlist">' + (ships || '<div class="muted">None yet</div>') + '</div></div>' + dates + hist + '</div>'
      + footBtns([['Open orders board ↗', dashUrl('index.html', { section: 'orders' })], ['Ship this order ↗', dashUrl('entry.html', { action: 'ship', order: o.order_id }), true]]));
  }

  function openER(id) {
    const e = failed(data.er) ? null : data.er.expected_receipts.find((x) => String(x.id) === String(id));
    if (!e) return;
    selection = { kind: 'er', label: e.reference_number || 'ER ' + e.id };
    renderChrome();
    showInspector(insHead('Expected receipt · ' + e.supplier_name, e.reference_number || 'ER ' + e.id, esc(e.product_name) + (e.expected_date ? ' · expected ' + short(e.expected_date) : ' · no expected date'))
      + '<div class="ins-body">' + stat3([['Expected', fmt(e.expected_qty) + ' lb'], ['Received', fmt(e.received_qty) + ' lb'], ['Remaining', fmt(e.remaining) + ' lb']])
      + (e.is_overdue ? '<div class="warnbox">' + e.days_overdue + ' days overdue.</div>' : '')
      + '<div class="sect"><h4>Details</h4><div class="txlist hist"><div><span>Expected date</span><span>' + esc(stampET(e.expected_date) || 'No date') + '</span></div><div><span>Receipts so far</span><span>' + e.receipt_count + '</span></div><div><span>Last received</span><span>' + esc(stamp(e.last_received_at) || '—') + '</span></div></div>'
      + (e.notes ? '<div class="muted" style="font-size:13px">' + esc(e.notes) + '</div>' : '') + '</div>'
      + history([['Created', stamp(e.created_at)], ['Created by', e.created_by || null], ['Last updated', stamp(e.updated_at)], ['Updated by', null]], 'GET /expected-receipts has no updated-by field.') + '</div>'
      + footBtns([['Expected receipts ↗', dashUrl('index.html', { section: 'expected' })], ['Receive ↗', taskUrl('receive'), true]]));
  }

  // ---------- ingredients: view mode, sorting, relationship map ----------
  const FAM = window.FL_FAMILIES || { approved: false, products: {} };
  const famApproved = () => FAM.approved === true;
  const familyOf = (pid) => (famApproved() && FAM.products[pid] ? FAM.products[pid][0] : null);
  const USE_DAYS = 90;
  let usePromise = null;

  // Ingredient use from posted make entries over the window, split by the batch product made.
  // One request, cached for the page.
  function loadUse() {
    if (!usePromise) {
      const since = addDays(TODAY, -(USE_DAYS - 1));
      usePromise = API.get('/transactions/history', { transaction_type: 'make', since, limit: 1000 }).then((r) => {
        const makes = r.transactions.filter((t) => t.status === 'posted' && t.type === 'make');
        const byIng = new Map();
        for (const t of makes) {
          const out = (t.lines || []).find((l) => num(l.quantity_lb) > 0);
          if (!out) continue;
          for (const l of t.lines) {
            if (l.product_type !== 'ingredient' || num(l.quantity_lb) >= 0) continue;
            let u = byIng.get(l.product_id);
            if (!u) { u = { total: 0, by: new Map() }; byIng.set(l.product_id, u); }
            const q = -num(l.quantity_lb);
            u.total += q;
            const b = u.by.get(out.product_id) || { name: out.product_name, q: 0 };
            b.q += q;
            u.by.set(out.product_id, b);
          }
        }
        const dates = makes.map((t) => t.business_date).sort();
        return { byIng, makes: makes.length, since, truncated: r.transactions.length >= 1000, first: dates[0], last: dates[dates.length - 1] };
      }).catch((e) => { usePromise = null; throw e; });
    }
    return usePromise;
  }

  // Split an ingredient's use by the family of the batch product made. Families come from the
  // approved list only; anything else counts as unclassified.
  function splitOf(x) {
    const r = { coconut: 0, granola: 0, unclassified: 0 };
    for (const [pid, b] of x.by) r[familyOf(pid) || 'unclassified'] += b.q;
    return r;
  }

  // Recipe basis: each approved batch product's formula (one batch), by weight. Formula lines
  // excluded from inventory (sweetening water) are left out. One GET per batch product,
  // queued by the API client (max 4 at a time) and cached for the page.
  let recipePromise = null;
  function loadRecipes() {
    if (!recipePromise) {
      const ids = Object.keys(FAM.products);
      recipePromise = Promise.allSettled(ids.map((id) => API.get('/bom/batches/' + id + '/formula'))).then((rs) => {
        const byIng = new Map();
        let failedN = 0;
        rs.forEach((r, i) => {
          if (r.status !== 'fulfilled') { failedN++; return; }
          const f = r.value, pid = ids[i];
          for (const l of f.ingredients || []) {
            if (l.exclude_from_inventory || !(num(l.quantity_lb) > 0)) continue;
            let u = byIng.get(l.ingredient_product_id);
            if (!u) { u = { total: 0, by: new Map() }; byIng.set(l.ingredient_product_id, u); }
            const q = num(l.quantity_lb);
            u.total += q;
            const b = u.by.get(pid) || { name: f.batch_name, q: 0 };
            b.q += q;
            u.by.set(pid, b);
          }
        });
        if (failedN === ids.length) throw new Error('no recipe could be read');
        return { byIng, recipes: ids.length - failedN, failed: failedN };
      }).catch((e) => { recipePromise = null; throw e; });
    }
    return recipePromise;
  }
  function setMapBasis(b) {
    if (b !== 'recipe' && b !== 'usage') return;
    view.basis = b;
    renderRelMap();
  }

  function setIngredientMode(mode) {
    view.mode.ingredient = mode;
    const tile = nodes.get('t-ingredient');
    if (!tile || !tile.el) return;
    const t2 = tile.el.querySelector('.t2');
    t2.classList.toggle('mapmode', mode === 'map');
    t2.querySelectorAll('[data-view]').forEach((b) => { const on = b.dataset.view === mode; b.classList.toggle('on', on); b.setAttribute('aria-pressed', on); });
    t2.querySelector('.sortwrap').hidden = mode === 'map';
    layoutCards('ingredient');
    if (mode === 'map') renderRelMap();
    scheduleUpdate();
  }
  function setIngredientSort(k) {
    if (!SORTS[k]) return;
    view.sort.ingredient = k;
    const tile = nodes.get('t-ingredient');
    tile.el.querySelectorAll('[data-sort]').forEach((b) => { const on = b.dataset.sort === k; b.classList.toggle('on', on); b.setAttribute('aria-pressed', on); });
    layoutCards('ingredient');
    scheduleUpdate();
  }

  const rmChip = (p, reason) => '<button type="button" class="rm-chip" data-ing="' + p.id + '" title="' + esc(p.name + ' · ' + qtyText(p) + (reason ? ' · ' + reason : '')) + '"><b>' + esc(p.name) + '</b><span>' + qtyText(p) + '</span></button>';
  const winLabel = (u) => short(u.since) + ' – ' + short(TODAY);

  async function renderRelMap() {
    const el = world.querySelector('[data-relmap]');
    if (!el) return;
    const ings = ofType('ingredient').slice().sort(SORTS.qty.fn);
    const basis = famApproved() ? view.basis : 'usage';
    const seq = (renderRelMap.seq = (renderRelMap.seq || 0) + 1);
    el.innerHTML = '<div class="rm-loading">' + (basis === 'recipe' ? 'Loading recipes…' : 'Loading make history…') + '</div>';
    let u;
    try { u = await (basis === 'recipe' ? loadRecipes() : loadUse()); } catch (e) { el.innerHTML = '<div class="rm-loading t-red">Couldn’t load ' + (basis === 'recipe' ? 'recipes' : 'make history') + ' (' + esc(e.message) + ')</div>'; return; }
    if (view.mode.ingredient !== 'map' || seq !== renderRelMap.seq) return;
    const basisSeg = famApproved() ? '<div class="seg rm-basis" role="group">' + [['recipe', 'Recipes'], ['usage', 'Actual use · ' + USE_DAYS + ' days']].map(([v, l]) => '<button type="button" data-basis="' + v + '" aria-pressed="' + (v === basis) + '"' + (v === basis ? ' class="on"' : '') + '>' + l + '</button>').join('') + '</div>' : '';
    const legend = '<div class="rm-legend">' + basisSeg + (basis === 'recipe'
      ? '<div><b>Based on recipes, not actual usage.</b> Position = an ingredient’s share of recipe weight in coconut batch products ÷ coconut + granola batch products, one batch of each recipe (' + u.recipes + ' recipes' + (u.failed ? ', ' + u.failed + ' couldn’t load' : '') + '). Left: coconut-heavy · middle: shared · right: granola-heavy. Labels show on-hand quantity.</div>'
      : '<div><b>Based on actual usage.</b> Position = coconut share of each ingredient’s use in posted makes, last ' + USE_DAYS + ' days (' + winLabel(u) + ', ' + u.makes + ' makes' + (u.truncated ? ', first 1,000 only' : '') + '). Left: used mostly in coconut products · middle: shared · right: mostly in granola. Labels show on-hand quantity.</div>') + '</div>';
    const band = '<div class="rm-band"><span>Coconut-heavy</span><span>Shared</span><span>Granola-heavy</span></div>';
    if (!famApproved()) {
      const counts = Object.values(FAM.products).reduce((m, [f]) => { m[f] = (m[f] || 0) + 1; return m; }, {});
      el.innerHTML = legend + '<div class="rm-plot rm-wait"><div class="rm-notice"><b>Positions are waiting for approval.</b> Factory Ledger has no coconut/granola flag on products, so each batch product needs a family first. Proposed (by product name, ' + esc(FAM.proposedOn || '') + '): '
        + (counts.coconut || 0) + ' coconut and ' + (counts.granola || 0) + ' granola batch products, listed in <span class="mono">canvas/families.js</span>. Nothing is placed until that list is approved.</div></div>'
        + band + '<div class="rm-tray"><div class="rm-tray-head">Unplaced · ' + ings.length + ' ingredients on hand (waiting for the family list)</div><div class="rm-chips">' + ings.map((p) => rmChip(p, 'not placed: family list not approved')).join('') + '</div></div>';
      fitTray(el);
      return;
    }
    const placed = [], tray = [];
    for (const p of ings) {
      const x = u.byIng.get(p.id);
      const f = x ? splitOf(x) : null;
      const known = f ? f.coconut + f.granola : 0;
      if (!x) tray.push([p, basis === 'recipe' ? 'in no recipe' : 'no use in posted makes in the window']);
      else if (!known) tray.push([p, 'used only in products without a family']);
      else placed.push({ p, share: f.coconut / known });
    }
    el.innerHTML = legend + '<div class="rm-plot"><svg class="rm-lines" aria-hidden="true"></svg>'
      + placed.map((n) => '<button type="button" class="rm-node" data-ing="' + n.p.id + '"><b>' + esc(n.p.name) + '</b><span>' + qtyText(n.p) + '</span></button>').join('') + '</div>'
      + band + '<div class="rm-tray"><div class="rm-tray-head">Unplaced · ' + tray.length + '</div><div class="rm-chips">' + (tray.map(([p, r]) => rmChip(p, r)).join('') || '<span class="dim">None</span>') + '</div></div>';
    const noRoom = placeNodes(el, placed);
    if (noRoom.length) {
      const head = el.querySelector('.rm-tray-head'), chips = el.querySelector('.rm-chips');
      head.textContent = 'Unplaced · ' + (tray.length + noRoom.length);
      chips.insertAdjacentHTML('afterbegin', noRoom.map((n) => rmChip(n.p, 'no room on the map at this size (' + Math.round(n.share * 100) + '% coconut)')).join(''));
    }
    fitTray(el);
  }

  // The tray never clips silently: chips that don't fit collapse into "+N more".
  function fitTray(el) {
    const box = el.querySelector('.rm-chips');
    if (!box) return;
    const chips = [...box.querySelectorAll('.rm-chip')];
    let hidden = 0;
    while (box.scrollHeight > box.clientHeight + 1 && chips.length - hidden > 1) {
      chips[chips.length - 1 - hidden].hidden = true;
      hidden++;
      let more = box.querySelector('.rm-more');
      if (!more) { more = document.createElement('span'); more.className = 'rm-more'; box.appendChild(more); }
      more.textContent = '+' + hidden + ' more · see Grid view';
    }
  }

  // Each node sits as close as it can above its exact position on the axis; when it has to
  // move sideways a thin leader line ties it back to its tick. Larger stock first.
  function placeNodes(el, placed) {
    const plot = el.querySelector('.rm-plot');
    const PW = plot.clientWidth, PH = plot.clientHeight, PAD = 12, AX = PH - 10;
    const maxQ = Math.max(1, ...placed.map((n) => n.p.qty));
    const btns = plot.querySelectorAll('.rm-node');
    const noRoom = [];
    placed.forEach((n, i) => { n.el = btns[i]; n.el.style.fontSize = (12 + 6 * Math.sqrt(n.p.qty / maxQ)).toFixed(1) + 'px'; });
    placed.forEach((n) => { n.w = n.el.offsetWidth; n.h = n.el.offsetHeight; n.ax = PAD + (1 - n.share) * (PW - 2 * PAD); });
    const boxes = [];
    const hits = (b) => boxes.some((o) => b.x < o.x + o.w + 6 && o.x < b.x + b.w + 6 && b.y < o.y + o.h + 4 && o.y < b.y + b.h + 4);
    let lines = '';
    for (const n of placed) {
      let best = null;
      for (let k = 0; ; k++) {
        const y = AX - 18 - n.h - k * Math.ceil(n.h / 2);
        if (y < 0) break;
        for (let j = 0; j <= Math.ceil(PW / 8); j++) {
          const dx = (j % 2 ? 1 : -1) * Math.ceil(j / 2) * 16;
          const x = Math.max(0, Math.min(PW - n.w, n.ax - n.w / 2 + dx));
          const b = { x, y, w: n.w, h: n.h };
          if (hits(b)) continue;
          const cost = Math.abs(x + n.w / 2 - n.ax) + k * 30;
          if (!best || cost < best.cost) best = Object.assign(b, { cost });
          break;
        }
      }
      if (!best) { n.el.remove(); noRoom.push(n); continue; }
      boxes.push(best);
      Object.assign(n.el.style, { left: best.x + 'px', top: best.y + 'px' });
      const cx = best.x + n.w / 2;
      lines += '<line class="tick" x1="' + n.ax + '" y1="' + (AX - 6) + '" x2="' + n.ax + '" y2="' + (AX + 6) + '"></line>';
      if (Math.abs(cx - n.ax) > 4 || best.y + n.h < AX - 26) lines += '<line x1="' + n.ax + '" y1="' + (AX - 6) + '" x2="' + Math.max(best.x + 6, Math.min(best.x + n.w - 6, n.ax)) + '" y2="' + (best.y + n.h) + '"></line>';
    }
    const svg = plot.querySelector('.rm-lines');
    svg.setAttribute('width', PW); svg.setAttribute('height', PH);
    svg.innerHTML = '<line class="axis" x1="0" y1="' + AX + '" x2="' + PW + '" y2="' + AX + '"></line>' + lines;
    return noRoom;
  }

  async function openIngredient(pid) {
    const p = products.find((x) => String(x.id) === String(pid));
    if (!p) return;
    selection = { kind: 'ingredient', label: p.name, pid: p.id };
    renderChrome();
    const head = insHead('Ingredient' + (p.sku ? ' · SKU ' + p.sku : ''), p.name, esc(qtyText(p)) + ' on hand · ' + p.lots.length + ' lot' + (p.lots.length === 1 ? '' : 's'));
    const foot = '<div class="ins-foot"><button type="button" class="fa-btn" data-ing-grid="' + p.id + '">Show in grid</button><a class="fa-btn fa-primary" href="' + esc(taskUrl('receive')) + '" target="_blank" rel="noopener">Receive ↗</a></div>';
    showInspector(head + '<div class="ins-body"><div class="loading">Loading use…</div></div>' + foot);
    const [uR, rR] = await Promise.allSettled([loadUse(), famApproved() ? loadRecipes() : Promise.reject(new Error('not approved'))]);
    const u = uR.status === 'fulfilled' ? uR.value : null;
    if (!selection || selection.kind !== 'ingredient' || selection.pid !== p.id) return;
    const x = u && u.byIng.get(p.id);
    const lb = (v) => fmt(v) + ' lb';
    let recipe = '';
    if (rR.status === 'fulfilled') {
      const rx = rR.value.byIng.get(p.id);
      if (!rx) recipe = '<div class="muted" style="font-size:13px">In no recipe.</div>';
      else {
        const f = splitOf(rx), known = f.coconut + f.granola;
        recipe = '<div class="txlist hist"><div><span>Coconut recipes</span><span class="mono">' + lb(f.coconut) + '</span></div><div><span>Granola recipes</span><span class="mono">' + lb(f.granola) + '</span></div>'
          + '<div><span>Coconut share</span><span class="mono">' + (known ? Math.round(f.coconut / known * 100) + '%' : '—') + '</span></div></div>'
          + '<div class="dim" style="font-size:12px">In ' + rx.by.size + ' recipe' + (rx.by.size === 1 ? '' : 's') + ': ' + esc([...rx.by.values()].sort((a, b) => b.q - a.q).map((b) => b.name.replace(/^Batch\s+/i, '') + ' ' + fmt(b.q) + ' lb').join(' · ')) + '</div>';
      }
      recipe = '<div class="sect"><h4>In recipes · per one batch (based on recipes, not actual usage)</h4>' + recipe + '</div>';
    }
    let split;
    if (!u) split = '<div class="t-red" style="font-size:13px">Make history couldn’t load.</div>';
    else if (!x) split = '<div class="muted" style="font-size:13px">Not used in posted makes in the last ' + USE_DAYS + ' days.</div>';
    else if (!famApproved()) split = '<div class="muted" style="font-size:13px">The coconut / granola split appears once the product-family list is approved.</div>';
    else {
      const f = splitOf(x), known = f.coconut + f.granola;
      split = '<div class="txlist hist"><div><span>Coconut products</span><span class="mono">' + lb(f.coconut) + '</span></div><div><span>Granola products</span><span class="mono">' + lb(f.granola) + '</span></div><div><span>Products without a family</span><span class="mono">' + lb(f.unclassified) + '</span></div>'
        + '<div><span>Coconut share</span><span class="mono">' + (known ? Math.round(f.coconut / known * 100) + '%' : '—') + '</span></div></div>';
    }
    const by = x ? [...x.by.values()].sort((a, b) => b.q - a.q).slice(0, 8).map((b) => '<div><span>' + esc(b.name) + '</span><span class="mono">' + lb(b.q) + '</span></div>').join('') : '';
    showInspector(head + '<div class="ins-body">' + stat3([['On hand', esc(qtyText(p))], ['Lots', p.lots.length], ['Used · ' + USE_DAYS + ' days', x ? lb(x.total) : '—']])
      + recipe + '<div class="sect"><h4>Actual use by product family · posted makes, ' + (u ? winLabel(u) : 'last ' + USE_DAYS + ' days') + '</h4>' + split + '</div>'
      + (by ? '<div class="sect"><h4>Used in</h4><div class="txlist">' + by + '</div></div>' : '') + '</div>' + foot);
  }

  ins.addEventListener('click', (e) => {
    const b = e.target.closest('[data-ing-grid]');
    if (!b) return;
    const id = 'p' + b.dataset.ingGrid;
    closeInspector();
    if (view.mode.ingredient !== 'grid') setIngredientMode('grid');
    flyTo(id);
  });

  // ---------- theme ----------
  const THEMES = ['industrial', 'warm', 'blueprint'];
  function applyTheme(t) {
    if (!THEMES.includes(t)) t = 'industrial';
    document.documentElement.setAttribute('data-theme', t);
    document.querySelectorAll('[data-theme-pick]').forEach((b) => b.setAttribute('aria-checked', b.dataset.themePick === t));
  }
  document.querySelectorAll('[data-theme-pick]').forEach((b) => {
    b.onclick = () => { applyTheme(b.dataset.themePick); try { localStorage.setItem('flcanvas.theme', b.dataset.themePick); } catch (e) { /* not remembered */ } };
  });
  applyTheme(document.documentElement.getAttribute('data-theme'));

  // ---------- tasks ----------
  function taskUrl(action) {
    const params = { action };
    if (action === 'ship' && selection && selection.kind === 'order' && selection.orderId) params.order = selection.orderId;
    return dashUrl('entry.html', params);
  }
  function buildTasks() {
    const el = $('#tasks');
    [['Receive', 'receive'], ['Make', 'make'], ['Pack', 'pack'], ['Ship', 'ship'], ['Count', 'adjust']].forEach(([label, action]) => {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'fa-btn'; b.textContent = label;
      b.title = action === 'adjust' ? 'Opens the dashboard entry page on Adjust (there is no separate count page; pick reason Physical count)' : 'Opens the dashboard entry page: ' + label;
      b.onclick = () => openTab(taskUrl(action));
      el.appendChild(b);
    });
  }

  // ---------- search ----------
  let searchIndex = [];
  function buildIndex() {
    searchIndex = [];
    for (const [id, a] of Object.entries(AREAS)) searchIndex.push({ label: a.name, kind: 'Area', go: () => flyTo(id) });
    for (const [t, tile] of Object.entries(TILES)) searchIndex.push({ label: tile.name, kind: 'Inventory category', go: () => flyTo('t-' + t) });
    for (const p of products) {
      searchIndex.push({ label: p.name, kind: (TILES[p.type] ? TILES[p.type].name : p.type) + ' · ' + qtyText(p), extra: p.sku, go: () => { closeInspector(); flyTo('p' + p.id); } });
      for (const l of p.lots) searchIndex.push({ label: l.code, kind: 'Lot · ' + p.name, go: () => { flyTo('p' + p.id); openLot(l.code, p.id, p.name); } });
    }
    if (!failed(data.orders)) for (const o of data.orders.orders) searchIndex.push({ label: o.order_number + ' · ' + o.customer, kind: 'Order', extra: o.customer_po, go: () => { flyTo('shipping'); openOrder(o.order_id); } });
    if (!failed(data.er)) for (const e of data.er.expected_receipts) searchIndex.push({ label: e.product_name + ' · ' + (e.reference_number || 'ER ' + e.id), kind: 'Expected receipt', go: () => { flyTo('receiving'); openER(e.id); } });
  }

  const qEl = $('#q'), resEl = $('#results');
  let results = [], active = 0, searchSeq = 0;
  function renderResults() {
    resEl.innerHTML = results.length ? '' : '<div class="empty">No matches on the canvas</div>';
    results.forEach((r, i) => {
      const b = document.createElement('button');
      b.type = 'button'; if (i === active) b.className = 'active';
      b.innerHTML = '<span>' + esc(r.label) + '</span><span>' + esc(r.kind) + '</span>';
      b.onmousedown = (e) => { e.preventDefault(); pick(r); };
      resEl.appendChild(b);
    });
    resEl.hidden = false;
  }
  function pick(r) { resEl.hidden = true; qEl.blur(); r.go(); }
  async function runSearch() {
    const q = qEl.value.trim().toLowerCase();
    const seq = ++searchSeq;
    const local = q ? searchIndex.filter((r) => (r.label + ' ' + r.kind + ' ' + (r.extra || '')).toLowerCase().includes(q)) : searchIndex.filter((r) => r.kind === 'Area' || r.kind === 'Inventory category');
    local.sort((a, b) => (a.label.toLowerCase().startsWith(q) ? 0 : 1) - (b.label.toLowerCase().startsWith(q) ? 0 : 1));
    results = local.slice(0, 9); active = 0; renderResults();
    if (q.length < 3) return;
    // Server search finds lots and orders that are not on hand / not open.
    try {
      const s = await API.get('/dashboard/api/search', { q });
      if (seq !== searchSeq) return;
      const have = new Set(results.map((r) => r.label));
      const extra = [];
      for (const l of s.lots || []) {
        const code = l.lot_code || l.code; if (!code || have.has(code)) continue;
        extra.push({ label: code, kind: 'Lot · ' + (l.product_name || '') + ' (record only)', go: () => openLot(code, l.product_id, l.product_name) });
      }
      for (const o of s.orders || []) {
        const lbl = (o.order_number || '') + ' · ' + (o.customer || o.customer_name || '');
        if (have.has(lbl)) continue;
        extra.push({ label: lbl, kind: 'Order (record only)', go: () => openOrder(o.order_id || o.id || o.order_number) });
      }
      if (extra.length) { results = results.concat(extra).slice(0, 12); renderResults(); }
    } catch (e) { /* local results are enough */ }
  }
  let st = 0;
  qEl.addEventListener('input', () => { clearTimeout(st); st = setTimeout(runSearch, 160); });
  qEl.addEventListener('focus', runSearch);
  qEl.addEventListener('blur', () => setTimeout(() => { resEl.hidden = true; }, 150));
  qEl.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') { active = Math.min(results.length - 1, active + 1); renderResults(); e.preventDefault(); }
    else if (e.key === 'ArrowUp') { active = Math.max(0, active - 1); renderResults(); e.preventDefault(); }
    else if (e.key === 'Enter' && results[active]) pick(results[active]);
    else if (e.key === 'Escape') { resEl.hidden = true; qEl.blur(); }
  });

  // ---------- input: wheel, drag, click ----------
  vp.addEventListener('wheel', (e) => {
    e.preventDefault();
    const r = vp.getBoundingClientRect();
    const px = e.clientX - r.left - r.width / 2, py = e.clientY - r.top - r.height / 2;
    const dy = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
    const s = cam.s * Math.exp(-dy * 0.0018);
    const L = limits();
    const ns = Math.max(L.min, Math.min(L.max, s));
    const wx = cam.cx + px / cam.s, wy = cam.cy + py / cam.s;
    setCam({ s: ns, cx: wx - px / ns, cy: wy - py / ns }, false);
  }, { passive: false });

  let down = null, dragged = false;
  vp.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return;
    down = { x: e.clientX, y: e.clientY, cam: Object.assign({}, cam), id: e.pointerId };
    dragged = false;
  });
  vp.addEventListener('pointermove', (e) => {
    if (!down) return;
    const dx = e.clientX - down.x, dy = e.clientY - down.y;
    if (!dragged && Math.abs(dx) + Math.abs(dy) < 5) return;
    if (!dragged) { dragged = true; vp.classList.add('dragging'); try { vp.setPointerCapture(down.id); } catch (_) {} }
    setCam({ s: down.cam.s, cx: down.cam.cx - dx / down.cam.s, cy: down.cam.cy - dy / down.cam.s }, false);
  });
  const endDrag = () => { if (!down) return; down = null; vp.classList.remove('dragging'); setTimeout(() => { dragged = false; }, 0); };
  vp.addEventListener('pointerup', endDrag);
  vp.addEventListener('pointercancel', endDrag);
  vp.addEventListener('dragstart', (e) => e.preventDefault());
  vp.addEventListener('click', (e) => { if (dragged) { e.stopPropagation(); e.preventDefault(); } }, true);

  world.addEventListener('click', (e) => {
    const t = e.target.closest('[data-view],[data-sort],[data-basis],[data-ing],[data-fly],[data-lot],[data-order],[data-er]');
    if (!t) return;
    if (t.dataset.view) setIngredientMode(t.dataset.view);
    else if (t.dataset.basis) setMapBasis(t.dataset.basis);
    else if (t.dataset.sort) setIngredientSort(t.dataset.sort);
    else if (t.dataset.ing) openIngredient(t.dataset.ing);
    else if (t.dataset.fly) { closeInspector(); flyTo(t.dataset.fly); }
    else if (t.dataset.lot) openLot(t.dataset.lot, t.dataset.pid, t.dataset.pname, t.dataset.tx);
    else if (t.dataset.order) openOrder(t.dataset.order);
    else if (t.dataset.er) openER(t.dataset.er);
  });

  function zoomOut() {
    if (selection) { closeInspector(); return; }
    const id = focus.product ? focus.tile : focus.tile ? focus.area : null;
    if (id) flyTo(id); else goHome();
  }
  $('#home').onclick = goHome;
  $('#zout').onclick = zoomOut;
  $('#zin').onclick = () => setCam({ cx: cam.cx, cy: cam.cy, s: cam.s * 1.8 }, true);
  document.addEventListener('keydown', (e) => {
    if (e.target.closest && e.target.closest('input')) return;
    if (e.key === 'Escape') zoomOut();
    else if (e.key === '+' || e.key === '=') $('#zin').click();
    else if (e.key === '-') zoomOut();
    else if (e.key === '/') { e.preventDefault(); qEl.focus(); }
  });
  window.addEventListener('resize', () => setCam(cam, false));

  // ---------- start ----------
  async function start() {
    const env = $('#env');
    env.textContent = CFG.env;
    env.classList.toggle('production', CFG.env === 'production');
    buildTasks();
    banner('Loading live ' + CFG.env + ' data…');
    try { await loadAll(); } catch (e) { banner('Couldn’t load data: ' + e.message, true); return; }
    if (failed(data.inv)) { banner('Couldn’t load inventory from ' + API.base + ' (' + data.inv.error + ')', true); }
    else banner('');
    buildProducts();
    // Card heights are measured from rendered text, so wait for the web fonts.
    try { await document.fonts.ready; } catch (e) { /* measure with fallback fonts */ }
    buildWorld();
    buildIndex();
    setCam(fit(BOX, 0.95), false);
  }
  start();
})();
