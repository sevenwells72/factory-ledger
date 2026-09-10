// Browser-side mechanical checks. Everything in CHECK_SOURCE runs inside the
// page; the Node side only reads the results back.
//
// Scope of what is and is not checked is deliberate. Each rule below has a
// clause that a rendered page can settle without judgement — a measured size,
// a computed ratio, a scroll width, an occlusion test, a shift score. The
// clauses that need a human (hover affordance, Spanish labels, floor lighting)
// are not touched here and are not reported as passes.
//
// Two groups live here: the five original rules (TOUCH-003, ACCESS-008,
// LAYOUT-003/ACCESS-001, LAYOUT-011, LAYOUT-020) and the eight machine-checkable
// clauses of category 17, Status & Data Display (STATUS-002, -004, -005, -006,
// -007, -008, -010, -011), at the bottom of CHECK_SOURCE.

export const CHECK_SOURCE = `
(() => {
  const MIN_HIT_PT = 44;

  const INTERACTIVE = [
    'a[href]', 'button', 'input:not([type="hidden"])', 'select', 'textarea', 'summary',
    '[role="button"]', '[role="tab"]', '[role="link"]', '[role="checkbox"]',
    '[onclick]', '[tabindex]:not([tabindex="-1"])',
    // Interaction attached to non-controls (SYS-3). These carry click handlers
    // in dashboard.js with no role, so a role-based selector alone misses them.
    '.lot-link', '.order-row', '.search-item', '.er-product-option', '.product-lot-row',
    'tr.expandable', '.supply-item-row', '.day-card-trigger', '.disambig-btn',
    '.collapsible-header', '.attention-chip', '.notes-filter-btn', '.note-checkbox',
    '.dir-btn', '.trace-btn', '.export-btn', '.graph-ctrl-btn', '.recent-lot',
    '.cell', '.copyday', '.more-toggle', '.otbtn', '.act', '.tab', '.supplies-subtab'
  ].join(',');

  function rectOf(el) { return el.getBoundingClientRect(); }

  function isVisible(el) {
    const r = rectOf(el);
    if (r.width <= 0 || r.height <= 0) return false;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden') return false;
    if (Number(cs.opacity) === 0) return false;
    // Screen-reader-only text is not a rendering failure.
    if (cs.clipPath === 'inset(50%)' || (cs.clip && cs.clip !== 'auto' && r.width <= 1)) return false;
    let p = el.parentElement;
    while (p) {
      const pcs = getComputedStyle(p);
      if (pcs.display === 'none' || pcs.visibility === 'hidden' || Number(pcs.opacity) === 0) return false;
      p = p.parentElement;
    }
    return true;
  }

  function pathOf(el) {
    const bits = [];
    let node = el;
    for (let i = 0; node && i < 4; i++) {
      let seg = node.tagName.toLowerCase();
      if (node.id) { seg += '#' + node.id; bits.unshift(seg); break; }
      const cls = (node.getAttribute('class') || '').trim().split(/\\s+/).filter(Boolean).slice(0, 2);
      if (cls.length) seg += '.' + cls.join('.');
      bits.unshift(seg);
      node = node.parentElement;
    }
    return bits.join(' > ');
  }

  function labelOf(el) {
    const text = (el.getAttribute('aria-label') || el.textContent || '').replace(/\\s+/g, ' ').trim();
    return text.length > 48 ? text.slice(0, 45) + '…' : text;
  }

  function root(sel) {
    if (!sel) return document.body;
    return document.querySelector(sel) || document.body;
  }

  function inRoot(el, rootEl) {
    return rootEl === document.body ? true : rootEl.contains(el);
  }

  function collect(sel, rootEl) {
    const all = Array.from(document.querySelectorAll(sel));
    return all.filter(el => inRoot(el, rootEl) && isVisible(el));
  }

  // ── TOUCH-003 · hit region ≥ 44 pt ────────────────────────────────────────
  // The region a finger can land on is not always the element's own box. Two
  // things the browser hit-tests as part of a control are invisible to
  // getBoundingClientRect() and are unioned in here:
  //   1. an absolutely-positioned ::before / ::after the control generates. A
  //      pointer event on generated content is dispatched to the originating
  //      element, which is how dashboard.css extends a row-action control's
  //      hit region without growing the row (IMP-011's "::before pseudo-element
  //      expanding the hit area"). Only an out-of-flow box can lie outside the
  //      element; an in-flow one is already inside the measured rect. A pseudo
  //      with pointer-events: none is not hit-tested and is not counted.
  //   2. a <label> that wraps a checkbox or radio. Activating a label activates
  //      its control (HTML §4.10.4), so the label is the hit region —
  //      07-systemic-clusters T4: "the wrapping label is what actually carries
  //      the hit region". Only a wrapping label is counted, because its box is
  //      one contiguous region that contains the control; a detached
  //      label[for] elsewhere on the page would make the union a bounding box
  //      of two separate regions. Labels of other control types are not
  //      counted: a label on a text field only focuses it.
  // Both are measured from geometry, not trusted from a flag. Where the two
  // differ, the element's own box is recorded (boxW, boxH) beside the hit
  // region (w, h) so they can be told apart in results.json.
  function pseudoRect(el, r, which) {
    const ps = getComputedStyle(el, which);
    if (!ps || ps.content === 'none' || ps.content === 'normal' || ps.display === 'none') return null;
    if (ps.position !== 'absolute') return null;
    if (ps.pointerEvents === 'none' || ps.visibility === 'hidden') return null;
    const cs = getComputedStyle(el);
    // The containing block of an absolutely-positioned pseudo is the nearest
    // positioned ancestor. Only the case where that is the element itself is
    // measured; anything else is not a hit-region extension of this control.
    if (cs.position === 'static') return null;
    const top = parseFloat(ps.top), left = parseFloat(ps.left);
    const w = parseFloat(ps.width), h = parseFloat(ps.height);
    if (![top, left, w, h].every(Number.isFinite)) return null;
    const x = r.left + parseFloat(cs.borderLeftWidth) + left;
    const y = r.top + parseFloat(cs.borderTopWidth) + top;
    return { left: x, top: y, right: x + w, bottom: y + h };
  }

  function unionRect(a, b) {
    if (!b) return a;
    return {
      left: Math.min(a.left, b.left), top: Math.min(a.top, b.top),
      right: Math.max(a.right, b.right), bottom: Math.max(a.bottom, b.bottom),
    };
  }

  function ownHitRect(el) {
    const r = rectOf(el);
    let hit = { left: r.left, top: r.top, right: r.right, bottom: r.bottom };
    hit = unionRect(hit, pseudoRect(el, r, '::before'));
    hit = unionRect(hit, pseudoRect(el, r, '::after'));
    return hit;
  }

  function hitRegion(el) {
    const r = rectOf(el);
    const grew = (a, b) => (b.right - b.left) > (a.right - a.left) + 0.5 || (b.bottom - b.top) > (a.bottom - a.top) + 0.5;
    const own = { left: r.left, top: r.top, right: r.right, bottom: r.bottom };
    let hit = ownHitRect(el);
    const via = [];
    if (grew(own, hit)) via.push('pseudo');
    const type = el.tagName === 'INPUT' ? (el.getAttribute('type') || 'text').toLowerCase() : '';
    if ((type === 'checkbox' || type === 'radio') && el.labels) {
      const before = hit;
      for (const label of el.labels) {
        if (!label.contains(el) || !isVisible(label)) continue;
        hit = unionRect(hit, ownHitRect(label));
      }
      if (grew(before, hit)) via.push('label');
    }
    return { hit, via: via.join('+') };
  }

  function touchTargets(rootSel) {
    const rootEl = root(rootSel);
    const els = collect(INTERACTIVE, rootEl);
    const seen = new Set();
    const items = [];
    const extended = [];
    for (const el of els) {
      if (seen.has(el)) continue;
      seen.add(el);
      if (el.tagName === 'OPTION' || el.closest('select')) continue;
      const r = rectOf(el);
      const boxW = Math.round(r.width * 10) / 10;
      const boxH = Math.round(r.height * 10) / 10;
      const { hit, via } = hitRegion(el);
      const w = Math.round((hit.right - hit.left) * 10) / 10;
      const h = Math.round((hit.bottom - hit.top) * 10) / 10;
      // A larger interactive ancestor means the small glyph sits inside a big
      // target. Recorded, but it does not excuse the element: the two carry
      // different actions wherever both have their own handler.
      let nested = false;
      let anc = el.parentElement;
      while (anc && anc !== rootEl.parentElement) {
        if (anc.matches && anc.matches(INTERACTIVE)) {
          const ar = rectOf(anc);
          if (ar.width >= MIN_HIT_PT && ar.height >= MIN_HIT_PT) { nested = true; break; }
        }
        anc = anc.parentElement;
      }
      const fails = w < MIN_HIT_PT || h < MIN_HIT_PT;
      if (fails) {
        // tag and the full class list are recorded because pathOf() drops the
        // class once an element has an id, which is exactly what makes a
        // failure hard to attribute to the CSS rule that sized it.
        const item = { path: pathOf(el), label: labelOf(el), w, h, nested, tag: el.tagName.toLowerCase(), cls: (el.getAttribute('class') || '').trim().slice(0, 120) };
        if (via) { item.boxW = boxW; item.boxH = boxH; item.hitVia = via; }
        items.push(item);
      } else if (via && (boxW < MIN_HIT_PT || boxH < MIN_HIT_PT)) {
        extended.push({ path: pathOf(el), label: labelOf(el), boxW, boxH, w, h, hitVia: via });
      }
    }
    items.sort((a, b) => (a.w * a.h) - (b.w * b.h));
    return {
      checked: els.length,
      failures: items.length,
      failuresNotNested: items.filter(i => !i.nested).length,
      // Targets whose own box is under 44pt but whose hit region — a
      // generated box or an associated label — is not. Counted so a pass
      // earned that way is visible in results.json rather than silent.
      passedViaExtension: extended,
      // Every failing measurement, not the twelve worst. The cap that used to
      // sit here dropped 2,910 of 4,968 TOUCH-003 measurements, which made any
      // arithmetic over results.json — cluster footprints, cells-cleared,
      // before/after counts — wrong. report.mjs already caps its own tables.
      worst: items,
    };
  }

  // ── ACCESS-008 · WCAG AA contrast ─────────────────────────────────────────
  function parseColor(value) {
    const m = String(value).match(/rgba?\\(([^)]+)\\)/);
    if (!m) return null;
    const parts = m[1].split(/[\\s,\\/]+/).filter(Boolean).map(Number);
    return { r: parts[0], g: parts[1], b: parts[2], a: parts.length > 3 ? parts[3] : 1 };
  }

  function over(fg, bg) {
    const a = fg.a + bg.a * (1 - fg.a);
    if (a === 0) return { r: 0, g: 0, b: 0, a: 0 };
    return {
      r: (fg.r * fg.a + bg.r * bg.a * (1 - fg.a)) / a,
      g: (fg.g * fg.a + bg.g * bg.a * (1 - fg.a)) / a,
      b: (fg.b * fg.a + bg.b * bg.a * (1 - fg.a)) / a,
      a,
    };
  }

  function luminance(c) {
    const f = v => {
      const s = v / 255;
      return s <= 0.03928 ? s / 12.92 : Math.pow((s + 0.055) / 1.055, 2.4);
    };
    return 0.2126 * f(c.r) + 0.7152 * f(c.g) + 0.0722 * f(c.b);
  }

  function ratio(a, b) {
    const la = luminance(a), lb = luminance(b);
    const hi = Math.max(la, lb), lo = Math.min(la, lb);
    return (hi + 0.05) / (lo + 0.05);
  }

  // Composite every translucent fill between the element and the first opaque
  // surface. This is the clause the code audit could not settle: the declared
  // hex of .so-ready-pill and .readiness-chip.severity-* is not the colour the
  // eye receives.
  function backdropOf(el) {
    const layers = [];
    let node = el;
    let imaged = false;
    while (node && node !== document.documentElement.parentElement) {
      const cs = getComputedStyle(node);
      if (cs.backgroundImage && cs.backgroundImage !== 'none') imaged = true;
      const c = parseColor(cs.backgroundColor);
      if (c && c.a > 0) {
        layers.push(c);
        if (c.a >= 0.999) break;
      }
      node = node.parentElement;
    }
    let base = layers.length && layers[layers.length - 1].a >= 0.999
      ? layers.pop()
      : { r: 255, g: 255, b: 255, a: 1 };
    for (let i = layers.length - 1; i >= 0; i--) base = over(layers[i], base);
    return { color: base, imaged };
  }

  function effectiveOpacity(el) {
    let o = 1, node = el;
    while (node && node !== document.documentElement.parentElement) {
      const v = Number(getComputedStyle(node).opacity);
      if (Number.isFinite(v)) o *= v;
      node = node.parentElement;
    }
    return o;
  }

  function hasOwnText(el) {
    for (const n of el.childNodes) {
      if (n.nodeType === 3 && n.textContent.trim().length) return true;
    }
    return false;
  }

  function contrast(rootSel) {
    const rootEl = root(rootSel);
    const scope = rootEl === document.body ? document.body : rootEl;
    const els = Array.from(scope.querySelectorAll('*')).filter(el => hasOwnText(el) && isVisible(el));
    // Placeholder text is rendered text and is governed by the same rule; it
    // is also INPUT-022's clause, which no code read could settle because the
    // colour comes from the user agent wherever no ::placeholder rule exists.
    const placeholders = Array.from(scope.querySelectorAll('input[placeholder], textarea[placeholder]'))
      .filter(el => isVisible(el) && el.value === '' && el.getAttribute('placeholder').trim());
    const failures = [];
    const indeterminate = [];
    let checked = 0;
    for (const el of placeholders) {
      const pseudo = getComputedStyle(el, '::placeholder');
      const fg = parseColor(pseudo.color) || parseColor(getComputedStyle(el).color);
      if (!fg) continue;
      const { color: bg } = backdropOf(el);
      const size = parseFloat(pseudo.fontSize || getComputedStyle(el).fontSize) || 16;
      const weight = Number(pseudo.fontWeight || getComputedStyle(el).fontWeight) || 400;
      const large = size >= 24 || (size >= 18.66 && weight >= 700);
      const target = large ? 3 : 4.5;
      const opacity = effectiveOpacity(el) * (Number(pseudo.opacity) || 1);
      const composed = over({ ...fg, a: fg.a * opacity }, bg);
      const r = Math.round(ratio(composed, bg) * 100) / 100;
      checked++;
      if (r + 0.005 < target) {
        failures.push({
          path: pathOf(el) + '::placeholder',
          text: el.getAttribute('placeholder').slice(0, 44),
          ratio: r, target, fontPx: Math.round(size * 10) / 10, weight,
          fg: pseudo.color, bg: 'rgb(' + [bg.r, bg.g, bg.b].map(Math.round).join(', ') + ')',
          opacity: Math.round(opacity * 100) / 100,
          placeholder: true,
        });
      }
    }
    for (const el of els) {
      const cs = getComputedStyle(el);
      const fg = parseColor(cs.color);
      if (!fg) continue;
      const { color: bg, imaged } = backdropOf(el);
      const size = parseFloat(cs.fontSize) || 16;
      const weight = Number(cs.fontWeight) || 400;
      const large = size >= 24 || (size >= 18.66 && weight >= 700);
      const target = large ? 3 : 4.5;
      const opacity = effectiveOpacity(el);
      const composedFg = over({ ...fg, a: fg.a * opacity }, bg);
      const r = Math.round(ratio(composedFg, bg) * 100) / 100;
      checked++;
      const record = {
        path: pathOf(el), text: labelOf(el), ratio: r, target,
        fontPx: Math.round(size * 10) / 10, weight,
        fg: cs.color, bg: 'rgb(' + [bg.r, bg.g, bg.b].map(Math.round).join(', ') + ')',
        opacity: Math.round(opacity * 100) / 100,
      };
      if (imaged) { indeterminate.push(record); continue; }
      if (r + 0.005 < target) failures.push(record);
    }
    failures.sort((a, b) => a.ratio - b.ratio);
    return {
      checked,
      placeholdersChecked: placeholders.length,
      placeholderFailures: failures.filter(f => f.placeholder).length,
      failures: failures.length,
      indeterminate: indeterminate.length,
      // Every failing measurement — see the note in touchTargets(). The cap
      // here dropped 1,483 of 3,418 ACCESS-008 measurements.
      worst: failures,
    };
  }

  // ── LAYOUT-003 / ACCESS-001 · no horizontal overflow ──────────────────────
  function horizontalOverflow() {
    const doc = document.documentElement;
    const overflowPx = Math.round((doc.scrollWidth - doc.clientWidth) * 10) / 10;
    const limit = doc.clientWidth;
    const offenders = [];
    if (overflowPx > 1) {
      for (const el of Array.from(document.body.querySelectorAll('*'))) {
        if (!isVisible(el)) continue;
        const r = rectOf(el);
        if (r.right <= limit + 1 && r.left >= -1) continue;
        // Content wider than its own scroll container is that container's
        // business, not the document's — .table-scroll et al. are legitimate.
        let scroller = el.parentElement, contained = false;
        while (scroller && scroller !== document.body) {
          const cs = getComputedStyle(scroller);
          if ((cs.overflowX === 'auto' || cs.overflowX === 'scroll') &&
              scroller.scrollWidth > scroller.clientWidth + 1) { contained = true; break; }
          scroller = scroller.parentElement;
        }
        if (contained) continue;
        offenders.push({
          path: pathOf(el), text: labelOf(el),
          right: Math.round(r.right), left: Math.round(r.left),
          width: Math.round(r.width), viewport: limit,
        });
      }
    }
    // Deepest-first, then keep only the outermost distinct offenders.
    const pruned = offenders.filter(o => !offenders.some(p => p !== o && o.path.startsWith(p.path + ' >')));
    return { overflowPx, viewport: limit, offenders: pruned.slice(0, 10), offenderCount: pruned.length };
  }

  // ── LAYOUT-011 · fixed bars must not cover the last row ───────────────────
  function fixedBars() {
    return Array.from(document.body.querySelectorAll('*')).filter(el => {
      const cs = getComputedStyle(el);
      if (cs.position !== 'fixed' && cs.position !== 'sticky') return false;
      if (!isVisible(el)) return false;
      const r = rectOf(el);
      return r.width > 40 && r.height > 8;
    });
  }

  // The rule is about the END of a scroll region: "scrollable regions include
  // end insets so the last items scroll fully into view above those bars." A
  // middle row passing under a top sticky header is not a failure — scrolling
  // up reveals it. Two things are failures, and only these are reported:
  //   1. anything occluded by a bar anchored to the bottom of the viewport at
  //      maximum scroll (there is no scroll left to lift it clear), and
  //   2. the last actionable item of a scroll region still occluded, or cut
  //      off by the viewport, once that region is scrolled to its end.
  function occlusionAtBottom(rootSel) {
    const rootEl = root(rootSel);
    const scrollers = Array.from(document.querySelectorAll('*')).filter(el => {
      const cs = getComputedStyle(el);
      // > 8px, not > 1px: a horizontal scrollbar inside an overflow-x wrapper
      // makes scrollHeight exceed clientHeight by a pixel or two, which is not
      // a vertical scroll region.
      return (cs.overflowY === 'auto' || cs.overflowY === 'scroll') && el.scrollHeight > el.clientHeight + 8;
    });
    for (const s of scrollers) s.scrollTop = s.scrollHeight;
    const docScrolls = document.documentElement.scrollHeight > document.documentElement.clientHeight + 1;
    window.scrollTo(0, document.documentElement.scrollHeight);

    const bars = fixedBars();
    const barInfo = bars.map(b => {
      const r = rectOf(b);
      const anchor = r.bottom >= window.innerHeight - 2 ? 'bottom'
        : r.top <= 2 ? 'top' : 'floating';
      return { path: pathOf(b), anchor, top: Math.round(r.top), bottom: Math.round(r.bottom), height: Math.round(r.height) };
    });
    const bottomBars = bars.filter((b, i) => barInfo[i].anchor === 'bottom');

    function blocker(el, barSet) {
      const r = rectOf(el);
      if (r.bottom <= 0 || r.top >= window.innerHeight) return null;
      const probes = [
        [r.left + Math.min(r.width / 2, 40), r.top + Math.min(r.height / 2, 12)],
        [r.left + 4, r.top + 4],
        [r.right - 4, r.bottom - 4],
      ];
      for (const [x, y] of probes) {
        if (x < 0 || y < 0 || x > window.innerWidth || y > window.innerHeight) continue;
        const hit = document.elementFromPoint(x, y);
        if (!hit || hit === el || el.contains(hit) || hit.contains(el)) continue;
        const bar = barSet.find(b => b === hit || b.contains(hit));
        if (bar && !bar.contains(el)) return pathOf(bar);
      }
      return null;
    }

    // A top bar covering something at maximum scroll is not a failure: one
    // scroll up moves it clear. Reachability is settled by trying — park the
    // element at 60% of the viewport and probe again.
    function isReachable(el, barSet) {
      const before = window.scrollY;
      const r = rectOf(el);
      const targetY = window.scrollY + r.top - window.innerHeight * 0.6;
      window.scrollTo(0, Math.max(0, targetY));
      const still = blocker(el, barSet);
      window.scrollTo(0, before);
      return !still;
    }

    const targets = collect(INTERACTIVE, rootEl);
    const covered = [];

    // (1) anything under a bottom-anchored bar at maximum scroll
    if (bottomBars.length) {
      for (const el of targets) {
        const by = blocker(el, bottomBars);
        if (by) covered.push({ path: pathOf(el), label: labelOf(el), blockedBy: by, why: 'under a bottom-anchored bar at maximum scroll' });
      }
    }

    // (2) the last actionable item of each scroll region, once at its end
    const regions = [{ el: null, targets }].concat(scrollers.map(s => ({
      el: s,
      targets: targets.filter(t => s.contains(t)),
    })));
    for (const region of regions) {
      const list = region.targets;
      if (!list.length) continue;
      const last = list[list.length - 1];
      const r = rectOf(last);
      const by = blocker(last, bars);
      const clipped = region.el
        ? r.bottom > rectOf(region.el).bottom + 1
        : (docScrolls && r.bottom > window.innerHeight + 1);
      if (clipped) {
        covered.push({
          path: pathOf(last), label: labelOf(last),
          blockedBy: '(clipped by its own scroll region)',
          why: 'last item of a scroll region does not scroll fully into view',
        });
      } else if (by && !isReachable(last, bars)) {
        covered.push({
          path: pathOf(last), label: labelOf(last), blockedBy: by,
          why: 'last item of a scroll region stays behind a fixed bar at every scroll position',
        });
      }
    }

    // Informational only: rows that pass under a top bar mid-scroll. Normal
    // sticky-header behaviour, recorded so the pass is not mistaken for
    // "no bar overlaps anything, ever".
    let passUnderTop = 0;
    const topBars = bars.filter((b, i) => barInfo[i].anchor === 'top');
    if (topBars.length) {
      for (const el of targets) if (blocker(el, topBars)) passUnderTop++;
    }

    const seen = new Set();
    const unique = covered.filter(c => {
      const k = c.path + c.blockedBy;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    });
    return {
      bars: barInfo,
      covered: unique.slice(0, 10),
      coveredCount: unique.length,
      passUnderTopBar: passUnderTop,
      documentScrolls: docScrolls,
      scrollRegions: scrollers.length,
    };
  }

  // ── LAYOUT-020 · nothing moves under the user on refresh ──────────────────
  function anchorKey(el, i) {
    const cls = (el.getAttribute('class') || '').trim().split(/\\s+/).slice(0, 2).join('.');
    return [el.tagName, el.id || '', cls, labelOf(el).slice(0, 28), i].join('|');
  }

  function anchors() {
    const out = new Map();
    const els = collect(INTERACTIVE + ',h1,h2,h3,th', root(null));
    els.forEach((el, i) => {
      const r = rectOf(el);
      out.set(anchorKey(el, i), { x: Math.round(r.left), y: Math.round(r.top + window.scrollY) });
    });
    return out;
  }

  function beginShiftWatch() {
    window.__flShift = { cls: 0, entries: 0, before: anchors() };
    try {
      const po = new PerformanceObserver(list => {
        for (const e of list.getEntries()) {
          if (e.hadRecentInput) continue;
          window.__flShift.cls += e.value;
          window.__flShift.entries++;
        }
      });
      po.observe({ type: 'layout-shift', buffered: false });
      window.__flShift.observer = po;
      window.__flShift.supported = true;
    } catch (err) {
      window.__flShift.supported = false;
    }
    return true;
  }

  function endShiftWatch() {
    const s = window.__flShift;
    if (!s) return null;
    if (s.observer) s.observer.disconnect();
    const after = anchors();
    let moved = 0, maxDelta = 0, vanished = 0;
    const worst = [];
    for (const [k, before] of s.before.entries()) {
      const now = after.get(k);
      if (!now) { vanished++; continue; }
      const d = Math.max(Math.abs(now.x - before.x), Math.abs(now.y - before.y));
      if (d > 2) {
        moved++;
        if (d > maxDelta) maxDelta = d;
        worst.push({ key: k.split('|').slice(0, 4).join(' '), dx: now.x - before.x, dy: now.y - before.y });
      }
    }
    worst.sort((a, b) => Math.max(Math.abs(b.dx), Math.abs(b.dy)) - Math.max(Math.abs(a.dx), Math.abs(a.dy)));
    return {
      cls: Math.round(s.cls * 10000) / 10000,
      shiftEntries: s.entries,
      clsSupported: s.supported,
      anchorsBefore: s.before.size,
      anchorsAfter: after.size,
      moved, vanished,
      maxDeltaPx: maxDelta,
      worst: worst.slice(0, 8),
    };
  }


  // ══ Category 17 · Status & Data Display ═══════════════════════════════════
  //
  // Eight of the fourteen STATUS rules have a clause a rendered page can settle.
  // The six that do not — STATUS-001 (orthogonal dimensions), -003 (context not
  // repeated), -009 (chip subtext), -012 (three health levels), -013 (case
  // summary), -014 (tabs with counts) — need someone to read the screen and are
  // not touched here.
  //
  // STATUS-005's hook does not exist in the product yet. Its check is written
  // against the markup the redesign will introduce, so it fails on every chip
  // today and passes once the chips carry it. That is deliberate: the number it
  // reports now is the size of the work.

  // ── Shared: what counts as a chip, a row, and a danger/warning tone ───────

  // Anything shaped like a condition marker. Deliberately generous — a class
  // name is the only signal the markup gives — and pruned three ways: form
  // controls are not chips, navigation and toolbar controls are filters rather
  // than conditions, and only the innermost match is kept so a chip container
  // is not counted alongside the chip.
  const CHIP_SEL = [
    '[data-explain]',
    '[class*="chip"]', '[class*="badge"]', '[class*="pill"]', '[class*="-tag"]', '[class*="status"]',
  ].join(',');

  const CHIP_EXCLUDE_CONTAINER = '.tab-bar,.site-nav,.mobile-nav,.orders-toolbar,.notes-toolbar,.supplies-inventory-toolbar,.supplies-page-header';

  // A plural or wrapper class name is a box of chips, not a chip: .readiness-chips
  // holds three .readiness-chip elements and must not be measured as a fourth.
  const CHIP_CONTAINER_CLASS = /(chips|badges|pills|tags)(?![-\\w])|(-list|-group|-wrap|-wrapper|-container|-cell|-row|-bar)(?![-\\w])/i;

  function ownText(el) {
    let s = '';
    for (const n of el.childNodes) if (n.nodeType === 3) s += n.textContent;
    return s.replace(/\\s+/g, ' ').trim();
  }

  function allText(el) {
    return (el.textContent || '').replace(/\\s+/g, ' ').trim();
  }

  function chips(rootEl) {
    const matches = collect(CHIP_SEL, rootEl).filter(el => {
      if (/^(SELECT|INPUT|TEXTAREA|OPTION|FORM|LABEL)$/.test(el.tagName)) return false;
      if (el.closest(CHIP_EXCLUDE_CONTAINER)) return false;
      if (CHIP_CONTAINER_CLASS.test(el.getAttribute('class') || '')) return false;
      return allText(el).length > 0;
    });
    // The chip is the outermost surviving match: it is the element that carries
    // the fill, the tooltip today and the hook tomorrow. Its label and detail
    // spans are parts of it, not chips of their own.
    return matches.filter(el => !matches.some(o => o !== el && o.contains(el)));
  }

  // Rows of a list or table. Innermost only, so a nested table's rows are not
  // counted twice, and header rows are excluded — they carry no record state.
  const ROW_SEL = [
    'tbody tr', '.order-row', '.er-row', '.supply-item-row', '.allocation-row',
    '.recent-entry-card', '.note-card', '.product-lot-row', '[role="row"]',
  ].join(',');

  // A row the user opened is not a list row: it is the detail the list rule
  // pushes off the row in the first place.
  const EXPANSION_ROW = /(^|[\\s-])(detail|details|expand|expanded|lines?-row|drawer|sub-?row|child|inventory-row|allocation-detail|order-lines)/i;

  function isExpansionRow(el) {
    const cls = el.getAttribute('class') || '';
    if (EXPANSION_ROW.test(cls)) return true;
    if (el.id && EXPANSION_ROW.test(el.id)) return true;
    return false;
  }

  function listRows(rootEl, { includeExpansion = false } = {}) {
    return collect(ROW_SEL, rootEl).filter(el => {
      if (el.querySelector(ROW_SEL)) return false;
      if (el.closest('thead')) return false;
      if (!includeExpansion && isExpansionRow(el)) return false;
      return allText(el).length > 0;
    });
  }

  // Tone is decided by hue against the two semantic tokens the product defines,
  // not by class name: --danger and --warning survive a redesign that renames
  // every class, and color-mix()ed fills resolve to the same hue at a lower
  // alpha. Both token sets (light #dc2626/#d97706, dark #f16161/#f59e0b) sit
  // ~32° apart, so a ±12° window separates them cleanly.
  function hsl(c) {
    const r = c.r / 255, g = c.g / 255, b = c.b / 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    let h = 0;
    if (d !== 0) {
      if (mx === r) h = ((g - b) / d) % 6;
      else if (mx === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h *= 60;
      if (h < 0) h += 360;
    }
    const l = (mx + mn) / 2;
    const s = d === 0 ? 0 : d / (1 - Math.abs(2 * l - 1));
    return { h, s, l };
  }

  function hueGap(a, b) {
    const d = Math.abs(a - b) % 360;
    return d > 180 ? 360 - d : d;
  }

  function tokenHues() {
    const cs = getComputedStyle(document.documentElement);
    const read = name => {
      const raw = cs.getPropertyValue(name).trim();
      if (!raw) return null;
      // A custom property's value is a token stream, not a colour: resolve it
      // by painting it on a throwaway element.
      const probe = document.createElement('span');
      probe.style.cssText = 'position:absolute;left:-9999px;color:' + raw;
      document.body.appendChild(probe);
      const c = parseColor(getComputedStyle(probe).color);
      probe.remove();
      return c ? hsl(c) : null;
    };
    return { danger: read('--danger'), warning: read('--warning') };
  }

  const HUE_WINDOW = 12;
  const TONE_MIN_SAT = 0.22;
  const TONE_MIN_ALPHA = 0.04;

  function toneOfColor(c, hues) {
    if (!c || c.a < TONE_MIN_ALPHA) return null;
    const p = hsl(c);
    if (p.s < TONE_MIN_SAT) return null;
    if (hues.danger && hueGap(p.h, hues.danger.h) <= HUE_WINDOW) return 'danger';
    if (hues.warning && hueGap(p.h, hues.warning.h) <= HUE_WINDOW) return 'warning';
    return null;
  }

  // Returns the tone this element *introduces*. Background, border and outline
  // are the element's own box, so any tone there counts. 'color' inherits, so a
  // tone matching the parent's is the parent's mark, not a second one — that is
  // what keeps a red row from reading as a dozen red alarms.
  function toneIntroduced(el, hues) {
    const cs = getComputedStyle(el);
    const bg = toneOfColor(parseColor(cs.backgroundColor), hues);
    if (bg) return { tone: bg, via: 'background' };
    for (const side of ['Top', 'Right', 'Bottom', 'Left']) {
      if (parseFloat(cs['border' + side + 'Width']) > 0) {
        const t = toneOfColor(parseColor(cs['border' + side + 'Color']), hues);
        if (t) return { tone: t, via: 'border' };
      }
    }
    // outline-color resolves to currentColor and Chrome reports a non-zero
    // outline-width even where outline-style is none, so both have to be
    // checked or every red-texted descendant reads as its own alarm.
    if (cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth) > 0) {
      const t = toneOfColor(parseColor(cs.outlineColor), hues);
      if (t) return { tone: t, via: 'outline' };
    }
    const own = toneOfColor(parseColor(cs.color), hues);
    if (own) {
      const parent = el.parentElement;
      const inherited = parent ? toneOfColor(parseColor(getComputedStyle(parent).color), hues) : null;
      if (inherited !== own) return { tone: own, via: 'text' };
    }
    return null;
  }

  function isColoured(el, hues) {
    const cs = getComputedStyle(el);
    for (const v of [cs.backgroundColor, cs.borderTopColor, cs.borderLeftColor, cs.color]) {
      const c = parseColor(v);
      if (!c || c.a < TONE_MIN_ALPHA) continue;
      if (hsl(c).s >= 0.12) return true;
    }
    return false;
  }

  // ── STATUS-002 · a nominal value is not a badge ───────────────────────────
  // The phrase list is the one written into the rule, plus the two check
  // glyphs. Matching is on the chip's whole text, so "Checks passed" is a hit
  // and "Short 240 lb" is not.
  const NOMINAL = [
    'ok', 'okay', 'good', 'confirmed', 'normal', 'fine', 'healthy', 'nominal',
    'on track', 'no issues', 'all clear', 'all good', 'checks passed', 'no problems',
    'within spec', '\\u2713', '\\u2714',
  ];

  function nominalBadges(rootSel) {
    const rootEl = root(rootSel);
    const hues = tokenHues();
    const found = [];
    const neutral = [];
    const all = chips(rootEl);
    for (const el of all) {
      const text = allText(el).toLowerCase().replace(/[.!·—–-]+$/, '').trim();
      if (!NOMINAL.includes(text)) continue;
      const record = { path: pathOf(el), text: allText(el).slice(0, 48) };
      if (isColoured(el, hues)) found.push(record); else neutral.push(record);
    }
    return {
      checked: all.length,
      failures: found.length,
      // A nominal badge with no colour is still a badge on a normal row, but
      // the rule's hard clause is the coloured one. Recorded, not counted.
      neutralNominal: neutral,
      worst: found,
    };
  }

  // ── STATUS-004 · at most one alarm per row ────────────────────────────────
  function alarmsPerRow(rootSel) {
    const rootEl = root(rootSel);
    const hues = tokenHues();
    if (!hues.danger && !hues.warning) {
      return { applicable: false, reason: 'This surface defines neither --danger nor --warning.' };
    }
    const rows = listRows(rootEl);
    const offenders = [];
    let maxAlarms = 0;
    for (const row of rows) {
      const marks = [];
      const candidates = [row].concat(Array.from(row.querySelectorAll('*')));
      for (const el of candidates) {
        if (!isVisible(el)) continue;
        const t = toneIntroduced(el, hues);
        if (t) marks.push({ path: pathOf(el), label: labelOf(el), tone: t.tone, via: t.via });
      }
      if (marks.length > maxAlarms) maxAlarms = marks.length;
      if (marks.length > 1) {
        offenders.push({
          row: pathOf(row), label: labelOf(row).slice(0, 48),
          alarms: marks.length, marks: marks.slice(0, 6),
        });
      }
    }
    offenders.sort((a, b) => b.alarms - a.alarms);
    return {
      applicable: true,
      checked: rows.length,
      failures: offenders.length,
      maxAlarms,
      worst: offenders,
    };
  }

  // ── STATUS-005 · every condition chip carries the explanation hook ────────
  // The hook is fixed by the rule so the redesign has one thing to implement:
  //   data-explain="<id>"  +  aria-describedby containing that id  +  a focus stop.
  // 'title' is measured separately: it is what the product uses today and what
  // the rule replaces, so counting it shows the size of the migration.
  function explainHooks(rootSel) {
    const rootEl = root(rootSel);
    const all = chips(rootEl);
    const failures = [];
    let titleOnly = 0;
    let complete = 0;
    for (const el of all) {
      const reasons = [];
      const hook = (el.getAttribute('data-explain') || '').trim();
      if (!hook) reasons.push('no data-explain');
      else if (!document.getElementById(hook)) reasons.push('data-explain points at no element');
      const describedBy = (el.getAttribute('aria-describedby') || '').trim().split(/\\s+/).filter(Boolean);
      if (!hook || !describedBy.includes(hook)) reasons.push('aria-describedby does not name the explanation');
      const tabindex = el.getAttribute('tabindex');
      const nativelyFocusable = el.tagName === 'BUTTON' || (el.tagName === 'A' && el.hasAttribute('href'));
      if (!nativelyFocusable && !(tabindex !== null && Number(tabindex) >= 0)) reasons.push('not focusable');
      if (el.hasAttribute('title')) titleOnly++;
      if (!reasons.length) { complete++; continue; }
      failures.push({
        path: pathOf(el), text: allText(el).slice(0, 48),
        reasons, hasTitle: el.hasAttribute('title'),
      });
    }
    return {
      checked: all.length,
      failures: failures.length,
      complete,
      // Chips whose only explanation today is a 'title' tooltip: invisible on
      // touch, and the thing the rule's hook replaces.
      titleOnly,
      worst: failures,
    };
  }

  // ── STATUS-006 · one number formatter ─────────────────────────────────────
  // Four kinds of failure, counted apart so the total is readable:
  //   raw-decimal      three or more decimal places — a stored value on screen
  //   no-separator     1,000 or more written without a thousands separator
  //   lb-precision     a pound value with any decimal
  //   pallet-precision a pallet value with more than one decimal
  //   not-tabular      digits in an aligned column without tabular figures
  // Numbers that are not quantities are skipped: anything glued to letters or a
  // hyphen (LAT codes, SKUs), anything inside a date or clock time, and bare
  // four-digit years.
  const NUM_TOKEN = /(^|[^\\w.,\\-])(\\d[\\d,]*(?:\\.\\d+)?)(?![\\w\\-])/g;
  const DATEISH = /\\d[\\d/:\\-]*[/:]|\\b(19|20)\\d\\d[-/]/;
  const CODEISH_HOST = '[class*="code"],[class*="lot"],[class*="sku"],[class*="ref"],[class*="id"],time,code,pre';

  function numberFormat(rootSel) {
    const rootEl = root(rootSel);
    const kinds = {};
    const worst = [];
    let numbersChecked = 0;

    // A column whose header names an identifier holds names written with
    // digits, not quantities: a numeric SKU is neither misread for want of a
    // thousands separator nor helped by tabular figures.
    const ID_HEADER = /\\b(sku|code|id|lot|ref|reference|order|line|batch|no|number|#)\\b/i;
    function columnHeader(cell) {
      if (!cell || cell.tagName !== 'TD') return '';
      const row = cell.parentElement;
      const table = cell.closest('table');
      if (!row || !table) return '';
      const index = Array.prototype.indexOf.call(row.children, cell);
      const headRow = table.querySelector('thead tr') || table.rows[0];
      if (!headRow || headRow === row) return '';
      const th = headRow.children[index];
      return th ? (th.textContent || '').trim() : '';
    }
    const inIdColumn = el => ID_HEADER.test(columnHeader(el.closest('td')));

    const walker = document.createTreeWalker(rootEl, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);

    for (const node of nodes) {
      const el = node.parentElement;
      if (!el || !isVisible(el)) continue;
      if (el.closest(CODEISH_HOST)) continue;
      if (inIdColumn(el)) continue;
      const text = node.textContent.replace(/\\s+/g, ' ');
      if (!/\\d/.test(text)) continue;
      if (DATEISH.test(text)) continue;

      NUM_TOKEN.lastIndex = 0;
      let m;
      while ((m = NUM_TOKEN.exec(text)) !== null) {
        const raw = m[2];
        const after = text.slice(m.index + m[0].length, m.index + m[0].length + 12).toLowerCase();
        // An identifier is not a quantity: "Line #1010" and "Order 1042" are
        // names written with digits, and a separator in them would be wrong.
        if (m[1] === '#') continue;
        const before = text.slice(0, m.index + m[1].length).toLowerCase();
        if (/(line|order|invoice|receipt|batch|lot|item|ref|reference|po|so|id|no\\.?|#)\\s*$/.test(before)) continue;
        const intPart = raw.split('.')[0].replace(/,/g, '');
        const decimals = raw.includes('.') ? raw.split('.')[1].length : 0;
        // A bare year is a date, not a quantity.
        if (!raw.includes(',') && decimals === 0 && intPart.length === 4) {
          const n = Number(intPart);
          if (n >= 1900 && n <= 2100) continue;
        }
        numbersChecked++;
        const hit = kind => {
          kinds[kind] = (kinds[kind] || 0) + 1;
          worst.push({ kind, value: raw, context: text.trim().slice(0, 60), path: pathOf(el) });
        };
        if (decimals >= 3) hit('raw-decimal');
        else if (!raw.includes(',') && intPart.length >= 4) hit('no-separator');
        else if (decimals > 0 && /^\\s*(lb|lbs|pound)/.test(after)) hit('lb-precision');
        else if (decimals > 1 && /^\\s*pallet/.test(after)) hit('pallet-precision');
      }
    }

    // Tabular figures, where numbers stack: a cell whose whole content is one
    // quantity. A cell holding a date, a code or a sentence does not align
    // digit-over-digit and is not what the clause is about.
    // One quantity, optionally with a unit, optionally repeated after a
    // separator: "3.5", "4,500 lb", "4,500 lb · 356 units". A date, a code or
    // a sentence in the same column does not match and is not measured.
    const QTY = '[\\\\d,]+(?:\\\\.\\\\d+)?(?:\\\\s*(?:%|[a-z]{1,8}))?';
    const QUANTITY_CELL = new RegExp('^[$\\\\s]*' + QTY + '(?:\\\\s*[\\u00b7,/]\\\\s*' + QTY + ')*$', 'i');
    // A lone one- or two-digit integer is not what the tabular clause is for:
    // proportional figures only misalign a column once the values differ in
    // width, and a calendar's day numbers are dates, which DATA-012 owns. The
    // clause bites where a quantity carries a separator, a decimal, a unit, or
    // three or more digits.
    const WIDTH_MATTERS = /[,.]|[a-z%]|\\d{3}/i;
    const aligned = collect('td,th,[class*="num"],[class*="qty"]', rootEl)
      .filter(el => {
        const t = ownText(el);
        if (!QUANTITY_CELL.test(t) || !WIDTH_MATTERS.test(t)) return false;
        return !inIdColumn(el);
      });
    let notTabular = 0;
    for (const el of aligned) {
      const cs = getComputedStyle(el);
      const fvn = cs.fontVariantNumeric || '';
      const ffs = cs.fontFeatureSettings || '';
      if (/tabular-nums/.test(fvn) || /["']tnum["']/.test(ffs)) continue;
      notTabular++;
      if (worst.length < 400) worst.push({ kind: 'not-tabular', value: ownText(el).slice(0, 24), context: '', path: pathOf(el) });
    }
    if (notTabular) kinds['not-tabular'] = notTabular;

    const failures = Object.values(kinds).reduce((a, b) => a + b, 0);
    return {
      checked: numbersChecked,
      alignedCells: aligned.length,
      failures,
      byKind: kinds,
      worst: worst.slice(0, 200),
    };
  }

  // ── STATUS-007 · no orphan placeholders ───────────────────────────────────
  // A dash inside a table cell holds the column's alignment and is part of the
  // grid — the rule's stated exception. A dash on its own anywhere else is the
  // artefact.
  const PLACEHOLDER_ONLY = /^(\\u2014|\\u2013|-{1,2}|n\\/a|none|null|undefined|\\u2014\\s*\\u2014)$/i;

  function orphanPlaceholders(rootSel) {
    const rootEl = root(rootSel);
    const found = [];
    for (const el of collect('*', rootEl)) {
      if (el.tagName === 'TD' || el.tagName === 'TH') continue;
      if (el.closest('td,th') && !el.querySelector('*') && el.parentElement && (el.parentElement.tagName === 'TD' || el.parentElement.tagName === 'TH')) {
        // A lone span wrapping a cell's dash is the cell's placeholder.
        if (ownText(el) === allText(el.closest('td,th'))) continue;
      }
      const own = ownText(el);
      if (!own || own !== allText(el)) continue;
      if (!PLACEHOLDER_ONLY.test(own)) continue;
      found.push({ path: pathOf(el), text: own, tag: el.tagName.toLowerCase(), cls: (el.getAttribute('class') || '').slice(0, 80) });
    }
    return { checked: found.length, failures: found.length, worst: found };
  }

  // ── STATUS-008 · list rows are single-line, ≤ 56 px at desktop width ──────
  function rowHeights(rootSel, opts) {
    const rootEl = root(rootSel);
    const width = (opts && opts.width) || window.innerWidth;
    if (width < 1200) {
      return { applicable: false, reason: 'Desktop clause only; viewport is ' + width + 'px.' };
    }
    const rows = listRows(rootEl);
    if (!rows.length) return { applicable: true, checked: 0, failures: 0, tallest: 0, worst: [] };
    const over = [];
    let tallest = 0;
    for (const row of rows) {
      const h = Math.round(rectOf(row).height * 10) / 10;
      if (h > tallest) tallest = h;
      if (h > 56) over.push({ path: pathOf(row), label: labelOf(row).slice(0, 48), height: h });
    }
    over.sort((a, b) => b.height - a.height);
    return { applicable: true, checked: rows.length, failures: over.length, tallest, worst: over };
  }

  // ── STATUS-010 · no developer vocabulary in user-facing copy ──────────────
  // "returns" is deliberately absent: on a factory floor it means returned
  // goods. The phrase "returns null" is caught by the null term.
  const DEV_TERMS = [
    'api', 'apis', 'endpoint', 'endpoints', 'null', 'undefined', 'nan', 'ttl',
    'cache', 'cached', 'payload', 'json', 'timeout', 'timed out', 'http', 'https',
    'cors', 'localhost', 'stack trace', 'traceback', 'exception', 'unhandled',
    'stringify', 'nullable', 'uuid', 'foreign key', 'primary key', 'schema',
  ];
  const DEV_RE = new RegExp('(^|[^\\\\w])(' + DEV_TERMS.map(t => t.replace(/ /g, '\\\\s+')).join('|') + ')(?![\\\\w])', 'i');
  const SNAKE_RE = /(^|[^\\w])([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?![\\w])/;

  function devVocabulary(rootSel) {
    const rootEl = root(rootSel);
    const found = [];
    const seen = new Set();
    const els = collect('*', rootEl).filter(el => ownText(el));
    for (const el of els) {
      const text = ownText(el);
      const hits = [];
      const dev = DEV_RE.exec(text);
      if (dev) hits.push(dev[2].toLowerCase());
      const snake = SNAKE_RE.exec(text);
      if (snake) hits.push(snake[2]);
      if (!hits.length) continue;
      const key = hits.join('|') + '@' + pathOf(el);
      if (seen.has(key)) continue;
      seen.add(key);
      found.push({ terms: hits, text: text.slice(0, 70), path: pathOf(el) });
    }
    // Attribute copy is user-facing too: a tooltip and a placeholder are read.
    for (const el of collect('[title],[placeholder],[aria-label]', rootEl)) {
      for (const attr of ['title', 'placeholder', 'aria-label']) {
        const v = (el.getAttribute(attr) || '').trim();
        if (!v) continue;
        const dev = DEV_RE.exec(v);
        const snake = SNAKE_RE.exec(v);
        if (!dev && !snake) continue;
        const terms = [dev && dev[2].toLowerCase(), snake && snake[2]].filter(Boolean);
        const key = terms.join('|') + '@' + attr + '@' + pathOf(el);
        if (seen.has(key)) continue;
        seen.add(key);
        found.push({ terms, text: '[' + attr + '] ' + v.slice(0, 60), path: pathOf(el) });
      }
    }
    return { checked: els.length, failures: found.length, worst: found.slice(0, 120) };
  }

  // ── STATUS-011 · a disclaimer appears once per screen ─────────────────────
  // Forty characters is the rule's own threshold: long enough that a repeat is
  // prose rather than a shared label, short enough to catch a one-line caveat.
  function repeatedSentences(rootSel) {
    const rootEl = root(rootSel);
    const counts = new Map();
    for (const el of collect('*', rootEl)) {
      const t = ownText(el);
      if (t.length < 40 || !/\\s/.test(t)) continue;
      let e = counts.get(t);
      if (!e) { e = { text: t, count: 0, paths: [] }; counts.set(t, e); }
      e.count++;
      if (e.paths.length < 4) e.paths.push(pathOf(el));
    }
    const repeated = [...counts.values()].filter(e => e.count > 1).sort((a, b) => b.count - a.count);
    return {
      checked: counts.size,
      failures: repeated.length,
      repeatedInstances: repeated.reduce((n, e) => n + e.count - 1, 0),
      worst: repeated.map(e => ({ text: e.text.slice(0, 90), count: e.count, paths: e.paths })),
    };
  }

  window.__FL_AUDIT = {
    touchTargets, contrast, horizontalOverflow, occlusionAtBottom,
    beginShiftWatch, endShiftWatch,
    // Category 17 — Status & Data Display
    nominalBadges, alarmsPerRow, explainHooks, numberFormat,
    orphanPlaceholders, rowHeights, devVocabulary, repeatedSentences,
    scrollTop: () => { window.scrollTo(0, 0); return true; },
    themeOf: () => document.documentElement.getAttribute('data-theme') || '(none)',
  };
})();
`;

// Registered before any page script so the app's own background-refresh
// callbacks can be invoked on demand instead of waiting out a 60-second timer.
export const INTERVAL_CAPTURE_SOURCE = `
(() => {
  const real = window.setInterval;
  window.__FL_INTERVALS = [];
  window.setInterval = function (fn, ms, ...rest) {
    try { window.__FL_INTERVALS.push({ fn, ms }); } catch (e) {}
    return real.call(window, fn, ms, ...rest);
  };
})();
`;
