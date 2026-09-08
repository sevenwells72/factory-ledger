// Browser-side mechanical checks. Everything in CHECK_SOURCE runs inside the
// page; the Node side only reads the results back.
//
// Scope of what is and is not checked is deliberate. Each rule below has a
// clause that a rendered page can settle without judgement — a measured size,
// a computed ratio, a scroll width, an occlusion test, a shift score. The
// clauses that need a human (hover affordance, Spanish labels, floor lighting)
// are not touched here and are not reported as passes.

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

  window.__FL_AUDIT = {
    touchTargets, contrast, horizontalOverflow, occlusionAtBottom,
    beginShiftWatch, endShiftWatch,
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
