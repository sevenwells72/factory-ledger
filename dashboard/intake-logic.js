/* === Intake review-state logic (ER + SO; renamed from er-intake-logic.js
   per docs/designs/sales-order-intake.md ruling 10) ===
   Pure functions only — no DOM, no fetch — so the review screen's state rules
   are testable under node:test (tests/test_er_intake_logic.js) exactly like
   pallet-calculations.js. dashboard.js owns rendering and wiring; every state
   transition on a review line goes through here. The ER API is unchanged;
   the sales-order flow reuses the same state machine by normalizing its match
   lines into the shared field names (so* functions at the bottom).

   Audit finding 1 (P1): a fuzzy match is a SUGGESTION, never a selection.
   `chosen` is set only for alias/exact matches or an explicit human pick;
   fuzzy products land in `suggested` and are rendered as a pre-highlighted
   candidate. Pounds are computed only when `chosen` is set. */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.ERIntake = api;
}(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  const CONFIRMED_SOURCES = ['alias', 'exact'];
  /* Conversions that came from a PRODUCT (not from the line's own text/unit):
     they die with a product change. Text/unit-derived and manual conversions
     survive a product change but not a unit change. */
  const PRODUCT_DEPENDENT_LB_SOURCES = ['alias', 'case_size'];

  function roundLb(value) {
    return Math.round(value * 100) / 100;
  }

  function normalizeUnit(unit) {
    const u = (unit == null ? '' : String(unit)).trim().toLowerCase().replace(/\.+$/, '');
    return u || null;
  }

  /* One /expected-receipts/match line → the review screen's working copy.
     Only alias/exact matches arrive confirmed; a fuzzy product is demoted to
     `suggested` and its (server-null) pounds stay empty until a human picks.
     The server's own verdict is kept in server_match_source/confidence so
     "Change" can restore the badge after a human pick set it to 'chosen'.
     qty_lb_source says how the EXPECTED-LB value itself was produced —
     'computed' (quantity × lb/unit) or 'manual' (typed directly). Only a
     direct edit of a field may ever tag that field 'manual'. */
  function buildReviewLine(matchLine) {
    const confirmed = CONFIRMED_SOURCES.includes(matchLine.match_source);
    return {
      ...matchLine,
      server_match_source: matchLine.match_source,
      server_confidence: matchLine.confidence,
      include: true,
      chosen: confirmed ? (matchLine.product || null) : null,
      suggested: !confirmed ? (matchLine.product || null) : null,
      qty_source: 'document',
      qty_lb: confirmed ? matchLine.expected_qty_lb : null,
      qty_lb_source: confirmed && matchLine.expected_qty_lb != null ? 'computed' : null,
      // Audit fix 3: alias learning is opt-out per line; defaults on while the
      // conversion is untouched, off once expected lb is overridden directly.
      save_alias: true,
      save_alias_touched: false,
    };
  }

  function recomputeQtyLb(line) {
    if (line.chosen && line.lb_per_unit > 0 && line.quantity > 0) {
      line.qty_lb = roundLb(line.quantity * line.lb_per_unit);
      line.qty_lb_source = 'computed';
    }
    return line;
  }

  function applyQuantityChange(line, quantity) {
    if (line.matching) return line; // audit-2 fix 4b: locked while a match request is in flight
    const next = Number(quantity) > 0 ? Number(quantity) : 0;
    if (next !== line.quantity) line.qty_source = 'edited';
    line.quantity = next;
    return recomputeQtyLb(line);
  }

  function applyLbPerUnitChange(line, lbPerUnit) {
    if (line.matching) return line; // audit-2 fix 4b: locked while a match request is in flight
    const v = Number(lbPerUnit);
    line.lb_per_unit = v > 0 ? v : null;
    line.lb_source = line.lb_per_unit != null ? 'manual' : 'none';
    return recomputeQtyLb(line);
  }

  function applyQtyLbOverride(line, qtyLb) {
    if (line.matching) return line; // audit-2 fix 4b: locked while a match request is in flight
    const v = Number(qtyLb);
    line.qty_lb = v > 0 ? v : null;
    line.qty_lb_source = line.qty_lb != null ? 'manual' : null;
    // Audit fix 3: an overridden expected-lb means the conversion no longer
    // explains the value — stop teaching it unless the user re-opts in.
    if (!line.save_alias_touched) line.save_alias = false;
    return line;
  }

  /* Audit fix 3: changing the unit invalidates the conversion AND the pounds
     computed from it — every lb_source is per-unit. A cosmetic edit that
     normalizes to the same unit ("BAG" → "bag.") keeps the values. */
  function applyUnitChange(line, unit) {
    if (line.matching) return line; // audit-2 fix 4b: locked while a match request is in flight
    const next = (unit == null ? '' : String(unit).trim()) || null;
    const changed = normalizeUnit(next) !== normalizeUnit(line.unit);
    line.unit = next;
    if (changed) {
      line.qty_source = 'edited';
      line.lb_per_unit = null;
      line.lb_source = 'none';
      line.qty_lb = null;
      line.qty_lb_source = null;
    }
    return line;
  }

  /* The ONLY way a product becomes chosen outside alias/exact: an explicit
     human pick (suggestion click, typeahead pick). The pick sets
     match_source to 'chosen' — the Fuzzy/No-match badge must not survive a
     human decision (the product is no longer a guess). Text-derived
     conversions may compute pounds now that a human confirmed the product. */
  function applyProductPick(line, product) {
    if (line.matching) return line; // audit-2 fix 4b: locked while a match request is in flight
    line.chosen = {
      product_id: Number(product.product_id),
      name: product.name,
      odoo_code: product.odoo_code || null,
    };
    // SO intake (rulings 2/3): the private-label first-sale badge needs these
    // on human picks too. ER picks don't carry them — their chosen shape is
    // unchanged (the ER test suite pins it).
    if ('label_type' in product || 'prior_sales' in product) {
      line.chosen.label_type = product.label_type || null;
      line.chosen.prior_sales = product.prior_sales === true ? true
        : product.prior_sales === false ? false : null;
    }
    line.match_source = 'chosen';
    line.confidence = 1.0;
    if (line.qty_lb == null) recomputeQtyLb(line);
    return line;
  }

  /* Audit fix 3: un-choosing (the "Change" button) is the first half of a
     product change — a conversion that came from the OLD product (alias /
     case weight) and the pounds computed from it are stale for the next pick.
     Text/unit-derived and manual conversions survive; the next pick recomputes
     pounds from them. */
  function clearChosen(line) {
    if (line.matching) return line; // audit-2 fix 4b: locked while a match request is in flight
    line.chosen = null;
    // A 'chosen' badge without a chosen product would lie — fall back to
    // what the server actually said about this line.
    line.match_source = line.server_match_source;
    line.confidence = line.server_confidence;
    if (PRODUCT_DEPENDENT_LB_SOURCES.includes(line.lb_source)) {
      line.lb_per_unit = null;
      line.lb_source = 'none';
    }
    line.qty_lb = null;
    line.qty_lb_source = null;
    return line;
  }

  function applySaveAliasToggle(line, checked) {
    if (line.matching) return line; // audit-2 fix 4b: locked while a match request is in flight
    line.save_alias = Boolean(checked);
    line.save_alias_touched = true;
    return line;
  }

  /* Audit fix 3: the conversion is sent for alias learning only when it
     actually explains the approved pounds (quantity × lb/unit = expected lb,
     to the same 2-decimal rounding the UI computes with). */
  function aliasConversionConsistent(line) {
    return Boolean(
      line.lb_per_unit > 0 && line.quantity > 0 && line.qty_lb != null &&
      Math.abs(roundLb(line.quantity * line.lb_per_unit) - line.qty_lb) < 0.005
    );
  }

  /* The /extract/approve line payload for an included, approvable line. */
  function approveLinePayload(line) {
    const sendConversion = Boolean(line.save_alias) && aliasConversionConsistent(line);
    return {
      product_id: line.chosen.product_id,
      expected_qty_lb: line.qty_lb,
      vendor_description: line.vendor_description,
      quantity: line.quantity,
      unit: line.unit,
      lb_per_unit: sendConversion ? line.lb_per_unit : null,
      save_alias: Boolean(line.save_alias),
    };
  }

  /* Approval requires an explicit product AND positive pounds — typing pounds
     into a product-less line must never make it approvable. */
  function lineApprovable(line) {
    return Boolean(line.chosen && line.chosen.product_id && line.qty_lb > 0);
  }

  /* Audit fix 4: a supplier re-match must not throw the review away. Lines
     correspond by index (same extraction). Per line: keep the exclusion flag
     and any explicit save_alias choice; keep a user-chosen product only when
     the new alias/exact result names the same product (the fresh line then
     carries the new supplier's conversion); otherwise the line goes back to
     unconfirmed. Everything else — match data, conversions — comes from the
     fresh result, which was computed from the CURRENT (edited) qty/unit. */
  function mergeRematch(prevLines, matchLines, build) {
    const buildLine = build || buildReviewLine;
    return matchLines.map((ml, i) => {
      const fresh = buildLine(ml);
      const prev = prevLines && prevLines[i];
      if (!prev) return fresh;
      fresh.include = prev.include;
      if (prev.save_alias_touched) {
        fresh.save_alias = prev.save_alias;
        fresh.save_alias_touched = true;
      }
      if (prev.chosen) {
        const agrees = fresh.chosen && fresh.chosen.product_id === prev.chosen.product_id;
        if (!agrees) {
          fresh.chosen = null;
          fresh.qty_lb = null;
          fresh.qty_lb_source = null;
        }
      }
      return fresh;
    });
  }

  /* Audit-2 fix 4b: while a /match request is in flight the whole review is
     about to be replaced — an edit made now would either be lost to the
     merge or silently applied against stale data. dashboard.js locks the
     lines when a match starts (and disables the rendered inputs); every
     mutator above refuses edits on a locked line. A successful match
     replaces the lines (fresh, unlocked); a failed one goes through
     applyRematchFailure, which unlocks. */
  function lockLines(lines) {
    (lines || []).forEach(l => { l.matching = true; });
    return lines;
  }

  function unlockLines(lines) {
    (lines || []).forEach(l => { l.matching = false; });
    return lines;
  }

  /* Audit-2 fix 4a: a FAILED supplier re-match leaves every line's match
     data computed against the wrong supplier. The newly selected supplier is
     kept (the user's choice stands), but every line drops back to
     unconfirmed — chosen product, conversion, and pounds are all stale until
     a re-match against the new supplier succeeds. Exclusions and explicit
     save_alias choices survive, same as mergeRematch. */
  function applyRematchFailure(lines) {
    return (lines || []).map(l => ({
      ...l,
      matching: false,
      chosen: null,
      match_source: l.server_match_source != null ? l.server_match_source : l.match_source,
      confidence: l.server_confidence != null ? l.server_confidence : l.confidence,
      lb_per_unit: null,
      lb_source: 'none',
      qty_lb: null,
      qty_lb_source: null,
    }));
  }

  /* Clipboard paste support: pick the first PNG/JPEG file from a paste
     event's clipboardData.files. Only images — text stays pasteable into the
     modal's fields, and clipboard PDFs (rare, unreliable across browsers)
     keep going through drop/browse. */
  function clipboardImageFile(files) {
    for (const f of Array.from(files || [])) {
      if (f && (f.type === 'image/png' || f.type === 'image/jpeg')) return f;
    }
    return null;
  }

  /* Pasted images have no useful name ("image.png") — stamp one so the stored
     original_filename says where and when it came from. */
  function clipboardFilename(date, mimeType) {
    const p = n => String(n).padStart(2, '0');
    const stamp = `${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}`
      + `-${p(date.getHours())}${p(date.getMinutes())}${p(date.getSeconds())}`;
    return `clipboard-${stamp}.${mimeType === 'image/jpeg' ? 'jpg' : 'png'}`;
  }

  /* Audit fix 4: the duplicate override is bound to the exact reviewed
     (supplier_id, normalized reference) pair — editing either disarms it. */
  function forceKey(supplierId, reference) {
    const ref = (reference == null ? '' : String(reference)).trim().replace(/\s+/g, ' ').toLowerCase();
    return `${supplierId || ''}|${ref}`;
  }

  /* ── Sales-order intake (docs/designs/sales-order-intake.md) ──
     The SO flow reuses the whole state machine above by mapping its match
     response into the shared field names: description → vendor_description,
     case_size_lb → lb_per_unit, case_size_source → lb_source ('product' →
     'case_size' so product-derived conversions die with a product change),
     quantity_lb → expected_qty_lb. customer_item_code and unit_price ride
     along untouched. */

  const SO_LB_UNITS = ['lb', 'lbs', 'lb.', 'lbs.', 'pound', 'pounds', '#'];
  const SO_CASE_UNITS = ['case', 'cases', 'cs', 'box', 'boxes', 'ctn', 'carton', 'cartons'];

  function soNormalizeMatchLine(ml) {
    // An lb-unit line converts at 1 lb per unit — exactly how the ER match
    // response models it — so a product pick (and clearChosen → re-pick)
    // recomputes its pounds instead of losing them. The conversion survives
    // product changes (unit_is_lb is not product-dependent).
    const isLb = SO_LB_UNITS.includes(normalizeUnit(ml.unit) || '');
    return {
      ...ml,
      vendor_description: ml.description,
      lb_per_unit: ml.case_size_lb != null ? ml.case_size_lb : (isLb ? 1 : null),
      lb_source: ml.case_size_source === 'product' ? 'case_size' : (ml.case_size_source || 'none'),
      expected_qty_lb: ml.quantity_lb != null ? ml.quantity_lb : null,
    };
  }

  function soBuildReviewLine(ml) {
    return buildReviewLine(soNormalizeMatchLine(ml));
  }

  function soMergeRematch(prevLines, matchLines) {
    return mergeRematch(prevLines, matchLines, soBuildReviewLine);
  }

  /* Owner ruling 5: a price is stored only when its basis is unambiguous —
     per-case on a cases line, per-lb on an lb line. Anything else (no unit,
     'ea', 'pallet', …) renders a "price basis unclear" tag and sends null. */
  function soPriceBasis(unit) {
    const u = normalizeUnit(unit) || '';
    if (SO_LB_UNITS.includes(u)) return 'per_lb';
    if (SO_CASE_UNITS.includes(u)) return 'per_case';
    return 'unclear';
  }

  function applyUnitPriceChange(line, price) {
    if (line.matching) return line; // locked while a match request is in flight
    const v = Number(price);
    line.unit_price = price !== '' && v >= 0 && isFinite(v) ? v : null;
    return line;
  }

  /* Ruling 2: warn — never block — when a chosen private-label product has no
     prior sales to this customer. prior_sales === null (unknown, e.g. an ER
     pick) never warns. */
  function soPrivateLabelWarning(line) {
    return Boolean(line.chosen && line.chosen.label_type === 'private_label'
      && line.chosen.prior_sales === false);
  }

  /* The /sales/orders/extract/approve line payload for an included,
     approvable line. The conversion (case_size_lb) is sent for alias learning
     only when it explains the approved pounds — same invariant as ER. */
  function soApproveLinePayload(line) {
    // An lb line's 1-lb-per-unit conversion is a display device, never a
    // learnable case size — teaching case_size_lb=1 could corrupt a future
    // cases line through the alias.
    const isLb = SO_LB_UNITS.includes(normalizeUnit(line.unit) || '');
    const sendConversion = Boolean(line.save_alias) && !isLb && aliasConversionConsistent(line);
    return {
      product_id: line.chosen.product_id,
      quantity_lb: line.qty_lb,
      quantity: line.quantity,
      unit: line.unit,
      case_size_lb: sendConversion ? line.lb_per_unit : null,
      unit_price: soPriceBasis(line.unit) === 'unclear' ? null
        : (line.unit_price != null ? line.unit_price : null),
      customer_item_code: line.customer_item_code || null,
      customer_description: line.vendor_description,
      save_alias: Boolean(line.save_alias),
    };
  }

  return {
    buildReviewLine,
    applyQuantityChange,
    applyLbPerUnitChange,
    applyQtyLbOverride,
    applyUnitChange,
    applyProductPick,
    clearChosen,
    applySaveAliasToggle,
    aliasConversionConsistent,
    approveLinePayload,
    lineApprovable,
    mergeRematch,
    applyRematchFailure,
    lockLines,
    unlockLines,
    clipboardImageFile,
    clipboardFilename,
    forceKey,
    normalizeUnit,
    roundLb,
    soNormalizeMatchLine,
    soBuildReviewLine,
    soMergeRematch,
    soPriceBasis,
    applyUnitPriceChange,
    soPrivateLabelWarning,
    soApproveLinePayload,
  };
}));
