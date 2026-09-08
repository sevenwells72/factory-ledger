/* === ER intake review-state logic (audit fix, docs/designs/er-intake-audit-1.md) ===
   Pure functions only — no DOM, no fetch — so the review screen's state rules
   are testable under node:test (tests/test_er_intake_logic.js) exactly like
   pallet-calculations.js. dashboard.js owns rendering and wiring; every state
   transition on a review line goes through here.

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

  function roundLb(value) {
    return Math.round(value * 100) / 100;
  }

  /* One /expected-receipts/match line → the review screen's working copy.
     Only alias/exact matches arrive confirmed; a fuzzy product is demoted to
     `suggested` and its (server-null) pounds stay empty until a human picks. */
  function buildReviewLine(matchLine) {
    const confirmed = CONFIRMED_SOURCES.includes(matchLine.match_source);
    return {
      ...matchLine,
      include: true,
      chosen: confirmed ? (matchLine.product || null) : null,
      suggested: !confirmed ? (matchLine.product || null) : null,
      qty_lb: confirmed ? matchLine.expected_qty_lb : null,
      qty_lb_source: confirmed && matchLine.expected_qty_lb != null ? matchLine.lb_source : null,
    };
  }

  function recomputeQtyLb(line, source) {
    if (line.chosen && line.lb_per_unit > 0 && line.quantity > 0) {
      line.qty_lb = roundLb(line.quantity * line.lb_per_unit);
      line.qty_lb_source = source;
    }
    return line;
  }

  function applyQuantityChange(line, quantity) {
    line.quantity = Number(quantity) > 0 ? Number(quantity) : 0;
    return recomputeQtyLb(line, line.lb_source);
  }

  function applyLbPerUnitChange(line, lbPerUnit) {
    const v = Number(lbPerUnit);
    line.lb_per_unit = v > 0 ? v : null;
    line.lb_source = line.lb_per_unit != null ? 'manual' : 'none';
    return recomputeQtyLb(line, 'manual');
  }

  function applyQtyLbOverride(line, qtyLb) {
    const v = Number(qtyLb);
    line.qty_lb = v > 0 ? v : null;
    line.qty_lb_source = line.qty_lb != null ? 'manual' : null;
    return line;
  }

  /* The ONLY way a product becomes chosen outside alias/exact: an explicit
     human pick (suggestion click, typeahead pick). Text-derived conversions
     may compute pounds now that a human has confirmed the product. */
  function applyProductPick(line, product) {
    line.chosen = {
      product_id: Number(product.product_id),
      name: product.name,
      odoo_code: product.odoo_code || null,
    };
    if (line.qty_lb == null) recomputeQtyLb(line, line.lb_source);
    return line;
  }

  function clearChosen(line) {
    line.chosen = null;
    return line;
  }

  /* Approval requires an explicit product AND positive pounds — typing pounds
     into a product-less line must never make it approvable. */
  function lineApprovable(line) {
    return Boolean(line.chosen && line.chosen.product_id && line.qty_lb > 0);
  }

  return {
    buildReviewLine,
    applyQuantityChange,
    applyLbPerUnitChange,
    applyQtyLbOverride,
    applyProductPick,
    clearChosen,
    lineApprovable,
    roundLb,
  };
}));
