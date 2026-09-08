// ER intake review-state logic — regression tests for the audit fixes
// (docs/designs/er-intake-audit-1.md). Run: node --test tests/test_er_intake_logic.js
// (also wrapped by tests/test_er_intake_logic_js.py so pytest runs it).
const test = require('node:test');
const assert = require('node:assert/strict');

const ERIntake = require('../dashboard/er-intake-logic.js');

function matchLine(overrides = {}) {
  return {
    vendor_description: 'VNDR THING 50LB',
    quantity: 4,
    unit: 'BAG',
    match_source: 'none',
    confidence: 0,
    product: null,
    candidates: [],
    lb_per_unit: null,
    lb_source: 'none',
    expected_qty_lb: null,
    ...overrides,
  };
}

const PROD = { product_id: 7, name: 'Thing 50 LB', odoo_code: 'T-50' };

// ── Audit fix 1: fuzzy is never confirmed ──────────────────────────────────

test('alias match arrives chosen with computed lb', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'alias', confidence: 1, product: PROD,
    lb_per_unit: 50, lb_source: 'alias', expected_qty_lb: 200,
  }));
  assert.deepEqual(l.chosen, PROD);
  assert.equal(l.suggested, null);
  assert.equal(l.qty_lb, 200);
  assert.equal(l.qty_lb_source, 'alias');
  assert.equal(ERIntake.lineApprovable(l), true);
});

test('exact match arrives chosen', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'exact', confidence: 1, product: PROD,
    lb_per_unit: 25, lb_source: 'case_size', expected_qty_lb: 100,
  }));
  assert.deepEqual(l.chosen, PROD);
  assert.equal(ERIntake.lineApprovable(l), true);
});

test('fuzzy match is a suggestion, not a selection', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'fuzzy', confidence: 0.8, product: PROD, candidates: [PROD],
    lb_per_unit: 50, lb_source: 'parsed_description',
  }));
  assert.equal(l.chosen, null);
  assert.deepEqual(l.suggested, PROD);
  assert.equal(l.qty_lb, null);
  assert.equal(ERIntake.lineApprovable(l), false);
});

test('quantity change never computes lb for an unconfirmed (fuzzy) line', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'fuzzy', confidence: 0.8, product: PROD,
    lb_per_unit: 50, lb_source: 'parsed_description',
  }));
  ERIntake.applyQuantityChange(l, 10);
  assert.equal(l.qty_lb, null, 'pounds must stay empty until a human picks a product');
  ERIntake.applyLbPerUnitChange(l, 40);
  assert.equal(l.qty_lb, null);
});

test('typing pounds directly does not make a product-less line approvable', () => {
  const l = ERIntake.buildReviewLine(matchLine({ match_source: 'fuzzy', product: PROD }));
  ERIntake.applyQtyLbOverride(l, 500);
  assert.equal(l.qty_lb, 500);
  assert.equal(ERIntake.lineApprovable(l), false, 'approval requires an explicit product pick');
});

test('explicit pick confirms the product and computes text-derived lb', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'fuzzy', product: PROD,
    lb_per_unit: 50, lb_source: 'parsed_description',
  }));
  ERIntake.applyProductPick(l, PROD);
  assert.deepEqual(l.chosen, PROD);
  assert.equal(l.qty_lb, 200); // 4 × 50
  assert.equal(l.qty_lb_source, 'parsed_description');
  assert.equal(ERIntake.lineApprovable(l), true);
});

test('clearing the chosen product de-confirms the line', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'exact', product: PROD, lb_per_unit: 25,
    lb_source: 'case_size', expected_qty_lb: 100,
  }));
  ERIntake.clearChosen(l);
  assert.equal(ERIntake.lineApprovable(l), false);
});

// ── Audit fix 3: product/unit changes invalidate conversions; alias saving is
// per-line, opt-out, and only sends a conversion that explains the pounds ──

test('unit change invalidates lb_per_unit and expected lb', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'alias', product: PROD, unit: 'BAG',
    lb_per_unit: 50, lb_source: 'alias', expected_qty_lb: 200,
  }));
  ERIntake.applyUnitChange(l, 'CASE');
  assert.equal(l.lb_per_unit, null);
  assert.equal(l.lb_source, 'none');
  assert.equal(l.qty_lb, null);
  assert.equal(ERIntake.lineApprovable(l), false);
});

test('cosmetic unit edit (same normalized unit) keeps the conversion', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'alias', product: PROD, unit: 'BAG',
    lb_per_unit: 50, lb_source: 'alias', expected_qty_lb: 200,
  }));
  ERIntake.applyUnitChange(l, ' bag. ');
  assert.equal(l.lb_per_unit, 50);
  assert.equal(l.qty_lb, 200);
});

test('product change drops product-derived conversions and stale pounds', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'exact', product: PROD, unit: 'CASE',
    lb_per_unit: 25, lb_source: 'case_size', expected_qty_lb: 100,
  }));
  ERIntake.clearChosen(l); // "Change" button — first half of a product change
  assert.equal(l.lb_per_unit, null, 'old product’s case weight must not carry over');
  assert.equal(l.qty_lb, null);
  ERIntake.applyProductPick(l, { product_id: 9, name: 'Other', odoo_code: null });
  assert.equal(l.qty_lb, null, 'no conversion left — pounds need manual entry');
});

test('product change keeps text-derived conversion and recomputes pounds', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'exact', product: PROD, unit: 'BAG',
    lb_per_unit: 50, lb_source: 'parsed_description', expected_qty_lb: 200,
  }));
  ERIntake.clearChosen(l);
  assert.equal(l.lb_per_unit, 50, 'the description’s own weight token is product-independent');
  assert.equal(l.qty_lb, null, 'but stale pounds are dropped until re-pick');
  ERIntake.applyProductPick(l, { product_id: 9, name: 'Other', odoo_code: null });
  assert.equal(l.qty_lb, 200);
});

test('save_alias defaults on; overriding expected lb flips it off', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'alias', product: PROD, unit: 'BAG',
    lb_per_unit: 50, lb_source: 'alias', expected_qty_lb: 200,
  }));
  assert.equal(l.save_alias, true);
  ERIntake.applyQtyLbOverride(l, 175);
  assert.equal(l.save_alias, false, 'overridden pounds must not teach the conversion');
});

test('an explicit save_alias choice survives a later lb override', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'alias', product: PROD, unit: 'BAG',
    lb_per_unit: 50, lb_source: 'alias', expected_qty_lb: 200,
  }));
  ERIntake.applySaveAliasToggle(l, true);
  ERIntake.applyQtyLbOverride(l, 175);
  assert.equal(l.save_alias, true);
});

test('approve payload sends the conversion only when it explains the pounds', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'alias', product: PROD, unit: 'BAG',
    lb_per_unit: 50, lb_source: 'alias', expected_qty_lb: 200,
  }));
  assert.equal(ERIntake.aliasConversionConsistent(l), true);
  assert.equal(ERIntake.approveLinePayload(l).lb_per_unit, 50);
  assert.equal(ERIntake.approveLinePayload(l).save_alias, true);

  ERIntake.applySaveAliasToggle(l, true); // keep saving the product mapping…
  ERIntake.applyQtyLbOverride(l, 175);    // …but the conversion no longer fits
  assert.equal(ERIntake.aliasConversionConsistent(l), false);
  const payload = ERIntake.approveLinePayload(l);
  assert.equal(payload.save_alias, true);
  assert.equal(payload.lb_per_unit, null, 'inconsistent conversion must not be taught');
  assert.equal(payload.expected_qty_lb, 175);
});

test('save_alias false sends no conversion', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'alias', product: PROD, unit: 'BAG',
    lb_per_unit: 50, lb_source: 'alias', expected_qty_lb: 200,
  }));
  ERIntake.applySaveAliasToggle(l, false);
  const payload = ERIntake.approveLinePayload(l);
  assert.equal(payload.save_alias, false);
  assert.equal(payload.lb_per_unit, null);
});

test('manual lb-per-unit correction stays teachable and consistent', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'exact', product: PROD, unit: 'BAG',
    lb_per_unit: 25, lb_source: 'case_size', expected_qty_lb: 100,
  }));
  ERIntake.applyLbPerUnitChange(l, 30); // human corrects the conversion
  assert.equal(l.qty_lb, 120);
  assert.equal(l.save_alias, true);
  const payload = ERIntake.approveLinePayload(l);
  assert.equal(payload.lb_per_unit, 30);
});

test('no-match line starts fully unconfirmed', () => {
  const l = ERIntake.buildReviewLine(matchLine());
  assert.equal(l.chosen, null);
  assert.equal(l.suggested, null);
  assert.equal(l.include, true);
  assert.equal(ERIntake.lineApprovable(l), false);
});
