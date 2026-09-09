// Intake review-state logic — regression tests for the ER audit fixes
// (docs/designs/er-intake-audit-1.md) plus the SO intake config
// (docs/designs/sales-order-intake.md). The module was renamed
// er-intake-logic.js → intake-logic.js (SO design ruling 10); every ER
// assertion below is unchanged. Run: node --test tests/test_er_intake_logic.js
// (also wrapped by tests/test_er_intake_logic_js.py so pytest runs it).
const test = require('node:test');
const assert = require('node:assert/strict');

const ERIntake = require('../dashboard/intake-logic.js');

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
  assert.equal(l.qty_lb_source, 'computed', 'expected lb is qty × lb/unit — tagged computed, never the conversion source');
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
  assert.equal(l.qty_lb_source, 'computed');
  assert.equal(l.match_source, 'chosen', 'a human pick must replace the Fuzzy badge');
  assert.equal(ERIntake.lineApprovable(l), true);
});

// ── 2026-09-08 live-smoke regression: the exact /match payload the prod
// smoke produced ("tote of honey", Dutch Gold Honey, fuzzy 43% → Honey
// 11030). An untouched fuzzy line must never be approvable, and Approve
// stays blocked until a human picks the product AND supplies pounds. ──────
test('live case: fuzzy 43% honey line cannot enable Approve untouched', () => {
  const honey = { product_id: 30, name: 'Honey', odoo_code: '11030', similarity: 0.429 };
  const l = ERIntake.buildReviewLine({
    vendor_description: 'tote of honey', quantity: 1.0, unit: 'tote',
    match_source: 'fuzzy', confidence: 0.429,
    product: honey, candidates: [honey],
    lb_per_unit: null, lb_source: 'none', expected_qty_lb: null,
  });
  assert.equal(l.chosen, null);
  assert.deepEqual(l.suggested, honey);
  assert.equal(l.qty_lb, null);
  assert.equal(l.match_source, 'fuzzy', 'badge still says Fuzzy while unpicked');
  assert.equal(ERIntake.lineApprovable(l), false);
  const ready = [l].filter(x => x.include).every(ERIntake.lineApprovable);
  assert.equal(ready, false, 'the Approve gate must be closed');

  // Typing pounds alone must not open it either.
  ERIntake.applyQtyLbOverride(l, 640);
  assert.equal(ERIntake.lineApprovable(l), false);

  // Picking the suggestion clears the Fuzzy badge → 'chosen', and only the
  // combination of a pick + pounds approves.
  ERIntake.applyProductPick(l, honey);
  assert.equal(l.match_source, 'chosen');
  assert.equal(ERIntake.lineApprovable(l), true);
});

test('Change restores the server badge; re-picking sets chosen again', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'fuzzy', confidence: 0.429, product: PROD, candidates: [PROD],
  }));
  ERIntake.applyProductPick(l, PROD);
  assert.equal(l.match_source, 'chosen');
  ERIntake.clearChosen(l); // the "Change" button
  assert.equal(l.match_source, 'fuzzy', 'no product chosen → back to the server verdict');
  assert.equal(l.confidence, 0.429);
  ERIntake.applyProductPick(l, { product_id: 9, name: 'Other', odoo_code: null });
  assert.equal(l.match_source, 'chosen');
});

test('Change on an exact line restores Exact, not Chosen', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'exact', confidence: 1, product: PROD,
  }));
  ERIntake.clearChosen(l);
  assert.equal(l.match_source, 'exact');
});

// ── Untouched fields are never labeled "manual"; qty/unit track edits ─────
test('editing lb/unit never marks the untouched expected-lb field manual', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'exact', product: PROD, unit: 'BAG',
  }));
  ERIntake.applyLbPerUnitChange(l, 30);
  assert.equal(l.lb_source, 'manual', 'the field the user actually typed in');
  assert.equal(l.qty_lb, 120);
  assert.equal(l.qty_lb_source, 'computed', 'derived value is computed, not manual');
  ERIntake.applyQtyLbOverride(l, 100);
  assert.equal(l.qty_lb_source, 'manual', 'a direct override IS manual');
});

test('qty/unit source starts as document and flips to edited on a real change', () => {
  const l = ERIntake.buildReviewLine(matchLine({ quantity: 4, unit: 'BAG' }));
  assert.equal(l.qty_source, 'document');
  ERIntake.applyQuantityChange(l, 4);      // unchanged value
  assert.equal(l.qty_source, 'document');
  ERIntake.applyUnitChange(l, ' bag. ');   // cosmetic unit edit
  assert.equal(l.qty_source, 'document');
  ERIntake.applyQuantityChange(l, 6);
  assert.equal(l.qty_source, 'edited');
});

test('a failed re-match never leaves a Chosen badge on a product-less line', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'fuzzy', confidence: 0.429, product: PROD, candidates: [PROD],
  }));
  ERIntake.applyProductPick(l, PROD);
  const [reset] = ERIntake.applyRematchFailure([l]);
  assert.equal(reset.chosen, null);
  assert.equal(reset.match_source, 'fuzzy', 'badge falls back to the server verdict');
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

// ── Audit fix 4: supplier re-match preserves the review; force is keyed ────

test('mergeRematch preserves exclusions', () => {
  const prev = [ERIntake.buildReviewLine(matchLine())];
  prev[0].include = false;
  const merged = ERIntake.mergeRematch(prev, [matchLine()]);
  assert.equal(merged[0].include, false);
});

test('mergeRematch keeps a user pick the new alias/exact result agrees with', () => {
  const prev = [ERIntake.buildReviewLine(matchLine({ match_source: 'fuzzy', product: PROD }))];
  ERIntake.applyProductPick(prev[0], PROD); // human confirmed product 7
  const merged = ERIntake.mergeRematch(prev, [matchLine({
    match_source: 'alias', product: PROD,
    lb_per_unit: 50, lb_source: 'alias', expected_qty_lb: 200,
  })]);
  assert.equal(merged[0].chosen.product_id, PROD.product_id);
  assert.equal(merged[0].qty_lb, 200, 'the new supplier’s conversion applies');
});

test('mergeRematch resets a pick the new result does not confirm', () => {
  const prev = [ERIntake.buildReviewLine(matchLine({ match_source: 'fuzzy', product: PROD }))];
  ERIntake.applyProductPick(prev[0], PROD);
  ERIntake.applyQtyLbOverride(prev[0], 500);
  // New supplier: only a fuzzy suggestion of the same product — not confirmation.
  const merged = ERIntake.mergeRematch(prev, [matchLine({ match_source: 'fuzzy', product: PROD })]);
  assert.equal(merged[0].chosen, null, 'stale pick must go back to unconfirmed');
  assert.equal(merged[0].qty_lb, null);
  assert.equal(ERIntake.lineApprovable(merged[0]), false);
});

test('mergeRematch resets a pick when the new result names a different product', () => {
  const other = { product_id: 8, name: 'Other Thing', odoo_code: null };
  const prev = [ERIntake.buildReviewLine(matchLine({ match_source: 'exact', product: PROD, expected_qty_lb: 100, lb_per_unit: 25, lb_source: 'case_size' }))];
  const merged = ERIntake.mergeRematch(prev, [matchLine({
    match_source: 'alias', product: other,
    lb_per_unit: 30, lb_source: 'alias', expected_qty_lb: 120,
  })]);
  assert.equal(merged[0].chosen, null);
  assert.equal(merged[0].qty_lb, null);
  assert.deepEqual(merged[0].candidates, []);
});

test('mergeRematch keeps an explicit save_alias choice, resets defaults', () => {
  const prevTouched = ERIntake.buildReviewLine(matchLine({ match_source: 'alias', product: PROD }));
  ERIntake.applySaveAliasToggle(prevTouched, false);
  const prevDefault = ERIntake.buildReviewLine(matchLine({ match_source: 'alias', product: PROD }));
  ERIntake.applyQtyLbOverride(prevDefault, 10); // auto-off, not user-chosen
  const merged = ERIntake.mergeRematch(
    [prevTouched, prevDefault],
    [matchLine({ match_source: 'alias', product: PROD }), matchLine({ match_source: 'alias', product: PROD })]
  );
  assert.equal(merged[0].save_alias, false, 'explicit unchecking survives');
  assert.equal(merged[1].save_alias, true, 'auto-defaults recompute for the fresh line');
});

// ── Audit-2 fix 4a: a failed re-match resets every line to unconfirmed ─────

test('applyRematchFailure: old product and 200 lb are not approvable after a failed rematch', () => {
  // Reviewed under the OLD supplier: alias-confirmed product, 4 × 50 = 200 lb.
  const prev = [ERIntake.buildReviewLine(matchLine({
    match_source: 'alias', confidence: 1, product: PROD,
    lb_per_unit: 50, lb_source: 'alias', expected_qty_lb: 200,
  }))];
  assert.equal(ERIntake.lineApprovable(prev[0]), true, 'sanity: approvable before the failure');
  // Supplier changed → re-match request failed. Everything computed for the
  // old supplier is stale: the line must drop back to unconfirmed.
  const reset = ERIntake.applyRematchFailure(prev);
  assert.equal(reset[0].chosen, null);
  assert.equal(reset[0].lb_per_unit, null);
  assert.equal(reset[0].lb_source, 'none');
  assert.equal(reset[0].qty_lb, null);
  assert.equal(reset[0].qty_lb_source, null);
  assert.equal(ERIntake.lineApprovable(reset[0]), false, 'old product + 200 lb must not be approvable');
});

test('applyRematchFailure keeps exclusions and explicit save_alias choices', () => {
  const a = ERIntake.buildReviewLine(matchLine({ match_source: 'alias', product: PROD }));
  a.include = false;
  const b = ERIntake.buildReviewLine(matchLine({ match_source: 'alias', product: PROD }));
  ERIntake.applySaveAliasToggle(b, false);
  const reset = ERIntake.applyRematchFailure([a, b]);
  assert.equal(reset[0].include, false);
  assert.equal(reset[1].save_alias, false);
  assert.equal(reset[1].save_alias_touched, true);
});

// ── Audit-2 fix 4b: edits are rejected while a match request is in flight ──

test('edits are rejected while matching=true', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'alias', confidence: 1, product: PROD,
    lb_per_unit: 50, lb_source: 'alias', expected_qty_lb: 200,
  }));
  ERIntake.lockLines([l]);
  const before = JSON.stringify(l);

  ERIntake.applyQuantityChange(l, 99);
  ERIntake.applyUnitChange(l, 'CASE');
  ERIntake.applyLbPerUnitChange(l, 1);
  ERIntake.applyQtyLbOverride(l, 1);
  ERIntake.applySaveAliasToggle(l, false);
  ERIntake.applyProductPick(l, { product_id: 8, name: 'Other' });
  ERIntake.clearChosen(l);

  assert.equal(JSON.stringify(l), before, 'no mutator may change a locked line');
  assert.equal(l.quantity, 4);
  assert.equal(l.chosen.product_id, PROD.product_id);
  assert.equal(l.qty_lb, 200);
});

test('unlockLines re-enables edits; applyRematchFailure also unlocks', () => {
  const l = ERIntake.buildReviewLine(matchLine({
    match_source: 'alias', product: PROD, lb_per_unit: 50, lb_source: 'alias', expected_qty_lb: 200,
  }));
  ERIntake.lockLines([l]);
  ERIntake.unlockLines([l]);
  ERIntake.applyQuantityChange(l, 3);
  assert.equal(l.quantity, 3, 'unlocked line accepts edits again');

  const locked = ERIntake.lockLines([ERIntake.buildReviewLine(matchLine())]);
  const reset = ERIntake.applyRematchFailure(locked);
  assert.equal(reset[0].matching, false);
  ERIntake.applyQuantityChange(reset[0], 7);
  assert.equal(reset[0].quantity, 7, 'a failed rematch must not leave lines locked');
});

// ── Clipboard paste support ────────────────────────────────────────────────

test('clipboardImageFile picks the first PNG/JPEG and ignores everything else', () => {
  const png = { type: 'image/png' };
  const jpg = { type: 'image/jpeg' };
  assert.equal(ERIntake.clipboardImageFile([{ type: 'text/plain' }, png, jpg]), png);
  assert.equal(ERIntake.clipboardImageFile([jpg]), jpg);
  assert.equal(ERIntake.clipboardImageFile([{ type: 'text/plain' }, { type: 'application/pdf' }]), null,
    'text and clipboard PDFs are not hijacked');
  assert.equal(ERIntake.clipboardImageFile([]), null);
  assert.equal(ERIntake.clipboardImageFile(null), null);
  assert.equal(ERIntake.clipboardImageFile(undefined), null);
});

test('clipboardFilename stamps clipboard-YYYYMMDD-HHMMSS with the right extension', () => {
  const d = new Date(2026, 8, 8, 14, 55, 7); // 2026-09-08 14:55:07 local
  assert.equal(ERIntake.clipboardFilename(d, 'image/png'), 'clipboard-20260908-145507.png');
  assert.equal(ERIntake.clipboardFilename(d, 'image/jpeg'), 'clipboard-20260908-145507.jpg');
  const early = new Date(2026, 0, 3, 4, 5, 6); // zero-padding on every field
  assert.equal(ERIntake.clipboardFilename(early, 'image/png'), 'clipboard-20260103-040506.png');
});

test('forceKey binds the override to the normalized (supplier, reference) pair', () => {
  assert.equal(ERIntake.forceKey(3, ' PO-777 '), ERIntake.forceKey(3, 'po-777'));
  assert.notEqual(ERIntake.forceKey(3, 'PO-777'), ERIntake.forceKey(4, 'PO-777'));
  assert.notEqual(ERIntake.forceKey(3, 'PO-777'), ERIntake.forceKey(3, 'PO-778'));
  assert.equal(ERIntake.forceKey(3, 'A  B'), ERIntake.forceKey(3, 'a b'));
});

// ── SO intake config (docs/designs/sales-order-intake.md) ──────────────────

function soMatchLine(overrides = {}) {
  return {
    customer_item_code: 'CQ-77',
    description: 'THEIR GRANOLA 10LB CASE',
    quantity: 3,
    unit: 'CASE',
    unit_price: 32.5,
    match_source: 'none',
    confidence: 0,
    product: null,
    candidates: [],
    case_size_lb: null,
    case_size_source: 'none',
    quantity_lb: null,
    ...overrides,
  };
}

const SO_PROD = { product_id: 9, name: 'Granola 10 LB', odoo_code: '1614',
  label_type: 'house', prior_sales: true };
const SO_PL_PROD = { product_id: 11, name: 'Setton Secret 25 LB', odoo_code: '70056',
  label_type: 'private_label', prior_sales: false };

test('soNormalizeMatchLine maps sales fields onto the shared shape', () => {
  const n = ERIntake.soNormalizeMatchLine(soMatchLine({
    case_size_lb: 10, case_size_source: 'product', quantity_lb: 30 }));
  assert.equal(n.vendor_description, 'THEIR GRANOLA 10LB CASE');
  assert.equal(n.lb_per_unit, 10);
  assert.equal(n.lb_source, 'case_size'); // 'product' → dies with a product change
  assert.equal(n.expected_qty_lb, 30);
  assert.equal(n.customer_item_code, 'CQ-77');
  assert.equal(n.unit_price, 32.5);
});

test('so alias/exact match arrives chosen with computed lb; fuzzy stays a suggestion', () => {
  const exact = ERIntake.soBuildReviewLine(soMatchLine({
    match_source: 'exact', confidence: 1, product: SO_PROD,
    case_size_lb: 10, case_size_source: 'product', quantity_lb: 30 }));
  assert.equal(exact.chosen.product_id, 9);
  assert.equal(exact.qty_lb, 30);
  assert.equal(exact.qty_lb_source, 'computed');
  const fuzzy = ERIntake.soBuildReviewLine(soMatchLine({
    match_source: 'fuzzy', confidence: 0.8, product: SO_PROD }));
  assert.equal(fuzzy.chosen, null, 'a fuzzy product is a suggestion, never a selection');
  assert.equal(fuzzy.suggested.product_id, 9);
  assert.equal(fuzzy.qty_lb, null);
});

test('so lb-unit line converts at 1 lb/unit; a pick computes and a re-pick restores', () => {
  const l = ERIntake.soBuildReviewLine(soMatchLine({
    unit: 'LB', quantity: 500, quantity_lb: 500, case_size_source: 'unit_is_lb' }));
  assert.equal(l.lb_per_unit, 1);
  assert.equal(l.lb_source, 'unit_is_lb');
  assert.equal(l.qty_lb, null, 'pounds wait for a product, exactly like ER lb lines');
  assert.equal(ERIntake.lineApprovable(l), false);
  ERIntake.applyProductPick(l, SO_PROD);
  assert.equal(l.qty_lb, 500, 'pick computes the document pounds');
  ERIntake.clearChosen(l);
  assert.equal(l.lb_per_unit, 1, 'unit_is_lb survives a product change');
  ERIntake.applyProductPick(l, { ...SO_PROD, product_id: 42 });
  assert.equal(l.qty_lb, 500, 're-pick restores the pounds');
  // The 1-lb display conversion must never be taught as an alias case size.
  ERIntake.applySaveAliasToggle(l, true);
  assert.equal(ERIntake.soApproveLinePayload(l).case_size_lb, null);
});

test('soMergeRematch preserves exclusions and agreeing picks like the ER merge', () => {
  const prev = ERIntake.soBuildReviewLine(soMatchLine());
  prev.include = false;
  ERIntake.applyProductPick(prev, SO_PROD);
  const merged = ERIntake.soMergeRematch([prev], [soMatchLine({
    match_source: 'exact', confidence: 1, product: SO_PROD,
    case_size_lb: 10, case_size_source: 'product', quantity_lb: 30 })]);
  assert.equal(merged[0].include, false, 'exclusion survives');
  assert.equal(merged[0].chosen.product_id, 9, 'agreeing pick survives');
  const disagree = ERIntake.soMergeRematch([prev], [soMatchLine({
    match_source: 'exact', confidence: 1, product: { ...SO_PROD, product_id: 99 },
    case_size_lb: 10, case_size_source: 'product', quantity_lb: 30 })]);
  assert.equal(disagree[0].chosen, null,
    'a result disagreeing with the user pick resets the line to unconfirmed');
  assert.equal(disagree[0].qty_lb, null);
});

// ── SO review follow-up (fix/so-intake-followup) ────────────────────────────

test('soMergeRematch preserveResolved: a PO-number re-match keeps human picks, re-matches only unresolved lines', () => {
  // Finding 1: editing the PO only re-runs the duplicate check — the customer
  // is unchanged, so a line the user resolved must come through untouched.
  const picked = ERIntake.soBuildReviewLine(soMatchLine());          // fuzzy/none → human picks
  ERIntake.applyProductPick(picked, SO_PROD);
  ERIntake.applyLbPerUnitChange(picked, 12);                          // typed case size → 36 lb
  const open = ERIntake.soBuildReviewLine(soMatchLine({ description: 'STILL UNKNOWN' }));
  ERIntake.lockLines([picked, open]);
  assert.equal(picked.qty_lb, 36);

  // The fresh result for line 0 is still fuzzy (no chosen product) — the old
  // merge would have wiped the pick; line 1 now carries a suggestion.
  const fresh = [
    soMatchLine({ match_source: 'fuzzy', confidence: 0.7, product: SO_PROD }),
    soMatchLine({ description: 'STILL UNKNOWN', match_source: 'fuzzy', confidence: 0.6, product: SO_PL_PROD }),
  ];
  const merged = ERIntake.soMergeRematch([picked, open], fresh, { preserveResolved: true });
  assert.equal(merged[0], picked, 'the resolved line object is kept whole');
  assert.equal(merged[0].chosen.product_id, 9, 'human pick survives the PO re-match');
  assert.equal(merged[0].match_source, 'chosen');
  assert.equal(merged[0].lb_per_unit, 12, 'typed conversion survives');
  assert.equal(merged[0].qty_lb, 36, 'pounds survive');
  assert.equal(merged[0].matching, false, 'and the line is unlocked');
  assert.equal(merged[1].chosen, null, 'unresolved line is re-matched');
  assert.equal(merged[1].suggested.product_id, 11, 'and takes the fresh suggestion');
  assert.ok(!merged[1].matching, 'fresh line is not locked');

  // Default (customer re-match) behaviour is unchanged: a disagreeing result
  // still resets the pick.
  const reset = ERIntake.soMergeRematch([picked, open], fresh);
  assert.equal(reset[0].chosen, null, 'a customer re-match still resets a pick the result does not confirm');
});

test('applyProductPick pulls the product-master case size so cases→lb converts without typing', () => {
  // Finding 2: picker results and suggestion chips carry case_size_lb.
  const l = ERIntake.soBuildReviewLine(soMatchLine({ quantity: 3, unit: 'CASE' }));
  assert.equal(l.lb_per_unit, null);
  ERIntake.applyProductPick(l, { ...SO_PROD, case_size_lb: 10 });
  assert.equal(l.lb_per_unit, 10, 'case size comes from the product master');
  assert.equal(l.lb_source, 'case_size', 'tagged as product-derived so it dies with a product change');
  assert.equal(l.qty_lb, 30, '3 cases × 10 lb');
  assert.equal(l.qty_lb_source, 'computed');
  assert.deepEqual(ERIntake.soApproveLinePayload(l), {
    product_id: 9, quantity_lb: 30, quantity: 3, unit: 'CASE', case_size_lb: 10,
    unit_price: 32.5, customer_item_code: 'CQ-77', customer_description: 'THEIR GRANOLA 10LB CASE', save_alias: true,
  });

  // Change → re-pick a product with a different case size: the old product's
  // conversion is dropped and the new one's is applied.
  ERIntake.clearChosen(l);
  assert.equal(l.lb_per_unit, null);
  ERIntake.applyProductPick(l, { ...SO_PL_PROD, case_size_lb: 25 });
  assert.equal(l.lb_per_unit, 25);
  assert.equal(l.qty_lb, 75);

  // A product with no case size falls back to what the user typed.
  const typed = ERIntake.soBuildReviewLine(soMatchLine({ quantity: 2, unit: 'CASE' }));
  ERIntake.applyLbPerUnitChange(typed, 8);
  ERIntake.applyProductPick(typed, { ...SO_PROD, case_size_lb: null });
  assert.equal(typed.lb_per_unit, 8, 'typed case size kept when the product has none');
  assert.equal(typed.lb_source, 'manual');
  assert.equal(typed.qty_lb, 16);

  // A typed case size is never overwritten by the master (customer packs differ).
  const typedFirst = ERIntake.soBuildReviewLine(soMatchLine({ quantity: 2, unit: 'CASE' }));
  ERIntake.applyLbPerUnitChange(typedFirst, 8);
  ERIntake.applyProductPick(typedFirst, { ...SO_PROD, case_size_lb: 10 });
  assert.equal(typedFirst.lb_per_unit, 8, 'explicit manual conversion wins over the master');

  // An lb-unit line keeps its 1-lb identity; the product case size is irrelevant.
  const lbLine = ERIntake.soBuildReviewLine(soMatchLine({ quantity: 500, unit: 'LB', case_size_source: 'unit_is_lb', quantity_lb: 500 }));
  ERIntake.applyProductPick(lbLine, { ...SO_PROD, case_size_lb: 10 });
  assert.equal(lbLine.lb_per_unit, 1);
  assert.equal(lbLine.qty_lb, 500);

  // ER picks carry no case_size_lb — their behaviour is unchanged.
  const er = ERIntake.buildReviewLine(matchLine());
  ERIntake.applyProductPick(er, PROD);
  assert.equal(er.lb_per_unit, null);
  assert.deepEqual(er.chosen, PROD);
});

test('soPriceBasis: per-case on cases, per-lb on lb, unclear otherwise (ruling 5)', () => {
  assert.equal(ERIntake.soPriceBasis('CASE'), 'per_case');
  assert.equal(ERIntake.soPriceBasis(' cs. '), 'per_case');
  assert.equal(ERIntake.soPriceBasis('LB'), 'per_lb');
  assert.equal(ERIntake.soPriceBasis('pounds'), 'per_lb');
  assert.equal(ERIntake.soPriceBasis('EA'), 'unclear');
  assert.equal(ERIntake.soPriceBasis(null), 'unclear');
  assert.equal(ERIntake.soPriceBasis(''), 'unclear');
});

test('soApproveLinePayload sends the price only when the basis is unambiguous', () => {
  const cases = ERIntake.soBuildReviewLine(soMatchLine({
    match_source: 'exact', confidence: 1, product: SO_PROD,
    case_size_lb: 10, case_size_source: 'product', quantity_lb: 30 }));
  assert.equal(ERIntake.soApproveLinePayload(cases).unit_price, 32.5);
  const unclear = ERIntake.soBuildReviewLine(soMatchLine({
    unit: 'EA', match_source: 'exact', confidence: 1, product: SO_PROD }));
  ERIntake.applyQtyLbOverride(unclear, 30);
  assert.equal(ERIntake.soApproveLinePayload(unclear).unit_price, null,
    'unclear basis → price not stored');
});

test('soApproveLinePayload teaches the conversion only when it explains the pounds', () => {
  const l = ERIntake.soBuildReviewLine(soMatchLine({
    match_source: 'alias', confidence: 1, product: SO_PROD,
    case_size_lb: 10, case_size_source: 'alias', quantity_lb: 30 }));
  assert.equal(l.save_alias, true);
  let p = ERIntake.soApproveLinePayload(l);
  assert.equal(p.case_size_lb, 10);
  assert.equal(p.quantity_lb, 30);
  assert.equal(p.customer_item_code, 'CQ-77');
  assert.equal(p.customer_description, 'THEIR GRANOLA 10LB CASE');
  // A direct lb override breaks the conversion → case_size_lb must not teach.
  ERIntake.applyQtyLbOverride(l, 35);
  ERIntake.applySaveAliasToggle(l, true);
  p = ERIntake.soApproveLinePayload(l);
  assert.equal(p.case_size_lb, null);
  assert.equal(p.quantity_lb, 35);
});

test('applyUnitPriceChange edits and clears the price, locked while matching', () => {
  const l = ERIntake.soBuildReviewLine(soMatchLine());
  ERIntake.applyUnitPriceChange(l, '40');
  assert.equal(l.unit_price, 40);
  ERIntake.applyUnitPriceChange(l, '');
  assert.equal(l.unit_price, null);
  ERIntake.applyUnitPriceChange(l, '-3');
  assert.equal(l.unit_price, null);
  l.matching = true;
  ERIntake.applyUnitPriceChange(l, '55');
  assert.equal(l.unit_price, null, 'locked line refuses edits');
});

test('soPrivateLabelWarning: warn only on a known first sale (rulings 2/3)', () => {
  const warn = ERIntake.soBuildReviewLine(soMatchLine());
  ERIntake.applyProductPick(warn, SO_PL_PROD);
  assert.equal(ERIntake.soPrivateLabelWarning(warn), true);
  const ok = ERIntake.soBuildReviewLine(soMatchLine());
  ERIntake.applyProductPick(ok, { ...SO_PL_PROD, prior_sales: true });
  assert.equal(ERIntake.soPrivateLabelWarning(ok), false);
  const unknown = ERIntake.soBuildReviewLine(soMatchLine());
  ERIntake.applyProductPick(unknown, { product_id: 7, name: 'ER pick' });
  assert.equal(ERIntake.soPrivateLabelWarning(unknown), false,
    'unknown label/prior_sales (ER picks) never warns');
});
