# Whole-reset proof on hosted staging — 2026-10-14 (O-3)

**Plain English first.** The cutover posts the reset one area group at a time. Each group has its own signed plan and executor journal. The new `verify_reset.py --whole-reset` proves all groups together. We ran three small groups one after another on the **deployed** FastAPI-staging. The proof said **FULL RESET SCOPE RECONCILED** (exit 0). Then we broke it on purpose, twice, and it named the gap each time (exit 2): once with a group left out of the manifest, once with an unsigned owner count post. No production read or write. The synthetic owner is **not** Michael.

## What ran

| Item | Value |
|---|---|
| When | 2026-10-14 12:39–12:41 EDT |
| API | `https://fastapi-staging-production-dd7b.up.railway.app` (hosted). Railway FastAPI-staging deployment `a91e3778`, SUCCESS from `d962a1f`. Later main merges changed no API code, so they skipped. |
| Database | guarded staging DB (`staging_safety.assert_staging_database`, port 5432); synthetic products ≥ 10⁹ |
| Script | `scripts/count_rehearsal_staging.py --hosted-groups` (new mode on this branch) |
| Receipt | `docs/deployments/whole-reset-proof-hosted-staging-receipt.json`, reference `STG-CNT-36A1AD` |

The files were cumulative, as on the real morning. Each group's count, coverage and review copy held every area transcribed so far. Nothing was edited after it was signed. The final inputs are the last group's signed copies.

| Group | Areas | Approved | Posts | What it proved |
|---|---|---|---|---|
| Dry storage | Dry storage | oats, sugar | ADJ-261014-009 (−5, estimated), ADJ-261014-010 (**−680**, owner hold released under the signed list) | ordinary + >500 lb hold |
| Warehouse, Cooler | Warehouse + Cooler | granola cases, extra-kosher batch | ADJ-261014-011 (−25 lb, 39 cases), FND-261014-205 (+300 lb NEW lot, attested) | **multi-area lot**: the extra-kosher lot was counted in both areas (180 + 120). It was reset once, by this group. It is held in the Office group's plan. |
| Office | Office | ingredient 1 | none | **no-change group**: counted at book, signed, no journal (runbook step 18 no-change branch) |

## The proof

| Manifest | Exit | Result |
|---|---|---|
| all three groups | **0** | FULL RESET SCOPE RECONCILED. 5/5 rehearsal-scope products; multi-area lot `STG-CNT-36A1AD-EK-1` reset by Warehouse+Cooler, held in Office |
| Office group left out | 2 | `MISSING_PRODUCT` 1000000177: has stock, but no signed group approves it |
| after an owner `physical_count` post (ADJ-261014-012, +1 lb on oats, through the hosted API, outside every plan) | 2 | `POST_OUTSIDE_APPROVED_PLAN` TX 1000000549, plus `FINAL_HOLD`: the movement sits within 30 minutes of the count and is unclassified |

Each proof took about 1 s (one read-only staging snapshot). The posting pauses took 18 s and 20 s, the same as the one-area rehearsal.

`rehearsal_scope_product_ids` limited the scope to the five rehearsal products, because the rest of the staging catalog was not counted. The tool refuses that key unless the target is staging. Production uses `expected_reset_products: 178`.

## Left behind on staging

Three synthetic sets: `STG-CNT-785330`, `STG-CNT-947168` and `STG-CNT-36A1AD`. The first two came from earlier runs of the same mode on this branch. Run 1 failed only in its final evidence read-back; run 2 showed the multi-area listing used the sheet's lot spelling. Both are fixed. All three sets have products, lots, receipts and the deliberate +1 lb post. Actors are deactivated and the staging owner key file is removed. Harmless; later staging tests of those ids must expect them.

## Fix round 1 re-run — STG-CNT-DCF5EB (Codex review 6173615593)

The review found six ways to get a false exit 0. All are fixed, each with a regression test that fails on 3c5195f and passes now:
- receipts with bogus ids or unknown outcomes;
- a second held ticket beside a posted reset;
- `physical_count` posts by other actors, or without a ticket;
- edits to duplicate-excluded rows;
- zero-stock products with no approving group;
- the step-20 journal paths.

The run was repeated on the **deployed** FastAPI-staging with three groups. The Office group now also signs for a zero-stock product (batch, no lot, coverage confirmed). Every proof ran against a saved read-only snapshot.

| Case | Exit | Gaps |
|---|---|---|
| complete (3 groups, 6 products) | **0** | FULL RESET SCOPE RECONCILED |
| Office group left out | 2 | `MISSING_PRODUCT` ×2 (one zero-stock, one with stock) |
| Office signed without the zero-stock product | 2 | `MISSING_PRODUCT` (zero stock) |
| forged `refused` receipt, bogus `ADJ-999999-999` / TX, chain re-hashed | 2 | `INVALID_RECEIPT` ×2, `RECEIPT_CONTRADICTS_FL` |
| duplicate-ticked oats row added after signing, excluded by an owner answer | 2 | `MISSING_OBSERVATION` |
| owner `physical_count` post ADJ-261014-016 (+1 lb) | 2 | `POST_OUTSIDE_APPROVED_PLAN` + `FINAL_HOLD` |
| **floor actor (Arturo)** `physical_count` post ADJ-261014-017 (+2 lb) | 2 | adds `POST_OUTSIDE_APPROVED_PLAN` "by Arturo" + `FINAL_HOLD` |
| floor −505 lb count adjust, held for the owner (hold #254) | 2 | adds `OPEN_HOLD` |
| after the owner declined hold #254 | 2 | `OPEN_HOLD` gone; the two posts remain |

Group posts: ADJ-261014-013/014 (Dry storage, incl. the −680 lb held-and-released change), ADJ-261014-015 and FND-261014-206 (Warehouse+Cooler, the multi-area extra-kosher lot reset once), none for Office.

**Raw artifacts for the reviewer:** `docs/deployments/whole-reset-proof-evidence/STG-CNT-DCF5EB/` holds the signed previews, approvals, inputs, manifests, redacted journals (ticket token only), scoped snapshots and proof outputs, plus `SHA256SUMS-originals.txt`. All 9 cases replayed identically from the scoped snapshots. A scan found no API key, ticket token or database URL.

Also left on staging: the STG-CNT-DCF5EB synthetic set, including the +1/+2 lb posts and declined hold #254. Actors are deactivated.
