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

## Fix round 2 re-run — STG-CNT-08370B (Codex re-check 6174135183)

The re-check found two more gaps. Both are fixed, each with a regression test that fails on b8da383 and passes now:
- **Fabricated hold identity.** A re-chained journal could name an owner hold FL does not have (exception `424242` or `-1`, or the wrong hold kind) on a `held` receipt or an `approve` intent, and still get exit 0. `reconcile_tickets` (shared by the executor preflight and the proof) now binds each named hold to FL's exception: id, kind, ticket, product, lot (`lot_id`, or `lot_code` for a new extra-kosher lot), a kind the signed plan lists, `Owner approves held entries: Y`, and an `approved` release by the named actor. A mismatch is `HOLD_NOT_IN_FL`. The scope snapshot query now also reads each hold's `lot_id` and `detail->>'lot_code'`. The legitimate retry (held → approve refused → approve ok) still proves.
- **Zero-stock-only apply.** A group approving only zero-stock products crashed the confirmed `--apply` in the read-back (`IndexError`). The read-back now uses fixed columns. The apply posts nothing, writes no journal and a header-only `readback.csv`, then runs its verification; the group is listed as `no_change: true`.

Hosted staging, four groups this time. The new **Zero-stock sign-off** group (area Office) approves one product with no lot and runs the real confirmed `--apply`: no API write, no journal, header-only read-back, the product unheld in its verification (area-scoped exit 2 as for the other groups).

| Case | Exit | Gaps |
|---|---|---|
| complete (4 groups, 7 products) | **0** | FULL RESET SCOPE RECONCILED |
| Dry storage journal: held receipt names exception #424242 | 2 | `HOLD_NOT_IN_FL` "names exception #424242, which FL does not have; FL has #258 LARGE_CORRECTION resolved on ticket 1187", plus the group's posts left unproved |
| Dry storage journal: approve intent names KOSHER_ATTESTATION | 2 | `HOLD_NOT_IN_FL` "FL exception #258 is LARGE_CORRECTION on ticket 1187", plus the group's posts left unproved |
| Office group left out | 2 | `MISSING_PRODUCT` ×2 |
| Office signed without its zero-stock product | 2 | `MISSING_PRODUCT` (zero stock) |
| forged `refused` receipt | 2 | `INVALID_RECEIPT` ×2, `RECEIPT_CONTRADICTS_FL` |
| duplicate row added after signing | 2 | `MISSING_OBSERVATION` |
| owner post ADJ-261014-026 / floor post ADJ-261014-027 | 2 | `POST_OUTSIDE_APPROVED_PLAN` + `FINAL_HOLD` |
| floor −505 lb held (hold #260), then declined | 2 | `OPEN_HOLD`, then gone |

Group posts: ADJ-261014-023/024 (Dry storage, incl. the −680 lb hold #258 released), ADJ-261014-025 and FND-261014-208 (Warehouse+Cooler); none for Office or Zero-stock sign-off. Evidence: `docs/deployments/whole-reset-proof-evidence/STG-CNT-08370B/` (87 files); all 11 cases replayed identically from the scoped snapshots; the secret scan was clean.

Also left on staging: synthetic sets STG-CNT-780841 (a first round-2 run whose "zero-stock unapproved" case read the wrong approval file; the rehearsal script is fixed and that evidence was not kept) and STG-CNT-08370B, with their +1/+2 lb posts and declined holds. Actors are deactivated.
