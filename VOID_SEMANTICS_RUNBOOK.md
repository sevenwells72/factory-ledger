# Void Semantics Migration — Post-Deploy Cleanup Runbook

Branch: `fix/void-semantics`. Status convention after this branch deploys:
**`status='posted'` is the single source of truth for all balance math.**
Voided transactions count nowhere; `POST /void/{id}` flips status and posts
NO reversal transaction.

## ⚠️ Critical sequencing

1. **Deploy AFTER floor operations end for the day.** Between the deploy and
   step 3 below, posted-only balances expose the 12 historical reversal
   transactions: lot **610 reads −3,400 lb** and lot **617 reads −1,000 lb**
   (negative!), lots 612–615 read thousands of lb low, and lot 582 reads
   −2,907. Negative/low availability can block `/make`, `/pack`, and `/ship`
   during this window.
2. **Merge → Railway auto-deploys from main.** Confirm the new code is live:
   `POST /void/{id}` on any test attempt must return
   `"reversal_transaction_id": null`. (The cleanup script also self-checks
   this and aborts if the old code answers.)
3. **Immediately run** `FACTORY_API_KEY=... ./scripts/cleanup_void_reversals.sh`.
   ⚠️ **This script must ONLY run against the new code.** Under the old code
   it would insert 12 NEW posted reversals and make every number worse.
4. The script verifies all 16 affected lots against the expected end state
   and exits non-zero on any mismatch. **The end state equals the
   pre-migration dashboard numbers exactly — nothing visible should shift.**

## What the script voids, in order, and the expected posted-only balance after each step

| Step | Void reversal # | (was reversal of) | Lot(s) affected | Posted-only balance after this step |
|---|---|---|---|---|
| 1 | 470 | #469 (pack) | 314 → **0**, 293 → **0** | lots 314/293 healed |
| 2 | 864 | #863 (make) | 582 → **0**; ingredients 33 → 0, 34 → 801.8, 337 → 0, 566 → 0, 500 → 0, 48 → 1372.12, 501 → 0 | lot 582 + 7 ingredient lots healed |
| 3 | 932 | #911 (adjust) | 610 → **6,600** | lot 610 no longer negative |
| 4 | 933 | #913 (adjust) | 612 → −1,900 | half-healed |
| 5 | 934 | #924 (adjust) | 612 → **0** | lot 612 healed (the −3,800 was 100% void artifact — no real adjustment needed) |
| 6 | 935 | #914 (adjust) | 613 → 35,075 | |
| 7 | 936 | #918 (adjust) | 617 → **9,000** | lot 617 no longer negative |
| 8 | 937 | #925 (adjust) | 613 → **36,150** | lot 613 healed |
| 9 | 938 | #915 (adjust) | 614 → 8,200 | |
| 10 | 939 | #926 (adjust) | 614 → **11,500** | lot 614 healed |
| 11 | 940 | #916 (adjust) | 615 → 6,600 | |
| 12 | 941 | #927 (adjust) | 615 → **10,000** | lot 615 healed |

All 12 pairs were verified line-by-line against prod on 2026-06-09 (read-only
session): every reversal's lines exactly negate its voided original.

Orphan voided transactions #80, #83, #84, #177 ("ghost make" audit voids,
single 0.0-lb line each) have no reversals and need no cleanup — they are
numerically inert under any convention.

## Re-run safety

The new `POST /void/{id}` returns HTTP 400 for an already-voided transaction
and changes nothing (covered by `tests/test_void_semantics.py::
test_double_void_fails_cleanly_and_changes_nothing`). If the script stops
partway, remove the already-completed IDs from `REVERSAL_IDS` and re-run, or
just re-run as-is and treat the 400 on the first IDs as the stop point to
trim from.

## If a verification mismatch appears

Do NOT adjust inventory immediately. A mismatch means real floor activity
occurred between the simulation (2026-06-09) and the cleanup — recompute the
expected value from the ledger (posted-only) before touching anything.
