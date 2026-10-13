# A12 Classic #9 kosher tiers

Builder branch `feat/kosher-tier`; Claude Code reviews. No merge or production deployment.
Migration 075 was selected after inspecting all open PR filenames on 2026-10-13:
073 is reserved by #96, 068 by #90, 066 by #91; none reserve a number after 074.

Apply `075_kosher_tiers.sql` explicitly to guarded **staging only**, as app/table
owner via port 5432 in a transaction, with search_path=public, lock_timeout=5s,
statement_timeout=60s. No startup migration. The migration updates product tiers
and labels, leaves formula 284 unchanged, and never reclassifies historical lots.
Paper-log extra-kosher lots are reconciled on cutover day under design §10.1 step 1a.

A9 integration: the receipt draft/response and transaction/lot objects carry
`kosher_tier`; the shift-summary contract already defines `EXTRA_KOSHER` with
`refs.kosher_tier`. Held makes have `status=awaiting_approval` and belong under
NOT ENTERED YET?; downgrades carry `transaction_kosher_downgrade=true`.

## Verification — 2026-10-13

Full Python suite on freshly created local `fl_a12_verified`: **2,457 passed**,
zero failures/skips (39.53 s). Node suite: **69 passed**, zero failures/skips.
Focused final source/concurrency suite: 422 passed. Migration 075 reapplication,
all catalog mappings, formula preservation, approval races and legacy-route
bypass checks are included. The Python suite uses the pinned Python 3.12 runtime.
Logs: `a12-tests-python.log`, `a12-tests-node.log`; `git diff --check` passed.

Guarded staging migration 075 applied at **2026-10-13 11:03:31 ET** via port 5432.
Acceptance uses this branch's local TestClient against the staging DB, without
FastAPI lifespan/startup, hosted deployment, or any other lane's service changes.
Retained evidence: `a12-staging-receipt.json`, reference `STG-A12-46454C41E879`.

1. Floor actor prepares extra-kosher make, ticket 64, receipt reference
   **MK-261013-001** → HTTP 202 HELD, `posted=false`, `success=false`, exception 12,
   zero transactions. Seven-day expiry: Oct 20 at 11:03:35 ET.
2. Synthetic owner approves from a separate authenticated identity → the SAME
   **MK-261013-001**, transaction 1000000055, output lot 1000000046,
   `B26-1013-001`; preparer actor 1000000050 stays entered_by, owner actor
   1000000051/time is stored on transaction, lot and ticket attestation.
3. Extra finished pack → **PK-261013-001**, transaction 1000000056,
   lot 1000000047, **1 case / 5 lb**, `kosher_tier=extra_kosher`.
4. Extra batch packed as regular → **PK-261013-002**, transaction 1000000057,
   downgrade flag and output lot event 1. A second held make was declined
   (ticket 67 / exception 13), with zero posts. Floor approval returns 403.

All examples are synthetic; they do not claim Michael attested a real batch.
Temporary floor/owner keys existed only in memory; both actors were deactivated.
The migration and rows were written to staging only. Production was not queried
or changed. No service was deployed; PR #101 remains draft and unmerged.

## Integration and rollout

- A9: consume `kosher_tier`, `kosher_label`, attesting actor/time and
  `transaction_kosher_downgrade` from stored receipts; use `EXTRA_KOSHER` and
  `KOSHER_DOWNGRADE` flag refs from the updated contract. List held makes under
  NOT ENTERED YET? and downgrades on the weekly view. A9 implementation is in
  its own lane; this PR supplies the JSON fields and contract.
- F1/D2: HTTP 202 is HELD with `posted=false`, never Recorded. Owner decisions
  use existing `/exceptions/{id}/approve` or `/reject`, from the owner's identity;
  no PIN or impersonation on the floor device. A11 is not a dependency.
- Before any later merge/deploy, the operator must authorize and apply 075 to
  the target database as app/table owner. Builder performed staging only.
  New columns are required by the app; no startup migration runs for 075.
- Cutover §10.1 step 1a already specifies paper-log reconciliation. Historical
  lots default regular; tag only evidenced paper-log lots in the one-time
  owner-controlled cutover data step, with count-sheet photo/receipt evidence.
  Ordinary lot-tier edits remain forbidden. Formula 284 and Aug 12 records
  are preserved. This request adds/checks the checklist; no cutover action ran.
