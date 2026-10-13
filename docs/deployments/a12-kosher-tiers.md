# A12 Classic #9 kosher tiers

Builder branch `feat/kosher-tier`; Claude Code reviews. No merge or production deployment.
Migration 075 was selected after inspecting all open PR filenames on 2026-10-13:
073 is reserved by #96, 068 by #90, 066 by #91; none reserve a number after 074.

Apply `075_kosher_tiers.sql` explicitly to guarded **staging only**, as app/table
owner via port 5432 in a transaction, with search_path=public, lock_timeout=5s,
statement_timeout=60s. No startup migration. The migration updates product tiers
and labels, leaves formula 284 unchanged, and never reclassifies historical lots.
Counted extra-kosher stock is loaded through the owner-approved cutover reset under
products 283/284 and 285–290 (design §10.1 step 1a); regular-product lots are never retagged.

A9 integration: the receipt draft/response and transaction/lot objects carry
`kosher_tier`; the shift-summary contract already defines `EXTRA_KOSHER` with
`refs.kosher_tier`. Held makes and found tickets have `status=awaiting_approval` and belong under
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

- **Merge order (Michael, 2026-10-13): fixed PR #102 (A9) merges FIRST.**
  This fix round rebases #101 onto `origin/main` at `a5b0ff6` (F1 merged).
  PR #102 is still open; after it lands, #101 must rebase again and implement
  the summary integration described in [Codex's #102 review](https://github.com/sevenwells72/factory-ledger/pull/102#issuecomment-6149165326)
  before #101 can merge. Keep contract changes minimal until then.
- A9 follow-up in #101: wire the five stored kosher fields and
  `EXTRA_KOSHER` / `KOSHER_DOWNGRADE` flags into the report and clients;
  show held make/found receipts under NOT ENTERED YET?, with owner/expiry.
  Use immutable lot evidence for ship/adjust rows whose transaction tier
  defaults regular; retain preparer attribution separately from the attester.
  Reconcile contract §1.3 with the kosher hashing prose: include stored tier,
  approving actor ID/time and downgrade evidence, exclude labels/names, live
  catalog data and held expectations. Test held → approved using the same
  receipt, tier/downgrade flags, catalog/label hash stability and stale confirm
  after approval. Preserve 068/075/076 and both receipt null guards during
  integration. Weekly downgrade rendering remains separate weekly-view work.
  Full integration checklist: FOLLOWUPS P1.22.
- Extra-kosher lot creation: receive of batches 283/284 is refused for every
  role; extra-kosher finished receive also refuses and directs callers to
  pack/found. Found of any extra-kosher product always holds for a separate
  owner approval, even when prepared by the owner; no lot/tier exists until
  approval. Direct extra-kosher make/found requires tickets; batch pack outputs
  and any reassign/merge touching extra-kosher stock are refused. Extra-kosher
  finished packs require extra-kosher sources. No retag route exists.
- A10: retire legacy `POST /ship` (misleading preview; commit is tier-safe) and
  retire/tier-gate the archived one-shot direct-SQL poster
  `scripts/inv_recon_post_2026_08_17.py`; **do not rerun it after 075**. Its
  generic lot creator and phase 3.A regular 90002 → extra 70013 conversion
  bypass the HTTP gates. See FOLLOWUPS P1.22.
- F1/D2: HTTP 202 is HELD with `posted=false`, never Recorded. Owner decisions
  use existing `/exceptions/{id}/approve` or `/reject`, from the owner's identity;
  no PIN or impersonation on the floor device. A11 is not a dependency.
- **Production deploy precondition (Michael, 2026-10-13): apply
  `075_kosher_tiers.sql` to production as the table-owner role (`postgres`)
  before deploying code.** Use port 5432 and the explicit guarded transaction
  above; no startup migration runs for 075. This is an operator deployment
  instruction; A12 review fixes do not apply it or change production.
- The migration creates `ticket_attestations` and `lot_events` with **RLS enabled
  and no policies**, and revokes PUBLIC access. App database access must run
  as the table owner (`postgres`); ordinary API roles have no policy granting
  access to these tables.
- Applying 075 in production also renames **products 283–290**: 283 becomes
  `Batch Classic Granola #9 — EXTRA-KOSHER (SS)`, 284 becomes
  `Batch Classic Chocolate Chip Granola #9 — EXTRA-KOSHER (SS)`, and finished
  products 285–290 retain their SS Classic #9 labels with
  ` — EXTRA-KOSHER (SS)` appended (removing any old `(Kosher Ignition)` text).
  Formula 284 and the Aug 12 historical records are preserved.
- Cutover §10.1 step 1a loads counted extra-kosher stock under 283/284 and
  285–290 through the reset with owner approval and count-sheet/paper-log
  evidence. Historical regular lots remain regular; there is no retag route.
  The current count tools list these products separately but cannot create
  new lots or store kosher approval; count-workstream changes are recorded
  in FOLLOWUPS P1.21. No count-tool change or cutover action ran here.

## Review fixes verified — 2026-10-13

Michael's decisions from PR #101 review comment 6149065092 are implemented:
extra batch receive refuses every role; all extra found tickets (including owner
preparers) wait for a separate owner approval, and the new lot gets its tier and
owner/time attestation only in the approved posting transaction. Photos cannot
release a kosher hold. Legacy extra receive/make/found, extra batch pack output,
and reassign/merge paths are closed; extra finished packs remain source-tier safe.
Count cutover uses products 283/284 and 285–290 with owner approval, never retagging.
The count-workstream and A10 gaps are recorded in FOLLOWUPS P1.21/P1.22.

Validation on a fresh UTF-8 PostgreSQL 17 database in a dedicated temporary local
cluster (`127.0.0.1:57612`), rebuilt from `tests/schema/schema.sql` with migration
075 before the final run:

- Final full Python suite: **2,550 passed**, zero failures or skips (37.70 s).
- Node suite: **69 passed**, zero failures or skips.
- **93 new regression cases**: corrected tests against a temporary source archive
  of reviewed head `928b601` produced **81 failed / 12 passed**; all pass after
  fixes. The passing baseline cases cover existing authorization/staleness gates.
- Focused kosher, review, correction-hold and shipment-evidence suites:
  **263 passed**. The full run covers migration idempotence and approval races.
- `git diff --check` passes. Independent read-only path review found no remaining
  HTTP extra-kosher creation bypass. No staging acceptance was repeated in this
  fix round; the earlier staging receipt and test logs above remain historical.

Production rollout is still an operator step: apply 075 as `postgres` before code,
verify `ticket_attestations` and `lot_events` are owned by `postgres` with RLS on
and zero policies, and account for the product 283–290 renames. No production
access, other worktree changes, deployment or merge occurred.

## Post-F1 rebase verification — 2026-10-13

Rebased source `4795107` includes current `origin/main` `a5b0ff6` (F1 merged),
all original A12 guards, the F1 route hooks and both migrations 068/075.
A new isolated UTF-8 PostgreSQL 17 database was built from the current schema
plus `pg_trgm`; **2,629 Python passed** (41.65 s), **75 Node passed**, zero
failures/errors/skips. All **93 A12 regressions pass**; the independently rerun
`928b601` baseline has **81 failed / 12 passed**, zero errors/skips.
Evidence: `a12-review-rebase-tests.json`. The local cluster was stopped and its
temporary database removed. Stored historical test/staging evidence above is
unchanged. All project and regression changelog history from both heads was retained;
the global iCloud log remains outside this worktree and was not written.

Independent final review confirms F1 uses the guarded prepare/commit routes and
preserves HTTP 202 holds. PR #102 is still open: its first merge and the subsequent
#101 summary integration in FOLLOWUPS P1.22 remain required before #101 merges.
Count loading and A10 legacy retirement remain in their respective workstreams.
No production/staging access, count-tool edit, other-worktree edit, deploy or merge.
