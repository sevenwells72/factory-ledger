# Receiving and hand additions — migration 098

**DRAFT — not merged. Cross-reviewed 2026-11-20 (code approved); replayed onto main 2026-11-24 after the shared-key shutdown (PR #234) went live.**

The final owner decision keeps reason-list policy in code, gated by marker 098. The migration does not edit or add columns to `correction_reasons`; applying it under old code leaves that catalog and old reason validation unchanged. Removing the marker restores the prior inventory/reason policy. The shared-key shutdown is not part of this PR any more: it shipped separately as PR #234 and has been live in production since 2026-11-24 (`docs/named-actor-writes.md`).

Nothing from this PR is applied to staging or production. Each staging or production step needs separate owner approval; production also needs a verified backup first and verification afterwards. Keep this PR draft until the owner approves the merge.

Each purchased-ingredient delivery gets its own Receive lot. Effective stock-in prevents receiving into an old lot; voided receipts do not. Adjust and existing-lot Found cannot increase the effective hand-entry sum since migration 098 above zero. Count-to-zero may use strictly less than the smaller of the effective receipt pack weight and 100 lb, cumulatively per lot. Product pack weights are never used. Lifetime allowance rows survive voids. Wrong-lot moves are one transaction with two opposite lines, excluded from give-back on both lots, and may open a shortage on the used lot. No tolerance or owner override.

## Switches and deploy order

1. Complete review and local Python, JavaScript and real-browser verification. Keep the PR draft.
2. With separate approval, deploy code to staging, apply 098 in one transaction as table owner and verify. Marker absence preserves old inventory rules. Applying 098 under old code leaves the reason catalog unchanged.
3. Before any approved production change, run `~/Backups/fl/fl-backup-and-verify.sh` and verify its restore and key checks. Approval for a merge does not authorize a migration or deploy.
4. Deploy approved code. Merging deploys the API (Railway) and the dashboard (Netlify): the merge commit must not contain `[skip netlify]`, or the dashboard changes will not publish. With the marker absent, no inventory rule changes yet.
5. Confirm the new dashboard is live before the next step. Migration 098 is the switch: applying it while the old dashboard is still being served would turn the new rules on under screens that do not explain them.
6. Run the go-live checks below, read-only.
7. Separately apply migration 098 with `ON_ERROR_STOP`, transaction-local search_path, lock timeout and statement timeout. Verify marker, unchanged reason catalog and both new RLS tables.

### Go-live checks before 098 (read-only)

- **No adjust or Found ticket is waiting for approval.** An addition held under the old rules cannot be approved once the marker is on: approve answers 409 `TICKET_STALE` / `NO_STOCK_BY_HAND` and the balance does not change; reject still works. That is intended (no owner override). Check `write_tickets` for `status = 'awaiting_approval'` with `action IN ('adjust', 'found')` and settle each one first.
- **List ingredient lots with an open shortage** (`exceptions` with `kind = 'SHORTAGE'` and `status` open or escalated, joined to lots of ingredient products). After 098, resolving a shortage as "Counted" adds stock by hand, so it stops working for those lots unless the lot is short by less than one pack. Decide each one before the switch, or expect to resolve it by a receipt or a paired move.

### Shared key

The shared master key is already read-only in production (PR #234). GETs and `POST /resolve` and `/products/resolve` remain available; a named actor key with role owner reaches every route. Emergency `SHARED_KEY_WRITES=allow` restores the old shared-key reach. Setting it is a production credential change requiring explicit owner approval, a verified backup and verification. Never set it as a test-suite default.

## Verification

Use the new HTTP tests as the staging checklist: existing received lot rejected; new/placeholder and voided receipt allowed; both count routes refuse exactly one pack, cumulative allowance exhausted and missing pack weight; paired move has two lines, one receipt and one shortage if needed. Verify shared-key write refusal before any row is written, named-owner route reach, and floor denial. Check receipt, history, shift roll-forward and nightly email show both move legs. Reattach photos when receiving again.

## Rollback

Fastest first. Each step is a production change: owner approval for that step, a verified backup before it and verification after.

1. **Turn the new inventory rules off: run the 098 down file alone.** It deletes only the marker, in one statement, and the old rules apply again at once with no deploy and no code revert. It keeps lifetime count usage and move identities as audit evidence and never changes the reason catalog. No ledger record is edited or deleted. Preserve the 096 marker, reasons, alert outbox and held-ticket compatibility.
2. **A code revert is optional afterwards.** With the marker absent the new code behaves like the old code for inventory rules, so there is no need to revert in a hurry. If the code is reverted, do it after the marker is down, not before.
3. **Putting the marker back starts a new window.** The give-back sum and the under-one-pack count allowance are both measured from the marker's time, so taking 098 down and applying it again restarts both from the new time. Usage recorded in the earlier window is kept as evidence but no longer counts against a lot.

The shared-key shutdown is not rolled back from here. It is PR #234; once this PR is merged, reverting #234 conflicts in the test tree, so its rollback is `SHARED_KEY_WRITES=allow` (see "Shared key" above), not a revert.
