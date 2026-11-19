# Receiving and hand additions — migration 098

**DRAFT — local verification and independent Claude Code cross-review required.**

The final owner decision keeps reason-list policy in code, gated by marker 098. The migration does not edit or add columns to `correction_reasons`; applying it under old code leaves that catalog and old reason validation unchanged. Removing the marker restores the prior inventory/reason policy. Shared-key shutdown is independent of the marker.

Nothing applied to staging or production. Each later staging or production step needs separate owner approval; production also needs a verified backup first and verification afterwards. Keep this PR draft; no merge or deploy is authorized.

Each purchased-ingredient delivery gets its own Receive lot. Effective stock-in prevents receiving into an old lot; voided receipts do not. Adjust and existing-lot Found cannot increase the effective hand-entry sum since migration 098 above zero. Count-to-zero may use strictly less than the smaller of the effective receipt pack weight and 100 lb, cumulatively per lot. Product pack weights are never used. Lifetime allowance rows survive voids. Wrong-lot moves are one transaction with two opposite lines, excluded from give-back on both lots, and may open a shortage on the used lot. No tolerance or owner override.

## Switches and deploy order

1. Complete review and local Python, JavaScript and real-browser verification. Keep the PR draft.
2. With separate approval, deploy code to staging, apply 098 in one transaction as table owner and verify. Marker absence preserves old inventory rules. Applying 098 under old code leaves the reason catalog unchanged.
3. Before any approved production change, run `~/Backups/fl/fl-backup-and-verify.sh` and verify its restore and key checks. Approval for a merge does not authorize a migration or deploy.
4. Deploy approved code; then separately apply migration 098 with `ON_ERROR_STOP`, transaction-local search_path, lock timeout and statement timeout. Verify marker, unchanged reason catalog and both new RLS tables.

The shared master key becomes read-only **on code deployment**, independently of the marker. GET and POST /resolve and /products/resolve remain available. A named actor key with role owner gains every route, matching Michael selected by name; no other role gains access. This widens one credential. API_KEY remains required at startup.

Emergency `SHARED_KEY_WRITES=allow` restores old shared-key reach. Setting it is a production credential change requiring explicit owner approval, verified backup and verification. Never set it as a test-suite default.

## Verification

Use the new HTTP tests as the staging checklist: existing received lot rejected; new/placeholder and voided receipt allowed; both count routes refuse exactly one pack, cumulative allowance exhausted and missing pack weight; paired move has two lines, one receipt and one shortage if needed. Verify shared-key write refusal before any row is written, named-owner route reach, and floor denial. Check receipt, history, shift roll-forward and nightly email show both move legs. Reattach photos when receiving again.

## Rollback

Revert the independent shared-key commit alone to restore its previous reach. Revert inventory code before applying the 098 down file in a separately approved transaction. The down file removes only the marker; it keeps lifetime count usage and move identities as audit evidence and never changes the reason catalog. No ledger record is edited or deleted. Preserve the 096 marker, reasons, alert outbox and held-ticket compatibility.
