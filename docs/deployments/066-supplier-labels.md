# Deferred migration 066: supplier display labels

**Draft; do not merge or apply until Michael completes and approves supplier
cleanup (Dutch Valley duplicates and the `DUTC Valley` typo).** This is split out
of A5 PR #88 and stacked on `feat/lot-confirmation`. After A5 merges, retarget to
main and preserve the cleanup gate. No production migration has been run.

A5 already works with nullable labels: migration 064 supplies the supplier FK
columns and immutable lot supplier IDs. This PR only seeds unique four-letter
labels and installs the future-supplier INSERT allocator. IDs remain the source
of supplier identity; no lot prefix, historical receipt or FK is rewritten.
Existing non-NULL labels are preserved. Inactive real vendors are included;
pseudo-suppliers are excluded. Labels are assigned in supplier ID order, so
cleanup can change proposed labels. Today's preview is not approval to apply.

Run the read-only report before cleanup and again after Michael approves it:

```sh
python scripts/dry_run_supplier_labels.py --database-url-file "$HOME/Documents/fl-secrets/production-db-url.txt"
```

The script reads a mode-600 URI file and uses a transaction-scoped read-only
connection on remote port 5432. It prints each supplier's ID/name, active state,
current/proposed label and allocation decision. It tolerates the short_code
column not existing yet. It never invokes the SQL allocator or imports the app,
and suppresses connection error details to keep credentials out of output.

The migration is explicit, not in the schema fixture, staging acceptance runner
or application startup. After cleanup approval, rehearse on staging first and
verify the resulting labels against the dry-run before any separately approved
production rollout. Apply as table owner, with ON_ERROR_STOP, in one transaction:

```sql
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL search_path=public;
SET LOCAL factory_ledger.supplier_cleanup_confirmed='michael_approved';
-- Execute migrations/066_supplier_labels.sql here.
COMMIT;
```

Without the explicit cleanup acknowledgment, 066 refuses before any mutation.
The setting records the operator's assertion that Michael approved cleanup; it
does not perform cleanup or replace that approval. A5 064 must already exist.
Reruns preserve labels. Do not remove/reassign labels after printing them; code
rollback can leave these additive labels and trigger in place.

## Validation and current-catalog production dry-run

Full suite: **1,877 Python + 69 JavaScript tests passed**, zero failures/skips.
The local schema fixture still omits 066; its dedicated tests apply it only
inside rolled-back local test transactions and verify the cleanup gate,
allocator parity, pseudo exclusions, preserved labels and repeatability.
The pre-schema preview and database-enforced read-only connection are tested.

Read-only production preview on 2026-10-08: **51 suppliers**, no changes applied.
These assignments will be recalculated after Michael cleans up the catalog.

| ID | Current supplier name | Proposed label |
|---|---|---|
| 11 | DUTC Valley | DUTC |
| 12 | Dutch Gold | DUTA |
| 13 | Dutch Gold Honey | DUTB |
| 14 | Dutch Valley | DUTD |
| 15 | Dutch Valley Food Dist. | DUTE |
| 16 | Dutch Valley Foods | DUTF |
