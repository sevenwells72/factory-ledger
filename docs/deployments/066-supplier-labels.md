# Migration 066: supplier display labels (after the Dutch supplier cleanup)

**Gated on Michael's supplier cleanup.** Decided 2026-10-08: only two real
companies — **13 Dutch Gold Honey = `DUTG`** and **16 Dutch Valley Foods = `DUTV`**
(explicit labels, not auto-assigned). 12 Dutch Gold merges into 13; 11 DUTC Valley
(typo), 14 Dutch Valley and 15 Dutch Valley Food Dist. merge into 16. Merge =
repoint every FK reference (`lots`, `transactions`, `lot_supplier_codes`,
`expected_receipts`, `supplier_product_aliases`, `search_aliases`) to the canonical
id, then **deactivate** the duplicate. Rows are never deleted; historical lot codes
(`DUTC…`) and free-text supplier names stay as typed.

A5 already works with nullable labels: migration 064 supplies the supplier FK
columns, the unique `suppliers.short_code` index and immutable lot supplier IDs.
066 only seeds unique four-letter labels and installs the allocator trigger. IDs
remain the source of supplier identity; no lot prefix, historical receipt or FK is
rewritten. `main.py` does not read `short_code` yet.

## Rules 066 enforces

* An explicit (non-NULL) `short_code` is **never overwritten** — the backfill only
  touches `short_code IS NULL`, and the allocator reserves existing codes.
* **Deactivated suppliers are skipped.** The trigger fires `BEFORE INSERT OR UPDATE
  OF active` and labels a supplier when it is active and label-less, so a
  reactivated vendor gets its code at that moment. The trigger is dropped and
  re-created on every run (replaces the INSERT-only version staging received from
  an earlier A5 build under marker `066_receipt_suppliers`).
* Pseudo-suppliers (FOUND, PHYSICAL COUNT, UNKNOWN, …) never get a label.
* Labels are assigned in supplier-id order: natural token (first four letters,
  `X`-padded), collisions take the next free fourth letter.

## Order of operations

1. **Dry-run (read-only)** — preview the post-cleanup labels before writing anything:

   ```sh
   python scripts/dry_run_supplier_labels.py \
       --database-url-file "$HOME/Documents/fl-secrets/production-db-url.txt" \
       --assume-inactive 11,12,14,15 --assume-label 13=DUTG --assume-label 16=DUTV
   ```

   The script reads a mode-600 URI file, uses a transaction-scoped read-only
   connection on port 5432, never invokes the SQL allocator or imports the app,
   and suppresses connection error details. Without `--assume-*` it previews the
   catalog as it is.

2. **Cleanup** — `scripts/supplier_cleanup_dutch_2026_10_08.sql`, one transaction
   (`lock_timeout='5s'`, `statement_timeout='60s'`), staging first, then production:

   ```sh
   psql "$URL" -v ON_ERROR_STOP=1 -X -f scripts/supplier_cleanup_dutch_2026_10_08.sql
   ```

   It guards on the exact id/name pairs, repoints all six FK tables, deactivates
   11/12/14/15 (clearing any auto label they carry), sets `DUTG`/`DUTV`, and
   raises if any reference to a deactivated id remains. If a `lots` or
   `transactions` row ever references a duplicate, the immutable/append-only
   triggers abort the transaction — stop and re-plan rather than disable them.

3. **066** — as table owner, ON_ERROR_STOP, one transaction:

   ```sql
   BEGIN;
   SET LOCAL lock_timeout='5s';
   SET LOCAL statement_timeout='60s';
   SET LOCAL search_path=public;
   SET LOCAL factory_ledger.supplier_cleanup_confirmed='michael_approved';
   \i migrations/066_supplier_labels.sql
   COMMIT;
   ```

   Without the acknowledgment setting, 066 refuses before any mutation. It records
   the operator's assertion that the cleanup was approved and applied; it does not
   perform the cleanup. 064 must already be applied. Reruns preserve labels and
   the marker timestamp.

4. **Verify** — re-run the dry-run with no `--assume-*` flags: every active real
   vendor must be `retained` with the label from step 1, and `SELECT … FROM
   suppliers WHERE short_code IS NOT NULL` must match the table below.

Do not remove or reassign labels once receipts have printed them; a code rollback
can leave these additive labels and the trigger in place.

## Production dry-run 2026-10-08 (read-only, cleanup assumed)

51 suppliers: 37 assigned + 2 explicit = **39 labelled active vendors**, 4 skipped
inactive (the Dutch duplicates), 8 excluded pseudo-suppliers.

| ID | Supplier | Label | | ID | Supplier | Label |
|---|---|---|---|---|---|---|
| 1 | A1 Baker | ABAK | | 31 | Jack's Eggs | JACA |
| 2 | A1 Bakery | ABAA | | 32 | JOEL | JOEL |
| 3 | A1 Bakery Supply | ABAB | | 33 | Kadouri | KADO |
| 4 | Acme Foods | ACME | | 34 | LaCrosse Milling | LACR |
| 5 | Barry Callebaut | BARR | | 35 | Linking Logistics | LINK |
| 6 | Blender Warehouse | BLEN | | 36 | National Harvest | NATI |
| 7 | Blue Stripes | BLUE | | 37 | NEW ENGLAND | NEWE |
| 8 | CBS Food | CBSF | | 38 | Parker Flavors | PARK |
| 9 | Creative Foods | CREA | | 39 | Phildesco | PHIL |
| 10 | David Rosen | DAVI | | 40 | Phildesco c/o Moran Logistics | PHIA |
| **13** | **Dutch Gold Honey** | **DUTG** (explicit) | | 42 | Quali Pack | QUAL |
| **16** | **Dutch Valley Foods** | **DUTV** (explicit) | | 43 | Refrig-IT | REFR |
| 17 | Essex Food | ESSE | | 44 | SAEM | SAEM |
| 18 | Essex Foods | ESSA | | 45 | Sam International | SAMI |
| 19 | Euro | EURO | | 46 | Star Snacks | STAR |
| 20 | Euro Good | EURA | | 47 | Sweet New England | SWEE |
| 21 | Euro Goods | EURB | | 48 | Tilley | TILL |
| 24 | Franklin Baker | FRAN | | 49 | Tri State | TRIS |
| 25 | Grain Supply | GRAI | | 51 | Vinnapro | VINN |
| 30 | Jack's Egg's | JACK | | | | |

Deactivated, no label: 11 DUTC Valley, 12 Dutch Gold, 14 Dutch Valley, 15 Dutch
Valley Food Dist. Staging (which ran the earlier 066 build) already carries these
exact labels for every non-Dutch supplier; the cleanup script corrects the six
Dutch rows there, and re-running 066 replaces the trigger and adds the
`066_supplier_labels` marker.

## Validation

`tests/test_supplier_label_backfill.py` (14 tests, local DB, rolled back): cleanup
gate, 064 precondition, allocator parity incl. >26 collisions, pseudo exclusion,
inactive skipped then labelled on activation, explicit labels kept, INSERT-only
trigger replaced, the cleanup script's repoint/deactivate/label end state and its
catalog guard, assumption overlay without a database, read-only connection guards.
