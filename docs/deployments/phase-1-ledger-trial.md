# Phase 1 deployment note — timestamp integrity and certification cutoff

Status: isolated database verified (67/67 tests); not applied to production.

Original transaction headers and line rows are now database append-only, not only timestamp-protected. Administrative lot reassignment/merge paths append effective line-correction events, so current balance reads move while raw historical lines remain unchanged.

## What becomes provable

- Every core ledger row has a database-generated, immutable `created_at` and a provenance marker that distinguishes database-created rows from legacy or migration backfills.
- Operational `occurred_at` / `business_date` is separate from creation time.
- Voids and amendments are append-only correction events. The original transaction row and its creation time remain unchanged.
- Each business day has one append-only certification chain with an independently stored cutoff time.
- The late-entry JSON and CSV endpoints classify `created_at > certified_at` and always disclose correction events.

## Operator-visible change

- Voiding now requires a reason and returns a correction-event ID.
- The owner records the evening cutoff through `POST /records/certifications` (or corrects it through the certification correction endpoint).
- `GET /records/late` and `GET /records/late.csv` provide the cutoff evidence.

## Production sequence

1. Run `migrations/dry-runs/039_phase1_timestamp_integrity_dry_run.sql` and save its read-only output.
2. Take a complete application-schema-and-data backup and prove it can be restored into an isolated database.
3. Review the migration's additive DDL and existing-row backfill count. Migration 039 updates existing rows to populate creation-time provenance and transaction business-time fields; do not proceed until that rewrite is explicitly approved.
4. Apply `migrations/039_trial_timestamp_integrity.sql`.
5. Deploy the matching API revision.
6. Run the Phase 1 test/smoke checklist against a disposable transaction and certification; do not change historical creation times.

The migration is forward-safe. Existing timestamps are preserved. Missing creation times receive the single migration timestamp and the explicit `migration_backfill_039` marker; existing times are marked `legacy_unverified`.
