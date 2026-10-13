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

Acceptance evidence will be recorded here after implementation and verification.
