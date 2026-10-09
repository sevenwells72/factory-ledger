# A7 — Order and expected-receipt tickets (PR #93)

Rebased `feat/order-tickets` onto `origin/main` **ce7bb57** (merged A2, A5 and A3b), in `~/dev/fl-a7-wt`. The conflict resolutions preserve both A3b exception/approval hooks and A7 metadata dispatch in `write_tickets.py`, the combined actor route contract, migration fixture ordering and all historical changelog entries.

Design: `docs/design/phase1-safe-operating-system.md` §1, §1.3, §1.5, §4.3 and §10/A7. No merge or production changes are part of this work.

## Review fixes

1. **Durable reference replay.** `external_order_reference` is a global immutable create key. Reference locks, the direct-create receipt lookup and order reference indexes no longer include the mutable customer. Ticket prepare/commit looks up the original committed ticket by reference and compares the original stored intent (excluding entry day and duplicate-PO acknowledgement). The same original request returns the original order and receipt after customer reassignment or catalog deactivation. Changed intent conflicts. A replay ticket stores the original response/result reference, with `replayed: true`, and has no new receipt number or order. Its own token remains bound to its preparing identity. The original committed ticket is immutable under A1; it is the durable receipt, including for previously committed A7 tickets. Direct creates retain their original immutable `sales_order_create_receipts` response.
2. **Concurrent SO allocation.** Migration 072 replaces the `MAX + 1` trigger with an atomic per-day counter upsert. Its row lock lasts until commit or savepoint rollback, so a prepare cannot select a number being committed by another request. Initial allocation starts after existing numbers, explicit historic numbers are respected and numbers above 999 are not truncated. Order date-prefix behavior remains the existing database `CURRENT_DATE` convention.
3. **Mixed quantity forms.** Duplicate detection sorts a serialized line key, preserving nulls and multiplicity without comparing `None` with numbers. Repeated products mixing `quantity`/`unit` and `quantity_lb` work for create and add-lines, regardless of line order.
4. **Expected-receipt tickets.** Manual create, update/close, cancel and atomic extracted-document approval share the A1 lifecycle and A2 office/owner permissions. The intake path retains source-document validation/approval, per-line expected receipts, alias learning and duplicate-reference acknowledgement. The existing master/direct APIs and core signatures used by PO sync remain compatible. Expected receipts remain metadata and never add inventory.

## Routes and permissions

All prepares use ID-only inputs and commit through `POST /tickets/{ticket}/commit`. A1 still provides identity and payload binding, expiry, supersession, warning acknowledgement, current-state validation, immutable receipts and concurrent double-commit replay. Actor status and role are rechecked under lock. The shared dashboard key gets no new grants; authenticated office/owner identities can use the expected-receipt prepares with `client_source: dashboard` (30-minute expiry). Master access retains the existing A2 compatibility grant.

| Action | POST prepare route | Named roles |
|---|---|---|
| `create_order` | `/sales/orders/prepare` | office, owner |
| `add_order_lines` | `/sales/orders/{id}/lines/prepare` | office, owner |
| `update_order_line` | `/sales/orders/{id}/lines/{line_id}/update/prepare` | office, owner |
| `cancel_order_line` | `/sales/orders/{id}/lines/{line_id}/cancel/prepare` | office, owner |
| `update_order_header` | `/sales/orders/{id}/header/prepare` | office, owner |
| `update_order_status` | `/sales/orders/{id}/status/prepare` | office, owner; reason checks also apply |
| `mark_order_ready` | `/sales/orders/{id}/ready/prepare` | floor, office, owner |
| `cancel_order` | `/sales/orders/{id}/cancel/prepare` | office, owner |
| `close_order` | `/sales/orders/{id}/close/prepare` | office, owner; unrecorded shipment owner only |
| `reopen_order` | `/sales/orders/{id}/reopen/prepare` | owner |
| `create_expected_receipt` | `/expected-receipts/prepare` | office, owner |
| `create_expected_receipt` (document intake) | `/expected-receipts/extract/approve/prepare` | office, owner |
| `update_expected_receipt` | `/expected-receipts/{expected_receipt_id}/update/prepare` | office, owner |
| `cancel_expected_receipt` | `/expected-receipts/{expected_receipt_id}/cancel/prepare` | office, owner |

Order `{id}` accepts a numeric ID or SO number. Expected-receipt IDs are numeric. Receipt prefixes: **ORD** for sales orders (SO already identifies order numbers), **ER** for expected deliveries. `GET /receipts/{number}` includes `order` or `expected_receipts` along with the original response and actor attribution.

Manual ER input: `product_id`, `supplier_id`, positive finite `expected_qty` in pounds, optional `expected_date`, `reference_number`, `notes`, `source_document_id`. Updates preserve omitted fields; explicit null clears nullable fields. `status: closed` or the cancel prepare requires an open record at commit. Intake input keeps the existing reviewed line contract (`document_id`, `supplier_id`, reference/date, `lines`, optional `force`), rejects caller-supplied identity, and uses the authenticated actor. Prepare rolls back receipt rows, document approval and aliases together; commit posts them together. A duplicate reference is a draft warning, acknowledged explicitly or through the existing `force` override.

These four routes provide the backend replacements for the dashboard's manual/edit/cancel/document intake paths at A10. A regression removes the old expected-receipt routes from both direct allowlists and verifies the named-office ticket path still works. A10's documented gate remains: verify the dashboard's named-identity integration before closing its legacy routes.

## Shared cores and hooks

`order_tickets.py` and `expected_receipt_tickets.py` use the live main.py cores inside a savepoint to build drafts, roll back all business writes, then issue the ticket. Commit uses the same cores after shared lifecycle checks. Business failures roll back the post and persist `TICKET_STALE`/rejection; unexpected failures roll back the whole transaction. Dry-run identity sequence gaps are harmless. Provisional IDs are omitted from create drafts.

Existing order rules survive: duplicate-PO acknowledgement, No PO flag, service lines and pricing, actor audit, status/state rules, reservation release and `MANUAL_TRANSITIONS`. `shipment_gate()` remains A6's named hook for `shipped_recorded`/invoiced. A3b holds, photo evidence, shortage handling and approval logic remain on their original ledger commit path. The order/ER path rejects unused lot confirmations and correction photos.

## Migrations and deployment order

- **067** (already in this PR): order actions, creating-ticket FKs on orders/lines and receipt lookup index. Historical staging application: **2026-10-09 00:50:28Z**; production was not changed.
- **072** (new review migration): global reference indexes, immutable original-ticket reference index, atomic daily SO counters/trigger, expected-receipt actions, nullable `expected_receipts.ticket_id` and index, marker `072_order_ticket_review_fixes`. Rerunnable. RLS on the counter table; no ledger view replacement. Applied only to disposable local test databases during this review.

Apply 067, then 072 **before** deploying the revised code. Use the app/table owner, port 5432, `ON_ERROR_STOP`, an explicit transaction, `SET LOCAL lock_timeout='5s'`, `statement_timeout='60s'`, and `search_path=public`. Migration 072 refuses historical references that identify multiple orders; reconcile those explicitly before retrying. It never picks a winner or deletes business data.

Rollback: stop revised writers and revert the code before 072 down, then optionally 067 down. The 072 down refuses any expected-receipt ticket evidence; archive under an explicit maintenance procedure first. Both migrations' test schema includes are pending until an authorized production rollout and later schema dump. No staging or production migration was applied in this review.

## Validation

The first **13** review regressions were executed against rebased pre-fix code and all failed: reference retry after reassignment; both mixed-quantity paths (`TypeError`); a synchronized live prepare/commit race (`sales_orders_order_number_key`, HTTP 500); missing ER create/edit/cancel/intake routes and role behavior. All pass with the fixes. Additional coverage checks immutable replay after catalog changes, independent ticket retries, parallel prepares/commits, ER double commit, lifecycle/strict inputs, stale suppliers/status, intake aliases and duplicate acknowledgement, direct PO-sync API compatibility, A10 allowlist removal, migration rerun/down/up/guard and SO counters past 999.

Fresh-database full suite: **2,315 Python passed, 0 failed, 0 skipped** (36.67 seconds); **69 Node passed, 0 failed, 0 skipped**. Final suite counts are also recorded in CHANGE_LOG.md. Every full Python run rebuilds `factory_ledger_test_a7_review` from `tests/schema/schema.sql` first; this avoids the existing committed race-fixture contamination between runs. Node uses `node --test tests/test_*.js`.

## Historical staging evidence (original A7, before review fixes)

`docs/deployments/a7-staging-receipt.json` records the original 067-only check: synthetic customer `1000000002`, order **SO-261009-001**, create receipt **ORD-261008-001**, add lines **ORD-261008-002**, floor ready **ORD-261008-003**. Replay and permission denials were checked and temporary actors deactivated. The differing day prefixes reflect database UTC order numbering versus plant-day receipts. This file is historical evidence, not verification of 072 or the revised code.
