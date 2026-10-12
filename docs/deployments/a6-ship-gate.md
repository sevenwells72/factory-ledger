# A6 — ship gate and private evidence (PR #99; rollout prerequisites)

Based on origin/main 3e52d21; migration 074 is reserved after checking open PRs #90, #91, #96. 074 was applied to isolated staging only during the build. This PR performs no production access or merge. The production-before-merge procedure below is required operator work.

Apply 074 as table owner via port 5432 in a transaction with ON_ERROR_STOP, lock_timeout 5s, statement_timeout 60s and search_path public. Migration is additive and safe to repeat. Attachments have RLS and immutable uploader, creation time, hash, size and kind.

Storage: configure SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY for a private `evidence` bucket (public buckets fail closed), or, **outside production only**, EVIDENCE_STORAGE_DIR for an absolute directory on a persistent private volume. The directory and files use 0700/0600. Files are downloaded through authenticated FL routes; no public or bearer storage links. Local staging acceptance uses a dedicated persistent private evidence directory and the guarded staging DB; it does not deploy or alter another staging lane.

POST /attachments accepts multipart file, kind, optional ticket_id and informational extracted JSON. JPEG/PNG/WebP/PDF, 15 MiB maximum; correction photos require an image. Use returned attachment_id or attachment_ref. Prepare/commit never treats extracted metadata as posted values.

P1.13: correction commit and counted shortage resolution verify existence, uploader and context, and read/hash-check the actual file. An unbound upload by the same actor is claimed by that ticket/exception. A previously bound file cannot be reused for another context. Explicit owner approval may still post a held correction without a photo (decision 7); any supplied/stored photo reference must be verified.

Ship prepares: `/sales/orders/{id}/ship/prepare` (ship_all or explicit line quantities), `/ship/prepare` (customer_id/product_id/optional lot_id) and `/ship/bulk/prepare` (customer_id and resolved lines). All commit through `/tickets/{token}/commit`. Supply lot_confirmations at prepare or commit, BOL in the prepared payload and attachment_ids at prepare or commit. No photo still posts; proof_due_at is 23:59 America/New_York on the shipment day.

The existing explicit exceptions sweep opens one BOL_PHOTO_MISSING per overdue shipment. It never runs on startup and no scheduler is created. A late BOL image resolves it; a signed packing slip or truck photo attaches without satisfying BOL-photo proof. Receipt responses carry short-lived HMAC packing-slip links bound to order and receipt; legacy key links stay available until A10. A7 shipment_gate applies to ticket close and invoiced transitions.

## Verified acceptance — 2026-10-12

PR #99 remains draft, unmerged. Code built on main 3e52d21. 2,380 Python tests passed with zero failures/skips on a freshly created UTF-8 PostgreSQL 17 database at local port 57674; 69 Node tests passed, zero failures/skips. Affected integration run: 745 passed; A6 ship acceptance: 49 passed. GPT schema remains at 30 operations.

Migration 074 applied ONLY to isolated staging via port 5432 and the staging guard. Acceptance used local TestClient against staging without startup or hosted deployment. It retained synthetic order **SO-261012-001** (ID 1000000007), **ticket 51**, **shipment 1**, **SHP-261012-001**, 50 lb, BOL **STG-A6-EC3429B9180B-BOL** and BOL image **attachment 1**, proof complete. The image is a readable synthetic BOL, 14,865 bytes; upload/download hash verified. Private evidence persists at the local staging acceptance directory; its path is not a public URL. `docs/deployments/a6-staging-receipt.json` contains the sanitized database proof. Both ephemeral actors were deactivated; their keys were never stored or printed.

The existing explicit sweep must be scheduled and verified after the 23:59 plant-time deadline to open missing-photo exceptions (P1.14, operator recipe below). No cron or production changes were performed by the builder. For hosted staging, configure a private Supabase evidence bucket or a persistent private filesystem volume before deploying this branch. The local staging acceptance did not change the hosted staging service or another lane.

### Minimal client flow

1. POST multipart `/attachments` with a named actor, `kind=bol`, and a JPEG/PNG/WebP image. Retain `attachment_id`. Signed packing slips use `kind=signed_packing_slip`; they do not satisfy BOL-photo proof.
2. POST `/sales/orders/{order_id}/ship/prepare` with `ship_all:true`, confirmed `bol_reference`, `attachment_ids:[id]`, and `lot_confirmations:[{lot_id,method:"full_code",value:"confirmed lot code"}]`. Inspect the frozen draft.
3. POST `/tickets/{ticket}/commit` with its `payload_hash` and required warning codes. Missing lot or photo evidence may be added in `lot_confirmations` / `attachment_ids`; frozen quantities, customer, BOL and event time cannot change. Record the SHP receipt.
4. Download files only through authenticated `/attachments/{id}/file`. A late BOL image may be linked through `/shipments/{id}/attachments`; it marks proof complete and resolves the one pending photo exception.

## Required production prerequisites — before merge (operator checklist)

- [ ] Obtain the separate production migration/deployment approval. Apply **074 to PRODUCTION before merge/code deploy**, as the **same app/table-owner role** used by FastAPI's database connection, on port **5432**. This is a rollout instruction, not a production action performed by the builder.
- [ ] **Verify ownership first** in the approved session. Confirm the app connection's `current_user` is the role below, and that every existing target table has `owner_is_current_user=true`; inspect `attachments` as well if it already exists. Stop and resolve a mismatch before applying 074. A granted non-owner app role cannot use `attachments`: **RLS enabled, zero policies**. Do not add permissive policies or disable RLS to work around wrong ownership.

```sql
SELECT current_user AS app_role, c.relname,
       pg_get_userbyid(c.relowner) AS table_owner,
       c.relowner = (SELECT oid FROM pg_roles WHERE rolname=current_user) AS owner_is_current_user
FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace
WHERE n.nspname='public' AND c.relname IN
  ('shipments','write_tickets','exceptions','migration_markers','attachments')
ORDER BY c.relname;
```

- [ ] Apply from the A6 checkout in a `psql` session with `ON_ERROR_STOP=1`; no credentials in command output. Migration is additive and rerunnable; no in-app startup migration applies it.

```sql
\set ON_ERROR_STOP on
BEGIN;
SET LOCAL lock_timeout='5s';
SET LOCAL statement_timeout='60s';
SET LOCAL search_path=public;
\i migrations/074_ship_gate_attachments.sql
COMMIT;
SELECT name FROM migration_markers WHERE name='074_ship_gate_attachments';
SELECT current_user AS app_role, pg_get_userbyid(c.relowner) AS table_owner,
       c.relrowsecurity AS rls_enabled, c.relforcerowsecurity AS force_rls,
       (SELECT count(*) FROM pg_policy p WHERE p.polrelid=c.oid) AS policies
FROM pg_class c WHERE c.oid='public.attachments'::regclass;
SELECT count(*) AS attachment_rows FROM public.attachments;
```

- [ ] Verify after apply using the **actual app role**: table owner matches `current_user`, `rls_enabled=true`, `force_rls=false`, `policies=0`, and the count query succeeds. Record only the marker/ownership checks, never a connection URI or key.
- [ ] Set `ENVIRONMENT=production`. Configure `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` through Railway secrets. Production evidence storage is **private Supabase bucket `evidence` only**; verify bucket `public=false`. The app checks privacy on every storage operation. Remove/unset `EVIDENCE_STORAGE_DIR`: startup **refuses** any configured local directory in production, including Railway's filesystem. Staging's local acceptance directory is not production evidence storage.
- [ ] Configure a dedicated nonblank **`PACKING_SLIP_SIGNING_KEY`** through Railway secrets. Production startup **refuses** a missing key before opening the DB pool; production signing **never falls back to `API_KEY`**. Keep this key stable across FastAPI workers; rotation invalidates outstanding 10-minute slip links.
- [ ] Configure the P1.14 sweep service below and verify it exits successfully. A sweep runs separately; startup does not open photo exceptions.
- [ ] On actual cutover day, **set `ORDER_LEGACY_CLOSE_CUTOFF=YYYY-MM-DD`** in FastAPI environment/config and redeploy (§10.1). This is an America/New_York **creation date**, not order_date or UTC date. Unset means all orders are pre-cutover; invalid date formats refuse startup. Verify the owner's pre-cutover note-bearing close and post-cutover 403. Do not backdate the setting to the build date.

### P1.14 Railway cron recipe (operator action, not created by this PR)

- [ ] Create a separate Railway service from the same repo and deployed branch as FastAPI, using `python scripts/exceptions_sweep.py` as its start command; set its dashboard cron schedule to **`30 7 * * *` UTC** (02:30 EST / 03:30 EDT on the next day, after 23:59 America/New_York in either season).
- [ ] Use a reference variable `DATABASE_URL=${{FastAPI.DATABASE_URL}}`, `ENVIRONMENT=production`, and `PRODUCTION_DATABASE_HOST` matching the actual FastAPI DB host. Do not copy or print credentials. Staging uses `${{FastAPI-staging.DATABASE_URL}}`, `ENVIRONMENT=staging` and the same isolation guard variables as FastAPI-staging; never point the staging job at production.
- [ ] Redeploy the cron service after script changes if Railway watch rules omit `scripts/`. Run once manually, verify successful exit and the sanitized `Escalated N overdue exception(s) and M shortage flag(s)` log. Repeated runs do not duplicate `BOL_PHOTO_MISSING`, including resolved/waived rows.

## Hosted staging check after merge (manual checklist)

This is **pending hosted acceptance**, distinct from the earlier local TestClient/staging-DB proof above. Use the hosted FastAPI-staging URL, its isolated DB and named test actors; verify guarded staging identity first. Configure its private storage and dedicated signer. Use synthetic orders/stock only; retain sanitized order/ticket/receipt/attachment IDs and deactivate temporary actors afterwards.

1. [ ] **Upload:** POST multipart `/attachments` as the shipping actor, `kind=bol`, with a readable synthetic BOL image. Retain `attachment_id`; confirm uploader, content type, size and SHA-256 metadata. No unauthenticated file access or public storage link.
2. [ ] **Prepare:** POST `/sales/orders/{order_id}/ship/prepare` with `ship_all:true`, the confirmed BOL number, `attachment_ids:[id]` and full-code lot confirmations. Inspect the frozen draft and required warnings; confirm no stock movement yet.
3. [ ] **Commit:** POST `/tickets/{ticket}/commit` with the returned payload hash and warning codes. Verify one SHP receipt, one shipment, the BOL, authenticated actor, inventory deduction, proof complete and the signed packing-slip link; retry the same commit and verify replay with no second shipment.
4. [ ] **Read/attach:** Look up that receipt and download `/attachments/{id}/file` as an authorized actor; verify matching SHA-256, unauthorized actor refusal and no-key 401. Upload and link `kind=signed_packing_slip` through `/shipments/{id}/attachments`; confirm it attaches. For a second synthetic order, commit a BOL-bearing shipment without a BOL photo: it posts with `photo_pending` and the shipment-day 23:59 NY deadline. A packing slip or truck photo must leave it pending.

- [ ] **One sweep run after 23:59:** leave the second shipment photo-pending until its actual plant-day deadline, run the staging sweep service once, and verify exactly one `BOL_PHOTO_MISSING` for its shipment/ticket/receipt. Confirm no exception on the photo-complete shipment. Upload a late BOL image as the same actor and link it; verify proof complete and that exception resolved. Confirm subsequent scheduled runs do not duplicate it.
- [ ] **Policy checks:** with the real staging clock, owner/floor can acknowledge their own `SHIP_NOT_SAME_DAY`; office is refused. Receipts filtered by `policy_code` list the acknowledgement actor/dates and legacy close note/actor for the future A9 weekly view. Test cutoff using staging configuration only; do not alter production.

## Review-fix validation — 2026-10-12 (Claude review at 382e0d7)

Fixes 1–5 implemented; decisions 41–42 recorded in the design. New regression module has 35 cases; its corrected initial before-fix run had 21 failures and 7 passes (remaining configuration/preserved-behavior cases added afterwards). Expanded focused integration run passed 132 tests before the final two missing-storage-config cases were added. The **complete final suite passed 2,415 Python tests / 69 Node tests, zero failures/skips**, using newly created UTF-8 PostgreSQL 17 database `fl_a6_fix_final` at local port 57674, loaded from `tests/schema/schema.sql`. No new migration/schema changes. Hosted staging acceptance, production 074, secret configuration and cron creation are pending operator steps, not actions performed by this update. PR #99 stays draft and unmerged for Claude Code's next review.
