# A6 — ship gate and private evidence (built; staging acceptance verified)

Based on origin/main 3e52d21; migration 074 is reserved after checking open PRs #90, #91, #96. Staging only. No startup migration, production access or merge.

Apply 074 as table owner via port 5432 in a transaction with ON_ERROR_STOP, lock_timeout 5s, statement_timeout 60s and search_path public. Migration is additive and safe to repeat. Attachments have RLS and immutable uploader, creation time, hash, size and kind.

Storage: configure SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY for a private `evidence` bucket (public buckets fail closed), or EVIDENCE_STORAGE_DIR for an absolute directory on a persistent private volume. The directory and files use 0700/0600. Files are downloaded through authenticated FL routes; no public or bearer storage links. Local staging acceptance uses a dedicated persistent private evidence directory and the guarded staging DB; it does not deploy or alter another staging lane.

POST /attachments accepts multipart file, kind, optional ticket_id and informational extracted JSON. JPEG/PNG/WebP/PDF, 15 MiB maximum; correction photos require an image. Use returned attachment_id or attachment_ref. Prepare/commit never treats extracted metadata as posted values.

P1.13: correction commit and counted shortage resolution verify existence, uploader and context, and read/hash-check the actual file. An unbound upload by the same actor is claimed by that ticket/exception. A previously bound file cannot be reused for another context. Explicit owner approval may still post a held correction without a photo (decision 7); any supplied/stored photo reference must be verified.

Ship prepares: `/sales/orders/{id}/ship/prepare` (ship_all or explicit line quantities), `/ship/prepare` (customer_id/product_id/optional lot_id) and `/ship/bulk/prepare` (customer_id and resolved lines). All commit through `/tickets/{token}/commit`. Supply lot_confirmations at prepare or commit, BOL in the prepared payload and attachment_ids at prepare or commit. No photo still posts; proof_due_at is 23:59 America/New_York on the shipment day.

The existing explicit exceptions sweep opens one BOL_PHOTO_MISSING per overdue shipment. It never runs on startup and no scheduler is created. A late BOL image resolves it; a signed packing slip or truck photo attaches without satisfying BOL-photo proof. Receipt responses carry short-lived HMAC packing-slip links bound to order and receipt; legacy key links stay available until A10. A7 shipment_gate applies to ticket close and invoiced transitions.

## Verified acceptance — 2026-10-12

PR #99 remains draft, unmerged. Code built on main 3e52d21. 2,380 Python tests passed with zero failures/skips on a freshly created UTF-8 PostgreSQL 17 database at local port 57674; 69 Node tests passed, zero failures/skips. Affected integration run: 745 passed; A6 ship acceptance: 49 passed. GPT schema remains at 30 operations.

Migration 074 applied ONLY to isolated staging via port 5432 and the staging guard. Acceptance used local TestClient against staging without startup or hosted deployment. It retained synthetic order **SO-261012-001** (ID 1000000007), **ticket 51**, **shipment 1**, **SHP-261012-001**, 50 lb, BOL **STG-A6-EC3429B9180B-BOL** and BOL image **attachment 1**, proof complete. The image is a readable synthetic BOL, 14,865 bytes; upload/download hash verified. Private evidence persists at the local staging acceptance directory; its path is not a public URL. `docs/deployments/a6-staging-receipt.json` contains the sanitized database proof. Both ephemeral actors were deactivated; their keys were never stored or printed.

The existing explicit sweep must be run after the 23:59 plant-time deadline to open missing-photo exceptions. FOLLOWUPS P1.14 still owns scheduling; no cron or production changes are included here. For hosted staging, configure a private Supabase evidence bucket or a persistent private filesystem volume before deploying this branch. The local staging acceptance did not change the hosted staging service or another lane.

### Minimal client flow

1. POST multipart `/attachments` with a named actor, `kind=bol`, and a JPEG/PNG/WebP image. Retain `attachment_id`. Signed packing slips use `kind=signed_packing_slip`; they do not satisfy BOL-photo proof.
2. POST `/sales/orders/{order_id}/ship/prepare` with `ship_all:true`, confirmed `bol_reference`, `attachment_ids:[id]`, and `lot_confirmations:[{lot_id,method:"full_code",value:"confirmed lot code"}]`. Inspect the frozen draft.
3. POST `/tickets/{ticket}/commit` with its `payload_hash` and required warning codes. Missing lot or photo evidence may be added in `lot_confirmations` / `attachment_ids`; frozen quantities, customer, BOL and event time cannot change. Record the SHP receipt.
4. Download files only through authenticated `/attachments/{id}/file`. A late BOL image may be linked through `/shipments/{id}/attachments`; it marks proof complete and resolves the one pending photo exception.
