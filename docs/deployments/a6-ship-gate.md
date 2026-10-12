# A6 — ship gate and private evidence (implementation; validation pending)

Based on origin/main 3e52d21; migration 074 is reserved after checking open PRs #90, #91, #96. Staging only. No startup migration, production access or merge.

Apply 074 as table owner via port 5432 in a transaction with ON_ERROR_STOP, lock_timeout 5s, statement_timeout 60s and search_path public. Migration is additive and safe to repeat. Attachments have RLS and immutable uploader, creation time, hash, size and kind.

Storage: configure SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY for a private `evidence` bucket (public buckets fail closed), or EVIDENCE_STORAGE_DIR for an absolute directory on a persistent private volume. The directory and files use 0700/0600. Files are downloaded through authenticated FL routes; no public or bearer storage links. Local staging acceptance uses a dedicated persistent private evidence directory and the guarded staging DB; it does not deploy or alter another staging lane.

POST /attachments accepts multipart file, kind, optional ticket_id and informational extracted JSON. JPEG/PNG/WebP/PDF, 15 MiB maximum; correction photos require an image. Use returned attachment_id or attachment_ref. Prepare/commit never treats extracted metadata as posted values.

P1.13: correction commit and counted shortage resolution verify existence, uploader and context, and read/hash-check the actual file. An unbound upload by the same actor is claimed by that ticket/exception. A previously bound file cannot be reused for another context. Explicit owner approval may still post a held correction without a photo (decision 7); any supplied/stored photo reference must be verified.

Ship prepares: `/sales/orders/{id}/ship/prepare` (ship_all or explicit line quantities), `/ship/prepare` (customer_id/product_id/optional lot_id) and `/ship/bulk/prepare` (customer_id and resolved lines). All commit through `/tickets/{token}/commit`. Supply lot_confirmations at prepare or commit, BOL in the prepared payload and attachment_ids at prepare or commit. No photo still posts; proof_due_at is 23:59 America/New_York on the shipment day.

The existing explicit exceptions sweep opens one BOL_PHOTO_MISSING per overdue shipment. It never runs on startup and no scheduler is created. A late BOL image resolves it; a signed packing slip or truck photo attaches without satisfying BOL-photo proof. Receipt responses carry short-lived HMAC packing-slip links bound to order and receipt; legacy key links stay available until A10. A7 shipment_gate applies to ticket close and invoiced transitions.
