# A6 — ship gate and private evidence (draft)

Based on origin/main 3e52d21; migration 074 is reserved after checking open PRs #90, #91, #96. Staging only. No startup migration, production access or merge.

Apply 074 as table owner via port 5432 in a transaction with ON_ERROR_STOP, lock_timeout 5s, statement_timeout 60s and search_path public. Migration is additive and safe to repeat. Attachments have RLS and immutable uploader, creation time, hash, size and kind.

Storage: configure SUPABASE_URL/SUPABASE_SERVICE_ROLE_KEY for a private `evidence` bucket (public buckets fail closed), or EVIDENCE_STORAGE_DIR for an absolute directory on a persistent private volume. The directory and files use 0700/0600. Files are downloaded through authenticated FL routes; no public or bearer storage links. Local staging acceptance uses a dedicated persistent private evidence directory and the guarded staging DB; it does not deploy or alter another staging lane.

POST /attachments accepts multipart file, kind, optional ticket_id and informational extracted JSON. JPEG/PNG/WebP/PDF, 15 MiB maximum; correction photos require an image. Use returned attachment_id or attachment_ref. Prepare/commit never treats extracted metadata as posted values.

P1.13: correction commit and counted shortage resolution verify existence, uploader and context, and read/hash-check the actual file. An unbound upload by the same actor is claimed by that ticket/exception. A previously bound file cannot be reused for another context. Explicit owner approval may still post a held correction without a photo (decision 7); any supplied/stored photo reference must be verified.
