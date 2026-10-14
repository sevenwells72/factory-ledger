# D2 office and void pilot — October 14

Builder: Codex. Independent reviewer: Claude Code. Keep the PR draft; do not merge.

Base verified: origin/main contains PR #107, merge 611d0c4. Worktree: ~/dev/fl-d2office-wt; branch feat/dash-office-void. The shared repository twice failed checkout with `mmap failed: Operation canceled`; an isolated bare Git store at ~/dev/fl-d2office-source.git supplies this worktree. No other checkout was modified.

## Part 1 — office

A7 prepare → reviewed draft → Record for orders, order lines, headers/status, close/cancel, expected deliveries and extracted PO/SO approval. Customer and product IDs come from /resolve; pouch conversion comes from FL. Preserve duplicate PO acknowledgement, No PO, service lines and legacy-close policy. Missing receipt searches are neutral.

## Part 2 — void

Receipt-specific FL permissions control void. Prepare with a fixed reason, commit the ticket, show its VD receipt and the original receipt’s VOIDED backlink.

## Acceptance

All new controls require key_sessions_enabled. Preview hosts use staging only. EN/ES, 390 px, per-action receipts, FL warnings unchanged and failure feedback remain required. Node and browser runners cover forms and both flag modes; full Python suite runs on one fresh throwaway DB at a time. Staging walkthrough and per-part review requests will be recorded here with exact evidence.

## Part 1 evidence (October 14)

- Full fresh UTF-8 PostgreSQL 17 suite: **3,077 passed**, zero failures/errors/skips; scripts/run_single_db_suite.py preserves destructive migration and concurrent-commit fixture semantics while using at most one temporary database. All 21 groups completed and the database was dropped.
- All Node runners: **152 passed**, zero failures/skips.
- Real Chrome, 390 px: run-office.mjs exercises all 14 forms against the live local API, 16 action receipts, duplicate-PO acknowledgement, No PO, pallet service quantity, FL pouch conversion/refusal, whole-case blocker, edits, EN/ES and permission restrictions. Flag-off runs every action URL and verifies no forms or writes. Existing D1/D2 flag-off runner passes.
- Extracted sales-document approval now has an atomic A7 adapter using the existing approval core; commit rechecks document/PO state. Actor-scoped office metadata supplies legal status/close choices. No schema change.
- Shared ticket cards require this action's ORD or ER receipt before green. FL warning text remains verbatim. Missing receipt lookups are neutral.
- Preview routing continues to force staging; existing production UI stays unchanged while the flag is off.
- Staging walkthrough follows Part 2. Claude Code review requested separately; this is builder validation, not an independent review.

## Part 2 and staging walkthrough

The receipt API supplies each entry's allowed_actions using the same live require_void policy as prepare and commit. The receipt screen shows the button only with key_sessions_enabled and a signed-in session. It sends the fixed reason **data_entry_error**, presents FL's draft, and accepts only a **VD** receipt as success. Owner and floor-own-same-day behavior, changed roles, old dates, retry/replay, and the original receipt's VOIDED backlink are covered. Office has no void button. The missing-receipt message stays neutral in EN and ES.

The hosted staging service has sign-in enabled but does not yet expose this branch's new SO adapter. The walkthrough therefore follows the repository's existing acceptance pattern: real Chrome at 390 px, branch UI and HTTP handlers against the guarded staging database, with application lifespan/startup disabled. **No hosted deployment, migration, production access or merge occurred.** Do not describe this as hosted-preview acceptance.

1. Office signed in with a temporary personal key. Customer and products were selected through /resolve.
2. The order draft used PO **000-STG-D2OFF-77262046**, two 25 lb cases at $30/case, and three Pallet Charges at $12.50 each. Pallet service pounds were zero; total **50 lb / $97.50**.
3. Record returned **ORD-261014-001**, order ID **1000000025**. Same-ticket replay returned that receipt without a second order.
4. A temporary floor person recorded 10 lb of synthetic ingredient stock as **RCV-261014-001**, transaction **1000000441**.
5. From that receipt, the floor person prepared the fixed-reason void and pressed Record. FL returned **VD-261014-001**. The original showed **VOIDED**, linked to VD; the VD page named the original receipt. EN/ES and replay passed.
6. Direct staging reads verified one committed ticket per receipt, one receive posting, and its voided effective status. Both temporary users were deactivated and any remaining sessions revoked; numbered evidence was retained.

Evidence: [sanitized results](office-void-staging/evidence.json), [order draft](office-void-staging/order-draft-390.png), [recorded order](office-void-staging/order-recorded-390.png), [void draft](office-void-staging/void-draft-390.png), [voided original](office-void-staging/voided-original-390.png), [Spanish original](office-void-staging/voided-original-es-390.png), [VD receipt](office-void-staging/void-receipt-390.png).

Local Chrome acceptance passes all 14 office forms and owner/floor void with the gate on. Gate off: all 14 action URLs hide forms, receipt void/sign-in controls are absent, no new writes occur, and existing D1/D2 legacy behavior passes. Netlify origin tests cover 13 pages × 4 hosts, including an unsafe override; previews always select staging, with zero hosted API calls in that test.

## Final validation

**3,086 Python tests and 155 Node tests passed**, zero failures/errors/skips. The full suite used fresh UTF-8 PostgreSQL 17 databases in 21 sequential groups; the sole disposable database was dropped after every group. No browser database or local server remains. Disk checked before heavy steps (6.3 GiB free at final suite start). All browser checks above passed on the final application behavior, including bilingual neutral lookup. Source syntax and diff checks pass. No additional schema migration.

Part 1 pushed at **9b073b1**, with its Claude Code review-request note on PR #115. Part 2 is pushed separately with its own review request. These are builder checks; Claude Code's independent review remains pending.
