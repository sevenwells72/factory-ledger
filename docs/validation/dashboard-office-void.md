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
