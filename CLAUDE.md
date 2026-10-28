# Global Instructions for Claude Code

## Production freeze: LIFTED (Michael, 2026-10-27) — cutover CLOSED 2026-10-28

The Oct 21 freeze ended 2026-10-27 (RESET COMPLETE). All FREEZE-LIFT-PLAN deployments are live (#145, #146, migration 088 + #153, #154; production main `9a9a9536`). **Current state, decisions and open work: [`audits/fresh-start/v3/MASTER-HANDOFF-2026-10-28.md`](audits/fresh-start/v3/MASTER-HANDOFF-2026-10-28.md) in this repo — read it first, then re-verify live state read-only.** Staff workflows and architecture: `docs/operating-model/`. Every production change (deploy, migration, inventory posting, credential change) still needs Michael's explicit approval for that step, a verified backup first (`~/Backups/fl/fl-backup-and-verify.sh` on Michael's Mac) and verification after.

Operating principles (owner, 2026-10-28): accurate inventory and traceability going forward over reconstructing history; label historical assumptions as assumptions, never as physically verified; never edit or delete transactions — correct with new records carrying a reason, receipt and named actor; record supplier lots exactly or explicitly "No lot on label"; never identify pooled inventory as one supplier lot; staff use their own production accounts; keep daily workflows simple. Inventory queries must sum only `transactions.status='posted'` lines (voided history otherwise shows false negative lots).

## Change Log Protocol

You must maintain TWO change logs every time you create, edit, or delete any file on my behalf:

### 1. Project-Level Log: CHANGE_LOG.md

Located at the root of whatever project you're currently working in.

Format:

```
# Change Log

## [YYYY-MM-DD HH:MM] — Brief Summary
- **File(s) changed:** `path/to/file`
- **What changed:** Clear description of what was added, modified, or removed
- **Why:** Context or reason for the change (if known)

---
```

Rules:
* Append a new entry immediately after making any change.
* Newest entries go at the top, just below the `# Change Log` header.
* If multiple files are changed as part of a single task, group them under one entry.
* If the file doesn't exist yet, create it with the header before adding the first entry.
* Never delete or modify previous log entries — this file is append-only.
* Keep descriptions concise but specific enough that someone reading months later would understand what happened.

Example entry:

```
## 2026-02-25 14:30 — Added error handling to shipping endpoint
- **File(s) changed:** `src/api/shipping.py`, `tests/test_shipping.py`
- **What changed:** Added try/except blocks around database calls in the ship_order function; added 3 new test cases for error scenarios
- **Why:** User reported unhandled exceptions when database connection drops mid-transaction

---
```

### 2. Global Log: ~/change-log.md

A single unified log across ALL projects. Each entry is a compact one-liner that includes the project path.

Format:

```
# Global Change Log

| Date | Project | Summary | Files |
|------|---------|---------|-------|
| 2026-02-25 14:30 | `/path/to/project` | Added error handling to shipping endpoint | `shipping.py`, `test_shipping.py` |
```

Rules:
* CANONICAL PATH: the real file is `~/Library/Mobile Documents/com~apple~CloudDocs/Claude Logs/change-log.md` (iCloud Drive). `~/change-log.md` is a symlink to it and MUST stay a symlink — never replace it with a regular file (no atomic write-and-rename, no `mv`/`cp` over it). If `~/change-log.md` is not a symlink, stop and report it instead of writing; the log forked once (2026-09-14 → 2026-10-07) because of this.
* Newest entries go at the top of the table, directly below the `|------|` separator row — never between the header row and the separator.
* If `~/change-log.md` doesn't exist yet, create it with the header and table headers before adding the first entry.
* Never delete or modify previous entries.
* The "Project" column should be the absolute path to the project root.
* The "Summary" column should be a brief one-line description (same as the project-level heading).
* The "Files" column should list the filenames (not full paths) separated by commas.

### General Rules

* Both logs must be updated on every change. No exceptions.
* Update the logs immediately after making the changes, not before.
* If you are working outside of a project directory (e.g. directly in `~`), still log to `~/change-log.md` and skip the project-level log.
* Use the current date and time for each entry.
* HARD RULE: openapi-gpt-v3.yaml has a 30-operation limit. Count operations before and after any schema change. Never exceed 30.
* HARD RULE: never use `set_session(readonly=True)` or any other session-level readonly setting (`SET SESSION CHARACTERISTICS`, `default_transaction_read_only`, `PGOPTIONS`) against the port-6543 pooler URL. It poisons shared pooled connections and breaks production writes. For read-only safety, connect via port 5432 (session pooler) or wrap every query in an explicit transaction: `BEGIN TRANSACTION READ ONLY; ... COMMIT;` With psycopg2, run `SET TRANSACTION READ ONLY` immediately after each `BEGIN` instead.

## Hard rules

* NEVER push to origin/main or merge into main without an explicit, per-action approval in the current session. Deploy sequences are confirmed one discrete step at a time.
* This checkout is shared by concurrent sessions: stage hunks (`git add -p`) or HEAD-based staged blobs, never whole files or `git add -A`, and re-read `git status`/`git log -3` immediately before every commit and push.

## Regression Guard

Before making ANY changes to `main.py`, GPT instructions, migrations, or the dashboard:

1. **Read `FACTORY_LEDGER_CHANGELOG.md`** — scan the "Breaks If Reverted" column for the area you're touching
2. **Check "Known Root Causes"** section — your fix might be a symptom of a deeper issue
3. **Check "Permanent Rules"** section — if editing GPT instructions, all 10 rules must survive
4. **After deploying** — add a new row to the changelog immediately. Format:
   `| # | Date | Area | What Changed | Problem It Solved | Breaks If Reverted | Migration/File |`

### Key files:
- `FACTORY_LEDGER_CHANGELOG.md` — full change history with regression guards
- `main.py` — FastAPI backend (Supabase/Postgres)
- GPT instructions — 8,000 char limit, version-controlled in repo
- `migrations/` — SQL migrations for Supabase

### Architecture:
- Supabase (Postgres) + Railway (FastAPI) + ChatGPT Custom GPT
- GitHub: sevenwells72/factory-ledger (auto-deploys to Railway)
- Dashboard: cns-factory-ledger.netlify.app
- API: fastapi-production-b73a.up.railway.app
