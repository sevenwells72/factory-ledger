# Global Instructions for Claude Code

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
* Newest entries go at the top of the table (just below the header row).
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
