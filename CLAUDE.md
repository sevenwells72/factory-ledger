# Global Instructions for Claude Code

## Production freeze: LIFTED (Michael, 2026-10-27) — cutover CLOSED 2026-10-28

The Oct 21 freeze ended 2026-10-27 (RESET COMPLETE). All FREEZE-LIFT-PLAN deployments are live (#145, #146, migration 088 + #153, #154; production main `9a9a9536`). **Current state, decisions and open work: [`audits/fresh-start/v3/MASTER-HANDOFF-2026-10-28.md`](audits/fresh-start/v3/MASTER-HANDOFF-2026-10-28.md) in this repo — read it first, then re-verify live state read-only.** Staff workflows and architecture: `docs/operating-model/`. Every production change (deploy, migration, inventory posting, credential change) still needs Michael's explicit approval for that step, a verified backup first (`~/Backups/fl/fl-backup-and-verify.sh` on Michael's Mac) and verification after.

Operating principles (owner, 2026-10-28): accurate inventory and traceability going forward over reconstructing history; label historical assumptions as assumptions, never as physically verified; never edit or delete transactions — correct with new records carrying a reason, receipt and named actor; record supplier lots exactly or explicitly "No lot on label"; never identify pooled inventory as one supplier lot; staff use their own production accounts; keep daily workflows simple. Inventory queries must sum only `transactions.status='posted'` lines (voided history otherwise shows false negative lots).

**Finding code:** `docs/ARCHITECTURE-MAP.md` maps each module (inventory, receiving, production, packing, shipping, sales orders, dashboard, auth, Assistant, database, deployment) to its files, `main.py` section anchors and tests. Open it when a task needs code. It is deliberately not loaded into every session. Then follow "Working efficiently" below.

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
* Write both entries with `python3 scripts/log_change.py --summary "..." --files "a.py, b.js" --what "..." --why "..."`, not by reading or editing the logs. `CHANGE_LOG.md` is about 1 MB.
  * The script enforces the rules above: top-of-file and below-separator placement, the symlink and its canonical target, no duplicates.
  * It locks each file while writing and restores the original if a write fails.
  * Add `--dry-run` to validate first, or `--project-only` when working outside the Mac that has the global log.
  * If it refuses, fix the cause, or stop and report it. Never bypass it by hand-editing.
* HARD RULE: openapi-gpt-v3.yaml has a 30-operation limit. Count operations before and after any schema change. Never exceed 30.
* HARD RULE: never use `set_session(readonly=True)` or any other session-level readonly setting (`SET SESSION CHARACTERISTICS`, `default_transaction_read_only`, `PGOPTIONS`) against the port-6543 pooler URL. It poisons shared pooled connections and breaks production writes. For read-only safety, connect via port 5432 (session pooler) or wrap every query in an explicit transaction: `BEGIN TRANSACTION READ ONLY; ... COMMIT;` With psycopg2, run `SET TRANSACTION READ ONLY` immediately after each `BEGIN` instead.

## Hard rules

* NEVER push to origin/main or merge into main without an explicit, per-action approval in the current session. Deploy sequences are confirmed one discrete step at a time.
* This checkout is shared by concurrent sessions: stage hunks (`git add -p`) or HEAD-based staged blobs, never whole files or `git add -A`, and re-read `git status`/`git log -3` immediately before every commit and push.

## Working efficiently (risk tiers)

Classify each task before reading code. The tier depends on what the change can break, not on the size of the diff. When unsure, pick the higher tier. Saving tokens never relaxes any rule in this file.

* **Small:** wording, CSS, docs, test-only changes, or one dashboard view that does not write.
  * Open only the files `docs/ARCHITECTURE-MAP.md` names and their tests.
  * Run those test files: `node --test tests/<file>.js` or `scripts/run_tests.sh tests/<file>.py -x`.
* **Medium:** logic inside one module, a dashboard flow that calls write endpoints, an Assistant intent, or sales-order editing.
  * Also search for callers of every changed function, including `api.<name>` calls from feature modules (`grep -n '<name>' *.py`).
  * Run the module's tests from the map, plus any test that mentions the changed code (`grep -l '<name>' tests/`).
* **High:** anything touching:
  * inventory postings (`transactions`, `transaction_lines`), lot or supplier-lot identity, make/pack consumption;
  * shipping, sales-order allocation or close, voids and corrections;
  * authentication, keys, roles or PINs;
  * migrations or startup DDL, idempotency, or helpers shared across modules;
  * GPT schema or instructions;
  * deploy config (`railway.json`, `Procfile`, `netlify.toml`, `requirements.txt`).

  For High changes:
  * Trace dependencies both ways.
  * Read the related runbooks (`VOID_SEMANTICS_RUNBOOK.md`, `IDEMPOTENCY_KEY_PLAN.md`, `docs/named-actor-writes.md`).
  * On the final candidate run `scripts/run_full_suite.sh` once, plus `node --test tests/*.js`, plus browser acceptance checks for any changed UI flow (see Tests).
  * Prove it on staging before any production step.

Saving context without cutting corners:
* **Map first, search second, never scan.**
* **Search code, not evidence:** `git grep -n 'term' -- ':!docs' ':!audits' ':!archive' ':!*.md'`. Open `docs/validation/`, `docs/design/audit/`, `audits/` and `archive/` only when the task is about that evidence.
* **Never read large files whole:** this covers `main.py`, `CHANGE_LOG.md` and `FACTORY_LEDGER_CHANGELOG.md`. Find the spot with `grep -n`, then read about 200 lines around it.
* **Reuse what this session already established** unless `git log` or `git diff` shows the code changed since.
* **Smaller models are fine for locating code:** for "where is X" questions across many files, an Explore subagent on Haiku or Sonnet is fine. Design, review, safety checks and High-tier work stay on the main model.
* **Editing (Claude Code):** change files with Edit/Write, not shell or Python scripts that rewrite files. Routine test, log and check commands are pre-allowed in `.claude/settings.json`; shell edits are not, so each one waits for approval.
* **Tests:**
  * During development run targeted tests only: the map's tests plus `grep -l '<name>' tests/`. After each change rerun the affected tests (`-x`, then `--lf`).
  * Full suite: `scripts/run_full_suite.sh` on the final candidate only. It checks the test interpreter against `requirements.txt`, runs both key-session modes at once on fresh disposable databases, drops them, and records the result by exact file contents. `scripts/run_full_suite.sh --verify` says whether the current files already have a passing record; if so, don't rerun it. After a full run, a change to non-test code in a High-risk area needs one more full run; test-only or doc-only changes need only the affected tests. Never use a shared or leftover database for a full run.
  * Classic visual audit (`npm run test:visual`): only when a shared style or shell file changes (`dashboard.css`, `interaction.css`, `ledger-forms.css`, `shell-layout.css`, `wallpaper.css`, `shell.js`, `ledger-ui.js`) or when asked; otherwise narrow it with `--screens`. Browser acceptance checks for a changed UI flow are always required.
  * Tests use only local databases; `tests/conftest.py` refuses production hosts. Never point tests at production or staging.
* **Keep the map current:** if a map pointer is wrong, or you add a module or a `main.py` section, update `docs/ARCHITECTURE-MAP.md` in the same change. Then run `python3 scripts/check_architecture_map.py`.

## Two-agent work (Codex manages, Claude Code implements)

* **Findings:** send critical ones at once (security, data integrity, attribution or inventory, wrong design or scope). Batch everything else and send it at the next checkpoint: after the plan, after the backend slice with its targeted tests, after the frontend slice, and on the final candidate. Never interrupt a running command or test for a non-critical finding.
* **Test evidence:** reuse a passing `scripts/run_full_suite.sh` record for the same files (`--verify`) and archived targeted-test logs. Rerun only to reproduce a specific suspected defect.

## Regression Guard

Before making ANY changes to `main.py`, GPT instructions, migrations, or the dashboard:

1. **Search `FACTORY_LEDGER_CHANGELOG.md` for the area you're touching. Don't read the whole file (about 450 KB).**
   * Search, for example: `grep -n -i -E 'ship|allocation' FACTORY_LEDGER_CHANGELOG.md | cut -c1-400`.
   * Use the area's terms: module, route, table, function and file names.
   * Read the full "Breaks If Reverted" text of every matching row.
   * Medium and High tiers (see "Working efficiently"): also search the names of every function, table and route the change touches.
   * High tier: also read the matching rows in full, and widen the search until no new related rows appear.
2. **Check "Known Root Causes"** section — your fix might be a symptom of a deeper issue
3. **Check "Permanent Rules"** section — if editing GPT instructions, all 10 rules must survive
   * Steps 2–3 are mandatory for every change in scope. Both sections together are about 2 KB. This command prints exactly those two sections and stops at the next `## ` heading: `awk '/^## /{p=($0 ~ /^## (Known Root Causes|Permanent Rules)/)} p' FACTORY_LEDGER_CHANGELOG.md`.
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
