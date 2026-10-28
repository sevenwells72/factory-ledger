# Agent Instructions (Codex and other non-Claude agents)

## Production freeze: LIFTED (Michael, 2026-10-27) — cutover CLOSED 2026-10-28

The Oct 21 freeze ended 2026-10-27 (RESET COMPLETE). All FREEZE-LIFT-PLAN deployments are live (#145, #146, migration 088 + #153, #154; production main `9a9a9536`). **Current state, decisions and open work: [`audits/fresh-start/v3/MASTER-HANDOFF-2026-10-28.md`](audits/fresh-start/v3/MASTER-HANDOFF-2026-10-28.md) in this repo — read it first, then re-verify live state read-only.** Staff workflows and architecture: `docs/operating-model/`. Every production change (deploy, migration, inventory posting, credential change) still needs Michael's explicit approval for that step, a verified backup first (`~/Backups/fl/fl-backup-and-verify.sh` on Michael's Mac) and verification after.

Operating principles (owner, 2026-10-28): accurate inventory and traceability going forward over reconstructing history; label historical assumptions as assumptions, never as physically verified; never edit or delete transactions — correct with new records carrying a reason, receipt and named actor; record supplier lots exactly or explicitly "No lot on label"; never identify pooled inventory as one supplier lot; staff use their own production accounts; keep daily workflows simple. Inventory queries must sum only `transactions.status='posted'` lines (voided history otherwise shows false negative lots).

Follow `CLAUDE.md` in this directory in full — it is the single source of project rules (change-log protocol, hard rules, regression guard). This file only restates the rules that have been broken by agents that did not read it.

## Global change-log: one file, one path

* The global change-log's CANONICAL PATH is `~/Library/Mobile Documents/com~apple~CloudDocs/Claude Logs/change-log.md` (iCloud Drive). Append global-log rows there and nowhere else.
* `~/change-log.md` is a symlink to that file and MUST stay a symlink. Never replace it with a regular file: no atomic write-and-rename, no `mv`/`cp` over it, no "create if missing". If `~/change-log.md` is not a symlink, stop and report it instead of writing.
* New rows go directly below the `|------|` separator row, never between the header row and the separator.
* Why: the two paths forked between 2026-09-14 and 2026-10-07 (Claude sessions wrote `~/change-log.md` after it became a regular file; Codex sessions wrote the iCloud file) and had to be hand-merged (493 rows, 2026-10-07).
