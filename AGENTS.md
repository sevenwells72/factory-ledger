# Agent Instructions (Codex and other non-Claude agents)

Follow `CLAUDE.md` in this directory in full — it is the single source of project rules (change-log protocol, hard rules, regression guard). This file only restates the rules that have been broken by agents that did not read it.

## Global change-log: one file, one path

* The global change-log's CANONICAL PATH is `~/Library/Mobile Documents/com~apple~CloudDocs/Claude Logs/change-log.md` (iCloud Drive). Append global-log rows there and nowhere else.
* `~/change-log.md` is a symlink to that file and MUST stay a symlink. Never replace it with a regular file: no atomic write-and-rename, no `mv`/`cp` over it, no "create if missing". If `~/change-log.md` is not a symlink, stop and report it instead of writing.
* New rows go directly below the `|------|` separator row, never between the header row and the separator.
* Why: the two paths forked between 2026-09-14 and 2026-10-07 (Claude sessions wrote `~/change-log.md` after it became a regular file; Codex sessions wrote the iCloud file) and had to be hand-merged (493 rows, 2026-10-07).
