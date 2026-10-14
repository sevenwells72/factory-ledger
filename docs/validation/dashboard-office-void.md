# D2 office and void pilot — October 14

Builder: Codex. Independent reviewer: Claude Code. Keep the PR draft; do not merge.

Base verified: origin/main contains PR #107, merge 611d0c4. Worktree: ~/dev/fl-d2office-wt; branch feat/dash-office-void. The shared repository twice failed checkout with `mmap failed: Operation canceled`; an isolated bare Git store at ~/dev/fl-d2office-source.git supplies this worktree. No other checkout was modified.

## Part 1 — office

A7 prepare → reviewed draft → Record for orders, order lines, headers/status, close/cancel, expected deliveries and extracted PO/SO approval. Customer and product IDs come from /resolve; pouch conversion comes from FL. Preserve duplicate PO acknowledgement, No PO, service lines and legacy-close policy. Missing receipt searches are neutral.

## Part 2 — void

Receipt-specific FL permissions control void. Prepare with a fixed reason, commit the ticket, show its VD receipt and the original receipt’s VOIDED backlink.

## Acceptance

All new controls require key_sessions_enabled. Preview hosts use staging only. EN/ES, 390 px, per-action receipts, FL warnings unchanged and failure feedback remain required. Node and browser runners cover forms and both flag modes; full Python suite runs on one fresh throwaway DB at a time. Staging walkthrough and per-part review requests will be recorded here with exact evidence.
