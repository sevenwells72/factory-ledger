# A9 — end-of-shift summary + D3-lite page (PR: `feat/shift-summary`)

Built on `origin/main` **bc57da4** (A6 #99 + docs #100 merged) in `~/dev/fl-a9-wt`. Contract: `docs/contracts/shift-summary.md` (frozen 2026-10-08, decisions 1–4). Design: `docs/design/phase1-safe-operating-system.md` §7.3, §10 (A9 / D3-lite rows), §10.2. No merge, no production access and no hosted-staging deploy are part of this work; migration 076 is **staging only** until the owner approves the production step.

## Part 1 — API (`shift_summary.py`)

| Route | Keys | Behaviour |
|---|---|---|
| `GET /reports/shift-summary?date&actor&lang` | dashboard key, master key, every actor key (both allowlists) | Read-only. Default `date` = today's plant day, default `actor` = the caller. Shared keys with no `actor` get the **all-actors view** (`actor: null`, `can_confirm: false`, title `— all / — todos`). `actor` = id or exact name; floor keys may only ask for themselves (`403 ACTOR_NOT_ALLOWED`); unknown → `422 ACTOR_UNKNOWN`; future → `422 DATE_IN_FUTURE`. |
| `POST /reports/shift-summary/confirm` | both allowlists; **named actors only** at the handler (`403 ACTOR_REQUIRED` for the shared keys) | `{date, outcome: match\|discrepancy, notes, summary_hash, receipts_seen[, lang]}` → `201` with the §3.3 row. `discrepancy` needs notes (`422 NOTES_REQUIRED`), opens exactly one `exceptions(SHIFT_DISCREPANCY, warn, owner_actor_id = active owner, due_at NULL)`. Stale hash → `409 SHIFT_SUMMARY_STALE` with the fresh document at `detail.summary`. Re-confirm **appends** (latest wins). Per-(actor, day) advisory lock serialises double taps. |

Ten sections in fixed order (`received, made, packed, shipped, adjusted, voided, not_entered_yet, lot_balances_touched, open_exceptions, late_entries`), each with `label_en/label_es`, `empty_en/empty_es`, `count`, `rows`, and the `open_exceptions` roll-up. Every `*_en/*_es` pair carries an un-suffixed `lang` copy (`title`, `label`, `empty`, `line`, `display`, `message`, `summary`, `rollup`, `reason`).

**Where each value comes from (contract §6):**

- Receipt set: `write_tickets.receipt_numbers()` — the same query `GET /receipts` now calls (refactored out of `list_receipts`), with `[name, id]` as the actor terms. Pre-ticket history never appears.
- Rows: `write_tickets.receipt_detail()` (the `GET /receipts/{n}` reader) + the ticket's stored `payload` / `draft` / `response`. Quantities are the posted effective ledger lines of the receipt's transactions; `effective_status` is `voided` when every transaction is voided.
- Flags: `UNIDENTIFIED_LOT` (lot `identity_status` + the open exception), `SUPPLIER_MISSING`, `SHORT` (response `shortages` → flag/exception ids and due date), `LATE_ENTRY` (`entry_timing.late_entry_exception_id`), `LARGE_CORRECTION` (`correction_review` rules; exception id when a hold was approved), `UNKNOWN_REASON`, `PHOTO_PENDING` (`shipments.proof_status` + any `BOL_PHOTO_MISSING` exception), `NOT_SAME_DAY` (`policy_acknowledgements`), `POSSIBLE_DUPLICATE_ACKED` (`acknowledged` + the warning's earlier receipt), `VOIDED`. `INVOICE_PENDING` / `EXTRA_KOSHER` wait for A8 / A12.
- Reason labels: `correction_reasons` (061), never hard-coded.
- Lot balances: `lot_on_hand()` now (`after_lb`); `before_lb = after_lb − net of the listed receipts' posted lines`; `short` from open `shortage_flags`.
- Open exceptions: `exceptions` with `status IN (open, escalated)` owned by the actor (all owners in the all-actors view); overdue first, then `due_at`, `opened_at`.
- NOT ENTERED YET: open `expected_receipts` with `expected_date ≤ date` and no transaction carrying a `receipt_number`; `production_runs` on `date` still `planned`/`in_progress`. **Boundary:** `production_runs` has no ticket/transaction link column, so run status is the only link FL has today (documented in the module docstring and FOLLOWUPS P1.21).

**`summary_hash` (§1.3, decision 4):** sha256 of the canonical JSON of the seven receipt sections reduced to `receipt_number, action, status, effective_status, happened_at, entered_at, late_entry, product.id, lot.id, quantity.lb, quantity.cases, detail, transaction_ids, flags[{code, refs}]`, computed on the **serialised** rows with the un-suffixed language copies stripped — so it is identical for `lang=en` and `lang=es`, reproducible by a client from either response (`shift_summary.summary_hash`), and unmoved by balances, exception escalation, the clock, expectations or labels. Verified by tests: sweep escalation, another actor adjusting a shared lot, a new open exception, a new expectation and a 3-hour clock jump all leave the hash unchanged; the person's own new receipt or a BOL photo completing a shipment changes it.

`_structured_error` (main.py) now copies `message_es` into `error_detail` when the raised detail carries one (contract §4); the existing `{code, message}` shape is unchanged for every route that raises without it.

## Migration 076 (`migrations/076_shift_confirmations.sql`, down file beside it)

Contract §3.4 verbatim: `shift_confirmations` (identity PK, `business_date`, `actor_id → actors`, `outcome` CHECK, `notes` non-blank on discrepancy, `receipts_seen ≥ 0`, 64-char `summary_hash`, `summary_snapshot` jsonb array, `exception_id → exceptions`, `confirmed_at`), index `(business_date, actor_id, confirmed_at DESC)`, append-only trigger on the 039 `ledger_block_append_only_change()` function, RLS on with zero policies (owner-only, 058/074 pattern), marker `076_shift_confirmations`. Additive, rerunnable; the down file refuses while rows exist. Numbering checked against open branches and the staging markers: 066 (#91, on hold), 068 (staging), 072/073 (#96 PIN login), 075 (A12 `feat/kosher-tier`, already applied on staging as `075_kosher_tiers`) are taken; 074 merged; **076 is free**.

Apply order (same recipe as 074): staging first as the table owner on port 5432 inside `BEGIN; SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s'; SET LOCAL search_path=public; \i migrations/076_shift_confirmations.sql; COMMIT;` — **before** the A9 code goes live on that environment (the GET reads the table). Production only with the owner's separate approval. `tests/schema/schema.sql` carries the pending `\ir` include until the prod apply + re-dump.

## Permissions / allowlists

`DASHBOARD_KEY_ALLOWLIST |= shift_summary.ROUTES` (the GET and the POST — contract §1 "both allowlists"); actor keys reach the same set through the existing `DASHBOARD_KEY_ALLOWLIST | ACTOR_WRITE_ALLOWLIST` check. The POST is listed in `permissions.UNGATED_ROUTES` (no §4.3 row: any named actor confirms their **own** shift; the handler refuses the shared keys). `write_tickets.ACTOR_ROUTES`/`DASHBOARD_ROUTES` untouched.

## Tests

`tests/test_shift_summary.py` — 19 cases, real HTTP on the rolled-back test transaction: every section with a ticketed receipt (receive, unidentified receive, short make, pack, order ship with/without BOL photo, adjust unknown reason, found, large correction, 49-h late make, owner late entry, expectations + planned runs), legacy-key receipts and the all-actors view, `GET /receipts` parity, hash determinism/scope, confirm match → notes re-confirm → append-only, discrepancy → one owner exception + owner's view, stale 409 with fresh summary, yesterday late / future refused / extra body fields refused, the permission matrix, attribution from the key, and 076 up/down/up on an isolated database. `tests/test_write_tickets.py`'s isolated-DB fixture applies 076.

## Part 2 — D3-lite page (`dashboard/shift-summary.html|js|css`)

See the Part 2 section appended below once built.
