# Daily Verification module — scope (read-only audit)

Date: 2026-11-01 · Audited: `origin/main` `87f500f` (worktree `~/dev/fl-verify-scope-wt`) plus production DB, read-only
(session pooler port 5432, every query inside `BEGIN TRANSACTION READ ONLY; … COMMIT;`). No code changes, no DB writes, nothing committed.

> Note: `~/Documents/factory-ledger` is a stale detached checkout (`0fa518f6`, Oct 15). Everything below is from current main.

## Live facts this scope relies on (production, 2026-11-01)

| Fact | Value |
| --- | --- |
| Correction reasons (`correction_reasons`, migration 061) | 8 codes: `physical_count`, `missing_receipt`, `missing_production`, `wrong_lot`, `damage_disposal`, `unrecorded_usage`, `data_entry_error`, `unknown` (note required). No `count_correction`. |
| Product types | `ingredient` 78, `finished` 75, `packaging` 32, `batch` 25 (= bulk granola WIP) |
| Lots with posted balance > 0 | ingredient 75 (411,977 lb), finished 50 (183,788 lb), batch **8** (7,868 lb), packaging 2 |
| Actors | Blubber (owner), Arturo (floor), Luz (office), Miriam (office) |
| Attribution gap | Since 2026-10-28, **18 of 36** transactions have no `entered_by_actor_id`: they were posted through `operator_id='legacy-shared-key'` (7 make, 6 pack, 4 adjust, 1 receive) |
| `shift_confirmations` | **0 rows ever.** The end-of-shift sign-off (A9) is live, and in Arturo's guide, but nobody has used it |
| `attachments` (private `evidence` bucket) | Live in production, 7 rows, all `bol` |
| Exceptions | LARGE_CORRECTION resolved 62; UNIDENTIFIED_LOT open 3, waived 7 |
| Activity (posted, by `business_date`) | Oct 28–30: 16/10/10 a day; no rows Oct 24–26 or Oct 31 |
| Backdating signal | `transactions` has `business_date`, `occurred_at`, `created_at`, `entry_backfilled`. Oct 22 (reset): 11 entered after the day, 129 backfilled. No late entries since. |
| Shipments / receipts since Oct 28 | 8 ship, 1 receive. `purchase_documents` 6 |
| OpenAPI | Office `openapi-gpt-v3.yaml` = **30/30** operations. Floor `openapi-floor.yaml` = 22, pinned by `tests/test_batch1_correctness_security.py:241` |
| Workers | Railway cron services `expire-tickets` and `exceptions-sweep` (`scripts/exceptions_sweep.py`, 02:30 ET), plus staging copies |
| Next migration number | 090 |

---

## Part 1 — Daily blind spot count (3 lots/day: ingredient, bulk granola WIP, finished good)

**Reuse**
- `correction_reasons` table (reason picker and `note_required` for `unknown`); `exceptions_enforcement.resolve_reason()`, which validates a reason for a given source.
- Posting a correction: the existing ticketed adjust path (`ticket_actions.py` adjust → `POST /adjust/commit`, `exceptions_enforcement.correction_review`). It already applies the >500 lb rule (`photo_required` / hold for owner approval, `LARGE_CORRECTION` exception) and `close_shortages_by_count`.
- Posted-only lot balance: `lot_on_hand()` in main.py, or the posted-only effective lines used by `shift_summary.lot_balances`.
- Named-actor enforcement: `attachments.named()` / `permissions.NAMED`; `api.get_plant_now()` and `PLANT_TIMEZONE` for the plant day.
- Photo evidence for a large variance: `POST /attachments` (kinds include `correction_photo`, `count_sheet`).
- Product type split: `products.type` in (`ingredient`, `batch`, `finished`).

**Missing**
- Nothing picks lots, stores assignments or counts. No table name contains `count` or `verif`; earlier blind counts were offline PDFs (`audits/fresh-start/build_v3_materials.py`).
- A blind entry flow. Adjust preview (`POST /adjust`, ticket preview) **shows the book balance first** (`quantity_on_hand → new_balance`), so it can't be reused as the count screen.
- Unit handling for finished goods. Counters count cases, so the count needs a `case_size_lb` conversion. The batch-lot pool is thin: 8 lots, so the same WIP lot will come up often.

**Smallest build — M**
1. Migration 090 creates `spot_count_assignments`: business_date, lot_id, category, picked_at, a seed so the pick can be reproduced. It also creates `spot_counts`: assignment_id, counted_qty, unit, counted_lb, expected_lb snapshot taken *at submit*, variance_lb, reason_code FK → `correction_reasons`, note, actor_id, counted_at, attachment_id. Both tables append-only with the 039 trigger and RLS on, like 076.
2. `GET /verification/spot-counts/today` picks the day's 3 lots on first call, randomly from lots with posted balance > 0 in each category, seeded by date. It returns only lot code, product, location and unit, **never the expected quantity**.
3. `POST /verification/spot-counts/{id}` takes the counted quantity from a named actor and computes expected from posted lines at that moment. If the variance is beyond tolerance, reason (and note for `unknown`) is required. The response only then reveals expected and variance.
4. A variance is recorded only. It never auto-posts. A separate "Post correction" button runs the existing adjust ticket with `physical_count` (or the chosen reason), so the >500 lb photo/owner rule still applies. That matches "corrections are new records with reason, receipt, named actor".
5. One dashboard page (`spot-count.html`) plus an optional FL Assistant intent. Keep the routes out of both GPT schemas (`include_in_schema=False`).

Open owner decisions: variance tolerance (e.g. ±2 % or ±5 lb); whether to skip lots with activity in the last N hours (counting mid-move); whether to exclude POOLED, held or zero-movement lots.

---

## Part 2 — Automated outside cross-check (QBO invoices vs FL shipments; QBO received POs vs FL receipts)

**Reuse**
- FL side: shipments (`transactions.type='ship'`, `shipments`, `sales_order_shipments` → `sales_orders.customer_po` and order number); receipts (`transactions.type='receive'` with `expected_receipt_id` → `expected_receipts`, `purchase_documents` with extracted `po_number`); the dashboard's Shipped export (CSV/Matrix), which Liz uses for the weekly manual compare (`docs/operating-model/04-guide-liz.md`).
- Exceptions table / dashboard Exceptions page as the place the morning list appears.
- Intuit app-review pages exist (`dashboard/legal/eula.html`, `privacy.html`, describing a read-only PO sync).

**QBO PO sync status: does not exist.** There is no QuickBooks code in the app: no `qbo`/`intuit` references in `main.py` or workers, no tables, no OAuth/token storage, no env var read. `docs/operating-model/01-architecture.md` says "Not connected"; F-18 in `07-issues-and-recommendations.md` says manual weekly compare is enough. **I couldn't confirm production keys are live.** Even if Intuit production keys were issued, nothing in FL reads them. I did not open Railway variables, to avoid printing secrets. The only QBO access today is the QuickBooks connector inside Michael's Claude sessions. It has invoice/sales tools and no purchase-order or bill read.

**Missing:** everything. Intuit OAuth2 app (production approval), refresh-token storage and rotation, QBO API client, matching rules, mismatch storage.

**Smallest build**
- **Step A — S (no QBO integration):** a server report `GET /verification/cross-check/fl-side?date=` lists yesterday's FL shipments (customer, SO, customer_po, lb/cases) and receipts (supplier, PO/expected receipt, lot, lb). A scheduled Claude task, or Liz, compares invoices using the existing QuickBooks connector and records the result. This covers the invoices-vs-shipments half. POs are still manual.
- **Step B — L (real integration):** Intuit OAuth (consent once by Michael; tokens in a new `integration_tokens` table or Railway secret, refresh every run, 100-day refresh-token expiry). A nightly job reads QBO `Invoice` (by TxnDate, `CustomerRef`, `DocNumber`/PO memo) and `PurchaseOrder` plus linked `Bill`/`ItemReceipt` (QBO has no "received" PO state; Closed/linked is the proxy). It matches on SO number or customer PO, and on PO number, then writes `cross_check_results` rows and opens exceptions for unmatched or quantity-mismatched items. Matching keys need cleanup first: QBO invoice ↔ FL SO linkage isn't stored anywhere today.

At current volume (8 shipments and 1 receipt in 5 days) Step A gives most of the value. Do Step B only if Step A keeps finding errors.

---

## Part 3 — End-of-day closeout (today's entries, photo of production board, "Day closed")

**Reuse — most of this exists**
- `GET /reports/shift-summary` and `POST /reports/shift-summary/confirm` (`shift_summary.py`, contract `docs/contracts/shift-summary.md`). Built on receipts from `shift_receipts.load`, posted-only balances, voided rows flagged, `summary_hash` of exactly what was seen, outcome `match`/`discrepancy` (notes required on discrepancy).
- Dashboard page `dashboard/shift-summary.html` / `shift-summary.js`, and the FL Assistant "resumen del día" / MCP `endOfShift`. Already in Arturo's guide as "Close the day" (`docs/operating-model/03-guide-arturo.md` §End of day).
- `shift_confirmations` table (migration 076, append-only, `business_date`, `actor_id`, `confirmed_at`, `confirmed_late`).
- Photo upload: `POST /attachments` with the private evidence bucket, already live in production.
- Backdating data: `transactions.business_date` vs `created_at` (plant time), `entry_backfilled`, `created_at_source`.
- `exceptions_enforcement.business_deadline()` (Mon–Fri business days).

**Missing**
- A photo on the confirmation. `shift_confirmations` has no attachment column, and `attachments` has no `shift_confirmation_id` or suitable kind (the kind CHECK lists `bol, truck_photo, correction_photo, lot_tag, count_sheet, signed_packing_slip, other`).
- A plant-wide "day closed" status. Today it is per actor, and **18 of the last 36 postings came from the legacy shared key with no actor**, so a per-actor summary misses half the day.
- Flags: no-activity business day; missing closeout; entries backdated after the day was closed. There is also no `plant_closure_days` table (FOLLOWUPS P1.12), so holidays would show up as false "no activity" alerts.
- Adoption: 0 confirmations so far. The feature exists but isn't part of anyone's routine.

**Smallest build — S/M**
1. Migration: add kind `production_board` to the `attachments` CHECK. Add a nullable `attachment_id` FK on `shift_confirmations` (ADD COLUMN is allowed; the append-only trigger only blocks UPDATE/DELETE).
2. Confirm accepts an optional `attachment_ref` (required for the plant-wide floor close). The page gets a camera/upload button and the existing confirm buttons, relabelled "Day closed".
3. A closeout check function, also used by Part 5, returns for date D:
   - `no_activity`: a business day with zero posted transactions and not a closure day;
   - `missing_closeout`: a business day with no floor confirmation by 23:59;
   - `late_entries`: rows with `business_date = D` and plant-local `created_at` after the latest confirmation (or after D), plus `entry_backfilled`.
   Optionally open an exception (new kind, same migration).
4. Optional and small: `plant_closure_days` (P1.12), so no-activity alerts skip holidays.

Separate and needed first: retire the `legacy-shared-key` write path for Floor GPT make/pack (principle 8, handoff item 10). Otherwise the "who entered it" column of the closeout stays half empty.

---

## Part 4 — Traceability challenge 2×/week (random finished lot, forward + reverse, saved and timed)

**Reuse**
- `GET /trace/batch/{lot_code}` returns the lot, its ingredient lots (reverse) **and** customer shipments with SO, customer and shipment id (forward).
- `GET /trace/ingredient/{lot_code}` (forward from an ingredient) and `GET /trace/supplier-lot/{code}` (recall exposure, including the `lot_supplier_codes` commingled table).
- `GET /dashboard/api/production/trace` (make/pack trace by product/date/lot) and `GET /production/day-summary`.
- Unidentified, pooled and provisional lot markers: the `UNIDENTIFIED_LOT` exceptions, the #154 POOLED guard, and provisional lot notes.

**Missing**
- Picker, persistence, timer, pass/fail rules.
- To verify before building: whether `/trace/batch` on a **pack** (finished) lot walks back through its parent bulk-granola lot to ingredient lots, or stops one hop up. If it stops, the challenge does two hops: FG → batch → ingredients → supplier lots.

**Smallest build — S/M**
1. Migration: `trace_challenges` table with lot_id, picked_at, started_at, completed_at, actor_id, result snapshot (jsonb), outcome (`pass` / `pass_with_known_gaps` / `fail`), notes. Append-only.
2. `POST /verification/trace-challenges` picks a random finished lot with production in the last 90 days, Tue/Fri or on demand. `POST …/{id}/complete` composes the existing trace functions server-side and stores the snapshot and duration.
3. Pass rules: every input lot resolves to a supplier lot, or is explicitly marked unidentified, pooled or provisional. Shipments reconcile with posted ship lines. Known gaps (e.g. pooled Coconut Flake 26-10-27-FOUND-011) are labelled `pass_with_known_gaps`, never presented as verified (principle 2).
4. Timing: the clock starts when the person opens the challenge and stops when they submit, which measures the person's response time, not API latency. Add a "mock recall" mode where a person must produce the customer list in a set number of minutes (FSMA-style).

---

## Part 5 — Daily owner email (pass/fail per check, exceptions only)

**Reuse**
- The scheduled-worker pattern: `scripts/exceptions_sweep.py` / `expire_tickets.py` with the production/staging DB guards (`staging_safety.py`), deployed as Railway cron services. Doc: `docs/deployments/a3b-exceptions-enforcement.md`.
- Data sources: Parts 1–4 plus open or escalated `exceptions` and `shortage_flags`.

**Missing:** there is **no email or notification code anywhere**: no SMTP, SendGrid, Resend, Postmark or webhook. `scripts/daily-health-ping.sh` only curls `/health`. A provider account, sender domain (SPF/DKIM for sevenwellsgranola.com) and API-key secret are all needed.

**Smallest build — M**
1. `verification_report.py`: one function `build_report(cur, day)` returning `[{check, status: pass|fail|skipped, items:[…]}]` for spot count done and variances, cross-check mismatches, closeout (missing / no activity / late entries), trace challenge (due/failed), and new or escalated exceptions. Also exposed read-only as `GET /verification/daily?date=` for the dashboard.
2. `scripts/daily_verification_email.py` renders a plain-text and HTML email showing only failures (a one-line "all checks passed" otherwise) and sends it through one transactional provider over HTTPS.
3. Run it **inside the existing `exceptions-sweep` cron** (02:30 ET, after the plant day closes) rather than a new service. That avoids a seventh Railway service while hosting is being reconsidered. If a later send time is wanted, add a second schedule only then.
4. Each send is recorded (`verification_runs`: day, sent_at, status hash) so a rerun doesn't double-send.

---

## Conflicts with standing constraints

| Constraint | Impact | How to stay inside it |
| --- | --- | --- |
| **30-op OpenAPI cap** | Office schema is at 30/30; Floor at 22 (test-pinned). Any new GPT action breaks a cap or a test. | No new GPT operations. All verification routes `include_in_schema=False` (like `/adjust/preview`), used by the dashboard and FL Assistant only. Custom GPTs stop after **Dec 11** anyway (handoff item 15), so plan these as MCP/plugin tools in that migration, not as GPT ops. |
| **`feat/planner-v2` untouched** | `origin/feat/planner-v2` exists. Nothing here touches `production_schedule`, `production_runs` or planner files. | Build in a fresh worktree off `origin/main`; never check out or rebase planner-v2. |
| **Three changelogs** | Each PR must update `CHANGE_LOG.md` (project), `~/change-log.md` (symlink to iCloud; must stay a symlink) and `FACTORY_LEDGER_CHANGELOG.md` (regression row with "Breaks If Reverted" after each deploy). | Budget a changelog step per PR. Migrations 090+ each need a backup, Michael's per-step approval, staging first and verification after. |
| **Render migration** | No Render migration plan exists in this repo: only stale `onrender.com` placeholders in `DEPLOYMENT.md`. Also Railway config-as-code stops **Dec 1** (handoff item 16). New Railway cron services add hosting-specific wiring to move later. | Add no new services. Put the email job inside the existing `exceptions-sweep` run; write jobs as plain `python scripts/…` with env vars only (host-agnostic); add nothing to `railway.json`. Michael: confirm the Render plan and timing before Part 5 ships. |
| Operating principles | A blind count must not silently "fix" stock. | Variances are recorded only. Posting is a separate adjust with reason, receipt and named actor; >500 lb needs a photo or owner approval. Pooled/provisional lots are reported as such. Counts are summed from posted lines only. |
| Named accounts | Spot counts and closeouts need a person; the shared key can't own them. | Reuse `named()`; refuse shared keys. Fix the legacy-shared-key Floor path first (Part 3 note). |

## Suggested order and size

| # | Item | Size | Depends on |
| --- | --- | --- | --- |
| 0 | Retire legacy-shared-key writes / get Arturo onto his own key on every path | S | — (handoff item 10) |
| 3 | Closeout photo + day-closed + flags (+ `plant_closure_days`) | S/M | migration 090 |
| 1 | Blind spot count | M | migration (same or 091) |
| 4 | Trace challenge | S/M | confirm the multi-hop trace on a pack lot |
| 5 | Owner email inside the `exceptions-sweep` run | M | email provider plus DNS; Parts 1/3/4 report functions |
| 2A | FL-side cross-check report + Claude/QuickBooks-connector compare | S | — |
| 2B | Real QBO integration (OAuth, invoices + POs) | L | Intuit production approval; only if 2A shows recurring errors |

One combined migration (090) for Parts 1, 3 and 4 keeps the production approvals to one backup-and-apply step.
