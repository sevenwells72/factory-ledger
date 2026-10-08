# Followups

Deferred work from Pass 1 (2026-04-20). Not shipped in Pass 1 — tracked here for a
future PR.

---

## Phase 1 — next up (added 2026-10-07)

**P1.1 `/products/resolve` must not auto-pick when several products match.**
Staging smoke after the catalog copy: "Classic" auto-resolved to Granola Classic
25 LB (136) over four other Classic SKUs, and "chocolate chip" to White Chocolate
Chips (53) over Chocolate Chips Sugar Free / Real 1,000 CT / Real 4,000 CT and
Granola Chocolate Chip 25 LB — both tagged `keyword` / `medium`. When more than one
product matches, return the candidates unresolved instead of picking one.
Needs CNS shorthand aliases (SS = Sunshine, BS = Blue Stripes): "Sunshine 9"
resolved to nothing. The alias tables are nearly empty
(customer_product_aliases 2 rows, supplier_product_aliases 1 row in production).
Seed list drafted 2026-10-07: `~/Documents/fl-audits/shorthand-draft.md` (Michael's
answers applied); catalog duplicates to resolve first:
`~/Documents/fl-audits/catalog-cleanup.csv`.

**P1.2 "Pouches" → cases conversion (A4 unit resolution + F1 draft) — Michael, 2026-10-07.**
People sometimes say "pouches" for quantity, but the Sunshine 12x10 OZ items (145,
146, 147, 148, 149) are sold and stocked in cases. Rule: when the user gives a
quantity in pouches for a pouch product, FL converts to cases using that product's
pouches-per-case (12 today) and shows the conversion in the draft — "60 pouches =
5 cases — confirm". If the quantity does not divide evenly, FL asks instead of
rounding. Design doc §3.2 now records this approved conversion rule (A4 part 1,
PR #81); the old each/cases ambiguity wording is superseded. Needs a `pouches_per_case` (or
reuse of `case_size_lb` / pack format) on the product, the `/resolve kind=unit`
response to return the converted draft quantity, and F1 to render the sentence.
Non-pouch products: "pouches" stays an error, never a silent unit change.

**P1.3 F1 must log unmatched words during the pilot (D5 Spanish/shorthand collection).**
The Spanish and floor-shorthand entries in `shorthand-draft.md` are guesses; nobody
has recorded Arturo's actual vocabulary. During the F1 pilot every `/resolve` call
whose outcome is `none` or whose chosen candidate was not the top-ranked one is
to be written to `resolution_log` in A4 part 2 after A1 merges (design §3.3) — F1 must additionally store the
raw user utterance (typed text or edited transcript) with the log row, and the
dashboard Aliases tab gets an "Unmatched this week" list the office reviews to add
aliases. Review cadence: weekly with the owner view until the list is empty for two
consecutive weeks.

**P1.4 Supplier tracking on receipts — APPROVED (Michael, 2026-10-07; design §5.2).**
Today `generate_lot_code` (main:5429) stores only the first four typed letters of the
supplier in the lot code and nothing stores a `supplier_id` (production: `DUTC` on 39
lots is both Dutch Valley and Dutch Gold; the `suppliers` table is referenced by two
expected receipts). From cutover: `receive/prepare` requires a resolved `supplier_id`
(`POST /resolve kind=supplier`, 422 `SUPPLIER_REQUIRED`), stored on `transactions` and
`lots`; the lot-code prefix becomes a label from `suppliers.short_code`. A5 +½ d, D2/F1
+½ d; readiness gate 10(e). Prerequisite: merge the supplier duplicates listed in
`~/Documents/fl-audits/catalog-cleanup.csv` (reviewed separately — not applied here).

**P1.5 Classic #9 Regular vs Extra-Kosher tiers (Michael, 2026-10-07; design §5.3, row A12).**
Two products per flavour already exist (107/108 regular, 283/284 "Kosher Ignition")
with identical formulas except 284's chip size (1,000 CT vs 108's 4,000 CT — resolved: chips
are interchangeable, standard is 4,000 CT, 284's formula → 72 via `catalog-cleanup.csv`,
1,000 CT stays a recorded substitution).
Rules: owner PIN attestation on every extra-kosher make ticket; extra-kosher finished
goods (285–290) pack only from extra-kosher lots (`KOSHER_SOURCE_REQUIRED` — the
2026-08-12 pack of 63 cases of 286 from a regular 107 lot is the case this prevents — left
uncorrected as a probable recording error, per Michael);
extra-kosher lots may be packed as regular, recorded as a downgrade; `SS`+`#9` resolves
to extra-kosher only, `#9` alone shows both. A12 3–3½ d after A5 and A11; cannot slip —
Sunshine order SO-260817-001 (10,000 lb of #9 bulk) is open. Mapping proposal in §5.3
awaits Michael's confirmation; no catalog change is made by the docs. **Interim until
cutover:** extra-kosher batches logged on paper (date + lot number); cutover checklist step
1a (design §10.1) tags those lots `extra_kosher` in FL from the paper log.

**P1.6 Schedule rev 3.4 — 3 Codex lanes (Michael, 2026-10-07; design §10.2, §11 item 20).**
A3 split into A3a (tables, 1 d) + A3b; A9 runs beside A12; D3-lite (shift-summary page +
Confirm) before the pilot, the rest of D3 Nov 23–27; G1 engineering merged Oct 30 behind an
unset `READONLY_API_KEY`; A10 at cutover+1 in two stages (ledger routes Nov 23, order routes
after A7). Pilot soft-start Nov 2, gate (d) Nov 9–13, decision Nov 17, cutover Nov 20. The
"simplify for cutover" set (§10.2 item 9) is the Nov 6 lever with A8. Lane plan with dates and
the daily owner-acceptance slot: `~/Documents/fl-audits/lane-schedule.md`. Open: PR #78 records
a PR-0 "rotate at cutover" override that the design doc still needs Michael to confirm.

**P1.7 `/products/resolve` is master-key only — decide in A2 whether office/dashboard should reach it (found 2026-10-08, PR #81 rollout).**
Live check after the A4 part 1 deploy: `POST /resolve` accepts the dashboard-scoped key
(`DASHBOARD_KEY_ALLOWLIST` has `('POST', '/resolve')` and `('GET', '/aliases')`), but
`POST /products/resolve` is on the office-GPT allowlist only, so the dashboard key gets
`403 API key not authorized for this endpoint` and the bulk OCR/order-confirmation path
needs the master key. Not a regression — the route was never on the dashboard list — but
A2 (key kinds / scoped keys) must decide whether office and dashboard clients should call
the bulk resolver, and if so add the `('POST', '/products/resolve')` pair to the relevant
allowlist(s) with a test, or document that bulk resolution stays master/office-key only.

**P1.8 `transactions.reason_code` after migration 061 — NULL window until A3b, and the effective view does not expose it (found 2026-10-08, PR #85 review).**
061's backfill is one-time at apply: every `type='adjust'` row existing then gets a code
(legacy map → already-new code → count-like text → `unknown`). Adjust and found rows that
PR #83's tickets insert between the prod apply and A3b carry `reason_code NULL`, because
nothing sets it at INSERT until A3b (`transactions` is append-only, so it cannot be
patched afterwards without the 046/061 trigger dance). 061 is rerunnable as a sweep
(`WHERE reason_code IS NULL`), so A3b's rollout should either re-run 061 once after its
deploy or include the sweep in its own migration. Separately, `ledger_current_transactions`
has an explicit column list and `effective_record` is `to_jsonb(t.*)`, so the new column is
only visible in the jsonb blob — A3b/A9 must add `reason_code` to the view (and to
`_TRANSACTION_AMENDABLE_FIELDS` if a reason may be amended) before reading it through
effective rows. The §7.3 contract (`docs/contracts/shift-summary.md` §2.2.1) reads
`reason_code` from the receipt's transaction, so this is on A9's path.

---

**P1.9 Create the `expire-tickets` Railway cron service before the F1 pilot (deferred by Michael 2026-10-08 after PR #83 rollout).**
`scripts/expire_tickets.py` is live but nothing schedules it (commit still enforces expiry
synchronously, so this is hygiene, not safety). Before pilot: add a service from the same
repo/branch as FastAPI with `ENVIRONMENT=production`, `PRODUCTION_DATABASE_HOST` = the host of
FastAPI's `DATABASE_URL` (`aws-1-us-east-1.pooler.supabase.com`), and
`DATABASE_URL=${{FastAPI.DATABASE_URL}}` as a reference variable; dashboard-set schedule
`15 7 * * *` + start command `python scripts/expire_tickets.py`. No `railway.cron.json`
(Railway config-as-code is deprecated after 2026-12-01), so the shared `railway.json`
watch patterns apply and `scripts/` changes require a manual redeploy of that service.
First manual run: restart its deployment, expect `Expired N prepared tickets` in the log,
verify read-only that only `write_tickets` changed. Same recipe for FastAPI-staging with
`ENVIRONMENT=staging` + the staging guard variables. See
`docs/deployments/a1-write-tickets-part2.md` "Nightly ticket expiry".

**P1.10 Sunshine billing rules — decided by Michael 2026-10-08 (design rev 3.7, §8.5, §11 items 27–31); invoicing trigger and price list still OPEN.**
Decided: (1) yield credits = **none** — CNS supplies all ingredients, no expected-vs-actual
adjustments of any kind; (2) Sunshine is billed only for **finished product sold** —
pouches (12x10 oz) and Mini 100 **per case**, bulk granola **per actual lb sold**; (3) one
batch may go partly to bulk and partly into pouches/minis, FL tracks the split from the
batch, only quantity *sold as bulk* is billable as bulk, packed quantity is billed only
through the finished product, never double-billed, bulk awaiting packing is not billable;
(4) **automatic invoicing at packing is NOT approved** — every "invoiced at pack" wording
in the design (R7, §7.2, §7.3, §8.4, A8 `pouch_pack`) is superseded and the trigger is
**unresolved**; (5) **QuickBooks is the source of truth for prices** — the FL price list
(design §8.5.3) stays **DRAFT** until every row is reconciled with QBO and confirmed by
Michael; conflicts are flagged, never guessed. No FL data and no QBO data were changed.
Read-only lookup 2026-10-08 (design §8.5.2): 9 of the 10 QBO rows map to FL products
(146, 145, 147, 148+149, 285 [+286/287], 185, 183, 288 [+289/290], 184); **unmapped:**
Chocolate Chip #9 Mini 100 (QBO 144) — no FL product. Found on the way: QBO "Low Carb
12x10" (72) covers two FL products (148, 149); FL 185 "SS Mini 100" is inactive, per-lb and
parented to batch #1 (116), not #9; FL ships TX2535/TX2536 to Sunshine on 2026-09-30 (600 cs
of 145 + 1,500 cs of 146, SO-260909-001) have no QBO invoice after Aug 13.
**Open for Michael** (design §11 after item 31): Original #1 bulk 1.97 or 1.85; create a QBO
item for Chocolate Chip #9 bulk at 1.70/lb; is Original #9 Mini 100 really a case (fractional
quantities billed) and how to fix FL 185; B'gan Chocolate sold or retired; **when an
invoice is created**; keep or split QBO Low Carb item 72; the Sep 30 ships' invoice status;
create an FL product for Chocolate Chip #9 Mini 100. A8 cannot build `invoice_triggers`
creation until the trigger is decided; it can still build `bins`, `bulk_dispatch_lines`
and the ownership flag.

---

**P1.11 A5 part 2 — supplier-lot correction ticket (resolves UNIDENTIFIED_LOT inside FL).**
**Owner: Codex. Due: before Nov 2 pilot.** Add the reason-coded supplier-lot
correction prepare/commit flow with attribution and atomic resolution of the
lot's open UNIDENTIFIED_LOT exception. Another receive must never clear it.

**F1/D2 A5 contract:** `lot_confirmations[].method='last4'` means the literal
last **four characters**, including the hyphen: lot `26-10-08-DUTC-004` needs
`value="-004"` (not `"004"`). For `method='pallet'`, `value` must echo the **full
lot code**, plus a matching latest move to production within 24 hours. Pass the
operator's evidence; never fabricate it from FL's suggested lot.

**A5 supplier labels / deferred 066:** Michael must clean up the Dutch Valley
duplicates and `DUTC Valley` typo before the separate supplier-label backfill PR
is applied. A5's nullable label and supplier-ID schema is in 064 and works before
066. Re-run the read-only label dry-run after cleanup; today's labels are a
preview of the current catalog, not an approved assignment.

---

## 1. Backfill NULL addresses on recurring customers

**Context.** During Pass 1 we added an address-similarity tiebreaker to
`resolve_customer_id` so the GPT can supply `customer_address` from a PO and
silently collapse fuzzy name matches to the correct existing customer.

**Finding.** The tiebreaker only helps when the existing row actually has an
address on file. Spot-check of production:

```
SELECT id, name, address FROM customers WHERE LOWER(name) LIKE '%setton%';
→ {"id": 5, "name": "Setton Farms", "address": null}
```

Setton Farms is one of the top recurring customers. Its `address` is NULL,
which means the tiebreaker cannot fire for Setton POs until the row is
backfilled. A broader audit of top customers in `customers` is needed — many
are likely in the same state.

**Action.**
- Audit: list active customers by order volume where `address IS NULL`.
- Backfill: pull canonical addresses from the most recent PO / order-confirmation
  attachments and UPDATE the `customers.address` column.
- Do not auto-derive from sales_orders alone — PO addresses have historically
  drifted.

**Out of scope for this PR** — don't let a data cleanup block the code change.

---

## 2. Normalize the remaining ~25 4xx raise sites to dict shape

**Context.** Pass 1 normalized 10 raise sites across the two product resolvers,
`resolve_order_id`, `resolve_customer_id`, and the 5 sales-order endpoints
(`createOrder`, `getOrder`, `updateOrderHeader`, `addOrderLines`,
`updateOrderLine`). All now return the standard structured error shape:

```json
{"detail": {"error_code": "...", "message": "...", "input": "...", "suggestions": []}}
```

**Remaining work.** The following GPT-facing endpoints still raise
`HTTPException` with plain-string detail. When the GPT hits one of these it
falls back to the old "something went wrong" handling because there is no
`detail.error_code` to parse — the exact failure mode Pass 1 was trying to
kill.

| Endpoint | Location | Current | Proposed normalization |
|---|---|---|---|
| `/ship` | main.py ~2370, ~2476 | 400 "No inventory available for..." | 409 INVENTORY_EMPTY |
| `/ship` | main.py ~2374, ~2481 | 400 "Insufficient total inventory..." | 409 INSUFFICIENT_INVENTORY (with `available_lb`, `needed_lb`) |
| `/ship` | main.py ~2381, ~2458 | 404 "Lot '...' not found or empty" | 404 LOT_NOT_FOUND |
| `/ship` | main.py ~2535 | 400 (long validation message) | 400 VALIDATION_ERROR |
| `/make` | main.py ~2775, ~2786 | 400 make rejected / batch size 0 | 400 MAKE_REJECTED |
| `/make` | main.py ~2882, ~2904 | 400 ingredient inventory | 409 INSUFFICIENT_INGREDIENT (with ingredient_id) |
| `/pack` | main.py ~3096, ~3169 | 400 case weight required | 400 CASE_WEIGHT_REQUIRED |
| `/pack` | main.py ~3181, ~3208 | 400 batch inventory | 409 INSUFFICIENT_INVENTORY |
| `/pack` | main.py ~3199 | 400 lot not found/empty | 404 LOT_NOT_FOUND |
| `/pack` | main.py ~3204, ~3219 | 400 allocation mismatch | 400 ALLOCATION_MISMATCH |
| `/pack` | main.py ~3250, ~3290 | 400 add-in insufficient | 409 INSUFFICIENT_INGREDIENT |
| `/pack` | main.py ~3380 | 404 "Lot '...' not found for product" | 404 LOT_NOT_FOUND |
| `/adjust` | main.py ~3521 | 404 lot not found | 404 LOT_NOT_FOUND |
| `/lots/by-code` | main.py ~1808, ~1856 | 404 string | 404 LOT_NOT_FOUND |
| `/lots/{lot_code}/supplier-lot` | main.py ~1913 | 404 string | 404 LOT_NOT_FOUND |
| `/trace/supplier-lot` | main.py ~1770 | 404 string | 404 SUPPLIER_LOT_NOT_FOUND |
| `POST /customers` | main.py ~4762 | 409 "already exists" | 409 CUSTOMER_DUPLICATE |
| `PATCH /customers/{id}` | main.py ~4777/4789/4794/4824 | 400/404/409 strings | consistent dict shape |
| `PATCH /sales/orders/{id}/status` | main.py ~5372/5376 | 400 string | 400 INVALID_STATUS_TRANSITION |
| `PATCH /sales/orders/{id}/lines/{line_id}/cancel` | main.py ~5778 | 404 string | 404 LINE_NOT_FOUND |
| `POST /sales/orders/{id}/ship` | main.py ~5658/5660/5706/5708/5728 | 400/404 strings | consistent dict shape |
| `/production/day-summary` | main.py ~8090 | 400 date format | 400 INVALID_DATE |

(Line numbers drift — grep by message string before editing.)

**Suggested grouping for the PR.** One endpoint-family per commit:
`/ship`, then `/make`, then `/pack`, then `/adjust`, then customer CRUD, then
the remaining sales-order endpoints. Keeps diffs reviewable and makes
bisection easy if anything regresses.

**Add OpenAPI response blocks** for each operation at the same time so the
error shape is contractual. Reference the existing
`components.schemas.ErrorResponse` added in Pass 1.

---

## 3. Tune `_pick_by_address` thresholds after real traffic

**Context.** Current defaults are 0.6 absolute / 0.2 gap, chosen conservatively
for Pass 1. The Setton case ("85 Austin Blvd, Commack, NY 11725" vs whatever
Setton Farms gets backfilled to — see #1) specifically may or may not clear
the gap.

**Action.**
- Revisit after 2–4 weeks of real order-entry traffic.
- Instrument: log every `_pick_by_address` call that falls through to the 409
  (no winner picked), capturing the top two `addr_sim` scores and the names.
- From those logs, check whether loosening to 0.5 / 0.15 would have helped
  without creating false positives.
- Watch for false negatives where the gap check rejects legitimate matches —
  especially multi-tenant addresses or customers with multiple locations in
  the same city.
- **Tune from data, not intuition.**

---

## 4. GPT instruction headroom

**Context.** GPT instructions at 7,987 / 8,000 chars — 13 char headroom. Next
instruction edit will likely overflow.

**Action.**
- Before adding any new instruction content, free ~200+ chars via ROUTING
  RULES consolidation. The section has visible redundancy across the intent
  hierarchy and endpoint-specific sections that wasn't trimmed in Pass 1.

**Estimate calibration.** Pass 1 estimated ~52 chars of additions; actual was
91 — a 39-char delta. Before the next edit, review the Pass 1 diff
(`git log -p -- gpt-instructions-v3.md` around 2026-04-20) and identify where
the extra chars went, so future estimates are more accurate. If the source of
the overage can't be pinned down from the diff, record "delta unaccounted —
review diff before next edit to calibrate estimates" here and treat the next
pass's estimate as lower-bound only.

---

## 5. Audit createOrder auto-create default for typo-duplicates

createOrder calls resolve_customer_id without auto_create=False, meaning a
typo'd customer name (e.g., "Setton Fams" instead of "Setton Farms") with no
address provided will silently create a duplicate customer row rather than
raise CUSTOMER_AMBIGUOUS or prompt for disambiguation. The address tiebreaker
mitigates this when address is present, but address is optional. Consider
adding a name-similarity guard before the auto-create branch fires — if the
new name has >0.7 trigram similarity to any existing customer, raise
CUSTOMER_AMBIGUOUS with the near-matches as suggestions instead of
auto-creating. Tune threshold after observing real traffic.

---

## 6. `factory-ledger` service in `gleaming-solace` is crashlooping

**Context.** During Pass 1 verification (running pytest via `railway ssh`
into the FastAPI service container), discovered the sibling `factory-ledger`
service in the same Railway project (`gleaming-solace`) is crashlooping with
`password authentication failed for user "postgres"` against the Supabase
pooler.

**Impact.** Prod traffic is **unaffected** — the `FastAPI` service
(`fastapi-production-b73a.up.railway.app`) is what serves GPT requests and
that service is healthy. But the dead `factory-ledger` service is consuming
restart budget and polluting the project's deployment log.

**Action.** One of:
- Fix its `DATABASE_URL` if it's meant to be running (pull the working value
  from the `FastAPI` service's env and set it on `factory-ledger`).
- Delete the service if it's a leftover from a rename / split.
- Document what it was intended for if keeping it around for future use.

Cheap to investigate — 5 minutes in the Railway dashboard. Worth closing out
before it becomes unexplained project clutter.

---

## 7. Staging follow-ups from PR #71 review (2026-10-07) — next staging PR

Non-blocking items found while reviewing and rolling out PR #71 (`infra/staging`,
merged as `8a56f92`). Do not implement outside a PR.

**7a. `tests/test_seed_staging.py` is order/DB-state dependent.**
`test_seed_is_idempotent_and_uses_no_known_actor_key` calls `sync_sequences()`,
which `setval`s `products_id_seq` (and others) to ≥1,000,000,000. `setval` is
non-transactional, so on a reused test DB every later committed row from the
race/lock tests lands in the fixture range and the `count(*) WHERE id >= FIXTURE_BASE == 3`
assertion fails (`15 == 3` observed). Passes on a fresh DB (full suite 1,471 green).
*Action:* count only the seed's own rows (`id BETWEEN FIXTURE_BASE+1 AND FIXTURE_BASE+3`
or `name LIKE 'STAGING %'`), or don't call `sync_sequences` in the test.

**7b. Document that the inlined staging launcher is intentionally kept.**
After the merge, `main.py` carries the guard itself, so the 3,130-char `python -c`
start command on FastAPI-staging is redundant. Keep it as defence in depth
(it also enforces `ENVIRONMENT=staging`), and say so in `docs/staging.md` so
nobody "cleans it up" later. It must be regenerated with
`scripts/staging_start_command.py` whenever `staging_safety.py` changes.

**7c. Fixture alias rows collide with `--copy-master-data`.** (Found on the
first real run, 2026-10-07.) `seed_fixtures()` inserts the synthetic
`customer_product_aliases` / `supplier_product_aliases` rows without explicit ids,
so they take serial id=1 — which is exactly production's alias id=1, and the copy
aborts with `customer_product_aliases_pkey` duplicate key (correctly rolled back,
staging untouched). Worked around in staging by moving both rows to id 1000000001
and rerunning. **Fixed in PR `docs/staging-followups`:** both inserts now use
explicit `FIXTURE_BASE+1` ids and
`test_fixture_aliases_use_reserved_ids_so_source_id_1_copies` seeds fixtures then
copies id=1 source rows. (Staging's hand-moved rows already sit at 1000000001.)
