# Followups

Deferred work from Pass 1 (2026-04-20). Not shipped in Pass 1 — tracked here for a
future PR.

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
