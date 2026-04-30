# Pack-Preview Response Contract v1

**Status:** Locked
**Version:** 1.0
**Last updated:** 2026-04-30
**Companion document:** `WARNING_CATALOG_v1.md`

---

## 1. Overview

The `pack-preview` endpoint is the primary planning aid for the order-to-ship workflow. Given a sales order, it returns a complete, structured picture of:

- What the customer ordered
- What's already been packed and/or shipped against this specific order
- For lines that still need work: the mapped batch source, available lots, and a FIFO-default lot allocation that the operator can confirm or modify
- Operational warnings (allergen-name conflicts, brand mismatches, missing supplier lot codes, etc.) at the appropriate scope (line, lot, or allocation)

It does **not** mutate state. Every call is a planning call. The companion `pack-commit` endpoint (designed separately) consumes the same allocation shape and writes the actual transactions.

**Who calls it:**

- The Factory Ledger Custom GPT, when the operator references an order or uploads a BOL (BOL handling lives in the separate `reconcile-preview` endpoint, not here)
- The dashboard, when rendering an order's pack status
- Internal scripts and audits that need a structured view of order fulfillment

**What problem it solves:** The pre-v1 workflow required 10–15 conversational turns between operator and GPT to plan and execute a multi-line order. Each turn re-fetched data and asked the operator to disambiguate things the system already knew. This endpoint collapses the planning phase to a single API call returning everything needed for one consolidated confirmation.

---

## 2. Request Contract

```
POST /orders/{order_id}/pack-preview
```

**Path parameters:**

- `order_id` (integer, required) — internal `sales_orders.id`. The endpoint also accepts the customer-facing order number (`SO-260417-001` etc.) at the existing `resolve_order_id()` layer; both work.

**Request body (optional):**

```json
{
  "lot_allocations": {
    "<line_id>": [
      {"lot_id": 8501, "lb": 600},
      {"lot_id": 8502, "lb": 200}
    ]
  }
}
```

**Body semantics:**

- If body is omitted or empty, the API computes FIFO defaults for every line that needs packing.
- If body contains entries for some lines, those are treated as operator-specified allocations; the API validates them and re-computes the response. Lines without an entry get FIFO defaults as usual.
- The API is stateless. Multiple rounds of refinement (operator changes a lot, re-calls the endpoint, gets updated response) require no session, no preview-id, and no server-side state.

**Authorization:** Whatever the existing API-key auth requires; no new permissions for v1.

---

## 3. Response Contract

Always returns HTTP 200 unless the order doesn't exist (404) or the API key is invalid (401). A cancelled order returns 200 with a sparse response — see §5 (Cancelled Orders Short-Circuit).

### Top-level fields

| Field | Type | Nullable | Description |
|---|---|---|---|
| `order_id` | integer | No | Internal `sales_orders.id` |
| `order_number` | string | No | Customer-facing order number (e.g., `SO-260417-001`) |
| `customer_id` | uuid | No | Internal customer ID |
| `customer_name` | string | No | Customer's display name |
| `order_status` | enum | No | One of: `open`, `partially_shipped`, `fully_shipped`, `closed`, `cancelled` |
| `is_blocked` | boolean | No | True if any line has a `block`-severity warning |
| `requires_acknowledgment` | boolean | No | True if any line has an `acknowledge`-severity warning AND `is_blocked` is false |
| `is_actionable` | boolean | No | True if at least one line has status `partial`, `needs_pack`, or `needs_make` |
| `summary` | object | No | Aggregate counts; see below |
| `lines` | array | No | One entry per order line; see below. May be empty for cancelled orders. |
| `generated_at` | string (ISO 8601 UTC) | No | Timestamp set when the response object is assembled |

### `summary` object

| Field | Type | Description |
|---|---|---|
| `total_lines` | integer | Count of all order lines |
| `ready_to_ship` | integer | Count of lines with status `ready_to_ship` |
| `partial` | integer | Count of lines with status `partial` |
| `needs_pack` | integer | Count of lines with status `needs_pack` |
| `needs_make` | integer | Count of lines with status `needs_make` |
| `blocked` | integer | Count of lines with status `blocked` |
| `not_applicable` | integer | Count of lines with status `not_applicable` |
| `total_cases_ordered` | integer | Sum of `cases_ordered` across all lines |
| `total_cases_shipped` | integer | Sum of `cases_shipped` across all lines |
| `total_cases_remaining` | integer | Sum of `cases_remaining` across all lines |

### Line object

| Field | Type | Nullable | Description |
|---|---|---|---|
| `line_id` | integer | No | Internal `sales_order_lines.id` |
| `product_id` | integer | No | The finished-good product ID |
| `product_name` | string | No | Product display name |
| `product_label_type` | enum | No | One of: `house`, `private_label` |
| `private_label_brand_owner` | object | Yes | `{id, name}` of the brand owner; null if `product_label_type != 'private_label'` |
| `cases_ordered` | integer | No | Original line quantity |
| `cases_shipped` | integer | No | Cases already shipped against this line (per shipment records) |
| `cases_remaining` | integer | No | `cases_ordered - cases_shipped - (fg_packed_for_order_lb / pack_size_lb)` for non-partial lines; the actual cases-still-to-pack figure for partial lines |
| `pack_size_lb` | numeric | No | Pack size in pounds (e.g., 25.0 for a 25 LB case) |
| `lb_remaining` | numeric | No | Computed convenience: `cases_remaining * pack_size_lb` |
| `status` | enum | No | One of: `ready_to_ship`, `partial`, `needs_pack`, `needs_make`, `blocked`, `not_applicable` |
| `fg_packed_for_order_lb` | numeric | No | Total FG inventory packed against this specific order (via normalized `transactions.order_reference` match) |
| `fg_inventory_on_hand_lb` | numeric | No | Total FG inventory of this product across all orders. Informational. |
| `fg_lots_for_order` | array | No | FG lots packed against this order (may be empty); see below |
| `batch_source` | object | Yes | `{batch_product_id, batch_product_name}`; null for `ready_to_ship`, `not_applicable`, and dual-role-NULL lines |
| `batch_lots_available` | array | No | Available lots of the mapped batch source, sorted oldest-first; empty when `batch_source` is null |
| `lot_allocation` | array | No | The allocation that will go on the packing slip if confirmed; empty for `ready_to_ship` and `not_applicable` |
| `lot_allocation_source` | enum | Yes | One of: `fifo_default`, `user_specified`; null when `lot_allocation` is empty |
| `warnings` | array | No | Line-level warnings (may be empty); see Warning Catalog |

### `fg_lots_for_order` entry

| Field | Type | Description |
|---|---|---|
| `lot_id` | integer | Internal lot ID |
| `lot_code` | string | Lot code (date-style or structured) |
| `lb` | numeric | Pounds in this lot attributable to this order |
| `match_confidence` | enum | One of: `exact`, `normalized`, `missing_reference` |
| `warnings` | array | Lot-level warnings |

### `batch_lots_available` entry

| Field | Type | Description |
|---|---|---|
| `lot_id` | integer | Internal lot ID |
| `lot_code` | string | Lot code |
| `lb` | numeric | Pounds available in this lot |
| `is_fifo_pick` | boolean | True if FIFO would consume this lot (in whole or part) for the current demand |
| `warnings` | array | Lot-level warnings (e.g., `LOT_AGE_EXCESSIVE`, `BATCH_NAME_INDICATOR_CONFLICT`, `SUPPLIER_LOT_MISSING`) |

### `lot_allocation` entry

| Field | Type | Description |
|---|---|---|
| `lot_id` | integer | Internal lot ID being drawn from |
| `lot_code` | string | Lot code |
| `lb` | numeric | Pounds drawn from this lot for the pack |
| `warnings` | array | Allocation-level warnings (e.g., `DUPLICATE_LOT_IN_ALLOCATION`) |

---

## 4. Status Enums

### Line status

| Value | Definition |
|---|---|
| `ready_to_ship` | FG lots packed against this order cover `cases_remaining` in full. The work is done; only shipping remains. |
| `partial` | Some FG lots packed against this order exist but don't cover the demand. `lot_allocation` is populated for the remaining cases. |
| `needs_pack` | No FG packed against this order yet; mapped batch inventory is sufficient to do the pack. `lot_allocation` is fully populated. |
| `needs_make` | No FG packed against this order yet; mapped batch inventory is insufficient. Batch production is required before packing. |
| `blocked` | Has a `block`-severity warning preventing action (e.g., archived product, brand mismatch, missing batch mapping). |
| `not_applicable` | Service items, billing-only lines, or other lines that aren't packable from inventory. |

### Order status

| Value | Definition |
|---|---|
| `open` | No shipments yet. |
| `partially_shipped` | At least one shipment exists; not all lines are fully shipped. |
| `fully_shipped` | Every line's `cases_shipped` equals `cases_ordered`. |
| `closed` | Order has been explicitly closed (manual, force-close, or post-shipment lifecycle event). |
| `cancelled` | Order was cancelled before fulfillment. Triggers the short-circuit response — see §5. |

---

## 5. Cancelled Orders: Short-Circuit Response

When `order_status = 'cancelled'`, the response returns HTTP 200 with **only** these fields populated:

```json
{
  "order_id": 1234,
  "order_number": "SO-260417-001",
  "customer_id": "uuid",
  "customer_name": "Platinum Group",
  "order_status": "cancelled",
  "generated_at": "2026-04-30T15:23:00Z"
}
```

All other fields are omitted (or `null` if the consumer requires them present). Consumers must check `order_status` before assuming `lines`, `summary`, or other fields are present.

Rationale: cancelled is a valid state of the order, not an error. The endpoint shouldn't return 4xx for a question it can validly answer ("the order is cancelled, there's nothing to pack").

---

## 6. Invariants

These rules hold for every response. Implementations must enforce them; consumers can rely on them.

1. **Cases is the source of truth.** `cases_ordered`, `cases_shipped`, `cases_remaining`, and the summary `total_cases_*` fields are authoritative for outstanding quantity. `lb_remaining`, `fg_packed_for_order_lb`, `fg_inventory_on_hand_lb`, and per-allocation `lb` values are computed-convenience fields. Implementations must not derive demand from `lb` fields.

2. **Cases arithmetic.** For non-partial lines: `cases_remaining = cases_ordered - cases_shipped - (fg_packed_for_order_lb / pack_size_lb)`. For partial lines, `cases_remaining` reflects what's still to pack (the residual after `fg_packed_for_order_lb` has covered some of the line's demand).

3. **Partial-line dual population.** A line with `status = 'partial'` has both `fg_lots_for_order` (non-empty) and `lot_allocation` (non-empty). Together they cover `cases_ordered - cases_shipped`. See worked example #2.

4. **Lot-allocation arithmetic.** Sum of `lb` across all entries in a line's `lot_allocation` equals `lb_remaining` (= `cases_remaining * pack_size_lb`).

5. **Status implies field population.**
   - `ready_to_ship` → `lot_allocation` empty, `batch_source` null, `fg_lots_for_order` non-empty
   - `partial` → `lot_allocation` non-empty, `batch_source` non-null, `fg_lots_for_order` non-empty
   - `needs_pack` → `lot_allocation` non-empty, `batch_source` non-null, `fg_lots_for_order` empty
   - `needs_make` → `lot_allocation` empty, `batch_source` non-null, `fg_lots_for_order` empty
   - `blocked` → at least one warning of severity `block` in the line's warnings array
   - `not_applicable` → `lot_allocation` empty, `batch_source` null

6. **Boolean flag derivation.**
   - `is_blocked = ANY line has a block-severity warning`
   - `requires_acknowledgment = (NOT is_blocked) AND (ANY line has an acknowledge-severity warning)`
   - `is_actionable = ANY line has status in {partial, needs_pack, needs_make}`

7. **Always 200 unless the resource is missing or auth fails.** Cancelled orders, blocked orders, and orders requiring acknowledgment all return 200. The response payload tells the consumer what state the order is in.

---

## 7. Worked Examples

### Example 1: Clean `ready_to_ship` line

A line for Granola Classic 25 LB (24 cases ordered) where the operator already packed 24 cases against this order earlier in the week.

```json
{
  "line_id": 5678,
  "product_id": 136,
  "product_name": "Granola Classic 25 LB",
  "product_label_type": "house",
  "private_label_brand_owner": null,
  "cases_ordered": 24,
  "cases_shipped": 0,
  "cases_remaining": 0,
  "pack_size_lb": 25.0,
  "lb_remaining": 0.0,
  "status": "ready_to_ship",
  "fg_packed_for_order_lb": 600.0,
  "fg_inventory_on_hand_lb": 1200.0,
  "fg_lots_for_order": [
    {
      "lot_id": 9001,
      "lot_code": "MAR 24 2026",
      "lb": 600.0,
      "match_confidence": "exact",
      "warnings": []
    }
  ],
  "batch_source": null,
  "batch_lots_available": [],
  "lot_allocation": [],
  "lot_allocation_source": null,
  "warnings": []
}
```

### Example 2: Partial line with the canonical 24/10/14 split

A line for Granola Cinnamon Almond 25 LB (24 cases ordered, 10 cases already packed against this order, 14 still to pack from batch).

```json
{
  "line_id": 5679,
  "product_id": 140,
  "product_name": "Granola Cinnamon Almond 25 LB",
  "product_label_type": "private_label",
  "private_label_brand_owner": {"id": "setton-uuid", "name": "Setton Farms"},
  "cases_ordered": 24,
  "cases_shipped": 0,
  "cases_remaining": 14,
  "pack_size_lb": 25.0,
  "lb_remaining": 350.0,
  "status": "partial",
  "fg_packed_for_order_lb": 250.0,
  "fg_inventory_on_hand_lb": 250.0,
  "fg_lots_for_order": [
    {
      "lot_id": 9050,
      "lot_code": "APR 27 2026",
      "lb": 250.0,
      "match_confidence": "exact",
      "warnings": []
    }
  ],
  "batch_source": {
    "batch_product_id": 109,
    "batch_product_name": "Batch Setton Cinnamon Almond Granola #14"
  },
  "batch_lots_available": [
    {
      "lot_id": 8501,
      "lot_code": "APR 27 2026",
      "lb": 410.0,
      "is_fifo_pick": true,
      "warnings": []
    }
  ],
  "lot_allocation": [
    {
      "lot_id": 8501,
      "lot_code": "APR 27 2026",
      "lb": 350.0,
      "warnings": []
    }
  ],
  "lot_allocation_source": "fifo_default",
  "warnings": []
}
```

Note: 10 cases (250 lb) already packed + 14 cases (350 lb) to pack = 24 cases (600 lb) = `cases_ordered * pack_size_lb`. The arithmetic invariants hold.

### Example 3: Cancelled order short-circuit

```json
{
  "order_id": 1234,
  "order_number": "SO-260417-001",
  "customer_id": "platinum-group-uuid",
  "customer_name": "Platinum Group",
  "order_status": "cancelled",
  "generated_at": "2026-04-30T15:23:00Z"
}
```

No `lines`, no `summary`, no boolean flags. Consumers must dispatch on `order_status` first.

---

## 8. Design Rationale

The non-obvious choices in this contract, with reasoning. Captured at design time so future maintainers (or future-you) understand why the shape is what it is.

- **`fg_packed_for_order_lb` is separate from `fg_inventory_on_hand_lb`.** These describe two different operational realities: "what's allocated to this customer" vs "what's available across the warehouse." Collapsing them would force the consumer to reconstruct the distinction, and would invite bugs where a line shows `ready_to_ship` based on inventory that isn't actually allocated. Surfacing both lets the GPT generate honest sentences like "24 cases packed for this order; 7 more on hand for other orders."

- **Pack transactions carry `order_reference`; no separate stage event.** A staging-dock event would be a more first-class model, but it would require operator behavior change (a new step in the workflow). The existing `transactions.order_reference` field is already populated correctly today. v1 uses what's there; v2 can introduce explicit staging if operational complexity warrants.

- **FG-to-order matching uses string normalization with a confidence flag.** A separate pack-to-order linking table would be more rigorous, but `transactions.order_reference` is the existing source of truth and is being populated. Normalization (`SO-260417-001` ≡ `so260417001`) handles the easy 90%; the `match_confidence` field admits where the matching was fuzzy, surfaced via `LOT_MATCH_FUZZY` info-level warnings. If the data quality declines, the `MISSING_ORDER_REFERENCE` and fuzzy-match warning rates will rise visibly in logs.

- **Warnings nest by scope rather than flat-with-scope-tags.** A flat `warnings: [{scope: 'lot', ...}]` array would be slightly simpler to iterate but harder to render. Nesting puts each warning next to the data it concerns, which is what the GPT and dashboard need to render natural per-line, per-lot, or per-allocation messages without re-joining.

- **`cases_remaining` is authoritative; `lb_remaining` is convenience.** Floating-point arithmetic on lb values is a known source of off-by-fractional-pound bugs. Cases are integers, deterministic, and what the customer ordered. Implementations must derive lb from cases, not the other way around.

- **Cancelled orders return 200 with a sparse response.** A 4xx would conflate "the resource is in a state I can't help with" (semantic) with "your request was malformed" (syntactic). Cancelled is a valid query result, just one with no actionable lines. Consumers dispatch on `order_status`.

- **`generated_at` is reserved without staleness enforcement.** Adding the field now means v1.1 can introduce freshness validation at commit endpoints without a contract change. Reserving fields is cheap; renaming or adding fields post-launch is expensive.

- **The endpoint is stateless and idempotent.** Multiple calls with the same input return the same output (modulo `generated_at` and any concurrent inventory changes). No preview-id, no session, no optimistic locking — those are commit-endpoint concerns. Re-validation of operator-modified allocations is just another call to the same endpoint.

- **`lot_allocation` is named for what it is, not what generated it.** The earlier draft called this `fifo_plan`, which framed FIFO as a recommendation the operator argues with. Renaming to `lot_allocation` with `lot_allocation_source: fifo_default | user_specified` matches the operator's mental model: this is the actuals, with FIFO defaults pre-filled to save typing.

- **`private_label_brand_owner` is a structured field, not a string in a warning message.** When `PRIVATE_LABEL_BRAND_MISMATCH` fires, both the warning and the line carry the structured brand owner data. This lets dashboards and audit reports query and filter without parsing message strings.

---

## 9. Versioning

This document describes **v1.0** of the response contract.

Future changes are versioned via a path prefix on the endpoint (e.g., `/v2/orders/{order_id}/pack-preview`) or via a `Content-Version` header. Breaking changes require a version bump. Additive changes (new fields with sensible defaults) can land within v1 if and only if all existing invariants continue to hold.

The `generated_at` field is reserved for v1.1 staleness validation work; see `factory-ledger-followups.txt`.

---

## 10. Related Documents

- `WARNING_CATALOG_v1.md` — the 15-entry warning catalog with codes, severities, and triggers
- `factory-ledger-followups.txt` (outside repo) — deferred work items including allergen schema, historical pack reconciliation, staleness enforcement, and pallet staging
- `FACTORY_LEDGER_CHANGELOG.md` — regression-guard log; the row for this design lock-in is the entry point
