# Warning Catalog v1

**Status:** Locked
**Version:** 1.0
**Last updated:** 2026-04-30
**Companion document:** `PACK_PREVIEW_RESPONSE_v1.md`

---

## 1. Overview

The warning catalog defines every condition the `pack-preview` (and later `reconcile-preview`) endpoints can flag in their responses. Each warning has:

- **Code** — a stable, uppercase identifier (e.g., `NO_BATCH_MAPPING`). Codes do not change between versions; new versions add or deprecate entries.
- **Scope** — `line`, `lot`, or `allocation`. Determines where the warning attaches in the response.
- **Severity** — `info`, `acknowledge`, or `block`. Determines how consumers should handle it.
- **Trigger predicate** — a mechanical condition expressible as a SQL or Python check. No "warns when something seems off."
- **Example message** — the human-readable string the consumer can render.
- **Notes** — edge cases, known limitations, or related work.

### Severity scheme

| Severity | Consumer behavior |
|---|---|
| `info` | Render alongside the relevant data; no special handling required. The operator can ignore it. |
| `acknowledge` | The operator must explicitly confirm before commit. Commit endpoints reject silent passthrough. |
| `block` | Commit will be rejected. The condition must be resolved (data fix, mapping change, etc.) before proceeding. |

### Where warnings live

Warnings nest by scope inside the response:

- **Line-scope** → `lines[i].warnings`
- **Lot-scope** → `lines[i].fg_lots_for_order[j].warnings` or `lines[i].batch_lots_available[j].warnings`
- **Allocation-scope** → `lines[i].lot_allocation[j].warnings`

A line is `blocked` (line status enum) when any warning on the line itself or on any of its lots or allocations has severity `block`. The boolean flags `is_blocked` and `requires_acknowledgment` at the response top level aggregate across all lines.

---

## 2. Catalog

### Line-scope warnings

#### `NO_BATCH_MAPPING`

| Field | Value |
|---|---|
| **Scope** | line |
| **Severity** | block |
| **Trigger** | `product.parent_batch_product_id IS NULL` AND `product.id NOT IN (the documented dual-role SKU list, see CLAUDE.md "Dual-role finished goods")` |
| **Example message** | "No batch source mapped for this product. Cannot determine what to pack from." |
| **Notes** | Dual-role SKUs intentionally have NULL `parent_batch_product_id` and must not trigger this warning. The dual-role list is maintained in CLAUDE.md and should be cached or table-ized for efficient lookup. |

#### `PRODUCT_INACTIVE`

| Field | Value |
|---|---|
| **Scope** | line |
| **Severity** | dynamic (`block` or `acknowledge` — see below) |
| **Trigger** | `product.active = false` |
| **Severity rule** | If `fg_packed_for_order_lb / pack_size_lb >= cases_remaining` → `acknowledge` (existing FG covers the line; we can ship inventory of an archived product). Otherwise → `block` (cannot fulfill demand for an archived product). |
| **Example messages** | Block: "Product is archived and there is insufficient existing FG inventory to fulfill this line." Acknowledge: "Product is archived but existing FG inventory covers this line. Ship existing inventory only." |
| **Notes** | The dynamic severity is intentional. Some archived products legitimately have remaining inventory that should be shipped to clear it out. Splitting into two codes was considered and rejected (more catalog surface for similar meaning). |

#### `INSUFFICIENT_BATCH_INVENTORY`

| Field | Value |
|---|---|
| **Scope** | line |
| **Severity** | acknowledge |
| **Trigger** | `SUM(batch_lots_available[*].lb) > 0` AND `SUM(batch_lots_available[*].lb) < lb_remaining` |
| **Example message** | "Batch inventory is short. Available: 410 lb. Needed: 600 lb. A partial pack of 410 lb (16 cases) is possible; remaining 190 lb (8 cases) requires production." |
| **Notes** | Maps to line status `partial` only when `fg_packed_for_order_lb` is also positive; otherwise the line is `needs_make` because no work has been done yet against this order. |

#### `NO_BATCH_INVENTORY`

| Field | Value |
|---|---|
| **Scope** | line |
| **Severity** | acknowledge |
| **Trigger** | `SUM(batch_lots_available[*].lb) = 0` AND `batch_source IS NOT NULL` |
| **Example message** | "No batch inventory on hand. A batch must be produced before this line can be packed." |
| **Notes** | This is a normal operational state ("we need to make a batch"), not an error. Severity is `acknowledge` so the GPT can surface it as actionable next-step guidance. The line status is `needs_make`. |

#### `PARTIAL_FG_AVAILABLE`

| Field | Value |
|---|---|
| **Scope** | line |
| **Severity** | info |
| **Trigger** | `0 < fg_packed_for_order_lb < lb_remaining_at_order_open` (i.e., some pack has happened against this order but more is needed) |
| **Example message** | "10 cases (250 lb) already packed against this order; 14 more cases (350 lb) to be packed from batch." |
| **Notes** | Always paired with line status `partial`. Purely informational; the operator doesn't need to do anything special. |

#### `PRIVATE_LABEL_BRAND_MISMATCH`

| Field | Value |
|---|---|
| **Scope** | line |
| **Severity** | block |
| **Trigger** | `product.label_type = 'private_label'` AND `order.customer_id != product.brand_owner_id` (or whatever the brand-owner attribution column is named) |
| **Example message** | "This is a Setton-branded private-label product but the order is for Platinum Group. Refusing to pack." |
| **Notes** | This rule already exists in the codebase (the 19 private-label SKU guard). Surfacing it pre-flight via this catalog means the operator sees it during planning, not at commit. Both the line and the warning carry the structured `private_label_brand_owner` field for clear rendering. |

#### `SERVICE_ITEM_AS_PRODUCT`

| Field | Value |
|---|---|
| **Scope** | line |
| **Severity** | block |
| **Trigger** | `product.is_service = true` AND the request body's `lot_allocations` includes an entry for this line (or a downstream commit is attempted) |
| **Example message** | "This line is a service item (e.g., pallet charge) and cannot be packed from inventory." |
| **Notes** | A service-item line normally has status `not_applicable` and no warning fires. This warning only triggers when a consumer tries to actively pack a service item — a usage error. The line being a service item by itself is not a warning condition. |

---

### Lot-scope warnings

#### `LOT_AGE_EXCESSIVE`

| Field | Value |
|---|---|
| **Scope** | lot (batch lots only; FG lots not currently checked) |
| **Severity** | info |
| **Trigger** | `now() - lot.created_at > 90 days` |
| **Example message** | "Lot APR 14 2026 is 95 days old." |
| **Notes** | Threshold is configurable (suggested: 90 days). The 90-day default reflects typical granola shelf-life concerns; coconut, sprinkles, and other categories may warrant different thresholds in v2. For v1, single global threshold. |

#### `LOT_INSUFFICIENT`

| Field | Value |
|---|---|
| **Scope** | lot |
| **Severity** | info |
| **Trigger** | `lot.lb < remaining_demand_at_this_position_in_FIFO_traversal` (i.e., the FIFO planner will need to combine this lot with the next one) |
| **Example message** | "Lot APR 14 2026 has 300 lb; remaining demand of 600 lb will be combined with the next FIFO lot." |
| **Notes** | This is informational because FIFO multi-lot consumption is normal. The warning exists to make the multi-lot allocation visible in cases where the operator might otherwise miss it. |

#### `BATCH_NAME_INDICATOR_CONFLICT`

| Field | Value |
|---|---|
| **Scope** | lot (batch lots) |
| **Severity** | acknowledge |
| **Trigger** | `regex_match (?i)(no\s+\w+) on batch_product.name` captures any group whose value (case-insensitive) appears as a substring of `finished_good_product.name` |
| **Example message** | "Batch name contains 'no almonds'; finished good name contains 'Almond'. Verify this is intentional before packing." |
| **Notes — IMPORTANT** | This is a **heuristic name check**, not a real allergen check. The regex is intentionally over-broad: it will fire on batch names like "no longer in production" if the captured word ("longer") happens to appear in an FG name. False-positive rate is acceptable because severity is `acknowledge`, not `block`. The proper allergen schema (typed allergen attributes on products and batches) is deferred to v2 — see `factory-ledger-followups.txt`. **Code comment requirement**: the implementation must include an inline comment explaining the heuristic nature and pointing to the followup. |

#### `LOT_PRODUCT_MISMATCH`

| Field | Value |
|---|---|
| **Scope** | lot |
| **Severity** | block |
| **Trigger** | A `lot_id` referenced in a request body's `lot_allocations` belongs to a `product_id` that doesn't match the line's expected product (the line's mapped batch product, or the line's FG product for `fg_lots_for_order`-type allocations) |
| **Example message** | "Lot 8501 belongs to Batch SS Original Granola #1, but this line expects Batch Setton Cinnamon Almond Granola #14." |
| **Notes** | Defends against the GPT or operator typing the wrong lot ID. Should never fire on FIFO defaults. |

#### `SUPPLIER_LOT_MISSING`

| Field | Value |
|---|---|
| **Scope** | lot (batch lots being consumed; this implies upstream traceability) |
| **Severity** | info |
| **Trigger** | `lot.supplier_lot_code IS NULL` for a batch lot included in `lot_allocation` |
| **Example message** | "Lot APR 27 2026 has no supplier lot code attached. Traceability for this pack will be incomplete." |
| **Notes** | Ties into the LAT Code Policy v1.1 work and the `PATCH /lots/{lot_code}/supplier-lot` endpoint. Not blocking because operationally the pack should still be allowed; the warning surfaces the trace gap. |

#### `LOT_MATCH_FUZZY`

| Field | Value |
|---|---|
| **Scope** | lot (FG lots in `fg_lots_for_order`) |
| **Severity** | info |
| **Trigger** | An FG lot was attributed to this order via normalized matching of `transactions.order_reference` (case, whitespace, or punctuation differences) rather than exact match |
| **Example message** | "Lot 9050 attributed via normalized order-reference match." |
| **Notes** | The normalization is silent and deterministic; the warning makes the fuzziness auditable. If `match_confidence = 'normalized'`, this warning fires. If `match_confidence = 'exact'`, no warning. |

#### `MISSING_ORDER_REFERENCE`

| Field | Value |
|---|---|
| **Scope** | lot (FG lots in `fg_lots_for_order` — surfaces only in edge cases) |
| **Severity** | info |
| **Trigger** | A pack transaction relevant to an FG lot has `order_reference IS NULL` but the lot is included in the response anyway (e.g., for a related-order heuristic in v2) |
| **Example message** | "Lot 9050 has no order reference on its pack transaction." |
| **Notes** | In v1, `fg_lots_for_order` only includes lots with non-null, matching order references. This warning is reserved for v1.1 if a "loose match" mode is added. Defining it in v1 keeps the catalog code-stable across minor versions. |

---

### Allocation-scope warnings

#### `DUPLICATE_LOT_IN_ALLOCATION`

| Field | Value |
|---|---|
| **Scope** | allocation |
| **Severity** | block |
| **Trigger** | A line's `lot_allocation` array contains the same `lot_id` more than once |
| **Example message** | "Lot 8501 appears twice in the allocation for line 5679. Combine into a single entry." |
| **Notes** | Defends against operator typo or GPT bug when manually overriding allocations. Only relevant when the request body includes a `lot_allocations` payload. |

---

## 3. Severity Behavior Summary

How consumers (GPT, dashboard, scripts) should treat each severity:

### `info`

- **Render:** Yes, alongside the relevant data.
- **Block commit?** No.
- **Operator action required?** No.
- **Examples:** `LOT_AGE_EXCESSIVE`, `LOT_INSUFFICIENT`, `SUPPLIER_LOT_MISSING`, `LOT_MATCH_FUZZY`, `PARTIAL_FG_AVAILABLE`.

### `acknowledge`

- **Render:** Yes, prominently.
- **Block commit?** Yes, unless the operator explicitly confirms.
- **Operator action required?** Confirm understanding (e.g., "yes, I know batch is short, partial pack is fine").
- **Examples:** `INSUFFICIENT_BATCH_INVENTORY`, `NO_BATCH_INVENTORY`, `BATCH_NAME_INDICATOR_CONFLICT`, dynamic-severity `PRODUCT_INACTIVE` (acknowledge variant).

### `block`

- **Render:** Yes, with clear explanation of why action is impossible.
- **Block commit?** Yes, absolutely. Commit endpoints must reject.
- **Operator action required?** Resolve the underlying condition (fix mapping, choose a different lot, get authorization, etc.) — there's no "force commit" path in v1.
- **Examples:** `NO_BATCH_MAPPING`, `PRIVATE_LABEL_BRAND_MISMATCH`, `SERVICE_ITEM_AS_PRODUCT`, `LOT_PRODUCT_MISMATCH`, `DUPLICATE_LOT_IN_ALLOCATION`, dynamic-severity `PRODUCT_INACTIVE` (block variant).

---

## 4. Top-Level Boolean Aggregation

The response's top-level boolean flags are derived from line-level warnings:

```
is_blocked = ANY(line.warnings has severity 'block'
                 OR ANY lot in line.fg_lots_for_order has severity 'block'
                 OR ANY lot in line.batch_lots_available has severity 'block'
                 OR ANY entry in line.lot_allocation has severity 'block')

requires_acknowledgment = (NOT is_blocked) AND
                          ANY(any warning at any scope on any line has severity 'acknowledge')
```

This means: if anything is `block`, `requires_acknowledgment` is false (the order can't proceed regardless). Acknowledgment only matters when nothing is blocking.

---

## 5. Catalog Counts

- **Total warnings in v1:** 15
- **By scope:** Line: 7. Lot: 7. Allocation: 1.
- **By severity (modulo dynamic):** Info: 6. Acknowledge: 5. Block: 5. (Dynamic `PRODUCT_INACTIVE` counted once, in whichever bucket it lands per request.)

---

## 6. Future Work

Deferred to later versions; tracked in `/Users/cns/factory-ledger-followups.txt`:

- **Proper allergen schema** — typed allergen attributes on products and batches, replacing the `BATCH_NAME_INDICATOR_CONFLICT` heuristic with a real `ALLERGEN_MISMATCH` rule.
- **Historical `order_reference` reconciliation** — pre-reliable-capture pack transactions are not queryable by order; either back-fill or accept the gap.
- **Pack-preview staleness** — `generated_at` is reserved; v1.1 should validate freshness at commit.
- **Pallet staging as first-class event** — model the staging dock state explicitly when operational complexity warrants.
- **Per-category lot-age thresholds** — currently a single global 90-day threshold; coconut and sprinkles may need different limits.
- **`MISSING_ORDER_REFERENCE` reaching production** — in v1 this warning is defined but not emitted; v1.1 may activate it for a loose-match mode.

---

## 7. Versioning

This document describes **v1.0** of the catalog.

- **Stable surface area:** warning codes, scopes, and trigger predicates.
- **Mutable surface area:** example messages, notes, severity rules (subject to in-version evolution as long as documented in changelog).
- **Breaking changes:** removing or renaming codes, changing scopes, raising severity from non-block to block. Require a version bump.
- **Additive changes:** new codes can land within v1 if they don't change existing behavior.

---

## 8. Related Documents

- `PACK_PREVIEW_RESPONSE_v1.md` — the response contract that consumes this catalog
- `/Users/cns/factory-ledger-followups.txt` — deferred work
- `FACTORY_LEDGER_CHANGELOG.md` — regression-guard log
- `CLAUDE.md` — dual-role finished goods list (relevant to `NO_BATCH_MAPPING`)
