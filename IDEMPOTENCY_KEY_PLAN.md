# Idempotency Key Plan — Ship / Make / Pack / Adjust / Ship_Order

> **Status:** Design artifact — not yet implemented.
> **Context:** Layer 3 of the readonly-transaction incident fix stack.
> Layers 1 (global tripwire, `a1da52e`) and 2 (connection discard + idempotent retry, `a8752ca`) are committed on `fix/readonly-tripwire-global-handler`.

---

## Why idempotency keys?

When a client (the GPT, the dashboard, or a script) sends a commit request and gets a 503 (readonly) or a network timeout, it cannot know whether the write landed. Without idempotency protection, retrying the same request risks double-shipping, double-making, double-packing, or double-adjusting. The tripwire (layer 1) tells the client to retry; the idempotency key (layer 3) makes that retry safe.

---

## Schema change (shared across all five endpoints)

```sql
-- Migration: NNN_create_idempotency_keys.sql

CREATE TABLE idempotency_keys (
    key           TEXT        PRIMARY KEY,
    endpoint      TEXT        NOT NULL,     -- e.g. 'ship', 'make', 'pack', 'adjust', 'ship_order'
    transaction_id INTEGER   REFERENCES transactions(id),
    request_hash  TEXT        NOT NULL,     -- SHA-256 of canonical request body (detects key reuse with different payload)
    response_json JSONB      NOT NULL,      -- stored response from the first successful execution
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at    TIMESTAMPTZ NOT NULL DEFAULT NOW() + INTERVAL '24 hours'
);

CREATE INDEX idx_idempotency_keys_expires ON idempotency_keys (expires_at);

-- Periodic cleanup (can be a cron or startup task):
-- DELETE FROM idempotency_keys WHERE expires_at < NOW();
```

### Client contract

- Client sends header `Idempotency-Key: <value>` on commit requests.
- If omitted, no idempotency protection — behaves as today (backward compatible).
- If provided and the key already exists with the same `request_hash`, return the stored `response_json` with HTTP 200 (replay).
- If provided and the key exists with a *different* `request_hash`, return HTTP 422 `IDEMPOTENCY_KEY_REUSED` — the client is misusing the key.
- If provided and the key does not exist, execute normally, store the key + response on commit, return the response.

### Key format recommendation for the GPT

The GPT should generate keys as: `{endpoint}-{natural_id}-{timestamp_minute}` — e.g. `ship_order-SO-260514-001-202605281300`. This makes keys human-readable in logs and naturally scoped to one logical operation attempt.

---

## Per-endpoint design

### 1. `POST /ship/commit` (standalone ship)

**What makes it unique:**
`product_name` + `customer_name` + `order_reference` + `quantity_lb`. No natural DB uniqueness constraint exists — the same product/customer/qty/ref can legitimately ship twice on different days (reorder). A purely natural key is therefore insufficient.

**Idempotency key:** Client-supplied `Idempotency-Key` header.

**Replay detection:**
1. Before opening a transaction, look up `idempotency_keys WHERE key = %s AND endpoint = 'ship'`.
2. If found and `request_hash` matches → return stored `response_json`, skip all writes.
3. If found and hash differs → 422.
4. If not found → execute, then INSERT into `idempotency_keys` with the new `transaction_id` and serialized response, all within the same DB transaction.

**What gets written (must all be skipped on replay):**
- `transactions` (1 row, type='ship')
- `transaction_lines` (1+ rows, negative qty for outbound)
- `shipments` (1 row)
- `shipment_lines` (1+ rows)

**Test: no-double-write**
1. Send `/ship/commit` with `Idempotency-Key: test-ship-1` → assert 200, note `transaction_id`.
2. Send identical request with same key → assert 200, same `transaction_id`, no new rows in `transactions` or `transaction_lines`.
3. Send different payload with same key → assert 422 `IDEMPOTENCY_KEY_REUSED`.
4. Send same payload with different key → assert 200, new `transaction_id` (legitimate new shipment).

---

### 2. `POST /make/commit`

**What makes it unique:**
`product_name` + `batches` + `lot_code` (if provided). A make with the same product/batches/lot_code on the same day is ambiguous — could be a genuine second run or a retry.

**Idempotency key:** Client-supplied `Idempotency-Key` header.

**Replay detection:** Same pattern as ship.

**What gets written (must all be skipped on replay):**
- `transactions` (1 row, type='make')
- `transaction_lines` (N rows: 1 positive output + N negative ingredient deductions)
- `lots` (1 row created or found-by-code)

**Additional concern:** `lot_code` auto-generation. If the first attempt creates a lot with an auto-generated code, the replay must return the *same* lot code. Storing the full response in `response_json` handles this — the replayed response includes the lot code from the first execution.

**Test: no-double-write**
1. `/make/commit` with key → 200, note `transaction_id` and `lot_code`.
2. Same request, same key → 200, same `transaction_id`, same `lot_code`, no new `transaction_lines`.
3. Different `batches` count, same key → 422.

---

### 3. `POST /pack/commit`

**What makes it unique:**
`source_product` + `target_product` + `cases` + `lot_allocations` (if explicit). Like make, a repeated pack of the same product/cases is ambiguous without a client key.

**Idempotency key:** Client-supplied `Idempotency-Key` header.

**Replay detection:** Same pattern as ship.

**What gets written (must all be skipped on replay):**
- `transactions` (1 row, type='pack' — technically stored as a second 'make' type in some flows; verify)
- `transaction_lines` (N rows: negative from batch lot(s), positive into FG lot)
- `lots` (1 row for target FG lot, created or found-by-code)

**Additional concern:** FIFO lot selection. If `lot_allocations` is omitted, the system picks lots by FIFO. A replay must not re-run FIFO (inventory may have changed between attempts). The stored `response_json` captures the exact allocations from the first run.

**Test: no-double-write**
1. `/pack/commit` with key → 200, note allocations and `transaction_id`.
2. Same request, same key → 200, same response, no new rows.
3. Different `cases`, same key → 422.

---

### 4. `POST /adjust/commit`

**What makes it unique:**
`product_name` + `lot_code` + `adjustment_lb` + `reason`. Adjustments are inherently non-idempotent in the business sense — adjusting +50 lb twice means +100 lb — so replay protection is critical.

**Idempotency key:** Client-supplied `Idempotency-Key` header.

**Replay detection:** Same pattern as ship.

**What gets written (must all be skipped on replay):**
- `transactions` (1 row, type='adjust')
- `transaction_lines` (1 row, signed quantity)

**Test: no-double-write**
1. `/adjust/commit` with key, `adjustment_lb: 50` → 200, note `transaction_id`.
2. Same request, same key → 200, same `transaction_id`. Verify lot on-hand unchanged from first call's result.
3. Same request, NO key → 200, NEW `transaction_id` (legitimate second adjustment — no idempotency protection requested).

---

### 5. `POST /sales/orders/{order_id}/ship/commit` (ship_order)

**What makes it unique:**
`order_id` is the natural scope. However, partial shipments mean the same order can be shipped multiple times legitimately. The combination of `order_id` + the specific `lines` payload (or `ship_all`) defines one shipment attempt.

**Idempotency key:** Client-supplied `Idempotency-Key` header.

**Natural guard (bonus):** After a successful ship_order, `sales_order_lines.quantity_shipped_lb` is updated. A naive replay would attempt to ship already-shipped quantities, but the existing `short` / fulfillment logic might produce a zero-shipment or a different allocation. This is NOT a reliable idempotency guard — it depends on timing and concurrent state. The explicit key is required.

**Replay detection:** Same pattern as ship.

**What gets written (must all be skipped on replay):**
- `transactions` (1 row, type='ship')
- `transaction_lines` (N rows, one per product/lot combination shipped)
- `shipments` (1 row)
- `shipment_lines` (N rows)
- `sales_order_shipments` (N rows, linking to `sales_order_lines`)
- `sales_order_lines.quantity_shipped_lb` UPDATEs (N rows)
- `sales_orders.status` UPDATE (potentially → 'shipped' or 'partial_ship')

This is the highest-stakes endpoint — the most tables touched, the most visible to customers.

**Test: no-double-write**
1. Create a test sales order with 2 lines.
2. `/sales/orders/{id}/ship/commit` with `ship_all: true`, key → 200, note `transaction_id`, shipment details.
3. Same request, same key → 200, same stored response. Verify: no new `transactions`, `shipment_lines`, or `sales_order_shipments` rows. `quantity_shipped_lb` unchanged.
4. Different lines payload, same key → 422.
5. Same payload, different key → should fail naturally (order already fully shipped → existing guard returns error).

---

## Implementation sequence (suggested)

1. **Migration:** Create `idempotency_keys` table.
2. **Shared middleware/helper:** `check_idempotency_key(cur, key, endpoint, request_hash) → stored_response | None` and `save_idempotency_key(cur, key, endpoint, transaction_id, request_hash, response)`. Both operate within the caller's transaction.
3. **Wire ship_order first** — highest risk, most tables, the endpoint that actually failed in the SO-260514-001 incident.
4. **Wire ship, make, pack, adjust** in that order.
5. **GPT instructions update** — teach the GPT to generate and send `Idempotency-Key` on commit calls after receiving a 503.
6. **Cleanup cron** — `DELETE FROM idempotency_keys WHERE expires_at < NOW()` on startup or every N hours.

## Open questions for review

1. **24-hour TTL** — is this sufficient? Too long? Keys must survive long enough for a human to retry after a Supabase failover (minutes), but short enough to not bloat the table. 24h seems conservative and safe.
2. **Should preview endpoints also be keyed?** Previews are read-only and safe to replay, so probably not — but including them would let the GPT cache preview results.
3. **Should the GPT auto-retry on 503?** The tripwire response says `"retryable": true`. If the GPT is instructed to auto-retry with the same idempotency key, the full loop is: 503 → wait 2s → retry with same key → either succeeds (fresh conn) or 503 again (still in failover window, give up). This is the ideal UX but requires GPT instruction changes.
4. **Conflict with `run_idempotent_write_with_readonly_retry`** — currently only wired to `PATCH /customers/{customer_id}`. Should routes that get idempotency keys also use the server-side retry helper, or is client-side retry with the key sufficient? Recommendation: client-side retry with key is sufficient for the five mutation endpoints; the server-side retry is for low-risk auxiliary writes (customer update, notes, etc.) where a client key would be overkill.
