Review complete. Full test output first, then findings.

## 10. Test suite — VERIFIED (186 passed)

First attempt is worth flagging: the repo's default `python3` (3.14 via Homebrew) has no pytest, and the failed run **still exited 0** — so a CI/agent check that only looks at the exit code would report green on zero tests. I ran the real suite via `/private/tmp/factory-ledger-test-venv4` (Python 3.12.13).

```
$ TEST_DATABASE_URL=postgresql://localhost:5432/factory_ledger_test \
    /private/tmp/factory-ledger-test-venv4/bin/python -m pytest -v

============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.1.1, pluggy-1.6.0
rootdir: /Users/michaelgross/Documents/factory-ledger
configfile: pytest.ini
testpaths: tests
plugins: anyio-3.7.1
collected 186 items

tests/test_batch1_correctness_security.py .....                          [  2%]
tests/test_daily_entries.py ...                                          [  4%]
tests/test_dashboard_api_key.py ..........                               [  9%]
tests/test_dashboard_b2.py ...                                           [ 11%]
tests/test_dashboard_production_calendar.py .                            [ 11%]
tests/test_expected_receipts.py ..................                       [ 21%]
tests/test_notes_auth.py ........                                        [ 25%]
tests/test_orders_matrix_export.py ...                                   [ 27%]
tests/test_phase1_ledger_integrity.py .......                            [ 31%]
tests/test_production_today_tile.py .......                              [ 34%]
tests/test_production_warning.py ......                                  [ 38%]
tests/test_readonly_tripwire.py .........                                [ 43%]
tests/test_recent_ledger.py ....                                         [ 45%]
tests/test_resolve_customer.py ....                                      [ 47%]
tests/test_sales_order_allocations.py ..............................     [ 63%]
tests/test_sales_order_line_fields.py ....                               [ 65%]
tests/test_sales_order_readiness.py .................                    [ 74%]
tests/test_ship_order_service_line.py ......                             [ 77%]
tests/test_supplies.py ............                                      [ 84%]
tests/test_void_semantics.py ....                                        [ 86%]
tests/test_write_response_contract.py .........................          [100%]

=============================== warnings summary ===============================
main.py:1387
  main.py:1387: SyntaxWarning: invalid escape sequence '\d'
    (substring(name FROM '(\d+)\s*x\s*\d+'))::numeric
main.py:1252
  main.py:1252: DeprecationWarning: on_event is deprecated, use lifespan event handlers instead.
    @app.on_event("startup")
fastapi/applications.py:4547 (x2)
  DeprecationWarning: on_event is deprecated, use lifespan event handlers instead.
    return self.router.on_event(event_type)
main.py:1589
  DeprecationWarning: on_event is deprecated ... @app.on_event("shutdown")
main.py:2203  PydanticDeprecatedSince20: V1 style @validator deprecated  @validator("category")
main.py:2209  PydanticDeprecatedSince20: @validator("priority")
main.py:2224  PydanticDeprecatedSince20: @validator("priority")
main.py:2230  PydanticDeprecatedSince20: @validator("status")
main.py:2281  PydanticDeprecatedSince20: @validator('unit', pre=True, always=True)
main.py:2285  PydanticDeprecatedSince20: @validator('quantity_lb', always=True, pre=True)
main.py:2267  PydanticDeprecatedSince20: class-based config deprecated  class OrderLineInput(BaseModel)
pydantic/_internal/_config.py:386
tests/test_batch1_correctness_security.py::test_admin_sql_route_is_removed
  UserWarning: Valid config keys have changed in V2: * 'underscore_attrs_are_private' has been removed
main.py:2353  PydanticDeprecatedSince20: class CommitShipOrderRequest(BaseModel)
tests/... (114 warnings across 15 modules)
  httpx/_client.py:690: DeprecationWarning: The 'app' shortcut is now deprecated.
tests/test_expected_receipts.py: 10 warnings
  main.py:4269: PydanticDeprecatedSince20: The `dict` method is deprecated  data = req.dict(exclude_unset=True)
tests/test_notes_auth.py, tests/test_write_response_contract.py: 3 warnings
  main.py:11913: PydanticDeprecatedSince20: The `dict` method is deprecated

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
====================== 186 passed, 153 warnings in 3.58s =======================
```

Builder's claim of 186 passing is accurate. (`-v` shows dots because `pytest.ini` sets `addopts = -ra -q`.) No new warnings introduced by this commit.

---

## 1. Void-coalesce/restore state machine — PASS on the unique-index invariant, FAIL on restore attribution

**Uniqueness holds.** I traced every INSERT/UPDATE that can produce `status='active'`:

- `_upsert_live_allocation` (`main.py:1086-1135`) — SELECTs the live row `lot_id IS NOT DISTINCT FROM %s AND status='active' … FOR UPDATE`, then UPDATEs or INSERTs, under `_lock_allocation_product` taken by the caller (`main.py:9263`).
- `_consume_allocation_row` partial split (`main.py:566-579`) — supersedes the original **before** inserting the leftover, so only one live row exists at insert time.
- `_shrink_active_allocations` (`main.py:745-752`) — inserts `status='released'`, never live.
- `_void_ship_allocations` (`main.py:781-813`) — flips shipped→active **only** when the `status='active'` leftover probe returns nothing; otherwise coalesces.
- `_coalesce_lot_allocations` (`main.py:993-1018`) — rewrites `lot_id` only when no live `(line, survivor)` pin exists.

This matches PR 1's hand-simulated `test_full_void_then_restore_cycle_holds_under_unique_index` (`tests/test_sales_order_allocations.py:488-600`) exactly, and `test_helper_full_void_then_restore_cycle` (`:711`) asserts the identical row shapes against the real helpers. **No path creates a second live row for the same unique key.**

**But the restore path mis-attributes rows.** `_consume_allocation_row` makes the split leftover inherit `last_ship_transaction_id` from the original row (`main.py:578`), and `_restore_ship_allocations` finds full-consume voids by `status='active' AND last_ship_transaction_id = txn` (`main.py:901-908`). Those two collide.

I confirmed this against the local test DB in a rolled-back transaction (probe script in scratchpad; no repo files touched, no DB state persisted):

```
seeded: line=1212 product=2453 lot=1786 allocation=1284 on_hand=100

--- after FULL ship 100 on txn A=2134 ---
  id=1284 status=shipped     qty= 100.0 ship_txn=2134 last_ship=2134
--- after VOID txn A ---
  id=1284 status=active      qty= 100.0 ship_txn=None last_ship=2134
--- after PARTIAL ship 40 on txn B=2135 ---
  id=1284 status=superseded  qty= 100.0 reason=split_on_ship   last_ship=2134
  id=1285 status=active      qty=  60.0 reason=None            last_ship=2134   <-- inherited A
  id=1286 status=shipped     qty=  40.0 ship_txn=2135          last_ship=2135

>>> _restore_ship_allocations(txn A=2134)
    returned: [{'allocation_id': 1285, 'quantity_lb': 60.0, 'coalesced': False}]

--- after RESTORE txn A ---
  id=1284 status=superseded  qty= 100.0 reason=split_on_ship  ship_txn=None
  id=1285 status=shipped     qty=  60.0 ship_txn=2134                          <-- WRONG
  id=1286 status=shipped     qty=  40.0 ship_txn=2135

live reserved total = 0.0
```

Restoring txn A marks the **60 lb live reservation** as shipped against A. Row 1284 — the row A actually consumed — stays `superseded/split_on_ship` forever. The ledger says A shipped 100 and B shipped 40; SOA says 60 + 40. The line silently loses its live reservation with no blocker and no error.

Reachable through supported API calls: ship an order, `POST /void/{A}`, re-ship a smaller quantity, then `POST /records/transactions/{A}/corrections` with `event_type=restore`. No test covers void → partial re-ship → restore.

## 2. Over-allocation guard — PASS

`_allocation_totals` (`main.py:1073-1082`) sums `quantity_lb` over `product_id = %s AND status='active'` with the TTL filter and **no order or line filter** — so `product_allocated` is all live allocations product-wide, across every SO. `_validate_allocation_addition` (`main.py:1154-1215`) then takes `coverable = min(line_coverable, product_coverable[, lot_coverable])`:

- `product_coverable = _product_on_hand - product_allocated` (`:1158-1159`)
- `line_coverable = remaining_effective - line_allocated` (`:1156-1157`) — covers "line allocate > remaining_effective 409s even if factory on-hand is huge"
- `lot_coverable = lot_on_hand - SUM(live pins on that lot, all lines)` (`:1184-1194`)

Sibling same-SKU lines on one order compete correctly, since `product_allocated` is order-agnostic — asserted by `test_http_manual_upsert_release_and_sibling_overallocation` (`:832-883`): line A allocates 80 of 100 on-hand, line B requests 30, gets `409 OVER_ALLOCATION` with `coverable_lb == 20`. Violation returns 409 with the design's envelope (`:1197-1210`).

Minor naming variance: the 409 field is `allocated_others_lb` but carries `product_allocated`, which **includes this line's own** live allocation. The arithmetic is right (own pounds are subtracted once on the line side, once on the product side, and `min()` picks the binding constraint); only the field name overstates what it holds.

## 3. Consume-on-ship — PASS

Unconditional: `_consume_sales_order_allocations` is called at `main.py:10190` in the `ship_order` commit body with no flag guard, and `ALLOCATIONS_ENFORCED` does not exist anywhere in `main.py` yet (correctly deferred to PR 5). Ship now routes through `_sales_order_ship_plan` (`main.py:10065` preview, `:10168` commit), which uses `available_lots_for_product` → `fifo_lot_balances` → `FIFO_LOT_ORDER_SQL` including `l.id ASC` (`main.py:324`) — the intended FIFO unification.

Split fields (`_consume_allocation_row`, `main.py:538-589`):

| Case | Row states | Fields |
|---|---|---|
| `T = Q` | original → `shipped` | `ship_transaction_id=txn`, `last_ship_transaction_id=txn`, `release_reason=NULL` (`:557-563`) |
| `T < Q` | original → `superseded/split_on_ship`; leftover `active`; slice `shipped` | both copies carry `split_from_id=original` (`:577`, `:585`); shipped slice gets `ship_transaction_id=txn` + `last_ship_transaction_id=txn` (`:586-587`) |

Own lot pins are walked before FIFO takeable, in `created_at ASC, id ASC` order (`_active_allocation_rows`, `main.py:427-438`; plan loop `:632-652`), then SKU-level rows cover the remainder (`:665-671`). Verified by `test_helper_full_void_then_restore_cycle` and `test_sku_level_allocation_is_consumed_and_split_on_ship`.

The one wrong field is the leftover's `last_ship_transaction_id` (`:578`) — see finding 1.

## 4. Release paths — PASS, with one gap

- **merge_lots**: `_coalesce_lot_allocations` (`main.py:979-1027`) called at `main.py:12420` *before* ledger lines move. Same-line live pin on survivor → `survivor.quantity_lb +=`, source → `superseded/lot_merged`; otherwise `UPDATE lot_id`. Shipped/superseded rows get `lot_id` rewritten (`:1020-1026`). SKU-level rows untouched (query filters `lot_id = source`). Covered by `test_merge_coalesces_two_live_pins_onto_surviving_lot`.
- **Order cancel**: `main.py:9622-9640` — locks each affected product, then `_release_active_allocations(order_id=…, reason='order_cancelled')`.
- **Line cancel**: `main.py:9881-9889` — `reason='line_cancelled'`.
- **Line edit**: `main.py:9945-10001` — 422 `QTY_BELOW_SHIPPED_EFFECTIVE` when `quantity_lb < shipped_effective`, then shrinks newest-first (`ORDER BY created_at DESC, id DESC`, `:9987`) with `reason='line_quantity_reduced'`.
- **Inventory void**: `_shrink_overallocated_products` (`main.py:921-978`) repairs per-lot deficits first, then product-level, ordering `requested_ship_date DESC NULLS LAST` as the design requires.

`_release_active_allocations`' parameter shuffle (`main.py:711-717`, `[params[0], params[-1], *params[1:-1]]`) is hard to read but correct — I traced the clause/param ordering.

**Gap:** restore of a *ship* txn re-posts a stock reduction but never calls `_shrink_overallocated_products` (`main.py:5901-5905` handles only reverse-coalesce). Void of a non-ship txn does shrink (`:5897`). So restoring a previously-voided ship can leave `SUM(active) > on_hand`. The design's event table doesn't require the shrink there either, so this is design-consistent — but it is a real hole worth naming before PR 5 claims enforcement is sound.

## 5. Auto-FIFO — PASS

- Ordered by `fifo_lot_balances()` → `COALESCE(l.received_at, l.created_at) ASC, l.id ASC` (`main.py:324`), consumed via `available_lots_for_product` at `main.py:9339`. `test_http_auto_fifo_splits_lots_and_sets_48_hour_ttl` asserts lot_a (60) then lot_b (90) — an `id ASC` tie-break, since both lots share a creation timestamp.
- `expires_at` on auto_fifo only: manual mode rejects a supplied `expires_at` with 422 `MANUAL_ALLOCATION_CANNOT_EXPIRE` (`:9274-9279`) and passes `expires_at=None` (`:9302`); `_upsert_live_allocation` sets `merged_expiry = expires_at if merged_source == "auto_fifo" else None` (`main.py:1114`), so promoting an auto row to manual/staged clears the TTL. Default 48h at `:9313`, asserted to the minute by the TTL test.
- Expired rows are never consumed: `_active_allocation_rows` filters `expires_at IS NULL OR expires_at > clock_timestamp()` (`main.py:433`), and every write path calls `_expire_auto_fifo_allocations` (`main.py:382-395`) first. `test_next_product_write_persists_expired_auto_fifo_release` asserts `plan["allocation_takes"] == []` and the row flipping to `released/expired`.
- GET does not UPDATE: despite its name, `_order_readiness_after_write` is pure read, and the readiness SQL filters expiry at `main.py:8424`. Design invariant holds.

## 6. PATCH /lots/{id}/received-at — PASS

`main.py:9504-9566`. Missing key / JSON `null` / `""` → 422 `RECEIVED_AT_REQUIRED` (`:9510-9516`); non-string or unparseable → 422 `INVALID_RECEIVED_AT`; naive (`tzinfo is None or utcoffset() is None`) → 422 `INVALID_RECEIVED_AT` (`:9534-9541`); future → 422 `RECEIVED_AT_IN_FUTURE` (`:9542-9548`); unknown lot → 404 `LOT_NOT_FOUND` (`:9557-9563`, rolled back since the UPDATE's `RETURNING` came back empty). Response carries the design's five fields plus `entry_source`, with `lot_is_incomplete` recomputed by the date-only rule. All four branches asserted in `test_received_at_patch_clears_missing_date_and_validates` (`:1065-1105`).

## 7. Endpoint exposure — PASS

```
$ grep -c 'operationId:' openapi-gpt-v3.yaml                       → 30
$ grep -c 'operationId:' gpt-configs/schemas/openapi-floor.yaml    → 22
```
No duplicate operationIds in either file, and `git show --stat ef709e5 -- openapi-gpt-v3.yaml gpt-configs/schemas/openapi-floor.yaml` is empty — neither schema was touched. All four new routes are on `DASHBOARD_KEY_ALLOWLIST` (`main.py:1787-1793`): `GET`/`POST /sales/orders/{order_id}/allocations`, `DELETE /sales/orders/{order_id}/allocations/{allocation_id}`, `PATCH /lots/{lot_id}/received-at`. `test_gpt_schema_operation_counts_unchanged` guards both counts.

## 8. Forbidden patterns — PASS

No triggers on the table (`pg_trigger` on `sales_order_allocations`, excluding internal: empty); migration 044 contains no `CREATE TRIGGER`/`CREATE FUNCTION`; the commit touches no migration. `set_config` appears nowhere in `main.py`; the sole `default_transaction_read_only` reference (`main.py:1624`) is the read-only tripwire *reading* `current_setting`. No `SET LOCAL`, no `SET SESSION`, no `set_session(readonly=True)`. `test_readonly_tripwire.py` passes (9 tests). Every f-string in the new SQL interpolates code-local literals only (`released_at_sql` `main.py:507`; joined clause list `:711`; `excluded` `:899`) — no user-controlled interpolation.

## 9. Design must-pass → test mapping

| Design must-pass case | Covering test |
|---|---|
| SKU-level allocate, `allocated`/`available` update | `test_http_manual_upsert_release_and_sibling_overallocation` |
| Duplicate allocate upserts, no second row | `test_http_manual_upsert_release_and_sibling_overallocation` |
| Release returns row to `released` | `test_http_manual_upsert_release_and_sibling_overallocation` |
| Auto-FIFO splits two lots, `source=auto_fifo`, `expires_at` set | `test_http_auto_fifo_splits_lots_and_sets_48_hour_ttl` |
| Expired auto_fifo not consumed; persisted on next write; GET no-write | `test_next_product_write_persists_expired_auto_fifo_release` |
| Allocate 100, ship 40, void → one live 100; restore → 60+40 | `test_helper_full_void_then_restore_cycle`, `test_http_allocate_ship_40_void_restore_cycle`, `test_full_void_then_restore_cycle_holds_under_unique_index` |
| Pins 40 on A + 60 on B; merge A→B → one live 100 on B | `test_merge_coalesces_two_live_pins_onto_surviving_lot` |
| Void of unrelated receive shrinks reservations | `test_void_inventory_releases_uncovered_lot_pin_even_if_product_total_is_covered` |
| SKU-level row consumed and split on ship | `test_sku_level_allocation_is_consumed_and_split_on_ship` |
| Ship uses own lot pins before FIFO takeable | `test_available_lots_subtracts_lot_pins_then_shadows_foreign_sku_fifo`, `test_helper_full_void_then_restore_cycle` |
| Consume-on-ship runs with flag off | Structural — no flag exists |
| Sibling same-SKU competition → `coverable_lb=20` | `test_http_manual_upsert_release_and_sibling_overallocation` |
| `PATCH received-at` 200 / 422 null / 422 future / 404 | `test_received_at_patch_clears_missing_date_and_validates` |
| Line-edit shrink, order cancel release | `test_line_edit_shrinks_then_order_cancel_releases_allocations` |
| Line qty < shipped_effective → 422 | `test_line_edit_rejects_quantity_below_effective_shipped` |
| Line cancel releases | `test_line_cancel_releases_allocation` |
| Dashboard-key scoping on new routes | `test_dashboard_scoped_key_can_use_allocation_and_received_at_routes` |
| Office 30 / Floor 22 | `test_gpt_schema_operation_counts_unchanged` |

**Must-pass cases with no covering test:**

1. **Service line allocate → 422** — `SERVICE_LINE_NOT_ALLOCATABLE` (`main.py:1052-1059`) is asserted nowhere.
2. **Cancelled / fulfilled / invoiced order allocate → 409** — `ORDER_NOT_ALLOCATABLE` (`main.py:1061-1070`) asserted nowhere.
3. **SKU-level 100 + lot-level 60 against the same 150 on-hand → 409 on the 60** — the "sum both kinds" case is untested.
4. **Line allocate > `remaining_effective` → 409 even when factory on-hand is huge** — the `line_coverable` limb is never the binding constraint in any test.
5. **Two concurrent transactions: second waits on `FOR UPDATE`, then 409s** — no concurrency test exists.
6. **`RESTORE_SPLIT_MISSING` 409** — the guard (`main.py:858-866`) is never exercised; the PR-1 test only asserts it *wouldn't* fire.
7. **Lot-level allocate; a second *order* cannot take that stock** — cross-order write-path competition is untested (only same-order siblings, plus PR 2's read-path test).
8. **Void → partial re-ship → restore** — untested, and this is exactly where finding 1 bites.

Also unasserted: `LOT_PRODUCT_MISMATCH`, `MANUAL_ALLOCATION_CANNOT_EXPIRE`, `ALLOCATION_NOT_ACTIVE`.

---

## Additional concerns (not in the numbered checks)

- **Release verb diverges from the design.** The doc specifies `POST /sales/orders/{id}/allocations/{allocation_id}/release`; the build ships `DELETE …/{allocation_id}` (`main.py:9430`). That forced a relaxation of the security guard in `tests/test_dashboard_api_key.py:97-100`, which previously asserted the dashboard key holds **no** DELETE outside `/dashboard/api/notes`. Functionally fine — the handler soft-releases, it doesn't delete rows — but the design's POST form would have left the guard untouched.
- **Auto-FIFO has no `/auto` sub-route**; it's a `mode` discriminator on the create route. Functionally equivalent, dashboard-only, no GPT-schema impact. Cosmetic.
- **`_copy_allocation_row` doesn't carry `created_at`** (`main.py:519-535`), so split leftovers get a fresh timestamp. That changes their position in the `created_at ASC` consume order and the `created_at DESC` shrink order relative to their parent. Not specified by the design; benign today, worth a deliberate decision.
- **`FACTORY_LEDGER_CHANGELOG.md` row 84 says "no push"** — stale as of this session's push of the branch to `origin/feat/044-so-allocations`. Row still correctly says not deployed / not on main.

---

## Verdict: **fix round required**

The PR is otherwise strong — the locking protocol, the over-allocation math, the unique-index discipline, consume-on-ship, the release paths, PATCH `received-at`, and the endpoint/exposure guards all hold up under direct code reading, and the 186/186 claim is real. Checks 2, 3, 5, 6, 7, and 8 pass cleanly.

What blocks acceptance is finding 1: a reproducible state-machine defect in `_restore_ship_allocations` that silently converts a live 60 lb reservation into a shipped row against the wrong transaction and strands the genuinely-consumed row in `superseded`. It fails silently — no error, no blocker, no unique violation — which is the worst failure mode for a reservation ledger. Root cause is the single line `main.py:578` feeding the selector at `main.py:901-908`; the fix is small (stop inheriting `last_ship_transaction_id` onto split leftovers, or tighten the restore selector to rows the void actually reactivated), but it needs a regression test for void → partial re-ship → restore.

Second, the coverage gaps in check 9 are broader than a nitpick: eight design must-pass cases have no test, including two error paths that are the only things standing between a user and allocating against a service line or a cancelled order. Those are cheap to add and belong in this PR, since PR 4 and PR 5 will build enforcement on top of them.

I modified no files. The one thing I created is a throwaway probe script under the session scratchpad (`scratchpad/probe_restore.py`); it ran inside a transaction that ended in `ROLLBACK`, so `factory_ledger_test` is unchanged.
