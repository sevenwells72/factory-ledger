# Sweetened coconut: 360 lb per pan, forward only

Owner decision: 2026-09-15, following [S1 amendment §2d](S1-amendment-bake-vs-pack.md#2d-coconut-pan-weight--what-the-ledger-shows-read-only-2026-09-15).
Changelog #147. Code and command prepared; **production data update not executed**.

Set only `yield_multiplier` to `1.0` on SKUs (`odoo_code`) 90003, 90004 and 90005.
Their `default_batch_lb` remains 360. Toasted coconut 90007 stays 300 lb.
The existing `PUT /admin/products/{product_id}` path writes catalog metadata only.
The shared per-pan helper retains legitimate multiplier logic for other products;
the production calendar, batch tile and today tile all use it. Admin product update
is the catalog writer, not a fourth pan-count reader.

After the owner runs the command, `/make` posts `360 * batches` lb. Readers use
`posted_lb / (360 * 1.0)`: 4,320 lb = 12 pans. No historical correction, backfill,
adjustment, void or amendment is part of this fix. Old 4,795.2-lb entries remain
4,795.2 lb; current-metadata displays derive 13.32 pans (calendar rounds to 13,
batch inventory to 13.3; today tile retains 13.32).
No `production_schedule`, planner branch, void/amend code or `trace_event_lots`
data is changed. The amendment's §2d remains a dated pre-correction survey.

## Exact owner-run API command

From the repository root, with the existing master/admin API key in `API_KEY`:

```sh
python3 scripts/set_sweetened_coconut_yield.py
```

The script resolves all three SKU codes using `GET /products/search?q=SKU&limit=100`,
requires one exact 360-lb batch match per SKU, then sends this request for each
resolved database ID (SKU numbers must not be used as IDs):

```http
PUT https://fastapi-production-b73a.up.railway.app/admin/products/{resolved_product_id}
X-API-Key: <master/admin API key>
Content-Type: application/json

{"yield_multiplier": 1.0}
```

Each update is its own transaction. The script stops on failure; a rerun is
idempotent. It checks the update response and prints each completed SKU.
Deployment does not invoke the script. The effective cutoff is when each admin
update commits; existing pounds are never recomputed.
