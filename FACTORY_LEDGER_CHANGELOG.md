# Factory Ledger Changelog — Regression Guard

Every fix is logged here so future sessions know what breaks if a change is reverted.

## Change History

| # | Date | Area | What Changed | Problem It Solved | Breaks If Reverted | Migration/File |
|---|------|------|-------------|-------------------|-------------------|----------------|
| 1 | 2026-03-19 | main.py | Exclude service/charge lines from SO weight totals | Service lines (pallets, freight) inflated weight totals on sales orders | SO weight summaries become inaccurate | `main.py` |
| 2 | 2026-03-19 | main.py | Auto-deduct add-in ingredients during pack | PB Chips, Banana Bites weren't consumed from inventory at hopper | Inventory ghost stock for add-in ingredients | `main.py` |
| 3 | 2026-03-19 | dashboard | Add 8oz BS panel and fix case weight rounding | 8oz products missing from dashboard; case weights showed too many decimals | 8oz products disappear from dashboard | `main.py`, `dashboard/` |
| 4 | 2026-03-19 | main.py | Fix day summary for pack consumption from prior-day batch lots | Pack runs using batches made on previous days didn't show in day summary | Day summary under-reports consumption | `main.py` |
| 5 | 2026-03-19 | dashboard | Fix dashboard API calls to use absolute Railway URL | Relative paths failed when dashboard hosted on Netlify | Dashboard API calls return 404 | `dashboard/` |
| 6 | 2026-03-19 | main.py | Replace `is_ingredient` with `type != 'ingredient'` in /make commit pack-prompt query | /make commit crashed with "column is_ingredient does not exist" | /make commit will crash again on auto-prompt /pack query | `main.py` |
| 7 | 2026-03-23 | main.py, dashboard, docs | "Ready to Ship" display label + ready→in_production reverse transition | "Ready" label was unclear; no way to move order back if production falls short | Dashboard shows "Ready" instead of "Ready to Ship"; can't reverse from ready status | `main.py`, `dashboard.js`, `index.html`, `gpt-instructions-v3.md`, `GUIDE.md`, `CONTEXT.md` |
| 8 | 2026-03-23 | main.py, dashboard | Backward trace handles ingredient lots (supplier trace + downstream batches) | Backward trace returned "Batch not found" for ingredient lots | Ingredient lots will 404 on backward trace again | `main.py`, `dashboard/traceability.html` |
| 9 | 2026-03-23 | main.py, dashboard | Lot code collision disambiguation (product_id param on 5 endpoints) + trace type validation on /trace/ingredient | Wrong lot returned when lot codes collide across products; /trace/ingredient returned false traces for finished goods lots | Lot collisions return wrong data; /trace/ingredient silently returns empty traces for batch lots; PATCH supplier-lot corrupts wrong lot | `main.py`, `dashboard/traceability.html` |
| 10 | 2026-03-23 | main.py, dashboard | Extend collision disambiguation to main dashboard — add product_id to shipment/receipt/search API responses, pass it through all lot links, handle 409 with disambiguation picker | Main dashboard showed raw 409 error when clicking colliding lot codes in shipping, receiving, search, or product panel | Dashboard lot links break for any colliding lot code; users see raw JSON error instead of picker | `main.py`, `dashboard/dashboard.js` |
| 11 | 2026-03-24 | GPT instructions | Order entry behavior fix — add ORDER ENTRY FROM CONFIRMATIONS section, strengthen BE CONCISE and ACT DON'T LOOP, tighten disambiguation for order entry, compress verbose sections | GPT acted as consultant during order entry: dumped SOPs, over-confirmed, took 6-7 exchanges instead of 1-2 | GPT reverts to verbose consultant mode during order entry; offers unprompted next steps; shows API payloads | `gpt-instructions-v3.md` |
| 12 | 2026-03-24 | main.py | Direct-ship trace + supplier-lot exposure: added direct_shipments to /trace/ingredient and _trace_ingredient_backward, new /trace/supplier-lot endpoint, normalized entry_source enums | Ingredient/resale lots shipped directly to customers invisible to trace (FDA recall gap) | Trace endpoints lose direct-ship visibility; /trace/supplier-lot endpoint disappears; entry_source checks may miss lots with new enum values | `main.py` |
| 13 | 2026-03-24 | OpenAPI + GPT instructions | Added format:date + description to requested_ship_date in both schemas; strengthened ORDER EDITING section with "CALL API IMMEDIATELY", date conversion rules, "NEVER show curl/docs" | GPT showed curl commands instead of calling updateOrderHeader for ship date changes | GPT reverts to showing curl commands for ship date edits; date format guidance lost | `openapi-v3.yaml`, `openapi-schema-gpt.yaml`, `gpt-instructions-v3.md` |
| 13 | 2026-03-24 | main.py | Option C trace: /trace/ingredient soft-handles output lots (returns upstream_ingredients, lot_origin, origin_note instead of 400 error); /trace/batch adds customer_shipments, total_shipped_lb, on_hand_lb | Output lots (e.g. 601141 Graham Cracker Crumbs NTF) blocked from /trace/ingredient despite shipping directly to customers | /trace/ingredient reverts to hard-rejecting output lots (shipment data invisible); /trace/batch loses customer shipment visibility | `main.py` |
| 14 | 2026-03-24 | main.py, migrations | Rename lot 324 from UNKNOWN to 25216 + new PATCH /lots/{lot_id}/rename endpoint | Chocolate Sprinkles lot stuck with lot_code='UNKNOWN'; no API to rename lots | Lot 324 reverts to UNKNOWN (trace by code fails); rename endpoint disappears, future UNKNOWN lots require raw SQL | `main.py`, `migrations/029_rename_unknown_lot_to_25216.sql` |
| 15 | 2026-03-24 | main.py, migrations | GAP-3: Standalone /ship writes shipments + shipment_lines; backfill migration 030 | Standalone shipments invisible to shipment tables (packing slips, integrity checker, reports) | Standalone ships stop creating shipment records; existing backfilled records remain but new ones won't be created; integrity checker flags all standalone ships | `main.py`, `migrations/030_backfill_standalone_shipment_records.sql` |
| 16 | 2026-03-25 | main.py, OpenAPI, GPT instructions | resolve_order_id dependency: all /sales/orders/{order_id} endpoints accept order_number strings (e.g. SO-260323-001) in addition to integer DB ids | GPT only knows order_number, not DB id — FastAPI rejected strings with validation error, GPT fell back to curl commands | All order endpoints revert to integer-only; GPT can't call any order update/ship/status endpoints by order_number | `main.py`, `openapi-v3.yaml`, `openapi-schema-gpt.yaml`, `gpt-instructions-v3.md` |
| 17 | 2026-03-25 | main.py, dashboard, OpenAPI, GPT | Unit Display: all qty endpoints return unit_count/batch_count derived from case_size_lb/default_batch_lb; dashboard shows dual "X lb · Y units" format; packing slip uses dual format; new /products/missing-case-size endpoint | Operators saw lb-only everywhere, had to mentally convert to units/cases | All unit counts disappear from API responses and dashboard; packing slip reverts to "N cs" format; /products/missing-case-size endpoint removed | `main.py`, `dashboard/dashboard.js`, `dashboard/traceability.html`, `openapi-v3.yaml`, `openapi-schema-gpt.yaml`, `GPT_INSTRUCTIONS.md`, `gpt-instructions-v3.md` |

## Known Root Causes

- **Weight calculation** — service/charge lines on SOs must always be filtered out (`line_type` or product category check)
- **Inventory accuracy** — add-in ingredients must be deducted at pack time, not just primary ingredients
- **Cross-origin hosting** — dashboard on Netlify, API on Railway; all API calls must use absolute URLs
- **Timezone** — all timestamps must use `America/New_York`; mixing UTC causes off-by-one on day boundaries

## Permanent Rules (GPT Instructions)

When editing GPT instructions (`GPT_INSTRUCTIONS.md` or `gpt-instructions-v3.md`), all of these rules must survive:

1. Always use ET timezone for all dates/times
2. Lot numbers follow format: `YY-MM-DD-XXXX-NNN`
3. Pack transactions auto-deduct add-in ingredients
4. Service/charge lines excluded from weight totals
5. `case_size_lb` must be set before packing (hard fail if missing)
6. Ship endpoint validates SO line quantities before committing
7. Receive transactions require supplier lot when available
8. Batch/make transactions consume ingredients proportionally
9. Dashboard API uses absolute URLs (Railway base)
10. All migrations are idempotent (IF NOT EXISTS / ON CONFLICT DO NOTHING)
