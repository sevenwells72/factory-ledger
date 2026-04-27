# Production Planning Knowledge Capture — Discrepancies

Discrepancies surfaced during the production-planning interview. Log-only for now; resolve separately.

_Interview started 2026-04-23._

---

## 1. Stale `has_bom` flag on `products`
- **What:** All 201 products have `has_bom = false`, despite 23 batches having real recipes in `batch_formulas` and 37 finished goods having BOMs in `product_bom`.
- **Impact:** Code that filters by `has_bom` will silently return zero rows.
- **Resolution TBD:** either backfill the flag from the two BOM tables, or drop it and update callers to use `EXISTS (...)`.

## 2. Coconut `default_batch_lb` vs. recipe sum mismatch — RESOLVED
- **What:** `default_batch_lb` and summed `batch_formulas.quantity_lb` disagree for all four coconut batches.
- **Resolution (2026-04-23):** the two numbers measure different things. `default_batch_lb` is the **post-processing yield** (what ends up in the bin after drying/toasting); the recipe sum (324.5 lb dry) is the ingredient draw. Owner confirmed yields: sweetened (Fancy/Flake/Medium) = **360 lb**, toasted (Flake) = **300 lb** (more water driven off during toasting). Both recipes for sweetened include 65 lb of Water marked `exclude_from_inventory=true` (389.5 lb wet → 360 lb dry → ~29.5 lb evaporation). Planner should schedule against `default_batch_lb`; the recipe sum is only for inventory consumption math.
- **Follow-up cleanup items spawned:** see #8 (90007 missing water line) and #9 (Glycol likely = Glycerin).

## 8. `90007` (Toasted Coconut Flake) BOM is missing the water line
- **What:** The three sweetened coconut batches (90003/90004/90005) each include a 65 lb Water line (`exclude_from_inventory=true`). `90007 Batch Coconut Toasted Sweetened Flake` has no water line at all — recipe jumps straight to dry ingredients, sum = 324.5 lb, yields 300 lb. The toasting step can't physically happen without a water/syrup phase; the line is almost certainly missing.
- **Action:** add a Water row (probably also 65 lb, `exclude_from_inventory=true`) to 90007's `batch_formulas` so the process is documented consistently across all 4 coconut SKUs. Low-urgency since water isn't tracked as inventory and the yield number is correct.

## 10. `shelf_life_days` is NULL on every finished good
- **What:** The `products.shelf_life_days` column is unpopulated for all 38 finished goods in scope.
- **Owner response 2026-04-23:** "not defined."
- **Impact:** Blocks demand planning from honoring shelf-life constraints (e.g., "don't produce more than N weeks of demand ahead"). Also blocks any date-code automation on cases.
- **Action:** capture per-product-family shelf life (granola vs coconut vs graham) with the owner and backfill.

## 11. No formal QA / hold gates defined on finished goods
- **What:** The prompt asks about metal detection, weight check, QA hold before cases ship. Owner response 2026-04-23: "not defined."
- **Impact:** Planner currently cannot reason about "cases packed today are not shippable until X hours of QA hold." Treat `pack_complete` as equivalent to `ready_to_ship` for planning.
- **Action:** confirm whether this is actually correct (no gates exist) or just not captured yet; document if/when any are added.

## 13. 7 FGs appear in open SOs but have NO `product_bom`
- **What:** Running `demand_planning_v1.sql` against the current open-SO snapshot surfaces 7 `products.type='finished'` SKUs that are `active=true` and referenced on at least one open sales-order line, but have zero `product_bom` rows — so they fall out of any FG → batch → capacity math:

  | odoo | name | notes |
  |---|---|---|
  | 10047 | Desiccated Flake 50 LB | Raw-coconut 50 LB repack / passthrough (sold direct to customers) |
  | 10302 | Sprinkles Rainbow 10 LB | Sourced ingredient sold as FG; owner previously said sprinkles are not manufactured |
  | 10303 | Sprinkles Chocolate 10 LB | Same — sourced |
  | 10305 | Sprinkles Rainbow 25 LB | Same — sourced |
  | 10306 | Sprinkles Chocolate 25 LB | Same — sourced |
  | 31011 | Graham Cracker Crumbs – 50 LB | Sourced-ingredient passthrough (paired with 31012's 10 LB repack which IS in scope) |
  | (null)| Pallet Charge | Service line (expected; not a physical product — already `is_service=true` in `products`) |
- **Impact:** These show in the planner detail as "REPACK (no batch)" only because the on-hand query lets them pass through. They are not part of the 37-FG production scope. Excluding them is correct for manufacturing planning but means they currently don't generate "buy/receive" signals either.
- **Action:** decide per-SKU whether to treat as (a) repack FG with `product_bom` = 1 row (sourced ingredient → case), same pattern as Graham Crumbs 31012; or (b) pure passthrough (sourced bulk sold as-is without repack). Either way, add the metadata so the planner can generate buy-signals for the upstream ingredient.

## 12. Changeover minutes between 25 LB granola SKUs not captured
- **What:** Owner response 2026-04-23: "not specified (includes label swap + box prep)." Policy recorded as "minimize where possible."
- **Impact:** Planner can't precisely cost SKU transitions. Safe default: treat same-case-format changeovers as cheap (maybe 5–15 min for label/box swap), but flag for future time-study.

## 9. "Glycol" ingredient is likely a typo for "Glycerin"
- **What:** All 4 coconut BOMs list 7.00 lb of **Glycol**. Glycol (propylene glycol or polyethylene glycol) is rarely used in food production; **Glycerin / Glycerine** is a standard humectant in sweetened coconut for moisture retention. Almost certainly a data-entry typo.
- **Action:** rename the `products` row and the `batch_formulas.ingredient_product_id` references. Low-urgency planning impact but matters for ingredient spec / supplier / regulatory docs.

## 3. Fruit Nut 25 LB finished good (`70061`) has only 1 BOM line
- **What:** Every other finished-good BOM has 3 lines (base batch + 2 packaging). Fruit Nut 25 LB has 1. Likely missing packaging components.
- **Action:** confirm during that SKU's interview; file a follow-up to add the missing rows.

## 4. Graham Cracker Crumbs 10 LB (`31012`) has no base batch
- **What:** Finished good with no matching `Batch Graham*` parent in `batch_formulas`.
- **Owner context (pre-interview):** legitimate repack from a sourced ingredient with no intermediate batch step.
- **Action:** during its FG interview, capture with `base_batch_sku: null` and a note explaining the repack workflow (yield, pack rate, labor).

## 5. `product_category` is NULL on every batch and finished good
- **What:** The `product_category` column on `products` is empty for all 60 products in scope.
- **Impact:** Non-blocking for this interview (I grouped by name). But downstream planning tools that key off `product_category` will break.
- **Action:** future cleanup — backfill from the family/brand grouping we use here.

## 6. Batches with zero downstream finished-good references — RESOLVED
- **What:** `90012` (Batch SS Chocolate Chip #5) and `90017` (Batch SS Original #4) have recipes but no `product_bom` row points to them.
- **Resolution (2026-04-23):** owner confirmed both are discontinued. Marked `status: discontinued` in YAML and skipped in interview. Follow-up: deactivate their `products.active` flag and/or archive their `batch_formulas` rows so they stop showing up in "batches we make" queries.

## 7. BS 6×8 OZ case line (70085/70086/70087/70088) is active in `products` but discontinued — RESOLVED AS DEACTIVATION NEEDED
- **What:** 4 finished-good products — `70085 BS Hazelnut Butter 6x8 OZ`, `70086 BS Almond Butter 6x8 OZ`, `70087 BS Dark Chocolate 6x8 OZ`, `70088 BS PB Banana 6x8 OZ` — are marked `active=true` in `products` but have no `product_bom` rows. Initial question raised at scoping thought 70088 was an active SKU missing a BOM.
- **Resolution (2026-04-23):** owner confirmed **the entire BS 6×8 OZ line is discontinued** — "there is only 7oz. 8oz is discontinued." Revised earlier decision to add 70088 to scope; it's out along with 70085/86/87. FG total remains 37, not 38.
- **Follow-up (low urgency):** set `products.active = false` for all four SKU IDs so future scoping queries don't re-surface them. Also remove any dashboard/SKU-list references to the 8 OZ line.

## 14. Pouch line load not computed in demand plan
- **What:** Pouch line is shown as info-only in Tier 1 of the demand plan (`render_demand_plan.py`). The YAML stores pouch throughput as bags/hr (750–1000), but `demand_planning_v1.sql` does not emit a per-pouch-FG hours estimate.
- **Impact:** The planner cannot flag pouch-line overload; only `granola_line` and `coconut_line` get real utilization percentages. A pouch-heavy week could quietly max out the pouch line without any Tier 1 warning.
- **Action:** extend `demand_planning_v1.sql` to emit estimated pouch-hours per pouch-FG SO line (cases × bags_per_case ÷ bags_per_hour), then aggregate in Tier 1 alongside granola and coconut. Address after the first few weeks of running if pouch bottlenecks actually surface in practice.
- **Identified:** 2026-04-23. Status: open.

## 15. Local dev machine is stuck on system Python 3.9.6
- **What:** `/Library/Developer/CommandLineTools/usr/bin/python3` (the only `python3` on PATH for user `cns` on this Mac) is version 3.9.6, with pip 21.2.4. No Homebrew, no pyenv. The `render_demand_plan.py` first-run succeeded only because the file uses `from __future__ import annotations` (stringifies all type hints) — PEP 604 unions (`X | Y`) in actual runtime positions would break, as would any library that eagerly resolves annotations (pydantic, dataclasses with modern typing, etc.). `main.py` already uses `dict | None` at runtime and cannot be imported locally for this reason.
- **Impact:** Trip hazard for any future local Python script — authors have to remember to stringify annotations. Also blocks local pytest runs against `main.py` (have to SSH into the Railway container instead). `pip install --break-system-packages` isn't supported on this pip version either; installs must use `--user`.
- **Action:** install Homebrew Python 3.11+ (or pyenv + a recent 3.11/3.12), repoint `/usr/local/bin/python3` or shell aliases, and reinstall this project's dev deps (`python-dotenv`, `jinja2`, plus `psycopg2-binary` + `pyyaml` if moving off system-site). Update `RUNBOOK_ACTUAL.md` install command once the interpreter changes.
- **Identified:** 2026-04-23 during demand-plan first run. Status: open, low urgency. Workaround in place: the renderer uses `from __future__ import annotations`; packages installed to user-site via `python3 -m pip install --user ...`.
