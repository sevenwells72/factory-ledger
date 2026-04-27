-- ============================================================================
-- demand_planning_v1.sql
-- First-pass demand planning query against Factory Ledger open sales orders.
-- Derived from production_planning_knowledge.yaml captured 2026-04-23.
--
-- Scope: sales orders with status NOT IN ('shipped','cancelled') and at least
-- one line with unshipped quantity. Computes, per SO line:
--   - FG remaining lb (quantity_lb - quantity_shipped_lb)
--   - on-hand FG inventory lb (positive means stock already exists)
--   - net shortfall to make via pack-out
--   - resolved base batch + upstream batch dependency (walks the 90008 / 95005
--     cold-mix cases to their real oven-baked upstream batch)
--   - on-hand batch inventory lb (for the BASE batch and the UPSTREAM batch,
--     if different)
--   - implied batches-to-run on each production line
--   - production line tag (granola_line / coconut_line / repack — graham)
--
-- Two result sets:
--   (A) detail — one row per open SO line, ranked by requested_ship_date
--   (B) summary — total batches needed on granola_line vs coconut_line for
--       SOs with requested_ship_date within the next 14 days (the lead-time
--       promise from the knowledge YAML), compared to weekly capacity
--       (granola ~75/wk, coconut ~57/wk).
--
-- Assumptions / simplifications:
--   * on_hand(product) = SUM(transaction_lines.quantity_lb) WHERE product_id = X
--     — positive receives/makes + negative ships/consumption nets to current qty
--   * batch target_lb sourced from products.default_batch_lb (owner-confirmed
--     for all active batches 2026-04-23)
--   * FG → base batch link uses product_bom (component type='batch')
--   * Cold-mix batches (90008 Fruit Nut, 95005 BS PB Banana) walk one level
--     further via batch_formulas to their upstream oven batch
--   * Graham Crumbs (31012) has base_batch_sku=null — classified as 'repack'
--     and excluded from line-capacity checks
--   * This is planning-only math. Does NOT consider packaging on-hand, lot
--     expiry, or shelf-life (shelf_life_days is NULL plant-wide — see
--     discrepancies.md #10).
-- ============================================================================

-- --- CTEs ------------------------------------------------------------------

WITH

-- Open sales-order lines with positive remaining qty
open_so_lines AS (
    SELECT  sol.id                                           AS line_id,
            so.id                                            AS so_id,
            so.order_number,
            so.customer_id,
            so.requested_ship_date,
            so.status                                        AS so_status,
            sol.product_id                                   AS fg_id,
            sol.quantity_lb                                  AS ordered_lb,
            COALESCE(sol.quantity_shipped_lb, 0)             AS shipped_lb,
            (sol.quantity_lb - COALESCE(sol.quantity_shipped_lb, 0)) AS remaining_lb,
            sol.line_status
    FROM    sales_orders so
    JOIN    sales_order_lines sol ON sol.sales_order_id = so.id
    WHERE   so.status NOT IN ('shipped','cancelled')
      AND   COALESCE(sol.line_status, 'pending') NOT IN ('shipped','cancelled','fulfilled')
      AND   (sol.quantity_lb - COALESCE(sol.quantity_shipped_lb, 0)) > 0
),

-- FG → base batch via product_bom (component type='batch')
fg_to_base_batch AS (
    SELECT  pb.finished_product_id                          AS fg_id,
            pb.component_product_id                         AS base_batch_id,
            p.odoo_code                                     AS base_batch_odoo,
            p.name                                          AS base_batch_name,
            p.default_batch_lb                              AS base_batch_target_lb
    FROM    product_bom pb
    JOIN    products p ON p.id = pb.component_product_id
    WHERE   p.type = 'batch'
),

-- Upstream batch for cold-mix batches (90008 Fruit Nut → 90002, 95005 → 95002)
base_to_upstream_batch AS (
    SELECT  bf.product_id                                   AS base_batch_id,
            bf.ingredient_product_id                        AS upstream_batch_id,
            up.odoo_code                                    AS upstream_batch_odoo,
            up.name                                         AS upstream_batch_name,
            up.default_batch_lb                             AS upstream_batch_target_lb,
            bf.quantity_lb                                  AS upstream_lb_per_base_batch
    FROM    batch_formulas bf
    JOIN    products up ON up.id = bf.ingredient_product_id
    WHERE   up.type = 'batch'
),

-- Current on-hand (sum of +in and -out per product)
on_hand AS (
    SELECT  tl.product_id,
            SUM(tl.quantity_lb)                             AS lb_on_hand
    FROM    transaction_lines tl
    GROUP BY tl.product_id
),

-- Classify each batch into production line
batch_line AS (
    SELECT  p.id                                            AS batch_id,
            p.odoo_code,
            p.name,
            CASE
                WHEN p.odoo_code IN ('90003','90004','90005','90007') THEN 'coconut_line'
                WHEN p.odoo_code IN ('90008','95005')                 THEN 'cold_mix_at_pack'  -- no oven cost here; upstream has the cost
                ELSE 'granola_line'
            END                                             AS production_line
    FROM    products p
    WHERE   p.type = 'batch'
),

-- Per-SO-line planning row, flattened
planning AS (
    SELECT
        osl.so_id,
        osl.order_number,
        osl.requested_ship_date,
        osl.so_status,
        osl.line_id,
        osl.fg_id,
        fg.odoo_code                                        AS fg_odoo,
        fg.name                                             AS fg_name,
        fg.case_size_lb,
        osl.remaining_lb                                    AS fg_remaining_lb,
        COALESCE(fg_oh.lb_on_hand, 0)                       AS fg_on_hand_lb,
        GREATEST(0, osl.remaining_lb - COALESCE(fg_oh.lb_on_hand, 0))
                                                             AS fg_shortfall_lb,

        -- base batch (may be null for graham repack)
        ftb.base_batch_id,
        ftb.base_batch_odoo,
        ftb.base_batch_target_lb,
        COALESCE(bb_oh.lb_on_hand, 0)                       AS base_batch_on_hand_lb,
        bl.production_line                                   AS base_batch_line,

        -- upstream batch (nullable; only 90008, 95005 have one)
        btu.upstream_batch_id,
        btu.upstream_batch_odoo,
        btu.upstream_batch_target_lb,
        btu.upstream_lb_per_base_batch,
        COALESCE(ub_oh.lb_on_hand, 0)                       AS upstream_batch_on_hand_lb,
        ul.production_line                                   AS upstream_batch_line

    FROM    open_so_lines osl
    JOIN    products fg ON fg.id = osl.fg_id
    LEFT JOIN fg_to_base_batch ftb ON ftb.fg_id = osl.fg_id
    LEFT JOIN on_hand fg_oh ON fg_oh.product_id = osl.fg_id
    LEFT JOIN on_hand bb_oh ON bb_oh.product_id = ftb.base_batch_id
    LEFT JOIN batch_line bl ON bl.batch_id = ftb.base_batch_id
    LEFT JOIN base_to_upstream_batch btu ON btu.base_batch_id = ftb.base_batch_id
    LEFT JOIN on_hand ub_oh ON ub_oh.product_id = btu.upstream_batch_id
    LEFT JOIN batch_line ul ON ul.batch_id = btu.upstream_batch_id
),

-- Expand to batches-needed math
batches_needed AS (
    SELECT
        p.*,
        -- base-batch shortfall after on-hand
        GREATEST(0, p.fg_shortfall_lb - p.base_batch_on_hand_lb) AS base_batch_shortfall_lb,
        CASE
            WHEN p.base_batch_target_lb IS NULL OR p.base_batch_target_lb = 0 THEN 0
            ELSE CEIL(GREATEST(0, p.fg_shortfall_lb - p.base_batch_on_hand_lb) / p.base_batch_target_lb)
        END AS base_batches_to_run,

        -- upstream shortfall math (applies only to cold-mix cases)
        -- Need: upstream_lb_per_base_batch × base_batches_to_run, minus on-hand upstream
        CASE WHEN p.upstream_batch_id IS NULL THEN NULL
             ELSE GREATEST(0,
                    (p.upstream_lb_per_base_batch *
                     CASE WHEN p.base_batch_target_lb IS NULL OR p.base_batch_target_lb = 0 THEN 0
                          ELSE CEIL(GREATEST(0, p.fg_shortfall_lb - p.base_batch_on_hand_lb) / p.base_batch_target_lb)
                     END)
                    - p.upstream_batch_on_hand_lb)
        END AS upstream_shortfall_lb,

        CASE WHEN p.upstream_batch_id IS NULL OR p.upstream_batch_target_lb IS NULL OR p.upstream_batch_target_lb = 0 THEN NULL
             ELSE CEIL(
                    GREATEST(0,
                      (p.upstream_lb_per_base_batch *
                       CASE WHEN p.base_batch_target_lb IS NULL OR p.base_batch_target_lb = 0 THEN 0
                            ELSE CEIL(GREATEST(0, p.fg_shortfall_lb - p.base_batch_on_hand_lb) / p.base_batch_target_lb)
                       END)
                      - p.upstream_batch_on_hand_lb
                    ) / p.upstream_batch_target_lb)
        END AS upstream_batches_to_run
    FROM planning p
)

-- =========================================================================
-- (A) DETAIL — ranked open-SO work list
-- =========================================================================
SELECT
    requested_ship_date,
    so_status,
    order_number,
    fg_odoo,
    fg_name,
    ROUND(fg_remaining_lb, 1)             AS fg_remaining_lb,
    ROUND(fg_on_hand_lb, 1)               AS fg_on_hand_lb,
    ROUND(fg_shortfall_lb, 1)             AS fg_shortfall_lb,

    base_batch_odoo,
    ROUND(base_batch_on_hand_lb, 1)       AS base_batch_on_hand_lb,
    ROUND(base_batch_shortfall_lb, 1)     AS base_batch_shortfall_lb,
    base_batches_to_run,
    base_batch_line,

    upstream_batch_odoo,
    ROUND(upstream_batch_on_hand_lb, 1)   AS upstream_batch_on_hand_lb,
    ROUND(upstream_shortfall_lb, 1)       AS upstream_shortfall_lb,
    upstream_batches_to_run,
    upstream_batch_line,

    CASE
        WHEN base_batch_odoo IS NULL THEN 'REPACK (no batch)'   -- e.g. Graham Crumbs
        WHEN upstream_batch_odoo IS NOT NULL THEN
             'COLD MIX — needs ' || upstream_batches_to_run || ' × ' || upstream_batch_odoo
             || ' upstream, then ' || base_batches_to_run || ' × ' || base_batch_odoo
        WHEN base_batches_to_run = 0 THEN 'SHIPPABLE FROM STOCK'
        ELSE base_batches_to_run || ' × ' || base_batch_odoo || ' on ' || base_batch_line
    END AS plan_note
FROM batches_needed
ORDER BY requested_ship_date NULLS LAST, order_number, fg_odoo;

-- =========================================================================
-- (B) SUMMARY — load by production line vs weekly capacity
--
-- Re-run the entire WITH block above, then append this final section in
-- place of the (A) SELECT:
-- =========================================================================

/*
, unioned AS (
    SELECT base_batch_line AS production_line, base_batches_to_run AS batches_due, requested_ship_date
    FROM batches_needed
    WHERE base_batches_to_run > 0 AND base_batch_line IN ('granola_line','coconut_line')
    UNION ALL
    SELECT upstream_batch_line AS production_line, upstream_batches_to_run AS batches_due, requested_ship_date
    FROM batches_needed
    WHERE upstream_batches_to_run > 0 AND upstream_batch_line IN ('granola_line','coconut_line')
)
SELECT
    production_line,
    SUM(CASE WHEN requested_ship_date <= CURRENT_DATE + INTERVAL '7 days'  THEN batches_due ELSE 0 END) AS batches_due_7d,
    SUM(CASE WHEN requested_ship_date <= CURRENT_DATE + INTERVAL '14 days' THEN batches_due ELSE 0 END) AS batches_due_14d,
    SUM(CASE WHEN requested_ship_date >  CURRENT_DATE + INTERVAL '14 days' THEN batches_due ELSE 0 END) AS batches_due_beyond_14d,
    SUM(batches_due)                                                                                     AS batches_due_total,
    CASE production_line WHEN 'granola_line' THEN  75 WHEN 'coconut_line' THEN  57 END AS weekly_capacity,
    CASE production_line WHEN 'granola_line' THEN 150 WHEN 'coconut_line' THEN 114 END AS two_week_capacity,
    CASE
        WHEN production_line = 'granola_line' AND
             SUM(CASE WHEN requested_ship_date <= CURRENT_DATE + INTERVAL '14 days' THEN batches_due ELSE 0 END) > 150 THEN 'OVER 2-WEEK CAPACITY'
        WHEN production_line = 'coconut_line' AND
             SUM(CASE WHEN requested_ship_date <= CURRENT_DATE + INTERVAL '14 days' THEN batches_due ELSE 0 END) > 114 THEN 'OVER 2-WEEK CAPACITY'
        ELSE 'within capacity'
    END AS capacity_flag_14d
FROM unioned
GROUP BY production_line
ORDER BY production_line;
*/
