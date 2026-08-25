-- found-lot-suppliers-2026-08-25.sql  ── DRAFT, NOT EXECUTED
-- Supplier retro-attribution for the found_inventory lots that break batch
-- trace completeness (report §3g-ii: 0/49 batches fully resolve, dominated
-- by these lots).
--
-- Field mapping (the lots table has no supplier/received_on/reference
-- columns): supplier + reference → entry_source_notes (appended, tagged
-- SUPPLIER-ATTRIB-2026-08-25); the supplier's own lot number →
-- supplier_lot_code; received_on → received_at.
--
-- Placeholders to fill before running:  :vendor  :received_on
-- (psql -v vendor="'Franklin Baker'" -v received_on="'2026-05-01'" style).
--
-- LIMITATION (deliberate): this updates the lots table only. It does NOT
-- create receive transactions, so /trace/* and the §3g metric (which look
-- for a posted receive with shipper_name) will still show these lots as
-- unresolved until either (a) backdated receive+offset txns are posted (an
-- owner decision — changes ledger history) or (b) trace/metric also read
-- lots.supplier_lot_code / entry_source_notes. This file records the facts
-- losslessly either way.
--
-- Verified 2026-08-25 (read-only): every lot below was credited by 'adjust'
-- Found-inventory txns (no shipper fields anywhere), entry_source =
-- 'found_inventory'.

BEGIN;

-- ── Desiccated coconut, found at the 2026-05-14 physical count ────────────
-- The numeric codes ARE the supplier's own lot numbers. Vendor unknown to
-- FL (Franklin Baker and Phildesco are the desiccated-coconut suppliers on
-- file) → :vendor placeholder.

-- 25120 — Coconut Macaroon Desiccated, lot 616, on-hand 1,013 lb
UPDATE lots
   SET supplier_lot_code = COALESCE(supplier_lot_code, '25120'),
       received_at       = COALESCE(received_at, :received_on),
       entry_source_notes = concat_ws(' | ', entry_source_notes,
         'SUPPLIER-ATTRIB-2026-08-25: supplier ' || :vendor ||
         ', supplier lot 25120, received ' || :received_on ||
         ' (found at 2026-05-14 physical count)')
 WHERE id = 616 AND lot_code = '25120';

-- 25120 — Coconut Medium Desiccated, lot 620, on-hand 16,375 lb
-- (same physical sticker family as lot 616; two FL lots because the count
-- split it across two products)
UPDATE lots
   SET supplier_lot_code = COALESCE(supplier_lot_code, '25120'),
       received_at       = COALESCE(received_at, :received_on),
       entry_source_notes = concat_ws(' | ', entry_source_notes,
         'SUPPLIER-ATTRIB-2026-08-25: supplier ' || :vendor ||
         ', supplier lot 25120, received ' || :received_on ||
         ' (found at 2026-05-14 physical count)')
 WHERE id = 620 AND lot_code = '25120';

-- 6013 — Coconut Flake Desiccated, lot 624, on-hand 6,750 lb
UPDATE lots
   SET supplier_lot_code = COALESCE(supplier_lot_code, '6013'),
       received_at       = COALESCE(received_at, :received_on),
       entry_source_notes = concat_ws(' | ', entry_source_notes,
         'SUPPLIER-ATTRIB-2026-08-25: supplier ' || :vendor ||
         ', supplier lot 6013, received ' || :received_on ||
         ' (found at 2026-05-14 physical count)')
 WHERE id = 624 AND lot_code = '6013';

-- Same family, uncomment if the vendor answer covers them too:
--   lot 610 '6012' Coconut Fancy Desiccated,  on-hand 2,800 lb
--   lot 612 '6012' Coconut Flake Desiccated,  on-hand 0 lb
--   lot 613 '6020' Coconut Flake Desiccated,  on-hand 0 lb
--   lot 617 '6020' Coconut Medium Desiccated, on-hand 7,000 lb

-- ── Parker Flavors — Flavor – Almond, lot 34 '26-02-03-FOUND-002' ─────────
-- Found 2026-02-03 tagged 'predates_system' (on-hand now 0, but it breaks
-- the trace of 9 of the last 30 days' batches). Receiving-form evidence:
-- Parker Flavor(s) is the almond-flavor vendor — deliveries 2026-04-21
-- (Almond Flavor 40 lb × 20, sticker 260521PARK001 [sic, = 260421]) and
-- 2026-07-21 (BOL 10032188, almond/chocolate/vanilla/cinnamon flavors).
-- The found lot itself predates those receipts; received_at stays as the
-- found date, the attribution is recorded as supplier identity + reference.
UPDATE lots
   SET entry_source_notes = concat_ws(' | ', entry_source_notes,
         'SUPPLIER-ATTRIB-2026-08-25: supplier Parker Flavors '
         '(almond-flavor vendor per receiving form: 2026-04-21 delivery '
         'sticker 260521PARK001 [sic], 2026-07-21 BOL 10032188); original '
         'receipt predates FL go-live')
 WHERE id = 34 AND lot_code = '26-02-03-FOUND-002';

-- ── Essex Food — Salt, lot 48 '26-02-03-FOUND-016' ────────────────────────
-- Found 2026-02-03 tagged 'predates_system'. Receiving-form evidence:
-- Essex Food (Ingredients) is the salt vendor — deliveries 2026-04-21
-- (Salt 50 lb × 49 + Corn Starch, sticker 260421ESSE, BOL 211226040498),
-- 2026-07-13 (sticker 260713ESSE, BOL 2120050), 2026-08-05 (sticker
-- 260805ESSE, BOL 2121095, Salt 50 lb × 98).
UPDATE lots
   SET entry_source_notes = concat_ws(' | ', entry_source_notes,
         'SUPPLIER-ATTRIB-2026-08-25: supplier Essex Food Ingredients '
         '(salt vendor per receiving form: 2026-04-21 BOL 211226040498, '
         '2026-07-13 BOL 2120050, 2026-08-05 BOL 2121095); original '
         'receipt predates FL go-live')
 WHERE id = 48 AND lot_code = '26-02-03-FOUND-016';

-- Sanity: exactly the intended rows were touched (expect 5 rows, or 9 with
-- the commented family uncommented).
SELECT id, lot_code, supplier_lot_code, received_at,
       right(entry_source_notes, 90) AS attrib_tail
  FROM lots
 WHERE entry_source_notes LIKE '%SUPPLIER-ATTRIB-2026-08-25%'
 ORDER BY id;

COMMIT;
