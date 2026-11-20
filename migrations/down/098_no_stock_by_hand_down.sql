-- Keep lifetime usage and move identity for audit. Reactivation starts a new marker window.
DELETE FROM migration_markers WHERE name='098_no_stock_by_hand';
