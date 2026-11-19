-- Keep lifetime usage and move identity for audit and safe reactivation.
DELETE FROM migration_markers WHERE name='098_no_stock_by_hand';
