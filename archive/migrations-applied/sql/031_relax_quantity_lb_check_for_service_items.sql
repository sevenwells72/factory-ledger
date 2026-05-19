-- Allow service items (Pallets, freight, etc.) with zero quantity_lb
-- Previously: quantity_lb > 0 (rejected service items)
-- Now: quantity_lb >= 0

ALTER TABLE sales_order_lines DROP CONSTRAINT IF EXISTS sales_order_lines_quantity_lb_check;
ALTER TABLE sales_order_lines ADD CONSTRAINT sales_order_lines_quantity_lb_check CHECK (quantity_lb >= 0);
