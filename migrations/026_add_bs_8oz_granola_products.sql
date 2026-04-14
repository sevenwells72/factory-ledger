-- Migration 026: Add Blue Stripes 8 OZ Granola SKUs
-- Run in Supabase SQL Editor (production: MyFirstProject, us-east-1)

INSERT INTO products (name, odoo_code, type, uom, case_size_lb, label_type, active)
VALUES
  ('BS Granola – Hazelnut Butter – 6x8 OZ Case',         '70085', 'finished', 'case', 3.0, 'private_label', true),
  ('BS Almond Butter Granola – 6x8 OZ Case',             '70086', 'finished', 'case', 3.0, 'private_label', true),
  ('BS Granola – Dark Chocolate – 6x8 OZ Case',          '70087', 'finished', 'case', 3.0, 'private_label', true),
  ('BS Granola – Peanut Butter Banana – 6x8 OZ Case',    '70088', 'finished', 'case', 3.0, 'private_label', true);
