-- Owner-approved Q8 data fix, separate from migration 054. Idempotent.
-- Run via port 5432. No existing assignments are edited.
INSERT INTO public.product_line_assignments (product_id, line_id)
SELECT p.id, l.id FROM public.products p CROSS JOIN public.production_lines l
WHERE p.odoo_code IN ('90008','90025','90026') AND p.type='batch' AND l.line_code='granola'
ON CONFLICT DO NOTHING;
