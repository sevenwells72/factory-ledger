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
