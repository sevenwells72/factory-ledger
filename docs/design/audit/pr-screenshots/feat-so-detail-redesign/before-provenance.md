# Frozen baseline provenance

Application baseline: `12b01b6be30c194af2de60fcb50d47a5ed32780a`, the synchronized main HEAD before the redesign. It is checked out detached at `/tmp/factory-ledger-so-detail-baseline`. The application, fixtures, screen recipes, measurement code and thresholds are unchanged in that checkout. The worktree uses the main checkout's installed Node dependencies through a `node_modules` symlink.

```sh
cd /tmp/factory-ledger-so-detail-baseline
node tests/visual/run-visual-audit.mjs --concurrency 4
```

The default run covers all 388 capturable screen/variant pairs: 1440px and 390px in light and dark, plus the existing 200% zoom cases. Every capture has a fresh browser context and stubbed API responses. It makes no production data changes. Date tokens resolve once per run in America/New_York.

Scopes follow PR #45: list is S-24–S-29 (36 captures); detail is S-30–S-40 excluding native dialogs S-35/S-38 (54 captures); other screens account for 298 captures. The generated report and raw results are retained as `before-browser-check.md` and `before-visual-results.json`; the command output is `before-visual.log`.

Selected captures use the same file naming layout as #45. S-24 is saved as `before-sales-orders-top-1440-light.png` and its other width/theme variants; S-25 as `before-sales-orders-1440-light.png`; S-30 as `before-sales-order-detail-1440-light.png`; S-31 as `before-sales-order-lines-1440-light.png`.

The frozen detail fixture retains its original legacy fields. It has no administrative state, fulfillment, or Health v2.1 fields. In particular fixture 109 has a legacy shipped status but effective shipped quantity 0 and remaining quantity 4,500 lb. Any after fixture migration must be disclosed separately; it does not alter this baseline.

GitHub metadata for PR #45 contains no submitted reviews and only the Netlify deployment bot comment. Blubber approval is therefore not established by that metadata. Its preview URL is `https://deploy-preview-45--cns-factory-ledger.netlify.app/`.

## Existing Python expectation regression

The frozen baseline reproduces the stale pallet-line expected dictionary in `test_order_list_exposes_compact_case_line_data_for_pallet_display`: the API includes the additive `line_status: 'pending'` field and the expected dictionary omits it. The single test fails before this redesign; output is retained in `before-python-regression.log`.

```sh
cd /tmp/factory-ledger-so-detail-baseline
env -u DATABASE_URL TEST_DATABASE_URL=postgresql://localhost:5432/factory_ledger_test /Users/michaelgross/dev/factory-ledger/.venv-test/bin/python -m pytest tests/test_sales_order_line_fields.py::test_order_list_exposes_compact_case_line_data_for_pallet_display
```

## Canonical baseline result

The full run completed 388/388 captures with zero runner errors: 3,037 passing rule cells, 633 failing cells, 64 warnings, and 1,310 not-applicable cells. Failing cells by scope: list 0; detail 212; other 421. The eight STATUS per-rule counts are recorded in `before-summary.md` and every verdict in `before-summary.json`.

All 388 original screenshots remain locally preserved in `/tmp/fl-so-detail-canonical-before-shots` independently of later diagnostic runs. Diagnostic reruns must retain separate raw results and must not replace or splice into the canonical baseline.

## List layout baseline

The unchanged baseline layout suite passes all 26 width/theme/state cases and all 260 assertions. Raw case results are `before-layout-results.json`; command output is `before-layout.log`.

```sh
cd /tmp/factory-ledger-so-detail-baseline
node tests/visual/run-sales-orders-layout.mjs dashboard /tmp/fl-so-detail-before-layout
```

## Rebuild failure tables

The evidence scripts evaluate the existing runner's exact `verdict` function to summarize the stored raw JSON. They do not change rule measurements or thresholds. From the repository root:

```sh
node docs/design/audit/pr-screenshots/feat-so-detail-redesign/summarize-audit.mjs . docs/design/audit/pr-screenshots/feat-so-detail-redesign before docs/design/audit/pr-screenshots/feat-so-detail-redesign/before-visual-results.json
node docs/design/audit/pr-screenshots/feat-so-detail-redesign/summarize-audit.mjs . docs/design/audit/pr-screenshots/feat-so-detail-redesign after docs/design/audit/pr-screenshots/feat-so-detail-redesign/after-visual-results.json
node docs/design/audit/pr-screenshots/feat-so-detail-redesign/compare-audits.mjs docs/design/audit/pr-screenshots/feat-so-detail-redesign
```

Instance counts use each STATUS check's `failures` field. For other rules, touch and contrast also use `failures`; horizontal overflow uses `offenderCount`; occlusion uses `coveredCount`. Refresh CLS has no single rendered-instance count, so its instance value is `null`; its failing cell count remains exact. The comparison keys every cell by screen, variant, and rule and never treats aggregate improvements as proof that no new cells failed.
