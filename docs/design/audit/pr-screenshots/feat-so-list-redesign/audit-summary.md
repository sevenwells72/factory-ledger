# Sales Orders list redesign — rendered evidence

The canonical before and after audits each render **388 screen/variant captures** using fresh browser contexts and the same four-worker concurrency. No STATUS rule, threshold, or measurement was relaxed. Per-rule numbers below count **failing captures**; a capture may fail more than one rule.

## Scope and STATUS results

List: **S-24–S-29, 36 captures**. Detail: **S-30–S-40, 54 captures**, excluding browser-native dialogs S-35/S-38. Other screens: **298 captures**. Detail remains a separate Step 4 scope.

| STATUS rule | List before | List after | Detail before | Detail after |
|---|---:|---:|---:|---:|
| STATUS-002 | 24 | 0 | 12 | 12 |
| STATUS-004 | 20 | 0 | 42 | 42 |
| STATUS-005 | 24 | 0 | 42 | 42 |
| STATUS-006 | 0 | 0 | 30 | 30 |
| STATUS-007 | 12 | 0 | 30 | 30 |
| STATUS-008 | 8 | 0 | 14 | 14 |
| STATUS-010 | 0 | 0 | 36 | 36 |
| STATUS-011 | 0 | 0 | 0 | 0 |

| STATUS rule | Other screens before | Other screens after |
|---|---:|---:|
| STATUS-002 | 0 | 0 |
| STATUS-004 | 32 | 32 |
| STATUS-005 | 102 | 102 |
| STATUS-006 | 28 | 28 |
| STATUS-007 | 0 | 0 |
| STATUS-008 | 32 | 32 |
| STATUS-010 | 48 | 38 |
| STATUS-011 | 16 | 16 |

## Layout regression

| Harness result | Before | After |
|---|---:|---:|
| Width/theme/state cases | 12 | 26 |
| Cases with any failed assertion | 0 | 0 |

Before covers Open at 1440, 1385, 1280, 1101, 1024, and 390px in both themes. After adds 1200px and Closed/Cancelled at 1440, 1200, and 390px in both themes. Identifiers use production-length SO numbers and a long customer name. All measured desktop rows at ≥1200px are 52px. Desktop fit, sticky sort controls, narrow-tablet scrolling, last-column reachability, and mobile cards remain checked. The removed Status/Blockers-column assertions are replaced by the eight-column order, expander-in-SO-cell, and ≤56px row assertions.

## All-rule comparison

The 13-rule comparison checks each screen, variant, and rule separately; aggregate improvements cannot hide a new failure. Counts in this table are **failing capture/rule pairs**, not unique captures.

| Scope | Before failing pairs | After failing pairs |
|---|---:|---:|
| list | 104 | 0 |
| detail | 212 | 212 |
| other | 431 | 421 |

Canonical after capture errors: **0**. New failing pairs outside the list: **16**.

| Screen | Variant | Rule | Before | After |
|---|---|---|---|---|
| S-01 | 1440-light | LAYOUT-020 | WARN | FAIL |
| S-02 | 390-dark | LAYOUT-020 | PASS | FAIL |
| S-03 | 1440-dark | LAYOUT-020 | PASS | FAIL |
| S-05 | 1440-dark | LAYOUT-020 | WARN | FAIL |
| S-08 | 390-dark | LAYOUT-020 | PASS | FAIL |
| S-08 | 1440-light | LAYOUT-020 | PASS | FAIL |
| S-09 | 390-dark | LAYOUT-020 | PASS | FAIL |
| S-10 | 390-light | LAYOUT-020 | PASS | FAIL |
| S-12 | 390-light | LAYOUT-020 | WARN | FAIL |
| S-13 | 1440-light | LAYOUT-020 | PASS | FAIL |
| S-16 | 1440-light | LAYOUT-020 | PASS | FAIL |
| S-17 | 390-light | LAYOUT-020 | WARN | FAIL |
| S-54 | 1440-light | LAYOUT-020 | WARN | FAIL |
| S-54 | 1440-dark | LAYOUT-020 | WARN | FAIL |
| S-55 | 1440-dark | LAYOUT-020 | WARN | FAIL |
| S-56 | 1440-light | LAYOUT-020 | WARN | FAIL |

Across the 32 LAYOUT-020 FAIL/non-FAIL swaps, 31 have identical settled anchor counts, moved/vanished anchors, maximum displacement, and worst-anchor lists. The remaining S-56 desktop-light settled geometry improves from one moved clear-search button (4px) to no movement. Only transient CLS differs in the other pairs. No new failure occurs outside LAYOUT-020 on detail or other screens. The strict no-new-failure-cell criterion across all rules is therefore **not met by the canonical matrices**. This limitation is retained, not converted to a pass.

Full per-rule/variant differences, including warnings, are in [comparison.json](comparison.json).

## Isolation and controlled rechecks

The original harness reused browser contexts between captures. The scheduler’s sample button appends orders and stores them in local storage, so later scheduler captures could contain different inputs depending on worker assignment. Canonical runs create a fresh context, install the API stub, capture one screen, and close that context in `finally`. The same isolation-only change was applied to the frozen baseline runner. Original summaries/logs remain as `*-original.*`; redundant original raw data and captures are preserved locally in `/tmp/fl-so-original-audit-evidence`.

A paired serial recheck of S-02/S-03/S-04/S-09/S-10/S-11/S-91 also records the earlier storage/timing variation in [before-regression-recheck.json](before-regression-recheck.json) and [after-regression-recheck.json](after-regression-recheck.json). These diagnostic runs do not replace or merge selected results into the canonical 388-capture matrices. A second experiment added 200 ms latency only during refresh, with initial setup unchanged, at S-01/S-02/S-54 and both 390/1440 themes, twice per snapshot. Ten of 12 cells stayed stable and matched, while current S-02 desktop alternated WARN (~0.228 CLS) and FAIL (0.258 CLS); baseline stayed WARN. Fixed latency alone did not remove the uncertainty. See [refresh-latency-experiment.md](refresh-latency-experiment.md) and the four *-latency-trial-*.json raw files. No latency change was adopted in the canonical harness. A final bounded source trace reproduced **0.258 / FAIL on the frozen baseline itself in both desktop themes**. The observer source rectangles identify existing Finished Goods, Production, Attention, and reference-calendar/header refresh changes being grouped into different paint frames; settled positions recover. This establishes existing refresh timing as a cause of the S-02 threshold crossing. The canonical 16 added/16 removed LAYOUT-020 cells remain explicit and the strict raw gate is not called clean. See [source trace table](refresh-cls-source-trace.md), [per-frame diagnosis](refresh-cls-source-diagnosis.md), and [raw source entries](refresh-cls-source-trace.json).

Screen recipes preserve IDs: S-26 now means Ready to ship rather than the removed Dispatch Queue filter; S-28 keeps the expanded ready-note state with its new label; detail recipes open the explicit Open order details action, and S-34 reaches fixture 109 through Shipped. Existing detail payloads are unchanged. The new list fixture and counts route model the new API separately.

## Commands and provenance

Before application baseline: `origin/main` commit `e772e36`, frozen before implementation at `/tmp/factory-ledger-so-baseline.sItrEs`. Its original fixtures/recipes are retained with only the browser-context isolation patch.

```sh
# In the frozen baseline directory
node tests/visual/run-visual-audit.mjs --concurrency 4
# In the redesigned checkout
node tests/visual/run-visual-audit.mjs --concurrency 4
node tests/visual/run-sales-orders-layout.mjs dashboard /tmp/factory-ledger-so-baseline.sItrEs/after-layout-terminal
node tests/visual/run-sales-orders-interactions.mjs
node tests/visual/run-so-exit-actions.mjs
```

The list integration suite passed at 1440 and 390px: tab membership/query parameters/count refreshes; customer/hide-ready refinements; hover/focus/tap/Escape explanations; expandable allocation detail; ready writes; preview/commit ordering and invalidation; cancellation note/related-SO requirements; the 409 close offer; reopening from both terminal tabs; and plain-language 503 errors. No browser page errors occurred. See [after-interaction-results.json](after-interaction-results.json).

## Captures and raw results

| View | Before | After |
|---|---|---|
| 1440-light | [List](before-sales-orders-1440-light.png) · [Tabs/top](before-sales-orders-top-1440-light.png) | [List](after-sales-orders-1440-light.png) · [Tabs/top](after-sales-orders-top-1440-light.png) |
| 1440-dark | [List](before-sales-orders-1440-dark.png) · [Tabs/top](before-sales-orders-top-1440-dark.png) | [List](after-sales-orders-1440-dark.png) · [Tabs/top](after-sales-orders-top-1440-dark.png) |
| 390-light | [List](before-sales-orders-390-light.png) · [Tabs/top](before-sales-orders-top-390-light.png) | [List](after-sales-orders-390-light.png) · [Tabs/top](after-sales-orders-top-390-light.png) |
| 390-dark | [List](before-sales-orders-390-dark.png) · [Tabs/top](before-sales-orders-top-390-dark.png) | [List](after-sales-orders-390-dark.png) · [Tabs/top](after-sales-orders-top-390-dark.png) |

Additional views: [Health details at 1440](after-sales-orders-health-1440-light.png), [Health details at 390](after-sales-orders-health-390-light.png), [Closed at 1440](after-sales-orders-closed-1440-light.png), [Closed at 390](after-sales-orders-closed-390-light.png).

Canonical raw results: [before](before-visual-results.json), [after](after-visual-results.json). Full rendered reports: [before](before-browser-check.md), [after](after-browser-check.md). Layout results: [before](before-layout-results.json), [after](after-layout-results.json).

The global generated report remains `docs/design/audit/06-browser-check.md` for the canonical after run. Screenshots linked inside full generated reports are build outputs; the selected stable screenshots above are committed review evidence.

Preview: [Netlify deploy preview](https://deploy-preview-45--cns-factory-ledger.netlify.app/) · [asset/startup smoke evidence](preview-smoke.md). The smoke used stubbed API responses and made no production writes.
