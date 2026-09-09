# Sales Orders design audit 5

Branch: `fix/design-audit-5`, PR #37, based on post-#36 `main` (`ea5c15b`). Review only; do not merge.

## Current behavior

The Sales Orders list uses the available desktop width. Above 1100px, its native table fits all columns, including Blockers, without horizontal scrolling. Status takes only its content width; the Factory Ready stamp sits below the status badge and wraps within 22ch. Dispatch badges, pallet descriptions and customer names can wrap instead of forcing large minimum column widths. Blockers labels wrap naturally without truncation or line clamping.

The sort row remains sticky beneath the navigation, measured header and tab bar. At 769–1100px the table uses local horizontal scrolling with the scroll hint. At 768px and below the existing responsive cards remain. The 44px desktop tab-bar assumption still applies to the sticky offset.

The shell stylesheet is **v5** on all five referencing pages. Other query versions remain intake **v12**, dashboard JS **v59**, and dashboard CSS **v42**, already above main's v11/v58/v41. Application JS, dashboard.css, backend and requirements content remain unchanged from main. Changelog row **128** remains unique; deployed intake row 127 is preserved.

## Why the original check missed the issue

PR #37 initially proved that Blockers could be reached *after scrolling*. It did not prove that the columns fit at desktop width. On the original live preview, a 1158px table had 1702px of content: Status occupied 366px and Pallets 344px. The unwrapped ready stamp and pallet description, plus the 1200px content cap, forced unnecessary overflow.

## Focused regression

The updated fixture test uses production-length order numbers, a PO attachment, a long customer name, Factory Ready stamps and mixed-pallet descriptions. It checks all columns **before any horizontal scrolling**, cell-content containment, Status width relative to its widest badge, two-line maximum Blockers labels, sticky controls, the narrow-width fallback and phone cards.

**12/12 scenarios pass**: 1440, 1385, 1280, 1101, 1024 and 390, each in light and dark. The same test against the previous PR head fails all eight fitted-desktop scenarios and both tablet fallback scenarios; both phone scenarios already pass.

| Viewport | Table / available width | Status width | Result |
|---|---|---|---|
| 1440 | 1398 / 1398px | 196.8px | All columns fit at scroll position zero |
| 1385 | 1343 / 1343px | 196.8px | All columns fit at scroll position zero |
| 1280 | 1238 / 1238px | 196.8px | All columns fit at scroll position zero |
| 1101 | 1059 / 1059px | 196.8px | All columns fit at scroll position zero |
| 1024 | 982px viewport for the table | — | Local scrolling and hint; sticky sort row |
| 390 | 364 / 364px | — | Existing cards; no page overflow |

Desktop, 1280px and phone captures were visually inspected. The fit checks explicitly reject the old failure mode where `overflow:clip` hid overflowing columns without widening the document.

## Full audit

Fresh full Playwright audit: **310 captures** at 1440 and 390 in both themes, **zero errors**.

| Width (both themes) | Previous PR PASS / FAIL / WARN | Updated PASS / FAIL / WARN |
|---|---|---|
| 1440 | 672 / 95 / 25 | 686 / 84 / 22 |
| 390 | 530 / 73 / 17 | 529 / 73 / 18 |
| Total | 1202 / 168 / 42 | 1215 / 157 / 40 |

Both runs have 138 N/A. Eight desktop TOUCH-003 verdicts on Sales Orders S-25–S-28 improve from FAIL to PASS (both themes). The Sales Orders group is now **108 PASS / 8 existing mobile touch FAIL / 4 N/A**, previously 100 / 16 / 4. Contrast, horizontal-overflow and fixed-bar verdicts are unchanged. The other 41 changed verdicts are refresh-layout-shift measurements outside Sales Orders; those timing-sensitive movements are not claimed as improvements. No non-refresh rule regresses.

All **47 intake Node tests pass**. Focused-harness syntax and `git diff --check` pass.

## Commands

```sh
npm run test:visual -- --variants 1440-light,1440-dark,390-light,390-dark --concurrency 12
node tests/visual/run-sales-orders-layout.mjs [dashboard-root] [output-dir]
node --test tests/test_er_intake_logic.js
node --check tests/visual/run-sales-orders-layout.mjs
git diff --check
```
