# Sales Orders design audit 5

Branch: `fix/design-audit-5`. Rebased onto `origin/main` `ea5c15b` after PR #36 merged (`30b4f11`). Prepared for PR review; do not merge without owner approval.

Desktop CSS gives the table its own horizontal scroll area and keeps a visible “Scroll sideways to see all columns →” hint beside sorting. The sort controls stay below the nav, measured header and 44px tab bar. The section uses `overflow:clip` to retain rounded corners without trapping sticky positioning; the outer wrapper uses visible overflow. Blocker chip text can wrap. Responsive cards at 390px are unchanged. No application JavaScript changes were required.

Changed asset: `shell-layout.css?v=3` → **v4** on index, history, sankey, process-flow and traceability. As requested, intake-logic references advance v11 → **v12**, dashboard.js v58 → **v59**, and dashboard.css v41 → **v42** (including the older history-page reference). Those three asset files are byte-identical to current main; their query versions are strictly above main. FACTORY_LEDGER_CHANGELOG uses **row 128**, preserving deployed row 127.

## Original before/after verification

Full Playwright audit: **310 captures before and 310 after**, all available screens at 1440 and 390, light and dark. Local fixture data only; no production data changed. Counts are rule verdicts, not unique defects.

| Width (both themes) | Before PASS / FAIL / WARN | After PASS / FAIL / WARN |
|---|---|---|
| 1440 | 672 / 95 / 25 | 674 / 92 / 26 |
| 390 | 529 / 69 / 22 | 532 / 65 / 23 |
| Total | 1201 / 164 / 47 | 1206 / 157 / 49 |

Both runs: 138 N/A, **0 errors**. Every verdict change is LAYOUT-020 refresh layout shift on screens outside Sales Orders. These timing-sensitive changes are not credited as improvements from this fix. TOUCH-003, ACCESS-008, LAYOUT-003/ACCESS-001 and LAYOUT-011 verdicts are identical across all captures.

Sales Orders S-24–S-29, all four variants: **100 PASS / 16 FAIL / 4 N/A before and after**. The remaining failures are existing touch-target findings. The broad audit does not test the new persistent scroll hint or intermediate-scroll sorting visibility, so a focused regression harness covers those requirements.

Focused harness: **4 desktop scenarios fail before → all 6 width/theme scenarios pass after** (1385, 1440, 390; light/dark). At desktop widths the 1618px table scrolls inside 1158px; all Blockers cells and chip text are reachable at the right edge, the document does not overflow, and sorting sits at y=182px, exactly the sticky-stack bottom. The 390px cards remain contained. Mobile does not use sticky sorting. An additional 1101px check with Resize columns expanded places the controls at y=182–532 with no document overflow.

Commands:

```sh
npm run test:visual -- --variants 1440-light,1440-dark,390-light,390-dark --concurrency 12
node tests/visual/run-sales-orders-layout.mjs [dashboard-root] [output-dir]
node --check tests/visual/run-sales-orders-layout.mjs
git diff --check
```

## Rebase verification

The rebase preserved the deployed SO intake logic and both changelog histories. Row 128 is unique and does not collide with row 127. The sticky offset relies on the existing 44px desktop tab-bar height; revisit if that chrome changes.

Fresh full Playwright audit against post-#36 main: **310 captures on each revision**, 1440 and 390 in light and dark, with fixture API responses only.

| Width (both themes) | Main PASS / FAIL / WARN | Rebased branch PASS / FAIL / WARN |
|---|---|---|
| 1440 | 666 / 95 / 31 | 672 / 95 / 25 |
| 390 | 526 / 75 / 19 | 530 / 73 / 17 |
| Total | 1192 / 170 / 50 | 1202 / 168 / 42 |

Both runs have 138 N/A and **zero errors**. All 38 changed verdicts are LAYOUT-020 refresh shifts; the touch, contrast, horizontal-overflow and fixed-bar verdicts are identical. These timing-sensitive changes are not credited as improvements. Sales Orders S-24–S-29 retains the same **16 existing touch-target failures**: main 99 PASS / 16 FAIL / 1 WARN / 4 N/A; branch 100 PASS / 16 FAIL / 4 N/A. Its only changed verdict is S-26 1440-light LAYOUT-020 WARN → PASS.

Focused Playwright checks: **6/6 pass** (1385, 1440 and 390, both themes): sticky controls visible, sideways-scroll hint present on desktop, no document overflow, responsive mobile cards retained, and blocker labels reachable. Desktop sorting sits at y=182, immediately below the sticky chrome. Desktop and phone captures were visually inspected.

Post-rebase intake Node tests: **47/47 pass**, including the merged CASE→LB→pick regression. `node --check tests/visual/run-sales-orders-layout.mjs` and `git diff --check` pass. No application JS, dashboard.css, backend or requirements content differs from main. Only shell-layout.css changes runtime styling; the higher query versions are intentional cache invalidation requested for this integration.
