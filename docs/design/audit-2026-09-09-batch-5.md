# Sales Orders design audit 5 — local commit only

Branch: `fix/design-audit-5`. Base: fetched `origin/main` `43960a4`, descendant of `8b391ab`. No push, PR, merge or deployment.

Desktop CSS gives the table its own horizontal scroll area and keeps a visible “Scroll sideways to see all columns →” hint beside sorting. The sort controls stay below the nav, measured header and 44px tab bar. The section uses `overflow:clip` to retain rounded corners without trapping sticky positioning; the outer wrapper uses visible overflow. Blocker chip text can wrap. Responsive cards at 390px are unchanged. No application JavaScript changes were required.

Changed asset: `shell-layout.css?v=3` → **v4** on index, history, sankey, process-flow and traceability, strictly above current main. Other assets are unchanged. Project CHANGE_LOG, FACTORY_LEDGER_CHANGELOG (row 127, LOCAL ONLY), and the global change log were updated.

## Verification

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

## Integration handoff

Await owner go-ahead before rebasing and pushing. `fix/so-intake-followup` is in flight. This branch does not edit dashboard.js; index.html contains only the shell CSS version bump. On the later rebase, preserve both changelog histories, renumber row 127 if it collides, and bump shell CSS above whatever main serves then. The sticky offset relies on the existing 44px desktop tab-bar height; revisit if that chrome changes.
