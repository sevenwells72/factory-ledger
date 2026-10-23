# Wallpaper refresh verification

Baseline: `origin/main` commit `27a2518655da43ae31860a0bcf8c1f73db114422`, copied before styling began. All browser requests use local static files and local API fixtures. No production data, external writes, deployment, or merge was involved.

## Results and scope

The existing Playwright harness completed **310 baseline and 310 after captures with zero capture errors**, at 1440×900 and 390×844 in light and dark themes. The raw reports retain existing dashboard findings; this is not an all-green claim for the entire design inventory.

| Check | Baseline failing cells | After failing cells | Interpretation |
|---|---:|---:|---|
| Touch targets | 94 | 94 | Same failing cells; no new failures |
| CSS-only text contrast | 9 | 28 | All 28 after cells concern inactive controls: Ascending sort buttons (24) or Create Expected Receipt (4). WCAG 1.4.3 exempts inactive UI controls. Raw results and thresholds remain unchanged. |
| Horizontal overflow | 4 | 4 | Same failing cells; no new failures |
| Bottom-bar occlusion | 8 | 8 | Same failing cells; no new failures |
| Numeric refresh CLS > 0.25 | 40 | 43 | Nine newly flagged cells and six resolved cells; all new cells were triggered by the explicit Refresh button. See the distinction below. |

**All 56 genuine background-interval captures in each full run recorded CLS 0 and zero moved anchors.** The existing harness's report and LAYOUT-020 distinguish background movement from changes caused by explicit user action. Its `#refresh-btn` probe programmatically presses that control; the numeric threshold still reports those shifts, although explicit action is allowed by the standard. `cls-trigger-comparison.json` preserves the matched 24-before/24-after subset, all triggered by `#refresh-btn`. Seven of the nine newly flagged cells fell below the numeric failure threshold on rerun; two still exceeded it (S-04 / 390-light and S-06 / 1440-dark). The baseline itself exceeded the threshold in three of those cells on rerun. These explicit-refresh repaint scores vary with timing; the original and repeated values remain recorded. No new automatic/background layout regression was found.

The focused final-source run covers Create Order, Record entry, FL Assistant, Exceptions, Runs, and History: **24 Chromium captures**, with 686 eligible text checks, no rendered-contrast or horizontal-overflow failures, and a minimum measured ratio of 5.71:1. **12 WebKit captures at 390px** add 326 eligible text checks, no rendered-contrast or overflow failures, and a minimum ratio of 5.76:1. Every focused final capture confirms `cover`, fixed wallpaper on desktop, and scrolling wallpaper on phones. Existing Runs touch findings and the Assistant's closed-details receipt-control finding are unchanged.

## Screenshots

The portable before/after images requested for review are:

```text
before-create-order-{1440,390}-{light,dark}.png
after-create-order-{1440,390}-{light,dark}.png
before-record-entry-{1440,390}-{light,dark}.png
after-record-entry-{1440,390}-{light,dark}.png
before-fl-assistant-{1440,390}-{light,dark}.png
after-fl-assistant-{1440,390}-{light,dark}.png
```

The same pattern covers Exceptions, Runs, and History. `after-webkit-*-390-{light,dark}.png` records the WebKit phone checks. The before and final screenshot sets were refreshed with the same corrected capture helper.

`before-browser-check.md` and `after-browser-check.md` are preserved raw harness reports. Their screenshot and standards links retain the original `docs/design/audit/` relative paths and are not portable from this folder. Broad-harness PNGs are temporary build output under `/tmp/fl-wallpaper-baseline-20261022` and `/tmp/fl-wallpaper-final-20261022`; the focused screenshots above are the committed image evidence. The baseline temporary PNG output was subsequently reused for scoped checks, so the full baseline PNG set is not archived. The full baseline JSON in this folder remains intact.

## Contrast supplement and provenance

The canonical CSS-only checker treats image/gradient backing as indeterminate. `tests/visual/lib/rendered-contrast.mjs` supplements it by sampling actual browser-composited pixels beneath text rectangles, including transparent cards and gradients. Text paint is temporarily hidden without changing layout; transitions are disabled while paint is hidden and restored. The helper normalizes computed CSS colors through a canvas so `color(srgb …)` values from `color-mix()` are measured correctly. Inactive controls are excluded from this supplement, without changing the canonical checker. Body/large-text thresholds remain 4.5:1 and 3:1.

| Evidence | CSS source and interpretation |
|---|---|
| `before-visual-results.json` | Original complete 310-case baseline |
| `after-visual-results.json`, `after-rendered-contrast-initial.json` | Complete 310-case after run at wallpaper SHA-256 `b6dbeac8af81f41c5cfdf316d4a7c4ca6647c642b3eff94c6c90a01edcd13a82` |
| `after-rendered-contrast-color-recheck.json` | 76 cases at that same CSS hash, repeated with correct canvas conversion of modern CSS colors; supersedes initial color-mix measurements |
| `before-rendered-contrast-recheck.json`, `before-rendered-contrast-trace.json` | 22 and 4 baseline cases, measured with the corrected color conversion |
| `after-scoped-visual-results.json`, `after-rendered-contrast-final-recheck.json` | 30 final-source captures: S-02, S-02b, S-16, S-17, S-18, S-63, S-64, S-68 in all applicable variants |
| `after-page-checks.json`, `after-webkit-page-checks.json` | Final-source focused checks and stylesheet hashes |

The only application change after the complete after run was `.late-lag { color: var(--fl-amber); }`. Final wallpaper SHA-256 is **`a8e7a89138588ffef1d1792a3a3b9e29a3cfc9d854c3cf4ed4cd78fc5d19f63f`**. The final 30-case recheck verifies that the affected late-entry timestamps pass in both themes and widths. The remaining seven raster-flagged cells have matching baseline findings: compact-calendar shipment dots inside text rectangles, trace text beneath dropdown shadows, and SVG edge labels. Trace/graph ratios improved. Calendar-dot samples remain conservative rectangle samples rather than a glyph-mask contrast proof; the existing tiny calendar and overlay details were not redesigned in this task. No newly failing active-text case remains relative to the measured baseline.

`visual-summary.json` contains exact new/resolved canonical failing-cell IDs. `visual-evidence-sha256.json` records hashes for the visual helpers, images, and reports. Existing unmatched `/auth/session/config` fixture requests in the general harness remain local 404s; focused runs have zero unknown, remote, or page-error events. These are fixture-driven browser checks, not physical-device, production-data, or exhaustive accessibility certification.

## Reproduction

```sh
node tests/visual/run-visual-audit.mjs --variants 1440-light,1440-dark,390-light,390-dark --concurrency 8
WALLPAPER_RASTER_OUT=/tmp/wallpaper-raster.json node --import ./tests/visual/lib/wallpaper-audit-hook.mjs tests/visual/run-visual-audit.mjs --variants 1440-light,1440-dark,390-light,390-dark --concurrency 8
WALLPAPER_PHASE=after node tests/visual/run-wallpaper-refresh.mjs
WALLPAPER_PHASE=after-webkit WALLPAPER_BROWSER=webkit WALLPAPER_WIDTH=390 node tests/visual/run-wallpaper-refresh.mjs
```

Use `WALLPAPER_SOURCE_ROOT=/absolute/frozen/source` for a focused baseline. Full-suite Python and JavaScript results are stored under `verification/`; source integrity and independent popup/print checks are in the adjacent JSON files.
