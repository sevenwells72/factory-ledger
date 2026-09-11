# S-02 CLS source diagnosis

Both frozen-before and current-after snapshots received identical 200 ms API latency only during refresh measurement. The observer was instrumented to report entry timing and source nodes/rectangles; no application files, recipes, measurement thresholds, or verdict rules changed. Canonical evidence and the prior latency experiment were preserved.

## Result

The frozen baseline itself changes from CLS 0.228 WARN to 0.258 FAIL across repeats in both desktop themes. This establishes that the extra ~0.03 can occur without the Sales Orders redesign. The recorded movement is existing Operations content and shared header refresh movement.

| Run | Variant | CLS | Entries |
|---|---|---:|---|
| after-trial-1 | 1440-light | 0.2481 | 0.172472 + 0.020033 + 0.053439 + 0.002148 |
| after-trial-1 | 1440-dark | 0.2579 | 0.172472 + 0.020033 + 0.065368 |
| after-trial-2 | 1440-light | 0.2481 | 0.172472 + 0.020033 + 0.053439 + 0.002156 |
| after-trial-2 | 1440-dark | 0.2481 | 0.172472 + 0.020033 + 0.053439 + 0.002156 |
| before-trial-1 | 1440-light | 0.228 | 0.172494 + 0.053218 + 0.002331 |
| before-trial-1 | 1440-dark | 0.228 | 0.172494 + 0.053218 + 0.002331 |
| before-trial-2 | 1440-light | 0.258 | 0.17255 + 0.020033 + 0.065431 |
| before-trial-2 | 1440-dark | 0.258 | 0.17255 + 0.020033 + 0.065431 |

## Sources of the extra ~0.03

Comparing frozen baseline trial 1 versus trial 2:

1. The initial loading paint is effectively identical: approximately 0.1725 CLS. Needs Attention (`main#tab-operations > section.section.attention-strip-section`) moves from y=595 to 417, Production (`section#section-production`) moves from y=812 to 634, Finished Goods (`section#section-finished-goods`) enters the viewport at y=790, and the shared header/reference-calendar panel move left 34.6875px.
2. In the higher-CLS capture, Finished Goods leaves the viewport in a separate entry worth **0.020033**. Production has an intermediate visible height 266px (the lower-CLS run's corresponding previous height was 132px).
3. Restoration is recorded differently. The higher run combines Needs Attention/Production moving down 178px with the header/reference-calendar panel moving right 34.6875px in one **0.065431** entry. The lower run records attention/production restoration and Finished Goods disappearance together at **0.053218**, then header restoration separately at **0.002331**.
4. Thus the higher sequence adds approximately 0.020033 from the separate Finished Goods disappearance and 0.009882 from the different grouping of restoration sources; initial-paint differences contribute only 0.000056. Total difference is approximately 0.029971.

The source trace supports a refresh rendering-order/paint-grouping explanation. CLS depends on the impacted viewport area and greatest movement in each rendered frame, so grouping the same content and header changes into different frames changes the cumulative score. This diagnosis does not make the existing refresh shifts acceptable or alter the canonical failure report; it identifies why before/after classifications can vary without a new application regression.

## Evidence

- [Compact per-entry values, timestamps, selectors, and source rectangles](refresh-cls-source-trace.json).
- Full diagnostic snapshots, reports, logs, and screenshots remain outside the repository at `/var/folders/61/x92cz33j1qb8vftv75f2bymh0000gn/T/factory-ledger-cls-source.74pw10j8`.
