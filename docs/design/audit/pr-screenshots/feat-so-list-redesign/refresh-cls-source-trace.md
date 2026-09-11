# Refresh CLS source trace

A bounded diagnostic logged the existing PerformanceObserver entries and their source rectangles for S-02 at 1440px in both themes, twice per frozen-before/current-after snapshot. The same 200 ms refresh-only API delay was used. Initial setup, application files, checks, and thresholds were unchanged; only diagnostic source metadata was added.

| Run | Variant | CLS | Verdict | Settled moved anchors |
|---|---|---:|---|---:|
| after-trial-1 | 1440-dark | 0.2579 | FAIL | 0 |
| after-trial-1 | 1440-light | 0.2481 | WARN | 0 |
| after-trial-2 | 1440-dark | 0.2481 | WARN | 0 |
| after-trial-2 | 1440-light | 0.2481 | WARN | 0 |
| before-trial-1 | 1440-dark | 0.228 | WARN | 0 |
| before-trial-1 | 1440-light | 0.228 | WARN | 0 |
| before-trial-2 | 1440-dark | 0.258 | FAIL | 0 |
| before-trial-2 | 1440-light | 0.258 | FAIL | 0 |

The frozen baseline itself reaches **0.258 / FAIL in both desktop themes** in its second trace repetition. The trace identifies existing Finished Goods, Production, Attention, and reference-calendar/header elements during refresh. In the higher-CLS frame sequence, Finished Goods disappears in a separate ~0.020 entry; Production temporarily occupies a taller viewport area; the subsequent Attention/Production return shares a frame with the header/calendar return (~0.065). In the lower sequence, these changes are grouped differently (~0.053 plus a separate small header entry).

This demonstrates existing refresh paint timing as a source of the S-02 threshold crossing. It does not convert a canonical failing cell to a pass: the canonical before/after matrices remain unchanged, with 16 added and 16 removed LAYOUT-020 failures. See [raw source entries and rectangles](refresh-cls-source-trace.json).
