# Frozen baseline refresh timing diagnostics

These runs investigate newly failing LAYOUT-020 cells from implementation review. They run the original harness against the detached `12b01b6be30c194af2de60fcb50d47a5ed32780a` worktree. Application files, fixtures, screen recipes, checks, thresholds, and response timing are unchanged. Each capture still starts in a fresh browser context with four workers. Only the selected screen and variant sets vary.

The canonical before raw JSON, generated report, and 388 screenshots are preserved independently. Diagnostics retain separate logs, reports, and raw results; none replace or splice into a canonical matrix. Each diagnostic reports every capture it ran, including cells that did not reproduce the failure.

Pass 1 comparison is `pass-1-refresh-reproduction.md` with complete measurements in `pass-1-refresh-reproduction.json`. Canonical settled geometry details are also retained in `pass-1-refresh-geometry.json`. A reproduced failure means the same screen, variant, and LAYOUT-020 rule crossed the existing FAIL threshold on the frozen baseline. The companion JSON separately records whether settled geometry also matches exactly.

Commands for diagnostic trials 1 and 2 (run twice, each preserved separately):

```sh
cd /tmp/factory-ledger-so-detail-baseline
node tests/visual/run-visual-audit.mjs --concurrency 4 --screens S-01,S-02,S-03,S-04,S-06,S-08,S-11,S-12,S-13,S-16,S-17,S-18,S-55,S-56 --variants 390-light,390-dark,1440-light,1440-dark
```

Trial 3 narrows to remaining unreproduced cells, with S-55 retained to inspect geometry:

```sh
cd /tmp/factory-ledger-so-detail-baseline
node tests/visual/run-visual-audit.mjs --concurrency 4 --screens S-01,S-03,S-04,S-11,S-55 --variants 390-dark,1440-light,1440-dark
```

Trial 4 narrows further:

```sh
cd /tmp/factory-ledger-so-detail-baseline
node tests/visual/run-visual-audit.mjs --concurrency 4 --screens S-04,S-11,S-55 --variants 1440-light,1440-dark
```

Rebuild the pass 1 comparison from the repository root:

```sh
node docs/design/audit/pr-screenshots/feat-so-detail-redesign/summarize-refresh-timing.mjs . docs/design/audit/pr-screenshots/feat-so-detail-redesign pass-1
```

The summary script uses the runner's existing verdict function. It reports every new other-screen LAYOUT-020 failure, every available diagnostic sample for that exact cell, and unreproduced cells explicitly. It does not grant exclusions or suppress changed geometry.

## Final audit qualification

The final 388-capture audit adds 22 other-screen LAYOUT-020 failure cells compared with the canonical baseline. All 22 exact cells reproduce on frozen `12b01b6`, and each has a diagnostic FAIL with identical settled anchor geometry to the final capture. The six diagnostic runs total 158 captures with zero runner errors. All samples, including non-failures, remain in `final-refresh-reproduction.md` and `final-refresh-reproduction.json`.

Twenty of the 22 final cells have identical settled geometry to their canonical baseline counterpart. The other two improve: S-01 at 1440-light changes six 5px movements to zero, and S-04 at 1440-dark changes one 3px movement to zero. Full canonical before/final measurements are in `final-refresh-geometry.json`. The raw canonical matrices still retain all 22 newly failing cells; the documented refresh-timing exclusion is supported by separate baseline evidence.

Trial 5 covers final new cells that earlier diagnostic runs had not proved:

```sh
cd /tmp/factory-ledger-so-detail-baseline
node tests/visual/run-visual-audit.mjs --concurrency 4 --screens S-02,S-02b,S-05,S-10 --variants 390-light,390-dark,1440-light,1440-dark
```

Trial 6 narrows to the final three unproved cells:

```sh
cd /tmp/factory-ledger-so-detail-baseline
node tests/visual/run-visual-audit.mjs --concurrency 4 --screens S-02,S-02b,S-05 --variants 390-dark,1440-light,1440-dark
```

Rebuild the final comparison from all preserved samples:

```sh
node docs/design/audit/pr-screenshots/feat-so-detail-redesign/summarize-refresh-timing.mjs . docs/design/audit/pr-screenshots/feat-so-detail-redesign after final
```
