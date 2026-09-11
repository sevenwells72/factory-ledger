# Sales Order detail redesign — audit evidence

Frozen baseline: `12b01b6`. Final application snapshot: `c0eed1c`. The specification was committed first as `26d2811`. Exactly two implementation review passes were completed; all corrections occurred between them.

Both canonical full audits contain **388 captures** with **zero runner errors**, fresh browser contexts, and four workers. They include 1440px and 390px in light/dark plus the existing zoom variants. No rule, threshold, or measurement was relaxed.

Counts below are failing screen/variant captures. List: S-24–S-29 (36 captures). Detail: S-30–S-40 (54 captures, excluding native dialogs S-35/S-38). Other: 298 captures.

| Rule | List before | List after | Detail before | Detail after | Other before | Other after |
|---|---:|---:|---:|---:|---:|---:|
| STATUS-002 | 0 | 0 | 12 | 0 | 0 | 0 |
| STATUS-004 | 0 | 0 | 42 | 0 | 32 | 32 |
| STATUS-005 | 0 | 0 | 42 | 0 | 102 | 102 |
| STATUS-006 | 0 | 0 | 30 | 0 | 28 | 28 |
| STATUS-007 | 0 | 0 | 30 | 0 | 0 | 0 |
| STATUS-008 | 0 | 0 | 14 | 0 | 32 | 32 |
| STATUS-010 | 0 | 0 | 36 | 0 | 38 | 38 |
| STATUS-011 | 0 | 0 | 0 | 0 | 16 | 16 |

## All-rule comparison

| Scope | Before failing cells | After failing cells |
|---|---:|---:|
| list | 0 | 0 |
| detail | 212 | 0 |
| other | 421 | 431 |

**No new failure occurs outside LAYOUT-020.** All **22/22** newly failing cells below also fail on frozen `12b01b6`, qualifying for the stated refresh-timing exclusion. Six separate diagnostic runs retained all 158 samples with zero runner errors; each exact cell has a baseline failure with matching settled geometry. The canonical before/after matrices remain unchanged. See [per-cell reproduction](final-refresh-reproduction.md) and [commands and provenance](refresh-diagnostics.md). Canonical settled geometry matches in 20/22 cells; the remaining two improve to zero moved anchors.

| Screen | Variant | Before | After |
|---|---|---|---|
| S-01 | 1440-light | WARN | FAIL |
| S-01 | 1440-dark | PASS | FAIL |
| S-02 | 390-light | PASS | FAIL |
| S-02 | 1440-dark | PASS | FAIL |
| S-02b | 390-light | PASS | FAIL |
| S-02b | 1440-light | PASS | FAIL |
| S-03 | 390-light | PASS | FAIL |
| S-03 | 390-dark | PASS | FAIL |
| S-04 | 390-dark | PASS | FAIL |
| S-04 | 1440-dark | WARN | FAIL |
| S-05 | 390-dark | PASS | FAIL |
| S-06 | 390-light | PASS | FAIL |
| S-10 | 390-light | PASS | FAIL |
| S-11 | 390-dark | WARN | FAIL |
| S-12 | 1440-dark | WARN | FAIL |
| S-13 | 1440-dark | PASS | FAIL |
| S-16 | 390-light | PASS | FAIL |
| S-16 | 390-dark | PASS | FAIL |
| S-17 | 390-dark | WARN | FAIL |
| S-18 | 390-light | WARN | FAIL |
| S-55 | 1440-light | WARN | FAIL |
| S-55 | 1440-dark | WARN | FAIL |

## Test and interaction results

| Check | Final result |
|---|---:|
| Python suite with local test database | 848 passed |
| JavaScript unit tests | 64 passed |
| Detail interactions, four viewport/theme variants | 36 passed |
| Shared exit dialog checks | 70 passed |
| List layout, widths/themes/states | 26 cases passed |
| List interactions | 1440px and 390px passed |

The first Python run was 847 passed/1 failed. Frozen main reproduces the same stale expected pallet-line dictionary, missing the additive `line_status` field deployed by PR #46. The test now asserts `line_status: pending`; no backend code changed.

Pass one reached zero requested STATUS failures and retained six existing S-33 TOUCH-003 failures. First-review regressions demonstrated off-page Ready-note loss, untracked service quantities rendered as zero, and server decimal warnings bypassing the shared formatter. The single correction window fixed those cases, real related-order URLs, signed fractional pallet differences, and edit-field touch targets. Final targeted and full rendered checks pass.

## Fixture reconciliation and limits

All original fields and values in detail fixtures 101–108 remain unchanged; state/fulfillment/health fields are additive. Fixture 109 previously claimed shipped while copying unshipped detail quantities and line IDs from 101; its effective quantities/identities now agree with the existing open/shipped list record. Closed/cancelled fixtures retain visible cancelled lines. Shipping preview previously received a generic write acknowledgement; it now exercises the existing request/shippable quantities and raw warning messages through the real preview renderer. See the visual README for exact changes.

The backend header-edit endpoint still uses its legacy status gate. The client offers editing for open/unshipped orders and explains server rejection without exposing raw responses. Missing Ready metadata is read from the matching list record; if the bounded 200-order query cannot locate a note, the checkbox change aborts instead of overwriting an unknown note. These backend contracts were not changed.

Human semantic/design approval remains separate from mechanical evidence. Blubber’s approval of the Netlify PR preview is pending; this branch is **NOT DEPLOYED** to production.
