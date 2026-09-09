# Batch B results — September 8, 2026

Implemented locally on `fix/design-audit-2`, based on merged Batch A (`32b96d2`). Not pushed or deployed. Working checkout: `/Users/cns/Documents/Codex/2026-09-08/ca/work/design-audit-2`.

| Finding | Standard rules | Result and evidence |
|---|---|---|
| 2 | LAYOUT-003, LAYOUT-004, LAYOUT-013, SEARCH-001, INPUT-004 | Header search retains usable width; calendar collapses behind an explicit reveal. Search measured 678 / 802 / 448 / 358 px at viewport widths 1440 / 850 / 480 / 390. |
| 3 | NAV-004, NAV-008, LAYOUT-003, OTHER-005 | Five mobile destinations: Operations, Activity, Sales Orders, Supplies, More. More reveals the remaining sections and top-level pages. No horizontally scrolling section strip. |
| 4 | NAV-001, LAYOUT-001, LAYOUT-004, TOUCH-001, OTHER-001 | Today So Far leads the Operations DOM; mobile quick actions appear before totals and use existing expected-receipt, supply-request and note editors. Clear attention categories are condensed. |
| 5 | TOUCH-003, LAYOUT-012 | Order expand and readiness label targets are separate 44×44 controls; trace recent-lot buttons also measure at least 44 px high. |
| 6 | ACCESS-003, ACTION-011, SEARCH-004 | Shared combobox supports Arrow Up/Down, Enter, Escape and announced result counts; obsolete responses cannot replace newer searches. Browser Arrow Down + Enter opened product detail with focus in its dialog. |
| 7 | ACCESS-003, DATA-004, NAV-012, TOUCH-003 | Recent-lot shortcuts are real buttons showing lot code and product together. Six appear initially, with an exact count and reveal control for the remaining indexed recent lots. Keyboard activation selects the product-specific lot. |
| 15 | LAYOUT-001, LAYOUT-009, LAYOUT-013, ACTION-011 | Every top-level page has a compact Reference calendar reveal that explicitly says it does not filter the page. The revealed calendar fits a 390 px viewport. |
| 16 | DATA-003, DATA-004, LAYOUT-003, LAYOUT-019, INPUT-004 | Order lists become cards at 1100 px and below, prioritizing identifier, customer, due date, dispatch state and remaining quantity. Secondary data remains available in detail. Ready-note edits are bounded. |
| 20 | TOUCH-001, TOUCH-004, LAYOUT-001 | Manual receipt Save and intake approval use mobile sticky action areas with safe-area spacing. Save measured 48 px high and remained within a simulated 390×500 viewport; mobile navigation hides behind dialogs. |
| 22 | SEARCH-001, SEARCH-002 | The same global search is available on Dashboard, Material Flow, Production Lines and Traceability. A Material Flow search opened the selected dashboard detail. Traceability separately labels its local lot search. |
| 31 | CHART-002, CHART-006, DATA-004, LAYOUT-003 | Desktop Material Flow assigns explicit semantic columns and label spacing. All 28 default fixture labels had zero pairwise overlaps and zero clipping at 1440 px. Selection reveals full names/connected flows. At phone widths the chart becomes readable source-to-destination cards. |

## Validation

- Browser checks used the read-only localhost fixture preview at port 8880. Operational API calls were intercepted and writes rejected.
- Header measured at 1440, 850, 480 and 390 px; no page-level horizontal overflow. Order cards measured 356 px within the 390 px viewport, with 44×44 readiness and expand targets at distinct x positions.
- Shared search tested with keyboard selection and cross-page navigation. Lot shortcuts tested with Enter. Reference calendar tested expanded on a phone viewport.
- Receipt Save checked in a 390×500 reduced viewport. This simulates reduced available space; it is not a physical iPhone/Android keyboard test.
- Material Flow checked as a desktop chart and phone flow list; desktop labels measured for overlap and clipping after the layout completed.
- **15 Python tests passed** (`test_recent_ledger.py`, `test_dashboard_b2.py`) against the isolated local test database; existing deprecation warnings remain.
- **40 Node tests passed** (`test_batch_a_ui.js`, `test_er_intake_logic.js`). Batch A semantics remain covered.
- Dashboard/shared scripts and modified inline scripts passed syntax checks. `git diff --check` passed.

## Scope and deployment

New shared assets: `shell.js?v=1`, `shell-layout.css?v=1`. Dashboard script v54 and CSS URL v37; inline page URLs: Material Flow v2, Production Lines v3, Traceability v3. Dashboard CSS itself is unchanged; the shared layout sheet supplies the new overrides.

Batch A deployment was verified separately: production serves dashboard JS v53/CSS v36 and the API returns `request_unit`. This is recorded as regression-guard row 121.

Batch B changes only dashboard assets, tests/version assertions and documentation. No backend logic, database records, migrations, or shipping rules were changed. Existing product, lot and order detail functions handle shared-search selections; existing editors handle quick actions.

Batch C remains deferred: findings 13, 14, 17, 18, 19, 21, 23, 24, 25, 26, 27, 29, 30, 32 and 34. This report does not claim that all design-standard rules or physical-device accessibility behaviors have been exhaustively verified.
