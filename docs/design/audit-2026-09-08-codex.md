# Factory Ledger: browser design audit

Reviewed September 8, 2026 against **FL Design Standards Master v1.0, September 7, 2026**.

**Finding:** The site has useful operational information and several good foundations, but needs changes to quantity presentation, responsive layout, keyboard access, status clarity, and shared controls before it aligns with the standard.

## Scope and interpretation

Inspected the live site at https://cns-factory-ledger.netlify.app/ through the browser. Reviewed Operations, Recent Entries, Activity, Notes, Sales Orders, Expected Receipts, and Supplies; Material Flow, Production Lines, and Traceability; an expanded sales order; new expected-receipt and note dialogs; a product-lot dialog; global SKU search; and a forward trace for an existing honey lot. Inspected screenshots, rendered DOM/accessibility information, and selected computed styles. Checked the initial approximately 850-pixel-wide window, a 1440×900 desktop viewport, and representative screens at 390×844.

No records were saved, marked ready, deleted, uploaded, or otherwise intentionally changed. The original viewport and dashboard were restored. The observed appearance was dark mode.

The PDF is an evaluation reference, not authorization to change the website. Its examples are distinguished from binding requirements. This is an observed-issue inventory, not a claim that every one of the 178 rules has been exhaustively tested. Priorities below are implementation priorities assigned by this review, not replacements for the PDF's importance/type fields. Rules marked as recommendations in the PDF remain recommendations.

## Fix first: correctness, access, and operational clarity

### 1. Correct the quantity/unit mismatch in Recent Entries — High

**Observed:** The September 8 Honey Nut shipment appears as `−1,000 25 lb case` in Recent Entries. Activity shows the matching product and timestamp as `1,000 lb · 40 units`. The Kookies & Kreme shipment likewise appears as `−3,000 25 lb case` versus `3,000 lb · 120 units`. This is a confirmed inconsistency between displayed quantities and units; the underlying storage error, if any, was not investigated.

**Fix:** Format the base weight and converted case count independently, for example `Shipped 40 cases · 1,000 lb`. Never append a case-size unit to an unconverted weight. Audit the same formatter in packing entries and Supplies, where packaging units also appear beside large on-hand values.

**Rules:** INPUT-008, DATA-005, LAYOUT-002. **Acceptance:** The same event has equivalent quantities and explicit units in Recent Entries, Activity, product history, and exports.

### 2. Stop the desktop header from collapsing search — High

**Observed:** At the initial window width, the global search input measured only **26 px wide**. Its placeholder disappeared visually, and Refresh extended past the right edge. The three-month calendar retained much of the header width.

**Fix:** Give search a usable minimum width; move or collapse the calendar before sacrificing search, refresh, and freshness information. Reflow controls without page-level horizontal overflow.

**Rules:** LAYOUT-003, LAYOUT-004, LAYOUT-013, SEARCH-001, INPUT-004. **Acceptance:** Search, refresh, and last-updated remain legible and reachable at full, half, and third desktop widths.

### 3. Replace the phone's horizontal section strip — High

**Observed:** At 390 px, the seven dashboard sections scroll horizontally; later destinations are offscreen. Top-level pages are behind a hamburger. There is no persistent bottom navigation for frequent floor destinations.

**Fix:** Define a consistent primary-navigation hierarchy. Use the standard's four-to-five prominent mobile destinations with an appropriate More destination; keep desktop/mobile labels and ordering aligned. Do not use a horizontally scrolling local tab strip to hold seven app areas.

**Rules:** NAV-004, NAV-008, LAYOUT-003, OTHER-005. **Acceptance:** Frequent destinations are recognizable and reachable without discovering offscreen tabs.

### 4. Restore a task-first phone home screen — High recommendation

**Observed:** Header, three miniature calendars, search, navigation, and seven attention tiles occupy the phone's opening screen. Today So Far is below the initial viewport. No direct Receive, Pack, Ship, or Log Coconut actions were visible in the reviewed dashboard.

**Fix:** Put the role's essential status and two-to-four frequent actions first. Reduce the calendar to a compact date control or secondary view; condense zero-count attention categories. Confirm which entry workflows this website is intended to expose before adding actions.

**Rules:** NAV-001, LAYOUT-001, LAYOUT-004, TOUCH-001, OTHER-001. **Acceptance:** A floor user can see today's status and start their authorized frequent task from the opening screen.

### 5. Increase undersized interactive targets — High

**Observed:** Sales-order expand buttons measured **22×22 px**; adjacent row checkboxes **20×20 px**, with no visible text label. Trace recent-lot chips measured **21 px high**. These are far below the standard's 44×44 target requirement.

**Fix:** Expand the actual hit area, not just the icon. Separate expand from readiness mutation and provide equivalent touch, keyboard, and pointer behavior.

**Rules:** TOUCH-003, LAYOUT-012. **Acceptance:** Every target has a minimum 44×44 logical hit area and does not overlap another target.

### 6. Repair keyboard access to global search results — High

**Observed:** Searching `70060` produced a result rendered as generic content. Arrow Down then Enter did not select/open it; clicking the same result opened the product-lot dialog.

**Fix:** Use a keyboard-operable search combobox/listbox or real result links. Provide active-option state, arrow navigation, Enter selection, Escape dismissal, and a clear accessible name.

**Rules:** ACCESS-003, ACTION-011, SEARCH-004. **Acceptance:** The tested SKU can be searched and opened with the keyboard alone.

### 7. Make recent-lot shortcuts semantic and distinguish duplicate codes — High

**Observed:** Traceability's recent lots are `span` elements without role or tabindex, with product names only in title tooltips. Codes such as `SEP 03 2026` and `AUG 31 2026` repeat for different products. Some chips were partially hidden at the bottom of their region.

**Fix:** Use buttons or links with visible product context, complete accessible names, and 44-pixel targets. Show an explicit overflow count/reveal control. Do not rely on hover to distinguish two lots on phones.

**Rules:** ACCESS-003, DATA-004, NAV-012, TOUCH-003. **Acceptance:** Keyboard and touch users can distinguish and select each repeated lot code without guessing.

### 8. Label icon controls and readiness checkboxes — High

**Observed:** Sales-order row checkboxes have no per-record accessible name; the visible Factory Ready column heading is blank. Trace zoom buttons expose only `+` and `−`, without descriptive labels or tooltips. Notes exposes glyph-only edit/remove controls in the accessibility tree.

**Fix:** Name controls by action and record, e.g. `Mark SO-… factory ready`, `Zoom in`, and `Edit note: …`. Show a visible readiness heading and conventional tooltips. Check the Notes controls' names and deletion behavior before choosing the final remove label.

**Rules:** ACCESS-010, ACTION-005, INPUT-002, DATA-005. **Acceptance:** Controls make sense when announced individually without surrounding visual context.

### 9. Gate saving an incomplete expected receipt — High

**Observed:** Save is enabled when Product, Supplier, and Expected qty are empty. The review did not submit the form, so server rejection behavior is unknown.

**Fix:** Mark required inputs explicitly, validate them reliably, and keep the commit action unavailable until the necessary values are valid. Explain missing inputs inline.

**Rules:** INPUT-019, INPUT-007. **Acceptance:** An empty or invalid receipt cannot be submitted; the user can identify what remains to be entered.

### 10. Resolve ambiguous READY/BLOCKED language — High

**Observed:** Some sales orders show `✓ READY · floor` and `BLOCKED` simultaneously. Expanded detail says `Computed view only; shipping is not gated by this status.`

**Fix:** Separate the concepts visibly: `Factory preparation: Ready` and `Dispatch checks: Issues found`, with specific remediation. Use `Blocked` only if it genuinely describes the operational restriction, or make its advisory scope unmistakable. Do not change shipping policy merely to fit a visual label.

**Rules:** NAV-003, FEEDBACK-008, ACCESS-004, OTHER-004. **Acceptance:** Users can explain whether shipping is allowed and what each status means without opening a technical explanation.

### 11. Finish loading the dashboard's dispatch attention count — High

**Observed:** Dispatch blocked remained `— / not loaded yet`, including after Sales Orders loaded and I returned to Operations. Other attention counts were populated.

**Fix:** Resolve this count from the same data as the order view. Distinguish loading, unavailable, failed, zero, and a real count. Provide a retry path for failure.

**Rules:** NAV-001, FEEDBACK-001, FEEDBACK-003, NOTIFY-011. **Acceptance:** A settled dashboard shows a verified count or an actionable unavailable/error state rather than an indefinite placeholder.

### 12. Replace the unexplained health score with actionable information — High

**Observed:** The header shows an amber `80`. Its tooltip explains a health score and lists raw identifiers such as `ship_missing_shipment_lines` and `lots_missing_supplier_lot_code`; the visible number has no label or repair path.

**Fix:** Label it `Data health: 80/100`, or preferably show a concise issue summary. Make it a semantic control opening readable issues and affected records. A score should not resemble an unhandled-item count.

**Rules:** NOTIFY-010, NOTIFY-011, ACCESS-004, LAYOUT-004. **Acceptance:** Users can understand the meaning, affected data, and next action without hovering over an unlabeled number.

### 13. Fix the Trace button contrast — High

**Observed:** After a trace, the Trace button computed to white 13-pixel text on `rgb(96,165,250)`. That pair has approximately **2.54:1** contrast, below the required 4.5:1 for body-sized text.

**Fix:** Use a darker accent fill or suitable dark label color across the button's normal, hover, and focus states. The regular blue buttons measured with a different fill are not covered by this specific failure.

**Rules:** ACCESS-008, ACTION-006. **Acceptance:** All enabled Trace button states meet the specified contrast threshold.

## Make layouts and controls consistent

### 14. Increase operational type sizes — Medium; threshold needs standard finalization

**Observed:** Sales-order headers measured 11 px; several expanded line-item headers 10 px; trace recent identifiers 11 px. Miniature calendar dates are visually difficult to read, especially on phones.

**Fix:** Establish and apply the shared type scale, using larger operational text and identifiers. The PDF proposes 14–16 px desktop body, 17 px mobile body, and no operational text below 12 px, but explicitly leaves final values unresolved.

**Rules:** ACCESS-005, ACCESS-006. **Acceptance:** Adopt final minimums, then verify on real floor devices rather than declaring the provisional numbers already binding.

### 15. Simplify and clarify the persistent calendar — Medium

**Observed:** Three miniature months appear throughout the app, including Traceability and Production Lines. They consume substantial header space and overflow horizontally on phones. The calendar's relation to page filters is unclear; Material Flow also has a separate Period control.

**Fix:** State what the calendar affects. Use a compact date/range control when needed and move the three-month reference calendar behind an explicit reveal action.

**Rules:** LAYOUT-001, LAYOUT-009, LAYOUT-013, ACTION-011. **Acceptance:** Date scope is explicit and calendar chrome does not displace the primary task.

### 16. Give sales-order rows room to remain readable — High on narrow layouts

**Observed:** At the initial width, SO numbers wrapped across three lines, customer labels wrapped heavily, and right-hand dispatch/quantity columns were outside the visible region. Expanded notes stretched across the wide table.

**Fix:** Prioritize identifier, customer, due date, and dispatch summary; move secondary information into detail. Use a compact record layout on phones, preserve distinctive identifiers, and bound edit-field width.

**Rules:** DATA-003, DATA-004, LAYOUT-003, LAYOUT-019, INPUT-004. **Acceptance:** Record identity and the next decision are visible without horizontal page scrolling; rows stay within the standard's two-line anatomy where applicable.

### 17. Provide discoverable table sorting and column resizing — Medium recommendation

**Observed:** Reviewed order, activity, and inventory headers show no active-sort indicator or discoverable resize handles. Sales-order headers had no `aria-sort` state.

**Fix:** Implement a shared sortable table header with reversible direction and announced state, appropriate defaults, and desktop resizing. Use a sort control for compact layouts.

**Rules:** DATA-008, DATA-009. **Acceptance:** Users can sort in both directions, identify the active sort, and widen important columns. Absence of an indicator is confirmed; unexposed implementation details were not audited.

### 18. Reduce competing primary actions — Medium

**Observed:** Sales Orders gives Export CSV, Export Matrix, and Refresh the same filled blue prominence, in addition to global Refresh. Expected Receipts similarly gives New and Refresh filled treatment.

**Fix:** Choose the actual primary action for the view. Make refresh and exports secondary, with related exports grouped if useful.

**Rules:** ACTION-003, ACTION-006, LAYOUT-021. **Acceptance:** The next meaningful action is apparent without several equally prominent blue buttons competing.

### 19. Use explicit action labels and signal dialogs — Medium

**Observed:** Notes uses `+ New`, the note dialog says `New Item`, and forms use `Save`. Expected-receipt creation opens a further-input dialog without an ellipsis/equivalent cue.

**Fix:** Prefer `New Note…`, an appropriate category-aware title, `Create Expected Receipt`, and `Save Factory Ready Note`. Keep labels concise but specific.

**Rules:** ACTION-005, ACTION-009, NAV-003. **Acceptance:** A button's effect and need for further input are predictable before selection.

### 20. Make mobile form commitment reachable — Medium

**Observed:** The expected-receipt dialog correctly stacks fields on phones, but Save is a small right-aligned button after the form, rather than a full-width bottom-anchored primary action. A long upload explanation precedes manual entry.

**Fix:** Provide a safe-area-aware action bar with an appropriately prominent primary action. Keep cancellation clear and let frequent manual entry start without navigating past secondary guidance.

**Rules:** TOUCH-001, TOUCH-004, LAYOUT-001. **Acceptance:** Commit remains reachable with the keyboard open and does not cover the last field.

### 21. Add persistent labels and one-action clearing to search/filter fields — Medium

**Observed:** Global search, customer filtering, Supplies search, and trace search rely on placeholders. Once a query is entered, its scope label disappears. No visible dedicated clear control appeared for the tested populated global and trace text fields.

**Fix:** Add compact persistent visible labels, meaningful accessible names, and clear controls for populated fields.

**Rules:** INPUT-002, INPUT-012, SEARCH-003. **Acceptance:** Scope remains understandable while typing and a query can be cleared in one action on phone and desktop.

### 22. Make global search available on every page — Medium

**Observed:** Material Flow and Production Lines have no visible global search. Traceability has a lot-specific search rather than the dashboard's global search.

**Fix:** Put the shared global search in the shared app shell, with the local trace search clearly scoped.

**Rules:** SEARCH-001, SEARCH-002. **Acceptance:** Users can start a cross-record search from any top-level page without returning to Dashboard.

### 23. Improve known-entity selection in forms — Medium

**Observed:** The note form asks for free-text `Entity Name / ID` even after choosing a known entity category. Receipt Supplier is a long plain select and exposes similar entries such as A1 Baker/A1 Bakery/A1 Bakery Supply.

**Fix:** Use searchable entity selectors with distinguishing identifiers. Review suspected supplier aliases separately; do not automatically merge similarly named suppliers based on this audit.

**Rules:** INPUT-011, INPUT-014, DATA-004. **Acceptance:** Users choose the intended existing entity from clearly differentiated results; arbitrary text cannot silently create or mis-link a record.

### 24. Replace small view-choice dropdowns where the standard calls for visible choices — Medium recommendation

**Observed:** Expected Receipts hides Open/Closed/Cancelled/All inside a dropdown; Activity's two date-basis choices are also a dropdown.

**Fix:** Use visible local segmented choices when they fit. The much longer sales-order status list is a reasonable dropdown candidate and should not be indiscriminately converted into many tabs.

**Rules:** NAV-007, ACTION-012. **Acceptance:** Small frequent view switches take one action and expose their current state.

### 25. Add a path beyond the 20 Recent Entries — Medium

**Observed:** The page says `Showing 20 most recently entered ledger events`, but exposes no View All, Load More, or history link. Activity's Shipping and Receiving sections do have explicit `Show all (96 more)` controls, which is a better pattern.

**Fix:** Link to full history or paginate with a remaining count; preserve the user's scope.

**Rules:** NAV-012, SEARCH-007. **Acceptance:** The subset is explicit and the rest of the history is reachable from this view.

### 26. Make Recent Entries and Notes scannable — Medium

**Observed:** A make event expands its full ingredient list inside the feed. A long inventory-cleanup note occupies a large portion of the desktop screen. Neither is a compact decision-oriented summary.

**Fix:** Use a consistent record summary with short identity, event, quantity, date, and status fields; expose long notes and ingredient detail on expansion. Offer a local filter for long content.

**Rules:** DATA-003, DATA-007, LAYOUT-001, LAYOUT-019, SEARCH-007. **Acceptance:** Users scan multiple records quickly, with complete details one action away.

### 27. Standardize date/time presentation — Medium

**Observed:** Sales Orders uses `06/29/26`; Activity uses `2026-09-08 01:05 PM ET`; Recent Entries uses `Sep 8, 2026, 2:34 PM EDT`; Material Flow's update time omits its time zone.

**Fix:** Define consistent operational date and time formats with clear zone handling. Keep literal supplier/lot identifiers unchanged even when they resemble dates.

**Rules:** DATA-012. **Acceptance:** Equivalent dates/timestamps use the shared format across screens; lot codes are not reformatted as dates.

### 28. Translate implementation vocabulary into operational language — Medium

**Observed:** UI includes `adjusted_in`, `pack_output`, `finished`, `Entity Name / ID`, `Effective Remaining`, and explanatory copy about values being `never stored`. The health tooltip exposes internal field identifiers.

**Fix:** Map technical values to plain labels such as `Inventory increased`, `Packed lot`, and `Remaining to ship`; retain technical detail only where it helps an audit or support task. Simplify receipt matching guidance while preserving its business meaning.

**Rules:** OTHER-004, ACCESS-004. **Acceptance:** Staff can explain labels without understanding database structures.

### 29. Give primary records a consistent linkable detail path — Medium

**Observed:** Opening the product-lot dialog leaves the URL at `/`; expanding a sales order also leaves it unchanged. The product dialog lists depleted lots but shows no obvious per-lot trace action. In contrast, running a trace correctly produced `?lot=…&direction=forward`.

**Fix:** Apply record-specific links to product, lot, order, and history detail views, and expose a direct trace action beside relevant lots. Preserve the working trace-link pattern.

**Rules:** SEARCH-008, NAV-009, NAV-003. **Acceptance:** Copying a record URL into a fresh tab opens the same record; users can move from a lot to its trace directly. Alternate unexposed deep-link routes remain unverified.

### 30. Use consistent names, icons, and control styling — Medium

**Observed:** Navigation says `Material Flow` while its heading says `☰ Product Flow — Sankey`; `Production Lines` opens `Production Process Flow`. Screens mix emoji, text glyphs, and other icon treatments, including a hamburger-like symbol in a page title and a floppy disk on Download Text Report.

**Fix:** Keep destination names recognizable across navigation and headings. Use one vector icon family and conventional action symbols, with text for domain-specific actions.

**Rules:** LAYOUT-002, NAV-003, ICON-001, ICON-002, ICON-004, ICON-005. **Acceptance:** The same concept has the same name and visual treatment throughout the app.

## Make analytical views usable

### 31. Fix overlapping and clipped Material Flow labels — High

**Observed:** At 1440×900, several finished-goods/customer labels overlap, and lower ingredient labels reach the chart edge. At 390 px, headings run together and most of the graph is outside the visible screen.

**Fix:** Use collision-aware labels, enough chart height, progressive subsets, and readable spacing. Provide a compact mobile list or staged flow view rather than squeezing the desktop graph. Expose complete labels in one step.

**Rules:** CHART-002, CHART-006, DATA-004, LAYOUT-003. **Acceptance:** Default charts have no colliding labels; mobile users can identify and inspect a complete path.

### 32. Explain what Material Flow means and provide a relationship table — Medium

**Observed:** A color legend and `All volumes shown in pounds` exist, but no plain-language takeaway, explanation of flow-width meaning, or adjacent relationship table appears. The native accessibility view exposes some node labels/values; it does not establish that source-to-target relationships and edge quantities are understandable without the image.

**Fix:** Add a concise period-specific takeaway, explain link widths and aggregation, and offer an accessible source → destination → pounds table with drill-through. Clarify what `Other` aggregates contain.

**Rules:** CHART-003, CHART-004, CHART-005, CHART-008. **Acceptance:** A user can understand the main result and inspect exact flows using keyboard/screen reader or the table. Do not treat the existing node labels as complete proof of chart accessibility.

### 33. Distinguish zero, unavailable, and not-applicable production values — Medium

**Observed:** Production Lines reports `0 / 4` active, while Produced Today, Avg Yield, and each stage display dashes. The screen shows idle lines but does not explain whether the dashes mean zero activity or unavailable data.

**Fix:** Use `0` for verified zero production, `Not applicable` for yield without a run, and `Unavailable` for missing data. Add a concise summary of today's state.

**Rules:** FEEDBACK-008, CHART-004, NAV-003. **Acceptance:** Zero activity cannot be mistaken for a failed load, and vice versa.

### 34. Provide manual refresh and useful drill-through on Production Lines — Medium

**Observed:** The page says it refreshes every 60 seconds and shows last-run dates, but exposes no manual refresh or link to the referenced run in the reviewed state.

**Fix:** Add a shared refresh control and make last-run or stage detail reachable where the underlying records exist. Preserve a stable layout during updates.

**Rules:** FEEDBACK-007, CHART-008. **Acceptance:** Users can request current data and inspect the run supporting a summary without finding it again elsewhere.

## What already aligns

- Dashboard includes Needs Attention and Today So Far, with actual links from attention items.
- Several standard buttons are already 44 pixels high; the smaller row controls need correction rather than a blanket rewrite of all controls.
- Receipt and note dialogs have headings, Close/Cancel, persistent field labels, and multiline notes. The receipt form stacks into one column on phones.
- Traceability disables Trace when empty, produces a stable lot/direction URL, provides a text audit table, and offers print/text reports.
- Activity exposes counts and Show All controls for hidden Shipping/Receiving rows.
- Statuses often include text and symbols in addition to color. These patterns should be retained.

## Items requiring further verification, not established failures

1. **Transactional safety:** save-in-progress behavior, duplicate suppression, destructive confirmations, reversal/correction paths, and validation after actual input. No live transaction was committed for testing. Rules FEEDBACK-001, ERROR-001–009, INPUT-007.
2. **Draft persistence:** refresh, backgrounding, device rotation, interrupted uploads, and recovery. Rule ERROR-002.
3. **Full accessibility matrix:** 200% browser zoom, user text-size/bold preferences, light and increased-contrast appearance, real screen-reader navigation, focus trapping/return, and all keyboard paths. This audit confirmed specific failures, not every possible one. Rules ACCESS-001–010.
4. **Color/type architecture:** semantic tokens, three appearance variants, shared status-color meanings, type-scale definitions, and prohibited hard-coded values require source inspection. Rules OTHER-010, FEEDBACK-012, ACCESS-005.
5. **Notifications and shared-device privacy:** permission timing, per-user history, deduplication, notification content and preferences were not exercised. Rules NOTIFY-001–012, SEARCH-006, OTHER-002.
6. **Drag/drop and uploads:** feedback, cancellation, undo, keyboard alternatives, and file extraction correctness were not exercised. Rules DRAG-001–011.
7. **Real floor testing:** gloves, glare, poor connectivity, device safe areas, hardware scanners, and Spanish labels need the actual deployment environment. Rules OTHER-006, TOUCH-001–005.
8. **Data integrity versus presentation:** the site itself reports missing shipment/lot/case-size information. The browser review cannot certify the underlying data, supplier aliases, allocation correctness, or the truth of a complete-trace claim.

The source standard itself lists unfinished work on page 71: final text-size limits, the status-color table/token file, per-operation stall timeouts, and standalone performance rules. Those decisions must be finished before an exact compliance sign-off is possible; this audit does not invent their final values.

## Suggested implementation order

1. Correct quantity/unit presentation, readiness wording, unresolved attention state, form gating, and the measured contrast failure.
2. Repair the shared shell and responsive navigation; fix search, target sizes, labels, and keyboard access.
3. Standardize tables, forms, dates, record links, and plain-language copy.
4. Redesign the Material Flow presentation and clarify production summaries.
5. Complete the untested transaction, accessibility, and real-device checks against the finalized standard.
