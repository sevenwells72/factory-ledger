# Factory Ledger — Batch C results

September 8, 2026. Implemented locally on `fix/design-audit-3`, based on Batch B commit `9732da0`. **Not pushed, merged, or deployed.** Batch B is also still local; this branch includes it as its parent.

## Findings addressed

| Finding | Rules | Implemented result |
|---|---|---|
| 13 — Trace contrast | ACCESS-008, ACTION-006 | Enabled Trace uses white on #195aea (5.69:1); hover/active use darker #1749b6. Normal/focus fill is shared across light and dark appearances. |
| 14 — Operational type | ACCESS-005, ACCESS-006 | Named caption/body/headline/title/display sizes replace pixel declarations in the reviewed dashboard and analytical pages. Baseline: 16 px desktop body, 17 px phone body, 14 px captions. Phone stage/report layouts reflow around the larger text. These are implementation choices using the proposed standard, not a claim that its unfinished type-size decision is finalized. |
| 17 — Sorting/resizing | DATA-008, DATA-009 | Native sortable header buttons, aria-sort, a compact sort selector and reversible direction; desktop column width sliders. Browser-local preferences survive refreshes. Orders default to earliest ship date; Dispatch Queue retains its priority default. Sorting keeps expanded lots, order lines, supply details and rowspan activity records together. Editable full-order tables retain their existing behavior. |
| 18 — Action hierarchy | ACTION-003, ACTION-006, LAYOUT-021 | Order exports/refresh and receipt refresh use secondary treatment; receipt creation remains prominent. |
| 19 — Action labels | ACTION-005, ACTION-009, NAV-003 | New Note… / New Expected Receipt…; category-aware note creation; Create Expected Receipt / Save Receipt Changes; Save Factory Ready Note. |
| 21 — Search labels/clearing | INPUT-002, INPUT-012, SEARCH-003 | Persistent visible labels and one-action Clear for global, customer, supplies, receipt, trace, recent-entry and note searches. Trace Clear also invalidates the previous selection. |
| 23 — Known entities | INPUT-011, INPUT-014, DATA-004 | Notes require choosing an existing search result when adding/changing a reference; stale/unconfirmed text is rejected before save. Existing references remain editable without forced migration. Supplier filtering shows exact names and IDs. No supplier merging. |
| 24 — Visible view choices | NAV-007, ACTION-012 | Receipt status and Activity date basis use labelled button groups with aria-pressed; existing select-backed request behavior remains intact. |
| 25 — Beyond Recent Entries | NAV-012, SEARCH-007 | New read-only Ledger History page uses the existing date/type-filtered transaction-history API. Recent Entries links to it, with per-transaction links from expanded entries. History shows its scope, corrections and the 1,000-record API limit. |
| 26 — Scannable content | DATA-003, DATA-007, LAYOUT-001, LAYOUT-019, SEARCH-007 | Recent entries show a concise first-line summary and line count, with complete ledger lines/history behind expansion. Notes show short previews and expandable full text. Local text filters search full loaded content. |
| 27 — Dates/times | DATA-012 | Calendar dates use Mon D, YYYY; timestamp displays use plant time with a zone where available. Sort keys handle 12-hour times correctly. Literal lot codes stay unchanged. Fixed the existing production helper that appended Z to already-offset timestamps and produced invalid dates. |
| 29 — Record paths | SEARCH-008, NAV-009, NAV-003 | Product, lot and order openings update shareable record URLs; expanded orders link to the full record on reload. Active and depleted product lots offer direct Trace links carrying product identity. History links preserve day/type/transaction selection. |
| 30 — Naming/controls | LAYOUT-002, NAV-003, ICON-001–005 | Material Flow, Production Lines and Traceability headings match navigation. Core menu/close/theme/order-expansion/sort controls use a shared vector treatment. Note edit/delete and report actions use explicit text instead of decorative emoji. Semantic status markers remain. |
| 32 — Flow explanation | CHART-003–005, CHART-008 | Period-specific largest displayed shipment relationship; explanation of band widths, separate stage aggregation, 50 lb threshold and truncation. Accessible relationship table with exact contributor expansion and product-search drill-through. Other top-count buckets and unclassified Other Production are distinguished. |
| 34 — Production refresh/detail | FEEDBACK-007, CHART-008 | Manual refresh with in-progress state; available last-run IDs link to the corresponding history record. Mobile stages use a readable vertical sequence. |

## Verification

- **46 Node tests:** existing Batch A/intake tests plus six Batch C regressions for time zones, calendar dates, literal record identity, Other-flow reconciliation, reference validation and 12-hour sort ordering.
- **15 Python tests:** Recent Entries and Activity contracts, against the isolated local test database.
- Modified standalone and inline JavaScript syntax checks; `git diff --check`.
- Browser checks used a **read-only local fixture server**. Non-GET API requests were blocked. No production records were written.
- Order customer sorting reversed correctly; the chosen sort survived reload; Customer widened to 400 px. Expanded order lines and inventory lot breakdowns remained with their parent records.
- Keyboard global search opened the product dialog; product/order URLs reopened the same records. Product-to-trace links retained product IDs, including depleted lots.
- Notes text filtering/clearing worked. An unconfirmed note reference was blocked; selecting the known product produced its ID and visible confirmation. Supplier filtering narrowed to the matching named/numbered supplier. Receipt Save remained gated on an incomplete form.
- Labelled global search, populated with a Clear button, measured 291 / 738 / 528 px at viewports 390 / 850 / 1440; page widths remained within those viewports.
- Material Flow at 1440: **28 labels, zero measured overlaps, zero clipped labels**. At 390: 30 compact flow cards and relationship table fit within the viewport.
- Populated Trace at 390 stayed within the viewport; after tracing, the eight fixture report rows also fit. Normal enabled contrast was **5.69:1** in light and dark modes; hover/active uses the darker shared token.
- Production run links resolved to valid plant dates and opened the selected transaction; manual refresh worked; phone production cards fit after reflow.
- No console errors in the final browser validation tab.

## Limits and preserved contracts

- This implements the remaining **15 audit findings**, not an exhaustive sign-off on all 178 design-standard rules. Real floor-device/glove/glare testing, actual OS keyboard/screen-reader testing, and the standard’s pending final decisions remain outstanding.
- Notes remain string-valued annotations under the existing API, not new database foreign-key relationships. Product selections use existing IDs; lot references include product context. Other callers and existing stored notes are not migrated.
- History uses the existing API: date-filtered current transactions with correction chains, capped at 1,000 per date/type. It is not an unlimited paginated copy of the append-only Recent Entries feed. Missing exact unit metadata is disclosed rather than inferred for non-weight products.
- Sorting applies to loaded rows; it does not add backend pagination or remote sorting. Column preferences are local to this browser, not synchronized per account.
- Material Flow preserves the existing aggregation and data-fetch limits. It does not certify ledger completeness or establish mass balance across stages.
- Batch A quantity/readiness/intake safeguards and Batch B shell behavior remain in place. No backend, schema, migration, or GPT instruction changes.

## Cache and logs

Dashboard JS **55**, dashboard CSS **38**, shell JS/CSS **2**, design controls **1**, history JS/page **1**. Navigation versions: Material Flow **3**, Production Lines **4**, Traceability **4**. Project change log, Factory Ledger regression log, and resolved global change log updated.
