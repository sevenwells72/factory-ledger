# IMPROVEMENTS — Master List

**Date:** 2026-09-07 · **Scope:** `dashboard/` including `dashboard/scheduler/`
**Standard:** [FL-Design-Standards-MASTER.md](../FL-Design-Standards-MASTER.md) — 178 rules
**Screens:** [00-screen-inventory.md](00-screen-inventory.md) — 91 screens (S-01…S-91)
**Source findings:** [01](01-nav-layout.md) · [02](02-actions-input-touch.md) · [03](03-feedback-error-notify.md) · [04](04-search-data-drag-chart.md) · [05](05-access-icon-other.md)
**Rendered verification:** [06-browser-check.md](06-browser-check.md) — 384 captures across 91 screens, 2026-09-08
**Systemic clusters:** [07-systemic-clusters.md](07-systemic-clusters.md) — the 648 TOUCH-003 / ACCESS-008 / overflow / fixed-bar failures grouped by root cause, ranked by failures-resolved-per-change
**Status:** Every FAIL and PARTIAL from the five group files, deduplicated across screens into one improvement each. **No application code was modified.**

---

## How this list was built

Each of the five group files records findings **per rule, per screen**. The same underlying defect frequently appears under several rules on many screens — the missing horizontal-scroll wrapper, for example, surfaces as LAYOUT-003, LAYOUT-011, LAYOUT-013, DATA-004, and ACCESS-001 across fourteen screens. This file collapses each defect into **one improvement**, listing every rule it satisfies and every screen it touches.

**Importance** is the highest importance among the rules an improvement resolves, taken from the standard.
**Effort** is a rough implementation size: **S** ≈ under a day, **M** ≈ one to three days, **L** ≈ a week or more.
**Screens** is the count of inventoried screens the change affects.

Sixty-seven improvements (IMP-001…IMP-067). Thirty-two resolve at least one Critical rule; twenty-two are High, twelve Medium, one Low. Eight are marked **DONE**.

---

## Proposed sequence

Ordered as the brief specifies: **Critical rules first, then by number of screens affected.** Within each band, one-line fixes that resolve a Critical rule are placed first — they are the cheapest risk reduction available.

### Band 0 — Critical, and small (do these first)

These six resolve Critical rules and are each an hour to a day of work.

| # | Improvement | Effort | Screens | Status | Commit |
|---|---|---|---|---|---|
| **IMP-001** | Fix `var(--bg-card, #fff)` — near-white text on white at Arturo's decision point | S | 1 | **DONE** | `4a96cc7` |
| **IMP-002** | Fix the sticky-offset overlap that hides the header on every phone | S | 5+ | **DONE** | `45be3eb` |
| **IMP-003** | Add horizontal-scroll wrappers to the six unwrapped tables | S | 14 | **DONE** | `4f34036` |
| **IMP-004** | Add a press state and disable every commit button synchronously | S | ~60 | **DONE** | `6083037` |
| **IMP-005** | Add a fetch timeout with a stall message and Retry | S | ~40 | **DONE** | `c451682` |
| **IMP-006** | Constrain `#er-qty` to numeric — `parseFloat("12O")` currently commits 12 | S | 1 | **DONE** | `44c5eb5` |

### Band 1 — Critical

| # | Improvement | Effort | Screens |
|---|---|---|---|
| **IMP-007** | Stop full-container re-renders on background refresh and per-row writes | M | 12 |
| **IMP-008** | Preserve drafts; stop blanking the view on a failed refresh | M | 20 |
| **IMP-009** | Raise operational text above the 12 px floor; add a mobile body size | M | ~50 |
| **IMP-010** | Fix the light-theme contrast failures and `--text-dimmed` | S | ~30 |
| **IMP-011** | Minimum 44 × 44 pt hit targets and a focus ring on every control | M | ~65 |
| **IMP-012** | Separate the product-family palette from the status palette; tokenise the hard-coded colours | M | ~40 |
| **IMP-013** | Introduce semantic button roles (primary / secondary / danger) | M | ~30 |
| **IMP-014** | Replace the native `confirm()`/`alert()` dialogs | M | 6 |
| **IMP-015** | Add a non-colour carrier to five status cues | S | 12 |
| **IMP-016** | Add reversal paths and undo for the four one-way doors | M | 7 |
| **IMP-017** | Mark required fields and gate every commit | S | 6 |
| **IMP-018** | Replace `#note-entity-id` free text with a type-ahead | M | 1 |
| **IMP-019** | Scope the Supplies search, or say that it is scoped | S | 2 |
| **IMP-020** | Fix the Traceability lot index and state its real scope | M | 2 |
| **IMP-021** | Add a `title` to truncated names; middle-truncate lot codes | S | 4 |
| **IMP-022** | Add a currency symbol and unit to the order-line edit inputs | S | 2 |
| **IMP-023** | Fix the Supplies "Incoming" column unit mismatch | S | 1 |
| **IMP-024** | Add inline cross-field validation to the allocation quantity | S | 1 |
| **IMP-025** | Style placeholders explicitly in the three modal forms | S | 3 |
| **IMP-026** | Add a confirmation to the scheduler's order-line delete | S | 1 |
| **IMP-027** | Give Sankey, Traceability, and the scheduler compact layouts | L | 28 |
| **IMP-028** | Add authentication and a role model | L | all |
| **IMP-065** | Make the two print views printable — the recall report prints white-on-white | S | 2 |
| **IMP-066** | Reclaim viewport height at 200 % zoom; the sticky stack strands the last row | M | 6+ |
| **IMP-067** | Replace the opacity dim on inactive rows with a token that still meets AA | S | 9 |

### Band 2 — High

| # | Improvement | Effort | Screens | Status | Commit |
|---|---|---|---|---|---|
| **IMP-029** | Signal truncation on every limited list | S | 12 | | |
| **IMP-030** | Promote attention items to the entry screen | M | 7 | **DONE** | `ae01e06` |
| **IMP-031** | Make every click target a real, focusable control | M | 25 | | |
| **IMP-032** | Add accessible names to the six icon-only buttons | S | 15 | | |
| **IMP-033** | Bind Enter to the primary action; trap focus in modals | S | 8 | | |
| **IMP-034** | Give every field a persistent label with its unit | S | 8 | | |
| **IMP-035** | Stop showing raw API bodies to the user | S | 17 | | |
| **IMP-036** | Auto-refresh the operational lists; fix the last-updated timestamp | M | 20 | | |
| **IMP-037** | Add a fixed spacing scale and a named type scale | M | all | | |
| **IMP-038** | Restructure the tab bar into primary navigation | L | ~35 | | |
| **IMP-039** | Add a shared page header carrying global search to all four pages | M | 14 | | |
| **IMP-040** | Replace four small-set dropdowns and toggles with segmented controls | S | 4 | | |
| **IMP-041** | Add sortable columns to the four tables where the task demands it | M | 10 | | |
| **IMP-042** | Trim the multi-line table rows | M | 5 | | |
| **IMP-043** | Stop dimming Factory-Ready rows; move the KPI row above the line table | S | 3 | | |
| **IMP-044** | Stop the fabricated sample-data fallback in Sankey and Process Flow | S | 6 | **DONE** | `c972526` `028c685` `684af02` `a6ae9d3` |
| **IMP-045** | Give the Sankey a takeaway, an accessible table, and drill-through | M | 1 | | |
| **IMP-046** | Adopt an accessibility checklist and write down the role → workflow map | S | all | | |

### Band 3 — Medium and below

| # | Improvement | Effort | Screens |
|---|---|---|---|
| **IMP-047** | Add URL state for tab, order, lot, and product | M | ~40 |
| **IMP-048** | Standardise the expand/navigate affordances and the lot row anatomy | M | 20 |
| **IMP-049** | Fix the table striping and the nested `<tbody>` | S | 6 |
| **IMP-050** | Unify the date/time formats; add elapsed values and lot age | M | 25 |
| **IMP-051** | Add next/previous between sibling records | M | 3 |
| **IMP-052** | Add find-within-view to the four long lists | S | 4 |
| **IMP-053** | Add expand-all / collapse-all | S | 10 |
| **IMP-054** | Add `type="search"`, a clear control, and a leading magnifier | S | 6 |
| **IMP-055** | Reduce Orders-toolbar and Order-Detail density | M | 4 |
| **IMP-056** | Let dense tables use the full window width | S | 4 |
| **IMP-057** | Fix the surface-level inversion; reduce to three levels | S | ~20 |
| **IMP-058** | Icon hygiene: the decorative hamburger and the emoji empty states | S | 8 |
| **IMP-059** | Convert the two single-line note fields to textareas | S | 2 |
| **IMP-060** | Confirm only consequential SO status transitions | S | 1 |
| **IMP-061** | Add a help control to the dashboard | M | ~50 |
| **IMP-062** | Add `prefers-reduced-motion` and `prefers-contrast` support | S | all |
| **IMP-063** | Cap and restructure the two disambiguation choice lists | S | 2 |
| **IMP-064** | Add a context label to the notes list and the depleted-lots table | S | 3 |

---

## The improvements

---

### IMP-001 — Fix `var(--bg-card, #fff)` in the lot-disambiguation panel

**Status:** **DONE** · commit `4a96cc7` (branch `fix/ux-band-0`) — Defined `--bg-card` in both theme blocks in `dashboard.css` and moved the disambiguation button styles into `.disambig-*` classes, removing the `#fff` fallback. No `var(--token)` reference in `dashboard/` now resolves to an undeclared token.

**Rules:** ACCESS-008 (Critical) · LAYOUT-002 · FEEDBACK-012 · OTHER-010 · INPUT-022
**Screens:** S-55 · **Importance:** Critical · **Effort:** S

**Browser check:** confirmed fixed. S-55 renders correctly in both themes across all captures; no `var(--token)` in the panel resolves to a fallback. ([06](06-browser-check.md))

`dashboard/dashboard.js:1379` — `background: var(--bg-card, #fff)`. The token `--bg-card` is defined nowhere in `dashboard.css`, so the `#fff` fallback applies while the text inherits `--text: #f1f5f9` from the panel. Contrast ≈ **1.1 : 1** — near-white text on a white background.

This is the screen Arturo lands on when a lot code matches more than one product, at the single moment he must choose correctly. A one-line fix.

**Fix:** replace the inline styles at `1377-1380` with the `.btn-secondary` class (`dashboard.css:1828-1836`), which already resolves through the theme system.

---

### IMP-002 — Fix the sticky offsets that hide the header on every phone

**Status:** **DONE** · commit `45be3eb` (branch `fix/ux-band-0`) — Replaced the hard-coded offsets with `--site-nav-h` / `--header-h`, seeded in `:root` at the previous desktop values and kept current by a `ResizeObserver` on `.site-nav` and `.app-header`. `.tab-bar` now uses `top: calc(var(--site-nav-h) + var(--header-h))`; the stale `.tab-bar { top: 91px }` in the `≤768px` block is gone. Verified by reading the CSS/JS, not in a browser — see the PR note on local verification.

**Rules:** LAYOUT-003 (Critical) · LAYOUT-011 · LAYOUT-015 · ACCESS-001 · TOUCH-001
**Screens:** S-03, S-05, and the top of every tab pane · **Importance:** Critical · **Effort:** S

**Browser check:** confirmed fixed. In all 384 captures no sticky bar overlaps another — at 390 px, at 1440 px, or at 200 % zoom. The "verified by reading the CSS/JS, not in a browser" caveat above is now discharged. The *height* the three bars consume together is a separate problem: see IMP-066. ([06](06-browser-check.md))

`dashboard/dashboard.css:140-142` sets `.app-header { position: sticky; top: 48px }` and `259-260` sets `.tab-bar { position: sticky; top: 91px }` — 48 px of site nav plus a 43 px single-row header. At ≤768 px the `@media` block at `2182-2184` deliberately wraps the header to three rows (brand, then a full-width `.header-right` containing the three-month mini-calendar, then a full-width search row), taking it well past 150 px. The same block re-states `.tab-bar { top: 91px }` at `2189` without recomputing it.

Result: **the tab bar sits on top of the app header on every phone**, hiding the global search field and the Refresh control. The same failure occurs at 200 % desktop zoom, which reports a sub-768 px viewport.

**Fix:** make the site nav, header, and tab bar one sticky container so the offsets compose, or set `--header-h` from a `ResizeObserver` on `.app-header` and use `top: calc(48px + var(--header-h))`.

---

### IMP-003 — Add horizontal-scroll wrappers to the six unwrapped tables

**Status:** **DONE** · commit `4f34036` (branch `fix/ux-band-0`) — Added a shared `.table-scroll` with per-table minimum widths and wrapped all eight tables across the six render functions listed above.

**Rules:** LAYOUT-003 (Critical) · LAYOUT-011 · LAYOUT-013 · DATA-004 · ACCESS-001
**Screens:** S-10, S-11, S-12, S-13, S-16, S-17, S-18, S-25, S-26, S-27, S-28, S-42, S-43 (13) · **Importance:** Critical · **Effort:** S

**Browser check:** confirmed fixed. No screen on `index.html` overflows the document horizontally at 390 px, 1440 px, or 200 % zoom. Every one of the 28 overflowing captures is on a surface outside `index.html`. ([06](06-browser-check.md))

Six tables have no `overflow-x` container, so a narrow window makes the **page** scroll horizontally, sliding content out from under the fixed nav and sticky header:

| Container | Columns | Evidence |
|---|---|---|
| `#orders-table-container` | 11 | `index.html:240`; `dashboard.js:2313` |
| `#er-table-container` | 9 | `index.html:283`; `dashboard.js:3424` |
| `.inv-table` ×3 (Finished Goods, Batch, Ingredients) | 4 / 3 / 2 | `dashboard.js:918`, `1002`, `1106` |
| `.activity-table` ×3 (Daily Entries, Shipping, Receiving) | 5 each | `dashboard.js:1191`, `1244`, `1315` |

The pattern already exists in the same stylesheet: `.order-detail-table-wrap` and `.allocation-table-wrap` (`dashboard.css:1812-1815`) and `.supplies-table-scroll` (`2337`).

**Fix:** wrap the six in `.supplies-table-scroll` (or a shared `.table-scroll`) with an appropriate `min-width`, matching `.order-readiness-table { min-width: 1040px }` (`1817`).

---

### IMP-004 — Add a press state and disable every commit button synchronously

**Status:** **DONE** · commit `6083037` (branch `fix/ux-band-0`) — Added `dashboard/interaction.css`, linked from all five pages, carrying the product's first `:active` rule plus `:disabled` and `.is-submitting`. All four commit paths now disable and relabel while in flight, matching `submitSupplyRequest`.

**Rules:** ACTION-002 (Critical) · FEEDBACK-001 (Critical)
**Screens:** every screen with an interactive control (~60) · **Importance:** Critical · **Effort:** S

**No `:active` rule exists in any of the six style sources** — `dashboard.css`, `mini-calendar.css`, `sankey.html`, `process-flow.html`, `traceability.html`, `scheduler/seven-wells-production-board.html` (verified by grep). Not one button in the product shows that a tap registered.

Four commit paths are additionally never disabled while in flight, so a double-tap fires twice:

| Commit | Evidence |
|---|---|
| Save a note — creates two notes | `dashboard.js:1809-1847`; `#note-save-btn` never touched |
| Toggle a note done | `dashboard.js:1741-1753` |
| Toggle Factory Ready | `dashboard.js:2455-2486` |
| Close / Cancel an expected receipt (after arming) | `dashboard.js:3465-3495` |

The rule names the causal chain exactly: *"nothing visibly changing invites a second tap and a duplicate submission."*

**Fix:** one global rule — `button:active, [role="button"]:active { transform: translateY(1px); filter: brightness(.92) }` — plus a `disabled = true` at the top of each of the four handlers with a `finally` restore, matching the pattern already used in `submitSupplyRequest` (`4058-4074`).

---

### IMP-005 — Add a fetch timeout with a stall message and a Retry

**Status:** **DONE** · commit `c451682` (branch `fix/ux-band-0`) — Added `dashboard/fetch-timeout.js` (`window.FL`) with a 15 s default timeout and a distinguishable `StallError`. All 13 `fetch` call sites in `dashboard/` route through it, and a stall renders a message with a working Retry. Rendering **over** the last-good view stays with IMP-008.

**Rules:** FEEDBACK-003 (Critical) · ERROR-001 · ERROR-002
**Screens:** every network-bound screen (~40) · **Importance:** Critical · **Effort:** S

No fetch anywhere in the product has a timeout, an `AbortController`, or a `signal`: `fetchAPI` (`dashboard.js:484-491`), `fetchSalesAPI` (`1914-1925`), the direct lot fetch (`1354`), `refreshHealthBadge` (`4114`), `exportOrdersMatrix` (`2199`), `api()` (`traceability.html:351-361`), and the Sankey and Process Flow fetchers.

If the API accepts a connection and then hangs — the common cold-container failure — every loading state waits forever. And the dashboard's indicator is **static text** (`.loading-indicator`, `dashboard.css:944-950`), which the rule treats as a freeze by definition.

**Fix:** wrap the two shared fetchers in an `AbortController` with a ~10 s timeout that rejects distinguishably, and render *"Server not responding — showing data from 3:42 PM. [Retry]"* **over** the last-good view rather than replacing it (see IMP-008). Process Flow already implements the stale-banner half correctly and can be the model.

---

### IMP-006 — Constrain `#er-qty` to numeric and validate on blur

**Status:** **DONE** · commit `44c5eb5` (branch `fix/ux-band-0`) — `#er-qty` is now `type="number" min="0" step="any" inputmode="decimal"` with a strict `readNumericInput` reader and a blur check. The scope was extended to every quantity input: the scheduler's ten `parseFloat` sites now go through an equivalent `numIn`/`rejectNum` pair. `"12O"` and `"2,000 lb"` are rejected where `parseFloat` returned 12 and 2.

**Rules:** INPUT-007 (Critical) · INPUT-008 (Critical) · INPUT-020
**Screens:** S-45 · **Importance:** Critical · **Effort:** S

`dashboard/index.html:419` declares the expected-quantity field as `<input type="text" inputmode="decimal">`. `dashboard/dashboard.js:3580-3581` validates it only at commit, with `parseFloat`.

`parseFloat` **silently coerces**: typing `12O` (letter O) commits **12**; pasting `"2,000 lb"` from a supplier email commits **2**. The rule's stated purpose is *"Prevents '12O' (letter O)"* and its hard clause is *"Invalid data is never silently accepted."* This is a 1000× error on an incoming-delivery quantity, accepted without a word.

**Fix:** change to `type="number" min="0" step="any" inputmode="decimal"`, matching `#supply-request-qty` (`index.html:364`) and `.allocation-quantity-input` (`dashboard.js:2923`), and add a blur check.

---

### IMP-007 — Stop full-container re-renders on background refresh and per-row writes

**Rules:** LAYOUT-020 (Critical) · NAV-005 (Hard rule) · FEEDBACK-006 · FEEDBACK-010 · NOTIFY-005
**Screens:** S-14, S-20, S-25, S-26, S-27, S-28, S-49, S-51, S-61, S-87 (10) · **Importance:** Critical · **Effort:** M

**Browser check:** measured. Running `refreshAll()` scores **CLS 0.62** on Daily Entries at 390 px, 0.469 on the Shipping and Receiving logs, and 0.23 on the calendar day-detail panel — the blank-and-repaint made visible. On the Orders tab it also collapses expanded rows, moving eight expand toggles up 62 px. By contrast every genuine background timer in the product measured CLS 0. ([06](06-browser-check.md))

Six places rebuild an entire list under the user:

| Trigger | What is destroyed | Evidence |
|---|---|---|
| Recent Entries 60 s poll | the whole feed, replaced first by a loading block | `dashboard.js:626-631` → `602-624` → `568-600` |
| Process Flow 60 s poll | the whole line grid | `process-flow.html`, `grid.innerHTML = html` |
| Factory Ready checkbox | the whole orders table, **twice** — optimistically and again on response; collapses every expanded row and **destroys unsaved note text on other rows** | `dashboard.js:2472`, `2479`, `2337` |
| Factory Ready note save | the whole orders table | `dashboard.js:2441` |
| Note done checkbox | the whole notes list; with "Show completed" off the row vanishes and everything below jumps | `dashboard.js:1749` → `1657-1674` |
| Supply lot fetch | the whole supplies table, twice, at an unpredictable moment after the tap | `dashboard.js:3799`, `3813`, `3858` |

The orders case is also a NAV-005 hard-rule failure: one row's action silently discards another row's unsaved input.

**Fix:** patch the changed row in place (update the checkbox, the `so-ready` class, the pill) instead of re-rendering; for the two polls, buffer into *"3 new entries — tap to refresh"* as LAYOUT-020 prescribes.

---

### IMP-008 — Preserve drafts; stop blanking the view on a failed refresh

**Rules:** ERROR-002 (Critical, Hard rule) · ERROR-004 · FEEDBACK-001 · FEEDBACK-008
**Screens:** S-08, S-10…S-13, S-16…S-18, S-20, S-22, S-25…S-28, S-30, S-31, S-33, S-40, S-45, S-49, S-51, S-53 (~20) · **Importance:** Critical · **Effort:** M

**Two separate defects with one root cause — the container is always rewritten.**

**(a) No draft is ever preserved, and every exit path discards silently.** `localStorage` holds one value (the theme, `dashboard.js:46`); `sessionStorage` holds one (expanded panels, `12`). All three modals discard on the X button, on Cancel, **and on a backdrop click** — `closeNoteModal` (`1804-1807`, bound at `1871-1873`), `closeErModal` (`3573-3576`, `3640-3642`), `closeSupplyRequestModal` (`4005-4008`, `4100-4107`, which also discards on **Escape**). Order Detail edit mode loses everything on `closeOrderDetail` (`3308-3319`), on the "Done" button (`2850-2856`), and on any browser refresh.

**(b) A transient network blip destroys the data on screen.** Thirteen `catch` blocks run `container.innerHTML = ''` before showing an error: `dashboard.js:700`, `899`, `969`, `1085`, `1180`, `1233`, `1298`, `1671`, `2271`, `2547`, `3380`, `3724`, `3915`. Luz reading the orders table loses it to a one-second Wi-Fi drop.

Related contradiction: `hideError` runs only at the top of each `refresh*`, never in the matching `render*`, so after a failure any filter change renders a complete table **beneath** a red "Failed to load" banner.

**Fix:** add a Save Draft / Discard / Cancel sheet to the three modals and to edit mode (the rule's own worked example); mirror form state into `sessionStorage` keyed by record id; and change every `catch` to leave existing content in place and render the error above it. Process Flow already does the latter (`else { return; // Keep showing last good data }`) and the scheduler already does the former via `localStorage` (`scheduler:376-378`).

---

### IMP-009 — Raise operational text above the 12 px floor; add a mobile body size

**Rules:** ACCESS-006 (Critical, Hard rule) · LAYOUT-005 · FEEDBACK-013 · CHART-006 · ACCESS-001
**Screens:** ~50 · **Importance:** Critical · **Effort:** M

The standard's floors: desktop body 14–16 px, **nothing operational below 12 px**, 11 px for non-operational metadata only, mobile body ≥ 17 px.

`body { font-size: 14px }` (`dashboard.css:83`) is at the bottom of the desktop range and has **no mobile override**, against a 17 px mobile floor.

**At least twenty-two selectors put operational text below 12 px.** The sharpest cases:

| Size | What it carries | Evidence |
|---|---|---|
| **8 px** | mini-calendar day numbers and ship-date indicators | `mini-calendar.css:63-74`, still 8 px at ≤1050 px (`129-133`) |
| 9–10 px | trace link quantities and confidence glyphs | `traceability.html:1051`, `1063`, `1131` |
| 10 px | **why an order cannot ship** (`.readiness-chip-detail`) | `dashboard.css:1628` |
| 10 px | **the lot code of a lot-level reservation** | `dashboard.css:2055-2062` |
| 10 px | **SKU** in the production detail | `dashboard.css:614-619` |
| 10 px | Dispatch Ready / Blocked, Factory Ready pill, allocation source and TTL, note category and priority, two sets of table headers | `1594`, `1584`, `2074`, `2046`, `1381`, `1050-1070` |

Plus fourteen selectors at 11 px carrying operational values where the standard permits 11 px only for metadata — including the ship-date weekday, the note due date, pallet counts, case and batch badges, and **every table `<th>` in the product**.

**Fix:** define `--text-xs: 12px` as a hard floor and `--text-caption: 11px` for metadata only; raise the twenty-two sub-12 px operational selectors; give the mini-calendar legible sizing or replace it with a compact count; add a mobile block raising `body` to 17 px.

---

### IMP-010 — Fix the light-theme contrast failures and `--text-dimmed`

**Rules:** ACCESS-008 (Critical, Hard rule) · FEEDBACK-012 · LAYOUT-005 · OTHER-010
**Screens:** ~30 · **Importance:** Critical · **Effort:** S

**Browser check:** measured. 151 of 190 light-theme captures carry at least one text node below AA; the worst non-print value is **1.16 : 1** (`.so-ready-pill`). 143 of 190 dark captures fail too, so this is not a light-theme-only defect — `span.lot-link` measures 2.32 : 1 on `--surface` in dark, and `.mini-calendar-dow` 1.8 : 1 on both. ([06](06-browser-check.md))

Computed WCAG ratios (full derivation in [05](05-access-icon-other.md)):

| Pair | Ratio | Target | Evidence |
|---|---|---|---|
| **`.so-ready-pill`** — `#86efac` on `rgba(34,197,94,.12)` over white | **≈ 1.27 : 1** | 4.5 : 1 | `dashboard.css:1576-1588` |
| **`.order-edit-message.success`** — `#86efac` on `rgba(52,211,153,.10)` over white | **≈ 1.27 : 1** | 4.5 : 1 | `dashboard.css:1885-1890` |
| `.order-edit-message.error` — `#fca5a5` on a near-white fill | ≈ 1.73 : 1 | 4.5 : 1 | `dashboard.css:1891-1898` |
| `--text-dimmed #64748b` on `--surface` (dark) | 3.07 : 1 | 4.5 : 1 | `dashboard.css:14` |
| `--text-dimmed #94a3b8` on white (light) | 2.56 : 1 | 4.5 : 1 | `dashboard.css:52` |
| `.tab.active` — `--primary` on `--surface` (dark) | 3.98 : 1 | 4.5 : 1 | `dashboard.css:279` |
| `.date-overdue` — `--danger` on `--surface` (dark) | 3.89 : 1 | 4.5 : 1 | `dashboard.css:1671` |
| `.lot-link` on `--row-header` (dark) at 12 px | 4.44 : 1 | 4.5 : 1 | `dashboard.css:840-848` |

The first two matter most: in the light theme the **"✓ READY" pill** and the **"Header saved."** confirmation are effectively invisible. Both confirm a write.

`--text-dimmed` fails in **both** themes across nine selectors, including `.note-meta`, which carries the due date and its overdue state.

**Every failure above is a hard-coded value or an under-contrasted token. Every colour that resolves through the theme system passes** — `--text` reaches 12.6 : 1 dark and 17.9 : 1 light.

**Fix:** darken `--text-dimmed` in both themes to clear 4.5 : 1; replace the nine hard-coded status colours with theme-aware tokens (see IMP-012); nudge `--primary` and `--danger` in the dark palette.

---

### IMP-011 — Minimum 44 × 44 pt hit targets and a focus ring on every control

**Rules:** TOUCH-003 (Critical, Hard rule) · LAYOUT-012 (Critical) · ACCESS-003 · ACTION-002
**Screens:** ~65 · **Importance:** Critical · **Effort:** M

**Browser check:** measured, and the estimates in the table below hold. `.order-ready-checkbox` **16 × 16**, `.note-checkbox` **18 × 18**, `.note-action-btn` **22.4 × 21**, `.order-expand-toggle` **22 × 22**, `.btn-sm` **≈ 24** tall, `.lot-link` **43.2 × 14**, `.btn-close` **12.9 × 22**. Three classes not in that table also fail and belong in scope: the scheduler's `button.copyday` at **12 × 12**, its `#leftpanel .ctl > input` at **15 × 15**, and its order-book `a.o-action` at **20 × 20**. 314 of 384 captures fail; 70 of the 74 screens that render a control at all. ([06](06-browser-check.md))

The rule applies *"regardless of input method and on desktop tables too."* Measured heights:

| Class | Declared | ≈ Height | Used for |
|---|---|---|---|
| `.note-action-btn` | `padding: 3px 6px; font-size: 11px` (`css:1111`) | **22 × 20** | Edit / **Delete** a note |
| `.order-ready-checkbox` | `16px` (`css:1336`) | **16 × 16** | the Factory Ready write |
| `.note-checkbox` | `18px` (`css:1026`) | 18 × 18 | mark a note done |
| `.show-more-btn` | `padding: 0` (`css:827`) | **≈16** | reveal 96 hidden rows |
| `.lot-link` | inline span, 12 px (`css:840`) | **≈15** | **the main drill path** |
| `.btn-close` | `font-size: 22px`, no padding (`css:880`) | ≈22 | close every modal |
| `.order-expand-toggle` | `22px` (`css:1344`) | 22 × 22 | expand order lines |
| `.btn-sm` | `padding: 4px 10px` (`css:394`) | **≈24** | ER Edit/Close/Cancel, Release, Cancel ×3, calendar arrows, Retry |
| `.btn-refresh` | `padding: 6px 14px` (`css:236`) | ≈30 | **every primary commit** |
| `.notes-filter-btn`, `.supplies-subtab`, `.site-nav-link`, `.search-item`, `.er-product-option` | | 27–33 | filters, navigation, pickers |

**Two controls in the whole product meet the rule** — `.recent-entries-refresh` and `.recent-entries-retry` (`dashboard.css:290`, `316`), both explicitly `min-height: 44px`.

Only three selectors define a focus indicator: `.day-card-trigger:focus-visible` (`436-441`), `.supplies-subtab:focus-visible` and `.supply-item-row:focus-visible` (`2300-2304`).

Two adjacencies compound this into the LAYOUT-012 Critical failure: Notes **Edit and Delete** are 22 × 20 px, 4 px apart, permanently visible on touch (`css:1102-1122`, `2193`); and the orders row's **expand toggle and Factory Ready checkbox** sit ~8 px apart at the leading edge (`css:1331-1342`).

**Fix:** a global `min-height: 44px; min-width: 44px` for touch, or a `::before` pseudo-element expanding the hit area without changing the visual size on desktop; plus one shared `:focus-visible` rule. Separately, move Delete out of the notes row into an overflow and put the SO number between the orders row's two controls.

---

### IMP-012 — Separate the family palette from the status palette; tokenise the hard-coded colours

**Rules:** FEEDBACK-012 (Critical, Hard rule) · ACCESS-008 (Critical) · OTHER-010 (Hard rule) · CHART-007 · LAYOUT-018 · LAYOUT-017
**Screens:** ~40 · **Importance:** Critical · **Effort:** M

**Four distinct defects, one root cause.**

**(a) Two tokens are byte-identical to status tokens.** `--category-granola #fbbf24` equals `--badge-amber-text #fbbf24` (`dashboard.css:19`, `28`), and `--category-coconut #60a5fa` equals `--primary-hover #60a5fa` (`17`, `16`). So on the Today So Far tile the word **"Granola"** renders in exactly the colour that means "overdue" everywhere else, and a coconut **product name** renders in the app's interactive blue.

**(b) Twelve raw values bypass the token system,** nine of them dark-mode-only status colours that produce the IMP-010 light-theme failures: `.site-nav` `#1a1a2e` (`98`), `.site-nav-link.active` `#3b82f6` (`115`), `#991b1b` ×2 (`743`, `940`), `.so-badge.status-ready/-partial_ship/-invoiced` (`1570-1573`), `.so-ready-pill` (`1576-1588`), `.order-ready-checkbox` `#22c55e` (`1341`), `.so-ready td:first-child` `#22c55e` (`1321`), `.order-edit-message.*` (`1885-1898`), `.detail-card #16213e` (`traceability.html:188`), `#93c5fd` (`mini-calendar.css:111`), and `var(--bg-card, #fff)` (IMP-001).

**(c) Five separate palettes, no shared file, and no high-contrast variant.** `dashboard.css:4-75`, `traceability.html:10-34`, `sankey.html`, `process-flow.html` (which uses `--text-dim` where the dashboard uses `--text-dimmed`), and `scheduler:8-18` (an entirely different light "paper" palette). The rule requires one token file with light, dark **and** increased-contrast variants; there is no `prefers-contrast: more` block anywhere.

**(d) Sankey and Traceability use the same four hex values for opposite meanings.** `#1D9E75` is "Ingredients" in one and **"Supplier"** in the other; `#378ADD` is "Customers" in one and **"Ingredient Lot"** in the other (`sankey.html:339-342` vs `traceability.html:28-33`). Coconut — the standard's own example of a colour that must never change — has five different treatments across the product.

**Fix:** one shared token file with `--surface-1/2/3`, `--text-1..4`, `--status-*`, `--entity-*`, and `--stage-*`, in light, dark, and high-contrast variants; give product families a hue range that does not overlap status; delete the five per-file `:root` blocks and the twelve raw values.

---

### IMP-013 — Introduce semantic button roles

**Rules:** ACTION-003 (Critical) · ACTION-006 (Critical, Hard rule) · ACTION-004 · LAYOUT-002 (Hard rule) · LAYOUT-018
**Screens:** ~30 · **Importance:** Critical · **Effort:** M

`.btn-refresh` (`dashboard.css:236-248`) is a filled accent button carrying **twelve different semantic roles**: Refresh ×5, Export CSV, Export Matrix, + New, + New Expected Receipt, Request Supply, Save ×2, Submit Request, Try again, Edit Order, Save Header, Save Lines, Allocate. The Orders toolbar therefore shows three filled accent buttons in a row (`index.html:233-235`), so nothing on the screen reads as *the* primary action.

**There is no destructive style at all** (no `.btn-danger` in the stylesheet). The four destructive actions each invent something different: note delete is neutral with red **only on hover** (`css:1122`); Release is red *text* on a neutral button (`2081`); ER Cancel turns **amber** when armed — the same treatment as the benign "Close" (`2230`, `dashboard.js:3470-3483`); the scheduler's delete is a neutral link (`scheduler:158`). Changing an order's status to `cancelled` is an ordinary `<option>` in a neutral select (`dashboard.js:2696-2698`).

And Cancel is a *different size* from Save in every modal — `.btn-sm` beside `.btn-refresh` (`index.html:390-391`, `438-439`, `500-501`) — which ACTION-004 prohibits.

**Fix:** define `.btn-primary` / `.btn-secondary` / `.btn-danger` at one shared size; reassign Refresh and both Exports to secondary, "+ New" and Save/Submit to primary, and Delete / Release / Cancel-record to danger; move the SO "Cancelled" transition out of the status select into a separate confirmed red action.

---

### IMP-014 — Replace the native `confirm()` and `alert()` dialogs

**Rules:** ACTION-008 (Critical, Hard rule) · ERROR-004 (Hard rule) · ERROR-006 · ERROR-007 (Critical) · ERROR-010 · ACCESS-004 · LAYOUT-002
**Screens:** S-22, S-23, S-35, S-38, S-80, S-89 (6) · **Importance:** Critical · **Effort:** M

A native `confirm()` makes **OK** both the visually dominant and the Enter-activated button — and OK is the destructive choice in all four instances: delete a note (`dashboard.js:1771`), change an SO status (`2819`), release a reservation (`3096`), wipe the entire scheduler plan (`scheduler:1321`). The audit test asks *"Is there any dialog where Enter destroys or reverses data?"* Four times. The standard's own example inverts it: *"'Keep Shipment' is primary/Enter; 'Void' is red and secondary"* — which a native dialog cannot express.

Three `alert()` calls deliver **form validation** as an interruption where an inline error belongs (`dashboard.js:1813`, `1845`; `scheduler:1565`), and native dialog text **cannot be selected or copied** (ERROR-010) — so `alert('Save failed: ' + err.message)` puts a raw API error in the one place it cannot be pasted anywhere.

**Fix:** route all six through the existing modal shell with the **non**-destructive option as primary and Enter-bound and the destructive one as `.btn-danger` (IMP-013); move the three validation alerts inline, matching `#er-modal-error` and `#supply-request-modal-error` which already do this correctly. Also give the ER two-step arm an explicit "no" — currently the only exits are waiting four seconds or clicking elsewhere (`3475-3481`).

---

### IMP-015 — Add a non-colour carrier to five status cues

**Rules:** FEEDBACK-011 (Critical, Hard rule) · FEEDBACK-013 · ACCESS-008
**Screens:** S-06, S-08, S-09, S-12, S-20, S-25, S-26, S-31, S-42, S-71, S-75, S-91 (12) · **Importance:** Critical · **Effort:** S

Most of the product passes — status badges carry text, late entries carry *"entered next day"*, Traceability carries ✓/⚠/✗ glyphs and dashed borders. **Five cues carry colour and nothing else:**

| Cue | Evidence | Survives greyscale as |
|---|---|---|
| Overdue ship date | `dashboard.css:1671` (`--danger` + `font-weight: 600`) | 13 px semibold vs regular |
| Overdue weekday line | `dashboard.css:1540-1542` (11 px, weight 500) | nothing |
| Line shortage | `dashboard.css:1819` (`--danger` on a number) | bold vs regular in a 9-column table |
| Note due-date overdue | `dashboard.css:1093` (11 px regular) | nothing |
| **Product family** | `dashboard.css:372-374`, `505-507`, `582-584`, `670-672` | nothing — a granola row and a coconut row are identical |

**S-75 additionally fails the colourblind clause:** the scheduler's `.d-better` (`--ok #2e7d4f`) and `.d-worse` (`--late #bf2f24`) sit at relative luminances of **0.158 and 0.132** — a red/green pair separated by hue alone, on the panel Blubber reads to judge whether a plan change helped.

The two print views (S-71, S-91) are the greyscale case the rule names explicitly.

**Fix:** `⚠ 3d overdue` instead of a red date; `Short 40 lb` instead of a red number; a leading family initial or shape beside the product name; add an ▲/▼ glyph to the delta panel. Also add `text-decoration: underline` to `.order-link` at rest (`css:1544-1553`), which is currently underlined only on hover so clickable SO numbers are indistinguishable in greyscale.

---

### IMP-016 — Add reversal paths and undo for the four one-way doors

**Rules:** ERROR-003 (Critical, Hard rule) · ACTION-008 · DRAG-003 (forward-looking)
**Screens:** S-20, S-23, S-37, S-38, S-42, S-43, S-51, S-80, S-87 (7 distinct) · **Importance:** Critical · **Effort:** M

**There is no undo anywhere in the product**, and four actions have no path back through the UI at all:

| Action | Why it is one-way | Evidence |
|---|---|---|
| Close an expected receipt | row actions render only `if (r.status === 'open')`; a closed record shows an empty action cell | `dashboard.js:3437`, `3458` |
| Cancel an expected receipt | same guard | `dashboard.js:3437`, `3461` |
| Mark a supply request Done | the Done button renders only `if (request.status === 'open')` | `dashboard.js:3943-3945` |
| Delete a scheduler order line | immediate filter with no confirmation and no recovery | `scheduler:1553-1554` |

`erSetStatus` (`3465`) already accepts an arbitrary status and would take `'open'`; nothing calls it that way.

**Every correction also requires retyping.** Edit modals prefill the *current* value (`1794-1799`, `3554-3557`) ✓ but nothing preserves the *previous* one, so undoing an edit means remembering it. And the ledger's correction model is **visible but not actionable**: the Recent Entries feed renders `amend`, `void`, and `restore` events with dedicated badges (`557-566`) while the dashboard offers no way to create one.

**Fix:** relax the two `if (status === 'open')` guards to render a Reopen action; add an Undo affordance to note delete and allocation release using the values already in `state`; add a Reverse action on Recent Entries rows that opens a pre-filled correction. The scheduler's baseline/delta (`scheduler:11`, `948`, `25`) is the product's one genuine forgiveness mechanism and shows the intent.

---

### IMP-017 — Mark required fields and gate every commit

**Rules:** INPUT-019 (Critical) · INPUT-007 · FEEDBACK-008 · ERROR-004
**Screens:** S-22, S-33, S-36, S-45, S-53, S-88 (6) · **Importance:** Critical · **Effort:** S

`dashboard/index.html` contains exactly **two** `required` attributes in the whole file (`346`, `375`), both inside a form declared `novalidate` (`343`). There is no `.required` class, no asterisk convention, and no `aria-required` anywhere. **Not one commit button in the product is ever gated.**

Meanwhile the *optional* fields are explicitly marked with `<span class="label-hint">(optional)</span>` on eight fields (`362`, `369`, `424`, `428`, `433`, `466`, `479`, `485`) — so requiredness is communicated only by **absence of an optional marker**, and every failure surfaces after the user has filled the whole form.

**Fix:** add a `*` marker and `aria-required` to the eight required fields; bind each form's commit button to a `checkValidity()`-driven disabled state. The Supply Request form is 80 % of the way there already — it has the attributes and field-specific inline messages; it needs the markers and the gate.

---

### IMP-018 — Replace `#note-entity-id` free text with a type-ahead

**Rules:** INPUT-011 (Critical, Hard rule) · INPUT-014 · SEARCH-004
**Screens:** S-22 · **Importance:** Critical · **Effort:** M

`dashboard/index.html:485-497` — "Pin to" is a `<select>` of entity **types** (Product / Lot / Customer / Supplier) and "Entity Name / ID" is a free `<input type="text">`. The typed string is stored verbatim (`dashboard.js:1821`) and rendered as a chip (`1720`).

All four types are enumerable and the endpoints already exist: `/products/search` (used at `3520`), `/suppliers` (`3499`), and the global `/search` covering products, lots, orders, and customers (`1514`). A note pinned to *"Granola Clasic 25 LB"* is a note nobody will ever find.

**Fix:** swap in a type-ahead select driven by the entity type, reusing the `#er-product-search` pattern (`3516-3540`). The Supply Request modal shows the correct shape for the exception case: a select over the enumerable set plus an explicit "not listed" branch that stores free text on the *request* rather than creating a master record (`index.html:346-349`; `dashboard.js:4042-4056`).

---

### IMP-019 — Scope the Supplies search, or say that it is scoped

**Rules:** SEARCH-003 (Critical, Hard rule) · SEARCH-007 · NAV-012
**Screens:** S-48, S-49 · **Importance:** Critical · **Effort:** S

`dashboard/index.html:311-314` — placeholder *"Search products..."*. `dashboard/dashboard.js:3830-3831` runs the filter over `supplyItemsForTab()` (`3656-3665`), which has **already narrowed to the active sub-tab**.

With Ingredients selected, searching "gloves" returns *"No products match this search."* (`3833`) — a definitive negative for an item that exists one tab over under Packaging. This is the exact failure the rule exists to prevent: *"Prevents an operator thinking a lot doesn't exist because they searched the wrong scope."* The consequence is a duplicate unlisted supply request for something already tracked.

**Fix:** bind the placeholder to the active sub-tab (*"Search ingredients…"*), and change the empty state to *"No ingredients match "gloves". [Search all supplies]"* with a control that switches to the All sub-tab and keeps the query.

---

### IMP-020 — Fix the Traceability lot index and state its real scope

**Rules:** SEARCH-003 (Critical) · NAV-012 · SEARCH-002 · DATA-004 (Critical)
**Screens:** S-63, S-65 · **Importance:** Critical · **Effort:** M

`dashboard/traceability.html:374-400` builds the entire lot search index client-side from `/transactions/history?limit=100`. The placeholder promises *"Search by lot code, supplier lot, or product name..."* (`276`); the actual scope is *"lots appearing in the last 100 transactions."*

A lot older than that returns *"No matches"* (`469`) — **the same message as a lot that does not exist**. Per-group results are additionally capped at 8 with no "more" cue (`462`). NAV-012 names this failure directly: *"The count stops Arturo concluding a lot doesn't exist because it's filtered out."* Today he would conclude exactly that.

It is also a second, divergent "global" search: the same lot code queried here and in the dashboard's `#global-search` (`/search?q=`, `dashboard.js:1514`) returns different results, which SEARCH-002 prohibits.

**Fix:** replace the client-side index with a server-side lot search (extending `/search` scoped to lots); until then, state the real scope in the placeholder and in the empty state, and add the "N more" cue to each group.

---

### IMP-021 — Add a `title` to truncated names; middle-truncate lot codes

**Rules:** DATA-004 (Critical, Hard rule) · INPUT-009 · TOUCH-002
**Screens:** S-63, S-68, S-83, S-87 (4) · **Importance:** Critical · **Effort:** S

**S-63 — no path to the full value at all.** `traceability.html:108` — `.search-item .product-name { max-width: 300px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap }`, rendered at `465` with **no `title`**. The product name is what distinguishes two lots with a shared code prefix; *"…Coconut 25 lb – Blue Str…"* and *"…– Red Str…"* clip identically.

**S-68 — truncated from the wrong end.** `traceability.html:1145-1148` — `truncate()` is an **end**-truncation, applied to lot codes at 22 characters (`1115`). The rule is explicit: *"LAT codes… share prefixes and differ at the end — end-truncation hides exactly the distinguishing part; middle-truncate."* The full value is in the hover tooltip ✓ but that is unreachable on touch.

**S-83, S-87** — `scheduler:151-153` truncates product names with no `title`.

**Fix:** add `title` to all four; change `truncate()` to a middle-truncation. `.er-notes` already shows the correct pattern — ellipsis **plus** `title` (`dashboard.css:2217`; `dashboard.js:3429`).

---

### IMP-022 — Add a currency symbol and unit to the order-line edit inputs

**Rules:** INPUT-008 (Critical, Hard rule) · INPUT-002 · DATA-005
**Screens:** S-31, S-33 · **Importance:** Critical · **Effort:** S

`dashboard/dashboard.js:3259-3266` — the quantity and price inputs in Order Detail edit mode have **no labels at all**. Their only identifier is the column header, which on a `min-width: 1040px` table inside a scroll wrapper (`dashboard.css:1817`, `1812`) can be scrolled out of view while the input stays.

The price input carries **no currency anywhere** — no `$`, no "USD", nothing in a label. The read-only rendering four lines away does show it: `'$' + Number(l.case_price).toFixed(2)` (`3270`). The quantity's unit appears only in an 11 px muted `<small>` beneath the field (`3261`).

Editing a price with no currency shown is a data-integrity risk on a field that PATCHes `unit_price` (`2776`).

**Fix:** add a `$` prefix adornment to the price input and a persistent unit adornment to the quantity input, matching `#supply-request-qty`'s live unit span (`index.html:363-366`) and `#er-qty`'s labelled unit (`418`).

---

### IMP-023 — Fix the Supplies "Incoming" column unit mismatch

**Rules:** INPUT-008 (Critical) · DATA-005
**Screens:** S-49 · **Importance:** Critical · **Effort:** S

`dashboard/dashboard.js:3845-3848` computes `unitIsPounds` from the item's own unit and, when it is false, renders the incoming total as `` `${fmtWt(incomingTotal)} lb` ``. The table header is `Name | On Hand | Unit | Incoming | Status` (`3838`).

So a row measured in "box" shows **On Hand in boxes and Incoming in pounds**, side by side, under a Unit column that says "box". The two numbers are not comparable and nothing warns the reader. The code is aware of the mismatch — it deliberately appends "lb" — which makes it a display decision rather than a bug, but the header does not carry it.

**Fix:** convert the incoming total to the item's unit where a conversion factor exists, or split into two columns (`Incoming (lb)` alongside `Incoming ({unit})`) so the header states what each number is.

---

### IMP-024 — Add inline cross-field validation to the allocation quantity

**Rules:** INPUT-007 (Critical, Hard rule) · FEEDBACK-008 · INPUT-018
**Screens:** S-36 · **Importance:** Critical · **Effort:** S

`dashboard/dashboard.js:3052-3059` validates that the quantity is a positive number and that a lot is chosen ✓ — but the **cross-field** check (requested versus coverable) is left entirely to the server, which returns `OVER_ALLOCATION` with a `coverable_lb` (`2977-2980`).

The client already holds `unallocated_need_lb` for the selected line and uses it to prefill the field (`3038-3039`). The rule requires cross-field validation *"as soon as both values exist"*; here both values exist before the user types.

The eventual message is excellent — *"Only 240 lb is coverable. Reduce the request to 240 lb or release a competing reservation."* — it simply arrives one round-trip too late.

**Fix:** run the comparison on `input` and surface the same message inline. The scheduler's pin modal is the reference: `feas()` recomputes station capacity on every keystroke and reports feasibility **with a remedy** (`scheduler:1457-1462`).

---

### IMP-025 — Style placeholders explicitly in the three modal forms

**Rules:** INPUT-022 (Critical, Hard rule) · ACCESS-008
**Screens:** S-22, S-45, S-53 · **Importance:** Critical · **Effort:** S

**Browser check:** measured, and the scope is wider than three forms. With no `::placeholder` rule, the user agent picks `rgb(117, 117, 117)`, which measures **2.69 : 1** against `--search-bg` in dark theme — a fail on all 36 measurements. In light theme the same default passes. Nine fields are affected: `#note-title`, `#note-body`, `#note-entity-id`, `#er-product-search`, `#er-reference`, `#er-notes`, `#supply-request-note`, `.order-ready-note-input` and `.allocation-note-input`. ([06](06-browser-check.md))

`dashboard/dashboard.css:1161-1175` styles `.form-group input / textarea / select` but defines **no `::placeholder` rule**, so every modal field inherits the user agent's default — typically the text colour at ~54 % opacity, which against `--search-bg: #283548` with `--text: #f1f5f9` lands close to `--text-secondary`, the treatment used for real secondary **values**.

The convention exists elsewhere in the same file — `#global-search::placeholder` (`166`), `#orders-customer-search::placeholder` (`1269`), `#supplies-search::placeholder` (`2335`) — and was simply not carried into the forms.

The rule's specific data-integrity example **passes** ✓: no numeric field uses a bare number as a placeholder; all use the `"e.g. 2000"` convention (`index.html:419`, `354`, `358`, `496`).

**Fix:** add `.form-group input::placeholder, .form-group textarea::placeholder { color: var(--text-muted); font-style: italic }`.

---

### IMP-026 — Add a confirmation to the scheduler's order-line delete

**Rules:** ACTION-008 (Critical, Hard rule) · LAYOUT-012 (Critical) · ERROR-003 (Critical) · ERROR-007 · ICON-001 · ICON-004
**Screens:** S-87 · **Importance:** Critical · **Effort:** S

`dashboard/scheduler/seven-wells-production-board.html:1536-1537` renders `✕` (Exclude from plan — **reversible**, restored by the `data-incl` handler at `:1551-1552`) immediately beside `⌫` (Delete order line — **irreversible**), styled identically (`:158`). The delete handler at `:1553-1554` runs at once with **no confirmation and no undo**.

Two remove-shaped glyphs, adjacent, one recoverable and one not — the exact shape LAYOUT-012 prohibits. `⌫` is also a keyboard-key glyph rather than a delete metaphor (ICON-001), and `✕` here is a third meaning for a glyph that already means "close" and "delete" elsewhere (ICON-004).

**Fix:** apply the ER two-step arm (`dashboard.js:3470-3483`) to the delete; separate the two controls; replace `⌫` with a trash glyph or a text label.

---

### IMP-027 — Give Sankey, Traceability, and the scheduler compact layouts

**Rules:** LAYOUT-003 (Critical, Hard rule) · LAYOUT-011 · LAYOUT-013 · OTHER-005 · CHART-006 · TOUCH-003
**Screens:** S-57…S-70, S-72…S-91 (28) · **Importance:** Critical · **Effort:** L

**Browser check:** measured. At 390 px, `traceability.html` overflows the document by **81 px** (135 px once the detail panel is open), `sankey.html` by **110–114 px**, and `process-flow.html` by **109–114 px**. The offender is `.header-right` on all three — the 768 px nav breakpoint wraps the links but not the header's own contents. The scheduler's pin modal overflows by **190 px**. All four surfaces also proved to have **no light palette at all**: their tokens sit on a bare `:root`, so the `data-theme` attribute two of them carry in markup does nothing. ([06](06-browser-check.md))

Three surfaces have no responsive rules beyond the shared nav:

| Surface | Evidence |
|---|---|
| **Sankey** | `sankey.html:49` — the file's only `@media` block styles `.site-nav`. Four columns and their labels compress into an illegible strip below ~700 px. |
| **Traceability** | `traceability.html:49-55` — same. `.search-row` (`75-77`) holds an input plus three buttons with no wrap; the graph layout is a hard 900 px wide (`992-993`) inside an SVG fixed at `height="500"` (`314`), so below ~980 px it exceeds its container from the first render and only pan/zoom recovers it. |
| **Scheduler** | `scheduler:185` — the file's **only** `@media` block is `print`. `#main { grid-template-columns: 240px minmax(0,1fr) 340px }` (`:56`) is fixed; `td.cell { min-width: 124px }` (`:114`); `#app { height: 100vh }` with three independent scroll regions (`:26`). Below ~1100 px the centre board is unusable and no panel can be collapsed. |

**Fix:** add breakpoints to all three; make the Traceability graph width-aware; give the scheduler's side panels a collapse control and let the board scroll independently.

---

### IMP-028 — Add authentication and a role model

**Rules:** INPUT-003 (Critical where applicable) · INPUT-017 (Critical, Hard rule) · OTHER-003 (Hard rule) · SEARCH-006 · NOTIFY-012
**Screens:** all 91 · **Importance:** Critical · **Effort:** L

`dashboard/dashboard.js:1882` — `const SALES_API_KEY = 'dashboard-key-2026';`, repeated in `mini-calendar.js:8`, `traceability.html:334`, `sankey.html`, and `process-flow.html`. It is sent as `X-API-Key` on every **write** path — notes create/update/delete, Factory Ready, order header/line/status PATCH, allocations create/release, expected receipts, supply requests. A write-capable credential ships in clear to every browser that loads the page.

Three design consequences follow from the same root:
- **INPUT-017:** `#supply-request-requested-by` asks who is requesting on every submission (`index.html:374-386`) because the app cannot know; and `postOrderReady` sends `by: 'floor'` as a hard-coded literal (`2414`) which the READY pill then displays as though it were audit data (`2232`).
- **OTHER-003:** there is no role scoping at all — `case_price` is rendered and editable (`3264`, `3270`), and customer names, contacts, and emails appear in global search (`1551`), on every screen, to everyone.
- **SEARCH-006:** per-user search history is not expressible.

**Fix:** a session-based auth layer with three roles (floor / office / owner), server-side scoping of pricing and customer contact data, and an actor identity that replaces both the `requested_by` picker and the `by: 'floor'` literal.

---

### IMP-029 — Signal truncation on every limited list

**Rules:** NAV-012 (High) · NOTIFY-010 (Hard rule) · DATA-005 · SEARCH-007
**Screens:** S-02, S-14, S-25, S-26, S-39, S-42, S-51, S-56, S-61, S-63, S-65 (11) · **Importance:** High · **Effort:** S

Nine list endpoints impose a server-side limit and only three tell the user:

| List | Limit | Cue? | Evidence |
|---|---|---|---|
| Recent Entries | 20 | **no** — the status line reads *"Showing 20 most recently entered ledger events"*, which reads as complete | `dashboard.js:611`, `614` |
| Orders | 200 | **no** | `dashboard.js:2247`, `2261-2263` |
| Expected Receipts | 500 | **no** — and the summary counts are computed from the truncated set | `dashboard.js:3374`, `3412-3413` |
| Supply Requests | 500 | **no** — and the open-count badge is likewise | `dashboard.js:3908`, `3923-3926` |
| Shipments / Receipts | 100 | **no** | `dashboard.js:1177`, `1230` |
| Mini-calendar ship dots | 200 | **no** — a busy day past the limit shows no dot | `mini-calendar.js:108` |
| Process Flow | 100 | `console.warn` only | `process-flow.html` |
| Product panel depleted lots | 10 | count shown, **no path to reveal** | `dashboard.js:1477`, `1485-1487` |

`#orders-dispatch-summary` compounds this: `total` comes from the server's `total_orders_checked` while `ready` is counted from the client's truncated list (`2286-2288`), so *"blocked"* is overstated — a NOTIFY-010 accuracy failure.

**Done correctly, and worth copying:** the allocation lot picker warns explicitly when the result count equals the limit (`2924`, `2993`, `3014`); the Sankey states its truncation (`sankey.html:809-820`); the activity show-more gives the exact hidden count (`1152-1157`); the trace fan-out gives a count and a reveal control (`traceability.html:764-772`).

**Fix:** a shared `truncationNotice(returned, limit)` helper beside `fetchSalesAPI`, called from every list renderer; fix the dispatch summary to count from one population.

---

### IMP-030 — Promote attention items to the entry screen

**Status:** **DONE** · branch `feat/entry-attention-strip`, one commit — `ae01e06` adds a "Needs Attention" strip above Today So Far on the Operations tab carrying all seven counts as chips that deep-link into the view (with the filter pre-applied) that resolves each one. Six counts are derived from data `refreshAll()` already fetches, so no request was added. The seventh — dispatch-blocked — needs `/sales/orders/fulfillment-check`, which the dashboard fetches *only* while the Dispatch Queue filter is selected and therefore never on entry; the table below was optimistic on that one point. Rather than add a request or print a zero it cannot stand behind, that chip reads "—" until the user has opened the dispatch queue once, and reports the real figure thereafter (NOTIFY-010: never fake a badge). Failed background refreshes are counted by a registry keyed on the load-error element ids, so the number mirrors exactly the set of load errors on screen; modal and per-action error ids are excluded by allowlist. FEEDBACK-011 is carried by the numeral, the left-edge bar weight (dotted / 2px / 5px solid), the count's font weight, and a flag glyph shown only when non-zero — verified under a grayscale filter. LAYOUT-020 is held by shipping all seven chips in the static markup with an em-dash placeholder, sizing the grid tracks from the container rather than the content, fixing the count track at 4.5ch, and capping the printed value at "999+". Zero states are neutral: shown, not hidden, not red.

**Remaining for a follow-up.** Two parts of this improvement are deliberately not in scope here. The dispatch-blocked count will only be available on entry once something fetches `fulfillment-check` on load — worth pairing with IMP-036 (auto-refresh) rather than adding a request solely for a badge. And NOTIFY-005 is only *mostly* resolved: the strip surfaces the counts, but there is still no per-item routing of who should see what (IMP-028's role model is the prerequisite).

**Rules:** NOTIFY-011 (High, Hard rule) · NAV-001 · NOTIFY-005 · NOTIFY-010
**Screens:** S-01, S-03, S-05, S-06, S-19, S-24, S-41, S-46, S-47, S-49, S-51 (7 distinct surfaces) · **Importance:** High · **Effort:** M

**There is no notification channel in this product** (no Notification API, no service worker, no push, no email trigger — verified by grep), so NOTIFY-011 carries the entire attention model. And nothing is promoted.

On open the user lands on Operations (`dashboard.js:10`; `index.html:48`, `58`): five read-only production views. The only attention-bearing elements in the chrome are the health badge — a **system-integrity** score, not an operational one (`4114`) — and the mini-calendar's ship dots.

Every operational attention item is behind a tab, several behind a tab *plus* a filter change — **and every count is already computed in code:**

| Item | Computed at | Currently visible only in |
|---|---|---|
| Overdue expected receipts | `dashboard.js:3411` | Tab 6 |
| Open supply requests | `dashboard.js:3923` | Tab 7 |
| Low-stock supplies | `dashboard.js:3674` | Tab 7 |
| Overdue sales orders | `dashboard.js:2009-2015` | Tab 5 + a checkbox |
| Dispatch-blocked orders | `dashboard.js:2287` | Tab 5 + a filter change |
| To-dos due today | `dashboard.js:1716` | Tab 4 |
| Failed background refreshes | swallowed by `Promise.allSettled` (`4163`) | nowhere |

**This is the highest-value single change in the audit.** It resolves NOTIFY-011, most of NOTIFY-005, the NOTIFY-010 absence, and a NAV-001 gap.

**Fix:** an attention strip at the top of the Operations tab rendering the seven counts as chips that deep-link to the relevant tab with the relevant filter pre-applied — *"3 overdue receipts · 2 blocked orders · 5 low stock · 1 supply request"*. The scheduler's topbar KPI strip (`scheduler:3-9`) is the model.

---

### IMP-031 — Make every click target a real, focusable control

**Rules:** ACCESS-003 (High, Hard rule) · ACTION-010 · ACTION-011 · INPUT-006 (Hard rule) · LAYOUT-007 · TOUCH-003 · ERROR-010
**Screens:** ~25 · **Importance:** High · **Effort:** M

Six element types carry click handlers with **no `tabindex`, no role, and no key handler**:

| Element | What it does | Evidence |
|---|---|---|
| `.lot-link` (a `<span>`) | **opens the lot panel — the main drill path** | `dashboard.js:935`, `1033`, `1120`, `1211`, `1264`; bound `1640-1647` |
| `tr.expandable` | expands a lot breakdown | `923`, `1007`, `1111`, `1196`, `1249`; bound `1624-1638` |
| `.order-row` | opens an order | `2319`; bound `2340-2347` |
| `.search-item` (a `<div>`) | **selects a global-search result** | `1530-1551`; bound `1563-1612` |
| `.er-product-option` (a `<div>`) | selects a product | `3526`; bound `3528-3533` |
| `.product-lot-row` | opens a lot | `1465`, `1478`; bound `1496-1500` |

Traceability adds `.search-item` with an inline `onclick` (`463`) and `.lot-pill` as a `<span>` (`411-421`); the scheduler adds `td.cell` (`:114`) and `.otbtn` (`:1345`).

**A keyboard-only user cannot open a lot, expand an inventory row, select a search result, or open an order** — the four most-used interactions in the dashboard. The same elements have no focus state, no button shape, and (for `.lot-link` and `.order-link`) swallow text selection because a click-drag ends in a `click` (ERROR-010).

**S-49 is the reference implementation:** `dashboard.js:3849` (`role="button" tabindex="0" aria-expanded`), `3863-3868` (Enter/Space with `preventDefault`), `dashboard.css:2300-2304` (`:focus-visible`).

**Fix:** convert `.lot-link` to `<button>` or `<a>`; give the row-level targets `tabindex`, a role, `aria-expanded`, and a key handler; give the dropdowns `role="listbox"`/`role="option"` with arrow-key navigation.

---

### IMP-032 — Add accessible names to the six icon-only buttons

**Rules:** ACCESS-010 (High, Hard rule) · ACTION-005 (Hard rule) · CHART-005
**Screens:** ~15 · **Importance:** High · **Effort:** S

A `title` does not override text content when computing the accessible name, so a button whose only content is a glyph is announced as that glyph:

| Control | Announced as | Evidence |
|---|---|---|
| Notes Edit / Delete | **"✎" / "✕"** | `dashboard.js:1732-1733` |
| Order expand | **"▸"** | `dashboard.js:2320` |
| Calendar previous / next | **"←" / "→"** | `index.html:76`, `78` |
| Theme toggle | **"☾"** | `index.html:40` |
| Graph zoom in / out | **"+" / "−"** (no `title` either) | `traceability.html:310-311` |

Also: `.collapsible-header` is a clickable `<div>` with no role, no `tabindex`, and no `aria-expanded` (`index.html:130`, `150`, `162`; `dashboard.js:910`, `1096`); and the Sankey SVG has no `role`, `aria-label`, or `<title>` (`sankey.html:350`).

**The correct pattern already exists in six places** — all four `.btn-close` (`aria-label="Close"`), `#navToggle`, both `.mini-calendar-nav` — and the scheduler does it best, naming the **record** in the label: `aria-label="Delete SO-1234 Granola 25 LB order line"` (`scheduler:1537`).

**Fix:** add `aria-label` to the six buttons; add `role="button"`, `tabindex="0"`, and `aria-expanded` to `.collapsible-header`; add `aria-hidden="true"` to `.chevron` and `.order-expand-caret` as `.supply-row-caret` already has (`3850`).

---

### IMP-033 — Bind Enter to the primary action; trap focus in modals

**Rules:** ACTION-007 (High) · INPUT-006 (Hard rule) · ACCESS-003 · ERROR-002
**Screens:** S-22, S-33, S-45, S-55, S-76, S-86, S-88 (7) · **Importance:** High · **Effort:** S

**Enter does nothing** in the Note modal (`index.html:452-503` — a `<div>`, not a `<form>`, no keydown handler), the ER modal (`404-441`, same), Order Detail edit mode (`dashboard.js:3256-3272` — inputs in table cells, no form), the disambiguation panel (`1387-1391`), and every scheduler form. In the ER modal this kills the fastest path entirely: type "Frank", Enter, Enter does nothing.

**No modal traps focus** — the overlays are plain `<div>`s and the page behind stays tabbable. Initial focus is set in two of three (`3570`, `4002`) and **not at all** in the Note modal (`1785-1802`); focus is restored on close in one of three (`3994`, `4007`).

**S-33 additionally has the commit buttons in the middle of the tab order:** `renderOrderEditActions` is emitted into the header at `3206`, before the line inputs (`3259-3266`) and the notes textarea (`3293`).

**Working correctly, as models:** the Supply Request modal is a real `<form>` with `type="submit"` (`index.html:343`, `390`); the allocation form likewise (`dashboard.js:2917`, `3164-3167`); Traceability binds Enter explicitly (`traceability.html:442-444`).

**Fix:** wrap the two modal bodies in `<form>` with a submit handler; add a focus trap and initial/restore focus to all three; move the Order Detail commit buttons after the notes field or replace them with one Save.

---

### IMP-034 — Give every field a persistent label with its unit

**Rules:** INPUT-002 (High, Hard rule) · LAYOUT-009 · ACCESS-010 · NAV-003
**Screens:** S-03, S-16, S-24, S-33, S-41, S-45, S-57, S-63 (8) · **Importance:** High · **Effort:** S

The rule is explicit that a placeholder *"can never be the sole identifier"*. Seven fields rely on one, or on nothing:

| Field | Identifier | Evidence |
|---|---|---|
| `#global-search` | placeholder only | `index.html:33` |
| `#orders-customer-search` | placeholder only | `index.html:223` |
| `#er-text-filter` | placeholder only | `index.html:267` |
| `#orders-status-filter` | **nothing** — no label, no `aria-label` | `index.html:205` |
| `#daily-entries-mode` | **nothing** | `index.html:138` |
| `#date-from` / `#date-to` | **nothing** — separated by the word "to" | `sankey.html:302`, `304` |
| `#lotSearch` | placeholder + a leading 🔎 | `traceability.html:276` |

Once Luz types a customer name into `#orders-customer-search`, nothing on screen says what that box filters.

**Also (S-45):** the ER modal in edit mode hides `#er-product-group` (`dashboard.js:3551`) so the modal titled *"Edit Expected Receipt #12"* **never names the product**. Luz edits an expected quantity without seeing what it is for.
**And (S-22):** the note modal's title is a generic *"Edit Item"* (`1787-1788`).

**Correct patterns already present:** `#supplies-search` has an `.sr-only` label (`index.html:311-314`); `#orders-dispatch-filter` has an `aria-label` (`218`); `#er-qty` is labelled *"Expected qty (lb)"* **with a format example** (`418-419`); `#supply-request-qty` has a live unit span (`362-366`); the scheduler's pin modal names the field, the unit **and** the record (`scheduler:1447-1450`).

**Fix:** add a visible or `.sr-only` label to the seven; render the product name in the ER edit modal; name the note in the modal title.

---

### IMP-035 — Stop showing raw API bodies to the user

**Rules:** ACCESS-004 (Medium) · ERROR-010 · FEEDBACK-008 · OTHER-007
**Screens:** ~17 · **Importance:** Medium (High in practice — it is the most-seen error surface) · **Effort:** S

`fetchAPI` and `fetchSalesAPI` throw `` new Error(`HTTP ${res.status}: ${body}`) `` with the **entire raw response body** (`dashboard.js:488`, `1919`).

`parseApiErrorMessage` exists (`2668-2685`) and correctly extracts `detail.message` — but it is called on only five paths (`2741`, `2806`, `2836`, `2984`, `3147`). **Seventeen other `catch` blocks concatenate `e.message` directly:** `901`, `970`, `1086`, `1181`, `1234`, `1299`, `1672`, `1751`, `1779`, `2182`, `2221`, `2272`, `2548`, `3381`, `3493`, `3725`, `3916`.

So Luz sees *"Failed to load sales orders: HTTP 500: {"detail":{"error_code":"DB_TIMEOUT","message":"…","trace_id":"…"}}"* where the rule asks for *"Lot not found"*. The extractor is already written; it simply is not used.

**Fix:** call `parseApiErrorMessage` in the remaining seventeen `catch` blocks, or move the extraction into the two fetchers so every thrown error is already clean.

---

### IMP-036 — Auto-refresh the operational lists; fix the last-updated timestamp

**Rules:** FEEDBACK-007 (High) · NOTIFY-005 · DATA-012 · FEEDBACK-008
**Screens:** ~20 · **Importance:** High · **Effort:** M

**Only two surfaces auto-refresh** — Recent Entries (`dashboard.js:626-631`) and Process Flow — and both do it by replacing `innerHTML` (IMP-007). Sales Orders, Expected Receipts, Supplies, Supply Requests, Notes, and all four inventory tables are fetched once on load and change only on a manual Refresh.

The rule's clause is explicit: *"SO lists, Expected Receipts, Today So Far… stay in sync between Luz and Arturo."* They do not. Luz can leave the Sales Orders tab open all afternoon while Arturo posts shipments through the Floor GPT and see none of it — in a multi-user ledger, exactly the condition that produces duplicate and conflicting transactions.

**And the timestamp lies.** `#last-refreshed` is set only at the end of `refreshAll`, after `await Promise.allSettled(ops)` (`4163-4171`). Two consequences: it ignores failures (nine of thirteen operations can reject and the header still reads *"Updated: 3:42:15 PM ET"*), and no section refresh updates it (`2237`, `3368`, `3693`, `3901`, `602` all leave it alone), so pressing the Orders Refresh at 4:15 leaves the header reading 3:42.

**Process Flow is the model:** it sets its timestamp from the real render, tracks `lastSuccessTime` separately, and shows a stale banner naming that time after three consecutive failures.

**Fix:** poll the operational lists on a visible-tab interval into a buffer, surfacing *"3 new — tap to refresh"*; set the timestamp per section from the actual response, marking any section whose last fetch failed.

---

### IMP-037 — Add a fixed spacing scale and a named type scale

**Rules:** LAYOUT-010 (High, Hard rule) · ACCESS-005 (High) · ACCESS-001 · LAYOUT-005 · DATA-004
**Screens:** all 91 · **Importance:** High · **Effort:** M

**No spacing scale.** `dashboard.css:37-38` defines only `--radius: 12px` and `--radius-sm: 8px`. Padding and gap values in use span **seventeen steps** — 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 32, 40 px — with no ratio: `.today-tile { padding: 14px 16px 16px }` (`356`), `.recent-entry-card { padding: 16px }` (`293`), `.note-card { padding: 12px 16px }` (`1012`), `.panel-header { padding: 14px 20px }` (`875`), `.supply-lot-card { padding: 9px 12px }` (`2371`). Nesting indentation is **three different values for one level**: 32 px (`730`), 24 px (`820`), 32 px dropping to 16 px (`2363`, `2434`).

**No type scale.** There is no named text-style set — thirteen distinct `font-size` values, and **seven different sizes for one structural level**: `.recent-entries-header h2` 22 px (`288`), `.supplies-page-header h2` 20 px (`2263`), `.production-detail-header h3` 17 px (`556`), `.panel-header h3` 15 px (`878`), `.order-section-heading h3` 15 px (`1928`), `.section > h2` 15 px (`337`), `.collapsible-header h2/h3` 14 px (`642`). Plus two synthetic weights — `650` (`560`) and `750` (`580`) — that a system font stack rounds to 700.

**And everything is `px`,** so browser text scaling has no effect on any text in the product (ACCESS-001).

**What is right and should be kept:** one system font stack plus one monospace stack for identifiers (`39-40`) ✓, and `font-variant-numeric: tabular-nums` correctly applied on ten selectors so digits align ✓.

**Fix:** define `--space-1..6` and `--text-title/-headline/-body/-subhead/-caption` in `rem`, set a root `font-size`, and replace the ad-hoc values. This is the foundation IMP-009 and IMP-057 both depend on.

---

### IMP-038 — Restructure the tab bar into primary navigation

**Rules:** NAV-004 (High) · NAV-008 · LAYOUT-003 · TOUCH-003 · NOTIFY-010
**Screens:** S-05 and every tab pane (~35) · **Importance:** High · **Effort:** L

`dashboard/index.html:47-55` — seven tabs: Operations, Recent Entries, Activity, Notes, Sales Orders, Expected Receipts, Supplies. These are not seven facets of one subject; they are seven separate areas with different data, different primary users, and different primary actions. The audit test asks *"Is any app section reachable only through a tab strip?"* — Sales Orders, Expected Receipts, Supplies, and Notes are reachable **only** here, and the site nav (`18-23`) carries only four page-level destinations.

Seven also exceeds the ~6 wide-screen and 5 phone limits, and `dashboard.css:262` implements the one behaviour NAV-008 forbids: `overflow-x: auto`. On a 390 px phone roughly three of seven tabs are visible and the rest are off-screen with no affordance.

**Fix:** promote the four record-owning sections into the site nav as top-level destinations, and keep a tab strip inside the Dashboard page for what genuinely is one subject — Operations / Recent Entries / Activity, all views of today's ledger. Add the IMP-030 counts to the promoted nav items.

---

### IMP-039 — Add a shared page header carrying global search to all four pages

**Rules:** SEARCH-001 (High) · LAYOUT-002 (Hard rule) · OTHER-010 · ACCESS-005
**Screens:** S-57…S-70 (14) · **Importance:** High · **Effort:** M

Global search exists **only on `index.html`** (`31-36`). The other three pages carry no search field: `sankey.html:282-288` and `process-flow.html` hold only a mini-calendar and a timestamp; `traceability.html:264-269` only a mini-calendar. From Material Flow or Production Lines, reaching global search means navigating back to the Dashboard — a full page load. The test is *"From any screen, can the user reach global search in one tap without leaving their context?"*

This also fixes the four-way duplication: the site nav's CSS is copy-pasted into `sankey.html:42-55`, `process-flow.html`, and `traceability.html:42-55`, with a duplicated `navToggle` inline script in each of the four files (`index.html:521`, `traceability.html:1398`, `sankey.html:864`), and `index.html` alone wraps the brand in `.site-nav-left` (`14-16`).

**Fix:** extract the site nav and app header into shared markup and a shared stylesheet linked by all four pages, carrying global search, the health badge, the theme toggle, and the last-updated slot.

---

### IMP-040 — Replace four small-set dropdowns and toggles with segmented controls

**Rules:** NAV-007 (High) · ACTION-001 (Hard rule) · ACTION-012 · NAV-004
**Screens:** S-08, S-16, S-24, S-41 · **Importance:** High · **Effort:** S

| Control | Options | Problem | Evidence |
|---|---|---|---|
| Production Calendar mode | 2 | a **button** whose label shows the mode you are *not* in; the current selection is never visible | `index.html:79`; `dashboard.js:676`, `683` |
| Daily Entries date mode | 2 | event-date vs entry-date — the distinction the whole timing model rests on — hidden in a `<select>` | `index.html:138-141` |
| Orders dispatch filter | 3 | `<select>` | `index.html:218-222` |
| ER status filter | 4 | `<select>` | `index.html:261-266` |

`.notes-filter-btn` (`dashboard.css:977-988`) is already the correct pattern in this codebase.

**Fix:** replace all four with segmented controls styled on `.notes-filter-btn`. Also split the mode out of `#orders-status-filter`, which currently mixes a **mode** (`dispatch_queue` — different endpoint, different columns, reveals a second control) with statuses (`index.html:206`; `dashboard.js:2244-2259`).

---

### IMP-041 — Add sortable columns to the four tables where the task demands it

**Rules:** DATA-008 (High) · DATA-009 · SEARCH-005
**Screens:** S-10…S-13, S-25, S-26, S-42, S-49, S-51, S-70 (10) · **Importance:** High · **Effort:** M

**No `<th>` anywhere carries a click handler, an `aria-sort`, or a sort affordance** (verified by grep), and there is no mobile sort control. Twelve tables, zero sortable columns. Luz cannot sort orders by ship date, or expected receipts by remaining quantity to find the biggest shortfall, or finished goods by cases to find what is nearly out.

**The fixed defaults are, in fairness, well chosen** — supply lots by lot date ascending for FIFO (`dashboard.js:3729-3738`), supply requests newest first (`3909-3911`), the Dispatch Queue by ready-then-ship-date (`2066-2072`), the scheduler order book by status priority (`scheduler:1519`). The gap is that nothing can be re-sorted.

Columns are also fixed-width and unresizable, so a clipped `.er-notes` (`max-width: 320px`, `dashboard.css:2217`) or a cramped blockers cell (`min-width: 190px; max-width: 320px`, `1325`) cannot be widened.

**Fix:** one shared `sortableTable()` helper with an `aria-sort` indicator, applied to Orders, Expected Receipts, Finished Goods, and Supplies; add column resize handles to the same four.

---

### IMP-042 — Trim the multi-line table rows

**Rules:** DATA-003 (High) · ACCESS-009 · LAYOUT-005 · DATA-006
**Screens:** S-14, S-25, S-26, S-31, S-33 (5) · **Importance:** High · **Effort:** M

The rule caps a list row at two lines. Three rows exceed it structurally:

- **Orders row** (`dashboard.js:2313`) — eleven columns, three of them multi-line: the Blockers cell wraps one chip per blocker in a `flex-wrap` container with `white-space: normal` on the detail (`css:1609-1632`), the Ship By cell stacks date and weekday (`1523-1527`), and the Status cell stacks a badge and a three-segment READY pill (`2229-2235`). Three blockers with details is four-plus lines.
- **Order Detail line row** (`3277`) — `renderLineReadiness(l, true)` renders every blocker's **full detail**, plus `.pallet-secondary` sub-lines under three numeric cells (`3261`, `3268`, `3274`). Five-plus lines.
- **Recent Entries card** (`589-598`) — topline, direction, optional correction, one `<li>` per transaction line, a two-column dates grid, and timing flags. Height unbounded.

`.readiness-chip { line-height: 1.25 }` (`1625`) makes those multi-line cells *tight*, which ACCESS-009 prohibits at three or more lines.

**Fix:** show the first blocker plus *"+2 more"* in the list view and reserve full detail for the detail view; move the weekday into the date string; collapse the READY pill to a check plus a tooltip.

---

### IMP-043 — Stop dimming Factory-Ready rows; move the KPI row above the line table

**Rules:** LAYOUT-005 (High, Hard rule) · LAYOUT-006 (High) · LAYOUT-004
**Screens:** S-25, S-26, S-30 (3) · **Importance:** High · **Effort:** S

**Browser check:** measured. `opacity: 0.6` on the Factory-Ready row takes `.so-ready-pill` to **1.16 : 1** in light, `.order-link` to 2.32 : 1 in dark, and the status badge, ship-by weekday, blockers cell and expand caret to 2.21–2.30 : 1 — the whole row below AA. The dim is not only a hierarchy inversion; it is a contrast failure. The same device on genuinely inactive rows fails too — see IMP-067. ([06](06-browser-check.md))

**(a)** `dashboard.css:1320` — `.orders-table tr.order-row.so-ready { opacity: 0.6 }`. Orders that are Factory Ready are dimmed to 60 %. A ready order is not a finished order; it is the one about to ship. This inverts the hierarchy for exactly the rows Luz is scanning for, and compounds the ACCESS-008 contrast margins. (Contrast the correct uses of the same device — `.er-inactive` (`2220`), `.allocation-status-released` (`2063`), `.supply-request-done` (`2402`), `.note-card.done` (`1020`) — all genuinely inactive.)

**(b)** `dashboard.js:3226-3231` **builds** the KPI row (Total Ordered / Shipped Effective / Remaining Effective / Pallets) and `3285` **appends it after** the line-items table closed at `3282`. The four headline numbers of the record sit below a nine-to-ten-column table. The variable is even named `summaryHtml` and constructed before the table.

**Fix:** (a) replace the dim with a green left border and the existing pill; (b) move line `3285` to immediately after `3209`.

---

### IMP-044 — Stop the fabricated sample-data fallback in Sankey and Process Flow

**Status:** **DONE** · branch `fix/error-surfacing`, four commits — `c972526` traceability's status bar renders API-controlled text with `textContent`/DOM nodes instead of `innerHTML` (`setStatus`, `suggestSimilar`, `showDisambiguation`; ERROR-010); `028c685` both chart pages' `apiFetch` now read and keep the response body on `!res.ok` (`HTTP <status> <path>: <body>` plus `error.status` / `error.body` / `error.payload`, the shape `fetchSalesAPI` already uses) and the banners show it verbatim in a selectable monospace block with a Copy button (ERROR-010); `684af02` deleted `getFallbackData()`, `getFallbackProductDetails()`, `useFallback`, and `getFallbackLinks()` — a failure with no prior successful load now renders a named failure state with the verbatim error and Retry, and sankey tracks `lastSuccessTime` so a failed refresh (including the debounced resize refetch) keeps the real chart under a *“Refresh failed — showing data from &lt;time&gt;”* banner instead of redrawing it; `a6ae9d3` `supplyApiErrorMessage` keeps the `HTTP <status>` prefix when unwrapping a structured error (ERROR-010).

**Note on IMP-035.** IMP-035 (*“Stop showing raw API bodies to the user”*) is **not** done and is in partial tension with the second commit here: on Sankey and Process Flow the verbatim body is now shown *deliberately*, because ERROR-010 asks for an error that can be *“pasted to Claude Code verbatim”* and these two pages have no other diagnostic surface. When IMP-035 is taken up, the resolution to aim for is a plain-language headline extracted via `parseApiErrorMessage` with the raw body kept behind the existing copyable detail block — not deleting the body.

**Not covered here:** the other ERROR-010 findings from [03-feedback-error-notify.md](03-feedback-error-notify.md) — the native `alert()`/`confirm()` bodies (IMP-014), the `.lot-link` / `.order-link` click handlers that swallow a click-drag selection (IMP-031), the `pointer-events: none` node tooltip (S-69), and the absence of a copy affordance on lot codes and SO numbers.

**Rules:** ERROR-002 · FEEDBACK-008 · CHART-001 · OTHER-003 (Hard rule) · OTHER-007
**Screens:** S-57, S-58, S-59, S-60, S-61, S-62 (6) · **Importance:** High · **Effort:** S

On an API failure both pages render **fabricated numbers as though they were data**:

- `sankey.html:824-828` — `catch (err) { showBanner('API unreachable — showing sample data', 'warning'); const links = getFallbackLinks(); renderSankey(links); }`. The chart then draws hard-coded flows (*"Graham Crumb 25lb → DOT Foods, 2,400"*) with the same styling as real data, behind a small banner.
- `process-flow.html` — `getFallbackData()` and `getFallbackProductDetails()` render invented lots (`GH-2026-0316`, `CS-2026-0087`) and yields on the first load if no successful fetch has occurred.

On an operational ledger this is worse than an error state: Blubber can read a plausible number off a chart that is describing nothing. The Sankey case is also reachable from a window **resize**, which triggers a debounced refetch (`sankey.html:851-856`).

**Fix:** replace both fallbacks with an explicit empty state naming the failure and offering Retry. Process Flow already does the right thing on *subsequent* failures (`else { return; // Keep showing last good data }`); the fallback path should be deleted rather than extended.

---

### IMP-045 — Give the Sankey a takeaway, an accessible table, and drill-through

**Rules:** CHART-004 (High) · CHART-005 (Hard rule) · CHART-008 · ACCESS-010 · SEARCH-008
**Screens:** S-58 · **Importance:** High · **Effort:** M

Three gaps on one screen:

- **No takeaway sentence.** The title names the chart type (*"Product Flow — Sankey"*, `sankey.html:283`) and the footer says only *"All volumes shown in pounds."* (`354-357`). Every input for a real headline — top ingredient, top customer, total volume — is already computed in `processData`.
- **No accessible values.** `<div id="sankey-chart">` (`350`) is filled with D3-generated SVG carrying no `role`, no `aria-label`, no `<title>`/`<desc>`, no adjacent table, and no "view as table" toggle. Up to ~40 link values exist only as SVG geometry and a `pointer-events: none` tooltip. CHART-005 is a **hard rule**.
- **No drill-through.** Hovering shows a value (`667-696`) but nothing is clickable — there is no `.on('click')` in the render. From a flow of 12,400 lb there is no path to the transactions behind it.

**Traceability solves all three and is the model:** the completeness badge is the takeaway (*"⚠ Partial trace — 2 legacy lot(s), 1 unknown supplier(s)"*, `1223-1249`); `renderDetailPanel` renders the identical data as a real `<table>` below the graph (`1254-1297`); and `onNodeClick` drills sideways into another trace (`1177-1194`).

**Fix:** add a headline line, render an adjacent `<table>` of the links, and make nodes and links clickable through to the underlying transactions.

---

### IMP-046 — Adopt an accessibility checklist and write down the role → workflow map

**Rules:** ACCESS-002 (High, Hard rule) · OTHER-006 (High, process) · OTHER-001 · OTHER-003
**Screens:** all 91 · **Importance:** High · **Effort:** S (process)

**No accessibility checklist exists** — `docs/design/` holds only the standards and this audit; `CLAUDE.md` has no accessibility section; there is no axe, Lighthouse, or pa11y configuration and no CI check. The code shows the signature of a per-screen retrofit: the **Supplies tab** has roles, `aria-selected`, `aria-expanded`, keyboard handlers, focus-visible styling, `aria-live` regions, and `aria-label`s updated on every render — while Orders, Expected Receipts, Notes, the inventory tables, and the lot panels have almost none. `.sr-only` is defined once (`dashboard.css:2244-2254`) and used **once** (`index.html:312`).

**No role → workflow map exists either.** The screen inventory records that *"the repo has no written statement of who uses which dashboard screen"* and that every Primary-user attribution is inferred — which is also why OTHER-003 has nothing to scope against.

The OTHER-006 testing clause is unverifiable from code, but six of its named conditions would have failed on first contact: the ≤768 px header overlap, 200 % zoom, light mode (`.so-ready-pill` at 1.27 : 1), Spanish labels (fifteen `nowrap` declarations), dead-zone Wi-Fi (no timeout), and arm's-length legibility (8 px calendar text).

**Fix:** an eight-item pre-ship checklist covering exactly what this audit found repeatedly — 44 px targets, focus-visible, accessible name on every icon-only control, keyboard operability of every click target, contrast in **both** themes, no operational text below 12 px, colour never the sole carrier, a label on every field — with the Supplies tab as the worked reference. Plus a one-page role → workflow map.

---

### IMP-047 — Add URL state for tab, order, lot, and product

**Rules:** SEARCH-008 (Medium) · NAV-001 (High) · ERROR-002 (Critical) · NAV-009 · FEEDBACK-009
**Screens:** ~40 · **Importance:** Medium *(supports a Critical rule)* · **Effort:** M

There is **no URL state anywhere in `dashboard.js`** — grep for `history`, `pushState`, `replaceState`, `location.hash`, and `searchParams` returns zero matches. `initTabs` toggles classes (`494-508`); `openOrderDetail` (`2526-2550`), `openLotPanel` (`1344`), and `openProductPanel` (`1438`) touch nothing.

Three consequences: **nothing can be shared** (a WhatsApp message cannot link to SO-1234; a QR label cannot open its lot); **the back button leaves the site** rather than returning to the list; and **a refresh always lands on Operations**, losing the open record and compounding IMP-008.

**Traceability is the reference:** `updateURL()` writes `?lot=…&direction=…&product_id=…` via `history.replaceState` (`1365-1372`) and `loadFromURL()` restores the state and auto-runs the trace (`1374-1387`).

**Fix:** adopt the same pattern — `#tab=orders&order=1234`, `#lot=26-0907-C` — read on load and written on every open.

---

### IMP-048 — Standardise the expand/navigate affordances and the lot row anatomy

**Rules:** NAV-010 (Medium) · DATA-007 (Medium) · LAYOUT-002 (Hard rule) · ICON-004 · ICON-002
**Screens:** ~20 · **Importance:** Medium · **Effort:** M

**Five affordances for "expand in place":** `.chevron::after` = ▶ (`dashboard.css:650-657`), `.order-expand-caret` = ▸ (`dashboard.js:2320`), `.supply-row-caret` = ▸ (`3850`), a text button *"Inventory"* (`3257`), a text button *"Show all (N more)"* (`1156`) — and **no affordance at all** on the four inventory tables' `tr.expandable` rows (`923`, `1007`, `1111`), whose only cue is `cursor: pointer`. Meanwhile "go deeper" (opening an order or a lot) has **no glyph anywhere**, so the conventional chevron is assigned to the wrong meaning.

**Seven anatomies for the lot row** — the product's most-repeated record: `.lot-row` in three inventory tables with 4/3/2 columns (`935`, `1033`, `1120`), inline text with `<br>` in shipment and receipt details (`1211`, `1264`), two inline-styled tables in the product panel (`1465`, `1476`), a `<dl>` card in Supplies FIFO (`3785-3793`), and a `<strong>`+`<span>` inside an allocation cell (`2941`). None carries a status indicator or a trailing accessory, and the quantity appears in four different formats.

**Fix:** one glyph per meaning — `▸`/`▾` for disclosure, a trailing `›` for navigate, nothing else — added to the four inventory tables; and one lot-row partial used everywhere: status dot · lot code · product/qty · accessory.

---

### IMP-049 — Fix the table striping and the nested `<tbody>`

**Rules:** DATA-006 (Medium) · LAYOUT-002 · ACCESS-008
**Screens:** S-10, S-11, S-12, S-13, S-17, S-18 (6) · **Importance:** Medium · **Effort:** S

**(a) The shading never appears on the Shipping and Receiving logs.** `dashboard.css:775` shades `.activity-table tr:nth-child(even)`, but `renderShipments` and `renderReceipts` emit **two** `<tr>` per record — a data row and an `.activity-detail` row (`1196`/`1204`, `1249`/`1258`). So data rows are always odd children and the even children are the detail rows, which are `display: none` (`css:818`) and set their own background when visible (`820`). **No visible row in either five-column log is ever shaded.**

**(b) The inventory tables emit invalid markup.** `dashboard.js:918` opens `<tbody>`, then per product emits a `<tr>` followed by a **`<tbody>` nested inside the open one** (`930`); same at `1019` and `1115`. Browsers recover by implicitly closing the outer `<tbody>`, so the DOM does not match the source and `tr:nth-child(even)` (`css:698`) counts unpredictably.

**(c)** `.order-lines-table` (`1376-1400`) and `.allocation-table` (`2029-2054`) define no striping at all — and the latter is explicitly `min-width: 820px`, a wide table with no row-tracking aid, which is the rule's primary target.

**Fix:** switch to a class applied by the renderer (`.row-alt` on every other **data** row) rather than `nth-child`; replace the nested `<tbody>` with a sibling `<tr class="lot-breakdown-row">`; add striping to the two wide tables.

---

### IMP-050 — Unify the date/time formats; add elapsed values and lot age

**Rules:** DATA-012 (Medium) · FEEDBACK-007 · ACCESS-004
**Screens:** ~25 · **Importance:** Medium · **Effort:** M

**Thirteen date/time renderings coexist:** `09/07/26` (`dashboard.js:1971-1976`), `Sep 7, 2026` (`511-518`), `Sep 7, 2026, 3:42 PM EDT` (`520-531`), `Sep 7, 2026, 3:42 PM ET` (`3872-3880`), `Monday, September 7, 2026` (`804-810`), `3:42 PM` (`1993-2002`), `3:42:15 PM ET` (`4166-4170`), raw `2026-09-07` (`768`), raw server strings (`1197`, `1250`, `1325`), Traceability's `dateStyle: 'medium'` (`traceability.html:1263`) and its long export form (`1310`), the scheduler's `Mon 8`, and the Sankey's bare `toLocaleTimeString()` with **no timezone** (`sankey.html:822-823`). Four different formats appear on the Operations tab alone.

**Time zones are handled with real care in two places** — `formatBusinessDate` anchors business dates at `T12:00:00Z` with an explanatory comment (`511-518`) ✓ and `formatEnteredAt` refuses to invent an offset for a naive value (`522-524`) ✓ — and loosely in three: `getLocalDateFromISO` derives the ship-date weekday from a **browser-local** Date (`1978-1991`), and both `getCalendarParams` (`644-667`) and `process-flow.html`'s `toET()` round-trip through `new Date(x.toLocaleString(...))`, an implementation-dependent parse. `getCalendarParams` also contains **dead code**: its `endDate`, `startDate`, and `fmt2` locals (`659-663`) are never used because the function returns `` `days=${totalDays}` `` at `666`.

**Live values:** one of six candidates updates itself — `updateAllocationCountdowns` (`3153-3157`, `3177`) ✓. `#last-refreshed`, the supply-request time, the recent-entry lag chip, and the stale-banner time are all fixed at render, and **there is no lot age anywhere**, though the rule names it directly.

**Fix:** one `formatDate`/`formatDateTime` pair used everywhere; replace the two `toLocaleString` round-trips; delete the dead branch in `getCalendarParams`; add a self-updating relative renderer and a lot-age column.

---

### IMP-051 — Add next/previous between sibling records

**Rules:** NAV-011 (Medium) · NAV-009 · FEEDBACK-009 (High)
**Screens:** S-30, S-54, S-56 (3) · **Importance:** Medium · **Effort:** M

Order Detail offers only *"← Back to Orders"* (`index.html:247`); the lot and product panels offer nothing (`dashboard.js:1400-1432`, `1447-1504`). Luz's dispatch-queue workflow is explicitly sequential — filter to blocked orders, open one, read the blockers, go back, open the next — and every record costs a round-trip through a list that is rebuilt on return.

Related (FEEDBACK-009): opening an order **replaces** the list (`2531`), so there is no "currently open" item to keep highlighted, and opening a lot panel leaves the source row unmarked behind a 440 px overlay. The Production Calendar shows the correct behaviour: `.day-card-trigger.selected` keeps an inset ring and `aria-expanded` while the detail is open (`854-859`; `css:442-447`).

**Fix:** add `‹ Prev / Next ›` to the Order Detail header scoped to the current filtered list, and the same to the lot panel scoped to the list it was opened from; highlight the source row while a panel is open.

---

### IMP-052 — Add find-within-view to the four long lists

**Rules:** SEARCH-007 (Medium) · SEARCH-005 · NAV-012
**Screens:** S-14, S-17, S-18, S-51 (4) · **Importance:** Medium · **Effort:** S

| List | Rows | Filter available |
|---|---|---|
| Shipping log | up to 100 | **none** — and 96 are `display: none` behind "Show all", so the browser's own find cannot reach them (`dashboard.css:822`) |
| Receiving log | up to 100 | **none**, same |
| Supply requests | up to 500 | **none** — only a show-completed checkbox (`index.html:324-327`) |
| Recent Entries | 20 | none |

`#er-text-filter` shows the right pattern: one field searching product, SKU, supplier, reference **and** notes (`dashboard.js:3391-3393`) ✓.

Also (S-24, S-25): `#orders-customer-search` filters by **customer only** (`2050-2052`), so within 200 loaded orders there is no way to find a specific SO number without the global search, which navigates away.

**Fix:** add an `#er-text-filter`-style field to the three logs and the supply requests list; extend the orders filter to cover SO number and product.

---

### IMP-053 — Add expand-all / collapse-all

**Rules:** DATA-010 (Medium) · NAV-012
**Screens:** ~10 · **Importance:** Medium · **Effort:** S

Expandable outlines are used well and widely — FG product → lots (`dashboard.js:923-941`), batch → lots (`1007-1039`), ingredient → lots (`1111-1126`), order → lines (`2490-2524`), line → inventory (`2638-2662`), supply item → FIFO lots and incoming receipts (`3853-3855`), transaction → lot detail (`1204`, `1258`) — but **there is no expand-all or collapse-all control anywhere**. The rule states the need directly: *"Expand-all for recall investigations."* A recall today means opening every lot row one at a time.

Related: allocations are a **separate flat table** below the line table (`2936-2955`) rather than nested under their line, linked only by a `Line #12` span (`2947`).

**Fix:** an Expand all / Collapse all pair in each section header, and nest allocations under their line.

---

### IMP-054 — Add `type="search"`, a clear control, and a leading magnifier

**Rules:** INPUT-012 (Medium) · INPUT-013 (Medium) · INPUT-010 (High) · SEARCH-004
**Screens:** S-03, S-04, S-24, S-41, S-45, S-63 (6) · **Importance:** Medium · **Effort:** S

Five of six search and filter fields are `type="text"`, so they get neither the user agent's clear button nor the search keyboard on mobile: `#global-search` (`index.html:33`), `#orders-customer-search` (`223`), `#er-text-filter` (`267`), `#er-product-search` (`407`), `#lotSearch` (`traceability.html:276`). Only `#supplies-search` uses `type="search"` (`index.html:313`) ✓ — a one-character precedent in the same file.

The leading-purpose / trailing-function convention exists in exactly one place: Traceability's leading 🔎 at `left: 11px` inside a field padded to match (`traceability.html:275`, `87-90`) ✓. The dashboard's four fields have no leading icon and no trailing control.

Minor related inconsistency: the global search box is cleared after selecting a product, order, or customer (`dashboard.js:1576`, `1586`, `1600`) but **not** after selecting a lot (`1563-1568`).

**Fix:** change the five to `type="search"`, add a leading magnifier to each, and clear the box on lot selection too.

---

### IMP-055 — Reduce Orders-toolbar and Order-Detail density

**Rules:** LAYOUT-001 (High) · LAYOUT-021 · CHART-002 · TOUCH-004
**Screens:** S-24, S-30, S-33, S-36 (4) · **Importance:** High · **Effort:** M

The Orders toolbar carries **eight controls** above the table — status select, dispatch select, customer search, two checkboxes, a summary span, and three buttons, two of which are exports (`index.html:203-237`). Exports are a monthly task competing for space with a daily one, and at ≤768 px `.orders-toolbar-actions` keeps `display: flex` with no column direction in the media block (`css:1237-1242`, `2194`), so three text buttons stay in one row on a phone (TOUCH-004 allows two).

`renderOrderDetail` (`dashboard.js:3180-3306`) emits, in one pass and all expanded: header, readiness card with blocker chips, a 9–10 column line table with a per-line inventory expander, a 4-tile KPI row, a four-field allocation form, an allocation history table, a shipping-preview section, and a notes card. Nothing is progressively disclosed.

**Fix:** move the two exports behind a "···" overflow; collapse Reservations and Shipping Preview behind disclosures that open on demand; give `.orders-toolbar-actions` a column direction at ≤768 px.

---

### IMP-056 — Let dense tables use the full window width

**Rules:** LAYOUT-011 (High) · LAYOUT-019 · DATA-006
**Screens:** S-25, S-26, S-27, S-42 (4) · **Importance:** High · **Effort:** S

`dashboard.css:282` — `.tab-content { max-width: 1200px; margin: 0 auto }` applies the reading-width cap to **every** tab pane, including the dense tables LAYOUT-019 explicitly exempts. On a 2560 px monitor an 11-column orders table is squeezed into 1200 px with 680 px of dead margin each side, while `.order-blockers-cell { min-width: 190px; max-width: 320px }` (`1325-1328`) fights for room inside it.

Prose is correctly constrained elsewhere — `.recent-entries-page { max-width: 760px }` (`286`), `.detail-panel { max-width: 1000px }` (`traceability.html:185`) ✓ — so the intent exists; it is simply applied at the wrong level.

**Fix:** set `max-width: none` on `#section-orders` and `#section-expected` (and their toolbars) while leaving the cap on the prose sections.

---

### IMP-057 — Fix the surface-level inversion; reduce to three levels

**Rules:** LAYOUT-017 (Medium) · OTHER-010 · ACCESS-008
**Screens:** ~20 · **Importance:** Medium · **Effort:** S

`dashboard.css:6-33` defines **seven** surface tokens, two pairs of which are byte-identical duplicates: `--surface-alt #1a2536` equals `--row-alt #1a2536`, and `--row-header #162032` equals `--panel-header-bg #162032`. That leaves four distinct levels against the rule's limit of three.

Worse, the levels do not nest monotonically: `--surface-alt` (#1a2536) and `--row-header` (#162032) are **darker** than `--surface` (#1e293b), so going one level deeper moves the surface *toward* the page background, inverting the containment cue. `traceability.html:188` adds a fifth, hard-coded `.detail-card { background: #16213e }`.

**Fix:** collapse to `--surface-1/2/3` with monotonic lightness in both themes; delete the two duplicate pairs and the hard-coded fifth.

---

### IMP-058 — Icon hygiene: the decorative hamburger and the emoji empty states

**Rules:** ICON-001 (High, Hard rule) · ICON-002 (Hard rule) · ICON-003 · ICON-005
**Screens:** S-20, S-21, S-29, S-44, S-51, S-58, S-59, S-87 (8) · **Importance:** High · **Effort:** S

**Browser check:** the point in (b) is confirmed and it defeats the measurement. The three emoji empty-state glyphs do render at `opacity: 0.5`, but a colour-emoji glyph paints its own pixels, so the computed `color` a contrast ratio is derived from never reaches the screen. The browser check records these separately and does not count them as ACCESS-008 failures — which is precisely the asymmetry this improvement describes: the same `opacity` rule that de-emphasises a monochrome glyph leaves a colour bitmap fully saturated. Their legibility needs an eye, not a ratio. ([06](06-browser-check.md))

**(a)** `sankey.html:283` — `<h1><span class="icon">&#9776;</span> Product Flow — Sankey</h1>`. The hamburger `☰` is used as a decorative "flow lines" glyph in the page title, **twenty pixels below the same glyph functioning as the menu button** (`274`). ICON-001 prohibits exactly this, and it collides with a conventional meaning.

**(b)** Four incompatible icon families are mixed, sometimes in one view: dingbat Unicode (`☰ ▶ ▸ × ← → ‹ › ✎ ✕ ✓ ✗ ⚠ ⧉ ⌫ ∅`), **full-colour emoji** (`📝 📦 🚛 🔍 🔎 🖨 💾`), one inline SVG (the process-flow arrow), and CSS-drawn shapes. The Traceability header pairs a colour 🔍 with monochrome ✕ and ⚠; its export row pairs colour 🖨 and 💾 with monochrome text buttons. The emoji are rendered at 32 px with `opacity: 0.5` (`dashboard.css:1214-1218`, `2171-2175`), which does not affect a colour bitmap glyph the way it does monochrome text, so they read as fully saturated illustrations in an otherwise monochrome interface.

**(c)** `▶` (`css:651`) and `▸` (`dashboard.js:2320`, `3850`) are two triangles for one meaning, both rendered at 10 px where they are indistinguishable.

**Fix:** replace the Sankey title glyph; swap the emoji for monochrome glyphs from the same dingbat family (or a small inline-SVG set); settle on one triangle.

---

### IMP-059 — Convert the two single-line note fields to textareas

**Rules:** INPUT-001 (Medium) · INPUT-004
**Screens:** S-28, S-36 · **Importance:** Medium · **Effort:** S

`dashboard/dashboard.js:2362` — the Factory Ready note is an `<input type="text">` labelled *"Factory Ready note"* with placeholder *"Optional note for the floor"*. Free-form prose in a single-line field; the display renderer confirms multi-line was expected: `.order-ready-note-text { white-space: pre-wrap; word-break: break-word }` (`css:1502-1509`).

`dashboard/dashboard.js:2925` — the allocation note is `<input type="text" maxlength="500" placeholder="Why this stock is reserved">`. Five hundred characters of explanation in a single-line field.

Every other note field in the product is correctly a textarea (`index.html:370`, `434`, `467`; `dashboard.js:3293`) ✓.

**Fix:** convert both to `<textarea rows="2">`.

---

### IMP-060 — Confirm only consequential SO status transitions

**Rules:** ERROR-005 (High) · ERROR-006 · ACTION-008
**Screens:** S-35 · **Importance:** High · **Effort:** S

`dashboard/dashboard.js:2818-2823` confirms **every** status transition, including benign forward moves like New → Confirmed. The rule warns that routine confirmations *"train people to tap through without reading — defeating the confirmations that matter."* The one that matters here is Ready → Cancelled.

The product otherwise does **not** over-confirm — all five existing confirmations protect something genuinely consequential ✓ — and this message is the best-written dialog title in the codebase (*"Change SO-1234 status from Ready to Ship to Cancelled?"*, naming the record and both states).

**Fix:** confirm only on backward transitions and on `cancelled`; let forward moves proceed with an inline success message.

---

### IMP-061 — Add a help control to the dashboard

**Rules:** OTHER-009 (Low) · LAYOUT-009 · OTHER-004
**Screens:** ~50 · **Importance:** Low · **Effort:** M

There is **no help control anywhere in the dashboard** — no "?", no help link, no documentation entry point on any of its 56 screens. Concepts that need one: the LAT lot-code policy; **Factory Ready vs Dispatch Ready** (two readiness concepts on the same row, one human-set and one computed, `dashboard.js:1948-1952` vs `2229-2235`); the three allocation modes and the 48-hour auto-FIFO TTL; what "Effective Remaining" and "Shipped Effective" mean; why an expected receipt auto-links by product + supplier.

Several are explained inline as prose ✓ (`index.html:281`, `293`; `dashboard.js:2896`, `2914`, `2963`) — good writing, but caption text competing for space rather than a control the user can reach on demand.

**The scheduler is the reference:** a consistent `.info` control immediately after each label it explains — five on the KPI strip (`scheduler:3-9`), one on the pin modal's quantity field (`:1448`), one on the order-book header (`:1525`) — each with a plain-language `title` and a delegated click handler that renders it as a positioned popover so it works on touch (`:1583-1597`) ✓.

**Fix:** adopt the scheduler's `.info` control on the dashboard's ten ambiguous labels.

---

### IMP-062 — Add `prefers-reduced-motion` and `prefers-contrast` support

**Rules:** ACCESS-008 (Critical) · OTHER-010 (Hard rule) · OTHER-007
**Screens:** all 91 · **Importance:** Critical *(the increased-contrast clause)* · **Effort:** S

**Browser check:** the ACCESS-001 half is now partly settled — at 200 % zoom no dashboard screen overflows, but six screens strand their last actionable row behind the sticky stack (IMP-066). The `prefers-reduced-motion` and `prefers-contrast` clauses were not exercised: the harness runs with `reducedMotion: 'reduce'` throughout precisely so captures are stable, which tests nothing about honouring it. ([06](06-browser-check.md))

There is no `@media (prefers-contrast: more)`, no `forced-colors` handling, and no `@media (prefers-reduced-motion)` in any of the five style sources (verified by grep). ACCESS-008 requires AA *"in light, dark, and increased-contrast modes"* and OTHER-010 requires all three token variants *"even if only one mode ships today."*

Motion that would need the guard: `scrollIntoView({ behavior: 'smooth' })` at `dashboard.js:887` and `traceability.html:1142`, the D3 zoom transitions (`traceability.html:1201-1211`), and roughly forty CSS transitions.

**Fix:** add a `prefers-contrast: more` token block raising `--text-muted`, `--text-dimmed`, and every border to full-contrast values, and a `prefers-reduced-motion: reduce` block setting `transition-duration: 0.01ms` and `scroll-behavior: auto`.

---

### IMP-063 — Cap and restructure the two disambiguation choice lists

**Rules:** ERROR-008 (High) · ERROR-007 · ERROR-009 · ACTION-003
**Screens:** S-55, S-67 · **Importance:** High · **Effort:** S

`dashboard/dashboard.js:1372-1391` — `renderLotDisambiguation` loops `for (const m of matches)` with **no cap**, rendering one button per matching product into `#lot-panel-body`, which is `overflow-y: auto` (`css:891-895`). Five matches means five choices in a scrolling panel; ERROR-008 caps a choice dialog at three plus Cancel and prohibits scrolling.

`dashboard/traceability.html:537-553` — `showDisambiguation` does the same into `#statusBar`, a flex container with `min-height: 44px` and **no scroll** (`171-174`), so the buttons overflow the bar entirely. It is also a choice arising from an action presented in a status region rather than a dialog (ERROR-009).

**Fix:** show the three most likely matches with the rest behind *"N more products…"*; move Traceability's version into a proper dialog; add a Cancel to both. (The `--bg-card` fix is IMP-001.)

---

### IMP-064 — Add a context label to the notes list and the depleted-lots table

**Rules:** DATA-005 (Medium) · NAV-012 · DATA-006
**Screens:** S-19, S-20, S-56 (3) · **Importance:** Medium · **Effort:** S

`dashboard/index.html:193` — `<div id="notes-list"></div>` has no heading, no count, and no context label. The rule requires *"a section label saying what it lists and how many."* Contrast `#supply-requests-list`, headed *"Supply Requests (3)"* (`322`) ✓, and `#recent-entries-feed`, headed with a title, a subtitle, and a live count (`115-120`) ✓.

`dashboard/dashboard.js:1476` — the depleted-lots table in the product panel opens straight into `<tr>` data rows with **no header row at all**, while the active-lots table immediately above it has one (`1463`). Three unlabelled columns of lot data at `opacity: 0.6`.

`dashboard/traceability.html:286` — the recent-lots strip is labelled only *"Recent:"*, with no count and no statement of what "recent" means (the last 20 lots from the last 100 transactions).

**Fix:** add *"Notes (12)"* above the notes list, a header row to the depleted-lots table, and a count and scope to the recent-lots label.

---

### IMP-065 — Make the two print views printable

**Status:** New — raised by the rendered check ([06](06-browser-check.md)), not visible from the code.

**Rules:** ACCESS-008 (Critical) · LAYOUT-003 (Critical) · FEEDBACK-011 · OTHER-006
**Screens:** S-71, S-91 (2) · **Importance:** Critical · **Effort:** S

Both print stylesheets restyle the *page* to white and leave the *text* on its dark-theme colour. Measured
under `@media print`:

| Screen | Element | Text | Composited ratio | Target |
|---|---|---|---:|---:|
| S-71 | `#traceDetail > h3` | the report's own title, *"Forward Trace (Ingredient → Batches → Customers)"* | **1.1 : 1** | 4.5 |
| S-71 | `#traceDetail … strong` | the traced lot code, `OAT-4471` | **1.1 : 1** | 4.5 |
| S-91 | `#topbar h1 span` | *"Production Board"* | **1.95 : 1** | 4.5 |
| S-91 | `#kpi-risk`, `#kpi-late` | Orders at risk, Late days | **2.29 : 1** | 4.5 |

`rgb(241, 245, 249)` on white paper. **The recall/audit report prints without its own heading and without the
lot code it is about** — the two things an auditor reads first, and the reason the screen exists. The table
body below them prints correctly, which is why nothing in the code review caught it.

Separately, S-91 lays out to **4,779 px against a 1,440 px page — 3,339 px of horizontal overflow**. The board
does not fit the paper it is printed on; the right-hand days are cut off.

**Fix:** in both `@media print` blocks, set an explicit ink colour on the elements the block re-grounds rather
than inheriting the screen token (`traceability.html:240-246`; `scheduler:185-208`), and give the scheduler
board a print width that fits — landscape `@page`, a scale transform, or column paging.

---

### IMP-066 — Reclaim viewport height at 200 % zoom

**Status:** New — raised by the rendered check ([06](06-browser-check.md)).

**Rules:** LAYOUT-011 (Critical wherever a fixed bar can hide an actionable row) · ACCESS-001 (High, Hard rule) · LAYOUT-015
**Screens:** S-12, S-30, S-32, S-33, S-39, S-52 measured; every long dashboard screen is exposed · **Importance:** Critical · **Effort:** M

IMP-002 fixed the *overlap* between the three sticky bars, and the browser check confirms that fix holds: at
390 px and at 200 % zoom the bars now stack correctly and never sit on top of one another. What IMP-002 did not
change is how much of the viewport they consume together.

At 200 % zoom a 1440 × 900 window exposes a **720 × 450 CSS-px** viewport. `.site-nav` + `.app-header` +
`.tab-bar` take a large share of those 450 px, and in twelve captures the last actionable element of a scroll
region could not be brought clear of them at **any** scroll position — verified by re-probing each candidate
after parking it at 60 % of the viewport:

| Screen | Element stranded | Behind |
|---|---|---|
| S-12 | the last row of the Batch Inventory table | `.app-header` |
| S-30, S-32, S-39 | **Preview all remaining lines** | `.tab-bar` |
| S-33 | the order notes `<textarea>` in edit mode | `.tab-bar` |
| S-52 | the Supply Requests **Refresh** button | `.app-header` |

ACCESS-001 requires that at 200 % zoom *"every task is still completable"*. Committing an order note is not.

**Fix:** collapse the header to a single compact row below a height threshold, or make the sticky stack
`position: static` under `@media (max-height: 560px)`, so vertical space goes to content. Overlaps with IMP-038
(restructure the tab bar) and IMP-055 (reduce Order-Detail density) — do them together.

---

### IMP-067 — Replace the opacity dim on inactive rows with a token that still meets AA

**Status:** New — raised by the rendered check ([06](06-browser-check.md)).

**Rules:** ACCESS-008 (Critical, Hard rule) · FEEDBACK-011 · LAYOUT-005
**Screens:** S-20, S-25, S-26, S-27, S-28, S-30, S-31, S-42, S-43 (9) · **Importance:** Critical · **Effort:** S

Four selectors mark "inactive" by dropping the whole row's `opacity`. IMP-043 argues one of them is applied to
the wrong rows; this is the separate point that **the device itself puts ordinary reading text below AA**, and
it does so on the rows that are genuinely inactive too. Measured composited ratios:

| Selector | Opacity | Worst measured | Target | Example |
|---|---:|---:|---:|---|
| `.note-card.done` (`css:1020`) | 0.55 | **1.55 : 1** light · 1.85 : 1 dark | 4.5 | *"supplier: Midstate Packaging"*, *"Due: 2026-09-04"*, the note body, Edit and Delete |
| `.orders-table tr.so-ready` (`css:1320`) | 0.6 | **1.16 : 1** light (`.so-ready-pill`) · 2.32 : 1 dark (`.order-link`) | 4.5 | the whole Factory-Ready row |
| `.er-inactive` (`css:2220`) | 0.6 | **2.21 : 1** light | 4.5 | the *Cancelled* badge, SKU, ship-by weekday |
| `.allocation-status-released` (`css:2063`) | 0.65 | **2.5 : 1** light | 4.5 | the released reservation's quantity, level, source |

A dimmed row is still a row a user reads — Luz reads the released reservation to decide whether to re-allocate,
and reads a completed to-do to confirm it was the right one. The audit's group-05 file recorded these as
*correct* uses of a dim device, which they are in intent; the rendered ratio is the part the code could not show.

**Fix:** drop the `opacity` and de-emphasise with a `--text-muted`-class token that is chosen to meet 4.5 : 1 on
each surface, plus the existing non-colour carriers (strikethrough on a done note, the *Cancelled* / *Released*
badge). One token change covers all four selectors, and it composes with IMP-010 and IMP-043.

---

## Systemic

Twelve root causes account for the large majority of the 400-plus individual findings. [07-systemic-clusters.md](07-systemic-clusters.md) prices four of them against the browser check: how many measured failures each root cause accounts for, and what one token or rule change clears. Fixing these twelve resolves most of the improvements above as a side effect; fixing the improvements without them means fixing the same thing repeatedly.

| # | Root cause | Evidence | Improvements it drives |
|---|---|---|---|
| **SYS-1** | **No shared component or token layer.** Five separate `:root` palettes, twelve semantic roles on one button class, six error-surfacing patterns, four confirmation patterns, twelve table treatments, seven heading sizes for one level, seventeen ad-hoc spacing steps, three nesting indents, four site-nav copies. | `dashboard.css:4-75`, `236-248`; `traceability.html:10-34`; `sankey.html`; `process-flow.html`; `scheduler:8-18` | IMP-010, 012, 013, 037, 039, 049, 057, 058 |
| **SYS-2** | **`innerHTML` replacement is the universal render, refresh, and error strategy.** Every list renderer rewrites its container; every `catch` blanks it first; every background poll tears the view down. | `dashboard.js:700`, `899`, `969`, `1085`, `1180`, `1233`, `1298`, `1671`, `2271`, `2337`, `2547`, `3380`, `3724`, `3858`, `3915` | IMP-007, 008, 036 |
| **SYS-3** | **Interaction is attached to non-controls.** Six element types (`.lot-link` span, `tr.expandable`, `.order-row`, `.search-item` div, `.er-product-option` div, `.product-lot-row`) carry click handlers with no role, no tabindex, no key handler, no focus state, and no button shape. | `dashboard.js:923`, `935`, `1465`, `1530`, `2319`, `3526` | IMP-011, 031, 048 |
| **SYS-4** | **No minimum-hit-size rule exists in the CSS.** Every interactive class was sized for visual density; two selectors in the whole product set 44 px, both on one screen. | `dashboard.css:290`, `316` versus `394`, `236`, `1111`, `1336`, `1344`, `827`, `880` | IMP-011 |
| **SYS-5** | **All type is `px`, with no scale.** 130+ declarations; no root `font-size`; no `rem`; thirteen sizes; twenty-two operational selectors below the 12 px floor. | `dashboard.css` throughout; `mini-calendar.css:63-74` | IMP-009, 037, and the ACCESS-001 half of IMP-062 |
| **SYS-6** | **Colour is carried by hard-coded values that escape the theme system.** Twelve raw values including nine dark-mode-only status colours and one undefined token; two category tokens byte-identical to status tokens. | `dashboard.css:16-19`, `98`, `115`, `1321`, `1341`, `1570-1588`, `1885-1898`; `dashboard.js:1379` | IMP-001, 010, 012, 015, 057 |
| **SYS-7** | **Native browser dialogs are the confirmation and validation mechanism.** Four `confirm()` and three `alert()` calls; the browser makes the destructive choice the Enter default, and their text cannot be copied. | `dashboard.js:1771`, `1813`, `1845`, `2819`, `3096`; `scheduler:1321`, `1565` | IMP-014, 016, 026, 060 |
| **SYS-8** | **No fetch has a timeout.** Every network call in every file is a bare `fetch()` with no `AbortController`. | `dashboard.js:484-491`, `1354`, `1914-1925`, `4114`; `traceability.html:351-361`; `sankey.html`; `process-flow.html` | IMP-005, 044 |
| **SYS-9** | **Lists are truncated server-side and rendered as if complete.** Nine endpoints impose a limit; three tell the user; two derive summary counts from the truncated set. | `dashboard.js:611`, `1177`, `1230`, `2247`, `2286-2288`, `3374`, `3412`, `3908`, `3923`; `mini-calendar.js:108`; `traceability.html:376` | IMP-020, 029 |
| **SYS-10** | **There is no application state in the URL.** No `history`, `pushState`, `replaceState`, or `searchParams` anywhere in `dashboard.js`; nothing is shareable, back-navigable, or refresh-survivable. | `dashboard.js:494-508`, `1344`, `1438`, `2526-2550` | IMP-047, and part of IMP-008 |
| **SYS-11** | **There is no identity.** One shared write-capable API key in client code; no login, no roles, no actor. Drives a public credential, a fabricated audit actor, an unscoped data model, and a required "who are you" field on every supply request. | `dashboard.js:1882`, `2414`, `4024-4035`; `mini-calendar.js:8`; `traceability.html:334` | IMP-028 |
| **SYS-12** | **Nothing is promoted to the entry screen.** Seven attention counts are computed inside their own tab's render function and never surfaced; there is no notification channel to compensate. | `dashboard.js:1716`, `2009-2015`, `2287`, `3411`, `3674`, `3923`, `4163` | IMP-030, 038 |

---

## Settled by the browser check

[06-browser-check.md](06-browser-check.md) rendered all 91 screens in Chromium — 384 captures at 390 px and
1440 px, light and dark, plus 1440 px at 200 % zoom for the four operational tabs — against stubbed API
fixtures. Five clauses that a rendered page can settle without judgement were measured. Each row below is
therefore no longer an open question.

| Rule | Verdict | What was measured | Resolves to |
|---|---|---|---|
| **TOUCH-003** | **FAIL** | Rendered hit boxes on every visible interactive element. **314 of 384 captures** carry at least one target under 44 pt; of the 74 screens that render an interactive control at all, **70 fail at least once** and four never do (S-14, S-15, S-57, S-91). Smallest measured: `button.copyday` **12 × 12**, the four filter checkboxes **13 × 13**, `.order-ready-checkbox` **16 × 16**, `.btn-close` **12.9 × 22**, `.note-checkbox` **18 × 18**, `.lot-link` **43.2 × 14**. The code review's estimates were accurate to a pixel or two throughout. | **IMP-011**, with the scheduler's `button.copyday`, `#leftpanel .ctl > input` and `a.o-action` now named in scope |
| **FEEDBACK-012 / ACCESS-008**, composited fills | **FAIL, both themes** | Every text node's computed colour composited over each translucent fill and inherited `opacity` up to the first opaque surface. **294 of 384 captures** fail — 151 light, 143 dark. The specific fills the audit could not settle: `.so-ready-pill` measures **1.16 : 1** in light (the audit estimated 1.27 : 1 from declared values; the row's `opacity: 0.6` accounts for the rest), `.readiness-chip.severity-*` and `.order-edit-message.*` pass. | **IMP-010** (light), **IMP-043** (the `.so-ready` dim), **IMP-067** (new — the dim device itself) |
| **FEEDBACK-012 / ACCESS-008**, print | **FAIL** | Measured under `@media print`. S-71's own title and the traced lot code render at **1.1 : 1** — light text on white paper. S-91's KPI values at **2.29 : 1**. | **IMP-065** (new) |
| **FEEDBACK-011**, print output | **FAIL** (partly) | The greyscale question is not settled — this run measures colour, not a greyscale conversion. What it does settle is that both print views are broken before greyscale enters into it: see IMP-065. | **IMP-065** (new); the greyscale clause stays open |
| **ACCESS-001**, 200 % zoom | **Mixed** | At a 720 × 450 CSS-px viewport: **no dashboard screen overflows horizontally** — IMP-003's scroll wrappers hold, and so does IMP-002's sticky fix. But in **12 captures across 6 screens** the last actionable element of a scroll region cannot be brought clear of the sticky stack at any scroll position. | Overflow clause → **PASS**; occlusion clause → **IMP-066** (new) |
| **INPUT-022** | **FAIL in dark, PASS in light** | Placeholder colour read from `getComputedStyle(el, '::placeholder')` — the colour the user agent actually chose where no rule exists. **930 measurements, 36 failures, every one in dark theme**, all at **2.69 : 1** (`rgb(117,117,117)` on `--search-bg` `rgb(40,53,72)`) against a 4.5 target. | **IMP-025**, whose scope widens from three modal forms to **nine fields** — the Note, Expected Receipt and Supply Request modals plus the order-ready note and the allocation note |

Two rules outside the original list were measured at the same time, and both are worth recording:

* **LAYOUT-003 / ACCESS-001, horizontal overflow.** 28 of 384 captures overflow, and **not one of them is on
  `index.html`**. Every failure is on a surface that has no compact layout: `sankey.html`, `process-flow.html`
  and `traceability.html` overflow by 81–135 px at 390 px (the culprit is `.header-right` in all three), the
  scheduler's pin modal by 190 px, and the scheduler's print view by 3,339 px. This is measured support for
  **IMP-027** and **IMP-065**, and a clean confirmation of **IMP-003**.
* **LAYOUT-020, layout shift.** Every real background timer in the product — the Recent Entries 60-second poll,
  the allocation-expiry countdown, Process Flow's 60-second auto-refresh — recorded **CLS 0 with nothing moved,
  in all 74 captures where one fired.** On this evidence the product has no background refresh that moves
  content under the user. The whole-dashboard `refreshAll()` is a different matter: it reaches **CLS 0.62** on
  the Activity tab and collapses expanded order rows by 62 px. That path is reached by the Refresh control and
  at app start, which LAYOUT-020 permits — but it is exactly the behaviour that would violate the rule the
  moment **IMP-036** puts these lists on a timer, and it is measured support for **IMP-007**.

### What the check confirmed already fixed

Three shipped improvements were verified in a browser rather than by reading the diff:

| Improvement | Verified |
|---|---|
| **IMP-001** | The lot-disambiguation panel renders correctly in both themes (S-55). The `#fff` fallback is gone. |
| **IMP-002** | The three sticky bars stack correctly at 390 px **and** at 200 % zoom; no bar overlaps another in any of the 384 captures. The PR note that this was "verified by reading the CSS/JS, not in a browser" can be closed. |
| **IMP-003** | No dashboard screen overflows the document horizontally at any width or zoom tested. |

---

## Still unverifiable — what a browser could not settle

Three of the original twenty-one rules are now fully settled — **TOUCH-003**, **FEEDBACK-012 / ACCESS-008**
and **INPUT-022**. Eighteen survive, two of them (**ACCESS-001**, **FEEDBACK-011**) only in part: their zoom
and print-colour clauses are closed above, their largest-text and greyscale clauses are not. Grouped below by
what would actually settle each one — five are within reach of this harness and were simply not attempted;
the rest need a device, a person, a printer, or an operational window a headless run does not have.

**Could be settled by extending this harness** — the measurement exists, this run simply did not make it:

| Rule | Screens | What to add |
|---|---|---|
| **ACTION-002** (disabled clause) | screens using the nine classes | Assert that no capture ever renders those classes with `disabled` — the harness already visits every state. |
| **DATA-006** | S-10…S-13 | Read back the parsed DOM and report what `tr:nth-child(even)` actually shades after the browser recovers the nested `<tbody>`. |
| **DATA-004** | S-63, S-68, S-83, S-87 | Compare `scrollWidth` against `clientWidth` per cell, and diff the visible strings of the real catalogue for collisions. |
| **SEARCH-007** | S-17, S-18 | Drive find-in-page through CDP and count matches with rows hidden versus shown. |
| **ERROR-010** | S-10…S-13, S-25…S-27, S-54…S-56 | Synthesise a mouse-down / move / up over `.lot-link` and assert whether `click` fires. |

**Needs a real device or a real person:**

| Rule | Screens | Why a headless run cannot answer it |
|---|---|---|
| **LAYOUT-014** · **ERROR-002** | S-01…S-05, S-10, S-11, S-13, S-24…S-56 | Rotation and app-switch are OS events. A viewport resize is not the same thing: it does not exercise the state loss an actual backgrounding causes. |
| **INPUT-010** | S-22, S-45, S-53, S-63 | Which soft keyboard iOS Safari and Android Chrome raise is a property of those browsers on those devices. |
| **FEEDBACK-002** · **FEEDBACK-003** | S-24, S-33, S-39, S-64, S-68, S-80 and every network-bound screen | Real elapsed time and the hang-versus-fail distribution against the Railway API. The stub answers instantly by design. |
| **FEEDBACK-011** (greyscale) | S-71, S-91 | Needs a greyscale conversion of the printed output, on paper. IMP-065 settles that both views are broken in colour first. |
| **CHART-005** | S-58, S-68 | What a screen reader announces for an SVG is a property of VoiceOver / NVDA, not of the DOM. |
| **CHART-006** · **ICON-007** | S-58, S-68, S-02, all | Legibility and optical centring at real device pixel ratios are judgements about a rendered glyph, not measurements. |
| **ICON-002 / ICON-005** | every screen using an emoji | Emoji render from the OS font stack. Chromium on this machine is one data point and not the floor phones'. |
| **ACCESS-001** (largest OS text size) | all | The 200 % zoom half is now settled. The OS text-size half needs the OS setting; browser zoom is not equivalent. |
| **DATA-012** | S-25, S-42 | The harness runs in `America/New_York`, so it cannot see the bug. Running it under a second `timezoneId` would settle it — one line, and worth doing. |
| **OTHER-006** | all 91 | Whether these screens were tested on the floor, in floor lighting, with Spanish labels. Not answerable from a repository or a browser. |

---

## What the product already does well

Recorded so that the improvements above do not read as a verdict on the whole product, and so these patterns are protected rather than refactored away.

| Area | What is right | Evidence |
|---|---|---|
| **Floor vocabulary** | Every term on screen is the term the floor uses — Lot, Batch, Cases, Pallets, On Hand, Ship By, BOL, Made, Packed, Bulk, Retail, Pan, Bake, Repack, Crew, Bottleneck. No invented jargon anywhere. OTHER-004 passes outright. | `dashboard.js:174-205`, `411-417`; `index.html:281` |
| **Units on displayed values** | `fmt`, `fmtInt`, `fmtWt`, `fmtLbs`, `fmtQty`, `fmtQtyCases`, `formatInventoryUnits` all emit an explicit unit; the Ingredients header adapts to whether rows share one. | `dashboard.js:66-105`, `393-399`, `1102-1110`, `1394-1398` |
| **Tabular figures** | `--mono` plus `font-variant-numeric: tabular-nums` on ten selectors, so digits align down every numeric column. | `dashboard.css:40`, `704`, `1515`, `2052`, and seven more |
| **Timezone care** | `formatBusinessDate` anchors business dates at `T12:00:00Z` to avoid off-by-one, with a comment explaining why; `formatEnteredAt` refuses to invent an offset for a naive value. | `dashboard.js:511-518`, `520-531` |
| **Error copy** | *"Only 240 lb is coverable. Reduce the request to 240 lb or release a competing reservation."* — states the problem, the number, and two remedies. | `dashboard.js:2977-2983` |
| **Truthful UI** | The readiness card says outright that dispatch readiness *"is not a shipping gate"*; the edit-locked notice explains the rule and the current status in prose rather than greying controls. | `dashboard.js:2896`, `2702-2704` |
| **Explicit incompleteness** | The allocation lot picker warns when the result count equals the limit — the one place in the product that admits truncation. | `dashboard.js:2924`, `2993`, `3014` |
| **Live inline validation** | The scheduler's pin modal recomputes station capacity on every keystroke and reports feasibility **with a remedy**. The reference implementation for INPUT-007. | `scheduler:1457-1462` |
| **State persistence** | The scheduler persists the entire plan to `localStorage` and restores it on load; a refresh loses nothing. | `scheduler:376-378` |
| **Forgiveness** | The scheduler's baseline / delta panel makes every planning change cheap and reversible. | `scheduler:11`, `25`, `948` |
| **Traceability as a whole** | Recents before typing, grouped type-ahead, match highlighting, near-miss correction, URL state, a completeness headline, an adjacent accessible data table, and a text export — the most complete screen in the product and the reference for SEARCH-004, SEARCH-008, CHART-004, and CHART-005. | `traceability.html:285-287`, `404-478`, `938-947`, `1223-1249`, `1254-1297`, `1306-1387` |
| **Accessible naming, done right** | The scheduler names the **record** in every icon-button label: *"Delete SO-1234 Granola 25 LB order line"*. | `scheduler:1343`, `1536-1537` |
| **Keyboard support, done right** | The supplies row implements `role`, `tabindex`, `aria-expanded`, Enter/Space with `preventDefault`, and `:focus-visible` — the model for SYS-3. | `dashboard.js:3849`, `3863-3868`; `dashboard.css:2300-2304` |
| **Selection feedback, done right** | The calendar day card keeps an inset ring, a `z-index` lift so the ring is not clipped, and a synced `aria-expanded` while its detail panel is open. | `dashboard.js:854-859`; `dashboard.css:442-447` |
| **Stall handling, done right** | Process Flow tracks consecutive failures, shows a stale banner naming the last success time, and keeps the last good data rather than blanking. | `process-flow.html` (`fetchData`, `showStale`, `refresh`) |
| **Charts that teach themselves** | Both novel visualizations carry an always-visible key — the Sankey a legend plus column labels, Traceability a legend covering node **and** edge types. CHART-003 passes on both. | `sankey.html:338-349`; `traceability.html:291-300` |
| **Copy to clipboard, done right** | The scheduler uses `navigator.clipboard` with an `execCommand` fallback and reports success or failure in the button label. | `scheduler:1079-1094` |
| **Theme architecture** | A complete paired light/dark token set defined once, persisted, and toggled — the right structure, which the twelve hard-coded values then escape. | `dashboard.css:4-75`; `dashboard.js:44-63` |
| **No sub-Regular weights** | ACCESS-007 passes on all 91 screens; secondary text is de-emphasised with colour and size, exactly as the rule prescribes. | verified by grep; `dashboard.css:1159`, `1658`, `783` |
| **FL-specific concepts get text labels** | Allocate, Release, Trace, Factory Ready, Dispatch Ready, Preview, Pin & replan — no unlabelled glyph was invented for any of them, which is precisely ICON-004's qualification. | throughout |

---

*End of IMPROVEMENTS-MASTER.md*
