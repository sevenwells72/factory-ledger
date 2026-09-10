# Factory Ledger — UX/UI Design Standards (Master)

**Version:** 1.1 · **Date:** 2026-09-10 · *(1.0 consolidated 2026-09-07; 1.1 adds category 17, Status & Data Display)*
**Status:** Standards under construction. No evaluation of the current Factory Ledger has been made.

## How this document was built

Five extraction threads processed Apple HIG pages in parallel and each assigned its own rule IDs. This master merges them into one registry with **permanent IDs**. Where two or more threads extracted the same principle, the rules were consolidated into one stronger rule; where a thread added a qualification to a rule another thread already had, the rule was expanded. The **Crosswalk** at the end maps every source-thread ID to its master ID so nothing is lost.

**Source threads**
- **M** — this thread: *Notifications, Progress indicators, Text fields, Steppers* (M1); *Designing for iOS, Design principles* (M2)
- **A** — *Searching, Drag and drop, Charting data, Entering data*
- **B** — *Segmented controls, Digit entry views, Action sheets, Buttons*
- **C** — *Labels, Collections, Tab views, Lists and tables*
- **D** — *Typography, Layout, Color, Icons*

**Fields per rule:** Rule · Meaning · Factory Ledger application · Platform (Desktop/Web, Mobile, Both) · Importance (Critical, High, Medium, Low/Contextual) · Type (Hard rule, Strong recommendation, Situational idea) · Audit test · Sources.

**Categories / ID prefixes:** NAV, LAYOUT, ACTION, INPUT, TOUCH, FEEDBACK, ERROR, SEARCH, DATA, ACCESS, PERF, NOTIFY, DRAG (direct manipulation), CHART (charts & dashboards), ICON (icons & imagery), STATUS (status & data display), OTHER.

---

## 1. Navigation & Information Architecture

### NAV-001 — Get people directly to the task; the entry screen shows status at a glance and offers one-tap access to the most frequent actions
- **Rule:** The interface stays out of the way. No splash, onboarding wall, or preamble between opening the app and working. Design for sessions as short as one minute: on open, the user sees what needs attention and can start their most common task in one tap. Where the platform offers entry shortcuts (home-screen quick actions, pinned dashboard tiles, bookmarkable deep links), expose the top 2–4 tasks there.
- **Meaning:** People use an operational tool to get something done and leave; every screen between them and the task is friction.
- **Factory Ledger:** Mobile home: today's status strip + big buttons for Receive / Pack / Ship / Log Coconut. Desktop: dashboard opens to Today So Far with direct links to open SOs and expected receipts. Deep links so a WhatsApp message can open a specific SO (see SEARCH-008).
- **Platform:** Both (Mobile primary) · **Importance:** High · **Type:** Strong recommendation
- **Audit:** From a cold open, how many taps to the most frequent task for each role? Is anything shown before the user can act that isn't required?
- **Sources:** M2 NAV-001 · Related: NOTIFY-011

### NAV-002 — Don't lock users into flows or modes; guided flows are escapable, and a genuinely fixed sequence is shown as steps, not tabs
- **Rule:** Let users reach any feature without being forced through a fixed sequence. When a step-by-step flow is genuinely required, provide an obvious exit and a way back to the main experience, and present it with a step indicator rather than a freely tappable tab strip (tabs imply "go anywhere"; steps imply "in this order"). *Qualification:* a transaction that must be atomic (multi-line commit) may prevent partial exit, but the user must be told they are inside such a boundary and how to cancel cleanly.
- **Meaning:** Modal lock-in makes users afraid to start things and traps them when reality doesn't match the script; tabs on a required sequence let required steps be skipped.
- **Factory Ledger:** A guided receiving flow (Scan PO → Confirm supplier lot → Enter qty → Assign LAT code → Confirm) shows steps so lot-code assignment can't be skipped, allows jumping back to any completed step, and can be cancelled with nothing saved. A user who only needs to correct a quantity shouldn't have to re-walk the whole flow.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation (hard rule for an always-available exit)
- **Audit:** For every multi-step flow, can the user leave at any step without side effects, is the exit visible, and does any required sequence use skippable tabs?
- **Sources:** M2 NAV-002 + C NAV-009 · Related: ERROR-001

### NAV-003 — Users always know which record they're on, where they are in it, and what comes next
- **Rule:** Every screen carries a clear heading that names the record and the task (which SO, which lot, which step) and states what the user can do here; the next expected action is obvious. Use a consistent structure so the same kind of screen looks and behaves the same everywhere.
- **Meaning:** Wayfinding failures in an operational tool lead to entries against the wrong record.
- **Factory Ledger:** Header shows "Receiving — PO 4471 · Franklin Baker" before Arturo types anything; "SO-1234 · Restaurant Depot · Packing 3 of 8 lines." Step indicators in multi-step flows. The primary next action sits in the same place on every screen.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** On any screen, can a user say in one sentence which record they're on and what the next step is, without scrolling or backing out?
- **Sources:** M2 NAV-003 + C LAYOUT-002 (view-heading clause)

### NAV-004 — Tabs and segmented controls switch between related views of one subject; primary navigation moves between sections
- **Rule:** Use a tab strip or segmented control only to divide closely related views of the same record, dataset, or task. Move between separate areas of the system with primary navigation (sidebar/main tabs on desktop; bottom tab bar on mobile). On desktop, switch views inside the main content area with a tab view; reserve segmented controls for toolbars, filter bars, and side panels, where they read as a local filter.
- **Meaning:** A tab strip signals enclosure — everything inside is about the same thing. Using it to hop between unrelated areas hides the app's structure and breaks back-navigation expectations; a small segmented control in the content area reads as a filter, not a section change.
- **Factory Ledger:** SO detail: Lines · Allocations · Shipments · History → tabs. "Open | Allocated | Shipped" views of one order list → segmented. Receiving, Packing, Shipping, Trace, Reports → primary navigation, never a tab strip or segmented control. "Day | Week | Month" in the Production Board toolbar → segmented.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Do all tabs/segments in this strip show facets of the same subject? Is any app section reachable only through a tab strip or segmented control?
- **Sources:** B NAV-001 + B NAV-002 + C NAV-001

### NAV-005 — Keep each tab pane self-contained
- **Rule:** Controls inside a tab pane affect only that pane's content. A pane must never silently change, save, or discard data belonging to another pane. If unsaved edits exist in a pane the user is leaving, persist them per-pane or warn explicitly. An action that genuinely spans panes moves out of the tab area to the record header (LAYOUT-021).
- **Meaning:** Panes are mutually exclusive; users assume what they can't see isn't being touched.
- **Factory Ledger:** On an SO detail with Lines and Allocations tabs, "Save" on Allocations must not also commit edited quantities on Lines without saying so. "Mark Factory Ready" (validates all tabs) lives in the record header.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Does any control inside a tab change data shown in another tab, or does switching tabs lose unsaved input?
- **Sources:** C NAV-002

### NAV-006 — Tab and segment labels are short, predictive nouns in Title Case
- **Rule:** Label each tab or segment with a noun or short noun phrase that lets users predict the contents or state before selecting it. Verbs belong on buttons; a verb is acceptable only when the pane *is* an action ("Receive").
- **Meaning:** Nouns say "show me this"; verbs imply something will run. A good label removes trial-and-error clicking.
- **Factory Ledger:** "Lots | SKUs," not "Show Lots | Show SKUs." "Day | Week," not "View Day." "Allocations," "Trace" — never "More," "Other," "Info."
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Can a first-time user say what's behind each tab/segment from its label alone, and are all labels nouns?
- **Sources:** C NAV-003 + B ACTION-017

### NAV-007 — Use a visible one-tap switcher for small sets of views or filters; a dropdown only when there are too many to show
- **Rule:** For a small set (≤ ~6) of closely related, mutually exclusive choices that change a view, state, or object — especially when the current choice should be visible at a glance and the group must stay together at any width — use a segmented control or tab strip, not a dropdown. Reserve dropdowns/menus for long sets. Multi-select segmented controls are acceptable on desktop for combinable attributes.
- **Meaning:** A visible switcher costs one tap and shows all options and the current selection; a dropdown costs two taps and hides them.
- **Factory Ledger:** Lot status filter (Active · On Hold · Consumed · Void) and dashboard range (Today · Week · Month) are segmented controls. Customer and product selection use searchable dropdowns (INPUT-014).
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Is any small set of view/filter options hidden behind a dropdown that costs an extra tap to see?
- **Sources:** C NAV-004 + B ACTION-014

### NAV-008 — Limit tabs and segments: at most 5 on phones; about 6 tabs / 7 segments on wide screens
- **Rule:** Beyond these limits, restructure: regroup, promote to primary navigation, or use a dropdown, menu, or filter panel. Never let tabs wrap, truncate, or scroll horizontally.
- **Meaning:** Too many options become unreadable, truncate labels, break layout, and slow selection.
- **Factory Ledger:** If a record detail grows past six facets, consolidate ("Notes" + "Attachments" → "Details") or move rarely used facets under an overflow. Eight product families → dropdown on mobile.
- **Platform:** Both (stricter on Mobile) · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does any tab strip or segmented control exceed the limit for its platform, or wrap/truncate?
- **Sources:** C NAV-005 + B ACTION-015

### NAV-009 — Express hierarchy with drillable lists; use list + detail side by side on wide screens
- **Rule:** Present hierarchical data as lists the user drills into. On wide screens show the list and the selected item's detail together (split view) so the user keeps their place; on narrow screens navigate list → detail with a clear back path.
- **Meaning:** Lists are the natural way to show "what's here and what's under it." Split views save a round-trip per record on desktop.
- **Factory Ledger:** Desktop: SO list on the left, selected SO detail on the right. Mobile: Lots → lot detail → trace events.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** On desktop, can the user move between records without losing their place in the list? On mobile, is the drill path and back path obvious?
- **Sources:** C NAV-006

### NAV-010 — Use one affordance for "go deeper" and a different one for "show details"
- **Rule:** A chevron/disclosure means navigate into a child level; an info/expand affordance means reveal more about the current item; disclosure triangles mean expand/collapse in place. Never use the same icon for two of these.
- **Meaning:** Users learn what a chevron does once; mixing meanings forces guessing.
- **Factory Ledger:** Chevron on an SO row opens the SO. An "i"/expand on a lot row shows lot metadata inline. A trace tree uses disclosure triangles.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does the same icon ever mean "navigate" in one place and "show details" in another?
- **Sources:** C NAV-007

### NAV-011 — Provide next/previous between sibling records inside a detail view for sequential review
- **Rule:** When users process records in sequence, offer next/previous from the detail view so they don't return to the list each time. Applies only when detail views are short enough not to scroll.
- **Meaning:** Removes one round-trip per record in batch-review workflows.
- **Factory Ledger:** Reviewing receipt lines, walking through today's pack records, verifying lots one by one — swipe or arrow to the next.
- **Platform:** Both · **Importance:** Medium · **Type:** Situational idea
- **Audit:** For sequential review tasks, must the user return to the list between every record?
- **Sources:** C NAV-008

### NAV-012 — Signal hidden content: collapsed, paged, scrolled-off, or filtered content is always evident, with a count and a path to reveal it; a limited list never looks complete
- **Rule:** Keep lists short by default by showing the most relevant subset, but make it unmistakable that more exists — "Show 12 more," a remaining count on "Load more," a count badge on collapsed sections, partially visible items at scroll edges — and provide an obvious path to the full set.
- **Meaning:** Hidden content with no cue is effectively invisible; a hidden item is a "missing" item to the user.
- **Factory Ledger:** Mobile Lots shows active lots with "Show consumed/void (142)." A lot detail showing five trace events says "12 more." Collapsed SO sections (allocations, flags, notes) show counts. Recent Entries shows the last 20 with "View all." The count stops Arturo concluding a lot doesn't exist because it's filtered out.
- **Platform:** Both (especially Mobile) · **Importance:** High · **Type:** Strong recommendation
- **Audit:** If this screen hides content, is there a visible cue that more exists, how much, and how to reach it?
- **Sources:** D NAV-001 + C DATA-011 · Related: LAYOUT-001

---

## 2. Screen Layout & Visual Hierarchy

### LAYOUT-001 — Show only what the primary task needs; keep secondary details and actions one interaction away
- **Rule:** Limit onscreen controls to those needed for the current task. Everything else is discoverable with minimal interaction (expand, long-press, overflow menu, secondary tab) rather than removed or cluttering the main view. Simplicity is not minimalism: the important things stay close; the rest falls away.
- **Meaning:** Every extra control dilutes attention and adds a chance of the wrong tap, especially on a small screen.
- **Factory Ledger:** Pack screen shows product, lot, qty, Submit. Void, reassign, notes, history live behind "···" or an expandable section. Dashboard tiles show the number; the breakdown appears on tap.
- **Platform:** Both (Critical-leaning on Mobile) · **Importance:** High · **Type:** Strong recommendation
- **Audit:** For each screen, can every visible control be justified by the primary task? Are secondary actions reachable in ≤1 extra interaction?
- **Sources:** M2 LAYOUT-001

### LAYOUT-002 — Once an element's appearance or behavior is established, apply it identically everywhere
- **Rule:** Same control, same look, same behavior, same position across all screens and both platforms. Don't invent a second way to do something already done elsewhere.
- **Meaning:** Consistency lets users learn once and trust that new screens work as expected.
- **Factory Ledger:** One style for primary buttons, one for destructive, one for status chips. A lot-code field looks and validates the same on Receiving, Packing, and Trace. Submit is always bottom-right (desktop) / bottom full-width (mobile).
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Pick any control type — does it look and behave the same on every screen? Are there two different patterns for the same job?
- **Sources:** M2 LAYOUT-002 · Instances: FEEDBACK-005, INPUT-013, ACTION-006, DATA-007

### LAYOUT-003 — The layout adapts to size, orientation, text size, and label length while staying recognizable; destinations and positions don't change across platforms
- **Rule:** Define behavior for compact (phone-width) and regular (tablet/desktop) sizes, both orientations, large text, and longer localized labels (Spanish runs ~20–30% longer). Across all of these, the same elements stay in recognizably the same places. Top-level destinations are identical in name and order on the desktop sidebar and the mobile tab bar (the tab bar carries the 4–5 most-used floor destinations plus "More"). Transitions use natural motion so users can follow what moved, appeared, or closed.
- **Meaning:** Someone who learned the desktop should not be lost on the phone; rotating a device or narrowing a window shouldn't scramble the screen.
- **Factory Ledger:** Lot detail: phone = single column; tablet landscape = two columns; desktop = table + side panel. Nav order Orders · Receiving · Production · Shipping · Trace everywhere. A record's status chip in the same relative spot on both platforms. Buttons and headers accommodate Spanish labels.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Does this screen render correctly and recognizably at phone, tablet, and desktop widths, in both orientations, with large text and Spanish labels? Are top-level destinations identical in name and order on desktop and mobile?
- **Sources:** M2 LAYOUT-003 + D LAYOUT-008 + D NAV-002 · Related: ACCESS-001, LAYOUT-013, LAYOUT-014

### LAYOUT-004 — Establish a clear visual hierarchy: the most important information gets room and is apparent at a glance; nonessential detail goes elsewhere
- **Rule:** Form and function should be readily apparent. Use size, position, weight, and standard control shapes so the user immediately identifies the key data. The most important information gets enough space to be read immediately; metadata and secondary detail go to secondary areas, expandable sections, or a detail view rather than crowding it. (The single primary action's prominence is governed by ACTION-003.)
- **Meaning:** If everything looks equally important, nothing is; crowding the key number with metadata makes it slower to find and easier to misread.
- **Factory Ledger:** Pack screen: SO number, product, target quantity, remaining quantity dominate; supplier lot numbers, notes, audit metadata sit below or behind a disclosure. SO detail: status and ship date are the largest text. Tables: identifying columns first; metadata columns collapsible.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Which single piece of data did the designer intend the user to read first, is it the easiest thing to see, and is anything nonessential competing with it?
- **Sources:** M2 LAYOUT-004 + D LAYOUT-002 · Related: ACTION-003

### LAYOUT-005 — Use a consistent text-emphasis hierarchy (primary / secondary / tertiary / faint) via size, weight, and color, preserved at every text size
- **Rule:** Define four emphasis levels — primary (key information), secondary (supporting), tertiary (unavailable/disabled or audit metadata), faint (placeholder/watermark) — and apply them the same way everywhere. Decision-critical data is never rendered at tertiary or faint emphasis. The relative hierarchy must survive user text-size changes, and primary elements stay toward the top even when text is very large.
- **Meaning:** Weight and color tell the eye what matters before it reads; a hierarchy that flips at large sizes is a broken one.
- **Factory Ledger:** Lot code and quantity on a row = primary; supplier and received date = secondary; consumed/void lot = tertiary; audit timestamps = tertiary. Quantities that matter now (remaining to pack) are larger/bolder than reference quantities (original ordered). A Factory Ready flag or a shortfall is never low-contrast gray.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Can a user identify primary, secondary, and tertiary text by size/weight/color alone, is the decision-relevant value at the strongest emphasis, and does that ordering hold at the largest text size?
- **Sources:** C LAYOUT-001 + D LAYOUT-014 · Related: ACCESS-005

### LAYOUT-006 — Position by importance in reading order
- **Rule:** Place the most important items near the top and leading (left) side, where scanning starts; order the remainder by decreasing importance. Blocking statuses appear before the content they block.
- **Meaning:** What's first is what's noticed.
- **Factory Ledger:** Screen title + primary identifier (SO#, lot code) top-left; the identifying column first in tables. A "Hold" or "Not Factory Ready" banner sits above the record body, not below it.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Reading top-to-bottom, left-to-right, does the user encounter the most important element first?
- **Sources:** D LAYOUT-003

### LAYOUT-007 — Group related items visually; controls are always distinguishable from content
- **Rule:** Use spacing, cards/background shapes, separators, and color to show which elements belong together and to divide the screen into distinct areas. Grouping must never blur the line between content (data) and controls (actions): a status chip must not look like a button, and a button must not look like a data chip.
- **Meaning:** Visual grouping is how users find what they're looking for without reading everything; control/content confusion causes wrong taps.
- **Factory Ledger:** Receiving form: supplier/PO/date in one group; lot/quantity/unit in another; actions in a footer group. Lot detail: Identity / Quantities / Trace / Actions as separate groups. Status badges are never styled like buttons.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Can a user tell at a glance which items belong together, and distinguish every control from every piece of data?
- **Sources:** D LAYOUT-001 · Related: ACTION-010

### LAYOUT-008 — Persistent control areas are a visibly separate layer from scrolling content
- **Rule:** Toolbars, tab bars, sticky action bars, and sidebars have a visual treatment distinct from scrolling content, with a clear transition at the edge (subtle shadow, fade, or blur) rather than a heavy background or no separation. Content under a translucent bar must stay legible at rest.
- **Meaning:** Users need to know what stays put and what scrolls.
- **Factory Ledger:** A sticky "Confirm Pack" bar on mobile is visibly separated from the list scrolling beneath it; the desktop filter/action toolbar is distinct from the table.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Is every persistent control area visually distinct from scrolling content, with a clear edge?
- **Sources:** D LAYOUT-005 · Related: ACCESS-008

### LAYOUT-009 — Caption a control only when it doesn't explain itself
- **Rule:** Don't add introductory text to a control whose purpose is clear from its content, position, or standard appearance (text-labeled segmented controls, add/remove buttons beside a table, a standard "?"). Do add a label or tooltip when a control is icon-only or its meaning depends on context. (Input fields always get a label — INPUT-002.)
- **Meaning:** Redundant captions add clutter and reading time; missing labels on ambiguous controls cause errors.
- **Factory Ledger:** "Open | Packed | Shipped" needs no "Filter:" caption. An icon-only "+" under the Lines table needs no caption but gets a desktop tooltip.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Is every caption on this screen doing work the control can't do on its own?
- **Sources:** B LAYOUT-001 · Related: ACCESS-010

### LAYOUT-010 — Align to a shared grid with a fixed spacing scale; use indentation to show nesting
- **Rule:** Align elements to a common grid and use a fixed spacing scale. Section titles get clear space above them and clear separation from their rows. Nested items indent one consistent level. Numeric columns align to a shared edge.
- **Meaning:** Alignment lets the eye track rows and columns; inconsistent spacing breaks the grid and makes scanning harder.
- **Factory Ledger:** Quantities right-aligned so they compare at a glance; form labels/values on a consistent baseline; lots under a product, components under a batch indented one level; scheduler rows on a uniform grid.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Do all elements sit on a consistent alignment grid with uniform spacing, and does indentation reliably indicate nesting?
- **Sources:** D LAYOUT-004

### LAYOUT-011 — Content fills the viewport, and nothing actionable is hidden behind fixed bars
- **Rule:** Backgrounds and scrollable layouts extend to the edges (no floating box with dead margins). Because navigation and action bars float above content, scrollable regions include end insets so the last items scroll fully into view above those bars.
- **Meaning:** Wasted space shrinks usable area; a fixed bar that covers the last row hides an actionable item.
- **Factory Ledger:** On a phone with a sticky bottom action bar, the last row of a pick list scrolls clear of the bar. Desktop tables use the full available width.
- **Platform:** Both · **Importance:** High (Critical wherever a fixed bar can hide an actionable row) · **Type:** Strong recommendation
- **Audit:** Can every item in a scrollable region be scrolled fully into view, unobstructed by fixed bars, and does the layout use the available space?
- **Sources:** D LAYOUT-006 · Related: LAYOUT-019

### LAYOUT-012 — Space and group controls so a consequential control can't be mis-hit for a routine one
- **Rule:** Provide enough space around interactive elements that adjacent controls are visibly distinct; group related controls and separate unrelated ones. Never place two different-purpose targets in the same edge zone of a row where using one triggers the other. Hover/focus states that enlarge a control must not overlap neighbors.
- **Meaning:** Unrelated controls placed close together are the classic cause of the wrong action being taken.
- **Factory Ledger:** "Void" and "Confirm" never sit adjacent with minimal spacing; +/- steppers are spaced away from "Submit." A lot row with a trailing chevron doesn't also carry a trailing "Consume" button on mobile — secondary actions go behind long-press or into the detail. Desktop row-action icons spaced so "edit" isn't clicked for "void."
- **Platform:** Both (Mobile primary) · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Are adjacent controls spaced and grouped so a user can't easily mis-hit one for another, especially where one is consequential? Are there rows with two different actions within thumb-width at the same edge?
- **Sources:** D LAYOUT-007 + C TOUCH-003 · Related: TOUCH-003

### LAYOUT-013 — Collapse to compact layouts as late as possible; hide tertiary panels first
- **Rule:** Design the full layout first and switch to a compact arrangement only when the full layout no longer fits. As width shrinks, hide tertiary panels (inspectors, side details) before secondary columns, and secondary columns before main content. Breakpoint changes never move navigation or the primary action. Test at half-screen, third-screen, and side-by-side sizes.
- **Meaning:** Stability across sizes keeps the UI familiar; premature collapse wastes space on large screens.
- **Factory Ledger:** Office users put the dashboard beside another window at half width — the table drops its detail side panel first, then non-key columns, before switching to cards.
- **Platform:** Both (desktop-heavy) · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** As the window narrows, does the layout drop tertiary elements first and remain stable, without surprising rearrangement at each breakpoint?
- **Sources:** D LAYOUT-009

### LAYOUT-014 — Support both orientations, or fail gracefully in one; rotation never loses state
- **Rule:** Aim to support portrait and landscape. If a screen only works in one, it works whichever way the device is rotated within that orientation and never asks the user to rotate. Rotation never loses entered data.
- **Meaning:** Floor devices get held and mounted in different ways.
- **Factory Ledger:** Phone pack/receive screens are portrait-first; a scheduler board on a tablet may be landscape-first; a mounted tablet may be fixed landscape. All survive rotation mid-entry.
- **Platform:** Mobile · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does this screen work in both orientations (or both rotations of its supported orientation) without losing state?
- **Sources:** D LAYOUT-011 · Related: ERROR-002

### LAYOUT-015 — Respect safe areas and system UI; keep the OS status bar visible
- **Rule:** Keep content and controls out of areas covered by device features (notch, camera housing, home indicator, rounded corners) and by system/browser chrome (status bar, browser toolbars, on-screen keyboard). Keep the status bar visible — it carries time, battery, and connectivity at little cost; hide it only for immersive experiences (none in FL).
- **Meaning:** Controls under the home indicator or keyboard are unreachable; hiding connectivity status on a floor device removes a diagnostic.
- **Factory Ledger:** Mobile web: bottom action bar padded above the home indicator; inputs scroll into view above the keyboard; nothing rendered under the status bar.
- **Platform:** Mobile (some Desktop/Web relevance for browser chrome) · **Importance:** High · **Type:** Hard rule
- **Audit:** Is any control or critical content obscured by device features, the status bar, browser chrome, or the keyboard?
- **Sources:** D LAYOUT-010

### LAYOUT-016 — Don't rely on the bottom edge of a desktop window for critical information or the only copy of an action
- **Rule:** People drag windows so the bottom edge is offscreen. Footer actions are sticky within the viewport or duplicated higher up; totals rows are visible without scrolling the window itself.
- **Meaning:** A "Save" below the screen edge is a "Save" the user can't find.
- **Factory Ledger:** Desktop forms keep Save/Submit sticky-within-viewport or in the header.
- **Platform:** Desktop/Web · **Importance:** Low/Contextual · **Type:** Situational idea
- **Audit:** If the bottom of the window is offscreen, is anything critical lost?
- **Sources:** D LAYOUT-012

### LAYOUT-017 — Use at most three background (surface) levels to express containment
- **Rule:** Define primary (page), secondary (grouping/cards), and tertiary (groups within groups) surfaces and use them consistently. Use an inset-card variant for grouped mobile lists and a plain full-bleed variant for tables. If content needs more than three levels, split the screen.
- **Meaning:** Nesting depth is read from background levels; more than three becomes unreadable.
- **Factory Ledger:** Page → SO card → nested allocation group. Inset cards for settings-style screens; plain surfaces for dense tables.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does this screen use at most three consistent background levels, each meaning one level of containment?
- **Sources:** D LAYOUT-013 · Related: OTHER-010

### LAYOUT-018 — Use color sparingly and for meaning; keep chrome neutral
- **Rule:** Reserve color for status indicators, the primary action, and selection. Keep toolbars, tab bars, and other chrome mostly monochromatic — especially when content itself carries color (status-tinted rows, charts). A brand accent works when content is otherwise neutral.
- **Meaning:** Color is a scarce attention resource; too much reduces the signal of the color that matters.
- **Factory Ledger:** Rows carrying status colors sit under neutral navigation and filter bars so status pops. Brand accent for primary buttons and selected tabs only.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Is every use of color on this screen doing a job (status, primary action, selection), and is the chrome neutral enough that status colors stand out?
- **Sources:** D LAYOUT-015 · Related: FEEDBACK-011..014

### LAYOUT-019 — Constrain reading width on wide screens; tables and boards may use full width
- **Rule:** Limit the width of prose, forms, and single-record detail views on very wide windows so lines stay readable and content stays near where the eye rests. Dense tables and boards use full width (complements LAYOUT-011).
- **Meaning:** Very long lines are hard to read; content pinned to the far left of a huge window is easy to lose.
- **Factory Ledger:** Lot notes, batch instructions, and single-record forms capped at a comfortable width; the production board and lot tables use the full window.
- **Platform:** Desktop/Web · **Importance:** Low/Contextual · **Type:** Situational idea
- **Audit:** On a wide monitor, is prose or form content constrained to a readable width while tables use the full width?
- **Sources:** D LAYOUT-016

### LAYOUT-020 — Never move content under the user
- **Rule:** Don't change layout, reorder items, or resize elements while the user is viewing or interacting, unless it is a direct response to their explicit action. Background refreshes must not shift items the user may be about to tap: buffer live updates ("3 new entries — tap to refresh") or append without shifting visible rows.
- **Meaning:** Layout shifts cause mis-taps and lost place; in a ledger a mis-tap is a wrong transaction.
- **Factory Ledger:** If the lot list auto-refreshes while Arturo is choosing a lot to consume, a new row inserting at the top moves his target and he consumes the wrong lot. Same for the Recent Entries feed and Today So Far tile.
- **Platform:** Both (Critical on Mobile) · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Can any background refresh or async load move, reorder, or resize an item the user is about to tap?
- **Sources:** C LAYOUT-003 · Constrains: FEEDBACK-007, FEEDBACK-010

### LAYOUT-021 — Placement communicates scope: record-level controls in the header, list-level controls adjacent to the list, tab-level controls inside the tab
- **Rule:** Buttons that act on a specific table, list, or panel (add row, remove row, reorder) are icon-only and sit within or directly beneath that view, not in the global header. Controls that apply to the whole record sit in the record header, visually outside (and inset from) the tab area. Global toolbars hold navigation and screen-level actions only.
- **Meaning:** Proximity tells the user what a control operates on; a "+" in a global header is ambiguous.
- **Factory Ledger:** "+ Line" / "− Line" directly under the order-lines table. "Void SO" and "Mark Factory Ready" in the record header above the Lines/Allocations tabs. Screen-level "New Order" in the header.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Is every add/remove/reorder control adjacent to the exact list it modifies, and is it visually clear which controls apply to the whole record versus the current tab?
- **Sources:** B ACTION-011 + C LAYOUT-004 · Related: NAV-005

---

## 3. Actions & Buttons

### ACTION-001 — Match the control type to the interaction: button = immediate action, toggle = on/off state, segmented = choose among a few options, menu = longer list
- **Rule:** Never mix segments that select a state with segments that fire an action within one segmented control, and never show selection state on an action segment.
- **Meaning:** Each control type carries an expectation about what happens on tap; a segment that "does" something next to siblings that "select" something makes the interface unpredictable.
- **Factory Ledger:** "Factory Ready" flag → toggle. Pack format choice → segmented. "Post Production" → button. Don't add "Refresh" as a fourth segment to a status filter.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Does each control's type match its behavior (action vs. state vs. selection), with no segmented control mixing the two?
- **Sources:** B ACTION-001

### ACTION-002 — Every button visibly shows pressed, hover (desktop), disabled, and selected/toggled states
- **Rule:** Never ship a custom button without a press state. Disabled must look disabled (and, where useful, say why). Reserve the selected/toggled look for controls that are actually toggled; don't reuse it as a plain button style.
- **Meaning:** Without a press state people can't tell whether the tap registered, so they tap again — the classic path to duplicate submissions. A disabled state that looks enabled invites dead taps.
- **Factory Ledger:** "Post Receipt" visibly depresses on tap and greys out while submitting. A toggled "Factory Ready" chip looks different from both an untoggled chip and a plain button.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Does every button visibly react on press, look clearly different when disabled, and show selection when applicable?
- **Sources:** B ACTION-003 · Related: FEEDBACK-001, FEEDBACK-008, INPUT-022

### ACTION-003 — One prominent (filled) primary action per view; at most two; emphasis comes from fill, not colored text
- **Rule:** Give the single most likely action the filled/accent style. Everything else uses quieter (plain/outlined) styles. If several controls in one area are filled, none is primary.
- **Meaning:** Fill and color draw the eye; more than two prominent buttons force the user to compare instead of act.
- **Factory Ledger:** Receiving form → "Post Receipt" is the only filled button; "Add Line," "Save Draft," "Cancel" are secondary. Order list → "New Order" prominent; filters/export quiet.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Can the user immediately tell which action is primary on this screen, is it the only filled control in its area, and are there ≤2 prominent buttons?
- **Sources:** B ACTION-004 + D ACTION-001 · Related: LAYOUT-004, ACTION-008

### ACTION-004 — Within a set of choices, style (not size) distinguishes the preferred option
- **Rule:** Buttons forming a set are the same size and height; the preferred one uses a more prominent style. Keep heights standard; a taller button only when content genuinely needs two lines or a tall icon.
- **Meaning:** Equal size signals "these belong together"; a size mismatch reads as an error or a different kind of control.
- **Factory Ledger:** "Ship Partial" / "Ship All" side by side: same size, "Ship All" filled.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Are buttons in the same choice set the same size, distinguished by style only?
- **Sources:** B ACTION-005

### ACTION-005 — Every button states its outcome: a verb-first label, a conventional icon, or both; icon-only controls carry a tooltip and accessible name
- **Rule:** Prefer a short verb-first text label in title case naming the specific action (Post Receipt, Void Lot, Add Line). Avoid vague labels (OK, Submit, Go, Done) when a specific verb is possible. In a confirmation dialog the confirming button repeats the action ("Void"). When using an icon, use the standard glyph (ICON-004), not a novel one. On desktop, icon-only buttons and segments must have tooltips; text-labeled buttons don't need them.
- **Meaning:** The label is the user's only preview of the consequence; floor users don't experiment — an unclear button is ignored or misused.
- **Factory Ledger:** "Post Receipt" not "Submit"; "Consume Lot" not "OK"; "Print Labels" rather than a printer glyph alone on mobile. Every icon-only desktop row action gets a tooltip.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Can a first-day user say exactly what will happen from each button's label or icon alone?
- **Sources:** B ACTION-006 + C ACTION-001 · Related: ACCESS-004, ACCESS-010

### ACTION-006 — Fixed semantic button roles applied everywhere: Primary (accent fill), Secondary (neutral), Cancel (neutral, no change), Destructive (red)
- **Rule:** Never restyle a role locally, never use the destructive color for non-destructive actions, and never present a destructive action in a non-destructive style.
- **Meaning:** Role styling is a signal people learn once and rely on. If red sometimes means "delete" and sometimes "rush," the signal is worthless.
- **Factory Ledger:** "Void Shipment," "Delete Line," "Unallocate" → red everywhere. "Post," "Save," "Ship" → accent. Red is never used for "Rush" or "Late" buttons.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Do buttons of the same role look identical across screens, and does red appear only on data-destroying actions?
- **Sources:** B ACTION-007 · Related: FEEDBACK-012, OTHER-010

### ACTION-007 — On desktop, the primary non-destructive button is the default: Enter triggers it and closes temporary views
- **Rule:** In every desktop form, sheet, or dialog, Enter/Return triggers the primary action and, for temporary views, dismisses the view. Never bind Enter to a destructive action (ACTION-008).
- **Meaning:** Keyboard-driven confirmation is what makes repetitive entry fast.
- **Factory Ledger:** Receiving line entry: type qty → Enter posts the line and advances. "Confirm Allocation" dialog: Enter confirms.
- **Platform:** Desktop/Web · **Importance:** High · **Type:** Strong recommendation
- **Audit:** In each desktop form/dialog, does Enter trigger the primary non-destructive action?
- **Sources:** B ACTION-008 · Related: INPUT-006

### ACTION-008 — Destructive actions are unmistakable, placed to be noticed, and never the default
- **Rule:** Style destructive options in the destructive (red) style and place them where they'll be seen (top of a mobile action sheet) so no one picks them by accident — while never making them the primary/default (Enter-key, visually dominant) button, even when they're the most likely choice. Give the primary role to a non-destructive option.
- **Meaning:** Visibility prevents accidental selection; non-default prevents reflexive selection. People pick the prominent button without fully reading it.
- **Factory Ledger:** "Void this shipment?" → "Keep Shipment" is primary/Enter; "Void" is red and secondary. "Discard unsaved receipt?" → "Discard Receipt" red at top of the sheet; "Save Draft" normal and default; "Cancel" at bottom.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Is there any dialog or screen where Enter or the most prominent button destroys or reverses data? Are destructive options red and visually distinct?
- **Sources:** B ACTION-009 + B ERROR-005 · Related: NOTIFY-009, DRAG-003

### ACTION-009 — Signal when a button opens further input rather than acting now
- **Rule:** Use a trailing ellipsis ("Adjust Quantity…") or an equivalent convention for buttons that open another view requiring more input; a plain label means "this happens now."
- **Meaning:** People hesitate over buttons if they're unsure whether tapping commits something.
- **Factory Ledger:** "Void…" opens a reason/confirm sheet; "Print…" opens a label-count dialog; "Post" acts immediately.
- **Platform:** Both (convention strongest on Desktop) · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Can the user tell before tapping whether a button acts now or opens another step?
- **Sources:** B ACTION-010

### ACTION-010 — Buttons look like buttons
- **Rule:** Give buttons a discernible shape with a contrasting fill so they read as tappable. Exception: buttons inside a toolbar, menu, or dialog whose own shape already sets them apart.
- **Meaning:** Text-only "buttons" get missed and are hard to hit; enclosed shapes are found and tapped faster.
- **Factory Ledger:** Floor screens: "Post," "Next," "Confirm" as filled or outlined blocks, not blue text links. Toolbar icons can stay borderless.
- **Platform:** Both · **Importance:** High (Critical on Mobile) · **Type:** Strong recommendation
- **Audit:** Does every action outside a toolbar/menu/dialog have a visible button shape and fill?
- **Sources:** B ACTION-012 · Related: LAYOUT-007

### ACTION-011 — Prefer standard/native controls; a custom control must be justified and must replicate every standard state
- **Rule:** Use native or shared design-system controls wherever they fit — they carry interaction states, accessibility, feedback, and size adaptation. For standard value types (date, time, item from a list) prefer the platform's built-in picker. A custom control is justified only when the standard one can't meet the need or a purpose-built control is measurably faster for a repetitive floor task, and it must replicate press/hover/focus/disabled states, keyboard support, and screen-reader labels.
- **Meaning:** Custom controls usually miss something; familiar controls need no learning.
- **Factory Ledger:** Native select/segmented/toggle/date inputs on mobile; one shared button component in the dashboard rather than per-page CSS. A large tappable lot-picker may justifiably replace a native select if it saves taps.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Is each custom control justified (documented speed reason), and does it replicate every standard state?
- **Sources:** B ACTION-013 + D INPUT-002 · Related: OTHER-004

### ACTION-012 — Segments within one segmented control are uniform: equal widths, similar content length, all text or all icons
- **Rule:** Never mix text and icon segments; icon-only segments get desktop tooltips or short labels beneath.
- **Meaning:** Uneven segments look broken and make the selected one harder to spot; mixed segments read as unrelated controls.
- **Factory Ledger:** "Open | Packed | Shipped," not "Open | Packed and Awaiting Pickup | Shipped." No truck icon for Shipped beside text labels.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Are segments uniform in width and content type, with icon-only segments labeled or tooltipped?
- **Sources:** B ACTION-016

---

## 4. Forms & Data Entry

### INPUT-001 — Single-line fields for short values; multi-line areas for notes
- **Rule:** Use a text field only for a small, specific value. Use a text area for free-form or longer text.
- **Meaning:** Field type signals expected length and behavior (Enter submits vs. adds a line).
- **Factory Ledger:** Lot code, PO number, quantity, case count → single-line. Receiving notes, QC observations, no-show reason → text area.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Is any free-form note squeezed into a single-line field, or any short code given a large text area?
- **Sources:** M1 INPUT-001

### INPUT-002 — Every field has a persistent label naming the value and its unit, with a format example as hint; a dedicated entry screen also names the record
- **Rule:** Show a visible label that remains while the user types and includes the unit ("Qty (cases)") and, for dates, which date ("Received on"). Placeholder text may add an example or format hint ("e.g. 26-0907-C") but disappears on input and can never be the sole identifier. Any dedicated entry screen or prompt carries a title naming what to enter, in what unit, and for which record.
- **Meaning:** Once typing starts, a placeholder-only field loses its identity; an entry box without context invites the right value in the wrong field or the wrong unit.
- **Factory Ledger:** On mobile forms with several quantity fields (cases, units, weight), persistent unit labels prevent entering weight in the case field. Focused keypad screen: "Cases Packed — Lot 26-0907-COC, Sweetened Coconut 5 lb."
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** After every field is filled, can you still identify each field and its unit without clearing it? Does every entry prompt name the field, unit, and record?
- **Sources:** M1 INPUT-002 + A INPUT-002 (label clause) + B INPUT-003 + C LAYOUT-002 (label clause) · Related: INPUT-022

### INPUT-003 — Mask secrets only; never mask operational values; never prefill a credential
- **Rule:** Any field holding a password, API key, token, or PIN uses obscured input and always requires fresh entry or biometrics. Quantities, lot codes, weights, and other operational data are never masked — they must stay visible for verification before commit.
- **Meaning:** Masking protects secrets; on operational data it removes the user's only chance to catch a typo.
- **Factory Ledger:** Login, API-key settings, any floor PIN sign-in → masked. Every quantity remains visible and checkable before "Post."
- **Platform:** Both · **Importance:** Low/Contextual (Critical where applicable) · **Type:** Hard rule
- **Audit:** Is any credential shown in clear or prefilled? Is any operational value masked?
- **Sources:** M1 INPUT-003 + A INPUT-009 + B INPUT-002

### INPUT-004 — Field width matches the expected length of the value
- **Rule:** Size each field to the quantity of text anticipated.
- **Meaning:** Width is an implicit hint about what to enter; a wide field for a 3-digit quantity invites over-typing or looks unfinished.
- **Factory Ledger:** Quantity: narrow. Lot code: fixed width for the LAT format. Customer name: medium. Notes: full width.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does the visual size of each field roughly match its typical content?
- **Sources:** M1 INPUT-004

### INPUT-005 — Space fields evenly, stack vertically, use consistent widths, and make label–field pairing unambiguous
- **Rule:** Lay out forms so each label clearly belongs to one field. Prefer vertical stacking; group related fields into consistent width classes. Mobile forms are single-column.
- **Meaning:** Tight or irregular layouts cause users to attach the wrong label to a field.
- **Factory Ledger:** Receiving form: Supplier / PO / Lot / Qty / Unit stacked; quantities share one width, identifiers another.
- **Platform:** Both (single-column mandatory on Mobile) · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Can a first-time user tell which label belongs to which field without hesitation? Are same-kind fields the same width?
- **Sources:** M1 INPUT-005 · Related: LAYOUT-010

### INPUT-006 — Focus/tab order follows the logical entry sequence and ends on the primary action
- **Rule:** Keyboard tabbing and mobile "Next" move through fields in the order a user naturally fills them — matching the physical process and the paper form — and land on Submit.
- **Meaning:** Fast repetitive entry depends on never touching the mouse; broken order costs time and lands data in the wrong field.
- **Factory Ledger:** Coconut form / pack form order matches the paper form Arturo reads from. Tab from the last field lands on the primary button.
- **Platform:** Both (keyboard on Desktop/Web; Next/Done on Mobile) · **Importance:** High · **Type:** Hard rule
- **Audit:** Can the whole form be completed with keyboard/Next only, in the order the work happens, ending on Submit?
- **Sources:** M1 INPUT-006 · Related: ACTION-007, ACCESS-003

### INPUT-007 — Validate as early as the check can be made reliably: impossible characters at keystroke, single-field rules on leaving the field, cross-field rules as soon as both values exist and always before commit
- **Rule:** Reject impossible input immediately (digits-only fields). Validate format and existence when the user leaves the field (lot exists, PO matches supplier). Validate cross-field and consequential rules inline as soon as the inputs are present, and never later than commit. Invalid data is never silently accepted. Choose timing so users learn of the problem before it compounds but aren't interrupted mid-keystroke.
- **Meaning:** Validation timing is a design choice: too early is annoying, too late causes bad records the user then has to hunt down.
- **Factory Ledger:** Qty field: non-numeric keystrokes blocked. Lot code: on blur, "Lot not found / Lot on hold" and LAT-policy mismatch flagged before moving on. Allocation: "Allocating 50 cases exceeds available 42" appears the moment both values are known — a block when ALLOCATIONS_ENFORCED is on, a warning when off.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule (validate) + strong recommendation (timing)
- **Audit:** For each field, is there a defined validation moment (keystroke / blur / inline cross-field / submit), does the user learn of an invalid value before leaving the field, and is invalid data ever silently accepted?
- **Sources:** M1 INPUT-007 + A INPUT-005

### INPUT-008 — Numeric fields accept only numbers and always show their unit and format
- **Rule:** Constrain numeric inputs to numeric characters (including pasted text); format output (decimals, thousands, %, currency) and display the unit adjacent to the value. Don't hardcode locale assumptions.
- **Meaning:** Prevents "12O" (letter O) and ambiguous magnitudes; consistent formatting prevents misreading quantities.
- **Factory Ledger:** Cases, units, lb/kg, prices. Fixed decimals for weights, none for case counts; "lb" beside the value.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Can any numeric field accept non-numeric input? Is the unit always visible next to the value?
- **Sources:** M1 INPUT-008 + A INPUT-005 + B INPUT-001

### INPUT-009 — Handle overflow deliberately: never clip silently; wrap or truncate with an indicator and expose the full value on hover/tap
- **Rule:** Long values wrap, or truncate with an ellipsis, and the full text is viewable in one step (tooltip on desktop, tap-to-expand or detail sheet on mobile). Silent clipping is prohibited.
- **Meaning:** Clipped identifiers lead to wrong selections.
- **Factory Ledger:** "Sweetened Toasted Coconut 25 lb – Blue Stripes" in table cells and pickers: middle-truncate keeping the distinguishing end; tooltip shows the full name.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Is any field or cell cutting off text with no ellipsis and no one-step way to see the full value?
- **Sources:** M1 INPUT-009 + A INPUT-008 · Governed for identifiers by DATA-004

### INPUT-010 — Show the keyboard that matches the field's content type; consider a focused large-keypad step for a single critical number
- **Rule:** Numeric fields open a numeric/decimal keypad; codes open the keyboard matching their format; email/URL fields their variants. For one critical value on mobile (cases packed, weight), consider a dedicated full-screen entry step with a large keypad.
- **Meaning:** Wrong keyboard adds taps and errors on every single entry; a big keypad is faster and more accurate with gloves.
- **Factory Ledger:** Quantity, case count, weight → numeric pad. Lot code → keyboard per LAT format. Never a full QWERTY for a count.
- **Platform:** Mobile · **Importance:** High · **Type:** Hard rule
- **Audit:** Tap every field on mobile — does the right keyboard appear each time?
- **Sources:** M1 INPUT-010 + B INPUT-001

### INPUT-011 — Prefer selection over typing; present a text field only when free text is truly required
- **Rule:** Whenever the set of valid values is known (customers, products, lots, reasons, locations, units), present a list, picker, chips, scan, or buttons instead of a text field — on desktop too. Reserve typing for genuinely open values. Essential on mobile, where typing is slow and error-prone.
- **Meaning:** Selection eliminates typos, enforces valid references, and is faster.
- **Factory Ledger:** Product = picker/search-select. Lot = scan or pick from open lots. No-show reason = buttons. Supplier = list. Free text only for notes.
- **Platform:** Both (Critical on Mobile) · **Importance:** Critical · **Type:** Hard rule
- **Audit:** For every text field: is the set of valid values enumerable? If yes, why is it a text field?
- **Sources:** M1 INPUT-011 + A INPUT-003 · Related: INPUT-016, INPUT-017

### INPUT-012 — Provide a one-tap Clear control in text fields
- **Rule:** Show a clear (×) at the trailing end of a field with content so users can erase in one tap.
- **Meaning:** Backspacing a wrong lot code character-by-character on a phone is slow.
- **Factory Ledger:** Lot code, search boxes, quantity fields on mobile.
- **Platform:** Mobile (primary); Desktop/Web for search fields · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Can any filled field be emptied in one tap?
- **Sources:** M1 INPUT-012

### INPUT-013 — Use the leading edge of a field for purpose cues and the trailing edge for functions
- **Rule:** A leading icon indicates what the field is for (barcode, search). Trailing controls offer actions (scan, lookup, clear).
- **Meaning:** Consistent placement makes affordances predictable.
- **Factory Ledger:** Lot field: leading barcode icon; trailing [Scan]. Search: leading magnifier; trailing clear.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Are in-field icons and buttons placed consistently across all forms (purpose left, action right)?
- **Sources:** M1 INPUT-013

### INPUT-014 — Pair free text with a list of choices (combo box / type-ahead select) when both are needed; free text never silently creates a master record
- **Rule:** When a value usually comes from a list but may need a new or exact entry, use a searchable select rather than a plain field or plain dropdown. Creating a new master record is an explicit "+ New" option.
- **Meaning:** Combines the speed of typing a few characters with the safety of choosing a valid record; prevents duplicate customers/products.
- **Factory Ledger:** Customer, product, supplier on desktop; type "Rest" → Restaurant Depot.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Do long-list selections support type-to-filter, and does free text ever silently create a new master record?
- **Sources:** M1 INPUT-014 · Related: SEARCH-004

### INPUT-015 — Steppers always show the value they change, are paired with a typed field when large changes are likely, and support accelerated increments
- **Rule:** A +/- control sits beside a visible, labelled value. When values vary widely, provide a typed field alongside. For large ranges, support fast increments (long-press acceleration on mobile, modifier-click or ×10 on desktop).
- **Meaning:** Steppers are ideal for small adjustments and terrible for entering 240; users need both paths.
- **Factory Ledger:** Case count: typed numeric field with +/- beside it for "one more case." Pallet count: stepper alone is fine.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** For every stepper, is the affected value adjacent and labelled? Can a user enter an arbitrary number without tapping repeatedly?
- **Sources:** M1 INPUT-015

### INPUT-016 — Use device and platform capabilities to eliminate data entry wherever possible, with the user's permission
- **Rule:** Prefer capturing data through camera/barcode scan, location, biometrics, and clipboard rather than typing. Ask permission with a clear reason (OTHER-002).
- **Meaning:** Every value the device can supply is one the user can't mistype.
- **Factory Ledger:** Scan lot/case barcodes instead of typing codes; biometric unlock instead of password on the floor phone; auto-stamp date/time/user.
- **Platform:** Mobile (primary); Desktop/Web (scanner input) · **Importance:** High · **Type:** Strong recommendation
- **Audit:** For each field the user types, could a scan, lookup, or device sensor have supplied it?
- **Sources:** M2 INPUT-016 · Related: INPUT-011, INPUT-017

### INPUT-017 — Never ask for data the system already knows or can derive
- **Rule:** Pre-fill from context, settings, permissions, the current record, or a scanned/selected reference. Each unnecessary field is a floor error waiting to happen.
- **Meaning:** Re-entering known data wastes time and introduces divergence between what the system knows and what was typed.
- **Factory Ledger:** Timestamp, logged-in operator, active work order, item from scanned lot, customer from selected SO, supplier from the PO, default location from device — all auto-filled, never typed.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** For each field on this form, could the system have filled it without asking?
- **Sources:** A INPUT-001

### INPUT-018 — Prefill sensible defaults; a default must be the most probable correct value, not merely a convenient one
- **Rule:** Default each field to the most likely value so users confirm rather than decide. A wrong default that gets accepted blindly is worse than no default — leave consequential fields blank when no reliable default exists.
- **Meaning:** Defaults reduce decisions and speed entry, but only when they're usually right.
- **Factory Ledger:** "Pack date" defaults to today; "Qty (cases)" defaults to the standard pallet count for that product; "Lot to consume" defaults to the FIFO lot but stays visibly overridable.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Does each field default to the most probable correct value, and is any default likely to be accepted blindly when wrong?
- **Sources:** A INPUT-002 (defaults clause)

### INPUT-019 — Required data is unmistakable and the commit action is gated until it's complete
- **Rule:** The Submit/Next control is unavailable until required fields are complete, and the user can see exactly which required inputs are missing.
- **Meaning:** Don't let an operator submit, fail, and re-enter everything.
- **Factory Ledger:** "Commit Ship" disabled until lot, qty, and SO line are set; missing fields highlighted and named.
- **Platform:** Both · **Importance:** Critical · **Type:** Strong recommendation
- **Audit:** Can the user tell exactly which required inputs are still missing, and is the commit action unavailable until they're provided?
- **Sources:** A INPUT-006 · Related: ACTION-002, FEEDBACK-008

### INPUT-020 — When pasted or dropped content contains more than the field needs, extract the relevant part
- **Rule:** Accept the richest content the field can use and pull out the meaningful portion rather than rejecting the paste.
- **Meaning:** Reduces cleanup and rejection.
- **Factory Ledger:** Pasting "LOT 26250-COC-01 — 40 cs" into the lot field yields `26250-COC-01`. Pasting an email signature into a contact field yields name + email.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** If the user pastes a messy string containing a valid value, does the field accept and clean it?
- **Sources:** A INPUT-007

### INPUT-021 — Match the text component to editability: read-only values look read-only; editable values look editable
- **Rule:** Show non-editable text as a plain label, not a disabled-looking input and not a silently editable one. Use a single-line field for short editable text and a text area for long editable text.
- **Meaning:** Users shouldn't have to tap to discover whether something can be changed.
- **Factory Ledger:** On a lot detail, LAT code and creation timestamp are labels. A value locked because the lot was consumed is shown as a label, not a grayed field that invites a tap.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Can the user tell at a glance which values are editable, and does any read-only value look like an input?
- **Sources:** C INPUT-001

### INPUT-022 — Hint text, entered values, and unavailable controls are visibly distinct
- **Rule:** Placeholder/hint text, user-entered values, and disabled/unavailable control text each use a distinct, consistent treatment so a user never mistakes a hint for a value or an unavailable control for an available one.
- **Meaning:** Misreading a hint as a value — or vice versa — produces wrong transactions.
- **Factory Ledger:** A quantity field showing placeholder "0" versus entered "0" is a data-integrity risk: placeholders are clearly hint-styled ("e.g., 1200"). A "Ship" button disabled because the SO isn't Factory Ready looks disabled with the reason nearby.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** On this form, can a user distinguish hint text from entered values, and available from unavailable controls, without interacting?
- **Sources:** D INPUT-001 · Related: INPUT-002, ACTION-002

---

## 5. Mobile & Touch Interaction

### TOUCH-001 — Primary and frequent controls live in the thumb zone (middle to bottom); the top bar holds navigation and secondary actions only
- **Rule:** Actions used most often, and actions taken one-handed, sit in the lower/middle display. The top bar is for back, related areas, and contextual secondary actions — never the primary commit action.
- **Meaning:** People hold phones and reach with the thumb; top-of-screen controls require regripping, which is slow and error-prone on a moving floor.
- **Factory Ledger:** Submit / Scan / +1 case anchored at the bottom; tab bar at the bottom. Top bar: back, "History," overflow (⋯). "Post Pack Run" is never top-right.
- **Platform:** Mobile · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Are the primary action and most frequent controls reachable with one thumb without regripping? Is the primary commit action absent from the top bar?
- **Sources:** M2 TOUCH-001 + B TOUCH-002

### TOUCH-002 — Support standard gestures (swipe back, swipe row actions, long-press); custom gestures are supplementary; every gesture has a visible equivalent
- **Rule:** Use the gestures users already expect. Add custom gestures only when the workflow needs them, and never as the only way to perform an action — gestures are invisible, so the same action must be reachable through a visible control.
- **Meaning:** Gestures remove taps for experienced users; visible equivalents keep them learnable.
- **Factory Ledger:** Swipe on an expected-receipt row → "Received" / "Reschedule," also on the row's detail. Swipe-to-void is fine as a shortcut, but Void is also a visible control. Don't invent a gesture for "consume lot."
- **Platform:** Mobile · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Is every gesture-triggered action also available through something visible?
- **Sources:** M2 TOUCH-002 + C TOUCH-002

### TOUCH-003 — Every tappable element has a hit region of at least 44×44 pt, padding around its glyph, visible hover/focus/selection states, and no overlap with neighbors
- **Rule:** Applies regardless of input method and on desktop tables too (a 16-px glyph still gets a 44-pt hit area). Avoid small/mini buttons in rows or stacks on touch screens.
- **Meaning:** Small or crowded targets cause wrong taps; in an operational system a wrong tap is a wrong transaction. Gloves and wet hands make this worse.
- **Factory Ledger:** Quantity steppers, lot rows, "Post" buttons all ≥44 pt with clear gaps; lot and SO rows on mobile are full-width targets; desktop row hover shows which row a click will hit.
- **Platform:** Both (stricter on Mobile) · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Is every interactive element ≥44×44 pt with clear separation, and is the hovered/focused item visibly indicated?
- **Sources:** B ACTION-002 + C TOUCH-001 · Related: LAYOUT-012

### TOUCH-004 — On phone screens the primary action is a full-width, bottom-anchored button that respects safe areas; no more than two text buttons or three icon buttons per row
- **Rule:** A full-width button is legitimate for a single dominant floor action if it aligns with safe areas and margins, sits clear of the home indicator, and is visually separated from content. If two actions must share a row, give them equal height and short labels. Multiple stacked full-width buttons dilute the benefit — extra actions move to a menu or above the fold.
- **Meaning:** Full-width buttons are easier to hit and unmistakably mark "the thing to do next"; crowded rows shrink targets.
- **Factory Ledger:** Packing screen → full-width "Post Pack" at bottom; "Back" / "Next" pair with equal heights; a row of three icon buttons (scan, note, more) is fine.
- **Platform:** Mobile · **Importance:** High · **Type:** Strong recommendation
- **Audit:** On mobile, is the primary action a full-width, bottom-anchored button respecting safe areas, and does any row hold more than two text buttons or three icon buttons?
- **Sources:** B TOUCH-001 + D ACTION-002 + D TOUCH-001 · Related: LAYOUT-015

### TOUCH-005 — Require an explicit select/edit mode before multi-selection, reordering, or bulk deletion on mobile lists
- **Rule:** Normal taps navigate; edit mode selects. A single accidental tap can never start a bulk or destructive operation.
- **Meaning:** Prevents accidental bulk actions from ordinary browsing taps.
- **Factory Ledger:** Choosing several lots to allocate enters a "Select" mode with checkboxes and a clear "Allocate 3 lots" action; a plain tap opens the lot.
- **Platform:** Mobile · **Importance:** Medium · **Type:** Situational idea
- **Audit:** Can a single accidental tap in a list start a bulk or destructive operation?
- **Sources:** C TOUCH-004 · Related: DRAG-004

---

## 6. Feedback, Status & Progress

### FEEDBACK-001 — Any operation that could appear stalled shows a progress indicator; commit buttons show in-progress state and become non-repeatable until the server responds; the indicator disappears exactly on completion
- **Rule:** Show progress during loads, saves, syncs, and long operations, within ~1 second of the action. When a button's action isn't instant, show a spinner inside the button, change its label to the in-progress form ("Post" → "Posting…"), and disable it until the result returns; then show the outcome. The indicator is transient: visible only while the operation runs.
- **Meaning:** Silence during a wait reads as "frozen"; nothing visibly changing invites a second tap and a duplicate submission; a lingering indicator reads as "still running."
- **Factory Ledger:** "Post Receipt" → "Posting…" with spinner, disabled until the API responds. Same for "Allocate," "Ship," "Emit Trace," `/make` commits, trace queries, report generation.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Does every network-bound action show progress within ~1 second, does every submit/commit button become non-repeatable until the server responds, and does the indicator disappear exactly when the operation ends?
- **Sources:** M1 FEEDBACK-001 + B STATUS-001 · Related: ACTION-002

### FEEDBACK-002 — Prefer determinate progress; be accurate and evenly paced; upgrade to determinate when possible; never switch spinner↔bar
- **Rule:** Use a determinate bar/ring whenever the total is known. Report advancement honestly with even pacing (no 90%-then-stall). If an indeterminate task becomes measurable, switch to determinate. Do not change indicator shape mid-operation.
- **Meaning:** Determinate progress lets users decide whether to wait, switch tasks, or abandon. Dishonest pacing feels deceptive; shape changes feel like a glitch.
- **Factory Ledger:** Bulk imports, multi-lot trace walks, matrix exports: "Processing lot 12 of 40."
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** For each multi-step operation, is the total known and shown? Does progress advance at a believable pace? Does the indicator keep one shape throughout?
- **Sources:** M1 FEEDBACK-002

### FEEDBACK-003 — Progress indicators keep moving; a stall is explained with what to do next
- **Rule:** A visibly stationary indicator is treated by users as a freeze. After a defined timeout without progress, replace or annotate the indicator with a message explaining the problem and the user's options.
- **Meaning:** Users need to distinguish "slow" from "broken."
- **Factory Ledger:** API unreachable: after a timeout, "Server not responding — your entry is saved locally. [Retry] [Keep working offline]" rather than an endless spinner.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** After N seconds without progress, does the UI change to explain the stall and offer a next step?
- **Sources:** M1 FEEDBACK-003 · Related: ERROR-002

### FEEDBACK-004 — Add a progress description only when it adds real information; skip vague labels and skip labels on small spinners
- **Rule:** If a label accompanies progress, make it specific in domain terms. Avoid "Loading…" / "Please wait." Small inline spinners generally need no label.
- **Meaning:** Vague labels add noise; specific labels reduce anxiety and aid support.
- **Factory Ledger:** "Generating forward trace for LOT 26-0907-C…" vs. "Loading."
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does every progress label say *what* is happening in domain terms, or is it a placeholder word?
- **Sources:** M1 FEEDBACK-004

### FEEDBACK-005 — Progress and status appear in a consistent location
- **Rule:** Define one convention: action progress appears on the action button itself; background sync status appears in a fixed header slot. Apply across all screens and both platforms.
- **Meaning:** Users learn where to look; inconsistency forces them to hunt.
- **Factory Ledger:** Same placement across dashboard tabs, scheduler, and mobile.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Is there a documented single location for (a) action progress and (b) background sync status, and do all screens follow it?
- **Sources:** M1 FEEDBACK-005 · Instance of LAYOUT-002

### FEEDBACK-006 — Use small inline spinners for background or space-constrained operations
- **Rule:** For asynchronous background tasks or progress inside a small control (field, button), use a compact spinner rather than a full-screen or full-width indicator; background work never blocks the UI.
- **Meaning:** Background work shouldn't dominate the screen.
- **Factory Ledger:** Customer/product lookup in a combo field, "checking lot availability" next to a quantity input, background refresh of Recent Entries.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Do background refreshes avoid blocking the UI, and does inline validation show compact progress in place?
- **Sources:** M1 FEEDBACK-006

### FEEDBACK-007 — Refresh data automatically without shifting content under the user; offer manual refresh; show last-updated time where staleness could cause a wrong transaction
- **Rule:** Keep operational lists current without user action (push or polling), but never move items the user may be about to tap (LAYOUT-020) — buffer updates behind a "N new — tap to refresh" cue or append without shifting. Provide manual refresh (pull-down on mobile, button/shortcut on desktop). If the refresh control has a label, use it for value ("Updated 2 min ago"), not instructions.
- **Meaning:** In a multi-user system, stale views cause duplicate or conflicting transactions; users shouldn't be responsible for every update, but a refresh that moves their target causes a wrong entry.
- **Factory Ledger:** SO lists, Expected Receipts, Today So Far, and the production board stay in sync between Luz and Arturo. "Last synced 3:42 PM" so a user on flaky Wi-Fi knows how fresh the data is.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Does every operational list update without manual action, without displacing on-screen items? Is last-sync time visible where staleness could cause a wrong transaction?
- **Sources:** M1 FEEDBACK-007 (expanded by C LAYOUT-003) · Related: DATA-012

### FEEDBACK-008 — Every state change gets a clear signal: control availability, content changes, and choices use standard patterns
- **Rule:** Show when controls are available or disabled (and why, where useful); visibly indicate when content changes; use standard alert/choice patterns rather than custom inventions.
- **Meaning:** Users need to know what they can do, what just happened, and what the system is asking them.
- **Factory Ledger:** "Ship" disabled with inline text "3 lines not yet packed." When a row's status changes to Allocated, the chip changes and briefly highlights. Confirmations use the same dialog pattern every time.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** For each control, is its enabled/disabled state visible and explained? Does every change to on-screen data produce a perceptible signal?
- **Sources:** M2 FEEDBACK-008 · Related: ACTION-002, NOTIFY-005, ERROR-004

### FEEDBACK-009 — Selection feedback matches the purpose of the list: navigation lists persistently highlight the open item; choice lists mark the chosen items
- **Rule:** In navigation lists, keep the currently open item highlighted so the user's path is visible — including a de-emphasized-but-visible selection when focus moves to a detail panel. In option/choice lists, highlight briefly and then mark the chosen item (checkmark) with a running total where relevant.
- **Meaning:** Feedback should answer the user's actual question: "where am I?" vs. "what did I pick?"
- **Factory Ledger:** Desktop split view: the open SO row stays highlighted (dimmed when the panel has focus) while its detail shows. Lot-allocation picker: checked lots show a checkmark and "3 lots · 48 cases." Active filter chips look selected.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** After selecting an item, can the user tell whether it's "open" or "chosen," and is the current item still indicated after scrolling or when another panel has focus?
- **Sources:** C FEEDBACK-001 + D DATA-002 (selection clause)

### FEEDBACK-010 — Animate list changes caused by the user's own action so they can track what moved
- **Rule:** When the user inserts, deletes, or reorders an item, use a brief standard animation so they see what changed and where it went. Keep it short (fast repetitive floor work). Background refreshes do not animate items into the user's path (LAYOUT-020).
- **Meaning:** Instant changes leave users unsure the action happened.
- **Factory Ledger:** When a lot is consumed and leaves the Active list, animate the row out; when a receipt posts, animate the row in.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** After the user adds, removes, or reorders an item, is the change visibly tracked rather than appearing instantly elsewhere?
- **Sources:** C FEEDBACK-002

### FEEDBACK-011 — Status, state, and interactivity are never communicated by color alone
- **Rule:** Every cue that uses color is also conveyed by a text label, icon/shape, or position, so it's understood by colorblind users, in poor lighting, and in monochrome contexts (print, screenshot, grayscale). Focus and selection states use outlines, scale, or fills, not color shifts alone.
- **Meaning:** A red dot alone is invisible to some users and ambiguous to everyone under bad lighting.
- **Factory Ledger:** "On Hold" = red + lock icon + the words ON HOLD. Factory Ready = green check + "Ready." Shortfall = warning icon + "Short 40." Desktop focus = visible outline.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** If this screen were viewed in grayscale, would every status, state, and interactive element still be identifiable?
- **Sources:** D FEEDBACK-001 · Related: ACCESS-002

### FEEDBACK-012 — One meaning per color, app-wide; the interactive color is used only on interactive things; status colors are immune to themes
- **Rule:** Assign each semantic color one meaning (e.g. green = complete/ready; amber = attention; red = blocked/error/void; accent = interactive/primary) and never reuse it. Maintain a status-color table in the standards. Meaning-bearing colors are not subject to theme or accent overrides.
- **Meaning:** Inconsistent color semantics teach users to ignore color.
- **Factory Ledger:** If the accent blue means link/button, non-interactive lot codes aren't blue. Red is never used for "important customer" branding.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Does every color on this screen carry the same meaning it carries everywhere else, and is the interactive color used only on interactive things?
- **Sources:** D FEEDBACK-002 · Related: ACTION-006, OTHER-010

### FEEDBACK-013 — Color-carried status goes on bold text, badges/chips, or filled shapes — never thin, small, regular-weight text or hairlines
- **Rule:** When color signals status or emphasis, apply it to bold text, a chip, or a filled shape.
- **Meaning:** A thin red hairline or 11-px regular red text doesn't register.
- **Factory Ledger:** Status words in tables render as chips or bold text; row-level status as a leading bar or background tint.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Is every color-coded status rendered as bold text, a badge, or a filled shape?
- **Sources:** D FEEDBACK-003

### FEEDBACK-014 — Background and row tints only when they communicate; no saturated long-lived backgrounds
- **Rule:** Use tints for status or grouping, not decoration. Avoid full-screen or large saturated backgrounds on screens that stay open for long periods.
- **Meaning:** A tint means something or it's noise; saturated backgrounds fatigue users and reduce contrast.
- **Factory Ledger:** Tint a row for "on hold"; don't tint an entire production board red because one line is behind. Floor dashboards that stay up all shift use neutral backgrounds.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does every colored background convey something specific, and are long-lived screens neutral?
- **Sources:** D FEEDBACK-004

---

## 7. Errors, Confirmation & Recovery

### ERROR-001 — Let users halt long operations when safe; pause when interruption loses work; confirm cancellation that discards progress
- **Rule:** Provide Cancel on interruptible operations with no side effects. If stopping could lose partial work, offer Pause alongside Cancel. When cancelling would lose progress, confirm with an alert that offers Resume. Never allow a silent cancel that leaves a partial transaction.
- **Meaning:** Users need an exit from a slow process, but must not accidentally discard work or half-apply a transaction.
- **Factory Ledger:** Bulk lot import: Cancel before commit is free; once posting begins, either make it atomic or show "Stopping now will leave 3 of 8 lots unposted. [Resume] [Stop anyway]."
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** For each long operation, is there a cancel path, is it clear whether cancelling is safe, and is a partial-commit state impossible or clearly surfaced with a resume option?
- **Sources:** M1 ERROR-001 · Related: NAV-002

### ERROR-002 — Preserve in-progress work across app switches, backgrounding, page refreshes, rotation, and interruptions
- **Rule:** A half-filled form or half-scanned pallet survives switching to the camera, WhatsApp, a call, a screen lock, rotation, or a browser refresh, and the user returns to exactly where they were. Drafts are clearly marked as unsaved.
- **Meaning:** Losing partially entered data is the most expensive friction: the work is redone and often re-entered wrong.
- **Factory Ledger:** Draft state of the coconut form or a pack run is held locally and restored on return. A desktop SO edit survives an accidental refresh.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Fill half a form, switch apps / lock screen / rotate / refresh, return — is everything still there, in the same place?
- **Sources:** M2 ERROR-002 + D LAYOUT-011 (rotation clause) · Related: FEEDBACK-003

### ERROR-003 — Build forgiveness in: routine actions are reversible, committed transactions have an equally easy correction path, and recovery never requires re-entering data
- **Rule:** Provide undo or return-to-previous-state for ordinary actions. Where a record is intentionally immutable (a posted transaction), provide an explicit correction path (void/reverse with audit trail) as easy as the original entry, with correct cascade.
- **Meaning:** Knowing mistakes are cheap lets users move fast and confidently.
- **Factory Ledger:** "Undo" toast after deleting a line. A mis-posted pack run has a one-tap "Reverse this run" that creates a pre-filled compensating entry. Voids cascade so the trace stays clean.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** For every action, what is the recovery path if it was wrong, how many steps does it take, and does it require retyping anything?
- **Sources:** M2 ERROR-003 · Related: DRAG-003, ACTION-008

### ERROR-004 — Choice dialogs for choices the user's action requires; alerts for unexpected problems — never the reverse
- **Rule:** When a user-initiated action needs a clarifying choice, present a choice dialog (action sheet / confirmation dialog) offering those choices. Reserve alerts for problems or changes the user didn't initiate. Errors are never delivered as notifications or toasts alone (NOTIFY-004).
- **Meaning:** Choice dialogs are expected and offer alternatives; alerts are interruptive and signal something went wrong. Mixing them trains users to dismiss alerts reflexively.
- **Factory Ledger:** Leaving a half-filled receiving form → sheet: "Save Draft / Discard / Cancel." Server rejects a post → alert.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Is every dialog triggered by a user action a choice dialog, and every unexpected condition an alert?
- **Sources:** B ERROR-001 · Related: NOTIFY-004, FEEDBACK-008

### ERROR-005 — Interrupt sparingly: confirm only consequential or irreversible actions; routine reversible actions proceed without interruption
- **Rule:** Every confirmation must protect against a consequential or irreversible outcome; otherwise remove it.
- **Meaning:** Frequent interruptions train people to tap through without reading — defeating the confirmations that matter.
- **Factory Ledger:** No "Are you sure?" on adding a line. Confirm on void, delete, ship-partial, or overriding a lock.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Is each confirmation in this flow protecting against a consequential or irreversible outcome?
- **Sources:** B ERROR-002 · Related: ERROR-003

### ERROR-006 — Dialog title is one specific line; add a message only when it's needed to choose correctly
- **Rule:** Keep the title to one line naming the action; add body text only for a non-obvious consequence.
- **Meaning:** Long titles truncate; unnecessary messages slow the decision.
- **Factory Ledger:** "Discard this receipt?" — not a paragraph. Add a message only for "Lot 26-0907 is already allocated to SO 1042."
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Is every dialog title one line, and does each message add information needed to decide?
- **Sources:** B ERROR-003 · Related: ACCESS-004

### ERROR-007 — Any dialog whose options could destroy or alter data includes a Cancel that does nothing, in the same predictable position everywhere
- **Rule:** Cancel dismisses with no change. Put it in one position (bottom of the sheet on mobile; the conventional dialog slot on desktop).
- **Meaning:** Users need a guaranteed safe exit they can find without reading; a Cancel that secretly saves or moves around is a trap.
- **Factory Ledger:** Void/Discard/Unallocate sheets always end with Cancel at the bottom on mobile; desktop dialogs place Cancel in the same slot every time.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Does every dialog with a destructive option have a Cancel that does nothing, in the standard position?
- **Sources:** B ERROR-004 · Related: ACTION-006

### ERROR-008 — Choice dialogs offer at most three choices plus Cancel and never scroll
- **Rule:** If more options are needed, redesign the step (a dedicated screen, or a menu the user opens deliberately).
- **Meaning:** Fewer options mean faster, more accurate decisions; scrolling a sheet risks tapping the wrong button.
- **Factory Ledger:** Insufficient stock on allocation → "Allocate Available / Backorder Remainder / Cancel," not seven options.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Does any choice dialog have more than three options plus Cancel, or require scrolling?
- **Sources:** B ERROR-006

### ERROR-009 — Choices that arise from an action appear in a dialog; menus appear only when the user opens them
- **Rule:** Don't pop a menu unexpectedly, and don't bury an action's required follow-up choice in a menu.
- **Meaning:** People expect menus when they open them and dialogs when they act; swapping the two feels unpredictable.
- **Factory Ledger:** Tapping "Ship" with partial stock → dialog with ship options. Row "⋯" → menu of secondary actions.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Do choice dialogs appear only in response to user actions, and menus only when the user opens them?
- **Sources:** B ERROR-007

### ERROR-010 — Error messages and identifiers are selectable and copyable in one gesture
- **Rule:** Any displayed text a user might need elsewhere — error messages, lot codes, PO/SO numbers, tracking numbers, addresses — must be copyable directly. Avoid non-selectable rendered text or controls that swallow selection.
- **Meaning:** Users shouldn't retype (and mistype) values the screen already shows.
- **Factory Ledger:** An API error in the dashboard is copyable so it can be pasted to Claude Code verbatim. A lot code has a copy affordance for WhatsApp, customer email, or trace search.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Can the user select or copy an error message, lot code, or order number directly from the screen?
- **Sources:** C ERROR-001

---

## 8. Search & Discovery

### SEARCH-001 — Search has a primary, persistent position reachable from anywhere in one action
- **Rule:** If finding records is a core task, search is never buried in a menu. Desktop: always-visible search field in the header. Mobile: a dedicated tab-bar or top-bar item.
- **Meaning:** Lot, SO, and item lookup are core tasks; each extra tap to reach search is paid dozens of times a day.
- **Factory Ledger:** Global search in the desktop header; a Search tab on mobile, not behind a hamburger.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** From any screen, can the user reach global search in one tap/click without leaving their context?
- **Sources:** A SEARCH-001

### SEARCH-002 — One global search finds every record type; section-level search boxes are filters on the current view
- **Rule:** Never have two "global" searches that return different things; every non-global search box behaves as a filter on its own view.
- **Meaning:** Users want a single known place to search everything.
- **Factory Ledger:** Global search returns lots, SOs, POs, items, customers, receipts. The Orders screen's search box filters only orders.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Is there exactly one place to search across all record types, and is every other search box clearly a filter on its own view?
- **Sources:** A SEARCH-002

### SEARCH-003 — The current search scope is unmistakable before typing
- **Rule:** Placeholder text, a scope selector, or a title states exactly what is being searched.
- **Meaning:** Prevents an operator thinking a lot doesn't exist because they searched the wrong scope.
- **Factory Ledger:** Placeholder "Search open sales orders," not "Search." On a lot-history page, "Search this lot's events."
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Can the user tell, before typing, exactly which records this search will cover?
- **Sources:** A SEARCH-003

### SEARCH-004 — Reduce typing with recent searches, suggestions, completions, and near-miss correction
- **Rule:** Show recents before typing; predict as they type; auto-correct near-misses; auto-complete identifier prefixes.
- **Meaning:** Big win on the floor where typing is slow and gloved.
- **Factory Ledger:** Typing "coc" suggests "Sweetened Coconut 5lb," "Coconut Toasted." Recent lots appear before typing.
- **Platform:** Both (larger payoff on Mobile) · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Can the user reach a common target with 3 or fewer keystrokes?
- **Sources:** A SEARCH-004 · Related: INPUT-014

### SEARCH-005 — Let users narrow by structured attributes; when filters are many or nested, use a persistent filter panel beside the content
- **Rule:** Beyond free text, allow narrowing by the attributes users think in (record type, status, date range, customer, item) via visible chips, tokens, or a filter panel. A segmented control holds only a handful of choices; on desktop, complex or nested filtering gets a persistent, visible panel so users can move between filters and content freely.
- **Meaning:** Free-text tricks don't scale; hidden filters get forgotten.
- **Factory Ledger:** Filter chips: Lot / SO / PO; Open / Shipped / Voided; date range; tokens like `customer:Restaurant Depot`. Trace/recall queries (product + lot range + date + customer) → filter sidebar.
- **Platform:** Both (panel: Desktop/Web) · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Can the user narrow results by the attributes they think in, and where filters exceed one segmented control, is there a persistent visible panel?
- **Sources:** A SEARCH-005 + B NAV-003

### SEARCH-006 — Search history is per-user and clearable
- **Rule:** On shared devices, one user's history must not leak to the next; provide a clear-history action.
- **Meaning:** Floor tablets are shared; others may see the screen.
- **Factory Ledger:** Per-user recents on shared floor devices.
- **Platform:** Both · **Importance:** Low/Contextual · **Type:** Situational idea
- **Audit:** On a shared device, does one user's search history leak to the next, and can it be cleared?
- **Sources:** A SEARCH-006

### SEARCH-007 — Support find-within-page on long content
- **Rule:** Long tables and documents get in-view find/highlight (Ctrl+F-style) on desktop; a filter field at the top of long lists serves the same purpose on mobile.
- **Meaning:** Record-level search doesn't help locate a value inside a 400-row report.
- **Factory Ledger:** A trace report or receiving log with 50+ rows.
- **Platform:** Desktop/Web (Mobile via list filter) · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** On any screen with 50+ rows, can the user locate a specific value without scrolling manually?
- **Sources:** A SEARCH-007

### SEARCH-008 — Every primary record has a stable, shareable link that opens straight to it
- **Rule:** Lots, SOs, POs, receipts, and shipments are addressable by URL so they can be pasted into WhatsApp/email or encoded in a QR label and open directly.
- **Meaning:** Content that can be reached from outside the app removes navigation entirely.
- **Factory Ledger:** A QR on a pallet label opens the lot; a link in a WhatsApp message opens the SO (see NAV-001).
- **Platform:** Both · **Importance:** Medium · **Type:** Situational idea
- **Audit:** Does every primary record have a shareable, stable link that opens straight to it?
- **Sources:** A SEARCH-008

---

## 9. Lists, Tables & Dense Operational Data

### DATA-001 — Present text-based records as rows; use card grids only for image-based or size-varying items
- **Rule:** Lists and tables scan and compare faster than card grids. On mobile, "rows" may be compact stacked cards — one per record, consistently structured, not a mosaic.
- **Meaning:** Card grids scatter text and waste space.
- **Factory Ledger:** Lots, receipts, SOs, shipments, trace events are rows. A card grid is only justified for a product catalog with photos.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Is any text-based operational record displayed in a card grid where a list or table would scan faster?
- **Sources:** C DATA-001

### DATA-002 — Use standard, predictable list and table layouts
- **Rule:** Avoid novel layouts that draw attention to themselves or make users learn how to read the screen. Custom visualizations are reserved for genuinely spatial or temporal data.
- **Meaning:** Familiar layouts are invisible; clever ones cost attention.
- **Factory Ledger:** Production board, lot list, receipts table read like any standard table. The pack-format calendar is a justified exception.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does the user need to learn how to read this layout before they can use it?
- **Sources:** C DATA-002 · Related: OTHER-004

### DATA-003 — Rows carry only identifying and decision-critical values; detail belongs in the detail view
- **Rule:** Keep row text short; no row wraps past two lines.
- **Meaning:** Short rows scan faster and truncate less.
- **Factory Ledger:** Lot row = LAT code, product, qty, status, age — not supplier notes or audit trail.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Does any list row wrap past two lines or include text the user doesn't need to choose between rows?
- **Sources:** C DATA-003

### DATA-004 — Minimize truncation; never truncate an identifier so two records look alike, and always keep a one-step path to the full value
- **Rule:** Avoid truncating text in lists, especially at larger text sizes; let meaningful labels wrap instead of clipping. When truncation is unavoidable, keep the distinguishing part visible (middle-truncate where both ends carry meaning) and expose the full value in one step (detail, tooltip, expand). Aim to show as much useful text at the largest text size as at the default.
- **Meaning:** A cut-off identifier is worse than useless: it looks like a different, valid identifier. Operators match records by eye.
- **Factory Ledger:** LAT codes and supplier lot numbers share prefixes and differ at the end — end-truncation ("LAT-260907-GRAN-…") hides exactly the distinguishing part; middle-truncate ("LAT-2609…-003") or guarantee the full code fits. Same for customer names differing by suffix and SKUs differing only in size. Product names wrap in cards.
- **Platform:** Both (Critical on Mobile) · **Importance:** Critical · **Type:** Hard rule
- **Audit:** At the narrowest supported width and the largest text size, can two records that differ only at the end of a code or name still be told apart, and can the full value be seen in one step?
- **Sources:** C DATA-004 + D DATA-001 · Related: INPUT-009

### DATA-005 — Every column has a descriptive header with unit; every headerless list has a context label
- **Rule:** Headers are nouns or short noun phrases, Title Case, no ending punctuation, including the unit where relevant. A single-column list without headers gets a section label saying what it lists and how many.
- **Meaning:** Headers are the legend for the table.
- **Factory Ledger:** "Qty (cases)," "Received," "LAT Code," "Status" — not "Amount," "Date," "Code." Mobile list headed "Today's Receipts · 7."
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does every column say what it contains and in what unit, and does every list say what it's a list of?
- **Sources:** C DATA-005

### DATA-006 — One consistent table style: visible grouping, alternating row shading on wide tables, standard separators/headers, and a selection that stays identifiable when another panel has focus
- **Rule:** Communicate grouping with section headers/footers and spacing. In wide multicolumn tables use alternating row backgrounds so the eye can follow a row across columns. Define one table style from the token system (header text, separators, alternating rows if used, selected-row treatment, de-emphasized selection when the table isn't the active pane) and use it for every table.
- **Meaning:** Grouping reduces scanning effort; row shading prevents reading the wrong row's value; a selection that vanishes when focus moves breaks the link between list and detail.
- **Factory Ledger:** Group Expected Receipts by supplier or day. The allocations matrix and SO line tables use alternating shading. Lot tables and SO tables share one style.
- **Platform:** Both (alternating shading: Desktop/Web) · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** In wide tables, can the user follow one row across all columns without losing it? Are related rows visibly grouped? Do all tables use the same treatments?
- **Sources:** C DATA-006 + D DATA-002 · Related: FEEDBACK-009

### DATA-007 — One consistent row anatomy per record type
- **Rule:** Define a standard row structure — leading status/icon, primary text, secondary text, trailing accessory (value, chevron, or action) — and use it for the same record type everywhere.
- **Meaning:** Once users learn where status and the key value sit, every list reads faster.
- **Factory Ledger:** A lot row looks the same on the Lots screen, in trace results, and in the allocation picker: status dot · LAT code · product/qty · chevron.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does a given record type use the same row layout in every list where it appears?
- **Sources:** C DATA-007 · Instance of LAYOUT-002

### DATA-008 — Sortable columns with reversible direction, a visible active-sort indicator, and a default sort matching the most common task
- **Rule:** Click a header to sort; click again to reverse; always show which column and direction is active. Mobile uses a sort control instead of clickable headers.
- **Meaning:** Sorting is how users find the oldest lot or the largest open order without a filter UI.
- **Factory Ledger:** Lot table sortable by age (FIFO check), qty, product; SO table by ship date, customer, status. Lots default oldest-first.
- **Platform:** Desktop/Web (Mobile: sort control) · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Can the user sort by the columns they decide on, reverse it, and see which sort is active?
- **Sources:** C DATA-008

### DATA-009 — Resizable columns on desktop tables, persisted per user where practical
- **Rule:** Let users widen what they're focused on and reveal clipped values without changing browser zoom.
- **Meaning:** Data width varies; fixed columns clip the wrong thing for someone.
- **Factory Ledger:** Office user widens "Customer" or "Notes"; a trace investigation widens "LAT Code."
- **Platform:** Desktop/Web · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Can the user widen a clipped column without changing browser zoom?
- **Sources:** C DATA-009

### DATA-010 — Show nested data as an expandable outline with expand-all/collapse-all
- **Rule:** For hierarchical data, use a tree with disclosure controls rather than a flat table or repeated drill-downs. Simplified on mobile.
- **Meaning:** Users see the structure and expand only what they need, staying on one screen.
- **Factory Ledger:** Forward/backward trace results (lot → consumed into → packed as → shipped on). SO → lines → allocated lots. Expand-all for recall investigations.
- **Platform:** Both (Desktop primary) · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Is parent-child data shown as an expandable tree, or must the user navigate away to see children?
- **Sources:** C DATA-010 · Related: NAV-010

### DATA-011 — Allow direct reordering where order carries meaning
- **Rule:** Let users reorder directly (drag on desktop; edit-mode handles on mobile) even if they can't add or remove items in that view. Not needed where order is system-determined.
- **Meaning:** Order is data; users need direct control rather than a workaround field.
- **Factory Ledger:** Reorder the production board queue or a pick sequence. Not for FIFO by receipt date.
- **Platform:** Both · **Importance:** Medium · **Type:** Situational idea
- **Audit:** Where sequence matters, can the user reorder directly rather than editing a "priority" number?
- **Sources:** C DATA-012 · Related: DRAG-001, TOUCH-005

### DATA-012 — One date/time format system-wide; time zones correct; live values update themselves
- **Rule:** Use one date/time format everywhere, and let relative/elapsed values ("last updated," lot age, running timers) update without manual refresh.
- **Meaning:** Inconsistent formats cause misreads; a stale "2 min ago" lies.
- **Factory Ledger:** Lot age ("14 d"), "Last synced 3 min ago," a running timer on an in-progress batch. Never mixing 9/7 and 7/9 styles.
- **Platform:** Both · **Importance:** Medium · **Type:** Situational idea
- **Audit:** Are dates formatted identically on every screen, and do relative/elapsed times update without user action?
- **Sources:** C DATA-013 · Related: FEEDBACK-007

---

## 10. Accessibility & Readability

### ACCESS-001 — Honor user text-size, bold-text, and light/dark settings; the layout survives the largest text size with hierarchy intact
- **Rule:** Text scales with system/browser text-size settings (relative units) and honors bold-text preferences; support dark mode and both orientations where applicable. At larger sizes: meaningful icons scale with text; inline/multi-column layouts stack and tables reduce columns rather than overflow; truncation is minimized (DATA-004); hierarchy and the position of primary elements stay the same; chrome such as tab labels need not scale. Assume viewing at arm's length, possibly in poor lighting. Verify at the largest accessibility size and at 200% browser zoom.
- **Meaning:** Users pick settings that work for their eyes and environment; an app that ignores them fails exactly the people who need help.
- **Factory Ledger:** An operator with large text still sees lot code, quantity, and Confirm without horizontal scrolling; rows reflow, nothing clips. Desktop at 200% zoom doesn't turn tables into unusable overflow. Dark mode for a dim warehouse.
- **Platform:** Both (Mobile primary) · **Importance:** High · **Type:** Hard rule
- **Audit:** At the largest system text size (mobile) or 200% zoom (desktop), and in dark mode, is every task still completable without overlap, clipping, or horizontal scrolling, and is the primary element still at the top?
- **Sources:** M2 ACCESS-001 + D ACCESS-004 + C ACCESS-001 (text-size clause) · Related: LAYOUT-003, LAYOUT-005

### ACCESS-002 — Design for everyone from the start; accessibility is a baseline requirement, not a later fix
- **Rule:** Consider the full range of users' abilities, experience levels, and language backgrounds during design. Accessibility is part of "done": color is never the sole carrier (FEEDBACK-011), icon-only controls have names (ACCESS-010), and messages are simple enough for a user whose first language isn't English.
- **Meaning:** Retrofitting accessibility is expensive and incomplete; designing inclusively usually makes the product better for everyone.
- **Factory Ledger:** An accessibility checklist applied to every new screen before ship.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Is there an accessibility checklist applied to every new screen before ship?
- **Sources:** M2 ACCESS-002

### ACCESS-003 — Every task is operable by touch, keyboard, and pointer, and every field accepts scan, paste, and pick as well as typing
- **Rule:** No task requires one specific input method: keyboard-only completion on desktop, touch-only on mobile. A hardware or camera scanner, clipboard paste, and drag all feed the same fields. Consider voice where it removes hands-on effort.
- **Meaning:** Different people and situations (gloves, wet hands, one hand free) call for different inputs; a field that only accepts typed input is a bottleneck.
- **Factory Ledger:** Full keyboard operation of desktop forms and tables (INPUT-006). Bluetooth scanner acts as keyboard input into the lot field; paste a lot code from WhatsApp into the same field.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Can each workflow be completed without a mouse on desktop and without a keyboard on mobile? Can the user get a value into this field by scan, paste, and pick — not only by typing?
- **Sources:** M2 ACCESS-003 + A INPUT-004 · Related: INPUT-016

### ACCESS-004 — Use the fewest exact words for every label, button, and message
- **Rule:** A label names the thing; a button names the outcome; a message states what happened and what to do. Remove filler.
- **Meaning:** Concise wording reads faster, translates better, and fits small screens.
- **Factory Ledger:** "Post" not "Click here to submit this record." "Lot not found" not "An error occurred while attempting to locate the requested lot."
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Can any label or message lose words without losing meaning?
- **Sources:** M2 ACCESS-004 · Related: NOTIFY-007, ERROR-006

### ACCESS-005 — A defined type scale with few named styles, one primary typeface plus a tabular-figure style for codes and numbers, using screen-legible system fonts
- **Rule:** Define a small set of named text styles (e.g. Title, Headline, Body, Subhead, Footnote, Caption), each with fixed size/weight/line-height, and use them exclusively. One primary typeface; at most a monospace/tabular-figure style for identifiers and quantities so digits align. Bold is an extra hierarchy step within a style, not a new style. Prefer system font stacks, which are tuned for screens and inherit user text-size behavior; any custom font is verified legible at all supported sizes on the actual devices.
- **Meaning:** Too many typefaces and ad-hoc sizes obscure hierarchy and look inconsistent; custom fonts often break at large or small sizes.
- **Factory Ledger:** Headline for record identifiers, Body for content, tabular style for lot codes and quantities, Caption for audit metadata. Same scale on dashboard and mobile.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Is every piece of text set in one of the named styles, and are no more than two typefaces in use?
- **Sources:** D ACCESS-005 + C ACCESS-001 (font clause) · Related: LAYOUT-005

### ACCESS-006 — Fixed minimum and default text sizes, set for floor conditions; nothing operational below the minimum
- **Rule:** Define a default body size and a hard minimum for any text a user must read to act. Apple's baselines (mobile 17 pt default / 11 pt min; desktop 13 pt / 10 pt) are the floor; Factory Ledger's context (arm's length, gloves, dust, motion, glare) argues for defaults at or above these and an operational minimum well above them. Thinner typefaces need larger sizes. **Proposed, to be fixed in the standards:** mobile body ≥ 17 px with lot codes and key quantities larger; desktop body 14–16 px; nothing operational below 12 px; 11-px captions only for non-operational metadata.
- **Meaning:** Text readable at a desk may not be readable on a phone held at arm's length on the floor.
- **Factory Ledger:** Lot codes and quantities at floor viewing distance in a bright warehouse.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Is any text the user needs to read to complete a task smaller than the standard's minimum, and is body text at the defined default?
- **Sources:** D ACCESS-001 + M2 ACCESS-001 (arm's-length clause)

### ACCESS-007 — Use Regular through Bold weights only; never Ultralight, Thin, or Light
- **Rule:** De-emphasize secondary text with color and size, not a lighter weight.
- **Meaning:** Light weights disappear at small sizes and in poor lighting.
- **Factory Ledger:** Audit metadata uses secondary color at Regular weight.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Is any text on this screen set in a weight lighter than Regular?
- **Sources:** D ACCESS-002

### ACCESS-008 — Meet WCAG AA contrast (4.5:1 body, 3:1 large text) in light, dark, and increased-contrast modes, including under overlays and on colored fills
- **Rule:** Avoid color pairs colorblind users can't separate (red/green of similar luminance). Content that scrolls under translucent bars is legible at rest. Text on images or busy backgrounds gets bold weight or a backing shape, not a shadow. Button labels contrast with both the button fill and surrounding content — on already-colorful content, prefer neutral label colors. If legibility fails, fix in order: larger size, more contrast, a more legible typeface.
- **Meaning:** Low contrast is the most common legibility failure and the easiest to measure.
- **Factory Ledger:** Status chip text on colored fills meets AA in both modes; "Today So Far" numbers on tinted tiles; text over scanned-label photos gets a backing shape; on status-colored order cards, button text stays neutral.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Does every text/icon element meet the contrast target in light, dark, and high-contrast modes, including over overlays, images, and colored fills?
- **Sources:** D ACCESS-003 + B ACCESS-001 · Related: LAYOUT-008

### ACCESS-009 — Generous line spacing for reading text; tight spacing only for two-line constrained rows, never three or more
- **Rule:** Multi-line reading text and wide columns get loose spacing. A two-line list row may be tight; a row that needs three lines is redesigned.
- **Meaning:** Compressed multi-line text is slow and error-prone to read.
- **Factory Ledger:** Batch instructions and notes get loose spacing; a product + lot row may be tight.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does any element show 3+ lines with tight spacing, and does long reading text have comfortable spacing?
- **Sources:** D ACCESS-006 · Related: DATA-003

### ACCESS-010 — Every icon-only control and custom icon has an accessible text name (and a desktop tooltip)
- **Rule:** Icon-only buttons and custom glyphs carry an accessible name describing the action or object, so screen readers can announce it and hover can reveal it. Doubles as on-hover training for new staff.
- **Meaning:** An unnamed icon is invisible to assistive technology and ambiguous to new users.
- **Factory Ledger:** Row action icons (edit, void, trace) have aria-labels; a barcode button is named "Scan barcode."
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Does every icon-only control and custom icon have an accessible name that describes what it does?
- **Sources:** D ACCESS-007 · Related: ACTION-005, LAYOUT-009

---

## 11. Performance & Perceived Speed

*No standalone rules yet. Performance-related constraints currently live in: FEEDBACK-001 (progress within ~1 s; double-submit prevention), FEEDBACK-003 (stall handling), FEEDBACK-006 (background work never blocks), FEEDBACK-007 (auto-refresh), LAYOUT-020 (no layout shift), ERROR-002 (state preservation), CHART-002 (progressive detail).*

---

## 12. Notifications & Attention

### NOTIFY-001 — Notifications carry only timely, high-value, glanceable information
- **Rule:** Send a notification only for something the recipient needs to know now and can understand in one glance. Keep it short.
- **Meaning:** Low-value or verbose notifications train users to ignore or disable all of them.
- **Factory Ledger:** Notify for "Truck for SO-1234 arrived," "Lot 26-0907-C failed QC hold," "Expected receipt overdue" — not for routine successful saves.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Does every notification type represent something the recipient would act on or want to know within minutes, and can it be understood without opening the app?
- **Sources:** M1 NOTIFY-001

### NOTIFY-002 — Never notify more than once for the same event
- **Rule:** One event, one notification. Do not re-send because the user hasn't responded; persistence lives in status, badges, or in-app lists. Escalation to a different recipient is a different event.
- **Meaning:** Repeats clutter and cause users to turn off all notifications.
- **Factory Ledger:** An overdue receipt fires once; its persistence lives in the Expected Receipts tab.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Can any single event produce more than one notification to the same user? Is there a dedup mechanism?
- **Sources:** M1 NOTIFY-002

### NOTIFY-003 — Do not use notifications to give instructions; offer inline actions instead
- **Rule:** A notification informs and, where possible, lets the user act directly. Never "go to X and do Y" — instructions in transient UI are lost.
- **Meaning:** Either the action lives in the notification, or the app surfaces it prominently on open.
- **Factory Ledger:** "Pickup for SO-1234 didn't arrive — [Mark No-Show] [Snooze 1h]."
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does any notification tell the user to perform steps they must remember later? Could that step be a button on the notification?
- **Sources:** M1 NOTIFY-003

### NOTIFY-004 — Errors are shown as in-app alerts, never as notifications
- **Rule:** Use the app's alert/inline error mechanism for errors; notifications are informational only. Background-job failures discovered later may notify, but the failure must also be visible in-app on entry (NOTIFY-011).
- **Meaning:** An error arriving as a push is easily missed and confuses the model of "what needs my attention now."
- **Factory Ledger:** A failed `/make` commit or trace-emission failure blocks with an alert in the workflow, not a toast or push after the fact.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Is any error condition communicated only through a notification/toast rather than an in-flow alert?
- **Sources:** M1 NOTIFY-004 · Related: ERROR-004

### NOTIFY-005 — When the user is already in the app, surface new information in context and make the change perceptible, not as an interruption
- **Rule:** If new information arrives while the user is viewing the relevant screen, insert it into the view (new row, updated counter) with a brief highlight so the change is noticed; no interruptive banner. Never shift items the user may be about to tap (LAYOUT-020).
- **Meaning:** Interrupting someone already looking at the data is redundant; silent insertion is insufficient.
- **Factory Ledger:** If Luz is on Receiving when a receipt is logged from the floor, the row appears (buffered if she's mid-selection); if she is on Sales Orders, a badge on the Receiving nav item increments.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** When a foreground user is on the screen where new data lands, does the data appear in place, perceptibly, without a modal or banner or displacing their target?
- **Sources:** M1 NOTIFY-005 (expanded M2) · Related: FEEDBACK-008, FEEDBACK-007

### NOTIFY-006 — Keep sensitive or confidential data out of notifications; provide a generic body for locked/preview-hidden states
- **Rule:** Notification text must be safe for anyone glancing at the device.
- **Meaning:** You can't control where the phone is when the notification arrives.
- **Factory Ledger:** Exclude customer pricing, exclusive-formula details, payroll/scorecard figures, API keys. "Receipt logged" is fine; "Blue Stripes formula BS-07 ratio changed" is not.
- **Platform:** Mobile (primary), Desktop/Web (OS/browser notifications) · **Importance:** Medium · **Type:** Hard rule
- **Audit:** Would any notification body reveal something you wouldn't want a visitor or a driver at the dock to read?
- **Sources:** M1 NOTIFY-006 · Related: OTHER-003

### NOTIFY-007 — Titles are short and specific; bodies are complete, unabbreviated sentences; no redundant app name
- **Rule:** Title = the one thing (order, lot, event). Body = one readable sentence. Let the system truncate; don't pre-truncate.
- **Meaning:** Titles are the most visible slot; wasting them on generic text throws away the glance.
- **Factory Ledger:** Title "SO-1234 ready to ship." Body "All 40 cases of Seven Wells Granola are packed and staged at Door 2." Not "Factory Ledger: Update."
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does each title identify the specific object or event? Is the body a full sentence a floor operator would understand instantly?
- **Sources:** M1 NOTIFY-007 · Related: ACCESS-004

### NOTIFY-008 — Sound and vibration supplement notifications; never rely on them for essential information
- **Rule:** A distinctive sound may aid attention on a noisy floor, but the same information must be visible on screen.
- **Meaning:** Users may have sound off or be unable to hear it.
- **Factory Ledger:** A tone for "truck arrived" on Arturo's phone, and the arrival also shows on the dashboard and Receiving screen.
- **Platform:** Mobile · **Importance:** Low/Contextual · **Type:** Situational idea
- **Audit:** If the device were silent, would the user still receive every important message visibly?
- **Sources:** M1 NOTIFY-008

### NOTIFY-009 — Notification actions are few, common, clearly labelled, safe by default, and ordered most-frequent first
- **Rule:** At most ~4 actions, each a frequent, time-saving task that avoids opening the app, labelled with a short phrase describing the result, with a simple icon. Prefer non-destructive actions; a destructive action is styled distinctly and given enough context. Never include an action that merely opens the app. Most-used action first (some devices trigger the first non-destructive action by a quick gesture).
- **Meaning:** Actions on notifications execute with almost no context and no undo screen.
- **Factory Ledger:** "Expected receipt overdue" → [Received] [Reschedule] [Not Coming]. Never [Void SO] from a notification.
- **Platform:** Mobile (primary), Desktop/Web where supported · **Importance:** High · **Type:** Strong recommendation (destructive handling is a hard rule)
- **Audit:** Does every notification action have a clear outcome in its label, avoid irreversible record changes, and is the most common one first?
- **Sources:** M1 NOTIFY-009 · Related: ACTION-008

### NOTIFY-010 — Badges show only counts of unhandled items, are always accurate, and are never faked
- **Rule:** Use badge counters solely for the number of items awaiting the user; clear the count the moment items are viewed/handled. Don't use badge-like elements for other numbers. Don't imitate a badge with a custom control the user can't turn off.
- **Meaning:** A stale or misused badge destroys trust in the signal.
- **Factory Ledger:** "Receiving (3)" means three unhandled receipts, decrementing the moment a receipt is opened/actioned. Inventory quantities are data, not red ovals.
- **Platform:** Both · **Importance:** Medium · **Type:** Hard rule
- **Audit:** Does every badge/counter represent unhandled items only, and does it reach zero immediately after handling?
- **Sources:** M1 NOTIFY-010

### NOTIFY-011 — Important information is findable on app entry, independent of any transient channel
- **Rule:** Never let a notification, badge, toast, or sound be the only way to learn something important. The home/dashboard surfaces it.
- **Meaning:** Users disable notifications, miss banners, or share devices; the app is the source of truth for "what needs attention."
- **Factory Ledger:** Today So Far / dashboard shows open holds, overdue receipts, missed pickups, and failed syncs even if every push was ignored.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** With all notifications disabled, can a user discover every pending attention item within one screen of opening the app?
- **Sources:** M1 NOTIFY-011 · Related: NAV-001

### NOTIFY-012 — Users control which notification categories they receive and their urgency
- **Rule:** Obtain consent before notifying; provide per-category toggles per user.
- **Meaning:** Different roles need different signals; forcing all notifications on everyone leads to all being turned off.
- **Factory Ledger:** Arturo: receipts/pickups; Luz: receiving and expected receipts; Blubber: exceptions and escalations.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Can each user turn categories on/off individually without losing all notifications?
- **Sources:** M1 NOTIFY-012 · Related: OTHER-002

---

## 13. Direct Manipulation / Drag & Drop

### DRAG-001 — Support drag and drop where users will instinctively try it, and always provide a non-drag alternative
- **Rule:** Drag is impossible for some users and situations (touch, assistive tech, small screens), so every drag operation is also achievable via a button, menu, or keyboard. On mobile the button path is the primary path.
- **Meaning:** Drag is a shortcut, not the only door.
- **Factory Ledger:** Scheduler: drag a job between days, and also "Move to…". Allocations: drag a lot onto an SO line, and also an "Allocate" button.
- **Platform:** Both (drag primarily Desktop/Web) · **Importance:** Critical (alternative) / High (drag support) · **Type:** Hard rule (alternative); strong recommendation (support)
- **Audit:** For every drag operation, is there a discoverable non-drag way to do the same thing?
- **Sources:** A DRAG-001 · Related: DATA-011, TOUCH-002

### DRAG-002 — Move-vs-copy semantics are predictable and choose the outcome least likely to lose data
- **Rule:** Within one container, drag = move; across containers, drag = copy/link — unless a different behavior is clearly expected. A drag never silently deletes or consumes a record.
- **Meaning:** Users must be able to predict whether the source item still exists after the drop.
- **Factory Ledger:** Dragging a job within the production board reorders it. Dragging a lot from inventory onto an SO line allocates it (creates a link; the lot stays in inventory).
- **Platform:** Desktop/Web · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Before dropping, can the user predict whether the source item will still exist where it was?
- **Sources:** A DRAG-002

### DRAG-003 — Every drag operation is undoable; if it can't be, confirm before completing it
- **Rule:** Provide Undo after a drop; where undo is impossible, confirm before an irreversible commit; where neither, provide a reversal path (ERROR-003).
- **Meaning:** Mis-drops are common.
- **Factory Ledger:** Dropping a job on the wrong day → Undo toast. Dropping a lot onto a shipment that commits inventory → confirm before commit, since a committed ship record needs a void cascade.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** After a drop, can the user reverse it in one action, or were they asked to confirm before an irreversible one?
- **Sources:** A DRAG-003 · Related: ERROR-003, ACTION-008

### DRAG-004 — Support multi-item drag when it saves work, with a count badge that updates if the destination accepts only some
- **Rule:** Let users select several items and move them together.
- **Meaning:** One drag instead of five.
- **Factory Ledger:** Select five pallets → drag to a shipment. Select three jobs → drag to next week.
- **Platform:** Desktop/Web · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Can the user move a group in one drag instead of repeating a drag per item?
- **Sources:** A DRAG-004 · Related: TOUCH-005

### DRAG-005 — Continuous, unambiguous feedback throughout a drag
- **Rule:** Show a translucent drag image immediately; highlight a destination only while hovering *and* only if it can accept the drop; show an explicit "not allowed" cue otherwise; highlight one destination at a time; change the pointer to signal the outcome; avoid a drag image that changes distractingly.
- **Meaning:** The user must always know whether the current target will accept the drop and what will happen.
- **Factory Ledger:** Dragging a lot over an SO line for a different item shows ⃠ and no highlight. Over a matching line, the row highlights. Over a fully-allocated line, a "will over-allocate" warning.
- **Platform:** Desktop/Web · **Importance:** High · **Type:** Strong recommendation
- **Audit:** While dragging, can the user always tell whether the current target will accept the drop and what will happen?
- **Sources:** A DRAG-005

### DRAG-006 — On an invalid or failed drop, visibly return the item to its source and say why
- **Rule:** Don't let a failed drop look like success. Snap back or clearly dismiss, and record nothing.
- **Meaning:** A silent failure becomes a phantom transaction in the user's mind.
- **Factory Ledger:** A lot dropped on a closed SO animates back to the inventory list with a brief message.
- **Platform:** Desktop/Web · **Importance:** Critical · **Type:** Hard rule
- **Audit:** If a drop fails, is it impossible for the user to believe it succeeded?
- **Sources:** A DRAG-006

### DRAG-007 — Auto-scroll the destination while dragging near its edge; stop when the drag leaves the container
- **Rule:** Long lists scroll as the user drags toward the boundary.
- **Meaning:** Targets off-screen when the drag started must still be reachable.
- **Factory Ledger:** Dragging a job to a week that's off-screen in the scheduler; a lot down a long SO list.
- **Platform:** Desktop/Web · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Can the user drop onto a target that wasn't visible when the drag started?
- **Sources:** A DRAG-007

### DRAG-008 — When a drop triggers work, show progress and a placeholder at the drop location
- **Rule:** If the drop starts a transfer or action, show it has begun, where the result will land, and how it's progressing (FEEDBACK-001).
- **Meaning:** A drop that takes more than ~0.5 s with no feedback looks like nothing happened.
- **Factory Ledger:** Dropping a CSV of receipts onto the receiving screen shows a placeholder row and progress until the API confirms.
- **Platform:** Desktop/Web · **Importance:** High · **Type:** Strong recommendation
- **Audit:** After a drop that takes more than ~0.5 s, does the user see that something is happening and where the result will appear?
- **Sources:** A DRAG-008

### DRAG-009 — After a drop, the dropped item is selected in the destination and deselected in the source
- **Rule:** Users expect to act on what they just moved.
- **Meaning:** Saves a re-selection step.
- **Factory Ledger:** After dragging a lot to an SO line, the allocation row is selected so the operator can immediately edit qty.
- **Platform:** Desktop/Web · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Immediately after a drop, is the moved item ready to act on without re-selecting it?
- **Sources:** A DRAG-009

### DRAG-010 — Select and start dragging in a single motion
- **Rule:** No click-to-select, then press-and-hold, then drag — unless multi-selecting.
- **Meaning:** Extra steps make drag slower than the button alternative.
- **Factory Ledger:** Grab a job card on the production board and move it in one gesture.
- **Platform:** Desktop/Web · **Importance:** Low/Contextual · **Type:** Strong recommendation
- **Audit:** Can a single item be dragged without a separate selection step first?
- **Sources:** A DRAG-010

### DRAG-011 — Hovering a dragged item over a tab, segment, or button activates that target so the drag can continue into another view
- **Rule:** Where drag-and-drop exists, one drag can cross views instead of drop → navigate → drag again.
- **Meaning:** Removes a full round-trip for cross-view moves.
- **Factory Ledger:** Dragging a job on the Production Board over the "Week" segment or another day tab switches the view mid-drag.
- **Platform:** Desktop/Web · **Importance:** Low/Contextual · **Type:** Situational idea
- **Audit:** Where drag-and-drop exists, can a dragged item reach a target in another view without dropping first?
- **Sources:** B OTHER-002

---

## 14. Charts & Dashboards

### CHART-001 — Use a chart only when it conveys insight text can't; otherwise use a sortable, searchable table
- **Rule:** Charts are prominent and draw attention. Use them for trends, the current state of a changing quantity, or comparisons across categories — not merely to display numbers.
- **Meaning:** A chart that doesn't answer a question is decoration that costs attention.
- **Factory Ledger:** "Coconut utilization over 10-day cadence" → line chart. "Lots received today" → table. Don't chart a list of open SOs.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Does this chart answer a question (trend, state, comparison) that a table wouldn't answer faster?
- **Sources:** A CHART-001

### CHART-002 — Keep charts simple and reveal detail progressively
- **Rule:** Don't pack every series in. Offer levels of detail or subsets the user can choose; interactive charts start with a simple default and offer richer modes.
- **Meaning:** A first-time viewer should read the main point in 3 seconds.
- **Factory Ledger:** Today So Far tile shows one number and a sparkline; tapping opens the full chart with per-item breakdown and date range controls.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Can a first-time viewer read the chart's main point in 3 seconds, with more detail available on demand?
- **Sources:** A CHART-002 · Related: NAV-012

### CHART-003 — Prefer common chart types; if a novel visualization is needed, teach it
- **Rule:** Bars and lines need no explanation. Anything unfamiliar gets a first-run explanation or an always-visible key.
- **Meaning:** An operator who has never seen the chart type must understand it without help.
- **Factory Ledger:** Bars and lines for production, inventory, and utilization. A capacity ring or trace tree gets a key.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Would an operator who has never seen this chart type understand it without help?
- **Sources:** A CHART-003

### CHART-004 — Pair every chart with a plain-language sentence stating the takeaway and what, if anything, needs action
- **Rule:** Title, subtitle, annotations, and a one-line headline summary. Supplements accessibility labels; does not replace them.
- **Meaning:** People grasp the point at a glance and see what's actionable.
- **Factory Ledger:** "Coconut line at 91% — 2 days of slack before RD order." Annotate anomalies ("Line down 8/14").
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Is there a plain-language sentence next to the chart stating what it shows and what needs action?
- **Sources:** A CHART-004

### CHART-005 — Every chart exposes its values to assistive technology or as an adjacent data table
- **Rule:** Provide screen-reader labels for the data and interactive elements, not just a visual.
- **Meaning:** A chart with no accessible data is invisible to some users and un-copyable for everyone.
- **Factory Ledger:** Each dashboard chart has an accessible description or a "view as table" toggle.
- **Platform:** Both · **Importance:** Medium · **Type:** Hard rule
- **Audit:** Can the chart's values be read by assistive technology or accessed as a table?
- **Sources:** A CHART-005 · Related: ACCESS-010

### CHART-006 — Size charts to their purpose: glanceable for tiles; large enough for legible labels, annotations, and scope controls when analysis is the goal
- **Rule:** A small chart previews; a large chart has room to be readable — especially on mobile.
- **Meaning:** A cramped chart is unreadable.
- **Factory Ledger:** Dashboard tiles use compact sparklines; the reports screen gives charts full width.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Are all labels and annotations legible at the chart's actual rendered size on this device?
- **Sources:** A CHART-006

### CHART-007 — Chart style is consistent; the same item, metric, or dataset uses the same color and marks everywhere; vary style only to signal a meaningful difference
- **Rule:** Charts of the same dataset at different zoom levels share type, colors, marks, and annotations.
- **Meaning:** Users transfer what they learn from one chart to the next.
- **Factory Ledger:** Coconut is always the same color everywhere. The tile sparkline and its expanded view use identical styling.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does every chart of the same item, metric, or dataset use the same color and style across the app?
- **Sources:** A CHART-007 · Related: FEEDBACK-012

### CHART-008 — Design charts from multiple perspectives: totals, meaningful subsets, and individual points, with drill-through to the underlying records
- **Rule:** Surface what macro (totals/averages), mid (subsets), and micro (specific values) levels each reveal.
- **Meaning:** From the summary the user should be able to reach the specific records behind any point.
- **Factory Ledger:** Production chart shows weekly total, per-line breakdown on demand, and tap-a-bar to see the lots that made up that day.
- **Platform:** Both · **Importance:** Low/Contextual · **Type:** Situational idea
- **Audit:** From the summary chart, can the user drill to the specific records behind any point?
- **Sources:** A CHART-008

---

## 15. Icons & Imagery

### ICON-001 — Icons are simple, recognizable, familiar metaphors that relate directly to their action or object
- **Rule:** Streamlined shapes, minimal detail; no clever or abstract glyphs.
- **Meaning:** Detail and cleverness reduce recognition speed.
- **Factory Ledger:** Box/pallet for inventory, truck for shipping, barcode for scan. Don't invent an unlabeled icon for "allocate" — pair with a text label until a convention exists.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Would a new operator correctly guess what this icon means without a label?
- **Sources:** D ICON-001

### ICON-002 — One consistent icon set: same size, detail, stroke weight, and perspective; never mix families
- **Rule:** Custom icons are drawn to the library's grid and stroke so visual weight matches.
- **Meaning:** Mixed families look broken and make icons harder to scan.
- **Factory Ledger:** One library for dashboard and mobile.
- **Platform:** Both · **Importance:** Medium · **Type:** Hard rule
- **Audit:** Do all icons on this screen appear to come from the same family?
- **Sources:** D ICON-002

### ICON-003 — Icon weight matches adjacent text and meaningful icons scale with text size
- **Rule:** Regular with regular, bold with bold, unless one is deliberately emphasized.
- **Meaning:** Mismatched weight makes one element look like an error.
- **Factory Ledger:** Status-chip icons match the chip's bold label; list-row glyphs scale when text size increases.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Do icons next to text match its weight and scale with it?
- **Sources:** D ICON-003 · Related: ACCESS-001

### ICON-004 — Conventional glyphs for standard actions; one glyph per meaning; FL-specific concepts get a text label
- **Rule:** Checkmark = done/save/select; X = cancel/close; trash = delete; plus = add; ellipsis = more; pencil = edit; magnifier = search; decreasing lines = filter; box-with-up-arrow = share/export; printer = print; U-turn arrows = undo/redo; folder = move to; paperclip = attach; person-in-circle = account; calendar = calendar; archive box = archive; document-on-document = copy. A glyph is never reused for a different action. **Qualification:** void, allocate, trace, hold have no universal glyph — they get a text label, with any icon as a supplement.
- **Meaning:** Conventions are free training; violating them costs errors.
- **Factory Ledger:** "Void" does not use the X; it gets a labeled destructive treatment. Save = checkmark consistently across the dashboard and any GPT-driven UI.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Does every standard action use its conventional glyph, is any glyph reused with a second meaning, and are non-standard actions labeled with text?
- **Sources:** D ICON-004 · Related: ACTION-005

### ICON-005 — Vector icons, recognizable at the smallest size shown
- **Rule:** Deliver icons as SVG; simplify detail for small variants so they stay distinguishable at ~16 px.
- **Meaning:** Blurry or indistinguishable small icons cause wrong clicks.
- **Factory Ledger:** Table-row action icons at 16 px are still distinguishable (edit vs. void).
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Is every icon a vector, and is it recognizable at its smallest rendered size?
- **Sources:** D ICON-005

### ICON-006 — Text inside icons only when the letter is the concept; inclusive, culturally neutral imagery
- **Rule:** If a letter is used, localize it. Depict people, if at all, in gender-neutral, culturally neutral ways.
- **Meaning:** Letters and culture-specific images fail for some users.
- **Factory Ledger:** Avoid letter-based icons that Spanish-speaking floor staff may not parse — use words.
- **Platform:** Both · **Importance:** Low/Contextual · **Type:** Situational idea
- **Audit:** Does any icon rely on a letter or culturally specific image to be understood?
- **Sources:** D ICON-006

### ICON-007 — Center asymmetric icons optically, not just geometrically
- **Rule:** Small padding adjustments in the asset so icons don't look misaligned in buttons and chips.
- **Meaning:** Small misalignments read as sloppiness.
- **Factory Ledger:** Download/upload arrows in circular buttons.
- **Platform:** Both · **Importance:** Low/Contextual · **Type:** Situational idea
- **Audit:** Do asymmetric icons look centered in their containers?
- **Sources:** D ICON-007

### ICON-008 — Selected state for icons via one consistent treatment, paired with a label or indicator
- **Rule:** Accent tint and/or filled variant, not hand-drawn alternate icons; never color alone (FEEDBACK-011).
- **Meaning:** One selection language across the app.
- **Factory Ledger:** Selected tab-bar item = filled glyph + accent + label.
- **Platform:** Both · **Importance:** Low/Contextual · **Type:** Situational idea
- **Audit:** Is the selected state of icons applied the same way everywhere?
- **Sources:** D ICON-008

### ICON-009 — Scale or crop images to fit; never stretch
- **Rule:** Keep the important content visible.
- **Meaning:** Distorted images are unreadable and look broken.
- **Factory Ledger:** Product photos, scanned labels, COA thumbnails.
- **Platform:** Both · **Importance:** Low/Contextual · **Type:** Situational idea
- **Audit:** Is any image stretched or squashed?
- **Sources:** D ICON-009

### ICON-010 — Label attachment types with a short recognizable term, not a cryptic extension
- **Rule:** "COA," "LABEL," "PDF," short enough to stay legible at list size.
- **Meaning:** Users recognize what a file is by a word.
- **Factory Ledger:** Attachments on a lot or receiving record.
- **Platform:** Both · **Importance:** Low/Contextual · **Type:** Situational idea
- **Audit:** Can a user tell what each attachment is from its label at list size?
- **Sources:** D ICON-010

---

## 16. Other

### OTHER-001 — Prioritize the core workflows and make them excellent before adding features
- **Rule:** Identify what the product is for and which tasks matter most to each role; invest design effort there. Evaluate every addition against that purpose.
- **Meaning:** A tool with a clear focus helps people more than a broad one with mediocre core paths.
- **Factory Ledger:** Receive, Pack, Ship, Coconut log, and Trace are the core; they get the most refinement, fastest paths, and most testing.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Is there a written list of core workflows per role, and do those have the shortest, most polished paths in the app?
- **Sources:** M2 OTHER-001

### OTHER-002 — Explain why before requesting a permission, and what data is used for
- **Rule:** Camera, location, notifications, and similar requests carry a one-line, task-specific reason at the moment of asking.
- **Meaning:** Unexplained requests get denied, and denied permissions break features.
- **Factory Ledger:** "Camera is used only to scan lot and case barcodes." "Notifications tell you when a truck arrives or a receipt is overdue."
- **Platform:** Mobile (primary); Desktop/Web (browser notifications, camera) · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does every permission prompt carry a one-line, task-specific reason?
- **Sources:** M2 OTHER-002 · Related: INPUT-016, NOTIFY-012

### OTHER-003 — Collect and show only what each role needs; anticipate misuse and unintended consequences and design guards against them
- **Rule:** Minimize data collected and displayed per role. Think through how a screen or action could be misused, or cause harm by accident, and add protections (role scoping, confirmations, limits, logging) at design time. Read-only roles have no mutation controls rendered.
- **Meaning:** Data integrity is a trust issue; the design, not just the backend, should make harmful outcomes hard.
- **Factory Ledger:** Floor role sees quantities and lots, not customer pricing or exclusive formulas. An action that could void many records is scoped, confirmed, and logged.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** For each role, is any data shown that the role doesn't need? For each consequential action, what stops accidental or malicious misuse?
- **Sources:** M2 OTHER-003 · Related: NOTIFY-006, NOTIFY-009

### OTHER-004 — Build on what people already know: factory vocabulary and standard, familiar controls
- **Rule:** Draw on real-world concepts (the paper form, the pallet, the dock door) and established software patterns. Don't invent novel interactions when a familiar one exists.
- **Meaning:** Familiar concepts need no training; novel ones cause errors.
- **Factory Ledger:** Screen and field names match what the floor says: "Receive," "Pack," "Ship," "Cases," "Lot." Form field order mirrors the paper form it replaces.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Does every term on screen match the term used on the floor and in the office? Is every control a type users have seen elsewhere?
- **Sources:** M2 OTHER-004 · Related: ACTION-011, DATA-002

### OTHER-005 — Give each platform first-class attention; mobile is not a shrunken desktop and desktop is not a stretched phone
- **Rule:** Design intentionally for each platform's strengths while keeping the shared structure (LAYOUT-003).
- **Meaning:** A responsive layout that merely reflows is not a mobile design; the floor phone and the office monitor serve different jobs.
- **Factory Ledger:** Mobile: scan-first, big targets, one task per screen. Desktop: dense tables, keyboard shortcuts, multi-record views.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Is each key workflow designed for the platform it runs on, or is one platform's layout simply reused?
- **Sources:** M2 OTHER-005

### OTHER-006 — Prototype, test on real floor devices under real conditions, and keep refining; quality is ongoing
- **Rule:** Try designs early and discard what doesn't work. Verify on the actual devices and environments: floor phones/tablets and office desktops; bright, dim, and glare lighting; each orientation; the largest and smallest layouts and text sizes first (they surface most problems); light and dark modes; gloves, noise, dead-zone Wi-Fi; devices with different display color profiles; Spanish labels. Fix failures by increasing size, increasing contrast, or switching to a more legible typeface. Shipping is not the finish line: schedule periodic review against these standards.
- **Meaning:** A simulator at a desk won't reveal what fails on the floor.
- **Factory Ledger:** Check the pack screen on the floor phone near the dock door mid-shift; check the dashboard at half-width beside another window.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule (process)
- **Audit:** Was this screen tested on the real floor device, in floor lighting, at the largest and smallest sizes, in both appearance modes, before release? When was it last reviewed against the standards?
- **Sources:** M2 OTHER-006 + D OTHER-001

### OTHER-007 — The intended feeling is confidence and calm; never let delight become decoration that slows the task
- **Rule:** For an operational ledger the target emotion is certainty, speed, and control; let that shape every moment including error messages. Character is welcome only where it doesn't add time or ambiguity.
- **Meaning:** People remember how the tool made them feel; in a factory that should be "I know exactly what happened and what to do."
- **Factory Ledger:** Confirmation states are clear and quick, not celebratory animations. No playful copy on consequential actions.
- **Platform:** Both · **Importance:** Low/Contextual · **Type:** Situational idea
- **Audit:** Does any animation, message, or flourish add time or ambiguity to a task? Does the tone stay steady in error states?
- **Sources:** M2 OTHER-007

### OTHER-008 — Use native file pickers for uploads and downloads
- **Rule:** System open/save dialogs include search and familiar navigation; don't build a custom one.
- **Meaning:** Familiar, accessible, zero training.
- **Factory Ledger:** Attaching a COA PDF to a receipt; exporting a trace report.
- **Platform:** Both · **Importance:** Low/Contextual · **Type:** Strong recommendation
- **Audit:** Does any upload/download path bypass the native file dialog?
- **Sources:** A OTHER-001

### OTHER-009 — One standard, recognizable help control per screen, in a predictable location, opening context-specific help
- **Rule:** Where help is offered, use a single "?" per screen in a consistent spot (lower corner of a dialog opposite OK/Cancel; lower corner of a settings pane), inside the content area, with no introductory text. It opens the help topic for the current context, falling back to top-level help only when no specific topic exists.
- **Meaning:** One consistent help control is predictable; several make people guess.
- **Factory Ledger:** A "?" on lot-code entry opening the LAT Code Policy section; one on the allocations screen opening allocation rules.
- **Platform:** Desktop/Web · **Importance:** Low/Contextual · **Type:** Situational idea
- **Audit:** Is there at most one help control per screen, in a consistent location, opening context-specific help?
- **Sources:** B OTHER-001

### OTHER-010 — Semantic color tokens with light, dark, and high-contrast variants, shared by all surfaces; never hard-coded, never repurposed
- **Rule:** Define colors by purpose (surface-1/2/3, text-1..4, separator, status-blocked, status-ready, action-primary, …) with all three variants, even if only one mode ships today. Components reference tokens, never raw values. A token is never used for a purpose other than its name. One token file consumed by dashboard and mobile; the status-color table (FEEDBACK-012) lives alongside it.
- **Meaning:** Tokens are what make consistent color semantics, contrast, and future dark mode achievable without rework.
- **Factory Ledger:** One token file for the Netlify dashboard and the mobile surface.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Does every color on this screen resolve to a named semantic token, used for its intended purpose, with all three variants defined?
- **Sources:** D OTHER-002 · Related: FEEDBACK-012, ACCESS-008, LAYOUT-017

### OTHER-011 — Confirm status colors carry the intended meaning for the actual user population
- **Rule:** Red/green "danger/positive" conventions vary across cultures; revisit if any surface reaches other markets.
- **Meaning:** Color meaning is learned, not universal.
- **Factory Ledger:** CNS staff are US-based English/Spanish speakers for whom red = stop and green = go are standard.
- **Platform:** Both · **Importance:** Low/Contextual · **Type:** Situational idea
- **Audit:** Do the chosen status colors mean the same thing to every user group?
- **Sources:** D OTHER-003

---

## 17. Status & Data Display

How a record's condition and its numbers are rendered. Sections 6 and 9 govern the mechanics — what a
status colour may be (FEEDBACK-011–014), how a row is built (DATA-003, DATA-007). This section governs
the semantics: which concepts are allowed to share a space, when a value should not be shown at all, and
how a number is written. It exists because the Sales Orders surface merged four independent concepts into
one column and printed raw database decimals beside them; every rule below is the general form of a
specific defect found there.

Eight of the fourteen have a clause a rendered page can settle without judgement and are checked by
`tests/visual/run-visual-audit.mjs`:

| Rule | Mechanically checked | Checked by |
|---|:--:|---|
| STATUS-001 Orthogonal dimensions | manual | — |
| STATUS-002 Silence means normal | **yes** | rendered chip text against a nominal-phrase list |
| STATUS-003 Context is not repeated | manual | — |
| STATUS-004 One alarm per row | **yes** | danger/warning-toned elements counted per row |
| STATUS-005 Chip explanations | **yes** (hook only) | `data-explain` hook, focusability; content is manual |
| STATUS-006 One number formatter | **yes** | rendered numeric text against the format |
| STATUS-007 No orphan placeholders | **yes** | dash-only elements outside table cells |
| STATUS-008 Single-line list rows | **yes** | measured row height at ≥ 1200 px |
| STATUS-009 Chip subtext | manual | — |
| STATUS-010 No developer vocabulary | **yes** | denylist scan over rendered text |
| STATUS-011 Disclaimers appear once | **yes** | repeated sentences within one capture |
| STATUS-012 Three health levels | manual | — |
| STATUS-013 Detail-page case summary | manual | — |
| STATUS-014 Tabs with counts | manual | — |

### STATUS-001 — State, Fulfillment, Readiness and Health are four independent dimensions and always render separately
- **Rule:** A record carries several conditions at once, and they vary independently: **State** (open / closed / cancelled), **Fulfillment** (unshipped / partial / shipped), **Readiness** (Factory Ready or not), and **Health** (critical / warning / quiet). Each gets its own column, chip, or field. Never combine two into one column, one chip, or one word, and never let one dimension's value suppress another's.
- **Meaning:** Combined dimensions are lossy. A single "Status" column that reads "Partial" cannot say whether the order is still open, and one that reads "Cancelled" hides that three pallets already shipped. The operator then has to open the record to learn what the list was supposed to tell them.
- **Factory Ledger:** A sales order that is open, half shipped, marked Factory Ready and overdue is four values, not one: State `Open` · Fulfillment `Partial` · Readiness `Factory Ready` · Health `overdue`. The order list gives each its own column; the detail header gives each its own field.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Take any record with a non-trivial combination. Can State, Fulfillment, Readiness and Health each be read off the list independently, or does one column carry more than one of them?
- **Sources:** FL-SO-2026-09 · Related: DATA-007, FEEDBACK-011, STATUS-012

### STATUS-002 — Silence means normal: a nominal value renders as nothing, never as a positive badge
- **Rule:** Only the exceptional gets a mark. A value that is what it should be renders as nothing at all, or as an em-dash where a column needs a placeholder. Never render a coloured "OK", "Good", "Confirmed", "Normal", "Fine", "Healthy", "On Track", "No Issues", "All Clear", or "Checks Passed" badge for the ordinary case.
- **Meaning:** A badge on every row is a badge on no row. Marking the normal case spends the operator's attention on the 95% of rows that need none, and leaves nothing left over to make the 5% that do stand out.
- **Factory Ledger:** An order with no blockers shows an empty Health cell, not a green "Checks passed" pill. A lot within date shows nothing in the age column; only an aged lot is marked.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** On a list where most rows are normal, how many rows carry a status badge? If it is most of them, what does the badge distinguish?
- **Sources:** FL-SO-2026-09 · Related: FEEDBACK-014, LAYOUT-018, STATUS-004

### STATUS-003 — A value implied by the active tab or filter is not repeated on every row
- **Rule:** When a view is already scoped to a value, that value does not appear again per row. It belongs in the view's heading or the tab label, once.
- **Meaning:** A column whose every cell reads the same thing carries no information and costs width the distinguishing columns need.
- **Factory Ledger:** Inside the **Open** tab, no row says "Open". Inside a supplier-filtered receipts list, the supplier is in the heading, not in every row. The moment the filter widens to All, the column earns its place again and returns.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** In each tab and under each default filter, is there a column whose value is the same on every row? Is that value already stated by the tab or filter?
- **Sources:** FL-SO-2026-09 · Related: NAV-003, DATA-003, STATUS-014

### STATUS-004 — At most one danger- or warning-toned element per list row
- **Rule:** A row gets one alarm. Where several conditions on one record would each be coloured, show the most severe and let the explanation (STATUS-005) carry the rest.
- **Meaning:** Two red marks on one row do not read as twice as urgent; they read as decoration, and they break the scan that finds the one row that matters.
- **Factory Ledger:** An overdue order that is also short on inventory shows one critical mark — the more severe — and names the second condition inside the explanation, not as a second red chip beside the first.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Count the danger- and warning-toned elements in the worst row of each list. Is the count above one?
- **Sources:** FL-SO-2026-09 · Related: FEEDBACK-013, LAYOUT-018, STATUS-012

### STATUS-005 — Every status, health and blocker chip opens an explanation, through one defined markup hook
- **Rule:** Any chip that reports a condition explains itself on demand: a generic definition of the condition, plus what it means for **this** record — the quantities, lines and dates behind it. The explanation opens on hover on pointer devices, on tap on touch devices, and on keyboard focus. One markup hook carries this everywhere: the chip element has **`data-explain`** whose value is the `id` of the element holding the explanation, is **`aria-describedby`**-linked to that same element, and is focusable (a `button`, or `tabindex="0"` where it must stay a `span`). A `title` attribute is not the hook: it is invisible on touch, unstyleable, and slow.
- **Meaning:** A chip that says "Blocked" without saying by what, and by how much, forces the operator to open the record — which is the trip the list was meant to save. One hook means the popover behaviour is written once and every chip inherits it.
- **Factory Ledger:** "Short 240 lb" opens: what a shortage is, then this order's two short lines with the quantity missing on each and the date the shortfall was measured. Applies to Health chips, blocker chips, Fulfillment chips and the Factory Ready marker alike.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Does every condition chip carry `data-explain`, an `aria-describedby` link and a focus stop? Does the explanation give both the definition and this record's specifics — quantities, lines, dates?
- **Sources:** FL-SO-2026-09 · Related: ACCESS-010, NAV-010, INPUT-009, ACCESS-003

### STATUS-006 — One number formatter for the whole product
- **Rule:** Every rendered number goes through one formatter. Thousands separators always. Pounds to whole numbers. Pallets to one decimal. Numerals are tabular wherever numbers stack or align. No raw stored value ever reaches the screen: `13500.0000` is a database representation, not a quantity.
- **Meaning:** Trailing zeros and unseparated digits make an operator re-read the number, and re-reading a quantity is how the wrong quantity gets shipped. Consistency also means an unusual-looking number is genuinely unusual, not just differently formatted.
- **Factory Ledger:** `13,500 lb` — not `13500.0000`, not `13500 lb`, not `13,500.00 lb`. `4.5 pallets` — not `4.50` and not `5`. Order totals, line quantities, on-hand, shortages and the allocation matrix all use the same function.
- **Platform:** Both · **Importance:** Critical · **Type:** Hard rule
- **Audit:** Search rendered text for a number with three or more decimal places, and for a value of 1,000 or more without a separator. Do the same quantity and unit render identically on the list, the detail page and the print view?
- **Sources:** FL-SO-2026-09 · Related: ACCESS-005, DATA-005, DATA-012

### STATUS-007 — No orphan placeholders: an empty secondary value is omitted, not dashed
- **Rule:** Where a secondary line, subtitle or caption has no value, the element is not rendered. A dangling "—" on a second line is not an empty state; it is a rendering artefact. Column placeholders inside a table cell are the one exception — a dash there holds the column's alignment and is read as part of the grid.
- **Meaning:** An empty stacked line still costs vertical space, still draws the eye, and tells the operator nothing that omitting it would not.
- **Factory Ledger:** An order row with no customer PO shows the order number alone, not the order number above a dash. A receipt with no notes shows no notes line.
- **Platform:** Both · **Importance:** Medium · **Type:** Hard rule
- **Audit:** In a row whose optional values are all empty, does any line render as a bare dash outside a table cell?
- **Sources:** FL-SO-2026-09 · Related: DATA-003, STATUS-008

### STATUS-008 — List rows are single-line: at most 56 px tall at desktop width
- **Rule:** A row in a list or table occupies one line and is at most 56 px tall at desktop width (≥ 1200 CSS px). Secondary detail belongs in the chip explanation (STATUS-005) or on the detail page. A detail row the user has explicitly expanded is not a list row and is not bound by this height.
- **Meaning:** Scanning is vertical. Every extra line per row is one fewer record visible, and a two-line row halves how much of the list the operator can hold in view while comparing.
- **Factory Ledger:** The Sales Orders row is one line: order number · customer · quantity · ship date · State · Fulfillment · Readiness · Health. SKU, PO number and notes live behind the expander and on the detail page.
- **Platform:** Desktop/Web (Mobile: stacked rows are governed by DATA-003's two-line limit) · **Importance:** High · **Type:** Strong recommendation
- **Audit:** At 1440 px, measure the tallest unexpanded row in each list. Is any above 56 px, and what is on the second line?
- **Sources:** FL-SO-2026-09 · Related: DATA-003, ACCESS-009, STATUS-007

### STATUS-009 — A chip's subtext never restates its title
- **Rule:** Where a chip carries both a label and a detail line, the detail adds facts the label does not already give — quantities, lines, dates. It never paraphrases the label.
- **Meaning:** Restated text is read twice and informs once, and it makes the chip wide enough to push the row past STATUS-008.
- **Factory Ledger:** "Inventory Short" · "240 lb across 2 lines" — not "Inventory Short" · "inventory is short".
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Read each chip's label and its subtext in turn. Does the subtext state a fact the label did not?
- **Sources:** FL-SO-2026-09 · Related: ACCESS-004, STATUS-005

### STATUS-010 — No developer vocabulary in user-facing copy
- **Rule:** Screen text, empty states, error messages, tooltips and explanations use factory and office vocabulary. Implementation terms never appear: "API", "endpoint", "returns", "null", "undefined", "NaN", "TTL", "cache", "payload", "JSON", "timeout", "500", "stack trace", and internal field or table names (`sales_order_id`, `qty_lb`, `order_lines`).
- **Meaning:** A message written in implementation terms cannot be acted on by the person reading it, and it tells them the tool is not finished.
- **Factory Ledger:** "Quantities are as of 6:15 AM" — not "cache TTL 900s". "This order has no lines yet" — not "lines returned null". "Couldn't reach the ledger — retry" — not "API 500 on /orders".
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** Scan all rendered text, including error and empty states, for implementation terms and for `snake_case` identifiers.
- **Sources:** FL-SO-2026-09 · Related: OTHER-004, ACCESS-004, ERROR-010

### STATUS-011 — A disclaimer appears once per screen, at first relevance
- **Rule:** A caveat, as-of note or limitation is stated once on a screen, at the first place it applies. The same sentence never repeats within one view — not per row, not per card, not per section.
- **Meaning:** A sentence repeated down a screen stops being read, including the one time it mattered. It also costs the space the data needs.
- **Factory Ledger:** "Bag count unavailable" belongs once above the tile, not on each of its rows. An as-of timestamp belongs in the section header, not in every cell.
- **Platform:** Both · **Importance:** Medium · **Type:** Strong recommendation
- **Audit:** Does any sentence of forty characters or more render more than once in a single view?
- **Sources:** FL-SO-2026-09 · Related: NOTIFY-002, ACCESS-004, LAYOUT-001

### STATUS-012 — Health has exactly three visual levels and no fourth tone
- **Rule:** Health renders in three levels only: **critical** (the danger token), **warning** (the warning token), and **quiet** (no colour at all). No other tone, tint, weight or icon is used for health — no informational blue, no positive green, no fourth severity.
- **Meaning:** Three levels can be learned at a glance and ranked without a legend. A fourth makes the operator ask which of two colours is worse.
- **Factory Ledger:** Overdue and short-inventory are critical. Due today and partially allocated are warnings. Everything else is quiet — no chip (STATUS-002). Positive green is not a health tone; Factory Ready is Readiness, and it renders as a readiness marker.
- **Platform:** Both · **Importance:** High · **Type:** Hard rule
- **Audit:** List every tone health takes on any screen. Are there exactly three, and does the third have no colour?
- **Sources:** FL-SO-2026-09 · Related: FEEDBACK-012, OTHER-010, STATUS-001

### STATUS-013 — A detail page opens with a one-line case summary
- **Rule:** The first line of a detail page states the case: identifier · party · quantity · key date with relative overdue · Fulfillment · Readiness · Health. It answers "what am I looking at and does it need me" before the reader scrolls.
- **Meaning:** Detail pages are opened to make one decision. The summary makes the common decision without reading the rest.
- **Factory Ledger:** `SO-1042 · Whole Foods NE · 13,500 lb · ships Sep 12 (2 days late) · Partial · Factory Ready · Short 240 lb`.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Does the top line of each detail page carry all seven elements, and is it readable without scrolling at the narrowest supported width?
- **Sources:** FL-SO-2026-09 · Related: NAV-003, LAYOUT-004, STATUS-001

### STATUS-014 — Tabs with counts are the primary filter on every list view; dropdowns are secondary refinements
- **Rule:** The division an operator uses most often is a row of tabs, each showing its count. Dropdowns and checkboxes refine within the selected tab; they never carry the primary division. A tab's count is live and matches the rows the tab shows.
- **Meaning:** A tab shows the shape of the work before it is clicked — how many are open, how many are stuck. A dropdown hides both the options and their sizes behind a click.
- **Factory Ledger:** Sales Orders: `Open 24 · Needs Review 3 · Shipped 61 · All 88`, with customer, date range and Factory Ready as dropdown refinements inside the chosen tab.
- **Platform:** Both · **Importance:** High · **Type:** Strong recommendation
- **Audit:** Is the most-used division of each list a visible tab row with counts, or is it hidden in a dropdown? Does each count match its tab's row count?
- **Sources:** FL-SO-2026-09 · Related: NAV-004, NAV-007, NAV-012, SEARCH-005, STATUS-003

---

## Appendix A — Rule count by category

| Category | Prefix | Rules |
|---|---|---|
| Navigation & IA | NAV | 12 |
| Screen Layout & Visual Hierarchy | LAYOUT | 21 |
| Actions & Buttons | ACTION | 12 |
| Forms & Data Entry | INPUT | 22 |
| Mobile & Touch | TOUCH | 5 |
| Feedback, Status & Progress | FEEDBACK | 14 |
| Errors, Confirmation & Recovery | ERROR | 10 |
| Search & Discovery | SEARCH | 8 |
| Lists, Tables & Dense Data | DATA | 12 |
| Accessibility & Readability | ACCESS | 10 |
| Performance & Perceived Speed | PERF | 0 (cross-refs only) |
| Notifications & Attention | NOTIFY | 12 |
| Direct Manipulation / Drag & Drop | DRAG | 11 |
| Charts & Dashboards | CHART | 8 |
| Icons & Imagery | ICON | 10 |
| Other | OTHER | 11 |
| Status & Data Display | STATUS | 14 |
| **Total** | | **192** |

Source rules consolidated: 216 (M1 35 · M2 24 · A 36 · B 37 · C 36 · D 48) → 178 master rules. Version 1.1 adds 14 STATUS rules from the Sales Orders field review (FL-SO-2026-09), for 192.

---

## Appendix B — Crosswalk (source-thread ID → master ID)

**M1 (this thread, batch 1):** all IDs kept as-is — NOTIFY-001–012, FEEDBACK-001–007, ERROR-001, INPUT-001–015.
**M2 (this thread, batch 2):** all IDs kept as-is — NAV-001–003, LAYOUT-001–004, TOUCH-001–002, INPUT-016, FEEDBACK-008, ERROR-002–003, ACCESS-001–004, OTHER-001–007.

**A — Searching / Drag & Drop / Charting / Entering data**
SEARCH-001→SEARCH-001 · SEARCH-002→SEARCH-002 · SEARCH-003→SEARCH-003 · SEARCH-004→SEARCH-004 · SEARCH-005→SEARCH-005 · SEARCH-006→SEARCH-006 · SEARCH-007→SEARCH-007 · SEARCH-008→SEARCH-008
INPUT-001→INPUT-017 · INPUT-002→INPUT-002 (label clause) + INPUT-018 (defaults) · INPUT-003→INPUT-011 (merged) · INPUT-004→ACCESS-003 (merged) · INPUT-005→INPUT-007 + INPUT-008 (merged) · INPUT-006→INPUT-019 · INPUT-007→INPUT-020 · INPUT-008→INPUT-009 (merged; identifiers under DATA-004) · INPUT-009→INPUT-003 (merged)
DRAG-001–010→DRAG-001–010 (unchanged)
CHART-001–008→CHART-001–008 (unchanged)
OTHER-001→OTHER-008

**B — Segmented controls / Digit entry / Action sheets / Buttons**
NAV-001→NAV-004 · NAV-002→NAV-004 (desktop clause) · NAV-003→SEARCH-005 (panel clause)
LAYOUT-001→LAYOUT-009
ACTION-001→ACTION-001 · ACTION-002→TOUCH-003 · ACTION-003→ACTION-002 · ACTION-004→ACTION-003 · ACTION-005→ACTION-004 · ACTION-006→ACTION-005 · ACTION-007→ACTION-006 · ACTION-008→ACTION-007 · ACTION-009→ACTION-008 · ACTION-010→ACTION-009 · ACTION-011→LAYOUT-021 · ACTION-012→ACTION-010 · ACTION-013→ACTION-011 · ACTION-014→NAV-007 · ACTION-015→NAV-008 · ACTION-016→ACTION-012 · ACTION-017→NAV-006
INPUT-001→INPUT-008 + INPUT-010 · INPUT-002→INPUT-003 · INPUT-003→INPUT-002
TOUCH-001→TOUCH-004 · TOUCH-002→TOUCH-001
STATUS-001→FEEDBACK-001 *(source-thread B's `STATUS-001`; unrelated to the master `STATUS-*` prefix introduced in §17)*
ERROR-001→ERROR-004 · ERROR-002→ERROR-005 · ERROR-003→ERROR-006 · ERROR-004→ERROR-007 · ERROR-005→ACTION-008 · ERROR-006→ERROR-008 · ERROR-007→ERROR-009
ACCESS-001→ACCESS-008
OTHER-001→OTHER-009 · OTHER-002→DRAG-011

**C — Labels / Collections / Tab views / Lists and tables**
NAV-001→NAV-004 · NAV-002→NAV-005 · NAV-003→NAV-006 · NAV-004→NAV-007 · NAV-005→NAV-008 · NAV-006→NAV-009 · NAV-007→NAV-010 · NAV-008→NAV-011 · NAV-009→NAV-002 (steps clause)
LAYOUT-001→LAYOUT-005 · LAYOUT-002→NAV-003 (heading) + INPUT-002 (labels) · LAYOUT-003→LAYOUT-020 · LAYOUT-004→LAYOUT-021
ACTION-001→ACTION-005
INPUT-001→INPUT-021
TOUCH-001→TOUCH-003 · TOUCH-002→TOUCH-002 · TOUCH-003→LAYOUT-012 · TOUCH-004→TOUCH-005
FEEDBACK-001→FEEDBACK-009 · FEEDBACK-002→FEEDBACK-010
ERROR-001→ERROR-010
DATA-001→DATA-001 · DATA-002→DATA-002 · DATA-003→DATA-003 · DATA-004→DATA-004 · DATA-005→DATA-005 · DATA-006→DATA-006 · DATA-007→DATA-007 · DATA-008→DATA-008 · DATA-009→DATA-009 · DATA-010→DATA-010 · DATA-011→NAV-012 · DATA-012→DATA-011 · DATA-013→DATA-012
ACCESS-001→ACCESS-001 (text-size) + ACCESS-005 (fonts)

**D — Typography / Layout / Color / Icons**
NAV-001→NAV-012 · NAV-002→LAYOUT-003
LAYOUT-001→LAYOUT-007 · LAYOUT-002→LAYOUT-004 · LAYOUT-003→LAYOUT-006 · LAYOUT-004→LAYOUT-010 · LAYOUT-005→LAYOUT-008 · LAYOUT-006→LAYOUT-011 · LAYOUT-007→LAYOUT-012 · LAYOUT-008→LAYOUT-003 · LAYOUT-009→LAYOUT-013 · LAYOUT-010→LAYOUT-015 · LAYOUT-011→LAYOUT-014 (+ ERROR-002 rotation clause) · LAYOUT-012→LAYOUT-016 · LAYOUT-013→LAYOUT-017 · LAYOUT-014→LAYOUT-005 · LAYOUT-015→LAYOUT-018 · LAYOUT-016→LAYOUT-019
ACTION-001→ACTION-003 · ACTION-002→TOUCH-004
INPUT-001→INPUT-022 · INPUT-002→ACTION-011
TOUCH-001→TOUCH-004
FEEDBACK-001→FEEDBACK-011 · FEEDBACK-002→FEEDBACK-012 · FEEDBACK-003→FEEDBACK-013 · FEEDBACK-004→FEEDBACK-014
DATA-001→DATA-004 · DATA-002→DATA-006 (+ FEEDBACK-009 selection clause)
ACCESS-001→ACCESS-006 · ACCESS-002→ACCESS-007 · ACCESS-003→ACCESS-008 · ACCESS-004→ACCESS-001 · ACCESS-005→ACCESS-005 · ACCESS-006→ACCESS-009 · ACCESS-007→ACCESS-010
ICON-001–010→ICON-001–010 (unchanged)
OTHER-001→OTHER-006 · OTHER-002→OTHER-010 · OTHER-003→OTHER-011

---

## Appendix C — Consolidation changelog (this merge)

**New rules added (from parallel threads, master IDs):**
NAV-004–012 · LAYOUT-005–021 · ACTION-001–012 · INPUT-017–022 · TOUCH-003–005 · FEEDBACK-009–014 · ERROR-004–010 · SEARCH-001–008 · DATA-001–012 · ACCESS-005–010 · DRAG-001–011 · CHART-001–008 · ICON-001–010 · OTHER-008–011

**Existing (M1/M2) rules expanded or modified by the merge:**
- NAV-002 — added "fixed sequence shown as steps, not tabs" (C NAV-009)
- NAV-003 — added view-heading requirement (C LAYOUT-002)
- LAYOUT-003 — expanded to size classes, orientation, text size, label length, identical destinations across platforms (D LAYOUT-008, D NAV-002)
- LAYOUT-004 — added "essential info gets room" (D LAYOUT-002); primary-action prominence moved to ACTION-003
- INPUT-002 — added unit in label, format-example placeholder, entry-screen prompt naming record (A INPUT-002, B INPUT-003, C LAYOUT-002); raised to Hard rule
- INPUT-003 — added "never mask operational values; never prefill a credential" (A INPUT-009, B INPUT-002)
- INPUT-007 — validation timing tightened: inline as soon as inputs exist; never silently accepted (A INPUT-005)
- INPUT-008 — pasted text also constrained; unit adjacent (A INPUT-005, B INPUT-001)
- INPUT-009 — one-step full-value access on mobile (A INPUT-008)
- INPUT-010 — focused large-keypad step option (B INPUT-001)
- INPUT-011 — "on desktop too"; chips (A INPUT-003)
- TOUCH-001 — top bar restricted to navigation/secondary (B TOUCH-002)
- TOUCH-002 — long-press; custom gestures supplementary (C TOUCH-002)
- FEEDBACK-001 — in-button progress, label change, disable-until-response (B STATUS-001); audit test expanded
- FEEDBACK-007 — must not shift content under the user; buffered updates (C LAYOUT-003)
- ERROR-002 — rotation added to preserved-state triggers (D LAYOUT-011)
- ACCESS-001 — full text-size/bold/zoom behavior and verification (D ACCESS-004, C ACCESS-001)
- ACCESS-003 — scan/paste/drag into every field (A INPUT-004)
- OTHER-006 — concrete real-device test matrix (D OTHER-001)

**Merged / no longer standalone (source IDs absorbed):**
A INPUT-003, A INPUT-004, A INPUT-005, A INPUT-008, A INPUT-009 · B NAV-002, B NAV-003, B ACTION-002, B ACTION-011, B ACTION-014, B ACTION-015, B ACTION-017, B ERROR-005, B STATUS-001, B TOUCH-002, B ACCESS-001, B INPUT-001, B INPUT-002, B INPUT-003 · C NAV-001, C NAV-009, C LAYOUT-002, C LAYOUT-004, C ACTION-001, C TOUCH-001, C TOUCH-002, C TOUCH-003, C DATA-011, C ACCESS-001 · D NAV-002, D LAYOUT-002, D LAYOUT-008, D LAYOUT-014, D ACTION-001, D ACTION-002, D INPUT-002, D TOUCH-001, D DATA-002, D ACCESS-004, D OTHER-001

**Conflicts reconciled:**
- "Label every control" (C) vs. "no captions on self-explanatory controls" (B): inputs always get labels (INPUT-002); other controls get captions only when not self-explanatory (LAYOUT-009); icon-only controls always get a tooltip/accessible name (ACCESS-010).
- "Validate on submit for cross-field rules" (M1) vs. "flag immediately, not on submit" (A): resolved in INPUT-007 as "as early as the check can be made reliably; inline as soon as both values exist; never later than commit."
- Primary-action audit test appeared in both LAYOUT-004 and ACTION-003: kept in ACTION-003; LAYOUT-004 now audits key-data hierarchy.
- Alternating-row shading (C DATA-006) and table style (D DATA-002): one table-style rule, DATA-006; selection persistence moved to FEEDBACK-009.

**Open items for the standards (flagged, not resolved):**
- ACCESS-006: final FL minimum/default text sizes to be fixed.
- FEEDBACK-012 / OTHER-010: the status-color table and token file are referenced but not yet authored.
- FEEDBACK-003: define the stall timeout (N seconds) per operation class.
- PERF: no standalone rules yet; expect HIG pages on loading, launching, and offline handling to populate this category.

---

## Appendix D — Version 1.1 changelog (Status & Data Display)

**Date:** 2026-09-10 · **Source:** FL-SO-2026-09, the Sales Orders field review.

**New category:** 17 — Status & Data Display, prefix `STATUS`. Every rule in it is the general form of a
defect found on the Sales Orders list and detail surfaces: four independent conditions merged into one
"Status" column, positive badges on ordinary rows, raw stored decimals, stacked dash placeholders, and
`title`-only chip tooltips that touch devices never show.

**New rules:** STATUS-001–014.

**No existing rule was renumbered, reworded or removed.** Where a STATUS rule sharpens an existing one,
the relationship is recorded in the `Related:` line of both, not by editing the older rule:
- STATUS-002 / STATUS-004 sharpen FEEDBACK-013 and FEEDBACK-014 for list rows.
- STATUS-006 gives DATA-005 and ACCESS-005 a single concrete number format.
- STATUS-008 gives DATA-003 a measured height at desktop width.
- STATUS-012 fixes the health palette that FEEDBACK-012 and OTHER-010 leave open — it does **not** close
  the open item above, which is the full status-colour table for every dimension, not just health.

**Prefix note:** source thread B used a `STATUS-` prefix of its own, mapped in Appendix B to `FEEDBACK-001`.
It has no relation to this category; no master rule has ever carried a `STATUS-` ID before version 1.1.

**Mechanical coverage:** STATUS-002, -004, -005 (hook only), -006, -007, -008, -010 and -011 are checked by
`tests/visual/run-visual-audit.mjs` and appear in `docs/design/audit/06-browser-check.md`. The other six
are manual review. STATUS-005's hook (`data-explain` + `aria-describedby` + a focus stop) does not exist in
the product yet: its check is written against the markup the redesign will introduce and fails everywhere
until then, by design.
