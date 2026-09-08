# 07 — Systemic Clusters (what one change actually buys)

**Date:** 2026-09-08 · **Source:** [06-browser-check.md](06-browser-check.md) · **Master list:** [IMPROVEMENTS-MASTER.md](IMPROVEMENTS-MASTER.md)
**Status:** Analysis only. **No application file was changed by this work.**

[06](06-browser-check.md) reports 659 rule failures across 384 captures but lists them one element at a time,
capped at twelve rows per capture. This file collapses them into **root causes**: for each cluster, the CSS
selector responsible, how many failures it accounts for, which screens, and the single token or rule change
that would clear it. It is written to answer one question — *which clusters go into the next PR* — so every
count below is stated in the unit the next PR will be judged in.

---

## How to read the counts

**A "failure" in 06 is a capture cell**, not an element: a screen × variant cell fails TOUCH-003 if *any*
interactive element in it is under 44 pt. So a cluster clears a cell only when it is the cell's **last**
remaining offender. Three numbers are given for every cluster, and they mean different things:

| Column | Meaning | Use it for |
|---|---|---|
| **Cells it appears in** | capture cells where this cluster contributes ≥ 1 failing element | how wide the defect is |
| **Cells it clears alone** | cells where this cluster is the *only* offender, so fixing it flips the cell to PASS by itself | what a single-cluster PR moves on the scoreboard |
| **Failing elements** | individual measurements | how much of the product the change touches |

**Cells it clears alone** is the honest per-change yield, and it is much smaller than the footprint —
because the dashboard's control classes fail *together*. The dashboard screens each carry a `.btn-sm`
**and** an input **and** a checkbox, so no one of them alone empties the screen. The bundle table at the
end is where that entanglement is priced.

## Where the data came from, and one caveat

The 06 run stores a per-capture `worst` list truncated to the twelve smallest / worst offenders
(`tests/visual/lib/checks.mjs:112` and `:256`), which drops 2,910 of the 4,968 TOUCH-003 measurements and
1,483 of the 3,418 ACCESS-008 measurements — enough to make cluster arithmetic wrong. The clustering below
therefore runs the same `checks.mjs` measurement functions with that cap lifted, from a scratch script
outside the repo. Nothing in `tests/visual/` or `dashboard/` was modified.

That re-run reproduces **762 of 768** TOUCH-003 / ACCESS-008 capture verdicts exactly. The six that differ
are `S-87` and `S-88` (scheduler, ACCESS-008) and `S-91` (print view, TOUCH-003), whose `checked` counts move
between runs — those three screens render a different amount of content depending on setup timing. Totals
here are consequently **315** failing TOUCH-003 cells and **290** failing ACCESS-008 cells, against 06's
**314** and **294**. The ±1–4 drift changes no cluster's rank, and every LAYOUT-003 and LAYOUT-011 number in
this file is read straight from the committed `screenshots/results.json` with no re-run at all.

---

## The one-line finding for each rule

* **TOUCH-003 — this is a vertical-rhythm problem, not a hit-target problem.** Of 4,967 failing
  measurements, **3,351 fail on height alone and only 36 on width alone**; the remaining 1,580 fail on both,
  and every one of those is an icon glyph or a checkbox. Nothing in the product is too narrow. Controls are
  24–28 pt tall because every control class was given `padding: 4–6px` and no `min-height`, and there is no
  `min-height` anywhere in `dashboard.css` for a control to inherit (**SYS-4**). One `--hit-min` token
  referenced from thirteen rules is the whole fix.
* **ACCESS-008 — most of it is one hex value being 0.6 : 1 short.** 1,401 of 3,418 failures sit within 0.7 of
  their AA target; the single largest cluster (`--text-muted` in light, 88 cells) never falls below
  **3.86 : 1** against a 4.5 target. The second largest is not a colour at all — it is `opacity` used as a
  de-emphasis device on rows that still contain readable text.
* **LAYOUT-003 — 24 of the 28 overflows are one missing declaration, copied three times.**
  `dashboard.css:2223-2225` already carries `@media (max-width: 768px) { .app-header { flex-wrap: wrap } }`,
  which is why no `index.html` screen overflows. `sankey.html`, `process-flow.html` and `traceability.html`
  each define the same `.header` flex row and **none of them has that media block**.
* **LAYOUT-011 — all 12 are one number.** At 200 % zoom the sticky stack measures **335 px of a 450 px
  viewport (74 %)**, leaving 115 px of content. Nothing about the individual screens is at fault.

---

## Ranked — all 27 clusters, by failures resolved per change

Each row is **one** token or rule change. Ranked by footprint; read the *clears alone* column for what a
single-cluster PR actually flips.

| Rank | Cluster | Rule | Cells it appears in | Cells it clears **alone** | Failing elements | Screens | Root cause |
|---:|---|---|---:|---:|---:|---:|---|
| 1 | **T1** | TOUCH-003 | 112 | 42 | 454 | 19 | .btn-sm — the shared small-button class |
| 2 | **A1** | ACCESS-008 | 102 | 28 | 864 | 18 | The `opacity` dim device (rows, badges, chips) |
| 3 | **T3** | TOUCH-003 | 96 | 6 | 188 | 17 | Dashboard text/number/date/search inputs |
| 4 | **A2** | ACCESS-008 | 88 | 38 | 644 | 32 | Light `--text-muted: #64748b` at 10–12 px |
| 5 | **T2** | TOUCH-003 | 74 | 6 | 128 | 13 | .btn-refresh — the shared primary-action class |
| 6 | **T6** | TOUCH-003 | 68 | 4 | 254 | 14 | Icon-only buttons (.btn-close, .order-expand-toggle, .note-action-btn, .btn-theme, .show-more-btn) |
| 7 | **T5** | TOUCH-003 | 64 | 36 | 412 | 14 | Non-control click surfaces (SYS-3) |
| 8 | **A4** | ACCESS-008 | 63 | 21 | 106 | 20 | White text on the `--primary` fill (3.68 : 1) |
| 9 | **A3** | ACCESS-008 | 60 | 27 | 366 | 11 | `--text-dimmed` — no AA-passing value in either theme |
| 10 | **T4** | TOUCH-003 | 60 | 2 | 270 | 11 | Dashboard checkboxes and radios (UA-sized) |
| 11 | **A8** | ACCESS-008 | 58 | 42 | 1196 | 21 | The four surfaces with no light palette (grey literals) |
| 12 | **A5** | ACCESS-008 | 38 | 12 | 114 | 12 | `--primary` as link text on a dark surface (3.37–4.44 : 1) |
| 13 | **T14** | TOUCH-003 | 37 | 17 | 1111 | 18 | Scheduler buttons and row actions (.act, .primary, .copyday, .otbtn, .more-toggle, .o-action, summary) |
| 14 | **T8** | TOUCH-003 | 30 | 6 | 48 | 5 | .btn-secondary and .btn-back |
| 15 | **A7** | ACCESS-008 | 26 | 0 | 36 | 9 | Unstyled placeholders — the UA default (INPUT-022) |
| 16 | **O1** | LAYOUT-003 | 24 | 24 | 24 | 12 | `.header` has no `flex-wrap: wrap` in the three inline stylesheets |
| 17 | **A6** | ACCESS-008 | 23 | 3 | 83 | 8 | Dark `--danger: #ef4444` as text (3.89 : 1) |
| 18 | **T13** | TOUCH-003 | 20 | 0 | 1874 | 9 | Scheduler grid inputs (`td > input`, `.floor-name`, `.ctl > input`, date/qty fields) |
| 19 | **T10** | TOUCH-003 | 18 | 4 | 36 | 4 | .mini-calendar-nav — the only width-driven cluster |
| 20 | **T12** | TOUCH-003 | 16 | 12 | 44 | 5 | Traceability controls (.trace-btn, .dir-btn, .export-btn, .graph-ctrl-btn, .search-input) |
| 21 | **T7** | TOUCH-003 | 16 | 6 | 76 | 3 | Tab / segmented chrome (.tab, .supplies-subtab, .notes-filter-btn) |
| 22 | **T9** | TOUCH-003 | 14 | 6 | 48 | 3 | Site navigation bar (.site-nav-link, .site-nav-toggle) — 4 copies |
| 23 | **F1** | LAYOUT-011 | 12 | 12 | 12 | 6 | The sticky stack is 335 px of a 450 px viewport at 200 % zoom |
| 24 | **T11** | TOUCH-003 | 8 | 0 | 24 | 2 | Sankey / Process Flow banner + retry buttons |
| 25 | **A9** | ACCESS-008 | 3 | 0 | 9 | 1 | Light category tokens (`--category-granola`, `--category-graham`) |
| 26 | **O2** | LAYOUT-003 | 2 | 2 | 2 | 1 | Scheduler has no compact layout (`#app` grid track sized by min-content) |
| 27 | **O3** | LAYOUT-003 | 2 | 2 | 2 | 1 | Scheduler print view renders the full horizon (`#topbar`/`#board` 4,779 px) |

---

## TOUCH-003 — 315 failing cells, 14 clusters

Every cluster below is one CSS rule. Every one of them is the same edit: add `min-height` (and, for the two
icon clusters, `min-width`) referencing a single new `--hit-min: 44px` token in `:root`. That token does not
exist today — introducing it is the change **SYS-4** is asking for, and it is what makes these thirteen edits
one decision rather than thirteen.

| Cluster | Selector / component pattern | Owning file | Cells | Clears alone | Failing elements | Smallest measured | Screens |
|---|---|---|---:|---:|---:|---|---|
| **T1** | `.btn-sm` and its nine variants (`.allocation-release-btn`, `.order-inventory-toggle`, `.er-edit/close/cancel-btn`, `#cal-prev`, `#cal-next`, `#cal-toggle`, `.today-tile-retry`, `.production-detail-close`, `.supply-request-done-btn`) | `dashboard.css:401` | 112 | 42 | 454 | 34 × 24 pt | S-07–S-09, S-22, S-27–S-28, S-30–S-34, S-36–S-37, S-42–S-43, S-45, S-51–S-53 |
| **T3** | `input` (types `text`, `number`, `date`, `search`), `select`, `textarea` — no shared rule exists, 17 ad-hoc id selectors | `dashboard.css` (new rule) | 96 | 6 | 188 | 110 × 19 pt | S-03–S-04, S-16, S-22, S-24, S-26–S-28, S-30, S-32–S-33, S-36, S-41, S-45, S-47–S-48, S-53 |
| **T2** | `.btn-refresh` and its variants (`#refresh-btn`, `#orders-refresh-btn`, `#orders-export-btn`, `.allocation-submit-btn`, `.order-save-lines-btn`, `#er-new-btn`, `#note-save-btn`, …) | `dashboard.css:243` | 74 | 6 | 128 | 57.6 × 27 pt | S-03, S-19, S-22, S-24, S-26, S-30, S-32–S-33, S-36, S-41, S-45–S-46, S-53 |
| **T6** | Icon-only buttons: `.btn-close`, `.order-expand-toggle`, `.note-action-btn`, `.btn-theme`, `.show-more-btn` | `dashboard.css` (new rule) | 68 | 4 | 254 | 12.9 × 15 pt | S-03, S-17–S-18, S-20, S-22, S-25–S-28, S-45, S-53–S-56 |
| **T5** | Non-controls carrying click handlers (**SYS-3**): `.lot-link`, `tr.expandable`, `tr.supply-item-row`, `tr.product-lot-row`, `.search-item`, `.collapsible-header`, `.disambig-btn` | `dashboard.css` (new rule) + `dashboard.js` | 64 | 36 | 412 | 43.2 × 14 pt | S-04, S-10–S-13, S-16–S-18, S-49–S-50, S-55–S-56, S-63–S-64 |
| **T4** | `input[type=checkbox]`, `input[type=radio]` — sized entirely by the user agent, no CSS at all | `dashboard.css` (new rule) | 60 | 2 | 270 | 13 × 13 pt | S-19–S-20, S-22, S-24–S-28, S-41, S-51–S-52 |
| **T14** | Scheduler controls: `.act`, `.act.sec`, `.primary`, `.copyday`, `.otbtn`, `.more-toggle`, `a.o-action`, `summary` | `scheduler/seven-wells-production-board.html` | 37 | 17 | 1,111 | 12 × 11 pt | S-72–S-73, S-75–S-88, S-90–S-91 |
| **T8** | `.btn-secondary`, `.btn-back` | `dashboard.css:1847`, `:1740` | 30 | 6 | 48 | 50.7 × 28 pt | S-30, S-32–S-34, S-39 |
| **T13** | Scheduler grid inputs: `td > input`, `.floor-name`, `.ctl > input`, `#pin-qty`, `#ao-*` | `scheduler/seven-wells-production-board.html` | 20 | 0 | 1,874 | 15 × 15 pt | S-76–S-82, S-86, S-88 |
| **T10** | `.mini-calendar-nav` — **the only width-driven cluster in the product** | `mini-calendar.css:12` | 18 | 4 | 36 | 24 × 97.8 pt | S-02–S-03, S-59, S-62 |
| **T7** | `.tab`, `.supplies-subtab`, `.notes-filter-btn` | `dashboard.css:272` | 16 | 6 | 76 | 39.3 × 26 pt | S-05, S-19, S-47 |
| **T12** | `.trace-btn`, `.dir-btn`, `.export-btn`, `.graph-ctrl-btn`, `.search-input` | `traceability.html` | 16 | 12 | 44 | 29 × 26 pt | S-63–S-64, S-68–S-70 |
| **T9** | `.site-nav-link`, `.site-nav-toggle` — **four byte-similar copies** (SYS-1) | `dashboard.css:116` + 3 inline copies | 14 | 6 | 48 | 37.8 × 26 pt | S-01, S-59, S-62 |
| **T11** | `.banner-btn`, `.data-failure-retry` | `sankey.html`, `process-flow.html` | 8 | 0 | 24 | 52 × 28 pt | S-59, S-62 |

### The change, per cluster

| Cluster | The single change |
|---|---|
| **T1** | `dashboard.css:401` — `.btn-sm { padding: 4px 10px }` → add `min-height: var(--hit-min); display: inline-flex; align-items: center`. Nothing else moves; the class is already inline. |
| **T3** | one new rule: `input:not([type=checkbox]):not([type=radio]), select, textarea { min-height: var(--hit-min) }`. The 17 id-scoped rules that size these fields today need no edit — none of them sets `height`. |
| **T2** | `dashboard.css:243` — same `min-height` on `.btn-refresh`. |
| **T6** | one new rule grouping the five icon classes at `min-height: var(--hit-min); min-width: var(--hit-min)`. `.btn-close` at **12.9 × 22** is the smallest control a user is asked to hit in the dashboard. |
| **T5** | the six handler-carrying non-controls need a control shape, not just a size — `min-height` plus the `role`, `tabindex` and key handler **IMP-031** already specifies. Sizing alone clears the measurement and leaves the keyboard defect. |
| **T4** | one new rule: `input[type=checkbox], input[type=radio] { width: 20px; height: 20px }` **plus** `min-height: var(--hit-min)` on the wrapping `label`, which is what actually carries the hit region. Setting 44 px on the box itself would be wrong. |
| **T14** | one new rule in the scheduler's inline `<style>` grouping the seven control classes. `.copyday` at **12 × 12** and `a.o-action` at **20 × 20** are the two smallest targets in the product. |
| **T8** | `dashboard.css:1847` / `:1740` — same `min-height` on `.btn-secondary` and `.btn-back`. |
| **T13** | one new rule: `input { min-height: var(--hit-min) }` in the scheduler. **This one is not free** — the board grid is 18 px per cell and 1,874 measurements deep; a 44 px row more than doubles the height of the plan table. It needs a layout decision, not a token. |
| **T10** | `mini-calendar.css:12` — `.mini-calendar-nav` is `width: 30px` (26 px at ≤1050 px, 24 px at ≤520 px) and 97.8 pt tall. Raise `min-width` to `var(--hit-min)`. |
| **T7** | `dashboard.css:272` — `.tab { padding: 10px 20px }` reaches 38 pt; the two sibling classes reach 26–34. One `min-height` on all three. |
| **T12** | one new rule in `traceability.html`'s inline style grouping the five control classes. |
| **T9** | `dashboard.css:116` — `.site-nav-link { padding: 5px 14px }` → `min-height`. Then the identical edit in `sankey.html:46`, `process-flow.html:33` and `traceability.html:46`, because the nav is copy-pasted four times (**SYS-1**). |
| **T11** | `.banner-btn` / `.data-failure-retry` in the two inline `<style>` blocks. |

---

## ACCESS-008 — 290 failing cells, 9 clusters

| Cluster | Colour / rule responsible | Owning file | Cells | Clears alone | Failing elements | Worst ratio | Themes | Screens |
|---|---|---|---:|---:|---:|---:|---|---|
| **A1** | `opacity: 0.5 / 0.55 / 0.6 / 0.65` used to de-emphasise rows that still carry readable text — `.order-row.so-ready`, the note cards, the expected-receipt rows, the allocation table, the empty-state icons | `dashboard.css` (every `opacity` on a text row) | 102 | 28 | 864 | **1.16 : 1** | dark + light | S-20–S-21, S-25–S-30, S-32–S-34, S-36–S-37, S-42–S-44, S-52, S-56 |
| **A2** | light `--text-muted: #64748b` at 10–12 px — table headers, `.num` cells, status badges | `dashboard.css:57` | 88 | 38 | 644 | 3.86 : 1 | light | S-04, S-09–S-19, S-21, S-24–S-28, S-30–S-34, S-36–S-37, S-41–S-43, S-46, S-49–S-51 |
| **A4** | white text on the `--primary` fill — dark `--primary: #3b82f6` plus four hard-coded `.site-nav-link.active { background: #3b82f6 }` copies | `dashboard.css:15`, `:122` + 3 inline copies | 63 | 21 | 106 | 3.68 : 1 | dark + light | S-01, S-03, S-14–S-15, S-19, S-22, S-24, S-26, S-30, S-32–S-33, S-36, S-41, S-45–S-46, S-53, S-59, S-62–S-64 |
| **A3** | `--text-dimmed` — `#64748b` in dark, `#94a3b8` in light; **neither value passes AA on its own surface** | `dashboard.css:14`, `:58` | 60 | 27 | 366 | 2.08 : 1 | dark + light | S-02–S-03, S-08–S-11, S-13, S-20, S-22, S-45, S-53 |
| **A8** | grey literals on the four surfaces with no light palette — `#555`, `#888`, `#b46a00`, `#64748b` in `sankey`, `process-flow`, `traceability` and the scheduler | four inline `<style>` blocks | 58 | 42 | 1,196 | **1.1 : 1** | same measurement twice | S-59–S-65, S-70–S-71, S-74–S-82, S-84, S-90–S-91 |
| **A5** | `--primary: #3b82f6` used as *link text* on `--surface` / `--surface-alt` — `.lot-link`, `.order-link` | `dashboard.css:15` | 38 | 12 | 114 | 3.37 : 1 | dark + light | S-04–S-05, S-11–S-13, S-17–S-18, S-25–S-28, S-56 |
| **A7** | no `::placeholder` rule anywhere — nine fields inherit the user agent's `rgb(117,117,117)` (**INPUT-022**) | `dashboard.css` (new rule) | 26 | 0 | 36 | 2.69 : 1 | dark only | S-22, S-27–S-28, S-30, S-32–S-33, S-36, S-45, S-53 |
| **A6** | dark `--danger: #ef4444` used as text — `.readiness-shortage`, `.allocation-release-btn`, `.note-due.overdue` | `dashboard.css:21` | 23 | 3 | 83 | 3.89 : 1 | dark only | S-20, S-30–S-34, S-36–S-37 |
| **A9** | light `--category-granola: #a16207` (3.99) and `--category-graham: #15803d` (4.07) | `dashboard.css:62`, `:63` | 3 | 0 | 9 | 3.99 : 1 | light | S-09 |

### The change, per cluster

| Cluster | The single change |
|---|---|
| **A1** | Retire `opacity` as the de-emphasis device and de-emphasise with a token colour that still clears AA. This is **IMP-067**, and it is the largest ACCESS-008 cluster in the product. `.so-ready-pill` measures **1.16 : 1** — a declared `rgb(134,239,172)` that would pass on its own, halved by the row's `opacity: 0.6`. No colour edit reaches it; only removing the multiplier does. |
| **A2** | `dashboard.css:57` — one hex value. `#64748b` reaches 4.34 : 1 on `--surface` and 3.86 : 1 on `--row-header`; about `#5a6472` clears 4.5 : 1 on all three light surfaces. **The single highest-yield, lowest-risk change in this file.** |
| **A4** | Split the token: a `--primary-fill` that clears 4.5 : 1 under white (`#2563eb`, which is already the light value) from `--primary` used as text. Then replace the four hard-coded `#3b82f6` site-nav copies with it. Note this cluster and **A5** pull the same token in opposite directions — that is the argument for the split, not a conflict. |
| **A3** | `dashboard.css:14` and `:58`. There is no value that works: `--text-dimmed` is the third step of a three-step scale whose second step (**A2**) is already failing. Retire the token and fold its uses into `--text-muted`, or accept that a third step needs a larger font, not a lighter colour. |
| **A8** | Give the four surfaces the paired light/dark token block `dashboard.css` already has. Their light and dark captures are byte-identical (06, "Surfaces with no light palette"), so these 58 cells are **29 real measurements reported twice** — the largest single number in this table is also the softest. Worst is `#traceDetail > h3` at **1.1 : 1**, near-white text on white paper under `@media print`. |
| **A5** | The lighter half of the A4 split — `#60a5fa`, already defined as `--primary-hover`. |
| **A7** | One `::placeholder` rule. All 36 failures are dark-theme, all at exactly 2.69 : 1, all `rgb(117,117,117)` on `--search-bg`. This is **IMP-025** at its full nine-field scope. |
| **A6** | `dashboard.css:21` — dark `--danger: #ef4444` → `#f87171` for text. The fill uses (`--badge-red-bg`) are a different token and are unaffected. |
| **A9** | `dashboard.css:62-63` — darken two light category values past 4.5 : 1. Three cells; listed for completeness. |

---

## LAYOUT-003 / ACCESS-001 — 28 failing cells, 3 clusters

| Cluster | Selector responsible | Owning file | Cells | Clears alone | Overflow | Screens |
|---|---|---|---:|---:|---:|---|
| **O1** | `.header { display: flex; justify-content: space-between }` with **no `flex-wrap`**, holding a 320 px `.mini-calendar-strip` | `sankey.html:59`, `process-flow.html:46`, `traceability.html:59` | 24 | 24 | 81–135 px | S-57–S-67, S-70 |
| **O2** | `#app { display: grid; height: 100vh }` — the implicit column is sized by `#topbar`'s 580 px min-content and there is no compact layout below it | `scheduler/seven-wells-production-board.html:26` | 2 | 2 | 190 px | S-86 |
| **O3** | `#topbar` / `#main` / `#board` render the full plan horizon at **4,779 px** under `@media print` | `scheduler/seven-wells-production-board.html:185` | 2 | 2 | 3,339 px | S-91 |

### The change, per cluster

**O1 — one declaration, already written, in the wrong file.** `mini-calendar.css:136-143` already carries the
mobile behaviour the strip needs:

```css
@media (max-width: 768px) {
  .mini-calendar-strip { order: 10; width: 100%; justify-content: center; overflow-x: auto; }
}
```

`order` and `width: 100%` only do anything inside a **wrapping** flex container. `dashboard.css:2223-2225`
provides exactly that for `index.html`:

```css
@media (max-width: 768px) {
  .app-header  { flex-wrap: wrap; gap: 8px; }
  .header-right { width: 100%; flex-wrap: wrap; }
}
```

The three standalone pages define the same `.header` / `.header-right` structure and **have no such block**, so
the strip cannot wrap and pushes `.header-right`'s right edge to 471–504 px inside a 390 px viewport. Copying
those three lines into each of the three inline stylesheets clears **all 24 cells** — and this is measured
support for **IMP-027** and a second demonstration of **SYS-1**: the fix exists, it just does not live anywhere
the other three pages can reach it.

**O2 / O3 — the scheduler, not the shared layer.** O2 is the same "no compact layout" defect as O1 but on a
grid app whose min-content is ~580 px, so it needs a real responsive pass rather than one declaration
(**IMP-027**). O3 is `@media print` rendering the whole horizon at 4,779 px on a 1,440 px page — **IMP-065**,
and unrelated to the other two.

---

## LAYOUT-011 — 12 failing cells, 1 cluster

All twelve are one number, and it is not a per-screen number.

| Cluster | Cause | Cells | Clears alone | Screens |
|---|---|---:|---:|---|
| **F1** | At a 720 × 450 CSS-px viewport the sticky stack measures **335 px — 74 % of the viewport**, leaving 115 px of content | 12 | 12 | S-12, S-30, S-32, S-33, S-39, S-52 |

Measured bar geometry, identical in all six screens and both themes:

| Bar | Position | Top → bottom | Height |
|---|---|---|---:|
| `nav#siteNav` | sticky, top-anchored | 0 → 48 | 48 px |
| `header.app-header` | sticky | 48 → 297 | **249 px** |
| `nav.tab-bar` | sticky | 297 → 335 | 38 px |

The 249 px header is not a bug in isolation — it is `dashboard.css:2223-2225` working correctly. At 720 CSS px
the `≤768px` block fires, `.app-header` wraps, `.header-right` takes a full row and `.mini-calendar-strip`
takes another. Three rows of sticky chrome is right for a phone held in the hand and wrong for a 450 px-tall
window, and nothing in the CSS distinguishes the two cases because **every media query in the product keys on
width alone**.

**The change:** one height-keyed rule — `@media (max-height: 600px) { .app-header { position: static } }`, or
the same guard moving `.mini-calendar-strip` out of the sticky region. `nav.tab-bar` then re-anchors at 48 px
and the stack drops from 335 px to 86 px (19 % of the viewport). That clears **all 12 cells**: the four blocked
by `.app-header` (S-12, S-52) and the eight blocked by `nav.tab-bar` (S-30, S-32, S-33, S-39), which is only
at 297 px because the header above it is 249 px tall. This is **IMP-066**.

---

## What a PR actually buys — bundles

Single clusters clear less than their footprint because the dashboard screens fail on several at once. These
are the measured totals for the combinations worth considering.

### TOUCH-003 (of 315 cells)

| Bundle | Cells cleared | Notes |
|---|---:|---|
| T1 alone | 42 | the largest single-cluster yield |
| T1 + T2 | 48 | +6 for the second button class — they co-occur almost everywhere |
| T1 + T2 + T8 | 60 | all four dashboard button classes |
| **T1–T4, T6–T8 — one `min-height` on every dashboard button and input** | **178** | 7 rules, one file, no layout decision beyond row height |
| **+ T5 (the SYS-3 non-controls)** | **238** | 8 rules; T5 also needs the `role`/`tabindex` work in IMP-031 |
| **+ T9, T10 (nav + mini-calendar)** | **254** | 10 rules; T9 is the same edit in four files |
| + T11, T12 (sankey / process-flow / traceability) | 278 | 12 rules |
| Scheduler only (T13 + T14) | 37 | T13 doubles the plan-table height — a layout decision, not a token |
| **All 14** | **315** | |

### ACCESS-008 (of 290 cells)

| Bundle | Cells cleared | Notes |
|---|---:|---|
| A2 alone | 38 | one hex value in `dashboard.css:57` |
| A1 alone | 28 | removing the `opacity` device |
| A2 + A3 | 74 | both muted tokens |
| **A1 + A2 + A3** | **136** | the dim device plus both muted tokens — three edits |
| **+ A4 + A5 (the `--primary` split)** | **192** | five edits, all in `dashboard.css` |
| **+ A6, A7, A9 — everything in `dashboard.css`** | **232** | eight edits, one file |
| A8 alone (the four unthemed surfaces) | 42 | but 29 distinct measurements — light and dark are the same capture |
| **All 9** | **290** | |

---

## Recommendation for the next PR

Ranked by cells cleared per unit of risk, not by footprint:

1. **A2** — one hex value, 38 cells, zero layout risk. Nothing else in this file has that ratio.
2. **O1** — three lines already written and proven on `index.html`, copied into three files, 24 cells.
3. **F1** — one height-keyed media query, 12 cells, and it closes LAYOUT-011 completely.
4. **A3 + A6 + A9** — three more token values, 33 further cells, still no layout risk.
5. **T1 + T2 + T8 + T7** — the four dashboard button classes behind one new `--hit-min` token, 76 cells. First
   change in this list that moves pixels; needs a look at the dense toolbars before it lands.
6. **A1** — 28 cells and the worst ratio in the product (1.16 : 1), but it is a visual-design decision
   (**IMP-067**) about how a de-emphasised row should read, not a token swap.

**Explicitly not recommended for the next PR:** **T13** (the scheduler grid, 1,874 measurements — a 44 px input
more than doubles the plan table and needs a layout decision first) and **A8** (the four unthemed surfaces,
which is the whole of **IMP-027** plus a token block per file, and whose 58 cells are 29 measurements counted
twice).

---

## Reproducing this

The clustering re-runs `tests/visual/lib/checks.mjs`'s `touchTargets()` and `contrast()` against the same
static server and API stub as 06, with the `worst: …slice(0, 12)` caps at `checks.mjs:112` and `:256` lifted and
the element's `tagName` and full `class` attribute recorded alongside each failure. Failures are then grouped
by owning CSS rule; a capture cell is counted as *cleared* by a cluster set only when every failing element in
it belongs to that set. LAYOUT-003 and LAYOUT-011 numbers are read directly from
`screenshots/results.json` — their per-capture lists are not truncated at the counts involved.
