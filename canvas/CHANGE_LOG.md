# Change Log

## 2026-10-21 19:14 — Mouse zoom clicks + Edit layout mode with saved layouts
- **File(s) changed:** `canvas/app.js`, `canvas/index.html`, `canvas/styles.css`, `canvas/README.md`, `canvas/layout.json` (new), `canvas/review/edit-layout-overview.png`, `canvas/review/edit-layout-categories.png` (new)
- **What changed:** Left click on empty canvas zooms in one step and right click anywhere on the canvas zooms out one step, both centred on the pointer; the browser context menu is suppressed on the canvas only; card click-to-fly, drag-pan, wheel, Esc, minimap and −/+ unchanged. New "Edit layout" mode (top-bar toggle + toolbar with Snap to 8px / Shift, Reset layout with confirm, Save layout to code, Done): move cards by dragging, resize from four corner handles with content-based minimum sizes; covers the 7 overview cards and the 4 Inventory category tiles (product grids follow the tile size). Connectors, labels, zones and minimap redraw live from the real card edges, switching to facing edges when moved cards no longer face. Layout auto-saves per browser in localStorage (`flcanvas.layout`, try/catch); `canvas/layout.json` (empty = automatic layout) is the code default, and "Save layout to code" downloads layout.json to be committed there. Layout code split into measure (`layoutWorld`) and place (`placeWorld`). Checked with Playwright at 1440×900 (39 checks: click/right-click zoom, drag ≠ click, move/resize/min size/snap, arrows follow, reload keeps layout, reset, blocked storage, GET only, max 4 API requests in flight).
- **Why:** Michael asked for mouse zoom and a way to arrange the canvas cards himself and keep the arrangement. No push, merge or deploy.

---

## 2026-10-21 17:58 — Relationship map: family list approved, recipe-based default, usage toggle
- **File(s) changed:** `canvas/families.js`, `canvas/app.js`, `canvas/styles.css`, `canvas/README.md`, `canvas/review/ingredients-relationship-map.png`
- **What changed:** Product → family list set to approved (90007 = product 128 already coconut; no other coconut batch products found on staging, inactive included, so nothing added). Map now defaults to recipe-based positions (one batch of each of the 25 recipes, by weight, inventory-excluded lines left out), labelled "Based on recipes, not actual usage"; the 90-day actual-use mode is a second option with its own label and tray. Ingredient panel shows the recipe split too. Map gives the plot most of the space and smaller nodes so all 44 recipe ingredients fit.
- **Why:** Michael approved the classification and asked for recipe-based positioning because staging has no recent coconut makes.

---

## 2026-10-21 17:38 — Canvas UI enhancements: icons, themes, content-aware cards, record times, ingredient sort + relationship map
- **File(s) changed:** `canvas/app.js`, `canvas/api.js`, `canvas/styles.css`, `canvas/index.html`, `canvas/README.md`, `canvas/families.js` (new), `canvas/vendor/lucide/icons.js` + `LICENSE` (new), `canvas/review/*.png` (new)
- **What changed:** Lucide icons (vendored, no CDN) on the 7 overview cards; 3 switchable background themes (industrial default, warm, operational map) remembered in localStorage; overview cards sized from measured content with paired heights and flow arrows re-anchored to real card bounds; Created/Updated lines on make, pack, expected-receipt and order rows and working-screen activity, plus Record history in the lot, order and expected-receipt panels (America/New_York, real fields only, "Not recorded" otherwise); ingredient cards lead with the name; ingredient sort (Quantity ↓, A–Z, Z–A); Grid view | Relationship map toggle with a 90-day make-use map that places nothing until the product → family list in `families.js` is approved; overdue orders moved off the Inventory card (stock alerts only); api.js now holds a request slot until the body is read so the 4-in-flight cap is exact.
- **Why:** Michael asked for a visual/usability pass on FL Canvas without changing data, math or the backend. No push, merge or deploy.

---

## 2026-10-19 16:05 — Experimental read-only FL Canvas site
- **File(s) changed:** `canvas/index.html`, `canvas/styles.css`, `canvas/app.js`, `canvas/api.js`, `canvas/config.js`, `canvas/README.md`, `canvas/CHANGE_LOG.md`
- **What changed:** New standalone static site in `canvas/`: zoomable infinite-canvas version of the dashboard (ConceptA design), live GET-only data from the staging API, lot/order/expected-receipt side panel, search, minimap, task links into the existing dashboard. Kept in `canvas/` because the repo-root `CHANGE_LOG.md` is off-limits for this experiment.
- **Why:** Michael asked for an experimental, read-only canvas site on branch `exp/fl-canvas`, separate from the existing app (no changes outside `canvas/`, no deploy until approved).

---
