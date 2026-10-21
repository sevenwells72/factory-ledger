# Change Log

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
