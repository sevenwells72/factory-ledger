# FL Canvas (experimental, read-only)

A zoomable, infinite-canvas view of Factory Ledger, built from the ConceptA design
prototype. It is a separate static site (plain HTML/CSS/JS, no build step) that
lives only in `canvas/` on branch `exp/fl-canvas`. It changes nothing in the
existing app and only sends GET requests.

## Run it

```sh
python3 -m http.server 8765 --directory canvas
```

Then open http://localhost:8765/ (run the command from the repo root).

## What it does

- One canvas with seven area cards joined by flow arrows: Production, Packing,
  Receiving, Inventory (hub), Orders & shipping, Quality & traceability, Reports.
- Semantic zoom. A card shows its next level once it fills enough of the screen:
  factory totals → area detail → inventory category (product cards) → a product's
  working screen (lots, KPIs, where it comes from and goes, recent activity).
- Click a lot to open its record in the side panel: in / out / on hand, trace back
  (ingredient lots), trace forward (batches and shipments), and its transactions.
  Orders and expected receipts open in the same panel.
- Mouse-wheel zoom around the pointer, drag to pan, click to fly, minimap,
  breadcrumb, "Entire factory" button, −/+ zoom, Esc to step out, `/` to search.
- Search jumps to areas, categories, products, on-hand lots, open orders and
  expected receipts. For 3+ characters it also asks `GET /dashboard/api/search`
  and lists lots and orders that are not on the canvas (record only).
- Task buttons (Receive, Make, Pack, Ship, Count) open the existing dashboard's
  `entry.html?action=…` in a new tab. The dashboard pre-fills only `order` for
  Ship, so Ship from an open order passes `&order=<id>`; nothing else is
  pre-filled. There is no count page: Count opens `action=adjust` (pick reason
  Physical count). Recording entries needs a signed-in actor key on that page.

## Configuration: `canvas/config.js`

- `apiBase`: **staging** by default
  (`https://fastapi-staging-production-dd7b.up.railway.app`).
- `apiKey`: left empty on purpose. The page fetches the dashboard key at load
  from `GET {apiBase}/auth/dashboard-key`, the same way the existing dashboard
  does. A committed key literal would fail the repo's `key-literal-guard` CI
  check, and the key rotates.
- `dashboardBase`: where task buttons and links go (staging's `/dashboard`).
- Auth header: `X-API-Key`. The API allows any origin (CORS `*`), so the page
  works from localhost or its own Netlify site.

## Data sources (all GET, dashboard key)

| Canvas part | Endpoint |
|---|---|
| Inventory totals, categories, product cards, lots on hand | `/inventory/current?limit=500` (one row per lot with stock) |
| Units per item, items at zero, low-stock flags | `/supplies/inventory?category=ingredient` and `=packaging` |
| Production, Packing, Reports corrections, large corrections | `/transactions/history?since=<7 days>&limit=1000` (posted only) |
| Planned runs (Production card, product screen) | `/production/runs?from=&to=` / `?product=&from=` |
| Receiving | `/expected-receipts?status=open` |
| Orders & shipping | `/sales/dashboard`, `/sales/orders/counts`, `/sales/orders?state=open&limit=200` |
| Order record | `/sales/orders/{id}` |
| Product working screen | `/dashboard/api/product/{id}/lots`, `/transactions/history?product_name=&since=<30 days>` |
| Lot record | `/dashboard/api/lot/{lot_code}?product_id=`, `/trace/batch/…` or `/trace/ingredient/…` |
| Search (extra results) | `/dashboard/api/search?q=` |
| Order created times (Orders area rows, read when that area opens; max 5) | `/sales/orders/{id}` (cached, reused by the order panel) |
| Ingredients relationship map + ingredient panel (read on first open) | `/transactions/history?transaction_type=make&since=<90 days>&limit=1000` |

The page sends at most 4 requests at a time, and only the product under the
centre of the screen loads its working-screen data, because the API's database
pool is small. Lot first-entry dates come from the 30-day history. Up to 20
older lots per product are then looked up one by one. Lots past that show no date.

## Look and layout

- **Themes.** Three backgrounds, switched with the three swatches in the header:
  Clean industrial (default), Warm modern, Operational map (faint drafting grid
  plus dashed stage zones: Inbound, Make & pack, Outbound, Oversight). The choice is
  remembered in `localStorage` (`flcanvas.theme`); if storage is blocked the page
  just uses the default.
- **Icons.** Lucide (lucide-static 0.460.0, ISC), vendored in `vendor/lucide/`
  (`icons.js` holds only the 7 icons used). No CDN.
- **Card sizes.** Each overview card is as tall as its content needs, at the
  overview and at the zoomed-in level; matching cards (Production/Packing,
  Receiving/Orders, Quality/Reports) share a height. The flow arrows are drawn from
  the measured card edges after layout (`layoutWorld()`), so they follow any resize.
- **Ingredients.** Cards lead with the name (wraps up to 3 lines), then quantity,
  then `lots · SKU`. Sort: Quantity ↓ (default), Name A–Z, Name Z–A; kept while
  the page is open, no requests. **Grid view | Relationship map** toggle.
- **Alerts.** Inventory shows stock alerts only (below reorder, open orders short
  on stock). Overdue orders are on Orders & shipping, overdue expected receipts on
  Receiving.

## Ingredients relationship map

Two bases, switched inside the map:

- **Recipes (default), labelled "Based on recipes, not actual usage."** Coconut
  share = the ingredient's recipe weight in coconut batch products ÷ its weight in
  coconut + granola batch products, one batch of each recipe
  (`GET /bom/batches/{id}/formula` for the 25 batch products in `families.js`,
  read once, max 4 at a time, cached). Formula lines excluded from inventory
  (sweetening water) are left out. Only ingredients in no recipe go to Unplaced.
- **Actual use · 90 days.** Coconut share of the ingredient's use in posted make
  entries over the last 90 days, split by the batch product made
  (`GET /transactions/history?transaction_type=make`). Ingredients with no use in
  the window go to Unplaced.

Nodes are labelled with on-hand quantity and sized by it. Each node keeps a thin
leader line to a tick at its exact position on the axis, so nodes moved sideways to
avoid overlap still show the true share.

Factory Ledger has no coconut/granola flag on products, so the product → family list
in `families.js` was approved by Michael on 2026-10-21 (4 coconut, 21 granola, by
name). Setting `approved: false` puts every ingredient back in the tray.

## Record timestamps

Summary rows show `Created … · Updated …` and side panels a **Record history**
section, all in America/New_York (`Oct 20, 2026 · 10:30 AM`). Only real API fields
are used; anything missing reads "Not recorded". Due / expected / ship dates are
shown separately from created / updated. What the API returns today:

| Record | Created | Created by | Updated | Updated by | Endpoint that would need the missing fields |
|---|---|---|---|---|---|
| Sales order | `created_date` + `created_time` (detail only) | missing | missing (`state_changed_at/by` covers state changes only) | missing | `GET /sales/orders` (list has no created time) and `GET /sales/orders/{id}` |
| Expected receipt | `created_at` | `created_by` | `updated_at` | missing | `GET /expected-receipts` |
| Make / pack / ship / adjust entry | `created_at` | `operator_id` | only `latest_correction_created_at` (entries are corrected, not edited) | missing (correction actor not returned) | `GET /transactions/history` |
| Shipment (on an order) | `shipped_date` + `shipped_time` | missing | missing | missing | `GET /sales/orders/{id}` → `shipments[]` |
| Lot | none of its own; first / latest ledger entry `created_at` | missing (timeline has no operator) | latest entry only | missing | `GET /dashboard/api/lot/{code}` → `timeline[]` |

## Not connected yet (shown with a striped "not connected yet" label)

- **Exceptions and holds.** `GET /exceptions` is actor-only, so the dashboard key
  gets 403. The page tries it, and shows real counts if it ever succeeds.
- **Assistant.** The box shows the current context, but asking needs a signed-in
  actor session, so the input is disabled.
- **Mock recall schedule** and **owner weekly view.** No endpoint exists.
- **Packing plan.** No read endpoint the dashboard key can use.
- **Kosher** is shown only where it is in the data: the `EXTRA-KOSHER (SS)`
  product names and the `kosher_label` on lot traces.

## Known limits

- Built for desktop. It loads on a phone without horizontal scroll, but there is
  no pinch zoom, and the working screen is small there.
- Data is loaded once at page load. Reload the page to refresh.
- The relationship map reads up to 1,000 make entries for its 90-day window and
  says so if it hits that limit.
- Staging's dashboard is served under `/dashboard/`. Some of its own links use
  root paths, so a few in-dashboard links may 404 there; the canvas's own links
  (`entry.html`, `traceability.html`, `history.html`, `shift-summary.html`) load.
