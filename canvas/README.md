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

The page sends at most 4 requests at a time, and only the product under the
centre of the screen loads its working-screen data, because the API's database
pool is small. Lot first-entry dates come from the 30-day history. Up to 20
older lots per product are then looked up one by one. Lots past that show no date.

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
- Staging's dashboard is served under `/dashboard/`. Some of its own links use
  root paths, so a few in-dashboard links may 404 there; the canvas's own links
  (`entry.html`, `traceability.html`, `history.html`, `shift-summary.html`) load.
