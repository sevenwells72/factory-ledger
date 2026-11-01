# Notes endpoints: security notes

Status: **remediated in the Notes auth PR (2026-11-01)**: the list now needs a key and every write by a named person is attributed. First documented, unchanged, in the main.py modularization PR0 (#163). `tests/test_notes_characterization.py` pins the rules below, so they can only change in their own reviewed PR.

Code: `main.py::# DASHBOARD API — Notes / To-Dos / Reminders` (5 routes) and the `NoteCreate` / `NoteUpdate` models. UI: the Notes panel in `dashboard/dashboard.js` (`refreshNotes`, and the save, toggle and delete handlers).

## What the endpoints do

| Route | Auth | Who is allowed | Actor attribution (`actor_write_audit`) |
|---|---|---|---|
| `GET /dashboard/api/notes` | `verify_api_key` | Master key, dashboard key (allowlisted), personal actor keys and sessions. No key gets 401. A wrong key or a deactivated actor's key gets 403 "Invalid API key". The read-only GPT key gets 403 "API key not authorized for this endpoint". | n/a |
| `POST /dashboard/api/notes` | `verify_api_key` | Same as GET | Every named person: personal actor key or session |
| `PUT /dashboard/api/notes/{note_id}` | `verify_api_key` | Same as GET | Same |
| `DELETE /dashboard/api/notes/{note_id}` | `verify_api_key` | Same as GET | Same |
| `PUT /dashboard/api/notes/{note_id}/toggle` | `verify_api_key` | Same as GET | Same |

The shared master and dashboard keys name a surface, not a person, so their writes record no actor (`_record_actor_write` does nothing for them). This is the same rule as every other attributed write.

Notes are annotations and to-dos. They are not inventory, production or traceability records, and no Notes route touches `transactions`, lots or allocations.

## What is still true

1. **The dashboard key is public.** `GET /auth/dashboard-key` hands the shared dashboard key to anyone, by design (row 204, O-7). A key is now needed to list notes, which stops URL-only access, crawlers and anyone who only has the API address. But someone who knows to fetch the dashboard key first can still read every note. This is the same exposure as every other route on `DASHBOARD_KEY_ALLOWLIST` (sales orders, customers, inventory, lots and more). Closing it for all of them needs one of these:
   - reads require a personal sign-in once every dashboard user signs in (key sessions are on in production), and the route stops publishing the key; or
   - the dashboard key is limited to screens that must work without a sign-in.

   Either is a product decision with its own PR.
2. **Hard delete.** `DELETE` removes the row. Nothing in the app recovers a deleted note. The attribution row (method, route, target id, actor) now records who deleted it. This is acceptable for annotations, but it is not the append-only model used for ledger records.
3. **No role check.** Every key that passes `verify_api_key` may write notes, including floor actors. `permissions.py` lists the four write routes in `UNGATED_ROUTES`, which means no role check. The GET has no `ROUTE_ACTIONS` entry, so every role may read.

## Other routes with no auth dependency

There are 18 others; `tests/fixtures/route_contract/routes.jsonl` is the authoritative list. This PR does not change them.

| Route | What it returns | Exposure |
|---|---|---|
| `GET /dashboard/api/activity/shipments` | Shipping log: customer name, order reference, transaction notes, products, lots and quantities | **High.** Customer names and free-text notes |
| `GET /dashboard/api/activity/receipts` | Receiving log: shipper or supplier name, BOL reference, notes, lots and quantities | **High.** Supplier names, BOLs, notes |
| `GET /dashboard/api/activity/daily-entries` | Every posted ledger entry for one day | **High.** Full daily ledger detail |
| `GET /dashboard/api/lot/{lot_code}` | Lot detail with its full transaction timeline: customer, shipper, order and BOL references, adjust reasons, notes | **High.** Traceability of any lot by code |
| `GET /dashboard/api/search` | Search across products, lots, sales orders and customers (including aliases) | **High.** Customer and order lookup |
| `GET /audit/integrity` | Integrity checks with lot codes, products, balances and transaction notes | **Medium.** Internal diagnostics |
| `GET /dashboard/api/production` | Daily make/pack calendar by product | **Medium.** Production volumes |
| `GET /dashboard/api/inventory/finished-goods`, `/batches`, `/ingredients` | On-hand inventory by product and lot | **Medium.** Stock levels |
| `GET /dashboard/api/product/{product_id}/lots` | Lots and on-hand quantity for one product | **Medium** |
| `GET /auth/dashboard-key` | The shared dashboard key | By design; see "What is still true", point 1 |
| `GET /auth/session/config` | Whether sign-in and PINs are on, and the idle timeout | **Low.** Public flags |
| `POST /auth/pin`, `POST /auth/pin/reset` | Phone PIN sign-in | Authenticated in the body by the device secret and PIN, not by a key |
| `GET /`, `GET /health` | App name, version, feature list; database and pool status | **Low** |
| `GET /dash/fl-assistant` | The static Assistant page (HTML only; its API calls need a key) | **Low** |

Recommended order for follow-up PRs, each with the same dashboard-first deploy as Notes:
1. Activity, lot detail and search, which hold customer and supplier names and free text.
2. Integrity, production and inventory.

The dashboard already has the key for each of these. Each follow-up is the Notes pattern: add `verify_api_key`, add the route to `DASHBOARD_KEY_ALLOWLIST`, and send the key from the page before the API requires it.

## Changing any of this again

- The dashboard (Netlify) and the API (Railway) deploy separately. Ship the page that sends the key first and confirm it is live. Then make the API require it. Row 32 of `FACTORY_LEDGER_CHANGELOG.md` records the same order for the earlier write-auth change.
- Update `tests/test_notes_characterization.py` and the route-contract snapshot in the same PR, and review the snapshot diff.
- Get the owner's approval and follow the High-risk tier in `CLAUDE.md`. These are auth changes.
