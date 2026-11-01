# Notes endpoints: security notes

Status: **documented, unchanged.** Written for the main.py modularization PR0 (2026-11-01). PR0 and the planned Notes extraction (PR1) keep this behaviour exactly as it is. `tests/test_notes_characterization.py` pins it, so it can only change in its own reviewed PR.

Code: `main.py::# DASHBOARD API — Notes / To-Dos / Reminders (NO AUTH)` (5 routes) and the `NoteCreate` / `NoteUpdate` models. UI: the Notes panel in `dashboard/dashboard.js` (`refreshNotes`, and the save, toggle and delete handlers).

## What the endpoints do today

| Route | Auth | Who is allowed | Actor attribution (`actor_write_audit`) |
|---|---|---|---|
| `GET /dashboard/api/notes` | **None** | Anyone who can reach the API. Any key, a wrong key or no key returns every note. | n/a |
| `POST /dashboard/api/notes` | `verify_api_key` | Master key, dashboard key (allowlisted), personal actor keys and sessions. The read-only key gets 403. A deactivated actor's key gets 403. | **Session tokens only** |
| `PUT /dashboard/api/notes/{note_id}` | `verify_api_key` | Same as POST | Session tokens only |
| `DELETE /dashboard/api/notes/{note_id}` | `verify_api_key` | Same as POST | Session tokens only |
| `PUT /dashboard/api/notes/{note_id}/toggle` | `verify_api_key` | Same as POST | Session tokens only |

Notes are annotations and to-dos. They are not inventory, production or traceability records, and no Notes route touches `transactions`, lots or allocations.

## Implications

1. **Unauthenticated read.** Anyone who knows the public Railway API URL can list every note, including titles, bodies, due dates and the linked entity (`entity_type` and `entity_id`, for example a lot or customer). Production notes may contain customer names, supplier issues or internal to-dos. The dashboard's other unauthenticated reads (`/dashboard/api/inventory/*`, `/dashboard/api/activity/*`, `/dashboard/api/lot/{lot_code}`, `/dashboard/api/search` and others) have the same exposure. Seventeen routes have no auth dependency in total; `tests/fixtures/route_contract/routes.jsonl` is the authoritative list. Notes is not unusual in this respect.
2. **Hard delete.** `DELETE` removes the row. Nothing in the app recovers a deleted note. This is acceptable for annotations, but it is not the append-only model used for ledger records.
3. **Partial attribution.** Each write handler calls `_record_actor_write` only when `request.state.key_kind == "session"`. A note written with a personal actor key directly, outside a session, records no actor. So does the shared dashboard key or the master key. Notes are the only writes with this condition; the other `_record_actor_write` call sites attribute every actor key. This conflicts with the "staff use their own production accounts / named actor" principle for this one surface.
4. **No role check.** Every key that passes `verify_api_key` may write notes, including floor actors. `permissions.py` lists the four write routes in `UNGATED_ROUTES`, which means no role check.

## Before changing any of this (a separate PR, not PR0 or PR1)

- **Adding auth to the GET:** the dashboard's `refreshNotes` must send `X-API-Key` first. It accepts a `headers` option but does not always pass one. The dashboard (Netlify) and the API (Railway) deploy separately, so ship the dashboard change first, or accept both behaviours for one release. Row 32 of `FACTORY_LEDGER_CHANGELOG.md` records the same race for the earlier Notes write-auth change. Also decide which keys may read: dashboard, read-only, actor and session.
- **Attributing personal-key writes:** drop the `key_kind == "session"` condition. `_record_actor_write` already does nothing for legacy keys. Update `test_non_session_writes_record_no_attribution` in the same PR.
- Get the owner's approval and follow the High-risk tier in `CLAUDE.md`. These are auth changes.
