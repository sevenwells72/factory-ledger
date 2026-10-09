# F1 assistant deployment

Merge order: **A5 #88 → A2 #89 → A3b #94 → F1 #90**. A5, A2 and A3b
are merged; F1 is rebased onto `ce7bb57`. F1 remains unmerged.
The `tests/schema/schema.sql` tail includes **068 only**; 062–065 and
069–071 are already in the merged production schema dump.
Migration 066 is a separate, unmerged supplier-label backfill; do not include
or apply it as an F1 prerequisite. Add its include only after it merges.

## Production prerequisites (manual; not performed by this PR)

Keep `ASSISTANT_ENABLED` unset or `0` throughout deployment. In that state,
`GET /dash/fl-assistant` returns 404 and assistant requests remain disabled.
The interface-neutral **GET /correction-reasons becomes reachable to master
and actor keys**, even while the assistant is disabled. It only reads the
existing A3a reason catalog; the dashboard key remains denied.

Before ever setting `ASSISTANT_ENABLED=1` in production, apply
`migrations/068_fl_assistant.sql` **by hand**, as the app/table owner, on the
verified production database through port **5432**. First confirm the A5
062–064, A2 065 and A3b 069–071 prerequisites were applied by their owners. F1 has no
startup migration. Use psql with `ON_ERROR_STOP=1` and a protected connection
configuration; never echo the URI or put credentials in shell arguments.
Run from the repository root in that authenticated psql session:

```sql
\set ON_ERROR_STOP on
BEGIN;
SET LOCAL lock_timeout = '5s';
SET LOCAL statement_timeout = '60s';
SET LOCAL search_path = public;
\ir migrations/068_fl_assistant.sql
COMMIT;
SELECT name FROM migration_markers WHERE name = '068_fl_assistant';
```

Verify all five assistant tables exist with RLS enabled, owned by the app
role, and inaccessible to PUBLIC/anon/authenticated. Only then, with separate
production approval, configure the service's OpenAI credential and spending
controls and enable the flag. Do not expose credentials to browser code or logs.
Application rollback clears the feature flag and retains the additive evidence
and receipt tables; do not delete committed transport state.

## Staging verification

Use only `FastAPI-staging` (`0d957be1-8787-41e5-ab38-de71287c30ce`), environment
`f4d219df-2fea-45e8-85de-466b36a86c07`, in project
`2206e070-d160-4528-a4f1-86a587ad88c3`. The environment's Railway label is
`production`, but this isolated service enforces `ENVIRONMENT=staging` and
its own Supabase host/project guard. Never target the production FastAPI service.

After the fresh-local full suite, upload this worktree to that staging service,
wait for deployment SUCCESS, and run `scripts/check_assistant_staging.py`.
The script reads only the protected staging URI, checks isolation before
connecting, creates synthetic fixtures and records receive/make/pack/adjust/found
through real hosted OpenAI function calls. The test operator supplies A5 lot
evidence explicitly. Concurrent retries must return one receipt and one ledger
post per ticket; the temporary actor is deactivated in finally. Save the evidence
in `docs/validation/fl-assistant-part1-staging.json`.

## Recovery and diagnostics

A definitive first-attempt commit 4xx clears `record_started_at`, allowing
Cancel. A changed attempt timestamp, earlier uncertain attempt, or already
committed ticket retains the marker so Cancel cannot claim an uncertain write
was unrecorded. Retry the same draft to establish its receipt in those cases.
Server failures and malformed success responses likewise retain the marker.

Chat leases last two minutes and renew between bounded model calls. After a
worker crash, retry once its lease expires; a replaced worker cannot save the
turn or clear the new worker's lease. OpenAI HTTP failures log only the status
code, never the response body, authorization header or credential.

A3b holds return HTTP 202 with `kind=awaiting_approval`. The assistant stores
that result without a receipt and keeps Cancel unavailable; Check approval
relays the same ticket to FL. Make/pack shortages remain warnings before
recording and visible flagged outcomes afterward. Photo attachments in F1
part 1 are not approval evidence: the model cannot set `attachment_ref`.
