# A11 PIN login and FL sessions — DORMANT delivery

Michael deferred activation on 2026-10-09. PR #96 may be merged **only after review
fixes are approved**, with `PIN_LOGIN_ENABLED` unset or `0`. This work does not
merge, deploy production, change production variables or publish Netlify.
Only the exact value `1` enables A11; having a pepper, PINs or migration 073 does
not activate it. Staging may remain ON for acceptance testing.

## Phase A — later approved merge, flag OFF

1. Leave the current Netlify production publish in place. Hold automatic Netlify
   production publishing for this merge; no frontend rollout is part of phase A.
   Check that the backend flag is unset or `0`. Do not set `PIN_PEPPER` yet.
2. **Before migration 073, connect with the actual backend `DATABASE_URL` role**
   (port 5432, privately supplied; never print the URL). Verify:

   ```sql
   BEGIN TRANSACTION READ ONLY;
   SELECT current_user AS application_role,
          pg_get_userbyid(relowner) AS actors_owner,
          relowner = (SELECT oid FROM pg_roles WHERE rolname=current_user) AS owned_by_app
   FROM pg_class WHERE oid='public.actors'::regclass;
   COMMIT;
   ```

   `owned_by_app` must be true. **Stop if not**; review ownership/RLS permissions
   before proceeding. A separate migrator being owner is not sufficient. 073
   enables RLS on `actors`, which would otherwise break existing actor-key
   lookup. The migration now also refuses a non-owning applying role before
   changing anything. No production role check was performed in this PR.
3. Apply `migrations/073_pin_sessions.sql` as that same app/table owner using
   `psql -v ON_ERROR_STOP=1`, port 5432 and one explicit transaction:
   `BEGIN; SET LOCAL search_path=public; SET LOCAL lock_timeout='5s';` then the
   migration and `COMMIT;`. It is additive/rerunnable; it is never run at startup.
   Retain actor/security-table RLS and revoked public/anon/authenticated grants.
4. Deploy the reviewed backend with the flag still OFF. Nothing else changes:
   the existing browser dashboard key, wildcard CORS policy, anonymous dashboard
   pages, shared-key allowlists, actor keys, owner checks and backdating work as
   main. No PIN re-entry. PIN session/admin APIs and backend admin assets return
   404; their API schema entries are hidden at startup.
5. Smoke the current Netlify pages and actor/shared-key routes without a pepper.
   Do not bootstrap PINs, enable the flag or deploy Netlify in phase A.

For the new dashboard assets, `/auth/config` returns a no-store mode response.
OFF keeps the existing public dashboard credential in the browser and sends it
on exactly the legacy keyed calls; public reads remain public. No actor-key or
session storage is deleted, no idle timers/login UI start, and no page asks for
sign-in. ON returns only the mode (no dashboard credential) and starts the full
A11 session client. A failed config request does not silently downgrade to a
shared credential. A separately hosted PIN admin URL remains hidden and returns
to the dashboard while OFF; backend-hosted PIN admin assets return 404.

## Phase B — future activation, separately authorized by Michael

Activation is deferred, with no scheduled date. Complete both FOLLOWUPS first:
**deliver and test an owner email alert for global PIN lockout**, and **decide
whether read-only dashboard pages should require sign-in**. Current ON browser
behavior requires sign-in for API-backed pages; this is not a settled rollout
policy. F1/A12 integration acceptance also belongs to that future rollout.

1. Set a random server-only `PIN_PEPPER` of at least 32 characters privately.
   Keep `PIN_LOGIN_ENABLED=0`. Never put the pepper/PIN in shell arguments,
   screenshots, logs or the browser. Pepper rotation requires resetting PINs.
2. **Michael bootstraps his own first PIN while the flag is still OFF.** On a
   private interactive shell with the backend DB role and pepper in the process
   environment, run `python scripts/bootstrap_owner_pin.py --actor-id <Michael-id>`.
   Resolve the existing Michael owner record first; do not create/reassign people.
   The command uses non-echoing prompts, refuses a noninteractive echo fallback,
   permits only an existing active owner with no PIN, verifies table ownership,
   serializes uniqueness and writes a management audit row. It cannot reset a
   PIN or turn on login. HTTP PIN routes remain 404 throughout bootstrap.
3. Configure the trusted Railway HTTP proxy launcher below; verify approved
   `DASHBOARD_ORIGINS`. Set **`PIN_LOGIN_ENABLED=1`**, deploy backend, and verify
   Michael's PIN/session, source/global limits and fresh protected-action proof
   privately against the Railway dashboard. Set other people’s PINs via owner
   administration; never publish or select PINs in this document.
4. **Deploy Netlify last**, after backend readiness and the read-only-page policy
   are signed off. Check every dashboard page, login/idle/logout, actor-key
   integrations, same-person ticket resume and protected owner actions.

Rollback: set the flag OFF and redeploy the backend, then reload dashboard tabs
so they read the no-store mode again. Keep migration/security audit tables and
RLS. Do not erase PIN/session evidence or run a destructive down migration.

## Railway proxy configuration — staging only now

`proxy_server.py` explicitly enables uvicorn `proxy_headers=True` and reads
`FORWARDED_ALLOW_IPS` from the environment (local default `127.0.0.1`). On Railway,
its outer adapter accepts the edge's overwritten `X-Real-IP` from a trusted socket
peer and normalizes it to XFF **before** uvicorn rewrites `request.client`.
Client-supplied XFF cannot choose a rate-limit bucket. Missing/malformed edge
identity falls back to the peer. The PIN verifier still hashes only the ASGI
client address and the independent device cookie; it does not parse raw headers.

Railway documents [X-Real-IP as the edge client address](https://docs.railway.com/networking/public-networking/specs-and-limits),
and [Railway staff confirm overwrite and no direct public bypass](https://station.railway.com/questions/need-authoritative-railway-client-ip-p-b7a7b4bd).
[Uvicorn proxy settings](https://www.uvicorn.org/settings/) accept an explicit
trusted-peer list or `*`. Railway does not publish stable proxy IPs: use
`FORWARDED_ALLOW_IPS=*` **only on this HTTP-edge-only Railway service**. Do not
expose it through a TCP proxy or an untrusted private caller. Local/direct
listeners keep a narrow peer allowlist. IP buckets distinguish public client IPs;
devices sharing a NAT still share an IP bucket and each retains its device bucket.

Only **FastAPI-staging** is configured now: service
`0d957be1-8787-41e5-ab38-de71287c30ce`, environment
`f4d219df-2fea-45e8-85de-466b36a86c07`, project
`2206e070-d160-4528-a4f1-86a587ad88c3`; verify `ENVIRONMENT=staging`, service name and
DB project `jygmyvxnxdjiiilhxseq` before every action. The platform environment is
named `production`; that label does not change the pinned staging service scope.
Generate its start command with `python scripts/staging_start_command.py`; it
embeds the database guard and proxy launcher and works with an older snapshot.
Use `PIN_LOGIN_ENABLED=1` and `FORWARDED_ALLOW_IPS=*` for staging acceptance.
No production variables, start command, Railway config file or Netlify settings
are changed by this PR. At future production activation, set the same reviewed
proxy environment and start with `python -m proxy_server` **on that separately
approved service only**. Do not change shared `railway.json` to force this now.

## Identity and security contract — flag ON only

- `POST /auth/session {pin}` identifies a person without a name picker. PINs are
  four ASCII digits; all-same digits, ascending/descending runs (including wrap),
  and repeated pairs are rejected. HMAC-SHA256 with a server-only `PIN_PEPPER`
  supports a single indexed lookup and uniqueness. PINs are never returned.
- A source gets a 15-minute lock after five failures in 15 minutes. IP and device
  are independent buckets, so replacing a cookie or changing IP alone cannot
  escape a lock. Fifty failures in an hour across all sources locks PIN checks
  for one hour. Login, current-PIN changes, and owner re-entry share these limits.
  A lock is checked before PIN lookup, even for a correct PIN. Unknown and locked
  PINs receive the same generic 401; `Retry-After` reports a lock.
- All workers share Postgres counters and an atomic check/count lock. Already
  locked requests only append a failed attempt and refuse; they do not serialize
  actor lookups or extend lock time. `pin_attempts` contains hashed sources,
  result, time, purpose and successful actor id, never the submitted PIN.
- `PIN_GLOBAL_LOCKOUT` is a credential-free ERROR event and the admin page shows
  the global lock/failure count. **Email delivery is not configured by this PR**;
  the event must be routed to the owner's alert channel before a production pilot.
- FL issues opaque `fls_` tokens (32 random bytes); only SHA256 token hashes are
  stored in `actor_sessions`. Header `X-API-Key` carries the token. The browser
  stores only this short-lived credential in `sessionStorage`; PINs and personal
  sign-in keys are cleared from password inputs and never persisted.
- Ten minutes idle ends a session. Active requests slide expiry; passive browser
  polling uses `X-FL-Background: 1` and cannot extend it. The browser also clears
  its token after ten minutes without human input, including a sleeping tab.
  Session auth checks current actor activity/role on every request, without the
  actor-key cache. Logout and PIN reset revoke sessions immediately.
- `/auth/whoami` reports `key_kind=session`; `permissions.py` accepts it as a
  named actor. Persisted tickets continue to use `key_kind=actor` because they
  belong to the person, not a credential. Re-login as the same actor can commit
  the same ticket; another person's session cannot commit or replay it.
- Owner step-up uses **one request's** `X-FL-Owner-PIN`, not an elevated session
  or reusable proof token. Approve/reject held corrections, supported write-off/
  waiver resolutions, and named-actor >14-day backdating require it. A3b's
  unsupported shortage resolutions remain unsupported; A11 does not add new
  ways to close a shortage. Wrong-person PINs fail under the same rate limits.
- A12 uses `require_owner_pin(api, request, purpose='kosher_attestation',
  allow_other_owner=True)` inside its ticket-attestation route, then stores the
  returned owner on that exact ticket in its transaction. This verifies Michael
  on Arturo's device without switching the signed-in person. A12 owns the
  attestation endpoint/table and the per-batch blocker; they are not invented here.
- Existing server-held actor keys and A2 legacy allowlists remain; shared keys
  gain no session/PIN administration rights. A2's grandfathered master-key
  backfill route stays as specified until A10 closes legacy direct writes.

## Staging people and first PINs

The previous staging bootstrap created Michael / Arturo / Luz / Miriam without
PINs and placed Michael's initial personal key in a private mode-0600 file. Do
not rerun it or overwrite that file. Staging may use its ON-only personal-key
bootstrap screen at [PIN administration](https://fastapi-staging-production-dd7b.up.railway.app/dashboard/pin-management.html).
Michael sets his own PIN there, signs in again, then sets other people’s unique
PINs. This staging-only convenience does not change the production sequence:
production uses the private offline first-owner bootstrap before enabling A11.

## F1 PR #90 handoff — no edits to feat/fl-assistant

At future activation, rebase F1 onto A11 after review. Honor `FLSession.ready` /
`PIN_LOGIN_ENABLED`; keep its existing key behavior while dormant and only require
a session after explicit opt-in. Its backend already uses `request_actor()` and
relays `X-API-Key`, so sessions pass the same actor/ticket checks. Required changes:

1. Load `/dashboard/fetch-timeout.js`, `/dashboard/session.css`, and
   `/dashboard/session.js` before `fl-assistant.js`. Replace the actor-key form
   with `await FLSession.requireSession()`; use its returned actor for the header
   and every draft's **Recording as …** label. The optional office/owner personal
   key exchange is already inside the shared sign-in dialog.
2. Route F1's `api()` through `FLSession.fetch(url, options)` with its existing
   JSON/FormData bodies, timeout and AbortSignal. Remove the `key` variable and
   direct `fetch` credential assembly. No browser actor key remains. The common
   client understands both FL errors and F1's nested `result.detail` errors.
3. Sign-out must await `FLSession.signOut()`, stop the mic, and clear in-memory
   chat/draft state. On `fl-session-change`, bind chat storage to the new actor;
   never display or submit another person's old draft. Chat IDs/preferences can
   remain in localStorage; they are not authentication credentials.
4. Preserve the exact draft/ticket, acknowledgements and confirmations after
   `SESSION_EXPIRED`. The shared client re-asks the PIN and retries only an
   explicit expiry response, only for the same actor. It never auto-retries a
   timeout, disconnect or uncertain post. F1's existing `record_started_at`
   handling already clears definitive first-attempt 4xx responses without
   clearing uncertainty from an earlier unknown outcome; preserve it.
5. For the explicit protected write/attest call only, F1 `relay()` must forward
   `X-FL-Owner-PIN` and the actual original ASGI client/device cookie context
   to FL. Do not forward step-up headers to reads, model calls, transcripts,
   tools' saved arguments, or logs. Its present hard-coded loopback client would
   otherwise collapse distinct devices onto one IP bucket. PIN verification
   must stay in FL, never in model prompts. A12 binds its attestation to one batch.
6. Add F1 acceptance cases for idle during a draft, same-person resume, wrong
   person after timeout, fresh PIN at protected record, and no credential storage.
   `permissions.py` route additions and A11's `pin_sessions.AUTH_ROUTES` must both
   survive the rebase. A7/A6 backdated commit paths must call the same owner-PIN
   verifier before their eventual post (main's direct validator already does).

## Review-fix validation (2026-10-09)

- Full release suite on a **fresh disposable PostgreSQL 17 database** at local
  port 57496: **2,280 Python tests passed, zero failures/skips**. Default OFF;
  the A11 security suite opts in explicitly. Existing actor/A2/A3b business
  regressions no longer seed PINs or attach step-up headers. The RLS fixture
  retains the documented requirement that the app owns `actors`.
- **69 JavaScript tests passed**. **93 browser checks passed** against current
  main `ce7bb57` across **all nine pre-existing HTML pages**, at 390/1440 px:
  identical rendered text and API requests/credential use; no sign-in UI or
  credential-storage deletion while dormant. All API/CDN calls used local
  fixtures; zero production requests. Config failure refuses silent downgrade.
- **22 enabled-session browser checks passed**, including login, protected
  owner actions, expiry retry/person binding, and admin handover.
- New DB/proxy cases cover disabled PIN surfaces with malformed/missing input,
  unchanged CORS preflights and browser/actor keys, prefixed actor-key priority,
  legacy owner actions/backdated commits with no PIN, auth without any PIN
  tables/columns, migration nonowner refusal, private pre-activation bootstrap
  and revocation, separate forwarded IP buckets and spoof-resistant Railway
  header normalization. `git diff --check` passes.
- Reproduce: load `tests/schema/schema.sql` into a fresh local PostgreSQL 17 DB,
  run Python 3.12 `pytest` with only its `TEST_DATABASE_URL`; run
  `node --test tests/test_*.js`, `npm run test:pin-dormant` (requires the baseline
  Git object `ce7bb57`) and `npm run test:pin-sessions` after `npm install` and
  Playwright Chromium setup. Historical PDF tests use ReportLab 4.4.9 plus
  pdfplumber, matching the committed historical fixture.

Staging review-fix deployment **`a38c1cbc-ac8e-4437-acc7-117d18cb7f00`** is
SUCCESS with the flag ON and trusted proxy launcher. The actual backend role
owns `actors` and all four security tables; all have RLS enabled. Live asset
hashes match the release-tested snapshot. Two failed requests from distinct
synthetic devices, each supplying forged XFF and X-Real-IP, were recorded under
the independently verified real public client IP. Security evidence is retained;
no real PIN was entered. See [the redacted staging receipt](a11-dormant-staging-receipt.json).
Production was not queried, configured, migrated or deployed; Netlify was not
published and PR #96 was not merged.

## Historical pre-review validation and staging rollout evidence

- Fresh disposable local PostgreSQL: **2,232 Python tests passed**, zero skips;
  the final role/key-precedence/shared-key follow-up passed all **115 focused
  authentication and permission tests**.
- **69 JavaScript tests passed**; **22 browser checks passed** at 390 and 1440 px.
- Real Postgres tests cover obvious PINs, uniqueness, independent IP/device locks,
  distributed and concurrent global limits, 10,000-candidate sweep, idle/passive
  reads, immediate revocation, bootstrap, own change, hashed storage, backdated
  commit step-up, fresh owner proof, wrong-person commit/replay, redaction, and RLS.
- `scripts/check_pin_sessions_staging.py` runs actual HTTP handlers and concurrent
  committed transactions in a private UUID namespace on real staging Postgres.
  The namespace is removed afterward; real people, PINs and live lockouts are
  untouched. `--apply-migration` separately applies public migration 073.
- Staging receipt: `docs/deployments/a11-staging-receipt.json`: **10,000 HTTP
  candidates, zero sign-ins, 9,995 blocked before PIN lookup**, distributed global
  ceiling after 50 failures; 240.7 seconds. The tested table definitions are
  unchanged by the 072→073 rename. Migration 073 was then idempotently applied;
  the historical `072_pin_sessions` marker remains as staging apply evidence.
- Deployed HTTP smoke: `docs/deployments/a11-live-staging-receipt.json`, including
  Michael's real bootstrap sign-in and all four named people with PINs unset.
  Temporary smoke accounts were deactivated and their PIN hashes cleared.
- Browser evidence: `docs/validation/a11-login-mobile.png` and
  `docs/validation/a11-pin-admin-desktop.png` (synthetic people, empty PIN fields).

Rollback application code first and retain all security evidence tables and
actor RLS. There is no destructive down migration. No production deployment,
production queries, merges, or changes to other worktrees are part of A11.


Staging rollout on 2026-10-09: server-only pepper configured without output;
public migration applied and then renumbered to 073; four people initialized
without PINs; protected owner sign-in file created with mode 0600. Backend
snapshot `847fb6c` is live at successful staging deployment
`046d335b-ea9c-47f3-8a8a-36b401ad6dc3`. This includes the final browser handover,
actor-key precedence and shared-key exclusion changes, plus migration 073.
The 10,000-candidate acceptance receipt predates those narrow follow-ups; the
final deployed HTTP smoke is in `a11-live-staging-receipt.json`; deployment,
public migration/RLS checks and matching static-asset hashes are in
`a11-staging-rollout.json`.
