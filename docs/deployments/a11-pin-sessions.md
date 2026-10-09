# A11 PIN login and FL sessions

Builder: Codex. Reviewer: Claude Code. Draft PR #96; **do not merge or deploy to
production**. Base `origin/main` at `ce7bb57`. Migration **073** is the next free number. Initial checks reserved 072; the final
open-PR check found A7 #93 had added `072_order_ticket_review_fixes.sql` during
this build, so A11 moved to 073. 066 #91 remains held; 067/072 belong to A7,
068 #90 to F1, and 069–071 are on main.

## Identity and security contract

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

## First PINs on staging

The staging service is **FastAPI-staging**, service
`0d957be1-8787-41e5-ab38-de71287c30ce`, environment
`f4d219df-2fea-45e8-85de-466b36a86c07` in project
`2206e070-d160-4528-a4f1-86a587ad88c3`. Its platform environment name is
`production`, but its application `ENVIRONMENT=staging` and database project
`jygmyvxnxdjiiilhxseq` are separately checked. Never target the production service.

1. Apply `migrations/073_pin_sessions.sql` to **staging** as the application/table
   owner, using port 5432, `ON_ERROR_STOP`, an explicit transaction and
   `SET LOCAL search_path=public; SET LOCAL lock_timeout='5s'`. It is rerunnable,
   additive, does not alter ledger rows, and does not auto-run at startup.
   The application must own `actors` and the new security tables (or have an
   explicitly reviewed RLS policy); public/anonymous/authenticated access is
   revoked. Keep actor RLS enabled.
2. Set a random server-only `PIN_PEPPER` of at least 32 characters on
   **FastAPI-staging**, before deploying A11. Do not print it. Rotating it makes
   all PIN hashes unresolvable; plan a full PIN reset when rotating it.
3. Run `python scripts/bootstrap_pin_staging.py` once. It verifies the exact
   staging database, creates **Michael / Arturo / Luz / Miriam with no PINs**, and
   saves Michael's random initial sign-in key, mode 0600, at
   `~/Documents/fl-secrets/staging-pin-owner-key.txt`. It never prints the key,
   overwrites an existing file, reactivates people or resets existing credentials.
4. Open [staging PIN administration](https://fastapi-staging-production-dd7b.up.railway.app/dashboard/pin-management.html).
   Expand **Office / owner sign-in**, paste the key from that private file, and
   choose **Use personal sign-in**. Set **Michael's first PIN**. This one bootstrap
   operation is allowed only for an owner key session whose own PIN is still unset.
5. Sign in with Michael's new PIN. In **Set or reset a person's PIN**, select
   Arturo, Luz or Miriam and set their unique PIN; re-enter Michael's PIN when
   prompted. Each person can then use **Change my PIN** with their current PIN.
   Reset/change signs that person out everywhere. The login screen never has a
   person picker; the name list exists only inside owner administration.

Dashboard writes use the shared session client on every existing dashboard page;
[staging dashboard](https://fastapi-staging-production-dd7b.up.railway.app/dashboard/index.html).
For a separately hosted dashboard set `DASHBOARD_ORIGINS` to its exact approved
origins (comma separated). The default is the existing Netlify dashboard. The
Railway-served staging dashboard uses the same staging origin. No production
Netlify deployment is performed by this PR. Uvicorn must trust only the actual
platform proxy when using forwarded client addresses; the code never trusts an
arbitrary `X-Forwarded-For` header itself.

## F1 PR #90 handoff — no edits to feat/fl-assistant

Rebase F1 onto A11 after review. Its backend already uses `request_actor()` and
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

## Validation and rollout evidence

- Fresh disposable local PostgreSQL: **2,232 Python tests passed**, zero skips;
  the final locked-role follow-up passed all **16 focused security tests**.
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
snapshot `1de4053` was verified live at deployment
`95d4308b-e6a0-40e0-8f8b-eed90714de92`. Final browser handover changes and the
073 filename/marker are included in the final deployment recorded in the PR.
