# O-5 — Production sign-in settings plan and read-only checker

Prepared 2026-10-15 (America/New_York). This is a preparation report, not a
deployment approval. This PR adds only this document and
`scripts/check_prod_signin_settings.py`. It changes no application code, tests,
change logs, Railway variables, Netlify configuration, DNS, or database data.

The plan follows **PR #122**, merged as `e41bc2e`: the dashboard obtains its
scoped read key from the API at runtime. Do not introduce Netlify key injection,
put a key in a static asset, or rebuild Netlify merely to rotate a key.

## Where each environment actually lives

| Target passed to the checker | Railway project | Railway environment | Railway service | Public API | Dashboard |
|---|---|---|---|---|---|
| `staging` | `2206e070-d160-4528-a4f1-86a587ad88c3` | `production` | `FastAPI-staging` | `https://fastapi-staging-production-dd7b.up.railway.app` | API `/dashboard/`; `https://deploy-preview-122--cns-factory-ledger.netlify.app` |
| `production` | same project | `production` | `FastAPI` | `https://fastapi-production-b73a.up.railway.app` | `https://cns-factory-ledger.netlify.app`; API `/dashboard/` |

Both services belong to the Railway environment named `production`. Selecting
a Railway environment named `staging` would inspect the wrong target. The
checker always supplies the project, environment, service, and `--no-local`;
it does not depend on a worktree's Railway link. `railway run` runs the checker
locally with the selected service's variables, not inside its live container.

## Inventory and timing

The desired values below are rules or nonsecret configuration. The two current
state columns record only presence, match, or verification status. They never
record key values, hashes, database URLs, or personal identities. The snapshot
was obtained on 2026-10-15 through the read-only checks below; unverified manual
requirements are called out separately.

“Prepare now” means suitable for a separately authorized preparation change;
this task does not make that change. Keep the production activation switches
off until the cutover sequence below.

### Railway and database

| Setting or prerequisite | What it does and correct production rule | Staging now | Production now | Timing |
|---|---|---|---|---|
| `FORWARDED_ALLOW_IPS` and Uvicorn proxy-header handling | Determines whose forwarded client address Uvicorn trusts. Sign-in limits use `request.client.host`. For this plan, `*` is acceptable only after the ingress review below; proxy handling must remain enabled and no start-command option may override the reviewed setting. Unset/loopback-only trust is not proof of distinct staff-client buckets behind Railway. | Present; matches proposed rule; trust review unverified | Absent; does not match proposed rule; trust review unverified | Prepare only after ingress review and staging proof; finish before sign-in activation. |
| `KEY_SESSIONS_ENABLED` | Exact `1` enables personal-key exchange, logout, and dashboard write controls. Unset, `0`, or `true` is off. | Present; matches; deployed flag matches | Absent; default off; deployed flag off | **Cutover day**, after all prerequisites. It exposes more than the sign-in button. |
| `PIN_LOGIN_ENABLED` | Keep unset or `0` (explicit `0` recommended); parked PIN login is not part of this rollout. Current merged session code does not read this variable. | Present; matches off invariant | Absent; matches parked/default-off invariant | May normalize to off in approved preparation. Never enable it for O-5. |
| `DATABASE_URL` | Required, usable database connection to the intended environment. Production must identify the production Supabase project; staging must be isolated. Compare internally, never print the URI. | Present; isolated identity and read-only connection match | Present; production identity and read-only connection match | Verify now. A database move is outside this plan. |
| `ENVIRONMENT` | Must identify the service correctly: `staging` for FastAPI-staging and `production` for FastAPI. Enables database-isolation and production storage/signing checks. | Present; matches service | Present; matches service | Prepare as a coordinated prerequisite bundle; setting production alone can prevent startup if storage/signing dependencies are missing. |
| `PRODUCTION_DATABASE_HOST` | Required by the staging safety guard; must identify the actual production host so staging refuses it. Not a production sign-in switch. | Present; isolation checks match | Absent; not required by production guard | Verify now. |
| `STAGING_DATABASE_HOST`, `STAGING_DATABASE_PROJECT_REF` | Optional additional staging destination guards. If present, must match the selected staging database. Production must never be substituted. | Both present; match database | Both absent; not required by production guard | Verify now; preserve correct guards. |
| `PGHOSTADDR`, `PGSERVICE`, `PGSERVICEFILE` | Must be absent for staging; they could reroute a validated database connection. The checker must also reject unsafe routing overrides before connecting. | All absent; matches | All absent; matches | Remove unintended overrides in approved preparation. |
| `API_KEY` | Required server master key. Keep private, nonempty, distinct from the dashboard and read-only keys. It cannot open a personal session. | Present; separation matches | Present; separation matches | Verify now. Master-key rotation belongs to the coordinated cutover, not O-5 preparation. |
| `DASHBOARD_API_KEY` | Required scoped dashboard key. Must differ from master, read-only, and **all** actor keys, including inactive actors. PR #122 publishes only this scope through `/auth/dashboard-key`. | Present; shared/actor separation and live delivery match | Present; shared/actor separation and live delivery match | Verify now. Rotate only through the coordinated O-7 procedure; no Netlify rebuild. |
| `READONLY_API_KEY` | Optional for personal sign-in; provision it if the coordinated read-only GPT cutover uses that credential. Must differ from master/dashboard and every active actor key. Never substitute it for a personal key. | Present; separation matches | Absent; optional for O-5 sign-in; separate GPT-cutover prerequisite | Can be provisioned privately before cutover. Switch GPT clients on cutover day under their runbook. |
| Migration `077_key_sessions` | Marker, `key_actor_sessions`, and `key_session_attempts` must exist; private tables require RLS, no public/client policy or grants, and access for the application role. Sessions store hashes only. | Present; marker, columns, privacy, role access and notes audit match | Present; marker, columns, privacy, role access and notes audit match | Can be applied additively under the separately approved migration procedure while the flag stays off. Do not replay or roll it back from this checker. |
| Named actors | Intended staff must have active accounts, appropriate `owner`/`floor`/`office` roles and privately delivered personal keys. Shared keys must not collide with actor hashes. Automated structural checks do not prove that the actual roster or key handoff is complete. | Active required roles and key structure match; handoff unverified | Active required roles and key structure match; handoff unverified | Prepare roster and private handoff before cutover. No synthetic identities are created here. |
| `ASSISTANT_ENABLED` | Exact `1` exposes the FL Assistant page and transport; separate from the session switch. Required if staff will use the Assistant at cutover. | Present; matches; deployed page enabled | Absent; default off; deployed page disabled | **Cutover day**, after session and Assistant prerequisites. |
| `OPENAI_API_KEY` | Private backend credential used for Assistant responses/transcription. Must be present; presence does not prove provider authorization, credit, or model access. | Present; provider capability unverified | Absent; does not match Assistant requirement | May provision privately before cutover while Assistant stays off. |
| `ASSISTANT_MODEL` | Optional override; unset selects the code default `gpt-6.1-sol`. If set, it must be a nonempty, approved and tested model. Empty-but-present does not use the default. | Absent; default matches | Absent; default matches | Preserve the tested choice; approve any change separately. |
| `ASSISTANT_TRANSCRIBE_MODEL` | Optional override; unset selects the code default `gpt-transcribe`. If set, it must be a nonempty, approved and tested model. | Absent; default matches | Absent; default matches | Same rule as the response model. |
| Assistant schema (`068_fl_assistant`) | Assistant tables and their migration marker must exist before enabling its routes. Includes sessions, turns, drafts, attachment links, and unmatched words. | Marker and all five tables present | Marker and all five tables present | Verify now; any migration is a separate approved action. |
| `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` | Required production evidence-storage configuration and startup dependencies. URL must identify the intended project; key stays server-side. Evidence storage must be private. | Absent; allowed staging directory backend configured | Both present; URL identity/format and secret presence match | Prepare before setting production mode or enabling staff workflows. |
| `EVIDENCE_STORAGE_DIR` | Must be absent in production; a local evidence directory is forbidden there. A staging volume is allowed only under the existing private/persistent-volume rule. | Present; absolute staging path matches policy; mount unverified | Absent; matches production policy | Verify now; do not silently change storage backends at activation. |
| `PACKING_SLIP_SIGNING_KEY` | Private, nonempty production signing key required at startup. Use a dedicated secret; nonproduction fallback to master is not the production rule. | Present; matches | Present; matches | May provision privately before cutover. Preserve existing valid signed links when planning rotations. |
| `ORDER_LEGACY_CLOSE_CUTOFF` | Adjacent O-4 control: unset is accepted by startup; if set, it must be a valid `YYYY-MM-DD` plant date. Any required cutover date comes from the approved master runbook, not the checker or today's date. | Absent; accepted startup default | Absent; accepted startup default; O-4 date remains a separate cutover action | **Cutover day** under O-4. A malformed value can prevent this same service from starting. |

### Browser, Netlify, and fixed behavior

| Setting or invariant | What it does and correct production rule | Staging now | Production now | Timing |
|---|---|---|---|---|
| `window.FL_SESSION_API_BASE` / `apiBase()` | Browser destination override, **not** a Railway environment setting and not automatically injected by a Netlify variable. Canonical `cns-factory-ledger.netlify.app` selects production; other `*.netlify.app` hosts always select staging, even if an override requests production. Railway-hosted pages select their own origin. | Live preview and Railway routing match; no HTML override | Live canonical and Railway routing match; no HTML override | Existing canonical hosts need no override. Verify now. |
| Custom dashboard domain — FOLLOWUPS P1.28 | Any other non-Netlify host defaults to its own origin. Before activating a custom domain, merge a reviewed `apiBase()` host rule, or provide the correct `window.FL_SESSION_API_BASE` **before** loading runtime scripts, on every relevant page. Then verify every page against the intended API. | Preview host uses staging; matches | Custom domain/aliases absent; canonical host matches | Code/routing first, domain/DNS activation only in a separate approved change. Do not point a new hostname at unchanged assets. |
| Netlify production branch and repository | Published production must use this repository's approved `main` deployment containing PR #122; backend `/auth/dashboard-key` must already work before those assets publish. | Preview isolation matches | Repository, main branch and ready post-O-7 publish match | Verify now; no settings changes in O-5. |
| Netlify build/publish configuration | Repository `netlify.toml` sets base `dashboard` and publish `.`. Static serving requires no key-generating build command. Check the effective site configuration as well as the repository file. | Common site configuration matches | Matches; UI base unset and publish normalizes to dashboard, compatible with repo configuration | Verify now. |
| Netlify environment variables | No `DASHBOARD_API_KEY`, master key, OpenAI key, session flag, or cookie variable is required in Netlify for this implementation. A Netlify variable named `FL_SESSION_API_BASE` has no effect without explicit code injection. Never add secret injection. | Not required; values deliberately not read | Not required; values deliberately not read | No sign-in variable change required. |
| Runtime dashboard-key delivery | `/auth/dashboard-key` must return the selected service's key, compared privately in memory, and `Cache-Control: no-store`. Runtime key cache is 60 seconds; failed refresh retains last good value; first failure fails closed. | Live delivery, private comparison and no-store match | Live delivery, private comparison and no-store match | PR #122 must be live before cutover. Key rotation does not require rebuild/cache clear. |
| Session configuration and UI gate | `/auth/session/config` must be no-store and report enabled with `idle_minutes: 180` at cutover. Dashboard writes require the exact boolean `true`; failure must leave controls unavailable. | Enabled and no-store match | Default off; not cutover-ready; no-store matches | Activation on cutover day; read-only checks now are safe. |
| Idle timeout | **180 minutes**, fixed in `key_sessions.py` and exchange SQL, sliding on user activity. Background reads send `X-FL-Background: 1` and do not slide expiry. No `SESSION_TIMEOUT` variable exists. | Live config matches | Live config matches | No setting to change. |
| Token storage and cookies | Session token in per-origin, per-tab `sessionStorage`; transmitted as `X-API-Key`. Personal key field clears on exchange. No session cookie, cookie domain, SameSite, Secure, HttpOnly, or cookie-secret environment setting is used. Moving origins requires a new sign-in. | Live token-storage asset and no-cookie config match | Live token-storage asset and no-cookie config match | No cookie configuration required. Keep HTTPS. |
| CORS | Current code allows all origins, methods and headers, with `allow_credentials=True`. Header-token requests need preflight permission for `Content-Type`, `X-API-Key`, and `X-FL-Background`; current browser flow does not require cross-origin cookies. There is no CORS environment variable. Verify GET/OPTIONS behavior from each dashboard origin. | GET and session-header preflight match | GET and session-header preflight match | Preserve current code for this rollout. Tightening CORS is a separate reviewed code change. |
| HTTPS and public legal pages | Dashboard/API HTTPS must work. Keep `/legal/*` publicly accessible; do not add a catch-all redirect, password gate, or Netlify Identity requirement. | Preview/API dashboard HTTPS matches; common site legal checks pass | HTTPS, HTTP redirect, public access and both legal pages match | Verify now; no DNS, TLS, or access-control changes here. |
| Other fixed timers | Sign-in/config/key and generic dashboard fetch timeout 15 seconds; Assistant browser default 95 seconds and turn timeout 570 seconds; Assistant upstream call timeout 60 seconds; dashboard tickets 30 minutes and other tickets 10 minutes; legacy actor cache 60 seconds. Session validation checks live actor/key state and bypasses that actor cache. None is an environment timeout knob. | Code invariant | Code invariant | No setting to change. |

The inventory covers the sign-in path and its startup/Assistant dependencies.
`PIN_LOGIN_ENABLED` is a parked operational invariant, not a current code read. Other application settings
are not sign-in switches: `ALLOCATIONS_ENFORCED`, `TRACE_EMIT_ENABLED`,
`FACTORY_READY_REQUIRED`, `FACTORY_TZ`, `SO_HEALTH_CRITICAL_DAYS`,
`SO_HEALTH_WARNING_DAYS`, `ANTHROPIC_API_KEY`, and `EXTRACTION_MODEL` control
inventory/readiness, trace, order display, or document extraction. Preserve
their existing approved values; do not make them part of an O-5 enablement
bundle. The shared ledger plant timezone remains `America/New_York` in code; the separate
order-health display can use its `FACTORY_TZ` override.

## Proxy trust is an explicit prerequisite

`key_sessions.py` hashes the ASGI peer address for persistent sign-in limits:
10 attempts/minute per source, five failures/15 minutes per source, and
50 failures/hour globally. Application code deliberately does not create a
new bucket from arbitrary forwarded headers. Uvicorn's handling of those
headers therefore matters.

`FORWARDED_ALLOW_IPS=*` trusts every connecting peer. It is acceptable only
after the operator records that public HTTP traffic comes through the trusted
Railway edge, client-supplied forwarding headers cannot choose the effective
client address, there is no untrusted direct/private/TCP path to the listener,
and the actual Uvicorn start command has proxy headers enabled without an
overriding allow-list. Review other services in the same project/environment,
not just the public hostname. Do not infer a stable Railway peer CIDR from
one observed address.

The checker's `--proxy-trust-reviewed` option is an operator attestation that
this review is complete. It does not change Railway, turn on proxy handling,
or independently test spoof resistance. Without that attestation, proxy trust
remains FAIL even when the variable matches. With it, an incorrect variable
still fails. If these ingress conditions cannot be established, stop: a
separate proxy/application design is needed before calling sign-in ready.

See [Uvicorn proxy guidance](https://www.uvicorn.org/deployment/) and
[Railway public networking specifications](https://docs.railway.com/networking/public-networking/specs-and-limits).
The repository pins Uvicorn `0.24.0`; do not assume a newer version's behavior
without validating the deployed server.

## Running the checker safely

Use Python with `psycopg2` installed, an authenticated Railway CLI, and the
Netlify access required for site metadata. From this worktree:

```sh
python3 scripts/check_prod_signin_settings.py staging
python3 scripts/check_prod_signin_settings.py production
```

After recording the ingress review, the same checks can acknowledge it:

```sh
python3 scripts/check_prod_signin_settings.py staging --proxy-trust-reviewed
python3 scripts/check_prod_signin_settings.py production --proxy-trust-reviewed
```

Use `--preview-url https://deploy-preview-122--cns-factory-ledger.netlify.app`
to select the preview explicitly; this is also the default preview. Replace
it only with the intended, reviewed Netlify preview when that preview changes.

The checker obtains Railway variables only through commands of this form,
with its own private inspection subcommand as the child process:

```text
railway run --project 2206e070-d160-4528-a4f1-86a587ad88c3 --environment production --service FastAPI-staging --no-local ...
railway run --project 2206e070-d160-4528-a4f1-86a587ad88c3 --environment production --service FastAPI --no-local ...
```

It prints PASS/FAIL by setting, never values. Exit **0** means every implemented
cutover-readiness check passed; missing evidence, unavailable dependencies,
timeouts, bad configuration, and disabled cutover switches fail closed. The
production checker is expected to fail while production remains deliberately
off. It must not turn a missing credential or unavailable platform API into a
pass.

The current model checks accept the two code defaults only, whether unset or
explicit. A different approved/tested model requires a reviewed checker-policy
update; arbitrary nonempty model names do not silently pass. The staging
storage checks accept an absolute configured evidence directory instead of
production Supabase credentials. They do not inspect the remote mount.
`READONLY_API_KEY` is optional for this sign-in gate: if supplied, separation
is checked; its absence is recorded in the inventory and must be addressed
by the separate GPT cutover if that client requires it. The cutoff check
validates startup syntax, not whether the master-runbook date is correct.

Its operations are public GET/OPTIONS, read-only platform metadata, and bounded
database SELECTs inside an explicit **transaction-scoped read-only** transaction.
It imports no `main` application module, starts no app, runs no migration,
changes no settings, and creates no sign-in or actor. It must never set
session-level read-only state on the port-6543 pooler. Connection routing is
validated before any connection and database errors are reduced to status,
not printed with connection strings.

This is configuration verification, not a functional sign-in test. It does
not exchange a personal key, POST a rejected credential, call logout, or send
actor/session-authenticated reads: those can write attempt rows, usage times,
session expiry, or revocation state. A pass does not prove provider billing,
model authorization, staff key possession, actual roster completeness, or an
end-to-end save. Perform the authorized human smoke test separately at cutover.

### Recorded read-only results

Results recorded 2026-10-15 at 08:49 America/New_York, against both live
services and the canonical/PR-122 Netlify hosts. No proxy-review attestation
was supplied, because this task has not verified ingress trust.

| Command | Exit | FAIL labels | All other emitted checks |
|---|---|---|---|
| `python3 scripts/check_prod_signin_settings.py staging` | **1** | `PROXY_TRUST_REVIEWED` | PASS |
| `python3 scripts/check_prod_signin_settings.py production` | **1** | `PROXY_TRUST_REVIEWED`, `FORWARDED_ALLOW_IPS`, `KEY_SESSIONS_ENABLED`, `ASSISTANT_ENABLED`, `OPENAI_API_KEY_PRESENT`, `DEPLOYED_ASSISTANT_FLAG`, `DEPLOYED_SESSION_FLAG` | PASS |

The two deployed-flag failures confirm the corresponding production switches
are still off; they are not additional settings. The inventory above is the
per-setting snapshot. Presence-only supplements for optional variables were
also obtained with explicitly targeted `railway run ... --no-local` children;
no environment dump or secret output was used. Netlify metadata was captured
in memory and reduced to rule matches; no Netlify environment values were read.

Successful checks include both database identities/connections, migration 077,
session columns, RLS/no-client grants, application-role access including RLS
bypass, notes audit constraints, Assistant schema, required active role/key
structure, shared/actor key separation, health, the fixed timeout, no-store,
CORS, runtime key delivery, canonical/preview/same-origin routing, token assets,
Netlify configuration, HTTPS redirect and public legal pages. Production
storage/signing prerequisites already match. Staging's existing directory
backend is allowed and does not require production Supabase credentials.

Local verification used in-memory assertions, without adding/editing tests:
14 assertions covered exit 0 for complete mocked success, refusal without
proxy review, a failed individual setting, malformed/secret-bearing CLI output,
argument redaction, database exception redaction, production/staging identity
and routing-override rejection, local environment scrubbing, and POST refusal.
An independent review also exercised Netlify null defaults, password gating,
stopped/skipped builds, preview disabling, custom domains and bad publish
configuration. Python syntax and the repository secret-literal guard pass. The guard also
passes with current credential values supplied only in memory from each
Railway service; neither values nor hashes are emitted.
Live commands exit 1 intentionally; this report does not claim cutover approval.

**Prepare before cutover:** privately provision the missing production OpenAI
credential; complete the ingress review and staging proof, then set the reviewed
production proxy rule in an approved preparation change. Existing database,
storage, signing, key-delivery and Netlify settings need no O-5 change. Complete
staff key handoff and the separately owned GPT/cutoff controls. **On cutover
day:** enable personal sessions and the Assistant only after the ordered gates
below. There is no timeout, cookie, CORS or Netlify secret variable to add.

## Cutover-day sequence and one-step rollback

Run these steps from the approved O-5 checkout. Production means Railway
project `2206e070-d160-4528-a4f1-86a587ad88c3` → environment `production` →
service `FastAPI`. No command in this document is authorization to mutate it
during preparation. Use the master cutover runbook for staff coordination,
GPT switches, migrations, and the O-4/O-7/A10 changes owned by other lanes.

Before starting, privately record the current Railway deployment ID, current
post-O-7 Netlify deploy ID, and each setting's previous presence/value in the
approved password manager or private change record. Never paste secrets into
a terminal argument, this document, a PR, or a chat. A rollback restores the
recorded state; it must not resurrect a retired exposed key or publish a
pre-O-7 build after key rotation.

| Order | Exact operator action and required evidence | One-step rollback / stop action |
|---|---|---|
| 1 | Announce the approved cutover window and keep staff on the existing workflow. Run both default checker commands above; confirm the expected off-state failures are understood and all prerequisite failures have owners. Record the reviewed deployment IDs and private rollback snapshot. | Cancel the cutover window; continue the existing workflow. No system mutation has occurred. |
| 2 | Verify FastAPI and Netlify run the approved build containing PR #122. Require `/auth/dashboard-key` success with no-store, effective Netlify base/publish settings, correct preview isolation, and canonical-host routing. If a replacement build is necessary, stop for the separately approved backend-first O-7 deployment. No custom-domain/DNS change is part of this cutover. | Stop and retain the published builds; this verification has changed nothing. |
| 3 | Verify the completed preparation bundle while both activation switches stay off: production database identity and schema markers, private session tables, active staff roster/key handoff, private storage, dedicated signing key, and OpenAI configuration. Require recorded acceptance of the separately approved migration/roster/configuration work; stop if anything is missing. The checker never applies that work. | Stop and retain the existing off-state; leave additive schema and audit/history tables in place. |
| 4 | Complete and record the ingress trust review. On **FastAPI → Variables**, verify the reviewed `FORWARDED_ALLOW_IPS` value (`*` only under the conditions above). If preparation has not already set it, set it, deploy, and wait for Success. Verify the actual startup command does not disable or override proxy handling. Repeat checker with `--proxy-trust-reviewed`; dependency checks must pass even though activation flags are still expected to fail. | Restore the previous presence/value of `FORWARDED_ALLOW_IPS` and deploy. |
| 5 | Verify acceptance of the master-runbook cutover controls: read-only GPT credential switch, O-4 legacy-close cutoff, O-7 dashboard rotation, and applicable A10 actor-only write controls. Their owners perform those actions in the master runbook's order. Require their recorded evidence before continuing; O-5 does not rotate credentials or perform these other controls. | Stop and keep personal sign-in off; this verification changes no setting. |
| 6 | On **FastAPI → Variables**, set `KEY_SESSIONS_ENABLED` to exact `1`, deploy, wait for Success, and read `/auth/session/config` without printing any credential. Require enabled and the fixed 180-minute duration. Keep PIN login off. This also opens dashboard write UI. | Set `KEY_SESSIONS_ENABLED=0` on FastAPI and deploy. |
| 7 | On **FastAPI → Variables**, set `ASSISTANT_ENABLED` to exact `1`, deploy, wait for Success, and verify the Assistant page is available. Use the already-provisioned backend credential/model configuration. | Set `ASSISTANT_ENABLED=0` on FastAPI and deploy. |
| 8 | Run `python3 scripts/check_prod_signin_settings.py production --proxy-trust-reviewed`; require every line PASS and exit 0. After recording staging's own ingress review, run `python3 scripts/check_prod_signin_settings.py staging --proxy-trust-reviewed` to confirm preview routing and separation. If either fails, hold release and resolve the specific failure. No Netlify key variable, rebuild, or cache clear is needed. | Restore both `KEY_SESSIONS_ENABLED=0` and `ASSISTANT_ENABLED=0` in one FastAPI variable-change deployment; keep staff on the previous workflow. |
| 9 | In an authorized staff browser on the canonical dashboard, sign in with that person's privately held key; confirm the correct name/role and usable forms. Check reload restoration and sign-out on that device. Check Assistant access with the same person on its own origin; a separate-origin sign-in is expected. Follow the master runbook for any actual ledger save and its evidence. Do not log keys or tokens. | Restore both activation flags to `0` in one FastAPI variable-change deployment; retain all sessions, attempts and business receipts. |
| 10 | Announce that staff may use the approved production links. Record status-only evidence and the successful deployment IDs in the authorized cutover record. Remind staff to sign out before handing over shared devices. | Close the release gate by restoring both activation flags to `0` in one FastAPI deployment; the existing staff fallback workflow resumes. |

Turning the session flag off prevents session resolution and hides new dashboard
write controls after configuration reload. It is **not** a revocation of
personal keys, already-issued session rows, or the legacy shared-key paths.
Do not mistake that UI rollback for A10's authorization controls. An Assistant
rollback is its separate flag. Preserve receipts and audit history throughout.

## Source map

- `key_sessions.py:13–18, 43–80, 85–145`: fixed idle limit, exact activation
  flag, live actor/key validation, background-read rule, rate limits, exchange
  and logout.
- `main.py:159–165, 355–365, 2438–2527, 2989, 3282–3380`: CORS, required
  credentials, startup guards, key-collision checks, actor cache, authorization
  and runtime dashboard-key delivery.
- `dashboard/dashboard-runtime.js:4–15, 42–91`: canonical/preview routing,
  override precedence, runtime key cache, exact-true write gate.
- `dashboard/key-session.js:4–65, 72–106` and `dashboard/fetch-timeout.js:16, 36–51`:
  per-tab storage, token headers, restoration, sign-out, and browser timeout.
- `dashboard/fl-assistant.js:83, 338` and `fl_assistant.py:31–32, 113–129, 285–313, 505–507`: browser/turn timeouts, model defaults,
  credential/transport timeout, feature flag and model override reads.
- `staging_safety.py:5–55`, `attachments.py:31–65`, `ship_tickets.py:74–87, 205–215`:
  database identities/guards, production storage and signing requirements.
- `migrations/068_fl_assistant.sql`, `migrations/077_key_sessions.sql`,
  `permissions.py:23–28`, `write_tickets.py:668`: schema, roles and independent
  ticket lifetimes.
- `netlify.toml`, `railway.json`, `requirements.txt`, `FOLLOWUPS.md` P1.28,
  `docs/contracts/key-sessions.md`, `docs/cutover/O-7-dashboard-key-rotation.md`:
  hosting, deployment watch patterns, pinned server, domain rule and O-7 contract.
