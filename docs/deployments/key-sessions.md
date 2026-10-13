# Pilot key sessions — staging rollout

Use only FastAPI-staging service `0d957be1-8787-41e5-ab38-de71287c30ce`, environment `f4d219df-2fea-45e8-85de-466b36a86c07`, project `2206e070-d160-4528-a4f1-86a587ad88c3`. Its Railway environment label is production but service ENVIRONMENT must equal staging. Never target the production FastAPI service.

1. Run the entire Python suite on a fresh UTF-8 PostgreSQL 17 DB and all Node tests. Run both Assistant browser flag modes and the dashboard session browser harness. Confirm numbering: merged A12/main c195a39 contains 075/076; 077 remains this pilot migration, separate from parked 073/#96. Keep all three schema includes.
2. Apply 077 explicitly as app/table owner through the staging guard in `scripts/check_key_sessions_staging.py --migrate` (session pooler port 5432; transaction-local lock timeout 5s, statement timeout 60s). Rerunning preserves pilot sessions and parked PIN tables. Never apply the full test schema to staging.
3. Set service-only KEY_SESSIONS_ENABLED=1 and PIN_LOGIN_ENABLED=0; keep ASSISTANT_ENABLED=1. Upload this worktree to the explicit staging service and wait for SUCCESS. No merge, production migration or production variable change.
4. Run `scripts/check_key_sessions_staging.py --check`. It creates four clearly synthetic Michael/Arturo/Luz/Miriam identities with ephemeral in-memory keys, exercises the real hosted Assistant for Arturo's 10 lb receive, verifies exactly one post and person attribution, tests logout/idle/revocation and deactivates all temporary actors. Retain the receipt and non-secret evidence in docs/validation/key-sessions-staging.json. Shared-key refusal/rate-limit/flag-off and private migration checks also have local DB tests.

The hosted staging dashboard at `/dashboard/index.html` and `/dashboard/runs.html` uses its Railway origin for pilot auth and existing ledger URL routing. For a separately hosted dashboard, set `window.FL_SESSION_API_BASE` to the intended API origin before loading key-session.js/fetch-timeout.js. No production session destination is configured by this build; enabling the flag on a separate API without configuring the static dashboard does not enable its session controls. This explicit hosting setting is a rollout prerequisite. The Assistant always uses its own origin.

Production remains unchanged. Future production enablement requires separate owner authorization, an explicit 077 application as app/table owner, trusted proxy source review, a configured dashboard API origin, and the designated Claude Code review evidence. PIN_LOGIN_ENABLED stays 0 until Michael decides. Default-off code can run without 077 tables.

Validation and live receipt evidence will be appended after staging acceptance; no reviewer verdict is claimed by the builder.

## Local acceptance — 2026-10-13

Final fresh UTF-8 PostgreSQL 17 database fl_signin_test at localhost:57777: **2,526 Python tests passed**, no failures/skips. **82 Node tests passed**, no failures/skips. Assistant Chrome harness passes KEY sessions on/off; pilot additionally verifies expiry re-ask/token/card clearing, new sign-in, reload recovery and server logout. Dashboard Chrome proves session-only writes, background-read parity, identity, reload, expiry and logout at 1440/390 widths. Screenshots visually inspected. GPT schema remains 30 operations; git diff --check and JS/Python syntax pass.

## Staging deployment — 2026-10-13

Deployment `01c88707-d856-4955-bf7b-181750412821` SUCCESS from tested app commit `bd9fd77`. Migration 077 applied only to guarded staging; service flags KEY_SESSIONS_ENABLED=1, PIN_LOGIN_ENABLED=0 and ASSISTANT_ENABLED=1. Both pilot tables have RLS on and zero policies. Parked PIN actor_sessions has no pilot columns. Production untouched. Hosted Arturo acceptance follows.

## Hosted acceptance — 2026-10-13

Four temporary synthetic identities (Michael/owner, Arturo/floor, Luz/office, Miriam/office) exchanged personal keys successfully with PIN off. Arturo's session drove real hosted OpenAI/FL Assistant prepare and Record for one 10 lb ingredient receive: **RCV-261013-005**, transaction **1000000072**, ticket **83**. Replay returned the same receipt; direct staging readback confirmed exactly one transaction and Arturo's actor id/name. Token hashes only; 180-minute session slide and unchanged 10-minute ticket verified. Sign out blocked another Record with SESSION_EXPIRED; the idle and revoked-key sessions were refused immediately; shared master refused. All temporary actors inactive and open sessions ended. Health 200 and five deployed browser assets match local bytes. Evidence: docs/validation/key-sessions-staging.json.


## Post-A12 rebase verification — 2026-10-13

Rebased feat/key-session onto merged/live A12 `c195a39`, preserving both changelog histories and schema includes 068/075/076/077. Final full fresh isolated UTF-8 PostgreSQL 17 suite: **2,731 Python / 107 Node passed**, zero failures/errors/skips; eight added session-authenticated A9/A12 integration regressions pass. The first run exposed only the historical A12 assertion that no 077 file existed; the corrected assertion permits only this unique `077_key_sessions.sql`, retaining collision checks. Assistant Chrome checks pass with the pilot flag both on/off; dashboard token custody, reload, expiry/re-sign-in and logout pass at 1440/390. Independent code/visual review found no blockers. Evidence: `docs/validation/key-sessions-rebase-tests.json`.

Guarded read-only staging preflight confirms 075/076/077 markers, private pilot RLS tables, no pilot columns in parked PIN actor_sessions, KEY_SESSIONS_ENABLED=1, PIN_LOGIN_ENABLED=0, ASSISTANT_ENABLED=1 and ENVIRONMENT=staging. The old main deployment returned 404 on `/auth/session/config`, so a new explicit staging CLI upload is required. Original hosted acceptance and `key-sessions-staging.json` stay historical; repeat acceptance will use `--output docs/validation/key-sessions-rebase-staging.json`. Production stays untouched; #103 remains draft/unmerged.

The newly merged standalone shift-summary page retains its own personal-key sign-in flow; backend summaries/confirmations accept pilot sessions and are covered by the new regressions. This rebase restores the existing Assistant/dashboard pilot session scope; migrating that separate page to token custody remains followup work.
