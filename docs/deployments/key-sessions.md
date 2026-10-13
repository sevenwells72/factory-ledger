# Pilot key sessions — staging rollout

Use only FastAPI-staging service `0d957be1-8787-41e5-ab38-de71287c30ce`, environment `f4d219df-2fea-45e8-85de-466b36a86c07`, project `2206e070-d160-4528-a4f1-86a587ad88c3`. Its Railway environment label is production but service ENVIRONMENT must equal staging. Never target the production FastAPI service.

1. Run the entire Python suite on a fresh UTF-8 PostgreSQL 17 DB and all Node tests. Run both Assistant browser flag modes and the dashboard session browser harness. Confirm 077 is free: main max 074; open PR reservations 066/#91, 073/#96, 075/#101, 076/#102.
2. Apply 077 explicitly as app/table owner through the staging guard in `scripts/check_key_sessions_staging.py --migrate` (session pooler port 5432; transaction-local lock timeout 5s, statement timeout 60s). Rerunning preserves pilot sessions and parked PIN tables. Never apply the full test schema to staging.
3. Set service-only KEY_SESSIONS_ENABLED=1 and PIN_LOGIN_ENABLED=0; keep ASSISTANT_ENABLED=1. Upload this worktree to the explicit staging service and wait for SUCCESS. No merge, production migration or production variable change.
4. Run `scripts/check_key_sessions_staging.py --check`. It creates four clearly synthetic Michael/Arturo/Luz/Miriam identities with ephemeral in-memory keys, exercises the real hosted Assistant for Arturo's 10 lb receive, verifies exactly one post and person attribution, tests logout/idle/revocation and deactivates all temporary actors. Retain the receipt and non-secret evidence in docs/validation/key-sessions-staging.json. Shared-key refusal/rate-limit/flag-off and private migration checks also have local DB tests.

The hosted staging dashboard at `/dashboard/index.html` and `/dashboard/runs.html` uses its Railway origin for pilot auth and existing ledger URL routing. For a separately hosted dashboard, set `window.FL_SESSION_API_BASE` to the intended API origin before loading key-session.js/fetch-timeout.js. No production session destination is configured by this build; enabling the flag on a separate API without configuring the static dashboard does not enable its session controls. This explicit hosting setting is a rollout prerequisite. The Assistant always uses its own origin.

Production remains unchanged. Future production enablement requires separate owner authorization, an explicit 077 application as app/table owner, trusted proxy source review, a configured dashboard API origin, and the designated Claude Code review evidence. PIN_LOGIN_ENABLED stays 0 until Michael decides. Default-off code can run without 077 tables.

Validation and live receipt evidence will be appended after staging acceptance; no reviewer verdict is claimed by the builder.

## Local acceptance — 2026-10-13

Final fresh UTF-8 PostgreSQL 17 database fl_signin_test at localhost:57777: **2,526 Python tests passed**, no failures/skips. **82 Node tests passed**, no failures/skips. Assistant Chrome harness passes KEY sessions on/off; pilot additionally verifies expiry re-ask/token/card clearing, new sign-in, reload recovery and server logout. Dashboard Chrome proves session-only writes, background-read parity, identity, reload, expiry and logout at 1440/390 widths. Screenshots visually inspected. GPT schema remains 30 operations; git diff --check and JS/Python syntax pass.
