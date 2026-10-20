# PR 142 review fixes — local validation

All three review items are fixed on `fix/pilot-ux-round`. PR #142 stays Draft; nothing was merged or deployed.

- Make and Pack have no photo control on the Record form or shared draft card. The independent Assistant draft renderer already has no upload control. Receive keeps BOL/lot-tag photos; Adjust and Found keep correction photos. Real local receipts confirm all three still save their evidence.
- The form and draft share all eight canonical correction reason codes and labels, including `wrong_lot` / **Wrong lot used** / **Lote equivocado**. Historical `wrong_lot_used` values also remain translated. EN/ES changes relabel an open form or draft in place.
- Office uses **Cases = Cajones**, **Boxes = Cajas**, and singular **cajón**. Order lines, added lines, expected deliveries, pouch conversions, extracted-document review, drafts and receipts agree. Quantities and stored unit codes are unchanged.

| Final check | Result |
|---|---|
| Full Python, sessions off | 6,916 passed; 0 failures, errors or skips |
| Full Python, sessions on | 6,916 passed; 0 failures, errors or skips |
| Full Node | 541 passed; 0 failures or skips |
| Dashboard runner | Both session modes passed |
| Assistant runner | Both session modes passed, including customer/order flows |
| Existing pilot screens | 60 passed across EN/ES, 390/1440 and both session modes |
| Full Luz/Arturo actual-page pilot walkthroughs | 96 passed; 44 local receipts, 20 explicit physical-lot confirmations, 16 saved photos |
| Added actual-page review checks | 34 passed at 390px; every reason, Office units and live language changes; no records |
| Hosting/navigation | 272 checks in 16 scenarios passed |

The final suites used fresh, disposable local PostgreSQL 17 databases, a clean environment and synthetic credentials. The browser bridge hard-refuses OpenAI and the browser blocks non-local traffic. There were zero OpenAI calls, staging/production database connections or Railway operations. The server, database pool, Assistant turn-handling code and migrations are unchanged. Every task database was dropped and the temporary clusters/servers were stopped; see `verification.json` for details.

The first Python attempt used a synthetic master key different from the older tests' fixed fixture key. It was interrupted, its output is retained under the raw output directory, and both full suites were restarted against fresh databases with the correct fixture. Two original Node assertions were updated to check the intended bilingual span and canonical Spanish label. Early review-runner attempts used an invalid correction sign or counted repeated fields as one; final assertions check every rendered reason cell. Final counts above exclude those setup attempts.

## Requested 390px draft screenshots

All six images were visually inspected and show the full draft through the Record button.

| Draft | English | Spanish |
|---|---|---|
| Make | [EN](screenshots/make-draft-en-390.png) | [ES](screenshots/make-draft-es-390.png) |
| Pack | [EN](screenshots/pack-draft-en-390.png) | [ES](screenshots/pack-draft-es-390.png) |
| Receive | [EN](screenshots/receive-draft-en-390.png) | [ES](screenshots/receive-draft-es-390.png) |

Machine counts, source hashes and cleanup: [verification.json](verification.json). Per-check browser results and final stdout logs are alongside this summary. Raw run outputs (including initial attempts and JUnit XML) remain at `/private/tmp/fl_pr142_validation`.

## Replay

Use the project Python 3.12 test environment and a fresh local PostgreSQL 17 cluster/database with `pg_trgm` and `tests/schema/schema.sql`. Run the complete `pytest` suite with `KEY_SESSIONS_ENABLED=0`, then `1`, using the fixed test master key `test-api-key`, synthetic dashboard key `test-dashboard-key`, local `TEST_DATABASE_URL`/`DATABASE_URL`, and no inherited model or hosted-service credentials. These runs used independent clusters to avoid cross-test connection-observer interference.

Run `node --test tests/test_*.js`, `tests/visual/run-key-sessions.mjs`, `tests/visual/run-fl-assistant.mjs` with `PILOT_SESSIONS=0` then `1`, `tests/visual/run-pilot-screens.mjs`, `tests/visual/run-pilot-ux-hosting.mjs`, `tests/visual/run-pilot-ux-round.mjs` and `tests/visual/run-pilot-ux-review.mjs`. Browser tooling accepts `PLAYWRIGHT_MODULE`, `PLAYWRIGHT_CHANNEL=chrome`, `PILOT_UX_PYTHON` and local `PILOT_UX_PG_PORT` overrides. The actual-page runners own and drop their `fl_pilotux_` databases; do not point them at hosted services.
