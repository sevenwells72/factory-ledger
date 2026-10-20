# Pilot UX round — local validation

All 17 requested items pass. No OpenAI calls, hosted data or setting changes, deployments, or migration changes were made by validation. Runtime pool/connection code and Assistant turn handling were not changed.

| Check | Final result |
|---|---|
| Full Python, fresh PG17, sessions off | 6,916 passed; 0 failures/errors/skips |
| Full Python, fresh PG17, sessions on | 6,916 passed; 0 failures/errors/skips |
| Node | 488 passed; 0 failures/skips |
| Actual local Record/Office pages | 96 checks; 8 fresh DB scenarios; 1440/390px; English/Spanish; sessions off/on |
| Actual local walkthrough Records | 44 receipts; 20 explicit physical-lot confirmations; 8 attached photos |
| Root/prefix hosting and navigation | 272 grouped checks; 16 scenarios; all 12 shared-header pages; 48 screenshots |
| Existing pilot-screen browser runner | 60 passed; both session modes; 1440/390px |
| Existing dashboard browser runner | Both session modes passed |
| Existing Assistant browser runner | Both session modes passed, including customer flow |

The sessions-off flag retains its existing write-UI cutover: navigation/style and zero-write gating are checked in both languages and widths. Complete named-person Record workflows run with sessions enabled. The actual-page bridge uses real authentication, FastAPI handlers, PostgreSQL and private local evidence storage. Assistant is actually disabled; its safe handoff links are tested without a model or turn call.

| Item | Done | Evidence |
|---|---|---|
| 1 | Yes | Button styling and phone layout on all 12 shared-header pages, both hosting layouts |
| 2 | Yes | Three visible Office start buttons; New customer opens without searching |
| 3 | Yes | Existing matches precede divider and secondary Add-new wording |
| 4 | Yes | Exact Add 'Setton Farms' → form → duplicate refusal → Use existing customer, all four active UI variants |
| 5 | Yes | Automatic debounced customer/product/supplier/lot matches, 1-character guard, Enter/Search retained |
| 6 | Yes | Regular Classic first; two separate EXTRA-KOSHER tags |
| 7 | Yes | Cases 10 × 25 lb = 250 lb and direct-lb path covered by browser/unit checks |
| 8 | Yes | Office and Record links use independent receipt Today; recorded/voided entries render while Assistant is off |
| 9 | Yes | Cancel clears drafts and returns empty forms with no commit |
| 10 | Yes | Receive lot-tag (English) and BOL (Spanish), plus Ship BOL: form → draft → receipt, real stored photos |
| 11 | Yes | Oats '-001' ambiguity shows both full codes; user chooses; every physical tag remains required; compact row per required ingredient lot |
| 12 | Yes | Zero Coconut warning before Record; real Make posts and creates its SHORTAGE flag |
| 13 | Yes | Classic #9 finds batch; newest just-made batch is first; Pack 4 × 25 lb still requires its tag |
| 14 | Yes | Oats product-name lot search for Adjust and Found, full-code selection; existing identity safety guard preserved |
| 15 | Yes | Floor/office backfill hidden; owner-only behavior retained in focused tests |
| 16 | Yes | Record home returns to Assistant/home |
| 17 | Yes | Truly disabled Assistant shows working Office/Record handoff links in both hosting layouts |

Luz's four practices use Setton Farms, 4 Fruit Nut cases + 2 Classic cases, No PO, ship by 2026-10-23, PRACTICE; exact Panaderia Practica Luz creation followed by 3 Sprinkles cases; expected Sunflower delivery on 2026-10-26 at 10 × 25 lb; typed receipt-number lookup, Today and Exceptions. Arturo's practices record Receive 4 × 50 lb with SUN-PRACTICE-A; one Classic #9 Make; Pack 4 × 25 lb from that new batch; manually select and Ship 25 lb with a BOL; Found 10 lb; Adjust -5 lb; explicit same-day Void; Today/receipt review. All run at both widths and languages. Synthetic catalog/lot IDs differ from staging. The synthetic Classic batch makes 200 lb from 150 lb Oats, 40 lb Sunflower and 10 lb Coconut so the four-case Pack can use the just-made batch. No lot is auto-confirmed.

Full screenshots/results and compressed JUnit: `/Users/cns/dev/fl-test-runs/pilot-ux-round/`. The 20 representative checked-in screenshots are in `screenshots/`; machine results are `verification.json`, `actual-browser-results.json` and `hosting-results.json`. Final stdout logs are in `logs/`.

## Commands and isolation

Python uses the pinned Python 3.12 environment with `OPENAI_API_KEY` and inherited `DATABASE_URL` unset, `TEST_DB_PREFIX=fl_pilotux_…`, and `python -m pytest --junitxml=…`. Sessions-on DB: `postgresql://127.0.0.1:5432/fl_pilotux_py_on`. Final sessions-off DB: `postgresql://127.0.0.1:55472/fl_pilotux_py_off_isolated` in disposable `/tmp/fl_pilotux_pg17_20261020`, PG17 UTF-8, no locale. Both load `tests/schema/schema.sql` and pg_trgm fresh. The temporary server is stopped. All fl_pilotux_ databases were dropped; all runner servers stopped.

The actual browser command is `PILOT_UX_PYTHON=<pinned Python> PLAYWRIGHT_MODULE=<Playwright module> node tests/visual/run-pilot-ux-round.mjs`. It owns eight fresh fl_pilotux_ databases and drops each in a finally block. The bridge applies the existing 083 customer migration only to its fresh synthetic DB; it never executes hosted startup hooks. Existing runners: `run-key-sessions.mjs`, `run-fl-assistant.mjs` with PILOT_SESSIONS=0 and 1, and `run-pilot-screens.mjs`. Hosting runner: `run-pilot-ux-hosting.mjs`.

## Verification limits and earlier attempts

Browser coverage is headless Chromium at CSS widths 390/1440; it is not a physical phone, camera or microphone test. Ordinary published-root and /dashboard/ layouts require every asset to succeed. The legacy /dash/ alias allows only its seven known speculative initial asset 404s and still requires the final correct /dashboard/ assets to return 200; navigation and UI must work. This is recorded in `hosting-results.json`.

Earlier shared-cluster attempts are retained under the full-output `logs/`: one run overlapped a main.py line change and invalidated source-inspection offsets; new lot-read test seeds were corrected to valid uppercase codes; simultaneous pool tests shared an application_name observer across databases. A later shared-cluster session-off run failed one pre-existing shipping-question case; its cause was not established. The final isolated session-off suite passed all 6,916 tests without pool or Assistant-turn code changes. Browser development exposed the Receive evidence field/kind mismatch, now fixed through the existing accepted evidence contract, plus test-selector/fixture races corrected before the final 96/96 exact-practice run.
