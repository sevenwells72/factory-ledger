# Full local validation — compact Make draft

All **26 required functional runner invocations passed**. Validation used local fixtures or disposable PostgreSQL 17 databases, with no hosted database access and **zero OpenAI calls**. Exact commands, environment overrides, every attempt, exclusions, and evidence paths are in [verification.json](verification.json).

## Final results

| Check | Result |
|---|---|
| Full Python, sessions off | **6,935 passed**, 0 failed, 0 skipped; 189.62 seconds |
| Full Python, sessions on | **6,935 passed**, 0 failed, 0 skipped; 168.82 seconds |
| Full Node | **577 passed**, 0 failed, 0 skipped |
| Pilot screens | **60 passed**; 390/1440px, sessions off/on |
| Actual API pilot walkthrough | **96 passed**, all 8 planned scenarios complete, 44 local receipts |
| Pilot UX review | **34 passed** |
| Photo/language | **13 passed**; 2 local receipts, 6 photos |
| Hosting layouts | **272 checks passed** across 16 scenarios |
| Assistant mobile, documented WebKit mode | **144 passed**; 375/390/393/430px, English/Spanish, sessions off/on |
| Assistant phone layout, Chromium and WebKit | **288 passed** |
| Assistant Round A | **30 passed per session mode** |
| Other current browser runners | Intake, dashboard key/retry, session/shift, Assistant off/on, final-floor, key sessions, Sales Order layout/interactions/detail/exit, production trace, run types and Runs interactions all exited 0 |

The focused actual-page Make acceptance run is separate: [browser-results.json](browser-results.json). Its requested before/after screenshots remain in [screenshots/](screenshots/).

Both full Python runs used independently fresh databases (`fl_makedraft_suite_off` and `fl_makedraft_suite_on`) on a new loopback-only PG17.10 cluster at `127.0.0.1:55473`. The interpreter was `/Users/cns/Documents/factory-ledger/.venv-test/bin/python` (Python 3.12.14, Pydantic 2.14.1, FastAPI 0.104.1, httpx 0.27.2). Node was 24.16.0 and Playwright 1.63.0. Each Python run reported 15,157 warnings.

The test processes inherited an explicit environment without hosted/model credentials. Python socket and Playwright/fetch guards refused external connections. The guard intercepted 54 optional cdnjs asset attempts; no model or service host attempts were recorded. The three owned suite databases were dropped and the isolated cluster stopped. The temporary checkout dependency symlink was removed. Bootstrap and cleanup commands are preserved under [logs/](logs/).

## Complete broad visual audit — diagnostic findings

The complete existing `npm run test:visual -- --concurrency 4` audit finished all **388 captures** across 92 inventory screens in 585.04 seconds. It retained its existing thresholds and reported:

| Status | Current cells | Committed historical snapshot |
|---|---:|---:|
| PASS | 3,229 | 3,251 |
| FAIL | **402** | 431 |
| WARN | **63** | 58 |
| N/A | 1,350 | 1,304 |
| ERROR | **0** | 0 |

The diagnostic run recorded the unmatched fixture endpoint `/auth/session/config`; it remained intercepted locally.

This audit is not a zero-findings functional gate. Its September 11 historical report is not an untouched-HEAD rendered baseline: dates, timing, calendar state and intervening changes can change cells. The comparison records 165 changed cells without attributing them to this task. Its `PAGE_URLS` cover index, sankey, process-flow, traceability and scheduler; it does **not** cover current Record Make. See the [complete matrix](logs/visual-audit/06-browser-check.md), [raw measurements](logs/visual-audit/screenshots/results.json.gz), and [historical comparison](logs/visual-audit/historical-comparison.json).

## Optional Chromium mobile check — existing failure retained

An extra Chromium expansion of the Assistant mobile runner timed out at line 140 waiting for “Upload failed — tap to retry” after its simulated upload socket dropout. The same failure reproduced using the untouched committed runner and dashboard assets from `27a2518655da43ae31860a0bcf8c1f73db114422`. The runner's documented default WebKit suite passed all 144 checks. The optional Chromium case remains unresolved; no Assistant runtime was changed. Both [current](logs/assistant-mobile.log) and [untouched-HEAD baseline](logs/assistant-mobile-chromium-baseline.log) failure logs are retained.

## Initial attempts and exclusions

Raw initial failures remain visible in `verification.json` and the logs:

- Fresh schema bootstrap exposed the dump's empty search path; selecting `public` before migration 085 repaired it.
- The initial full Python run found the schema-tail allowlist missing 085. The guard was updated without removing its existing checks.
- Reusing that database for the initial second mode produced receipt/actor count failures from committed fixtures. Independently fresh final databases eliminated these failures, and both complete runs passed.
- Initial Node and pilot-screen attempts overlapped the new lot pagination assertions. The complete final reruns passed 577/577 and 60/60 respectively.

Excluded workflows are listed individually with reasons in `verification.json`: hosted/staging deployment checks, real-model scripts that read secrets or invoke a model, the separately served deployment key checker, historical Assistant page runners hardcoding port 5432, and superseded D12/Office fixture-server runners requiring fixed-port fixture files. Current pilot/Assistant local suites and the focused actual-API Make walkthrough provide the applicable coverage.

## Screenshot archive

All **1,114 broad-suite PNGs** (128,274,692 bytes), including optional failure/baseline evidence, are preserved locally at `/private/tmp/fl-makedraft-full-suite-screenshots`. The [archive manifest](logs/screenshot-archive.json) maps original paths to archived paths and records SHA-256 checksums. Generated reports and result JSON retain their original capture paths; use this manifest to locate their images. No broad-suite PNG remains under `logs/`; the dedicated requested Make screenshots are unchanged.

Two delivery log copies had trailing whitespace removed for the Git diff check. Their original stdout bytes remain in `/private/tmp/fl-makedraft-raw-stdout`; the [stdout archive manifest](logs/raw-stdout-archive.json) records both checksums. Every diagnostic is retained.

The broad audit’s 93,318-line raw measurement JSON is delivered losslessly compressed to keep the PR review focused. The [raw-results manifest](logs/visual-audit/raw-results-archive.json) records its uncompressed checksum and original local copy.
