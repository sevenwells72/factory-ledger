# PR 142 fix round — ready for independent review, still Draft

This round builds on `976d0a72` and completes the remaining requests from the Desktop review. It does not merge or deploy anything.

- The pilot checker now fails on sign-in/setup errors, skipped checks and unstarted walkthroughs. It reports the reason and the missing checks. An executable regression forces a mid-run sign-in error and proves a nonzero exit; all four reporting regressions pass.
- Make and Pack have no photo control on the Record form or shared draft, even if a caller asks to enable one. The Assistant draft also has no upload control. Receive keeps BOL/lot-tag photos; Adjust and Found retain correction photos and record successfully.
- All eight correction reasons share the same form/draft labels in English and Spanish, including canonical `wrong_lot`: **Wrong lot used / Lote equivocado**. Historical `wrong_lot_used` still displays correctly.
- Office consistently uses **Cases = Cajones**, **Boxes = Cajas**; singular case is **cajón**. Stored unit codes and quantities are unchanged.
- Changing an attached photo's type keeps the image and updates its saved label. If the update fails, the original photo and its previous label stay intact. Uploading still blocks Record, Edit and Cancel until complete.
- Selected-item, batch, photo, warning, expiry, role and confirmation text updates when English/Spanish changes, without clearing typed values or evidence. Floor/Office/Owner labels are translated too.
- Pilot links have 44-pixel tap areas and phone tag fields use readable 16-pixel text. Asset versions were refreshed.

## Test counts

Every final run below passed; there were no failed, errored or skipped Python/Node cases.

| Check | Result |
|---|---|
| Full Python, sessions OFF | **6,916 passed; 0 failed/errors/skipped** |
| Full Python, sessions ON | **6,916 passed; 0 failed/errors/skipped** |
| Full Node | **559 passed; 0 failed/skipped**, including four fail-closed reporting regressions |
| Actual Luz/Arturo pilot walkthroughs | **96/96 planned checks; 8/8 complete scenarios**; EN/ES, 390/1440, sessions OFF/ON |
| Pilot local records/evidence | **44 receipts; 20 explicit physical-tag confirmations; 16 saved photos** |
| Actual reason/unit/live-language review | **34/34 checks** |
| Additional actual photo/live-language regression | **13/13 checks**; 2 extra Receive receipts, 6 uploads; identical image bytes through reclassification and correct final receipt labels |
| Dashboard/Assistant fixture runners | **20/20 jobs**, 0 failed |
| Dashboard access/retry | **48/48** |
| Dashboard key sessions | Both modes passed |
| Main Assistant/customer | Both modes passed |
| Assistant Round A | **30/30 each mode** |
| Existing pilot screens | **60/60** |
| Hosting/navigation | **272 checks; 16/16 scenarios** |
| Sales layout / list / detail / exit actions | **26 scenarios/312 groups; 2 scenarios/20 groups; 36 checks; 174 checks** — all passed |
| Assistant phone / mobile / final floor | **288; 144; 240** checks — all passed, including WebKit |
| Session/shift | Both modes passed |
| Production trace, run types, runs interactions | All four width/theme variants each passed |
| Integrated Intake | **4/4 scenarios**, eight upload-review measurements |

The two previously failed browser checkers are repaired:

- **Round A:** accepts only the seven documented speculative legacy `/dash/` asset 404s, once each, and requires every final `/dashboard/` asset to return 200; other missing requests still fail.
- **Intake:** uses the current Sales Order state tabs, readiness checkbox and eight-column table, while retaining its upload/review tests.

## Pilot-page visual findings only

The earlier **418 findings / 62 warnings** came from an old inventory of index, scheduler, traceability, Sankey and process-flow screens. It contained no Record entry, Office, Receipt, Exceptions or receipt-based Today captures. Those totals are not evidence of defects on these pilot pages.

I measured **92 actual pilot-page states** across those five pages during the complete walkthroughs. After the small fixes, there are **0 undersized tap targets and 0 horizontally overflowing captures**. The contrast checker flags 24 intentionally disabled controls at half opacity; enabled text and controls have no contrast flags.

| Staff-visible issue | Result |
|---|---|
| Small home/Today/receipt links | Fixed: 44-pixel tap areas |
| Small phone physical-tag input text | Fixed: 16-pixel input text |
| Spanish draft shows English role names | Fixed: Piso, Oficina, Propietario |
| Make requires substantial phone scrolling | Remains: the full ingredient quantities, stock, shortages and physical-tag review stay visible; a larger draft-layout redesign is outside this quick fix |
| Catalog product names and native file-picker text | Catalog names retain their identities; native picker wording follows browser/device language rather than the app switch |

There is no requested functional fix left outstanding. The application keeps Prepare → Record → numbered receipt, existing role gates, owner-only backfill and explicit physical-tag evidence. No server, pool, turn-handling, permission, migration or deployment code was changed by this round.

## Six inspected 390px draft screenshots

| Draft | English | Spanish |
|---|---|---|
| Make — no photo control | [EN](screenshots/make-draft-en-390.png) | [ES](screenshots/make-draft-es-390.png) |
| Pack — no photo control | [EN](screenshots/pack-draft-en-390.png) | [ES](screenshots/pack-draft-es-390.png) |
| Receive — photo retained | [EN](screenshots/receive-draft-en-390.png) | [ES](screenshots/receive-draft-es-390.png) |

All six were visually inspected through the Record button. They use synthetic local practice data.

## Isolation and replay

Zero OpenAI calls; no staging/production database access or Railway operations. Both PostgreSQL 17 clusters were local-only, with fresh UTF-8 synthetic databases, stripped hosted/model credentials and blocked external Python sockets. Browser routes blocked hosted traffic. All task databases were dropped, both private PostgreSQL servers stopped and all browser servers closed.

Full Python used the pinned Python 3.12.14/Pydantic 2.14.1 environment and `python -m pytest` in this Git checkout, with KEY_SESSIONS_ENABLED=0 then 1. Node used `node --test tests/test_*.js`.

Actual browser commands: `run-pilot-ux-round.mjs`, `run-pilot-ux-review.mjs` and `run-pilot-photo-language.mjs`, with `PILOT_UX_PG_PORT=55453`, `PILOT_UX_PYTHON=<local guarded Python wrapper>` and `PLAYWRIGHT_MODULE=<local Playwright module>`. For scoped visual measurements, preload `tests/visual/lib/pilot-page-audit.mjs` using Node's `--import` and set `PILOT_PAGE_AUDIT_OUT`.

The photo regression's initial bare-product-name assertion was corrected to compare the actual full catalog label; the failed attempt exited nonzero and made no application change. The final complete run above is retained separately from its raw attempt logs.

[Machine counts and source hashes](verification.json), [pilot plan/completion results](actual-browser-results.json), [reason/unit results](reason-unit-results.json), [photo-label results](photo-language-results.json), [scoped pilot visual measurements](pilot-page-audit.json), [browser job results](browser-suite-results.json), and final stdout logs are retained here. Full raw logs/JUnit and all 104 pilot screenshots remain at `/private/tmp/pr142-fix-runs/`.
