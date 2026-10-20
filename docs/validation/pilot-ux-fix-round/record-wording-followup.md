# PR #142 — Spanish Record wording follow-up (2026-10-20)

Draft branch `fix/pilot-ux-round`; not merged or deployed.

Only the five requested Record-screen case phrases needed changes. The shared draft already used **Cajones** and **lb por cajón**. Office **Boxes = Cajas** stays unchanged. The entry script cache version advances from 8 to 9.

| Record screen | Old Spanish | New Spanish |
|---|---|---|
| Receive | Cajas / envases recibidos | Cajones / envases recibidos |
| Receive | lb por caja / envase | lb por cajón / envase |
| Pack | Cajas para empacar | Cajones para empacar |
| Pack | lb por caja (opcional) | lb por cajón (opcional) |
| Pack warning | Escribe lb por caja para usar el lote seleccionado. | Escribe lb por cajón para usar el lote seleccionado. |

The Record source and every shared script loaded by Record were searched for `caja`/`cajas`. No other case-meaning phrase remains. The shared `boxes` translation remains `Cajas`.

## Fresh validation

- **562 Node tests passed**, zero failures, skips or cancellations. This includes every correction reason in EN/ES, form/draft unit agreement through ES → EN → ES, preserved form values, Office boxes, and the missing-weight warning refusing to prepare a Pack draft.
- **34/34 reason and unit browser checks passed**, EN/ES at 390px; no records made by this checker.
- **22/22 planned checks passed** in one complete Spanish 390px pilot walkthrough with sessions enabled. Receive and Pack both prepared, confirmed and recorded successfully. Eleven numbered local receipts were created across the full walkthrough.
- Zero OpenAI calls. No hosted databases, Railway, server, pool, turn-handling, migrations or deployment changes. All disposable databases were removed and the private test cluster stopped.

The first walkthrough attempt, run concurrently with the reason/unit checker, failed while loading Office before reaching Receive and Pack. It correctly reported only 3/22 passed checks and an incomplete scenario; the failed report is retained. The screenshot showed Office initialization had not finished; the precise cause was not captured. A fresh run by itself completed all 22 checks without runtime edits.

Machine results, original failed attempt and source hashes: [record-wording-results.json](record-wording-results.json).

## Commands

```sh
node --test tests/test_*.js
# Both browser commands used the guarded local Python wrapper and PG17 on 127.0.0.1:55453.
PILOT_UX_PG_PORT=55453 PILOT_UX_PYTHON=/private/tmp/pr142-fix-runs/browser/python-local PLAYWRIGHT_MODULE=/Users/cns/Documents/factory-ledger/node_modules/playwright/index.mjs node tests/visual/run-pilot-ux-review.mjs
PILOT_SESSIONS=1 PILOT_UX_WIDTHS=390 PILOT_UX_LANGUAGES=es PILOT_UX_PG_PORT=55453 PILOT_UX_PYTHON=/private/tmp/pr142-fix-runs/browser/python-local PLAYWRIGHT_MODULE=/Users/cns/Documents/factory-ledger/node_modules/playwright/index.mjs node tests/visual/run-pilot-ux-round.mjs
```
