# Compact Make browser walkthrough

All 48 checks passed: English and Spanish at 390px and 1440px. The real Record entry page called the real local FL handlers. Eight actual Make receipts were recorded across four disposable local PostgreSQL 17 databases; all four databases were dropped. No hosted database, Railway or OpenAI was contacted. Browser requests to other origins and Assistant turns were refused.

This walkthrough uses clearly synthetic local stock and a three-ingredient Classic Granola #9 test recipe (150 lb Oats, 40 lb Sunflower Seeds and 10 lb Coconut per 200 lb batch). It does not claim to reproduce the current production recipe or balances.

The unchanged main screen required 4,349px of draft scrolling in English and 4,380px in Spanish. The new test drafts measured 1,138px and 1,146px, about 74% shorter. All three ingredient rows, the red shortage and the Record button fit together in one 390 × 844 ingredient view. The entire draft, including its header and existing shortage acknowledgement, remains scrollable.

| 390px language | Before | Compact ingredient view | Change lot | Receipt method |
|---|---|---|---|---|
| English | [Before](screenshots/before-en-390-make-draft.png) | [After](screenshots/after-en-390-ingredient-rows.png) | [Choices](screenshots/after-en-390-change-lot.png) | [Tap evidence](screenshots/after-en-390-receipt-confirmation-method.png) |
| Spanish | [Antes](screenshots/before-es-390-make-draft.png) | [Después](screenshots/after-es-390-ingredient-rows.png) | [Lotes](screenshots/after-es-390-change-lot.png) | [Método](screenshots/after-es-390-receipt-confirmation-method.png) |

The same walkthrough checks the recipe hint before Prepare, oldest-first preselection without automatic confirmation, blocked Record until every row is checked, progress after each tap, Change with full lot codes/received dates/available pounds, and changing an already checked lot without silently checking the replacement. Both active oat lots end in C-01; the direct Change list and the optional ambiguous-suffix choices both let Arturo choose the full bag tag and then explicitly confirm it. The unused suggested oat lot stays at 500 lb; only the selected alternate lot is deducted.

It also checks the red zero-stock Coconut warning, real shortage exception, a language switch that keeps physical checks, the existing substitution form, Choose a different ingredient lot, and a second receipt with full-code, scan and tap evidence kept distinct. Every actual receipt is compared with immutable transaction confirmation rows and Arturo’s actor id. Scanner input is simulated by entering the scanned lot code; no physical barcode reader was used. There are no browser errors or unexpected direct ledger writes.

Results: [browser-results.json](browser-results.json). Unchanged-main captures: [browser-before-results.json](browser-before-results.json). Exact final capture source hashes and independent visual review: [browser-capture-manifest.json](browser-capture-manifest.json). Earlier capture chronology: [browser-capture-history.json](browser-capture-history.json).

Re-run with the pinned test Python and a disposable loopback-only PostgreSQL cluster:

```sh
MAKE_DRAFT_PYTHON=/Users/cns/Documents/factory-ledger/.venv-test/bin/python MAKE_DRAFT_PG_PORT=55473 MAKE_DRAFT_OUT=/tmp/fl-makedraft-browser node tests/visual/run-make-draft-compact.mjs
```

The Python runtime defaults to `MAKE_DRAFT_PYTHON`, then `PILOT_UX_PYTHON`, then the checkout’s `.venv-test/bin/python`. The command above explicitly uses the pinned runtime available for this worktree.

The runner creates/drops only databases named `fl_makedraft_*`. `MAKE_DRAFT_WIDTHS` and `MAKE_DRAFT_LANGUAGES` can narrow diagnostics; default coverage is English/Spanish at 390px/1440px. For before captures, set `MAKE_DRAFT_BEFORE=1` and `MAKE_DRAFT_BEFORE_ROOT` to an untouched base checkout containing `dashboard/`.
