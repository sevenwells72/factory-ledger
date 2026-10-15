**Approved real-model repeat:** all 68 checks pass with gpt-6.1-sol/Pydantic 2.14.1 in both modes; see the [B3 real-model evidence](final-b3/README.md#approved-real-model-repeat). All task DBs dropped.

**Latest: [final round B3](final-b3/README.md)** — rebased onto #130/main, Pydantic 2.14.1, 6,103 Python per mode, 353 Node and all requested browser runners pass. Earlier B2 evidence below is historical.

# PR #132 round B fix recheck

Rebased first onto `origin/fix/assistant-page-ux` at `c2b57e9`; complete changelog histories remain, row 210 above 209. PR #131 subsequently merged as `9b978b82`. This round owns the display files too. PR #130 and the other worktrees were not changed.

All page evidence uses the actual dashboard HTML, CSS and JavaScript, authenticated HTTP routes, FL handlers and fresh local PostgreSQL 17 databases. `real-page` replaces only model tool selection; `real-model` uses the actual `gpt-6.1-sol` model. The data is a local synthetic catalog, three customer orders (including SO-261015-002 Kookies & Kreme), 66 supplier identities and two make ingredients. It is not a staging-data or staging-deployment claim. No keys appear in the artifacts.

## Worker-visible fixes

Every row is exercised at 1440px and 390px in English and Spanish, with sessions off and on. Each mode has its own screenshot directory. Center-scrolled supplemental captures show every order and all controls above the fixed composer.

| Fix | Tests | English 1440 / 390 | Spanish 1440 / 390 |
|---|---|---|---|
| Open orders are a readable list | `test_assistant_fix_b2.py::test_kookies_ampersand_open_order_filter; test_assistant_display.js` | [1440px](real-page/sessions-off/1440-01-open-orders-en.png) / [390px](real-page/sessions-off/390-01-open-orders-en.png) | [1440px](real-page/sessions-off/1440-01-open-orders-es.png) / [390px](real-page/sessions-off/390-01-open-orders-es.png) |
| Lot questions state the actual pounds | `test_assistant_understanding_round_b.py; test_assistant_display.js` | [1440px](real-page/sessions-off/1440-02-lot-quantity-en.png) / [390px](real-page/sessions-off/390-02-lot-quantity-en.png) | [1440px](real-page/sessions-off/1440-02-lot-quantity-es.png) / [390px](real-page/sessions-off/390-02-lot-quantity-es.png) |
| Receive retains weight and offers six supplier buttons plus search | `test_assistant_fix_b2.py::test_supplier_list_limits_deduplicates_and_searches_real_ids; test_fl_assistant_ui.js` | [1440px](real-page/sessions-off/1440-03-receive-supplier-search-en.png) / [390px](real-page/sessions-off/390-03-receive-supplier-search-en.png) | [1440px](real-page/sessions-off/1440-03-receive-supplier-search-es.png) / [390px](real-page/sessions-off/390-03-receive-supplier-search-es.png) |
| Make has one instruction and Different lot / Otro lote for every ingredient | `test_assistant_make_round_b.py; actual-page Different lot Record test; test_fl_assistant_ui.js` | [1440px](real-page/sessions-off/1440-04-make-final-ingredient-en.png) / [390px](real-page/sessions-off/390-04-make-final-ingredient-en.png) | [1440px](real-page/sessions-off/1440-04-make-final-ingredient-es.png) / [390px](real-page/sessions-off/390-04-make-final-ingredient-es.png) |
| Out-of-scope uses only the requested Factory Ledger wording | `test_assistant_understanding_round_b.py; test_fl_assistant_ui.js` | [1440px](real-page/sessions-off/1440-05-out-of-scope-en.png) / [390px](real-page/sessions-off/390-05-out-of-scope-en.png) | [1440px](real-page/sessions-off/1440-05-out-of-scope-es.png) / [390px](real-page/sessions-off/390-05-out-of-scope-es.png) |
| Unknown lot plainly says No lot called X / No hay un lote llamado X | `test_assistant_fix_b2.py::test_nonexistent_lot_is_plain_permanent_answer; test_fl_assistant_ui.js` | [1440px](real-page/sessions-off/1440-06-unknown-lot-en.png) / [390px](real-page/sessions-off/390-06-unknown-lot-en.png) | [1440px](real-page/sessions-off/1440-06-unknown-lot-es.png) / [390px](real-page/sessions-off/390-06-unknown-lot-es.png) |
| Spanish questions select Spanish without pressing ES | `test_assistant_fix_b2.py; actual-page automatic-spanish assertions` | — | [1440px](real-page/sessions-off/1440-07-automatic-spanish-es.png) / [390px](real-page/sessions-off/390-07-automatic-spanish-es.png) |

[All deterministic page results](real-page/results.json) and [all actual-model page results](real-model/results.json) include per-case language, viewport, session mode, visible text, screenshots, API failures and model-call metadata. Also see [actual-model Kookies and exact Classic #9 checks](reviewed-queries-real-model/results.json).

## Reviewed bugs and worker clarity

- Kookies & Kreme now matches the ampersand product and returns SO-261015-002. Unknown product filters ask for clarification instead of reporting zero. [English screenshot](reviewed-queries-real-model/sessions-off/1440-10-kookies-order-filter-en.png), [Spanish screenshot](reviewed-queries-real-model/sessions-off/1440-10-kookies-order-filter-es.png). Tests: `test_kookies_ampersand_open_order_filter`, `test_unknown_product_order_filter_asks_for_product`.
- A receive correction updates the existing chat, cancels the old draft and removes its Record control. Exact 10 lb → supplier → 3 × 5 lb mismatch → 15 lb correction passes in EN/ES, keeps approved IDs, and posts nothing. Photos survive replacement; a concurrent/already-started Record prevents replacement. [English 390px](real-page/sessions-off/390-09-corrected-receive-quantity-en.png), [Spanish 390px](real-page/sessions-off/390-09-corrected-receive-quantity-es.png). Tests: `test_reviewed_receive_mismatch_can_be_corrected_in_same_chat`, the correction lifecycle/race/attachment tests in `test_assistant_fix_b2.py`.
- Exact product name, SKU or ID wins before the Classic #9 family shortcut. [English](reviewed-queries-real-model/sessions-off/1440-11-exact-classic-inventory-en.png), [Spanish](reviewed-queries-real-model/sessions-off/1440-11-exact-classic-inventory-es.png). Tests: `test_exact_full_classic_nine_inventory_never_expands_family` and resolution regressions.
- Receipt answers use effective status and visibly say **VOIDED / ANULADO**, including ADJ-261015-240. [English mobile screenshot](display-edge-cases/390-voided-en.png), [Spanish mobile screenshot](display-edge-cases/390-voided-es.png). Tests: `test_receipt_uses_effective_void_state` and `Round B receipt effective void status is visible` in `test_assistant_display.js`.
- Supplier suggestions rank actual product receive history and aliases, show at most six, and search the catalog directly. A1 Baker / A1 Bakery / A1 Bakery Supply and legal-suffix variants collapse only in the list. Supplier records/IDs and printed EXTRA-KOSHER labels remain unchanged. Tests: supplier list/search/ranking EN/ES tests and Node UI tests. Search screenshots are in the table above.
- UNVERIFIED Different lot evidence is excluded from physical-stock totals and displayed as unverified, awaiting follow-up. Underlying ledger debits are preserved. [English mobile screenshot](display-edge-cases/390-unverified-en.png), [Spanish mobile screenshot](display-edge-cases/390-unverified-es.png). Tests: `test_unverified_trace_debit_is_not_negative_inventory` EN/ES and Node unresolved-input tests.
- Singular text is `1 open order` / `1 pedido abierto`; Spanish intent detection includes normal question punctuation. Tests: bilingual order filter and language tests, real page checks.

## Traceability remains as reviewed

No ingredient is auto-confirmed or skipped, including a single-lot ingredient. Empty/wrong tag evidence is refused. Actual-page tests click Different lot, require each typed physical tag, explicitly press Record, inspect the outgoing confirmations, and verify an unresolved trace lot plus follow-up exception while suggested stock stays unchanged. Extra-kosher hold/approval, atomicity, replay and FIFO protections remain covered by the unchanged core and full suites. [Actual-page receipt](real-page/sessions-off/1440-08-different-lot-trace-en.png), [physical controls in Spanish](real-page/sessions-off/390-04-make-final-ingredient-es.png).

## Dependency and validation

`requirements.txt` pins pydantic **2.11.9**. A read-only check of the running staging container actually returned **2.14.1** on October 15, contrary to the review's older-version premise. The chosen pin follows the user's below-2.12 requirement and the review-proven compatible version. Local tests validate every generated strict function schema and forbid regex lookarounds. An explicitly approved real OpenAI request with the complete tool list returned HTTP 200. [Request check and staging observation](openai-request-validation.json). The official [Structured Outputs schema guide](https://developers.openai.com/api/docs/guides/structured-outputs) documents the supported schema subset.

- Fresh PostgreSQL, sessions off: **5,948 Python passed**, zero failures/errors/skips. This precedes the final two exact mismatch-loop test additions; both then passed independently.
- Second fresh PostgreSQL, sessions on, final Python files: **5,950 passed**, zero failures/errors/skips.
- Node: **321 passed**, zero failures/skips.
- Round A browser: **30/30 per mode**.
- Existing Assistant browser runner: passed both modes.
- Original B browser contract runner: **9/9 groups per mode**.
- Final new page/model counts, source hashes and database cleanup are recorded in [verification.json](verification.json).

The first diagnostic full run had fixture setup errors because PostgreSQL folded unquoted mixed-case task DB names; quoted identifiers fixed the test harness. All final runs above passed. Every database created for this round uses the `fl_fixB2_` prefix and is dropped after validation.

The additional [display-edge-cases/results.json](display-edge-cases/results.json) contains eight passing actual-dashboard renderer checks for VOIDED and unverified inventory (EN/ES, 1440px/390px), using explicit read-card fixtures generated by `assistant_replies.read_card`. These eight checks use neither a database nor a model. The [complete screenshot manifest](screenshot-manifest.json) lists all **248 PNGs**.
