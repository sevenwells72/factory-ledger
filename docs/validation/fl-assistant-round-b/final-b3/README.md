# PR #132 final round B3

Rebased the three B commits onto current main `f0ae3c2` (#130 merged), following review comment 6198313762. This checkpoint supersedes earlier B2 pin/base/test counts; earlier evidence remains historical.

## Conflict resolutions

- `assistant_tools.py`: keep B pack/make/physical-lot rules and all #130 customer rules; scope includes customers, lots and shift summary.
- `dashboard/fl-assistant.js`: B scope wording without ChatGPT; combined refusal rules; supplier helpers; both lot-exception receipts and CUS Continue with an order. `assistant-display.js` retains every #130 contact/order/line field and existing/add-new customer choice.
- `fl_assistant.py`: keep both query and supplier_search. A completely unmatched office/owner customer still offers Add new; B’s non-customer unknown-item clarification stays intact.
- `tests/test_fl_assistant_ui.js` and `tests/test_assistant_display.js`: retain both suites. Flip main’s ChatGPT assertion to the exact B sentence and reject ChatGPT in both languages. Retain the customer-ID trust refusal with B’s plain decorated response.
- All four scratch-prefix conflicts use main’s scratch_database_prefix(): test_exceptions_tables_061.py, test_lot_identity_047.py, test_trace_tables_048.py and test_write_tickets.py.
- Both complete changelog histories preserved; B row 211 follows main customer row 210 and A row 209. Assistant cache v9 and shared display v4 exceed main v8/v3.

## Final fixes

Pydantic is pinned to **2.14.1** and the pin assertion matches. The guard rejecting lookaround in every generated strict tool schema remains. `jsonschema` is already declared in `tests/requirements-test.txt` on current main and is retained; the clean test venv installs it from that file.

Different lot / Otro lote hides only that ingredient’s suggested lot codes, method pickers and hints. Other ingredients stay visible. Toggling back restores controls and clears the tag. Node and actual-page checks cover both languages, both session modes and 1440/390 widths.

All four duplicate-warning generators use **hace 1 minuto** and plural minutos otherwise; physical-lot Spanish references say **lotes**. Twenty-four new tests preserve English wording and receipt/actor/reference evidence.

Traceability remains as reviewed. Every ingredient needs a physical tag, including single-lot ingredients. Missing/wrong/blank tags refuse. Different lot still posts unresolved trace evidence with follow-up and leaves the suggested FIFO stock untouched. Extra-kosher hold/approval, atomicity and replay remain intact. `lot_confirmation.py` and `resolution.py` are byte-identical to reviewed B `14aecc6`.

## Validation

| Check | Result |
|---|---|
| Full Python, fresh PostgreSQL 17, sessions off | 6,103 passed; zero failures/errors/skips |
| Full Python, fresh PostgreSQL 17, sessions on | 6,103 passed; zero failures/errors/skips |
| Node | 353 passed; zero failures/skips |
| Existing Assistant + #130 new-customer browser flow | Both modes pass; Add, CUS → Continue → ORD, duplicate recovery, mobile ES |
| Round A browser | 30 per mode |
| B backend contract browser | 9 groups per mode |
| B actual dashboard page | 68 total, both modes, EN/ES, 1440/390 |

Python used a fresh venv from requirements.txt plus tests/requirements-test.txt: Python 3.12.14, Pydantic 2.14.1, pydantic-core 2.50.1, jsonschema 4.26.0; pip check passed. Modes ran sequentially. The saved single-database runner allocates a fresh DB per each of 22 schema/race groups and preserves original migration/fixture SQL; every collected test ran exactly once per mode. Its temporary allocation layer permits the requested mixed-case `fl_fixB3_` database names while retaining main’s lowercase-prefix helper in source.

Browser backend/page checks use real local handlers and PostgreSQL with synthetic data, queuing only model tool selection. Existing/A browser suites use fixture responses. All databases use `fl_fixB3_` and are dropped by runner cleanup. No hosted database or deployment was changed.

See [verification.json](verification.json), [page results](page-results.json), [contract results](contract.json), mode logs and full per-group Python logs. The two stale-import failures from the initial in-flight run occurred before the final customer integration edits and are excluded from final counts; the entire off suite was then repeated fresh and passed.

Selected inspected screenshots:

- [Different lot, desktop EN](different-lot-desktop-en.png), [mobile ES](different-lot-mobile-es.png), [mobile ES sessions on](different-lot-mobile-es-sessions-on.png)
- [Scope, desktop EN](scope-desktop-en.png), [mobile ES sessions on](scope-mobile-es-sessions-on.png)
- [CUS Continue with an order](customer-continue-desktop-en.png), [Spanish duplicate recovery](customer-duplicate-mobile-es.png)

Saved terminal logs have trailing whitespace normalized; result text and counts are unchanged.
