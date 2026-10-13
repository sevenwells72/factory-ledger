# D1 / D2 dashboard review evidence

Builder: Codex. Reviewer: Claude Code. Branch `feat/dash-d1-d2`, based on `f4e51b5`.
No production changes, schema migrations, or edits to parallel worktrees.

## Part 1 — D1

Receipt lookup is shared by dashboard HTML pages. Receipt details read `GET /receipts/{number}`; History, Recent Entries and all Activity tables show the stored receipt as the primary identifier. Legacy rows are explicitly labeled Legacy TX; no receipt is fabricated. The additive backend change projects stored receipt identities onto existing read responses.

Exceptions are fetched from `GET /exceptions?status=open`, which includes escalated items, grouped by kind and ordered by FL's overdue flag and due date. Rows name the owner and link to their source receipt when FL has issued one. Resolve reads the individual exception before showing counted adjustment, A6 photo, supplier-lot, or owner decision forms. Reserved extra-kosher receipts and other held drafts are never labeled recorded.

### Current API differences from the design

- A3b ships `/exceptions/{id}/resolve`, `/approve`, and `/reject`, not `/resolve/prepare`. D1 uses these actual operations after an explicit review card. FL performs the count calculation and all approval/evidence checks. No client-side rule or stock calculation was added.
- A3b counted resolution may post a correction without a receipt; metadata resolution and rejection also do not issue XR receipts. The UI reports the server's exception status without green success and instructs users to check History before retrying a counted adjustment. This backend gap is not silently represented as a ticket or a fabricated receipt.
- Large-correction holds may have no receipt number yet. Such rows say receipt unavailable. The current backend also has no ticketed void producer; the read projection can show a VD receipt if a committed void ticket links its `result_ref.correction_id`, and otherwise preserves the legacy correction trail.
- A5 part 2 is absent on this base. Supplier-lot forms use `POST /lots/{lot_id}/supplier-lot/prepare` with `supplier_lot_code`, `reason_code`, `note`, `client_source: dashboard`, followed by the normal ticket commit. There is no PATCH fallback. Check its final contract and run the identification/auto-resolution round trip once that endpoint reaches main.

### Verification

- Dedicated fresh local PostgreSQL databases; never the shared test database.
- Node assertions for receipt/hold truthfulness, verbatim FL messages, escaping and queue ordering.
- Python tests execute receive prepare/commit and verify receipt identity on History, Recent Entries, Activity and receipt detail.
- `tests/visual/run-d12.mjs` runs Chrome at 390 px against real local FastAPI routes. It asserts session auth headers, token-only storage, receipt lookup, supplier-lot prepare 404 without fallback, owner approval, logout and EN/ES. No mocked happy-path write routes.
- Browser fixtures are restricted to a named local database by `tests/visual/d12_fixture.py`; runtime evidence is under `.venv-test/d12-visual/`.

Part 1 final checks: 2,745 Python tests on fresh `fl_d12_20261013_p1_final`, 110 Node tests, zero failures/errors/skips; real-route Chrome passed. Part 2 and staging evidence follow in a separate commit.
