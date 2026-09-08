# Design audit B/C + deployed SO intake integration

Status: validated locally; prepared for push and review. Not merged into main or deployed.

Branch: `fix/design-audit-3-integrated`, cut from `b0eb2c3`. Merge direction: `origin/main` at `84b89e3` into this branch. The original design worktrees and shared checkout were not edited.

## Combined changes relative to main

Retains Batch B navigation/responsive layouts and Batch C controls, record URLs/history, sorting, typography, trace and production changes. Main's deployed SO intake remains authoritative. New integration-specific work:

- Order-number button and SO source-document button coexist in the same cell, with an explicit order-number sort value and no wrapping between controls. Existing paperclip stopPropagation and Factory Ready label guard remain.
- Main's shared intake-logic.js loads before dashboard.js; the deleted ER-specific module is not referenced. Design controls load before consumers, shell styles last, shell JS after page logic. New Sales Order primary action and full modal are retained.
- Dashboard JS 56, CSS 39, intake logic 9; other shared assets and page links retain Batch C versions.
- Both change histories are preserved in full. Main SO intake row stays 121; Codex A verification/B/C become 122/123/124. Historical deployment wording is retained as history.
- Shared intake quantity cells/source labels wrap at phone widths, fixing overlap exposed by larger text. Both ER/SO sticky footers and navigation hiding remain intact.
- Added a reproducible browser fixture harness at tests/visual/run-integrated-intake.mjs.

Protected files `main.py`, `extraction.py`, migration 050, schema dump and SO intake design document are byte-identical to origin/main. No dashboard allowlist change; POST /sales/orders was not added.

## Local database and automated validation

Migration 050 applied successfully with psql to `postgresql://localhost:5432/factory_ledger_design_audit_a`. Server identity confirmed database factory_ledger_design_audit_a, address ::1, port 5432. No production migration or data writes.

- `node --test tests/test_*.js`: **59 passed**, zero failures/skips (Batch A 5, Batch C 6, ER logic 35, SO logic 9, pallet logic 4).
- Requested four Python suites: **244 passed**.
- Final run including `tests/test_er_intake_logic_js.py`: **245 passed**, zero failures. Existing warnings remain (152 in final run).
- Actual Node intake file is still `tests/test_er_intake_logic.js`. It imports `dashboard/intake-logic.js`, covers ER and SO, and is wrapped by `tests/test_er_intake_logic_js.py`.
- JS syntax, git diff whitespace, protected-file equality, script/style ordering and preservation of both parent log histories passed.

Python command (pinned Python 3.12 environment):

```sh
TEST_DATABASE_URL=postgresql://localhost:5432/factory_ledger_design_audit_a \
DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib \
/Users/cns/Documents/factory-ledger/.venv-test.nosync/bin/python -m pytest \
tests/test_recent_ledger.py tests/test_dashboard_b2.py \
tests/test_sales_order_extract.py tests/test_expected_receipt_extract.py \
tests/test_er_intake_logic_js.py --disable-warnings -o addopts='' -q
```

## Browser results

Used isolated headless Chrome with local static assets and intercepted fixture APIs because the interactive Mac browser was locked. All network requests were either local or intercepted; no real API request or production write occurred.

Four configurations: **1440×900 and 390×900, each light and dark**. Both ER and SO exercised in each configuration: **eight upload-to-review flows**.

- Synthetic PDF file uploaded through the actual file input and multipart request; extraction and matching fixture responses drive real review rendering with three lines.
- Review fields populated, approval enabled for valid matched inputs, reference/PO edits exercised. Approval itself was not submitted in the browser; server approval behavior is covered by local Python tests.
- Both modal bodies have no horizontal overflow: desktop client/scroll width 518/518px; phone 388/388px. Overall page width equals viewport width.
- No quantity-cell/source-label overlap after responsive CSS adjustment.
- Sticky actions remain on screen: desktop buttons 44px high; phone ER 52px, SO 44px, with 17px action text. Phone navigation is hidden while either modal is open.
- Sales Orders sorting retains all eight rows with the original document IDs and corresponding detail rows.
- Order button and paperclip stay side by side. Clicking the paperclip requests the correct document URL and invokes document opening; the order list stays visible and detail view stays hidden.
- Zero page errors, unmatched fixture API routes or unexpected network requests in all four configurations.
- Captured and inspected top/bottom review screenshots, including phone dark ER and SO and desktop layouts.

Reproduce with installed Playwright and Chrome:

```sh
node tests/visual/run-integrated-intake.mjs
```

Optional environment variables: PLAYWRIGHT_MODULE for the installed module path; BROWSER_CHANNEL (defaults chrome); BROWSER_OUTPUT for screenshots/results.json. This is fixture validation, not real OCR/storage availability or real-device keyboard testing.

## PR constraint

GitHub supports changing a PR base branch but not its source branch. PR #34's original source is fix/design-audit-3. The user was asked to choose a replacement draft PR or an additional fast-forward of that original remote branch. This report does not authorize either workaround or a merge/deployment.
