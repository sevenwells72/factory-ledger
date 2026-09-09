# Design audit Batch 4 — CSS fixes

Branch: `fix/design-audit-4`, based on `origin/main` at `e9b1ee5` (deployed B/C). LOCAL ONLY, NOT DEPLOYED. No application JavaScript, backend, database or intake behavior changes.

## Changes

1. Responsive order cards place remaining quantity above a full-width status/readiness row. Readiness badges wrap inside the card; expand/readiness checkbox targets and handlers are unchanged.
2. Phone Sales Orders filters use a full-width status/customer field and paired checkboxes. Actions use a two-column grid and sorting shares a compact row with direction. Toolbar height in the same 390px fixture falls from **317px to 271px (46px / 15%)**.
3. Native text/search/number/date/select/textarea controls inherit surface, text and border tokens. Root color-scheme follows light/dark appearance, preserving native pickers and checkbox behavior.
4. History content gets 12px phone / up to 32px desktop inline padding, scoped to the history page.

Assets: dashboard.css **40**, shell-layout.css **3**. All five HTML references and Recent Ledger assertions updated. Dashboard JS remains 56, intake 9; other script/page-link versions unchanged. FACTORY row 125 records local-only status.

## Verification

- `node --test tests/test_*.js`: **59 passed**, zero failures/skips.
- `test_recent_ledger.py` using the local factory_ledger_design_audit_a DB: **12 passed**, zero failures (25 warnings).
- `git diff --check`: passed. Application JS/backend diff: empty. No remaining HTML CSS39/shell2 references.
- Existing local browser harness extended to reproduce readiness overlap and capture order cards, toolbar, native controls and history before/after. Baseline captured before CSS edits from this branch's main base; same fixture data and viewport configuration used after.
- Headless Chrome local harness at **390×900 and 1440×900, both light/dark**. All API traffic intercepted; no production reads/writes or real OCR/storage calls.
- **Eight SO/ER upload-to-review flows per run:** real file-input/multipart UI, synthetic extraction and matching, three review cards, valid approval gate, reference edit and close. Approval was not submitted.
- At 390px all three ready fixture rows place badges below quantity with **zero overlap**, in both themes. Desktop row layout remains unchanged.
- Paperclip click after Customer sorting opens only the fixture document URL; order detail stays closed. All eight order/document/detail groups remain paired.
- Modal client/scroll widths match (388/388px phone; 518/518px desktop). No horizontal overflow or review quantity-label collisions. Sticky actions remain visible, phone nav hidden behind dialogs; ER approval 52px high and SO 44px at 390px, both with 17px text.
- History phone padding changed from 0 to 12px. Dark select background changed from white to rgb(40,53,72), matching the existing search-surface token.
- Zero browser page errors, unmatched API fixtures or unexpected external requests.

Run fixture browser checks with `node tests/visual/run-integrated-intake.mjs`; PLAYWRIGHT_MODULE, BROWSER_CHANNEL and BROWSER_OUTPUT can select installed tooling and output directory. CAPTURE_BASELINE=1 captures the old layout without enforcing the new below-quantity assertion. Final run did not set that flag.

This is local fixture verification, not a production deployment or physical-phone keyboard/safe-area test. Before/after screenshots are included in the user-facing report.
