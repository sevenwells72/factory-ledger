# Receive lot capture validation

Branch `feat/receive-lot-required` starts from `origin/main` at `27a25186`. This change is unmerged and undeployed.

Purchased ingredient receipts require the printed supplier lot, or the explicit **No lot on label / Sin lote en la etiqueta** choice. Missing, blank and placeholder codes without the choice return HTTP 422 with English and Spanish instructions. The rule covers the dashboard, `/receive/prepare`, legacy `/receive`, and its preview/commit aliases. Valid commingled component codes remain accepted.

The explicit exception keeps the existing unidentified-lot workflow: seven business days after entry, with the existing New York deadline and correction-only identification. Supplier requirements and links remain; legacy named-vendor handling and expected-receipt matching are preserved. No database schema changes or migration files are needed.

The CNS lot code appears first and large on receive success, receipt read-back and Assistant receipts, followed by **Write this on the pallet tag / Escriba esto en la etiqueta del pallet**. The Floor GPT schema and instructions expose the explicit exception and CNS-first read-back. The Assistant only accepts a worker's direct label-absence assertion; model-invented flags, questions and instructions do not authorize it, and explicit denials clear a saved choice.

Validation on 2026-10-21:

| Check | Result |
|---|---|
| Full Python suite, final frozen source | 7,044 passed; zero failures, errors or skips; 151.43 seconds |
| Full JavaScript suite | 577 passed; zero failures or skips |
| Local browser checks | 12 passed; English/Spanish at 390px and 1440px |
| Floor GPT OpenAPI operations | 30 before / 30 after |
| Floor GPT instruction length | 7,997 characters; within 8,000 |
| Diff whitespace and Python/YAML syntax | Passed |

Python used the pinned Python 3.12.14 / Pydantic 2.14.1 environment and a fresh UTF-8 PostgreSQL 17 instance at `127.0.0.1:55471`. The runner explicitly removed `DATABASE_URL` and set `TEST_DATABASE_URL` to its disposable local database. JavaScript ran with `node --test tests/test_*.js`.

New coverage includes blank/sentinel rejection on every receive endpoint, strict boolean exception handling, unidentified status and the unchanged deadline, supplier links, normal receives, commingled receipts, non-purchased exemptions, old-ticket commit revalidation, server policy metadata, Assistant source binding, and bilingual receipt rendering. Existing unrelated receipt fixtures now supply actual lot evidence; intentional unidentified fixtures explicitly set the flag.

Browser QA used local mocked API responses with all external requests blocked. CNS lot font size was 26px on phones and 38px on desktop, above the receipt number; no horizontal overflow or browser errors. See [browser checks](browser-checks.json), [English receive success](entry-success-en-390.png), and [Spanish receipt read-back](receipt-es-390.png).

Cache versions: entry 9→10, receipt 9→10, ledger-ticket 9→10, ledger-ui 17→18, ledger-forms CSS 8→9, assistant-display 11→12, fl-assistant 15→16, fl-assistant CSS 8→9. All referencing dashboard pages were updated.

No production or staging access, deployment, production migration, merge, or main-branch push occurred. No changes touch the live count worktree, `audits/fresh-start/`, or count/reset logic. Historical changelog entries remain intact; the canonical global log was updated through its verified symlink target using in-place writes.
