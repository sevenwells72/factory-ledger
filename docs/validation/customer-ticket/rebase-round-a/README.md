# PR #130 rebase onto merged Round A

Rebased `feat/customer-ticket` onto `9b978b8` (PR #131). Customer row 210 follows preserved main rows 209 and 208. Earlier row-allocation and test-count notes describe their original checkpoints. Round B worktree was not accessed or changed. `assistant_tools.py` is byte-identical to the original PR head.

Customer contact fields, order PO/date/line display and the existing/add-new customer choice descriptors now live in `assistant-display.js`. The page renders those helpers while retaining Round A refusal, pending, session, Today and receipt behavior. Active duplicate customers offer an existing-customer choice with Record disabled. Assets: Assistant v8, display v3, shared ledger display v12 (all four consumers).

Final validation:

- Full Python on a newly created UTF-8 PG17 database: **5,947 passed**, zero failures or skips. `TEST_DB_PREFIX=fl_cust_rb_` covered suite-created child DBs.
- Node `node --test tests/*.js`: **327 passed**, zero failures or skips.
- Office real HTTP/database browser suite, sessions off/on: **PASS/PASS**. The first setup attempt omitted required migration 083; after rebuilding fresh DBs with 083 applied, both modes passed, then passed again with final shared asset URLs.
- Existing Assistant browser suite, sessions off/on: **PASS/PASS**, including Sunrise choices, CUS → Continue → separate order draft → ORD, duplicate recovery, Spanish/mobile, retry and session expiry. These tests use deterministic API fixtures.
- Round A browser suite, sessions off/on: **30/30 + 30/30**, preserving its 1440/390 layout, Today, draft, warning, receipt, timeout and recovery checks.
- Real local page, exact `New order for Sunrise Bakery`, **1440/390 × sessions off/on: 4/4**. Real seeded Sunshine Bakery remained a candidate beside **Add 'Sunrise Bakery' as a new customer**. Contact fields appeared, no customer existed before Record, and Record produced a real CUS receipt plus committed ticket/customer/audit rows. Only the model transport was stubbed because no model key was configured; HTTP/auth/resolve/prepare/Record/receipt/PostgreSQL were real. No page errors or horizontal overflow. All 12 screenshots were inspected.
- Servers stopped. Every task DB used `fl_cust_rb_`; final query returned zero. `git diff --check` passed. No hosted migration, merge or deployment.

Python command: `env -u DATABASE_URL TEST_DB_PREFIX=fl_cust_rb_ TEST_DATABASE_URL=postgresql://localhost:5432/fl_cust_rb_py scripts/run_tests.sh`. Each fresh DB was created from template0 with UTF-8, pg_trgm and `tests/schema/schema.sql`. Office and real-page runs also applied migration 083 before starting the app. Browser commands are the repository `tests/visual/run-office.mjs`, `run-fl-assistant.mjs` and `run-fl-assistant-round-a.mjs`, with matching session flags. Raw passing logs, source hashes, real-page requests/summary/screenshots and temporary harness sources (synthetic local credentials replaced with placeholders) are saved beside this note. The first pushed evidence triggered the key-literal guard on two synthetic harness credentials; replacing them with placeholders fixed that evidence-only CI finding without changing application code or weakening the guard.
