# Receive lot capture validation

Latest [wording re-review 6322849012](https://github.com/sevenwells72/factory-ledger/pull/145#issuecomment-6322849012): the Floor RECEIVE rule and schema description now explicitly cover unreadable lots as well as no lot on label, using the exact proposed wording. The generated body is 7,860 characters (7,990 including header). All five matching contract tests pass, including real local preview/commit. This follow-up changes configuration, pinned text and documentation only; the full runtime suite results below are from 59cad84d.

`feat/receive-lot-required` starts from `origin/main` at `27a25186`. PR #145 remains unmerged. This report covers [review comment 6322477292](https://github.com/sevenwells72/factory-ledger/pull/145#issuecomment-6322477292).

Purchased ingredient receipts require the printed supplier lot or an explicit **No lot on label / Sin lote en la etiqueta** choice. Blank/sentinel codes without it return bilingual HTTP 422. Dashboard, ticket prepare/commit, legacy receive and preview/commit aliases share the guard. Existing purchased-ingredient exemptions, valid commingled codes, supplier links, legacy named-vendor handling, expected-receipt matching and normal receives are preserved. The explicit exception retains the existing unidentified status and seven-business-day New York deadline. No database migration is needed.

The CNS lot appears first and large on success, receipt read-back and Assistant cards, followed by **Write this on the pallet tag / Escribe esto en la etiqueta del pallet**. New receive Spanish consistently uses tú: Ingresa, selecciona, Escribe and di.

Review fixes:

- The actual Floor GPT schema, `gpt-configs/schemas/openapi-floor.yaml`, is v4.2.0. Its receive schema makes supplier_lot_code optional and exposes boolean no_supplier_lot with default false. The reviewed descriptions replace stale UNKNOWN/fallback advice. The RECEIVE line is exact in `floor-specific.md`; `shared-rules.md` has no RECEIVE line and remains unchanged. The root `build_gpt_instructions.py floor` rebuilt the dist file from both sources.
- Assistant no-lot facts work anywhere in EN/ES sentences: “10 bags oats 50 lb from Grain Millers, no lot on label”, “no lot”, “lot unreadable”, “No tiene lote”, “Sin lote” and “lote ilegible”. The receive tool binds worker facts even without a received verb. Supplier-choice turns retain them; real supplier codes take precedence; model-invented flags are refused; explicit denials clear the flag. The error says **say “no lot on label” / di “sin lote en la etiqueta”**, never select.
- The receipt changelog entry is now **229**. PR #144 reserves 225; PR #143 uses 226–228. Historical CHANGE_LOG allocation records remain intact.
- Office instructions are v3.8.2 with 515 characters of headroom. Routing and transaction rule areas remain intact.

Final review-fix validation on 2026-10-21:

| Check | Result |
|---|---|
| Floor contract tests, latest wording | 5 passed; zero failures/skips |
| Last full Python suite (59cad84d) | 7,094 passed; zero failures/errors/skips; 155.42 seconds |
| Last full JavaScript suite (59cad84d) | 577 passed; zero failures/skips |
| Last browser checks (59cad84d) | 12 passed; EN/ES at 390px and 1440px |
| Office GPT operations | 30 before / 30 after |
| Floor GPT operations | 22 before / 22 after |
| Floor deployable body | 7,860 characters |
| Floor file including generated header | 7,990 characters; below 8,000 |
| Office instructions | 7,485 characters; below 8,000 |
| Changed Floor receive descriptions | 128 and 195 characters; below 300 |
| Diff whitespace and changed Python syntax | Passed |

Python used pinned Python 3.12.14 / Pydantic 2.14.1 and fresh UTF-8 PostgreSQL 17 at `127.0.0.1:55471`, with DATABASE_URL removed and TEST_DATABASE_URL explicitly local. JavaScript ran `node --test tests/test_*.js`. New tests exercise the deployable Floor schema plus real no-label preview/commit and actual Assistant prepare→Record→unidentified exception/deadline paths. Existing tests retain strict boolean refusal, blank-code guards, normal receipts, supplier provenance, identity corrections and old-ticket commit revalidation.

Browser checks used mocked local APIs with all external requests blocked: 26px phone and 38px desktop CNS lot text above the receipt number, no overflow or console/page errors. Spanish screenshots now use Escribe. See [checks](browser-checks.json), [English receive success](entry-success-en-390.png), and [Spanish read-back](receipt-es-390.png).

Final caches: ledger-ui 19, fl-assistant 17, entry/receipt/ledger-ticket 10, assistant-display 12, ledger-forms/fl-assistant CSS 9. All changed-asset consumers were refreshed.

Future approved rollout checklist — not executed here:

1. Verify the office GPT editor holds the intended source version; the prepared artifact is v3.8.2. Live editor contents were not inspected or changed.
2. Paste the Floor v4.2.0 schema and rebuilt instructions **before** backend deployment. The current backend ignores the extra flag and retains its old blank-code fallback, avoiding a no-label outage during that order. Log an actual paste only after it happens.
3. Just before an approved deployment, check open receive drafts. Old blank/no-flag drafts intentionally fail Record with 409 SUPPLIER_LOT_REQUIRED and need re-preparing.
4. Legacy receives now report identity, create the existing unidentified exception and enforce correction-only mismatch guards. These are intentional, locally tested behaviors.

No production/staging access, manual deployment, hosted GPT paste, production migration, merge or push to main occurred. No live count worktree, audits/fresh-start/ or count/reset changes. Required logs used the verified canonical global symlink target and in-place writes.
