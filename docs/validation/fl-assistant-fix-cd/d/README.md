# FL Assistant C/D completion report — 2026-10-15

Round C was pushed to PR #133 at `cef6be9`, with a fix/test comment and `review:merge-ready-for-recheck`, then merged into main as `440b8c6` on 2026-10-15. Round D is rebased onto that main commit in `fix/assistant-orders-speed`; draft PR #134 now targets main. The rebase completed without conflicts and preserved the entire tested repository tree byte-for-byte before this documentation update.

No Railway, Supabase, OpenAI settings or AI model were changed. The real-model runs used the existing `gpt-6.1-sol` configuration. Product/supplier lookup remains separate from prepare.

## Item status

| Item | Status and result | Validation |
|---|---|---|
| C1 — real lot codes | Done. Literal and near-literal codes remain opaque; spoken digits normalize only after exact identity checks. | Every one of 536 real lot codes passed four normalization variants and exact API lookup in both modes, including hyphenated, OPEN, B and UNVERIFIED formats. |
| C2 — Classic stock | Done. Core Batch Classic Granola #9 ranks first with its direct finished products; one plain answer, positive stock by default, short choices for true ambiguity. | Exact English/Spanish transcripts against the original staging catalog, including exact IDs/SKUs and zero-stock requests. |
| C3 — unknown packing lot | Done. `lot one two three four` offers the chosen product's parent bulk lots in FIFO order; a selected exact lot stays pinned. | English/Spanish real-model transcripts and browser selection; physical tags remain unconfirmed. |
| C4 — Fruit & Nut / phone Menu | Done. Spanish alias resolves; send and Today close the phone Menu. | English/Spanish transcript and Chromium/WebKit phone checks. |
| D5 — open orders | Done. Exclude lines/orders with no effective remaining quantity; retain existing close rules. Display natural case units, e.g. `1 case (50 lb)`. | Original data shows three real open orders, excludes seven fully shipped STG orders; cancelled/service-line and case-weight tests. Those seven orders were eligible for closure and were closed separately under D8. |
| D6 — packing | Done. Ask “Which batch did these come from?” / “¿De qué lote a granel salieron?” with the correct parent FIFO lots and Different batch. Staff wording uses Bulk batch/Lote a granel. | Exact coconut transcripts, real parent-product choices and lot ordering; model-supplied identity never becomes tag evidence. |
| D7 — shipping | Done. English/Spanish spoken and spaced SO numbers resolve; Sunshine/Setton route through customer orders and short choices. Not-found stays an order question with the named customer's open orders. | Exact transcripts include voice, spaces, a following quantity, malformed literal codes and not-found customer orders. Four browser walks reach BOL/photo and stop before Record. Shipping all remaining requires an explicit worker button. |
| D8 — STG cleanup | Done. Closed seven fully shipped STG-customer orders through supported order tickets, with note `STAGING TEST CLEANUP` and standard reason `shipped_recorded`. | Receipts ORD-261015-010 through ORD-261015-016. No open STG orders remain. Protected SO-261015-001/002/003 hashes unchanged. See staging-cleanup.json. |
| D9 — Today | Done. Both chat paths use summary mode; return a 20-entry preview and View all. History retains counts rather than full receipt results. | 301-entry real-data page shows 0 own / 301 team, then View all shows 301. Saved response about 16.8 KB; history 429–461 bytes. A separate 601-entry regression preserves complete counts. |
| D10 — make balances | Done. Candidate lot metadata and posted balances load together in one grouped query across candidate ingredient products. | Before/after Classic and SS #1 requirements, plans, blockers and physical-confirmation groups match exactly. Final real-model allocations match: seven Classic and six SS #1 lot allocations. Fourteen focused allocation tests. |
| D11 — sign-in / reasons | Done. Verify live identity once per incoming request, pass a Python-only capability internally, recheck destination permissions, cache the reason list in memory. | Expired, revoked, rotated, inactive and wrong-role requests fail before work; forged HTTP headers cannot create capability. Office make denied without draft/ticket. Both complete modes pass. |
| D12 — prompt cache | Code done; cache-read improvement not observed. Dynamic date/time moved after the stable instruction prefix without changing meaning. | Existing provider reported 0% cached-token reads before and after, including three identical warmed probes. Each probe reported 4,411 cache-write tokens. No model or provider parameters changed. |
| D13 — progress | Done. Stream plain English/Spanish lookup, lot-check and draft status lines as work happens. | Actual uvicorn page receives and displays progress before the final result. Legacy JSON clients still work. |
| D14 — concurrency | Done. Run the synchronous DB work of a chat turn in a worker so it does not block the ASGI event loop. | Two actual concurrent chat requests meet at a DB-wait barrier and both complete; full tests cover error lease release/retry and both session modes. |

## Speed measurements

Same original staging dump, catalog, ledger and existing model before/after. SQL and AI counts cover a logical message, including explicit product/supplier button continuations where needed. Sign-in/chat creation is outside those counts. Warmup calls are separate.

| Message | DB queries before → after | AI calls before → after |
|---|---:|---:|
| Oats stock | 36 → 20 | 1 → 1 |
| Classic #9 stock | 78 → 37 | 1 → 1 |
| Receive | 82 → 49 | 4 → 3 |
| Make Classic #9 | 288 → 74 | 2 → 2 |
| Make SS #1 | 260 → 61 | 2 → 2 |
| Open orders | 26 → 15 | 1 → 1 |
| Found 10 lb oats | 34 → 18 | 2 → 2 |
| Spanish receive | 76 → 49 | 3 → 3 |
| Pack | 67 → 36 | 2 → 2 |
| Today's entries | 19 → 14 | 0 → 0 |
| **Total** | **966 → 373 (61.39% fewer)** | **18 → 17** |

Latency was measured against local PostgreSQL; it does not represent the hosted 62 ms network hop. The AI call difference can include model variation. The C baseline already used the fast Today shortcut, so the old 232-second detail path was not reproduced in this comparison. `speed.json` includes local timings and measurement notes. Cached-token read share is **0% → 0%**; no cache-hit speed gain is claimed.

## Tests and evidence

| Stage | Python, sessions off | Python, sessions on | Node |
|---|---:|---:|---:|
| C | 6,354 | 6,354 | 371 |
| D | 6,443 | 6,443 | 379 |

All passed, zero failures/errors/skips. Python ran on fresh local databases: 22 isolated groups per C mode and 23 per D mode, including committed races and destructive migration tests.

All Assistant browser runners pass in both modes: existing Assistant/customer, A 60, B contract 18, B actual page 68, C transcript 72 and phone 288 checks. Final D adds four real-model shipping walkthroughs and four Today walkthroughs (English/Spanish × both modes). Shipping stops before Record; no browser Record requests were sent. The strengthened B packing browser test explicitly chooses a FIFO batch and verifies physical tags remain unconfirmed.

C real-model acceptance includes 28 exact English/Spanish transcripts, 12 affected-transcript final-source rechecks and four actual phone-page checks. Final D has 40 exact transcript checks, with 64 real model calls. Exact D transcript texts and assertions are recorded in `exact-transcripts.json`; the committed tests include Michael's exact phone strings. All 536 real lot codes were retested in both stages and modes.

The original staging copy contained 422 products, 536 lots, 841 transactions and 301 entries for the tested day. No synthetic catalog was substituted. The local harness added only isolated authentication/chat/draft artifacts. Full contents of 12 ledger/catalog/order/trace tables remain identical across the original C/D copies after all preview tests. FIFO suggestions and traceability requirements are unchanged; no physical lot was auto-confirmed or skipped.

`verification.json`, Python group summaries, real-browser-verification.json, runtime-sha256.json, allocation evidence, original-copy-content.json and the cleanup receipts are included here. Source digests match the code that received the final real/browser runs.

## Cleanup and remaining work

The read-only staging dump was deleted. All task-owned `fl_fixCD_` databases were dropped, including the original C/D copies; none remain. The shared local PostgreSQL server remains running. See `local-cleanup.json`.

Hosted staging changes were limited to these supported order closures:

- SO-261012-001, SO-261012-003, SO-261012-004, SO-261012-005
- SO-261013-001, SO-261014-001, SO-261014-002

No real-customer order was changed. SO-261015-001/002/003 are unchanged. This task performed no migration, deployment or PR merge. D's rebase after C merged is complete; physical iPhone microphone testing remains outside automated transcript/browser coverage.

## Rebase verification after C merged

C merged as `440b8c652db3e520cf21468aed2f473f6b0ef186`. Replaying the single D commit produced `67a7fd5925870bd8d97577ebe6e828e3f9c56798` with no conflicts. Its complete tree hash, `5f7c84d7f674ca1f95926a8eaff92599f335a5f7`, equals the original tested `317d7db` tree. All 14 saved runtime hashes match; main ancestry and `git diff --check` pass. Only this status report, rebase evidence and append-only change logs were then updated.

No source, test, schema or configuration bytes changed, so the existing 6,443 Python tests in each session mode, 379 Node tests and full real-model/browser acceptance results remain applicable. No tests or model calls were repeated, no fresh staging dump was made and no test database was recreated. See `rebase-verification.json`.
