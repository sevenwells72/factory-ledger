# BB110927 trace findings — 2026-11-09

Investigated fresh `origin/main` **4e70ae34**, including Nov 9 PR #194 (`f84937ef`) and PR #63 (`c7e361ed`). Production health was healthy and the deployed GET responses contained the PR #194 input tree. All database inspection used `BEGIN; SET TRANSACTION READ ONLY;` (`SHOW transaction_read_only` returned `on`), ending in `ROLLBACK`. No historical data was changed.

## Evidence and classification

Finished product **145**, lot **1723 / BB110927**, has pack **2987**: **3,855 lb** output, **3,855 lb** remaining and no posted shipments at inspection. Inputs are batch lots 1681 / NOV 02 2026 (523.5 lb), 1690 / B26-1103-002 (786 lb), 1696 / NOV 04 2026 (2,358 lb), and 1709 / NOV 06 2026 (187.5 lb).

| Symptom | Classification | Evidence / required fix |
|---|---|---|
| NOV 04 2026 / txn 2947 shows no ingredients | Report-rendering bug, not missing data | Ledger AND deployed input_tree contain oats 1,374; honey 282; chocolate chips 228; canola 132; coconut 114; almonds 114; sunflower 114 lb. Their sum is 2,358 lb. All are shared with other batches. Graph nodes are deduplicated by lot ID; detail/text export iterates those nodes, so it loses per-batch associations. Report transaction-to-lot edges independently of graph deduplication. |
| Shared oats shown only with first batch | Report-rendering bug | addNode keeps the first node detail/quantity and addTreeInputs skips subsequent expansions. The graph has links, but the exported node list retains first-use metadata. List every consuming transaction and parent batch. |
| Whole-batch amounts called used by this finished lot | Report-rendering / quantity interpretation bug | NOV 02 batch has makes 2926 (786 lb) + 2927 (1,965 lb), total 2,751 lb; this pack consumes only 523.5 lb. NOV 06 make 2965 produces 3,144 lb; this pack consumes 187.5 lb. Preserve recorded whole-make inputs; show proportional lot-share estimates separately. Same-code lots pool multiple makes; those estimates are assumptions, never physically verified assignments. |
| Legacy Gaps: 0 despite two found-inventory gaps | Report-rendering bug plus real origin gaps | traceBackwardTree hardcodes legacyCount=0; text export prints it. Coconut lot 616 / 25120 and honey lot 1653 / 26-10-27-FOUND-006 have no posted supplier receipts and are marked found inventory. Count actual unique gaps consistently; keep them visible. |
| Nation Harvest / National Harvest | Stored naming discrepancy, not a query/rendering bug | Receipts 2201 (sunflower lot 1398) and 2442 (almond lot 1477) carry free-text shipper Nation Harvest with no supplier_id. Receipt 2955 (almond lot 1702) references supplier 36, National Harvest. Only one matching suppliers-table record exists, not two supplier records. Report both recorded names; no merge or rewrite. |

## Forward trace: oats 26-10-22-QUAL-001

Product **32**, lot **1637**, supplier lot **261019**. Posted current ledger contains **17 makes** across **14 distinct output lots**: 2910, 2914, 2922, 2926, 2927, 2935, 2936, 2947, 2948, 2965, 2980, 2981, 2982, 2983, 2984, 2985, 2986. Voided make 2903 must be excluded. Remaining oats: **32,702 lb**.

The deployed `/trace/ingredient/26-10-22-QUAL-001?product_id=32` returns **16 rows**, dropping one of makes **2980 / 2985** (both consume 450 lb into output lot 1716). SQL DISTINCT omits transaction_id, so identical separate makes collapse. This is a **trace-query bug**. It also reads ingredient_lot_consumption rather than corrected current lines, so amended quantities/lot identities can diverge (a structural bug even where this sample agrees).

The old forward UI initially shows only five of these rows and looks only for direct batch shipments; it does not follow pack consumption through finished lots to customers. A complete forward report must traverse all posted make/pack edges by numeric lot ID, preserve each transaction, and enumerate every reachable shipment and remaining lot. Downstream amounts describe full potentially exposed lots, not a physically verified allocation of the starting ingredient.

## Implementation boundaries

Read-only API and UI; no migrations, no changes to openapi-gpt-v3.yaml, no deploy, no merge, no production writes. Keep numeric lot identity and product disambiguation, all posted current-ledger semantics, found/unknown origin warnings, units, and the existing mismatch check. Print/PDF and text must use the same complete report data. Claude Code cross-review is required before merge.

## Candidate verification (read-only, same day)

The revised trace returns all **17** posted oats makes (including both 2980 and 2985), **20** reachable lots, **25** make/pack edges and **3** shipments. BB110927 retains all four bulk lots and every ingredient association: **41** input rows, including all **7** for make 2947, with **2** unique origin gaps.

Additional query/rendering bug discovered during candidate verification: product 145 has catalog UOM `12x10 oz case`, but ledger make/pack quantities are pounds. The earlier input-tree helper treated that packaging label as a count unit, producing an input-tree output of zero despite the top-level 3,855 lb. Trace-specific unit handling now treats batch/finished weights as pounds and preserves count units for packaging supplies. The refreshed read-only response has **3,855 lb** at both levels.

Estimated oats shares, explicitly assuming proportional mixing: make 2926 **87.1549 lb** and make 2927 **217.8871 lb** (each recorded amount multiplied by 523.5 / 2,751); make 2936 **458 lb**; make 2947 **1,374 lb**; make 2965 **109.2557 lb** (1,832 × 187.5 / 3,144). These estimates do not establish physical lineage among repeated makes sharing a lot code. Recorded whole-make amounts remain alongside them.

Additional guard: commingled lots retain every recorded `lot_supplier_codes` entry in history and both trace directions. These lot-level supplier/code/quantity records are displayed separately from receipts and labelled as recorded metadata, not receipt allocation or physical verification. A commingled identity or recorded pooled note suppresses any claim that the inventory is one supplier lot.

## Final local validation

Fresh disposable local PostgreSQL: **7,469 Python tests passed**, zero failures/skips. Full JavaScript suite: **700 passed**, zero failures/skips. Real HTTP/FastAPI/PostgreSQL Playwright coverage passed at 1440, 820 and 390 px in light/dark themes: **138 actual requests carried X-API-Key**, including inventory, lot history, the existing lot panel and trace endpoints; no route stubs. Both directions, zero balances, repeated makes, customer/order details, text export, browser Print/Save PDF, touch targets and responsive layouts were exercised. The existing backward-trace browser regression also passed. PDF pages were rendered and visually inspected. Architecture-map, key-literal and whitespace checks passed. Existing route contracts are unchanged except the one new authenticated GET route.

The full suite was rerun on a freshly rebuilt local database after intermediate source edits and a repeat-run fixture-count failure; the final result above is from the stable final code. No production writes, merge or deploy occurred. Claude Code cross-review remains required before merge.

## PR #196 cross-review fixes

Review [6755271486](https://github.com/sevenwells72/factory-ledger/pull/196#issuecomment-6755271486) found correct answers but unacceptable forward-query cost. The complete report now lives only at `GET /dashboard/api/lot-report/{lot_code}` (explicit direction/product, authenticated). Existing `/trace/batch`, `/trace/ingredient`, `/trace/supplier-lot` and their helpers match `origin/main` `4e70ae34` byte-for-byte. In particular, GPT `ingredients` retains `ingredient_lot_consumption` per row 270. Earlier implementation notes describing corrected legacy responses are superseded: corrected current-ledger inputs and complete exposure are used only by the explicit audit reports.

Forward closure uses a recursive CTE with cycle-safe UNION and no depth/date/fanout cap. Four further set-based queries fetch balances/lot metadata, receipts, source entries and shipments/order references: **five queries independent of lot count**, or six with endpoint lookup. Tests pin five at 1, 80 and 500 output lots. Backward audit logic is isolated in `lot_trace_backward.py`.

The page accepts only the versioned complete shape. Missing or incomplete formats show clear errors without legacy answers. A visible building state and report-only 120-second timeout support large complete results; a browser check deliberately delays a report 16 seconds. Multi-output consumption is numeric only on its first row; subsequent rows say **Same consumption as above, do not add** in screen, print/PDF and text. The actual-network browser test now runs in `node --test tests/*.js` and `npm test`, observes protected API headers and requires a successful `/audit/integrity` request.

Initial read-only measurements from this Mac, including transaction setup, SQL round trips/result transfer, report construction and JSON serialization (excluding connection establishment):

| Lot | Reachable lots | Edges | Three measurements |
|---|---:|---:|---|
| Coconut 25120, product 13 | 277 | 401 | 0.371 / 0.252 / 0.238 seconds |
| 26-02-03-FOUND-016, product 39 | 481 | 686 | 0.281 / 0.294 / 0.243 seconds |

Each used six endpoint queries and included the entire result. Oats remained 17 makes / 20 lots / 25 edges / 3 shipments. BB110927 remained four batch lots with make input counts 7, 7, 8, 7, 8; all seven recorded inputs for 2947 independently matched the ledger; report model has 41 input rows, two gaps and 3,855 lb. Reproduce with `scripts/benchmark_lot_reports_readonly.py`; it uses port 5432 and explicit `BEGIN; SET TRANSACTION READ ONLY; ... ROLLBACK`, never app startup or production writes.

### Final review-fix verification

Both full suites ran against separately rebuilt local test databases: **7,474 Python passed** and **702 JavaScript passed**, zero failures/skips. The default JS suite includes the non-stubbed browser test: **270 actual HTTP requests carried X-API-Key**, including successful health, inventory, lot panel, history and complete report requests. The updated backward-report browser suite passed, including a 16-second response, missing/incomplete shape errors and clearing previous evidence when the next report fails. An old test that required the removed scanner implementation now checks real new-report shipment isolation for identical lot codes across products; its original GPT-response assertion remains.

Final production read-only repeats, after both full suites:

| Lot | Complete lots / edges | Times (seconds) | Endpoint queries |
|---|---|---|---:|
| 25120 / product 13 | 277 / 401 | 0.769, 0.268, 0.262 | 6 |
| 26-02-03-FOUND-016 / product 39 | 481 / 686 | 0.326, 0.325, 0.291 | 6 |

Both worst cases remain below five seconds; no results are capped or truncated. Oats again verified **17 makes / 20 lots / 25 edges / 3 shipments** (0.226 seconds). BB110927 backward again verified **four batches, full 7/7/8/7/8 make inputs, 41 associations, two gaps, 3,855 lb** (3.035 seconds); transaction 2947 matched an independent posted-ledger query. The corrected backward reader is separate and still uses recursive per-origin reads; the fixed five-query invariant applies to the forward traversal. Machine-readable evidence: `lot-report-performance-2026-11-09.json`.

The existing trace section remains byte-for-byte identical to `origin/main` `4e70ae34`. Contract comparison adds only the two dashboard GET routes and removes/changes no existing routes. Architecture-map, key-literal and whitespace checks pass. All three changelogs updated (regression row 272). Claude Code re-review is still required before merge; no merge, deploy or production write was performed.
