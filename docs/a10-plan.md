# A10 / O-14: retire direct writes

Planning only. Commissioned for the overnight review on **2026-10-14, America/New_York**. Source snapshot: `origin/main` commit `5592cdc`, inspected in `~/dev/fl-a10-wt` on `docs/a10-plan`. Existing code was read, not changed. No application import, database connection, test run, or staging/production write was used to produce this inventory. Deployment state and recent production usage have not been verified.

Michael's requested stage-1 date is **Monday, November 9, 2026**. Repository planning still places cutover on November 20 and A10 on November 23 (`FOLLOWUPS.md`, P1.6). This is an unresolved schedule conflict: the November 9 target does not establish that cutover prerequisites will already be complete. Stage 2 has no approved date in this request.

## Scope and count

There are **64 direct business-write HTTP method/path pairs** in this source snapshot: 63 in `main.py` and one direct exception-resolution route. This deliberately broad count includes inventory/transaction writes, order changes, lot/catalog maintenance, certifications, planning, supply requests, and notes. It prevents a narrow inventory-only definition from silently leaving other business writes outside the requested “prepare → commit → receipt” rule. Each HTTP alias counts separately; the five base inventory routes count once each even though their bodies accept preview and commit modes.

The narrower subtotal is **19 entrances capable of changing inventory/transaction or effective ledger state**: B03–B12, B14–B20, B38 and B50. These comprise 16 posting/void/correction entrances (including commit aliases), lot reassignment, lot merge, and direct exception resolution that can post a count adjustment. The other **45** change lot metadata, orders/customers/expected receipts, catalog/certifications, planning/allocations, supply requests or notes. This is not a claim of 64 separate inventory-posting operations.

Three document upload/extraction routes and other modern support/control routes are listed separately below. They write storage or workflow state and must receive an explicit cutover decision; they are not silently exempt. Existing ticket preparation, ticket commit, and approval of a held ticket are not legacy bypasses. Six forced-preview aliases and three lookup POSTs do not perform business writes and are excluded from 64. Authentication can update usage/session metadata even on GETs; that is also called out below.

This is a source inventory, not a claim that all 64 are in active production use. “No ticket equivalent” below means no equivalent registered HTTP prepare route was found at this commit. A replacement must be built and verified, the operation must be retired, or Michael must explicitly change the scope before closure can preserve that operation.

## Authorization facts

Table abbreviations: **M** = server master `API_KEY`; **D** = shared `DASHBOARD_API_KEY`; **P/S** = active personal actor key or enabled actor session. **No route in the 64-row inventory accepts the read-only key.** P/S means authentication can reach the route, subject to role/action and business guards; it does not mean every person can perform every operation. Sessions require the existing session feature to be enabled. Values are neither needed nor included here.

The source of truth is `main.py:2762` (dashboard allowlist), `main.py:2878` (read-only allowlist), `main.py:2936` (actor allowlist), and `main.py:3249` (authorization order). Master comparison returns before other key kinds; closing routes must therefore apply to **all** key kinds, including master, rather than only deleting actor/dashboard grants. Personal/session action checks are in `permissions.py:153` onward. Shared-key authentication passing is not a promise that a later business guard permits an operation; for example, closing an order as shipped without a shipment has owner-specific shared-core checks.

Personal/session roles: receive is available to all named roles; make/pack/adjust/found/void exclude office; standalone ship excludes floor; order creation/edits/cancel/close and customer management exclude floor; ready permits all named roles; reopen is owner-only. Exception resolution permits floor/owner with additional kind-specific owner rules. Planning, allocations, suppliers, notes, date repair, supply requests and intake staging are explicitly ungated named-actor route exemptions today (`permissions.py:202`–`238`). A personal key does **not** replace master access to a master-only catalog/admin route in this snapshot.

## Direct business-write inventory (64)

Line references identify route decorators at the snapshot, not deployment line numbers. All proposed replacements use `POST /tickets/{ticket}/commit` after the listed prepare route, with the returned receipt checked. “Close” means disable the external legacy HTTP entry point; shared implementation functions remain needed by ticket handlers.

### Inventory, lots, and transaction corrections (20)

| ID | Method and path | Source | What it writes | Accepted auth | Ticket replacement or gap |
|---|---|---|---|---|---|
| B01 | PATCH `/lots/{lot_code}/supplier-lot` | `main.py:5199` | Lot supplier code and supplier-code association; actor audit | M, P/S | `/lots/{lot_id}/supplier-lot/prepare`; resolve the stable lot ID first; named actor required there |
| B02 | PATCH `/lots/{lot_id}/rename` | `main.py:5267` | Lot code and actor audit | M, P/S | No rename prepare route found; build or retire |
| B03 | POST `/receive` | `main.py:5779` | Receive transaction/lines, lots, supplier-lot data and expected-receipt settlement when committing | M, P/S | `/receive/prepare` |
| B04 | POST `/ship` | `main.py:8557` | Standalone shipment, transaction and inventory lines when committing | M, P/S | `/ship/prepare`; `/ship/bulk/prepare` for the separate bulk workflow |
| B05 | POST `/make` | `main.py:9677` | Make transaction, ingredient consumption, output lot and trace when committing | M, P/S | `/make/prepare` |
| B06 | POST `/pack` | `main.py:9809` | Pack transaction, consumption and output lot when committing | M, P/S | `/pack/prepare` |
| B07 | POST `/adjust` | `main.py:9842` | Adjustment transaction/lines and inventory adjustment when committing | M, P/S | `/adjust/prepare` |
| B08 | POST `/void/{transaction_id}` | `main.py:10136` | Append-only void correction and related allocation effects; internally creates VD ticket/receipt | M, P/S | `/void/{transaction_id}/prepare`; existing inline receipt does not provide client prepare/review |
| B09 | POST `/records/transactions/{transaction_id}/corrections` | `main.py:10189` | Amend, void or restore correction and related trace/allocation effects | M | Void has `/void/{transaction_id}/prepare`; amend/restore have no equivalent prepare route found |
| B10 | POST `/lots/{lot_id}/reassign` | `main.py:11505` | Lot product identity, append-only line corrections and reassignment history | M | No equivalent prepare route found |
| B11 | POST `/inventory/found` | `main.py:11647` | Found-inventory transaction/lot/adjustment | M | `/inventory/found/prepare` for an existing product |
| B12 | POST `/inventory/found-with-new-product` | `main.py:11665` | New product plus found transaction/lot/adjustment | M | Found prepare does not cover creating the product; product creation is a gap |
| B13 | PATCH `/lots/{lot_id}/received-at` | `main.py:15462` | Lot receipt date used by age/FIFO views | M, D, P/S | No equivalent prepare route found |
| B14 | POST `/sales/orders/{order_id}/ship` | `main.py:16170` | Order shipment, transaction/lines, shipment/order/allocation effects when committing | M, P/S | `/sales/orders/{order_id}/ship/prepare` |
| B15 | POST `/receive/commit` | `main.py:16303` | Forces direct receive commit | M, D, P/S | `/receive/prepare` |
| B16 | POST `/ship/commit` | `main.py:16313` | Forces direct standalone ship commit | M | `/ship/prepare` |
| B17 | POST `/make/commit` | `main.py:16323` | Forces direct make commit | M | `/make/prepare` |
| B18 | POST `/pack/commit` | `main.py:16333` | Forces direct pack commit | M | `/pack/prepare` |
| B19 | POST `/adjust/commit` | `main.py:16343` | Forces direct adjust commit | M | `/adjust/prepare` |
| B20 | POST `/sales/orders/{order_id}/ship/commit` | `main.py:16360` | Forces direct order ship commit; named `commitShipOrder` in OpenAPI | M, D, P/S | `/sales/orders/{order_id}/ship/prepare` |

### Orders, customers, and expected receipts (18)

| ID | Method and path | Source | What it writes | Accepted auth | Ticket replacement or gap |
|---|---|---|---|---|---|
| B21 | POST `/suppliers` | `main.py:6096` | Supplier master row | M, D, P/S | No supplier creation prepare route found |
| B22 | POST `/expected-receipts` | `main.py:6161` | Expected receipt | M, D, P/S | `/expected-receipts/prepare` |
| B23 | PATCH `/expected-receipts/{expected_receipt_id}` | `main.py:6334` | Expected-receipt fields or cancellation | M, D, P/S | `/expected-receipts/{expected_receipt_id}/update/prepare` or `/cancel/prepare` |
| B24 | POST `/expected-receipts/extract/approve` | `main.py:7493` | Expected receipts and source-document approval state | M, D, P/S | `/expected-receipts/extract/approve/prepare` |
| B25 | POST `/sales/orders/extract/approve` | `main.py:7704` | Sales order/lines and source-document approval state | M, D, P/S | `/sales/orders/extract/approve/prepare` |
| B26 | POST `/customers` | `main.py:12025` | Customer and actor audit | M, P/S | No customer creation prepare route found; order prepare's customer handling is not a general customer-maintenance substitute |
| B27 | PATCH `/customers/{customer_id}` | `main.py:12050` | Customer fields, aliases and actor audit | M, P/S | No equivalent prepare route found |
| B28 | POST `/sales/orders` | `main.py:12398` | Order/lines; may create customer; stores creation-idempotency receipt | M, P/S | `/sales/orders/prepare`; historical idempotency receipt is not the prepare-review-commit contract |
| B29 | POST `/sales-orders/{so_number}/ready` | `main.py:14171` | Order ready flag | M, D, P/S | `/sales/orders/{order_id}/ready/prepare`; resolve ID; note legacy hyphenated path |
| B30 | POST `/sales/orders/{order_id}/close` | `main.py:14786` | Administrative close and related allocation/audit effects | M, D, P/S | `/sales/orders/{order_id}/close/prepare`; preserve shared-core close reason/owner guards |
| B31 | POST `/sales/orders/{order_id}/cancel` | `main.py:14874` | Cancellation and allocation/audit effects | M, D, P/S | `/sales/orders/{order_id}/cancel/prepare` |
| B32 | POST `/sales/orders/{order_id}/reopen` | `main.py:14949` | Reopen state and related checks | M, D, P/S | `/sales/orders/{order_id}/reopen/prepare`; owner action |
| B33 | PATCH `/sales/orders/{order_id}/status` | `main.py:15655` | Order status and related checks/effects | M, D, P/S | `/sales/orders/{order_id}/status/prepare`; keep closed-shipped reason checks in shared core |
| B34 | PATCH `/sales/orders/{order_id}` | `main.py:15804` | Order header fields | M, D, P/S | `/sales/orders/{order_id}/header/prepare` |
| B35 | POST `/sales/orders/{order_id}/lines` | `main.py:15828` | New order lines and actor audit | M, P/S | `/sales/orders/{order_id}/lines/prepare` |
| B36 | PATCH `/sales/orders/{order_id}/lines/{line_id}/cancel` | `main.py:15964` | Line cancellation and related allocation effects | M, P/S | `/sales/orders/{order_id}/lines/{line_id}/cancel/prepare` |
| B37 | PATCH `/sales/orders/{order_id}/lines/{line_id}/update` | `main.py:16123` | Line quantity/price and related checks | M, D, P/S | `/sales/orders/{order_id}/lines/{line_id}/update/prepare` |
| B38 | POST `/exceptions/{exception_id}/resolve` | `exceptions_enforcement.py:1031` | Resolve/waive exception, shortage evidence and possibly count adjustment; internally creates XR/ADJ receipts | P/S | `/exceptions/{exception_id}/resolve/prepare`; named floor/owner with kind-specific restrictions |

### Catalog, certifications, and administrative corrections (12)

| ID | Method and path | Source | What it writes | Accepted auth | Ticket replacement or gap |
|---|---|---|---|---|---|
| B39 | POST `/records/certifications` | `main.py:10847` | Certification row | M | No equivalent prepare route found |
| B40 | POST `/records/certifications/{certification_id}/corrections` | `main.py:10884` | Append-only certification correction/new certification version | M | No equivalent prepare route found |
| B41 | POST `/products/quick-create` | `main.py:11400` | Unverified product and verification history | M | No equivalent prepare route found |
| B42 | POST `/products/quick-create-batch` | `main.py:11456` | Batch product with production context | M | No equivalent prepare route found |
| B43 | POST `/products/{product_id}/verify` | `main.py:11813` | Verify/reject/archive product, optional name, verification history | M | No equivalent prepare route found |
| B44 | PUT `/admin/products/{product_id}` | `main.py:18204` | Product master fields | M | No equivalent prepare route found |
| B45 | POST `/admin/bom/{product_id}/lines` | `main.py:18325` | Batch-formula line | M | No equivalent prepare route found |
| B46 | PUT `/admin/bom/lines/{line_id}` | `main.py:18360` | Batch-formula line fields | M | No equivalent prepare route found |
| B47 | DELETE `/admin/bom/lines/{line_id}` | `main.py:18406` | Deletes batch-formula line | M | No equivalent prepare route found |
| B48 | POST `/admin/product-bom` | `main.py:18477` | Finished-product component mapping | M | No equivalent prepare route found |
| B49 | DELETE `/admin/product-bom/{mapping_id}` | `main.py:18509` | Deletes product-component mapping | M | No equivalent prepare route found |
| B50 | POST `/admin/lots/merge` | `main.py:18587` | Append-only line corrections, consumption references, lot merge state, allocations and trace markers | M | No equivalent prepare route found; do not mistake this for harmless catalog metadata |

### Planning, allocation, supply requests, and notes (14)

| ID | Method and path | Source | What it writes | Accepted auth | Ticket replacement or gap |
|---|---|---|---|---|---|
| B51 | POST `/supply-requests` | `main.py:7895` | Supply request | M, D, P/S | No equivalent prepare route found |
| B52 | PATCH `/supply-requests/{supply_request_id}` | `main.py:7971` | Marks supply request done | M, D, P/S | No equivalent prepare route found |
| B53 | POST `/sales/orders/{order_id}/allocations` | `main.py:15209` | Stock allocation and allocation audit | M, D, P/S | No equivalent prepare route found |
| B54 | POST `/sales/orders/{order_id}/allocations/{allocation_id}/release` | `main.py:15388` | Allocation release and audit | M, D, P/S | No equivalent prepare route found |
| B55 | POST `/dashboard/api/notes` | `main.py:18078` | Note and actor audit | M, D, P/S | No equivalent prepare route found |
| B56 | PUT `/dashboard/api/notes/{note_id}` | `main.py:18105` | Note fields and actor audit | M, D, P/S | No equivalent prepare route found |
| B57 | DELETE `/dashboard/api/notes/{note_id}` | `main.py:18150` | Deletes note and records actor audit | M, D, P/S | No equivalent prepare route found |
| B58 | PUT `/dashboard/api/notes/{note_id}/toggle` | `main.py:18168` | Note status and actor audit | M, D, P/S | No equivalent prepare route found |
| B59 | POST `/production/runs` | `main.py:19687` | Planned production run | M, D, P/S | No equivalent prepare route found |
| B60 | PATCH `/production/runs/{run_id}` | `main.py:19717` | Run fields/status | M, D, P/S | No equivalent prepare route found |
| B61 | POST `/production/runs/{run_id}/cancel` | `main.py:19783` | Run cancellation | M, D, P/S | No equivalent prepare route found |
| B62 | POST `/production/runs/{run_id}/complete` | `main.py:19869` | Run completion status/stamps; evidence is only returned, not a gate | M, D, P/S | No equivalent prepare route found |
| B63 | PUT `/production/runs/{run_id}/coverage` | `main.py:19924` | Replaces run-to-order coverage and touches run metadata | M, D, P/S | No equivalent prepare route found |
| B64 | POST `/schedule` | `main.py:20852` | `action=confirm` upserts `production_schedule` (`main.py:20716`); suggest/current branches read | M | No equivalent prepare route found; close confirm, preserve verified read actions only through an explicit allowlist or separate read route |

## Support/control writes: explicit cutover decisions required

These routes are outside the 64 legacy business-entry-point count. Their state changes still matter to the requirement that every write has a ticket and receipt. The proposed disposition is to keep necessary transport and existing ticket lifecycle operations available, while Michael decides whether support metadata is an explicit exception. If the requirement literally includes every storage/session/evidence write, missing prepare/receipt contracts are blockers, not permission to leave these writes undocumented.

| Method and path | Source | Auth | State written / disposition |
|---|---|---|---|
| POST `/expected-receipts/extract` | `main.py:7318` | M, D, P/S | Upload/source-document storage and intake state only; approval is B24. Needed before document approval prepare. Explicit support exemption or new receipt contract required |
| POST `/sales/orders/extract` | `main.py:7330` | M, D, P/S | Sales-document upload/source state only; approval is B25. Same decision |
| POST `/purchase-documents/{document_id}/extract` | `main.py:7342` | M, D, P/S | Re-extraction updates stored document/extraction state via `_run_extraction_and_store`; same decision |
| POST `/attachments` | `attachments.py:172` | P/S; named actor required even though master passes general auth | Stores evidence bytes and attachment record, optionally bound to actor's ticket; prerequisite for evidence-gated commits. No independent prepare/receipt |
| POST `/shipments/{shipment_id}/attachments` | `attachments.py:213` | P/S, uploader/ownership plus shipment context/office-owner rules | Links evidence; a BOL photo can directly mark shipment proof complete and resolve BOL-photo exception (`attachments.py:149`–`169`). This changes business evidence, not just upload metadata; needs explicit ticket/receipt coverage decision |
| POST `/reports/shift-summary/confirm` | `shift_summary.py:1001` | P/S only; shared keys rejected by handler | Inserts shift confirmation and creates/updates discrepancy exception after summary-hash check. No prepare/commit receipt route; decide contract before closure |
| POST `/exceptions/{exception_id}/reject` | `exceptions_enforcement.py:1135` | Owner P/S | Rejects an already-held ticket and records XR decision receipt inline. Preserve owner rejection lifecycle; confirm whether an additional prepared decision is required |
| POST `/exceptions/{exception_id}/approve` | `exceptions_enforcement.py:1093` | Owner P/S | Calls `write_tickets.execute_commit` on an existing held ticket with current-role/staleness checks and receipt; **not a direct posting bypass**. Preserve |
| POST `/auth/session` | `key_sessions.py:90` | Personal key supplied for exchange; shared keys refused; enabled feature required | Records successful/failed sign-in attempts and creates hashed actor-session record. Necessary auth infrastructure, not ledger posting; explicit metadata scope |
| POST `/auth/session/logout` | `key_sessions.py:134` | Session only | Ends actor session; preserve as auth infrastructure |
| POST `/assistant/session` | `fl_assistant.py:267` | P/S only | Creates assistant session; transport state |
| POST `/assistant/turn` | `fl_assistant.py:394` | P/S only | Stores turns, session lease/state, drafts; relays permitted prepare/read actions, not unrestricted direct commits |
| POST `/assistant/confirm-sku` | `fl_assistant.py:369` | P/S only | Prepares corrected draft and cancels superseded draft; transport plus ticket preparation |
| POST `/assistant/record` | `fl_assistant.py:507` | P/S only | Relays `/tickets/{ticket}/commit`, validates receipt, updates assistant draft/result; **preserve** ticket transport |
| POST `/assistant/cancel` | `fl_assistant.py:553` | P/S only | Cancels assistant draft with pending-outcome guards; preserve transport |
| POST `/assistant/attachment` | `fl_assistant.py:564` | P/S only | Relays evidence upload and records session/attachment link; same evidence decision as above |

`/assistant/resume` (`fl_assistant.py:274`), `/assistant/attachment/read` (`:588`), and `/assistant/transcribe` (`:607`) do not directly post ledger/business state; retain their documented read/transport behavior. Transcription calls an external service. Their authentication may still refresh session metadata.

Read-side metadata writes are observable: `_touch_actor_last_used` updates `actors.last_used_at`, throttled to ten minutes (`main.py:3133`–`3178`, called at `:3287`); session authentication can extend expiry or end a revoked/expired session (`key_sessions.py:41`–`81`). No additional direct business-write GET handler was found by static inspection. A “reads stay open” acceptance test must distinguish these metadata updates from ledger mutations rather than asserting every authenticated GET performs zero SQL writes.

## Reads, previews, and ticket routes that remain available

Keep authenticated/public GET reads with current authorization; the 16-entry `READONLY_KEY_ALLOWLIST`; read-only POST `/products/resolve` (`main.py:4707`), `/resolve` (`resolution.py:696`), `/expected-receipts/match` (`main.py:7381`) and `/sales/orders/match` (`main.py:7528`); receipt lookup; and the forced-preview aliases below. Do not assume all POSTs write or all authenticated GETs avoid metadata writes.

| Forced-preview route (POST) | Source | Auth today |
|---|---|---|
| `/receive/preview` | `main.py:16298` | M, D, P/S |
| `/ship/preview` | `main.py:16308` | M |
| `/make/preview` | `main.py:16318` | M |
| `/pack/preview` | `main.py:16328` | M |
| `/adjust/preview` | `main.py:16338` | M |
| `/sales/orders/{order_id}/ship/preview` | `main.py:16348` | M, D, P/S |

These wrappers forcibly set `mode="preview"` and then call the corresponding legacy Python handler. A guard placed inside that shared handler, or deleting it in stage 2 without extracting a preview service, can accidentally break preserved reads. Gate the registered external write route by method/template, and retain/refactor shared preview helpers independently. Closing the mixed-mode base paths also closes their historical preview mode; callers must use the explicit preview paths.

M0 was checked without entering its worktree: `git show origin/feat/mcp-readonly` at commit `188def49e6f1704909311c42f2df8d71f6bcff72` shows `mcp_server/src/factory_ledger_mcp/adapter.py:38`–`42` allowing `shipOrder` only at `/sales/orders/{order_id}/ship/preview`; `:98`–`103` forces preview mode. Its catalog points to that alias at `mcp_server/src/factory_ledger_mcp/catalog.json:814` and `:1403`; `source_path` naming the base route is schema provenance, not the request destination. Preserve that alias and verify the **deployed** M0 revision before closing base ship. A read-only plugin identity is separate from the narrowly scoped `READONLY_API_KEY`: that key does not currently allow the ship-preview alias. Do not switch the plugin's backing credential to it without an explicit scope design and tests.

Preserve the ticket route families and shared cores:

| Family | Registered prepare/read sources |
|---|---|
| Receive, make, pack, adjust, found, move lot, supplier-lot correction; generic commit; receipt reads | `write_tickets.py:754`–`882` |
| Order ship, standalone ship, bulk ship | `ship_tickets.py:363`–`369` |
| Order create, lines, line edit/cancel, header/status/ready, cancel/close/reopen; office options | `order_tickets.py:745`–`830` |
| Expected receipt create/edit/cancel and intake approval | `expected_receipt_tickets.py:168`–`182` |
| Sales-document intake approval | `sales_document_tickets.py:69` |
| Exception resolution and void | `resolution_receipts.py:140`–`153` |

`resolution_receipts.py:23`–`63` explains why direct void and direct exception resolution can already return receipt numbers: they issue/finish a ticket inside the same request when no prepared ticket was supplied. That is useful audit history but does not meet the requested two-step client review contract. Their prepared endpoints and generic commit must survive; the direct entrances close.



## Caller inventory and what breaks

“Breaks” means the checked-in caller receives the closure response if used unchanged. It does not establish that the caller is deployed or was used recently. Source review covered tracked Python, JavaScript, HTML, shell, YAML, Markdown and configuration files, including dynamic route builders and archived schemas. There are no tracked `.github/workflows/` files at this snapshot. Production logs, GPT editor contents, remote scheduler settings and other machines remain unverified.

### GPT Actions

The legacy office schema still exposes **17 business-write operations**; the floor schema exposes **11**, including its separate order-ship commit action. Each of the following will fail after closure if an editor still has that legacy schema. The replacement read-only files exist, but their existence does not prove the two GPT editors were updated.

| Business route IDs | Office schema evidence (`openapi-gpt-v3.yaml`) | Floor schema evidence (`gpt-configs/schemas/openapi-floor.yaml`) |
|---|---|---|
| B01 supplier lot | `:501` | `:429` |
| B02 rename | — | `:460` |
| B03 receive | `:532` | `:485` |
| B04 ship | `:562` | `:499` |
| B05 make | `:576` | `:515` |
| B06 pack | `:590` | `:529` |
| B07 adjust | `:604` | `:543` |
| B08 void | — | `:559` |
| B14 order ship | `:1009` | `:720` |
| B20 order ship commit | — | `:748` |
| B22 expected receipt create | `:546` | — |
| B26/B27 customer create/update | `:738`, `:765` | — |
| B28 order create | `:812` | — |
| B33 order status | `:893` | `:697` |
| B34 order header | `:862` | — |
| B35 add lines | `:913` | — |
| B36 cancel line | `:951` | — |
| B37 update line | `:970` | — |

Current instruction callers are `gpt-instructions-v3.md:45,50,55,67,82`, `gpt-configs/sources/shared-rules.md:23`, `gpt-configs/sources/floor-specific.md:4,10,21,26,32,34,37,45,50`, and generated `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md:27,46,52,63,68,74,76,79,87,92`. Update source instructions before regenerating; do not hand-edit generated output. `gpt-configs/README.md:31` describes the editor-copy workflow. The intended G1 replacements are `gpt-configs/readonly/openapi-office-readonly.yaml`, `openapi-floor-readonly.yaml`, `office-instructions.md`, `floor-instructions.md`, and `CUTOVER_CHECKLIST.md` in that directory. They preserve lookup actions and direct operators to FL Assistant/dashboard.

The archived `archive/superseded-schemas/openapi-schema.yaml`, `openapi-v3.yaml`, and `openapi-schema-gpt.yaml` also describe legacy writes. They are historical sources, not proof of a third active GPT. Retain their history and mark them non-deployable; do not paste them into an editor after cutover.

### Dashboard and browser callers

| Route IDs | Exact caller evidence | Closure effect / required disposition |
|---|---|---|
| B29 | `dashboard/dashboard.js:2512`; branch at `:2507` | Legacy ready-to-ship checkbox/note save fails. With session UI enabled it navigates to ticketed office ready form |
| B34 | `dashboard/dashboard.js:2822`; branch at `:2803` | Legacy order-header save fails; enabled UI navigates to office header prepare |
| B37 | `dashboard/dashboard.js:2885`; branch at `:2837` | Legacy quantity/price save fails; enabled UI navigates to office line prepare |
| B30/B31/B32 | `dashboard/so-list-actions.js:294,326`; redirect at `:94` | Legacy close/cancel/reopen preview and commit both fail; enabled UI navigates to corresponding office ticket form |
| B22/B23 | `dashboard/dashboard.js:3750,3740,3593`; branches at `:3570,3662,3718` | Legacy expected-receipt create/edit/status fails; enabled UI uses office prepare forms |
| B24 | `dashboard/dashboard.js:4266`; branch at `:4243` | Legacy extracted expected-receipt approval fails; enabled UI opens approval draft |
| B25 | `dashboard/dashboard.js:4851`; branch at `:4830` | Legacy extracted sales-order approval fails; enabled UI opens approval draft |
| B38 | `dashboard/exceptions.js:41,46`; dynamic action from `dashboard/ledger-ui.js:26`–`:30` | **Direct exception resolution fails**, including counted/missing-movement/evidence resolutions. The local review screen is not server ticket preparation. Migrate resolve to `/exceptions/{id}/resolve/prepare` plus generic commit; held-ticket approve/reject are separate lifecycle decisions |
| B21 | `dashboard/dashboard.js:4155` | **Unconditional direct supplier creation** in intake review fails. No supplier prepare replacement found |
| B53/B54 | `dashboard/dashboard.js:3120,3147` | **Unconditional direct reserve/release** fails. Both need ticket replacements or retirement |
| B55/B56/B57/B58 | `dashboard/dashboard.js:2109,2101,1991,1961` (`API_BASE` includes `/dashboard/api`) | **Unconditional notes create/edit/delete/toggle** fails. Need explicit scope decision and implementation |
| B51/B52 | `dashboard/dashboard.js:5345,5243` | **Unconditional supply request/create and mark-done** fails. Need ticket replacements or retirement |
| B59/B60 | `dashboard/runs.js:129` | **Unconditional production-run create/edit** fails |
| B61/B62/B63 | `dashboard/runs.js:184,197,231` | **Unconditional cancel/complete/coverage** fails. These run controls have no prepare replacement found |
| Preserved support | `dashboard/dashboard.js:3828,3850,4435,4451` | Upload/re-extract must stay usable under the explicit support-state decision; approvals remain ticketed |
| Preserved reads | `dashboard/dashboard.js:3905,4504,3196` | Matching and dedicated ship preview remain available; avoid broad POST or path-prefix bans |

**Dashboard fallback is a cutover blocker:** `dashboard/dashboard-runtime.js:15,26,32` turns on the new forms only when `/auth/session/config` reports `key_sessions_enabled=true`. A failed configuration request resolves the UI state to false (`:32`), selecting legacy fallbacks. Before closure, test timeout/error as well as success: a configuration failure must leave a clear unavailable/retry state rather than expose controls that can only hit retired paths. A stale cached dashboard may also take the old path. Check the served asset revision and reload behavior, not just the repository code.

Already-ticketed browser paths are `dashboard/entry.js:65` (receive/make/pack/adjust/found/ship preparation), `dashboard/office-logic.js:28`–`:42` (order and expected-receipt actions), `dashboard/ledger-ui.js:38` (void prepare), `dashboard/exceptions.js:38` (supplier-lot prepare only), and `dashboard/ledger-ticket.js:36` (generic commit). `dashboard/fl-assistant.js:136,229` posts through `/assistant/record`, whose backend commits a ticket. Preserve these callers. No direct write caller was found in the static scheduler HTML; `/schedule` remains an exposed server entry point and must still be handled.

### Scripts, rehearsal tools, and scheduled work

| Route IDs | Caller | What happens after closure |
|---|---|---|
| B07 | `audits/fresh-start/apply_reset.py:206,300` | **Legacy CSV reset backend breaks**: `LiveBackend.request` still permits direct `/adjust` preview/commit. The same current file already routes JSON/v3 previews through `run_v3` (`:473–476`), `LiveBackendV3` (`:707`), prepare (`:822`) and ticket commit (`:839`), with an explicit `--target` (`:1010–1011`). Preserve that v3 flow; retire the legacy backend and verify operators select the approved v3 packet. A personal-key substitution alone cannot save the old backend |
| B07 | `scripts/granola_writedown_0917.py:265,287` | Historical September write-down uses direct adjust preview/commit and requires a legacy shared identity. Retire it; any future correction needs a new reviewed ticket workflow |
| B08 | `scripts/cleanup_void_reversals.sh:44` | Historical reversal cleanup posts direct voids. Retire the executable cleanup after confirming completion; preserve evidence |
| B44 | `scripts/set_sweetened_coconut_yield.py:36` | One-off product-master change fails; retire completed operation or replace with approved catalog ticket functionality |
| B05/B06 | `scripts/check_recipe_corrections_staging.py:231,247,276,282` (make), `:263,264` (pack preview) | Staging acceptance script fails. Migrate writes to prepare/commit and pure previews to forced-preview routes; preserve corrected-formula and kosher assertions |
| B30/B31/B32/B33/B14/B20 | `scripts/check_direct_order_close_staging.py:439`–`:443`; browser proof at `:192` | Synthetic direct-close/status/ship matrix fails its old expectations. Replace with closure checks plus equivalent ticket-path owner/cutoff/shipment tests |
| B38 | `scripts/check_exceptions_staging.py:166,194` | Direct resolution acceptance fails. Replace with `/exceptions/{id}/resolve/prepare` and generic commit; retain shortage/waiver checks |

No checked-in cron definition or GitHub workflow was found invoking these HTTP writes. `scripts/daily-health-ping.sh`, `scripts/data_health.py`, `scripts/a5_readonly.py`, and reporting/cross-check scripts are read-oriented; ticket expiration and exception sweeps are server maintenance, not callers of the retired HTTP routes. The separate Mac audit inventories launchd/cron and local configuration; hosted scheduler configuration and other Macs cannot be inferred from this checkout. `scripts/inv_recon_post_2026_08_17.py` performs historical direct SQL repairs, not HTTP `/adjust` calls (its `adjust` is a local SQL helper); retiring HTTP routes will not disable it. Treat any remaining authorized direct-database maintenance as a separate scope question.

For B09–B13, B15–B19, B39–B43, B45–B50 and B64, no current non-test HTTP caller was found beyond the documentation and schema entries explicitly listed above (B50 has no active script caller here). This is evidence to check production logs, not permission to assume zero users. Route names can also be constructed by external clients absent from this repository.

### Stage-2 test changes: preserve behavior, replace transport

Do not delete entire regression suites merely because they call an old route. Move fixture/setup posts and business assertions to ticket preparation/commit or retained internal services, then delete only assertions whose sole purpose was legacy exposure, `confirmation_code`, or direct commit success. Keep a small permanent retirement suite proving old routes cannot write; decide deliberately whether stage-2 absence yields 404/405 or whether a lightweight 410 compatibility tombstone remains.

| Test family / evidence | Replace or remove | Retain through tickets/shared services |
|---|---|---|
| `tests/test_named_actor_writes.py:48`–`:55`; `tests/test_actor_attribution.py:894`–`:902`; `tests/test_dashboard_api_key.py:125`; `tests/test_roles_a2.py`; `tests/test_readonly_api_key.py` | Dynamic legacy route allow/deny and old floor-schema assertions; remove grants to deleted routes | Identity, role restrictions, shared/read-only-key refusal, correct actor on receipt, and zero mutation when denied |
| `tests/test_write_tickets.py`, `tests/test_write_tickets_part2.py`, `tests/test_ship_tickets.py`, `tests/test_order_tickets.py`, `tests/test_supplier_lot_tickets.py`, `tests/test_resolution_receipts.py` | Direct-route setup and explicit compatibility cases only | Ticket/hash/replay/stale/rollback/receipt guarantees; these remain primary positive tests |
| `tests/test_lot_identity_047.py`, `tests/test_audit_savepoints_lotcodes.py`, `tests/test_trace_emission.py`, `tests/test_trace_tables_048.py` | Direct posting/correction/identity transport | Lot identity, savepoints, trace integrity and mutation atomicity; build missing corrective ticket functions before retiring their coverage |
| `tests/test_void_semantics.py`, `tests/test_void_aware_views.py`, `tests/test_phase1_ledger_integrity.py`, `tests/test_recent_ledger.py`, `tests/test_daily_entries.py` | Direct void/correction/certification entry points | Append-only effective-state, balance and history behavior; amend/restore/certification gaps block removal of those capabilities |
| `tests/test_sales_order_state_model.py`, `tests/test_direct_order_close_guard.py`, `tests/test_sales_order_allocations.py`, `tests/test_released_by_attribution.py`, `tests/test_sales_order_readiness.py`, `tests/test_ship_order_service_line.py` | Direct close/ship/edit/allocation calls and fixtures | Close cutoff, owner checks, effective shipment evidence, release attribution, FIFO/reservation and service-line correctness |
| `tests/test_order_create_contract.py`, `tests/test_order_create_review.py`, `tests/test_sales_order_line_fields.py`, `tests/test_write_response_contract.py` | Old response-shape/transport expectations; raw legacy creation token expectations if actually unused | Duplicate prevention, stale edits, line precision, actor attribution, stable receipt/replay outcomes |
| `tests/test_expected_receipts.py`, `tests/test_expected_receipt_extract.py`, `tests/test_sales_order_extract.py`, `tests/test_a7_review_fixes.py` | Direct ER/order approval/create/update assertions | Upload/matching support, reviewed approval, duplicate/source linkage, receipt quantities and cancellation rules |
| `tests/test_production_runs.py`, `tests/test_run_types.py`, `tests/test_supplies.py`, `tests/test_notes_auth.py`, `tests/test_key_sessions.py` | Direct business-write cases only after replacement decision | Planning/coverage validation, supply behavior, note auth and session expiry/sign-out; auth itself remains open |
| `tests/test_recipe_corrections_078.py`, `tests/test_a12_review_fixes.py`, `tests/test_kosher_tiers.py`, `tests/test_lot_confirmation.py`, `tests/test_production_warning.py`, `tests/test_pilot_triage_backend.py:47,69`, `tests/test_exceptions_enforcement.py`, `tests/test_reset_shortage_close.py` | Legacy base/commit-alias cases and direct fixture posts | Correct formulas, kosher gates, lot confirmation, missing-recipe failures, shortage handling and owner restrictions |
| `tests/test_readonly_tripwire.py:296` | Direct-route matrix targets that disappear | Database read-only-error propagation on surviving read/ticket/shared-core paths; do not drop the incident regression |

Browser tests also encode the old transport: `tests/test_ready_ticket_ui.js:20`, `tests/test_order_create_review.js:62`, `tests/test_session_shift_fix_ui.js:49`, and visual scripts `tests/visual/run-d12-fixes.mjs:46,50`, `tests/visual/run-d12-staging.mjs:23`, `tests/visual/run-sales-orders-interactions.mjs:43,52`, `tests/visual/run-so-detail-interactions.mjs:80,102`, `tests/visual/run-so-exit-actions.mjs:184,387`, `tests/visual/run-runs-interactions.mjs:75,80`, `tests/visual/run-run-types.mjs:30,33`, `tests/visual/run-integrated-intake.mjs:45,51`, `tests/visual/run-office.mjs:35,55`, `tests/visual/run-d2.mjs:35`, and `tests/visual/lib/stub.mjs:89,157`. Distinguish intentional “no legacy write” assertions from mocks expecting old success. Update mocks and retain accessible controls, loading/error feedback, double-submit prevention, mode-off/config-failure behavior and receipt navigation. Do not run their staging writers during this planning task.

### Documentation to update or mark historical in stage 2

Update the live schemas/instruction sources and read-only cutover checklist above, plus `docs/data-entry-inventory.md:14` (older route/auth inventory), `docs/named-actor-writes.md:27`, `docs/order-create-contract.md:4`, `docs/design/phase1-safe-operating-system.md:776,824` (stage meaning and conflicting date), `FOLLOWUPS.md` (A10/cutover tracking), `FACTORY_LEDGER_SYSTEM_KNOWLEDGE.md` (legacy endpoint/operator guidance), `CONTEXT.md` (current architecture), `VOID_SEMANTICS_RUNBOOK.md:3`, and `audits/fresh-start/apply-guide.md:3` / `READINESS-2026-10-13.md:7` (reset transport). Update the relevant accepted scheduling, intake and allocation design documents when replacement decisions are made.

Historical deployment results, old audits and append-only change logs remain evidence. Add an explicit superseded/current-workflow pointer where needed rather than rewriting a historical claim as if it described the new system. The following reference index makes the remaining document review finite; a textual mention is not an executable caller. Each file has one or more legacy-route references; the first matching line is supplied so reviewers can locate the material without copying any credential-bearing text.

| Document reference | Classification |
|---|---|
| `CHANGE_LOG.md:12` | Historical evidence; retain context and add supersession pointer if reused |
| `CONTEXT.md:312` | Review current guidance; update or label superseded |
| `FACTORY_LEDGER_CHANGELOG.md:11` | Historical evidence; retain context and add supersession pointer if reused |
| `FACTORY_LEDGER_SYSTEM_KNOWLEDGE.md:5` | Review current guidance; update or label superseded |
| `FOLLOWUPS.md:281` | Review current guidance; update or label superseded |
| `IDEMPOTENCY_KEY_PLAN.md:52` | Review current guidance; update or label superseded |
| `VOID_SEMANTICS_RUNBOOK.md:5` | Review current guidance; update or label superseded |
| `archive/superseded-instructions/GPT_INSTRUCTIONS.md:6` | Historical evidence; retain context and add supersession pointer if reused |
| `archive/superseded-instructions/SALES_API.md:15` | Historical evidence; retain context and add supersession pointer if reused |
| `audits/fresh-start/HOSTED-REHEARSAL-2026-10-14.md:78` | Review current guidance; update or label superseded |
| `audits/fresh-start/READINESS-2026-10-13.md:7` | Review current guidance; update or label superseded |
| `audits/fresh-start/apply-guide.md:3` | Review current guidance; update or label superseded |
| `audits/reports/AUDIT_GPT_FABRICATION_2026-04-21.md:102` | Historical evidence; retain context and add supersession pointer if reused |
| `audits/reports/DASHBOARD_ACTIONABILITY_AUDIT.md:22` | Historical evidence; retain context and add supersession pointer if reused |
| `audits/reports/TRACEABILITY_AUDIT_2026-03-24.md:17` | Historical evidence; retain context and add supersession pointer if reused |
| `audits/reports/granola-batch-onhand-audit.md:5` | Historical evidence; retain context and add supersession pointer if reused |
| `audits/reports/granola-writedown-plan.md:85` | Historical evidence; retain context and add supersession pointer if reused |
| `audits/reports/inventory-summary-void-audit.md:32` | Historical evidence; retain context and add supersession pointer if reused |
| `audits/reports/packout-completeness-audit.md:5` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/audits/inventory-variance-recon-plan.md:174` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/audits/inventory-variance-txn-history.md:8` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/contracts/fl-assistant-part1.md:86` | Review current guidance; update or label superseded |
| `docs/contracts/resolution-receipts.md:40` | Review current guidance; update or label superseded |
| `docs/contracts/shift-summary.md:132` | Review current guidance; update or label superseded |
| `docs/data-entry-inventory.md:14` | Review current guidance; update or label superseded |
| `docs/data-health-baseline-2026-08-24.md:212` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/deployments/a1-write-tickets-part1.md:14` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/deployments/a1-write-tickets-part2.md:19` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/deployments/a12-kosher-tiers.md:61` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/deployments/a2-roles.md:84` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/deployments/a3b-exceptions-enforcement.md:151` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/deployments/a5-lot-confirmation.md:53` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/deployments/a5-supplier-lot-ticket.md:14` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/deployments/a6-ship-gate.md:13` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/deployments/a7-order-tickets.md:20` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/deployments/a9-shift-summary.md:17` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/deployments/f1-assistant.md:60` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/deployments/phase-1-ledger-trial.md:18` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/deployments/recipe-corrections-078.md:17` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/design/FL-Design-Standards-MASTER.md:116` | Review current guidance; update or label superseded |
| `docs/design/audit-2026-09-08-batch-a.md:12` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/design/audit-2026-09-08-integration.md:18` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/design/audit/00-screen-inventory.md:19` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/design/audit/02-actions-input-touch.md:642` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/design/audit/IMPROVEMENTS-MASTER.md:621` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/design/phase1-safe-operating-system.md:40` | Review current guidance; update or label superseded |
| `docs/design/runs-screen-spec.md:24` | Review current guidance; update or label superseded |
| `docs/design/sales-orders-redesign-spec.md:12` | Review current guidance; update or label superseded |
| `docs/design/scheduling-spec-draft.md:41` | Review current guidance; update or label superseded |
| `docs/design/so-state-model-findings.md:18` | Review current guidance; update or label superseded |
| `docs/designs/044-so-allocations-design.md:12` | Review current guidance; update or label superseded |
| `docs/designs/044-so-allocations-restore-addendum.md:31` | Review current guidance; update or label superseded |
| `docs/designs/045-write-foundation-design.md:37` | Review current guidance; update or label superseded |
| `docs/designs/TRACEABILITY_DESIGN.md:19` | Review current guidance; update or label superseded |
| `docs/designs/er-intake-audit-1-response.md:18` | Review current guidance; update or label superseded |
| `docs/designs/er-intake-audit-1.md:31` | Review current guidance; update or label superseded |
| `docs/designs/er-intake-audit-2.md:18` | Review current guidance; update or label superseded |
| `docs/designs/expected-receipt-intake.md:10` | Review current guidance; update or label superseded |
| `docs/designs/sales-order-intake.md:12` | Review current guidance; update or label superseded |
| `docs/incidents/2026-08-05-gpt-actions-dispatch.md:29` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/manual-tests/044-so-allocations-pr6-dashboard.md:21` | Review current guidance; update or label superseded |
| `docs/named-actor-writes.md:27` | Review current guidance; update or label superseded |
| `docs/order-create-contract.md:4` | Review current guidance; update or label superseded |
| `docs/resolution.md:54` | Review current guidance; update or label superseded |
| `docs/reviews/ef709e5-pr3-review.md:120` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/scheduling/S1-amendment-bake-vs-pack.md:14` | Review current guidance; update or label superseded |
| `docs/scheduling/coconut-yield-forward-only.md:8` | Review current guidance; update or label superseded |
| `docs/trace-preclean-worklist-2026-08.md:39` | Review current guidance; update or label superseded |
| `docs/traceability/FACTORY_LEDGER_TRACEABILITY_RULEBOOK.md:15` | Review current guidance; update or label superseded |
| `docs/validation/dashboard-d1-d2.md:17` | Historical evidence; retain context and add supersession pointer if reused |
| `docs/validation/dashboard-office-void.md:78` | Historical evidence; retain context and add supersession pointer if reused |
| `gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md:12` | Review current guidance; update or label superseded |
| `gpt-configs/readonly/CUTOVER_CHECKLIST.md:54` | Review current guidance; update or label superseded |
| `gpt-configs/sources/floor-specific.md:4` | Review current guidance; update or label superseded |
| `gpt-configs/sources/shared-rules.md:8` | Review current guidance; update or label superseded |
| `gpt-instructions-v3.md:6` | Review current guidance; update or label superseded |

### Per-route test/reference index

This index assigns every B ID to its tracked Python test references. It includes literal/f-string route references, parametrized matrices and direct Python handler calls, not only successful HTTP posts. It is a removal review aid: several rows are deliberate auth-denial tests that should survive with updated expectations. The prose above identifies current non-test callers and conditional dashboard branches. `—` means no matching Python test reference was found; it does not prove an untested external caller does not exist. Dynamic browser cases are indexed above. Only file paths and line numbers are reproduced.

| ID / retired route | Python test source evidence |
|---|---|
| B01 `PATCH /lots/{lot_code}/supplier-lot` | `tests/test_named_actor_writes.py:52` |
| B02 `PATCH /lots/{lot_id}/rename` | `tests/test_lot_identity_047.py:628,700`; `tests/test_named_actor_writes.py:52`; `tests/test_readonly_tripwire.py:306` |
| B03 `POST /receive` | `tests/test_a12_review_fixes.py:59,64`; `tests/test_actor_attribution.py:895`; `tests/test_audit_savepoints_lotcodes.py:486`; `tests/test_dashboard_api_key.py:88,136`; `tests/test_expected_receipts.py:118,345`; `tests/test_lot_identity_047.py:514,521`; `tests/test_named_actor_writes.py:49,57`; `tests/test_roles_a2.py:483`; `tests/test_supplier_lot_tickets.py:249`; `tests/test_trace_emission.py:237,258`; `tests/test_write_tickets.py:205` |
| B04 `POST /ship` | `tests/test_actor_attribution.py:895`; `tests/test_dashboard_api_key.py:88,134`; `tests/test_direct_order_close_guard.py:304,336`; `tests/test_named_actor_writes.py:49,57`; `tests/test_order_create_contract.py:317,323`; `tests/test_order_create_review.py:102,121`; `tests/test_roles_a2.py:446`; `tests/test_supplies.py:293`; `tests/test_trace_emission.py:371,400` |
| B05 `POST /make` | `tests/test_a12_review_fixes.py:209`; `tests/test_actor_attribution.py:894`; `tests/test_audit_savepoints_lotcodes.py:380,426`; `tests/test_dashboard_api_key.py:88,90`; `tests/test_expected_receipts.py:533`; `tests/test_key_sessions.py:44`; `tests/test_lot_identity_047.py:565,575`; `tests/test_named_actor_writes.py:49,57`; `tests/test_pilot_triage_backend.py:47`; `tests/test_production_warning.py:140,152`; `tests/test_readonly_api_key.py:334,339`; `tests/test_recipe_corrections_078.py:438,443`; `tests/test_roles_a2.py:437,481`; `tests/test_trace_emission.py:291,317` |
| B06 `POST /pack` | `tests/test_a12_review_fixes.py:282`; `tests/test_actor_attribution.py:894`; `tests/test_dashboard_api_key.py:88,91`; `tests/test_lot_identity_047.py:606,613`; `tests/test_named_actor_writes.py:50,57`; `tests/test_pilot_triage_backend.py:69`; `tests/test_recipe_corrections_078.py:466,468`; `tests/test_trace_emission.py:336,355` |
| B07 `POST /adjust` | `audits/fresh-start/test_apply_reset.py:521`; `tests/test_actor_attribution.py:894`; `tests/test_dashboard_api_key.py:88,92`; `tests/test_exceptions_enforcement.py:168`; `tests/test_inventory_occurred_at.py:76`; `tests/test_named_actor_writes.py:50,57`; `tests/test_phase1_ledger_integrity.py:165`; `tests/test_roles_a2.py:397,516`; `tests/test_trace_emission.py:478,497` |
| B08 `POST /void/{transaction_id}` | `tests/test_actor_attribution.py:896`; `tests/test_batch1_correctness_security.py:224`; `tests/test_dashboard_api_key.py:137`; `tests/test_expected_receipts.py:330`; `tests/test_named_actor_writes.py:50`; `tests/test_phase1_ledger_integrity.py:122`; `tests/test_resolution_receipts.py:60`; `tests/test_roles_a2.py:403,452`; `tests/test_sales_order_allocations.py:953,1117`; `tests/test_shift_summary.py:769`; `tests/test_ship_tickets.py:318,337`; `tests/test_supplies.py:159`; `tests/test_trace_emission.py:652`; `tests/test_void_aware_views.py:33`; `tests/test_void_semantics.py:124,186`; `tests/test_write_tickets.py:285`; `tests/test_write_tickets_part2.py:191` |
| B09 `POST /records/transactions/{transaction_id}/corrections` | `tests/test_batch1_correctness_security.py:200`; `tests/test_daily_entries.py:277`; `tests/test_phase1_ledger_integrity.py:137,178`; `tests/test_recent_ledger.py:121,150`; `tests/test_sales_order_allocations.py:1134,1162`; `tests/test_sales_order_readiness.py:329,524`; `tests/test_sales_order_state_model.py:187`; `tests/test_trace_emission.py:675,695` |
| B10 `POST /lots/{lot_id}/reassign` | `tests/test_a12_review_fixes.py:304`; `tests/test_audit_savepoints_lotcodes.py:229,259`; `tests/test_dashboard_api_key.py:144`; `tests/test_write_response_contract.py:329,417` |
| B11 `POST /inventory/found` | `tests/test_a12_review_fixes.py:196`; `tests/test_audit_savepoints_lotcodes.py:117,143`; `tests/test_exceptions_enforcement.py:173`; `tests/test_lot_identity_047.py:133`; `tests/test_trace_emission.py:511`; `tests/test_write_tickets_part2.py:221` |
| B12 `POST /inventory/found-with-new-product` | `tests/test_audit_savepoints_lotcodes.py:530`; `tests/test_trace_emission.py:531` |
| B13 `PATCH /lots/{lot_id}/received-at` | `tests/test_sales_order_allocations.py:1042,1255` |
| B14 `POST /sales/orders/{order_id}/ship` | `tests/test_direct_order_close_guard.py:304,336`; `tests/test_named_actor_writes.py:55,58`; `tests/test_order_create_contract.py:317,323`; `tests/test_order_create_review.py:102,121`; `tests/test_readonly_tripwire.py:212,232`; `tests/test_sales_order_state_model.py:3724,3991`; `tests/test_ship_order_service_line.py:241`; `tests/test_trace_emission.py:444,467` |
| B15 `POST /receive/commit` | `tests/test_a12_review_fixes.py:60`; `tests/test_named_actor_writes.py:491`; `tests/test_readonly_api_key.py:338` |
| B16 `POST /ship/commit` | `tests/test_actor_attribution.py:470`; `tests/test_dashboard_api_key.py:135`; `tests/test_released_by_attribution.py:255,316`; `tests/test_sales_order_allocations.py:1105,1187`; `tests/test_sales_order_state_model.py:2579` |
| B17 `POST /make/commit` | `tests/test_pilot_triage_backend.py:47` |
| B18 `POST /pack/commit` | `tests/test_pilot_triage_backend.py:69`; `tests/test_sales_order_allocations.py:1562,1751` |
| B19 `POST /adjust/commit` | — |
| B20 `POST /sales/orders/{order_id}/ship/commit` | `tests/test_actor_attribution.py:470`; `tests/test_released_by_attribution.py:255,316`; `tests/test_sales_order_allocations.py:1105,1187`; `tests/test_sales_order_state_model.py:2579`; `tests/test_ship_order_service_line.py:185,215` |
| B21 `POST /suppliers` | `tests/test_expected_receipts.py:444,447` |
| B22 `POST /expected-receipts` | `tests/test_a7_review_fixes.py:256,287`; `tests/test_expected_receipt_extract.py:1211,1224`; `tests/test_expected_receipts.py:100,142`; `tests/test_roles_a2.py:451` |
| B23 `PATCH /expected-receipts/{expected_receipt_id}` | `tests/test_a7_review_fixes.py:259,287`; `tests/test_expected_receipts.py:124,382` |
| B24 `POST /expected-receipts/extract/approve` | `tests/test_a7_review_fixes.py:288`; `tests/test_expected_receipt_extract.py:1048,1349` |
| B25 `POST /sales/orders/extract/approve` | `tests/test_named_actor_writes.py:677,679`; `tests/test_order_create_review.py:33`; `tests/test_sales_order_extract.py:834,864` |
| B26 `POST /customers` | `tests/test_dashboard_api_key.py:108`; `tests/test_named_actor_writes.py:51,130`; `tests/test_readonly_api_key.py:263,342`; `tests/test_roles_a2.py:444`; `tests/test_write_response_contract.py:255,360` |
| B27 `PATCH /customers/{customer_id}` | `tests/test_named_actor_writes.py:51,132`; `tests/test_roles_a2.py:445` |
| B28 `POST /sales/orders` | `tests/test_actor_attribution.py:760,764`; `tests/test_dashboard_api_key.py:107,143`; `tests/test_health_v3.py:195`; `tests/test_named_actor_writes.py:53,138`; `tests/test_order_create_contract.py:53,120`; `tests/test_order_create_review.py:38,43`; `tests/test_order_tickets.py:257,565`; `tests/test_readonly_api_key.py:265`; `tests/test_roles_a2.py:438,439`; `tests/test_sales_order_extract.py:1020,1208`; `tests/test_sales_order_line_fields.py:92,125`; `tests/test_sales_order_readiness.py:222,570`; `tests/test_sales_order_state_model.py:1904,1918`; `tests/test_write_response_contract.py:111,156` |
| B29 `POST /sales-orders/{so_number}/ready` | `tests/test_actor_attribution.py:396`; `tests/test_order_tickets.py:575`; `tests/test_roles_a2.py:488`; `tests/test_sales_order_state_model.py:2051,2071`; `tests/test_write_response_contract.py:144,167` |
| B30 `POST /sales/orders/{order_id}/close` | `tests/test_actor_attribution.py:334,679`; `tests/test_direct_order_close_guard.py:184,360`; `tests/test_order_tickets.py:592`; `tests/test_roles_a2.py:454,466`; `tests/test_sales_order_state_model.py:470,491` |
| B31 `POST /sales/orders/{order_id}/cancel` | `tests/test_actor_attribution.py:356`; `tests/test_direct_order_close_guard.py:276`; `tests/test_roles_a2.py:450,494`; `tests/test_sales_order_state_model.py:506,525` |
| B32 `POST /sales/orders/{order_id}/reopen` | `tests/test_actor_attribution.py:379`; `tests/test_direct_order_close_guard.py:453,581`; `tests/test_roles_a2.py:453`; `tests/test_sales_order_state_model.py:600,621` |
| B33 `PATCH /sales/orders/{order_id}/status` | `tests/test_actor_attribution.py:499,527`; `tests/test_direct_order_close_guard.py:230,273`; `tests/test_order_tickets.py:571`; `tests/test_released_by_attribution.py:277`; `tests/test_roles_a2.py:449`; `tests/test_sales_order_allocations.py:1313`; `tests/test_sales_order_state_model.py:2308,3012`; `tests/test_write_response_contract.py:182,372` |
| B34 `PATCH /sales/orders/{order_id}` | `tests/test_batch1_correctness_security.py:693`; `tests/test_fl_assistant.py:865`; `tests/test_health_v3.py:189`; `tests/test_order_create_contract.py:92,113`; `tests/test_order_create_review.py:267,333`; `tests/test_order_tickets.py:284,333`; `tests/test_pilot_triage_backend.py:94`; `tests/test_production_runs.py:950`; `tests/test_readonly_api_key.py:266`; `tests/test_roles_a2.py:448`; `tests/test_sales_order_allocations.py:1282`; `tests/test_sales_order_line_fields.py:99,131`; `tests/test_sales_order_readiness.py:230,238`; `tests/test_sales_order_state_model.py:809,812`; `tests/test_write_response_contract.py:197,380` |
| B35 `POST /sales/orders/{order_id}/lines` | `tests/test_actor_attribution.py:783`; `tests/test_named_actor_writes.py:53,138`; `tests/test_roles_a2.py:447`; `tests/test_sales_order_state_model.py:2627,2655`; `tests/test_write_response_contract.py:211` |
| B36 `PATCH /sales/orders/{order_id}/lines/{line_id}/cancel` | `tests/test_actor_attribution.py:784`; `tests/test_named_actor_writes.py:54`; `tests/test_order_create_review.py:189`; `tests/test_released_by_attribution.py:210,225`; `tests/test_sales_order_allocations.py:1356`; `tests/test_sales_order_state_model.py:3824`; `tests/test_write_response_contract.py:227` |
| B37 `PATCH /sales/orders/{order_id}/lines/{line_id}/update` | `tests/test_actor_attribution.py:573,608`; `tests/test_order_create_contract.py:286,354`; `tests/test_order_create_review.py:107,174`; `tests/test_sales_order_allocations.py:1301,1337`; `tests/test_sales_order_state_model.py:3805,4177`; `tests/test_write_response_contract.py:243` |
| B38 `POST /exceptions/{exception_id}/resolve` | `tests/test_a12_review_fixes.py:184`; `tests/test_exceptions_enforcement.py:377,569`; `tests/test_kosher_tiers.py:117`; `tests/test_named_actor_writes.py:174`; `tests/test_reset_shortage_close.py:87`; `tests/test_resolution_receipts.py:21,46`; `tests/test_ship_tickets.py:161,268` |
| B39 `POST /records/certifications` | `tests/test_phase1_ledger_integrity.py:367,412` |
| B40 `POST /records/certifications/{certification_id}/corrections` | `tests/test_phase1_ledger_integrity.py:431` |
| B41 `POST /products/quick-create` | `tests/test_audit_savepoints_lotcodes.py:161,184` |
| B42 `POST /products/quick-create-batch` | — |
| B43 `POST /products/{product_id}/verify` | `tests/test_audit_savepoints_lotcodes.py:201` |
| B44 `PUT /admin/products/{product_id}` | `tests/test_dashboard_api_key.py:140`; `tests/test_dashboard_b2.py:155` |
| B45 `POST /admin/bom/{product_id}/lines` | `tests/test_readonly_tripwire.py:313` |
| B46 `PUT /admin/bom/lines/{line_id}` | `tests/test_dashboard_api_key.py:139` |
| B47 `DELETE /admin/bom/lines/{line_id}` | `tests/test_dashboard_api_key.py:139` |
| B48 `POST /admin/product-bom` | — |
| B49 `DELETE /admin/product-bom/{mapping_id}` | — |
| B50 `POST /admin/lots/merge` | `tests/test_a12_review_fixes.py:316`; `tests/test_dashboard_api_key.py:138`; `tests/test_sales_order_allocations.py:1214`; `tests/test_trace_emission.py:806,821` |
| B51 `POST /supply-requests` | `tests/test_supplies.py:126,315` |
| B52 `PATCH /supply-requests/{supply_request_id}` | `tests/test_supplies.py:360,386` |
| B53 `POST /sales/orders/{order_id}/allocations` | `tests/test_actor_attribution.py:421`; `tests/test_sales_order_allocations.py:978,984`; `tests/test_sales_order_state_model.py:2672,2691` |
| B54 `POST /sales/orders/{order_id}/allocations/{allocation_id}/release` | `tests/test_actor_attribution.py:442`; `tests/test_released_by_attribution.py:300`; `tests/test_sales_order_allocations.py:1012,1046` |
| B55 `POST /dashboard/api/notes` | `tests/test_dashboard_api_key.py:94`; `tests/test_key_sessions.py:222`; `tests/test_notes_auth.py:76,91`; `tests/test_readonly_tripwire.py:320`; `tests/test_write_response_contract.py:265,276` |
| B56 `PUT /dashboard/api/notes/{note_id}` | `tests/test_key_sessions.py:225,226`; `tests/test_notes_auth.py:93,94`; `tests/test_write_response_contract.py:278,291` |
| B57 `DELETE /dashboard/api/notes/{note_id}` | `tests/test_key_sessions.py:225,226`; `tests/test_notes_auth.py:93,94`; `tests/test_write_response_contract.py:278,291` |
| B58 `PUT /dashboard/api/notes/{note_id}/toggle` | `tests/test_key_sessions.py:226`; `tests/test_notes_auth.py:95,117`; `tests/test_write_response_contract.py:303` |
| B59 `POST /production/runs` | `tests/test_production_runs.py:61,62`; `tests/test_run_types.py:40,52` |
| B60 `PATCH /production/runs/{run_id}` | `tests/test_production_runs.py:63,582`; `tests/test_run_types.py:31,55` |
| B61 `POST /production/runs/{run_id}/cancel` | `tests/test_production_runs.py:64,555`; `tests/test_run_types.py:109` |
| B62 `POST /production/runs/{run_id}/complete` | `tests/test_production_runs.py:65,643` |
| B63 `PUT /production/runs/{run_id}/coverage` | `tests/test_production_runs.py:67,318` |
| B64 `POST /schedule` | `tests/test_dashboard_api_key.py:88,141` |

## Stage 1 proposal — close the external entrances on November 9

This is a proposal, not an implemented flag or a production change. Use one explicitly configured flag, **`LEGACY_DIRECT_WRITES_ENABLED`**, and a checked-in exact method/route-template manifest covering B01–B64. Deploy the gate with the flag explicitly enabled for an observation period, migrate callers, then set it to false for closure. The completed implementation should fail closed for an absent or malformed setting; validate that behavior before rollout. Do not use a prefix such as `/sales/orders`, a blanket POST ban, or a suffix such as `/commit`: each would catch required reads or ticket operations.

**Apply the same gate to every key kind.** Removing dashboard/personal grants alone leaves master access open because master authentication returns early. Put the gate at the matched HTTP-route boundary, keeping existing authentication and authorization, before body-model/business dependencies and any business mutation. For an otherwise authorized request, a retired route returns 410 even when its old request body is malformed. Preserve existing credential, key-scope and named-role 401/403 refusals. This includes the named floor/owner check for B38: master passes general authentication but is refused by its route-role policy. Move/reuse that policy at the boundary without invoking the business handler. Payload-dependent business validation is unnecessary once an otherwise permitted route is closed; do not promise its former body-dependent statuses. The gate may still incur existing authentication/session metadata writes; it must not invoke the business handler, allocate receipts, create a ticket, upload a file, or resolve an order through a business dependency just to reject the operation. In particular, `resolve_order_id` performs a database lookup for order-number input (`main.py:3351`).

Use a small structured response with plain wording, for example:

> This direct write route is closed. Use FL Assistant or the dashboard to prepare and review a ticket, then select Record. Nothing was recorded. If that action is unavailable, ask Michael for the approved workflow.

Give it a stable error code such as `DIRECT_WRITE_RETIRED`, links to the actual deployed Assistant/dashboard, and non-cacheable headers. Verify those links at rollout; a link is not evidence that the replacement action exists. The existing write-response envelope (`main.py:209`) must preserve the status and message. Clients must stop automatic retries and show the message; never silently replay a failed legacy request through a ticket or claim it was recorded.

| Inventory group | Stage-1 disposition |
|---|---|
| B01, B03–B08, B11, B14–B20 | Close the old HTTP entrance. Require the listed prepare endpoint, explicit review, generic commit and receipt; retain posting/preview cores. B01/B11 replacement authorization differs from the old route, so verify the intended named operator |
| B22–B25, B28–B38 | Close direct order/ER/state/exception writes. Migrate both dashboard branches and scripts. Existing internally generated receipt numbers on void/resolve do not waive the two-step requirement |
| B02, B09–B10, B12–B13, B21, B26–B27, B39–B63 | These rows have missing or partial ticket replacements. Close their old entrances under the same flag; finish and accept replacements first, or explicitly retire/freeze the affected operation. Until that choice is made, closure is a readiness blocker, not a functioning replacement |
| B64 `POST /schedule` | Gate `action=confirm`. Preserve only the normalized, explicitly enumerated read actions `suggest` and `current` after verifying their transitive behavior. Missing/unknown actions must never reach a writer while closed. Do not allow a permissive “anything except confirm” fallback |
| Six forced-preview aliases | Keep current read behavior and authorization; they must force preview even if the caller sends `mode=commit`. Extract preview services as needed so the gate on a base HTTP route cannot block an internal preview call |
| Ticket preparations, generic commit, receipt reads, held-ticket approval/rejection, Assistant transport | Keep current authorization, holds, staleness and receipt guarantees. Preserve an existing legitimate prepared ticket across the flag change; do not re-create it or invalidate its saved replay response |
| Upload/intake/shift-confirmation/auth support rows | Keep only the necessary, explicitly agreed support/lifecycle scope described above. Shipment evidence and shift-confirmation changes need particular attention: calling them “metadata” does not settle whether the strict requirement permits them |

The broad no-ticket rows include daily notes, supply requests, supplier creation and planning. The plan does not assume those features are expendable. Michael must choose replacements or a deliberate feature freeze before the November 9 go/no-go. If the intended A10 scope is only inventory/order posting, record the reduced scope by explicit row IDs and account for the remaining writes; do not describe that smaller closure as “every write goes through tickets.”

Keep reads available, including G1 read-only GPT actions and M0's dedicated ship-preview path. M0 uses named backing credentials in its reviewed source; do not equate its read-only tool catalog with the narrower G1 `READONLY_API_KEY` allowlist. Confirm writes remain disabled in the deployed M0 service, not just hidden from its tool list.

### Observation and one-step rollback

Add sanitized observation before starting the 72-hour window: timestamp, deployment/revision, method, normalized route template, safe action name where needed for `/schedule`, status, key kind, actor ID where present, and a request correlation ID. A declared client source/user agent is a hint, not trusted identity. Do not log keys, authorization headers, raw query strings, bodies, packing-slip URLs, or credential hashes. Count allowed direct attempts during observation and rejected attempts after closure, across every replica and ingress.

Existing `write_response_envelope` is not a complete direct-route usage counter. `actor_write_audit` also omits legacy-key calls (`main.py:3226`), and actor “last used” timestamps do not prove that master-key use stopped. Confirm what production logs actually retain before promising a clean window.

**Stage-1 rollback is one configuration action:** set `LEGACY_DIRECT_WRITES_ENABLED=true` using the prepared service configuration action, which redeploys/reloads all replicas on the same reviewed stage-1 release. The runbook must record the exact target service and verify every replica has converged; saving an environment value is not instantaneous closure or rollback. No migration rollback, receipt deletion, source revert, or restoration of the old master key is part of this action.

This deliberately reopens all gated legacy routes and loses the strict ticket-only property. Make that visible to the operator, record the reason/time, bound the rollback period, and restore the gate after the incident. Rotated credentials stay rotated: an old caller still using the invalid key will not recover merely because routes reopen. Retain the close/shipment/kosher/role guards from the reviewed release during rollback. Wait for/drain in-flight legacy requests at closure and establish the effective cutover timestamp only after all instances report the closed state.

## Stage 2 proposal — remove obsolete implementation after stability

No removal date is approved. Proposed entry gate: at least **seven operating days** of stage-1 stability, including the normal weekly workflows, no unexplained legitimate direct-route attempts for the final 72 hours, accepted replacements/frozen-feature decisions for every gap row, and Michael's go-ahead. This is a proposed observation period, not a source-backed existing schedule.

1. Remove the external B01–B64 business-write handlers/decorators and direct commit aliases once each replacement or retirement is accepted. For `/schedule`, remove the confirm branch while retaining its verified read contract, or deliberately migrate its read callers first.
2. Remove each retired method/template from `DASHBOARD_KEY_ALLOWLIST`, `ACTOR_WRITE_ALLOWLIST` and the direct-route permission map. Keep permissions/actions still needed by tickets and previews. The G1 read-only allowlist should not need widening.
3. Delete legacy-only request fields, confirmation-code dispatch, response adapters and direct-create compatibility logic **only after** a dependency search proves no ticket/preview/receipt replay still needs them. `ticket_actions.post` uses shared `_*_commit_core` functions; `order_tickets.py:373`–`:431`, `expected_receipt_tickets.py:100`–`:114`, and `sales_document_tickets.py:60` reuse order/intake cores. Removing a decorator does not authorize deleting a core.
4. Preserve the six preview services and their current contracts; remove their dependency on deleted base handlers. Keep effective-ledger views, append-only corrections, old receipts, idempotency records, lookup history, migrations and database tables. This retirement does not require deleting or rewriting historical ledger data.
5. Remove obsolete dashboard fallback branches and direct-write controls after ticket replacements or explicit feature retirement. Configuration/network failures must show an unavailable state, not fall back to retired routes. Remove legacy write operations from deployable GPT schemas/instructions and update the editor configuration; archived schemas must be clearly historical.
6. Retire the one-shot scripts listed in the caller inventory; preserve their evidence. Disable the historical direct-SQL inventory poster as a separately reviewed source change. An HTTP flag and API-key rotation cannot stop a script using `DATABASE_URL`. Keep reviewed ticket-expiry/exception-sweep maintenance and document its distinct service/database authority.
7. Replace/delete test cases and update documentation according to the caller inventory above. Preserve policy/atomicity tests through the surviving ticket/service layer. Do not remove a test solely because it reveals a capability whose replacement is still missing.
8. Remove the stage-1 reopening flag after the removal release is accepted. Prefer actual route removal: a missing method/path will return 404 or 405 where another method still exists. If a long-lived 410 tombstone is wanted for operator guidance, decide that explicitly; it must contain no business writer. Stage 2 no longer has a flag-only rollback: recovery is redeploying the retained stage-1 release, with its gate closed by default, under the incident runbook.

## Validation plan (future implementation, not executed in this task)

This task performed static source/document checks only. It did not run pytest, Node/browser suites, app startup, database creation, staging writers, production write probes, or the one-shot scripts. Even an empty-body “should fail” POST is not an authorized production check here. The existing DB fixtures/imports can create databases, so test collection is not assumed harmless.

### Stage 1

| Area | Required evidence |
|---|---|
| Manifest completeness | Compare the registered route set with B01–B64 and all dynamically/module-registered routes. Assert no business writer is uncategorized. Check static aliases hidden from OpenAPI as well as visible routes |
| Closed-route matrix | For every applicable row, exercise master, dashboard, read-only, owner/floor/office personal keys, owner/floor/office sessions, missing/invalid/revoked/expired credentials, session flag on/off, preview/commit modes, valid/invalid bodies and path parameters. Otherwise-authorized closed calls yield 410; unauthorized calls remain refused. A sentinel around the business handler proves it never runs. Ledger, orders, lots, allocations, receipts and business audit remain unchanged; distinguish intentional auth metadata |
| Gate and rollback | Test explicit true/false, absent/malformed flag, startup/reload behavior, every replica, requests in flight, and a real rollback drill in an isolated environment. With the flag reopened, only pre-existing authorization/guards return; no new privilege is granted |
| Surviving reads | All G1 actions (14 office, 11 floor), M0's 17 read tools and six forced-preview aliases. Send commit-shaped bodies to forced previews and prove no business posting occurs. Verify schedule suggest/current remain read-only and malformed/unknown actions cannot fall into confirm |
| Surviving tickets | Prepare/review/commit/receipt for receive, make, pack, adjust, found, ship/order/bulk, void, supplier-lot/move, order/ER/intake and exception resolution. Include stale hashes, expired/wrong-actor tokens, duplicate/concurrent commit, same receipt on retry, and a ticket prepared before the closure flag changes |
| Business safeguards | Preserve source-lot confirmation, kosher source/attestation/held approval, photo requirements, shortage holds, backdating, append-only corrections, effective-shipment and legacy-close cutoff/note/owner checks. Include floor/office refusals and owner sessions; retain PR #121 shared-core rules already present in this snapshot |
| User workflows | Test named dashboard/Assistant workflows, sign-in/out, config failure/timeouts, stale cached assets, receipt links, drafts, 202 held vs recorded, explicit 410 feedback and no automatic resubmission. Verify every operationally required gap-row feature has its accepted replacement or visible retirement notice |
| Automation and reset | Preserve and test the existing v3 reset journal/recovery and ticket path; retire legacy CSV reset transport and migrate other remaining scripts before rehearsal. Prove the selected reset invocation uses the reviewed owner-approved ticket path, matching count quantities and receipts, including extra-kosher lots. Confirm no cron/automation invokes a retired HTTP route or bypasses it with unreviewed direct SQL |
| Observability | Verify sanitized counters distinguish methods/templates, every instance, key kinds, schedule read/confirm, expected test calls and real callers. Confirm the logs contain no credentials or sensitive URL/body material |

Start with route-gate/unit tests using isolated fakes and no DB creation. The meaningful transaction/atomicity and concurrency checks then need the project's controlled integration environment in a separate authorized implementation task; use only explicitly isolated fixtures and never infer safety from an environment label. Production acceptance for closure begins with read-only logs/metrics and approved reads; any future end-to-end posting rehearsal needs its own approved target and scope.

### Stage 2

Re-run the surviving ticket/read/policy regression families from the previous table and the retained legacy-route rejection suite. Additionally prove:

- The public route table, generated OpenAPI, GPT deployable schemas, dashboard assets and scripts have no callable retired writer or hidden commit alias. Search assembled/dynamic URLs as well as literal strings.
- All B01–B64 entrances are absent or inert tombstones (schedule read-only branch excepted); no retained function can be reached through a new alias that bypasses ticket review.
- No ticket or preview imports/calls a deleted handler/model. Receipt lookup/replay and historical effective balances/trace still work after removal.
- The ready forms, order/ER controls, intake, shipping proof, owner holds and required catalog/planning replacements work with named keys/sessions. G1 and M0 remain read-only and usable.
- Any intentionally deleted compatibility test is replaced by rejection or surviving business coverage. Static schema operation limits and existing incident regressions still pass.
- Redeploying the retained stage-1 release is rehearsed without undoing data, migrations, credentials, or receipts.

## November 9 pre-flight and go/no-go

All boxes below are **pending**. No production-log review or deployed-state verification was performed for this planning task.

- [ ] **Resolve the calendar first.** Confirm the cutover date, the November 9 closure time in America/New_York, operator and rollback owner. Reconcile the November 20/23 dates in the existing plan and `ORDER_LEGACY_CLOSE_CUTOFF`; do not change the cutoff merely to make dates look consistent.
- [ ] Confirm the deployed revision contains the complete gate, ticket replacements, client fixes, required migrations and shared-core guards. Record production and dashboard revisions and the deployed M0 revision/configuration; source inspection alone is insufficient.
- [ ] Accept or explicitly retire every gap-row capability. Confirm the support-write boundary, shipment evidence/shift-confirmation policy, and any catalog/planning scope decision. Required business work with no accepted replacement is a no-go.
- [ ] Verify the accepted count/reset invocation selects the JSON/v3 ticket backend and explicit target. This source snapshot already contains that backend, but also retains the legacy CSV backend calling `/adjust`; retire the legacy path and confirm the actual operator packet and receipt/journal recovery. Do not infer live readiness solely from source or a deployment note.
- [ ] Verify both live GPT editors contain the G1 read-only schemas, instructions and read-only credential, with no lingering legacy write action. Verify M0 hides **and refuses** writes and uses the dedicated shipping-preview route. Confirm signed packing-slip/evidence links do not rely on the rotated master.
- [ ] Inventory all external consumers, including this Mac's audit, other Macs, hosted cron services, spreadsheet/PO-sync jobs, browser tabs and manually installed GPT actions. Obtain an owner for each remaining caller; this Mac is not the entire deployment.
- [ ] Establish **complete production request coverage for the previous 72 hours**. For a Monday November 9 switch, that starts at the same local time Friday November 6. Check all 64 route templates, all methods and commit aliases, successful calls and refused/retried calls, with the schedule action distinguished. Require zero legitimate direct-write attempts, and explain any scanner/test traffic. Missing log retention, omitted instances or unclassified traffic is not zero.
- [ ] Add evidence from representative working shifts and the normal weekly jobs. A quiet Friday-to-Monday window can hide weekday-only consumers; 72 hours by itself is not proof that every workflow migrated.
- [ ] Review request logs **before and after key rotation**. A caller that turns from 200 into 401/403 still needs migration. Verify failed direct calls are not being endlessly retried or masked by a dashboard fallback.
- [ ] Corroborate logs with new-write audit/receipt evidence using entry/commit time, not a backdated event date. Check new postings for missing ticket/receipt links and unnamed identities, and inspect order/catalog changes separately. Receipt presence alone does not prove client preparation: direct void/resolve already create receipts inline.
- [ ] Check errors and latency on prepare/commit/receipt, pending/held tickets and evidence uploads, owner approval availability, correct actor/role/session behavior, and key-delivery readiness. Confirm ordinary reads and shared-key/public health/reporting consumers have their intended replacement credentials.
- [ ] Verify ticket-expiry and exception-sweep service status from their sanitized logs; they use database credentials, not the master API key. Confirm no retired inventory poster or one-shot cleanup is scheduled. Do not disable legitimate maintenance to obtain a clean HTTP chart.
- [ ] Complete the isolated stage-1 acceptance and rollback rehearsal; save revision, operator, timestamp and evidence references. Confirm the one-step configuration action targets every production replica and retains all safety guards.
- [ ] At the agreed time, apply the closed configuration and verify all instances have converged and pre-existing direct requests have drained. Observe reads, ticket receipts, 410 counts and user reports through a full working shift. Unexpected business-route traffic, blocked essential work or a broken ticket/read path triggers the agreed rollback/escalation decision, not an improvised credential bypass.

## Top risks and open questions for Michael

1. **People can lose working tools on closure day.** The current dashboard has unconditional direct writers and legacy fallbacks, the legacy CSV reset backend still calls direct adjust, and several catalog/planning operations have no ticket replacement. A personal key fixes identity, not the route or the missing feature.
2. **A partial fix can leave a bypass or break the new path.** Master access, hidden commit aliases, inline-receipt routes and the historical direct-SQL poster all need distinct handling. Conversely, deleting shared legacy functions can break ticket commits and M0 previews. The flag must guard HTTP entrances while preserving reviewed business cores.
3. **The proposed date and operational evidence are not settled.** November 9 conflicts with the current November 20/23 schedule; no live 72-hour usage evidence has been checked. This Mac also has stored legacy credential copies and uninspected automation/session-history areas. A zero-count assumption would hide risk rather than resolve it.

Decisions needed, without guessing:

- Is November 9 the new cutover/closure date, and at what local time? What should the existing legacy-close cutoff remain?
- Does “every write” include notes, planning, supplier/customer/catalog maintenance, shipment-evidence completion and shift confirmation, or is A10 limited to a specified subset? Which gap-row operations may actually be frozen, and who owns each replacement?
- Is the current v3 ticket-based count/reset accepted, and will the operator use its JSON preview and explicit target? Which real cutover count is still required?
- Who confirms the live GPT editor changes, M0 deployment/write-disable setting, other-device consumers and hosted schedules?
- Where are complete sanitized production route logs, how long are they retained, and who can certify the 72-hour window plus representative shifts?
- Who operates the closure and rollback, what outage/traffic condition triggers rollback, and how long may direct writes be reopened?
- What stability period and date should authorize stage 2? Should removed routes disappear (404/405) or retain a non-writing 410 tombstone?
- Who owns post-rotation cleanup of credential-bearing operational docs/history and verification of Shortcuts/session-history gaps in the companion Mac audit?

## Evidence and change boundary

The companion document is [Master-key audit of this Mac](master-key-audit-home-mac.md). This branch adds only these two planning documents. Existing code, schemas, tests, configuration, change logs and other worktrees were not edited. The user explicitly limited writes to these files, so the repository's usual project/global change-log additions are intentionally not performed.

Validation performed for this documentation: static parsing of registered route declarations and authorization sets, caller/reference searches, file/line evidence review, matching all 64 inventory rows to source, document consistency/secret checks, and Git diff scope checks. No application tests or database-creating test fixtures were run. All future deploys, configuration changes, key rotation, workflow migration and removal work remain separate execution work.
