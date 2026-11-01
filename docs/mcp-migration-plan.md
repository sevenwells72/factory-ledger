# Factory Ledger MCP migration plan

## Plain-English summary

- Replace the old GPT’s 30 actions with one shared connection for ChatGPT and Claude.
- Run that connection as a separate service on Railway, alongside the existing Factory Ledger app.
- Use individual sign-in instead of a shared password; choose the sign-in provider and who may make changes.
- Require explicit approval before every change, and return the internal SO number for every order change.
- Save pallet charges as Pallet Charge line items, product 176, and customer PO numbers in the dedicated PO field.
- Confirm which GPT staff actually use, name a migration owner, and choose a pilot group before switching.
- Budget 12–20 engineering days plus pilot time, with a target switch by November 20 ahead of the supplied December 11 deadline.

## Scope and evidence

Prepared September 28, 2026 against local checkout `54df62b`. This is a planning document only. No application code, configuration, credentials, deployments, or production records were changed or exercised to prepare it.

The inventory below is exhaustive for `openapi-gpt-v3.yaml`, whose declared API version is **3.5.0**: **30 operations, comprising 14 business reads and 16 writes**. A POST is not necessarily a write: `resolveProducts` only resolves names. Preview-capable tools count as writes because they can also commit.

Local evidence: `openapi-gpt-v3.yaml`; `gpt-instructions-v3.md` v3.8.0; `gpt-configs/README.md`; `main.py`; `requirements.txt`; `railway.json`; `CONTEXT.md`; migration `050_sales_doc_intake.sql`; the sales-order intake design; and the April 21 fabrication audit and August 5 dispatch incident report. Source references below are repository-relative descriptions, not claims that the live GPT editor or deployment was inspected.

OpenAI’s migration guidance confirms that instructions/knowledge can migrate but custom actions must be rebuilt. It also says the original becomes read-only after migration. The retrieved guidance concerns Enterprise and does not state December 11 in its text; **December 11, 2026 is the owner-supplied deadline**, pending confirmation for the actual account/workspace. [OpenAI migration guidance](https://learn.chatgpt.com/docs/migrate-custom-gpts)

### What the GPT actually relies on

The instructions establish workflow dependencies, not call frequency. No current action logs, usage export, or live GPT configuration were inspected. Do not label every exposed operation “used daily.” The inventory uses:

- **Core:** explicitly routed by the legacy instructions for ordinary work; prioritize in acceptance testing. Actual daily frequency remains unmeasured.
- **Conditional:** needed when a particular task arises, such as recall tracing, formulas, corrections, or new customers.
- **Exposed:** present in the action schema but without a clear named workflow in the current legacy instructions; retain for parity and validate with the owner.

The daily workflow chains documented in those instructions are:

1. **Office/order entry:** `searchCustomers` → `resolveProducts` (or `searchProducts`) → `createOrder`; `listOrders`/`getOrder` for follow-up; header, line, and status edits as requested.
2. **Floor operations:** `searchProducts`/`inventoryLookup`/`getLotByCode` → preview and commit of `receive`, `make`, `pack`, or `adjust`; `getDaySummary` for shift close.
3. **Dispatch:** `listOrders(status=open)`/`getOrder` → `shipOrder`; use standalone `ship` only when there is no appropriate open SO or the operator explicitly requests standalone shipping.
4. **Exceptions and traceability:** customer/product disambiguation, supplier-lot corrections, backward/forward recall trace, and transaction history.

There is an important deployment ambiguity: `gpt-configs/README.md` describes a live **Floor & Fulfillment** GPT with **22 actions**, while retaining the root legacy files. The August 5 incident calls the personal **Factory-Ledger 2.0** configuration obsolete and recommends removing or renaming it. That incident provides evidence of actual `getOrder`/shipping use, including a successful floor dispatch; the April audit provides evidence of actual order-status work. Neither proves current daily use of the named legacy GPT.

Before implementation, the owner should identify the exact GPTs, accounts, operators, live schemas, and instructions. Collect 2–4 weeks of existing request logs and representative conversations, redacting secrets/customer details. Count calls and failures by operation and distinguish GPT traffic from dashboard traffic where attribution permits. Save 10–15 familiar prompts/results. Retain all 30 operations for parity even if some are rarely used.

## 1. Complete operation inventory and proposed MCP tools

**One row equals one existing operation and one proposed MCP tool.** Keep the existing operationId as the MCP tool name. The endpoint column is relative to the current action server, `https://fastapi-production-b73a.up.railway.app`. “Write” means explicit, payload-specific confirmation is required before committing, including writes without inventory effects.

| # | Existing operation → MCP tool name | HTTP endpoint | What it does | Read/write | Workflow reliance |
|---|---|---|---|---|---|
| 1 | `searchProducts` | GET `/products/search` | Finds products by name/SKU using exact, keyword, and fuzzy matching; returns sizing metadata. | Read | Core: single-product preflight. |
| 2 | `resolveProducts` | POST `/products/resolve` | Resolves several raw/OCR product names, with confidence and alternatives. | Read | Core: multi-line order preflight. |
| 3 | `getBatchFormula` | GET `/bom/batches/{batch_id}/formula` | Returns ingredients, IDs, formula notes, and production warnings. | Read | Conditional: ingredient overrides, exclusions, and formula checks. |
| 4 | `inventoryLookup` | GET `/inventory/lookup` | Finds products, lots, and on-hand balances. | Read | Core: primary inventory question route and lot selection. |
| 5 | `getLotByCode` | GET `/lots/by-code/{lot_code}` | Returns internal lot details and supplier-lot references. | Read | Core: bare lot-code lookup. |
| 6 | `updateSupplierLot` | PATCH `/lots/{lot_code}/supplier-lot` | Corrects or attaches the supplier’s lot reference and notes. | Write | Conditional: label/packing-slip mismatch. |
| 7 | `receive` | POST `/receive` | Previews or records incoming inventory, lots, and supplier information. | Write; native preview | Core: receiving. |
| 8 | `createExpectedReceipt` | POST `/expected-receipts` | Records an expected supplier delivery; no stock movement and not a purchase order. | Write | Core: “expecting/incoming” workflow. |
| 9 | `ship` | POST `/ship` | Previews or records a standalone shipment with lot allocation. | Write; native preview | Conditional: standalone exception to SO shipping. |
| 10 | `make` | POST `/make` | Previews or records batch production and ingredient consumption. | Write; native preview | Core: production. |
| 11 | `pack` | POST `/pack` | Previews or records batch-to-finished-goods packing, including applicable add-ins. | Write; native preview | Core: packing; distinct from making. |
| 12 | `adjust` | POST `/adjust` | Previews or records an inventory increase/decrease with a reason. | Write; native preview | Core: requested stock correction. |
| 13 | `traceBatch` | GET `/trace/batch/{lot_code}` | Traces a batch backward to consumed ingredient lots. | Read | Conditional: trace/recall work. |
| 14 | `traceIngredient` | GET `/trace/ingredient/{lot_code}` | Traces an ingredient forward to where it was used. | Read | Conditional: trace/recall work. |
| 15 | `traceSupplierLot` | GET `/trace/supplier-lot/{supplier_lot_code}` | Finds internal lots and exposure associated with a supplier lot. | Read | Conditional: supplier-lot lookup/recall. |
| 16 | `getTransactionHistory` | GET `/transactions/history` | Returns recent transactions filtered by type, product, and dates. | Read | Core: operational history questions. |
| 17 | `getDaySummary` | GET `/production/day-summary` | Summarizes a day’s make/pack/adjust activity and consumption. | Read | Core: shift wrap-up. |
| 18 | `listCustomers` | GET `/customers` | Lists customers, optionally active only. | Read | Conditional: customer directory. |
| 19 | `createCustomer` | POST `/customers` | Creates a customer with contact/address information. | Write | Conditional: confirmed new customer after failed lookup. |
| 20 | `searchCustomers` | GET `/customers/search` | Finds customer names/aliases for identification and disambiguation. | Read | Core: customer lookup and order preflight. |
| 21 | `updateCustomer` | PATCH `/customers/{customer_id}` | Edits customer details, aliases, notes, or active status. | Write | Exposed: customer maintenance; frequency unverified. |
| 22 | `listOrders` | GET `/sales/orders` | Lists filtered orders, quantities, readiness, and blockers. | Read | Core: open orders, dispatch, and packing-slip lookup. |
| 23 | `createOrder` | POST `/sales/orders` | Creates an SO and its lines; current backend creates it as confirmed. | Write | Core: confirmation/PO entry. |
| 24 | `getOrder` | GET `/sales/orders/{order_id}` | Returns one SO’s header, lines, shipments, totals, and readiness. | Read | Core companion: exact order/detail lookup; actual use evidenced in floor incident. |
| 25 | `updateOrderHeader` | PATCH `/sales/orders/{order_id}` | Edits ship date, notes, or customer within allowed states. | Write | Core: order editing. |
| 26 | `updateOrderStatus` | PATCH `/sales/orders/{order_id}/status` | Makes an allowed manual status transition; shipping states are automatic. | Write | Core: status changes/cancellation; actual use evidenced in April audit. |
| 27 | `addOrderLines` | POST `/sales/orders/{order_id}/lines` | Adds resolved product/service lines to an editable SO. | Write | Exposed: supplemental order entry. |
| 28 | `cancelOrderLine` | PATCH `/sales/orders/{order_id}/lines/{line_id}/cancel` | Cancels an eligible line; current backend also releases its active reservations. | Write | Exposed: line removal. |
| 29 | `updateOrderLine` | PATCH `/sales/orders/{order_id}/lines/{line_id}/update` | Changes line quantity and/or price; currently supplied as query parameters. | Write | Core: quantity/price edits. |
| 30 | `shipOrder` | POST `/sales/orders/{order_id}/ship` | Previews or posts shipments against an SO and updates fulfillment/status. | Write; native preview | Core: normal dispatch. |

Preserve input types, bounds, optional filters, date handling, ambiguity errors, and bilingual fields. Support order IDs and internal SO strings wherever currently accepted. Preserve trace `product_id` disambiguation. Enrich sparse response schemas using actual handler contracts, not the YAML descriptions alone.

The initial MCP surface remains these **30 tools**. The floor-only `commitShipOrder` is not an additional operation in this inventory: preserve its proven approval/receipt behavior inside MCP `shipOrder`, backed by the legacy combined endpoint’s still-supported commit mode. Review floor-only capabilities separately if that GPT is also being replaced. Likewise packing slips, lot merges, document-upload/extraction endpoints, and administrative order exits are not additional tools in this 1:1 scope.

## 2. Write confirmation, receipts, and duplicate prevention

### A confirmation contract for all 16 writes

Keep one tool per operation and give each write tool an MCP-level `phase: preview | commit`, defaulting to preview. Translate it to the existing API `mode` only for the six native preview operations. For other writes, preparation uses reads and validation and does **not** call the mutating endpoint. Any required preview validation should be factored into shared backend logic during implementation; never create a temporary real order to obtain a preview.

1. Resolve the customer, product IDs, lots, quantities, prices, PO, dates, and target record. Display a concise summary of exactly what will change, including pallet charges and warnings.
2. Save a short-lived pending approval bound to the authenticated user, operation, canonical payload hash, environment, and relevant record versions. This is approval metadata, not a Factory Ledger business mutation.
3. Obtain **explicit approval of that exact summary**. Recommended enforceable baseline: an authenticated Factory Ledger approval page reached from the preview result, with Approve/Cancel. The page records the human decision server-side. A conversational “yes” or a model-supplied `confirmed: true` alone cannot prove approval to a remote server. Native client confirmation can improve the experience, but substitute it only if a verifiable per-call approval mechanism is demonstrated in both clients.
4. The same MCP tool’s commit phase supplies the pending approval ID. The server verifies approval, identity, expiry, payload, and record versions; rejects missing/changed/reused approvals; rechecks current business constraints; then calls the API. Changes to the proposal require a fresh approval. OAuth account consent is not transaction approval.
5. Persist approval consumption and the resulting receipt safely. No “Created,” “Updated,” or “Shipped” message until committed success is established. Declining or abandoning approval causes no business change.

Use `readOnlyHint: true` only on the 14 reads; use false on all 16 write-capable tools even during previews. Set destructive annotations according to actual consequences, including inventory posting and cancellation. Set `openWorldHint: false` for this bounded Factory Ledger account. Annotations guide clients but do not enforce approval. Do not carry forward the old workaround that labeled shipping commits non-consequential. [OpenAI MCP tool guidance](https://developers.openai.com/plugins/build/mcp-server)

Migrate instructions to remove “all info provided → no reconfirmation,” automatic order creation from uploads, and instructions to retry every failed write. Retain search-first, ambiguity handling, FIFO, traceability, warnings, bilingual output, and receipt-based success. A multi-step status change must preview the entire proposed sequence and report any partial completion; never imply it was atomic.

### Receipt requirements

Every successful write returns a normalized receipt with `success`, operation, request/approval ID, completion time, authenticated actor, changed record IDs, and meaningful resulting state. Retain the original API receipt fields and warnings.

| Write group | Required business receipt |
|---|---|
| `createOrder` | **Internal `order_number` (SO…), `order_id`, `customer_po`, customer, line IDs, amounts, pallet line, and resulting status.** Read the SO number produced by the database; never derive it from the PO or predict it. |
| `updateOrderHeader`, `updateOrderStatus`, `addOrderLines`, `cancelOrderLine`, `updateOrderLine`, `shipOrder` | **Internal SO number and order ID**, changed fields/line IDs and resulting state. Shipments also include shipment header ID, per-line transaction IDs, and confirmation codes when supplied. |
| `receive`, `ship`, `make`, `pack`, `adjust` | Transaction ID(s), confirmation code(s), relevant lot/record IDs, quantities and resulting balances/summary. Include a verified SO only when actually associated. |
| `createExpectedReceipt` | Expected-receipt ID, supplier/product, quantity, date, and status. |
| `createCustomer`, `updateCustomer` | Customer ID/name and changed fields. |
| `updateSupplierLot` | Internal lot identity and previous/new supplier-lot reference. |

Non-order operations do not inherently have an SO: return `order_number: null` where inapplicable, with the appropriate receipt above. Never invent an SO or create an unrelated order to satisfy the receipt format.

Some current order-line mutations omit `order_number`; resolve it from the verified order and enrich the result, or extend the backend receipt. Distinguish shipment-header IDs from per-line shipment-record IDs, a documented existing mismatch. Include `product_id` in order detail so pallet persistence can be verified. If a write succeeds but readback fails, report “saved; verification pending” with the received identifiers, rather than claiming full verification or resubmitting the write.

### Retry safety is additional build work

`IDEMPOTENCY_KEY_PLAN.md` explicitly remains a design artifact. Inspection found a limited safe retry helper for customer updates, not general durable request-key protection. PO duplicate warnings are not equivalent to idempotency.

Implement backend idempotency for all MCP write paths: a request key scoped to actor, operation, and payload; durable receipt storage **in the same database transaction as the mutation**; concurrency protection; replay returns the original receipt; changed payload with the same key fails. Cover order creation and receiving as well as ship/make/pack/adjust. An MCP-side cache alone cannot close the failure window after the API commits but before the adapter receives the response. Until this protection is verified, disable automatic write retries and reconcile uncertain outcomes using read tools.

## 3. Pallet charges and the dedicated PO field

These are required acceptance criteria, not optional follow-ups.

### Pallet Charge, product ID 176

- Persist every pallet charge as an actual `sales_order_lines` row with **`product_id = 176`**, using its quantity, unit price, and extended amount. Never save it only in notes, drop it as “non-inventory,” or choose a similarly named physical pallet product.
- Treat it as a service charge: units/each, no physical pounds, no stock reservation, FIFO allocation, lot creation, or production requirement. Preserve the current shipping behavior that auto-fulfills service lines and prevents service-only rows from bypassing the zero-physical-shipment guard.
- Product ID 176 is an owner-supplied requirement. Its current production name/active/service flags were not queried. Verify those flags read-only before implementation testing; do not silently recreate or repurpose the product if validation fails. A separate test database may need a fixture mapping.
- There is a real contract gap: root `OrderLineInput` accepts names, not product IDs, and its advertised units omit `each`. The backend stores service counts in the legacy `quantity_lb` column and interprets them as units on detail output. Add unambiguous service-quantity/product-ID handling at the API boundary and keep that legacy storage convention internal. Do not invent a case weight, present counts as pounds, or set the count to zero and lose the charge amount.
- Reconcile create/add/detail totals: `_create_sales_order_core` currently adds every line’s stored quantity to `total_lb`, while order detail excludes services from physical totals. The new receipt must not count pallets as product weight. Verify amount calculations, including zero prices, through create, edit, detail, packing slips, and fulfillment.

### Customer PO

- Save the PO in **`sales_orders.customer_po`**, preserving meaningful punctuation/leading zeros. It is distinct from the internal `order_number`, supplier delivery references, and free-text notes.
- Migration 050 already introduces this column; the shared `_create_sales_order_core(..., customer_po=...)` and dashboard document-approval path use it. Do not plan a second PO column.
- The legacy `OrderCreate`, `OrderHeaderUpdate`, and YAML schemas do **not** expose `customer_po`; `create_sales_order` does not pass it to the core. `getOrder` also omits it, although `listOrders` returns it. Extend create/header/detail contracts in the future implementation so the 1:1 MCP tools can save and read it. Merely adding an MCP argument would currently risk silently losing it.
- Reuse the existing normalized customer-plus-PO duplicate check and locking behavior for the new create path. Display conflicts; an intentional duplicate requires a separate explicit override in the approved proposal. Apply appropriate duplicate checks to PO/customer edits too.
- For entry from a customer PO, require the PO number before commit, consistent with the existing intake rule. For orders genuinely placed without a PO, have the owner choose whether those are allowed; if allowed, leave the dedicated field empty and record that decision, rather than inventing “N/A” as a purchase-order number.
- Persist order, PO, service lines, receipt, and idempotency record atomically. Read back and compare the PO and product 176 line before reporting verified completion.

## 4. Authentication and authorization

### Current GPT

The YAML declares global **API-key authentication in `X-API-Key`**. `main.py:verify_api_key` delegates to `_authorize_api_key`; the master credential comes from the `API_KEY` environment variable. Missing keys yield 401; invalid header keys yield 403. The app also supports a restricted dashboard key and actor keys, but these are route-allowlisted; `POST /sales/orders` is intentionally outside the dashboard allowlist. A generic dashboard key is therefore not a drop-in replacement for the legacy GPT credential.

This is the repository-defined authentication contract. The live GPT’s stored key and its current value were not inspected. The legacy packing-slip instruction embeds a static credential in a query-string link; do not copy that value into migrated skills or this document. Plan an authenticated/short-lived, narrowly scoped packing-slip link and coordinated credential rotation after dependent clients are identified. Do not rotate anything during planning.

### Recommended common approach: per-user OAuth

Use **OAuth authorization code with PKCE (S256), following MCP’s OAuth 2.1 authorization profile**, with a managed authorization provider. Each operator signs in; the MCP server validates tokens and authorizes each tool. This provides revocation, user attribution, and separate read/write permissions.

| Option | ChatGPT plugin | Claude remote custom connector | Decision |
|---|---|---|---|
| User OAuth | Supported and expected for authenticated MCP servers. | Supported for individual sign-in. | **Use this shared approach.** |
| Static API key/header | ChatGPT plugin documentation says custom API keys are not supported. Local-agent/API capabilities are a different integration surface. | Static request headers exist in a limited organization beta; not a dependable universal baseline. | Keep keys server-side for the adapter-to-ledger hop only. |
| No authentication | Technically possible for public tools. | Technically possible. | Inappropriate for private factory data and writes. |

[OpenAI plugin authentication](https://developers.openai.com/plugins/build/auth), [Claude connector authentication](https://claude.com/docs/connectors/building/authentication)

Implementation requirements:

1. Publish protected-resource metadata and authorization-server discovery; return an HTTP 401 with `WWW-Authenticate` when authentication is required. Use one canonical resource/audience and validate issuer, audience, expiry, and scopes on every request. Never pass a client token through to an unrelated upstream resource. [MCP authorization specification](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization)
2. Select a provider that supports PKCE, refresh/revocation, and compatible client registration. Prefer CIMD with public-client token authentication (`none`) for both hosts, with DCR available if needed; verify provider capabilities in a short integration spike. Copy the exact ChatGPT callback from its management page and support its issuer-identification requirements. Do not assume an existing Supabase database means a suitable OAuth authorization server is already configured. [OpenAI authentication requirements](https://developers.openai.com/plugins/build/auth)
3. For hosted Claude, allow its documented callback `https://claude.ai/api/mcp/auth_callback`. CIMD requires both `client_id_metadata_document_supported: true` and `none` in the advertised token-auth methods; otherwise Claude falls back to DCR. Return a real 401 to start sign-in and test refresh/reconnect. [Claude OAuth requirements](https://claude.com/docs/connectors/building/authentication)
4. Define permissions such as `ledger.read`, `orders.write`, `inventory.write`, `customers.write`, and `traceability.write`; map every write row to the narrowest permission. Limit users to this factory, resolve identity from validated tokens, and never trust a model-provided actor or `created_by` as identity. Preserve source tags separately from the human actor.
5. Give the adapter a dedicated, rotatable backend credential with an explicit route allowlist. The existing master key is too broad for the long-term bridge, while the current dashboard key cannot reach all 30 operations. Plan a dedicated MCP service-key path in Factory Ledger, and authenticated actor attribution/audit records. Do not expose this credential in either client, skills, receipts, URLs, or logs.

The common server should accept authorized traffic from both providers. If network allowlists or OpenAI client mTLS are added, do not accidentally exclude Claude. User OAuth remains the authorization boundary.

## 5. Hosting recommendation

**Recommend a separate MCP adapter service on Railway, in the existing project/region where practical, forwarding to Factory Ledger’s existing API.** Keep business logic and database writes in Factory Ledger. Use a stable HTTPS `/mcp` endpoint with Streamable HTTP, supported by both hosts. [OpenAI deployment guidance](https://developers.openai.com/plugins/build/mcp-server), [Claude server guidance](https://claude.com/docs/connectors/building)

Repository evidence identifies Railway for FastAPI/Uvicorn, Supabase PostgreSQL, and Netlify for the static dashboard. The action schema points at Railway, and incident/deployment records corroborate it. `DEPLOYMENT.md` contains older Render/bearer-token material; do not use that historical section to choose the architecture. The current Railway plan, instance settings, and production revision remain to be checked before building.

| Consideration | Mount MCP in existing `main.py` app | Separate Railway adapter |
|---|---|---|
| Deployment | One service and origin; can share internal business functions. | Independent start command, dependencies, logs, scaling, and rollback. |
| Dependencies | Existing FastAPI 0.104.1, Uvicorn 0.24.0, multipart 0.0.6 and explicitly constrained dependency stack require compatibility work. | Modern MCP/OAuth dependencies can be pinned separately. |
| Production impact | MCP lifecycle, streaming and auth changes share the operational app’s failure boundary. | Adapter failure stops assistant access while the ledger API/dashboard remain available. |
| Data access | Can reuse transactions directly, but must preserve FastAPI dependencies, actor checks, and startup lifecycle. | Calls the API; no direct access to business tables or duplicated FIFO/order logic. |
| Operational cost | Fewer services, but broader regression scope. | One additional service and network hop; some targeted API work still required. |

The official Python SDK’s current source requires newer AnyIO/Uvicorn/multipart dependencies than this app’s pins; its main branch is evidence of compatibility risk, not a version to install blindly. Resolve and lock an actual released SDK during the spike. That separation is the strongest reason to choose a second service here. [Official Python SDK dependency metadata](https://raw.githubusercontent.com/modelcontextprotocol/python-sdk/main/pyproject.toml)

Proposed request path: **ChatGPT plugin / Claude connector → OAuth-protected Railway MCP adapter → authenticated Factory Ledger API → existing Supabase transactions.** Approval records live in a dedicated store; the definitive mutation/idempotency receipt remains transactionally stored by the ledger backend. No business-table credentials in the adapter.

Operational requirements for the later build:

- Configure a dedicated service root/start command and watch patterns. Current `railway.json` watches root Python/config files, not a prospective nested MCP package; otherwise adapter changes may not deploy. Keep its environment separate from the legacy runtime.
- Check service sleep/cold starts, streaming/proxy timeouts, health checks, connection reuse, rate limits, and bounded concurrency against the ledger’s 2–20-connection pool. Do not assume a daily keepalive script proves readiness.
- Deploy additive backend changes first: PO/service fields, receipts, scoped bridge authentication, and durable idempotency. Verify schema prerequisites and use the existing guarded migration process. The separate service is not a claim that no backend work is needed.
- Add request IDs across both services and redacted audit events for previews, approvals, commits, errors, and replays. A tool that never reaches Railway must be distinguishable from a database failure.
- Prefer stateless MCP request handling where practical; keep approvals and idempotency durable across restarts/replicas. Maintain a read-only switch and a separate write-disable switch.

## 6. Risks and acceptance tests

| What could break | Mitigation / release requirement |
|---|---|
| Migrating the wrong GPT or losing a floor-only operation | Inventory live accounts/configurations and confirm scope. Replay office and floor examples; explicitly list any deferred floor-only features. |
| Client selects the wrong tool, stalls, or claims success without a request | Clear names/descriptions and exact-SO routing; receipt-only success; test actual ChatGPT and Claude conversations, not only direct API calls. Include the August dispatch and April status-change regressions. |
| Missing confirmation or a model forges a confirmation flag | Enforce authenticated, payload-bound approval; test no approval, deny, expiry, changed payload, different user, and replay. Client permission settings must not bypass it. |
| Timeout/retry creates duplicate orders or inventory movements | Transactional idempotency; concurrent same-key and dropped-response tests. An uncertain outcome is not permission to retry with a new key. |
| PO disappears or pallet charge is only a note | Read back `customer_po`, product ID 176, count, price and amount. Verify duplicate-PO handling and leading zeros. Test both hosts from the same example PO. |
| Pallets distort weight, stock shortages, or shipment completion | Service-aware totals/readiness; shipping auto-fulfillment regression tests; no service inventory movements or false service-only shipment success. |
| Unexpected customer creation | Current order creation calls customer resolution with auto-create enabled by default. The new path must require an existing resolved customer or separately approved `createCustomer`; revalidate at commit. |
| Case/lb conversion, wrong FG/private-label match, or missing warning | Require resolved identity and explicit units; preserve warning/disambiguation behavior and customer-specific restrictions from the newer intake flow where applicable. Never guess from OCR. |
| Wrong date, lot, state transition, or stale inventory | Preserve America/New_York event-time/backfill rules, FIFO/override rules, legal status transitions and row locks; reject stale proposals. Surface current readiness/blockers and supplier-lot ambiguity. |
| OAuth callback, expiry, role, or provider mismatch | Test first connection, refresh, reconnect, logout/revocation, wrong audience, read-only role attempting a write, and unauthorized user in both actual clients. |
| Shared-key exposure or loss of individual attribution | Server-side scoped bridge secret, per-user OAuth/audit, safe packing-slip access, coordinated rotation; redact logs and migrate no embedded keys. |
| Dependency/startup change breaks the ledger | Isolated adapter dependencies; backend regression checks in a test database. Do not start/import the production-configured app for a “read-only” check: startup performs migrations. |
| Large trace/history results or infrastructure limits cause truncation/timeouts | Preserve limits, bound response sizes, label truncation, and test large recall traces. Never silently turn incomplete results into a complete recall claim. |
| Instructions or knowledge files fail to carry over | Version a shared workflow guide; package it with the ChatGPT plugin and supply equivalent Claude guidance. A connector gives tools, not automatic transfer of GPT instructions/files. |
| Packing-slip workflow fails despite 30-tool parity | Replace credential-bearing URLs with authenticated or short-lived document access; test opening a slip as the intended operator. This companion web route need not add an MCP operation. |

Run all business-write tests against an isolated database with synthetic data. Existing targeted suites include `test_ship_order_service_line.py`, `test_sales_order_line_fields.py`, `test_sales_order_extract.py`, `test_sales_order_readiness.py`, `test_sales_order_state_model.py`, and `test_sales_order_allocations.py`; add contract/auth/approval/replay tests around the new adapter and backend changes. No tests were run as part of this planning-only task.

Go-live acceptance: exactly 30 unique mapped tools; all 14 reads work; all 16 writes refuse unapproved commits; real client order entry returns the saved internal SO and correct PO/pallet line; shipping returns real receipts; replay creates no duplicate; denied users cannot write; existing dashboard/API behavior passes regression testing. Preserve actionable 4xx/409 suggestions instead of replacing them with generic retries.

## 7. Build estimate, rollout, and owner decisions

Estimate for one engineer familiar with the repository, using a managed OAuth provider and an existing isolated test environment:

| Work | Engineering days |
|---|---:|
| Confirm live GPT usage; finalize contracts and OAuth/dependency spike | 1–2 |
| Separate Railway adapter and 30 tool mappings | 2–3 |
| OAuth, scopes, bridge credential and user attribution | 2–3 |
| Approval workflow, durable idempotency, PO/service/receipt corrections | 4–7 |
| Both-client acceptance, backend regression, packaging and rollout checks | 3–5 |
| **Total** | **12–20** |

Allow roughly **3–4 engineering weeks**, plus **one week of operator pilot/calendar buffer**. A read-only prototype could take 2–3 days, but it would not meet the order-entry requirement. Estimates exclude identity-provider procurement, unexpected database cleanup, public marketplace review, and replacing all floor-only/dashboard-only features. Bespoke identity infrastructure or substantial service-quantity redesign would add time; revise the estimate after the spike. Hosting/provider costs depend on the existing plans and have not been quoted.

Suggested milestones, to be agreed rather than scheduled by this document:

1. **By October 9:** identify live GPTs, owners, usage examples, access roles, and the OAuth provider; complete the compatibility spike.
2. **By November 6:** finish staging acceptance for all 30 tools and required business rules; prepare plugin instructions/knowledge and Claude guidance.
3. **November 9–13:** pilot with a small office/floor group. Compare reads freely, but designate only one client to commit each real business event; never dual-write for comparison.
4. **By November 20:** switch normal work after owner acceptance and workspace access checks. Verify each intended user can install/connect and authenticate.
5. **Before December 11:** retire obsolete entry points and coordinate legacy credential removal only after dependent systems are accounted for. Maintain a dashboard/manual fallback for critical operations; the original GPT cannot be the fallback after its retirement.

If the adapter fails, disable its writes and return operators to the existing API/dashboard workflows. Roll back the adapter independently; preserve committed business records and additive database fields. Reconcile pending/unknown writes before retrying. Before retirement, the old GPT may remain an operational fallback only if its exact configuration is still available and validated.

**Decisions for the owner:**

- Confirm whether scope is the personal Factory-Ledger 2.0 GPT, the workspace Floor & Fulfillment GPT, or both; supply the account-specific retirement notice.
- Approve the separate Railway-service approach, select the managed sign-in provider, and identify who may read, enter orders, and post inventory changes.
- Accept the authenticated approval-page baseline for all writes, or fund validation of a more integrated confirmation experience with equivalent server enforcement in both clients.
- Decide whether genuine no-PO orders are allowed; pallet product 176 and dedicated PO storage are already fixed requirements.
- Name the implementation owner and pilot operators; agree on the November 20 target and fallback owner.

This document authorizes no implementation, deployment, credential rotation, or production-data changes. Stop here for review.
