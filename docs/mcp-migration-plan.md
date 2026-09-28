# Factory Ledger MCP migration plan

## Plain-English summary

- Replace both active GPTs: **Factory-Ledger 2.0** (office) and **Factory Ledger - Floor 2.0** (floor). One MCP service exposes separate office and floor connections in ChatGPT and Claude.
- Run that connection as a separate service on Railway, alongside the existing Factory Ledger app.
- Use Google sign-in plus an email allowlist and roles from a Railway environment variable, with server-enforced office/floor permissions; unlisted users are denied.
- Require explicit approval before every change, and return the internal SO number for every order change.
- Save pallet charges as Pallet Charge line items, product 176, and customer PO numbers in the dedicated PO field.
- Michael Gross owns the migration and pilots first for about one week, followed by Luz or Miriam (office) plus Arturo (floor), then all staff.
- Budget **16–26 engineering days plus staged pilot time** for both GPTs; target cutover is **November 20, 2026**, ahead of the supplied December 11 deadline.

## Scope and evidence

Originally prepared September 28, 2026 against `54df62b`; corrected and expanded on the same date, with Phase 1 built on branch `feature/mcp-server` from local `origin/main` at `6480c91`. The owner explicitly authorized the isolated read-only build below. No production credentials, production data or deployments are exercised. Existing OpenAPI files and `main.py` behavior remain unchanged; real handlers are exercised only against disposable local test databases.

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

**Owner correction (2026-09-28): two GPTs are actively used and both must be replaced.** The office GPT is **Factory-Ledger 2.0**, inventoried from `openapi-gpt-v3.yaml`. The floor GPT is **Factory Ledger - Floor 2.0**, mapped to `gpt-configs/schemas/openapi-floor.yaml` (schema title “Factory Ledger — Floor & Fulfillment”, v4.1.0). The earlier incident's recommendation to retire the personal office GPT does not override the owner's current statement that it remains active. Repository README statements about a single GPT/planned three-GPT split are historical, not current migration scope.

The Floor schema was found, so the Step 1 stop condition does **not** apply. Its 22 operations are inventoried below. Exact live-editor parity remains a later acceptance check; no live GPT editor or traffic logs were inspected. Collect representative office/floor conversations and request counts for acceptance testing without dropping less-frequent operations.

## Decisions (2026-09-28)

- **Architecture:** one MCP server deployed as a separate Railway service, with two tool groups, `office` and `floor`. Expose them as **two separate ChatGPT plugins and two separate Claude connector entries**, using `/office/mcp` and `/floor/mcp` on the same service. Registration/visibility alone never grants permission; enforce roles at the server.
- **Authentication:** Google sign-in. Email allowlist and roles come from a **Railway environment variable**, never email addresses in code. Validate the Google identity first, then look up the verified email in the server-managed allowlist; a client-supplied email/domain does not authenticate anyone. Unlisted users are denied, including authenticated CNS users.
- **Owner:** Michael Gross.
- **Roles:** `admin_office_floor` (Michael, Luz, Miriam) = office + floor write, including reads; `floor` = floor write + office read, including **read-only office order reads**. Unlisted = denied. Phase 1 still exposes no business writes.
- **Pilot:** Michael for approximately one week → Luz or Miriam (office) + Arturo (floor) → all staff.
- **Target cutover:** **November 20, 2026**.
- **No-PO orders:** allowed, flagged **"No PO"** in summaries/receipts and displays. Leave `customer_po` empty; do not store the flag as a fabricated PO number.
- **Write confirmation:** every write first returns a **plain-English summary + one-time confirmation token** and saves only on a **second call with that token**. Bind the token to the authenticated user, exact operation/payload and relevant versions; reject expired, changed, cross-user or reused tokens.
- **Phase 1 authorization:** branch `feature/mcp-server`; separate service directory; **read-only tools for both groups only**; local/test database tests; explicit Google OAuth TODO; no production credentials. **No deployment, production-data access, changes to existing OpenAPI files, or `main.py` behavior.** Later writes, OAuth rollout and production access remain separate work.

| Role / user class (verified Google identity + allowlist required) | Office group | Floor group |
|---|---|---|
| `admin_office_floor` — Michael, Luz, Miriam | Read + write (writes in a later phase) | Read + write (writes in a later phase) |
| `floor` — allowlisted floor users | Read only, including office orders | Read + write (writes in a later phase) |
| Unlisted / unauthenticated | Denied | Denied |

The existing Phase 1 demo stub predates this decision and still uses synthetic
`reader`/`admin`/`floor` labels with its older group boundary. It does not implement
Google login or the decided user-role mapping. The auth workstream must replace that
boundary with this contract; no real user assignment or write is enabled here.

## 1. Complete operation inventories and proposed MCP tools

### Office GPT — Factory-Ledger 2.0

**One row equals one existing operation and one proposed tool within its group.** Keep the existing operationId as the MCP tool name; the two MCP URLs disambiguate shared names. The endpoint column is relative to the current action server, `https://fastapi-production-b73a.up.railway.app`. “Write” means explicit, payload-specific confirmation is required before committing, including writes without inventory effects.

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

### Floor GPT — Factory Ledger - Floor 2.0

Source: `gpt-configs/schemas/openapi-floor.yaml`, declared version **4.1.0**. **22 operations: 12 business reads (11 GETs plus a POST shipping preview) and 10 write-capable operations.** Classification follows behavior, not HTTP verb or the old `x-openai-isConsequential` hint. Mixed preview/commit operations count as writes.

| # | Operation → floor tool name | HTTP endpoint | Read/write | Overlap with office |
|---|---|---|---|---|
| 1 | `searchProducts` | GET `/products/search` | Read | Yes |
| 2 | `listProducts` | GET `/bom/products` | Read | **Floor only** |
| 3 | `getBatchFormula` | GET `/bom/batches/{batch_id}/formula` | Read | Yes |
| 4 | `inventoryLookup` | GET `/inventory/lookup` | Read | Yes |
| 5 | `getLotByCode` | GET `/lots/by-code/{lot_code}` | Read | Yes |
| 6 | `getLotsBySupplierLot` | GET `/lots/by-supplier-lot/{supplier_lot_code}` | Read | **Floor only** |
| 7 | `updateSupplierLot` | PATCH `/lots/{lot_code}/supplier-lot` | Write | Yes |
| 8 | `renameLot` | PATCH `/lots/{lot_id}/rename` | Write | **Floor only** |
| 9 | `receive` | POST `/receive` | Write; preview/commit | Yes |
| 10 | `ship` | POST `/ship` | Write; preview/commit | Yes |
| 11 | `make` | POST `/make` | Write; preview/commit | Yes |
| 12 | `pack` | POST `/pack` | Write; preview/commit | Yes |
| 13 | `adjust` | POST `/adjust` | Write; preview/commit | Yes |
| 14 | `voidTransaction` | POST `/void/{transaction_id}` | Write | **Floor only** |
| 15 | `traceSupplierLot` | GET `/trace/supplier-lot/{supplier_lot_code}` | Read | Yes |
| 16 | `getTransactionHistory` | GET `/transactions/history` | Read | Yes |
| 17 | `getDaySummary` | GET `/production/day-summary` | Read | Yes |
| 18 | `listOrders` | GET `/sales/orders` | Read | Yes |
| 19 | `getOrder` | GET `/sales/orders/{order_id}` | Read | Yes |
| 20 | `updateOrderStatus` | PATCH `/sales/orders/{order_id}/status` | Write | Yes |
| 21 | `shipOrder` | POST `/sales/orders/{order_id}/ship` | Read — preview only | Same name/path; office allows commit, Floor allows preview only |
| 22 | `commitShipOrder` | POST `/sales/orders/{order_id}/ship/commit` | Write | **Floor only** |

**Overlap:** 17 operation names/endpoints appear in both schemas. Nine are shared read operations; seven are shared write-capable operations; `shipOrder` is the remaining shared name with different permissions. The Floor schema restricts it to preview but the underlying combined route can commit. Phase 1 routes Floor `shipOrder` to the existing **`POST /sales/orders/{order_id}/ship/preview`** wrapper, validates `mode: preview`, and also overwrites the forwarded mode with `preview`. This avoids exposing the combined route's commit capability.

**Five Floor-only operations:** `listProducts`, `getLotsBySupplierLot`, `renameLot`, `voidTransaction`, `commitShipOrder`. The first two are reads; the final three are writes deferred from Phase 1. Preserve the Floor `shipOrder` → approval → `commitShipOrder` workflow for later write parity; do not merge it into the office combined action. Floor `getLotByCode` also advertises optional `product_id` disambiguation missing from the office schema; keep group-specific argument contracts.

**Combined scope:** 52 operation registrations across the two groups (30 office + 22 floor), representing 35 unique operation names. Phase 1 exposes **26 read-only registrations: 14 office + 12 floor**, with 17 unique names. The remaining 26 write-capable registrations (16 office + 10 floor) are deferred. Do not apply the legacy GPT 30-action cap to the combined MCP service. Other dashboard-only operations are outside this inventory.

### Older / third GPT schemas — retired

The repository contains these three additional, already archived schemas. Mark all **RETIRED — historical reference only; not a third active GPT and not a migration target**:

| Repository file | Declared version | Operations | Status |
|---|---|---:|---|
| `archive/superseded-schemas/openapi-schema.yaml` | 2.7.0 | 35 | Retired |
| `archive/superseded-schemas/openapi-schema-gpt.yaml` | 2.7.0 | 32 | Retired |
| `archive/superseded-schemas/openapi-v3.yaml` | 3.3.0 | 33 | Retired |

A search of tracked files and the working checkout found no other OpenAPI schema. `gpt-configs/README.md` mentions Sales & Admin and Trace & Recall as planned, not built; their separate schemas are absent. This confirms repository artifacts, not remote deletion of any GPT. Both active GPT schemas remain untouched.

## 2. Write confirmation, receipts, and duplicate prevention

### Future confirmation contract for all 26 write-capable group registrations

For the later write phase, keep one tool per operation per group and give write-capable tools an MCP-level `phase: preview | commit`, defaulting to preview. Preserve the distinct Floor `shipOrder` read-only preview and `commitShipOrder` write action; the latter must consume approval for the exact preview payload. None of these writes is implemented in Phase 1. Translate it to the existing API `mode` only for the six native preview operations. For other writes, preparation uses reads and validation and does **not** call the mutating endpoint. Any required preview validation should be factored into shared backend logic during implementation; never create a temporary real order to obtain a preview.

1. Resolve the customer, product IDs, lots, quantities, prices, PO, dates, and target record. The first call returns a **plain-English summary of exactly what will change and a one-time confirmation token**, including pallet charges, "No PO" where applicable, and warnings. It makes no business change.
2. Store a short-lived pending confirmation bound to the authenticated user, operation, canonical payload hash, environment, and relevant record versions. This is approval metadata, not a Factory Ledger business mutation. Never log the token.
3. Show the summary and obtain explicit operator approval. Only then make a **second call with the token**; a boolean `confirmed: true` is not a substitute. This two-call contract supersedes the earlier proposed approval-page baseline; a token enforces a second, payload-bound call but is not independently proof of human intent, so clients must still present the summary and require approval.
4. Verify the token, identity, expiry, payload and record versions; reject missing, changed, cross-user, expired or reused tokens. Recheck current business constraints before saving. Changes to the proposal require a fresh summary/token. OAuth consent is not transaction approval.
5. Consume the token exactly once with durable replay protection and persist the resulting receipt safely. Save only on the second call. No "Created," "Updated," or "Shipped" message until committed success is established. Declining or abandoning confirmation causes no business change.

Use `readOnlyHint: true` on the 26 read-only group registrations; use false on all 26 future write-capable registrations even during their previews. Set destructive annotations according to actual consequences, including inventory posting and cancellation. Set `openWorldHint: false` for this bounded Factory Ledger account. Annotations guide clients but do not enforce approval. Do not carry forward the old workaround that labeled shipping commits non-consequential. [OpenAI MCP tool guidance](https://developers.openai.com/plugins/build/mcp-server)

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
| Floor `renameLot` | Lot ID, product ID, previous/new internal lot code. |
| Floor `voidTransaction` | Voided transaction ID, reason and affected records; preserve append-only semantics. |
| Floor `commitShipOrder` | Internal SO number, order ID, shipment ID, per-line transaction IDs and resulting order status. |

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
- Product ID 176 is an owner-supplied requirement. Its current production name/active/service flags were not queried. Use synthetic fixtures during Phase 1; production flag verification is a later, separately authorized pre-cutover check. Do not silently recreate or repurpose the product if validation fails. A separate test database may need a fixture mapping.
- There is a real contract gap: root `OrderLineInput` accepts names, not product IDs, and its advertised units omit `each`. The backend stores service counts in the legacy `quantity_lb` column and interprets them as units on detail output. Add unambiguous service-quantity/product-ID handling at the API boundary and keep that legacy storage convention internal. Do not invent a case weight, present counts as pounds, or set the count to zero and lose the charge amount.
- Reconcile create/add/detail totals: `_create_sales_order_core` currently adds every line’s stored quantity to `total_lb`, while order detail excludes services from physical totals. The new receipt must not count pallets as product weight. Verify amount calculations, including zero prices, through create, edit, detail, packing slips, and fulfillment.

### Customer PO

- Save the PO in **`sales_orders.customer_po`**, preserving meaningful punctuation/leading zeros. It is distinct from the internal `order_number`, supplier delivery references, and free-text notes.
- Migration 050 already introduces this column; the shared `_create_sales_order_core(..., customer_po=...)` and dashboard document-approval path use it. Do not plan a second PO column.
- The legacy `OrderCreate`, `OrderHeaderUpdate`, and YAML schemas do **not** expose `customer_po`; `create_sales_order` does not pass it to the core. `getOrder` also omits it, although `listOrders` returns it. Extend create/header/detail contracts in the future implementation so the 1:1 MCP tools can save and read it. Merely adding an MCP argument would currently risk silently losing it.
- Reuse the existing normalized customer-plus-PO duplicate check and locking behavior for the new create path. Display conflicts; an intentional duplicate requires a separate explicit override in the approved proposal. Apply appropriate duplicate checks to PO/customer edits too.
- For entry from a supplied customer PO, capture its number before commit. Orders genuinely placed without a PO are **allowed**: keep the dedicated field empty and flag **"No PO"** in the preview, receipt and display. Do not invent “N/A” or store “No PO” as the PO number.
- Persist order, PO, service lines, receipt, and idempotency record atomically. Read back and compare the PO and product 176 line before reporting verified completion.

## 4. Authentication and authorization

### Current GPTs

The YAML declares global **API-key authentication in `X-API-Key`**. `main.py:verify_api_key` delegates to `_authorize_api_key`; the master credential comes from the `API_KEY` environment variable. Missing keys yield 401; invalid header keys yield 403. The app also supports a restricted dashboard key and actor keys, but these are route-allowlisted; `POST /sales/orders` is intentionally outside the dashboard allowlist. A generic dashboard key is therefore not a drop-in replacement for the legacy GPT credential.

This is the repository-defined authentication contract. The live GPT’s stored key and its current value were not inspected. The legacy packing-slip instruction embeds a static credential in a query-string link; do not copy that value into migrated skills or this document. Plan an authenticated/short-lived, narrowly scoped packing-slip link and coordinated credential rotation after dependent clients are identified. Do not rotate anything during planning.

### Decided common approach: Google sign-in with per-user OAuth

Use **OAuth authorization code with PKCE (S256), following MCP’s OAuth 2.1 authorization profile**, with a compatible authorization provider using Google as the identity source, restricted by a server-managed email allowlist and role mapping supplied through a Railway environment variable. No email addresses belong in code; Google/Workspace membership alone is insufficient and unlisted users are denied. Each operator signs in; the MCP server validates tokens and authorizes each tool. This provides revocation, user attribution, and separate read/write permissions.

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
4. Define group scopes `office.read`, `office.write`, `floor.read`, and `floor.write` per the Decisions table; map every operation to its group permission and optionally narrower action permissions. Limit users to this factory, resolve identity from validated tokens, and never trust a model-provided actor or `created_by` as identity. Preserve source tags separately from the human actor.
5. Give the adapter a dedicated, rotatable backend credential with an explicit route allowlist. The existing master key is too broad for the long-term bridge, while the current dashboard key cannot reach all 30 operations. Plan a dedicated MCP service-key path in Factory Ledger, and authenticated actor attribution/audit records. Do not expose this credential in either client, skills, receipts, URLs, or logs.

The common server should accept authorized traffic from both providers. If network allowlists or OpenAI client mTLS are added, do not accidentally exclude Claude. User OAuth remains the authorization boundary.

## 5. Hosting recommendation

**Recommend a separate MCP adapter service on Railway, in the existing project/region where practical, forwarding to Factory Ledger’s existing API.** Keep business logic and database writes in Factory Ledger. Use stable HTTPS `/office/mcp` and `/floor/mcp` endpoints with Streamable HTTP, one per client registration. Phase 1 serves these paths locally only. [OpenAI deployment guidance](https://developers.openai.com/plugins/build/mcp-server), [Claude server guidance](https://claude.com/docs/connectors/building)

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

- The branch-only `.railway/railway.ts` partial declares the new MCP service: separate `/mcp_server` root, Docker build, explicit start command, `/health`, restart policy, and MCP/config watch paths. Existing services and root `railway.json` remain unchanged. Railway no longer accepts JSON/TOML config for new services, with a December 1, 2026 cutoff for existing users; legacy-service migration is separate work. No Railway plan or apply was run. See the [IaC reference](https://docs.railway.com/infrastructure-as-code/reference).
- Check service sleep/cold starts, streaming/proxy timeouts, health checks, connection reuse, rate limits, and bounded concurrency against the ledger’s 2–20-connection pool. Do not assume a daily keepalive script proves readiness.
- Deploy additive backend changes first: PO/service fields, receipts, scoped bridge authentication, and durable idempotency. Verify schema prerequisites and use the existing guarded migration process. The separate service is not a claim that no backend work is needed.
- Add request IDs across both services and redacted audit events for previews, approvals, commits, errors, and replays. A tool that never reaches Railway must be distinguishable from a database failure.
- Prefer stateless MCP request handling where practical; keep approvals and idempotency durable across restarts/replicas. Maintain a read-only switch and a separate write-disable switch.

## 6. Risks and acceptance tests

| What could break | Mitigation / release requirement |
|---|---|
| Migrating the wrong GPT or losing a floor-only operation | Both GPTs are now confirmed in scope. Verify live-editor parity with the two repository schemas, replay office and floor examples, and include all five Floor-only operations in later acceptance. |
| Client selects the wrong tool, stalls, or claims success without a request | Clear names/descriptions and exact-SO routing; receipt-only success; test actual ChatGPT and Claude conversations, not only direct API calls. Include the August dispatch and April status-change regressions. |
| Missing confirmation or a model forges a confirmation flag | Enforce authenticated, payload-bound approval; test no approval, deny, expiry, changed payload, different user, and replay. Client permission settings must not bypass it. |
| Timeout/retry creates duplicate orders or inventory movements | Transactional idempotency; concurrent same-key and dropped-response tests. An uncertain outcome is not permission to retry with a new key. |
| PO disappears or pallet charge is only a note | Read back `customer_po`, product ID 176, count, price and amount. Verify duplicate-PO handling and leading zeros. Test both hosts from the same example PO. |
| Pallets distort weight, stock shortages, or shipment completion | Service-aware totals/readiness; shipping auto-fulfillment regression tests; no service inventory movements or false service-only shipment success. |
| Unexpected customer creation | Current order creation calls customer resolution with auto-create enabled by default. The new path must require an existing resolved customer or separately approved `createCustomer`; revalidate at commit. |
| Case/lb conversion, wrong FG/private-label match, or missing warning | Require resolved identity and explicit units; preserve warning/disambiguation behavior and customer-specific restrictions from the newer intake flow where applicable. Never guess from OCR. |
| Wrong date, lot, state transition, or stale inventory | Preserve America/New_York event-time/backfill rules, FIFO/override rules, legal status transitions and row locks; reject stale proposals. Surface current readiness/blockers and supplier-lot ambiguity. |
| OAuth callback, expiry, role, or provider mismatch | Test first connection, refresh, reconnect, logout/revocation, wrong audience, floor role attempting an office write, and unlisted user in both actual clients. |
| Shared-key exposure or loss of individual attribution | Server-side scoped bridge secret, per-user OAuth/audit, safe packing-slip access, coordinated rotation; redact logs and migrate no embedded keys. |
| Dependency/startup change breaks the ledger | Isolated adapter dependencies; backend regression checks in a test database. Do not start/import the production-configured app for a “read-only” check: startup performs migrations. |
| Large trace/history results or infrastructure limits cause truncation/timeouts | Preserve limits, bound response sizes, label truncation, and test large recall traces. Never silently turn incomplete results into a complete recall claim. |
| Instructions or knowledge files fail to carry over | Version a shared workflow guide; package it with the ChatGPT plugin and supply equivalent Claude guidance. A connector gives tools, not automatic transfer of GPT instructions/files. |
| Packing-slip workflow fails despite office/floor tool parity | Replace credential-bearing URLs with authenticated or short-lived document access; test opening a slip as the intended operator. This companion web route need not add an MCP operation. |

Run all business-write tests against an isolated database with synthetic data. Existing targeted suites include `test_ship_order_service_line.py`, `test_sales_order_line_fields.py`, `test_sales_order_extract.py`, `test_sales_order_readiness.py`, `test_sales_order_state_model.py`, and `test_sales_order_allocations.py`; add contract/auth/approval/replay tests around the new adapter and backend changes. Phase 1 adapter verification is described below. Existing backend write suites are deferred; the integration harness starts the unmodified legacy app in a separate process against a disposable local test database. The adapter runtime never imports it.

Full go-live acceptance: 30 office and 22 floor registrations (35 unique names); all 26 read registrations work; all 26 write-capable registrations refuse unapproved commits; real client order entry returns the saved internal SO and correct PO/pallet line; shipping returns real receipts; replay creates no duplicate; denied users cannot write; existing dashboard/API behavior passes regression testing. Preserve actionable 4xx/409 suggestions instead of replacing them with generic retries.

## 7. Build estimate, rollout, and owner decisions

Estimate for one engineer familiar with the repository, using a managed OAuth provider and an existing isolated test environment:

| Work | Engineering days |
|---|---:|
| Reconcile both GPT contracts, floor-only workflows, Google OAuth/dependency spike | 2–3 |
| Separate Railway service, two group catalogs, 52 registrations / 35 unique names | 3–5 |
| Google OAuth, CNS restrictions, group roles, bridge credential and attribution | 3–4 |
| Approval workflow, durable idempotency, PO/service/receipt corrections | 4–7 |
| Both-client office/floor acceptance, regressions, two packages and rollout checks | 4–7 |
| **Total** | **16–26** |

Allow roughly **3–5 engineering weeks**, plus **about two calendar weeks for staged pilots**. The original 12–20 days covered only the office inventory; the expanded 16–26 days includes Floor-only tools, group permissions, separate client registrations, and floor workflow acceptance. Phase 1 read-only scaffolding is not full order-entry or floor-write parity. Estimates exclude identity-provider procurement, unexpected database cleanup, public marketplace review, and dashboard-only features. All 22 Floor-schema operations are now included in the estimate. Bespoke identity infrastructure or substantial service-quantity redesign would add time; revise the estimate after the spike. Hosting/provider costs depend on the existing plans and have not been quoted.

Target milestones (owner: Michael Gross; dates are targets, not scheduled automation):

1. **By October 9:** configure the env-managed email/role allowlist, Google OAuth provider integration and representative prompts for both confirmed GPTs.
2. **By October 30:** complete staging acceptance for office and floor scope, including later write safeguards; prepare two ChatGPT plugins and two Claude connection guides.
3. **November 2–6:** Michael pilots first, for approximately one week. Do not start write pilots until approval, auth and idempotency acceptance passes.
4. **November 9–13:** expand to Luz or Miriam (office) plus Arturo (floor). Compare reads; only one client may commit each real event, never dual-write for comparison.
5. **By November 20:** expand to all staff after Michael accepts results and each intended user can authenticate with the correct group access.
6. **Before December 11:** retire both replaced GPT entry points and coordinate legacy credential removal after dependencies are accounted for. Preserve an API/dashboard/manual fallback.

If the adapter fails, disable its writes and return operators to the existing API/dashboard workflows. Roll back the adapter independently; preserve committed business records and additive database fields. Reconcile pending/unknown writes before retrying. Before retirement, the old GPT may remain an operational fallback only if its exact configuration is still available and validated.

**Remaining configuration / information for Michael:**

- Supply the verified Google email allowlist and role assignments through the Railway environment variable; configure the managed OAuth provider. Do not add the emails to repository files.
- Choose Luz or Miriam for the office pilot, name the fallback operator, and supply the account-specific retirement notice. Arturo is the floor pilot.
- Implement and validate the decided two-call confirmation-token contract in both clients. Product 176, dedicated PO storage, allowed "No PO" orders and floor office-read access are settled requirements.

## 8. Phase 1 implementation and verification

Branch: `feature/mcp-server`. New service: `mcp_server/`. One process exposes `/office/mcp` (14 reads) and `/floor/mcp` (12 reads), each with an independent catalog. No write tool is registered, including for the simulated admin role. Existing schemas are inputs to a generated, read-only JSON catalog and are not edited. Tool names/arguments remain group-specific.

The adapter makes bounded HTTP calls to a numeric loopback test API only, does not follow redirects or environment proxies, and never reads database credentials or imports `main.py`. Non-read operations are absent; POST is allowlisted only for office product resolution and Floor's dedicated shipping preview. JSON Schema validation rejects additional arguments and commit mode; path parameters cannot escape the selected route. Tool annotations are read-only and non-destructive.

Google OAuth is an explicit TODO. Authentication defaults to locked; the opt-in bearer-token stub runs only on loopback in local/test mode and refuses Railway/production. Its older synthetic floor role cannot access the office group; this is a demo-only boundary, not the decided `floor` role contract. The auth workstream must permit office reads for allowlisted floor users. The current stub does not authenticate CNS users or enable hosted client installation.

### Read-only boundary and accepted usage metadata exception

**Owner decision (2026-09-28): read-only means no ledger, order, inventory, lot, or
product changes. `actors.last_used_at` updates are permitted usage metadata.**
An actor backend key invokes the existing, throttled `_touch_actor_last_used()` during
authentication. We keep that behavior; no `main.py` change is needed. Backend actor keys
retain the real route allowlist (for example, product resolution is forbidden); they do
not confer Google identity or substitute for future MCP user authorization.

### MCP contract overrides (OpenAPI unchanged)

The catalog generator explicitly overrides `getLotByCode` in both groups with optional
integer query parameter `product_id`. Office's YAML omits it, but the real endpoint
requires it to resolve a 409 ambiguity. Return the ambiguity matches, then repeat the
lookup with the selected product ID. Source parity includes this documented override.

Floor shipping preview rejects `ship_all=true` together with `lines`; explicit `lines`
must be nonempty and require `ship_all=false` or omission. This prevents the backend
from previewing all remaining quantities when the caller requested selected lines.
Remove the inherited suggestion to call unavailable `commitShipOrder` from the MCP
parameter description. Preview still calls the existing preview-only wrapper.

### Cross-review validation and limitations

The service retains the independent lockfile, Dockerfile, health endpoint, synthetic
SQLite demo and routing/validation tests. The new reusable integration harness creates
a private PostgreSQL cluster from `tests/schema/schema.sql`, clones a fresh database
per test, and runs real `main.py` startup/handlers/auth in a separate dependency process.
It accepts no external database URL and forwards no inherited secrets. Each session has
its own socket/port, allowing the auth and write worktrees to reuse it concurrently.

Real integration scenarios cover missing/wrong backend keys (401/403), 10-lb single-line
versus two 100-lb full-order previews, conflicting inputs, and ambiguous/disambiguated lot
lookups in both groups. All 26 read registrations have explicit scenarios independent of
the generated catalog: full table/row/column and sequence snapshots allow only the actor's
`last_used_at` change. The actor's existing product-resolution restriction returns 403;
the synthetic master key exercises that endpoint successfully with no database change.

The new project-level Railway TypeScript declaration is a scoped partial managing only
`factory-ledger-mcp`; root legacy config stays byte-identical. Local SDK type-check and
evaluation do not validate remote reconciliation. Google OAuth, full write workflows,
production data, live ChatGPT/Claude linking, Docker execution and deployment remain
unverified. Run and harness details are in `mcp_server/README.md`.

**Local only — not deployed.** No production access, push or merge is performed.

Cross-review validation (2026-09-28): **98 tests passed** (91 routing/validation, 7 real-ledger integration cases); Ruff lint/format and package builds passed.
