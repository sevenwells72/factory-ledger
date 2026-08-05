# 2026-08-05 — Floor GPT sales-order dispatch failures

## Summary

The Factory Ledger — Floor 2.0 Custom GPT could answer some read-only questions but repeatedly failed to look up or dispatch a named sales order. It sometimes claimed it had no live Factory Ledger access, and after a successful shipment preview it never sent the commit POST. The FastAPI service itself was healthy: direct curl calls to the combined sales-order ship endpoint previewed and committed correctly.

The incident crossed three independent layers: the GPT action schema did not route intent strongly enough, the ChatGPT client swallowed consequential commit calls before they reached Railway, and the GPT instructions still described the older single-endpoint workflow. All three layers had to be corrected.

## Symptoms

- An operator supplied an exact SO number, but the GPT did not reliably call `getOrder` or `shipOrder`.
- The GPT sometimes searched products for an SO number or responded that it lacked a live connection.
- `shipOrder` preview calls reached the API and returned valid feasibility data.
- After explicit operator approval, no commit request appeared in Railway logs. ChatGPT's consequential-action confirmation dialog did not render, so the call died in the client.
- Direct curl preview and commit requests worked, ruling out API health, authentication, FIFO, inventory, or database transaction handling as the primary cause.

## Root cause: three layers

### 1. OpenAPI action definition and routing

The earlier action definition lacked enough connection and routing precision: server/auth configuration was incomplete in the failing configuration, summaries overlapped product search, inventory, order lookup, and shipping intent, and the combined `shipOrder` operation tried to represent both a harmless preview and a consequential commit. ChatGPT could select the wrong action or decline to call one.

Fixes:

- Canonical Floor schema advanced from v4.0.0 to v4.1.0.
- The schema has an explicit Railway `servers` URL and global `X-API-Key` security scheme.
- Action summaries/descriptions now distinguish SKU search, inventory lookup, exact SO lookup, preview, and commit.
- `shipOrder` is preview-only in the GPT schema.
- Added operation 22, `commitShipOrder`, at `POST /sales/orders/{order_id}/ship/commit` with a required mode-less body containing `ship_all` or `lines`.
- Both operations carry `x-openai-isConsequential: false`.

### 2. ChatGPT consequential-action handling

The combined action's commit request was classified as consequential. In this workspace, ChatGPT attempted to pause for its confirmation dialog, but the dialog silently failed to appear. The request was therefore never sent; Railway had no corresponding POST or server error to diagnose.

Fixes:

- Split the GPT-facing workflow into a safe `shipOrder` preview call and a dedicated `commitShipOrder` mutation.
- Marked both operations `x-openai-isConsequential: false` to bypass the broken client confirmation UI.
- The new server endpoint delegates to the existing `ship_order` service using `mode="commit"`; FIFO allocation, row locking, shipment creation, status transitions, 409/422 receipts, and readonly error handling remain single-sourced.
- The legacy combined `/sales/orders/{order_id}/ship` endpoint remains backward compatible.

#### Safety trade-off

Disabling ChatGPT's platform confirmation removes a client-side safety barrier. Safety now rests on the explicit two-stage protocol: the GPT must call preview, show the operator quantities/shortages/warnings, and call `commitShipOrder` only after the operator explicitly says to commit, dispatch, or ship. Receipt-anchored success remains mandatory; the GPT may claim shipment only when the commit response includes the shipment and transaction receipts. This protocol must not be weakened when editing the schema or instructions.

### 3. GPT instruction drift

The deployed instructions predated the split and described every transaction as `mode: preview` followed by `mode: commit` on one action. They did not tell the GPT that `commitShipOrder` was available, so the model could refuse the action or send the wrong body.

Fixes:

- Added the critical rule `ALL 22 ACTIONS ARE LIVE` and prohibited false no-access claims.
- Distinguished single-endpoint inventory transactions from two-action sales-order dispatch.
- Added exact SO-number + dispatch routing directly to `shipOrder`, never product search.
- Rewrote SHIP as two stages: `shipOrder` preview, then mode-less `commitShipOrder` after explicit approval.
- Required the GPT to quote `shipment_id`, per-line `transaction_id` and confirmation code, and the new order status.

## Verification and receipts

- Feature commit: `5b1ae75`.
- Initial Railway code deployment: `cbfb0169-157c-460a-b69b-773eb5e8b907`; current deployment after documentation sync: `0045c81f-79ba-4242-b810-d14bb99ad64b`.
- Automated tests: 62/62 passed, including preview safety, dedicated commit, full/partial status transitions, `QTY_EXCEEDS_REMAINING`, and existing 409 behavior.
- Redocly: valid schema; 22 operations, 22 unique operation IDs, below the 30-operation cap.
- Disposable production smoke test: `SO-260805-001` previewed and committed 1 lb through the new endpoint, then transaction 1846 was voided and the order cancelled; inventory returned to its starting balance and the voided shipment disappeared from reporting.
- Live operator verification then completed the intended dispatch chain for `SO-260723-001`: shipment header ID 319, transaction IDs 1847 and 1848, with the order transitioned to `shipped`.

## Known ChatGPT/editor quirks

- The first message in a new GPT chat can stall without sending an action. A short follow-up or “nudge” commonly causes the call to proceed.
- The GPT editor's **Allow** button can be flaky or fail to render/respond even when an action is configured correctly.
- Prefer the **Auto** model for floor operations. **Thinking** is more prone to pauses and tool-call friction in this workflow.
- These are client/editor behaviors; check Railway HTTP logs before attributing a missing request to the API.

## Follow-ups

1. Reconcile shipment identifier naming. The commit response returns the shipment header ID (319), while `GET /sales/orders/{id}` currently labels per-line `sales_order_shipments` IDs (421/422) as `shipment_id`. Rename or expose both fields so operators do not see conflicting “Shipment ID” values.
2. Apply the same server/security configuration, unambiguous action summaries, live-action rule, and preview/commit instruction hardening to the Sales & Admin GPT.
3. Consider a dashboard receive form as a ChatGPT-independent fallback for critical floor receiving work.
4. Delete or clearly rename the legacy personal-account **Factory-Ledger 2.0** GPT so operators cannot enter the obsolete configuration accidentally.

## Operational rule

When diagnosing a future missing GPT action, verify all three layers independently: inspect the live GPT schema/instructions, check whether the ChatGPT client rendered or swallowed confirmation, and check Railway HTTP logs for an actual request. A healthy curl path proves the server, but it does not prove ChatGPT sent the action.
