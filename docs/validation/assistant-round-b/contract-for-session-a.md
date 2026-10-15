# Round B assistant response contract for Session A

Round B changes server-side interpretation, resolution and traceability. Session A owns `dashboard/fl-assistant.html`, its JavaScript/CSS and card rendering. All fields below are additive; existing `result`, `prepared`, receipt and draft identifiers remain available. Render server-provided text as text, not HTML. The backend keeps tickets and payload hashes private.

## Language and plain answers

`POST /assistant/turn` accepts optional `language: "en" | "es"`. User text determines the reply language, with the session language as a fallback. Both the turn and each returned card include `language`. Catalog names remain the real names printed on product and bag labels.

Read cards return a plain sentence before their structured data:

```json
{
  "kind": "read",
  "tool": "inventory_lookup",
  "status": 200,
  "language": "en",
  "summary": "Batch Classic Granola #9: 17,678 lb in 6 lots.",
  "message": "Batch Classic Granola #9: 17,678 lb in 6 lots.",
  "rows": [],
  "result": {}
}
```

Display `summary` first, then render `rows`. `result` retains the existing full response for details. Supported read tools include inventory, open orders, lot quantity, receipts, today's entries and shift summary. An empty result also receives a complete sentence.

Open-order rows have `order_id`, `order_number`, `customer`, `items`, `ship_date`, `status`, `state` and `remaining_lb`. `items` retains existing order endpoint pallet-line data, including product identity and quantity. `ship_date` may be null. The tool accepts customer/product filters in the worker's words, including short customer names and SKUs. Neither reading orders nor filtering them posts ledger transactions.

## Real choices and unknown wording

`kind: "choices"` always has real entries in `result.candidates`. Display the card's localized `message`/`result.ask`, then the returned candidate labels. Do not construct an empty choice card. Unknown wording returns `kind: "clarify"`, the original `query`, a localized `message` identifying what was not recognized, and candidate hints when available. The worker may follow with a product number or SKU as normal text.

Ordinary choices still submit `choice_id: card.id` and `selected_id: candidate.id` to `/assistant/turn`, with a new `turn_id` and the current `session_id`. An explicit product choice establishes the make output SKU; a second "Confirm this output SKU" step is unnecessary. This never confirms physical ingredient lots.

## Missing receive details and suppliers

A receive missing-details card can contain:

```json
{
  "kind": "missing_facts",
  "language": "es",
  "fields": ["supplier_id", "cases", "case_size_lb", "bol_reference"],
  "missing_details": [
    {"field": "supplier_id", "label": "proveedor"},
    {"field": "cases", "label": "número de cajas"}
  ],
  "known_facts": {"quantity_lb": 10, "product_id": 283},
  "summary": "Conservo la cantidad: 10 lb. Agrega los datos que faltan.",
  "message": "Agrega los datos que faltan.",
  "choice_id": "UUID",
  "supplier_choices": [{"id": 12, "name": "Grain Supply", "label": "Grain Supply"}]
}
```

Render `summary` when present, localized `missing_details` labels, and real `supplier_choices`. A supplier button submits `choice_id: card.choice_id` and `selected_id: supplier.id` through the existing turn route. The chosen supplier then has the same server provenance as a normal resolver choice. `known_facts.quantity_lb` preserves the worker's given pounds; do not overwrite it with a blank field or silently change it when entering cases. The server returns a localized mismatch summary if cases multiplied by case weight contradict those pounds. A missing supplier is expressed through normal missing details, not raw `SUPPLIER_REQUIRED` or endpoint text.

## One physical-lot card for the whole make

A make draft includes `prepared.draft.lot_confirmation_group`:

```json
{
  "kind": "lot_confirmation",
  "instruction": "Type the last 4 characters from the tag on each bag you used.",
  "instruction_en": "Type the last 4 characters from the tag on each bag you used.",
  "instruction_es": "Escriba los últimos 4 caracteres de la etiqueta de cada bolsa que usó.",
  "ingredients": [
    {
      "product_id": 283,
      "product_name": "Sunflower Seeds",
      "needed_lb": 10,
      "suggested_lots": [
        {"lot_id": 21, "lot_code": "SUNFLOWER-ABCD", "quantity_lb": 10, "confirmed": false}
      ],
      "candidates": [
        {"lot_id": 21, "lot_code": "SUNFLOWER-ABCD", "available_lb": 100}
      ],
      "different_lot": {
        "method": "different_lot",
        "label": "Different lot",
        "label_en": "Different lot",
        "label_es": "Otro lote"
      }
    }
  ]
}
```

Show every ingredient in this one card, including any ingredient with just one listed lot. `suggested_lots` are FIFO suggestions, and `candidates` are real available lots in FIFO order. Every suggestion begins unconfirmed. An ingredient may need more than one suggested lot; collect evidence for every used input row. Keep each field empty until the worker enters evidence from the physical tag. The server enforces confirmation even when a single system lot exists.

The normal record request remains:

```json
{
  "draft_id": "UUID",
  "acknowledged_warnings": [],
  "lot_confirmations": [
    {"lot_id": 21, "method": "last4", "value": "ABCD"}
  ]
}
```

If the physical tag is not among the listed lots, the worker chooses `different_lot` and supplies the actual tag or tag ending. Send that evidence using the suggested input's `lot_id`, `method: "different_lot"`, and `value` containing what the worker read. The backend records the actual unresolved input on a separate trace lot and opens the existing follow-up exception. It never debits the unverified FIFO suggestion or calls the physical lot confirmed. The receipt returns `lot_exceptions`, including `exception_id`, `physical_tag`, `suggested_lot_id`, `suggested_lot_code`, product identity and quantity. Show the recorded make and its follow-up exception plainly. Extra-kosher provenance requirements still apply.

Wrong tag endings are rejected before any ledger post. Missing tags remain `LOT_NOT_CONFIRMED`. A rejection's localized `message` explains the correction. Explicit full codes/scans remain supported where the existing lot control supports them.

## Other wording

Display localized `message` on permanent refusals. For example, `ORDER_ALREADY_FULFILLED` says "This order is already fully shipped." and does not invite retrying. Out-of-scope text says "I can only record or look up Factory Ledger items." / "Solo puedo registrar o consultar elementos de Factory Ledger."

## Validation boundary

`tests/visual/run-assistant-round-b.mjs` runs a browser through a loopback HTTP bridge into actual TestClient handlers and a fresh local PostgreSQL database for each session mode. Only model tool selection is stubbed. It validates this data contract and database consequences. Existing `run-fl-assistant.mjs` covers the current dashboard renderer in both session modes. Session A must connect its presentation changes to this contract for the new unified lot-card layout to be visible.

## Integration with the customer-ticket PR

PR #130 currently offers **New customer** after an empty `choices` result. Round B intentionally replaces that result with `clarify` for `resolution_kind: "customer"`, retaining the original `query`. When rebasing that PR, attach its New customer action to this clarification case and keep real-choice cards nonempty. Its server prepare/receipt hooks remain separate from the Round B changes. The historical customer-ticket test expecting empty choices must adopt this contract.

For extra-kosher output makes with regular ingredients, an unmatched physical tag uses the existing owner-approval hold before posting. An ingredient whose catalog or lot itself is extra-kosher still needs provenance review; Different lot cannot create extra-kosher ingredient identity from an unknown bag.
