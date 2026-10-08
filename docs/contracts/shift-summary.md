# Contract — end-of-shift summary (A9)

**Status:** **APPROVED by Michael 2026-10-08** (design rev 3.5 §11 items 21–23) — day-1 contract freeze (design §10.2, lane schedule §2), written 2026-10-08 by lane 3. **Contract only** — no implementation. This document now changes only by design-doc revision; F1 (lane 2), D3-lite (lane 3) and the A9 implementation (lane 3, Oct 26–27) build against it.
**Source of truth for the rules:** `docs/design/phase1-safe-operating-system.md` §7.3 (what the summary shows), §2 (receipt numbers), §6 (happened/entered/late), §7.1 (exceptions, migration 061), §5.1 (reason codes carry `label_es`), §1.8 (receipts and the summary are endpoints first, screens second).
**Consumers:** the D3-lite dashboard page (phone), the F1 FL Assistant ("what did I enter today" / end of shift), and after cutover the MCP read tool `endOfShift` (M3). All three render this JSON; none computes anything from ledger rows.

The three points marked **[DECISION n — YES]** were approved by Michael on 2026-10-08 (PR #84); everything else follows the design text.

---

## 1. `GET /reports/shift-summary`

Read-only. Dashboard key and actor keys (both allowlists, A9 adds exactly this `GET` plus the `POST` in §3). Never writes; safe to call repeatedly.

### 1.1 Query parameters

| Param | Type | Default | Rules |
|---|---|---|---|
| `date` | `YYYY-MM-DD` plant day (America/New_York) | today (plant time) | future date → `422 DATE_IN_FUTURE`; any past day allowed (the page is also used to confirm yesterday) |
| `actor` | actor **id** or exact actor **name** (same rule as `GET /receipts?actor=`) | the authenticated actor | legacy master/dashboard key with no `actor` → the summary covers **all actors** and `actor` in the response is `null` **[DECISION 1 — YES]**; unknown → `422 ACTOR_UNKNOWN`; a floor actor asking for someone else → `403 ACTOR_NOT_ALLOWED` once A2 roles land (until A2: allowed) |
| `lang` | `en` \| `es` | `en` | affects only `label` / `message` / `summary` convenience fields; `*_en` and `*_es` are **always** both present so clients can switch without a refetch |

### 1.2 Response envelope

```json
{
  "date": "2026-10-07",
  "plant_timezone": "America/New_York",
  "generated_at": "2026-10-07T17:02:14-04:00",
  "actor": {"id": 7, "name": "Arturo", "role": "floor"},
  "title_en": "SHIFT SUMMARY — Tue Oct 7 — Arturo",
  "title_es": "RESUMEN DEL TURNO — mar 7 oct — Arturo",
  "receipts_today": 11,
  "summary_hash": "sha256-hex of the canonical JSON of `sections` (sorted keys, no whitespace)",
  "confirmation": {
    "status": "unconfirmed",
    "latest": null,
    "can_confirm": true,
    "reason_en": null, "reason_es": null
  },
  "sections": [ "...ten section objects, fixed order, see §2..." ]
}
```

- `receipts_today` = number of receipt rows across RECEIVED … VOIDED (each receipt counted once even when it appears again under LATE ENTRIES).
- `summary_hash` is echoed back on confirm (§3) so the server knows the person confirmed **what was shown**; it covers `sections` only (not `generated_at`).
- `confirmation.status`: `unconfirmed` \| `match` \| `discrepancy` — the **latest** `shift_confirmations` row for `(date, actor)`; `latest` is that row (§3.3 shape) or `null`. `can_confirm` is `false` with a reason only when the caller has no actor identity (legacy key, all-actors view) — confirmations are always by a person.
- `actor` is `null` for the all-actors view; `title_*` then ends with "— all / — todos".

## 2. Sections

`sections` is an **array in this fixed order**; a section with nothing to show is still present with `rows: []` and `count: 0` (the page prints the `empty_*` text, e.g. "VOIDED (none)"). Clients key on `key`, never on position or label.

| # | `key` | `label_en` | `label_es` | `empty_en` / `empty_es` | Row type |
|---|---|---|---|---|---|
| 1 | `received` | RECEIVED | RECIBIDO | none / ninguno | receipt row, action `receive` |
| 2 | `made` | MADE | PRODUCIDO | none / ninguno | receipt row, action `make` |
| 3 | `packed` | PACKED | EMPACADO | none / ninguno | receipt row, action `pack` |
| 4 | `shipped` | SHIPPED | ENVIADO | none / ninguno | receipt row, actions `ship_order` \| `ship_standalone` \| `ship_bulk` |
| 5 | `adjusted` | ADJUSTED | AJUSTADO | none / ninguno | receipt row, actions `adjust` \| `found` \| `rename_lot` \| `update_supplier_lot` |
| 6 | `voided` | VOIDED | ANULADO | none / ninguno | receipt row, action `void` (the `VD-` receipt; `refs.voided_receipt_number` names what it voided) |
| 7 | `not_entered_yet` | NOT ENTERED YET? | ¿FALTA REGISTRAR? | nothing expected / nada pendiente | expectation row (§2.3) |
| 8 | `lot_balances_touched` | LOT BALANCES TOUCHED TODAY | LOTES MOVIDOS HOY | none / ninguno | lot row (§2.4) |
| 9 | `open_exceptions` | OPEN EXCEPTIONS (yours) | EXCEPCIONES ABIERTAS (tuyas) | none / ninguna | exception row (§2.5) |
| 10 | `late_entries` | LATE ENTRIES TODAY | ENTRADAS TARDÍAS HOY | none / ninguna | receipt row (same object as above, repeated) |

Section object:

```json
{"key": "received", "label_en": "RECEIVED", "label_es": "RECIBIDO",
 "empty_en": "none", "empty_es": "ninguno", "count": 2, "rows": [ "..." ]}
```

### 2.1 Which receipts are "today's"

Exactly the `GET /receipts?date=<date>&actor=<actor>` set (design §7.3): every **committed** ticket whose `happened_at` plant day **or** `entered_at` (= `committed_at`) plant day is `date`, for the actor (or all actors). Each is listed under its action's section **once**, ordered by `happened_at` ascending; those with `late_entry: true` are listed **again** under LATE ENTRIES. A receipt that happened today but was entered on a later day shows on today's summary when that later summary is generated — it is marked `entered_on_later_day: true` so the person can see the day was back-filled after it was confirmed.

Pre-ticket (legacy) transactions never appear: no receipt number = not recorded (§2). Direct-route writes made during the overlap (before A10) therefore do not show here either — that is intended and is why the summary is the floor's check.

### 2.2 Receipt row (sections 1–6 and 10)

```json
{
  "receipt_number": "MK-261007-001",
  "action": "make",
  "status": "committed",
  "effective_status": "posted",
  "actor": {"id": 7, "name": "Arturo", "role": "floor"},
  "client_source": "fl_assistant",
  "happened_at": "2026-10-07T12:55:00-04:00",
  "entered_at":  "2026-10-07T12:58:41-04:00",
  "late_entry": false, "days_late": 0, "hours_late": 0, "entered_on_later_day": false,
  "product": {"id": 283, "name": "Batch SS Classic Granola #9 (Kosher Ignition)", "odoo_code": "90025", "type": "batch"},
  "lot": {"id": 9321, "lot_code": "26-10-07-CLS9-001", "supplier_lot_code": null, "identity_status": "identified"},
  "quantity": {"lb": 323, "cases": null, "unit": "lb", "display_en": "323 lb", "display_es": "323 lb"},
  "line_en": "MK-261007-001  SS Classic #9 Batch 90025  323 lb  lot 26-10-07-CLS9-001",
  "line_es": "MK-261007-001  SS Classic #9 Lote de producción 90025  323 lb  lote 26-10-07-CLS9-001",
  "detail": { "...per-action object, §2.2.1..." },
  "flags": [
    {"code": "SHORT", "severity": "warn",
     "message_en": "SHORT 12 lb oats lot 26-09-30-OATS-002 — resolve by Oct 9",
     "message_es": "FALTAN 12 lb avena lote 26-09-30-OATS-002 — resolver antes del 9 oct",
     "due_at": "2026-10-09T23:59:59-04:00",
     "refs": {"exception_id": 41, "shortage_flag_id": 12, "lot_id": 9123, "short_lb": 12}}
  ],
  "transaction_ids": [2451],
  "transactions": [{"id": 2451, "type": "make", "effective_status": "posted"}],
  "exceptions_opened": [41]
}
```

- `status` is the ticket status (always `committed` here); `effective_status` is the ledger's current state of the posted transaction(s): `posted` or `voided` (a voided receipt stays in its section with flag `VOIDED` and its `VD-` receipt under VOIDED).
- `quantity.lb` is always present (ledger truth); `cases` only for pack/ship/receive-by-case; `display_*` is the human form the page prints ("120 cases (900 lb)").
- `line_en` / `line_es` is the one-line rendering from §7.3's mock for clients that do not want to compose it (F1 chat, SMS-style). Fields, not lines, are the contract; the line is a convenience and may be truncated by the server.
- `flags[]` is the only place warnings live. Codes (closed list; clients render unknown codes generically):

| `code` | Section(s) | Meaning / source | `refs` |
|---|---|---|---|
| `UNIDENTIFIED_LOT` | received | lot with `identity_status='unidentified'`; message carries "resolve by <date>" (R8, +7 d) | `lot_id`, `exception_id`, `due_at` |
| `SUPPLIER_MISSING` | received | receive without `supplier_id` (gate 10(e) report; A5) | `transaction_id` |
| `SHORT` | made, packed | consumption posted short (R3); "resolve by" = `shortage_flags.due_at` | `shortage_flag_id`, `exception_id`, `lot_id`, `short_lb` |
| `LATE_ENTRY` | any, late_entries | happened ≥ 48 h before entered (§6.3) — an `exceptions(LATE_ENTRY)` was opened; < 48 h is `late_entry: true` without this flag | `exception_id`, `hours_late` |
| `LARGE_CORRECTION` | adjusted | > 500 lb or > 10 % of the lot (R2); photo required | `exception_id`, `delta_lb`, `pct` |
| `UNKNOWN_REASON` | adjusted | `reason_code='unknown'` (always highlighted, R2) | `transaction_id` |
| `PHOTO_PENDING` | shipped | `shipments.proof_status='photo_pending'` (R6) | `shipment_id`, `exception_id` |
| `NOT_SAME_DAY` | shipped | ship `happened_at` not on the plant day of entry (R6 blocker acknowledged by owner) | `shipment_id` |
| `INVOICE_PENDING` | packed, shipped | Sunshine-owned pack / bulk dispatch awaiting invoice (R7, A8) | `invoice_trigger_id` |
| `POSSIBLE_DUPLICATE_ACKED` | any | the committer acknowledged a possible-duplicate warning; names the earlier receipt | `receipt_number` |
| `VOIDED` | any | the receipt's transaction(s) are voided; names the `VD-` receipt | `voided_by_receipt_number` |
| `EXTRA_KOSHER` | made, packed, received | A12 follow-up (½ d): tier shown on the line | `kosher_tier` |

#### 2.2.1 `detail` per action

| action | `detail` |
|---|---|
| `receive` | `{supplier: {id, name} \| null, supplier_lot_code, cases, case_size_lb, bol_reference, expected_receipt: {id, reference_number} \| null}` |
| `make` | `{batch_product, output_lb, ingredients: [{product: {id, name}, lot: {id, lot_code}, quantity_lb, confirmation: {method: 'last4'\|'full_code'\|'scan'\|'pallet', value} \| null, short_lb: 0}], substitutions: [{ingredient_product, substitute_product, lot, reason_code: {code, label_en, label_es}}], production_warning}` — `ingredients[].confirmation` renders as "(last-4 typed)" |
| `pack` | `{source_product, source_lot: {id, lot_code}, target_product, cases, lb, ownership: 'cns'\|'sunshine', invoice_status: null\|'pending'\|'triggered'}` |
| `ship_order` / `ship_standalone` / `ship_bulk` | `{sales_order: {id, so_number, customer_name} \| null, customer: {id, name}, lines: [{product, cases, lb, lot}], bol_reference, proof_status: 'complete'\|'photo_pending', bins: [...] (bulk only)}` |
| `adjust` / `found` | `{lot, delta_lb (signed), lot_on_hand_before, lot_on_hand_after, reason_code: {code, label_en, label_es, note_required}, note, photo_attached: bool}` — `reason_code.*` comes from `correction_reasons` (061); legacy history has none and is never listed here (no receipt) |
| `rename_lot` / `update_supplier_lot` | `{lot, field: 'lot_code'\|'supplier_lot_code', old, new, reason_code: {...}}` |
| `void` | `{voided_receipt_number, voided_action, reason_code: {...}, note}` |

### 2.3 NOT ENTERED YET? rows

What FL expected today and has not seen a receipt for (design §7.3): open expected receipts due today or earlier without a linked `RCV`, and production runs planned today (status `planned` / `in_progress`) without a linked `MK` / `PK`.

```json
{"kind": "expected_receipt",
 "ref": {"expected_receipt_id": 88, "reference_number": "ER-…", "production_run_id": null},
 "product": {"id": 41, "name": "Graham Cracker Crumbs – 50 LB", "odoo_code": "…"},
 "expected": {"quantity": 1000, "unit": "lb", "date": "2026-10-07", "supplier": {"id": 12, "name": "…"}},
 "status": "open",
 "line_en": "expected receipt ER-… (Graham crumbs, due today) — no RCV receipt",
 "line_es": "recepción esperada ER-… (Graham crumbs, vence hoy) — sin recibo RCV"}
```

`kind`: `expected_receipt` \| `production_run` (`ref.production_run_id`, `expected.run_type`, `expected.planned_qty`/`planned_unit`). Ordered: overdue first, then by expected date, then product name. The all-actors view and the single-actor view show the **same** rows (expectations are not per person).

### 2.4 LOT BALANCES TOUCHED TODAY rows

One row per lot that any of today's listed receipts posted a line against, ordered by product name then lot code.

```json
{"lot": {"id": 9123, "lot_code": "24-09-30-COCO-001", "product": {"id": 12, "name": "Coconut Flake Desiccated"}},
 "before_lb": 260, "after_lb": 220, "delta_lb": -40,
 "short": false, "negative": false,
 "receipts": ["MK-261007-001"],
 "line_en": "coconut lot 24-09-30-COCO-001 260 → 220 lb",
 "line_es": "coco lote 24-09-30-COCO-001 260 → 220 lb"}
```

- `after_lb` = `lot_on_hand()` **now** (posted-only, effective). `before_lb` = `after_lb` minus the net of today's listed receipts' posted lines on that lot (so a lot touched twice shows one row, start → end). Both are lb; cases are never shown here.
- `short: true` when an open `shortage_flags` row exists for the lot; `negative: true` when `after_lb < 0` (the line prints "(short)").

### 2.5 OPEN EXCEPTIONS (yours) rows

Open or escalated `exceptions` (061) whose `owner_actor_id` is the actor (all owners in the all-actors view), overdue first, then by `due_at`, then `opened_at`.

```json
{"exception_id": 41, "kind": "SHORTAGE", "severity": "warn", "status": "open",
 "opened_at": "2026-10-07T13:01:00-04:00", "due_at": "2026-10-09T23:59:59-04:00", "overdue": false,
 "owner": {"id": 7, "name": "Arturo"},
 "refs": {"receipt_number": "MK-261007-002", "lot_id": 9123, "product_id": 12, "transaction_id": 2452, "shortage_flag_id": 12},
 "summary_en": "SHORT 12 lb oats lot 26-09-30-OATS-002 (from MK-261007-002) — resolve by Oct 9",
 "summary_es": "FALTAN 12 lb avena lote 26-09-30-OATS-002 (de MK-261007-002) — resolver antes del 9 oct"}
```

The section also carries a roll-up for the header line of §7.3's mock: `"rollup_en": "2 shortages (1 due Oct 9), 1 unidentified lot (due Oct 14)"`, `"rollup_es": "…"` on the section object (next to `count`). `kind` values are the 061 enum; `resolution` is **not** part of this read — resolving goes through `POST /exceptions/{id}/resolve/prepare` (A3b), not through the summary.

### 2.6 LATE ENTRIES TODAY

Receipt rows (same object as §2.2) for every receipt **entered** on `date` whose `happened_at` plant day is earlier (`late_entry: true`), ordered by `hours_late` descending. Rows with ≥ 48 h also carry the `LATE_ENTRY` flag (§6.3). This section is the only duplication in the document; clients that already show `late_entry` inline may hide it.

---

## 3. `POST /reports/shift-summary/confirm`

"Does this match what happened on the floor?  [Confirm]  [Confirm with notes]  [Something is missing]". One call, three outcomes. Actor keys only (a confirmation is always by a person); dashboard key → `403 ACTOR_REQUIRED` until D2 sign-in (A11) gives the page an FL session. Both allowlists gain this one `POST`.

### 3.1 Request

```json
{"date": "2026-10-07",
 "outcome": "match",
 "notes": "",
 "summary_hash": "…the hash from the GET…",
 "receipts_seen": 11}
```

| Field | Rules |
|---|---|
| `date` | required; the plant day confirmed; future → `422 DATE_IN_FUTURE` |
| `outcome` | `match` (Confirm), `match` + non-empty `notes` (Confirm with notes), `discrepancy` (Something is missing). `discrepancy` with blank `notes` → `422 NOTES_REQUIRED` |
| `notes` | ≤ 2,000 chars; stored verbatim; language free |
| `summary_hash` | required; must equal the hash of a `GET` for the same `(date, actor)` made **now** → else `409 SHIFT_SUMMARY_STALE` with the fresh summary in `detail.summary` so the client re-renders and asks again **[DECISION 2 — YES]** |
| `receipts_seen` | required; the `receipts_today` the client displayed (defence in depth with the hash; stored) |

The actor is the authenticated actor — never from the body (same rule as tickets). No ticket is involved: a confirmation is not a ledger write and is never replayed; a second confirm for the same `(date, actor)` **appends** a new row (the latest is the day's status; the weekly view counts the latest) **[DECISION 3 — YES]**.

### 3.2 Side effects

- `shift_confirmations` row (§3.3) with `summary_snapshot` = the exact `sections` JSON confirmed (so a later "it said 11 receipts" dispute is answerable).
- `outcome = 'discrepancy'` → one `exceptions` row: `kind='SHIFT_DISCREPANCY'`, `severity='warn'`, `owner_actor_id` = the **owner** actor (Michael), `detail = {business_date, actor_id, notes, receipts_seen, summary_hash}`, `due_at = NULL`. Its id is returned. Resolution is `acknowledged` (owner) via A3b's resolve endpoint.
- Nothing else. Not confirming is not blocked — it is visible (§7.2 item 8).

### 3.3 Response (`201`)

```json
{"confirmation_id": 17, "business_date": "2026-10-07",
 "actor": {"id": 7, "name": "Arturo", "role": "floor"},
 "outcome": "discrepancy", "notes": "The afternoon oats delivery was never entered",
 "confirmed_at": "2026-10-07T17:05:02-04:00", "confirmed_late": false,
 "receipts_seen": 11, "summary_hash": "…",
 "exception": {"id": 52, "kind": "SHIFT_DISCREPANCY"} ,
 "message_en": "Shift recorded as: something is missing. Michael will see this on the weekly view.",
 "message_es": "Turno registrado como: falta algo. Michael lo verá en la vista semanal."}
```

`confirmed_late: true` when `confirmed_at` plant day ≠ `business_date` (confirming yesterday is allowed and visible). `exception` is `null` for `match`.

### 3.4 Table (A9's migration, numbered at merge time)

```sql
CREATE TABLE shift_confirmations (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  business_date date NOT NULL,
  actor_id integer NOT NULL REFERENCES actors(id),
  outcome text NOT NULL CHECK (outcome IN ('match','discrepancy')),
  notes text NOT NULL DEFAULT '' CHECK (outcome <> 'discrepancy' OR btrim(notes) <> ''),
  receipts_seen integer NOT NULL CHECK (receipts_seen >= 0),
  summary_hash text NOT NULL CHECK (length(summary_hash) = 64),
  summary_snapshot jsonb NOT NULL CHECK (jsonb_typeof(summary_snapshot) = 'array'),
  exception_id bigint REFERENCES exceptions(id),
  confirmed_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX ON shift_confirmations (business_date, actor_id, confirmed_at DESC);
-- append-only (ledger_block_append_only_change trigger, 039 pattern); RLS on, owner only.
```

---

## 4. Errors

All errors use the standard envelope `{"error_detail": {"code", "message", "message_es", ...}}`.

| HTTP | `code` | When |
|---|---|---|
| 422 | `DATE_IN_FUTURE` | `date` after today's plant day |
| 422 | `ACTOR_UNKNOWN` | `actor` matches no actor |
| 403 | `ACTOR_NOT_ALLOWED` | (A2) floor actor asked for another actor's summary |
| 403 | `ACTOR_REQUIRED` | confirm with a key that has no actor identity |
| 422 | `NOTES_REQUIRED` | `outcome='discrepancy'` without notes |
| 422 | `OUTCOME_INVALID` | outcome not in the enum |
| 409 | `SHIFT_SUMMARY_STALE` | `summary_hash` ≠ current; `detail.summary` carries the fresh document |

---

## 5. Worked example — the §7.3 mock as JSON (abridged)

```json
{"date":"2026-10-07","plant_timezone":"America/New_York","generated_at":"2026-10-07T17:02:14-04:00",
 "actor":{"id":7,"name":"Arturo","role":"floor"},
 "title_en":"SHIFT SUMMARY — Tue Oct 7 — Arturo","title_es":"RESUMEN DEL TURNO — mar 7 oct — Arturo",
 "receipts_today":11,"summary_hash":"…",
 "confirmation":{"status":"unconfirmed","latest":null,"can_confirm":true,"reason_en":null,"reason_es":null},
 "sections":[
  {"key":"received","label_en":"RECEIVED","label_es":"RECIBIDO","empty_en":"none","empty_es":"ninguno","count":2,"rows":[
    {"receipt_number":"RCV-261007-001","action":"receive","happened_at":"2026-10-07T08:41:00-04:00","entered_at":"2026-10-07T08:43:10-04:00","late_entry":false,
     "product":{"id":12,"name":"Coconut Flake Desiccated"},"lot":{"id":9123,"lot_code":"24-09-30-COCO-001","supplier_lot_code":"8812-A","identity_status":"identified"},
     "quantity":{"lb":1100,"cases":44,"unit":"cases","display_en":"1,100 lb","display_es":"1,100 lb"},
     "detail":{"supplier":{"id":3,"name":"Dutch Valley Foods"},"supplier_lot_code":"8812-A","cases":44,"case_size_lb":25,"bol_reference":"77120","expected_receipt":null},
     "flags":[],"line_en":"RCV-261007-001  Coconut Flake Desiccated  1,100 lb  lot 24-09-30-COCO-001 (supplier lot 8812-A)  08:41"},
    {"receipt_number":"RCV-261007-002","action":"receive","product":{"id":30,"name":"Oats Rolled"},"lot":{"id":9124,"lot_code":"26-10-07-QUAL-001","supplier_lot_code":null,"identity_status":"unidentified"},
     "quantity":{"lb":2000,"display_en":"2,000 lb","display_es":"2,000 lb"},
     "flags":[{"code":"UNIDENTIFIED_LOT","severity":"warn","message_en":"UNIDENTIFIED — resolve by Oct 14","message_es":"SIN IDENTIFICAR — resolver antes del 14 oct","due_at":"2026-10-14T23:59:59-04:00","refs":{"lot_id":9124,"exception_id":40}}]}]},
  {"key":"made","label_en":"MADE","label_es":"PRODUCIDO","count":2,"rows":[
    {"receipt_number":"MK-261007-001","action":"make","product":{"id":283,"name":"Batch SS Classic Granola #9 (Kosher Ignition)","odoo_code":"90025"},"lot":{"id":9321,"lot_code":"26-10-07-CLS9-001"},"quantity":{"lb":323},
     "detail":{"ingredients":[{"product":{"id":12,"name":"Coconut Flake Desiccated"},"lot":{"id":9123,"lot_code":"24-09-30-COCO-001"},"quantity_lb":40,"confirmation":{"method":"last4","value":"-001"},"short_lb":0}]},"flags":[]},
    {"receipt_number":"MK-261007-002","action":"make","flags":[{"code":"SHORT","severity":"warn","message_en":"SHORT 12 lb oats lot 26-09-30-OATS-002 — resolve by Oct 9","message_es":"FALTAN 12 lb avena lote 26-09-30-OATS-002 — resolver antes del 9 oct","due_at":"2026-10-09T23:59:59-04:00","refs":{"shortage_flag_id":12,"exception_id":41,"lot_id":9100,"short_lb":12}}]}]},
  {"key":"packed","label_en":"PACKED","label_es":"EMPACADO","count":1,"rows":[
    {"receipt_number":"PK-261007-001","action":"pack","product":{"id":287,"name":"Granola SS Classic #9 10 LB"},"quantity":{"lb":900,"cases":120,"unit":"cases","display_en":"120 cases (900 lb)","display_es":"120 cajas (900 lb)"},
     "detail":{"source_lot":{"id":9321,"lot_code":"26-10-07-CLS9-001"},"ownership":"sunshine","invoice_status":"pending"},
     "flags":[{"code":"INVOICE_PENDING","severity":"info","message_en":"ownership: Sunshine → invoice pending","message_es":"propiedad: Sunshine → factura pendiente","refs":{}}]}]},
  {"key":"shipped","label_en":"SHIPPED","label_es":"ENVIADO","count":1,"rows":[
    {"receipt_number":"SHP-261007-001","action":"ship_order","quantity":{"cases":40,"lb":400,"display_en":"40 cases","display_es":"40 cajas"},
     "detail":{"sales_order":{"id":512,"so_number":"SO-261003-004","customer_name":"Setton Farms"},"bol_reference":"55821","proof_status":"complete"},"flags":[]}]},
  {"key":"adjusted","label_en":"ADJUSTED","label_es":"AJUSTADO","count":1,"rows":[
    {"receipt_number":"ADJ-261007-001","action":"adjust","product":{"id":5,"name":"Almonds"},"lot":{"id":8800,"lot_code":"26-09-12-ABAK-003"},"quantity":{"lb":-35,"display_en":"−35 lb","display_es":"−35 lb"},
     "detail":{"delta_lb":-35,"lot_on_hand_before":410,"lot_on_hand_after":375,"reason_code":{"code":"damage_disposal","label_en":"Damage/disposal","label_es":"Daño o desecho","note_required":false},"note":null,"photo_attached":false},"flags":[]}]},
  {"key":"voided","label_en":"VOIDED","label_es":"ANULADO","empty_en":"none","empty_es":"ninguno","count":0,"rows":[]},
  {"key":"not_entered_yet","label_en":"NOT ENTERED YET?","label_es":"¿FALTA REGISTRAR?","count":2,"rows":[
    {"kind":"expected_receipt","ref":{"expected_receipt_id":88,"reference_number":"ER-…"},"product":{"id":41,"name":"Graham Cracker Crumbs – 50 LB"},"expected":{"quantity":1000,"unit":"lb","date":"2026-10-07"},"status":"open","line_en":"expected receipt ER-… (Graham crumbs, due today) — no RCV receipt","line_es":"recepción esperada ER-… (Graham crumbs, vence hoy) — sin recibo RCV"},
    {"kind":"production_run","ref":{"production_run_id":88},"product":{"id":136,"name":"Granola SS Original 12x10 OZ"},"expected":{"run_type":"pack","planned_qty":100,"planned_unit":"cases","date":"2026-10-07"},"status":"planned","line_en":"production run #88 SS Original pack, planned today — status planned, no PK receipt","line_es":"corrida #88 SS Original empaque, planeada hoy — estado planeada, sin recibo PK"}]},
  {"key":"lot_balances_touched","label_en":"LOT BALANCES TOUCHED TODAY","label_es":"LOTES MOVIDOS HOY","count":2,"rows":[
    {"lot":{"id":9123,"lot_code":"24-09-30-COCO-001","product":{"id":12,"name":"Coconut Flake Desiccated"}},"before_lb":260,"after_lb":220,"delta_lb":-40,"short":false,"negative":false,"receipts":["MK-261007-001"],"line_en":"coconut lot 24-09-30-COCO-001 260 → 220 lb","line_es":"coco lote 24-09-30-COCO-001 260 → 220 lb"},
    {"lot":{"id":9100,"lot_code":"26-09-30-OATS-002","product":{"id":30,"name":"Oats Rolled"}},"before_lb":0,"after_lb":-12,"delta_lb":-12,"short":true,"negative":true,"receipts":["MK-261007-002"],"line_en":"oats lot 26-09-30-OATS-002 0 → −12 lb (short)","line_es":"avena lote 26-09-30-OATS-002 0 → −12 lb (faltante)"}]},
  {"key":"open_exceptions","label_en":"OPEN EXCEPTIONS (yours)","label_es":"EXCEPCIONES ABIERTAS (tuyas)","count":3,
   "rollup_en":"2 shortages (1 due Oct 9), 1 unidentified lot (due Oct 14)","rollup_es":"2 faltantes (1 vence 9 oct), 1 lote sin identificar (vence 14 oct)","rows":["…§2.5 rows…"]},
  {"key":"late_entries","label_en":"LATE ENTRIES TODAY","label_es":"ENTRADAS TARDÍAS HOY","empty_en":"none","empty_es":"ninguna","count":0,"rows":[]}
 ]}
```

---

## 6. Rules the implementation must keep (acceptance checklist for Oct 27)

1. Sections always present, fixed order, keyed by `key`; `label_en`/`label_es` on every section; `message_en`/`message_es` on every flag; `line_en`/`line_es` on every row.
2. Receipt set == `GET /receipts?date&actor` (same filter, same `late_entry` rule); no legacy transactions, no direct-route writes.
3. Lot balances are `lot_on_hand()` (posted-only, effective) — the summary never sums raw lines.
4. Reason codes are read from `correction_reasons` (061) so the Spanish label is never hard-coded in a client.
5. `summary_hash` is deterministic for identical data; confirm refuses a stale hash with the fresh document.
6. A `discrepancy` always opens exactly one `SHIFT_DISCREPANCY` exception owned by the owner, and the weekly view (§7.2 item 8) can count days by latest outcome.
7. The all-actors view is read-only (`can_confirm: false`).
8. Response time target: < 1.5 s for a 30-receipt day on staging (the page is opened on a phone at the end of a shift).

## 7. Decisions — recorded 2026-10-08 (Michael, PR #84; design §11 items 21–23)

- **[DECISION 1 — YES]** Legacy-key callers (today's dashboard, GPTs) get the all-actors view with `actor: null`, read-only. (Rejected alternative: require `actor` and 422.)
- **[DECISION 2 — YES]** Confirm requires the `summary_hash` from the summary shown; a changed summary is a 409 and the page re-asks. (Rejected alternative: no hash.)
- **[DECISION 3 — YES]** Re-confirming the same day appends (latest wins, history kept) rather than 409. (Rejected alternative: one confirmation per day per actor.)
- Section labels in Spanish (RECIBIDO / PRODUCIDO / EMPACADO / ENVIADO / AJUSTADO / ANULADO / ¿FALTA REGISTRAR? / LOTES MOVIDOS HOY / EXCEPCIONES ABIERTAS (tuyas) / ENTRADAS TARDÍAS HOY) — Arturo to confirm wording during pilot week 1 (D5 list).
