# Resolution and void receipts (D1/D2 backend)

Branch base: `1a423f7`. Additive migration **080_resolution_receipts** must precede this code.
This is the backend contract for PR #107 item 7 and the coordination note for PR #108.

## Counted shortage and resolution responses

`POST /exceptions/{id}/resolve` keeps its existing exception-shaped response. Its
`receipt_number` still identifies the **source posting**, never the resolution.
These additive fields appear on exception list rows, detail, and direct resolve:

| Field | Meaning |
|---|---|
| `resolution_receipt_number` | Latest recorded XR decision/activity for this exception; null on older/unrecorded resolutions. |
| `resolution_posting_receipt_number` | Number of the fixing/evidence ticket referenced by the existing `resolution_ticket_id`; ADJ for a nonzero counted correction. |
| `allowed_actions` | Ordered strings permitted for the signed-in caller, as detailed below. |

A nonzero `counted` correction inserts its transaction with an ADJ receipt and
ticket ID. `detail.resolution.receipt_number`, `.ticket_id`, and `.transaction_id`
identify that correction. The existing `resolution_ticket_id` on both the
exception and shortage flag points to the ADJ ticket. The stock correction,
exception/flag close, actor audit, ADJ, XR and receipt counters commit or roll
back together. The ADJ carries correction-review and frozen kosher evidence.

A zero delta closes with XR only: `detail.resolution.transaction_id` and
`resolution_ticket_id` remain null; no zero-quantity adjustment is fabricated.
An accepted partial `missing_movement` claim records an XR but leaves the
exception open, with `evidence.remaining_lb` and `detail.last_claim.closes=false`.
A later decision replaces the exception's latest XR pointer; previous XR receipt
snapshots remain immutable. Direct resolve retains its existing closed-exception
409 behavior; clients needing safe replay should use the ticket flow.

## Prepare and commit

- `POST /exceptions/{id}/resolve/prepare` accepts the existing resolve body plus
  optional `client_source` (`dashboard` by default). Notes are still required for
  resolve, including write-off; `counted_lb`, evidence receipt or photo rules are
  unchanged. Its standard A1 draft includes source receipt, resolution kind, note,
  and counted book/delta/review facts. Preparing does not post stock or issue a number.
- `POST /void/{transaction_id}/prepare` accepts `reason` and optional
  `client_source`. The reason must not be blank.
- Both commit at `POST /tickets/{ticket}/commit` with the usual `payload_hash`
  and acknowledgement envelope. Actor binding, active-role recheck, expiry,
  signed payload and current state checks run again. A changed count balance or
  target correction state returns 409 `TICKET_STALE`; prepare again. Evidence
  belongs in the resolution prepare body, not extra commit fields.

An XR ticket commit returns:

```json
{
  "success": true,
  "receipt_number": "XR-261013-001",
  "ticket_id": 123,
  "replayed": false,
  "source_receipt_number": "MK-261013-012",
  "adjustment_receipt_number": "ADJ-261013-027",
  "exception": {
    "receipt_number": "MK-261013-012",
    "resolution_receipt_number": "XR-261013-001",
    "resolution_posting_receipt_number": "ADJ-261013-027",
    "status": "resolved",
    "allowed_actions": []
  }
}
```

The ticket ID above is illustrative. `adjustment_receipt_number` is null without a
counted ADJ. Recommitting the same ticket returns its saved response with
`replayed=true`, without another posting or number. `GET /receipts/{number}` and
`GET /receipts?action=resolve_exception&date=YYYY-MM-DD` resolve/list XR through
the existing receipt reader. Its `response` preserves the decision snapshot;
XR has no fabricated inventory transaction. Counted ADJ appears once in the shift
summary's adjusted section; XR appears under `other_receipts`.

## Approval and rejection

Owner approve accepts omitted, null, empty and whitespace-only `note`, including
extra-kosher holds. Existing approval evidence and preparer/owner checks remain.
The approval response keeps the stock posting's `receipt_number` and adds
`resolution_receipt_number` (XR). The original stock ticket stores this same link
before becoming immutable; approval and stock-ticket retries preserve it. Photo
release of a held correction also records XR atomically.

Reject still requires `resolution_kind="declined"` and a nonblank note. Its
existing rejection response adds `resolution_receipt_number` (XR); rejection
retry returns the saved rejection response with `replayed=true`. No stock posts.
Write-off and waiver remain owner-only, and write-off still requires a note.

## Allowed actions for PR #107

`allowed_actions` is an array of strings, evaluated for the authenticated actor
using the A2 policy also enforced on writes. Do not derive buttons from a copied
frontend resolution-kind map. These are authorization choices; the server still
checks evidence, current state, hold expiry and all other posting rules.

| Exception/caller | Values |
|---|---|
| Open SHORTAGE / NEGATIVE_BALANCE, owner or floor | `counted`, `missing_movement`, `voided` |
| Open UNIDENTIFIED_LOT, owner | `identified`, `written_off` |
| Open UNIDENTIFIED_LOT, floor | `identified` |
| Open held LARGE_CORRECTION / KOSHER_ATTESTATION, owner | `approve`, `reject` |
| Open LATE_ENTRY / PRE_MAKE_ADJUST, owner | `acknowledged` |
| Open BOL_PHOTO_MISSING / SHIPMENT_PROOF_MISSING, owner | `photo_attached`, `waived` |
| Office caller, closed exception, floor on owner-only kind | `[]` |

For other kinds, the existing default choices are `resolved`, `waived` for owner,
and `resolved` for floor. Shared keys cannot access exception routes. Session
sign-in resolves to the same actor/role rules.

Map `approve` to `POST /exceptions/{id}/approve`, `reject` to
`POST /exceptions/{id}/reject` with `resolution_kind="declined"`. Other strings
are `resolution_kind` values for resolve/prepare (or the compatibility direct
resolve endpoint). Closed/waived exceptions always return `[]`.

## Void responses and receipt display

Both void prepare/commit and the compatibility `POST /void/{transaction_id}`
produce `VD-YYMMDD-NNN`. The direct response preserves reversal compatibility
fields and adds `receipt_number`, `ticket_id`, `source_receipt_number`, `replayed`.
The generic transaction-correction void path also records VD. A2 permits floor
voids only for their own same-plant-day posting; owner/master retain their scope.
A VD's `result_ref.correction_id` matches the append-only correction ID, with
`action="void"` and `status="committed"`, as expected by PR #107's projection.

The original receipt remains immutable and keeps `status="committed"`.
`GET /receipts/{original}` adds live `effective_status` (`posted`, `voided`, or
`partially_voided`; metadata receipts use their ticket status) and
`void_receipt_number`. Each entry in `transactions` has its own
`void_receipt_number`. For multiple transactions, the top-level number is the
first available VD; use the per-transaction links for complete detail. A later
restore clears the live backlink, while VD itself stays retrievable. Historical
voids without numbered tickets remain null. A subsequent metadata amendment does
not hide an effective void's VD link. Dashboard color/recorded state must use
current effective status, not committed ticket status alone.

## Case-unit corrections and PR #108 coordination

Adjust `delta_lb` and its book balance already hold pounds for weighted case
products. A3b now treats these as pounds when the product has a real
`case_size_lb` or `default_case_weight_lb`, rather than multiplying pounds by
case weight a second time. A -40 lb correction on a 15 lb case stays -40 lb.
Unweighted count-based supplies keep the existing percentage-only behavior;
explicit found-stock unit conversion is unchanged. >10% highlights; >500 lb
requires a photo or owner hold. Exact boundary values do not exceed a threshold.

Likely conflicts with PR #108: `exceptions_enforcement.py` counted evidence and
receipt helpers, `write_tickets.py` commit hooks, both changelogs, and FOLLOWUPS.
This branch does not change the adjust request model, ordinary payload hashes,
`refuse_cover_up`, or `ticket_actions.py` adjust posting. Preserve #108's exact
`resolves_shortage_ids` set, physical-count-only rule, shortage-before-lot lock
order and fixing ADJ `resolution_ticket_id` semantics when combining branches.
Do not replace that evidence pointer with XR. An XR records a resolution decision;
#108's ADJ remains the stock evidence.

Update #108's `audits/fresh-start/fresh_start_v3.py:a3b_lb_equivalent`,
`test_R6b_hold_prediction_mirrors_a3b_units`, and `READINESS-2026-10-13.md` predicted
case-unit hold after this lands: -40 lb is no longer predicted as 600 lb/held.
No count-tool files or its worktree were edited here.

## Deployment and validation

Apply only `migrations/080_resolution_receipts.sql` as the app/table owner in a
bounded transaction before deploying code. It preserves existing allowed ticket
actions, adds XR/VD actions, a VD lookup index and nullable exception XR FK; no
historical backfill or startup application. Existing RLS is unchanged. Application
rollback leaves additive schema and issued receipt evidence intact; do not drop
the new column/action support once receipts exist. Production application and merge
remain separate operator actions.

Staging validation uses `scripts/check_resolution_receipts_staging.py` through
local branch HTTP handlers against the guarded staging DB, without app startup,
production access or hosted deployment. Credentials are kept in memory and
synthetic actors are deactivated. See `docs/deployments/resolution-receipts-staging.json`
for persisted ADJ/XR/VD evidence. The migration is applied only to staging and
throwaway local databases in this task. Baseline and final test counts are in
`docs/deployments/resolution-receipts-tests.json`.
