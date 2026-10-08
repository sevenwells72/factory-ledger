# A5 — lot confirmation, substitutions, unidentified lots, suppliers

Builder: Codex. Reviewer: Claude Code. Draft; do not merge.
Base: origin/main 1f3f098 (design revision 3.7).

## Checkpoints

1. Lot confirmations and pallet-move evidence: complete; 175 targeted tests passed.
2. Substitutions with reasons: complete; 122 confirmation/substitution/A1 tests passed.
3. Unidentified lots and seven-business-day deadline: pending.
4. Real supplier identity on receipts: pending.

Commit and push each finished part. Recheck `feat/roles` before shared edits.
A5 owns confirmation validators; A3b owns insufficient-stock behavior.
Direct legacy routes remain unchanged. Migrations are explicit and staging-only.

## Integration baseline

The fetched main implements receive/make/pack/adjust/found tickets only.
Ship and supplier-lot correction tickets are absent. A5 integrates existing
consumption tickets and adds the specified pallet-move ticket; later ticket
adapters can use the same interface-neutral confirmation module.

## Shared hook points

- `ticket_actions.validate`: dispatch `move_lot`; call A5 confirmation validator after stock validation.
- `ticket_actions.post`: dispatch move evidence and save confirmations after core posting.
- `write_tickets`: typed request fields/move route; late-evidence merge before validation;
  recoverable 422 confirmation blockers; non-ledger move result refs/receipt reads.
- `main._make_commit_core`: optional validated `substitutions` argument and named
  formula mapping hook immediately after formula fetch. Legacy callers supply none.
- `ticket_actions.validate/post`: A5 substitution validation, specification and evidence.
- `choose_inputs`, `_post_prepared_inputs` and insufficient-stock branches: unchanged.

The original payload/hash stays immutable. Prepare returns `confirmed:false`
and `LOT_NOT_CONFIRMED` until evidence is supplied. Commit can add only
`lot_confirmations`; it cannot replace lot selections or quantities. Missing or
invalid evidence returns 422 and leaves the ticket prepared for a retry.
`last4` requires exactly four matching characters and a unique suffix among
positive-stock, non-merged lots of that product; `full_code`/`scan` match the full
code. `pallet` uses the full lot code plus the latest move being to production,
within 24 hours. A later storage/staging move invalidates that evidence.

Pallet moves use `/lots/{lot_id}/move/prepare` with `to_location`, `method`
(`last4`, `full_code`, `scan`) and `value`, followed by normal ticket commit.
They create a `LOT` receipt and immutable `lot_moves` row, without ledger stock
movement. Named actors retain attribution; dashboard scope is unchanged.

## Validation and staging evidence

Pending. No production access, deployment, or merge authorized.

## Substitution contract

Make accepts `substitutions: [{ingredient_product_id, substitute_product_id,
lot_id, reason_code, note}]`. The substitute takes the original pounds; the
master formula is unchanged. It must be an active stock ingredient/batch, with
a matching lot, and needs its own confirmation. Excluded/duplicate/self/output
substitutions are blocked. Multiple requirements for one replacement aggregate
and use one selected lot. Manual `excluded_ingredients` require a top-level
`reason_code` (optional `note`); automatic recipe exclusions retain their policy.
Reasons here are operator-supplied nonblank codes; the §5.1 fixed correction
catalog explicitly applies to corrections, not make substitutions.
`transaction_substitutions` stores original/replacement/lot/reason/note/actor
atomically, including null replacement for a manual exclusion. Receipt JSON
exposes the evidence; ledger and ingredient trace consume the actual substitute.
