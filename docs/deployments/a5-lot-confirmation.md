# A5 — lot confirmation, substitutions, unidentified lots, suppliers

Builder: Codex. Reviewer: Claude Code. Draft; do not merge.
Base: origin/main 1f3f098 (design revision 3.7).

## Checkpoints

1. Lot confirmations and pallet-move evidence: pending.
2. Substitutions with reasons: pending.
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

To be recorded with each implementation checkpoint.

## Validation and staging evidence

Pending. No production access, deployment, or merge authorized.
