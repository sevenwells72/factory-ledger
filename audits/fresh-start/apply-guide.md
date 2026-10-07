# V3 apply safeguards - for the owner and office

**Plain-English summary:** The count can run while production continues. Review the movements and exact proposed changes, then apply as soon as ready. There is no waiting period or age deadline. Only `apply_reset.py` can write inventory, and nothing was applied during this preparation. All other current count/preview/verification tools are write-incapable.

The new v3 path preserves signed preview hashes, exact original inputs, personal actor identity, fresh ledger checks, product/lot validation, rounded quantities, the process lock, a durable journal, exact typed confirmation, no automatic write retries, and independent posted-event readback. Ingredients use the same safeguards as finished goods; units remain explicit. Packaging is counted for information and excluded from apply, along with billing items Pallets 102 and Pallet Charge 176. The reset covers 178 of the 210 floor-count products. Archived/inactive products, service identities and unknown lots are not silently made writable. The former 171/209 physical-count exclusions are superseded in v3, but inactive-write holds remain.

Pre-decision v3 previews must be regenerated and reapproved. A scope-policy marker and explicit packaging/billing exclusions are checked when loading the signed plan and against the current catalog before posting. Packaging counts cannot authorize adjustments, even when signed or showing zero change.

## Before a future live use

1. Resolve all holds in the selected products. Keep original count CSV, coverage CSV, move log, owner-review JSON and photos. Run a fresh v3 preview and sign its exact JSON filename and SHA-256 using `approval.md`. Every proposed nonzero adjustment must be approved. For separate groups, issue separate reviewed input copies and previews containing only those completed product groups; do not hand-edit a generated preview.
2. **Dry run on production**, only when authorized: run the executor without `--apply`. It validates the signed inputs and current FL state and looks up the actor using a read-only query; it makes no application API calls. Use the private personal key at `~/.config/factory-ledger/blubber_key` with permissions 600. Shared/master keys are refused. This task did not read the real personal key or execute the live dry run.
3. **First live use: one small active product, whole pounds** (or the approved native unit of an eligible product). Separately approve that group, dry run, then apply it. **Check the recorded actor and dashboard balances** against the signed plan. **Only then approve and apply the rest.** A verified group is not full-scope sign-off.
4. Count-day movement continues normally. During the short actual posting/readback interval, coordinate no changes to the **approved products only**. Production elsewhere may continue. The API has no atomic compare-and-set or server idempotency token: any new ledger/catalog activity in approved products stops further posting and requires a refreshed review. This is a brief execution safeguard, not a plant-wide count freeze.

```sh
python3 -B audits/fresh-start/apply_reset.py \
  audits/fresh-start/v3/approved/reset-preview-v3.json \
  --approval audits/fresh-start/v3/approved/approval.md \
  --count-csv audits/fresh-start/v3/count-inputs/count.csv
```

A later separately authorized live run adds `--apply` and requires an interactive terminal and the exact typed phrase. All per-lot payloads use the approved native quantity, exact lot, per-lot effective cutoff derived from row times, and Estimated note. The signed v3 plan explicitly uses intentional historical posting (`backfill=true`) to preserve those true times; the existing application's normal date-window rule is not an apply deadline. Freshness, evidence and movement review govern readiness.

## If interrupted or something changes

Preserve `apply-state/<preview SHA-256>/journal.jsonl`. The journal records durable intent before every write, a complete transaction baseline, and validated receipt metadata afterward. A timeout or interruption is never retried blindly. On rerun, only one exact posted event with the approved product/lot, quantity, reason, cutoff and named actor can prove success; wrong/duplicate/voided/corrected events stop the process. Never delete the journal to force a retry. Already completed writes are not rolled back when a later check fails.

The executor verifies original source hashes throughout, refuses drift/new or omitted lots, compares the recomputed live-count reconciliation to the approved figures, and verifies every post. Standalone verification also reads and checks the journal without importing the write executor. Follow `tool-guide.md` for the verification command. Require **FULL RESET SCOPE RECONCILED** plus owner review of all reset-scope physical/ownership exceptions before declaring the reset complete. Informational packaging counts are reported separately and are not a reset sign-off requirement.

Earlier historical work included an owner-approved direct database write to archive product 185; `archive-185-execution.txt` is retained unchanged. This v3 preparation made no production writes or API calls and did not commit, create branches or open PRs.
