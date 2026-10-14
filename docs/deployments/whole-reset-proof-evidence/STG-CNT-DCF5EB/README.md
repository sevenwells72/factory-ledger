# STG-CNT-DCF5EB — raw whole-reset proof artifacts (hosted staging, 2026-10-14, PR #120 fix round 1)

Written by `scripts/count_rehearsal_staging.py --hosted-groups` (`export_evidence`). Staging only; synthetic owner/floor actors (deactivated afterwards), not Michael. Receipt: `docs/deployments/whole-reset-proof-hosted-staging-receipt.json`. Write-up: `audits/fresh-start/WHOLE-RESET-PROOF-2026-10-14.md`.

| Path | What |
|---|---|
| `group-*/approved/` | Signed `reset-preview-v3.json` + `approval.md`, **byte-identical** (`approval-without-zero-stock.md` = the negative case's narrower signature) |
| `group-*/inputs/` | Signed count/coverage/moves/owner-review copies, byte-identical (hashes match each preview's `sources`) |
| `apply-state-staging/<preview SHA>/journal.redacted.jsonl` | Executor journals. Only the `ticket` bearer token is replaced by `REDACTED-sha256:<16 hex>`; every other field, including each event's original `sha256` and `previous`, is kept. Header and receipt events re-hash exactly; intent events cannot (the token is part of their hash). |
| `forged/` | The forged-receipt journal (negative case), same redaction |
| `final-with-duplicate/` | Final count/review with the added duplicate-ticked oats row (negative case) |
| `manifest-<case>.json` | The manifests as run. Paths are relative to the run folder `audits/fresh-start/v3/rehearsal/STG-CNT-DCF5EB/`; journals at `../../apply-state-staging/<SHA>/journal.jsonl` |
| `snapshots/<case>.scoped.json` | The read-only staging snapshot each proof used, cut to the 6 rehearsal products, their lots/lines/tickets/holds and the two rehearsal actors (orders, catalog and schema dropped) |
| `whole-reset/<case>/` | Proof outputs (md/json/gaps csv) |
| `SHA256SUMS-originals.txt` | SHA-256 of every original file, unredacted journals included |

**Replay.** Each case was re-run from its scoped snapshot (`verify_reset.py --whole-reset … --snapshot snapshots/<case>.scoped.json`) and gave the same exit, verdict and gap list as the live run (receipt → `evidence.replay_from_scoped_snapshot`, all 9 true). The signed previews pin absolute input paths under `/Users/cns/dev/fl-o3-wt/audits/fresh-start/v3/rehearsal/STG-CNT-DCF5EB/`. To replay elsewhere, the files must be reachable at that path. A symlink from that path to your checkout's `audits/fresh-start` should work, since the tools resolve symlinks, but this was not tested. The journal-chain check needs the unredacted journals, which stay on this machine.
