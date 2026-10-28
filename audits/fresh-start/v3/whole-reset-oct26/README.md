# Whole-reset readiness check (2026-10-26, read-only, nothing posted)

`whole-reset-manifest.DRAFT.json` lists the four applied groups (G01 first 70, G01 rest, G02, final group 1)
with their signed previews, approvals and journals. Final inputs = ../count-inputs/oct26-small-group.
The small group must be added as a fifth entry once signed (as `no_change` if it posts nothing).

## Dry proof (current tool) — dry-proof/ (proof SHA 58d15c62…)
46 gaps, all expected except 5:
- 15 MISSING_PRODUCT, 12 COVERAGE_INCOMPLETE, 13 FINAL_HOLD, 1 UNEXPLAINED_DIFFERENCE (171 −5 lb):
  the small-group products only.
- 5 MISSING_LOT: 126 OCT 23 2026, 148 BB101927, 149 BB101927, 157 OCT 22 2026, 157 OCT 23 2026.
  All five are lots created by ordinary production on Oct 23 (TX2850–2860), after every count.
  They can never be in a signed plan, so the proof could never pass while the plant runs.

## Prepared fix (NOT applied — needs owner OK) — post-count-lots.patch
`whole_reset_proof.py`: an unsigned lot is not a gap when its first FL movement was entered after its count
cutoff AND it is coverage-only, READY, with zero change (its whole balance is logged post-count movement).
A lot with any movement before its cutoff is still MISSING_LOT. New test
`test_lot_made_after_the_count_is_not_a_missing_lot` fails without the patch and passes with it;
test_whole_reset.py 31/31 with it. Tool reverted after the check; `git apply post-count-lots.patch` applies cleanly.
Dry proof with the patch — dry-proof-patched/ (SHA db53f8a8…): 41 gaps, only the small-group products.

## Oct 23 top-ups (read-only investigation)
| TX | Item | + lb | Finding |
| --- | --- | ---: | --- |
| 2851 | Sugar – Monk Fruit (45) | 7 | Reset left 55 lb (= count); #8 recipe needed 62 |
| 2852 | Chocolate Chips – Sugar Free (6) | 58 | Reset left 0 (count 0); #8 recipe needed 58 |
| 2853 | Flax – Ground (27) | 28 | Reset left 0 in that lot; #8 recipe needed 28 |
| 2855 | Batch #8 (119) | 27.5 | 2 batches = 700 lb; 97 cases packed = 727.5 lb (pack yield) |
| 2859 | Batch Coconut Sweetened Flake (126) | 40 | 6 batches = 2,160 lb; 88 cases = 2,200 lb (same pattern +65 lb on Oct 21) |
All posted as ordinary adjustments (reason "unknown"), not physical_count, so they do not break the proof;
they appear as post-count movements. First three: either missed in the count or recipe ≠ actual use (Arturo R8).
Last two: batch yield above recipe (Arturo R9; see branch fix/coconut-yield-multiplier).
No open exceptions (42 resolved). No negative lots.
