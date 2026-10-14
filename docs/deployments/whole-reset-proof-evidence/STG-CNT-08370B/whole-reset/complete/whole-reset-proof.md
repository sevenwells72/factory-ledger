# Whole-reset proof

**FULL RESET SCOPE RECONCILED.** Every reset-scope product is approved by exactly one signed area group, every count observation sits inside that group's signed plan, every journal receipt matches FL, every posted reset is proved by its own group's journal, no physical_count post by any actor in the reset window falls outside a signed plan, no owner hold is still open, and the final whole-scope verification reconciles with zero unexplained differences.

Target **staging**, one read-only snapshot at 2026-10-14T17:37:20.419855+00:00. Reset scope 7 products (manifest expects 7); **staging rehearsal scope**, not the production catalog.

## Area groups

| Group | Areas | Preview SHA-256 | Products | Posts (receipts) | Journal |
| --- | --- | --- | --- | --- | --- |
| Dry storage | Dry storage | 878e5e72952919d9… | 2 | ADJ-261014-023, ADJ-261014-024 | apply-state-staging/878e5e72952919d98df802704fb0314e76e96b2d78b1552190061cccf9c48164/journal.jsonl |
| Warehouse, Cooler | Cooler, Warehouse | 9a788a745aa24597… | 2 | ADJ-261014-025, FND-261014-208 | apply-state-staging/9a788a745aa245978f7a006f933e1af2e21066a7ba9114ab64645f0dd0015386/journal.jsonl |
| Office | Office | b75be065f8c10b2e… | 2 | - | none (no-change group) |
| Zero-stock sign-off | Office | f51b4483ab62f466… | 1 | - | none (no-change group) |

## Multi-area lots (handled once)

| Product / lot | Areas | Reset by | Held in |
| --- | --- | --- | --- |
| 1000000224 / STG-CNT-08370B-EK-1 | Cooler, Warehouse | Warehouse, Cooler | Office, Zero-stock sign-off |

Zero-stock products (no counted or on-hand lot), each approved by one signed group: 2.
Count-post window checked from 2026-10-14T13:35:37-04:00 (all actors).

## Gaps

None.

Final whole-scope verification: `final/reset-verification-v3.md` (exit 0). Machine-readable: `whole-reset-proof.json`, `whole-reset-gaps.csv`.
