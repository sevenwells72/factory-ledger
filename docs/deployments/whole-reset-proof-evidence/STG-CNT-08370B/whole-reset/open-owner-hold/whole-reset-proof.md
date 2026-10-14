# Whole-reset proof

**WHOLE RESET NOT RECONCILED: 5 gap(s).** This is not a release sign-off. Each gap below names the group, product and lot; fix the cause and rerun. Do not edit signed files or concatenate journals.

Target **staging**, one read-only snapshot at 2026-10-14T17:37:37.617043+00:00. Reset scope 7 products (manifest expects 7); **staging rehearsal scope**, not the production catalog.

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

| Kind | Group | Product | Lot | Detail |
| --- | --- | --- | --- | --- |
| FINAL_HOLD |  | 1000000211 |  | Unclassified movement / allocation: TX1000000599 [a11fb43a7b6e8c2290332eefae87b0965a2ea72051630167ce39f22f06c0f8d7] |
| FINAL_HOLD |  | 1000000215 |  | Unclassified movement / allocation: TX1000000600 [2d701ba7106504ab0e18805b93f27d4f82559667d766c9cac4467959661fca35] |
| POST_OUTSIDE_APPROVED_PLAN |  | 1000000211 | STG-CNT-08370B-OATS-1 | TX 1000000599 ADJ-261014-026 1 by Michael (STAGING STG-CNT-08370B) is not a proved post of any signed group |
| POST_OUTSIDE_APPROVED_PLAN |  | 1000000215 | STG-CNT-08370B-GRANOLA-1 | TX 1000000600 ADJ-261014-027 2 by Arturo (STAGING STG-CNT-08370B) is not a proved post of any signed group |
| OPEN_HOLD |  | 1000000212 |  | LARGE_CORRECTION hold #260 on ticket 1194 is open: it can still post; resolve it (reject, or approve and prove it) before release |

Final whole-scope verification: `final/reset-verification-v3.md` (exit 2). Machine-readable: `whole-reset-proof.json`, `whole-reset-gaps.csv`.
