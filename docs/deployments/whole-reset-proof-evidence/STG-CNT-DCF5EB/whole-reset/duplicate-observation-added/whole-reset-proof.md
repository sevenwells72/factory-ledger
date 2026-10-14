# Whole-reset proof

**WHOLE RESET NOT RECONCILED: 1 gap(s).** This is not a release sign-off. Each gap below names the group, product and lot; fix the cause and rerun. Do not edit signed files or concatenate journals.

Target **staging**, one read-only snapshot at 2026-10-14T17:08:35.416785+00:00. Reset scope 6 products (manifest expects 6); **staging rehearsal scope**, not the production catalog.

## Area groups

| Group | Areas | Preview SHA-256 | Products | Posts (receipts) | Journal |
| --- | --- | --- | --- | --- | --- |
| Dry storage | Dry storage | 64c3f9b28a970020… | 2 | ADJ-261014-013, ADJ-261014-014 | apply-state-staging/64c3f9b28a9700204d1a8b45ee6e906c62ac5a88994bcdfad46f309b15b7fd74/journal.jsonl |
| Warehouse, Cooler | Cooler, Warehouse | 5d361ddd4e4724bc… | 2 | ADJ-261014-015, FND-261014-206 | apply-state-staging/5d361ddd4e4724bc2118c2d5542c3d4057814b168f3918969b812b57a4049b56/journal.jsonl |
| Office | Office | 7fc3a400fdbd67f8… | 2 | - | none (no-change group) |

## Multi-area lots (handled once)

| Product / lot | Areas | Reset by | Held in |
| --- | --- | --- | --- |
| 1000000196 / STG-CNT-DCF5EB-EK-1 | Cooler, Warehouse | Warehouse, Cooler | Office |

Zero-stock products (no counted or on-hand lot), each approved by one signed group: 1.
Count-post window checked from 2026-10-14T13:06:53-04:00 (all actors).

## Gaps

| Kind | Group | Product | Lot | Detail |
| --- | --- | --- | --- | --- |
| MISSING_OBSERVATION | Dry storage | 1000000184 | STG-CNT-DCF5EB-OATS-1 | observation RG1D-1-dup was not in the plan signed for this product |

Final whole-scope verification: `final/reset-verification-v3.md` (exit 0). Machine-readable: `whole-reset-proof.json`, `whole-reset-gaps.csv`.
