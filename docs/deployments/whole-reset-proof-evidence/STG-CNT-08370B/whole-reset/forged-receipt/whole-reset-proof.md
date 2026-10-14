# Whole-reset proof

**WHOLE RESET NOT RECONCILED: 3 gap(s).** This is not a release sign-off. Each gap below names the group, product and lot; fix the cause and rerun. Do not edit signed files or concatenate journals.

Target **staging**, one read-only snapshot at 2026-10-14T17:37:24.770372+00:00. Reset scope 7 products (manifest expects 7); **staging rehearsal scope**, not the production catalog.

## Area groups

| Group | Areas | Preview SHA-256 | Products | Posts (receipts) | Journal |
| --- | --- | --- | --- | --- | --- |
| Dry storage | Dry storage | 878e5e72952919d9… | 2 | ADJ-261014-023, ADJ-261014-024 | v3/rehearsal/STG-CNT-08370B/forged/group-1-journal-forged-receipt.jsonl |
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
| INVALID_RECEIPT | Dry storage | 1000000211 |  | journal receipt seq 8 for intent 7 (commit, ticket 1186): unexpected field(s) receipt_number on a 'refused' receipt |
| INVALID_RECEIPT | Dry storage | 1000000211 |  | journal receipt seq 8 for intent 7 (commit, ticket 1186): a 'refused' receipt names transaction 999999999 |
| RECEIPT_CONTRADICTS_FL | Dry storage | 1000000211 |  | journal receipt seq 8 for intent 7 (commit, ticket 1186): journal says the commit was refused but FL posted that ticket (ADJ-261014-023) |

Final whole-scope verification: `final/reset-verification-v3.md` (exit 0). Machine-readable: `whole-reset-proof.json`, `whole-reset-gaps.csv`.
