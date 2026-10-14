# Whole-reset proof

**WHOLE RESET NOT RECONCILED: 3 gap(s).** This is not a release sign-off. Each gap below names the group, product and lot; fix the cause and rerun. Do not edit signed files or concatenate journals.

Target **staging**, one read-only snapshot at 2026-10-14T17:08:34.561027+00:00. Reset scope 6 products (manifest expects 6); **staging rehearsal scope**, not the production catalog.

## Area groups

| Group | Areas | Preview SHA-256 | Products | Posts (receipts) | Journal |
| --- | --- | --- | --- | --- | --- |
| Dry storage | Dry storage | 64c3f9b28a970020… | 2 | ADJ-261014-013, ADJ-261014-014 | v3/rehearsal/STG-CNT-DCF5EB/forged/group-1-journal-forged-receipt.jsonl |
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
| INVALID_RECEIPT | Dry storage | 1000000184 |  | journal receipt seq 8 for intent 7 (commit, ticket 886): unexpected field(s) receipt_number on an refused receipt |
| INVALID_RECEIPT | Dry storage | 1000000184 |  | journal receipt seq 8 for intent 7 (commit, ticket 886): an refused receipt names transaction 999999999 |
| RECEIPT_CONTRADICTS_FL | Dry storage | 1000000184 |  | journal receipt seq 8 for intent 7 (commit, ticket 886): journal says the commit was refused but FL posted that ticket (ADJ-261014-013) |

Final whole-scope verification: `final/reset-verification-v3.md` (exit 0). Machine-readable: `whole-reset-proof.json`, `whole-reset-gaps.csv`.
