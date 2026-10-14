# Whole-reset proof

**FULL RESET SCOPE RECONCILED.** Every reset-scope product is approved by exactly one signed area group, every count observation sits inside that group's signed plan, every journal receipt matches FL, every posted reset is proved by its own group's journal, no physical_count post by any actor in the reset window falls outside a signed plan, no owner hold is still open, and the final whole-scope verification reconciles with zero unexplained differences.

Target **staging**, one read-only snapshot at 2026-10-14T17:08:31.316389+00:00. Reset scope 6 products (manifest expects 6); **staging rehearsal scope**, not the production catalog.

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

None.

Final whole-scope verification: `final/reset-verification-v3.md` (exit 0). Machine-readable: `whole-reset-proof.json`, `whole-reset-gaps.csv`.
