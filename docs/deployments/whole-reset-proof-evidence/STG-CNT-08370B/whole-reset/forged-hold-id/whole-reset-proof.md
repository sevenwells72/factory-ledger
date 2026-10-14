# Whole-reset proof

**WHOLE RESET NOT RECONCILED: 8 gap(s).** This is not a release sign-off. Each gap below names the group, product and lot; fix the cause and rerun. Do not edit signed files or concatenate journals.

Target **staging**, one read-only snapshot at 2026-10-14T17:37:25.80018+00:00. Reset scope 7 products (manifest expects 7); **staging rehearsal scope**, not the production catalog.

## Area groups

| Group | Areas | Preview SHA-256 | Products | Posts (receipts) | Journal |
| --- | --- | --- | --- | --- | --- |
| Dry storage | Dry storage | 878e5e72952919d9… | 2 | - | NOT PROVED |
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
| HOLD_NOT_IN_FL | Dry storage |  |  | Lot 1000000528: held receipt seq 4 names exception #424242, which FL does not have; FL has #258 LARGE_CORRECTION resolved on ticket 1187. |
| FINAL_HOLD |  | 1000000212 |  | Negative expected current stock |
| FINAL_LOT_HOLD |  |  |  | lot 1000000527: prior opening without journal proof |
| FINAL_LOT_HOLD |  |  |  | lot 1000000528: prior opening without journal proof |
| UNEXPLAINED_DIFFERENCE |  | 1000000211 | STG-CNT-08370B-OATS-1 | expected 390 vs FL 395 lb (difference -5) |
| UNEXPLAINED_DIFFERENCE |  | 1000000212 | STG-CNT-08370B-SUGAR-1 | expected -160 vs FL 520 lb (difference -680) |
| POST_OUTSIDE_APPROVED_PLAN |  | 1000000211 | STG-CNT-08370B-OATS-1 | TX 1000000595 ADJ-261014-023 -5 by Michael (STAGING STG-CNT-08370B) is not a proved post of any signed group |
| POST_OUTSIDE_APPROVED_PLAN |  | 1000000212 | STG-CNT-08370B-SUGAR-1 | TX 1000000596 ADJ-261014-024 -680 by Michael (STAGING STG-CNT-08370B) is not a proved post of any signed group |

Final whole-scope verification: `final/reset-verification-v3.md` (exit 2). Machine-readable: `whole-reset-proof.json`, `whole-reset-gaps.csv`.
