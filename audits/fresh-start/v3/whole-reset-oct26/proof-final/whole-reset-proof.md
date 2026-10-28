# Whole-reset proof

**FULL RESET SCOPE RECONCILED.** Every reset-scope product is approved by exactly one signed area group, every count observation sits inside that group's signed plan, every journal receipt matches FL, every posted reset is proved by its own group's journal, no physical_count post by any actor in the reset window falls outside a signed plan, no owner hold is still open, and the final whole-scope verification reconciles with zero unexplained differences.

Target **production**, one read-only snapshot at 2026-10-27T17:28:44.782167+00:00. Reset scope 178 products (manifest expects 178).

## Area groups

| Group | Areas | Preview SHA-256 | Products | Posts (receipts) | Journal |
| --- | --- | --- | --- | --- | --- |
| G01 first 70 | 01 Ingredients | 36b5ca0ecc0de989… | 1 | ADJ-261022-001 | apply-state/36b5ca0ecc0de989aebb2234ea4daf201af4b205f6e4fbb83ea134843f719f82/journal.jsonl |
| G01 rest | 01 Ingredients | 21570e83c294c199… | 8 | ADJ-261022-002, ADJ-261022-003, ADJ-261022-004, ADJ-261022-005, ADJ-261022-006, ADJ-261022-007, ADJ-261022-008, ADJ-261022-009, ADJ-261022-010, ADJ-261022-011, ADJ-261022-012 | apply-state/21570e83c294c19940c6af4213c63c84d448ab653b428a7f803bc6086d318e2b/journal.jsonl |
| G02 coconut raw | 02 Coconut raw materials | ff7ca42a935149ab… | 4 | ADJ-261022-013, ADJ-261022-014, ADJ-261022-015, ADJ-261022-016, ADJ-261022-017, ADJ-261022-018, ADJ-261022-019, ADJ-261022-020, ADJ-261022-021 | apply-state/ff7ca42a935149ab9cbcdd9188dd7dbce5726bbb20a0d7cfee9a844e959af100/journal.jsonl |
| Final group 1 | (all) | d834a086f4e16aad… | 150 | ADJ-261022-022, ADJ-261022-023, ADJ-261022-024, ADJ-261022-025, ADJ-261022-026, ADJ-261022-027, ADJ-261022-028, ADJ-261022-029, ADJ-261022-030, ADJ-261022-031, ADJ-261022-032, ADJ-261022-033, ADJ-261022-034, ADJ-261022-035, ADJ-261022-036, ADJ-261022-037, ADJ-261022-038, ADJ-261022-039, ADJ-261022-040, ADJ-261022-041, ADJ-261022-042, ADJ-261022-043, ADJ-261022-044, ADJ-261022-045, ADJ-261022-046, ADJ-261022-047, ADJ-261022-048, ADJ-261022-049, ADJ-261022-050, ADJ-261022-051, ADJ-261022-052, ADJ-261022-053, ADJ-261022-054, ADJ-261022-055, ADJ-261022-056, ADJ-261022-057, ADJ-261022-058, ADJ-261022-059, ADJ-261022-060, ADJ-261022-061, ADJ-261022-062, ADJ-261022-063, ADJ-261022-064, ADJ-261022-065, ADJ-261022-066, ADJ-261022-067, ADJ-261022-068, ADJ-261022-069, ADJ-261022-070, ADJ-261022-071, ADJ-261022-072, ADJ-261022-073, ADJ-261022-074, ADJ-261022-075, ADJ-261022-076, ADJ-261022-077, ADJ-261022-078, ADJ-261022-079, ADJ-261022-080, ADJ-261022-081, ADJ-261022-082, ADJ-261022-083, ADJ-261022-084, ADJ-261022-085, ADJ-261022-086, ADJ-261022-087, ADJ-261022-088, ADJ-261022-089, ADJ-261022-090, ADJ-261022-091, ADJ-261022-092, ADJ-261022-093, ADJ-261022-094, ADJ-261022-095, ADJ-261022-096, ADJ-261022-097, ADJ-261022-098, ADJ-261022-099, ADJ-261022-100, ADJ-261022-101, ADJ-261022-102, ADJ-261022-103, ADJ-261022-104, ADJ-261022-105, ADJ-261022-106, ADJ-261022-107, ADJ-261022-108, ADJ-261022-109, ADJ-261022-110, ADJ-261022-111, ADJ-261022-112, ADJ-261022-113, ADJ-261022-114, ADJ-261022-115, ADJ-261022-116, ADJ-261022-117, ADJ-261022-118 | apply-state/d834a086f4e16aad94014f17faac6370d4c9d590b245f36982299614a05d3ca4/journal.jsonl |
| Zero-stock 285/286 | (all) | 934c739710a9e9ed… | 2 | - | none (no-change group) |
| Small group B | (all) | 2d97861d5bdc5d31… | 11 | ADJ-261027-001, ADJ-261027-002, ADJ-261027-003, ADJ-261027-004, ADJ-261027-005, ADJ-261027-006, ADJ-261027-007, ADJ-261027-008, ADJ-261027-009, ADJ-261027-010 | apply-state/2d97861d5bdc5d31c2440f112053579b1d896366afa299a58fa2d9270ac84512/journal.jsonl |
| Provisional #9 | (all) | 6bc20f9228da5d13… | 2 | - | none (no-change group) |

## Multi-area lots (handled once)

None.

Zero-stock products (no counted or on-hand lot), each approved by one signed group: 42.
Count-post window checked from 2026-10-22T09:00:00-04:00 (all actors).

## Gaps

None.

Final whole-scope verification: `final/reset-verification-v3.md` (exit 0). Machine-readable: `whole-reset-proof.json`, `whole-reset-gaps.csv`.
