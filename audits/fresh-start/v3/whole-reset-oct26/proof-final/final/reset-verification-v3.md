# Live-count v3 verification

**Plain-English summary:** 1276 usable reset lot/location observations; 1275 ready lot comparisons; 0 product/identity groups, 0 lots and 0 sheets held. No stock was changed. FULL RESET SCOPE RECONCILED.

Reset scope: 178 products. Packaging and billing items are excluded (32 catalog products); see [the separate information-only report](packaging-information-only.md). Packaging coverage, unknown lots and quantities cannot generate reset adjustments or holds. This verdict concerns only the reset scope, not completion of the informational floor count.

**TIME SINCE COUNT: 0.05 to 124.48 hours** at snapshot 2026-10-27T17:28:44.782167+00:00.
No waiting period or age deadline. Every logged reset-scope movement since each cutoff is in `all-movements.csv`. New activity requires a refreshed review.

Counts use each row time, falling back to sheet end. Multiple locations cannot each be compared as if FL held separate location balances: FL only has the total lot balance. `row-cutoffs.csv` shows that whole-lot reference at each observed time. Classified movements are allocated once among locations to bring the combined count forward. The proposed adjustment is one per lot, dated at its latest count time, with all underlying row cutoffs retained.

Estimated counts are marked on the preview and in the exact adjustment reason. Unidentified reset-scope lots remain held with a seven-day follow-up; no new lots are created. Empty count rows are not zeros. Missing lot zeros require certified product coverage.

## Per-area reconciliation

| Sheet | Area | Start / end | Rows | Movement lines | Holds |
| --- | --- | --- | --- | --- | --- |
| A01 | 01 Ingredients | 2026-10-22T09:00:00-04:00 / 2026-10-22T11:18:00-04:00 | A01-2, A01-3, A01-4, A01-7, A01-9, A01-12-1, A01-12-2, A01-17, A01-18, A01-19, A01-21, A01-72-R6-1, A01-72-R6-2, A01-72-R6-3, A01-7-R6-1, A01-7-R6-2, A01-8-R6-1, A01-8-R6-2, A01-16-R6-1, A01-16-R6-2, A01-18-R6-1, A01-20-R6-1, A01-21-R6-1, A01-26-R6-1, A01-26-R6-2, A01-28-R6-1, A01-29-R6-1, A01-29-R6-2, A01-30-R6-1, A01-34-R6-1, A01-34-R6-2, A01-52-R6-1, A01-35-R6-1, A01-39-R6-1, A01-39-R6-2, A01-39-R6-3, A01-40-R6-1, A01-40-R6-2, A01-41-R6-1, A01-42-R6-1, A01-43-R6-1, A01-43-R6-2, A01-44-R6-1, A01-45-R6-1, A01-46-R6-1, A01-32-R6-1, A01-32-R6-2, A01-32-R6-3, A01-75-R6-1, A01-75-R6-2 | 8188, 8189, 8202, 8203, 8193, 8043, 8194, 8204, 8205, 8184, 8195, 8196 |  |
| A02 | 02 Coconut raw materials | 2026-10-22T09:00:00-04:00 / 2026-10-22T09:00:00-04:00 | A02-87, A02-90, A02-91, A02-92, A02-93, A02-94, A02-96, A02-97, A02-98, A02-99, A02-100, A02-101, A02-12-R6-1 | 8201, 8191 |  |
| A06 | 06 Finished goods - all customers | 2026-10-22T09:00:00-04:00 / 2026-10-22T13:26:00-04:00 | A06-196, A06-164-R5-1, A06-164-R5-2, A06-164-R5-3, A06-164-R5-4, A06-164-R5-5, A06-164-R5-6, A06-164-R5-7, A06-164-R5-8, A06-164-R5-9, A06-164-R5-10, A06-157-R5-1, A06-157-R5-2, A06-157-R5-3, A06-157-R5-4, A06-156-R5-1, A06-158-R5-1, A06-162-R5-1, A06-161-R5-1, A06-161-R5-2, A06-163-R5-1, A06-163-R5-2, A06-160-R5-1, A06-160-R5-2, A06-159-R5-1 | 8176, 8177, 8178, 8179 |  |
| A06B | 06 Finished goods - all customers | 2026-10-22T16:32:00-04:00 / 2026-10-22T16:41:00-04:00 | A06-137-1, A06-137-2, A06-137-3, A06-136-R5-1, A06-136-R5-2, A06-136-R5-3, A06-142-R5-1, A06-141-R5-1, A06-144-R5-1, A06-144-R5-2, A06-144-R5-3, A06B-145-R6-1, A06B-145-R6-2, A06B-145-R6-3, A06B-146-R6-1, A06B-146-R6-2, A06B-147-R6-1 | 8077, 8073, 8075 |  |
| A03 | 03 Other WIP - container stock | 2026-10-22T16:32:00-04:00 / 2026-10-22T16:41:00-04:00 | A03-291-R5-1 |  |  |
| A05C | 05 Granola bulk WIP | 2026-10-27T11:52:00-04:00 / 2026-10-27T12:17:00-04:00 | A05C-121-1 |  |  |
| A06C | 06 Finished goods - all customers | 2026-10-27T11:44:00-04:00 / 2026-10-27T12:17:00-04:00 | A06C-171-1, A06C-173-1, A06C-174-1, A06C-175-1, A06C-175-2, A06C-175-3, A06C-202-1, A06C-202-2, A06C-203-1 |  |  |
| P09 | 05 Granola bulk WIP | 2026-10-27T13:26:00-04:00 / 2026-10-27T13:26:00-04:00 | P09-107-1, P09-107-2, P09-107-3, P09-107-4, P09-107-5, P09-107-6 |  |  |

## Lot comparisons

| Item / lot | Unit | Physical total | After-count movement | Internal move correction | FL now | Expected now | Proposed change | Estimated | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 / 26-01-29-DUTVF-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 2 / 26-02-25-DUTC-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 2 / 26-04-27-DUTC-005 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 2 / 26-06-22-DUTC-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 2 / 26-08-03-DUTC-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 2 / 26-10-12-DUTC-002 | lb | 175 | -78 | 0 | 97 | 97 | 0 | False | READY |
| 3 / 26-01-29-DUTVF-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 3 / 26-02-25-STAR-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 3 / 26-04-22-SAMI-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 3 / 26-05-07-SAMI-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 3 / 26-07-24-REFR-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 3 / 26-09-22-NATI-001 | lb | 375 | 0 | 0 | 375 | 375 | 0 | False | READY |
| 4 / 26-02-16-DUTC-004 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 4 / 26-04-27-DUTC-004 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 4 / 08226 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 4 / 26-06-22-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 4 / 26-08-03-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 4 / 26-10-12-DUTC-001 | lb | 175 | -116 | 0 | 59 | 59 | 0 | False | READY |
| 6 / 26-03-02-JOEL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 6 / 26-05-04-ABAK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 6 / 26-08-04-ABAK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 7 / 26-02-06-FOUND-006 | lb | 97 | 0 | 0 | 97 | 97 | 0 | False | READY |
| 7 / 26-04-27-DUTC-001 | lb | 50 | 0 | 0 | 50 | 50 | 0 | False | READY |
| 8 / 26-02-06-FOUND-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 8 / 26-02-06-FOUND-004 | lb | 50 | 0 | 0 | 50 | 50 | 0 | False | READY |
| 8 / 26-04-27-DUTC-003 | lb | 100 | 0 | 0 | 100 | 100 | 0 | False | READY |
| 9 / 26-02-06-FOUND-007 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 9 / 6011 | lb | 12000 | 0 | 0 | 12000 | 12000 | 0 | False | READY |
| 9 / 25212 | lb | 4125 | 0 | 0 | 4125 | 4125 | 0 | False | READY |
| 9 / 25134 | lb | 2850 | 0 | 0 | 2850 | 2850 | 0 | False | READY |
| 9 / 25213 | lb | 550 | 0 | 0 | 550 | 550 | 0 | False | READY |
| 9 / 25133 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 10 / 5267 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 10 / 6012 | lb | 8400 | 0 | 0 | 8400 | 8400 | 0 | False | READY |
| 10 / 24307 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 10 / 6036 | lb | 4200 | 0 | 0 | 4200 | 4200 | 0 | False | READY |
| 12 / 26-01-30-PHIL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 12 / 26-01-30-FRAN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 12 / 26-02-05-FOUND-004 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 12 / 26-02-09-FRAN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 12 / 26-02-12-FOUND-021 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 12 / 26-02-18-BLEN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 12 / 26-02-24-PHIL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 12 / 26-05-01-FRAN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 12 / 6012 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 12 / 6020 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 12 / 6036 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 12 / 6037 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 12 / 6013 | lb | 8100 | -1200 | 0 | 6900 | 6900 | 0 | False | READY |
| 13 / 26-02-03-FOUND-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 13 / 26-02-05-SAEM-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 13 / 26-02-05-SAEM-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 13 / 25120 | lb | 3875 | -78 | 0 | 3797 | 3797 | 0 | False | READY |
| 14 / 26-02-05-FOUND-005 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 14 / 26-02-06-FOUND-008 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 14 / 6020 | lb | 6200 | 0 | 0 | 6200 | 6200 | 0 | False | READY |
| 14 / 5119 | lb | 550 | 0 | 0 | 550 | 550 | 0 | False | READY |
| 14 / 6044 | lb | 9850 | 0 | 0 | 9850 | 9850 | 0 | False | READY |
| 14 / 25120 | lb | 3000 | 0 | 0 | 3000 | 3000 | 0 | False | READY |
| 14 / 24289 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 14 / VINA202524 | lb | 13200 | 0 | 0 | 13200 | 13200 | 0 | False | READY |
| 16 / 26-02-05-FOUND-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 16 / 26-04-21-ESSE-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 16 / 26-07-13-ESSE-001 | lb | 635 | -30 | 0 | 605 | 605 | 0 | False | READY |
| 16 / 26-10-19-ESSE-001 | lb | 4500 | 0 | 0 | 4500 | 4500 | 0 | False | READY |
| 17 / 26-02-04-FOUN-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 17 / 26-02-16-DUTC-006 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 17 / 26-05-04-DUTC-004 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 17 / 26-08-17-ABAK-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 17 / 26-09-16-ABAK-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 17 / 26-10-12-DUTC-004 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 18 / 26-02-03-FOUND-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 18 / 26-04-21-PARK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 18 / 26-07-21-PARK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 18 / 26-10-07-PARK-001 | lb | 720 | 0 | 0 | 720 | 720 | 0 | False | READY |
| 19 / 26-02-03-FOUND-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 20 / 26-02-03-FOUND-004 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 20 / 26-07-21-PARK-003 | lb | 24 | 0 | 0 | 24 | 24 | 0 | False | READY |
| 21 / 26-02-03-FOUND-005 | lb | 24 | 0 | 0 | 24 | 24 | 0 | False | READY |
| 22 / 26-02-03-FOUND-006 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 23 / 26-02-03-FOUND-007 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 25 / 26-02-03-FOUND-008 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 26 / 26-02-03-FOUND-009 | lb | 480 | 0 | 0 | 480 | 480 | 0 | False | READY |
| 26 / 26-07-21-PARK-002 | lb | 80 | 0 | 0 | 80 | 80 | 0 | False | READY |
| 27 / 26-02-03-FOUND-010 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 27 / 26-03-02-FOUN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 27 / 26-05-04-DUTC-005 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 27 / 26-10-20-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 28 / 26-02-03-FOUND-011 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 28 / 26-02-25-DUTC-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 28 / 26-04-27-DUTC-006 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 28 / 26-06-22-DUTC-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 28 / 26-08-03-DUTC-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 28 / 26-10-12-DUTC-003 | lb | 50 | 0 | 0 | 50 | 50 | 0 | False | READY |
| 29 / 26-02-03-FOUND-012 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 29 / 26-04-24-TILL-001 | lb | 948 | -42 | 0 | 906 | 906 | 0 | False | READY |
| 29 / 26-09-28-TILL-001 | lb | 7680 | 0 | 0 | 7680 | 7680 | 0 | False | READY |
| 30 / 26-02-03-FOUND-013 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 30 / 26-02-11-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 30 / 26-03-11-DAVI-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 30 / 26-03-19-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 30 / 26-04-28-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 30 / 26-07-10-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 30 / 26-09-03-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 30 / 26-10-06-DUTC-001 | lb | 1741.8 | 0 | 0 | 1741.8 | 1741.8 | 0 | False | READY |
| 31 / 26-02-12-LACR-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 31 / 26-04-14-QUAL-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 31 / 26-04-27-QUAL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 32 / 26-01-28-DUTV-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 32 / 26-01-30-QUAL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 32 / 26-02-10-FOUND-004 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 32 / 26-04-14-QUAL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 32 / 26-05-08-QUAL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 32 / 26-06-05-QUAL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 32 / 26-06-12-QUAL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 32 / 26-06-30-QUAL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 32 / 26-07-09-QUAL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 32 / 26-07-15-QUAL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 32 / 26-08-04-QUAL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 32 / 26-08-21-QUAL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 32 / 26-08-28-QUAL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 32 / 26-09-28-QUAL-001 | lb | 500 | -156 | 0 | 344 | 344 | 0 | False | READY |
| 32 / 26-10-12-DUTC-005 | lb | 2000 | 0 | 0 | 2000 | 2000 | 0 | False | READY |
| 32 / 26-10-22-QUAL-001 | lb | 43750 | 0 | 0 | 43750 | 43750 | 0 | False | READY |
| 33 / 26-02-25-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 33 / 251121N | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 33 / 26-05-04-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 33 / 26-06-08-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 33 / 26-08-17-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 33 / 26-09-28-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 34 / 26-02-03-FOUND-014 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 34 / 26-02-09-CBSF-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 34 / 26-03-13-CBSF-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 34 / 26-04-29-CBSF-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 34 / 26-05-04-CBSF-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 34 / 26-05-28-CBSF-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 34 / 26-06-25-CBSF-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 34 / 26-07-01-CBSF-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 34 / 26-07-24-CBSF-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 34 / 26-08-25-CBSF-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 34 / 26-09-17-CBSF-001 | lb | 500 | -44 | 0 | 456 | 456 | 0 | False | READY |
| 34 / 26-10-20-CBSF-001 | lb | 6000 | 0 | 0 | 6000 | 6000 | 0 | False | READY |
| 35 / 26-02-06-FOUND-005 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 35 / 26-04-27-DUTC-007 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 35 / 26-06-29-DUTC-002 | lb | 110 | 0 | 0 | 110 | 110 | 0 | False | READY |
| 36 / 26-02-04-FOUN-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 36 / 26-02-16-DUTC-005 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 36 / 26-05-04-DUTC-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 36 / 26-09-16-ABAK-004 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 36 / 26-10-06-ABAK-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 38 / 26-02-03-FOUND-015 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 38 / 26-02-16-DUTC-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 38 / 26-02-18-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 38 / 26-04-27-DUTC-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 38 / 26-06-29-DUTC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 39 / 26-02-03-FOUND-016 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 39 / 26-04-21-ESSE-001 | lb | 493.3 | -15 | 0 | 478.3 | 478.3 | 0 | False | READY |
| 39 / 26-07-13-ESSE-002 | lb | 2450 | 0 | 0 | 2450 | 2450 | 0 | False | READY |
| 39 / 26-08-05-ESSE-001 | lb | 4706.7 | 0 | 0 | 4706.7 | 4706.7 | 0 | False | READY |
| 40 / 26-02-03-FOUND-017 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 40 / 26-02-06-JACK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 40 / 26-02-17-JACK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 40 / 26-03-04-JACK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 40 / 26-04-14-LINK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 40 / 26-06-10-NEWE-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 40 / 26-08-06-SWEE-001 | lb | 8347.51 | -660 | 0 | 7687.51 | 7687.51 | 0 | False | READY |
| 40 / 26-09-30-NEWE-001 | lb | 44000 | 0 | 0 | 44000 | 44000 | 0 | False | READY |
| 41 / 26-02-03-FOUND-018 | lb | 200 | 0 | 0 | 200 | 200 | 0 | False | READY |
| 42 / 26-02-03-FOUND-019 | lb | 1050 | 0 | 0 | 1050 | 1050 | 0 | False | READY |
| 43 / 26-01-29-DUTVF-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 43 / 26-02-03-FOUND-020 | lb | 100 | 0 | 0 | 100 | 100 | 0 | False | READY |
| 43 / 26-02-16-DUTC-001 | lb | 500 | 0 | 0 | 500 | 500 | 0 | False | READY |
| 44 / 26-02-03-FOUND-021 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 44 / 26-02-09-FOUND-005 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 44 / 26-03-12-JACK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 44 / 26-04-29-JACK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 44 / 26-05-05-JACK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 44 / 26-05-26-JACK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 44 / 26-06-28-JACK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 44 / 26-07-06-JACK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 44 / 26-07-09-JACK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 44 / 26-08-04-JACK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 44 / 26-08-26-JACK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 44 / 26-09-30-JACK-001 | lb | 1850 | 0 | 0 | 1850 | 1850 | 0 | False | READY |
| 45 / 26-02-03-FOUND-022 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 45 / 26-08-26-HONI-001 | lb | 55 | -55 | 0 | 0 | 0 | 0 | False | READY |
| 46 / 26-01-30-FOUND-018 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 46 / 26-02-03-FOUND-023 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 46 / 26-02-05-STAR-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 46 / 26-03-18-STAR-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 46 / 26-04-14-STAR-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 46 / 26-05-05-KADO-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 46 / 26-06-24-STAR-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 46 / 26-07-06-NATI-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 46 / 26-09-04-NATI-001 | lb | 2650 | -78 | 0 | 2572 | 2572 | 0 | False | READY |
| 49 / 26-02-04-FOUN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 49 / 26-02-16-DUTC-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 49 / 26-05-04-DUTC-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 49 / 26-09-16-ABAK-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 50 / 26-02-05-FOUND-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 50 / 26-02-05-FOUND-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 52 / 26-02-03-FOUND-024 | lb | 455 | 0 | 0 | 455 | 455 | 0 | False | READY |
| 54 / 26-01-30-FOUND-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 55 / 26-01-30-FOUND-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 55 / 26-03-10-BLUE-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 55 / 26-03-19-BLUE-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 56 / 26-01-30-FOUND-003 | lb | 484 | 0 | 0 | 484 | 484 | 0 | False | READY |
| 58 / 26-01-30-FOUND-004 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 58 / 26-03-03-BLUE-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 58 / 26-05-11-BLUE-002 | lb | 1400 | 0 | 0 | 1400 | 1400 | 0 | False | READY |
| 59 / 26-01-30-FOUND-005 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 60 / 26-01-30-FOUND-006 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 61 / 26-01-30-FOUND-007 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 61 / 26-01-30-FOUND-017 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 61 / 26-02-03-FOUND-025 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 61 / 26-02-03-BS-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 61 / 26-02-06-BLUE-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 61 / 26-03-25-BLUE-001 | lb | 2200 | 0 | 0 | 2200 | 2200 | 0 | False | READY |
| 61 / 26-05-11-BLUE-001 | lb | 2200 | 0 | 0 | 2200 | 2200 | 0 | False | READY |
| 61 / 26-06-30-BLUE-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 62 / 26-01-30-FOUND-008 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 62 / 26-02-04-BLUE-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 62 / 26-06-01-BLUE-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 64 / 26-01-30-FOUND-009 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 65 / 26-01-30-FOUND-010 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 66 / 26-01-30-FOUND-011 | lb | 1725 | 0 | 0 | 1725 | 1725 | 0 | False | READY |
| 67 / 26-01-30-FOUND-012 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 67 / 26-02-12-BLUE-001 | lb | 2000 | 0 | 0 | 2000 | 2000 | 0 | False | READY |
| 67 / 121125 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 68 / 26-01-30-FOUND-013 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 68 / 26-03-03-BLUE-001 | lb | 3200 | 0 | 0 | 3200 | 3200 | 0 | False | READY |
| 69 / 26-01-30-FOUND-014 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 70 / 26-01-30-FOUND-015 | lb | 110 | 0 | 0 | 110 | 110 | 0 | False | READY |
| 71 / 26-01-30-FOUND-016 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 72 / 26-02-12-BARR-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 72 / 26-04-14-UNKN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 72 / 26-05-13-ABAK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 72 / 26-06-09-ABAK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 72 / 26-08-04-ABAK-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 72 / 26-08-17-ABAK-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 72 / 26-09-10-ABAK-001 | lb | 525 | 0 | 0 | 525 | 525 | 0 | False | READY |
| 72 / 26-09-16-ABAK-001 | lb | 500 | 0 | 0 | 500 | 500 | 0 | False | READY |
| 72 / 26-10-06-ABAK-001 | lb | 500 | 0 | 0 | 500 | 500 | 0 | False | READY |
| 73 / 26-02-10-CALL-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 75 / 26-02-06-CREA-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 75 / 26-04-16-UNKN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 75 / 26-06-09-CREA-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 75 / 26-07-28-CREA-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 75 / 26-09-22-CREA-001 | lb | 3950 | 0 | 0 | 3950 | 3950 | 0 | False | READY |
| 75 / 26-10-06-CREA-001 | lb | 8900 | 0 | 0 | 8900 | 8900 | 0 | False | READY |
| 107 / JAN 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / B26-0204-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / FEB-03-2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / FEB-04-26 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / 26-02-05-FOUND-006 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / FEB-02-26 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / FEB-05-2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / FEB 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / FEB 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / FEB 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / FEB 10 2026-02 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / FEB 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAR 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / B26-0304-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAR 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAR 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAR 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAR 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAR 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAR 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAR 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / APR 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / 2026-04-15 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / 2026-04-21 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / ABR 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / APR 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / APR 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / 26-05-04-GRAN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / 26-05-05-GRAN-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / 26-05-06-GRAN-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAY 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAY 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAY 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAY 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAY 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAY 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAY 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAY 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / MAY 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUN 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUN 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUN 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUN 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUN 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUN 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUN 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUN 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / Jul 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL 2 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL 07 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL15 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL 15 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / Jul 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / JUL 31 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / AUG 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / AUG 04 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / AUG 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / AUG 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / AUG 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / AUG 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / AUG 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / AUG 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / AUG 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / AUG 31 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / SEP 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / SEP 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / SEP 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / SEP 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / SEP 15 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / SEP 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / SEP 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / SEP 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / SEP 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / SEP 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / SEP 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 107 / OCT 05 2026 | lb | 1387 | 0 | 0 | 1387 | 1387 | 0 | False | READY |
| 107 / OCT 12 2026 | lb | 768 | 0 | 0 | 768 | 768 | 0 | False | READY |
| 107 / OCT 13 2026 | lb | 2376 | 0 | 0 | 2376 | 2376 | 0 | False | READY |
| 107 / OCT 14 2026 | lb | 1292 | 0 | 0 | 1292 | 1292 | 0 | False | READY |
| 107 / OCT 15 2026 | lb | 122 | 0 | 0 | 122 | 122 | 0 | False | READY |
| 107 / OCT 16 2026 | lb | 1938 | 0 | 0 | 1938 | 1938 | 0 | False | READY |
| 108 / FEB 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / FEB 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / MAR 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / 2026-04-20 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / APR 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / APR 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / MAY 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / MAY 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / JUN 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / JUN 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / JUN 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / JUN 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / JUL 07 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / JUL 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / JUL 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / AUG 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / AUG 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / AUG 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / SEP 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / SEP 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / SEP 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / SEP 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / OCT 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 108 / OCT 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 109 / APR 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 109 / MAY 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 109 / JUN 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 111 / FEB 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 111 / MAY 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 111 / JUN 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 111 / JUN 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 112 / FEB 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 112 / FEB 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 112 / FEB 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 112 / APR 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 112 / 26-05-05-VAN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 112 / 26-05-06-VAN-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 112 / JUN 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 112 / JUL 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 112 / SEP 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 112 / SEP 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 113 / JAN 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 113 / MAY 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 113 / JUN 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 113 / SEP 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / B26-0212-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / FEB 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / FEB 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / FEB 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / 26-02-25 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / FEB 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / APR 10 2026 Lot | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / APR 12 2026 Lote | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / APR 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / APR 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / APR 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / APR 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / MAY 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / 26-05-04-SSCHOC-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / MAY 15 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / MAY 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / MAY 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / JUN 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / JUN 15 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / JUN 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / JUN 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / JUN 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / JUN 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / JUL 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / JUL 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / JUL 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / AUG 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / AUG 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / AUG 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / AUG 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / AUG 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / SEP 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / SEP 04 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / SEP 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / SEP 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / SEP 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 114 / OCT 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / FEB 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / B26-0302-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / MAR 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / APR 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / APR 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / 2026-04-17 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / APR 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / APR 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / MAY 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / MAY 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / JUN 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / JUN 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / JUN 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / JUN 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / AUG 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / AUG 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / AUG 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / AUG 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / SEP 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / SEP 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / SEP 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / SEP 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / SEP 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / SEP 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / OCT 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / OCT 07 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / OCT 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / OCT 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 116 / OCT 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 118 / 26-05-05-CRAN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 118 / MAY 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 118 / JUN 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 118 / AUG 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 118 / BB081727 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 118 / OCT 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 119 / B26-0302-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 119 / 26-05-04-LOWCARB-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 119 / JUN 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 119 / AUG 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 119 / AUG 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 119 / OCT 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 120 / B26-0302-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 120 / 26-05-05-LOWCARB-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 120 / JUN 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 120 / AUG 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 120 / OCT 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 121 / B26-0202-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 121 / B26-0203-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 121 / B26-0306-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 121 / MAR 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 121 / MAR 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 121 / MAR 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 121 / MAR 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 121 / MAR 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 121 / JUL 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 121 / JUL 24 2026 | lb | 1050 | 0 | 0 | 1050 | 1050 | 0 | False | READY |
| 122 / JUL 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 123 / B26-0305-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 123 / B26-0306-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 124 / FEB 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 124 / FEB 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 124 / B26-0304-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 124 / 26-05-06-HAZE-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 124 / MAY 07 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / FEB 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / MAR 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / MAR 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / MAR 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / MAR 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / MAR 31 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / APR 10 2026 Lot | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / 2026-04-20 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / APR 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / MAY 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / MAY 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / JUN 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / JUN 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / JUL 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / JUL 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / AUG 04 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / AUG 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / SEP 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / SEP 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / SEP 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / SEP 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / SEP 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 125 / OCT 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB-03-2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / 020526 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / B26-0212-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / TEST-WATER-3 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / FEB 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAR 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAR 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAR 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAR 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAR 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAR 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAR 31 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / APR 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / 2026-04-14 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / 2026-04-15 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / 2026-04-16 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / 2026-04-21 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / APR 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / APR 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / ABR 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / APR 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / APR 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / B26-0430-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / 26-05-04-COCO-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / 26-05-05-COCO-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / 26-05-06-COCO-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAY 07 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAY 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAY 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAY 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAY 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAY 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAY 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAY 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAY 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAY 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAY 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAY 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / MAY 31 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 04 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 15 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUN 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUL 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUL 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUL 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUL 07 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUL 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUL 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUL 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUL 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUL 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUL 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / JUL 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 04 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 07 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / AUG 31 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / SEP 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / SEP 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / SEP 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / SEP 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / SEP 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / SEP 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / SEP 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / SEP 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / SEP 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / SEP 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / SEP 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / SEP 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / SEP 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / SEP 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / OCT 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / OCT 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / OCT 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / OCT 15 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / OCT 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / OCT 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / OCT 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 126 / OCT 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / FEB 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / MAR 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / MAR 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / 2026-04-17 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / APR 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / MAY 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / JUN 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / JUN 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / JUL 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / AUG 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / AUG 04 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / AUG 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / SEP 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / SEP 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / SEP 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 127 / SEP 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / MAR 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / APR 10 2026 Lot | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / 2026-04-17 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / APR 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / MAY 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / MAY 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / MAY 15 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / MAY 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / MAY 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUN 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUN 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUN 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUN 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUN 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUN 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUN 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUN 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUL 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUL 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUL 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUL 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUL 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUL 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUL 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUL 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / JUL 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / OCT 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / OCT 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / OCT 07 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / OCT 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / OCT 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 128 / OCT 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 129 / OCT 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 132 / MAY 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 133 / FEB 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 133 / FEB 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 133 / MAY 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 133 / JUL 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 134 / FEB 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 134 / APR 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 134 / JUN 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 134 / SEP 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 134 / SEP 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 135 / JAN 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 135 / APR 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 135 / MAY 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 135 / JUN 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 135 / SEP 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / FEB 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / FEB 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / B26-0304-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / MAR 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / MAR 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / MAR 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / MAR 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / 2026-04-13 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / 2026-04-17 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / ABR 15 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / APR 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / MAY 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / MAY 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / MAY 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / MAY 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / MAY 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / JUN 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / JUN 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / JUL 07 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / JUL 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / JUL 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / AUG 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / AUG 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / AUG 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / AUG 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / AUG 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / SEP 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / SEP 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / SEP 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / SEP 15 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / SEP 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / SEP 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / SEP 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 136 / OCT 12 2026 | lb | 3250 | 0 | 0 | 3250 | 3250 | 0 | False | READY |
| 136 / OCT 13 2026 | lb | 1500 | 0 | 0 | 1500 | 1500 | 0 | False | READY |
| 136 / OCT 15 2026 | lb | 3000 | 0 | 0 | 3000 | 3000 | 0 | False | READY |
| 137 / FEB 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / FEB 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / MAR 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / MAR 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / MAR 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / APR 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / MAY 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / MAY 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / MAY 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / MAY 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / MAY 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / JUN 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / JUL 07 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / AUG 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / AUG 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / AUG 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / AUG 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / AUG 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / AUG 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 137 / AUG 31 2026 | lb | 100 | 0 | 0 | 100 | 100 | 0 | False | READY |
| 137 / SEP 17 2026 | lb | 440 | 0 | 0 | 440 | 440 | 0 | False | READY |
| 137 / OCT 15 2026 | lb | 1400 | 0 | 0 | 1400 | 1400 | 0 | False | READY |
| 138 / MAR 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 138 / MAR 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 138 / MAR 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 138 / MAY 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 138 / MAY 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 138 / MAY 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 138 / JUN 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 138 / JUL 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 138 / AUG 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 138 / SEP 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 139 / FEB 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 139 / JUN 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 140 / APR 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 140 / MAY 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 140 / JUN 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 141 / MAR 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 141 / 2026-04-17 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 141 / APR 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 141 / JUN 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 141 / AUG 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 141 / SEP 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 141 / OCT 14 2026 | lb | 1250 | 0 | 0 | 1250 | 1250 | 0 | False | READY |
| 142 / FN-0219 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 142 / MAY 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 142 / JUN 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 142 / SEP 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 142 / OCT 06 2026 | lb | 550 | 0 | 0 | 550 | 550 | 0 | False | READY |
| 143 / ABR 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 143 / MAY 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / 26-02-05-FOUND-007 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / 26-02-06-FOUND-009 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / 26-02-09-FOUND-006 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / 26-02-09-FOUND-007 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / 26-02-09-FOUND-008 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / 26-02-09-FOUND-009 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / 26-02-10-FOUND-006 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / 26-02-10-FOUND-007 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / 26-02-13-FOUND-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / FEB 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / FEB 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / MAR 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / MAR 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / MAR 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / MAR 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / 26-03-23-PHYS-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / MAR 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / ABR 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / MAY 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / MAY 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / MAY 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / MAY 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUN 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUN 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUN 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUN 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUN 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUN 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUN 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUL 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUL 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUL 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUL 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUL 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUL 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUL 15 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUL 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUL 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUL 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUL 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUL 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUL 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / JUL 31 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / AUG 04 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / AUG 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / AUG 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / AUG 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / AUG 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / AUG 31 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / SEP 17 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / SEP 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 144 / SEP 29 2026 | lb | 4200 | 0 | 0 | 4200 | 4200 | 0 | False | READY |
| 144 / OCT 05 2026 | lb | 2800 | 0 | 0 | 2800 | 2800 | 0 | False | READY |
| 144 / OCT 12 2026 | lb | 1400 | 0 | 0 | 1400 | 1400 | 0 | False | READY |
| 145 / BB021627 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / FEB 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB022427 Lot | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / FEB 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB030327 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / 26-02-25 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB041327 Lot | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB041327 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / MAY 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB050427 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB052627 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB060127 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB060827 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB061527 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB062227 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB070827 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB071327 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB072027 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB080327 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB081027 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB081727 USA | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB081727 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB082427 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB090827 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB091427 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB092827 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 145 / BB093027 | lb | 2272.5 | 0 | 0 | 2272.5 | 2272.5 | 0 | False | READY |
| 145 / BB100527 | lb | 1597.5 | 0 | 0 | 1597.5 | 1597.5 | 0 | False | READY |
| 145 / BB101227 | lb | 1755 | 0 | 0 | 1755 | 1755 | 0 | False | READY |
| 146 / FEB 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 146 / BB022427 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 146 / BB030327 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 146 / BB041327 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 146 / BB061527 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 146 / BB070827 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 146 / BB081027 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 146 / BB081727 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 146 / BB082427 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 146 / BB083127 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 146 / BB090827 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 146 / BB091427 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 146 / BB092427 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 146 / BB100527 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 146 / BB101227 | lb | 450 | 0 | 0 | 450 | 450 | 0 | False | READY |
| 146 / BB101927 | lb | 4567.5 | 0 | 0 | 4567.5 | 4567.5 | 0 | False | READY |
| 147 / BB070827 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 147 / BB081727 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 147 / BB082427 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 147 / BB101927 | lb | 675 | 0 | 0 | 675 | 675 | 0 | False | READY |
| 148 / B26-0302-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 148 / BB070827 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 148 / BB082427 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 148 / BB101927 | lb | 0 | 727.5 | 0 | 727.5 | 727.5 | 0 | False | READY |
| 149 / B26-0302-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 149 / BB070827 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 149 / BB082427 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 149 / BB101927 | lb | 0 | 667.5 | 0 | 667.5 | 667.5 | 0 | False | READY |
| 150 / SW2607890 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 150 / SW2611090 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 150 / MAR 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 150 / SW2620590 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 151 / 26-02-10-FOUND-005 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 151 / 26-02-12-FOUND-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 151 / SW2607708 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 151 / SW2620891 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 152 / B26-0305-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 152 / SW2607593 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 153 / SW2606892 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 153 / SW2612692 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 154 / APR 10 2026 Lot | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 155 / 26-02-12-FOUND-014 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / 26-02-10-FOUND-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / 26-02-10-FOUND-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / 26-02-10-FOUND-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / 26-02-12-FOUND-012 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / 26-02-12-FOUND-013 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / LOT FEB 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / FEB 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / APR 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / MAY 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / MAY 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / JUN 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / JUN 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / JUL 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / JUL 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / JUL 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / AUG 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / AUG 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / AUG 31 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / SEP 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / SEP 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / OCT 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 156 / OCT 06 2026 | lb | 1150 | 0 | 0 | 1150 | 1150 | 0 | False | READY |
| 157 / 26-02-04-VINN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / 26-02-12-FOUND-010 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / 26-02-12-FOUND-011 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / 2026-04-21 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / MAY 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / MAY 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / JUN 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / JUL 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / JUL 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / JUL 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / AUG 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / AUG 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / SEP 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / SEP 04 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / SEP 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 157 / SEP 17 2026 | lb | 450 | -450 | 0 | 0 | 0 | 0 | False | READY |
| 157 / SEP 28 2026 | lb | 1100 | -1100 | 0 | 0 | 0 | 0 | False | READY |
| 157 / OCT 20 2026 | lb | 725 | -725 | 0 | 0 | 0 | 0 | False | READY |
| 157 / OCT 21 2026 | lb | 725 | -225 | 0 | 500 | 500 | 0 | False | READY |
| 157 / OCT 22 2026 | lb | 0 | 1100 | 0 | 1100 | 1100 | 0 | False | READY |
| 157 / OCT 23 2026 | lb | 0 | 1100 | 0 | 1100 | 1100 | 0 | False | READY |
| 158 / 26-02-12-FOUND-015 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 158 / MAR 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 158 / AUG 04 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 158 / AGO 19 2026 | lb | 160 | 0 | 0 | 160 | 160 | 0 | False | READY |
| 159 / MAR 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 159 / MAY 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 159 / JUN 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 159 / JUL 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 159 / JUL 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 159 / JUL 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 159 / JUL 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 159 / AUG 21 2026 | lb | 110 | 0 | 0 | 110 | 110 | 0 | False | READY |
| 160 / 26-02-12-FOUND-016 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / MAR 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / 2026-04-20 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / MAY 04 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / MAY 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / MAY 15 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / MAY 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / MAY 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / MAY 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUN 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUN 15 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUN 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUN 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUN 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUN 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUN 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUL 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / APR 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUL 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUL 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUL 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUL 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JAN 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUL 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUL 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUL 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / JUL 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / OCT 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / OCT 07 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / OCT 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / OCT 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 160 / OCT 12 2026 | lb | 1375 | 0 | 0 | 1375 | 1375 | 0 | False | READY |
| 160 / OCT 13 2026 | lb | 1000 | 0 | 0 | 1000 | 1000 | 0 | False | READY |
| 161 / 26-02-12-FOUND-019 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / FEB 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / MAR 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / MAR 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / MAR 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / MAR 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / MAR 31 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / 2026-04-20 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / APR 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / APR 10 2026 Lot | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / MAY 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / MAY 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / MAY 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / JUN 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / JUN 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / JUL 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / JUL 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / AUG 04 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / AUG 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / AUG 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / SEP 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / SEP 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / SEP 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / SEP 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 161 / SEP 30 2026 | lb | 690 | 0 | 0 | 690 | 690 | 0 | False | READY |
| 161 / OCT 20 2026 | lb | 1260 | 0 | 0 | 1260 | 1260 | 0 | False | READY |
| 162 / 26-02-12-FOUND-017 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / 26-02-12-FOUND-018 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / LOT FEB 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / FEB 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / MAR 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / MAR 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / MAR 31 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / APR 13 2026 lot | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / 2026-04-17 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / APR 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / MAY 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / JUN 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / JUN 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / JUN 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / JUN 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / JUL 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / JUL 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / JUL 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / AUG 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / AUG 04 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / AUG 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / AUG 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / STAGED-FEESERS-67476 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / SEP 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / SEP 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / SEP 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 162 / OCT 05 2026 | lb | 830 | 0 | 0 | 830 | 830 | 0 | False | READY |
| 163 / 26-02-12-FOUND-020 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / FEB 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / MAR 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / MAR 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / 2026-04-17 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / APR 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / MAY 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / JUN 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / JUN 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / JUL 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / JUL 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / JUL 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / AUG 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / AUG 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / SEP 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / SEP 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 163 / SEP 29 2026 | lb | 1280 | 0 | 0 | 1280 | 1280 | 0 | False | READY |
| 163 / SEP 30 2026 | lb | 1090 | 0 | 0 | 1090 | 1090 | 0 | False | READY |
| 164 / FEB-04-2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / FEB-03-2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-02-05-FOUND-008 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-02-05-FOUND-009 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-02-06-FOUND-010 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-02-06-FOUND-011 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-02-10-FOUND-008 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-02-11-FOUND-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / FEB 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-02-12-FOUND-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-02-12-FOUND-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-02-12-FOUND-004 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-02-12-FOUND-005 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-02-12-FOUND-006 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-02-12-FOUND-007 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-02-12-FOUND-008 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-02-16-FOUND-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / FEB 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / FEB 16 2026-A | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / FEB 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / FEB 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / FEB 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / FEB 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAR 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAR 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAR 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAR 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 26-03-23-PHYS-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / APR 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 2026-04-13 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 2026-04-14 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 2026-04-15 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 2026-04-16 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / 2026-04-21 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / ABR 21 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / ABR 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / ABR 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / APR 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / APR 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / APR 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / APR 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / ABR 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 04 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 05 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 07 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 011 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / MAY 31 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 03 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 04 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 15 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 19 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 22 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 23 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUN 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUL 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUL 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUL 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUL 07 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUL 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUL 28 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / JUL 30 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 07 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 13 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 14 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 17 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 18 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 20 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 24 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 25 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 26 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 26 2027 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 27 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / AUG 31 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / SEP 01 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / SEP 02 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / SEP 08 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / SEP 09 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / SEP 10 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / SEP 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 164 / SEP 22 2026 | lb | 1400 | 0 | 0 | 1400 | 1400 | 0 | False | READY |
| 164 / SEP 23 2026 | lb | 4200 | 0 | 0 | 4200 | 4200 | 0 | False | READY |
| 164 / SEP 24 2026 | lb | 4200 | 0 | 0 | 4200 | 4200 | 0 | False | READY |
| 164 / SEP 25 2026 | lb | 3740 | 0 | 0 | 3740 | 3740 | 0 | False | READY |
| 164 / SEP 28 2026 | lb | 3260 | 0 | 0 | 3260 | 3260 | 0 | False | READY |
| 164 / OCT 13 2026 | lb | 1400 | 0 | 0 | 1400 | 1400 | 0 | False | READY |
| 164 / OCT 15 2026 | lb | 4200 | 0 | 0 | 4200 | 4200 | 0 | False | READY |
| 164 / OCT 16 2026 | lb | 2800 | 0 | 0 | 2800 | 2800 | 0 | False | READY |
| 164 / OCT 20 2026 | lb | 1400 | 0 | 0 | 1400 | 1400 | 0 | False | READY |
| 164 / OCT 21 2026 | lb | 1000 | 0 | 0 | 1000 | 1000 | 0 | False | READY |
| 165 / 26-02-12-FOUND-009 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 602271 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 26-03-12-FOUN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 603061 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 601141 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 603231 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 26-02-06-CREA-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 604171 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 26-04-16-UNKN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 605081 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 605291 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 606011 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 606051 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 606121 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 606261 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 607021 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 607011 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 607101 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 607231 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 607311 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 608041 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 608101 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 608261 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 608281 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 609111 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 609221 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 6091511 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 609151 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 609051 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 610021 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 610071 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 165 / 610161 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 166 / 26-01-30-PH-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 166 / FEB-04-2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 169 / 26-02-04-ACME-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 171 / 26-02-06-FOUND-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 171 / 26-02-06-FOUND-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 171 / 26-02-09-FOUND-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 171 / 26-02-09-FOUND-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 171 / 26-02-09-FOUND-003 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 171 / 26-02-09-FOUND-004 | lb | 5 | 0 | 0 | 5 | 5 | 0 | False | READY |
| 173 / 26-02-10-CREA-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 173 / 26-02-10-CREA-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 173 / 26-06-02-CREA-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 173 / 26-06-02-CREA-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 173 / 26-09-23-CREA-001 | lb | 34375 | 0 | 0 | 34375 | 34375 | 0 | False | READY |
| 174 / 25132 | lb | 12050 | 0 | 0 | 12050 | 12050 | 0 | False | READY |
| 174 / 25114 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 175 / 26-02-16-EURO-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 175 / 26-03-03-INVE-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 175 / 26-03-03-INVE-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 175 / 26-03-10-FOUN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 175 / 26-04-17-EURO-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 175 / 26054 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 175 / 261212585122 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 175 / 261222585122 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 175 / 26-06-10-TRIS-001 | lb | 4650 | 0 | 0 | 4650 | 4650 | 0 | False | READY |
| 175 / 26-09-24-TRIS-001 | lb | 28600 | 0 | 0 | 28600 | 28600 | 0 | False | READY |
| 178 / 601262 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 178 / 26-02-10-CREA-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 178 / 605182 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 178 / 26-06-02-CREA-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 178 / 605192 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 178 / 610211 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 179 / FN-0219 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 179 / MAY 12 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 179 / JUN 29 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 179 / SEP 16 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 179 / OCT 06 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 183 / BULK-#1-YTD | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 189 / 6036 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 190 / 26-04-22-FRAN-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 194 / 6020 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 196 / 6086 | lb | 11375 | 0 | 0 | 11375 | 11375 | 0 | False | READY |
| 202 / 26-02-27-EURO-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 202 / 26-03-20-INVE-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 202 / 26-05-01-EURO-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 202 / 26054 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 202 / 550075853 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 202 / 261222444122 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 202 / 26-06-10-TRIS-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 202 / 26-08-03-EURO-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 202 / 26-08-04-EURO-001 | lb | 1300 | 0 | 0 | 1300 | 1300 | 0 | False | READY |
| 202 / 26-09-24-TRIS-002 | lb | 12600 | 0 | 0 | 12600 | 12600 | 0 | False | READY |
| 203 / 26-03-12-FOUN-002 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 203 / 25216 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 203 / 25114 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 203 / 26-06-05-EURO-001 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 203 / 26-08-14-EURO-001 | lb | 3625 | 0 | 0 | 3625 | 3625 | 0 | False | READY |
| 206 / SW2605619 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 209 / SW2607708 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 285 / BULK-#9-YTD | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 286 / AUG 11 2026 | lb | 0 | 0 | 0 | 0 | 0 | 0 | False | READY |
| 291 / AUG 14 MIX | container | 1 | 0 | 0 | 1 | 1 | 0 | False | READY |

## Owner-approved posts in this preview

Every reset post is an FL ticket under the named owner key with reason `physical_count` and a receipt number. Rows below meet an FL owner hold or close an open shortage. Signing the approval with `Owner approves held entries: Y` and listing every new extra-kosher lot authorizes the executor to approve exactly these holds with the same key; anything else stops.

| Item / lot | FL action | Change | Owner holds | Closes shortages |
| --- | --- | --- | --- | --- |

## Holds


## Move-log follow-up

Correct these rows in the move log and rerun. An unidentified move does not block review of unrelated catalog products; see `move-follow-up.csv`.

| Row reference | Move ID | Product ID as entered | Follow-up |
| --- | --- | --- | --- |

Moves are caught by the tool only when logged, tagged, or noted; for anything else, the H1 multi-area hold is the control.

## Owner review

Fill `owner-review-template.json` in a separate inputs folder. Answers bind exact movement, row times and quantities through fingerprints. Each required movement needs before/after answers; multi-area movements also need signed quantity allocations. Late entries need explicit confirmation. Move-log entries need both endpoint rows and inclusion decisions. Duplicate ticks and corrections need owner disposition. Each multi-area lot also needs confirmation of no unlogged moves; sheet follow-ups and coverage-without-rows reviews must be resolved.

Only the separate `apply_reset.py` can submit an approved reset. This report and generator have no production-write capability. Intentional historical posting (`backfill=true`) is included in the signed v3 plan so actual count times can be retained even on an older reviewed count. There is no automatic retry of an uncertain write.
