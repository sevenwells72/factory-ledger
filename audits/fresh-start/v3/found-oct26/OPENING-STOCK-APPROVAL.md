# Opening stock — owner approval sheet (prepared 2026-10-26, nothing posted)

These are the counted amounts that could not be matched to an existing FL lot (or were left over after
matching, rules rev 6). Each becomes ONE new "opening stock" lot per product, posted as Found inventory
only **after** the whole-reset proof says FULL RESET SCOPE RECONCILED.

| ID | Product | Opening lot (lb) | FL today (lb) | FL after posting (lb) | Over 500 lb? |
| --- | --- | ---: | ---: | ---: | --- |
| OPEN-12 | Coconut Flake Desiccated | 159,250 | 6,900 | 166,150 | yes |
| OPEN-10 | Coconut Fancy Desiccated | 21,800 | 12,600 | 34,400 | yes |
| OPEN-30 | Honey | 1,558.2 | 1,741.8 | 3,300 | yes |
| OPEN-16 | Corn Starch | 1,365 | 5,105 | 6,470 | yes |
| OPEN-73 | Chocolate Chips – Real – 1,000 CT | 1,250 | 0 | 1,250 | yes |
| OPEN-29 | Glycol | 972 | 8,586 | 9,558 | yes |
| OPEN-40 | Sugar – 6X | 152.49 | 51,687.51 | 51,840 | no |
| OPEN-7 | Cinnamon Ground | 20 | 147 | 167 | no |

"FL today" is the read-only balance at 2026-10-26 ~13:00 ET; it already includes Oct 23 production
(coconut flake −1,200, corn starch −30, glycol −42, sugar 6X −660), so "after posting" is lower than the
counted total by exactly that usage.

Each lot's note lists the receipts it may have come from (`opening-lot-notes.csv`).

**Not on this sheet:** FB Toasted Flake 3,700 lb (lot 6125) and 2,950 lb (lot 6126). The product must be
created first; see `found-rows-new-product.PENDING.csv`.

## To approve
1. Owner decision: approve these 8 entries (including the 6 over 500 lb).
2. After the proof passes, fill in `found-approval.DRAFT.md` (proof SHA-256, date, signature) and rename it
   `found-approval.md`. Rows file SHA-256 is already filled; any edit to `found-rows.csv` invalidates it.
3. Post one row at a time with `post_found_after_reset.py --approval found-approval.md --opening-notes opening-lot-notes.csv --post`
   (shows each draft and waits for "yes").
