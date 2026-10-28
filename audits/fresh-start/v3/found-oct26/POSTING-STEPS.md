# Opening stock — posting steps (prepared 2026-10-27; run only after the #9 group and the passing proof)

Re-checked 2026-10-27 (read-only, last transaction 2870): the 8 opening-stock balances in
OPENING-STOCK-APPROVAL.md are unchanged.

## A. Before posting (Claude, read-only)
1. #9 group signed, applied by the owner, verified.
2. Whole-reset proof with all groups → `../whole-reset-oct26/proof-final/whole-reset-proof.json` must print
   **FULL RESET SCOPE RECONCILED**. Keep that report: a later proof rerun would list the found posts as
   outside the plan, by design.
3. If the #9 count leaves an opening-lot remainder (OPEN-107 / OPEN-283), add it to `found-rows.csv` and
   `opening-lot-notes.csv` (the rows SHA-256 then changes).
4. Fill `found-approval.DRAFT.md`: Rows SHA-256 (`shasum -a 256 found-rows.csv`), Whole-reset proof SHA-256
   (`shasum -a 256 ../whole-reset-oct26/proof-final/whole-reset-proof.json`), the exception ID list, date.

## B. Owner signs
5. ★ Michael signs: Signature + date in `found-approval.DRAFT.md`, rename it to `found-approval.md`.
   "Owner approves held entries: Y" approves the FL holds on the entries over 500 lb (six of the eight).

## C. Owner posts, one row per run (own Terminal)
Start with the smallest (OPEN-7, cinnamon, 20 lb) and check it, then the rest in this order:
OPEN-7, OPEN-40, OPEN-29, OPEN-73, OPEN-16, OPEN-30, OPEN-10, OPEN-12.

```bash
cd ~/dev/fl-countday-wt && DATABASE_URL="$(cat ~/.config/factory-ledger/db_url)" python3 -B audits/fresh-start/post_found_after_reset.py --proof audits/fresh-start/v3/whole-reset-oct26/proof-final/whole-reset-proof.json --rows audits/fresh-start/v3/found-oct26/found-rows.csv --opening-notes audits/fresh-start/v3/found-oct26/opening-lot-notes.csv --approval audits/fresh-start/v3/found-oct26/found-approval.md --target production --exception OPEN-7 --post
```

Each run: read-only checks (no earlier post names the ID, no lot collision) → shows the draft → waits for
"yes" → posts with the owner key, reason physical_count → approves its own over-500-lb hold if listed.
Change only `--exception` between runs. A row with any logged outcome is never posted twice; if a run says
"held" or "uncertain", stop and send the output to Claude.

## D. After posting (Claude, read-only)
6. `final_reconciliation.py` with the final manifest, the kept proof and `found-rows.csv`: every difference 0,
   nothing open → RESET COMPLETE.
7. Then, as ordinary Found inventory (not part of the reset): ground flax 50 lb and monk fruit sugar 55 lb
   found by Arturo on 2026-10-27 (FL 0; checked not counted elsewhere: no pending tickets, no found entries,
   not in the opening rows). Note: monk fruit was counted 55 lb on Oct 22 and FL then used 62 lb by recipe on
   Oct 23; the 55 lb still on the shelf suggests the #8 recipe overstates monk fruit.
8. ★ Michael confirms the reset is complete. The production freeze stays until he separately approves lifting it.
