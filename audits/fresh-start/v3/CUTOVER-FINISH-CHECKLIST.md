# Finishing the inventory reset — checklist (2026-10-26)

★ = needs Michael. Everything else is prepared and can run as soon as the ★ step before it is done.

## Done
- [x] Groups 01, 02, final group 1 applied (Oct 22); verified read-only.
- [x] 285 / 286 signed as their own zero-stock group (Oct 26); dry run 0 posts.
- [x] Whole-reset check fixed for lots made after the count (Oct 26, owner-approved).
- [x] Opening-stock sheet, Arturo questions, small-group builder ready.
- [x] Small group B (11 products, −12,397.5 lb) approved by owner and applied 2026-10-27 (ADJ-261027-001…010 = TX2861–2870, 7 large corrections approved); verified read-only.
- [x] Whole-reset check after small group: only #9 granola (107 / 283) open. Pre-posting reconciliation: 0 differences, 8 opening rows not yet posted.

## To finish the reset
1. ~~Send Arturo the questions~~ done. Remaining: ★ **#9 granola** — Arturo weighs the one and a half containers on the new scale and says regular #9 or SS #9 (107 / 283), then its own small group (steps 2–4).
   Original: **Send Arturo the questions** — `count-inputs/oct26-small-group/QUESTIONS-FOR-ARTURO.txt` (R1–R9).
2. Enter his answers in `arturo-answers.csv` → `build_sg.py` → read-only preview → plain-English summary.
3. ★ **Approve the small group** (K&K, sprinkles, #9/#8/#7/BS Dark Choc batches, white choc chips, 171).
   Confirm "empty" products; approve any change over 500 lb.
   Product 171: if Arturo finds the 5 lb, nothing changes; otherwise ★ owner runs the switch-on / switch-off
   steps (`count-inputs/oct22-small-group-prep/README.md`).
4. Dry run → ★ **owner applies** the small group (owner's own terminal) → read-only verify.
5. Whole-reset proof (`whole-reset-oct26/`) must print **FULL RESET SCOPE RECONCILED**. Keep the report.
6. ★ **Sign the opening stock** — `found-oct26/OPENING-STOCK-APPROVAL.md` (8 products; fill proof SHA, date, signature)
   plus any opening lots from the small group (`build/found-after-reset.csv`).
7. ★ **FB Toasted Flake** — create the product (or decide not to), then its 6,650 lb (lots 6125/6126) follow step 8.
8. Post opening stock one row at a time (`post_found_after_reset.py`, shows each draft, waits for "yes").
9. Final reconciliation (`final_reconciliation.py`): every difference 0, no open exception.
10. ★ **Confirm the whole-reset proof passed** → production freeze lifts.

## After the freeze lifts (each ★)
11. Deploy #142 (already on main) with #145, then #146; read-only production checks.
12. PR #153 (duplicate-shipment guard): follow its "Deploy order" — staging rehearsal, migration 088 applied and
    verified first, then code. Required before staff ship through the GPT.
13. Yield adjustments: make a short note **required** when the reason is processing yield correction
    (`hydration_yield`), so it always shows in the daily report (owner decision 2026-10-27; build after the freeze).
14. API-key rotation about two weeks after stable operation.

## Things to know
- Arturo found ground flax 50 lb and monk fruit 55 lb on 2026-10-27 that FL shows as 0. Recommended: enter them as an ordinary
  Found entry after the opening stock is posted (the opening-stock tool only takes the Oct 21–22 count wording).
- Owner rule 2026-10-27: Arturo may make same-day yield adjustments so reports close the same day, using the
  processing yield correction reason and a short note; the owner reviews them in the daily report.
- Five Oct 23 top-ups (monk fruit 7, SF chips 58, flax 28, #8 batch 27.5, coconut batch 40 lb) are ordinary
  adjustments; they do not block the proof. Arturo R8/R9 explain them.
- Production keeps running normally meanwhile; each rebuild uses fresh read-only balances.
