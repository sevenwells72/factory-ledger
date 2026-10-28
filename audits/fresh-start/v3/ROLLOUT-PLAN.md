# Fastest safe path to go live — plan for owner approval (2026-10-27)

Nothing in this plan has been executed. ★ = owner approval needed. All checks stay on; nothing is bypassed.

## Can FL take later corrections without losing history? Yes (checked in code and in production)
- `transactions` and `transaction_lines` cannot be edited or deleted: database triggers
  `trg_transactions_original_append_only` / `trg_transaction_lines_original_append_only` are live in production.
- A correction is always a new record: an adjustment (reason `count_correction`, `damage`, `hydration_yield`, …) or
  Found entry on the exact product and lot, or a void/amend correction that keeps the original and stores the
  previous and replacement values, reason, operator and named actor.
- Every post has a ticket, a receipt number (ADJ-/FND-) and the named actor; notes can cite the original receipt.

## No technical or data-integrity reason to wait
Only routine items: PR #145 needs a rebase onto main (conflicts with the Oct 23 dashboard change); the Floor GPT
contract for #145 must be confirmed pasted before its backend deploy; a whole-reset proof re-run after the
opening-stock posts lists them as "outside the plan" by design (we keep the pre-posting report).

## Step 1 — #9 granola accepted provisionally (★ A1)
- One zero-change group for #9 Classic (107: 7,883 lb in 6 lots) and SS Extra-Kosher #9 (283: 0 lb):
  each FL lot "counted" at its current FL balance, so **nothing is posted**.
- Labelled honestly in the signed inputs: counter "Owner — FL balance accepted, NOT physically counted",
  every row and coverage note "PROVISIONAL", and a written exception in the whole-reset manifest notes.
- Listed in `UNVERIFIED-BALANCES.md` for Arturo's later weigh-in.
- Read-only preview → ★ owner signs → dry run (0 posts). It is a no-change group: no apply step.

## Step 2 — Whole-reset proof (Claude, read-only)
Must print FULL RESET SCOPE RECONCILED with all 7 groups. Kept as the official pre-posting report.

## Step 3 — Opening stock (★ A2 sign, ★ A3 each "yes")
`found-oct26/POSTING-STEPS.md`: 8 entries (159,250 lb coconut flake, 21,800 lb fancy coconut, honey, corn starch,
chocolate chips, glycol, sugar 6X, cinnamon). Owner signs once, then posts one per run, OPEN-7 first.

## Step 4 — Final reconciliation (Claude, read-only)
Every difference 0, nothing open → ★ A4 owner confirms the reset is complete.

## Step 5 — Lift the freeze and roll out, one step at a time (★ A5 lift; ★ each deploy)
1. Redeploy `main` → ships #142 (pilot screens, merged Oct 20). Verify health, deployment SUCCESS, screens.
2. #145 (supplier lot on receipts): rebase, full tests, confirm Floor GPT contract pasted → merge → verify.
3. #146 (signed packing-slip links): retarget to main → merge → verify.
4. #153 (duplicate-shipment guard): its Deploy order — staging rehearsal, migration 088 applied and verified
   first, then merge → verify. Required before staff ship through the GPT.
5. Then, as separate decisions: #147, #148, #144, #143, #141; yield-note requirement; key rotation (~2 weeks).

## Step 6 — Later findings as ordinary corrections
- Arturo's #9 weights → `count_correction` adjustment per lot (or Found for an unknown lot), note citing
  "provisional #9, reset group <name>"; history keeps both the provisional balance and the correction.
- Ground flax 50 lb and monk fruit 55 lb → Found entries right after Step 3 (already agreed; checked not
  counted elsewhere).
