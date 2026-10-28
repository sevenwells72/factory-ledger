# G. Standard operating procedures — Factory Ledger

These SOPs apply from the **switch-on date** (see [07](07-issues-and-recommendations.md) §1). Before that date, follow the "Today (interim)" parts of each guide.

## SOP-1: One person, one account, one primary tool

| Person | Account | Primary | Allowed backup |
|---|---|---|---|
| Arturo | Arturo (floor) | FL Assistant | Record entry form |
| Luz (Liz) | Luz (office), or her own if Liz is a different person | Dashboard + Office screen | FL Assistant |
| Michael | Blubber (owner) | Dashboard | Read-only chat |

- **Never share keys.** Never record under someone else's sign-in.
- Keys live only in a password manager.
- Sign out on shared devices.

## SOP-2: Who records what

Each kind of entry has **one owner**. Nobody else records it, so nothing is recorded twice.

| Entry | Records it | Checks it |
|---|---|---|
| Ingredient / supply receipt | **Arturo**, at the dock, with BOL and lot-tag photos | Luz, against the paperwork |
| Granola / coconut batch (make) | **Arturo** | Michael (daily review) |
| Packing | **Arturo** | Michael |
| Shipment | **Arturo**, against the sales order, with BOL and signed-slip photos | Luz (order shows Shipped, invoice raised) |
| Found / Adjust (discrepancies, counts) | **Arturo** | Michael approves anything held |
| Sales orders, customers, expected receipts | **Luz** | Michael (weekly) |
| Voids | Same-day own mistakes: **the person who made them**. Everything else: **Michael**. | — |
| Approvals, kosher attestations, write-offs | **Michael** only | — |

If the owner of an entry is absent, Michael names a stand-in for the day. The stand-in uses **their own** account.

## SOP-3: Record at the time it happens

- Record within the hour, and always the same day.
- Entries 2–14 days late automatically open an exception for Michael. Entries more than 14 days late need Michael.
- **A receipt number (RCV-, MK-, PK-, SHP-, ADJ-, FND-…) is the only proof something was recorded.** No number means it did not happen: check "Today's entries" before trying again.

## SOP-4: Read the draft before Record

- Check the product, quantity, lot and supplier lot.
- Type the physical lot-tag codes when asked; never copy them from the screen.
- **"Possible duplicate"** warning: stop, look at "Today's entries", and only tick and Record if it really is a second event.

## SOP-5: Supplier lots and traceability

- Every ingredient receipt carries the **supplier lot code** from the label. If there is genuinely none, choose **"No lot on label"** and photograph the label.
- Write the FL lot code from the receipt on the pallet tag.
- An unidentified lot must be identified within 7 business days, on the Exceptions page. Arturo owns these.
- Never assign a supplier lot to a **pooled** lot. The system refuses this anyway.

## SOP-6: Discrepancies and corrections

- **More than expected:** Found, with a photo.
- **Less than expected:** Adjust, with reason "physical count" (or the true reason) and a photo.
- Corrections over 500 lb, or over 10% of a lot, need a photo. Without one they wait for Michael.
- **Never** "fix" stock by recording a fake receipt, batch or shipment.
- An open shortage must be resolved on the Exceptions page, not covered with a Found entry. The system blocks that anyway.

## SOP-7: Daily close

1. **Arturo, end of shift:**
   - check "Today's entries" against the paperwork;
   - resolve or note his exceptions;
   - **Shift Summary → Confirm** (or **Confirm with notes**, or **Something is missing**).
2. **Luz, end of day:**
   - every shipped order shows Shipped;
   - every new PO is entered;
   - arrived deliveries are matched to expected receipts.
3. **Michael, next morning (15 minutes):**
   - Needs Attention;
   - yesterday's Shift Summary;
   - the Exceptions queue.

## SOP-8: Weekly review (Michael, about 45 minutes)

- History by person, looking for any unnamed entries;
- recurring exception kinds;
- shipped orders vs QuickBooks invoices (with Luz);
- one traceability spot check;
- yields vs recipe.

## SOP-9: Production vs staging

- Work only at **`cns-factory-ledger.netlify.app`**.
- Any other address is the test system: never record real work there.
- If in doubt, look for "STAGING" names in the data, and ask Michael.

## SOP-10: Chat tools

- **The FL Assistant (or the Record entry form) is the normal way to write.** The Floor and Office GPTs stay available as **fallbacks** (Michael, 2026-10-28): use one for an entry only when the new path fails, and never record the same event in both. Making the GPTs read-only is a later, separate decision.
- Never paste a key, password or PIN into any chat.
- **If the Assistant is down:**
  1. use the Record entry form;
  2. if Factory Ledger itself is down, write it on paper with the time;
  3. enter it the same day once it's back.

## SOP-11: Changes to the system

- Any change to production settings, GPT configuration, accounts or code needs Michael's explicit approval for that step.
- New accounts and deactivations are owner actions.
- When someone leaves, deactivate their account the same day.
