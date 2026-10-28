# D3. Michael — Owner guide

**Primary tool: the dashboard.** You use these parts of it:
- **Operations → Needs Attention**;
- **Exceptions**, for approvals;
- **History**;
- **Traceability**;
- **Shift Summary**.

**Secondary:** the Office GPT, and later FL Assistant or the read-only connector, for quick questions ("how much coconut flake is on hand by lot?").
**Aim: no routine entries.** Your entries should be approvals, decisions and the rare owner-only correction.

Your production account is **"Blubber", role owner**. It is the only account that has been used, and it was used from engineering sessions, not from the dashboard.

> **Today, the Exceptions page is hidden in production** (personal sign-in is off). That is why your approvals for Oct 22–27 (58 approvals and 7 resolutions) were done through Claude sessions using your owner key. After switch-on, you do them yourself, on the page.

---

## Daily (about 15 minutes, morning)

| Purpose | Where | What to look at |
|---|---|---|
| **Operational overview** | Dashboard → **Operations** → **Needs Attention** chips | Overdue expected receipts, overdue orders, blocked dispatch, low stock, supply requests, refresh failures |
| **Production status** | Operations → **Today So Far**; **Production Calendar** | Batches, pans and cases recorded; compare with what you expected |
| **Yesterday's sign-off** | **Shift Summary** → yesterday | Did Arturo press **Confirm**? Read every "Confirm with notes" and "Something is missing" |
| **Exceptions / approvals** | **Exceptions** page | Large corrections (over 500 lb with no photo) waiting for you; late entries; overdue shortages and unidentified lots. Approve or reject each one. Approving produces an XR receipt. |
| **Outstanding orders** | **Sales Orders → Overdue / Ready to ship** | Orders at risk |
| **Shortages** | Needs Attention → low stock; Exceptions → SHORTAGE | Ingredients short for planned work |
| **Unidentified supplier lots** | Exceptions → UNIDENTIFIED_LOT | Each has a 7-business-day deadline. Today 2 are open: flax (66) and monk fruit (67), from the Oct 28 Found entries. |
| **Inventory accuracy** | Operations → **ledger health checks** (integrity panel) | Any red check is a data problem; ask for it to be investigated |

## Weekly (about 45 minutes)

| Purpose | Where | What |
|---|---|---|
| **Employee activity and accountability** | **History**, filtered by week; each entry shows who entered it | After switch-on, every entry has a name. Look for entries by "legacy-shared-key": after the cutover there should be **none**. |
| **Weekly operations report** | History (Production, Packing, Shipping) + Sales Orders → Shipped + **Export Matrix** | Volume made, packed and shipped |
| **Production efficiency** | **Runs** page (if runs are planned there); Production Calendar; batch yields compared to the recipe | Yields and the #8 monk-fruit recipe question are listed in `UNVERIFIED-BALANCES.md` |
| **Recurring problems** | Exceptions page, resolved and waived, by kind | Same product short again and again, same supplier with no lot code, many late entries by one person |
| **Traceability spot check** | **Traceability** → pick one finished lot → trace back to supplier lots | Should resolve without gaps. Note: Coconut Flake FOUND-011 is a **pooled** lot (159,250 lb from at least 10 supplier lots). A recall on any of those supplier lots covers the whole pool and lot 6013. |
| **QuickBooks check** | With Luz | Shipped orders vs invoices |

## What should reach you automatically

**Built and working:** these become owner exceptions and appear on the Exceptions page.
- Any correction over 500 lb **with no photo** (a block: it waits for you).
- Any entry 2–14 days late, made by anyone but you.
- Kosher attestations for extra-kosher production.
- Shortages and unidentified lots that pass their deadline (they show as overdue).
- Assistant outage (an `ASSISTANT_OFFLINE` exception to you).

**Gaps:**
- **There is no push notification.** You must open the page; nothing is emailed or texted.
- **Corrections made through the GPTs never reach you**, because the direct route skips the approval rule. That is a reason to make the GPTs read-only.
- Over 500 lb **with** a photo posts without your approval and is only highlighted. Decide whether that is acceptable.

## Owner-only actions

- Approve or reject held corrections.
- Kosher attestation.
- Void anyone's entry.
- Backdate more than 14 days.
- Close an order as "shipped, not recorded".
- Waive or write off an exception.

Do them on the **Exceptions** and **Receipt** pages once they are visible. Until then, they need an engineering session.

## Decisions waiting for you now

1. **Switch-on of personal sign-in and FL Assistant in production.** See [07](07-issues-and-recommendations.md) §1.
2. **#9 Regular:** Factory Ledger shows 7,883 lb; Arturo counted 1,682. Decide the lot split (container labels or oldest first).
3. **Confirm** that "Almond Vanilla" = product 113 and "PB and Banana Mix" = product 122 (Found 475 lb and 1,120 lb).
4. **Is the absence of receive, make, pack and ship entries since Oct 23 real?** If work happened, it needs to be entered (late-entry rules will apply).
5. **Master-key rotation** (it has been exposed) and the read-only GPT cutover date.
