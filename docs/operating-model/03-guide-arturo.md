# D1. Arturo — Plant Manager daily guide

**Primary tool: FL Assistant on your phone.**
**Backup: the Record entry screen on the dashboard** (same sign-in, same records).
**For looking things up: the dashboard.**

> **Two versions of this guide.**
> - **Part 1 (target)** is how Arturo works once Michael approves switching on personal sign-in and the FL Assistant in production.
> - **Part 2 (today / interim)** is how to work until then, using the Floor GPT.
>
> Do not mix the two: on any given day, record everything in one place.

Arturo's account in production: **"Arturo", role floor**. It is active but **has never been used**.

## What the floor role can and cannot do

| Can do | Cannot do (ask Luz or Michael) |
|---|---|
| Receive, Make (granola and coconut), Pack, Ship against a sales order (or several), Adjust, Found | Create or change sales orders or customers |
| Mark an order ready to ship | Ship without a sales order (standalone) |
| Void **your own** entry **on the same plant day** | Void someone else's entry, or an older one |
| Fix a supplier lot code; rename a lot | Approve large corrections (owner only) |
| Resolve shortage and unidentified-lot exceptions | Backdate more than 14 days |

---

## Part 1 — Target workflow (after switch-on)

### Bookmarks on the phone (home-screen icons)

1. **FL Assistant:** `https://cns-factory-ledger.netlify.app/fl-assistant.html`
2. **Dashboard:** `https://cns-factory-ledger.netlify.app/`
3. **Shift Summary:** `https://cns-factory-ledger.netlify.app/shift-summary.html`

Every link must start with exactly `cns-factory-ledger.netlify.app`. **Any other address that ends in `netlify.app` is the test system.**

### Signing in, once per shift

1. Open FL Assistant.
2. Paste your **personal actor key** once (keep it in the phone's password manager, never in WhatsApp).
3. You stay signed in until you have been idle for 3 hours, or until you press Sign out.
4. On a shared device, always **Sign out** at the end of the shift.

### Beginning of day (about 10 minutes)

| Step | Where | How |
|---|---|---|
| Review today's production plan | **Dashboard → Runs** (if Michael or Luz plan runs there), otherwise the plan you get from Michael | ⚠ Factory Ledger holds no production plan today (the schedule is empty and the Runs page has never been used). Michael should decide whether the plan lives in **Runs**; see [07](07-issues-and-recommendations.md) F-12. |
| Check ingredients | **Dashboard → Supplies** (ingredients and packaging on hand), or ask the Assistant: "¿Cuánta avena tenemos?" | The dashboard gives the full list; the Assistant is quicker for one item |
| Pending orders and shortages | **Dashboard → Sales Orders → "Ready to ship" and "Overdue"**; **Operations → Needs Attention** | Needs Attention shows overdue orders, blocked dispatch, low stock and supply requests |
| Assign work | Outside Factory Ledger (in person) | Factory Ledger does not assign tasks |

### During production: one rule, record it when it happens

Everything below is done in **FL Assistant**. You describe what happened, the Assistant shows a **draft card**, you check it, tick any warnings, type the lot-tag codes it asks for, then press **Record**. You get a receipt number (RCV-…, MK-…, PK-…, SHP-…, ADJ-…, FND-…). **If there is no receipt number, it was not recorded.**

| Action | What to say / do | Notes |
|---|---|---|
| **Ingredients received** | "Recibí 40 cajas de avena de Quali Pack, BOL 12345, lote del proveedor 26-0915." Take a photo of the BOL and the lot tag. | The **supplier lot is required**. If the label has none, say "no lot on label". It is still recorded, but it opens a follow-up to identify the lot within 7 days. Write the **FL lot code** shown on the receipt on the pallet tag. |
| **Granola batch** | "Hicimos 3 batches de #8 Regular." | The Assistant offers the recipe's ingredient lots, oldest first. Confirm each physical lot tag you actually used: **this is how ingredient consumption is recorded**. Nothing is recorded separately. |
| **Coconut batch** | "Hicimos 4 pans de coco endulzado fancy." | Same "make" action with the coconut product. |
| **Packing** | "Empacamos 120 cajas de Granola Setton 25 LB del batch de hoy." | Pick the source batch from the buttons (oldest first) or tap "Different batch". |
| **Shipping** | "Enviamos la orden SO-261027-002 completa, BOL 7788." Photo of the BOL and the signed slip. | The BOL is required. If you are not sure the order is ready, check the dashboard first. The packing-slip link is not yet shown in the Assistant (issue F-6): print slips from the office. |
| **Discrepancy: more than the system says** | "Encontré 50 lb de linaza sin registrar." → **Found** | Add a photo. A Found entry with no lot code opens an "unidentified lot" follow-up. |
| **Discrepancy: less than the system says** | "Ajuste: el lote X tiene 30 lb menos, conteo físico." → **Adjust** with reason *physical count* | **Over 500 lb, or over 10% of a lot:** attach a photo, or it waits for Michael's approval ("Waiting for owner approval"). |
| **Mistake you just made today** | Type "anular RCV-261028-003". The Assistant opens the **Receipt** page. There, press **Prepare void for entry #…**, then Record. Then record the correct entry. | Only your own entries, same day. Otherwise tell Michael with the receipt number. |

**When to switch to the Record entry form instead of the Assistant:**
- the Assistant is down (it shows an "offline" card with a link to Record entry);
- it keeps misunderstanding you;
- you are entering a long list of identical items.

Same sign-in, same rules, same receipts.

**If you get "possible duplicate":** stop. Open "Today's entries" and check. If it really is a second delivery or batch, tick the warning and Record. If not, cancel.

### End of day (about 10 minutes)

| Step | Where | How |
|---|---|---|
| Confirm everything was recorded | **FL Assistant → "Today's entries"** (or Dashboard → Receipt → Today) | Compare with the BOLs and batch sheets in your hand. Anything missing: record it now. Entries 2–14 days late open an exception for Michael. |
| Check inventory accuracy | **Dashboard → Operations** stock panels for what you used today | If something looks wrong, count it and enter Found or Adjust with a photo. |
| Unfinished work | **Dashboard → Sales Orders → Ready to ship** | Orders that did not ship today stay there. |
| Exceptions | **Dashboard → Exceptions** (button on the tool bar) | Shortages and unidentified lots are assigned to you by default. Resolve them, or write in "Confirm with notes" why you can't. |
| Close the day | **Shift Summary → pick today → Confirm** (or **Confirm with notes**, or **Something is missing**) | This is the daily sign-off Michael reviews. |

---

## Part 2 — Today (interim, until switch-on)

Use the **Floor GPT ("Factory Ledger - Floor 2.0")** for all entries. Look things up on the dashboard.

| Action | Floor GPT |
|---|---|
| Receive, make, pack, adjust | Describe it. The GPT shows a **preview**. Say "sí, registrar" only if the preview is right. A real entry always ends with a **receipt number**; if there is none, it did not happen. |
| Ship an order | The GPT previews (`shipOrder`). You approve explicitly. It records (`commitShipOrder`) and returns a **packing-slip link valid for 10 minutes**. Open it right away. |
| Found inventory | **Not possible in the GPT.** Write it down and send it to Michael. |
| Void | Possible, using the transaction ID from History. Confirm with Michael first while the GPT has no role limits. |
| End of day | Ask "resumen del día"; compare with your paperwork. |

**Interim warnings:**
- Your entries are recorded as "legacy-shared-key", not as Arturo. Keep your paperwork; it is the only proof of who did what.
- Do not ask the Office GPT, Luz or Michael to record the same entry "just in case". There is no duplicate check between GPT entries except for order shipments.
