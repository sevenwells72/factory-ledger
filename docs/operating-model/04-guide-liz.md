# D2. Liz (Luz) — Office daily guide

> **Name check.** Production has **no "Liz" account**. The office accounts are **"Luz" (office)** and **"Miriam" (office)**, and every document says Luz. This guide assumes Liz = Luz. If Liz is a different person, she needs her own account (an owner action) and must **not** share Luz's key.

**Primary tool: the dashboard on a desktop**, with these screens:
- **Sales Orders** tab;
- **Expected Receipts** tab;
- **Activity** tab;
- the **Office** screen, for changes.

**Secondary: FL Assistant**, for a quick order or customer from a phone message.
**Interim (today): the Office GPT** for writes the dashboard can't do yet.

Luz's account: **"Luz", role office**. It is active but **has never been used**.

## What the office role can and cannot do

| Can do | Cannot do (Arturo or Michael) |
|---|---|
| Create and change customers, sales orders and lines; close, cancel and reopen orders | Make, Pack, Adjust, Found |
| Expected receipts (create, edit, cancel; approve from a supplier document) | Void any entry |
| Receive ingredients (if the office logs a delivery) | Resolve or approve exceptions |
| All shipment types, including standalone | Kosher attestation, large-correction approval |
| Fix a supplier lot code; rename a lot | |

So when the office makes an **inventory** mistake, such as a wrong receive, Luz cannot void it herself. **Send the receipt number to Michael.** Mistakes on orders and expected receipts she fixes herself on the Office screen.

---

## Part 1 — Target workflow (after switch-on)

### Bookmarks (desktop browser)

- Dashboard: `https://cns-factory-ledger.netlify.app/`
- Office: `https://cns-factory-ledger.netlify.app/office.html`
- History: `https://cns-factory-ledger.netlify.app/history.html`
- Traceability: `https://cns-factory-ledger.netlify.app/traceability.html`

Sign in once per day with the **personal actor key**, which is stored in the browser's password manager. Sign out when leaving.

### Daily routine

| Time | Activity | Where | How / buttons |
|---|---|---|---|
| Morning | **Incoming orders** | Dashboard → **Sales Orders → Open** | New PO by email: **New Sales Order** → upload the PO PDF or photo → check the extracted lines → **approve**. This opens a draft in **Office**; press Record. Watch for **"duplicate customer PO"** warnings. Phone or text order: Office → Create order (or FL Assistant). |
| Morning | **Expected receipts** | Dashboard → **Expected Receipts** | Add each PO you placed with a supplier (Office → Create expected delivery, or upload the supplier confirmation and approve it). Overdue ones appear in Needs Attention. |
| On delivery | **Supplier paperwork** | Dashboard → **Activity → Receipts**, then open the receipt | Arturo records the receipt on the floor with a BOL photo. Check the receipt against the paper BOL and invoice: product, cases, lb, supplier lot. If wrong, send the receipt number to Arturo (same day) or Michael. Keep the paper filed by receipt number (RCV-…). |
| Anytime | **Inventory and lots** | Operations stock panels → click a product → lot detail; **Traceability** for "where did lot X go" | Read-only. |
| Midday | **Production progress** | Operations → **Today So Far** tile; **Activity** | Shows today's batches, pans and cases as Arturo records them. |
| Afternoon | **Prepare shipments** | Sales Orders → **Ready to ship** / **Overdue** → open the order → **Preview** (shipping capacity) | Preview shows whether stock covers each line. Set aside stock with **allocations** if needed. Mark ready on the Office screen. |
| Afternoon | **Packing slips / shipping documents** | ⚠ **No packing-slip button on the dashboard today.** | Interim: the Floor GPT returns a 10-minute slip link when a shipment is recorded. Proposed fix: a **Packing slip** button on the order (issue F-6). |
| Anytime | **Correct office errors** | **Office** → choose the action (update header, update / cancel line, cancel / reopen order, update / cancel expected delivery) | Every change is a draft → Record, with your name on it. |
| Daily | **Exceptions** | Dashboard → Operations → **Needs Attention** | Office can see exceptions but cannot resolve them. Chase Arturo or Michael on overdue ones. |
| Weekly | **QuickBooks reconciliation** | Dashboard → Sales Orders → **Shipped** → **Export CSV / Export Matrix**, compared against QuickBooks invoices | ⚠ **There is no QuickBooks connection.** QuickBooks is the price and invoice source of truth; Factory Ledger is the quantity and lot source of truth. Check every shipped order has an invoice, and every invoice a shipment. |
| End of day | **Office review** | **History** (filter today) and **Sales Orders** counts | Check every order shipped today shows Shipped, every new PO is entered, and expected receipts that arrived are matched. |

---

## Part 2 — Today (interim, until switch-on)

| Task | Use |
|---|---|
| New orders from a PO, order changes, customers | Dashboard **Sales Orders** tab (New Sales Order from PDF or photo; edit; close; cancel). It works today but records **no name**. Or the **Office GPT**. |
| Expected receipts | Dashboard **Expected Receipts** tab (works today). |
| Review everything | Dashboard tabs (all work today). |
| Shipping / packing slips | Arturo ships through the Floor GPT; open the 10-minute slip link from his GPT chat, or ask him to forward it immediately. |

**Rule for the interim:** pick **one** place per kind of work. Orders go on the dashboard tab, so the GPT is only for when the dashboard can't do it. Never enter the same order in both.

**Office GPT warnings:**
- It **creates orders straight from an uploaded PO without asking again**. Check the order on the dashboard afterwards.
- It can adjust inventory, which the office role normally cannot. **Don't use it for adjustments.**
