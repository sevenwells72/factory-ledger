# Sunshine pouch ownership reconciliation - preparation only

**Plain-English summary:** Count every physical Sunshine pouch case normally. FL's shipment records and invoice references can help the office reconcile ownership afterward, but **they do not prove who owns each pallet still at CNS**. The new physical count and the owner's QuickBooks invoice detail have not been supplied yet, so no counted pallet has been classified or changed. This is a read-only matching plan with FL evidence.

Snapshot: 2026-10-06T18:37:34.872652+00:00. The five pouch SKUs below are 12 x 10 oz cases (7.5 lb/case). Keep this office report away from blind counters.

| Product | SKU | Name | FL lb | FL case equivalent | Positive lots |
| --- | --- | --- | --- | --- | --- |
| 145 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | 30345 | 4046 | 7 |
| 146 | 70002 | Granola SS Original 12x10 OZ Case | 8032.5 | 1071 | 3 |
| 147 | 70011 | Granola SS Cranberry 12x10 OZ Case | 2025 | 270 | 2 |
| 148 | 70070 | Granola SS Chocolate Chip Low Carb 12x10 OZ Case | 1365 | 182 | 1 |
| 149 | 70010 | Granola SS Original Low Carb 12x10 OZ Case | 2032.5 | 271 | 1 |

## What FL actually gives us

The scope snapshot contains only three orders with these pouch products:

| Order | Date | Legacy status | Current state / reason | Notes |
| --- | --- | --- | --- | --- |
| SO-260909-001 | 2026-09-08 | shipped | open / None |  |
| SO-260814-002 | 2026-08-14 | shipped | closed / shipped_recorded | PO 81326 |
| SO-260908-005 | 2026-09-08 | cancelled | cancelled / other | PO: 09062026 |

SO-260909-001 has a legacy `shipped` status while its newer state is `open`; that disagreement needs review, not an ownership assumption. Posted ships TX2535/2536 explicitly cite that order. The only pouch allocation in the snapshot is a **released 100-lb, unpinned allocation** for product 145 / SO-260814-002; it does not reserve or identify a remaining physical lot.

Historical reconciliation ships TX1960-1963 mention billed quantities and invoice references. TX1960 lists invoices 28259, 28260, 28263, 28266, 28268, 28265, 28183, 28181, 28276, 28277, 28323, 28295, 28297, 28320, 28321, 28322 and 28257. Those notes are **aggregate retrospective references**, not invoice-line-to-pallet proof. Keep them as candidate matches and prevent reuse of the same invoice quantity. Do not undo a historical reconciliation merely because a pallet is now found.

`sunshine-ship-evidence.csv` contains every pouch shipment line, exact lot, date, status, pounds/cases, customer/order reference and notes. `sunshine-order-evidence.csv` contains the matching order lines, including cancelled records. The inspected live schema has no dedicated invoice number or invoiced-owner field. There are no current location balances or ownership balances.

## Office matching after the count

1. Start with each counted SKU + exact physical lot + location + row time and any **physical marker text** the counter copied. Do not infer ownership from the Sunshine product name or from a bare “sold” label alone.
2. Match posted FL shipments to order lines and order references, then match the owner's QuickBooks export by invoice number/date/customer/SKU/quantity. Retain voids, credits, returns, partials and cancellations. FL ship pounds and invoice cases must be converted consistently (7.5 lb/case for these five pouch SKUs).
3. Obtain actual dispatch/delivery evidence. An invoice can precede shipment; a retrospective FL ship can be an accounting reconciliation. Neither by itself proves physical departure or ownership transfer.
4. Propose **CNS-owned**, **Sunshine-owned held at CNS**, or **unresolved** only with the owner's ownership terms and a supported quantity allocation to the counted lot/location. A physical stack may need a split; do not double count it. No automatic FIFO invoice matching or relabeling.
5. Compare physical custody and CNS-owned book stock separately. If customer-owned cases were already removed through a posted ship but remain onsite, an ordinary count reset would wrongly add them back to CNS-owned stock. **Hold Sunshine product adjustments until this office reconciliation is complete and supported stock treatment exists.** Keep customer-owned physical quantities on the count and on an exception list; do not silently subtract them from the original count.

## Missing inputs

- Completed v3 count rows and times, locations, tag IDs and physical marker notes.
- Owner-supplied QuickBooks invoice and credit detail (number, dates, customer, item, cases/lb, price, order/PO links, void/credit status). No QuickBooks/API lookup was made.
- Ownership-transfer terms and delivery/BOL/pickup evidence, including billed stock intentionally held at CNS.
- Evidence allocating aggregate invoice/reconciliation quantities to actual physical lots and pallets; unallocated amounts remain unresolved.
- A supported way to represent Sunshine-owned stock held at CNS (future backlog). The generic reset must not overwrite this distinction.
