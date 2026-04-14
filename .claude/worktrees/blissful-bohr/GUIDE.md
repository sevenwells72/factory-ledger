# Factory Ledger — Dummy Guide (v2.5.0)

How to use each workflow, with real-world examples from the factory floor.

---

## 1. RECEIVE — Incoming Goods

**When:** A shipment arrives at the dock.

**What you tell the GPT:**
```
Received 40 cases of Organic Coconut Oil 35 LB from Caribbean Imports, BOL# CI-20260209
```

**What happens:**
- Finds the product "Organic Coconut Oil 35 LB"
- Calculates total: 40 cases × 35 lb = 1,400 lb
- Generates a lot code: `26-02-09-CARI-001`
- Shows you a preview → you confirm → inventory goes up by 1,400 lb

**Result:**
```
📥 Received 1,400 lb Organic Coconut Oil 35 LB
Lot: 26-02-09-CARI-001
Shipper: Caribbean Imports
BOL: CI-20260209
```

---

## 2. SHIP (Standalone) — Ship Without a Sales Order

**When:** Someone says "send 500 lb of granola to customer X" and there's no formal order in the system.

**What you tell the GPT:**
```
Ship 500 lb Classic Granola 25 LB to Tropical Foods
```

**What happens:**
- Finds the product and checks lot inventory (oldest lot first — FIFO)
- If the customer has open sales orders, you'll see a warning: "⚠️ This customer has open orders — ship against the order instead?"
- Shows preview with lot and quantity → you confirm → inventory goes down

**Result:**
```
🚚 Shipped 500 lb Classic Granola 25 LB → Tropical Foods
Lot: 26-01-15-PROD-003
Remaining in lot: 200 lb
```

**Watch out for:** If the oldest lot doesn't have enough, it'll tell you. Use multi-lot ship instead.

---

## 3. MULTI-LOT SHIP — Ship Across Multiple Lots

**When:** You need to ship 2,000 lb but no single lot has that much. The system splits it across lots automatically.

**What you tell the GPT:**
```
Ship 2,000 lb Classic Granola 25 LB to Whole Foods, multi-lot
```

**What happens:**
- Finds all lots with stock, oldest first
- Allocates across lots: Lot A (800 lb) + Lot B (750 lb) + Lot C (450 lb) = 2,000 lb
- Shows the breakdown → you confirm → all lots deducted

**Result:**
```
🚚 Shipped 2,000 lb Classic Granola 25 LB → Whole Foods
  Lot 26-01-10-PROD-001: 800 lb (depleted)
  Lot 26-01-15-PROD-002: 750 lb (depleted)
  Lot 26-01-22-PROD-003: 450 lb (550 lb remaining)
```

---

## 4. MAKE — Production Run

**When:** The production team makes a batch of finished product from ingredients.

**What you tell the GPT:**
```
Make 2 batches of Classic Granola 25 LB
```

**What happens:**
- Looks up the batch formula (recipe) for Classic Granola
- Each batch = e.g. 500 lb, so 2 batches = 1,000 lb output
- Checks that all ingredients have enough stock (oats, honey, coconut oil, etc.)
- Shows ingredient requirements → you confirm → ingredients consumed, finished goods created

**Result:**
```
🧪 Produced 1,000 lb Classic Granola 25 LB
Output Lot: B26-0209-001
Ingredients consumed:
  - Rolled Oats: 400 lb from lot 26-02-01-GRAIN-001
  - Honey: 150 lb from lot 26-01-28-SWEET-002
  - Coconut Oil: 100 lb from lot 26-02-05-CARI-001
  - Almonds: 200 lb from lot 26-01-20-NUTS-003
  - Cinnamon: 50 lb from lot 26-02-03-SPICE-001
```

**Special options:**
- **Exclude an ingredient:** "Make 2 batches of Classic Granola, skip almonds"
- **Force a specific lot:** "Make 2 batches of Classic Granola, use lot 26-02-01-GRAIN-002 for oats"

**Auto-excluded ingredients:** Water and other utility ingredients are automatically excluded from production — they'll never block a run. You'll see them listed as "auto-excluded" in the preview.

---

## 4a. PACK — Internal Packing (Batch → Finished Good) / Empaque Interno

**When:** The production team has made a batch product (e.g., "Batch Classic Granola #9") and now needs to pack it into customer-ready finished-good cases (e.g., "CQ Granola 10 LB").

**This is different from MAKE.** Make = raw ingredients into batch product. Pack = batch product into finished-good cases.

**What you tell the GPT:**
```
Pack 140 cases of CQ Granola 10 LB from Batch Classic Granola #9, FIFO
```

**What happens:**
- Finds the batch product and checks available lot inventory (FIFO — oldest lots first)
- Calculates total: 140 cases × 10 lb = 1,400 lb needed
- Shows allocation preview → you confirm → batch inventory goes down, finished-good inventory goes up

**Result:**
```
📦 Packed 140 cases (1,400 lb) of CQ Granola 10 LB
Source: Batch Classic Granola #9
Lot consumed: FEB 06 2026 (1,400 lb)
Output lot: FEB 06 2026
```

**Split across lots:**
```
Pack 296 cases of CQ Coconut Sweetened Flake 10 LB from Batch Coconut Sweetened Flake:
- 16 cases from lot FEB 12 2026
- 280 cases from lot FEB 13 2026
```

**Result:**
```
📦 Packed 296 cases (2,960 lb) of CQ Coconut Sweetened Flake 10 LB
Lots consumed:
  - FEB 12 2026: 160 lb
  - FEB 13 2026: 2,800 lb
Output lot: FEB 12 2026
```

**Key points:**
- The finished-good lot inherits the lot code from the primary batch lot
- Full traceability: "Trace batch FEB 06 2026" will show it came from the batch product
- FIFO is the default — specify lots only when you need to control which batches are used

---

## 5. ADJUST — Fix Inventory Manually

**When:** You counted inventory and it doesn't match, or product was damaged/discarded.

**What you tell the GPT:**
```
Adjust lot 26-01-15-PROD-003 of Classic Granola down by 50 lb, reason: damaged in storage
```

**What happens:**
- Finds the product and lot
- Applies the adjustment directly (no preview step)
- Logs the reason for audit

**Result:**
```
⚖️ Adjusted -50 lb Classic Granola 25 LB
Lot: 26-01-15-PROD-003
Reason: damaged in storage
```

To add inventory (e.g., found extra on a shelf):
```
Adjust lot 26-01-15-PROD-003 of Classic Granola up by 25 lb, reason: found extra pallet
```

---

## 6. CREATE SALES ORDER — Customer Places an Order

**When:** A customer calls in with an order.

**What you tell the GPT:**
```
New order from Quali-Pack: 360 cases Classic Granola 25 LB at $4.50, ship by Feb 15
```

**What happens:**
- Finds (or creates) the customer "Quali-Pack"
- Converts cases to pounds: 360 cases × 25 lb = 9,000 lb
- Creates the order with a number like SO-260209-001

**Result:**
```
📦 Order Created: SO-260209-001
Customer: QUALI-PACK USA
1. Classic Granola 25 LB — 360 cases × 25 lb = 9,000 lb @ $4.50/lb
Ship by: Feb 15, 2026
Status: New
```

**Multi-line order:**
```
New order from Tropical Foods:
- 200 cases Classic Granola 25 LB at $4.50
- 100 cases Chocolate Granola 25 LB at $5.00
Ship by Feb 20
```

**Ordering in pounds (old way still works):**
```
New order from Tropical Foods: 5,000 lb Classic Granola at $4.50
```

---

## 7. LIST / VIEW ORDERS — Check Order Status

**When:** You want to see what's in the pipeline.

**What you tell the GPT:**
```
Show open orders
```
```
What orders does Quali-Pack have?
```
```
What's overdue?
```
```
Show me order SO-260209-001
```

**What you get back:**

For a list:
```
Open Orders:
1. SO-260206-003 | Quali-Pack USA | 360 lb ordered, 0 lb shipped | Status: New
2. SO-260209-001 | Quali-Pack USA | 9,000 lb ordered, 0 lb shipped | Status: New
3. SO-260208-001 | Tropical Foods | 5,000 lb ordered, 2,000 lb shipped | Status: Partial Ship
```

For a single order — full detail with lines, shipment history, and totals.

---

## 8. UPDATE ORDER STATUS — Move an Order Through the Pipeline

**When:** The order moves to the next stage.

**Status flow:**
```
📦 New → ✅ Confirmed → 🏭 In Production → 📋 Ready → 🚚 Shipped → 💰 Invoiced
```

**What you tell the GPT:**
```
Confirm order SO-260209-001
```
```
Mark SO-260209-001 as in production
```
```
SO-260209-001 is ready to ship
```

**Result:**
```
✅ SO-260209-001 updated to: Confirmed
```

---

## 9. MODIFY ORDER LINES — Add, Cancel, or Change Lines

**When:** The customer changes their mind.

**Add a line:**
```
Add 50 cases Chocolate Granola 25 LB at $5.00 to order SO-260209-001
```

**Cancel a line:**
```
Cancel the Chocolate Granola line on SO-260209-001
```

**Change quantity:**
```
Update SO-260209-001 Classic Granola to 10,000 lb
```

**Rules:**
- Can't modify lines that are already fulfilled (fully shipped)
- Can't modify cancelled lines
- Can't add lines to shipped/invoiced/cancelled orders

---

## 10. SHIP AGAINST ORDER — Fulfill a Sales Order

**When:** The order is ready and you're shipping it out. This ties the shipment to the order so balances update.

**Ship everything on the order:**
```
Ship order SO-260209-001
```

**Ship partial (only some quantity):**
```
Ship 5,000 lb of Classic Granola on SO-260209-001
```

**What happens:**
- Checks what's remaining on each order line
- Checks inventory availability for each product
- Uses FIFO lots (oldest first)
- Shows preview with what can ship → you confirm → inventory deducted, order updated

**Result:**
```
🚚 Shipped against SO-260209-001
1. Classic Granola 25 LB — 5,000 lb shipped
   Lots used: 26-02-09-PROD-001 (3,000 lb), 26-02-09-PROD-002 (2,000 lb)
   Remaining on order: 4,000 lb

Order status: Partial Ship
```

**Ship all remaining:**
```
Ship the rest of SO-260209-001
```

**Key difference from standalone ship:** This updates the order's shipped quantity and line status. Standalone ship does not touch orders at all (that's why Fix 2 warns you).

---

---
---

# Part 2: Lookups, Traceability & Admin (Items 11–20)

---

## 11. SEARCH PRODUCTS — Find a Product

**When:** You're not sure of the exact product name, or want to see what matches.

**What you tell the GPT:**
```
Search for coconut
```
```
What granola products do we have?
```
```
Find product OD-1234
```

**What happens:**
- Searches by name or Odoo code (fuzzy match — partial names work)
- Returns up to 20 matches, best matches first

**Result:**
```
Found 4 products matching "coconut":
1. Organic Coconut Oil 35 LB (raw_material)
2. Coconut Flakes 25 LB (raw_material)
3. Batch Coconut Granola #5 (batch)
4. Coconut Granola 25 LB (finished_good)
```

---

## 12. GET INVENTORY — Check Stock Levels

**When:** You need to know how much of something is on hand.

**What you tell the GPT:**
```
How much Classic Granola do we have?
```
```
What's in stock for coconut oil?
```
```
Inventory for all granola products
```

**What happens:**
- Finds matching products
- Adds up all lot quantities for each product

**Result:**
```
Classic Granola 25 LB: 3,200 lb on hand
  Lot 26-01-15-PROD-001: 800 lb
  Lot 26-02-01-PROD-002: 1,200 lb
  Lot 26-02-09-PROD-003: 1,200 lb
```

**For a whole category:**
```
How much granola do we have total?
```
The GPT will search all granola products (bulk + finished) and give you a grand total.

---

## 13. TRANSACTION HISTORY — What Happened Recently

**When:** You want to see recent activity — what was received, shipped, made, or adjusted.

**What you tell the GPT:**
```
Recent transactions
```
```
What shipped today?
```
```
Show receive transactions for coconut oil
```
```
Last 10 adjustments
```

**What happens:**
- Pulls transactions filtered by type and/or product
- Shows most recent first

**Result:**
```
Recent Transactions:
1. 🚚 Ship | Feb 9, 2:30 PM | 100 lb Classic Granola → Quali-Pack | Lot: 26-02-09-FOUND-004
2. 📥 Receive | Feb 9, 10:15 AM | 1,400 lb Coconut Oil | Lot: 26-02-09-CARI-001 | BOL: CI-20260209
3. 🧪 Make | Feb 8, 3:00 PM | 1,000 lb Classic Granola | Lot: B26-0208-001
4. ⚖️ Adjust | Feb 8, 11:00 AM | -50 lb Classic Granola | Lot: 26-01-15-PROD-003 | Reason: damaged
```

---

## 14. BATCH FORMULA — See a Recipe

**When:** You want to know what ingredients go into a batch product before making it.

**What you tell the GPT:**
```
What's the formula for Classic Granola?
```
```
What's in Batch Coconut Granola #5?
```
```
Show me the recipe for Chocolate Granola
```

**What happens:**
- Looks up the batch product's formula
- Shows each ingredient and how much per batch

**Result:**
```
Formula: Classic Granola 25 LB
Batch size: 500 lb

Ingredients per batch:
  - Rolled Oats: 200 lb
  - Honey: 75 lb
  - Coconut Oil: 50 lb
  - Almonds: 100 lb
  - Cinnamon: 25 lb
  - Vanilla Extract: 10 lb
  - Brown Sugar: 40 lb
```

**Useful before production:** Check the formula first, then decide if you need to exclude or override any ingredients.

---

## 15. TRACEABILITY — Track Lots Forward and Backward

### 15a. Trace a Batch (backward) — "What went into this?"

**When:** A customer complains about lot B26-0209-001 and you need to know what ingredients were used.

**What you tell the GPT:**
```
Trace batch B26-0209-001
```

**Result:**
```
Batch: B26-0209-001 — Classic Granola 25 LB
Produced: Feb 9, 2026 | Output: 1,000 lb

Ingredients used:
  - Rolled Oats: 400 lb from lot 26-02-01-GRAIN-001
  - Honey: 150 lb from lot 26-01-28-SWEET-002
  - Coconut Oil: 100 lb from lot 26-02-05-CARI-001
  - Almonds: 200 lb from lot 26-01-20-NUTS-003
  - Cinnamon: 50 lb from lot 26-02-03-SPICE-001
```

### 15b. Trace an Ingredient (forward) — "Where was this lot used?"

**When:** A supplier says their coconut oil lot had an issue. You need to know which batches it went into.

**What you tell the GPT:**
```
Where was lot 26-02-05-CARI-001 used?
```

**Result:**
```
Ingredient: Coconut Oil — Lot 26-02-05-CARI-001

Used in:
  1. B26-0209-001 — Classic Granola 25 LB (100 lb used)
  2. B26-0209-002 — Coconut Granola 25 LB (150 lb used)
  3. B26-0210-001 — Chocolate Granola 25 LB (80 lb used)
```

**This is critical for recalls:** You can trace from a bad ingredient to every finished product it touched.

---

## 16. CUSTOMERS — Manage Customer List

**When:** You need to add, find, or update a customer.

**List all customers:**
```
Show me all customers
```

**Search for one:**
```
Search customer Quali
```

**Add a new customer:**
```
Add customer Tropical Foods, contact: Maria Lopez, phone: 305-555-1234, email: maria@tropicalfoods.com
```

**Update a customer:**
```
Update Tropical Foods phone to 305-555-9999
```

**Result (new customer):**
```
✅ Customer created: Tropical Foods
Contact: Maria Lopez
Phone: 305-555-1234
Email: maria@tropicalfoods.com
```

**Note:** Customers are also auto-created when you create a sales order for a name that doesn't exist yet.

---

## 17. SALES DASHBOARD — Big Picture Overview

**When:** Start of the day, or you need a quick status check on all orders.

**What you tell the GPT:**
```
Sales dashboard
```
```
Order summary
```

**What happens:**
- Shows order counts by status
- Lists overdue orders
- Lists orders due this week
- Shows shipments from the last 7 days

**Result:**
```
📊 Sales Dashboard — Feb 9, 2026

Order Status:
  📦 New: 3
  ✅ Confirmed: 2
  🏭 In Production: 1
  📋 Ready: 1
  🚚 Partial Ship: 2

⚠️ Overdue (2):
  SO-260203-001 | Tropical Foods | Due Feb 7 | 2,000 lb remaining
  SO-260205-002 | Whole Foods | Due Feb 8 | 500 lb remaining

📅 Due This Week (3):
  SO-260209-001 | Quali-Pack USA | Due Feb 15 | 9,000 lb remaining
  SO-260208-001 | Tropical Foods | Due Feb 12 | 3,000 lb remaining
  SO-260207-003 | Fresh Market | Due Feb 14 | 1,500 lb remaining

🚚 Recent Shipments (7 days):
  SO-260209-001 | Quali-Pack USA | 100 lb shipped | Feb 9
  SO-260208-001 | Tropical Foods | 2,000 lb shipped | Feb 8
```

---

## 18. QUICK CREATE PRODUCT — Add a New Product On the Fly

**When:** You're receiving or making something and the product doesn't exist in the system yet.

**What you tell the GPT:**
```
Create new product: Organic Almond Butter 5 LB, type: finished_good
```
```
Quick create: Batch Dark Chocolate Granola #1, type: batch
```

**What happens:**
- Creates the product with status "unverified" (flagged for review)
- You can start using it immediately for receives, orders, etc.

**Result:**
```
✅ Product created: Organic Almond Butter 5 LB
Type: finished_good
Status: ⚠️ Unverified (needs review)
```

**Product types:** `raw_material`, `finished_good`, `batch`, `packaging`

**To see unverified products later:**
```
Show unverified products
```

---

## 19. FOUND INVENTORY — Log Unmarked Product You Discover

**When:** Someone finds pallets on the floor with no lot tag, or product that wasn't in the system.

**What you tell the GPT:**
```
Found 500 lb of Classic Granola in the back warehouse, unknown age, might be from Caribbean Imports
```

**What happens:**
- Creates a special "FOUND" lot code: `26-02-09-FOUND-001`
- Adds the inventory with metadata (location, estimated age, suspected source)
- Goes into the Found Inventory Queue for review

**Result:**
```
✅ Found inventory logged
Product: Classic Granola 25 LB
Quantity: 500 lb
Lot: 26-02-09-FOUND-001
Location: back warehouse
Estimated age: unknown
Suspected supplier: Caribbean Imports
```

**Check the found inventory queue:**
```
Show found inventory queue
```
This shows all found items that still have stock — useful for periodic review.

---

## 20. LOT REASSIGNMENT — Move a Lot to a Different Product

**When:** A lot was received under the wrong product name, or you need to reclassify it.

**Example:** Lot `26-02-09-CARI-001` was received as "Coconut Oil 35 LB" but it's actually "Organic Coconut Oil 35 LB" (different product in the system).

**What you tell the GPT:**
```
Reassign lot 26-02-09-CARI-001 to Organic Coconut Oil 35 LB, reason: received under wrong product name
```

**What happens:**
- Moves the lot from old product to new product
- Updates all transaction records
- Logs the change in an audit table
- Reports if the lot was used in any production batches (those records get updated too)

**Result:**
```
✅ Lot reassigned
Lot: 26-02-09-CARI-001
From: Coconut Oil 35 LB
To: Organic Coconut Oil 35 LB
Reason: received under wrong product name
Production records updated: 2 batches
```

**Warning:** This changes the product on ALL transactions for that lot. Use carefully.

---

## Quick Reference — All 21 Workflows

| # | Say This | System Does |
|---|----------|-------------|
| 1 | "Received 40 cases of X from Y, BOL Z" | Creates lot, adds inventory |
| 2 | "Ship 500 lb X to Y" | Deducts from oldest lot |
| 3 | "Ship 2,000 lb X to Y, multi-lot" | Splits across multiple lots |
| 4 | "Make 2 batches of X" | Consumes ingredients, creates output |
| 4a | "Pack 140 cases of FG from Batch X" | Converts batch → FG cases |
| 5 | "Adjust lot X down 50 lb, reason: Y" | Corrects inventory |
| 6 | "New order from X: 360 cases Y at $4.50" | Creates sales order |
| 7 | "Show open orders" / "What's overdue?" | Lists orders |
| 8 | "Confirm order SO-..." | Updates status |
| 9 | "Add / cancel / change line on SO-..." | Modifies order |
| 10 | "Ship order SO-..." | Fulfills order, updates balances |
| 11 | "Search for coconut" | Finds matching products |
| 12 | "How much granola do we have?" | Shows stock levels by lot |
| 13 | "Recent transactions" / "What shipped today?" | Shows activity log |
| 14 | "What's the formula for Classic Granola?" | Shows batch recipe |
| 15 | "Trace batch B26-0209-001" / "Where was lot X used?" | Forward & backward traceability |
| 16 | "Add customer X" / "Show customers" | Manage customer list |
| 17 | "Sales dashboard" | Status, overdue, due soon, recent ships |
| 18 | "Create new product: X, type: finished_good" | Adds product (unverified) |
| 19 | "Found 500 lb of X in warehouse" | Logs found inventory with FOUND lot |
| 20 | "Reassign lot X to product Y" | Moves lot to correct product |
