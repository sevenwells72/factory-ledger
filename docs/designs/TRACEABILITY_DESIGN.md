# TRACEABILITY_DESIGN — Canonical Lot Genealogy for Factory Ledger

**Status:** DRAFT — design only, no migrations written.
**Date:** 2026-08-31 · **Author:** Claude (session for Michael) · **Schema basis:** `tests/schema/schema.sql` (prod dump 2026-08-20, post-046 columns verified against `migrations/046_inventory_occurred_at.sql`), `main.py` @ `65bca82` + the three 8/31 fixes on `fix/audit-insert-savepoints` (`84c4797`, `dc6a9c0`, `66f7aee`).
**Audit basis:** `docs/data-health-baseline-2026-08-24.md`, `docs/data-entry-inventory.md` (both on main), and `docs/traceability-audit-2026-08.md` (branch `audit/traceability-2026-08`, commit `fda4f9a`, not merged).

> **A note on "CC-1/CC-2/CC-3":** no findings under those labels exist anywhere in the repo (searched all branches). This doc treats the three headline data-health findings as the intended referents and labels them here for cross-reference:
> - **CC-1 — lot-code collisions:** 159 lot codes resolve to >1 lot record (baseline §3i); date-style codes span up to 14 products in one day; a code-only backward trace on `AUG 21 2026` attaches granola transactions to a coconut shipment (§3i-iii/iv). No global uniqueness on `lots.lot_code`; the only constraint is `UNIQUE (product_id, lot_code)`.
> - **CC-2 — trace incompleteness:** 0/49 batches in the last 30 d fully resolve to supplier-bearing receives (§3g) — dominated by long-lived `found_inventory` coconut lots with no receive by design; forward trace from a batch lot dead-ends in code (`trace_ingredient` hard-codes `batches = []` for output lots, §4) even though pack ILC rows exist.
> - **CC-3 — event time is not captured:** `occurred_at` was server-derived at entry for every write until PR #18 (2026-08-25), so occurred ≡ entered for all floor history (§3a, "Reading notes"); the only real occurred/recorded gap ever recorded is the 8/17 recon's deliberate backdate. Adjustments are reconciliation catch-up (84% entered late, p50 3 d), not day-of recording.

---

## 1. Problem statement

The ledger already has better raw genealogy than most small-plant systems: every make and pack writes `ingredient_lot_consumption` (input lots) and `transaction_lines` (output lot), every ship has lot-level lines, and the 039 append-only ledger means history cannot be silently rewritten. What it does **not** have:

1. **A single canonical graph.** A recall trace today stitches four tables (`transaction_lines`, `ingredient_lot_consumption`, `shipment_lines`, `sales_order_shipments`) with per-type join logic, filtered through `ledger_current_transactions.effective_status`, and the one code path that tries (`trace_ingredient`, `main.py:6761`) has a known false-premise bug for output lots.
2. **Collision-safe lot identity** (CC-1). `lot_code` is a display string, not a key; `/pack` *intentionally* reuses the source batch lot's code for the FG lot, so global code uniqueness is impossible in this domain.
3. **Honest event time** (CC-3). `occurred_at` exists and (post-PR #18) is caller-suppliable, but nothing surfaces or flags occurred-vs-recorded gaps.
4. **A machine-checkable completeness story** (CC-2): orphans, dead-ends, and mass-balance drift are only discovered by ad-hoc audits.

## 2. Three architectures, evaluated against the actual schema

### Option 1 — `lot_links` genealogy table only

A single edge table: `(parent_lot_id, child_lot_id, quantity_lb, via_transaction_id)`.

| For | Against |
|---|---|
| Trivial to build; backward/forward trace is one recursive CTE. | **Ships, receives, and adjusts are not lot→lot edges.** A shipment is a terminal event on the *same* lot; a receive has no parent lot; an adjust is signed quantity on one lot. They'd need NULL-parent / NULL-child pseudo-edges, at which point the table is an event table with the semantics stripped off. |
| No new write-path semantics. | No `occurred_at`/`recorded_at` per event (CC-3 unaddressed); no place for customer/supplier parties; no role vocabulary, so a "moved" vs "shipped" vs "consumed" distinction is unrepresentable. |
| | Mass balance is impossible from edges alone — you still join the four operational tables, so the "single canonical graph" goal fails. |
| | GS1 EPCIS event-type awareness (a stated requirement) cannot be modeled: EPCIS is event-centric, not edge-centric. |

**Verdict: rejected.** It answers only the backward/forward CTE requirement and none of the others.

### Option 2 — unified event tables replacing parts of the operational model

Replace `transactions`/`transaction_lines`/`ingredient_lot_consumption` with a normalized `events` + `event_lots` pair; existing tables become views.

| For | Against |
|---|---|
| One model, no duplication; EPCIS-native. | **The blast radius is the entire system.** ~50 balance queries read `POSTED_LINES`/`lot_on_hand()`; the 039 append-only triggers, `ledger_corrections`, and both `ledger_current_*` views are keyed to `transactions`/`transaction_lines`; migration 044's `sales_order_allocations` carries FKs to `transactions` (`ship_transaction_id`, `last_ship_transaction_id`) and its void/restore machinery reads `ledger_corrections` rows targeting those tables; two GPT schemas (30-op and 22-op, both at hard caps) and the dashboard read the current shapes. |
| | The 2026-06 void-semantics rebuild and the 039 append-only migration were each multi-week efforts on this exact surface. Redoing them inside a rewrite, on a live single-`main.py` app with concurrent-session development, is the highest-risk option by an order of magnitude. |
| | Backfill becomes a *migration of record* rather than a derivation — if the transform is wrong, the operational history is wrong, not just the trace copy. |

**Verdict: rejected.** Right destination shape, wrong cost/risk for a 14.8k-line production `main.py` with append-only history already load-bearing.

### Option 3 — canonical trace-event layer (RECOMMENDED)

`trace_events` + `trace_event_lots`, written **atomically inside the same DB transaction** as the operational commit (RECEIVE / MAKE / PACK / SHIP / ADJUST and their void/restore corrections). Existing tables remain the domain model and the source of on-hand truth; the trace layer is the canonical **genealogy graph** — the one place recall queries, the integrity report, and the dashboard trace UI read from.

**The rule that makes it canonical:** *traceability is never entered independently.* There is no endpoint that writes `trace_events` directly; rows exist only because a real operational transaction emitted them in the same commit. A trace row with no operational parent is by construction a bug, and the integrity report treats it as one.

| For | Against / mitigations |
|---|---|
| Zero change to on-hand math, allocations, void semantics, GPT schemas, or any existing read. | Duplication: quantities exist in both layers. Mitigated by (a) emission being same-transaction and fail-hard, (b) the nightly integrity report proving 1:1 coverage and quantity equality against the operational tables (§8, checks R1–R2). |
| EPCIS event semantics fit naturally (§4). | Two more tables to keep append-only. They reuse the existing `ledger_enforce_created_at` / append-only trigger pattern from 039 — no new machinery. |
| Backfill is a pure derivation from existing history (§9) — re-runnable, verifiable, discardable. | Emission code must be added to 5 commit paths + the correction path. These are exactly the paths inventoried line-by-line in `docs/data-entry-inventory.md`; hook points are known. |
| Fixes CC-2 structurally: the forward-trace dead-end disappears because pack events are first-class transformation events, not an `is_output_lot` special case. | |

**Verdict: adopt Option 3.** The rest of this document designs it.

---

## 3. Schema design (DDL sketch — illustrative, not a migration)

Everything here follows house conventions: append-only via the 039 trigger pair, `created_at`/`created_at_source` DB-owned, `operator_id` as source tag (FR-15 interim rule — plain tags, no fake users, nullable rather than `'legacy-shared-key'` on new tables, matching the approved `sales_order_allocations.created_by` deviation).

### 3.1 Lot identity: `lots.lot_uuid` (CC-1)

```sql
ALTER TABLE lots ADD COLUMN lot_uuid uuid NOT NULL DEFAULT gen_random_uuid();
ALTER TABLE lots ADD CONSTRAINT lots_lot_uuid_key UNIQUE (lot_uuid);
```

- The **UUID is the machine identity** (QR payload, API scan key). It is minted once, survives `PATCH /lots/{lot_id}/rename`, and is never ambiguous.
- The **code stays human**: `UNIQUE (product_id, lot_code)` (already exists, `lots_product_id_lot_code_key`) remains the only code constraint that is *true* in this domain — `/pack` legitimately reuses the batch lot's code across FG SKUs, so global code uniqueness would break the plant's actual labeling practice. Every code-based lookup must carry `product_id` (the API already 409s `ambiguous_lot_code` without it).

Additional DB-level collision hardening — a **two-tier scheme** (safe against the 159 existing collisions because it constrains **new rows only** or normalizes within the existing key; tier choice validated against full lot history 2026-08-31, evidence below):

**Tier 1 — HARD unique index (blocks the write).** Case + whitespace normalization plus the trailing `LOT` noise-token strip — the strongest normalization the historical data affirmatively supports for a blocking constraint:

```sql
-- Code twins within one product ("APR 10 2026 Lot" vs "APR 10 2026", "BB041327 Lot" vs
-- "BB041327"): upper-case, trim + collapse internal whitespace, drop a trailing 'LOT'
-- noise token, so casing/spacing/'Lot'-suffix variants of the same physical lot
-- collide at mint time.
-- Merged lots keep their lot_code (merge sets status='merged' without renaming), so the
-- index MUST exclude them — otherwise the first twin-merge leaves a colliding merged row
-- that makes the index impossible to build and future twin-merges impossible to complete.
CREATE UNIQUE INDEX lots_product_code_norm_uniq
    ON lots (product_id,
             regexp_replace(
                 regexp_replace(upper(btrim(lot_code)), '\s+', ' ', 'g'),  -- case + whitespace
                 '\s+LOT$', ''))                                           -- trailing 'LOT' token
    WHERE status IS DISTINCT FROM 'merged';
```

**Tier 2 — SOFT mint-time warning (never blocks the write).** The full aggressive normalization — tier 1 plus stripping *all* remaining non-alphanumerics (internal whitespace and punctuation `[./#-]` etc.) — moves out of the constraint and into detection. At every code-minting path (`/receive`, `/make`, `/pack`, found-inventory entry), if the new code's aggressive key matches an existing non-merged lot of the same product while their tier-1 keys differ, the write **succeeds** and the API response carries a `suspicious_code_similarity` warning naming the near-twin lot; the §8 T2 integrity check lists all such within-product near-twin pairs for human review (→ `/admin/lots/merge` when they turn out to be real twins).

```sql
-- SOFT key (warning + §8 T2 report only — never a unique index or constraint):
regexp_replace(
    regexp_replace(upper(btrim(lot_code)), '\s+LOT$', ''),
    '[^A-Z0-9]', '', 'g')
```

**Why the split — validation evidence (2026-08-31, read-only sweep of all 1,048 historical lots, all statuses incl. the 2 merged, all products):**

- *Case + whitespace-collapse tier:* **0** within-product collision groups anywhere in history.
- *+ trailing-LOT strip (= the tier-1 index):* exactly **1** additional group — `BB041327 Lot` / `BB041327` (product 145, lots 401/410), both house `pack_output` codes, a confirmed twin already merged in step-0. Under the index's non-merged predicate: **0** violations — the tier-1 index builds clean today.
- *Full aggressive tier:* exactly **1** further group — `JUL15 2026` / `JUL 15 2026` (product 107, lots 999/1003), both house `production_output`, also a confirmed step-0-merged twin. A punctuation-only variant (strip `[./#-]` but keep whitespace) creates **0** additional groups; the aggressive tier's extra catch came entirely from internal-whitespace removal.
- *Supplier-assigned codes:* **no tier collapses any pair of supplier-entered lots** (`received` + `found_inventory`, 314 lots), and the `lots.supplier_lot_code` side-channel (112 real values; `lot_supplier_codes` table is empty) has **0** aggressive-key twins. So no plausibly-distinct supplier pair is collapsed — but this is *vacuous* rather than affirmative evidence for punctuation-stripping: 273 of the punctuation-bearing supplier/manual codes are house-format receiving codes (`26-01-28-DUTV-001` style, where dash-stripping is harmless), only ~43 lot codes are genuinely supplier-style (bare numerics like `25120`, `261212585122`), and almost none carry punctuation. The punctuation rule has effectively never been exercised by a real supplier code.
- Every real twin the stronger tiers ever caught was **house-generated**, and the generator races that minted those twins are being closed separately (advisory lock 3, suffix-safe probe). A future supplier pair like `1234-1` vs `123-41` would be collapsed — and the receive **hard-blocked at the dock** — by an aggressive unique index, with zero historical precedent to justify that risk. Blocking a legitimate receive is the costlier failure mode; promoting the soft key into the hard index later (if T2 stays quiet) is a cheap follow-up migration, while demoting it after a dock incident is not. Hence: aggressive = warn, never block.
- Step-0 pre-clean cross-check (`docs/trace-preclean-worklist-2026-08.md`): the 2 twin groups found there are exactly the 2 groups above, both merged pre-047; the "expected ≈3" APR 10 2026 twins are cross-product and never violated any per-product tier.

```sql
-- Format discipline for newly minted codes, without invalidating history:
ALTER TABLE lots ADD CONSTRAINT lots_code_format_chk
    CHECK (lot_code ~ '^[A-Z0-9][A-Z0-9 ./#-]{2,49}$' AND lot_code !~ '(ENE|ABR|AGO|DIC|SET) ')
    NOT VALID;   -- checks INSERT/UPDATE only; existing rows untouched, never VALIDATEd
```

Generator races are already being closed on this branch (advisory lock 3 for `/make`, suffix-safe sequence probe — commits `dc6a9c0`, `66f7aee`); this design assumes those land.

### 3.2 `trace_events` — one row per physical event

```sql
CREATE TABLE trace_events (
    id                bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    event_uuid        uuid NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    -- what happened (domain verb) and its EPCIS classification
    event_type        text NOT NULL CHECK (event_type IN
                        ('receive','make','pack','ship','adjust','void','restore','amend','merge')),
    epcis_type        text NOT NULL CHECK (epcis_type IN ('object','transformation','aggregation')),
    -- provenance: the operational transaction that emitted this event
    transaction_id    integer NOT NULL REFERENCES transactions(id),
    correction_id     uuid REFERENCES ledger_corrections(id),   -- set for void/restore/amend events
    -- time (CC-3)
    occurred_at       timestamptz NOT NULL,   -- copied from transactions.occurred_at (claimed physical time)
    recorded_at       timestamptz NOT NULL DEFAULT clock_timestamp(),  -- DB-owned, trigger-enforced
    business_date     date NOT NULL,
    late_recorded     boolean GENERATED ALWAYS AS
                        ((recorded_at - occurred_at) > interval '24 hours') STORED,
    -- parties and place (free text mirrors the operational row; FKs where they exist)
    biz_location      text NOT NULL DEFAULT 'seven-wells-plant',
    source_party      text,          -- receive: shipper_name
    destination_party text,          -- ship: customer_name
    customer_id       integer REFERENCES customers(id),   -- when resolvable (SO-ship path)
    sales_order_id    integer REFERENCES sales_orders(id),
    operator_id       text,
    created_at        timestamptz NOT NULL DEFAULT clock_timestamp(),
    created_at_source text NOT NULL DEFAULT 'database',
    CONSTRAINT trace_events_correction_pairing CHECK
        ((event_type IN ('void','restore','amend')) = (correction_id IS NOT NULL)),
    CONSTRAINT trace_events_txn_type_uniq UNIQUE (transaction_id, event_type, correction_id)
);
CREATE INDEX trace_events_txn_idx ON trace_events (transaction_id);
CREATE INDEX trace_events_occurred_idx ON trace_events (occurred_at);
CREATE INDEX trace_events_late_idx ON trace_events (business_date) WHERE late_recorded;
```

`late_recorded` implements the required occurred/recorded gap flag at a 24 h threshold; the integrity report (§8, check T1) also reports the raw gap distribution so the threshold can be tuned without a migration (it's a generated column — changing the threshold is one `ALTER ... DROP/ADD COLUMN`, data-free). Deliberate reconstructions (PR #18's `backfill: true` → `transactions.entry_backfilled`) are reported separately from accidental lateness.

### 3.3 `trace_event_lots` — the lots an event touched, with roles

```sql
CREATE TABLE trace_event_lots (
    id              bigint GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
    trace_event_id  bigint NOT NULL REFERENCES trace_events(id),
    lot_id          integer NOT NULL REFERENCES lots(id),
    role            text NOT NULL CHECK (role IN
                      ('input','output','shipped','received','adjusted','moved')),
    quantity_lb     numeric(14,4) NOT NULL CHECK (quantity_lb <> 0),
    -- sign convention: positive = into the lot (output/received/found), negative = out of it
    -- (input/shipped/consumed); 'adjusted' may be either sign; matches transaction_lines.
    created_at        timestamptz NOT NULL DEFAULT clock_timestamp(),
    created_at_source text NOT NULL DEFAULT 'database',
    CONSTRAINT tel_role_sign CHECK (
        (role IN ('output','received') AND quantity_lb > 0)
     OR (role IN ('input','shipped')  AND quantity_lb < 0)
     OR (role IN ('adjusted','moved'))
    ),
    CONSTRAINT tel_event_lot_role_uniq UNIQUE (trace_event_id, lot_id, role)
);
CREATE INDEX tel_lot_idx ON trace_event_lots (lot_id);
CREATE INDEX tel_lot_role_idx ON trace_event_lots (lot_id, role);
```

**Event-type awareness (GS1 EPCIS requirement):**

- **Transformations** (`make`, `pack`) → `epcis_type='transformation'`: N rows `role='input'` (negative) + 1 row `role='output'` (positive). Inputs and outputs are different lots.
- **Moves and ships** reference the **same lot with a role, never a new lot**: `ship` → `epcis_type='object'`, one row per shipped lot with `role='shipped'` (negative), `destination_party` = customer. A future internal move is `role='moved'` with location columns — the vocabulary slot exists today.
- **Receives** → `epcis_type='object'`, `role='received'` (positive), `source_party` = supplier.
- **Adjusts** (including found-inventory) → `epcis_type='object'`, `role='adjusted'`, signed.
- `aggregation` is reserved for pallets (§7).

Both tables get the standard 039 protections: `trg_*_created_at` (`ledger_enforce_created_at`) and `trg_*_append_only` (`ledger_block_append_only_change`) — the trace layer is append-only like the ledger it mirrors.

### 3.4 Void/correction semantics — one source of truth

Voids do **not** delete or flip trace rows. When `POST /void/{id}` (or the corrections endpoint) appends a `ledger_corrections` row, the same commit appends a `trace_events` row (`event_type='void'|'restore'|'amend'`, `correction_id` set, **no** `trace_event_lots` rows) — an audit marker, not a quantity event.

**Visibility is resolved exactly one way:** trace traversals join `ledger_current_transactions` on `trace_events.transaction_id` and keep only `effective_status = 'posted'`. Void state is never duplicated into the trace layer, so it can never disagree with the ledger. (This mirrors the hard-won lesson of June 2026: one posted-only subquery, everywhere.)

---

## 4. Emission: where and how

Emission happens **in application code, inside the existing commit transaction, fail-hard** — explicitly *not* the swallowed-`except` "best-effort" pattern whose failure mode (silently rolled-back primary writes reported as success) was demonstrated on 2026-08-25 (`docs/data-entry-inventory.md` §Risk 3) and is being fixed with savepoints on this branch. A trace-emission failure must abort the operational commit: an untraceable transaction is worse than a retried one.

One helper, called at the end of each commit path (after all `transaction_lines`/`ingredient_lot_consumption`/allocation writes, before `conn.commit()`):

```
emit_trace_event(cur, txn_id, event_type, lot_roles=[(lot_id, role, qty), ...], **parties)
```

| Path (main.py @ 65bca82) | Emits | Lot roles |
|---|---|---|
| `/receive` commit (~L4169) | `receive` (object) | received lot, `+qty`; `source_party=shipper_name` |
| `/make` commit (~L5613) | `make` (transformation) | each ILC row → `input` (−); output lot → `output` (+) |
| `/pack` commit (~L6103) | `pack` (transformation) | each source-batch ILC row → `input` (−); FG lot → `output` (+) |
| `/ship` commit (~L5224 path) and SO-ship commit (~L10786 path) | `ship` (object) | each shipped lot line → `shipped` (−); `destination_party=customer_name`, `customer_id`/`sales_order_id` when resolved |
| `/adjust` commit (~L6268), `/inventory/found*` | `adjust` (object) | lot, signed |
| `POST /void/{id}`, `POST /records/transactions/{id}/corrections` | `void`/`restore`/`amend` marker | none (correction_id set) |
| `POST /admin/lots/merge` | `merge` marker on the amending correction + **repoint** is forbidden — see §10 conflict M1 | none |

Why app code and not triggers: a complete event spans three tables whose rows arrive in sequence inside the commit (`transactions` → `transaction_lines` → `ingredient_lot_consumption`); a row-level trigger sees fragments, and a deferred constraint-trigger assembly would re-derive what the endpoint already holds in memory. The integrity report's coverage checks (R1/R2) are the guard against a forgotten hook — any posted transaction without exactly its expected trace event fails the nightly report loudly. A slim safety-net `AFTER INSERT` statement trigger on `transactions` that only *queues* an id into a one-column `trace_emission_debt` table (drained and asserted empty by the report) is an optional belt-and-braces addition, listed as an open question (§12).

`occurred_at` on the event copies `transactions.occurred_at` — which, post-PR #18, callers can supply (with the >14 d `backfill` gate). The QR scan flow (§6) supplies it from scan time, which is the first mechanism this plant has ever had that captures physical event time at the moment it happens (CC-3).

---

## 5. Trace queries (recursive CTEs over the trace layer only)

Backward (any lot → its supplier receipts). Forward is the mirror image.

```sql
-- posted-only lens, resolved once
WITH posted_events AS (
    SELECT te.*
    FROM trace_events te
    JOIN ledger_current_transactions lct ON lct.id = te.transaction_id
    WHERE lct.effective_status = 'posted'
      AND te.event_type IN ('receive','make','pack','ship','adjust')
),
RECURSIVE upstream AS (
    SELECT l.id AS lot_id, 0 AS depth, ARRAY[l.id] AS path
    FROM lots l WHERE l.lot_uuid = $1            -- or (product_id, lot_code)
  UNION ALL
    SELECT tel_in.lot_id, u.depth + 1, u.path || tel_in.lot_id
    FROM upstream u
    JOIN trace_event_lots tel_out ON tel_out.lot_id = u.lot_id
                                 AND tel_out.role IN ('output','received','adjusted')
    JOIN posted_events pe ON pe.id = tel_out.trace_event_id
    JOIN trace_event_lots tel_in ON tel_in.trace_event_id = pe.id
                                AND tel_in.role = 'input'
    WHERE NOT tel_in.lot_id = ANY(u.path)        -- cycle guard
      AND u.depth < 20
)
SELECT DISTINCT ON (u.lot_id) u.lot_id, u.depth, l.lot_code, p.name,
       pe.source_party AS supplier, pe.occurred_at AS received_at
FROM upstream u
JOIN lots l ON l.id = u.lot_id
JOIN products p ON p.id = l.product_id
LEFT JOIN trace_event_lots tel ON tel.lot_id = u.lot_id AND tel.role = 'received'
LEFT JOIN posted_events pe ON pe.id = tel.trace_event_id
ORDER BY u.lot_id, u.depth;
```

Forward swaps the join: from `role='input'` rows find the event, take its `output` lot, recurse; terminal `role='shipped'` rows carry `destination_party` — the customer list. Both are single statements, indexable (`tel_lot_role_idx`), and depth in this plant is ≤3 (ingredient → batch → FG), so the <5 min recall criterion is met with orders of magnitude of headroom (expected: milliseconds).

Exposed as `GET /trace/lots/{lot_uuid}/backward|forward` (also accepting `?lot_code=&product_id=`), replacing the buggy `trace_ingredient` traversal; the dashboard `traceability.html` switches to these.

---

## 6. QR labels and scan entry (lot-level)

**Label** (printed at receive / make / pack commit, from the commit response):

- Human line: `lot_code` + product name + business_date + quantity.
- Machine line: QR encoding `FL1:<lot_uuid>` (versioned prefix, 41 chars — small QR, survives label wear better than a URL; the scanner app owns the base URL).

**Endpoints:**

| Endpoint | Purpose |
|---|---|
| `GET /lots/{lot_id}/label` | render label payload (JSON now; PNG/ZPL a dashboard concern) |
| `GET /scan/{lot_uuid}` | resolve: lot, product, on-hand, open SO allocations, last 5 events — the "what is this pallet-less bag" screen |
| `POST /scan/{lot_uuid}/event` | **routes to the existing operational endpoints — it never writes trace rows itself.** Body: `{action: receive_confirm\|consume\|pack_source\|ship\|adjust, quantity_lb, occurred_at?}`. The handler translates to the corresponding `/make`/`/pack`/`/ship`/`/adjust` commit call with the lot pinned by id (no code ambiguity possible) and `occurred_at` defaulted to scan time. The no-independent-traceability rule holds: a scan is just a better-keyed way to enter the real transaction. |

**Pallet aggregation later, without schema change:** EPCIS aggregation events fit the existing shape. A pallet becomes a `lots` row on a `packaging`-type pseudo-product (`entry_source='container'` — new enum value, text column, no DDL) with its own `lot_uuid`/QR; palletizing emits a `trace_events` row with `epcis_type='aggregation'` and `trace_event_lots` rows `role='moved'` (contained FG lots, qty 0-signed exempted via role) plus the pallet lot as `role='output'`. Only CHECK-constraint enum extensions (`role`: add `'contained'`; `event_type`: add `'aggregate'`) — `ALTER TABLE ... DROP/ADD CONSTRAINT`, metadata-only, no table rewrite, no new tables.

## 6.5 Offline scanning

*(Inserted as §6.5 rather than renumbering — the cross-references to §§7–12 throughout this doc stay valid.)*

The scanner UI is an **offline-first PWA**: a service worker caches the app shell and an IndexedDB local queue stores scans on-device, so scanning never depends on connectivity — a scan in a dead zone succeeds instantly and syncs later.

**Each queued scan stores:**

- the scan payload (`lot_uuid`, action, quantity, any action-specific fields),
- `occurred_at` — **captured at scan time on the device**, not at sync time,
- `device_id`,
- a **client-generated idempotency UUID**, minted when the scan is taken.

**Sync:** when connectivity returns, the queue drains to the FastAPI scan endpoint **in order**. The server deduplicates on the idempotency key, so a re-drained or retried queue is harmless.

**Idempotency must ride through to the operational call.** Because `POST /scan/{lot_uuid}/event` routes to the existing operational endpoints (§6 rule — it never writes trace rows itself), the idempotency key must be passed through to the underlying `/make`/`/pack`/`/ship`/`/adjust` commit call so that a re-synced scan cannot double-post the operational transaction. Deduplicating only at the scan-endpoint layer would leave a window (operational commit succeeded, response lost in transit) where a client retry posts the transaction twice.

**Validation failures do not block the sync.** A synced scan that fails server-side validation (insufficient on-hand, inactive lot, constraint violation, …) is accepted off the device queue and lands in an **exceptions queue** — full payload, `device_id`, `occurred_at`, rejection reason — for human resolution (replay after fixing the cause, or dismissal). The rest of the queue keeps draining. Open exceptions are surfaced by the §8 integrity report as check class **E1**.

**Expected occurred/recorded gaps:** dead-zone scans routinely sync with `occurred_at`/`recorded_at` gaps of up to **~60 min**. This is additional justification for the 24 h `late_recorded` threshold (§3.2, §7) rather than a tighter one — a tighter threshold would flag routine offline sync as lateness.

---

## 7. occurred_at vs recorded_at (CC-3)

Already designed into §3.2; summarized as policy:

- `occurred_at` = claimed physical event time. Caller-suppliable since PR #18 (≤5 min future rejected; >14 d past requires `backfill: true` → `entry_backfilled`). Scan entry supplies it automatically.
- `recorded_at` = DB clock at insert, trigger-owned, never caller-settable (existing `ledger_enforce_created_at` pattern).
- `late_recorded` (generated, >24 h) is the flag; the integrity report (T1) breaks late events down by type/operator and separates `entry_backfilled` reconstructions (expected, e.g. the 8/17 recon's 75.8 h backdate — the *correct* use) from unexplained lateness (the adjust-discipline problem the audit graded D).

---

## 8. Automated integrity report

`scripts/trace_integrity.py` (read-only: 6543 pooler, per-query `BEGIN TRANSACTION READ ONLY`, per the hard rule) + `GET /dashboard/api/trace/integrity` for the dashboard. Nightly via the existing `daily-health-ping.sh` slot. Every check returns count + drill-down rows; the report FAILS (non-zero exit, dashboard red) on any nonzero R/O/M/E-class result.

| # | Check | Definition |
|---|---|---|
| R1 | Coverage | every posted-effective transaction of the 5 types has exactly its expected `trace_events` row(s); every trace event has a live parent transaction. (Guards forgotten emission hooks and drift.) |
| R2 | Quantity mirror | per (transaction, lot): Σ`trace_event_lots.quantity_lb` = Σ posted `transaction_lines.quantity_lb`; make/pack inputs match `ingredient_lot_consumption`. |
| O1 | **Orphan lots** | lots with any outbound/consumption event but **no origin event** (`received`/`output`/positive `adjusted`). `entry_source='found_inventory'` lots *with* their found-adjust origin are **explained** — reported informationally, not failures (this is the CC-2 population: `25120`, `6013`, the FOUND-series). A lot with no origin event at all is unexplained → FAIL. |
| O2 | **Outputs with no inputs** | transformation events (`make`/`pack`) having an `output` row but zero `input` rows. Grandfathered seed population = the **8 posted + 1 voided legacy ILC-less packs** (full-history census, §9 step 0 — the old "2 known" figure was the 90-day-windowed audit count), in two shapes: (i) txns **1964, 1966** — batch debits exist in `transaction_lines` but have no ILC mirror; (ii) Feb-05 txns **77, 78, 94, 95, 97, 98** — single negative batch line, no ILC *and no FG output line at all*. The backfill synthesizes whatever each shape actually has and marks every synthesized row `created_at_source='trace_backfill_048'`; O2 exempts marked events and FAILs on any unmarked occurrence. |
| O3 | **Shipments with no source lot** | `ship` events with zero `shipped` rows, or shipped lots with no origin event. |
| O4 | **Consumption from nonexistent inventory** | `input`/`shipped` rows whose lot has no prior positive event, or whose cumulative lot balance (by `occurred_at`) goes negative at the event. |
| M1 | **Mass balance — ingredient lots** | per lot: received − consumed(inputs) − |negative adjusts| + positive adjusts = on-hand (`lot_on_hand()`); tolerance 0.01 lb. |
| M2 | **Mass balance — batch lots** | produced(output) − packed(inputs of pack events) ± adjusts = remaining. |
| M3 | **Mass balance — FG lots** | packed(output) − shipped ± adjusts = remaining. |
| T1 | Timeliness | `late_recorded` events by type/operator; `entry_backfilled` split out. Informational. |
| T2 | Code hygiene | Two parts, mirroring §3.1's two tiers. (a) *Hard-rule violations:* new lots violating the tier-1 normalized-uniq or format rules (should be impossible post-047; belt-and-braces). (b) **Near-twin listing (soft tier):** within-product non-merged lot pairs whose §3.1 aggressive keys match while their tier-1 keys differ — i.e. codes differing only by internal whitespace/punctuation — the same population the mint-time `suspicious_code_similarity` API warning fires on. Listed with lot ids/codes/`entry_source` for human review (→ `/admin/lots/merge` for real twins); informational, never blocks a write and never fails the report. Historical baseline at design time: 0 pairs (both known twins merged in step-0). |
| E1 | **Scan sync exceptions** | open entries in the offline-scan exceptions queue (§6.5) — synced scans rejected by server-side validation, awaiting replay or dismissal. Reported with payload/device/occurred_at drill-down; counts as a failing class while any remain open (a rejected scan is a real physical event the ledger hasn't recorded). |

M1–M3 are the requirement's mass-balance identity split by lot tier; because trace quantities mirror `transaction_lines` (R2) and on-hand is `lot_on_hand()` over the same lines, M-checks catch *semantic* holes (missing events, wrong roles) rather than arithmetic drift.

---

## 9. Migration & backfill plan (sequenced; each step separately approvable per house rules)

0. **Pre-clean (data-only, no DDL):** DONE 2026-08-31 — `docs/trace-preclean-worklist-2026-08.md`. (a) Twin dry run: **0 violations** of the original case/whitespace-only index; **2 within-product twin groups** under the stronger §3.1 normalization (1003→999 product 107, 401→410 product 145), merged via `/admin/lots/merge` with per-merge owner approval. (b) ILC-less pack census (full history, not the 90-day audit window): **8 posted + 1 voided** legacy packs in two shapes — txns 1964/1966 (lines but no ILC mirror) and Feb-05 txns 77/78/94/95/97/98 (no ILC, no FG output line) — annotated in the worklist as the approved grandfather population; the marker itself is applied by step 4.
1. **Migration 047 — identity:** `lots.lot_uuid` + unique; **tier-1** normalized unique index (§3.1 hard tier — case/whitespace/trailing-LOT only); `NOT VALID` format CHECK. (Backfills UUIDs via default on ADD COLUMN; table is small — ~1.3k rows.) The §3.1 tier-2 `suspicious_code_similarity` warning is API code, not DDL — it ships with the mint-path/emission changes (step 3 at the latest).
2. **Migration 048 — trace tables:** `trace_events`, `trace_event_lots`, indexes, append-only + created_at triggers. Zero rows, zero behavior.
3. **Deploy emission code** (all 6 hook sites + `emit_trace_event`, fail-hard). From this moment forward-writes are dual. Smoke: one make on prod, verify event + roles.
4. **Backfill script** (idempotent, keyed on `transaction_id`): walk `ledger_current_transactions` **base rows** (all statuses — void markers come from `ledger_corrections` replay) ordered by id; synthesize events per §4's table. Sources: `transaction_lines` for roles/quantities, `ingredient_lot_consumption` for inputs, `shipments`/`sales_order_shipments` for customer/SO ids, `shipper_name`/`customer_name` for parties. Time columns: `occurred_at` from the transaction; `recorded_at` from `created_at` **where `created_at_source='database'`**, else from legacy `"timestamp"` (the 039 backfill stamped `created_at` with the migration time — using it would mark 500+ historical rows falsely late); `created_at_source='trace_backfill_048'` on every synthesized row (the append-only trigger's created_at enforcement needs the same `api_backfill`-style carve-out 046 used — noted in the migration). For the 9 legacy ILC-less packs the backfill **synthesizes what actually exists**: txns 1964/1966 get `input` rows taken from their negative `transaction_lines` (quantities are right there — real genealogy preserved, no ILC needed); the Feb-05 six (77/78/94/95/97/98) get their input side from lines but **no `output` row** (none was ever posted). All 9 carry the `trace_backfill_048` marker. |
5. **Verify:** run the full §8 report. Gate: R1/R2 = 0, O-class = 0 unexplained (found-inventory origins explained; the 8 posted + 1 voided legacy packs grandfathered via the `trace_backfill_048` marker), M-class within tolerance — with **R2 and M2 excluding the marked population**: the Feb-05 shape has no output row so its quantity mirror and batch mass balance can never reconcile, and 1964/1966 have no ILC to mirror; marked rows report informationally instead of failing the gate. Fix-forward and re-run until green.
6. **Cut reads over:** new `/trace/*` endpoints + dashboard `traceability.html` switch; retire `trace_ingredient`'s traversal (the ≤5-line `batches=[]` fix from baseline §4 is superseded but worth shipping independently *now* as a stopgap).
7. **Labels/scan** (§6) — separate PR; needs no further schema.

Rollback story: steps 1–2 are additive; step 3's emission can be flag-gated (`TRACE_EMIT_ENABLED`, default on, off = revert to pre-trace behavior with the debt visible in R1); the backfill is truncate-and-rerun safe until reads cut over.

---

## 10. Conflicts with the 044 SO-allocations work (flagged as required)

- **M1 — `/admin/lots/merge` mutates ILC in place.** The merge endpoint repoints `ingredient_lot_consumption` rows and amends lines via `ledger_corrections`, and 044 taught it to coalesce/repoint `sales_order_allocations`. Backfilled trace rows derived from pre-merge ILC would go stale, and in-place mutation contradicts the append-only trace layer. **Resolution:** merge must emit a `merge` marker event and *append* corrected role rows (or the traversal must follow `lots.merged_into_lot_id`, which the CTE can do with one extra join); pick one in review — this doc recommends following `merged_into_lot_id` at read time, leaving history untouched.
- **Allocations are intent, not physical events — deliberately out of the trace layer.** `sales_order_allocations` rows (and their expire/shrink side-writes `_expire_auto_fifo_allocations`/`_shrink_overallocated_products`, which fire inside ship/pack commits) do not emit trace events. Only the resulting ship/pack transaction does. This keeps the graph physical and avoids coupling to 044's still-evolving lifecycle (the `inventory_shipped` shrink addendum §4.a.1).
- **Ordering inside the commit:** emission must run *after* 044's allocation side-writes (they can change which lots a ship actually draws — pinned-lot rules per the `be41b6d` fix round) so roles reflect the final deduction plan. Hook placement in §4 already states "after all … allocation writes".
- **`ALLOCATIONS_ENFORCED` flip (PR 5 flag, currently OFF):** when enforcement turns on, ship commits can newly fail/roll back; fail-hard emission rolls back with them — no action needed, but the enforcement-flip test plan should include a trace-coverage assertion.
- **Shared checkout / branch collision:** this design's emission hooks touch the same `main.py` commit paths as the in-flight `fix/audit-insert-savepoints` commits (savepoints around best-effort inserts, advisory lock 3). Land that branch first; emission code must sit *outside* any best-effort savepoint scope.
- **`sales_order_allocation_reactivations` (045)** already links ship-void corrections; the trace `void` marker carries the same `correction_id` — joinable, no duplication, no conflict.

---

## 11. Worked example — both directions

Illustrative ids; codes follow the real generators. Locations: `seven-wells-plant` throughout (single site today). Product names are real; `CQ Granola 10 LB` is SKU 1614 ("Chef Quality").

**The story:** 2,000 lb of gluten-free oats arrive from Dutch Valley on 9/01; 9/02 they go into a 1,938 lb batch of Classic Granola #9; 9/03 350 lb of that batch is packed into 35 × 10 lb CQ Granola cases; 9/04 140 lb (14 cases) ships to Restaurant Depot/Jetro Haines City #407; 9/04 the ship is discovered to have been entered against the wrong order and is voided, then re-entered correctly.

### Operational + trace rows produced

| # | Operational commit | trace_events row | trace_event_lots rows |
|---|---|---|---|
| 1 | `/receive` txn **2101**, 9/01 10:12 ET occurred, 10:14 recorded; shipper "Dutch Valley", BOL DV-88213; lot **1401** `26-09-01-DUTC-001` (Oats – Gluten Free), supplier lot `251121N` | ev 1: `receive`/object, txn 2101, occ 2026-09-01 14:12Z, rec 14:14Z, src "Dutch Valley", late=false | lot 1401 `received` **+2000.0** |
| 2 | `/make` txn **2102**, 9/02 08:40 occurred (scan-entered at 08:41); output lot **1402** `B26-0902-001` (Batch Classic Granola #9); ILC: oats 1401 −600.0, plus other ingredient lots (honey 1377 −180.0, coconut 1380 −220.0, …) | ev 2: `make`/**transformation**, txn 2102 | 1401 `input` **−600.0**; 1377 `input` −180.0; 1380 `input` −220.0; …; lot 1402 `output` **+1938.0** |
| 3 | `/pack` txn **2103**, 9/03 09:15; source lot 1402, target lot **1403** code `B26-0902-001` (CQ Granola 10 LB — code inherited, same string, different lot_id/uuid: CC-1 handled) | ev 3: `pack`/**transformation**, txn 2103 | 1402 `input` **−350.0**; 1403 `output` **+350.0** |
| 4 | SO-ship txn **2104**, SO-260904-002, 9/04 07:30 occurred, 07:32 recorded; customer #43 RESTAURANT DEPOT/Jetro Haines City #407 | ev 4: `ship`/object, txn 2104, dest party + customer_id 43 + SO id | 1403 `shipped` **−140.0** |
| 5 | `POST /void/2104` 9/04 11:05, reason "entered against SO-260904-002; belongs to -003"; `ledger_corrections` row c-77 | ev 5: `void` marker, txn 2104, correction_id c-77, **no lot rows** | — |
| 6 | SO-ship txn **2105** (correct SO-260904-003), same physical facts, `occurred_at` back-set to 07:30 | ev 6: `ship`/object, txn 2105 | 1403 `shipped` **−140.0** |

### Backward trace from the shipped FG lot (`lot_uuid` of 1403)

Posted-only lens drops ev 4 (txn 2104 `effective_status='voided'`). Result:

```
depth 0: lot 1403  B26-0902-001  CQ Granola 10 LB        (350 packed, 140 shipped, 210 on hand)
depth 1: lot 1402  B26-0902-001  Batch Classic Granola #9 (make ev 2, occurred 9/02 08:40)
depth 2: lot 1401  26-09-01-DUTC-001  Oats – Gluten Free  ← supplier: Dutch Valley, BOL DV-88213,
         supplier lot 251121N, received 9/01 10:12
         lot 1377, 1380, …                                 ← their suppliers likewise
```

### Forward trace from the supplier oats lot (1401)

```
lot 1401 → [ev 2 input] → lot 1402 (batch, 1,938 lb)
lot 1402 → [ev 3 input] → lot 1403 (CQ Granola 10 LB, 350 lb)
lot 1403 → [ev 6 shipped, −140 lb] → RESTAURANT DEPOT/Jetro Haines City #407, SO-260904-003, 9/04 07:30
(ev 4 excluded: parent txn voided; ev 5 visible in the event log as the audit marker)
```

Mass balance after the dust settles: lot 1403 packed 350 − shipped 140 = 210 remaining ✓ (the voided ship contributes nothing — `effective_status` filtering, not trace-layer bookkeeping, removed it). Lot 1402: produced 1,938 − packed 350 = 1,588 remaining ✓. Lot 1401: received 2,000 − consumed 600 = 1,400 on hand ✓.

---

## 12. Acceptance criteria (restated for sign-off) & open questions

1. **(a) Mock recall <5 min:** pick any shipped lot; backward trace to supplier lots *and* forward trace to all customer shipments each complete in <5 minutes wall-clock including a human reading the output. (Design expectation: each is one indexed CTE, sub-second.)
2. **(b) Backward completeness:** every FG lot shipped in the last 90 d reaches ≥1 supplier-bearing receive event or an *explained* found-inventory origin; unexplained dead-ends = 0.
3. **(c) Forward completeness:** every supplier lot consumed in the last 90 d enumerates every customer shipment that drew on it, with quantities.
4. **(d) Mass balance:** M1–M3 hold within 0.01 lb for all lots with post-backfill activity; legacy-only lots reported but not gated.
5. **(e) Zero unexplained orphans:** O1 unexplained = 0 (found-inventory lots with found-origin events count as explained).

### Decisions (2026-08-31)

- **Q1 — safety-net `trace_emission_debt` trigger: no.** Fail-hard emission plus nightly R1 is the guard. Revisit only if R1 ever catches a missed emission.
- **Q2 — lot merges: follow `lots.merged_into_lot_id` at read time** in the trace CTEs; trace history is never rewritten. (Resolves §10 conflict M1 per the recommendation there.)
- **Q3 — `late_recorded` threshold stays 24 h; T1 remains informational (non-gating).** The adjust-discipline expectation is to be communicated to the floor; gating will be reconsidered in October.
- **Q4 — the scan `ship` action is standalone-ship only in v1.** SO-allocation pre-fill is deferred until after the `ALLOCATIONS_ENFORCED` flip.
