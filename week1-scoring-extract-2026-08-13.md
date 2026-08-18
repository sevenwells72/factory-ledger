# Week 1 Scoring Extract — CNS Floor Manager Trial (v8.1)

## 1. Header

| Field | Value |
|---|---|
| Environment | **PRODUCTION** — Supabase Postgres, project `vrafvwcdpcijvxdvefpr`, via `aws-1-us-east-1.pooler.supabase.com:6543` (PostgreSQL 17.6). Confirmed not the local test DB. |
| Access mode | Read-only (`default_transaction_read_only=on`); no writes performed |
| Schema version | **REMEDIATED SCHEMA** — `transactions.business_date` / `occurred_at` / `created_at` + `created_at_source` / `operator_id` present; `ledger_corrections` present; `certifications` + `current_certifications` present; append-only Phase 1 ledger (migration 039) live |
| Dispatch proof fields | **ABSENT** — no proof/photo/signature/POD/attachment columns exist anywhere in the schema (see §4) |
| Extraction window | Mon 2026-08-10 00:00 ET → Thu 2026-08-13 14:35 ET (time of run; Thursday is a **partial day**) |
| Run timestamp | 2026-08-13 14:35:29 ET (18:35:29 UTC) |
| Service | Factory Ledger System v3.1.1, live at `fastapi-production-b73a.up.railway.app` (health: OK). Local repo `main` HEAD: `62ea3fe` (2026-08-12 14:49 ET) |

**Critical timing caveat that shapes everything below:** the remediation migration (039) went live in prod at **2026-08-11 10:32 ET**. Entries made before that moment (i.e., all of Monday 8/10) carry a backfilled `created_at` equal to the migration timestamp and `created_at_source='migration_backfill_039'`; their only entry-time evidence is the **legacy mutable `timestamp` column, which is NOT independently verifiable**. Entries from 8/11 10:35 ET onward have database-generated `created_at` (`created_at_source='database'`) and ARE verifiable. Timing rows below are labeled ✅ verifiable or ⚠️ legacy accordingly.

All times below are America/New_York unless marked UTC. `business_date` is the authoritative work day.

---

## 2. Per-day sections

### Monday 2026-08-10 — 9 transactions (1882–1890), ALL ⚠️ legacy timing

#### Production (makes)

| Txn | Product (batch) | Batches | Output lot | Output qty (lb) | Operator | Entry time |
|---|---|---|---|---|---|---|
| 1886 | Batch Classic Granola #9 | 16 | 1143 `AUG 10 2026` | 5,168.0 | legacy-unattributed | ⚠️ 15:55 (legacy ts) |
| 1888 | Batch Coconut Sweetened Flake (auto-excluded IDs: [50]) | 12 | 1145 `AUG 10 2026` | 4,795.2 | legacy-unattributed | ⚠️ 16:10 (legacy ts) |

Ingredient lots consumed (transaction_lines and ingredient_lot_consumption agree):

- **1886**: Coconut Macaroon Desiccated `25120` 200.0; Flavor – Almond `26-02-03-FOUND-002` 20.6 + `26-04-21-PARK-001` 8.2; Honey `26-07-10-DUTC-001` 25.6; Oats `26-07-15-QUAL-001` 2,215.2 + `26-08-04-QUAL-001` 1,384.8; Oil – Canola `26-07-24-CBSF-001` 480.0; Salt `26-04-21-ESSE-001` 27.2; Sugar – Light Brown `26-07-09-JACK-001` 800.0
- **1888**: Coconut Flake Desiccated `6013` 2,400.0; Corn Starch `26-04-21-ESSE-002` 60.0; Glycol `26-04-24-TILL-001` 84.0; Salt `26-04-21-ESSE-001` 30.0; Sugar – 6X `26-06-10-NEWE-001` 1,320.0

#### Packing / inventory movements

| Txn | Packed | Qty (lb) | From lot(s) | Into lot | Entry time |
|---|---|---|---|---|---|
| 1887 | 420 cases Graham Cracker Crumbs – 10 LB | 4,200.0 | GCC–50 LB `26-07-28-CREA-001` | 1144 `608101` | ⚠️ 16:02 |
| 1889 | 6 cases CQ Coconut Sweetened Flake 10 LB | 60.0 | Batch lot 1145 `AUG 10 2026` | 1142 `AUG 07 2026` (pre-existing lot) | ⚠️ 16:12 |
| 1890 | 434 cases CQ Coconut Sweetened Flake 10 LB | 4,340.0 | Batch lot 1145 `AUG 10 2026` | 1147 `AUG 10 2026` | ⚠️ 16:13 |

#### Shipments / dispatch (feeds 1b)

| Shipment | Order | Customer | Contents (product / qty lb / source lots) | Dispatch time | Proof fields |
|---|---|---|---|---|---|
| 323 | SO-260723-003 | Nac Foods Corp. | Coconut Sweetened Flake CNS 25 LB / 2,500 / `JUN 09 2026` 675 + `JUL 09 2026` 650 + `JUL 13 2026` 1,000 + `JUL 27 2026` 175 (txn 1882) • Desiccated Flake 50 LB / 400 / `26-04-22-FRAN-001` (1883) • Sprinkles Chocolate 10 LB / 100 / `25132` (1884) | ⚠️ 11:13 | none (absent from schema) |
| 324 | SO-260805-005 | David Rosen Bakery Supply | Granola Crunchy CNS 10 LB Case / 50 / `JUL 07 2026` (1885) | ⚠️ 14:11 | none (absent from schema) |

#### Receipts
None.

#### Entry-timing table (⚠️ all from legacy mutable `timestamp`; `created_at` on every row = backfill stamp 2026-08-11 10:32:52)

| Txn | Type | Legacy entry time (ET) | Verifiable? |
|---|---|---|---|
| 1882–1884 | ship | 11:13:24 | No — legacy |
| 1885 | ship | 14:11:17 | No — legacy |
| 1886 | make | 15:55:27 | No — legacy |
| 1887 | pack | 16:02:38 | No — legacy |
| 1888 | make | 16:10:52 | No — legacy |
| 1889 | pack | 16:12:17 | No — legacy |
| 1890 | pack | 16:13:30 | No — legacy |

---

### Tuesday 2026-08-11 — 10 transactions (1891–1900), ALL ✅ database-verifiable timing

Note: 1891 is a Phase-1 deployment smoke-test adjustment (+0.01 lb, reason `PHASE1_PRODUCTION_SMOKE_TEST_20260811_DO_NOT_DELETE`), voided ~35 s later via `ledger_corrections` (effective status: voided). Listed for completeness; not floor activity.

#### Production (makes)

| Txn | Product (batch) | Batches | Output lot | Output qty (lb) | Operator | Entry time ✅ |
|---|---|---|---|---|---|---|
| 1892 | Batch Classic Granola #9 | 16 | 1148 `AUG 11 2026` | 5,168.0 | legacy-shared-key | 15:36:38 |
| 1898 | Batch Coconut Sweetened Flake (auto-excluded IDs: [50]) | 12 | 1154 `AUG 11 2026` | 4,795.2 | legacy-shared-key | 15:51:50 |

Ingredient lots consumed:

- **1892**: Coconut Macaroon Desiccated `25120` 200.0; Flavor – Almond `26-04-21-PARK-001` 28.8; Honey `26-07-10-DUTC-001` 25.6; Oats `26-08-04-QUAL-001` 3,600.0; Oil – Canola `26-07-24-CBSF-001` 480.0; Salt `26-04-21-ESSE-001` 27.2; Sugar – Light Brown `26-07-09-JACK-001` 800.0
- **1898**: Coconut Flake Desiccated `6013` 2,400.0; Corn Starch `26-04-21-ESSE-002` 60.0; Glycol `26-04-24-TILL-001` 84.0; Salt `26-04-21-ESSE-001` 30.0; Sugar – 6X `26-06-10-NEWE-001` 1,320.0

#### Packing / inventory movements

| Txn | Packed | Qty (lb) | From lot | Into lot | Entry time ✅ |
|---|---|---|---|---|---|
| 1893 | 420 cases CQ Granola 10 LB | 4,200.0 | Batch Classic #9 lot 1143 `AUG 10 2026` | 1149 `AUG 10 2026` (new) | 15:39:34 |
| 1894 | 96 cases CQ Granola 10 LB | 960.0 | Batch Classic #9 lot 1143 `AUG 10 2026` | 1139 `AUG 04 2026` (pre-existing) | 15:41:15 |
| 1895 | 44 cases CQ Granola 10 LB | 440.0 | Batch Classic #9 lot 912 `JUN 30 2026` | 1139 `AUG 04 2026` (pre-existing) | 15:42:18 |
| 1896 | 11 cases Granola Crunchy CNS 10 LB Case | 110.0 | Batch Classic #9 lot 1148 `AUG 11 2026` | 1152 `AUG 10 2026` (new) | 15:44:16 |
| 1897 | 10 cases Coconut Toasted Swt Flake CNS 10 LB | 100.0 | Batch Coconut Toasted lot 518 `MAY 01 2026` | 1153 `JUL 27 2026` (new) | 15:45:22 |
| 1899 | 126 cases CQ Coconut Sweetened Flake 10 LB | 1,260.0 | Batch Coconut lot 1154 `AUG 11 2026` | 1147 `AUG 10 2026` (pre-existing) | 16:17:30 |
| 1900 | 314 cases CQ Coconut Sweetened Flake 10 LB | 3,140.0 | Batch Coconut lot 1154 `AUG 11 2026` | 1156 `AUG 11 2026` (new) | 16:18:22 |

#### Shipments / dispatch
None.

#### Receipts
None.

#### Entry-timing table ✅

| Txn | Type | created_at (ET) | Same day as business_date? | After 6 PM? |
|---|---|---|---|---|
| 1891 | adjust (smoke test, voided) | 10:35:07 | yes | no |
| 1892 | make | 15:36:38 | yes | no |
| 1893–1897 | pack | 15:39:34 – 15:45:22 | yes | no |
| 1898 | make | 15:51:50 | yes | no |
| 1899–1900 | pack | 16:17:30 – 16:18:22 | yes | no |

---

### Wednesday 2026-08-12 — 13 transactions (1901–1913), ALL ✅ database-verifiable timing

#### Production (makes)

| Txn | Product (batch) | Batches | Output lot | Output qty (lb) | Operator | Entry time ✅ |
|---|---|---|---|---|---|---|
| 1904 | Batch Coconut Sweetened Flake (auto-excluded IDs: [50]) | 12 | 1157 `AUG 12 2026` | 4,795.2 | legacy-shared-key | 16:35:47 |
| 1905 | Batch Classic Granola #9 | 18 | 1158 `AUG 12 2026` | 5,814.0 | legacy-shared-key | 16:36:21 |

Ingredient lots consumed:

- **1904**: Coconut Flake Desiccated `6013` 2,400.0; Corn Starch `26-04-21-ESSE-002` 60.0; Glycol `26-04-24-TILL-001` 84.0; Salt `26-04-21-ESSE-001` 30.0; Sugar – 6X `26-06-10-NEWE-001` 1,320.0
- **1905**: Coconut Macaroon Desiccated `25120` 225.0; Flavor – Almond `26-04-21-PARK-001` 32.4; Honey `26-07-10-DUTC-001` 28.8; Oats `26-08-04-QUAL-001` 4,050.0; Oil – Canola `26-07-24-CBSF-001` 540.0; Salt `26-04-21-ESSE-001` 30.6; Sugar – Light Brown `26-07-09-JACK-001` 410.0 + `26-08-04-JACK-001` 490.0

#### Packing / inventory movements (including the corrected-pack sequence)

| Txn | Type | Detail | Qty (lb) | Lots | Entry time ✅ |
|---|---|---|---|---|---|
| 1906 | pack | 459 cases Granola SS Chocolate Chip 12x10 OZ Case | 3,442.5 | from Batch SS Choc Chip #2 `JUN 23 2026` 957.0 + `JUL 08 2026` 2,485.5 → new lot 1159 `BB081027` | 16:38:19 |
| 1907 | pack | 63 cases **Granola SS Classic #9 25 LB** (product 286) | 1,575.0 | from Batch Classic #9 lot 1148 `AUG 11 2026` → new lot 1160 `AUG 11 2026` | 16:41:12 |
| 1908 | pack | 106 cases CQ Coconut Sweetened Flake 10 LB | 1,060.0 | from batch lot 1157 `AUG 12 2026` → lot 1156 `AUG 11 2026` (pre-existing) | 16:43:54 |
| 1909 | pack | 335 cases CQ Coconut Sweetened Flake 10 LB | 3,350.0 | from batch lot 1157 `AUG 12 2026` → new lot 1162 `AUG 12 2026` | 16:44:48 |
| 1910 | adjust | −1,575.0 on product 286 lot 1160 — reason: “The system assigned the wrong output product.” | −1,575.0 | lot 1160 | 16:50:03 |
| 1911 | adjust | +1,575.0 on Batch Classic #9 lot 1148 — reason: “Restore source batch consumed by incorrect output product assignment.” | +1,575.0 | lot 1148 | 16:50:09 |
| 1912 | pack | 63 cases **Granola Classic 25 LB** (product 136) — re-entry of the corrected pack | 1,575.0 | from lot 1148 `AUG 11 2026` → new lot 1163 `AUG 11 2026` | 16:50:16 |

#### Shipments / dispatch (feeds 1b)

| Shipment | Order | Customer | Contents (product / qty lb / source lots) | Dispatch time ✅ | Proof fields |
|---|---|---|---|---|---|
| 325 | SO-260729-001 | Inter-County Bakers, Inc. | Coconut Sweetened Flake CNS 25 LB / 500 / `JUL 27 2026` 450 + `AUG 06 2026` 50 (txn 1901) • Granola Crunchy CNS 10 LB Case / 400 / `JUL 07 2026` 200 + `AUG 03 2026` 200 (1902) • Kookies & Kreme – 25 LB / 1,500 / `26-06-02-CREA-001` (1903) | 14:45 | none (absent from schema) |
| 326 | SO-260812-002 | Grassland Food & Snacks LLC | Granola Classic 25 LB / 1,575 / `JUL 21 2026` 625 + `AUG 03 2026` 750 + `AUG 11 2026` (lot 1163) 200 (txn 1913) | 16:50 | none (absent from schema) |

#### Receipts
None.

#### Entry-timing table ✅

| Txn | Type | created_at (ET) | Same day as business_date? | After 6 PM? |
|---|---|---|---|---|
| 1901–1903 | ship | 14:45:39 – 14:45:41 | yes | no |
| 1904–1905 | make | 16:35:47 – 16:36:21 | yes | no |
| 1906–1909 | pack | 16:38:19 – 16:44:48 | yes | no |
| 1910–1911 | adjust | 16:50:03 – 16:50:09 | yes | no |
| 1912 | pack | 16:50:16 | yes | no |
| 1913 | ship | 16:50:40 | yes | no |

---

### Thursday 2026-08-13 (partial — through 14:35 ET) — 3 transactions (1914–1916), ALL ✅

#### Production (makes)
None as of run time.

#### Packing / inventory movements
None as of run time.

#### Shipments / dispatch (feeds 1b)

| Shipment | Order | Customer | Contents (product / qty lb / source lots) | Dispatch time ✅ | Proof fields |
|---|---|---|---|---|---|
| 327 | SO-260721-002 | Kohl Wholesale Dist. | Sprinkles Rainbow 10 LB / 1,980 / `261212585122` 360 + `261222585122` 1,620 (txn 1914) • Coconut Sweetened Flake UNIPRO 10 LB / 400 / `AUG 03 2026` (1915) • Coconut Sweetened Medium UNIPRO 10 LB / 400 / `AUG 03 2026` (1916) | 08:53 | none (absent from schema) |

#### Receipts
None.

#### Entry-timing table ✅

| Txn | Type | created_at (ET) | Same day as business_date? | After 6 PM? |
|---|---|---|---|---|
| 1914–1916 | ship | 08:53:48 – 08:53:50 | yes | no |

---

## 3. Timing analysis

**Late-entry query** (created_at NY-date ≠ business_date, OR created_at NY-time > 18:00, guarded on `created_at_source='database'` — same logic as the Daily Entries endpoint): **0 rows** for the window.

1. **Entries after 6:00 PM on their production day:** none among verifiable (database-source) entries, Tue–Thu. For Monday 8/10, the legacy (unverifiable) timestamps also all fall before 6 PM ET (latest: 16:13); presented for completeness, not as verified fact.
2. **Entries created on a different day than the work they describe:** none among verifiable entries — every database-source transaction's `created_at` NY calendar date equals its `business_date`. Monday 8/10 entries cannot be assessed on `created_at` (backfilled); their legacy timestamps indicate same-day entry but are mutable and unverifiable.
3. **Correction/modification records:** exactly one `ledger_corrections` row in the window — a `void` of txn 1891 (the Phase-1 deployment smoke-test adjustment), reason `PHASE1_PRODUCTION_SMOKE_CLEANUP_20260811_PRESERVE_VOIDED_EVIDENCE`, operator `legacy-shared-key`, at 2026-08-11 10:35:42 ET. No corrections target any floor entry. Note: the 8/12 wrong-product pack was corrected via paired `adjust` transactions (1910/1911) + re-pack (1912), not via `ledger_corrections` (see §4.6).

Verifiability boundary: the append-only ledger and DB-generated `created_at` became active in prod at 2026-08-11 10:32–10:35 ET. Entry times are independently verifiable from that moment forward only.

---

## 4. Gaps & anomalies (flags only — no conclusions)

1. **No `receive` transactions the entire week.** Last receive in the system is txn 1862, business_date 2026-08-06. Whether deliveries occurred this week is not determinable from the DB.
2. **No production or packing entries for Thursday 8/13 as of the 14:35 ET run.** (Partial day; ships were entered at 08:53.)
3. **No ship transactions on Tuesday 8/11.** (Other window days each have ships.)
4. **Certifications table contains zero real evening-certification records for Week 1.** The only rows in `certifications` are two smoke-test artifacts self-labeled TEST ONLY with business_date 1900-01-01 (created 2026-08-11 during Phase-1 deployment verification). No certification-linked snapshot exists for the evenings of 8/10, 8/11, or 8/12.
5. **No per-operator identity in use.** Every database-source entry has `operator_id='legacy-shared-key'`; all backfilled entries are `legacy-unattributed`. The schema supports per-operator identity but no distinct operator identities appear anywhere in the window.
6. **Wed 8/12 wrong-output-product sequence (raw record):** 16:41 pack 1907 recorded 1,575 lb as product 286 “Granola SS Classic #9 25 LB” (a product whose record carries kosher-ignition `verification_notes`); 16:50 adjust 1910 (−1,575, reason “The system assigned the wrong output product.”) and adjust 1911 (+1,575 batch restore); 16:50 pack 1912 re-entered the 1,575 lb as product 136 “Granola Classic 25 LB”; 16:50 ship 1913 dispatched product 136 to Grassland (200 lb of it from the re-packed lot 1163). Elapsed time from wrong entry to correction: ~9 minutes. Lot 1160 (product 286) nets to 0.
7. **Dispatch proof fields absent from schema.** No proof/photo/signature/POD/attachment columns exist in any table. On all 16 ship transactions this week, `bol_reference`, `shipper_name`, and `shipper_code` are NULL; `shipments.transaction_id` is NULL on all 5 shipment headers (lines link via `shipment_lines.transaction_id`). Dispatch evidence is limited to the ledger entries themselves.
8. **Pack-output lot codes that differ from the entry date** (raw observations, no interpretation): 1893 output coded `AUG 10 2026` entered 8/11 (source batch was AUG 10); 1896 output coded `AUG 10 2026` entered 8/11 from an AUG 11 batch lot; 1897 output coded `JUL 27 2026` entered 8/11 from a MAY 01 batch lot; 1899 packed into pre-existing lot coded `AUG 10 2026`; 1908 packed into pre-existing lot coded `AUG 11 2026` from an AUG 12 batch; 1894/1895 packed into pre-existing lot coded `AUG 04 2026`; 1889 (8/10, legacy) packed into pre-existing lot coded `AUG 07 2026`. Also 1907/1912 outputs coded `AUG 11 2026` entered 8/12.
9. **Missing lot codes:** none — every transaction line in the window has a lot_id and lot code (0 NULLs).
10. **Lot ID sequence gaps:** lot IDs 1146, 1150, 1151, 1155, 1161 do not exist in `lots` (neighbors on both sides exist). Cause not determinable from the DB (sequence gaps can arise from rolled-back inserts).
11. **`production_schedule` table remains empty** (0 rows) — no schedule data exists to compare planned vs. actual.
12. **Monday 8/10 output quantity vs. Tuesday:** makes 1886 and 1892 are both “16 batches of Batch Classic Granola #9” with identical output (5,168 lb) but different ingredient-lot splits; stated as data, both days’ figures are as recorded.

---

## 5. Caveats — schema limitations affecting verifiability this week

1. **Monday 8/10 timing is not verifiable.** All 9 Monday entries predate the append-only migration (live 2026-08-11 10:32 ET). Their `created_at` is a backfill stamp; the only entry-time evidence is the legacy `timestamp` column, which was mutable under the old schema. Do not treat Monday entry times as verified.
2. **Tue 8/11 (from ~10:35 ET) through Thu 8/13 timing IS verifiable**: DB-generated `created_at` under append-only triggers; voids/amendments can only append to `ledger_corrections`.
3. **No operator attribution**: single shared API key; `operator_id` distinguishes nothing this week. Entries cannot be tied to a person from the DB.
4. **No dispatch proof capability**: the schema has no fields for BOL scans, signatures, photos, or any dispatch evidence; ship entries are self-reported.
5. **No evening certification mechanism was used**: the certifications table is live but contains only deployment test rows; there is no snapshot record certifying any evening's counts this week.
6. **Thursday is a partial day** (extraction ran 14:35 ET); Thursday absences (production/packing) may simply reflect the run time.
7. `occurred_at` currently mirrors the entry timestamp on database-source rows (not an independently captured event time); `business_date` is the authoritative work-day assignment.
