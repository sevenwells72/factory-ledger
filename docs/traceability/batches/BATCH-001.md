# Batch 001 — GS1 Global Traceability Standard

Processed: 2026-09-07. Source collection remains **open**.

Source: SRC-001, *GS1 Global Traceability Standard*, Release 2.0, ratified August 2017, supplied as a 59-page website-print PDF. All pages were read; figures and tables were visually checked. The original source is preserved with its checksum in the [source register](/Users/michaelgross/Documents/factory-ledger/docs/traceability/SOURCE_REGISTER.md).

## Batch result

| Requested change report | Result |
| --- | --- |
| New rules added | **80**, across 17 categories; permanent ID ranges below |
| Existing rules strengthened | **0** — first source batch; no pre-existing source-based rules |
| Rules merged | **0 existing IDs** — overlapping guidance was consolidated before initial IDs were assigned; for example, R10/R11 share SCAN-001, and Appendix A does not duplicate §5 rules |
| Potential contradictions/ambiguities | **12 recorded qualifications/issues**, including R21/R23 precision, hybrid/global identity, historical standards versions, and distinction between explicit GS1 minima and derived engineering requirements |
| Important material intentionally excluded | Specialized healthcare, coupon, OEM/MRO, consumer marketing, and unrelated website/contributor material; reusable identity, hierarchy, and custody lessons retained where relevant |
| Final source reconciliation | **Not performed** — awaiting the user's explicit declaration that all sources are supplied |
| Factory Ledger architecture/database audit | **Not performed** — no application behavior, schema, or compliance conclusions asserted |

## Permanent rules introduced

| Category | New IDs | Count |
| --- | --- | ---: |
| A. Traceability fundamentals | TRACE-001 through TRACE-004 | 4 |
| B. Identification | ID-001 through ID-008 | 8 |
| C. Lots and batches | LOT-001 through LOT-005 | 5 |
| D. Critical Tracking Events | CTE-001 through CTE-004 | 4 |
| E. Event data / KDEs | EVENT-001 through EVENT-005 | 5 |
| F. Receiving | REC-001 through REC-003 | 3 |
| G. Production | PROD-001 through PROD-005 | 5 |
| H. Packing / aggregation | PACK-001 through PACK-005 | 5 |
| I. Inventory / location | INV-001 through INV-004 | 4 |
| J. Shipping | SHIP-001 through SHIP-004 | 4 |
| K. Master data | MASTER-001 through MASTER-004 | 4 |
| L. Data integrity | DATA-001 through DATA-005 | 5 |
| M. Barcode / capture | SCAN-001 through SCAN-005 | 5 |
| N. Recall / queries | RECALL-001 through RECALL-005 | 5 |
| O. Data quality | QUAL-001 through QUAL-004 | 4 |
| P. Interoperability | EXCH-001 through EXCH-006 | 6 |
| Q. Additional architecture / operations | ARCH-001 through ARCH-004 | 4 |
| **Total** | | **80** |

No retired IDs or alias mappings exist in this initial batch. Future additions take the next unused ID in their category; merges must preserve retired-ID mappings rather than reassigning numbers.

## Strength and priority distribution

| Source strength | Rules |
| --- | ---: |
| GS1 requirement | 17 |
| GS1 recommendation | 5 |
| Derived architectural principle | 56 |
| Optional enhancement | 2 |

All **18 numbered source requirements** are mapped in the source register. R10/R11 retain their “should” wording and are consolidated as one recommendation; R21 is separately applied to receiving and shipping. These choices explain why source requirement and rule counts differ.

| Priority | Rules |
| --- | ---: |
| Critical | 39 |
| High | 37 |
| Medium | 1 |
| Contextual | 3 |

These priorities rank the risk/value of **proposed design obligations**, not discovered Factory Ledger defects. Actual implementation priorities require the later authorized audit and applicability assessment.

## Principal architectural outcomes

- Preserve connected, bidirectional genealogy through receipts, actual consumption, transformations, packing, effective containment, dispatch, and recipient sites.
- Retain many-to-many lot/run/output relationships, partial quantities, intermediate and rework ancestry, and all applicable packaging contributors.
- Keep product definitions, lots, serials, logistic units, reusable assets, transactions, and physical occurrences conceptually distinct.
- Preserve the histories of pallet membership, custody, disposition, master-data meaning, and corrections; current state must remain explainable.
- Match traceability claims to evidence: GTIN-only, lot-level, and serialized identity support different precision. Unknown or pooled provenance must remain explicit.
- Separate the internal architectural lessons from exact GS1 technology adoption. EPCIS compatibility, RFID, GDSN, EDI, and specialized identifiers are conditional choices.

## Main ambiguities and source limits

The full issue register is in SOURCE_REGISTER.md. The most consequential issues are:

1. R21 permits class-level receipt/shipment identification, while R23's general CTE minimum lists lot or instance identity. The proposed policy requires lot/instance precision for risk-relevant food traceability and treats class-only cases explicitly.
2. R01 permits internal IDs for intermediate products; governed hybrid deployment is discussed elsewhere. Neither establishes blanket full conformance for all-local external identities.
3. Correction history, record-entry timestamps, quantity/UOM reconciliation, formula versions, idempotency, and transaction consistency are derived engineering rules, not verbatim GS1 database requirements.
4. The standard and its reference editions date to 2017 or earlier. No claim is made that referenced technology limitations or services describe their current state.
5. Lot-level cases on an SSCC pallet are a valid illustrated model; individual-case serialization is not universally mandated. A precise case route requires adequate identifiers and historical links.
6. Response times, retention periods, process tolerances, detailed identifier allocation, and jurisdiction-specific food obligations need separately established requirements.

## Verification of this deliverable

The rulebook was checked for unique permanent IDs, all required rule fields, explicit source strength/applicability/priority, section/page references, valid cross-references, and coverage of every numbered source requirement. The source copy's hash matches the supplied PDF. The relationship contract and 14 proposed acceptance scenarios are design/audit aids, not executed system tests.

Canonical master: [Factory Ledger Traceability & Data Architecture Rulebook](/Users/michaelgross/Documents/factory-ledger/docs/traceability/FACTORY_LEDGER_TRACEABILITY_RULEBOOK.md).

For the next batch: read the new source completely; compare to this master; update existing rules where possible; add only genuinely new obligations; preserve IDs; record source-specific strength and applicability; and append a new batch report. Do not regenerate an unrelated rulebook or begin auditing the application during source collection.
