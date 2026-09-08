# Factory Ledger Traceability & Data Architecture Rulebook

Version: **1.0 — Final reconciliation**, 2026-09-08. Status: **Final for the complete SRC-001 corpus; source collection closed.**

This rulebook translates the supplied material into design rules for a food-manufacturing operational system. It describes desired behavior, not observed Factory Ledger capabilities. No codebase or database audit has been performed. Examples are hypothetical. Physical implementation choices remain open where equivalent designs preserve the required facts and relationships.

## Final Reconciliation

Completed **2026-09-08** after rereading the rulebook end to end against SOURCE_REGISTER.md and BATCH-001.md. **SRC-001 (GS1 Global Traceability Standard 2.0) is the complete corpus.** No new source, application audit, schema inspection, or assertion of Factory Ledger capabilities is part of this reconciliation.

**Final inventory: 79 active rules, 80 preserved permanent IDs, 1 merged alias, 17 categories (A–Q).** Each active rule has an explicit Rule ID, Rule, Why, Factory Ledger application, Priority, Source strength, Source, Applicability, and Audit question; examples are included where useful. The retained alias carries the same fields and resolves to its canonical obligation. Counts below exclude that alias from active rules and priorities.

| Category | Active rules | Critical | High | Medium | Contextual | Alias IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A. Traceability fundamentals | 4 | 4 | 0 | 0 | 0 | 0 |
| B. Identification and unique IDs | 8 | 3 | 5 | 0 | 0 | 0 |
| C. Lots and batches | 4 | 3 | 1 | 0 | 0 | 1 |
| D. Critical Tracking Events | 4 | 3 | 1 | 0 | 0 | 0 |
| E. Event data and KDEs | 5 | 2 | 3 | 0 | 0 | 0 |
| F. Receiving and supplier traceability | 3 | 2 | 1 | 0 | 0 | 0 |
| G. Production and transformation | 5 | 3 | 2 | 0 | 0 | 0 |
| H. Packing and aggregation | 5 | 3 | 2 | 0 | 0 | 0 |
| I. Inventory and location | 4 | 2 | 2 | 0 | 0 | 0 |
| J. Shipping and customer traceability | 4 | 2 | 2 | 0 | 0 | 0 |
| K. Master data | 4 | 1 | 3 | 0 | 0 | 0 |
| L. Data integrity and auditability | 5 | 4 | 1 | 0 | 0 | 0 |
| M. Barcode, QR, and automatic identification | 5 | 1 | 4 | 0 | 0 | 0 |
| N. Recall and traceability queries | 5 | 3 | 2 | 0 | 0 | 0 |
| O. Data quality and validation | 4 | 2 | 2 | 0 | 0 | 0 |
| P. Interoperability and external data exchange | 6 | 0 | 3 | 1 | 2 | 0 |
| Q. Additional architectural and operational principles | 4 | 0 | 3 | 0 | 1 | 0 |
| **Total** | **79** | **38** | **37** | **1** | **3** | **1** |

**Merge:** LOT-005 → PROD-002. Mixing/pooling contributor preservation now belongs to the same derived allocation obligation as other transformations; carryover, conservative attribution and evidenced boundaries are retained. LOT-005 remains in category C as a permanent alias, so historical references still resolve. No ID was removed, renumbered or reused. Other overlaps were reconciled by responsibility: TRACE-003/004 govern precision and gaps; RECALL-003 governs incident decisions; PROD-001 retains direct R24 strength while PROD-002 supplies derived mechanics. Event minima, workflow capture, consistency validation and query outcomes remain separately testable obligations. General historical-data rules and their process-specific applications remain distinct checks, not additional copies of the same rule.

**Source-strength counts (active):** 17 GS1 requirement; 5 GS1 recommendation; 55 Derived architectural principle; 2 Optional enhancement. All 18 numbered source requirements retain their direct anchors from the source register. R10/R11 remain recommendations; R21 remains applied separately to receipt and dispatch. No derived engineering choice was upgraded to a GS1 mandate.

**Coverage conclusion:** Both trace-back and trace-forward are specified across raw-material receipts, all transformation generations, intermediates/rework, packing, historical containment, dispatch and recipient sites. The two mandatory recall scenario contracts below define exact expected product, lot, supplier and customer sets, along with evidence, quantities and ambiguity handling. They establish testable rulebook coverage, **not successful execution against Factory Ledger**. Full external-chain answers still depend on available partner evidence; a gap must prevent an unsupported completeness claim.

**Source limits remain explicit:** R21/R23 precision tension, intermediate-ID exceptions, conditional GS1 exchange, historical technical versions, and absent universal retention periods, deadlines and process limits are not resolved by inventing requirements. Collection closure does not turn unsupplied referenced standards into evidence. The source register and Batch 001 are preserved as historical records; this section supersedes their provisional collection status and counts.

## Reading and maintaining the rules

Every rule has a permanent ID. “Must” expresses the proposed Factory Ledger design obligation **within the rule's applicability**; it does not by itself mean GS1 mandates that engineering implementation. Conditional capabilities activate only when their stated business context applies. The complete corpus does not establish a regulatory compliance profile for Factory Ledger.

**Source strength:** GS1 requirement = direct requirement in this source's interoperability framework; GS1 recommendation = guidance, not a mandate; Derived architectural principle = an engineering consequence or extension of cited guidance; Optional enhancement = a useful capability whose need depends on scope. Applications and examples are Factory Ledger interpretations unless explicitly stated otherwise. A citation on a derived rule identifies its conceptual basis, not an assertion that GS1 specifies every detail.

**Applicability:** A = Essential principle; B = Useful architecture; C = Interoperability consideration. D = Not relevant to the stated scope; these concepts are documented in the source register rather than imposed as rules. A rule may carry more than one classification when its internal principle and external implementation differ.

**Priority:** Critical = failure can break genealogy, omit affected product/customers, or destroy evidence; High = material operational, identification, or response risk; Medium = improves maintainability or assurance; Contextual = activated by a particular operation or exchange need. Priority is independent of source strength.

**Evidence:** SRC-001 is *GS1 Global Traceability Standard*, Release 2.0, ratified August 2017, supplied as a 59-page website-print PDF. Page references below are **PDF pages**, not pagination of a different GS1 edition. Section and R-number references are included for stability. See [source register](/Users/michaelgross/Documents/factory-ledger/docs/traceability/SOURCE_REGISTER.md) for provenance, coverage, exclusions, and unresolved interpretations. This rulebook does not claim to verify current versions of the standards referenced by the 2017 document.

SRC-001 is the complete source corpus. No further batches are expected. The source register and BATCH-001 remain historical provenance records; their open-collection and reconciliation-pending statements are superseded by this final version. Standards cited by SRC-001 but not supplied are outside the corpus. They cannot supply missing requirements or resolve source ambiguities by implication. Retain all permanent IDs, including merge aliases; never reuse them.

**Testability and applicability:** The Rule, Factory Ledger application, and Audit question together define each active obligation. “Relevant,” “required,” “where appropriate,” and conditional features must resolve to a documented scope/profile decision under TRACE-001, with an owner and rationale; they are not permission to silently omit data. Before assessing a conditional rule, establish whether its trigger applies. For time targets, retention, review cadence, measurement tolerances, release criteria, supported encodings, and access permissions, require an explicit policy value or decision and evidence of its application. SRC-001 does not supply universal values. An absent required policy or unavailable evidence yields unknown/incomplete, not a pass. Examples illustrate the obligation without prescribing a storage technology or asserting existing functionality.

## A. Traceability fundamentals

<a id="trace-001"></a>

### TRACE-001 — Define the traceability boundary and required precision

**Rule ID:** TRACE-001.

**Rule:** Maintain an explicit scope covering traceable objects, materials, packaging, locations, parties, lifecycle steps, upstream/downstream tiers, and required identification precision.

**Why:** Completeness cannot be measured against an undefined population; omitted materials can become recall blind spots.

**Factory Ledger application:** Maintain a scope matrix for ingredients, intermediate and finished goods, primary/secondary packaging, cases, pallets, outsourced operations, and relevant equipment. Document risk-based exclusions and partner needs. Do not assume all secondary packaging is irrelevant because the source's tuna example tracks cartons only at class level.

**Priority:** Critical. **Source strength:** GS1 recommendation. **Applicability:** A.

**Source:** SRC-001 §§2.2, 3.4, 5.1, Appendix C.3 steps 1–5; pp. 6, 17, 34–35, 48–49.

**Audit question:** Is every in-scope material and workflow assigned an explicit traceability level and boundary, with exclusions justified?

<a id="trace-002"></a>

### TRACE-002 — Preserve one connected, bidirectionally traversable history

**Rule ID:** TRACE-002.

**Rule:** Preserve links across internal processing and external handoffs so each in-scope output can be traced upstream and each input can be tracked downstream.

**Why:** A lot register without its connecting relationships cannot identify affected production or recipients.

**Factory Ledger application:** Connect supplier/source location, receipt, input lot, consumption, production, output lot, packing, containment, shipment, and receiving party/location. Maintain reverse traversal over the same evidence rather than separately maintained forward and backward narratives. One-up/one-down is an external baseline, not permission to omit internal processing.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A.

**Source:** SRC-001 §§3.1–3.2, 3.5; R21, R24–R25; pp. 11–12, 18–20, 37.

**Audit question:** Can every link in a finished-lot-to-supplier path also be traversed from the supplier lot toward affected shipments?

<a id="trace-003"></a>

### TRACE-003 — Make traceability claims match recorded precision

**Rule ID:** TRACE-003.

**Rule:** Report no finer identity, route, or location certainty than the identification and event evidence support.

**Why:** GTIN-only data cannot distinguish lots, and lot-level data cannot distinguish otherwise identical cases within one lot.

**Factory Ledger application:** Record precision per object and data source. A lot distributed to three locations has three quantity positions; it does not have one unique current location. A serial can support an individual route only where observations and containment links support it. Distinguish recorded observations from inferred positions.

**Example:** A complaint with only product plus lot may implicate every case of that lot; a particular pallet cannot be selected merely because it was the latest shipment.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A.

**Source:** SRC-001 §§2.2, 3.3.2, 4.1.1, Tables 3-1/4-2; pp. 6, 13–14, 22–23.

**Audit question:** Do queries expose ambiguous identities and multiple possible paths instead of choosing an unsupported unique answer?

<a id="trace-004"></a>

### TRACE-004 — Distinguish known endpoints from missing traceability

**Rule ID:** TRACE-004.

**Rule:** Represent unknown identity, missing events, unavailable partner data, and intentional scope boundaries explicitly.

**Why:** An empty query result can otherwise be mistaken for proof that no affected product or customer exists.

**Factory Ledger application:** Mark unresolved links and the last verified observation. Distinguish unknown final consumers from a missing business recipient. Preserve a route to request further information from direct partners when deeper supply-chain information is unavailable.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A.

**Source:** SRC-001 §§3.2, 3.3.4–3.3.5, 4.3.3, §6 traceability party; pp. 12, 15–16, 31–32, 40.

**Audit question:** Can a recall report distinguish a verified endpoint, an intentional boundary, and an unresolved gap?

## B. Identification and unique IDs

<a id="id-001"></a>

### ID-001 — Give entities durable, typed identities

**Rule ID:** ID-001.

**Rule:** Give each entity a stable internal identity and distinguish its type from its names, labels, and external identifiers.

**Why:** Renaming an ingredient or customer must not sever history; identical strings in different schemes need not identify the same thing.

**Factory Ledger application:** Distinguish products, lots, serialized objects, logistic units, assets, parties, locations, documents, runs, and events. Store identifier scheme and issuer/namespace alongside external values. Prevent identifier reassignment from making historical references resolve to a different entity.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §§2.2, 4.1.1–4.1.4, R01–R06; pp. 5–6, 21–25, 35–36.

**Audit question:** Can two unrelated objects share a displayed code without being merged, and do references survive a name change?

<a id="id-002"></a>

### ID-002 — Support globally unique identifiers for in-scope traceable objects

**Rule ID:** ID-002.

**Rule:** For the GS1 interoperability profile, identify each created or managed traceable object globally and maintain its associated master data; intermediate products may use internal identifiers.

**Why:** Trading partners must resolve the same identity without guessing from local names.

**Factory Ledger application:** Support GTIN for trade-item types, SSCC for logistic-unit instances, and appropriate asset keys when relevant. The R01 intermediate-product exception permits internal work-in-process IDs but does not excuse broken input/output links. If non-intermediate objects use local IDs in a hybrid deployment, document that interoperability limitation rather than claiming full R01 coverage.

**Priority:** High. **Source strength:** GS1 requirement. **Applicability:** C; internal identity principle A.

**Source:** SRC-001 §5.2 R01; §2.5; pp. 10, 35.

**Audit question:** What percentage of in-scope objects meets R01, and are intermediate exceptions and other deviations distinguished?

<a id="id-003"></a>

### ID-003 — Retain identifiers assigned by other parties

**Rule ID:** ID-003.

**Rule:** Record externally assigned object identifiers in captured traceability data and retrieve relevant associated master data where appropriate.

**Why:** Replacing a supplier's identifier with an unlinked local code breaks cross-company tracing.

**Factory Ledger application:** Keep supplier product identifiers, lot codes, and received SSCCs alongside internal identities. Preserve their issuer and original representation. An internal label may add a local identifier but must retain a resolvable mapping to the incoming object. Conflicting or ambiguous identifiers require resolution, not silent consolidation.

**Priority:** Critical. **Source strength:** GS1 requirement. **Applicability:** A/C.

**Source:** SRC-001 §5.2 R02; pp. 35; §4.1.1 pp. 21–23.

**Audit question:** Can a supplier's original product/lot or SSCC locate all associated receipts and downstream uses?

<a id="id-004"></a>

### ID-004 — Select identification granularity to meet traceability needs

**Rule ID:** ID-004.

**Rule:** Identify produced, managed, or sold objects at the level needed by relevant parties across the supply chain.

**Why:** Insufficient precision cannot be recovered by adding more descriptive fields later.

**Factory Ledger application:** Require at least lot-level identification for risk-relevant food materials under the agreed scope; serialize where individual-object tracing is required. Preserve lot associations even when serials exist if partners need lot-level handling. GTIN alone identifies a product type, not a particular physical unit.

**Priority:** Critical. **Source strength:** GS1 requirement. **Applicability:** A/C.

**Source:** SRC-001 R03; §§2.2, 4.1.1; pp. 6, 22–23, 35.

**Audit question:** Does every object meet its documented precision requirement through every relevant handoff?

<a id="id-005"></a>

### ID-005 — Identify parties independently of their roles

**Rule ID:** ID-005.

**Rule:** For GS1 interoperability, assign or retain globally unique party identifiers and associated master data.

**Why:** A broker, manufacturer, carrier, customer, and site operator can be different parties; one party can play several roles.

**Factory Ledger application:** Support GLN-based party identity and separate role assignments. Do not use a customer shipping address as the party's sole identity. Local party keys may remain internal implementation details, with explicit mappings to external identities.

**Priority:** High. **Source strength:** GS1 requirement. **Applicability:** C; party/role distinction A.

**Source:** SRC-001 §4.1.2; R04; §6 party; pp. 24, 35, 39.

**Audit question:** Can the same party be a supplier and customer without duplicate identities, and can distinct participants in one shipment be identified?

<a id="id-006"></a>

### ID-006 — Identify physical locations independently of parties

**Rule ID:** ID-006.

**Rule:** For GS1 interoperability, use globally unique identifiers for managed physical locations and maintain their master data.

**Why:** Knowing who owns goods does not establish where they were received, processed, or stored.

**Factory Ledger application:** Support site GLNs and finer internal location identities, including GLN extensions where applicable. Keep the location's identity separate from the operator's identity even when the same organization manages it. Include coordinates when useful for locations outside an ordinary facility hierarchy.

**Priority:** High. **Source strength:** GS1 requirement. **Applicability:** C; distinct location identity A.

**Source:** SRC-001 §4.1.3, Table 4-4; R05; pp. 24–25, 36.

**Audit question:** Do event locations resolve to an identified physical area rather than only an organization or free-text address?

<a id="id-007"></a>

### ID-007 — Make shared document and transaction references unambiguous

**Rule ID:** ID-007.

**Rule:** Use globally unique identifiers for documents or transactions created and shared with other parties, and retain the associated data.

**Why:** A reference such as “invoice 1001” may occur at many companies.

**Factory Ledger application:** Keep document type, issuing party, identifier, and associated record; support GDTI where appropriate. Connect receipt, purchase-order, shipment, sales-order, certificate, and exchange-message references to relevant events without making a document number the physical object's identity.

**Priority:** High. **Source strength:** GS1 requirement. **Applicability:** C.

**Source:** SRC-001 §4.1.4; R06; pp. 25, 36.

**Audit question:** Can a recipient resolve a shared reference to exactly one issuing party's document and its retained data?

<a id="id-008"></a>

### ID-008 — Govern identifier issuance and lifecycle

**Rule ID:** ID-008.

**Rule:** Define who may issue, associate, replace, or retire each identifier scheme, and preserve historical assignments.

**Why:** Uncontrolled issuance creates collisions and can overwrite identity history.

**Factory Ledger application:** Distinguish brand-owner GTIN responsibility, logistic-unit builder/brand-owner SSCC responsibility, and asset-owner identifier responsibility. Record issuer and assignment evidence. Verify detailed allocation, reuse, and product-change rules against an explicitly adopted specification before implementing them; its detailed algorithms and reuse periods are outside this corpus.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/C.

**Source:** SRC-001 §5.6, Appendix B identifier responsibilities and exceptions; pp. 39, 45–47.

**Audit question:** Is every issued identifier attributable to an authorized issuer, and can retirement or relabeling occur without redirecting old history?

## C. Lots and batches

<a id="lot-001"></a>

### LOT-001 — Define lot boundaries and distinguish them from runs and receipts

**Rule ID:** LOT-001.

**Rule:** Define the conditions that create and close a lot; do not treat a production run, delivery, or date as inherently equivalent to a lot.

**Why:** These objects answer different questions and need not have one-to-one relationships.

**Factory Ledger application:** Document lot-assignment policy by material/product/process. Give runs their own identity and connect their actual outputs to lot identities. One delivery may contain several lots; a lot may arrive repeatedly. New lot boundaries require an explicit business reason and genealogy, not a cosmetic code change.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A.

**Source:** SRC-001 §§2.2, 3.3.1, 4.1.1; R24; pp. 6, 13, 22–23, 37.

**Audit question:** Are lot boundaries defined independently of receipt and run IDs, including multiple-lot and repeated-delivery cases?

<a id="lot-002"></a>

### LOT-002 — Resolve lots and serials within their proper product namespace

**Rule ID:** LOT-002.

**Rule:** Never identify a lot by bare lot text alone; resolve lot and serial identifiers with the product and applicable issuer namespace.

**Why:** Two products or suppliers can use the same printed lot code.

**Factory Ledger application:** Support the GS1 combinations GTIN plus lot and GTIN plus serial. For non-GS1 incoming material, retain supplier/issuer and product identity with the lot code. Internal surrogate keys may simplify references, but must not erase the original composite identity. Treat identifier text as text so formatting is preserved.

**Example:** Supplier A's flour lot “042” and Supplier B's oil lot “042” remain unrelated lots.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A/C.

**Source:** SRC-001 §4.1.1, Table 4-2; R02–R03; pp. 22–23, 35.

**Audit question:** Does importing two identical lot strings for different products or issuers preserve distinct identities?

<a id="lot-003"></a>

### LOT-003 — Preserve lot identity across quantity splits and repeated receipts

**Rule ID:** LOT-003.

**Rule:** Represent separate holdings and receipt contributions without silently creating or merging lot identities.

**Why:** One lot can occupy multiple bins, arrive on multiple receipts, and be consumed in many runs.

**Factory Ledger application:** Track quantity by lot and relevant location/container/state. Preserve receipt-level contributions where known. If repeated receipts of the same supplier lot are indistinguishably pooled, report that provenance at pooled precision instead of inventing which receipt supplied a particular consumption.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A.

**Source:** SRC-001 §§3.3.2, 4.1.1; R21, R24; pp. 13–14, 23, 37.

**Audit question:** Can one lot span receipts, bins, and runs without either duplicate stock or falsely precise receipt attribution?

<a id="lot-004"></a>

### LOT-004 — Preserve date meaning and lot-specific attributes

**Rule ID:** LOT-004.

**Rule:** Store production, receipt, expiry, and best-before dates as distinct concepts, retaining their source and applicable lot or instance.

**Why:** Receipt date is not production date, and different date meanings support different operational decisions.

**Factory Ledger application:** Store received date attributes as supplied; record corrections with evidence. Do not recalculate an old lot's shelf life from today's product master. Link any authorized shelf-life reassessment to its decision record. The applicable operational policy determines which dates gate use.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §3.5 Figures 3-8/3-9; §§4.2–4.2.1; Appendix C.3 step 7; pp. 18–19, 25–26, 49.

**Audit question:** Can the system distinguish the lot's original expiry from receipt time and any later authorized change?

<a id="lot-005"></a>

### LOT-005 — Retain every contributor when material is mixed or pooled

**Rule ID:** LOT-005. **Status:** Merged alias of [PROD-002](#prod-002). No independent obligation or separate compliance score.

**Rule:** Apply PROD-002 to mixing, blending, pooling, and carryover; its complete allocation and contributor-preservation rule is authoritative.

**Why:** Pooling is the same contributor-preservation problem as production allocation; separate requirements risk divergent treatment.

**Factory Ledger application:** Follow PROD-002 for identified pooling contexts, all contributor links, conservative candidate sets, and evidenced boundaries. Existing LOT-005 references resolve to that rule.

**Example:** A partially refilled tank must retain its prior and new contributors; see the acceptance check in PROD-002.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A; continuous-process detail conditional. These classifications are inherited from the surviving obligation, not counted twice.

**Source:** SRC-001 §2.3 transformation, §3.3.2, R24; pp. 7, 13–14, 37; consolidated with PROD-002.

**Audit question:** Does the PROD-002 assessment include partially refilled or pooled material wherever that workflow is in scope?

## D. Critical Tracking Events

<a id="cte-001"></a>

### CTE-001 — Record every in-scope traceability-relevant occurrence durably

**Rule ID:** CTE-001.

**Rule:** Every completed in-scope physical step or traceability-relevant state transition must produce durable evidence of what occurred.

**Why:** A current quantity or status cannot reconstruct how it arose.

**Factory Ledger application:** Define CTEs for receiving, consuming, producing, packing, moving, loading, unpacking, shipping, and relevant interventions. One business action may yield several linked facts; implementations may use domain records rather than a single universal event table. Preserve the history required by the agreed scope.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A.

**Source:** SRC-001 §§2.3, 3.1; R21–R26; pp. 7, 10–11, 37–38.

**Audit question:** Can every relevant change in stock, identity, containment, or disposition be explained by retained occurrence records?

<a id="cte-002"></a>

### CTE-002 — Record the initial creation of traceable objects

**Rule ID:** CTE-002.

**Rule:** Record every CTE in which a traceable object is initially created.

**Why:** Outputs with no origin event create disconnected genealogy.

**Factory Ledger application:** Record commissioning/creation of intermediate lots, finished lots, and logistic units. A record opened in advance is a plan until physical creation is confirmed. Link produced outputs to their transformation; creating an SSCC does not prove a pallet was physically packed.

**Priority:** Critical. **Source strength:** GS1 requirement. **Applicability:** A.

**Source:** SRC-001 R22; Appendix A; pp. 37, 44.

**Audit question:** Does every created physical object have an origin event distinguishable from a planned database record?

<a id="cte-003"></a>

### CTE-003 — Distinguish observation, transformation, and containment events

**Rule ID:** CTE-003.

**Rule:** Model an event according to its physical meaning; do not collapse all operations into an undifferentiated inventory adjustment.

**Why:** Transformation creates input/output genealogy, while aggregation preserves identifiable contents within a parent.

**Factory Ledger application:** Moving sauce totes records location change; cooking ingredients into sauce records transformation; placing cases on a pallet records containment. A packing operation may involve both output creation and aggregation and must preserve both meanings. EPCIS event categories inform this separation without requiring EPCIS storage.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §2.3; R24–R25; Appendix A event categories; pp. 7, 37, 44.

**Audit question:** Can queries tell whether materials were transformed, merely moved, or placed inside a parent?

<a id="cte-004"></a>

### CTE-004 — Include relevant exceptions and end-of-life steps

**Rule ID:** CTE-004.

**Rule:** Record additional CTEs wherever they affect in-scope traceability, including applicable disposal or destruction.

**Why:** Product remains relevant to a recall until its disposition can be explained.

**Factory Ledger application:** Include returns, sampling, waste, destruction, quarantine, release, and rework when those operations exist. Preserve identity, quantity where applicable, time, place, and responsible party. An exception workflow must not erase a lot merely because it is no longer available for sale.

**Priority:** High. **Source strength:** GS1 requirement. **Applicability:** A, activated by the operation.

**Source:** SRC-001 §1.2; R26; Appendix C.3 step 7; pp. 2, 38, 49. The named Factory Ledger event catalog is an application of the source's “where applicable” requirement.

**Audit question:** Are all applicable non-routine and terminal movements represented in the event catalog and retained history?

## E. Event data and KDEs

<a id="event-001"></a>

### EVENT-001 — Capture the minimum business context for each CTE

**Rule ID:** EVENT-001.

**Rule:** Record event date/time including time zone and UTC offset; object identity at lot or instance level; event location; business step and disposition; and responsible party when different from the location manager.

**Why:** An object ID without contextual evidence does not explain what happened.

**Factory Ledger application:** Enforce a CTE-specific required-data contract. Resolve any default responsible party from historically valid location management, not today's master data. Treat process step and disposition separately. Additional KDEs are required where needed by other rules. Document class-level scope exceptions against the R21/R23 ambiguity in the source register.

**Priority:** Critical. **Source strength:** GS1 requirement. **Applicability:** A/C.

**Source:** SRC-001 §5.4 R23; §2.3; pp. 7, 37.

**Audit question:** Can every CTE resolve all R23 fields at its event time, including the responsible-party exception?

<a id="event-002"></a>

### EVENT-002 — Separate physical occurrence time from record-entry time

**Rule ID:** EVENT-002.

**Rule:** Preserve both when an event physically occurred and when it was recorded, including corrections or late submissions.

**Why:** Entry order can differ from physical order; conflating them distorts custody and stock history.

**Factory Ledger application:** Keep occurrence time, recorded-at time, original offset, and source precision. Order trace narratives by physical evidence while retaining entry chronology for audit. Unknown times remain explicit; do not substitute upload time as an asserted physical fact. Flag conflicts caused by late events.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §§2.3, 3.3.4; R23; pp. 7, 15, 37. R23 requires event time; the separate entry timestamp is derived.

**Audit question:** Does a receiving event entered the next day retain both timestamps without silently rewriting the physical timeline?

<a id="event-003"></a>

### EVENT-003 — Preserve event participants and their roles

**Rule ID:** EVENT-003.

**Rule:** Associate each event with its relevant parties and role-specific responsibilities, and record the operator or originating system where applicable.

**Why:** Site manager, physical handler, owner, and person entering data need not be the same actor.

**Factory Ledger application:** Model multiple event participants instead of one overloaded supplier/customer field. Retain operator attribution separately from the responsible organization and record a system identity for automated capture.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §§2.3, 4.1.2, 4.2; R23; Appendix B; pp. 7, 24–25, 37, 46.

**Audit question:** Can an event identify the carrier, responsible site operator, owner when relevant, and data-entry actor without substituting one for another?

<a id="event-004"></a>

### EVENT-004 — Qualify every quantity by object, role, and unit

**Rule ID:** EVENT-004.

**Rule:** Quantity-bearing event lines must identify the object, quantity, unit of measure, and business role of the amount.

**Why:** “20” is meaningless without knowing whether it means kilograms consumed, cases packed, or units shipped.

**Factory Ledger application:** Distinguish planned, measured, accepted, rejected, consumed, and produced quantities. Preserve original units and governed conversion factors; record actual weights for variable-measure goods when needed. Avoid incompatible summations, silent rounding, and unexplained negative amounts. Define precision and tolerances by process.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A.

**Source:** SRC-001 §3.5 Figures 3-8/3-9; §4.1.1; pp. 18–19, 23. Quantities appear in examples; a universal UOM schema is not specified by R23.

**Audit question:** Can every inventory-affecting amount be interpreted and reconciled in compatible units without relying on a mutable default?

<a id="event-005"></a>

### EVENT-005 — Link events to transactions without confusing intent and execution

**Rule ID:** EVENT-005.

**Rule:** Preserve references between events and business documents while keeping planned or commercial activity distinct from physical completion.

**Why:** An order or invoice alone does not prove the lot actually received or shipped.

**Factory Ledger application:** Link receipts to purchase-order lines and shipments to sales-order lines using actual fulfillment relationships. Permit partial and combined fulfillment. Cancellation of an order must not delete the event evidence of goods already handled.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §3.3.1; §4.3.1; pp. 13, 27–28.

**Audit question:** Can one order line be fulfilled by several physical events, with each event's actual identities and quantities retained?

## F. Receiving and supplier traceability

<a id="rec-001"></a>

### REC-001 — Record each actual receipt and its source

**Rule ID:** REC-001.

**Rule:** Record every in-scope receipt CTE with traceable object identity, source party and source location, and receipt date.

**Why:** The supplier relationship alone does not identify which material physically arrived from where.

**Factory Ledger application:** Receipt lines must resolve the actual ingredient or packaging lots, quantities/units, received logistic units where relevant, and receiving location. Use EVENT-001 for the fuller event context. Preserve separate deliveries even when they concern the same supplier lot. Apply the agreed precision profile rather than downgrading risk-relevant materials to GTIN-only capture.

**Priority:** Critical. **Source strength:** GS1 requirement. **Applicability:** A/C.

**Source:** SRC-001 R21; §3.5 receiving example; pp. 19, 37.

**Audit question:** Can each received lot quantity be linked to its actual receipt, supplying party, source site, and receipt date?

<a id="rec-002"></a>

### REC-002 — Reconcile received goods against expectations before release

**Rule ID:** REC-002.

**Rule:** Compare physical receipt evidence with expected identity and quantity; preserve discrepancies and the acceptance decision.

**Why:** An advance notice can describe goods that were short, substituted, damaged, or never received.

**Factory Ledger application:** Compare scans and counts against purchase orders or despatch data. Preserve expected and actual lines, rejected quantities, unknown lots, and inspection results. Keep unreconciled material unavailable under the applicable hold policy until an authorized resolution is recorded.

**Example:** A notice says 20 cases of lot A, but receiving finds 18 of A and 2 of B; preserve the actual two-lot receipt and the discrepancy.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A.

**Source:** SRC-001 §3.5 receiving example; §3.3.4; pp. 15, 19. The example checks incoming goods before making them available; detailed discrepancy handling is derived.

**Audit question:** Can an incorrect advance notice be reconciled without overwriting actual observations or releasing unresolved stock?

<a id="rec-003"></a>

### REC-003 — Preserve relevant origin and third-party custody information

**Rule ID:** REC-003.

**Rule:** Distinguish the commercial supplier from the physical source, manufacturer, carrier, or third-party handler when they differ and are relevant.

**Why:** A broker's invoice address may not identify where contaminated material originated or who handled it.

**Factory Ledger application:** Attach available origin identities and supporting certificates to the receipt or lot, with provenance. Record the direct supplying party even if deeper origin is unknown. For outsourced storage or manufacture, retain the actual facility and operator, not just an internal “external” location.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A; deeper-tier information C.

**Source:** SRC-001 §§2.3, 3.2, 3.5 upstream query, 4.1.2; §5.1; pp. 7, 12, 20, 24, 34.

**Audit question:** Does a broker-supplied or externally stored lot retain the known physical source and custody parties independently of billing information?

## G. Production and transformation

<a id="prod-001"></a>

### PROD-001 — Preserve transformation input/output relationships

**Rule ID:** PROD-001.

**Rule:** Record the relationship between inputs and outputs for every transformation CTE.

**Why:** This is the essential bridge from incoming materials to affected finished products.

**Factory Ledger application:** Identify every actual contributing lot and every resulting output lot through the production/transformation context. Support multiple inputs and multiple outputs. Do not replace actual genealogy with a recipe's ingredient types or a single ingredient-lot field on finished goods.

**Example:** One sauce run consumes tomato lots T1/T2 and oil lot O1 and creates output lots S1/S2; preserve all applicable links.

**Priority:** Critical. **Source strength:** GS1 requirement. **Applicability:** A.

**Source:** SRC-001 R24; §2.3; §3.5 manufacturing; pp. 7, 18–19, 37.

**Audit question:** For every finished production lot, can Factory Ledger identify every ingredient lot actually consumed in producing it?

<a id="prod-002"></a>

### PROD-002 — Record actual many-to-many consumption and production allocations

**Rule ID:** PROD-002.

**Rule:** Preserve every actual input contributor and output relationship, with individual consumption/output quantities and units, for transformations including mixing, blending, and pooling. Permit many lots per process and many processes per lot; retain uncertainty where exact allocations are not evidenced.

**Why:** One ingredient lot commonly feeds several runs, while a run draws from several lots of the same ingredient.

**Factory Ledger application:** Retain repeated additions as distinguishable consumption facts linked to an identified run/process or pooling context and its output. Record partial withdrawals and outputs over time. Where output-specific allocation is known, preserve it; otherwise retain the conservative candidate contributor set and applicable time interval. Carryover in tanks or continuous processes must preserve prior and new contributors until an evidenced boundary excludes them. Do not assume empty-to-empty boundaries or infer exact contaminated fractions from proportional bookkeeping. This rule incorporates former LOT-005; PROD-001 remains the direct R24 relationship requirement, while these allocation mechanics are derived.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A.

**Source:** SRC-001 §2.3 transformation, §3.3.2, R24; §3.5 Figure 3-8; pp. 7, 13–14, 18, 37.

**Audit question:** Can two lots feed one run, either lot feed another run, and a partially refilled vessel retain both prior and new contributors, with quantities and uncertain output attribution preserved?

<a id="prod-003"></a>

### PROD-003 — Carry genealogy through intermediates and rework

**Rule ID:** PROD-003.

**Rule:** Preserve identifiable intermediate and reworked material across every subsequent transformation.

**Why:** Reusing an earlier output can carry a problem into production long after the original run.

**Factory Ledger application:** Link intermediate outputs to their later consumption; permit internal IDs under R01's exception. Treat rework as a traceable input with its existing ancestry. Returning unused untransformed material to storage must be distinguishable from creating a new transformed output. Record successive occurrences so rework cannot become an unexplained circular link.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A; rework branch conditional.

**Source:** SRC-001 §1.2; R01 exception, R24; pp. 2, 35, 37.

**Audit question:** Can a recalled input be followed through an intermediate, a finished lot reused as rework, and every later output?

<a id="prod-004"></a>

### PROD-004 — Preserve the process specification used by each run

**Rule ID:** PROD-004.

**Rule:** Associate production with the historical formula/recipe and relevant process specification actually used, independently of actual consumption evidence.

**Why:** A current recipe cannot explain earlier production or prove which lots were consumed.

**Factory Ledger application:** Retain a version or snapshot of the authorized formula, expected yields, and relevant parameters for the run. Record substitutions and deviations explicitly. A formula revision must not change the description of old runs. Actual lot consumption remains authoritative for genealogy.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §§3.3–3.3.1, 3.3.4; R24; pp. 12–15, 37. Recipe versioning is not explicitly mandated in GTS.

**Audit question:** After a recipe changes, can an old run still show the specification it used and its actual substitutions and consumed lots?

<a id="prod-005"></a>

### PROD-005 — Reconcile transformation quantities with justified process effects

**Rule ID:** PROD-005.

**Rule:** Reconcile input, output, loss, waste, samples, and retained work-in-process using compatible measurement bases and documented tolerances.

**Why:** Unexplained quantity gaps can conceal missing outputs or missing ingredient contributions.

**Factory Ledger application:** Compare actual amounts using appropriate mass/count conversions. Account for evaporation, added water, by-products, and other relevant effects rather than forcing unlike amounts to balance. Preserve measured versus estimated values and investigation of excessive variance. Do not require exact kilogram-to-case equality without a supported conversion.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §3.3.4; §3.5 Figure 3-8; R24; Appendix D fresh-food waste; pp. 15, 18, 37, 51.

**Audit question:** Does each closed run reconcile on a valid measurement basis, with unexplained variance visible and resolved under policy?

## H. Packing and aggregation

<a id="pack-001"></a>

### PACK-001 — Preserve each parent/child containment level

**Rule ID:** PACK-001.

**Rule:** Record parent/child relationships for every aggregation and disaggregation CTE at each containment level.

**Why:** A pallet ID cannot identify its affected contents without those links.

**Factory Ledger application:** Preserve lot/units-to-case, case-or-lot-quantity-to-pallet, and pallet-to-transport associations at the chosen precision. Unpacking must record what was removed. Do not skip an operationally relevant level merely because the shipping screen displays only pallets.

**Priority:** Critical. **Source strength:** GS1 requirement. **Applicability:** A/C.

**Source:** SRC-001 R25; §4.1.1 Figure 4-1; §3.5; pp. 18–19, 23–24, 37.

**Audit question:** Can a shipped pallet's contents be traced down through every recorded containment level, including prior disaggregation?

<a id="pack-002"></a>

### PACK-002 — Make containment historical and enforce physical consistency

**Rule ID:** PACK-002.

**Rule:** Preserve when containment starts and ends, and prevent physically impossible active relationships at the represented precision.

**Why:** A mutable current pallet ID erases repacking history and can misidentify an earlier shipment's contents.

**Factory Ledger application:** Retain pack, unpack, and repack events. A uniquely identified case has at most one immediate physical parent at a time; a lot may legitimately be split among many parents with allocated quantities. Prevent self-containment and cycles. Support nested ancestors without confusing them with multiple immediate parents.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 R25; §4.1.1 serial-location distinction; pp. 23–24, 37.

**Audit question:** After repalletizing a case, can the system reconstruct its old and new parents at the correct times without simultaneous contradictory membership?

<a id="pack-003"></a>

### PACK-003 — Represent serialized cases or lot-count contents honestly

**Rule ID:** PACK-003.

**Rule:** Choose and preserve the actual case-identification level; never represent a batch count as individually observed case identities.

**Why:** “20 cases of lot L” is valid lot-level evidence but does not identify case number 7's route.

**Factory Ledger application:** Store either explicit case instances or product/lot/quantity contents as appropriate. Support mixed-lot pallets and partial removals. If a case can itself contain multiple lots, record that composition or enforce a documented single-lot packing policy. Do not infer homogeneous contents from the pallet's first line.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A.

**Source:** SRC-001 §3.5 Figures 3-8/3-9; §4.1.1; R25; pp. 18–19, 22–24, 37.

**Audit question:** Can mixed-lot and partially unpacked pallets be represented without losing lot quantities or inventing case-level precision?

<a id="pack-004"></a>

### PACK-004 — Distinguish packaging product definitions from actual packed contents

**Rule ID:** PACK-004.

**Rule:** Keep packaging-level trade-item definitions and conversion relationships separate from actual packing events and their material genealogy.

**Why:** A case product definition describes a standard configuration; it does not establish what a particular case contained.

**Factory Ledger application:** Define each traded packaging configuration and relevant identifier mapping. Link bulk/finished product to actual packed output, and link risk-relevant packaging material lots to the packing process. Preserve the configuration used at the time. Do not assume every packaging change requires a new GTIN without consulting the applicable allocation rules.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B/C.

**Source:** SRC-001 §§2.2, 4.1.1; R24–R25; Appendix D retail CPG; pp. 6, 21–24, 37, 50.

**Audit question:** Can an actual packing record identify both the filled product lot and applicable packaging lots, independently of today's case definition?

<a id="pack-005"></a>

### PACK-005 — Separate reusable assets from their changing contents

**Rule ID:** PACK-005.

**Rule:** Keep a reusable pallet, tote, crate, tank, or transport asset's identity separate from the product/logistic-unit contents associated with it over time.

**Why:** Reusing an asset must not merge the histories of successive loads.

**Factory Ledger application:** Use an asset identity for the enduring physical carrier and a separate load/logistic-unit identity where appropriate. Retain timed loading and unloading associations. GS1 distinguishes GIAI/GRAI assets from GTIN/SSCC goods; exact key adoption depends on scope. Empty asset movements can be recorded when operationally relevant.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B; GS1 keys C.

**Source:** SRC-001 §4.1.1, Figure 4-1; Appendix D transport/logistics; pp. 21–24, 52.

**Audit question:** Can the same physical tote carry two successive lots without either lot inheriting the other's load history?

## I. Inventory and location

<a id="inv-001"></a>

### INV-001 — Derive current inventory from retained physical evidence

**Rule ID:** INV-001.

**Rule:** Current inventory must be explainable from retained receipts, transformations, movements, shipments, and authorized corrections.

**Why:** A mutable balance alone cannot support historical tracing or quantity reconciliation.

**Factory Ledger application:** Maintain balances by the dimensions needed for identity, location, containment, and disposition. A cached balance is acceptable if it can be reconciled to retained facts. Do not count both a pallet and its contained case quantities as separate stock. If partitioning a lot across receipts is uncertain, preserve that limitation.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §§2.3, 3.3.1–3.3.2; R21–R26; pp. 7, 13–14, 37–38. GS1 does not mandate an event-sourced database.

**Audit question:** Can present and historical inventory be reconciled to physical evidence without double-counting containment levels?

<a id="inv-002"></a>

### INV-002 — Record locations at useful physical granularity

**Rule ID:** INV-002.

**Rule:** Represent the facility and subordinate physical areas needed to answer traceability and intervention questions.

**Why:** A facility-level location may be inadequate to locate or isolate a lot spread across storage and production.

**Factory Ledger application:** Identify receiving docks, staging areas, bins, lines, tanks, and external warehouses where required by scope. Separate a scan's read point from the business location or destination when they differ. Preserve historical location names and operator relationships through MASTER-003.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §§2.3, 4.1.3, 4.2; Figures 3-8/3-9; pp. 7, 18–19, 24–25.

**Audit question:** Can an affected lot be located at the physical level required to find and isolate it, rather than only at company level?

<a id="inv-003"></a>

### INV-003 — Preserve movements, staging, and relevant in-transit intervals

**Rule ID:** INV-003.

**Rule:** Record the origin, destination, moved identity/quantity, and timing of relevant movements, including transit when departure and arrival are separate facts.

**Why:** Overwriting location loses the route; instantaneous relocation can conceal stock awaiting receipt.

**Factory Ledger application:** Connect a movement's departure and arrival while allowing partial receipt, discrepancy, or pending transit. A pallet move may imply movement of known contents only for the effective membership at that time. Retain internal staging steps when needed by the CTE scope.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §§2.3, 3.3.1, 3.5; R23, R26; pp. 7, 13, 19, 37–38.

**Audit question:** Can stock in transit or moved within a facility be distinguished from confirmed stock at its destination?

<a id="inv-004"></a>

### INV-004 — Preserve hold and release decisions separately from physical location

**Rule ID:** INV-004.

**Rule:** Record quarantine, hold, release, and other relevant disposition changes with scope, reason, time, and authorized responsibility; enforce the resulting use restrictions.

**Why:** Goods can be physically present but unavailable or unsafe to consume or ship.

**Factory Ledger application:** Apply holds to the identified lot, instance, container contents, or documented affected scope. Retain decision history and evidence. Prevent normal consumption/shipping of held material; any authorized exception must be explicit. Release cannot erase the prior hold, and a location move alone must not imply release.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A.

**Source:** SRC-001 R23 disposition; §3.5 receiving; Appendix C.3 step 7; pp. 19, 37, 49.

**Audit question:** Does held stock remain blocked through moves and repacking until an attributable release or authorized exception exists?

## J. Shipping and customer traceability

<a id="ship-001"></a>

### SHIP-001 — Record each actual shipment and destination

**Rule ID:** SHIP-001.

**Rule:** Record every in-scope shipment/despatch CTE with object identity, destination party and location, and despatch date.

**Why:** A sales total or order cannot establish which customer received an affected lot.

**Factory Ledger application:** Retain shipment lines for actual lots, quantities, cases/pallets, ship-from location, and ship-to party/site. Link order references without substituting them for fulfillment evidence. Support split shipments and multiple recipient sites for the same customer under EVENT-005.

**Priority:** Critical. **Source strength:** GS1 requirement. **Applicability:** A/C.

**Source:** SRC-001 R21; §3.5 downstream query; pp. 19–20, 37.

**Audit question:** For an affected finished lot, can every actual shipment and receiving customer site be identified?

<a id="ship-002"></a>

### SHIP-002 — Preserve the contents actually dispatched

**Rule ID:** SHIP-002.

**Rule:** Reconstruct shipment contents as they were at dispatch, independently of later container, order, or master-data changes.

**Why:** A current pallet composition is not reliable evidence of an earlier shipment.

**Factory Ledger application:** Retain the shipment's effective manifest as a snapshot or reconstructible event-linked membership. Resolve contained product/lot quantities. Preserve loading changes before dispatch and corrections after it as separate evidence. Expected contents and dispatched contents remain distinguishable.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 R21, R25; §3.5; pp. 18–20, 37.

**Audit question:** After a pallet is unpacked or rebuilt, does an earlier shipment still return the original dispatched lot composition?

<a id="ship-003"></a>

### SHIP-003 — Distinguish custody, ownership, shipment, and transport groupings

**Rule ID:** SHIP-003.

**Rule:** Preserve physical custody transfers independently of commercial ownership and of shipment/consignment groupings.

**Why:** A carrier can hold goods without owning them, and a freight consignment can contain multiple shipments.

**Factory Ledger application:** Record role-specific parties and relevant transport legs/assets. Keep ordered delivery groupings separate from carrier transport groupings. GSIN/GINC support these different groupings when needed; neither replaces the SSCC identifying a logistic unit. Track confirmation status where delivery evidence is available rather than treating dispatch as proof of arrival.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B; GSIN/GINC C.

**Source:** SRC-001 §§3.3.1, 4.1.1–4.1.2; §6 custody/ownership; Appendix B; pp. 13, 21–24, 39, 46.

**Audit question:** Can a third-party carrier transport several shipments without being recorded as their customer or owner?

<a id="ship-004"></a>

### SHIP-004 — Treat returns and redirections as new occurrences

**Rule ID:** SHIP-004.

**Rule:** Record returns, refusals, and redirections without deleting or negating the historical fact of the original dispatch.

**Why:** Returned product still reached or was exposed to an earlier custody path and may enter new workflows.

**Factory Ledger application:** Link returned quantities and identities to the original shipment when supported, inspect and disposition them, and preserve subsequent restocking, rework, or destruction. Distinguish unknown returned identity from verified identity. Financial credits do not by themselves establish physical return.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A, when these operations occur.

**Source:** SRC-001 §§1.2, 3.3.1; R21, R26; pp. 2, 13, 37–38.

**Audit question:** Can a returned lot retain its original customer exposure and its subsequent disposition without corrupting shipped or on-hand totals?

## K. Master data

<a id="master-001"></a>

### MASTER-001 — Separate master, relation, transaction, and visibility data

**Rule ID:** MASTER-001.

**Rule:** Preserve distinct meanings and ownership for descriptive master data, partner relations, business transactions, and actual event evidence.

**Why:** Each supplies different context; one cannot substitute for another.

**Factory Ledger application:** Product specifications and standard case sizes are master data; approved supplier/product/site associations are relation data; purchase orders are transactions; receipts and production are occurrences. Maintain a canonical source for shared concepts across modules while linking these data domains. Separate tables or services are options, not GS1 prescriptions.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §3.3.1, Figure 3-3; §4.3.1; pp. 12–13, 27–28.

**Audit question:** Are actual lot genealogy and fulfillment facts stored independently of mutable product definitions, supplier lists, and orders?

<a id="master-002"></a>

### MASTER-002 — Maintain time-bounded partner/product/location relationships

**Rule ID:** MASTER-002.

**Rule:** Keep supply-chain partner relation records by trade item/category or class, source/destination party, source/destination location, and validity period.

**Why:** Current partner lists cannot show which relationships were relevant in an earlier period.

**Factory Ledger application:** Cover suppliers, customers, and third-party service providers. Permit multiple partners/sites per product and changes over time. Use these relationships for preparation and validation, while relying on actual events for the path of a particular lot.

**Priority:** High. **Source strength:** GS1 requirement. **Applicability:** A/C.

**Source:** SRC-001 §5.4 R20; §3.3.1 relation data; pp. 13, 36–37.

**Audit question:** Can partner/product/site relationships be retrieved for a historical date without using today's list as proof of an actual delivery?

<a id="master-003"></a>

### MASTER-003 — Preserve historical master-data meaning

**Rule ID:** MASTER-003.

**Rule:** Historical traceability records must retain the master-data meaning relevant at the time, despite subsequent edits, merges, or deactivation.

**Why:** Renamed products, changed site operators, or revised unit conversions can alter an old record's interpretation.

**Factory Ledger application:** Use effective-dated versions, immutable references, or appropriate snapshots for material attributes, party/site details, pack sizes, and other consequential data. Retain old identifier aliases. Deactivation must not orphan historical events; entity merges require preserved mappings and evidence.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §§3.3.1, 3.3.4; R20, R31; pp. 13, 15, 36, 38. The source requires validity/accessibility, not a particular versioning model.

**Audit question:** Do old events remain correctly interpretable after a product conversion, customer name, or location operator changes?

<a id="master-004"></a>

### MASTER-004 — Link supporting evidence to its subject and validity

**Rule ID:** MASTER-004.

**Rule:** Retain relevant certificates, inspection/lab records, and other supporting evidence with their identified subject, issuer/source, date, and applicable validity or scope.

**Why:** A generic attachment cannot establish which lot, supplier site, or production period it supports.

**Factory Ledger application:** Link evidence to lots, receipts, parties/sites, runs, or events as appropriate. Keep superseded evidence retrievable. Evaluate certificate expiry or other exceptions under the applicable policy, and restrict sensitive quality data under DATA-005 and EXCH-005.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §§3.3.1, 3.3.3, 3.5 upstream query, 4.1.4, 4.4; Appendix C.3 step 7; pp. 13–15, 20, 25, 33, 49.

**Audit question:** Can a certificate or lab result be resolved to the specific subjects and period it supports, including its prior versions?

## L. Data integrity and auditability

<a id="data-001"></a>

### DATA-001 — Correct recorded facts without silently erasing history

**Rule ID:** DATA-001.

**Rule:** Corrections to confirmed traceability facts must preserve the original assertion, its correction, responsible actor, time, reason, and resulting effective interpretation.

**Why:** Silent edits can conceal earlier decisions and change recall results without explanation.

**Factory Ledger application:** Use linked amendments, reversals, or equivalent auditable version history. Distinguish a clerical correction from a real physical reversal or return. Do not hard-delete confirmed evidence within its retention period. Recalculate affected genealogy and balances under the corrected interpretation while retaining the prior one. Draft cleanup need not be treated as a physical event.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §§3.3.1, 3.3.4; R31; pp. 13, 15, 38. GTS does not prescribe append-only storage or a correction-event schema.

**Audit question:** Can a wrong lot consumption be corrected while showing both what was originally asserted and what is now considered correct?

<a id="data-002"></a>

### DATA-002 — Prevent duplicate facts from retries and repeated capture

**Rule ID:** DATA-002.

**Rule:** Distinguish a duplicate submission from a separate physical occurrence and prevent unintended duplicate effects.

**Why:** Retries or repeated scans can inflate received, consumed, or shipped quantities and distort genealogy.

**Factory Ledger application:** Give commands/events stable identities and use source-aware deduplication or idempotency controls. Retain legitimate repeated partial consumptions even if item, quantity, and timestamp resemble an earlier event. Conflicting replays must be investigated, not silently accepted or discarded.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §§3.3.4, 4.2.2; R21–R25; pp. 15, 27, 37. Idempotency is an engineering inference, not an explicit GS1 requirement here.

**Audit question:** Does replaying a receipt submission leave one physical receipt, while two genuine partial receipts remain distinct?

<a id="data-003"></a>

### DATA-003 — Keep inventory effects and traceability links consistent

**Rule ID:** DATA-003.

**Rule:** A confirmed operation must not leave durable stock effects without their corresponding identities, event evidence, and required relationships, or vice versa.

**Why:** Partial writes can create material with no origin or consume stock without identifying affected output.

**Factory Ledger application:** Use atomic transactions or a durable recoverable workflow with explicit pending states and reconciliation. Enforce referential integrity and valid relationship targets. Prevent concurrent allocation from confirming the same uniquely held stock twice. An incomplete operation must remain visible and controlled, not look successfully completed.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §3.3.4 consistency/completeness; R24–R25; pp. 15, 37.

**Audit question:** If processing fails between stock updates and genealogy recording, can the system recover without a falsely complete operation or orphaned material?

<a id="data-004"></a>

### DATA-004 — Retain usable traceability data for the required period

**Rule ID:** DATA-004.

**Rule:** Retain traceability data and keep it accessible to authorized parties for the period needed to meet relevant supply-chain requirements.

**Why:** Data that exists only in an unreadable archive cannot support a timely trace request.

**Factory Ledger application:** Define retention and archive retrieval procedures covering events, links, identifiers, master-data context, and supporting evidence. Include partners' retention/access capabilities where their data is needed. Determine periods from applicable legal and commercial requirements; SRC-001 supplies no universal number of years.

**Priority:** Critical. **Source strength:** GS1 requirement. **Applicability:** A/C.

**Source:** SRC-001 R31; §3.3; Appendix C.3 step 6; pp. 12, 38, 49.

**Audit question:** Can an authorized user retrieve and interpret a trace chain near the end of its required retention period within the required response time?

<a id="data-005"></a>

### DATA-005 — Preserve provenance and restrict changes to accountable actors

**Rule ID:** DATA-005.

**Rule:** Record where traceability assertions came from and restrict creation, correction, and sensitive access according to defined responsibilities.

**Why:** Conflicting sources and unauthorized changes undermine both operational decisions and audit confidence.

**Factory Ledger application:** Attribute records to operator/system, source document/message, import, or partner as applicable. Distinguish observed, partner-reported, estimated, and inferred data. Maintain access and change evidence at a level that supports investigations. Apply internal access boundaries to personnel, formula, quality, and commercial data rather than assuming all internal users need all data.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §§3.3.3–3.3.4, 4.2, 4.3.3; Appendix B; pp. 14–15, 25, 31–32, 45–47.

**Audit question:** For a disputed consumption or shipment, can reviewers identify its source and authorized changes without exposing unrelated sensitive data?

## M. Barcode, QR, and automatic identification

<a id="scan-001"></a>

### SCAN-001 — Use open AIDC standards at the required precision

**Rule ID:** SCAN-001.

**Rule:** Where automatic identification is required, use open standards for objects marked internally and for handling objects marked by other parties.

**Why:** A readable common carrier reduces transcription and cross-party interpretation errors.

**Factory Ledger application:** Define supported carriers and precision by object/workflow, including supplier labels. Select equipment that can read the required carrier; the source notes that 2D symbols need image-based readers. Verify supported lot/serial attributes rather than assuming any product barcode supplies them. Evaluate cost and existing equipment as part of the capture design.

**Priority:** High. **Source strength:** GS1 recommendation. **Applicability:** A/C, when AIDC is required.

**Source:** SRC-001 §5.3 R10–R11 use “should”; §§4.2.1–4.2.2; pp. 26–27, 36. Their placement in a requirements table does not turn the wording into “must.”

**Audit question:** Can each required workflow capture the necessary identity precision from both Factory Ledger and supported supplier labels?

<a id="scan-002"></a>

### SCAN-002 — Decode identity and attributes with explicit carrier semantics

**Rule ID:** SCAN-002.

**Rule:** Parse scans according to their declared scheme and supported encoding, preserving meaningful identifier and attribute boundaries.

**Why:** A GTIN, lot code, serial, and expiry date are different fields; an arbitrary QR payload is not automatically GS1 data.

**Factory Ledger application:** Retain the raw scan where useful for diagnosis plus normalized, validated fields. Preserve leading zeros and lot text. Validate checks/structure under the selected specification. Do not manufacture lot precision from a GTIN-only scan or treat a URL as proof of a physical event. Detailed Application Identifier and Digital Link encodings are outside this corpus; an implementation must name and validate its adopted specification before claiming support.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B/C.

**Source:** SRC-001 §§2.5, 4.2–4.2.1, Table 4-6; pp. 9, 25–26.

**Audit question:** Do supported scans produce the correct typed fields, while GTIN-only and unsupported payloads remain visibly limited?

<a id="scan-003"></a>

### SCAN-003 — Bind issued labels to the correct physical objects

**Rule ID:** SCAN-003.

**Rule:** Preserve and verify the association between a printed/applied label or tag and the actual object and traceability data it represents.

**Why:** Correct data attached to the wrong case produces a false trace chain.

**Factory Ledger application:** Link label issuance/application to the packing or identification workflow. Verify product/lot/date and relevant SSCC contents. Reprints must retain identity; replacements must preserve prior mappings and reasons. A prewritten RFID tag must be explicitly associated with its object. Label generation alone does not confirm production or packing.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A; RFID conditional.

**Source:** SRC-001 §4.2.1 barcode linkage, logistics labels, and prewritten RFID; p. 26.

**Audit question:** Can a label reprint or swapped label be detected without creating another physical case or silently assigning the wrong lot?

<a id="scan-004"></a>

### SCAN-004 — Preserve complete capture during label or connectivity failures

**Rule ID:** SCAN-004.

**Rule:** Provide controlled exception capture for unreadable labels, unavailable devices, or delayed synchronization, preserving required KDEs and uncertainty.

**Why:** Operational failures must not force users to invent identities or bypass traceability.

**Factory Ledger application:** Allow attributed manual entry with validation and later resolution. If offline capture is supported, preserve occurrence time, local submission identity, synchronization status, and conflict handling under DATA-002. Unknown identity must trigger a visible exception and appropriate hold rather than a guessed lot.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B; offline implementation conditional.

**Source:** SRC-001 §§3.3.4, 4.2.2; R23; pp. 15, 27, 37. GTS does not mandate offline software.

**Audit question:** Can receiving continue under an approved exception process without losing KDEs or releasing material with guessed identity?

<a id="scan-005"></a>

### SCAN-005 — Convert scans into contextual business evidence

**Rule ID:** SCAN-005.

**Rule:** An identification read must create an operational effect only through a defined business action with the necessary context.

**Why:** The same pallet can be scanned for inquiry, picking, checking, or shipping; these are different actions.

**Factory Ledger application:** Capture location/read point, actor/device, time, action, and relevant quantities. Distinguish inquiry and repeated observations from stock-changing confirmations. Validate that the action is compatible with the object's state, and apply duplicate controls to confirmations.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §§2.5, 4.2, 4.2.2; pp. 9, 25, 27.

**Audit question:** Can a pallet be scanned for lookup or verification without accidentally shipping it or duplicating an inventory movement?

## N. Recall and traceability queries

<a id="recall-001"></a>

### RECALL-001 — Support evidence-linked traceback from a customer complaint

**Rule ID:** RECALL-001.

**Rule:** Given a supported finished-product identifier, return its upstream production, material, receipt, supplier, and relevant processing/custody history with supporting evidence and limitations.

**Why:** A complaint must resolve to actual potential origins, not merely the ingredients normally in the product.

**Factory Ledger application:** Accept supported product/lot, serial, case, pallet, or shipment references together with the customer's identity and any available delivery context. Resolve every matching candidate at the narrowest evidenced precision. Traverse dispatch-time contents, packing, every transformation generation, intermediates and rework to all contributing raw-material and relevant packaging lots; then return all supported receipts, direct suppliers and source sites, and available deeper origin/custody evidence. Include dates, locations, historical process specifications and linked quality evidence where in scope. For pooled provenance, retain candidate receipts rather than selecting one arbitrarily. A product-only complaint returns candidate lots or a visible identity gap; it cannot yield an invented unique lot or case route.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A.

**Source:** SRC-001 §§3.1–3.2, 3.5 upstream query; §6 tracing; pp. 11–12, 20, 40.

**Audit question:** For the customer-case fixture below, does traceback return the complete expected raw-material lot, receipt, supplier and source-site set through every generation, plus supporting events and explicit candidate paths where case or receipt identity is ambiguous?

<a id="recall-002"></a>

### RECALL-002 — Support complete forward impact and quantity disposition

**Rule ID:** RECALL-002.

**Rule:** Given an affected input lot or other supported incident scope, return every potentially affected descendant, its quantity/status/location, and all recorded downstream shipments and recipients.

**Why:** Omitting one branch of production, repacking, or redistribution can leave affected goods in commerce.

**Factory Ledger application:** Resolve the incoming lot using product and supplier/issuer identifiers, and search all its receipts and uses across the retained scope. Traverse every consumption branch, intermediate, pooling context, rework generation, finished lot, packing output, and effective containment relationship without a fixed generation limit. Return each finished product identity and lot, the evidence path that included it, and every dispatched quantity, shipment date/reference, customer party and ship-to site. Preserve all historical recipient exposures, including returns and redirections; do not net customers out because goods came back. Distinguish confirmed receipt from dispatch/pending delivery, and retain available onward evidence with explicit partner-data gaps. Report current quantity/status/location separately from cumulative production and shipment flows, using compatible units and avoiding double-counting across generations or containment levels. Missing evidence prevents a completeness claim; an empty join is not proof of no impact.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A.

**Source:** SRC-001 §§3.2, 3.5 downstream query, 4.1.1 lot-level traceability; R24–R25; pp. 12, 20, 23, 37.

**Audit question:** For the contaminated-ingredient fixture below, does the result equal the expected finished-product/lot and customer/site sets, retain every shipment and return exposure, reconcile current dispositions, and show each evidence path and unresolved gap?

<a id="recall-003"></a>

### RECALL-003 — Preserve conservative scope and explain exclusions

**Rule ID:** RECALL-003.

**Rule:** Apply TRACE-003 precision limits and TRACE-004 gap handling to incident scope decisions: record each inclusion or exclusion, its evidence and uncertainty, and retain earlier scope decisions when new evidence changes the affected set.

**Why:** False precision can incorrectly declare product unaffected.

**Factory Ledger application:** Show confirmed and potentially affected objects separately, with unresolved links and inclusion/exclusion reasons. A broad lot-level result is appropriate when serial or time-separated allocation evidence is missing. Further investigation may narrow scope through attributable new evidence; retain earlier conclusions and their basis.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A.

**Source:** SRC-001 §§3.3.2–3.3.4, 4.1.1; pp. 13–15, 23.

**Audit question:** If a run's ingredient allocation is incomplete, does the report expose uncertainty and potential exposure rather than silently excluding the run?

<a id="recall-004"></a>

### RECALL-004 — Record intervention execution through closeout

**Rule ID:** RECALL-004.

**Rule:** Maintain a recall/intervention record linking scope decisions, accountable roles, affected objects, notifications, removal confirmations, and closeout evidence.

**Why:** Finding recipients is only the beginning; the organization needs evidence of what action followed.

**Factory Ledger application:** Preserve incident criteria and query/evidence versions, authorized initiation/approval, contacts, communication status, customer responses, quarantined/returned/destroyed quantities, unresolved actions, and closeout rationale. Do not equate notification delivery with product removal. Exact message standards are outside this corpus; procedures and any adopted message profile must be explicitly established by company policy.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B; standardized external messages C.

**Source:** SRC-001 Appendix B recall responsibilities; Appendix C.3 step 7 and C.5 step 12; pp. 47, 49–50.

**Audit question:** Can an incident show which recipients confirmed removal, what quantities remain unresolved, who approved decisions, and why it was closed?

<a id="recall-005"></a>

### RECALL-005 — Test both trace directions against operational scenarios

**Rule ID:** RECALL-005.

**Rule:** Exercise traceback and forward impact at a documented cadence and after material traceability changes, using independently specified expected results, response-time targets, and quantity reconciliation.

**Why:** Data fields can exist while real trace paths fail under split lots, rework, repacking, or archive retrieval.

**Factory Ledger application:** Run documented mock incidents, including partner participation where needed. Use both mandatory recall fixtures and the edge-case scenarios below; compare exact expected and actual identity/recipient sets as well as missing or extra paths. Record completeness, false exclusions, conservative inclusions, unresolved gaps, retrieval time, quantity reconciliation, and communication readiness. Management must set the cadence and time targets from applicable requirements; the source supplies no universal interval or deadline.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 R30; Appendix C.4 step 10 and C.5 steps 12–13; pp. 38, 49–50.

**Audit question:** Is there recent evidence that both trace directions, including archived and partner-dependent paths, meet defined targets?

## O. Data quality and validation

<a id="qual-001"></a>

### QUAL-001 — Enforce CTE-specific completeness before confirmation

**Rule ID:** QUAL-001.

**Rule:** Define required KDEs and relationships for each CTE and prevent incomplete data from being represented as a fully confirmed traceability record.

**Why:** A nominally completed event with no input lot or destination hides a critical gap.

**Factory Ledger application:** Validate required fields and relationship cardinalities by action and precision profile. Explicit exceptions may capture incomplete evidence, but must carry status, owner, and operational restrictions. Distinguish not applicable from unknown; do not use a shared “unknown lot” as a real merged identity.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §3.3.4; R21–R25; pp. 15, 37.

**Audit question:** Can production or shipping appear fully confirmed while its required genealogy or recipient fields are missing?

<a id="qual-002"></a>

### QUAL-002 — Validate semantic and physical consistency across records

**Rule ID:** QUAL-002.

**Rule:** Check that valid-looking values form a coherent physical and business history, not merely that they pass field-format validation.

**Why:** A correctly formatted lot can still be the wrong ingredient or appear in impossible locations.

**Factory Ledger application:** Validate product/lot compatibility, unit dimensions, chronology, quantity availability, object existence, and physical containment constraints. Consider late-entry evidence before declaring a chronological conflict. Distinguish a blocking impossibility from a suspicious but investigable discrepancy; retain exception decisions.

**Priority:** Critical. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §3.3.4; §4.1.1 single-instance location; pp. 15, 23.

**Audit question:** Are incompatible units, incorrect material lots, impossible active parents, and unexplained stock deficits detected across workflows?

<a id="qual-003"></a>

### QUAL-003 — Measure coverage and quality against defined denominators

**Rule ID:** QUAL-003.

**Rule:** Monitor identification, event recording, relationship completeness, and data quality relative to the documented scope.

**Why:** An unqualified “100% traceable” claim can conceal excluded facilities or missing event types.

**Factory Ledger application:** Report numerators, denominators, scope/version, and collection period for R01–R32 coverage where relevant. Assess completeness, accuracy, consistency, and temporal validity separately. If using the source's 1–5 quality scale, distinguish unknown quality, weak quality, usable without regular assurance, regularly assured, and continually monitored quality. A score must be backed by evidence.

**Priority:** High. **Source strength:** GS1 recommendation. **Applicability:** A/B/C.

**Source:** SRC-001 §3.3.4; §5 introduction/5.1 and KPI notes; pp. 15, 34–38.

**Audit question:** Do reported coverage and quality metrics disclose scope and denominator and reveal missing events rather than only invalid recorded events?

<a id="qual-004"></a>

### QUAL-004 — Turn detected exceptions into accountable action

**Rule ID:** QUAL-004.

**Rule:** Route traceability gaps and relevant condition/certificate exceptions to an accountable resolution process with retained outcomes.

**Why:** Detection alone does not prevent use of questionable material or repair missing evidence.

**Factory Ledger application:** Track missing links, identifier conflicts, expired certificates, quantity discrepancies, and available temperature exceptions with severity, owner, disposition, and closure evidence. Connect the case to affected material and hold/release decisions. Avoid silent auto-filling of missing facts merely to clear a dashboard.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §3.3.4; Appendix C.3 step 7, C.5 step 13; pp. 15, 49–50.

**Audit question:** Does each material exception have an owner and evidenced resolution, and does its status affect the relevant operational decision?

## P. Interoperability and external data exchange

<a id="exch-001"></a>

### EXCH-001 — Exchange traceability data securely within agreed timeframes

**Rule ID:** EXCH-001.

**Rule:** Enable both provision and receipt of traceability data through effective, secure mechanisms within the required timeframe.

**Why:** Complete internal history is insufficient if needed partner data cannot be obtained or communicated in time.

**Factory Ledger application:** Define response expectations, responsible contacts, permitted recipients, request/response tracking, and escalation for unavailable partners. Support the necessary direct-partner exchange even if it initially uses a controlled manual process. Do not invent a universal response deadline from this source.

**Priority:** High. **Source strength:** GS1 requirement. **Applicability:** A/C.

**Source:** SRC-001 §5.5 R30; §3.1; pp. 11, 38.

**Audit question:** Can Factory Ledger supply and obtain a usable trace response within the agreed timeframe using authorized mechanisms?

<a id="exch-002"></a>

### EXCH-002 — Use open standards for automated cross-party exchange

**Rule ID:** EXCH-002.

**Rule:** Traceability data electronically exchanged between parties for automated processing must use open data-sharing standards under the GS1 interoperability profile.

**Why:** Proprietary meanings and formats prevent reliable automated interpretation across systems.

**Factory Ledger application:** Select a supported exchange profile for the business need: master-data synchronization, transaction messages, or visibility events. Validate applicable identifiers, required semantics, versions, and partner compatibility. A custom JSON API does not become standards-compliant merely by containing a GTIN field.

**Priority:** Contextual. **Source strength:** GS1 requirement. **Applicability:** C, when automated exchange is implemented.

**Source:** SRC-001 R32; §§4.3.1–4.3.2, 4.4; pp. 27–30, 33, 38–39.

**Audit question:** Can each automated partner exchange identify and validate the open standard/profile it actually implements?

<a id="exch-003"></a>

### EXCH-003 — Separate business meaning from exchange transport

**Rule ID:** EXCH-003.

**Rule:** Keep the meaning of identifiers, quantities, relations, and events independent of their delivery channel.

**Why:** Switching from file transfer to API or subscription should not change what “received” or “lot” means.

**Factory Ledger application:** Define stable internal semantics and adapters for push, pull, or publish/subscribe as needed. Govern business-step and disposition vocabularies, with explicit mappings to external vocabularies such as CBV. Preserve schema/profile versions so historical messages remain interpretable.

**Priority:** Medium. **Source strength:** Derived architectural principle. **Applicability:** B/C.

**Source:** SRC-001 §4.3.1 separation principle, §4.3.2 EPCIS/CBV; pp. 27–30.

**Audit question:** Can the same event be shared through two supported channels without changing its identity or business meaning?

<a id="exch-004"></a>

### EXCH-004 — Govern hybrid identifiers and non-standard extensions

**Rule ID:** EXCH-004.

**Rule:** Document how internal, legacy, non-GS1, and GS1 representations map, including their limits and any non-standard extensions.

**Why:** Uncontrolled mappings and private conventions create conflicting identities and partner-specific code paths.

**Factory Ledger application:** Retain reversible identifier crosswalks where possible, explicitly describe lossy mappings, and reject ambiguous matches. Prefer established fields/vocabularies before adding private ones. Test that exchange round trips preserve required genealogy and precision. Do not force a supplier's unsupported GS1 identity into existence by inventing one.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** B/C.

**Source:** SRC-001 §2.5 hybrid implementations and extensions; §§4.3.2, 4.4; pp. 10, 28–30, 33.

**Audit question:** Can each external identity be mapped unambiguously, with any loss of precision or interoperability recorded?

<a id="exch-005"></a>

### EXCH-005 — Control discovery, trust, and disclosure by purpose and role

**Rule ID:** EXCH-005.

**Rule:** Authorize access to traceability data according to the requesting party and permissible data scope, including parties beyond direct trading relationships.

**Why:** Supply-chain membership does not automatically justify access to recipes, personnel information, or unrelated customers.

**Factory Ledger application:** Define internal/external data classifications, partner permissions, authentication, secure channels, and approved redaction. For deeper-tier queries, establish who holds the data and the trust basis before disclosure. Maintain enough disclosed identity and evidence to answer the authorized question; record withheld or unavailable portions as limitations.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B/C.

**Source:** SRC-001 §§3.3.3, 3.3.5, 4.3.3; R30 authorization KPI; pp. 14–16, 31–32, 38.

**Audit question:** Can a legitimate recall partner obtain relevant evidence without accessing another customer's unrelated shipments or internal formulations?

<a id="exch-006"></a>

### EXCH-006 — Adopt EPCIS compatibility only against a defined need

**Rule ID:** EXCH-006.

**Rule:** Preserve EPCIS-compatible architectural concepts where useful, and implement actual EPCIS/CBV exchange only when justified by a defined partner or business requirement.

**Why:** The source is technology-neutral; internal traceability does not inherently require an EPCIS server or a particular storage engine.

**Factory Ledger application:** Preserve contextual events, object/lot identity, business steps/dispositions, transformation genealogy, and aggregation relationships. Before claiming EPCIS compatibility, select and validate a specific supported standard version/profile using its actual specification. Treat centralized, networked, cumulative, and replicated sharing models as design options with different visibility and access costs; no blockchain is required.

**Priority:** Contextual. **Source strength:** Optional enhancement. **Applicability:** B/C.

**Source:** SRC-001 §§1.2, 3.3.5, 4.3.2–4.3.4, Appendix A; pp. 2, 15–16, 29–33, 44–45.

**Audit question:** Is any claimed EPCIS capability backed by an explicit use case and tested profile, rather than inferred from having an event table?

## Q. Additional architectural and operational principles

<a id="arch-001"></a>

### ARCH-001 — Assign traceability responsibilities and train operators

**Rule ID:** ARCH-001.

**Rule:** Define accountable roles, written procedures, training, and contact/escalation paths for capture, master data, interventions, and trace requests.

**Why:** A data model cannot compensate for unclear responsibility at a physical handoff.

**Factory Ledger application:** Assign ownership for identifier issuance, receiving, production declarations, packing, shipping, corrections, and data-quality resolution. Maintain current crisis contacts and decision responsibilities, including separation of recall initiation/approval where the procedure requires it. Map software permissions to those responsibilities.

**Priority:** High. **Source strength:** GS1 recommendation. **Applicability:** A.

**Source:** SRC-001 §5.6, Appendix B, Appendix C.5 step 12; pp. 39, 45–47, 50.

**Audit question:** Is each required capture and intervention step assigned to a trained role with a written procedure and usable escalation contact?

<a id="arch-002"></a>

### ARCH-002 — Keep requirements aligned with changing operations

**Rule ID:** ARCH-002.

**Rule:** Reassess traceability scope, precision, KDEs, and workflows when products, processes, facilities, partners, or applicable requirements change.

**Why:** A once-complete design can become incomplete after a new production or distribution step is introduced.

**Factory Ledger application:** Maintain versioned requirement profiles and impact review for new workflows. Use documented gap analysis and product/facility pilots before rollout, followed by monitoring. This is an implementation obligation; this reconciliation does not perform a system audit or rollout.

**Priority:** High. **Source strength:** GS1 recommendation. **Applicability:** A/B.

**Source:** SRC-001 §§2.1, 2.4; Appendix C.3–C.5 steps 1–13; pp. 5, 8, 48–50.

**Audit question:** Does adding a new process or external warehouse trigger review of event coverage, identifiers, relationships, and trace-query tests?

<a id="arch-003"></a>

### ARCH-003 — Preserve traceability across component boundaries and recovery

**Rule ID:** ARCH-003.

**Rule:** Ensure the logical traceability repository remains coherent and retrievable across its constituent systems, archives, and recovery procedures.

**Why:** A chain distributed across disconnected applications or unrestorable data is functionally broken.

**Factory Ledger application:** Define authoritative sources and consistent references across modules or services. A single database, integrated domain records, or a federated repository may all be valid. Back up related facts and mappings, verify restoration, and detect missing interfaces or delayed synchronization that affect query completeness. Preserve practical query performance for the required retained volume.

**Priority:** High. **Source strength:** Derived architectural principle. **Applicability:** A/B.

**Source:** SRC-001 §§3.3, 3.4; R31; Appendix C.3 step 6; pp. 12, 17, 38, 49. Backup and recovery mechanics are derived.

**Audit question:** Can a restored or archived system reconstruct a complete trace chain across all required components within its response target?

<a id="arch-004"></a>

### ARCH-004 — Link condition and equipment evidence when it serves traceability

**Rule ID:** ARCH-004.

**Rule:** When condition monitoring or equipment history is in scope, associate observations with the relevant device/asset, time interval, place, and affected material context.

**Why:** A temperature reading or equipment incident is actionable only if the potentially exposed material can be identified.

**Factory Ledger application:** Link available sensor readings or process-condition evidence to storage/transport/production intervals and effective contents. Retain measured values, units, source, and uncertainty. Avoid assigning every historical load of a truck or tank to one isolated incident. Apply exception management to actionable excursions.

**Priority:** Contextual. **Source strength:** Optional enhancement. **Applicability:** B; external sensor exchange C.

**Source:** SRC-001 §§4.3.4, 4.1.1 assets; Appendix C.3 step 7; pp. 21–24, 32–33, 49. Specific sensor schemas, calibration requirements, and food-process limits are not supplied here.

**Audit question:** Can a condition exception identify the material actually associated with the affected asset/location during the relevant interval?

## Relationship contract for a future design or audit

This is a logical contract derived from the rules above, **not a proposed database migration**. Named entities may be implemented differently if their distinct meanings, cardinalities, history, and queryability survive. “Many” is permitted where real operations need it; it is not an instruction to manufacture relationships that did not occur. All substantive edges need the applicable event, quantity/precision, time, and provenance context. Source basis: SRC-001 §§2.3, 3.3.1, 3.5, 4.1.1–4.1.4, R20–R25, and the rules cited in each row.

| Relationship to preserve | Cardinality and historical meaning | Governing rules |
| --- | --- | --- |
| Product ↔ identifiers ↔ packaging definitions | One product/configuration can have supported identifier aliases; identifiers have schemes and issuers. Distinct traded packaging configurations remain distinguishable. | ID-001–ID-004, LOT-002, PACK-004 |
| Party ↔ role ↔ site | A party has multiple roles/sites; a site's operator and role assignments may change over time. | ID-005–ID-006, EVENT-003, MASTER-003 |
| Product/category ↔ partner ↔ source/destination site | Many partners and sites per product, with validity periods. These are possible business relations, not proof of a lot's actual route. | MASTER-002 |
| Supplier/source site → receipt → received lot quantity | A receipt can contain many lots; a lot can occur on many receipts. Preserve actual receipt lines and known attribution limits. | REC-001–REC-003, LOT-003 |
| Order line ↔ fulfillment event line | Many partial or combined fulfillments are possible. Commercial and physical records remain distinct. | EVENT-005, REC-002, SHIP-001 |
| Input lot → consumption occurrence → run/transformation | Many lots per run and many runs per lot; repeated additions retain amounts, units, and relevant times. | PROD-001–PROD-002 |
| Run/transformation → output occurrence → output lot | One transformation may create multiple outputs; a lot may have multiple production declarations if its boundary policy permits. | LOT-001, PROD-001–PROD-002 |
| Intermediate/rework output → later consumption → later outputs | Preserve every generation and its actual sequence. An aggregate lot-level representation must not cause query loops or conceal a later use. | LOT-005, PROD-003 |
| Run → historical formula/process specification | A run resolves the specification actually used and its deviations; the specification alone does not identify consumed lots. | PROD-004, MASTER-003 |
| Finished/bulk lot + applicable packaging lots → packing → packed product | Supports multiple source lots where allowed, multiple pack configurations, and separate product versus packaging contributions. | PACK-004, PROD-001 |
| Serialized case/unit → immediate parent | At most one immediate physical parent at a time, with many historical memberships. Nested ancestors are allowed; cycles are not. | PACK-001–PACK-002 |
| Product/lot quantity ↔ case/pallet contents | Many lot quantities per container and many containers per lot. Quantity allocation substitutes for individual identity only at declared lot-level precision. | PACK-003, TRACE-003 |
| Reusable asset ↔ load/content | Many successive loads over time; asset identity survives changes of content. | PACK-005, ARCH-004 |
| Object/quantity → movement → origin/destination | One lot can occupy many locations; a serial has one physical position at a time. Distinguish transit from confirmed arrival. | INV-001–INV-003 |
| Shipment ↔ manifest lines ↔ logistic units/lot quantities | A dispatch fixes an effective composition. Returned/reused assets may participate later; old manifests remain reproducible. | SHIP-001–SHIP-002 |
| Shipment → recipient party + actual site | A customer may receive many shipments at many sites. A single invoice account does not identify all physical destinations. | SHIP-001 |
| Logistic units ↔ shipment ↔ consignment/transport legs | Delivery and carrier transport groupings have different meanings; retain the role and time of each association. | SHIP-003 |
| Event ↔ participating parties/operators ↔ documents/evidence | Multiple participants and evidence records; preserve role, issuer, source, and validity, with appropriate access restrictions. | EVENT-003, EVENT-005, MASTER-004, DATA-005 |
| Confirmed assertion → correction/amendment | Preserve original and corrected interpretations, attribution, and downstream consequences. A physical return is a separate occurrence. | DATA-001, SHIP-004 |
| Incident ↔ scoped objects ↔ decisions/actions/recipient responses | Many affected objects and actions per incident; scope can evolve with evidence without losing earlier decisions. | INV-004, RECALL-003–RECALL-004 |

Both directions use these same links. The following paths are examples of supported queries, not a single linear schema:

```text
Supplier + source site
  → actual receipt lines → ingredient/packaging lots
  → actual consumption lines → transformations/runs
  → intermediate or finished output lots
  → packing outputs → effective case/pallet contents
  → dispatch manifest → shipments → customers + receiving sites

Branches that must remain connected when applicable:
  intermediate/output lot → rework or later transformation → further outputs
  lot quantity → multiple bins, containers, receipts, or shipments
  case → unpack/repack → different parent at a later time
  shipment → return → hold / release / rework / destruction
```

## Event-data contract

This matrix consolidates existing rules rather than adding new IDs. It is a Factory Ledger design aid. Only the fields explicitly attributed to R21–R25 are source-prescribed minima; the detailed internal record shape, quantities/UOM requirements, entry timestamps, validation states, and correction model are derived.

**Common context:** stable occurrence identity and provenance; actual event time and zone/offset; record-entry time; object references at declared precision; event location; step; disposition; resolvable responsible party; operator/source system where applicable; linked transactions/evidence; and completeness/correction status. Quantity-bearing lines carry amount, unit, and role. Do not add meaningless quantity fields to purely qualitative observations. See EVENT-001–EVENT-005 and DATA-001–DATA-003.

| Event family | Additional relationships/data needed | Meaning that must not be lost |
| --- | --- | --- |
| Receive | Source party/site, actual received lots or serials, received units/containers, quantities, expected-document links, discrepancy and acceptance evidence | Receipt is an observation of arrival, not the order or advance notice. R21 plus REC-001–REC-003. |
| Create/commission | New object identity, product/lot context, quantity if applicable, origin process | Identifier reservation is not physical creation. R22 plus CTE-002. |
| Consume/transform/produce | Actual input/output line sets, run/process context, quantity/UOM, intermediate/rework links, relevant process-specification version | Many-to-many material ancestry; planned recipe is not actual consumption. R24 plus PROD-001–PROD-005. |
| Pack/aggregate | Parent and child identities or lot quantities, containment level, effective membership, applicable packed-output and packaging-input links | Actual contents are distinct from standard pack definitions. R25 plus PACK-001–PACK-004. |
| Unpack/disaggregate/repack | Removed child/lot quantities, original parent, effective time, destination/new membership as applicable | Ending membership does not delete its historical existence. R25 plus PACK-002–PACK-003. |
| Move/store/load/unload | Object or allocated contents, source/destination/read point as relevant, transport asset/leg, departure/arrival evidence | A repeated read is not necessarily a move; in-transit is distinct from received. INV-002–INV-003, SHIP-003. |
| Ship | Destination party/site, ship-from site, dispatch date, actual manifest and lot quantities, relevant carrier/order links | Later repacking cannot alter the dispatched contents. R21 plus SHIP-001–SHIP-003. |
| Inspect/hold/release | Subject scope, decision/disposition, reason/evidence, responsible authorization, applicable time | State restriction is distinct from location. INV-004, MASTER-004, QUAL-004. |
| Return/dispose/destroy/sample | Original path where known, actual identity/quantity, disposition, decision/evidence, new custody/location where relevant | Terminal or exception activity retains ancestry and quantity effects. CTE-004, SHIP-004. |
| Correct/adjust | Referenced original facts, original and effective values, actor/time/reason, stock/genealogy consequences | An administrative correction does not assert an unobserved physical event. DATA-001. |
| Condition observation, if used | Device/asset, reading/unit, observation interval, relevant material/location association, source and uncertainty | Exposures depend on time and effective contents. ARCH-004. |

## Mandatory recall scenario contracts

These are hypothetical design fixtures, not Factory Ledger records or executed tests. Both queries must use the relationship and event-data contracts above, return evidence references for each path, identify the query cutoff and effective correction version, and expose scope/precision limitations. The expected sets below are defined independently of any implementation. Additional uncertain paths must be reported as candidates with reasons, not silently discarded or presented as proven contamination.

### Shared hypothetical fixture

Supplier Alpha at source site A supplies ingredient lot I1 on receipts RA1 and RA2; Supplier Beta at B supplies I2 on RB1; Supplier Gamma at C supplies I3 on RC1. Each lot has a distinct product/issuer-qualified identity. RA1 and RA2 are pooled, so subsequent I1 use is attributable to I1 but not to one particular receipt. For this fixture, all I1 and M1 quantities are consumed in the listed processes; no unlisted residual stock or process branches exist. An implementation fixture must supply compatible measured input/output quantities and justified process losses for production reconciliation.

| Process | Actual inputs | Actual outputs |
| --- | --- | --- |
| T1 | I1 and I2 | Intermediate M1 |
| T2 | All of M1 | Finished product P / lot F1, packed as 100 cases |
| T3 | I1 and I3 | Finished product Q / lot F2, packed as 50 cases |
| T4, later rework | 10 case-equivalents of F1 and I3 | Finished product R / lot F3, packed as 10 cases |
| T0, unrelated control | I2 only | Finished product U / lot F0, 20 cases held on hand |

Packing records link each packed lot to its bulk output and any in-scope packaging lots. The fixture retains event times, sites, parties, historical specifications and evidence for all processes. Case-equivalents and quantities here are fixture assumptions for disposition checks, not prescribed yield or unit-conversion rules.

| Dispatch / disposition | Evidence and quantities |
| --- | --- |
| D1 | 30 F1 cases dispatched to Customer C1 / site S1; receipt confirmed |
| D2 | 20 F1 cases dispatched to Customer C1 / site S2; receipt confirmed; 5 subsequently returned and held |
| D3 | 25 F2 cases dispatched to Customer C2 / site S3; receipt confirmed |
| D4 | 5 F3 cases dispatched to Customer C3 / site S4; arrival unconfirmed at the query cutoff |
| D5 | 5 F2 cases first dispatched to Customer C4 / site S5, refused, then redirected and confirmed received by Customer C5 / site S6; retain both legs |
| Remaining stock | F1: 40 never-dispatched cases plus the 5 returned cases on hold; F2: 20 cases on hand; F3: 5 cases on hand. The 10 F1 case-equivalents used by T4 are consumed history, not remaining F1 stock. |

### Scenario 1 — Contaminated incoming ingredient lot → products and customers

Start with Supplier Alpha's product/lot identifier for I1. The finished-product/lot result is exactly **{P/F1, Q/F2, R/F3}** through T1/T2, T3, and T4 respectively. Intermediate M1 and all I1 receipts/uses remain visible on the paths. F0 is excluded because its complete recorded ancestry contains I2 only; the matching presence of I2 in T1 is not a downstream path from I1 to F0.

The historical dispatch/exposure result is exactly **{C1/S1, C1/S2, C2/S3, C3/S4, C4/S5, C5/S6}**, with D1–D5 and both D5 legs. Confirmed receiving sites are C1/S1, C1/S2, C2/S3 and C5/S6; C3/S4 remains a pending recipient, and C4/S5 a refused destination with its recorded custody evidence. Neither is silently dropped. Returned stock does not remove C1/S2 from the report. Customer summaries may deduplicate party identity, but must retain each site, shipment, quantity and delivery status underneath.

At the cutoff, current affected finished-case holdings are 45 F1 cases (including 5 held returns), 20 F2 cases, and 5 F3 cases; outstanding dispatched quantities are 45 F1, 30 F2, and 5 F3 cases, with confirmation status preserved. F1 reconciles as 100 = 45 currently held + 45 outstanding dispatched + 10 consumed as rework; F2 as 50 = 20 + 30; F3 as 10 = 5 + 5. Gross dispatch, returns, redirections and rework are separate flows. Do not sum production across F1 and F3 to claim distinct remaining product. Unknown onward recipients are a visible boundary requiring partner evidence, not a claim that the full external chain is known.

**Pass criterion:** All expected product/lot and exposure pairs appear with evidence paths and quantity/disposition reconciliation; the unrelated control is excluded on positive evidence; every gap, candidate and delivery-status distinction is visible. Primary rules: RECALL-002–RECALL-003, PROD-001–PROD-003, PACK-001–PACK-004, SHIP-001–SHIP-004, TRACE-003–TRACE-004.

### Scenario 2 — Customer-reported finished case → raw materials and suppliers

Customer C3 reports a case labeled product R / lot F3, with D4 as delivery context. Without serialization, the supported result is F3's history and the candidate case route, not an invented unique case identity or proof that D4 arrived. Trace back F3 → T4 → F1 → T2 → M1 → T1, and the direct I3 input to T4. The complete raw-material set is exactly **{I1, I2, I3}**; the receipt set is **{RA1, RA2, RB1, RC1}**; and the supplier/source-site set is **{Alpha/A, Beta/B, Gamma/C}**. RA1 and RA2 remain pooled candidate contributions. Include the actual packing records and their relevant packaging-lot/supplier ancestry, custody and location events, dates, specifications, quality evidence and correction history throughout the path. F2, F0 and T3 are not ancestors of F3 merely because they share ingredients.

Repeat with an individually identified case whose packing/dispatch associations are recorded: the result may narrow its route only to the extent those associations justify it. Repeat with product-only or conflicting label data: return the supported candidate set or identity gap and required follow-up, without falsely claiming a complete unique history.

**Pass criterion:** All expected raw-material, receipt and supplier/site identities appear with every supporting generation and its provenance; no recipe-only substitution or loss of rework/packaging history occurs; identity and pooled-receipt ambiguity are explicit. Primary rules: RECALL-001, ID-003, LOT-002–LOT-003, REC-001–REC-003, PROD-001–PROD-004, PACK-002–PACK-004, MASTER-003–MASTER-004, DATA-001.

## Acceptance scenarios for a later audit

These are **proposed acceptance tests, not tests executed against Factory Ledger**. They supplement each rule's audit question and make relationship failures observable. A later auditor should implement fixtures suited to the actual system, capture query evidence, and grade each applicable rule individually.

| Scenario | Expected result and evidence | Rules primarily exercised |
| --- | --- | --- |
| 1. Two suppliers use lot code “042” for different products | Two distinct lot identities. A recall of one does not select the other just because the string matches. Original codes and issuers remain retrievable. | ID-003, LOT-002, TRACE-003 |
| 2. Same supplier lot arrives twice and is split into three bins | Both receipt events remain; total on-hand reconciles; all bin quantities resolve to the lot. If mixed, the system does not invent exact receipt-to-consumption attribution. | LOT-003, REC-001, INV-001–INV-003 |
| 3. One run consumes two lots of the same ingredient, and one of those lots feeds another run | Traceback from either output returns the right input set. Forward tracing the shared input reaches both runs and every applicable output; repeated additions are not collapsed into one arbitrary lot. | TRACE-002, PROD-001–PROD-002 |
| 4. An intermediate is used, and part of a finished lot is later reworked | Forward tracing reaches all later outputs without looping or treating rework as fresh, unrelated raw material. Earlier and later production remain distinct occurrences. | LOT-005, PROD-003, RECALL-002 |
| 5. A pallet contains 20 cases of lot A; 7 are moved to another pallet | Historical first-pallet contents show 20 before and 13 after. The new pallet shows 7. Total remains 20. A nonserialized case cannot be claimed to have a uniquely known route. | PACK-001–PACK-003, INV-001 |
| 6. A dispatched pallet is later unpacked or its reusable asset is reused | The original shipment still resolves its original lot quantities and recipient. The asset's later load does not replace the old contents. | PACK-005, SHIP-002 |
| 7. One contaminated input lot reaches multiple outputs and customer sites | Query returns all descendant branches, on-hand/held material, shipments and direct recipients, with unknown onward paths explicit. Results do not add intermediate and final quantities as if they were distinct remaining stock. | RECALL-002–RECALL-003, TRACE-004 |
| 8. A customer reports only a product and lot rather than a serialized case | Traceback returns the supported lot-level history and candidate shipment paths, not a fabricated individual-case history. | TRACE-003, PACK-003, RECALL-001 |
| 9. Receipt is entered late, a submission is replayed, and a wrong input lot is corrected | Physical and entry timestamps remain distinct; replay does not duplicate inventory; correction retains the original assertion and revises affected genealogy transparently. | EVENT-002, DATA-001–DATA-003 |
| 10. Recipe, case conversion, supplier name, and site operator change | Older runs, quantities, parties, and responsible-party defaults retain the meaning applicable to their event times. | PROD-004, MASTER-003, EVENT-001 |
| 11. A label is unreadable or a required consumption link is missing | An attributed exception is visible; affected operations follow hold/completeness controls. Recall results expose potential exposure and missing evidence rather than reporting an empty safe result. | SCAN-004, QUAL-001, RECALL-003 |
| 12. A customer returns part of a dispatched lot | Original dispatch remains recorded. Returned, outstanding, restocked, reworked, and destroyed quantities are distinguishable and reconcile without treating a credit note as physical receipt. | SHIP-004, RECALL-004, INV-001 |
| 13. Archived data and partner data are needed during a mock incident | Authorized personnel can retrieve readable evidence within defined targets. Partner delays, redactions, and missing records are visible and escalated. | DATA-004, EXCH-001, EXCH-005, ARCH-003 |
| 14. A mixed-lot pallet or packing-material incident is traced | Every affected content lot or packaging-use branch is included; no homogeneous-pallet or ingredients-only assumption narrows the result. | TRACE-001, PACK-003–PACK-004, RECALL-002 |

For any later audit result, record: Rule ID, applicability and rationale, pass/partial/fail/unknown/not-applicable status, concrete evidence, failing scenario, affected workflow/data population, risk/value, and remediation acceptance criterion. “Unknown” is not a pass; “not applicable” requires scope evidence. No system audit results have been populated by this reconciliation.
