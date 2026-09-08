# Factory Ledger traceability source register

Status: Cumulative; source collection open. Last updated: 2026-09-07, Batch 001.

## SRC-001 — GS1 Global Traceability Standard

| Field | Recorded value |
| --- | --- |
| Title | GS1 Global Traceability Standard — GS1's framework for the design of interoperable traceability systems for supply chains |
| Release | 2.0, ratified August 2017, as stated on PDF pp. 1, 53, 58 |
| Supplied artifact | 59-page website-print PDF, supplied by the user |
| Original path | `/Users/michaelgross/Downloads/GS1 Global Traceability Standard _ GS1.pdf` |
| Preserved source | [SRC-001 PDF](/Users/michaelgross/Documents/factory-ledger/docs/traceability/sources/SRC-001-gs1-gts-release-2.0.pdf) |
| SHA-256 | `905c9b31c58ae393e317004bb92b8a65c2ae11f8a193b2dccf831421b162b518` |
| PDF metadata creation | 2026-09-07 20:18:40 UTC; this is the print-file timestamp, **not** the standard's publication date |
| Source website printed in PDF | [GS1 source page](https://www.gs1.org/standards/gs1-global-traceability-standard/current-standard) — provenance only; this batch analyzed the supplied snapshot |
| Read coverage | All 59 pages of extracted text read, including glossary, references, requirements, appendices, contributors/change log, and website footer |
| Visual coverage | All 59 pages rendered and visually surveyed; event/KDE diagrams, containment diagram, and requirement/role tables checked against text; Figures 3-8/3-9 examined at enlarged resolution for quantities and relationships |
| Source instructions | Document content treated as evidence. Navigation prompts, contacts, and links did not authorize browsing, communication, installation, or changes to Factory Ledger code. |
| Output supported | Rulebook v0.1, 80 rules, relationship/event contracts, and proposed acceptance scenarios |

Page numbers throughout the rulebook refer to this exact 59-page PDF. Its diagrams sometimes follow their figure captions on the next PDF page. The source is a website print, so contributor-table rightmost columns are clipped; this does not affect the architectural requirements. The material technical text and diagrams used for rules are legible. Full contents of referenced external standards were **not** provided or processed in this batch.

## Coverage by section

| Source span | Material considered | Result in rulebook |
| --- | --- | --- |
| §1.1–1.4, pp. 1–4 | Objective, lifecycle scope, technology neutrality, layered standards, document usage | TRACE-001–TRACE-002, CTE-004, EXCH-006, ARCH-002 |
| §2.1, pp. 4–5 | Operational/recall uses, data accessibility, cross-functional needs, evolving requirements | RECALL-001–RECALL-005, ARCH-001–ARCH-002; no legal obligations inferred |
| §2.2, pp. 5–6; Figure 2-3 | Traceable objects, class/lot/instance precision, risk and material/packaging scope | TRACE-001/003, ID-001/004, LOT-001/002, PACK-003/004 |
| §2.3, pp. 6–8; Figures 2-4/2-5 | Who/what/where/when/why; creation of internal and multi-party data; irreversible input/output combinations | CTE-001/003, EVENT-001–EVENT-003, PROD-001 |
| §2.4–2.5, pp. 8–10; Figures 2-6/2-7 | Identify/capture/share, interoperability, extensions, hybrid standards and maturity | ID-002/003, SCAN-001/002/005, EXCH-002–EXCH-004/006 |
| §3.1–3.2, pp. 10–12; Figures 3-1/3-2 | CTE/KDE definitions, one-up/one-down, bidirectional full-chain tracing, emergent relationships | TRACE-002/004, CTE-001, RECALL-001/002, EXCH-001/005 |
| §3.3.1, pp. 12–13; Figure 3-3 | Static and relation master data, transactions, visibility events | MASTER-001–MASTER-003, EVENT-005, PROD-004 |
| §3.3.2, pp. 13–14; Figure 3-4/Table 3-1 | Identification versus record granularity; precision limits | TRACE-003, LOT-003/005, PACK-003, RECALL-003 |
| §3.3.3, pp. 14–15; Table 3-2 | Internal/external sensitivity examples and internal access restrictions | DATA-005, EXCH-005; examples not copied as an automatic disclosure policy |
| §3.3.4, p. 15 | Completeness, accuracy, consistency, temporal validity | QUAL-001–QUAL-004, EVENT-002/004, DATA-001–DATA-003, MASTER-003 |
| §3.3.5, pp. 15–16; Figure 3-5 | Five sharing choreographies, asymmetric visibility, repository access differences | EXCH-001/005/006; technology choices retained as conditional considerations |
| §3.4, pp. 16–17; Figure 3-6 | Organizational traceability system, risk/cost balance, scope, multiple IT components | TRACE-001, ARCH-001–ARCH-003 |
| §3.5, pp. 17–20; Figures 3-7–3-11 | Ingredient/packaging supply, manufacturing, lot quantities, pallets, truck loading, receipt checks, upstream/downstream queries | REC-001–REC-003, PROD-001/002/005, PACK-001–PACK-004, SHIP-001/002, RECALL-001/002; quantity rules explicitly derived |
| §4.1.1, pp. 21–24; Tables 4-1/4-2, Figure 4-1 | Object keys, product/lot/serial distinction, logistics hierarchy, asset versus contents | ID-001–ID-004, LOT-002/003, PACK-001–PACK-005, SHIP-003 |
| §4.1.2–4.1.4, pp. 24–25; Tables 4-3–4-5 | Parties, roles, sites, sublocations, coordinates, documents | ID-005–ID-007, EVENT-003/005, INV-002, MASTER-004 |
| §4.2–4.2.2, pp. 25–27; Figures 4-2/4-3, Table 4-6 | Carrier capabilities, dynamic marking, object/label link, RFID commissioning, scanner context and process selection | SCAN-001–SCAN-005, EVENT-002/003; healthcare-specific stakeholders excluded |
| §4.3.1–4.3.3, pp. 27–32; Tables 4-7/4-8 | Content/transport separation, master/transaction/event standards, push/pull/subscriptions, discovery and trust | EXCH-001–EXCH-006, DATA-005 |
| §4.3.4–4.4, pp. 32–34; Figures 4-4/4-5 | Sensors/context, ecosystem integration, proprietary lock-in, partial interoperability, certification references | ARCH-004, MASTER-004, EXCH-002/004/006 |
| §5.1–5.6, pp. 34–39 | Scope-relative requirements/KPIs and data responsibilities | Explicit R01–R32 crosswalk below; QUAL-003, ARCH-001 |
| §6–6.1, pp. 39–42 | Custody/ownership, trace versus track, objects/events, party roles, unknown end-consumers, abbreviations | SHIP-003, TRACE-004, CTE-001/003, RECALL-001/002; no separate duplicate glossary rules |
| §7, pp. 42–43 | Normative/non-normative reference list with dated editions | Version/dependency caveats and pending-source list below; referenced documents not assumed read |
| Appendix A, pp. 43–45 | Summary of identifiers, carriers, event categories and exchange mechanisms | Consolidated into ID/SCAN/CTE/EXCH rules, not duplicated |
| Appendix B, pp. 45–47 | Identifier issuance, master/event/transaction ownership, recall notification/removal/closeout roles | ID-008, ARCH-001, RECALL-004; allocation exceptions reserved for source standards |
| Appendix C.1–C.5, pp. 47–50 | Goals → information needs → processes → identification → data → repository → interventions → gap analysis → pilot → rollout → training → monitoring | TRACE-001, ARCH-001–ARCH-003, RECALL-004/005, QUAL-004; future audit only |
| Appendix D, pp. 50–53 | CPG, fresh foods, healthcare, technical industries, transport/logistics | Packaging, perishability, waste, containment and asset/custody lessons retained; sector-specific exclusions below |
| Contributors/change log/footer, pp. 53–59 | Release history, contributors, copyright and website navigation | Provenance retained; no design rules generated |

## Direct GS1 requirement crosswalk

This table ensures every numbered requirement was considered. “Direct anchor” identifies the rule that carries the corresponding obligation without converting all supporting implementation details into GS1 mandates.

| Source requirement | PDF pages | Direct anchor | Qualification |
| --- | --- | --- | --- |
| R01 — globally identified traceable objects and master data | 35 | ID-002 | Explicit exception for intermediate products; other local-ID deployments are deviations/hybrid limitations, not this exception. |
| R02 — retain other parties' object identifiers | 35 | ID-003 | Retrieve relevant master data where appropriate. |
| R03 — identification at required precision | 35 | ID-004 | Coordinate required granularity with supply-chain needs. |
| R04 — globally identified parties/master data | 35 | ID-005 | Party identity is separate from its roles. |
| R05 — globally identified physical locations/master data | 36 | ID-006 | Scope and needed site/sublocation granularity apply. |
| R06 — globally identified shared documents/transactions | 36 | ID-007 | Applies to created records shared with other parties. |
| R10 — open AIDC for own objects | 36 | SCAN-001 | Source says “should”; classified as recommendation, conditional on requiring automatic identification. |
| R11 — open AIDC for other parties' objects | 36 | SCAN-001 | Consolidated with R10; inbound coverage remains an explicit audit dimension. |
| R20 — partner/product/site relation data and validity | 36–37 | MASTER-002 | Includes suppliers, customers, and third-party providers in its KPIs. |
| R21 — receipt and shipment CTEs | 37 | REC-001, SHIP-001 | Split by workflow for auditable responsibilities; note precision ambiguity below. |
| R22 — initial creation CTEs | 37 | CTE-002 | Object creation distinct from mere reservation of an identifier. |
| R23 — minimum KDEs | 37 | EVENT-001 | Responsible party required if different from location manager; no quantity/UOM or record-entry timestamp listed in its minimum. |
| R24 — transformation input/output relationships | 37 | PROD-001 | Many-to-many representation and detailed allocation mechanics are derived in PROD-002. |
| R25 — aggregation/disaggregation at each containment level | 37 | PACK-001 | Temporal integrity and parent constraints are derived in PACK-002. |
| R26 — additional applicable CTEs | 38 | CTE-004 | The internal catalog is scope-dependent; installation is not automatically required for food products. |
| R30 — effective, secure, timely two-way sharing | 38 | EXCH-001 | Authorization elaborated in EXCH-005; no universal time limit specified. |
| R31 — retention and authorized accessibility | 38 | DATA-004 | Includes partner accessibility; no universal retention duration specified. |
| R32 — open standards for automated electronic exchange | 38–39 | EXCH-002 | Conditional on that exchange activity; does not impose every listed GS1 technology. |

There are 18 numbered source requirements. Some numerical slots are absent in GS1's numbering; they are not missing pages or omitted requirements.

## Ambiguities, qualifications, and pending decisions

1. **R21 versus R23 precision.** R21 allows class-, lot-, or instance-level IDs for shipments/receipts; R23 lists lot or instance identity for each CTE. Sections 2.2/3.3.2 also illustrate class-level secondary materials. This batch does not erase that tension. Factory Ledger's proposed policy requires lot/instance precision for risk-relevant food traceability; any class-only category needs an explicit scope justification and cannot claim lot-level capability. An eventual strict GS1 conformance assessment needs the applicable profile or clarification.
2. **Global identifiers versus an internal/hybrid system.** R01–R06 articulate interoperability requirements; §2.5 permits governed hybrid implementations and gradual adoption. This is not blanket permission to call any local-ID system fully compliant. ID-002 preserves the explicit intermediate-product exception; local design decisions and external conformance are tracked separately.
3. **Framework minima versus engineering requirements.** R23 does not enumerate quantities/UOM, record-entry time, an immutable audit ledger, SQL constraints, idempotency, formula versioning, or a correction schema. Those are derived rules grounded in accurate, consistent, retrievable history. They remain clearly labeled even when assigned Critical priority.
4. **No universal case serialization.** The examples use lot-identified cases and SSCC pallets with lot quantities. Serial identity has higher precision and cost. Lot-level cases may be appropriate; an individual-case route cannot be promised without additional evidence.
5. **Aggregation versus transformation in packing.** Physical grouping preserves contents; creation of a new packed trade item can also require output identity and material input relationships. The word “packing” alone does not settle the event design. Preserve both meanings where both occur; do not infer a new GTIN for every transformation from Appendix D's simplified CPG example.
6. **Technical versions are historical.** The reference list includes EPCIS 1.2 (2016), CBV 1.2.1 (2017), General Specifications v17 (2017), and other dated documents. Statements that discovery services or sensor capabilities were under development describe the source's period. This batch makes no claim about their present availability or the correct current deployment version.
7. **External/internal data categories are examples.** Table 3-2 is not a universal authorization policy. Supplier/customer relations and quality data may require different disclosure controls depending on the actual question and agreement.
8. **Relation data standards.** R20 specifies relation content; Appendix A describes bilateral relation sharing without a designated standard, while R32 requires open standards for automated traceability exchange. An implementation must select an appropriate supported exchange/profile and disclose extensions; the source does not provide a universal relation-data wire schema.
9. **Retention, response time, and tolerances are unassigned.** GTS ties retention and response to relevant requirements but provides no universal duration or deadline. It also does not specify process yields, lot sizes, mass-balance tolerances, or release limits. The rulebook requires explicit policies without inventing values.
10. **Custody does not establish exact contamination.** Recorded relationships establish potential exposure paths. They do not prove the contaminated fraction of a mixed output. Exact physical attribution must not be inferred from proportional bookkeeping or nominal recipes.
11. **Full-chain visibility has external dependencies.** One-up/one-down queries, discovery, and trust can extend the chain; Factory Ledger cannot prove unknown onward paths from internal data alone. Missing partner evidence must remain visible.
12. **GTS CTE/KDE terms are not a legal determination.** This source is sector-neutral and complementary to other standards. No jurisdiction-specific food tracing law, mandated event list, or filing deadline is established here.

## Concept applicability and deliberate exclusions

| Concept or source material | Classification | Treatment and reason |
| --- | --- | --- |
| Object identity, lot/instance precision, CTE/KDE context, bidirectional genealogy | A | Core rules; necessary to answer the two incident questions. |
| Quantified many-to-many allocations, temporal containment, corrected history, historical master data | A/B | Derived internal design rules; no prescribed database technology. |
| GTIN, GLN, SSCC | A principle / C exact scheme | Preserve their distinct identity meanings; exact GS1 adoption follows interoperability scope. |
| GSIN versus GINC | C | Conditional delivery/transport grouping support in SHIP-003, not required for every internal shipment. |
| GIAI/GRAI | B/C | Useful if reusable assets/transport equipment are tracked; keep asset/content distinction regardless of exact key. |
| GDTI | C | Useful for shared document identity; local documents still need stable references. |
| GSRN | C | Consider only if externally identified service relationships are needed. Do not require it for every operator login. |
| GPC | B/C | Optional product-category mapping for relation records/analysis. No mandated internal product classification service. |
| EPCIS/CBV | B/C | Adopt event semantics and vocabularies where useful; actual compatibility requires a separate specification/profile. |
| GDSN and GS1 EDI | C | Conditional master-data synchronization and transactional exchange; not prerequisites for internal genealogy. |
| GLN Service, GS1 Source, GS1 SmartSearch | C | Historical examples of identity/master-data discovery and publication channels. No deployment requirement or present availability claim. |
| GS1-128, GS1 DataMatrix, GS1 QR, other barcodes, EPC/RFID | B/C | Carrier choices driven by precision, process, partners, and equipment. Detailed encodings remain pending. |
| Watermarks and image recognition | B, optional | Alternative capture technologies acknowledged; no separate rule because SCAN-001–SCAN-005 already express the needed capture outcomes. |
| Sensors/actuators and IoT | B/C, optional | ARCH-004 preserves the business-context lesson, particularly for condition exposure. No sensor installation mandate. |
| Centralized/networked/cumulative/decentralized sharing | B/C | Design alternatives. Cumulative downstream sharing alone does not provide upstream parties full downstream visibility. No mandatory shared ledger. |
| Blockchain hashes, encryption, nested signatures | C, optional | Retain access/tamper-evidence and storage tradeoffs as context; no blockchain, signature, or cryptographic architecture mandated from this discussion. |
| Food packaging, perishability, relevant waste and certifications | A or contextual B | Included through scope, date, packing, evidence, exception and terminal-event rules. No food-specific technical limits invented. |
| Patient/caregiver, medical records, bedside administration, surgical-device workflows | D | Healthcare examples are not Factory Ledger workflows; generic identity and evidence lessons were retained separately. |
| GCN, digital/paper coupons and coupon management | D | No connection to the stated food-manufacturing traceability scope. |
| CPID and OEM part-number allocation | D | Specialized OEM parts mechanism is not indicated by the stated scope; generic stable-identity lessons remain included. |
| Rail/defence MRO, long-life manufactured equipment lifecycle, installed subsystem management | D for specialized workflow | No food-product MRO subsystem recommended. Reusable equipment identity and historical composition lessons retained where relevant. |
| Consumer marketing, search optimization, omnichannel sales experience | D for product scope | No consumer marketing/search feature mandated. Relevant downstream recipient and data-sharing limits retained. |
| Direct identification of every final consumer | D as a blanket requirement | The glossary notes end-consumers may be unknown and not traceability parties. Actual direct recipients are retained at available/required precision. |
| Animal regulatory IDs and BIC intermodal examples | C illustration / D implementation in present scope | Illustrate governed hybrid IDs; no animal registry or intermodal subsystem added. |
| Contributor names, acknowledgments, navigation, website service links | D | Provenance only; no engineering rules or external actions derived. |

## Referenced material not yet processed

The document points to the GS1 System Architecture, General Specifications, GTIN Management Rules, GLN Allocation Rules, EPCIS/CBV and their implementation guideline, Logistics Label guideline, EDI and Product Recall standards, Tag Data Standard, food traceability compliance criteria, and other sector guidance. Their appearance in the bibliography is not evidence that their full requirements were read.

Future supplied batches may substantiate or revise identifier allocation/reuse, Application Identifier syntax, EPCIS correction and event-time semantics, detailed food KDEs, label contents, recall messages, and conformance tests. Do not silently substitute this source's references for those documents or relabel derived rules as explicit requirements without supporting text.
