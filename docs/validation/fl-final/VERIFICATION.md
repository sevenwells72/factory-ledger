# PR 136 — final round E acceptance

All 17 requested items are implemented and verified locally. The branch was rebased onto merged PR 137 / origin/main `60be28ae8454bc9f37a64d1d65002375490b9f17` before edits. Both change-log histories survive; Assistant assets are display v9, Assistant v14 and CSS v7. Regression-guard row 216 records this round.

The separate `/Users/michaelgross/dev/fl_final_venv` exactly matches every requirements.txt pin, including Pydantic 2.14.1. All 41 installed dependencies are compatible. The model and hosted settings were not changed. The staging source was dumped read-only through port 5432 and restored only to local databases named `fl_final_…`. Staging received no writes. Every disposable database, staging-derived dump and private full model trace has now been removed; both owned fresh-test clusters were stopped and deleted.

| # | Done | Result and proof |
|---|---|---|
| 1 | Yes | Found accepts the original note in either language. Real-model found Record, receipt readback and replay: 24/24; 12 Spanish records had notes_es and no English note. Includes the reason-button path, three repeats per language and mode. |
| 2 | Yes | Supplier lot and BOL appear on the receipt card, receipt page and chat lookup. Actual receive Record/readback/lookup in both languages plus iPhone-width WebKit evidence. |
| 3 | Yes | Both short and complete Spanish/English bag phrases produce 10 × 50 = 500 lb. Every exact form passed three times per mode. |
| 4 | Yes | Receives show one product at a time and retain the rest visibly in a persistent queue. Cannot advance past an unfinished first item. Both short and full EN/ES sentences passed 3/3 per mode; first item recorded and second Oats draft verified. |
| 5 | Yes | Quota/billing failures show the requested offline wording with Record entry and Office links, no Retry. Temporary failures retain Retry. Clear error log plus a deduplicated owner-only Exceptions notice is the alert channel. Fault-injection, owner permissions and EN/ES WebKit checks pass. |
| 6 | Yes | Change/cancel order links open the identified order in Office; future deliveries link to expected-delivery creation. All EN/ES exact cases passed 3/3 per mode without a provider call. |
| 7 | Yes | Spanish Different lot succeeds after explicit valid physical tags. Wrong/ambiguous tags post nothing; full-code recovery succeeds. Actual model Record proof is 3/3 per language/mode. Display quantities have at most two decimals, including 88.8; Spanish errors explain the issue. |
| 8 | Yes | Spanish refusal reasons cover kosher, stock, lot and normal validation families. Real-model kosher and missing-Coffee make refusals plus 28 coded and five legacy-text reason families pass. |
| 9 | Yes | A fully shipped order is identified as already fully shipped, with its SO number, in EN/ES. Three repetitions per mode. |
| 10 | Yes | Stock shortages name the product, available cases and required cases, with next actions. Exact 12/20 wording is tested; staging-copy real model verifies actual Kookies stock of 5 versus 20 needed, 3/3 per language/mode. |
| 11 | Yes | Add Setton and Add Setton as a new customer offer existing Setton Farms; they cannot silently create a duplicate. EN/ES real-model cases pass 3/3 per mode. |
| 12 | Yes | Spoken 283 and 70013, S S original number one and la cross milling resolve correctly. Fuzzy supplier/customer matches still require a human choice. Actual model and deterministic choice guards pass. |
| 13 | Yes | Exceptions use plain labels, newest unresolved first, and collapsed opening-stock rows. Ordering is applied before the server limit. EN/ES WebKit grouping passes with sessions on; eight sessions-off checks preserve the existing dashboard cutover gate. |
| 14 | Yes | Original full_run.log shows EMAXCONNSESSION session-pool client exhaustion on order/product reads. Pool reduced from 2/20 to 1/10 with connect timeout/keepalives; broken sockets discarded; only audited product/inventory/order reads retry once. No write is replayed. Tests cover capacity, disconnect, bounded retry and bilingual failure. Hosted behavior still needs observation after deployment. |
| 15 | Yes | Phone buttons meet the 44 px minimum, with 48 px primary controls. WebKit checks cover 375/390/393/430 px, EN/ES and both modes. |
| 16 | Yes | Jump to latest stays clear of tag inputs, photo-only guidance uses a neutral status instead of the red error box, and unused photoSeq is removed. WebKit and Node checks pass. |
| 17 | Yes | Authenticated lot-tag photos attach atomically to RECEIVE receipts and remain available on card/page/chat lookup. Wrong owner/type/context or missing bytes cannot post; replay creates one post. Actual model photo/receipt proof and deterministic rollback/privacy tests pass. |

Final complete Python suites: **6,709 off + 6,709 on = 13,418 passed**, zero failures/errors/skips. Node: **411/411**. Browser runners: existing Assistant/customer both modes; A **60/60**; B contract **18/18 groups**; B page **68/68**; C transcript **72/72**; C phone **288/288**; mobile UX **144/144**; final floor UI **200/200**, plus **8/8** sessions-off Exceptions gate checks. See [browser summary](browser/SUMMARY.md).

Exact sentence acceptance: **312/312 selected checks**, covering **52 sentence/scenario rows × 3 repeats × 2 modes**. Of these, **240 are provider-backed** and **72 intentionally use deterministic navigation/status handling**. There are **679 distinct completed real-provider responses**, all with `store:false`. No model call or physical-tag confirmation was replaced by a stub in this proof. Browser fixtures are reported separately and are not counted as real-model evidence. See [every exact sentence](exact-sentences.md) and [machine-readable acceptance](acceptance-summary.json).

The last real-model pass found and fixed a narrow bare-list bug: the receive splitter had added its inferred “Received” prefix only to later items. It now adds the same prefix to the first item. All 12 short multi-item and 12 full multi-item checks were rerun on the final source with every provider response completed. Other earlier model proofs remain applicable because they never enter that bare-list branch; every other runtime source hash is unchanged. The exact source hashes, failed attempts and affected reruns are retained in [supplemental evidence](supplement-final-summary.json). Both complete Python modes were repeated after the fix. No failed attempt was relabelled as passing.

Earlier validation failures are retained: two corrected stale receive-state regressions, an invalid resolved-exception test fixture, a mobile sign-in helper race, and the real bare-list defect. The generic misunderstood/incomplete bare-list attempt was rejected, not accepted by weakening the assertion.

Traceability remains explicit: the app/model never confirms a physical lot. Wrong or ambiguous tags produce zero writes; the test worker types each full physical code; Different lot retains an unresolved exception and leaves the suggested Salt stock unchanged. Receipt replay is one post. This is local WebKit phone emulation and typed spoken wording, not a physical iPhone microphone test.

Deployment note: migration **084_assistant_service_alert.sql** must be applied through the normal approved deployment process for the durable owner outage notice. It was tested locally only. No Railway, Supabase or OpenAI setting was changed, and no main merge was performed.

Cleanup evidence: [all local databases and copies removed](fresh-db-cleanup.json), [original staging-copy removal](root-data-cleanup.json), and the supplemental cleanup report. Independent UI/backend/assistant reviews found no remaining concrete defect in the requested scope.
