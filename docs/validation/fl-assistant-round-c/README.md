# FL Assistant fix round C: voice + phone

Branch `fix/assistant-voice`, worktree `~/dev/fl-fixC-wt`. Built on round B `dc62638`, then rebased onto its merged main commit `c04274e`. Round B’s worktree was not edited. Changelog row **212**. No migration or deployment.

| Requested item | Result |
|---|---|
| Spoken product numbers | Done: Classic #9 and SS #1/#2/#3, including Sunshine and Spanish número spellings. |
| Spoken quantities and units | Done: English/Spanish cardinal quantities, bags/cases/batches and lb; preserve decimal/thousands magnitude and party names. |
| Spoken lot codes | Done: individual digits and grouped letter codes, including P fifteen eighty-five / P one five eight five and Spanish equivalents. No lot or tag confirmation is inferred. |
| Misheard names and BOL | Done: actual catalog candidates become human buttons; unknown wording is named. POL, B.O.L., bowl and B O L with codes normalize to BOL. |
| Customer choices | Done: candidates retain customer-table provenance and names appear only on buttons. A guarded read-only staging audit found A-1 Bakery Supply (36) and Bakers Depot (69) are active customers; neither name matched a supplier. |
| Today by voice | Done: supported English/Spanish review phrases bypass model availability and show complete newest-first entries with separate own/team counts. Corrections and void requests keep their action intent; the assistant does not certify correctness. |
| Phone layout | Done: compact Menu contains Receipt lookup, Exceptions, Record entry and Office; no sideways page scroll at 375, 390, 393 or 430px, including focused receipt/tag fields. |
| Small wording | Done: no developer order assignment note, omit draft midnight, plain temporary-unavailability copy, no raw Internal server error. |

Every B physical tag remains required, including single-lot ingredients. Wrong tags refuse. Different lot still records unresolved trace evidence without substituting suggested FIFO stock; extra-kosher holds and replay rules remain intact. The core traceability files are unchanged from main.

| Final check | Passed |
|---|---:|
| Full Python, fresh local PostgreSQL 17, sessions off | 6,317 |
| Full Python, fresh local PostgreSQL 17, sessions on | 6,317 |
| Node | 369 |
| Exact-transcript actual-page checks, Chromium + WebKit, EN/ES, both modes | 72 |
| Approved real gpt-6.1-sol actual-page checks, EN/ES, both modes | 36 |
| Phone layout/focus checks, both engines/modes/languages, all four widths | 288 |
| Round A browser | 30 per mode |
| Round B backend browser contract | 9 per mode |
| Round B actual dashboard page | 68 total |
| Existing Assistant and customer browser runners | Both modes pass |

Full Python has zero failures/errors/skips. Each mode ran every collected test exactly once across 22 fresh schema/race groups using the saved allocation runner and unchanged original migration/fixture SQL. Python 3.12.14, Pydantic 2.14.1. Node has zero failures/skips. Intermediate runs caught unfinished test assertions and old wording/layout expectations; final counts above come from complete runs on the reviewed rebased source.

All seven exact English phone transcripts and Spanish versions run through real local page/API/DB handlers. API acceptance includes 36 cases per mode and records receive/make/pack/order with supplied facts and human physical-tag evidence. Browser acceptance stops before Record: C’s queued and real-model runs post zero Record calls and no ledger writes. Legacy B runners use their explicit preview-only mode; legacy fixture-only UI runners retain simulated receipt behavior. The real-model run used 60 completed OpenAI Responses calls after explicit user approval.

The delivered screenshots and results are in [phone](phone/results.json), [transcripts](transcripts/results.json) and [real-model](real-model/results.json). Full counts, source hashes and cleanup are in [verification.json](verification.json), with Python/Node/browser logs beside it. All `fl_fixC_` databases were dropped. Hosted access was limited to the read-only customer-name audit; no staging or production writes occurred.

These checks use text transcripts and synthetic local catalog/ledger data. They do not exercise microphone transcription, a physical iPhone, or Safari’s real keyboard/address bar. Extra spelling/unit/code variants are covered by unit/API tests; real-model coverage focuses on the seven transcripts and a grouped P-code. Populated Today lists and pagination are verified through API/UI fixtures; real-model Today fixtures contain no entries.
