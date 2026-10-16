# Final browser and Node evidence

All final runs passed on the actual current assets, locally. Both session modes were exercised. No staging or external service writes were used.

| Runner | Final result | Coverage |
|---|---:|---|
| Existing assistant + customer browser | Passed in both modes | Draft/Record/retry, evidence, choices, reload, ES, 390px |
| Round A | 60/60 | 30 checks per mode |
| Round B backend contract | 18/18 groups | 9 groups per mode; real local API/DB |
| Round B page | 68/68 | Both modes; real local API/DB, queued model tools |
| Round C transcripts | 72/72 | Chromium + WebKit, both modes; real local API/DB, queued model tools |
| Round C phone | 288/288 | Chromium + WebKit, both modes, EN/ES, 375/390/393/430px |
| Mobile UX | 144/144 | WebKit, both modes, EN/ES, 375/390/393/430px; zero Record requests |
| Final floor UI | 200/200 + 8/8 cutover checks | WebKit, both modes, EN/ES, all four widths; evidence in ../ui and exceptions-cutover-off.json |
| All Node tests | 411/411 | No failures or skips |

The first mobile UX run failed on the final context because the test attempted to fill the hidden actor-key field while the saved session was restoring. The sign-in helper now waits for the existing auth-loading state to finish. No app assertion was weakened. The full rerun passed all 144 checks. Initial failure log and matrix are retained separately; matrix.json records the successful final run with the initial attempt nested.

Browser tests use fixture/queued model output and do not constitute real-model proof; separate real-model evidence covers AI behavior. WebKit at iPhone widths is browser emulation, not a physical iPhone or microphone test. Browser temporary databases used the fl_final_ prefix and were dropped by their runners. Bulk screenshots remain outside the repository; the final-floor evidence retains representative 390px EN/ES captures.

Exceptions grouping is visible only with key sessions enabled, by the existing dashboard cutover design. Its names/order/collapsed opening-stock group passed EN/ES at all four widths. The additional eight sessions-off WebKit checks prove the existing gate hides unsupported exception controls and rows. The backend exception ordering/source-marker tests passed 12/12 in each mode on a fresh fl_final_exception_queue database, which was dropped. The resolved fixture was corrected to include resolved_at; application assertions are unchanged.
