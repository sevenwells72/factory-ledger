# ER intake — independent audit #2 (round 2, on 3c19d93..c7f724c; received 2026-09-08, verbatim)

Findings 3 and 4 remain merge blockers. 44 Python extractor/parser tests and 23 JavaScript tests passed. Database-writing tests and migration application not run.

1. Fixed — fuzzy product confirmation. er-intake-logic.js:36 separates suggestions from chosen; :163 requires a chosen product for approval.
2. Fixed — alias unit mismatch. main.py:5424 requires normalized unit equality. 50-lb/BAG alias on 100 LB now returns 100 lb.
3. Partially fixed — P1 remains. UI gates at er-intake-logic.js:139, but main.py:5834 saves the supplied conversion without checking it against approved pounds. Reproduced: qty 4, 50 lb/BAG, expected 175, save_alias=true → alias saved with 50.
4. Partially fixed — P1 remains. Sequencing improved at dashboard.js:4019. But: on supplier-rematch failure, dashboard.js:4029 clears `matching` and retains old lines under the new supplier (old product + 200 lb approvable). A response overwrites edits made while matching at dashboard.js:4024 (175 reverted to 200).
5. Fixed — retry vs approve. main.py:5612, :5623 condition on status <> 'approved' with rowcount check.
6. Fixed within limited scope — rate limits at main.py:5497; requests 31 and 121 return 429 with Retry-After. Process-local.
7. Fixed — weight parser. main.py:5294. `.5 LB` → 0.5; `4 x 5 LB` → 20; malformed → None.
8. Partially fixed — ID can still be lost: upload returns the ID only after storage at main.py:5680; if that request times out, dashboard.js:3949 shows only an upload error with no recovery path.
9. Fixed — file validation. main.py:5649 bounded read, magic-byte MIME, PDF page count before writes.
10. Fixed — advisory lock at main.py:5810 before duplicate check. Tests do not run two full concurrent approvals asserting the loser gets 409.
11. Partially fixed — extraction.py:67 accepts "2026-2-3", violating exact YYYY-MM-DD.
12. Fixed — upload_failed status and approval eligibility; migration 049:47 constraint replacement structurally safe and schema-idempotent, though the "every rerun is a no-op" comment is inaccurate.

Manual POST /expected-receipts: no fix-induced regression. No-merge.
