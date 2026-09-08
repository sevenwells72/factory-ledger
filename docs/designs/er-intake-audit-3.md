# ER intake — independent audit #3 (round 3; received 2026-09-08, verbatim)

3 — Fixed. 4 × 50 vs 175, save_alias=true → 422 ALIAS_CONVERSION_MISMATCH before any DB access (main.py:5825).
4 — Fixed for both reproductions. Failed rematch clears old product and 200 lb; matchStale blocks approval after manual edits. During matching the 175-lb edit is refused, inputs disabled (er-intake-logic.js:221, dashboard.js:4013, :4333).
8 — Not fully fixed. Sequential identical uploads return 200 with the existing ID. But if Storage saves the object and its response is lost, the row becomes upload_failed; healing retries with x-upsert: false, receives "already exists," and stays stuck — reproduced 502 STORAGE_UPLOAD_FAILED with status still upload_failed (main.py:5680, :5692, :5181). Also the dedupe SELECT and INSERT are not serialized; concurrent uploads can create siblings (main.py:5719).
11 — Fixed. "2026-2-3" rejected for both date fields (extraction.py:69).
12 — Fixed. Comment accurate (049:19).
Retry matching — Pass. Uses current state; overlapping retries discard older responses; approval blocked until latest succeeds (dashboard.js:4022, :4032).
no-merge
