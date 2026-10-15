# Staging pilot cleanup — dry run, 2026-10-15T10:53:42-04:00

API: `https://fastapi-staging-production-dd7b.up.railway.app`. Reason/note: `STAGING PILOT CLEANUP`. Plan SHA-256: `da131b521f801ea181c0f1969ab82dd332e60899a2fab7b180f369256156b5f7`.

Read-only discovery only: no prepare, commit, approval, admin mutation or direct SQL write was run.

Apply uses only ticket adapters; unsupported items are skipped and remain blocked. Direct product/customer deactivation APIs exist but do not meet the required ticket/receipt contract.

UNIDENTIFIED_LOT uses `written_off` (canonical API decision, final status `waived`). Shortages cannot be waived; no counted resolution is substituted.

Live reconciliation: the morning 95 lots/~5,867 lb was the real-product subset. All STG- lots, including synthetic products and zero balances, are listed below. Live shortages/exceptions can exceed the morning 4/6; every selected item has explicit test product/customer/lot/shift evidence. Owner identity alone never selects an exception.

Apply checks each lot again before prepare and verifies zero balances and exact committed receipts. The server can recalculate at commit; a detected concurrent change stops subsequent items after that commit for reconciliation. This cannot prevent the one in-flight change; use a quiet staging window for a future apply.

The apply command below is provided for review and has NOT been run. A changed snapshot or contract requires a new dry run and reviewed hash. Partial/uncertain apply stops; inspect the adjacent append-only .receipts.jsonl journal before preparing another plan.

```sh
/Users/cns/Documents/factory-ledger/.venv-test.nosync/bin/python /Users/cns/.codex/worktrees/staging-pilot-cleanup/factory-ledger/scripts/staging_pilot_cleanup.py --apply --plan /Users/cns/dev/staging-cleanup-plan-2026-10-15.md --expected-plan-sha256 da131b521f801ea181c0f1969ab82dd332e60899a2fab7b180f369256156b5f7 --api-base https://fastapi-staging-production-dd7b.up.railway.app
```

## Counts

```json
{
  "lots": {
    "total": 266,
    "planned": 239,
    "blocked": 4,
    "noop": 23
  },
  "products": {
    "total": 212,
    "planned": 0,
    "blocked": 212,
    "noop": 0
  },
  "customers": {
    "total": 9,
    "planned": 0,
    "blocked": 9,
    "noop": 0
  },
  "orders": {
    "total": 6,
    "planned": 6,
    "blocked": 0,
    "noop": 0
  },
  "exceptions": {
    "total": 8,
    "planned": 8,
    "blocked": 0,
    "noop": 0
  },
  "shortages": {
    "total": 6,
    "planned": 0,
    "blocked": 6,
    "noop": 0
  },
  "nonzero_real_product_lots": {
    "lots": 95,
    "products": 20,
    "balance_lb": "5867.4000"
  },
  "preserved_shipped_test_orders": 7,
  "protected_open_lots": 205,
  "protected_exceptions": 56
}
```

## Lots

| ID | Item | Before → desired after | Status / path / limits |
|---|---|---|---|
| 1000000003 | STG-SMOKE-5BB2511D5EA3; product 1000000001 STAGING Test Almonds; kosher=regular | 25.0000 → 0 | planned; /adjust/prepare;  |
| 1000000004 | STG-A1-7DA20A23D052; product 1000000001 STAGING Test Almonds; kosher=regular | 0.0100 → 0 | planned; /adjust/prepare;  |
| 1000000005 | STG-A1P2-294CDA1C784E-INPUT; product 1000000004 STG-A1P2-294CDA1C784E ingredient; kosher=regular | 0.0900 → 0 | planned; /adjust/prepare;  |
| 1000000009 | STG-A1P2-57FB94002B5A-INPUT; product 1000000007 STG-A1P2-57FB94002B5A ingredient; kosher=regular | 0.0900 → 0 | planned; /adjust/prepare;  |
| 1000000023 | STG-F1-425FD98B13-LOT; product 1000000016 STG-F1-425FD98B13 ingredient; kosher=regular | 90.0000 → 0 | planned; /adjust/prepare;  |
| 1000000026 | STG-F1-85AAC3C9FC-LOT; product 1000000019 STG-F1-85AAC3C9FC ingredient; kosher=regular | 88.0000 → 0 | planned; /adjust/prepare;  |
| 1000000029 | STG-F1-85AAC3C9FC-FOUND; product 1000000019 STG-F1-85AAC3C9FC ingredient; kosher=regular | 2.0000 → 0 | planned; /adjust/prepare;  |
| 1000000030 | STG-F1-B7D27BA79C-LOT; product 1000000022 STG-F1-B7D27BA79C ingredient; kosher=regular | 88.0000 → 0 | planned; /adjust/prepare;  |
| 1000000033 | STG-F1-B7D27BA79C-FOUND; product 1000000022 STG-F1-B7D27BA79C ingredient; kosher=regular | 2.0000 → 0 | planned; /adjust/prepare;  |
| 1000000034 | STG-A3B-0D78DDA6AF2C-ING; product 1000000027 STG-A3B-0D78DDA6AF2C ingredient; kosher=regular | -606.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000036 | STG-F1-EC5678F88B-LOT; product 1000000029 STG-F1-EC5678F88B ingredient; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000037 | STG-F1-7FF9E7880F-LOT; product 1000000032 STG-F1-7FF9E7880F ingredient; kosher=regular | -610.0000 → 0 | blocked; /adjust/prepare; SHORTAGE_OPEN_RESOLVE_INSTEAD: positive correction would cover an open shortage; waiver unsupported. |
| 1000000040 | STG-F1-7FF9E7880F-FOUND; product 1000000034 STG-F1-7FF9E7880F finished; kosher=regular | 2.0000 → 0 | planned; /adjust/prepare;  |
| 1000000041 | STG-A6-EC3429B9180B-LOT; product 1000000035 STG-A6-EC3429B9180B granola; kosher=regular | 50.0000 → 0 | planned; /adjust/prepare;  |
| 1000000042 | STG-A6H-62A8547011-LOT; product 1000000036 STG-A6H-62A8547011 granola; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000043 | STG-A6H-83911280CA-LOT; product 1000000037 STG-A6H-83911280CA granola; kosher=regular | 75.0000 → 0 | planned; /adjust/prepare;  |
| 1000000044 | STG-A6H-A52A97B6FF-LOT; product 1000000038 STG-A6H-A52A97B6FF granola; kosher=regular | 50.0000 → 0 | planned; /adjust/prepare;  |
| 1000000045 | STG-A12-46454C41E879-ING; product 1000000039 STG-A12-46454C41E879 ingredient; kosher=regular | 80.0000 → 0 | planned; /adjust/prepare;  |
| 1000000049 | STG-A9-573C5E4443C3-LOT; product 1000000043 STG-A9-573C5E4443C3 oats; kosher=regular | 97.0000 → 0 | planned; /adjust/prepare;  |
| 1000000050 | STG-A9-AFDC1124A00C-LOT; product 1000000044 STG-A9-AFDC1124A00C oats; kosher=regular | 97.0000 → 0 | planned; /adjust/prepare;  |
| 1000000051 | STG-F1-AABCF507E9-LOT; product 1000000045 STG-F1-AABCF507E9 ingredient; kosher=regular | -610.0000 → 0 | blocked; /adjust/prepare; SHORTAGE_OPEN_RESOLVE_INSTEAD: positive correction would cover an open shortage; waiver unsupported. |
| 1000000052 | STG-A9-9901D12E1A33-LOT; product 1000000048 STG-A9-9901D12E1A33 oats; kosher=regular | 97.0000 → 0 | planned; /adjust/prepare;  |
| 1000000055 | STG-F1-AABCF507E9-FOUND; product 1000000047 STG-F1-AABCF507E9 finished; kosher=regular | 10.0000 → 0 | planned; /adjust/prepare;  |
| 1000000057 | STG-078-CLASSIC9; product 107 Batch Classic Granola #9; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000058 | STG-078-ALMONDS; product 3 Almonds – Sliced; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000059 | STG-078-WALNUTS; product 49 Walnuts; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000060 | STG-078-RAISINS; product 36 Raisins; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000061 | STG-078-CRANBERRIES; product 17 Cranberries Dried; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000062 | STG-078-CHIPS-4000CT; product 72 Chocolate Chips – Real – 4,000 CT; kosher=regular | 10.0000 → 0 | planned; /adjust/prepare;  |
| 1000000063 | STG-078-COCONUT; product 13 Coconut Macaroon Desiccated; kosher=regular | 7.5000 → 0 | planned; /adjust/prepare;  |
| 1000000064 | STG-078-ALMOND-FLAVOR; product 18 Flavor – Almond; kosher=regular | 1.4000 → 0 | planned; /adjust/prepare;  |
| 1000000065 | STG-078-HONEY; product 30 Honey; kosher=regular | 1.8000 → 0 | planned; /adjust/prepare;  |
| 1000000066 | STG-078-OATS; product 32 Oats; kosher=regular | 75.0000 → 0 | planned; /adjust/prepare;  |
| 1000000067 | STG-078-CANOLA; product 34 Oil – Canola; kosher=regular | 10.0000 → 0 | planned; /adjust/prepare;  |
| 1000000068 | STG-078-SALT; product 39 Salt; kosher=regular | 1.6000 → 0 | planned; /adjust/prepare;  |
| 1000000069 | STG-078-BROWN-SUGAR; product 44 Sugar – Light Brown; kosher=regular | 20.0000 → 0 | planned; /adjust/prepare;  |
| 1000000070 | STG-078-FRUIT-NUT-001; product 179 Granola Fruit Nut Batch; kosher=regular | 328.3000 → 0 | planned; /adjust/prepare;  |
| 1000000071 | STG-078-KOSHER-CC-001; product 284 Batch Classic Chocolate Chip Granola #9 — EXTRA-KOSHER (SS); kosher=extra_kosher | 348.0000 → 0 | planned; /adjust/prepare;  |
| 1000000073 | STG-078-R2-CLASSIC9; product 107 Batch Classic Granola #9; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000074 | STG-078-R2-ALMONDS; product 3 Almonds – Sliced; kosher=regular | 21.0000 → 0 | planned; /adjust/prepare;  |
| 1000000075 | STG-078-R2-WALNUTS; product 49 Walnuts; kosher=regular | 21.0000 → 0 | planned; /adjust/prepare;  |
| 1000000076 | STG-078-R2-RAISINS; product 36 Raisins; kosher=regular | 21.0000 → 0 | planned; /adjust/prepare;  |
| 1000000077 | STG-078-R2-CRANBERRIES; product 17 Cranberries Dried; kosher=regular | 21.0000 → 0 | planned; /adjust/prepare;  |
| 1000000078 | STG-078-R2-CHIPS-4000CT; product 72 Chocolate Chips – Real – 4,000 CT; kosher=regular | 60.0000 → 0 | planned; /adjust/prepare;  |
| 1000000079 | STG-078-R2-COCONUT; product 13 Coconut Macaroon Desiccated; kosher=regular | 7.5000 → 0 | planned; /adjust/prepare;  |
| 1000000080 | STG-078-R2-ALMOND-FLAVOR; product 18 Flavor – Almond; kosher=regular | 5.0000 → 0 | planned; /adjust/prepare;  |
| 1000000081 | STG-078-R2-HONEY; product 30 Honey; kosher=regular | 5.0000 → 0 | planned; /adjust/prepare;  |
| 1000000082 | STG-078-R2-OATS; product 32 Oats; kosher=regular | 75.0000 → 0 | planned; /adjust/prepare;  |
| 1000000083 | STG-078-R2-CANOLA; product 34 Oil – Canola; kosher=regular | 10.0000 → 0 | planned; /adjust/prepare;  |
| 1000000084 | STG-078-R2-SALT; product 39 Salt; kosher=regular | 5.0000 → 0 | planned; /adjust/prepare;  |
| 1000000085 | STG-078-R2-BROWN-SUGAR; product 44 Sugar – Light Brown; kosher=regular | 20.0000 → 0 | planned; /adjust/prepare;  |
| 1000000086 | STG-078-FRUIT-NUT-002; product 179 Granola Fruit Nut Batch; kosher=regular | 328.3000 → 0 | planned; /adjust/prepare;  |
| 1000000087 | STG-078-R3-CLASSIC9; product 107 Batch Classic Granola #9; kosher=regular | 371.1000 → 0 | planned; /adjust/prepare;  |
| 1000000088 | STG-078-R3-ALMONDS; product 3 Almonds – Sliced; kosher=regular | 30.0000 → 0 | planned; /adjust/prepare;  |
| 1000000089 | STG-078-R3-WALNUTS; product 49 Walnuts; kosher=regular | 30.0000 → 0 | planned; /adjust/prepare;  |
| 1000000090 | STG-078-R3-RAISINS; product 36 Raisins; kosher=regular | 30.0000 → 0 | planned; /adjust/prepare;  |
| 1000000091 | STG-078-R3-CRANBERRIES; product 17 Cranberries Dried; kosher=regular | 30.0000 → 0 | planned; /adjust/prepare;  |
| 1000000092 | STG-078-R3-CHIPS-4000CT; product 72 Chocolate Chips – Real – 4,000 CT; kosher=regular | 60.0000 → 0 | planned; /adjust/prepare;  |
| 1000000093 | STG-078-R3-COCONUT; product 13 Coconut Macaroon Desiccated; kosher=regular | 20.0000 → 0 | planned; /adjust/prepare;  |
| 1000000094 | STG-078-R3-ALMOND-FLAVOR; product 18 Flavor – Almond; kosher=regular | 5.0000 → 0 | planned; /adjust/prepare;  |
| 1000000095 | STG-078-R3-HONEY; product 30 Honey; kosher=regular | 5.0000 → 0 | planned; /adjust/prepare;  |
| 1000000096 | STG-078-R3-OATS; product 32 Oats; kosher=regular | 300.0000 → 0 | planned; /adjust/prepare;  |
| 1000000097 | STG-078-R3-CANOLA; product 34 Oil – Canola; kosher=regular | 40.0000 → 0 | planned; /adjust/prepare;  |
| 1000000098 | STG-078-R3-SALT; product 39 Salt; kosher=regular | 5.0000 → 0 | planned; /adjust/prepare;  |
| 1000000099 | STG-078-R3-BROWN-SUGAR; product 44 Sugar – Light Brown; kosher=regular | 70.0000 → 0 | planned; /adjust/prepare;  |
| 1000000100 | STG-078-FRUIT-NUT-003; product 179 Granola Fruit Nut Batch; kosher=regular | 328.3000 → 0 | planned; /adjust/prepare;  |
| 1000000101 | STG-PR104-CODEX-18F18501E995-284; product 284 Batch Classic Chocolate Chip Granola #9 — EXTRA-KOSHER (SS); kosher=extra_kosher | 348.0000 → 0 | planned; /adjust/prepare;  |
| 1000000102 | STG-A5P2-F9454E5513; product 1000000051 STG-A5P2-F9454E5513 synthetic ingredient; kosher=regular | 4.0000 → 0 | planned; /adjust/prepare;  |
| 1000000108 | STG-CNT-A575D9-OATS-1; product 1000000058 STG STG-CNT-A575D9 Oats; kosher=regular | 400.0000 → 0 | planned; /adjust/prepare;  |
| 1000000109 | STG-CNT-A575D9-SUGAR-1; product 1000000059 STG STG-CNT-A575D9 Sugar; kosher=regular | 1200.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000110 | STG-CNT-A575D9-COCONUT-1; product 1000000060 STG STG-CNT-A575D9 Coconut; kosher=regular | -20.0000 → 0 | blocked; /adjust/prepare; SHORTAGE_OPEN_RESOLVE_INSTEAD: positive correction would cover an open shortage; waiver unsupported. |
| 1000000111 | STG-CNT-A575D9-GRANOLA-1; product 1000000062 STG STG-CNT-A575D9 Granola 25 LB; kosher=regular | 1000.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000113 | STG-CNT-3A3FEE-OATS-1; product 1000000063 STG STG-CNT-3A3FEE Oats; kosher=regular | 395.0000 → 0 | planned; /adjust/prepare;  |
| 1000000114 | STG-CNT-3A3FEE-SUGAR-1; product 1000000064 STG STG-CNT-3A3FEE Sugar; kosher=regular | 520.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000115 | STG-CNT-3A3FEE-COCONUT-1; product 1000000065 STG STG-CNT-3A3FEE Coconut; kosher=regular | -20.0000 → 0 | blocked; /adjust/prepare; SHORTAGE_OPEN_RESOLVE_INSTEAD: positive correction would cover an open shortage; waiver unsupported. |
| 1000000116 | STG-CNT-3A3FEE-GRANOLA-1; product 1000000067 STG STG-CNT-3A3FEE Granola 25 LB; kosher=regular | 1000.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000118 | STG-CNT-C5CDEB-OATS-1; product 1000000068 STG STG-CNT-C5CDEB Oats; kosher=regular | 395.0000 → 0 | planned; /adjust/prepare;  |
| 1000000119 | STG-CNT-C5CDEB-SUGAR-1; product 1000000069 STG STG-CNT-C5CDEB Sugar; kosher=regular | 520.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000120 | STG-CNT-C5CDEB-COCONUT-1; product 1000000070 STG STG-CNT-C5CDEB Coconut; kosher=regular | 25.0000 → 0 | planned; /adjust/prepare;  |
| 1000000121 | STG-CNT-C5CDEB-GRANOLA-1; product 1000000072 STG STG-CNT-C5CDEB Granola 25 LB; kosher=regular | 975.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000123 | STG-CNT-C5CDEB-EK-1; product 285 Granola SS Classic #9 Bulk per/lb — EXTRA-KOSHER (SS); kosher=extra_kosher | 180.0000 → 0 | planned; /adjust/prepare;  |
| 1000000124 | STG-CNT-EA2A30-OATS-1; product 1000000073 STG STG-CNT-EA2A30 Oats; kosher=regular | 395.0000 → 0 | planned; /adjust/prepare;  |
| 1000000125 | STG-CNT-EA2A30-SUGAR-1; product 1000000074 STG STG-CNT-EA2A30 Sugar; kosher=regular | 520.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000126 | STG-CNT-EA2A30-COCONUT-1; product 1000000075 STG STG-CNT-EA2A30 Coconut; kosher=regular | 25.0000 → 0 | planned; /adjust/prepare;  |
| 1000000127 | STG-CNT-EA2A30-GRANOLA-1; product 1000000077 STG STG-CNT-EA2A30 Granola 25 LB; kosher=regular | 950.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000129 | STG-CNT-EA2A30-EK-1; product 288 Granola SS Classic Chocolate Chip #9 Bulk per/lb — EXTRA-KOSHER (SS); kosher=extra_kosher | 180.0000 → 0 | planned; /adjust/prepare;  |
| 1000000130 | STG-CNT-9F7F22-OATS-1; product 1000000078 STG STG-CNT-9F7F22 Oats; kosher=regular | 395.0000 → 0 | planned; /adjust/prepare;  |
| 1000000131 | STG-CNT-9F7F22-SUGAR-1; product 1000000079 STG STG-CNT-9F7F22 Sugar; kosher=regular | 520.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000132 | STG-CNT-9F7F22-COCONUT-1; product 1000000080 STG STG-CNT-9F7F22 Coconut; kosher=regular | 25.0000 → 0 | planned; /adjust/prepare;  |
| 1000000133 | STG-CNT-9F7F22-GRANOLA-1; product 1000000082 STG STG-CNT-9F7F22 Granola 25 LB; kosher=regular | 950.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000134 | STG-CNT-9F7F22-MISC1-1; product 1000000083 STG STG-CNT-9F7F22 Ingredient 1; kosher=regular | 99.0000 → 0 | planned; /adjust/prepare;  |
| 1000000135 | STG-CNT-9F7F22-MISC2-1; product 1000000084 STG STG-CNT-9F7F22 Ingredient 2; kosher=regular | 98.0000 → 0 | planned; /adjust/prepare;  |
| 1000000136 | STG-CNT-9F7F22-MISC3-1; product 1000000085 STG STG-CNT-9F7F22 Ingredient 3; kosher=regular | 97.0000 → 0 | planned; /adjust/prepare;  |
| 1000000137 | STG-CNT-9F7F22-MISC4-1; product 1000000086 STG STG-CNT-9F7F22 Ingredient 4; kosher=regular | 96.0000 → 0 | planned; /adjust/prepare;  |
| 1000000138 | STG-CNT-9F7F22-MISC5-1; product 1000000087 STG STG-CNT-9F7F22 Ingredient 5; kosher=regular | 95.0000 → 0 | planned; /adjust/prepare;  |
| 1000000139 | STG-CNT-9F7F22-MISC6-1; product 1000000088 STG STG-CNT-9F7F22 Ingredient 6; kosher=regular | 94.0000 → 0 | planned; /adjust/prepare;  |
| 1000000141 | STG-CNT-9F7F22-EK-1; product 283 Batch Classic Granola #9 — EXTRA-KOSHER (SS); kosher=extra_kosher | 180.0000 → 0 | planned; /adjust/prepare;  |
| 1000000142 | STG-XR-CCAF19C68F-INGREDIENT; product 1000000089 STG-XR-CCAF19C68F ingredient; kosher=regular | 4.0000 → 0 | planned; /adjust/prepare;  |
| 1000000144 | STG-XR-CCAF19C68F-CASE; product 1000000091 STG-XR-CCAF19C68F case; kosher=regular | 1600.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000164 | STG-CNT-625A72-OATS-1; product 1000000116 STG STG-CNT-625A72 Oats; kosher=regular | 395.0000 → 0 | planned; /adjust/prepare;  |
| 1000000165 | STG-CNT-625A72-SUGAR-1; product 1000000117 STG STG-CNT-625A72 Sugar; kosher=regular | 520.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000166 | STG-CNT-625A72-COCONUT-1; product 1000000118 STG STG-CNT-625A72 Coconut; kosher=regular | 25.0000 → 0 | planned; /adjust/prepare;  |
| 1000000167 | STG-CNT-625A72-GRANOLA-1; product 1000000120 STG STG-CNT-625A72 Granola 25 LB; kosher=regular | 950.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000168 | STG-CNT-625A72-CASES15-1; product 1000000121 STG STG-CNT-625A72 Granola 15 LB case; kosher=regular | 255.0000 → 0 | planned; /adjust/prepare;  |
| 1000000169 | STG-CNT-625A72-MISC1-1; product 1000000122 STG STG-CNT-625A72 Ingredient 1; kosher=regular | 99.0000 → 0 | planned; /adjust/prepare;  |
| 1000000170 | STG-CNT-625A72-MISC2-1; product 1000000123 STG STG-CNT-625A72 Ingredient 2; kosher=regular | 98.0000 → 0 | planned; /adjust/prepare;  |
| 1000000171 | STG-CNT-625A72-MISC3-1; product 1000000124 STG STG-CNT-625A72 Ingredient 3; kosher=regular | 97.0000 → 0 | planned; /adjust/prepare;  |
| 1000000172 | STG-CNT-625A72-MISC4-1; product 1000000125 STG STG-CNT-625A72 Ingredient 4; kosher=regular | 96.0000 → 0 | planned; /adjust/prepare;  |
| 1000000173 | STG-CNT-625A72-MISC5-1; product 1000000126 STG STG-CNT-625A72 Ingredient 5; kosher=regular | 95.0000 → 0 | planned; /adjust/prepare;  |
| 1000000174 | STG-CNT-625A72-MISC6-1; product 1000000127 STG STG-CNT-625A72 Ingredient 6; kosher=regular | 94.0000 → 0 | planned; /adjust/prepare;  |
| 1000000176 | STG-CNT-625A72-EK-1; product 1000000128 STG STG-CNT-625A72 Extra-kosher batch; kosher=extra_kosher | 300.0000 → 0 | planned; /adjust/prepare;  |
| 1000000177 | STG-PILOT-261014-D60C7286-OATS-UNKNOWN; product 32 Oats; kosher=regular | 601.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000178 | STG-PILOT-261014-D60C7286-CLASSIC; product 107 Batch Classic Granola #9; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000179 | STG-PILOT-261014-D60C7286-CRANBERRIES; product 17 Cranberries Dried; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000180 | STG-PILOT-261014-D60C7286-RAISINS; product 36 Raisins; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000181 | STG-PILOT-261014-D60C7286-WALNUTS; product 49 Walnuts; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000182 | STG-PILOT-261014-D60C7286-ALMONDS; product 3 Almonds – Sliced; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000183 | STG-PILOT-261014-D60C7286-FRUIT-NUT; product 179 Granola Fruit Nut Batch; kosher=regular | 3.3000 → 0 | planned; /adjust/prepare;  |
| 1000000184 | STG-PILOT-261014-D60C7286-FRUIT-FINISHED; product 142 Granola Fruit Nut 25 LB; kosher=regular | 250.0000 → 0 | planned; /adjust/prepare;  |
| 1000000185 | STG-PILOT-261014-D60C7286-EXTRA-KOSHER; product 283 Batch Classic Granola #9 — EXTRA-KOSHER (SS); kosher=extra_kosher | 298.0000 → 0 | planned; /adjust/prepare;  |
| 1000000186 | STG-PILOT-261014-D60C7286-KOSHER-FINISHED; product 285 Granola SS Classic #9 Bulk per/lb — EXTRA-KOSHER (SS); kosher=extra_kosher | 25.0000 → 0 | planned; /adjust/prepare;  |
| 1000000187 | STG-PILOT-261014-D60C7286-SHORTAGE-MAKE; product 179 Granola Fruit Nut Batch; kosher=regular | 328.3000 → 0 | planned; /adjust/prepare;  |
| 1000000188 | STG-CNT-A13CF3-OATS-1; product 1000000129 STG STG-CNT-A13CF3 Oats; kosher=regular | 395.0000 → 0 | planned; /adjust/prepare;  |
| 1000000189 | STG-CNT-A13CF3-SUGAR-1; product 1000000130 STG STG-CNT-A13CF3 Sugar; kosher=regular | 520.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000190 | STG-CNT-A13CF3-COCONUT-1; product 1000000131 STG STG-CNT-A13CF3 Coconut; kosher=regular | 25.0000 → 0 | planned; /adjust/prepare;  |
| 1000000191 | STG-CNT-A13CF3-GRANOLA-1; product 1000000133 STG STG-CNT-A13CF3 Granola 25 LB; kosher=regular | 1000.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000192 | STG-CNT-A13CF3-CASES15-1; product 1000000134 STG STG-CNT-A13CF3 Granola 15 LB case; kosher=regular | 255.0000 → 0 | planned; /adjust/prepare;  |
| 1000000193 | STG-CNT-A13CF3-MISC1-1; product 1000000135 STG STG-CNT-A13CF3 Ingredient 1; kosher=regular | 99.0000 → 0 | planned; /adjust/prepare;  |
| 1000000194 | STG-CNT-A13CF3-MISC2-1; product 1000000136 STG STG-CNT-A13CF3 Ingredient 2; kosher=regular | 98.0000 → 0 | planned; /adjust/prepare;  |
| 1000000195 | STG-CNT-A13CF3-MISC3-1; product 1000000137 STG STG-CNT-A13CF3 Ingredient 3; kosher=regular | 97.0000 → 0 | planned; /adjust/prepare;  |
| 1000000196 | STG-CNT-A13CF3-MISC4-1; product 1000000138 STG STG-CNT-A13CF3 Ingredient 4; kosher=regular | 96.0000 → 0 | planned; /adjust/prepare;  |
| 1000000197 | STG-CNT-A13CF3-MISC5-1; product 1000000139 STG STG-CNT-A13CF3 Ingredient 5; kosher=regular | 95.0000 → 0 | planned; /adjust/prepare;  |
| 1000000198 | STG-CNT-A13CF3-MISC6-1; product 1000000140 STG STG-CNT-A13CF3 Ingredient 6; kosher=regular | 94.0000 → 0 | planned; /adjust/prepare;  |
| 1000000200 | STG-CNT-A13CF3-EK-1; product 1000000141 STG STG-CNT-A13CF3 Extra-kosher batch; kosher=extra_kosher | 180.0000 → 0 | planned; /adjust/prepare;  |
| 1000000407 | STG-TRIAGE-4860BBF0F5-00; product 3 Almonds – Sliced; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000408 | STG-TRIAGE-4860BBF0F5-01; product 13 Coconut Macaroon Desiccated; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000409 | STG-TRIAGE-4860BBF0F5-02; product 30 Honey; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000410 | STG-TRIAGE-4860BBF0F5-03; product 32 Oats; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000411 | STG-TRIAGE-4860BBF0F5-04; product 34 Oil – Canola; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000412 | STG-TRIAGE-4860BBF0F5-05; product 46 Sunflower Seeds; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000413 | STG-TRIAGE-4860BBF0F5-06; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000414 | STG-TRIAGE-4860BBF0F5-07; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000415 | STG-TRIAGE-4860BBF0F5-08; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000416 | STG-TRIAGE-4860BBF0F5-09; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000417 | STG-TRIAGE-4860BBF0F5-10; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000418 | STG-TRIAGE-4860BBF0F5-11; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000419 | STG-TRIAGE-4860BBF0F5-12; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000420 | STG-TRIAGE-4860BBF0F5-13; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000421 | STG-TRIAGE-4860BBF0F5-14; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000422 | STG-TRIAGE-4860BBF0F5-15; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000423 | STG-TRIAGE-4860BBF0F5-16; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000424 | STG-TRIAGE-4860BBF0F5-17; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000425 | STG-TRIAGE-4860BBF0F5-18; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000426 | STG-TRIAGE-4860BBF0F5-19; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000427 | STG-TRIAGE-4860BBF0F5-20; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000428 | STG-TRIAGE-4860BBF0F5-21; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000429 | STG-TRIAGE-4860BBF0F5-22; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000430 | STG-TRIAGE-4860BBF0F5-23; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000431 | STG-TRIAGE-4860BBF0F5-24; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000432 | STG-TRIAGE-4860BBF0F5-25; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000433 | STG-TRIAGE-4860BBF0F5-26; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000434 | STG-TRIAGE-4860BBF0F5-27; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000435 | STG-TRIAGE-4860BBF0F5-28; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000436 | STG-TRIAGE-4860BBF0F5-SS1; product 116 Batch SS Original Granola #1; kosher=regular | 350.0000 → 0 | planned; /adjust/prepare;  |
| 1000000437 | STG-TRIAGE-C31C73D8FC-00; product 3 Almonds – Sliced; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000438 | STG-TRIAGE-C31C73D8FC-01; product 13 Coconut Macaroon Desiccated; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000439 | STG-TRIAGE-C31C73D8FC-02; product 30 Honey; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000440 | STG-TRIAGE-C31C73D8FC-03; product 32 Oats; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000441 | STG-TRIAGE-C31C73D8FC-04; product 34 Oil – Canola; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000442 | STG-TRIAGE-C31C73D8FC-05; product 46 Sunflower Seeds; kosher=regular | 0.0000 → 0 | noop; /adjust/prepare; Already zero; no zero-delta ticket. |
| 1000000443 | STG-TRIAGE-C31C73D8FC-06; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000444 | STG-TRIAGE-C31C73D8FC-07; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000445 | STG-TRIAGE-C31C73D8FC-08; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000446 | STG-TRIAGE-C31C73D8FC-09; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000447 | STG-TRIAGE-C31C73D8FC-10; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000448 | STG-TRIAGE-C31C73D8FC-11; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000449 | STG-TRIAGE-C31C73D8FC-12; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000450 | STG-TRIAGE-C31C73D8FC-13; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000451 | STG-TRIAGE-C31C73D8FC-14; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000452 | STG-TRIAGE-C31C73D8FC-15; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000453 | STG-TRIAGE-C31C73D8FC-16; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000454 | STG-TRIAGE-C31C73D8FC-17; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000455 | STG-TRIAGE-C31C73D8FC-18; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000456 | STG-TRIAGE-C31C73D8FC-19; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000457 | STG-TRIAGE-C31C73D8FC-20; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000458 | STG-TRIAGE-C31C73D8FC-21; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000459 | STG-TRIAGE-C31C73D8FC-22; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000460 | STG-TRIAGE-C31C73D8FC-23; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000461 | STG-TRIAGE-C31C73D8FC-24; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000462 | STG-TRIAGE-C31C73D8FC-25; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000463 | STG-TRIAGE-C31C73D8FC-26; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000464 | STG-TRIAGE-C31C73D8FC-27; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000465 | STG-TRIAGE-C31C73D8FC-28; product 32 Oats; kosher=regular | 1.0000 → 0 | planned; /adjust/prepare;  |
| 1000000466 | STG-TRIAGE-C31C73D8FC-SS1; product 116 Batch SS Original Granola #1; kosher=regular | 350.0000 → 0 | planned; /adjust/prepare;  |
| 1000000467 | STG-CNT-785330-OATS-1; product 1000000145 STG STG-CNT-785330 Oats; kosher=regular | 396.0000 → 0 | planned; /adjust/prepare;  |
| 1000000468 | STG-CNT-785330-SUGAR-1; product 1000000146 STG STG-CNT-785330 Sugar; kosher=regular | 520.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000469 | STG-CNT-785330-COCONUT-1; product 1000000147 STG STG-CNT-785330 Coconut; kosher=regular | 10.0000 → 0 | planned; /adjust/prepare;  |
| 1000000470 | STG-CNT-785330-GRANOLA-1; product 1000000149 STG STG-CNT-785330 Granola 25 LB; kosher=regular | 975.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000471 | STG-CNT-785330-CASES15-1; product 1000000150 STG STG-CNT-785330 Granola 15 LB case; kosher=regular | 300.0000 → 0 | planned; /adjust/prepare;  |
| 1000000472 | STG-CNT-785330-MISC1-1; product 1000000151 STG STG-CNT-785330 Ingredient 1; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000473 | STG-CNT-785330-MISC2-1; product 1000000152 STG STG-CNT-785330 Ingredient 2; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000474 | STG-CNT-785330-MISC3-1; product 1000000153 STG STG-CNT-785330 Ingredient 3; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000475 | STG-CNT-785330-MISC4-1; product 1000000154 STG STG-CNT-785330 Ingredient 4; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000476 | STG-CNT-785330-MISC5-1; product 1000000155 STG STG-CNT-785330 Ingredient 5; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000477 | STG-CNT-785330-MISC6-1; product 1000000156 STG STG-CNT-785330 Ingredient 6; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000478 | STG-CNT-785330-EK-1; product 1000000157 STG STG-CNT-785330 Extra-kosher batch; kosher=extra_kosher | 300.0000 → 0 | planned; /adjust/prepare;  |
| 1000000479 | STG-CNT-947168-OATS-1; product 1000000158 STG STG-CNT-947168 Oats; kosher=regular | 396.0000 → 0 | planned; /adjust/prepare;  |
| 1000000480 | STG-CNT-947168-SUGAR-1; product 1000000159 STG STG-CNT-947168 Sugar; kosher=regular | 520.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000481 | STG-CNT-947168-COCONUT-1; product 1000000160 STG STG-CNT-947168 Coconut; kosher=regular | 10.0000 → 0 | planned; /adjust/prepare;  |
| 1000000482 | STG-CNT-947168-GRANOLA-1; product 1000000162 STG STG-CNT-947168 Granola 25 LB; kosher=regular | 975.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000483 | STG-CNT-947168-CASES15-1; product 1000000163 STG STG-CNT-947168 Granola 15 LB case; kosher=regular | 300.0000 → 0 | planned; /adjust/prepare;  |
| 1000000484 | STG-CNT-947168-MISC1-1; product 1000000164 STG STG-CNT-947168 Ingredient 1; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000485 | STG-CNT-947168-MISC2-1; product 1000000165 STG STG-CNT-947168 Ingredient 2; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000486 | STG-CNT-947168-MISC3-1; product 1000000166 STG STG-CNT-947168 Ingredient 3; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000487 | STG-CNT-947168-MISC4-1; product 1000000167 STG STG-CNT-947168 Ingredient 4; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000488 | STG-CNT-947168-MISC5-1; product 1000000168 STG STG-CNT-947168 Ingredient 5; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000489 | STG-CNT-947168-MISC6-1; product 1000000169 STG STG-CNT-947168 Ingredient 6; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000490 | STG-CNT-947168-EK-1; product 1000000170 STG STG-CNT-947168 Extra-kosher batch; kosher=extra_kosher | 300.0000 → 0 | planned; /adjust/prepare;  |
| 1000000491 | STG-CNT-36A1AD-OATS-1; product 1000000171 STG STG-CNT-36A1AD Oats; kosher=regular | 396.0000 → 0 | planned; /adjust/prepare;  |
| 1000000492 | STG-CNT-36A1AD-SUGAR-1; product 1000000172 STG STG-CNT-36A1AD Sugar; kosher=regular | 520.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000493 | STG-CNT-36A1AD-COCONUT-1; product 1000000173 STG STG-CNT-36A1AD Coconut; kosher=regular | 10.0000 → 0 | planned; /adjust/prepare;  |
| 1000000494 | STG-CNT-36A1AD-GRANOLA-1; product 1000000175 STG STG-CNT-36A1AD Granola 25 LB; kosher=regular | 975.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000495 | STG-CNT-36A1AD-CASES15-1; product 1000000176 STG STG-CNT-36A1AD Granola 15 LB case; kosher=regular | 300.0000 → 0 | planned; /adjust/prepare;  |
| 1000000496 | STG-CNT-36A1AD-MISC1-1; product 1000000177 STG STG-CNT-36A1AD Ingredient 1; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000497 | STG-CNT-36A1AD-MISC2-1; product 1000000178 STG STG-CNT-36A1AD Ingredient 2; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000498 | STG-CNT-36A1AD-MISC3-1; product 1000000179 STG STG-CNT-36A1AD Ingredient 3; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000499 | STG-CNT-36A1AD-MISC4-1; product 1000000180 STG STG-CNT-36A1AD Ingredient 4; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000500 | STG-CNT-36A1AD-MISC5-1; product 1000000181 STG STG-CNT-36A1AD Ingredient 5; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000501 | STG-CNT-36A1AD-MISC6-1; product 1000000182 STG STG-CNT-36A1AD Ingredient 6; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000502 | STG-CNT-36A1AD-EK-1; product 1000000183 STG STG-CNT-36A1AD Extra-kosher batch; kosher=extra_kosher | 300.0000 → 0 | planned; /adjust/prepare;  |
| 1000000503 | STG-CNT-DCF5EB-OATS-1; product 1000000184 STG STG-CNT-DCF5EB Oats; kosher=regular | 396.0000 → 0 | planned; /adjust/prepare;  |
| 1000000504 | STG-CNT-DCF5EB-SUGAR-1; product 1000000185 STG STG-CNT-DCF5EB Sugar; kosher=regular | 520.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000505 | STG-CNT-DCF5EB-COCONUT-1; product 1000000186 STG STG-CNT-DCF5EB Coconut; kosher=regular | 10.0000 → 0 | planned; /adjust/prepare;  |
| 1000000506 | STG-CNT-DCF5EB-GRANOLA-1; product 1000000188 STG STG-CNT-DCF5EB Granola 25 LB; kosher=regular | 977.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000507 | STG-CNT-DCF5EB-CASES15-1; product 1000000189 STG STG-CNT-DCF5EB Granola 15 LB case; kosher=regular | 300.0000 → 0 | planned; /adjust/prepare;  |
| 1000000508 | STG-CNT-DCF5EB-MISC1-1; product 1000000190 STG STG-CNT-DCF5EB Ingredient 1; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000509 | STG-CNT-DCF5EB-MISC2-1; product 1000000191 STG STG-CNT-DCF5EB Ingredient 2; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000510 | STG-CNT-DCF5EB-MISC3-1; product 1000000192 STG STG-CNT-DCF5EB Ingredient 3; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000511 | STG-CNT-DCF5EB-MISC4-1; product 1000000193 STG STG-CNT-DCF5EB Ingredient 4; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000512 | STG-CNT-DCF5EB-MISC5-1; product 1000000194 STG STG-CNT-DCF5EB Ingredient 5; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000513 | STG-CNT-DCF5EB-MISC6-1; product 1000000195 STG STG-CNT-DCF5EB Ingredient 6; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000514 | STG-CNT-DCF5EB-EK-1; product 1000000196 STG STG-CNT-DCF5EB Extra-kosher batch; kosher=extra_kosher | 300.0000 → 0 | planned; /adjust/prepare;  |
| 1000000515 | STG-CNT-780841-OATS-1; product 1000000197 STG STG-CNT-780841 Oats; kosher=regular | 396.0000 → 0 | planned; /adjust/prepare;  |
| 1000000516 | STG-CNT-780841-SUGAR-1; product 1000000198 STG STG-CNT-780841 Sugar; kosher=regular | 520.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000517 | STG-CNT-780841-COCONUT-1; product 1000000199 STG STG-CNT-780841 Coconut; kosher=regular | 10.0000 → 0 | planned; /adjust/prepare;  |
| 1000000518 | STG-CNT-780841-GRANOLA-1; product 1000000201 STG STG-CNT-780841 Granola 25 LB; kosher=regular | 977.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000519 | STG-CNT-780841-CASES15-1; product 1000000202 STG STG-CNT-780841 Granola 15 LB case; kosher=regular | 300.0000 → 0 | planned; /adjust/prepare;  |
| 1000000520 | STG-CNT-780841-MISC1-1; product 1000000203 STG STG-CNT-780841 Ingredient 1; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000521 | STG-CNT-780841-MISC2-1; product 1000000204 STG STG-CNT-780841 Ingredient 2; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000522 | STG-CNT-780841-MISC3-1; product 1000000205 STG STG-CNT-780841 Ingredient 3; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000523 | STG-CNT-780841-MISC4-1; product 1000000206 STG STG-CNT-780841 Ingredient 4; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000524 | STG-CNT-780841-MISC5-1; product 1000000207 STG STG-CNT-780841 Ingredient 5; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000525 | STG-CNT-780841-MISC6-1; product 1000000208 STG STG-CNT-780841 Ingredient 6; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000526 | STG-CNT-780841-EK-1; product 1000000210 STG STG-CNT-780841 Extra-kosher batch; kosher=extra_kosher | 300.0000 → 0 | planned; /adjust/prepare;  |
| 1000000527 | STG-CNT-08370B-OATS-1; product 1000000211 STG STG-CNT-08370B Oats; kosher=regular | 396.0000 → 0 | planned; /adjust/prepare;  |
| 1000000528 | STG-CNT-08370B-SUGAR-1; product 1000000212 STG STG-CNT-08370B Sugar; kosher=regular | 520.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000529 | STG-CNT-08370B-COCONUT-1; product 1000000213 STG STG-CNT-08370B Coconut; kosher=regular | 10.0000 → 0 | planned; /adjust/prepare;  |
| 1000000530 | STG-CNT-08370B-GRANOLA-1; product 1000000215 STG STG-CNT-08370B Granola 25 LB; kosher=regular | 977.0000 → 0 | planned; /adjust/prepare; Over 500 lb: normal LARGE_CORRECTION hold, then named owner approval and ADJ + XR receipts. |
| 1000000531 | STG-CNT-08370B-CASES15-1; product 1000000216 STG STG-CNT-08370B Granola 15 LB case; kosher=regular | 300.0000 → 0 | planned; /adjust/prepare;  |
| 1000000532 | STG-CNT-08370B-MISC1-1; product 1000000217 STG STG-CNT-08370B Ingredient 1; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000533 | STG-CNT-08370B-MISC2-1; product 1000000218 STG STG-CNT-08370B Ingredient 2; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000534 | STG-CNT-08370B-MISC3-1; product 1000000219 STG STG-CNT-08370B Ingredient 3; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000535 | STG-CNT-08370B-MISC4-1; product 1000000220 STG STG-CNT-08370B Ingredient 4; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000536 | STG-CNT-08370B-MISC5-1; product 1000000221 STG STG-CNT-08370B Ingredient 5; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000537 | STG-CNT-08370B-MISC6-1; product 1000000222 STG STG-CNT-08370B Ingredient 6; kosher=regular | 100.0000 → 0 | planned; /adjust/prepare;  |
| 1000000538 | STG-CNT-08370B-EK-1; product 1000000224 STG STG-CNT-08370B Extra-kosher batch; kosher=extra_kosher | 300.0000 → 0 | planned; /adjust/prepare;  |

## Products

| ID | Item | Before → desired after | Status / path / limits |
|---|---|---|---|
| 1000000001 | STAGING Test Almonds | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000002 | STAGING Test Batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000003 | STAGING Pallet Charge | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000004 | STG-A1P2-294CDA1C784E ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000005 | STG-A1P2-294CDA1C784E batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000006 | STG-A1P2-294CDA1C784E finished | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000007 | STG-A1P2-57FB94002B5A ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000008 | STG-A1P2-57FB94002B5A batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000009 | STG-A1P2-57FB94002B5A finished | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000010 | STG-A5-E25438A1C0FC ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000011 | STG-A5-E25438A1C0FC substitute | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000012 | STG-A5-E25438A1C0FC batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000013 | STG-A5-58F9C7C8DB95 ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000014 | STG-A5-58F9C7C8DB95 substitute | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000015 | STG-A5-58F9C7C8DB95 batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000016 | STG-F1-425FD98B13 ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000017 | STG-F1-425FD98B13 batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000018 | STG-F1-425FD98B13 finished | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000019 | STG-F1-85AAC3C9FC ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000020 | STG-F1-85AAC3C9FC batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000021 | STG-F1-85AAC3C9FC finished | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000022 | STG-F1-B7D27BA79C ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000023 | STG-F1-B7D27BA79C batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000024 | STG-F1-B7D27BA79C finished | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000025 | STG-A7-E5A62D939A90 granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000026 | STG-A7-E5A62D939A90 pallet charge | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000027 | STG-A3B-0D78DDA6AF2C ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000028 | STG-A3B-0D78DDA6AF2C batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000029 | STG-F1-EC5678F88B ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000030 | STG-F1-EC5678F88B batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000031 | STG-F1-EC5678F88B finished | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000032 | STG-F1-7FF9E7880F ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000033 | STG-F1-7FF9E7880F batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000034 | STG-F1-7FF9E7880F finished | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000035 | STG-A6-EC3429B9180B granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000036 | STG-A6H-62A8547011 granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000037 | STG-A6H-83911280CA granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000038 | STG-A6H-A52A97B6FF granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000039 | STG-A12-46454C41E879 ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000040 | STG-A12-46454C41E879 batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000041 | STG-A12-46454C41E879 extra_finished | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000042 | STG-A12-46454C41E879 regular_finished | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000043 | STG-A9-573C5E4443C3 oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000044 | STG-A9-AFDC1124A00C oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000045 | STG-F1-AABCF507E9 ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000046 | STG-F1-AABCF507E9 batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000047 | STG-F1-AABCF507E9 finished | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000048 | STG-A9-9901D12E1A33 oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000049 | STG-SIGNIN-11404F4E Almonds | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000050 | STG-SIGNIN-CAEA4CCD Almonds | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000051 | STG-A5P2-F9454E5513 synthetic ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000052 | STG-D12-12D6B802 Oat Bran | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000053 | STG-D12-12D6B802 Rolled Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000054 | STG-D12-12D6B802 Batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000055 | STG-D12-EB83887D Oat Bran | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000056 | STG-D12-EB83887D Rolled Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000057 | STG-D12-EB83887D Batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000058 | STG STG-CNT-A575D9 Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000059 | STG STG-CNT-A575D9 Sugar | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000060 | STG STG-CNT-A575D9 Coconut | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000061 | STG STG-CNT-A575D9 Batch granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000062 | STG STG-CNT-A575D9 Granola 25 LB | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000063 | STG STG-CNT-3A3FEE Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000064 | STG STG-CNT-3A3FEE Sugar | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000065 | STG STG-CNT-3A3FEE Coconut | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000066 | STG STG-CNT-3A3FEE Batch granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000067 | STG STG-CNT-3A3FEE Granola 25 LB | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000068 | STG STG-CNT-C5CDEB Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000069 | STG STG-CNT-C5CDEB Sugar | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000070 | STG STG-CNT-C5CDEB Coconut | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000071 | STG STG-CNT-C5CDEB Batch granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000072 | STG STG-CNT-C5CDEB Granola 25 LB | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000073 | STG STG-CNT-EA2A30 Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000074 | STG STG-CNT-EA2A30 Sugar | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000075 | STG STG-CNT-EA2A30 Coconut | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000076 | STG STG-CNT-EA2A30 Batch granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000077 | STG STG-CNT-EA2A30 Granola 25 LB | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000078 | STG STG-CNT-9F7F22 Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000079 | STG STG-CNT-9F7F22 Sugar | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000080 | STG STG-CNT-9F7F22 Coconut | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000081 | STG STG-CNT-9F7F22 Batch granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000082 | STG STG-CNT-9F7F22 Granola 25 LB | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000083 | STG STG-CNT-9F7F22 Ingredient 1 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000084 | STG STG-CNT-9F7F22 Ingredient 2 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000085 | STG STG-CNT-9F7F22 Ingredient 3 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000086 | STG STG-CNT-9F7F22 Ingredient 4 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000087 | STG STG-CNT-9F7F22 Ingredient 5 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000088 | STG STG-CNT-9F7F22 Ingredient 6 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000089 | STG-XR-CCAF19C68F ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000090 | STG-XR-CCAF19C68F batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000091 | STG-XR-CCAF19C68F case | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000092 | STG-D12-7C09159C Oat Bran | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000093 | STG-D12-7C09159C Rolled Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000094 | STG-D12-7C09159C Batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000095 | STG-D12-F0DF2E28 Oat Bran | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000096 | STG-D12-F0DF2E28 Rolled Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000097 | STG-D12-F0DF2E28 Batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000098 | STG-D12-56581DBB Oat Bran | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000099 | STG-D12-56581DBB Rolled Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000100 | STG-D12-56581DBB Batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000101 | STG-D12-815187CF Oat Bran | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000102 | STG-D12-815187CF Rolled Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000103 | STG-D12-815187CF Batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000116 | STG STG-CNT-625A72 Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000117 | STG STG-CNT-625A72 Sugar | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000118 | STG STG-CNT-625A72 Coconut | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000119 | STG STG-CNT-625A72 Batch granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000120 | STG STG-CNT-625A72 Granola 25 LB | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000121 | STG STG-CNT-625A72 Granola 15 LB case | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000122 | STG STG-CNT-625A72 Ingredient 1 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000123 | STG STG-CNT-625A72 Ingredient 2 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000124 | STG STG-CNT-625A72 Ingredient 3 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000125 | STG STG-CNT-625A72 Ingredient 4 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000126 | STG STG-CNT-625A72 Ingredient 5 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000127 | STG STG-CNT-625A72 Ingredient 6 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000128 | STG STG-CNT-625A72 Extra-kosher batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000129 | STG STG-CNT-A13CF3 Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000130 | STG STG-CNT-A13CF3 Sugar | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000131 | STG STG-CNT-A13CF3 Coconut | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000132 | STG STG-CNT-A13CF3 Batch granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000133 | STG STG-CNT-A13CF3 Granola 25 LB | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000134 | STG STG-CNT-A13CF3 Granola 15 LB case | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000135 | STG STG-CNT-A13CF3 Ingredient 1 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000136 | STG STG-CNT-A13CF3 Ingredient 2 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000137 | STG STG-CNT-A13CF3 Ingredient 3 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000138 | STG STG-CNT-A13CF3 Ingredient 4 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000139 | STG STG-CNT-A13CF3 Ingredient 5 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000140 | STG STG-CNT-A13CF3 Ingredient 6 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000141 | STG STG-CNT-A13CF3 Extra-kosher batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000142 | STG-D2OFF-77262046 Cases | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000143 | STG-D2OFF-77262046 Pallet Charge | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000144 | STG-D2OFF-77262046 Ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000145 | STG STG-CNT-785330 Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000146 | STG STG-CNT-785330 Sugar | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000147 | STG STG-CNT-785330 Coconut | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000148 | STG STG-CNT-785330 Batch granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000149 | STG STG-CNT-785330 Granola 25 LB | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000150 | STG STG-CNT-785330 Granola 15 LB case | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000151 | STG STG-CNT-785330 Ingredient 1 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000152 | STG STG-CNT-785330 Ingredient 2 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000153 | STG STG-CNT-785330 Ingredient 3 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000154 | STG STG-CNT-785330 Ingredient 4 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000155 | STG STG-CNT-785330 Ingredient 5 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000156 | STG STG-CNT-785330 Ingredient 6 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000157 | STG STG-CNT-785330 Extra-kosher batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000158 | STG STG-CNT-947168 Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000159 | STG STG-CNT-947168 Sugar | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000160 | STG STG-CNT-947168 Coconut | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000161 | STG STG-CNT-947168 Batch granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000162 | STG STG-CNT-947168 Granola 25 LB | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000163 | STG STG-CNT-947168 Granola 15 LB case | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000164 | STG STG-CNT-947168 Ingredient 1 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000165 | STG STG-CNT-947168 Ingredient 2 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000166 | STG STG-CNT-947168 Ingredient 3 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000167 | STG STG-CNT-947168 Ingredient 4 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000168 | STG STG-CNT-947168 Ingredient 5 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000169 | STG STG-CNT-947168 Ingredient 6 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000170 | STG STG-CNT-947168 Extra-kosher batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000171 | STG STG-CNT-36A1AD Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000172 | STG STG-CNT-36A1AD Sugar | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000173 | STG STG-CNT-36A1AD Coconut | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000174 | STG STG-CNT-36A1AD Batch granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000175 | STG STG-CNT-36A1AD Granola 25 LB | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000176 | STG STG-CNT-36A1AD Granola 15 LB case | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000177 | STG STG-CNT-36A1AD Ingredient 1 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000178 | STG STG-CNT-36A1AD Ingredient 2 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000179 | STG STG-CNT-36A1AD Ingredient 3 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000180 | STG STG-CNT-36A1AD Ingredient 4 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000181 | STG STG-CNT-36A1AD Ingredient 5 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000182 | STG STG-CNT-36A1AD Ingredient 6 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000183 | STG STG-CNT-36A1AD Extra-kosher batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000184 | STG STG-CNT-DCF5EB Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000185 | STG STG-CNT-DCF5EB Sugar | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000186 | STG STG-CNT-DCF5EB Coconut | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000187 | STG STG-CNT-DCF5EB Batch granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000188 | STG STG-CNT-DCF5EB Granola 25 LB | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000189 | STG STG-CNT-DCF5EB Granola 15 LB case | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000190 | STG STG-CNT-DCF5EB Ingredient 1 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000191 | STG STG-CNT-DCF5EB Ingredient 2 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000192 | STG STG-CNT-DCF5EB Ingredient 3 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000193 | STG STG-CNT-DCF5EB Ingredient 4 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000194 | STG STG-CNT-DCF5EB Ingredient 5 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000195 | STG STG-CNT-DCF5EB Ingredient 6 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000196 | STG STG-CNT-DCF5EB Extra-kosher batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000197 | STG STG-CNT-780841 Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000198 | STG STG-CNT-780841 Sugar | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000199 | STG STG-CNT-780841 Coconut | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000200 | STG STG-CNT-780841 Batch granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000201 | STG STG-CNT-780841 Granola 25 LB | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000202 | STG STG-CNT-780841 Granola 15 LB case | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000203 | STG STG-CNT-780841 Ingredient 1 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000204 | STG STG-CNT-780841 Ingredient 2 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000205 | STG STG-CNT-780841 Ingredient 3 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000206 | STG STG-CNT-780841 Ingredient 4 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000207 | STG STG-CNT-780841 Ingredient 5 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000208 | STG STG-CNT-780841 Ingredient 6 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000209 | STG STG-CNT-780841 Empty-shelf ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000210 | STG STG-CNT-780841 Extra-kosher batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000211 | STG STG-CNT-08370B Oats | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000212 | STG STG-CNT-08370B Sugar | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000213 | STG STG-CNT-08370B Coconut | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000214 | STG STG-CNT-08370B Batch granola | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000215 | STG STG-CNT-08370B Granola 25 LB | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000216 | STG STG-CNT-08370B Granola 15 LB case | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000217 | STG STG-CNT-08370B Ingredient 1 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000218 | STG STG-CNT-08370B Ingredient 2 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000219 | STG STG-CNT-08370B Ingredient 3 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000220 | STG STG-CNT-08370B Ingredient 4 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000221 | STG STG-CNT-08370B Ingredient 5 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000222 | STG STG-CNT-08370B Ingredient 6 | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000223 | STG STG-CNT-08370B Empty-shelf ingredient | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000224 | STG STG-CNT-08370B Extra-kosher batch | True → False | blocked; ; Direct API exists (/admin/products/{product_id}) but has no supported prepare → commit / receipt path for deactivation. |

## Customers

| ID | Item | Before → desired after | Status / path / limits |
|---|---|---|---|
| 1000000001 | STAGING Test Customer | True → False | blocked; ; Direct API exists (/customers/{customer_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000002 | STG-A7-E5A62D939A90 customer | True → False | blocked; ; Direct API exists (/customers/{customer_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000003 | STG-A6-EC3429B9180B customer | True → False | blocked; ; Direct API exists (/customers/{customer_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000004 | STG-A6H-62A8547011 customer | True → False | blocked; ; Direct API exists (/customers/{customer_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000005 | STG-A6H-83911280CA customer | True → False | blocked; ; Direct API exists (/customers/{customer_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000006 | STG-A6H-A52A97B6FF customer | True → False | blocked; ; Direct API exists (/customers/{customer_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000007 | STG-F1-AABCF507E9 customer | True → False | blocked; ; Direct API exists (/customers/{customer_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000008 | STG-PILOT-261014-D60C7286 TEST CUSTOMER | True → False | blocked; ; Direct API exists (/customers/{customer_id}) but has no supported prepare → commit / receipt path for deactivation. |
| 1000000009 | STG-D2OFF-77262046 customer | True → False | blocked; ; Direct API exists (/customers/{customer_id}) but has no supported prepare → commit / receipt path for deactivation. |

## Orders

| ID | Item | Before → desired after | Status / path / limits |
|---|---|---|---|
| 1000000001 | SO-STAGING-TEST | {"fulfillment":"unshipped","state":"open","status":"confirmed"} → {"fulfillment":"unshipped","state":"cancelled","status":"cancelled"} | planned; /sales/orders/1000000001/cancel/prepare;  |
| 1000000002 | SO-261007-001 | {"fulfillment":"unshipped","state":"open","status":"confirmed"} → {"fulfillment":"unshipped","state":"cancelled","status":"cancelled"} | planned; /sales/orders/1000000002/cancel/prepare;  |
| 1000000004 | SO-261009-001 | {"fulfillment":"unshipped","state":"open","status":"confirmed"} → {"fulfillment":"unshipped","state":"cancelled","status":"cancelled"} | planned; /sales/orders/1000000004/cancel/prepare;  |
| 1000000009 | SO-261012-002 | {"fulfillment":"unshipped","state":"open","status":"confirmed"} → {"fulfillment":"unshipped","state":"cancelled","status":"cancelled"} | planned; /sales/orders/1000000009/cancel/prepare;  |
| 1000000017 | SO-261012-006 | {"fulfillment":"unshipped","state":"open","status":"confirmed"} → {"fulfillment":"unshipped","state":"cancelled","status":"cancelled"} | planned; /sales/orders/1000000017/cancel/prepare;  |
| 1000000025 | SO-261014-003 | {"fulfillment":"unshipped","state":"open","status":"confirmed"} → {"fulfillment":"unshipped","state":"cancelled","status":"cancelled"} | planned; /sales/orders/1000000025/cancel/prepare;  |

## Shortages

| ID | Item | Before → desired after | Status / path / limits |
|---|---|---|---|
| 7 | STG-F1-7FF9E7880F-LOT; flags [2] | {"kind":"SHORTAGE","status":"escalated"} → {"resolution_kind":"waived","status":"waived"} | blocked; /exceptions/7/resolve/prepare; No supported waiver; allowed resolutions: counted, missing_movement, voided. |
| 8 | B26-1009-001; flags [3] | {"kind":"SHORTAGE","status":"escalated"} → {"resolution_kind":"waived","status":"waived"} | blocked; /exceptions/8/resolve/prepare; No supported waiver; allowed resolutions: counted, missing_movement, voided. |
| 17 | STG-F1-AABCF507E9-LOT; flags [4] | {"kind":"SHORTAGE","status":"open"} → {"resolution_kind":"waived","status":"waived"} | blocked; /exceptions/17/resolve/prepare; No supported waiver; allowed resolutions: counted, missing_movement, voided. |
| 18 | B26-1013-002; flags [5] | {"kind":"SHORTAGE","status":"open"} → {"resolution_kind":"waived","status":"waived"} | blocked; /exceptions/18/resolve/prepare; No supported waiver; allowed resolutions: counted, missing_movement, voided. |
| 23 | STG-CNT-A575D9-COCONUT-1; flags [6] | {"kind":"SHORTAGE","status":"open"} → {"resolution_kind":"waived","status":"waived"} | blocked; /exceptions/23/resolve/prepare; No supported waiver; allowed resolutions: counted, missing_movement, voided. |
| 24 | STG-CNT-3A3FEE-COCONUT-1; flags [7] | {"kind":"SHORTAGE","status":"open"} → {"resolution_kind":"waived","status":"waived"} | blocked; /exceptions/24/resolve/prepare; No supported waiver; allowed resolutions: counted, missing_movement, voided. |

## Exceptions

| ID | Item | Before → desired after | Status / path / limits |
|---|---|---|---|
| 1 | 26-10-08-DUTB-003 | {"kind":"UNIDENTIFIED_LOT","status":"open"} → {"resolution_kind":"written_off","status":"waived"} | planned; /exceptions/1/resolve/prepare;  |
| 2 | 26-10-08-DUTB-006 | {"kind":"UNIDENTIFIED_LOT","status":"open"} → {"resolution_kind":"written_off","status":"waived"} | planned; /exceptions/2/resolve/prepare;  |
| 3 | STG-F1-B7D27BA79C-FOUND | {"kind":"UNIDENTIFIED_LOT","status":"open"} → {"resolution_kind":"written_off","status":"waived"} | planned; /exceptions/3/resolve/prepare;  |
| 11 | BOL_PHOTO_MISSING | {"kind":"BOL_PHOTO_MISSING","status":"escalated"} → {"resolution_kind":"waived","status":"waived"} | planned; /exceptions/11/resolve/prepare;  |
| 14 | STG-A9-573C5E4443C3-LOT | {"kind":"UNIDENTIFIED_LOT","status":"open"} → {"resolution_kind":"written_off","status":"waived"} | planned; /exceptions/14/resolve/prepare;  |
| 15 | STG-A9-AFDC1124A00C-LOT | {"kind":"UNIDENTIFIED_LOT","status":"open"} → {"resolution_kind":"written_off","status":"waived"} | planned; /exceptions/15/resolve/prepare;  |
| 16 | STG-A9-9901D12E1A33-LOT | {"kind":"UNIDENTIFIED_LOT","status":"open"} → {"resolution_kind":"written_off","status":"waived"} | planned; /exceptions/16/resolve/prepare;  |
| 55 | SHIFT_DISCREPANCY | {"kind":"SHIFT_DISCREPANCY","status":"open"} → {"resolution_kind":"waived","status":"waived"} | planned; /exceptions/55/resolve/prepare;  |

## Exact supported prepare requests (never sent during dry run)

```jsonl
{"body":{"client_source":"api","delta_lb":-25.0,"lot_id":1000000003,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000003,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-0.01,"lot_id":1000000004,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000004,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-0.09,"lot_id":1000000005,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000005,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-0.09,"lot_id":1000000009,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000009,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-90.0,"lot_id":1000000023,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000023,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-88.0,"lot_id":1000000026,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000026,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-2.0,"lot_id":1000000029,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000029,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-88.0,"lot_id":1000000030,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000030,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-2.0,"lot_id":1000000033,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000033,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":606.0,"lot_id":1000000034,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000034,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000036,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000036,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-2.0,"lot_id":1000000040,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000040,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-50.0,"lot_id":1000000041,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000041,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000042,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000042,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-75.0,"lot_id":1000000043,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000043,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-50.0,"lot_id":1000000044,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000044,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-80.0,"lot_id":1000000045,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000045,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-97.0,"lot_id":1000000049,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000049,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-97.0,"lot_id":1000000050,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000050,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-97.0,"lot_id":1000000052,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000052,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-10.0,"lot_id":1000000055,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000055,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-10.0,"lot_id":1000000062,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000062,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-7.5,"lot_id":1000000063,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000063,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.4,"lot_id":1000000064,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000064,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.8,"lot_id":1000000065,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000065,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-75.0,"lot_id":1000000066,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000066,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-10.0,"lot_id":1000000067,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000067,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.6,"lot_id":1000000068,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000068,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-20.0,"lot_id":1000000069,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000069,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-328.3,"lot_id":1000000070,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000070,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-348.0,"lot_id":1000000071,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000071,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-21.0,"lot_id":1000000074,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000074,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-21.0,"lot_id":1000000075,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000075,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-21.0,"lot_id":1000000076,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000076,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-21.0,"lot_id":1000000077,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000077,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-60.0,"lot_id":1000000078,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000078,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-7.5,"lot_id":1000000079,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000079,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-5.0,"lot_id":1000000080,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000080,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-5.0,"lot_id":1000000081,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000081,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-75.0,"lot_id":1000000082,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000082,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-10.0,"lot_id":1000000083,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000083,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-5.0,"lot_id":1000000084,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000084,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-20.0,"lot_id":1000000085,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000085,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-328.3,"lot_id":1000000086,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000086,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-371.1,"lot_id":1000000087,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000087,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-30.0,"lot_id":1000000088,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000088,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-30.0,"lot_id":1000000089,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000089,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-30.0,"lot_id":1000000090,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000090,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-30.0,"lot_id":1000000091,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000091,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-60.0,"lot_id":1000000092,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000092,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-20.0,"lot_id":1000000093,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000093,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-5.0,"lot_id":1000000094,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000094,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-5.0,"lot_id":1000000095,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000095,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-300.0,"lot_id":1000000096,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000096,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-40.0,"lot_id":1000000097,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000097,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-5.0,"lot_id":1000000098,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000098,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-70.0,"lot_id":1000000099,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000099,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-328.3,"lot_id":1000000100,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000100,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-348.0,"lot_id":1000000101,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000101,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-4.0,"lot_id":1000000102,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000102,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-400.0,"lot_id":1000000108,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000108,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1200.0,"lot_id":1000000109,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000109,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1000.0,"lot_id":1000000111,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000111,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-395.0,"lot_id":1000000113,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000113,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-520.0,"lot_id":1000000114,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000114,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1000.0,"lot_id":1000000116,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000116,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-395.0,"lot_id":1000000118,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000118,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-520.0,"lot_id":1000000119,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000119,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-25.0,"lot_id":1000000120,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000120,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-975.0,"lot_id":1000000121,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000121,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-180.0,"lot_id":1000000123,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000123,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-395.0,"lot_id":1000000124,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000124,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-520.0,"lot_id":1000000125,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000125,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-25.0,"lot_id":1000000126,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000126,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-950.0,"lot_id":1000000127,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000127,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-180.0,"lot_id":1000000129,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000129,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-395.0,"lot_id":1000000130,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000130,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-520.0,"lot_id":1000000131,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000131,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-25.0,"lot_id":1000000132,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000132,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-950.0,"lot_id":1000000133,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000133,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-99.0,"lot_id":1000000134,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000134,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-98.0,"lot_id":1000000135,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000135,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-97.0,"lot_id":1000000136,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000136,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-96.0,"lot_id":1000000137,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000137,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-95.0,"lot_id":1000000138,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000138,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-94.0,"lot_id":1000000139,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000139,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-180.0,"lot_id":1000000141,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000141,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-4.0,"lot_id":1000000142,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000142,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1600.0,"lot_id":1000000144,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000144,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-395.0,"lot_id":1000000164,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000164,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-520.0,"lot_id":1000000165,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000165,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-25.0,"lot_id":1000000166,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000166,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-950.0,"lot_id":1000000167,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000167,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-255.0,"lot_id":1000000168,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000168,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-99.0,"lot_id":1000000169,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000169,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-98.0,"lot_id":1000000170,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000170,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-97.0,"lot_id":1000000171,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000171,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-96.0,"lot_id":1000000172,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000172,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-95.0,"lot_id":1000000173,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000173,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-94.0,"lot_id":1000000174,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000174,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-300.0,"lot_id":1000000176,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000176,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-601.0,"lot_id":1000000177,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000177,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-3.3,"lot_id":1000000183,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000183,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-250.0,"lot_id":1000000184,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000184,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-298.0,"lot_id":1000000185,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000185,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-25.0,"lot_id":1000000186,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000186,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-328.3,"lot_id":1000000187,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000187,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-395.0,"lot_id":1000000188,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000188,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-520.0,"lot_id":1000000189,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000189,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-25.0,"lot_id":1000000190,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000190,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1000.0,"lot_id":1000000191,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000191,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-255.0,"lot_id":1000000192,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000192,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-99.0,"lot_id":1000000193,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000193,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-98.0,"lot_id":1000000194,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000194,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-97.0,"lot_id":1000000195,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000195,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-96.0,"lot_id":1000000196,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000196,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-95.0,"lot_id":1000000197,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000197,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-94.0,"lot_id":1000000198,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000198,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-180.0,"lot_id":1000000200,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000200,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000413,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000413,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000414,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000414,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000415,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000415,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000416,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000416,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000417,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000417,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000418,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000418,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000419,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000419,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000420,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000420,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000421,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000421,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000422,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000422,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000423,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000423,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000424,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000424,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000425,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000425,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000426,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000426,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000427,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000427,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000428,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000428,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000429,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000429,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000430,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000430,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000431,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000431,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000432,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000432,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000433,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000433,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000434,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000434,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000435,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000435,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-350.0,"lot_id":1000000436,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000436,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000443,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000443,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000444,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000444,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000445,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000445,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000446,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000446,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000447,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000447,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000448,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000448,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000449,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000449,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000450,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000450,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000451,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000451,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000452,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000452,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000453,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000453,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000454,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000454,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000455,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000455,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000456,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000456,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000457,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000457,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000458,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000458,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000459,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000459,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000460,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000460,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000461,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000461,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000462,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000462,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000463,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000463,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000464,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000464,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-1.0,"lot_id":1000000465,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000465,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-350.0,"lot_id":1000000466,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000466,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-396.0,"lot_id":1000000467,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000467,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-520.0,"lot_id":1000000468,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000468,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-10.0,"lot_id":1000000469,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000469,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-975.0,"lot_id":1000000470,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000470,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-300.0,"lot_id":1000000471,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000471,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000472,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000472,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000473,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000473,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000474,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000474,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000475,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000475,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000476,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000476,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000477,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000477,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-300.0,"lot_id":1000000478,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000478,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-396.0,"lot_id":1000000479,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000479,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-520.0,"lot_id":1000000480,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000480,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-10.0,"lot_id":1000000481,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000481,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-975.0,"lot_id":1000000482,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000482,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-300.0,"lot_id":1000000483,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000483,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000484,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000484,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000485,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000485,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000486,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000486,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000487,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000487,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000488,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000488,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000489,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000489,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-300.0,"lot_id":1000000490,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000490,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-396.0,"lot_id":1000000491,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000491,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-520.0,"lot_id":1000000492,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000492,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-10.0,"lot_id":1000000493,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000493,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-975.0,"lot_id":1000000494,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000494,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-300.0,"lot_id":1000000495,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000495,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000496,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000496,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000497,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000497,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000498,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000498,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000499,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000499,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000500,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000500,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000501,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000501,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-300.0,"lot_id":1000000502,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000502,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-396.0,"lot_id":1000000503,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000503,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-520.0,"lot_id":1000000504,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000504,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-10.0,"lot_id":1000000505,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000505,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-977.0,"lot_id":1000000506,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000506,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-300.0,"lot_id":1000000507,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000507,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000508,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000508,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000509,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000509,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000510,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000510,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000511,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000511,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000512,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000512,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000513,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000513,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-300.0,"lot_id":1000000514,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000514,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-396.0,"lot_id":1000000515,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000515,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-520.0,"lot_id":1000000516,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000516,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-10.0,"lot_id":1000000517,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000517,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-977.0,"lot_id":1000000518,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000518,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-300.0,"lot_id":1000000519,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000519,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000520,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000520,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000521,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000521,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000522,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000522,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000523,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000523,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000524,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000524,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000525,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000525,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-300.0,"lot_id":1000000526,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000526,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-396.0,"lot_id":1000000527,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000527,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-520.0,"lot_id":1000000528,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000528,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-10.0,"lot_id":1000000529,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000529,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-977.0,"lot_id":1000000530,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000530,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-300.0,"lot_id":1000000531,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000531,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000532,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000532,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000533,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000533,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000534,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000534,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000535,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000535,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000536,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000536,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-100.0,"lot_id":1000000537,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000537,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","delta_lb":-300.0,"lot_id":1000000538,"note":"STAGING PILOT CLEANUP","reason_code":"physical_count"},"category":"lots","id":1000000538,"method":"POST","path":"/adjust/prepare"}
{"body":{"client_source":"api","note":"STAGING PILOT CLEANUP","reason":"other"},"category":"orders","id":1000000001,"method":"POST","path":"/sales/orders/1000000001/cancel/prepare"}
{"body":{"client_source":"api","note":"STAGING PILOT CLEANUP","reason":"other"},"category":"orders","id":1000000002,"method":"POST","path":"/sales/orders/1000000002/cancel/prepare"}
{"body":{"client_source":"api","note":"STAGING PILOT CLEANUP","reason":"other"},"category":"orders","id":1000000004,"method":"POST","path":"/sales/orders/1000000004/cancel/prepare"}
{"body":{"client_source":"api","note":"STAGING PILOT CLEANUP","reason":"other"},"category":"orders","id":1000000009,"method":"POST","path":"/sales/orders/1000000009/cancel/prepare"}
{"body":{"client_source":"api","note":"STAGING PILOT CLEANUP","reason":"other"},"category":"orders","id":1000000017,"method":"POST","path":"/sales/orders/1000000017/cancel/prepare"}
{"body":{"client_source":"api","note":"STAGING PILOT CLEANUP","reason":"other"},"category":"orders","id":1000000025,"method":"POST","path":"/sales/orders/1000000025/cancel/prepare"}
{"body":{"client_source":"api","note":"STAGING PILOT CLEANUP","resolution_kind":"written_off"},"category":"exceptions","id":1,"method":"POST","path":"/exceptions/1/resolve/prepare"}
{"body":{"client_source":"api","note":"STAGING PILOT CLEANUP","resolution_kind":"written_off"},"category":"exceptions","id":2,"method":"POST","path":"/exceptions/2/resolve/prepare"}
{"body":{"client_source":"api","note":"STAGING PILOT CLEANUP","resolution_kind":"written_off"},"category":"exceptions","id":3,"method":"POST","path":"/exceptions/3/resolve/prepare"}
{"body":{"client_source":"api","note":"STAGING PILOT CLEANUP","resolution_kind":"waived"},"category":"exceptions","id":11,"method":"POST","path":"/exceptions/11/resolve/prepare"}
{"body":{"client_source":"api","note":"STAGING PILOT CLEANUP","resolution_kind":"written_off"},"category":"exceptions","id":14,"method":"POST","path":"/exceptions/14/resolve/prepare"}
{"body":{"client_source":"api","note":"STAGING PILOT CLEANUP","resolution_kind":"written_off"},"category":"exceptions","id":15,"method":"POST","path":"/exceptions/15/resolve/prepare"}
{"body":{"client_source":"api","note":"STAGING PILOT CLEANUP","resolution_kind":"written_off"},"category":"exceptions","id":16,"method":"POST","path":"/exceptions/16/resolve/prepare"}
{"body":{"client_source":"api","note":"STAGING PILOT CLEANUP","resolution_kind":"waived"},"category":"exceptions","id":55,"method":"POST","path":"/exceptions/55/resolve/prepare"}
```

## Protected OPEN261014 lots — unchanged

| Lot ID | Code | Product | Before → after lb |
|---|---|---|---|
| 1000000201 | OPEN261014-P857 | 2 | 14.0000 → 14.0000 |
| 1000000202 | OPEN261014-P1097 | 2 | 325.0000 → 325.0000 |
| 1000000203 | OPEN261014-P1582 | 2 | 325.0000 → 325.0000 |
| 1000000204 | OPEN261014-P1477 | 3 | 1255.0000 → 1255.0000 |
| 1000000205 | OPEN261014-P1096 | 4 | 53.0000 → 53.0000 |
| 1000000206 | OPEN261014-P1581 | 4 | 400.0000 → 400.0000 |
| 1000000207 | OPEN261014-P1055 | 121 | 1050.0000 → 1050.0000 |
| 1000000208 | OPEN261014-P1450 | 108 | 2000.0000 → 2000.0000 |
| 1000000209 | OPEN261014-P1485 | 108 | 1044.0000 → 1044.0000 |
| 1000000210 | OPEN261014-P1498 | 108 | 2784.0000 → 2784.0000 |
| 1000000211 | OPEN261014-P1593 | 108 | 1392.0000 → 1392.0000 |
| 1000000212 | OPEN261014-P1512 | 107 | 1051.0000 → 1051.0000 |
| 1000000213 | OPEN261014-P1521 | 107 | 2368.0000 → 2368.0000 |
| 1000000214 | OPEN261014-P1547 | 107 | 4891.7000 → 4891.7000 |
| 1000000215 | OPEN261014-P1586 | 107 | 5168.0000 → 5168.0000 |
| 1000000216 | OPEN261014-P1592 | 107 | 3876.0000 → 3876.0000 |
| 1000000217 | OPEN261014-P1478 | 126 | 160.0000 → 160.0000 |
| 1000000218 | OPEN261014-P1542 | 114 | 1354.5000 → 1354.5000 |
| 1000000219 | OPEN261014-P1573 | 116 | 2097.5000 → 2097.5000 |
| 1000000220 | OPEN261014-P1435 | 112 | 240.0000 → 240.0000 |
| 1000000221 | OPEN261014-P1463 | 112 | 115.0000 → 115.0000 |
| 1000000222 | OPEN261014-P9 | 54 | 1963.6000 → 1963.6000 |
| 1000000223 | OPEN261014-P338 | 55 | 4407.8500 → 4407.8500 |
| 1000000224 | OPEN261014-P11 | 56 | 578.8100 → 578.8100 |
| 1000000225 | OPEN261014-P571 | 58 | 718.4000 → 718.4000 |
| 1000000226 | OPEN261014-P14 | 60 | 1324.2000 → 1324.2000 |
| 1000000227 | OPEN261014-P58 | 61 | 891.6000 → 891.6000 |
| 1000000228 | OPEN261014-P59 | 61 | 508.0000 → 508.0000 |
| 1000000229 | OPEN261014-P92 | 61 | 2200.0000 → 2200.0000 |
| 1000000230 | OPEN261014-P360 | 61 | 2200.0000 → 2200.0000 |
| 1000000231 | OPEN261014-P570 | 61 | 2200.0000 → 2200.0000 |
| 1000000232 | OPEN261014-P909 | 61 | 1800.0000 → 1800.0000 |
| 1000000233 | OPEN261014-P61 | 62 | 512.8000 → 512.8000 |
| 1000000234 | OPEN261014-P715 | 62 | 6160.0000 → 6160.0000 |
| 1000000235 | OPEN261014-P17 | 64 | 1098.8000 → 1098.8000 |
| 1000000236 | OPEN261014-P18 | 65 | 1860.0000 → 1860.0000 |
| 1000000237 | OPEN261014-P19 | 66 | 10000.0000 → 10000.0000 |
| 1000000238 | OPEN261014-P20 | 67 | 1932.6000 → 1932.6000 |
| 1000000239 | OPEN261014-P143 | 67 | 8000.0000 → 8000.0000 |
| 1000000240 | OPEN261014-P206 | 67 | 32.0000 → 32.0000 |
| 1000000241 | OPEN261014-P256 | 68 | 2737.8000 → 2737.8000 |
| 1000000242 | OPEN261014-P22 | 69 | 65.8200 → 65.8200 |
| 1000000243 | OPEN261014-P23 | 70 | 387.8000 → 387.8000 |
| 1000000244 | OPEN261014-P1421 | 72 | 1340.0000 → 1340.0000 |
| 1000000245 | OPEN261014-P1445 | 72 | 500.0000 → 500.0000 |
| 1000000246 | OPEN261014-P1553 | 72 | 500.0000 → 500.0000 |
| 1000000247 | OPEN261014-P531 | 6 | 19.0000 → 19.0000 |
| 1000000248 | OPEN261014-P1111 | 6 | 250.0000 → 250.0000 |
| 1000000249 | OPEN261014-P97 | 7 | 105.0000 → 105.0000 |
| 1000000250 | OPEN261014-P469 | 7 | 50.0000 → 50.0000 |
| 1000000251 | OPEN261014-P94 | 8 | 94.0000 → 94.0000 |
| 1000000252 | OPEN261014-P95 | 8 | 50.0000 → 50.0000 |
| 1000000253 | OPEN261014-P471 | 8 | 100.0000 → 100.0000 |
| 1000000254 | OPEN261014-P605 | 9 | 12000.0000 → 12000.0000 |
| 1000000255 | OPEN261014-P606 | 9 | 4125.0000 → 4125.0000 |
| 1000000256 | OPEN261014-P607 | 9 | 2850.0000 → 2850.0000 |
| 1000000257 | OPEN261014-P608 | 9 | 550.0000 → 550.0000 |
| 1000000258 | OPEN261014-P609 | 9 | 275.0000 → 275.0000 |
| 1000000259 | OPEN261014-P611 | 10 | 640.0000 → 640.0000 |
| 1000000260 | OPEN261014-P635 | 10 | 10000.0000 → 10000.0000 |
| 1000000261 | OPEN261014-P624 | 12 | 14500.0000 → 14500.0000 |
| 1000000262 | OPEN261014-P616 | 13 | 749.0000 → 749.0000 |
| 1000000263 | OPEN261014-P617 | 14 | 4800.0000 → 4800.0000 |
| 1000000264 | OPEN261014-P618 | 14 | 2200.0000 → 2200.0000 |
| 1000000265 | OPEN261014-P619 | 14 | 9850.0000 → 9850.0000 |
| 1000000266 | OPEN261014-P620 | 14 | 16375.0000 → 16375.0000 |
| 1000000267 | OPEN261014-P621 | 14 | 1725.0000 → 1725.0000 |
| 1000000268 | OPEN261014-P622 | 14 | 13200.0000 → 13200.0000 |
| 1000000269 | OPEN261014-P1527 | 161 | 1250.0000 → 1250.0000 |
| 1000000270 | OPEN261014-P1537 | 161 | 1090.0000 → 1090.0000 |
| 1000000271 | OPEN261014-P1466 | 156 | 400.0000 → 400.0000 |
| 1000000272 | OPEN261014-P1539 | 156 | 370.0000 → 370.0000 |
| 1000000273 | OPEN261014-P1550 | 156 | 720.0000 → 720.0000 |
| 1000000274 | OPEN261014-P1562 | 156 | 1460.0000 → 1460.0000 |
| 1000000275 | OPEN261014-P1459 | 157 | 475.0000 → 475.0000 |
| 1000000276 | OPEN261014-P1468 | 157 | 1100.0000 → 1100.0000 |
| 1000000277 | OPEN261014-P1519 | 157 | 1100.0000 → 1100.0000 |
| 1000000278 | OPEN261014-P1540 | 162 | 1450.0000 → 1450.0000 |
| 1000000279 | OPEN261014-P1549 | 162 | 2180.0000 → 2180.0000 |
| 1000000280 | OPEN261014-P1308 | 158 | 160.0000 → 160.0000 |
| 1000000281 | OPEN261014-P1528 | 163 | 1280.0000 → 1280.0000 |
| 1000000282 | OPEN261014-P1538 | 163 | 1090.0000 → 1090.0000 |
| 1000000283 | OPEN261014-P1321 | 159 | 310.0000 → 310.0000 |
| 1000000284 | OPEN261014-P1587 | 160 | 1375.0000 → 1375.0000 |
| 1000000285 | OPEN261014-P1594 | 160 | 1000.0000 → 1000.0000 |
| 1000000286 | OPEN261014-P982 | 16 | 810.0000 → 810.0000 |
| 1000000287 | OPEN261014-P1406 | 164 | 2800.0000 → 2800.0000 |
| 1000000288 | OPEN261014-P1418 | 164 | 2800.0000 → 2800.0000 |
| 1000000289 | OPEN261014-P1428 | 164 | 4200.0000 → 4200.0000 |
| 1000000290 | OPEN261014-P1432 | 164 | 4200.0000 → 4200.0000 |
| 1000000291 | OPEN261014-P1481 | 164 | 2800.0000 → 2800.0000 |
| 1000000292 | OPEN261014-P1491 | 164 | 4200.0000 → 4200.0000 |
| 1000000293 | OPEN261014-P1502 | 164 | 4200.0000 → 4200.0000 |
| 1000000294 | OPEN261014-P1506 | 164 | 3740.0000 → 3740.0000 |
| 1000000295 | OPEN261014-P1518 | 164 | 3260.0000 → 3260.0000 |
| 1000000296 | OPEN261014-P1597 | 164 | 520.0000 → 520.0000 |
| 1000000297 | OPEN261014-P1390 | 144 | 1400.0000 → 1400.0000 |
| 1000000298 | OPEN261014-P1515 | 144 | 1400.0000 → 1400.0000 |
| 1000000299 | OPEN261014-P1524 | 144 | 4200.0000 → 4200.0000 |
| 1000000300 | OPEN261014-P1598 | 144 | 2800.0000 → 2800.0000 |
| 1000000301 | OPEN261014-P1289 | 17 | 154.5000 → 154.5000 |
| 1000000302 | OPEN261014-P1447 | 17 | 25.0000 → 25.0000 |
| 1000000303 | OPEN261014-P1584 | 17 | 100.0000 → 100.0000 |
| 1000000304 | OPEN261014-P453 | 190 | 100700.0000 → 100700.0000 |
| 1000000305 | OPEN261014-P1345 | 196 | 11050.0000 → 11050.0000 |
| 1000000306 | OPEN261014-P439 | 18 | 223.0000 → 223.0000 |
| 1000000307 | OPEN261014-P1027 | 18 | 798.2000 → 798.2000 |
| 1000000308 | OPEN261014-P1566 | 18 | 800.0000 → 800.0000 |
| 1000000309 | OPEN261014-P35 | 19 | 8.6600 → 8.6600 |
| 1000000310 | OPEN261014-P36 | 20 | 38.0800 → 38.0800 |
| 1000000311 | OPEN261014-P1029 | 20 | 40.0000 → 40.0000 |
| 1000000312 | OPEN261014-P37 | 21 | 40.9600 → 40.9600 |
| 1000000313 | OPEN261014-P38 | 22 | 80.0800 → 80.0800 |
| 1000000314 | OPEN261014-P39 | 23 | 48.7000 → 48.7000 |
| 1000000315 | OPEN261014-P40 | 25 | 8.6600 → 8.6600 |
| 1000000316 | OPEN261014-P41 | 26 | 552.4000 → 552.4000 |
| 1000000317 | OPEN261014-P1028 | 26 | 80.0000 → 80.0000 |
| 1000000318 | OPEN261014-P43 | 28 | 350.0000 → 350.0000 |
| 1000000319 | OPEN261014-P235 | 28 | 75.0000 → 75.0000 |
| 1000000320 | OPEN261014-P474 | 28 | 150.0000 → 150.0000 |
| 1000000321 | OPEN261014-P858 | 28 | 100.0000 → 100.0000 |
| 1000000322 | OPEN261014-P1098 | 28 | 125.0000 → 125.0000 |
| 1000000323 | OPEN261014-P1583 | 28 | 125.0000 → 125.0000 |
| 1000000324 | OPEN261014-P461 | 29 | 1193.0000 → 1193.0000 |
| 1000000325 | OPEN261014-P1510 | 29 | 7680.0000 → 7680.0000 |
| 1000000326 | OPEN261014-P1565 | 165 | 400.0000 → 400.0000 |
| 1000000327 | OPEN261014-P1591 | 165 | 1540.0000 → 1540.0000 |
| 1000000328 | OPEN261014-P1473 | 75 | 5360.0000 → 5360.0000 |
| 1000000329 | OPEN261014-P1555 | 75 | 8900.0000 → 8900.0000 |
| 1000000330 | OPEN261014-P1476 | 136 | 2500.0000 → 2500.0000 |
| 1000000331 | OPEN261014-P1489 | 136 | 1500.0000 → 1500.0000 |
| 1000000332 | OPEN261014-P1545 | 136 | 4000.0000 → 4000.0000 |
| 1000000333 | OPEN261014-P1552 | 136 | 1250.0000 → 1250.0000 |
| 1000000334 | OPEN261014-P1389 | 137 | 300.0000 → 300.0000 |
| 1000000335 | OPEN261014-P1513 | 137 | 1400.0000 → 1400.0000 |
| 1000000336 | OPEN261014-P1561 | 142 | 550.0000 → 550.0000 |
| 1000000337 | OPEN261014-P1560 | 179 | 219.0400 → 219.0400 |
| 1000000338 | OPEN261014-P1291 | 145 | 2850.0000 → 2850.0000 |
| 1000000339 | OPEN261014-P1298 | 145 | 7012.5000 → 7012.5000 |
| 1000000340 | OPEN261014-P1327 | 145 | 2287.5000 → 2287.5000 |
| 1000000341 | OPEN261014-P1402 | 145 | 7252.5000 → 7252.5000 |
| 1000000342 | OPEN261014-P1440 | 145 | 6217.5000 → 6217.5000 |
| 1000000343 | OPEN261014-P1536 | 145 | 2167.5000 → 2167.5000 |
| 1000000344 | OPEN261014-P1544 | 145 | 2557.5000 → 2557.5000 |
| 1000000345 | OPEN261014-P1563 | 145 | 1597.5000 → 1597.5000 |
| 1000000346 | OPEN261014-P1590 | 145 | 1755.0000 → 1755.0000 |
| 1000000347 | OPEN261014-P1359 | 148 | 1365.0000 → 1365.0000 |
| 1000000348 | OPEN261014-P1322 | 147 | 1852.5000 → 1852.5000 |
| 1000000349 | OPEN261014-P1328 | 147 | 172.5000 → 172.5000 |
| 1000000350 | OPEN261014-P1424 | 146 | 780.0000 → 780.0000 |
| 1000000351 | OPEN261014-P1439 | 146 | 1177.5000 → 1177.5000 |
| 1000000352 | OPEN261014-P1499 | 146 | 6075.0000 → 6075.0000 |
| 1000000353 | OPEN261014-P1570 | 146 | 5482.5000 → 5482.5000 |
| 1000000354 | OPEN261014-P1589 | 146 | 1170.0000 → 1170.0000 |
| 1000000355 | OPEN261014-P1360 | 149 | 2032.5000 → 2032.5000 |
| 1000000356 | OPEN261014-P1525 | 138 | 2000.0000 → 2000.0000 |
| 1000000357 | OPEN261014-P1564 | 30 | 2731.8000 → 2731.8000 |
| 1000000358 | OPEN261014-P1420 | 178 | 50.0000 → 50.0000 |
| 1000000359 | OPEN261014-P725 | 173 | 6375.0000 → 6375.0000 |
| 1000000360 | OPEN261014-P1482 | 173 | 40250.0000 → 40250.0000 |
| 1000000361 | OPEN261014-P1511 | 32 | 13271.2000 → 13271.2000 |
| 1000000362 | OPEN261014-P1585 | 32 | 2000.0000 → 2000.0000 |
| 1000000363 | OPEN261014-P1509 | 33 | 2000.0000 → 2000.0000 |
| 1000000364 | OPEN261014-P1461 | 34 | 3016.8700 → 3016.8700 |
| 1000000365 | OPEN261014-P56 | 52 | 475.0000 → 475.0000 |
| 1000000366 | OPEN261014-P63 | 169 | 50.0000 → 50.0000 |
| 1000000367 | OPEN261014-P335 | 176 | 85.0000 → 85.0000 |
| 1000000368 | OPEN261014-P224 | 102 | 242.0000 → 242.0000 |
| 1000000369 | OPEN261014-P891 | 35 | 131.0000 → 131.0000 |
| 1000000370 | OPEN261014-P66 | 36 | 103.2500 → 103.2500 |
| 1000000371 | OPEN261014-P174 | 36 | 30.0000 → 30.0000 |
| 1000000372 | OPEN261014-P526 | 36 | 30.0000 → 30.0000 |
| 1000000373 | OPEN261014-P1448 | 36 | 30.0000 → 30.0000 |
| 1000000374 | OPEN261014-P1554 | 36 | 30.0000 → 30.0000 |
| 1000000375 | OPEN261014-P890 | 38 | 139.8100 → 139.8100 |
| 1000000376 | OPEN261014-P440 | 39 | 625.0000 → 625.0000 |
| 1000000377 | OPEN261014-P983 | 39 | 2450.0000 → 2450.0000 |
| 1000000378 | OPEN261014-P1125 | 39 | 4898.3000 → 4898.3000 |
| 1000000379 | OPEN261014-P223 | 174 | 10610.0000 → 10610.0000 |
| 1000000380 | OPEN261014-P654 | 174 | 1760.0000 → 1760.0000 |
| 1000000381 | OPEN261014-P1173 | 203 | 4200.0000 → 4200.0000 |
| 1000000382 | OPEN261014-P795 | 175 | 7470.0000 → 7470.0000 |
| 1000000383 | OPEN261014-P1494 | 175 | 28600.0000 → 28600.0000 |
| 1000000384 | OPEN261014-P1113 | 202 | 1725.0000 → 1725.0000 |
| 1000000385 | OPEN261014-P1495 | 202 | 14700.0000 → 14700.0000 |
| 1000000386 | OPEN261014-P1132 | 40 | 12197.5100 → 12197.5100 |
| 1000000387 | OPEN261014-P1530 | 40 | 44000.0000 → 44000.0000 |
| 1000000388 | OPEN261014-P50 | 41 | 1565.0000 → 1565.0000 |
| 1000000389 | OPEN261014-P51 | 42 | 1750.0000 → 1750.0000 |
| 1000000390 | OPEN261014-P52 | 43 | 362.5000 → 362.5000 |
| 1000000391 | OPEN261014-P170 | 43 | 500.0000 → 500.0000 |
| 1000000392 | OPEN261014-P1342 | 44 | 410.0000 → 410.0000 |
| 1000000393 | OPEN261014-P1529 | 44 | 4950.0000 → 4950.0000 |
| 1000000394 | OPEN261014-P1343 | 45 | 458.0000 → 458.0000 |
| 1000000395 | OPEN261014-P941 | 46 | 5110.4000 → 5110.4000 |
| 1000000396 | OPEN261014-P1398 | 46 | 5479.0000 → 5479.0000 |
| 1000000397 | OPEN261014-P5 | 166 | 3975.0000 → 3975.0000 |
| 1000000398 | OPEN261014-P65 | 49 | 103.2500 → 103.2500 |
| 1000000399 | OPEN261014-P172 | 49 | 30.0000 → 30.0000 |
| 1000000400 | OPEN261014-P525 | 49 | 30.0000 → 30.0000 |
| 1000000401 | OPEN261014-P1446 | 49 | 60.0000 → 60.0000 |
| 1000000402 | OPEN261014-P1181 | 291 | 1.0000 → 1.0000 |
| 1000000403 | OPEN261014-TEST-MK107 | 107 | 323.0000 → 323.0000 |
| 1000000404 | OPEN261014-TEST-MK116 | 116 | 350.0000 → 350.0000 |
| 1000000405 | OPEN261014-TEST-MK179 | 179 | 328.3000 → 328.3000 |

## Protected copied missing-supplier-lot exceptions — unchanged

| ID | Kind | Lot | Before → after status |
|---|---|---|---|
| 71 | UNIDENTIFIED_LOT | OPEN261014-P9 | open → open |
| 74 | UNIDENTIFIED_LOT | OPEN261014-P11 | open → open |
| 77 | UNIDENTIFIED_LOT | OPEN261014-P14 | open → open |
| 79 | UNIDENTIFIED_LOT | OPEN261014-P58 | open → open |
| 81 | UNIDENTIFIED_LOT | OPEN261014-P59 | open → open |
| 83 | UNIDENTIFIED_LOT | OPEN261014-P92 | open → open |
| 88 | UNIDENTIFIED_LOT | OPEN261014-P61 | open → open |
| 91 | UNIDENTIFIED_LOT | OPEN261014-P17 | open → open |
| 93 | UNIDENTIFIED_LOT | OPEN261014-P18 | open → open |
| 95 | UNIDENTIFIED_LOT | OPEN261014-P19 | open → open |
| 97 | UNIDENTIFIED_LOT | OPEN261014-P20 | open → open |
| 99 | UNIDENTIFIED_LOT | OPEN261014-P143 | open → open |
| 100 | UNIDENTIFIED_LOT | OPEN261014-P206 | open → open |
| 102 | UNIDENTIFIED_LOT | OPEN261014-P22 | open → open |
| 103 | UNIDENTIFIED_LOT | OPEN261014-P23 | open → open |
| 105 | UNIDENTIFIED_LOT | OPEN261014-P97 | open → open |
| 106 | UNIDENTIFIED_LOT | OPEN261014-P94 | open → open |
| 107 | UNIDENTIFIED_LOT | OPEN261014-P95 | open → open |
| 109 | UNIDENTIFIED_LOT | OPEN261014-P605 | open → open |
| 111 | UNIDENTIFIED_LOT | OPEN261014-P606 | open → open |
| 113 | UNIDENTIFIED_LOT | OPEN261014-P607 | open → open |
| 115 | UNIDENTIFIED_LOT | OPEN261014-P608 | open → open |
| 116 | UNIDENTIFIED_LOT | OPEN261014-P609 | open → open |
| 118 | UNIDENTIFIED_LOT | OPEN261014-P611 | open → open |
| 120 | UNIDENTIFIED_LOT | OPEN261014-P635 | open → open |
| 122 | UNIDENTIFIED_LOT | OPEN261014-P624 | open → open |
| 124 | UNIDENTIFIED_LOT | OPEN261014-P616 | open → open |
| 126 | UNIDENTIFIED_LOT | OPEN261014-P617 | open → open |
| 128 | UNIDENTIFIED_LOT | OPEN261014-P618 | open → open |
| 130 | UNIDENTIFIED_LOT | OPEN261014-P619 | open → open |
| 132 | UNIDENTIFIED_LOT | OPEN261014-P620 | open → open |
| 134 | UNIDENTIFIED_LOT | OPEN261014-P621 | open → open |
| 136 | UNIDENTIFIED_LOT | OPEN261014-P622 | open → open |
| 166 | UNIDENTIFIED_LOT | OPEN261014-P439 | open → open |
| 169 | UNIDENTIFIED_LOT | OPEN261014-P35 | open → open |
| 170 | UNIDENTIFIED_LOT | OPEN261014-P36 | open → open |
| 171 | UNIDENTIFIED_LOT | OPEN261014-P37 | open → open |
| 172 | UNIDENTIFIED_LOT | OPEN261014-P38 | open → open |
| 173 | UNIDENTIFIED_LOT | OPEN261014-P39 | open → open |
| 174 | UNIDENTIFIED_LOT | OPEN261014-P40 | open → open |
| 176 | UNIDENTIFIED_LOT | OPEN261014-P41 | open → open |
| 177 | UNIDENTIFIED_LOT | OPEN261014-P43 | open → open |
| 178 | UNIDENTIFIED_LOT | OPEN261014-P235 | open → open |
| 215 | UNIDENTIFIED_LOT | OPEN261014-P56 | open → open |
| 216 | UNIDENTIFIED_LOT | OPEN261014-P63 | open → open |
| 217 | UNIDENTIFIED_LOT | OPEN261014-P66 | open → open |
| 218 | UNIDENTIFIED_LOT | OPEN261014-P174 | open → open |
| 220 | UNIDENTIFIED_LOT | OPEN261014-P440 | open → open |
| 233 | UNIDENTIFIED_LOT | OPEN261014-P50 | open → open |
| 235 | UNIDENTIFIED_LOT | OPEN261014-P51 | open → open |
| 236 | UNIDENTIFIED_LOT | OPEN261014-P52 | open → open |
| 237 | UNIDENTIFIED_LOT | OPEN261014-P170 | open → open |
| 242 | UNIDENTIFIED_LOT | OPEN261014-P5 | open → open |
| 243 | UNIDENTIFIED_LOT | OPEN261014-P65 | open → open |
| 244 | UNIDENTIFIED_LOT | OPEN261014-P172 | open → open |
| 245 | UNIDENTIFIED_LOT | OPEN261014-P1181 | open → open |

## Other test orders with shipped quantity — unchanged

| ID | Order | Before → after |
|---|---|---|
| 1000000007 | SO-261012-001 | open/shipped → open/shipped |
| 1000000011 | SO-261012-003 | open/shipped → open/shipped |
| 1000000013 | SO-261012-004 | open/shipped → open/shipped |
| 1000000015 | SO-261012-005 | open/shipped → open/shipped |
| 1000000019 | SO-261013-001 | open/shipped → open/shipped |
| 1000000021 | SO-261014-001 | open/shipped → open/shipped |
| 1000000023 | SO-261014-002 | open/shipped → open/shipped |
