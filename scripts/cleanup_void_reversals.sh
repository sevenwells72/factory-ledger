#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
# Historical void-reversal cleanup — Void-semantics migration 2026-06
# ═══════════════════════════════════════════════════════════════════════════
#
# ⚠️  RUN ONLY AGAINST THE NEW CODE (branch fix/void-semantics deployed).
#     Under the OLD code, POST /void/{id} would itself insert ANOTHER posted
#     reversal transaction for each ID below, making the corruption worse.
#     Verify the deploy first: voiding under new code returns
#     "reversal_transaction_id": null.
#
# ⚠️  TIMING: deploy after floor operations end for the day, then run this
#     script IMMEDIATELY after Railway picks up the deploy. Between deploy
#     and this script, posted-only balances expose the historical reversals
#     (lot 610 reads -3,400 lb and lot 617 reads -1,000 lb), which can block
#     production transactions.
#
# What it does: marks the 12 historical reversal transactions voided via the
# app's own POST /void/{id} (originals are already voided), then verifies all
# 16 affected lots against the expected end-state balances. Safe to re-run:
# an already-voided ID returns HTTP 400, which the loop treats as fatal by
# design — review and rerun with the remaining IDs if a partial run occurred.
#
# Usage:
#   FACTORY_API_KEY=... ./scripts/cleanup_void_reversals.sh
#   (optional) FACTORY_API_BASE=https://... to override the API base URL.
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

API_BASE="${FACTORY_API_BASE:-https://fastapi-production-b73a.up.railway.app}"
API_KEY="${FACTORY_API_KEY:?Set FACTORY_API_KEY in the environment — never hardcode it}"
REASON="Void-semantics migration 2026-06: marking historical reversal transaction voided; original already voided"

# The 12 posted reversal transactions (notes = 'Reversal of transaction #N'),
# verified 2026-06-09 against prod with line-by-line negation of their
# voided originals (#469, #863, #911, #913, #924, #914, #925, #915, #926,
# #916, #927, #918 respectively).
REVERSAL_IDS=(470 864 932 933 934 935 936 937 938 939 940 941)

echo "── Step 1: void the 12 historical reversal transactions ──"
for id in "${REVERSAL_IDS[@]}"; do
  echo -n "Voiding reversal transaction #${id} ... "
  response=$(curl -sS -X POST "${API_BASE}/void/${id}" \
    -H "X-API-Key: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d "{\"reason\": \"${REASON}\"}")
  if echo "${response}" | grep -q '"success"[[:space:]]*:[[:space:]]*true'; then
    echo "OK"
  else
    echo "FAILED"
    echo "Response: ${response}"
    echo "STOPPING — fix the failure, then re-run with the remaining IDs."
    exit 1
  fi
  if echo "${response}" | grep -q '"reversal_transaction_id"[[:space:]]*:[[:space:]]*[0-9]'; then
    echo "FATAL: the API posted a NEW reversal — the OLD code is still deployed!"
    echo "Response: ${response}"
    exit 1
  fi
done

echo ""
echo "── Step 2: verify posted-only balances against expected end state ──"
# lot_id:expected_lb — from the 2026-06-09 read-only simulation (these equal
# the pre-migration app-visible/unfiltered balances; nothing should shift).
EXPECTED="
610:6600
612:0
613:36150
614:11500
615:10000
617:9000
582:0
293:0
314:0
33:0
34:801.8
337:0
48:1372.12
500:0
501:0
566:0
"

fail=0
for pair in ${EXPECTED}; do
  lot_id="${pair%%:*}"
  expected="${pair##*:}"
  actual=$(curl -sS "${API_BASE}/lots/${lot_id}" -H "X-API-Key: ${API_KEY}" \
    | sed -n 's/.*"quantity_on_hand"[[:space:]]*:[[:space:]]*\(-\{0,1\}[0-9.]*\).*/\1/p')
  if [ -z "${actual}" ]; then
    echo "lot ${lot_id}: ERROR — could not read quantity_on_hand"
    fail=1
    continue
  fi
  # numeric compare with tolerance for trailing zeros / float dust
  if awk -v a="${actual}" -v e="${expected}" 'BEGIN { d = a - e; if (d < 0) d = -d; exit (d < 0.01 ? 0 : 1) }'; then
    echo "lot ${lot_id}: ${actual} lb (expected ${expected})  OK"
  else
    echo "lot ${lot_id}: ${actual} lb (expected ${expected})  *** MISMATCH ***"
    fail=1
  fi
done

echo ""
if [ "${fail}" -eq 0 ]; then
  echo "✅ Cleanup complete — all 16 lots match the expected end state."
else
  echo "❌ One or more lots mismatched. Investigate before any further adjustments."
  exit 1
fi
