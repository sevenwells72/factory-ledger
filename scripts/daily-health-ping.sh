#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   HEALTH_URL="https://your-api-endpoint/health" ./scripts/daily-health-ping.sh
# or:
#   ./scripts/daily-health-ping.sh "https://your-api-endpoint/health"

HEALTH_URL="${1:-${HEALTH_URL:-}}"

if [[ -z "${HEALTH_URL}" ]]; then
  echo "Error: set HEALTH_URL or pass the health URL as argument."
  exit 1
fi

timestamp="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"

http_code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 20 "${HEALTH_URL}")"

if [[ "${http_code}" == "200" ]]; then
  echo "[${timestamp}] ping ok (${http_code}) ${HEALTH_URL}"
else
  echo "[${timestamp}] ping failed (${http_code}) ${HEALTH_URL}"
  exit 1
fi
