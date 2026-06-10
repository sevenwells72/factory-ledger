#!/usr/bin/env bash
# Refresh tests/schema/schema.sql from the production database.
# SCHEMA ONLY — read-only catalog operation, never dumps data.
#
# pg_dump cannot use the transaction-mode pooler (port 6543); this script
# rewrites the URL to the session-mode pooler on port 5432.
#
# Usage: DATABASE_URL=<prod url> scripts/dump_prod_schema.sh
#        (or it will read DATABASE_URL from the repo .env)

set -euo pipefail

PG_BIN="/opt/homebrew/opt/postgresql@17/bin"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/tests/schema/schema.sql"

URL="${DATABASE_URL:-}"
if [ -z "$URL" ] && [ -f "$ROOT/.env" ]; then
    URL="$(grep '^DATABASE_URL=' "$ROOT/.env" | cut -d= -f2-)"
fi
[ -n "$URL" ] || { echo "DATABASE_URL not set and not found in .env" >&2; exit 1; }

# Transaction-mode pooler (6543) does not support pg_dump — use session mode.
URL="${URL/:6543\//:5432\/}"

"$PG_BIN/pg_dump" --schema-only -n public --no-owner --no-privileges \
    --no-comments "$URL" \
    | sed 's/^CREATE SCHEMA public;$/CREATE SCHEMA IF NOT EXISTS public;/' \
    > "$OUT"

if grep -qE '^(COPY|INSERT)' "$OUT"; then
    echo "ERROR: dump contains data statements — refusing to keep it." >&2
    rm -f "$OUT"
    exit 1
fi

echo "Wrote $(wc -l < "$OUT" | tr -d ' ') lines to $OUT (verified zero data rows)."
