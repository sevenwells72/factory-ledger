#!/usr/bin/env bash
# Create/refresh the LOCAL test database for the pytest suite.
#
# Usage:
#   scripts/setup_test_db.sh           # create DB + load schema if missing
#   scripts/setup_test_db.sh --fresh   # drop and rebuild from schema.sql
#
# Then run tests with:
#   TEST_DATABASE_URL=postgresql://localhost:5432/factory_ledger_test python3 -m pytest
#
# Requires: brew install postgresql@17  (matches prod server 17.x)
# Schema source: tests/schema/schema.sql — a schema-only pg_dump of prod
# (zero data rows). Refresh it with scripts/dump_prod_schema.sh if the prod
# schema changes.

set -euo pipefail

PG_BIN="/opt/homebrew/opt/postgresql@17/bin"
DB_NAME="factory_ledger_test"
HOST="localhost"
PORT="5432"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SCHEMA="$ROOT/tests/schema/schema.sql"

if [ ! -x "$PG_BIN/psql" ]; then
    echo "postgresql@17 not found — run: brew install postgresql@17" >&2
    exit 1
fi

PG_DATA="/opt/homebrew/var/postgresql@17"
PG_LOG="/opt/homebrew/var/log/postgresql@17.log"

if ! "$PG_BIN/pg_isready" -q -h "$HOST" -p "$PORT"; then
    echo "Starting postgresql@17 via pg_ctl..."
    "$PG_BIN/pg_ctl" -D "$PG_DATA" -l "$PG_LOG" start
    for _ in $(seq 1 30); do
        "$PG_BIN/pg_isready" -q -h "$HOST" -p "$PORT" && break
        sleep 1
    done
    "$PG_BIN/pg_isready" -q -h "$HOST" -p "$PORT" || {
        echo "Postgres did not come up on $HOST:$PORT (see $PG_LOG)" >&2; exit 1;
    }
fi

if [ "${1:-}" = "--fresh" ]; then
    "$PG_BIN/dropdb" -h "$HOST" -p "$PORT" --if-exists "$DB_NAME"
fi

if "$PG_BIN/psql" -h "$HOST" -p "$PORT" -d postgres -tAc \
        "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" | grep -q 1; then
    echo "Database $DB_NAME already exists (use --fresh to rebuild)."
else
    "$PG_BIN/createdb" -h "$HOST" -p "$PORT" "$DB_NAME"
    # pg_trgm: on Supabase it lives in the 'extensions' schema and is not in
    # the public-schema dump; /products/resolve & customer matching need it.
    "$PG_BIN/psql" -h "$HOST" -p "$PORT" -d "$DB_NAME" -v ON_ERROR_STOP=1 \
        -c "CREATE EXTENSION IF NOT EXISTS pg_trgm"
    "$PG_BIN/psql" -h "$HOST" -p "$PORT" -d "$DB_NAME" -v ON_ERROR_STOP=1 \
        -q -f "$SCHEMA"
    echo "Database $DB_NAME created and schema loaded."
fi

echo
echo "Run tests with:"
echo "  TEST_DATABASE_URL=postgresql://$HOST:$PORT/$DB_NAME python3 -m pytest"
