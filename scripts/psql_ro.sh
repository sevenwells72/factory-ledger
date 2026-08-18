#!/usr/bin/env bash
# Open psql against production for read-only investigation.
#
# Connects via the SESSION-mode pooler (rewrites :6543/ to :5432/) so nothing
# you do can land on a server connection the app might receive next. Read-only
# enforcement is transaction-scoped — wrap every transaction yourself:
#
#     BEGIN TRANSACTION READ ONLY;
#     -- ... SELECTs ...
#     COMMIT;
#
# Do NOT "fix" this script with PGOPTIONS, SET SESSION CHARACTERISTICS, or
# psycopg2 set_session(readonly=True): session-level read-only GUCs leak onto
# shared 6543 pool connections (2026-08-17 SO-260811-002 READONLY_TRIPWIRE)
# and the session pooler silently drops startup options anyway. See CONTEXT.md
# "Read-only investigation access".
#
# Usage: scripts/psql_ro.sh [psql args...]
#        (reads DATABASE_URL from env or the repo .env)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

URL="${DATABASE_URL:-}"
if [ -z "$URL" ] && [ -f "$ROOT/.env" ]; then
    URL="$(grep '^DATABASE_URL=' "$ROOT/.env" | cut -d= -f2-)"
fi
[ -n "$URL" ] || { echo "DATABASE_URL not set and not found in .env" >&2; exit 1; }

# Never investigate through the transaction-mode pooler the app draws from.
URL="${URL/:6543\//:5432/}"

cat >&2 <<'BANNER'
== factory-ledger read-only investigation session (session-mode pooler) ==
Wrap every transaction:  BEGIN TRANSACTION READ ONLY; ... COMMIT;
This session is NOT read-only until you do.
BANNER

exec psql "$URL" "$@"
