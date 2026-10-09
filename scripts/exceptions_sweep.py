#!/usr/bin/env python3
"""Nightly exceptions sweep (A3b, design §7.1): escalate overdue items.

Idempotent — an item is escalated once (status 'open' → 'escalated',
escalated_at set) and never touched again; resolved/waived rows are skipped.
Explicit operation, never run by app startup. Same database guard as
scripts/expire_tickets.py: DATABASE_URL must name a local DB, pass the staging
isolation guard, or match PRODUCTION_DATABASE_HOST with explicit
ENVIRONMENT=production. Never schedules itself; the Railway cron is documented
in docs/deployments/a3b-exceptions-enforcement.md and is NOT created here.

Scope today: overdue SHORTAGE / UNIDENTIFIED_LOT (and any other kind carrying a
due_at) exceptions plus their shortage_flags rows. The design's other nightly
openers (SHIPMENT_PROOF_MISSING, NEGATIVE_BALANCE, UNSHIPPED_PAST_DUE) belong to
A6/A9 and are listed in FOLLOWUPS.
"""
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from staging_safety import assert_staging_database, database_host


def sweep(cur):
    """Escalate overdue open exceptions and shortage flags; returns (exceptions, flags)."""
    cur.execute("""UPDATE exceptions SET status='escalated', escalated_at=clock_timestamp()
                   WHERE status='open' AND due_at IS NOT NULL AND due_at < clock_timestamp()""")
    exceptions = cur.rowcount
    cur.execute("""UPDATE shortage_flags SET status='escalated'
                   WHERE status='open' AND due_at < clock_timestamp()""")
    return exceptions, cur.rowcount


def main():
    url = os.environ.get('DATABASE_URL', '')
    if not url:
        raise RuntimeError('DATABASE_URL is required')
    environment = os.environ.get('ENVIRONMENT', '').strip().lower()
    if environment == 'staging':
        assert_staging_database(url)
    elif environment == 'production':
        host = database_host(url)
        expected = os.environ.get('PRODUCTION_DATABASE_HOST', '').strip().lower().rstrip('.')
        if not expected or host != expected:
            raise RuntimeError('Production requires a matching PRODUCTION_DATABASE_HOST')
        if any(os.environ.get(k) for k in ('PGHOSTADDR', 'PGSERVICE', 'PGSERVICEFILE')):
            raise RuntimeError('Connection routing overrides are not supported')
    else:
        parts = urlsplit(url)
        if parts.scheme not in ('postgres', 'postgresql') or parts.hostname not in ('localhost', '127.0.0.1', '::1') or parts.query:
            raise RuntimeError('Only a local database or explicitly guarded staging/production is supported')
        if any(os.environ.get(k) for k in ('PGHOSTADDR', 'PGSERVICE', 'PGSERVICEFILE')):
            raise RuntimeError('Connection routing overrides are not supported')
    with psycopg2.connect(url) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout='5s'")
        cur.execute("SET LOCAL statement_timeout='60s'")
        exceptions, flags = sweep(cur)
    print(f'Escalated {exceptions} overdue exception(s) and {flags} shortage flag(s)')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Never print a DSN or credential-bearing driver error.
        print(f'Exceptions sweep failed ({type(exc).__name__})', file=sys.stderr)
        sys.exit(1)
