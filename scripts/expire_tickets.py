#!/usr/bin/env python3
"""Expire prepared tickets. Explicit operation; never run by app startup.

DATABASE_URL must name a local DB, pass the staging isolation guard, or match
PRODUCTION_DATABASE_HOST with explicit ENVIRONMENT=production. Never schedules
itself; Railway cron setup is documented separately.
"""
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from staging_safety import assert_staging_database, database_host


def expire(cur):
    cur.execute("UPDATE write_tickets SET status='expired' WHERE status='prepared' AND expires_at < now()")
    return cur.rowcount


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
        count = expire(cur)
    print(f'Expired {count} prepared tickets')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Never print a DSN or credential-bearing driver error.
        print(f'Ticket expiry failed ({type(exc).__name__})', file=sys.stderr)
        sys.exit(1)
