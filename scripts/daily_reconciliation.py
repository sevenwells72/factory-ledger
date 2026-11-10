#!/usr/bin/env python3
"""Nightly inventory reconciliation (reconciliation.py, migration 094).

    python scripts/daily_reconciliation.py                 the nightly run
    python scripts/daily_reconciliation.py --dry-run       same check, prints the email, writes nothing
    python scripts/daily_reconciliation.py --dry-run --date 2026-11-09
                                                           replay one past plant day from timestamps

A plain command that reads only environment variables, so the same line runs
from the Railway exceptions-sweep cron today and as a Render Cron Job later:

    DATABASE_URL, ENVIRONMENT, PRODUCTION_DATABASE_HOST   same guard as scripts/exceptions_sweep.py
    RECONCILIATION_EMAIL_ENABLED   "1" to send; default off (the run is still saved)
    ALERT_EMAIL_TO, RESEND_API_KEY recipient(s) and the Resend key
    RECONCILIATION_EMAIL_FROM      optional sender; needs a domain verified in Resend

Order: snapshot -> check the business day -> save flags -> send email -> record
the send, all in one transaction, so a run is either fully saved or not at all.
If the email fails the run and its flags are still saved with the error, and
the command exits 2. It never writes to any table except the three of
migration 094. --dry-run runs inside a READ ONLY transaction (transaction-scoped,
never a session setting) and needs no migration when --date is given.

Exit codes: 0 done; 1 the job failed; 2 the job worked but the email did not send.
"""
import argparse
from datetime import date
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import reconciliation
from staging_safety import assert_staging_database, database_host


def guarded_database_url():
    """Same routing rules as scripts/exceptions_sweep.py; never connects."""
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
    return url


def run(cur, *, dry_run, day=None):
    """One job inside the caller's transaction. Returns reconciliation.run_job's outcome."""
    if day is not None:
        return reconciliation.replay(cur, day)
    if not reconciliation.migration_applied(cur):
        if dry_run:
            # Nothing saved yet to compare with: replay the plant day that just ended.
            cur.execute("SELECT (clock_timestamp() AT TIME ZONE 'America/New_York')::date - 1")
            return reconciliation.replay(cur, cur.fetchone()[0])
        raise RuntimeError('Migration 094 is not applied')
    return reconciliation.run_job(cur, dry_run=dry_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Nightly inventory reconciliation')
    parser.add_argument('--dry-run', action='store_true', help='print the email; write nothing, send nothing')
    parser.add_argument('--date', type=date.fromisoformat, help='replay this plant day (YYYY-MM-DD); needs --dry-run')
    args = parser.parse_args(argv)
    if args.date and not args.dry_run:
        parser.error('--date replays a past day and never saves; use it with --dry-run')
    url = guarded_database_url()
    conn = psycopg2.connect(url)
    try:
        with conn.cursor() as cur:
            # First statement of the transaction: one consistent view of the ledger for
            # the whole run, and for a dry run a transaction that cannot write at all.
            cur.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ' + (' READ ONLY' if args.dry_run else ''))
            cur.execute("SET LOCAL lock_timeout='5s'")
            cur.execute("SET LOCAL statement_timeout='120s'")
            outcome = run(cur, dry_run=args.dry_run, day=args.date)
        if args.dry_run:
            conn.rollback()
        else:
            conn.commit()
    except Exception as exc:
        conn.rollback()
        if not args.dry_run:
            try:
                with conn.cursor() as cur:
                    if reconciliation.migration_applied(cur):
                        reconciliation.record_failed_run(cur, type(exc).__name__)
                conn.commit()
            except Exception:
                conn.rollback()
        raise
    finally:
        conn.close()
    analysis, email = outcome['analysis'], outcome['email']
    if args.dry_run:
        print(f'DRY RUN. Nothing saved, nothing sent. Subject: {email["subject"]}\n')
        print(email['text'])
        return 0
    print(f'Reconciliation run {outcome["run_id"]} for {analysis["plant_date"]}: result {analysis["result"]}, '
          f'{len(analysis["flags"])} flag(s), email {outcome["email_status"]}')
    return 2 if outcome['email_status'] == 'failed' else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        # Never print a DSN or credential-bearing driver error.
        print(f'Daily reconciliation failed ({type(exc).__name__})', file=sys.stderr)
        sys.exit(1)
