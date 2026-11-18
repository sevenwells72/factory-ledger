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
openers (NEGATIVE_BALANCE, UNSHIPPED_PAST_DUE) are listed in FOLLOWUPS. The unused
SHIPMENT_PROOF_MISSING kind is superseded by BOL_PHOTO_MISSING (P1.14); no schema change. A6 opens BOL_PHOTO_MISSING once per
shipment at its 23:59 plant-time deadline, serialized with late uploads.
"""
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlsplit

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from staging_safety import assert_staging_database, database_host


def sweep(cur):
    """Escalate overdue open exceptions and shortage flags; returns (exceptions, flags)."""
    from ship_tickets import open_missing_photos
    open_missing_photos(cur)
    cur.execute("""UPDATE exceptions SET status='escalated', escalated_at=clock_timestamp()
                   WHERE status='open' AND due_at IS NOT NULL AND due_at < clock_timestamp()""")
    exceptions = cur.rowcount
    cur.execute("""UPDATE shortage_flags SET status='escalated'
                   WHERE status='open' AND due_at < clock_timestamp()""")
    return exceptions, cur.rowcount


def guarded_database_url():
    """DATABASE_URL, refused unless it is local or an explicitly guarded staging/production target."""
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


def main():
    url = guarded_database_url()
    with psycopg2.connect(url) as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout='5s'")
        cur.execute("SET LOCAL statement_timeout='60s'")
        exceptions, flags = sweep(cur)
    print(f'Escalated {exceptions} overdue exception(s) and {flags} shortage flag(s)')


def send_adjustment_alerts(pauses=(0, 45, 90), sleep=time.sleep):
    """Nightly retry: owner emails for corrections over 500 lb that are still owed.

    The app queues one `adjustment_alerts` row per large correction (migration 096)
    and tries to send it itself; whatever is still pending — the app has no email
    settings, or the send failed — goes out here, with this cron service's Resend
    settings. Its own connections, after the sweep's transaction has ended. It never
    raises and never changes the cron's exit status: a failed email stays pending
    for the next run. A retry never sends a second copy (see adjustment_alerts.py).
    """
    try:
        from contextlib import contextmanager
        from psycopg2.extras import RealDictCursor
        import adjustment_alerts
        url = guarded_database_url()

        @contextmanager
        def transaction():
            conn = psycopg2.connect(url)
            try:
                with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SET LOCAL lock_timeout='5s'")
                    cur.execute("SET LOCAL statement_timeout='60s'")
                    yield cur
            finally:
                conn.close()

        # An attempt with no answer (timeout, dropped connection) is repeated in this same
        # run, with the same idempotency key: Resend then returns the first answer or
        # sends it for the first time. Waiting for tomorrow night would be past the 23
        # hours in which a repeat is safe, and the alert would be left for a person.
        for pause in pauses:
            if pause:
                sleep(pause)
            result = adjustment_alerts.deliver(transaction)
            print(f'Large-adjustment emails: {result["status"]} ({result["count"]})'
                  + (f', {result["unconfirmed"]} left unconfirmed for a person to check' if result.get('unconfirmed') else ''))
            if not result.get('unanswered'):
                break
    except Exception as exc:
        # Never print a DSN or credential-bearing driver error.
        print(f'Large-adjustment emails failed ({type(exc).__name__}); they stay pending', file=sys.stderr)


def run_daily_reconciliation():
    """Nightly hook: run the daily reconciliation after the sweep, as its own plain command.

    Off unless RECONCILIATION_ENABLED=1 on this cron service (migration 094 must be
    applied first). It runs as a separate process — `python scripts/daily_reconciliation.py`,
    the exact line a Render Cron Job would use — so it shares nothing with the sweep's
    transaction and cannot undo or block it. Returns its exit code (0 when off).
    """
    if os.environ.get('RECONCILIATION_ENABLED', '').strip().lower() not in ('1', 'true', 'yes', 'on'):
        return 0
    script = Path(__file__).resolve().parent / 'daily_reconciliation.py'
    return subprocess.run([sys.executable, str(script)], check=False).returncode


if __name__ == '__main__':
    status = 0
    try:
        main()
    except Exception as exc:
        # Never print a DSN or credential-bearing driver error.
        print(f'Exceptions sweep failed ({type(exc).__name__})', file=sys.stderr)
        status = 1
    send_adjustment_alerts()
    # The reconciliation runs even when the sweep failed; either failure fails the cron run.
    sys.exit(run_daily_reconciliation() or status)
