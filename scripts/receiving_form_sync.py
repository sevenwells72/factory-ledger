#!/usr/bin/env python3
"""Nightly read of the receiving Form (receiving_form.py, migration 095).

    python scripts/receiving_form_sync.py             the nightly run
    python scripts/receiving_form_sync.py --dry-run   same read and matching; prints the result; writes nothing

Reads the Google Sheet "Receiving# (Responses)" (read-only), saves each row as
seen, links every row that clearly matches an FL receipt (date + internal lot
+ product) and lists the rest for the owner as "Receiving form row with no
matching FL receipt". It never guesses a match and never changes the sheet,
the photos or any ledger table: its only writes are INSERTs into the three
receiving tables of migration 095.

A plain command that reads only environment variables, started by the Railway
exceptions-sweep cron when RECEIVING_FORM_SYNC_ENABLED=1:

    DATABASE_URL, ENVIRONMENT, PRODUCTION_DATABASE_HOST   same guard as scripts/exceptions_sweep.py
    GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET, GOOGLE_OAUTH_REFRESH_TOKEN
    RECEIVING_FORM_START_DATE     optional; default 2026-11-11

--dry-run runs inside a READ ONLY transaction (transaction-scoped, never a
session setting) and needs no migration. Exit codes: 0 done; 1 the job failed.
"""
import argparse
from pathlib import Path
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import google_evidence
import receiving_form
from scripts.daily_reconciliation import guarded_database_url

LOCK_NAME = 'factory_ledger.receiving_form_sync'
ALREADY_RUNNING = 'Another receiving Form read is in progress. This run stopped: nothing saved.'


def read_sheet(client=None):
    """The Form tab, fetched before any database transaction is opened."""
    client = client or receiving_form.client_factory()
    sheet = client.sheet(receiving_form.SHEET_ID, receiving_form.TAB)
    return dict(sheet, sheet_id=receiving_form.SHEET_ID, tab=receiving_form.TAB)


def run(url, sheet, *, dry_run):
    conn = psycopg2.connect(url, cursor_factory=RealDictCursor)
    try:
        with conn.cursor() as cur:
            if dry_run:
                cur.execute('SET TRANSACTION READ ONLY')      # this transaction only; cannot write at all
            cur.execute("SET LOCAL lock_timeout='5s'")
            cur.execute("SET LOCAL statement_timeout='120s'")
            if not dry_run:
                if not receiving_form.migration_applied(cur):
                    raise RuntimeError('Migration 095 is not applied')
                # Transaction-level, so it also holds on the 6543 transaction pooler.
                cur.execute('SELECT pg_try_advisory_xact_lock(hashtext(%s)) AS held', (LOCK_NAME,))
                if not cur.fetchone()['held']:
                    conn.rollback()
                    return None
            outcome = receiving_form.sync(cur, sheet, dry_run=dry_run)
        conn.rollback() if dry_run else conn.commit()
        return outcome
    except Exception as exc:
        conn.rollback()
        if not dry_run:
            try:
                with conn.cursor() as cur:
                    if receiving_form.migration_applied(cur):
                        receiving_form.record_failed_sync(cur, type(exc).__name__)
                conn.commit()
            except Exception:
                conn.rollback()
        raise
    finally:
        conn.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description='Nightly read of the receiving Form')
    parser.add_argument('--dry-run', action='store_true', help='print the result; write nothing')
    args = parser.parse_args(argv)
    url = guarded_database_url()
    try:
        sheet = read_sheet()
    except Exception as exc:
        # The sheet could not be read: leave a failed-run note so the owner list is not mistaken for current.
        if not args.dry_run:
            _note_failure(url, exc)
        raise
    outcome = run(url, sheet, dry_run=args.dry_run)
    if outcome is None:
        print(ALREADY_RUNNING)
        return 0
    label = 'DRY RUN. Nothing saved. ' if args.dry_run else f'Receiving Form read {outcome["run_id"]}: '
    print(f'{label}{outcome["rows_read"]} row(s) on or after the start date, {outcome["rows_linked"]} linked to an FL '
          f'receipt ({outcome["links_new"]} new link record(s)), {len(outcome["unmatched"])} with no clear match.')
    for item in outcome['unmatched']:
        print(f'  Row {item["row_number"]} ({item["timestamp"]}): {" ".join(item["reasons"])}')
    return 0


def _note_failure(url, exc):
    try:
        conn = psycopg2.connect(url, cursor_factory=RealDictCursor)
        try:
            with conn.cursor() as cur:
                if receiving_form.migration_applied(cur):
                    code = exc.code if isinstance(exc, google_evidence.GoogleError) else type(exc).__name__
                    receiving_form.record_failed_sync(cur, str(code))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        # Never print a DSN, a token or a credential-bearing driver error.
        detail = f': {exc}' if isinstance(exc, (google_evidence.GoogleError, receiving_form.FormLayoutError)) else ''
        print(f'Receiving Form read failed ({type(exc).__name__}){detail}', file=sys.stderr)
        sys.exit(1)
