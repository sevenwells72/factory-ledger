#!/usr/bin/env python3
"""Trial recall drill (trial_recall.py, migration 097).

    python scripts/trial_recall_drill.py               run the drill now and save it
    python scripts/trial_recall_drill.py --scheduled   the cron form: only Monday / Wednesday / Friday
                                                 (plant time), and only once per day
    python scripts/trial_recall_drill.py --dry-run     same drill; prints the email; saves nothing, sends nothing

STRICTLY READ-ONLY on the ledger: everything it reads from Factory Ledger is
read inside READ ONLY transactions (transaction-scoped, never a session
setting). Its one write is a single INSERT into trial_recall_runs, on a
separate connection, after the drill is finished. It reads the receiving Form
sheet and the BOL photos through google_evidence.py, which can only read.

A plain command that reads only environment variables, started by the Railway
exceptions-sweep cron (about 02:30 / 03:30 Eastern) when TRIAL_RECALL_ENABLED=1:

    DATABASE_URL, ENVIRONMENT, PRODUCTION_DATABASE_HOST   same guard as scripts/exceptions_sweep.py
    GOOGLE_OAUTH_CLIENT_ID, GOOGLE_OAUTH_CLIENT_SECRET, GOOGLE_OAUTH_REFRESH_TOKEN
    ANTHROPIC_API_KEY (and optional EXTRACTION_MODEL)     the extraction model that reads the BOL photo
    TRIAL_RECALL_EMAIL_ENABLED     "1" to send; default off (the run is still saved)
    ALERT_EMAIL_TO, RESEND_API_KEY, RECONCILIATION_EMAIL_FROM   the daily reconciliation's email setup

Only one run at a time (a transaction-level advisory lock on a guard
connection, as scripts/daily_reconciliation.py). Exit codes: 0 done (or not a
recall day, or already run today); 1 the job failed; 2 the job worked but the
email did not send.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import psycopg2
from psycopg2.extras import RealDictCursor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import google_evidence
import receiving_form
import trial_recall
from scripts.daily_reconciliation import guarded_database_url

ALREADY_RUNNING = 'Another trial recall is in progress. This run stopped: nothing saved, nothing sent.'


def read_only(url, work):
    """Run `work(cur)` in one READ ONLY, REPEATABLE READ transaction and return its answer."""
    conn = psycopg2.connect(url, cursor_factory=RealDictCursor)
    try:
        with conn.cursor() as cur:
            # First statement: one consistent view of the books, in a transaction that cannot write.
            cur.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            cur.execute("SET LOCAL lock_timeout='5s'")
            cur.execute("SET LOCAL statement_timeout='120s'")
            return work(cur)
    finally:
        conn.rollback()
        conn.close()


def write(url, work):
    """The one short writing transaction: an INSERT into trial_recall_runs."""
    conn = psycopg2.connect(url, cursor_factory=RealDictCursor)
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL lock_timeout='5s'")
            cur.execute("SET LOCAL statement_timeout='60s'")
            answer = work(cur)
        conn.commit()
        return answer
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def drill(url, *, client=None, extractor=None, rng=None, now=None):
    """Sheet -> books (read-only) -> photos -> names (read-only) -> verdicts -> email text."""
    started_at = now or datetime.now(timezone.utc)
    client = client or receiving_form.client_factory()
    sheet = dict(client.sheet(receiving_form.SHEET_ID, receiving_form.TAB),
                 sheet_id=receiving_form.SHEET_ID, tab=receiving_form.TAB)
    data = read_only(url, lambda cur: trial_recall.load(cur, sheet, rng=rng))
    trial_recall.gather_evidence(data, client, extractor)
    read_only(url, lambda cur: trial_recall.confirm_names(cur, data))
    analysis = trial_recall.analyse(data)
    return started_at, analysis, trial_recall.render_email(analysis, started_at)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Trial recall drill')
    parser.add_argument('--dry-run', action='store_true', help='print the email; save nothing, send nothing')
    parser.add_argument('--scheduled', action='store_true', help='run only on Mon/Wed/Fri plant time, once per day')
    args = parser.parse_args(argv)
    url = guarded_database_url()
    now = datetime.now(timezone.utc)
    if args.scheduled and not trial_recall.is_recall_day(now):
        print('Not a trial recall day (Monday, Wednesday, Friday). Nothing run.')
        return 0
    if args.dry_run:
        _, analysis, email = drill(url, now=now)
        print(f'DRY RUN. Nothing saved, nothing sent. Subject: {email["subject"]}\n')
        print(email['text'])
        return 0
    guard = psycopg2.connect(url)
    try:
        with guard.cursor() as cur:
            cur.execute('SELECT pg_try_advisory_xact_lock(hashtext(%s))', (trial_recall.RUN_LOCK_NAME,))
            if not cur.fetchone()[0]:
                print(ALREADY_RUNNING)
                return 0
        ready, done = read_only(url, lambda cur: (trial_recall.migration_applied(cur),
                                                  trial_recall.migration_applied(cur) and trial_recall.ran_today(cur, now)))
        if not ready:
            raise RuntimeError('Migration 097 is not applied')
        if args.scheduled and done:
            print('Today\'s trial recall already ran. Nothing run.')
            return 0
        try:
            started_at, analysis, email = drill(url, now=now)
        except Exception as exc:
            code = exc.code if isinstance(exc, google_evidence.GoogleError) else type(exc).__name__
            try:
                write(url, lambda cur: trial_recall.record_failed_run(cur, str(code), now))
            except Exception:
                pass
            raise
        delivery = trial_recall.deliver(email)
        run_id = write(url, lambda cur: trial_recall.save_run(cur, started_at=started_at, analysis=analysis,
                                                               email=email, delivery=delivery))
    finally:
        try:
            guard.rollback()            # ends the guard transaction: the lock is released only now
        finally:
            guard.close()
    print(f'Trial recall run {run_id}: forward {analysis["forward"]["result"]}, backward '
          f'{analysis["backward"]["result"]}, full recall possible {analysis["full_recall_possible"]}, '
          f'email {delivery["status"]}')
    return 2 if delivery['status'] == 'failed' else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        # Never print a DSN, a token or a credential-bearing driver error.
        detail = f': {exc}' if isinstance(exc, (google_evidence.GoogleError, receiving_form.FormLayoutError)) else ''
        print(f'Trial recall failed ({type(exc).__name__}){detail}', file=sys.stderr)
        sys.exit(1)
