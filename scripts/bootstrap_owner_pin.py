#!/usr/bin/env python3
"""Michael's first PIN, before activation. No HTTP route or flag bypass.

Run privately with the backend DATABASE_URL and PIN_PEPPER in the environment.
Only an existing active owner without a PIN is eligible. No value is echoed,
accepted on the command line, saved to disk or included in errors.
"""
import argparse
import getpass
import logging
import os
from pathlib import Path
import sys
import warnings

import psycopg2
from psycopg2.extras import RealDictCursor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pin_sessions as pins


def bootstrap(cur, actor_id, value):
    if not pins.valid_pin(value):
        raise ValueError('PIN policy')
    hashed = pins.pin_hash(value)
    cur.execute("SELECT relowner=(SELECT oid FROM pg_roles WHERE rolname=current_user) AS owned FROM pg_class WHERE oid='actors'::regclass")
    if not cur.fetchone()['owned']:
        raise ValueError('Application must own actors')
    cur.execute('SELECT pg_advisory_xact_lock(724111)')
    cur.execute('SELECT active,role,pin_hash FROM actors WHERE id=%s FOR UPDATE', (actor_id,))
    owner = cur.fetchone()
    if not owner or not owner['active'] or owner['role'] != 'owner' or owner['pin_hash'] is not None:
        raise ValueError('An active owner with no PIN is required')
    cur.execute('SELECT 1 FROM actors WHERE pin_hash=%s', (hashed,))
    if cur.fetchone():
        raise ValueError('PIN unavailable')
    cur.execute('UPDATE actors SET pin_hash=%s,pin_set_at=clock_timestamp() WHERE id=%s', (hashed, actor_id))
    cur.execute("UPDATE actor_sessions SET ended_at=clock_timestamp(),ended_reason='pin_reset' WHERE actor_id=%s AND ended_at IS NULL", (actor_id,))
    cur.execute('INSERT INTO pin_management_audit(actor_id,changed_by_actor_id) VALUES (%s,%s)', (actor_id, actor_id))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor-id', type=int, required=True)
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    # Refuse getpass's echoing fallback (e.g. a piped non-interactive session).
    warnings.simplefilter('error', getpass.GetPassWarning)
    if pins.enabled():
        raise ValueError('Bootstrap must precede activation')
    value = getpass.getpass('Michael: new PIN (hidden): ')
    if value != getpass.getpass('Confirm PIN (hidden): '):
        raise ValueError('Confirmation differs')
    with psycopg2.connect(os.environ['DATABASE_URL'], connect_timeout=15) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SET LOCAL search_path=public; SET LOCAL lock_timeout='5s'")
            bootstrap(cur, args.actor_id, value)
    print('First owner PIN saved. PIN login remains disabled.')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('Bootstrap refused or failed; no credentials displayed. Check owner eligibility, PIN policy and configuration.', file=sys.stderr)
        sys.exit(1)
