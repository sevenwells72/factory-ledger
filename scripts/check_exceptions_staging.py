#!/usr/bin/env python3
"""A3b acceptance on guarded staging through local HTTP, never hosted deployment.

Same shape as scripts/check_lot_confirmation_staging.py and
check_order_tickets_staging.py: reads only the protected staging URI; no
production configuration or API keys; TestClient without a lifespan context, so
no startup migrations or sweeps run. `--apply-migration` applies 069 and 070 in one
guarded transaction each (the script refuses to run the examples without them);
`--migrations-only` stops after that and reports the markers.

Three examples Michael asked for, all retained as evidence:
  1. a short make — posted with a shortage flag + SHORTAGE exception;
  2. a > 500 lb adjust without a photo — HELD (awaiting_approval), then approved
     once by a temporary owner (replay returns the original receipt);
  3. a denied permission attempt — the floor actor tries to approve, the
     office actor tries to resolve, the master key tries to list.
Synthetic products/lots and two temporary actors (floor, owner) are created;
the actor keys exist only in memory and both actors are deactivated in finally.
"""
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import logging
import os
from pathlib import Path
import secrets
import sys
from urllib.parse import urlsplit
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staging_safety import assert_staging_database, PRODUCTION_DATABASE_HOST
from scripts.seed_staging import secret_file

MIGRATIONS = ('069_exceptions_enforcement', '070_pre_make_adjust')


def check(apply_migration=False, migrations_only=False):
    uri = secret_file(Path.home() / 'Documents/fl-secrets/staging-db-url.txt')
    assert_staging_database(uri, 'staging', PRODUCTION_DATABASE_HOST)
    if urlsplit(uri).hostname != 'aws-0-us-east-1.pooler.supabase.com':
        raise RuntimeError('Not the documented staging host')

    @contextmanager
    def connection():
        conn = psycopg2.connect(uri, port=5432, connect_timeout=10, sslmode='require')
        try:
            with conn:
                with conn.cursor() as cur:
                    cur.execute("SET LOCAL search_path=public")
                    cur.execute("SET LOCAL lock_timeout='5s'")
                    cur.execute("SET LOCAL statement_timeout='60s'")
                yield conn
        finally:
            conn.close()

    if apply_migration:
        for migration in MIGRATIONS:
            with connection() as conn, conn.cursor() as cur:
                cur.execute("SELECT 1 FROM migration_markers WHERE name IN ('061_exceptions_tables','065_entered_by')")
                if len(cur.fetchall()) != 2:
                    raise RuntimeError('069/070 need 061 and 065 on staging first')
                cur.execute((ROOT / 'migrations' / f'{migration}.sql').read_text())
    with connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT name, applied_at FROM migration_markers WHERE name = ANY(%s) ORDER BY name", (list(MIGRATIONS),))
        markers = {row[0]: row[1] for row in cur.fetchall()}
        missing = [m for m in MIGRATIONS if m not in markers]
        if missing:
            raise RuntimeError(f'Apply {", ".join(missing)} to staging before acceptance (--apply-migration)')
        if migrations_only:
            return {'mode': 'migrations only', 'migration_applied_now': apply_migration, 'markers': markers}
        cur.execute("SELECT count(*) FROM correction_reasons WHERE active")
        if cur.fetchone()[0] != 8:
            raise RuntimeError('Staging correction_reasons seed is not the fixed list of 8')

    os.environ['DATABASE_URL'] = uri
    os.environ['ENVIRONMENT'] = 'staging'
    os.environ['PRODUCTION_DATABASE_HOST'] = PRODUCTION_DATABASE_HOST
    os.environ['API_KEY'] = secrets.token_urlsafe(40)
    os.environ['DASHBOARD_API_KEY'] = secrets.token_urlsafe(40)
    logging.disable(logging.CRITICAL)
    import main
    main.get_db_connection = connection
    main._reset_actor_cache()
    reference = 'STG-A3B-' + uuid4().hex[:12].upper()
    keys = {role: secrets.token_urlsafe(40) for role in ('floor', 'owner', 'office')}
    actor_ids = {}
    client = TestClient(main.app)
    try:
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            for role, key in keys.items():
                cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES (%s,%s,%s,true) RETURNING id",
                            (f'{reference} {role}', role, sha256(key.encode()).hexdigest()))
                actor_ids[role] = cur.fetchone()['id']
            products = {}
            for label, kind in [('ingredient', 'ingredient'), ('batch', 'batch')]:
                cur.execute("""INSERT INTO products(name,odoo_code,type,uom,default_batch_lb,active)
                    VALUES (%s,%s,%s,'lb',10,true) RETURNING id""", (f'{reference} {label}', f'{reference}-{label}', kind))
                products[label] = cur.fetchone()['id']
                assert products[label] >= 1_000_000_000
            cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,10)',
                        (products['batch'], products['ingredient']))
            # One ingredient lot with 4 lb on hand: the make below needs 10.
            cur.execute('INSERT INTO lots(product_id,lot_code,entry_source) VALUES (%s,%s,%s) RETURNING id,lot_code',
                        (products['ingredient'], f'{reference}-ING', 'found_inventory'))
            lot = cur.fetchone()
            cur.execute("INSERT INTO transactions(type,notes,operator_id) VALUES ('adjust',%s,'staging-fixture') RETURNING id",
                        (f'{reference} fixture 4 lb',))
            cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,4)',
                        (cur.fetchone()['id'], products['ingredient'], lot['id']))

        def headers(role):
            return {'X-API-Key': keys[role]}

        def post(path, payload, role, status=200):
            response = client.post(path, json=payload, headers=headers(role))
            if response.status_code != status:
                raise RuntimeError(f'Unexpected HTTP {response.status_code} on {path.split("/")[1]} (wanted {status})')
            return response.json()

        def commit(draft, role, status=200, **extra):
            return post('/tickets/' + draft['ticket'] + '/commit', {'payload_hash': draft['payload_hash'],
                        'acknowledged_warnings': [w['code'] for w in draft['warnings'] if w.get('requires_ack')], **extra},
                        role, status)

        # ── 1. short make: posted + flagged, never blocked ──────────────────
        draft = post('/make/prepare', {'product_id': products['batch'], 'batches': 1,
                     'lot_confirmations': [{'lot_id': lot['id'], 'method': 'full_code', 'value': lot['lot_code']}]}, 'floor')
        assert draft['can_commit'], draft['blockers']
        warning = next(w for w in draft['warnings'] if w['code'] == 'WILL_CREATE_SHORTAGE')
        assert warning['requires_ack'] is False and warning['refs']['short_lb'] == 6.0
        made = commit(draft, 'floor')
        assert made['shortages'][0]['short_lb'] == 6.0
        assert commit(draft, 'floor') == made | {'replayed': True}
        shortage = made['shortages'][0]
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT count(*) AS n FROM shortage_flags WHERE transaction_id=%s', (made['transaction_id'],))
            assert cur.fetchone()['n'] == 1
            cur.execute('SELECT status,kind,due_at,owner_actor_id FROM exceptions WHERE id=%s', (shortage['exception_id'],))
            exc = cur.fetchone()
            assert exc['kind'] == 'SHORTAGE' and exc['status'] == 'open'
            balance = main.lot_on_hand(cur, lot['id'])
        assert balance == -6.0

        # ── 3a. never add stock to cover: refused with a pointer to the exception ──
        refused = post('/adjust/prepare', {'lot_id': lot['id'], 'delta_lb': 6, 'reason_code': 'missing_receipt'}, 'floor', 409)
        assert refused['detail']['error_code'] == 'SHORTAGE_OPEN_RESOLVE_INSTEAD'

        # ── 2. > 500 lb correction without a photo: held, then approved once ──
        big = post('/adjust/prepare', {'lot_id': lot['id'], 'delta_lb': -600, 'reason_code': 'damage_disposal'}, 'floor')
        assert big['can_commit'] and big['draft']['correction_review']['photo_required']
        held = commit(big, 'floor', 202)
        assert held['held'] and held['status'] == 'awaiting_approval'
        exception_id = held['exception_id']
        held_again = commit(big, 'floor', 202)
        assert held_again['exception_id'] == exception_id
        # ── 3b. denied permission attempts ───────────────────────────────
        denied_floor = post(f'/exceptions/{exception_id}/approve', {}, 'floor', 403)
        assert denied_floor['detail']['error_code'] == 'ROLE_NOT_ALLOWED'
        denied_office = post(f'/exceptions/{shortage["exception_id"]}/resolve',
                             {'resolution_kind': 'counted', 'note': 'office trying'}, 'office', 403)
        assert denied_office['detail']['error_code'] == 'ROLE_NOT_ALLOWED'
        master = client.get('/exceptions', headers={'X-API-Key': os.environ['API_KEY']})
        assert master.status_code == 403 and master.json()['detail']['error_code'] == 'ROLE_NOT_ALLOWED'
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT count(*) AS n FROM transactions WHERE ticket_id=%s', (big['ticket_id'],))
            assert cur.fetchone()['n'] == 0
        approved = post(f'/exceptions/{exception_id}/approve', {'note': 'Photo seen on the floor (staging acceptance)'}, 'owner')
        assert approved['replayed'] is False and approved['approval']['approved_by']['id'] == actor_ids['owner']
        replay = post(f'/exceptions/{exception_id}/approve', {}, 'owner')
        assert replay == approved | {'replayed': True}
        assert commit(big, 'floor') == approved | {'replayed': True}
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SELECT count(*) AS n, min(entered_by_actor_id) AS entered_by, min(reason_code) AS reason_code '
                        'FROM transactions WHERE ticket_id=%s', (big['ticket_id'],))
            posted = cur.fetchone()
            assert posted['n'] == 1 and posted['entered_by'] == actor_ids['floor'] and posted['reason_code'] == 'damage_disposal'
            cur.execute('SELECT status,resolution_kind,resolved_by_actor_id FROM exceptions WHERE id=%s', (exception_id,))
            exc2 = cur.fetchone()
            assert (exc2['status'], exc2['resolution_kind'], exc2['resolved_by_actor_id']) == ('resolved', 'approved', actor_ids['owner'])
            cur.execute('SELECT reason_code FROM ledger_current_transactions WHERE id=%s', (approved['transaction_id'],))
            assert cur.fetchone()['reason_code'] == 'damage_disposal'
            balance_after = main.lot_on_hand(cur, lot['id'])
        assert balance_after == -606.0

        # ── resolve the shortage as the floor (who/when), then list it ───────
        # 'counted' needs the physical count and posts the counted correction atomically (0 lb → +6).
        resolved = post(f'/exceptions/{shortage["exception_id"]}/resolve',
                        {'resolution_kind': 'counted', 'counted_lb': 0, 'note': 'Counted: nothing left on the pallet (staging acceptance)'}, 'floor')
        assert resolved['status'] == 'resolved' and resolved['resolved_by']['id'] == actor_ids['floor']
        assert resolved['detail']['resolution']['transaction_id'] and resolved['detail']['resolution']['new_balance_lb'] == 0.0
        listed = client.get('/exceptions?status=all&lot_id=' + str(lot['id']), headers=headers('owner')).json()
        assert {e['id'] for e in listed['exceptions']} >= {shortage['exception_id'], exception_id}
        return {'reference': reference, 'mode': 'local branch HTTP routes against STAGING; no hosted deployment',
                'migration_applied_now': apply_migration,
                'example_1_short_make': {'receipt': made['receipt_number'], 'transaction_id': made['transaction_id'],
                                         'lot_id': lot['id'], 'lot_code': lot['lot_code'], 'short_lb': shortage['short_lb'],
                                         'balance_after_lb': balance, 'exception_id': shortage['exception_id'],
                                         'shortage_flag_id': shortage['shortage_flag_id'], 'due_at': shortage['due_at'],
                                         'warning': warning['message']},
                'example_2_large_correction': {'ticket_id': big['ticket_id'], 'held_response': held,
                                               'exception_id': exception_id, 'approved_receipt': approved['receipt_number'],
                                               'transaction_id': approved['transaction_id'],
                                               'entered_by_actor_id': actor_ids['floor'], 'approved_by_actor_id': actor_ids['owner'],
                                               'balance_after_lb': balance_after},
                'example_3_denied': {'floor_approve': denied_floor['detail']['error_code'],
                                     'office_resolve': denied_office['detail']['error_code'],
                                     'master_key_list': master.json()['detail']['error_code'],
                                     'cover_up_adjust': refused['detail']['error_code']},
                'shortage_resolved_by': resolved['resolved_by'], 'shortage_resolved_at': resolved['resolved_at'],
                'shortage_resolution': resolved['detail']['resolution'],
                'actors': actor_ids, 'actors_deactivated_after_check': True, 'products': products}
    finally:
        client.close()
        if actor_ids:
            with connection() as conn, conn.cursor() as cur:
                cur.execute('UPDATE actors SET active=false WHERE id=ANY(%s)', (list(actor_ids.values()),))
        main._reset_actor_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply-migration', action='store_true')
    parser.add_argument('--migrations-only', action='store_true', help='apply/check the markers and stop; no examples')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = check(args.apply_migration, args.migrations_only)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + '\n')
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    except Exception as exc:
        # Never emit DB connection exception strings or tracebacks with secrets.
        print('A3b staging acceptance failed: ' + type(exc).__name__ + ': ' + str(exc)[:200].replace(os.environ.get('DATABASE_URL', '\0'), '<redacted>'), file=sys.stderr)
        raise SystemExit(1)
