#!/usr/bin/env python3
"""A4 acceptance on staging; optional explicit migration 060, never an app deploy.

Imports only the independent resolver, never main (which has startup writes).
Acceptance uses persisted aliases and a transaction-scoped READ ONLY transaction.
--apply-migration applies only 060 in a separate, explicit staging transaction.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import resolution
from staging_safety import assert_staging_database, PRODUCTION_DATABASE_HOST


def check(*, apply_migration=False):
    path = Path.home() / 'Documents/fl-secrets/staging-db-url.txt'
    if path.stat().st_mode & 0o077:
        raise RuntimeError('Staging URI file must be private (mode 600)')
    uri = path.read_text().strip()
    assert_staging_database(uri, 'staging', PRODUCTION_DATABASE_HOST)
    # Refuse an accidental non-staging host even when it is not production.
    if resolution.normalize(psycopg2.extensions.parse_dsn(uri)['host']) != 'aws-0-us-east-1.pooler.supabase.com':
        raise RuntimeError('URI does not point to the documented staging host')
    conn = psycopg2.connect(uri, connect_timeout=10)
    try:
        expected_seeds = {'ss': 'sunshine', 'bs': 'blue stripes', 'cls': 'classic',
                          'choc': 'chocolate', '#9': '9'}
        if apply_migration:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SET LOCAL lock_timeout = '5s'")
                cur.execute("SET LOCAL statement_timeout = '60s'")
                cur.execute((Path(__file__).resolve().parents[1] / 'migrations/060_search_aliases.sql').read_text())
                rows, available = resolution.read_aliases(cur)
                stored = {row['alias_norm']: resolution.normalize(row['expansion'])
                          for row in rows if row['kind'] == 'token' and row['active']}
                assert available and all(stored.get(key) == value for key, value in expected_seeds.items())
            conn.commit()
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            cur.execute("SET LOCAL statement_timeout = '15s'")
            rows, available = resolution.read_aliases(cur)
            stored = {row['alias_norm']: resolution.normalize(row['expansion'])
                      for row in rows if row['kind'] == 'token' and row['active']}
            assert available and all(stored.get(key) == value for key, value in expected_seeds.items())
            results = []
            for query, context, limit in [('Classic', {}, 5), ('chocolate chip', {}, 8),
                                           ('Sunshine 9', {}, 8), ('Sunshine 9', {'action': 'make'}, 8),
                                           ('#9', {}, 25), ('SS 9', {}, 8), ('SS', {}, 8),
                                           ('SSX', {}, 8), ('glass jar', {}, 8), ('CLS Specialty', {}, 8)]:
                result = resolution.resolve(cur, resolution.ResolveRequest(kind='product', query=query, context=context, limit=limit))
                if query in ('Classic', 'chocolate chip', 'Sunshine 9', 'SS', '#9', 'SS 9'):
                    assert result['outcome'] == 'ambiguous' and result['match'] is None
                if query == 'Sunshine 9':
                    expected = {283, 284} if context else set(range(283, 291))
                    assert {c['id'] for c in result['candidates']} == expected
                if query == '#9':
                    assert {107, 108, *range(283, 291)} <= {c['id'] for c in result['candidates']}
                if query == 'SS 9':
                    assert {c['id'] for c in result['candidates']} == set(range(283, 291))
                if query == 'CLS Specialty':
                    assert result['outcome'] == 'none' and result['match'] is None
                if query == 'SS':
                    assert all('ss' in c['name'].lower().split() for c in result['candidates'])
                    assert all(c['id'] != 136 for c in result['candidates'])
                if query in ('SSX', 'glass jar'):
                    assert result['expansions_applied'] == []
                if query == 'SSX':
                    assert result['outcome'] == 'none'
                results.append(dict(query=query, context=context, **result))
            # Exercise the actual APIRouter via HTTP against the same guarded
            # staging cursor. Auth is deliberately local; no hosted API claim.
            @contextmanager
            def transaction():
                yield cur
            app = FastAPI()
            app.include_router(resolution.build_router(transaction, lambda: True))
            with TestClient(app) as client:
                for query in ('Classic', 'chocolate chip'):
                    response = client.post('/resolve', json={'kind': 'product', 'query': query})
                    assert response.status_code == 200 and response.json()['match'] is None
            for quantity in (24, 25):
                result = resolution.resolve(cur, resolution.ResolveRequest(kind='unit', query='pouches',
                            context={'product_id': 145}, quantity=quantity))
                if quantity == 24:
                    assert result['draft']['quantity'] == 2 and result['draft']['unit'] == 'cases'
                else:
                    assert result['needs_clarification'] and 'draft' not in result
                results.append(dict(query=f'{quantity} pouches', context={'product_id': 145}, **result))
            cur.execute('SHOW transaction_read_only')
            assert cur.fetchone()['transaction_read_only'] == 'on'
            print(json.dumps({'checked_at': datetime.now(timezone.utc).isoformat(),
                              'mode': 'local A4 resolver/HTTP router against staging; no deployment',
                              'aliases': 'persisted migration 060 seeds; no synthetic fixture',
                              'migration_060_applied': apply_migration,
                              'transaction_read_only': True, 'results': results}, default=str, indent=2))
    finally:
        conn.rollback()
        conn.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply-migration', action='store_true', help='Apply only migration 060 to the verified staging database before acceptance')
    args = parser.parse_args()
    try:
        check(apply_migration=args.apply_migration)
    except Exception as exc:
        # Connection errors can include credentials/DSNs; report only the class.
        print('Staging acceptance failed: ' + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
