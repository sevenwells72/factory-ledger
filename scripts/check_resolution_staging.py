#!/usr/bin/env python3
"""A4 acceptance on live staging data, without deploying or writing any rows.

Imports only the independent resolver, never main (which has startup writes).
SS/Sunshine is a synthetic in-memory fixture; no draft shorthand is read/seeded.
The database enforces an explicit transaction-scoped READ ONLY transaction.
"""
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


def check():
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
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            cur.execute("SET LOCAL statement_timeout = '15s'")
            fixture = dict(id=0, kind='token', alias='SS', alias_norm='ss', expansion='Sunshine', active=True)
            db_aliases, _ = resolution.read_aliases(cur)
            aliases = db_aliases + [fixture]
            results = []
            for query, context in [('Classic', {}), ('chocolate chip', {}), ('Sunshine 9', {}),
                                   ('Sunshine 9', {'action': 'make'}), ('SS', {}), ('SSX', {}), ('glass jar', {})]:
                result = resolution.resolve(cur, resolution.ResolveRequest(kind='product', query=query, context=context), aliases=aliases)
                if query in ('Classic', 'chocolate chip', 'Sunshine 9', 'SS'):
                    assert result['outcome'] == 'ambiguous' and result['match'] is None
                if query == 'Sunshine 9':
                    expected = {283, 284} if context else set(range(283, 291))
                    assert {c['id'] for c in result['candidates']} == expected
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
                              'alias_fixture': 'SS=Sunshine in memory only; alias_id=0 is synthetic',
                              'transaction_read_only': True, 'results': results}, default=str, indent=2))
    finally:
        conn.rollback()
        conn.close()


if __name__ == '__main__':
    try:
        check()
    except (psycopg2.Error, OSError) as exc:
        # Connection errors can include credentials/DSNs; report only the class.
        print('Staging acceptance failed: ' + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
