#!/usr/bin/env python3
"""A11 staging acceptance, including the full 10,000-candidate sweep.

Only the documented staging DB is accepted. Migration is separately committed;
Synthetic actors, PINs, sessions, limits and attempts live only in a uniquely
named private staging namespace, which is removed at the end.
Never prints request/response bodies or any credential. No production access.
"""
import argparse
from contextlib import contextmanager
from datetime import timedelta
import json
import logging
import os
from pathlib import Path
import secrets
import sys
import time
from uuid import uuid4

import psycopg2
from psycopg2.extras import RealDictCursor
from fastapi.testclient import TestClient
from starlette.requests import Request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staging_safety import assert_staging_database, PRODUCTION_DATABASE_HOST
from scripts.seed_staging import secret_file


def stage_uri():
    from urllib.parse import urlsplit
    uri = secret_file(Path.home()/'Documents/fl-secrets/staging-db-url.txt')
    assert_staging_database(uri, 'staging', PRODUCTION_DATABASE_HOST)
    if urlsplit(uri).hostname != 'aws-0-us-east-1.pooler.supabase.com' or 'jygmyvxnxdjiiilhxseq' not in uri:
        raise RuntimeError('Refusing: not the documented staging database')
    return uri


def check(apply_migration=False):
    from concurrent.futures import ThreadPoolExecutor
    from psycopg2.pool import ThreadedConnectionPool
    from psycopg2 import sql
    uri = stage_uri()
    schema = 'a11_acceptance_' + uuid4().hex
    admin = psycopg2.connect(uri, port=5432, sslmode='require', connect_timeout=15)
    pool = None
    try:
        with admin, admin.cursor() as cur:
            cur.execute("SET LOCAL search_path=public; SET LOCAL lock_timeout='5s'")
            if apply_migration:
                cur.execute((ROOT/'migrations/072_pin_sessions.sql').read_text())
            cur.execute("SELECT current_user=pg_get_userbyid(relowner) FROM pg_class WHERE oid='public.actors'::regclass")
            assert cur.fetchone()[0], 'Staging backend must own its actors table'
            # A private, uniquely named acceptance namespace on REAL staging
            # Postgres. No real person's PIN or live lockout is exercised.
            cur.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
            cur.execute(sql.SQL('SET LOCAL search_path={},public').format(sql.Identifier(schema)))
            cur.execute('CREATE TABLE actors (LIKE public.actors INCLUDING ALL)')
            cur.execute('CREATE TABLE migration_markers (name text PRIMARY KEY,applied_at timestamptz DEFAULT clock_timestamp())')
            cur.execute((ROOT/'migrations/072_pin_sessions.sql').read_text())
        os.environ.update(DATABASE_URL=uri, ENVIRONMENT='staging', PRODUCTION_DATABASE_HOST=PRODUCTION_DATABASE_HOST,
                          PIN_PEPPER=secrets.token_hex(32), API_KEY=secrets.token_urlsafe(40), DASHBOARD_API_KEY=secrets.token_urlsafe(40))
        logging.disable(logging.CRITICAL)
        import main
        import pin_sessions as pins
        pool = ThreadedConnectionPool(1,12,uri,port=5432,sslmode='require',connect_timeout=15)
        @contextmanager
        def connection():
            conn=pool.getconn()
            try:
                with conn:
                    with conn.cursor() as cur:
                        cur.execute(sql.SQL('SET LOCAL search_path={},public').format(sql.Identifier(schema)))
                        cur.execute("SET LOCAL lock_timeout='30s'; SET LOCAL statement_timeout='60s'")
                    yield conn
            finally:pool.putconn(conn)
        main.get_db_connection = connection
        people = {}
        with connection() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
            for role in ('owner','floor','office'):
                while True:
                    pin = str(secrets.randbelow(7000)+2000)
                    if pins.valid_pin(pin) and pin not in [p['pin'] for p in people.values()]: break
                key = secrets.token_urlsafe(40)
                cur.execute('INSERT INTO actors(name,role,key_hash,pin_hash,active) VALUES (%s,%s,%s,%s,true) RETURNING id',
                            ('STG-A11-'+role+'-'+uuid4().hex[:10],role,pins.digest(key),pins.pin_hash(pin)))
                people[role] = {'id':cur.fetchone()['id'],'pin':pin,'key':key}
        main._reset_actor_cache()
        client = TestClient(main.app)
        def login(role):
            r = client.post('/auth/session', json={'pin':people[role]['pin']})
            assert r.status_code == 200, 'Synthetic PIN login failed'
            return r.json()['session_token']
        def auth(token): return {'X-API-Key':token}
        owner, floor = login('owner'), login('floor')
        assert client.get('/auth/whoami',headers=auth(floor)).json()['actor']['id'] == people['floor']['id']
        assert client.get('/actors/pins',headers=auth(floor)).status_code == 403
        assert client.post('/actors/'+str(people['office']['id'])+'/pin',json={'pin':people['office']['pin']},headers=auth(owner)).status_code == 403
        # Auth before resource lookup is covered locally; this nonexistent
        # exception is read from public only after PIN verification succeeds.
        assert client.post('/exceptions/2147483000/approve',json={},headers=auth(owner)).status_code == 403
        assert client.post('/exceptions/2147483000/approve',json={},headers=auth(owner)|{'X-FL-Owner-PIN':people['floor']['pin']}).status_code == 401
        assert client.post('/exceptions/2147483000/approve',json={},headers=auth(owner)|{'X-FL-Owner-PIN':people['owner']['pin']}).status_code == 404
        assert client.post('/exceptions/2147483000/approve',json={},headers=auth(owner)).status_code == 403
        with connection() as conn,conn.cursor() as cur:
            cur.execute("UPDATE actor_sessions SET expires_at=clock_timestamp()-interval '1 second' WHERE token_hash=%s",(pins.digest(floor),))
        assert client.get('/auth/whoami',headers=auth(floor)).status_code == 401
        assert client.post('/actors/'+str(people['office']['id'])+'/pin',json={'pin':str(1)*4},headers=auth(owner)).status_code == 422
        with connection() as conn,conn.cursor() as cur:
            cur.execute('TRUNCATE pin_rate_limits,pin_attempts')  # private acceptance namespace only
        started=time.monotonic()
        def candidate(i):
            r=client.post('/auth/session',json={'pin':str(i).zfill(4)})
            return r.status_code, 'Retry-After' in r.headers
        # Establish the source lock before parallelizing the remaining space.
        first=[candidate(i) for i in range(5)]
        assert all(status==401 for status,_ in first) and first[-1][1]
        statuses=list(first)
        with ThreadPoolExecutor(max_workers=8) as executor:
            for result in executor.map(candidate,range(5,10000)):
                statuses.append(result)
                if len(statuses)%1000==0:
                    print(json.dumps({'sweep_attempts':len(statuses),'authenticated':sum(s==200 for s,_ in statuses)}),flush=True)
        assert all(status==401 for status,_ in statuses), 'Sweep unexpectedly bypassed rejection'
        with connection() as conn,conn.cursor() as cur:
            cur.execute('SELECT count(*),count(*) FILTER (WHERE blocked) FROM pin_attempts')
            attempts,blocked=cur.fetchone()
            assert (attempts,blocked)==(10000,9995)
            cur.execute('TRUNCATE pin_rate_limits')
        for i in range(50):
            req=Request({'type':'http','method':'POST','path':'/auth/session','client':('203.0.113.'+str(i),1234),'headers':[]})
            actor,err,_=pins.verify_pin(main,req,'invalid',purpose='staging-distributed')
            assert actor is None and err
        req=Request({'type':'http','method':'POST','path':'/auth/session','client':('192.0.2.100',1234),'headers':[]})
        actor,err,_=pins.verify_pin(main,req,people['owner']['pin'],purpose='staging-distributed')
        assert actor is None and err.headers.get('Retry-After')
        return {'environment':'staging','mode':'actual FastAPI HTTP routes; committed concurrent transactions in private namespace on real staging Postgres',
                'migration':'072_pin_sessions','sweep_attempts':attempts,'sweep_authenticated':0,
                'source_lock_at_attempt':5,'blocked_before_PIN_lookup':blocked,'global_failures_before_lock':50,
                'checks':['PIN identity','weak PIN refusal','owner-only PIN administration','fresh owner proof','wrong-person PIN denial','idle expiry','full concurrent sweep','distributed global lock'],
                'elapsed_seconds':round(time.monotonic()-started,1),'cleanup':'private acceptance namespace removed; public people/PINs/sessions/limits untouched'}
    finally:
        if pool:pool.closeall()
        admin.rollback()
        # Only this invocation's UUID namespace is removed. Public schema and
        # its additive migration remain. Identifiers never come from a user.
        with admin,admin.cursor() as cur:
            cur.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(schema)))
        admin.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply-migration',action='store_true');parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    try:
        result=check(args.apply_migration)
        encoded=json.dumps(result,indent=2)+'\n'
        if args.output:args.output.write_text(encoded)
        print(encoded)
    except BaseException as exc:
        # Raw psycopg error/query context might include credential hashes.
        print('A11 staging acceptance failed: '+type(exc).__name__,file=sys.stderr)
        sys.exit(1)
