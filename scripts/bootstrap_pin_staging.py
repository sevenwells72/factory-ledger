#!/usr/bin/env python3
"""Create the four DECIDED staging people with NO PINs, exactly once.

Michael's initial owner sign-in key is saved with mode 0600, never displayed.
Use the dashboard to set Michael's first PIN, then each person's PIN. The
script refuses existing people or an existing bootstrap file; it never resets.
"""
import os
from pathlib import Path
import secrets
import sys
import hashlib
import psycopg2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.check_pin_sessions_staging import stage_uri


def bootstrap():
    destination=Path.home()/'Documents/fl-secrets/staging-pin-owner-key.txt'
    people=[('Michael','owner'),('Arturo','floor'),('Luz','office'),('Miriam','office')]
    conn=psycopg2.connect(stage_uri(),port=5432,sslmode='require',connect_timeout=15)
    created=False
    try:
        with conn,conn.cursor() as cur:
            cur.execute("SET LOCAL search_path=public; SET LOCAL lock_timeout='5s'")
            cur.execute('SELECT pg_advisory_xact_lock(724111)')
            cur.execute('SELECT name FROM actors WHERE name=ANY(%s)',([n for n,_ in people],))
            if cur.fetchone() or destination.exists():raise RuntimeError('Staging people or bootstrap file already exist; no changes made')
            owner_key=secrets.token_urlsafe(40)
            fd=os.open(destination,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
            created=True
            with os.fdopen(fd,'w') as f:f.write(owner_key+'\n')
            for name,role in people:
                key=owner_key if role=='owner' else secrets.token_urlsafe(40)
                cur.execute('INSERT INTO actors(name,role,key_hash,active) VALUES (%s,%s,%s,true)',(name,role,hashlib.sha256(key.encode()).hexdigest()))
        print('Staging people created with no PINs. Michael’s initial sign-in key is in '+str(destination))
    except BaseException:
        if created:destination.unlink(missing_ok=True)
        raise
    finally:conn.close()


if __name__=='__main__':
    try:bootstrap()
    except BaseException as exc:
        print('Staging bootstrap refused or failed: '+type(exc).__name__,file=sys.stderr);sys.exit(1)
