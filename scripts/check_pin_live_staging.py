#!/usr/bin/env python3
"""Credential-redacted smoke of the deployed A11 staging service only."""
from pathlib import Path
import json
import logging
import os
import secrets
import subprocess
import sys
from uuid import uuid4

import httpx
import psycopg2

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.check_pin_sessions_staging import stage_uri
import pin_sessions as pins

PROJECT='2206e070-d160-4528-a4f1-86a587ad88c3'
SERVICE='0d957be1-8787-41e5-ab38-de71287c30ce'
ENVIRONMENT='f4d219df-2fea-45e8-85de-466b36a86c07'
BASE='https://fastapi-staging-production-dd7b.up.railway.app'


def check():
    logging.disable(logging.CRITICAL)
    r=subprocess.run(['railway','variable','list','--project',PROJECT,'--service',SERVICE,'--environment',ENVIRONMENT,'--json'],capture_output=True,text=True,check=True)
    config=json.loads(r.stdout)
    assert config['ENVIRONMENT']=='staging' and config['RAILWAY_SERVICE_NAME']=='FastAPI-staging'
    assert config['STAGING_DATABASE_PROJECT_REF']=='jygmyvxnxdjiiilhxseq'
    os.environ['PIN_PEPPER']=config['PIN_PEPPER']
    conn=psycopg2.connect(stage_uri(),port=5432,sslmode='require',connect_timeout=15)
    made={};tokens=[]
    try:
        with conn,conn.cursor() as cur:
            for role in ('owner','floor'):
                while True:
                    value=str(secrets.randbelow(9000)+1000)
                    if not pins.valid_pin(value):continue
                    cur.execute('SELECT 1 FROM actors WHERE pin_hash=%s',(pins.pin_hash(value),))
                    if not cur.fetchone():break
                cur.execute('INSERT INTO actors(name,role,key_hash,pin_hash,active) VALUES (%s,%s,%s,%s,true) RETURNING id',
                            ('STG-A11-LIVE-'+role+'-'+uuid4().hex[:10],role,pins.digest(secrets.token_urlsafe(40)),pins.pin_hash(value)))
                made[role]={'id':cur.fetchone()[0],'pin':value}
        with httpx.Client(base_url=BASE,timeout=30) as http:
            def login(role):
                response=http.post('/auth/session',json={'pin':made[role]['pin']});assert response.status_code==200
                token=response.json()['session_token'];tokens.append(token);return {'X-API-Key':token}
            owner=login('owner');floor=login('floor')
            who=http.get('/auth/whoami',headers=floor);assert who.status_code==200
            assert who.json()['key_kind']=='session' and who.json()['actor']['id']==made['floor']['id']
            assert http.get('/actors/pins',headers=floor).status_code==403
            assert http.post('/exceptions/2147483000/approve',json={},headers=owner).status_code==403
            assert http.post('/exceptions/2147483000/approve',json={},headers=owner|{'X-FL-Owner-PIN':made['floor']['pin']}).status_code==401
            assert http.post('/exceptions/2147483000/approve',json={},headers=owner|{'X-FL-Owner-PIN':made['owner']['pin']}).status_code==404
            assert http.post('/exceptions/2147483000/approve',json={},headers=owner).status_code==403
            assert http.post('/actors/'+str(made['floor']['id'])+'/pin',json={'pin':str(1)*4},headers=owner).status_code==422
            with conn,conn.cursor() as cur:
                cur.execute("UPDATE actor_sessions SET expires_at=clock_timestamp()-interval '1 second' WHERE actor_id=%s",(made['floor']['id'],))
            assert http.get('/auth/session',headers=floor).status_code==401
            assert http.delete('/auth/session',headers=owner).status_code==200
            assert http.get('/auth/whoami',headers=owner).status_code==401
            page=http.get('/dashboard/pin-management.html');assert page.status_code==200 and 'session.js' in page.text
            js=http.get('/dashboard/session.js');assert js.status_code==200 and 'OWNER_PIN_REQUIRED' in js.text
            bootstrap=Path.home()/'Documents/fl-secrets/staging-pin-owner-key.txt'
            assert bootstrap.stat().st_mode & 0o777 == 0o600
            initial=http.post('/auth/session/key',json={'actor_key':bootstrap.read_text().strip()});assert initial.status_code==200
            initial_header={'X-API-Key':initial.json()['session_token']}
            assert initial.json()['actor']['name']=='Michael'
            people=http.get('/actors/pins',headers=initial_header);assert people.status_code==200
            people={a['name']:a for a in people.json()['actors']}
            assert all(people[name]['active'] and not people[name]['pin_set'] for name in ('Michael','Arturo','Luz','Miriam'))
            assert http.delete('/auth/session',headers=initial_header).status_code==200
        return {'environment':'staging','api':BASE,'checks':[
            'live PIN-to-person login','live session whoami','floor admin denial','fresh owner step-up',
            'wrong-person PIN denial','weak PIN refusal','idle expiry','logout invalidation',
            'served dashboard/session assets','Michael bootstrap sign-in','four people ready with no PINs set'],
            'synthetic_actors':'deactivated; PIN hashes cleared; sessions revoked','production_touched':False}
    finally:
        conn.rollback()
        if made:
            with conn,conn.cursor() as cur:
                ids=[p['id'] for p in made.values()]
                cur.execute('UPDATE actors SET active=false,pin_hash=NULL WHERE id=ANY(%s)',(ids,))
                cur.execute("UPDATE actor_sessions SET ended_at=clock_timestamp(),ended_reason='revoked' WHERE actor_id=ANY(%s) AND ended_at IS NULL",(ids,))
        conn.close()


if __name__=='__main__':
    try:
        result=check();encoded=json.dumps(result,indent=2)+'\n'
        if len(sys.argv)>1:Path(sys.argv[1]).write_text(encoded)
        print(encoded)
    except BaseException as exc:
        print('Live staging PIN smoke failed: '+type(exc).__name__,file=sys.stderr);sys.exit(1)
