#!/usr/bin/env python3
"""Guarded pilot acceptance: synthetic named people/stock, no printed secrets.

Uses only the protected staging URI via the existing isolation guard. No main
import or production reads. Temporary keys live only in process memory.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import secrets
import time
from uuid import uuid4
import httpx
from psycopg2.extras import RealDictCursor
from check_assistant_staging import connect, BASE, ROOT


def migrate():
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s'; SET LOCAL search_path=public")
        cur.execute((ROOT/'migrations/077_key_sessions.sql').read_text())
        cur.execute("SELECT name FROM migration_markers WHERE name='077_key_sessions'")
        assert cur.fetchone()[0]=='077_key_sessions'
    print('077 applied to verified staging only; parked PIN tables untouched.',flush=True)


def check(output=None):
    marker='STG-SIGNIN-'+uuid4().hex[:8].upper()
    actors={}; keys={name:secrets.token_urlsafe(40) for name in ('Arturo','Michael','Luz','Miriam')}
    sessions={}; h={}; evidence={'reference':marker,'base_url':BASE,'checked_at':datetime.now(timezone.utc).isoformat(),'checks':[]}
    def ok(response):
        assert response.status_code in (200,202), 'Staging request refused: HTTP '+str(response.status_code)
        return response.json()
    try:
        with connect() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SET LOCAL lock_timeout='5s'; SET LOCAL search_path=public")
            for name,key in keys.items():
                role='floor' if name=='Arturo' else 'owner' if name=='Michael' else 'office'
                cur.execute('INSERT INTO actors(name,role,key_hash,active) VALUES (%s,%s,%s,true) RETURNING id,name,role',
                            (name+' (STAGING '+marker+')',role,sha256(key.encode()).hexdigest()))
                actors[name]=dict(cur.fetchone())
            cur.execute("INSERT INTO products(name,odoo_code,type,uom,active) VALUES (%s,%s,'ingredient','lb',true) RETURNING id,name",(marker+' Almonds',marker+'-ALM'))
            product=dict(cur.fetchone())
            cur.execute('INSERT INTO suppliers(name) VALUES (%s) RETURNING id,name',(marker+' Supplier',))
            supplier=dict(cur.fetchone())
        with httpx.Client(base_url=BASE,timeout=120) as http:
            config=ok(http.get('/auth/session/config'))
            assert config['key_sessions_enabled'] and config['idle_minutes']==180
            for name,key in keys.items():
                result=ok(http.post('/auth/session',json={'actor_key':key}))
                assert result['key_kind']=='session' and result['actor']['id']==actors[name]['id']
                sessions[name]=result['session_token'];h[name]={'X-API-Key':result['session_token']}
                who=ok(http.get('/auth/whoami',headers=h[name]));assert who['key_kind']=='session'
            evidence['checks'].append('All four synthetic named people exchanged keys with PIN off')
            with connect() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
                for name,token in sessions.items():
                    cur.execute('SELECT token_hash,expires_at-last_seen_at AS idle FROM key_actor_sessions WHERE token_hash=%s',(sha256(token.encode()).hexdigest(),))
                    row=cur.fetchone();assert row and row['idle'].total_seconds()==180*60
            evidence['checks'].append('Only token hashes persisted; 180-minute sliding session expiry')
            sid=ok(http.post('/assistant/session',headers=h['Arturo']))['session_id']
            body={'session_id':sid,'turn_id':str(uuid4()),'text':f'Receive 1 case of {product["name"]}, 10 lb per case, from {supplier["name"]}, BOL {marker}, supplier lot {marker}-LOT. Prepare a draft for my review.'}
            response=None
            for attempt in range(3):
                raw=http.post('/assistant/turn',headers=h['Arturo'],json=body)
                if raw.status_code==502 and attempt<2:
                    time.sleep(2);continue
                response=ok(raw);break
            cards=response.get('cards',[])
            # If the real resolver asks, explicitly choose this synthetic fixture.
            for _ in range(4):
                choices=next((c for c in cards if c['kind']=='choices'),None)
                if not choices:break
                target=product['id'] if choices['resolution_kind']=='product' else supplier['id']
                candidates=choices['result'].get('candidates',[])
                assert any(c['id']==target for c in candidates),'Expected staging fixture not offered'
                response=ok(http.post('/assistant/turn',headers=h['Arturo'],json={'session_id':sid,'turn_id':str(uuid4()),'text':'','choice_id':choices['id'],'selected_id':target}))
                cards=response.get('cards',[])
            card=next(c for c in cards if c['kind']=='draft')
            assert card['prepared']['actor']['id']==actors['Arturo']['id']
            record=ok(http.post('/assistant/record',headers=h['Arturo'],json={'draft_id':card['id']}))
            assert record['kind']=='receipt'; result=record['result']
            replay=ok(http.post('/assistant/record',headers=h['Arturo'],json={'draft_id':card['id']}))
            assert replay['result']['receipt_number']==result['receipt_number']
            with connect() as conn,conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute('SELECT id,actor_id,expires_at-prepared_at AS ttl FROM write_tickets WHERE receipt_number=%s',(result['receipt_number'],))
                ticket=cur.fetchone();assert ticket['actor_id']==actors['Arturo']['id'] and ticket['ttl'].total_seconds()==600
                cur.execute('SELECT id,entered_by_actor_id,operator_id FROM transactions WHERE ticket_id=%s',(ticket['id'],))
                rows=cur.fetchall();assert len(rows)==1 and rows[0]['entered_by_actor_id']==actors['Arturo']['id']
                entry=dict(rows[0])
            evidence.update({'actor':actors['Arturo'],'receipt_number':result['receipt_number'],'transaction':entry,'ticket_id':ticket['id'],'quantity_lb':10,'assistant_chat_id':sid,'draft_id':card['id']})
            evidence['checks'].append('Arturo real-model Assistant prepare/Record/replay: one person-attributed 10 lb receive, 10-minute ticket')
            ok(http.post('/auth/session/logout',headers=h['Arturo']))
            refused=http.post('/assistant/record',headers=h['Arturo'],json={'draft_id':card['id']})
            assert refused.status_code==401 and refused.json()['detail']['error_code']=='SESSION_EXPIRED'
            evidence['checks'].append('Sign out ends token; subsequent Record receives SESSION_EXPIRED')
            with connect() as conn,conn.cursor() as cur:
                cur.execute("UPDATE key_actor_sessions SET expires_at=clock_timestamp()-interval '1 second',last_seen_at=clock_timestamp()-interval '180 minutes' WHERE token_hash=%s",(sha256(sessions['Luz'].encode()).hexdigest(),))
                cur.execute('UPDATE actors SET active=false WHERE id=%s',(actors['Miriam']['id'],))
            for name in ('Luz','Miriam'):
                r=http.get('/auth/whoami',headers=h[name]);assert r.status_code==401 and r.json()['detail']['error_code']=='SESSION_EXPIRED'
            evidence['checks'].append('180-minute idle and revoked key are rejected immediately')
            # Master credentials are read only from their protected staging file.
            master=(Path.home()/'Documents/fl-secrets/staging-api-key.txt').read_text().strip()
            r=http.post('/auth/session',json={'actor_key':master});assert r.status_code==401
            evidence['checks'].append('Shared staging master key refused')
    finally:
        if actors:
            with connect() as conn,conn.cursor() as cur:
                ids=[a['id'] for a in actors.values()]
                cur.execute('UPDATE actors SET active=false WHERE id=ANY(%s)',(ids,))
                cur.execute("UPDATE key_actor_sessions SET ended_at=clock_timestamp(),ended_reason='revoked' WHERE actor_id=ANY(%s) AND ended_at IS NULL",(ids,))
            evidence['temporary_actors_inactive']=True
    dest=Path(output) if output else ROOT/'docs/validation/key-sessions-staging.json';dest.write_text(json.dumps(evidence,indent=2,default=str)+'\n')
    print('Staging Arturo receive:',evidence['receipt_number'],'transaction',entry['id'],'10 lb; tokens/keys omitted.',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--migrate',action='store_true');parser.add_argument('--check',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.migrate:migrate()
    if args.check:check(args.output)
