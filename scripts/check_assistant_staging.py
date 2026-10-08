#!/usr/bin/env python3
"""Explicit F1 migration or real hosted acceptance. Staging only; no main import.

Reads only the protected staging URI file. Creates synthetic acceptance stock
and a temporary named actor, deactivated in finally. Never prints credentials.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
import json
from pathlib import Path
import secrets
import sys
import time
from uuid import uuid4

import httpx
import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from staging_safety import assert_staging_database, PRODUCTION_DATABASE_HOST

BASE = 'https://fastapi-staging-production-dd7b.up.railway.app'


def connect():
    file = Path.home() / 'Documents/fl-secrets/staging-db-url.txt'
    if file.stat().st_mode & 0o077:
        raise RuntimeError('Staging URI file must be private')
    uri = file.read_text().strip()
    assert_staging_database(uri, 'staging', PRODUCTION_DATABASE_HOST)
    dsn = psycopg2.extensions.parse_dsn(uri)
    assert dsn['host'] == 'aws-0-us-east-1.pooler.supabase.com'
    assert 'jygmyvxnxdjiiilhxseq' in dsn.get('user', '')
    dsn['port'] = '5432'
    return psycopg2.connect(**dsn, connect_timeout=15)


def migrate():
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SET LOCAL lock_timeout='5s'; SET LOCAL statement_timeout='60s'; SET LOCAL search_path=public")
        cur.execute((ROOT / 'migrations/068_fl_assistant.sql').read_text())
        cur.execute("SELECT name FROM migration_markers WHERE name='068_fl_assistant'")
        assert cur.fetchone()[0] == '068_fl_assistant'
    print('Migration 068 applied to verified STAGING only.', flush=True)


def check(audio_file=None):
    token = 'STG-F1-' + uuid4().hex[:10].upper()
    raw_key = secrets.token_urlsafe(40)
    actor_id = None
    evidence = {'reference': token, 'base_url': BASE, 'builder': 'Codex', 'receipts': {}, 'checks': []}
    try:
        with connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SET LOCAL lock_timeout='5s'; SET LOCAL search_path=public")
            cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES (%s,'floor',%s,true) RETURNING id", (token, sha256(raw_key.encode()).hexdigest()))
            actor_id = cur.fetchone()['id']
            products = {}
            for kind in ('ingredient', 'batch', 'finished'):
                name = f'{token} {kind}'
                cur.execute("INSERT INTO products(name,odoo_code,type,uom,default_batch_lb,case_size_lb,active) VALUES (%s,%s,%s,'lb',10,5,true) RETURNING id,name", (name, f'{token}-{kind}', kind))
                products[kind] = dict(cur.fetchone())
            cur.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s', (products['batch']['id'], products['finished']['id']))
            cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,10)', (products['batch']['id'], products['ingredient']['id']))
            supplier = token + ' supplier'
            cur.execute('INSERT INTO suppliers(name) VALUES (%s) RETURNING id', (supplier,))
            supplier_id = cur.fetchone()['id']
        with httpx.Client(base_url=BASE, headers={'X-API-Key': raw_key}, timeout=580) as http:
            # Existing actor-key cache can take 60 seconds to discover a new actor.
            for _ in range(15):
                who = http.get('/auth/whoami')
                if who.status_code == 200:
                    break
                time.sleep(5)
            assert who.status_code == 200, 'Staging actor authentication failed'
            def post(path, **kwargs):
                response = http.post(path, **kwargs)
                assert response.status_code == 200, f'{path} HTTP {response.status_code}: {response.json().get("detail", response.json().get("result", {}))}'
                return response.json()
            def conversation(text, action, attachments=None):
                sid = post('/assistant/session')['session_id']
                if attachments is not None:
                    photo = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl6UAAAAABJRU5ErkJggg==')
                    attached = post('/assistant/attachment', params={'session_id': sid}, files={'file': ('tag.png', photo, 'image/png')})
                    attachments.append(attached['id'])
                    assert attached['attachment_only']
                body = {'session_id': sid, 'turn_id': str(uuid4()), 'text': text, 'attachment_ids': attachments or []}
                result = post('/assistant/turn', json=body)
                # An HTTP retry must return the same card/draft without another model call.
                assert post('/assistant/turn', json=body) == result
                for _ in range(6):
                    card = result['cards'][0]
                    if card['kind'] != 'choices':
                        break
                    candidates = card['result']['candidates']
                    exact = card['result'].get('query_normalized', '').lower()
                    selected = [c for c in candidates if c.get('name', '').lower() == exact]
                    # The test operator knows its synthetic product and presses
                    # that actual FL choice. The application/model never picks.
                    if not selected and card['resolution_kind'] == 'product' and action != 'pack':
                        expected = products['batch' if action == 'make' else 'ingredient']['id']
                        selected = [c for c in candidates if c['id'] == expected]
                    if not selected and card['resolution_kind'] == 'supplier':
                        selected = [c for c in candidates if c['id'] == supplier_id]
                    assert len(selected) == 1, f'Unexpected choices: {card}'
                    result = post('/assistant/turn', json={'session_id': sid, 'turn_id': str(uuid4()), 'choice_id': card['id'], 'selected_id': selected[0]['id'], 'attachment_ids': attachments or []})
                assert card['kind'] == 'draft', f'Expected {action} draft: {card}'
                assert card['prepared']['action'] == action
                if any(b['code'] == 'SKU_CONFIRMATION_REQUIRED' for b in card['prepared']['blockers']):
                    card = post('/assistant/confirm-sku', json={'draft_id': card['id']})['card']
                assert card['prepared']['can_commit'], card['prepared']['blockers']
                assert 'ticket' not in card['prepared'] and 'payload_hash' not in card['prepared']
                commit_body = {'draft_id': card['id'], 'acknowledged_warnings': [w['code'] for w in card['prepared']['warnings'] if w.get('requires_ack')]}
                with ThreadPoolExecutor(max_workers=2) as pool:
                    responses = [f.result() for f in [pool.submit(post, '/assistant/record', json=commit_body) for _ in range(2)]]
                assert all(r['kind'] == 'receipt' for r in responses)
                assert responses[0]['result']['receipt_number'] == responses[1]['result']['receipt_number']
                resumed = post('/assistant/resume', json={'session_id': sid})
                assert any(d['result'] and d['result']['receipt_number'] == responses[0]['result']['receipt_number'] for d in resumed['drafts'])
                evidence['receipts'][action] = responses[0]['result']
                print(action + ': ' + responses[0]['result']['receipt_number'], flush=True)
                return responses[0]['result'], sid

            lot = token + '-LOT'
            receive, _ = conversation(f'Receive 10 cases of {products["ingredient"]["name"]}, 10 lb per case, from {supplier}. BOL {token}. Supplier lot {token}-SUP. Internal lot code {lot}.', 'receive', [])
            conversation(f'Make 1 batch of {products["batch"]["name"]}.', 'make')
            conversation(f'Pack 2 cases of {products["finished"]["name"]}, 5 lb per case, from {products["batch"]["name"]}.', 'pack')
            conversation(f'Adjust lot {lot} of {products["ingredient"]["name"]} down by 2 lb. Correction reason physical_count.', 'adjust')
            _, last_sid = conversation(f'Encontré 2 lb de {products["ingredient"]["name"]}. Motivo: physical_count. Lote nuevo {token}-FOUND.', 'found')
            for query in ('What did I enter today?', f'Look up inventory for {products["ingredient"]["name"]}'):
                result = post('/assistant/turn', json={'session_id': last_sid, 'turn_id': str(uuid4()), 'text': query})
                assert result['cards'][0]['kind'] == 'read', result
            evidence['checks'] += ['five real Responses function-tool drafts and commits', 'concurrent same-ticket replay for all five actions', 'turn idempotency', 'private photo attachment', 'English/Spanish', 'today receipts', 'inventory read', 'saved draft/receipt recovery']
            if audio_file:
                transcript = post('/assistant/transcribe', files={'file': ('dictation.wav', Path(audio_file).read_bytes(), 'audio/wav')})
                assert transcript['editable'] and transcript['sent'] is False
                assert 'two' in transcript['text'].lower() or '2' in transcript['text']
                evidence['transcript'] = transcript
                evidence['checks'].append('real OpenAI transcription, editable and unsent')
            assert http.get('/dash/fl-assistant').status_code == 200
        with connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('BEGIN TRANSACTION READ ONLY')
            cur.execute("SELECT t.id,t.receipt_number,t.client_source,count(x.id) AS ledger_posts FROM write_tickets t JOIN transactions x ON x.ticket_id=t.id WHERE t.actor_id=%s GROUP BY t.id ORDER BY t.id", (actor_id,))
            rows = [dict(r) for r in cur.fetchall()]
            assert len(rows) == 5 and all(r['ledger_posts'] == 1 and r['client_source'] == 'fl_assistant' for r in rows)
            evidence['verified_tickets'] = rows
        evidence['actor_id'] = actor_id
        evidence['synthetic_products'] = products
        return evidence
    finally:
        if actor_id:
            with connect() as conn, conn.cursor() as cur:
                cur.execute('UPDATE actors SET active=false WHERE id=%s', (actor_id,))
            print('Temporary staging actor deactivated.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply-migration', action='store_true')
    parser.add_argument('--audio-file')
    parser.add_argument('--output', default='/tmp/fl-f1-staging.json')
    args = parser.parse_args()
    try:
        if args.apply_migration:
            migrate()
        else:
            result = check(args.audio_file)
            Path(args.output).write_text(json.dumps(result, indent=2, default=str) + '\n')
            print('Staging acceptance passed. Evidence: ' + args.output)
    except Exception as exc:
        # Connection exceptions can contain URI credentials. Only controlled
        # assertion text (HTTP response data) is printable; no traceback/locals.
        print('Staging check failed: ' + type(exc).__name__ + (': ' + str(exc) if isinstance(exc, AssertionError) else ''), file=sys.stderr)
        raise SystemExit(1)
