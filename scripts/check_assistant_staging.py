#!/usr/bin/env python3
"""Explicit F1 migration or real hosted acceptance. Staging only; no main import.

Reads only the protected staging URI file. Creates synthetic acceptance stock
and a temporary named actor, deactivated in finally. Never prints credentials.
"""
import argparse
import io
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


def evidence_image(reference, kind, detail):
    """Readable synthetic evidence; bytes go only through the A6 upload API."""
    from PIL import Image, ImageDraw
    image = Image.new('RGB', (1000, 700), 'white')
    ImageDraw.Draw(image).multiline_text((40, 40),
        'STAGING TEST EVIDENCE\n\n' + kind.upper().replace('_', ' ') + '\n' + reference +
        '\n\n' + detail + '\n\nSynthetic acceptance evidence only', fill='black', spacing=18)
    stream = io.BytesIO()
    image.save(stream, format='PNG')
    return stream.getvalue()


def check(audio_file=None):
    token = 'STG-F1-' + uuid4().hex[:10].upper()
    keys = {role: secrets.token_urlsafe(40) for role in ('floor', 'owner', 'privacy_floor')}
    actors = {}
    evidence = {'reference': token, 'base_url': BASE, 'builder': 'Codex',
                'receipts': {}, 'attachments': {}, 'checks': []}
    ship_actions = ('ship_order', 'ship_standalone', 'ship_bulk')
    expected_actions = ('receive', 'make', 'pack', 'adjust', 'found') + ship_actions
    try:
        with connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SET LOCAL lock_timeout='5s'; SET LOCAL search_path=public")
            cur.execute("SELECT name FROM migration_markers WHERE name=ANY(%s)",
                        (['068_fl_assistant', '074_ship_gate_attachments'],))
            assert {r['name'] for r in cur.fetchall()} == {'068_fl_assistant', '074_ship_gate_attachments'}
            for role, key in keys.items():
                cur.execute('INSERT INTO actors(name,role,key_hash,active) VALUES (%s,%s,%s,true) RETURNING id',
                            (token + ' ' + role, 'floor' if role == 'privacy_floor' else role,
                             sha256(key.encode()).hexdigest()))
                actors[role] = cur.fetchone()['id']
            products = {}
            for kind in ('ingredient', 'batch', 'finished'):
                name = f'{token} {kind}'
                cur.execute("INSERT INTO products(name,odoo_code,type,uom,default_batch_lb,case_size_lb,active,is_service) VALUES (%s,%s,%s,'lb',10,5,true,false) RETURNING id,name",
                            (name, f'{token}-{kind}', kind))
                products[kind] = dict(cur.fetchone())
            cur.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s',
                        (products['batch']['id'], products['finished']['id']))
            cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,10)',
                        (products['batch']['id'], products['ingredient']['id']))
            supplier = token + ' supplier'
            cur.execute('INSERT INTO suppliers(name) VALUES (%s) RETURNING id', (supplier,))
            supplier_id = cur.fetchone()['id']
            customer = token + ' customer'
            cur.execute('INSERT INTO customers(name,active) VALUES (%s,true) RETURNING id', (customer,))
            customer_id = cur.fetchone()['id']
            assert min(customer_id, supplier_id, *(p['id'] for p in products.values())) >= 1_000_000_000
        with httpx.Client(base_url=BASE, headers={'X-API-Key': keys['floor']}, timeout=580) as http:
            # Existing actor-key cache can take 60 seconds to discover new actors.
            for role, key in keys.items():
                for _ in range(15):
                    who = http.get('/auth/whoami', headers={'X-API-Key': key})
                    if who.status_code == 200:
                        break
                    time.sleep(5)
                assert who.status_code == 200 and who.json()['actor']['id'] == actors[role]

            def post(path, role='floor', **kwargs):
                for attempt in range(3):
                    response = http.post(path, headers={'X-API-Key': keys[role]}, **kwargs)
                    if (path == '/assistant/turn' and response.status_code == 502
                            and response.json().get('detail', {}).get('error_code') == 'OPENAI_UNAVAILABLE'
                            and attempt < 2):
                        # Same turn ID/body: retry without another ledger draft.
                        print('Model temporarily unavailable; retrying the same turn.', flush=True)
                        time.sleep(5 * (attempt + 1))
                        continue
                    break
                assert response.status_code in (200, 202), f'{path} HTTP {response.status_code}'
                return response.json()

            def upload(sid, action, kind, detail, role):
                photo = evidence_image(token, kind, detail)
                attached = post('/assistant/attachment', role,
                    params={'session_id': sid}, data={'kind': kind},
                    files={'file': (action + '.png', photo, 'image/png')})
                aid = attached['attachment_id']
                assert type(aid) is int and attached['id'] == aid and attached['attachment_only']
                # F1 only links the session: the A6 API owns bytes, hash and ACLs.
                restored = http.post('/assistant/attachment/read', headers={'X-API-Key': keys[role]},
                                     json={'attachment_id': aid})
                assert restored.status_code == 200 and restored.content == photo
                owner_read = http.get(f'/attachments/{aid}/file', headers={'X-API-Key': keys['owner']})
                assert owner_read.status_code == 200 and sha256(owner_read.content).digest() == sha256(photo).digest()
                private = http.get(f'/attachments/{aid}/file', headers={'X-API-Key': keys['privacy_floor']})
                assert private.status_code == 403
                assert httpx.get(BASE + f'/attachments/{aid}/file', timeout=30).status_code == 401
                assert http.post('/assistant/attachment/read', headers={'X-API-Key': keys['privacy_floor']},
                                 json={'attachment_id': aid}).status_code == 404
                metadata = http.get(f'/attachments/{aid}', headers={'X-API-Key': keys[role]})
                assert metadata.status_code == 200
                row = metadata.json()
                assert row['kind'] == kind and row['uploaded_by_actor_id'] == actors[role]
                assert row['bytes'] == len(photo) and row['sha256'] == sha256(photo).hexdigest()
                evidence['attachments'][action] = {k: row[k] for k in
                    ('id', 'kind', 'sha256', 'bytes', 'content_type', 'uploaded_by_actor_id')}
                return aid

            def conversation(text, action, *, photo_kind=None, photo_detail='', role='floor'):
                sid = post('/assistant/session', role)['session_id']
                attachment_ids = [upload(sid, action, photo_kind, photo_detail or text, role)] if photo_kind else []
                body = {'session_id': sid, 'turn_id': str(uuid4()), 'text': text, 'attachment_ids': attachment_ids}
                result = post('/assistant/turn', role, json=body)
                assert post('/assistant/turn', role, json=body) == result
                for _ in range(10):
                    card = result['cards'][0]
                    if card['kind'] != 'choices':
                        break
                    candidates = card['result']['candidates']
                    exact = card['result'].get('query_normalized', '').lower()
                    selected = [c for c in candidates if c.get('name', '').lower() == exact]
                    # The operator selects its known fixture; no automatic model choice.
                    kind = card['resolution_kind']
                    if not selected and kind == 'product' and action != 'pack':
                        product_kind = 'finished' if action in ship_actions or action == 'found' else ('batch' if action == 'make' else 'ingredient')
                        selected = [c for c in candidates if c['id'] == products[product_kind]['id']]
                    if not selected and kind in ('supplier', 'customer', 'order'):
                        expected = {'supplier': supplier_id, 'customer': customer_id,
                                    'order': evidence.get('fixture_order', {}).get('order_id')}[kind]
                        selected = [c for c in candidates if c['id'] == expected]
                    if not selected and kind == 'lot':
                        codes = {token + '-LOT', token + '-FOUND'}
                        selected = [c for c in candidates if c.get('lot_code') in codes]
                    assert len(selected) == 1, 'Unexpected or ambiguous staging fixture choice'
                    result = post('/assistant/turn', role, json={'session_id': sid, 'turn_id': str(uuid4()),
                        'choice_id': card['id'], 'selected_id': selected[0]['id'], 'attachment_ids': attachment_ids})
                assert card['kind'] == 'draft' and card['prepared']['action'] == action
                if any(b['code'] == 'SKU_CONFIRMATION_REQUIRED' for b in card['prepared']['blockers']):
                    card = post('/assistant/confirm-sku', role, json={'draft_id': card['id']})['card']
                prepared = card['prepared']
                assert prepared['can_commit'] or (prepared['blockers'] and all(b['code'] == 'LOT_NOT_CONFIRMED' for b in prepared['blockers']))
                assert 'ticket' not in prepared and 'payload_hash' not in prepared
                if action in ('make', 'pack'):
                    assert any(w['code'] == 'WILL_CREATE_SHORTAGE' for w in prepared['warnings'])
                if action in ship_actions:
                    assert prepared['draft']['bol_reference'] == token + '-' + action.upper() + '-BOL'
                    assert prepared['draft']['photo'] == 'attached'
                    assert card['attachment_ids'] == attachment_ids
                if action == 'adjust':
                    held = post('/assistant/record', role, json={'draft_id': card['id']})
                    assert held['kind'] == 'awaiting_approval' and held['result']['held']
                    assert 'receipt_number' not in held['result']
                    assert post('/assistant/record', role, json={'draft_id': card['id']}) == held
                    resumed = post('/assistant/resume', role, json={'session_id': sid})
                    assert any(d['result'] == held['result'] for d in resumed['drafts'])
                    assert http.post('/assistant/cancel', headers={'X-API-Key': keys[role]},
                                     json={'draft_id': card['id']}).status_code == 409
                    with connect() as conn, conn.cursor() as cur:
                        cur.execute('SELECT count(*) FROM transactions WHERE ticket_id=%s', (prepared['ticket_id'],))
                        assert cur.fetchone()[0] == 0
                    approved = post(f"/exceptions/{held['result']['exception_id']}/approve", 'owner',
                        json={'note': 'Synthetic F1 acceptance: verified staged correction'})
                    evidence['held_correction'] = {'exception_id': held['result']['exception_id'], 'kind': held['kind'],
                        'no_post_before_approval': True, 'resume_verified': True, 'receipt_number': approved['receipt_number']}
                commit_body = {'draft_id': card['id'], 'acknowledged_warnings':
                               [w['code'] for w in prepared['warnings'] if w.get('requires_ack')]}
                if action in ('make', 'pack') + ship_actions:
                    # Human-only physical confirmation of each displayed fixture lot.
                    commit_body['lot_confirmations'] = [{'lot_id': i['lot_id'], 'method': 'full_code', 'value': i['lot_code']}
                                                         for i in prepared['draft']['input_plan']]
                with ThreadPoolExecutor(max_workers=2) as pool:
                    responses = [f.result() for f in [pool.submit(post, '/assistant/record', role, json=commit_body) for _ in range(2)]]
                assert all(r['kind'] == 'receipt' for r in responses)
                receipt = responses[0]['result']
                assert receipt['receipt_number'] == responses[1]['result']['receipt_number']
                resumed = post('/assistant/resume', role, json={'session_id': sid})
                assert any(d['result'] and d['result']['receipt_number'] == receipt['receipt_number'] for d in resumed['drafts'])
                # Signed packing-slip URLs are transient bearer capabilities; retain only the receipt data.
                evidence['receipts'][action] = {k: v for k, v in receipt.items() if k != 'packing_slip_url'}
                if action in ('make', 'pack'):
                    assert receipt['shortages'][0]['short_lb'] == 10
                if action == 'adjust':
                    assert receipt['receipt_number'] == evidence['held_correction']['receipt_number']
                if action in ship_actions:
                    assert receipt['proof_status'] == 'complete' and receipt['bol_reference'] == prepared['draft']['bol_reference']
                    if action == 'ship_order':
                        slip = httpx.get(BASE + receipt['packing_slip_url'], timeout=30)
                        assert slip.status_code == 200 and slip.content.startswith(b'%PDF')
                print(action + ': ' + receipt['receipt_number'], flush=True)
                return receipt, sid

            lot = token + '-LOT'
            conversation(f'Receive 10 cases of {products["ingredient"]["name"]}, 10 lb per case, from {supplier}. BOL {token}. Supplier lot {token}-SUP. Internal lot code {lot}.',
                         'receive', photo_kind='lot_tag')
            conversation(f'Make 11 batches of {products["batch"]["name"]}.', 'make')
            conversation(f'Pack 24 cases of {products["finished"]["name"]}, 5 lb per case, from {products["batch"]["name"]}.', 'pack')
            conversation(f'Adjust lot {lot} of {products["ingredient"]["name"]} down by 600 lb. Correction reason physical_count.', 'adjust')
            _, last_sid = conversation(f'Encontré 20 lb de {products["finished"]["name"]}. Motivo: physical_count. Lote nuevo {token}-FOUND.',
                                      'found', photo_kind='correction_photo')

            # A7 prepares/commits the synthetic fixture order; this setup receipt is
            # separate from the eight Assistant action receipts under acceptance.
            order = post('/sales/orders/prepare', 'owner', json={'customer_id': customer_id,
                'customer_po': token + '-PO', 'external_order_reference': token + '-ORDER',
                'lines': [{'product_id': products['finished']['id'], 'quantity_lb': 5, 'unit_price': 1}]})
            created = post('/tickets/' + order['ticket'] + '/commit', 'owner', json={'payload_hash': order['payload_hash'],
                'acknowledged_warnings': [w['code'] for w in order['warnings'] if w.get('requires_ack')]})
            evidence['fixture_order'] = {'order_id': created['order_id'], 'order_number': created['order_number'],
                                         'receipt_number': created['receipt_number']}
            for action in ship_actions:
                bol = token + '-' + action.upper() + '-BOL'
                if action == 'ship_order':
                    text = f'Ship all remaining quantities of sales order {created["order_number"]}. BOL {bol}.'
                else:
                    mode = 'standalone shipment' if action == 'ship_standalone' else 'bulk shipment with one resolved product line'
                    text = f'Record a {mode} of 5 lb of {products["finished"]["name"]} from lot {token}-FOUND to {customer}. Order reference {token}-{action.upper()}. BOL {bol}.'
                conversation(text, action, photo_kind='bol', role='owner',
                             photo_detail=f'BILL OF LADING\nBOL: {bol}\nCustomer: {customer}\nProduct: {products["finished"]["name"]}\nQuantity: 5 lb\nCarrier: STAGING TEST TRUCK')
            for query in ('What did I enter today?', f'Look up inventory for {products["ingredient"]["name"]}'):
                result = post('/assistant/turn', json={'session_id': last_sid, 'turn_id': str(uuid4()), 'text': query})
                assert result['cards'][0]['kind'] == 'read'
            evidence['checks'] += ['eight real Responses function-tool drafts and commits',
                'concurrent same-ticket replay for all eight actions', 'turn idempotency',
                'A6 private authenticated photo upload/download with floor isolation and anonymous refusal',
                'A6 integer attachment IDs in Assistant custody; no independent F1 byte storage',
                'English/Spanish', 'today receipts', 'inventory read', 'saved draft/receipt recovery',
                'A5 explicit operator lot confirmations for make, pack and all ship actions',
                'three BOL-bearing photo-complete shipments', 'signed order packing-slip PDF']
            if audio_file:
                transcript = post('/assistant/transcribe', files={'file': ('dictation.wav', Path(audio_file).read_bytes(), 'audio/wav')})
                assert transcript['editable'] and transcript['sent'] is False
                assert 'two' in transcript['text'].lower() or '2' in transcript['text']
                evidence['transcript'] = transcript
                evidence['checks'].append('real OpenAI transcription, editable and unsent')
            assert http.get('/dash/fl-assistant').status_code == 200
        with connect() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('BEGIN TRANSACTION READ ONLY')
            cur.execute("SELECT t.id,t.action,t.actor_id,t.receipt_number,t.client_source,count(x.id) AS ledger_posts FROM write_tickets t JOIN transactions x ON x.ticket_id=t.id WHERE t.actor_id=ANY(%s) AND t.client_source='fl_assistant' GROUP BY t.id ORDER BY t.id",
                        (list(actors.values()),))
            rows = [dict(r) for r in cur.fetchall()]
            assert len(rows) == 8 and {r['action'] for r in rows} == set(expected_actions)
            assert all(r['ledger_posts'] == 1 for r in rows)
            assert all(r['actor_id'] == actors['owner' if r['action'] in ship_actions else 'floor'] for r in rows)
            evidence['verified_tickets'] = rows
            ticket_ids = [r['id'] for r in rows]
            cur.execute('SELECT count(*) AS n FROM transactions x JOIN write_tickets t ON t.id=x.ticket_id WHERE t.id=ANY(%s) AND x.entered_by_actor_id=t.actor_id', (ticket_ids,))
            assert cur.fetchone()['n'] == 8
            cur.execute('SELECT t.action,count(c.transaction_id) AS confirmations FROM transaction_lot_confirmations c JOIN transactions x ON x.id=c.transaction_id JOIN write_tickets t ON t.id=x.ticket_id WHERE t.id=ANY(%s) GROUP BY t.action', (ticket_ids,))
            confirmations = {r['action']: r['confirmations'] for r in cur.fetchall()}
            assert confirmations == {action: 1 for action in ('make', 'pack') + ship_actions}
            evidence['persisted_lot_confirmations'] = confirmations
            cur.execute('SELECT s.id,s.ticket_id,s.bol_reference,s.proof_status,s.entered_by_actor_id,t.action FROM shipments s JOIN write_tickets t ON t.id=s.ticket_id WHERE t.id=ANY(%s) ORDER BY s.id', (ticket_ids,))
            shipments = [dict(r) for r in cur.fetchall()]
            assert len(shipments) == 3 and {s['action'] for s in shipments} == set(ship_actions)
            assert all(s['proof_status'] == 'complete' and s['entered_by_actor_id'] == actors['owner']
                       and s['bol_reference'] == token + '-' + s['action'].upper() + '-BOL' for s in shipments)
            evidence['verified_shipments'] = shipments
            for s in shipments:
                cur.execute('SELECT id,kind,ticket_id,shipment_id,uploaded_by_actor_id FROM attachments WHERE id=%s',
                            (evidence['attachments'][s['action']]['id'],))
                attachment = dict(cur.fetchone())
                assert attachment['kind'] == 'bol' and attachment['ticket_id'] == s['ticket_id']
                assert attachment['shipment_id'] == s['id'] and attachment['uploaded_by_actor_id'] == actors['owner']
                evidence['attachments'][s['action']].update(attachment)
            found = evidence['receipts']['found']
            cur.execute('SELECT id,kind,ticket_id,transaction_id,uploaded_by_actor_id FROM attachments WHERE id=%s',
                        (evidence['attachments']['found']['id'],))
            correction = dict(cur.fetchone())
            assert correction['kind'] == 'correction_photo' and correction['ticket_id'] == found['ticket_id']
            assert correction['transaction_id'] == found['transaction_id'] and correction['uploaded_by_actor_id'] == actors['floor']
            evidence['attachments']['found'].update(correction)
            evidence['checks'] += ['A2 entered-by actor on all eight posts', 'A5 persisted lot evidence',
                'A6 single shipment per ship ticket; BOL/photo ownership and ticket/shipment links',
                'A6 correction photo claimed by trusted ID and linked to found transaction',
                'A3b make/pack shortage warnings and flagged receipts',
                'A3b held correction: replay, resume, no Cancel, owner approval then one post']
        evidence['actor_id'] = actors['floor']
        evidence['owner_id'] = actors['owner']
        evidence['privacy_actor_id'] = actors['privacy_floor']
        evidence['synthetic_products'] = products
        evidence['synthetic_customer_id'] = customer_id
        return evidence
    finally:
        if actors:
            with connect() as conn, conn.cursor() as cur:
                cur.execute('UPDATE actors SET active=false WHERE id=ANY(%s)', (list(actors.values()),))
                cur.execute('SELECT count(*) FROM actors WHERE id=ANY(%s) AND active', (list(actors.values()),))
                assert cur.fetchone()[0] == 0
            evidence['temporary_actors_inactive'] = True
            print('Temporary staging actors deactivated.', flush=True)


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
        # Never print exception text, HTTP bodies, credentials or locals.
        print('Staging check failed: ' + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
