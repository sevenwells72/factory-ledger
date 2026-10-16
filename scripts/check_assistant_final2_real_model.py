#!/usr/bin/env python3
"""Final2 re-check acceptance: actual provider, local staging copy, explicit test-worker Record.

No hosted DB connection, no model replacement, no automatic app lot confirmation.
The simulated worker deliberately supplies each tag and retains wrong-tag refusal.
Full traces remain private; the shareable report keeps exact inputs and provider IDs.
"""
import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from hashlib import sha256
import base64
import io
import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sessions', type=int, choices=(0, 1), required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--only', default='')
    args = parser.parse_args()
    url = urlsplit(args.database_url)
    assert url.hostname == '127.0.0.1' and re.fullmatch(r'/fl_final2_[a-z0-9_]+', url.path)
    assert not url.query and not url.fragment and args.repeats >= 3
    os.environ.update(DATABASE_URL=args.database_url, ASSISTANT_ENABLED='1',
                      API_KEY=uuid4().hex, DASHBOARD_API_KEY=uuid4().hex,
                      KEY_SESSIONS_ENABLED=str(args.sessions),
                      EVIDENCE_STORAGE_DIR='/private/tmp/fl_final2/photos')
    for key in ('PGOPTIONS', 'PGSERVICE', 'PGSERVICEFILE', 'PGHOSTADDR'):
        os.environ.pop(key, None)
    os.environ['OPENAI_API_KEY'] = (Path.home() / 'Documents/fl-secrets/openai-api-key.txt').read_text().strip()
    sys.path.insert(0, str(ROOT))
    logging.disable(logging.CRITICAL)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        import main as api
        import fl_assistant
    import psycopg2
    from psycopg2.extras import RealDictCursor
    from fastapi.testclient import TestClient

    @contextmanager
    def connection():
        conn = psycopg2.connect(args.database_url)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    api.get_db_connection = connection
    api._reset_actor_cache()
    actor_key = 'fl-final-owner-' + uuid4().hex
    with connection() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES(%s,'owner',%s,true)",
                    ('Final local validation ' + uuid4().hex[:8], sha256(actor_key.encode()).hexdigest()))
    client = TestClient(api.app)
    headers = {'X-API-Key': actor_key}
    if args.sessions:
        r = client.post('/auth/session', json={'actor_key': actor_key})
        assert r.status_code == 200, r.text
        headers = {'X-API-Key': r.json()['session_token']}
    owner_headers = headers
    floor_key = 'fl-final2-floor-' + uuid4().hex
    with connection() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES(%s,'floor',%s,true)",
                    ('Final2 local floor ' + uuid4().hex[:8], sha256(floor_key.encode()).hexdigest()))
    floor_headers = {'X-API-Key': floor_key}
    if args.sessions:
        r = client.post('/auth/session', json={'actor_key': floor_key})
        assert r.status_code == 200, r.text
        floor_headers = {'X-API-Key': r.json()['session_token']}
    original = fl_assistant.openai
    events = []

    async def observe(path, *, body=None, file=None):
        event = {'request_model': (body or {}).get('model'), 'store': (body or {}).get('store')}
        events.append(event)
        result = await original(path, body=body, file=file)
        event.update(response_id=result.get('id'), model=result.get('model'), status=result.get('status'),
                     tools=[x.get('name') for x in result.get('output', []) if x.get('type') == 'function_call'])
        return result

    fl_assistant.openai = observe
    cases = [
        ('sunshine_en', 'en', 'Add Sunshine Granola as a new customer', 'customer'),
        ('sunshine_es', 'es', 'Agrega Sunshine Granola como cliente nuevo', 'customer'),
        ('semantic_en', 'en', 'We need Sunshine Granola in our customer list.', 'customer'),
        ('semantic_es', 'es', 'Necesitamos Sunshine Granola en nuestra lista de clientes.', 'customer'),
        ('setton_en', 'en', 'Add Setton', 'customer'),
        ('setton_es', 'es', 'Agrega Setton', 'customer'),
        ('setton_new_en', 'en', 'Add Setton as a new customer', 'customer'),
        ('setton_new_es', 'es', 'Agrega Setton como cliente nuevo', 'customer'),
        ('expected_en', 'en', 'Grain Supply is sending 20 cases of Oats next Tuesday, PO C4F14D. Add it as an expected delivery.', 'handoff'),
        ('expected_es', 'es', 'Grain Supply va a mandar 20 cajas de avena el próximo martes, PO C4F14D. Agrégalo como entrega esperada.', 'handoff'),
        ('bol_en', 'en', 'We received 2 cases of Oats from LaCrosse Milling, 50 lb per case, BOL PH1777, supplier lot LM-PH1777. Photo attached.', 'photo'),
        ('bol_es', 'es', 'Recibimos 2 cajas de avena de LaCrosse Milling, 50 lb por caja, BOL PH1777, lote del proveedor LM-PH1777. Foto adjunta.', 'photo'),
        ('both_en', 'en', 'We received 2 cases of Oats from LaCrosse Milling, 50 lb per case, BOL PH1777, supplier lot LM-PH1777. Photos attached.', 'photo'),
        ('both_es', 'es', 'Recibimos 2 cajas de avena de LaCrosse Milling, 50 lb por caja, BOL PH1777, lote del proveedor LM-PH1777. Fotos adjuntas.', 'photo'),
    ]
    if args.only:
        selected = set(args.only.split(','))
        cases = [c for c in cases if c[0] in selected]
        assert len(cases) == len(selected)
    sources = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    results = []
    full = []
    report = {'sessions': args.sessions, 'database_local_only': True, 'real_model': True,
              'repeats': args.repeats, 'source_sha256': sources, 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    private = args.output.with_suffix('.private.json')

    def save():
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        private.write_text(json.dumps(full, indent=2, ensure_ascii=False, default=str) + '\n')
        private.chmod(0o600)

    def send(sid, language, text='', **extra):
        response = client.post('/assistant/turn', headers=headers, json={
            'session_id': sid, 'turn_id': str(uuid4()), 'text': text, 'language': language, **extra})
        data = response.json()
        steps.append({'text': text, 'status': response.status_code, 'result': data})
        assert response.status_code == 200, json.dumps(data)
        assert not any(c.get('error_code') == 'ASSISTANT_QUOTA_EXHAUSTED' for c in data.get('cards', [])), 'PROVIDER_QUOTA'
        return data

    def cards(data, kind):
        return [c for c in data.get('cards', []) if c.get('kind') == kind]

    def follow(sid, language, data, purpose):
        for _ in range(6):
            if cards(data, 'draft'):
                return data
            choices = cards(data, 'choices')
            if choices:
                choice = choices[0]
                opts = choice.get('result', {}).get('candidates', [])
                targets = ('LaCrosse Milling',) if purpose == 'supplier' and choice.get('resolution_kind') == 'supplier' else (
                    'LaCrosse Milling', 'Grain Supply', 'Oats', 'Honey', 'Sunflower Seeds', 'Batch Classic Granola #9')
                opt = next((c for t in targets for c in opts if c.get('name') == t), None)
                assert opt, 'No exact human-selected catalog choice: ' + json.dumps(choice)
                data = send(sid, language, choice_id=choice.get('id') or choice['choice_id'], selected_id=opt['id'])
                continue
            reason = next((c for c in data.get('cards', []) if c.get('reason_required')), None)
            if reason:
                data = send(sid, language, 'El motivo de corrección es physical_count.' if language == 'es' else 'The correction reason is physical_count.')
                continue
            missing = cards(data, 'missing_facts')
            if missing:
                fields = missing[0].get('fields', [])
                if set(fields) <= {'supplier_id', 'bol_reference'}:
                    data = send(sid, language, 'De Grain Supply, BOL 5521.' if language == 'es' else 'From Grain Supply, BOL 5521.')
                    continue
            return data
        raise AssertionError('Conversation did not converge')

    for label, language, sentence, purpose in cases:
        headers = floor_headers if purpose == "photo" else owner_headers
        for repeat in range(1, args.repeats + 1):
            events.clear(); steps = []; started = time.perf_counter()
            row = {'case': label, 'language': language, 'text': sentence, 'run': repeat}
            try:
                r = client.post('/assistant/session', headers=headers)
                assert r.status_code == 200, r.text
                sid = r.json()['session_id']
                extra = {}; attachment_ids = []
                if purpose == 'photo':
                    png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+iDbcAAAAASUVORK5CYII=')
                    for kind in (['bol', 'lot_tag'] if label.startswith('both') else ['bol']):
                        upload = client.post('/assistant/attachment', params={'session_id': sid}, headers=headers,
                            data={'kind': kind}, files={'file': ('local-' + kind + '.png', png, 'image/png')})
                        assert upload.status_code == 200, upload.text
                        attachment_ids.append(upload.json()['id'])
                    extra['attachment_ids'] = attachment_ids
                data = send(sid, language, sentence, **extra)
                if purpose in ('customer', 'sku', 'handoff', 'shipped', 'refusal', 'shortage'):
                    text = json.dumps(data, ensure_ascii=False)
                    if purpose == 'customer':
                        assert ('Sunshine Granola' if label.startswith(('sunshine', 'semantic')) else 'Setton Farms') in text, text
                        assert cards(data, 'choices') or cards(data, 'existing_customer') or (cards(data, 'draft') and not cards(data, 'draft')[0]['prepared']['can_commit']), text
                        choices = cards(data, 'choices')
                        if choices:
                            before = len(events)
                            choice = choices[0]
                            target = 'Sunshine Granola' if label.startswith(('sunshine', 'semantic')) else 'Setton Farms'
                            selected = next(c for c in choice['result']['candidates'] if c['name'] == target)
                            chosen = send(sid, language, choice_id=choice['id'], selected_id=selected['id'])
                            assert cards(chosen, 'handoff') and len(events) == before, chosen
                            row['selection_finished_without_model'] = True
                    elif purpose == 'sku':
                        assert cards(data, 'read'), text
                        assert ('Classic' if label == 'sku_283' else 'Granola SS Classic #9 Bulk') in text, text
                    elif purpose == 'handoff':
                        assert cards(data, 'handoff') and '/office' in text and 'expected' in text, text
                        if language == 'es': assert 'entrega esperada' in text, text
                        assert not events, 'Office handoff must not wait for a model call'
                    elif purpose == 'shipped':
                        assert ('ya está completamente enviado' if language == 'es' else 'already fully shipped') in text, text
                    elif purpose == 'shortage':
                        data = follow(sid, language, data, purpose)
                        text = json.dumps(data, ensure_ascii=False)
                        assert 'INSUFFICIENT_STOCK' in text and 'Kookies' in text, text
                        assert ('Solo hay 5 cajas' if language == 'es' else 'Only 5 cases') in text, text
                        assert ('necesita 20' if language == 'es' else 'needs 20') in text, text
                        assert cards(data, 'refusal')[0].get('actions'), text
                    else:
                        data = follow(sid, language, data, purpose)
                        text = json.dumps(data, ensure_ascii=False)
                        assert cards(data, 'refusal') or (cards(data, 'draft') and not cards(data, 'draft')[0]['prepared']['can_commit']), text
                        assert 'Revise los datos de este borrador' not in text, text
                else:
                    if purpose == 'multi':
                        assert cards(data, 'receive_queue'), json.dumps(data)
                    data = follow(sid, language, data, purpose)
                    draft = cards(data, 'draft')[0]
                    d = draft['prepared']['draft']
                    row['draft_action'] = draft['prepared']['action']
                    assert purpose == 'photo', purpose
                    assert d['supplier_lot_code'] == 'LM-PH1777' and d['bol_reference'] == 'PH1777', d
                    assert set(a['id'] for a in d['attachments']) == set(attachment_ids), d
                    recorded = client.post('/assistant/record', headers=headers, json={
                        'draft_id': draft['id'], 'acknowledged_warnings': [w['code'] for w in draft['prepared'].get('warnings', [])]})
                    assert recorded.status_code == 200, recorded.text
                    result = recorded.json()['result']; row['receipt'] = result['receipt_number']
                    assert set(a['id'] for a in result['attachments']) == set(attachment_ids), result
                    detail_response = client.get('/receipts/' + result['receipt_number'], headers=headers)
                    assert detail_response.status_code == 200, detail_response.text
                    detail = detail_response.json()
                    assert set(a['id'] for a in detail['attachments']) == set(attachment_ids), detail
                    assert all(a['transaction_id'] == result['transaction_id'] for a in detail['attachments']), detail
                    lookup = send(sid, language, ('Muestra el recibo ' if language == 'es' else 'Show receipt ') + result['receipt_number'])
                    read = cards(lookup, 'read'); assert read, lookup
                    assert set(a['id'] for a in read[0]['result']['attachments']) == set(attachment_ids), lookup
                    with connection() as conn, conn.cursor() as cur:
                        cur.execute('SELECT id, ticket_id, transaction_id FROM attachments WHERE id=ANY(%s)', (attachment_ids,))
                        linked=cur.fetchall()
                        assert len(linked)==len(attachment_ids) and all(r[1]==draft['prepared']['ticket_id'] and r[2]==result['transaction_id'] for r in linked), linked
                    replay=client.post('/assistant/record', headers=headers, json={'draft_id':draft['id']})
                    assert replay.status_code==200 and replay.json()['result']['receipt_number']==result['receipt_number'], replay.text
                    row.update(photo_kinds=[a['kind'] for a in detail['attachments']], draft_and_receipt_and_lookup_match=True,
                               attachment_ticket_transaction_linked=True, replay_same_receipt=True)
                assert purpose == 'handoff' or (events and all(e.get('status') == 'completed' and e.get('response_id') for e in events)), events
                row['status'] = 'passed'
            except Exception as error:
                row['status'] = 'failed'; row['error'] = str(error)[:1500]
            row['seconds'] = round(time.perf_counter() - started, 3)
            row['provider'] = list(events)
            results.append(row); full.append({**row, 'steps': steps}); save()
            print(f"{label} {repeat}/{args.repeats}: {row['status']} ({row['seconds']}s)", flush=True)
            if 'PROVIDER_QUOTA' in row.get('error', ''):
                raise SystemExit('Provider quota exhausted: remaining real-model cases not run')
    report['source_unchanged'] = all(sha256((ROOT / p).read_bytes()).hexdigest() == digest for p, digest in sources.items())
    report['passed'] = sum(r['status'] == 'passed' for r in results)
    report['failed'] = len(results) - report['passed']; save()
    raise SystemExit(bool(report['failed']))


if __name__ == '__main__':
    main()
