#!/usr/bin/env python3
"""Final repair acceptance: actual provider, local staging copy, explicit test-worker Record.

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
    assert url.hostname == '127.0.0.1' and re.fullmatch(r'/fl_final_[a-z0-9_]+', url.path)
    assert not url.query and not url.fragment and args.repeats >= 3
    os.environ.update(DATABASE_URL=args.database_url, ASSISTANT_ENABLED='1',
                      API_KEY=uuid4().hex, DASHBOARD_API_KEY=uuid4().hex,
                      KEY_SESSIONS_ENABLED=str(args.sessions),
                      EVIDENCE_STORAGE_DIR='/private/tmp/fl_final/photos')
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
        ('found_en', 'en', 'We found 2 cases of Honey, 5 lb each, during the physical count.', 'found'),
        ('found_es', 'es', 'Encontramos 2 cajas de miel, 5 lb cada una, en el conteo físico', 'found'),
        ('found_reason_es', 'es', 'Encontramos 2 cajas de miel, 5 lb cada una.', 'found'),
        ('receive_en', 'en', 'We received 6 cases of Sunflower Seeds from Grain Supply, 50 lb per case, BOL FS8996, supplier lot SL-64AED4', 'receive'),
        ('receive_es', 'es', 'Recibimos 6 cajas de semillas de girasol de Grain Supply, 50 lb por caja, BOL FS8996, lote del proveedor SL-64AED4', 'receive'),
        ('bags_es', 'es', 'Recibimos 10 bolsas de avena de 50 libras de Grain Supply, BOL 5521, lote del proveedor SL-4471.', 'bags'),
        ('bags_each_en', 'en', 'We received 10 bags of oats, 50 lb each, from Grain Supply, BOL 5521, supplier lot SL-4471.', 'bags'),
        ('bags_en', 'en', 'We received 10 bags of oats, 50 lb bags, from Grain Supply, BOL 5521, supplier lot SL-4471.', 'bags'),
        ('multi_en', 'en', 'We received 5 cases of Sunflower Seeds and 3 cases of Oats from Grain Supply, 50 lb per case, BOL MULTI-5521.', 'multi'),
        ('multi_es', 'es', 'Recibimos 5 cajas de semillas de girasol y 3 cajas de avena de Grain Supply, 50 lb por caja, BOL MULTI-5521.', 'multi'),
        ('setton_en', 'en', 'Add Setton as a new customer', 'customer'),
        ('setton_es', 'es', 'Agrega Setton como cliente nuevo', 'customer'),
        ('sku_283', 'en', 'how much two eighty three do we have', 'sku'),
        ('sku_70013', 'en', 'how much seven zero zero one three do we have', 'sku'),
        ('supplier_voice', 'en', 'we received four cases of oats from la cross milling fifty pounds a case B O L five five two one', 'supplier'),
        ('supplier_es', 'es', 'Recibimos 4 cajas de avena de la cross milling, 50 lb por caja, BOL 5521.', 'supplier'),
        ('make_en', 'en', 'We made 1 batch of Classic Granola #9', 'make'),
        ('make_es', 'es', 'Hicimos 1 lote de Classic Granola #9', 'make'),
        ('kosher_en', 'en', 'We received 2 cases of product 283 from Grain Supply, 25 lb per case, BOL 5521.', 'refusal'),
        ('kosher_es', 'es', 'Recibimos 2 cajas del producto 283 de Grain Supply, 25 lb por caja, BOL 5521.', 'refusal'),
        ('change_en', 'en', 'Change order SO-261015-001 to 3 cases', 'handoff'),
        ('change_es', 'es', 'Cambia el pedido SO-261015-001 a 3 cajas', 'handoff'),
        ('cancel_en', 'en', 'Cancel order SO-261015-001', 'handoff'),
        ('cancel_es', 'es', 'Cancela el pedido SO-261015-001', 'handoff'),
        ('expected_en', 'en', 'Grain Supply is sending 20 cases of Oats next Tuesday, PO FINAL-1. Add it as an expected delivery.', 'handoff'),
        ('expected_es', 'es', 'Grain Supply enviará 20 cajas de avena el próximo martes, PO FINAL-1. Agrégalo como entrega esperada.', 'handoff'),
        ('shipped_en', 'en', 'Ship SO-261012-001', 'shipped'),
        ('shipped_es', 'es', 'Envía el pedido SO-261012-001', 'shipped'),
        ('shortage_en', 'en', 'Ship all remaining on order SO-261016-047, BOL FINAL-SHIP.', 'shortage'),
        ('shortage_es', 'es', 'Envía todo lo pendiente del pedido SO-261016-047, BOL FINAL-SHIP.', 'shortage'),
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
                    'Grain Supply', 'Oats', 'Honey', 'Sunflower Seeds', 'Batch Classic Granola #9')
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
        for repeat in range(1, args.repeats + 1):
            events.clear(); steps = []; started = time.perf_counter()
            row = {'case': label, 'language': language, 'text': sentence, 'run': repeat}
            try:
                r = client.post('/assistant/session', headers=headers)
                assert r.status_code == 200, r.text
                sid = r.json()['session_id']
                data = send(sid, language, sentence)
                if purpose in ('customer', 'sku', 'handoff', 'shipped', 'refusal', 'shortage'):
                    text = json.dumps(data, ensure_ascii=False)
                    if purpose == 'customer':
                        assert 'Setton Farms' in text and not cards(data, 'draft'), text
                        assert cards(data, 'choices') or cards(data, 'existing_customer'), text
                    elif purpose == 'sku':
                        assert cards(data, 'read'), text
                        assert ('Classic' if label == 'sku_283' else 'Granola SS Classic #9 Bulk') in text, text
                    elif purpose == 'handoff':
                        assert cards(data, 'handoff') and '/office' in text, text
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
                    if purpose == 'bags':
                        assert d['cases'] == 10 and d['case_size_lb'] == 50 and d['total_lb'] == 500, d
                    elif purpose == 'receive':
                        assert d['supplier_lot_code'] == 'SL-64AED4' and d['bol_reference'] == 'FS8996', d
                    elif purpose == 'supplier':
                        assert d['supplier_id'] == 34, d
                    elif purpose == 'found':
                        assert draft['prepared']['can_commit'], draft
                    elif purpose == 'multi':
                        assert d['product_id'] == 46 and d['cases'] == 5, d
                        queued = cards(data, 'receive_queue')[0]
                        assert 'Oats' in json.dumps(queued) or 'avena' in json.dumps(queued), queued
                    elif purpose == 'make':
                        assert d['input_plan'] and not draft['prepared']['can_commit'], draft
                        assert all(not line.get('confirmed') for line in d['input_plan']), d
                    row['draft_facts'] = {k: d[k] for k in ('product_id', 'supplier_id', 'cases', 'case_size_lb', 'total_lb', 'supplier_lot_code', 'bol_reference', 'reason_code') if k in d}
                    if purpose == 'multi':
                        continuation = 'Continúa la siguiente recepción pendiente' if language == 'es' else 'Continue the next queued receive'
                        blocked = send(sid, language, continuation)
                        assert cards(blocked, 'receive_queue')[0].get('finish_current'), blocked
                        recorded = client.post('/assistant/record', headers=headers, json={
                            'draft_id': draft['id'], 'acknowledged_warnings': [w['code'] for w in draft['prepared'].get('warnings', [])]})
                        assert recorded.status_code == 200, recorded.text
                        row['first_receipt'] = recorded.json()['result']['receipt_number']
                        resumed = client.post('/assistant/resume', headers=headers, json={'session_id': sid})
                        assert resumed.status_code == 200
                        next_data = follow(sid, language, send(sid, language, continuation), purpose)
                        next_draft = cards(next_data, 'draft')[0]
                        assert next_draft['prepared']['draft']['product_id'] == 32 and next_draft['prepared']['draft']['cases'] == 3, next_draft
                        row['second_draft_product_id'] = 32
                        draft = next_draft
                    # Cancel the final review draft; only explicitly selected queue records post.
                    r = client.post('/assistant/cancel', headers=headers, json={'draft_id': draft['id']})
                    assert r.status_code == 200, r.text
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
