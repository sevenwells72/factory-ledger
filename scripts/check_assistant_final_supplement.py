#!/usr/bin/env python3
"""Supplement exact staging/user sentences against real AI and a local data copy.

Run separately from the main proof. Only the simulated worker's explicit Record
posts found inventory or the first queued receive; no lot confirmation is supplied.
"""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from hashlib import sha256
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
CASES = (
    ('found_record_en', 'en', 'We found 2 cases of Honey, 5 lb each, during the physical count', 'found'),
    ('found_record_es', 'es', 'Encontramos 2 cajas de miel, 5 lb cada una, en el conteo físico', 'found'),
    ('found_reason_record_en', 'en', 'We found 2 cases of Honey, 5 lb each', 'found_reason'),
    ('found_reason_record_es', 'es', 'Encontramos 2 cajas de miel, 5 lb cada una', 'found_reason'),
    ('bags_exact_es', 'es', '10 bolsas de avena de 50 libras', 'bags'),
    ('bags_each_exact_en', 'en', '10 bags of oats, 50 lb each', 'bags'),
    ('bags_exact_en', 'en', '10 bags of oats, 50 lb bags', 'bags'),
    ('setton_bare_en', 'en', 'Add Setton', 'customer'),
    ('setton_bare_es', 'es', 'Agrega Setton', 'customer'),
    ('ss_original_voice', 'en', 'we made one batch of S S original number one', 'ss_make'),
    ('multi_short_en', 'en', '5 cases of Sunflower Seeds and 3 cases of Oats', 'multi'),
    ('multi_short_es', 'es', '5 cajas de semillas de girasol y 3 cajas de avena', 'multi'),
    ('change_report_en', 'en', 'Change order SO-261015-001: Kookies & Kreme should be 6 cases, not 4', 'handoff'),
    ('change_report_es', 'es', 'Cambia el pedido SO-261015-001: Kookies & Kreme debe ser 6 cajas, no 4', 'handoff'),
    ('expected_short_en', 'en', 'Grain Supply is sending 20 cases of Oats', 'handoff'),
    ('expected_short_es', 'es', 'Grain Supply está enviando 20 cajas de avena', 'handoff'),
    ('make_missing_en', 'en', 'We made 1 batch of Setton Coffee Granola #12', 'missing_make'),
    ('make_missing_es', 'es', 'Hicimos 1 lote de Setton Coffee Granola #12', 'missing_make'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sessions', type=int, choices=(0, 1), required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--only', default='')
    args = parser.parse_args()
    url = urlsplit(args.database_url)
    assert url.hostname == '127.0.0.1' and re.fullmatch(r'/fl_final_supplement[a-z0-9_]*', url.path)
    assert not url.query and not url.fragment and args.repeats >= 3
    for key in ('PGOPTIONS', 'PGSERVICE', 'PGSERVICEFILE', 'PGHOSTADDR'):
        os.environ.pop(key, None)
    os.environ.update(DATABASE_URL=args.database_url, ASSISTANT_ENABLED='1',
                      API_KEY=uuid4().hex, DASHBOARD_API_KEY=uuid4().hex,
                      KEY_SESSIONS_ENABLED=str(args.sessions),
                      EVIDENCE_STORAGE_DIR='/private/tmp/fl_final/supplement-photos')
    secret = Path.home() / 'Documents/fl-secrets/openai-api-key.txt'
    assert secret.stat().st_mode & 0o777 == 0o600
    os.environ['OPENAI_API_KEY'] = secret.read_text().strip()
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
    key = 'fl-final-supplement-' + uuid4().hex
    with connection() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES(%s,'owner',%s,true)",
                    ('Final supplemental local worker ' + uuid4().hex[:8], sha256(key.encode()).hexdigest()))
    client = TestClient(api.app)
    headers = {'X-API-Key': key}
    if args.sessions:
        signed = client.post('/auth/session', json={'actor_key': key})
        assert signed.status_code == 200, signed.text
        headers = {'X-API-Key': signed.json()['session_token']}
    provider = fl_assistant.openai
    events = []

    async def observe(path, *, body=None, file=None):
        event = {'request_model': (body or {}).get('model'), 'store': (body or {}).get('store')}
        events.append(event)
        result = await provider(path, body=body, file=file)
        event.update(response_id=result.get('id'), model=result.get('model'), status=result.get('status'),
                     incomplete_details=result.get('incomplete_details'),
                     tools=[x.get('name') for x in result.get('output', []) if x.get('type') == 'function_call'])
        return result

    fl_assistant.openai = observe
    cases = CASES
    if args.only:
        selected = set(args.only.split(','))
        cases = [c for c in cases if c[0] in selected]
        assert len(cases) == len(selected)
    sources = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    results, full = [], []
    report = {'sessions': args.sessions, 'real_provider': True, 'local_copy_only': True,
              'repeats': args.repeats, 'source_sha256': sources, 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    private = args.output.with_suffix('.private.json')

    def save():
        report.update(passed=sum(r['status'] == 'passed' for r in results),
                      failed=sum(r['status'] != 'passed' for r in results))
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        private.write_text(json.dumps(full, indent=2, ensure_ascii=False, default=str) + '\n')
        private.chmod(0o600)

    def cards(data, kind):
        return [c for c in data.get('cards', []) if c.get('kind') == kind]

    def send(sid, language, text='', **extra):
        response = client.post('/assistant/turn', headers=headers, json={
            'session_id': sid, 'turn_id': str(uuid4()), 'language': language, 'text': text, **extra})
        data = response.json()
        steps.append({'text': text, 'status': response.status_code, 'response': data})
        assert response.status_code == 200, data
        assert not any(c.get('error_code') == 'ASSISTANT_QUOTA_EXHAUSTED' for c in data.get('cards', [])), 'PROVIDER_QUOTA'
        return data

    def follow(sid, language, data, purpose):
        clarified_receiving = False
        for _ in range(7):
            if cards(data, 'draft'):
                return data
            choice = next(iter(cards(data, 'choices')), None)
            if choice:
                targets = ('Honey', 'Oats', 'Sunflower Seeds', 'Grain Supply',
                           'Batch SS Original Granola #1', 'Batch Setton Coffee Granola #12')
                opts = choice.get('result', {}).get('candidates', [])
                selected = next((c for name in targets for c in opts if c.get('name', '').startswith(name)), None)
                assert selected, choice
                data = send(sid, language, choice_id=choice.get('id') or choice['choice_id'], selected_id=selected['id'])
                continue
            reason = next((c for c in data.get('cards', []) if c.get('reason_required')), None)
            if reason:
                assert any(r['code'] == 'physical_count' for r in reason.get('reasons', [])), reason
                row['reason_button_selected'] = True
                data = send(sid, language, 'El motivo de corrección es physical_count.' if language == 'es' else 'The correction reason is physical_count.')
                continue
            if cards(data, 'missing_facts') and purpose in ('bags', 'multi'):
                # Supply only facts absent from the original sentence. Never
                # repeat/correct the package count or bag weight under test.
                facts = ('De Grain Supply, BOL SUPPLEMENT.' if language == 'es' else 'From Grain Supply, BOL SUPPLEMENT.')
                if purpose == 'multi':
                    facts = ('De Grain Supply, 50 lb por caja, BOL SUPPLEMENT.' if language == 'es' else 'From Grain Supply, 50 lb per case, BOL SUPPLEMENT.')
                data = send(sid, language, facts)
                continue
            if purpose == 'multi' and not clarified_receiving and any(
                    cards(data, kind) for kind in ('clarify', 'misunderstood', 'unsupported_request')):
                # The literal user phrase contains no action/supplier/weight/BOL.
                # Clarify those facts without repeating either product or count.
                clarified_receiving = True
                data = send(sid, language,
                    'Son artículos que recibimos de Grain Supply, 50 lb por caja, BOL SUPPLEMENT.'
                    if language == 'es' else
                    'These are items we received from Grain Supply, 50 lb per case, BOL SUPPLEMENT.')
                continue
            return data
        raise AssertionError('Conversation did not converge')

    def transaction_count(ticket_id):
        with connection() as conn, conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM transactions WHERE ticket_id=%s', (ticket_id,))
            return cur.fetchone()[0]

    def record(draft):
        response = client.post('/assistant/record', headers=headers, json={'draft_id': draft['id'],
            'acknowledged_warnings': [w['code'] for w in draft['prepared'].get('warnings', []) if w.get('requires_ack')]})
        data = response.json()
        steps.append({'record': True, 'status': response.status_code, 'response': data})
        assert response.status_code == 200 and data.get('kind') == 'receipt', data
        return data

    for label, language, sentence, purpose in cases:
        for repeat in range(1, args.repeats + 1):
            events.clear(); steps = []; started = time.perf_counter()
            row = {'case': label, 'language': language, 'text': sentence, 'run': repeat}
            try:
                opened = client.post('/assistant/session', headers=headers)
                assert opened.status_code == 200, opened.text
                sid = opened.json()['session_id']
                if purpose == 'multi':
                    with connection() as conn, conn.cursor() as cur:
                        cur.execute('SELECT count(*) FROM transactions')
                        before = cur.fetchone()[0]
                data = send(sid, language, sentence)
                if purpose == 'multi':
                    assert any(cards(data, kind) for kind in ('draft', 'choices', 'missing_facts')), data
                    assert not any(cards(data, kind) for kind in ('misunderstood', 'unsupported_request', 'unavailable')), data
                    first_queue = cards(data, 'receive_queue')[0]
                    queued = ' '.join(first_queue['queued_text'])
                    assert '3' in queued and ('Oats' in queued or 'avena' in queued), first_queue
                    with connection() as conn, conn.cursor() as cur:
                        cur.execute('SELECT count(*) FROM transactions')
                        assert cur.fetchone()[0] == before, 'Initial request posted without Record'
                        cur.execute('SELECT state FROM assistant_sessions WHERE id=%s', (sid,))
                        current = cur.fetchone()[0]['receive_queue_current']
                    assert '5' in current and ('Sunflower' in current or 'girasol' in current), current
                    row.update(first_reply_retains_queue=True, first_reply_posts_nothing=True)
                if purpose == 'handoff':
                    serialized = json.dumps(data)
                    assert cards(data, 'handoff') and '/dashboard/office.html?' in serialized, data
                    assert ('create_expected_receipt' if label.startswith('expected') else 'update_order_line') in serialized
                    assert not events, 'Deterministic Office handoff called a model'
                elif purpose == 'customer':
                    choice = cards(data, 'choices')[0]
                    assert choice.get('existing_customer_only'), choice
                    opt = next(c for c in choice['result']['candidates'] if c['name'] == 'Setton Farms')
                    data = send(sid, language, choice_id=choice.get('id') or choice['choice_id'], selected_id=opt['id'])
                    assert cards(data, 'handoff') and 'Setton Farms' in json.dumps(data) and not cards(data, 'draft'), data
                else:
                    data = follow(sid, language, data, purpose)
                    if purpose == 'missing_make':
                        serialized = json.dumps(data, ensure_ascii=False)
                        assert 'INSUFFICIENT_STOCK' in serialized and 'Coffee' in serialized, data
                        assert ('necesita 0.5 lb' if language == 'es' else 'needs 0.5 lb') in serialized, data
                        assert ('Registre primero la recepción' if language == 'es' else 'receive it first') in serialized, data
                        assert 'Revise los datos de este borrador' not in serialized
                    else:
                        draft = cards(data, 'draft')[0]
                        d = draft['prepared']['draft']
                        if purpose.startswith('found'):
                            assert d['product_id'] == 30 and d['quantity'] == 10 and d['reason_code'] == 'physical_count', d
                            assert draft['prepared']['can_commit'], draft
                            if purpose == 'found_reason':
                                assert row.get('reason_button_selected'), 'Reason was invented instead of chosen'
                            ticket_id = draft['prepared']['ticket_id']
                            assert transaction_count(ticket_id) == 0
                            recorded = record(draft)['result']
                            assert transaction_count(ticket_id) == 1
                            assert record(draft)['result']['receipt_number'] == recorded['receipt_number']
                            assert transaction_count(ticket_id) == 1
                            detail = client.get('/receipts/' + recorded['receipt_number'], headers=headers)
                            assert detail.status_code == 200 and detail.json().get('transactions'), detail.text
                            with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                                cur.execute('SELECT reason_code,reason_notes,reason_notes_es FROM inventory_adjustments WHERE lot_id=%s', (recorded['lot_id'],))
                                audit = cur.fetchone()
                                assert audit['reason_code'] == 'physical_count' and (audit['reason_notes'] or audit['reason_notes_es']), audit
                            row.update(receipt_number=recorded['receipt_number'], single_post=True, replay=True,
                                       note_en_present=bool(d.get('notes')), note_es_present=bool(d.get('notes_es')))
                        elif purpose == 'bags':
                            assert d['product_id'] == 32 and d['cases'] == 10 and d['case_size_lb'] == 50 and d['total_lb'] == 500, d
                            row['draft_total_lb'] = d['total_lb']
                        elif purpose == 'ss_make':
                            assert d['product_id'] == 116 and d['input_plan'], d
                            assert not draft['prepared']['can_commit'] and all(not i.get('confirmed') for i in d['input_plan'])
                            row['draft_product_id'] = d['product_id']
                        elif purpose == 'multi':
                            assert d['product_id'] == 46 and d['cases'] == 5, d
                            queue = cards(data, 'receive_queue')[0]
                            assert 'Oats' in json.dumps(queue) or 'avena' in json.dumps(queue), queue
                            continuation = 'Continúa la siguiente recepción pendiente' if language == 'es' else 'Continue the next queued receive'
                            blocked = send(sid, language, continuation)
                            assert cards(blocked, 'receive_queue')[0].get('finish_current'), blocked
                            row['first_receipt'] = record(draft)['result']['receipt_number']
                            resumed = client.post('/assistant/resume', headers=headers, json={'session_id': sid})
                            assert resumed.status_code == 200
                            data = follow(sid, language, send(sid, language, continuation), purpose)
                            draft = cards(data, 'draft')[0]; d = draft['prepared']['draft']
                            assert d['product_id'] == 32 and d['cases'] == 3, d
                            row['second_product_id'] = d['product_id']
                        if not purpose.startswith('found'):
                            cancelled = client.post('/assistant/cancel', headers=headers, json={'draft_id': draft['id']})
                            assert cancelled.status_code == 200, cancelled.text
                if purpose != 'handoff':
                    assert events and all(e.get('response_id') and e.get('status') == 'completed' for e in events), events
                row['status'] = 'passed'
            except Exception as error:
                row['status'] = 'failed'; row['error'] = str(error)[:2000]
            row['seconds'] = round(time.perf_counter() - started, 3)
            row['provider'] = list(events)
            results.append(row); full.append({**row, 'steps': steps}); save()
            print(f"{label} {repeat}/{args.repeats}: {row['status']} ({row['seconds']}s)", flush=True)
            if 'PROVIDER_QUOTA' in row.get('error', ''):
                raise SystemExit('Provider credits exhausted; remaining proof not run')
    report['source_unchanged'] = all(sha256((ROOT / p).read_bytes()).hexdigest() == digest for p, digest in sources.items())
    save()
    raise SystemExit(bool(report['failed']) or not report['source_unchanged'])


if __name__ == '__main__':
    main()
