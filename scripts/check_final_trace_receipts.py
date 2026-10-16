#!/usr/bin/env python3
"""Real-model receipt/physical-tag proof on a disposable local staging copy.

This test worker explicitly reads and types each tag. The app/model never confirms
one. Records are local only. No app lifespan or remote database is permitted.
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
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ('receive_en', 'en', 'We received 6 cases of Sunflower Seeds from Grain Supply, 50 lb per case, BOL FS8996, supplier lot SL-64AED4'),
    ('receive_es', 'es', 'Recibimos 6 cajas de semillas de girasol de Grain Supply, 50 lb por caja, BOL FS8996, lote del proveedor SL-64AED4'),
    ('different_en', 'en', 'We made 1 batch of Classic #9'),
    ('different_es', 'es', 'Hicimos 1 lote de Classic #9'),
)
PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-url', required=True)
    parser.add_argument('--sessions', type=int, choices=(0, 1), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    url = urlsplit(args.database_url)
    assert url.hostname == '127.0.0.1' and re.fullmatch('/fl_final_trace[a-z0-9_]*', url.path)
    assert not url.query and not url.fragment and args.repeats >= 3
    for key in ('PGOPTIONS', 'PGSERVICE', 'PGSERVICEFILE', 'PGHOSTADDR'):
        os.environ.pop(key, None)
    os.environ.update(DATABASE_URL=args.database_url, ASSISTANT_ENABLED='1',
        API_KEY=uuid4().hex, DASHBOARD_API_KEY=uuid4().hex,
        KEY_SESSIONS_ENABLED=str(args.sessions), EVIDENCE_STORAGE_DIR='/private/tmp/fl_final/trace-photos')
    secret = Path.home() / 'Documents/fl-secrets/openai-api-key.txt'
    assert secret.stat().st_mode & 0o777 == 0o600
    os.environ['OPENAI_API_KEY'] = secret.read_text().strip()
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, str(ROOT))
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        import main as api
        import fl_assistant
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import RealDictCursor
    from fastapi.testclient import TestClient

    active_url = args.database_url
    admin_url = urlunsplit(url._replace(path="/postgres"))
    @contextmanager
    def connection():
        conn = psycopg2.connect(active_url)
        try:
            with conn:
                yield conn
        finally:
            conn.close()
    api.get_db_connection = connection
    api._reset_actor_cache()
    key = 'final-trace-floor-' + uuid4().hex
    with connection() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES(%s,'floor',%s,true)",
                    ('Final trace local worker ' + uuid4().hex[:8], sha256(key.encode()).hexdigest()))
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
                     tools=[x.get('name') for x in result.get('output', []) if x.get('type') == 'function_call'])
        return result
    fl_assistant.openai = observe
    results, full = [], []
    sources = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    report = {'sessions': args.sessions, 'real_provider': True, 'local_copy_only': True,
              'explicit_test_worker_tags': True, 'source_sha256': sources, 'results': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    private = args.output.with_suffix('.private.json')
    def save():
        report.update(passed=sum(r['status'] == 'passed' for r in results), failed=sum(r['status'] != 'passed' for r in results))
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        private.write_text(json.dumps(full, indent=2, ensure_ascii=False, default=str) + '\n')
        private.chmod(0o600)
    def send(sid, language, text='', **extra):
        response = client.post('/assistant/turn', headers=headers, json={
            'session_id': sid, 'turn_id': str(uuid4()), 'language': language, 'text': text, **extra})
        data = response.json(); steps.append({'text': text, 'status': response.status_code, 'response': data})
        assert response.status_code == 200, data
        assert not any(c.get('code') == 'ASSISTANT_CREDIT_OUTAGE' for c in data.get('cards', [])), 'PROVIDER_QUOTA'
        return data
    def follow(sid, language, data):
        for _ in range(7):
            drafts = [c for c in data.get('cards', []) if c.get('kind') == 'draft']
            if drafts:
                assert len(drafts) == 1
                return drafts[0]
            choice = next((c for c in data.get('cards', []) if c.get('kind') == 'choices'), None)
            if choice:
                candidates = choice.get('result', {}).get('candidates', [])
                targets = ('Grain Supply', 'Sunflower Seeds', 'Batch Classic Granola #9')
                selected = next((c for name in targets for c in candidates if c.get('name') == name), None)
                assert selected, choice
                data = send(sid, language, choice_id=choice.get('id') or choice['choice_id'], selected_id=selected['id'])
                continue
            raise AssertionError(json.dumps(data, ensure_ascii=False))
        raise AssertionError('No converged draft')
    def count(ticket_id):
        with connection() as conn, conn.cursor() as cur:
            cur.execute('SELECT count(*) FROM transactions WHERE ticket_id=%s', (ticket_id,))
            return cur.fetchone()[0]
    def balance(lot_id):
        with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            return api.lot_on_hand(cur, lot_id)
    def record(draft, confirmations=None):
        response = client.post('/assistant/record', headers=headers, json={'draft_id': draft['id'],
            'acknowledged_warnings': [w['code'] for w in draft['prepared'].get('warnings', []) if w.get('requires_ack')],
            'lot_confirmations': confirmations or []})
        data = response.json(); steps.append({'record': True, 'status': response.status_code, 'response': data})
        return response.status_code, data
    for label, language, sentence in CASES:
        for repeat in range(1, args.repeats + 1):
            events.clear(); steps = []; started = time.perf_counter()
            row = {'case': label, 'language': language, 'text': sentence, 'run': repeat}
            # Each run starts from the same staging-stock snapshot. Prior makes
            # must not exhaust the ambiguous FIFO lot needed by the next proof.
            case_db = f'fl_final_trace_case_{args.sessions}_{label}_{repeat}'
            admin = psycopg2.connect(admin_url); admin.autocommit = True
            try:
                with admin.cursor() as cur:
                    cur.execute(sql.SQL('CREATE DATABASE {} TEMPLATE {}').format(
                        sql.Identifier(case_db), sql.Identifier(url.path[1:])))
            finally:
                admin.close()
            active_url = urlunsplit(url._replace(path='/' + case_db))
            api._reset_actor_cache()
            try:
                response = client.post('/assistant/session', headers=headers)
                assert response.status_code == 200, response.text
                sid = response.json()['session_id']
                extra = {}
                if label.startswith('receive'):
                    photo = client.post('/assistant/attachment', params={'session_id': sid}, headers=headers,
                        data={'kind': 'lot_tag'}, files={'file': ('local-test-lot-tag.png', PNG, 'image/png')})
                    assert photo.status_code == 200, photo.text
                    aid = photo.json()['id']; extra['attachment_ids'] = [aid]
                draft = follow(sid, language, send(sid, language, sentence, **extra))
                d = draft['prepared']['draft']; ticket_id = draft['prepared']['ticket_id']
                assert events and all(e.get('status') == 'completed' and e.get('response_id') for e in events), events
                assert count(ticket_id) == 0
                if label.startswith('receive'):
                    assert d['supplier_lot_code'] == 'SL-64AED4' and d['bol_reference'] == 'FS8996'
                    assert d['attachments'][0]['id'] == aid
                    status, recorded = record(draft)
                    assert status == 200 and recorded['kind'] == 'receipt', recorded
                    result = recorded['result']
                    assert result['supplier_lot_code'] == 'SL-64AED4' and result['bol_reference'] == 'FS8996'
                    assert 'Supplier lot: SL-64AED4' in result['receipt_text']
                    assert result['attachments'][0]['id'] == aid
                    detail = client.get('/receipts/' + result['receipt_number'], headers=headers).json()
                    assert detail['lots'][0]['supplier_lot_code'] == 'SL-64AED4'
                    assert detail['transactions'][0]['bol_reference'] == 'FS8996'
                    assert detail['attachments'][0]['id'] == aid
                    assert detail['attachments'][0]['transaction_id'] == result['transaction_id']
                    chat_read = send(sid, language, ('Muestra el recibo ' if language == 'es' else 'Show receipt ') + result['receipt_number'])
                    serialized = json.dumps(chat_read, ensure_ascii=False)
                    assert 'SL-64AED4' in serialized and 'FS8996' in serialized and 'lot_tag' in serialized
                    row.update(photo_linked=True, supplier_lot='SL-64AED4', bol='FS8996')
                else:
                    assert not draft['prepared']['can_commit'] and all(not i['confirmed'] for i in d['input_plan'])
                    salt = [i for i in d['input_plan'] if i['product_name'] == 'Salt']; assert salt, d
                    old = {i['lot_id']: balance(i['lot_id']) for i in salt}
                    physical = 'DIFF-FINAL-TEST-WORKER-' + uuid4().hex[:8]
                    confirmations = [{'lot_id': i['lot_id'], 'method': 'different_lot' if i['product_name'] == 'Salt' else 'full_code',
                        'value': physical if i['product_name'] == 'Salt' else i['lot_code']} for i in d['input_plan']]
                    normal = next(i for i in d['input_plan'] if i['product_name'] != 'Salt')
                    wrong = [dict(e, value='WRONG-TAG') if e['lot_id'] == normal['lot_id'] else e for e in confirmations]
                    status, failed = record(draft, wrong)
                    assert status == 422 and 'LOT_CONFIRMATION_MISMATCH' in json.dumps(failed), failed
                    assert count(ticket_id) == 0
                    ambiguous = None
                    with connection() as conn, conn.cursor() as cur:
                        for line in d['input_plan']:
                            if line['product_name'] == 'Salt': continue
                            cur.execute(f'''SELECT count(*) FROM (SELECT l.id FROM lots l LEFT JOIN {api.POSTED_LINES} tl ON tl.lot_id=l.id
                                WHERE l.product_id=%s AND l.status IS DISTINCT FROM 'merged' AND right(upper(l.lot_code),4)=%s
                                GROUP BY l.id HAVING COALESCE(sum(tl.quantity_lb),0)>0) matches''', (line['product_id'], line['lot_code'][-4:].upper()))
                            if cur.fetchone()[0] > 1:
                                ambiguous = line; break
                    assert ambiguous, 'Staging copy no longer has ambiguous last4 in this make plan'
                    short = [dict(e, method='last4', value=ambiguous['lot_code'][-4:]) if e['lot_id'] == ambiguous['lot_id'] else e for e in confirmations]
                    status, failed = record(draft, short)
                    assert status == 422 and 'AMBIGUOUS_SUFFIX' in json.dumps(failed), failed
                    if language == 'es':
                        assert 'código completo' in json.dumps(failed, ensure_ascii=False), failed
                    assert count(ticket_id) == 0
                    status, recorded = record(draft, confirmations)
                    assert status == 200 and recorded['kind'] == 'receipt', recorded
                    result = recorded['result']
                    assert all(balance(lid) == before for lid, before in old.items())
                    assert result['lot_exceptions'] and any(e['physical_tag'] == physical for e in result['lot_exceptions'])
                    with connection() as conn, conn.cursor() as cur:
                        for exception in result['lot_exceptions']:
                            cur.execute('SELECT status FROM exceptions WHERE id=%s', (exception['exception_id'],))
                            assert cur.fetchone()[0] == 'open'
                    row.update(wrong_tag_zero_writes=True, ambiguous_suffix_zero_writes=True, full_code_recovery=True,
                        suggested_salt_unchanged=True, physical_lot_unresolved=True)
                assert count(ticket_id) == 1
                status, replay = record(draft)
                assert status == 200 and replay['result']['receipt_number'] == result['receipt_number'] and replay['result']['replayed']
                assert count(ticket_id) == 1
                row.update(status='passed', receipt_number=result['receipt_number'], single_post=True, replay=True)
            except Exception as error:
                row.update(status='failed', error=str(error)[:1800])
            row.update(seconds=round(time.perf_counter() - started, 3), provider=list(events))
            results.append(row); full.append({**row, 'steps': steps}); save()
            active_url = args.database_url
            admin = psycopg2.connect(admin_url); admin.autocommit = True
            try:
                with admin.cursor() as cur:
                    cur.execute(sql.SQL('DROP DATABASE {}').format(sql.Identifier(case_db)))
            finally:
                admin.close()
            row['case_database_dropped'] = True
            save()
            print(f"{label} {repeat}/{args.repeats}: {row['status']} ({row['seconds']}s)", flush=True)
            if 'PROVIDER_QUOTA' in row.get('error', ''): break
    report['source_sha256_after'] = {p: sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}
    report['changed_source_files'] = [p for p in sources if sources[p] != report['source_sha256_after'][p]]
    report['source_unchanged'] = not report['changed_source_files']
    core = ('main.py', 'fl_assistant.py', 'ticket_actions.py', 'lot_confirmation.py', 'write_tickets.py', 'attachments.py', 'assistant_tools.py', 'assistant_receive.py', 'assistant_voice.py')
    report['trace_recording_source_unchanged'] = all(p not in report['changed_source_files'] for p in core)
    save()
    raise SystemExit(bool(report['failed']))


if __name__ == '__main__':
    main()
