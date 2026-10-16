#!/usr/bin/env python3
"""Real-provider customer/order regression proof on disposable local staging copies.

Only prepares drafts. Existing catalog choices and Add-new follow-ups reproduce
human actions; no Record route is called and catalog/ledger fingerprints must stay fixed.
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-url', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sessions', type=int, choices=(0, 1), required=True)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    url = urlsplit(args.database_url)
    assert url.hostname == '127.0.0.1' and re.fullmatch(r'/fl_final2_customer_paths_[01]', url.path)
    assert url.path.endswith(str(args.sessions)) and not url.query and not url.fragment
    assert args.repeats >= 3
    for key in ('PGOPTIONS', 'PGSERVICE', 'PGSERVICEFILE', 'PGHOSTADDR'):
        os.environ.pop(key, None)
    os.environ.update(DATABASE_URL=args.database_url, ASSISTANT_ENABLED='1',
                      API_KEY=uuid4().hex, DASHBOARD_API_KEY=uuid4().hex,
                      KEY_SESSIONS_ENABLED=str(args.sessions))
    os.environ['OPENAI_API_KEY'] = (Path.home() / 'Documents/fl-secrets/openai-api-key.txt').read_text().strip()
    sys.path.insert(0, str(ROOT))
    logging.disable(logging.CRITICAL)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        import main as api
        import fl_assistant
    assert os.getenv('ASSISTANT_MODEL', fl_assistant.MODEL) == 'gpt-6.1-sol'
    import psycopg2
    from fastapi.testclient import TestClient

    @contextmanager
    def connection():
        conn = psycopg2.connect(args.database_url)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def fingerprints():
        result = {}
        with connection() as conn, conn.cursor() as cur:
            for table in ('customers', 'sales_orders', 'sales_order_lines', 'transactions', 'transaction_lines', 'lots'):
                cur.execute('SELECT row_to_json(t)::text FROM ' + table + ' t ORDER BY id')
                rows = [row[0] for row in cur.fetchall()]
                result[table] = {'rows': len(rows), 'sha256': sha256('\n'.join(rows).encode()).hexdigest()}
        return result

    before = fingerprints()
    api.get_db_connection = connection
    api._reset_actor_cache()
    actor_key = 'fl-final2-customer-proof-' + uuid4().hex
    with connection() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES(%s,'owner',%s,true)",
                    ('Final2 customer proof ' + uuid4().hex[:8], sha256(actor_key.encode()).hexdigest()))
        cur.execute("SELECT id,name,case_size_lb FROM products WHERE name='Kookies & Kreme – 10 LB' AND active")
        product_id, product_name, case_size = cur.fetchone()
        assert case_size == 10
        cur.execute("SELECT id FROM customers WHERE name='Sunshine Granola' AND active")
        customer_id = cur.fetchone()[0]
    client = TestClient(api.app)
    headers = {'X-API-Key': actor_key}
    if args.sessions:
        signed = client.post('/auth/session', json={'actor_key': actor_key})
        assert signed.status_code == 200, signed.status_code
        headers = {'X-API-Key': signed.json()['session_token']}

    provider = fl_assistant.openai
    events = []

    async def observe(path, *, body=None, file=None):
        assert path == 'responses' and body['store'] is False and body['model'] == 'gpt-6.1-sol'
        event = {'request_model': body['model'], 'store': body['store']}
        events.append(event)
        result = await provider(path, body=body, file=file)
        event.update(response_id=result.get('id'), model=result.get('model'), status=result.get('status'),
                     tools=[x.get('name') for x in result.get('output', []) if x.get('type') == 'function_call'])
        return result

    fl_assistant.openai = observe
    sources = {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    results, private_results = [], []
    report = {'sessions': args.sessions, 'real_model': True, 'repeats': args.repeats,
              'database_local_only': True, 'record_calls': 0,
              'source_sha256': sources, 'results': results}
    private = Path('/private/tmp/fl_final2') / f'customer-paths-{args.sessions}.private.json'
    private.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        private.write_text(json.dumps(private_results, ensure_ascii=False, indent=2, default=str) + '\n')
        private.chmod(0o600)

    def send(sid, language, text='', **extra):
        response = client.post('/assistant/turn', headers=headers, json={
            'session_id': sid, 'turn_id': str(uuid4()), 'text': text, 'language': language, **extra})
        data = response.json()
        steps.append({'text': text, 'status': response.status_code, 'result': data})
        assert response.status_code == 200, f'turn HTTP {response.status_code}'
        assert not any(c.get('error_code') == 'ASSISTANT_QUOTA_EXHAUSTED' for c in data.get('cards', [])), 'PROVIDER_QUOTA'
        return data

    for purpose in ('existing_order', 'new_customer'):
        for language in ('en', 'es'):
            for repeat in range(1, args.repeats + 1):
                events.clear()
                steps = []
                started = time.perf_counter()
                customer_name = 'Final2 Boreal Orchard ' + uuid4().hex[:8]
                po = f'FINAL2-PATH-{args.sessions}-{language}-{repeat}'
                text = ((f'Create an order for Sunshine Granola for 2 cases of {product_name}, PO {po}.' if language == 'en' else
                         f'Crea un pedido para Sunshine Granola de 2 cajas de {product_name}, PO {po}.')
                        if purpose == 'existing_order' else
                        (f'Add {customer_name} as a new customer.' if language == 'en' else
                         f'Agrega {customer_name} como cliente nuevo.'))
                row = {'case': purpose, 'language': language, 'run': repeat, 'text': text}
                try:
                    created = client.post('/assistant/session', headers=headers)
                    assert created.status_code == 200, f'chat HTTP {created.status_code}'
                    sid = created.json()['session_id']
                    data = send(sid, language, text)
                    for followup in range(4):
                        draft = next((c for c in data.get('cards', []) if c['kind'] == 'draft'), None)
                        if draft:
                            break
                        choice = next((c for c in data.get('cards', []) if c['kind'] == 'choices'), None)
                        assert choice, 'No draft or human choice: ' + ','.join(c['kind'] for c in data.get('cards', []))
                        if purpose == 'new_customer':
                            assert choice.get('resolution_kind') == 'customer' and not choice.get('existing_customer_only')
                            assert choice.get('query') == customer_name
                            assert followup == 0, 'Add-new choice looped instead of preparing the customer'
                            # Exact follow-up emitted by dashboard/assistant-display.js.
                            selection = (f'Create a new customer named {customer_name} first, then continue the order.' if language == 'en' else
                                         f'Crea primero un cliente nuevo llamado {customer_name} y luego continúa el pedido.')
                            data = send(sid, language, selection)
                            row['human_add_new_choice'] = True
                        else:
                            kind = choice.get('resolution_kind')
                            ident = customer_id if kind == 'customer' else product_id
                            assert kind in ('customer', 'product') and any(c['id'] == ident for c in choice['result']['candidates'])
                            data = send(sid, language, choice_id=choice['id'], selected_id=ident)
                    assert draft, 'No prepared draft'
                    prepared = draft['prepared']
                    assert prepared['action'] == ('create_order' if purpose == 'existing_order' else 'create_customer')
                    assert prepared['can_commit'], 'Draft is blocked: ' + ','.join(b['code'] for b in prepared['blockers'])
                    facts = prepared['draft']
                    if purpose == 'existing_order':
                        assert facts['customer'] == 'Sunshine Granola' and facts['customer_po'] == po
                        assert len(facts['lines']) == 1 and facts['lines'][0]['product_id'] == product_id
                        assert facts['lines'][0]['quantity'] == 2 and facts['lines'][0]['unit'] == 'cases' and facts['total_lb'] == 20
                        assert not any('prepare_create_customer' in event['tools'] for event in events), 'Ordinary order took customer-create path'
                    else:
                        assert facts['customer']['name'] == customer_name
                    assert events and all(e.get('response_id') and e.get('status') == 'completed' for e in events)
                    row.update(status='passed', draft_action=prepared['action'], can_commit=True)
                except Exception as error:
                    row.update(status='failed', error_type=type(error).__name__, error=str(error)[:250])
                row['seconds'] = round(time.perf_counter() - started, 3)
                row['provider'] = list(events)
                results.append(row)
                private_results.append({**row, 'steps': steps})
                save()
                print(f'{purpose} {language} {repeat}/{args.repeats}: {row["status"]} ({row["seconds"]}s)', flush=True)
                if row.get('error') == 'PROVIDER_QUOTA':
                    raise SystemExit('Provider quota exhausted')
    ids = [event['response_id'] for row in results for event in row['provider'] if event.get('response_id')]
    report.update(passed=sum(r['status'] == 'passed' for r in results), failed=sum(r['status'] != 'passed' for r in results),
                  unique_completed_provider_ids=len(set(ids)), provider_ids_unique=len(ids) == len(set(ids)),
                  core_data_unchanged=before == fingerprints(),
                  source_unchanged=all(sha256((ROOT / path).read_bytes()).hexdigest() == digest for path, digest in sources.items()))
    save()
    assert report['provider_ids_unique'] and report['core_data_unchanged'] and report['source_unchanged']
    raise SystemExit(bool(report['failed']))


if __name__ == '__main__':
    main()
