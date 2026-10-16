#!/usr/bin/env python3
"""Repeat Michael's exact Round E transcripts using real OpenAI on a local staging copy.

Prepare the copy with a read-only public-schema staging dump and PG17 restore.
This runner refuses every non-loopback database, never starts app lifespan,
never seeds catalog/inventory/order data, and never sends Record. The sole
local fixture is an owner actor so normal auth/session routes can be used.
The OpenAI key is read in-process from the protected existing secret file.
"""
import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime
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
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2 import sql
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
TRANSCRIPTS = (
    ('ship_spoken', 'Ship S O two six one oh one five dash zero zero one.'),
    ('ship_spaced', 'SO 261015 001'),
    ('pack_coconut', 'Packed 10 cases of sweetened flake coconut.'),
)
CORE_TABLES = ('products', 'suppliers', 'customers', 'lots', 'transactions',
               'transaction_lines', 'sales_orders', 'sales_order_lines',
               'sales_order_allocations', 'batch_formulas')
SECRET_FIELDS = {'ticket', 'authorization', 'api_key', 'key_hash', 'session_token',
                 'payload_hash', 'download_url', 'packing_slip_url'}


def sanitized(value):
    if isinstance(value, dict):
        return {key: sanitized(item) for key, item in value.items()
                if key.lower() not in SECRET_FIELDS}
    if isinstance(value, list):
        return [sanitized(item) for item in value]
    return value


def fingerprint(connection):
    """Hash every core row, preserving evidence without exporting staging rows."""
    result = {}
    with connection() as conn, conn.cursor() as cur:
        for table in CORE_TABLES:
            cur.execute(sql.SQL('SELECT row_to_json(t)::text FROM {} t ORDER BY id')
                        .format(sql.Identifier(table)))
            digest = sha256()
            count = 0
            for (row,) in cur:
                digest.update(row.encode() + b'\n')
                count += 1
            result[table] = {'rows': count, 'sha256': digest.hexdigest()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-url', required=True)
    parser.add_argument('--staging-dump', type=Path, required=True,
                        help='Private dump used to create this local copy (hashed only)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--sessions', type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    parsed = urlsplit(args.database_url)
    if (parsed.scheme not in ('postgres', 'postgresql') or parsed.hostname != '127.0.0.1'
            or not re.fullmatch(r'/fl_round_e_[a-z0-9_]+', parsed.path)
            or parsed.query or parsed.fragment):
        raise RuntimeError('Round E proof requires its disposable loopback database')
    if args.repeats < 5:
        raise RuntimeError('Each exact transcript requires at least five repetitions')
    dump_hash = sha256(args.staging_dump.read_bytes()).hexdigest()
    for name in ('PGOPTIONS', 'PGSERVICE', 'PGSERVICEFILE', 'PGHOSTADDR'):
        os.environ.pop(name, None)
    os.environ.update(DATABASE_URL=args.database_url, ASSISTANT_ENABLED='1',
                      API_KEY='round-e-local-master-' + uuid4().hex,
                      DASHBOARD_API_KEY='round-e-local-read-' + uuid4().hex,
                      KEY_SESSIONS_ENABLED=str(args.sessions),
                      EVIDENCE_STORAGE_DIR='/private/tmp/fl_round_e_evidence')
    if not os.environ.get('OPENAI_API_KEY'):
        secret = Path.home() / 'Documents/fl-secrets/openai-api-key.txt'
        if secret.stat().st_mode & 0o777 != 0o600:
            raise RuntimeError('OpenAI secret file permissions must be 600')
        os.environ['OPENAI_API_KEY'] = secret.read_text().strip()
    logging.disable(logging.CRITICAL)
    sys.path.insert(0, str(ROOT))
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        import main as api
        import fl_assistant
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
    actor_key = 'round-e-local-owner-' + uuid4().hex
    with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("INSERT INTO actors(name,role,key_hash,active) VALUES (%s,'owner',%s,true) RETURNING id",
                    ('Round E real-model validation ' + uuid4().hex[:8], sha256(actor_key.encode()).hexdigest()))
        actor_id = cur.fetchone()['id']
        cur.execute("SELECT id,order_number FROM sales_orders WHERE order_number='SO-261015-001'")
        order = dict(cur.fetchone() or {})
        if not order:
            raise RuntimeError('The real staging order is missing from the copy')
        cur.execute("SELECT id,name FROM products WHERE type='batch' AND name='Batch Coconut Sweetened Flake' AND active IS DISTINCT FROM false")
        coconut_parent = dict(cur.fetchone() or {})
        if not coconut_parent:
            raise RuntimeError('The real sweetened-flake coconut bulk parent is missing')
    before = fingerprint(connection)
    traces = []
    provider = fl_assistant.openai
    original_relay = fl_assistant.relay
    current_events = []

    async def real_openai(path, *, body=None, file=None):
        """Observe the actual transport, without changing request or response."""
        event = {'kind': 'model', 'path': path, 'request_model': (body or {}).get('model'),
                 'allowed_tools': [item.get('name') for item in (body or {}).get('tools', [])],
                 'order_in_model_input': order['order_number'] in json.dumps((body or {}).get('input', []))}
        current_events.append(event)
        started = time.perf_counter()
        result = await provider(path, body=body, file=file)
        event.update(seconds=round(time.perf_counter() - started, 4),
                     response_id=result.get('id'), response_model=result.get('model'),
                     status=result.get('status'), usage=result.get('usage'),
                     tools=[{'name': item.get('name'),
                             'arguments': json.loads(item.get('arguments') or '{}')}
                            for item in result.get('output', []) if item.get('type') == 'function_call'])
        return result

    async def observe_relay(app, request, method, path, **kwargs):
        current_events.append({'kind': 'fl_route', 'method': method, 'path': path,
                               'body': sanitized(kwargs.get('body')),
                               'params': sanitized(kwargs.get('params'))})
        return await original_relay(app, request, method, path, **kwargs)

    fl_assistant.openai = real_openai
    fl_assistant.relay = observe_relay
    client = TestClient(api.app)  # No context manager: app lifespan never runs.
    headers = {'X-API-Key': actor_key}
    if args.sessions:
        signed = client.post('/auth/session', json={'actor_key': actor_key})
        if signed.status_code != 200:
            raise RuntimeError('Local actor key-session sign-in failed')
        headers = {'X-API-Key': signed.json()['session_token']}

    def turn(sid, text='', **extra):
        current_events.clear()
        response = client.post('/assistant/turn', headers=headers, json={
            'session_id': sid, 'turn_id': str(uuid4()), 'text': text, 'language': 'en', **extra})
        return {'text': text, 'status': response.status_code,
                'response': sanitized(response.json()), 'trace': sanitized(list(current_events))}

    def validate(label, steps, sid):
        errors = []
        if any(step['status'] != 200 for step in steps):
            errors.append('HTTP status was not 200')
        cards = steps[-1]['response'].get('cards', [])
        all_cards = [card for step in steps for card in step['response'].get('cards', [])]
        models = [event for step in steps for event in step['trace'] if event['kind'] == 'model']
        if not models or any(event.get('status') != 'completed' for event in models):
            errors.append('No completed real provider request, or a provider request failed')
        if any(card.get('kind') in ('lookup_failed', 'misunderstood', 'refusal') for card in all_cards):
            errors.append('Lookup/model/refusal card shown')
        if label.startswith('ship_'):
            events = steps[0]['trace']
            first_model = next((index for index, event in enumerate(events) if event['kind'] == 'model'), -1)
            detail = '/sales/orders/' + str(order['id'])
            if first_model < 0 or not any(event['kind'] == 'fl_route' and event['path'] == detail
                                         for event in events[:first_model]):
                errors.append('FL did not resolve and read the real order before the model')
            if any(event['kind'] == 'fl_route' and event['path'] == '/resolve'
                   and (event.get('body') or {}).get('kind') == 'product' for event in events):
                errors.append('Order number was routed to product resolution')
            if any(card.get('resolution_kind') == 'product' for card in all_cards):
                errors.append('Product choices shown for order number')
            flow = next((card for card in cards if card.get('shipping_quantity_needed')), None)
            if not flow or flow.get('order', {}).get('order_number') != order['order_number']:
                errors.append('Exact order-number transcript did not start the correct ship quantity flow')
        else:
            choice = next((card for card in cards if card.get('packing_lots')), None)
            if not choice or choice.get('kind') != 'choices':
                errors.append('No real parent bulk-batch buttons')
            else:
                if choice.get('message') != 'Which batch did these come from?':
                    errors.append('Batch question does not use the requested wording')
                candidates = choice.get('result', {}).get('candidates', [])
                lots = [item for item in candidates if isinstance(item.get('id'), int)]
                parents = {item.get('product_id') for item in lots}
                if parents != {coconut_parent['id']}:
                    errors.append('Lot buttons changed the exact coconut family (for example to toasted coconut)')
                with connection() as conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute('SELECT state FROM assistant_sessions WHERE id=%s', (sid,))
                    state = cur.fetchone()['state']
                    target = (state.get('pending') or {}).get('packing_context', {}).get('target_product_id')
                    cur.execute('SELECT name,type,parent_batch_product_id FROM products WHERE id=%s', (target,))
                    product = cur.fetchone()
                    parent = product['parent_batch_product_id'] if product else None
                    if product and product['type'] != 'finished' and len(parents) == 1:
                        # The raw name can identify several finished case sizes.
                        # FL may ask for the shared bulk lot before that size.
                        parent = next(iter(parents))
                        cur.execute("SELECT name FROM products WHERE parent_batch_product_id=%s AND type='finished' AND active IS DISTINCT FROM false", (parent,))
                        outputs = [row['name'] for row in cur.fetchall()]
                        if not outputs or not all('coconut' in name.lower() for name in outputs):
                            errors.append('Bulk lot is not a real parent of the coconut finished family')
                    if not product or parents != {parent}:
                        errors.append('Lot buttons are not for the real finished product parent')
                    expected = [row['id'] for row in api.fifo_lot_balances(cur, parent)] if parent else []
                if not lots or [item['id'] for item in lots] != expected:
                    errors.append('Lot buttons do not match all available parent lots in FIFO order')
                if not choice.get('packing_lots'):
                    errors.append('Different batch affordance absent')
            if any(card.get('kind') == 'missing_facts' and 'source_product_id' in card.get('fields', [])
                   for card in all_cards):
                errors.append('Bare missing bulk-batch details shown')
        return errors

    try:
        for label, transcript in TRANSCRIPTS:
            for repeat in range(1, args.repeats + 1):
                created = client.post('/assistant/session', headers=headers)
                if created.status_code != 200:
                    raise RuntimeError('Local Assistant session creation failed')
                sid = created.json()['session_id']
                steps = [turn(sid, transcript)]
                if label == 'pack_coconut':
                    # Finished case sizes can legitimately need a human product button.
                    for _ in range(2):
                        choices = next((card for card in steps[-1]['response'].get('cards', [])
                                        if card.get('kind') == 'choices' and card.get('resolution_kind') == 'product'), None)
                        if not choices:
                            break
                        candidates = choices.get('result', {}).get('candidates', [])
                        selected = next((item for item in candidates
                                         if 'coconut' in item.get('name', '').lower()
                                         and float(item.get('case_size_lb') or 0) == 10), None)
                        if selected is None:
                            selected = next((item for item in candidates
                                             if 'coconut' in item.get('name', '').lower()), None)
                        if selected is None:
                            break
                        follow = turn(sid, choice_id=choices.get('id') or choices.get('choice_id'),
                                      selected_id=selected['id'])
                        follow['human_product_button'] = selected
                        steps.append(follow)
                errors = validate(label, steps, sid)
                row = {'label': label, 'repeat': repeat, 'exact_transcript': transcript,
                       'passed': not errors, 'errors': errors, 'steps': steps}
                traces.append(row)
                print(json.dumps({key: row[key] for key in ('label', 'repeat', 'passed', 'errors')}), flush=True)
    finally:
        after = fingerprint(connection)
        result = {'created_at': datetime.now(ZoneInfo('America/New_York')).isoformat(),
                  'local_database': parsed.path.lstrip('/'), 'staging_dump_sha256': dump_hash,
                  'real_model': True, 'transport': 'original fl_assistant.openai; observer only',
                  'sessions_enabled': bool(args.sessions), 'repeats_per_exact_transcript': args.repeats,
                  'synthetic_catalog_seeded': False, 'record_requests': 0,
                  'verified_real_order': order, 'verified_real_coconut_parent': coconut_parent,
                  'ledger_and_catalog_unchanged': before == after, 'before': before, 'after': after,
                  'source_sha256': {name: sha256((ROOT / name).read_bytes()).hexdigest()
                                    for name in ('fl_assistant.py', 'assistant_voice.py', 'assistant_tools.py',
                                                 'assistant_replies.py', 'assistant_receive.py', 'write_tickets.py',
                                                 'scripts/check_assistant_round_e_real_model.py')},
                  'results': traces}
        result['passed'] = before == after and len(traces) == len(TRANSCRIPTS) * args.repeats and all(row['passed'] for row in traces)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, 'w') as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(json.dumps(result, indent=2, ensure_ascii=False, default=str) + '\n')
        client.close()
        api._reset_actor_cache()
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Transport diagnostics can contain DSNs; report the type only.
        print('Round E validation failed: ' + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
