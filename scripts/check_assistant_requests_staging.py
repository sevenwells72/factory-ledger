#!/usr/bin/env python3
"""F1 request smoke: hosted model, inventory, receipts and editable dictation.

Staging only. Creates assistant transport sessions/turns, never ledger drafts or
records. Reads the protected staging Michael key in memory; never prints it.
"""
import argparse
from datetime import datetime
from hashlib import sha256
import json
import re
from pathlib import Path
import sys
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx

from check_assistant_staging import BASE, ROOT, connect

ORIGINAL = "Smoke test only. Please reply 'Staging chat OK' and calculate 7 + 5. Do not create, save, update, or record anything"
OBSERVED = 'Smoke test only. Please reply ‘Staging chat OK’ and calculate 7 + 5. Do not create, save, update, or record anything in Factory Ledger.'
SCENARIOS = [
    ('original_chat', ORIGINAL, 'unsupported_request'),
    ('observed_chat', OBSERVED, 'unsupported_request'),
    ('spanish_chat', 'Solo una prueba. Calcula 7 + 5. No crees, guardes, actualices ni registres nada.', 'unsupported_request'),
    ('inventory', 'Check inventory for the product named SMOKE-TEST-NO-SUCH-ITEM. This is a read-only lookup. Do not create, save, update, or record any inventory transactions.', 'read'),
    ('today_entries', 'What did I enter today?', 'read'),
    ('missing_receipt', 'Look up receipt RCV-000000-999. Do not record anything.', 'lookup_failed'),
]


def check(audio_file):
    key_file = Path.home() / 'Documents/fl-secrets/staging-michael-key.txt'
    assert not key_file.stat().st_mode & 0o077, 'Staging key file must be private'
    evidence = {'validated_at': datetime.now(ZoneInfo('America/New_York')).isoformat(),
                'base_url': BASE, 'scenarios': {}, 'production_accessed': False}
    # Guard the known staging database before any HTTP request.
    with connect() as conn, conn.cursor() as cur:
        cur.execute('SET TRANSACTION READ ONLY')
        cur.execute("SELECT name FROM migration_markers WHERE name='068_fl_assistant'")
        assert cur.fetchone()
    with httpx.Client(base_url=BASE, headers={'X-API-Key': key_file.read_text().strip()}, timeout=570) as http:
        identity = http.get('/auth/whoami')
        assert identity.status_code == 200
        actor = identity.json()['actor']
        assert actor['name'] == 'Michael (staging)' and actor['role'] == 'owner'
        evidence['actor'] = actor['name']
        for filename in ['fl-assistant.js', 'fl-assistant.css']:
            asset = http.get('/dashboard/' + filename)
            assert asset.status_code == 200
            assert asset.content == (ROOT / 'dashboard' / filename).read_bytes()
            evidence.setdefault('asset_sha256', {})[filename] = sha256(asset.content).hexdigest()
        for name, text, expected in SCENARIOS:
            response = http.post('/assistant/session')
            assert response.status_code == 200
            sid = response.json()['session_id']
            body = {'session_id': sid, 'turn_id': str(uuid4()), 'text': text}
            response = http.post('/assistant/turn', json=body)
            assert response.status_code == 200
            result = response.json()
            assert [c['kind'] for c in result['cards']] == [expected], name
            assert http.post('/assistant/turn', json=body).json() == result
            with connect() as conn, conn.cursor() as cur:
                cur.execute('SET TRANSACTION READ ONLY')
                cur.execute('SELECT state FROM assistant_sessions WHERE id=%s AND actor_id=%s', (sid, actor['id']))
                history = cur.fetchone()[0]['history']
                calls = [{'name': c['name'], 'arguments': json.loads(c['arguments'])}
                         for c in history if c.get('type') == 'function_call']
                assert not any(c['name'].startswith('prepare_') for c in calls)
                if expected == 'unsupported_request':
                    assert calls == [{'name': 'unsupported_request', 'arguments': {}}]
                cur.execute('SELECT count(*) FROM assistant_drafts WHERE session_id=%s', (sid,))
                assert cur.fetchone()[0] == 0
            evidence['scenarios'][name] = {'status': 'pass', 'session_id': sid, 'turn_id': body['turn_id'],
                'kind': expected, 'model_calls': calls, 'replay_identical': True, 'draft_count': 0}
            print(name + ': pass', flush=True)
        # Transcription must not submit an assistant turn or prepare a ledger draft.
        with connect() as conn, conn.cursor() as cur:
            cur.execute('SET TRANSACTION READ ONLY')
            cur.execute('SELECT count(*) FROM assistant_turns t JOIN assistant_sessions s ON s.id=t.session_id WHERE s.actor_id=%s', (actor['id'],))
            before = cur.fetchone()[0]
        audio = Path(audio_file)
        response = http.post('/assistant/transcribe', files={'file': (audio.name, audio.read_bytes(), 'audio/mp4')})
        assert response.status_code == 200
        transcript = response.json()
        assert transcript['editable'] and transcript['sent'] is False
        assert re.search(r'\b(two|2) cases\b', transcript['text'], re.IGNORECASE)
        with connect() as conn, conn.cursor() as cur:
            cur.execute('SET TRANSACTION READ ONLY')
            cur.execute('SELECT count(*) FROM assistant_turns t JOIN assistant_sessions s ON s.id=t.session_id WHERE s.actor_id=%s', (actor['id'],))
            assert cur.fetchone()[0] == before
        evidence['scenarios']['dictation'] = {'status': 'pass', **transcript, 'assistant_turns_unchanged': True}
        print('dictation: pass (editable, unsent)', flush=True)
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio-file', required=True)
    parser.add_argument('--output', default='/tmp/fl-f1chat-smoke.json')
    args = parser.parse_args()
    try:
        evidence = check(args.audio_file)
        Path(args.output).write_text(json.dumps(evidence, indent=2) + '\n')
        print('Staging request smoke passed. Evidence: ' + args.output)
    except Exception as exc:
        print('Staging request smoke failed: ' + type(exc).__name__, file=sys.stderr)
        raise SystemExit(1)
