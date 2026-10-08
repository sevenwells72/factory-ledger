"""F1 transport: FL owns every business decision; only a human Record commits.

No network hop to another FL deployment, process-local ticket store, model prose
renderer, hosted MCP, browser OpenAI key, or model-accessible commit tool.
"""
import asyncio
import hashlib
import json
import logging
import os
import re
from pathlib import Path
from uuid import UUID, uuid4

import httpx
from fastapi import Depends, File, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field, StrictInt, StrictStr
from psycopg2.extras import Json

import assistant_tools as catalog
import write_tickets

ROUTES = ('session', 'resume', 'turn', 'record', 'cancel', 'confirm-sku', 'attachment', 'attachment/read', 'transcribe')
ACTOR_ROUTES = frozenset({('POST', '/assistant/' + p) for p in ROUTES} |
                         {('GET', '/correction-reasons')})
MAX_PHOTO = 5 * 1024 * 1024
MAX_AUDIO = 20 * 1024 * 1024
MAX_STEPS = 8
MODEL = 'gpt-6.1-sol'  # PoC choice; service may explicitly override it.
TRANSCRIBE_MODEL = 'gpt-transcribe'
logger = logging.getLogger(__name__)


def fail(status, code, message):
    raise HTTPException(status, {'error_code': code, 'message': message})


class Body(BaseModel):
    class Config:
        extra = 'forbid'


class SessionBody(Body):
    session_id: UUID


class TurnBody(SessionBody):
    turn_id: UUID
    text: str = Field('', max_length=4000)
    choice_id: UUID | None = None
    selected_id: StrictInt | StrictStr | None = None
    next_page: bool = False
    attachment_ids: list[UUID] = Field(default_factory=list, max_items=5)


class DraftBody(Body):
    draft_id: UUID


class RecordBody(DraftBody):
    acknowledged_warnings: list[str] = Field(default_factory=list, max_items=25)
    # Opaque FL evidence; validated only by CommitRequest/A5, never inferred here.
    lot_confirmations: list[dict] = Field(default_factory=list, max_items=100)


class AttachmentRead(Body):
    attachment_id: UUID


async def openai(path, *, body=None, file=None):
    key = os.getenv('OPENAI_API_KEY', '').strip()
    if not key:
        fail(503, 'ASSISTANT_NOT_CONFIGURED', 'OpenAI is unavailable. NOT recorded / NO registrado.')
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=False) as client:
            kwargs = {'json': body} if file is None else {
                'files': {'file': file},
                'data': {'model': os.getenv('ASSISTANT_TRANSCRIBE_MODEL', TRANSCRIBE_MODEL), 'response_format': 'json'}}
            response = await client.post('https://api.openai.com/v1/' + path,
                                         headers={'Authorization': 'Bearer ' + key}, **kwargs)
        if response.status_code != 200:
            logger.warning('OpenAI request failed: status=%s', response.status_code)
            fail(502, 'OPENAI_UNAVAILABLE', 'OpenAI request failed. NOT recorded / NO registrado.')
        return response.json()
    except (httpx.HTTPError, ValueError):
        fail(502, 'OPENAI_UNAVAILABLE', 'OpenAI connection failed. NOT recorded / NO registrado.')


async def relay(app, request, method, path, *, body=None, params=None):
    """Invoke actual FL routes including their auth, validation and error envelopes.

    A small ASGI client avoids logging the ticket in an HTTP client's URL log.
    Destination is the current application, never a configurable remote URL.
    """
    from urllib.parse import urlencode
    content = json.dumps(body, default=str).encode() if body is not None else b''
    headers = [(b'x-api-key', request.headers.get('x-api-key', '').encode()),
               (b'content-type', b'application/json'), (b'content-length', str(len(content)).encode())]
    scope = {'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
             'method': method, 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
             'query_string': urlencode(params or {}).encode(), 'root_path': '',
             'headers': headers, 'server': ('fl-internal', 80), 'client': ('127.0.0.1', 0)}
    messages, sent = [], False
    finished = asyncio.Event()

    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {'type': 'http.request', 'body': content, 'more_body': False}
        await finished.wait()
        return {'type': 'http.disconnect'}

    async def send(message):
        messages.append(message)
        if message['type'] == 'http.response.body' and not message.get('more_body', False):
            finished.set()

    await app(scope, receive, send)
    status = next(m['status'] for m in messages if m['type'] == 'http.response.start')
    raw = b''.join(m.get('body', b'') for m in messages if m['type'] == 'http.response.body')
    try:
        result = json.loads(raw)
    except ValueError:
        fail(502, 'FL_RESPONSE_INVALID', 'FL returned no usable response. NOT recorded / NO registrado.')
    return status, result


def session_row(api, session_id, actor_id, *, lock=False):
    with api.get_transaction() as cur:
        cur.execute('SELECT * FROM assistant_sessions WHERE id=%s AND actor_id=%s' +
                    (' FOR UPDATE' if lock else ''), (str(session_id), actor_id))
        row = cur.fetchone()
    if not row:
        fail(404, 'SESSION_NOT_FOUND', 'Chat not found for this person / Chat no encontrado para esta persona.')
    return row


def draft_row(api, draft_id, actor_id, *, lock=False, cur=None):
    if cur is None:
        with api.get_transaction() as cursor:
            return draft_row(api, draft_id, actor_id, lock=lock, cur=cursor)
    cur.execute('SELECT * FROM assistant_drafts WHERE id=%s AND actor_id=%s' +
                (' FOR UPDATE' if lock else ''), (str(draft_id), actor_id))
    row = cur.fetchone()
    if not row:
        fail(404, 'DRAFT_NOT_FOUND', 'Draft not found for this person / Borrador no encontrado.')
    return row


def approve_id(state, kind, candidate):
    values = state.setdefault('resolved', {}).setdefault(kind, [])
    if candidate.get('id') not in values:
        values.append(candidate['id'])


def untrusted_ids(args, state):
    """Transport provenance only. FL still validates identity and eligibility."""
    kinds = {'product_id': 'product', 'source_product_id': 'product', 'target_product_id': 'product',
             'ingredient_product_id': 'product', 'substitute_product_id': 'product',
             'supplier_id': 'supplier', 'lot_id': 'lot'}
    missing = []

    def walk(value):
        if isinstance(value, dict):
            for name, item in value.items():
                if name in kinds and item is not None and item not in state.get('resolved', {}).get(kinds[name], []):
                    missing.append(name)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(args)
    return missing


def receipt_ok(status, result):
    return (status == 200 and isinstance(result, dict) and result.get('success') is not False
            and isinstance(result.get('receipt_number'), str)
            and re.fullmatch(r'(?:RCV|MK|PK|ADJ|FND)-\d{6}-\d{3,}', result['receipt_number']) is not None)


def register_routes(app, api):
    def authenticate(request: Request, _: bool = Depends(api.verify_api_key)):
        if os.getenv('ASSISTANT_ENABLED') != '1':
            fail(503, 'ASSISTANT_DISABLED', 'FL Assistant is not enabled / FL Assistant no está habilitado.')
        actor = api.request_actor(request)
        if not actor:
            fail(403, 'PERSONAL_IDENTITY_REQUIRED', 'Sign in with your personal key / Usa tu clave personal.')
        return actor

    @app.get('/correction-reasons')
    def correction_reasons(_: bool = Depends(api.verify_api_key)):
        # An interface-neutral FL read, from the A3a fixed catalog.
        with api.get_transaction() as cur:
            cur.execute('SELECT code,label_en,label_es,applies_to,adjust_sign,note_required FROM correction_reasons ORDER BY sort_order,code')
            return {'reasons': [dict(r) for r in cur.fetchall()]}

    @app.get('/dash/fl-assistant', include_in_schema=False)
    def page():
        if os.getenv('ASSISTANT_ENABLED') != '1':
            raise HTTPException(404, 'Not found')
        return FileResponse(Path(__file__).parent / 'dashboard' / 'fl-assistant.html',
                            headers={'Cache-Control': 'no-store'})

    @app.post('/assistant/session')
    def create_session(actor=Depends(authenticate)):
        sid = str(uuid4())
        with api.get_transaction() as cur:
            cur.execute('INSERT INTO assistant_sessions(id,actor_id) VALUES (%s,%s)', (sid, actor['id']))
        return {'session_id': sid, 'actor': {'id': actor['id'], 'name': actor['name'], 'role': actor['role']}}

    @app.post('/assistant/resume')
    def resume(body: SessionBody, actor=Depends(authenticate)):
        row = session_row(api, body.session_id, actor['id'])
        with api.get_transaction() as cur:
            cur.execute('SELECT response FROM assistant_turns WHERE session_id=%s ORDER BY created_at,id', (str(body.session_id),))
            turns = [r['response'] for r in cur.fetchall()]
            cur.execute('SELECT id,status,result,card,record_started_at FROM assistant_drafts WHERE session_id=%s ORDER BY created_at,id', (str(body.session_id),))
            drafts = [dict(r) for r in cur.fetchall()]
        return {'session_id': str(body.session_id), 'turns': turns, 'drafts': drafts,
                'pending': row['state'].get('pending'),
                'actor': {'id': actor['id'], 'name': actor['name'], 'role': actor['role']}}

    async def execute_tool(request, body, actor, state, name, args):
        if name == 'resolve':
            status, result = await relay(app, request, 'POST', '/resolve', body=args)
            if status == 200 and result.get('outcome') == 'match' and not result.get('needs_clarification'):
                approve_id(state, args['kind'], result['match'])
                return status, result, None
            if status == 200:
                choice_id = str(uuid4())
                card = {'kind': 'choices', 'id': choice_id, 'resolution_kind': args['kind'], 'result': result}
                state['pending'] = {'id': choice_id, 'args': args, 'result': result}
                if result.get('outcome') == 'none':
                    with api.get_transaction() as cur:
                        cur.execute('INSERT INTO assistant_unmatched_words(session_id,actor_id,kind,query,context) VALUES (%s,%s,%s,%s,%s)',
                                    (str(body.session_id), actor['id'], args['kind'], args.get('query', ''), Json(args.get('context', {}))))
                return status, result, card
            return status, result, {'kind': 'refusal', 'status': status, 'result': result}
        if name in catalog.PREPARES:
            if set(args) & catalog.UI_FIELDS:
                fail(502, 'MODEL_EVIDENCE_FORBIDDEN', 'Confirmation must come from the page / Confirma en la página.')
            unknown = untrusted_ids(args, state)
            if unknown:
                return 422, {'detail': {'error_code': 'RESOLVE_REQUIRED', 'fields': unknown}}, None
            path, _ = catalog.PREPARES[name]
            status, result = await relay(app, request, 'POST', path, body={**args, 'client_source': 'fl_assistant'})
            if status != 200 or not result.get('ticket') or not result.get('payload_hash') or not isinstance(result.get('draft'), dict):
                return status, result, {'kind': 'refusal', 'status': status, 'result': catalog.safe_result(result)}
            card = store_draft(result, args, body.session_id, actor['id'], body.attachment_ids)
            return status, result, card
        if name == 'today_entries':
            path, params = '/receipts', {'date': api.get_plant_now().date().isoformat(), 'actor': str(actor['id'])}
        elif name == 'inventory_lookup':
            path, params = '/inventory/lookup', args
        elif name == 'receipt_lookup':
            number = args.get('receipt_number', '')
            if not re.fullmatch(r'[A-Z]{2,3}-\d{6}-\d{3,}', number):
                fail(422, 'INVALID_RECEIPT', 'Enter the exact FL receipt number / Escribe el número del recibo FL.')
            path, params = '/receipts/' + number, {}
        elif name == 'shift_summary' and any(getattr(r, 'path', '') == '/reports/shift-summary' for r in app.routes):
            path, params = '/reports/shift-summary', {'date': api.get_plant_now().date().isoformat(), 'actor': 'me'}
        else:
            fail(502, 'TOOL_NOT_ALLOWED', 'No permitted FL tool completed / Ninguna herramienta FL completó.')
        status, result = await relay(app, request, 'GET', path, params=params)
        return status, result, {'kind': 'read' if status == 200 else 'refusal', 'tool': name, 'status': status, 'result': result}

    def store_draft(result, args, sid, actor_id, attachments):
        draft_id = str(uuid4())
        card = {'kind': 'draft', 'id': draft_id, 'prepared': catalog.safe_result(result),
                'attachment_ids': [str(a) for a in attachments]}
        original = {**args, 'occurred_at': result['draft']['happened_at']}
        with api.get_transaction() as cur:
            cur.execute('INSERT INTO assistant_drafts(id,session_id,actor_id,ticket,payload_hash,card,prepare_body) VALUES (%s,%s,%s,%s,%s,%s,%s)',
                        (draft_id, str(sid), actor_id, result['ticket'], result['payload_hash'], Json(card), Json(original)))
        return card

    @app.post('/assistant/confirm-sku')
    async def confirm_sku(body: DraftBody, request: Request, actor=Depends(authenticate)):
        row = draft_row(api, body.draft_id, actor['id'])
        prepared = row['card']['prepared']
        if (row['status'] != 'pending' or row['record_started_at'] or prepared.get('action') != 'make'
                or not any(b.get('code') == 'SKU_CONFIRMATION_REQUIRED' for b in prepared.get('blockers', []))):
            fail(409, 'SKU_CONFIRMATION_UNAVAILABLE', 'Use the current FL draft / Usa el borrador actual de FL.')
        args = {**row['prepare_body'], 'confirmed_sku': True, 'client_source': 'fl_assistant'}
        status, result = await relay(app, request, 'POST', '/make/prepare', body=args)
        if status != 200 or not result.get('ticket'):
            return JSONResponse(status_code=status, content=catalog.safe_result(result))
        card = store_draft(result, args, row['session_id'], actor['id'], row['card'].get('attachment_ids', []))
        with api.get_transaction() as cur:
            cur.execute("UPDATE assistant_drafts SET status='cancelled' WHERE id=%s AND status='pending' AND record_started_at IS NULL", (str(body.draft_id),))
        return {'card': card}

    def renew_lease(sid, lease):
        # A crashed worker holds the chat for at most two minutes. Active turns
        # renew between bounded OpenAI calls; a replaced worker cannot save.
        with api.get_transaction() as cur:
            cur.execute("""UPDATE assistant_sessions SET lease_until=clock_timestamp()+interval '2 minutes'
                WHERE id=%s AND lease_id=%s""", (sid, lease))
            if cur.rowcount != 1:
                fail(409, 'CHAT_BUSY', 'Chat changed; reload / El chat cambió; recarga.')

    @app.post('/assistant/turn')
    async def turn(body: TurnBody, request: Request, actor=Depends(authenticate)):
        sid, tid, lease = str(body.session_id), str(body.turn_id), str(uuid4())
        digest = write_tickets.canonical_hash(jsonable_encoder(body))
        session_row(api, sid, actor['id'])
        with api.get_transaction() as cur:
            cur.execute('SELECT request_hash,response FROM assistant_turns WHERE session_id=%s AND id=%s', (sid, tid))
            previous = cur.fetchone()
            if previous:
                if previous['request_hash'] != digest:
                    fail(409, 'TURN_CHANGED', 'Retry the original message / Reintenta el mensaje original.')
                return previous['response']
            cur.execute("""UPDATE assistant_sessions SET lease_id=%s,lease_until=clock_timestamp()+interval '2 minutes'
                WHERE id=%s AND actor_id=%s AND (lease_until IS NULL OR lease_until < clock_timestamp()) RETURNING state""",
                        (lease, sid, actor['id']))
            claimed = cur.fetchone()
            if not claimed:
                fail(409, 'CHAT_BUSY', 'Wait for the current message / Espera el mensaje actual.')
        state = claimed['state']
        try:
            # Store names/bytes outside model context. Validate every attachment owner.
            with api.get_transaction() as cur:
                for aid in body.attachment_ids:
                    cur.execute('SELECT id FROM assistant_attachments WHERE id=%s AND session_id=%s AND actor_id=%s',
                                (str(aid), sid, actor['id']))
                    if not cur.fetchone():
                        fail(404, 'ATTACHMENT_NOT_FOUND', 'Attachment unavailable / Adjunto no disponible.')
            history = state.setdefault('history', [])
            pending = state.get('pending')
            if body.choice_id:
                if not pending or str(body.choice_id) != pending['id']:
                    fail(409, 'CHOICE_EXPIRED', 'Choose from the current choices / Usa las opciones actuales.')
                if body.next_page:
                    args = {**pending['args'], 'offset': pending['result'].get('next_offset', 0)}
                    if not pending['result'].get('has_more'):
                        fail(422, 'NO_MORE_CHOICES', 'No more choices / No hay más opciones.')
                    status, result, card = await execute_tool(request, body, actor, state, 'resolve', args)
                    cards = [card] if card else [{'kind': 'read', 'tool': 'resolve', 'result': result}]
                    return save_turn(state, cards, body, actor, digest, lease)
                candidates = pending['result'].get('candidates', [])
                selected = next((c for c in candidates if c['id'] == body.selected_id), None)
                if not selected:
                    if body.selected_id is not None:
                        fail(422, 'CHOICE_INVALID', 'Choose a displayed option / Elige una opción mostrada.')
                    state.pop('pending', None)
                    cards = [{'kind': 'clarify', 'message': 'Give a fuller name or exact code / Escribe el nombre completo o código exacto.'}]
                    history.append({'role': 'user', 'content': 'None of these candidates; ask me for a fuller name.'})
                    return save_turn(state, cards, body, actor, digest, lease)
                approve_id(state, pending['args']['kind'], selected)
                state.pop('pending', None)
                history.append({'role': 'user', 'content': 'I selected this FL candidate: ' + json.dumps(selected, ensure_ascii=False)})
            else:
                if not body.text.strip():
                    fail(422, 'TEXT_REQUIRED', 'Enter a message / Escribe un mensaje.')
                state.pop('pending', None)
                history.append({'role': 'user', 'content': body.text.strip()})
            reasons_status, reasons = await relay(app, request, 'GET', '/correction-reasons')
            if reasons_status != 200:
                return save_turn(state, [{'kind': 'refusal', 'status': reasons_status, 'result': reasons}], body, actor, digest, lease)
            functions = catalog.tools(reasons['reasons'], shift_summary=any(getattr(r, 'path', '') == '/reports/shift-summary' for r in app.routes))
            allowed = {f['name'] for f in functions}
            cards = []
            for _ in range(MAX_STEPS):
                renew_lease(sid, lease)
                response = await openai('responses', body={
                    'model': os.getenv('ASSISTANT_MODEL', MODEL), 'store': False,
                    'instructions': catalog.INSTRUCTIONS + '\nCurrent FL plant time (America/New_York): ' + api.get_plant_now().isoformat()
                        + '\nFL reason catalog: ' + json.dumps(reasons['reasons'], ensure_ascii=False),
                    'input': history, 'tools': functions, 'tool_choice': 'required',
                    'parallel_tool_calls': False, 'max_output_tokens': 4000,
                    'reasoning': {'effort': 'low'},
                })
                output = response.get('output', [])
                calls = [c for c in output if c.get('type') == 'function_call']
                if response.get('status') != 'completed' or len(calls) != 1 or calls[0].get('name') not in allowed:
                    fail(502, 'TOOL_REQUIRED', 'No FL tool completed. NOT recorded / NO registrado.')
                call = calls[0]
                try:
                    args = json.loads(call['arguments'])
                    if not isinstance(args, dict):
                        raise ValueError()
                except (KeyError, ValueError, TypeError):
                    fail(502, 'TOOL_ARGUMENTS_INVALID', 'Invalid tool response. NOT recorded / NO registrado.')
                # Retain reasoning items for Responses continuity; discard all model prose.
                history.extend(item for item in output if item.get('type') in ('reasoning', 'function_call'))
                status, result, card = await execute_tool(request, body, actor, state, call['name'], catalog.clean_arguments(args))
                history.append(catalog.output_item(call['call_id'], result))
                if card:
                    if card['kind'] == 'refusal':
                        card['reasons'] = reasons['reasons']
                    cards.append(card)
                    break
            if not cards:
                fail(502, 'TOOL_LIMIT', 'Please give more precise details. NOT recorded / NO registrado.')
            # Bounded transcript, trim only at user-message boundaries to keep call/output pairs.
            while len(json.dumps(history)) > 90000 and sum(x.get('role') == 'user' for x in history) > 1:
                next_user = next(i for i, x in enumerate(history[1:], 1) if x.get('role') == 'user')
                del history[:next_user]
            return save_turn(state, cards, body, actor, digest, lease)
        finally:
            with api.get_transaction() as cur:
                cur.execute('UPDATE assistant_sessions SET lease_id=NULL,lease_until=NULL WHERE id=%s AND lease_id=%s', (sid, lease))

    def save_turn(state, cards, body, actor, digest, lease):
        result = {'turn_id': str(body.turn_id), 'text': body.text, 'cards': cards}
        with api.get_transaction() as cur:
            cur.execute('UPDATE assistant_sessions SET state=%s,updated_at=clock_timestamp() WHERE id=%s AND actor_id=%s AND lease_id=%s',
                        (Json(state), str(body.session_id), actor['id'], lease))
            if cur.rowcount != 1:
                fail(409, 'CHAT_BUSY', 'Chat changed; reload / El chat cambió; recarga.')
            cur.execute('INSERT INTO assistant_turns(session_id,id,request_hash,response) VALUES (%s,%s,%s,%s)',
                        (str(body.session_id), str(body.turn_id), digest, Json(result)))
        return result

    @app.post('/assistant/record')
    async def record(body: RecordBody, request: Request, actor=Depends(authenticate)):
        # Durably mark the attempt BEFORE FL commit. A lost response must never
        # permit Cancel to claim NOT recorded for a ticket that actually posted.
        with api.get_transaction() as cur:
            row = draft_row(api, body.draft_id, actor['id'], lock=True, cur=cur)
            if row['status'] == 'cancelled':
                fail(409, 'DRAFT_CANCELLED', 'Draft cancelled. NOT recorded / Borrador cancelado. NO registrado.')
            prior_attempt = row['record_started_at']
            cur.execute('UPDATE assistant_drafts SET record_started_at=clock_timestamp() WHERE id=%s RETURNING record_started_at', (str(body.draft_id),))
            attempt = cur.fetchone()['record_started_at']
        payload = {'payload_hash': row['payload_hash'], 'acknowledged_warnings': body.acknowledged_warnings}
        if body.lot_confirmations:
            payload['lot_confirmations'] = body.lot_confirmations
        # Always relay even after a saved result: FL re-authenticates and replays.
        status, result = await relay(app, request, 'POST', '/tickets/' + row['ticket'] + '/commit', body=payload)
        if not receipt_ok(status, result):
            if 400 <= status < 500 and prior_attempt is None:
                # Clear only this definitive attempt. A concurrent retry changes
                # the marker; a prior lost response remains uncertain even when
                # a later role check denies replay of an already-posted ticket.
                with api.get_transaction() as cur:
                    cur.execute("""UPDATE assistant_drafts SET record_started_at=NULL
                        WHERE id=%s AND status='pending' AND record_started_at=%s
                        AND NOT EXISTS (SELECT 1 FROM write_tickets
                            WHERE ticket_hash=%s AND status='committed')""",
                        (str(body.draft_id), attempt, write_tickets.token_hash(row['ticket'])))
            return JSONResponse(status_code=status if status >= 400 else 502,
                                content=jsonable_encoder({'kind': 'not_recorded', 'result': catalog.safe_result(result)}))
        with api.get_transaction() as cur:
            cur.execute("UPDATE assistant_drafts SET status='committed',result=%s WHERE id=%s", (Json(result), str(body.draft_id)))
        return {'kind': 'receipt', 'draft_id': str(body.draft_id), 'result': catalog.safe_result(result)}

    @app.post('/assistant/cancel')
    def cancel(body: DraftBody, actor=Depends(authenticate)):
        with api.get_transaction() as cur:
            row = draft_row(api, body.draft_id, actor['id'], lock=True, cur=cur)
            if row['status'] == 'committed':
                fail(409, 'ALREADY_RECORDED', 'This draft already has an FL receipt / Este borrador ya tiene recibo FL.')
            if row['record_started_at']:
                fail(409, 'RECORD_OUTCOME_PENDING', 'Record was attempted. Retry the same ticket to verify the receipt / Reintenta Registrar para verificar el recibo.')
            cur.execute("UPDATE assistant_drafts SET status='cancelled' WHERE id=%s", (str(body.draft_id),))
        return {'kind': 'cancelled', 'draft_id': str(body.draft_id)}

    @app.post('/assistant/attachment')
    async def attachment(session_id: UUID, file: UploadFile = File(...), actor=Depends(authenticate)):
        session_row(api, session_id, actor['id'])
        data = await file.read(MAX_PHOTO + 1)
        await file.close()
        mime = ('image/jpeg' if data.startswith(b'\xff\xd8\xff') else
                'image/png' if data.startswith(b'\x89PNG\r\n\x1a\n') else
                'image/webp' if data[:4] == b'RIFF' and data[8:12] == b'WEBP' else None)
        if not data or len(data) > MAX_PHOTO or not mime or file.content_type != mime:
            fail(422, 'PHOTO_INVALID', 'Use a JPEG, PNG or WebP photo up to 5 MB / Foto de hasta 5 MB.')
        aid = str(uuid4())
        filename = Path(file.filename or 'photo').name[:150]
        with api.get_transaction() as cur:
            # Bound durable storage to 50 MB per conversation, independent of model input.
            cur.execute('SELECT id FROM assistant_sessions WHERE id=%s FOR UPDATE', (str(session_id),))
            cur.execute('SELECT coalesce(sum(octet_length(content)),0) AS size FROM assistant_attachments WHERE session_id=%s', (str(session_id),))
            if cur.fetchone()['size'] + len(data) > 50 * 1024 * 1024:
                fail(413, 'ATTACHMENT_LIMIT', 'Chat attachment limit reached / Límite de adjuntos alcanzado.')
            cur.execute('INSERT INTO assistant_attachments(id,session_id,actor_id,filename,media_type,sha256,content) VALUES (%s,%s,%s,%s,%s,%s,%s)',
                        (aid, str(session_id), actor['id'], filename, mime, hashlib.sha256(data).hexdigest(), data))
        return {'id': aid, 'filename': filename, 'media_type': mime, 'size': len(data), 'attachment_only': True}

    @app.post('/assistant/attachment/read')
    def read_attachment(body: AttachmentRead, actor=Depends(authenticate)):
        with api.get_transaction() as cur:
            cur.execute('SELECT content,media_type FROM assistant_attachments WHERE id=%s AND actor_id=%s', (str(body.attachment_id), actor['id']))
            row = cur.fetchone()
        if not row:
            fail(404, 'ATTACHMENT_NOT_FOUND', 'Attachment unavailable / Adjunto no disponible.')
        return Response(bytes(row['content']), media_type=row['media_type'],
                        headers={'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})

    @app.post('/assistant/transcribe')
    async def transcribe(file: UploadFile = File(...), actor=Depends(authenticate)):
        data = await file.read(MAX_AUDIO + 1)
        await file.close()
        mime = (file.content_type or '').split(';')[0]
        extensions = {'audio/webm': 'webm', 'video/webm': 'webm', 'audio/mp4': 'mp4',
                      'audio/mpeg': 'mp3', 'audio/wav': 'wav', 'audio/ogg': 'ogg'}
        if mime not in extensions or not data or len(data) > MAX_AUDIO:
            fail(422, 'AUDIO_INVALID', 'Audio must be at most 20 MB / Audio de hasta 20 MB.')
        result = await openai('audio/transcriptions', file=('dictation.' + extensions[mime], data, mime))
        if not isinstance(result.get('text'), str) or len(result['text']) > 4000:
            fail(502, 'TRANSCRIPT_INVALID', 'Transcript unavailable; type instead / Escribe el mensaje.')
        return {'text': result['text'], 'editable': True, 'sent': False}
