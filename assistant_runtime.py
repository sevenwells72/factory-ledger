"""Request-local chat execution and plain, live progress; no business decisions."""
import asyncio
import json

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from starlette.concurrency import run_in_threadpool


def progress(request, stage, query='', language='en'):
    callback = request.scope.get('_fl_chat_progress')
    if not callback:
        return
    es = language == 'es'
    query = str(query or '').strip()[:100]
    messages = {
        'read': ('Reading your message…', 'Leyendo tu mensaje…'),
        'lookup': (f'Looking up {query}…' if query else 'Looking that up…',
                   f'Buscando {query}…' if query else 'Buscando…'),
        'lots': ('Checking lots…', 'Revisando lotes…'),
        'draft': ('Preparing draft…', 'Preparando el borrador…'),
        'today': ('Loading today’s entries…', 'Cargando las entradas de hoy…'),
    }
    callback({'type': 'progress', 'message': messages[stage][es]})


async def run_turn(handler, request):
    """Keep synchronous PostgreSQL waits off the server's event loop.

    Each request retains its own event loop inside the worker. Internal async
    FL routes and OpenAI calls share that request's verified actor and state.
    """
    def work():
        return asyncio.run(handler())

    if 'application/x-ndjson' not in request.headers.get('accept', ''):
        return await run_in_threadpool(work)

    loop = asyncio.get_running_loop()
    events = asyncio.Queue()

    def emit(event):
        loop.call_soon_threadsafe(events.put_nowait, event)

    request.scope['_fl_chat_progress'] = emit

    async def produce():
        try:
            result = await run_in_threadpool(work)
            await events.put({'type': 'result', 'result': jsonable_encoder(result)})
        except HTTPException as error:
            await events.put({'type': 'error', 'status': error.status_code, 'detail': error.detail})
        except Exception:
            await events.put({'type': 'error', 'status': 500,
                              'detail': 'Factory Ledger is temporarily unavailable — try again in a minute.'})
        finally:
            await events.put(None)

    async def chunks():
        producer = asyncio.create_task(produce())
        try:
            while True:
                event = await events.get()
                if event is None:
                    break
                yield json.dumps(event, ensure_ascii=False, default=str) + '\n'
        finally:
            # A disconnected client must not abandon an actor-bound draft or
            # leave a lease stuck. Retrying uses the same durable turn ID.
            await asyncio.shield(producer)

    return StreamingResponse(chunks(), media_type='application/x-ndjson',
                             headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})


def history_result(tool, result):
    """A day read stays in its UI; the model retains only bounded counts."""
    if tool == 'today_entries' and isinstance(result, dict):
        return {key: result[key] for key in ('date', 'own_count', 'team_count') if key in result}
    return result


def compact_history(history):
    """Repair older saved day outputs while preserving call/output pairing."""
    day_calls = {item.get('call_id') for item in history
                 if item.get('type') == 'function_call' and item.get('name') == 'today_entries'}
    for item in history:
        if item.get('type') == 'function_call_output' and item.get('call_id') in day_calls:
            try:
                result = json.loads(item.get('output', '{}'))
            except (ValueError, TypeError):
                continue
            item['output'] = json.dumps(history_result('today_entries', result), ensure_ascii=False, default=str)


def day_preview(receipts, limit=20):
    """Keep a small readable day preview; full entry evidence stays in receipts.

    The Today button reads the complete fast summary directly. Neither model
    history nor persisted chat responses need hundreds of full ticket drafts.
    """
    result = []
    draft_fields = ('product_name', 'target_product_name', 'cases', 'total_cases',
                    'total_lb', 'total_output_lb', 'total_weight_lb', 'quantity',
                    'current_quantity_lb', 'new_balance_lb', 'delta_lb')
    for receipt in receipts[:limit]:
        row = {key: receipt.get(key) for key in
               ('receipt_number', 'action', 'entered_at', 'happened_at', 'actor', 'effective_status')}
        draft = receipt.get('draft') or {}
        row['draft'] = {key: draft[key] for key in draft_fields if key in draft}
        lines = [line for transaction in receipt.get('transactions', []) for line in transaction.get('lines', [])]
        outputs = [line for line in lines if float(line.get('quantity_lb') or 0) > 0]
        relevant = outputs or lines
        row['transactions'] = [{'lines': [{key: line.get(key) for key in ('product_name', 'quantity_lb')} for line in relevant]}] if relevant else []
        result.append(row)
    return result
