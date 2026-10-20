"""Tell failures a DB-pool wait timeout caused apart from real failures.

Every PoolTimeout carries a random wait id. The pool records each id it raises
on the current request, so a 5xx answer is turned into a retryable "busy"
only when it is that timeout, an exception explicitly raised from it
(`raise ... from exc`), or an error whose text quotes its message — the way
`except Exception as e: return JSONResponse({"error": str(e)}, 500)` handlers
answer. A request that merely waited earlier and then failed for another
reason keeps its real error (PR #141 review, comment 6294160112).
"""
import contextvars
import secrets

from psycopg2 import pool

BUSY_MESSAGE = 'FL is busy right now — try again in a moment.'
BUSY_MESSAGE_ES = 'FL está ocupado en este momento — inténtalo de nuevo en un momento.'

# A set shared by reference: sync handlers run in a worker thread on a copy of
# the request's context, and still add to the same set.
_request_wait_ids = contextvars.ContextVar('fl_request_pool_wait_ids', default=None)


class PoolTimeout(pool.PoolError):
    """getconn found no free connection within the wait limit."""

    def __init__(self, message):
        self.wait_id = 'pool-wait-' + secrets.token_hex(6)
        super().__init__(f'{message} ({self.wait_id})')
        ids = _request_wait_ids.get()
        if ids is not None:
            ids.add(self.wait_id)


def begin_request():
    """Start collecting wait ids for one request; returns (reset token, ids)."""
    ids = set()
    return _request_wait_ids.set(ids), ids


def end_request(token):
    _request_wait_ids.reset(token)


def _texts(error):
    yield str(error)
    detail = getattr(error, 'detail', None)
    if detail is not None:
        yield str(detail)


def caused_by_timeout(error, ids=None):
    """True when `error` is a pool timeout or was raised from / quotes one.

    Only the explicit cause chain counts: an exception merely raised while a
    timeout was being handled (`__context__`) is a different failure."""
    ids = _request_wait_ids.get() if ids is None else ids
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, PoolTimeout):
            return True
        if ids and any(wait_id in text for text in _texts(error) for wait_id in ids):
            return True
        error = error.__cause__
    return False


def body_mentions_timeout(body, ids):
    """True when a response body quotes a timeout raised for this request."""
    return any(wait_id.encode('ascii') in body for wait_id in ids)


def busy_detail():
    return {'error_code': 'DATABASE_BUSY', 'message': BUSY_MESSAGE, 'message_es': BUSY_MESSAGE_ES,
            'retryable': True}
