"""Classify permanent provider billing outages and record an owner notice."""
import logging
from psycopg2.extras import Json

CODE = 'ASSISTANT_QUOTA_EXHAUSTED'
MESSAGE = 'The assistant is offline right now. Use Record entry or Office to record, and tell the office.'
MESSAGE_ES = 'El asistente no está disponible ahora. Use Registrar entrada u Oficina para registrar y avise a la oficina.'
logger = logging.getLogger(__name__)


def quota_exhausted(status, payload):
    if status not in (400, 402, 403, 429):
        return False
    error = payload.get('error', payload) if isinstance(payload, dict) else {}
    if not isinstance(error, dict):
        return False
    values = ' '.join(str(error.get(key) or '') for key in ('code', 'type', 'message')).lower()
    if 'billing' in ' '.join(str(error.get(key) or '') for key in ('code', 'type')).lower():
        return True
    return any(word in values for word in (
        'insufficient_quota', 'credit_balance_exhausted', 'billing_hard_limit',
        'billing_not_active', 'billing limit', 'no credits', 'credits remaining',
        'credit balance', 'exceeded your current quota', 'quota exhausted'))


def offline_card(language='en'):
    es = language == 'es'
    return {'kind': 'handoff', 'error_code': CODE, 'message': MESSAGE_ES if es else MESSAGE,
            'message_en': MESSAGE, 'message_es': MESSAGE_ES, 'actions': [
                {'href': '/dashboard/entry.html', 'label': 'Registrar entrada' if es else 'Record entry'},
                {'href': '/dashboard/office.html', 'label': 'Oficina' if es else 'Office'}]}


def record_outage(api, actor):
    """One durable open owner alert per outage; never log provider bodies/keys."""
    logger.error('ASSISTANT_QUOTA_EXHAUSTED: provider credits/quota/billing exhausted; owner action required')
    try:
        with api.get_transaction() as cur:
            cur.execute("SELECT pg_advisory_xact_lock(hashtext('assistant-credit-outage'))")
            cur.execute("SELECT id FROM exceptions WHERE kind='ASSISTANT_OFFLINE' AND status IN ('open','escalated') LIMIT 1")
            previous = cur.fetchone()
            if previous:
                return previous['id']
            cur.execute("SELECT id FROM actors WHERE role='owner' AND active ORDER BY id LIMIT 1")
            owner = cur.fetchone()
            cur.execute("""INSERT INTO exceptions(kind,severity,owner_actor_id,detail)
                VALUES ('ASSISTANT_OFFLINE','block',%s,%s) RETURNING id""", (
                    owner['id'] if owner else None, Json({'code': CODE, 'message': MESSAGE,
                        'message_es': MESSAGE_ES, 'entered_by_actor_id': actor.get('id'),
                        'follow_up': 'The owner must restore assistant provider billing or credits, then acknowledge this notice.'})))
            return cur.fetchone()['id']
    except Exception:
        # The worker must still receive offline guidance if the DB is unavailable.
        logger.exception('ASSISTANT_QUOTA_EXHAUSTED: could not persist owner exception notice')
        return None
