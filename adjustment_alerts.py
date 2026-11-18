"""Owner email for corrections over 500 lb (owner decision 2026-11-18, migration 096).

Nothing over 500 lb is held for approval any more: the correction posts and the
owner is told by email. The email must never get in the way of the correction, so
the two are separated:

* `queue_if_large` runs in the posting transaction, inside a SAVEPOINT, and only
  writes one `adjustment_alerts` row ("an email is owed for transaction N"). Any
  failure rolls back the savepoint alone, is logged, and the correction posts.
* `deliver` runs later on its own connections, one alert = one email: claim a
  pending row, send its email through the Resend setup in `reconciliation.py`,
  record the outcome. A failed send leaves the row pending with the error and the
  attempt count; nothing is ever undone.
* Retry: the web process tries a few times after each large correction
  (`RETRY_DELAYS`), and the nightly cron (`scripts/exceptions_sweep.py`) sends
  whatever is still pending. The web process only sends when `RESEND_API_KEY` and
  `ALERT_EMAIL_TO` are set on that service; otherwise the nightly job does it.

A retry never sends a second copy (Codex review 5964179039 #2):

* Each row has a fixed `idempotency_key`, sent on every attempt as Resend's
  `Idempotency-Key` header. For 24 hours Resend answers a repeat of the same key
  and body with its first answer and sends nothing again.
* The email (`message`: subject, text, html) is built from the ledger at the first
  attempt, stored on the row, and never rebuilt, so every attempt sends the same
  body. A photo or a void that arrives later is on the receipt, not in the email.
* A row becomes `sent` only when Resend confirms (a 2xx answer). A send that was
  interrupted — no answer, a crash, the status write failing — leaves the row
  pending; the next attempt repeats the same key and Resend returns the original
  answer, which is then recorded.
* `unconfirmed_since` is the start of the earliest attempt whose answer never
  arrived. It is set when an attempt starts, before the send, so a crash cannot
  lose it; it is cleared only when Resend answers that attempt with a refusal
  (4xx: nothing went out). Resend forgets a key after 24 hours, so a row that is
  still unconfirmed after `CONFIRM_WINDOW` (23 hours) is marked `unconfirmed` and
  NOT sent again: the choice is one email at most, and a person checks it
  (`ADJUSTMENT_ALERT_EMAIL unconfirmed` in the log; the nightly job prints the
  count). Refusals with nothing in doubt are retried without any limit.
"""
import json
import logging
import os
import threading
import time
from html import escape

import exceptions_enforcement as a3b
import reconciliation

logger = logging.getLogger(__name__)

RETRY_DELAYS = (5, 60, 300, 900)    # seconds between the web process's passes after a large correction
CLAIM_COOLDOWN = '30 seconds'       # a row is not claimed again sooner (covers one send, 20 s timeout)
CONFIRM_WINDOW = '23 hours'         # Resend keeps an Idempotency-Key for 24 hours
BATCH = 25                          # alerts (emails) per pass
UNCONFIRMED_ERROR = ('No answer confirmed this email within 23 hours of the first attempt. It was not sent again, '
                     'so the owner cannot get it twice; check whether it arrived.')
DEFAULT_DASHBOARD_URL = 'https://cns-factory-ledger.netlify.app'
_worker = {'lock': threading.Lock(), 'thread': None, 'again': False}


# ---------------------------------------------------------------------------
# 1. In the posting transaction: record that an email is owed. Never raises.
# ---------------------------------------------------------------------------
def queue_if_large(api, cur, transaction_id, action, product, quantity, uom=None):
    """Queue the owner email when this correction is over 500 lb. Returns True when queued.

    Called by the adjust and found posting cores right after the ledger rows are
    written. Off until migration 096's marker exists. Everything happens inside a
    savepoint: whatever goes wrong here, the correction itself is untouched."""
    queued = False
    try:
        cur.execute('SAVEPOINT adjustment_alert_sp')
    except Exception:
        logger.exception('ADJUSTMENT_ALERT_QUEUE_FAILED transaction_id=%s (correction preserved)', transaction_id)
        return False
    try:
        if a3b.destination_rules(cur):
            _, lb = a3b.correction_lb(api, action, product, quantity, uom)
            if a3b.is_large(lb):
                cur.execute('''INSERT INTO adjustment_alerts(transaction_id, action, lb) VALUES (%s,%s,%s)
                               ON CONFLICT (transaction_id) DO NOTHING''', (transaction_id, action, lb))
                queued = True
        cur.execute('RELEASE SAVEPOINT adjustment_alert_sp')
    except Exception:
        queued = False
        logger.exception('ADJUSTMENT_ALERT_QUEUE_FAILED transaction_id=%s (correction preserved)', transaction_id)
        try:
            cur.execute('ROLLBACK TO SAVEPOINT adjustment_alert_sp')
            cur.execute('RELEASE SAVEPOINT adjustment_alert_sp')
        except Exception:
            logger.exception('ADJUSTMENT_ALERT_QUEUE_FAILED could not restore the savepoint, transaction_id=%s', transaction_id)
        return False
    if queued:
        logger.info('ADJUSTMENT_ALERT queued transaction_id=%s action=%s', transaction_id, action)
        try:
            kick(api)
        except Exception:
            logger.exception('ADJUSTMENT_ALERT could not start the sender; the nightly job will send it')
    return queued


# ---------------------------------------------------------------------------
# 2. The email, built from the ledger once, at the first attempt.
# ---------------------------------------------------------------------------
def _lb(value):
    text = f'{abs(float(value)):,.2f}'.rstrip('0').rstrip('.')
    return ('−' if float(value) < 0 else '+') + text + ' lb'


def facts(cur, alert, env=None):
    """Product, lot, pounds, person, reason, destination and photo for one alert."""
    env = os.environ if env is None else env
    cur.execute('''SELECT t.id, t.receipt_number, t.ticket_id, t.adjust_reason, t.reason_code, t.notes, t.operator_id,
                          t.created_at, a.name AS actor_name, r.label_en AS reason_label, r.removal_destination,
                          ct.effective_status
                   FROM transactions t
                   LEFT JOIN actors a ON a.id = t.entered_by_actor_id
                   LEFT JOIN correction_reasons r ON r.code = t.reason_code
                   LEFT JOIN ledger_current_transactions ct ON ct.id = t.id
                   WHERE t.id = %s''', (alert['transaction_id'],))
    tx = cur.fetchone()
    if not tx:
        return None
    cur.execute('''SELECT tl.quantity_lb, p.name AS product_name, l.lot_code
                   FROM transaction_lines tl JOIN products p ON p.id = tl.product_id
                   LEFT JOIN lots l ON l.id = tl.lot_id
                   WHERE tl.transaction_id = %s ORDER BY tl.id''', (tx['id'],))
    lines = cur.fetchall()
    removed = float(alert['lb']) < 0
    order = None
    if tx['ticket_id'] is not None:
        cur.execute('''SELECT so.order_number, c.name AS customer_name
                       FROM write_tickets w
                       JOIN sales_orders so ON so.id = NULLIF(w.payload->>'sales_order_id', '')::int
                       LEFT JOIN customers c ON c.id = so.customer_id
                       WHERE w.id = %s''', (tx['ticket_id'],))
        order = cur.fetchone()
    cur.execute("SELECT count(*) AS n FROM attachments WHERE transaction_id = %s AND kind = 'correction_photo'", (tx['id'],))
    photos = cur.fetchone()['n']
    reason = tx['reason_label'] or tx['adjust_reason'] or tx['reason_code'] or 'not given'
    if not removed:
        destination = 'Not applicable: stock was added'
    elif order:
        destination = 'Sales order ' + order['order_number'] + (f' ({order["customer_name"]})' if order['customer_name'] else '')
    elif tx['removal_destination']:
        destination = reason
    else:
        destination = 'None: recorded as ' + reason
    base = (env.get('DASHBOARD_URL', '').strip() or DEFAULT_DASHBOARD_URL).rstrip('/')
    link = f'{base}/receipt.html?number={tx["receipt_number"]}' if tx['receipt_number'] else None
    note = (tx['notes'] or '').strip()
    return {
        'transaction_id': tx['id'], 'receipt_number': tx['receipt_number'],
        'product': ', '.join(dict.fromkeys(line['product_name'] for line in lines)) or 'unknown product',
        'lot': ', '.join(dict.fromkeys(line['lot_code'] or '?' for line in lines)) or 'unknown lot',
        'change': _lb(alert['lb']) + (' (removed)' if removed else ' (added)'),
        'person': reconciliation._who(tx['actor_name'], tx['operator_id']) or 'No name on it (shared login)',
        'reason': reason, 'destination': destination, 'note': note or None,
        'photo': (f'Attached. Open the receipt to see it: {link}' if link else 'Attached') if photos else 'None attached',
        'receipt_link': link,
        'entered': tx['created_at'].astimezone(a3b.PLANT_TIMEZONE).strftime('%b %-d, %Y %-I:%M %p ET'),
        'still_counts': tx['effective_status'] in (None, 'posted'),
    }


_ROWS = (('Product', 'product'), ('Lot', 'lot'), ('Change', 'change'), ('Person', 'person'), ('Reason', 'reason'),
         ('Destination', 'destination'), ('Note', 'note'), ('Photo', 'photo'), ('Receipt', 'receipt_number'),
         ('Entered', 'entered'))


def compose(item, env=None):
    """{'subject', 'text', 'html'} for the email about one alert (`item`: its `facts`)."""
    env = os.environ if env is None else env
    stage = env.get('ENVIRONMENT', '').strip().lower()
    tag = '' if stage in ('', 'production') else f'[{stage}] '
    subject = f'{tag}Large inventory adjustment: {item["change"].split(" (")[0]} {item["product"]}, lot {item["lot"]}'
    intro = (f'An inventory adjustment over {int(a3b.LARGE_CORRECTION_LB)} lb was recorded. It counted in stock right away; '
             'nothing is waiting for your approval.')
    rows = [(label, item[key]) for label, key in _ROWS if item.get(key)]
    if not item['still_counts']:
        rows.append(('Status', 'This entry has since been voided.'))
    text = '\n'.join([intro, ''] + [f'{label}: {value}' for label, value in rows] + [''])
    html = (f'<p style="font:14px sans-serif">{escape(intro)}</p>\n'
            '<table style="font:14px sans-serif;border-collapse:collapse;margin:0 0 16px">' + ''.join(
                f'<tr><td style="padding:2px 12px 2px 0;color:#555">{escape(label)}</td><td style="padding:2px 0">{escape(str(value))}</td></tr>'
                for label, value in rows) + '</table>')
    return {'subject': subject, 'text': text, 'html': html}


# ---------------------------------------------------------------------------
# 3. Deliver what is pending. Never raises; a failure leaves the row pending.
# ---------------------------------------------------------------------------
def configured(env=None):
    env = os.environ if env is None else env
    return bool(env.get('RESEND_API_KEY', '').strip() and env.get('ALERT_EMAIL_TO', '').strip())


def _claim(transaction, env):
    """Start one send attempt on the next due alert, in its own transaction.

    None when nothing is due. Otherwise the alert's id plus either what to send
    ({'key', 'message', 'attempt', 'fresh'}) or why nothing is sent ({'error', ...}).
    The attempt is on record — count, time, `unconfirmed_since` — before any email
    leaves, so an interrupted send is still known to be in doubt afterwards."""
    with transaction() as cur:
        cur.execute("""SELECT id, transaction_id, action, lb, idempotency_key, message, unconfirmed_since,
                              (unconfirmed_since IS NOT NULL
                               AND unconfirmed_since <= clock_timestamp() - %s::interval) AS lapsed
                       FROM adjustment_alerts
                       WHERE status = 'pending' AND (last_attempt_at IS NULL
                             OR last_attempt_at < clock_timestamp() - %s::interval)
                       ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED""", (CONFIRM_WINDOW, CLAIM_COOLDOWN))
        alert = cur.fetchone()
        if alert is None:
            return None
        if alert['lapsed']:
            # Resend may no longer remember the key: another send could be a second copy.
            cur.execute("""UPDATE adjustment_alerts SET status = 'unconfirmed', last_attempt_at = clock_timestamp(),
                               last_error = %s WHERE id = %s""", (UNCONFIRMED_ERROR, alert['id']))
            logger.error('ADJUSTMENT_ALERT_EMAIL unconfirmed alert=%s transaction_id=%s: not sent again, check by hand',
                         alert['id'], alert['transaction_id'])
            return {'id': alert['id'], 'unconfirmed': True, 'error': UNCONFIRMED_ERROR}
        message = alert['message']
        if message is None:
            item = facts(cur, alert, env)
            if item is None:
                error = 'The ledger entry for this alert was not found'
                cur.execute("""UPDATE adjustment_alerts SET attempts = attempts + 1, last_attempt_at = clock_timestamp(),
                                   last_error = %s WHERE id = %s""", (error, alert['id']))
                return {'id': alert['id'], 'error': error}
            message = compose(item, env)
        cur.execute("""UPDATE adjustment_alerts SET attempts = attempts + 1, last_attempt_at = clock_timestamp(),
                           message = %s::jsonb, unconfirmed_since = COALESCE(unconfirmed_since, clock_timestamp())
                       WHERE id = %s RETURNING attempts""", (json.dumps(message), alert['id']))
        return {'id': alert['id'], 'key': alert['idempotency_key'], 'message': message,
                'attempt': cur.fetchone()['attempts'], 'fresh': alert['unconfirmed_since'] is None}


def _record(transaction, claim, outcome):
    """Write what Resend said about one attempt. Returns 'sent', 'refused' or 'unanswered'."""
    answered = outcome.get('http_status')
    with transaction() as cur:
        if outcome.get('status') == 'sent':
            cur.execute("""UPDATE adjustment_alerts SET status = 'sent', sent_at = clock_timestamp(), email_to = %s,
                               provider_id = %s, last_error = NULL, unconfirmed_since = NULL
                           WHERE id = %s AND status <> 'sent' """, (outcome.get('to'), outcome.get('provider_id'), claim['id']))
            return 'sent'
        # A 4xx answer is Resend refusing THIS attempt: nothing went out because of it.
        # That settles the doubt only if no earlier attempt is still in doubt (`fresh`)
        # and no other worker has started one since (`attempts` unchanged).
        refused = answered is not None and 400 <= int(answered) < 500
        cur.execute("""UPDATE adjustment_alerts SET last_error = %s,
                           unconfirmed_since = CASE WHEN %s AND attempts = %s THEN NULL ELSE unconfirmed_since END
                       WHERE id = %s AND status = 'pending' """,
                    ((outcome.get('error') or 'send failed')[:500], bool(refused and claim['fresh']), claim['attempt'], claim['id']))
        return 'refused' if refused else 'unanswered'


def deliver(transaction, env=None, send=None):
    """One pass: for each due alert (up to BATCH) send its one email and record the result.

    `transaction` is a zero-argument context manager that yields a dict cursor and
    commits on exit (`api.get_transaction` in the app). `send(message, env,
    idempotency_key=...)` defaults to the Resend sender. Returns a summary dict:
    status is unavailable | not_configured | nothing_pending | sent | failed | error;
    `unanswered` counts attempts whose outcome is not known (worth another pass soon)."""
    send = send or reconciliation.send_email_once
    ids, sent, unconfirmed, unanswered, error, in_flight = [], [], [], 0, None, False
    try:
        if not configured(env):
            return {'status': 'not_configured', 'count': 0}
        with transaction() as cur:
            cur.execute("SELECT to_regclass('public.adjustment_alerts') AS relation")
            if cur.fetchone()['relation'] is None:
                return {'status': 'unavailable', 'count': 0}
        for _ in range(BATCH):
            claim = _claim(transaction, env)
            if claim is None:
                break
            ids.append(claim['id'])
            if 'key' not in claim:                       # nothing to send for this row
                error = claim['error']
                if claim.get('unconfirmed'):
                    unconfirmed.append(claim['id'])
                continue
            in_flight = True
            outcome = send(claim['message'], env, idempotency_key=claim['key'])
            result = _record(transaction, claim, outcome)
            in_flight = False
            if result == 'sent':
                sent.append(claim['id'])
                logger.info('ADJUSTMENT_ALERT_EMAIL sent alert=%s attempt=%s', claim['id'], claim['attempt'])
            else:
                error = outcome.get('error') or 'send failed'
                unanswered += result == 'unanswered'
                logger.warning('ADJUSTMENT_ALERT_EMAIL failed alert=%s attempt=%s %s error=%s (will retry, same key)',
                               claim['id'], claim['attempt'], result, error)
        status = 'nothing_pending' if not ids else 'sent' if len(sent) == len(ids) else 'failed'
        return {'status': status, 'count': len(ids), 'alert_ids': ids, 'sent': len(sent), 'unconfirmed': len(unconfirmed),
                'unanswered': unanswered, 'error': error}
    except Exception as exc:   # never the caller's problem; the row stays pending and in doubt, and the retry reuses its key
        logger.error('ADJUSTMENT_ALERT_EMAIL error alerts=%s (%s); will retry', ids, type(exc).__name__)
        return {'status': 'error', 'count': 0, 'alert_ids': ids, 'sent': len(sent), 'unconfirmed': len(unconfirmed),
                'unanswered': unanswered + in_flight, 'error': type(exc).__name__}


# ---------------------------------------------------------------------------
# 4. The web process's sender: one background thread, a few spaced passes.
# ---------------------------------------------------------------------------
def kick(api, delays=RETRY_DELAYS, sleep=time.sleep):
    """Start (or extend) the background sender. Returns False when this service
    has no email settings — then the nightly job sends the alert."""
    if not configured():
        return False
    with _worker['lock']:
        if _worker['thread'] is not None and _worker['thread'].is_alive():
            _worker['again'] = True
            return True
        _worker['again'] = False
        _worker['thread'] = threading.Thread(target=_run, args=(api, delays, sleep), daemon=True,
                                             name='adjustment-alerts')
        _worker['thread'].start()
    return True


def _run(api, delays, sleep):
    while True:
        for delay in delays:
            sleep(delay)
            deliver(api.get_transaction)
        with _worker['lock']:
            if not _worker['again']:
                _worker['thread'] = None
                return
            _worker['again'] = False
