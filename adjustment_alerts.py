"""Owner email for corrections over 500 lb (owner decision 2026-11-18, migration 095).

Nothing over 500 lb is held for approval any more: the correction posts and the
owner is told by email. The email must never get in the way of the correction, so
the two are separated:

* `queue_if_large` runs in the posting transaction, inside a SAVEPOINT, and only
  writes one `adjustment_alerts` row ("an email is owed for transaction N"). Any
  failure rolls back the savepoint alone, is logged, and the correction posts.
* `deliver` runs later on its own connections: claim the pending rows, build the
  email from the ledger as it is now, send it through the Resend setup in
  `reconciliation.send_email`, record the outcome. A failed send leaves the row
  pending with the error and the attempt count; nothing is ever undone.
* Retry: the web process tries a few times after each large correction
  (`RETRY_DELAYS`), and the nightly cron (`scripts/exceptions_sweep.py`) sends
  whatever is still pending. The web process only sends when `RESEND_API_KEY` and
  `ALERT_EMAIL_TO` are set on that service; otherwise the nightly job does it.

One email covers every alert pending at that moment (normally one). Delivery is
at-least-once: a crash between the send and the record can repeat an email.
"""
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
BATCH = 25
DEFAULT_DASHBOARD_URL = 'https://cns-factory-ledger.netlify.app'
_worker = {'lock': threading.Lock(), 'thread': None, 'again': False}


# ---------------------------------------------------------------------------
# 1. In the posting transaction: record that an email is owed. Never raises.
# ---------------------------------------------------------------------------
def queue_if_large(api, cur, transaction_id, action, product, quantity, uom=None):
    """Queue the owner email when this correction is over 500 lb. Returns True when queued.

    Called by the adjust and found posting cores right after the ledger rows are
    written. Off until migration 095's marker exists. Everything happens inside a
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
# 2. The email, built from the ledger at send time.
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
        destination = 'Not recorded'
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


def compose(items, env=None):
    """{'subject', 'text', 'html'} for one email covering `items` (facts dicts)."""
    env = os.environ if env is None else env
    stage = env.get('ENVIRONMENT', '').strip().lower()
    tag = '' if stage in ('', 'production') else f'[{stage}] '
    if len(items) == 1:
        only = items[0]
        subject = f'{tag}Large inventory adjustment: {only["change"].split(" (")[0]} {only["product"]}, lot {only["lot"]}'
    else:
        subject = f'{tag}{len(items)} large inventory adjustments (over {int(a3b.LARGE_CORRECTION_LB)} lb each)'
    intro = (f'An inventory adjustment over {int(a3b.LARGE_CORRECTION_LB)} lb was recorded. It counted in stock right away; '
             'nothing is waiting for your approval.')
    lines, html = [intro, ''], [f'<p style="font:14px sans-serif">{escape(intro)}</p>']
    for item in items:
        rows = [(label, item[key]) for label, key in _ROWS if item.get(key)]
        if not item['still_counts']:
            rows.append(('Status', 'This entry has since been voided.'))
        lines += [f'{label}: {value}' for label, value in rows] + ['']
        html.append('<table style="font:14px sans-serif;border-collapse:collapse;margin:0 0 16px">' + ''.join(
            f'<tr><td style="padding:2px 12px 2px 0;color:#555">{escape(label)}</td><td style="padding:2px 0">{escape(str(value))}</td></tr>'
            for label, value in rows) + '</table>')
    return {'subject': subject, 'text': '\n'.join(lines), 'html': '\n'.join(html)}


# ---------------------------------------------------------------------------
# 3. Deliver what is pending. Never raises; a failure leaves the rows pending.
# ---------------------------------------------------------------------------
def configured(env=None):
    env = os.environ if env is None else env
    return bool(env.get('RESEND_API_KEY', '').strip() and env.get('ALERT_EMAIL_TO', '').strip())


def deliver(transaction, env=None, send=None):
    """One pass: claim pending alerts, send one email, record the result.

    `transaction` is a zero-argument context manager that yields a dict cursor and
    commits on exit (`api.get_transaction` in the app). Returns a summary dict:
    status is unavailable | not_configured | nothing_pending | sent | failed | error."""
    send = send or reconciliation.send_email
    ids = []
    try:
        if not configured(env):
            return {'status': 'not_configured', 'count': 0}
        with transaction() as cur:
            cur.execute("SELECT to_regclass('public.adjustment_alerts') AS relation")
            if cur.fetchone()['relation'] is None:
                return {'status': 'unavailable', 'count': 0}
            cur.execute('''UPDATE adjustment_alerts SET attempts = attempts + 1, last_attempt_at = clock_timestamp()
                           WHERE id IN (SELECT id FROM adjustment_alerts
                                        WHERE status = 'pending' AND (last_attempt_at IS NULL
                                              OR last_attempt_at < clock_timestamp() - %s::interval)
                                        ORDER BY id LIMIT %s FOR UPDATE SKIP LOCKED)
                           RETURNING id, transaction_id, action, lb''', (CLAIM_COOLDOWN, BATCH))
            claimed = sorted(cur.fetchall(), key=lambda row: row['id'])
            items = []
            for alert in claimed:
                item = facts(cur, alert, env)
                if item:
                    ids.append(alert['id'])
                    items.append(item)
        if not items:
            return {'status': 'nothing_pending', 'count': 0}
        outcome = send(compose(items, env), env)
        sent = outcome.get('status') == 'sent'
        with transaction() as cur:
            if sent:
                cur.execute('''UPDATE adjustment_alerts SET status = 'sent', sent_at = clock_timestamp(), email_to = %s,
                                   provider_id = %s, last_error = NULL WHERE id = ANY(%s) AND status = 'pending' ''',
                            (outcome.get('to'), outcome.get('provider_id'), ids))
            else:
                cur.execute("UPDATE adjustment_alerts SET last_error = %s WHERE id = ANY(%s) AND status = 'pending'",
                            ((outcome.get('error') or 'send failed')[:500], ids))
        if sent:
            logger.info('ADJUSTMENT_ALERT_EMAIL sent alerts=%s', ids)
        else:
            logger.warning('ADJUSTMENT_ALERT_EMAIL failed alerts=%s error=%s (will retry)', ids, outcome.get('error'))
        return {'status': 'sent' if sent else 'failed', 'count': len(ids), 'alert_ids': ids, 'error': outcome.get('error')}
    except Exception as exc:   # never the caller's problem; the rows stay pending for the next pass
        logger.error('ADJUSTMENT_ALERT_EMAIL error alerts=%s (%s); will retry', ids, type(exc).__name__)
        return {'status': 'error', 'count': 0, 'alert_ids': ids, 'error': type(exc).__name__}


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
