"""A2: roles enforced in FL — the per-action permission matrix and the back-dating rule.

One module, no database state: the matrix is code (design rev 3.7 §4.3), the
role comes from the authenticated identity (`actors.role` for actor keys, the
key kind for the two shared keys), and every denial is the same structured 403.
Later chunks reuse `require()` / `allowed()` unchanged: A3b `/exceptions/*`
(`list_exceptions`, `resolve_exception`, `approve_exception`), A4 part 2
`/aliases` writes (`manage_aliases`), A6/A7 ship and order tickets, A11 PIN
sessions (a session resolves to an actor, so it lands in the same role column)
and F1 (greys out by `GET /auth/whoami` `permissions`, FL still enforces).

Actions are keyed exactly like `write_tickets.action` where a ticket exists
(`receive`, `make`, `pack`, `adjust`, `found`), so the ticket layer enforces on
the stored action, never on the route.
"""
from datetime import timedelta

from fastapi import HTTPException

OWNER, FLOOR, OFFICE = 'owner', 'floor', 'office'
# The two shared keys have no person behind them. They keep exactly what they
# reach today and get nothing new (owner decision, A2 brief); A10 retires them.
LEGACY_LEDGER, LEGACY_DASHBOARD = 'legacy_ledger', 'legacy_dashboard'
ROLES = (OWNER, FLOOR, OFFICE, LEGACY_LEDGER, LEGACY_DASHBOARD)
NAMED = frozenset({OWNER, FLOOR, OFFICE})
_ALL = frozenset(ROLES)
_NAMED_AND_MASTER = NAMED | {LEGACY_LEDGER}

# action -> roles allowed. Design §4.3 rows, in the same order; "✗" = absent.
ROLE_PERMISSIONS = {
    # Ledger tickets (write_tickets.action). The dashboard key keeps only the
    # A1 part-1 receive grant; the master key keeps every route it reaches today.
    'receive': _ALL,
    'make': _NAMED_AND_MASTER - {OFFICE},
    'pack': _NAMED_AND_MASTER - {OFFICE},
    'adjust': _NAMED_AND_MASTER - {OFFICE},           # > 500 lb / > 10 % hold is A3b
    'found': _NAMED_AND_MASTER - {OFFICE},
    'void': _NAMED_AND_MASTER - {OFFICE},             # floor: own posts, same plant day (A3b)
    'rename_lot': _NAMED_AND_MASTER,
    'update_supplier_lot': _NAMED_AND_MASTER,
    'move_lot': _NAMED_AND_MASTER,                    # A5 pallet-move ticket (LOT receipts)
    # Shipping (A6 tickets)
    'ship_order': _NAMED_AND_MASTER,
    'ship_standalone': _NAMED_AND_MASTER - {FLOOR},
    'ship_bulk': _NAMED_AND_MASTER,
    # Orders (A7 tickets; today's direct routes are unchanged)
    'create_order': _NAMED_AND_MASTER - {FLOOR},
    'add_order_lines': _NAMED_AND_MASTER - {FLOOR},
    'update_order_line': _NAMED_AND_MASTER - {FLOOR},
    'cancel_order_line': _NAMED_AND_MASTER - {FLOOR},
    'update_order_header': _NAMED_AND_MASTER - {FLOOR},
    'update_order_status': _NAMED_AND_MASTER - {FLOOR},
    'mark_order_ready': _NAMED_AND_MASTER,            # floor's only status change
    'cancel_order': _NAMED_AND_MASTER - {FLOOR},
    'close_order_shipped_not_recorded': frozenset({OWNER, LEGACY_LEDGER}),
    'reopen_order': frozenset({OWNER, LEGACY_LEDGER}),
    'create_expected_receipt': _NAMED_AND_MASTER - {FLOOR},
    'manage_customers': _NAMED_AND_MASTER - {FLOOR},
    # New in Phase 1 — no shared key reaches any of these.
    'manage_aliases': frozenset({OWNER, OFFICE}),
    'list_exceptions': NAMED,
    'resolve_exception': frozenset({OWNER, FLOOR}),   # shortage, unidentified lot
    'approve_exception': frozenset({OWNER}),          # held corrections, late-entry ack, proof waiver
    'kosher_attestation': frozenset({OWNER}),
    # Back-dating > 14 days (§6.3). The master key keeps today's backfill path.
    'backdate_over_14d': frozenset({OWNER, LEGACY_LEDGER}),
}
ACTIONS = tuple(ROLE_PERMISSIONS)

_LABELS = {
    'receive': ('record a receive', 'registrar una recepción'),
    'make': ('record a make', 'registrar una producción'),
    'pack': ('record a pack', 'registrar un empaque'),
    'adjust': ('adjust inventory', 'ajustar inventario'),
    'found': ('record found inventory', 'registrar inventario encontrado'),
    'void': ('void a posting', 'anular un registro'),
    'move_lot': ('move a lot', 'mover un lote'),
    'backdate_over_14d': ('record an entry more than 14 days old', 'registrar una entrada de más de 14 días'),
}
_ROLE_NAMES = {OWNER: 'owner', FLOOR: 'floor', OFFICE: 'office',
               LEGACY_LEDGER: 'the shared ledger key', LEGACY_DASHBOARD: 'the shared dashboard key'}


def fail(http_status, code, message, **extra):
    raise HTTPException(http_status, {'error_code': code, 'message': message, **extra})


def role_of(identity):
    """`write_tickets.identity()` shape: {id, name, role, key_kind}. Unknown → None (denied)."""
    if identity.get('key_kind') == 'actor':
        return identity.get('role') if identity.get('role') in NAMED else None
    return identity.get('key_kind') if identity.get('key_kind') in (LEGACY_LEDGER, LEGACY_DASHBOARD) else None


def allowed(action, role):
    return role in ROLE_PERMISSIONS.get(action, frozenset())


def permissions_for(identity):
    role = role_of(identity)
    return {action: allowed(action, role) for action in ACTIONS}


def label(action):
    return _LABELS.get(action, (action.replace('_', ' '), action.replace('_', ' ')))


def require(action, identity):
    """403 ROLE_NOT_ALLOWED naming the action, the role and the person."""
    role = role_of(identity)
    if allowed(action, role):
        return role
    who = identity.get('name') or _ROLE_NAMES.get(role, 'this key')
    en, es = label(action)
    fail(403, 'ROLE_NOT_ALLOWED',
         f'{who} ({_ROLE_NAMES.get(role, "unknown role")}) is not allowed to {en}.',
         message_es=f'{who} no tiene permiso para {es}.',
         action=action, role=role, actor=identity.get('name'), key_kind=identity.get('key_kind'))


# ---------------------------------------------------------------------------
# Back-dating (design §6.3, decision D6). All comparisons are absolute
# (timezone-aware) so DST never shifts a boundary; display is plant time.
# ---------------------------------------------------------------------------
FUTURE_GRACE = timedelta(minutes=5)   # = main.INVENTORY_OCCURRED_AT_FUTURE_GRACE (clock skew)
LATE_ENTRY_AFTER = timedelta(hours=48)
BACKFILL_AFTER = timedelta(days=14)   # = main.INVENTORY_OCCURRED_AT_STANDARD_WINDOW


def timing(happened_at, now):
    """Classify `now - happened_at`: future | normal (≤ 48 h) | late (48 h–14 d) | backfill (> 14 d).

    'future' is informational here; `validate_inventory_occurred_at` is what
    rejects it (400 OCCURRED_AT_IN_FUTURE, a draft blocker).
    """
    delta = now - happened_at
    hours = round(delta.total_seconds() / 3600, 2)
    if delta < -FUTURE_GRACE:
        status = 'future'
    elif delta <= LATE_ENTRY_AFTER:
        status = 'normal'
    elif delta <= BACKFILL_AFTER:
        status = 'late'
    else:
        status = 'backfill'
    return {'status': status, 'hours_late': max(hours, 0.0), 'days_late': max(int(hours // 24), 0),
            'happened_at': happened_at.isoformat()}


def require_backdating(identity, happened_at, now):
    """> 14 days is owner-only, at prepare and again at commit. Returns the timing dict."""
    info = timing(happened_at, now)
    if info['status'] == 'backfill' and not allowed('backdate_over_14d', role_of(identity)):
        who = identity.get('name') or _ROLE_NAMES.get(role_of(identity), 'this key')
        fail(403, 'BACKFILL_OWNER_ONLY',
             f'{who} cannot record an entry {info["days_late"]} days after it happened; '
             'entries older than 14 days are owner-only.',
             message_es=f'{who} no puede registrar una entrada {info["days_late"]} días después; '
                        'las entradas de más de 14 días son solo del dueño.',
             action='backdate_over_14d', role=role_of(identity), actor=identity.get('name'),
             days_late=info['days_late'])
    return info


def opens_late_entry(identity, info):
    """Floor, office and the shared keys get a LATE_ENTRY exception; the owner does not (§6.3)."""
    return info['status'] == 'late' and role_of(identity) != OWNER


def late_entry_warning(identity, info):
    if not opens_late_entry(identity, info):
        return None
    days = info['days_late']
    return {'code': 'LATE_ENTRY', 'requires_ack': False,
            'message': f'This will be recorded as a late entry ({days} days) for the owner to acknowledge.',
            'message_es': f'Se registrará como entrada tardía ({days} días) para que el dueño la revise.',
            'refs': {'days_late': days, 'hours_late': info['hours_late']}}


def owner_actor_id(cur):
    cur.execute("SELECT id FROM actors WHERE role='owner' AND active ORDER BY id LIMIT 1")
    row = cur.fetchone()
    return row['id'] if row else None


def open_late_entry(cur, *, identity, info, action, transaction_id, receipt_number, ticket_id,
                    entered_at, product_id=None, lot_id=None, client_source=None):
    """Append the 061 `exceptions(LATE_ENTRY)` row in the caller's transaction; returns its id."""
    from psycopg2.extras import Json
    detail = {'days_late': info['days_late'], 'hours_late': info['hours_late'],
              'happened_at': info['happened_at'], 'entered_at': entered_at.isoformat(),
              'action': action, 'actor_id': identity.get('id'), 'actor': identity.get('name'),
              'role': role_of(identity), 'client_source': client_source}
    cur.execute('''INSERT INTO exceptions(kind,status,severity,product_id,lot_id,transaction_id,
                       receipt_number,ticket_id,detail,owner_actor_id)
                   VALUES ('LATE_ENTRY','open','warn',%s,%s,%s,%s,%s,%s,%s) RETURNING id''',
                (product_id, lot_id, transaction_id, receipt_number, ticket_id, Json(detail),
                 owner_actor_id(cur)))
    return cur.fetchone()['id']


def matrix_markdown():
    """The §4.3 table as rendered in the PR description (kept in sync with the code by a test)."""
    head = '| Action | ' + ' | '.join(ROLES) + ' |'
    rows = [head, '|' + '---|' * (len(ROLES) + 1)]
    for action in ACTIONS:
        rows.append(f'| `{action}` | ' + ' | '.join('✓' if allowed(action, r) else '✗' for r in ROLES) + ' |')
    return '\n'.join(rows)
