"""Real owner PIN proof for pre-A11 business-rule tests (never bypass auth)."""
import secrets
import main
import pin_sessions

OWNER_PINS = {}


def seed_owner_pin(cur, actor_id, key):
    cur.execute('SELECT pin_hash FROM actors WHERE pin_hash IS NOT NULL')
    used = {row['pin_hash'] if isinstance(row, dict) else row[0] for row in cur.fetchall()}
    while True:
        value = str(secrets.randbelow(9000)+1000)
        if pin_sessions.valid_pin(value) and pin_sessions.pin_hash(value) not in used:
            break
    cur.execute('UPDATE actors SET pin_hash=%s,pin_set_at=clock_timestamp() WHERE id=%s', (pin_sessions.pin_hash(value), actor_id))
    OWNER_PINS[key] = value


def headers(key=None):
    result = {'X-API-Key':key or main.API_KEY}
    if key in OWNER_PINS:
        result['X-FL-Owner-PIN'] = OWNER_PINS[key]
    return result


def commit(client, prepared, key=None, **updates):
    return client.post('/tickets/'+prepared['ticket']+'/commit',
                       json={'payload_hash':prepared['payload_hash'],**updates}, headers=headers(key))
