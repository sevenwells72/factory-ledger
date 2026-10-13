"""A12: immutable lot tiers, signed-in owner approval and downgrade evidence.

The caller owns the transaction. No PIN, client-supplied actor or startup DDL.
"""
from datetime import datetime
from fastapi import HTTPException
from psycopg2.extras import Json
import permissions

EXTRA = 'extra_kosher'

def tier(row):
    return row.get('kosher_tier') or 'regular'

def fields(row):
    value = tier(row)
    return {'kosher_tier': value, 'kosher_label': 'EXTRA-KOSHER (SS)' if value == EXTRA else 'Regular'}

def owner(cur, identity):
    if not identity or identity.get('id') is None:
        return None
    cur.execute('SELECT id,name,role,active FROM actors WHERE id=%s FOR SHARE', (identity['id'],))
    current = cur.fetchone()
    return dict(current) if current and current['active'] and current['role']=='owner' else None

def attest(cur, identity):
    current = owner(cur, identity)
    if not current:
        raise HTTPException(403, {'error_code':'KOSHER_ATTESTATION_REQUIRED',
            'message':'Use a make ticket; Michael must approve this extra-kosher make from his own signed-in device.'})
    cur.execute('SELECT clock_timestamp() AS at')
    return {'actor_id':current['id'], 'actor_name':current['name'], 'attested_at':cur.fetchone()['at']}

def output_matches(cur, lot_id, product):
    cur.execute('SELECT kosher_tier FROM lots WHERE id=%s', (lot_id,))
    if tier(cur.fetchone()) != tier(product):
        raise HTTPException(409, {'error_code':'LOT_KOSHER_TIER_MISMATCH',
            'message':'This output lot has a different immutable kosher tier; use a new lot.'})

def require_sources(cur, target, lot_ids):
    if tier(target) != EXTRA:
        return
    cur.execute('SELECT id,kosher_tier FROM lots WHERE id=ANY(%s)', (list(lot_ids),))
    rows = cur.fetchall()
    if not rows or len(rows) != len(set(lot_ids)) or any(tier(r)!=EXTRA for r in rows):
        raise HTTPException(409, {'error_code':'KOSHER_SOURCE_REQUIRED',
            'message':'EXTRA-KOSHER finished goods require extra-kosher source lots.'})

def pack_lots(cur, source, target, lots, allocations=None):
    if tier(target) != EXTRA:
        return lots
    if allocations:
        cur.execute('SELECT id FROM lots WHERE product_id=%s AND lower(lot_code)=ANY(%s)',
            (source['id'], [a.lot_code.lower() for a in allocations]))
        ids = [r['id'] for r in cur.fetchall()]
        if len(ids) != len(allocations):
            raise HTTPException(409, {'error_code':'KOSHER_SOURCE_REQUIRED', 'message':'Resolve an extra-kosher source lot.'})
        require_sources(cur, target, ids)
    cur.execute('SELECT id,kosher_tier FROM lots WHERE id=ANY(%s)',([r['lot_id'] for r in lots],))
    tiers = {r['id']:r['kosher_tier'] for r in cur.fetchall()}
    eligible = [dict(r,kosher_tier=tiers[r['lot_id']]) for r in lots if tiers.get(r['lot_id'])==EXTRA]
    if not eligible and not allocations:
        # Depleted lots may still be pinned for A3b shortages, but never regular lots.
        cur.execute("SELECT id FROM lots WHERE product_id=%s AND kosher_tier='extra_kosher' AND status IS DISTINCT FROM 'merged' LIMIT 1", (source['id'],))
        if not cur.fetchone():
            raise HTTPException(409, {'error_code':'KOSHER_SOURCE_REQUIRED', 'message':'Make an extra-kosher batch before packing this product.'})
    return eligible

def downgrade(cur, target, lot_ids):
    if tier(target) == EXTRA:
        return False
    cur.execute("SELECT EXISTS(SELECT 1 FROM lots WHERE id=ANY(%s) AND kosher_tier='extra_kosher') AS yes", (list(lot_ids),))
    return cur.fetchone()['yes']

def record_downgrade(cur, lot_id, transaction_id, source_ids, request_actor_id, occurred_at, ticket_id):
    cur.execute("""INSERT INTO lot_events(lot_id,event,old,new,happened_at,entered_by_actor_id,ticket_id,transaction_id)
        VALUES (%s,'kosher_downgrade',%s,%s,%s,%s,%s,%s)""",
        (lot_id,Json({'kosher_tier':EXTRA,'source_lot_ids':list(source_ids)}),Json({'kosher_tier':'regular'}),
         occurred_at,request_actor_id,ticket_id,transaction_id))

def record_attestation(cur, lot_id, transaction_id, ticket_id, attestation):
    cur.execute('UPDATE lots SET kosher_attested_by_actor_id=%s,kosher_attested_at=%s WHERE id=%s',
        (attestation['actor_id'],attestation['attested_at'],lot_id))
    if ticket_id is not None:
        cur.execute("INSERT INTO ticket_attestations(ticket_id,kind,actor_id,attested_at) VALUES (%s,'kosher_procedure',%s,%s)",
            (ticket_id,attestation['actor_id'],attestation['attested_at']))

def hold(api, cur, row, actor, draft, acknowledged, effective_payload):
    import write_tickets
    if actor['id'] is None:
        raise HTTPException(403, {'error_code':'KOSHER_ATTESTATION_REQUIRED',
            'message':'A named actor must prepare the extra-kosher make for Michael to approve.'})
    if row['status']=='awaiting_approval':
        return row['response']
    receipt = write_tickets.allocate_receipt(cur,'make',datetime.fromisoformat(row['payload']['occurred_at']).astimezone(api.PLANT_TIMEZONE).date())
    detail = {'payload_hash':row['payload_hash'],'action':'make','kosher_tier':EXTRA,
        'prepared_by_actor_id':actor['id'],'product_id':row['payload']['product_id'],
        'happened_at':row['payload']['occurred_at'],'lot_code':draft.get('lot_code')}
    cur.execute("""INSERT INTO exceptions(kind,severity,product_id,ticket_id,receipt_number,detail,owner_actor_id)
        VALUES ('KOSHER_ATTESTATION','block',%s,%s,%s,%s,%s) RETURNING id""",
        (row['payload']['product_id'],row['id'],receipt,Json(detail),permissions.owner_actor_id(cur)))
    exception_id = cur.fetchone()['id']
    response = {'held':True,'posted':False,'status':'awaiting_approval','ticket_id':row['id'],
        'receipt_number':receipt,'exception_id':exception_id,'kosher_tier':EXTRA,'kosher_label':'EXTRA-KOSHER (SS)',
        'error_code':'KOSHER_ATTESTATION_REQUIRED',
        'message':"HELD — waiting for Michael's kosher approval. Nothing was posted.",
        'message_es':'EN ESPERA — falta la aprobación kosher de Michael. No se registró nada.'}
    cur.execute("""UPDATE write_tickets SET status='awaiting_approval',receipt_number=%s,response=%s,
        expires_at=clock_timestamp()+interval '7 days',acknowledged=%s,
        draft=draft || jsonb_build_object('held_at',clock_timestamp(),'held_lot_confirmations',%s::jsonb)
        WHERE id=%s RETURNING expires_at""",
        (receipt,Json(response),Json(sorted(set(acknowledged))),Json(effective_payload.get('lot_confirmations') or []),row['id']))
    response['expires_at']=cur.fetchone()['expires_at'].isoformat()
    cur.execute('UPDATE write_tickets SET response=%s WHERE id=%s',(Json(response),row['id']))
    return response

def receipt_evidence(cur, transactions, lots):
    for transaction in transactions:
        cur.execute("""SELECT t.kosher_tier,t.kosher_attested_by_actor_id,t.kosher_attested_at,
            t.transaction_kosher_downgrade,a.name AS kosher_attested_by_name
            FROM transactions t LEFT JOIN actors a ON a.id=t.kosher_attested_by_actor_id WHERE t.id=%s""",(transaction['id'],))
        transaction.update(dict(cur.fetchone()))
        transaction.update(fields(transaction))
    for lot in lots:
        cur.execute("""SELECT l.kosher_tier,l.kosher_attested_by_actor_id,l.kosher_attested_at,
            a.name AS kosher_attested_by_name FROM lots l LEFT JOIN actors a ON a.id=l.kosher_attested_by_actor_id WHERE l.id=%s""",(lot['id'],))
        lot.update(dict(cur.fetchone()))
        lot.update(fields(lot))
