"""A6 shipping through the A1 lifecycle and the existing cursor-owned cores.

Bulk here means multiple resolved product/lot lines. A8 bins and invoicing
are deliberately absent. Quantities/customer/BOL/time stay in the signed draft.
"""
import hashlib
import hmac
import os
import time
from datetime import date, datetime, time as day_time
from typing import List, Optional
from urllib.parse import urlencode

from fastapi import Depends, HTTPException, Query, Request
from psycopg2.extras import Json
from pydantic import Field, root_validator, validator, confloat

import attachments
import kosher_tiers
import lot_confirmation as a5
import order_tickets
import permissions
import ticket_actions
import write_tickets as wt

ACTIONS=('ship_order','ship_standalone','ship_bulk')
ROUTES=frozenset({('POST','/ship/prepare'),('POST','/ship/bulk/prepare'),('POST','/sales/orders/{order_id}/ship/prepare')})

class ShipBase(wt.IdInput):
    occurred_at: Optional[datetime]=None
    backfill: bool=False
    client_source: wt.ClientSource='api'
    bol_reference: str=Field('',max_length=200)
    attachment_ids: List[wt.PositiveId]=Field(default_factory=list,max_length=20)
    lot_confirmations: List[a5.LotConfirmation]=Field(default_factory=list)

    @root_validator(pre=True)
    def aliases(cls,values):
        values=dict(values)
        if 'happened_at' in values:
            if 'occurred_at' in values: wt.fail(422,'AMBIGUOUS_EVENT_TIME','Supply one event time.')
            values['occurred_at']=values.pop('happened_at')
        if 'customer_name' in values: wt.fail(422,'IDS_REQUIRED','Use customer_id from /resolve.')
        return values

    @validator('bol_reference')
    def trim(cls,value): return value.strip()

class OrderLine(wt.IdInput):
    lot_allocations: Optional[List[wt.LotAllocation]] = Field(None, max_length=100)
    line_id: wt.PositiveId
    quantity_lb: confloat(ge=0,allow_inf_nan=False)

class OrderShip(ShipBase):
    ship_all: bool=False
    lines: List[OrderLine]=Field(default_factory=list,max_length=100)

class BulkLine(wt.IdInput):
    lot_allocations: Optional[List[wt.LotAllocation]] = Field(None, max_length=100)
    product_id: wt.PositiveId
    lot_id: Optional[wt.PositiveId]=None
    quantity_lb: confloat(gt=0,allow_inf_nan=False)

class StandaloneShip(ShipBase):
    lot_allocations: Optional[List[wt.LotAllocation]] = Field(None, max_length=100)
    customer_id: wt.PositiveId
    product_id: wt.PositiveId
    lot_id: Optional[wt.PositiveId]=None
    quantity_lb: confloat(gt=0,allow_inf_nan=False)
    order_reference: str=Field(min_length=1,max_length=200)

class BulkShip(ShipBase):
    customer_id: wt.PositiveId
    lines: List[BulkLine]=Field(min_length=1,max_length=100)
    order_reference: str=Field(min_length=1,max_length=200)

def legacy_close_cutoff():
    value=os.environ.get('ORDER_LEGACY_CLOSE_CUTOFF','').strip()
    if not value: return None
    try:
        cutoff=date.fromisoformat(value)
        if cutoff.isoformat()!=value: raise ValueError
        return cutoff
    except ValueError:
        raise RuntimeError('ORDER_LEGACY_CLOSE_CUTOFF must be a YYYY-MM-DD America/New_York date') from None

def validate_configuration():
    legacy_close_cutoff()
    if os.environ.get('ENVIRONMENT','').strip().lower()=='production' and not os.environ.get('PACKING_SLIP_SIGNING_KEY','').strip():
        raise RuntimeError('PACKING_SLIP_SIGNING_KEY is required in production')

def same_day(api,payload,actor):
    happened=datetime.fromisoformat(payload['occurred_at']).astimezone(api.PLANT_TIMEZONE).date()
    entered=api.get_plant_now().astimezone(api.PLANT_TIMEZONE).date()
    if happened==entered: return []
    if permissions.role_of(actor) not in ('owner','floor'):
        wt.fail(422,'SHIP_NOT_SAME_DAY','Only owner or floor may acknowledge a shipment outside the same America/New_York plant day.')
    return [{'code':'SHIP_NOT_SAME_DAY','requires_ack':True,
             'message':'This shipment is outside the same plant day; acknowledge before recording.',
             'message_es':'Este envío es de otro día de planta; confirme antes de registrarlo.',
             'refs':{'happened_on':happened.isoformat(),'entered_on':entered.isoformat()}}]

def stock_shortage(product, available_lb, needed_lb, *, order=False):
    """Return safe case/lb specifics from the same allocation plan that refused."""
    from assistant_replies import number
    size = float(product.get('case_size_lb') or 0)
    available = number(available_lb / size if size else available_lb)
    needed = number(needed_lb / size if size else needed_lb)
    unit, unit_es = ('cases', 'cajas') if size else ('lb', 'lb')
    subject, subject_es = ('the order', 'el pedido') if order else ('this shipment', 'este envío')
    wt.fail(409, 'INSUFFICIENT_STOCK',
        f"Only {available} {unit} of {product['name']} available; {subject} needs {needed}.",
        message_es=f"Solo hay {available} {unit_es} de {product['name']}; {subject_es} necesita {needed}.",
        product=product['name'], available_lb=float(available_lb), needed_lb=float(needed_lb),
        case_size_lb=size or None)


def validate(api,cur,action,payload,actor,*,lock=False,ticket_id=None):
    day_warnings=same_day(api,payload,actor)
    occurred,source=api.validate_inventory_occurred_at(datetime.fromisoformat(payload['occurred_at']),payload['backfill'])
    order=None
    if action=='ship_order':
        order=order_tickets.load_order(cur,payload['order_id'],lock=lock)
        api._require_open_state(order['state'],order['order_number'],order['id'],'shipping')
        if order['status'] in ('new','invoiced','cancelled'): wt.fail(409,'ORDER_NOT_SHIPPABLE','Confirm an open order before shipping.')
        if lock: api._lock_sales_order_lines(cur,order['id'])
        original=order_tickets.order_lines(cur,order['id'])
        lines=[]
        for requested in payload['lines']:
            row=next((r for r in original if r['id']==requested['line_id']),None)
            if not row: wt.fail(404,'LINE_NOT_FOUND','Line does not belong to this order.')
            if row['line_status']=='cancelled': wt.fail(409,'LINE_CANCELLED','Cancelled lines cannot ship.')
            p=ticket_actions.product(api,cur,row['product_id'],False)
            if p['is_service'] and requested.get('lot_allocations') is not None:
                wt.fail(422,'UNUSED_LOT_SELECTION','Service charges do not consume lots.')
            remaining=float(row['quantity_lb'])-float(row['quantity_shipped_lb'])
            if row['line_status']=='fulfilled': wt.fail(409,'LINE_ALREADY_FULFILLED','Line is already fulfilled.')
            if not p['is_service'] and (requested['quantity_lb']<=0 or requested['quantity_lb']>remaining):
                wt.fail(422,'QTY_EXCEEDS_REMAINING','Quantity must be positive and within the remaining order quantity.')
            lines.append(dict(requested,product_id=p['id'],product=p,remaining_lb=remaining))
        customer=order_tickets.customer(cur,order['customer_id'],lock=False)
    else:
        customer=order_tickets.customer(cur,payload['customer_id'],lock=False)
        lines=[]
        for row in payload['lines']:
            p=ticket_actions.product(api,cur,row['product_id'],False)
            choices=row.get('lot_allocations')
            if choices is not None:
                if (row.get('lot_id') is not None or not choices
                        or len({r['lot_id'] for r in choices}) != len(choices)
                        or abs(sum(float(r['quantity_lb']) for r in choices)-float(row['quantity_lb']))>api.BALANCE_EPSILON):
                    wt.fail(422,'INVALID_LOT_ALLOCATIONS','Select each lot once; lot quantities must add up to the shipment quantity.')
                lines.extend(dict(row,**choice,product=p) for choice in choices)
            else:
                lines.append(dict(row,product=p))
        original=[]
        if any(r['product']['is_service'] for r in lines): wt.fail(422,'SERVICE_STANDALONE_UNSUPPORTED','Service charges ship against an order.')
    if not lines: wt.fail(409,'ORDER_ALREADY_FULFILLED','No remaining lines to ship.')
    if lock: api._lock_allocation_products(cur,[r['product_id'] for r in lines if not r['product']['is_service']])
    plan=[]; states={}; display=[]; requirements=[]
    totals={}
    for row in lines:
        p=row['product']; qty=row['quantity_lb']; allocated=[]
        if not p['is_service']:
            if action=='ship_order':
                allocation=api._sales_order_ship_plan(cur,p['id'],row['line_id'],qty,lock=lock,persist_expired=False,lot_allocations=row.get('lot_allocations'))
                allocated=allocation['lots']
                if float(allocation['actual_ship_lb'])+api.BALANCE_EPSILON<qty:
                    stock_shortage(p, float(allocation['actual_ship_lb']), qty, order=True)
            else:
                lots=api.available_lots_for_product(cur,p['id'],lock=lock,persist_expired=False)
                lots=kosher_tiers.sale_lots(cur,p['id'],lots)
                if row.get('lot_id') is not None: lots=[l for l in lots if l['lot_id']==row['lot_id']]
                allocation=api._takeable_deduction_plan(lots,qty)
                if allocation['short_lb']>api.BALANCE_EPSILON:
                    stock_shortage(p, qty-float(allocation['short_lb']), qty)
                allocated=[{'lot_id':r['lot']['lot_id'],'lot_code':r['lot']['lot_code'],'quantity_lb':r['quantity_lb']} for r in allocation['lots']]
            kosher_tiers.require_sources(cur,p,[take['lot_id'] for take in allocated])
            for take in allocated:
                lot=ticket_actions.lot(api,cur,take['lot_id'],p['id'],False)
                states[lot['id']]=lot
                totals[lot['id']]=totals.get(lot['id'],0)+float(take['quantity_lb'])
                if totals[lot['id']]>float(lot['on_hand_lb'])+api.BALANCE_EPSILON:
                    stock_shortage(p, float(lot['on_hand_lb']), totals[lot['id']])
                plan.append(dict(take,product_id=p['id'],line_id=row.get('line_id')))
            requirements.extend(allocated)
        display.append({'line_id':row.get('line_id'),'product_id':p['id'],'product':p['name'],'quantity_lb':qty,
                        'cases':qty/float(p['case_size_lb']) if p.get('case_size_lb') else None,'is_service':p['is_service'],'lots':allocated})
    evidence_plan=[{'lot_id':lid,'quantity_lb':qty} for lid,qty in sorted(totals.items())]
    evidence,blockers=a5.validate_plan(api,cur,evidence_plan,list(states.values()),payload['lot_confirmations'],commit=lock)
    if not payload['bol_reference']:
        if lock: wt.fail(422,'BOL_REQUIRED','BOL number is required before recording a shipment.')
        blockers.append({'code':'BOL_REQUIRED','message':'Enter the BOL number before recording.'})
    photo=False
    for ref in payload.get('attachment_ids',[]):
        att=attachments.verified(cur,ref,actor,ticket_id=ticket_id,kinds=('bol','truck_photo','signed_packing_slip'),claim=lock)
        attachments.load(api,att)
        photo |= att['kind']=='bol' and att['content_type'].startswith('image/')
    warnings=list(day_warnings)
    if action!='ship_order':
        opened=api.check_open_orders_for_ship(cur,customer['id'],customer['name'])
        if opened:
            if lock and permissions.role_of(actor) not in ('office','owner','legacy_ledger'):
                wt.fail(403,'ROLE_NOT_ALLOWED','Only office/owner may acknowledge open sales orders.')
            warnings.append({'code':'OPEN_SALES_ORDER_EXISTS','requires_ack':True,'message':'Customer has open orders; confirm this standalone shipment is intentional.','refs':opened})
    draft={'order_id':order['id'] if order else None,'order':order['order_number'] if order else None,
           'customer_id':customer['id'],'customer':customer['name'],'lines':display,'total_lb':sum(r['quantity_lb'] for r in display if not r['is_service']),
           'total_cases':sum(r['cases'] or 0 for r in display),'bol_reference':payload['bol_reference'],
           'photo':'attached' if photo else 'pending','input_plan':evidence,'blockers':blockers}
    warnings+=possible_duplicate(cur,action,payload,customer['id'],draft['total_lb'])
    state={'order':order,'lines':original,'lots':list(states.values()),'plan':plan}
    return {'draft':draft,'state':state,'lines':lines,'customer':customer,'order':order,'plan':plan,'evidence':evidence,'warnings':warnings,'occurred':occurred,'source':source}

def possible_duplicate(cur,action,payload,customer_id,total_lb):
    clause='AND sl.sales_order_line_id=ANY(%s)' if action=='ship_order' else ''
    having='' if action=='ship_order' else 'HAVING SUM(sl.quantity_lb)=%s'
    args=[customer_id]
    if action=='ship_order':args.append([r['line_id'] for r in payload['lines']])
    else:args.append(total_lb)
    cur.execute('''SELECT s.id,COALESCE(w.receipt_number,MAX(raw.receipt_number)) AS receipt_number,
                          MAX(raw.operator_id) AS operator_id,
                          EXTRACT(EPOCH FROM(clock_timestamp()-s.created_at))/60 AS minutes_ago
                   FROM shipments s JOIN shipment_lines sl ON sl.shipment_id=s.id
                   JOIN ledger_current_transactions effective ON effective.id=sl.transaction_id AND effective.effective_status='posted'
                   JOIN transactions raw ON raw.id=sl.transaction_id
                   LEFT JOIN write_tickets w ON w.id=s.ticket_id
                   WHERE s.customer_id=%s AND s.created_at>=clock_timestamp()-interval '24 hours' '''+clause+'''
                   GROUP BY s.id,w.receipt_number '''+having+''' ORDER BY s.created_at DESC,s.id DESC LIMIT 1''',args)
    row=cur.fetchone()
    if not row:return []
    receipt=row['receipt_number'] or f"legacy shipment {row['id']}"
    minutes=max(0,int(row['minutes_ago']))
    return [{'code':'POSSIBLE_DUPLICATE','requires_ack':True,
             'message':f"A matching shipment was recorded {minutes} minutes ago as {receipt} by {row['operator_id']}.",
             'message_es':f"Un envío similar se registró hace {minutes} {'minuto' if minutes == 1 else 'minutos'} como {receipt} por {row['operator_id']}.",
             'refs':dict(row)}]

def _sign(order_id,receipt,expires):
    key=os.environ.get('PACKING_SLIP_SIGNING_KEY')
    if not key and os.environ.get('ENVIRONMENT','').strip().lower()!='production':
        key=os.environ.get('API_KEY')
    if key and not key.strip(): key=None
    if not key: wt.fail(503,'SIGNING_NOT_CONFIGURED','Packing slip signing is unavailable.')
    return hmac.new(key.encode(),f'{order_id}:{receipt}:{expires}'.encode(),hashlib.sha256).hexdigest()

def packing_link(order_id,receipt):
    expires=int(time.time())+600
    return f'/sales/orders/{order_id}/packing-slip?'+urlencode({'receipt':receipt,'expires':expires,'sig':_sign(order_id,receipt,expires)})

def verify_packing_link(api,request,receipt,expires,sig):
    try: order_id=int(request.path_params['order_id'])
    except (ValueError,KeyError): wt.fail(401,'PACKING_SLIP_SIGNATURE_INVALID','Invalid packing slip link.')
    if expires is None or not int(time.time())<=expires<=int(time.time())+600 or not hmac.compare_digest(sig or '',_sign(order_id,receipt,expires)):
        wt.fail(401,'PACKING_SLIP_SIGNATURE_INVALID','Packing slip link is invalid or expired.')
    with api.get_transaction() as cur:
        cur.execute('''SELECT s.id FROM shipments s JOIN write_tickets t ON t.id=s.ticket_id
                       WHERE s.sales_order_id=%s AND t.receipt_number=%s AND t.status='committed' ''',(order_id,receipt))
        if not cur.fetchone(): wt.fail(401,'PACKING_SLIP_SIGNATURE_INVALID','Receipt does not identify this order shipment.')
    return True

def commit(api,cur,row,body,actor,request,entry_timing):
    if actor['id'] is not None:
        cur.execute('SELECT active FROM actors WHERE id=%s FOR SHARE',(actor['id'],))
        if not cur.fetchone()['active']: wt.fail(403,'ACTOR_INACTIVE','Preparing actor is inactive.')
    payload=a5.merge_confirmations(row['payload'],[r.dict() for r in body.lot_confirmations])
    if body.attachment_ref: wt.fail(422,'UNUSED_ATTACHMENT','Use attachment_ids for shipment evidence.')
    payload['attachment_ids']=sorted(set(payload.get('attachment_ids',[])+body.attachment_ids))
    cur.execute('SAVEPOINT ticket_post')
    try:
        ctx=validate(api,cur,row['action'],payload,actor,lock=True,ticket_id=row['id'])
        missing={w['code'] for w in ctx['warnings'] if w.get('requires_ack')}-set(body.acknowledged_warnings)
        if missing: wt.fail(409,'WARNING_NOT_ACKNOWLEDGED','Acknowledge current shipment warnings.',missing=sorted(missing))
        receipt=wt.allocate_receipt(cur,row['action'],ctx['occurred'].astimezone(api.PLANT_TIMEZONE).date())
        if row['action']=='ship_order':
            req=api.ShipOrderRequest(mode='commit',ship_all=False,lines=[api.ShipOrderLineRequest(**r) for r in payload['lines']])
            result=api._ship_order_commit_core(cur,request,payload['order_id'],req,ctx['occurred'],ctx['source'],ticket_id=row['id'],receipt_number=receipt,bol_reference=payload['bol_reference'],
                lot_allocations_by_line={r['line_id']:r['lot_allocations'] for r in payload['lines'] if r.get('lot_allocations') is not None})
        else:
            result=None
            for line in ctx['lines']:
                lot_code=next((p['lot_code'] for p in ctx['plan'] if p['lot_id']==line.get('lot_id')),None)
                req=api.ShipRequest(mode='commit',product_name=line['product']['name'],customer_name=ctx['customer']['name'],quantity_lb=line['quantity_lb'],lot_code=lot_code,
                                    order_reference=payload['order_reference'],force_standalone=True)
                posted=api._ship_commit_core(cur,req,request,ctx['occurred'],ctx['source'],product=line['product'],customer=ctx['customer'],ticket_id=row['id'],receipt_number=receipt,bol_reference=payload['bol_reference'])
                if result is None: result=posted
                else:
                    cur.execute('UPDATE shipment_lines SET shipment_id=%s WHERE shipment_id=%s',(result['shipment_id'],posted['shipment_id']))
                    cur.execute('DELETE FROM shipments WHERE id=%s',(posted['shipment_id'],))
                    result['quantity_shipped']+=posted['quantity_shipped'];result['lots_used']+=posted['lots_used']
        sid=result['shipment_id']
        if not sid: wt.fail(409,'SHIPMENT_REQUIRED','Shipping must record a shipment.')
        due=datetime.combine(ctx['occurred'].astimezone(api.PLANT_TIMEZONE).date(),day_time(23,59),api.PLANT_TIMEZONE)
        cur.execute("""UPDATE shipments SET bol_reference=%s,proof_status='photo_pending',proof_due_at=%s,ticket_id=%s,entered_by_actor_id=%s,client_source=%s WHERE id=%s""",
                    (payload['bol_reference'],due,row['id'],actor['id'],row['client_source'],sid))
        cur.execute('SELECT DISTINCT transaction_id FROM shipment_lines WHERE shipment_id=%s AND transaction_id IS NOT NULL ORDER BY transaction_id',(sid,))
        txns=[r['transaction_id'] for r in cur.fetchall()]
        cur.execute('SELECT DISTINCT lot_id FROM transaction_lines WHERE transaction_id=ANY(%s) AND lot_id IS NOT NULL ORDER BY lot_id',(txns,))
        actual_lots=[r['lot_id'] for r in cur.fetchall()]
        if set(actual_lots)!=set(r['lot_id'] for r in ctx['evidence']): wt.fail(422,'LOT_NOT_CONFIRMED','The shipping plan changed; confirm the actual lots.')
        for txn in txns:
            cur.execute('SELECT DISTINCT lot_id FROM transaction_lines WHERE transaction_id=%s AND quantity_lb<0',(txn,))
            consumed={r['lot_id'] for r in cur.fetchall()}
            a5.record_confirmations(cur,txn,[r for r in ctx['evidence'] if r['lot_id'] in consumed],actor['id'])
        for att in payload['attachment_ids']: attachments.attach_shipment(api,cur,sid,att,actor)
        result['attachments'] = attachments.for_receipt(cur, row['id'])
        cur.execute('SELECT proof_status,created_at FROM shipments WHERE id=%s',(sid,));shipment=cur.fetchone()
        result.update(receipt_number=receipt,ticket_id=row['id'],bol_reference=payload['bol_reference'],proof_status=shipment['proof_status'],proof_due_at=due,
                      replayed=False,state_changed=wt.canonical_hash(wt.json_value(api,ctx['state']))!=row['state_hash'],success=True,
                      entry_timing={**entry_timing,'entered_at':shipment['created_at'],'entered_by':actor,'late_entry_exception_id':None})
        late_id=None
        if permissions.opens_late_entry(actor,entry_timing):
            late_id=permissions.open_late_entry(cur,identity=actor,info=entry_timing,action=row['action'],
                transaction_id=txns[0] if txns else None,receipt_number=receipt,ticket_id=row['id'],
                entered_at=shipment['created_at'],client_source=row['client_source'])
        result['entry_timing']['late_entry_exception_id']=late_id
        day_warning=next((w for w in ctx['warnings'] if w['code']=='SHIP_NOT_SAME_DAY'),None)
        if day_warning:
            result['policy_acknowledgements']=[{'code':'SHIP_NOT_SAME_DAY','actor':dict(actor),
                'acknowledged_at':shipment['created_at'],**day_warning['refs']}]
        if row['action']=='ship_order': result['packing_slip_url']=packing_link(payload['order_id'],receipt)
        kosher_tiers.snapshot_response(cur,result,row['action'],ctx['draft'],lot_ids=actual_lots)
        result=wt.json_value(api,result)
    except HTTPException as exc:
        if exc.status_code>=500: raise
        cur.execute('ROLLBACK TO SAVEPOINT ticket_post')
        code=wt.blocker(exc)['code']
        if code in ('BOL_REQUIRED','SHIP_NOT_SAME_DAY','LOT_NOT_CONFIRMED','LOT_CONFIRMATION_MISMATCH','AMBIGUOUS_SUFFIX','UNUSED_LOT_CONFIRMATION','DUPLICATE_LOT_CONFIRMATION',
                    'ATTACHMENT_NOT_FOUND','ATTACHMENT_WRONG_ACTOR','ATTACHMENT_WRONG_CONTEXT','ATTACHMENT_KIND_INVALID','WARNING_NOT_ACKNOWLEDGED'):
            return api.JSONResponse(status_code=exc.status_code,content=wt.json_value(api,{'detail':exc.detail}))
        cur.execute("UPDATE write_tickets SET status='rejected',reject_reason=%s WHERE id=%s",(Json([wt.blocker(exc)]),row['id']))
        return api.JSONResponse(status_code=409,content={'detail':{'error_code':'TICKET_STALE','message':'Draft no longer valid; prepare again.','blockers':[wt.blocker(exc)]}})
    cur.execute('RELEASE SAVEPOINT ticket_post')
    ref={'shipment_ids':[sid],'transaction_ids':txns,'lot_ids':actual_lots,'order_id':payload.get('order_id')}
    cur.execute('''UPDATE write_tickets SET status='committed',committed_at=clock_timestamp(),receipt_number=%s,response=%s,result_ref=%s,acknowledged=%s WHERE id=%s''',
                (receipt,Json(result),Json(ref),Json(body.acknowledged_warnings),row['id']))
    return result

def shipment_gate(api,cur,order,close_reason):
    # Proof photo is due later; a recorded shipment is the close gate. Effective
    # stock lines exclude voided shipments; service-only orders still have a header.
    cur.execute('''SELECT s.id FROM shipments s WHERE s.sales_order_id=%s AND (
      EXISTS(SELECT 1 FROM shipment_lines sl JOIN ledger_current_transactions t ON t.id=sl.transaction_id
             WHERE sl.shipment_id=s.id AND t.effective_status='posted' AND sl.quantity_lb>0)
      OR NOT EXISTS(SELECT 1 FROM sales_order_lines sol JOIN products p ON p.id=sol.product_id
                    WHERE sol.sales_order_id=s.sales_order_id AND NOT COALESCE(p.is_service,false))) LIMIT 1''',(order['id'],))
    if cur.fetchone(): return []
    # Older recorded order shipments predate the unified header. Require a
    # positive, effective ship posting; counters and voided links do not count.
    cur.execute('''SELECT sos.id FROM sales_order_shipments sos
      JOIN sales_order_lines sol ON sol.id=sos.sales_order_line_id
      JOIN ledger_current_transactions t ON t.id=sos.transaction_id
      WHERE sol.sales_order_id=%s AND sos.quantity_lb>0
        AND t.type='ship' AND t.effective_status='posted' LIMIT 1''',(order['id'],))
    if cur.fetchone(): return []
    if close_reason=='shipped_not_recorded':
        cutoff=legacy_close_cutoff()
        if cutoff is not None and order['created_at'].astimezone(api.PLANT_TIMEZONE).date()>=cutoff:
            wt.fail(403,'LEGACY_CLOSE_CUTOFF','Orders created on or after the cutover date require a recorded shipment, including for the owner.')
        return []
    return [{'code':'SHIPMENT_REQUIRED','message':'Record a shipment before closing this order as shipped.'}]

def open_missing_photos(cur):
    # Header lock serializes the sweeper with late uploads. Unique index survives
    # repeated sweeps and prevents reopening waived/resolved historical exceptions.
    cur.execute("SELECT id FROM shipments WHERE ticket_id IS NOT NULL AND proof_status='photo_pending' AND proof_due_at<=clock_timestamp() ORDER BY id FOR UPDATE")
    ids=[r[0] if not isinstance(r,dict) else r['id'] for r in cur.fetchall()]
    added=0
    for sid in ids:
        cur.execute('''INSERT INTO exceptions(kind,severity,shipment_id,sales_order_id,ticket_id,receipt_number,owner_actor_id,due_at,detail)
                       SELECT 'BOL_PHOTO_MISSING','warn',s.id,s.sales_order_id,s.ticket_id,t.receipt_number,
                         (SELECT id FROM actors WHERE active AND role='owner' ORDER BY id LIMIT 1),s.proof_due_at,
                         jsonb_build_object('bol_reference',s.bol_reference,'proof_due_at',s.proof_due_at)
                       FROM shipments s JOIN write_tickets t ON t.id=s.ticket_id WHERE s.id=%s AND s.proof_status='photo_pending'
                       ON CONFLICT DO NOTHING''',(sid,));added+=cur.rowcount
    return added

def register_routes(app,api):
    @app.get('/ship/lots')
    def shipping_lots(request:Request, product_id:int=Query(...,gt=0), line_id:Optional[int]=Query(None,gt=0),
                      quantity_lb:Optional[float]=Query(None,gt=0,allow_inf_nan=False), _:bool=Depends(api.verify_api_key)):
        actor=wt.identity(api,request)
        permissions.require('ship_order' if line_id else 'ship_bulk',actor)
        with api.get_transaction() as cur:
            p=ticket_actions.product(api,cur,product_id,False)
            if line_id:
                cur.execute('SELECT product_id FROM sales_order_lines WHERE id=%s',(line_id,))
                row=cur.fetchone()
                if not row or row['product_id']!=product_id:
                    wt.fail(404,'LINE_NOT_FOUND','The order line does not match this product.')
            lots=kosher_tiers.sale_lots(cur,product_id,api.available_lots_for_product(cur,product_id,line_id))
            choices=[]
            for lot in lots:
                own=min(float(lot['reserved_this_line_lot'] or 0),
                        max(0.0,float(lot['on_hand'])-float(lot['reserved_others_lot'] or 0)))
                available=(own+float(lot['takeable'] or 0)) if line_id else (
                    float(lot['takeable'] or 0) if api._allocations_enforced() else float(lot['on_hand']))
                if available>api.BALANCE_EPSILON:
                    choices.append({'lot_id':lot['lot_id'],'lot_code':lot['lot_code'],'available_lb':available})
            suggested=[]
            if quantity_lb is not None:
                if line_id:
                    suggested=api._sales_order_ship_plan(cur,product_id,line_id,quantity_lb,lock=False,persist_expired=False)['lots']
                else:
                    suggested=[{'lot_id':r['lot']['lot_id'],'lot_code':r['lot']['lot_code'],'quantity_lb':r['quantity_lb']}
                               for r in api._takeable_deduction_plan(lots,quantity_lb)['lots']]
            return {'product_id':product_id,'product':p['name'],'case_size_lb':p.get('case_size_lb'),
                    'lots':choices,'suggested_lots':suggested}

    def prepare(action,req,request,order_id=None):
        actor=wt.identity(api,request);permissions.require(action,actor)
        now=api.get_plant_now();occurred=req.occurred_at or now
        if occurred.tzinfo is None: occurred=occurred.replace(tzinfo=api.PLANT_TIMEZONE)
        timing=permissions.require_backdating(actor,occurred,now)
        payload=wt.json_value(api,req.dict());payload['occurred_at']=occurred.isoformat()
        with api.get_transaction() as cur:
            if action=='ship_order':
                payload['order_id']=order_id
                if req.ship_all:
                    payload['lines']=[{'line_id':r['id'],'quantity_lb':max(0,float(r['quantity_lb'])-float(r['quantity_shipped_lb']))} for r in order_tickets.order_lines(cur,order_id) if r['line_status'] not in ('fulfilled','cancelled')]
                if len({r['line_id'] for r in payload['lines']})!=len(payload['lines']): wt.fail(422,'DUPLICATE_LINE','Supply each order line once.')
            elif action=='ship_standalone': payload['lines']=[{k:payload.pop(k) for k in ('product_id','lot_id','quantity_lb','lot_allocations')}]
            ctx=validate(api,cur,action,payload,actor)
            late=permissions.late_entry_warning(actor,timing)
            if late: ctx['warnings'].append(late)
            draft=ctx['draft'];draft['entry_timing']=timing
            return wt.issue_ticket(api,cur,actor=actor,action=action,payload=payload,draft=draft,state=ctx['state'],warnings=wt.json_value(api,ctx['warnings']),
                                   blockers=draft['blockers'],client_source=payload['client_source'],event_time=occurred)

    @app.post('/sales/orders/{order_id}/ship/prepare')
    def order(order_id:int,body:OrderShip,request:Request,_:bool=Depends(api.verify_api_key)):
        return prepare('ship_order',body,request,order_id)
    @app.post('/ship/prepare')
    def standalone(body:StandaloneShip,request:Request,_:bool=Depends(api.verify_api_key)):
        return prepare('ship_standalone',body,request)
    @app.post('/ship/bulk/prepare')
    def bulk(body:BulkShip,request:Request,_:bool=Depends(api.verify_api_key)):
        return prepare('ship_bulk',body,request)
