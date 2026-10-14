"""Sales document approval through the existing A7 create_order ticket lifecycle."""
from datetime import date
from typing import List, Optional

from fastapi import Depends, Request
from pydantic import Field, confloat, validator

import write_tickets as wt
import ticket_actions


class IntakeLine(wt.IdInput):
    product_id: wt.PositiveId
    quantity_lb: confloat(gt=0)
    quantity: Optional[confloat(gt=0)] = None
    unit: Optional[str] = None
    case_size_lb: Optional[confloat(gt=0)] = None
    unit_price: Optional[confloat(ge=0)] = None
    customer_item_code: Optional[str] = None
    customer_description: str
    save_alias: bool = True


class IntakeRequest(wt.IdInput):
    client_source: wt.ClientSource = 'api'
    document_id: wt.PositiveId
    customer_id: wt.PositiveId
    customer_po: str = Field(min_length=1)
    order_date: Optional[date] = None
    requested_ship_date: Optional[date] = None
    notes: Optional[str] = None
    lines: List[IntakeLine] = Field(min_length=1)
    allow_duplicate_po: bool = False

    @validator('customer_po')
    def required_po(cls, value):
        if not value.strip():
            wt.fail(422, 'PO_REQUIRED', 'customer_po is required')
        return value.strip()


def validate(api, cur, payload, *, lock=False):
    import order_tickets
    customer = order_tickets.customer(cur, payload['customer_id'], lock=lock)
    products = [ticket_actions.product(api, cur, pid, lock)
                for pid in sorted({line['product_id'] for line in payload['lines']})]
    # Same document -> PO lock order as the direct core.
    if lock:
        cur.execute('SELECT id FROM purchase_documents WHERE id=%s FOR UPDATE', (payload['document_id'],))
        api._lock_customer_po(cur, customer['id'], payload['customer_po'])
    duplicates = api._dedupe_existing_sales_orders(cur, customer['id'], payload['customer_po'])
    return {'blockers': [], 'customer': customer, 'duplicates': duplicates,
            'state': {'customer': customer, 'duplicates': duplicates,
                      'products': [{'id': p['id'], 'active': p['active']} for p in products]}}


def post(api, cur, payload, request, ticket_id, allow_duplicate):
    values = {k: v for k, v in payload.items() if k not in ('occurred_at', 'allow_duplicate_po')}
    req = api.SalesOrderApproveRequest(**values, force=allow_duplicate)
    result = api._approve_extracted_sales_order_core(cur, req, request)
    if ticket_id is not None:
        cur.execute('UPDATE sales_orders SET ticket_id=%s WHERE id=%s', (ticket_id, result['order_id']))
        cur.execute('UPDATE sales_order_lines SET ticket_id=%s WHERE sales_order_id=%s',
                    (ticket_id, result['order_id']))
    return result


def register_routes(app, api, prepare):
    @app.post('/sales/orders/extract/approve/prepare')
    def prepare_sales_document(req: IntakeRequest, request: Request,
                               _: bool = Depends(api.verify_api_key)):
        return prepare('create_order', req, request)
