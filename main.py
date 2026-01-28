from fastapi import FastAPI, HTTPException, Header, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, date
from zoneinfo import ZoneInfo
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import re

app = FastAPI(title="Factory Ledger System")

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
API_KEY = (os.getenv("API_KEY") or "").strip()

PLANT_TIMEZONE = ZoneInfo("America/New_York")
TIMEZONE_LABEL = "ET"


def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


# --- Pydantic Models ---

class CommandRequest(BaseModel):
    raw_text: str

class ReceiveRequest(BaseModel):
    product_name: str
    cases: int
    case_size_lb: float
    shipper_name: str
    bol_reference: str
    shipper_code_override: Optional[str] = None

class ReceivePreviewResponse(BaseModel):
    product_id: int
    product_name: str
    odoo_code: str
    cases: int
    case_size_lb: float
    total_lb: float
    shipper_name: str
    shipper_code: str
    shipper_code_auto: bool
    lot_code: str
    bol_reference: str
    preview_message: str

class ReceiveCommitResponse(BaseModel):
    success: bool
    transaction_id: int
    lot_id: int
    lot_code: str
    total_lb: float
    receipt_text: str
    receipt_html: str
    message: str

class ShipLotAllocation(BaseModel):
    lot_code: str
    lot_id: int
    available_lb: float
    use_lb: float

class ShipRequest(BaseModel):
    product_name: str
    quantity_lb: float
    customer_name: str
    order_reference: str
    lot_code: Optional[str] = None

class ShipPreviewResponse(BaseModel):
    product_id: int
    product_name: str
    odoo_code: str
    quantity_lb: float
    customer_name: str
    order_reference: str
    allocated_lots: List[ShipLotAllocation]
    total_allocated_lb: float
    sufficient: bool
    multi_lot: bool
    preview_message: str

class ShipCommitResponse(BaseModel):
    success: bool
    transaction_id: int
    lots_shipped: List[dict]
    total_quantity_lb: float
    slip_text: str
    slip_html: str
    message: str

class AdjustRequest(BaseModel):
    product_name: str
    quantity_lb: float
    lot_code: str
    reason: str

class AdjustPreviewResponse(BaseModel):
    product_id: int
    product_name: str
    odoo_code: str
    lot_code: str
    lot_id: int
    adjustment_lb: float
    current_balance_lb: float
    new_balance_lb: float
    reason: str
    preview_message: str

class AdjustCommitResponse(BaseModel):
    success: bool
    transaction_id: int
    lot_code: str
    adjustment_lb: float
    new_balance_lb: float
    reason: str
    message: str

class IngredientLotAllocation(BaseModel):
    lot_code: str
    lot_id: int
    available_lb: float
    use_lb: float

class IngredientAllocation(BaseModel):
    product_name: str
    product_id: int
    odoo_code: str
    required_lb: float
    allocated_lots: List[IngredientLotAllocation]
    sufficient: bool

class IngredientLotOverride(BaseModel):
    lot_code: str
    use_lb: float

class MakeRequest(BaseModel):
    product_name: str
    batches: float
    lot_code: str
    ingredient_lot_overrides: Optional[dict[str, List[IngredientLotOverride]]] = None

class MakePreviewResponse(BaseModel):
    batch_product_id: int
    batch_product_name: str
    batch_odoo_code: str
    batches: float
    batch_size_lb: float
    total_yield_lb: float
    output_lot_code: str
    ingredients: List[IngredientAllocation]
    all_sufficient: bool
    preview_message: str

class MakeCommitResponse(BaseModel):
    success: bool
    transaction_id: int
    batch_product_name: str
    batch_odoo_code: str
    batches: float
    produced_lb: float
    output_lot_code: str
    consumed: List[dict]
    message: str

class RepackRequest(BaseModel):
    source_product: str
    source_lot: str
    source_quantity_lb: float
    target_product: str
    target_quantity_lb: float
    target_lot_code: str
    notes: Optional[str] = None

class RepackPreviewResponse(BaseModel):
    source_product_id: int
    source_product_name: str
    source_odoo_code: str
    source_lot_code: str
    source_lot_id: int
    source_available_lb: float
    source_consume_lb: float
    target_product_id: int
    target_product_name: str
    target_odoo_code: str
    target_lot_code: str
    target_produce_lb: float
    yield_pct: float
    notes: Optional[str]
    preview_message: str

class RepackCommitResponse(BaseModel):
    success: bool
    transaction_id: int
    source_product: str
    source_lot: str
    consumed_lb: float
    target_product: str
    target_lot: str
    produced_lb: float
    message: str


# --- Helper Functions ---

def generate_shipper_code(shipper_name: str) -> str:
    words = shipper_name.strip().split()
    if not words:
        return "UNK"
    if len(words) == 1:
        code = words[0][:3].upper()
    else:
        code = words[0][:3].upper()
        for word in words[1:]:
            if word:
                code += word[0].upper()
    return code[:5]

def generate_lot_code(shipper_code: str, receive_date: date = None) -> str:
    if receive_date is None:
        receive_date = date.today()
    return f"{receive_date.strftime('%y-%m-%d')}-{shipper_code}"

def generate_lot_code_with_sequence(cur, product_id: int, shipper_code: str, receive_date: date = None) -> str:
    if receive_date is None:
        receive_date = date.today()
    
    base_code = f"{receive_date.strftime('%y-%m-%d')}-{shipper_code}"
    
    cur.execute("""
        SELECT lot_code FROM lots 
        WHERE product_id = %s AND lot_code LIKE %s
        ORDER BY lot_code DESC
    """, (product_id, f"{base_code}%"))
    existing = cur.fetchall()
    
    if not existing:
        return f"{base_code}-001"
    
    max_seq = 0
    for row in existing:
        code = row["lot_code"]
        if code == base_code:
            max_seq = max(max_seq, 0)
        elif code.startswith(base_code + "-"):
            try:
                seq = int(code.split("-")[-1])
                max_seq = max(max_seq, seq)
            except ValueError:
                pass
    
    return f"{base_code}-{max_seq + 1:03d}"

def localize_timestamp(dt: datetime) -> datetime:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
    return dt.astimezone(PLANT_TIMEZONE)

def format_timestamp(dt: datetime) -> tuple[str, str]:
    local_dt = localize_timestamp(dt)
    date_str = local_dt.strftime('%B %d, %Y')
    time_str = f"{local_dt.strftime('%I:%M %p')} {TIMEZONE_LABEL}"
    return date_str, time_str

def format_history_timestamp(ts):
    if ts is None:
        return None
    local_ts = localize_timestamp(ts)
    return {
        "iso": local_ts.isoformat(),
        "display": f"{local_ts.strftime('%b %d, %Y %I:%M %p')} {TIMEZONE_LABEL}"
    }

def get_available_lots_fifo(cur, product_id: int) -> list:
    cur.execute("""
        SELECT l.id as lot_id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available_lb
        FROM lots l
        LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
        WHERE l.product_id = %s
        GROUP BY l.id, l.lot_code
        HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
        ORDER BY l.id ASC
    """, (product_id,))
    return cur.fetchall()

def allocate_from_lots(available_lots: list, required_lb: float) -> tuple[list, bool]:
    allocations = []
    remaining = required_lb
    for lot in available_lots:
        if remaining <= 0:
            break
        use_lb = min(float(lot["available_lb"]), remaining)
        allocations.append({
            "lot_code": lot["lot_code"],
            "lot_id": lot["lot_id"],
            "available_lb": float(lot["available_lb"]),
            "use_lb": round(use_lb, 2)
        })
        remaining -= use_lb
    is_sufficient = remaining <= 0.001
    return allocations, is_sufficient

def allocate_with_overrides(cur, product_id: int, odoo_code: str, required_lb: float, overrides: Optional[List] = None) -> tuple[list, bool, str]:
    allocations = []
    remaining = required_lb
    used_lot_ids = set()
    error_msg = None
    
    if overrides:
        for override in overrides:
            lot_code = override.lot_code if hasattr(override, 'lot_code') else override['lot_code']
            use_lb = override.use_lb if hasattr(override, 'use_lb') else override['use_lb']
            
            cur.execute("""
                SELECT l.id as lot_id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available_lb
                FROM lots l
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE l.product_id = %s AND l.lot_code = %s
                GROUP BY l.id, l.lot_code
            """, (product_id, lot_code))
            lot = cur.fetchone()
            
            if not lot:
                error_msg = f"Override lot '{lot_code}' not found for product {odoo_code}"
                continue
            
            available = float(lot["available_lb"])
            if available < use_lb:
                error_msg = f"Override lot '{lot_code}' has {available:.0f} lb, requested {use_lb:.0f} lb"
                use_lb = min(available, use_lb)
            
            if use_lb > 0:
                allocations.append({
                    "lot_code": lot["lot_code"],
                    "lot_id": lot["lot_id"],
                    "available_lb": available,
                    "use_lb": round(use_lb, 2),
                    "override": True
                })
                used_lot_ids.add(lot["lot_id"])
                remaining -= use_lb
    
    if remaining > 0.001:
        available_lots = get_available_lots_fifo(cur, product_id)
        for lot in available_lots:
            if lot["lot_id"] in used_lot_ids:
                continue
            if remaining <= 0:
                break
            use_lb = min(float(lot["available_lb"]), remaining)
            allocations.append({
                "lot_code": lot["lot_code"],
                "lot_id": lot["lot_id"],
                "available_lb": float(lot["available_lb"]),
                "use_lb": round(use_lb, 2),
                "override": False
            })
            remaining -= use_lb
    
    is_sufficient = remaining <= 0.001
    return allocations, is_sufficient, error_msg


# --- Receipt/Slip Generators ---

def generate_receipt_text(product_name, odoo_code, cases, case_size_lb, total_lb, shipper_name, lot_code, bol_reference, timestamp):
    date_str, time_str = format_timestamp(timestamp)
    return f"""
╔══════════════════════════════════════════════════╗
║         CNS CONFECTIONERY PRODUCTS               ║
║              RECEIVING RECEIPT                   ║
╠══════════════════════════════════════════════════╣
║  Date:        {date_str:<32}║
║  Time:        {time_str:<32}║
╠══════════════════════════════════════════════════╣
║  LOT CODE:    {lot_code:<32}║
╠══════════════════════════════════════════════════╣
║  Product:     {product_name:<32}║
║  Odoo Code:   {odoo_code:<32}║
║  Quantity:    {f'{cases} cases × {case_size_lb:.0f} lb = {total_lb:,.0f} lb':<32}║
║  Shipper:     {shipper_name:<32}║
║  BOL Ref:     {bol_reference:<32}║
╠══════════════════════════════════════════════════╣
║  ☐ Label pallets with lot code                   ║
║  ☐ Store in designated area                      ║
╚══════════════════════════════════════════════════╝
""".strip()

def generate_receipt_html(product_name, odoo_code, cases, case_size_lb, total_lb, shipper_name, lot_code, bol_reference, timestamp):
    date_str, time_str = format_timestamp(timestamp)
    return f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 400px; margin: 20px auto; }}
        .receipt {{ border: 2px solid #333; padding: 20px; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 15px; }}
        .header h1 {{ margin: 0; font-size: 16px; }}
        .header h2 {{ margin: 5px 0 0 0; font-size: 14px; font-weight: normal; }}
        .lot-code {{ background: #f0f0f0; padding: 15px; text-align: center; margin: 15px 0; border: 1px solid #333; }}
        .lot-code span {{ font-size: 24px; font-weight: bold; font-family: monospace; }}
        .details {{ margin: 15px 0; }}
        .row {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px dotted #ccc; }}
        .label {{ font-weight: bold; }}
        .checklist {{ margin-top: 20px; padding-top: 15px; border-top: 2px solid #333; }}
        .checklist div {{ margin: 8px 0; }}
        .checkbox {{ display: inline-block; width: 16px; height: 16px; border: 1px solid #333; margin-right: 10px; }}
    </style>
</head>
<body>
    <div class="receipt">
        <div class="header"><h1>CNS CONFECTIONERY PRODUCTS</h1><h2>RECEIVING RECEIPT</h2></div>
        <div class="details">
            <div class="row"><span class="label">Date:</span><span>{date_str}</span></div>
            <div class="row"><span class="label">Time:</span><span>{time_str}</span></div>
        </div>
        <div class="lot-code"><div>LOT CODE</div><span>{lot_code}</span></div>
        <div class="details">
            <div class="row"><span class="label">Product:</span><span>{product_name}</span></div>
            <div class="row"><span class="label">Odoo Code:</span><span>{odoo_code}</span></div>
            <div class="row"><span class="label">Quantity:</span><span>{cases} cases × {case_size_lb:.0f} lb = {total_lb:,.0f} lb</span></div>
            <div class="row"><span class="label">Shipper:</span><span>{shipper_name}</span></div>
            <div class="row"><span class="label">BOL Ref:</span><span>{bol_reference}</span></div>
        </div>
        <div class="checklist">
            <div><span class="checkbox"></span>Label pallets with lot code</div>
            <div><span class="checkbox"></span>Store in designated area</div>
        </div>
    </div>
</body>
</html>"""

def generate_packing_slip_text(product_name, odoo_code, quantity_lb, customer_name, lot_code, order_reference, timestamp):
    date_str, time_str = format_timestamp(timestamp)
    return f"""
╔══════════════════════════════════════════════════╗
║         CNS CONFECTIONERY PRODUCTS               ║
║                PACKING SLIP                      ║
╠══════════════════════════════════════════════════╣
║  Date:        {date_str:<32}║
║  Time:        {time_str:<32}║
╠══════════════════════════════════════════════════╣
║  SHIP TO:     {customer_name:<32}║
║  Order Ref:   {order_reference:<32}║
╠══════════════════════════════════════════════════╣
║  Product:     {product_name:<32}║
║  Odoo Code:   {odoo_code:<32}║
║  Quantity:    {f'{quantity_lb:,.0f} lb':<32}║
║  Lot Code:    {lot_code:<32}║
╠══════════════════════════════════════════════════╣
║  ☐ Verify quantity                               ║
║  ☐ Check lot labels match                        ║
║  ☐ Load and secure                               ║
╚══════════════════════════════════════════════════╝
""".strip()

def generate_packing_slip_html(product_name, odoo_code, quantity_lb, customer_name, lot_code, order_reference, timestamp):
    date_str, time_str = format_timestamp(timestamp)
    return f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 400px; margin: 20px auto; }}
        .slip {{ border: 2px solid #333; padding: 20px; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 15px; }}
        .header h1 {{ margin: 0; font-size: 16px; }}
        .header h2 {{ margin: 5px 0 0 0; font-size: 14px; font-weight: normal; }}
        .customer {{ background: #f0f0f0; padding: 15px; margin: 15px 0; border: 1px solid #333; }}
        .customer .name {{ font-size: 20px; font-weight: bold; }}
        .customer .ref {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .details {{ margin: 15px 0; }}
        .row {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px dotted #ccc; }}
        .label {{ font-weight: bold; }}
        .lot-code {{ background: #fff3cd; padding: 10px; text-align: center; margin: 15px 0; border: 1px solid #333; }}
        .lot-code span {{ font-size: 20px; font-weight: bold; font-family: monospace; }}
        .checklist {{ margin-top: 20px; padding-top: 15px; border-top: 2px solid #333; }}
        .checklist div {{ margin: 8px 0; }}
        .checkbox {{ display: inline-block; width: 16px; height: 16px; border: 1px solid #333; margin-right: 10px; }}
    </style>
</head>
<body>
    <div class="slip">
        <div class="header"><h1>CNS CONFECTIONERY PRODUCTS</h1><h2>PACKING SLIP</h2></div>
        <div class="details">
            <div class="row"><span class="label">Date:</span><span>{date_str}</span></div>
            <div class="row"><span class="label">Time:</span><span>{time_str}</span></div>
        </div>
        <div class="customer"><div class="name">{customer_name}</div><div class="ref">Order: {order_reference}</div></div>
        <div class="details">
            <div class="row"><span class="label">Product:</span><span>{product_name}</span></div>
            <div class="row"><span class="label">Odoo Code:</span><span>{odoo_code}</span></div>
            <div class="row"><span class="label">Quantity:</span><span>{quantity_lb:,.0f} lb</span></div>
        </div>
        <div class="lot-code"><div>LOT CODE</div><span>{lot_code}</span></div>
        <div class="checklist">
            <div><span class="checkbox"></span>Verify quantity</div>
            <div><span class="checkbox"></span>Check lot labels match</div>
            <div><span class="checkbox"></span>Load and secure</div>
        </div>
    </div>
</body>
</html>"""

def generate_packing_slip_text_multi(product_name, odoo_code, lots_shipped, customer_name, order_reference, timestamp):
    date_str, time_str = format_timestamp(timestamp)
    total_lb = sum(l["quantity_lb"] for l in lots_shipped)
    
    if len(lots_shipped) == 1:
        lot_section = f"║  Lot Code:    {lots_shipped[0]['lot_code']:<32}║"
    else:
        lot_lines = ["║  Lot Codes:                                      ║"]
        for lot in lots_shipped:
            line = f"║    • {lot['lot_code']}: {lot['quantity_lb']:,.0f} lb"
            line = line + " " * (51 - len(line)) + "║"
            lot_lines.append(line)
        lot_section = "\n".join(lot_lines)
    
    return f"""
╔══════════════════════════════════════════════════╗
║         CNS CONFECTIONERY PRODUCTS               ║
║                PACKING SLIP                      ║
╠══════════════════════════════════════════════════╣
║  Date:        {date_str:<32}║
║  Time:        {time_str:<32}║
╠══════════════════════════════════════════════════╣
║  SHIP TO:     {customer_name:<32}║
║  Order Ref:   {order_reference:<32}║
╠══════════════════════════════════════════════════╣
║  Product:     {product_name:<32}║
║  Odoo Code:   {odoo_code:<32}║
║  Quantity:    {f'{total_lb:,.0f} lb':<32}║
{lot_section}
╠══════════════════════════════════════════════════╣
║  ☐ Verify quantity                               ║
║  ☐ Check lot labels match                        ║
║  ☐ Load and secure                               ║
╚══════════════════════════════════════════════════╝
""".strip()

def generate_packing_slip_html_multi(product_name, odoo_code, lots_shipped, customer_name, order_reference, timestamp):
    date_str, time_str = format_timestamp(timestamp)
    total_lb = sum(l["quantity_lb"] for l in lots_shipped)
    
    if len(lots_shipped) == 1:
        lot_html = f'<div class="lot-code"><div>LOT CODE</div><span>{lots_shipped[0]["lot_code"]}</span></div>'
    else:
        lot_items = "".join([
            f'<div class="lot-item"><span class="lot-name">{lot["lot_code"]}</span><span class="lot-qty">{lot["quantity_lb"]:,.0f} lb</span></div>' 
            for lot in lots_shipped
        ])
        lot_html = f'<div class="lot-codes"><div class="lot-header">LOT CODES</div>{lot_items}</div>'
    
    return f"""<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 400px; margin: 20px auto; }}
        .slip {{ border: 2px solid #333; padding: 20px; }}
        .header {{ text-align: center; border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 15px; }}
        .header h1 {{ margin: 0; font-size: 16px; }}
        .header h2 {{ margin: 5px 0 0 0; font-size: 14px; font-weight: normal; }}
        .customer {{ background: #f0f0f0; padding: 15px; margin: 15px 0; border: 1px solid #333; }}
        .customer .name {{ font-size: 20px; font-weight: bold; }}
        .customer .ref {{ font-size: 14px; color: #666; margin-top: 5px; }}
        .details {{ margin: 15px 0; }}
        .row {{ display: flex; justify-content: space-between; padding: 5px 0; border-bottom: 1px dotted #ccc; }}
        .label {{ font-weight: bold; }}
        .lot-code {{ background: #fff3cd; padding: 10px; text-align: center; margin: 15px 0; border: 1px solid #333; }}
        .lot-code span {{ font-size: 20px; font-weight: bold; font-family: monospace; }}
        .lot-codes {{ background: #fff3cd; padding: 10px; margin: 15px 0; border: 1px solid #333; }}
        .lot-codes .lot-header {{ text-align: center; font-weight: bold; margin-bottom: 10px; }}
        .lot-codes .lot-item {{ display: flex; justify-content: space-between; padding: 5px 10px; font-family: monospace; }}
        .lot-codes .lot-name {{ font-weight: bold; }}
        .checklist {{ margin-top: 20px; padding-top: 15px; border-top: 2px solid #333; }}
        .checklist div {{ margin: 8px 0; }}
        .checkbox {{ display: inline-block; width: 16px; height: 16px; border: 1px solid #333; margin-right: 10px; }}
    </style>
</head>
<body>
    <div class="slip">
        <div class="header"><h1>CNS CONFECTIONERY PRODUCTS</h1><h2>PACKING SLIP</h2></div>
        <div class="details">
            <div class="row"><span class="label">Date:</span><span>{date_str}</span></div>
            <div class="row"><span class="label">Time:</span><span>{time_str}</span></div>
        </div>
        <div class="customer"><div class="name">{customer_name}</div><div class="ref">Order: {order_reference}</div></div>
        <div class="details">
            <div class="row"><span class="label">Product:</span><span>{product_name}</span></div>
            <div class="row"><span class="label">Odoo Code:</span><span>{odoo_code}</span></div>
            <div class="row"><span class="label">Quantity:</span><span>{total_lb:,.0f} lb</span></div>
        </div>
        {lot_html}
        <div class="checklist">
            <div><span class="checkbox"></span>Verify quantity</div>
            <div><span class="checkbox"></span>Check lot labels match</div>
            <div><span class="checkbox"></span>Load and secure</div>
        </div>
    </div>
</body>
</html>"""


# --- API Endpoints ---

@app.get("/")
def root():
    return {
        "name": "Factory Ledger System",
        "version": "1.4.0",
        "status": "online",
        "timezone": f"{PLANT_TIMEZONE} ({TIMEZONE_LABEL})",
        "endpoints": {
            "GET /health": "Health check",
            "GET /inventory/{item_name}": "Get current inventory",
            "GET /products/search": "Search products",
            "GET /transactions/history": "Transaction history",
            "GET /bom/{product}": "Get BOM/recipe",
            "POST /receive/preview": "Preview receive",
            "POST /receive/commit": "Commit receive",
            "POST /ship/preview": "Preview ship (supports multi-lot)",
            "POST /ship/commit": "Commit ship (supports multi-lot)",
            "POST /adjust/preview": "Preview adjustment",
            "POST /adjust/commit": "Commit adjustment",
            "POST /make/preview": "Preview production",
            "POST /make/commit": "Commit production",
            "POST /repack/preview": "Preview repack/rework",
            "POST /repack/commit": "Commit repack/rework",
            "GET /trace/batch/{lot}": "Trace batch ingredients",
            "GET /trace/ingredient/{lot}": "Trace ingredient usage",
            "GET /bom/products": "List BOM products",
            "GET /bom/batches": "List all batches",
            "GET /bom/batches/{ref}/formula": "Get batch formula",
            "GET /bom/finished/{ref}/bom": "Get finished product BOM",
            "POST /bom/production/requirements": "Calculate production requirements",
            "GET /bom/allergens": "List allergens",
            "GET /bom/search": "Search BOM products",
        },
    }


@app.get("/health")
def health_check():
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
        conn.close()
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "database": "disconnected", "error": str(e)})


@app.get("/inventory/{item_name}")
def get_inventory(item_name: str, _: bool = Depends(verify_api_key)):
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if item_name.isdigit():
                    cur.execute("SELECT p.name, p.odoo_code, COALESCE(SUM(tl.quantity_lb), 0) AS total FROM products p LEFT JOIN transaction_lines tl ON tl.product_id = p.id WHERE p.odoo_code = %s GROUP BY p.id", (item_name,))
                    result = cur.fetchone()
                else:
                    cur.execute("SELECT p.name, p.odoo_code, COALESCE(SUM(tl.quantity_lb), 0) AS total FROM products p LEFT JOIN transaction_lines tl ON tl.product_id = p.id WHERE LOWER(p.name) = LOWER(%s) GROUP BY p.id", (item_name,))
                    result = cur.fetchone()
                    if not result:
                        cur.execute("SELECT p.name, p.odoo_code, COALESCE(SUM(tl.quantity_lb), 0) AS total FROM products p LEFT JOIN transaction_lines tl ON tl.product_id = p.id WHERE p.name ILIKE %s GROUP BY p.id ORDER BY LENGTH(p.name) ASC LIMIT 1", (f"%{item_name}%",))
                        result = cur.fetchone()
        if not result:
            return JSONResponse(status_code=404, content={"error": "Product not found", "query": item_name})
        return {"item": result["name"], "odoo_code": result["odoo_code"], "on_hand_lb": float(result["total"])}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/products/search")
def search_products(q: str, limit: int = Query(default=20, ge=1, le=100), _: bool = Depends(verify_api_key)):
    if not q or len(q.strip()) < 2:
        return JSONResponse(status_code=400, content={"error": "Query must be at least 2 characters"})
    query = q.strip()
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if query.isdigit():
                    cur.execute("SELECT name, odoo_code, type FROM products WHERE odoo_code = %s LIMIT %s", (query, limit))
                    results = cur.fetchall()
                    if results:
                        return {"query": query, "matches": results}
                cur.execute("""
                    SELECT name, odoo_code, type 
                    FROM products 
                    WHERE name ILIKE %s 
                    ORDER BY 
                        CASE WHEN LOWER(name) = LOWER(%s) THEN 0 ELSE 1 END,
                        LENGTH(name) ASC 
                    LIMIT %s
                """, (f"%{query}%", query, limit))
                results = cur.fetchall()
        return {"query": query, "matches": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/transactions/history")
def get_transaction_history(_: bool = Depends(verify_api_key), limit: int = Query(default=10, ge=1, le=100), type: Optional[str] = Query(default=None), product: Optional[str] = Query(default=None)):
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = "SELECT t.id as transaction_id, t.type, t.timestamp, t.notes, t.adjust_reason, json_agg(json_build_object('product', p.name, 'odoo_code', p.odoo_code, 'lot', l.lot_code, 'quantity_lb', tl.quantity_lb) ORDER BY tl.id) as lines FROM transactions t LEFT JOIN transaction_lines tl ON tl.transaction_id = t.id LEFT JOIN products p ON p.id = tl.product_id LEFT JOIN lots l ON l.id = tl.lot_id"
                conditions, params = [], []
                if type:
                    conditions.append("t.type = %s")
                    params.append(type.lower())
                if product:
                    if product.isdigit():
                        conditions.append("t.id IN (SELECT DISTINCT tl2.transaction_id FROM transaction_lines tl2 JOIN products p2 ON p2.id = tl2.product_id WHERE p2.odoo_code = %s)")
                    else:
                        conditions.append("t.id IN (SELECT DISTINCT tl2.transaction_id FROM transaction_lines tl2 JOIN products p2 ON p2.id = tl2.product_id WHERE p2.name ILIKE %s)")
                        product = f"%{product}%"
                    params.append(product)
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                query += " GROUP BY t.id ORDER BY t.timestamp DESC, t.id DESC LIMIT %s"
                params.append(limit)
                cur.execute(query, params)
                transactions = cur.fetchall()
        result = [{"transaction_id": tx["transaction_id"], "type": tx["type"], "timestamp": format_history_timestamp(tx["timestamp"]), "notes": tx["notes"], "adjust_reason": tx.get("adjust_reason"), "lines": [l for l in (tx["lines"] or []) if l.get("product")]} for tx in transactions]
        return {"count": len(result), "filters": {"limit": limit, "type": type, "product": product}, "transactions": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/recipe/{product}")
def get_bom(product: str, expand: bool = Query(default=False), _: bool = Depends(verify_api_key)):
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                batch = None
                if product.isdigit():
                    cur.execute("SELECT id, name, odoo_code, default_batch_lb FROM products WHERE odoo_code = %s", (product,))
                    batch = cur.fetchone()
                if not batch:
                    cur.execute("SELECT id, name, odoo_code, default_batch_lb FROM products WHERE LOWER(name) = LOWER(%s)", (product,))
                    batch = cur.fetchone()
                if not batch:
                    cur.execute("SELECT id, name, odoo_code, default_batch_lb FROM products WHERE name ILIKE %s AND name ILIKE 'Batch%%' ORDER BY LENGTH(name) ASC LIMIT 5", (f"%{product}%",))
                    matches = cur.fetchall()
                    if len(matches) == 1:
                        batch = matches[0]
                    elif len(matches) > 1:
                        return {"status": "multiple_matches", "query": product, "message": "Multiple products match.", "suggestions": [{"name": m["name"], "odoo_code": m["odoo_code"]} for m in matches]}
                if not batch:
                    return JSONResponse(status_code=404, content={"error": f"No batch products found matching: {product}"})
                cur.execute("SELECT p.name AS ingredient, p.odoo_code AS ingredient_code, bf.quantity_lb FROM batch_formulas bf JOIN products p ON p.id = bf.ingredient_product_id WHERE bf.product_id = %s ORDER BY bf.quantity_lb DESC", (batch["id"],))
                ingredients = cur.fetchall()
                if not ingredients:
                    return JSONResponse(status_code=404, content={"error": f"No BOM found for: {batch['name']}"})
                result = {"product": batch["name"], "odoo_code": batch["odoo_code"], "batch_size_lb": float(batch["default_batch_lb"]) if batch["default_batch_lb"] else None, "ingredient_count": len(ingredients), "ingredients": [{"name": ing["ingredient"], "odoo_code": ing["ingredient_code"], "quantity_lb": float(ing["quantity_lb"])} for ing in ingredients]}
                if expand:
                    expanded = expand_bom(cur, ingredients, batch.get("default_batch_lb"))
                    if expanded:
                        result["expanded_ingredients"] = expanded.get("expanded_ingredients")
                return result
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# --- RECEIVE ENDPOINTS ---

@app.post("/receive/preview", response_model=ReceivePreviewResponse)
def receive_preview(req: ReceiveRequest, _: bool = Depends(verify_api_key)):
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id, name, odoo_code, uom FROM products WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) = LOWER(%s) LIMIT 5", (f"%{req.product_name}%", req.product_name))
                products = cur.fetchall()
                if not products:
                    raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")
                if len(products) > 1:
                    raise HTTPException(status_code=400, detail=f"Multiple products match '{req.product_name}'. Please be more specific:\n" + "\n".join([f"• {p['name']} ({p['odoo_code']})" for p in products]))
                product = products[0]
        shipper_code = req.shipper_code_override.upper()[:5] if req.shipper_code_override else generate_shipper_code(req.shipper_name)
        shipper_code_auto = not req.shipper_code_override
        lot_code = generate_lot_code(shipper_code)
        total_lb = req.cases * req.case_size_lb
        code_note = "(auto-generated)" if shipper_code_auto else "(override)"
        preview_message = f"📦 RECEIVE PREVIEW\n────────────────────────────────\nProduct:      {product['name']} ({product['odoo_code']})\nQuantity:     {req.cases} cases × {req.case_size_lb:.0f} lb = {total_lb:,.0f} lb\nShipper:      {req.shipper_name}\nShipper Code: {shipper_code} {code_note}\nLot Code:     {lot_code}\nBOL Ref:      {req.bol_reference}\n────────────────────────────────\n✓ Say \"confirm\" to proceed\n✎ Or correct any errors"
        return ReceivePreviewResponse(product_id=product["id"], product_name=product["name"], odoo_code=product["odoo_code"], cases=req.cases, case_size_lb=req.case_size_lb, total_lb=total_lb, shipper_name=req.shipper_name, shipper_code=shipper_code, shipper_code_auto=shipper_code_auto, lot_code=lot_code, bol_reference=req.bol_reference, preview_message=preview_message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/receive/commit", response_model=ReceiveCommitResponse)
def receive_commit(req: ReceiveRequest, _: bool = Depends(verify_api_key)):
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        conn.autocommit = False
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, name, odoo_code FROM products WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) = LOWER(%s) LIMIT 1", (f"%{req.product_name}%", req.product_name))
        product = cur.fetchone()
        if not product:
            raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")
        shipper_code = req.shipper_code_override.upper()[:5] if req.shipper_code_override else generate_shipper_code(req.shipper_name)
        lot_code = generate_lot_code_with_sequence(cur, product["id"], shipper_code)
        total_lb = req.cases * req.case_size_lb
        timestamp = datetime.now()
        cur.execute("SELECT id FROM lots WHERE lot_code = %s AND product_id = %s", (lot_code, product["id"]))
        lot_row = cur.fetchone()
        lot_id = lot_row["id"] if lot_row else None
        if not lot_id:
            cur.execute("INSERT INTO lots (lot_code, product_id) VALUES (%s, %s) RETURNING id", (lot_code, product["id"]))
            lot_id = cur.fetchone()["id"]
        cur.execute("INSERT INTO transactions (type, bol_reference, shipper_name, shipper_code, cases_received, case_size_lb) VALUES ('receive', %s, %s, %s, %s, %s) RETURNING id", (req.bol_reference, req.shipper_name, shipper_code, req.cases, req.case_size_lb))
        transaction_id = cur.fetchone()["id"]
        cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (transaction_id, product["id"], lot_id, total_lb))
        conn.commit()
        cur.close()
        receipt_text = generate_receipt_text(product["name"], product["odoo_code"], req.cases, req.case_size_lb, total_lb, req.shipper_name, lot_code, req.bol_reference, timestamp)
        receipt_html = generate_receipt_html(product["name"], product["odoo_code"], req.cases, req.case_size_lb, total_lb, req.shipper_name, lot_code, req.bol_reference, timestamp)
        return ReceiveCommitResponse(success=True, transaction_id=transaction_id, lot_id=lot_id, lot_code=lot_code, total_lb=total_lb, receipt_text=receipt_text, receipt_html=receipt_html, message=f"✅ Received {total_lb:,.0f} lb {product['name']} into lot {lot_code}")
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn: conn.close()


# --- SHIP ENDPOINTS ---

@app.post("/ship/preview", response_model=ShipPreviewResponse)
def ship_preview(req: ShipRequest, _: bool = Depends(verify_api_key)):
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id, name, odoo_code FROM products WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) = LOWER(%s) LIMIT 5", (f"%{req.product_name}%", req.product_name))
                products = cur.fetchall()
                if not products:
                    raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")
                if len(products) > 1:
                    raise HTTPException(status_code=400, detail=f"Multiple products match '{req.product_name}'. Please be more specific:\n" + "\n".join([f"• {p['name']} ({p['odoo_code']})" for p in products]))
                product = products[0]
                lots_with_inventory = get_available_lots_fifo(cur, product["id"])
                if not lots_with_inventory:
                    raise HTTPException(status_code=400, detail=f"No inventory available for {product['name']}")
                if req.lot_code:
                    selected_lot = next((l for l in lots_with_inventory if l["lot_code"] == req.lot_code), None)
                    if not selected_lot:
                        raise HTTPException(status_code=400, detail=f"Lot '{req.lot_code}' not found or has no inventory. Available lots:\n" + "\n".join([f"• {l['lot_code']} ({l['available_lb']:,.0f} lb)" for l in lots_with_inventory]))
                    if float(selected_lot["available_lb"]) < req.quantity_lb:
                        raise HTTPException(status_code=400, detail=f"Insufficient inventory in lot {selected_lot['lot_code']}. Requested: {req.quantity_lb:,.0f} lb, Available: {selected_lot['available_lb']:,.0f} lb.\n\nRemove lot specification to allow multi-lot shipping, or choose from:\n" + "\n".join([f"• {l['lot_code']} ({l['available_lb']:,.0f} lb)" for l in lots_with_inventory]))
                    allocations = [ShipLotAllocation(lot_code=selected_lot["lot_code"], lot_id=selected_lot["lot_id"], available_lb=float(selected_lot["available_lb"]), use_lb=req.quantity_lb)]
                    multi_lot = False
                else:
                    allocations = []
                    remaining = req.quantity_lb
                    for lot in lots_with_inventory:
                        if remaining <= 0:
                            break
                        use_lb = min(float(lot["available_lb"]), remaining)
                        allocations.append(ShipLotAllocation(lot_code=lot["lot_code"], lot_id=lot["lot_id"], available_lb=float(lot["available_lb"]), use_lb=round(use_lb, 2)))
                        remaining -= use_lb
                    multi_lot = len(allocations) > 1
                total_allocated = sum(a.use_lb for a in allocations)
                sufficient = total_allocated >= req.quantity_lb - 0.001
                if multi_lot:
                    lot_lines = "\n".join([f"    • {a.lot_code}: {a.use_lb:,.0f} lb (of {a.available_lb:,.0f} lb)" for a in allocations])
                    lot_section = f"Lots (FIFO):\n{lot_lines}"
                else:
                    a = allocations[0]
                    lot_note = "(specified)" if req.lot_code else "(FIFO)"
                    lot_section = f"Lot:          {a.lot_code} {lot_note}\nAvailable:    {a.available_lb:,.0f} lb in this lot"
                if sufficient:
                    status = "✓ Say \"confirm\" to proceed"
                else:
                    status = f"⚠️ INSUFFICIENT: Need {req.quantity_lb:,.0f} lb, can only allocate {total_allocated:,.0f} lb"
                preview_message = f"""📦 SHIP PREVIEW
────────────────────────────────
Product:      {product['name']} ({product['odoo_code']})
Quantity:     {req.quantity_lb:,.0f} lb
Customer:     {req.customer_name}
Order Ref:    {req.order_reference}
{lot_section}
────────────────────────────────
{status}"""
                return ShipPreviewResponse(product_id=product["id"], product_name=product["name"], odoo_code=product["odoo_code"], quantity_lb=req.quantity_lb, customer_name=req.customer_name, order_reference=req.order_reference, allocated_lots=allocations, total_allocated_lb=total_allocated, sufficient=sufficient, multi_lot=multi_lot, preview_message=preview_message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ship/commit", response_model=ShipCommitResponse)
def ship_commit(req: ShipRequest, _: bool = Depends(verify_api_key)):
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        conn.autocommit = False
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, name, odoo_code FROM products WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) = LOWER(%s) LIMIT 1", (f"%{req.product_name}%", req.product_name))
        product = cur.fetchone()
        if not product:
            raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")
        lots_with_inventory = get_available_lots_fifo(cur, product["id"])
        if not lots_with_inventory:
            raise HTTPException(status_code=400, detail=f"No inventory available for {product['name']}")
        if req.lot_code:
            selected_lot = next((l for l in lots_with_inventory if l["lot_code"] == req.lot_code), None)
            if not selected_lot:
                raise HTTPException(status_code=400, detail=f"Lot '{req.lot_code}' not found or has no inventory")
            if float(selected_lot["available_lb"]) < req.quantity_lb:
                raise HTTPException(status_code=400, detail=f"Insufficient inventory in lot {selected_lot['lot_code']}. Requested: {req.quantity_lb:,.0f} lb, Available: {selected_lot['available_lb']:,.0f} lb")
            allocations = [{"lot_code": selected_lot["lot_code"], "lot_id": selected_lot["lot_id"], "use_lb": req.quantity_lb}]
        else:
            allocations = []
            remaining = req.quantity_lb
            for lot in lots_with_inventory:
                if remaining <= 0:
                    break
                use_lb = min(float(lot["available_lb"]), remaining)
                allocations.append({"lot_code": lot["lot_code"], "lot_id": lot["lot_id"], "use_lb": round(use_lb, 2)})
                remaining -= use_lb
            if remaining > 0.001:
                total_allocated = sum(a["use_lb"] for a in allocations)
                raise HTTPException(status_code=400, detail=f"Insufficient total inventory. Need {req.quantity_lb:,.0f} lb, only {total_allocated:,.0f} lb available.")
        timestamp = datetime.now()
        cur.execute("INSERT INTO transactions (type, customer_name, order_reference) VALUES ('ship', %s, %s) RETURNING id", (req.customer_name, req.order_reference))
        transaction_id = cur.fetchone()["id"]
        lots_shipped = []
        for alloc in allocations:
            cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (transaction_id, product["id"], alloc["lot_id"], -alloc["use_lb"]))
            lots_shipped.append({"lot_code": alloc["lot_code"], "quantity_lb": alloc["use_lb"]})
        conn.commit()
        cur.close()
        total_shipped = sum(l["quantity_lb"] for l in lots_shipped)
        slip_text = generate_packing_slip_text_multi(product["name"], product["odoo_code"], lots_shipped, req.customer_name, req.order_reference, timestamp)
        slip_html = generate_packing_slip_html_multi(product["name"], product["odoo_code"], lots_shipped, req.customer_name, req.order_reference, timestamp)
        lot_summary = ", ".join([f"{l['lot_code']} ({l['quantity_lb']:,.0f} lb)" for l in lots_shipped])
        return ShipCommitResponse(success=True, transaction_id=transaction_id, lots_shipped=lots_shipped, total_quantity_lb=total_shipped, slip_text=slip_text, slip_html=slip_html, message=f"✅ Shipped {total_shipped:,.0f} lb {product['name']} to {req.customer_name}\nLots: {lot_summary}")
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn: conn.close()


# --- ADJUST ENDPOINTS ---

@app.post("/adjust/preview", response_model=AdjustPreviewResponse)
def adjust_preview(req: AdjustRequest, _: bool = Depends(verify_api_key)):
    if not req.reason or len(req.reason.strip()) < 3:
        raise HTTPException(status_code=400, detail="Reason is required (minimum 3 characters)")
    if req.quantity_lb == 0:
        raise HTTPException(status_code=400, detail="Adjustment quantity cannot be zero")
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id, name, odoo_code FROM products WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) = LOWER(%s) LIMIT 5", (f"%{req.product_name}%", req.product_name))
                products = cur.fetchall()
                if not products:
                    raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")
                if len(products) > 1:
                    raise HTTPException(status_code=400, detail=f"Multiple products match '{req.product_name}'. Please be more specific:\n" + "\n".join([f"• {p['name']} ({p['odoo_code']})" for p in products]))
                product = products[0]
                cur.execute("SELECT l.id as lot_id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as current_balance FROM lots l LEFT JOIN transaction_lines tl ON tl.lot_id = l.id WHERE l.product_id = %s AND l.lot_code = %s GROUP BY l.id, l.lot_code", (product["id"], req.lot_code))
                lot = cur.fetchone()
                if not lot:
                    cur.execute("SELECT l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as balance FROM lots l LEFT JOIN transaction_lines tl ON tl.lot_id = l.id WHERE l.product_id = %s GROUP BY l.id, l.lot_code HAVING COALESCE(SUM(tl.quantity_lb), 0) != 0 ORDER BY l.lot_code", (product["id"],))
                    existing_lots = cur.fetchall()
                    if existing_lots:
                        raise HTTPException(status_code=400, detail=f"Lot '{req.lot_code}' not found for {product['name']}. Existing lots:\n" + "\n".join([f"• {l['lot_code']} ({l['balance']:,.0f} lb)" for l in existing_lots]))
                    else:
                        raise HTTPException(status_code=400, detail=f"Lot '{req.lot_code}' not found for {product['name']}. No lots exist for this product.")
                current_balance = float(lot["current_balance"])
                new_balance = current_balance + req.quantity_lb
                if new_balance < 0:
                    raise HTTPException(status_code=400, detail=f"Cannot adjust. Lot {req.lot_code} only has {current_balance:,.0f} lb. Adjustment of {req.quantity_lb:,.0f} lb would result in {new_balance:,.0f} lb (negative not allowed).")
        adj_display = f"+{req.quantity_lb:,.0f}" if req.quantity_lb > 0 else f"{req.quantity_lb:,.0f}"
        adj_type = "ADD" if req.quantity_lb > 0 else "REMOVE"
        preview_message = f"📋 ADJUST PREVIEW\n────────────────────────────────\nProduct:      {product['name']} ({product['odoo_code']})\nLot:          {req.lot_code}\nAdjustment:   {adj_display} lb ({adj_type})\n────────────────────────────────\nCurrent:      {current_balance:,.0f} lb\nAfter:        {new_balance:,.0f} lb\n────────────────────────────────\nReason:       {req.reason}\n────────────────────────────────\n✓ Say \"confirm\" to proceed"
        return AdjustPreviewResponse(product_id=product["id"], product_name=product["name"], odoo_code=product["odoo_code"], lot_code=lot["lot_code"], lot_id=lot["lot_id"], adjustment_lb=req.quantity_lb, current_balance_lb=current_balance, new_balance_lb=new_balance, reason=req.reason, preview_message=preview_message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/adjust/commit", response_model=AdjustCommitResponse)
def adjust_commit(req: AdjustRequest, _: bool = Depends(verify_api_key)):
    if not req.reason or len(req.reason.strip()) < 3:
        raise HTTPException(status_code=400, detail="Reason is required (minimum 3 characters)")
    if req.quantity_lb == 0:
        raise HTTPException(status_code=400, detail="Adjustment quantity cannot be zero")
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        conn.autocommit = False
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, name, odoo_code FROM products WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) = LOWER(%s) LIMIT 1", (f"%{req.product_name}%", req.product_name))
        product = cur.fetchone()
        if not product:
            raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")
        cur.execute("SELECT l.id as lot_id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as current_balance FROM lots l LEFT JOIN transaction_lines tl ON tl.lot_id = l.id WHERE l.product_id = %s AND l.lot_code = %s GROUP BY l.id, l.lot_code", (product["id"], req.lot_code))
        lot = cur.fetchone()
        if not lot:
            raise HTTPException(status_code=400, detail=f"Lot '{req.lot_code}' not found for {product['name']}")
        current_balance = float(lot["current_balance"])
        new_balance = current_balance + req.quantity_lb
        if new_balance < 0:
            raise HTTPException(status_code=400, detail=f"Cannot adjust. Lot {req.lot_code} only has {current_balance:,.0f} lb. Adjustment of {req.quantity_lb:,.0f} lb would result in negative balance.")
        cur.execute("INSERT INTO transactions (type, adjust_reason) VALUES ('adjust', %s) RETURNING id", (req.reason,))
        transaction_id = cur.fetchone()["id"]
        cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (transaction_id, product["id"], lot["lot_id"], req.quantity_lb))
        conn.commit()
        cur.close()
        adj_type = "Added" if req.quantity_lb > 0 else "Removed"
        return AdjustCommitResponse(success=True, transaction_id=transaction_id, lot_code=lot["lot_code"], adjustment_lb=req.quantity_lb, new_balance_lb=new_balance, reason=req.reason, message=f"✅ {adj_type} {abs(req.quantity_lb):,.0f} lb {product['name']} (lot {lot['lot_code']}). New balance: {new_balance:,.0f} lb")
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn: conn.close()


# --- MAKE ENDPOINTS ---

def _norm(s): return re.sub(r"\s+", " ", s.strip())

def expand_bom(cur, ingredients, parent_batch_size):
    expanded, has_batches = [], False
    for ing in ingredients:
        cur.execute("SELECT bf.quantity_lb, p.name, p.odoo_code, parent.default_batch_lb FROM batch_formulas bf JOIN products p ON p.id = bf.ingredient_product_id JOIN products parent ON parent.id = bf.product_id WHERE bf.product_id = (SELECT id FROM products WHERE odoo_code = %s OR name = %s LIMIT 1) ORDER BY bf.quantity_lb DESC", (ing["ingredient_code"], ing["ingredient"]))
        sub = cur.fetchall()
        if sub:
            has_batches = True
            scale = float(ing["quantity_lb"]) / float(sub[0]["default_batch_lb"]) if sub[0]["default_batch_lb"] else 1
            expanded.append({"parent": ing["ingredient"], "parent_odoo_code": ing["ingredient_code"], "parent_qty_lb": float(ing["quantity_lb"]), "raw_ingredients": [{"name": s["name"], "odoo_code": s["odoo_code"], "quantity_lb": round(float(s["quantity_lb"]) * scale, 2)} for s in sub if float(s["quantity_lb"]) > 0]})
        else:
            expanded.append({"name": ing["ingredient"], "odoo_code": ing["ingredient_code"], "quantity_lb": float(ing["quantity_lb"]), "is_raw": True})
    return {"expanded_ingredients": expanded} if has_batches else {}

def get_or_create_lot(cur, product_id, lot_code):
    cur.execute("SELECT id FROM lots WHERE product_id=%s AND lot_code=%s", (product_id, lot_code))
    row = cur.fetchone()
    if row: return row["id"]
    cur.execute("INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id", (product_id, lot_code))
    return cur.fetchone()["id"]

def find_product(cur, token):
    t = _norm(token)
    if t.isdigit():
        cur.execute("SELECT id, name, odoo_code, default_batch_lb, type FROM products WHERE odoo_code=%s LIMIT 1", (t,))
        row = cur.fetchone()
        if row: return (row, [row])
    cur.execute("SELECT id, name, odoo_code, default_batch_lb, type FROM products WHERE LOWER(name)=LOWER(%s) LIMIT 1", (t,))
    row = cur.fetchone()
    if row: return (row, [row])
    cur.execute("SELECT id, name, odoo_code, default_batch_lb, type FROM products WHERE name ILIKE %s ORDER BY LENGTH(name) ASC LIMIT 5", (f"%{t}%",))
    matches = cur.fetchall()
    if len(matches) == 1: return (matches[0], matches)
    elif len(matches) > 1: return (None, matches)
    return (None, [])


@app.post("/make/preview", response_model=MakePreviewResponse)
def make_preview(req: MakeRequest, _: bool = Depends(verify_api_key)):
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                batch, matches = find_product(cur, req.product_name)
                if not batch and len(matches) > 1:
                    raise HTTPException(status_code=400, detail=f"Multiple products match '{req.product_name}'. Please be more specific:\n" + "\n".join([f"• {m['name']} ({m['odoo_code']})" for m in matches]))
                if not batch:
                    raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")
                if batch.get("default_batch_lb") is None:
                    raise HTTPException(status_code=400, detail=f"No batch size defined for '{batch['name']}'")
                batch_size_lb = float(batch["default_batch_lb"])
                total_yield_lb = req.batches * batch_size_lb
                cur.execute("""
                    SELECT bf.ingredient_product_id, bf.quantity_lb, 
                           p.name as ingredient_name, p.odoo_code as ingredient_odoo_code
                    FROM batch_formulas bf
                    JOIN products p ON p.id = bf.ingredient_product_id
                    WHERE bf.product_id = %s
                    ORDER BY bf.quantity_lb DESC
                """, (batch["id"],))
                bom_lines = cur.fetchall()
                if not bom_lines:
                    raise HTTPException(status_code=400, detail=f"No BOM/recipe found for '{batch['name']}'")
                ingredients = []
                all_sufficient = True
                override_warnings = []
                for line in bom_lines:
                    required_lb = float(line["quantity_lb"]) * req.batches
                    product_id = line["ingredient_product_id"]
                    odoo_code = line["ingredient_odoo_code"]
                    overrides = None
                    if req.ingredient_lot_overrides and odoo_code in req.ingredient_lot_overrides:
                        overrides = req.ingredient_lot_overrides[odoo_code]
                    allocations, sufficient, error_msg = allocate_with_overrides(cur, product_id, odoo_code, required_lb, overrides)
                    if error_msg:
                        override_warnings.append(error_msg)
                    if not sufficient:
                        all_sufficient = False
                    ingredients.append(IngredientAllocation(product_name=line["ingredient_name"], product_id=product_id, odoo_code=odoo_code, required_lb=round(required_lb, 2), allocated_lots=[IngredientLotAllocation(lot_code=a["lot_code"], lot_id=a["lot_id"], available_lb=a["available_lb"], use_lb=a["use_lb"]) for a in allocations], sufficient=sufficient))
                ing_lines = []
                for ing in ingredients:
                    status = "✓" if ing.sufficient else "⚠️ INSUFFICIENT"
                    lot_details = ", ".join([f"{a.lot_code}: {a.use_lb} lb" for a in ing.allocated_lots])
                    if not lot_details:
                        lot_details = "NO INVENTORY"
                    ing_lines.append(f"  • {ing.product_name}: {ing.required_lb} lb {status}\n    Lots: {lot_details}")
                status_line = "✓ All ingredients available" if all_sufficient else "⚠️ INSUFFICIENT INVENTORY - cannot proceed"
                warning_line = ""
                if override_warnings:
                    warning_line = "\n⚠️ Override warnings:\n  " + "\n  ".join(override_warnings) + "\n"
                preview_message = f"""�icing PRODUCTION PREVIEW
────────────────────────────────
Product:      {batch['name']} ({batch['odoo_code']})
Batches:      {req.batches} × {batch_size_lb:,.0f} lb = {total_yield_lb:,.0f} lb
Output Lot:   {req.lot_code}
────────────────────────────────
INGREDIENTS TO CONSUME:
{chr(10).join(ing_lines)}
────────────────────────────────{warning_line}
{status_line}
────────────────────────────────
✓ Say "confirm" to proceed"""
                return MakePreviewResponse(batch_product_id=batch["id"], batch_product_name=batch["name"], batch_odoo_code=batch.get("odoo_code", ""), batches=req.batches, batch_size_lb=batch_size_lb, total_yield_lb=total_yield_lb, output_lot_code=req.lot_code, ingredients=ingredients, all_sufficient=all_sufficient, preview_message=preview_message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/make/commit", response_model=MakeCommitResponse)
def make_commit(req: MakeRequest, _: bool = Depends(verify_api_key)):
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        conn.autocommit = False
        cur = conn.cursor(cursor_factory=RealDictCursor)
        batch, matches = find_product(cur, req.product_name)
        if not batch and len(matches) > 1:
            raise HTTPException(status_code=400, detail=f"Multiple products match '{req.product_name}'")
        if not batch:
            raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")
        if batch.get("default_batch_lb") is None:
            raise HTTPException(status_code=400, detail=f"No batch size defined for '{batch['name']}'")
        batch_size_lb = float(batch["default_batch_lb"])
        total_yield_lb = req.batches * batch_size_lb
        cur.execute("""
            SELECT bf.ingredient_product_id, bf.quantity_lb, 
                   p.name as ingredient_name, p.odoo_code as ingredient_odoo_code
            FROM batch_formulas bf
            JOIN products p ON p.id = bf.ingredient_product_id
            WHERE bf.product_id = %s
            ORDER BY bf.quantity_lb DESC
        """, (batch["id"],))
        bom_lines = cur.fetchall()
        if not bom_lines:
            raise HTTPException(status_code=400, detail=f"No BOM/recipe found for '{batch['name']}'")
        for line in bom_lines:
            required_lb = float(line["quantity_lb"]) * req.batches
            odoo_code = line["ingredient_odoo_code"]
            overrides = None
            if req.ingredient_lot_overrides and odoo_code in req.ingredient_lot_overrides:
                overrides = req.ingredient_lot_overrides[odoo_code]
            allocations, sufficient, _ = allocate_with_overrides(cur, line["ingredient_product_id"], odoo_code, required_lb, overrides)
            if not sufficient:
                total_allocated = sum(a["use_lb"] for a in allocations)
                raise HTTPException(status_code=400, detail=f"Insufficient {line['ingredient_name']}: need {required_lb:,.0f} lb, can allocate {total_allocated:,.0f} lb")
        cur.execute("INSERT INTO transactions (type, notes) VALUES ('make', %s) RETURNING id", (f"Make {req.batches} batch(es) {batch['name']} lot {req.lot_code}",))
        tx_id = cur.fetchone()["id"]
        output_lot_id = get_or_create_lot(cur, batch["id"], req.lot_code)
        consumed = []
        for line in bom_lines:
            required_lb = float(line["quantity_lb"]) * req.batches
            product_id = line["ingredient_product_id"]
            odoo_code = line["ingredient_odoo_code"]
            overrides = None
            if req.ingredient_lot_overrides and odoo_code in req.ingredient_lot_overrides:
                overrides = req.ingredient_lot_overrides[odoo_code]
            allocations, _, _ = allocate_with_overrides(cur, product_id, odoo_code, required_lb, overrides)
            lots_used = []
            for alloc in allocations:
                use_lb = alloc["use_lb"]
                lot_id = alloc["lot_id"]
                lot_code = alloc["lot_code"]
                cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (tx_id, product_id, lot_id, -use_lb))
                cur.execute("INSERT INTO ingredient_lot_consumption (transaction_id, ingredient_product_id, ingredient_lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (tx_id, product_id, lot_id, use_lb))
                lots_used.append({"lot": lot_code, "qty_lb": round(use_lb, 2)})
            consumed.append({"ingredient": line["ingredient_name"], "odoo_code": odoo_code, "total_lb": round(required_lb, 2), "lots": lots_used})
        cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (tx_id, batch["id"], output_lot_id, total_yield_lb))
        conn.commit()
        cur.close()
        consumed_summary = "\n".join([f"  • {c['ingredient']}: {c['total_lb']} lb from {', '.join([l['lot'] for l in c['lots']])}" for c in consumed])
        return MakeCommitResponse(success=True, transaction_id=tx_id, batch_product_name=batch["name"], batch_odoo_code=batch.get("odoo_code", ""), batches=req.batches, produced_lb=total_yield_lb, output_lot_code=req.lot_code, consumed=consumed, message=f"✅ Produced {total_yield_lb:,.0f} lb {batch['name']} (lot {req.lot_code})\n\nIngredients consumed:\n{consumed_summary}")
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn: conn.close()


# --- REPACK ENDPOINTS ---

@app.post("/repack/preview", response_model=RepackPreviewResponse)
def repack_preview(req: RepackRequest, _: bool = Depends(verify_api_key)):
    if req.source_quantity_lb <= 0:
        raise HTTPException(status_code=400, detail="Source quantity must be positive")
    if req.target_quantity_lb <= 0:
        raise HTTPException(status_code=400, detail="Target quantity must be positive")
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                source, source_matches = find_product(cur, req.source_product)
                if not source and len(source_matches) > 1:
                    raise HTTPException(status_code=400, detail=f"Multiple source products match '{req.source_product}':\n" + "\n".join([f"• {m['name']} ({m['odoo_code']})" for m in source_matches]))
                if not source:
                    raise HTTPException(status_code=404, detail=f"Source product not found: {req.source_product}")
                cur.execute("""
                    SELECT l.id as lot_id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available_lb
                    FROM lots l
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.product_id = %s AND l.lot_code = %s
                    GROUP BY l.id, l.lot_code
                """, (source["id"], req.source_lot))
                source_lot = cur.fetchone()
                if not source_lot:
                    available_lots = get_available_lots_fifo(cur, source["id"])
                    if available_lots:
                        raise HTTPException(status_code=400, detail=f"Lot '{req.source_lot}' not found for {source['name']}. Available lots:\n" + "\n".join([f"• {l['lot_code']} ({l['available_lb']:,.0f} lb)" for l in available_lots]))
                    else:
                        raise HTTPException(status_code=400, detail=f"No lots found for {source['name']}")
                source_available = float(source_lot["available_lb"])
                if source_available < req.source_quantity_lb:
                    raise HTTPException(status_code=400, detail=f"Insufficient inventory in lot {req.source_lot}. Available: {source_available:,.0f} lb, Requested: {req.source_quantity_lb:,.0f} lb")
                target, target_matches = find_product(cur, req.target_product)
                if not target and len(target_matches) > 1:
                    raise HTTPException(status_code=400, detail=f"Multiple target products match '{req.target_product}':\n" + "\n".join([f"• {m['name']} ({m['odoo_code']})" for m in target_matches]))
                if not target:
                    raise HTTPException(status_code=404, detail=f"Target product not found: {req.target_product}")
                yield_pct = (req.target_quantity_lb / req.source_quantity_lb) * 100
                yield_note = ""
                if yield_pct < 95:
                    yield_note = f"\n⚠️ Low yield ({yield_pct:.1f}%) - {req.source_quantity_lb - req.target_quantity_lb:.1f} lb loss"
                elif yield_pct > 100:
                    yield_note = f"\n⚠️ Yield over 100% ({yield_pct:.1f}%) - verify quantities"
                notes_line = f"\nNotes:        {req.notes}" if req.notes else ""
                preview_message = f"""🔄 REPACK PREVIEW
────────────────────────────────
FROM (consume):
  Product:    {source['name']} ({source['odoo_code']})
  Lot:        {req.source_lot}
  Available:  {source_available:,.0f} lb
  Consume:    {req.source_quantity_lb:,.0f} lb

TO (produce):
  Product:    {target['name']} ({target['odoo_code']})
  Lot:        {req.target_lot_code}
  Produce:    {req.target_quantity_lb:,.0f} lb

Yield:        {yield_pct:.1f}%{yield_note}{notes_line}
────────────────────────────────
✓ Say "confirm" to proceed"""
                return RepackPreviewResponse(source_product_id=source["id"], source_product_name=source["name"], source_odoo_code=source.get("odoo_code", ""), source_lot_code=source_lot["lot_code"], source_lot_id=source_lot["lot_id"], source_available_lb=source_available, source_consume_lb=req.source_quantity_lb, target_product_id=target["id"], target_product_name=target["name"], target_odoo_code=target.get("odoo_code", ""), target_lot_code=req.target_lot_code, target_produce_lb=req.target_quantity_lb, yield_pct=round(yield_pct, 2), notes=req.notes, preview_message=preview_message)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/repack/commit", response_model=RepackCommitResponse)
def repack_commit(req: RepackRequest, _: bool = Depends(verify_api_key)):
    if req.source_quantity_lb <= 0:
        raise HTTPException(status_code=400, detail="Source quantity must be positive")
    if req.target_quantity_lb <= 0:
        raise HTTPException(status_code=400, detail="Target quantity must be positive")
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        conn.autocommit = False
        cur = conn.cursor(cursor_factory=RealDictCursor)
        source, _ = find_product(cur, req.source_product)
        if not source:
            raise HTTPException(status_code=404, detail=f"Source product not found: {req.source_product}")
        cur.execute("""
            SELECT l.id as lot_id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available_lb
            FROM lots l
            LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
            WHERE l.product_id = %s AND l.lot_code = %s
            GROUP BY l.id, l.lot_code
        """, (source["id"], req.source_lot))
        source_lot = cur.fetchone()
        if not source_lot:
            raise HTTPException(status_code=400, detail=f"Lot '{req.source_lot}' not found for {source['name']}")
        source_available = float(source_lot["available_lb"])
        if source_available < req.source_quantity_lb:
            raise HTTPException(status_code=400, detail=f"Insufficient inventory. Available: {source_available:,.0f} lb, Requested: {req.source_quantity_lb:,.0f} lb")
        target, _ = find_product(cur, req.target_product)
        if not target:
            raise HTTPException(status_code=404, detail=f"Target product not found: {req.target_product}")
        notes = req.notes or f"Repack {source['name']} to {target['name']}"
        cur.execute("INSERT INTO transactions (type, notes) VALUES ('repack', %s) RETURNING id", (notes,))
        tx_id = cur.fetchone()["id"]
        cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (tx_id, source["id"], source_lot["lot_id"], -req.source_quantity_lb))
        target_lot_id = get_or_create_lot(cur, target["id"], req.target_lot_code)
        cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (tx_id, target["id"], target_lot_id, req.target_quantity_lb))
        cur.execute("INSERT INTO ingredient_lot_consumption (transaction_id, ingredient_product_id, ingredient_lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (tx_id, source["id"], source_lot["lot_id"], req.source_quantity_lb))
        conn.commit()
        cur.close()
        return RepackCommitResponse(success=True, transaction_id=tx_id, source_product=source["name"], source_lot=req.source_lot, consumed_lb=req.source_quantity_lb, target_product=target["name"], target_lot=req.target_lot_code, produced_lb=req.target_quantity_lb, message=f"✅ Repacked {req.source_quantity_lb:,.0f} lb {source['name']} (lot {req.source_lot}) → {req.target_quantity_lb:,.0f} lb {target['name']} (lot {req.target_lot_code})")
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn: conn.close()


# --- TRACEABILITY ENDPOINTS ---

@app.get("/trace/batch/{lot_code}")
def trace_batch(lot_code: str, _: bool = Depends(verify_api_key)):
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT t.id as transaction_id, t.timestamp, t.notes,
                           p.name as product_name, p.odoo_code,
                           tl.quantity_lb as produced_lb, l.lot_code
                    FROM transactions t
                    JOIN transaction_lines tl ON tl.transaction_id = t.id
                    JOIN lots l ON l.id = tl.lot_id
                    JOIN products p ON p.id = tl.product_id
                    WHERE t.type IN ('make', 'repack') AND l.lot_code = %s AND tl.quantity_lb > 0
                    ORDER BY t.timestamp DESC LIMIT 1
                """, (lot_code,))
                batch_info = cur.fetchone()
                if not batch_info:
                    return JSONResponse(status_code=404, content={"error": f"No production record found for lot {lot_code}"})
                cur.execute("""
                    SELECT p.name as ingredient_name, p.odoo_code as ingredient_code,
                           l.lot_code as ingredient_lot, ilc.quantity_lb
                    FROM ingredient_lot_consumption ilc
                    JOIN products p ON p.id = ilc.ingredient_product_id
                    JOIN lots l ON l.id = ilc.ingredient_lot_id
                    WHERE ilc.transaction_id = %s
                    ORDER BY ilc.quantity_lb DESC
                """, (batch_info["transaction_id"],))
                ingredients = cur.fetchall()
                date_str, time_str = format_timestamp(batch_info["timestamp"])
                return {
                    "lot_code": lot_code,
                    "product": batch_info["product_name"],
                    "odoo_code": batch_info["odoo_code"],
                    "produced_lb": float(batch_info["produced_lb"]),
                    "produced_at": f"{date_str} {time_str}",
                    "transaction_id": batch_info["transaction_id"],
                    "ingredients_consumed": [{"ingredient": ing["ingredient_name"], "odoo_code": ing["ingredient_code"], "lot": ing["ingredient_lot"], "quantity_lb": float(ing["quantity_lb"])} for ing in ingredients]
                }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/trace/ingredient/{lot_code}")
def trace_ingredient(lot_code: str, used_only: bool = Query(default=False), _: bool = Depends(verify_api_key)):
    """
    Trace which batches used a given ingredient lot.
    Returns ALL products that have this lot code (lot codes aren't globally unique).
    """
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT l.id, l.lot_code, l.product_id, p.name as product_name, p.odoo_code
                    FROM lots l
                    JOIN products p ON p.id = l.product_id
                    WHERE l.lot_code = %s
                """, (lot_code,))
                lots = cur.fetchall()
                
                if not lots:
                    return JSONResponse(status_code=404, content={"error": f"No lots found with code '{lot_code}'"})
                
                products_traced = []
                total_batch_count = 0
                
                for lot in lots:
                    lot_id = lot["id"]
                    
                    cur.execute("""
                        SELECT ilc.quantity_lb, ilc.transaction_id, t.timestamp,
                               output_lot.lot_code as batch_lot,
                               output_product.name as batch_name,
                               output_product.odoo_code as batch_odoo_code
                        FROM ingredient_lot_consumption ilc
                        JOIN transactions t ON t.id = ilc.transaction_id
                        JOIN transaction_lines output_tl ON output_tl.transaction_id = t.id AND output_tl.quantity_lb > 0
                        JOIN lots output_lot ON output_lot.id = output_tl.lot_id
                        JOIN products output_product ON output_product.id = output_tl.product_id
                        WHERE ilc.ingredient_lot_id = %s
                        ORDER BY t.timestamp DESC
                    """, (lot_id,))
                    consumption = cur.fetchall()
                    
                    batches_used = []
                    for c in consumption:
                        date_str, time_str = format_timestamp(c["timestamp"])
                        batches_used.append({
                            "batch_lot": c["batch_lot"],
                            "batch_product": c["batch_name"],
                            "batch_odoo_code": c["batch_odoo_code"],
                            "quantity_lb": float(c["quantity_lb"]),
                            "transaction_id": c["transaction_id"],
                            "timestamp": f"{date_str} {time_str}"
                        })
                    
                    products_traced.append({
                        "ingredient": lot["product_name"],
                        "odoo_code": lot["odoo_code"],
                        "lot_id": lot_id,
                        "used_in_batches": batches_used,
                        "batch_count": len(batches_used)
                    })
                    total_batch_count += len(batches_used)

                if used_only:
                    products_traced = [p for p in products_traced if p["batch_count"] > 0]
                
                return {                
                    "lot_code": lot_code,
                    "products_with_lot": len(products_traced),
                    "products": products_traced,
                    "total_batch_count": total_batch_count
                }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# BOM ENDPOINTS
# ═══════════════════════════════════════════════════════════════

# --- Pydantic Models for BOM ---

class BOMProductResponse(BaseModel):
    id: int
    internal_ref: str
    name: str
    product_type: str
    brand: Optional[str] = None
    odoo_code: Optional[str] = None
    unit: Optional[str] = None
    notes: Optional[str] = None

class BatchFormulaItem(BaseModel):
    ingredient_ref: str
    ingredient_name: str
    quantity: float
    unit: str
    percentage: Optional[float] = None

class BatchFormulaResponse(BaseModel):
    batch_ref: str
    batch_name: str
    batch_weight_lb: Optional[float] = None
    yields_batches: Optional[int] = None
    ingredients: List[BatchFormulaItem]
    allergens: List[str] = []

class FinishedBOMResponse(BaseModel):
    finished_ref: str
    finished_name: str
    brand: Optional[str] = None
    batch_ref: Optional[str] = None
    batch_name: Optional[str] = None
    batch_qty: Optional[float] = None
    packaging_items: List[dict] = []
    allergens: List[str] = []

class ProductionRequirementRequest(BaseModel):
    finished_ref: str
    quantity_cases: int

class IngredientRequirement(BaseModel):
    ingredient_ref: str
    ingredient_name: str
    quantity_needed_lb: float
    unit: str

class ProductionRequirementResponse(BaseModel):
    finished_ref: str
    finished_name: str
    cases_requested: int
    batches_needed: float
    ingredients: List[IngredientRequirement]

class UpdateBatchRequest(BaseModel):
    batch_weight_lb: Optional[float] = None
    yields_batches: Optional[int] = None
    notes: Optional[str] = None


# --- BOM Endpoints ---

@app.get("/bom/products")
def get_bom_products(
    product_type: Optional[str] = Query(None, description="Filter by type: ingredient, batch, packaging, finished"),
    brand: Optional[str] = Query(None, description="Filter by brand: CNS, Setton, SS, BS, CQ, UNIPRO"),
    search: Optional[str] = Query(None, description="Search by name or internal_ref"),
    limit: int = Query(100, le=500),
    _: bool = Depends(verify_api_key)
):
    """List all products with optional filters"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        query = "SELECT id, internal_ref, name, product_type, brand, odoo_code, unit, notes FROM products WHERE 1=1"
        params = []
        
        if product_type:
            query += " AND product_type = %s"
            params.append(product_type)
        
        if brand:
            query += " AND brand = %s"
            params.append(brand)
        
        if search:
            query += " AND (name ILIKE %s OR internal_ref ILIKE %s)"
            params.extend([f"%{search}%", f"%{search}%"])
        
        query += " ORDER BY internal_ref LIMIT %s"
        params.append(limit)
        
        cur.execute(query, params)
        products = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return {"count": len(products), "products": products}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/bom/batches")
def get_all_batches(
    brand: Optional[str] = Query(None),
    _: bool = Depends(verify_api_key)
):
    """List all batches with their summary info"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        query = "SELECT * FROM v_batch_summary WHERE 1=1"
        params = []
        
        if brand:
            query += " AND brand = %s"
            params.append(brand)
        
        query += " ORDER BY batch_ref"
        
        cur.execute(query, params)
        batches = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return {"count": len(batches), "batches": batches}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/bom/batches/{batch_ref}/formula")
def get_batch_formula(
    batch_ref: str,
    _: bool = Depends(verify_api_key)
):
    """Get the ingredient formula for a specific batch"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT id, internal_ref, name, batch_weight_lb, yields_batches 
            FROM products 
            WHERE internal_ref = %s AND product_type = 'batch'
        """, (batch_ref,))
        batch = cur.fetchone()
        
        if not batch:
            cur.close()
            conn.close()
            return JSONResponse(status_code=404, content={"error": f"Batch {batch_ref} not found"})
        
        cur.execute("""
            SELECT ingredient_ref, ingredient_name, quantity, unit, percentage
            FROM v_batch_ingredients
            WHERE batch_ref = %s
            ORDER BY percentage DESC NULLS LAST
        """, (batch_ref,))
        ingredients = cur.fetchall()
        
        cur.execute("""
            SELECT a.name 
            FROM product_allergens pa
            JOIN allergens a ON pa.allergen_id = a.id
            JOIN products p ON pa.product_id = p.id
            WHERE p.internal_ref = %s
        """, (batch_ref,))
        allergens = [row["name"] for row in cur.fetchall()]
        
        cur.close()
        conn.close()
        
        return {
            "batch_ref": batch["internal_ref"],
            "batch_name": batch["name"],
            "batch_weight_lb": batch["batch_weight_lb"],
            "yields_batches": batch["yields_batches"],
            "ingredient_count": len(ingredients),
            "ingredients": ingredients,
            "allergens": allergens
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/bom/finished/{finished_ref}/bom")
def get_finished_product_bom(
    finished_ref: str,
    _: bool = Depends(verify_api_key)
):
    """Get the full BOM for a finished product (batch + packaging)"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT id, internal_ref, name, brand, case_size, cases_per_pallet
            FROM products 
            WHERE internal_ref = %s AND product_type = 'finished'
        """, (finished_ref,))
        product = cur.fetchone()
        
        if not product:
            cur.close()
            conn.close()
            return JSONResponse(status_code=404, content={"error": f"Finished product {finished_ref} not found"})
        
        cur.execute("""
            SELECT * FROM v_finished_product_bom WHERE finished_ref = %s
        """, (finished_ref,))
        bom = cur.fetchone()
        
        cur.execute("""
            SELECT p.internal_ref, p.name, pb.quantity, p.unit
            FROM product_bom pb
            JOIN products p ON pb.component_id = p.id
            JOIN products fp ON pb.finished_product_id = fp.id
            WHERE fp.internal_ref = %s AND p.product_type = 'packaging'
        """, (finished_ref,))
        packaging = cur.fetchall()
        
        allergens = []
        if bom and bom.get("batch_ref"):
            cur.execute("""
                SELECT a.name 
                FROM product_allergens pa
                JOIN allergens a ON pa.allergen_id = a.id
                JOIN products p ON pa.product_id = p.id
                WHERE p.internal_ref = %s
            """, (bom["batch_ref"],))
            allergens = [row["name"] for row in cur.fetchall()]
        
        cur.close()
        conn.close()
        
        return {
            "finished_ref": product["internal_ref"],
            "finished_name": product["name"],
            "brand": product["brand"],
            "case_size": product["case_size"],
            "cases_per_pallet": product["cases_per_pallet"],
            "batch_ref": bom["batch_ref"] if bom else None,
            "batch_name": bom["batch_name"] if bom else None,
            "batch_qty_per_case": bom["batch_qty"] if bom else None,
            "packaging_items": packaging,
            "allergens": allergens
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/bom/production/requirements")
def calculate_production_requirements(
    req: ProductionRequirementRequest,
    _: bool = Depends(verify_api_key)
):
    """Calculate ingredient requirements for a production run"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT fp.id, fp.name as finished_name, fp.internal_ref,
                   b.internal_ref as batch_ref, b.name as batch_name,
                   b.batch_weight_lb, b.yields_batches,
                   pb.quantity as batch_qty_per_case
            FROM products fp
            JOIN product_bom pb ON fp.id = pb.finished_product_id
            JOIN products b ON pb.component_id = b.id AND b.product_type = 'batch'
            WHERE fp.internal_ref = %s AND fp.product_type = 'finished'
        """, (req.finished_ref,))
        product = cur.fetchone()
        
        if not product:
            cur.close()
            conn.close()
            return JSONResponse(status_code=404, content={"error": f"Finished product {req.finished_ref} not found or has no batch"})
        
        batch_qty_per_case = product["batch_qty_per_case"] or 1
        yields_batches = product["yields_batches"] or 1
        batch_weight = product["batch_weight_lb"] or 0
        
        total_batch_qty_needed = req.quantity_cases * batch_qty_per_case
        batches_needed = total_batch_qty_needed / yields_batches if yields_batches else total_batch_qty_needed
        
        cur.execute("""
            SELECT ingredient_ref, ingredient_name, quantity, unit
            FROM v_batch_ingredients
            WHERE batch_ref = %s
        """, (product["batch_ref"],))
        ingredients = cur.fetchall()
        
        scaled_ingredients = []
        for ing in ingredients:
            scaled_qty = (ing["quantity"] or 0) * batches_needed
            scaled_ingredients.append({
                "ingredient_ref": ing["ingredient_ref"],
                "ingredient_name": ing["ingredient_name"],
                "quantity_needed": round(scaled_qty, 2),
                "unit": ing["unit"]
            })
        
        cur.close()
        conn.close()
        
        return {
            "finished_ref": req.finished_ref,
            "finished_name": product["finished_name"],
            "cases_requested": req.quantity_cases,
            "batch_ref": product["batch_ref"],
            "batch_name": product["batch_name"],
            "batches_needed": round(batches_needed, 2),
            "total_batch_weight_lb": round(batches_needed * batch_weight, 2) if batch_weight else None,
            "ingredients": scaled_ingredients
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.put("/bom/batches/{batch_ref}")
def update_batch(
    batch_ref: str,
    req: UpdateBatchRequest,
    _: bool = Depends(verify_api_key)
):
    """Update batch weight, yields, or notes"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT id FROM products WHERE internal_ref = %s AND product_type = 'batch'
        """, (batch_ref,))
        batch = cur.fetchone()
        
        if not batch:
            cur.close()
            conn.close()
            return JSONResponse(status_code=404, content={"error": f"Batch {batch_ref} not found"})
        
        updates = []
        params = []
        
        if req.batch_weight_lb is not None:
            updates.append("batch_weight_lb = %s")
            params.append(req.batch_weight_lb)
        
        if req.yields_batches is not None:
            updates.append("yields_batches = %s")
            params.append(req.yields_batches)
        
        if req.notes is not None:
            updates.append("notes = %s")
            params.append(req.notes)
        
        if not updates:
            return JSONResponse(status_code=400, content={"error": "No fields to update"})
        
        params.append(batch_ref)
        query = f"UPDATE products SET {', '.join(updates)} WHERE internal_ref = %s RETURNING *"
        
        cur.execute(query, params)
        updated = cur.fetchone()
        conn.commit()
        
        cur.close()
        conn.close()
        
        return {"message": f"Batch {batch_ref} updated", "batch": updated}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/bom/allergens")
def get_allergens_list(_: bool = Depends(verify_api_key)):
    """Get all allergens and which batches contain them"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT a.id, a.name, a.code,
                   COALESCE(json_agg(
                       json_build_object('batch_ref', p.internal_ref, 'batch_name', p.name)
                   ) FILTER (WHERE p.id IS NOT NULL), '[]') as batches
            FROM allergens a
            LEFT JOIN product_allergens pa ON a.id = pa.allergen_id
            LEFT JOIN products p ON pa.product_id = p.id
            GROUP BY a.id, a.name, a.code
            ORDER BY a.name
        """)
        allergens = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return {"allergens": allergens}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/bom/search")
def search_bom(
    q: str = Query(..., description="Search term"),
    _: bool = Depends(verify_api_key)
):
    """Search across all products, batches, and formulas"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        search_term = f"%{q}%"
        
        cur.execute("""
            SELECT internal_ref, name, product_type, brand
            FROM products
            WHERE name ILIKE %s OR internal_ref ILIKE %s OR odoo_code ILIKE %s
            ORDER BY product_type, internal_ref
            LIMIT 25
        """, (search_term, search_term, search_term))
        products = cur.fetchall()
        
        cur.close()
        conn.close()
        
        return {
            "query": q,
            "result_count": len(products),
            "results": products
        }
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
