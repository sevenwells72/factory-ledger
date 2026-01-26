from fastapi import FastAPI, HTTPException, Header, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from datetime import datetime, date
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import re

app = FastAPI(title="Factory Ledger System")

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
API_KEY = (os.getenv("API_KEY") or "").strip()


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
    lot_code: str
    lot_id: int
    available_lb: float
    lot_selection: str
    preview_message: str


class ShipCommitResponse(BaseModel):
    success: bool
    transaction_id: int
    lot_code: str
    quantity_lb: float
    slip_text: str
    slip_html: str
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


# --- Receipt/Slip Generators ---

def generate_receipt_text(product_name, odoo_code, cases, case_size_lb, total_lb, shipper_name, lot_code, bol_reference, timestamp):
    return f"""
╔══════════════════════════════════════════════════╗
║         CNS CONFECTIONERY PRODUCTS               ║
║              RECEIVING RECEIPT                   ║
╠══════════════════════════════════════════════════╣
║  Date:        {timestamp.strftime('%B %d, %Y'):<32}║
║  Time:        {timestamp.strftime('%I:%M %p'):<32}║
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
            <div class="row"><span class="label">Date:</span><span>{timestamp.strftime('%B %d, %Y')}</span></div>
            <div class="row"><span class="label">Time:</span><span>{timestamp.strftime('%I:%M %p')}</span></div>
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
    return f"""
╔══════════════════════════════════════════════════╗
║         CNS CONFECTIONERY PRODUCTS               ║
║                PACKING SLIP                      ║
╠══════════════════════════════════════════════════╣
║  Date:        {timestamp.strftime('%B %d, %Y'):<32}║
║  Time:        {timestamp.strftime('%I:%M %p'):<32}║
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
            <div class="row"><span class="label">Date:</span><span>{timestamp.strftime('%B %d, %Y')}</span></div>
            <div class="row"><span class="label">Time:</span><span>{timestamp.strftime('%I:%M %p')}</span></div>
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


# --- API Endpoints ---

@app.get("/")
def root():
    return {
        "name": "Factory Ledger System",
        "version": "0.8.0",
        "status": "online",
        "endpoints": {
            "GET /health": "Health check",
            "GET /inventory/{item_name}": "Get current inventory",
            "GET /products/search": "Search products",
            "GET /transactions/history": "Transaction history",
            "GET /bom/{product}": "Get BOM/recipe",
            "POST /command/preview": "Preview command",
            "POST /command/commit": "Execute make command",
            "POST /receive/preview": "Preview receive",
            "POST /receive/commit": "Commit receive",
            "POST /ship/preview": "Preview ship",
            "POST /ship/commit": "Commit ship",
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
def search_products(q: str, _: bool = Depends(verify_api_key)):
    if not q or len(q.strip()) < 2:
        return JSONResponse(status_code=400, content={"error": "Query must be at least 2 characters"})
    query = q.strip()
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if query.isdigit():
                    cur.execute("SELECT name, odoo_code, type FROM products WHERE odoo_code = %s LIMIT 5", (query,))
                    results = cur.fetchall()
                    if results:
                        return {"query": query, "matches": results}
                cur.execute("SELECT name, odoo_code, type FROM products WHERE name ILIKE %s ORDER BY CASE WHEN LOWER(name) = LOWER(%s) THEN 0 ELSE 1 END, LENGTH(name) ASC LIMIT 5", (f"%{query}%", query))
                results = cur.fetchall()
        return {"query": query, "matches": results}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/transactions/history")
def get_transaction_history(_: bool = Depends(verify_api_key), limit: int = Query(default=10, ge=1, le=100), type: Optional[str] = Query(default=None), product: Optional[str] = Query(default=None)):
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = "SELECT t.id as transaction_id, t.type, t.timestamp, t.notes, json_agg(json_build_object('product', p.name, 'odoo_code', p.odoo_code, 'lot', l.lot_code, 'quantity_lb', tl.quantity_lb) ORDER BY tl.id) as lines FROM transactions t LEFT JOIN transaction_lines tl ON tl.transaction_id = t.id LEFT JOIN products p ON p.id = tl.product_id LEFT JOIN lots l ON l.id = tl.lot_id"
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
        result = [{"transaction_id": tx["transaction_id"], "type": tx["type"], "timestamp": tx["timestamp"].isoformat() if tx["timestamp"] else None, "notes": tx["notes"], "lines": [l for l in (tx["lines"] or []) if l.get("product")]} for tx in transactions]
        return {"count": len(result), "filters": {"limit": limit, "type": type, "product": product}, "transactions": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/bom/{product}")
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


@app.post("/command/preview")
def preview_command(request: CommandRequest, _: bool = Depends(verify_api_key)):
    text = request.raw_text.lower().strip()
    if "receive" in text:
        return {"status": "ready", "type": "receive", "preview": f"Ready to receive: {request.raw_text}"}
    if "make" in text or "batch" in text:
        return {"status": "ready", "type": "make", "preview": f"Ready to manufacture: {request.raw_text}"}
    if "adjust" in text:
        return {"status": "ready", "type": "adjust", "preview": f"Ready to adjust: {request.raw_text}"}
    return {"status": "error", "message": "Unknown command."}


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
        lot_code = generate_lot_code(shipper_code)
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
                cur.execute("SELECT l.id as lot_id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available_lb FROM lots l LEFT JOIN transaction_lines tl ON tl.lot_id = l.id WHERE l.product_id = %s GROUP BY l.id, l.lot_code HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0 ORDER BY l.lot_code ASC", (product["id"],))
                lots_with_inventory = cur.fetchall()
                if not lots_with_inventory:
                    raise HTTPException(status_code=400, detail=f"No inventory available for {product['name']}")
                if req.lot_code:
                    selected_lot = next((l for l in lots_with_inventory if l["lot_code"] == req.lot_code), None)
                    if not selected_lot:
                        raise HTTPException(status_code=400, detail=f"Lot '{req.lot_code}' not found or has no inventory. Available lots:\n" + "\n".join([f"• {l['lot_code']} ({l['available_lb']:,.0f} lb)" for l in lots_with_inventory]))
                    lot_selection = "manual"
                else:
                    selected_lot = lots_with_inventory[0]
                    lot_selection = "fifo"
                if selected_lot["available_lb"] < req.quantity_lb:
                    raise HTTPException(status_code=400, detail=f"Insufficient inventory in lot {selected_lot['lot_code']}. Requested: {req.quantity_lb:,.0f} lb, Available: {selected_lot['available_lb']:,.0f} lb.\n\nAvailable lots:\n" + "\n".join([f"• {l['lot_code']} ({l['available_lb']:,.0f} lb)" for l in lots_with_inventory]))
        lot_note = "(auto-selected, FIFO)" if lot_selection == "fifo" else "(specified)"
        preview_message = f"📦 SHIP PREVIEW\n────────────────────────────────\nProduct:      {product['name']} ({product['odoo_code']})\nQuantity:     {req.quantity_lb:,.0f} lb\nCustomer:     {req.customer_name}\nOrder Ref:    {req.order_reference}\nLot:          {selected_lot['lot_code']} {lot_note}\nAvailable:    {selected_lot['available_lb']:,.0f} lb in this lot\n────────────────────────────────\n✓ Say \"confirm\" to proceed\n✎ Or specify a different lot"
        return ShipPreviewResponse(product_id=product["id"], product_name=product["name"], odoo_code=product["odoo_code"], quantity_lb=req.quantity_lb, customer_name=req.customer_name, order_reference=req.order_reference, lot_code=selected_lot["lot_code"], lot_id=selected_lot["lot_id"], available_lb=float(selected_lot["available_lb"]), lot_selection=lot_selection, preview_message=preview_message)
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
        cur.execute("SELECT l.id as lot_id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available_lb FROM lots l LEFT JOIN transaction_lines tl ON tl.lot_id = l.id WHERE l.product_id = %s GROUP BY l.id, l.lot_code HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0 ORDER BY l.lot_code ASC", (product["id"],))
        lots_with_inventory = cur.fetchall()
        if not lots_with_inventory:
            raise HTTPException(status_code=400, detail=f"No inventory available for {product['name']}")
        if req.lot_code:
            selected_lot = next((l for l in lots_with_inventory if l["lot_code"] == req.lot_code), None)
            if not selected_lot:
                raise HTTPException(status_code=400, detail=f"Lot '{req.lot_code}' not found or has no inventory")
        else:
            selected_lot = lots_with_inventory[0]
        if selected_lot["available_lb"] < req.quantity_lb:
            raise HTTPException(status_code=400, detail=f"Insufficient inventory in lot {selected_lot['lot_code']}. Requested: {req.quantity_lb:,.0f} lb, Available: {selected_lot['available_lb']:,.0f} lb")
        timestamp = datetime.now()
        cur.execute("INSERT INTO transactions (type, customer_name, order_reference) VALUES ('ship', %s, %s) RETURNING id", (req.customer_name, req.order_reference))
        transaction_id = cur.fetchone()["id"]
        cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (transaction_id, product["id"], selected_lot["lot_id"], -req.quantity_lb))
        conn.commit()
        cur.close()
        slip_text = generate_packing_slip_text(product["name"], product["odoo_code"], req.quantity_lb, req.customer_name, selected_lot["lot_code"], req.order_reference, timestamp)
        slip_html = generate_packing_slip_html(product["name"], product["odoo_code"], req.quantity_lb, req.customer_name, selected_lot["lot_code"], req.order_reference, timestamp)
        return ShipCommitResponse(success=True, transaction_id=transaction_id, lot_code=selected_lot["lot_code"], quantity_lb=req.quantity_lb, slip_text=slip_text, slip_html=slip_html, message=f"✅ Shipped {req.quantity_lb:,.0f} lb {product['name']} to {req.customer_name} from lot {selected_lot['lot_code']}")
    except HTTPException:
        if conn: conn.rollback()
        raise
    except Exception as e:
        if conn: conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn: conn.close()


# --- MAKE COMMAND ---

MAKE_RE = re.compile(r"^\s*make\s+(?P<n>\d+(?:\.\d+)?)\s+batch\s+(?P<batch>.+?)\s+lot\s+(?P<lot>[A-Za-z0-9\-_]+)\s*$", re.IGNORECASE)

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

@app.post("/command/commit")
def commit_command(request: CommandRequest, _: bool = Depends(verify_api_key)):
    raw = request.raw_text.strip()
    m = MAKE_RE.match(raw)
    if not m:
        return JSONResponse(status_code=400, content={"error": "Unsupported command format.", "expected": "Make 1 batch <Name OR Code> lot <LOT>"})
    n_batches, batch_token, output_lot_code = float(m.group("n")), m.group("batch"), m.group("lot")
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        conn.autocommit = False
        cur = conn.cursor(cursor_factory=RealDictCursor)
        batch, matches = find_product(cur, batch_token)
        if not batch and len(matches) > 1:
            conn.close()
            return JSONResponse(status_code=409, content={"error": "Multiple products match.", "matches": [{"name": m["name"], "odoo_code": m["odoo_code"]} for m in matches]})
        if not batch:
            conn.close()
            return JSONResponse(status_code=404, content={"error": f"Batch not found: {batch_token}"})
        if batch.get("default_batch_lb") is None:
            conn.close()
            return JSONResponse(status_code=400, content={"error": f"default_batch_lb is NULL for '{batch['name']}'"})
        yield_lb = float(batch["default_batch_lb"])
        output_qty = n_batches * yield_lb
        cur.execute("SELECT bf.ingredient_product_id, bf.quantity_lb, p.name AS ingredient_name, p.odoo_code AS ingredient_odoo_code FROM batch_formulas bf JOIN products p ON p.id = bf.ingredient_product_id WHERE bf.product_id = %s ORDER BY p.name", (batch["id"],))
        bom_lines = cur.fetchall()
        if not bom_lines:
            conn.close()
            return JSONResponse(status_code=400, content={"error": f"No BOM found for '{batch['name']}'"})
        cur.execute("INSERT INTO transactions (type, notes) VALUES ('make', %s) RETURNING id", (f"commit: {raw}",))
        tx_id = cur.fetchone()["id"]
        batch_lot_id = get_or_create_lot(cur, batch["id"], output_lot_code)
        consumed = []
        for line in bom_lines:
            ing_id, ing_name, per_batch = int(line["ingredient_product_id"]), line["ingredient_name"], float(line["quantity_lb"])
            qty = -(n_batches * per_batch)
            ing_lot_id = get_or_create_lot(cur, ing_id, "UNKNOWN")
            cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (tx_id, ing_id, ing_lot_id, qty))
            consumed.append({"ingredient": ing_name, "qty_lb": abs(qty), "lot": "UNKNOWN"})
        cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (tx_id, batch["id"], batch_lot_id, output_qty))
        conn.commit()
        conn.close()
        return {"status": "committed", "transaction_id": tx_id, "action": "make", "batch": {"name": batch["name"], "odoo_code": batch.get("odoo_code"), "batches": n_batches, "yield_lb_per_batch": yield_lb, "produced_lb": output_qty, "lot": output_lot_code}, "consumed": consumed, "note": "Ingredient lots defaulted to UNKNOWN (MVP)."}
    except Exception as e:
        if conn:
            try: conn.rollback(); conn.close()
            except: pass
        return JSONResponse(status_code=500, content={"error": str(e)})
