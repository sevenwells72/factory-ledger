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


# --- Shipper Code Generator ---

def generate_shipper_code(shipper_name: str) -> str:
    """
    Generate shipper code from name (Rule 1C-ii):
    - Single word → first 3 letters: Betrimex → BET
    - Multiple words → first 3 + first letter of each subsequent: Better Foods → BETF
    - Max 5 characters, uppercase
    """
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
    """Generate lot code in format YY-MM-DD-SRC"""
    if receive_date is None:
        receive_date = date.today()
    return f"{receive_date.strftime('%y-%m-%d')}-{shipper_code}"


# --- Receipt Generators ---

def generate_receipt_text(
    product_name: str, odoo_code: str, cases: int, case_size_lb: float,
    total_lb: float, shipper_name: str, lot_code: str, bol_reference: str,
    timestamp: datetime
) -> str:
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


def generate_receipt_html(
    product_name: str, odoo_code: str, cases: int, case_size_lb: float,
    total_lb: float, shipper_name: str, lot_code: str, bol_reference: str,
    timestamp: datetime
) -> str:
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
        @media print {{ body {{ margin: 0; }} }}
    </style>
</head>
<body>
    <div class="receipt">
        <div class="header">
            <h1>CNS CONFECTIONERY PRODUCTS</h1>
            <h2>RECEIVING RECEIPT</h2>
        </div>
        <div class="details">
            <div class="row"><span class="label">Date:</span><span>{timestamp.strftime('%B %d, %Y')}</span></div>
            <div class="row"><span class="label">Time:</span><span>{timestamp.strftime('%I:%M %p')}</span></div>
        </div>
        <div class="lot-code">
            <div>LOT CODE</div>
            <span>{lot_code}</span>
        </div>
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


@app.get("/")
def root():
    return {
        "name": "Factory Ledger System",
        "version": "0.7.0",
        "status": "online",
        "endpoints": {
            "GET /health": "Health check",
            "GET /inventory/{item_name}": "Get current inventory",
            "GET /products/search": "Search products by name or code",
            "GET /transactions/history": "Get transaction history",
            "GET /bom/{product}": "Get BOM/recipe for a batch product",
            "POST /command/preview": "Preview a command",
            "POST /command/commit": "Execute a command and write to ledger",
            "POST /receive/preview": "Preview a receive transaction",
            "POST /receive/commit": "Commit a receive transaction",
        },
    }


@app.get("/health")
def health_check():
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            cur.fetchone()
        conn.close()
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "database": "disconnected", "error": str(e)},
        )


@app.get("/inventory/{item_name}")
def get_inventory(item_name: str, _: bool = Depends(verify_api_key)):
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if item_name.isdigit():
                    cur.execute(
                        """
                        SELECT p.name, p.odoo_code, COALESCE(SUM(tl.quantity_lb), 0) AS total
                        FROM products p
                        LEFT JOIN transaction_lines tl ON tl.product_id = p.id
                        WHERE p.odoo_code = %s
                        GROUP BY p.id
                        """,
                        (item_name,),
                    )
                    result = cur.fetchone()
                else:
                    cur.execute(
                        """
                        SELECT p.name, p.odoo_code, COALESCE(SUM(tl.quantity_lb), 0) AS total
                        FROM products p
                        LEFT JOIN transaction_lines tl ON tl.product_id = p.id
                        WHERE LOWER(p.name) = LOWER(%s)
                        GROUP BY p.id
                        """,
                        (item_name,),
                    )
                    result = cur.fetchone()
                    if not result:
                        cur.execute(
                            """
                            SELECT p.name, p.odoo_code, COALESCE(SUM(tl.quantity_lb), 0) AS total
                            FROM products p
                            LEFT JOIN transaction_lines tl ON tl.product_id = p.id
                            WHERE p.name ILIKE %s
                            GROUP BY p.id
                            ORDER BY LENGTH(p.name) ASC
                            LIMIT 1
                            """,
                            (f"%{item_name}%",),
                        )
                        result = cur.fetchone()

        if not result:
            return JSONResponse(
                status_code=404,
                content={"error": "Product not found", "query": item_name},
            )

        return {
            "item": result["name"],
            "odoo_code": result["odoo_code"],
            "on_hand_lb": float(result["total"]),
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/products/search")
def search_products(q: str, _: bool = Depends(verify_api_key)):
    if not q or len(q.strip()) < 2:
        return JSONResponse(
            status_code=400,
            content={"error": "Query must be at least 2 characters"}
        )

    query = q.strip()

    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                if query.isdigit():
                    cur.execute(
                        "SELECT name, odoo_code, type FROM products WHERE odoo_code = %s LIMIT 5",
                        (query,),
                    )
                    results = cur.fetchall()
                    if results:
                        return {"query": query, "matches": results}

                cur.execute(
                    """
                    SELECT name, odoo_code, type FROM products
                    WHERE name ILIKE %s
                    ORDER BY CASE WHEN LOWER(name) = LOWER(%s) THEN 0 ELSE 1 END, LENGTH(name) ASC
                    LIMIT 5
                    """,
                    (f"%{query}%", query),
                )
                results = cur.fetchall()

        return {"query": query, "matches": results}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/transactions/history")
def get_transaction_history(
    _: bool = Depends(verify_api_key),
    limit: int = Query(default=10, ge=1, le=100),
    type: Optional[str] = Query(default=None),
    product: Optional[str] = Query(default=None),
):
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT t.id as transaction_id, t.type, t.timestamp, t.notes,
                        json_agg(json_build_object(
                            'product', p.name, 'odoo_code', p.odoo_code,
                            'lot', l.lot_code, 'quantity_lb', tl.quantity_lb
                        ) ORDER BY tl.id) as lines
                    FROM transactions t
                    LEFT JOIN transaction_lines tl ON tl.transaction_id = t.id
                    LEFT JOIN products p ON p.id = tl.product_id
                    LEFT JOIN lots l ON l.id = tl.lot_id
                """
                conditions, params = [], []
                if type:
                    conditions.append("t.type = %s")
                    params.append(type.lower())
                if product:
                    if product.isdigit():
                        conditions.append("t.id IN (SELECT DISTINCT tl2.transaction_id FROM transaction_lines tl2 JOIN products p2 ON p2.id = tl2.product_id WHERE p2.odoo_code = %s)")
                        params.append(product)
                    else:
                        conditions.append("t.id IN (SELECT DISTINCT tl2.transaction_id FROM transaction_lines tl2 JOIN products p2 ON p2.id = tl2.product_id WHERE p2.name ILIKE %s)")
                        params.append(f"%{product}%")
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                query += " GROUP BY t.id, t.type, t.timestamp, t.notes ORDER BY t.timestamp DESC, t.id DESC LIMIT %s"
                params.append(limit)
                cur.execute(query, params)
                transactions = cur.fetchall()

        result = []
        for tx in transactions:
            lines = [line for line in (tx["lines"] or []) if line.get("product")]
            result.append({
                "transaction_id": tx["transaction_id"],
                "type": tx["type"],
                "timestamp": tx["timestamp"].isoformat() if tx["timestamp"] else None,
                "notes": tx["notes"],
                "lines": lines,
            })

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
                        return {"status": "multiple_matches", "query": product, "message": "Multiple products match. Please specify using one of these Odoo codes:", "suggestions": [{"name": m["name"], "odoo_code": m["odoo_code"]} for m in matches]}
                if not batch:
                    search_terms = []
                    for word in product.lower().replace('#', '').split():
                        if word not in ('batch', 'the', 'a', 'an', 'in', 'for'):
                            number_map = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
                            search_terms.append(number_map.get(word, word))
                    suggestions = []
                    if search_terms:
                        patterns = [f"%{term}%" for term in search_terms]
                        q = "SELECT DISTINCT name, odoo_code FROM products WHERE name ILIKE 'Batch%%' AND (" + " OR ".join(["name ILIKE %s"] * len(patterns)) + ") ORDER BY name LIMIT 5"
                        cur.execute(q, patterns)
                        suggestions = cur.fetchall()
                    if suggestions:
                        return {"status": "suggestions", "query": product, "message": "No exact match found. Did you mean one of these?", "suggestions": [{"name": s["name"], "odoo_code": s["odoo_code"]} for s in suggestions]}
                    else:
                        return JSONResponse(status_code=404, content={"error": f"No batch products found matching: {product}", "suggestion": "Try /products/search?q=..."})
                cur.execute("SELECT p.name AS ingredient, p.odoo_code AS ingredient_code, bf.quantity_lb FROM batch_formulas bf JOIN products p ON p.id = bf.ingredient_product_id WHERE bf.product_id = %s ORDER BY bf.quantity_lb DESC", (batch["id"],))
                ingredients = cur.fetchall()
                if not ingredients:
                    return JSONResponse(status_code=404, content={"error": f"No BOM found for: {batch['name']}", "suggestion": "This product may not have a recipe in batch_formulas."})
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
    return {"status": "error", "message": "Unknown command. Try: 'Make 1 batch Batch Classic Granola #9 lot B0120'"}


# -----------------------------
# RECEIVE ENDPOINTS
# -----------------------------

@app.post("/receive/preview", response_model=ReceivePreviewResponse)
def receive_preview(req: ReceiveRequest, _: bool = Depends(verify_api_key)):
    """Preview a receive transaction before committing."""
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT id, name, odoo_code, uom FROM products
                    WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) = LOWER(%s)
                    LIMIT 5""",
                    (f"%{req.product_name}%", req.product_name)
                )
                products = cur.fetchall()

                if not products:
                    raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")

                if len(products) > 1:
                    product_list = [f"• {p['name']} ({p['odoo_code']})" for p in products]
                    raise HTTPException(status_code=400, detail=f"Multiple products match '{req.product_name}'. Please be more specific:\n" + "\n".join(product_list))

                product = products[0]

        if req.shipper_code_override:
            shipper_code = req.shipper_code_override.upper()[:5]
            shipper_code_auto = False
        else:
            shipper_code = generate_shipper_code(req.shipper_name)
            shipper_code_auto = True

        lot_code = generate_lot_code(shipper_code)
        total_lb = req.cases * req.case_size_lb
        code_note = "(auto-generated)" if shipper_code_auto else "(override)"

        preview_message = f"""📦 RECEIVE PREVIEW
────────────────────────────────
Product:      {product['name']} ({product['odoo_code']})
Quantity:     {req.cases} cases × {req.case_size_lb:.0f} lb = {total_lb:,.0f} lb
Shipper:      {req.shipper_name}
Shipper Code: {shipper_code} {code_note}
Lot Code:     {lot_code}
BOL Ref:      {req.bol_reference}
────────────────────────────────
✓ Say "confirm" to proceed
✎ Or correct any errors"""

        return ReceivePreviewResponse(
            product_id=product["id"], product_name=product["name"], odoo_code=product["odoo_code"],
            cases=req.cases, case_size_lb=req.case_size_lb, total_lb=total_lb,
            shipper_name=req.shipper_name, shipper_code=shipper_code, shipper_code_auto=shipper_code_auto,
            lot_code=lot_code, bol_reference=req.bol_reference, preview_message=preview_message
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/receive/commit", response_model=ReceiveCommitResponse)
def receive_commit(req: ReceiveRequest, _: bool = Depends(verify_api_key)):
    """Commit a receive transaction. Creates lot, transaction, and transaction_line."""
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        conn.autocommit = False
        cur = conn.cursor(cursor_factory=RealDictCursor)

        cur.execute(
            """SELECT id, name, odoo_code FROM products
            WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) = LOWER(%s)
            LIMIT 1""",
            (f"%{req.product_name}%", req.product_name)
        )
        product = cur.fetchone()
        if not product:
            raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")

        if req.shipper_code_override:
            shipper_code = req.shipper_code_override.upper()[:5]
        else:
            shipper_code = generate_shipper_code(req.shipper_name)

        lot_code = generate_lot_code(shipper_code)
        total_lb = req.cases * req.case_size_lb
        timestamp = datetime.now()

        # Get or create lot
        cur.execute("SELECT id FROM lots WHERE lot_code = %s AND product_id = %s", (lot_code, product["id"]))
        lot_row = cur.fetchone()
        if lot_row:
            lot_id = lot_row["id"]
        else:
            cur.execute("INSERT INTO lots (lot_code, product_id) VALUES (%s, %s) RETURNING id", (lot_code, product["id"]))
            lot_id = cur.fetchone()["id"]

        # Create transaction
        cur.execute(
            """INSERT INTO transactions (type, bol_reference, shipper_name, shipper_code, cases_received, case_size_lb)
            VALUES ('receive', %s, %s, %s, %s, %s) RETURNING id""",
            (req.bol_reference, req.shipper_name, shipper_code, req.cases, req.case_size_lb)
        )
        transaction_id = cur.fetchone()["id"]

        # Create transaction line (positive quantity)
        cur.execute(
            "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)",
            (transaction_id, product["id"], lot_id, total_lb)
        )

        conn.commit()
        cur.close()

        receipt_text = generate_receipt_text(product["name"], product["odoo_code"], req.cases, req.case_size_lb, total_lb, req.shipper_name, lot_code, req.bol_reference, timestamp)
        receipt_html = generate_receipt_html(product["name"], product["odoo_code"], req.cases, req.case_size_lb, total_lb, req.shipper_name, lot_code, req.bol_reference, timestamp)

        return ReceiveCommitResponse(
            success=True, transaction_id=transaction_id, lot_id=lot_id, lot_code=lot_code,
            total_lb=total_lb, receipt_text=receipt_text, receipt_html=receipt_html,
            message=f"✅ Received {total_lb:,.0f} lb {product['name']} into lot {lot_code}"
        )

    except HTTPException:
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()


# -----------------------------
# MAKE COMMAND (COMMIT)
# -----------------------------

MAKE_RE = re.compile(
    r"^\s*make\s+(?P<n>\d+(?:\.\d+)?)\s+batch\s+(?P<batch>.+?)\s+lot\s+(?P<lot>[A-Za-z0-9\-_]+)\s*$",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def expand_bom(cur, ingredients, parent_batch_size):
    expanded, has_batches = [], False
    for ing in ingredients:
        cur.execute(
            """SELECT bf.quantity_lb, p.name, p.odoo_code, parent.default_batch_lb
            FROM batch_formulas bf JOIN products p ON p.id = bf.ingredient_product_id
            JOIN products parent ON parent.id = bf.product_id
            WHERE bf.product_id = (SELECT id FROM products WHERE odoo_code = %s OR name = %s LIMIT 1)
            ORDER BY bf.quantity_lb DESC""",
            (ing["ingredient_code"], ing["ingredient"])
        )
        sub_ingredients = cur.fetchall()
        if sub_ingredients:
            has_batches = True
            sub_batch_size = float(sub_ingredients[0]["default_batch_lb"]) if sub_ingredients[0]["default_batch_lb"] else 1
            scale_factor = float(ing["quantity_lb"]) / sub_batch_size if sub_batch_size else 1
            expanded.append({"parent": ing["ingredient"], "parent_odoo_code": ing["ingredient_code"], "parent_qty_lb": float(ing["quantity_lb"]), "raw_ingredients": [{"name": sub["name"], "odoo_code": sub["odoo_code"], "quantity_lb": round(float(sub["quantity_lb"]) * scale_factor, 2)} for sub in sub_ingredients if float(sub["quantity_lb"]) > 0]})
        else:
            expanded.append({"name": ing["ingredient"], "odoo_code": ing["ingredient_code"], "quantity_lb": float(ing["quantity_lb"]), "is_raw": True})
    return {"expanded_ingredients": expanded} if has_batches else {}


def get_or_create_lot(cur, product_id: int, lot_code: str) -> int:
    cur.execute("SELECT id FROM lots WHERE product_id=%s AND lot_code=%s", (product_id, lot_code))
    row = cur.fetchone()
    if row:
        return row["id"]
    cur.execute("INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id", (product_id, lot_code))
    return cur.fetchone()["id"]


def find_product(cur, token: str):
    t = _norm(token)
    if t.isdigit():
        cur.execute("SELECT id, name, odoo_code, default_batch_lb, type FROM products WHERE odoo_code=%s LIMIT 1", (t,))
        row = cur.fetchone()
        if row:
            return (row, [row])
    cur.execute("SELECT id, name, odoo_code, default_batch_lb, type FROM products WHERE LOWER(name)=LOWER(%s) LIMIT 1", (t,))
    row = cur.fetchone()
    if row:
        return (row, [row])
    cur.execute("SELECT id, name, odoo_code, default_batch_lb, type FROM products WHERE name ILIKE %s ORDER BY LENGTH(name) ASC LIMIT 5", (f"%{t}%",))
    matches = cur.fetchall()
    if len(matches) == 1:
        return (matches[0], matches)
    elif len(matches) > 1:
        return (None, matches)
    return (None, [])


@app.post("/command/commit")
def commit_command(request: CommandRequest, _: bool = Depends(verify_api_key)):
    raw = request.raw_text.strip()
    m = MAKE_RE.match(raw)
    if not m:
        return JSONResponse(status_code=400, content={"error": "Unsupported command format (MVP supports make only).", "expected": "Make 1 batch <Batch Name OR Odoo Code> lot <LOT_CODE>", "example": "Make 1 batch Batch Classic Granola #9 lot B0120"})

    n_batches = float(m.group("n"))
    batch_token = m.group("batch")
    output_lot_code = m.group("lot")

    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        conn.autocommit = False
        cur = conn.cursor(cursor_factory=RealDictCursor)

        batch, matches = find_product(cur, batch_token)

        if not batch and len(matches) > 1:
            cur.close()
            conn.close()
            return JSONResponse(status_code=409, content={"error": "Multiple products match. Please specify using Odoo code.", "query": batch_token, "matches": [{"name": m["name"], "odoo_code": m["odoo_code"], "type": m["type"]} for m in matches]})

        if not batch:
            cur.close()
            conn.close()
            return JSONResponse(status_code=404, content={"error": f"Batch product not found: {batch_token}", "suggestion": "Try searching with /products/search?q=..."})

        if batch.get("default_batch_lb") is None:
            cur.close()
            conn.close()
            return JSONResponse(status_code=400, content={"error": f"default_batch_lb is NULL for batch '{batch['name']}'", "suggestion": "Set default_batch_lb in products table for this batch."})

        yield_lb = float(batch["default_batch_lb"])
        output_qty = n_batches * yield_lb

        cur.execute("SELECT bf.ingredient_product_id, bf.quantity_lb, p.name AS ingredient_name, p.odoo_code AS ingredient_odoo_code FROM batch_formulas bf JOIN products p ON p.id = bf.ingredient_product_id WHERE bf.product_id = %s ORDER BY p.name", (batch["id"],))
        bom_lines = cur.fetchall()
        if not bom_lines:
            cur.close()
            conn.close()
            return JSONResponse(status_code=400, content={"error": f"No BOM found in batch_formulas for batch '{batch['name']}'", "suggestion": "Add BOM lines to batch_formulas table for this product."})

        cur.execute("INSERT INTO transactions (type, notes) VALUES ('make', %s) RETURNING id", (f"commit: {raw}",))
        tx_id = cur.fetchone()["id"]

        batch_lot_id = get_or_create_lot(cur, batch["id"], output_lot_code)

        consumed = []
        for line in bom_lines:
            ing_id = int(line["ingredient_product_id"])
            ing_name = line["ingredient_name"]
            per_batch = float(line["quantity_lb"])
            qty = -(n_batches * per_batch)
            ing_lot_id = get_or_create_lot(cur, ing_id, "UNKNOWN")
            cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (tx_id, ing_id, ing_lot_id, qty))
            consumed.append({"ingredient": ing_name, "qty_lb": abs(qty), "lot": "UNKNOWN"})

        cur.execute("INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)", (tx_id, batch["id"], batch_lot_id, output_qty))

        conn.commit()
        cur.close()
        conn.close()

        return {"status": "committed", "transaction_id": tx_id, "action": "make", "batch": {"name": batch["name"], "odoo_code": batch.get("odoo_code"), "batches": n_batches, "yield_lb_per_batch": yield_lb, "produced_lb": output_qty, "lot": output_lot_code}, "consumed": consumed, "note": "Ingredient lots defaulted to UNKNOWN (MVP)."}

    except Exception as e:
        if conn:
            try:
                conn.rollback()
                conn.close()
            except:
                pass
        return JSONResponse(status_code=500, content={"error": str(e)})
