from fastapi import FastAPI, HTTPException, Header, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
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


@app.get("/")
def root():
    return {
        "name": "Factory Ledger System",
        "version": "0.5.2",
        "status": "online",
        "endpoints": {
            "GET /health": "Health check (real DB check)",
            "GET /inventory/{item_name}": "Get current inventory (requires API key)",
            "GET /products/search": "Search products by name or code (requires API key)",
            "GET /transactions/history": "Get transaction history (requires API key)",
            "GET /bom/{product}": "Get BOM/recipe for a batch product (requires API key)",
            "POST /command/preview": "Preview a command (requires API key)",
            "POST /command/commit": "Execute a command and write to ledger (requires API key)",
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
                        """
                        SELECT name, odoo_code, type
                        FROM products
                        WHERE odoo_code = %s
                        LIMIT 5
                        """,
                        (query,),
                    )
                    results = cur.fetchall()
                    if results:
                        return {"query": query, "matches": results}

                cur.execute(
                    """
                    SELECT name, odoo_code, type
                    FROM products
                    WHERE name ILIKE %s
                    ORDER BY 
                        CASE WHEN LOWER(name) = LOWER(%s) THEN 0 ELSE 1 END,
                        LENGTH(name) ASC
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
    limit: int = Query(default=10, ge=1, le=100, description="Number of transactions to return"),
    type: Optional[str] = Query(default=None, description="Filter by transaction type: make, receive, adjust"),
    product: Optional[str] = Query(default=None, description="Filter by product name or Odoo code"),
):
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT 
                        t.id as transaction_id,
                        t.type,
                        t.timestamp,
                        t.notes,
                        json_agg(
                            json_build_object(
                                'product', p.name,
                                'odoo_code', p.odoo_code,
                                'lot', l.lot_code,
                                'quantity_lb', tl.quantity_lb
                            ) ORDER BY tl.id
                        ) as lines
                    FROM transactions t
                    LEFT JOIN transaction_lines tl ON tl.transaction_id = t.id
                    LEFT JOIN products p ON p.id = tl.product_id
                    LEFT JOIN lots l ON l.id = tl.lot_id
                """
                
                conditions = []
                params = []
                
                if type:
                    conditions.append("t.type = %s")
                    params.append(type.lower())
                
                if product:
                    if product.isdigit():
                        conditions.append("""
                            t.id IN (
                                SELECT DISTINCT tl2.transaction_id 
                                FROM transaction_lines tl2 
                                JOIN products p2 ON p2.id = tl2.product_id 
                                WHERE p2.odoo_code = %s
                            )
                        """)
                        params.append(product)
                    else:
                        conditions.append("""
                            t.id IN (
                                SELECT DISTINCT tl2.transaction_id 
                                FROM transaction_lines tl2 
                                JOIN products p2 ON p2.id = tl2.product_id 
                                WHERE p2.name ILIKE %s
                            )
                        """)
                        params.append(f"%{product}%")
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += """
                    GROUP BY t.id, t.type, t.timestamp, t.notes
                    ORDER BY t.timestamp DESC, t.id DESC
                    LIMIT %s
                """
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

        return {
            "count": len(result),
            "filters": {"limit": limit, "type": type, "product": product},
            "transactions": result,
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/bom/{product}")
def get_bom(product: str, _: bool = Depends(verify_api_key)):
    """
    Get the Bill of Materials (recipe) for a batch product.
    Accepts product name or Odoo code.
    Returns suggestions if no exact match found.
    """
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                batch = None
                
                # Try exact odoo_code match
                if product.isdigit():
                    cur.execute(
                        "SELECT id, name, odoo_code, default_batch_lb FROM products WHERE odoo_code = %s",
                        (product,)
                    )
                    batch = cur.fetchone()
                
                # Try exact name match
                if not batch:
                    cur.execute(
                        "SELECT id, name, odoo_code, default_batch_lb FROM products WHERE LOWER(name) = LOWER(%s)",
                        (product,)
                    )
                    batch = cur.fetchone()
                
                # Try fuzzy search - single match
                if not batch:
                    cur.execute(
                        """
                        SELECT id, name, odoo_code, default_batch_lb 
                        FROM products 
                        WHERE name ILIKE %s AND name ILIKE 'Batch%%'
                        ORDER BY LENGTH(name) ASC
                        LIMIT 5
                        """,
                        (f"%{product}%",)
                    )
                    matches = cur.fetchall()
                    
                    if len(matches) == 1:
                        batch = matches[0]
                    elif len(matches) > 1:
                        # Multiple matches - return suggestions (200 so GPT can read it)
                        return {
                            "status": "multiple_matches",
                            "query": product,
                            "message": "Multiple products match. Please specify using one of these Odoo codes:",
                            "suggestions": [
                                {"name": m["name"], "odoo_code": m["odoo_code"]}
                                for m in matches
                            ]
                        }
                
                # Still no match - try broader search for suggestions
                if not batch:
                    # Extract keywords (handle "batch nine", "granola 9", "#9", etc.)
                    search_terms = []
                    for word in product.lower().replace('#', '').split():
                        if word not in ('batch', 'the', 'a', 'an', 'in', 'for'):
                            # Convert number words to digits
                            number_map = {'one': '1', 'two': '2', 'three': '3', 'four': '4', 
                                         'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 
                                         'nine': '9', 'ten': '10'}
                            search_terms.append(number_map.get(word, word))
                    
                    # Search for any batch products matching keywords
                    suggestions = []
                    if search_terms:
                        # Build search pattern
                        patterns = [f"%{term}%" for term in search_terms]
                        query = """
                            SELECT DISTINCT name, odoo_code 
                            FROM products 
                            WHERE name ILIKE 'Batch%%' AND (
                        """
                        query += " OR ".join(["name ILIKE %s"] * len(patterns))
                        query += ") ORDER BY name LIMIT 5"
                        
                        cur.execute(query, patterns)
                        suggestions = cur.fetchall()
                    
                    if suggestions:
                        return {
                            "status": "suggestions",
                            "query": product,
                            "message": "No exact match found. Did you mean one of these?",
                            "suggestions": [
                                {"name": s["name"], "odoo_code": s["odoo_code"]}
                                for s in suggestions
                            ]
                        }
                    else:
                        return JSONResponse(
                            status_code=404,
                            content={
                                "error": f"No batch products found matching: {product}",
                                "suggestion": "Try /products/search?q=... to find the product, or use the Odoo code directly."
                            }
                        )
                
                # Found a batch - get BOM lines
                cur.execute(
                    """
                    SELECT 
                        p.name AS ingredient,
                        p.odoo_code AS ingredient_code,
                        bf.quantity_lb
                    FROM batch_formulas bf
                    JOIN products p ON p.id = bf.ingredient_product_id
                    WHERE bf.product_id = %s
                    ORDER BY bf.quantity_lb DESC
                    """,
                    (batch["id"],)
                )
                ingredients = cur.fetchall()
                
                if not ingredients:
                    return JSONResponse(
                        status_code=404,
                        content={
                            "error": f"No BOM found for: {batch['name']}",
                            "suggestion": "This product may not have a recipe in batch_formulas."
                        }
                    )
                
                return {
                    "product": batch["name"],
                    "odoo_code": batch["odoo_code"],
                    "batch_size_lb": float(batch["default_batch_lb"]) if batch["default_batch_lb"] else None,
                    "ingredient_count": len(ingredients),
                    "ingredients": [
                        {
                            "name": ing["ingredient"],
                            "odoo_code": ing["ingredient_code"],
                            "quantity_lb": float(ing["quantity_lb"])
                        }
                        for ing in ingredients
                    ]
                }
                
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

    return {
        "status": "error",
        "message": "Unknown command. Try: 'Make 1 batch Batch Classic Granola #9 lot B0120'",
    }


# -----------------------------
# COMMIT IMPLEMENTATION
# -----------------------------

MAKE_RE = re.compile(
    r"^\s*make\s+(?P<n>\d+(?:\.\d+)?)\s+batch\s+(?P<batch>.+?)\s+lot\s+(?P<lot>[A-Za-z0-9\-_]+)\s*$",
    re.IGNORECASE,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def get_or_create_lot(cur, product_id: int, lot_code: str) -> int:
    cur.execute(
        "SELECT id FROM lots WHERE product_id=%s AND lot_code=%s",
        (product_id, lot_code),
    )
    row = cur.fetchone()
    if row:
        return row["id"]
    cur.execute(
        "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
        (product_id, lot_code),
    )
    return cur.fetchone()["id"]


def find_product(cur, token: str):
    t = _norm(token)
    
    if t.isdigit():
        cur.execute(
            "SELECT id, name, odoo_code, default_batch_lb, type FROM products WHERE odoo_code=%s LIMIT 1",
            (t,)
        )
        row = cur.fetchone()
        if row:
            return (row, [row])

    cur.execute(
        "SELECT id, name, odoo_code, default_batch_lb, type FROM products WHERE LOWER(name)=LOWER(%s) LIMIT 1",
        (t,)
    )
    row = cur.fetchone()
    if row:
        return (row, [row])

    cur.execute(
        """
        SELECT id, name, odoo_code, default_batch_lb, type 
        FROM products 
        WHERE name ILIKE %s
        ORDER BY LENGTH(name) ASC
        LIMIT 5
        """,
        (f"%{t}%",)
    )
    matches = cur.fetchall()
    
    if len(matches) == 1:
        return (matches[0], matches)
    elif len(matches) > 1:
        return (None, matches)
    else:
        return (None, [])


@app.post("/command/commit")
def commit_command(request: CommandRequest, _: bool = Depends(verify_api_key)):
    raw = request.raw_text.strip()
    m = MAKE_RE.match(raw)
    if not m:
        return JSONResponse(
            status_code=400,
            content={
                "error": "Unsupported command format (MVP supports make only).",
                "expected": "Make 1 batch <Batch Name OR Odoo Code> lot <LOT_CODE>",
                "example": "Make 1 batch Batch Classic Granola #9 lot B0120",
            },
        )

    n_batches = float(m.group("n"))
    batch_token = m.group("batch")
    output_lot_code = m.group("lot")

    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        conn.autocommit = False

        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            batch, matches = find_product(cur, batch_token)

            if not batch and len(matches) > 1:
                conn.close()
                return JSONResponse(
                    status_code=409,
                    content={
                        "error": "Multiple products match. Please specify using Odoo code.",
                        "query": batch_token,
                        "matches": [
                            {"name": m["name"], "odoo_code": m["odoo_code"], "type": m["type"]}
                            for m in matches
                        ],
                    },
                )

            if not batch:
                conn.close()
                return JSONResponse(
                    status_code=404,
                    content={
                        "error": f"Batch product not found: {batch_token}",
                        "suggestion": "Try searching with /products/search?q=..."
                    },
                )

            if batch.get("default_batch_lb") is None:
                conn.close()
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"default_batch_lb is NULL for batch '{batch['name']}'",
                        "suggestion": "Set default_batch_lb in products table for this batch."
                    },
                )

            yield_lb = float(batch["default_batch_lb"])
            output_qty = n_batches * yield_lb

            cur.execute(
                """
                SELECT bf.ingredient_product_id, bf.quantity_lb, p.name AS ingredient_name, p.odoo_code AS ingredient_odoo_code
                FROM batch_formulas bf
                JOIN products p ON p.id = bf.ingredient_product_id
                WHERE bf.product_id = %s
                ORDER BY p.name
                """,
                (batch["id"],),
            )
            bom_lines = cur.fetchall()
            if not bom_lines:
                conn.close()
                return JSONResponse(
                    status_code=400,
                    content={
                        "error": f"No BOM found in batch_formulas for batch '{batch['name']}'",
                        "suggestion": "Add BOM lines to batch_formulas table for this product."
                    },
                )

            cur.execute(
                "INSERT INTO transactions (type, notes) VALUES ('make', %s) RETURNING id",
                (f"commit: {raw}",),
            )
            tx_id = cur.fetchone()["id"]

            batch_lot_id = get_or_create_lot(cur, batch["id"], output_lot_code)

            consumed = []
            for line in bom_lines:
                ing_id = int(line["ingredient_product_id"])
                ing_name = line["ingredient_name"]
                per_batch = float(line["quantity_lb"])
                qty = -(n_batches * per_batch)

                ing_lot_id = get_or_create_lot(cur, ing_id, "UNKNOWN")

                cur.execute(
                    """
                    INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (tx_id, ing_id, ing_lot_id, qty),
                )
                consumed.append({"ingredient": ing_name, "qty_lb": abs(qty), "lot": "UNKNOWN"})

            cur.execute(
                """
                INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                VALUES (%s, %s, %s, %s)
                """,
                (tx_id, batch["id"], batch_lot_id, output_qty),
            )

            conn.commit()

        conn.close()

        return {
            "status": "committed",
            "transaction_id": tx_id,
            "action": "make",
            "batch": {
                "name": batch["name"],
                "odoo_code": batch.get("odoo_code"),
                "batches": n_batches,
                "yield_lb_per_batch": yield_lb,
                "produced_lb": output_qty,
                "lot": output_lot_code,
            },
            "consumed": consumed,
            "note": "Ingredient lots defaulted to UNKNOWN (MVP).",
        }

    except Exception as e:
        try:
            conn.rollback()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        return JSONResponse(status_code=500, content={"error": str(e)})
