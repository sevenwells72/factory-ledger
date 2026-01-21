from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import re

app = FastAPI(title="Factory Ledger System")

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
API_KEY = (os.getenv("API_KEY") or "").strip()


def verify_api_key(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="API key required")
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


class CommandRequest(BaseModel):
    raw_text: str


@app.get("/")
def root():
    return {
        "name": "Factory Ledger System",
        "version": "0.2.0",
        "status": "online",
        "endpoints": {
            "GET /health": "Health check (real DB check)",
            "GET /inventory/{item_name}": "Get current inventory (requires API key)",
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
def get_inventory(item_name: str, authorization: str = Header(None)):
    verify_api_key(authorization)

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
                    # 1) exact case-insensitive match
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

                    # 2) fallback: forgiving match
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


@app.post("/command/preview")
def preview_command(request: CommandRequest, authorization: str = Header(None)):
    verify_api_key(authorization)

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
# COMMIT IMPLEMENTATION (MVP)
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
    """
    Accept either:
      - exact Odoo code (all digits) -> match products.odoo_code
      - otherwise -> match products.name (case-insensitive exact)
    """
    t = _norm(token)
    if t.isdigit():
        cur.execute("SELECT id, name, odoo_code, default_batch_lb, type FROM products WHERE odoo_code=%s LIMIT 1", (t,))
        return cur.fetchone()

    cur.execute("SELECT id, name, odoo_code, default_batch_lb, type FROM products WHERE LOWER(name)=LOWER(%s) LIMIT 1", (t,))
    return cur.fetchone()

@app.post("/command/commit")
def commit_command(request: CommandRequest, authorization: str = Header(None)):
    verify_api_key(authorization)

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
            # 1) identify batch product
            batch = find_product(cur, batch_token)
            if not batch:
                raise Exception(f"Batch product not found: {batch_token}")

            # yield per batch must be set
            if batch.get("default_batch_lb") is None:
                raise Exception(f"default_batch_lb is NULL for batch '{batch['name']}' (set it in products.default_batch_lb).")

            yield_lb = float(batch["default_batch_lb"])
            output_qty = n_batches * yield_lb

            # 2) load BOM lines
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
                raise Exception(f"No BOM found in batch_formulas for batch '{batch['name']}'.")

            # 3) create transaction header
            cur.execute(
                "INSERT INTO transactions (type, notes) VALUES ('make', %s) RETURNING id",
                (f"commit: {raw}",),
            )
            tx_id = cur.fetchone()["id"]

            # 4) lots
            batch_lot_id = get_or_create_lot(cur, batch["id"], output_lot_code)

            # Ingredient lots default to UNKNOWN (MVP)
            # Ensure UNKNOWN lots exist and write negative consumption lines
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

            # 5) write output line (+)
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
