from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
import os

app = FastAPI(title="Factory Ledger System")

# IMPORTANT: strip() removes hidden newlines/spaces that can break sslmode parsing
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
        "version": "0.1.0",
        "status": "online",
        "endpoints": {
            "GET /health": "Health check (real DB check)",
            "GET /inventory/{item_name}": "Get current inventory (requires API key)",
            "POST /command/preview": "Preview a command (requires API key)",
        },
    }


@app.get("/health")
def health_check():
    """
    Real health check: actually connects to Postgres and runs SELECT 1.
    """
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
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT COALESCE(SUM(tl.quantity_lb), 0) AS total
                FROM transaction_lines tl
                JOIN products p ON tl.product_id = p.id
                WHERE LOWER(p.name) = LOWER(%s)
                """,
                (item_name,),
            )
            result = cur.fetchone()
        conn.close()

        total = float(result["total"]) if result and result.get("total") is not None else 0.0
        return {"item": item_name, "on_hand_lb": total}

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
        "message": "Unknown command. Try: 'Receive X lb item lot Y', 'Make batch', or 'Adjust X to Y lb'",
    }

