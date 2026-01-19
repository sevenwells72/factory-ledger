from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from datetime import datetime
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import re

app = FastAPI(title="Factory Ledger System")

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
API_KEY = os.getenv("API_KEY")

def get_db_connection():
      return psycopg2.connect(DATABASE_URL)

def verify_api_key(authorization: str = Header(None)):
      if not authorization:
                raise HTTPException(status_code=401, detail="API key required")
            if authorization != f"Bearer {API_KEY}":
                      raise HTTPException(status_code=403, detail="Invalid API key")
                  return True

class CommandRequest(BaseModel):
      raw_text: str

class CommitRequest(BaseModel):
      preview_id: str
    confirmed: bool

# ===== PARSER =====
def parse_command(raw_text: str):
      text = raw_text.lower().strip()

    # Receive
    match = re.match(r'receive\s+(\d+)\s+lb\s+(.+?)\s+lot\s+([a-z0-9]+)', text)
    if match:
              return {
                            "type": "receive",
                            "quantity_lb": float(match.group(1)),
                            "item": match.group(2).strip(),
                            "lot": match.group(3).strip()
              }

    # Make batch
    match = re.match(r'make\s+(\d+)?\s*batch\s+(.+?)\s+lot\s+([a-z0-9]+)(?:\s+using\s+(.+))?', text)
    if match:
              batch_count = int(match.group(1)) if match.group(1) else 1
              product = match.group(2).strip()
              output_lot = match.group(3).strip()
              ingredients_text = match.group(4)

        ingredients = []
        if ingredients_text:
                      ing_matches = re.findall(r'(\d+)\s+lb\s+(.+?)\s+lot\s+([a-z0-9]+)', ingredients_text)
                      for qty, item, lot in ing_matches:
                                        ingredients.append({
                                                              "quantity_lb": float(qty),
                                                              "item": item.strip(),
                                                              "lot": lot.strip()
                                        })

                  return {
                                "type": "make",
                                "batch_count": batch_count,
                                "product": product,
                                "output_lot": output_lot,
                                "ingredients": ingredients
                  }

    # Adjust
    match = re.match(r'adjust\s+(.+?)\s+to\s+(\d+)\s+lb', text)
    if match:
              return {
                            "type": "adjust",
                            "item": match.group(1).strip(),
                            "quantity_lb": float(match.group(2))
              }

    # Query
    if "how much" in text or ("what" in text and "have" in text):
              item_match = re.search(r'how much\s+(.+?)\s+do we', text)
              if item_match:
                            item = item_match.group(1).strip()
    else:
            item_match = re.search(r'what\s+(.+?)\s+do we', text)
                  item = item_match.group(1).strip() if item_match else None

        return {
                      "type": "query",
                      "query_type": "inventory",
                      "item": item
        }

    if "what went into" in text or "what batches used" in text:
              return {
                            "type": "query",
                            "query_type": "batch_or_ingredient",
                            "text": text
              }

    return None

# ===== DATABASE HELPERS =====
def get_product_id(conn, product_name: str):
      with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id FROM products WHERE LOWER(name) = LOWER(%s)", (product_name,))
                result = cur.fetchone()
                return result['id'] if result else None

def get_lot_id(conn, product_id: int, lot_code: str):
      with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                              "SELECT id FROM lots WHERE product_id = %s AND LOWER(lot_code) = LOWER(%s)",
                              (product_id, lot_code)
                )
                result = cur.fetchone()
                return result['id'] if result else None

def get_inventory(conn, item_name: str):
      product_id = get_product_id(conn, item_name)
    if not product_id:
              return None

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
              cur.execute(
                            "SELECT COALESCE(SUM(quantity_lb), 0) as total FROM transaction_lines WHERE product_id = %s",
                            (product_id,)
              )
              result = cur.fetchone()
              return float(result['total'])

# ===== ENDPOINTS =====
@app.get("/health")
def health_check():
      return {"status": "ok"}

@app.post("/command/preview")
def preview_command(request: CommandRequest, authorized: bool = Depends(verify_api_key)):
      parsed = parse_command(request.raw_text)

    if not parsed:
              return JSONResponse(
                            status_code=400,
                            content={"error": "Could not parse command", "raw_text": request.raw_text}
              )

    conn = get_db_connection()

    try:
              if parsed["type"] == "receive":
                            product_id = get_product_id(conn, parsed["item"])
                            if not product_id:
                                              return JSONResponse(
                                                                    status_code=400,
                                                                    content={
                                                                                              "status": "needs_clarification",
                                                                                              "question": f"Unknown product: {parsed['item']}. Did you mean one of these?",
                                                                                              "options": ["Classic Granola", "Sugar", "Oats"]
                                                                    }
                                              )

                            return JSONResponse(
                                status_code=200,
                                content={
                                    "status": "ready",
                                    "preview": {
                                        "type": "receive",
                                        "item": parsed["item"],
                                        "quantity_lb": parsed["quantity_lb"],
                                        "lot": parsed["lot"]
                                    }
                                }
                            )

              elif parsed["type"] == "make":
                            product_id = get_product_id(conn, parsed["product"])
                            if not product_id:
                                              return JSONResponse(
                                                                    status_code=400,
                                                                    content={"error": f"Unknown product: {parsed['product']}"}
                                              )

                            if parsed["ingredients"]:
                                              preview_ingredients = []
                                              for ing in parsed["ingredients"]:
                                                                    ing_id = get_product_id(conn, ing["item"])
                                                                    if not ing_id:
                                                                                              return JSONResponse(
                                                                                                                            status_code=400,
                                                                                                                            content={"error": f"Unknown ingredient: {ing['item']}"}
                                                                                                )
                                                                                          preview_ingredients.append({
                                                                        "item": ing["item"],
                                                                        "quantity_lb": ing["quantity_lb"],
                                                                        "lot": ing["lot"]
                                                                    })
                            else:
                                              preview_ingredients = []

                            produced_lb = parsed["batch_count"] * 300

            return JSONResponse(
                              status_code=200,
                              content={
                                                    "status": "ready",
                                                    "preview": {
                                                                              "type": "make",
                                                                              "product": parsed["product"],
                                                                              "batch_count": parsed["batch_count"],
                                                                              "produced_lb": produced_lb,
                                                                              "output_lot": parsed["output_lot"],
                                                                              "ingredients": preview_ingredients
                                                    }
                              }
            )

elif parsed["type"] == "adjust":
            product_id = get_product_id(conn, parsed["item"])
            if not product_id:
                              return JSONResponse(
                                                    status_code=400,
                                                    content={"error": f"Unknown product: {parsed['item']}"}
                              )

            current = get_inventory(conn, parsed["item"])

            return JSONResponse(
                              status_code=200,
                              content={
                                                    "status": "ready",
                                                    "preview": {
                                                                              "type": "adjust",
                                                                              "item": parsed["item"],
                                                                              "current_lb": current,
                                                                              "adjusted_to_lb": parsed["quantity_lb"],
                                                                              "delta_lb": parsed["quantity_lb"] - current
                                                    }
                              }
            )

elif parsed["type"] == "query":
            return JSONResponse(
                              status_code=200,
                              content={
                                                    "status": "ready",
                                                    "preview": {
                                                                              "type": "query",
                                                                              "query": parsed
                                                    }
                              }
            )

finally:
        conn.close()

    return JSONResponse(status_code=400, content={"error": "Unknown command type"})

@app.post("/command/commit")
def commit_command(request: CommitRequest, authorized: bool = Depends(verify_api_key)):
      if not request.confirmed:
                return JSONResponse(
                              status_code=400,
                              content={"error": "Command not confirmed"}
                )

    return JSONResponse(
              status_code=200,
              content={
                            "status": "success",
                            "message": "Command would be committed (implementation pending)"
              }
    )

@app.get("/inventory/{item_name}")
def get_item_inventory(item_name: str, authorized: bool = Depends(verify_api_key)):
      conn = get_db_connection()
    try:
              inventory = get_inventory(conn, item_name)
              if inventory is None:
                            return JSONResponse(
                                              status_code=404,
                                              content={"error": f"Unknown product: {item_name}"}
                            )

              return JSONResponse(
                  status_code=200,
                  content={
                      "item": item_name,
                      "on_hand_lb": inventory
                  }
              )
finally:
        conn.close()

@app.get("/")
def root():
      return {
                "name": "Factory Ledger System",
                "version": "0.1.0",
                "endpoints": {
                              "POST /command/preview": "Preview a command",
                              "POST /command/commit": "Commit a command",
                              "GET /inventory/{item_name}": "Get current inventory",
                              "GET /health": "Health check"
                }
      }
