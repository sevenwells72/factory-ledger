from fastapi import FastAPI, HTTPException, Header, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
import os
import re
import logging
import secrets

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Factory Ledger System", version="2.1.0")

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
API_KEY = (os.getenv("API_KEY") or "").strip()

# Timezone configuration
PLANT_TIMEZONE = ZoneInfo("America/New_York")
TIMEZONE_LABEL = "ET"

# Connection pool (initialized on startup)
db_pool = None


@app.on_event("startup")
async def startup():
    global db_pool
    try:
        db_pool = pool.ThreadedConnectionPool(minconn=2, maxconn=20, dsn=DATABASE_URL)
        logger.info("Database connection pool created")
    except Exception as e:
        logger.error(f"Failed to create connection pool: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    global db_pool
    if db_pool:
        db_pool.closeall()
        logger.info("Database connection pool closed")


@contextmanager
def get_db_connection():
    conn = db_pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        db_pool.putconn(conn)


@contextmanager
def get_transaction():
    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur


def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    if not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


def format_timestamp(dt):
    if dt is None:
        return None, None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    local_dt = dt.astimezone(PLANT_TIMEZONE)
    return local_dt.strftime("%Y-%m-%d"), local_dt.strftime("%I:%M %p") + f" {TIMEZONE_LABEL}"


def get_plant_now():
    return datetime.now(PLANT_TIMEZONE)


# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════

class CommandRequest(BaseModel):
    raw_text: str

class ReceiveRequest(BaseModel):
    product_name: str
    cases: int
    case_size_lb: float
    shipper_name: str
    bol_reference: str
    shipper_code_override: Optional[str] = None

class ShipRequest(BaseModel):
    product_name: str
    quantity_lb: float
    customer_name: str
    order_reference: str
    lot_code: Optional[str] = None

class MultiLotShipRequest(BaseModel):
    product_name: str
    quantity_lb: float
    customer_name: str
    order_reference: str
    lot_allocations: Optional[List[Dict]] = None

class MakeRequest(BaseModel):
    product_name: str
    batches: int
    lot_code: Optional[str] = None
    ingredient_lot_overrides: Optional[Dict[str, str]] = None

class AdjustRequest(BaseModel):
    product_name: str
    lot_code: str
    adjustment_lb: float
    reason: str

# --- Quick Create Models ---
class QuickCreateProductRequest(BaseModel):
    product_name: str
    product_type: str  # ingredient, finished_good, packaging
    uom: str = "lb"
    storage_type: str = "ambient"
    name_confidence: str = "exact"  # exact, approximate
    notes: Optional[str] = None
    performed_by: str = "system"

class QuickCreateBatchProductRequest(BaseModel):
    product_name: str
    category: str  # granola, coconut, chocolate, etc.
    production_context: str  # test_batch, sample, private_label, one_off, standard
    name_confidence: str = "exact"
    notes: Optional[str] = None
    performed_by: str = "system"

# --- Lot Reassignment Models ---
class LotReassignmentRequest(BaseModel):
    to_product_id: int
    reason_code: str  # incorrect_receive, product_merge, data_entry_error, supplier_relabel, other
    reason_notes: Optional[str] = None
    performed_by: str = "system"

# --- Found Inventory Models ---
class AddFoundInventoryRequest(BaseModel):
    product_id: int
    quantity: float
    uom: str = "lb"
    reason_code: str  # found_during_count, found_back_stock, predates_system, unreceived_delivery
    found_location: Optional[str] = None
    estimated_age: str = "unknown"  # recent, 1-2_weeks, 1_month_plus, unknown
    suspected_supplier: Optional[str] = None
    suspected_bol: Optional[str] = None
    notes: Optional[str] = None
    performed_by: str = "system"

class AddFoundInventoryWithNewProductRequest(BaseModel):
    product_name: str
    product_type: str
    quantity: float
    reason_code: str
    uom: str = "lb"
    storage_type: str = "ambient"
    found_location: Optional[str] = None
    estimated_age: str = "unknown"
    suspected_supplier: Optional[str] = None
    notes: Optional[str] = None
    performed_by: str = "system"

# --- Product Verification Models ---
class VerifyProductRequest(BaseModel):
    action: str  # verify, reject, archive
    verified_name: Optional[str] = None
    notes: Optional[str] = None
    performed_by: str = "system"


# ═══════════════════════════════════════════════════════════════
# HEALTH & ROOT ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "name": "Factory Ledger System",
        "version": "2.1.0",
        "status": "online",
        "features": ["receive", "ship", "make", "adjust", "trace", "bom", "quick-create", "lot-reassign", "found-inventory"]
    }


@app.get("/health")
def health_check():
    try:
        with get_transaction() as cur:
            cur.execute("SELECT 1")
        pool_status = f"active, {db_pool.minconn}-{db_pool.maxconn} connections" if db_pool else "not initialized"
        return {"status": "healthy", "database": "connected", "pool": pool_status}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e)})


# ═══════════════════════════════════════════════════════════════
# PRODUCT SEARCH ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/products/search")
def search_products(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=100),
    _: bool = Depends(verify_api_key)
):
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT id, name, odoo_code, type, uom, active,
                       COALESCE(verification_status, 'verified') as verification_status
                FROM products
                WHERE (LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) LIKE LOWER(%s))
                  AND COALESCE(active, true) = true
                ORDER BY 
                    CASE WHEN LOWER(name) = LOWER(%s) THEN 0
                         WHEN LOWER(odoo_code) = LOWER(%s) THEN 1
                         WHEN LOWER(name) LIKE LOWER(%s) THEN 2
                         ELSE 3 END,
                    name
                LIMIT %s
            """, (f"%{q}%", f"%{q}%", q, q, f"{q}%", limit))
            products = cur.fetchall()
        return {"count": len(products), "products": products}
    except Exception as e:
        logger.error(f"Product search failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/products/{product_id}")
def get_product(product_id: int, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("SELECT * FROM products WHERE id = %s", (product_id,))
            product = cur.fetchone()
        if not product:
            raise HTTPException(status_code=404, detail="Product not found")
        return product
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get product failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# INVENTORY ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/inventory/{item_name}")
def get_inventory(item_name: str, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT p.id, p.name, p.odoo_code, p.type, p.uom,
                       COALESCE(SUM(tl.quantity_lb), 0) as total_on_hand
                FROM products p
                LEFT JOIN lots l ON l.product_id = p.id
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE LOWER(p.name) LIKE LOWER(%s) OR LOWER(p.odoo_code) LIKE LOWER(%s)
                GROUP BY p.id
            """, (f"%{item_name}%", f"%{item_name}%"))
            results = cur.fetchall()
        return {"count": len(results), "inventory": results}
    except Exception as e:
        logger.error(f"Get inventory failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/inventory/current")
def get_current_inventory(
    product_type: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    _: bool = Depends(verify_api_key)
):
    try:
        with get_transaction() as cur:
            query = """
                SELECT p.id as product_id, p.name as product_name, p.odoo_code, p.type as product_type,
                       l.id as lot_id, l.lot_code,
                       COALESCE(SUM(tl.quantity_lb), 0) as quantity_on_hand
                FROM products p
                JOIN lots l ON l.product_id = p.id
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE COALESCE(p.active, true) = true
            """
            params = []
            if product_type:
                query += " AND p.type = %s"
                params.append(product_type)
            query += " GROUP BY p.id, l.id HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0 ORDER BY p.name, l.lot_code LIMIT %s"
            params.append(limit)
            cur.execute(query, params)
            inventory = cur.fetchall()
        return {"count": len(inventory), "inventory": inventory}
    except Exception as e:
        logger.error(f"Get current inventory failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# LOT ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/lots/by-code/{lot_code}")
def get_lot_by_code(lot_code: str, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT l.id, l.lot_code, l.product_id, p.name as product_name, p.odoo_code,
                       COALESCE(SUM(tl.quantity_lb), 0) as quantity_on_hand,
                       l.entry_source, l.found_location, l.estimated_age
                FROM lots l
                JOIN products p ON p.id = l.product_id
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE LOWER(l.lot_code) = LOWER(%s)
                GROUP BY l.id, p.id
            """, (lot_code,))
            lot = cur.fetchone()
        if not lot:
            raise HTTPException(status_code=404, detail=f"Lot '{lot_code}' not found")
        return lot
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get lot by code failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/lots/{lot_id}")
def get_lot(lot_id: int, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT l.*, p.name as product_name, p.odoo_code,
                       COALESCE(SUM(tl.quantity_lb), 0) as quantity_on_hand
                FROM lots l
                JOIN products p ON p.id = l.product_id
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE l.id = %s
                GROUP BY l.id, p.id
            """, (lot_id,))
            lot = cur.fetchone()
        if not lot:
            raise HTTPException(status_code=404, detail="Lot not found")
        return lot
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get lot failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# RECEIVE ENDPOINTS
# ═══════════════════════════════════════════════════════════════

def generate_lot_code(cur, shipper_name: str, shipper_code_override: str = None) -> tuple:
    now = get_plant_now()
    date_part = now.strftime("%y-%m-%d")
    
    if shipper_code_override:
        shipper_code = shipper_code_override.upper()[:4]
        auto = False
    else:
        shipper_code = ''.join(c for c in shipper_name.upper() if c.isalpha())[:4]
        auto = True
    
    shipper_code = shipper_code or "UNKN"
    
    # Find next sequence number
    cur.execute("""
        SELECT lot_code FROM lots 
        WHERE lot_code LIKE %s 
        ORDER BY lot_code DESC LIMIT 1
    """, (f"{date_part}-{shipper_code}-%",))
    existing = cur.fetchone()
    
    if existing:
        try:
            last_seq = int(existing['lot_code'].split('-')[-1])
            seq = last_seq + 1
        except (ValueError, IndexError):
            seq = 1
    else:
        seq = 1
    
    lot_code = f"{date_part}-{shipper_code}-{seq:03d}"
    return lot_code, shipper_code, auto


@app.post("/receive/preview")
def receive_preview(req: ReceiveRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT id, name, odoo_code FROM products 
                WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) LIKE LOWER(%s)
                LIMIT 1
            """, (f"%{req.product_name}%", f"%{req.product_name}%"))
            product = cur.fetchone()
            
            if not product:
                return JSONResponse(status_code=404, content={
                    "error": f"Product '{req.product_name}' not found",
                    "suggestion": "Use /products/quick-create to add it first"
                })
            
            lot_code, shipper_code, auto = generate_lot_code(cur, req.shipper_name, req.shipper_code_override)
            total_lb = req.cases * req.case_size_lb
            
            return {
                "product_id": product['id'],
                "product_name": product['name'],
                "odoo_code": product['odoo_code'],
                "cases": req.cases,
                "case_size_lb": req.case_size_lb,
                "total_lb": total_lb,
                "shipper_name": req.shipper_name,
                "shipper_code": shipper_code,
                "shipper_code_auto": auto,
                "lot_code": lot_code,
                "bol_reference": req.bol_reference,
                "preview_message": f"Ready to receive {req.cases} cases ({total_lb} lb) of {product['name']} as lot {lot_code}"
            }
    except Exception as e:
        logger.error(f"Receive preview failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/receive/commit")
def receive_commit(req: ReceiveRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Lock for sequence generation
                cur.execute("SELECT pg_advisory_xact_lock(1)")
                
                cur.execute("""
                    SELECT id, name, odoo_code FROM products 
                    WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) LIKE LOWER(%s)
                    LIMIT 1
                """, (f"%{req.product_name}%", f"%{req.product_name}%"))
                product = cur.fetchone()
                
                if not product:
                    raise HTTPException(status_code=404, detail=f"Product '{req.product_name}' not found")
                
                lot_code, shipper_code, _ = generate_lot_code(cur, req.shipper_name, req.shipper_code_override)
                total_lb = req.cases * req.case_size_lb
                now = get_plant_now()
                
                # Create lot
                cur.execute("""
                    INSERT INTO lots (product_id, lot_code, entry_source)
                    VALUES (%s, %s, 'received')
                    RETURNING id
                """, (product['id'], lot_code))
                lot_id = cur.fetchone()['id']
                
                # Create transaction
                cur.execute("""
                    INSERT INTO transactions (type, timestamp, bol_reference, shipper_name, shipper_code, cases_received, case_size_lb)
                    VALUES ('receive', %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (now, req.bol_reference, req.shipper_name, shipper_code, req.cases, req.case_size_lb))
                txn_id = cur.fetchone()['id']
                
                # Create transaction line
                cur.execute("""
                    INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                    VALUES (%s, %s, %s, %s)
                """, (txn_id, product['id'], lot_id, total_lb))
                
                date_str, time_str = format_timestamp(now)
                receipt = f"RECEIVED: {req.cases} cases ({total_lb} lb) {product['name']}\nLot: {lot_code}\nBOL: {req.bol_reference}\n{date_str} {time_str}"
                
                logger.info(f"Receive committed: {lot_code} - {total_lb} lb of {product['name']}")
                
                return {
                    "success": True,
                    "transaction_id": txn_id,
                    "lot_id": lot_id,
                    "lot_code": lot_code,
                    "total_lb": total_lb,
                    "receipt_text": receipt,
                    "message": f"Received {total_lb} lb as lot {lot_code}"
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Receive commit failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# SHIP ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/ship/preview")
def ship_preview(req: ShipRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT id, name, odoo_code FROM products 
                WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) LIKE LOWER(%s)
                LIMIT 1
            """, (f"%{req.product_name}%", f"%{req.product_name}%"))
            product = cur.fetchone()
            
            if not product:
                raise HTTPException(status_code=404, detail=f"Product '{req.product_name}' not found")
            
            # Find lots with inventory
            cur.execute("""
                SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                FROM lots l
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE l.product_id = %s
                GROUP BY l.id
                HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                ORDER BY l.id ASC
            """, (product['id'],))
            lots = cur.fetchall()
            
            if not lots:
                raise HTTPException(status_code=400, detail=f"No inventory available for {product['name']}")
            
            if req.lot_code:
                selected = next((l for l in lots if l['lot_code'].lower() == req.lot_code.lower()), None)
                if not selected:
                    raise HTTPException(status_code=404, detail=f"Lot '{req.lot_code}' not found or empty")
                lot_selection = "specified"
            else:
                selected = lots[0]
                lot_selection = "FIFO (oldest)"
            
            if float(selected['available']) < req.quantity_lb:
                return JSONResponse(status_code=400, content={
                    "error": f"Insufficient inventory. Lot {selected['lot_code']} has {selected['available']} lb, need {req.quantity_lb} lb",
                    "available_lots": [{"lot_code": l['lot_code'], "available_lb": float(l['available'])} for l in lots]
                })
            
            return {
                "product_id": product['id'],
                "product_name": product['name'],
                "odoo_code": product['odoo_code'],
                "quantity_lb": req.quantity_lb,
                "customer_name": req.customer_name,
                "order_reference": req.order_reference,
                "lot_code": selected['lot_code'],
                "lot_id": selected['id'],
                "available_lb": float(selected['available']),
                "lot_selection": lot_selection,
                "preview_message": f"Ready to ship {req.quantity_lb} lb of {product['name']} from lot {selected['lot_code']}"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ship preview failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ship/commit")
def ship_commit(req: ShipRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, name, odoo_code FROM products 
                    WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) LIKE LOWER(%s)
                    LIMIT 1
                """, (f"%{req.product_name}%", f"%{req.product_name}%"))
                product = cur.fetchone()
                
                if not product:
                    raise HTTPException(status_code=404, detail=f"Product '{req.product_name}' not found")
                
                # Lock and get lot
                if req.lot_code:
                    cur.execute("""
                        SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                        FROM lots l
                        LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                        WHERE l.product_id = %s AND LOWER(l.lot_code) = LOWER(%s)
                        GROUP BY l.id
                        FOR UPDATE OF l
                    """, (product['id'], req.lot_code))
                else:
                    cur.execute("""
                        SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                        FROM lots l
                        LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                        WHERE l.product_id = %s
                        GROUP BY l.id
                        HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                        ORDER BY l.id ASC
                        FOR UPDATE OF l
                        LIMIT 1
                    """, (product['id'],))
                
                lot = cur.fetchone()
                if not lot or float(lot['available']) < req.quantity_lb:
                    raise HTTPException(status_code=400, detail="Insufficient inventory")
                
                now = get_plant_now()
                
                # Create transaction
                cur.execute("""
                    INSERT INTO transactions (type, timestamp, customer_name, order_reference)
                    VALUES ('ship', %s, %s, %s)
                    RETURNING id
                """, (now, req.customer_name, req.order_reference))
                txn_id = cur.fetchone()['id']
                
                # Create transaction line (negative for ship)
                cur.execute("""
                    INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                    VALUES (%s, %s, %s, %s)
                """, (txn_id, product['id'], lot['id'], -req.quantity_lb))
                
                remaining = float(lot['available']) - req.quantity_lb
                date_str, time_str = format_timestamp(now)
                
                logger.info(f"Ship committed: {req.quantity_lb} lb of {product['name']} to {req.customer_name}")
                
                return {
                    "success": True,
                    "transaction_id": txn_id,
                    "lot_code": lot['lot_code'],
                    "quantity_shipped": req.quantity_lb,
                    "remaining_in_lot": remaining,
                    "message": f"Shipped {req.quantity_lb} lb from lot {lot['lot_code']}. {remaining} lb remaining."
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ship commit failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# MULTI-LOT SHIP ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/ship/multi-lot/preview")
def multi_lot_ship_preview(req: MultiLotShipRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT id, name, odoo_code FROM products 
                WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) LIKE LOWER(%s)
                LIMIT 1
            """, (f"%{req.product_name}%", f"%{req.product_name}%"))
            product = cur.fetchone()
            
            if not product:
                raise HTTPException(status_code=404, detail=f"Product '{req.product_name}' not found")
            
            cur.execute("""
                SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                FROM lots l
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE l.product_id = %s
                GROUP BY l.id
                HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                ORDER BY l.id ASC
            """, (product['id'],))
            lots = cur.fetchall()
            
            total_available = sum(float(l['available']) for l in lots)
            
            if total_available < req.quantity_lb:
                raise HTTPException(status_code=400, detail=f"Insufficient total inventory. Have {total_available} lb, need {req.quantity_lb} lb")
            
            # Calculate FIFO allocation
            allocations = []
            remaining_need = req.quantity_lb
            for lot in lots:
                if remaining_need <= 0:
                    break
                take = min(float(lot['available']), remaining_need)
                allocations.append({
                    "lot_id": lot['id'],
                    "lot_code": lot['lot_code'],
                    "available_lb": float(lot['available']),
                    "allocated_lb": take
                })
                remaining_need -= take
            
            return {
                "product_id": product['id'],
                "product_name": product['name'],
                "quantity_lb": req.quantity_lb,
                "customer_name": req.customer_name,
                "order_reference": req.order_reference,
                "allocations": allocations,
                "total_available": total_available,
                "preview_message": f"Will ship {req.quantity_lb} lb from {len(allocations)} lot(s)"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-lot ship preview failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ship/multi-lot/commit")
def multi_lot_ship_commit(req: MultiLotShipRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, name, odoo_code FROM products 
                    WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) LIKE LOWER(%s)
                    LIMIT 1
                """, (f"%{req.product_name}%", f"%{req.product_name}%"))
                product = cur.fetchone()
                
                if not product:
                    raise HTTPException(status_code=404, detail=f"Product '{req.product_name}' not found")
                
                # Lock all lots for this product
                cur.execute("""
                    SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                    FROM lots l
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.product_id = %s
                    GROUP BY l.id
                    HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                    ORDER BY l.id ASC
                    FOR UPDATE OF l
                """, (product['id'],))
                lots = cur.fetchall()
                
                total_available = sum(float(l['available']) for l in lots)
                if total_available < req.quantity_lb:
                    raise HTTPException(status_code=400, detail="Insufficient inventory")
                
                now = get_plant_now()
                
                cur.execute("""
                    INSERT INTO transactions (type, timestamp, customer_name, order_reference)
                    VALUES ('ship', %s, %s, %s)
                    RETURNING id
                """, (now, req.customer_name, req.order_reference))
                txn_id = cur.fetchone()['id']
                
                shipped_lots = []
                remaining_need = req.quantity_lb
                for lot in lots:
                    if remaining_need <= 0:
                        break
                    take = min(float(lot['available']), remaining_need)
                    cur.execute("""
                        INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                        VALUES (%s, %s, %s, %s)
                    """, (txn_id, product['id'], lot['id'], -take))
                    shipped_lots.append({"lot_code": lot['lot_code'], "shipped_lb": take})
                    remaining_need -= take
                
                logger.info(f"Multi-lot ship committed: {req.quantity_lb} lb of {product['name']} from {len(shipped_lots)} lots")
                
                return {
                    "success": True,
                    "transaction_id": txn_id,
                    "quantity_shipped": req.quantity_lb,
                    "lots_used": shipped_lots,
                    "message": f"Shipped {req.quantity_lb} lb from {len(shipped_lots)} lot(s)"
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-lot ship commit failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# MAKE (PRODUCTION) ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/make/preview")
def make_preview(req: MakeRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT id, name, odoo_code, default_batch_lb FROM products 
                WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) LIKE LOWER(%s)
                LIMIT 1
            """, (f"%{req.product_name}%", f"%{req.product_name}%"))
            product = cur.fetchone()
            
            if not product:
                raise HTTPException(status_code=404, detail=f"Product '{req.product_name}' not found")
            
            batch_size = float(product.get('default_batch_lb') or 0)
            total_output = batch_size * req.batches
            
            # Get formula
            cur.execute("""
                SELECT bf.ingredient_product_id, p.name as ingredient_name, bf.quantity_lb
                FROM batch_formulas bf
                JOIN products p ON p.id = bf.ingredient_product_id
                WHERE bf.batch_product_id = %s
            """, (product['id'],))
            formula = cur.fetchall()
            
            ingredients_needed = []
            for ing in formula:
                needed = float(ing['quantity_lb']) * req.batches
                cur.execute("""
                    SELECT COALESCE(SUM(tl.quantity_lb), 0) as available
                    FROM lots l
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.product_id = %s
                """, (ing['ingredient_product_id'],))
                avail = float(cur.fetchone()['available'])
                ingredients_needed.append({
                    "ingredient_id": ing['ingredient_product_id'],
                    "ingredient_name": ing['ingredient_name'],
                    "needed_lb": needed,
                    "available_lb": avail,
                    "sufficient": avail >= needed
                })
            
            all_sufficient = all(i['sufficient'] for i in ingredients_needed)
            
            lot_code = req.lot_code or f"B{get_plant_now().strftime('%m%d')}"
            
            return {
                "product_id": product['id'],
                "product_name": product['name'],
                "batches": req.batches,
                "batch_size_lb": batch_size,
                "total_output_lb": total_output,
                "lot_code": lot_code,
                "ingredients": ingredients_needed,
                "all_ingredients_available": all_sufficient,
                "preview_message": f"Ready to make {req.batches} batch(es) of {product['name']} ({total_output} lb)"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Make preview failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/make/commit")
def make_commit(req: MakeRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, name, odoo_code, default_batch_lb FROM products 
                    WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) LIKE LOWER(%s)
                    LIMIT 1
                """, (f"%{req.product_name}%", f"%{req.product_name}%"))
                product = cur.fetchone()
                
                if not product:
                    raise HTTPException(status_code=404, detail=f"Product '{req.product_name}' not found")
                
                batch_size = float(product.get('default_batch_lb') or 0)
                total_output = batch_size * req.batches
                now = get_plant_now()
                lot_code = req.lot_code or f"B{now.strftime('%m%d')}"
                
                # Create output lot
                cur.execute("""
                    INSERT INTO lots (product_id, lot_code, entry_source)
                    VALUES (%s, %s, 'production_output')
                    RETURNING id
                """, (product['id'], lot_code))
                output_lot_id = cur.fetchone()['id']
                
                # Create transaction
                cur.execute("""
                    INSERT INTO transactions (type, timestamp, notes)
                    VALUES ('make', %s, %s)
                    RETURNING id
                """, (now, f"{req.batches} batch(es) of {product['name']}"))
                txn_id = cur.fetchone()['id']
                
                # Add output
                cur.execute("""
                    INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                    VALUES (%s, %s, %s, %s)
                """, (txn_id, product['id'], output_lot_id, total_output))
                
                # Consume ingredients
                cur.execute("""
                    SELECT bf.ingredient_product_id, bf.quantity_lb
                    FROM batch_formulas bf
                    WHERE bf.batch_product_id = %s
                """, (product['id'],))
                formula = cur.fetchall()
                
                consumed = []
                for ing in formula:
                    needed = float(ing['quantity_lb']) * req.batches
                    
                    # Check for override
                    override_lot = None
                    if req.ingredient_lot_overrides and str(ing['ingredient_product_id']) in req.ingredient_lot_overrides:
                        override_code = req.ingredient_lot_overrides[str(ing['ingredient_product_id'])]
                        cur.execute("""
                            SELECT l.id, l.lot_code FROM lots l
                            WHERE l.product_id = %s AND LOWER(l.lot_code) = LOWER(%s)
                        """, (ing['ingredient_product_id'], override_code))
                        override_lot = cur.fetchone()
                    
                    if override_lot:
                        cur.execute("""
                            INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                            VALUES (%s, %s, %s, %s)
                        """, (txn_id, ing['ingredient_product_id'], override_lot['id'], -needed))
                        consumed.append({"lot_code": override_lot['lot_code'], "consumed_lb": needed})
                    else:
                        # FIFO consumption
                        cur.execute("""
                            SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                            FROM lots l
                            LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                            WHERE l.product_id = %s
                            GROUP BY l.id
                            HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                            ORDER BY l.id ASC
                            FOR UPDATE OF l
                        """, (ing['ingredient_product_id'],))
                        lots = cur.fetchall()
                        
                        remaining = needed
                        for lot in lots:
                            if remaining <= 0:
                                break
                            take = min(float(lot['available']), remaining)
                            cur.execute("""
                                INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                                VALUES (%s, %s, %s, %s)
                            """, (txn_id, ing['ingredient_product_id'], lot['id'], -take))
                            
                            # Record traceability
                            cur.execute("""
                                INSERT INTO ingredient_lot_consumption (batch_transaction_id, ingredient_product_id, ingredient_lot_id, quantity_consumed)
                                VALUES (%s, %s, %s, %s)
                            """, (txn_id, ing['ingredient_product_id'], lot['id'], take))
                            
                            consumed.append({"lot_code": lot['lot_code'], "consumed_lb": take})
                            remaining -= take
                
                logger.info(f"Make committed: {lot_code} - {total_output} lb of {product['name']}")
                
                return {
                    "success": True,
                    "transaction_id": txn_id,
                    "lot_id": output_lot_id,
                    "lot_code": lot_code,
                    "output_lb": total_output,
                    "ingredients_consumed": consumed,
                    "message": f"Produced {total_output} lb as lot {lot_code}"
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Make commit failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# ADJUST ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/adjust/commit")
def adjust_commit(req: AdjustRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT p.id as product_id, p.name, l.id as lot_id, l.lot_code
                    FROM products p
                    JOIN lots l ON l.product_id = p.id
                    WHERE (LOWER(p.name) LIKE LOWER(%s) OR LOWER(p.odoo_code) LIKE LOWER(%s))
                      AND LOWER(l.lot_code) = LOWER(%s)
                """, (f"%{req.product_name}%", f"%{req.product_name}%", req.lot_code))
                result = cur.fetchone()
                
                if not result:
                    raise HTTPException(status_code=404, detail=f"Product/lot combination not found")
                
                now = get_plant_now()
                
                cur.execute("""
                    INSERT INTO transactions (type, timestamp, adjust_reason, notes)
                    VALUES ('adjust', %s, %s, %s)
                    RETURNING id
                """, (now, req.reason, f"Adjustment: {req.adjustment_lb} lb"))
                txn_id = cur.fetchone()['id']
                
                cur.execute("""
                    INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                    VALUES (%s, %s, %s, %s)
                """, (txn_id, result['product_id'], result['lot_id'], req.adjustment_lb))
                
                logger.info(f"Adjust committed: {req.adjustment_lb} lb to lot {result['lot_code']}")
                
                return {
                    "success": True,
                    "transaction_id": txn_id,
                    "lot_code": result['lot_code'],
                    "adjustment_lb": req.adjustment_lb,
                    "reason": req.reason,
                    "message": f"Adjusted lot {result['lot_code']} by {req.adjustment_lb} lb"
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Adjust commit failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# TRACE ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/trace/batch/{lot_code}")
def trace_batch(lot_code: str, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            # Get the batch
            cur.execute("""
                SELECT t.id as transaction_id, t.timestamp, l.lot_code, p.name as product_name,
                       tl.quantity_lb as output_lb
                FROM transactions t
                JOIN transaction_lines tl ON tl.transaction_id = t.id
                JOIN lots l ON l.id = tl.lot_id
                JOIN products p ON p.id = tl.product_id
                WHERE t.type = 'make' AND LOWER(l.lot_code) = LOWER(%s) AND tl.quantity_lb > 0
            """, (lot_code,))
            batch = cur.fetchone()
            
            if not batch:
                raise HTTPException(status_code=404, detail=f"Batch '{lot_code}' not found")
            
            # Get ingredients
            cur.execute("""
                SELECT p.name as ingredient_name, l.lot_code as ingredient_lot, ilc.quantity_consumed
                FROM ingredient_lot_consumption ilc
                JOIN products p ON p.id = ilc.ingredient_product_id
                JOIN lots l ON l.id = ilc.ingredient_lot_id
                WHERE ilc.batch_transaction_id = %s
            """, (batch['transaction_id'],))
            ingredients = cur.fetchall()
            
            date_str, time_str = format_timestamp(batch['timestamp'])
            
            return {
                "batch_lot_code": batch['lot_code'],
                "product_name": batch['product_name'],
                "output_lb": float(batch['output_lb']),
                "produced_date": date_str,
                "produced_time": time_str,
                "ingredients": [
                    {
                        "ingredient_name": ing['ingredient_name'],
                        "lot_code": ing['ingredient_lot'],
                        "quantity_lb": float(ing['quantity_consumed'])
                    } for ing in ingredients
                ]
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trace batch failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/trace/ingredient/{lot_code}")
def trace_ingredient(lot_code: str, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT l.lot_code, p.name as ingredient_name, l.product_id
                FROM lots l
                JOIN products p ON p.id = l.product_id
                WHERE LOWER(l.lot_code) = LOWER(%s)
            """, (lot_code,))
            lot = cur.fetchone()
            
            if not lot:
                raise HTTPException(status_code=404, detail=f"Lot '{lot_code}' not found")
            
            cur.execute("""
                SELECT DISTINCT bl.lot_code as batch_lot, bp.name as batch_product, ilc.quantity_consumed
                FROM ingredient_lot_consumption ilc
                JOIN lots il ON il.id = ilc.ingredient_lot_id
                JOIN transactions t ON t.id = ilc.batch_transaction_id
                JOIN transaction_lines tl ON tl.transaction_id = t.id AND tl.quantity_lb > 0
                JOIN lots bl ON bl.id = tl.lot_id
                JOIN products bp ON bp.id = bl.product_id
                WHERE LOWER(il.lot_code) = LOWER(%s)
            """, (lot_code,))
            batches = cur.fetchall()
            
            return {
                "ingredient_lot_code": lot['lot_code'],
                "ingredient_name": lot['ingredient_name'],
                "used_in_batches": [
                    {
                        "batch_lot_code": b['batch_lot'],
                        "batch_product": b['batch_product'],
                        "quantity_used": float(b['quantity_consumed'])
                    } for b in batches
                ]
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trace ingredient failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# TRANSACTION HISTORY ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/transactions/history")
def get_transaction_history(
    limit: int = Query(default=20, ge=1, le=100),
    transaction_type: Optional[str] = Query(default=None),
    product_name: Optional[str] = Query(default=None),
    _: bool = Depends(verify_api_key)
):
    try:
        with get_transaction() as cur:
            query = """
                SELECT t.id, t.type, t.timestamp, t.bol_reference, t.shipper_name, 
                       t.customer_name, t.order_reference, t.adjust_reason, t.notes,
                       json_agg(json_build_object(
                           'product_name', p.name,
                           'lot_code', l.lot_code,
                           'quantity_lb', tl.quantity_lb
                       )) as lines
                FROM transactions t
                LEFT JOIN transaction_lines tl ON tl.transaction_id = t.id
                LEFT JOIN products p ON p.id = tl.product_id
                LEFT JOIN lots l ON l.id = tl.lot_id
                WHERE 1=1
            """
            params = []
            
            if transaction_type:
                query += " AND t.type = %s"
                params.append(transaction_type)
            
            if product_name:
                query += " AND p.name ILIKE %s"
                params.append(f"%{product_name}%")
            
            query += " GROUP BY t.id ORDER BY t.timestamp DESC LIMIT %s"
            params.append(limit)
            
            cur.execute(query, params)
            transactions = cur.fetchall()
            
            for txn in transactions:
                date_str, time_str = format_timestamp(txn['timestamp'])
                txn['date'] = date_str
                txn['time'] = time_str
            
            return {"count": len(transactions), "transactions": transactions}
    except Exception as e:
        logger.error(f"Get transaction history failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# QUICK-CREATE PRODUCT ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/products/quick-create")
def quick_create_product(req: QuickCreateProductRequest, _: bool = Depends(verify_api_key)):
    """Quick-create a product during receive when it doesn't exist."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if product already exists
                cur.execute("SELECT id, name FROM products WHERE LOWER(name) = LOWER(%s)", (req.product_name,))
                existing = cur.fetchone()
                if existing:
                    return JSONResponse(status_code=409, content={
                        "error": f"Product '{req.product_name}' already exists",
                        "existing_product_id": existing['id']
                    })
                
                verification_notes = f"Quick-created. Name confidence: {req.name_confidence}."
                if req.notes:
                    verification_notes += f" {req.notes}"
                
                cur.execute("""
                    INSERT INTO products (name, type, uom, storage_type, verification_status, verification_notes, created_via, active)
                    VALUES (%s, %s, %s, %s, 'unverified', %s, 'quick_create', true)
                    RETURNING id, name, type, uom, verification_status
                """, (req.product_name, req.product_type, req.uom, req.storage_type, verification_notes))
                product = cur.fetchone()
                
                # Create audit record if table exists
                try:
                    cur.execute("""
                        INSERT INTO product_verification_history (product_id, from_status, to_status, action, action_notes, performed_by)
                        VALUES (%s, NULL, 'unverified', 'created', %s, %s)
                    """, (product['id'], f"Quick-created during receive. {verification_notes}", req.performed_by))
                except Exception:
                    pass  # Table might not exist yet
                
                logger.info(f"Quick-created product: {product['name']} (ID: {product['id']})")
                
                return {
                    "success": True,
                    "product_id": product['id'],
                    "product_name": product['name'],
                    "product_type": product['type'],
                    "verification_status": product['verification_status'],
                    "message": f"Created '{product['name']}' - flagged for verification"
                }
    except Exception as e:
        logger.error(f"Quick-create product failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/products/quick-create-batch")
def quick_create_batch_product(req: QuickCreateBatchProductRequest, _: bool = Depends(verify_api_key)):
    """Quick-create a batch product during production."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id, name FROM products WHERE LOWER(name) = LOWER(%s)", (req.product_name,))
                existing = cur.fetchone()
                if existing:
                    return JSONResponse(status_code=409, content={
                        "error": f"Product '{req.product_name}' already exists",
                        "existing_product_id": existing['id']
                    })
                
                verification_notes = f"Quick-created for production. Category: {req.category}. Context: {req.production_context}. Name confidence: {req.name_confidence}."
                if req.notes:
                    verification_notes += f" {req.notes}"
                
                cur.execute("""
                    INSERT INTO products (name, type, uom, verification_status, verification_notes, production_context, created_via, active)
                    VALUES (%s, 'batch', 'lb', 'unverified', %s, %s, 'quick_create_batch', true)
                    RETURNING id, name, type, verification_status
                """, (req.product_name, verification_notes, req.production_context))
                product = cur.fetchone()
                
                logger.info(f"Quick-created batch product: {product['name']} (ID: {product['id']})")
                
                return {
                    "success": True,
                    "product_id": product['id'],
                    "product_name": product['name'],
                    "product_type": product['type'],
                    "production_context": req.production_context,
                    "verification_status": product['verification_status'],
                    "message": f"Created batch '{product['name']}' - flagged for verification"
                }
    except Exception as e:
        logger.error(f"Quick-create batch product failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# LOT REASSIGNMENT ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/lots/{lot_id}/reassign")
def reassign_lot(lot_id: int, req: LotReassignmentRequest, _: bool = Depends(verify_api_key)):
    """Reassign a lot to a different product."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get current lot info
                cur.execute("""
                    SELECT l.id, l.lot_code, l.product_id, p.name as product_name
                    FROM lots l
                    JOIN products p ON p.id = l.product_id
                    WHERE l.id = %s
                    FOR UPDATE OF l
                """, (lot_id,))
                lot = cur.fetchone()
                
                if not lot:
                    raise HTTPException(status_code=404, detail=f"Lot ID {lot_id} not found")
                
                if lot['product_id'] == req.to_product_id:
                    return JSONResponse(status_code=400, content={
                        "error": f"Lot is already assigned to {lot['product_name']}"
                    })
                
                # Get target product
                cur.execute("SELECT id, name FROM products WHERE id = %s", (req.to_product_id,))
                to_product = cur.fetchone()
                
                if not to_product:
                    raise HTTPException(status_code=404, detail=f"Target product ID {req.to_product_id} not found")
                
                # Check if lot was used in production
                cur.execute("""
                    SELECT COUNT(*) as count FROM ingredient_lot_consumption WHERE ingredient_lot_id = %s
                """, (lot_id,))
                usage = cur.fetchone()
                
                # Update the lot
                cur.execute("UPDATE lots SET product_id = %s WHERE id = %s", (req.to_product_id, lot_id))
                
                # Update transaction_lines
                cur.execute("""
                    UPDATE transaction_lines SET product_id = %s WHERE lot_id = %s
                """, (req.to_product_id, lot_id))
                
                # Record reassignment if table exists
                try:
                    cur.execute("""
                        INSERT INTO lot_reassignments (lot_id, from_product_id, to_product_id, reason_code, reason_notes, reassigned_by)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (lot_id, lot['product_id'], req.to_product_id, req.reason_code, req.reason_notes, req.performed_by))
                except Exception:
                    pass
                
                logger.info(f"Reassigned lot {lot['lot_code']} from {lot['product_name']} to {to_product['name']}")
                
                return {
                    "success": True,
                    "lot_id": lot_id,
                    "lot_code": lot['lot_code'],
                    "from_product": lot['product_name'],
                    "to_product": to_product['name'],
                    "reason_code": req.reason_code,
                    "production_usage_updated": usage['count'] if usage else 0,
                    "message": f"Reassigned lot {lot['lot_code']} to {to_product['name']}"
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lot reassignment failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# FOUND INVENTORY ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/inventory/found")
def add_found_inventory(req: AddFoundInventoryRequest, _: bool = Depends(verify_api_key)):
    """Add inventory found during counts that was never formally received."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Verify product exists
                cur.execute("SELECT id, name FROM products WHERE id = %s", (req.product_id,))
                product = cur.fetchone()
                
                if not product:
                    raise HTTPException(status_code=404, detail=f"Product ID {req.product_id} not found")
                
                now = get_plant_now()
                date_part = now.strftime("%y-%m-%d")
                
                # Generate FOUND lot code
                cur.execute("""
                    SELECT lot_code FROM lots WHERE lot_code LIKE %s ORDER BY lot_code DESC LIMIT 1
                """, (f"{date_part}-FOUND-%",))
                existing = cur.fetchone()
                
                if existing:
                    try:
                        last_seq = int(existing['lot_code'].split('-')[-1])
                        seq = last_seq + 1
                    except (ValueError, IndexError):
                        seq = 1
                else:
                    seq = 1
                
                lot_code = f"{date_part}-FOUND-{seq:03d}"
                
                # Create lot
                cur.execute("""
                    INSERT INTO lots (product_id, lot_code, entry_source, entry_source_notes, found_location, estimated_age)
                    VALUES (%s, %s, 'found_inventory', %s, %s, %s)
                    RETURNING id
                """, (req.product_id, lot_code, req.notes, req.found_location, req.estimated_age))
                lot_id = cur.fetchone()['id']
                
                # Create transaction
                cur.execute("""
                    INSERT INTO transactions (type, timestamp, notes)
                    VALUES ('adjust', %s, %s)
                    RETURNING id
                """, (now, f"Found inventory: {req.reason_code}"))
                txn_id = cur.fetchone()['id']
                
                # Create transaction line
                cur.execute("""
                    INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                    VALUES (%s, %s, %s, %s)
                """, (txn_id, req.product_id, lot_id, req.quantity))
                
                # Record adjustment if table exists
                try:
                    cur.execute("""
                        INSERT INTO inventory_adjustments 
                        (lot_id, product_id, adjustment_type, quantity_before, quantity_after, reason_code, reason_notes, suspected_supplier, adjusted_by)
                        VALUES (%s, %s, 'found', 0, %s, %s, %s, %s, %s)
                    """, (lot_id, req.product_id, req.quantity, req.reason_code, req.notes, req.suspected_supplier, req.performed_by))
                except Exception:
                    pass
                
                logger.info(f"Added found inventory: {lot_code} - {req.quantity} {req.uom} of {product['name']}")
                
                return {
                    "success": True,
                    "lot_id": lot_id,
                    "lot_code": lot_code,
                    "product_name": product['name'],
                    "quantity": req.quantity,
                    "uom": req.uom,
                    "entry_source": "found_inventory",
                    "message": f"Added {req.quantity} {req.uom} of {product['name']} as lot {lot_code}"
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add found inventory failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/inventory/found-with-new-product")
def add_found_inventory_with_new_product(req: AddFoundInventoryWithNewProductRequest, _: bool = Depends(verify_api_key)):
    """Create a new product AND add found inventory in one transaction."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Check if product exists
                cur.execute("SELECT id, name FROM products WHERE LOWER(name) = LOWER(%s)", (req.product_name,))
                existing = cur.fetchone()
                if existing:
                    return JSONResponse(status_code=409, content={
                        "error": f"Product '{req.product_name}' already exists",
                        "existing_product_id": existing['id'],
                        "suggestion": "Use /inventory/found with the existing product_id"
                    })
                
                # Create product
                verification_notes = f"Quick-created during inventory count. {req.notes or ''}"
                cur.execute("""
                    INSERT INTO products (name, type, uom, storage_type, verification_status, verification_notes, created_via, active)
                    VALUES (%s, %s, %s, %s, 'unverified', %s, 'quick_create_found_inventory', true)
                    RETURNING id, name
                """, (req.product_name, req.product_type, req.uom, req.storage_type, verification_notes))
                product = cur.fetchone()
                
                now = get_plant_now()
                date_part = now.strftime("%y-%m-%d")
                
                # Generate lot code
                cur.execute("SELECT lot_code FROM lots WHERE lot_code LIKE %s ORDER BY lot_code DESC LIMIT 1", (f"{date_part}-FOUND-%",))
                existing_lot = cur.fetchone()
                seq = (int(existing_lot['lot_code'].split('-')[-1]) + 1) if existing_lot else 1
                lot_code = f"{date_part}-FOUND-{seq:03d}"
                
                # Create lot
                cur.execute("""
                    INSERT INTO lots (product_id, lot_code, entry_source, entry_source_notes, found_location, estimated_age)
                    VALUES (%s, %s, 'found_inventory', %s, %s, %s)
                    RETURNING id
                """, (product['id'], lot_code, req.notes, req.found_location, req.estimated_age))
                lot_id = cur.fetchone()['id']
                
                # Create transaction
                cur.execute("""
                    INSERT INTO transactions (type, timestamp, notes)
                    VALUES ('adjust', %s, %s)
                    RETURNING id
                """, (now, f"Found inventory with new product: {req.reason_code}"))
                txn_id = cur.fetchone()['id']
                
                cur.execute("""
                    INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                    VALUES (%s, %s, %s, %s)
                """, (txn_id, product['id'], lot_id, req.quantity))
                
                logger.info(f"Created product and found inventory: {product['name']} - {lot_code}")
                
                return {
                    "success": True,
                    "product_id": product['id'],
                    "product_name": product['name'],
                    "verification_status": "unverified",
                    "lot_id": lot_id,
                    "lot_code": lot_code,
                    "quantity": req.quantity,
                    "message": f"Created '{product['name']}' and added {req.quantity} {req.uom} as lot {lot_code}"
                }
    except Exception as e:
        logger.error(f"Add found inventory with new product failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# REVIEW QUEUE ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/products/unverified")
def get_unverified_products(limit: int = Query(default=50, ge=1, le=200), _: bool = Depends(verify_api_key)):
    """Get products that need verification."""
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT id, name, type, uom, verification_status, verification_notes, created_via,
                       created_at
                FROM products
                WHERE COALESCE(verification_status, 'verified') = 'unverified'
                  AND COALESCE(active, true) = true
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            products = cur.fetchall()
        return {"count": len(products), "products": products}
    except Exception as e:
        logger.error(f"Get unverified products failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/products/test-batches")
def get_test_batches(limit: int = Query(default=50, ge=1, le=200), _: bool = Depends(verify_api_key)):
    """Get test batch products that may need review."""
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT id, name, type, verification_status, production_context, verification_notes
                FROM products
                WHERE COALESCE(production_context, 'standard') IN ('test_batch', 'sample', 'one_off')
                  AND COALESCE(active, true) = true
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            products = cur.fetchall()
        return {"count": len(products), "products": products}
    except Exception as e:
        logger.error(f"Get test batches failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/inventory/found/queue")
def get_found_inventory_queue(limit: int = Query(default=50, ge=1, le=200), _: bool = Depends(verify_api_key)):
    """Get found inventory lots that may need review."""
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT l.id as lot_id, l.lot_code, p.id as product_id, p.name as product_name,
                       COALESCE(SUM(tl.quantity_lb), 0) as quantity_on_hand,
                       l.entry_source, l.found_location, l.estimated_age, l.entry_source_notes
                FROM lots l
                JOIN products p ON p.id = l.product_id
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE l.entry_source = 'found_inventory'
                GROUP BY l.id, p.id
                HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                ORDER BY l.id DESC
                LIMIT %s
            """, (limit,))
            lots = cur.fetchall()
        return {"count": len(lots), "found_inventory": lots}
    except Exception as e:
        logger.error(f"Get found inventory queue failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# PRODUCT VERIFICATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/products/{product_id}/verify")
def verify_product(product_id: int, req: VerifyProductRequest, _: bool = Depends(verify_api_key)):
    """Verify, reject, or archive a product."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id, name, verification_status FROM products WHERE id = %s FOR UPDATE", (product_id,))
                product = cur.fetchone()
                
                if not product:
                    raise HTTPException(status_code=404, detail=f"Product ID {product_id} not found")
                
                old_status = product.get('verification_status', 'unverified')
                
                if req.action == 'verify':
                    new_status = 'verified'
                    new_name = req.verified_name or product['name']
                    cur.execute("""
                        UPDATE products SET verification_status = %s, name = %s, verification_notes = %s
                        WHERE id = %s
                    """, (new_status, new_name, req.notes, product_id))
                    
                elif req.action == 'reject':
                    new_status = 'rejected'
                    cur.execute("""
                        UPDATE products SET verification_status = %s, active = false, verification_notes = %s
                        WHERE id = %s
                    """, (new_status, req.notes, product_id))
                    
                elif req.action == 'archive':
                    new_status = 'archived'
                    cur.execute("""
                        UPDATE products SET verification_status = %s, active = false, verification_notes = %s
                        WHERE id = %s
                    """, (new_status, req.notes, product_id))
                else:
                    raise HTTPException(status_code=400, detail=f"Invalid action: {req.action}")
                
                # Record history if table exists
                try:
                    cur.execute("""
                        INSERT INTO product_verification_history (product_id, from_status, to_status, action, action_notes, performed_by)
                        VALUES (%s, %s, %s, %s, %s, %s)
                    """, (product_id, old_status, new_status, req.action, req.notes, req.performed_by))
                except Exception:
                    pass
                
                logger.info(f"Product {product_id} {req.action}: {old_status} -> {new_status}")
                
                return {
                    "success": True,
                    "product_id": product_id,
                    "action": req.action,
                    "from_status": old_status,
                    "to_status": new_status,
                    "message": f"Product {req.action}d successfully"
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verify product failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# BOM ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/bom/products")
def list_bom_products(
    product_type: Optional[str] = Query(default=None),
    q: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    _: bool = Depends(verify_api_key)
):
    """List products with optional type filter and search."""
    try:
        with get_transaction() as cur:
            query = "SELECT id, name, odoo_code, type, uom, default_batch_lb FROM products WHERE COALESCE(active, true) = true"
            params = []
            
            if product_type:
                query += " AND type = %s"
                params.append(product_type)
            
            if q:
                query += " AND (LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) LIKE LOWER(%s))"
                params.extend([f"%{q}%", f"%{q}%"])
            
            query += " ORDER BY name LIMIT %s"
            params.append(limit)
            
            cur.execute(query, params)
            products = cur.fetchall()
        return {"count": len(products), "products": products}
    except Exception as e:
        logger.error(f"List BOM products failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/bom/batches/{batch_id}/formula")
def get_batch_formula(batch_id: int, _: bool = Depends(verify_api_key)):
    """Get the formula (ingredients) for a batch product."""
    try:
        with get_transaction() as cur:
            cur.execute("SELECT id, name, default_batch_lb FROM products WHERE id = %s", (batch_id,))
            batch = cur.fetchone()
            
            if not batch:
                raise HTTPException(status_code=404, detail=f"Batch product ID {batch_id} not found")
            
            cur.execute("""
                SELECT bf.ingredient_product_id, p.name as ingredient_name, p.odoo_code, bf.quantity_lb
                FROM batch_formulas bf
                JOIN products p ON p.id = bf.ingredient_product_id
                WHERE bf.batch_product_id = %s
                ORDER BY bf.quantity_lb DESC
            """, (batch_id,))
            ingredients = cur.fetchall()
            
            return {
                "batch_id": batch['id'],
                "batch_name": batch['name'],
                "batch_weight_lb": float(batch['default_batch_lb']) if batch['default_batch_lb'] else None,
                "ingredients": ingredients
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get batch formula failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# REASON CODES ENDPOINT
# ═══════════════════════════════════════════════════════════════

@app.get("/reason-codes")
def get_reason_codes(_: bool = Depends(verify_api_key)):
    """Get available reason codes for adjustments and reassignments."""
    return {
        "found_inventory_reasons": [
            {"code": "found_during_count", "description": "Discovered during physical inventory count"},
            {"code": "found_back_stock", "description": "Found in back stock or secondary location"},
            {"code": "predates_system", "description": "Inventory that existed before system go-live"},
            {"code": "unreceived_delivery", "description": "Delivery that was never formally received"}
        ],
        "lot_reassignment_reasons": [
            {"code": "incorrect_receive", "description": "Originally received against wrong product"},
            {"code": "product_merge", "description": "Products being merged/consolidated"},
            {"code": "data_entry_error", "description": "Simple data entry mistake"},
            {"code": "supplier_relabel", "description": "Supplier changed product labeling"},
            {"code": "other", "description": "Other reason (specify in notes)"}
        ],
        "adjustment_reasons": [
            {"code": "damage", "description": "Product damaged"},
            {"code": "spoilage", "description": "Product spoiled or expired"},
            {"code": "count_correction", "description": "Correction from physical count"},
            {"code": "sample", "description": "Used for samples"},
            {"code": "other", "description": "Other reason (specify in notes)"}
        ]
    }
