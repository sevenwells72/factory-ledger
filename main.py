from fastapi import FastAPI, HTTPException, Header, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Union
import json
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

app = FastAPI(title="Factory Ledger System", version="2.4.0")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
API_KEY = (os.getenv("API_KEY") or "").strip()

# Timezone configuration
PLANT_TIMEZONE = ZoneInfo("America/New_York")
TIMEZONE_LABEL = "ET"

# ═══════════════════════════════════════════════════════════════
# SKU PROTECTION — Private-Label Guardrails
# ═══════════════════════════════════════════════════════════════

MERGE_KEYWORDS = ["merge", "deprecat", "consolidat", "migrat"]

# Odoo codes for verified private-label finished goods + exclusive batches
PRIVATE_LABEL_ODOO_CODES = [
    '893',    # CQ Coconut Sweetened Flake 10 LB (Chef Quality)
    '1614',   # CQ Granola 10 LB (Chef Quality)
    '67470',  # Coconut Sweetened Fancy UNIPRO 10 LB
    '67473',  # Coconut Sweetened Medium UNIPRO 10 LB
    '67476',  # Coconut Sweetened Flake UNIPRO 10 LB
    '70002',  # Granola SS Original 12x10 OZ Case (Sunshine)
    '70003',  # Granola SS Chocolate Chip 12x10 OZ Case (Sunshine)
    '70010',  # Granola SS Original Low Carb 12x10 OZ Case (Sunshine)
    '70011',  # Granola SS Cranberry 12x10 OZ Case (Sunshine)
    '70056',  # Granola Setton Cocoa Crunch 25 LB
    '70070',  # Granola SS Chocolate Chip Low Carb 12x10 OZ Case (Sunshine)
    '70073',  # BS Granola – Peanut Butter Banana – 6x7 OZ Case (Blue Stripes)
    '70074',  # BS Granola – Dark Chocolate – 6x7 OZ Case (Blue Stripes)
    '70077',  # Granola Setton Cinnamon Spice Almond 25 LB
    '70078',  # Granola Setton Morning Latte Crunch 25 LB
    '70079',  # BS Almond Butter Granola – 6x7 OZ Case (Blue Stripes)
    '70080',  # BS Granola – Hazelnut Butter – 6x7 OZ Case (Blue Stripes)
    '70081',  # Granola Setton Good Ol 25 LB
    '70082',  # Granola Setton French Vanilla 25 LB
]


def check_private_label_merge(product_name: str, label_type: str, reason: str, quantity: float):
    """Check if an adjustment would violate private-label SKU protection.
    Returns a warning message string if blocked, or None if allowed."""
    if label_type != 'private_label':
        return None
    reason_lower = reason.lower()
    is_merge_reason = any(kw in reason_lower for kw in MERGE_KEYWORDS)
    if is_merge_reason and quantity < 0:
        return (
            f"BLOCKED: Cannot merge/deprecate a private-label SKU. "
            f"'{product_name}' is identity-protected. "
            f"If this is a physical repack, use reason 'repack' instead."
        )
    return None

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

    # Migration: Add label_type column for SKU protection
    try:
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                # Add column if it doesn't exist
                cur.execute("""
                    ALTER TABLE products ADD COLUMN IF NOT EXISTS label_type TEXT DEFAULT 'house'
                """)

                # Set private-label flags on verified finished goods by odoo_code
                cur.execute("""
                    UPDATE products SET label_type = 'private_label'
                    WHERE odoo_code = ANY(%s) AND COALESCE(label_type, 'house') != 'private_label'
                """, (PRIVATE_LABEL_ODOO_CODES,))
                updated_by_code = cur.rowcount

                # Set private-label flags on Blue Stripes exclusive batch products
                cur.execute("""
                    UPDATE products SET label_type = 'private_label'
                    WHERE name ILIKE 'Batch BS %%'
                      AND COALESCE(label_type, 'house') != 'private_label'
                """)
                updated_by_name = cur.rowcount

                # Set private-label flags on Setton batch products
                cur.execute("""
                    UPDATE products SET label_type = 'private_label'
                    WHERE name ILIKE 'Batch Setton %%'
                      AND COALESCE(label_type, 'house') != 'private_label'
                """)
                updated_by_name += cur.rowcount

                conn.commit()
                if updated_by_code + updated_by_name > 0:
                    logger.info(f"SKU protection migration: flagged {updated_by_code + updated_by_name} products as private_label")
                else:
                    logger.info("SKU protection: label_type column up to date")
        finally:
            db_pool.putconn(conn)
    except Exception as e:
        logger.warning(f"SKU protection migration warning (non-fatal): {e}")


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
# BILINGUAL SUPPORT HELPERS
# ═══════════════════════════════════════════════════════════════

def validate_bilingual(english_val, spanish_val, field_name: str):
    """Validate bilingual field pair: English required when Spanish is provided."""
    if spanish_val and not english_val:
        raise HTTPException(400,
            f"English version required. Provide '{field_name}' along with '{field_name}_es'."
        )


def bilingual_response(english_val, spanish_val, field_name: str) -> dict:
    """Return bilingual fields for a response dict. Only includes _es if it has a value."""
    result = {field_name: english_val}
    if spanish_val:
        result[f"{field_name}_es"] = spanish_val
    return result


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
    ingredient_lot_overrides: Optional[Union[Dict[str, str], str]] = None
    excluded_ingredients: Optional[List[int]] = None
    
    def get_lot_overrides(self) -> Optional[Dict[str, str]]:
        """Parse ingredient_lot_overrides whether it's a dict or JSON string"""
        if self.ingredient_lot_overrides is None:
            return None
        if isinstance(self.ingredient_lot_overrides, str):
            try:
                parsed = json.loads(self.ingredient_lot_overrides)
                if isinstance(parsed, dict):
                    return parsed
                return None
            except (json.JSONDecodeError, TypeError):
                return None
        return self.ingredient_lot_overrides

class AdjustRequest(BaseModel):
    product_name: str
    lot_code: str
    adjustment_lb: float
    reason: str
    reason_es: Optional[str] = None

class QuickCreateProductRequest(BaseModel):
    product_name: str
    product_type: str
    uom: str = "lb"
    storage_type: str = "ambient"
    name_confidence: str = "exact"
    notes: Optional[str] = None
    notes_es: Optional[str] = None
    performed_by: str = "system"

class QuickCreateBatchProductRequest(BaseModel):
    product_name: str
    category: str
    production_context: str
    name_confidence: str = "exact"
    notes: Optional[str] = None
    notes_es: Optional[str] = None
    performed_by: str = "system"

class LotReassignmentRequest(BaseModel):
    to_product_id: int
    reason_code: str
    reason_notes: Optional[str] = None
    reason_notes_es: Optional[str] = None
    performed_by: str = "system"

class AddFoundInventoryRequest(BaseModel):
    product_id: int
    quantity: float
    uom: str = "lb"
    reason_code: str
    found_location: Optional[str] = None
    estimated_age: str = "unknown"
    suspected_supplier: Optional[str] = None
    suspected_bol: Optional[str] = None
    notes: Optional[str] = None
    notes_es: Optional[str] = None
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
    notes_es: Optional[str] = None
    performed_by: str = "system"

class VerifyProductRequest(BaseModel):
    action: str
    verified_name: Optional[str] = None
    notes: Optional[str] = None
    notes_es: Optional[str] = None
    performed_by: str = "system"


# ═══════════════════════════════════════════════════════════════
# SALES PYDANTIC MODELS (v2.3.0)
# ═══════════════════════════════════════════════════════════════

class CustomerCreate(BaseModel):
    name: str
    contact_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    notes: Optional[str] = None
    notes_es: Optional[str] = None

class CustomerUpdate(BaseModel):
    name: Optional[str] = None
    contact_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    notes: Optional[str] = None
    notes_es: Optional[str] = None
    active: Optional[bool] = None

class OrderLineInput(BaseModel):
    product_name: str
    quantity: Optional[float] = None
    unit: Optional[str] = None               # None = not explicitly provided
    case_weight_lb: Optional[float] = None
    quantity_lb: Optional[float] = None
    unit_price: Optional[float] = None
    notes: Optional[str] = None
    notes_es: Optional[str] = None
    _unit_explicitly_set: bool = False        # internal tracking flag

    class Config:
        underscore_attrs_are_private = True

    @validator('unit', pre=True, always=True)
    def track_unit_explicit(cls, v):
        return v  # actual tracking done in calculate_quantity_lb

    @validator('quantity_lb', always=True, pre=True)
    def calculate_quantity_lb(cls, v, values):
        unit = values.get('unit')
        quantity = values.get('quantity')
        case_weight = values.get('case_weight_lb')

        # If quantity_lb is directly provided, use it (backward compatible)
        if v is not None:
            return v

        # Default unit to lb if not provided
        if unit is None:
            unit = 'lb'

        # If quantity + unit provided, calculate
        if quantity is not None:
            if unit == 'lb':
                return quantity
            elif unit in ('cases', 'bags', 'boxes'):
                if case_weight is None:
                    raise ValueError(f"case_weight_lb is required when unit is '{unit}'")
                return quantity * case_weight
            else:
                raise ValueError(f"Unknown unit: {unit}. Use 'lb', 'cases', 'bags', or 'boxes'")

        raise ValueError("Either quantity_lb or (quantity + unit) must be provided")

class OrderCreate(BaseModel):
    customer_name: str
    requested_ship_date: Optional[str] = None
    lines: List[OrderLineInput]
    notes: Optional[str] = None
    notes_es: Optional[str] = None

class OrderStatusUpdate(BaseModel):
    status: str

class AddOrderLines(BaseModel):
    lines: List[OrderLineInput]

class ShipOrderLineRequest(BaseModel):
    line_id: int
    quantity_lb: float

class ShipOrderRequest(BaseModel):
    ship_all: Optional[bool] = False
    lines: Optional[List[ShipOrderLineRequest]] = None


# ═══════════════════════════════════════════════════════════════
# SALES HELPER FUNCTIONS (v2.3.0)
# ═══════════════════════════════════════════════════════════════

def resolve_product_id(cur, product_name: str) -> tuple:
    """Find product by name (case-insensitive). Returns (product_id, product_name)."""
    cur.execute(
        "SELECT id, name FROM products WHERE LOWER(name) = LOWER(%s) AND COALESCE(active, true) = true",
        (product_name,)
    )
    row = cur.fetchone()
    if row:
        return row['id'], row['name']
    # Try fuzzy
    cur.execute(
        "SELECT id, name FROM products WHERE LOWER(name) LIKE LOWER(%s) AND COALESCE(active, true) = true ORDER BY name LIMIT 5",
        (f"%{product_name}%",)
    )
    rows = cur.fetchall()
    if len(rows) == 1:
        return rows[0]['id'], rows[0]['name']
    elif len(rows) > 1:
        suggestions = [r['name'] for r in rows]
        raise HTTPException(400, f"Multiple products match '{product_name}': {suggestions}")
    raise HTTPException(404, f"Product not found: '{product_name}'")


def resolve_customer_id(cur, customer_name: str, auto_create: bool = True) -> tuple:
    """Find or create customer by name. Returns (customer_id, customer_name)."""
    cur.execute(
        "SELECT id, name FROM customers WHERE LOWER(name) = LOWER(%s)",
        (customer_name,)
    )
    row = cur.fetchone()
    if row:
        return row['id'], row['name']
    # Try fuzzy
    cur.execute(
        "SELECT id, name FROM customers WHERE LOWER(name) LIKE LOWER(%s) AND active = true ORDER BY name LIMIT 5",
        (f"%{customer_name}%",)
    )
    rows = cur.fetchall()
    if len(rows) == 1:
        return rows[0]['id'], rows[0]['name']
    elif len(rows) > 1:
        suggestions = [r['name'] for r in rows]
        raise HTTPException(400, f"Multiple customers match '{customer_name}': {suggestions}")
    if auto_create:
        cur.execute(
            "INSERT INTO customers (name) VALUES (%s) RETURNING id, name",
            (customer_name,)
        )
        row = cur.fetchone()
        return row['id'], row['name']
    raise HTTPException(404, f"Customer not found: '{customer_name}'")


# ═══════════════════════════════════════════════════════════════
# DASHBOARD ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/dashboard/inventory")
def dashboard_inventory(_: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("SELECT * FROM inventory_summary WHERE on_hand > 0")
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Dashboard inventory failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/dashboard/low-stock")
def dashboard_low_stock(_: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("SELECT * FROM low_stock_alerts")
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Dashboard low-stock failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/dashboard/today")
def dashboard_today(_: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("SELECT * FROM todays_transactions")
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Dashboard today failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/dashboard/lots")
def dashboard_lots(_: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("SELECT * FROM lot_balances LIMIT 100")
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Dashboard lots failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/dashboard/production")
def dashboard_production(_: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("SELECT * FROM production_history LIMIT 50")
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Dashboard production failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# HEALTH & ROOT ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "name": "Factory Ledger System",
        "version": "2.2.0",
        "status": "online",
        "features": ["receive", "ship", "make", "adjust", "trace", "bom", "quick-create", "lot-reassign", "found-inventory", "ingredient-exclusion", "ingredient-lot-override", "dashboard", "sales-orders", "customers"]
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


# ═══════════════════════════════════════════════════════════════
# REVIEW QUEUE ENDPOINTS (MUST BE BEFORE /products/{product_id})
# ═══════════════════════════════════════════════════════════════

@app.get("/products/unverified")
def get_unverified_products(limit: int = Query(default=50, ge=1, le=200), _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT id, name, type, uom, 
                       COALESCE(verification_status, 'verified') as verification_status,
                       verification_notes, created_via
                FROM products
                WHERE COALESCE(verification_status, 'verified') = 'unverified'
                  AND COALESCE(active, true) = true
                ORDER BY id DESC
                LIMIT %s
            """, (limit,))
            products = cur.fetchall()
        return {"count": len(products), "products": products}
    except Exception as e:
        logger.error(f"Get unverified products failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/products/test-batches")
def get_test_batches(limit: int = Query(default=50, ge=1, le=200), _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT id, name, type, 
                       COALESCE(verification_status, 'verified') as verification_status,
                       COALESCE(production_context, 'standard') as production_context,
                       verification_notes
                FROM products
                WHERE COALESCE(production_context, 'standard') IN ('test_batch', 'sample', 'one_off')
                  AND COALESCE(active, true) = true
                ORDER BY id DESC
                LIMIT %s
            """, (limit,))
            products = cur.fetchall()
        return {"count": len(products), "products": products}
    except Exception as e:
        logger.error(f"Get test batches failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# PRODUCT BY ID (AFTER specific /products/* routes)
# ═══════════════════════════════════════════════════════════════

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
                
                cur.execute("""
                    INSERT INTO lots (product_id, lot_code, entry_source)
                    VALUES (%s, %s, 'received')
                    RETURNING id
                """, (product['id'], lot_code))
                lot_id = cur.fetchone()['id']
                
                cur.execute("""
                    INSERT INTO transactions (type, timestamp, bol_reference, shipper_name, shipper_code, cases_received, case_size_lb)
                    VALUES ('receive', %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (now, req.bol_reference, req.shipper_name, shipper_code, req.cases, req.case_size_lb))
                txn_id = cur.fetchone()['id']
                
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
            
            # Check for open sales orders from this customer
            open_orders_warning = None
            cur.execute("""
                SELECT so.order_number, so.status,
                       SUM(sol.quantity_lb - sol.quantity_shipped_lb) as remaining_lb
                FROM sales_orders so
                JOIN sales_order_lines sol ON sol.sales_order_id = so.id
                WHERE so.customer_id IN (
                    SELECT id FROM customers WHERE LOWER(name) LIKE LOWER(%s)
                )
                AND so.status NOT IN ('shipped', 'invoiced', 'cancelled')
                AND sol.line_status NOT IN ('fulfilled', 'cancelled')
                GROUP BY so.id, so.order_number, so.status
                HAVING SUM(sol.quantity_lb - sol.quantity_shipped_lb) > 0
            """, (f"%{req.customer_name}%",))
            open_orders = cur.fetchall()

            if open_orders:
                order_list = "; ".join([
                    f"{o['order_number']} ({o['status']}, {o['remaining_lb']:,.0f} lb remaining)"
                    for o in open_orders
                ])
                open_orders_warning = (
                    f"⚠️ WARNING: {req.customer_name} has open sales order(s): {order_list}. "
                    f"Consider using shipOrderPreview to ship against the order instead."
                )

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
                "open_orders_warning": open_orders_warning,
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
                
                if req.lot_code:
                    cur.execute("""
                        SELECT l.id, l.lot_code FROM lots l
                        WHERE l.product_id = %s AND LOWER(l.lot_code) = LOWER(%s)
                    """, (product['id'], req.lot_code))
                    lot_row = cur.fetchone()
                    if not lot_row:
                        raise HTTPException(status_code=404, detail=f"Lot '{req.lot_code}' not found")
                    
                    cur.execute("SELECT id FROM lots WHERE id = %s FOR UPDATE", (lot_row['id'],))
                    
                    cur.execute("""
                        SELECT COALESCE(SUM(quantity_lb), 0) as available
                        FROM transaction_lines WHERE lot_id = %s
                    """, (lot_row['id'],))
                    available = float(cur.fetchone()['available'])
                    lot = {"id": lot_row['id'], "lot_code": lot_row['lot_code'], "available": available}
                else:
                    cur.execute("""
                        SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                        FROM lots l
                        LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                        WHERE l.product_id = %s
                        GROUP BY l.id
                        HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                        ORDER BY l.id ASC
                    """, (product['id'],))
                    candidates = cur.fetchall()
                    
                    if not candidates:
                        raise HTTPException(status_code=400, detail=f"No inventory available for {product['name']}")
                    
                    cur.execute("SELECT id FROM lots WHERE id = %s FOR UPDATE", (candidates[0]['id'],))
                    
                    cur.execute("""
                        SELECT COALESCE(SUM(quantity_lb), 0) as available
                        FROM transaction_lines WHERE lot_id = %s
                    """, (candidates[0]['id'],))
                    available = float(cur.fetchone()['available'])
                    lot = {"id": candidates[0]['id'], "lot_code": candidates[0]['lot_code'], "available": available}
                
                if lot['available'] < req.quantity_lb:
                    raise HTTPException(status_code=400, detail="Insufficient inventory")
                
                now = get_plant_now()
                
                cur.execute("""
                    INSERT INTO transactions (type, timestamp, customer_name, order_reference)
                    VALUES ('ship', %s, %s, %s)
                    RETURNING id
                """, (now, req.customer_name, req.order_reference))
                txn_id = cur.fetchone()['id']
                
                cur.execute("""
                    INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                    VALUES (%s, %s, %s, %s)
                """, (txn_id, product['id'], lot['id'], -req.quantity_lb))
                
                remaining = lot['available'] - req.quantity_lb
                
                # Check for open sales orders from this customer
                open_orders_warning = None
                cur.execute("""
                    SELECT so.order_number, so.status,
                           SUM(sol.quantity_lb - sol.quantity_shipped_lb) as remaining_lb
                    FROM sales_orders so
                    JOIN sales_order_lines sol ON sol.sales_order_id = so.id
                    WHERE so.customer_id IN (
                        SELECT id FROM customers WHERE LOWER(name) LIKE LOWER(%s)
                    )
                    AND so.status NOT IN ('shipped', 'invoiced', 'cancelled')
                    AND sol.line_status NOT IN ('fulfilled', 'cancelled')
                    GROUP BY so.id, so.order_number, so.status
                    HAVING SUM(sol.quantity_lb - sol.quantity_shipped_lb) > 0
                """, (f"%{req.customer_name}%",))
                open_orders = cur.fetchall()

                if open_orders:
                    order_list = "; ".join([
                        f"{o['order_number']} ({o['status']}, {o['remaining_lb']:,.0f} lb remaining)"
                        for o in open_orders
                    ])
                    open_orders_warning = (
                        f"⚠️ WARNING: {req.customer_name} has open sales order(s): {order_list}. "
                        f"This shipment was standalone and NOT applied to the order."
                    )

                logger.info(f"Ship committed: {req.quantity_lb} lb of {product['name']} to {req.customer_name}")

                return {
                    "success": True,
                    "transaction_id": txn_id,
                    "lot_code": lot['lot_code'],
                    "quantity_shipped": req.quantity_lb,
                    "remaining_in_lot": remaining,
                    "open_orders_warning": open_orders_warning,
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
                
                cur.execute("""
                    SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                    FROM lots l
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.product_id = %s
                    GROUP BY l.id
                    HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                    ORDER BY l.id ASC
                """, (product['id'],))
                candidate_lots = cur.fetchall()
                
                if not candidate_lots:
                    raise HTTPException(status_code=400, detail=f"No inventory available for {product['name']}")
                
                lot_ids = [lot['id'] for lot in candidate_lots]
                cur.execute(
                    "SELECT id FROM lots WHERE id = ANY(%s) ORDER BY id ASC FOR UPDATE",
                    (lot_ids,)
                )
                
                cur.execute("""
                    SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                    FROM lots l
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.id = ANY(%s)
                    GROUP BY l.id
                    HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                    ORDER BY l.id ASC
                """, (lot_ids,))
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
            
            excluded_ids = set(req.excluded_ingredients or [])
            
            cur.execute("""
                SELECT bf.ingredient_product_id, p.name as ingredient_name, bf.quantity_lb
                FROM batch_formulas bf
                JOIN products p ON p.id = bf.ingredient_product_id
                WHERE bf.product_id = %s
            """, (product['id'],))
            formula = cur.fetchall()
            
            ingredients_needed = []
            excluded_ingredients = []
            lot_overrides_applied = []
            
            lot_overrides = req.get_lot_overrides()
            
            for ing in formula:
                ing_id = ing['ingredient_product_id']
                needed = float(ing['quantity_lb']) * req.batches
                
                if ing_id in excluded_ids:
                    excluded_ingredients.append({
                        "ingredient_id": ing_id,
                        "ingredient_name": ing['ingredient_name'],
                        "would_need_lb": needed,
                        "excluded": True
                    })
                    continue
                
                if lot_overrides and str(ing_id) in lot_overrides:
                    override_code = lot_overrides[str(ing_id)]
                    cur.execute("""
                        SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                        FROM lots l
                        LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                        WHERE l.product_id = %s AND LOWER(l.lot_code) = LOWER(%s)
                        GROUP BY l.id
                    """, (ing_id, override_code))
                    override_lot = cur.fetchone()
                    
                    if not override_lot:
                        ingredients_needed.append({
                            "ingredient_id": ing_id,
                            "ingredient_name": ing['ingredient_name'],
                            "needed_lb": needed,
                            "available_lb": 0,
                            "sufficient": False,
                            "override_lot": override_code,
                            "override_error": f"Lot '{override_code}' not found for this ingredient"
                        })
                        continue
                    
                    avail = float(override_lot['available'])
                    lot_overrides_applied.append({
                        "ingredient_id": ing_id,
                        "ingredient_name": ing['ingredient_name'],
                        "lot_code": override_lot['lot_code'],
                        "needed_lb": needed,
                        "available_lb": avail,
                        "sufficient": avail >= needed
                    })
                    ingredients_needed.append({
                        "ingredient_id": ing_id,
                        "ingredient_name": ing['ingredient_name'],
                        "needed_lb": needed,
                        "available_lb": avail,
                        "sufficient": avail >= needed,
                        "override_lot": override_lot['lot_code']
                    })
                else:
                    cur.execute("""
                        SELECT COALESCE(SUM(tl.quantity_lb), 0) as available
                        FROM lots l
                        LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                        WHERE l.product_id = %s
                    """, (ing_id,))
                    avail = float(cur.fetchone()['available'])
                    ingredients_needed.append({
                        "ingredient_id": ing_id,
                        "ingredient_name": ing['ingredient_name'],
                        "needed_lb": needed,
                        "available_lb": avail,
                        "sufficient": avail >= needed
                    })
            
            all_sufficient = all(i['sufficient'] for i in ingredients_needed)
            
            if req.lot_code:
                lot_code = req.lot_code
            else:
                now = get_plant_now()
                date_part = now.strftime("%y-%m%d")
                cur.execute("""
                    SELECT lot_code FROM lots 
                    WHERE lot_code LIKE %s 
                    ORDER BY lot_code DESC LIMIT 1
                """, (f"B{date_part}-%",))
                existing = cur.fetchone()
                if existing:
                    try:
                        last_seq = int(existing['lot_code'].split('-')[-1])
                        seq = last_seq + 1
                    except (ValueError, IndexError):
                        seq = 1
                else:
                    seq = 1
                lot_code = f"B{date_part}-{seq:03d}"
            
            response = {
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
            
            if lot_overrides_applied:
                response["lot_overrides"] = lot_overrides_applied
                response["preview_message"] += f" (with {len(lot_overrides_applied)} lot override(s))"
            
            if excluded_ingredients:
                response["excluded_ingredients"] = excluded_ingredients
                response["preview_message"] += f" (excluding {len(excluded_ingredients)} ingredient(s))"
            
            return response
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
                
                excluded_ids = set(req.excluded_ingredients or [])
                
                if req.lot_code:
                    lot_code = req.lot_code
                else:
                    date_part = now.strftime("%y-%m%d")
                    cur.execute("""
                        SELECT lot_code FROM lots 
                        WHERE lot_code LIKE %s 
                        ORDER BY lot_code DESC LIMIT 1
                    """, (f"B{date_part}-%",))
                    existing = cur.fetchone()
                    if existing:
                        try:
                            last_seq = int(existing['lot_code'].split('-')[-1])
                            seq = last_seq + 1
                        except (ValueError, IndexError):
                            seq = 1
                    else:
                        seq = 1
                    lot_code = f"B{date_part}-{seq:03d}"
                
                cur.execute("""
                    INSERT INTO lots (product_id, lot_code, entry_source)
                    VALUES (%s, %s, 'production_output')
                    RETURNING id
                """, (product['id'], lot_code))
                output_lot_id = cur.fetchone()['id']
                
                exclusion_note = ""
                if excluded_ids:
                    exclusion_note = f" (excluded ingredient IDs: {sorted(excluded_ids)})"
                
                cur.execute("""
                    INSERT INTO transactions (type, timestamp, notes)
                    VALUES ('make', %s, %s)
                    RETURNING id
                """, (now, f"{req.batches} batch(es) of {product['name']}{exclusion_note}"))
                txn_id = cur.fetchone()['id']
                
                cur.execute("""
                    INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                    VALUES (%s, %s, %s, %s)
                """, (txn_id, product['id'], output_lot_id, total_output))
                
                cur.execute("""
                    SELECT bf.ingredient_product_id, bf.quantity_lb
                    FROM batch_formulas bf
                    WHERE bf.product_id = %s
                """, (product['id'],))
                formula = cur.fetchall()
                
                consumed = []
                excluded_from_run = []
                
                lot_overrides = req.get_lot_overrides()
                
                for ing in formula:
                    ing_id = ing['ingredient_product_id']
                    needed = float(ing['quantity_lb']) * req.batches
                    
                    if ing_id in excluded_ids:
                        cur.execute("SELECT name FROM products WHERE id = %s", (ing_id,))
                        ing_name = cur.fetchone()
                        excluded_from_run.append({
                            "ingredient_id": ing_id,
                            "ingredient_name": ing_name['name'] if ing_name else f"ID {ing_id}",
                            "skipped_lb": needed
                        })
                        continue
                    
                    override_lot = None
                    if lot_overrides and str(ing_id) in lot_overrides:
                        override_code = lot_overrides[str(ing_id)]
                        cur.execute("""
                            SELECT l.id, l.lot_code FROM lots l
                            WHERE l.product_id = %s AND LOWER(l.lot_code) = LOWER(%s)
                        """, (ing_id, override_code))
                        override_lot = cur.fetchone()
                    
                    if override_lot:
                        cur.execute("SELECT id FROM lots WHERE id = %s FOR UPDATE", (override_lot['id'],))
                        
                        cur.execute("""
                            SELECT COALESCE(SUM(tl.quantity_lb), 0) as available
                            FROM transaction_lines tl WHERE tl.lot_id = %s
                        """, (override_lot['id'],))
                        available = float(cur.fetchone()['available'])
                        
                        if available < needed:
                            raise HTTPException(
                                status_code=400, 
                                detail=f"Override lot {override_lot['lot_code']} has {available} lb, need {needed} lb"
                            )
                        
                        cur.execute("""
                            INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                            VALUES (%s, %s, %s, %s)
                        """, (txn_id, ing_id, override_lot['id'], -needed))
                        
                        cur.execute("""
                            INSERT INTO ingredient_lot_consumption (transaction_id, ingredient_product_id, ingredient_lot_id, quantity_lb)
                            VALUES (%s, %s, %s, %s)
                        """, (txn_id, ing_id, override_lot['id'], needed))
                        
                        consumed.append({"lot_code": override_lot['lot_code'], "consumed_lb": needed, "override": True})
                    else:
                        cur.execute("""
                            SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                            FROM lots l
                            LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                            WHERE l.product_id = %s
                            GROUP BY l.id
                            HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                            ORDER BY l.id ASC
                        """, (ing_id,))
                        candidate_lots = cur.fetchall()
                        
                        if not candidate_lots:
                            raise HTTPException(
                                status_code=400,
                                detail=f"No inventory available for ingredient ID {ing_id}"
                            )
                        
                        lot_ids = [lot['id'] for lot in candidate_lots]
                        cur.execute(
                            "SELECT id FROM lots WHERE id = ANY(%s) ORDER BY id ASC FOR UPDATE",
                            (lot_ids,)
                        )
                        
                        cur.execute("""
                            SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                            FROM lots l
                            LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                            WHERE l.id = ANY(%s)
                            GROUP BY l.id
                            HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                            ORDER BY l.id ASC
                        """, (lot_ids,))
                        lots = cur.fetchall()
                        
                        remaining = needed
                        for lot in lots:
                            if remaining <= 0:
                                break
                            take = min(float(lot['available']), remaining)
                            cur.execute("""
                                INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                                VALUES (%s, %s, %s, %s)
                            """, (txn_id, ing_id, lot['id'], -take))
                            
                            cur.execute("""
                                INSERT INTO ingredient_lot_consumption (transaction_id, ingredient_product_id, ingredient_lot_id, quantity_lb)
                                VALUES (%s, %s, %s, %s)
                            """, (txn_id, ing_id, lot['id'], take))
                            
                            consumed.append({"lot_code": lot['lot_code'], "consumed_lb": take})
                            remaining -= take
                        
                        if remaining > 0.001:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Insufficient inventory for ingredient ID {ing_id}. Missing {remaining:.2f} lb"
                            )
                
                logger.info(f"Make committed: {lot_code} - {total_output} lb of {product['name']}")
                
                response = {
                    "success": True,
                    "transaction_id": txn_id,
                    "lot_id": output_lot_id,
                    "lot_code": lot_code,
                    "output_lb": total_output,
                    "ingredients_consumed": consumed,
                    "message": f"Produced {total_output} lb as lot {lot_code}"
                }
                
                if excluded_from_run:
                    response["excluded_ingredients"] = excluded_from_run
                    response["message"] += f" (excluded {len(excluded_from_run)} ingredient(s))"
                
                return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Make commit failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# ADJUST ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/adjust/preview")
def adjust_preview(req: AdjustRequest, _: bool = Depends(verify_api_key)):
    validate_bilingual(req.reason, req.reason_es, "reason")
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT p.id as product_id, p.name, p.odoo_code,
                       COALESCE(p.label_type, 'house') as label_type,
                       l.id as lot_id, l.lot_code,
                       COALESCE(SUM(tl.quantity_lb), 0) as quantity_on_hand
                FROM products p
                JOIN lots l ON l.product_id = p.id
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE (LOWER(p.name) LIKE LOWER(%s) OR LOWER(p.odoo_code) LIKE LOWER(%s))
                  AND LOWER(l.lot_code) = LOWER(%s)
                GROUP BY p.id, l.id
            """, (f"%{req.product_name}%", f"%{req.product_name}%", req.lot_code))
            result = cur.fetchone()

            if not result:
                return JSONResponse(status_code=404, content={
                    "error": f"Product/lot combination not found for '{req.product_name}' / '{req.lot_code}'"
                })

            # SKU protection check
            warning = check_private_label_merge(
                result['name'], result['label_type'], req.reason, req.adjustment_lb
            )
            if warning:
                return JSONResponse(status_code=403, content={
                    "blocked": True,
                    "warning": warning,
                    "product_name": result['name'],
                    "label_type": result['label_type']
                })

            quantity_on_hand = float(result['quantity_on_hand'])
            new_balance = quantity_on_hand + req.adjustment_lb

            response = {
                "product_id": result['product_id'],
                "product_name": result['name'],
                "odoo_code": result['odoo_code'],
                "label_type": result['label_type'],
                "lot_code": result['lot_code'],
                "current_quantity_lb": quantity_on_hand,
                "adjustment_lb": req.adjustment_lb,
                "new_balance_lb": new_balance,
                "reason": req.reason,
                "preview_message": f"Will adjust {result['name']} lot {result['lot_code']} by {req.adjustment_lb} lb ({quantity_on_hand} → {new_balance} lb)"
            }
            if req.reason_es:
                response["reason_es"] = req.reason_es
            if new_balance < 0:
                response["balance_warning"] = f"This will result in negative inventory ({new_balance} lb)"
            return response
    except Exception as e:
        logger.error(f"Adjust preview failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/adjust/commit")
def adjust_commit(req: AdjustRequest, _: bool = Depends(verify_api_key)):
    validate_bilingual(req.reason, req.reason_es, "reason")
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT p.id as product_id, p.name,
                           COALESCE(p.label_type, 'house') as label_type,
                           l.id as lot_id, l.lot_code
                    FROM products p
                    JOIN lots l ON l.product_id = p.id
                    WHERE (LOWER(p.name) LIKE LOWER(%s) OR LOWER(p.odoo_code) LIKE LOWER(%s))
                      AND LOWER(l.lot_code) = LOWER(%s)
                """, (f"%{req.product_name}%", f"%{req.product_name}%", req.lot_code))
                result = cur.fetchone()

                if not result:
                    raise HTTPException(status_code=404, detail=f"Product/lot combination not found")

                # SKU protection check
                warning = check_private_label_merge(
                    result['name'], result['label_type'], req.reason, req.adjustment_lb
                )
                if warning:
                    return JSONResponse(status_code=403, content={
                        "blocked": True,
                        "warning": warning,
                        "product_name": result['name'],
                        "label_type": result['label_type']
                    })

                now = get_plant_now()

                cur.execute("""
                    INSERT INTO transactions (type, timestamp, adjust_reason, adjust_reason_es, notes)
                    VALUES ('adjust', %s, %s, %s, %s)
                    RETURNING id
                """, (now, req.reason, req.reason_es, f"Adjustment: {req.adjustment_lb} lb"))
                txn_id = cur.fetchone()['id']

                cur.execute("""
                    INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                    VALUES (%s, %s, %s, %s)
                """, (txn_id, result['product_id'], result['lot_id'], req.adjustment_lb))

                logger.info(f"Adjust committed: {req.adjustment_lb} lb to lot {result['lot_code']}")

                response = {
                    "success": True,
                    "transaction_id": txn_id,
                    "lot_code": result['lot_code'],
                    "adjustment_lb": req.adjustment_lb,
                    "reason": req.reason,
                    "message": f"Adjusted lot {result['lot_code']} by {req.adjustment_lb} lb"
                }
                if req.reason_es:
                    response["reason_es"] = req.reason_es
                return response
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
            
            cur.execute("""
                SELECT p.name as ingredient_name, l.lot_code as ingredient_lot, ilc.quantity_lb as quantity_consumed
                FROM ingredient_lot_consumption ilc
                JOIN products p ON p.id = ilc.ingredient_product_id
                JOIN lots l ON l.id = ilc.ingredient_lot_id
                WHERE ilc.transaction_id = %s
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
                SELECT DISTINCT bl.lot_code as batch_lot, bp.name as batch_product, ilc.quantity_lb as quantity_consumed
                FROM ingredient_lot_consumption ilc
                JOIN lots il ON il.id = ilc.ingredient_lot_id
                JOIN transactions t ON t.id = ilc.transaction_id
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
    validate_bilingual(req.notes, req.notes_es, "notes")
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

                verification_notes = f"Quick-created. Name confidence: {req.name_confidence}."
                if req.notes:
                    verification_notes += f" {req.notes}"
                verification_notes_es = req.notes_es

                cur.execute("""
                    INSERT INTO products (name, type, uom, storage_type, verification_status, verification_notes, verification_notes_es, created_via, active)
                    VALUES (%s, %s, %s, %s, 'unverified', %s, %s, 'quick_create', true)
                    RETURNING id, name, type, uom, verification_status
                """, (req.product_name, req.product_type, req.uom, req.storage_type, verification_notes, verification_notes_es))
                product = cur.fetchone()

                try:
                    cur.execute("""
                        INSERT INTO product_verification_history (product_id, from_status, to_status, action, action_notes, action_notes_es, performed_by)
                        VALUES (%s, NULL, 'unverified', 'created', %s, %s, %s)
                    """, (product['id'], f"Quick-created during receive. {verification_notes}", verification_notes_es, req.performed_by))
                except Exception:
                    pass

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
    validate_bilingual(req.notes, req.notes_es, "notes")
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
                verification_notes_es = req.notes_es

                cur.execute("""
                    INSERT INTO products (name, type, uom, verification_status, verification_notes, verification_notes_es, production_context, created_via, active)
                    VALUES (%s, 'batch', 'lb', 'unverified', %s, %s, %s, 'quick_create_batch', true)
                    RETURNING id, name, type, verification_status
                """, (req.product_name, verification_notes, verification_notes_es, req.production_context))
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
    validate_bilingual(req.reason_notes, req.reason_notes_es, "reason_notes")
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT l.id, l.lot_code, l.product_id, p.name as product_name,
                           COALESCE(p.label_type, 'house') as label_type,
                           COALESCE(SUM(tl.quantity_lb), 0) as quantity_on_hand
                    FROM lots l
                    JOIN products p ON p.id = l.product_id
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.id = %s
                    GROUP BY l.id, p.id
                    FOR UPDATE OF l
                """, (lot_id,))
                lot = cur.fetchone()

                if not lot:
                    raise HTTPException(status_code=404, detail=f"Lot ID {lot_id} not found")

                if lot['product_id'] == req.to_product_id:
                    return JSONResponse(status_code=400, content={
                        "error": f"Lot is already assigned to {lot['product_name']}"
                    })

                cur.execute("""
                    SELECT id, name, COALESCE(label_type, 'house') as label_type
                    FROM products WHERE id = %s
                """, (req.to_product_id,))
                to_product = cur.fetchone()

                if not to_product:
                    raise HTTPException(status_code=404, detail=f"Target product ID {req.to_product_id} not found")

                # SKU protection: block product_merge if either side is private-label
                if req.reason_code == 'product_merge':
                    if lot['label_type'] == 'private_label':
                        return JSONResponse(status_code=403, content={
                            "blocked": True,
                            "warning": (
                                f"BLOCKED: Cannot merge lots from private-label SKU '{lot['product_name']}'. "
                                f"Private-label products are identity-protected and cannot be merged or consolidated. "
                                f"If this is a correction, use reason_code 'incorrect_receive' or 'data_entry_error' instead."
                            ),
                            "source_product": lot['product_name'],
                            "source_label_type": lot['label_type']
                        })
                    if to_product['label_type'] == 'private_label':
                        return JSONResponse(status_code=403, content={
                            "blocked": True,
                            "warning": (
                                f"BLOCKED: Cannot merge lots into private-label SKU '{to_product['name']}'. "
                                f"Private-label products are identity-protected and cannot be merged or consolidated. "
                                f"If this is a correction, use reason_code 'incorrect_receive' or 'data_entry_error' instead."
                            ),
                            "target_product": to_product['name'],
                            "target_label_type": to_product['label_type']
                        })
                
                cur.execute("""
                    SELECT COUNT(*) as count FROM ingredient_lot_consumption WHERE ingredient_lot_id = %s
                """, (lot_id,))
                usage = cur.fetchone()
                
                cur.execute("UPDATE lots SET product_id = %s WHERE id = %s", (req.to_product_id, lot_id))
                
                cur.execute("""
                    UPDATE transaction_lines SET product_id = %s WHERE lot_id = %s
                """, (req.to_product_id, lot_id))
                
                try:
                    cur.execute("""
                        INSERT INTO lot_reassignments
                        (lot_id, lot_code, from_product_id, from_product_name, to_product_id, to_product_name,
                         quantity_affected, uom, reason_code, reason_notes, reason_notes_es, reassigned_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (lot_id, lot['lot_code'], lot['product_id'], lot['product_name'],
                          req.to_product_id, to_product['name'], float(lot['quantity_on_hand']), 'lb',
                          req.reason_code, req.reason_notes, req.reason_notes_es, req.performed_by))
                except Exception as e:
                    logger.warning(f"Failed to record lot reassignment history: {e}")

                logger.info(f"Reassigned lot {lot['lot_code']} from {lot['product_name']} to {to_product['name']}")

                response = {
                    "success": True,
                    "lot_id": lot_id,
                    "lot_code": lot['lot_code'],
                    "from_product": lot['product_name'],
                    "to_product": to_product['name'],
                    "reason_code": req.reason_code,
                    "production_usage_updated": usage['count'] if usage else 0,
                    "message": f"Reassigned lot {lot['lot_code']} to {to_product['name']}"
                }
                if req.reason_notes:
                    response["reason_notes"] = req.reason_notes
                if req.reason_notes_es:
                    response["reason_notes_es"] = req.reason_notes_es
                return response
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
    validate_bilingual(req.notes, req.notes_es, "notes")
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id, name FROM products WHERE id = %s", (req.product_id,))
                product = cur.fetchone()
                
                if not product:
                    raise HTTPException(status_code=404, detail=f"Product ID {req.product_id} not found")
                
                cur.execute("SELECT pg_advisory_xact_lock(2)")
                
                now = get_plant_now()
                date_part = now.strftime("%y-%m-%d")
                
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
                
                cur.execute("""
                    INSERT INTO lots (product_id, lot_code, entry_source, entry_source_notes, entry_source_notes_es, found_location, estimated_age)
                    VALUES (%s, %s, 'found_inventory', %s, %s, %s, %s)
                    RETURNING id
                """, (req.product_id, lot_code, req.notes, req.notes_es, req.found_location, req.estimated_age))
                lot_id = cur.fetchone()['id']

                cur.execute("""
                    INSERT INTO transactions (type, timestamp, notes)
                    VALUES ('adjust', %s, %s)
                    RETURNING id
                """, (now, f"Found inventory: {req.reason_code}"))
                txn_id = cur.fetchone()['id']

                cur.execute("""
                    INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                    VALUES (%s, %s, %s, %s)
                """, (txn_id, req.product_id, lot_id, req.quantity))

                try:
                    cur.execute("""
                        INSERT INTO inventory_adjustments
                        (lot_id, product_id, adjustment_type, quantity_before, quantity_adjustment, quantity_after,
                         uom, reason_code, reason_notes, reason_notes_es, found_location, estimated_age, suspected_supplier, adjusted_by)
                        VALUES (%s, %s, 'found', 0, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (lot_id, req.product_id, req.quantity, req.quantity, req.uom,
                          req.reason_code, req.notes, req.notes_es, req.found_location, req.estimated_age,
                          req.suspected_supplier, req.performed_by))
                except Exception as e:
                    logger.warning(f"Failed to record inventory adjustment: {e}")
                
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
    validate_bilingual(req.notes, req.notes_es, "notes")
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT id, name FROM products WHERE LOWER(name) = LOWER(%s)", (req.product_name,))
                existing = cur.fetchone()
                if existing:
                    return JSONResponse(status_code=409, content={
                        "error": f"Product '{req.product_name}' already exists",
                        "existing_product_id": existing['id'],
                        "suggestion": "Use /inventory/found with the existing product_id"
                    })
                
                verification_notes = f"Quick-created during inventory count. {req.notes or ''}"
                verification_notes_es = req.notes_es
                cur.execute("""
                    INSERT INTO products (name, type, uom, storage_type, verification_status, verification_notes, verification_notes_es, created_via, active)
                    VALUES (%s, %s, %s, %s, 'unverified', %s, %s, 'quick_create_found_inventory', true)
                    RETURNING id, name
                """, (req.product_name, req.product_type, req.uom, req.storage_type, verification_notes, verification_notes_es))
                product = cur.fetchone()
                
                cur.execute("SELECT pg_advisory_xact_lock(2)")
                
                now = get_plant_now()
                date_part = now.strftime("%y-%m-%d")
                
                cur.execute("SELECT lot_code FROM lots WHERE lot_code LIKE %s ORDER BY lot_code DESC LIMIT 1", (f"{date_part}-FOUND-%",))
                existing_lot = cur.fetchone()
                seq = (int(existing_lot['lot_code'].split('-')[-1]) + 1) if existing_lot else 1
                lot_code = f"{date_part}-FOUND-{seq:03d}"
                
                cur.execute("""
                    INSERT INTO lots (product_id, lot_code, entry_source, entry_source_notes, entry_source_notes_es, found_location, estimated_age)
                    VALUES (%s, %s, 'found_inventory', %s, %s, %s, %s)
                    RETURNING id
                """, (product['id'], lot_code, req.notes, req.notes_es, req.found_location, req.estimated_age))
                lot_id = cur.fetchone()['id']
                
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


@app.get("/inventory/found/queue")
def get_found_inventory_queue(limit: int = Query(default=50, ge=1, le=200), _: bool = Depends(verify_api_key)):
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
    validate_bilingual(req.notes, req.notes_es, "notes")
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
                        UPDATE products SET verification_status = %s, name = %s, verification_notes = %s, verification_notes_es = %s
                        WHERE id = %s
                    """, (new_status, new_name, req.notes, req.notes_es, product_id))

                elif req.action == 'reject':
                    new_status = 'rejected'
                    cur.execute("""
                        UPDATE products SET verification_status = %s, active = false, verification_notes = %s, verification_notes_es = %s
                        WHERE id = %s
                    """, (new_status, req.notes, req.notes_es, product_id))

                elif req.action == 'archive':
                    new_status = 'archived'
                    cur.execute("""
                        UPDATE products SET verification_status = %s, active = false, verification_notes = %s, verification_notes_es = %s
                        WHERE id = %s
                    """, (new_status, req.notes, req.notes_es, product_id))
                else:
                    raise HTTPException(status_code=400, detail=f"Invalid action: {req.action}")

                try:
                    cur.execute("""
                        INSERT INTO product_verification_history (product_id, from_status, to_status, action, action_notes, action_notes_es, performed_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (product_id, old_status, new_status, req.action, req.notes, req.notes_es, req.performed_by))
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
                WHERE bf.product_id = %s
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


# ═══════════════════════════════════════════════════════════════
# CUSTOMER ENDPOINTS (v2.3.0)
# ═══════════════════════════════════════════════════════════════

@app.get("/customers")
def list_customers(active_only: bool = True, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            if active_only:
                cur.execute("SELECT id, name, contact_name, email, phone, active FROM customers WHERE active = true ORDER BY name")
            else:
                cur.execute("SELECT id, name, contact_name, email, phone, active FROM customers ORDER BY name")
            return {"customers": cur.fetchall()}
    except Exception as e:
        logger.error(f"List customers failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/customers/search")
def search_customers(q: str = Query(..., min_length=1), _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute(
                "SELECT id, name, contact_name, phone, email FROM customers WHERE LOWER(name) LIKE LOWER(%s) AND active = true ORDER BY name",
                (f"%{q}%",)
            )
            rows = cur.fetchall()
        return {"results": rows}
    except Exception as e:
        logger.error(f"Search customers failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/customers")
def create_customer(req: CustomerCreate, _: bool = Depends(verify_api_key)):
    validate_bilingual(req.notes, req.notes_es, "notes")
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """INSERT INTO customers (name, contact_name, email, phone, address, notes, notes_es)
                       VALUES (%s, %s, %s, %s, %s, %s, %s) RETURNING id, name""",
                    (req.name, req.contact_name, req.email, req.phone, req.address, req.notes, req.notes_es)
                )
                row = cur.fetchone()
                logger.info(f"Created customer: {row['name']} (ID: {row['id']})")
                return {"customer_id": row['id'], "name": row['name'], "message": f"Customer '{row['name']}' created"}
    except Exception as e:
        if "unique" in str(e).lower():
            raise HTTPException(409, f"Customer '{req.name}' already exists")
        logger.error(f"Create customer failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.patch("/customers/{customer_id}")
def update_customer(customer_id: int, req: CustomerUpdate, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                updates = req.dict(exclude_none=True)
                if not updates:
                    raise HTTPException(400, "No fields to update")
                set_clause = ", ".join(f"{k} = %s" for k in updates)
                values = list(updates.values()) + [customer_id]
                cur.execute(
                    f"UPDATE customers SET {set_clause} WHERE id = %s RETURNING id, name",
                    values
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(404, "Customer not found")
                return {"customer_id": row['id'], "name": row['name'], "message": "Customer updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update customer failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# SALES ORDER ENDPOINTS (v2.3.0)
# ═══════════════════════════════════════════════════════════════

# Status state machine — all valid transitions (includes auto-transitions from shipOrderCommit)
VALID_TRANSITIONS = {
    'new':            ['confirmed', 'cancelled'],
    'confirmed':      ['in_production', 'cancelled'],
    'in_production':  ['ready', 'cancelled'],
    'ready':          ['shipped', 'partial_ship', 'cancelled'],
    'partial_ship':   ['shipped', 'cancelled'],
    'shipped':        ['invoiced'],
    'invoiced':       [],   # terminal
    'cancelled':      [],   # terminal
}

# Manual transitions — subset excluding shipped/partial_ship (those are auto-only via shipOrderCommit)
MANUAL_TRANSITIONS = {
    'new':            ['confirmed', 'cancelled'],
    'confirmed':      ['in_production', 'cancelled'],
    'in_production':  ['ready', 'cancelled'],
    'ready':          ['cancelled'],
    'partial_ship':   ['cancelled'],
    'shipped':        ['invoiced'],
    'invoiced':       [],
    'cancelled':      [],
}

@app.post("/sales/orders")
def create_sales_order(req: OrderCreate, _: bool = Depends(verify_api_key)):
    validate_bilingual(req.notes, req.notes_es, "notes")
    for line in req.lines:
        validate_bilingual(line.notes, line.notes_es, "notes")
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                customer_id, customer_name = resolve_customer_id(cur, req.customer_name)

                cur.execute(
                    """INSERT INTO sales_orders (customer_id, requested_ship_date, notes, notes_es, order_number)
                       VALUES (%s, %s, %s, %s, '')
                       RETURNING id, order_number""",
                    (customer_id, req.requested_ship_date, req.notes, req.notes_es)
                )
                row = cur.fetchone()
                order_id, order_number = row['id'], row['order_number']

                line_results = []
                total_lb = 0
                warnings = []
                for line in req.lines:
                    product_id, prod_name = resolve_product_id(cur, line.product_name)

                    # Fix #2: Auto-lookup case weight from product if not provided
                    effective_case_weight = line.case_weight_lb
                    used_unit = line.unit or 'lb'
                    if used_unit in ('cases', 'bags', 'boxes') and effective_case_weight is None:
                        cur.execute(
                            "SELECT default_case_weight_lb FROM products WHERE id = %s",
                            (product_id,)
                        )
                        prod_row = cur.fetchone()
                        if prod_row and prod_row.get('default_case_weight_lb'):
                            effective_case_weight = float(prod_row['default_case_weight_lb'])
                        else:
                            raise HTTPException(400,
                                f"case_weight_lb is required for '{prod_name}' when ordering in {used_unit}. "
                                f"No default case weight is set for this product."
                            )
                        # Recalculate quantity_lb with looked-up weight
                        line.quantity_lb = line.quantity * effective_case_weight

                    # Fix #1: Warn if unit was not explicitly provided and quantity was given
                    if line.quantity is not None and line.unit is None:
                        warnings.append(
                            f"⚠️ '{prod_name}': No unit specified for quantity {line.quantity:,.0f} — "
                            f"defaulting to lb. Did you mean cases?"
                        )

                    cur.execute(
                        """INSERT INTO sales_order_lines (sales_order_id, product_id, quantity_lb, unit_price, notes, notes_es)
                           VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                        (order_id, product_id, line.quantity_lb, line.unit_price, line.notes, line.notes_es)
                    )
                    line_id = cur.fetchone()['id']
                    total_lb += line.quantity_lb

                    # Fix #3: Quantity sanity check — compare to customer's average order size
                    cur.execute("""
                        SELECT AVG(sol.quantity_lb) as avg_qty
                        FROM sales_order_lines sol
                        JOIN sales_orders so ON so.id = sol.sales_order_id
                        WHERE so.customer_id = %s AND sol.product_id = %s
                          AND sol.id != %s
                          AND sol.line_status != 'cancelled'
                    """, (customer_id, product_id, line_id))
                    avg_row = cur.fetchone()
                    if avg_row and avg_row['avg_qty'] and line.quantity_lb < float(avg_row['avg_qty']) * 0.25:
                        warnings.append(
                            f"⚠️ '{prod_name}': {line.quantity_lb:,.0f} lb is unusually low for {customer_name}. "
                            f"Their average order is {float(avg_row['avg_qty']):,.0f} lb. Double-check the quantity."
                        )

                    line_results.append({
                        "line_id": line_id,
                        "product": prod_name,
                        "quantity_lb": line.quantity_lb,
                        "original_quantity": line.quantity,
                        "original_unit": used_unit,
                        "case_weight_lb": effective_case_weight,
                        "unit_price": line.unit_price
                    })

                logger.info(f"Created sales order {order_number} for {customer_name} with {len(line_results)} lines")
                return {
                    "order_id": order_id,
                    "order_number": order_number,
                    "customer": customer_name,
                    "requested_ship_date": req.requested_ship_date,
                    "status": "new",
                    "total_lb": total_lb,
                    "lines": line_results,
                    "warnings": warnings if warnings else None,
                    "message": f"Order {order_number} created with {len(line_results)} line(s)"
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create sales order failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/sales/orders")
def list_sales_orders(
    status: Optional[str] = None,
    customer: Optional[str] = None,
    overdue_only: bool = False,
    limit: int = Query(default=50, ge=1, le=200),
    _: bool = Depends(verify_api_key)
):
    try:
        with get_transaction() as cur:
            query = """
                SELECT so.id, so.order_number, c.name AS customer,
                       so.order_date, so.requested_ship_date, so.status,
                       COUNT(sol.id) AS line_count,
                       COALESCE(SUM(sol.quantity_lb), 0) AS total_lb,
                       COALESCE(SUM(sol.quantity_shipped_lb), 0) AS shipped_lb
                FROM sales_orders so
                JOIN customers c ON c.id = so.customer_id
                LEFT JOIN sales_order_lines sol ON sol.sales_order_id = so.id
                WHERE 1=1
            """
            params = []
            if status:
                query += " AND so.status = %s"
                params.append(status)
            if customer:
                query += " AND LOWER(c.name) LIKE LOWER(%s)"
                params.append(f"%{customer}%")
            if overdue_only:
                query += " AND so.requested_ship_date < CURRENT_DATE AND so.status NOT IN ('shipped', 'invoiced', 'cancelled')"

            query += " GROUP BY so.id, c.name ORDER BY so.requested_ship_date ASC NULLS LAST LIMIT %s"
            params.append(limit)
            cur.execute(query, params)
            rows = cur.fetchall()

            orders = []
            for r in rows:
                total = float(r['total_lb'] or 0)
                shipped = float(r['shipped_lb'] or 0)
                ship_date = r['requested_ship_date']
                orders.append({
                    "order_id": r['id'],
                    "order_number": r['order_number'],
                    "customer": r['customer'],
                    "order_date": str(r['order_date']),
                    "requested_ship_date": str(ship_date) if ship_date else None,
                    "status": r['status'],
                    "line_count": r['line_count'],
                    "total_lb": total,
                    "shipped_lb": shipped,
                    "remaining_lb": total - shipped,
                    "overdue": ship_date is not None and ship_date < date.today() and r['status'] not in ('shipped', 'invoiced', 'cancelled')
                })
            return {"orders": orders, "count": len(orders)}
    except Exception as e:
        logger.error(f"List sales orders failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/sales/orders/fulfillment-check")
def fulfillment_check(
    customer_name: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    order_id: Optional[int] = Query(default=None),
    _: bool = Depends(verify_api_key)
):
    """
    Read-only fulfillment feasibility check across open orders.
    Returns which orders can be fully fulfilled from current inventory.
    """
    OPEN_STATUSES = ('confirmed', 'in_production', 'ready')

    if status and status not in OPEN_STATUSES:
        raise HTTPException(400,
            f"Invalid status filter '{status}'. Must be one of: {list(OPEN_STATUSES)}"
        )

    try:
        with get_transaction() as cur:
            # Build dynamic query for orders
            query = """
                SELECT so.id, so.order_number, so.status, so.requested_ship_date,
                       c.name AS customer
                FROM sales_orders so
                JOIN customers c ON c.id = so.customer_id
                WHERE so.status IN %s
            """
            params: list = [OPEN_STATUSES]

            if order_id is not None:
                query += " AND so.id = %s"
                params.append(order_id)

            if customer_name:
                query += " AND LOWER(c.name) LIKE LOWER(%s)"
                params.append(f"%{customer_name}%")

            if status:
                query += " AND so.status = %s"
                params.append(status)

            query += " ORDER BY so.requested_ship_date ASC NULLS LAST, so.id ASC"
            cur.execute(query, tuple(params))
            orders = cur.fetchall()

            results = []
            summary = {"total_orders_checked": 0, "fulfillable": 0, "partially_fulfillable": 0, "blocked": 0}

            for order in orders:
                # Get active lines with remaining quantity
                cur.execute(
                    """SELECT sol.id, p.id AS product_id, p.name,
                              sol.quantity_lb, sol.quantity_shipped_lb
                       FROM sales_order_lines sol
                       JOIN products p ON p.id = sol.product_id
                       WHERE sol.sales_order_id = %s
                         AND sol.line_status NOT IN ('fulfilled', 'cancelled')
                       ORDER BY sol.id""",
                    (order['id'],)
                )
                lines = cur.fetchall()

                order_lines = []
                total_remaining = 0
                total_on_hand = 0
                total_shortfall = 0
                lines_fulfillable = 0
                lines_checked = 0

                for line in lines:
                    remaining = float(line['quantity_lb']) - float(line['quantity_shipped_lb'])
                    if remaining <= 0:
                        continue

                    # Same inventory query as shipOrderPreview
                    cur.execute(
                        """SELECT COALESCE(SUM(tl.quantity_lb), 0) as on_hand
                           FROM lots l
                           JOIN transaction_lines tl ON tl.lot_id = l.id
                           WHERE l.product_id = %s""",
                        (line['product_id'],)
                    )
                    on_hand = float(cur.fetchone()['on_hand'])
                    shortfall = max(0, remaining - on_hand)
                    can_fulfill = on_hand >= remaining

                    lines_checked += 1
                    if can_fulfill:
                        lines_fulfillable += 1

                    total_remaining += remaining
                    total_on_hand += on_hand
                    total_shortfall += shortfall

                    order_lines.append({
                        "line_id": line['id'],
                        "product": line['name'],
                        "ordered_lb": float(line['quantity_lb']),
                        "shipped_lb": float(line['quantity_shipped_lb']),
                        "remaining_lb": remaining,
                        "on_hand_lb": on_hand,
                        "can_fulfill": can_fulfill,
                        "shortfall_lb": shortfall
                    })

                # Skip orders with nothing remaining
                if lines_checked == 0:
                    continue

                order_fulfillable = (lines_fulfillable == lines_checked)

                # Classify for summary
                summary["total_orders_checked"] += 1
                if lines_fulfillable == lines_checked:
                    summary["fulfillable"] += 1
                elif lines_fulfillable > 0:
                    summary["partially_fulfillable"] += 1
                else:
                    summary["blocked"] += 1

                results.append({
                    "order_id": order['id'],
                    "order_number": order['order_number'],
                    "customer": order['customer'],
                    "status": order['status'],
                    "requested_ship_date": str(order['requested_ship_date']) if order['requested_ship_date'] else None,
                    "fulfillable": order_fulfillable,
                    "lines": order_lines,
                    "total_remaining_lb": total_remaining,
                    "total_on_hand_lb": total_on_hand,
                    "total_shortfall_lb": total_shortfall
                })

            # Sort: fulfillable first within each date group
            results.sort(key=lambda o: (
                o['requested_ship_date'] or '9999-12-31',
                0 if o['fulfillable'] else 1
            ))

            return {
                "summary": summary,
                "orders": results
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fulfillment check failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/sales/orders/{order_id}")
def get_sales_order(order_id: int, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute(
                """SELECT so.id, so.order_number, c.name AS customer, so.order_date,
                          so.requested_ship_date, so.status, so.notes, so.notes_es, so.created_at
                   FROM sales_orders so
                   JOIN customers c ON c.id = so.customer_id
                   WHERE so.id = %s""",
                (order_id,)
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(404, f"Order #{order_id} not found")

            date_str, time_str = format_timestamp(row['created_at'])
            order = {
                "order_id": row['id'],
                "order_number": row['order_number'],
                "customer": row['customer'],
                "order_date": str(row['order_date']),
                "requested_ship_date": str(row['requested_ship_date']) if row['requested_ship_date'] else None,
                "status": row['status'],
                "notes": row['notes'],
                "created_date": date_str,
                "created_time": time_str
            }
            if row.get('notes_es'):
                order["notes_es"] = row['notes_es']

            cur.execute(
                """SELECT sol.id, p.name, sol.quantity_lb, sol.quantity_shipped_lb,
                          sol.unit_price, sol.line_status, sol.notes, sol.notes_es
                   FROM sales_order_lines sol
                   JOIN products p ON p.id = sol.product_id
                   WHERE sol.sales_order_id = %s
                   ORDER BY sol.id""",
                (order_id,)
            )
            lines = []
            total_ordered = 0
            total_shipped = 0
            total_value = 0
            for r in cur.fetchall():
                qty = float(r['quantity_lb'])
                shipped = float(r['quantity_shipped_lb'])
                price = float(r['unit_price']) if r['unit_price'] else None
                total_ordered += qty
                total_shipped += shipped
                if price:
                    total_value += qty * price
                line_data = {
                    "line_id": r['id'],
                    "product": r['name'],
                    "quantity_lb": qty,
                    "quantity_shipped_lb": shipped,
                    "remaining_lb": qty - shipped,
                    "unit_price": price,
                    "line_value": round(qty * price, 2) if price else None,
                    "line_status": r['line_status'],
                    "notes": r['notes']
                }
                if r.get('notes_es'):
                    line_data["notes_es"] = r['notes_es']
                lines.append(line_data)

            cur.execute(
                """SELECT sos.id, sol.id AS line_id, p.name AS product,
                          sos.quantity_lb, sos.shipped_at, t.id AS transaction_id
                   FROM sales_order_shipments sos
                   JOIN sales_order_lines sol ON sol.id = sos.sales_order_line_id
                   JOIN products p ON p.id = sol.product_id
                   JOIN transactions t ON t.id = sos.transaction_id
                   WHERE sol.sales_order_id = %s
                   ORDER BY sos.shipped_at DESC""",
                (order_id,)
            )
            shipments = []
            for r in cur.fetchall():
                s_date, s_time = format_timestamp(r['shipped_at'])
                shipments.append({
                    "shipment_id": r['id'],
                    "line_id": r['line_id'],
                    "product": r['product'],
                    "quantity_lb": float(r['quantity_lb']),
                    "shipped_date": s_date,
                    "shipped_time": s_time,
                    "transaction_id": r['transaction_id']
                })

            order["lines"] = lines
            order["shipments"] = shipments
            order["totals"] = {
                "total_ordered_lb": total_ordered,
                "total_shipped_lb": total_shipped,
                "remaining_lb": total_ordered - total_shipped,
                "total_value": round(total_value, 2) if total_value > 0 else None
            }
            return order
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get sales order failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.patch("/sales/orders/{order_id}/status")
def update_order_status(order_id: int, req: OrderStatusUpdate, _: bool = Depends(verify_api_key)):
    all_statuses = list(VALID_TRANSITIONS.keys())
    if req.status not in all_statuses:
        raise HTTPException(400, f"Invalid status. Must be one of: {all_statuses}")

    # Block manual setting of shipped/partial_ship — those are auto-only via shipOrderCommit
    if req.status in ('shipped', 'partial_ship'):
        raise HTTPException(400,
            f"'{req.status}' status is set automatically when an order is shipped. "
            f"Use the ship endpoint instead."
        )

    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Get current status first
                cur.execute(
                    "SELECT order_number, status FROM sales_orders WHERE id = %s",
                    (order_id,)
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(404, f"Order #{order_id} not found")

                current = row['status']
                allowed = MANUAL_TRANSITIONS.get(current, [])

                if req.status not in allowed:
                    if not allowed:
                        raise HTTPException(400,
                            f"Order {row['order_number']} is '{current}' — this is a terminal status. "
                            f"No further status changes are allowed."
                        )
                    raise HTTPException(400,
                        f"Invalid status transition: '{current}' → '{req.status}'. "
                        f"Allowed transitions from '{current}': {allowed}."
                    )

                cur.execute(
                    "UPDATE sales_orders SET status = %s WHERE id = %s RETURNING order_number, status",
                    (req.status, order_id)
                )
                updated = cur.fetchone()
                logger.info(f"Order {updated['order_number']} status: {current} → {req.status}")
                return {
                    "order_number": updated['order_number'],
                    "previous_status": current,
                    "status": updated['status'],
                    "message": f"Order {updated['order_number']}: {current} → {req.status}"
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update order status failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/sales/orders/{order_id}/lines")
def add_order_lines(order_id: int, req: AddOrderLines, _: bool = Depends(verify_api_key)):
    for line in req.lines:
        validate_bilingual(line.notes, line.notes_es, "notes")
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT order_number, status FROM sales_orders WHERE id = %s", (order_id,))
                row = cur.fetchone()
                if not row:
                    raise HTTPException(404, f"Order #{order_id} not found")
                if row['status'] in ('shipped', 'invoiced', 'cancelled'):
                    raise HTTPException(400, f"Cannot add lines to {row['status']} order")

                results = []
                warnings = []
                for line in req.lines:
                    product_id, prod_name = resolve_product_id(cur, line.product_name)

                    # Fix #2: Auto-lookup case weight from product if not provided
                    effective_case_weight = line.case_weight_lb
                    used_unit = line.unit or 'lb'
                    if used_unit in ('cases', 'bags', 'boxes') and effective_case_weight is None:
                        cur.execute(
                            "SELECT default_case_weight_lb FROM products WHERE id = %s",
                            (product_id,)
                        )
                        prod_row = cur.fetchone()
                        if prod_row and prod_row.get('default_case_weight_lb'):
                            effective_case_weight = float(prod_row['default_case_weight_lb'])
                        else:
                            raise HTTPException(400,
                                f"case_weight_lb is required for '{prod_name}' when ordering in {used_unit}. "
                                f"No default case weight is set for this product."
                            )
                        line.quantity_lb = line.quantity * effective_case_weight

                    # Fix #1: Warn if unit was not explicitly provided
                    if line.quantity is not None and line.unit is None:
                        warnings.append(
                            f"⚠️ '{prod_name}': No unit specified for quantity {line.quantity:,.0f} — "
                            f"defaulting to lb. Did you mean cases?"
                        )

                    cur.execute(
                        """INSERT INTO sales_order_lines (sales_order_id, product_id, quantity_lb, unit_price, notes, notes_es)
                           VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                        (order_id, product_id, line.quantity_lb, line.unit_price, line.notes, line.notes_es)
                    )
                    line_id = cur.fetchone()['id']
                    results.append({
                        "line_id": line_id,
                        "product": prod_name,
                        "quantity_lb": line.quantity_lb,
                        "original_quantity": line.quantity,
                        "original_unit": used_unit,
                        "case_weight_lb": effective_case_weight
                    })

                response = {"order_number": row['order_number'], "lines_added": results, "message": f"Added {len(results)} line(s) to {row['order_number']}"}
                if warnings:
                    response["warnings"] = warnings
                return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Add order lines failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.patch("/sales/orders/{order_id}/lines/{line_id}/cancel")
def cancel_order_line(order_id: int, line_id: int, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """UPDATE sales_order_lines SET line_status = 'cancelled'
                       WHERE id = %s AND sales_order_id = %s AND line_status != 'fulfilled'
                       RETURNING id""",
                    (line_id, order_id)
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(404, "Line not found or already fulfilled")
                return {"line_id": line_id, "line_status": "cancelled", "message": "Line cancelled"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cancel order line failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.patch("/sales/orders/{order_id}/lines/{line_id}/update")
def update_order_line(
    order_id: int,
    line_id: int,
    quantity_lb: Optional[float] = Query(default=None),
    unit_price: Optional[float] = Query(default=None),
    _: bool = Depends(verify_api_key)
):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                fields = []
                values = []
                if quantity_lb is not None:
                    fields.append("quantity_lb = %s")
                    values.append(quantity_lb)
                if unit_price is not None:
                    fields.append("unit_price = %s")
                    values.append(unit_price)
                if not fields:
                    raise HTTPException(400, "Nothing to update")
                values.extend([line_id, order_id])
                cur.execute(
                    f"""UPDATE sales_order_lines SET {', '.join(fields)}
                        WHERE id = %s AND sales_order_id = %s AND line_status NOT IN ('fulfilled', 'cancelled')
                        RETURNING id, quantity_lb, unit_price""",
                    values
                )
                row = cur.fetchone()
                if not row:
                    raise HTTPException(404, "Line not found or already fulfilled/cancelled")
                return {"line_id": row['id'], "quantity_lb": float(row['quantity_lb']), "unit_price": float(row['unit_price']) if row['unit_price'] else None}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update order line failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# SHIP AGAINST ORDER ENDPOINTS (v2.3.0)
# ═══════════════════════════════════════════════════════════════

@app.post("/sales/orders/{order_id}/ship/preview")
def ship_order_preview(order_id: int, req: Optional[ShipOrderRequest] = None, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute(
                """SELECT so.order_number, so.status, c.name
                   FROM sales_orders so
                   JOIN customers c ON c.id = so.customer_id
                   WHERE so.id = %s""",
                (order_id,)
            )
            order_row = cur.fetchone()
            if not order_row:
                raise HTTPException(404, f"Order #{order_id} not found")
            if order_row['status'] == 'new':
                raise HTTPException(400,
                    f"Cannot ship order {order_row['order_number']} — status is 'new'. "
                    f"Confirm the order first (update status to 'confirmed' or later)."
                )
            if order_row['status'] in ('invoiced', 'cancelled'):
                raise HTTPException(400, f"Cannot ship {order_row['status']} order")

            ship_all = (req is None) or (req.ship_all)

            cur.execute(
                """SELECT sol.id, p.id AS product_id, p.name, sol.quantity_lb, sol.quantity_shipped_lb
                   FROM sales_order_lines sol
                   JOIN products p ON p.id = sol.product_id
                   WHERE sol.sales_order_id = %s AND sol.line_status NOT IN ('fulfilled', 'cancelled')
                   ORDER BY sol.id""",
                (order_id,)
            )
            lines = cur.fetchall()

            preview = []
            warnings = []
            for line in lines:
                remaining = float(line['quantity_lb']) - float(line['quantity_shipped_lb'])
                if remaining <= 0:
                    continue

                cur.execute(
                    """SELECT COALESCE(SUM(tl.quantity_lb), 0) as on_hand
                       FROM lots l
                       JOIN transaction_lines tl ON tl.lot_id = l.id
                       WHERE l.product_id = %s""",
                    (line['product_id'],)
                )
                on_hand = float(cur.fetchone()['on_hand'])

                if ship_all:
                    ship_qty = remaining
                elif req and req.lines:
                    match = next((rl for rl in req.lines if rl.line_id == line['id']), None)
                    if not match:
                        continue
                    ship_qty = match.quantity_lb
                else:
                    ship_qty = remaining

                can_ship = min(ship_qty, on_hand)
                if can_ship < ship_qty:
                    warnings.append(f"{line['name']}: only {on_hand:.1f} lb on hand, need {ship_qty:.1f} lb")

                preview.append({
                    "line_id": line['id'],
                    "product": line['name'],
                    "ordered_lb": float(line['quantity_lb']),
                    "already_shipped_lb": float(line['quantity_shipped_lb']),
                    "remaining_lb": remaining,
                    "requested_ship_lb": ship_qty,
                    "can_ship_lb": can_ship,
                    "on_hand_lb": on_hand,
                    "short": max(0, ship_qty - on_hand)
                })

            return {
                "order_number": order_row['order_number'],
                "customer": order_row['name'],
                "status": order_row['status'],
                "lines": preview,
                "warnings": warnings,
                "message": "Preview only — call /ship/commit to execute"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ship order preview failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/sales/orders/{order_id}/ship/commit")
def ship_order_commit(order_id: int, req: Optional[ShipOrderRequest] = None, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """SELECT so.id, so.order_number, so.status, c.name
                       FROM sales_orders so
                       JOIN customers c ON c.id = so.customer_id
                       WHERE so.id = %s""",
                    (order_id,)
                )
                order_row = cur.fetchone()
                if not order_row:
                    raise HTTPException(404, f"Order #{order_id} not found")
                if order_row['status'] == 'new':
                    raise HTTPException(400,
                        f"Cannot ship order {order_row['order_number']} — status is 'new'. "
                        f"Confirm the order first (update status to 'confirmed' or later)."
                    )
                if order_row['status'] in ('invoiced', 'cancelled'):
                    raise HTTPException(400, f"Cannot ship {order_row['status']} order")

                ship_all = (req is None) or (req.ship_all)

                if ship_all:
                    cur.execute(
                        """SELECT sol.id, sol.product_id, sol.quantity_lb, sol.quantity_shipped_lb, p.name
                           FROM sales_order_lines sol
                           JOIN products p ON p.id = sol.product_id
                           WHERE sol.sales_order_id = %s
                             AND sol.line_status NOT IN ('fulfilled', 'cancelled')
                           ORDER BY sol.id""",
                        (order_id,)
                    )
                    lines_to_ship = [
                        {"line_id": r['id'], "product_id": r['product_id'],
                         "quantity_lb": float(r['quantity_lb']) - float(r['quantity_shipped_lb']),
                         "product_name": r['name']}
                        for r in cur.fetchall()
                        if float(r['quantity_lb']) - float(r['quantity_shipped_lb']) > 0
                    ]
                else:
                    lines_to_ship = []
                    for rl in (req.lines or []):
                        cur.execute(
                            """SELECT sol.id, sol.product_id, sol.quantity_lb, sol.quantity_shipped_lb, p.name
                               FROM sales_order_lines sol
                               JOIN products p ON p.id = sol.product_id
                               WHERE sol.id = %s AND sol.sales_order_id = %s""",
                            (rl.line_id, order_id)
                        )
                        r = cur.fetchone()
                        if not r:
                            raise HTTPException(404, f"Line #{rl.line_id} not found on order #{order_id}")
                        remaining = float(r['quantity_lb']) - float(r['quantity_shipped_lb'])
                        ship_qty = min(rl.quantity_lb, remaining)
                        if ship_qty > 0:
                            lines_to_ship.append({
                                "line_id": r['id'], "product_id": r['product_id'],
                                "quantity_lb": ship_qty, "product_name": r['name']
                            })

                if not lines_to_ship:
                    raise HTTPException(400, "Nothing to ship — all lines fulfilled or cancelled")

                now = get_plant_now()
                results = []
                all_fully_shipped = True

                for item in lines_to_ship:
                    qty_to_ship = item["quantity_lb"]

                    # FIFO 3-step pattern (v2.1.3)
                    cur.execute(
                        """SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) AS balance
                           FROM lots l
                           JOIN transaction_lines tl ON tl.lot_id = l.id
                           WHERE l.product_id = %s
                           GROUP BY l.id
                           HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                           ORDER BY l.created_at ASC""",
                        (item["product_id"],)
                    )
                    candidates = cur.fetchall()
                    lot_ids = [c['id'] for c in candidates]

                    if lot_ids:
                        cur.execute("SELECT id FROM lots WHERE id = ANY(%s) FOR UPDATE", (lot_ids,))
                        cur.execute(
                            """SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) AS balance
                               FROM lots l
                               JOIN transaction_lines tl ON tl.lot_id = l.id
                               WHERE l.id = ANY(%s)
                               GROUP BY l.id
                               HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                               ORDER BY l.created_at ASC""",
                            (lot_ids,)
                        )
                        lots = cur.fetchall()
                    else:
                        lots = []

                    available = sum(float(lt['balance']) for lt in lots)
                    actual_ship = min(qty_to_ship, available)

                    if actual_ship <= 0:
                        results.append({
                            "line_id": item["line_id"],
                            "product": item["product_name"],
                            "requested_lb": qty_to_ship,
                            "shipped_lb": 0,
                            "status": "no_stock"
                        })
                        all_fully_shipped = False
                        continue

                    # Create ship transaction
                    cur.execute(
                        """INSERT INTO transactions (type, timestamp, customer_name, notes)
                           VALUES ('ship', %s, %s, %s) RETURNING id""",
                        (now, order_row['name'], f"Sales order {order_row['order_number']} — {item['product_name']}")
                    )
                    txn_id = cur.fetchone()['id']

                    # FIFO consumption
                    remaining_to_ship = actual_ship
                    lots_used = []
                    for lot in lots:
                        if remaining_to_ship <= 0:
                            break
                        balance = float(lot['balance'])
                        take = min(remaining_to_ship, balance)
                        cur.execute(
                            """INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                               VALUES (%s, %s, %s, %s)""",
                            (txn_id, item["product_id"], lot['id'], -take)
                        )
                        remaining_to_ship -= take
                        lots_used.append({"lot_code": lot['lot_code'], "quantity_lb": take})

                    # Update order line
                    cur.execute(
                        """UPDATE sales_order_lines
                           SET quantity_shipped_lb = quantity_shipped_lb + %s
                           WHERE id = %s
                           RETURNING quantity_lb, quantity_shipped_lb""",
                        (actual_ship, item["line_id"])
                    )
                    updated = cur.fetchone()
                    ordered = float(updated['quantity_lb'])
                    new_shipped = float(updated['quantity_shipped_lb'])

                    if new_shipped >= ordered:
                        new_line_status = 'fulfilled'
                    elif new_shipped > 0:
                        new_line_status = 'partial'
                        all_fully_shipped = False
                    else:
                        new_line_status = 'pending'
                        all_fully_shipped = False

                    cur.execute("UPDATE sales_order_lines SET line_status = %s WHERE id = %s", (new_line_status, item["line_id"]))

                    # Record shipment link
                    cur.execute(
                        """INSERT INTO sales_order_shipments (sales_order_line_id, transaction_id, quantity_lb)
                           VALUES (%s, %s, %s)""",
                        (item["line_id"], txn_id, actual_ship)
                    )

                    results.append({
                        "line_id": item["line_id"],
                        "product": item["product_name"],
                        "requested_lb": qty_to_ship,
                        "shipped_lb": actual_ship,
                        "short_lb": max(0, qty_to_ship - actual_ship),
                        "lots_used": lots_used,
                        "transaction_id": txn_id,
                        "line_status": new_line_status
                    })

                    if actual_ship < qty_to_ship:
                        all_fully_shipped = False

                # Update order status
                new_order_status = 'shipped' if all_fully_shipped else 'partial_ship'
                cur.execute("UPDATE sales_orders SET status = %s WHERE id = %s", (new_order_status, order_id))

                logger.info(f"Ship order {order_row['order_number']}: {'fully' if all_fully_shipped else 'partially'} shipped")
                return {
                    "order_number": order_row['order_number'],
                    "customer": order_row['name'],
                    "order_status": new_order_status,
                    "lines_shipped": results,
                    "message": f"Order {order_row['order_number']} {'fully' if all_fully_shipped else 'partially'} shipped"
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ship order commit failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# SALES DASHBOARD (v2.3.0)
# ═══════════════════════════════════════════════════════════════

@app.get("/sales/dashboard")
def sales_dashboard(_: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            # Status counts
            cur.execute(
                """SELECT status, COUNT(*) as cnt FROM sales_orders
                   WHERE status NOT IN ('invoiced', 'cancelled')
                   GROUP BY status ORDER BY status"""
            )
            status_counts = {r['status']: r['cnt'] for r in cur.fetchall()}

            # Overdue
            cur.execute(
                """SELECT so.order_number, c.name AS customer, so.requested_ship_date,
                          SUM(sol.quantity_lb - sol.quantity_shipped_lb) AS remaining_lb
                   FROM sales_orders so
                   JOIN customers c ON c.id = so.customer_id
                   JOIN sales_order_lines sol ON sol.sales_order_id = so.id
                   WHERE so.requested_ship_date < CURRENT_DATE
                     AND so.status NOT IN ('shipped', 'invoiced', 'cancelled')
                   GROUP BY so.id, c.name
                   HAVING SUM(sol.quantity_lb - sol.quantity_shipped_lb) > 0
                   ORDER BY so.requested_ship_date ASC"""
            )
            overdue = [
                {"order_number": r['order_number'], "customer": r['customer'],
                 "requested_ship_date": str(r['requested_ship_date']), "remaining_lb": float(r['remaining_lb'])}
                for r in cur.fetchall()
            ]

            # Due this week
            cur.execute(
                """SELECT so.order_number, c.name AS customer, so.requested_ship_date,
                          SUM(sol.quantity_lb - sol.quantity_shipped_lb) AS remaining_lb
                   FROM sales_orders so
                   JOIN customers c ON c.id = so.customer_id
                   JOIN sales_order_lines sol ON sol.sales_order_id = so.id
                   WHERE so.requested_ship_date BETWEEN CURRENT_DATE AND CURRENT_DATE + INTERVAL '7 days'
                     AND so.status NOT IN ('shipped', 'invoiced', 'cancelled')
                   GROUP BY so.id, c.name
                   HAVING SUM(sol.quantity_lb - sol.quantity_shipped_lb) > 0
                   ORDER BY so.requested_ship_date ASC"""
            )
            due_this_week = [
                {"order_number": r['order_number'], "customer": r['customer'],
                 "requested_ship_date": str(r['requested_ship_date']), "remaining_lb": float(r['remaining_lb'])}
                for r in cur.fetchall()
            ]

            # Recent shipments
            cur.execute(
                """SELECT so.order_number, c.name AS customer, SUM(sos.quantity_lb) AS shipped_lb,
                          MAX(sos.shipped_at) AS last_shipped
                   FROM sales_order_shipments sos
                   JOIN sales_order_lines sol ON sol.id = sos.sales_order_line_id
                   JOIN sales_orders so ON so.id = sol.sales_order_id
                   JOIN customers c ON c.id = so.customer_id
                   WHERE sos.shipped_at > now() - INTERVAL '7 days'
                   GROUP BY so.id, c.name
                   ORDER BY last_shipped DESC"""
            )
            recent_shipments = []
            for r in cur.fetchall():
                s_date, s_time = format_timestamp(r['last_shipped'])
                recent_shipments.append({
                    "order_number": r['order_number'], "customer": r['customer'],
                    "shipped_lb": float(r['shipped_lb']),
                    "last_shipped_date": s_date, "last_shipped_time": s_time
                })

            now_date, now_time = format_timestamp(get_plant_now())
            return {
                "status_summary": status_counts,
                "overdue_orders": overdue,
                "overdue_count": len(overdue),
                "due_this_week": due_this_week,
                "due_this_week_count": len(due_this_week),
                "recent_shipments_7d": recent_shipments,
                "as_of_date": now_date,
                "as_of_time": now_time
            }
    except Exception as e:
        logger.error(f"Sales dashboard failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
