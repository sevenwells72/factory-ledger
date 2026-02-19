from fastapi import FastAPI, HTTPException, Header, Query, Depends
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Union
import json
import pathlib
from datetime import datetime, date, timezone, timedelta
from zoneinfo import ZoneInfo
from contextlib import contextmanager
import psycopg2
from psycopg2.extras import RealDictCursor
import calendar
from psycopg2 import pool
import os
import re
import logging
import secrets
import math
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Factory Ledger System", version="2.5.0")
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
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL env var required — app cannot start without a database")
    if not API_KEY:
        raise RuntimeError("API_KEY env var required — app cannot start without authentication")
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

    # Migration 004: Add exclude_from_inventory flag to batch_formulas
    # Allows utility ingredients (e.g. Water) to remain visible in formulas
    # without blocking production or creating phantom inventory shortages.
    try:
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    ALTER TABLE batch_formulas
                    ADD COLUMN IF NOT EXISTS exclude_from_inventory BOOLEAN DEFAULT false
                """)

                # Flag Water as excluded in all formulas
                cur.execute("""
                    UPDATE batch_formulas bf
                    SET exclude_from_inventory = true
                    FROM products p
                    WHERE p.id = bf.ingredient_product_id
                      AND LOWER(p.name) = 'water'
                      AND bf.exclude_from_inventory = false
                """)
                water_rows = cur.rowcount

                conn.commit()
                if water_rows > 0:
                    logger.info(f"Migration 004: flagged {water_rows} Water formula row(s) as exclude_from_inventory")
                else:
                    logger.info("Migration 004: exclude_from_inventory column up to date")
        finally:
            db_pool.putconn(conn)
    except Exception as e:
        logger.warning(f"Migration 004 warning (non-fatal): {e}")

    # Migration 005: Add yield_multiplier to products
    # Allows products that gain/lose weight during processing (e.g. coconut hydration)
    # to record an expected yield factor. Default 1.0 = no change.
    try:
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    ALTER TABLE products
                    ADD COLUMN IF NOT EXISTS yield_multiplier FLOAT DEFAULT 1.0
                """)
                conn.commit()
                logger.info("Migration 005: yield_multiplier column up to date")
        finally:
            db_pool.putconn(conn)
    except Exception as e:
        logger.warning(f"Migration 005 warning (non-fatal): {e}")

    # Migration 006: Add case_size_lb to products
    # Stores the weight per sellable unit (e.g., 25 for "25 LB" case, 10 for "10 LB" case).
    # Required for correct line_value calculation: cases * unit_price (not lb * unit_price).
    try:
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    ALTER TABLE products
                    ADD COLUMN IF NOT EXISTS case_size_lb NUMERIC(10,2)
                """)

                # Auto-populate from product names
                cur.execute("UPDATE products SET case_size_lb = 25 WHERE name LIKE '%25 LB%' AND case_size_lb IS NULL")
                updated_25 = cur.rowcount
                cur.execute("UPDATE products SET case_size_lb = 10 WHERE name LIKE '%10 LB%' AND case_size_lb IS NULL")
                updated_10 = cur.rowcount
                cur.execute("UPDATE products SET case_size_lb = 50 WHERE name LIKE '%50 LB%' AND case_size_lb IS NULL")
                updated_50 = cur.rowcount

                conn.commit()
                total = updated_25 + updated_10 + updated_50
                if total > 0:
                    logger.info(f"Migration 006: case_size_lb populated for {total} products (25lb:{updated_25}, 10lb:{updated_10}, 50lb:{updated_50})")
                else:
                    logger.info("Migration 006: case_size_lb column up to date")
        finally:
            db_pool.putconn(conn)
    except Exception as e:
        logger.warning(f"Migration 006 warning (non-fatal): {e}")

    # Migration 007: Migrate legacy 'new' orders to 'confirmed'
    # Phase 3 changed default status to 'confirmed', but pre-existing orders may still be 'new'.
    try:
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("UPDATE sales_orders SET status = 'confirmed' WHERE status = 'new'")
                migrated = cur.rowcount
                conn.commit()
                if migrated > 0:
                    logger.info(f"Migration 007: Migrated {migrated} orders from 'new' to 'confirmed'")
                else:
                    logger.info("Migration 007: No legacy 'new' orders to migrate")
        finally:
            db_pool.putconn(conn)
    except Exception as e:
        logger.warning(f"Migration 007 warning (non-fatal): {e}")

    # Migration 008: Lot merge support columns
    # Adds status, merged_into_lot_id, merged_at, merge_reason to lots table
    # for controlled lot merge operations (POST /admin/lots/merge).
    try:
        conn = db_pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("ALTER TABLE lots ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'active'")
                cur.execute("ALTER TABLE lots ADD COLUMN IF NOT EXISTS merged_into_lot_id INTEGER REFERENCES lots(id)")
                cur.execute("ALTER TABLE lots ADD COLUMN IF NOT EXISTS merged_at TIMESTAMPTZ")
                cur.execute("ALTER TABLE lots ADD COLUMN IF NOT EXISTS merge_reason TEXT")
                conn.commit()
                logger.info("Migration 008: lot merge columns up to date")
        finally:
            db_pool.putconn(conn)
    except Exception as e:
        logger.warning(f"Migration 008 warning (non-fatal): {e}")


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


def generate_confirmation_code(transaction_id: int) -> str:
    """Generate a short unique confirmation code from a transaction ID."""
    import hashlib
    hash_input = f"txn-{transaction_id}-cns"
    short_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:6].upper()
    return f"TXN-{short_hash}"


def get_daily_production_summary(cur, target_date=None):
    """Query all make+pack transactions for a given date (plant timezone).
    Returns dict with 'production' list and 'adjustments' list."""
    if target_date is None:
        target_date = get_plant_now().date()

    # Build timezone-aware start/end for the target date
    day_start = datetime(target_date.year, target_date.month, target_date.day,
                         tzinfo=PLANT_TIMEZONE)
    day_end = day_start + timedelta(days=1)

    # Production + packing summary
    cur.execute("""
        SELECT p.name as product_name, p.type as product_type,
               t.type as transaction_type,
               SUM(tl.quantity_lb) FILTER (WHERE tl.quantity_lb > 0) as output_lb,
               COUNT(DISTINCT t.id) as transaction_count
        FROM transactions t
        JOIN transaction_lines tl ON tl.transaction_id = t.id
        JOIN products p ON p.id = tl.product_id
        WHERE t.type IN ('make', 'pack')
          AND t.timestamp >= %s AND t.timestamp < %s
        GROUP BY p.id, p.name, p.type, t.type
        ORDER BY t.type, p.name
    """, (day_start, day_end))
    prod_rows = cur.fetchall()

    production = [
        {
            "product_name": r['product_name'],
            "product_type": r['product_type'],
            "transaction_type": r['transaction_type'],
            "total_lb": float(r['output_lb'] or 0),
            "transaction_count": r['transaction_count']
        }
        for r in prod_rows
    ]

    # Adjustments for the day
    cur.execute("""
        SELECT p.name as product_name, l.lot_code,
               tl.quantity_lb as adjustment_lb,
               t.adjust_reason as reason
        FROM transactions t
        JOIN transaction_lines tl ON tl.transaction_id = t.id
        JOIN products p ON p.id = tl.product_id
        JOIN lots l ON l.id = tl.lot_id
        WHERE t.type = 'adjust'
          AND t.timestamp >= %s AND t.timestamp < %s
        ORDER BY t.timestamp
    """, (day_start, day_end))
    adj_rows = cur.fetchall()

    adjustments = [
        {
            "product_name": r['product_name'],
            "lot_code": r['lot_code'],
            "adjustment_lb": float(r['adjustment_lb']),
            "reason": r['reason']
        }
        for r in adj_rows
    ]

    return {"production": production, "adjustments": adjustments}


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
    # Lot Identity Policy: If lot_code is provided, find-or-create by (product_id, lot_code).
    # Only auto-generate if lot_code is omitted.
    lot_code: Optional[str] = None

class ShipRequest(BaseModel):
    product_name: str
    quantity_lb: float
    customer_name: str
    order_reference: str
    lot_code: Optional[str] = None

class MakeRequest(BaseModel):
    product_name: str
    batches: int
    # Lot Identity Policy: If lot_code is provided, find-or-create by (product_id, lot_code).
    # Only auto-generate if lot_code is omitted.
    lot_code: Optional[str] = None
    ingredient_lot_overrides: Optional[Union[Dict[str, str], str]] = None
    excluded_ingredients: Optional[List[int]] = None
    confirmed_sku: Optional[bool] = None  # Must be True when sibling SKUs exist
    
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

class PackLotAllocation(BaseModel):
    lot_code: str
    quantity_lb: float

class PackRequest(BaseModel):
    source_product: str          # Batch product name or code (e.g., "Batch Classic Granola #9" or "90002")
    target_product: str          # Finished good name or code (e.g., "CQ Granola 10 LB" or "1614")
    cases: int
    case_weight_lb: Optional[float] = None  # Override; defaults to target product's default_case_weight_lb
    lot_allocations: Optional[List[PackLotAllocation]] = None  # Explicit lot splits; FIFO if omitted
    # Lot Identity Policy: If target_lot_code is provided, find-or-create by (product_id, lot_code).
    # Only auto-generate (inherit from batch lot) if target_lot_code is omitted.
    target_lot_code: Optional[str] = None

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
    # Lot Identity Policy: If lot_code is provided, find-or-create by (product_id, lot_code).
    # Only auto-generate if lot_code is omitted.
    lot_code: Optional[str] = None

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
    # Lot Identity Policy: If lot_code is provided, find-or-create by (product_id, lot_code).
    # Only auto-generate if lot_code is omitted.
    lot_code: Optional[str] = None

class VerifyProductRequest(BaseModel):
    action: str
    verified_name: Optional[str] = None
    notes: Optional[str] = None
    notes_es: Optional[str] = None
    performed_by: str = "system"


# ═══════════════════════════════════════════════════════════════
# NOTES / TO-DOS / REMINDERS PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════

class NoteCreate(BaseModel):
    category: str  # 'note', 'todo', 'reminder'
    title: str
    body: Optional[str] = ""
    priority: Optional[str] = "normal"  # 'low', 'normal', 'high'
    due_date: Optional[str] = None  # YYYY-MM-DD
    entity_type: Optional[str] = None  # 'product', 'lot', 'customer', 'supplier'
    entity_id: Optional[str] = None

    @validator("category")
    def validate_category(cls, v):
        if v not in ("note", "todo", "reminder"):
            raise ValueError("category must be 'note', 'todo', or 'reminder'")
        return v

    @validator("priority")
    def validate_priority(cls, v):
        if v not in ("low", "normal", "high"):
            raise ValueError("priority must be 'low', 'normal', or 'high'")
        return v

class NoteUpdate(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    priority: Optional[str] = None
    status: Optional[str] = None  # 'open', 'done', 'dismissed'
    due_date: Optional[str] = None  # YYYY-MM-DD or empty string to clear
    entity_type: Optional[str] = None
    entity_id: Optional[str] = None

    @validator("priority")
    def validate_priority(cls, v):
        if v is not None and v not in ("low", "normal", "high"):
            raise ValueError("priority must be 'low', 'normal', or 'high'")
        return v

    @validator("status")
    def validate_status(cls, v):
        if v is not None and v not in ("open", "done", "dismissed"):
            raise ValueError("status must be 'open', 'done', or 'dismissed'")
        return v


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

class OrderHeaderUpdate(BaseModel):
    requested_ship_date: Optional[str] = None
    notes: Optional[str] = None
    notes_es: Optional[str] = None
    customer_id: Optional[int] = None

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
    # Try fuzzy (name + odoo_code)
    cur.execute(
        """SELECT id, name FROM products
           WHERE (LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) LIKE LOWER(%s))
             AND COALESCE(active, true) = true
           ORDER BY name LIMIT 5""",
        (f"%{product_name}%", f"%{product_name}%")
    )
    rows = cur.fetchall()
    if len(rows) == 1:
        return rows[0]['id'], rows[0]['name']
    elif len(rows) > 1:
        suggestions = [r['name'] for r in rows]
        raise HTTPException(400, f"Multiple products match '{product_name}': {suggestions}")
    raise HTTPException(404, f"Product not found: '{product_name}'")


def resolve_product_full(cur, product_name: str) -> dict:
    """Find product by name/odoo_code. Returns full row dict with id, name, odoo_code, etc.
    Used by receive/ship/make endpoints that need extra columns."""
    # Exact name match
    cur.execute(
        """SELECT id, name, odoo_code, default_batch_lb, default_case_weight_lb,
                       COALESCE(yield_multiplier, 1.0) as yield_multiplier
           FROM products WHERE LOWER(name) = LOWER(%s) AND COALESCE(active, true) = true""",
        (product_name,)
    )
    row = cur.fetchone()
    if row:
        return dict(row)
    # Exact odoo_code match
    cur.execute(
        """SELECT id, name, odoo_code, default_batch_lb, default_case_weight_lb,
                       COALESCE(yield_multiplier, 1.0) as yield_multiplier
           FROM products WHERE LOWER(odoo_code) = LOWER(%s) AND COALESCE(active, true) = true""",
        (product_name,)
    )
    row = cur.fetchone()
    if row:
        return dict(row)
    # Fuzzy match (name + odoo_code)
    cur.execute(
        """SELECT id, name, odoo_code, default_batch_lb, default_case_weight_lb,
                       COALESCE(yield_multiplier, 1.0) as yield_multiplier
           FROM products
           WHERE (LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) LIKE LOWER(%s))
             AND COALESCE(active, true) = true
           ORDER BY name LIMIT 5""",
        (f"%{product_name}%", f"%{product_name}%")
    )
    rows = cur.fetchall()
    if len(rows) == 1:
        return dict(rows[0])
    elif len(rows) > 1:
        suggestions = [f"{r['name']} ({r['odoo_code'] or 'no code'})" for r in rows]
        raise HTTPException(400, f"Multiple products match '{product_name}': {suggestions}")
    raise HTTPException(404, f"Product not found: '{product_name}'")


def get_sibling_skus(cur, product_id: int) -> list:
    """Find other finished-good products that share the exact same BOM ingredients.
    If product A and product B both have identical ingredient sets in batch_formulas,
    they are 'siblings' — same batch source, different labels/packaging.
    Returns list of dicts: [{id, name, odoo_code}, ...] (excluding the given product)."""
    # Get the set of ingredient product IDs for this product
    cur.execute(
        "SELECT ingredient_product_id FROM batch_formulas WHERE product_id = %s ORDER BY ingredient_product_id",
        (product_id,)
    )
    my_ingredients = [row['ingredient_product_id'] for row in cur.fetchall()]

    if not my_ingredients:
        return []  # No formula → no siblings

    # Find all products that have a formula, grouped by their ingredient set
    cur.execute("""
        SELECT bf.product_id, ARRAY_AGG(bf.ingredient_product_id ORDER BY bf.ingredient_product_id) as ingredients
        FROM batch_formulas bf
        JOIN products p ON p.id = bf.product_id AND COALESCE(p.active, true) = true
        WHERE bf.product_id != %s
        GROUP BY bf.product_id
        HAVING ARRAY_AGG(bf.ingredient_product_id ORDER BY bf.ingredient_product_id) = %s::int[]
    """, (product_id, my_ingredients))
    sibling_ids = [row['product_id'] for row in cur.fetchall()]

    if not sibling_ids:
        return []

    cur.execute(
        "SELECT id, name, odoo_code FROM products WHERE id = ANY(%s) ORDER BY name",
        (sibling_ids,)
    )
    return [dict(r) for r in cur.fetchall()]


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
        "version": app.version,
        "status": "online",
        "features": ["receive", "ship", "make", "adjust", "trace", "bom", "quick-create", "lot-reassign", "found-inventory", "ingredient-exclusion", "ingredient-lot-override", "dashboard", "sales-orders", "customers", "fulfillment-check", "bilingual", "sku-disambiguation", "production-scheduling"]
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
# LOT IDENTITY — Find-or-Create Pattern
# ═══════════════════════════════════════════════════════════════
# Lot Identity Policy: A physical lot must map to exactly one canonical lot_id.
# If lot_code is provided, find-or-create by (product_id, lot_code).
# Only auto-generate if lot_code is omitted.

def find_or_create_lot(cur, product_id: int, lot_code: str, entry_source: str,
                       entry_source_notes: str = None, entry_source_notes_es: str = None,
                       found_location: str = None, estimated_age: str = None) -> tuple:
    """Find existing lot or create a new one. Returns (lot_id, is_new).

    Uses INSERT ... ON CONFLICT DO NOTHING + SELECT to guarantee exactly one lot
    per (product_id, lot_code) pair, leveraging the unique index.
    """
    # Build dynamic INSERT with optional columns
    columns = ["product_id", "lot_code", "entry_source"]
    values = [product_id, lot_code, entry_source]
    placeholders = ["%s", "%s", "%s"]

    if entry_source_notes:
        columns.append("entry_source_notes")
        values.append(entry_source_notes)
        placeholders.append("%s")
    if entry_source_notes_es:
        columns.append("entry_source_notes_es")
        values.append(entry_source_notes_es)
        placeholders.append("%s")
    if found_location:
        columns.append("found_location")
        values.append(found_location)
        placeholders.append("%s")
    if estimated_age:
        columns.append("estimated_age")
        values.append(estimated_age)
        placeholders.append("%s")

    col_str = ", ".join(columns)
    ph_str = ", ".join(placeholders)

    cur.execute(f"""
        INSERT INTO lots ({col_str})
        VALUES ({ph_str})
        ON CONFLICT (product_id, lot_code) DO NOTHING
    """, values)
    is_new = cur.rowcount > 0

    # Fetch the lot (whether just created or already existed)
    cur.execute("SELECT id FROM lots WHERE product_id = %s AND lot_code = %s", (product_id, lot_code))
    lot_id = cur.fetchone()['id']

    if not is_new:
        logger.info(f"Found existing lot {lot_code} (id={lot_id}) for product_id={product_id}")

    return lot_id, is_new


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
            product = resolve_product_full(cur, req.product_name)

            # Lot Identity Policy: honor physical lot code if provided
            if req.lot_code:
                lot_code = req.lot_code
                shipper_code = req.shipper_code_override or ''.join(c for c in req.shipper_name.upper() if c.isalpha())[:4] or "UNKN"
                auto = False
                # Check if lot already exists
                cur.execute("SELECT id FROM lots WHERE product_id = %s AND lot_code = %s", (product['id'], req.lot_code))
                existing = cur.fetchone()
            else:
                lot_code, shipper_code, auto = generate_lot_code(cur, req.shipper_name, req.shipper_code_override)
                existing = None
            total_lb = req.cases * req.case_size_lb

            response = {
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
            if existing:
                response["lot_exists"] = True
                response["existing_lot_id"] = existing['id']
                response["preview_message"] += f" (lot already exists — will add to existing)"
            return response
    except Exception as e:
        logger.error(f"Receive preview failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/receive/commit")
def receive_commit(req: ReceiveRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT pg_advisory_xact_lock(1)")
                product = resolve_product_full(cur, req.product_name)

                # Lot Identity Policy: honor physical lot code if provided
                if req.lot_code:
                    lot_code = req.lot_code
                    shipper_code = req.shipper_code_override or ''.join(c for c in req.shipper_name.upper() if c.isalpha())[:4] or "UNKN"
                else:
                    lot_code, shipper_code, _ = generate_lot_code(cur, req.shipper_name, req.shipper_code_override)
                total_lb = req.cases * req.case_size_lb
                now = get_plant_now()

                lot_id, is_new_lot = find_or_create_lot(cur, product['id'], lot_code, 'received')
                
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
                
                lot_verb = "created" if is_new_lot else "found existing"
                logger.info(f"Receive committed: {lot_code} ({lot_verb}) - {total_lb} lb of {product['name']}")

                return {
                    "success": True,
                    "transaction_id": txn_id,
                    "confirmation_code": generate_confirmation_code(txn_id),
                    "lot_id": lot_id,
                    "lot_code": lot_code,
                    "lot_is_new": is_new_lot,
                    "total_lb": total_lb,
                    "receipt_text": receipt,
                    "message": f"Received {total_lb} lb as lot {lot_code}" + ("" if is_new_lot else " (existing lot)")
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
            product = resolve_product_full(cur, req.product_name)

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

            total_available = sum(float(l['available']) for l in lots)
            if total_available < req.quantity_lb:
                raise HTTPException(status_code=400,
                    detail=f"Insufficient total inventory for {product['name']}. "
                           f"Have {total_available} lb across {len(lots)} lot(s), need {req.quantity_lb} lb")

            if req.lot_code:
                selected = next((l for l in lots if l['lot_code'].lower() == req.lot_code.lower()), None)
                if not selected:
                    raise HTTPException(status_code=404, detail=f"Lot '{req.lot_code}' not found or empty")
                lot_selection = "specified"
            else:
                selected = lots[0]
                lot_selection = "FIFO (oldest)"

            single_lot_sufficient = float(selected['available']) >= req.quantity_lb

            if single_lot_sufficient:
                ship_mode = "single_lot"
                allocations = [{"lot_code": selected['lot_code'], "lot_id": selected['id'],
                                "available_lb": float(selected['available']), "allocated_lb": req.quantity_lb}]
                preview_msg = f"Ready to ship {req.quantity_lb} lb of {product['name']} from lot {selected['lot_code']}"
            else:
                # Auto multi-lot FIFO preview
                ship_mode = "multi_lot_fifo"
                allocations = []
                remaining_need = req.quantity_lb
                for lot in lots:
                    if remaining_need <= 0:
                        break
                    take = min(float(lot['available']), remaining_need)
                    allocations.append({
                        "lot_code": lot['lot_code'],
                        "lot_id": lot['id'],
                        "available_lb": float(lot['available']),
                        "allocated_lb": take
                    })
                    remaining_need -= take
                preview_msg = (f"Will ship {req.quantity_lb} lb of {product['name']} "
                               f"from {len(allocations)} lot(s) (auto multi-lot FIFO)")

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
                "ship_mode": ship_mode,
                "allocations": allocations,
                "total_available_lb": total_available,
                "open_orders_warning": open_orders_warning,
                "preview_message": preview_msg
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
                product = resolve_product_full(cur, req.product_name)

                # ── Gather candidate lots (FIFO by lot id) ──
                if req.lot_code:
                    # User pinned a specific lot — try it first, but still
                    # fall back to multi-lot FIFO if it's insufficient.
                    cur.execute("""
                        SELECT l.id, l.lot_code FROM lots l
                        WHERE l.product_id = %s AND LOWER(l.lot_code) = LOWER(%s)
                    """, (product['id'], req.lot_code))
                    lot_row = cur.fetchone()
                    if not lot_row:
                        raise HTTPException(status_code=404, detail=f"Lot '{req.lot_code}' not found")
                    pinned_lot_id = lot_row['id']
                else:
                    pinned_lot_id = None

                # Always load every lot with positive inventory (FIFO order)
                cur.execute("""
                    SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                    FROM lots l
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.product_id = %s
                    GROUP BY l.id
                    HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                    ORDER BY l.id ASC
                """, (product['id'],))
                all_lots = cur.fetchall()

                if not all_lots:
                    raise HTTPException(status_code=400, detail=f"No inventory available for {product['name']}")

                # ── Decide single-lot vs multi-lot ──
                if pinned_lot_id:
                    primary = next((l for l in all_lots if l['id'] == pinned_lot_id), None)
                    if not primary:
                        raise HTTPException(status_code=400,
                            detail=f"Lot '{req.lot_code}' has no available inventory")
                else:
                    primary = all_lots[0]  # FIFO oldest

                single_lot_sufficient = float(primary['available']) >= req.quantity_lb

                if single_lot_sufficient:
                    # ── Single-lot path (original behaviour) ──
                    cur.execute("SELECT id FROM lots WHERE id = %s FOR UPDATE", (primary['id'],))
                    cur.execute("""
                        SELECT COALESCE(SUM(quantity_lb), 0) as available
                        FROM transaction_lines WHERE lot_id = %s
                    """, (primary['id'],))
                    locked_avail = float(cur.fetchone()['available'])

                    if locked_avail < req.quantity_lb:
                        # Race condition — fell below threshold after lock;
                        # fall through to multi-lot below.
                        single_lot_sufficient = False

                if single_lot_sufficient:
                    now = get_plant_now()
                    cur.execute("""
                        INSERT INTO transactions (type, timestamp, customer_name, order_reference)
                        VALUES ('ship', %s, %s, %s) RETURNING id
                    """, (now, req.customer_name, req.order_reference))
                    txn_id = cur.fetchone()['id']

                    cur.execute("""
                        INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                        VALUES (%s, %s, %s, %s)
                    """, (txn_id, product['id'], primary['id'], -req.quantity_lb))

                    remaining = locked_avail - req.quantity_lb
                    shipped_lots = [{"lot_code": primary['lot_code'], "shipped_lb": req.quantity_lb}]
                    ship_mode = "single_lot"
                    log_msg = f"Shipped {req.quantity_lb} lb from lot {primary['lot_code']}. {remaining} lb remaining in lot."

                else:
                    # ── Multi-lot FIFO fallback ──
                    lot_ids = [lot['id'] for lot in all_lots]
                    cur.execute(
                        "SELECT id FROM lots WHERE id = ANY(%s) ORDER BY id ASC FOR UPDATE",
                        (lot_ids,)
                    )
                    # Re-read balances after lock
                    cur.execute("""
                        SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                        FROM lots l
                        LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                        WHERE l.id = ANY(%s)
                        GROUP BY l.id
                        HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                        ORDER BY l.id ASC
                    """, (lot_ids,))
                    locked_lots = cur.fetchall()

                    total_available = sum(float(l['available']) for l in locked_lots)
                    if total_available < req.quantity_lb:
                        raise HTTPException(status_code=400,
                            detail=f"Insufficient total inventory for {product['name']}. "
                                   f"Have {total_available} lb across {len(locked_lots)} lot(s), need {req.quantity_lb} lb")

                    now = get_plant_now()
                    cur.execute("""
                        INSERT INTO transactions (type, timestamp, customer_name, order_reference)
                        VALUES ('ship', %s, %s, %s) RETURNING id
                    """, (now, req.customer_name, req.order_reference))
                    txn_id = cur.fetchone()['id']

                    shipped_lots = []
                    remaining_need = req.quantity_lb
                    for lot in locked_lots:
                        if remaining_need <= 0:
                            break
                        take = min(float(lot['available']), remaining_need)
                        cur.execute("""
                            INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                            VALUES (%s, %s, %s, %s)
                        """, (txn_id, product['id'], lot['id'], -take))
                        shipped_lots.append({"lot_code": lot['lot_code'], "shipped_lb": take})
                        remaining_need -= take

                    remaining = None  # not meaningful for multi-lot
                    ship_mode = "multi_lot_fifo"
                    log_msg = f"Shipped {req.quantity_lb} lb from {len(shipped_lots)} lot(s) (auto multi-lot FIFO)."

                # ── Open-orders warning (shared by both paths) ──
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

                logger.info(f"Ship committed: {req.quantity_lb} lb of {product['name']} to {req.customer_name} ({ship_mode})")

                return {
                    "success": True,
                    "transaction_id": txn_id,
                    "confirmation_code": generate_confirmation_code(txn_id),
                    "quantity_shipped": req.quantity_lb,
                    "ship_mode": ship_mode,
                    "lots_used": shipped_lots,
                    "remaining_in_lot": remaining,
                    "open_orders_warning": open_orders_warning,
                    "message": log_msg
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ship commit failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# MAKE (PRODUCTION) ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/make/preview")
def make_preview(req: MakeRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            product = resolve_product_full(cur, req.product_name)
            
            batch_size = float(product.get('default_batch_lb') or 0)
            yield_multiplier = float(product.get('yield_multiplier') or 1.0)
            formula_weight_lb = batch_size * req.batches
            total_output = formula_weight_lb * yield_multiplier

            manual_excluded_ids = set(req.excluded_ingredients or [])

            cur.execute("""
                SELECT bf.ingredient_product_id, p.name as ingredient_name, bf.quantity_lb,
                       COALESCE(bf.exclude_from_inventory, false) as exclude_from_inventory
                FROM batch_formulas bf
                JOIN products p ON p.id = bf.ingredient_product_id
                WHERE bf.product_id = %s
            """, (product['id'],))
            formula = cur.fetchall()

            # Auto-exclude ingredients flagged in batch_formulas (e.g. Water)
            auto_excluded_ids = set()
            for ing in formula:
                if ing.get('exclude_from_inventory'):
                    auto_excluded_ids.add(ing['ingredient_product_id'])
            excluded_ids = manual_excluded_ids | auto_excluded_ids

            ingredients_needed = []
            excluded_ingredients = []
            lot_overrides_applied = []

            lot_overrides = req.get_lot_overrides()

            # Bulk-fetch all lots with inventory for all BOM ingredients (single query)
            all_ing_ids = [ing['ingredient_product_id'] for ing in formula
                           if ing['ingredient_product_id'] not in excluded_ids]
            ingredient_lots_map = {}  # ing_id -> [{id, lot_code, available}, ...]
            if all_ing_ids:
                cur.execute("""
                    SELECT l.product_id, l.id, l.lot_code,
                           COALESCE(SUM(tl.quantity_lb), 0) as available
                    FROM lots l
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.product_id = ANY(%s)
                    GROUP BY l.id
                    HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                    ORDER BY l.product_id, l.id ASC
                """, (all_ing_ids,))
                for row in cur.fetchall():
                    pid = row['product_id']
                    if pid not in ingredient_lots_map:
                        ingredient_lots_map[pid] = []
                    ingredient_lots_map[pid].append(dict(row))

            for ing in formula:
                ing_id = ing['ingredient_product_id']
                needed = float(ing['quantity_lb']) * req.batches

                if ing_id in excluded_ids:
                    excluded_ingredients.append({
                        "ingredient_id": ing_id,
                        "ingredient_name": ing['ingredient_name'],
                        "would_need_lb": needed,
                        "excluded": True,
                        "exclusion_type": "auto" if ing_id in auto_excluded_ids else "manual"
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
                    available_lots = ingredient_lots_map.get(ing_id, [])
                    total_avail = sum(float(lot['available']) for lot in available_lots)
                    lot_details = [
                        {"lot_code": lot['lot_code'], "available_lb": float(lot['available'])}
                        for lot in available_lots
                    ]
                    ingredients_needed.append({
                        "ingredient_id": ing_id,
                        "ingredient_name": ing['ingredient_name'],
                        "needed_lb": needed,
                        "available_lb": total_avail,
                        "sufficient": total_avail >= needed,
                        "lot_count": len(available_lots),
                        "lots": lot_details
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
            
            # Check for sibling SKUs (same BOM ingredients, different finished-good label)
            siblings = get_sibling_skus(cur, product['id'])

            yield_note = ""
            if yield_multiplier != 1.0:
                yield_note = f" (estimated yield with {yield_multiplier}x multiplier; actual weight may differ)"

            response = {
                "product_id": product['id'],
                "product_name": product['name'],
                "batches": req.batches,
                "batch_size_lb": batch_size,
                "yield_multiplier": yield_multiplier,
                "formula_weight_lb": formula_weight_lb,
                "estimated_yield_lb": total_output,
                "total_output_lb": total_output,
                "lot_code": lot_code,
                "ingredients": ingredients_needed,
                "all_ingredients_available": all_sufficient,
                "preview_message": f"Ready to make {req.batches} batch(es) of {product['name']} ({total_output} lb){yield_note}"
            }

            if siblings:
                sibling_names = [s['name'] for s in siblings]
                response["sibling_skus"] = siblings
                response["sku_confirmation_required"] = True
                response["sku_warning"] = (
                    f"This batch source has {len(siblings) + 1} finished-good SKUs with the same formula. "
                    f"You selected '{product['name']}'. Other options: {sibling_names}. "
                    f"Confirm this is the correct output SKU before committing."
                )

            if lot_overrides_applied:
                response["lot_overrides"] = lot_overrides_applied
                response["preview_message"] += f" (with {len(lot_overrides_applied)} lot override(s))"

            if excluded_ingredients:
                auto_count = sum(1 for e in excluded_ingredients if e.get('exclusion_type') == 'auto')
                manual_count = len(excluded_ingredients) - auto_count
                response["excluded_ingredients"] = excluded_ingredients
                parts = []
                if auto_count:
                    parts.append(f"{auto_count} auto-excluded")
                if manual_count:
                    parts.append(f"{manual_count} manually excluded")
                response["preview_message"] += f" ({', '.join(parts)} ingredient(s))"

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
                product = resolve_product_full(cur, req.product_name)

                # Block if sibling SKUs exist and operator hasn't confirmed
                siblings = get_sibling_skus(cur, product['id'])
                if siblings and not req.confirmed_sku:
                    sibling_names = [s['name'] for s in siblings]
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"SKU confirmation required. '{product['name']}' shares a batch formula with: "
                            f"{sibling_names}. Set confirmed_sku=true to confirm this is the correct "
                            f"output SKU. Never assume — ask the operator which SKU they are packing."
                        )
                    )

                batch_size = float(product.get('default_batch_lb') or 0)
                yield_multiplier = float(product.get('yield_multiplier') or 1.0)
                formula_weight_lb = batch_size * req.batches
                total_output = formula_weight_lb * yield_multiplier
                now = get_plant_now()

                manual_excluded_ids = set(req.excluded_ingredients or [])
                auto_excluded_ids = set()

                # Lot Identity Policy: honor physical lot code if provided
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

                output_lot_id, is_new_lot = find_or_create_lot(cur, product['id'], lot_code, 'production_output')
                
                cur.execute("""
                    SELECT bf.ingredient_product_id, bf.quantity_lb,
                           COALESCE(bf.exclude_from_inventory, false) as exclude_from_inventory
                    FROM batch_formulas bf
                    WHERE bf.product_id = %s
                """, (product['id'],))
                formula = cur.fetchall()

                # Auto-exclude ingredients flagged in batch_formulas (e.g. Water)
                auto_excluded_ids = set()
                for ing in formula:
                    if ing.get('exclude_from_inventory'):
                        auto_excluded_ids.add(ing['ingredient_product_id'])
                excluded_ids = manual_excluded_ids | auto_excluded_ids

                exclusion_note = ""
                if manual_excluded_ids:
                    exclusion_note += f" (manually excluded IDs: {sorted(manual_excluded_ids)})"
                if auto_excluded_ids:
                    exclusion_note += f" (auto-excluded IDs: {sorted(auto_excluded_ids)})"

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

                consumed_by_ingredient = {}  # keyed by ingredient_product_id
                excluded_from_run = []

                lot_overrides = req.get_lot_overrides()

                # Pre-fetch ingredient names for all BOM ingredients
                all_formula_ids = [ing['ingredient_product_id'] for ing in formula]
                ing_names = {}
                if all_formula_ids:
                    cur.execute(
                        "SELECT id, name FROM products WHERE id = ANY(%s)",
                        (all_formula_ids,)
                    )
                    for row in cur.fetchall():
                        ing_names[row['id']] = row['name']

                for ing in formula:
                    ing_id = ing['ingredient_product_id']
                    needed = float(ing['quantity_lb']) * req.batches
                    ing_name = ing_names.get(ing_id, f"ID {ing_id}")

                    if ing_id in excluded_ids:
                        excluded_from_run.append({
                            "ingredient_id": ing_id,
                            "ingredient_name": ing_name,
                            "skipped_lb": needed,
                            "exclusion_type": "auto" if ing_id in auto_excluded_ids else "manual"
                        })
                        continue

                    # Initialize grouping entry for this ingredient
                    if ing_id not in consumed_by_ingredient:
                        consumed_by_ingredient[ing_id] = {
                            "ingredient_id": ing_id,
                            "ingredient_name": ing_name,
                            "total_consumed_lb": 0.0,
                            "lots": []
                        }

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

                        consumed_by_ingredient[ing_id]["total_consumed_lb"] += needed
                        consumed_by_ingredient[ing_id]["lots"].append({
                            "lot_code": override_lot['lot_code'], "consumed_lb": needed, "override": True
                        })
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

                            consumed_by_ingredient[ing_id]["total_consumed_lb"] += take
                            consumed_by_ingredient[ing_id]["lots"].append({
                                "lot_code": lot['lot_code'], "consumed_lb": take
                            })
                            remaining -= take

                        if remaining > 0.001:
                            raise HTTPException(
                                status_code=400,
                                detail=f"Insufficient inventory for ingredient ID {ing_id}. Missing {remaining:.2f} lb"
                            )

                # Build backward-compatible flat list + grouped view
                consumed_flat = []
                for group in consumed_by_ingredient.values():
                    for lot_entry in group["lots"]:
                        consumed_flat.append({
                            "ingredient_id": group["ingredient_id"],
                            "ingredient_name": group["ingredient_name"],
                            **lot_entry
                        })
                
                logger.info(f"Make committed: {lot_code} - {total_output} lb of {product['name']}")
                
                response = {
                    "success": True,
                    "transaction_id": txn_id,
                    "confirmation_code": generate_confirmation_code(txn_id),
                    "lot_id": output_lot_id,
                    "lot_code": lot_code,
                    "yield_multiplier": yield_multiplier,
                    "formula_weight_lb": formula_weight_lb,
                    "estimated_yield_lb": total_output,
                    "output_lb": total_output,
                    "ingredients_consumed": consumed_flat,
                    "ingredients_consumed_grouped": list(consumed_by_ingredient.values()),
                    "message": f"Produced {total_output} lb as lot {lot_code}"
                }
                
                if siblings:
                    response["confirmed_sku"] = True
                    response["sibling_skus"] = [s['name'] for s in siblings]

                if excluded_from_run:
                    auto_count = sum(1 for e in excluded_from_run if e.get('exclusion_type') == 'auto')
                    manual_count = len(excluded_from_run) - auto_count
                    response["excluded_ingredients"] = excluded_from_run
                    parts = []
                    if auto_count:
                        parts.append(f"{auto_count} auto-excluded")
                    if manual_count:
                        parts.append(f"{manual_count} manually excluded")
                    response["message"] += f" ({', '.join(parts)} ingredient(s))"

                response["daily_production_summary"] = get_daily_production_summary(cur)
                return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Make commit failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# PACK (BATCH → FINISHED GOOD) ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/pack/preview")
def pack_preview(req: PackRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            # Resolve source (batch) product
            source = resolve_product_full(cur, req.source_product)

            # Resolve target (finished good) product
            target = resolve_product_full(cur, req.target_product)

            # Determine case weight
            case_weight = req.case_weight_lb
            if case_weight is None:
                case_weight = float(target.get('default_case_weight_lb') or 0)
            if case_weight <= 0:
                raise HTTPException(400, f"Case weight required. Product '{target['name']}' has no default_case_weight_lb set.")

            total_lb = req.cases * case_weight

            # Get available batch lots (FIFO order)
            cur.execute("""
                SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                FROM lots l
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE l.product_id = %s
                GROUP BY l.id
                HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                ORDER BY l.id ASC
            """, (source['id'],))
            available_lots = cur.fetchall()

            total_available = sum(float(lot['available']) for lot in available_lots)

            # Build allocation plan
            if req.lot_allocations:
                # Explicit allocations provided
                allocations = []
                for alloc in req.lot_allocations:
                    matched = None
                    for lot in available_lots:
                        if lot['lot_code'].lower() == alloc.lot_code.lower():
                            matched = lot
                            break
                    if not matched:
                        allocations.append({
                            "lot_code": alloc.lot_code,
                            "available_lb": 0,
                            "allocated_lb": alloc.quantity_lb,
                            "sufficient": False,
                            "error": f"Lot '{alloc.lot_code}' not found or has no inventory for {source['name']}"
                        })
                        continue
                    allocations.append({
                        "lot_id": matched['id'],
                        "lot_code": matched['lot_code'],
                        "available_lb": float(matched['available']),
                        "allocated_lb": alloc.quantity_lb,
                        "sufficient": float(matched['available']) >= alloc.quantity_lb
                    })
                alloc_total = sum(a['allocated_lb'] for a in allocations)
                if abs(alloc_total - total_lb) > 0.01:
                    return JSONResponse(status_code=400, content={
                        "error": f"Lot allocations sum to {alloc_total} lb but {total_lb} lb needed ({req.cases} cases x {case_weight} lb)"
                    })
            else:
                # FIFO allocation
                allocations = []
                remaining = total_lb
                for lot in available_lots:
                    if remaining <= 0:
                        break
                    take = min(float(lot['available']), remaining)
                    allocations.append({
                        "lot_id": lot['id'],
                        "lot_code": lot['lot_code'],
                        "available_lb": float(lot['available']),
                        "allocated_lb": take,
                        "sufficient": True
                    })
                    remaining -= take

            all_sufficient = all(a.get('sufficient', False) for a in allocations)

            # Determine output lot code
            if req.target_lot_code:
                output_lot_code = req.target_lot_code
            elif allocations and allocations[0].get('lot_code'):
                # Inherit from primary (first) source lot
                output_lot_code = allocations[0]['lot_code']
            else:
                output_lot_code = "UNKNOWN"

            return {
                "source_product_id": source['id'],
                "source_product_name": source['name'],
                "target_product_id": target['id'],
                "target_product_name": target['name'],
                "cases": req.cases,
                "case_weight_lb": case_weight,
                "total_lb": total_lb,
                "output_lot_code": output_lot_code,
                "allocations": allocations,
                "all_lots_sufficient": all_sufficient,
                "total_batch_available_lb": total_available,
                "source_lot_count": len(available_lots),
                "source_lots": [
                    {"lot_code": lot['lot_code'], "available_lb": float(lot['available'])}
                    for lot in available_lots
                ],
                "preview_message": f"Ready to pack {req.cases} cases ({total_lb} lb) of {target['name']} from {source['name']} ({len(available_lots)} batch lot(s))"
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pack preview failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/pack/commit")
def pack_commit(req: PackRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Resolve and validate products
                source = resolve_product_full(cur, req.source_product)
                target = resolve_product_full(cur, req.target_product)

                # Determine case weight
                case_weight = req.case_weight_lb
                if case_weight is None:
                    case_weight = float(target.get('default_case_weight_lb') or 0)
                if case_weight <= 0:
                    raise HTTPException(400, f"Case weight required for '{target['name']}'")

                total_lb = req.cases * case_weight
                now = get_plant_now()

                # Get available batch lots (FIFO order)
                cur.execute("""
                    SELECT l.id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available
                    FROM lots l
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.product_id = %s
                    GROUP BY l.id
                    HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                    ORDER BY l.id ASC
                """, (source['id'],))
                candidate_lots = cur.fetchall()

                if not candidate_lots:
                    raise HTTPException(400, f"No batch inventory available for {source['name']}")

                # Lock lots
                lot_ids = [lot['id'] for lot in candidate_lots]
                cur.execute(
                    "SELECT id FROM lots WHERE id = ANY(%s) ORDER BY id ASC FOR UPDATE",
                    (lot_ids,)
                )

                # Re-read availability after lock
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
                lots_by_code = {lot['lot_code'].lower(): lot for lot in lots}

                # Build allocation plan
                if req.lot_allocations:
                    alloc_plan = []
                    for alloc in req.lot_allocations:
                        lot = lots_by_code.get(alloc.lot_code.lower())
                        if not lot:
                            raise HTTPException(400, f"Lot '{alloc.lot_code}' not found or empty for {source['name']}")
                        if float(lot['available']) < alloc.quantity_lb:
                            raise HTTPException(400, f"Lot '{alloc.lot_code}' has {lot['available']} lb, need {alloc.quantity_lb} lb")
                        alloc_plan.append((lot, alloc.quantity_lb))
                    alloc_total = sum(qty for _, qty in alloc_plan)
                    if abs(alloc_total - total_lb) > 0.01:
                        raise HTTPException(400, f"Allocations sum to {alloc_total} lb, need {total_lb} lb ({req.cases} cases x {case_weight} lb)")
                else:
                    # FIFO allocation
                    total_available = sum(float(l['available']) for l in lots)
                    if total_available < total_lb:
                        raise HTTPException(400, f"Insufficient batch inventory. Have {total_available} lb, need {total_lb} lb")
                    alloc_plan = []
                    remaining = total_lb
                    for lot in lots:
                        if remaining <= 0:
                            break
                        take = min(float(lot['available']), remaining)
                        alloc_plan.append((lot, take))
                        remaining -= take
                    if remaining > 0.001:
                        raise HTTPException(400, f"Could not fully allocate. Missing {remaining:.2f} lb")

                # Determine output lot code (inherit from primary batch lot)
                if req.target_lot_code:
                    output_lot_code = req.target_lot_code
                else:
                    output_lot_code = alloc_plan[0][0]['lot_code']

                # Lot Identity Policy: find-or-create output lot for the finished good
                output_lot_id, is_new_lot = find_or_create_lot(cur, target['id'], output_lot_code, 'pack_output')

                # Create the pack transaction
                source_lot_summary = ", ".join(f"{lot['lot_code']} ({qty} lb)" for lot, qty in alloc_plan)
                cur.execute("""
                    INSERT INTO transactions (type, timestamp, notes)
                    VALUES ('pack', %s, %s)
                    RETURNING id
                """, (now, f"Pack {req.cases} cases of {target['name']} from {source['name']} lots: {source_lot_summary}"))
                txn_id = cur.fetchone()['id']

                # Positive line: finished good output
                cur.execute("""
                    INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                    VALUES (%s, %s, %s, %s)
                """, (txn_id, target['id'], output_lot_id, total_lb))

                # Negative lines: batch inventory deductions + traceability
                consumed = []
                for lot, qty in alloc_plan:
                    cur.execute("""
                        INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                        VALUES (%s, %s, %s, %s)
                    """, (txn_id, source['id'], lot['id'], -qty))

                    # Record in ingredient_lot_consumption for traceability
                    cur.execute("""
                        INSERT INTO ingredient_lot_consumption (transaction_id, ingredient_product_id, ingredient_lot_id, quantity_lb)
                        VALUES (%s, %s, %s, %s)
                    """, (txn_id, source['id'], lot['id'], qty))

                    consumed.append({
                        "lot_code": lot['lot_code'],
                        "consumed_lb": qty
                    })

                logger.info(f"Pack committed: {output_lot_code} - {total_lb} lb of {target['name']} from {source['name']}")

                response = {
                    "success": True,
                    "transaction_id": txn_id,
                    "confirmation_code": generate_confirmation_code(txn_id),
                    "output_lot_id": output_lot_id,
                    "output_lot_code": output_lot_code,
                    "target_product_name": target['name'],
                    "source_product_name": source['name'],
                    "cases": req.cases,
                    "case_weight_lb": case_weight,
                    "total_lb": total_lb,
                    "batch_lots_consumed": consumed,
                    "message": f"Packed {req.cases} cases ({total_lb} lb) of {target['name']} as lot {output_lot_code}"
                }

                response["daily_production_summary"] = get_daily_production_summary(cur)
                return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pack commit failed: {e}")
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
                product = resolve_product_full(cur, req.product_name)
                cur.execute(
                    "SELECT id as lot_id, lot_code FROM lots WHERE product_id = %s AND LOWER(lot_code) = LOWER(%s)",
                    (product['id'], req.lot_code)
                )
                lot = cur.fetchone()
                if not lot:
                    raise HTTPException(404, f"Lot '{req.lot_code}' not found for product '{product['name']}'")
                # Fetch label_type for SKU protection check
                cur.execute("SELECT COALESCE(label_type, 'house') as label_type FROM products WHERE id = %s", (product['id'],))
                lt_row = cur.fetchone()
                result = {**product, 'product_id': product['id'], 'lot_id': lot['lot_id'], 'lot_code': lot['lot_code'],
                          'label_type': lt_row['label_type'] if lt_row else 'house'}

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

                # Compute new balance after adjustment
                cur.execute("""
                    SELECT COALESCE(SUM(tl.quantity_lb), 0) as new_balance
                    FROM transaction_lines tl
                    WHERE tl.lot_id = %s
                """, (result['lot_id'],))
                new_balance = float(cur.fetchone()['new_balance'])

                logger.info(f"Adjust committed: {req.adjustment_lb} lb to lot {result['lot_code']} (balance: {new_balance} lb)")

                response = {
                    "success": True,
                    "transaction_id": txn_id,
                    "confirmation_code": generate_confirmation_code(txn_id),
                    "product_id": result['product_id'],
                    "product_name": result['name'],
                    "lot_code": result['lot_code'],
                    "adjustment_lb": req.adjustment_lb,
                    "new_balance_lb": new_balance,
                    "reason": req.reason,
                    "message": f"Adjusted lot {result['lot_code']} by {req.adjustment_lb} lb (new balance: {new_balance} lb)"
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
                WHERE t.type IN ('make', 'pack') AND LOWER(l.lot_code) = LOWER(%s) AND tl.quantity_lb > 0
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

                # Lot Identity Policy: honor physical lot code if provided
                if req.lot_code:
                    lot_code = req.lot_code
                else:
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

                lot_id, is_new_lot = find_or_create_lot(
                    cur, req.product_id, lot_code, 'found_inventory',
                    entry_source_notes=req.notes, entry_source_notes_es=req.notes_es,
                    found_location=req.found_location, estimated_age=req.estimated_age
                )

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

                # Lot Identity Policy: honor physical lot code if provided
                if req.lot_code:
                    lot_code = req.lot_code
                else:
                    date_part = now.strftime("%y-%m-%d")
                    cur.execute("SELECT lot_code FROM lots WHERE lot_code LIKE %s ORDER BY lot_code DESC LIMIT 1", (f"{date_part}-FOUND-%",))
                    existing_lot = cur.fetchone()
                    seq = (int(existing_lot['lot_code'].split('-')[-1]) + 1) if existing_lot else 1
                    lot_code = f"{date_part}-FOUND-{seq:03d}"

                lot_id, is_new_lot = find_or_create_lot(
                    cur, product['id'], lot_code, 'found_inventory',
                    entry_source_notes=req.notes, entry_source_notes_es=req.notes_es,
                    found_location=req.found_location, estimated_age=req.estimated_age
                )
                
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
                SELECT bf.ingredient_product_id, p.name as ingredient_name, p.odoo_code, bf.quantity_lb,
                       COALESCE(bf.exclude_from_inventory, false) as exclude_from_inventory
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
            {"code": "hydration_yield", "description": "Hydration/processing yield correction"},
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
                    """INSERT INTO sales_orders (customer_id, requested_ship_date, notes, notes_es, order_number, status)
                       VALUES (%s, %s, %s, %s, '', 'confirmed')
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
                    "status": "confirmed",
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
                if status == 'open':
                    query += " AND so.status NOT IN ('shipped', 'invoiced', 'cancelled')"
                else:
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
                is_open = r['status'] not in ('shipped', 'invoiced', 'cancelled')

                # Proactive warnings for open orders
                order_warnings = []
                if is_open:
                    if ship_date is None:
                        order_warnings.append("⚠️ No ship date set")
                    if 'test' in r['customer'].lower():
                        order_warnings.append("⚠️ Possible test order")
                    if total == 0 and r['line_count'] == 0:
                        order_warnings.append("⚠️ Empty order — no line items")

                order = {
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
                    "overdue": ship_date is not None and ship_date < date.today() and is_open
                }
                if order_warnings:
                    order["warnings"] = order_warnings
                orders.append(order)
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
                          sol.unit_price, sol.line_status, sol.notes, sol.notes_es,
                          p.case_size_lb
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
                case_size = float(r['case_size_lb']) if r['case_size_lb'] else None
                cases = round(qty / case_size) if case_size else None
                total_ordered += qty
                total_shipped += shipped

                # line_value = cases * price_per_case (not lb * price)
                line_value = None
                if price and cases:
                    line_value = round(cases * price, 2)
                    total_value += line_value
                elif price:
                    # Fallback for products without case_size_lb: treat unit_price as price/lb
                    line_value = round(qty * price, 2)
                    total_value += line_value

                # Detect non-weight items (pallets, freight, surcharges, etc.)
                product_name_lower = r['name'].lower()
                non_weight_keywords = ('pallet', 'freight', 'delivery', 'surcharge', 'charge', 'fee')
                is_non_weight = any(kw in product_name_lower for kw in non_weight_keywords)

                if is_non_weight:
                    price_basis = "per_unit"
                elif case_size:
                    price_basis = "per_case"
                else:
                    price_basis = "per_lb"

                line_data = {
                    "line_id": r['id'],
                    "product": r['name'],
                    "quantity_lb": qty,
                    "cases": cases,
                    "case_size_lb": case_size,
                    "quantity_shipped_lb": shipped,
                    "remaining_lb": qty - shipped,
                    "case_price": price,
                    "price_basis": price_basis,
                    "line_value": line_value,
                    "line_status": r['line_status'],
                    "notes": r['notes']
                }
                if is_non_weight:
                    line_data["is_non_weight"] = True
                    line_data["unit_quantity"] = int(qty) if qty == int(qty) else qty
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


@app.patch("/sales/orders/{order_id}")
def update_order_header(order_id: int, req: OrderHeaderUpdate, _: bool = Depends(verify_api_key)):
    """Update order header fields (ship date, notes, customer). Only allowed when status is 'new' or 'confirmed'."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, order_number, status, customer_id, requested_ship_date, notes, notes_es FROM sales_orders WHERE id = %s",
                    (order_id,)
                )
                order = cur.fetchone()
                if not order:
                    raise HTTPException(404, f"Order #{order_id} not found")

                if order['status'] not in ('new', 'confirmed'):
                    raise HTTPException(400,
                        f"Order {order['order_number']} is '{order['status']}' — header edits only allowed when status is 'new' or 'confirmed'."
                    )

                updates = {}
                if req.requested_ship_date is not None:
                    updates['requested_ship_date'] = req.requested_ship_date if req.requested_ship_date else None
                if req.notes is not None:
                    updates['notes'] = req.notes if req.notes else None
                if req.notes_es is not None:
                    updates['notes_es'] = req.notes_es if req.notes_es else None
                if req.customer_id is not None:
                    # Verify customer exists
                    cur.execute("SELECT id, name FROM customers WHERE id = %s", (req.customer_id,))
                    cust = cur.fetchone()
                    if not cust:
                        raise HTTPException(404, f"Customer ID {req.customer_id} not found")
                    updates['customer_id'] = req.customer_id

                if not updates:
                    raise HTTPException(400, "No fields to update")

                set_clause = ", ".join(f"{k} = %s" for k in updates)
                values = list(updates.values()) + [order_id]
                cur.execute(
                    f"UPDATE sales_orders SET {set_clause} WHERE id = %s RETURNING id, order_number, status, customer_id, requested_ship_date, notes, notes_es",
                    values
                )
                updated = cur.fetchone()

                # Get customer name for response
                cur.execute("SELECT name FROM customers WHERE id = %s", (updated['customer_id'],))
                customer_name = cur.fetchone()['name']

                changes = list(updates.keys())
                logger.info(f"Order {updated['order_number']} header updated: {changes}")
                return {
                    "order_id": updated['id'],
                    "order_number": updated['order_number'],
                    "status": updated['status'],
                    "customer_id": updated['customer_id'],
                    "customer_name": customer_name,
                    "requested_ship_date": str(updated['requested_ship_date']) if updated['requested_ship_date'] else None,
                    "notes": updated['notes'],
                    "notes_es": updated['notes_es'],
                    "fields_updated": changes,
                    "message": f"Order {updated['order_number']} updated: {', '.join(changes)}"
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Update order header failed: {e}")
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
                        "confirmation_code": generate_confirmation_code(txn_id),
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


# ═══════════════════════════════════════════════════════════════
# WEB DASHBOARD API (no auth — read-only, same-origin)
# ═══════════════════════════════════════════════════════════════

_DASHBOARD_CONFIG_PATH = pathlib.Path(__file__).parent / "dashboard" / "dashboard_config.json"

def _load_dashboard_config():
    try:
        with open(_DASHBOARD_CONFIG_PATH) as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dashboard config: {e}")
        return None


@app.get("/dashboard/api/production")
def dashboard_api_production(
    days: int = Query(default=5, ge=1, le=31),
    month: Optional[str] = Query(default=None)
):
    """Rolling production calendar — batches made + finished goods packed.
    Excludes ship/receive/adjust. Only 'make' and 'pack' transactions."""
    try:
        with get_transaction() as cur:
            # Timestamps are stored in ET via get_plant_now(), so use DATE()
            # directly — no timezone conversion needed.
            if month:
                # Full month view: e.g. month=2026-02
                try:
                    parts = month.split("-")
                    y, m = int(parts[0]), int(parts[1])
                    start_date = f"{y}-{m:02d}-01"
                    last_day = calendar.monthrange(y, m)[1]
                    end_date = f"{y}-{m:02d}-{last_day}"
                except (ValueError, IndexError):
                    raise HTTPException(400, "month must be YYYY-MM format")
                date_filter = "DATE(t.timestamp) BETWEEN %s AND %s"
                params = [start_date, end_date]
            else:
                now_et = get_plant_now()
                start = (now_et - timedelta(days=days - 1)).strftime("%Y-%m-%d")
                date_filter = "DATE(t.timestamp) >= %s"
                params = [start]

            cur.execute(f"""
                SELECT DATE(t.timestamp) as prod_date,
                       p.name as product_name, p.type as product_type,
                       p.default_batch_lb,
                       SUM(tl.quantity_lb) FILTER (WHERE tl.quantity_lb > 0) as total_lbs,
                       COUNT(DISTINCT t.id) as txn_count
                FROM transactions t
                JOIN transaction_lines tl ON tl.transaction_id = t.id
                JOIN products p ON p.id = tl.product_id
                WHERE t.type IN ('make', 'pack')
                  AND tl.quantity_lb > 0
                  AND {date_filter}
                GROUP BY prod_date, p.id
                ORDER BY prod_date DESC, p.name
            """, params)
            rows = cur.fetchall()

        # Group by day
        days_map = {}
        for r in rows:
            d = str(r['prod_date'])
            if d not in days_map:
                dt = r['prod_date']
                day_name = dt.strftime("%A") if hasattr(dt, 'strftime') else d
                days_map[d] = {"date": d, "day_name": day_name, "batches": [], "finished_goods": []}
            entry = {
                "product_name": r['product_name'],
                "total_lbs": float(r['total_lbs'] or 0),
                "product_type": r['product_type']
            }
            if r['product_type'] == 'batch':
                batch_size = float(r['default_batch_lb']) if r['default_batch_lb'] else None
                entry["standard_batch_size_lbs"] = batch_size
                days_map[d]["batches"].append(entry)
            else:
                days_map[d]["finished_goods"].append(entry)

        return {"days": list(days_map.values())}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard production API failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/dashboard/api/inventory/finished-goods")
def dashboard_api_finished_goods():
    """On-hand inventory for finished goods, grouped by panel with lot breakdown."""
    config = _load_dashboard_config()
    if not config:
        return JSONResponse(status_code=500, content={"error": "Dashboard config not found"})
    try:
        panels = list(config.get("finished_goods_panels", []))
        coconut = config.get("coconut_panel")
        if coconut:
            panels.append(coconut)

        # Collect all SKU names
        all_skus = []
        for panel in panels:
            all_skus.extend(panel.get("skus", []))

        with get_transaction() as cur:
            # Get on-hand per product
            cur.execute("""
                SELECT p.id, p.name, p.default_case_weight_lb,
                       COALESCE(SUM(tl.quantity_lb), 0) as on_hand_lbs
                FROM products p
                LEFT JOIN lots l ON l.product_id = p.id
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE COALESCE(p.active, true) = true
                  AND LOWER(p.name) = ANY(SELECT LOWER(unnest(%s::text[])))
                GROUP BY p.id
            """, (all_skus,))
            product_rows = {r['name'].lower(): dict(r) for r in cur.fetchall()}

            # Get lot breakdown for all matched products
            matched_ids = [r['id'] for r in product_rows.values()]
            lot_map = {}
            if matched_ids:
                cur.execute("""
                    SELECT l.product_id, l.lot_code,
                           COALESCE(SUM(tl.quantity_lb), 0) as on_hand_lbs,
                           l.id as lot_id
                    FROM lots l
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.product_id = ANY(%s)
                    GROUP BY l.id
                    HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                    ORDER BY l.id ASC
                """, (matched_ids,))
                for lr in cur.fetchall():
                    pid = lr['product_id']
                    if pid not in lot_map:
                        lot_map[pid] = []
                    lot_map[pid].append({
                        "lot_code": lr['lot_code'],
                        "on_hand_lbs": float(lr['on_hand_lbs'])
                    })

        result_panels = []
        for panel in panels:
            panel_data = {
                "id": panel.get("id", ""),
                "title": panel.get("title", ""),
                "case_weight_lb": panel.get("case_weight_lb"),
                "products": [],
                "missing_skus": []
            }
            for sku in panel.get("skus", []):
                prow = product_rows.get(sku.lower())
                if prow:
                    pid = prow['id']
                    on_hand = float(prow['on_hand_lbs'])
                    case_wt = panel.get("case_weight_lb")
                    if case_wt is None and prow.get('default_case_weight_lb'):
                        case_wt = float(prow['default_case_weight_lb'])
                    product_entry = {
                        "product_name": prow['name'],
                        "on_hand_lbs": on_hand,
                        "case_weight_lb": case_wt,
                        "lots": lot_map.get(pid, [])
                    }
                    panel_data["products"].append(product_entry)
                else:
                    panel_data["missing_skus"].append(sku)
            # Sort by on_hand descending
            panel_data["products"].sort(key=lambda x: x["on_hand_lbs"], reverse=True)
            result_panels.append(panel_data)

        return {"panels": result_panels}
    except Exception as e:
        logger.error(f"Dashboard finished goods API failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/dashboard/api/inventory/batches")
def dashboard_api_batches():
    """Batch inventory on-hand with estimated batch counts and lot breakdown."""
    config = _load_dashboard_config()
    if not config:
        return JSONResponse(status_code=500, content={"error": "Dashboard config not found"})
    try:
        batch_skus = config.get("batch_skus", [])
        sku_names = [b["name"] for b in batch_skus]
        # Build a lookup for standard batch sizes from config
        config_batch_sizes = {b["name"].lower(): b.get("standard_batch_size_lbs") for b in batch_skus}

        with get_transaction() as cur:
            cur.execute("""
                SELECT p.id, p.name, p.default_batch_lb,
                       COALESCE(SUM(tl.quantity_lb), 0) as on_hand_lbs
                FROM products p
                LEFT JOIN lots l ON l.product_id = p.id
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE COALESCE(p.active, true) = true
                  AND LOWER(p.name) = ANY(SELECT LOWER(unnest(%s::text[])))
                GROUP BY p.id
                ORDER BY COALESCE(SUM(tl.quantity_lb), 0) DESC
            """, (sku_names,))
            product_rows = cur.fetchall()

            matched_ids = [r['id'] for r in product_rows]
            lot_map = {}
            if matched_ids:
                cur.execute("""
                    SELECT l.product_id, l.lot_code,
                           COALESCE(SUM(tl.quantity_lb), 0) as on_hand_lbs
                    FROM lots l
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.product_id = ANY(%s)
                    GROUP BY l.id
                    HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                    ORDER BY l.id ASC
                """, (matched_ids,))
                for lr in cur.fetchall():
                    pid = lr['product_id']
                    if pid not in lot_map:
                        lot_map[pid] = []
                    lot_map[pid].append({
                        "lot_code": lr['lot_code'],
                        "on_hand_lbs": float(lr['on_hand_lbs'])
                    })

        found_names = {r['name'].lower() for r in product_rows}
        missing_skus = [n for n in sku_names if n.lower() not in found_names]

        batches = []
        for r in product_rows:
            on_hand = float(r['on_hand_lbs'])
            # Prefer config batch size, fall back to DB default_batch_lb
            batch_size = config_batch_sizes.get(r['name'].lower())
            if batch_size is None and r['default_batch_lb']:
                batch_size = float(r['default_batch_lb'])
            batches.append({
                "product_name": r['name'],
                "on_hand_lbs": on_hand,
                "standard_batch_size_lbs": batch_size,
                "lots": lot_map.get(r['id'], [])
            })

        return {"batches": batches, "missing_skus": missing_skus}
    except Exception as e:
        logger.error(f"Dashboard batches API failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/dashboard/api/inventory/ingredients")
def dashboard_api_ingredients(category: Optional[str] = Query(default=None)):
    """Ingredient/raw material on-hand grouped by category."""
    config = _load_dashboard_config()
    if not config:
        return JSONResponse(status_code=500, content={"error": "Dashboard config not found"})
    try:
        categories = config.get("ingredient_categories", [])
        if category:
            categories = [c for c in categories if c["id"] == category]

        all_names = []
        for cat in categories:
            all_names.extend(cat.get("items", []))

        with get_transaction() as cur:
            cur.execute("""
                SELECT p.id, p.name,
                       COALESCE(SUM(tl.quantity_lb), 0) as on_hand
                FROM products p
                LEFT JOIN lots l ON l.product_id = p.id
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE COALESCE(p.active, true) = true
                  AND LOWER(p.name) = ANY(SELECT LOWER(unnest(%s::text[])))
                GROUP BY p.id
            """, (all_names,))
            rows = cur.fetchall()
            product_map = {r['name'].lower(): {"id": r['id'], "name": r['name'], "on_hand": float(r['on_hand'])} for r in rows}

            # Fetch lot-level breakdown for all matched products
            matched_ids = [r['id'] for r in rows]
            lot_map = {}
            if matched_ids:
                cur.execute("""
                    SELECT l.product_id, l.lot_code,
                           COALESCE(SUM(tl.quantity_lb), 0) as on_hand_lbs
                    FROM lots l
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.product_id = ANY(%s)
                    GROUP BY l.id
                    HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                    ORDER BY l.id ASC
                """, (matched_ids,))
                for lr in cur.fetchall():
                    pid = lr['product_id']
                    if pid not in lot_map:
                        lot_map[pid] = []
                    lot_map[pid].append({
                        "lot_code": lr['lot_code'],
                        "on_hand_lbs": float(lr['on_hand_lbs'])
                    })

        result = []
        for cat in categories:
            items = []
            missing = []
            for item_name in cat.get("items", []):
                pdata = product_map.get(item_name.lower())
                if pdata:
                    items.append({
                        "name": pdata["name"],
                        "on_hand": pdata["on_hand"],
                        "lots": lot_map.get(pdata["id"], [])
                    })
                else:
                    missing.append(item_name)
            # Preserve config ordering (don't sort by on_hand)
            result.append({
                "id": cat["id"],
                "title": cat["title"],
                "unit": cat.get("unit", "lb"),
                "items": items,
                "missing_skus": missing,
                "total_skus_expected": len(cat.get("items", []))
            })

        return {"categories": result}
    except Exception as e:
        logger.error(f"Dashboard ingredients API failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/dashboard/api/activity/shipments")
def dashboard_api_shipments(limit: int = Query(default=100, ge=1, le=500)):
    """Shipping log — most recent first."""
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT t.id, t.timestamp, t.customer_name, t.order_reference, t.notes,
                       json_agg(json_build_object(
                           'product_name', p.name,
                           'lot_code', l.lot_code,
                           'quantity_lb', tl.quantity_lb
                       ) ORDER BY p.name) as lines
                FROM transactions t
                JOIN transaction_lines tl ON tl.transaction_id = t.id
                JOIN products p ON p.id = tl.product_id
                LEFT JOIN lots l ON l.id = tl.lot_id
                WHERE t.type = 'ship'
                GROUP BY t.id
                ORDER BY t.timestamp DESC
                LIMIT %s
            """, (limit,))
            shipments = cur.fetchall()

        result = []
        for s in shipments:
            d, tm = format_timestamp(s['timestamp'])
            total_lbs = sum(abs(float(ln['quantity_lb'] or 0)) for ln in (s['lines'] or []))
            result.append({
                "transaction_id": s['id'],
                "date": d,
                "time": tm,
                "customer_name": s['customer_name'],
                "order_reference": s['order_reference'],
                "total_lbs": total_lbs,
                "lines": s['lines'] or [],
                "notes": s['notes']
            })
        return {"shipments": result}
    except Exception as e:
        logger.error(f"Dashboard shipments API failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/dashboard/api/activity/receipts")
def dashboard_api_receipts(limit: int = Query(default=100, ge=1, le=500)):
    """Receiving log — most recent first."""
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT t.id, t.timestamp, t.shipper_name, t.bol_reference, t.notes,
                       t.cases_received, t.case_size_lb,
                       json_agg(json_build_object(
                           'product_name', p.name,
                           'lot_code', l.lot_code,
                           'quantity_lb', tl.quantity_lb
                       ) ORDER BY p.name) as lines
                FROM transactions t
                JOIN transaction_lines tl ON tl.transaction_id = t.id
                JOIN products p ON p.id = tl.product_id
                LEFT JOIN lots l ON l.id = tl.lot_id
                WHERE t.type = 'receive'
                GROUP BY t.id
                ORDER BY t.timestamp DESC
                LIMIT %s
            """, (limit,))
            receipts = cur.fetchall()

        result = []
        for r in receipts:
            d, tm = format_timestamp(r['timestamp'])
            total_lbs = sum(float(ln['quantity_lb'] or 0) for ln in (r['lines'] or []))
            result.append({
                "transaction_id": r['id'],
                "date": d,
                "time": tm,
                "shipper_name": r['shipper_name'],
                "bol_reference": r['bol_reference'],
                "total_lbs": total_lbs,
                "cases_received": r['cases_received'],
                "case_size_lb": float(r['case_size_lb']) if r['case_size_lb'] else None,
                "lines": r['lines'] or [],
                "notes": r['notes']
            })
        return {"receipts": result}
    except Exception as e:
        logger.error(f"Dashboard receipts API failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/dashboard/api/lot/{lot_code}")
def dashboard_api_lot_detail(lot_code: str):
    """Lot detail with full transaction timeline."""
    try:
        with get_transaction() as cur:
            # Lot info
            cur.execute("""
                SELECT l.id, l.lot_code, l.product_id, p.name as product_name,
                       l.entry_source,
                       COALESCE(SUM(tl.quantity_lb), 0) as on_hand_lbs
                FROM lots l
                JOIN products p ON p.id = l.product_id
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE LOWER(l.lot_code) = LOWER(%s)
                GROUP BY l.id, p.id
            """, (lot_code,))
            lot = cur.fetchone()
            if not lot:
                raise HTTPException(404, f"Lot '{lot_code}' not found")

            # First transaction to get original quantity
            cur.execute("""
                SELECT tl.quantity_lb
                FROM transaction_lines tl
                JOIN transactions t ON t.id = tl.transaction_id
                WHERE tl.lot_id = %s
                ORDER BY t.timestamp ASC
                LIMIT 1
            """, (lot['id'],))
            first_txn = cur.fetchone()
            original_qty = float(first_txn['quantity_lb']) if first_txn else 0

            # Full timeline
            cur.execute("""
                SELECT t.id as transaction_id, t.type, t.timestamp,
                       tl.quantity_lb,
                       t.customer_name, t.shipper_name, t.order_reference,
                       t.bol_reference, t.adjust_reason, t.notes
                FROM transaction_lines tl
                JOIN transactions t ON t.id = tl.transaction_id
                WHERE tl.lot_id = %s
                ORDER BY t.timestamp ASC
            """, (lot['id'],))
            timeline_rows = cur.fetchall()

        timeline = []
        for tr in timeline_rows:
            d, tm = format_timestamp(tr['timestamp'])
            timeline.append({
                "transaction_id": tr['transaction_id'],
                "type": tr['type'],
                "date": d,
                "time": tm,
                "quantity_lb": float(tr['quantity_lb']),
                "customer_name": tr['customer_name'],
                "shipper_name": tr['shipper_name'],
                "order_reference": tr['order_reference'],
                "bol_reference": tr['bol_reference'],
                "adjust_reason": tr['adjust_reason'],
                "notes": tr['notes']
            })

        return {
            "lot_code": lot['lot_code'],
            "product_name": lot['product_name'],
            "entry_source": lot['entry_source'],
            "original_quantity_lbs": original_qty,
            "on_hand_lbs": float(lot['on_hand_lbs']),
            "timeline": timeline
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard lot detail API failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/dashboard/api/search")
def dashboard_api_search(q: str = Query(min_length=1)):
    """Global search across products, lots, orders, customers, suppliers."""
    try:
        term = f"%{q}%"
        results = {}
        with get_transaction() as cur:
            # Products
            cur.execute("""
                SELECT p.name, p.type, p.odoo_code,
                       COALESCE(SUM(tl.quantity_lb), 0) as on_hand_lbs
                FROM products p
                LEFT JOIN lots l ON l.product_id = p.id
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE COALESCE(p.active, true) = true
                  AND (p.name ILIKE %s OR p.odoo_code ILIKE %s)
                GROUP BY p.id
                ORDER BY p.name
                LIMIT 20
            """, (term, term))
            results["products"] = [dict(r) for r in cur.fetchall()]
            for p in results["products"]:
                p["on_hand_lbs"] = float(p["on_hand_lbs"])

            # Lots
            cur.execute("""
                SELECT l.lot_code, p.name as product_name,
                       COALESCE(SUM(tl.quantity_lb), 0) as on_hand_lbs
                FROM lots l
                JOIN products p ON p.id = l.product_id
                LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                WHERE l.lot_code ILIKE %s
                GROUP BY l.id, p.id
                ORDER BY l.id DESC
                LIMIT 20
            """, (term,))
            results["lots"] = [dict(r) for r in cur.fetchall()]
            for lt in results["lots"]:
                lt["on_hand_lbs"] = float(lt["on_hand_lbs"])

            # Sales orders
            cur.execute("""
                SELECT so.order_number, c.name as customer, so.status,
                       so.order_date
                FROM sales_orders so
                JOIN customers c ON c.id = so.customer_id
                WHERE so.order_number ILIKE %s OR c.name ILIKE %s
                ORDER BY so.id DESC
                LIMIT 20
            """, (term, term))
            results["orders"] = [
                {**dict(r), "order_date": str(r["order_date"]) if r["order_date"] else None}
                for r in cur.fetchall()
            ]

            # Customers
            cur.execute("""
                SELECT name, contact_name, email, phone
                FROM customers
                WHERE name ILIKE %s AND active = true
                ORDER BY name
                LIMIT 20
            """, (term,))
            results["customers"] = [dict(r) for r in cur.fetchall()]

        return results
    except Exception as e:
        logger.error(f"Dashboard search API failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# DASHBOARD API — Notes / To-Dos / Reminders (NO AUTH)
# ═══════════════════════════════════════════════════════════════

def _note_row_to_dict(row):
    """Convert a notes DB row to a JSON-safe dict."""
    d = dict(row)
    for key in ("created_at", "updated_at"):
        if d.get(key):
            date_str, time_str = format_timestamp(d[key])
            d[key] = f"{date_str} {time_str}"
    if d.get("due_date"):
        d["due_date"] = str(d["due_date"])
    return d


@app.get("/dashboard/api/notes")
def dashboard_api_notes(
    category: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    entity_type: Optional[str] = Query(default=None),
    entity_id: Optional[str] = Query(default=None),
):
    """List notes/todos/reminders with optional filters. NO AUTH."""
    try:
        with get_transaction() as cur:
            clauses = []
            params = []
            if category:
                clauses.append("category = %s")
                params.append(category)
            if status:
                clauses.append("status = %s")
                params.append(status)
            if entity_type:
                clauses.append("entity_type = %s")
                params.append(entity_type)
            if entity_id:
                clauses.append("entity_id = %s")
                params.append(entity_id)

            where = ""
            if clauses:
                where = "WHERE " + " AND ".join(clauses)

            cur.execute(f"""
                SELECT * FROM notes
                {where}
                ORDER BY
                    CASE WHEN status = 'open' THEN 0 ELSE 1 END,
                    CASE priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END,
                    due_date ASC NULLS LAST,
                    created_at DESC
            """, params)
            rows = [_note_row_to_dict(r) for r in cur.fetchall()]
            return {"notes": rows}
    except Exception as e:
        logger.error(f"Dashboard notes list failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/dashboard/api/notes")
def dashboard_api_notes_create(req: NoteCreate):
    """Create a note/todo/reminder. NO AUTH."""
    try:
        with get_transaction() as cur:
            due = None
            if req.due_date:
                due = date.fromisoformat(req.due_date)

            cur.execute("""
                INSERT INTO notes (category, title, body, priority, due_date, entity_type, entity_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING *
            """, (req.category, req.title, req.body or "", req.priority, due,
                  req.entity_type, req.entity_id))
            row = cur.fetchone()
            return _note_row_to_dict(row)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Dashboard notes create failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.put("/dashboard/api/notes/{note_id}")
def dashboard_api_notes_update(note_id: int, req: NoteUpdate):
    """Update a note/todo/reminder. NO AUTH."""
    try:
        with get_transaction() as cur:
            # Build SET clause dynamically from provided fields
            sets = []
            params = []
            data = req.dict(exclude_unset=True)
            if not data:
                return JSONResponse(status_code=400, content={"error": "No fields to update"})

            for field, value in data.items():
                if field == "due_date":
                    if value == "" or value is None:
                        sets.append("due_date = NULL")
                    else:
                        sets.append("due_date = %s")
                        params.append(date.fromisoformat(value))
                else:
                    sets.append(f"{field} = %s")
                    params.append(value)

            sets.append("updated_at = NOW()")
            params.append(note_id)

            cur.execute(f"""
                UPDATE notes SET {', '.join(sets)}
                WHERE id = %s
                RETURNING *
            """, params)
            row = cur.fetchone()
            if not row:
                return JSONResponse(status_code=404, content={"error": "Note not found"})
            return _note_row_to_dict(row)
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        logger.error(f"Dashboard notes update failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/dashboard/api/notes/{note_id}")
def dashboard_api_notes_delete(note_id: int):
    """Delete a note/todo/reminder. NO AUTH."""
    try:
        with get_transaction() as cur:
            cur.execute("DELETE FROM notes WHERE id = %s RETURNING id", (note_id,))
            row = cur.fetchone()
            if not row:
                return JSONResponse(status_code=404, content={"error": "Note not found"})
            return {"deleted": True, "id": note_id}
    except Exception as e:
        logger.error(f"Dashboard notes delete failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.put("/dashboard/api/notes/{note_id}/toggle")
def dashboard_api_notes_toggle(note_id: int):
    """Toggle a note's status between open and done. NO AUTH."""
    try:
        with get_transaction() as cur:
            cur.execute("SELECT status FROM notes WHERE id = %s", (note_id,))
            row = cur.fetchone()
            if not row:
                return JSONResponse(status_code=404, content={"error": "Note not found"})
            new_status = "done" if row["status"] == "open" else "open"
            cur.execute("""
                UPDATE notes SET status = %s, updated_at = NOW()
                WHERE id = %s RETURNING *
            """, (new_status, note_id))
            return _note_row_to_dict(cur.fetchone())
    except Exception as e:
        logger.error(f"Dashboard notes toggle failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# ADMIN PRODUCT & BOM MANAGEMENT
# ═══════════════════════════════════════════════════════════════

class ProductUpdate(BaseModel):
    default_case_weight_lb: Optional[float] = None
    default_batch_lb: Optional[float] = None
    yield_multiplier: Optional[float] = None
    active: Optional[bool] = None


@app.put("/admin/products/{product_id}")
def admin_update_product(product_id: int, req: ProductUpdate, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("SELECT id, name FROM products WHERE id = %s", (product_id,))
            product = cur.fetchone()
            if not product:
                raise HTTPException(status_code=404, detail=f"Product ID {product_id} not found")

            updates = []
            params = []
            if req.default_case_weight_lb is not None:
                updates.append("default_case_weight_lb = %s")
                params.append(req.default_case_weight_lb)
            if req.default_batch_lb is not None:
                updates.append("default_batch_lb = %s")
                params.append(req.default_batch_lb)
            if req.yield_multiplier is not None:
                updates.append("yield_multiplier = %s")
                params.append(req.yield_multiplier)
            if req.active is not None:
                updates.append("active = %s")
                params.append(req.active)

            if not updates:
                return {"updated": False, "message": "No fields to update"}

            params.append(product_id)
            cur.execute(f"UPDATE products SET {', '.join(updates)} WHERE id = %s", params)

        return {
            "updated": True,
            "product_id": product_id,
            "product_name": product['name'],
            "changes": {
                k: v for k, v in {
                    "default_case_weight_lb": req.default_case_weight_lb,
                    "default_batch_lb": req.default_batch_lb,
                    "yield_multiplier": req.yield_multiplier,
                    "active": req.active
                }.items() if v is not None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin product update failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


class BomLineCreate(BaseModel):
    ingredient_product_id: int
    quantity_lb: float
    exclude_from_inventory: Optional[bool] = False

class BomLineUpdate(BaseModel):
    quantity_lb: Optional[float] = None
    exclude_from_inventory: Optional[bool] = None


@app.get("/admin/bom/search")
def admin_bom_search(
    product_name: str = Query(...),
    _: bool = Depends(verify_api_key)
):
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT id, name, odoo_code, type, default_batch_lb
                FROM products
                WHERE COALESCE(active, true) = true
                  AND LOWER(name) LIKE LOWER(%s)
                ORDER BY name
                LIMIT 20
            """, (f"%{product_name}%",))
            products = cur.fetchall()
        return {"count": len(products), "products": products}
    except Exception as e:
        logger.error(f"Admin BOM search failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/admin/bom/{product_id}/lines")
def admin_bom_lines(product_id: int, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("SELECT id, name FROM products WHERE id = %s", (product_id,))
            product = cur.fetchone()
            if not product:
                raise HTTPException(status_code=404, detail=f"Product ID {product_id} not found")

            cur.execute("""
                SELECT bf.id AS line_id, bf.ingredient_product_id, p.name AS ingredient_name,
                       bf.quantity_lb, COALESCE(bf.exclude_from_inventory, false) AS exclude_from_inventory
                FROM batch_formulas bf
                JOIN products p ON p.id = bf.ingredient_product_id
                WHERE bf.product_id = %s
                ORDER BY bf.quantity_lb DESC
            """, (product_id,))
            lines = cur.fetchall()

        return {
            "product_id": product['id'],
            "product_name": product['name'],
            "line_count": len(lines),
            "lines": lines
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin BOM lines failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/admin/bom/{product_id}/lines")
def admin_bom_add_line(product_id: int, req: BomLineCreate, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("SELECT id, name FROM products WHERE id = %s", (product_id,))
            if not cur.fetchone():
                raise HTTPException(status_code=404, detail=f"Product ID {product_id} not found")

            cur.execute("SELECT id, name FROM products WHERE id = %s", (req.ingredient_product_id,))
            ingredient = cur.fetchone()
            if not ingredient:
                raise HTTPException(status_code=404, detail=f"Ingredient product ID {req.ingredient_product_id} not found")

            cur.execute("""
                INSERT INTO batch_formulas (product_id, ingredient_product_id, quantity_lb, exclude_from_inventory)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (product_id, req.ingredient_product_id, req.quantity_lb, req.exclude_from_inventory))
            new_line = cur.fetchone()

        return {
            "created": True,
            "line_id": new_line['id'],
            "ingredient_name": ingredient['name'],
            "quantity_lb": req.quantity_lb,
            "exclude_from_inventory": req.exclude_from_inventory
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin BOM add line failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.put("/admin/bom/lines/{line_id}")
def admin_bom_update_line(line_id: int, req: BomLineUpdate, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT bf.id, bf.quantity_lb, COALESCE(bf.exclude_from_inventory, false) AS exclude_from_inventory,
                       p.name AS ingredient_name
                FROM batch_formulas bf
                JOIN products p ON p.id = bf.ingredient_product_id
                WHERE bf.id = %s
            """, (line_id,))
            existing = cur.fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail=f"BOM line ID {line_id} not found")

            updates = []
            params = []
            if req.quantity_lb is not None:
                updates.append("quantity_lb = %s")
                params.append(req.quantity_lb)
            if req.exclude_from_inventory is not None:
                updates.append("exclude_from_inventory = %s")
                params.append(req.exclude_from_inventory)

            if not updates:
                return {"updated": False, "message": "No fields to update"}

            params.append(line_id)
            cur.execute(f"UPDATE batch_formulas SET {', '.join(updates)} WHERE id = %s", params)

        return {
            "updated": True,
            "line_id": line_id,
            "ingredient_name": existing['ingredient_name'],
            "previous_quantity_lb": float(existing['quantity_lb']),
            "new_quantity_lb": req.quantity_lb if req.quantity_lb is not None else float(existing['quantity_lb']),
            "exclude_from_inventory": req.exclude_from_inventory if req.exclude_from_inventory is not None else existing['exclude_from_inventory']
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin BOM update line failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/admin/bom/lines/{line_id}")
def admin_bom_delete_line(line_id: int, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT bf.id, bf.product_id, bf.ingredient_product_id, bf.quantity_lb,
                       p.name AS ingredient_name, prod.name AS product_name
                FROM batch_formulas bf
                JOIN products p ON p.id = bf.ingredient_product_id
                JOIN products prod ON prod.id = bf.product_id
                WHERE bf.id = %s
            """, (line_id,))
            existing = cur.fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail=f"BOM line ID {line_id} not found")

            cur.execute("DELETE FROM batch_formulas WHERE id = %s", (line_id,))

        return {
            "deleted": True,
            "line_id": line_id,
            "product_name": existing['product_name'],
            "ingredient_name": existing['ingredient_name'],
            "quantity_lb": float(existing['quantity_lb'])
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin BOM delete line failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# ADMIN: FG → BATCH PRODUCT MAPPING (product_bom)
# ═══════════════════════════════════════════════════════════════

class ProductBomCreate(BaseModel):
    finished_product_id: int
    component_product_id: int
    quantity: Optional[float] = 1.0
    uom: Optional[str] = "unit"


@app.get("/admin/product-bom")
def admin_list_product_bom(
    fg_only: bool = Query(False, description="Only show batch product components"),
    _: bool = Depends(verify_api_key)
):
    """List all FG → component mappings from product_bom."""
    try:
        with get_transaction() as cur:
            query = """
                SELECT pb.id, pb.finished_product_id, fg.name AS finished_good_name,
                       pb.component_product_id, cp.name AS component_name, cp.type AS component_type,
                       pb.quantity, pb.uom
                FROM product_bom pb
                JOIN products fg ON fg.id = pb.finished_product_id
                JOIN products cp ON cp.id = pb.component_product_id
            """
            if fg_only:
                query += " WHERE cp.type = 'batch'"
            query += " ORDER BY fg.name, cp.type"
            cur.execute(query)
            rows = cur.fetchall()
        return {"count": len(rows), "mappings": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"Admin product-bom list failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/admin/product-bom")
def admin_create_product_bom(req: ProductBomCreate, _: bool = Depends(verify_api_key)):
    """Add a component to a finished good's product_bom."""
    try:
        with get_transaction() as cur:
            cur.execute("SELECT id, name FROM products WHERE id = %s", (req.finished_product_id,))
            fg = cur.fetchone()
            if not fg:
                raise HTTPException(404, f"Finished good ID {req.finished_product_id} not found")

            cur.execute("SELECT id, name FROM products WHERE id = %s", (req.component_product_id,))
            cp = cur.fetchone()
            if not cp:
                raise HTTPException(404, f"Component product ID {req.component_product_id} not found")

            cur.execute("""
                INSERT INTO product_bom (finished_product_id, component_product_id, quantity, uom)
                VALUES (%s, %s, %s, %s)
                RETURNING id
            """, (req.finished_product_id, req.component_product_id, req.quantity, req.uom))
            row = cur.fetchone()

        return {"created": True, "id": row['id'], "finished_good": fg['name'], "component": cp['name'],
                "quantity": req.quantity, "uom": req.uom}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin product-bom create failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/admin/product-bom/{mapping_id}")
def admin_delete_product_bom(mapping_id: int, _: bool = Depends(verify_api_key)):
    """Remove a component from a finished good's product_bom."""
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT pb.id, fg.name AS finished_good_name, cp.name AS component_name
                FROM product_bom pb
                JOIN products fg ON fg.id = pb.finished_product_id
                JOIN products cp ON cp.id = pb.component_product_id
                WHERE pb.id = %s
            """, (mapping_id,))
            existing = cur.fetchone()
            if not existing:
                raise HTTPException(404, f"Mapping ID {mapping_id} not found")

            cur.execute("DELETE FROM product_bom WHERE id = %s", (mapping_id,))

        return {"deleted": True, "finished_good": existing['finished_good_name'], "component": existing['component_name']}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin product-bom delete failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# ADMIN: READ-ONLY SQL QUERY (diagnostics)
# ═══════════════════════════════════════════════════════════════

class AdminSQLQuery(BaseModel):
    sql: str

@app.post("/admin/sql")
def admin_sql_query(req: AdminSQLQuery, _: bool = Depends(verify_api_key)):
    """Read-only SQL for admin diagnostics. Only SELECT allowed."""
    sql = req.sql.strip()
    if not sql.upper().startswith("SELECT"):
        raise HTTPException(400, "Only SELECT queries allowed")
    try:
        with get_transaction() as cur:
            cur.execute(sql)
            rows = cur.fetchall()
        return {"count": len(rows), "rows": [dict(r) for r in rows]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# LOT TRACEABILITY — Duplicate Scanner & Merge
# ═══════════════════════════════════════════════════════════════

@app.get("/admin/lots/duplicates")
def scan_lot_duplicates(_: bool = Depends(verify_api_key)):
    """Scan for duplicate (product_id, lot_code) pairs across the lots table.
    Returns grouped results for review before merging."""
    try:
        with get_transaction() as cur:
            cur.execute("""
                SELECT p.name AS product_name, p.id AS product_id, l.lot_code,
                       COUNT(*) AS duplicate_count,
                       ARRAY_AGG(l.id ORDER BY l.created_at) AS lot_ids,
                       ARRAY_AGG(l.created_at ORDER BY l.created_at) AS created_dates
                FROM lots l
                JOIN products p ON p.id = l.product_id
                WHERE l.lot_code IS NOT NULL
                  AND COALESCE(l.status, 'active') = 'active'
                GROUP BY p.id, p.name, l.lot_code
                HAVING COUNT(*) > 1
                ORDER BY COUNT(*) DESC
            """)
            rows = cur.fetchall()

            groups = []
            for r in rows:
                groups.append({
                    "product_name": r['product_name'],
                    "product_id": r['product_id'],
                    "lot_code": r['lot_code'],
                    "duplicate_count": r['duplicate_count'],
                    "lot_ids": r['lot_ids'],
                    "created_dates": [str(d) for d in r['created_dates']]
                })

            return {
                "duplicate_groups": groups,
                "total_groups": len(groups)
            }
    except Exception as e:
        logger.error(f"Scan lot duplicates failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


class LotMergeRequest(BaseModel):
    source_lot_id: int
    target_lot_id: int
    reason: str


@app.post("/admin/lots/merge")
def merge_lots(req: LotMergeRequest, _: bool = Depends(verify_api_key)):
    """Merge source lot into target lot. Moves all transaction_lines and
    ingredient_lot_consumption references, marks source as merged.
    Both lots must belong to the same product."""
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # 1. Validate both lots exist
                cur.execute("SELECT id, product_id, lot_code, status FROM lots WHERE id = %s", (req.source_lot_id,))
                source = cur.fetchone()
                if not source:
                    raise HTTPException(404, f"Source lot ID {req.source_lot_id} not found")

                cur.execute("SELECT id, product_id, lot_code, status FROM lots WHERE id = %s", (req.target_lot_id,))
                target = cur.fetchone()
                if not target:
                    raise HTTPException(404, f"Target lot ID {req.target_lot_id} not found")

                # 2. Validate neither is already merged
                if source.get('status') == 'merged':
                    raise HTTPException(400,
                        f"Source lot {source['lot_code']} (id={req.source_lot_id}) is already merged. "
                        f"Cannot merge an already-merged lot."
                    )
                if target.get('status') == 'merged':
                    raise HTTPException(400,
                        f"Target lot {target['lot_code']} (id={req.target_lot_id}) is already merged. "
                        f"Cannot merge into an already-merged lot."
                    )

                # 3. Validate same product
                if source['product_id'] != target['product_id']:
                    raise HTTPException(400,
                        f"Cannot merge lots from different products. "
                        f"Source lot {source['lot_code']} is product_id={source['product_id']}, "
                        f"target lot {target['lot_code']} is product_id={target['product_id']}."
                    )

                # 4. Lock both lots within transaction
                cur.execute(
                    "SELECT id FROM lots WHERE id IN (%s, %s) ORDER BY id FOR UPDATE",
                    (req.source_lot_id, req.target_lot_id)
                )

                rows_moved = {}

                # 5. Move transaction_lines
                cur.execute(
                    "UPDATE transaction_lines SET lot_id = %s WHERE lot_id = %s",
                    (req.target_lot_id, req.source_lot_id)
                )
                rows_moved["transaction_lines"] = cur.rowcount

                # Move ingredient_lot_consumption
                cur.execute(
                    "UPDATE ingredient_lot_consumption SET ingredient_lot_id = %s WHERE ingredient_lot_id = %s",
                    (req.target_lot_id, req.source_lot_id)
                )
                rows_moved["ingredient_lot_consumption"] = cur.rowcount

                # 6. Mark source lot as merged
                now = get_plant_now()
                cur.execute("""
                    UPDATE lots
                    SET status = 'merged',
                        merged_into_lot_id = %s,
                        merged_at = %s,
                        merge_reason = %s
                    WHERE id = %s
                """, (req.target_lot_id, now, req.reason, req.source_lot_id))

                # 7. Recalculate target lot balance from ledger
                cur.execute("""
                    SELECT COALESCE(SUM(quantity_lb), 0) AS computed_balance
                    FROM transaction_lines
                    WHERE lot_id = %s
                """, (req.target_lot_id,))
                computed_balance = float(cur.fetchone()['computed_balance'])

                total_rows = sum(rows_moved.values())
                logger.info(
                    f"Lot merge: {source['lot_code']} (id={req.source_lot_id}) → "
                    f"{target['lot_code']} (id={req.target_lot_id}). "
                    f"Moved {total_rows} rows. Balance: {computed_balance} lb. "
                    f"Reason: {req.reason}"
                )

                return {
                    "merged": True,
                    "source_lot_id": req.source_lot_id,
                    "source_lot_code": source['lot_code'],
                    "target_lot_id": req.target_lot_id,
                    "target_lot_code": target['lot_code'],
                    "product_id": source['product_id'],
                    "rows_moved": rows_moved,
                    "target_lot_new_balance": computed_balance,
                    "audit_note": req.reason
                }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Lot merge failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# PRODUCTION REQUIREMENTS
# ═══════════════════════════════════════════════════════════════

@app.get("/production/requirements")
def production_requirements(
    product_name: str = Query(..., description="Finished good or batch product name"),
    cases: Optional[int] = Query(None, description="Number of cases (for finished goods)"),
    batches: Optional[int] = Query(None, description="Number of batches (for batch products)"),
    _: bool = Depends(verify_api_key)
):
    """Given a finished good + cases OR batch product + batches, return the full ingredient breakdown."""
    try:
        with get_transaction() as cur:
            product = resolve_product_full(cur, product_name)
            pid = product['id']
            pname = product['name']

            # Determine if this is a finished good or batch product
            cur.execute("SELECT type FROM products WHERE id = %s", (pid,))
            ptype = cur.fetchone()['type']

            batch_product_id = None
            batch_product_name = None
            num_batches = batches
            total_output_lb = None

            if ptype == 'finished':
                if not cases:
                    raise HTTPException(400, "cases parameter required for finished goods")

                # Look up the batch product from product_bom
                cur.execute("""
                    SELECT pb.component_product_id AS batch_product_id, p.name AS batch_name, p.default_batch_lb
                    FROM product_bom pb
                    JOIN products p ON p.id = pb.component_product_id
                    WHERE pb.finished_product_id = %s AND p.type = 'batch'
                """, (pid,))
                link = cur.fetchone()

                if not link:
                    raise HTTPException(404, f"No batch product linked to '{pname}'. Add a product_bom mapping.")

                batch_product_id = link['batch_product_id']
                batch_product_name = link['batch_name']
                batch_size = float(link['default_batch_lb'] or 0)

                # Calculate how many lbs needed
                case_weight = float(product.get('default_case_weight_lb') or 0)
                if case_weight <= 0:
                    raise HTTPException(400, f"No default_case_weight_lb set for '{pname}'")

                total_output_lb = cases * case_weight
                if batch_size > 0:
                    import math
                    num_batches = math.ceil(total_output_lb / batch_size)
                else:
                    raise HTTPException(400, f"No default_batch_lb set for batch product '{batch_product_name}'")

            elif ptype == 'batch':
                batch_product_id = pid
                batch_product_name = pname
                if not num_batches:
                    num_batches = 1
                batch_size = float(product.get('default_batch_lb') or 0)
                total_output_lb = num_batches * batch_size
            else:
                raise HTTPException(400, f"Product '{pname}' is type '{ptype}', expected 'finished' or 'batch'")

            # Get the BOM for the batch product
            cur.execute("""
                SELECT bf.ingredient_product_id, p.name AS ingredient_name, bf.quantity_lb,
                       COALESCE(bf.exclude_from_inventory, false) AS exclude_from_inventory
                FROM batch_formulas bf
                JOIN products p ON p.id = bf.ingredient_product_id
                WHERE bf.product_id = %s
                ORDER BY bf.quantity_lb DESC
            """, (batch_product_id,))
            formula = cur.fetchall()

            if not formula:
                raise HTTPException(404, f"No BOM found for batch product '{batch_product_name}'")

            # Check if any ingredient is itself a batch product (nested BOM, e.g. PB Banana)
            ingredients = []
            for ing in formula:
                needed = float(ing['quantity_lb']) * num_batches
                excluded = ing['exclude_from_inventory']

                # Check if this ingredient has its own BOM (nested)
                cur.execute("SELECT COUNT(*) AS cnt FROM batch_formulas WHERE product_id = %s", (ing['ingredient_product_id'],))
                has_sub_bom = cur.fetchone()['cnt'] > 0

                # Get current inventory
                cur.execute("""
                    SELECT COALESCE(SUM(tl.quantity_lb), 0) AS available
                    FROM lots l
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.product_id = %s
                """, (ing['ingredient_product_id'],))
                available = float(cur.fetchone()['available'])

                ing_data = {
                    "ingredient_id": ing['ingredient_product_id'],
                    "ingredient_name": ing['ingredient_name'],
                    "per_batch_lb": float(ing['quantity_lb']),
                    "total_needed_lb": needed,
                    "available_lb": available,
                    "sufficient": available >= needed or excluded,
                    "excluded": excluded
                }

                if has_sub_bom:
                    # Expand the sub-BOM
                    sub_batches_needed = needed  # lbs needed of this sub-batch
                    cur.execute("SELECT default_batch_lb FROM products WHERE id = %s", (ing['ingredient_product_id'],))
                    sub_batch_size = float(cur.fetchone()['default_batch_lb'] or 0)
                    if sub_batch_size > 0:
                        import math
                        sub_num_batches = math.ceil(sub_batches_needed / sub_batch_size)
                    else:
                        sub_num_batches = 1

                    cur.execute("""
                        SELECT bf.ingredient_product_id, p.name AS ingredient_name, bf.quantity_lb,
                               COALESCE(bf.exclude_from_inventory, false) AS exclude_from_inventory
                        FROM batch_formulas bf
                        JOIN products p ON p.id = bf.ingredient_product_id
                        WHERE bf.product_id = %s
                        ORDER BY bf.quantity_lb DESC
                    """, (ing['ingredient_product_id'],))
                    sub_formula = cur.fetchall()

                    sub_ingredients = []
                    for sub in sub_formula:
                        sub_needed = float(sub['quantity_lb']) * sub_num_batches
                        cur.execute("""
                            SELECT COALESCE(SUM(tl.quantity_lb), 0) AS available
                            FROM lots l LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                            WHERE l.product_id = %s
                        """, (sub['ingredient_product_id'],))
                        sub_avail = float(cur.fetchone()['available'])
                        sub_excluded = sub['exclude_from_inventory']
                        sub_ingredients.append({
                            "ingredient_id": sub['ingredient_product_id'],
                            "ingredient_name": sub['ingredient_name'],
                            "per_batch_lb": float(sub['quantity_lb']),
                            "total_needed_lb": sub_needed,
                            "available_lb": sub_avail,
                            "sufficient": sub_avail >= sub_needed or sub_excluded,
                            "excluded": sub_excluded
                        })

                    ing_data["is_sub_batch"] = True
                    ing_data["sub_batches_needed"] = sub_num_batches
                    ing_data["sub_ingredients"] = sub_ingredients

                ingredients.append(ing_data)

            all_sufficient = all(
                i['sufficient'] and all(s['sufficient'] for s in i.get('sub_ingredients', []))
                for i in ingredients
            )

            result = {
                "product_name": pname,
                "product_type": ptype,
                "batch_product": batch_product_name,
                "batches_needed": num_batches,
                "total_output_lb": total_output_lb,
                "all_ingredients_sufficient": all_sufficient,
                "ingredients": ingredients
            }

            if ptype == 'finished':
                result["cases"] = cases
                result["case_weight_lb"] = case_weight

            return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Production requirements failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# PRODUCTION DAY SUMMARY
# ═══════════════════════════════════════════════════════════════

@app.get("/production/day-summary")
def production_day_summary(
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format; defaults to today"),
    _: bool = Depends(verify_api_key)
):
    """Return all make/pack/adjust activity for a given day, grouped by product with lot-level detail."""
    try:
        with get_transaction() as cur:
            if date:
                try:
                    target_date = datetime.strptime(date, "%Y-%m-%d").date()
                except ValueError:
                    raise HTTPException(400, "date must be YYYY-MM-DD format")
            else:
                target_date = get_plant_now().date()

            day_start = datetime(target_date.year, target_date.month, target_date.day,
                                 tzinfo=PLANT_TIMEZONE)
            day_end = day_start + timedelta(days=1)

            # ── Batch production (make transactions) ──
            cur.execute("""
                SELECT p.id as product_id, p.name as product_name,
                       l.id as lot_id, l.lot_code,
                       tl.quantity_lb, t.id as transaction_id
                FROM transactions t
                JOIN transaction_lines tl ON tl.transaction_id = t.id
                JOIN products p ON p.id = tl.product_id
                JOIN lots l ON l.id = tl.lot_id
                WHERE t.type = 'make'
                  AND t.timestamp >= %s AND t.timestamp < %s
                  AND tl.quantity_lb > 0
                ORDER BY t.timestamp
            """, (day_start, day_end))
            make_rows = cur.fetchall()

            # ── Pack consumption from batch lots (negative lines on batch products) ──
            cur.execute("""
                SELECT l.id as lot_id, l.lot_code,
                       ABS(tl.quantity_lb) as packed_lb,
                       t.id as transaction_id
                FROM transactions t
                JOIN transaction_lines tl ON tl.transaction_id = t.id
                JOIN lots l ON l.id = tl.lot_id
                WHERE t.type = 'pack'
                  AND t.timestamp >= %s AND t.timestamp < %s
                  AND tl.quantity_lb < 0
                ORDER BY t.timestamp
            """, (day_start, day_end))
            pack_consume_rows = cur.fetchall()

            # ── Pack output (finished goods produced) ──
            cur.execute("""
                SELECT p.id as product_id, p.name as product_name,
                       l.lot_code, tl.quantity_lb,
                       t.notes
                FROM transactions t
                JOIN transaction_lines tl ON tl.transaction_id = t.id
                JOIN products p ON p.id = tl.product_id
                JOIN lots l ON l.id = tl.lot_id
                WHERE t.type = 'pack'
                  AND t.timestamp >= %s AND t.timestamp < %s
                  AND tl.quantity_lb > 0
                ORDER BY t.timestamp
            """, (day_start, day_end))
            pack_output_rows = cur.fetchall()

            # ── Adjustments for the day ──
            cur.execute("""
                SELECT p.name as product_name, l.id as lot_id, l.lot_code,
                       tl.quantity_lb as adjustment_lb,
                       t.adjust_reason as reason
                FROM transactions t
                JOIN transaction_lines tl ON tl.transaction_id = t.id
                JOIN products p ON p.id = tl.product_id
                JOIN lots l ON l.id = tl.lot_id
                WHERE t.type = 'adjust'
                  AND t.timestamp >= %s AND t.timestamp < %s
                ORDER BY t.timestamp
            """, (day_start, day_end))
            adjust_rows = cur.fetchall()

            # ── Build batch lot summaries ──
            # Collect all lot_ids touched by make today
            batch_lots = {}  # lot_id -> {lot_code, product_name, produced_lb, packed_lb, adjusted_lb}
            for r in make_rows:
                lid = r['lot_id']
                if lid not in batch_lots:
                    batch_lots[lid] = {
                        "lot_code": r['lot_code'],
                        "product_id": r['product_id'],
                        "product_name": r['product_name'],
                        "produced_lb": 0.0,
                        "packed_lb": 0.0,
                        "adjusted_lb": 0.0,
                    }
                batch_lots[lid]["produced_lb"] += float(r['quantity_lb'])

            # Add pack consumption
            for r in pack_consume_rows:
                lid = r['lot_id']
                if lid in batch_lots:
                    batch_lots[lid]["packed_lb"] += float(r['packed_lb'])

            # Add adjustments
            for r in adjust_rows:
                lid = r['lot_id']
                if lid in batch_lots:
                    batch_lots[lid]["adjusted_lb"] += float(r['adjustment_lb'])

            # Get current on-hand for each batch lot
            if batch_lots:
                lot_ids = list(batch_lots.keys())
                cur.execute("""
                    SELECT l.id as lot_id, COALESCE(SUM(tl.quantity_lb), 0) as on_hand
                    FROM lots l
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.id = ANY(%s)
                    GROUP BY l.id
                """, (lot_ids,))
                for row in cur.fetchall():
                    if row['lot_id'] in batch_lots:
                        batch_lots[row['lot_id']]["current_on_hand_lb"] = float(row['on_hand'])

            # Group batch lots by product
            products_map = {}
            for lid, info in batch_lots.items():
                pid = info["product_id"]
                if pid not in products_map:
                    products_map[pid] = {
                        "product_id": pid,
                        "product_name": info["product_name"],
                        "total_produced_lb": 0.0,
                        "total_packed_lb": 0.0,
                        "lots": []
                    }
                products_map[pid]["total_produced_lb"] += info["produced_lb"]
                products_map[pid]["total_packed_lb"] += info["packed_lb"]
                products_map[pid]["lots"].append({
                    "lot_code": info["lot_code"],
                    "produced_lb": round(info["produced_lb"], 2),
                    "packed_lb": round(info["packed_lb"], 2),
                    "adjusted_lb": round(info["adjusted_lb"], 2),
                    "current_on_hand_lb": round(info.get("current_on_hand_lb", 0), 2)
                })

            # ── Build finished goods section ──
            finished_goods = []
            for r in pack_output_rows:
                finished_goods.append({
                    "product_name": r['product_name'],
                    "lot_code": r['lot_code'],
                    "total_lb": float(r['quantity_lb']),
                })

            # ── Adjustments list ──
            adjustments = [
                {
                    "product_name": r['product_name'],
                    "lot_code": r['lot_code'],
                    "adjustment_lb": float(r['adjustment_lb']),
                    "reason": r['reason']
                }
                for r in adjust_rows
            ]

            return {
                "date": str(target_date),
                "day_name": target_date.strftime("%A"),
                "batch_products": list(products_map.values()),
                "finished_goods": finished_goods,
                "adjustments": adjustments
            }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Production day-summary failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# PRODUCTION SCHEDULING — 7-Day Tactical Scheduler
# ═══════════════════════════════════════════════════════════════

def _build_schedule_calendar(start_date: date, horizon_days: int, friday_modifier: float):
    """Build list of working days with capacity modifiers."""
    days = []
    current = start_date
    working_days_added = 0
    max_scan = horizon_days * 3  # scan enough calendar days
    scanned = 0
    while working_days_added < horizon_days and scanned < max_scan:
        dow = current.strftime("%A")
        if dow in ("Saturday", "Sunday"):
            current += timedelta(days=1)
            scanned += 1
            continue
        modifier = friday_modifier if dow == "Friday" else 1.0
        days.append({
            "date": current,
            "day_of_week": dow,
            "capacity_modifier": modifier,
        })
        working_days_added += 1
        current += timedelta(days=1)
        scanned += 1
    return days


def _load_line_config(cur):
    """Load production lines and their capacity modes from DB."""
    cur.execute("""
        SELECT pl.id, pl.name, pl.line_code, pl.active,
               json_agg(json_build_object(
                   'mode_id', lcm.id, 'mode_name', lcm.mode_name,
                   'workers_required', lcm.workers_required,
                   'batches_per_day', lcm.batches_per_day,
                   'pallets_per_day', lcm.pallets_per_day,
                   'bags_per_day', lcm.bags_per_day,
                   'pack_size_lb', lcm.pack_size_lb,
                   'is_default', lcm.is_default
               ) ORDER BY lcm.is_default DESC, lcm.workers_required) AS modes
        FROM production_lines pl
        LEFT JOIN line_capacity_modes lcm ON lcm.line_id = pl.id
        WHERE pl.active = true
        GROUP BY pl.id, pl.name, pl.line_code, pl.active
        ORDER BY pl.name
    """)
    lines = {}
    for row in cur.fetchall():
        lines[row['line_code']] = {
            'id': row['id'],
            'name': row['name'],
            'line_code': row['line_code'],
            'modes': row['modes'] or [],
        }
    return lines


def _load_product_line_map(cur):
    """Load product→line assignments. Returns dict: product_id → line_code."""
    cur.execute("""
        SELECT pla.product_id, pl.line_code
        FROM product_line_assignments pla
        JOIN production_lines pl ON pl.id = pla.line_id
    """)
    return {row['product_id']: row['line_code'] for row in cur.fetchall()}


def _load_demand(cur, horizon_end: date):
    """Load open/confirmed sales orders within or overdue relative to horizon."""
    cur.execute("""
        SELECT so.id AS order_id, so.order_number, so.requested_ship_date, so.status,
               sol.id AS line_id, sol.product_id, sol.quantity_lb, sol.quantity_shipped_lb,
               p.name AS product_name, p.type AS product_type
        FROM sales_orders so
        JOIN sales_order_lines sol ON sol.sales_order_id = so.id
        JOIN products p ON p.id = sol.product_id
        WHERE so.status IN ('confirmed', 'in_production', 'ready')
          AND sol.line_status IN ('pending', 'partial')
          AND (so.requested_ship_date IS NULL OR so.requested_ship_date <= %s)
        ORDER BY so.requested_ship_date ASC NULLS LAST, so.id ASC
    """, (horizon_end,))
    return cur.fetchall()


def _load_finished_inventory(cur):
    """Load on-hand inventory for finished and batch products."""
    cur.execute("""
        SELECT p.id AS product_id, p.name AS product_name,
               COALESCE(SUM(tl.quantity_lb), 0) AS on_hand_lb
        FROM products p
        LEFT JOIN lots l ON l.product_id = p.id
        LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
        WHERE p.active = true AND p.type IN ('finished', 'batch')
        GROUP BY p.id, p.name
    """)
    return {row['product_id']: float(row['on_hand_lb']) for row in cur.fetchall()}


def _load_ingredient_inventory(cur):
    """Load on-hand inventory for ingredient products."""
    cur.execute("""
        SELECT p.id AS product_id, p.name AS product_name,
               COALESCE(SUM(tl.quantity_lb), 0) AS on_hand_lb
        FROM products p
        LEFT JOIN lots l ON l.product_id = p.id
        LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
        WHERE p.active = true AND p.type = 'ingredient'
        GROUP BY p.id, p.name
    """)
    return {row['product_id']: {'name': row['product_name'], 'on_hand': float(row['on_hand_lb'])} for row in cur.fetchall()}


def _load_bom_structure(cur):
    """Load full BOM: finished→batch (product_bom) and batch→ingredients (batch_formulas).
    Returns:
      fg_to_batch: {finished_product_id: {batch_product_id, batch_name, quantity, uom}}
      batch_to_ingredients: {batch_product_id: [{ingredient_product_id, ingredient_name, quantity_lb, exclude}]}
      batch_sizes: {batch_product_id: default_batch_lb}
    """
    # Finished good → batch product mapping
    cur.execute("""
        SELECT pb.finished_product_id, pb.component_product_id, p.name AS component_name,
               p.type AS component_type, pb.quantity, pb.uom, p.default_batch_lb
        FROM product_bom pb
        JOIN products p ON p.id = pb.component_product_id
    """)
    fg_to_batch = {}
    for row in cur.fetchall():
        fid = row['finished_product_id']
        if fid not in fg_to_batch:
            fg_to_batch[fid] = []
        fg_to_batch[fid].append({
            'component_product_id': row['component_product_id'],
            'component_name': row['component_name'],
            'component_type': row['component_type'],
            'quantity': float(row['quantity'] or 1),
            'uom': row['uom'] or 'unit',
            'default_batch_lb': float(row['default_batch_lb']) if row['default_batch_lb'] else None,
        })

    # Batch product → ingredients
    cur.execute("""
        SELECT bf.product_id, bf.ingredient_product_id, p.name AS ingredient_name,
               bf.quantity_lb, COALESCE(bf.exclude_from_inventory, false) AS exclude_from_inventory
        FROM batch_formulas bf
        JOIN products p ON p.id = bf.ingredient_product_id
        ORDER BY bf.product_id, bf.quantity_lb DESC
    """)
    batch_to_ingredients = {}
    for row in cur.fetchall():
        bid = row['product_id']
        if bid not in batch_to_ingredients:
            batch_to_ingredients[bid] = []
        batch_to_ingredients[bid].append({
            'ingredient_product_id': row['ingredient_product_id'],
            'ingredient_name': row['ingredient_name'],
            'quantity_lb': float(row['quantity_lb']),
            'exclude': row['exclude_from_inventory'],
        })

    # Batch sizes
    cur.execute("SELECT id, default_batch_lb FROM products WHERE type = 'batch' AND active = true AND default_batch_lb IS NOT NULL")
    batch_sizes = {row['id']: float(row['default_batch_lb']) for row in cur.fetchall()}

    return fg_to_batch, batch_to_ingredients, batch_sizes


def _simulated_allocation(demand_rows, inventory, fg_to_batch, batch_sizes, product_line_map):
    """
    Walk demand in ship-date order. For each order line, allocate from available
    finished goods inventory. Whatever remains becomes a production requirement.
    Returns list of production requirements (what to make).
    """
    available = dict(inventory)  # copy — will be mutated during simulation
    production_reqs = []  # [{product_id, product_name, batch_product_id, batch_name, needed_lb, batches, overproduction_lb, order_numbers, line_code}]

    # Group demand by product to consolidate
    product_demand = {}  # product_id → [{order_number, needed_lb, ship_date}]
    for row in demand_rows:
        pid = row['product_id']
        remaining = float(row['quantity_lb']) - float(row['quantity_shipped_lb'] or 0)
        if remaining <= 0:
            continue
        if pid not in product_demand:
            product_demand[pid] = []
        product_demand[pid].append({
            'order_number': row['order_number'],
            'needed_lb': remaining,
            'ship_date': row['requested_ship_date'],
            'product_name': row['product_name'],
            'product_type': row['product_type'],
        })

    # Process each product's demand
    for pid, demands in product_demand.items():
        total_needed = sum(d['needed_lb'] for d in demands)
        on_hand = available.get(pid, 0)

        # Allocate from inventory
        allocated = min(on_hand, total_needed)
        available[pid] = on_hand - allocated
        net_need = total_needed - allocated

        if net_need <= 0:
            continue

        order_numbers = [d['order_number'] for d in demands]
        product_name = demands[0]['product_name']
        product_type = demands[0]['product_type']
        earliest_ship = min((d['ship_date'] for d in demands if d['ship_date']), default=None)

        # Determine what to produce
        if product_type == 'finished':
            # Look up batch product from BOM
            bom_components = fg_to_batch.get(pid, [])
            batch_component = next((c for c in bom_components if c['component_type'] == 'batch'), None)

            if not batch_component:
                # No BOM — can't schedule production, will be flagged
                production_reqs.append({
                    'product_id': pid,
                    'product_name': product_name,
                    'product_type': 'finished',
                    'batch_product_id': None,
                    'batch_name': None,
                    'needed_lb': net_need,
                    'batches': None,
                    'batch_size_lb': None,
                    'overproduction_lb': 0,
                    'order_numbers': order_numbers,
                    'earliest_ship_date': earliest_ship,
                    'line_code': product_line_map.get(pid),
                    'warning': f"No batch product in BOM for '{product_name}'",
                })
                continue

            batch_pid = batch_component['component_product_id']
            batch_name = batch_component['component_name']
            batch_size = batch_sizes.get(batch_pid, 0)

            if batch_size <= 0:
                production_reqs.append({
                    'product_id': pid,
                    'product_name': product_name,
                    'product_type': 'finished',
                    'batch_product_id': batch_pid,
                    'batch_name': batch_name,
                    'needed_lb': net_need,
                    'batches': None,
                    'batch_size_lb': 0,
                    'overproduction_lb': 0,
                    'order_numbers': order_numbers,
                    'earliest_ship_date': earliest_ship,
                    'line_code': product_line_map.get(batch_pid),
                    'warning': f"No default_batch_lb for '{batch_name}'",
                })
                continue

            # Also check batch inventory — maybe we have batch product on hand
            batch_on_hand = available.get(batch_pid, 0)
            net_need_after_batch = max(0, net_need - batch_on_hand)
            if batch_on_hand > 0:
                used = min(batch_on_hand, net_need)
                available[batch_pid] = batch_on_hand - used

            if net_need_after_batch <= 0:
                continue

            num_batches = math.ceil(net_need_after_batch / batch_size)
            total_output = num_batches * batch_size
            overproduction = total_output - net_need_after_batch

            production_reqs.append({
                'product_id': pid,
                'product_name': product_name,
                'product_type': 'finished',
                'batch_product_id': batch_pid,
                'batch_name': batch_name,
                'needed_lb': net_need_after_batch,
                'batches': num_batches,
                'batch_size_lb': batch_size,
                'overproduction_lb': round(overproduction, 2),
                'order_numbers': order_numbers,
                'earliest_ship_date': earliest_ship,
                'line_code': product_line_map.get(batch_pid),
                'warning': None,
            })

        elif product_type == 'batch':
            # Direct batch product demand
            batch_size = batch_sizes.get(pid, 0)
            line_code = product_line_map.get(pid)

            if batch_size <= 0:
                production_reqs.append({
                    'product_id': pid,
                    'product_name': product_name,
                    'product_type': 'batch',
                    'batch_product_id': pid,
                    'batch_name': product_name,
                    'needed_lb': net_need,
                    'batches': None,
                    'batch_size_lb': 0,
                    'overproduction_lb': 0,
                    'order_numbers': order_numbers,
                    'earliest_ship_date': earliest_ship,
                    'line_code': line_code,
                    'warning': f"No default_batch_lb for '{product_name}'",
                })
                continue

            num_batches = math.ceil(net_need / batch_size)
            total_output = num_batches * batch_size
            overproduction = total_output - net_need

            production_reqs.append({
                'product_id': pid,
                'product_name': product_name,
                'product_type': 'batch',
                'batch_product_id': pid,
                'batch_name': product_name,
                'needed_lb': net_need,
                'batches': num_batches,
                'batch_size_lb': batch_size,
                'overproduction_lb': round(overproduction, 2),
                'order_numbers': order_numbers,
                'earliest_ship_date': earliest_ship,
                'line_code': line_code,
                'warning': None,
            })

    return production_reqs


def _explode_ingredients(production_reqs, batch_to_ingredients, ingredient_inventory):
    """Calculate total ingredient needs and check for shortages."""
    ingredient_needs = {}  # ingredient_product_id → total_lb_needed

    for req in production_reqs:
        if req['batches'] is None or req['batch_product_id'] is None:
            continue
        formula = batch_to_ingredients.get(req['batch_product_id'], [])
        for ing in formula:
            if ing['exclude']:
                continue
            iid = ing['ingredient_product_id']
            needed = ing['quantity_lb'] * req['batches']
            ingredient_needs[iid] = ingredient_needs.get(iid, 0) + needed

    # Check against inventory
    ingredient_summary = []
    for iid, needed in sorted(ingredient_needs.items(), key=lambda x: x[1], reverse=True):
        info = ingredient_inventory.get(iid, {'name': f'Unknown ({iid})', 'on_hand': 0})
        on_hand = info['on_hand']
        shortage = max(0, needed - on_hand)
        ingredient_summary.append({
            'ingredient_name': info['name'],
            'ingredient_id': iid,
            'required_lb': round(needed, 2),
            'on_hand_lb': round(on_hand, 2),
            'shortage_lb': round(shortage, 2),
            'status': '⚠️ Ingredient Risk' if shortage > 0 else '✅ OK',
        })

    return ingredient_summary


def _schedule_runs_to_days(production_reqs, calendar_days, line_config, total_workers, strategy='earliest'):
    """
    Assign production runs to days respecting capacity and labor constraints.
    strategy: 'earliest' = pull forward, 'latest' = push back (closer to ship date)
    Returns (scheduled_days, unscheduled_orders).
    """
    # Initialize day structures
    day_schedules = []
    for day_info in calendar_days:
        day_sched = {
            'date': day_info['date'].isoformat(),
            'day_of_week': day_info['day_of_week'],
            'capacity_modifier': day_info['capacity_modifier'],
            'total_labor_used': 0,
            'lines': {},
        }
        for lc, linfo in line_config.items():
            day_sched['lines'][lc] = {
                'line_name': linfo['name'],
                'workers_assigned': 0,
                'runs': [],
                'warnings': [],
                'remaining_batches': None,  # will be set when line activated
                'remaining_bags': None,
                'remaining_pallets': None,
            }
        day_schedules.append(day_sched)

    unscheduled = []

    # Sort reqs: by earliest_ship_date for 'earliest', reverse for 'latest'
    sorted_reqs = sorted(
        production_reqs,
        key=lambda r: (r['earliest_ship_date'] or date(2099, 1, 1), r.get('needed_lb', 0)),
        reverse=(strategy == 'latest')
    )

    for req in sorted_reqs:
        if req['batches'] is None or req['batches'] <= 0:
            if req.get('warning'):
                unscheduled.append({
                    'order_numbers': req['order_numbers'],
                    'product_name': req['product_name'],
                    'reason': req['warning'],
                })
            continue

        line_code = req['line_code']
        if not line_code or line_code not in line_config:
            unscheduled.append({
                'order_numbers': req['order_numbers'],
                'product_name': req['product_name'],
                'reason': f"No production line assigned for '{req.get('batch_name') or req['product_name']}'",
            })
            continue

        linfo = line_config[line_code]
        modes = linfo['modes']
        if not modes:
            unscheduled.append({
                'order_numbers': req['order_numbers'],
                'product_name': req['product_name'],
                'reason': f"No capacity modes configured for line '{linfo['name']}'",
            })
            continue

        remaining_batches = req['batches']
        day_indices = range(len(day_schedules)) if strategy == 'earliest' else reversed(range(len(day_schedules)))

        for di in day_indices:
            if remaining_batches <= 0:
                break

            day = day_schedules[di]
            modifier = day['capacity_modifier']
            line_day = day['lines'][line_code]

            # Pick the best capacity mode that fits labor
            available_labor = total_workers - day['total_labor_used']

            # If this line already has workers assigned today, use that mode
            if line_day['workers_assigned'] > 0:
                # Line already active — use remaining capacity
                can_do = line_day.get('remaining_batches') or 0
                if can_do <= 0:
                    continue

                run_batches = min(remaining_batches, can_do)
                batch_size = req['batch_size_lb']
                run_qty = run_batches * batch_size

                line_day['runs'].append({
                    'product_name': req.get('batch_name') or req['product_name'],
                    'product_id': req.get('batch_product_id') or req['product_id'],
                    'batches': run_batches,
                    'quantity_lb': round(run_qty, 2),
                    'for_orders': req['order_numbers'],
                    'overproduction_lb': round(req['overproduction_lb'], 2) if remaining_batches - run_batches <= 0 else 0,
                    'overproduction_reason': 'Batch Rounding' if (remaining_batches - run_batches <= 0 and req['overproduction_lb'] > 0) else None,
                })
                line_day['remaining_batches'] = can_do - run_batches
                remaining_batches -= run_batches
                continue

            # Line not active yet today — need to activate with a mode
            best_mode = None
            for mode in modes:
                w = mode['workers_required']
                if w <= available_labor:
                    best_mode = mode
                    break  # modes sorted by default first, then lowest workers

            if not best_mode:
                continue  # can't fit any mode on this day

            workers = best_mode['workers_required']
            raw_capacity = best_mode.get('batches_per_day') or 0
            day_capacity = max(1, int(raw_capacity * modifier))

            run_batches = min(remaining_batches, day_capacity)
            batch_size = req['batch_size_lb']
            run_qty = run_batches * batch_size

            # Activate the line
            day['total_labor_used'] += workers
            line_day['workers_assigned'] = workers
            line_day['remaining_batches'] = day_capacity - run_batches

            line_day['runs'].append({
                'product_name': req.get('batch_name') or req['product_name'],
                'product_id': req.get('batch_product_id') or req['product_id'],
                'batches': run_batches,
                'quantity_lb': round(run_qty, 2),
                'for_orders': req['order_numbers'],
                'overproduction_lb': round(req['overproduction_lb'], 2) if remaining_batches - run_batches <= 0 else 0,
                'overproduction_reason': 'Batch Rounding' if (remaining_batches - run_batches <= 0 and req['overproduction_lb'] > 0) else None,
            })
            remaining_batches -= run_batches

        if remaining_batches > 0:
            unscheduled.append({
                'order_numbers': req['order_numbers'],
                'product_name': req.get('batch_name') or req['product_name'],
                'reason': f"Insufficient capacity in window ({remaining_batches} batches remaining)",
            })

    # Format output
    formatted_days = []
    for day in day_schedules:
        lines_out = []
        for lc in ['granola', 'coconut', 'bulk_pack', 'pouch']:
            if lc in day['lines']:
                ld = day['lines'][lc]
                lines_out.append({
                    'line_name': ld['line_name'],
                    'workers_assigned': ld['workers_assigned'],
                    'runs': ld['runs'],
                    'warnings': ld['warnings'],
                })
        formatted_days.append({
            'date': day['date'],
            'day_of_week': day['day_of_week'],
            'capacity_modifier': day['capacity_modifier'],
            'total_labor_used': day['total_labor_used'],
            'lines': lines_out,
        })

    return formatted_days, unscheduled


def _handle_schedule_suggest(body: dict):
    """Generate a proposed 7-day production schedule based on open orders, inventory, and capacity."""
    start_date_str = body.get('start_date')
    horizon_days = body.get('horizon_days', 7)
    total_workers = body.get('total_workers', 10)
    friday_modifier = body.get('friday_modifier', 0.5)

    if start_date_str:
        try:
            start = date.fromisoformat(start_date_str)
        except ValueError:
            raise HTTPException(400, f"Invalid start_date format: '{start_date_str}'. Use YYYY-MM-DD.")
    else:
        start = get_plant_now().date() + timedelta(days=1)

    with get_transaction() as cur:
        # 1. Load config
        line_config = _load_line_config(cur)
        product_line_map = _load_product_line_map(cur)

        # 2. Build calendar
        calendar_days = _build_schedule_calendar(start, horizon_days, friday_modifier)
        if not calendar_days:
            return {"error": "No working days in the specified horizon"}
        horizon_end = calendar_days[-1]['date']

        # 3. Load demand
        demand = _load_demand(cur, horizon_end)

        # 4. Load supply
        inventory = _load_finished_inventory(cur)
        ingredient_inv = _load_ingredient_inventory(cur)

        # 5. Load BOM structure
        fg_to_batch, batch_to_ingredients, batch_sizes = _load_bom_structure(cur)

        # 6. Simulated allocation → net requirements
        production_reqs = _simulated_allocation(
            demand, inventory, fg_to_batch, batch_sizes, product_line_map
        )

        if not production_reqs:
            return {
                "schedule_id": str(uuid.uuid4()),
                "horizon": {"start": start.isoformat(), "end": horizon_end.isoformat()},
                "total_workers_available": total_workers,
                "message": "No production needed — all open orders covered by current inventory.",
                "scenarios": [],
            }

        # 7. Explode ingredients
        ingredient_summary = _explode_ingredients(
            production_reqs, batch_to_ingredients, ingredient_inv
        )

        # 8. Schedule runs — try earliest-first
        days_earliest, unscheduled_earliest = _schedule_runs_to_days(
            production_reqs, calendar_days, line_config, total_workers, strategy='earliest'
        )

        scenarios = []

        if not unscheduled_earliest:
            # Everything fits — single recommended scenario
            scenarios.append({
                'scenario_name': 'Recommended',
                'days': days_earliest,
                'ingredient_summary': [i for i in ingredient_summary if i['shortage_lb'] > 0],
                'unscheduled_orders': [],
            })
        else:
            # Conflicts — generate two scenarios
            scenarios.append({
                'scenario_name': 'Scenario A: Pull Production Earlier',
                'days': days_earliest,
                'ingredient_summary': [i for i in ingredient_summary if i['shortage_lb'] > 0],
                'unscheduled_orders': unscheduled_earliest,
            })

            # Scenario B: push production later
            days_latest, unscheduled_latest = _schedule_runs_to_days(
                production_reqs, calendar_days, line_config, total_workers, strategy='latest'
            )
            scenarios.append({
                'scenario_name': 'Scenario B: Push Production Later',
                'days': days_latest,
                'ingredient_summary': [i for i in ingredient_summary if i['shortage_lb'] > 0],
                'unscheduled_orders': unscheduled_latest,
            })

        # Build summary of production requirements for context
        production_summary = []
        for req_item in production_reqs:
            production_summary.append({
                'product': req_item.get('batch_name') or req_item['product_name'],
                'for_finished_good': req_item['product_name'] if req_item['product_type'] == 'finished' else None,
                'needed_lb': round(req_item['needed_lb'], 2),
                'batches': req_item['batches'],
                'total_output_lb': round(req_item['batches'] * req_item['batch_size_lb'], 2) if req_item['batches'] and req_item['batch_size_lb'] else None,
                'overproduction_lb': req_item['overproduction_lb'],
                'line': req_item['line_code'],
                'for_orders': req_item['order_numbers'],
                'warning': req_item.get('warning'),
            })

        return {
            "schedule_id": str(uuid.uuid4()),
            "horizon": {"start": start.isoformat(), "end": horizon_end.isoformat()},
            "total_workers_available": total_workers,
            "open_orders_in_window": len(set(r['order_number'] for r in demand)),
            "production_requirements": production_summary,
            "scenarios": scenarios,
            "all_ingredient_status": ingredient_summary,
        }


def _handle_schedule_confirm(body: dict):
    """Confirm a proposed (or edited) production schedule — saves to production_schedule table."""
    runs_data = body.get('runs')
    if not runs_data or not isinstance(runs_data, list):
        raise HTTPException(400, "confirm action requires a 'runs' array")

    with get_db_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Load line lookup
            cur.execute("SELECT id, line_code FROM production_lines WHERE active = true")
            line_lookup = {r['line_code']: r['id'] for r in cur.fetchall()}

            confirmed_ids = []
            for run in runs_data:
                line_code = run.get('line_code')
                line_id = line_lookup.get(line_code)
                if not line_id:
                    raise HTTPException(400, f"Unknown line_code: '{line_code}'")

                run_date_str = run.get('date')
                try:
                    run_date = date.fromisoformat(run_date_str)
                except (ValueError, TypeError):
                    raise HTTPException(400, f"Invalid date: '{run_date_str}'")

                product_id = run.get('product_id')
                if not product_id:
                    raise HTTPException(400, "Each run requires a 'product_id'")

                cur.execute("""
                    INSERT INTO production_schedule
                        (schedule_date, line_id, product_id, planned_batches, planned_quantity_lb,
                         planned_bags, workers_assigned, status, linked_order_numbers,
                         overproduction_lb, overproduction_reason, notes, confirmed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, 'confirmed', %s, %s, %s, %s, NOW())
                    ON CONFLICT (schedule_date, line_id, product_id)
                    DO UPDATE SET
                        planned_batches = EXCLUDED.planned_batches,
                        planned_quantity_lb = EXCLUDED.planned_quantity_lb,
                        planned_bags = EXCLUDED.planned_bags,
                        workers_assigned = EXCLUDED.workers_assigned,
                        linked_order_numbers = EXCLUDED.linked_order_numbers,
                        overproduction_lb = EXCLUDED.overproduction_lb,
                        overproduction_reason = EXCLUDED.overproduction_reason,
                        notes = EXCLUDED.notes,
                        status = 'confirmed',
                        confirmed_at = NOW()
                    RETURNING id
                """, (
                    run_date, line_id, product_id,
                    run.get('planned_batches'), run.get('planned_quantity_lb'),
                    run.get('planned_bags'), run.get('workers_assigned', 0),
                    run.get('linked_order_numbers'),
                    run.get('overproduction_lb', 0), run.get('overproduction_reason'),
                    run.get('notes'),
                ))
                row = cur.fetchone()
                if row:
                    confirmed_ids.append(row['id'])

            conn.commit()

    return {
        "confirmed": True,
        "runs_saved": len(confirmed_ids),
        "schedule_ids": confirmed_ids,
    }


def _handle_schedule_current(body: dict):
    """View the current confirmed production schedule."""
    start_date_str = body.get('start_date')
    days = body.get('days', 7)

    if start_date_str:
        try:
            start = date.fromisoformat(start_date_str)
        except ValueError:
            raise HTTPException(400, f"Invalid start_date: '{start_date_str}'")
    else:
        start = get_plant_now().date()

    end = start + timedelta(days=days)

    with get_transaction() as cur:
        cur.execute("""
            SELECT ps.id, ps.schedule_date, ps.planned_batches, ps.planned_quantity_lb,
                   ps.planned_bags, ps.workers_assigned, ps.status,
                   ps.linked_order_numbers, ps.overproduction_lb, ps.overproduction_reason,
                   ps.notes, ps.confirmed_at,
                   pl.name AS line_name, pl.line_code,
                   p.name AS product_name, p.id AS product_id
            FROM production_schedule ps
            JOIN production_lines pl ON pl.id = ps.line_id
            JOIN products p ON p.id = ps.product_id
            WHERE ps.schedule_date >= %s AND ps.schedule_date < %s
              AND ps.status != 'cancelled'
            ORDER BY ps.schedule_date, pl.name, p.name
        """, (start, end))
        rows = cur.fetchall()

    # Group by date
    by_date = {}
    for r in rows:
        d = r['schedule_date'].isoformat()
        if d not in by_date:
            by_date[d] = {
                'date': d,
                'day_of_week': r['schedule_date'].strftime("%A"),
                'runs': [],
                'total_workers': 0,
            }
        by_date[d]['runs'].append({
            'schedule_id': r['id'],
            'line_name': r['line_name'],
            'line_code': r['line_code'],
            'product_name': r['product_name'],
            'product_id': r['product_id'],
            'planned_batches': r['planned_batches'],
            'planned_quantity_lb': float(r['planned_quantity_lb']) if r['planned_quantity_lb'] else None,
            'planned_bags': r['planned_bags'],
            'workers_assigned': r['workers_assigned'],
            'status': r['status'],
            'linked_orders': r['linked_order_numbers'],
            'overproduction_lb': float(r['overproduction_lb']) if r['overproduction_lb'] else 0,
            'notes': r['notes'],
        })
        by_date[d]['total_workers'] += r['workers_assigned']

    return {
        "period": {"start": start.isoformat(), "end": end.isoformat()},
        "days": list(by_date.values()),
        "total_runs": len(rows),
    }


@app.post("/schedule")
def schedule_dispatch(request_body: dict, _: bool = Depends(verify_api_key)):
    """Unified scheduling endpoint. Dispatches based on the 'action' field.

    Actions:
      - suggest:  Generate a proposed schedule (optional: start_date, horizon_days, total_workers, friday_modifier)
      - confirm:  Confirm/save a schedule (requires: runs[])
      - current:  View the confirmed schedule (optional: start_date, days)
    """
    action = request_body.get("action")
    if not action:
        raise HTTPException(400, "Missing required field: 'action'. Use 'suggest', 'confirm', or 'current'.")

    action = action.strip().lower()

    try:
        if action == "suggest":
            return _handle_schedule_suggest(request_body)
        elif action == "confirm":
            return _handle_schedule_confirm(request_body)
        elif action == "current":
            return _handle_schedule_current(request_body)
        else:
            raise HTTPException(400, f"Unknown action: '{action}'. Use 'suggest', 'confirm', or 'current'.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Schedule ({action}) failed: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# STATIC FILE SERVING — Dashboard UI (must be LAST)
# ═══════════════════════════════════════════════════════════════

_dashboard_dir = pathlib.Path(__file__).parent / "dashboard"
if _dashboard_dir.is_dir():
    app.mount("/dashboard", StaticFiles(directory=str(_dashboard_dir), html=True), name="dashboard-ui")
