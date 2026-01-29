from fastapi import FastAPI, HTTPException, Header, Query, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime, date, timezone
from zoneinfo import ZoneInfo
from contextlib import contextmanager
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from psycopg2 import errors as pg_errors
import os
import re
import logging
import secrets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("factory_ledger")

app = FastAPI(title="Factory Ledger System", version="2.0.0")

DATABASE_URL = (os.getenv("DATABASE_URL") or "").strip()
API_KEY = (os.getenv("API_KEY") or "").strip()

PLANT_TIMEZONE = ZoneInfo("America/New_York")
TIMEZONE_LABEL = "ET"

# Constants
WEIGHT_TOLERANCE_LB = 0.01
MAX_PRODUCT_MATCHES = 5
MAX_LOT_CODE_RETRIES = 5
DB_POOL_MIN_CONN = 1
DB_POOL_MAX_CONN = 20

# Connection pool (initialized on startup)
db_pool: Optional[pool.SimpleConnectionPool] = None


# ═══════════════════════════════════════════════════════════════
# DATABASE CONNECTION MANAGEMENT
# ═══════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup_event():
    """Initialize database connection pool on startup."""
    global db_pool
    try:
        db_pool = pool.SimpleConnectionPool(
            DB_POOL_MIN_CONN,
            DB_POOL_MAX_CONN,
            DATABASE_URL,
            connect_timeout=5
        )
        logger.info(f"Database pool initialized (min={DB_POOL_MIN_CONN}, max={DB_POOL_MAX_CONN})")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise


@app.on_event("shutdown")
def shutdown_event():
    """Close all database connections on shutdown."""
    global db_pool
    if db_pool:
        db_pool.closeall()
        logger.info("Database pool closed")


@contextmanager
def get_db_connection(autocommit: bool = True):
    """
    Get a database connection from the pool.
    
    Args:
        autocommit: If True, connection auto-commits. Set False for transactions.
    """
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")
    
    conn = db_pool.getconn()
    try:
        conn.autocommit = autocommit
        yield conn
    finally:
        db_pool.putconn(conn)


@contextmanager
def get_transaction():
    """
    Get a database connection configured for a transaction.
    Automatically rolls back on exception, commits on success.
    """
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")
    
    conn = db_pool.getconn()
    conn.autocommit = False
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        db_pool.putconn(conn)


# ═══════════════════════════════════════════════════════════════
# AUTHENTICATION
# ═══════════════════════════════════════════════════════════════

def verify_api_key(x_api_key: str = Header(None, alias="X-API-Key")):
    """Verify API key using constant-time comparison to prevent timing attacks."""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    if not API_KEY or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


# ═══════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════

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
    ingredient_lot_overrides: Optional[Dict[str, List[IngredientLotOverride]]] = None


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


class UpdateBatchRequest(BaseModel):
    batch_weight_lb: Optional[float] = None


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def _norm(s: str) -> str:
    """Normalize whitespace in a string."""
    return re.sub(r"\s+", " ", s.strip())


def is_weight_sufficient(allocated: float, required: float) -> bool:
    """Check if allocated weight meets requirement within tolerance."""
    return allocated >= required - WEIGHT_TOLERANCE_LB


def is_weight_zero(weight: float) -> bool:
    """Check if weight is effectively zero within tolerance."""
    return abs(weight) < WEIGHT_TOLERANCE_LB


def now_local() -> datetime:
    """Get current time in plant timezone (timezone-aware)."""
    return datetime.now(tz=PLANT_TIMEZONE)


def now_utc() -> datetime:
    """Get current time in UTC (timezone-aware)."""
    return datetime.now(tz=timezone.utc)


def find_product(cur, token: str, product_type: Optional[str] = None):
    """
    Unified product lookup. Returns (product, matches) tuple.
    
    Lookup priority:
    1. Exact odoo_code match
    2. Exact name match (case-insensitive)
    3. Partial name match (ILIKE)
    
    Returns:
        - If exact match found: (product_dict, [product_dict])
        - If multiple matches: (None, [matches...])
        - If no matches: (None, [])
    """
    t = _norm(token)
    type_clause = "AND type = %s" if product_type else ""
    type_params = (product_type,) if product_type else ()
    
    # 1. Try exact odoo_code match (works for any format: numeric, alphanumeric, etc.)
    query = f"""
        SELECT id, name, odoo_code, default_batch_lb, type, uom 
        FROM products 
        WHERE odoo_code = %s {type_clause} 
        LIMIT 1
    """
    cur.execute(query, (t,) + type_params)
    row = cur.fetchone()
    if row:
        return (row, [row])
    
    # 2. Try exact name match (case-insensitive)
    query = f"""
        SELECT id, name, odoo_code, default_batch_lb, type, uom 
        FROM products 
        WHERE LOWER(name) = LOWER(%s) {type_clause} 
        LIMIT 1
    """
    cur.execute(query, (t,) + type_params)
    row = cur.fetchone()
    if row:
        return (row, [row])
    
    # 3. Try partial name match
    query = f"""
        SELECT id, name, odoo_code, default_batch_lb, type, uom 
        FROM products 
        WHERE name ILIKE %s {type_clause} 
        ORDER BY LENGTH(name) ASC 
        LIMIT %s
    """
    cur.execute(query, (f"%{t}%",) + type_params + (MAX_PRODUCT_MATCHES,))
    matches = cur.fetchall()
    
    if len(matches) == 1:
        return (matches[0], matches)
    elif len(matches) > 1:
        return (None, matches)
    return (None, [])


def format_product_matches(matches: list) -> str:
    """Format product matches for error messages."""
    return "\n".join([f"• {m['name']} ({m['odoo_code']})" for m in matches])


def generate_shipper_code(shipper_name: str) -> str:
    """Generate a shipper code from shipper name."""
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


def generate_lot_code_base(shipper_code: str, receive_date: Optional[date] = None) -> str:
    """Generate the base lot code (without sequence number)."""
    if receive_date is None:
        receive_date = now_local().date()
    return f"{receive_date.strftime('%y-%m-%d')}-{shipper_code}"


def generate_lot_code_tentative(cur, product_id: int, shipper_code: str, receive_date: Optional[date] = None) -> str:
    """
    Generate a tentative lot code for preview purposes.
    This does NOT lock rows - use create_lot_with_retry for commits.
    """
    if receive_date is None:
        receive_date = now_local().date()
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
        if code.startswith(base_code + "-"):
            try:
                seq = int(code.split("-")[-1])
                max_seq = max(max_seq, seq)
            except ValueError:
                pass
    return f"{base_code}-{max_seq + 1:03d}"


def create_lot_with_retry(cur, product_id: int, shipper_code: str, receive_date: Optional[date] = None) -> tuple:
    """
    Create a lot with automatic retry on unique constraint violation.
    
    This handles race conditions where two concurrent requests might try
    to create the same lot code.
    
    Returns:
        (lot_id, lot_code) tuple
        
    Raises:
        HTTPException if max retries exceeded
    """
    if receive_date is None:
        receive_date = now_local().date()
    base_code = f"{receive_date.strftime('%y-%m-%d')}-{shipper_code}"
    
    for attempt in range(MAX_LOT_CODE_RETRIES):
        # Find current max sequence
        cur.execute("""
            SELECT lot_code FROM lots 
            WHERE product_id = %s AND lot_code LIKE %s
            ORDER BY lot_code DESC
            LIMIT 1
            FOR UPDATE
        """, (product_id, f"{base_code}%"))
        existing = cur.fetchone()
        
        if not existing:
            next_seq = 1
        else:
            try:
                current_seq = int(existing["lot_code"].split("-")[-1])
                next_seq = current_seq + 1
            except ValueError:
                next_seq = 1
        
        lot_code = f"{base_code}-{next_seq:03d}"
        
        try:
            cur.execute("""
                INSERT INTO lots (product_id, lot_code) 
                VALUES (%s, %s) 
                RETURNING id
            """, (product_id, lot_code))
            lot_id = cur.fetchone()["id"]
            return (lot_id, lot_code)
        except pg_errors.UniqueViolation:
            # Another request beat us - retry with next sequence
            logger.warning(f"Lot code collision on {lot_code}, retrying (attempt {attempt + 1})")
            continue
    
    raise HTTPException(
        status_code=500, 
        detail=f"Failed to generate unique lot code after {MAX_LOT_CODE_RETRIES} attempts"
    )


def get_or_create_lot(cur, product_id: int, lot_code: str) -> int:
    """Get existing lot ID or create new lot."""
    cur.execute(
        "SELECT id FROM lots WHERE product_id = %s AND lot_code = %s", 
        (product_id, lot_code)
    )
    row = cur.fetchone()
    if row:
        return row["id"]
    
    try:
        cur.execute(
            "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id", 
            (product_id, lot_code)
        )
        return cur.fetchone()["id"]
    except pg_errors.UniqueViolation:
        # Race condition - lot was created by another request
        cur.execute(
            "SELECT id FROM lots WHERE product_id = %s AND lot_code = %s", 
            (product_id, lot_code)
        )
        return cur.fetchone()["id"]


def localize_timestamp(dt: datetime) -> datetime:
    """Convert a timestamp to plant timezone."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        # Treat naive timestamps as UTC (DB default)
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(PLANT_TIMEZONE)


def format_timestamp(dt: datetime) -> tuple:
    """Format timestamp for display. Returns (date_str, time_str)."""
    local_dt = localize_timestamp(dt)
    date_str = local_dt.strftime('%B %d, %Y')
    time_str = f"{local_dt.strftime('%I:%M %p')} {TIMEZONE_LABEL}"
    return date_str, time_str


def format_history_timestamp(ts) -> Optional[dict]:
    """Format timestamp for API history responses."""
    if ts is None:
        return None
    local_ts = localize_timestamp(ts)
    return {
        "iso": local_ts.isoformat(),
        "display": f"{local_ts.strftime('%b %d, %Y %I:%M %p')} {TIMEZONE_LABEL}"
    }


def get_available_lots_fifo(cur, product_id: int, lock: bool = False) -> list:
    """
    Get lots with available inventory in FIFO order.
    
    Args:
        cur: Database cursor
        product_id: Product to get lots for
        lock: If True, lock the lot rows (use in commit operations only)
    """
    # FOR UPDATE cannot be used with GROUP BY, so we use a subquery approach
    # First get lots with positive balances, then optionally lock them
    if lock:
        # Lock the lot rows first, then calculate balances
        cur.execute("""
            SELECT l.id as lot_id, l.lot_code
            FROM lots l
            WHERE l.product_id = %s
            FOR UPDATE
        """, (product_id,))
        locked_lots = cur.fetchall()
        
        if not locked_lots:
            return []
        
        # Now get balances for locked lots
        lot_ids = [lot["lot_id"] for lot in locked_lots]
        cur.execute("""
            SELECT l.id as lot_id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available_lb
            FROM lots l
            LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
            WHERE l.id = ANY(%s)
            GROUP BY l.id, l.lot_code
            HAVING COALESCE(SUM(tl.quantity_lb), 0) > %s
            ORDER BY l.id ASC
        """, (lot_ids, WEIGHT_TOLERANCE_LB))
    else:
        cur.execute("""
            SELECT l.id as lot_id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as available_lb
            FROM lots l
            LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
            WHERE l.product_id = %s
            GROUP BY l.id, l.lot_code
            HAVING COALESCE(SUM(tl.quantity_lb), 0) > %s
            ORDER BY l.id ASC
        """, (product_id, WEIGHT_TOLERANCE_LB))
    
    return cur.fetchall()


def get_lot_with_balance(cur, product_id: int, lot_code: str, lock: bool = False) -> Optional[dict]:
    """
    Get a specific lot with its current balance.
    
    Args:
        cur: Database cursor
        product_id: Product ID
        lot_code: Lot code to look up
        lock: If True, lock the lot row (use in commit operations only)
    """
    # Get the lot first (with optional lock)
    # FOR UPDATE cannot be used with GROUP BY, so we split the queries
    if lock:
        cur.execute("""
            SELECT id as lot_id, lot_code 
            FROM lots 
            WHERE product_id = %s AND lot_code = %s
            FOR UPDATE
        """, (product_id, lot_code))
    else:
        cur.execute("""
            SELECT id as lot_id, lot_code 
            FROM lots 
            WHERE product_id = %s AND lot_code = %s
        """, (product_id, lot_code))
    
    lot = cur.fetchone()
    if not lot:
        return None
    
    # Get the balance separately (aggregation doesn't need lock)
    cur.execute("""
        SELECT COALESCE(SUM(quantity_lb), 0) as current_balance
        FROM transaction_lines
        WHERE lot_id = %s
    """, (lot["lot_id"],))
    balance = cur.fetchone()
    
    return {
        "lot_id": lot["lot_id"],
        "lot_code": lot["lot_code"],
        "current_balance": float(balance["current_balance"])
    }


def allocate_from_lots(available_lots: list, required_lb: float) -> tuple:
    """
    Allocate inventory from lots using FIFO.
    
    Returns:
        (allocations, is_sufficient) tuple
    """
    allocations = []
    remaining = required_lb
    
    for lot in available_lots:
        if is_weight_zero(remaining) or remaining < 0:
            break
        use_lb = min(float(lot["available_lb"]), remaining)
        allocations.append({
            "lot_code": lot["lot_code"],
            "lot_id": lot["lot_id"],
            "available_lb": float(lot["available_lb"]),
            "use_lb": round(use_lb, 2)
        })
        remaining -= use_lb
    
    total_allocated = sum(a["use_lb"] for a in allocations)
    return allocations, is_weight_sufficient(total_allocated, required_lb)


def allocate_with_overrides(
    cur, 
    product_id: int, 
    odoo_code: str, 
    required_lb: float, 
    overrides: Optional[List] = None,
    lock: bool = False
) -> tuple:
    """
    Allocate inventory with optional lot overrides.
    
    Args:
        cur: Database cursor
        product_id: Product ID
        odoo_code: Product code (for error messages)
        required_lb: Amount needed
        overrides: Optional list of lot overrides
        lock: If True, lock lot rows
        
    Returns:
        (allocations, is_sufficient, error_msg) tuple
    """
    allocations = []
    remaining = required_lb
    used_lot_ids = set()
    error_msg = None
    
    # Process overrides first
    if overrides:
        for override in overrides:
            lot_code = override.lot_code if hasattr(override, 'lot_code') else override['lot_code']
            use_lb = override.use_lb if hasattr(override, 'use_lb') else override['use_lb']
            
            lot = get_lot_with_balance(cur, product_id, lot_code, lock=lock)
            
            if not lot:
                error_msg = f"Override lot '{lot_code}' not found for product {odoo_code}"
                continue
            
            available = float(lot["current_balance"])
            if available < use_lb:
                error_msg = f"Override lot '{lot_code}' has {available:.0f} lb, requested {use_lb:.0f} lb"
                use_lb = min(available, use_lb)
            
            if use_lb > WEIGHT_TOLERANCE_LB:
                allocations.append({
                    "lot_code": lot["lot_code"],
                    "lot_id": lot["lot_id"],
                    "available_lb": available,
                    "use_lb": round(use_lb, 2),
                    "override": True
                })
                used_lot_ids.add(lot["lot_id"])
                remaining -= use_lb
    
    # Fill remaining from FIFO
    if remaining > WEIGHT_TOLERANCE_LB:
        available_lots = get_available_lots_fifo(cur, product_id, lock=lock)
        for lot in available_lots:
            if lot["lot_id"] in used_lot_ids:
                continue
            if is_weight_zero(remaining) or remaining < 0:
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
    
    total_allocated = sum(a["use_lb"] for a in allocations)
    return allocations, is_weight_sufficient(total_allocated, required_lb), error_msg


# ═══════════════════════════════════════════════════════════════
# RECEIPT/SLIP GENERATORS
# ═══════════════════════════════════════════════════════════════

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


def generate_packing_slip_text_multi(product_name, odoo_code, lots_shipped, customer_name, order_reference, timestamp):
    date_str, time_str = format_timestamp(timestamp)
    total_lb = sum(lot["quantity_lb"] for lot in lots_shipped)
    
    if len(lots_shipped) == 1:
        lot_section = f"║  Lot Code:    {lots_shipped[0]['lot_code']:<32}║"
    else:
        lot_lines = ["║  Lot Codes:                                      ║"]
        for lot in lots_shipped:
            line = f"║    • {lot['lot_code']}: {lot['quantity_lb']:,.0f} lb"
            lot_lines.append(line + " " * (51 - len(line)) + "║")
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
    total_lb = sum(lot["quantity_lb"] for lot in lots_shipped)
    
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


# ═══════════════════════════════════════════════════════════════
# API ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {
        "name": "Factory Ledger System",
        "version": "2.0.0",
        "status": "online",
        "timezone": f"{PLANT_TIMEZONE} ({TIMEZONE_LABEL})"
    }


@app.get("/health")
def health_check():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        return {"status": "ok", "database": "connected", "pool_size": db_pool.maxconn if db_pool else 0}
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "database": "disconnected", "error": str(e)})


@app.get("/inventory/{item_name}")
def get_inventory(item_name: str, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                product, matches = find_product(cur, item_name)
                
                if not product and len(matches) > 1:
                    return JSONResponse(
                        status_code=400,
                        content={
                            "error": f"Multiple products match '{item_name}'",
                            "suggestions": [{"name": m["name"], "odoo_code": m["odoo_code"]} for m in matches]
                        }
                    )
                if not product:
                    return JSONResponse(status_code=404, content={"error": "Product not found", "query": item_name})
                
                cur.execute("""
                    SELECT COALESCE(SUM(tl.quantity_lb), 0) AS total 
                    FROM transaction_lines tl 
                    WHERE tl.product_id = %s
                """, (product["id"],))
                result = cur.fetchone()
                
        return {
            "item": product["name"],
            "odoo_code": product["odoo_code"],
            "on_hand_lb": float(result["total"])
        }
    except Exception as e:
        logger.error(f"Inventory lookup failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/products/search")
def search_products(q: str, limit: int = Query(default=20, ge=1, le=100), _: bool = Depends(verify_api_key)):
    if not q or len(q.strip()) < 2:
        return JSONResponse(status_code=400, content={"error": "Query must be at least 2 characters"})
    
    query_str = q.strip()
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Try exact odoo_code first
                cur.execute(
                    "SELECT name, odoo_code, type FROM products WHERE odoo_code = %s LIMIT %s",
                    (query_str, limit)
                )
                results = cur.fetchall()
                if results:
                    return {"query": query_str, "matches": results}
                
                # Fall back to name search
                cur.execute("""
                    SELECT name, odoo_code, type 
                    FROM products 
                    WHERE name ILIKE %s 
                    ORDER BY CASE WHEN LOWER(name) = LOWER(%s) THEN 0 ELSE 1 END, LENGTH(name) ASC 
                    LIMIT %s
                """, (f"%{query_str}%", query_str, limit))
                results = cur.fetchall()
        return {"query": query_str, "matches": results}
    except Exception as e:
        logger.error(f"Product search failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/transactions/history")
def get_transaction_history(
    _: bool = Depends(verify_api_key),
    limit: int = Query(default=10, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    type: Optional[str] = Query(default=None),
    product: Optional[str] = Query(default=None)
):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = """
                    SELECT t.id as transaction_id, t.type, t.timestamp, t.notes, t.adjust_reason,
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
                    product_filter = product if product.isdigit() else f"%{product}%"
                    field = "p2.odoo_code = %s" if product.isdigit() else "p2.name ILIKE %s"
                    conditions.append(f"""
                        t.id IN (
                            SELECT DISTINCT tl2.transaction_id
                            FROM transaction_lines tl2
                            JOIN products p2 ON p2.id = tl2.product_id
                            WHERE {field}
                        )
                    """)
                    params.append(product_filter)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " GROUP BY t.id ORDER BY t.timestamp DESC, t.id DESC LIMIT %s OFFSET %s"
                params.extend([limit, offset])
                
                cur.execute(query, params)
                transactions = cur.fetchall()
        
        result = [{
            "transaction_id": tx["transaction_id"],
            "type": tx["type"],
            "timestamp": format_history_timestamp(tx["timestamp"]),
            "notes": tx["notes"],
            "adjust_reason": tx.get("adjust_reason"),
            "lines": [line for line in (tx["lines"] or []) if line.get("product")]
        } for tx in transactions]
        
        return {
            "count": len(result),
            "filters": {"limit": limit, "offset": offset, "type": type, "product": product},
            "transactions": result
        }
    except Exception as e:
        logger.error(f"Transaction history failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/recipe/{product}")
def get_recipe(product: str, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                batch, matches = find_product(cur, product)
                
                if not batch and len(matches) > 1:
                    return {
                        "status": "multiple_matches",
                        "query": product,
                        "message": "Multiple products match.",
                        "suggestions": [{"name": m["name"], "odoo_code": m["odoo_code"]} for m in matches]
                    }
                
                if not batch:
                    return JSONResponse(status_code=404, content={"error": f"No batch products found matching: {product}"})
                
                cur.execute("""
                    SELECT p.name AS ingredient, p.odoo_code AS ingredient_code, bf.quantity_lb
                    FROM batch_formulas bf
                    JOIN products p ON p.id = bf.ingredient_product_id
                    WHERE bf.product_id = %s
                    ORDER BY bf.quantity_lb DESC
                """, (batch["id"],))
                ingredients = cur.fetchall()
                
                if not ingredients:
                    return JSONResponse(status_code=404, content={"error": f"No recipe found for: {batch['name']}"})
                
                return {
                    "product": batch["name"],
                    "odoo_code": batch["odoo_code"],
                    "batch_size_lb": float(batch["default_batch_lb"]) if batch["default_batch_lb"] else None,
                    "ingredient_count": len(ingredients),
                    "ingredients": [
                        {"name": ing["ingredient"], "odoo_code": ing["ingredient_code"], "quantity_lb": float(ing["quantity_lb"])}
                        for ing in ingredients
                    ]
                }
    except Exception as e:
        logger.error(f"Recipe lookup failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# RECEIVE ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/receive/preview", response_model=ReceivePreviewResponse)
def receive_preview(req: ReceiveRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                product, matches = find_product(cur, req.product_name)
                
                if not product and len(matches) > 1:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Multiple products match '{req.product_name}'. Please be more specific:\n{format_product_matches(matches)}"
                    )
                if not product:
                    raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")
                
                shipper_code = req.shipper_code_override.upper()[:5] if req.shipper_code_override else generate_shipper_code(req.shipper_name)
                shipper_code_auto = not req.shipper_code_override
                
                # Tentative lot code for preview (may change on commit if race condition)
                lot_code = generate_lot_code_tentative(cur, product["id"], shipper_code)
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
            product_id=product["id"],
            product_name=product["name"],
            odoo_code=product["odoo_code"],
            cases=req.cases,
            case_size_lb=req.case_size_lb,
            total_lb=total_lb,
            shipper_name=req.shipper_name,
            shipper_code=shipper_code,
            shipper_code_auto=shipper_code_auto,
            lot_code=lot_code,
            bol_reference=req.bol_reference,
            preview_message=preview_message
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Receive preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/receive/commit", response_model=ReceiveCommitResponse)
def receive_commit(req: ReceiveRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            product, _ = find_product(cur, req.product_name)
            if not product:
                raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")
            
            shipper_code = req.shipper_code_override.upper()[:5] if req.shipper_code_override else generate_shipper_code(req.shipper_name)
            total_lb = req.cases * req.case_size_lb
            timestamp = now_local()
            
            # Create lot with retry logic for race conditions
            lot_id, lot_code = create_lot_with_retry(cur, product["id"], shipper_code)
            
            cur.execute("""
                INSERT INTO transactions (type, bol_reference, shipper_name, shipper_code, cases_received, case_size_lb)
                VALUES ('receive', %s, %s, %s, %s, %s)
                RETURNING id
            """, (req.bol_reference, req.shipper_name, shipper_code, req.cases, req.case_size_lb))
            transaction_id = cur.fetchone()["id"]
            
            cur.execute("""
                INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                VALUES (%s, %s, %s, %s)
            """, (transaction_id, product["id"], lot_id, total_lb))
            
            logger.info(f"RECEIVE: {total_lb:.0f} lb {product['name']} lot={lot_code} tx={transaction_id}")
            
            receipt_text = generate_receipt_text(
                product["name"], product["odoo_code"], req.cases, req.case_size_lb,
                total_lb, req.shipper_name, lot_code, req.bol_reference, timestamp
            )
            receipt_html = generate_receipt_html(
                product["name"], product["odoo_code"], req.cases, req.case_size_lb,
                total_lb, req.shipper_name, lot_code, req.bol_reference, timestamp
            )
            
            return ReceiveCommitResponse(
                success=True,
                transaction_id=transaction_id,
                lot_id=lot_id,
                lot_code=lot_code,
                total_lb=total_lb,
                receipt_text=receipt_text,
                receipt_html=receipt_html,
                message=f"✅ Received {total_lb:,.0f} lb {product['name']} into lot {lot_code}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Receive commit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# SHIP ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/ship/preview", response_model=ShipPreviewResponse)
def ship_preview(req: ShipRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                product, matches = find_product(cur, req.product_name)
                
                if not product and len(matches) > 1:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Multiple products match '{req.product_name}'. Please be more specific:\n{format_product_matches(matches)}"
                    )
                if not product:
                    raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")
                
                # No lock for preview
                lots_with_inventory = get_available_lots_fifo(cur, product["id"], lock=False)
                if not lots_with_inventory:
                    raise HTTPException(status_code=400, detail=f"No inventory available for {product['name']}")
                
                if req.lot_code:
                    selected_lot = next((l for l in lots_with_inventory if l["lot_code"] == req.lot_code), None)
                    if not selected_lot:
                        lot_list = "\n".join([f"• {l['lot_code']} ({l['available_lb']:,.0f} lb)" for l in lots_with_inventory])
                        raise HTTPException(
                            status_code=400,
                            detail=f"Lot '{req.lot_code}' not found or has no inventory. Available lots:\n{lot_list}"
                        )
                    if float(selected_lot["available_lb"]) < req.quantity_lb:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Insufficient inventory in lot {selected_lot['lot_code']}. Requested: {req.quantity_lb:,.0f} lb, Available: {selected_lot['available_lb']:,.0f} lb."
                        )
                    allocations = [ShipLotAllocation(
                        lot_code=selected_lot["lot_code"],
                        lot_id=selected_lot["lot_id"],
                        available_lb=float(selected_lot["available_lb"]),
                        use_lb=req.quantity_lb
                    )]
                    multi_lot = False
                else:
                    raw_allocations, _ = allocate_from_lots(lots_with_inventory, req.quantity_lb)
                    allocations = [ShipLotAllocation(**a) for a in raw_allocations]
                    multi_lot = len(allocations) > 1
                
                total_allocated = sum(a.use_lb for a in allocations)
                sufficient = is_weight_sufficient(total_allocated, req.quantity_lb)
                
                if multi_lot:
                    lot_lines = "\n".join([f"    • {a.lot_code}: {a.use_lb:,.0f} lb (of {a.available_lb:,.0f} lb)" for a in allocations])
                    lot_section = f"Lots (FIFO):\n{lot_lines}"
                else:
                    a = allocations[0]
                    lot_note = "(specified)" if req.lot_code else "(FIFO)"
                    lot_section = f"Lot:          {a.lot_code} {lot_note}\nAvailable:    {a.available_lb:,.0f} lb in this lot"
                
                status = "✓ Say \"confirm\" to proceed" if sufficient else f"⚠️ INSUFFICIENT: Need {req.quantity_lb:,.0f} lb, can only allocate {total_allocated:,.0f} lb"
                
                preview_message = f"""📦 SHIP PREVIEW
────────────────────────────────
Product:      {product['name']} ({product['odoo_code']})
Quantity:     {req.quantity_lb:,.0f} lb
Customer:     {req.customer_name}
Order Ref:    {req.order_reference}
{lot_section}
────────────────────────────────
{status}"""
                
                return ShipPreviewResponse(
                    product_id=product["id"],
                    product_name=product["name"],
                    odoo_code=product["odoo_code"],
                    quantity_lb=req.quantity_lb,
                    customer_name=req.customer_name,
                    order_reference=req.order_reference,
                    allocated_lots=allocations,
                    total_allocated_lb=total_allocated,
                    sufficient=sufficient,
                    multi_lot=multi_lot,
                    preview_message=preview_message
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ship preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ship/commit", response_model=ShipCommitResponse)
def ship_commit(req: ShipRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            product, _ = find_product(cur, req.product_name)
            if not product:
                raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")
            
            # LOCK inventory rows to prevent race conditions
            lots_with_inventory = get_available_lots_fifo(cur, product["id"], lock=True)
            if not lots_with_inventory:
                raise HTTPException(status_code=400, detail=f"No inventory available for {product['name']}")
            
            if req.lot_code:
                selected_lot = next((l for l in lots_with_inventory if l["lot_code"] == req.lot_code), None)
                if not selected_lot:
                    raise HTTPException(status_code=400, detail=f"Lot '{req.lot_code}' not found or has no inventory")
                if float(selected_lot["available_lb"]) < req.quantity_lb:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Insufficient inventory in lot {selected_lot['lot_code']}. Available: {selected_lot['available_lb']:,.0f} lb"
                    )
                allocations = [{"lot_code": selected_lot["lot_code"], "lot_id": selected_lot["lot_id"], "use_lb": req.quantity_lb}]
            else:
                allocations, sufficient = allocate_from_lots(lots_with_inventory, req.quantity_lb)
                if not sufficient:
                    total_allocated = sum(a["use_lb"] for a in allocations)
                    raise HTTPException(
                        status_code=400,
                        detail=f"Insufficient total inventory. Need {req.quantity_lb:,.0f} lb, only {total_allocated:,.0f} lb available."
                    )
            
            timestamp = now_local()
            
            cur.execute("""
                INSERT INTO transactions (type, customer_name, order_reference)
                VALUES ('ship', %s, %s)
                RETURNING id
            """, (req.customer_name, req.order_reference))
            transaction_id = cur.fetchone()["id"]
            
            lots_shipped = []
            for alloc in allocations:
                cur.execute("""
                    INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                    VALUES (%s, %s, %s, %s)
                """, (transaction_id, product["id"], alloc["lot_id"], -alloc["use_lb"]))
                lots_shipped.append({"lot_code": alloc["lot_code"], "quantity_lb": alloc["use_lb"]})
            
            total_shipped = sum(l["quantity_lb"] for l in lots_shipped)
            logger.info(f"SHIP: {total_shipped:.0f} lb {product['name']} to {req.customer_name} tx={transaction_id}")
            
            slip_text = generate_packing_slip_text_multi(
                product["name"], product["odoo_code"], lots_shipped,
                req.customer_name, req.order_reference, timestamp
            )
            slip_html = generate_packing_slip_html_multi(
                product["name"], product["odoo_code"], lots_shipped,
                req.customer_name, req.order_reference, timestamp
            )
            
            lot_summary = ", ".join([f"{l['lot_code']} ({l['quantity_lb']:,.0f} lb)" for l in lots_shipped])
            return ShipCommitResponse(
                success=True,
                transaction_id=transaction_id,
                lots_shipped=lots_shipped,
                total_quantity_lb=total_shipped,
                slip_text=slip_text,
                slip_html=slip_html,
                message=f"✅ Shipped {total_shipped:,.0f} lb {product['name']} to {req.customer_name}\nLots: {lot_summary}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ship commit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# ADJUST ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/adjust/preview", response_model=AdjustPreviewResponse)
def adjust_preview(req: AdjustRequest, _: bool = Depends(verify_api_key)):
    if not req.reason or len(req.reason.strip()) < 3:
        raise HTTPException(status_code=400, detail="Reason is required (minimum 3 characters)")
    if is_weight_zero(req.quantity_lb):
        raise HTTPException(status_code=400, detail="Adjustment quantity cannot be zero")
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                product, matches = find_product(cur, req.product_name)
                
                if not product and len(matches) > 1:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Multiple products match '{req.product_name}'. Please be more specific:\n{format_product_matches(matches)}"
                    )
                if not product:
                    raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")
                
                lot = get_lot_with_balance(cur, product["id"], req.lot_code, lock=False)
                
                if not lot:
                    available_lots = get_available_lots_fifo(cur, product["id"], lock=False)
                    if available_lots:
                        lot_list = "\n".join([f"• {l['lot_code']} ({l['available_lb']:,.0f} lb)" for l in available_lots])
                        raise HTTPException(
                            status_code=400,
                            detail=f"Lot '{req.lot_code}' not found for {product['name']}. Existing lots:\n{lot_list}"
                        )
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Lot '{req.lot_code}' not found for {product['name']}. No lots exist for this product."
                        )
                
                current_balance = float(lot["current_balance"])
                new_balance = current_balance + req.quantity_lb
                
                if new_balance < -WEIGHT_TOLERANCE_LB:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot adjust. Lot {req.lot_code} only has {current_balance:,.0f} lb."
                    )
        
        adj_display = f"+{req.quantity_lb:,.0f}" if req.quantity_lb > 0 else f"{req.quantity_lb:,.0f}"
        adj_type = "ADD" if req.quantity_lb > 0 else "REMOVE"
        
        preview_message = f"""📋 ADJUST PREVIEW
────────────────────────────────
Product:      {product['name']} ({product['odoo_code']})
Lot:          {req.lot_code}
Adjustment:   {adj_display} lb ({adj_type})
────────────────────────────────
Current:      {current_balance:,.0f} lb
After:        {new_balance:,.0f} lb
────────────────────────────────
Reason:       {req.reason}
────────────────────────────────
✓ Say "confirm" to proceed"""
        
        return AdjustPreviewResponse(
            product_id=product["id"],
            product_name=product["name"],
            odoo_code=product["odoo_code"],
            lot_code=lot["lot_code"],
            lot_id=lot["lot_id"],
            adjustment_lb=req.quantity_lb,
            current_balance_lb=current_balance,
            new_balance_lb=new_balance,
            reason=req.reason,
            preview_message=preview_message
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Adjust preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/adjust/commit", response_model=AdjustCommitResponse)
def adjust_commit(req: AdjustRequest, _: bool = Depends(verify_api_key)):
    if not req.reason or len(req.reason.strip()) < 3:
        raise HTTPException(status_code=400, detail="Reason is required (minimum 3 characters)")
    if is_weight_zero(req.quantity_lb):
        raise HTTPException(status_code=400, detail="Adjustment quantity cannot be zero")
    
    try:
        with get_transaction() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            product, _ = find_product(cur, req.product_name)
            if not product:
                raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")
            
            # LOCK the lot row
            lot = get_lot_with_balance(cur, product["id"], req.lot_code, lock=True)
            if not lot:
                raise HTTPException(status_code=400, detail=f"Lot '{req.lot_code}' not found for {product['name']}")
            
            current_balance = float(lot["current_balance"])
            new_balance = current_balance + req.quantity_lb
            
            if new_balance < -WEIGHT_TOLERANCE_LB:
                raise HTTPException(
                    status_code=400,
                    detail=f"Cannot adjust. Lot {req.lot_code} only has {current_balance:,.0f} lb."
                )
            
            cur.execute(
                "INSERT INTO transactions (type, adjust_reason) VALUES ('adjust', %s) RETURNING id",
                (req.reason,)
            )
            transaction_id = cur.fetchone()["id"]
            
            cur.execute("""
                INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                VALUES (%s, %s, %s, %s)
            """, (transaction_id, product["id"], lot["lot_id"], req.quantity_lb))
            
            adj_type = "Added" if req.quantity_lb > 0 else "Removed"
            logger.info(f"ADJUST: {req.quantity_lb:+.0f} lb {product['name']} lot={req.lot_code} reason='{req.reason}' tx={transaction_id}")
            
            return AdjustCommitResponse(
                success=True,
                transaction_id=transaction_id,
                lot_code=lot["lot_code"],
                adjustment_lb=req.quantity_lb,
                new_balance_lb=new_balance,
                reason=req.reason,
                message=f"✅ {adj_type} {abs(req.quantity_lb):,.0f} lb {product['name']} (lot {lot['lot_code']}). New balance: {new_balance:,.0f} lb"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Adjust commit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# MAKE ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/make/preview", response_model=MakePreviewResponse)
def make_preview(req: MakeRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                batch, matches = find_product(cur, req.product_name)
                
                if not batch and len(matches) > 1:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Multiple products match '{req.product_name}'. Please be more specific:\n{format_product_matches(matches)}"
                    )
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
                    
                    # No lock for preview
                    allocations, sufficient, error_msg = allocate_with_overrides(
                        cur, product_id, odoo_code, required_lb, overrides, lock=False
                    )
                    
                    if error_msg:
                        override_warnings.append(error_msg)
                    if not sufficient:
                        all_sufficient = False
                    
                    ingredients.append(IngredientAllocation(
                        product_name=line["ingredient_name"],
                        product_id=product_id,
                        odoo_code=odoo_code,
                        required_lb=round(required_lb, 2),
                        allocated_lots=[
                            IngredientLotAllocation(
                                lot_code=a["lot_code"],
                                lot_id=a["lot_id"],
                                available_lb=a["available_lb"],
                                use_lb=a["use_lb"]
                            ) for a in allocations
                        ],
                        sufficient=sufficient
                    ))
                
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
                
                preview_message = f"""🏭 PRODUCTION PREVIEW
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
                
                return MakePreviewResponse(
                    batch_product_id=batch["id"],
                    batch_product_name=batch["name"],
                    batch_odoo_code=batch.get("odoo_code", ""),
                    batches=req.batches,
                    batch_size_lb=batch_size_lb,
                    total_yield_lb=total_yield_lb,
                    output_lot_code=req.lot_code,
                    ingredients=ingredients,
                    all_sufficient=all_sufficient,
                    preview_message=preview_message
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Make preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/make/commit", response_model=MakeCommitResponse)
def make_commit(req: MakeRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as conn:
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
            
            # Pre-validate all ingredients with locks
            for line in bom_lines:
                required_lb = float(line["quantity_lb"]) * req.batches
                odoo_code = line["ingredient_odoo_code"]
                overrides = None
                if req.ingredient_lot_overrides and odoo_code in req.ingredient_lot_overrides:
                    overrides = req.ingredient_lot_overrides[odoo_code]
                
                allocations, sufficient, _ = allocate_with_overrides(
                    cur, line["ingredient_product_id"], odoo_code, required_lb, overrides, lock=True
                )
                if not sufficient:
                    total_allocated = sum(a["use_lb"] for a in allocations)
                    raise HTTPException(
                        status_code=400,
                        detail=f"Insufficient {line['ingredient_name']}: need {required_lb:,.0f} lb, can allocate {total_allocated:,.0f} lb"
                    )
            
            cur.execute("""
                INSERT INTO transactions (type, notes)
                VALUES ('make', %s)
                RETURNING id
            """, (f"Make {req.batches} batch(es) {batch['name']} lot {req.lot_code}",))
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
                
                # Already locked above, but allocate again to get final values
                allocations, _, _ = allocate_with_overrides(
                    cur, product_id, odoo_code, required_lb, overrides, lock=False
                )
                
                lots_used = []
                for alloc in allocations:
                    use_lb = alloc["use_lb"]
                    lot_id = alloc["lot_id"]
                    lot_code = alloc["lot_code"]
                    
                    cur.execute("""
                        INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                        VALUES (%s, %s, %s, %s)
                    """, (tx_id, product_id, lot_id, -use_lb))
                    
                    cur.execute("""
                        INSERT INTO ingredient_lot_consumption (transaction_id, ingredient_product_id, ingredient_lot_id, quantity_lb)
                        VALUES (%s, %s, %s, %s)
                    """, (tx_id, product_id, lot_id, use_lb))
                    
                    lots_used.append({"lot": lot_code, "qty_lb": round(use_lb, 2)})
                
                consumed.append({
                    "ingredient": line["ingredient_name"],
                    "odoo_code": odoo_code,
                    "total_lb": round(required_lb, 2),
                    "lots": lots_used
                })
            
            cur.execute("""
                INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                VALUES (%s, %s, %s, %s)
            """, (tx_id, batch["id"], output_lot_id, total_yield_lb))
            
            logger.info(f"MAKE: {total_yield_lb:.0f} lb {batch['name']} lot={req.lot_code} tx={tx_id}")
            
            consumed_summary = "\n".join([
                f"  • {c['ingredient']}: {c['total_lb']} lb from {', '.join([l['lot'] for l in c['lots']])}"
                for c in consumed
            ])
            
            return MakeCommitResponse(
                success=True,
                transaction_id=tx_id,
                batch_product_name=batch["name"],
                batch_odoo_code=batch.get("odoo_code", ""),
                batches=req.batches,
                produced_lb=total_yield_lb,
                output_lot_code=req.lot_code,
                consumed=consumed,
                message=f"✅ Produced {total_yield_lb:,.0f} lb {batch['name']} (lot {req.lot_code})\n\nIngredients consumed:\n{consumed_summary}"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Make commit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# REPACK ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.post("/repack/preview", response_model=RepackPreviewResponse)
def repack_preview(req: RepackRequest, _: bool = Depends(verify_api_key)):
    if req.source_quantity_lb <= 0:
        raise HTTPException(status_code=400, detail="Source quantity must be positive")
    if req.target_quantity_lb <= 0:
        raise HTTPException(status_code=400, detail="Target quantity must be positive")
    
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                source, source_matches = find_product(cur, req.source_product)
                if not source and len(source_matches) > 1:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Multiple source products match '{req.source_product}':\n{format_product_matches(source_matches)}"
                    )
                if not source:
                    raise HTTPException(status_code=404, detail=f"Source product not found: {req.source_product}")
                
                source_lot = get_lot_with_balance(cur, source["id"], req.source_lot, lock=False)
                
                if not source_lot:
                    available_lots = get_available_lots_fifo(cur, source["id"], lock=False)
                    if available_lots:
                        lot_list = "\n".join([f"• {l['lot_code']} ({l['available_lb']:,.0f} lb)" for l in available_lots])
                        raise HTTPException(
                            status_code=400,
                            detail=f"Lot '{req.source_lot}' not found for {source['name']}. Available lots:\n{lot_list}"
                        )
                    else:
                        raise HTTPException(status_code=400, detail=f"No lots found for {source['name']}")
                
                source_available = float(source_lot["current_balance"])
                if source_available < req.source_quantity_lb:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Insufficient inventory in lot {req.source_lot}. Available: {source_available:,.0f} lb, Requested: {req.source_quantity_lb:,.0f} lb"
                    )
                
                target, target_matches = find_product(cur, req.target_product)
                if not target and len(target_matches) > 1:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Multiple target products match '{req.target_product}':\n{format_product_matches(target_matches)}"
                    )
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
                
                return RepackPreviewResponse(
                    source_product_id=source["id"],
                    source_product_name=source["name"],
                    source_odoo_code=source.get("odoo_code", ""),
                    source_lot_code=source_lot["lot_code"],
                    source_lot_id=source_lot["lot_id"],
                    source_available_lb=source_available,
                    source_consume_lb=req.source_quantity_lb,
                    target_product_id=target["id"],
                    target_product_name=target["name"],
                    target_odoo_code=target.get("odoo_code", ""),
                    target_lot_code=req.target_lot_code,
                    target_produce_lb=req.target_quantity_lb,
                    yield_pct=round(yield_pct, 2),
                    notes=req.notes,
                    preview_message=preview_message
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Repack preview failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/repack/commit", response_model=RepackCommitResponse)
def repack_commit(req: RepackRequest, _: bool = Depends(verify_api_key)):
    if req.source_quantity_lb <= 0:
        raise HTTPException(status_code=400, detail="Source quantity must be positive")
    if req.target_quantity_lb <= 0:
        raise HTTPException(status_code=400, detail="Target quantity must be positive")
    
    try:
        with get_transaction() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            source, _ = find_product(cur, req.source_product)
            if not source:
                raise HTTPException(status_code=404, detail=f"Source product not found: {req.source_product}")
            
            # LOCK the source lot
            source_lot = get_lot_with_balance(cur, source["id"], req.source_lot, lock=True)
            if not source_lot:
                raise HTTPException(status_code=400, detail=f"Lot '{req.source_lot}' not found for {source['name']}")
            
            source_available = float(source_lot["current_balance"])
            if source_available < req.source_quantity_lb:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient inventory. Available: {source_available:,.0f} lb, Requested: {req.source_quantity_lb:,.0f} lb"
                )
            
            target, _ = find_product(cur, req.target_product)
            if not target:
                raise HTTPException(status_code=404, detail=f"Target product not found: {req.target_product}")
            
            notes = req.notes or f"Repack {source['name']} to {target['name']}"
            
            cur.execute(
                "INSERT INTO transactions (type, notes) VALUES ('repack', %s) RETURNING id",
                (notes,)
            )
            tx_id = cur.fetchone()["id"]
            
            cur.execute("""
                INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                VALUES (%s, %s, %s, %s)
            """, (tx_id, source["id"], source_lot["lot_id"], -req.source_quantity_lb))
            
            target_lot_id = get_or_create_lot(cur, target["id"], req.target_lot_code)
            
            cur.execute("""
                INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
                VALUES (%s, %s, %s, %s)
            """, (tx_id, target["id"], target_lot_id, req.target_quantity_lb))
            
            cur.execute("""
                INSERT INTO ingredient_lot_consumption (transaction_id, ingredient_product_id, ingredient_lot_id, quantity_lb)
                VALUES (%s, %s, %s, %s)
            """, (tx_id, source["id"], source_lot["lot_id"], req.source_quantity_lb))
            
            logger.info(f"REPACK: {req.source_quantity_lb:.0f} lb {source['name']} -> {req.target_quantity_lb:.0f} lb {target['name']} tx={tx_id}")
            
            return RepackCommitResponse(
                success=True,
                transaction_id=tx_id,
                source_product=source["name"],
                source_lot=req.source_lot,
                consumed_lb=req.source_quantity_lb,
                target_product=target["name"],
                target_lot=req.target_lot_code,
                produced_lb=req.target_quantity_lb,
                message=f"✅ Repacked {req.source_quantity_lb:,.0f} lb {source['name']} (lot {req.source_lot}) → {req.target_quantity_lb:,.0f} lb {target['name']} (lot {req.target_lot_code})"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Repack commit failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════
# TRACEABILITY ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/trace/batch/{lot_code}")
def trace_batch(lot_code: str, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
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
                    ORDER BY t.timestamp DESC
                    LIMIT 1
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
                    "ingredients_consumed": [
                        {
                            "ingredient": ing["ingredient_name"],
                            "odoo_code": ing["ingredient_code"],
                            "lot": ing["ingredient_lot"],
                            "quantity_lb": float(ing["quantity_lb"])
                        } for ing in ingredients
                    ]
                }
    except Exception as e:
        logger.error(f"Trace batch failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/trace/ingredient/{lot_code}")
def trace_ingredient(lot_code: str, used_only: bool = Query(default=False), _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
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
        logger.error(f"Trace ingredient failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# BOM ENDPOINTS
# ═══════════════════════════════════════════════════════════════

@app.get("/bom/products")
def get_bom_products(
    product_type: Optional[str] = Query(None, description="Filter by type"),
    brand: Optional[str] = Query(None, description="Filter by brand"),
    search: Optional[str] = Query(None, description="Search by name or odoo_code"),
    limit: int = Query(100, le=500),
    _: bool = Depends(verify_api_key)
):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = "SELECT id, odoo_code, name, type, brand, uom, default_batch_lb, active FROM products WHERE 1=1"
                params = []
                
                if product_type:
                    query += " AND type = %s"
                    params.append(product_type)
                if brand:
                    query += " AND brand = %s"
                    params.append(brand)
                if search:
                    query += " AND (name ILIKE %s OR odoo_code ILIKE %s)"
                    params.extend([f"%{search}%", f"%{search}%"])
                
                query += " ORDER BY odoo_code LIMIT %s"
                params.append(limit)
                
                cur.execute(query, params)
                products = cur.fetchall()
        
        return {"count": len(products), "products": products}
    except Exception as e:
        logger.error(f"BOM products failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/bom/batches")
def get_all_batches(brand: Optional[str] = Query(None), _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                query = "SELECT * FROM v_batch_summary WHERE 1=1"
                params = []
                
                if brand:
                    query += " AND brand = %s"
                    params.append(brand)
                
                query += " ORDER BY internal_ref"
                cur.execute(query, params)
                batches = cur.fetchall()
        
        return {"count": len(batches), "batches": batches}
    except Exception as e:
        logger.error(f"BOM batches failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/bom/batches/{batch_ref}/formula")
def get_batch_formula(batch_ref: str, _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, odoo_code, name, default_batch_lb
                    FROM products
                    WHERE odoo_code = %s AND type = 'batch'
                """, (batch_ref,))
                batch = cur.fetchone()
                
                if not batch:
                    return JSONResponse(status_code=404, content={"error": f"Batch {batch_ref} not found"})
                
                cur.execute("""
                    SELECT p.odoo_code as ingredient_ref, p.name as ingredient_name,
                           bf.quantity_lb as quantity, p.uom as unit
                    FROM batch_formulas bf
                    JOIN products p ON p.id = bf.ingredient_product_id
                    WHERE bf.product_id = %s
                    ORDER BY bf.quantity_lb DESC
                """, (batch["id"],))
                ingredients = cur.fetchall()
                
                cur.execute("""
                    SELECT a.name
                    FROM product_allergens pa
                    JOIN allergens a ON pa.allergen_id = a.id
                    WHERE pa.product_id = %s
                """, (batch["id"],))
                allergens = [row["name"] for row in cur.fetchall()]
        
        return {
            "batch_ref": batch["odoo_code"],
            "batch_name": batch["name"],
            "batch_weight_lb": batch["default_batch_lb"],
            "ingredient_count": len(ingredients),
            "ingredients": ingredients,
            "allergens": allergens
        }
    except Exception as e:
        logger.error(f"Batch formula failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/bom/allergens")
def get_allergens_list(_: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT a.id, a.name,
                           COALESCE(json_agg(
                               json_build_object('batch_ref', p.odoo_code, 'batch_name', p.name)
                           ) FILTER (WHERE p.id IS NOT NULL), '[]') as batches
                    FROM allergens a
                    LEFT JOIN product_allergens pa ON a.id = pa.allergen_id
                    LEFT JOIN products p ON pa.product_id = p.id
                    GROUP BY a.id, a.name
                    ORDER BY a.name
                """)
                allergens = cur.fetchall()
        
        return {"allergens": allergens}
    except Exception as e:
        logger.error(f"Allergens list failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/bom/search")
def search_bom(q: str = Query(..., description="Search term"), _: bool = Depends(verify_api_key)):
    try:
        with get_db_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT odoo_code, name, type, brand
                    FROM products
                    WHERE name ILIKE %s OR odoo_code ILIKE %s
                    ORDER BY type, odoo_code
                    LIMIT 25
                """, (f"%{q}%", f"%{q}%"))
                products = cur.fetchall()
        
        return {"query": q, "result_count": len(products), "results": products}
    except Exception as e:
        logger.error(f"BOM search failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.put("/bom/batches/{batch_ref}")
def update_batch(batch_ref: str, req: UpdateBatchRequest, _: bool = Depends(verify_api_key)):
    try:
        with get_transaction() as conn:
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            cur.execute("SELECT id FROM products WHERE odoo_code = %s AND type = 'batch'", (batch_ref,))
            batch = cur.fetchone()
            
            if not batch:
                return JSONResponse(status_code=404, content={"error": f"Batch {batch_ref} not found"})
            
            updates = []
            params = []
            
            if req.batch_weight_lb is not None:
                updates.append("default_batch_lb = %s")
                params.append(req.batch_weight_lb)
            
            if not updates:
                return JSONResponse(status_code=400, content={"error": "No fields to update"})
            
            params.append(batch_ref)
            query = f"UPDATE products SET {', '.join(updates)} WHERE odoo_code = %s RETURNING *"
            cur.execute(query, params)
            updated = cur.fetchone()
            
            logger.info(f"Updated batch {batch_ref}")
            
            return {"message": f"Batch {batch_ref} updated", "batch": updated}
    except Exception as e:
        logger.error(f"Update batch failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# ═══════════════════════════════════════════════════════════════
# DATABASE SETUP - RUN THESE ON YOUR SUPABASE DATABASE
# ═══════════════════════════════════════════════════════════════
"""
-- Required unique constraint for lot code race condition handling
ALTER TABLE lots ADD CONSTRAINT lots_product_id_lot_code_key UNIQUE (product_id, lot_code);

-- Recommended indexes for performance
CREATE INDEX IF NOT EXISTS idx_lots_product_id ON lots(product_id);
CREATE INDEX IF NOT EXISTS idx_lots_lot_code ON lots(lot_code);
CREATE INDEX IF NOT EXISTS idx_transaction_lines_lot_id ON transaction_lines(lot_id);
CREATE INDEX IF NOT EXISTS idx_transaction_lines_product_id ON transaction_lines(product_id);
CREATE INDEX IF NOT EXISTS idx_transaction_lines_transaction_id ON transaction_lines(transaction_id);
CREATE INDEX IF NOT EXISTS idx_transactions_type ON transactions(type);
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_ingredient_lot_consumption_transaction_id ON ingredient_lot_consumption(transaction_id);
CREATE INDEX IF NOT EXISTS idx_ingredient_lot_consumption_ingredient_lot_id ON ingredient_lot_consumption(ingredient_lot_id);
CREATE INDEX IF NOT EXISTS idx_batch_formulas_product_id ON batch_formulas(product_id);
CREATE INDEX IF NOT EXISTS idx_products_odoo_code ON products(odoo_code);
CREATE INDEX IF NOT EXISTS idx_products_type ON products(type);
"""
