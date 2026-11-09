"""Supplies (dashboard-only): packaging / consumables inventory view, low-stock
thresholds and the supply-request queue. Migration 043.

Moved unchanged from main.py (modularization PR2). main.py registers these
routes at the spot the section used to occupy, after the sales-order routes and
before the ship endpoints, so route order is unchanged. Not in
openapi-gpt-v3.yaml on purpose (30-op cap; GPTs don't need these).

main is reached through `api` at call time: every route needs a key
(`api.verify_api_key`); POST and PATCH /supply-requests stay in
permissions.py UNGATED_ROUTES (any named actor), unchanged. on_hand and lot balances are read-only sums of
posted ledger lines (`api.POSTED_LINES`, `api.fifo_lot_balances`);
`api.ledger_quantity_unit` stays in main because other code uses it too.

Supply requests are a queue, not inventory: no route here writes
transactions, lots or allocations.
"""
import re
from typing import Literal, Optional

from fastapi import Depends, HTTPException, Query
from pydantic import BaseModel


class SupplyRequestCreate(BaseModel):
    """Supplies: one "we need X" request from the floor/office. Exactly one of
    product_id (a catalogued product) or item_text (free text for anything not
    in products) — never both, never neither. qty is optional (product uom /
    whatever the text implies). requested_by is a plain name/source tag."""
    product_id: Optional[int] = None
    item_text: Optional[str] = None
    qty: Optional[float] = None
    note: Optional[str] = None
    requested_by: str


class SupplyRequestUpdate(BaseModel):
    """PATCH body. The only transition is open -> done (sets done_at)."""
    status: Literal["done"]


# The Supplies filter intentionally exposes only supply-relevant categories;
# the unfiltered endpoint still returns every products.type value.
PRODUCT_CATEGORIES = ("ingredient", "packaging", "consumable")


# ── Supply requests ────────────────────────────────────────────

SUPPLY_REQUEST_SELECT_SQL = """
    SELECT sr.id, sr.product_id, sr.item_text, sr.qty, sr.note, sr.requested_by,
           sr.status, sr.created_at, sr.done_at,
           p.name AS product_name, p.odoo_code, p.type AS category, p.uom
    FROM supply_requests sr
    LEFT JOIN products p ON p.id = sr.product_id
"""


def _serialize_supply_request(row: dict) -> dict:
    return {
        "id": row["id"],
        "product_id": row["product_id"],
        "product_name": row.get("product_name"),
        "sku": row.get("odoo_code"),
        "category": row.get("category"),
        "unit": (row.get("uom") or "lb") if row["product_id"] is not None else None,
        "item_text": row["item_text"],
        "display_name": row.get("product_name") if row["product_id"] is not None else row["item_text"],
        "qty": float(row["qty"]) if row["qty"] is not None else None,
        "note": row["note"],
        "requested_by": row["requested_by"],
        "status": row["status"],
        "created_at": row["created_at"].isoformat() if row.get("created_at") else None,
        "done_at": row["done_at"].isoformat() if row.get("done_at") else None,
    }


def fetch_supply_request(cur, supply_request_id: int) -> Optional[dict]:
    cur.execute(SUPPLY_REQUEST_SELECT_SQL + " WHERE sr.id = %s", (supply_request_id,))
    row = cur.fetchone()
    return _serialize_supply_request(dict(row)) if row else None


def register_routes(app, api):
    # Every product (active or not, zero inventory or not) with its posted-only
    # ledger SUM. LEFT JOINs so a product with no lots / no lines still appears
    # with on_hand 0. This is a read of the ledger, never of a stored balance.
    SUPPLIES_INVENTORY_SQL = f"""
        SELECT p.id AS product_id, p.name, p.odoo_code, p.type AS category, p.uom,
               p.active, p.case_size_lb, p.low_stock_threshold,
               COALESCE(SUM(tl.quantity_lb), 0) AS on_hand
        FROM products p
        LEFT JOIN lots l ON l.product_id = p.id
        LEFT JOIN {api.POSTED_LINES} tl ON tl.lot_id = l.id
    """


    def _serialize_supplies_item(row: dict) -> dict:
        on_hand = float(row["on_hand"] or 0)
        threshold = float(row["low_stock_threshold"]) if row.get("low_stock_threshold") is not None else None
        return {
            "product_id": row["product_id"],
            "name": row["name"],
            "sku": row.get("odoo_code"),
            "category": row["category"],
            "unit": api.ledger_quantity_unit(row.get("uom")),
            "request_unit": row.get("uom") or "lb",  # Requests use purchasing units, not on-hand ledger units.
            "active": bool(row["active"]) if row.get("active") is not None else True,
            "case_size_lb": float(row["case_size_lb"]) if row.get("case_size_lb") is not None else None,
            "on_hand": round(on_hand, 4),
            "low_stock_threshold": threshold,
            # is_low is false whenever no threshold is set (NULL = no alerting)
            # and for an archived product (products.active=false): archived
            # items never raise low-stock alerts (product_archive.py).
            "is_low": bool(threshold is not None and on_hand < threshold
                           and row.get("active") is not False),
        }


    @app.get("/supplies/inventory")
    def supplies_inventory(
        category: Optional[str] = Query(None, description="ingredient | packaging | consumable"),
        active_only: bool = Query(False, description="Drop inactive products (default: every product)"),
        low_only: bool = Query(False, description="Only rows with is_low = true"),
        _: bool = Depends(api.verify_api_key),
    ):
        """ALL products with on_hand as the posted-only ledger SUM (zero-inventory
        products included), low_stock_threshold and is_low (on_hand < threshold;
        false when threshold is null). Alphabetical by name."""
        where, params = [], []
        if category:
            cat = category.strip().lower()
            if cat not in PRODUCT_CATEGORIES:
                raise HTTPException(
                    status_code=422,
                    detail={"error_code": "INVALID_CATEGORY",
                            "message": f"category must be one of {', '.join(PRODUCT_CATEGORIES)}",
                            "input": category},
                )
            where.append("p.type = %s")
            params.append(cat)
        if active_only:
            where.append("COALESCE(p.active, true) = true")
        sql = SUPPLIES_INVENTORY_SQL
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " GROUP BY p.id ORDER BY lower(p.name), p.id"
        with api.get_transaction() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        items = [_serialize_supplies_item(dict(r)) for r in rows]
        if low_only:
            items = [i for i in items if i["is_low"]]
        return {
            "items": items,
            "count": len(items),
            "low_count": sum(1 for i in items if i["is_low"]),
            "category": category.strip().lower() if category else None,
        }


    @app.get("/supplies/inventory/{product_id}/lots")
    def supplies_product_lots(
        product_id: int,
        include_empty: bool = Query(False, description="Also list depleted / negative lots"),
        _: bool = Depends(api.verify_api_key),
    ):
        """FIFO lot breakdown for one product: lot code, lot date (the FIFO key),
        remaining per lot as the posted-only ledger SUM, in the exact order the
        allocation paths consume lots (fifo_lot_balances)."""
        with api.get_transaction() as cur:
            cur.execute(
                "SELECT id AS product_id, name, odoo_code, type AS category, uom, active, case_size_lb, low_stock_threshold "
                "FROM products WHERE id = %s",
                (product_id,),
            )
            prod = cur.fetchone()
            if not prod:
                raise HTTPException(status_code=404, detail={"error_code": "PRODUCT_NOT_FOUND", "message": f"Product id {product_id} not found"})
            lots = api.fifo_lot_balances(cur, product_id, include_empty=include_empty)
        unit = api.ledger_quantity_unit(prod["uom"])
        out_lots = []
        for rank, l in enumerate(lots, start=1):
            out_lots.append({
                "fifo_rank": rank,
                "lot_id": l["id"],
                "lot_code": l["lot_code"],
                "lot_date": l["lot_date"].isoformat() if l.get("lot_date") else None,
                "received_at": l["received_at"].isoformat() if l.get("received_at") else None,
                "created_at": l["created_at"].isoformat() if l.get("created_at") else None,
                "entry_source": l.get("entry_source"),
                "supplier_lot_code": l.get("supplier_lot_code"),
                "lot_status": l.get("status") or "active",
                "remaining": round(float(l["available"] or 0), 4),
                "unit": unit,
            })
        total = round(sum(x["remaining"] for x in out_lots), 4)
        threshold = float(prod["low_stock_threshold"]) if prod["low_stock_threshold"] is not None else None
        return {
            "product_id": prod["product_id"],
            "name": prod["name"],
            "sku": prod["odoo_code"],
            "category": prod["category"],
            "unit": unit,
            "active": bool(prod["active"]) if prod["active"] is not None else True,
            "low_stock_threshold": threshold,
            "on_hand": total,
            "is_low": bool(threshold is not None and total < threshold),
            "fifo_order": "COALESCE(received_at, created_at) ASC",
            "include_empty": include_empty,
            "lot_count": len(out_lots),
            "lots": out_lots,
        }



    @app.post("/supply-requests", status_code=201)
    def create_supply_request(req: SupplyRequestCreate, _: bool = Depends(api.verify_api_key)):
        """Create a supply request. Exactly one of product_id / item_text (422
        otherwise). No inventory effect."""
        item_text = re.sub(r"\s+", " ", (req.item_text or "").strip()) or None
        has_product = req.product_id is not None
        if has_product and item_text:
            raise HTTPException(status_code=422, detail={
                "error_code": "SUPPLY_REQUEST_TARGET_AMBIGUOUS",
                "message": "Send either product_id or item_text, not both",
            })
        if not has_product and not item_text:
            raise HTTPException(status_code=422, detail={
                "error_code": "SUPPLY_REQUEST_TARGET_REQUIRED",
                "message": "Either product_id (catalogued product) or item_text (free text) is required",
            })
        requested_by = re.sub(r"\s+", " ", (req.requested_by or "").strip())
        if not requested_by:
            raise HTTPException(status_code=422, detail={"error_code": "REQUESTED_BY_REQUIRED", "message": "requested_by is required"})
        if req.qty is not None and req.qty <= 0:
            raise HTTPException(status_code=422, detail={"error_code": "INVALID_QUANTITY", "message": "qty must be > 0 when provided"})

        with api.get_transaction() as cur:
            if has_product:
                cur.execute("SELECT id FROM products WHERE id = %s", (req.product_id,))
                if not cur.fetchone():
                    raise HTTPException(status_code=404, detail={"error_code": "PRODUCT_NOT_FOUND", "message": f"Product id {req.product_id} not found"})
            cur.execute(
                """INSERT INTO supply_requests (product_id, item_text, qty, note, requested_by)
                   VALUES (%s, %s, %s, %s, %s) RETURNING id""",
                (req.product_id if has_product else None, item_text, req.qty,
                 (req.note or "").strip() or None, requested_by[:120]),
            )
            new_id = cur.fetchone()["id"]
            record = fetch_supply_request(cur, new_id)

        api.logger.info(f"Supply request {new_id} created by {requested_by}: {record['display_name']} qty={record['qty']}")
        return {
            "supply_request_id": new_id,
            "supply_request": record,
            "message": f"Supply request #{new_id} opened for {record['display_name']}",
        }


    @app.get("/supply-requests")
    def list_supply_requests(
        status: Optional[str] = Query("all", description="open | done | all (default all)"),
        limit: int = Query(500, ge=1, le=2000),
        _: bool = Depends(api.verify_api_key),
    ):
        """List supply requests, newest first."""
        where, params = [], []
        st = (status or "all").strip().lower()
        if st != "all":
            if st not in ("open", "done"):
                raise HTTPException(status_code=422, detail={"error_code": "INVALID_STATUS", "message": "status must be open, done or all"})
            where.append("sr.status = %s")
            params.append(st)
        sql = SUPPLY_REQUEST_SELECT_SQL
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY sr.created_at DESC, sr.id DESC LIMIT %s"
        params.append(limit)
        with api.get_transaction() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
        items = [_serialize_supply_request(dict(r)) for r in rows]
        return {
            "supply_requests": items,
            "count": len(items),
            "open_count": sum(1 for i in items if i["status"] == "open"),
            "done_count": sum(1 for i in items if i["status"] == "done"),
            "status": st,
        }


    @app.patch("/supply-requests/{supply_request_id}")
    def update_supply_request(supply_request_id: int, req: SupplyRequestUpdate, _: bool = Depends(api.verify_api_key)):
        """The only transition: open -> done (sets done_at). Anything else 409."""
        with api.get_transaction() as cur:
            cur.execute("SELECT id, status FROM supply_requests WHERE id = %s FOR UPDATE", (supply_request_id,))
            sr = cur.fetchone()
            if not sr:
                raise HTTPException(status_code=404, detail={"error_code": "SUPPLY_REQUEST_NOT_FOUND", "message": f"Supply request {supply_request_id} not found"})
            if sr["status"] != "open":
                raise HTTPException(status_code=409, detail={
                    "error_code": "SUPPLY_REQUEST_NOT_OPEN",
                    "message": f"Supply request {supply_request_id} is already {sr['status']}; only open requests can be marked done.",
                    "status": sr["status"],
                })
            cur.execute(
                "UPDATE supply_requests SET status = 'done', done_at = clock_timestamp() WHERE id = %s",
                (supply_request_id,),
            )
            record = fetch_supply_request(cur, supply_request_id)
        return {
            "supply_request_id": supply_request_id,
            "supply_request": record,
            "changed_fields": ["status", "done_at"],
            "message": f"Supply request #{supply_request_id} marked done",
        }
