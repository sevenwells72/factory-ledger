"""Synthetic SQLite response fixtures, NOT the real ledger or business logic.

Only this demonstration process creates a temporary in-memory database. The MCP
adapter never opens a database. No environment files, keys or URLs are loaded.
"""

import json
import sqlite3
from contextlib import asynccontextmanager

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

from .adapter import CATALOG


def fixture_payloads():
    product = {
        "id": 1,
        "name": "Demo Sprinkles",
        "odoo_code": "DEMO-001",
        "case_size_lb": 10,
        "default_batch_lb": 100,
        "match_tier": "exact",
    }
    lot = {
        "id": 1,
        "lot_code": "DEMO-LOT",
        "product_id": 1,
        "quantity_on_hand": 100,
        "supplier_lot_code": "DEMO-SUPPLIER",
    }
    order = {
        "id": 1,
        "order_number": "SO-DEMO-001",
        "customer_po": "000123-DEMO",
        "status": "confirmed",
        "customer_name": "Demo Customer",
        "lines": [
            {"id": 1, "product_id": 1, "quantity_lb": 10, "quantity_shipped_lb": 0},
            {"id": 2, "product_id": 176, "is_service": True, "quantity_units": 1},
        ],
    }
    customer = {"id": 1, "name": "Demo Customer"}
    return {
        "searchProducts": {"count": 1, "products": [product]},
        "resolveProducts": {
            "resolved": [{"input": "Demo Sprinkles", "match": product}],
            "summary": {"total": 1, "resolved": 1, "unresolved": 0},
        },
        "listProducts": [product],
        "getBatchFormula": {
            "ingredients": [{"product_id": 1, "quantity_lb": 100}],
            "verification_notes": "Demo warning",
            "verification_notes_es": "Aviso",
        },
        "inventoryLookup": {
            "query": "Demo",
            "results": [{"product": product, "lots": [lot], "total_on_hand": 100}],
        },
        "getLotByCode": lot,
        "getLotsBySupplierLot": {"lots": [lot]},
        "traceBatch": {"lot_code": "DEMO-LOT", "ingredients": [lot]},
        "traceIngredient": {"lot_code": "DEMO-LOT", "usage": []},
        "traceSupplierLot": {"supplier_lot_code": "DEMO-SUPPLIER", "lots": [lot]},
        "getTransactionHistory": {"transactions": [{"id": 1, "type": "receive"}]},
        "getDaySummary": {
            "date": "2026-09-28",
            "batch_products": [],
            "finished_goods": [],
            "adjustments": [],
        },
        "listCustomers": [customer],
        "searchCustomers": {"customers": [customer]},
        "listOrders": {"orders": [order]},
        "getOrder": order,
        "shipOrder": {
            "mode": "preview",
            "order_number": "SO-DEMO-001",
            "lines": [],
            "warnings": ["Synthetic preview only"],
            "message": "No changes made",
        },
    }


def create_demo_app():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE responses (name TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    db.executemany(
        "INSERT INTO responses VALUES (?, ?)",
        [(name, json.dumps(payload)) for name, payload in fixture_payloads().items()],
    )
    db.commit()
    db.execute("PRAGMA query_only = ON")
    requests = []

    def endpoint(spec):
        async def read(request):
            body = await request.json() if request.method == "POST" else None
            requests.append(
                {
                    "method": request.method,
                    "path": request.url.path,
                    "params": dict(request.query_params),
                    "body": body,
                    "headers": dict(request.headers),
                }
            )
            if spec["name"] == "shipOrder" and body.get("mode") != "preview":
                return JSONResponse({"error": "demo rejects writes"}, status_code=400)
            if spec["name"] == "getOrder" and request.path_params["order_id"] not in {
                "1",
                "SO-DEMO-001",
            }:
                return JSONResponse({"error": "ORDER_NOT_FOUND"}, status_code=404)
            row = db.execute(
                "SELECT payload FROM responses WHERE name = ?", (spec["name"],)
            ).fetchone()
            return JSONResponse(json.loads(row[0]), headers={"X-Synthetic-Fixture": "true"})

        return read

    routes, seen = [], set()
    for specs in CATALOG.values():
        for spec in specs:
            key = (spec["path"], spec["method"])
            if key not in seen:
                routes.append(Route(spec["path"], endpoint(spec), methods=[spec["method"]]))
                seen.add(key)

    @asynccontextmanager
    async def lifespan(app):
        yield
        db.close()

    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.db = db
    app.state.requests = requests
    return app


def main():
    uvicorn.run(create_demo_app(), host="127.0.0.1", port=8100)


if __name__ == "__main__":
    main()
