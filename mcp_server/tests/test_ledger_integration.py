"""Business assertions against real ledger handlers/auth and a private PostgreSQL DB."""

import copy

import pytest

from .conftest import rpc
from .ledger_harness import ACTOR_KEY


async def call(client, group, name, **arguments):
    return (await rpc(client, group, "tools/call", {"name": name, "arguments": arguments}))[
        "result"
    ]


@pytest.mark.parametrize("key,status", [("", 401), ("wrong-test-key", 403)])
async def test_real_backend_rejects_missing_and_wrong_keys(real_stack, key, status):
    # A mapped user whose actor key is blank or wrong gets the backend's own denial;
    # the adapter never substitutes a shared key.
    async with real_stack(actor_key=key) as (client, db):
        before = db.snapshot()
        result = await call(client, "office", "searchProducts", q="MCP Test")
        assert result["isError"]
        assert result["structuredContent"]["status"] == status
        assert db.snapshot() == before


@pytest.mark.parametrize("group", ["office", "floor"])
async def test_ambiguous_lot_then_product_disambiguation(real_stack, group):
    async with real_stack() as (client, _):
        result = await call(client, group, "getLotByCode", lot_code="MCP-SHARED-LOT")
        assert result["isError"]
        error = result["structuredContent"]
        assert error["status"] == 409
        assert error["details"]["error"] == "ambiguous_lot_code"
        assert {row["product_id"] for row in error["details"]["matches"]} == {1, 2}
        for product_id in (1, 2):
            resolved = await call(
                client, group, "getLotByCode", lot_code="MCP-SHARED-LOT", product_id=product_id
            )
            assert not resolved["isError"], resolved
            data = resolved["structuredContent"]["data"]
            assert data["product_id"] == product_id
            assert data["quantity_on_hand"] == 250


async def test_preview_respects_quantities_and_rejects_conflicting_inputs(real_stack):
    async with real_stack() as (client, db):
        before = db.snapshot()
        conflict = await call(
            client,
            "floor",
            "shipOrder",
            order_id="1",
            mode="preview",
            ship_all=True,
            lines=[{"line_id": 1, "quantity_lb": 10}],
        )
        assert conflict["isError"]
        assert conflict["structuredContent"]["error"] == "invalid_arguments"
        assert db.snapshot() == before  # Rejected before backend auth even stamps usage.
        for extra in ({}, {"ship_all": False}):
            preview = await call(
                client,
                "floor",
                "shipOrder",
                order_id="1",
                mode="preview",
                lines=[{"line_id": 1, "quantity_lb": 10}],
                **extra,
            )
            assert not preview["isError"], preview
            rows = preview["structuredContent"]["data"]["lines"]
            assert [
                (row["line_id"], row["requested_ship_lb"], row["can_ship_lb"]) for row in rows
            ] == [(1, 10, 10)]
        preview = await call(
            client, "floor", "shipOrder", order_id="1", mode="preview", ship_all=True
        )
        assert not preview["isError"], preview
        assert [
            (row["line_id"], row["requested_ship_lb"])
            for row in preview["structuredContent"]["data"]["lines"]
        ] == [(1, 100), (2, 100)]


# Explicit scenarios, not generated from catalog.json or the demo backend.
SHARED_READS = [
    ("searchProducts", {"q": "MCP Test"}),
    ("getBatchFormula", {"batch_id": 2}),
    ("inventoryLookup", {"q": "MCP Test"}),
    ("getLotByCode", {"lot_code": "MCP-SHARED-LOT", "product_id": 1}),
    ("traceSupplierLot", {"supplier_lot_code": "MCP-SUPPLIER-LOT"}),
    ("getTransactionHistory", {}),
    ("getDaySummary", {}),
    ("listOrders", {}),
    ("getOrder", {"order_id": "1"}),
]
READS = [(group, name, args) for group in ("office", "floor") for name, args in SHARED_READS] + [
    ("office", "listCustomers", {}),
    ("office", "searchCustomers", {"q": "MCP Test"}),
    ("office", "traceBatch", {"lot_code": "MCP-SHARED-LOT", "product_id": 2}),
    ("office", "traceIngredient", {"lot_code": "MCP-SHARED-LOT", "product_id": 1}),
    ("office", "resolveProducts", {"names": ["MCP Test Almonds"]}),
    ("floor", "listProducts", {}),
    ("floor", "getLotsBySupplierLot", {"supplier_lot_code": "MCP-SUPPLIER-LOT"}),
    ("floor", "shipOrder", {"order_id": "1", "mode": "preview", "ship_all": True}),
]


async def test_all_reads_change_only_actor_last_used_at(real_stack):
    # Phase 2: reads use the caller's named-actor key only. The former master-key
    # scenario is gone on purpose; a shared key is never sent by this service.
    async with real_stack(actor_key=ACTOR_KEY) as (client, db):
        before = db.snapshot()
        assert before["actors"][0]["last_used_at"] is None
        for group, name, arguments in READS:
            result = await call(client, group, name, **arguments)
            # The existing backend forbids actor keys on product resolution (not in PR #66).
            if name == "resolveProducts":
                assert result["isError"] and result["structuredContent"]["status"] == 403
            else:
                assert not result["isError"], (group, name, result)
        after = db.snapshot()
        assert after["actors"][0]["last_used_at"] is not None
        normalized = copy.deepcopy(after)
        normalized["actors"][0]["last_used_at"] = None
        assert normalized == before
