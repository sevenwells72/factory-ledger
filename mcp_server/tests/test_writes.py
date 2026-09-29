"""Write acceptance through the real, isolated ledger HTTP handlers.

Blocked creates are tested as BLOCKED, not reported as successful persistence.
No fake/demo backend can satisfy the business assertions in this module.
"""

import asyncio
import json
from pathlib import Path

import httpx
import pytest
import yaml

from factory_ledger_mcp import identity
from factory_ledger_mcp.adapter import (
    CATALOG,
    WRITE_CATALOG,
    ConfirmationStore,
    LedgerReader,
    ToolFailure,
)

from .conftest import rpc
from .ledger_harness import ACTOR_KEY, MASTER_KEY, invalid_write_probe, seed_write_database

ADMIN = identity.Identity("office@example.invalid", "admin_office_floor", ACTOR_KEY)
FLOOR = identity.Identity("floor@example.invalid", "floor", ACTOR_KEY)
HEADER = {"order_id": "SO-MCP-TEST", "notes": "Approved office note"}
SHIP = {"order_id": "SO-MCP-TEST", "lines": [{"line_id": 1, "quantity_lb": 10}]}
EXPECTED = {
    "product_id": 1,
    "supplier_name": "MCP Test Supplier",
    "expected_qty": 40,
    "reference_number": "REF-001",
}
ORDER = {
    "customer_id": 1,
    "external_order_ref": "ORD28100",
    "customer_po": "00009-B",
    "lines": [
        {"product_id": 1, "quantity": 10, "unit": "lb", "unit_price": 2},
        {"product_id": 176, "quantity": 2, "unit": "each", "unit_price": 15},
    ],
}
SAMPLES = {
    "updateSupplierLot": {
        "lot_code": "MCP-SHARED-LOT",
        "product_id": 1,
        "supplier_lot_code": "NEW-LOT",
    },
    "receive": {
        "product_name": "MCP Test Almonds",
        "cases": 1,
        "case_size_lb": 10,
        "shipper_name": "MCP Test Supplier",
        "bol_reference": "BOL1",
        "supplier_lot_code": "SUP1",
    },
    "createExpectedReceipt": EXPECTED,
    "ship": {
        "product_name": "MCP Test Almonds",
        "quantity_lb": 10,
        "customer_name": "MCP Test Customer",
        "order_reference": "REF1",
    },
    "make": {"product_name": "MCP Test Batch", "batches": 1},
    "pack": {"source_product": "MCP Test Batch", "target_product": "MCP Test Almonds", "cases": 1},
    "adjust": {
        "product_name": "MCP Test Almonds",
        "lot_code": "MCP-SHARED-LOT",
        "adjustment_lb": 1,
        "reason": "Test count",
    },
    "createCustomer": {"name": "New Synthetic Customer"},
    "updateCustomer": {"customer_id": 1, "notes": "Note"},
    "createOrder": ORDER,
    "updateOrderHeader": HEADER,
    "updateOrderStatus": {"order_id": "SO-MCP-TEST", "status": "cancelled"},
    "addOrderLines": {
        "order_id": "SO-MCP-TEST",
        "lines": [{"product_name": "MCP Test Almonds", "quantity_lb": 10}],
    },
    "cancelOrderLine": {"order_id": "SO-MCP-TEST", "line_id": 1},
    "updateOrderLine": {
        "order_id": "SO-MCP-TEST",
        "line_id": 1,
        "quantity_lb": 90,
        "unit_price": 0,
    },
    "shipOrder": SHIP,
    "commitShipOrder": SHIP,
    "renameLot": {"lot_id": 1, "new_lot_code": "MCP-NEW-LOT"},
    "voidTransaction": {"transaction_id": 1, "reason": "Test correction"},
}


def business_snapshot(db):
    snapshot = db.snapshot()
    for actor in snapshot["actors"]:
        actor["last_used_at"] = None
    return snapshot


@pytest.fixture
async def writer(ledger, tmp_path, monkeypatch):
    db, url = ledger
    seed_write_database(db)
    monkeypatch.setattr(identity, "current_identity", lambda: ADMIN)
    clock = [1_800_000_000.0]
    store = ConfirmationStore(tmp_path / "confirmations.sqlite3", clock=lambda: clock[0])
    # Deliberately poisonous shared-key default: all write flow requests must override it.
    async with httpx.AsyncClient(
        base_url=url, headers={"X-API-Key": MASTER_KEY}, trust_env=False
    ) as client:
        yield LedgerReader(client, store), db, clock


async def commit(reader, group, name, args, preview=None):
    preview = preview or await reader.call(group, name, args)
    return await reader.call(
        group, name, args | {"phase": "commit", "confirmation_token": preview["confirmation_token"]}
    )


def assert_error(exc, code):
    assert exc.value.payload["error"] == code, exc.value.payload


def test_write_inventory_and_read_catalog_remain_separate():
    root = Path(__file__).resolve().parents[2]
    for group, source, count in [
        ("office", "openapi-gpt-v3.yaml", 16),
        ("floor", "gpt-configs/schemas/openapi-floor.yaml", 10),
    ]:
        doc = yaml.safe_load((root / source).read_text())
        expected = {
            op["operationId"]: (method.upper(), path)
            for path, methods in doc["paths"].items()
            for method, op in methods.items()
            if method in {"post", "patch"}
            and op["operationId"] not in {s["name"] for s in CATALOG[group]}
        }
        assert {s["name"]: (s["method"], s["path"]) for s in WRITE_CATALOG[group]} == expected
        assert len(expected) == count


async def test_all_26_writes_require_tokens_before_any_backend_request(writer):
    reader, db, _ = writer
    before = db.snapshot()
    for group, specs in WRITE_CATALOG.items():
        for spec in specs:
            with pytest.raises(ToolFailure) as exc:
                await reader.call(group, spec["name"], SAMPLES[spec["name"]] | {"phase": "commit"})
            assert_error(exc, "confirmation_required")
    assert db.snapshot() == before


async def test_all_underlying_write_routes_named_actor_authorization(writer):
    reader, db, _ = writer
    before = business_snapshot(db)
    seen = set()
    for specs in WRITE_CATALOG.values():
        for spec in specs:
            key = (spec["method"], spec["path"])
            if key in seen:
                continue
            seen.add(key)
            assert spec["named_actor_allowed"], spec["name"]
            path, body = invalid_write_probe(spec["path"])
            # Invalid/missing required fields safely probe the REAL auth dependency.
            response = await reader.client.request(
                spec["method"],
                path,
                headers={"X-API-Key": ACTOR_KEY},
                json=body,
            )
            assert response.status_code in {400, 422}, (spec["name"], response.text)
    assert len(seen) == 19
    assert business_snapshot(db) == before


async def test_all_previews_are_read_only_and_blocked_routes_never_save(writer):
    reader, db, _ = writer
    before = business_snapshot(db)
    for group, specs in WRITE_CATALOG.items():
        for spec in specs:
            args = SAMPLES[spec["name"]]
            preview = await reader.call(group, spec["name"], args)
            assert not preview["saved"]
            assert preview["summary"] and preview["confirmation_token"]
            if spec["name"] == "createOrder":
                assert not preview["can_commit"]
                with pytest.raises(ToolFailure) as exc:
                    await commit(reader, group, spec["name"], args, preview)
                assert_error(exc, "backend_write_blocked")
            else:
                assert preview["can_commit"], (group, spec["name"], preview)
    assert business_snapshot(db) == before


async def test_token_is_payload_user_operation_group_bound_and_expires(writer, monkeypatch):
    reader, db, clock = writer
    preview = await reader.call("office", "updateOrderHeader", HEADER)
    before = business_snapshot(db)
    for args in [HEADER | {"notes": "Changed"}, HEADER | {"notes_es": "Nueva"}]:
        with pytest.raises(ToolFailure) as exc:
            await commit(reader, "office", "updateOrderHeader", args, preview)
        assert_error(exc, "invalid_confirmation")
    monkeypatch.setattr(
        identity,
        "current_identity",
        lambda: identity.Identity(
            "other@example.invalid", "admin_office_floor", "mcp-test-other-actor-key"
        ),
    )
    with pytest.raises(ToolFailure) as exc:
        await commit(reader, "office", "updateOrderHeader", HEADER, preview)
    assert_error(exc, "invalid_confirmation")
    monkeypatch.setattr(identity, "current_identity", lambda: ADMIN)
    for group, name, args in [
        ("floor", "updateOrderStatus", SAMPLES["updateOrderStatus"]),
        ("office", "updateOrderStatus", SAMPLES["updateOrderStatus"]),
    ]:
        with pytest.raises(ToolFailure) as exc:
            await commit(reader, group, name, args, preview)
        assert_error(exc, "invalid_confirmation")
    with monkeypatch.context() as patch:
        patch.setattr(reader, "environment", "different-environment")
        with pytest.raises(ToolFailure) as exc:
            await commit(reader, "office", "updateOrderHeader", HEADER, preview)
        assert_error(exc, "invalid_confirmation")
    clock[0] += 120
    with pytest.raises(ToolFailure) as exc:
        await commit(reader, "office", "updateOrderHeader", HEADER, preview)
    assert_error(exc, "confirmation_expired")
    assert business_snapshot(db) == before


async def test_durable_token_single_use_concurrent_call_and_restart(writer):
    reader, db, clock = writer
    preview = await reader.call("office", "createExpectedReceipt", EXPECTED)
    # Independent store connection simulates another worker/restarted service.
    restarted = LedgerReader(
        reader.client, ConfirmationStore(reader.confirmations.path, clock=lambda: clock[0])
    )
    results = await asyncio.gather(
        commit(reader, "office", "createExpectedReceipt", EXPECTED, preview),
        commit(restarted, "office", "createExpectedReceipt", EXPECTED, preview),
        return_exceptions=True,
    )
    receipts = [r for r in results if isinstance(r, dict)]
    failures = [r for r in results if isinstance(r, ToolFailure)]
    assert len(receipts) == len(failures) == 1, results
    assert failures[0].payload["error"] == "confirmation_used"
    receipt = receipts[0]
    assert receipt["actor"] == {
        "email": ADMIN.email,
        "role": ADMIN.role,
        "name": "Synthetic MCP actor",
    }
    assert receipt["order_number"] is None
    assert db.sql("SELECT count(*) FROM expected_receipts") == "1"
    assert db.sql("SELECT created_by FROM expected_receipts") == "Synthetic MCP actor"
    with reader.confirmations.connect() as journal:
        record = journal.execute(
            "SELECT * FROM confirmations WHERE approval_id=?", (preview["approval_id"],)
        ).fetchone()
        assert record["state"] == "saved"
        assert json.loads(record["receipt"]) == receipt
    persisted = reader.confirmations.path.read_bytes()
    assert ACTOR_KEY.encode() not in persisted
    assert preview["confirmation_token"].encode() not in persisted
    with pytest.raises(ToolFailure) as exc:
        await commit(restarted, "office", "createExpectedReceipt", EXPECTED, preview)
    assert_error(exc, "confirmation_used")


async def test_stale_order_inventory_and_permission_changes_refuse_commit(writer, monkeypatch):
    reader, db, _ = writer
    preview = await reader.call("office", "updateOrderHeader", HEADER)
    db.sql("UPDATE sales_orders SET notes='Concurrent edit' WHERE id=1")
    with pytest.raises(ToolFailure) as exc:
        await commit(reader, "office", "updateOrderHeader", HEADER, preview)
    assert_error(exc, "stale_confirmation")
    preview = await reader.call("floor", "commitShipOrder", SHIP)
    db.sql("""WITH t AS (INSERT INTO transactions (type) VALUES ('adjust') RETURNING id)
        INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
        SELECT id, 1, 1, -10 FROM t""")
    with pytest.raises(ToolFailure) as exc:
        await commit(reader, "floor", "commitShipOrder", SHIP, preview)
    assert_error(exc, "stale_confirmation")
    preview = await reader.call("office", "updateOrderHeader", HEADER)
    monkeypatch.setattr(identity, "current_identity", lambda: FLOOR)
    with pytest.raises(ToolFailure) as exc:
        await commit(reader, "office", "updateOrderHeader", HEADER, preview)
    assert_error(exc, "forbidden")
    assert db.sql("SELECT notes FROM sales_orders WHERE id=1") == "Concurrent edit"
    assert db.sql("SELECT count(*) FROM shipments") == "0"


@pytest.mark.parametrize(
    "person", [None, identity.Identity("legacy@example.invalid", "admin", ACTOR_KEY), FLOOR]
)
async def test_office_writes_deny_missing_legacy_and_floor_roles(writer, monkeypatch, person):
    reader, db, _ = writer

    def resolve():
        if person is None:
            raise PermissionError("no user")
        return person

    monkeypatch.setattr(identity, "current_identity", resolve)
    before = db.snapshot()
    with pytest.raises(ToolFailure) as exc:
        await reader.call("office", "updateOrderHeader", HEADER)
    assert_error(exc, "unauthenticated" if person is None else "forbidden")
    assert db.snapshot() == before


async def test_shared_key_mapping_and_caller_supplied_identity_are_rejected(writer, monkeypatch):
    reader, db, _ = writer
    before = business_snapshot(db)
    for key in ["actor", "created_by", "confirmed", "mode"]:
        with pytest.raises(ToolFailure) as exc:
            await reader.call("office", "createExpectedReceipt", EXPECTED | {key: "spoofed"})
        assert_error(exc, "invalid_arguments")
    monkeypatch.setattr(
        identity, "current_identity", lambda: identity.Identity(ADMIN.email, ADMIN.role, MASTER_KEY)
    )
    with pytest.raises(ToolFailure) as exc:
        await reader.call("office", "createExpectedReceipt", EXPECTED)
    assert_error(exc, "named_actor_required")
    assert business_snapshot(db) == before


async def test_order_header_line_and_status_receipts_keep_so_po_and_actor(writer):
    reader, db, _ = writer
    for name, args in [
        ("updateOrderHeader", HEADER),
        ("updateOrderLine", SAMPLES["updateOrderLine"]),
        ("updateOrderStatus", SAMPLES["updateOrderStatus"]),
    ]:
        receipt = await commit(reader, "office", name, args)
        assert receipt["success"] and receipt["saved"]
        assert receipt["order_number"] == "SO-MCP-TEST"
        assert receipt["order_id"] == 1
        assert receipt["customer_po"] == "0007-A"
        assert receipt["flags"] == []
        assert receipt["verification"] == ("pending" if name == "updateOrderLine" else "verified")
        assert receipt["actor"]["name"] == "Synthetic MCP actor"
    assert db.sql("SELECT notes FROM sales_orders WHERE id=1") == HEADER["notes"]
    assert db.sql("SELECT unit_price FROM sales_order_lines WHERE id=1") == "0.0000"
    assert db.sql("SELECT state_changed_by FROM sales_orders WHERE id=1") == "Synthetic MCP actor"


async def test_floor_shipping_commits_only_explicit_ten_pounds_and_returns_real_ids(
    writer, monkeypatch
):
    reader, db, _ = writer
    monkeypatch.setattr(identity, "current_identity", lambda: FLOOR)
    preview = await reader.call("floor", "commitShipOrder", SHIP)
    assert [r["requested_ship_lb"] for r in preview["proposal"]["shipping_preview"]["lines"]] == [
        10
    ]
    receipt = await commit(reader, "floor", "commitShipOrder", SHIP, preview)
    assert receipt["order_number"] == "SO-MCP-TEST" and receipt["order_id"] == 1
    assert db.sql("SELECT quantity_shipped_lb FROM sales_order_lines WHERE id=1") == "10.0000"
    assert db.sql("SELECT quantity_shipped_lb FROM sales_order_lines WHERE id=2") == "0.0000"
    assert str(receipt["shipment_id"]) == db.sql("SELECT id FROM shipments")
    assert receipt["transaction_ids"] == [
        int(db.sql("SELECT id FROM transactions WHERE type='ship'"))
    ]
    assert len(receipt["confirmation_codes"]) == 1
    assert receipt["actor"]["email"] == FLOOR.email


async def test_shipping_all_preserves_pallet_service_without_inventory_movements(writer):
    reader, db, _ = writer
    receipt = await commit(reader, "floor", "commitShipOrder", {"order_id": "1", "ship_all": True})
    assert receipt["success"]
    assert db.sql("SELECT line_status FROM sales_order_lines WHERE product_id=176") == "fulfilled"
    assert db.sql("SELECT count(*) FROM transaction_lines WHERE product_id=176") == "0"
    assert (
        db.sql("SELECT quantity_lb * unit_price FROM sales_order_lines WHERE product_id=176")
        == "30.00000000"
    )
    assert receipt["resulting_state"]["totals"]["total_ordered_lb"] == 200


@pytest.mark.parametrize(
    "extra",
    [
        {"ship_all": True, "lines": [{"line_id": 1, "quantity_lb": 10}]},
        {"lines": []},
        {"lines": [{"line_id": 1, "quantity_lb": 10}, {"line_id": 1, "quantity_lb": 10}]},
        {},
    ],
)
async def test_shipping_conflicts_empty_and_duplicate_lines_never_reach_backend(writer, extra):
    reader, db, _ = writer
    before = db.snapshot()
    with pytest.raises(ToolFailure) as exc:
        await reader.call("floor", "commitShipOrder", {"order_id": "1"} | extra)
    assert_error(exc, "invalid_arguments")
    assert db.snapshot() == before


async def test_create_order_proposal_pallet_po_no_po_and_backend_gap(writer):
    reader, db, _ = writer
    before = business_snapshot(db)
    for po in ["00009-B", None, ""]:
        args = ORDER | {"customer_po": po}
        preview = await reader.call("office", "createOrder", args)
        proposed = preview["proposal"]
        pallet = proposed["lines"][1]
        assert (
            pallet["product_id"],
            pallet["product"],
            pallet["unit"],
            pallet["quantity"],
            pallet["unit_price"],
            pallet["amount"],
        ) == (176, "Pallet Charge", "each", 2, 15, 30)
        assert "Pallet Charge" in preview["summary"]
        assert proposed["customer_po"] == (po or None)
        assert proposed["flags"] == ([] if po else ["No PO"])
        if not po:
            assert "No PO" in preview["summary"]
        assert not preview["can_commit"]
        with pytest.raises(ToolFailure) as exc:
            await commit(reader, "office", "createOrder", args, preview)
        assert_error(exc, "backend_write_blocked")
    assert business_snapshot(db) == before


async def test_external_reference_duplicate_rejected_before_create(writer):
    reader, db, _ = writer
    db.sql("UPDATE sales_orders SET notes='Customer confirmation ORD28100' WHERE id=1")
    before = business_snapshot(db)
    with pytest.raises(ToolFailure) as exc:
        await reader.call("office", "createOrder", ORDER)
    assert_error(exc, "duplicate_order")
    assert exc.value.payload["details"]["order_number"] == "SO-MCP-TEST"
    assert business_snapshot(db) == before


async def test_po_duplicate_and_changed_po_refuse_unsafe_save(writer):
    reader, db, _ = writer
    with pytest.raises(ToolFailure) as exc:
        await reader.call("office", "createOrder", ORDER | {"customer_po": " 0007-a "})
    assert_error(exc, "duplicate_po")
    args = HEADER | {"customer_po": "NEW-PO"}
    preview = await reader.call("office", "updateOrderHeader", args)
    assert not preview["can_commit"]
    with pytest.raises(ToolFailure) as exc:
        await commit(reader, "office", "updateOrderHeader", args, preview)
    assert_error(exc, "backend_write_blocked")
    assert db.sql("SELECT customer_po FROM sales_orders WHERE id=1") == "0007-A"
    db.sql("UPDATE sales_orders SET customer_po=NULL WHERE id=1")
    receipt = await commit(reader, "office", "updateOrderHeader", HEADER)
    assert receipt["customer_po"] is None
    assert receipt["flags"] == ["No PO"]
    assert "No PO" in (await reader.call("office", "updateOrderHeader", HEADER))["summary"]


async def test_catalog_permissions_and_real_write_through_mcp(real_stack, monkeypatch):
    monkeypatch.setattr(identity, "current_identity", lambda: FLOOR)
    async with real_stack() as (client, db):
        seed_write_database(db)
        for group, count in [("office", 14), ("floor", 22)]:
            listed = (await rpc(client, group, "tools/list"))["result"]["tools"]
            assert len(listed) == count
            assert all(
                t["annotations"]["readOnlyHint"]
                == (t["name"] in {s["name"] for s in CATALOG[group]})
                for t in listed
            )
        monkeypatch.setattr(identity, "current_identity", lambda: ADMIN)
        listed = (await rpc(client, "office", "tools/list"))["result"]["tools"]
        assert len(listed) == 30
        params = {"name": "updateOrderHeader", "arguments": HEADER}
        preview = (await rpc(client, "office", "tools/call", params))["result"][
            "structuredContent"
        ]["data"]
        params["arguments"] = HEADER | {
            "phase": "commit",
            "confirmation_token": preview["confirmation_token"],
        }
        result = (await rpc(client, "office", "tools/call", params))["result"]
        assert not result["isError"], result
        assert result["structuredContent"]["data"]["order_number"] == "SO-MCP-TEST"


class FaultAfterRealWrite(httpx.AsyncBaseTransport):
    """Forward to real HTTP handlers; inject only the response-loss boundary."""

    def __init__(self, *, lose_write=False, lose_readback=False):
        self.network = httpx.AsyncHTTPTransport(retries=0)
        self.lose_write = lose_write
        self.lose_readback = lose_readback
        self.sent = False
        self.requests = []

    async def handle_async_request(self, request):
        self.requests.append((request.method, request.url.path, request.headers["x-api-key"]))
        if self.sent and self.lose_readback and request.method == "GET":
            return httpx.Response(503, json={"error": "synthetic failed readback"})
        response = await self.network.handle_async_request(request)
        if request.method in {"PATCH", "POST"} and not request.url.path.endswith("/preview"):
            self.sent = True
            await response.aread()
            if self.lose_write:
                await response.aclose()
                raise httpx.ReadTimeout("Dropped after real commit", request=request)
        return response

    async def aclose(self):
        await self.network.aclose()


async def test_dropped_response_persists_uncertainty_and_never_retries(writer):
    reader, db, _ = writer
    transport = FaultAfterRealWrite(lose_write=True)
    async with httpx.AsyncClient(
        base_url=reader.client.base_url, transport=transport, headers={"X-API-Key": MASTER_KEY}
    ) as client:
        failing = LedgerReader(client, reader.confirmations)
        preview = await failing.call("office", "createExpectedReceipt", EXPECTED)
        with pytest.raises(ToolFailure) as exc:
            await commit(failing, "office", "createExpectedReceipt", EXPECTED, preview)
        assert_error(exc, "write_outcome_uncertain")
        assert db.sql("SELECT count(*) FROM expected_receipts") == "1"
        with pytest.raises(ToolFailure) as exc:
            await commit(failing, "office", "createExpectedReceipt", EXPECTED, preview)
        assert_error(exc, "confirmation_used")
        with pytest.raises(ToolFailure) as exc:
            await failing.call("office", "createExpectedReceipt", EXPECTED)
        assert_error(exc, "reconciliation_required")
    assert len([r for r in transport.requests if r[0] == "POST"]) == 1
    assert {r[2] for r in transport.requests} == {ACTOR_KEY}
    with reader.confirmations.connect() as journal:
        assert journal.execute("SELECT state FROM confirmations").fetchone()[0] == "uncertain"


async def test_readback_failure_returns_saved_pending_with_so_and_no_second_mutation(writer):
    reader, db, _ = writer
    transport = FaultAfterRealWrite(lose_readback=True)
    async with httpx.AsyncClient(base_url=reader.client.base_url, transport=transport) as client:
        failing = LedgerReader(client, reader.confirmations)
        receipt = await commit(failing, "office", "updateOrderHeader", HEADER)
    assert receipt["saved"] and receipt["success"]
    assert receipt["verification"] == "pending"
    assert receipt["order_number"] == "SO-MCP-TEST"
    assert "Saved; verification pending" in receipt["message"]
    assert db.sql("SELECT notes FROM sales_orders WHERE id=1") == HEADER["notes"]
    assert len([r for r in transport.requests if r[0] == "PATCH"]) == 1
    assert {r[2] for r in transport.requests} == {ACTOR_KEY}


async def test_expiry_during_precommit_validation_consumes_without_saving(writer, monkeypatch):
    reader, db, clock = writer
    preview = await reader.call("office", "updateOrderHeader", HEADER)
    original = reader._prepare

    async def delayed(*args):
        result = await original(*args)
        clock[0] += 121
        return result

    monkeypatch.setattr(reader, "_prepare", delayed)
    before = business_snapshot(db)
    with pytest.raises(ToolFailure) as exc:
        await commit(reader, "office", "updateOrderHeader", HEADER, preview)
    assert_error(exc, "confirmation_expired")
    assert business_snapshot(db) == before


async def test_stale_fifo_lot_date_and_actor_rotation_invalidate_approval(writer, monkeypatch):
    reader, db, _ = writer
    preview = await reader.call("floor", "commitShipOrder", SHIP)
    db.sql("UPDATE lots SET received_at=COALESCE(received_at, now()) - interval '1 day' WHERE id=1")
    with pytest.raises(ToolFailure) as exc:
        await commit(reader, "floor", "commitShipOrder", SHIP, preview)
    assert_error(exc, "stale_confirmation")
    preview = await reader.call("floor", "commitShipOrder", SHIP)
    monkeypatch.setattr(
        identity,
        "current_identity",
        lambda: identity.Identity(ADMIN.email, ADMIN.role, "mcp-test-other-actor-key"),
    )
    with pytest.raises(ToolFailure) as exc:
        await commit(reader, "floor", "commitShipOrder", SHIP, preview)
    assert_error(exc, "invalid_confirmation")
    assert db.sql("SELECT count(*) FROM shipments") == "0"


async def test_zero_pallet_price_is_preserved_and_wrong_product_flag_is_rejected(writer):
    reader, db, _ = writer
    args = ORDER | {"lines": [ORDER["lines"][1] | {"unit_price": 0}]}
    preview = await reader.call("office", "createOrder", args)
    assert preview["proposal"]["lines"][0]["amount"] == 0
    db.sql("UPDATE products SET is_service=false WHERE id=176")
    with pytest.raises(ToolFailure) as exc:
        await reader.call("office", "createOrder", args)
    assert_error(exc, "invalid_pallet_product")


async def test_po_header_proposal_shows_requested_dedicated_field(writer):
    reader, _, _ = writer
    preview = await reader.call("office", "updateOrderHeader", HEADER | {"customer_po": "NEW-0009"})
    assert preview["proposal"]["customer_po"] == "NEW-0009"
    assert "NEW-0009" in preview["summary"]
    assert not preview["can_commit"]


async def test_mcp_write_resolves_identity_once_and_sanitizes_validation(real_stack, monkeypatch):
    calls = []

    def resolve():
        calls.append(1)
        return ADMIN

    monkeypatch.setattr(identity, "current_identity", resolve)
    async with real_stack() as (client, _):
        result = (
            await rpc(
                client,
                "office",
                "tools/call",
                {
                    "name": "updateOrderHeader",
                    "arguments": HEADER | {"actor_key": "secret-should-not-echo"},
                },
            )
        )["result"]
    assert result["isError"]
    assert calls == [1]
    assert "secret-should-not-echo" not in json.dumps(result)
