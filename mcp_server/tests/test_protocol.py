import importlib.util
import sqlite3
from pathlib import Path

import pytest

from factory_ledger_mcp.adapter import CATALOG
from factory_ledger_mcp.demo import fixture_payloads

from .conftest import rpc

READS = [(group, spec) for group, specs in CATALOG.items() for spec in specs]


def arguments_for(spec):
    values = {
        "q": "Demo",
        "batch_id": 1,
        "lot_code": "DEMO-LOT",
        "supplier_lot_code": "DEMO-SUPPLIER",
        "order_id": "SO-DEMO-001",
        "names": ["Demo Sprinkles"],
        "mode": "preview",
    }
    return {key: values[key] for key in spec["input_schema"]["required"]}


async def test_protocol_initialize_and_group_catalogs(stack):
    async with stack() as (client, backend):
        for group, count in [("office", 14), ("floor", 12)]:
            initialized = await rpc(
                client,
                group,
                "initialize",
                {
                    "protocolVersion": "2025-11-25",
                    "capabilities": {},
                    "clientInfo": {"name": "local-test", "version": "1"},
                },
            )
            assert "tools" in initialized["result"]["capabilities"]
            tools = (await rpc(client, group, "tools/list"))["result"]["tools"]
            assert len(tools) == count
            assert {tool["name"] for tool in tools} == {spec["name"] for spec in CATALOG[group]}
            assert all(tool["annotations"]["readOnlyHint"] for tool in tools)
            assert all(tool["annotations"]["destructiveHint"] is False for tool in tools)
        assert backend.state.requests == []


@pytest.mark.parametrize("group,spec", READS, ids=[f"{g}-{s['name']}" for g, s in READS])
async def test_every_read_over_http_mcp_with_sqlite_fixtures(stack, group, spec):
    async with stack() as (client, backend):
        before = list(backend.state.db.iterdump())
        args = arguments_for(spec)
        result = (
            await rpc(client, group, "tools/call", {"name": spec["name"], "arguments": args})
        )["result"]
        assert result["isError"] is False, result
        assert result["structuredContent"]["data"] == fixture_payloads()[spec["name"]]
        assert list(backend.state.db.iterdump()) == before
        assert len(backend.state.requests) == 1
        request = backend.state.requests[0]
        assert request["method"] == spec["method"]
        expected_path = spec["path"]
        for param in spec["path_parameters"]:
            expected_path = expected_path.replace("{" + param + "}", str(args[param]))
        assert request["path"] == expected_path
        assert request["body"] == (
            {k: args[k] for k in spec["body_parameters"] if k in args}
            if spec["method"] == "POST"
            else None
        )
        assert "authorization" not in request["headers"]
        assert "x-api-key" not in request["headers"]


async def test_optional_filters_and_bilingual_fields_survive(stack):
    async with stack() as (client, backend):
        args = {
            "limit": 8,
            "since": "2026-09-01",
            "until": "2026-09-28",
            "transaction_type": "make",
            "product_name": "Demo",
        }
        await rpc(
            client, "floor", "tools/call", {"name": "getTransactionHistory", "arguments": args}
        )
        assert backend.state.requests[-1]["params"] == {k: str(v) for k, v in args.items()}
        result = (
            await rpc(
                client,
                "floor",
                "tools/call",
                {
                    "name": "getBatchFormula",
                    "arguments": {"batch_id": 1},
                },
            )
        )["result"]["structuredContent"]["data"]
        assert result["verification_notes_es"] == "Aviso"


async def test_order_not_found_remains_actionable_error(stack):
    async with stack() as (client, backend):
        result = (
            await rpc(
                client,
                "office",
                "tools/call",
                {
                    "name": "getOrder",
                    "arguments": {"order_id": "SO-NOT-FOUND"},
                },
            )
        )["result"]
        assert result["isError"]
        assert result["structuredContent"]["status"] == 404
        assert result["structuredContent"]["details"]["error"] == "ORDER_NOT_FOUND"


async def test_fixture_database_rejects_writes(stack):
    async with stack() as (_, backend):
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            backend.state.db.execute("DELETE FROM responses")


def test_catalog_matches_unmodified_source_schemas():
    path = Path(__file__).resolve().parents[1] / "scripts/build_catalog.py"
    spec = importlib.util.spec_from_file_location("build_catalog", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build() == CATALOG
