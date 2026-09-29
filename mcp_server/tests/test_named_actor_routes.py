"""PR #66 route authorization and adapter commits against the disposable real ledger."""

import httpx
import pytest

from factory_ledger_mcp.adapter import WRITE_CATALOG

from .ledger_harness import ACTOR_KEY, LedgerDatabase, invalid_write_probe, seed_write_database
from .test_writes import (  # noqa: F401  (fixture re-exported for this module)
    business_snapshot,
    commit,
    writer,
)

PR66_ROUTES = [
    ("PATCH", "/customers/{customer_id}"),
    ("PATCH", "/lots/{lot_code}/supplier-lot"),
    ("PATCH", "/lots/{lot_id}/rename"),
    ("PATCH", "/sales/orders/{order_id}/lines/{line_id}/cancel"),
    ("POST", "/adjust"),
    ("POST", "/customers"),
    ("POST", "/make"),
    ("POST", "/pack"),
    ("POST", "/receive"),
    ("POST", "/sales/orders"),
    ("POST", "/sales/orders/{order_id}/lines"),
    ("POST", "/sales/orders/{order_id}/ship"),
    ("POST", "/ship"),
    ("POST", "/void/{transaction_id}"),
]


def test_exactly_the_fourteen_pr66_routes_are_enabled():
    enabled = [
        spec
        for specs in WRITE_CATALOG.values()
        for spec in specs
        if (spec["method"], spec["path"]) in PR66_ROUTES
    ]
    assert len(enabled) == 20
    assert {(s["method"], s["path"]) for s in enabled} == set(PR66_ROUTES)
    assert all(s["named_actor_allowed"] for s in enabled)


@pytest.mark.parametrize("method,path", PR66_ROUTES, ids=[f"{m}-{p}" for m, p in PR66_ROUTES])
async def test_named_actor_key_is_authorized_on_route(ledger, method, path):
    db, url = ledger
    seed_write_database(db)
    before = business_snapshot(db)
    path, body = invalid_write_probe(path)
    async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=10) as client:
        # An invalid body probes the REAL auth dependency without persisting anything.
        response = await client.request(method, path, headers={"X-API-Key": ACTOR_KEY}, json=body)
    assert response.status_code != 403, response.text
    assert response.status_code in {400, 422}, response.text
    assert business_snapshot(db) == before


async def test_create_customer_commits_through_the_adapter(writer):  # noqa: F811
    reader, db, _ = writer
    receipt = await commit(reader, "office", "createCustomer", {"name": "New Synthetic Customer"})
    assert receipt["saved"] is True
    assert db.sql("SELECT count(*) FROM customers WHERE name='New Synthetic Customer'") == "1"
    assert (
        db.sql(
            "SELECT count(*) FROM actor_write_audit a JOIN customers c ON c.id=a.target_id "
            "WHERE a.target_table='customers' AND a.actor_id=1 AND a.route='/customers' "
            "AND c.name='New Synthetic Customer'"
        )
        == "1"
    )


async def test_rename_lot_commits_through_the_adapter(writer):  # noqa: F811
    reader, db, _ = writer
    receipt = await commit(reader, "floor", "renameLot", {"lot_id": 1, "new_lot_code": "MCP-NEW"})
    assert receipt["saved"] is True
    assert db.sql("SELECT lot_code FROM lots WHERE id=1") == "MCP-NEW"
    assert (
        db.sql(
            "SELECT count(*) FROM actor_write_audit WHERE target_table='lots' "
            "AND target_id=1 AND actor_id=1 AND route='/lots/{lot_id}/rename'"
        )
        == "1"
    )


def test_disposable_schema_includes_056(postgres):
    # Inspect the template loaded by schema.sql, before any ledger process starts.
    db = LedgerDatabase(postgres.pg_bin, postgres.socket_dir, "mcp_template")
    assert (
        db.sql("SELECT count(*) FROM migration_markers WHERE name='056_actor_write_audit'") == "1"
    )
    assert db.sql("SELECT count(*) FROM actor_write_audit") == "0"
    assert (
        db.sql(
            "SELECT count(*) FROM pg_trigger WHERE tgrelid='actor_write_audit'::regclass "
            "AND tgname='actor_write_audit_append_only' AND tgenabled='O'"
        )
        == "1"
    )


@pytest.mark.parametrize("role", ["floor", "office", "owner"])
async def test_all_admin_routes_remain_blocked(ledger, role):
    db, url = ledger
    db.sql(f"UPDATE actors SET role='{role}' WHERE id=1")
    before = business_snapshot(db)
    routes = [
        ("DELETE", "/admin/bom/lines/1"),
        ("DELETE", "/admin/product-bom/1"),
        ("GET", "/admin/bom/search"),
        ("GET", "/admin/bom/1/lines"),
        ("GET", "/admin/lots/duplicates"),
        ("GET", "/admin/product-bom"),
        ("POST", "/admin/bom/1/lines"),
        ("POST", "/admin/lots/merge"),
        ("POST", "/admin/product-bom"),
        ("PUT", "/admin/bom/lines/1"),
        ("PUT", "/admin/products/1"),
    ]
    async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=10) as client:
        for method, path in routes:
            response = await client.request(
                method, path, headers={"X-API-Key": ACTOR_KEY}, json={"unsupported_probe": True}
            )
            assert response.status_code == 403, (role, method, path, response.text)
            assert response.json()["detail"] == "API key not authorized for this endpoint"
    assert business_snapshot(db) == before
