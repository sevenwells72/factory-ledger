"""Backend routes that still reject named-actor keys: expected to fail until PR #66 lands.

`fix/named-actor-writes` (PR #66) adds these 14 routes to the backend's actor-key
allowlist. Nothing here merges or cherry-picks that branch. Each case probes the real,
unmodified ledger with the synthetic named-actor key and is marked xfail(strict=False),
so it reports XPASS once this branch is rebased onto a main that contains PR #66.

After that rebase: drop the markers here, flip `named_actor_allowed` for the same routes
in catalog.json (and the mirrored expectation in test_writes.py's route-authorization
test), and re-run the write suite so the adapter's "Save blocked" blockers disappear.
"""

import httpx
import pytest

from factory_ledger_mcp.adapter import WRITE_CATALOG

from .ledger_harness import ACTOR_KEY, seed_write_database
from .test_writes import commit, writer  # noqa: F401  (fixture re-exported for this module)

PR66_REASON = (
    "blocked until PR #66 (fix/named-actor-writes) is merged; "
    "flips to pass after rebasing onto main"
)
ADAPTER_REASON = (
    PR66_REASON + "; additionally requires flipping named_actor_allowed in catalog.json "
    "for this route so the adapter stops blocking the commit"
)
BLOCKED_ROUTES = sorted(
    {
        (spec["method"], spec["path"])
        for specs in WRITE_CATALOG.values()
        for spec in specs
        if not spec["named_actor_allowed"]
    }
)


def probe_path(path):
    for parameter, value in [
        ("{lot_code}", "MCP-SHARED-LOT"),
        ("{lot_id}", "1"),
        ("{order_id}", "1"),
        ("{line_id}", "1"),
        ("{customer_id}", "1"),
        ("{transaction_id}", "1"),
    ]:
        path = path.replace(parameter, value)
    assert "{" not in path, path
    return path


def test_exactly_the_fourteen_pr66_routes_are_pending():
    assert len(BLOCKED_ROUTES) == 14


@pytest.mark.xfail(reason=PR66_REASON, strict=False)
@pytest.mark.parametrize("method,path", BLOCKED_ROUTES, ids=[f"{m}-{p}" for m, p in BLOCKED_ROUTES])
async def test_named_actor_key_is_authorized_on_route(ledger, method, path):
    db, url = ledger
    seed_write_database(db)
    before = db.snapshot()
    async with httpx.AsyncClient(base_url=url, trust_env=False, timeout=10) as client:
        # An invalid body probes the REAL auth dependency without persisting anything.
        response = await client.request(
            method, probe_path(path), headers={"X-API-Key": ACTOR_KEY}, json={"probe": True}
        )
    # Today: 403 "API key not authorized for this endpoint". After PR #66: validation 400/422.
    assert response.status_code != 403, response.text
    assert response.status_code in {400, 422}, response.text
    after = db.snapshot()
    after["actors"] = before["actors"]  # only last_used_at usage metadata may move
    assert after == before


@pytest.mark.xfail(reason=ADAPTER_REASON, strict=False)
async def test_create_customer_commits_through_the_adapter(writer):  # noqa: F811
    reader, db, _ = writer
    receipt = await commit(reader, "office", "createCustomer", {"name": "New Synthetic Customer"})
    assert receipt["saved"] is True
    assert db.sql("SELECT count(*) FROM customers WHERE name='New Synthetic Customer'") == "1"


@pytest.mark.xfail(reason=ADAPTER_REASON, strict=False)
async def test_rename_lot_commits_through_the_adapter(writer):  # noqa: F811
    reader, db, _ = writer
    receipt = await commit(reader, "floor", "renameLot", {"lot_id": 1, "new_lot_code": "MCP-NEW"})
    assert receipt["saved"] is True
    assert db.sql("SELECT lot_code FROM lots WHERE id=1") == "MCP-NEW"
