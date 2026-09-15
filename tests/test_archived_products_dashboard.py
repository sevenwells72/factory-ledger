"""Archived products (products.active = false) on the finished-goods panels.

Ruling 2026-09-15 (Blubber): the four 6x8 OZ Case SKUs are discontinued and
archived. The dashboard must hide archived products, and a panel whose every
configured SKU is archived must be dropped instead of rendering as
"No inventory on hand" with a "Missing SKUs" list. A SKU name that matches
no product at all is still reported as missing, so a config typo cannot hide
behind the archive rule.
"""

from contextlib import contextmanager
import shutil
import subprocess
from pathlib import Path

import pytest
from psycopg2.extras import RealDictCursor

import main

ROOT = Path(__file__).resolve().parent.parent
JS_TEST = ROOT / "tests" / "test_archived_products_dashboard.js"


def _insert_finished(cur, *, sku, name, active, on_hand_lb=0, case_size_lb=3):
    cur.execute(
        """
        INSERT INTO products (odoo_code, name, type, active, uom, case_size_lb)
        VALUES (%s, %s, 'finished', %s, 'lb', %s)
        RETURNING id
        """,
        (sku, name, active, case_size_lb),
    )
    product_id = cur.fetchone()["id"]
    if on_hand_lb:
        cur.execute(
            "INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id",
            (product_id, f"ARCH-{sku}-LOT"),
        )
        lot_id = cur.fetchone()["id"]
        cur.execute(
            """
            INSERT INTO transactions (
                type, timestamp, status, occurred_at, business_date, operator_id
            )
            VALUES (
                'adjust',
                clock_timestamp() AT TIME ZONE 'UTC',
                'posted',
                clock_timestamp(),
                timezone('America/New_York', clock_timestamp())::date,
                'archive-test'
            )
            RETURNING id
            """
        )
        txn_id = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
            "VALUES (%s, %s, %s, %s)",
            (txn_id, product_id, lot_id, on_hand_lb),
        )
    return product_id


@pytest.fixture
def api_txn(_db_connection, monkeypatch):
    @contextmanager
    def _test_transaction():
        with _db_connection.cursor(cursor_factory=RealDictCursor) as api_cur:
            yield api_cur

    monkeypatch.setattr(main, "get_transaction", _test_transaction)
    return _db_connection


def _panels(monkeypatch, panels):
    monkeypatch.setattr(main, "_load_dashboard_config",
                        lambda: {"finished_goods_panels": panels})
    response = main.dashboard_api_finished_goods()
    assert isinstance(response, dict), response
    return {p["id"]: p for p in response["panels"]}


@pytest.mark.db
def test_all_archived_panel_is_dropped(api_txn, db_cursor, monkeypatch):
    names = [f"ARCH Test {flavor} 6x8 OZ Case" for flavor in ("A", "B", "C", "D")]
    for i, name in enumerate(names):
        # One archived SKU still carries a posted balance: archiving hides it
        # regardless, the ledger is not what the panel keys on.
        _insert_finished(db_cursor, sku=f"ARCH-{i}", name=name, active=False,
                         on_hand_lb=2406 if i == 1 else 0)
    _insert_finished(db_cursor, sku="ARCH-LIVE", name="ARCH Live 6x7 OZ Case",
                     active=True, on_hand_lb=30)

    by_id = _panels(monkeypatch, [
        {"id": "retail_bs_8oz_t", "title": "6x8 OZ Retail Cases (BS Line)",
         "case_weight_lb": None, "skus": names},
        {"id": "retail_bs_t", "title": "6x7 OZ Retail Cases (BS Line)",
         "case_weight_lb": None, "skus": ["ARCH Live 6x7 OZ Case"]},
    ])

    assert "retail_bs_8oz_t" not in by_id
    live = by_id["retail_bs_t"]
    assert [p["product_name"] for p in live["products"]] == ["ARCH Live 6x7 OZ Case"]
    assert live["missing_skus"] == []
    assert live["archived_skus"] == []


@pytest.mark.db
def test_archived_sku_hidden_but_panel_kept_when_siblings_active(api_txn, db_cursor, monkeypatch):
    _insert_finished(db_cursor, sku="ARCH-GONE", name="ARCH Gone 6x8 OZ Case",
                     active=False, on_hand_lb=12)
    _insert_finished(db_cursor, sku="ARCH-HERE", name="ARCH Here 6x8 OZ Case",
                     active=True, on_hand_lb=0)

    by_id = _panels(monkeypatch, [
        {"id": "mixed_t", "title": "Mixed", "case_weight_lb": None,
         "skus": ["ARCH Gone 6x8 OZ Case", "ARCH Here 6x8 OZ Case"]},
    ])

    panel = by_id["mixed_t"]
    assert [p["product_name"] for p in panel["products"]] == ["ARCH Here 6x8 OZ Case"]
    assert panel["missing_skus"] == []
    assert panel["archived_skus"] == ["ARCH Gone 6x8 OZ Case"]


@pytest.mark.db
def test_unmatched_sku_is_still_missing_not_archived(api_txn, db_cursor, monkeypatch):
    _insert_finished(db_cursor, sku="ARCH-ONLY", name="ARCH Only Archived Case",
                     active=False)

    by_id = _panels(monkeypatch, [
        {"id": "typo_t", "title": "Typo", "case_weight_lb": None,
         "skus": ["ARCH Only Archived Case", "ARCH Name That Matches Nothing"]},
    ])

    # Kept: the missing name must stay visible even though the other is archived.
    panel = by_id["typo_t"]
    assert panel["products"] == []
    assert panel["missing_skus"] == ["ARCH Name That Matches Nothing"]
    assert panel["archived_skus"] == ["ARCH Only Archived Case"]


@pytest.mark.db
def test_null_active_still_counts_as_active(api_txn, db_cursor, monkeypatch):
    _insert_finished(db_cursor, sku="ARCH-NULL", name="ARCH Null Active Case",
                     active=None, on_hand_lb=3)

    by_id = _panels(monkeypatch, [
        {"id": "null_t", "title": "Null", "case_weight_lb": None,
         "skus": ["ARCH Null Active Case"]},
    ])
    assert [p["product_name"] for p in by_id["null_t"]["products"]] == ["ARCH Null Active Case"]


def test_render_finished_goods_panels_js():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    result = subprocess.run(
        [node, "--test", str(JS_TEST)],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
