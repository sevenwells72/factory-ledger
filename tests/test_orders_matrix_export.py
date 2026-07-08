"""Regression coverage for the dashboard-only styled orders matrix export."""

from contextlib import contextmanager
from datetime import date
from decimal import Decimal
from io import BytesIO
from zipfile import ZipFile

from fastapi.testclient import TestClient
from openpyxl import load_workbook

import main


class _SeededCursor:
    def __init__(self, rows):
        self.rows = rows

    def execute(self, query, params=None):
        assert "/export/orders-matrix.xlsx" not in query

    def fetchall(self):
        return self.rows


def test_orders_matrix_export_workbook(monkeypatch):
    seeded_lines = [
        {
            "customer": "Sunday Customer", "order_id": "SO-TEST-001",
            "due_date": date(2026, 7, 12), "sku": "70050",
            "product_name": "Granola Classic 25 LB", "qty": Decimal("2"),
            "uom": "25 lb case",
        },
        {
            "customer": "Monday Customer", "order_id": "SO-TEST-002",
            "due_date": date(2026, 7, 13), "sku": "10001",
            "product_name": "Coconut Sweetened Flake CNS 10 LB", "qty": Decimal("3"),
            "uom": "10 lb case",
        },
        {
            "customer": "Sunday Customer", "order_id": "SO-TEST-001",
            "due_date": date(2026, 7, 12), "sku": "70073",
            "product_name": "BS Granola – Peanut Butter Banana – 6x7 OZ Case",
            "qty": Decimal("1.5"), "uom": "6x7 oz case",
        },
        {
            "customer": "Monday Customer", "order_id": "SO-TEST-002",
            "due_date": date(2026, 7, 13), "sku": "31012",
            "product_name": "Graham Cracker Crumbs 10 LB Case",
            "qty": Decimal("4"), "uom": "10 lb case",
        },
    ]

    @contextmanager
    def seeded_transaction():
        yield _SeededCursor(seeded_lines)

    monkeypatch.setattr(main, "get_transaction", seeded_transaction)
    monkeypatch.setattr(main, "API_KEY", "matrix-test-key")

    client = TestClient(main.app)
    try:
        response = client.get(
            "/export/orders-matrix.xlsx",
            headers={"X-API-Key": "matrix-test-key"},
        )
    finally:
        client.close()

    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    workbook = load_workbook(BytesIO(response.content), data_only=False)
    assert workbook.sheetnames == ["Cases", "Pounds"]

    cases = workbook["Cases"]
    pounds = workbook["Pounds"]
    assert cases.column_dimensions["D"].hidden is True
    assert cases["A1"].font.sz == 7
    assert cases.freeze_panes == "E2"

    granola_col = next(cell.column for cell in cases[1] if cell.value == "Granola Classic 25#")
    fractional_col = next(cell.column for cell in cases[1] if cell.value == "BS Granola Peanut Butter Banana 6x7 OZ")
    coconut_col = next(cell.column for cell in cases[1] if cell.value == "Coco Swt Flake CNS 10#")
    graham_col = next(cell.column for cell in cases[1] if cell.value == "Graham Cracker Crumbs 10#")
    assert cases.cell(2, granola_col).value == 2
    assert pounds.cell(2, granola_col).value == 50
    assert cases.cell(3, coconut_col).value == 3
    assert pounds.cell(3, coconut_col).value == 30
    assert cases.cell(2, fractional_col).value == 1.5
    assert pounds.cell(2, fractional_col).value == 3.9375
    assert cases.cell(2, fractional_col).number_format == '#,##0.#;(#,##0.#);"—"'
    assert pounds.cell(2, fractional_col).number_format == '#,##0.#;(#,##0.#);"—"'
    assert cases.cell(2, granola_col).number_format == '#,##0;(#,##0);"—"'
    assert "0.2 pans" in cases.cell(2, granola_col).comment.text
    assert "0.2 pans" in pounds.cell(2, granola_col).comment.text
    assert "pans" in cases.cell(2, granola_col).comment.text
    assert "pans" in pounds.cell(2, granola_col).comment.text
    assert "repack" in cases.cell(3, graham_col).comment.text.lower()
    assert "repack" in pounds.cell(3, graham_col).comment.text.lower()
    assert "<0.1 pans" in cases.cell(2, fractional_col).comment.text
    assert "<0.1 pans" in pounds.cell(2, fractional_col).comment.text
    assert "0.0 pans" not in cases.cell(2, fractional_col).comment.text
    assert "0.0 pans" not in pounds.cell(2, fractional_col).comment.text
    assert cases.cell(2, granola_col).comment.author == "Factory Ledger"
    with ZipFile(BytesIO(response.content)) as archive:
        comment_shapes = "".join(
            archive.read(name).decode("utf-8")
            for name in archive.namelist()
            if name.endswith(".vml")
        )
    assert "width:260px;height:80px" in comment_shapes.replace(" ", "")

    total_col = cases.max_column
    total_row = 4
    assert pounds.cell(total_row, total_col).value == f"=SUM(E{total_row}:{pounds.cell(total_row, total_col - 1).column_letter}{total_row})"
    assert sum(
        pounds.cell(row, col).value or 0
        for row in (2, 3)
        for col in range(5, total_col)
    ) == 123.9375
    assert cases["A3"].border.top.style == "medium"
    assert cases.auto_filter.ref.endswith("3")

    input_row = 7
    batches_row = 8
    assert "CNS Production Source of Truth" in cases.cell(input_row, granola_col).comment.text
    assert cases.cell(batches_row, granola_col).comment.text == "50 lb ÷ 322.6 lb/pan"


def test_orders_matrix_rejects_unknown_uom(monkeypatch):
    rows = [{
        "customer": "Bad UOM Customer", "order_id": "SO-BAD-UOM",
        "due_date": date(2026, 7, 13), "sku": "70999",
        "product_name": "Granola Unknown Pack", "qty": Decimal("1"),
        "uom": "mystery case",
    }]

    @contextmanager
    def seeded_transaction():
        yield _SeededCursor(rows)

    monkeypatch.setattr(main, "get_transaction", seeded_transaction)
    monkeypatch.setattr(main, "API_KEY", "matrix-test-key")
    client = TestClient(main.app)
    try:
        response = client.get(
            "/export/orders-matrix.xlsx",
            headers={"X-API-Key": "matrix-test-key"},
        )
    finally:
        client.close()
    assert response.status_code == 422
    assert response.json()["detail"]["offending_skus"] == ["70999"]
