#!/usr/bin/env python3
"""Propose or apply pack_format values for active production-board SKUs.

The default mode is read-only: it reads the active SKU catalog, applies the
approved name heuristics to granola products, and emits a Markdown table for
human review. Non-granola products are intentionally proposed as NULL.

The explicit --apply mode transactionally writes the approved mapping after
validating that migration 040 and its constraint exist. SKU 70004 is the sole
approved unclassified granola SKU and intentionally remains NULL because bulk
per-lb product is excluded from case counts.

Usage:
    DATABASE_URL=postgresql://... python3 scripts/propose_pack_format_backfill.py
    DATABASE_URL=postgresql://... python3 scripts/propose_pack_format_backfill.py --apply

If DATABASE_URL is not set, the script also checks a .env file at the repository
root. The production-board catalog currently contains 58 entries even though its
source comment says 57.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor


REPO_ROOT = Path(__file__).resolve().parents[1]

# Mirrors dashboard/scheduler/seven-wells-production-board.html on main.
# Keeping the review scope explicit prevents raw materials and packaging rows in
# the broader products table from silently expanding the proposed backfill.
ACTIVE_SKU_IDS = (
    "10001", "10002", "10006", "10007", "10020", "67470", "67473", "67476",
    "893", "10010", "10029", "10045", "10046", "10047", "10048", "10049",
    "10051", "10052", "10053", "10054", "10055", "10056", "10058", "10059",
    "70051", "31012", "31011", "10301", "10302", "10303", "10304", "10305",
    "10306", "70050", "70012", "10300", "1614", "70053", "70048", "70052",
    "70057", "70061", "70060", "70056", "70077", "70081", "70082", "70059",
    "70002", "70003", "70010", "70011", "70070", "70004", "70073", "70074",
    "70079", "70080",
)

APPROVED_UNCLASSIFIED_NULL_SKUS = {"70004"}

TEN_LB_RE = re.compile(r"\b10\s*LB\b", re.IGNORECASE)
TWENTY_FIVE_LB_RE = re.compile(r"\b25\s*LB\b", re.IGNORECASE)
OUNCE_RE = re.compile(r"\bOZ\b", re.IGNORECASE)
POUCH_COUNT_RE = re.compile(
    r"(?:\b\d+\s*[x×]\s*\d+(?:\.\d+)?\b|\b\d+(?:\.\d+)?\s*OZ\s*[x×]\s*\d+\b)",
    re.IGNORECASE,
)


def database_url() -> str:
    """Return DATABASE_URL from the environment or the repository .env file."""
    value = os.getenv("DATABASE_URL")
    if value:
        return value

    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("DATABASE_URL="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    raise RuntimeError("DATABASE_URL is not set and was not found in the repository .env")


def propose_pack_format(name: str) -> tuple[str | None, str]:
    """Return the proposed value and its review status for one product name."""
    if "GRANOLA" not in name.upper():
        return None, "not applicable"
    if TEN_LB_RE.search(name):
        return "10lb", "matched 10 LB/10LB"
    if TWENTY_FIVE_LB_RE.search(name):
        return "25lb", "matched 25 LB"
    if OUNCE_RE.search(name) or POUCH_COUNT_RE.search(name):
        return "bagged", "matched OZ/pouch count"
    return None, "UNCLASSIFIED"


def markdown_cell(value: object) -> str:
    """Escape a value for one Markdown table cell."""
    if value is None:
        return "NULL"
    return str(value).replace("|", "\\|").replace("\n", " ")


def load_rows(conn) -> list[dict]:
    """Load the full approved catalog scope, including inactive rows for checks."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT odoo_code AS sku_id, name, type, COALESCE(active, true) AS active
              FROM products
             WHERE odoo_code = ANY(%s)
             ORDER BY CASE odoo_code
                WHEN '893' THEN 893
                WHEN '1614' THEN 1614
                ELSE odoo_code::integer
             END,
             odoo_code
            """,
            (list(ACTIVE_SKU_IDS),),
        )
        return cur.fetchall()


def validate_scope(rows: list[dict]) -> tuple[dict[str, dict], list[str]]:
    """Validate that every approved catalog SKU exists, is active, and is unique."""
    rows_by_sku = {row["sku_id"]: row for row in rows}
    if len(rows_by_sku) != len(rows):
        raise RuntimeError("duplicate odoo_code rows found in the approved catalog scope")

    missing = [sku for sku in ACTIVE_SKU_IDS if sku not in rows_by_sku]
    inactive = [sku for sku in ACTIVE_SKU_IDS if sku in rows_by_sku and not rows_by_sku[sku]["active"]]
    if missing:
        raise RuntimeError(f"approved SKU(s) missing from products: {', '.join(missing)}")
    if inactive:
        raise RuntimeError(f"approved SKU(s) are no longer active: {', '.join(inactive)}")

    unclassified = [
        sku
        for sku, row in rows_by_sku.items()
        if propose_pack_format(row["name"])[1] == "UNCLASSIFIED"
    ]
    unexpected = sorted(set(unclassified) - APPROVED_UNCLASSIFIED_NULL_SKUS, key=int)
    approved_missing = sorted(APPROVED_UNCLASSIFIED_NULL_SKUS - set(unclassified), key=int)
    if unexpected or approved_missing:
        raise RuntimeError(
            "unclassified SKU set changed; "
            f"unexpected={unexpected or 'none'}, approved_missing={approved_missing or 'none'}"
        )

    return rows_by_sku, sorted(unclassified, key=int)


def assert_migration_applied(conn) -> None:
    """Require both the pack_format column and the migration's check constraint."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                       SELECT 1
                         FROM information_schema.columns
                        WHERE table_schema = 'public'
                          AND table_name = 'products'
                          AND column_name = 'pack_format'
                   ) AS column_exists,
                   EXISTS (
                       SELECT 1
                         FROM pg_constraint
                        WHERE conrelid = 'public.products'::regclass
                          AND conname = 'products_pack_format_check'
                   ) AS constraint_exists
            """
        )
        state = cur.fetchone()
    if not state["column_exists"] or not state["constraint_exists"]:
        raise RuntimeError(
            "migration 040 is incomplete: products.pack_format and "
            "products_pack_format_check are both required"
        )


def apply_mapping(conn, rows_by_sku: dict[str, dict]) -> None:
    """Write the exact approved mapping in one transaction and verify it."""
    expected: dict[str, str | None] = {}
    with conn.cursor() as cur:
        for sku in ACTIVE_SKU_IDS:
            proposed, _ = propose_pack_format(rows_by_sku[sku]["name"])
            expected[sku] = proposed
            cur.execute(
                """
                UPDATE products
                   SET pack_format = %s
                 WHERE odoo_code = %s
                   AND COALESCE(active, true) = true
                """,
                (proposed, sku),
            )
            if cur.rowcount != 1:
                raise RuntimeError(f"expected to update one active row for SKU {sku}; updated {cur.rowcount}")

        cur.execute(
            """
            SELECT odoo_code AS sku_id, pack_format
              FROM products
             WHERE odoo_code = ANY(%s)
            """,
            (list(ACTIVE_SKU_IDS),),
        )
        actual = {row["sku_id"]: row["pack_format"] for row in cur.fetchall()}

    mismatches = {
        sku: {"expected": value, "actual": actual.get(sku)}
        for sku, value in expected.items()
        if actual.get(sku) != value
    }
    if mismatches:
        raise RuntimeError(f"post-update mapping verification failed: {mismatches}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="transactionally apply the approved mapping (default is proposal-only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with psycopg2.connect(database_url(), cursor_factory=RealDictCursor) as conn:
        conn.set_session(readonly=not args.apply)
        rows = load_rows(conn)
        rows_by_sku, unclassified = validate_scope(rows)
        if args.apply:
            assert_migration_applied(conn)
            apply_mapping(conn, rows_by_sku)
            conn.commit()

    print("| SKU id | Name | Proposed pack_format | Review flag |")
    print("|---:|---|---|---|")

    for sku in sorted(ACTIVE_SKU_IDS, key=int):
        row = rows_by_sku[sku]
        proposed, status = propose_pack_format(row["name"])
        print(
            "| {sku} | {name} | {proposed} | {status} |".format(
                sku=markdown_cell(sku),
                name=markdown_cell(row["name"]),
                proposed=markdown_cell(proposed),
                status=markdown_cell(status),
            )
        )

    print()
    print(f"Catalog entries: {len(ACTIVE_SKU_IDS)}")
    print(f"Database rows found: {len(rows)}")
    print(f"Unclassified granola SKUs: {', '.join(unclassified) if unclassified else 'none'}")
    print("Missing SKUs: none")
    print("Inactive SKUs: none")
    if args.apply:
        non_null_count = sum(
            propose_pack_format(row["name"])[0] is not None
            for row in rows_by_sku.values()
        )
        print(
            f"Applied approved mapping: {len(rows_by_sku)} rows "
            f"({non_null_count} non-NULL, {len(rows_by_sku) - non_null_count} NULL)"
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"pack_format proposal failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
