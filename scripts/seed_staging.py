#!/usr/bin/env python3
"""Seed staging with synthetic fixtures; optionally SELECT approved master data.

Based on mcp_server/tests/ledger_harness.py seed_database/seed_write_database at
feature/mcp-server d74f747d2766393c75d07e2ca295f6ada49b6c74 (read via git show).
That branch and its files are not changed. No test keys are reused.
"""
import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import secrets
import stat
import sys

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from staging_safety import assert_staging_database, database_host, PRODUCTION_DATABASE_HOST

SECRETS_DIR = Path.home() / "Documents/fl-secrets"
FIXTURE_BASE = 1_000_000_000
FIXTURE_MARKER = "staging_synthetic_fixture_v1"
MASTER_MARKER = "staging_master_data_copy_v1"
# This schema stores units in products.uom and supplier_product_aliases.unit /
# lb_per_unit; it has no units or product_aliases table. Never expand dynamically.
MASTER_TABLES = (
    "customers", "suppliers", "products",
    "customer_product_aliases", "supplier_product_aliases",
)
SYNC_TABLES = MASTER_TABLES + (
    "actors", "lots", "transactions", "transaction_lines",
    "sales_orders", "sales_order_lines",
)


def secret_file(path):
    path = Path(path).expanduser()
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise RuntimeError("Secret file must have permissions 600")
    value = path.read_text().strip()
    if not value or "\n" in value or "\r" in value:
        raise RuntimeError("Secret file must contain one nonempty URI")
    return value


def marked(cur, name):
    cur.execute("SELECT 1 FROM public.migration_markers WHERE name=%s", (name,))
    return cur.fetchone() is not None


def mark(cur, name):
    cur.execute("INSERT INTO public.migration_markers(name) VALUES (%s)", (name,))


def copy_master_data(target_cur, production_url, staging_url):
    """Only this optional function opens a source connection; never writes to it."""
    if database_host(production_url) == database_host(staging_url):
        raise RuntimeError("Master-data source and staging hosts must differ")
    if marked(target_cur, MASTER_MARKER):
        return {"already_copied": True}
    counts = {}
    with closing(psycopg2.connect(production_url, connect_timeout=10,
                                 sslmode="require")) as source:
        source.autocommit = True
        with source.cursor() as cur:
            # Transaction scoped: NEVER set_session/default_transaction_read_only.
            cur.execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            try:
                for table in MASTER_TABLES:
                    target_cur.execute(
                        """SELECT column_name FROM information_schema.columns
                           WHERE table_schema='public' AND table_name=%s
                             AND is_generated='NEVER' ORDER BY ordinal_position""",
                        (table,),
                    )
                    columns = [row[0] for row in target_cur.fetchall()]
                    if not columns:
                        raise RuntimeError("Required master-data table missing in staging")
                    names = sql.SQL(",").join(map(sql.Identifier, columns))
                    # Source statements can only SELECT from the fixed allowlist.
                    cur.execute(sql.SQL("SELECT {} FROM public.{} ORDER BY id").format(
                        names, sql.Identifier(table)))
                    rows = cur.fetchall()
                    id_index = columns.index("id")
                    if any(row[id_index] >= FIXTURE_BASE for row in rows):
                        raise RuntimeError("Source IDs overlap the reserved synthetic fixture range")
                    if rows:
                        execute_values(target_cur,
                            sql.SQL("INSERT INTO public.{} ({}) VALUES %s").format(
                                sql.Identifier(table), names), rows)
                    counts[table] = len(rows)
            finally:
                cur.execute("ROLLBACK")
    mark(target_cur, MASTER_MARKER)
    return counts


def seed_fixtures(cur):
    if marked(cur, FIXTURE_MARKER):
        return False
    # Reserved IDs avoid production startup mappings and catalog ID collisions.
    # Inactive actor preserves the harness fixture without deploying a known key.
    actor_hash = hashlib.sha256(secrets.token_bytes(32)).hexdigest()
    cur.execute("""INSERT INTO actors(id,name,role,key_hash,active)
                   VALUES (%s,'STAGING Synthetic Actor','floor',%s,false)""",
                (FIXTURE_BASE + 1, actor_hash))
    cur.execute("""INSERT INTO customers(id,name) VALUES (%s,'STAGING Test Customer');
                   INSERT INTO suppliers(id,name,active) VALUES (%s,'STAGING Test Supplier',true)""",
                (FIXTURE_BASE + 1, FIXTURE_BASE + 1))
    cur.execute("""INSERT INTO products
        (id,name,type,active,case_size_lb,default_batch_lb,uom,is_service)
        VALUES (%s,'STAGING Test Almonds','ingredient',true,NULL,NULL,'lb',false),
               (%s,'STAGING Test Batch','batch',true,10,100,'lb',false),
               (%s,'STAGING Pallet Charge','finished',true,NULL,NULL,'each',true)""",
        (FIXTURE_BASE + 1, FIXTURE_BASE + 2, FIXTURE_BASE + 3))
    cur.execute("""INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb)
                   VALUES (%s,%s,100)""", (FIXTURE_BASE + 2,FIXTURE_BASE + 1))
    for offset, kind in ((1,"ingredient"),(2,"batch")):
        cur.execute("""INSERT INTO lots(id,product_id,lot_code,supplier_lot_code,lot_type)
                       VALUES (%s,%s,'STAGING-SHARED-LOT','STAGING-SUPPLIER-LOT',%s)""",
                    (FIXTURE_BASE+offset,FIXTURE_BASE+offset,kind))
    cur.execute("INSERT INTO transactions(id,type,notes) VALUES (%s,'receive','Synthetic staging fixture')",
                (FIXTURE_BASE+1,))
    for offset in (1,2):
        cur.execute("""INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb)
                       VALUES (%s,%s,%s,250)""",
                    (FIXTURE_BASE+1,FIXTURE_BASE+offset,FIXTURE_BASE+offset))
    cur.execute("""INSERT INTO sales_orders(id,order_number,customer_id,status,customer_po)
                   VALUES (%s,'SO-STAGING-TEST',%s,'confirmed','0007-A')""",
                (FIXTURE_BASE+1,FIXTURE_BASE+1))
    for offset, qty, price in ((1,100,None),(2,100,None),(3,2,15)):
        cur.execute("""INSERT INTO sales_order_lines
                       (id,sales_order_id,product_id,quantity_lb,unit_price)
                       VALUES (%s,%s,%s,%s,%s)""",
                    (FIXTURE_BASE+offset,FIXTURE_BASE+1,FIXTURE_BASE+offset,qty,price))
    # Realistic alias/unit fixtures without copying any production rows.
    cur.execute("""INSERT INTO supplier_product_aliases
                   (supplier_id,vendor_description,product_id,lb_per_unit,unit)
                   VALUES (%s,'STAGING ALMONDS 25 LB',%s,25,'case')""",
                (FIXTURE_BASE+1,FIXTURE_BASE+1))
    cur.execute("""INSERT INTO customer_product_aliases
                   (customer_id,customer_item_code,product_id,case_size_lb)
                   VALUES (%s,'STAGING-BATCH-10',%s,10)""",
                (FIXTURE_BASE+1,FIXTURE_BASE+2))
    mark(cur, FIXTURE_MARKER)
    return True


def sync_sequences(cur):
    for table in SYNC_TABLES:
        cur.execute("SELECT pg_get_serial_sequence(%s,'id')", ("public."+table,))
        sequence = cur.fetchone()[0]
        if sequence:
            cur.execute(sql.SQL(
                "SELECT setval(%s, GREATEST(COALESCE(MAX(id),1), "
                "(SELECT last_value FROM {})), true) FROM public.{}"
            ).format(sql.Identifier(*sequence.split(".")),sql.Identifier(table)), (sequence,))


def run(args):
    staging_url = secret_file(args.db_url_file)
    assert_staging_database(staging_url, "staging", args.production_host)
    source_url = None
    if args.copy_master_data:
        if not args.production_db_url_file:
            raise RuntimeError("--copy-master-data requires --production-db-url-file")
        source_url = secret_file(args.production_db_url_file)
        database_host(source_url)
    with closing(psycopg2.connect(staging_url, connect_timeout=10,
                                 sslmode="require")) as target:
        with target:
            with target.cursor() as cur:
                # Serialize reruns before checking the durable fixture markers.
                cur.execute("SELECT pg_advisory_xact_lock(768732490519)")
                copied = copy_master_data(cur, source_url, staging_url) if source_url else {}
                inserted = seed_fixtures(cur)
                sync_sequences(cur)
    return {"synthetic_fixtures": "inserted" if inserted else "already_present",
            "master_data_copied": copied}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-url-file", type=Path, default=SECRETS_DIR/"staging-db-url.txt")
    parser.add_argument("--production-host", default=PRODUCTION_DATABASE_HOST)
    parser.add_argument("--copy-master-data", action="store_true")
    parser.add_argument("--production-db-url-file", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), sort_keys=True))
    except (psycopg2.Error, OSError):
        # Connection errors and SQL details can contain secrets/catalog rows.
        print("Staging seed failed; sensitive database diagnostics suppressed.", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
