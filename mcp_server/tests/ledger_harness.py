"""TEST ONLY: disposable Postgres + unmodified ledger in its own dependency process.

No supplied DB URL is accepted. Every database lives in the private cluster created
here; the child receives only a minimal environment and synthetic authentication.
The MCP process keeps its modern dependencies separate from legacy FastAPI.
"""

import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[2]
MASTER_KEY = "mcp-test-master-key"
ACTOR_KEY = "mcp-test-actor-key"
DASHBOARD_KEY = "mcp-test-dashboard-key"


def clean_environment():
    # In particular: no inherited DATABASE_URL, API keys, proxies or storage/AI secrets.
    return {"PATH": os.defpath, "LANG": "C", "PYTHONUNBUFFERED": "1"}


def run(command, **kwargs):
    result = subprocess.run(
        [str(arg) for arg in command],
        capture_output=True,
        text=True,
        env=clean_environment(),
        **kwargs,
    )
    if result.returncode:
        raise RuntimeError(f"Test harness command failed: {command[0]}\n{result.stderr}")
    return result.stdout.strip()


@dataclass
class LedgerDatabase:
    pg_bin: Path
    socket_dir: Path
    name: str
    port: int = 5432  # Private Unix socket; PostgreSQL never opens a TCP listener.

    @property
    def dsn(self):
        return f"dbname={self.name} host={self.socket_dir} port={self.port} user=mcp_test"

    def sql(self, sql):
        return run(
            [self.pg_bin / "psql", "-XAt", "-v", "ON_ERROR_STOP=1", "--dbname", self.dsn],
            input=sql,
        )

    def snapshot(self):
        # Include every table, row, column and sequence, not a selected business subset.
        # Capture in one transaction so timestamps and concurrent reads stay consistent.
        tables = self.sql(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
        ).splitlines()
        statements = ["BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;"]
        for table in tables:
            statements.append(
                f"SELECT json_build_object('table', '{table}', 'rows', "
                f"COALESCE(jsonb_agg(to_jsonb(t) ORDER BY to_jsonb(t)::text), '[]'::jsonb)) "
                f'FROM public."{table}" t;'
            )
        sequences = self.sql(
            "SELECT sequencename FROM pg_sequences WHERE schemaname='public' ORDER BY sequencename"
        ).splitlines()
        for sequence in sequences:
            statements.append(
                f"SELECT json_build_object('table', 'sequence:{sequence}', "
                "'rows', jsonb_build_array(jsonb_build_object("
                "'last_value', last_value, 'is_called', is_called))) "
                f'FROM public."{sequence}";'
            )
        statements.append("COMMIT;")
        return {
            row["table"]: row["rows"]
            for line in self.sql("\n".join(statements)).splitlines()
            if line.startswith("{")
            for row in [json.loads(line)]
        }


@contextmanager
def postgres_cluster():
    configured = os.getenv("MCP_TEST_PG_BIN")
    found = shutil.which("initdb")
    pg_bin = Path(
        configured or (str(Path(found).parent) if found else "/opt/homebrew/opt/postgresql@17/bin")
    )
    if not (pg_bin / "initdb").is_file():
        raise RuntimeError("Install PostgreSQL 17; set MCP_TEST_PG_BIN to its bin directory")
    # Short path avoids macOS's Unix-socket path limit; random per session/worktree.
    with tempfile.TemporaryDirectory(prefix="mcp-pg-", dir="/tmp") as directory:
        root = Path(directory)
        data = root / "data"
        run(
            [
                pg_bin / "initdb",
                "-D",
                data,
                "-U",
                "mcp_test",
                "-A",
                "trust",
                "--no-locale",
                "--encoding=UTF8",
            ]
        )
        run(
            [
                pg_bin / "pg_ctl",
                "-D",
                data,
                "-l",
                root / "postgres.log",
                "-w",
                "start",
                "-o",
                f"-k {root} -h '' -F",
            ]
        )
        try:
            template = LedgerDatabase(pg_bin, root, "mcp_template")
            admin = LedgerDatabase(pg_bin, root, "postgres")
            admin.sql("CREATE DATABASE mcp_template;")
            template.sql("CREATE EXTENSION pg_trgm;")
            run(
                [
                    pg_bin / "psql",
                    "-Xq",
                    "-v",
                    "ON_ERROR_STOP=1",
                    "--dbname",
                    template.dsn,
                    "-f",
                    ROOT / "tests/schema/schema.sql",
                ]
            )
            yield admin
        finally:
            run([pg_bin / "pg_ctl", "-D", data, "-m", "immediate", "-w", "stop"])


def seed_database(db):
    # Static, synthetic business scenario independent of the MCP-generated catalog.
    key_hash = hashlib.sha256(ACTOR_KEY.encode()).hexdigest()
    db.sql(f"""
        INSERT INTO actors (id, name, role, key_hash)
            VALUES (1, 'Synthetic MCP actor', 'floor', '{key_hash}');
        INSERT INTO customers (id, name) VALUES (1, 'MCP Test Customer');
        INSERT INTO products (id, name, type, active, case_size_lb, default_batch_lb)
            VALUES (1, 'MCP Test Almonds', 'ingredient', true, NULL, NULL),
                   (2, 'MCP Test Batch', 'batch', true, 10, 100);
        INSERT INTO batch_formulas (product_id, ingredient_product_id, quantity_lb)
            VALUES (2, 1, 100);
        INSERT INTO lots (id, product_id, lot_code, supplier_lot_code, lot_type)
            VALUES (1, 1, 'MCP-SHARED-LOT', 'MCP-SUPPLIER-LOT', 'ingredient'),
                   (2, 2, 'MCP-SHARED-LOT', 'MCP-SUPPLIER-LOT', 'batch');
        INSERT INTO transactions (id, type) VALUES (1, 'receive');
        INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
            VALUES (1, 1, 1, 250), (1, 2, 2, 250);
        INSERT INTO sales_orders (id, order_number, customer_id, status)
            VALUES (1, 'SO-MCP-TEST', 1, 'confirmed');
        INSERT INTO sales_order_lines (id, sales_order_id, product_id, quantity_lb)
            VALUES (1, 1, 1, 100), (2, 1, 2, 100);
    """)
    # Make subsequent write tests safe to reuse these seeded IDs through normal sequences.
    db.sql("""
        DO $$ DECLARE r record; BEGIN
          FOR r IN SELECT table_name, column_name FROM information_schema.columns
                   WHERE table_schema='public' AND column_default LIKE 'nextval(%'
          LOOP
            EXECUTE format('SELECT setval(%L, COALESCE(MAX(%I), 1), MAX(%I) IS NOT NULL) FROM %I',
              pg_get_serial_sequence('public.' || quote_ident(r.table_name), r.column_name),
              r.column_name, r.column_name, r.table_name);
          END LOOP;
        END $$;
    """)


@contextmanager
def ledger_process(db, log_path):
    python = Path(
        os.getenv("MCP_TEST_LEDGER_PYTHON", ROOT / "mcp_server/.venv-ledger-test/bin/python")
    )
    if not python.is_file():
        raise RuntimeError("Create .venv-ledger-test with ../requirements.txt (see README)")
    env = clean_environment() | {
        "DATABASE_URL": db.dsn,
        "API_KEY": MASTER_KEY,
        "DASHBOARD_API_KEY": DASHBOARD_KEY,
    }
    # Pass the bound socket to uvicorn: no race from selecting then releasing a port.
    with socket.socket() as listener, log_path.open("w+") as log:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        process = subprocess.Popen(
            [str(python), "-m", "uvicorn", "main:app", "--fd", str(listener.fileno())],
            cwd=ROOT,
            env=env,
            pass_fds=(listener.fileno(),),
            stdout=log,
            stderr=log,
        )
        try:
            ready = False
            with httpx.Client(base_url=url, trust_env=False, timeout=0.5) as client:
                deadline = time.monotonic() + 25
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    try:
                        if client.get("/health").status_code == 200:
                            ready = True
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.05)
            if not ready:
                log.seek(0)
                raise RuntimeError(f"Test ledger did not start:\n{log.read()}")
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
