"""Run all suites with at most ONE populated disposable database, then remove it.

Run from the repo root with mcp_server/.venv/bin/python. No supplied database URL
or credentials are inherited. The existing backend migration/race tests allocate
scratch DBs themselves, so they run only AFTER the main test DB is dropped.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAILURES = []
sys.path.insert(0, str(ROOT / "mcp_server/tests"))
from ledger_harness import (  # noqa: E402
    LedgerDatabase,
    clean_environment,
    load_schema,
    postgres_cluster,
)


def disk_guard():
    subprocess.run(["df", "-h", "/"], check=True)
    if shutil.disk_usage("/").free < 5 * 1024**3:
        raise SystemExit("Under 5 GB free: refusing heavy steps")


def invoke(command, *, env=None):
    result = subprocess.run(command, cwd=ROOT, env=env or clean_environment())
    if result.returncode:
        # Finish independent suites even when current-main fixtures fail; never
        # turn a partial run into a green result or leave a database behind.
        FAILURES.append(result.returncode)


def backend(phase="both"):
    disk_guard()
    with postgres_cluster(tcp=True) as admin:
        base = f"postgresql://mcp_test@127.0.0.1:{admin.port}/"
        env = clean_environment() | {
            "TEST_DATABASE_URL": base + "fl_m0_test",
            "PYTHONPATH": str(Path(__file__).parent),
            "M0_DATABASE_PHASE": "regular",
        }
        command = [
            str(ROOT / ".venv-test/bin/python"),
            "-m",
            "pytest",
            "tests",
            "-p",
            "single_db_partition",
            "--tb=short",
        ]
        if phase != "scratch":
            admin.sql("CREATE DATABASE fl_m0_test;")
            try:
                load_schema(
                    LedgerDatabase(admin.pg_bin, admin.socket_dir, "fl_m0_test", admin.port)
                )
                invoke(command, env=env)
            finally:
                admin.sql("DROP DATABASE fl_m0_test WITH (FORCE);")
        if phase != "regular":
            disk_guard()
            env.update(TEST_DATABASE_URL=base + "postgres", M0_DATABASE_PHASE="scratch")
            invoke(command, env=env)
        remaining = admin.sql(
            "SELECT datname FROM pg_database WHERE NOT datistemplate AND datname != 'postgres'"
        )
        if remaining:
            raise SystemExit("Scratch fixture left a database behind")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite", choices=["all", "backend", "mcp", "node", "audits"], default="all"
    )
    parser.add_argument("--database-phase", choices=["both", "regular", "scratch"], default="both")
    args = parser.parse_args()
    if args.suite in {"all", "backend"}:
        backend(args.database_phase)
    if args.suite in {"all", "mcp"}:
        disk_guard()
        invoke(
            [
                str(ROOT / "mcp_server/.venv/bin/python"),
                "-m",
                "pytest",
                "mcp_server/tests",
                "--tb=short",
                "-q",
            ]
        )
    if args.suite == "audits":
        invoke(
            [
                str(ROOT / ".venv-test/bin/python"),
                "-m",
                "pytest",
                "audits/fresh-start",
                "--tb=short",
            ]
        )
    if args.suite in {"all", "node"}:
        node = shutil.which("node")
        if not node:
            raise SystemExit("Node is required; tests cannot be skipped")
        invoke([node, "--test", *map(str, sorted((ROOT / "tests").glob("test_*.js")))])
    if FAILURES:
        raise SystemExit(f"{len(FAILURES)} test phase(s) failed; see the results above")


if __name__ == "__main__":
    main()
