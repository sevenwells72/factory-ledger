"""FL runs exactly one server process: the Assistant's instant chat unlock lives
in one process's memory (PR #141 review). Raising the worker count fails here."""

import ast
from pathlib import Path
import shlex
import subprocess
import sys
import textwrap
import time

import pytest
from uvicorn.main import main as uvicorn_cli

import server_workers

ROOT = Path(__file__).resolve().parents[1]


def _procfile_args():
    web = next(line.partition(":")[2] for line in (ROOT / "Procfile").read_text().splitlines()
               if line.startswith("web:"))
    command = shlex.split(web.replace("$PORT", "8000"))
    assert command[0] == "uvicorn", "production must launch through the uvicorn CLI"
    return command[1:]


def test_production_procfile_pins_one_worker_even_with_web_concurrency(monkeypatch):
    # Railway or a teammate setting WEB_CONCURRENCY must not add workers.
    monkeypatch.setenv("WEB_CONCURRENCY", "4")
    context = uvicorn_cli.make_context("uvicorn", _procfile_args())
    assert context.params["workers"] == 1
    assert server_workers.configured_workers(["uvicorn", *_procfile_args()], {"WEB_CONCURRENCY": "4"}) == 1


def test_staging_launcher_pins_one_worker():
    tree = ast.parse((ROOT / "scripts/staging_start_command.py").read_text())
    launch = next(ast.literal_eval(node.value) for node in tree.body
                  if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "LAUNCH" for t in node.targets))
    runs = [node for node in ast.walk(ast.parse(launch))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and getattr(node.func.value, "id", None) == "uvicorn" and node.func.attr == "run"]
    assert len(runs) == 1
    workers = next((k.value for k in runs[0].keywords if k.arg == "workers"), None)
    assert workers is not None and ast.literal_eval(workers) == 1, "uvicorn.run without workers=1 honours WEB_CONCURRENCY"


@pytest.mark.parametrize("argv,env,expected", [
    (["uvicorn", "main:app"], {}, 1),
    (["uvicorn", "main:app"], {"WEB_CONCURRENCY": "3"}, 3),
    (["uvicorn", "main:app", "--workers", "2"], {}, 2),
    (["uvicorn", "main:app", "--workers=2"], {}, 2),
    (["uvicorn", "main:app", "--workers", "1"], {"WEB_CONCURRENCY": "3"}, 1),
    (["uvicorn", "main:app", "--workers", "two"], {}, None),
])
def test_configured_workers(argv, env, expected):
    assert server_workers.configured_workers(argv, env) == expected


def test_problem_reported_for_extra_workers_or_child_process():
    one = ["uvicorn", "main:app", "--workers", "1"]
    assert server_workers.multiple_workers_problem(one, {}, parent=False) is None
    assert "2 server workers" in server_workers.multiple_workers_problem(
        ["uvicorn", "main:app", "--workers", "2"], {}, parent=False)
    assert "child process" in server_workers.multiple_workers_problem(["-c"], {}, parent=object())
    assert server_workers.multiple_workers_problem([*one, "--reload"], {}, parent=object()) is None


def test_startup_logs_single_worker_error():
    source = (ROOT / "main.py").read_text()
    startup = source[source.index("async def startup():"):source.index("_run_startup_migrations()\n    _assert")]
    assert "server_workers.multiple_workers_problem()" in startup
    assert "SINGLE_WORKER_REQUIRED" in startup


@pytest.mark.parametrize("workers,detected", [(1, False), (2, True)])
def test_real_uvicorn_workers_are_detected(tmp_path, workers, detected):
    # Each real server process reports what the guard sees at startup.
    (tmp_path / "probe_app.py").write_text(textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(ROOT)!r})
        import server_workers

        async def app(scope, receive, send):
            if scope["type"] == "lifespan":
                await receive()
                problem = server_workers.multiple_workers_problem()
                with open({str(tmp_path / "seen")!r}, "a") as out:
                    out.write(f"{{os.getpid()}} {{bool(problem)}}\\n")
                await send({{"type": "lifespan.startup.complete"}})
                await receive()
                await send({{"type": "lifespan.shutdown.complete"}})
    """))
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "probe_app:app", "--host", "127.0.0.1", "--port", "0",
         "--workers", str(workers), "--log-level", "error"],
        cwd=tmp_path, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    seen = tmp_path / "seen"
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if seen.exists() and len(seen.read_text().splitlines()) >= workers:
                break
            assert server.poll() is None, server.stderr.read().decode()
            time.sleep(0.1)
        lines = seen.read_text().splitlines()
    finally:
        server.terminate()
        server.wait(timeout=15)
    assert len(lines) == workers
    assert {line.split()[1] for line in lines} == {str(detected)}
