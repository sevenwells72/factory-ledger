"""The app's DB pool queues requests for a free connection instead of failing.

Production 2026-10-18/19: opening the dashboard fired ~20 reads at once at a
10-connection psycopg2 pool, and every request past the tenth failed instantly
with "connection pool exhausted". These tests run the real app in a local
Uvicorn process (startup builds the pool exactly as on Railway) with each SQL
statement slowed down, so the burst overlaps the way it does over the network.
"""
import asyncio
import os
from pathlib import Path
import socket
import subprocess
import sys
import textwrap
import threading
import time

import httpx
import psycopg2
import pytest

import main


ROOT = Path(__file__).resolve().parents[1]
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")

APPLICATION_NAME = "fl_pool_loadtest"
STATEMENT_DELAY_SECONDS = 0.15

# What dashboard.js refreshAll() requests on first load (plus a not-found
# order, so a request that errors after using the DB is in the burst too).
DASHBOARD_PATHS = [
    "/production/today-tile",
    "/dashboard/api/production?days=14",
    "/dashboard/api/inventory/finished-goods",
    "/dashboard/api/inventory/batches",
    "/dashboard/api/inventory/ingredients",
    "/dashboard/api/activity/shipments?limit=100",
    "/dashboard/api/activity/receipts?limit=100",
    "/sales/orders/counts",
    "/sales/orders?limit=50",
    "/expected-receipts?status=open&limit=500",
    "/supplies/inventory",
    "/supply-requests?status=all&limit=500",
    "/sales/orders/fulfillment-check",
]
MISSING_ORDER_PATH = "/sales/orders/2147483000"

LAUNCHER = textwrap.dedent('''
    import sys, time
    import psycopg2, psycopg2.extensions, psycopg2.extras
    import uvicorn

    port, delay, application_name = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
    slow = [False]
    _cursor_classes = {}

    def _slow_cursor_class(base):
        if base not in _cursor_classes:
            def execute(self, *args, **kwargs):
                if slow[0]:
                    time.sleep(delay)  # network round trip to the hosted DB
                return base.execute(self, *args, **kwargs)
            _cursor_classes[base] = type("Slow" + base.__name__, (base,), {"execute": execute})
        return _cursor_classes[base]

    class SlowConnection(psycopg2.extensions.connection):
        def cursor(self, *args, **kwargs):
            base = kwargs.pop("cursor_factory", None) or self.cursor_factory or psycopg2.extensions.cursor
            return super().cursor(*args, cursor_factory=_slow_cursor_class(base), **kwargs)

    _connect = psycopg2.connect
    def connect(*args, **kwargs):
        kwargs.setdefault("connection_factory", SlowConnection)
        kwargs.setdefault("application_name", application_name)
        return _connect(*args, **kwargs)
    psycopg2.connect = connect

    import main

    @main.app.on_event("startup")
    async def _slow_after_startup():
        # Registered after main's own startup: migrations run at full speed.
        slow[0] = True

    @main.app.get("/__pool")
    def _pool_state():
        return {"used": len(main.db_pool._used), "maxconn": main.db_pool.maxconn}

    uvicorn.run(main.app, host="127.0.0.1", port=port, log_level="warning")
''')


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def local_server(tmp_path):
    launcher = tmp_path / "launch_pool_app.py"
    launcher.write_text(LAUNCHER)
    port = _free_port()
    env = dict(os.environ, DATABASE_URL=TEST_DATABASE_URL, PYTHONPATH=str(ROOT))
    env.pop("ENVIRONMENT", None)
    log = (tmp_path / "server.log").open("w")
    process = subprocess.Popen(
        [sys.executable, str(launcher), str(port), str(STATEMENT_DELAY_SECONDS), APPLICATION_NAME],
        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            assert process.poll() is None, (tmp_path / "server.log").read_text()
            try:
                if httpx.get(base + "/__pool", timeout=2).status_code == 200:
                    break
            except httpx.TransportError:
                pass
            assert time.monotonic() < deadline, "local server did not start"
            time.sleep(0.2)
        yield base, tmp_path / "server.log"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


async def _burst(base, paths):
    headers = {"X-API-Key": os.environ["DASHBOARD_API_KEY"]}
    limits = httpx.Limits(max_connections=len(paths), max_keepalive_connections=0)
    async with httpx.AsyncClient(base_url=base, headers=headers, limits=limits, timeout=60) as client:
        return await asyncio.gather(*(client.get(path) for path in paths))


def _server_sessions():
    with psycopg2.connect(TEST_DATABASE_URL) as conn, conn.cursor() as cur:
        cur.execute("SELECT state, count(*) FROM pg_stat_activity "
                    "WHERE application_name = %s GROUP BY state", (APPLICATION_NAME,))
        return dict(cur.fetchall())


def test_25_concurrent_dashboard_requests_wait_for_the_10_connection_pool(local_server):
    base, server_log = local_server
    assert httpx.get(base + "/__pool").json()["maxconn"] == 10
    paths = [DASHBOARD_PATHS[i % len(DASHBOARD_PATHS)] for i in range(24)] + [MISSING_ORDER_PATH]

    # Each request succeeds on its own, so any failure below is the burst.
    headers = {"X-API-Key": os.environ["DASHBOARD_API_KEY"]}
    alone = {path: httpx.get(base + path, headers=headers, timeout=30).status_code
             for path in sorted(set(paths))}
    assert alone == {path: 404 if path == MISSING_ORDER_PATH else 200 for path in alone}

    started = time.monotonic()
    responses = asyncio.run(_burst(base, paths))
    elapsed = time.monotonic() - started

    failed = [(str(r.url), r.status_code, r.text[:200]) for r in responses
              if r.status_code != (404 if r.url.path == MISSING_ORDER_PATH else 200)]
    assert failed == [], failed
    assert "pool exhausted" not in server_log.read_text()
    assert elapsed < main.DB_POOL_WAIT_SECONDS

    # No leaked connections: the pool has none checked out, and no server
    # session is left mid-transaction.
    assert httpx.get(base + "/__pool").json()["used"] == 0
    sessions = _server_sessions()
    assert set(sessions) <= {"idle"}, sessions
    assert sum(sessions.values()) <= 10


def _waiting_pool(maxconn, wait_seconds):
    return main.WaitingConnectionPool(0, maxconn, dsn=TEST_DATABASE_URL, wait_seconds=wait_seconds)


def _sleep_on(db_pool, seconds, started=None):
    conn = db_pool.getconn()
    try:
        if started:
            started.set()
        with conn.cursor() as cur:
            cur.execute("SELECT pg_sleep(%s)", (seconds,))
        conn.rollback()
    finally:
        db_pool.putconn(conn)


def test_a_slow_query_does_not_starve_other_requests():
    db_pool = _waiting_pool(maxconn=2, wait_seconds=5)
    try:
        started = threading.Event()
        slow = threading.Thread(target=_sleep_on, args=(db_pool, 4, started))
        slow.start()
        assert started.wait(5)
        finished = []

        def quick(n):
            conn = db_pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT %s", (n,))
                    finished.append((cur.fetchone()[0], time.monotonic()))
                conn.rollback()
            finally:
                db_pool.putconn(conn)

        begin = time.monotonic()
        quick_threads = [threading.Thread(target=quick, args=(n,)) for n in range(20)]
        for thread in quick_threads:
            thread.start()
        for thread in quick_threads:
            thread.join(10)
        # All 20 shared the one free connection while the slow query ran on.
        assert sorted(n for n, _ in finished) == list(range(20))
        assert max(at for _, at in finished) - begin < 3
        assert slow.is_alive()
        slow.join(10)
        assert db_pool._used == {}
    finally:
        db_pool.closeall()


def test_waiting_is_bounded_when_every_connection_stays_busy():
    db_pool = _waiting_pool(maxconn=2, wait_seconds=1)
    try:
        sleepers = [threading.Thread(target=_sleep_on, args=(db_pool, 3)) for _ in range(2)]
        for thread in sleepers:
            thread.start()
        deadline = time.monotonic() + 5
        while len(db_pool._used) < 2:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        begin = time.monotonic()
        with pytest.raises(main.pool.PoolError, match="connection pool exhausted"):
            db_pool.getconn()
        assert 0.9 <= time.monotonic() - begin < 2

        # A longer wait is served as soon as a slow query returns its connection.
        db_pool.wait_seconds = 5
        conn = db_pool.getconn()
        assert time.monotonic() - begin < 4
        db_pool.putconn(conn)
        for thread in sleepers:
            thread.join(10)
        assert db_pool._used == {}
    finally:
        db_pool.closeall()


# Every dashboard panel that 500ed on prod 2026-10-18/19; the last three catch
# their own exceptions and used to turn the pool error into a 500.
DASHBOARD_PANEL_PATHS = [
    "/supplies/inventory",
    "/supply-requests?status=all&limit=500",
    "/expected-receipts?status=open&limit=500",
    "/production/today-tile",
    "/sales/orders/fulfillment-check",
    "/sales/orders/counts",
]


@pytest.mark.parametrize("path", DASHBOARD_PANEL_PATHS)
def test_pool_exhaustion_is_a_retryable_503_not_a_500(monkeypatch, caplog, path):
    db_pool = _waiting_pool(maxconn=1, wait_seconds=0.2)
    held = db_pool.getconn()
    monkeypatch.setattr(main, "db_pool", db_pool)
    try:
        async def call():
            transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                return await client.get(path, headers={"X-API-Key": os.environ["DASHBOARD_API_KEY"]})
        response = asyncio.run(call())
        assert response.status_code == 503, response.text
        assert response.json()["error_code"] == "DATABASE_BUSY"
        assert response.json()["retryable"] is True
        assert "DB_POOL_CAPACITY: all 1 app connections stayed busy for 0.2 s" in caplog.text
    finally:
        db_pool.putconn(held)
        db_pool.closeall()


def test_errors_and_threadpool_work_return_their_connections(monkeypatch):
    import assistant_runtime
    db_pool = _waiting_pool(maxconn=2, wait_seconds=1)
    monkeypatch.setattr(main, "db_pool", db_pool)
    try:
        with pytest.raises(psycopg2.ProgrammingError):
            with main.get_transaction() as cur:
                cur.execute("SELECT no_such_column FROM products")
        with pytest.raises(RuntimeError):
            with main.get_transaction() as cur:
                cur.execute("SELECT 1")
                raise RuntimeError("handler failed mid-transaction")
        assert db_pool._used == {}

        class Request:
            headers = {}
            scope = {}

        async def handler():
            with main.get_transaction() as cur:
                cur.execute("SELECT 1")
                raise RuntimeError("assistant turn failed")

        for _ in range(5):  # more than the pool holds: a leak would exhaust it
            with pytest.raises(RuntimeError, match="assistant turn failed"):
                asyncio.run(assistant_runtime.run_turn(handler, Request()))
        assert db_pool._used == {}
    finally:
        db_pool.closeall()


# ── Review round 1 (PR #138 comment 6271859908) ─────────────────────────────

def _route_calls():
    """One call per (method, route) the app serves, path params filled in."""
    from fastapi.routing import APIRoute
    calls = []
    for route in main.app.routes:
        if not isinstance(route, APIRoute):
            continue
        path = route.path_format
        for param in route.dependant.path_params:
            path = path.replace("{" + param.name + "}", "1")
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            calls.append((method, path))
    return calls


def test_no_route_answers_500_while_every_connection_stays_busy(monkeypatch):
    """Nine dashboard reads (and other routes) caught the pool timeout in their
    own `except Exception` and answered 500. Every route now answers 503
    DATABASE_BUSY instead, whatever its own error handling does."""
    monkeypatch.setenv("ASSISTANT_ENABLED", "1")
    db_pool = _waiting_pool(maxconn=1, wait_seconds=0.05)
    held = db_pool.getconn()
    monkeypatch.setattr(main, "db_pool", db_pool)
    calls = _route_calls()
    assert len(calls) > 200
    try:
        async def run():
            transport = httpx.ASGITransport(app=main.app, raise_app_exceptions=False)
            async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30) as client:
                results = []
                for method, path in calls:
                    response = await client.request(method, path, json={},
                                                    headers={"X-API-Key": os.environ["API_KEY"]})
                    results.append((method, path, response))
                return results
        results = asyncio.run(run())
    finally:
        db_pool.putconn(held)
        db_pool.closeall()
    failed = [(m, p, r.status_code, r.text[:120]) for m, p, r in results
              if r.status_code >= 500 and (r.status_code != 503 or "DATABASE_BUSY" not in r.text)]
    assert failed == [], "\n".join(map(str, failed))
    # Every panel the dashboard opens with reached the DB and was told to retry.
    by_path = {(m, p): r for m, p, r in results}
    for path in DASHBOARD_PATHS:
        response = by_path[("GET", path.split("?")[0])]
        assert response.status_code == 503, (path, response.text[:200])
        assert response.json()["error_code"] == "DATABASE_BUSY"
        assert response.json()["retryable"] is True


class _FakeConnection:
    """Enough of a psycopg2 connection for the pool's own bookkeeping."""

    def __init__(self, rollback_seconds=0):
        self.closed = 0
        self.rollback_seconds = rollback_seconds
        self.status = psycopg2.extensions.TRANSACTION_STATUS_IDLE

    @property
    def info(self):
        return type("Info", (), {"transaction_status": self.status})()

    def rollback(self):
        time.sleep(self.rollback_seconds)
        self.status = psycopg2.extensions.TRANSACTION_STATUS_IDLE

    def close(self):
        self.closed = 1


@pytest.fixture
def fake_connect(monkeypatch):
    """psycopg2.connect replaced by a fake whose open takes `delay[0]` seconds."""
    delay = [0.0]
    opened = []

    def connect(*args, **kwargs):
        time.sleep(delay[0])
        conn = _FakeConnection()
        opened.append(conn)
        return conn
    monkeypatch.setattr(psycopg2, "connect", connect)
    return delay, opened


def _wait_until(condition, seconds=5):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "condition never became true"
        time.sleep(0.005)


def test_a_returned_connection_goes_to_the_oldest_waiter_not_a_newcomer(fake_connect):
    db_pool = main.WaitingConnectionPool(0, 1, wait_seconds=5)
    order = []

    def take(name):
        conn = db_pool.getconn()
        order.append(name)
        time.sleep(0.05)
        db_pool.putconn(conn)

    held = db_pool.getconn()
    oldest = threading.Thread(target=take, args=("oldest",))
    oldest.start()
    time.sleep(0.2)  # oldest is now queued for the one connection
    # The holder returns its connection and asks again straight away, the way
    # a handler running several short transactions does.
    db_pool.putconn(held)
    take("newcomer")
    oldest.join(5)
    assert order == ["oldest", "newcomer"]
    assert db_pool._used == {}


def test_waiters_are_served_in_arrival_order_under_contention(fake_connect):
    # One connection, so each thread records its turn while it holds the only
    # slot: the recorded order is exactly the order the pool served them in.
    waiters = 30
    db_pool = main.WaitingConnectionPool(0, 1, wait_seconds=10)
    order = []
    held = db_pool.getconn()

    def take(n):
        conn = db_pool.getconn()
        order.append(n)
        time.sleep(0.005)
        db_pool.putconn(conn)

    threads = []
    for n in range(waiters):
        thread = threading.Thread(target=take, args=(n,))
        thread.start()
        threads.append(thread)
        time.sleep(0.02)  # arrive one after another

    # The holder keeps returning its connection and asking again at once; it
    # may never jump ahead of a thread that queued before it.
    conn = held
    for _ in range(5):
        db_pool.putconn(conn)
        conn = db_pool.getconn()
        order.append("holder")
    db_pool.putconn(conn)
    for thread in threads:
        thread.join(15)
    assert order == list(range(waiters)) + ["holder"] * 5
    assert db_pool._used == {}


def test_slow_connection_opens_never_stretch_the_wait_past_its_limit(fake_connect):
    """Review: three 6 s opens under the mutex made the third getconn take 18 s
    against a 10 s limit. Opens now run outside the mutex, within the deadline."""
    delay, _ = fake_connect
    delay[0] = 0.6
    db_pool = main.WaitingConnectionPool(0, 3, wait_seconds=1.0)
    durations, errors = [], []

    def get():
        begin = time.monotonic()
        try:
            conn = db_pool.getconn()
        except main.pool.PoolError as exc:
            errors.append(str(exc))
        finally:
            durations.append(time.monotonic() - begin)

    threads = [threading.Thread(target=get) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert errors == []
    assert max(durations) < 1.0, durations  # opened side by side, not one after another


def test_an_open_slower_than_the_limit_times_out_on_time_and_frees_its_slot(fake_connect):
    delay, opened = fake_connect
    delay[0] = 1.5
    db_pool = main.WaitingConnectionPool(0, 1, wait_seconds=0.5)
    begin = time.monotonic()
    with pytest.raises(main.pool.PoolError, match="connection pool exhausted"):
        db_pool.getconn()
    assert time.monotonic() - begin < 0.65
    # The abandoned open finishes in the background; its connection is closed
    # or kept idle, never leaked, and the slot serves the next caller.
    delay[0] = 0
    _wait_until(lambda: len(opened) == 1)
    conn = db_pool.getconn()
    db_pool.putconn(conn)
    assert db_pool._used == {}


def test_returning_a_connection_is_not_blocked_by_network_work(fake_connect):
    delay, _ = fake_connect
    db_pool = main.WaitingConnectionPool(0, 3, wait_seconds=5)
    returned = db_pool.getconn()
    delay[0] = 1.0
    opener = threading.Thread(target=lambda: db_pool.putconn(db_pool.getconn()))
    opener.start()
    time.sleep(0.1)  # a slow open is in progress
    begin = time.monotonic()
    db_pool.putconn(returned)
    assert time.monotonic() - begin < 0.1
    # A rollback on return (network round trip) does not block other callers either.
    delay[0] = 0
    dirty = db_pool.getconn()
    dirty.status = psycopg2.extensions.TRANSACTION_STATUS_INTRANS
    dirty.rollback_seconds = 1.0
    roller = threading.Thread(target=db_pool.putconn, args=(dirty,))
    roller.start()
    time.sleep(0.1)
    begin = time.monotonic()
    db_pool.putconn(db_pool.getconn())
    assert time.monotonic() - begin < 0.1
    opener.join(5)
    roller.join(5)
    assert db_pool._used == {}


def test_a_retried_read_shares_one_wait_limit(fake_connect, monkeypatch):
    """retry_database_read's second attempt must not get a fresh 10 s wait."""
    db_pool = main.WaitingConnectionPool(0, 1, wait_seconds=1.0)
    monkeypatch.setattr(main, "db_pool", db_pool)
    held = db_pool.getconn()

    def other():
        time.sleep(0.3)
        conn = db_pool.getconn()  # queues behind the read, then keeps the slot
        time.sleep(3)
        db_pool.putconn(conn)

    def holder():
        time.sleep(0.7)
        db_pool.putconn(held)
    threading.Thread(target=other, daemon=True).start()
    threading.Thread(target=holder, daemon=True).start()

    @main.retry_database_read
    def read():
        with main.get_db_connection():
            raise psycopg2.OperationalError("server closed the connection unexpectedly")

    begin = time.monotonic()
    with pytest.raises(main.HTTPException) as caught:
        read()
    elapsed = time.monotonic() - begin
    assert caught.value.status_code == 503
    # 0.7 s first wait + 0.1 s pause + the 0.3 s left, not another full 1 s.
    assert elapsed < 1.0 + 0.25, elapsed


def test_busy_middleware_replaces_a_handlers_own_500_and_an_escaping_error():
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    def timed_out():
        try:
            raise main.WaitingConnectionPool._timeout(type("P", (), {"maxconn": 1, "wait_seconds": 0.1})())
        except main.pool.PoolError:
            pass

    async def own_500(request):
        timed_out()
        return JSONResponse({"error": "Lookup failed"}, status_code=500)

    async def escapes(request):
        timed_out()
        raise RuntimeError("wrapped the pool error")

    async def not_busy(request):
        return JSONResponse({"error": "real bug"}, status_code=500)

    async def busy_but_ok(request):
        timed_out()  # handled: a degraded 200 stays a 200
        return JSONResponse({"partial": True})

    async def streamed(request):
        async def body():
            yield b'{"started":true}'
            timed_out()
            yield b'{"late":true}'
        return StreamingResponse(body(), status_code=200)

    app = Starlette(routes=[Route(p, f) for p, f in [
        ("/own-500", own_500), ("/escapes", escapes), ("/not-busy", not_busy),
        ("/busy-but-ok", busy_but_ok), ("/streamed", streamed)]])
    app.add_middleware(main.DatabaseBusyMiddleware)

    async def run():
        transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            return {p: await client.get(p) for p in
                    ["/own-500", "/escapes", "/not-busy", "/busy-but-ok", "/streamed"]}
    got = asyncio.run(run())
    for path in ["/own-500", "/escapes"]:
        assert got[path].status_code == 503, path
        assert got[path].json()["error_code"] == "DATABASE_BUSY"
    assert got["/not-busy"].status_code == 500
    assert got["/busy-but-ok"].json() == {"partial": True}
    assert got["/streamed"].status_code == 200  # already sent; cannot be changed
