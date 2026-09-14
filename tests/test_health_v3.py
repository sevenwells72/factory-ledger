"""Scheduling S2 — Health v3: availability and planned-run coverage.

What this file defends (docs/design/so-state-model-findings.md § "Health v3",
docs/design/scheduling-spec-draft.md §4c):

  * Availability — an open line's available pounds are its own explicit
    active allocations (honoured first) plus its share of the SKU's
    unallocated on-hand under the competing-orders waterfall: open orders
    only, cancelled lines out, remaining = effective pounds, priority
    requested_ship_date ASC NULLS LAST → order id → line id, another order's
    explicit reservation off the pool first. The same pound is never
    attributed to two lines. inventory_ready no longer requires an allocation.
  * Coverage — covered_lb sums run_coverage from runs in status planned or
    in_progress ONLY. done and cancelled runs cover nothing.
  * Tiers — only UNCOVERED pounds tier (critical / warning / info by the
    v2.1 windows); a covered shortage is info with the run date(s); a
    covering run planned after the ship date is a warning ("late run"); a
    covering run still `planned` past its planned date is a warning ("run
    overdue"); overdue-with-stock and Not-Ready keep their v2.1 guards on the
    shortage ON PAPER; closed and cancelled orders are silent.
  * Strings — STATUS-006 formatting (ROUND_HALF_UP, separators), product
    names, one aggregated line per order, and no enforcement suffix anywhere:
    main.py, dashboard/, tests/.
  * Response shape — list and detail rows gain available_lb / covered_lb /
    uncovered_lb (per order) and the per-line readiness gains the same plus
    coverage_runs; nothing removed.
  * Health is read-only: the readiness SELECT carries no lock clause and no
    DML.

Unit tests drive compute_so_health() with hand-built readiness rows and a
pinned today; end-to-end tests go through the endpoints against the test
database with runs and coverage inserted directly.
"""

from contextlib import contextmanager
from datetime import date, timedelta
from itertools import count
from pathlib import Path
from uuid import uuid4

import pytest

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover
    pytest.skip("fastapi/httpx not installed", allow_module_level=True)

import main

ROOT = Path(__file__).resolve().parent.parent
_TODAY = date(2026, 6, 15)


# ─────────────────────────────────────────────────────────────────
# Fixtures — same TestClient + savepoint-proxy pattern as the state-model file
# ─────────────────────────────────────────────────────────────────

class _ConnProxy:
    def __init__(self, conn, savepoint):
        self._conn = conn
        self._savepoint = savepoint
        with conn.cursor() as cur:
            cur.execute(f"SAVEPOINT {savepoint}")

    def cursor(self, *args, **kwargs):
        return self._conn.cursor(*args, **kwargs)

    def commit(self):
        with self._conn.cursor() as cur:
            cur.execute(f"RELEASE SAVEPOINT {self._savepoint}")
            cur.execute(f"SAVEPOINT {self._savepoint}")

    def rollback(self):
        with self._conn.cursor() as cur:
            cur.execute(f"ROLLBACK TO SAVEPOINT {self._savepoint}")
            cur.execute(f"SAVEPOINT {self._savepoint}")


@pytest.fixture
def client(_db_connection, monkeypatch):
    @contextmanager
    def _fake_get_conn():
        proxy = _ConnProxy(_db_connection, "health_v3_api")
        try:
            yield proxy
            proxy.commit()
        except Exception:
            proxy.rollback()
            raise

    monkeypatch.setattr(main, "get_db_connection", _fake_get_conn)
    with TestClient(main.app) as test_client:
        test_client.headers["X-API-Key"] = main.API_KEY
        yield test_client


def _seed_customer(cur):
    token = uuid4().hex[:10].upper()
    name = f"HV3 Customer {token}"
    cur.execute("INSERT INTO customers (name, active) VALUES (%s, true) RETURNING id", (name,))
    return cur.fetchone()["id"], name, token


def _seed_product(cur, token, *, stock=0):
    suffix = uuid4().hex[:6].upper()
    cur.execute(
        "INSERT INTO products (name, type, odoo_code, uom, is_service, active) "
        "VALUES (%s, 'finished', %s, 'lb', false, true) RETURNING id",
        (f"HV3 FG {token} {suffix}", f"HV3-{token}-{suffix}"),
    )
    product_id = cur.fetchone()["id"]
    cur.execute(
        "INSERT INTO lots (product_id, lot_code, entry_source, received_at) "
        "VALUES (%s, %s, 'received', NOW()) RETURNING id",
        (product_id, f"HV3-LOT-{token}-{suffix}"),
    )
    lot_id = cur.fetchone()["id"]
    if stock:
        cur.execute("INSERT INTO transactions (type, timestamp) VALUES ('receive', NOW()) RETURNING id")
        txn = cur.fetchone()["id"]
        cur.execute(
            "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) "
            "VALUES (%s, %s, %s, %s)",
            (txn, product_id, lot_id, stock),
        )
    return product_id, lot_id


def _seed_order(cur, customer_id, token, *, ship_date=None, floor_ready=True,
                state=None, reason=None):
    order_number = f"HV3-SO-{token}-{uuid4().hex[:5]}"
    cur.execute(
        "INSERT INTO sales_orders (customer_id, order_number, status, requested_ship_date) "
        "VALUES (%s, %s, 'confirmed', %s) RETURNING id",
        (customer_id, order_number, ship_date),
    )
    order_id = cur.fetchone()["id"]
    if state is not None:
        cur.execute(
            "UPDATE sales_orders SET state = %s, state_reason = %s WHERE id = %s",
            (state, reason, order_id),
        )
    if floor_ready:
        cur.execute(
            "INSERT INTO sales_order_flags (so_number, ready, ready_at, ready_by) "
            "VALUES (%s, true, NOW(), 'test')",
            (order_number,),
        )
    return order_id, order_number


def _add_line(cur, order_id, product_id, qty, *, status="pending"):
    cur.execute(
        "INSERT INTO sales_order_lines "
        "(sales_order_id, product_id, quantity_lb, quantity_shipped_lb, line_status) "
        "VALUES (%s, %s, %s, 0, %s) RETURNING id",
        (order_id, product_id, qty, status),
    )
    return cur.fetchone()["id"]


def _allocate(cur, order_id, line_id, product_id, qty):
    cur.execute(
        "INSERT INTO sales_order_allocations "
        "(sales_order_id, sales_order_line_id, product_id, quantity_lb, source) "
        "VALUES (%s, %s, %s, %s, 'manual') RETURNING id",
        (order_id, line_id, product_id, qty),
    )
    return cur.fetchone()["id"]


def _run(cur, product_id, qty, planned_date, *, status="planned"):
    cur.execute(
        "INSERT INTO production_runs (product_id, planned_qty_lb, planned_qty, planned_unit, "
        "planned_date, status) VALUES (%s, %s, %s, 'lb', %s, %s) RETURNING id",
        (product_id, qty, qty, planned_date, status),
    )
    return cur.fetchone()["id"]


def _cover(cur, run_id, line_id, qty):
    cur.execute(
        "INSERT INTO run_coverage (run_id, sales_order_line_id, qty_lb) VALUES (%s, %s, %s)",
        (run_id, line_id, qty),
    )


def _detail(client, order_id):
    response = client.get(f"/sales/orders/{order_id}")
    assert response.status_code == 200, response.text
    return response.json()


def _listed(client, customer_name):
    response = client.get("/sales/orders", params={"customer": customer_name, "limit": 50})
    assert response.status_code == 200, response.text
    return {row["order_id"]: row for row in response.json()["orders"]}


def _codes(payload):
    return {item["code"]: item["severity"] for item in payload["blockers"]}


def _fmt(d):
    return main._so_run_date(d, main._factory_today())


# ─────────────────────────────────────────────────────────────────
# Unit helpers — readiness rows shaped like _line_readiness() yields
# ─────────────────────────────────────────────────────────────────

_run_ids = count(9000)


def _r(planned, qty, status="planned", run_id=None):
    """One coverage_runs entry, dates as the jsonb carries them (ISO text)."""
    return {"run_id": run_id or next(_run_ids), "planned_date": planned.isoformat(),
            "status": status, "qty_lb": qty}


def _line(shortage=0.0, *, covered=None, runs=(), unallocated=0.0, line_id=1,
          sku=None, product=None, v21=False):
    """`v21=True` builds the pre-S2 row (no coverage keys at all)."""
    readiness = {"shortage_lb": shortage, "unallocated_need_lb": unallocated}
    if not v21:
        if covered is None:
            covered = sum(r["qty_lb"] for r in runs)
        readiness.update({
            "covered_lb": covered,
            "uncovered_lb": max(0.0, shortage - covered),
            "coverage_runs": list(runs),
        })
    return {"line_id": line_id, "sku": sku or f"SKU-{line_id}",
            "product": product, "readiness": readiness}


def _health(lines, *, days_out=None, floor_ready=True, state="open",
            fulfillment="unshipped", today=_TODAY):
    return main.compute_so_health(
        state=state,
        fulfillment=fulfillment,
        requested_ship_date=(None if days_out is None
                             else today + timedelta(days=days_out)),
        floor_ready=floor_ready,
        line_readiness=lines,
        today=today,
    )


def _d(days):
    return _TODAY + timedelta(days=days)


# ═════════════════════════════════════════════════════════════════
# Covered shortage — info, never a tier, stated with the run date(s)
# ═════════════════════════════════════════════════════════════════

def test_covered_shortage_is_info_with_the_run_date():
    health = _health([_line(500, runs=[_r(_d(5), 500)])], days_out=8)
    assert health["level"] == "quiet", health
    assert health["reasons"] == []
    assert health["info"] == ["Short 500 lb — covered by run on Jun 20"]


def test_covered_shortage_across_two_lines_lists_both_run_dates():
    health = _health([
        _line(500, runs=[_r(_d(5), 500)], line_id=1),
        _line(200, runs=[_r(_d(7), 200)], line_id=2),
    ], days_out=8)
    assert health["level"] == "quiet"
    assert health["info"] == [
        "Short 700 lb on 2 lines — covered by runs on Jun 20 and Jun 22"]


def test_two_runs_on_one_day_say_runs_on_one_date():
    health = _health([_line(500, runs=[_r(_d(5), 300), _r(_d(5), 200)])],
                     days_out=8)
    assert health["info"] == ["Short 500 lb — covered by runs on Jun 20"]


def test_three_run_dates_are_comma_joined_with_a_final_and():
    health = _health([_line(600, runs=[_r(_d(3), 200), _r(_d(5), 200),
                                        _r(_d(7), 200)])], days_out=8)
    assert health["info"] == [
        "Short 600 lb — covered by runs on Jun 18, Jun 20 and Jun 22"]


def test_coverage_beyond_the_shortage_reports_only_the_shortage():
    """A run that covers more than the line is short does not invent pounds:
    covered_short is min(shortage, covered)."""
    health = _health([_line(300, runs=[_r(_d(5), 500)])], days_out=8)
    assert health["info"] == ["Short 300 lb — covered by run on Jun 20"]


def test_half_covered_order_reads_as_two_facts():
    """Uncovered remainder tiers with the v2.1 wording; the covered part is a
    separate info line carrying the run date."""
    health = _health([_line(500, runs=[_r(_d(2), 200)])], days_out=3)
    assert health["level"] == "critical", health
    assert health["reasons"] == ["Short 300 lb — ships in 3 days"]
    assert health["info"] == ["Short 200 lb — covered by run on Jun 17"]


def test_half_covered_counts_lines_separately_for_each_sentence():
    """Line 1 is half covered, line 2 wholly uncovered, line 3 wholly covered:
    'on N lines' counts the lines in each sentence, not the order's lines."""
    health = _health([
        _line(500, runs=[_r(_d(1), 200)], line_id=1),
        _line(100, line_id=2),
        _line(50, runs=[_r(_d(2), 50)], line_id=3),
    ], days_out=3)
    assert health["reasons"] == ["Short 400 lb on 2 lines — ships in 3 days"]
    assert health["info"] == [
        "Short 250 lb on 2 lines — covered by runs on Jun 16 and Jun 17"]


# ═════════════════════════════════════════════════════════════════
# Windows — uncovered tiers exactly as v2.1, covered never tiers
# ═════════════════════════════════════════════════════════════════

_WINDOWS = [
    # (days_out, uncovered level, where the uncovered sentence lands)
    (-1, "critical", "reasons"),
    (0, "critical", "reasons"),
    (5, "critical", "reasons"),      # == today + SO_HEALTH_CRITICAL_DAYS
    (6, "warning", "reasons"),
    (10, "warning", "reasons"),      # == today + SO_HEALTH_WARNING_DAYS
    (11, "quiet", "info"),
    (None, "warning", "reasons"),    # no ship date: attention, not a crisis
]


@pytest.mark.parametrize("days_out,level,where", _WINDOWS,
                         ids=[f"days={w[0]}" for w in _WINDOWS])
def test_uncovered_shortage_tiers_by_the_v21_windows(days_out, level, where,
                                                     monkeypatch):
    monkeypatch.delenv("SO_HEALTH_CRITICAL_DAYS", raising=False)
    monkeypatch.delenv("SO_HEALTH_WARNING_DAYS", raising=False)
    health = _health([_line(500)], days_out=days_out)
    assert health["level"] == level, health
    phrase = main._so_ship_phrase(
        None if days_out is None else _TODAY + timedelta(days=days_out), _TODAY)
    expected = "Short 500 lb" + (f" — {phrase}" if phrase else "")
    assert health[where] == [expected], health
    other = "info" if where == "reasons" else "reasons"
    assert health[other] == []


@pytest.mark.parametrize("days_out", [w[0] for w in _WINDOWS],
                         ids=[f"days={w[0]}" for w in _WINDOWS])
def test_covered_shortage_never_tiers_at_any_window(days_out, monkeypatch):
    """The same shortage, fully covered by a run before the ship date (or any
    run when there is no ship date): quiet everywhere, info everywhere."""
    monkeypatch.delenv("SO_HEALTH_CRITICAL_DAYS", raising=False)
    monkeypatch.delenv("SO_HEALTH_WARNING_DAYS", raising=False)
    run_day = _d(-3) if days_out is not None and days_out < 0 else _d(0)
    # in_progress so a run in the past is not "overdue" — what is under test
    # here is the window, not the run's own state.
    health = _health([_line(500, runs=[_r(run_day, 500, "in_progress")])],
                     days_out=days_out)
    assert health["level"] == "quiet", health
    assert health["reasons"] == []
    assert health["info"] == [
        f"Short 500 lb — covered by run on {main._so_run_date(run_day, _TODAY)}"]


def test_critical_is_gated_on_the_uncovered_pounds_not_the_paper_shortage():
    """Short on paper inside the critical window but fully scheduled: not
    critical, not warning — the window belongs to 'nobody is making it'."""
    health = _health([_line(500, runs=[_r(_d(1), 500)])], days_out=2)
    assert health["level"] == "quiet", health


# ═════════════════════════════════════════════════════════════════
# Late run — warning, never critical
# ═════════════════════════════════════════════════════════════════

def test_late_run_is_a_warning():
    health = _health([_line(500, runs=[_r(_d(5), 500)])], days_out=3)
    assert health["level"] == "warning", health
    assert health["reasons"] == [
        "Run for 500 lb planned Jun 20 — after ship date Jun 18"]
    assert health["info"] == ["Short 500 lb — covered by run on Jun 20"]


def test_late_run_is_never_critical_even_inside_the_critical_window():
    health = _health([_line(500, runs=[_r(_d(2), 500)])], days_out=1)
    assert health["level"] == "warning", health


def test_late_run_on_the_ship_date_itself_is_not_late():
    health = _health([_line(500, runs=[_r(_d(3), 500)])], days_out=3)
    assert health["level"] == "quiet", health
    assert health["reasons"] == []


def test_late_run_with_an_uncovered_remainder_is_critical_and_lists_both():
    health = _health([_line(500, runs=[_r(_d(5), 200)])], days_out=3)
    assert health["level"] == "critical", health
    assert health["reasons"] == [
        "Short 300 lb — ships in 3 days",
        "Run for 200 lb planned Jun 20 — after ship date Jun 18",
    ]


def test_two_late_runs_aggregate_to_one_reason():
    health = _health([
        _line(500, runs=[_r(_d(5), 500)], line_id=1),
        _line(200, runs=[_r(_d(7), 200)], line_id=2),
    ], days_out=3)
    assert health["reasons"] == [
        "Runs for 700 lb planned Jun 20 and Jun 22 — after ship date Jun 18"]


def test_late_run_pounds_are_the_coverage_on_this_order_not_the_run_plan():
    """A run may cover several orders; the reason states what it makes for
    THIS order's lines (the coverage row), which is all the row carries."""
    health = _health([_line(500, runs=[_r(_d(5), 120)])], days_out=3)
    assert "Run for 120 lb planned Jun 20" in health["reasons"][1]


def test_no_ship_date_means_no_late_run():
    health = _health([_line(500, runs=[_r(_d(40), 500)])], days_out=None)
    assert health["level"] == "quiet", health
    assert health["reasons"] == []


# ═════════════════════════════════════════════════════════════════
# Run overdue — warning, status planned only
# ═════════════════════════════════════════════════════════════════

def test_run_overdue_is_a_warning():
    health = _health([_line(500, runs=[_r(_d(-3), 500)])], days_out=8)
    assert health["level"] == "warning", health
    assert health["reasons"] == ["Run for 500 lb planned Jun 12 has not started"]
    assert health["info"] == ["Short 500 lb — covered by run on Jun 12"]


def test_in_progress_run_past_its_date_is_not_overdue():
    health = _health([_line(500, runs=[_r(_d(-3), 500, "in_progress")])],
                     days_out=8)
    assert health["level"] == "quiet", health
    assert health["reasons"] == []


def test_run_planned_today_is_not_overdue():
    health = _health([_line(500, runs=[_r(_d(0), 500)])], days_out=8)
    assert health["reasons"] == []


def test_two_overdue_runs_say_have_not_started():
    health = _health([_line(500, runs=[_r(_d(-5), 300), _r(_d(-3), 200)])],
                     days_out=8)
    assert health["reasons"] == [
        "Runs for 500 lb planned Jun 10 and Jun 12 have not started"]


def test_an_overdue_run_that_is_also_late_lists_both_reasons():
    """Ship date passed, run planned after it, still not started."""
    health = _health([_line(500, runs=[_r(_d(-2), 500)])], days_out=-4)
    assert health["level"] == "warning", health
    assert health["reasons"] == [
        "Run for 500 lb planned Jun 13 — after ship date Jun 11",
        "Run for 500 lb planned Jun 13 has not started",
    ]


def test_run_overdue_is_measured_on_the_factory_date(monkeypatch):
    """The run's own date is compared with the same `today` every other
    health date uses."""
    run_day = date(2026, 9, 10)
    line = _line(500, runs=[_r(run_day, 500)])
    on_the_day = _health([line], days_out=8, today=run_day)
    next_day = _health([line], days_out=8, today=run_day + timedelta(days=1))
    assert on_the_day["reasons"] == []
    assert next_day["reasons"] == ["Run for 500 lb planned Sep 10 has not started"]


# ═════════════════════════════════════════════════════════════════
# Which runs count as "covering"
# ═════════════════════════════════════════════════════════════════

def test_a_run_on_a_line_that_is_not_short_is_neither_late_nor_overdue():
    """Coverage on a line whose stock is there is not doing work Health can
    be late about."""
    health = _health([_line(0, runs=[_r(_d(-3), 500)])], days_out=1)
    assert health == {"level": "quiet", "reasons": [], "info": [],
                      "info_detail": []}


def test_a_v21_readiness_row_without_coverage_keys_is_wholly_uncovered():
    """Older fixtures and any caller that has not learned the new keys read
    exactly as before S2 — coverage can only shrink what tiers."""
    health = _health([_line(500, v21=True)], days_out=3)
    assert health["level"] == "critical"
    assert health["reasons"] == ["Short 500 lb — ships in 3 days"]
    assert health["info"] == []


# ═════════════════════════════════════════════════════════════════
# The v2.1 guards keep their meaning on the shortage ON PAPER
# ═════════════════════════════════════════════════════════════════

def test_overdue_with_stock_is_suppressed_when_short_but_covered():
    """Short on paper, fully scheduled, past the ship date: not 'stock on
    hand' — the covered sentence is the whole story."""
    health = _health([_line(500, runs=[_r(_d(-3), 500, "in_progress")])],
                     days_out=-2)
    assert health["level"] == "quiet", health
    assert health["reasons"] == []
    assert not any("stock on hand" in r for r in health["reasons"] + health["info"])


def test_overdue_with_stock_is_kept_when_nothing_is_short():
    health = _health([_line(0)], days_out=-2)
    assert health["reasons"] == ["2 days overdue — stock on hand"]


def test_not_ready_is_suppressed_when_short_but_covered():
    health = _health([_line(500, runs=[_r(_d(0), 500, "in_progress")])],
                     days_out=1, floor_ready=False)
    assert not any("Not Ready" in r for r in health["reasons"]), health
    assert health["level"] == "quiet"


def test_not_ready_is_kept_when_nothing_is_short():
    health = _health([_line(0)], days_out=1, floor_ready=False)
    assert health["reasons"] == ["Not Ready to Ship — ships tomorrow"]


@pytest.mark.parametrize("state", ["closed", "cancelled"])
def test_closed_and_cancelled_are_silent_even_with_late_and_overdue_runs(state):
    health = _health([_line(500, runs=[_r(_d(-3), 200), _r(_d(9), 300)],
                            unallocated=500)],
                     days_out=-12, floor_ready=False, state=state)
    assert health == {"level": "quiet", "reasons": [], "info": [],
                      "info_detail": []}


# ═════════════════════════════════════════════════════════════════
# Strings — STATUS-006, dates, product names, no enforcement suffix
# ═════════════════════════════════════════════════════════════════

def test_covered_and_run_pounds_round_half_up_with_separators():
    health = _health([_line(1606.5, runs=[_r(_d(5), 1606.5)])], days_out=3)
    assert health["reasons"] == [
        "Run for 1,607 lb planned Jun 20 — after ship date Jun 18"]
    assert health["info"] == ["Short 1,607 lb — covered by run on Jun 20"]


def test_uncovered_pounds_round_half_up_too():
    health = _health([_line(1000.5, runs=[_r(_d(1), 394)])], days_out=3)
    assert health["reasons"] == ["Short 607 lb — ships in 3 days"]  # 606.5 → 607


def test_run_date_carries_the_year_only_when_it_is_not_this_year():
    assert main._so_run_date(date(2026, 9, 5), _TODAY) == "Sep 5"
    assert main._so_run_date("2026-09-05", _TODAY) == "Sep 5"
    assert main._so_run_date(date(2027, 1, 5), _TODAY) == "Jan 5, 2027"
    health = _health([_line(500, runs=[_r(date(2027, 1, 5), 500)])], days_out=8)
    assert health["reasons"] == [
        "Run for 500 lb planned Jan 5, 2027 — after ship date Jun 23"]


def test_ship_date_in_the_late_run_reason_uses_the_same_formatter():
    health = _health([_line(500, runs=[_r(date(2027, 2, 1), 500)])],
                     days_out=None if False else 200)
    assert health["reasons"][0].endswith("— after ship date Jan 1, 2027")


@pytest.mark.parametrize("items,expected", [
    (["Jun 20"], "Jun 20"),
    (["Jun 20", "Jun 22"], "Jun 20 and Jun 22"),
    (["Jun 18", "Jun 20", "Jun 22"], "Jun 18, Jun 20 and Jun 22"),
    ([], ""),
])
def test_so_join(items, expected):
    assert main._so_join(items) == expected


def test_no_v3_string_carries_a_decimal_point_or_a_bare_thousand(monkeypatch):
    """The STATUS-006 audit against the assembled v3 sentences."""
    monkeypatch.setattr(main, "_allocations_enforced", lambda: False)
    health = _health([
        _line(13500.25, runs=[_r(_d(-3), 4000.5), _r(_d(9), 2000.75)],
              unallocated=13500.25, line_id=1),
        _line(1234.5, runs=[_r(_d(2), 1234.5)], line_id=2),
    ], days_out=3)
    strings = health["reasons"] + health["info"]
    assert len(strings) >= 4, strings
    for text in strings:
        assert "." not in text.replace("Jun ", ""), text
        for token in text.replace(",", "").split():
            if token.isdigit() and int(token) >= 1000:
                assert f"{int(token):,}" in text, (token, text)


def test_unallocated_info_never_carries_the_enforcement_suffix(monkeypatch):
    needle = "allocations not " + "enforced"
    for flag in (True, False):
        monkeypatch.setattr(main, "_allocations_enforced", lambda flag=flag: flag)
        health = _health([_line(0, unallocated=60, product="Granola Maple")],
                         days_out=30)
        assert health["info"] == ["60 lb not allocated on Granola Maple (SKU-1)"]
        assert needle not in " ".join(health["info"] + health["reasons"])
        assert health["info_detail"] == [{
            "line_id": 1, "sku": "SKU-1", "product_name": "Granola Maple",
            "unallocated_lb": 60.0}]


def test_health_no_longer_reads_the_enforcement_flag(monkeypatch):
    """The flag is not consulted at all: a flag that raises must not reach
    compute_so_health()."""
    def _boom():
        raise AssertionError("compute_so_health() read _allocations_enforced()")
    monkeypatch.setattr(main, "_allocations_enforced", _boom)
    health = _health([_line(500, runs=[_r(_d(5), 200)], unallocated=500)],
                     days_out=3)
    assert health["level"] == "critical"


def test_the_enforcement_string_is_gone_from_every_consumer():
    """Merge criterion: grepping main.py, dashboard/ and tests/ for the suffix
    returns nothing. The needle is assembled so this file itself does
    not trip it."""
    needle = "allocations not " + "enforced"
    files = [ROOT / "main.py"]
    files += sorted((ROOT / "dashboard").glob("*.js"))
    files += sorted((ROOT / "dashboard").glob("*.html"))
    files += sorted((ROOT / "tests").rglob("*.py"))
    files += sorted((ROOT / "tests").rglob("*.js"))
    files += sorted((ROOT / "tests").rglob("*.mjs"))
    files += sorted((ROOT / "tests").rglob("*.json"))
    offenders = [
        str(path.relative_to(ROOT)) for path in files
        if "node_modules" not in path.parts
        and needle in path.read_text(encoding="utf-8", errors="ignore").lower()
    ]
    assert offenders == [], offenders


def test_health_is_read_only_no_lock_clause_no_dml():
    sql = main.SALES_ORDER_READINESS_SQL.upper()
    for forbidden in ("FOR UPDATE", "FOR NO KEY UPDATE", "FOR SHARE",
                      "FOR KEY SHARE", "INSERT ", "UPDATE ", "DELETE "):
        assert forbidden not in sql, forbidden
    assert "RUN_COVERAGE" in sql and "PRODUCTION_RUNS" in sql


# ═════════════════════════════════════════════════════════════════
# End to end — the competing-orders waterfall
# ═════════════════════════════════════════════════════════════════

@pytest.mark.db
def test_competing_orders_never_share_a_pound(db_cursor, client):
    """100 lb on hand. A closed order still holds an explicit 10 lb
    reservation (foreign, off the pool first). Three open orders compete for
    the remaining 90: A (created first, ships in 10 days) wants 80, B (created
    second, ships in 5 days) wants 80, C (undated) wants 50. A cancelled
    1,000 lb line on B must not count.

    Priority is the ship date, so B goes first despite the higher id."""
    customer_id, customer_name, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, stock=100)
    today = main._factory_today()

    closed_id, _ = _seed_order(db_cursor, customer_id, token, ship_date=today,
                               state="closed", reason="short_closed")
    closed_line = _add_line(db_cursor, closed_id, product_id, 10)
    _allocate(db_cursor, closed_id, closed_line, product_id, 10)

    a_id, _ = _seed_order(db_cursor, customer_id, token, ship_date=today + timedelta(days=10))
    a_line = _add_line(db_cursor, a_id, product_id, 80)
    b_id, _ = _seed_order(db_cursor, customer_id, token, ship_date=today + timedelta(days=5))
    b_line = _add_line(db_cursor, b_id, product_id, 80)
    _add_line(db_cursor, b_id, product_id, 1000, status="cancelled")
    c_id, _ = _seed_order(db_cursor, customer_id, token, ship_date=None)
    c_line = _add_line(db_cursor, c_id, product_id, 50)

    rows = _listed(client, customer_name)
    assert rows[b_id]["available_lb"] == pytest.approx(80)
    assert rows[b_id]["shortage_lb"] == pytest.approx(0)
    assert rows[b_id]["inventory_ready"] is True
    assert rows[a_id]["available_lb"] == pytest.approx(10)
    assert rows[a_id]["shortage_lb"] == pytest.approx(70)
    assert rows[c_id]["available_lb"] == pytest.approx(0)
    assert rows[c_id]["shortage_lb"] == pytest.approx(50)
    assert rows[c_id]["health"]["level"] == "warning"       # undated shortage
    assert rows[c_id]["health"]["reasons"] == ["Short 50 lb"]
    # No double attribution: the open orders' availability sums to exactly
    # what is on hand minus the foreign reservation.
    assert (rows[a_id]["available_lb"] + rows[b_id]["available_lb"]
            + rows[c_id]["available_lb"]) == pytest.approx(90)
    assert rows[closed_id]["health"] == {"level": "quiet", "reasons": [],
                                         "info": [], "info_detail": []}

    # Detail rows carry the same numbers per line.
    a_ready = _detail(client, a_id)["lines"][0]["readiness"]
    assert a_ready["available_lb"] == pytest.approx(10)
    assert a_ready["coverable_lb"] == pytest.approx(10)
    assert a_ready["shortage_lb"] == pytest.approx(70)
    assert a_ready["uncovered_lb"] == pytest.approx(70)
    assert a_ready["covered_lb"] == pytest.approx(0)
    assert a_ready["coverage_runs"] == []

    # Now A reserves 30 explicitly. Explicit allocations win first: the pool
    # drops to 100 − 10 − 30 = 60, B takes all 60, A has its 30 and nothing
    # more, C still nothing. Still exactly 90 handed out.
    _allocate(db_cursor, a_id, a_line, product_id, 30)
    rows = _listed(client, customer_name)
    assert rows[b_id]["available_lb"] == pytest.approx(60)
    assert rows[b_id]["shortage_lb"] == pytest.approx(20)
    assert rows[a_id]["available_lb"] == pytest.approx(30)
    assert rows[a_id]["shortage_lb"] == pytest.approx(50)
    assert rows[c_id]["available_lb"] == pytest.approx(0)
    assert (rows[a_id]["available_lb"] + rows[b_id]["available_lb"]
            + rows[c_id]["available_lb"]) == pytest.approx(90)
    del b_line, c_line


@pytest.mark.db
def test_same_ship_date_breaks_ties_on_order_id_then_line_id(db_cursor, client):
    customer_id, customer_name, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, stock=100)
    today = main._factory_today()
    a_id, _ = _seed_order(db_cursor, customer_id, token, ship_date=today + timedelta(days=7))
    a_one = _add_line(db_cursor, a_id, product_id, 60)
    a_two = _add_line(db_cursor, a_id, product_id, 60)
    b_id, _ = _seed_order(db_cursor, customer_id, token, ship_date=today + timedelta(days=7))
    _add_line(db_cursor, b_id, product_id, 60)

    by_line = {line["line_id"]: line["readiness"] for line in _detail(client, a_id)["lines"]}
    assert by_line[a_one]["available_lb"] == pytest.approx(60)
    assert by_line[a_two]["available_lb"] == pytest.approx(40)
    assert by_line[a_two]["shortage_lb"] == pytest.approx(20)
    rows = _listed(client, customer_name)
    assert rows[a_id]["available_lb"] == pytest.approx(100)
    assert rows[b_id]["available_lb"] == pytest.approx(0)
    assert rows[b_id]["shortage_lb"] == pytest.approx(60)


@pytest.mark.db
def test_an_order_off_the_page_still_competes(db_cursor, client):
    """The waterfall runs over the SKU's whole open demand, not the page:
    fetching only the later order still sees the earlier one's claim."""
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, stock=100)
    today = main._factory_today()
    first_id, _ = _seed_order(db_cursor, customer_id, token, ship_date=today + timedelta(days=2))
    _add_line(db_cursor, first_id, product_id, 100)
    later_id, _ = _seed_order(db_cursor, customer_id, token, ship_date=today + timedelta(days=9))
    _add_line(db_cursor, later_id, product_id, 100)

    later = _detail(client, later_id)
    assert later["available_lb"] == pytest.approx(0)
    assert later["shortage_lb"] == pytest.approx(100)
    assert later["health"]["level"] == "warning"
    assert later["health"]["reasons"] == ["Short 100 lb — ships in 9 days"]


@pytest.mark.db
def test_a_reservation_beyond_on_hand_is_capped_at_the_stock(db_cursor, client):
    """A stale 100 lb allocation on 0 lb of stock does not read as available:
    explicit allocations win first, but only as far as stock exists."""
    customer_id, _, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, stock=0)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=main._factory_today() + timedelta(days=3))
    line_id = _add_line(db_cursor, order_id, product_id, 100)
    _allocate(db_cursor, order_id, line_id, product_id, 100)

    body = _detail(client, order_id)
    assert body["available_lb"] == pytest.approx(0)
    assert body["shortage_lb"] == pytest.approx(100)
    assert body["inventory_ready"] is False
    assert body["health"]["level"] == "critical"


@pytest.mark.db
def test_inventory_ready_no_longer_requires_an_allocation(db_cursor, client):
    """S2 removes the `allocated >= remaining` gate: stock available under the
    waterfall is inventory-ready. The `unallocated` blocker is informational
    (allocation is a reservation, not a dispatch gate) and dispatch_ready
    is true."""
    customer_id, customer_name, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, stock=100)
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=main._factory_today() + timedelta(days=30))
    _add_line(db_cursor, order_id, product_id, 100)

    body = _detail(client, order_id)
    assert body["inventory_ready"] is True
    assert body["lines"][0]["readiness"]["inventory_ready"] is True
    assert body["shortage_lb"] == pytest.approx(0)
    assert _codes(body) == {"unallocated": "info"}
    assert body["dispatch_ready"] is True
    assert body["health"]["level"] == "quiet"
    assert _listed(client, customer_name)[order_id]["inventory_ready"] is True


# ═════════════════════════════════════════════════════════════════
# End to end — coverage
# ═════════════════════════════════════════════════════════════════

def _short_order(db_cursor, *, days_out=8, floor_ready=True):
    customer_id, customer_name, token = _seed_customer(db_cursor)
    product_id, _ = _seed_product(db_cursor, token, stock=0)
    today = main._factory_today()
    order_id, _ = _seed_order(db_cursor, customer_id, token,
                              ship_date=today + timedelta(days=days_out),
                              floor_ready=floor_ready)
    line_id = _add_line(db_cursor, order_id, product_id, 500)
    return customer_name, product_id, order_id, line_id, today


@pytest.mark.db
def test_planned_run_covers_the_shortage_end_to_end(db_cursor, client):
    customer_name, product_id, order_id, line_id, today = _short_order(db_cursor)
    run_day = today + timedelta(days=3)
    run_id = _run(db_cursor, product_id, 500, run_day)
    _cover(db_cursor, run_id, line_id, 500)

    body = _detail(client, order_id)
    line = body["lines"][0]["readiness"]
    assert line["available_lb"] == pytest.approx(0)
    assert line["shortage_lb"] == pytest.approx(500)
    assert line["covered_lb"] == pytest.approx(500)
    assert line["uncovered_lb"] == pytest.approx(0)
    assert line["coverage_runs"] == [{
        "run_id": run_id, "planned_date": run_day.isoformat(),
        "status": "planned", "qty_lb": 500.0}]
    assert body["covered_lb"] == pytest.approx(500)
    assert body["uncovered_lb"] == pytest.approx(0)
    assert body["available_lb"] == pytest.approx(0)
    assert body["health"]["level"] == "quiet", body["health"]
    assert body["health"]["reasons"] == []
    # The covered sentence, then the (unchanged, suffix-free) unallocated one.
    assert body["health"]["info"][0] == \
        f"Short 500 lb — covered by run on {_fmt(run_day)}"
    assert len(body["health"]["info"]) == 2
    assert body["health"]["info"][1].startswith("500 lb not allocated on ")

    row = _listed(client, customer_name)[order_id]
    assert row["covered_lb"] == pytest.approx(500)
    assert row["uncovered_lb"] == pytest.approx(0)
    assert row["available_lb"] == pytest.approx(0)
    assert row["health"] == body["health"]


@pytest.mark.db
def test_in_progress_run_covers_too(db_cursor, client):
    _, product_id, order_id, line_id, today = _short_order(db_cursor)
    run_id = _run(db_cursor, product_id, 500, today - timedelta(days=1), status="in_progress")
    _cover(db_cursor, run_id, line_id, 500)
    body = _detail(client, order_id)
    assert body["covered_lb"] == pytest.approx(500)
    assert body["health"]["level"] == "quiet", body["health"]
    assert body["health"]["reasons"] == []


@pytest.mark.parametrize("status", ["done", "cancelled"])
@pytest.mark.db
def test_done_and_cancelled_runs_do_not_cover(db_cursor, client, status):
    """A completed run never masks a shortage — the ledger governs; a
    cancelled run's coverage rows stay in the table and are ignored."""
    _, product_id, order_id, line_id, today = _short_order(db_cursor)
    run_id = _run(db_cursor, product_id, 500, today + timedelta(days=3), status=status)
    _cover(db_cursor, run_id, line_id, 500)

    body = _detail(client, order_id)
    line = body["lines"][0]["readiness"]
    assert line["covered_lb"] == pytest.approx(0)
    assert line["uncovered_lb"] == pytest.approx(500)
    assert line["coverage_runs"] == []
    assert body["covered_lb"] == pytest.approx(0)
    assert body["uncovered_lb"] == pytest.approx(500)
    assert body["health"]["level"] == "warning", body["health"]
    assert body["health"]["reasons"] == ["Short 500 lb — ships in 8 days"]
    assert not any("covered" in i for i in body["health"]["info"])


@pytest.mark.db
def test_a_done_run_on_a_half_covered_line_leaves_only_the_live_run(db_cursor, client):
    _, product_id, order_id, line_id, today = _short_order(db_cursor)
    live = _run(db_cursor, product_id, 200, today + timedelta(days=3))
    _cover(db_cursor, live, line_id, 200)
    done = _run(db_cursor, product_id, 300, today + timedelta(days=1), status="done")
    _cover(db_cursor, done, line_id, 300)

    body = _detail(client, order_id)
    assert body["covered_lb"] == pytest.approx(200)
    assert body["uncovered_lb"] == pytest.approx(300)
    assert [r["run_id"] for r in body["lines"][0]["readiness"]["coverage_runs"]] == [live]
    assert body["health"]["level"] == "warning"
    assert body["health"]["reasons"] == ["Short 300 lb — ships in 8 days"]
    assert body["health"]["info"][0] == \
        f"Short 200 lb — covered by run on {_fmt(today + timedelta(days=3))}"


@pytest.mark.db
def test_late_run_end_to_end(db_cursor, client):
    _, product_id, order_id, line_id, today = _short_order(db_cursor, days_out=3)
    run_day = today + timedelta(days=5)
    run_id = _run(db_cursor, product_id, 500, run_day)
    _cover(db_cursor, run_id, line_id, 500)
    health = _detail(client, order_id)["health"]
    assert health["level"] == "warning", health
    assert health["reasons"] == [
        f"Run for 500 lb planned {_fmt(run_day)} — after ship date "
        f"{_fmt(today + timedelta(days=3))}"]


@pytest.mark.db
def test_run_overdue_end_to_end(db_cursor, client):
    _, product_id, order_id, line_id, today = _short_order(db_cursor)
    run_day = today - timedelta(days=2)
    run_id = _run(db_cursor, product_id, 500, run_day)
    _cover(db_cursor, run_id, line_id, 500)
    health = _detail(client, order_id)["health"]
    assert health["level"] == "warning", health
    assert health["reasons"] == [
        f"Run for 500 lb planned {_fmt(run_day)} has not started"]


@pytest.mark.db
def test_coverage_does_not_change_availability(db_cursor, client):
    """Coverage answers 'is the shortage scheduled', never 'is there stock':
    a covered line still shows shortage_lb and is not inventory-ready."""
    _, product_id, order_id, line_id, today = _short_order(db_cursor)
    run_id = _run(db_cursor, product_id, 500, today + timedelta(days=3))
    _cover(db_cursor, run_id, line_id, 500)
    body = _detail(client, order_id)
    assert body["shortage_lb"] == pytest.approx(500)
    assert body["inventory_ready"] is False
    assert _codes(body)["shortage"] == "block"


# ═════════════════════════════════════════════════════════════════
# Response shape — additive only
# ═════════════════════════════════════════════════════════════════

_V21_LIST_FIELDS = (
    "order_id", "order_number", "customer", "order_date", "requested_ship_date",
    "status", "customer_po", "source_document_id", "line_count", "total_lb",
    "shipped_lb", "remaining_lb", "total_units", "shipped_units",
    "remaining_units", "pallet_lines", "ready", "ready_at", "ready_by", "note",
    "overdue", "inventory_ready", "dispatch_ready", "fulfillment_diverged",
    "shortage_lb", "allocated_lb", "remaining_effective_lb", "blockers",
    "state", "fulfillment", "health",
)
_V21_LINE_READINESS = (
    "ordered_lb", "shipped_recorded_lb", "shipped_effective_lb", "remaining_lb",
    "on_hand_lb", "allocated_lb", "allocated_sku_lb", "allocated_lot_lb",
    "available_lb", "coverable_lb", "shortage_lb", "unallocated_need_lb",
    "inbound_open_lb", "inventory_ready", "fulfillment_diverged", "blockers",
)


@pytest.mark.db
def test_list_and_detail_gain_the_additive_fields_and_lose_nothing(db_cursor, client):
    customer_name, product_id, order_id, line_id, today = _short_order(db_cursor)
    run_id = _run(db_cursor, product_id, 200, today + timedelta(days=3))
    _cover(db_cursor, run_id, line_id, 200)

    row = _listed(client, customer_name)[order_id]
    for field in _V21_LIST_FIELDS:
        assert field in row, f"list lost {field}"
    for field in ("available_lb", "covered_lb", "uncovered_lb"):
        assert isinstance(row[field], float), field
    assert set(row["health"]) == {"level", "reasons", "info", "info_detail"}

    body = _detail(client, order_id)
    for field in ("available_lb", "covered_lb", "uncovered_lb"):
        assert isinstance(body[field], float), field
    line = body["lines"][0]["readiness"]
    for field in _V21_LINE_READINESS:
        assert field in line, f"line readiness lost {field}"
    for field in ("available_lb", "covered_lb", "uncovered_lb", "coverage_runs"):
        assert field in line, field
    assert body["uncovered_lb"] == pytest.approx(300)
    assert body["covered_lb"] == pytest.approx(200)
    assert set(body["health"]["info_detail"][0]) == {
        "line_id", "sku", "product_name", "unallocated_lb"}
