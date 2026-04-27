"""
render_demand_plan.py

Reads DATABASE_URL from .env, executes demand_planning_v1.sql, loads capacity
ceilings and dependency rules from production_planning_knowledge.yaml, and
renders a three-tier HTML dashboard to demand_plan.html in the repo root.

Tiers:
  1. Capacity summary — granola_line, coconut_line, pouch_line load vs weekly
     cap, with traffic-light coloring (green <70%, amber 70-95%, red >95%).
  2. Batch production plan — one row per batch SKU (base + upstream), sorted
     by earliest downstream requested ship date.
  3. SO fulfillment detail — one row per open SO line, sorted by ship date.

Usage:
    python3 render_demand_plan.py

Env:
    DATABASE_URL  (from .env via python-dotenv)
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from datetime import datetime, date, timedelta
from decimal import Decimal
from pathlib import Path

import psycopg2
import psycopg2.extras
import yaml
from dotenv import load_dotenv
from jinja2 import Template

REPO_ROOT = Path(__file__).resolve().parent
SQL_FILE = REPO_ROOT / "demand_planning_v1.sql"
YAML_FILE = REPO_ROOT / "production_planning_knowledge.yaml"
OUT_FILE = REPO_ROOT / "demand_plan.html"

# Planning horizon: YAML describes a 14-day lead-time promise; traffic-light
# uses 7-day load vs weekly capacity per user spec.
HORIZON_7D = 7
HORIZON_14D = 14


def load_env() -> str:
    load_dotenv(REPO_ROOT / ".env")
    url = os.environ.get("DATABASE_URL")
    if not url:
        print("ERROR: DATABASE_URL not set in .env", file=sys.stderr)
        sys.exit(1)
    return url


def load_knowledge() -> dict:
    with YAML_FILE.open() as f:
        return yaml.safe_load(f)


def extract_capacity(k: dict) -> dict:
    """Pull weekly caps + discontinued SKU list from the knowledge YAML."""
    plant = k.get("plant", {}) or {}
    derived = plant.get("derived_weekly_capacity", {}) or {}

    # YAML writes caps as "~75" (string with tilde) — parse defensively.
    def _to_int(v, fallback):
        if v is None:
            return fallback
        if isinstance(v, (int, float)):
            return int(v)
        s = str(v).lstrip("~").strip()
        try:
            return int(float(s))
        except ValueError:
            return fallback

    discontinued = set()
    scope = k.get("scope", {}) or {}
    for item in scope.get("discontinued", []) or []:
        sku = item.get("sku") if isinstance(item, dict) else None
        if sku:
            discontinued.add(str(sku))

    return {
        "granola_weekly": _to_int(derived.get("granola_batches"), 75),
        "coconut_weekly": _to_int(derived.get("coconut_batches"), 57),
        "pouch_weekly_hours": float(plant.get("weekly_working_hours", 39.0) or 39.0),
        "discontinued_skus": discontinued,
    }


def run_sql(db_url: str) -> list[dict]:
    sql_text = SQL_FILE.read_text()
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql_text)
            rows = [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
    return rows


def _f(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, Decimal):
        return float(v)
    return float(v)


def _i(v) -> int:
    if v is None:
        return 0
    if isinstance(v, Decimal):
        return int(v)
    return int(v)


def filter_discontinued(rows: list[dict], discontinued: set[str]) -> tuple[list[dict], int, set[str]]:
    """Defensive post-filter: drop any row referencing a discontinued SKU.

    Returns (kept_rows, dropped_count, matched_skus). Caller surfaces a
    stderr warning if dropped_count > 0 — a non-empty result means the
    SQL-side filter (or product-active flags) missed something.
    """
    kept = []
    dropped_count = 0
    matched_skus: set[str] = set()
    for r in rows:
        codes = {str(r.get(k)) for k in ("fg_odoo", "base_batch_odoo", "upstream_batch_odoo") if r.get(k)}
        hits = codes & discontinued
        if hits:
            dropped_count += 1
            matched_skus |= hits
            continue
        kept.append(r)
    return kept, dropped_count, matched_skus


def compute_tier1(rows: list[dict], caps: dict) -> list[dict]:
    """Aggregate batches_to_run per production line, bucketed by horizon."""
    # Tier 1 aggregation — mirrors the commented summary block in demand_planning_v1.sql.
    # If the SQL summary is ever uncommented and promoted to canonical source of truth,
    # delete this function and replace with a direct query.
    # --- Begin SQL reference (copy verbatim from demand_planning_v1.sql) ---
    # /*
    # , unioned AS (
    #     SELECT base_batch_line AS production_line, base_batches_to_run AS batches_due, requested_ship_date
    #     FROM batches_needed
    #     WHERE base_batches_to_run > 0 AND base_batch_line IN ('granola_line','coconut_line')
    #     UNION ALL
    #     SELECT upstream_batch_line AS production_line, upstream_batches_to_run AS batches_due, requested_ship_date
    #     FROM batches_needed
    #     WHERE upstream_batches_to_run > 0 AND upstream_batch_line IN ('granola_line','coconut_line')
    # )
    # SELECT
    #     production_line,
    #     SUM(CASE WHEN requested_ship_date <= CURRENT_DATE + INTERVAL '7 days'  THEN batches_due ELSE 0 END) AS batches_due_7d,
    #     SUM(CASE WHEN requested_ship_date <= CURRENT_DATE + INTERVAL '14 days' THEN batches_due ELSE 0 END) AS batches_due_14d,
    #     SUM(CASE WHEN requested_ship_date >  CURRENT_DATE + INTERVAL '14 days' THEN batches_due ELSE 0 END) AS batches_due_beyond_14d,
    #     SUM(batches_due)                                                                                     AS batches_due_total,
    #     CASE production_line WHEN 'granola_line' THEN  75 WHEN 'coconut_line' THEN  57 END AS weekly_capacity,
    #     CASE production_line WHEN 'granola_line' THEN 150 WHEN 'coconut_line' THEN 114 END AS two_week_capacity,
    #     CASE
    #         WHEN production_line = 'granola_line' AND
    #              SUM(CASE WHEN requested_ship_date <= CURRENT_DATE + INTERVAL '14 days' THEN batches_due ELSE 0 END) > 150 THEN 'OVER 2-WEEK CAPACITY'
    #         WHEN production_line = 'coconut_line' AND
    #              SUM(CASE WHEN requested_ship_date <= CURRENT_DATE + INTERVAL '14 days' THEN batches_due ELSE 0 END) > 114 THEN 'OVER 2-WEEK CAPACITY'
    #         ELSE 'within capacity'
    #     END AS capacity_flag_14d
    # FROM unioned
    # GROUP BY production_line
    # ORDER BY production_line;
    # */
    # --- End SQL reference ---

    today = date.today()
    d7 = today + timedelta(days=HORIZON_7D)
    d14 = today + timedelta(days=HORIZON_14D)

    # {line: {'7d': n, '14d': n, 'beyond': n, 'total': n}}
    load = defaultdict(lambda: {"7d": 0, "14d": 0, "beyond": 0, "total": 0})

    def _bucket(line_name, batches, ship_date):
        if not line_name or not batches:
            return
        b = _i(batches)
        if b <= 0:
            return
        load[line_name]["total"] += b
        if ship_date is None:
            load[line_name]["beyond"] += b
            return
        if ship_date <= d7:
            load[line_name]["7d"] += b
        if ship_date <= d14:
            load[line_name]["14d"] += b
        if ship_date > d14:
            load[line_name]["beyond"] += b

    for r in rows:
        rsd = r.get("requested_ship_date")
        _bucket(r.get("base_batch_line"), r.get("base_batches_to_run"), rsd)
        _bucket(r.get("upstream_batch_line"), r.get("upstream_batches_to_run"), rsd)

    cap_map = {
        "granola_line": caps["granola_weekly"],
        "coconut_line": caps["coconut_weekly"],
    }

    tier1 = []
    for line in ("granola_line", "coconut_line"):
        weekly = cap_map[line]
        l = load[line]
        pct_7d = (l["7d"] / weekly * 100.0) if weekly else 0.0
        if pct_7d < 70:
            tl = "green"
        elif pct_7d <= 95:
            tl = "amber"
        else:
            tl = "red"
        tier1.append({
            "line": line,
            "weekly_cap": weekly,
            "two_week_cap": weekly * 2,
            "load_7d": l["7d"],
            "load_14d": l["14d"],
            "load_beyond": l["beyond"],
            "load_total": l["total"],
            "pct_7d": round(pct_7d, 1),
            "traffic_light": tl,
        })

    # Pouch line is hours-based in YAML; we don't have hour estimates per
    # order in the SQL yet, so surface as info row with no load calc.
    # See discrepancies.md #14.
    tier1.append({
        "line": "pouch_line",
        "weekly_cap": caps["pouch_weekly_hours"],
        "two_week_cap": None,
        "load_7d": None, "load_14d": None, "load_beyond": None, "load_total": None,
        "pct_7d": None,
        "traffic_light": "info",
    })
    return tier1


def compute_tier2(rows: list[dict]) -> list[dict]:
    """One row per (batch_odoo, production_line). Sum batches across SO lines.

    Tracks min requested_ship_date for sort and a list of downstream FGs.
    """
    # key -> aggregate
    agg = {}

    def _add(odoo, line, batches, ship_date, fg_odoo, fg_name):
        if not odoo or not batches:
            return
        b = _i(batches)
        if b <= 0:
            return
        key = (str(odoo), line or "")
        if key not in agg:
            agg[key] = {
                "batch_odoo": str(odoo),
                "production_line": line,
                "batches_to_run": 0,
                "earliest_due": None,
                "downstream_fgs": {},
            }
        a = agg[key]
        a["batches_to_run"] += b
        if ship_date is not None:
            if a["earliest_due"] is None or ship_date < a["earliest_due"]:
                a["earliest_due"] = ship_date
        if fg_odoo:
            a["downstream_fgs"].setdefault(str(fg_odoo), fg_name or "")

    for r in rows:
        _add(
            r.get("base_batch_odoo"),
            r.get("base_batch_line"),
            r.get("base_batches_to_run"),
            r.get("requested_ship_date"),
            r.get("fg_odoo"),
            r.get("fg_name"),
        )
        _add(
            r.get("upstream_batch_odoo"),
            r.get("upstream_batch_line"),
            r.get("upstream_batches_to_run"),
            r.get("requested_ship_date"),
            r.get("fg_odoo"),
            r.get("fg_name"),
        )

    out = []
    for a in agg.values():
        a["downstream_fgs_list"] = sorted(a["downstream_fgs"].keys())
        out.append(a)
    out.sort(key=lambda x: (x["earliest_due"] or date.max, x["batch_odoo"]))
    return out


# --------------------------------------------------------------------------
# Template — inline, dark industrial theme matching /dashboard/dashboard.css
# --------------------------------------------------------------------------
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<title>Demand Plan — Factory Ledger</title>
<style>
  :root {
    --bg: #0f172a; --surface: #1e293b; --surface-alt: #1a2536;
    --border: #334155; --text: #f1f5f9; --text-secondary: #cbd5e1;
    --text-muted: #94a3b8; --text-dimmed: #64748b;
    --primary: #3b82f6; --accent: #10b981;
    --badge-green-bg: #064e3b; --badge-green-text: #6ee7b7;
    --badge-amber-bg: #78350f; --badge-amber-text: #fbbf24;
    --badge-red-bg: #7f1d1d; --badge-red-text: #fca5a5;
    --badge-info-bg: #1e3a5f; --badge-info-text: #93c5fd;
    --radius: 12px; --radius-sm: 8px;
    --font: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --mono: "SF Mono", "Fira Code", Menlo, monospace;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: var(--font); background: var(--bg); color: var(--text);
    font-size: 14px; line-height: 1.5; padding: 24px;
  }
  h1 { font-size: 22px; font-weight: 700; letter-spacing: -0.3px; }
  h2 { font-size: 16px; font-weight: 600; color: var(--text-secondary);
       margin: 32px 0 12px; padding-bottom: 8px;
       border-bottom: 1px solid var(--border); }
  .meta { color: var(--text-muted); font-size: 12px; margin-top: 4px; }
  .tier {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 16px; margin-bottom: 16px;
  }
  .capacity-grid {
    display: grid; gap: 12px;
    grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  }
  .cap-card {
    background: var(--surface-alt); border: 1px solid var(--border);
    border-radius: var(--radius-sm); padding: 14px;
    border-left: 4px solid var(--border);
  }
  .cap-card.green { border-left-color: #10b981; }
  .cap-card.amber { border-left-color: #f59e0b; }
  .cap-card.red   { border-left-color: #ef4444; }
  .cap-card.info  { border-left-color: #3b82f6; }
  .cap-title { font-weight: 600; font-size: 13px; color: var(--text-secondary);
               text-transform: uppercase; letter-spacing: 0.5px; }
  .cap-metric { font-size: 24px; font-weight: 700; margin-top: 6px;
                font-family: var(--mono); }
  .cap-sub { font-size: 12px; color: var(--text-muted); margin-top: 4px; }
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 0.3px;
  }
  .badge.green { background: var(--badge-green-bg); color: var(--badge-green-text); }
  .badge.amber { background: var(--badge-amber-bg); color: var(--badge-amber-text); }
  .badge.red   { background: var(--badge-red-bg);   color: var(--badge-red-text); }
  .badge.info  { background: var(--badge-info-bg);  color: var(--badge-info-text); }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 8px 10px; text-align: left;
           border-bottom: 1px solid var(--border); }
  th { background: var(--surface-alt); color: var(--text-secondary);
       font-size: 11px; text-transform: uppercase; letter-spacing: 0.4px;
       font-weight: 600; position: sticky; top: 0; }
  tbody tr:hover { background: var(--surface-alt); }
  td.mono, th.mono { font-family: var(--mono); }
  td.right, th.right { text-align: right; }
  .empty { color: var(--text-dimmed); font-style: italic; padding: 12px; }
  .footer { margin-top: 24px; font-size: 11px; color: var(--text-dimmed); }
</style>
</head>
<body>

<h1>Demand Plan</h1>
<div class="meta">
  Generated {{ generated_at }} ·
  {{ tier3|length }} open SO line(s) ·
  source: <span class="mono">demand_planning_v1.sql</span>
</div>

<h2>Tier 1 — Production Capacity (14-day horizon; color from 7-day load)</h2>
<div class="tier">
  <div class="capacity-grid">
    {% for c in tier1 %}
    <div class="cap-card {{ c.traffic_light }}">
      <div class="cap-title">
        {{ c.line }}
        <span class="badge {{ c.traffic_light }}">
          {% if c.traffic_light == 'info' %}info{% else %}{{ c.pct_7d }}%{% endif %}
        </span>
      </div>
      {% if c.load_14d is not none %}
        <div class="cap-metric">{{ c.load_14d }} / {{ c.two_week_cap }}</div>
        <div class="cap-sub">
          batches due in 14d · 2-wk cap {{ c.two_week_cap }} ·
          7d load {{ c.load_7d }} (color basis, weekly cap {{ c.weekly_cap }}) ·
          beyond {{ c.load_beyond }}
        </div>
      {% else %}
        <div class="cap-metric">{{ c.weekly_cap }} hrs</div>
        <div class="cap-sub">weekly pouch line — no per-SO hour estimate yet (discrepancies.md #14)</div>
      {% endif %}
    </div>
    {% endfor %}
  </div>
</div>

<h2>Tier 2 — Batch Production Plan</h2>
<div class="tier">
  {% if tier2 %}
  <table>
    <thead>
      <tr>
        <th>Batch SKU</th>
        <th>Line</th>
        <th class="right">Batches to Run</th>
        <th>Earliest Due</th>
        <th>Downstream FGs</th>
      </tr>
    </thead>
    <tbody>
    {% for b in tier2 %}
      <tr>
        <td class="mono">{{ b.batch_odoo }}</td>
        <td>{{ b.production_line or '—' }}</td>
        <td class="right mono">{{ b.batches_to_run }}</td>
        <td class="mono">{{ b.earliest_due.isoformat() if b.earliest_due else '—' }}</td>
        <td class="mono">{{ ', '.join(b.downstream_fgs_list) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% else %}
  <div class="empty">No batches currently needed — all open SOs shippable from stock.</div>
  {% endif %}
</div>

<h2>Tier 3 — SO Fulfillment Detail</h2>
<div class="tier">
  {% if tier3 %}
  <table>
    <thead>
      <tr>
        <th>Due</th>
        <th>Order</th>
        <th>Status</th>
        <th>FG</th>
        <th>FG Name</th>
        <th class="right">Remaining lb</th>
        <th class="right">On-Hand lb</th>
        <th class="right">Shortfall lb</th>
        <th>Base Batch</th>
        <th class="right">Batches</th>
        <th>Upstream Batch</th>
        <th class="right">U. Batches</th>
        <th>Plan Note</th>
      </tr>
    </thead>
    <tbody>
    {% for r in tier3 %}
      <tr>
        <td class="mono">{{ r.requested_ship_date.isoformat() if r.requested_ship_date else '—' }}</td>
        <td class="mono">{{ r.order_number or '—' }}</td>
        <td>{{ r.so_status or '—' }}</td>
        <td class="mono">{{ r.fg_odoo or '—' }}</td>
        <td>{{ r.fg_name or '—' }}</td>
        <td class="right mono">{{ r.fg_remaining_lb if r.fg_remaining_lb is not none else '—' }}</td>
        <td class="right mono">{{ r.fg_on_hand_lb if r.fg_on_hand_lb is not none else '—' }}</td>
        <td class="right mono">{{ r.fg_shortfall_lb if r.fg_shortfall_lb is not none else '—' }}</td>
        <td class="mono">{{ r.base_batch_odoo or '—' }}</td>
        <td class="right mono">{{ r.base_batches_to_run if r.base_batches_to_run is not none else '—' }}</td>
        <td class="mono">{{ r.upstream_batch_odoo or '—' }}</td>
        <td class="right mono">{{ r.upstream_batches_to_run if r.upstream_batches_to_run is not none else '—' }}</td>
        <td>{{ r.plan_note or '—' }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  {% else %}
  <div class="empty">No open sales-order lines matched the planning query.</div>
  {% endif %}
</div>

<div class="footer">
  Capacity ceilings from production_planning_knowledge.yaml ·
  Discontinued-SKU filter applied defensively.
</div>

</body>
</html>
"""


def render(tier1, tier2, tier3) -> str:
    tmpl = Template(TEMPLATE)
    return tmpl.render(
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z").strip(),
        tier1=tier1,
        tier2=tier2,
        tier3=tier3,
    )


def main() -> None:
    db_url = load_env()
    knowledge = load_knowledge()
    caps = extract_capacity(knowledge)

    rows = run_sql(db_url)
    rows, dropped_count, dropped_skus = filter_discontinued(rows, caps["discontinued_skus"])
    if dropped_count > 0:
        print(
            f"WARNING: Python-side filter dropped {dropped_count} rows for "
            f"discontinued SKUs {sorted(dropped_skus)}. SQL-side filter may "
            f"be missing a case — consider tightening demand_planning_v1.sql.",
            file=sys.stderr,
        )

    tier1 = compute_tier1(rows, caps)
    tier2 = compute_tier2(rows)
    tier3 = rows

    html = render(tier1, tier2, tier3)
    OUT_FILE.write_text(html)

    granola_total = next((t["load_total"] for t in tier1 if t["line"] == "granola_line"), 0) or 0
    coconut_total = next((t["load_total"] for t in tier1 if t["line"] == "coconut_line"), 0) or 0
    print(
        f"Rendered {granola_total} granola batches, "
        f"{coconut_total} coconut batches, "
        f"{len(tier3)} open SO lines -> {OUT_FILE.name}"
    )


if __name__ == "__main__":
    main()
