"""
render_planner_v2.py

Three-week rolling production planner. Reads DATABASE_URL from .env, runs
demand_planning_v1.sql, loads production_planning_knowledge.yaml, and writes
planner_v2.html to the repo root.

Output is a static, self-contained HTML report — no JS, no CDN. Three week
cards (granola + coconut batch totals, batch breakdown, finished-goods
case counts) followed by a single sales-orders panel covering every open
SO line that falls inside the 3-week horizon (or is past-due).

Usage:
    python3 render_planner_v2.py
"""
from __future__ import annotations

import math
import os
import re
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
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
OUT_FILE = REPO_ROOT / "planner_v2.html"


# ---------------------------------------------------------------------------
# Helpers copied verbatim from render_demand_plan.py
# ---------------------------------------------------------------------------
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


def filter_discontinued(rows, discontinued):
    kept = []
    dropped_count = 0
    matched_skus = set()
    for r in rows:
        codes = {str(r.get(k)) for k in ("fg_odoo", "base_batch_odoo", "upstream_batch_odoo") if r.get(k)}
        hits = codes & discontinued
        if hits:
            dropped_count += 1
            matched_skus |= hits
            continue
        kept.append(r)
    return kept, dropped_count, matched_skus


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------
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


def discontinued_set(knowledge: dict) -> set[str]:
    out: set[str] = set()
    scope = knowledge.get("scope", {}) or {}
    for item in scope.get("discontinued", []) or []:
        sku = item.get("sku") if isinstance(item, dict) else None
        if sku:
            out.add(str(sku))
    return out


def case_size_map(knowledge: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for fg in knowledge.get("finished_goods", []) or []:
        odoo = fg.get("odoo")
        cs = fg.get("case_size_lb")
        if odoo is None or cs is None:
            continue
        try:
            out[str(odoo)] = float(cs)
        except (TypeError, ValueError):
            continue
    return out


def batch_name_map(knowledge: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for b in knowledge.get("batches", []) or []:
        sku = b.get("sku")
        nm = b.get("name")
        if sku and nm:
            out[str(sku)] = str(nm)
    return out


def batch_target_lb_map(knowledge: dict) -> dict[str, float]:
    """{batch_sku: target_batch_lb} from YAML's owner-confirmed values.
    YAML's `target_batch_lb` is the canonical recipe-sum / owner-confirmed
    figure. The DB's products.default_batch_lb is compared against this at
    startup; mismatches log to stderr."""
    out: dict[str, float] = {}
    for b in knowledge.get("batches", []) or []:
        sku = b.get("sku")
        tgt = b.get("target_batch_lb")
        if sku is None or tgt is None:
            continue
        try:
            out[str(sku)] = float(tgt)
        except (TypeError, ValueError):
            continue
    return out


def fetch_db_batch_targets(db_url: str) -> dict[str, float]:
    """Pull products.default_batch_lb for type='batch' so we can warn on
    YAML-vs-DB drift."""
    sql = """
        SELECT odoo_code, default_batch_lb
        FROM products
        WHERE type = 'batch' AND default_batch_lb IS NOT NULL
    """
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return {str(r["odoo_code"]): float(r["default_batch_lb"]) for r in cur.fetchall()}
    finally:
        conn.close()


def upstream_rule_map(knowledge: dict) -> dict[str, dict]:
    """Derive per-base-batch upstream rules from YAML upstream_dependencies.

    Two rule types are recognized:
      * per_batch — qty_per_unit upstream batches per 1 base batch (95005 case)
      * per_case  — qty_per_unit lb of upstream per FG case (90008 case)
    Any other shape is logged to stderr and skipped."""
    out: dict[str, dict] = {}
    targets = batch_target_lb_map(knowledge)
    for b in knowledge.get("batches", []) or []:
        sku = str(b.get("sku") or "")
        if not sku:
            continue
        for dep in b.get("upstream_dependencies", []) or []:
            up_sku = dep.get("batch_odoo")
            if not up_sku:
                continue  # ingredient dep, not a batch dep
            up_sku = str(up_sku)
            qty = dep.get("qty_lb_per_batch")
            try:
                qty = float(qty) if qty is not None else 0.0
            except (TypeError, ValueError):
                qty = 0.0
            up_target = targets.get(up_sku)
            if not up_target or up_target <= 0:
                print(
                    f"WARN: upstream {up_sku} missing target_batch_lb in YAML; "
                    f"skipping rule for base {sku}",
                    file=sys.stderr,
                )
                continue
            if sku == "95005":
                rule = {
                    "upstream_sku": up_sku,
                    "upstream_target_lb": up_target,
                    "rule_type": "per_batch",
                    "qty_per_unit": 1,
                }
            elif sku == "90008":
                rule = {
                    "upstream_sku": up_sku,
                    "upstream_target_lb": up_target,
                    "rule_type": "per_case",
                    "qty_per_unit": qty,  # lb of upstream per FG case
                }
            else:
                print(
                    f"WARN: unknown upstream rule shape for base {sku} → {up_sku}; skipping",
                    file=sys.stderr,
                )
                continue
            out[sku] = rule
    return out


def gross_base_batches(fg_remaining_lb: float, base_batch_target_lb: float | None) -> int:
    """Gross base batches needed to fulfill fg_remaining_lb, ignoring on-hand
    inventory. Returns 0 for REPACK FGs (target is None) or non-positive demand."""
    if base_batch_target_lb is None or base_batch_target_lb <= 0:
        return 0
    if fg_remaining_lb is None or fg_remaining_lb <= 0:
        return 0
    return math.ceil(float(fg_remaining_lb) / float(base_batch_target_lb))


def gross_upstream_batches(
    base_batch_count: int,
    fg_remaining_lb: float,
    case_size_lb: float | None,
    upstream_rule: dict | None,
) -> int:
    """Gross upstream batches for a row, given the rule payload from
    upstream_rule_map(). Returns 0 when no rule applies."""
    if not upstream_rule:
        return 0
    rule_type = upstream_rule.get("rule_type")
    qty_per_unit = float(upstream_rule.get("qty_per_unit") or 0)
    up_target = float(upstream_rule.get("upstream_target_lb") or 0)
    if qty_per_unit <= 0:
        return 0
    if rule_type == "per_batch":
        return int(base_batch_count * qty_per_unit)
    if rule_type == "per_case":
        if case_size_lb is None or case_size_lb <= 0 or up_target <= 0:
            return 0
        if fg_remaining_lb is None or fg_remaining_lb <= 0:
            return 0
        cases = math.ceil(float(fg_remaining_lb) / float(case_size_lb))
        upstream_lb = cases * qty_per_unit
        return int(math.ceil(upstream_lb / up_target))
    return 0


# Sanity checks for the gross-requirement math.
assert gross_base_batches(625, 323) == 2,  "625/323 → 2 batches"
assert gross_base_batches(323, 323) == 1,  "exactly one batch"
assert gross_base_batches(324, 323) == 2,  "one over → two batches"
assert gross_base_batches(0, 323)   == 0,  "zero demand → zero batches"
assert gross_base_batches(100, None) == 0, "REPACK (no base) → zero batches"

_FRUIT_NUT_RULE_TEST = {
    "upstream_sku": "90002", "upstream_target_lb": 323,
    "rule_type": "per_case", "qty_per_unit": 20,
}
assert gross_upstream_batches(2, 600, 25, _FRUIT_NUT_RULE_TEST) == 2, \
    "Fruit Nut: 24 cases × 20 lb / 323 → 2 upstream batches"

_PB_BANANA_RULE_TEST = {
    "upstream_sku": "95002", "upstream_target_lb": 350,
    "rule_type": "per_batch", "qty_per_unit": 1,
}
assert gross_upstream_batches(4, 1746, 7, _PB_BANANA_RULE_TEST) == 4, \
    "BS PB Banana: 4 base × 1 → 4 upstream batches"


def fetch_customer_map(db_url: str) -> dict[str, str]:
    """Map order_number → customer name. Run once after the planning query."""
    sql = """
        SELECT so.order_number, COALESCE(c.name, '') AS customer_name
        FROM sales_orders so
        LEFT JOIN customers c ON c.id = so.customer_id
        WHERE so.status NOT IN ('shipped', 'cancelled')
    """
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql)
            return {r["order_number"]: r["customer_name"] for r in cur.fetchall()}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Working-week math
# ---------------------------------------------------------------------------
WEEKDAY_LETTERS = {0: "M", 1: "T", 2: "W", 3: "Th", 4: "F"}


def next_tuesday(today: date) -> date:
    """Tuesday of the upcoming work-week. If today is a Tuesday, returns today."""
    delta = (1 - today.weekday()) % 7  # weekday: Mon=0, Tue=1
    return today + timedelta(days=delta)


def working_days_for_week(start_tue: date) -> list[dict]:
    """Tue + Wed + Thu + Fri + Mon-of-next-week — 5 calendar slots."""
    days = []
    for offset in (0, 1, 2, 3, 6):
        d = start_tue + timedelta(days=offset)
        days.append(
            {
                "day_num": d.day,
                "letter": WEEKDAY_LETTERS[d.weekday()],
                "date": d.isoformat(),
            }
        )
    return days


def build_weeks(today: date) -> list[dict]:
    """Return three week dicts with start, end, working_days, and an empty payload."""
    w1_start = next_tuesday(today)
    weeks = []
    for i in range(3):
        start = w1_start + timedelta(days=7 * i)
        end = start + timedelta(days=6)  # Tue → Mon inclusive
        weeks.append(
            {
                "week_number": i + 1,
                "start": start,
                "end": end,
                "working_days": working_days_for_week(start),
            }
        )
    return weeks


def working_week_tuesday(d: date) -> date:
    """Tuesday of d's Tue→Mon work-week. Sat/Sun map to the work-week that contains them."""
    return d - timedelta(days=(d.weekday() - 1) % 7)


# Boundary-case assertions — guard against future "improvements" of the formula.
# Use 2026-05-12 (Tue) → 2026-05-18 (Mon) as the reference work-week.
assert working_week_tuesday(date(2026, 5, 18)) == date(2026, 5, 12), "Mon → prior Tue"
assert working_week_tuesday(date(2026, 5, 12)) == date(2026, 5, 12), "Tue → same Tue"
assert working_week_tuesday(date(2026, 5, 13)) == date(2026, 5, 12), "Wed → Tue"
assert working_week_tuesday(date(2026, 5, 15)) == date(2026, 5, 12), "Fri → Tue"
assert working_week_tuesday(date(2026, 5, 16)) == date(2026, 5, 12), "Sat → prior Tue"
assert working_week_tuesday(date(2026, 5, 17)) == date(2026, 5, 12), "Sun → prior Tue"


def _format_date_range_label(min_d: date, max_d: date) -> str:
    """Render the Beyond card subtitle.

    Rules (tested below):
      empty                              → "—"
      single date                        → "May 19"
      same year, same month              → "May 19 – 27"
      same year, different months        → "May 19 – Jun 24"
      different years (year crossing)    → "Dec 30 – Jan 12, 2027"
    """
    if min_d == max_d:
        return min_d.strftime("%b %-d")
    if min_d.year == max_d.year:
        if min_d.month == max_d.month:
            return f"{min_d.strftime('%b %-d')} – {max_d.day}"
        return f"{min_d.strftime('%b %-d')} – {max_d.strftime('%b %-d')}"
    return f"{min_d.strftime('%b %-d')} – {max_d.strftime('%b %-d, %Y')}"


assert _format_date_range_label(date(2026, 5, 19), date(2026, 5, 19)) == "May 19", "single date"
assert _format_date_range_label(date(2026, 5, 19), date(2026, 5, 27)) == "May 19 – 27", "same month"
assert _format_date_range_label(date(2026, 5, 19), date(2026, 6, 24)) == "May 19 – Jun 24", "diff month"
assert _format_date_range_label(date(2026, 12, 30), date(2027, 1, 12)) == "Dec 30 – Jan 12, 2027", "year cross"


# ---------------------------------------------------------------------------
# Name cleanup
# ---------------------------------------------------------------------------
_BATCH_TRAILING_RE = re.compile(
    r"\s*("
    r"Granola\s+\d+(?:\s*lb)?|"   # "Granola 350", "Granola 380 lb"
    r"Granola|"                    # bare trailing "Granola"
    r"#\d+|"                       # "#9"
    r"\d+\s*lb"                    # "380 lb"
    r")\s*$",
    re.IGNORECASE,
)


def clean_batch_name(raw: str) -> str:
    if not raw:
        return ""
    s = str(raw).strip()
    if s.lower().startswith("batch "):
        s = s[6:].strip()
    for _ in range(3):
        s2 = _BATCH_TRAILING_RE.sub("", s).strip()
        if s2 == s:
            break
        s = s2
    if s.lower().startswith("coconut "):
        s = s[8:].strip()
    return s


def clean_fg_name(raw: str) -> str:
    if not raw:
        return ""
    s = str(raw).strip()
    s = re.sub(r"^\d{3,6}\s+", "", s)        # strip leading SKU code if present
    s = re.sub(r"^(Granola|Coconut)\s+", "", s, flags=re.IGNORECASE)
    return s.strip()


# ---------------------------------------------------------------------------
# Bucketing + aggregation
# ---------------------------------------------------------------------------
GRANOLA_LIKE_LINES = {"granola_line", "cold_mix_at_pack"}
COCONUT_LINES = {"coconut_line"}


def _coerce_date(v):
    if v is None:
        return None
    if isinstance(v, date) and not isinstance(v, datetime):
        return v
    if isinstance(v, datetime):
        return v.date()
    try:
        return datetime.fromisoformat(str(v)).date()
    except (TypeError, ValueError):
        return None


def bucket_rows_into_weeks(
    rows: list[dict], weeks: list[dict]
) -> tuple[list[list[dict]], list[dict]]:
    """Return ([week1_rows, week2_rows, week3_rows], beyond_rows).

    Past-due rolls into Week 1. Anything past Week 3 lands in the beyond
    bucket — no upper cap. Bad/missing dates log to stderr."""
    out: list[list[dict]] = [[] for _ in weeks]
    beyond: list[dict] = []
    for r in rows:
        rsd = _coerce_date(r.get("requested_ship_date"))
        if rsd is None:
            print(
                f"WARN: dropping row with unparseable ship date "
                f"(order={r.get('order_number')}, fg={r.get('fg_odoo')})",
                file=sys.stderr,
            )
            continue
        if rsd <= weeks[0]["end"]:
            out[0].append(r)
            continue
        if rsd <= weeks[1]["end"]:
            out[1].append(r)
            continue
        if rsd <= weeks[2]["end"]:
            out[2].append(r)
            continue
        beyond.append(r)
    return out, beyond


def aggregate_beyond(
    beyond_rows: list[dict],
    case_sizes: dict[str, float],
    batch_names: dict[str, str],
    batch_targets: dict[str, float],
    upstream_rules: dict[str, dict],
) -> dict:
    """Compute the Beyond Week 3 card payload using gross batch requirements.

    Groups rows by working-week Tuesday. Sums batch counts (base + upstream,
    both lines combined). Picks the dominant batch family per group
    (highest single contribution; ties broken case-insensitively).

    Returns: {granola_total, coconut_total, date_range_label, next_big_weeks}.
    """
    granola_total = 0
    coconut_total = 0
    by_week: dict[date, dict] = {}
    min_d: date | None = None
    max_d: date | None = None

    def _record(odoo, fallback_name, count, line, week_tue):
        nonlocal granola_total, coconut_total
        n = _i(count)
        if n <= 0:
            return
        sku = str(odoo) if odoo else ""
        raw = batch_names.get(sku) or fallback_name or sku
        cleaned = clean_batch_name(raw)
        if not cleaned:
            return
        if line in COCONUT_LINES:
            coconut_total += n
        else:
            granola_total += n
        bucket = by_week.setdefault(week_tue, {"total": 0, "by_name": defaultdict(int)})
        bucket["total"] += n
        bucket["by_name"][cleaned] += n

    for r in beyond_rows:
        rsd = _coerce_date(r.get("requested_ship_date"))
        if rsd is None:
            continue
        if min_d is None or rsd < min_d:
            min_d = rsd
        if max_d is None or rsd > max_d:
            max_d = rsd
        wtue = working_week_tuesday(rsd)

        fg_odoo = str(r.get("fg_odoo") or "")
        rem = _f(r.get("fg_remaining_lb"))
        base_sku = str(r.get("base_batch_odoo") or "")
        base_target = batch_targets.get(base_sku) if base_sku else None

        gross_base = gross_base_batches(rem, base_target)
        if gross_base > 0:
            _record(base_sku, r.get("base_batch_name"), gross_base, r.get("base_batch_line"), wtue)

        rule = upstream_rules.get(base_sku) if base_sku else None
        if rule and gross_base > 0:
            cs_for_up = case_sizes.get(fg_odoo)
            gross_up = gross_upstream_batches(gross_base, rem, cs_for_up, rule)
            if gross_up > 0:
                up_line = r.get("upstream_batch_line") or "granola_line"
                _record(rule["upstream_sku"], None, gross_up, up_line, wtue)

    if min_d is None or max_d is None:
        date_range_label = "—"
    else:
        date_range_label = _format_date_range_label(min_d, max_d)

    # Build the per-week payload. Skip weeks that contributed zero counted
    # batches (rows existed but base_batches_to_run / upstream were 0 — those
    # are stock-covered and don't need a callout).
    week_payloads = []
    for wtue, info in by_week.items():
        if info["total"] <= 0:
            continue
        # Descending count, ascending case-insensitive name as tiebreaker.
        dominant_name = sorted(
            info["by_name"].items(),
            key=lambda kv: (-kv[1], kv[0].lower()),
        )[0][0]
        week_payloads.append(
            {
                "week_tuesday": wtue,
                "week_tuesday_label": wtue.strftime("%b %-d"),
                "total_batches": info["total"],
                "dominant_batch_name": dominant_name,
            }
        )

    week_payloads.sort(key=lambda x: (-x["total_batches"], x["week_tuesday"]))
    next_big_weeks = week_payloads[:3]

    return {
        "granola_total": granola_total,
        "coconut_total": coconut_total,
        "date_range_label": date_range_label,
        "next_big_weeks": next_big_weeks,
    }


def aggregate_week(
    week_rows: list[dict],
    case_sizes: dict[str, float],
    batch_names: dict[str, str],
    batch_targets: dict[str, float],
    upstream_rules: dict[str, dict],
) -> dict:
    """Build batches list, finished-goods list, and totals for one week.
    Batch counts are gross requirements (ignore inventory) computed from
    fg_remaining_lb against YAML target_batch_lb."""
    batch_agg: dict[tuple, dict] = {}
    fg_agg: dict[str, int] = defaultdict(int)

    def _add_batch(odoo, fallback_name, count, line):
        n = _i(count)
        if n <= 0:
            return
        sku = str(odoo) if odoo else ""
        raw_name = batch_names.get(sku) or fallback_name or sku
        cleaned = clean_batch_name(raw_name)
        if not cleaned:
            return
        bucket_line = "coconut" if line in COCONUT_LINES else "granola"
        key = (cleaned, bucket_line)
        if key not in batch_agg:
            batch_agg[key] = {"name": cleaned, "count": 0, "line": bucket_line}
        batch_agg[key]["count"] += n

    for r in week_rows:
        fg_odoo = str(r.get("fg_odoo") or "")
        rem = _f(r.get("fg_remaining_lb"))
        base_sku = str(r.get("base_batch_odoo") or "")
        base_target = batch_targets.get(base_sku) if base_sku else None

        gross_base = gross_base_batches(rem, base_target)
        if gross_base > 0:
            _add_batch(base_sku, r.get("base_batch_name"), gross_base, r.get("base_batch_line"))

        rule = upstream_rules.get(base_sku) if base_sku else None
        if rule and gross_base > 0:
            cs_for_up = case_sizes.get(fg_odoo)
            gross_up = gross_upstream_batches(gross_base, rem, cs_for_up, rule)
            if gross_up > 0:
                up_line = r.get("upstream_batch_line") or "granola_line"
                _add_batch(rule["upstream_sku"], None, gross_up, up_line)

        cs = case_sizes.get(fg_odoo)
        if cs and cs > 0 and rem > 0:
            cases = int(math.ceil(rem / cs))
            fg_name = clean_fg_name(r.get("fg_name"))
            if fg_name:
                fg_agg[fg_name] += cases

    granola_total = sum(b["count"] for b in batch_agg.values() if b["line"] == "granola")
    coconut_total = sum(b["count"] for b in batch_agg.values() if b["line"] == "coconut")

    batches_sorted = sorted(
        batch_agg.values(),
        key=lambda x: (0 if x["line"] == "granola" else 1, -x["count"], x["name"]),
    )
    fgs_sorted = sorted(
        ({"name": k, "cases": v} for k, v in fg_agg.items()),
        key=lambda x: (-x["cases"], x["name"]),
    )

    return {
        "granola_total": granola_total,
        "coconut_total": coconut_total,
        "batches": batches_sorted,
        "finished_goods": fgs_sorted,
    }


# ---------------------------------------------------------------------------
# SO panel
# ---------------------------------------------------------------------------
def build_so_rows(
    rows: list[dict],
    weeks: list[dict],  # noqa: ARG001 — kept for signature stability; no longer filters
    customer_map: dict[str, str],
    today: date,
) -> list[dict]:
    """Group SQL rows by order_number → one display row per SO. No horizon
    cap — every open SO appears, sorted past-due first then ascending ship date."""
    groups: dict[str, dict] = {}
    for r in rows:
        rsd = _coerce_date(r.get("requested_ship_date"))
        if rsd is None:
            continue  # unparseable — already warned during bucketing
        on = r.get("order_number") or ""
        g = groups.get(on)
        if g is None:
            g = {
                "order_number": on,
                "customer": customer_map.get(on, ""),
                "ship_date": rsd,
                "items": [],
                "total_lb": 0.0,
            }
            groups[on] = g
        else:
            # SO ship date is per-order in the source, but be defensive.
            if rsd < g["ship_date"]:
                g["ship_date"] = rsd
        g["items"].append(
            {
                "name": clean_fg_name(r.get("fg_name")),
                "lb": _f(r.get("fg_remaining_lb")),
            }
        )
        g["total_lb"] += _f(r.get("fg_remaining_lb"))

    out = []
    for g in groups.values():
        days_late = (today - g["ship_date"]).days
        is_past_due = days_late > 0
        items = g["items"]
        if len(items) == 1:
            items_text = items[0]["name"]
        elif len(items) == 2:
            items_text = " · ".join(i["name"] for i in items)
        else:
            items_text = items[0]["name"] + f" · +{len(items) - 1} more"
        out.append(
            {
                "order_number": g["order_number"],
                "customer": g["customer"],
                "ship_date": g["ship_date"],
                "ship_date_label": g["ship_date"].strftime("%b %-d"),
                "is_past_due": is_past_due,
                "days_late": days_late if is_past_due else 0,
                "items_text": items_text,
                "total_lb": round(g["total_lb"], 0),
            }
        )

    out.sort(key=lambda r: (0 if r["is_past_due"] else 1, r["ship_date"], r["order_number"]))
    return out


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------
TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Production planner — Factory Ledger</title>
<style>
  :root {
    --color-bg: #f8fafc;
    --color-surface: #ffffff;
    --color-border: rgba(15, 23, 42, 0.08);
    --color-divider: rgba(15, 23, 42, 0.06);
    --color-text-primary: #0f172a;
    --color-text-secondary: #475569;
    --color-text-tertiary: #94a3b8;
    --color-row-alt: #f8fafc;

    --granola-bg: #E6F1FB;
    --granola-label: #185FA5;
    --granola-number: #0C447C;
    --granola-bg-muted: #EFF5FB;
    --granola-label-muted: #3D7AB5;
    --granola-number-muted: #1B5489;

    --coconut-bg: #E1F5EE;
    --coconut-label: #0F6E56;
    --coconut-number: #085041;
    --coconut-bg-muted: #ECF5F0;
    --coconut-label-muted: #3F8169;
    --coconut-number-muted: #1A6850;

    --pastdue-bg: #FCEBEB;
    --pastdue-border: #E24B4A;
    --pastdue-text: #791F1F;
  }

  @media (prefers-color-scheme: dark) {
    :root {
      --color-bg: #0f172a;
      --color-surface: #1e293b;
      --color-border: rgba(241, 245, 249, 0.10);
      --color-divider: rgba(241, 245, 249, 0.08);
      --color-text-primary: #f1f5f9;
      --color-text-secondary: #cbd5e1;
      --color-text-tertiary: #94a3b8;
      --color-row-alt: #1a2536;

      --granola-bg: #16334F;
      --granola-label: #93C5FD;
      --granola-number: #DBEAFE;
      --granola-bg-muted: #112942;
      --granola-label-muted: #6EA0CC;
      --granola-number-muted: #B5CFEB;

      --coconut-bg: #14443A;
      --coconut-label: #6EE7B7;
      --coconut-number: #D1FAE5;
      --coconut-bg-muted: #0F362F;
      --coconut-label-muted: #4FB494;
      --coconut-number-muted: #A8DEC4;

      --pastdue-bg: #4A1818;
      --pastdue-border: #E24B4A;
      --pastdue-text: #FCA5A5;
    }
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  html, body {
    background: var(--color-bg);
    color: var(--color-text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    font-size: 14px;
    line-height: 1.5;
    font-weight: 400;
  }

  body { padding: 28px 32px; }

  .page-header { margin-bottom: 20px; }
  .page-title {
    font-size: 22px;
    font-weight: 500;
    letter-spacing: -0.2px;
  }
  .page-sub {
    margin-top: 4px;
    font-size: 13px;
    color: var(--color-text-secondary);
  }

  .week-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
  }

  .beyond-subtitle {
    font-size: 11px;
    color: var(--color-text-secondary);
    font-variant-numeric: tabular-nums;
  }

  .week-card {
    background: var(--color-surface);
    border: 0.5px solid var(--color-border);
    border-radius: 12px;
    padding: 14px;
  }

  .week-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 12px;
  }
  .week-label {
    font-size: 13px;
    font-weight: 500;
    color: var(--color-text-secondary);
  }
  .working-days {
    display: grid;
    grid-template-columns: repeat(5, minmax(22px, auto));
    gap: 6px;
    text-align: center;
    font-variant-numeric: tabular-nums;
  }
  .wd-num {
    font-size: 13px;
    font-weight: 500;
    color: var(--color-text-primary);
    line-height: 1.2;
  }
  .wd-letter {
    font-size: 10px;
    color: var(--color-text-tertiary);
    line-height: 1.2;
  }

  .metrics {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
    margin-bottom: 14px;
  }
  .metric {
    border-radius: 10px;
    padding: 10px 12px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .metric.granola { background: var(--granola-bg); }
  .metric.coconut { background: var(--coconut-bg); }
  .metric.granola-muted { background: var(--granola-bg-muted); }
  .metric.coconut-muted { background: var(--coconut-bg-muted); }
  .metric-label {
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.2px;
  }
  .metric.granola .metric-label { color: var(--granola-label); }
  .metric.coconut .metric-label { color: var(--coconut-label); }
  .metric.granola-muted .metric-label { color: var(--granola-label-muted); }
  .metric.coconut-muted .metric-label { color: var(--coconut-label-muted); }
  .metric-number {
    font-size: 30px;
    font-weight: 500;
    line-height: 1.1;
    font-variant-numeric: tabular-nums;
  }
  .metric.granola .metric-number { color: var(--granola-number); }
  .metric.coconut .metric-number { color: var(--coconut-number); }
  .metric.granola-muted .metric-number { color: var(--granola-number-muted); }
  .metric.coconut-muted .metric-number { color: var(--coconut-number-muted); }
  .metric-sub {
    font-size: 11px;
    font-weight: 400;
  }
  .metric.granola .metric-sub { color: var(--granola-label); }
  .metric.coconut .metric-sub { color: var(--coconut-label); }
  .metric.granola-muted .metric-sub { color: var(--granola-label-muted); }
  .metric.coconut-muted .metric-sub { color: var(--coconut-label-muted); }

  .next-big-weeks {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }
  .next-week-row {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 8px;
    font-size: 12px;
    line-height: 1.5;
  }
  .next-week-date {
    color: var(--color-text-primary);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }
  .next-week-stack {
    text-align: right;
    min-width: 0;
  }
  .next-week-count {
    color: var(--color-text-primary);
    font-variant-numeric: tabular-nums;
  }
  .next-week-family {
    font-size: 11px;
    color: var(--color-text-secondary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .section-label {
    text-transform: uppercase;
    letter-spacing: 0.6px;
    font-size: 11px;
    font-weight: 500;
    color: var(--color-text-tertiary);
    margin: 4px 0 6px;
  }

  .item-list {
    font-size: 12px;
    line-height: 1.6;
    color: var(--color-text-primary);
  }
  .item-list .row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
  }
  .item-list .row .name {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .item-list .row .val {
    font-variant-numeric: tabular-nums;
    color: var(--color-text-secondary);
  }
  .item-list .divider {
    border-top: 0.5px solid var(--color-divider);
    margin-top: 4px;
    padding-top: 4px;
  }
  .item-list .more {
    color: var(--color-text-tertiary);
    font-size: 11px;
    margin-top: 2px;
  }
  .item-list .empty {
    color: var(--color-text-tertiary);
    font-size: 11px;
    font-style: italic;
  }

  .week-section + .week-section { margin-top: 12px; }

  .so-panel {
    margin-top: 1.5rem;
    background: var(--color-surface);
    border: 0.5px solid var(--color-border);
    border-radius: 12px;
    overflow: hidden;
  }
  .so-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 16px;
    border-bottom: 0.5px solid var(--color-border);
  }
  .so-title { font-size: 13px; font-weight: 500; color: var(--color-text-primary); }
  .so-count { font-size: 12px; color: var(--color-text-secondary); }

  table.so-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  table.so-table th, table.so-table td {
    text-align: left;
    padding: 8px 12px;
    font-variant-numeric: tabular-nums;
  }
  table.so-table thead th {
    background: var(--color-row-alt);
    color: var(--color-text-secondary);
    font-weight: 500;
    font-size: 11px;
    letter-spacing: 0.3px;
  }
  table.so-table tbody tr {
    border-top: 0.5px solid var(--color-divider);
  }
  table.so-table tbody tr:nth-child(even) {
    background: #F1F5F9;
  }
  table.so-table td.right, table.so-table th.right {
    text-align: right;
  }
  table.so-table td.ship  { width: 80px; }
  table.so-table td.order { width: 110px; }
  table.so-table td.cust  { width: 130px; }
  table.so-table td.tot   { width: 70px; }
  table.so-table td.items {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 0;
  }

  table.so-table tbody tr.past-due,
  table.so-table tbody tr.past-due:nth-child(even) {
    background: var(--pastdue-bg);
    box-shadow: inset 3px 0 0 0 var(--pastdue-border);
  }
  tr.past-due td.ship .late {
    color: var(--pastdue-text);
    font-size: 11px;
    margin-left: 4px;
  }

  .empty-state {
    padding: 24px;
    text-align: center;
    color: var(--color-text-tertiary);
    font-size: 12px;
  }
</style>
</head>
<body>

<div class="page-header">
  <div class="page-title">Production planner</div>
  <div class="page-sub">3-week rolling view · generated {{ generated_at }} · {{ open_lines }} open SO line{{ '' if open_lines == 1 else 's' }}</div>
</div>

<div class="week-grid">
{% for w in weeks %}
  <div class="week-card">
    <div class="week-head">
      <div class="week-label">Week {{ w.week_number }}</div>
      <div class="working-days">
        {% for d in w.working_days %}
          <div>
            <div class="wd-num">{{ d.day_num }}</div>
            <div class="wd-letter">{{ d.letter }}</div>
          </div>
        {% endfor %}
      </div>
    </div>

    <div class="metrics">
      <div class="metric granola">
        <div class="metric-label">Granola</div>
        <div class="metric-number">{{ w.granola_total }}</div>
        <div class="metric-sub">batches</div>
      </div>
      <div class="metric coconut">
        <div class="metric-label">Coconut</div>
        <div class="metric-number">{{ w.coconut_total }}</div>
        <div class="metric-sub">batches</div>
      </div>
    </div>

    <div class="week-section">
      <div class="section-label">Batches</div>
      <div class="item-list">
        {% set granola_batches = w.batches | selectattr('line','equalto','granola') | list %}
        {% set coconut_batches = w.batches | selectattr('line','equalto','coconut') | list %}
        {% set all_batches = granola_batches + coconut_batches %}
        {% if not all_batches %}
          <div class="empty">No batches scheduled</div>
        {% else %}
          {% set ns = namespace(saw_coconut=false, saw_granola=false) %}
          {% for b in all_batches %}
            {% if b.line == 'coconut' and not ns.saw_coconut and ns.saw_granola %}
              <div class="divider"></div>
            {% endif %}
            <div class="row"><span class="name">{{ b.name }}</span><span class="val">{{ b.count }}</span></div>
            {% if b.line == 'granola' %}{% set ns.saw_granola = true %}{% endif %}
            {% if b.line == 'coconut' %}{% set ns.saw_coconut = true %}{% endif %}
          {% endfor %}
        {% endif %}
      </div>
    </div>

    <div class="week-section">
      <div class="section-label">Finished goods</div>
      <div class="item-list">
        {% if not w.finished_goods %}
          <div class="empty">Nothing to pack</div>
        {% else %}
          {% for fg in w.finished_goods %}
            <div class="row"><span class="name">{{ fg.name }}</span><span class="val">{{ fg.cases }} cs</span></div>
          {% endfor %}
        {% endif %}
      </div>
    </div>
  </div>
{% endfor %}

  <div class="week-card beyond-card">
    <div class="week-head">
      <div class="week-label">Beyond Week 3</div>
      <div class="beyond-subtitle">{{ beyond.date_range_label }}</div>
    </div>

    <div class="metrics">
      <div class="metric granola-muted">
        <div class="metric-label">Granola</div>
        <div class="metric-number">{{ beyond.granola_total }}</div>
        <div class="metric-sub">batches</div>
      </div>
      <div class="metric coconut-muted">
        <div class="metric-label">Coconut</div>
        <div class="metric-number">{{ beyond.coconut_total }}</div>
        <div class="metric-sub">batches</div>
      </div>
    </div>

    <div class="week-section">
      <div class="section-label">Next big week</div>
      <div class="next-big-weeks">
        {% if not beyond.next_big_weeks %}
          <div class="empty">No production scheduled</div>
        {% else %}
          {% for nb in beyond.next_big_weeks %}
            <div class="next-week-row">
              <div class="next-week-date">{{ nb.week_tuesday_label }}</div>
              <div class="next-week-stack">
                <div class="next-week-count">{{ nb.total_batches }} batches</div>
                <div class="next-week-family">{{ nb.dominant_batch_name }}</div>
              </div>
            </div>
          {% endfor %}
        {% endif %}
      </div>
    </div>
  </div>
</div>

<div class="so-panel">
  <div class="so-head">
    <div class="so-title">Sales orders</div>
    <div class="so-count">{{ so_rows|length }} of {{ so_rows|length }} shown · sorted by ship date</div>
  </div>
  {% if so_rows %}
  <table class="so-table">
    <thead>
      <tr>
        <th class="ship">Ship</th>
        <th class="order">Order</th>
        <th class="cust">Customer</th>
        <th class="items">Items</th>
        <th class="right tot">Total lb</th>
      </tr>
    </thead>
    <tbody>
      {% for r in so_rows %}
      <tr class="{{ 'past-due' if r.is_past_due else '' }}">
        <td class="ship">
          {{ r.ship_date_label }}{% if r.is_past_due %}<span class="late">({{ r.days_late }}d late)</span>{% endif %}
        </td>
        <td class="order">{{ r.order_number }}</td>
        <td class="cust">{{ r.customer }}</td>
        <td class="items">{{ r.items_text }}</td>
        <td class="right tot">{{ '{:,.0f}'.format(r.total_lb) }}</td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <div class="empty-state">No open sales orders inside the 3-week horizon.</div>
  {% endif %}
</div>

</body>
</html>
"""


def render(weeks, beyond, so_rows, open_lines, generated_at):
    tmpl = Template(TEMPLATE)
    return tmpl.render(
        weeks=weeks,
        beyond=beyond,
        so_rows=so_rows,
        open_lines=open_lines,
        generated_at=generated_at,
    )


def audit_fg_names(rows: list[dict]) -> None:
    """First-run audit: print before/after for every unique FG name to stderr."""
    seen: dict[str, str] = {}
    for r in rows:
        raw = r.get("fg_name")
        if not raw:
            continue
        s = str(raw)
        if s in seen:
            continue
        seen[s] = clean_fg_name(s)
    print(f"FG name cleanup audit ({len(seen)} unique names):", file=sys.stderr)
    width = max((len(k) for k in seen), default=0)
    for raw, cleaned in sorted(seen.items()):
        marker = "  " if raw == cleaned else "→ "
        print(f"  {raw.ljust(width)}  {marker}{cleaned}", file=sys.stderr)


def main() -> None:
    db_url = load_env()
    knowledge = load_knowledge()
    case_sizes = case_size_map(knowledge)
    batch_names = batch_name_map(knowledge)
    batch_targets = batch_target_lb_map(knowledge)
    upstream_rules = upstream_rule_map(knowledge)
    discontinued = discontinued_set(knowledge)

    # Surface YAML-vs-DB target_batch_lb drift up front.
    db_targets = fetch_db_batch_targets(db_url)
    for sku, yaml_t in sorted(batch_targets.items()):
        db_t = db_targets.get(sku)
        if db_t is not None and abs(db_t - yaml_t) > 0.5:
            print(
                f"WARN: target_batch_lb drift for {sku}: YAML={yaml_t} DB={db_t} "
                f"(planner uses YAML).",
                file=sys.stderr,
            )

    rows = run_sql(db_url)
    rows, dropped_count, dropped_skus = filter_discontinued(rows, discontinued)
    if dropped_count > 0:
        print(
            f"WARNING: dropped {dropped_count} discontinued-SKU rows ({sorted(dropped_skus)}).",
            file=sys.stderr,
        )

    audit_fg_names(rows)

    customer_map = fetch_customer_map(db_url)

    today = date.today()
    weeks = build_weeks(today)
    bucketed, beyond_rows = bucket_rows_into_weeks(rows, weeks)
    for w, week_rows in zip(weeks, bucketed):
        agg = aggregate_week(week_rows, case_sizes, batch_names, batch_targets, upstream_rules)
        w.update(agg)
        print(
            f"Week {w['week_number']} batches: {len(agg['batches'])} entries, all shown | "
            f"Week {w['week_number']} FGs: {len(agg['finished_goods'])} entries, all shown",
            file=sys.stderr,
        )
    beyond = aggregate_beyond(beyond_rows, case_sizes, batch_names, batch_targets, upstream_rules)

    so_rows = build_so_rows(rows, weeks, customer_map, today)

    html = render(
        weeks=weeks,
        beyond=beyond,
        so_rows=so_rows,
        open_lines=len(rows),
        generated_at=today.strftime("%b %-d, %Y"),
    )
    OUT_FILE.write_text(html)

    g_total = sum(w["granola_total"] for w in weeks)
    c_total = sum(w["coconut_total"] for w in weeks)
    print(
        f"Rendered W1-W3: {g_total} granola, {c_total} coconut batches (gross) | "
        f"Beyond: {beyond['granola_total']} granola, {beyond['coconut_total']} coconut "
        f"({beyond['date_range_label']}) | "
        f"{len(so_rows)} SOs across {len(rows)} open lines -> {OUT_FILE.name}"
    )


if __name__ == "__main__":
    main()
