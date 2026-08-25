#!/usr/bin/env python3
"""Read-only data-health baseline for Factory Ledger (2026-08 audit).

Talks to production Postgres through the pgbouncer transaction pooler
(port 6543) using ONLY transaction-scoped read-only transactions:
every query runs inside an explicit `BEGIN TRANSACTION READ ONLY; ...; COMMIT;`
frame on an autocommit connection. No session-level GUCs are ever set
(see CLAUDE.md hard rule — session GUCs leak through the shared 6543 pool).

Prints a markdown report to stdout. Usage:
    .venv-test/bin/python scripts/data_health.py > docs/data-health-baseline-2026-08-24.md
"""
from __future__ import annotations

import os
import re
import sys
from collections import defaultdict
from datetime import date, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor

# ── connection ────────────────────────────────────────────────────────────
def load_database_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
        with open(env_path) as fh:
            for line in fh:
                if line.startswith("DATABASE_URL="):
                    url = line.split("=", 1)[1].strip()
    if not url:
        raise SystemExit("DATABASE_URL not found")
    return url


CONN = psycopg2.connect(load_database_url())
CONN.autocommit = True  # no implicit BEGIN — we frame every query explicitly


def q(sql: str, params=None) -> list[dict]:
    """Run one query wrapped in BEGIN TRANSACTION READ ONLY ... COMMIT."""
    with CONN.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("BEGIN TRANSACTION READ ONLY")
        try:
            cur.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
            cur.execute("COMMIT")
            return [dict(r) for r in rows]
        except Exception:
            cur.execute("ROLLBACK")
            raise


# ── constants ─────────────────────────────────────────────────────────────
EPOCH = date(2026, 8, 11)
TODAY = q("SELECT (now() AT TIME ZONE 'America/New_York')::date AS d")[0]["d"]
WINDOWS = {
    "last 7 d": (TODAY - timedelta(days=6), TODAY),
    "last 30 d": (TODAY - timedelta(days=29), TODAY),
    "post-epoch": (EPOCH, TODAY),
}
POSTED_T = "(SELECT * FROM ledger_current_transactions WHERE effective_status = 'posted')"
POSTED_L = (
    "(SELECT tl.* FROM ledger_current_transaction_lines tl "
    " JOIN ledger_current_transactions ct ON ct.id = tl.transaction_id "
    " WHERE ct.effective_status = 'posted')"
)


def fmt(v):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:,.1f}"
    return str(v)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(fmt(c) for c in r) + " |")
    if not rows:
        out.append("| _no rows_ " + "| " * (len(headers) - 1) + "|")
    return "\n".join(out)


def emit(*lines):
    for ln in lines:
        print(ln)
    print()


# ══════════════════════════════════════════════════════════════════════════
emit(
    "# Factory Ledger — data-health baseline (read-only audit)",
    "",
    f"Generated {TODAY} against production via pgbouncer 6543, per-query "
    "`BEGIN TRANSACTION READ ONLY`. Epoch = 2026-08-11 (bulk backfill "
    "boundary; migration_backfill_039 ran 2026-08-11 14:32 UTC). All "
    "windows are by `business_date` (authoritative ET business day). "
    "'Live' rows = `created_at_source = 'database'`; 'backfilled' = "
    "`created_at_source IN ('migration_backfill_039','legacy_unverified')`.",
)

# ── denominators ──────────────────────────────────────────────────────────
rows = []
for wname, (a, b) in WINDOWS.items():
    for r in q(
        f"""SELECT type, count(*) n,
                   sum((created_at_source <> 'database')::int) backfilled
            FROM {POSTED_T} t WHERE business_date BETWEEN %s AND %s
            GROUP BY type ORDER BY type""",
        (a, b),
    ):
        rows.append([wname, r["type"], r["n"], r["backfilled"]])
emit("## 0. Posted transaction counts (denominators)", "",
     table(["window", "type", "posted txns", "backfilled"], rows))

# ── a. entry lag ──────────────────────────────────────────────────────────
rows = []
for wname, (a, b) in WINDOWS.items():
    for r in q(
        f"""SELECT type, count(*) n,
              percentile_cont(0.5) WITHIN GROUP (ORDER BY lag_h) med_h,
              percentile_cont(0.9) WITHIN GROUP (ORDER BY lag_h) p90_h,
              round(avg((same_day)::int)*100, 1) pct_same_day,
              round(avg((lag_h > 24)::int)*100, 1) pct_over_24h,
              round(avg((abs(lag_h) < 0.034)::int)*100, 1) pct_lag_lt_2min
            FROM (
              SELECT type,
                extract(epoch FROM (created_at - occurred_at))/3600.0 lag_h,
                business_date = (created_at AT TIME ZONE 'America/New_York')::date same_day
              FROM {POSTED_T} t
              WHERE created_at_source = 'database'
                AND business_date BETWEEN %s AND %s) s
            GROUP BY type ORDER BY type""",
        (a, b),
    ):
        rows.append([wname, r["type"], r["n"], round(float(r["med_h"]), 2),
                     round(float(r["p90_h"]), 2), r["pct_same_day"],
                     r["pct_over_24h"], r["pct_lag_lt_2min"]])
emit("## 3a. Entry lag (occurred_at → created_at), LIVE rows only", "",
     table(["window", "type", "n", "median h", "p90 h", "% same-day",
            "% >24 h", "% |lag|<2 min"], rows))

# is occurred_at just defaulted? distribution of lag rounded to 0.5h, post-epoch
rows = [
    [float(r["lag_bin_h"]), r["n"]]
    for r in q(
        f"""SELECT round(extract(epoch FROM (created_at - occurred_at))/1800.0)/2 lag_bin_h,
                   count(*) n
            FROM {POSTED_T} t
            WHERE created_at_source = 'database' AND business_date >= %s
            GROUP BY 1 ORDER BY n DESC LIMIT 8""",
        (EPOCH,),
    )
]
emit("### 3a-ii. Lag distribution, live rows post-epoch (0.5 h bins — a single "
     "dominant bin means occurred_at is defaulted at entry time and lag is NOT "
     "measurable)", "", table(["lag bin (h)", "txns"], rows))

# ── b. backfill rate weekly from Jul 1 ────────────────────────────────────
rows = [
    [str(r["wk"]), r["type"], r["n"], r["backfilled"],
     round(100.0 * r["backfilled"] / r["n"], 1)]
    for r in q(
        f"""SELECT date_trunc('week', business_date)::date wk, type,
                   count(*) n, sum((created_at_source <> 'database')::int) backfilled
            FROM {POSTED_T} t WHERE business_date >= '2026-07-01'
            GROUP BY 1, 2 ORDER BY 1, 2"""
    )
]
emit("## 3b. Backfill rate by type, weekly from Jul 1 (week = Monday start)",
     "", table(["week", "type", "txns", "backfilled", "% backfilled"], rows))

# ── c. entry bursts ───────────────────────────────────────────────────────
rows = []
flagged = []
for r in q(
    f"""WITH e AS (
          SELECT created_at,
                 (created_at AT TIME ZONE 'America/New_York')::date entry_day,
                 CASE WHEN lag(created_at) OVER (ORDER BY created_at, id) IS NULL
                        OR created_at - lag(created_at) OVER (ORDER BY created_at, id)
                           > interval '5 minutes'
                      THEN 1 ELSE 0 END nb
          FROM {POSTED_T} t
          WHERE created_at_source = 'database'
            AND business_date >= %s),
        b AS (SELECT *, sum(nb) OVER (ORDER BY created_at) burst_id FROM e),
        c AS (SELECT *, count(*) OVER (PARTITION BY burst_id) burst_n FROM b)
        SELECT entry_day, count(*) txns, count(DISTINCT burst_id) bursts,
               max(burst_n) biggest_burst
        FROM c GROUP BY entry_day ORDER BY entry_day""",
    (TODAY - timedelta(days=29),),
):
    pct = round(100.0 * r["biggest_burst"] / r["txns"], 1)
    flag = "**FLAG**" if pct > 80 else ""
    if pct > 80:
        flagged.append(str(r["entry_day"]))
    rows.append([str(r["entry_day"]), r["txns"], r["bursts"],
                 r["biggest_burst"], pct, flag])
emit("## 3c. Entry bursts (live rows, last 30 d by entry day; burst = txns "
     "entered ≤5 min apart)", "",
     table(["entry day (ET)", "txns", "bursts", "biggest burst",
            "% in biggest", ">80% in one burst"], rows),
     "", f"Flagged days: {', '.join(flagged) or 'none'}")

# ── d. production-day coverage ────────────────────────────────────────────
rows = [
    [str(r["d"]), r["dow"], r["makes"] or 0,
     str(r["first_entered_et"]) if r["first_entered_et"] else "—",
     "" if r["makes"] else "**NO MAKE**"]
    for r in q(
        f"""WITH days AS (
              SELECT d::date d, to_char(d, 'Dy') dow
              FROM generate_series(%s::date, %s::date, '1 day') d
              WHERE extract(isodow FROM d) < 6),
            mk AS (
              SELECT business_date, count(*) makes,
                     min(created_at AT TIME ZONE 'America/New_York') first_entered_et
              FROM {POSTED_T} t WHERE type = 'make' GROUP BY 1)
            SELECT days.d, days.dow, mk.makes, mk.first_entered_et
            FROM days LEFT JOIN mk ON mk.business_date = days.d
            ORDER BY days.d""",
        (EPOCH, TODAY),
    )
]
emit("## 3d. Production-day coverage (weekdays post-epoch)", "",
     table(["weekday", "dow", "make txns", "first make entered (ET)", "gap"],
           rows))

# lag > 24h attribution
rows = [
    [r["id"], r["type"], str(r["business_date"]), round(float(r["lag_h"]), 1),
     r["operator_id"], (r["notes"] or "—")[:48]]
    for r in q(
        f"""SELECT t.id, t.type, t.business_date, t.operator_id, t.notes,
                   extract(epoch FROM (created_at - occurred_at))/3600.0 lag_h
            FROM {POSTED_T} t
            WHERE created_at_source = 'database' AND business_date >= %s
              AND extract(epoch FROM (created_at - occurred_at))/3600.0 > 24
            ORDER BY t.id LIMIT 100""",
        (EPOCH,),
    )
]
emit("### 3a-iii. Live txns post-epoch with lag > 24 h (who/what they are)",
     "", table(["txn", "type", "business_date", "lag h", "operator_id",
                "notes (first 48 ch)"], rows))

# ── e. ship linkage ───────────────────────────────────────────────────────
SHIP_BASE = f"""
    SELECT t.id, t.business_date, t.customer_name,
           nullif(btrim(coalesce(t.customer_name, '')), '') IS NOT NULL has_customer,
           nullif(btrim(coalesce(t.order_reference, '')), '') IS NOT NULL has_ordref,
           EXISTS (SELECT 1 FROM sales_order_shipments sos
                   WHERE sos.transaction_id = t.id) has_so,
           EXISTS (SELECT 1 FROM shipment_lines sl
                   WHERE sl.transaction_id = t.id) has_shiprow,
           coalesce((SELECT sum(abs(tl.quantity_lb)) FROM {POSTED_L} tl
                     WHERE tl.transaction_id = t.id AND tl.quantity_lb < 0), 0) lb
    FROM {POSTED_T} t
    WHERE t.type = 'ship' AND t.business_date BETWEEN %s AND %s"""
rows = []
for wname, (a, b) in WINDOWS.items():
    r = q(
        f"""SELECT count(*) n, sum(lb) lb,
                   round(avg(has_customer::int)*100, 1) pct_cust_n,
                   round(100*sum(lb*has_customer::int)/nullif(sum(lb),0), 1) pct_cust_lb,
                   round(avg(has_ordref::int)*100, 1) pct_ordref,
                   round(avg(has_so::int)*100, 1) pct_so,
                   round(100*sum(lb*has_so::int)/nullif(sum(lb),0), 1) pct_so_lb,
                   round(avg(has_shiprow::int)*100, 1) pct_shiprow,
                   round(avg((has_ordref OR has_so)::int)*100, 1) pct_any
            FROM ({SHIP_BASE}) s""",
        (a, b),
    )[0]
    rows.append([wname, r["n"], r["lb"], r["pct_cust_n"], r["pct_cust_lb"],
                 r["pct_ordref"], r["pct_so"], r["pct_so_lb"],
                 r["pct_shiprow"], r["pct_any"]])
emit("## 3e. Ship linkage — customer and order linkage split by evidence "
     "type (`shipment_lines` rows are written mechanically on every /ship "
     "commit, so they are shown separately and NOT counted as an SO link)",
     "",
     table(["window", "ship txns", "lb", "% customer (n)", "% customer (lb)",
            "% order_reference", "% SO link (n)", "% SO link (lb)",
            "% shipment row", "% order_ref OR SO"], rows))

rows = [
    [r["id"], str(r["business_date"]), r["customer_name"] or "—", r["lb"]]
    for r in q(
        f"""SELECT * FROM ({SHIP_BASE}) s
            WHERE NOT (has_ordref OR has_so) ORDER BY business_date, id""",
        (TODAY - timedelta(days=29), TODAY),
    )
]
emit("### 3e-ii. Ship txns with neither order_reference nor SO link, last 30 d",
     "", table(["txn id", "business_date", "customer_name", "lb"], rows))

# ── f. receipt linkage ────────────────────────────────────────────────────
rows = []
for wname, (a, b) in WINDOWS.items():
    r = q(
        f"""SELECT count(*) n,
              round(avg((nullif(btrim(coalesce(t.shipper_name,'')),'') IS NOT NULL)::int)*100,1) pct_supplier,
              round(avg((nullif(btrim(coalesce(t.bol_reference,'')),'') IS NOT NULL)::int)*100,1) pct_bol,
              round(avg((nullif(btrim(coalesce(t.shipper_name,'')),'') IS NOT NULL
                     AND nullif(btrim(coalesce(t.bol_reference,'')),'') IS NOT NULL)::int)*100,1) pct_both,
              round(avg((bt.expected_receipt_id IS NOT NULL)::int)*100,1) pct_er
            FROM {POSTED_T} t
            JOIN transactions bt ON bt.id = t.id
            WHERE t.type = 'receive' AND t.business_date BETWEEN %s AND %s""",
        (a, b),
    )[0]
    rows.append([wname, r["n"], r["pct_supplier"], r["pct_bol"],
                 r["pct_both"], r["pct_er"]])
er = q("SELECT count(*) n, count(*) FILTER (WHERE status = 'open') open "
       "FROM expected_receipts")[0]
emit("## 3f. Receipt linkage (expected_receipt_id read from base "
     "`transactions` — the ledger view predates migration 041 and does not "
     "expose it)", "",
     table(["window", "receive txns", "% supplier", "% BOL/ref",
            "% supplier+BOL", "% matched to expected receipt"], rows),
     "",
     f"`expected_receipts` rows ever created: **{er['n']}** "
     f"({er['open']} open) — receipts cannot match what was never entered.")

# ── g. trace completeness ─────────────────────────────────────────────────
r = q(
    f"""WITH mk AS (SELECT id FROM {POSTED_T} t
                    WHERE type = 'make' AND business_date BETWEEN %s AND %s),
        ing AS (SELECT mk.id txn_id, ilc.ingredient_lot_id lot_id
                FROM mk JOIN ingredient_lot_consumption ilc ON ilc.transaction_id = mk.id),
        res AS (SELECT DISTINCT lot_id,
                  EXISTS (SELECT 1 FROM {POSTED_T} t
                          JOIN ledger_current_transaction_lines tl ON tl.transaction_id = t.id
                          WHERE t.type = 'receive' AND tl.lot_id = ing.lot_id
                            AND nullif(btrim(coalesce(t.shipper_name,'')),'') IS NOT NULL) ok
                FROM ing),
        per_batch AS (SELECT ing.txn_id, bool_and(res.ok) all_ok, count(*) n_lots
                      FROM ing JOIN res USING (lot_id) GROUP BY ing.txn_id)
        SELECT count(*) batches,
               sum(all_ok::int) fully_resolved,
               round(avg(all_ok::int)*100, 1) pct
        FROM per_batch""",
    (TODAY - timedelta(days=29), TODAY),
)[0]
emit("## 3g. Trace completeness — batches (make txns) last 30 d whose consumed "
     "ingredient lots ALL resolve to a posted receive with a supplier name", "",
     table(["batches", "fully resolved", "%"],
           [[r["batches"], r["fully_resolved"], r["pct"]]]))

rows = [
    [r["lot_code"], r["product"], r["entry_source"] or "—",
     r["on_hand"], r["age_days"]]
    for r in q(
        f"""WITH mk AS (SELECT id FROM {POSTED_T} t
                        WHERE type = 'make' AND business_date BETWEEN %s AND %s),
            bad AS (SELECT DISTINCT ilc.ingredient_lot_id lot_id
                    FROM mk JOIN ingredient_lot_consumption ilc ON ilc.transaction_id = mk.id
                    WHERE NOT EXISTS (
                      SELECT 1 FROM {POSTED_T} t
                      JOIN ledger_current_transaction_lines tl ON tl.transaction_id = t.id
                      WHERE t.type = 'receive' AND tl.lot_id = ilc.ingredient_lot_id
                        AND nullif(btrim(coalesce(t.shipper_name,'')),'') IS NOT NULL))
            SELECT l.lot_code, p.name product, l.entry_source,
                   coalesce((SELECT sum(tl.quantity_lb) FROM {POSTED_L} tl
                             WHERE tl.lot_id = l.id), 0) on_hand,
                   round(extract(epoch FROM (now() - coalesce(l.received_at,
                         l.created_at AT TIME ZONE 'UTC')))/86400.0) age_days
            FROM bad JOIN lots l ON l.id = bad.lot_id
            JOIN products p ON p.id = l.product_id
            ORDER BY p.name, l.lot_code""",
        (TODAY - timedelta(days=29), TODAY),
    )
]
emit("### 3g-ii. Unresolved ingredient lots (consumed last 30 d, no "
     "supplier-bearing receive)", "",
     table(["lot_code", "product", "entry_source", "on-hand lb", "age days"],
           rows))

# ── h. lot hygiene ────────────────────────────────────────────────────────
rows = [
    [r["product"], r["ptype"], r["n_lots"], r["lb"]]
    for r in q(
        f"""WITH oh AS (
              SELECT l.id, l.product_id,
                     coalesce((SELECT sum(tl.quantity_lb) FROM {POSTED_L} tl
                               WHERE tl.lot_id = l.id), 0) bal,
                     extract(epoch FROM (now() - coalesce(l.received_at,
                        l.created_at AT TIME ZONE 'UTC')))/86400.0 age_d
              FROM lots l)
            SELECT p.name product, p.type ptype, count(*) n_lots,
                   round(sum(oh.bal), 1) lb
            FROM oh JOIN products p ON p.id = oh.product_id
            WHERE oh.bal > 0 AND oh.age_d > 30
              AND ((p.type = 'batch' AND p.default_batch_lb > 0
                    AND oh.bal < 0.5 * p.default_batch_lb)
                OR (p.type = 'finished' AND p.case_size_lb > 0
                    AND oh.bal < p.case_size_lb))
            GROUP BY p.name, p.type ORDER BY lb DESC""",
    )
]
emit("## 3h. Lot hygiene", "",
     "### 3h-i. Stale fractional lots: on-hand > 0, age > 30 d, and "
     "< 0.5 batch (batch products) or < 1 case (finished)", "",
     table(["product", "type", "lots", "total lb"], rows))

rows = [
    [r["lot_code"], r["product"], r["ptype"], r["bal"]]
    for r in q(
        f"""SELECT l.lot_code, p.name product, p.type ptype,
                   round(coalesce((SELECT sum(tl.quantity_lb) FROM {POSTED_L} tl
                                   WHERE tl.lot_id = l.id), 0), 2) bal
            FROM lots l JOIN products p ON p.id = l.product_id
            WHERE coalesce((SELECT sum(tl.quantity_lb) FROM {POSTED_L} tl
                            WHERE tl.lot_id = l.id), 0) < 0
            ORDER BY bal""",
    )
]
emit("### 3h-ii. Negative lot balances (posted-only)", "",
     table(["lot_code", "product", "type", "balance lb"], rows))

# lot-code formats per family + Spanish months + near-dupes (Python side)
lots = q(
    """SELECT l.id, l.lot_code, p.name product, p.type ptype,
              p.product_category, p.parent_batch_product_id,
              pp.name parent_name
       FROM lots l JOIN products p ON p.id = l.product_id
       LEFT JOIN products pp ON pp.id = p.parent_batch_product_id"""
)

def family(r):
    if r["ptype"] == "finished" and r["parent_name"]:
        return r["parent_name"] + " (family)"
    return r["product"]

def sig(code):
    s = re.sub(r"[A-Za-z]+", "A", code.strip())
    s = re.sub(r"[0-9]+", "9", s)
    return s

fam_formats = defaultdict(lambda: defaultdict(list))
for r in lots:
    fam_formats[family(r)][sig(r["lot_code"])].append(r["lot_code"])
rows = []
for fam, fmts in sorted(fam_formats.items()):
    if len(fmts) > 1:
        ex = "; ".join(f"`{sorted(v)[0]}`" for v in fmts.values())
        rows.append([fam, len(fmts), ex[:120]])
emit("### 3h-iii. Product families with >1 distinct lot-code format "
     "(format = letter/digit-run signature; family = parent batch product "
     "for finished goods, else product)", "",
     table(["family", "formats", "one example per format"], rows))

ES = {"ENE": "JAN", "ABR": "APR", "AGO": "AUG", "DIC": "DEC", "SET": "SEP"}
es_rows, near = [], []
codes_norm = defaultdict(list)  # normalized -> [(code, product)]
for r in lots:
    codes_norm[r["lot_code"].strip().upper()].append(r)
for r in lots:
    up = r["lot_code"].strip().upper()
    toks = [t for t in ES if re.search(rf"(?<![A-Z]){t}(?![A-Z])", up)]
    if toks:
        es_rows.append([r["lot_code"], r["product"], ",".join(toks)])
        norm = up
        for t in toks:
            norm = re.sub(rf"(?<![A-Z]){t}(?![A-Z])", ES[t], norm)
        if norm != up and norm in codes_norm:
            for other in codes_norm[norm]:
                near.append([r["lot_code"], other["lot_code"],
                             r["product"], other["product"]])
emit("### 3h-iv. Lot codes containing non-English month tokens "
     "(ENE/ABR/AGO/DIC/SET)", "",
     table(["lot_code", "product", "tokens"], es_rows), "",
     "Near-duplicates differing only by month spelling:", "",
     table(["Spanish-month code", "English twin", "product (ES)",
            "product (EN)"], near))

# ── i. lot-code collisions ────────────────────────────────────────────────
rows = [
    [r["code"], r["n"], r["products"][:160]]
    for r in q(
        """SELECT upper(btrim(l.lot_code)) code, count(*) n,
                  string_agg(p.name || ' [' || p.type || '] lot ' || l.id,
                             ' ; ' ORDER BY l.id) products
           FROM lots l JOIN products p ON p.id = l.product_id
           GROUP BY 1 HAVING count(*) > 1 ORDER BY n DESC, code"""
    )
]
emit("## 3i. Lot-code collisions (same code, >1 lot record)", "",
     f"Total colliding codes: **{len(rows)}**", "",
     table(["code", "lots", "products"], rows))

rows = [
    [r["lot_id"], r["product"], r["odoo_code"] or "—", r["ptype"],
     r["entry_source"] or "—", r["bal"]]
    for r in q(
        f"""SELECT l.id lot_id, p.name product, p.odoo_code, p.type ptype,
                   l.entry_source,
                   round(coalesce((SELECT sum(tl.quantity_lb) FROM {POSTED_L} tl
                                   WHERE tl.lot_id = l.id), 0), 1) bal
            FROM lots l JOIN products p ON p.id = l.product_id
            WHERE lower(l.lot_code) = lower('AUG 21 2026') ORDER BY l.id"""
    )
]
emit('### 3i-ii. Every lot record coded "AUG 21 2026"', "",
     table(["lot id", "product", "SKU", "type", "entry_source", "on-hand lb"],
           rows))

rows = [
    [r["txn"], r["type"], str(r["business_date"]), r["product"],
     r["odoo_code"] or "—", r["qty"], r["customer_name"] or "—",
     r["eff"], "lot " + str(r["lot_id"])]
    for r in q(
        """SELECT t.id txn, t.type, t.business_date, t.customer_name,
                  t.effective_status eff, p.name product, p.odoo_code,
                  tl.quantity_lb qty, tl.lot_id
           FROM ledger_current_transaction_lines tl
           JOIN ledger_current_transactions t ON t.id = tl.transaction_id
           JOIN lots l ON l.id = tl.lot_id
           JOIN products p ON p.id = tl.product_id
           WHERE lower(l.lot_code) = lower('AUG 21 2026')
           ORDER BY t.id"""
    )
]
emit('### 3i-iii. Every transaction the backward trace would attach to code '
     '"AUG 21 2026" (all lots with that code)', "",
     table(["txn", "type", "business_date", "product", "SKU", "qty lb",
            "customer", "status", "lot"], rows))

rows = [
    [r["txn"], r["type"], str(r["business_date"]), r["product"],
     r["odoo_code"] or "—", r["lot_code"], r["qty"], r["customer_name"] or "—"]
    for r in q(
        """SELECT t.id txn, t.type, t.business_date, t.customer_name,
                  p.name product, p.odoo_code, l.lot_code, tl.quantity_lb qty
           FROM ledger_current_transaction_lines tl
           JOIN ledger_current_transactions t ON t.id = tl.transaction_id
           JOIN lots l ON l.id = tl.lot_id
           JOIN products p ON p.id = tl.product_id
           WHERE t.id IN (2087, 2090) ORDER BY t.id, tl.id"""
    )
]
emit("### 3i-iv. Txns #2087 and #2090 as actually recorded", "",
     table(["txn", "type", "business_date", "product", "SKU", "lot_code",
            "qty lb", "customer"], rows))

# ── j. on-hand drift ──────────────────────────────────────────────────────
D30 = TODAY - timedelta(days=29)
rows = []
for r in q(
    f"""WITH bal AS (
          SELECT tl.product_id,
                 sum(tl.quantity_lb) today,
                 sum(tl.quantity_lb) FILTER (WHERE t.business_date < %s) at_epoch
          FROM {POSTED_L} tl JOIN {POSTED_T} t ON t.id = tl.transaction_id
          GROUP BY tl.product_id),
        made AS (
          SELECT tl.product_id, sum(tl.quantity_lb) lb
          FROM {POSTED_L} tl JOIN {POSTED_T} t ON t.id = tl.transaction_id
          WHERE t.type = 'make' AND tl.quantity_lb > 0
            AND t.business_date BETWEEN %s AND %s GROUP BY 1),
        packed_out AS (
          SELECT l.product_id, sum(ilc.quantity_lb) lb
          FROM ingredient_lot_consumption ilc
          JOIN {POSTED_T} t ON t.id = ilc.transaction_id
          JOIN lots l ON l.id = ilc.ingredient_lot_id
          WHERE t.type = 'pack' AND t.business_date BETWEEN %s AND %s
          GROUP BY 1)
        SELECT p.name, coalesce(bal.at_epoch, 0) at_epoch,
               coalesce(bal.today, 0) today,
               coalesce(made.lb, 0) made30, coalesce(packed_out.lb, 0) packed30
        FROM products p
        LEFT JOIN bal ON bal.product_id = p.id
        LEFT JOIN made ON made.product_id = p.id
        LEFT JOIN packed_out ON packed_out.product_id = p.id
        WHERE p.type = 'batch' AND coalesce(p.active, true)
          AND (bal.today <> 0 OR made.lb IS NOT NULL OR packed_out.lb IS NOT NULL)
        ORDER BY p.name""",
    (EPOCH, D30, TODAY, D30, TODAY),
):
    ep, td = float(r["at_epoch"]), float(r["today"])
    m, pk = float(r["made30"]), float(r["packed30"])
    ratio = f"{m:,.0f}:{pk:,.0f}" + (f" ({m/pk:.2f})" if pk else "")
    flag = "**FLAG**" if ep > 0 and td > 1.2 * ep and m > 0 and pk > 0 else ""
    rows.append([r["name"], round(ep, 1), round(td, 1), ratio, flag])
emit("## 3j. On-hand drift — batch products (epoch vs today; made:packed "
     "over last 30 d; flag = on-hand rose >20% with both flows non-zero)", "",
     table(["batch product", "on-hand @ epoch", "on-hand today",
            "made:packed 30 d", "flag"], rows))

rows = []
for r in q(
    f"""WITH ship30 AS (
          SELECT tl.product_id, sum(abs(tl.quantity_lb)) lb
          FROM {POSTED_L} tl JOIN {POSTED_T} t ON t.id = tl.transaction_id
          WHERE t.type = 'ship' AND tl.quantity_lb < 0
            AND t.business_date BETWEEN %s AND %s GROUP BY 1),
        pack30 AS (
          SELECT tl.product_id, sum(tl.quantity_lb) lb
          FROM {POSTED_L} tl JOIN {POSTED_T} t ON t.id = tl.transaction_id
          WHERE t.type = 'pack' AND tl.quantity_lb > 0
            AND t.business_date BETWEEN %s AND %s GROUP BY 1),
        bal AS (
          SELECT tl.product_id, sum(tl.quantity_lb) today,
                 sum(tl.quantity_lb) FILTER (WHERE t.business_date < %s) at_epoch
          FROM {POSTED_L} tl JOIN {POSTED_T} t ON t.id = tl.transaction_id
          GROUP BY 1)
        SELECT p.name, p.odoo_code,
               coalesce(bal.at_epoch, 0) at_epoch, coalesce(bal.today, 0) today,
               coalesce(pack30.lb, 0) packed30, coalesce(ship30.lb, 0) shipped30
        FROM products p
        LEFT JOIN bal ON bal.product_id = p.id
        LEFT JOIN pack30 ON pack30.product_id = p.id
        LEFT JOIN ship30 ON ship30.product_id = p.id
        WHERE p.type = 'finished'
        ORDER BY coalesce(ship30.lb, 0) DESC, coalesce(bal.today, 0) DESC
        LIMIT 20""",
    (D30, TODAY, D30, TODAY, EPOCH),
):
    ep, td = float(r["at_epoch"]), float(r["today"])
    pk, sh = float(r["packed30"]), float(r["shipped30"])
    ratio = f"{pk:,.0f}:{sh:,.0f}" + (f" ({pk/sh:.2f})" if sh else "")
    flag = "**FLAG**" if ep > 0 and td > 1.2 * ep and pk > 0 and sh > 0 else ""
    rows.append([r["name"], r["odoo_code"] or "—", round(ep, 1), round(td, 1),
                 ratio, flag])
emit("### 3j-ii. On-hand drift — top 20 finished SKUs by 30-d shipped lb "
     "(packed:shipped over 30 d)", "",
     table(["finished SKU", "SKU code", "on-hand @ epoch", "on-hand today",
            "packed:shipped 30 d", "flag"], rows))

# ── k. customer entity hygiene ────────────────────────────────────────────
SUFFIX = re.compile(
    r"\b(INC|LLC|CORP|CO|COMPANY|LTD|USA|DBA)\b\.?", re.I)

def norm_name(s):
    s = re.sub(r"[^\w\s]", " ", s.upper())
    s = SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

cust = q("SELECT id, name, active FROM customers ORDER BY id")
groups = defaultdict(list)
for c in cust:
    groups[norm_name(c["name"])].append(c)
rows = [
    [k, "; ".join(f"#{c['id']} `{c['name']}`"
                  + ("" if c["active"] else " (inactive)") for c in v)]
    for k, v in sorted(groups.items()) if len(v) > 1
]
emit("## 3k. Customer entity hygiene", "",
     f"`customers` rows: {len(cust)}. Normalized-name collisions "
     "(case/punctuation/suffix folded):", "",
     table(["normalized", "customer records"], rows))

ship_names = q(
    f"""SELECT coalesce(nullif(btrim(customer_name), ''), '(blank)') name,
               count(*) n, round(min(business_date - date '1970-01-01')) f,
               max(business_date) last_seen
        FROM {POSTED_T} t WHERE type = 'ship' GROUP BY 1"""
)
sgroups = defaultdict(list)
for c in ship_names:
    sgroups[norm_name(c["name"])].append(c)
rows = [
    [k, "; ".join(f"`{c['name']}` ({c['n']}×, last {c['last_seen']})"
                  for c in v)]
    for k, v in sorted(sgroups.items()) if len(v) > 1
]
emit("### 3k-ii. Free-text `transactions.customer_name` collisions on ship "
     "txns (this is what the Sankey nodes come from)", "",
     table(["normalized", "variants"], rows))

rd = [c for c in ship_names + cust
      if re.search(r"RESTAURANT\s*DEPOT|JETRO", str(c.get("name", "")), re.I)]
rows = [[c.get("name"), c.get("n", "customers-table row #%s" % c.get("id"))]
        for c in rd]
emit("### 3k-iii. Restaurant Depot / Jetro variants seen anywhere", "",
     table(["name", "ship-txn count / source"], rows))

# ── l. pack allocation ────────────────────────────────────────────────────
rows = [
    [r["txn"], str(r["business_date"]), r["product"] or "—", r["n_lots"],
     r["lots"][:120]]
    for r in q(
        f"""SELECT t.id txn, t.business_date, count(DISTINCT ilc.ingredient_lot_id) n_lots,
                   min(p.name) product,
                   string_agg(DISTINCT l.lot_code, ', ') lots
            FROM {POSTED_T} t
            JOIN ingredient_lot_consumption ilc ON ilc.transaction_id = t.id
            JOIN lots l ON l.id = ilc.ingredient_lot_id
            JOIN products p ON p.id = ilc.ingredient_product_id
            WHERE t.type = 'pack' AND t.business_date BETWEEN %s AND %s
            GROUP BY t.id, t.business_date
            HAVING count(DISTINCT ilc.ingredient_lot_id) >= 3
            ORDER BY t.business_date""",
        (TODAY - timedelta(days=59), TODAY),
    )
]
emit("## 3l. Packs drawing from 3+ source lots, last 60 d", "",
     f"Count: **{len(rows)}**", "",
     table(["pack txn", "business_date", "source product", "lots drawn",
            "lot codes"], rows))

CONN.close()
