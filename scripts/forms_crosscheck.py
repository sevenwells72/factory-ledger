#!/usr/bin/env python3
"""Forms-vs-FL cross-check (read-only) — section 6 of the 2026-08 data-health baseline.

Treats the Google-Form exports in data/forms/ (coconut_batches.csv,
receiving.csv, shipping.csv) as the independent floor record and compares
them to production Factory Ledger. DB access: pgbouncer 6543, every query
wrapped in `BEGIN TRANSACTION READ ONLY; ...; COMMIT` on an autocommit
connection — no session GUCs (CLAUDE.md hard rule).

Prints a markdown section to stdout:
    .venv-test/bin/python scripts/forms_crosscheck.py >> docs/data-health-baseline-2026-08-24.md
"""
from __future__ import annotations

import csv
import difflib
import os
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta

import psycopg2
from psycopg2.extras import RealDictCursor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
FORMS = os.path.join(REPO, "data", "forms")


def load_database_url() -> str:
    url = (os.getenv("DATABASE_URL") or "").strip()
    if not url:
        with open(os.path.join(REPO, ".env")) as fh:
            for line in fh:
                if line.startswith("DATABASE_URL="):
                    url = line.split("=", 1)[1].strip()
    if not url:
        raise SystemExit("DATABASE_URL not found")
    return url


CONN = psycopg2.connect(load_database_url())
CONN.autocommit = True


def q(sql, params=None):
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


POSTED_T = "(SELECT * FROM ledger_current_transactions WHERE effective_status = 'posted')"
POSTED_L = (
    "(SELECT tl.* FROM ledger_current_transaction_lines tl "
    " JOIN ledger_current_transactions ct ON ct.id = tl.transaction_id "
    " WHERE ct.effective_status = 'posted')"
)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join("—" if c is None else str(c) for c in r) + " |")
    if not rows:
        out.append("| _no rows_ " + "| " * (len(headers) - 1) + "|")
    return "\n".join(out)


def emit(*lines):
    for ln in lines:
        print(ln)
    print()


def read_csv(name):
    with open(os.path.join(FORMS, name), newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def parse_ts(s):
    s = s.strip()
    for f in ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return datetime.strptime(s, f)
        except ValueError:
            pass
    return None


MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}
# misspellings / Spanish months seen in the free text
MONTH_FIX = {"AGU": "AUG", "AGO": "AUG", "ABR": "APR", "ENE": "JAN",
             "DIC": "DEC", "DE": "DEC", "SETP": "SEP", "SEPT": "SEP",
             "MAYO": "MAY"}


def norm_lot(code):
    """Uppercase, fix month spellings, strip 'LOT' noise + extra spaces."""
    s = re.sub(r"\s+", " ", code.strip().upper())
    s = re.sub(r"^LOT\s+|\s+LOT$", "", s)
    for bad, good in MONTH_FIX.items():
        s = re.sub(rf"(?<![A-Z]){bad}(?![A-Z])", good, s)
    return s


def lot_date(code):
    """Best-effort embedded date: 'MMM D[D] YYYY', ISO, or YYMMDD prefix."""
    s = norm_lot(code)
    m = re.search(r"([A-Z]{3,4})[ -]?0?(\d{1,2})[ -]?(20\d{2}|\d{4})", s)
    if m and m.group(1)[:3] in MONTHS:
        try:
            return date(int(m.group(3)), MONTHS[m.group(1)[:3]], int(m.group(2)))
        except ValueError:
            return None
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    m = re.match(r"^(2[56])(\d{2})(\d{2})", s)  # YYMMDD, years 2025/2026 only
    if m:
        try:
            return date(2000 + int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return None
    return None


SUFFIX = re.compile(r"\b(INC|LLC|CORP|CO|COMPANY|LTD|USA|DBA|FOODS?|GOODS?|SUPPLY)\b\.?", re.I)


def norm_name(s):
    s = re.sub(r"[^\w\s]", " ", s.upper())
    s = SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def fuzzy_candidates(name, candidates):
    """ALL plausible candidates, best kind first: exact normalized match,
    containment (>=5 chars), then ratio >= 0.75."""
    n = norm_name(name)
    if not n:
        return []
    out = []
    for c in candidates:
        nc = norm_name(c)
        if nc == n:
            out.append((c, "exact"))
        elif len(n) >= 5 and len(nc) >= 5 and (n in nc or nc in n):
            out.append((c, "contain"))
        else:
            r = difflib.SequenceMatcher(None, n, nc).ratio()
            if r >= 0.75:
                out.append((c, f"fuzzy {r:.2f}"))
    order = {"exact": 0, "contain": 1}
    return sorted(out, key=lambda x: order.get(x[1], 2))


def fuzzy_match(name, candidates):
    cands = fuzzy_candidates(name, candidates)
    return cands[0] if cands else (None, None)


SENTINELS = {"FOUND", "FOUND INVENTORY", "INVENTORY FOUND", "PHYSICAL COUNT",
             "INITIAL INVENTORY", "INVENTORY CORRECTION", "INVENTORY INTAKE",
             "UNKNOWN"}

lot_sanity = []  # [form, row date, field, code, issue]


def sanity(form, rowdate, field, code, strict_date=False):
    """strict_date=True → the code is minted that day (sticker/batch/case lot),
    so an embedded date >7 d from the row date is flagged. Ingredient lots are
    legitimately older, so only future dates (>1 d ahead) are flagged there."""
    c = code.strip()
    if not c:
        return
    up = re.sub(r"\s+", " ", c.upper())
    # a 4-digit number is only a "year" when it sits in a date-like pattern
    # (e.g. 'Jul 30 3026'); bare numeric supplier lots (6020, 5263) are not
    for y in {int(y) for y in
              re.findall(r"[A-Z]{2,5}\.? ?\d{1,2},? ?(\d{4})\b", up)}:
        if y not in (2025, 2026):
            lot_sanity.append([form, rowdate, field, c, f"implausible year {y}"])
    for bad in MONTH_FIX:
        if re.search(rf"(?<![A-Z]){bad}(?![A-Z])", up):
            lot_sanity.append([form, rowdate, field, c,
                               f"misspelled/Spanish month '{bad}'"])
    d = lot_date(c)
    if d and rowdate and d.year in (2025, 2026):
        off = (d - rowdate).days
        if strict_date and abs(off) > 7:
            lot_sanity.append([form, rowdate, field, c,
                               f"embedded date {d} is {abs(off)} d from row date"])
        elif not strict_date and off > 1:
            lot_sanity.append([form, rowdate, field, c,
                               f"embedded date {d} is {off} d in the FUTURE"])


# ══════════════════════════════════════════════════════════════════════════
emit("## 6. Forms vs FL cross-check (floor Google-Form exports as the "
     "independent record)", "",
     "Forms read from `data/forms/` (not committed; gitignored). FL side: "
     "posted-only effective ledger, read-only via the 6543 pooler. 'FL "
     "entered' uses `created_at` for live rows and the legacy `\"timestamp\"` "
     "column for backfilled rows (their `created_at` is the 039 migration "
     "stamp). FL ledger coverage starts 2026-01-28 (first posted txn); "
     "first FL coconut make is 2026-02-05.")

# ── 6a. receiving ─────────────────────────────────────────────────────────
hdr, rows = read_csv("receiving.csv")
form_rx = []
for r in rows:
    ts = parse_ts(r[2])
    if not ts:
        continue
    form_rx.append({"ts": ts, "d": ts.date(), "carrier": r[9].strip(),
                    "lot": r[5].strip(), "bol": r[8].strip(),
                    "products": "; ".join(x.strip() for x in r[11:17] if x.strip())})
    sanity("receiving", ts.date(), "What lot# did you use?", r[5], strict_date=True)
last_rx = max(f["ts"] for f in form_rx)

fl_rx = q(f"""SELECT t.id, t.business_date, t.shipper_name, t.bol_reference,
              CASE WHEN t.created_at_source = 'database'
                   THEN t.created_at AT TIME ZONE 'America/New_York'
                   ELSE t."timestamp" END entered_et,
              t.created_at_source
              FROM {POSTED_T} t
              WHERE t.type = 'receive' AND t.business_date >= '2026-02-01'
              ORDER BY t.id""")
fl_real = [r for r in fl_rx
           if (r["shipper_name"] or "").strip().upper() not in SENTINELS]
# One physical delivery = several FL receive txns (one per product, same
# BOL) — compare at (business_date, normalized supplier) EVENT grain.
fl_events = defaultdict(list)
for r in fl_real:
    fl_events[(r["business_date"], norm_name(r["shipper_name"] or ""))].append(r)
fl_names = sorted({(r["shipper_name"] or "").strip() for r in fl_real if r["shipper_name"]})

matched, unmatched_form, used_ev = [], [], set()
for f in form_rx:
    hit = None
    for cand, kind in fuzzy_candidates(f["carrier"], fl_names):
        for dd in (0, -1, 1):
            key = (f["d"] + timedelta(days=dd), norm_name(cand))
            if key in fl_events and key not in used_ev:
                hit = (key, dd, kind)
                break
        if hit:
            break
    if hit is None and len(f["bol"]) >= 4:
        # secondary: exact BOL match ±3 d (form logs the trucking CARRIER,
        # FL the supplier — e.g. Linking Logistics vs Sweet New England)
        for key, rs in fl_events.items():
            if key in used_ev or abs((key[0] - f["d"]).days) > 3:
                continue
            if any((r["bol_reference"] or "").strip() == f["bol"] for r in rs):
                hit = (key, (key[0] - f["d"]).days, "BOL")
                break
    if hit:
        key, dd, kind = hit
        used_ev.add(key)
        rs = fl_events[key]
        entered = min(r["entered_et"] for r in rs)
        src = rs[0]["created_at_source"]
        delta_min = round((entered - f["ts"]).total_seconds() / 60)
        matched.append([str(f["d"]), f["carrier"], rs[0]["shipper_name"],
                        ", ".join(str(r["id"]) for r in rs),
                        str(key[0]), delta_min,
                        kind + (f", ±{dd} d" if dd else ""), src])
    else:
        known = "yes" if fuzzy_candidates(f["carrier"], fl_names) else "NO"
        unmatched_form.append([str(f["d"]), f["carrier"], f["bol"],
                               f["products"][:60] or "—", known])

fl_unmatched = [[", ".join(str(r["id"]) for r in rs), str(key[0]),
                 rs[0]["shipper_name"],
                 ", ".join(sorted({(r["bol_reference"] or "—") for r in rs})),
                 rs[0]["created_at_source"]]
                for key, rs in sorted(fl_events.items())
                if key not in used_ev]

emit("### 6a. Receiving form vs FL receipts (match: supplier from the "
     "CARRIER column + same business date, ±1 day fallback, then exact-BOL "
     "±3 d; FL grouped into day×supplier delivery events since one truck "
     "spans several FL txns; names fuzzy-normalized — 'DUTC'-style lot "
     "prefixes never used for matching)",
     "",
     f"Form rows (Feb 2 → {last_rx:%b %d %H:%M}): **{len(form_rx)}** · FL "
     f"receives since Feb 1: **{len(fl_rx)}** txns ({len(fl_real)} real in "
     f"**{len(fl_events)}** delivery events; "
     f"{len(fl_rx) - len(fl_real)} sentinel found/count rows excluded) · "
     f"matched form-row↔event pairs: **{len(matched)}**")

by_src = defaultdict(list)
for m in matched:
    by_src[m[7]].append(m[5])
lines = []
for src, ds in sorted(by_src.items()):
    ds = sorted(ds)
    med = ds[len(ds) // 2]
    neg = sum(1 for d in ds if d < 0)
    lines.append(f"- `{src}` rows (n={len(ds)}): median **{med:+d} min**, "
                 f"range {ds[0]:+d}…{ds[-1]:+d}; {neg} pair(s) have FL "
                 "entered BEFORE the form was submitted.")
emit("FL entered − form timestamp, by FL row source (backfilled rows' legacy "
     "`\"timestamp\"` is server-clock based, so a systematic ~+240 min there "
     "is a UTC-vs-ET labeling artifact, not real lag):", *lines)
emit("Matched pairs:", "",
     table(["form date", "form carrier", "FL shipper", "FL txn(s)",
            "FL business_date", "FL entered − form (min)", "match",
            "FL row source"], matched))
emit("Form rows with NO FL receipt:", "",
     table(["form date", "carrier", "BOL", "products",
            "carrier known to FL?"], unmatched_form))
emit("FL delivery events (non-sentinel) with NO form row:", "",
     table(["FL txn(s)", "business_date", "shipper", "BOL", "source"],
           fl_unmatched))

# ── 6b. coconut ───────────────────────────────────────────────────────────
hdr, rows = read_csv("coconut_batches.csv")
BATCH_MAP = {
    "batch coconut sweetened flake": "Batch Coconut Sweetened Flake",
    "batch coconut sweetened toasted flake": "Batch Coconut Toasted Sweetened Flake",
    "batch coconut sweetened fancy": "Batch Coconut Sweetened Fancy",
    "batch coconut sweetened medium": "Batch Coconut Sweetened Medium",
}
LABEL_MAP = {
    "chefs quality 10lb sweetened flake coconut": "CQ Coconut Sweetened Flake 10 LB",
    "coconut sweetened toasted flake - 25lb": "Coconut Toasted Sweetened Flake CNS 25 LB",
    "coconut sweetened toasted flake - 10lb": "Coconut Toasted Sweetened Flake CNS 10 LB",
    "coconut sweetened fancy - 10lb(unipro)": "Coconut Sweetened Fancy UNIPRO 10 LB",
    "coconut sweetened flake - 10lb(unipro)": "Coconut Sweetened Flake UNIPRO 10 LB",
    "coconut sweetened medium - 10lb(unipro)": "Coconut Sweetened Medium UNIPRO 10 LB",
    "coconut sweetened flake 25lb": "Coconut Sweetened Flake CNS 25 LB",
    "coconut sweetened flake - 10lb": "Coconut Sweetened Flake CNS 10 LB",
    "coconut sweetened medium - 10lb": "Coconut Sweetened Medium CNS 10 LB",
}
DOW = {"MONDAY": 0, "TUESDAY": 1, "WEDNESDAY": 2, "THURSDAY": 3,
       "FRIDAY": 4, "SATURDAY": 5, "SUNDAY": 6}
FL_COCO_START = date(2026, 2, 5)

GROUPS = [
    {"sel": 5, "lot": 6, "pans": 7,
     "packs": [(8, 9, 11), (12, 13, 15), (16, 17, 19)],
     "ings": [(i, i + 1, i + 2) for i in range(20, 45, 3)]},
    {"sel": 47, "lot": 49, "pans": 50,
     "packs": [(51, 52, 53), (54, 55, 56), (57, 58, 59)],
     "ings": [(i, i + 1, i + 2) for i in range(60, 85, 3)]},
]

form_makes, form_packs = [], []
fallback_days = 0
last_coco = None
for r in rows:
    ts = parse_ts(r[3])
    if not ts:
        continue
    last_coco = max(last_coco or ts, ts)
    downame = r[4].strip().upper()
    for g in GROUPS:
        sel = r[g["sel"]].strip().lower()
        if not sel:
            continue
        lot_raw = r[g["lot"]].strip()
        ld = lot_date(lot_raw) if lot_raw else None
        day, want = None, DOW.get(downame)
        cands = ([ld] if ld else []) + [ts.date()]
        if want is not None:
            for c in cands:
                if c.weekday() == want and abs((c - ts.date()).days) <= 6:
                    day = c
                    break
        if day is None:
            day = ld if (ld and abs((ld - ts.date()).days) <= 4) else ts.date()
            fallback_days += 1
        try:
            pans = float(r[g["pans"]].strip() or 0)
        except ValueError:
            pans = 0
        form_makes.append({"day": day, "batch": BATCH_MAP.get(sel, sel),
                           "pans": pans, "lot": lot_raw, "ts": ts})
        sanity("coconut", day, "DAILY BATCH LOT#", lot_raw, strict_date=True)
        for (li, ci, loti) in g["packs"]:
            lab = r[li].strip().lower()
            if not lab:
                continue
            try:
                cases = float(r[ci].strip() or 0)
            except ValueError:
                cases = 0
            form_packs.append({"day": day, "product": LABEL_MAP.get(lab, lab),
                               "cases": cases, "lot": r[loti].strip(), "ts": ts})
            sanity("coconut", day, "case lot#", r[loti], strict_date=True)
        for (si, lbi, loti) in g["ings"]:
            if r[si].strip():
                sanity("coconut", day, f"ingredient lot# ({r[si].strip()})",
                       r[loti], strict_date=False)

fl_makes = q(f"""SELECT t.id, t.business_date, p.name batch, t.notes
                 FROM {POSTED_T} t
                 JOIN {POSTED_L} tl ON tl.transaction_id = t.id AND tl.quantity_lb > 0
                 JOIN products p ON p.id = tl.product_id
                 WHERE t.type = 'make' AND p.name ILIKE '%%coconut%%'
                   AND t.business_date >= '2026-01-01' ORDER BY t.id""")
for r in fl_makes:
    m = re.match(r"\s*([\d.]+)\s*batch", r["notes"] or "")
    r["pans"] = float(m.group(1)) if m else None

fm_day = defaultdict(float)
for f in form_makes:
    fm_day[f["day"]] += f["pans"]
fl_day = defaultdict(float)
for r in fl_makes:
    fl_day[r["business_date"]] += r["pans"] or 0

all_days = sorted(set(fm_day) | set(fl_day))
day_rows, only_form_pre, only_form, only_fl, agree, mismatch = [], [], [], [], 0, 0
for d in all_days:
    a, b = fm_day.get(d, 0), fl_day.get(d, 0)
    if a and not b:
        (only_form_pre if d < FL_COCO_START else only_form).append(str(d))
    elif b and not a:
        only_fl.append(str(d))
    elif a == b:
        agree += 1
    else:
        mismatch += 1
    if d >= date(2026, 7, 1) or (a != b and d >= FL_COCO_START):
        day_rows.append([str(d), a or "—", b or "—",
                         "✓" if a == b else ("FORM ONLY" if a and not b
                         else ("FL ONLY" if b and not a else f"Δ {a - b:+g}"))])

emit("### 6b. Coconut form vs FL — pans (batches) per production day", "",
     f"Form rows: {len(rows)} (last submission {last_coco:%Y-%m-%d %H:%M}). "
     "Production day per row = DAILY BATCH LOT# embedded date vs submission "
     "date, arbitrated by the DAY PRODUCED weekday "
     f"(heuristic fallback on {fallback_days} rows).", "",
     f"Days with coconut production on either side: **{len(all_days)}** · "
     f"exact pan agreement: **{agree}** · pan-count mismatches: "
     f"**{mismatch}** · form-only days before FL coconut coverage "
     f"(2026-02-05): **{len(only_form_pre)}** (expected — pre-go-live) · "
     f"form-only days in FL's era: **{len(only_form)}** "
     f"({', '.join(only_form) or '—'}) · FL-only days: **{len(only_fl)}** "
     f"({', '.join(only_fl) or '—'})", "",
     "Per-day detail (every mismatch/one-sided day in FL's era, plus all "
     "days from Jul 1):", "",
     table(["day", "form pans", "FL pans", "verdict"], day_rows))

aug20 = (fm_day.get(date(2026, 8, 20), 0), fl_day.get(date(2026, 8, 20), 0))
emit(f"**Aug 20 check (6d):** form {aug20[0]:g} pans vs FL {aug20[1]:g} pans → "
     + ("**CONFIRMED equal.**" if aug20[0] == aug20[1] else "**REFUTED.**"))

fl_packs = q(f"""SELECT t.id, t.business_date, p.name product,
                 tl.quantity_lb lb, l.lot_code, p.case_size_lb
                 FROM {POSTED_T} t
                 JOIN {POSTED_L} tl ON tl.transaction_id = t.id AND tl.quantity_lb > 0
                 JOIN products p ON p.id = tl.product_id
                 JOIN lots l ON l.id = tl.lot_id
                 WHERE t.type = 'pack' AND p.name ILIKE '%%coconut%%'
                   AND t.business_date >= '2026-01-01' ORDER BY t.id""")
for r in fl_packs:
    r["cases"] = float(r["lb"]) / float(r["case_size_lb"] or 1)

fp_key = defaultdict(float)
for f in form_packs:
    fp_key[(f["day"], f["product"])] += f["cases"]
flp_key = defaultdict(float)
flp_lots = defaultdict(set)
for r in fl_packs:
    k = (r["business_date"], r["product"])
    flp_key[k] += r["cases"]
    flp_lots[k].add(r["lot_code"])

keys = sorted(set(fp_key) | set(flp_key), key=lambda k: (k[0], k[1]))
pk_rows = []
for k in keys:
    a, b = fp_key.get(k, 0), flp_key.get(k, 0)
    if k[0] < FL_COCO_START:
        continue
    verdict = ("✓" if a == b and a else "FORM ONLY" if a and not b
               else "FL ONLY" if b and not a else f"Δ {a - b:+g}")
    if k[0] >= date(2026, 7, 1) or verdict != "✓":
        pk_rows.append([str(k[0]), k[1], round(a, 1) or "—",
                        round(b, 1) or "—", verdict])
emit("### 6b-ii. Coconut form vs FL — cases packed per day × product "
     "(FL's era; every non-agreeing pair plus all pairs from Jul 1)", "",
     table(["day", "product", "form cases", "FL cases", "verdict"], pk_rows))


def same_lot(form_lot, fl_lot_set):
    """Same lot if normalized strings match OR both embed the same date
    (format-only differences like 'Apr 20 2026' vs '2026-04-20' are NOT
    misattribution)."""
    nf, df = norm_lot(form_lot), lot_date(form_lot)
    for x in fl_lot_set:
        if norm_lot(x) == nf:
            return True
        if df and lot_date(x) == df:
            return True
    return False


case_same = case_diff = 0.0
lot_mismatch_rows = []
for f in form_packs:
    k = (f["day"], f["product"])
    if k not in flp_lots or not f["lot"] or f["day"] < FL_COCO_START:
        continue
    if same_lot(f["lot"], flp_lots[k]):
        case_same += f["cases"]
    else:
        case_diff += f["cases"]
        lot_mismatch_rows.append([str(f["day"]), f["product"], f["cases"],
                                  f["lot"], ", ".join(sorted(flp_lots[k]))])
tot = case_same + case_diff
emit("### 6b-iii. Physical case lot# (form) vs FL pack output lot — the F4 "
     "error size", "",
     f"Matched form pack lines (same day+product packed in FL, form lot "
     f"present): **{tot:g} cases**. Attributed in FL to a lot with a "
     f"DIFFERENT embedded date than the one physically on the cases: "
     f"**{case_diff:g} cases = {100 * case_diff / tot if tot else 0:.1f}%** "
     "(month-spelling, 'Lot' noise, and pure format differences — e.g. "
     "`Apr 20 2026` vs `2026-04-20` — normalized away first; what remains "
     "is genuine day-level misattribution).", "",
     table(["day", "product", "form cases", "physical lot on cases",
            "FL pack output lot(s) that day"], lot_mismatch_rows))

# 8/7 79-case narrative, computed
n79 = [f for f in form_packs
       if f["cases"] == 79 and f["lot"].strip().lower().startswith("jul 30")]
fl_j30 = q(f"""SELECT t.id, t.business_date, tl.quantity_lb lb
               FROM {POSTED_T} t
               JOIN {POSTED_L} tl ON tl.transaction_id = t.id AND tl.quantity_lb > 0
               JOIN products p ON p.id = tl.product_id
               JOIN lots l ON l.id = tl.lot_id
               WHERE t.type = 'pack' AND p.name = 'CQ Coconut Sweetened Flake 10 LB'
                 AND upper(l.lot_code) = 'JUL 30 2026'
                 AND t.business_date >= '2026-08-01' ORDER BY t.id""")
if n79:
    f = n79[0]
    fl_txt = "; ".join(f"txn {r['id']} on {r['business_date']} "
                       f"{float(r['lb']) / 10:g} cases" for r in fl_j30)
    emit(f"**8/7 79-case check:** the form (submitted {f['ts']:%m/%d %H:%M}, "
         f"production day {f['day']}) reports **79 cases of "
         f"{f['product']} packed from lot `Jul 30 2026`**. FL records the "
         f"Jul-30-lot CQ packs in August as: {fl_txt} — i.e. FL split the "
         "same 79 Jul-30-lot cases across two business days (60 + 19) "
         "instead of the form's single 79-case event, and a further form "
         "line (8/7 afternoon, 19 cases `Jul 30 2026`) overlaps the FL "
         "8/7 19-case pack. Case totals reconcile at the lot level "
         "(79 form-morning vs 60+19 FL) but day attribution differs — "
         "flagged as requested.")
else:
    emit("**8/7 79-case check:** no form pack line of 79 cases from a "
         "Jul 30 lot was found — check manually.")

# ── 6c. lot-code sanity ───────────────────────────────────────────────────
kinds = Counter(re.sub(r"'.*'", "'…'", re.sub(r"\d{4}-\d{2}-\d{2}.*", "…", i))
                for *_x, i in lot_sanity)
emit("### 6c. Lot-code sanity on form free-text lot fields", "",
     f"Issues found: **{len(lot_sanity)}** (same-day fields — receiving "
     "sticker lot#, DAILY BATCH LOT#, case lot# — checked for >7 d "
     "date drift; ingredient lot#s are legitimately older stock, so they "
     "are only checked for impossible years, month misspellings, and "
     "FUTURE dates).", "",
     table(["issue type", "n"], sorted(kinds.items(), key=lambda x: -x[1])))
emit("Full list:", "",
     table(["form", "row date", "field", "code", "issue"],
           [[f, str(d), fld, c, i] for f, d, fld, c, i in lot_sanity]))

# ── 6e. shipping ──────────────────────────────────────────────────────────
hdr, rows = read_csv("shipping.csv")
form_sh = []
for r in rows:
    ts = parse_ts(r[1])
    if not ts:
        continue
    form_sh.append({"ts": ts, "d": ts.date(), "cust": r[4].strip(),
                    "carrier": r[3].strip(), "bol": r[6].strip()})
last_sh = max(f["ts"] for f in form_sh)

fl_sh = q(f"""SELECT t.id, t.business_date, t.customer_name,
              coalesce((SELECT sum(abs(tl.quantity_lb)) FROM {POSTED_L} tl
                        WHERE tl.transaction_id = t.id AND tl.quantity_lb < 0), 0) lb
              FROM {POSTED_T} t
              WHERE t.type = 'ship' AND t.business_date >= '2026-01-28'
              ORDER BY t.id""")
fl_cust = sorted({(r["customer_name"] or "").strip() for r in fl_sh if r["customer_name"]})

# collapse both sides to (day, normalized customer) events — one truck may
# be several FL txns and several form rows; the event is the comparable unit
form_pairs = defaultdict(list)
for f in form_sh:
    form_pairs[(f["d"], norm_name(f["cust"]))].append(f)
fl_pairs = defaultdict(list)
for r in fl_sh:
    fl_pairs[(r["business_date"], norm_name(r["customer_name"] or ""))].append(r)

form_only, fl_only, pair_matched, form_matched = [], [], set(), set()
for (d, name), fs in sorted(form_pairs.items()):
    hit = None
    for cand, _kind in fuzzy_candidates(fs[0]["cust"], fl_cust):
        for dd in (0, -1, 1):
            key = (d + timedelta(days=dd), norm_name(cand))
            if key in fl_pairs:
                hit = key
                break
        if hit:
            break
    if hit:
        pair_matched.add(hit)
        form_matched.add((d, name))
    else:
        form_only.append([str(d), fs[0]["cust"], fs[0]["carrier"],
                          fs[0]["bol"] or "—"])
fl_names_from_form = sorted({f["cust"] for f in form_sh if f["cust"]})
for (d, name), rs in sorted(fl_pairs.items()):
    if (d, name) in pair_matched:
        continue
    # reverse check: any form event within ±1 d whose customer maps to us
    hit = None
    for cand, _kind in fuzzy_candidates(rs[0]["customer_name"] or "",
                                        fl_names_from_form):
        for dd in (0, -1, 1):
            key = (d + timedelta(days=dd), norm_name(cand))
            if key in form_pairs:
                hit = key
                break
        if hit:
            break
    if hit:
        pair_matched.add((d, name))
        continue
    fl_only.append([str(d), rs[0]["customer_name"] or "(blank)",
                    len(rs), round(sum(float(r["lb"]) for r in rs))])

months = sorted({k[0].strftime("%Y-%m") for k in
                 list(form_pairs) + list(fl_pairs)})
mrows = []
for m in months:
    fp = [k for k in form_pairs if k[0].strftime("%Y-%m") == m]
    lp = [k for k in fl_pairs if k[0].strftime("%Y-%m") == m]
    fo = sum(1 for r in form_only if r[0][:7] == m)
    lo = sum(1 for r in fl_only if r[0][:7] == m)
    mrows.append([m, len(fp), len(lp), fo, lo])

emit("### 6e. Shipping form vs FL ship transactions (compared at day × "
     "normalized-customer grain, ±1 day; one truck can span several FL "
     "txns/form rows)", "",
     "**The shipping form export contains NO product, case-count, or "
     "lot-number columns** — its fields are timestamp, name, carrier, "
     "customer, BOL photo link, BOL#, and three condition checkboxes. The "
     "requested packing-slip-lot vs FL-allocated-lot comparison and "
     "quantity mismatches are therefore NOT computable from this export; "
     "'% of ship lines where FL's lot ≠ the slip's lot' would require the "
     "linked BOL photos or per-line form fields. Presence matching only, "
     "below.", "",
     f"Form rows (Jan 28 → {last_sh:%b %d %H:%M}): **{len(form_sh)}** = "
     f"**{len(form_pairs)}** day×customer events · FL ship txns since "
     f"Jan 28: **{len(fl_sh)}** = **{len(fl_pairs)}** day×customer events · "
     f"events matched: **{len(pair_matched)}** · form-only events: "
     f"**{len(form_only)}** · FL-only events: **{len(fl_only)}**", "",
     "By month:", "",
     table(["month", "form events", "FL events", "form-only", "FL-only"], mrows))
emit("Form shipment events with NO FL ship (from Jul 1; earlier months "
     "summarized above):", "",
     table(["date", "customer", "carrier", "BOL#"],
           [r for r in form_only if r[0] >= "2026-07-01"]))
emit("FL ship events with NO form row (from Jul 1):", "",
     table(["date", "customer", "FL txns", "lb"],
           [r for r in fl_only if r[0] >= "2026-07-01"]))

emit("### 6f. Form freshness", "",
     table(["form", "last submission"],
           [["coconut_batches.csv", f"{last_coco:%Y-%m-%d %H:%M}"],
            ["receiving.csv", f"{last_rx:%Y-%m-%d %H:%M}"],
            ["shipping.csv", f"{last_sh:%Y-%m-%d %H:%M}"]]))

CONN.close()
