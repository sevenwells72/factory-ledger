#!/usr/bin/env python3
"""One-shot inventory recon poster. Invoked only after explicit 'go'.
Posts the confirmed 2026-08-17 checklist in one DB transaction and rolls back
if the post-write on-hand check fails."""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

import psycopg2
from psycopg2.extras import RealDictCursor

NY = ZoneInfo("America/New_York")
NOTE = "INV-RECON-2026-08-17"
COUNT_REF = "physical-count-2026-08-14.md"
OPERATOR = "inv-recon-2026-08-17"
LOG_PATH = "/tmp/inv_recon_post_2026_08_17.json"

INV_70003 = (
    "28259, 28260, 28263, 28266, 28268, 28265, 28183, 28181, "
    "28276, 28277, 28323, 28295, 28297, 28320, 28321, 28322, 28257"
)


def D(x) -> Decimal:
    return Decimal(str(x))


def plant_noon(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, 12, 0, 0, tzinfo=NY)


class Poster:
    def __init__(self, cur):
        self.cur = cur
        self.log = []
        self.products = {}

    def sku(self, code: str) -> dict:
        if code not in self.products:
            self.cur.execute(
                "SELECT id, odoo_code, name, case_size_lb, type FROM products WHERE odoo_code = %s",
                (code,),
            )
            row = self.cur.fetchone()
            if not row:
                raise RuntimeError(f"SKU {code} not found")
            self.products[code] = dict(row)
        return self.products[code]

    def lot(self, product_id: int, lot_code: str, entry_source: str = "found_inventory") -> int:
        self.cur.execute(
            """
            INSERT INTO lots (product_id, lot_code, entry_source, entry_source_notes)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (product_id, lot_code) DO NOTHING
            """,
            (product_id, lot_code, entry_source, NOTE),
        )
        self.cur.execute(
            "SELECT id FROM lots WHERE product_id = %s AND lot_code = %s",
            (product_id, lot_code),
        )
        return self.cur.fetchone()["id"]

    def txn(self, *, typ: str, biz: date, notes: str, customer: str | None = None,
            order_ref: str | None = None, adjust_reason: str | None = None) -> int:
        occurred = plant_noon(biz)
        ts = occurred.replace(tzinfo=None)
        self.cur.execute(
            """
            INSERT INTO transactions
                (type, timestamp, notes, customer_name, order_reference,
                 adjust_reason, occurred_at, business_date, operator_id, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'posted')
            RETURNING id
            """,
            (typ, ts, notes, customer, order_ref, adjust_reason, occurred, biz, OPERATOR),
        )
        return self.cur.fetchone()["id"]

    def line(self, txn_id: int, product_id: int, lot_id: int, qty: Decimal):
        self.cur.execute(
            """
            INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb)
            VALUES (%s, %s, %s, %s)
            """,
            (txn_id, product_id, lot_id, qty),
        )

    def shipment(self, txn_id: int, customer_id: int, biz: date, lines: list[tuple[int, Decimal]]) -> int:
        self.cur.execute(
            """
            INSERT INTO shipments (transaction_id, shipped_at, customer_id, sales_order_id)
            VALUES (%s, %s, %s, NULL) RETURNING id
            """,
            (txn_id, plant_noon(biz), customer_id),
        )
        sid = self.cur.fetchone()["id"]
        for product_id, qty_lb in lines:
            self.cur.execute(
                """
                INSERT INTO shipment_lines (shipment_id, transaction_id, product_id, quantity_lb)
                VALUES (%s, %s, %s, %s)
                """,
                (sid, txn_id, product_id, abs(qty_lb)),
            )
        return sid

    def record(self, checklist_id: str, txn_id: int, **extra):
        rec = {"id": checklist_id, "txn_id": txn_id, **extra}
        self.log.append(rec)
        print(f"  posted {checklist_id} txn={txn_id} {extra}", flush=True)

    def on_hand(self, sku: str) -> Decimal:
        p = self.sku(sku)
        self.cur.execute(
            """
            SELECT COALESCE(SUM(l.quantity_lb), 0) AS lb
            FROM ledger_current_transaction_lines l
            JOIN ledger_current_transactions t ON t.id = l.transaction_id
            WHERE l.product_id = %s AND t.effective_status = 'posted'
            """,
            (p["id"],),
        )
        return D(self.cur.fetchone()["lb"])

    def lot_oh(self, sku: str, lot_code: str) -> Decimal:
        p = self.sku(sku)
        self.cur.execute(
            """
            SELECT COALESCE(SUM(l.quantity_lb), 0) AS lb
            FROM ledger_current_transaction_lines l
            JOIN ledger_current_transactions t ON t.id = l.transaction_id
            JOIN lots lt ON lt.id = l.lot_id
            WHERE l.product_id = %s AND lt.lot_code = %s AND t.effective_status = 'posted'
            """,
            (p["id"], lot_code),
        )
        return D(self.cur.fetchone()["lb"])


def adjust(p: Poster, cid: str, sku: str, lot: str, qty: Decimal, biz: date, reason: str,
           entry_source: str = "found_inventory"):
    prod = p.sku(sku)
    lot_id = p.lot(prod["id"], lot, entry_source)
    before = p.lot_oh(sku, lot)
    after = before + qty
    if after < D("-0.0001"):
        raise RuntimeError(f"{cid} would take {sku}/{lot} {before} → {after}")
    tid = p.txn(
        typ="adjust",
        biz=biz,
        notes=f"{NOTE} | {reason}",
        adjust_reason=reason,
    )
    p.line(tid, prod["id"], lot_id, qty)
    p.record(cid, tid, sku=sku, lot=lot, qty=str(qty), type="adjust")
    return tid


def pack_bulk(p: Poster, cid: str, src_sku: str, dst_sku: str, dst_lot: str,
              sources: list[tuple[str, Decimal]], biz: date, reason: str):
    src = p.sku(src_sku)
    dst = p.sku(dst_sku)
    total = sum((q for _, q in sources), D(0))
    dst_lot_id = p.lot(dst["id"], dst_lot, "pack_output")
    tid = p.txn(typ="pack", biz=biz, notes=f"{NOTE} | {reason}")
    for lot_code, qty in sources:
        if p.lot_oh(src_sku, lot_code) + D("0.0001") < qty:
            raise RuntimeError(f"{cid} {src_sku}/{lot_code} short for {qty}")
        p.line(tid, src["id"], p.lot(src["id"], lot_code), -qty)
    p.line(tid, dst["id"], dst_lot_id, total)
    p.record(cid, tid, sku=dst_sku, lot=dst_lot, qty=str(total), type="pack", sources=len(sources))
    return tid


def ship_group(p: Poster, cid: str, customer: str, customer_id: int, order_ref: str,
               biz: date, lines: list[tuple[str, str, Decimal]], notes: str):
    tid = p.txn(
        typ="ship",
        biz=biz,
        notes=f"{NOTE} | {notes}",
        customer=customer,
        order_ref=order_ref,
    )
    ship_lines = []
    for sku, lot, qty in lines:
        if qty >= 0:
            raise RuntimeError("ship qty must be negative")
        prod = p.sku(sku)
        available = p.lot_oh(sku, lot)
        if available + qty < D("-0.0001"):
            raise RuntimeError(f"{cid} {sku}/{lot} have {available} need {-qty}")
        p.line(tid, prod["id"], p.lot(prod["id"], lot), qty)
        ship_lines.append((prod["id"], qty))
    sid = p.shipment(tid, customer_id, biz, ship_lines)
    p.record(cid, tid, shipment_id=sid, type="ship", n_lines=len(lines), ref=order_ref)
    return tid


def preflight(p: Poster):
    expected = {
        "70003": D("29535"),
        "70002": D("6180"),
        "70010": D("975"),
        "70070": D("1050"),
        "70011": D("1342.5"),
        "70073": D("3613.62"),
        "70074": D("4902.32"),
        "70080": D("4614.74"),
        "67470": D("1190"),
        "67473": D("620"),
        "67476": D("3170"),
        "90002": D("34316"),
        "90001": D("17748"),
        "90016": D("1620"),
        "95005": D("2.38"),
    }
    for sku, exp in expected.items():
        got = p.on_hand(sku)
        if abs(got - exp) > D("0.001"):
            raise RuntimeError(f"preflight abort: {sku} on-hand {got} != {exp}")
    print("preflight on-hand OK", flush=True)


def expected_after() -> dict[str, Decimal]:
    return {
        "70003": D("4500"),
        "70002": D("1500"),
        "70010": D("0"),
        "70070": D("0"),
        "70011": D("0"),
        "70073": D("0"),
        "70074": D("0"),
        "70080": D("0"),
        "70013": D("0"),
        "70004": D("0"),
        "70050": D("450"),
        "70082": D("0"),
        "70059": D("0"),
        "70052": D("0"),
        "1614": D("0"),
        "31012": D("2800"),
        "10300": D("0"),
        "893": D("7000"),
        "67476": D("3170"),
        "67470": D("260"),
        "10020": D("0"),
        "10029": D("1250"),
        "67473": D("600"),
        "10002": D("160"),
        "10001": D("0"),
        "10007": D("0"),
        "10010": D("60"),
        "10006": D("0"),
        "15999": D("1"),
        "90002": D("10013"),
        "90001": D("0"),
        "90011": D("3930"),
        "95002": D("1050"),
        "90016": D("0"),
        "90020": D("0"),
        "90024": D("0"),
        "90010": D("0"),
        "90019": D("0"),
        "90015": D("0"),
        "90013": D("379"),
        "95005": D("2.38"),
        "70088": D("2406"),
    }


def post_all(p: Poster):
    d0512 = date(2026, 5, 12)
    d0724 = date(2026, 7, 24)
    d0729 = date(2026, 7, 29)
    d0730 = date(2026, 7, 30)
    d0814 = date(2026, 8, 14)

    # C1 product
    p.cur.execute(
        """
        INSERT INTO products
            (name, type, odoo_code, uom, brand, storage_type, verification_status,
             verification_notes, created_via, active, label_type, parent_batch_product_id)
        VALUES
            (%s, 'ingredient', '15999', 'container', 'Blue Stripes', 'ambient', 'verified',
             %s, %s, true, 'house',
             (SELECT id FROM products WHERE odoo_code = '95005'))
        RETURNING id, odoo_code, name, case_size_lb, type
        """,
        (
            "WIP Banana / PB-chip mix (for PBB)",
            "1 container counted 2026-08-14; usable as inclusions in a future PBB /make; "
            "not granola. 95005 leftover stays. " + NOTE,
            NOTE,
        ),
    )
    row = dict(p.cur.fetchone())
    p.products["15999"] = row
    p.record("C1", 0, sku="15999", product_id=row["id"], type="product_create")

    # C2
    adjust(
        p, "C2", "15999", "AUG 14 MIX", D("1.00"), d0814,
        f"{COUNT_REF} — 1 container token (not a scale weight); usable in future PBB run",
        entry_source="found_inventory",
    )

    # Phase 1
    adjust(p, "1.1", "70073", "SW2620891", D("2627.37"), d0729,
           "28337-I GRPB under-pack 999 cs")
    adjust(p, "1.2", "70074", "SW2620590", D("149.91"), d0724,
           "28337-I GRDC under-pack 57 cs")
    adjust(p, "1.3", "70074", "SW2607890", D("165.69"), d0512,
           "28220-I GRDC under-pack 63 cs")
    adjust(p, "1.4", "70010", "BB070827", D("1522.50"), d0814,
           "QB Low Carb under-pack 203 cs (964 billed)")
    adjust(p, "1.5", "70070", "BB070827", D("1567.50"), d0814,
           "QB Low Carb under-pack 209 cs (964 billed)")
    adjust(p, "1.6", "70011", "BB070827", D("780.00"), d0814,
           "QB Cranberry under-pack 104 cs (283 billed)")
    adjust(p, "1.7", "90016", "JUN 24 2026", D("255.00"), d0814,
           "QB #1 Bulk 1875 vs ledger 1620 under-record")

    # Phase 2 ships
    ship_group(
        p, "2.1-2.4", "Blue Stripes", 17, "28220-I", d0512,
        [
            ("70080", "SW2612692", D("-4581.46")),
            ("70074", "SW2611090", D("-2651.04")),
            ("70074", "MAR 20 2026", D("-1115.12")),
            ("70074", "SW2607890", D("-165.69")),
        ],
        "Blue Stripes invoice 28220-I; standalone; not SO-260629-003",
    )
    ship_group(
        p, "2.5-2.6", "Blue Stripes", 17, "28337-I", d0730,
        [
            ("70073", "SW2620891", D("-6240.99")),
            ("70074", "SW2620590", D("-1286.07")),
        ],
        "Blue Stripes invoice 28337-I; standalone; not SO-260629-003",
    )
    ship_group(
        p, "2.7-2.14", "Sunshine Granola", 217, "QB-70003-YTD-NET", d0814,
        [
            ("70003", "BB060827", D("-2820.00")),
            ("70003", "BB061527", D("-3712.50")),
            ("70003", "BB062227", D("-4162.50")),
            ("70003", "BB070827", D("-1125.00")),
            ("70003", "BB071327", D("-2865.00")),
            ("70003", "BB072027", D("-3375.00")),
            ("70003", "BB080327", D("-4177.50")),
            ("70003", "BB081027", D("-3090.00")),
        ],
        f"YTD billed 8016 cs minus SUNSHINE-RECON-2026 4639 cs. Invoices: {INV_70003}. "
        "Do not void recon. force_standalone; not SO-260814-002.",
    )
    ship_group(
        p, "2.15", "Sunshine Granola", 217, "QB-70002-YTD-NET", d0814,
        [("70002", "BB061527", D("-112.50"))],
        "YTD billed ~915 minus recon 900. force_standalone; not SO-260814-002.",
    )
    ship_group(
        p, "2.16-2.17", "Sunshine Granola", 217, "QB-LOWCARB-YTD-NET", d0814,
        [
            ("70010", "BB070827", D("-2497.50")),
            ("70070", "BB070827", D("-2617.50")),
        ],
        "QB Low Carb 964 cs allocated 475/489; recon already 142/140. force_standalone.",
    )
    ship_group(
        p, "2.18", "Sunshine Granola", 217, "QB-70011-YTD", d0814,
        [("70011", "BB070827", D("-2122.50"))],
        "QB Cranberry 283 cs. force_standalone; not SO-260814-002.",
    )

    # Phase 3.A Classic #9 → 70013
    pack_bulk(
        p, "3.A", "90002", "70013", "BULK-#9-YTD",
        [
            ("JUN 30 2026", D("224")),
            ("Jul 01 2026", D("753")),
            ("JUL 2 2026", D("429")),
            ("JUL 06 2026", D("1075")),
            ("JUL 07 2026", D("1726")),
            ("JUL 10 2026", D("9")),
            ("JUL 13 2026", D("1075")),
            ("JUL 14 2026", D("321")),
            ("JUL 15 2026", D("1938")),
            ("JUL15 2026", D("6")),
            ("JUL 16 2026", D("644")),
            ("JUL 17 2026", D("814")),
            ("JUL 21 2026", D("2122")),
            ("JUL 27 2026", D("979")),
        ],
        d0814,
        "QB Original (#9) Bulk per/lb 12,115 lb",
    )
    ship_group(
        p, "3.A16", "Sunshine Granola", 217, "QB-70013-BULK-YTD", d0814,
        [("70013", "BULK-#9-YTD", D("-12115"))],
        "QB Original (#9) Bulk per/lb 12,115 lb. force_standalone.",
    )

    # Phase 3.B #1 Bulk
    pack_bulk(
        p, "3.B", "90016", "70004", "BULK-#1-YTD",
        [
            ("JUN 24 2026", D("1305")),
            ("JUN 09 2026", D("220")),
            ("JUN 29 2026", D("100")),
            ("AUG 05 2026", D("250")),
        ],
        d0814,
        "QB #1 Bulk 1,875 lb",
    )
    ship_group(
        p, "3.B3", "Sunshine Granola", 217, "QB-70004-BULK-YTD", d0814,
        [("70004", "BULK-#1-YTD", D("-1875"))],
        "QB #1 Bulk 1,875 lb. force_standalone.",
    )

    # Phase 3.D generic consumes
    generic = [
        ("3.D1", "90002", "JUL 27 2026", D("-3543")),
        ("3.D2", "90002", "JUL 28 2026", D("-6")),
        ("3.D3", "90002", "Jul 29 2026", D("-214")),
        ("3.D4", "90002", "JUL 30 2026", D("-537")),
        ("3.D5", "90002", "JUL 31 2026", D("-1399")),
        ("3.D6", "90002", "AUG 03 2026", D("-465")),
        ("3.D7", "90002", "AUG 04 2026", D("-968")),
        ("3.D8", "90002", "AUG 10 2026", D("-8")),
        ("3.D9", "90002", "AUG 11 2026", D("-3433")),
        ("3.D10", "90002", "AUG 12 2026", D("-1615")),
        ("3.D11", "90001", "JUN 10 2026", D("-2088")),
        ("3.D12", "90001", "JUN 22 2026", D("-1740")),
        ("3.D13", "90001", "JUN 23 2026", D("-1740")),
        ("3.D14", "90001", "JUL 07 2026", D("-1392")),
        ("3.D15", "90001", "JUL 08 2026", D("-696")),
        ("3.D16", "90001", "JUL 22 2026", D("-6264")),
        ("3.D17", "90001", "AUG 03 2026", D("-3828")),
        ("3.D18", "90011", "JUL 20 2026", D("-4656")),
        ("3.D19", "90011", "AUG 05 2026", D("-4323")),
        ("3.D20", "90011", "AUG 06 2026", D("-4323")),
        ("3.D21", "95002", "JUL 23 2026", D("-263.84")),
        ("3.D22", "95002", "JUL 24 2026", D("-3150")),
        ("3.D23", "90013", "JUN 24 2026", D("-173.50")),
        ("3.D24", "90020", "JUN 03 2026", D("-2366")),
        ("3.D25", "90020", "JUN 30 2026", D("-76")),
        ("3.D26", "90020", "MAY 01 2026", D("-266")),
        ("3.D27", "90024", "26-05-06-VAN-002", D("-765")),
        ("3.D28", "90024", "JUL 21 2026", D("-15")),
        ("3.D29", "90024", "JUN 29 2026", D("-140")),
        ("3.D30", "90010", "JUN 29 2026", D("-160")),
        ("3.D31", "90010", "MAY 12 2026", D("-85")),
        ("3.D32", "90019", "JUN 30 2026", D("-60")),
        ("3.D33", "90019", "MAY 12 2026", D("-90")),
        ("3.D34", "90015", "JUN 24 2026", D("-75")),
    ]
    for cid, sku, lot, qty in generic:
        adjust(p, cid, sku, lot, qty, d0814, f"{COUNT_REF} unrecorded consumption")

    # Phase 4
    adjust(p, "4.1", "70003", "BB081027", D("292.50"), d0814,
           f"{COUNT_REF} count 600 vs 561 after billed ships")
    adjust(p, "4.2", "70002", "BB061527", D("-2137.50"), d0814,
           f"{COUNT_REF} count 200")
    adjust(p, "4.3", "70002", "BB070827", D("-2430.00"), d0814,
           f"{COUNT_REF} count 200")
    adjust(p, "4.4", "70080", "SW2612692", D("-5.26"), d0814,
           f"{COUNT_REF} BS 6x7 = 0; exact remaining 5.26 lb")
    adjust(p, "4.5", "70080", "SW2606892", D("-28.02"), d0814,
           f"{COUNT_REF} BS 6x7 = 0; exact lot remainder 28.02 lb")
    adjust(p, "4.6", "70050", "AUG 11 2026", D("-1375.00"), d0814,
           f"{COUNT_REF} clear wrong lot")
    adjust(p, "4.7", "70050", "MAY 13 2026", D("125.00"), d0814,
           f"{COUNT_REF} counted 5 cs")
    adjust(p, "4.8", "70050", "JUL 21 2026", D("325.00"), d0814,
           f"{COUNT_REF} counted 13 cs")
    adjust(p, "4.9", "70082", "JUL 21 2026", D("-750.00"), d0814, f"{COUNT_REF} count 0")
    adjust(p, "4.10", "70059", "JUN 30 2026", D("-600.00"), d0814, f"{COUNT_REF} count 0")
    adjust(p, "4.11", "70052", "JUN 29 2026", D("-350.00"), d0814, f"{COUNT_REF} count 0")
    adjust(p, "4.12", "31012", "608101", D("-1400.00"), d0814, f"{COUNT_REF} count 280 Clark")
    adjust(p, "4.13", "10300", "AUG 10 2026", D("-110.00"), d0814, f"{COUNT_REF} count 0")
    adjust(p, "4.14", "10300", "AUG 11 2026", D("-50.00"), d0814, f"{COUNT_REF} count 0")
    adjust(p, "4.15", "893", "AUG 14 2026", D("-100.00"), d0814, f"{COUNT_REF} count 700")
    adjust(p, "4.16", "67470", "AUG 13 2026", D("-930.00"), d0814, f"{COUNT_REF} count 26")
    adjust(p, "4.17", "10020", "AUG 06 2026", D("-2950.00"), d0814, f"{COUNT_REF} count 0")
    adjust(p, "4.18", "10029", "JUL 27 2026", D("-350.00"), d0814, f"{COUNT_REF} count 50")
    adjust(p, "4.19", "67473", "AUG 13 2026", D("-20.00"), d0814, f"{COUNT_REF} count 60")
    adjust(p, "4.20", "10002", "MAR 26 2026", D("-600.00"), d0814, f"{COUNT_REF} count 16 on AUG 04")
    adjust(p, "4.21", "10002", "AUG 04 2026", D("-200.00"), d0814, f"{COUNT_REF} 36 → 16")
    adjust(p, "4.22", "10007", "26-02-12-FOUND-014", D("-125.00"), d0814, f"{COUNT_REF} count 0")
    adjust(p, "4.23", "67476", "AUG 05 2026", D("-970.00"), d0814, f"{COUNT_REF} 237 → 140")
    adjust(p, "4.24", "67476", "AUG 06 2026", D("170.00"), d0814, f"{COUNT_REF} 80 → 97")
    adjust(
        p, "4.25", "67476", "STAGED-FEESERS-67476", D("800.00"), d0814,
        f"{COUNT_REF} 80 cs staged for Feesers order — actual lot dates to be recorded "
        "before ship posts (trace requirement).",
        entry_source="found_inventory",
    )


def verify(p: Poster) -> list[dict]:
    diffs = []
    for sku, exp in expected_after().items():
        got = p.on_hand(sku)
        ok = abs(got - exp) <= D("0.001")
        diffs.append({"sku": sku, "expected_lb": str(exp), "actual_lb": str(got),
                      "delta_lb": str(got - exp), "ok": ok})
        mark = "OK" if ok else "FAIL"
        print(f"  verify {sku:6} expected={exp:>12} actual={got:>12} {mark}", flush=True)
    lot_checks = [
        ("70003", "BB081027", D("4500")),
        ("70002", "BB081027", D("1500")),
        ("90002", "AUG 13 2026", D("5814")),
        ("90002", "AUG 12 2026", D("4199")),
        ("90011", "AUG 14 2026", D("3930")),
        ("95002", "JUL 24 2026", D("1050")),
        ("90013", "AUG 06 2026", D("379")),
        ("67476", "AUG 05 2026", D("1400")),
        ("67476", "AUG 06 2026", D("970")),
        ("67476", "STAGED-FEESERS-67476", D("800")),
        ("893", "AUG 14 2026", D("2800")),
        ("70050", "MAY 13 2026", D("125")),
        ("70050", "JUL 21 2026", D("325")),
        ("15999", "AUG 14 MIX", D("1")),
        ("95005", "JUL 27 2026", D("2.38")),
    ]
    for sku, lot, exp in lot_checks:
        got = p.lot_oh(sku, lot)
        ok = abs(got - exp) <= D("0.001")
        diffs.append({"sku": sku, "lot": lot, "expected_lb": str(exp),
                      "actual_lb": str(got), "delta_lb": str(got - exp), "ok": ok})
        mark = "OK" if ok else "FAIL"
        print(f"  verify {sku}/{lot} expected={exp} actual={got} {mark}", flush=True)
    return diffs


def main():
    url = os.environ["DATABASE_URL"]
    conn = psycopg2.connect(url)
    conn.autocommit = False
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SET TRANSACTION READ WRITE")
            p = Poster(cur)
            preflight(p)
            print("posting…", flush=True)
            post_all(p)
            print("verifying…", flush=True)
            diffs = verify(p)
            failed = [d for d in diffs if not d["ok"]]
            payload = {"posted": p.log, "diffs": diffs, "failed": failed}
            with open(LOG_PATH, "w") as f:
                json.dump(payload, f, indent=2, default=str)
            if failed:
                conn.rollback()
                print(f"ROLLBACK — {len(failed)} verification failures. See {LOG_PATH}", file=sys.stderr)
                sys.exit(2)
            conn.commit()
            print(f"COMMITTED {len(p.log)} entries. Log {LOG_PATH}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
