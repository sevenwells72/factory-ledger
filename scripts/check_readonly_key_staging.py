#!/usr/bin/env python3
"""G1 staging check: the read-only GPT key against the hosted FastAPI-staging service.

Sweeps every authenticated route of THIS checkout's code (the code uploaded to
FastAPI-staging) with READONLY_API_KEY:
  * each GPT read (READONLY_KEY_ALLOWLIST) must get past auth — 200, or a
    handler 404 for a lookup of a value that does not exist;
  * every other authenticated route — every write, every other GET — must
    answer 403 "API key not authorized for this endpoint";
  * POST /auth/session with the read-only key as actor_key must be refused.
Write calls carry an empty JSON body and placeholder path ids; they are refused
by the auth dependency before any handler runs, so nothing is written.

The key is read from ~/Documents/fl-secrets/staging-readonly-key.txt and never
printed. Refuses to run against anything but the staging host.

    python scripts/check_readonly_key_staging.py --base https://<staging domain> \
        [--out docs/deployments/g1-readonly-staging.json]
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
KEY_FILE = Path.home() / "Documents/fl-secrets/staging-readonly-key.txt"
PRODUCTION_HOST = "fastapi-production-b73a.up.railway.app"
NOT_AUTHORIZED = "API key not authorized for this endpoint"

# Same live calls as tests/test_readonly_api_key.py::GPT_READ_CALLS.
GPT_READ_CALLS = {
    ("POST", "/products/resolve"): ("POST", "/products/resolve", {"names": ["granola"]}),
    ("GET", "/products/search"): ("GET", "/products/search?q=granola", None),
    ("GET", "/bom/products"): ("GET", "/bom/products?limit=1", None),
    ("GET", "/bom/batches/{batch_id}/formula"): ("GET", "/bom/batches/999999999/formula", None),
    ("GET", "/inventory/lookup"): ("GET", "/inventory/lookup?q=granola", None),
    ("GET", "/lots/by-code/{lot_code}"): ("GET", "/lots/by-code/NO-SUCH-LOT", None),
    ("GET", "/lots/by-supplier-lot/{supplier_lot_code}"): ("GET", "/lots/by-supplier-lot/NO-SUCH-LOT", None),
    ("GET", "/trace/batch/{lot_code}"): ("GET", "/trace/batch/NO-SUCH-LOT", None),
    ("GET", "/trace/ingredient/{lot_code}"): ("GET", "/trace/ingredient/NO-SUCH-LOT", None),
    ("GET", "/trace/supplier-lot/{supplier_lot_code}"): ("GET", "/trace/supplier-lot/NO-SUCH-LOT", None),
    ("GET", "/transactions/history"): ("GET", "/transactions/history?limit=1", None),
    ("GET", "/production/day-summary"): ("GET", "/production/day-summary", None),
    ("GET", "/customers"): ("GET", "/customers", None),
    ("GET", "/customers/search"): ("GET", "/customers/search?q=acme", None),
    ("GET", "/sales/orders"): ("GET", "/sales/orders?limit=1", None),
    ("GET", "/sales/orders/{order_id}"): ("GET", "/sales/orders/999999999", None),
}


def _routes():
    # Import the app only to enumerate its routes: no startup, no DB.
    os.environ.pop("DATABASE_URL", None)
    import main
    from fastapi.routing import APIRoute

    auth = {main.verify_api_key, main.verify_api_key_flexible}

    def calls(dependant):
        for sub in dependant.dependencies:
            yield sub.call
            yield from calls(sub)

    routes = sorted(
        (m, r.path)
        for r in main.app.routes
        if isinstance(r, APIRoute) and any(c in auth for c in calls(r.dependant))
        for m in r.methods
    )
    assert set(GPT_READ_CALLS) == main.READONLY_KEY_ALLOWLIST
    return routes, main.READONLY_KEY_ALLOWLIST


def check(base):
    host = urlsplit(base).hostname or ""
    if host == PRODUCTION_HOST or "staging" not in host:
        raise SystemExit(f"refusing: {host!r} is not the staging service")
    key = KEY_FILE.read_text().strip()
    routes, allowlist = _routes()
    ro = {"X-API-Key": key}
    reads, writes, other_gets, failures = [], [], [], []
    with httpx.Client(base_url=base, timeout=30) as http:
        whoami_unknown = http.get("/customers", headers={"X-API-Key": "not-a-real-key"})
        for route in sorted(allowlist):
            method, url, body = GPT_READ_CALLS[route]
            r = http.request(method, url, headers=ro, json=body)
            detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else None
            ok = r.status_code == 200 or (r.status_code == 404 and detail != NOT_AUTHORIZED)
            reads.append({"route": f"{route[0]} {route[1]}", "status": r.status_code, "ok": ok})
            if not ok:
                failures.append(reads[-1])
        for method, path in routes:
            if (method, path) in allowlist:
                continue
            url = re.sub(r"\{[^}]+\}", "1", path)
            r = http.request(method, url, headers=ro, json={} if method in ("POST", "PUT", "PATCH") else None)
            detail = r.json().get("detail") if r.headers.get("content-type", "").startswith("application/json") else None
            ok = r.status_code == 403 and detail == NOT_AUTHORIZED
            row = {"route": f"{method} {path}", "status": r.status_code, "ok": ok}
            (other_gets if method == "GET" else writes).append(row)
            if not ok:
                failures.append(row)
        # Codex FIX ROUND 1: the read-only key never opens a session (401
        # ACTOR_KEY_INVALID; 404 when KEY_SESSIONS_ENABLED is off).
        r = http.post("/auth/session", json={"actor_key": key})
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        code = body.get("detail", {}).get("error_code") if isinstance(body.get("detail"), dict) else body.get("detail")
        exchange = {"route": "POST /auth/session (read-only key as actor_key)", "status": r.status_code,
                    "detail": code, "ok": r.status_code in (401, 404) and "session_token" not in r.text}
        if not exchange["ok"]:
            failures.append({k: exchange[k] for k in ("route", "status", "ok")})
    return {
        "check": "G1 READONLY_API_KEY on hosted FastAPI-staging",
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base": base,
        "unknown_key_control": {"status": whoami_unknown.status_code,
                                "detail": whoami_unknown.json().get("detail")},
        "gpt_reads_allowed": f"{sum(r['ok'] for r in reads)}/{len(reads)}",
        "writes_refused_403": f"{sum(r['ok'] for r in writes)}/{len(writes)}",
        "other_gets_refused_403": f"{sum(r['ok'] for r in other_gets)}/{len(other_gets)}",
        "session_exchange_refused": exchange,
        "failures": failures,
        "reads": reads,
        "writes": writes,
        "other_gets": other_gets,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out")
    args = ap.parse_args()
    result = check(args.base.rstrip("/"))
    summary = {k: result[k] for k in ("base", "unknown_key_control", "gpt_reads_allowed",
                                      "writes_refused_403", "other_gets_refused_403",
                                      "session_exchange_refused", "failures")}
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    sys.exit(1 if result["failures"] else 0)
