#!/usr/bin/env python3
"""O-7 rotation probe: is the API serving and honouring the NEW dashboard key?

Prints status codes and sha256 prefixes only — never a key value. Reads keys
from files (mode 600, e.g. ~/Documents/fl-secrets/staging-dashboard-key.txt).

  * GET /auth/dashboard-key is 200, no-store, and serves the new key (compared
    by fingerprint);
  * the new key reads (/auth/whoami -> 200, key_kind legacy_dashboard);
  * the old key, when given, is dead (/auth/whoami -> 403);
  * the new key still cannot open a personal session (POST /auth/session ->
    401, or 404 while key sessions are off) nor prepare make/pack/adjust/
    found/ship (403 before any handler runs, so nothing is written).

    python3 scripts/check_dashboard_key_rotation.py --base https://<api host> \
        --new-key-file ~/Documents/fl-secrets/staging-dashboard-key.txt \
        [--old-key-file <file>]

Exit 0 = every check passed. A production --base needs --allow-production.
"""
import argparse
import hashlib
import sys
from pathlib import Path
from urllib.parse import urlsplit

import httpx

PRODUCTION_HOST = "fastapi-production-b73a.up.railway.app"
PREPARE_ROUTES = ["/make/prepare", "/pack/prepare", "/adjust/prepare", "/inventory/found/prepare",
                  "/ship/prepare", "/ship/bulk/prepare", "/sales/orders/999999999/ship/prepare"]


def fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def read_key(path):
    value = Path(path).expanduser().read_text().strip()
    if len(value) < 16:
        raise SystemExit(f"{path}: key file looks empty or truncated")
    return value


def run(client, new, old=None):
    results = []

    def check(label, ok, detail):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'}  {label}: {detail}")

    served = client.get("/auth/dashboard-key")
    body = served.json() if served.headers.get("content-type", "").startswith("application/json") else {}
    value = body.get("dashboard_key") or ""
    check("GET /auth/dashboard-key", served.status_code == 200 and value == new
          and served.headers.get("cache-control") == "no-store",
          f"{served.status_code}, cache-control={served.headers.get('cache-control')}, "
          f"served sha256 {fingerprint(value) if value else '-'} vs new {fingerprint(new)}")
    who = client.get("/auth/whoami", headers={"X-API-Key": new})
    kind = who.json().get("key_kind") if who.status_code == 200 else None
    check("new key GET /auth/whoami", who.status_code == 200 and kind == "legacy_dashboard",
          f"{who.status_code}, key_kind={kind}")
    if old is not None:
        dead = client.get("/auth/whoami", headers={"X-API-Key": old})
        check("old key GET /auth/whoami", dead.status_code == 403, f"{dead.status_code} (want 403)")
    session = client.post("/auth/session", json={"actor_key": new})
    check("new key POST /auth/session", session.status_code in (401, 404),
          f"{session.status_code} (want 401, or 404 with sessions off)")
    for route in PREPARE_ROUTES:
        response = client.post(route, json={}, headers={"X-API-Key": new})
        check(f"new key POST {route}", response.status_code == 403, f"{response.status_code} (want 403)")
    return all(results)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True)
    parser.add_argument("--new-key-file", required=True)
    parser.add_argument("--old-key-file")
    parser.add_argument("--allow-production", action="store_true")
    args = parser.parse_args(argv)
    if urlsplit(args.base).hostname == PRODUCTION_HOST and not args.allow_production:
        raise SystemExit("Refusing production without --allow-production")
    new = read_key(args.new_key_file)
    old = read_key(args.old_key_file) if args.old_key_file else None
    if old is not None and old == new:
        raise SystemExit("old and new key files hold the same value")
    with httpx.Client(base_url=args.base.rstrip("/"), timeout=30) as client:
        ok = run(client, new, old)
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
