#!/usr/bin/env python3
"""Fail if an API key literal is (re)committed. Prints locations, never values.

O-7 (cutover runbook): the dashboard reads its scoped key from
GET /auth/dashboard-key, so no tracked file may carry a key value again.
Three checks over `git ls-files`:

  1. Retired keys, by SHA-256 fingerprint (the values are not in this file).
     Every token that looks like a key is hashed and compared. Append-only
     history (change logs, design/audit docs) may still name a retired key —
     it is dead once rotated — so HISTORY_PATHS is exempt from this check only.
  2. Served dashboard code (dashboard/**) must not hard-code any X-API-Key
     header value or assign a string literal to an *API_KEY* / *DASHBOARD_KEY* / readKey name.
  3. Live values: FL_KEY_GUARD_VALUES (newline-separated, e.g. a CI secret or
     `$(cat ~/Documents/fl-secrets/staging-dashboard-key.txt)`) are searched in
     every tracked file, with no exemptions.

Usage: python3 scripts/check_no_key_literals.py [--root DIR]
Exit 0 = clean, 1 = findings (each printed as path:line + redacted reason).
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

# sha256 of retired key values. Add the old value's fingerprint on every
# rotation: python3 -c "import hashlib,sys;print(hashlib.sha256(sys.stdin.read().strip().encode()).hexdigest())"
RETIRED_FINGERPRINTS = {
    "ab98de1625ea9054743ac441b2e6ace163bd82cff9e74e9ee8613d6e9691677d": "dashboard key (pre-O-7 literal)",
}
HISTORY_PATHS = re.compile(r"^(CHANGE_LOG\.md|FACTORY_LEDGER_CHANGELOG\.md|docs/|archive/|audits/)")
TOKEN = re.compile(r"[A-Za-z0-9_\-]{12,128}")
HEADER_LITERAL = re.compile(r"""['"]X-API-Key['"]\s*:\s*['"`][^'"`]+['"`]""", re.I)
KEY_ASSIGN = re.compile(r"""\b\w*(?:API_?KEY|DASHBOARD_?KEY|readKey)\s*[:=]\s*['"`][A-Za-z0-9_\-]{8,}['"`]""", re.I)
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".xlsx", ".woff", ".woff2", ".zip"}


def _fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest()


def tracked_files(root):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    return [p for p in out.decode().split("\0") if p]


def scan(root, files, live_values=()):
    findings = []
    live = [(v, _fingerprint(v)[:8]) for v in live_values if len(v) >= 8]
    for rel in files:
        path = Path(root, rel)
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        history = bool(HISTORY_PATHS.match(rel))
        served = rel.startswith("dashboard/")
        for number, line in enumerate(text.splitlines(), 1):
            if not history:
                for token in TOKEN.findall(line):
                    label = RETIRED_FINGERPRINTS.get(_fingerprint(token))
                    if label:
                        findings.append(f"{rel}:{number}: retired {label} literal [redacted]")
            if served and (HEADER_LITERAL.search(line) or KEY_ASSIGN.search(line)):
                findings.append(f"{rel}:{number}: hard-coded API key in served dashboard code [redacted]")
            for value, short in live:
                if value in line:
                    findings.append(f"{rel}:{number}: live key value (sha256 {short}…) [redacted]")
    return findings


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    root = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=args.root, check=True,
                          capture_output=True, text=True).stdout.strip()
    live = [v.strip() for v in os.environ.get("FL_KEY_GUARD_VALUES", "").splitlines() if v.strip()]
    findings = scan(root, tracked_files(root), live)
    for finding in findings:
        print(finding)
    print(f"key-literal guard: {len(findings)} finding(s); {len(live)} live value(s) checked")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
