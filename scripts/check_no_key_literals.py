#!/usr/bin/env python3
"""Fail if an API key literal is (re)committed. Prints locations, never values.

O-7 (cutover runbook): the dashboard reads its scoped key from
GET /auth/dashboard-key, so no tracked file may carry a key value again.
Four checks over `git ls-files`:

  1. Retired keys, by SHA-256 fingerprint (the values are not in this file).
     Every token that looks like a key is hashed and compared, in every
     tracked file — history included (it was redacted in the O-7 fix round).
  2. Served dashboard code (dashboard/**) must not hard-code any X-API-Key
     header value or assign a string literal to an *API_KEY* / *DASHBOARD_KEY* / readKey name.
  3. Any key-shaped literal anywhere outside tests/ (fixtures are never served):
     a quoted X-API-Key header value, a curl-style `X-API-Key: value`,
     `Bearer value`, a `?key=value` query or `*API_KEY=value`, 12+ characters,
     unless it is a <placeholder>, $VARIABLE or one of DOCUMENTED_FAKES.
  4. Live values: FL_KEY_GUARD_VALUES (newline-separated, e.g. a CI secret or
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
    "93637d56f6c8ea900f2fdf6c8a7ecbaadda648664f49bf3a1d890f3164c0e78f": "master key (pre-cutover documented literal)",
}
# Exact fake values that documentation outside tests/ names on purpose.
DOCUMENTED_FAKES = {"test-dashboard-key", "not-a-real-key"}
LITERAL_PATTERNS = [
    re.compile(r"""['"]X-API-Key['"]\s*:\s*['"`]([^'"`]{12,})['"`]""", re.I),
    re.compile(r"""X-API-Key:\s+([A-Za-z0-9_\-.]{12,})""", re.I),
    re.compile(r"""\bBearer\s+([A-Za-z0-9_\-.]{12,})"""),
    re.compile(r"""[?&]key=([A-Za-z0-9_\-]{12,})"""),
    re.compile(r"""\b[A-Z_]*API_KEY\s*[=:]\s*['"`]?([A-Za-z0-9_\-]{12,})"""),
]
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
        served = rel.startswith("dashboard/")
        fixture = rel.startswith("tests/")
        for number, line in enumerate(text.splitlines(), 1):
            retired = False
            for token in TOKEN.findall(line):
                label = RETIRED_FINGERPRINTS.get(_fingerprint(token))
                if label:
                    retired = True
                    findings.append(f"{rel}:{number}: retired {label} literal [redacted]")
            if served and (HEADER_LITERAL.search(line) or KEY_ASSIGN.search(line)):
                findings.append(f"{rel}:{number}: hard-coded API key in served dashboard code [redacted]")
            elif not fixture and not retired and any(
                    m.group(1) not in DOCUMENTED_FAKES
                    for pattern in LITERAL_PATTERNS for m in pattern.finditer(line)):
                findings.append(f"{rel}:{number}: key-shaped literal [redacted]")
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
