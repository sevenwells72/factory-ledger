#!/usr/bin/env python3
"""Fail if an API key literal is (re)committed. Prints locations, never values.

O-7 (cutover runbook): the dashboard reads its scoped key from
GET /auth/dashboard-key, so no tracked file may carry a key value again.
Four checks over `git ls-files`:

  1. Retired keys, by SHA-256 fingerprint (the values are not in this file).
     Every token that looks like a key is hashed and compared, in every
     tracked file — history included (it was redacted in the O-7 fix round).
  2. Served dashboard code (dashboard/**) must not hard-code any X-API-Key
     header value, including Headers.set/append and header tuples across
     lines. Simple literal variables/aliases feeding those headers are also
     checked, as are *API_KEY* / *DASHBOARD_KEY* / readKey assignments.
  3. Any key-shaped literal anywhere outside tests/ (fixtures are never served):
     a quoted X-API-Key header value, a curl-style `X-API-Key: value`,
     `Bearer value`, a `?key=value` query or `*API_KEY=value`, 12+ characters,
     unless it is a <placeholder>, $VARIABLE or one of DOCUMENTED_FAKES.
  4. Live values: FL_KEY_GUARD_VALUES is plain text, one raw current key per
     line (production and staging), without names, quotes, commas, JSON, or
     shell syntax. Blank lines are ignored; a trailing newline is fine.
     Values are searched in every tracked file, with no exemptions. This
     catches arbitrary names even when no static header use can be traced.

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
KEY_ASSIGN = re.compile(r"""\b\w*(?:API_?KEY|DASHBOARD_?KEY|readKey)\s*[:=]\s*['"`][A-Za-z0-9_\-]{8,}['"`]""", re.I)
# These deliberately bounded checks are not a JavaScript parser. They cover
# direct literals and simple same-file assignments/aliases; live values are
# still needed for computed strings, unrelated declarations and other syntax.
JS_NAME = r"[A-Za-z_$][\w$]*"
JS_STRING = r'''(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|`(?:\\.|[^`\\])*`)'''
JS_VALUE = rf"(?:{JS_STRING}|{JS_NAME}(?![\w$]))"
HEADER_VALUE = re.compile(
    rf'''(?:['"]X-API-Key['"]\s*:\s*|'''
    rf'''\.\s*(?:set|append)\s*\(\s*['"]X-API-Key['"]\s*,\s*|'''
    rf'''\[\s*['"]X-API-Key['"]\s*,\s*)(?P<value>{JS_VALUE})''', re.I)
SIMPLE_ASSIGN = re.compile(
    rf"(?<![\w$.])(?P<name>{JS_NAME})\s*=(?!=|>)\s*(?P<value>{JS_VALUE})"
    r"\s*(?=[;,\r\n]|$)")
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".xlsx", ".woff", ".woff2", ".zip"}


def _fingerprint(value):
    return hashlib.sha256(value.encode()).hexdigest()


def tracked_files(root):
    out = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True).stdout
    return [p for p in out.decode().split("\0") if p]


def _is_static_string(value):
    # Empty strings and interpolated templates are not literal credentials.
    return len(value) > 2 and value[0] in "\"'`" and "${" not in value


def _served_literal_lines(text):
    """Locations only; never include a matched literal in diagnostics."""
    assignments = [(m.group("name"), m.group("value"))
                   for m in SIMPLE_ASSIGN.finditer(text)]
    literal_names = set()
    # A fixed point also handles aliases declared before their source name.
    for _ in range(len(assignments)):
        found = {name for name, value in assignments
                 if _is_static_string(value) or value in literal_names}
        if found <= literal_names:
            break
        literal_names.update(found)
    offsets = [m.start() for m in KEY_ASSIGN.finditer(text)]
    offsets.extend(m.start() for m in HEADER_VALUE.finditer(text)
                   if _is_static_string(m.group("value"))
                   or m.group("value") in literal_names)
    return {text.count("\n", 0, offset) + 1 for offset in offsets}


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
        served_literal_lines = _served_literal_lines(text) if served else set()
        for number, line in enumerate(text.splitlines(), 1):
            retired = False
            for token in TOKEN.findall(line):
                label = RETIRED_FINGERPRINTS.get(_fingerprint(token))
                if label:
                    retired = True
                    findings.append(f"{rel}:{number}: retired {label} literal [redacted]")
            if number in served_literal_lines:
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
    if not live and os.environ.get("GITHUB_ACTIONS") == "true":
        print("::warning::FL_KEY_GUARD_VALUES is empty; live-key matching is disabled. "
              "Set the repository secret to one raw production/staging dashboard key per line. "
              "Structural and retired-key checks still run.")
    findings = scan(root, tracked_files(root), live)
    for finding in findings:
        print(finding)
    print(f"key-literal guard: {len(findings)} finding(s); {len(live)} live value(s) checked")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
