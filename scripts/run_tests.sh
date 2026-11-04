#!/bin/bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Interpreter: FL_TEST_VENV if set, else this checkout's .venv-test, else the
# main checkout's .venv-test (worktrees share it instead of building their own).
if [[ -n "${FL_TEST_VENV:-}" ]]; then
    python_bin="$FL_TEST_VENV/bin/python"
else
    python_bin="$repo_dir/.venv-test/bin/python"
    if [[ ! -x "$python_bin" ]]; then
        common_dir="$(git -C "$repo_dir" rev-parse --path-format=absolute --git-common-dir 2>/dev/null || true)"
        if [[ -n "$common_dir" && -x "$(dirname "$common_dir")/.venv-test/bin/python" ]]; then
            python_bin="$(dirname "$common_dir")/.venv-test/bin/python"
        fi
    fi
fi

if [[ ! -x "$python_bin" ]]; then
    echo "Pinned test interpreter not found: $python_bin" >&2
    echo "Rebuild it in the main checkout with: python3.12 -m venv .venv-test (or set FL_TEST_VENV)" >&2
    exit 1
fi

if [[ "${FL_CHECK_PINS:-0}" == "1" ]]; then
    # Exact "name==version" pins in requirements.txt must match the interpreter.
    "$python_bin" -I - "$repo_dir/requirements.txt" <<'PY' || exit 3
import re, sys
from importlib import metadata
bad = []
for line in open(sys.argv[1]):
    m = re.match(r"\s*([A-Za-z0-9_.\[\]-]+)\s*==\s*([^\s;#]+)", line)
    if not m:
        continue
    name, want = re.sub(r"\[.*\]", "", m.group(1)), m.group(2)
    try:
        have = metadata.version(name)
    except metadata.PackageNotFoundError:
        have = "missing"
    if have != want:
        bad.append(f"{name}: requirements.txt pins {want}, test interpreter has {have}")
if bad:
    print("Test interpreter does not match requirements.txt:", *bad, sep="\n  ", file=sys.stderr)
    print(f"Fix: {sys.executable} -m pip install -r requirements.txt -r tests/requirements-test.txt", file=sys.stderr)
    sys.exit(1)
PY
fi

export TEST_DATABASE_URL="${TEST_DATABASE_URL:-postgresql://localhost:5432/factory_ledger_test}"

if ! "$python_bin" -c 'import pyexpat' >/dev/null 2>&1; then
    expat_lib="/opt/homebrew/opt/expat/lib"
    if [[ -d "$expat_lib" ]]; then
        export DYLD_LIBRARY_PATH="$expat_lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    fi
fi

cd "$repo_dir"
exec "$python_bin" -m pytest "$@"
