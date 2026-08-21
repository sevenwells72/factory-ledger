#!/bin/bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$repo_dir/.venv-test/bin/python"

if [[ ! -x "$python_bin" ]]; then
    echo "Pinned test interpreter not found: $python_bin" >&2
    echo "Rebuild it with: python3.12 -m venv .venv-test" >&2
    exit 1
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
