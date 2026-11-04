#!/bin/bash
# Full Python regression run for a final candidate (High tier and pre-merge).
#
# Both key-session modes run at the same time, each on its own fresh disposable
# LOCAL database built from tests/schema/schema.sql. The databases are dropped
# afterwards, and the result is recorded against the exact files tested, so
# later steps and reviewers can reuse it instead of re-running the suite.
#
#   scripts/run_full_suite.sh              # run both modes, record the result
#   scripts/run_full_suite.sh --verify     # exit 0 if the current files AND test environment already have a passing record
#   scripts/run_full_suite.sh -x           # extra pytest args are passed through (result is not reusable)
#   FL_KEEP_DBS=1 scripts/run_full_suite.sh   # keep the databases for debugging
#
# Records live in <git common dir>/fl-test-evidence/ (shared by every worktree
# of this clone, never committed): <files-hash>-<env-hash>.json for a passing
# full run, runs/<run-id>.* for every run's logs and summary. The environment
# hash covers the Python version, every installed package version of the test
# interpreter, and the PostgreSQL version, so a changed environment needs a new run.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
PG_BIN="/opt/homebrew/opt/postgresql@17/bin"
common_dir="$(git rev-parse --path-format=absolute --git-common-dir)"
evidence_dir="$common_dir/fl-test-evidence"
mkdir -p "$evidence_dir/runs"

# Hash of the exact working files (tracked and untracked, minus ignored files),
# computed in a throwaway index so the real index is untouched.
files_hash() {
    local tmp
    tmp="$(mktemp -d)"
    GIT_INDEX_FILE="$tmp/index" git read-tree HEAD
    GIT_INDEX_FILE="$tmp/index" git add -A
    GIT_INDEX_FILE="$tmp/index" git write-tree
    rm -rf "$tmp"
}

# Hash of the test environment: the interpreter run_tests.sh would use (Python
# version and every installed package version) plus the PostgreSQL version.
env_hash() {
    local python_env pg_version
    python_env="$(FL_ENV_FINGERPRINT=1 "$repo_dir/scripts/run_tests.sh")" || return 1
    pg_version="$("$PG_BIN/postgres" --version)" || return 1
    [[ -n "$python_env" && -n "$pg_version" ]] || return 1
    printf '%s\n%s\n' "$python_env" "$pg_version" | shasum -a 256 | cut -c1-16
}

if [[ "${1:-}" == "--verify" ]]; then
    hash="$(files_hash)"
    env="$(env_hash)" || { echo "Could not fingerprint the test environment." >&2; exit 2; }
    record="$evidence_dir/$hash-$env.json"
    if [[ -f "$record" ]]; then
        echo "PASSING FULL-SUITE RECORD for the current files and test environment: $record"
        cat "$record"
        exit 0
    fi
    echo "No passing full-suite record for the current files ($hash) and test environment ($env)."
    exit 1
fi

run_id="fl_full_$$_${RANDOM}"
db0="${run_id}_ks0"
db1="${run_id}_ks1"
log0="$evidence_dir/runs/$run_id-ks0.log"
log1="$evidence_dir/runs/$run_id-ks1.log"

cleanup() {
    if [[ "${FL_KEEP_DBS:-0}" == "1" ]]; then
        echo "Keeping databases with prefix $run_id (FL_KEEP_DBS=1)."
        return
    fi
    "$PG_BIN/psql" -h localhost -p 5432 -d postgres -Atc \
        "SELECT datname FROM pg_database WHERE datname LIKE '${run_id}%'" 2>/dev/null |
        while read -r db; do
            [[ -n "$db" ]] && "$PG_BIN/dropdb" -h localhost -p 5432 --if-exists --force "$db" || true
        done
}
trap cleanup EXIT

# Refuse to run on a test interpreter whose packages differ from the exact pins
# in requirements.txt: results from a stale environment are not valid evidence.
FL_CHECK_PINS=1 "$repo_dir/scripts/run_tests.sh" --version > /dev/null

hash_before="$(files_hash)"
env_before="$(env_hash)" || { echo "Could not fingerprint the test environment." >&2; exit 2; }
echo "Run $run_id: files $hash_before, environment $env_before"
"$repo_dir/scripts/setup_test_db.sh" --db "$db0" > "$evidence_dir/runs/$run_id-setup.log" 2>&1
"$repo_dir/scripts/setup_test_db.sh" --db "$db1" >> "$evidence_dir/runs/$run_id-setup.log" 2>&1

run_mode() {
    local mode="$1" db="$2" prefix="$3" log="$4"
    shift 4
    echo "# KEY_SESSIONS_ENABLED=$mode database=$db started=$(date +%H:%M:%S)" > "$log"
    KEY_SESSIONS_ENABLED="$mode" \
    TEST_DATABASE_URL="postgresql://localhost:5432/$db" \
    TEST_DB_PREFIX="$prefix" \
        "$repo_dir/scripts/run_tests.sh" -p no:cacheprovider -rfE "$@" >> "$log" 2>&1
}

start="$(date +%s)"
run_mode 0 "$db0" "${run_id}_a_" "$log0" "$@" &
pid0=$!
run_mode 1 "$db1" "${run_id}_b_" "$log1" "$@" &
pid1=$!
rc0=0; wait "$pid0" || rc0=$?
rc1=0; wait "$pid1" || rc1=$?
seconds=$(( $(date +%s) - start ))
hash_after="$(files_hash)"
env_after="$(env_hash)" || env_after="unavailable"

summary() { grep -E '^=*[ ]*[0-9]+ (passed|failed|errors?)\b.* in [0-9.]+s' "$1" | tail -n 1 || true; }
sum0="$(summary "$log0")"
sum1="$(summary "$log1")"

full=true
[[ $# -gt 0 ]] && full=false
reusable=false
if [[ "$full" == true && $rc0 -eq 0 && $rc1 -eq 0 && "$hash_before" == "$hash_after" && "$env_before" == "$env_after" ]]; then
    reusable=true
fi

RUN_ID="$run_id" FILES_HASH="$hash_before" FILES_HASH_AFTER="$hash_after" \
ENV_HASH="$env_before" ENV_HASH_AFTER="$env_after" \
HEAD_COMMIT="$(git rev-parse HEAD)" BRANCH="$(git rev-parse --abbrev-ref HEAD)" \
RC0="$rc0" RC1="$rc1" SUM0="$sum0" SUM1="$sum1" SECONDS_TAKEN="$seconds" \
FULL="$full" REUSABLE="$reusable" ARGS="$*" LOG0="$log0" LOG1="$log1" \
OUT_RUN="$evidence_dir/runs/$run_id.json" \
OUT_PASS="$evidence_dir/$hash_before-$env_before.json" \
/usr/bin/python3 -I - <<'PY'
import json, os, datetime
e = os.environ
record = {
    "run_id": e["RUN_ID"],
    "finished_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    "files_hash": e["FILES_HASH"],
    "files_changed_during_run": e["FILES_HASH"] != e["FILES_HASH_AFTER"],
    "environment_hash": e["ENV_HASH"],
    "environment_changed_during_run": e["ENV_HASH"] != e["ENV_HASH_AFTER"],
    "head_commit": e["HEAD_COMMIT"],
    "branch": e["BRANCH"],
    "full_suite": e["FULL"] == "true",
    "extra_args": e["ARGS"],
    "seconds": int(e["SECONDS_TAKEN"]),
    "key_sessions_off": {"exit": int(e["RC0"]), "summary": e["SUM0"], "log": e["LOG0"]},
    "key_sessions_on": {"exit": int(e["RC1"]), "summary": e["SUM1"], "log": e["LOG1"]},
    "reusable": e["REUSABLE"] == "true",
}
text = json.dumps(record, indent=2) + "\n"
open(e["OUT_RUN"], "w").write(text)
if record["reusable"]:
    open(e["OUT_PASS"], "w").write(text)
PY

echo "key sessions OFF: exit $rc0  $sum0"
echo "key sessions ON:  exit $rc1  $sum1"
echo "Took ${seconds}s. Logs: $log0 $log1"
if [[ "$reusable" == true ]]; then
    echo "Recorded passing result for files $hash_before, environment $env_before ($evidence_dir/$hash_before-$env_before.json)."
elif [[ "$hash_before" != "$hash_after" || "$env_before" != "$env_after" ]]; then
    echo "Files or test environment changed during the run: result NOT recorded as reusable."
fi
[[ $rc0 -eq 0 && $rc1 -eq 0 ]]
