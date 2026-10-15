"""PR #122 guard regressions: real CLI exit status and redacted output."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts/check_no_key_literals.py"
PLANTED = "fld_synthetic_pr122_literal_9741"


def run_guard(tmp_path, source, *, live="", ci=False):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    page = tmp_path / "dashboard/page.js"
    page.parent.mkdir()
    page.write_text(source, encoding="utf-8")
    subprocess.run(["git", "-C", str(tmp_path), "add", "dashboard/page.js"], check=True)
    env = {**os.environ, "FL_KEY_GUARD_VALUES": live, "GITHUB_ACTIONS": "true" if ci else "false"}
    return subprocess.run([sys.executable, str(GUARD), "--root", str(tmp_path)],
                          env=env, capture_output=True, text=True)


@pytest.mark.parametrize("source", [
    "headers.set('X-API-Key', 'VALUE');",
    'headers.append("X-API-Key", "VALUE");',
    "new Headers([['X-API-Key', 'VALUE']]);",
    "headers.set(\n  'X-API-Key',\n  'VALUE'\n);",
    "new Headers([\n  [\n    'X-API-Key',\n    'VALUE'\n  ]\n]);",
    "const headers = {\n 'X-API-Key':\n 'VALUE'\n};",
])
def test_header_literal_forms_fail_with_redacted_output(tmp_path, source):
    result = run_guard(tmp_path, source.replace("VALUE", PLANTED))
    assert result.returncode == 1, result.stdout
    assert "dashboard/page.js:" in result.stdout
    assert "[redacted]" in result.stdout
    assert PLANTED not in result.stdout + result.stderr
    assert result.stderr == ""


@pytest.mark.parametrize("source", [
    "const RK = 'VALUE';\nheaders.set('X-API-Key', RK);",
    "let innocent = 'VALUE';\nconst alias = innocent;\nheaders.append('X-API-Key', alias);",
    "const innocent = 'VALUE';\nnew Headers([['X-API-Key', innocent]]);",
    "const innocent = 'VALUE';\nfetch(url, {headers: {'X-API-Key': innocent}});",
])
def test_simple_literal_variables_used_in_headers_fail_redacted(tmp_path, source):
    result = run_guard(tmp_path, source.replace("VALUE", PLANTED))
    assert result.returncode == 1, result.stdout
    assert "[redacted]" in result.stdout
    assert PLANTED not in result.stdout + result.stderr


def test_arbitrary_innocuous_name_is_caught_by_newline_live_values(tmp_path):
    result = run_guard(tmp_path, f"const RK = '{PLANTED}';\n",
                       live=f"synthetic-second-environment-key\n{PLANTED}\n")
    assert result.returncode == 1, result.stdout
    assert "2 live value(s) checked" in result.stdout
    assert PLANTED not in result.stdout + result.stderr
    assert "synthetic-second-environment-key" not in result.stdout + result.stderr


def test_empty_live_list_still_succeeds_without_findings(tmp_path):
    result = run_guard(tmp_path, "headers.set('X-API-Key', await FLDashboard.key());\n",
                       live="\n\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 finding(s); 0 live value(s) checked" in result.stdout
    assert "[redacted]" not in result.stdout
    assert result.stderr == ""


def test_ci_warns_clearly_when_live_secret_is_missing_but_stays_clean(tmp_path):
    result = run_guard(tmp_path, "const name = 'ordinary UI text';\n", ci=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "::warning::FL_KEY_GUARD_VALUES is empty; live-key matching is disabled" in result.stdout
    assert "0 finding(s); 0 live value(s) checked" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("source", [
    "const label = 'ordinary text';\nheaders.set('X-API-Key', await FLDashboard.key());",
    "const key = await FLDashboard.key();\nheaders.append('X-API-Key', key);",
    "const token = session.token;\nnew Headers([['X-API-Key', token]]);",
])
def test_runtime_header_values_still_pass(tmp_path, source):
    result = run_guard(tmp_path, source)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "0 finding(s)" in result.stdout
