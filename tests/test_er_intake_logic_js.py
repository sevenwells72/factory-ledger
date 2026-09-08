"""Runs the ER-intake review-state JS tests (tests/test_er_intake_logic.js,
node:test over dashboard/er-intake-logic.js) inside the pytest suite, so the
audit-fix regression coverage can't be skipped by habit. Skips only when node
is not installed."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
JS_TEST = ROOT / "tests" / "test_er_intake_logic.js"


def test_er_intake_logic_js():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    result = subprocess.run(
        [node, "--test", str(JS_TEST)],
        capture_output=True, text=True, cwd=ROOT, timeout=120,
    )
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
