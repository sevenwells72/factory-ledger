"""Print a secret-free Railway start command that also works on unmerged main.

The command embeds the same safety module used by main.py. Set it ONLY on the
FastAPI-staging service, never in railway.json (which production also consumes).
"""
from pathlib import Path
import shlex

ROOT = Path(__file__).resolve().parents[1]
LAUNCH = '''
if os.getenv("ENVIRONMENT", "").strip().lower() != "staging":
    raise RuntimeError("This launcher requires ENVIRONMENT=staging")
assert_staging_database(os.getenv("DATABASE_URL", ""))
import uvicorn
uvicorn.run("main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
'''


def start_command():
    source = (ROOT / "staging_safety.py").read_text() + LAUNCH
    return "python -c " + shlex.quote(source)


if __name__ == "__main__":
    print(start_command())
