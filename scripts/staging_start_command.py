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
run()
'''


def start_command():
    # Embed the launcher as well: staging may still run an older snapshot.
    proxy = (ROOT / "proxy_server.py").read_text().split("if __name__ == '__main__':")[0]
    source = (ROOT / "staging_safety.py").read_text() + "\n" + proxy + LAUNCH
    return "python -c " + shlex.quote(source)


if __name__ == "__main__":
    print(start_command())
