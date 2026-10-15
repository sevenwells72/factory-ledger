#!/usr/bin/env python3
"""Read-only, status-only check of O-7's dashboard-key boot prerequisite.

Read the target service's DATABASE_URL, DASHBOARD_API_KEY, API_KEY and optional
READONLY_API_KEY from the environment. Do not import the application: its
startup runs migrations. stdout contains only OK or COLLISION and the kinds;
an incomplete check exits 2 with no output on either stream.
"""

from hashlib import sha256
import os
import sys


def actor_key_collision(database_url: str, dashboard_key: str) -> bool:
    import psycopg2

    connection = psycopg2.connect(database_url, connect_timeout=5)
    try:
        # psycopg2 issues BEGIN before this first statement. Transaction-scoped
        # read-only/timeout settings cannot poison a shared transaction pooler.
        with connection:
            with connection.cursor() as cursor:
                cursor.execute("SET TRANSACTION READ ONLY")
                cursor.execute("SET LOCAL statement_timeout = '5s'")
                cursor.execute(
                    "SELECT EXISTS (SELECT 1 FROM public.actors WHERE key_hash = %s)",
                    (sha256(dashboard_key.encode("utf-8")).hexdigest(),),
                )
                # Intentionally no active filter: retired personal keys count.
                return bool(cursor.fetchone()[0])
    finally:
        connection.close()


def main() -> int:
    dashboard_key = (os.environ.get("DASHBOARD_API_KEY") or "").strip()
    master_key = (os.environ.get("API_KEY") or "").strip()
    readonly_key = (os.environ.get("READONLY_API_KEY") or "").strip()
    database_url = (os.environ.get("DATABASE_URL") or "").strip()
    if not all((dashboard_key, master_key, database_url)):
        return 2

    collisions = []
    if dashboard_key == master_key:
        collisions.append("master")
    if readonly_key and dashboard_key == readonly_key:
        collisions.append("readonly")
    try:
        if actor_key_collision(database_url, dashboard_key):
            collisions.append("personal")
    except (Exception, KeyboardInterrupt):
        # Driver exceptions may contain the DSN or query parameters. Deliberately
        # do not print/log/chain them, including dependency and timeout failures.
        return 2

    if collisions:
        print("COLLISION " + " ".join(collisions))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
