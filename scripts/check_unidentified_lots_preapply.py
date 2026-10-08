#!/usr/bin/env python3
"""Read-only pre-064 check. Exit 1 on duplicate open/escalated lot exceptions.

Usage: python scripts/check_unidentified_lots_preapply.py --database-url-file PATH
Reads a mode-600 URI file, never prints credentials, never changes exceptions.
Remote databases must use port 5432; no session-level read-only settings.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.a5_readonly import readonly_cursor


def duplicate_open_lots(cur):
    cur.execute("""SELECT lot_id, count(*) AS open_count, array_agg(id ORDER BY id) AS exception_ids
        FROM exceptions WHERE kind='UNIDENTIFIED_LOT' AND status IN ('open','escalated')
          AND lot_id IS NOT NULL
        GROUP BY lot_id HAVING count(*) > 1 ORDER BY lot_id""")
    return [dict(row) for row in cur.fetchall()]


def require_no_duplicates(cur):
    if duplicate_open_lots(cur):
        raise RuntimeError('Duplicate open UNIDENTIFIED_LOT exceptions; run the pre-apply report and reconcile before 064')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database-url-file', required=True)
    args = parser.parse_args()
    try:
        with readonly_cursor(args.database_url_file) as cur:
            duplicates = duplicate_open_lots(cur)
        for row in duplicates:
            print(f"lot {row['lot_id']}: {row['open_count']} open exceptions {row['exception_ids']}")
        print('FAIL: reconcile duplicate exceptions before 064' if duplicates else
              'PASS: no duplicate open UNIDENTIFIED_LOT exceptions per lot')
        return 1 if duplicates else 0
    except Exception:
        # Connection and libpq errors can contain credentials. Never echo them.
        print('Pre-apply check failed; verify the protected URI file, port 5432 and schema.', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(main())
