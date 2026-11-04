#!/usr/bin/env python3
"""Print a reviewed, self-checking SQL transaction that seeds `person_choices` (091).

It never connects to a database. A human applies the printed SQL in an
approved step, after a verified backup, over port 5432:

    scripts/seed_person_choices.py --environment production \
        --michael 1:Blubber --luz 3:Luz --arturo 2:Arturo --miriam 4:Miriam

Each mapping is ACTOR_ID:EXPECTED_ACTORS_NAME. Inside the transaction every row
is checked against the live `actors` table — exact id, exact name, the routine
role the person needs, active — and the whole transaction aborts on the first
mismatch. Nothing is ever substituted, renamed or deactivated; `actors.name`
(for example the owner's historical 'Blubber') stays as it is. The display
names are fixed: Michael (Owner), Luz (Office), Arturo (Floor Lead),
Miriam (Office), in that order.
"""
import argparse
import re
import sys

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__))))
import trusted_browsers  # noqa: E402  (pure constants; no database, no app import)

KEYS = ('michael', 'luz', 'arturo', 'miriam')
MAPPING = re.compile(r'([1-9][0-9]{0,9}):(.{1,60})')


def literal(value):
    return "'" + value.replace("'", "''") + "'"


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--environment', required=True, choices=('production', 'staging', 'test'))
    for key in KEYS:
        parser.add_argument('--' + key, required=True, metavar='ID:ACTORS_NAME')
    args = parser.parse_args(argv)
    rows = []
    for key, (display, role_label, role) in zip(KEYS, trusted_browsers.EXPECTED_CHOICES):
        match = MAPPING.fullmatch(getattr(args, key))
        if not match:
            parser.error(f'--{key} must be ACTOR_ID:ACTORS_NAME')
        rows.append({'actor_id': int(match[1]), 'name': match[2], 'display': display,
                     'role_label': role_label, 'role': role})
    if len({r['actor_id'] for r in rows}) != len(rows):
        parser.error('the four people must be four different actor ids')
    return args, rows


def build_sql(environment, rows):
    checks, inserts = [], []
    for order, row in enumerate(rows, start=1):
        checks.append(
            f"    IF NOT EXISTS (SELECT 1 FROM actors WHERE id={row['actor_id']} AND name={literal(row['name'])}"
            f" AND role={literal(row['role'])} AND active) THEN\n"
            f"        RAISE EXCEPTION 'person_choices seed refused: actor {row['actor_id']} is not the active "
            f"{row['role']} named %', {literal(row['name'])};\n    END IF;")
        inserts.append(f"({row['actor_id']}, {literal(row['display'])}, {literal(row['role_label'])}, {order}, true)")
    return '\n'.join([
        f'-- person_choices seed for {environment}. Review every line; apply only in an approved step.',
        'BEGIN;',
        "SET LOCAL lock_timeout = '5s';",
        "SET LOCAL statement_timeout = '60s';",
        "DO $$ BEGIN",
        "    IF to_regclass('public.person_choices') IS NULL THEN",
        "        RAISE EXCEPTION 'person_choices seed refused: migration 091 is not applied';",
        "    END IF;",
        "    IF EXISTS (SELECT 1 FROM person_choices) THEN",
        "        RAISE EXCEPTION 'person_choices seed refused: the table already has rows; review them instead';",
        "    END IF;",
        *checks,
        "END $$;",
        'INSERT INTO person_choices(actor_id, display_name, role_label, sort_order, active) VALUES',
        '    ' + ',\n    '.join(inserts) + ';',
        'SELECT c.sort_order, c.actor_id, a.name AS actors_name, c.display_name, c.role_label, a.role',
        '  FROM person_choices c JOIN actors a ON a.id=c.actor_id ORDER BY c.sort_order;',
        'COMMIT;',
        '',
    ])


def main(argv=None):
    args, rows = parse(argv)
    sys.stdout.write(build_sql(args.environment, rows))


if __name__ == '__main__':
    main()
