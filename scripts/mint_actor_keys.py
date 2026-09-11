#!/usr/bin/env python3
"""Mint per-actor API keys for FR-15 attribution (migration 052).

    python3 scripts/mint_actor_keys.py "Blubber:owner" "Arturo:floor" \
                                       "Luz:office" "Miriam:office"

For each NAME:ROLE it generates a fresh 32-byte urlsafe key, prints the
plaintext to stdout ONCE, and writes `actors_insert.sql` containing only the
sha256 hashes.

  ┌─────────────────────────────────────────────────────────────────┐
  │ THE SQL THIS WRITES IS PASTED INTO THE SUPABASE SQL EDITOR.     │
  │ Apply migrations/052_actors.sql there first — this file only    │
  │ populates the table that migration creates.                     │
  └─────────────────────────────────────────────────────────────────┘

PLAINTEXT KEYS ARE NEVER WRITTEN TO DISK. They exist in this process's stdout
and nowhere else, which is the entire security property of the scheme: the
database stores hashes, so a database dump is not a set of credentials. The
consequence is that a key not captured from this run is GONE — re-run the
script for that actor to mint a replacement. That is a feature; there is no
recovery path by design.

Hand each key to its person over a channel you would send a password over.
Do not paste them into the repo, a ticket, or a chat that is archived.

ROTATION. Re-running for an existing NAME mints a new key and the generated
INSERT ... ON CONFLICT (name) DO UPDATE replaces that actor's hash in place,
so the old key stops working the moment the SQL is applied (plus up to the
API's 60-second actor-cache TTL). The actor's id, and therefore every row
already attributed to them, is preserved.

REVOCATION without rotation is a one-liner in the SQL editor:

    update actors set active = false where name = 'Arturo';

VERIFY after applying, with one of the minted keys:

    curl -sH "X-API-Key: <key>" \
        https://fastapi-production-b73a.up.railway.app/auth/whoami
    -> {"actor":{"name":"Arturo","role":"floor"},"key_kind":"actor"}
"""

import argparse
import hashlib
import re
import secrets
import sys
from pathlib import Path

# Mirrors the CHECK constraint in migrations/052_actors.sql. Kept in sync by
# hand and asserted by tests/test_actor_attribution.py, so a role this script
# accepts can never be one the database rejects at paste time — a failure
# there happens after the keys have already been printed and distributed.
ROLES = ("owner", "floor", "office")

# actors.name lands in text columns that other writers cap at 60 characters
# (caller_source_tag). A longer name would be stored inconsistently depending
# on which path wrote it, so it is refused at mint time rather than truncated.
NAME_MAX = 60

OUTPUT_FILE = "actors_insert.sql"

# 32 bytes of os.urandom, urlsafe-base64 encoded to 43 characters.
KEY_BYTES = 32


def parse_actor(spec: str) -> tuple:
    """'Arturo:floor' -> ('Arturo', 'floor'), or exit with a usable message."""
    if spec.count(":") != 1:
        sys.exit(f"error: expected exactly one ':' in {spec!r} — use \"Name:role\"")
    name, role = (part.strip() for part in spec.split(":"))
    if not name:
        sys.exit(f"error: empty name in {spec!r}")
    if len(name) > NAME_MAX:
        sys.exit(f"error: name {name!r} is {len(name)} characters; "
                 f"the limit is {NAME_MAX}")
    # The name is interpolated into SQL below. Restricting it to a plain
    # identifier is simpler and safer than quoting rules, and every real name
    # this system has fits.
    if not re.fullmatch(r"[A-Za-z0-9 .'\-]+", name):
        sys.exit(f"error: name {name!r} may contain only letters, digits, "
                 f"spaces, dots, apostrophes and hyphens")
    if role not in ROLES:
        sys.exit(f"error: role {role!r} is not one of {', '.join(ROLES)} "
                 f"(in {spec!r})")
    return name, role


def sql_literal(value: str) -> str:
    """Single-quoted SQL string with quotes doubled — for names like O'Brien."""
    return "'" + value.replace("'", "''") + "'"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Mint per-actor API keys and write hash-only INSERT SQL.",
        epilog='example: mint_actor_keys.py "Blubber:owner" "Arturo:floor"',
    )
    parser.add_argument("actors", nargs="+", metavar="NAME:ROLE",
                        help=f"role is one of: {', '.join(ROLES)}")
    parser.add_argument("--out", default=OUTPUT_FILE,
                        help=f"SQL output path (default: {OUTPUT_FILE})")
    args = parser.parse_args(argv)

    parsed = [parse_actor(spec) for spec in args.actors]

    names = [name for name, _ in parsed]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        # Two rows for one name cannot both survive ON CONFLICT (name), so the
        # second key would be silently dead. Refuse rather than mint it.
        sys.exit(f"error: duplicate name(s): {', '.join(duplicates)}")

    minted = []
    for name, role in parsed:
        key = secrets.token_urlsafe(KEY_BYTES)
        minted.append((name, role, key, hashlib.sha256(key.encode("utf-8")).hexdigest()))

    lines = [
        "-- actors_insert.sql — generated by scripts/mint_actor_keys.py",
        "--",
        "-- Paste into the Supabase SQL editor, AFTER applying",
        "-- migrations/052_actors.sql. Contains sha256 hashes only; the",
        "-- plaintext keys were printed once by the script that wrote this and",
        "-- are not recoverable from this file.",
        "--",
        "-- Re-runnable: an existing name has its hash replaced and is",
        "-- reactivated, keeping its id and therefore its attribution history.",
        "",
    ]
    for name, role, _key, key_hash in minted:
        lines.append(
            f"INSERT INTO public.actors (name, role, key_hash, active)\n"
            f"     VALUES ({sql_literal(name)}, {sql_literal(role)}, "
            f"'{key_hash}', true)\n"
            f"ON CONFLICT (name) DO UPDATE\n"
            f"        SET key_hash = EXCLUDED.key_hash,\n"
            f"            active   = true;"
        )
        lines.append("")
    lines.append("SELECT id, name, role, active, created_at, last_used_at")
    lines.append("  FROM public.actors")
    lines.append(" ORDER BY id;")
    lines.append("")

    out_path = Path(args.out)
    out_path.write_text("\n".join(lines))

    width = max(len(name) for name, _, _, _ in minted)
    print()
    print("Minted keys — copy them NOW, they are not stored anywhere:")
    print()
    for name, role, key, _hash in minted:
        print(f"  {name.ljust(width)}  {role.ljust(6)}  {key}")
    print()
    print(f"Wrote {out_path} ({len(minted)} actor"
          f"{'' if len(minted) == 1 else 's'}, hashes only).")
    print("Next: paste it into the Supabase SQL editor, then verify with")
    print('  curl -sH "X-API-Key: <key>" $API/auth/whoami')
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
