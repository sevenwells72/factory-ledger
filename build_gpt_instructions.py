#!/usr/bin/env python3
"""
build_gpt_instructions.py — Generator for Factory Ledger GPT instruction files.

Concatenates shared-rules.md + {role}-specific.md into a single deployable file.
Exists to prevent the drift problem: the canonical source is the split files,
and the deployed file is always regenerated, never hand-edited.

Usage:
    python build_gpt_instructions.py floor
    python build_gpt_instructions.py sales    # (once sales-specific.md exists)
    python build_gpt_instructions.py trace    # (once trace-specific.md exists)

Char limits (per OpenAI Custom GPT instructions field):
    - Soft warning at 7,500 chars (printed, does not fail)
    - Hard fail at 8,000 chars (exit 1)

Exit codes:
    0 = success
    1 = output over hard limit
    2 = source file missing
    3 = invalid role argument
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

SOFT_LIMIT = 7500
HARD_LIMIT = 8000

VALID_ROLES = {"floor", "sales", "trace"}
SHARED_SOURCE = "shared-rules.md"

REPO_ROOT = Path(__file__).resolve().parent
SOURCES_DIR = REPO_ROOT / "gpt-configs" / "sources"
DIST_DIR = REPO_ROOT / "gpt-configs" / "dist"


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in VALID_ROLES:
        print(f"Usage: python {Path(argv[0]).name} {{{'|'.join(sorted(VALID_ROLES))}}}", file=sys.stderr)
        return 3

    role = argv[1]
    role_source = SOURCES_DIR / f"{role}-specific.md"
    shared_source = SOURCES_DIR / SHARED_SOURCE
    output_path = DIST_DIR / f"GPT_{role.upper()}_INSTRUCTIONS.md"

    # Source file check
    for path in (shared_source, role_source):
        if not path.is_file():
            print(f"ERROR: source file not found: {path}", file=sys.stderr)
            return 2

    # Read sources
    shared_text = shared_source.read_text(encoding="utf-8").rstrip()
    role_text = role_source.read_text(encoding="utf-8").rstrip()

    # Generated header — explicitly marks this file as built, not hand-edited
    built_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = (
        f"<!-- GENERATED FILE — do not edit. -->\n"
        f"<!-- Source: {SHARED_SOURCE} + {role}-specific.md -->\n"
        f"<!-- Built: {built_at} -->\n"
    )

    # Compose: header, blank line, shared, blank line, role-specific
    output_text = f"{header}\n{shared_text}\n\n{role_text}\n"

    # Char budget check — use len on the text that will actually be pasted
    # into the GPT admin panel (i.e., excluding the HTML comment header,
    # since the operator will strip that before pasting — or leave it in,
    # in which case it also counts. We measure WITH the header to be safe.)
    char_count = len(output_text)

    # Ensure dist dir exists and write
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text(output_text, encoding="utf-8")

    # Report
    print(f"Wrote: {output_path.relative_to(REPO_ROOT)}")
    print(f"Chars: {char_count:,} (shared: {len(shared_text):,}, {role}: {len(role_text):,})")

    if char_count >= HARD_LIMIT:
        print(
            f"FAIL: output is {char_count:,} chars, over the {HARD_LIMIT:,} hard limit. "
            f"Trim source files before deploying.",
            file=sys.stderr,
        )
        return 1

    if char_count >= SOFT_LIMIT:
        remaining = HARD_LIMIT - char_count
        print(
            f"WARNING: output is {char_count:,} chars, over the {SOFT_LIMIT:,} soft limit. "
            f"{remaining:,} chars of headroom remain before hard fail.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
