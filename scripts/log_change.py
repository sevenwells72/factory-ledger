#!/usr/bin/env python3
"""Add one entry to the project CHANGE_LOG.md and one row to the global change log.

Neither log is printed, so a session never loads the 1 MB project log into context.
Follows the Change Log Protocol in CLAUDE.md:
  - project entry goes directly below '# Change Log';
  - global row goes directly below the '|------|' separator row;
  - ~/change-log.md must be a symlink to the canonical iCloud file and stays one
    (the file it points to is rewritten in place, never replaced).
Refuses (exit 2, nothing written) on a bad structure, a wrong symlink, or an entry whose
heading or row key already exists with different text.
Retry-safe: a log that already holds the identical entry is left untouched, so re-running the
same command (same --when) after a partial failure only adds what is missing (exit 0).
If the global write fails after the project entry was written, it exits 1 and prints that command.
Each file is locked with flock while it is read, checked and written, so two runs of this
helper cannot interleave; tools that do not lock (the Edit tool, other agents) are not blocked.
If a write fails or reads back wrong, the original bytes are restored.

Usage:
  scripts/log_change.py --summary "Fixed X" --files "main.py, tests/test_x.py" \
      --what "What changed" --why "Why"   [--when "YYYY-MM-DD HH:MM"] [--project-only] [--dry-run]
"""
import argparse
import datetime
import fcntl
import os
import sys

HEADER = '# Change Log\n\n'
GLOBAL_HEADER_ROW = '| Date | Project | Summary | Files |'
GLOBAL_SEPARATOR = '|------|---------|---------|-------|'
CANONICAL_GLOBAL = '~/Library/Mobile Documents/com~apple~CloudDocs/Claude Logs/change-log.md'


class Refused(Exception):
    pass


class PartialWrite(Exception):
    pass


def project_entry(when, summary, files, what, why):
    listed = ', '.join(f'`{f}`' for f in files)
    return (f'## {when} — {summary}\n'
            f'- **File(s) changed:** {listed}\n'
            f'- **What changed:** {what}\n'
            f'- **Why:** {why}\n\n---\n\n')


def global_row(when, project_root, summary, files):
    names = ', '.join(f'`{os.path.basename(f)}`' for f in files)
    return f'| {when} | `{project_root}` | {summary} | {names} |'


def add_project_entry(text, entry):
    """Return (new_text, added). An identical existing entry is kept as is (retry)."""
    if not text.startswith(HEADER):
        raise Refused('project log does not start with "# Change Log" and a blank line')
    if '\n' + entry in text:
        return text, False
    heading = entry.split('\n', 1)[0]
    if '\n' + heading + '\n' in text:
        raise Refused(f'project log already has this heading with different text: {heading}')
    return HEADER + entry + text[len(HEADER):], True


def add_global_row(text, row):
    """Return (new_text, added). An identical existing row is kept as is (retry)."""
    lines = text.split('\n')
    if lines.count(GLOBAL_SEPARATOR) != 1:
        raise Refused('global log must contain exactly one table separator row')
    i = lines.index(GLOBAL_SEPARATOR)
    if i == 0 or lines[i - 1] != GLOBAL_HEADER_ROW:
        raise Refused('global log separator row is not directly below the table header row')
    if row in lines:
        return text, False
    key = row.rsplit(' | ', 1)[0] + ' | '  # date, project and summary columns
    if any(line.startswith(key) for line in lines):
        raise Refused('global log already has a row for this date, project and summary with different files')
    lines.insert(i + 1, row)
    return '\n'.join(lines), True


def resolve_global(link, canonical):
    if not os.path.islink(link):
        raise Refused(f'{link} is not a symlink; stop and report it (CLAUDE.md change-log rule)')
    target = os.path.realpath(link)
    if target != os.path.realpath(canonical):
        raise Refused(f'{link} points to {target}, expected {canonical}')
    if not os.path.isfile(target):
        raise Refused(f'global log target is missing: {target}')
    return target


def rewrite_locked(path, transform, dry_run):
    """Read, transform and rewrite `path` in place while holding an exclusive flock.

    `transform` returns (new_text, added); nothing is written when added is False.
    Returns added.
    """
    with open(path, 'r+', encoding='utf-8', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            old = f.read()
            new, added = transform(old)
            if dry_run or not added:
                return added
            try:
                f.seek(0)
                f.write(new)
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
                f.seek(0)
                if f.read() != new:
                    raise OSError('read-back after write did not match')
            except Exception:
                f.seek(0)
                f.write(old)
                f.truncate()
                f.flush()
                os.fsync(f.fileno())
                raise
            return True
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run(args):
    files = [f.strip() for f in args.files.split(',') if f.strip()]
    for field in ('summary', 'what', 'why'):
        if not getattr(args, field).strip() or '\n' in getattr(args, field):
            raise Refused(f'--{field} must be one non-empty line')
    if not files:
        raise Refused('--files must name at least one file')
    if '|' in args.summary:
        raise Refused('--summary must not contain "|" (it would break the global table)')
    when = args.when or datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    datetime.datetime.strptime(when, '%Y-%m-%d %H:%M')
    root = os.path.abspath(args.project_root)
    project_log = os.path.join(root, 'CHANGE_LOG.md')
    if os.path.islink(project_log) or not os.path.isfile(project_log):
        raise Refused(f'project log must be a regular file: {project_log}')

    entry = project_entry(when, args.summary, files, args.what, args.why)
    row = global_row(when, root, args.summary, files)
    target = None
    if not args.project_only:
        target = resolve_global(os.path.expanduser(args.global_log), os.path.expanduser(args.canonical))
        with open(target, encoding='utf-8') as f:
            add_global_row(f.read(), row)  # validate before touching either file

    def report(added, what):
        verb = ('would add' if args.dry_run else 'added') if added else 'already present, unchanged'
        print(f'{verb}: {what}')

    report(rewrite_locked(project_log, lambda text: add_project_entry(text, entry), args.dry_run),
           f'project entry {entry.splitlines()[0]}')
    if target:
        try:
            added = rewrite_locked(target, lambda text: add_global_row(text, row), args.dry_run)
        except (OSError, Refused) as exc:
            raise PartialWrite(
                f'project entry is in {project_log} but the global row was NOT written ({exc}). '
                f'Fix the cause, then re-run the same command with --when "{when}"; '
                'the existing project entry is kept and only the global row is added.') from exc
        report(added, f'global row in {target}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--summary', required=True)
    p.add_argument('--files', required=True, help='comma-separated repo-relative paths')
    p.add_argument('--what', required=True)
    p.add_argument('--why', required=True)
    p.add_argument('--when', help='YYYY-MM-DD HH:MM (default: now, local time)')
    p.add_argument('--project-root', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p.add_argument('--global-log', default='~/change-log.md')
    p.add_argument('--canonical', default=CANONICAL_GLOBAL)
    p.add_argument('--project-only', action='store_true', help='skip the global log')
    p.add_argument('--dry-run', action='store_true', help='validate only; write nothing')
    args = p.parse_args(argv)
    try:
        run(args)
    except (Refused, ValueError) as exc:
        print(f'REFUSED: {exc}', file=sys.stderr)
        return 2
    except PartialWrite as exc:
        print(f'PARTIAL: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
