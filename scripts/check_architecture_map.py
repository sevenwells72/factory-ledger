#!/usr/bin/env python3
"""Check that docs/ARCHITECTURE-MAP.md still points at real code.

Fails (exit 1) when:
  - a backticked path or glob in the map matches nothing;
  - a `file::text` anchor's text no longer appears in that file;
  - main.py has a `# ... ENDPOINT(S)` section banner the map does not name;
  - a root-level *.py module is not covered by any path in the map.
Read-only; prints one line per problem. Usage: python3 scripts/check_architecture_map.py
"""
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAP = os.path.join(ROOT, 'docs', 'ARCHITECTURE-MAP.md')
MAX_LINES = 160

TOKEN_RE = re.compile(r'`([^`]+)`')
PATH_RE = re.compile(r'^[\w.*?\[\]/-]+$')
EXT_RE = re.compile(r'\.(py|js|mjs|md|sql|html|css|json|toml|yml|yaml|sh)$')
BANNER_RE = re.compile(r'^# ([A-Z][A-Z (&)/-]* ENDPOINTS?)$', re.M)


def is_path(token):
    if not PATH_RE.match(token) or token.startswith('/'):
        return False
    return '/' in token or bool(EXT_RE.search(token))


def matches(token):
    pattern = os.path.join(ROOT, token)
    if token.endswith('/'):
        return [p for p in glob.glob(pattern.rstrip('/')) if os.path.isdir(p)]
    return glob.glob(pattern)


def check(map_text):
    problems = []
    covered = set()
    lines = map_text.count('\n') + 1
    if lines > MAX_LINES:
        problems.append(f'map is {lines} lines; keep it under {MAX_LINES}')
    for token in TOKEN_RE.findall(map_text):
        if '::' in token:
            path, text = token.split('::', 1)
            full = os.path.join(ROOT, path)
            if not os.path.isfile(full):
                problems.append(f'anchor file missing: {path}')
            elif text not in open(full, encoding='utf-8').read():
                problems.append(f'anchor text not found: {token}')
            covered.add(os.path.normpath(full))
        elif is_path(token):
            found = matches(token)
            if not found:
                problems.append(f'path matches nothing: {token}')
            covered.update(os.path.normpath(p) for p in found)
    main_py = open(os.path.join(ROOT, 'main.py'), encoding='utf-8').read()
    for banner in BANNER_RE.findall(main_py):
        if f'main.py::# {banner}' not in map_text:
            problems.append(f'main.py section not in map: # {banner}')
    for module in sorted(glob.glob(os.path.join(ROOT, '*.py'))):
        if os.path.normpath(module) not in covered:
            problems.append(f'root module not in map: {os.path.basename(module)}')
    return problems


def main():
    problems = check(open(MAP, encoding='utf-8').read())
    for problem in problems:
        print(problem)
    print(f'architecture map: {len(problems)} problem(s)')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
