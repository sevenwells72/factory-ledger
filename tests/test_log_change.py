"""scripts/log_change.py: adds change-log entries safely. Uses only temp files; no database."""
import importlib.util
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, 'scripts', 'log_change.py')
spec = importlib.util.spec_from_file_location('log_change', SCRIPT)
log_change = importlib.util.module_from_spec(spec)
spec.loader.exec_module(log_change)

PROJECT = ('# Change Log\n\n## 2026-01-02 10:00 — Older entry\n- **File(s) changed:** `a.py`\n'
           '- **What changed:** x\n- **Why:** y\n\n---\n\n')
GLOBAL = ('# Global Change Log\n\n| Date | Project | Summary | Files |\n|------|---------|---------|-------|\n'
          '| 2026-01-02 10:00 | `/p` | Older entry | `a.py` |\n')


@pytest.fixture
def logs(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    (root / 'CHANGE_LOG.md').write_text(PROJECT)
    cloud = tmp_path / 'cloud'
    cloud.mkdir()
    canonical = cloud / 'change-log.md'
    canonical.write_text(GLOBAL)
    link = tmp_path / 'change-log.md'
    link.symlink_to(canonical)
    return root, link, canonical


def args(root, link, canonical, summary='New thing', *extra):
    return ['--summary', summary, '--files', 'main.py, tests/test_x.py', '--what', 'Did it',
            '--why', 'Because', '--when', '2026-10-31 21:00', '--project-root', str(root),
            '--global-log', str(link), '--canonical', str(canonical), *extra]


def test_adds_entry_and_row_and_preserves_existing_text(logs):
    root, link, canonical = logs
    assert log_change.main(args(root, link, canonical)) == 0
    project = (root / 'CHANGE_LOG.md').read_text()
    assert project == ('# Change Log\n\n## 2026-10-31 21:00 — New thing\n'
                       '- **File(s) changed:** `main.py`, `tests/test_x.py`\n'
                       '- **What changed:** Did it\n- **Why:** Because\n\n---\n\n') + PROJECT[len('# Change Log\n\n'):]
    lines = canonical.read_text().split('\n')
    sep = lines.index('|------|---------|---------|-------|')
    assert lines[sep + 1] == f'| 2026-10-31 21:00 | `{root}` | New thing | `main.py`, `test_x.py` |'
    assert canonical.read_text().replace(lines[sep + 1] + '\n', '', 1) == GLOBAL
    assert link.is_symlink() and os.path.realpath(link) == str(canonical)


def test_duplicate_is_refused_and_nothing_changes(logs):
    root, link, canonical = logs
    assert log_change.main(args(root, link, canonical)) == 0
    before = ((root / 'CHANGE_LOG.md').read_bytes(), canonical.read_bytes())
    assert log_change.main(args(root, link, canonical)) == 2
    assert ((root / 'CHANGE_LOG.md').read_bytes(), canonical.read_bytes()) == before


@pytest.mark.parametrize('breakage', ['regular_file', 'wrong_target', 'missing_separator', 'separator_above_header'])
def test_bad_global_log_is_refused_before_any_write(logs, tmp_path, breakage):
    root, link, canonical = logs
    if breakage == 'regular_file':
        link.unlink()
        link.write_text(GLOBAL)
    elif breakage == 'wrong_target':
        other = tmp_path / 'other.md'
        other.write_text(GLOBAL)
        link.unlink()
        link.symlink_to(other)
    elif breakage == 'missing_separator':
        canonical.write_text(GLOBAL.replace('|------|---------|---------|-------|\n', ''))
    else:
        canonical.write_text(GLOBAL.replace('| Date | Project | Summary | Files |\n|------|---------|---------|-------|\n',
                                            '|------|---------|---------|-------|\n| Date | Project | Summary | Files |\n'))
    before = ((root / 'CHANGE_LOG.md').read_bytes(), canonical.read_bytes())
    assert log_change.main(args(root, link, canonical)) == 2
    assert ((root / 'CHANGE_LOG.md').read_bytes(), canonical.read_bytes()) == before
    if breakage == 'regular_file':
        assert not link.is_symlink()


def test_bad_project_log_and_inputs_are_refused(logs):
    root, link, canonical = logs
    (root / 'CHANGE_LOG.md').write_text('Change Log\n' + PROJECT)
    assert log_change.main(args(root, link, canonical)) == 2
    (root / 'CHANGE_LOG.md').write_text(PROJECT)
    assert log_change.main(args(root, link, canonical, 'two\nlines')) == 2
    assert log_change.main(args(root, link, canonical, 'a | b')) == 2
    assert (root / 'CHANGE_LOG.md').read_text() == PROJECT and canonical.read_text() == GLOBAL


def test_dry_run_writes_nothing(logs):
    root, link, canonical = logs
    assert log_change.main(args(root, link, canonical, 'New thing', '--dry-run')) == 0
    assert (root / 'CHANGE_LOG.md').read_text() == PROJECT and canonical.read_text() == GLOBAL


def test_failed_write_restores_original(logs, monkeypatch):
    root, link, canonical = logs
    real_fsync, calls = os.fsync, []

    def failing_fsync(fd):
        calls.append(fd)
        if len(calls) == 1:
            raise OSError('disk full')
        real_fsync(fd)

    monkeypatch.setattr(log_change.os, 'fsync', failing_fsync)
    with pytest.raises(OSError):
        log_change.main(args(root, link, canonical, 'New thing', '--project-only'))
    assert (root / 'CHANGE_LOG.md').read_text() == PROJECT


def test_concurrent_writers_do_not_lose_entries(logs):
    root, link, canonical = logs
    procs = [subprocess.Popen([sys.executable, SCRIPT, *args(root, link, canonical, f'Writer {n}')],
                              stdout=subprocess.DEVNULL, stderr=subprocess.PIPE) for n in range(8)]
    assert [p.wait() for p in procs] == [0] * 8
    project, glob_text = (root / 'CHANGE_LOG.md').read_text(), canonical.read_text()
    for n in range(8):
        assert project.count(f'## 2026-10-31 21:00 — Writer {n}\n') == 1
        assert glob_text.count(f'| Writer {n} |') == 1
    assert project.startswith('# Change Log\n\n') and project.endswith(PROJECT[len('# Change Log\n\n'):])
    assert glob_text.endswith('| 2026-01-02 10:00 | `/p` | Older entry | `a.py` |\n')
