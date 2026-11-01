"""docs/ARCHITECTURE-MAP.md must keep pointing at real code (scripts/check_architecture_map.py)."""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location('check_architecture_map',
                                              os.path.join(ROOT, 'scripts', 'check_architecture_map.py'))
check_map = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_map)


def test_map_points_at_existing_code():
    with open(check_map.MAP, encoding='utf-8') as f:
        assert check_map.check(f.read()) == []


def test_checker_reports_stale_pointers():
    problems = check_map.check('`no_such_module.py` `main.py::def no_such_function` `tests/test_no_such_*`')
    assert 'path matches nothing: no_such_module.py' in problems
    assert 'anchor text not found: main.py::def no_such_function' in problems
    assert 'path matches nothing: tests/test_no_such_*' in problems
    assert any(p.startswith('main.py section not in map: # RECEIVE ENDPOINTS') for p in problems)
    assert 'root module not in map: main.py' not in problems  # covered via the anchor's file


def test_map_length_is_not_a_failure():
    with open(check_map.MAP, encoding='utf-8') as f:
        text = f.read()
    padding = ''.join(f'\n- note {n}' for n in range(300))  # a much longer, still valid map
    assert check_map.check(text + padding) == []
