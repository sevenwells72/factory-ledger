"""Feature modules never import main (main.py modularization PR0).

main.py imports every feature module at its top, then hands itself over as
`api` in `register_routes(app, api)`. A feature module that imports main
instead would create a circular import (main is half-initialized while its
imports run) and would bind helpers at import time, so tests that monkeypatch
`main.get_db_connection` & co. would silently stop reaching the moved code.
The rule: reach main only through the `api` argument, looked up at call time.

Also pins that every module defining `register_routes` is actually registered
by main or by a module main registers; an extracted module that is never
wired in would make its routes 404 without failing any import.

Static (AST) checks only; nothing is imported or executed.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENTRY = "main"


def _root_modules():
    return sorted(p for p in ROOT.glob("*.py") if p.stem != ENTRY)


def _const_str(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def main_imports(source, filename="<src>"):
    """Every way `source` reaches the main module: imports, dynamic imports,
    or sys.modules lookups by name. Returns ["line N: ..."]."""
    found = []
    for node in ast.walk(ast.parse(source, filename)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == ENTRY:
                    found.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and (node.module or "").split(".")[0] == ENTRY:
                found.append(f"line {node.lineno}: from {node.module} import ...")
            elif node.level and any(a.name == ENTRY for a in node.names):
                found.append(f"line {node.lineno}: relative import of {ENTRY}")
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            args = [_const_str(a) for a in node.args[:1]]
            if name in ("import_module", "__import__") and args and args[0] == ENTRY:
                found.append(f"line {node.lineno}: {name}('{ENTRY}')")
            if (name == "get" and isinstance(func, ast.Attribute)
                    and isinstance(func.value, ast.Attribute) and func.value.attr == "modules"
                    and args and args[0] == ENTRY):
                found.append(f"line {node.lineno}: sys.modules.get('{ENTRY}')")
        elif isinstance(node, ast.Subscript):
            if (isinstance(node.value, ast.Attribute) and node.value.attr == "modules"
                    and _const_str(node.slice) == ENTRY):
                found.append(f"line {node.lineno}: sys.modules['{ENTRY}']")
    return found


def test_no_feature_module_imports_main():
    offenders = {}
    for path in _root_modules():
        hits = main_imports(path.read_text(), str(path))
        if hits:
            offenders[path.name] = hits
    assert not offenders, (
        "Feature modules must reach main only via the `api` argument of "
        f"register_routes(app, api), never by importing it: {offenders}"
    )


def test_guard_catches_every_import_form():
    """The checker itself: each spelling is caught, look-alikes are not."""
    caught = [
        "import main",
        "import main as api",
        "from main import get_transaction",
        "from main import *",
        "import importlib\nimportlib.import_module('main')",
        "__import__('main')",
        "import sys\nsys.modules['main']",
        "import sys\nsys.modules.get('main')",
        "def f():\n    import main\n",
    ]
    for source in caught:
        assert main_imports(source), source
    allowed = [
        "import mainframe",
        "from domain import main",
        "import sys\nsys.modules[__name__]",
        "x = 'main'",
        "def register_routes(app, api):\n    return api.get_transaction()\n",
    ]
    for source in allowed:
        assert not main_imports(source), source


def _module_aliases(tree, modules):
    """{local name: module} for `import x` / `import x as y` / `from . import x`."""
    names = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in modules:
                    names[a.asname or a.name] = a.name
        elif isinstance(node, ast.ImportFrom) and node.module is None:
            for a in node.names:
                if a.name in modules:
                    names[a.asname or a.name] = a.name
    return names


def test_every_register_routes_module_is_registered():
    sources = {p.stem: p.read_text() for p in ROOT.glob("*.py")}
    trees = {name: ast.parse(src, name) for name, src in sources.items()}
    defines = {name for name, tree in trees.items()
               if any(isinstance(n, ast.FunctionDef) and n.name == "register_routes" for n in tree.body)}
    called = set()
    for name, tree in trees.items():
        aliases = _module_aliases(tree, defines)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "register_routes"
                    and isinstance(node.func.value, ast.Name) and node.func.value.id in aliases):
                called.add(aliases[node.func.value.id])
    assert defines, "expected feature modules exposing register_routes"
    assert defines <= called, f"register_routes defined but never called: {sorted(defines - called)}"
