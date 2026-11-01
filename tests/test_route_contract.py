"""Route-registration and OpenAPI contract snapshot (main.py modularization PR0).

Moving a section of main.py into a feature module must not change what the API
serves. This test pins, from the live `main.app`:

  * the route table IN ORDER (FastAPI/Starlette use the first matching route,
    and the /dashboard static mount swallows any /dashboard/* route registered
    after it): kind, path, methods, route name, status code, response model and
    class, schema inclusion, def vs async def, every parameter, and every
    dependency (by module.qualname, so a swapped auth dependency shows up);
  * the app-level middleware and exception handlers;
  * a normalized OpenAPI document: $refs are inlined and generated identifiers
    (operationId, summary, operation description, schema titles) are dropped,
    so renaming or moving a Pydantic model or a handler's docstring is not a
    false positive, while any change to a path, method, parameter, request
    body, response or field type is.

What a code move IS allowed to change: the handler's __module__ (not
recorded). Everything else must stay byte-identical.

Fixtures: tests/fixtures/route_contract/{routes,openapi}.jsonl, one JSON line
per route/operation, so a diff names exactly what moved.

Deliberate API changes regenerate the fixtures; the regenerating run is
reported as skipped, never green. Review the resulting git diff line by line:

    FL_UPDATE_ROUTE_CONTRACT=1 scripts/run_tests.sh tests/test_route_contract.py

No database is needed or touched.
"""

import copy
import inspect
import json
import os
import re
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from starlette.routing import Mount, Route

import main

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "route_contract"
ROUTES_FILE = FIXTURES / "routes.jsonl"
OPENAPI_FILE = FIXTURES / "openapi.jsonl"
UPDATE = os.environ.get("FL_UPDATE_ROUTE_CONTRACT") == "1"


# ─────────────────────────────────────────────────────────────────
# Route table
# ─────────────────────────────────────────────────────────────────

def _type_name(annotation):
    """Stable, module-free spelling of a parameter or model type."""
    if annotation is None:
        return None
    if isinstance(annotation, type):
        return annotation.__qualname__
    # typing constructs: drop module prefixes ("typing.", "main.", "datetime.")
    return re.sub(r"\b[A-Za-z_][A-Za-z0-9_]*\.", "", str(annotation))


def _call_name(call):
    if call is None:
        return None
    module = getattr(call, "__module__", None) or type(call).__module__
    name = getattr(call, "__qualname__", None) or type(call).__qualname__
    return f"{module}.{name}"


def _params(fields):
    return [
        {
            "name": f.name,
            "alias": f.alias,
            "required": bool(f.required),
            "type": _type_name(getattr(f.field_info, "annotation", None)),
        }
        for f in fields
    ]


def _dependant(dep):
    """Recursive dependency tree: the order of sub-dependencies is preserved."""
    return {
        "call": _call_name(dep.call),
        "path": _params(dep.path_params),
        "query": _params(dep.query_params),
        "header": _params(dep.header_params),
        "cookie": _params(dep.cookie_params),
        "body": _params(dep.body_params),
        "request": dep.request_param_name is not None,
        "dependencies": [_dependant(d) for d in dep.dependencies],
    }


def route_entry(index, route):
    if isinstance(route, APIRoute):
        tree = _dependant(route.dependant)
        tree.pop("call")  # the handler itself; its module may change on a move
        return {
            "i": index,
            "kind": "api",
            "path": route.path,
            "methods": sorted(route.methods),
            "name": route.name,
            "status_code": route.status_code,
            "response_model": _type_name(route.response_model),
            "response_class": _type_name(route.response_class.value
                                         if hasattr(route.response_class, "value")
                                         else route.response_class),
            "include_in_schema": route.include_in_schema,
            "deprecated": route.deprecated,
            "tags": list(route.tags or []),
            "async": inspect.iscoroutinefunction(route.endpoint),
            "route_dependencies": [_call_name(d.dependency) for d in route.dependencies],
            "dependant": tree,
        }
    if isinstance(route, Mount):
        return {"i": index, "kind": "mount", "path": route.path, "name": route.name,
                "app": type(route.app).__qualname__}
    if isinstance(route, Route):
        return {"i": index, "kind": "route", "path": route.path, "name": route.name,
                "methods": sorted(route.methods or [])}
    return {"i": index, "kind": type(route).__qualname__, "path": getattr(route, "path", None)}


def app_entry(app):
    return {
        "i": "app",
        "kind": "app",
        "middleware": [
            {"cls": m.cls.__qualname__,
             "dispatch": _call_name(m.options.get("dispatch")) if m.options.get("dispatch") else None}
            for m in app.user_middleware
        ],
        "exception_handlers": sorted(
            f"{_type_name(k) if isinstance(k, type) else k} -> {_call_name(v)}"
            for k, v in app.exception_handlers.items()
        ),
        "default_response_class": _type_name(app.router.default_response_class.value
                                             if hasattr(app.router.default_response_class, "value")
                                             else app.router.default_response_class),
    }


def route_table(app):
    return [app_entry(app)] + [route_entry(i, r) for i, r in enumerate(app.routes)]


# ─────────────────────────────────────────────────────────────────
# OpenAPI normalization
# ─────────────────────────────────────────────────────────────────

_GENERATED_OPERATION_KEYS = ("operationId", "summary", "description")
_UNORDERED_LISTS = ("required", "enum", "tags")


def _normalize_schema(node, components, stack=(), properties=False):
    if isinstance(node, list):
        return [_normalize_schema(v, components, stack) for v in node]
    if not isinstance(node, dict):
        return node
    if properties:
        # Keys of a `properties` mapping are field names, not schema keywords:
        # keep every one (a field may be called "title").
        return {k: _normalize_schema(v, components, stack) for k, v in node.items()}
    if "$ref" in node:
        name = node["$ref"].rsplit("/", 1)[-1]
        if name in stack:
            return {"$ref": "<recursive>"}
        return _normalize_schema(components[name], components, stack + (name,))
    out = {}
    for key, value in node.items():
        if key == "title":
            continue  # generated from class/field names ("Body_<handler>_<path>")
        value = _normalize_schema(value, components, stack, properties=(key == "properties"))
        if key in _UNORDERED_LISTS and isinstance(value, list) and all(isinstance(v, str) for v in value):
            value = sorted(value)
        out[key] = value
    return out


def normalize_openapi(spec):
    """{"<METHOD> <path>": normalized operation} plus an "__info__" entry."""
    spec = copy.deepcopy(spec)
    components = spec.get("components", {}).get("schemas", {})
    out = {"__info__": {"info": spec.get("info"), "openapi": spec.get("openapi"),
                        "securitySchemes": spec.get("components", {}).get("securitySchemes")}}
    for path, item in spec.get("paths", {}).items():
        for method, operation in item.items():
            op = {k: v for k, v in operation.items() if k not in _GENERATED_OPERATION_KEYS}
            op = _normalize_schema(op, components)
            if "parameters" in op:
                op["parameters"] = sorted(op["parameters"], key=lambda p: (p.get("in", ""), p.get("name", "")))
            out[f"{method.upper()} {path}"] = op
    return out


# ─────────────────────────────────────────────────────────────────
# Fixture I/O and readable diffs
# ─────────────────────────────────────────────────────────────────

def _dump(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(_dump(r) + "\n" for r in rows))


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _deep_diff(old, new, where="", out=None, limit=20):
    out = [] if out is None else out
    if len(out) >= limit:
        return out
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new), key=str):
            if k not in old:
                out.append(f"{where}/{k}: added {_dump(new[k])[:200]}")
            elif k not in new:
                out.append(f"{where}/{k}: removed (was {_dump(old[k])[:200]})")
            else:
                _deep_diff(old[k], new[k], f"{where}/{k}", out, limit)
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, (a, b) in enumerate(zip(old, new)):
            _deep_diff(a, b, f"{where}[{i}]", out, limit)
    elif old != new:
        out.append(f"{where or '/'}: {_dump(old)[:200]} -> {_dump(new)[:200]}")
    return out


def _regenerate_or_load(path, live_rows):
    if UPDATE:
        _write_jsonl(path, live_rows)
        pytest.skip(f"FL_UPDATE_ROUTE_CONTRACT=1: rewrote {path.name}; review the git diff, then rerun without it")
    assert path.exists(), f"{path} missing; generate it with FL_UPDATE_ROUTE_CONTRACT=1"
    return _read_jsonl(path)


HOW_TO = ("\nIf this API change is deliberate, regenerate with FL_UPDATE_ROUTE_CONTRACT=1 "
          "and get the fixture diff reviewed. A pure code move must never need that.")


# ─────────────────────────────────────────────────────────────────
# The contract tests
# ─────────────────────────────────────────────────────────────────

def test_route_table_matches_snapshot():
    live = json.loads(_dump(route_table(main.app)))
    pinned = _regenerate_or_load(ROUTES_FILE, live)
    problems = []
    if len(live) != len(pinned):
        problems.append(f"route count {len(pinned) - 1} -> {len(live) - 1}")
    for old, new in zip(pinned, live):
        if old != new:
            label = f"#{old.get('i')} {old.get('kind')} {old.get('methods', '')} {old.get('path', '')}"
            problems.append(label + "\n    " + "\n    ".join(_deep_diff(old, new)))
        if len(problems) >= 15:
            break
    if len(live) != len(pinned):
        tail = live[len(pinned):] or pinned[len(live):]
        problems += [f"unmatched: {r.get('methods', '')} {r.get('path')}" for r in tail[:10]]
    assert not problems, "Route table changed (order matters):\n" + "\n".join(problems) + HOW_TO


def test_openapi_matches_snapshot():
    live = normalize_openapi(main.app.openapi())
    live_rows = [{"op": k, "spec": json.loads(_dump(v))} for k, v in sorted(live.items())]
    pinned = {r["op"]: r["spec"] for r in _regenerate_or_load(OPENAPI_FILE, live_rows)}
    current = {r["op"]: r["spec"] for r in live_rows}
    problems = [f"added operation: {k}" for k in sorted(set(current) - set(pinned))]
    problems += [f"removed operation: {k}" for k in sorted(set(pinned) - set(current))]
    for key in sorted(set(current) & set(pinned)):
        if current[key] != pinned[key]:
            problems.append(f"changed: {key}\n    " + "\n    ".join(_deep_diff(pinned[key], current[key])))
    assert not problems, "OpenAPI contract changed:\n" + "\n".join(problems[:30]) + HOW_TO


def test_snapshot_covers_every_route():
    """Guards the snapshot itself: every registered route is represented."""
    table = route_table(main.app)
    assert len(table) - 1 == len(main.app.routes)
    api_keys = [(tuple(sorted(r.methods)), r.path) for r in main.app.routes if isinstance(r, APIRoute)]
    assert len(api_keys) == len(set(api_keys)), "duplicate METHOD+path registrations"


def test_dashboard_routes_registered_before_static_mount():
    """Starlette stops at the first full match. The /dashboard StaticFiles mount
    fully matches every /dashboard/* path, so an API route under /dashboard that
    is registered after it is unreachable. Extracted modules that own
    /dashboard/api/* routes must register before the mount."""
    mounts = [i for i, r in enumerate(main.app.routes) if isinstance(r, Mount) and r.path == "/dashboard"]
    assert len(mounts) == 1
    late = [f"{sorted(r.methods)} {r.path}" for i, r in enumerate(main.app.routes)
            if isinstance(r, APIRoute) and r.path.startswith("/dashboard/") and i > mounts[0]]
    assert not late, f"unreachable behind the /dashboard static mount: {late}"


# ─────────────────────────────────────────────────────────────────
# The normalizer itself: no false positives, no false negatives
# ─────────────────────────────────────────────────────────────────

def _spec(model_name="NoteCreate", field_type="string", extra_param=None, op_id="create_note_post"):
    params = [{"name": "b", "in": "query", "required": False, "schema": {"type": "string", "title": "B"}},
              {"name": "a", "in": "query", "required": True, "schema": {"type": "integer", "title": "A"}}]
    if extra_param:
        params.append(extra_param)
    return {
        "openapi": "3.1.0",
        "info": {"title": "t", "version": "1"},
        "paths": {"/notes": {"post": {
            "operationId": op_id, "summary": "Create Note", "description": "docstring",
            "parameters": params,
            "requestBody": {"content": {"application/json": {
                "schema": {"$ref": f"#/components/schemas/{model_name}"}}}, "required": True},
            "responses": {"200": {"description": "Successful Response"}},
        }}},
        "components": {"schemas": {model_name: {
            "title": model_name, "type": "object", "required": ["title", "category"],
            "properties": {"title": {"type": field_type, "title": "Title"},
                           "category": {"type": "string", "title": "Category"}},
        }}},
    }


def test_normalizer_ignores_generated_identifiers_and_ordering():
    base = normalize_openapi(_spec())
    renamed = normalize_openapi(_spec(model_name="notes_api__NoteCreate", op_id="other_id"))
    reordered = _spec()
    reordered["paths"]["/notes"]["post"]["parameters"].reverse()
    reordered["components"]["schemas"]["NoteCreate"]["required"].reverse()
    assert renamed == base
    assert normalize_openapi(reordered) == base
    # a field literally named "title" survives title-stripping
    body = base["POST /notes"]["requestBody"]["content"]["application/json"]["schema"]
    assert set(body["properties"]) == {"title", "category"}


def test_normalizer_detects_meaningful_changes():
    base = normalize_openapi(_spec())
    assert normalize_openapi(_spec(field_type="integer")) != base
    assert normalize_openapi(_spec(extra_param={"name": "c", "in": "header", "required": True,
                                                "schema": {"type": "string"}})) != base
    required_changed = _spec()
    required_changed["components"]["schemas"]["NoteCreate"]["required"] = ["category"]
    assert normalize_openapi(required_changed) != base
