"""Generate the reviewed read-only catalog, never modify the source OpenAPI files."""

import copy
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SOURCES = {"office": "openapi-gpt-v3.yaml", "floor": "gpt-configs/schemas/openapi-floor.yaml"}
OUTPUT = ROOT / "mcp_server/src/factory_ledger_mcp/catalog.json"


def resolve(value, document):
    if isinstance(value, list):
        return [resolve(item, document) for item in value]
    if not isinstance(value, dict):
        return value
    if "$ref" in value:
        ref = document
        for segment in value["$ref"].removeprefix("#/").split("/"):
            ref = ref[segment]
        return resolve(copy.deepcopy(ref), document)
    result = {key: resolve(item, document) for key, item in value.items()}
    if result.get("type") == "object":
        result["additionalProperties"] = False
    return result


def build():
    catalog = {}
    for group, source in SOURCES.items():
        document = yaml.safe_load((ROOT / source).read_text())
        entries = []
        for path, methods in document["paths"].items():
            for method, operation in methods.items():
                if method not in {"get", "post", "patch", "put", "delete"}:
                    continue
                name = operation["operationId"]
                if not (
                    method == "get"
                    or name == "resolveProducts"
                    or (group == "floor" and name == "shipOrder")
                ):
                    continue
                properties, required, path_names, query_names = {}, [], [], []
                for param in operation.get("parameters", []):
                    properties[param["name"]] = resolve(param["schema"], document)
                    if "description" in param:
                        properties[param["name"]]["description"] = param["description"]
                    if param.get("required"):
                        required.append(param["name"])
                    (path_names if param["in"] == "path" else query_names).append(param["name"])
                body = resolve(
                    operation.get("requestBody", {})
                    .get("content", {})
                    .get("application/json", {})
                    .get("schema", {}),
                    document,
                )
                properties.update(body.get("properties", {}))
                required.extend(body.get("required", []))
                description = operation["summary"]
                if name == "shipOrder":
                    description = (
                        "Preview a sales-order shipment. Read-only: cannot dispatch or "
                        "commit. Show quantities, shortages and warnings to the operator."
                    )
                entries.append(
                    {
                        "name": name,
                        "method": method.upper(),
                        # Dedicated backend wrapper overwrites mode=preview too: defense in depth.
                        "path": path + "/preview" if name == "shipOrder" else path,
                        "source_path": path,
                        "source": source,
                        "description": description,
                        "path_parameters": path_names,
                        "query_parameters": query_names,
                        "body_parameters": list(body.get("properties", {})),
                        "input_schema": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": False,
                        },
                    }
                )
        catalog[group] = entries
    return catalog


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
