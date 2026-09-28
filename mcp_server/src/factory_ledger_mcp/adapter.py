"""Finite read-only API adapter. No database, root app import, or write routing."""

import json
from importlib.resources import files
from urllib.parse import quote

import httpx
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

CATALOG = json.loads(files("factory_ledger_mcp").joinpath("catalog.json").read_text())
MAX_RESPONSE_BYTES = 2_000_000
SAFE_POSTS = {
    ("office", "resolveProducts", "/products/resolve"),
    ("floor", "shipOrder", "/sales/orders/{order_id}/ship/preview"),
}


class ToolFailure(Exception):
    def __init__(self, code, message, status=None, details=None):
        self.payload = {"error": code, "message": message}
        if status is not None:
            self.payload["status"] = status
        if details is not None:
            self.payload["details"] = details
        super().__init__(message)


class LedgerReader:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def call(self, group, name, arguments):
        spec = next((item for item in CATALOG.get(group, []) if item["name"] == name), None)
        if spec is None:
            raise ToolFailure("unknown_tool", "This tool is unavailable in this read-only group")
        if spec["method"] != "GET" and (group, name, spec["path"]) not in SAFE_POSTS:
            raise ToolFailure("write_blocked", "Phase 1 prohibits write operations")
        try:
            Draft202012Validator(spec["input_schema"], format_checker=FormatChecker()).validate(
                arguments
            )
        except ValidationError as exc:
            # Do not echo arbitrary arguments or credentials in validation errors.
            path = ".".join(str(part) for part in exc.absolute_path) or "arguments"
            raise ToolFailure("invalid_arguments", f"Invalid {path}: {exc.validator} constraint")
        path = spec["path"]
        for parameter in spec["path_parameters"]:
            value = str(arguments[parameter])
            # Prevent path normalization/encoded separators from routing to another endpoint.
            if not value or value in {".", ".."} or any(c in value for c in "/\\%?#\x00\r\n"):
                raise ToolFailure("invalid_arguments", f"Unsafe path parameter: {parameter}")
            path = path.replace("{" + parameter + "}", quote(value, safe=""))
        query = {key: arguments[key] for key in spec["query_parameters"] if key in arguments}
        body = {key: arguments[key] for key in spec["body_parameters"] if key in arguments}
        if group == "floor" and name == "shipOrder":
            body["mode"] = "preview"
        kwargs = {"params": query}
        if spec["method"] == "POST":
            kwargs["json"] = body
        try:
            async with self.client.stream(spec["method"], path, **kwargs) as response:
                data = bytearray()
                async for chunk in response.aiter_bytes():
                    data.extend(chunk)
                    if len(data) > MAX_RESPONSE_BYTES:
                        raise ToolFailure(
                            "response_too_large", "Result exceeds 2 MB; narrow filters"
                        )
                # Never forward upstream redirects, secrets, HTML or 5xx internals.
                if 300 <= response.status_code < 400:
                    raise ToolFailure("redirect_blocked", "Ledger redirects are disabled")
                if response.status_code >= 500:
                    raise ToolFailure(
                        "upstream_failure", "Local ledger failed", response.status_code
                    )
                try:
                    result = json.loads(data)
                except (ValueError, UnicodeDecodeError):
                    raise ToolFailure("invalid_response", "Ledger did not return valid JSON")
                if response.status_code >= 400:
                    raise ToolFailure(
                        "ledger_error", "Ledger rejected the read", response.status_code, result
                    )
                if isinstance(result, dict) and (
                    result.get("success") is False or "error" in result
                ):
                    raise ToolFailure(
                        "ledger_error", "Ledger reported a failed read", details=result
                    )
                return result
        except httpx.TimeoutException:
            raise ToolFailure("timeout", "Local ledger timed out; no automatic retry was attempted")
        except httpx.RequestError:
            raise ToolFailure("unavailable", "Local ledger is unavailable")
