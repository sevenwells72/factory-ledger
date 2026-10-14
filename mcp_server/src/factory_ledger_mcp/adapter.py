"""Thin M0 read adapter: fixed FL routes, caller credentials, no business logic."""

import json
from importlib.resources import files
from urllib.parse import quote

import httpx
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from . import identity

CATALOG = json.loads(files("factory_ledger_mcp").joinpath("catalog.json").read_text())
# Historical names are a denial list, never executable tools. shipOrder is the
# read-only preview in floor/all; office retains its old write-name refusal.
WRITE_NAMES = frozenset(
    {
        "updateSupplierLot",
        "renameLot",
        "receive",
        "createExpectedReceipt",
        "ship",
        "make",
        "pack",
        "adjust",
        "voidTransaction",
        "createCustomer",
        "updateCustomer",
        "createOrder",
        "updateOrderHeader",
        "updateOrderStatus",
        "addOrderLines",
        "cancelOrderLine",
        "updateOrderLine",
        "commitShipOrder",
    }
)
MAX_RESPONSE_BYTES = 2_000_000
SAFE_POSTS = {
    ("office", "resolveProducts", "/products/resolve"),
    ("floor", "shipOrder", "/sales/orders/{order_id}/ship/preview"),
    ("all", "resolveProducts", "/products/resolve"),
    ("all", "shipOrder", "/sales/orders/{order_id}/ship/preview"),
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

    @staticmethod
    def caller_headers():
        """The calling user's own named-actor key, or no credential at all.

        Reads outside an authenticated request (the loopback demo stub without
        MCP_DEV_EMAIL) send nothing, so the backend answers 401 itself. A shared
        or settings-level key is never substituted.
        """
        try:
            return {"X-API-Key": identity.current_identity().actor_key}
        except PermissionError:
            return {}

    async def call(self, group, name, arguments):
        if name in WRITE_NAMES or (group == "office" and name == "shipOrder"):
            raise ToolFailure("write_blocked", "M0 is read-only; writes are disabled", 403)
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
        if name == "shipOrder":
            if body.get("ship_all") and "lines" in body:
                raise ToolFailure(
                    "invalid_arguments", "ship_all=true cannot be combined with explicit lines"
                )
            body["mode"] = "preview"
        kwargs = {"params": query, "headers": self.caller_headers()}
        if spec["method"] == "POST":
            kwargs["json"] = body
        return await self._request(spec["method"], path, **kwargs)

    async def _request(self, method, path, **kwargs):
        # The credential is decided per request from the caller's identity. Whatever the
        # client was constructed with, no other X-API-Key ever leaves this adapter.
        headers = dict(kwargs.pop("headers", None) or {})
        actor_key = headers.pop("X-API-Key", None)
        request = self.client.build_request(method, path, headers=headers, **kwargs)
        request.headers.pop("X-API-Key", None)
        request.headers.pop("Authorization", None)
        request.headers.pop("Cookie", None)
        if actor_key:
            request.headers["X-API-Key"] = actor_key
        try:
            response = await self.client.send(request, stream=True, follow_redirects=False)
            try:
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
                    raise ToolFailure("upstream_failure", "Ledger failed", response.status_code)
                try:
                    result = json.loads(data)
                except (ValueError, UnicodeDecodeError):
                    raise ToolFailure("invalid_response", "Ledger did not return valid JSON")
                if response.status_code >= 400:
                    raise ToolFailure(
                        "ledger_error", "Ledger rejected the request", response.status_code, result
                    )
                if isinstance(result, dict) and (
                    result.get("success") is False or "error" in result
                ):
                    raise ToolFailure(
                        "ledger_error", "Ledger reported a failed request", details=result
                    )
                return result
            finally:
                await response.aclose()
        except httpx.TimeoutException:
            raise ToolFailure("timeout", "Ledger timed out; no automatic retry was attempted")
        except httpx.RequestError:
            raise ToolFailure("unavailable", "Ledger is unavailable")
