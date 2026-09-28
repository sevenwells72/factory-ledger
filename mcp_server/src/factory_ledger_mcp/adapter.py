"""Bounded ledger adapter with payload-bound, two-call write approval.

Approval metadata is durable SQLite, separate from the ledger. Consume BEFORE
sending a mutation; never retry an uncertain outcome. The unmodified backend has
no transactional idempotency or conditional-version API: the read/check/write gap
and end-to-end idempotency remain backend work, not guarantees of this adapter.
Blocked named-actor routes are never redirected to a less restricted endpoint.

Credentials: every ledger request, read or write, carries only the calling user's
own named-actor key from the request-scoped identity. There is no service-level or
shared key; a default X-API-Key on the HTTP client is dropped before sending.
"""

import hashlib
import json
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from urllib.parse import quote

import httpx
from jsonschema import Draft202012Validator, FormatChecker, ValidationError

from . import identity

CATALOG = json.loads(files("factory_ledger_mcp").joinpath("catalog.json").read_text())
WRITE_CATALOG = CATALOG.pop("_writes")
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
    def __init__(self, client: httpx.AsyncClient, confirmations=None, *, environment="local"):
        self.client = client
        self.confirmations = confirmations
        self.environment = environment

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
        write = next((s for s in WRITE_CATALOG.get(group, []) if s["name"] == name), None)
        if write is not None:
            return await self._write(group, write, arguments)
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
        if actor_key:
            request.headers["X-API-Key"] = actor_key
        try:
            response = await self.client.send(request, stream=True)
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
                    raise ToolFailure(
                        "upstream_failure", "Local ledger failed", response.status_code
                    )
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
            raise ToolFailure("timeout", "Local ledger timed out; no automatic retry was attempted")
        except httpx.RequestError:
            raise ToolFailure("unavailable", "Local ledger is unavailable")

    async def _write(self, group, spec, arguments):
        # Resolve exactly once. Never accept email, role, actor, or keys in tool arguments.
        try:
            person = identity.current_identity()
        except PermissionError:
            raise ToolFailure(
                "unauthenticated", "An authenticated, allowlisted identity is required", 401
            )
        permission = identity.can_write_office if group == "office" else identity.can_write_floor
        if not permission(person):
            raise ToolFailure("forbidden", f"Your role cannot write to {group}", 403)
        validate(spec, arguments)
        if not self.confirmations:
            raise ToolFailure("confirmation_unavailable", "Durable approval storage is unavailable")
        phase = arguments.get("phase", "preview")
        token = arguments.get("confirmation_token")
        payload = {k: v for k, v in arguments.items() if k not in {"phase", "confirmation_token"}}
        # Check path components even for blocked tools; none can escape their route.
        route(spec["path"], payload, spec["path_parameters"])
        if spec["name"] in {"shipOrder", "commitShipOrder"}:
            shipping_arguments(payload)
        binding = digest(
            [self.environment, str(self.client.base_url), group, spec["name"], payload]
        )
        principal = digest([person.email, person.role, person.actor_key])
        if phase == "commit":
            if not token:
                raise ToolFailure(
                    "confirmation_required", "Preview first, then approve with its token"
                )
            # Durable atomic claim before any await: concurrent/restarted callers cannot reuse it.
            pending = self.confirmations.claim(token, principal, binding)
        elif token:
            raise ToolFailure("invalid_arguments", "A token is only accepted with phase=commit")
        else:
            pending = None

        headers = {"X-API-Key": person.actor_key}
        sent = False
        try:
            # A trusted mapping must still name a backend actor, never a shared/master key.
            who = await self._request("GET", "/auth/whoami", headers=headers)
            if who.get("key_kind") != "actor" or not who.get("actor", {}).get("name"):
                raise ToolFailure(
                    "named_actor_required", "The mapped backend key must name a person", 403
                )
            actor = {"email": person.email, "role": person.role, "name": who["actor"]["name"]}
            state, proposal, summary = await self._prepare(spec, payload, headers)
            blockers = []
            if not spec["named_actor_allowed"]:
                blockers.append(
                    "This backend endpoint rejects named-actor keys; no save is available."
                )
            if spec["name"] == "createOrder":
                blockers.append(
                    "The backend create contract lacks dedicated customer_po, external-reference "
                    "uniqueness, and resolved product/service inputs. No order will be created."
                )
            if spec["name"] == "updateOrderHeader" and "customer_po" in payload:
                blockers.append("The backend header contract cannot persist customer_po.")
            revision = digest(state)
            if pending is None:
                issued = self.confirmations.issue(principal, binding, revision)
                return {
                    "success": False,
                    "phase": "preview",
                    "saved": False,
                    "summary": summary
                    + (" Save blocked: " + " ".join(blockers) if blockers else ""),
                    "proposal": proposal,
                    "confirmation_token": issued["token"],
                    "approval_id": issued["approval_id"],
                    "expires_at": issued["expires_at"],
                    "can_commit": not blockers,
                    "blockers": blockers,
                    "warnings": state.get("preview", {}).get("warnings", []),
                    "instruction": "Show this summary and warnings to the operator. After explicit "
                    "approval, call again with phase=commit, the unchanged inputs and token.",
                }
            if pending["revision"] != revision:
                raise ToolFailure(
                    "stale_confirmation", "Relevant records changed; preview again", 409
                )
            if self.confirmations.clock() >= pending["expires"]:
                raise ToolFailure("confirmation_expired", "Approval expired; preview again", 409)
            if blockers:
                raise ToolFailure("backend_write_blocked", " ".join(blockers), 403)
            # All routes below were explicitly checked with real handlers and actor keys.
            path = route(spec["path"], payload, spec["path_parameters"])
            query = {k: payload[k] for k in spec["query_parameters"] if k in payload}
            body = {k: payload[k] for k in spec["body_parameters"] if k in payload}
            if spec["name"] == "createExpectedReceipt":
                body["product_id"] = state["product"]["id"]
                body.pop("product_name", None)
            sent = True
            result = await self._request(
                spec["method"], path, params=query, json=body, headers=headers
            )
            receipt = {
                "success": True,
                "saved": True,
                "operation": spec["name"],
                "approval_id": pending["approval_id"],
                "request_id": pending["approval_id"],
                "completed_at": timestamp(self.confirmations.clock()),
                "actor": actor,
                "order_number": state.get("order", {}).get("order_number"),
                "order_id": state.get("order", {}).get("order_id"),
                "changed_fields": payload,
                "result": result,
                "warnings": result.get("warnings") or [],
                "verification": "verified",
                "resulting_state": result,
            }
            if "order" in state:
                receipt["customer_po"] = state["order"]["customer_po"]
                receipt["flags"] = state["order"]["flags"]
                try:
                    fresh = await self._order(payload["order_id"], headers)
                    receipt["resulting_state"] = fresh
                    verify_order_readback(spec["name"], payload, state["order"], fresh, result)
                except ToolFailure:
                    receipt["verification"] = "pending"
                    receipt["message"] = "Saved; verification pending. Do not resubmit the write."
            if spec["name"] == "commitShipOrder":
                receipt["shipment_id"] = result.get("shipment_id")
                receipt["transaction_ids"] = [
                    row["transaction_id"]
                    for row in result.get("lines_shipped", [])
                    if "transaction_id" in row
                ]
                receipt["confirmation_codes"] = [
                    row["confirmation_code"]
                    for row in result.get("lines_shipped", [])
                    if "confirmation_code" in row
                ]
            receipt["changed_record_ids"] = {
                k: v
                for k, v in (result | {"order_id": receipt["order_id"]}).items()
                if k.endswith("_id") and v is not None
            }
            self.confirmations.finish(pending["approval_id"], "saved", receipt)
            return receipt
        except BaseException as exc:
            if pending:
                # Even an HTTP error may follow a committed mutation. Reconcile with reads.
                self.confirmations.finish(
                    pending["approval_id"], "uncertain" if sent else "rejected"
                )
            if sent and isinstance(exc, ToolFailure):
                raise ToolFailure(
                    "write_outcome_uncertain",
                    "The write was sent but completion could not be "
                    "verified. Do not retry; reconcile using read tools.",
                    details={"approval_id": pending["approval_id"], "cause": exc.payload},
                ) from exc
            raise

    async def _order(self, order_id, headers):
        detail = await self._request(
            "GET",
            route("/sales/orders/{order_id}", {"order_id": order_id}, ["order_id"]),
            headers=headers,
        )
        # Legacy detail omits customer_po; list is the existing read contract that exposes it.
        orders = await self._request(
            "GET",
            "/sales/orders",
            params={"customer": detail["customer"], "limit": 200},
            headers=headers,
        )
        row = next(
            (r for r in orders["orders"] if r["order_number"] == detail["order_number"]), None
        )
        if row is None:
            raise ToolFailure(
                "verification_incomplete",
                "Cannot verify the dedicated PO field in the bounded order list",
            )
        detail["customer_po"] = row.get("customer_po")
        detail["flags"] = [] if detail["customer_po"] else ["No PO"]
        return detail

    async def _product(self, product_id, headers):
        result = await self._request("GET", f"/products/{product_id}", headers=headers)
        # Real /products/{id} returns a product object, not a search result.
        if not result.get("active", True):
            raise ToolFailure("invalid_product", "The selected product is inactive", 409)
        return result

    async def _prepare(self, spec, payload, headers):
        name = spec["name"]
        state = {}
        proposal = dict(payload)
        if "order_id" in payload:
            state["order"] = await self._order(payload["order_id"], headers)
            order = state["order"]
            state["allocations"] = await self._request(
                "GET", f"/sales/orders/{order['order_id']}/allocations", headers=headers
            )
            target_po = payload.get("customer_po", order["customer_po"])
            proposal.update(
                order_number=order["order_number"],
                customer_po=target_po,
                flags=[] if target_po else ["No PO"],
            )
            if name == "updateOrderHeader":
                if not set(payload) - {"order_id"}:
                    raise ToolFailure("invalid_arguments", "Supply at least one header field")
                if order["status"] not in {"new", "confirmed"}:
                    raise ToolFailure(
                        "invalid_state", "Only new or confirmed order headers may be edited", 409
                    )
                if "customer_id" in payload:
                    state["customer"] = await self._customer(payload["customer_id"], headers)
                    candidates = await self._request(
                        "GET",
                        "/sales/orders",
                        params={"customer": state["customer"]["name"], "limit": 200},
                        headers=headers,
                    )
                    if len(candidates["orders"]) >= 200:
                        raise ToolFailure(
                            "duplicate_check_incomplete",
                            "Cannot check the new customer's PO duplicates",
                            409,
                        )
                    for candidate in candidates["orders"]:
                        if (
                            candidate["order_id"] != order["order_id"]
                            and order["customer_po"]
                            and normalize_po(candidate.get("customer_po"))
                            == normalize_po(order["customer_po"])
                        ):
                            raise ToolFailure(
                                "duplicate_po", "The new customer already has this PO", 409
                            )
            if name == "updateOrderStatus":
                transitions = {
                    "new": ["confirmed", "cancelled"],
                    "confirmed": ["in_production", "cancelled"],
                    "in_production": ["ready", "cancelled"],
                    "ready": ["in_production", "cancelled"],
                    "partial_ship": ["cancelled"],
                    "shipped": ["invoiced"],
                }
                if payload["status"] not in transitions.get(order["status"], []):
                    raise ToolFailure(
                        "invalid_state", "Requested status transition is not permitted", 409
                    )
                if payload["status"] == "cancelled" and order.get("fulfillment") != "unshipped":
                    raise ToolFailure(
                        "invalid_state", "An order with shipments cannot be cancelled", 409
                    )
            if "line_id" in payload:
                line = next((r for r in order["lines"] if r["line_id"] == payload["line_id"]), None)
                if not line or line["line_status"] in {"fulfilled", "cancelled"}:
                    raise ToolFailure("invalid_line", "Select an editable line on this order", 409)
                if name == "updateOrderLine" and not {"quantity_lb", "unit_price"}.intersection(
                    payload
                ):
                    raise ToolFailure("invalid_arguments", "Supply quantity_lb or unit_price")
            if name in {"commitShipOrder", "shipOrder"}:
                body = {k: v for k, v in payload.items() if k != "order_id"} | {"mode": "preview"}
                state["preview"] = await self._request(
                    "POST",
                    f"/sales/orders/{order['order_id']}/ship/preview",
                    json=body,
                    headers=headers,
                )
                rows = state["preview"]["lines"]
                if not rows or (
                    "lines" in payload
                    and {r["line_id"] for r in rows} != {r["line_id"] for r in payload["lines"]}
                ):
                    raise ToolFailure(
                        "invalid_line", "Shipping preview did not include every requested line", 409
                    )
                for row in rows:
                    if row["requested_ship_lb"] > row["remaining_lb"]:
                        raise ToolFailure(
                            "invalid_quantity", "Shipment exceeds the remaining order quantity", 409
                        )
                # Include lot identities/balances, not just an aggregate inventory count.
                state["inventory"] = []
                for row in rows:
                    if not row.get("is_service"):
                        inventory = await self._request(
                            "GET",
                            "/inventory/lookup",
                            params={"q": row["product"], "limit": 50},
                            headers=headers,
                        )
                        exact = [
                            r
                            for r in inventory["results"]
                            if r["product"].casefold() == row["product"].casefold()
                        ]
                        if len(exact) != 1:
                            raise ToolFailure("ambiguous_product", "Cannot bind the shipping lots")
                        products = await self._request(
                            "GET",
                            "/products/search",
                            params={"q": row["product"], "limit": 100},
                            headers=headers,
                        )
                        selected = [
                            p
                            for p in products["products"]
                            if p["name"].casefold() == row["product"].casefold()
                        ]
                        if len(selected) != 1:
                            raise ToolFailure(
                                "ambiguous_product", "Cannot identify shipping product"
                            )
                        lots = []
                        for lot in exact[0]["lots"]:
                            lot_path = route(
                                "/lots/by-code/{lot_code}", {"lot_code": lot["lot"]}, ["lot_code"]
                            )
                            resolved = await self._request(
                                "GET",
                                lot_path,
                                params={"product_id": selected[0]["id"]},
                                headers=headers,
                            )
                            # The numeric lookup includes received_at, which controls FIFO.
                            lots.append(
                                await self._request(
                                    "GET",
                                    f"/lots/{resolved['id']}",
                                    headers=headers,
                                )
                            )
                        state["inventory"].append({"product": selected[0], "lots": lots})
                proposal["shipping_preview"] = state["preview"]
        elif name == "createOrder":
            state["customer"] = await self._customer(payload["customer_id"], headers)
            proposal["customer"] = state["customer"]["name"]
            proposal["customer_po"] = payload.get("customer_po") or None
            if isinstance(proposal["customer_po"], str) and not proposal["customer_po"].strip():
                proposal["customer_po"] = None
            proposal["flags"] = [] if proposal["customer_po"] else ["No PO"]
            external = payload["external_order_ref"].strip()
            if not external:
                raise ToolFailure("invalid_arguments", "An external order reference is required")
            orders = await self._request(
                "GET", "/sales/orders", params={"limit": 200}, headers=headers
            )
            state["duplicates"] = orders
            for row in orders["orders"]:
                detail = await self._request(
                    "GET", f"/sales/orders/{row['order_id']}", headers=headers
                )
                fields = [
                    row.get("external_order_ref", ""),
                    row["order_number"],
                    detail.get("notes") or "",
                    detail.get("notes_es") or "",
                ]
                fields += [r.get("notes") or "" for r in detail.get("lines", [])]
                if any(
                    re.search(r"(?<!\w)" + re.escape(external) + r"(?!\w)", value, re.I)
                    for value in fields
                ):
                    raise ToolFailure(
                        "duplicate_order",
                        f"External reference {external} already appears on {row['order_number']}",
                        409,
                        {"order_number": row["order_number"], "order_id": row["order_id"]},
                    )
                if (
                    row["customer"] == state["customer"]["name"]
                    and proposal["customer_po"]
                    and normalize_po(row.get("customer_po"))
                    == normalize_po(proposal["customer_po"])
                ):
                    raise ToolFailure(
                        "duplicate_po", f"Customer PO already exists on {row['order_number']}", 409
                    )
            if len(orders["orders"]) >= 200:
                raise ToolFailure(
                    "duplicate_check_incomplete",
                    "The backend cannot check external references beyond 200 orders; "
                    "creation remains blocked",
                    409,
                )
            state["products"] = []
            proposal["lines"] = []
            for line in payload["lines"]:
                product = await self._product(line["product_id"], headers)
                state["products"].append(product)
                if line["product_id"] == 176:
                    if product.get("name") != "Pallet Charge" or not product.get("is_service"):
                        raise ToolFailure(
                            "invalid_pallet_product",
                            "Product 176 must be the active Pallet Charge service",
                        )
                    if line["unit"] != "each" or "unit_price" not in line:
                        raise ToolFailure(
                            "invalid_pallet_line",
                            "Pallet Charge requires an each count and unit price",
                        )
                elif "pallet" in product.get("name", "").lower():
                    raise ToolFailure(
                        "invalid_pallet_product", "Use product 176 for pallet charges"
                    )
                if line["unit"] == "each" and not product.get("is_service"):
                    raise ToolFailure(
                        "invalid_units", "Physical products require explicit lb or cases"
                    )
                if product.get("is_service") and line["unit"] != "each":
                    raise ToolFailure("invalid_units", "Service charges require each units")
                if line["unit"] == "cases" and not product.get("case_size_lb"):
                    raise ToolFailure("invalid_units", "The selected product has no case weight")
                proposed = line | {
                    "product": product["name"],
                    "is_service": bool(product.get("is_service")),
                    "amount": round(line["quantity"] * line["unit_price"], 2)
                    if "unit_price" in line
                    else None,
                }
                proposal["lines"].append(proposed)
        elif name == "createExpectedReceipt":
            if payload.get("product_id"):
                state["product"] = await self._product(payload["product_id"], headers)
            else:
                found = await self._request(
                    "GET",
                    "/products/search",
                    params={"q": payload.get("product_name", ""), "limit": 100},
                    headers=headers,
                )
                matches = [
                    p
                    for p in found["products"]
                    if p["name"].casefold() == payload.get("product_name", "").casefold()
                ]
                if len(matches) != 1:
                    raise ToolFailure(
                        "ambiguous_product", "Select an exact product_id before previewing"
                    )
                state["product"] = await self._product(matches[0]["id"], headers)
            suppliers = await self._request("GET", "/suppliers", headers=headers)
            matches = [
                r
                for r in suppliers["suppliers"]
                if r["name"].strip().casefold() == payload["supplier_name"].strip().casefold()
                and r.get("active", True)
            ]
            if len(matches) != 1:
                raise ToolFailure("invalid_supplier", "Select an existing active supplier")
            state["supplier"] = matches[0]
            proposal["product_id"] = state["product"]["id"]
            proposal["product_name"] = state["product"]["name"]
        elif name == "renameLot":
            state["lot"] = await self._request("GET", f"/lots/{payload['lot_id']}", headers=headers)
            proposal["previous_lot"] = state["lot"]
        elif name == "updateSupplierLot":
            path = route("/lots/by-code/{lot_code}", payload, ["lot_code"])
            state["lot"] = await self._request(
                "GET",
                path,
                params={k: payload[k] for k in ["product_id"] if k in payload},
                headers=headers,
            )
            proposal["previous_lot"] = state["lot"]
        # Blocked routes have no mutating preview calls. The proposal is explicitly unsaved.
        return state, proposal, describe(name, proposal)

    async def _customer(self, customer_id, headers):
        customers = await self._request("GET", "/customers", headers=headers)
        customer = next((r for r in customers["customers"] if r["id"] == customer_id), None)
        if not customer or not customer.get("active", True):
            raise ToolFailure(
                "invalid_customer",
                "Select an existing active customer; customer creation needs separate approval",
            )
        return customer


def verify_order_readback(name, payload, before, after, result):
    """Never call a lossy legacy response verified just because the GET succeeded."""
    comparisons = [
        after["order_id"] == before["order_id"],
        after["order_number"] == before["order_number"],
        after["customer_po"] == before["customer_po"],
    ]
    if name == "updateOrderHeader":
        for field in ("requested_ship_date", "notes", "notes_es"):
            if field in payload:
                comparisons.append((after.get(field) or None) == (payload[field] or None))
        if "customer_id" in payload:
            comparisons.append(result.get("customer_id") == payload["customer_id"])
            comparisons.append(after["customer"] == result.get("customer_name"))
    elif name == "updateOrderStatus":
        comparisons.append(after["status"] == payload["status"])
    elif name == "updateOrderLine":
        line = next((r for r in after["lines"] if r["line_id"] == payload["line_id"]), {})
        for request_field, response_field in [
            ("quantity_lb", "quantity_lb"),
            ("unit_price", "case_price"),
        ]:
            if request_field in payload:
                # Legacy handlers serialize a zero price as null: retain the approved
                # zero in changed_fields, but do not invent a verified readback value.
                comparisons.append(line.get(response_field) == payload[request_field])
    elif name == "commitShipOrder":
        comparisons.append(after["status"] == result.get("order_status"))
        comparisons.append(after["order_number"] == result.get("order_number"))
        transaction_ids = {r["transaction_id"] for r in after["shipments"]}
        comparisons.extend(
            r["transaction_id"] in transaction_ids
            for r in result.get("lines_shipped", [])
            if "transaction_id" in r
        )
    if not all(comparisons):
        raise ToolFailure("readback_mismatch", "Saved fields could not be fully verified")


def validate(spec, arguments):
    try:
        # Reject NaN/Infinity as well as unknown identity/control fields.
        json.dumps(arguments, allow_nan=False)
        Draft202012Validator(spec["input_schema"], format_checker=FormatChecker()).validate(
            arguments
        )
    except (ValueError, TypeError, ValidationError):
        raise ToolFailure("invalid_arguments", "Arguments do not match this tool's schema")


def route(template, arguments, parameters):
    for parameter in parameters:
        value = str(arguments[parameter])
        if not value or value in {".", ".."} or any(c in value for c in "/\\%?#\x00\r\n"):
            raise ToolFailure("invalid_arguments", f"Unsafe path parameter: {parameter}")
        template = template.replace("{" + parameter + "}", quote(value, safe=""))
    return template


def shipping_arguments(payload):
    if payload.get("ship_all") and "lines" in payload:
        raise ToolFailure(
            "invalid_arguments", "ship_all=true cannot be combined with explicit lines"
        )
    if not payload.get("ship_all") and not payload.get("lines"):
        raise ToolFailure("invalid_arguments", "Supply explicit lines or ship_all=true")
    ids = [row["line_id"] for row in payload.get("lines", [])]
    if len(ids) != len(set(ids)):
        raise ToolFailure("invalid_arguments", "A shipping line may appear only once")


def normalize_po(value):
    return " ".join((value or "").split()).casefold()


def describe(name, p):
    """Human-readable sentences include exact values; structured proposal retains all fields."""
    labels = {
        "updateOrderHeader": "Update the header of",
        "updateOrderStatus": "Change the status of",
        "updateOrderLine": "Edit a line on",
        "addOrderLines": "Add lines to",
        "cancelOrderLine": "Cancel a line on",
        "shipOrder": "Ship",
        "commitShipOrder": "Ship",
    }
    if name == "createOrder":
        lines = "; ".join(
            f"{r['quantity']:g} {r['unit']} of {r['product']} (product {r['product_id']})"
            + (
                f" at {r['unit_price']:g} per {r['unit']}, amount {r['amount']:g}"
                if r.get("unit_price") is not None
                else ""
            )
            for r in p["lines"]
        )
        sentence = (
            f"Create an order for {p['customer']}, external reference {p['external_order_ref']}, "
            + (f"PO {p['customer_po']}" if p["customer_po"] else "No PO")
            + f". Order lines: {lines}."
        )
    elif name in labels:
        po = "PO " + p["customer_po"] if p["customer_po"] else "No PO"
        sentence = f"{labels[name]} {p['order_number']} ({po})."
    elif name == "createExpectedReceipt":
        sentence = (
            f"Record an expected delivery of {p['expected_qty']:g} lb "
            f"of {p['product_name']} from {p['supplier_name']}."
        )
    elif name == "renameLot":
        sentence = f"Rename lot {p['lot_id']} to {p['new_lot_code']}."
    elif name == "voidTransaction":
        sentence = f"Void transaction {p['transaction_id']} with reason {p.get('reason', '')}."
    else:
        words = re.sub(r"([A-Z])", r" \1", name).lower()
        sentence = f"Propose {words} with the following values."
    details = "; ".join(
        f"{k.replace('_', ' ')}: {json.dumps(v, ensure_ascii=False)}" for k, v in p.items()
    )
    return sentence + " Exact proposal: " + details + ". No changes have been saved."


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def timestamp(seconds):
    return datetime.fromtimestamp(seconds, UTC).isoformat()


class ConfirmationStore:
    """Durable one-time claims across processes/restarts, with no raw tokens or keys.

    This is an approval journal, NOT transactional ledger idempotency. An inflight
    record left by a crash is uncertain and blocks reissuing the same proposal.
    Receipts must be retained for operator reconciliation; no automatic retries.
    """

    def __init__(self, path, *, clock=time.time, ttl=120):
        self.path = Path(path)
        self.clock = clock
        self.ttl = ttl

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        db.execute("""CREATE TABLE IF NOT EXISTS confirmations (
            approval_id TEXT PRIMARY KEY, token_hash TEXT UNIQUE NOT NULL,
            principal TEXT NOT NULL, binding TEXT NOT NULL, revision TEXT NOT NULL,
            expires REAL NOT NULL, state TEXT NOT NULL, receipt TEXT)""")
        try:
            with db:
                yield db
        finally:
            db.close()

    def issue(self, principal, binding, revision):
        token = secrets.token_urlsafe(32)
        approval_id = secrets.token_hex(16)
        expires = self.clock() + self.ttl
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute(
                "SELECT 1 FROM confirmations WHERE principal=? AND binding=? "
                "AND state IN ('inflight','uncertain')",
                (principal, binding),
            ).fetchone():
                raise ToolFailure(
                    "reconciliation_required",
                    "An earlier identical write has an uncertain outcome. "
                    "Reconcile it before preparing another",
                    409,
                )
            db.execute(
                "UPDATE confirmations SET state='superseded' "
                "WHERE principal=? AND binding=? AND state='pending'",
                (principal, binding),
            )
            db.execute(
                "INSERT INTO confirmations VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL)",
                (approval_id, digest(token), principal, binding, revision, expires),
            )
        return {"token": token, "approval_id": approval_id, "expires_at": timestamp(expires)}

    def claim(self, token, principal, binding):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM confirmations WHERE token_hash=?", (digest(token),)
            ).fetchone()
            if row is None or row["principal"] != principal or row["binding"] != binding:
                raise ToolFailure(
                    "invalid_confirmation",
                    "Token does not match this user, operation and exact payload",
                    409,
                )
            if row["state"] != "pending":
                raise ToolFailure(
                    "confirmation_used", "This token has already been consumed; do not retry", 409
                )
            if self.clock() >= row["expires"]:
                raise ToolFailure("confirmation_expired", "Approval expired; preview again", 409)
            db.execute(
                "UPDATE confirmations SET state='inflight' WHERE approval_id=?",
                (row["approval_id"],),
            )
        return dict(row)

    def finish(self, approval_id, state, receipt=None):
        with self.connect() as db:
            db.execute(
                "UPDATE confirmations SET state=?, receipt=? WHERE approval_id=?",
                (state, json.dumps(receipt) if receipt is not None else None, approval_id),
            )
