"""Vision extraction for AI-assisted document intake.

One public function — extract_purchase_document(file_bytes, mime_type, kind) —
turns a document (PNG, JPEG, or PDF) into the strict JSON shape the matching
step consumes. kind='purchase' (default) reads a vendor PO / order
confirmation for the expected-receipt flow; kind='sales' reads a CUSTOMER PO
for the sales-order flow (docs/designs/sales-order-intake.md). The API vendor
is swappable by replacing this module: nothing else in the codebase imports an
LLM SDK.

HARD RULE (docs/designs/expected-receipt-intake.md): the model never writes to
the DB. This module has no DB imports — extraction is a pure
bytes → validated-dict function, and matching/approval live elsewhere.
"""

import base64
import math
import os
import re
from datetime import datetime
from typing import List, Optional

import anthropic
from pydantic import BaseModel, ValidationError, field_validator

DEFAULT_EXTRACTION_MODEL = "claude-sonnet-5"

ALLOWED_MIME_TYPES = ("image/png", "image/jpeg", "application/pdf")

MAX_FILE_BYTES = 15 * 1024 * 1024  # 15 MB cap (owner-approved)

MAX_PDF_PAGES = 20  # owner-approved cap; enforced by the upload endpoint


class ExtractionError(Exception):
    """Extraction failed: API error, refusal, or output that doesn't fit the
    schema. The caller keeps the stored file (status 'extraction_failed') so a
    retry needs no re-upload."""


DOCUMENT_KINDS = ("purchase", "sales")


def _validate_iso_date_value(v: Optional[str]) -> Optional[str]:
    """A date is exactly YYYY-MM-DD and a real calendar date, or null — never
    a free-text string the model happened to emit. The format check must run
    BEFORE strptime, which accepts unpadded fields ('2026-2-3') that the
    contract forbids. (Audit fix 11, shared by both document kinds.)"""
    if v is None:
        return v
    if not isinstance(v, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
        raise ValueError("date must be YYYY-MM-DD or null")
    try:
        datetime.strptime(v, "%Y-%m-%d")
    except (ValueError, TypeError):
        raise ValueError("date must be YYYY-MM-DD or null")
    return v


class ExtractedLine(BaseModel):
    vendor_description: str
    quantity: float
    unit: Optional[str] = None

    # Audit fix 11: the "strict" contract really is strict — NaN/±inf and
    # non-positive quantities are schema violations, not passthroughs.
    @field_validator("quantity")
    @classmethod
    def _quantity_finite_positive(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError("quantity must be a finite number > 0")
        return v


class ExtractionResult(BaseModel):
    supplier_name: str
    reference_number: Optional[str] = None
    document_date: Optional[str] = None           # YYYY-MM-DD
    expected_delivery_date: Optional[str] = None  # YYYY-MM-DD
    lines: List[ExtractedLine]

    @field_validator("document_date", "expected_delivery_date")
    @classmethod
    def _valid_iso_date(cls, v: Optional[str]) -> Optional[str]:
        return _validate_iso_date_value(v)


class SalesExtractedLine(BaseModel):
    customer_item_code: Optional[str] = None
    description: str
    quantity: float
    unit: Optional[str] = None
    unit_price: Optional[float] = None

    @field_validator("quantity")
    @classmethod
    def _quantity_finite_positive(cls, v: float) -> float:
        if not math.isfinite(v) or v <= 0:
            raise ValueError("quantity must be a finite number > 0")
        return v

    # A printed 0.00 (no-charge line) is legitimate; NaN/±inf/negative are
    # schema violations, same strictness as quantity.
    @field_validator("unit_price")
    @classmethod
    def _price_finite_nonnegative(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if not math.isfinite(v) or v < 0:
            raise ValueError("unit_price must be a finite number >= 0 or null")
        return v


class SalesExtractionResult(BaseModel):
    customer_name: str
    po_number: Optional[str] = None
    document_date: Optional[str] = None          # YYYY-MM-DD
    requested_ship_date: Optional[str] = None    # YYYY-MM-DD
    lines: List[SalesExtractedLine]

    @field_validator("document_date", "requested_ship_date")
    @classmethod
    def _valid_iso_date(cls, v: Optional[str]) -> Optional[str]:
        return _validate_iso_date_value(v)


# Forced tool-use schema. reference_number / document_date /
# expected_delivery_date / unit are all nullable — the prompt forbids guessing.
EXTRACTION_TOOL = {
    "name": "record_purchase_document",
    "description": "Record the structured contents of a vendor purchase order or order confirmation.",
    "input_schema": {
        "type": "object",
        "properties": {
            "supplier_name": {
                "type": "string",
                "description": "The vendor/supplier company name as printed on the document.",
            },
            "reference_number": {
                "type": ["string", "null"],
                "description": "PO / order / confirmation number as printed. null if none is printed.",
            },
            "document_date": {
                "type": ["string", "null"],
                "description": "Date the document was issued, YYYY-MM-DD. null if not printed.",
            },
            "expected_delivery_date": {
                "type": ["string", "null"],
                "description": "Promised/expected delivery or ship date, YYYY-MM-DD. null if not printed.",
            },
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "vendor_description": {
                            "type": "string",
                            "description": "The line's product description EXACTLY as printed, verbatim.",
                        },
                        "quantity": {
                            "type": "number",
                            "description": "Quantity ordered on the line, as printed.",
                        },
                        "unit": {
                            "type": ["string", "null"],
                            "description": "Unit as printed (e.g. LB, CASE, BAG, EA). null if no unit is printed.",
                        },
                    },
                    "required": ["vendor_description", "quantity", "unit"],
                },
            },
        },
        "required": [
            "supplier_name",
            "reference_number",
            "document_date",
            "expected_delivery_date",
            "lines",
        ],
    },
}

EXTRACTION_PROMPT = """Extract the purchase order / order confirmation in this document using the record_purchase_document tool.

Rules:
- Extract ONLY what is printed on the document. NEVER guess or infer a missing value — use null for any field that is not printed.
- vendor_description must be VERBATIM: the supplier's exact wording, including pack sizes, codes, and abbreviations. Do not normalize, translate, or expand it.
- Dates in YYYY-MM-DD. If only a month/day is printed with no year, use null rather than guessing the year.
- quantity is the quantity ORDERED. If the document distinguishes ordered vs shipped/backordered, use ordered.
- Include every product line item. Exclude freight, tax, deposits, and subtotal/total rows.
"""


# Forced tool-use schema for kind='sales' — a CUSTOMER purchase order (we are
# the vendor). Nullable fields mirror the purchase schema: the prompt forbids
# guessing.
SALES_EXTRACTION_TOOL = {
    "name": "record_customer_purchase_order",
    "description": "Record the structured contents of a customer's purchase order sent to the vendor.",
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_name": {
                "type": "string",
                "description": "The BUYER company that issued this purchase order (the customer placing the order), as printed.",
            },
            "po_number": {
                "type": ["string", "null"],
                "description": "The customer's PO / order number as printed. null if none is printed.",
            },
            "document_date": {
                "type": ["string", "null"],
                "description": "Date the PO was issued, YYYY-MM-DD. null if not printed.",
            },
            "requested_ship_date": {
                "type": ["string", "null"],
                "description": "Requested ship/delivery date, YYYY-MM-DD. null if not printed.",
            },
            "lines": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "customer_item_code": {
                            "type": ["string", "null"],
                            "description": "The customer's item/SKU code for the line, exactly as printed. null if none is printed.",
                        },
                        "description": {
                            "type": "string",
                            "description": "The line's product description EXACTLY as printed, verbatim.",
                        },
                        "quantity": {
                            "type": "number",
                            "description": "Quantity ordered on the line, as printed.",
                        },
                        "unit": {
                            "type": ["string", "null"],
                            "description": "Unit as printed (e.g. CASE, CS, LB, EA). null if no unit is printed.",
                        },
                        "unit_price": {
                            "type": ["number", "null"],
                            "description": "Price per unit as printed on the line. null if no price is printed.",
                        },
                    },
                    "required": ["customer_item_code", "description", "quantity", "unit", "unit_price"],
                },
            },
        },
        "required": [
            "customer_name",
            "po_number",
            "document_date",
            "requested_ship_date",
            "lines",
        ],
    },
}

SALES_EXTRACTION_PROMPT = """Extract the customer purchase order in this document using the record_customer_purchase_order tool.

This is a purchase order a CUSTOMER sent to us, the vendor (Seven Wells Granola). customer_name is the BUYER that issued the PO — the company placing the order. It is NEVER the vendor/supplier/remit-to party the order is addressed to.

Rules:
- Extract ONLY what is printed on the document. NEVER guess or infer a missing value — use null for any field that is not printed.
- description must be VERBATIM: the customer's exact wording, including pack sizes, codes, and abbreviations. Do not normalize, translate, or expand it.
- customer_item_code is the customer's own item/SKU number for the line, if the PO prints one.
- Dates in YYYY-MM-DD. If only a month/day is printed with no year, use null rather than guessing the year.
- quantity is the quantity ORDERED. If the document distinguishes ordered vs shipped/backordered, use ordered.
- unit_price is the per-unit price printed on the line, if any. Do not derive it from an extended/total amount.
- Include every product line item. Exclude freight, tax, deposits, and subtotal/total rows.
"""


_KIND_CONFIG = {
    "purchase": (EXTRACTION_TOOL, EXTRACTION_PROMPT, ExtractionResult),
    "sales": (SALES_EXTRACTION_TOOL, SALES_EXTRACTION_PROMPT, SalesExtractionResult),
}


def _client() -> anthropic.Anthropic:
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise ExtractionError("ANTHROPIC_API_KEY is not configured")
    return anthropic.Anthropic(api_key=api_key)


def extract_purchase_document(file_bytes: bytes, mime_type: str,
                              kind: str = "purchase") -> dict:
    """Send the document to the vision model and return the validated
    extraction as a plain dict (ExtractionResult shape for kind='purchase',
    SalesExtractionResult shape for kind='sales'). Raises ExtractionError on
    any API/validation failure — never returns a guess."""
    if kind not in _KIND_CONFIG:
        raise ExtractionError(f"Unknown document kind: {kind}")
    tool, prompt, result_model = _KIND_CONFIG[kind]
    if mime_type not in ALLOWED_MIME_TYPES:
        raise ExtractionError(f"Unsupported mime type: {mime_type}")
    if len(file_bytes) > MAX_FILE_BYTES:
        raise ExtractionError(f"File exceeds {MAX_FILE_BYTES // (1024 * 1024)} MB limit")

    data_b64 = base64.standard_b64encode(file_bytes).decode("ascii")
    if mime_type == "application/pdf":
        file_block = {
            "type": "document",
            "source": {"type": "base64", "media_type": mime_type, "data": data_b64},
        }
    else:
        file_block = {
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": data_b64},
        }

    model = (os.getenv("EXTRACTION_MODEL") or "").strip() or DEFAULT_EXTRACTION_MODEL
    try:
        message = _client().messages.create(
            model=model,
            max_tokens=4096,
            tools=[tool],
            tool_choice={"type": "tool", "name": tool["name"]},
            messages=[
                {
                    "role": "user",
                    "content": [file_block, {"type": "text", "text": prompt}],
                }
            ],
        )
    except anthropic.APIError as exc:
        raise ExtractionError(f"Vision API call failed: {exc}") from exc

    tool_input = next(
        (block.input for block in message.content if block.type == "tool_use"), None
    )
    if tool_input is None:
        raise ExtractionError("Model returned no structured extraction")

    try:
        result = result_model(**tool_input)
    except (ValidationError, TypeError) as exc:
        raise ExtractionError(f"Extraction did not match schema: {exc}") from exc

    if not result.lines:
        raise ExtractionError("No product lines found on the document")

    return {"extraction": result.model_dump(), "extraction_model": model}
