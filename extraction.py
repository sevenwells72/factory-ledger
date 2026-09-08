"""Vision extraction for AI-assisted expected-receipt intake.

One public function — extract_purchase_document(file_bytes, mime_type) — turns
a vendor PO / order confirmation (PNG, JPEG, or PDF) into the strict JSON shape
the matching step consumes. The API vendor is swappable by replacing this
module: nothing else in the codebase imports an LLM SDK.

HARD RULE (docs/designs/expected-receipt-intake.md): the model never writes to
the DB. This module has no DB imports — extraction is a pure
bytes → validated-dict function, and matching/approval live elsewhere.
"""

import base64
import os
from typing import List, Optional

import anthropic
from pydantic import BaseModel, ValidationError

DEFAULT_EXTRACTION_MODEL = "claude-sonnet-5"

ALLOWED_MIME_TYPES = ("image/png", "image/jpeg", "application/pdf")

MAX_FILE_BYTES = 15 * 1024 * 1024  # 15 MB cap (owner-approved)

MAX_PDF_PAGES = 20  # owner-approved cap; enforced by the upload endpoint


class ExtractionError(Exception):
    """Extraction failed: API error, refusal, or output that doesn't fit the
    schema. The caller keeps the stored file (status 'extraction_failed') so a
    retry needs no re-upload."""


class ExtractedLine(BaseModel):
    vendor_description: str
    quantity: float
    unit: Optional[str] = None


class ExtractionResult(BaseModel):
    supplier_name: str
    reference_number: Optional[str] = None
    document_date: Optional[str] = None           # YYYY-MM-DD
    expected_delivery_date: Optional[str] = None  # YYYY-MM-DD
    lines: List[ExtractedLine]


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


def _client() -> anthropic.Anthropic:
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise ExtractionError("ANTHROPIC_API_KEY is not configured")
    return anthropic.Anthropic(api_key=api_key)


def extract_purchase_document(file_bytes: bytes, mime_type: str) -> dict:
    """Send the document to the vision model and return the validated
    extraction as a plain dict (ExtractionResult shape). Raises
    ExtractionError on any API/validation failure — never returns a guess."""
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
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "record_purchase_document"},
            messages=[
                {
                    "role": "user",
                    "content": [file_block, {"type": "text", "text": EXTRACTION_PROMPT}],
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
        result = ExtractionResult(**tool_input)
    except (ValidationError, TypeError) as exc:
        raise ExtractionError(f"Extraction did not match schema: {exc}") from exc

    if not result.lines:
        raise ExtractionError("No product lines found on the document")

    return {"extraction": result.model_dump(), "extraction_model": model}
