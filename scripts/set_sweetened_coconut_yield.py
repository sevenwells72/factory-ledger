"""Owner-run forward-only catalog update. This script is NOT run by deployment."""
import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def main():
    base = "https://fastapi-production-b73a.up.railway.app"
    key = os.environ["API_KEY"]  # Master/admin key; never printed.

    def call(path, body=None):
        request = Request(
            base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"X-API-Key": key, "Content-Type": "application/json"},
            method="PUT" if body is not None else "GET",
        )
        with urlopen(request, timeout=30) as response:
            return json.load(response)

    # These are SKU/odoo_code values, NOT database product IDs. Resolve and
    # validate all three before issuing any update. Re-running is idempotent.
    products = []
    for sku in ("90003", "90004", "90005"):
        matches = call("/products/search?" + urlencode({"q": sku, "limit": 100}))
        exact = [p for p in matches["products"] if str(p["odoo_code"]) == sku]
        if len(exact) != 1:
            raise RuntimeError(f"Expected one exact product for SKU {sku}")
        product = exact[0]
        if product["type"] != "batch" or float(product["default_batch_lb"] or 0) != 360:
            raise RuntimeError(f"SKU {sku} must be a 360-lb batch")
        products.append((sku, product["id"]))

    for sku, product_id in products:
        result = call(f"/admin/products/{product_id}", {"yield_multiplier": 1.0})
        if (result.get("updated") is not True or result.get("product_id") != product_id
                or result.get("changes", {}).get("yield_multiplier") != 1.0):
            raise RuntimeError(f"Unexpected update response for SKU {sku}: {result}")
        print(f"SKU {sku} (product ID {product_id}): yield_multiplier = 1.0")


if __name__ == "__main__":
    main()
