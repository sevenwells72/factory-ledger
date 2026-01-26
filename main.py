# ============================================================
# ADJUST COMMAND - Add to main.py
# ============================================================

# --- Add these Pydantic models after the Ship models ---

class AdjustRequest(BaseModel):
    product_name: str
    quantity_lb: float  # Positive to add, negative to remove
    lot_code: str       # Required - no FIFO guessing
    reason: str         # Required for audit trail


class AdjustPreviewResponse(BaseModel):
    product_id: int
    product_name: str
    odoo_code: str
    lot_code: str
    lot_id: int
    adjustment_lb: float
    current_balance_lb: float
    new_balance_lb: float
    reason: str
    preview_message: str


class AdjustCommitResponse(BaseModel):
    success: bool
    transaction_id: int
    lot_code: str
    adjustment_lb: float
    new_balance_lb: float
    reason: str
    message: str


# --- Adjust Endpoints ---

@app.post("/adjust/preview", response_model=AdjustPreviewResponse)
def adjust_preview(req: AdjustRequest, _: bool = Depends(verify_api_key)):
    """Preview an inventory adjustment. Requires lot code and reason."""
    
    if not req.reason or len(req.reason.strip()) < 3:
        raise HTTPException(status_code=400, detail="Reason is required (minimum 3 characters)")
    
    if req.quantity_lb == 0:
        raise HTTPException(status_code=400, detail="Adjustment quantity cannot be zero")
    
    try:
        with psycopg2.connect(DATABASE_URL, connect_timeout=5) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Find product
                cur.execute(
                    """SELECT id, name, odoo_code FROM products
                    WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) = LOWER(%s)
                    LIMIT 5""",
                    (f"%{req.product_name}%", req.product_name)
                )
                products = cur.fetchall()

                if not products:
                    raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")

                if len(products) > 1:
                    product_list = [f"• {p['name']} ({p['odoo_code']})" for p in products]
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Multiple products match '{req.product_name}'. Please be more specific:\n" + "\n".join(product_list)
                    )

                product = products[0]

                # Find the specific lot
                cur.execute(
                    """SELECT l.id as lot_id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as current_balance
                    FROM lots l
                    LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                    WHERE l.product_id = %s AND l.lot_code = %s
                    GROUP BY l.id, l.lot_code""",
                    (product["id"], req.lot_code)
                )
                lot = cur.fetchone()

                if not lot:
                    # Check if product has any lots to give helpful error
                    cur.execute(
                        """SELECT l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as balance
                        FROM lots l
                        LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
                        WHERE l.product_id = %s
                        GROUP BY l.id, l.lot_code
                        HAVING COALESCE(SUM(tl.quantity_lb), 0) != 0
                        ORDER BY l.lot_code""",
                        (product["id"],)
                    )
                    existing_lots = cur.fetchall()
                    
                    if existing_lots:
                        lot_list = [f"• {l['lot_code']} ({l['balance']:,.0f} lb)" for l in existing_lots]
                        raise HTTPException(
                            status_code=400,
                            detail=f"Lot '{req.lot_code}' not found for {product['name']}. Existing lots:\n" + "\n".join(lot_list)
                        )
                    else:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Lot '{req.lot_code}' not found for {product['name']}. No lots exist for this product."
                        )

                current_balance = float(lot["current_balance"])
                new_balance = current_balance + req.quantity_lb

                # Block if adjustment would make balance negative (Rule 2A)
                if new_balance < 0:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Cannot adjust. Lot {req.lot_code} only has {current_balance:,.0f} lb. "
                               f"Adjustment of {req.quantity_lb:,.0f} lb would result in {new_balance:,.0f} lb (negative not allowed)."
                    )

        # Format adjustment display
        adj_display = f"+{req.quantity_lb:,.0f}" if req.quantity_lb > 0 else f"{req.quantity_lb:,.0f}"
        adj_type = "ADD" if req.quantity_lb > 0 else "REMOVE"

        preview_message = f"""📋 ADJUST PREVIEW
────────────────────────────────
Product:      {product['name']} ({product['odoo_code']})
Lot:          {req.lot_code}
Adjustment:   {adj_display} lb ({adj_type})
────────────────────────────────
Current:      {current_balance:,.0f} lb
After:        {new_balance:,.0f} lb
────────────────────────────────
Reason:       {req.reason}
────────────────────────────────
✓ Say "confirm" to proceed"""

        return AdjustPreviewResponse(
            product_id=product["id"],
            product_name=product["name"],
            odoo_code=product["odoo_code"],
            lot_code=lot["lot_code"],
            lot_id=lot["lot_id"],
            adjustment_lb=req.quantity_lb,
            current_balance_lb=current_balance,
            new_balance_lb=new_balance,
            reason=req.reason,
            preview_message=preview_message
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/adjust/commit", response_model=AdjustCommitResponse)
def adjust_commit(req: AdjustRequest, _: bool = Depends(verify_api_key)):
    """Commit an inventory adjustment."""
    
    if not req.reason or len(req.reason.strip()) < 3:
        raise HTTPException(status_code=400, detail="Reason is required (minimum 3 characters)")
    
    if req.quantity_lb == 0:
        raise HTTPException(status_code=400, detail="Adjustment quantity cannot be zero")
    
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL, connect_timeout=5)
        conn.autocommit = False
        cur = conn.cursor(cursor_factory=RealDictCursor)

        # Find product
        cur.execute(
            """SELECT id, name, odoo_code FROM products
            WHERE LOWER(name) LIKE LOWER(%s) OR LOWER(odoo_code) = LOWER(%s)
            LIMIT 1""",
            (f"%{req.product_name}%", req.product_name)
        )
        product = cur.fetchone()
        if not product:
            raise HTTPException(status_code=404, detail=f"Product not found: {req.product_name}")

        # Find the specific lot
        cur.execute(
            """SELECT l.id as lot_id, l.lot_code, COALESCE(SUM(tl.quantity_lb), 0) as current_balance
            FROM lots l
            LEFT JOIN transaction_lines tl ON tl.lot_id = l.id
            WHERE l.product_id = %s AND l.lot_code = %s
            GROUP BY l.id, l.lot_code""",
            (product["id"], req.lot_code)
        )
        lot = cur.fetchone()

        if not lot:
            raise HTTPException(status_code=400, detail=f"Lot '{req.lot_code}' not found for {product['name']}")

        current_balance = float(lot["current_balance"])
        new_balance = current_balance + req.quantity_lb

        # Block if adjustment would make balance negative (Rule 2A)
        if new_balance < 0:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot adjust. Lot {req.lot_code} only has {current_balance:,.0f} lb. "
                       f"Adjustment of {req.quantity_lb:,.0f} lb would result in negative balance."
            )

        # Create transaction
        cur.execute(
            """INSERT INTO transactions (type, adjust_reason)
            VALUES ('adjust', %s) RETURNING id""",
            (req.reason,)
        )
        transaction_id = cur.fetchone()["id"]

        # Create transaction line
        cur.execute(
            "INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, %s)",
            (transaction_id, product["id"], lot["lot_id"], req.quantity_lb)
        )

        conn.commit()
        cur.close()

        adj_display = f"+{req.quantity_lb:,.0f}" if req.quantity_lb > 0 else f"{req.quantity_lb:,.0f}"
        adj_type = "Added" if req.quantity_lb > 0 else "Removed"

        return AdjustCommitResponse(
            success=True,
            transaction_id=transaction_id,
            lot_code=lot["lot_code"],
            adjustment_lb=req.quantity_lb,
            new_balance_lb=new_balance,
            reason=req.reason,
            message=f"✅ {adj_type} {abs(req.quantity_lb):,.0f} lb {product['name']} (lot {lot['lot_code']}). New balance: {new_balance:,.0f} lb"
        )

    except HTTPException:
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        if conn:
            conn.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if conn:
            conn.close()
