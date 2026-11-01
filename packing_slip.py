"""Packing slip PDF: GET /sales/orders/{order_id}/packing-slip.

Moved unchanged from main.py (modularization PR3). main.py registers the route
at the spot the section used to occupy, after the sales-order ship routes and
before the sales dashboard, so route order is unchanged.

main is reached through `api` at call time: `api.resolve_order_id` (id or order
number), `api.verify_api_key_flexible` (X-API-Key header, ?key= query
parameter, or a signed ?receipt=&expires=&sig= link checked by
ship_tickets.verify_packing_link), `api.get_transaction`, `api.POSTED_LINES`,
`api.get_plant_now` and `api.logger`. Auth, link signing
(ship_tickets.py), the commitShipOrder link (main.py::_floor_packing_slip),
the dashboard-key refusal (permissions / DASHBOARD_KEY_ALLOWLIST) and the
access-log redaction of `key=` all stay where they were.

Read-only: the slip reads orders, posted ledger lines and lots; nothing here
writes.
"""
import io
import math

from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse


def register_routes(app, api):
    """Returns the handler so main can keep it as main.generate_packing_slip."""
    @app.get("/sales/orders/{order_id}/packing-slip")
    def generate_packing_slip(order_id: int = Depends(api.resolve_order_id), _: bool = Depends(api.verify_api_key_flexible)):
        """Generate a printable packing slip PDF for a sales order."""
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.lib.colors import HexColor
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Spacer, Paragraph
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER

        try:
            with api.get_transaction() as cur:
                # ── Fetch order header + customer ──
                cur.execute("""
                    SELECT so.id, so.order_number, so.order_date, so.requested_ship_date,
                           so.status, so.notes,
                           c.name AS customer_name, c.address AS customer_address
                    FROM sales_orders so
                    JOIN customers c ON c.id = so.customer_id
                    WHERE so.id = %s
                """, (order_id,))
                order = cur.fetchone()
                if not order:
                    raise HTTPException(404, f"Order #{order_id} not found")
                if order['status'] == 'cancelled':
                    raise HTTPException(400, f"Order {order['order_number']} is cancelled — cannot generate packing slip")

                # ── Fetch order lines ──
                cur.execute("""
                    SELECT sol.id, sol.product_id, p.name AS product_name,
                           sol.quantity_lb, sol.quantity_shipped_lb,
                           p.case_size_lb, sol.line_status, sol.ordered_quantity,
                           COALESCE(p.is_service, false) AS is_service
                    FROM sales_order_lines sol
                    JOIN products p ON p.id = sol.product_id
                    WHERE sol.sales_order_id = %s AND sol.line_status != 'cancelled'
                    ORDER BY sol.id
                """, (order_id,))
                lines = cur.fetchall()

                # ── Lot allocation: actual shipment data or FIFO preview ──
                line_allocations = []

                # Check if this order has committed shipment records
                cur.execute("""
                    SELECT s.id
                    FROM shipments s
                    JOIN shipment_lines sl ON sl.shipment_id = s.id
                    JOIN ledger_current_transactions t ON t.id = sl.transaction_id
                    WHERE s.sales_order_id = %s
                      AND t.effective_status = 'posted'
                    LIMIT 1
                """, (order_id,))
                has_shipments = cur.fetchone() is not None

                if has_shipments:
                    # ── Post-shipment: pull actual lot allocations from shipment records ──
                    cur.execute("""
                        SELECT p.name AS product_name, p.case_size_lb,
                               l.lot_code, l.supplier_lot_code,
                               ABS(tl.quantity_lb) AS quantity_lb
                        FROM shipment_lines sl
                        JOIN ledger_current_transactions t ON t.id = sl.transaction_id
                            AND t.effective_status = 'posted'
                        JOIN ledger_current_transaction_lines tl ON tl.transaction_id = sl.transaction_id
                        JOIN lots l ON l.id = tl.lot_id
                        JOIN products p ON p.id = tl.product_id
                        WHERE sl.shipment_id IN (
                            SELECT s.id FROM shipments s WHERE s.sales_order_id = %s
                        )
                        AND tl.quantity_lb != 0
                        ORDER BY p.name, l.created_at ASC
                    """, (order_id,))
                    shipped_lots = cur.fetchall()

                    for row in shipped_lots:
                        product_name = row['product_name']
                        case_size = float(row['case_size_lb']) if row['case_size_lb'] else None
                        qty = float(row['quantity_lb'])

                        product_lower = product_name.lower()
                        non_weight_keywords = ('pallet', 'freight', 'delivery', 'surcharge', 'charge', 'fee')
                        is_non_weight = any(kw in product_lower for kw in non_weight_keywords)

                        if is_non_weight:
                            qty_display = str(int(qty)) if qty == int(qty) else str(qty)
                        elif case_size and case_size > 0:
                            cases = round(qty / case_size)
                            qty_display = f"{qty:g} lb \u00b7 {cases} units"
                        else:
                            qty_display = f"{qty:g} lb"

                        line_allocations.append({
                            "product_name": product_name,
                            "lot_code": row['lot_code'],
                            "supplier_lot_code": row['supplier_lot_code'],
                            "qty_display": qty_display
                        })

                    # Include non-weight order lines that may not have shipment lot records
                    shipped_product_names = {r['product_name'] for r in shipped_lots}
                    for line in lines:
                        product_name = line['product_name']
                        product_lower = product_name.lower()
                        non_weight_keywords = ('pallet', 'freight', 'delivery', 'surcharge', 'charge', 'fee')
                        is_non_weight = (line['is_service'] if line['ordered_quantity'] is not None else
                                         line['is_service'] or any(kw in product_lower for kw in non_weight_keywords))
                        if is_non_weight and product_name not in shipped_product_names:
                            count = float(line['ordered_quantity'] if line['ordered_quantity'] is not None else line['quantity_lb'])
                            qty_display = f"{count:g}"
                            line_allocations.append({
                                "product_name": product_name,
                                "lot_code": "N/A",
                                "qty_display": qty_display
                            })

                else:
                    # ── Pre-shipment: FIFO lot allocation preview ──
                    for line in lines:
                        product_name = line['product_name']
                        qty_lb = float(line['quantity_lb'])
                        remaining_lb = qty_lb - float(line['quantity_shipped_lb'])
                        case_size = float(line['case_size_lb']) if line['case_size_lb'] else None

                        # Detect non-weight items
                        product_lower = product_name.lower()
                        non_weight_keywords = ('pallet', 'freight', 'delivery', 'surcharge', 'charge', 'fee')
                        is_non_weight = (line['is_service'] if line['ordered_quantity'] is not None else
                                         line['is_service'] or any(kw in product_lower for kw in non_weight_keywords))

                        if is_non_weight:
                            count = float(line['ordered_quantity'] if line['ordered_quantity'] is not None else line['quantity_lb'])
                            qty_display = f"{count:g}"
                            line_allocations.append({
                                "product_name": product_name,
                                "lot_code": "N/A",
                                "qty_display": qty_display
                            })
                            continue

                        # Query FIFO lots
                        cur.execute(f"""
                            SELECT l.id, l.lot_code, l.supplier_lot_code,
                                   COALESCE(SUM(tl.quantity_lb), 0) AS available
                            FROM lots l
                            LEFT JOIN {api.POSTED_LINES} tl ON tl.lot_id = l.id
                            WHERE l.product_id = %s
                            GROUP BY l.id
                            HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                            ORDER BY COALESCE(l.received_at, l.created_at) ASC
                        """, (line['product_id'],))
                        lots = cur.fetchall()

                        remaining_need = remaining_lb if remaining_lb > 0 else qty_lb
                        allocated = False
                        for lot in lots:
                            if remaining_need <= 0:
                                break
                            avail = float(lot['available'])
                            take = min(avail, remaining_need)
                            qty_display_val = take
                            if case_size and case_size > 0:
                                cases = round(take / case_size)
                                qty_display = f"{take:g} lb \u00b7 {cases} units"
                            else:
                                qty_display = f"{take:g} lb"
                            line_allocations.append({
                                "product_name": product_name,
                                "lot_code": lot['lot_code'],
                                "supplier_lot_code": lot['supplier_lot_code'],
                                "qty_display": qty_display
                            })
                            remaining_need -= take
                            allocated = True

                        if remaining_need > 0:
                            # Shortfall — show INSUFFICIENT row
                            if case_size and case_size > 0:
                                short_cases = math.ceil(remaining_need / case_size)
                                qty_display = f"{remaining_need:g} lb \u00b7 {short_cases} units"
                            else:
                                qty_display = f"{remaining_need:g} lb"
                            insufficient_entry = {
                                "product_name": product_name,
                                "lot_code": "INSUFFICIENT",
                                "qty_display": qty_display
                            }
                            # Check if unpacked batch inventory exists for this FG product.
                            # If /make was run but /pack was not, batch inventory sits idle
                            # while the FG SKU shows zero on-hand.
                            cur.execute(f"""
                                SELECT p2.name AS batch_name,
                                       COALESCE(SUM(tl.quantity_lb), 0) AS batch_available
                                FROM products p
                                JOIN products p2 ON p2.id = p.parent_batch_product_id
                                JOIN lots l ON l.product_id = p2.id
                                JOIN {api.POSTED_LINES} tl ON tl.lot_id = l.id
                                WHERE p.id = %s AND p.parent_batch_product_id IS NOT NULL
                                GROUP BY p2.name
                                HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                            """, (line['product_id'],))
                            batch_row = cur.fetchone()
                            if batch_row:
                                batch_avail = float(batch_row['batch_available'])
                                insufficient_entry["batch_hint"] = (
                                    f"Note: {batch_avail:g} lb of {batch_row['batch_name']} is available "
                                    f"— run /pack to convert to finished goods."
                                )
                            line_allocations.append(insufficient_entry)
                        elif not allocated:
                            # No lots at all
                            if case_size and case_size > 0:
                                cases = round(qty_lb / case_size)
                                qty_display = f"{qty_lb:g} lb \u00b7 {cases} units"
                            else:
                                qty_display = f"{qty_lb:g} lb"
                            insufficient_entry = {
                                "product_name": product_name,
                                "lot_code": "INSUFFICIENT",
                                "qty_display": qty_display
                            }
                            # Same batch-inventory cross-reference for zero-lot case
                            cur.execute(f"""
                                SELECT p2.name AS batch_name,
                                       COALESCE(SUM(tl.quantity_lb), 0) AS batch_available
                                FROM products p
                                JOIN products p2 ON p2.id = p.parent_batch_product_id
                                JOIN lots l ON l.product_id = p2.id
                                JOIN {api.POSTED_LINES} tl ON tl.lot_id = l.id
                                WHERE p.id = %s AND p.parent_batch_product_id IS NOT NULL
                                GROUP BY p2.name
                                HAVING COALESCE(SUM(tl.quantity_lb), 0) > 0
                            """, (line['product_id'],))
                            batch_row = cur.fetchone()
                            if batch_row:
                                batch_avail = float(batch_row['batch_available'])
                                insufficient_entry["batch_hint"] = (
                                    f"Note: {batch_avail:g} lb of {batch_row['batch_name']} is available "
                                    f"— run /pack to convert to finished goods."
                                )
                            line_allocations.append(insufficient_entry)

                # ── Build PDF ──
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=letter,
                                        topMargin=0.5*inch, bottomMargin=0.5*inch,
                                        leftMargin=0.6*inch, rightMargin=0.6*inch)

                styles = getSampleStyleSheet()
                elements = []

                # Colors
                dark_gray = HexColor('#333333')
                med_gray = HexColor('#666666')
                light_gray = HexColor('#999999')
                header_bg = HexColor('#4a4a4a')
                white = HexColor('#FFFFFF')
                row_alt = HexColor('#F5F5F5')

                # Custom styles
                style_company = ParagraphStyle('Company', parent=styles['Normal'],
                                               fontSize=12, leading=14, textColor=dark_gray,
                                               fontName='Helvetica-Bold')
                style_company_detail = ParagraphStyle('CompanyDetail', parent=styles['Normal'],
                                                       fontSize=8, leading=10, textColor=med_gray)
                style_title = ParagraphStyle('SlipTitle', parent=styles['Normal'],
                                              fontSize=18, leading=22, textColor=dark_gray,
                                              fontName='Helvetica-Bold', spaceAfter=6)
                style_label = ParagraphStyle('Label', parent=styles['Normal'],
                                              fontSize=7, leading=9, textColor=light_gray,
                                              fontName='Helvetica-Bold')
                style_value = ParagraphStyle('Value', parent=styles['Normal'],
                                              fontSize=9, leading=11, textColor=dark_gray)
                style_value_bold = ParagraphStyle('ValueBold', parent=styles['Normal'],
                                                  fontSize=9, leading=11, textColor=dark_gray,
                                                  fontName='Helvetica-Bold')
                style_small_gray = ParagraphStyle('SmallGray', parent=styles['Normal'],
                                                  fontSize=7, leading=9, textColor=light_gray)
                style_footer = ParagraphStyle('Footer', parent=styles['Normal'],
                                               fontSize=7, leading=9, textColor=light_gray)
                style_section_label = ParagraphStyle('SectionLabel', parent=styles['Normal'],
                                                      fontSize=10, leading=12, textColor=dark_gray,
                                                      fontName='Helvetica-Bold', spaceBefore=12,
                                                      spaceAfter=4)
                style_sig_line = ParagraphStyle('SigLine', parent=styles['Normal'],
                                                fontSize=9, leading=18, textColor=dark_gray)

                # ── HEADER: Company info ──
                company_block = [
                    Paragraph("CNS Confectionery Products LLC", style_company),
                    Paragraph("33 Hook Road", style_company_detail),
                    Paragraph("Bayonne, NJ 07002 US", style_company_detail),
                    Paragraph("(201) 823-1400", style_company_detail),
                    Paragraph("miriam@cnscoinc.com", style_company_detail),
                ]
                for p in company_block:
                    elements.append(p)

                elements.append(Spacer(1, 0.25*inch))

                # ── TITLE ──
                elements.append(Paragraph("Packing Slip", style_title))
                elements.append(Spacer(1, 0.15*inch))

                # ── INFO ROW (4 columns) ──
                # Customer address
                cust_name = order['customer_name'] or ""
                cust_addr = order['customer_address'] or "Address on file"
                addr_lines = cust_addr.split('\n') if cust_addr else ["Address on file"]
                bill_to_text = f"<b>{cust_name}</b><br/>" + "<br/>".join(addr_lines)
                ship_to_text = bill_to_text  # Mirror for now

                ship_date = str(order['requested_ship_date']) if order['requested_ship_date'] else "TBD"
                order_date = str(order['order_date']) if order['order_date'] else ""
                order_number = order['order_number']

                style_info = ParagraphStyle('InfoCell', parent=styles['Normal'],
                                             fontSize=8, leading=10, textColor=dark_gray)

                col1 = [Paragraph("BILL TO", style_label), Paragraph(bill_to_text, style_info)]
                col2 = [Paragraph("SHIP TO", style_label), Paragraph(ship_to_text, style_info)]
                col3 = [
                    Paragraph("SHIP DATE", style_label), Paragraph(ship_date, style_info),
                    Spacer(1, 4),
                    Paragraph("SHIP VIA", style_label), Paragraph("Customer Pick Up", style_info)
                ]
                col4 = [
                    Paragraph("SO #", style_label), Paragraph(f"<b>{order_number}</b>", style_info),
                    Spacer(1, 4),
                    Paragraph("DATE", style_label), Paragraph(order_date, style_info)
                ]

                # Render the 4-column info as a table
                from reportlab.platypus import KeepTogether
                info_table_data = [[col1, col2, col3, col4]]
                info_table = Table(info_table_data, colWidths=[2.0*inch, 2.0*inch, 1.5*inch, 1.5*inch])
                info_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('TOPPADDING', (0, 0), (-1, -1), 0),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
                ]))
                elements.append(info_table)
                elements.append(Spacer(1, 0.15*inch))

                # ── PURCHASE ORDER ──
                po_ref = order['notes'] if order['notes'] else ""
                if po_ref:
                    elements.append(Paragraph("PURCHASE ORDER", style_label))
                    elements.append(Paragraph(po_ref, style_value))
                    elements.append(Spacer(1, 0.1*inch))

                # ── ITEMS TABLE ──
                style_table_header = ParagraphStyle('TableHeader', parent=styles['Normal'],
                                                     fontSize=8, leading=10, textColor=white,
                                                     fontName='Helvetica-Bold')
                style_table_cell = ParagraphStyle('TableCell', parent=styles['Normal'],
                                                   fontSize=8, leading=10, textColor=dark_gray)
                style_table_cell_bold = ParagraphStyle('TableCellBold', parent=styles['Normal'],
                                                        fontSize=8, leading=10, textColor=dark_gray,
                                                        fontName='Helvetica-Bold')

                table_data = [[
                    Paragraph("DATE", style_table_header),
                    Paragraph("ACTIVITY", style_table_header),
                    Paragraph("LOT #", style_table_header),
                    Paragraph("QTY", style_table_header),
                ]]

                style_supplier_ref = ParagraphStyle('SupplierRef', parent=styles['Normal'],
                                                        fontSize=6, leading=8, textColor=light_gray)

                style_batch_hint = ParagraphStyle('BatchHint', parent=styles['Normal'],
                                                        fontSize=7, leading=9, textColor=HexColor('#CC6600'),
                                                        fontName='Helvetica-Oblique')

                order_date_str = str(order['order_date']) if order['order_date'] else ""
                for alloc in line_allocations:
                    lot_style = style_table_cell_bold if alloc['lot_code'] == 'INSUFFICIENT' else style_table_cell
                    # Build lot cell with optional supplier lot reference
                    lot_text = alloc['lot_code']
                    supplier_code = alloc.get('supplier_lot_code')
                    if supplier_code and alloc['lot_code'] not in ('N/A', 'INSUFFICIENT'):
                        lot_cell = [Paragraph(lot_text, lot_style),
                                    Paragraph(f"(Supplier: {supplier_code})", style_supplier_ref)]
                    else:
                        lot_cell = Paragraph(lot_text, lot_style)
                    table_data.append([
                        Paragraph(order_date_str, style_table_cell),
                        Paragraph(alloc['product_name'], style_table_cell),
                        lot_cell,
                        Paragraph(alloc['qty_display'], style_table_cell),
                    ])
                    # If this INSUFFICIENT line has a batch hint, add a note row
                    if alloc.get('batch_hint'):
                        table_data.append([
                            Paragraph("", style_table_cell),
                            Paragraph(alloc['batch_hint'], style_batch_hint),
                            Paragraph("", style_table_cell),
                            Paragraph("", style_table_cell),
                        ])

                items_table = Table(table_data, colWidths=[1.0*inch, 3.2*inch, 1.5*inch, 1.3*inch])

                # Build table style with alternating rows
                table_style_cmds = [
                    ('BACKGROUND', (0, 0), (-1, 0), header_bg),
                    ('TEXTCOLOR', (0, 0), (-1, 0), white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                    ('TOPPADDING', (0, 0), (-1, 0), 6),
                    ('LEFTPADDING', (0, 0), (-1, -1), 6),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 6),
                    ('ALIGN', (3, 0), (3, -1), 'RIGHT'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('LINEBELOW', (0, 0), (-1, 0), 0.5, dark_gray),
                    ('LINEBELOW', (0, -1), (-1, -1), 0.5, light_gray),
                ]
                # Alternating row backgrounds
                for i in range(1, len(table_data)):
                    if i % 2 == 0:
                        table_style_cmds.append(('BACKGROUND', (0, i), (-1, i), row_alt))

                items_table.setStyle(TableStyle(table_style_cmds))
                elements.append(items_table)

                # ── Dashed separator ──
                elements.append(Spacer(1, 0.2*inch))
                from reportlab.platypus import HRFlowable
                elements.append(HRFlowable(width="100%", thickness=0.5, color=light_gray,
                                            dash=[3, 3], spaceBefore=0, spaceAfter=0))
                elements.append(Spacer(1, 0.2*inch))

                # ── WAREHOUSE CONFIRMATION ──
                elements.append(Paragraph("WAREHOUSE CONFIRMATION", style_section_label))
                elements.append(Spacer(1, 0.1*inch))
                elements.append(Paragraph("Picked By: ___________________________________", style_sig_line))
                elements.append(Paragraph("Verified By: ___________________________________", style_sig_line))
                elements.append(Paragraph("Date: ___________________________________", style_sig_line))
                elements.append(Spacer(1, 0.2*inch))

                # ── Traceability note ──
                elements.append(Paragraph(
                    "All lot numbers are system-assigned internal codes from Factory Ledger. "
                    "Do not substitute with supplier lot numbers.",
                    style_small_gray
                ))
                elements.append(Spacer(1, 0.3*inch))

                # ── Footer ──
                now_et = api.get_plant_now()
                footer_ts = now_et.strftime("%Y-%m-%d %I:%M %p ET")
                footer_table_data = [[
                    Paragraph(f"Generated by Factory Ledger | {footer_ts}", style_footer),
                    Paragraph("Page 1 of 1", ParagraphStyle('FooterRight', parent=style_footer, alignment=TA_RIGHT))
                ]]
                footer_table = Table(footer_table_data, colWidths=[5.0*inch, 2.0*inch])
                footer_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'BOTTOM'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 0),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 0),
                ]))
                elements.append(footer_table)

                # ── Render ──
                doc.build(elements)
                buffer.seek(0)

                return StreamingResponse(
                    buffer,
                    media_type="application/pdf",
                    headers={
                        "Content-Disposition": f'inline; filename="packing_slip_{order_number}.pdf"'
                    }
                )

        except HTTPException:
            raise
        except Exception as e:
            api.logger.error(f"Packing slip generation failed: {e}")
            return JSONResponse(status_code=500, content={"error": str(e)})

    return generate_packing_slip
