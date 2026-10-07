#!/usr/bin/env python3
"""DRY RUN ONLY. Reads FL via psql_ro.sh; writes local review files. No apply path."""
from fresh_start_common import (D, OUT, fmt, cell, table, snapshot, analyze,
                                run_arguments, write_csv, movement_section, package_note, opening_balance_reason, stamp,
                                write_late_review, late_review_section, SCOPE_NOTE)

COLUMNS = ['row_type', 'group_status', 'status', 'product_id', 'product_active', 'SKU', 'name', 'lot_id',
           'lot_code', 'paper_row', 'suggestions', 'count_unit', 'case_weight_lb', 'case_weight_source', 'package_decision', 'counted_qty', 'counted_lb',
           'fl_cutoff_lb', 'count_minus_cutoff_lb', 'late_pre_cutoff_lb', 'existing_opening_lb',
           'adjustment_basis_lb', 'candidate_adjustment_lb', 'rounded_adjustment_lb', 'rounding_delta_lb', 'planned_balance_lb', 'adjustment_lb', 'post_cutoff_movement_lb',
           'current_lb', 'expected_current_lb', 'reason', 'detail',
           'transaction_id', 'line_id', 'quantity_lb', 'occurred_at', 'entered_at',
           'entered_after_cutoff', 'confirmation', 'review_flag', 'fingerprint', 'operator_id', 'correction_id', 'correction_target',
           'input_sha256', 'cutoff', 'snapshot_at']


def write_preview(a, output):
    output.mkdir(parents=True, exist_ok=True)
    md = output / f"reset-preview-{a['date']}.md"
    csv = output / f"reset-preview-{a['date']}.csv"
    if {str(csv.resolve()),str(md.resolve())} & {a['input_path'],a['confirmations_path']}:
        raise ValueError('Output would overwrite the count input; use a different output directory.')
    assert a['reason'] == opening_balance_reason(stamp(a['cutoff']))
    review_path = write_late_review(a, output)
    ready = sum(s.startswith('READY') for s in a['group_status'].values())
    changed = sum(r['adjustment_lb'] is not None and r['adjustment_lb'] != 0 for r in a['rows'])
    lines = ['# Reset preview — DRY RUN ONLY', '',
             f"No inventory was changed. {ready} of {len(a['products'])} product groups are complete and ready for owner review; {changed} lot rows have calculated nonzero differences. A calculated difference is not approval. Hold every incomplete or flagged product group.", '',
             SCOPE_NOTE, '',
             f"Physical count cutoff: **{a['cutoff']}**. Read-only FL snapshot: {a['snapshot_at']}.",
             f"Count source: `{a['input_path']}`. SHA-256: `{a['input_sha256']}`.", '',
             f"Exact reason for any later approved adjustment: **{a['reason']}**.", '',
             'The ledger is stored in pounds. BLUE STRIPES products 150–153 use the owner-approved local override of 2.625 lb/case. Products 183, 184, 185, 186, 285 and 288 are expected zero: any positive count is held for owner decision, even if a catalog weight is later supplied. Other products use usable, unambiguous catalog case weights. An explicit or confirmed missing-lot zero needs no case conversion. No new lot is created. Products absent from the sheet remain NOT COUNTED. See package-decisions.md.', '',
             'Planned adjustments are rounded to 4 decimal places (nearest 0.0001 lb; exact halfway ties to even). The original calculation, rounded amount and proposed change are shown separately. Physical counts stay unrounded. A nonzero adjustment on an inactive product is HELD – inactive in FL, owner decision needed; that product cannot be approved.', '',
             'FL at cutoff uses stored physical times; the adjustment basis uses owner-confirmed Y/N classifications for entries typed later. Basis = current FL minus ordinary movements confirmed after the count, including any existing opening adjustment so it is not proposed twice. Unconfirmed entries withhold adjustments for their whole product. No lot suggestion is matched automatically.', '',
             'Original entry times marked migration_backfill_039 are historical estimates; they are not proof of the actual typing time. Current corrections and later entries are listed below. Use one agreed cutoff and freeze stock movements while counting, or reconcile the count back to that cutoff before entering it.', '']
    lines += [f'Owner confirmation template: `{review_path}`. Preserve its fingerprint/identity fields; fill Y/N and rerun with `--late-entry-confirmations`. New or changed entries require a fresh confirmation.', '']
    lines += late_review_section(a)
    ranked = sorted((r for r in a['rows'] if r['adjustment_lb'] is not None and r['adjustment_lb'] != 0),
                    key=lambda r: abs(r['adjustment_lb']), reverse=True)[:20]
    lines += ['## Spot-check before approval — 20 largest proposed changes', '',
              'Compare each row with the photographed paper sheet and physical label. For an absent lot, the cited completion row is the evidence that the whole product was checked. Held calculations with no proposed change are excluded.', '']
    lines += table(['Product / lot','Original lb','Rounded proposed lb','Paper-sheet row','Group status','Owner checked'],
                   [[f"{r['product_id']} / {r['lot_code']}",r['candidate_adjustment_lb'],r['adjustment_lb'],r['paper_row'],r['group_status'],'☐'] for r in ranked])
    if not ranked:
        lines += ['', 'No nonzero adjustments are proposed yet.']
    lines += ['']
    for message in a['general']:
        lines += [f'- {message}']
    lines += ['Warning: '+w for w in a['warnings']]
    lines += ['', '## Product approval register', '']
    lines += table(['Product', 'SKU', 'Name', 'Active in FL', 'Status', 'Owner decision'],
                   [[pid, p.get('odoo_code'), p['name'], 'yes' if p.get('active') else 'no', a['group_status'][pid], 'Pending' if a['group_status'][pid].startswith('READY') else 'NOT APPROVABLE']
                    for pid,p in a['products'].items()])
    records = []
    for pid,p in a['products'].items():
        rs = [r for r in a['rows'] if r['product_id']==pid]
        lines += ['', f"## Product {pid} · SKU {p.get('odoo_code')} · {p['name']}", '',
                  a['group_status'][pid], '', 'Owner approval: ____________________  Date/time: ____________________' if a['group_status'][pid].startswith('READY') else 'NOT APPROVABLE — resolve the hold or incomplete count first.', '']
        if package_note(p):
            lines += [package_note(p), '']
        if p['type'] == 'finished':
            weight = next((r['case_weight_lb'] for r in rs if r['case_weight_lb'] is not None), None)
            lines += [f"Conversion used: {fmt(weight)} lb per case." if weight is not None else 'Case conversion is unresolved; only zero counts can be calculated.', '']
        if pid in a['issues']:
            lines += ['- '+i for i in a['issues'][pid]] + ['']
        lines += table(['Lot / ID', 'Paper row', 'Suggestions (never auto-match)', 'Count / unit', 'Count lb', 'FL at cutoff lb', 'Count minus cutoff lb', 'Late earlier lb (included)',
                        'Existing opening lb', 'Adjustment basis lb', 'Original change lb', 'Rounded change lb', 'Proposed change lb', 'Planned balance lb',
                        'Later movements lb', 'Current FL lb', 'Status / detail'],
                       [[f"{r['lot_code']} / {r['lot_id'] or 'none'}", r['paper_row'], r['suggestions'],
                         f"{fmt(r['counted_qty'])} {r['count_unit']}" if r['counted_qty'] is not None else 'NOT COUNTED',
                         r['counted_lb'], r['fl_cutoff_lb'], r['count_minus_cutoff_lb'], r['late_pre_cutoff_lb'], r['existing_opening_lb'],
                         r['adjustment_basis_lb'],r['candidate_adjustment_lb'],r['rounded_adjustment_lb'],r['adjustment_lb'],r['planned_balance_lb'],r['post_cutoff_movement_lb'],
                         r['current_lb'],r['status']+'; '+r['detail']] for r in rs])
        for r in rs:
            records.append(dict(r, row_type='LOT_REVIEW'))
    for x in a['late_entries']:
        records.append(dict(x, row_type='LATE_ENTRY_OWNER_REVIEW', status=x['confirmation_status']))
    for x in a['movements']:
        records.append(dict(x, row_type='MOVEMENT_EXPLANATION', status=x['movement_kind']))
    for x in a['corrections']:
        records.append(dict(row_type='CORRECTION_EXPLANATION', correction_id=x['id'],
                            correction_target=f"{x['target_table']} / {x['target_id']}",
                            status=x['event_type'],entered_at=x['created_at'],operator_id=x['operator_id'],detail=x['reason']))
    for x in a['unassigned']:
        records.append(dict(x,row_type='UNASSIGNED_LEDGER_EXCEPTION',status='HOLD — lot identity missing/mismatched'))
    for warning in a['warnings']:
        records.append(dict(row_type='WARNING',detail=warning))
    for message in a['general']:
        records.append(dict(row_type='GENERAL_EXCEPTION',detail=message))
    for pid,issues in a['issues'].items():
        for issue in issues:
            records.append(dict(row_type='PRODUCT_EXCEPTION',product_id=pid,detail=issue))
    if a['unassigned']:
        lines += ['', '## Posted lines without a usable lot', '']
        lines += table(['TX / line','Product','Lot ID','lb'],
                       [[f"{x['transaction_id']} / {x['line_id']}",x['product_id'],x['lot_id'],x['quantity_lb']] for x in a['unassigned']])
    lines += ['']+movement_section(a)+['', 'This tool has no API client, database mutation, apply mode, or inventory write capability.']
    for record in records:
        record.update(input_sha256=a['input_sha256'],cutoff=a['cutoff'],snapshot_at=a['snapshot_at'])
    write_csv(csv, COLUMNS, records)
    md.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'DRY RUN ONLY: {ready}/{len(a["products"])} product groups ready for owner review; {changed} calculated nonzero lot differences.')
    print(md)
    print(csv)


def main():
    import sys
    if '--v3' in sys.argv:
        sys.argv.remove('--v3')
        from fresh_start_v3 import main as live_count_main
        return live_count_main(verify=False)
    path, output, cutoff, confirmations = run_arguments(__doc__)
    write_preview(analyze(snapshot(), path, cutoff, confirmations), output)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError) as e:
        raise SystemExit(str(e))
