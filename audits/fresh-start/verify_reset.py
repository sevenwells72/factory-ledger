#!/usr/bin/env python3
"""READ ONLY. Compare counted opening stock plus later movements to current FL."""
from fresh_start_common import (TOLERANCE, fmt, table, snapshot, analyze,
                                run_arguments, write_csv, movement_section, late_review_section,
                                opening_balance_reason, stamp, SCOPE_NOTE)


def write_verification(a, output):
    output.mkdir(parents=True, exist_ok=True)
    md = output / f"reset-verification-{a['date']}.md"
    csv = output / f"reset-verification-{a['date']}.csv"
    if {str(csv.resolve()),str(md.resolve())} & {a['input_path'],a['confirmations_path']}:
        raise ValueError('Output would overwrite the count input; use a different output directory.')
    assert a['reason'] == opening_balance_reason(stamp(a['cutoff']))
    counted = [r for r in a['rows'] if r['counted_for_verdict']]
    matched = [r for r in counted if r['unexplained_lb'] is not None and abs(r['unexplained_lb']) <= TOLERANCE]
    exceptions = [r for r in counted if r not in matched]
    verdict = f'{len(counted)} lots counted, {len(matched)} reconciled, {len(exceptions)} unexplained differences'
    incomplete = {pid: s for pid,s in a['group_status'].items() if not s.startswith('READY')}
    complete = not exceptions and not incomplete and not a['general'] and not a['unassigned']
    lines = [verdict, '',
             'FULL SCOPE RECONCILED.' if complete else 'EXCEPTIONS OR INCOMPLETE SCOPE — this is not a full-reset sign-off.', '',
             SCOPE_NOTE, '',
             f"Physical count cutoff: {a['cutoff']}. Read-only snapshot: {a['snapshot_at']}.",
             f"Count source: `{a['input_path']}`. SHA-256: `{a['input_sha256']}`.", '',
             'Expected current FL = counted pounds + ordinary movements after cutoff, using owner Y/N confirmations for entries typed later. Existing adjustments with the exact opening-balance reason remain in actual FL and are excluded from ordinary movements. Late-entered physical movements at/before cutoff are already covered by the physical count, so they are not added again. Differences within 0.0001 lb are reconciled (FL floating-point tolerance); exact amounts remain in CSV.', '',
             'The lot count includes explicit counted rows and nonzero proposed/already-posted opening adjustments; it excludes untouched old zero lots. A numerically matching lot is not proof that an adjustment was authorized or posted. Unknown lots and quantities without a usable case weight remain exceptions. Lots inferred absent are included only for explicitly completed products. All products absent from the sheet remain NOT COUNTED.', '',
             'BS means BLUE STRIPES. Products 150–153 use 2.625 lb/case under the local owner decision. Products 183, 184, 185, 186, 285 and 288 are expected zero; stock found under them remains an owner-decision exception. Product 185 requires cases, packs per case and weight per pack (including its unit) in notes. See package-decisions.md.', '',
             '## Counted-lot exceptions', '']
    for warning in a['warnings']:
        lines += ['WARNING (does not fail matching counts): '+warning,'']
    lines += table(['Product / lot', 'Count / unit', 'Expected current lb', 'Actual FL lb', 'Unexplained lb', 'Reason'],
                   [[f"{r['product_id']} / {r['lot_code']}",f"{fmt(r['counted_qty'])} {r['count_unit']}",
                     r['expected_current_lb'],r['current_lb'],r['unexplained_lb'],r['status']+'; '+r['detail']] for r in exceptions])
    if not exceptions:
        lines += ['', 'No unexplained differences among the counted, convertible lots.']
    lines += ['', '## Products still uncounted or held', '']
    lines += table(['Product', 'Name', 'Status', 'Details'],
                   [[pid,a['products'][pid]['name'],status,'; '.join(a['issues'].get(pid,[]))]
                    for pid,status in incomplete.items()])
    for item in a['general']:
        lines += ['', item]
    records = []
    for r in a['rows']:
        result = 'RECONCILED' if r in matched else 'EXCEPTION' if r in counted else 'NOT COUNTED'
        records.append(dict(r, verification=result, cutoff=a['cutoff'], snapshot_at=a['snapshot_at'], input_sha256=a['input_sha256']))
    columns = ['verification','group_status','product_id','SKU','name','lot_id','lot_code','paper_row','counted_for_verdict','count_unit',
               'case_weight_lb','case_weight_source','package_decision','counted_qty','counted_lb','current_lb','post_cutoff_movement_lb',
               'expected_current_lb','unexplained_lb','existing_opening_lb','status','detail',
               'cutoff','snapshot_at','input_sha256']
    write_csv(csv,columns,records)
    lines += ['']+late_review_section(a)+movement_section(a)
    md.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(verdict)
    print('FULL SCOPE RECONCILED.' if complete else f'Exceptions remain: {len(incomplete)} product groups uncounted/held; see report.')
    print(md)
    print(csv)
    return 0 if complete else 2


def main():
    import sys
    if '--v3' in sys.argv:
        sys.argv.remove('--v3')
        from fresh_start_v3 import main as live_count_main
        return live_count_main(verify=True)
    path, output, cutoff, confirmations = run_arguments(__doc__)
    return write_verification(analyze(snapshot(), path, cutoff, confirmations), output)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as e:
        raise SystemExit(str(e))
