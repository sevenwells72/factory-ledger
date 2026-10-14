#!/usr/bin/env python3
"""READ ONLY. Whole-reset proof across every area group of one cutover reset (runbook O-3, step 20).

`verify_reset.py --v3` proves ONE signed preview/approval/journal triplet. A cutover
posts several area groups, each with its own signed plan and executor journal, so the
release gate needs the union: every reset-scope product approved by exactly one group,
every count observation inside that group's signed plan, every journal receipt mapped to
an approved row, no reset post outside a signed plan, and the whole scope reconciled.

Input is a manifest (JSON) listing every group. Journals are never concatenated: each is
proved on its own against its own signed preview, then the proved posts are unioned.

  python3 audits/fresh-start/verify_reset.py --whole-reset "$RUN/whole-reset-manifest.json" \
    --output-dir "$RUN/verification/whole-reset"

Exit 0 = FULL RESET SCOPE RECONCILED. Exit 2 = a precise list of gaps. A malformed
manifest exits with a message. One read-only snapshot; no API client, no writes to FL.
"""
import argparse
import csv
import json
import re
from collections import defaultdict
from decimal import Decimal as D
from pathlib import Path

import fresh_start_common as f
import fresh_start_v3 as v

check = v.check
MANIFEST_KEYS = {'version', 'target', 'expected_reset_products', 'final_inputs', 'groups', 'rehearsal_scope_product_ids', 'notes'}
GROUP_KEYS = {'name', 'areas', 'approved_preview', 'approval', 'journal', 'no_change', 'notes'}
INPUT_KEYS = {'count', 'coverage', 'moves', 'review'}
# = apply_reset.APPROVAL_FIELDS_V3. Kept here so this proof never imports the write executor.
APPROVAL_FIELDS = ('Preview file', 'SHA-256', 'Approved product IDs', 'Owner name', 'Named actor', 'Date', 'Signature',
                   'Opening entry classification', 'Owner approves held entries', 'Extra-kosher lots')
SHARED_ACTORS = ('legacy-shared-key', 'shared', 'master')
VERDICT_OK = 'FULL RESET SCOPE RECONCILED'


def gap(gaps, kind, detail, group=None, product_id=None, lot_code=None):
    gaps.append(dict(kind=kind, group=group or '', product_id=product_id if product_id is not None else '',
                     lot_code=lot_code or '', detail=detail))


def load_manifest(path):
    path = f.local_path(path)
    m = json.loads(path.read_text(encoding='utf-8'))
    check(isinstance(m, dict) and set(m) <= MANIFEST_KEYS, 'Manifest has unknown keys: ' + ', '.join(sorted(set(m) - MANIFEST_KEYS)) if isinstance(m, dict) else 'Manifest must be a JSON object')
    check(m.get('version') == 1, 'Manifest version must be 1')
    check(m.get('target') in f.TARGETS, 'Manifest target must be production or staging')
    check(type(m.get('expected_reset_products')) is int and m['expected_reset_products'] > 0,
          'expected_reset_products is required (178 for the Oct 13 packet); a catalog change must be reconciled explicitly')
    base = path.parent

    def resolve(value, label):
        check(isinstance(value, str) and value, label + ' path is required')
        p = Path(value)
        return f.local_path(p if p.is_absolute() else base / p)

    final = m.get('final_inputs')
    check(isinstance(final, dict) and set(final) <= INPUT_KEYS and {'count', 'coverage', 'moves'} <= set(final),
          'final_inputs needs count, coverage and moves (review optional)')
    m['final_inputs'] = {k: resolve(val, 'final_inputs.' + k) for k, val in final.items() if val}
    groups = m.get('groups')
    check(isinstance(groups, list) and groups, 'Manifest lists no area groups')
    names = set()
    for g in groups:
        check(isinstance(g, dict) and set(g) <= GROUP_KEYS, 'Group has unknown keys')
        check(isinstance(g.get('name'), str) and g['name'].strip() and g['name'] not in names, 'Each group needs a unique name')
        names.add(g['name'])
        g['approved_preview'] = resolve(g.get('approved_preview'), g['name'] + ' approved_preview')
        g['approval'] = resolve(g.get('approval'), g['name'] + ' approval')
        no_change = g.setdefault('no_change', False)
        check(type(no_change) is bool, g['name'] + ': no_change must be true or false')
        # Documented no-change case (runbook step 18): no posts, so the executor wrote no journal.
        check(bool(g.get('journal')) != no_change, g['name'] + ': give a journal, or no_change: true with no journal')
        g['journal'] = resolve(g['journal'], g['name'] + ' journal') if g.get('journal') else None
        if 'areas' in g:
            check(isinstance(g['areas'], list) and all(isinstance(a, str) and a for a in g['areas']), g['name'] + ': areas must be a list')
    scope = m.get('rehearsal_scope_product_ids')
    if scope is not None:
        # Staging rehearsals count a few synthetic products, never the whole catalog.
        check(m['target'] == 'staging', 'rehearsal_scope_product_ids is allowed only for a staging rehearsal')
        check(isinstance(scope, list) and scope and all(type(x) is int for x in scope) and len(set(scope)) == len(scope),
              'rehearsal_scope_product_ids must be distinct product ids')
    m['path'] = path
    return m


def signed_plan(preview, approval, target):
    """Read-only twin of apply_reset.load_v3_plan's approval rules: the signed preview, its
    owner fields, the approved scope and the still-unchanged signed inputs."""
    raw, signed = preview.read_bytes(), approval.read_bytes()
    plan = json.loads(raw, parse_float=D)
    check(plan.get('version') == 3, 'Not a v3 preview')
    check(plan.get('reset_scope_policy') == f.RESET_SCOPE_POLICY, 'Reset scope decision changed; regenerate and reapprove preview')
    check(plan.get('target', 'production') == target, f'Preview was made for {plan.get("target", "production")}, not {target}')
    fields = {}
    for line in signed.decode().splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            if key in APPROVAL_FIELDS:
                check(key not in fields, 'Duplicate approval field')
                fields[key] = value.strip()
    check(set(fields) == set(APPROVAL_FIELDS) and all(fields.values()), 'Approval is incomplete or unsigned')
    check(fields['Preview file'] == preview.name and fields['SHA-256'].lower() == v.sha(raw), 'Signed preview hash/filename mismatch')
    check(fields['Signature'] == fields['Owner name'], 'Owner signature mismatch')
    check(fields['Named actor'] not in SHARED_ACTORS, 'Named personal actor required')
    check(fields['Opening entry classification'] == 'N', 'Administrative opening classification must be N')
    check(fields['Owner approves held entries'] in ('Y', 'N'), 'Owner approves held entries must be Y or N')
    check(re.fullmatch(r'[1-9]\d*(?:\s*,\s*[1-9]\d*)*', fields['Approved product IDs']), 'Invalid approved product IDs')
    ids = [int(x) for x in fields['Approved product IDs'].split(',')]
    check(len(ids) == len(set(ids)) and not set(ids) & f.BILLING_RESET_EXCLUDED_IDS, 'Duplicate or billing product in approval')
    check(not plan['general'], 'Signed preview still has general exceptions')
    approved = {}
    for r in plan['rows']:
        check(not f.reset_exclusion_reason(dict(id=r['product_id'], type=r['product_type'])), 'Packaging/billing item in signed plan')
        change = f.number(r['adjustment_lb']) if r['adjustment_lb'] is not None else None
        check(not change or r['product_id'] in ids, 'A proposed change is outside the approved products')
        if r['product_id'] not in ids:
            continue
        check(r['status'] == 'READY' and not r['holds'], f'Approved product {r["product_id"]} has a held lot')
        key = (r['product_id'], r['lot_code'])
        check(key not in approved, 'Duplicate lot proposal')
        approved[key] = dict(r, change=change or D(0))
    check(set(ids) == {pid for pid, _ in approved}, 'Approved product absent from lot preview')
    needs = [r for r in approved.values() if r['change'] and r.get('owner_approvals')]
    check(not needs or fields['Owner approves held entries'] == 'Y', 'Plan has owner-held entries but approval says N')
    kosher = {f'{r["product_id"]}:{r["lot_code"]}' for r in approved.values() if r.get('new_lot') and r['change']}
    listed = set() if fields['Extra-kosher lots'].lower() == 'none' else {x.strip() for x in fields['Extra-kosher lots'].split(',') if x.strip()}
    check(listed == kosher, 'Extra-kosher lots do not match the plan')
    # The signed inputs must be immutable: same bytes, and the same embedded count/review.
    src = plan['sources']
    for item in src.values():
        check(v.sha(f.local_path(item['path']).read_bytes()) == item['sha256'], 'A signed input file changed after approval')
    data = v.load_inputs(src['count']['path'], src['coverage']['path'], src['moves']['path'], src.get('review', {}).get('path'))
    check(v.canonical(data) == v.canonical(plan['data']), 'Embedded count/review differs from the signed input files')
    return dict(plan=plan, fields=fields, ids=set(ids), approved=approved, data=data, sha=v.sha(raw))


def prove_group(g, target, s, gaps):
    """One group exactly as step 18 proves it, plus: every approved change posted once,
    nothing held/uncertain, every journal receipt matching its proved post."""
    name = g['name']
    try:
        sp = signed_plan(g['approved_preview'], g['approval'], target)
    except (ValueError, RuntimeError, OSError, KeyError) as e:
        gap(gaps, 'UNSIGNED_OR_INVALID_GROUP', f'{type(e).__name__}: {e}' if not isinstance(e, ValueError) else str(e), name)
        return None
    plan = sp['plan']
    if 'areas' in g and sorted(g['areas']) != (plan.get('area_group') or []):
        gap(gaps, 'AREA_MISMATCH', f'manifest areas {sorted(g["areas"])} but the signed preview was made for {plan.get("area_group")}', name)
    result = dict(name=name, areas=plan.get('area_group'), preview_sha256=sp['sha'], approved_products=sorted(sp['ids']),
                  approved=sp['approved'], data=sp['data'], actor=sp['fields']['Named actor'], owner=sp['fields']['Owner name'],
                  snapshot_at=plan['snapshot_at'], plan_rows=plan['rows'], completed={}, journal=None, no_change=g['no_change'])
    changes = {k: r for k, r in sp['approved'].items() if r['change']}
    if g['no_change']:
        if changes:
            gap(gaps, 'NO_CHANGE_GROUP_HAS_CHANGES', f'declared no-change but its signed plan approves {len(changes)} change(s); list its journal', name)
        default = f.OUT / f.target_config(target)['state'] / sp['sha'] / 'journal.jsonl'
        if default.exists():
            gap(gaps, 'NO_CHANGE_GROUP_HAS_JOURNAL', f'an executor journal exists at {default.relative_to(f.OUT)}; list it instead of no_change', name)
        return result
    try:
        proof = v.journal_proof(g['approved_preview'], g['approval'], g['journal'], s, sp['data'])
    except (ValueError, RuntimeError, KeyError, OSError) as e:
        gap(gaps, 'JOURNAL_PROOF_FAILED', str(e) if isinstance(e, (ValueError, RuntimeError)) else f'{type(e).__name__}: {e}', name)
        return result
    result['journal'] = dict(path=str(g['journal'].relative_to(f.OUT)), sha256=proof['journal_sha256'])
    by_lid = {(r['lot_key'] if r.get('new_lot') else r['lot_id']): k for k, r in sp['approved'].items()}
    for lid, line in proof['completed'].items():
        result['completed'][by_lid[lid]] = line
    for lid in proof['held']:
        pid, code = by_lid[lid]
        gap(gaps, 'HELD_NOT_RELEASED', 'ticket is still awaiting the owner; the approved change is not posted', name, pid, code)
    for lid in proof['uncertain']:
        pid, code = by_lid[lid]
        gap(gaps, 'UNCERTAIN_COMMIT', 'a commit has no definite outcome; rerun the same apply (step 19) before release', name, pid, code)
    for key, r in changes.items():
        lid = r['lot_key'] if r.get('new_lot') else r['lot_id']
        if key not in result['completed'] and lid not in proof['held'] and lid not in proof['uncertain']:
            gap(gaps, 'APPROVED_CHANGE_NOT_POSTED', f'approved change {f.fmt(r["change"])} {r["ledger_unit"]} has no proved post', name, *key)
    for seq, intent in proof['intents'].items():
        receipt = proof['receipts'].get(seq)
        if not receipt or receipt.get('outcome') != 'ok':
            continue
        line = proof['completed'].get(intent['lot_id'])
        if not line or line['receipt_number'] != receipt.get('receipt_number') or line['transaction_id'] != receipt.get('transaction_id'):
            gap(gaps, 'RECEIPT_NOT_MAPPED', f'journal receipt {receipt.get("receipt_number")} (seq {seq}) does not match the one proved post for its approved row', name, intent['product_id'])
    return result


def scoped_snapshot(s, ids):
    if ids is None:
        return s
    keep = set(ids)
    return dict(s, products=[p for p in s['products'] if p['id'] in keep])


def union(groups, a, s, final_data, gaps):
    """Product, lot, observation, receipt and post coverage across all proved groups."""
    scope = set(a['products'])
    final_rows = {(r['product_id'], r['lot_code']): r for r in a['rows']}
    final_count = {r['row_id']: r for r in final_data['count']}
    approvers = defaultdict(list)
    for g in groups:
        for pid in g['approved_products']:
            approvers[pid].append(g)
            if pid not in scope:
                gap(gaps, 'OUTSIDE_RESET_SCOPE', 'approved product is not in the reset scope of the final analysis', g['name'], pid)
    products_with_rows = {pid for pid, _ in final_rows}
    nothing = []
    for pid in sorted(scope):
        gs = approvers.get(pid, [])
        if len(gs) > 1:
            gap(gaps, 'OVERLAP', 'approved in more than one group: ' + ', '.join(g['name'] for g in gs), None, pid)
        elif not gs and pid in products_with_rows:
            gap(gaps, 'MISSING_PRODUCT', 'has counted or on-hand lots but no signed group approves it', None, pid)
        elif not gs:
            nothing.append(pid)
    multi = []
    for key, r in sorted(final_rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        gs = approvers.get(key[0], [])
        if len(gs) != 1:
            continue
        g = gs[0]
        signed = g['approved'].get(key)
        if not signed:
            gap(gaps, 'MISSING_LOT', 'lot is in the final count but not in the signed plan of the group that approved its product', g['name'], *key)
            continue
        final_ids, signed_ids = set(r['row_ids']), set(signed['row_ids'])
        for rid in sorted(final_ids - signed_ids):
            gap(gaps, 'MISSING_OBSERVATION', f'observation {rid} was not in the signed plan that reset this lot', g['name'], *key)
        for rid in sorted(signed_ids - final_ids):
            gap(gaps, 'DROPPED_OBSERVATION', f'signed observation {rid} is absent from the final count', g['name'], *key)
        signed_count = {x['row_id']: x for x in g['data']['count']}
        for rid in sorted(final_ids & signed_ids):
            if rid in signed_count and final_count.get(rid) != signed_count[rid]:
                gap(gaps, 'CHANGED_OBSERVATION', f'count row {rid} changed after the group was signed', g['name'], *key)
        # By row id: a sheet may spell a new extra-kosher lot code differently from FL's canonical code.
        areas = sorted({o['area'] for o in a['row_comparison'] if o['row_id'] in final_ids and not o['row_id'].startswith('COVERAGE-')})
        if len(areas) > 1:
            elsewhere = sorted(o['name'] for o in groups if o is not g and any((x['product_id'], x['lot_code']) == key for x in o['plan_rows']))
            multi.append(dict(product_id=key[0], lot_code=key[1], areas=areas, handled_by=g['name'], held_in=elsewhere,
                              row_ids=sorted(final_ids)))
    for g in groups:
        for key in g['approved']:
            if key not in final_rows:
                gap(gaps, 'SIGNED_LOT_NOT_IN_FINAL', 'signed lot does not appear in the final analysis', g['name'], *key)
    # Each lot is posted once, by one group's journal.
    posted = defaultdict(list)
    for g in groups:
        for key, line in g['completed'].items():
            posted[key].append((g['name'], line))
    for key, items in posted.items():
        if len(items) > 1:
            gap(gaps, 'DOUBLE_POST', 'reset posted by several groups: ' + ', '.join(f'{n} {x["receipt_number"]}' for n, x in items), None, *key)
    # No executor post outside an approved plan: any reset-text post, or a physical_count
    # ticket by a group's named actor since the first signed preview, must be a proved post.
    own = {line['line_id'] for g in groups for line in g['completed'].values()}
    actors = {g['actor'] for g in groups}
    since = min((f.stamp(g['snapshot_at']) for g in groups), default=None)
    lots = {l['id']: l for l in s['lots']}
    extra = []
    for x in s['lines']:
        if x['product_id'] not in scope or x['effective_status'] != 'posted' or x['line_id'] in own:
            continue
        lot = lots.get(x.get('lot_id'))
        opening = any(re.match(v.OPENING, t) for t in v.opening_texts(x, lot))
        entered = f.entered_at(x)
        executor = (x.get('reason_code') == 'physical_count' and x.get('ticket_id') and x.get('operator_id') in actors
                    and since is not None and entered is not None and entered >= since)
        if opening or executor:
            extra.append(x)
            gap(gaps, 'POST_OUTSIDE_APPROVED_PLAN',
                f'TX {x["transaction_id"]} {x.get("receipt_number") or "(no receipt)"} {f.fmt(f.number(x["quantity_lb"]))} by {x.get("operator_id")} is not a proved post of any signed group',
                None, x['product_id'], lot['lot_code'] if lot else None)
    # Owner answers signed with a group may not be changed in the final review.
    final_review = final_data.get('review') or {}
    for g in groups:
        for section, answers in (g['data'].get('review') or {}).items():
            for fp, answer in (answers or {}).items():
                later = (final_review.get(section) or {}).get(fp)
                if later is not None and later != answer:
                    gap(gaps, 'REVIEW_CHANGED', f'{section} answer {fp[:12]} differs from the one signed with this group', g['name'])
    return dict(nothing_to_reset=nothing, multi_area=multi, extra_posts=[x['line_id'] for x in extra], proved_line_ids=sorted(own))


def final_gaps(a, exit_code, gaps):
    if exit_code == 0:
        return
    for pid in sorted(set(a['products']) - set(a['coverage_complete'])):
        gap(gaps, 'COVERAGE_INCOMPLETE', 'all-location coverage not certified in the final inputs', None, pid)
    for pid, hs in a['holds'].items():
        gap(gaps, 'FINAL_HOLD', '; '.join(hs), None, int(pid) if pid.isdigit() else None)
    for lid, hs in a['lot_holds'].items():
        gap(gaps, 'FINAL_LOT_HOLD', f'lot {lid}: ' + '; '.join(hs))
    for sid, items in a['sheet_follow_up'].items():
        gap(gaps, 'SHEET_FOLLOW_UP', f'sheet {sid}: ' + '; '.join(i['hold'] for i in items))
    for item in a['move_follow_up']:
        gap(gaps, 'MOVE_FOLLOW_UP', f'{item.get("row_ref")}: {item.get("detail")}')
    for item in a['general']:
        gap(gaps, 'GENERAL', item)
    for r in a['rows']:
        diff = f.number(r['expected_current_lb']) - f.number(r['current_lb'])
        if abs(diff) > f.TOLERANCE:
            gap(gaps, 'UNEXPLAINED_DIFFERENCE', f'expected {f.fmt(r["expected_current_lb"])} vs FL {f.fmt(r["current_lb"])} {r["ledger_unit"]} (difference {f.fmt(diff)})', None, r['product_id'], r['lot_code'])
    if not any(g['kind'].startswith(('FINAL', 'COVERAGE', 'SHEET', 'MOVE', 'GENERAL', 'UNEXPLAINED')) for g in gaps):
        gap(gaps, 'FINAL_NOT_RECONCILED', 'final whole-scope verification did not reach FULL RESET SCOPE RECONCILED; see final/reset-verification-v3.md')


def run(manifest_path, output, snapshot_path=None, target=None):
    m = load_manifest(manifest_path)
    check(target in (None, m['target']), 'Manifest target differs from --target')
    output = f.local_path(output)
    inputs = [m['path'], *m['final_inputs'].values()] + [p for g in m['groups'] for p in (g['approved_preview'], g['approval'], g['journal']) if p]
    check(not any(p.is_relative_to(output) for p in inputs), 'Keep the manifest and signed inputs outside the output directory')
    output.mkdir(parents=True, exist_ok=True)
    if snapshot_path:
        s = json.loads(f.local_path(snapshot_path).read_text(), parse_float=D)
        check(s.get('target', m['target']) == m['target'], 'Saved snapshot was read from another target')
    else:
        s = v.snapshot(m['target'])
    s['target'] = m['target']
    gaps = []
    for kind, key in (('previews', 'approved_preview'), ('journals', 'journal')):
        paths = [g[key] for g in m['groups'] if g[key]]
        if len(paths) != len(set(paths)):
            gap(gaps, 'DUPLICATE_GROUP_FILE', f'the same {kind[:-1]} is listed for more than one group')
    groups = [r for r in (prove_group(g, m['target'], s, gaps) for g in m['groups']) if r]
    shas = [g['preview_sha256'] for g in groups]
    if len(shas) != len(set(shas)):
        gap(gaps, 'DUPLICATE_GROUP_FILE', 'the same signed preview (by hash) is listed for more than one group')
    final_data = v.load_inputs(m['final_inputs']['count'], m['final_inputs']['coverage'], m['final_inputs']['moves'], m['final_inputs'].get('review'))
    own = {line['line_id'] for g in groups for line in g['completed'].values()}
    scope_ids = m.get('rehearsal_scope_product_ids')
    a = v.analyze(scoped_snapshot(s, scope_ids), final_data, own)
    if len(a['products']) != m['expected_reset_products']:
        gap(gaps, 'SCOPE_CHANGED', f'reset scope is {len(a["products"])} products, manifest expects {m["expected_reset_products"]}; reconcile the catalog change explicitly')
    final_exit = v.report(a, output / 'final', verify=True)
    final_gaps(a, final_exit, gaps)
    covered = union(groups, a, s, final_data, gaps)
    ok = not gaps
    verdict = VERDICT_OK if ok else f'WHOLE RESET NOT RECONCILED: {len(gaps)} gap(s)'
    summary = [dict(name=g['name'], areas=g['areas'], preview_sha256=g['preview_sha256'], journal=g['journal'], no_change=g['no_change'],
                    approved_products=g['approved_products'], approved_lots=len(g['approved']),
                    posts=sorted((x['receipt_number'] for x in g['completed'].values())), actor=g['actor'], owner=g['owner'])
               for g in groups]
    proof = dict(verdict=verdict, exit_code=0 if ok else 2, target=m['target'], snapshot_at=s['snapshot_at'],
                 manifest=str(m['path'].relative_to(f.OUT)), manifest_sha256=v.sha(m['path'].read_bytes()),
                 reset_scope_products=len(a['products']), expected_reset_products=m['expected_reset_products'],
                 rehearsal_scope=scope_ids is not None, final_inputs={k: dict(path=str(p.relative_to(f.OUT)), sha256=v.sha(p.read_bytes())) for k, p in m['final_inputs'].items()},
                 final_verification_exit=final_exit, groups=summary, products_nothing_to_reset=covered['nothing_to_reset'],
                 multi_area_lots=covered['multi_area'], proved_line_ids=covered['proved_line_ids'], gaps=gaps)
    (output / 'whole-reset-proof.json').write_text(json.dumps(proof, indent=2, default=str) + '\n')
    with (output / 'whole-reset-gaps.csv').open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=['kind', 'group', 'product_id', 'lot_code', 'detail'])
        w.writeheader()
        w.writerows(gaps)
    lines = ['# Whole-reset proof', '',
             f'**{verdict}.** ' + ('Every reset-scope product is approved by exactly one signed area group, every count observation sits inside that group\'s signed plan, every posted reset is proved by its own group\'s journal, no reset post exists outside a signed plan, and the final whole-scope verification reconciles with zero unexplained differences.'
                                   if ok else 'This is not a release sign-off. Each gap below names the group, product and lot; fix the cause and rerun. Do not edit signed files or concatenate journals.'), '',
             f'Target **{m["target"]}**, one read-only snapshot at {s["snapshot_at"]}. Reset scope {len(a["products"])} products (manifest expects {m["expected_reset_products"]})'
             + ('; **staging rehearsal scope**, not the production catalog.' if scope_ids is not None else '.'), '',
             '## Area groups', '']
    lines += f.table(['Group', 'Areas', 'Preview SHA-256', 'Products', 'Posts (receipts)', 'Journal'],
                     [[g['name'], ', '.join(g['areas'] or ['(all)']), g['preview_sha256'][:16] + '…', len(g['approved_products']),
                       ', '.join(g['posts']) or '-', 'none (no-change group)' if g['no_change'] else (g['journal'] or {}).get('path', 'NOT PROVED')]
                      for g in summary])
    lines += ['', '## Multi-area lots (handled once)', '']
    lines += f.table(['Product / lot', 'Areas', 'Reset by', 'Held in'],
                     [[f'{x["product_id"]} / {x["lot_code"]}', ', '.join(x['areas']), x['handled_by'], ', '.join(x['held_in']) or '-'] for x in covered['multi_area']]) if covered['multi_area'] else ['None.']
    lines += ['', f'Products with no counted or on-hand lots (coverage-certified, nothing to reset): {len(covered["nothing_to_reset"])}.',
              '', '## Gaps', '']
    lines += f.table(['Kind', 'Group', 'Product', 'Lot', 'Detail'], [[x['kind'], x['group'], x['product_id'], x['lot_code'], x['detail']] for x in gaps]) if gaps else ['None.']
    lines += ['', 'Final whole-scope verification: `final/reset-verification-v3.md` (exit ' + str(final_exit) + '). Machine-readable: `whole-reset-proof.json`, `whole-reset-gaps.csv`.']
    (output / 'whole-reset-proof.md').write_text('\n'.join(lines) + '\n')
    print(verdict + '.')
    for x in gaps:
        print(f'- {x["kind"]}: ' + ' '.join(str(y) for y in (x['group'], x['product_id'], x['lot_code']) if y != '') + ' — ' + x['detail'])
    print(output / 'whole-reset-proof.md')
    return 0 if ok else 2


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--whole-reset', required=True, metavar='MANIFEST', help='JSON manifest of every area group (see apply-guide.md)')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--snapshot', help='Saved read-only evidence; omitted means one fresh read-only snapshot')
    p.add_argument('--target', choices=sorted(f.TARGETS), help='Optional; must equal the manifest target')
    args = p.parse_args(argv)
    return run(args.whole_reset, args.output_dir, args.snapshot, args.target)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, OSError) as e:
        raise SystemExit(str(e))
