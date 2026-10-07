"""Fresh-start preparation only. Fixed read-only SQL; no inventory write/API path."""
import csv
import hashlib
import json
import os
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
from pathlib import Path
from zoneinfo import ZoneInfo
from lot_review import lot_suggestions

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'audits/fresh-start'
WRAPPER = ROOT / 'scripts/psql_ro.sh'
CREDENTIAL = Path.home() / '.config/factory-ledger/db_url'
PLANT = ZoneInfo('America/New_York')
D = Decimal
ZERO = D('0')
TOLERANCE = D('0.0001')  # FL BALANCE_EPSILON, in pounds.
ADJUSTMENT_QUANTUM = D('0.0001')
INACTIVE_HOLD = 'HELD – inactive in FL, owner decision needed'
COMPLETE = 'PRODUCT COUNT COMPLETE'
# Owner decisions recorded in package-decisions.md; never written to the catalog.
APPROVED_CASE_WEIGHTS = {pid: D('2.625') for pid in (150, 151, 152, 153)}
EXPECTED_ZERO_PRODUCTS = frozenset((183, 184, 185, 186, 285, 288))
RESET_SCOPE_POLICY = 'v3-packaging-information-only-2026-10-06'
BILLING_RESET_EXCLUDED_IDS = frozenset((102, 176))


def reset_exclusion_reason(product):
    """Owner decision: floor information is not authorization to reset supplies."""
    if product['id'] in BILLING_RESET_EXCLUDED_IDS:
        return 'Billing item; excluded from the reset by owner decision.'
    if product.get('type') == 'packaging':
        return 'Packaging is information only; FL does not deduct packaging on pack.'
    return ''


EXCLUDED_PRODUCT_IDS = frozenset((171, 209))
SCOPE_NOTE = 'Owner scope exclusion: products 171 and 209 are not counted, adjusted or required for full-scope sign-off.'
COLUMNS = ['area', 'product id', 'SKU', 'name', 'lot code', 'count unit',
           'counted qty', 'partial cases', 'notes']
V2_COLUMNS = ['paper row', 'area', 'product id', 'SKU', 'name', 'lot code',
              'lot date', 'count unit', 'counted qty', 'partial cases', 'notes', 'product checked']
CHECKED_LABEL = '☑ Product checked – complete'
UNCHECKED_LABEL = '☐ Product checked – complete'
LATE_COLUMNS = ['cutoff', 'transaction id', 'line id', 'product id', 'lot code',
                'physical time', 'entry time', 'qty lb', 'posted status', 'fingerprint',
                'happened after count', 'owner initials', 'notes']


def opening_balance_reason(cutoff):
    """The sole executable definition of the exact opening-balance reason."""
    return f"OPENING BALANCE {cutoff.astimezone(PLANT).date().isoformat()} – physical count"

# No supplied SQL, identifiers, filenames, cutoff, or CSV values enter this query.
# Use effective status, not raw transactions.status, and keep even nonposted rows
# only to identify recent activity. Balance calculations below use posted only.
SQL = """
BEGIN;
SET TRANSACTION READ ONLY;
SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;
WITH p AS (
 SELECT id,odoo_code,name,type,active,is_service,parent_batch_product_id,
        case_size_lb,default_case_weight_lb,pack_format
 FROM products WHERE (type='finished' OR (type='batch' AND name ILIKE '%granola%'))
 AND id NOT IN (__RESET_EXCLUDED_IDS__)
), x AS (
 SELECT l.id line_id,l.transaction_id,l.product_id,l.lot_id,l.quantity_lb,
        l.created_at line_created_at,l.created_at_source line_created_at_source,
        l.latest_correction_created_at line_correction_at,
        t.type,t.occurred_at,t.business_date,t.created_at transaction_created_at,
        t.created_at_source transaction_created_at_source,t.effective_status,
        t.latest_correction_created_at transaction_correction_at,
        t.adjust_reason,t.notes,t.operator_id
 FROM ledger_current_transaction_lines l
 JOIN ledger_current_transactions t ON t.id=l.transaction_id
 JOIN p ON p.id=l.product_id
), c AS (
 SELECT c.id,c.target_table,c.target_id,c.event_type,c.created_at,c.operator_id,c.reason
 FROM ledger_corrections c
 WHERE (c.target_table='transactions' AND c.target_id IN (SELECT transaction_id FROM x))
    OR (c.target_table='transaction_lines' AND c.target_id IN
        (SELECT id FROM ledger_current_transaction_lines
         WHERE transaction_id IN (SELECT transaction_id FROM x)))
)
SELECT json_build_object(
 'snapshot_at',current_timestamp,
 'products',COALESCE((SELECT json_agg(p ORDER BY id) FROM p),'[]'::json),
 'lots',COALESCE((SELECT json_agg(q ORDER BY product_id,id) FROM
   (SELECT l.id,l.product_id,l.lot_code,l.status,l.merged_into_lot_id,
           l.received_at, l.created_at AT TIME ZONE 'America/New_York' AS lot_created_at,
           l.created_at_source AS lot_created_at_source
    FROM lots l JOIN p ON p.id=l.product_id)q),'[]'::json),
 'lines',COALESCE((SELECT json_agg(x ORDER BY line_id) FROM x),'[]'::json),
 'corrections',COALESCE((SELECT json_agg(c ORDER BY created_at,id) FROM c),'[]'::json));
ROLLBACK;
""".replace("__RESET_EXCLUDED_IDS__", ", ".join(str(pid) for pid in sorted(EXCLUDED_PRODUCT_IDS)))


def scoped_snapshot(s):
    """Apply owner exclusions to live or saved snapshots without mutating input."""
    result = dict(s)
    result['products'] = [p for p in s['products'] if p['id'] not in EXCLUDED_PRODUCT_IDS]
    result['lots'] = [lot for lot in s['lots'] if lot['product_id'] not in EXCLUDED_PRODUCT_IDS]
    result['lines'] = [line for line in s['lines'] if line['product_id'] not in EXCLUDED_PRODUCT_IDS]
    removed = [line for line in s['lines'] if line['product_id'] in EXCLUDED_PRODUCT_IDS]
    removed_lines = {line['line_id'] for line in removed}
    removed_tx = {line['transaction_id'] for line in removed} - {line['transaction_id'] for line in result['lines']}
    result['corrections'] = [c for c in s['corrections'] if not (
        (c['target_table'] == 'transactions' and c['target_id'] in removed_tx) or
        (c['target_table'] == 'transaction_lines' and c['target_id'] in removed_lines))]
    return result


def number(value):
    try:
        value = D(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError('Quantities must be plain numbers (no commas, fractions, or units).') from None
    if not value.is_finite():
        raise ValueError('Quantities must be finite numbers.')
    return value


def round_adjustment(value):
    """Single planning rule: nearest 0.0001 lb, exact halfway ties to even.

    Ties-to-even also makes a half-quantum residual round to zero on rerun.
    Keep the unrounded calculation separately for the owner and verifier.
    """
    return number(value).quantize(ADJUSTMENT_QUANTUM, rounding=ROUND_HALF_EVEN)


def fmt(value):
    if value is None:
        return ''
    if isinstance(value, Decimal):
        if not value:
            return '0'
        return format(value, 'f').rstrip('0').rstrip('.') if '.' in format(value, 'f') else str(value)
    return str(value)


def cell(value):
    return fmt(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def stamp(value):
    if not value:
        return None
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError('Timestamp needs an explicit UTC offset, for example 2026-10-05T16:00:00-04:00.')
    return result


def local_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(OUT.resolve()):
        raise ValueError('Count inputs and report outputs must stay under audits/fresh-start/.')
    return path


def snapshot():
    if Path(__file__).resolve().parent != OUT or not WRAPPER.is_file():
        raise RuntimeError('Run this tool from a repository checkout containing scripts/psql_ro.sh.')
    # Always override any ambient DATABASE_URL; never fall back to .env.
    url = CREDENTIAL.read_text().strip()
    if not url or '\n' in url:
        raise RuntimeError('Configured database URL is missing or invalid (value suppressed).')
    env = dict(os.environ, DATABASE_URL=url, PGCONNECT_TIMEOUT='15')
    env.pop('PGOPTIONS', None)  # No session-level GUCs on shared pool connections.
    result = subprocess.run([str(WRAPPER), '-X', '-qAt', '-v', 'ON_ERROR_STOP=1'],
                            input=SQL, text=True, capture_output=True,
                            cwd=ROOT, env=env, timeout=120)
    if result.returncode:
        raise RuntimeError('Read-only psql query failed; database details and credentials suppressed.')
    try:
        return json.loads(result.stdout, parse_float=D)
    except (ValueError, TypeError):
        raise RuntimeError('Read-only query did not return one valid snapshot.') from None


def case_weight(p):
    if p['type'] == 'batch':
        return D('1'), ''
    if p['id'] in EXPECTED_ZERO_PRODUCTS:
        return None, package_note(p)
    if p['id'] in APPROVED_CASE_WEIGHTS:
        return APPROVED_CASE_WEIGHTS[p['id']], ''
    a = number(p['case_size_lb']) if p.get('case_size_lb') is not None else None
    b = number(p['default_case_weight_lb']) if p.get('default_case_weight_lb') is not None else None
    if a is not None and b is not None and a != b:
        return None, f'Conflicting case weights: case_size_lb={fmt(a)}, default_case_weight_lb={fmt(b)}. Owner decision needed.'
    value = a if a is not None else b
    if value is None or value <= 0:
        return None, 'No usable case weight. Owner must confirm physical package and pounds per case.'
    return value, ''


def package_note(p):
    if p['id'] in APPROVED_CASE_WEIGHTS:
        return 'BLUE STRIPES: owner-approved 6 × 7 oz case = 2.625 lb. Local override; catalog unchanged.'
    if p['id'] == 185:
        return ('EXPECTED 0 CASES. If found: record cases, packs per case and weight per pack '
                '(with its unit) in notes; HOLD for owner decision on case weight. '
                'QuickBooks name supplied by owner: Sunshine#9/Mini 100; last billed June 2026.')
    if p['id'] in EXPECTED_ZERO_PRODUCTS:
        return ('EXPECTED 0 CASES. Bulk granola is counted only as batch pounds. '
                'Any stock found here: HOLD for owner decision; no case conversion allowed.')
    return ''


def area(p):
    name = p['name'].lower()
    if p['type'] == 'batch':
        return '01 Granola bulk / batch containers'
    if p['id'] in (145, 146, 147, 148, 149):
        return '02 Finished goods / Sunshine pouches'
    if 'coconut' in name or 'desiccated' in name:
        return '03 Finished goods / coconut'
    if 'bs granola' in name or 'bs almond' in name or 'bs hazelnut' in name:
        return '04 Finished goods / Blue Stripes pouches'
    if p['id'] in EXPECTED_ZERO_PRODUCTS:
        return '05 Finished goods / Sunshine expected 0' + (' / INACTIVE' if not p['active'] else '')
    if 'granola ss' in name:
        return '05 Finished goods / other Sunshine (confirm packaging)'
    if 'granola' in name:
        return '06 Finished goods / other granola'
    return '07 Finished goods / other packaged goods'


def unit(p):
    return 'lb' if p['type'] == 'batch' else 'cases'


def entered_at(line):
    dates = [stamp(line.get(k)) for k in ('line_created_at', 'transaction_created_at')]
    return max(dates) if all(dates) else None


def is_reset(line, reason):
    return line['type'] == 'adjust' and line.get('adjust_reason') == reason


def partial_value(text):
    """Return a rational pair; multiply by case weight before dividing."""
    match = re.fullmatch(r'(\d+)\s*of\s*(\d+)', text.strip(), re.I)
    if match:
        n,d = map(D, match.groups())
        if d <= 0:
            raise ValueError('Partial cases need a positive denominator, for example 5 of 12.')
        return n,d
    value = number(text)
    if value < 0 or value >= 1:
        raise ValueError('Use X of Y for loose packs, or a decimal partial case below 1.')
    return value,D(1)


def completion_flags(text, *, checked_column=False):
    complete = none = found = False
    for part in re.split(r'[;/\n]', text or ''):
        part = part.strip()
        if part.upper() == COMPLETE:
            complete = True
            continue
        if not checked_column:
            continue  # Notes only accept the explicit legacy completion marker.
        if part.casefold() in ('yes','y','true','checked','☑','☒','✓','[x]'):
            complete = True
            continue
        if not re.match(r'^(?:☑|☒|✓|✅|\[x\])\s*', part, re.I):
            continue
        words = re.sub(r'^(?:☑|☒|✓|✅|\[x\])\s*', '', part, flags=re.I).casefold()
        if 'none found' in words:
            complete = none = True
        elif words.startswith('found'):
            found = True
        elif 'product checked' in words and 'complete' in words:
            complete = True
    return complete,none,found


def parse_sheet(path, products):
    present, complete, counts, issues = set(), set(), {}, defaultdict(list)
    completion_rows, none_found, found_flag = {}, set(), set()
    seen, references = set(), set()
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        version = 2 if reader.fieldnames == V2_COLUMNS else 1 if reader.fieldnames == COLUMNS else None
        if version is None:
            raise ValueError('CSV headers must match the issued v1 or v2 count sheet.')
        for rownum,row in enumerate(reader,2):
            if None in row or any(v is None for v in row.values()):
                raise ValueError(f'CSV row {rownum}: wrong number of fields.')
            raw_code = row['lot code']
            row = {k:v.strip() for k,v in row.items()}
            row['lot code'] = raw_code if raw_code.strip() else ''
            if not any(row.values()):
                continue
            try:
                pid = int(row['product id'])
            except ValueError:
                raise ValueError(f'CSV row {rownum}: product id must be a catalog ID.') from None
            if pid in EXCLUDED_PRODUCT_IDS:
                continue  # Old v1/v2 sheets may still contain these; never count or zero them.
            if pid not in products:
                raise ValueError(f'CSV row {rownum}: product {pid} is outside current reset scope.')
            p = products[pid];present.add(pid)
            ref = row.get('paper row') or f'legacy CSV row {rownum} (paper row unavailable)'
            if version == 2:
                if not re.fullmatch(rf'P{pid}-[A-Za-z0-9]+', ref):
                    raise ValueError(f'CSV row {rownum}: paper row must start P{pid}- and have a unique suffix.')
                if ref in references:
                    raise ValueError(f'Duplicate paper row {ref}. Give added rows unique references.')
                references.add(ref)
            if row['SKU'] != str(p.get('odoo_code') or '') or row['name'] != p['name']:
                issues[pid].append(f'{ref}: name/SKU differs from current catalog; recheck identity.')
            if row['count unit'] != unit(p):
                issues[pid].append(f'{ref}: count unit must be {unit(p)}.')
            done,none,found = completion_flags(row.get('product checked',''), checked_column=True)
            done = done or completion_flags(row['notes'])[0]
            if done:
                complete.add(pid);completion_rows.setdefault(pid,ref)
            if none:
                none_found.add(pid)
            if found:
                found_flag.add(pid)
            code,qty,partial = row['lot code'],row['counted qty'],row['partial cases']
            if code:
                duplicate = (pid,code.casefold())
                if duplicate in seen:
                    raise ValueError(f'{ref}: duplicate counted product/lot. Combine all locations onto one row.')
                seen.add(duplicate)
            if not code:
                if qty or partial:
                    issues[pid].append(f'{ref}: quantity without a lot; identify the lot before approval.')
                elif any(x.strip() and not completion_flags(x)[0] and not completion_flags(x)[2]
                         for x in row['notes'].split(';')):
                    issues[pid].append(f'{ref}: notes on an unidentified-lot row need owner review.')
                continue
            data = dict(code=code,qty=None,notes=row['notes'],paper_row=ref,csv_row=rownum,
                        whole_qty=None,partial_n=ZERO,partial_d=D(1))
            if not qty and not partial:
                counts[(pid,code)] = data
                continue
            total = number(qty) if qty else ZERO
            if total < 0:
                raise ValueError(f'{ref}: physical counts cannot be negative.')
            if p['type'] == 'batch':
                if partial:
                    raise ValueError(f'{ref}: batches use lb; leave partial cases blank.')
                n,d = ZERO,D(1)
            else:
                if total != total.to_integral_value():
                    raise ValueError(f'{ref}: put whole cases in counted qty and X of Y in partial cases.')
                n,d = partial_value(partial) if partial else (ZERO,D(1))
            data.update(qty=total+n/d,whole_qty=total,partial_n=n,partial_d=d)
            counts[(pid,code)] = data
            if pid in EXPECTED_ZERO_PRODUCTS and data['qty'] > 0:
                issues[pid].append(f'{ref}: stock found under an expected-zero product. '+package_note(p))
    for pid in none_found:
        if any(k[0] == pid and c['qty'] is not None and c['qty'] > 0 for k,c in counts.items()):
            issues[pid].append('“Checked, none found” conflicts with a positive counted row; correct the tick or the count.')
    for pid in found_flag:
        if not any(k[0] == pid and c['qty'] is not None and c['qty'] > 0 for k,c in counts.items()):
            issues[pid].append('“Found – write below” is checked but no positive lot count was entered.')
    return dict(version=version,present=present,complete=complete,counts=counts,
                issues=issues,completion_rows=completion_rows)


def late_fingerprint(line, cutoff, lot_code):
    data = {k:line.get(k) for k in ('transaction_id','line_id','product_id','lot_id','type',
            'occurred_at','line_created_at','transaction_created_at','line_created_at_source',
            'transaction_created_at_source','effective_status','adjust_reason','notes','operator_id',
            'line_correction_at','transaction_correction_at')}
    data.update(quantity_lb=fmt(number(line['quantity_lb'])),lot_code=lot_code,
                cutoff=cutoff.astimezone(timezone.utc).isoformat())
    return hashlib.sha256(json.dumps(data,sort_keys=True,default=str).encode()).hexdigest()


def read_confirmations(path):
    result = {}
    if path is None:
        return result
    with path.open(newline='',encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != LATE_COLUMNS:
            raise ValueError('Late-entry confirmation headers must match the generated review CSV.')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('Malformed late-entry confirmation row.')
            answer = row['happened after count'].strip().upper()
            if answer in ('','Y/N'):
                continue
            if answer not in ('Y','N'):
                raise ValueError('Each late-entry answer must be Y, N, or blank; no answer is inferred.')
            key = (int(row['transaction id']),int(row['line id']))
            if key in result:
                raise ValueError(f'Duplicate late-entry confirmation for TX/line {key}.')
            result[key] = dict(row,answer=answer)
    return result


def analyze(s, path, cutoff, confirmations_path=None):
    s = scoped_snapshot(s)
    at = stamp(s['snapshot_at'])
    if cutoff > at:
        raise ValueError('Cutoff cannot be in the future relative to the database snapshot.')
    reason = opening_balance_reason(cutoff)
    products = {p['id']:p for p in s['products']}
    parsed = parse_sheet(path,products)
    present,complete,counts = parsed['present'],parsed['complete'],parsed['counts']
    issues = parsed['issues'];data_issues = defaultdict(list)
    general,warnings = [],[]
    def data_issue(pid, category, detail, fix):
        data_issues[pid].append(dict(category=category,detail=detail,fix=fix))
        issues[pid].append(detail)
    manifest_path = OUT / ('scope-manifest-v2.json' if parsed['version']==2 else 'scope-manifest.json')
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest:
        original = {p['id']:p for p in manifest['products'] if p['id'] not in EXCLUDED_PRODUCT_IDS}
        for pid in original.keys()-products.keys():
            general.append(f'Issued-scope product {pid} disappeared or changed type: NOT COUNTED; never auto-zero.')
        for pid in products.keys()-original.keys():
            data_issue(pid,'scope','New in-scope product since sheet issue.','Add it to the physical count and reissue scope for review.')
        for pid in products.keys()&original.keys():
            if any(str(products[pid].get(k)) != str(original[pid].get(k)) for k in ('case_size_lb','default_case_weight_lb','type')):
                data_issue(pid,'catalog change','Case weight or product type changed since sheet issue.','Owner must review the changed package/type and count conversion.')
    else:
        general.append('Issued scope manifest missing; full scope cannot be certified.')
    lots = {l['id']:l for l in s['lots']}
    by_key,by_product,by_fold,events = defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
    balances = defaultdict(lambda:ZERO)
    for line in s['lines']:
        if line['effective_status']=='posted':
            balances[line['lot_id']] += number(line['quantity_lb'])
    for lot in s['lots']:
        by_key[(lot['product_id'],lot['lot_code'] or '')].append(lot)
        by_fold[(lot['product_id'],(lot['lot_code'] or '').casefold())].append(lot)
        by_product[lot['product_id']].append(lot)
    ignored_merged = []
    for lot in s['lots']:
        merged = lot.get('merged_into_lot_id') is not None or lot.get('status') == 'merged'
        if merged:
            if balances[lot['id']] == 0:
                ignored_merged.append(lot['id'])
            else:
                data_issue(lot['product_id'],'merged lot',f"Merged lot {lot['lot_code']} (ID {lot['id']}) still has {fmt(balances[lot['id']])} lb.",
                           'Owner must trace the merge and settle its remaining balance in a separately authorized task; do not count the alias twice.')
    for (pid,code),matches in by_fold.items():
        if len(matches)>1:
            data_issue(pid,'duplicate lot code',f"Code {code!r} matches multiple FL lot IDs: {', '.join(str(l['id']) for l in matches)}.",
                       'Compare labels and lot history; decide which identity is correct before any adjustment. Never auto-merge.')
    confirmations = read_confirmations(confirmations_path)
    late_entries,movements,unassigned,unconfirmed = [],[],[],defaultdict(list)
    after_flags = {}
    for line in s['lines']:
        pid,lid = line['product_id'],line['lot_id']
        occurred,entered = stamp(line.get('occurred_at')),entered_at(line)
        code = (lots.get(lid) or {}).get('lot_code') or 'MISSING LOT'
        key = (line['transaction_id'],line['line_id'])
        ordinary_after = bool(occurred and occurred>cutoff)
        answer = None
        if entered and entered>cutoff:
            fingerprint = late_fingerprint(line,cutoff,code)
            confirmation = confirmations.get(key)
            valid = bool(confirmation and confirmation['fingerprint']==fingerprint
                         and stamp(confirmation['cutoff'])==cutoff
                         and int(confirmation['product id'])==pid
                         and confirmation['lot code']==code
                         and stamp(confirmation['physical time'])==occurred
                         and stamp(confirmation['entry time'])==entered
                         and number(confirmation['qty lb'])==number(line['quantity_lb'])
                         and confirmation['posted status']==line['effective_status'])
            if valid:
                answer = confirmation['answer']
            else:
                unconfirmed[pid].append(key)
            possible = bool(occurred and occurred>cutoff and abs((occurred-entered).total_seconds())<=1800)
            entry = dict(line,lot_code=code,entered_at=entered.isoformat(),fingerprint=fingerprint,
                         confirmation=answer or '',owner_initials=confirmation['owner initials'] if valid else '',
                         confirmation_notes=confirmation['notes'] if valid else '',
                         review_flag='possibly not back-dated – check' if possible else '',
                         entry_time_quality='database-recorded' if line.get('line_created_at_source')=='database' and line.get('transaction_created_at_source')=='database' else 'entry time inferred/backfilled; actual typing time not proven',
                         confirmation_status='confirmed' if valid else 'STALE — reconfirm changed entry' if confirmation else 'UNCONFIRMED')
            late_entries.append(entry)
            if answer:
                ordinary_after = answer=='Y'
        # Opening adjustments are administrative count entries, never ordinary
        # physical movement; they still require the requested owner Y/N review.
        after_flags[line['line_id']] = ordinary_after and not is_reset(line,reason)
        if line['effective_status'] != 'posted':
            continue
        if lid not in lots or lots[lid]['product_id'] != pid:
            data_issue(pid,'lot identity',f"TX{line['transaction_id']} line {line['line_id']} has a missing/mismatched lot.",
                       'Identify the correct product/lot from paperwork; correction requires separate authorization.')
            unassigned.append(line)
        else:
            events[lid].append(line)
        if occurred is None or entered is None:
            data_issue(pid,'missing timestamp',f"TX{line['transaction_id']} line {line['line_id']} lacks physical or entry time.",
                       'Check the original paperwork and establish the real event/entry date before approval.')
        if occurred and occurred>at:
            data_issue(pid,'future timestamp',f"TX{line['transaction_id']} has a future physical time.",
                       'Owner must establish the actual time; do not assume a future-dated entry already happened.')
        if (entered and entered>cutoff) or (occurred and occurred>cutoff):
            kind = ('EXISTING_OPENING_ADJUSTMENT' if is_reset(line,reason) else
                    'AFTER_COUNT_MOVEMENT' if after_flags[line['line_id']] else 'PRE_COUNT_ENTRY — already covered by count')
            movements.append(dict(line,movement_kind=kind,entered_at=entered.isoformat() if entered else '',
                                  entered_after_cutoff=bool(entered and entered>cutoff),lot_code=code,
                                  confirmation=answer or 'unconfirmed' if entered and entered>cutoff else 'not required'))
    for pid,keys in unconfirmed.items():
        issues[pid].append(f'{len(keys)} entries typed after cutoff need owner confirmation: happened after count Y/N. No adjustment is proposed until confirmed.')
    corrections = [c for c in s['corrections'] if stamp(c['created_at'])>cutoff]
    if corrections:
        warnings.append(f'{len(corrections)} corrections on in-scope transactions were entered after cutoff; effective balances already include them. Warning only when counts reconcile.')
    rows = []
    for pid,p in sorted(products.items(),key=lambda x:(area(x[1]),x[1]['name'],x[0])):
        weight,weight_issue = case_weight(p)
        keys = {k for k in counts if k[0]==pid} | {(pid,l['lot_code'] or '') for l in by_product[pid]}
        if not keys:
            keys = {(pid,'')}
        for key in sorted(keys):
            matches = by_key.get(key,[]);counted = counts.get(key)
            explicit = bool(counted and counted['qty'] is not None)
            qty = counted['qty'] if counted else None
            paper_row = counted['paper_row'] if counted else ''
            r = dict(product_id=pid,product_active=p.get('active') is True,SKU=p.get('odoo_code') or '',name=p['name'],area=area(p),
                     count_unit=unit(p),case_weight_lb=weight if p['type']=='finished' else None,
                     package_decision=package_note(p),case_weight_source='owner-approved local override' if pid in APPROVED_CASE_WEIGHTS else 'owner hold: expected zero' if pid in EXPECTED_ZERO_PRODUCTS else 'catalog' if weight is not None else 'unresolved',
                     lot_id=None,lot_code=key[1],counted_qty=qty,counted_lb=None,fl_cutoff_lb=None,
                     late_pre_cutoff_lb=None,count_minus_cutoff_lb=None,existing_opening_lb=None,
                     adjustment_basis_lb=None,candidate_adjustment_lb=None,rounded_adjustment_lb=None,
                     rounding_delta_lb=None,planned_balance_lb=None,adjustment_lb=None,
                     post_cutoff_movement_lb=None,current_lb=None,expected_current_lb=None,unexplained_lb=None,
                     reason=reason,status='',detail='',inferred_zero=False,has_counted_row=explicit,
                     counted_for_verdict=explicit,paper_row=paper_row,suggestions='')
            if not matches and key[1]:
                if explicit:
                    suggestions = lot_suggestions(key[1],by_product[pid])
                    r['suggestions'] = '; '.join(f"{x['lot_code']} (FL lot {x['lot_id']}, similarity {x['similarity']:.0%})" for x in suggestions) or 'No FL lot codes exist for this product.'
                    r.update(status='FLOOR LOT NOT EXACTLY MATCHED — OWNER DECISION',detail='Suggested codes are not matched automatically. Recheck the paper/label, then explicitly correct the CSV if appropriate. '+package_note(p))
                    issues[pid].append(f"{paper_row}: counted code {key[1]!r} does not exactly match FL; owner must resolve it.")
                else:
                    r.update(status='UNCOUNTED PRINTED CODE NO LONGER IN FL',detail='No physical quantity entered; no adjustment proposed for this code.')
                rows.append(r);continue
            if len(matches)>1:
                r.update(status='AMBIGUOUS LOT — OWNER DECISION',detail='Duplicate FL identities; no adjustment calculated.')
                rows.append(r);continue
            lot = matches[0] if matches else None
            ls = events[lot['id']] if lot else []
            current = sum((number(x['quantity_lb']) for x in ls),ZERO)
            raw_cutoff = sum((number(x['quantity_lb']) for x in ls if stamp(x.get('occurred_at')) and stamp(x['occurred_at'])<=cutoff),ZERO)
            after = sum((number(x['quantity_lb']) for x in ls if after_flags[x['line_id']]),ZERO)
            opening = sum((number(x['quantity_lb']) for x in ls if is_reset(x,reason)),ZERO)
            late_before = sum((number(x['quantity_lb']) for x in ls if entered_at(x) and entered_at(x)>cutoff and not after_flags[x['line_id']] and not is_reset(x,reason)),ZERO)
            if qty is None and pid in complete and lot:
                qty = ZERO;r['inferred_zero'] = True
                paper_row = parsed['completion_rows'].get(pid,'') + ' (product checked; not found)'
            r.update(lot_id=lot['id'] if lot else None,lot_code=lot['lot_code'] if lot else key[1],counted_qty=qty,
                     paper_row=paper_row,current_lb=current,fl_cutoff_lb=raw_cutoff,late_pre_cutoff_lb=late_before,
                     existing_opening_lb=opening,post_cutoff_movement_lb=after,adjustment_basis_lb=current-after)
            merged_zero = bool(lot and lot['id'] in ignored_merged)
            if merged_zero and not (qty is not None and qty>0):
                r.update(status='ZERO-BALANCE MERGED LOT — IGNORED',detail='Historical merged identity; no hold or adjustment. Count only physical stock actually found.')
                if explicit:
                    r.update(counted_lb=ZERO,expected_current_lb=ZERO,unexplained_lb=current)
            elif pid not in present:
                r.update(status='NOT COUNTED — PRODUCT ABSENT',detail='Never auto-zero an absent product.')
            elif qty is None:
                r.update(status='COMPLETE — NO LOTS FOUND' if pid in complete and not lot else 'NOT COUNTED',
                         detail='No adjustment needed.' if pid in complete and not lot else 'Record only what you find, then tick the product completion box.')
            elif not lot or not lot.get('lot_code'):
                r.update(status='LOT IDENTITY — OWNER DECISION',detail='A unique nonempty FL lot code is required.')
            elif lot.get('merged_into_lot_id') is not None or lot.get('status') not in (None,'active'):
                r.update(status='LOT STATUS — OWNER DECISION',detail='Nonzero balance or physical stock under a merged/inactive lot requires investigation.')
                issues[pid].append(f"Lot {lot['lot_code']} has stock/balance under a merged or inactive identity.")
            elif qty>0 and pid in EXPECTED_ZERO_PRODUCTS:
                r.update(status='EXPECTED ZERO — STOCK FOUND — OWNER DECISION',detail=package_note(p))
            elif qty!=0 and weight is None:
                r.update(status='CASE WEIGHT — OWNER DECISION',detail=weight_issue)
                issues[pid].append(f"Lot {lot['lot_code']}: {weight_issue}")
            elif any(stamp(x.get('occurred_at')) is None or entered_at(x) is None for x in ls):
                r.update(status='TIMESTAMP — OWNER DECISION',detail='Cannot allocate stock reliably around cutoff.')
            else:
                if weight is None:
                    count_lb = ZERO
                elif explicit:
                    count_lb = counted['whole_qty']*weight + counted['partial_n']*weight/counted['partial_d']
                else:
                    count_lb = qty*weight
                change = count_lb-(current-after)
                rounded = round_adjustment(change)
                r.update(counted_lb=count_lb,count_minus_cutoff_lb=count_lb-raw_cutoff,candidate_adjustment_lb=change,
                         rounded_adjustment_lb=rounded,rounding_delta_lb=rounded-change,planned_balance_lb=current+rounded,
                         expected_current_lb=count_lb+after,unexplained_lb=current-(count_lb+after),
                         status='CALCULATED — OWNER REVIEW REQUIRED',
                         detail='Absent under completed product check; target zero.' if r['inferred_zero'] else '')
                if rounded and p.get('active') is not True:
                    r.update(status=INACTIVE_HOLD,detail=INACTIVE_HOLD)
                    issues[pid].append(INACTIVE_HOLD)
                elif not data_issues[pid] and not unconfirmed[pid]:
                    r['adjustment_lb'] = rounded
                else:
                    r['detail'] += ' Adjustment withheld while product data or late-entry confirmation is unresolved.'
                r['counted_for_verdict'] = explicit or (r['adjustment_lb'] is not None and r['adjustment_lb']!=0) or opening!=0
            rows.append(r)
    groups = {}
    for pid in products:
        prs = [r for r in rows if r['product_id']==pid]
        if INACTIVE_HOLD in issues[pid]:
            status = INACTIVE_HOLD
            for r in prs:
                r['adjustment_lb'] = None  # Entire inactive product is unapprovable.
        elif issues[pid] or any('DECISION' in r['status'] for r in prs):
            status = 'HOLD — resolve data, count or late-entry exceptions'
        elif pid not in present:
            status = 'NOT COUNTED — product absent from sheet'
        elif pid not in complete:
            status = 'INCOMPLETE — product not checked complete'
        else:
            status = 'READY FOR OWNER REVIEW — no approval assumed'
        groups[pid] = status
        for r in prs:
            r['group_status'] = status
    return dict(snapshot_at=s['snapshot_at'],cutoff=cutoff.isoformat(),date=cutoff.astimezone(PLANT).date().isoformat(),
                reason=reason,input_path=str(path),input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                sheet_version=parsed['version'],products=products,present=present,complete=complete,rows=rows,
                group_status=groups,issues={k:sorted(set(v)) for k,v in issues.items() if v},
                data_issues={k:v for k,v in data_issues.items() if v},general=general,warnings=warnings,
                movements=movements,late_entries=late_entries,unconfirmed_late=dict(unconfirmed),
                corrections=corrections,unassigned=unassigned,ignored_zero_merged=ignored_merged,
                confirmations_path=str(confirmations_path) if confirmations_path else '',
                confirmations_sha256=hashlib.sha256(confirmations_path.read_bytes()).hexdigest() if confirmations_path else '')

def table(headers, records):
    return ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |'] + [
        '| ' + ' | '.join(cell(v) for v in record) + ' |' for record in records]


def write_late_review(a, output):
    """Never overwrite the owner's filled confirmation file on a rerun."""
    path = output / f"late-entry-review-{a['date']}.csv"
    suffix = 1
    while path.exists():
        path = output / f"late-entry-review-{a['date']}-new-{suffix}.csv"
        suffix += 1
    rows = []
    for x in a['late_entries']:
        rows.append(dict(zip(LATE_COLUMNS,[a['cutoff'],x['transaction_id'],x['line_id'],x['product_id'],
                         x['lot_code'],x.get('occurred_at'),x['entered_at'],x['quantity_lb'],
                         x['effective_status'],x['fingerprint'],x['confirmation'],x['owner_initials'],
                         x['confirmation_notes']])))
    write_csv(path,LATE_COLUMNS,rows)
    return path


def late_review_section(a):
    pending = sum(len(v) for v in a['unconfirmed_late'].values())
    lines = ['## STOP — owner review of every entry typed after cutoff', '',
             f'**{len(a["late_entries"])} entries recorded after cutoff; {pending} unconfirmed. Products with unconfirmed entries stay HELD.**', '',
             'For each line confirm whether the physical movement happened after the count: Y or N. A timestamp close to typing time may mean older paperwork was entered without back-dating. A Y carries that movement forward; an N includes it in the opening count basis. These answers change this local calculation only. Opening-balance adjustments are administrative count entries, never ordinary physical movements; review those too (normally N). Voided entries are listed for review but never affect stock.', '',
             'Entry time is the later of transaction and line creation. Historical migration/backfill timestamps are labeled as estimates; actual typing time cannot be recovered from those alone. Corrections to existing entries are shown separately as warnings, not as second stock movements.', '']
    if not a['late_entries']:
        return lines+['No entries were recorded after this cutoff.','']
    lines += table(['TX / line','Product / lot','Physical time in FL','Entry time','lb / status','Review flag','Owner confirmation'],
                   [[f"{x['transaction_id']} / {x['line_id']}",f"{x['product_id']} / {x['lot_code']}",
                     x.get('occurred_at'),x['entered_at'],f"{fmt(x['quantity_lb'])} / {x['effective_status']}",
                     '; '.join(filter(None,[x['review_flag'],x['entry_time_quality'],x['confirmation_status']])),
                     'happened after count: '+(x['confirmation'] if x['confirmation'] else 'Y/N ______')]
                    for x in a['late_entries']])
    return lines+['']


def movement_section(a):
    lines = ['## Movements and entries after the cutoff', '',
             'These are explanations, not additional adjustments. Owner-confirmed Y/N answers classify entries typed after cutoff; otherwise classification is provisional and the product is held. Matching opening adjustments remain in current FL and never enter ordinary movement totals.', '']
    lines += table(['Kind', 'TX / line', 'Product / lot', 'Physical time', 'Entered time', 'Entered after cutoff?', 'lb', 'Actor'],
                   [[x['movement_kind'], f"{x['transaction_id']} / {x['line_id']}",
                     f"{x['product_id']} / {x['lot_code']}", x['occurred_at'], x['entered_at'],
                     x['entered_after_cutoff'], x['quantity_lb'], x['operator_id']] for x in a['movements']])
    if not a['movements']:
        lines += ['', 'None.']
    lines += ['', '## Corrections entered after cutoff', '',
              'These amendments/voids/restores are already applied by ledger_current. Do not add their values again. This report uses the current effective posted ledger, sliced by physical time; it does not reconstruct an old FL screen.', '']
    lines += table(['Correction', 'Target', 'Type', 'Entered', 'Actor', 'Reason'],
                   [[x['id'], f"{x['target_table']} / {x['target_id']}", x['event_type'],
                     x['created_at'], x['operator_id'], x['reason']] for x in a['corrections']])
    if not a['corrections']:
        lines += ['', 'None.']
    return lines


def run_arguments(description):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument('count_csv', help='Filled CSV under audits/fresh-start/.')
    p.add_argument('--cutoff', required=True, help='Physical count cutoff with offset, e.g. 2026-10-05T16:00:00-04:00.')
    p.add_argument('--output-dir', default=str(OUT), help='Optional subdirectory under audits/fresh-start/.')
    p.add_argument('--late-entry-confirmations', help='Owner-filled late-entry review CSV under audits/fresh-start/.')
    args = p.parse_args()
    path, output = local_path(args.count_csv), local_path(args.output_dir)
    if not path.is_file():
        raise ValueError('Count CSV does not exist.')
    cutoff = stamp(args.cutoff)
    if cutoff is None:
        raise ValueError('A cutoff timestamp is required.')
    confirmations = local_path(args.late_entry_confirmations) if args.late_entry_confirmations else None
    if confirmations and not confirmations.is_file():
        raise ValueError('Late-entry confirmation CSV does not exist.')
    return path, output, cutoff, confirmations


def write_csv(path, columns, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: fmt(row.get(k)) for k in columns})
