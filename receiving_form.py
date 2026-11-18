"""Receiving Form rows and BOL photos, linked to FL receipts (migration 095;
docs/deployments/trial-recall.md).

The receiving Form's answers (Google Sheet "Receiving# (Responses)", tab
"Form Responses 1") and the BOL photos it links to in Google Drive are
INDEPENDENT EVIDENCE: this module reads them and never changes them.

Nightly (`sync`, run by scripts/receiving_form_sync.py):

  * read the tab, save each row as it was seen (a changed row is saved again as
    a new version; nothing is ever updated);
  * match each row to its FL receipt by DATE + INTERNAL LOT + PRODUCT. A clear
    match stores a link record (Form row + Drive file id) for the receipt;
  * anything else is never guessed: it goes on the owner's list
    "Receiving form row with no matching FL receipt", with the reason.

On a receipt page, "BOL photo" streams the linked photo from Drive through the
API to a signed-in, named person. No copy of a photo is stored anywhere.

READ-ONLY toward the ledger. The only statements that write are INSERTs into
three tables of migration 095 (receiving_form_sync_runs, receiving_form_rows,
receipt_bol_links), all append-only; a test pins that. Feature-module rule:
never imports main; routes reach it via `api`.
"""
from datetime import date, datetime, timedelta
import hashlib
import json
import os
import re
from zoneinfo import ZoneInfo

from fastapi import Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from psycopg2.extras import Json

import google_evidence
import permissions
import resolution

PLANT_TIMEZONE = ZoneInfo('America/New_York')   # = main.PLANT_TIMEZONE
MARKER = '095_trial_recall'
SHEET_ID = '1_zHWbCa0SHlmqVFhI07qxcD9ea35T66iHMMDJIapXJs'
TAB = 'Form Responses 1'
# Form rows before this plant date are not matched or listed (same day as the
# trial recall cutoff). RECEIVING_FORM_START_DATE overrides it.
DEFAULT_START = date(2026, 11, 11)
# The Form is sometimes filled the evening of, or the morning after, the day the
# receipt is dated in FL. More than this many days apart is not a clear match.
DATE_TOLERANCE_DAYS = 1
KG_TO_LB = 2.2046226218
# What a BOL photo may be. Anything else is refused, never served.
PHOTO_TYPES = frozenset({'image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/heic', 'image/heif',
                         'application/pdf'})

PHOTO_ROUTES = frozenset({
    ('GET', '/receipts/{receipt_number}/bol-photos'),
    ('GET', '/receipts/{receipt_number}/bol-photos/{photo_id}/file'),
})
OWNER_ROUTES = frozenset({('GET', '/verification/receiving-form/unmatched')})
ROUTES = PHOTO_ROUTES | OWNER_ROUTES

# Seam for tests: what builds the read-only Google client.
client_factory = google_evidence.Client


class FormLayoutError(Exception):
    """The tab no longer has the columns this module knows. The job stops; it never guesses a column."""


# ---------------------------------------------------------------------------
# Reading a Form row. Pure functions: no database, no network.
# ---------------------------------------------------------------------------

def _header_key(text):
    return ' '.join((text or '').lower().split())


_SINGLE_COLUMNS = {
    'timestamp': lambda h: h == 'timestamp',
    'name': lambda h: h in ('name:', 'name'),
    'stickers': lambda h: 'stickers' in h,
    'lot': lambda h: 'what lot' in h,
    'truck': lambda h: h.startswith('truck condition'),
    'approved': lambda h: 'approved supplier' in h,
    'bol': lambda h: h.startswith('bol'),
    'carrier': lambda h: h.startswith('carrier'),
    'photo': lambda h: 'picture of bol' in h,
}
_REQUIRED_COLUMNS = ('timestamp', 'lot', 'bol', 'photo')
_TIMESTAMP_FORMATS = ('%m/%d/%Y %H:%M:%S', '%m/%d/%Y %H:%M', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S')
_SIZE = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*([a-zA-Z]*)\.?')
_UNITS = {'lb': 'lb', 'lbs': 'lb', 'pound': 'lb', 'pounds': 'lb', 'kg': 'kg', 'kgs': 'kg', 'kilo': 'kg', 'kilos': 'kg'}
_DRIVE_ID = re.compile(r'(?:[?&]id=|/d/)([A-Za-z0-9_-]{10,200})')


def columns(header):
    """Where each known column is, found by its heading (the tabs do not share one order)."""
    found, products = {}, []
    for index, cell in enumerate(header):
        key = _header_key(cell)
        if key.startswith('name of product'):
            products.append(index)
            continue
        for field, fits in _SINGLE_COLUMNS.items():
            if fits(key):
                if field in found:
                    raise FormLayoutError(f'The tab has two "{field}" columns.')
                found[field] = index
    missing = [field for field in _REQUIRED_COLUMNS if field not in found]
    if missing or not products:
        raise FormLayoutError('The tab is missing a column this check needs: '
                              + ', '.join(missing + ([] if products else ['product'])))
    return found, products


def _number(text):
    cleaned = (text or '').replace(',', '').strip()
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if value > 0 and value != float('inf') else None


def parse_product(text):
    """"Name-case size-cases" (e.g. "Sugar 6X-50lbs-225"), read from the right.

    total_lb is filled only when the case size states pounds or kilograms; a
    bare number ("25") is never assumed to be pounds.
    """
    text = ' '.join((text or '').split())
    entry = {'text': text, 'name': text, 'case_size': None, 'case_unit': None, 'cases': None, 'total_lb': None}
    parts = [part.strip() for part in text.rsplit('-', 2)]
    if len(parts) == 3 and parts[0]:
        size, cases = _SIZE.fullmatch(parts[1]), _number(parts[2])
        if size and cases is not None:
            entry.update(name=parts[0], case_size=_number(size.group(1)), cases=cases,
                         case_unit=_UNITS.get(size.group(2).lower(), size.group(2).lower() or None))
            if entry['case_size'] and entry['case_unit'] in ('lb', 'kg'):
                factor = KG_TO_LB if entry['case_unit'] == 'kg' else 1
                entry['total_lb'] = round(entry['case_size'] * cases * factor, 4)
    return entry


def parse_lots(text):
    """Every lot number typed in "What lot# did you use?", in order, once each."""
    lots = []
    for part in re.split(r'[,;/&+\n]|\s+(?:and|y)\s+|\s+', text or ''):
        part = part.strip()
        if part and part not in lots:
            lots.append(part)
    return lots


def lot_key(code):
    """Lot codes compare on letters and digits only (26-11-12-STAR-001 = 261112star001)."""
    return re.sub(r'[^A-Z0-9]', '', (code or '').upper())


def photo_ids(text):
    ids = []
    for file_id in _DRIVE_ID.findall(text or ''):
        if file_id not in ids:
            ids.append(file_id)
    return ids


def _timestamp(text, zone):
    text = (text or '').strip()
    for layout in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(text, layout).replace(tzinfo=zone)
        except ValueError:
            continue
    return None


def parse_rows(values, time_zone=None):
    """Every answered row of the tab as plain data. Row 1 is the heading."""
    if not values:
        raise FormLayoutError('The tab is empty.')
    found, product_columns = columns(values[0])
    try:
        zone = ZoneInfo(time_zone) if time_zone else PLANT_TIMEZONE
    except Exception:  # an unknown zone name: the plant's own zone is the honest default
        zone = PLANT_TIMEZONE
    header, rows, seen = values[0], [], {}

    def cell(row, field):
        index = found.get(field)
        return row[index].strip() if index is not None and index < len(row) else ''

    for offset, row in enumerate(values[1:], start=2):
        if not any((value or '').strip() for value in row):
            continue
        stamp_text = cell(row, 'timestamp')
        stamp = _timestamp(stamp_text, zone)
        occurrence = seen[stamp_text] = seen.get(stamp_text, 0) + 1
        products = [parse_product(row[i]) for i in product_columns if i < len(row) and row[i].strip()]
        raw = {(header[i].strip() or f'column {i + 1}'): value for i, value in enumerate(row) if (value or '').strip()}
        rows.append({
            'row_number': offset,
            'row_key': f'{stamp_text}|{occurrence}' if stamp_text else f'no timestamp|row {offset}',
            'content_hash': hashlib.sha256(json.dumps([offset, row], ensure_ascii=False).encode()).hexdigest(),
            'timestamp_text': stamp_text,
            'submitted_at': stamp,
            'submitted_date': stamp.astimezone(PLANT_TIMEZONE).date() if stamp else None,
            'submitted_by': cell(row, 'name'), 'stickers': cell(row, 'stickers'),
            'lot_text': cell(row, 'lot'), 'lots': parse_lots(cell(row, 'lot')),
            'truck_condition': cell(row, 'truck'), 'approved_supplier': cell(row, 'approved'),
            'bol_number': cell(row, 'bol'), 'carrier': cell(row, 'carrier'),
            'photo_text': cell(row, 'photo'), 'photo_file_ids': photo_ids(cell(row, 'photo')),
            'products': products, 'raw': raw,
        })
    return rows


def current_rows(rows, start):
    """Rows dated on/after `start`, plus undated rows that sit below the first such row."""
    first = next((row['row_number'] for row in rows if row['submitted_date'] and row['submitted_date'] >= start), None)
    return [row for row in rows
            if (row['submitted_date'] and row['submitted_date'] >= start)
            or (row['submitted_date'] is None and first is not None and row['row_number'] > first)]


# ---------------------------------------------------------------------------
# Reading the ledger. Everything below this banner only SELECTs.
# ---------------------------------------------------------------------------

def migration_applied(cur):
    cur.execute('SELECT 1 FROM migration_markers WHERE name = %s', (MARKER,))
    return cur.fetchone() is not None


def _dicts(cur):
    names = [column[0] for column in cur.description]
    return [dict(row) if isinstance(row, dict) else dict(zip(names, row)) for row in cur.fetchall()]


def load_receipts(cur, since):
    """Posted FL receipts (current ledger) dated on/after `since`: one row per (receipt, lot)."""
    cur.execute("""
        SELECT t.id AS transaction_id, tx.receipt_number, tx.business_date, t.timestamp, t.bol_reference,
               t.shipper_name, tx.supplier_id, s.name AS supplier_name, tx.entered_by_actor_id, tx.operator_id,
               i.lot_id, l.lot_code, l.supplier_lot_code, l.identity_status, l.entry_source_notes,
               i.product_id, p.name AS product_name, p.type AS product_type, p.uom, SUM(i.quantity_lb) AS quantity
        FROM ledger_current_transactions t
        JOIN ledger_current_transaction_lines i ON i.transaction_id = t.id
        JOIN transactions tx ON tx.id = t.id
        LEFT JOIN suppliers s ON s.id = tx.supplier_id
        JOIN lots l ON l.id = i.lot_id
        JOIN products p ON p.id = i.product_id
        WHERE t.effective_status = 'posted' AND t.type = 'receive' AND i.quantity_lb > 0
          AND tx.business_date >= %s
        GROUP BY t.id, tx.receipt_number, tx.business_date, t.timestamp, t.bol_reference, t.shipper_name,
                 tx.supplier_id, s.name, tx.entered_by_actor_id, tx.operator_id, i.lot_id, l.lot_code,
                 l.supplier_lot_code, l.identity_status, l.entry_source_notes, i.product_id, p.name, p.type, p.uom
        ORDER BY t.id, i.lot_id""", (since,))
    receipts = _dicts(cur)
    for line in receipts:
        line['quantity'] = float(line['quantity'])
    return receipts


def receipt_index(receipts):
    index = {}
    for line in receipts:
        index.setdefault(lot_key(line['lot_code']), []).append(line)
    index.pop('', None)
    return index


def _name_key(text):
    return frozenset(re.findall(r'[a-z0-9]+', (text or '').lower()))


class ProductMatcher:
    """Does the product typed on the Form fit the product on the FL receipt?

    The receipt is already pinned by lot and date; this only confirms the
    product. It uses the shared FL resolver (names and learned aliases), inside
    a savepoint so a resolver error cannot spoil the caller's transaction:
      same_name          the two names use the same words
      resolver_match     the resolver names exactly that FL product for the Form wording
      resolver_candidate the Form wording is less specific ("Sugar") and that FL
                         product is one of the products it could mean
    Anything else is no fit (None).
    """

    def __init__(self, cur):
        self.cur, self.cache = cur, {}

    def lookup(self, text):
        if text not in self.cache:
            found = {'match_id': None, 'candidate_ids': frozenset()}
            self.cur.execute('SAVEPOINT receiving_form_resolve')
            try:
                result = resolution.resolve(self.cur, resolution.ResolveRequest(kind='product', query=text[:500]))
                found = {'match_id': (result.get('match') or {}).get('id'),
                         'candidate_ids': frozenset(c['id'] for c in result.get('candidates') or [])}
                self.cur.execute('RELEASE SAVEPOINT receiving_form_resolve')
            except Exception:  # an unreadable name is simply not a fit
                self.cur.execute('ROLLBACK TO SAVEPOINT receiving_form_resolve')
                self.cur.execute('RELEASE SAVEPOINT receiving_form_resolve')
            self.cache[text] = found
        return self.cache[text]

    def basis(self, form_name, product_id, product_name):
        if not (form_name or '').strip():
            return None
        if _name_key(form_name) == _name_key(product_name):
            return 'same_name'
        found = self.lookup(form_name.strip())
        if found['match_id'] == product_id:
            return 'resolver_match'
        if found['match_id'] is None and product_id in found['candidate_ids']:
            return 'resolver_candidate'
        return None


def _receipt_name(line):
    return line['receipt_number'] or f'entry #{line["transaction_id"]}'


def match_row(row, index, matcher, tolerance=DATE_TOLERANCE_DAYS):
    """One Form row against the FL receipts: date + internal lot + product.

    Returns {'status': matched | partly_matched | unmatched, 'links': [...], 'reasons': [...]}.
    A link is made only when exactly one FL receipt line fits a Form product.
    Nothing is ever picked among several.
    """
    def result(links, reasons):
        status = 'unmatched' if not links else ('matched' if not reasons else 'partly_matched')
        return {'status': status, 'links': links, 'reasons': reasons}

    if row['submitted_date'] is None:
        return result([], [f'The Form timestamp ("{row["timestamp_text"]}") could not be read.'])
    if not row['lots']:
        return result([], ['The Form row has no internal lot number.'])
    if not row['products']:
        return result([], ['The Form row names no product.'])
    near, far = [], []
    for lot in row['lots']:
        for line in index.get(lot_key(lot), []):
            days = (line['business_date'] - row['submitted_date']).days
            (near if abs(days) <= tolerance else far).append((line, days))
    if not near:
        if far:
            return result([], [f'Lot {line["lot_code"]} is on FL receipt {_receipt_name(line)} dated '
                               f'{line["business_date"].isoformat()}, {abs(days)} days from the Form date '
                               f'{row["submitted_date"].isoformat()}.' for line, days in far])
        return result([], ['No FL receipt uses lot ' + ', '.join(row['lots']) + '.'])
    reasons, fits = [], {}
    for line, days in near:
        fitted = False
        for position, entry in enumerate(row['products']):
            basis = matcher.basis(entry['name'], line['product_id'], line['product_name'])
            if basis:
                fits.setdefault(position, []).append((line, days, basis))
                fitted = True
        if not fitted:
            reasons.append(f'Lot and date match FL receipt {_receipt_name(line)}, but the product differs: the Form '
                           f'says {", ".join(e["text"] for e in row["products"])}; FL says {line["product_name"]}.')
    links, linked = [], set()
    for position, entry in enumerate(row['products']):
        options = fits.get(position, [])
        distinct = {(line['transaction_id'], line['lot_id']) for line, _, _ in options}
        if not options:
            reasons.append(f'No FL receipt was found for "{entry["text"]}".')
        elif len(distinct) > 1:
            reasons.append(f'More than one FL receipt fits "{entry["text"]}" ('
                           + ', '.join(sorted({_receipt_name(line) for line, _, _ in options})) + '). Not linked.')
        else:
            line, days, basis = options[0]
            if (line['transaction_id'], line['lot_id']) in linked:
                continue
            linked.add((line['transaction_id'], line['lot_id']))
            links.append({
                'transaction_id': line['transaction_id'], 'receipt_number': line['receipt_number'],
                'lot_id': line['lot_id'], 'lot_code': line['lot_code'], 'product_id': line['product_id'],
                'product_name': line['product_name'], 'business_date': line['business_date'].isoformat(),
                'quantity': line['quantity'], 'form_product': entry['text'],
                'basis': {'lot': 'same lot code', 'date_difference_days': days, 'product': basis}})
    return result(links, reasons)


def start_date(env=None):
    env = os.environ if env is None else env
    text = (env.get('RECEIVING_FORM_START_DATE') or '').strip()
    return date.fromisoformat(text) if text else DEFAULT_START


def describe_row(row, sheet_id=SHEET_ID, gid=None):
    """A Form row as JSON-able facts for a list or an email."""
    return {'row_number': row['row_number'], 'timestamp': row['timestamp_text'],
            'submitted_date': row['submitted_date'].isoformat() if row['submitted_date'] else None,
            'submitted_by': row['submitted_by'], 'lot_text': row['lot_text'], 'bol_number': row['bol_number'],
            'carrier': row['carrier'], 'products': [entry['text'] for entry in row['products']],
            'photo_file_ids': row['photo_file_ids'],
            'sheet_row_url': google_evidence.sheet_row_url(sheet_id, gid, row['row_number']) if gid is not None else None,
            'photo_urls': [google_evidence.drive_file_url(file_id) for file_id in row['photo_file_ids']]}


# ---------------------------------------------------------------------------
# The nightly sync. The ONLY writes in this module are the INSERTs below.
# ---------------------------------------------------------------------------

def _clock(cur):
    cur.execute('SELECT clock_timestamp() AS now')
    row = cur.fetchone()
    return row['now'] if isinstance(row, dict) else row[0]


def _one(cur):
    row = cur.fetchone()
    if row is None:
        return None
    return list(row.values())[0] if isinstance(row, dict) else row[0]


def sync(cur, sheet, *, env=None, sheet_id=SHEET_ID, tab=TAB, dry_run=False):
    """Save tonight's read of the Form tab, link the clear matches, list the rest.

    `sheet` is google_evidence.Client.sheet()'s answer, fetched by the caller
    before this transaction. dry_run=True reads and matches and writes nothing.
    The caller owns the transaction.
    """
    started_at = _clock(cur)
    start = start_date(env)
    rows = current_rows(parse_rows(sheet['values'], sheet.get('time_zone')), start)
    index = receipt_index(load_receipts(cur, start - timedelta(days=DATE_TOLERANCE_DAYS)))
    matcher = ProductMatcher(cur)
    rows_new = links_new = 0
    unmatched, linked_rows, current_links = [], 0, []
    for row in rows:
        outcome = match_row(row, index, matcher)
        if outcome['links']:
            linked_rows += 1
        if outcome['status'] != 'matched':
            unmatched.append(dict(describe_row(row, sheet_id, sheet.get('gid')), status=outcome['status'],
                                  reasons=outcome['reasons'],
                                  linked_receipts=sorted({link['receipt_number'] or f'#{link["transaction_id"]}'
                                                          for link in outcome['links']})))
        if dry_run:
            continue
        cur.execute("""INSERT INTO receiving_form_rows
                           (sheet_id, tab, row_key, row_number, content_hash, submitted_at, submitted_date,
                            submitted_by, lot_text, bol_number, carrier, photo_file_ids, products, raw)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                       ON CONFLICT (sheet_id, tab, row_key, content_hash) DO NOTHING RETURNING id""",
                    (sheet_id, tab, row['row_key'], row['row_number'], row['content_hash'], row['submitted_at'],
                     row['submitted_date'], row['submitted_by'], row['lot_text'], row['bol_number'], row['carrier'],
                     row['photo_file_ids'], Json(row['products']), Json(row['raw'])))
        form_row_id = _one(cur)
        if form_row_id is not None:
            rows_new += 1
        else:
            cur.execute("""SELECT id FROM receiving_form_rows
                           WHERE sheet_id = %s AND tab = %s AND row_key = %s AND content_hash = %s""",
                        (sheet_id, tab, row['row_key'], row['content_hash']))
            form_row_id = _one(cur)
        for link in outcome['links']:
            for file_id in row['photo_file_ids'] or ['']:
                cur.execute("""INSERT INTO receipt_bol_links
                                   (form_row_id, transaction_id, receipt_number, lot_id, product_id, drive_file_id,
                                    match_basis)
                               VALUES (%s, %s, %s, %s, %s, %s, %s)
                               ON CONFLICT (form_row_id, transaction_id, lot_id, drive_file_id) DO NOTHING
                               RETURNING id""",
                            (form_row_id, link['transaction_id'], link['receipt_number'], link['lot_id'],
                             link['product_id'], file_id, Json(dict(link['basis'], form_product=link['form_product']))))
                link_id = _one(cur)
                if link_id is not None:
                    links_new += 1
                else:
                    cur.execute("""SELECT id FROM receipt_bol_links
                                   WHERE form_row_id = %s AND transaction_id = %s AND lot_id = %s
                                     AND drive_file_id = %s""",
                                (form_row_id, link['transaction_id'], link['lot_id'], file_id))
                    link_id = _one(cur)
                current_links.append(link_id)
    # link_ids = the links tonight's read and matching stand behind. Only these are shown on a
    # receipt: a row edited back to an earlier wording, a row removed from the sheet, or a receipt
    # voided since, all drop out here without anything being updated or deleted.
    summary = {'start_date': start.isoformat(), 'date_tolerance_days': DATE_TOLERANCE_DAYS,
               'rows_linked': linked_rows, 'gid': sheet.get('gid'), 'unmatched': unmatched,
               'link_ids': current_links}
    outcome = {'run_id': None, 'rows_read': len(rows), 'rows_new': rows_new, 'links_new': links_new,
               'rows_linked': linked_rows, 'unmatched': unmatched, 'saved': not dry_run}
    if not dry_run:
        cur.execute("""INSERT INTO receiving_form_sync_runs
                           (started_at, job_status, sheet_id, tab, rows_read, rows_new, links_new, unmatched_count,
                            summary)
                       VALUES (%s, 'completed', %s, %s, %s, %s, %s, %s, %s) RETURNING id""",
                    (started_at, sheet_id, tab, len(rows), rows_new, links_new, len(unmatched), Json(summary)))
        outcome['run_id'] = _one(cur)
    return outcome


def record_failed_sync(cur, error_name, *, sheet_id=SHEET_ID, tab=TAB):
    """Best-effort note that tonight's read failed, so a missing list is never mistaken for an empty one."""
    cur.execute("""INSERT INTO receiving_form_sync_runs (started_at, job_status, job_error, sheet_id, tab)
                   VALUES (clock_timestamp(), 'failed', %s, %s, %s)""", (error_name[:200], sheet_id, tab))


def latest_sync(cur):
    """The newest read that worked, and whether a later one failed."""
    cur.execute("""SELECT id, started_at, finished_at, rows_read, rows_new, links_new, unmatched_count, summary
                   FROM receiving_form_sync_runs WHERE job_status = 'completed' ORDER BY id DESC LIMIT 1""")
    done = _dicts(cur)
    cur.execute("""SELECT id, started_at, job_error FROM receiving_form_sync_runs
                   WHERE job_status = 'failed' AND id > %s ORDER BY id DESC LIMIT 1""",
                (done[0]['id'] if done else 0,))
    failed = _dicts(cur)
    return (done[0] if done else None), (failed[0] if failed else None)


# A link counts only while the latest completed nightly read still stands behind it.
_CURRENT_LINK = """l.id IN (SELECT (jsonb_array_elements_text(s.summary -> 'link_ids'))::bigint
                            FROM receiving_form_sync_runs s
                            WHERE s.id = (SELECT max(id) FROM receiving_form_sync_runs
                                          WHERE job_status = 'completed'))"""


def photos_for_receipt(cur, receipt_number):
    """The linked BOL photos of one receipt, one entry per Drive file. [] until migration 095 exists."""
    if not migration_applied(cur):
        return []
    cur.execute("""
        SELECT l.id, l.drive_file_id, r.row_number, r.bol_number, r.submitted_date, r.carrier
        FROM receipt_bol_links l
        JOIN receiving_form_rows r ON r.id = l.form_row_id
        JOIN transactions t ON t.id = l.transaction_id
        WHERE t.receipt_number = %s AND l.drive_file_id <> '' AND """ + _CURRENT_LINK + """
        ORDER BY l.id""", (receipt_number,))
    photos, seen = [], set()
    for row in _dicts(cur):
        if row['drive_file_id'] in seen:
            continue
        seen.add(row['drive_file_id'])
        photos.append({'photo_id': row['id'], 'form_row_number': row['row_number'], 'bol_number': row['bol_number'],
                       'form_date': row['submitted_date'].isoformat() if row['submitted_date'] else None,
                       'carrier': row['carrier']})
    return photos


# ---------------------------------------------------------------------------
# Routes. Never in either GPT schema (include_in_schema=False).
# ---------------------------------------------------------------------------

def _fail(status, code, message):
    raise HTTPException(status_code=status, detail={'error_code': code, 'message': message})


def _named(api, request):
    actor = api._actor_identity(request)
    if actor['id'] is None or permissions.role_of(actor) not in permissions.PEOPLE:
        _fail(403, 'ROLE_NOT_ALLOWED', 'A BOL photo can only be opened by a signed-in, named person.')
    return actor


def _owner_or_office(api, request):
    role = permissions.role_of(api._actor_identity(request))
    if role not in {permissions.OWNER, permissions.SELECTED_OWNER, permissions.LEGACY_LEDGER, permissions.OFFICE}:
        _fail(403, 'ROLE_NOT_ALLOWED', 'This list is for the owner and the office.')


_GOOGLE_STATUS = {'NOT_CONFIGURED': 503, 'AUTH_FAILED': 503, 'SCOPE_TOO_WIDE': 503, 'UNAVAILABLE': 502,
                  'NOT_FOUND': 404, 'FORBIDDEN': 502, 'TOO_LARGE': 413, 'BAD_FILE_ID': 404}


def open_photo(file_id, env=None):
    """Check one Drive file is a servable BOL photo and open it as a stream: (mime type, chunks)."""
    env = os.environ if env is None else env
    client = client_factory()
    meta = client.file_metadata(file_id)
    if meta['trashed']:
        raise google_evidence.GoogleError('NOT_FOUND', 'The photo is in the Drive trash.')
    if meta['mime_type'] not in PHOTO_TYPES:
        raise google_evidence.GoogleError('NOT_FOUND', 'The linked Drive file is not a photo or a PDF.')
    if meta['size'] is not None and meta['size'] > google_evidence.MAX_PHOTO_BYTES:
        raise google_evidence.GoogleError('TOO_LARGE', 'The photo is larger than 15 MB.')
    folder = (env.get('RECEIVING_FORM_PHOTO_FOLDER_ID') or '').strip()
    if folder and folder not in meta['parents']:
        # Optional hardening: only files in the Form's own upload folder are ever served.
        raise google_evidence.GoogleError('NOT_FOUND', "The linked Drive file is not in the receiving Form's photo folder.")
    return meta['mime_type'], client.open_media(file_id)


def register_routes(app, api):
    @app.get('/receipts/{receipt_number}/bol-photos', include_in_schema=False)
    def receipt_bol_photos(receipt_number: str, request: Request, _: bool = Depends(api.verify_api_key)):
        """Which BOL photos (from the receiving Form) are linked to this receipt. Reads only."""
        _named(api, request)
        with api.get_transaction() as cur:
            return {'receipt_number': receipt_number, 'photos': photos_for_receipt(cur, receipt_number)}

    @app.get('/receipts/{receipt_number}/bol-photos/{photo_id}/file', include_in_schema=False)
    def receipt_bol_photo_file(receipt_number: str, photo_id: int, request: Request,
                               _: bool = Depends(api.verify_api_key)):
        """Stream one linked BOL photo from Google Drive. Nothing is stored."""
        _named(api, request)
        # A short transaction: the database connection is given back before Drive is called.
        with api.get_transaction() as cur:
            file_id = None
            if migration_applied(cur):
                cur.execute("""SELECT l.drive_file_id FROM receipt_bol_links l
                               JOIN transactions t ON t.id = l.transaction_id
                               WHERE l.id = %s AND t.receipt_number = %s AND l.drive_file_id <> ''
                                 AND """ + _CURRENT_LINK,
                            (photo_id, receipt_number))
                file_id = _one(cur)
        if file_id is None:
            _fail(404, 'BOL_PHOTO_NOT_FOUND', 'No BOL photo is linked to this receipt under that number.')
        try:
            mime_type, chunks = open_photo(file_id)
        except google_evidence.GoogleError as exc:
            _fail(_GOOGLE_STATUS.get(exc.code, 502), 'BOL_PHOTO_' + exc.code, str(exc))
        return StreamingResponse(chunks, media_type=mime_type, headers={
            'Cache-Control': 'private, no-store', 'X-Content-Type-Options': 'nosniff',
            'Content-Disposition': 'inline'})

    @app.get('/verification/receiving-form/unmatched', include_in_schema=False)
    def receiving_form_unmatched(request: Request, _: bool = Depends(api.verify_api_key)):
        """The owner's list: Receiving form row with no matching FL receipt (from the latest nightly read)."""
        _owner_or_office(api, request)
        with api.get_transaction() as cur:
            if not migration_applied(cur):
                _fail(503, 'RECEIVING_FORM_UNAVAILABLE', 'The receiving Form check is not switched on yet.')
            done, failed = latest_sync(cur)
            if done is None:
                return {'title': 'Receiving form row with no matching FL receipt', 'sync': None,
                        'later_failure': failed, 'unmatched': []}
            summary = done.pop('summary') or {}
            return {'title': 'Receiving form row with no matching FL receipt', 'sync': done,
                    'later_failure': failed, 'unmatched': summary.get('unmatched', [])}
