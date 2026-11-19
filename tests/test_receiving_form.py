"""Receiving Form rows and BOL photo links (receiving_form.py, google_evidence.py, migration 097).

The Form sheet and the Drive photos are independent evidence: these tests pin
that they are only read, that a link is stored only for a clear match (date +
internal lot + product), that everything else is listed and never guessed, and
that the job writes to nothing except the three receiving tables of 097.
"""
import ast
from datetime import date
import re

import psycopg2
import pytest
import yaml

import google_evidence
import main
import receiving_form
from scripts import exceptions_sweep, receiving_form_sync
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_trusted_browsers import migrated, people  # noqa: F401
from tests.trial_recall_kit import (DOWN, HEADER, MIGRATION, NEW_TABLES, PHOTO, PHOTO_2, ROOT, FakeGoogle, Plant,
                                    day, existing_tables, fingerprint, form_row, sheet, stamp)

# ───────────────────────── reading a Form row (no database) ─────────────────────────


def test_columns_are_found_by_heading_in_any_order():
    shuffled = ['NOTES', 'Timestamp', 'Name:', 'CARRIER/TRANSPORTADOR', 'What lot# did you use?', ' BOL#',
                'PICTURE OF BOL', 'Name of Product: Case Size: Number of Cases (example:coconut-50lbs-32)']
    rows = receiving_form.parse_rows([shuffled, ['', '2/3/2026 11:47:46', 'Arturo', 'Jack’s Eggs', '260203JA', '632192',
                                                 f'https://drive.google.com/open?id={PHOTO}', 'Sugar brown-50lbs-100']])
    row = rows[0]
    assert (row['row_number'], row['submitted_by'], row['carrier'], row['lot_text'], row['bol_number']) == (
        2, 'Arturo', 'Jack’s Eggs', '260203JA', '632192')
    assert row['photo_file_ids'] == [PHOTO] and row['submitted_date'] == date(2026, 2, 3)
    assert row['products'][0]['name'] == 'Sugar brown' and row['products'][0]['total_lb'] == 5000


@pytest.mark.parametrize('header', [
    [h for h in HEADER if 'What lot' not in h],                 # a needed column is gone
    [h for h in HEADER if not h.startswith('Name of Product')],  # no product column
    [*HEADER, 'Timestamp'],                                      # two Timestamp columns
    [],
])
def test_a_changed_layout_stops_the_job_instead_of_guessing_a_column(header):
    with pytest.raises(receiving_form.FormLayoutError):
        receiving_form.parse_rows([header, form_row(day(1), 'L1', ['Sugar-50lbs-1'])] if header else [])


@pytest.mark.parametrize('text,name,size,unit,cases,total', [
    ('Sugar 6X-50lbs-225', 'Sugar 6X', 50, 'lb', 225, 11250),
    ('Monk fruit-25kg-5', 'Monk fruit', 25, 'kg', 5, 275.5778),
    ('Chocolate 4M-25lbs-1,600', 'Chocolate 4M', 25, 'lb', 1600, 40000),
    ('Dark chocolate chips 70%-44lbs-50', 'Dark chocolate chips 70%', 44, 'lb', 50, 2200),
    ('Cacao beans hazalnut-25-120', 'Cacao beans hazalnut', 25, None, 120, None),   # no unit: never assumed pounds
    ('Almond flavor-33pcs-33', 'Almond flavor', 33, 'pcs', 33, None),
    ('3 liners roll', '3 liners roll', None, None, None, None),                     # not in the three-part form
    ('Drie mango cubes-5.00kg-231', 'Drie mango cubes', 5, 'kg', 231, 2546.3391),
])
def test_product_text_is_read_from_the_right(text, name, size, unit, cases, total):
    entry = receiving_form.parse_product(text)
    assert (entry['name'], entry['case_size'], entry['case_unit'], entry['cases'], entry['total_lb']) == (
        name, size, unit, cases, total)
    assert entry['text'] == text


def test_lots_photos_and_timestamps():
    assert receiving_form.parse_lots('26-11-12-STAR-001, 26-11-12-STAR-002 / 261112JA') == [
        '26-11-12-STAR-001', '26-11-12-STAR-002', '261112JA']
    assert receiving_form.lot_key('26-11-12-star-001') == receiving_form.lot_key('261112STAR001') == '261112STAR001'
    assert receiving_form.photo_ids(f'https://drive.google.com/open?id={PHOTO}, '
                                    f'https://drive.google.com/file/d/{PHOTO_2}/view') == [PHOTO, PHOTO_2]
    rows = receiving_form.parse_rows([HEADER, form_row(None, 'L1', ['Sugar-50lbs-1']),
                                      form_row(day(1), 'L2', ['Sugar-50lbs-1']), [], ['', '', 'not a date', 'x']],
                                     'America/New_York')
    assert [r['row_number'] for r in rows] == [2, 3, 5]                 # the blank row is skipped, numbering is the sheet's
    assert rows[0]['submitted_date'] is None and rows[2]['submitted_date'] is None
    assert rows[1]['submitted_at'].utcoffset().total_seconds() == -5 * 3600
    # Rows before the start date are left out; an undated row counts only when it sits below a current one.
    assert [r['row_number'] for r in receiving_form.current_rows(rows, day(0))] == [3, 5]
    assert receiving_form.current_rows(rows, day(5)) == []


def test_two_answers_in_the_same_second_stay_two_rows():
    rows = receiving_form.parse_rows([HEADER, form_row(day(1), 'L1', ['Sugar-50lbs-1']),
                                      form_row(day(1), 'L2', ['Honey-50lbs-1'])])
    assert len({r['row_key'] for r in rows}) == 2 and len({r['content_hash'] for r in rows}) == 2


# ───────────────────────── matching (no database) ─────────────────────────

class Names:
    """A stand-in for the product check: fits when the first word is the same."""

    def basis(self, form_name, product_id, product_name):
        return 'same_name' if form_name.split()[0].lower() == product_name.split()[0].lower() else None


def line(tx, lot_id, code, product, on, name=None, quantity=5000.0):
    return {'transaction_id': tx, 'receipt_number': f'RCV-{tx}', 'business_date': on, 'lot_id': lot_id,
            'lot_code': code, 'product_id': product, 'product_name': name or f'Sugar {product}', 'quantity': quantity}


def match(row_cells, *lines):
    row = receiving_form.parse_rows([HEADER, row_cells])[0]
    return receiving_form.match_row(row, receiving_form.receipt_index(list(lines)), Names())


def test_a_clear_match_links_date_lot_and_product():
    outcome = match(form_row(day(1), '26-11-12-star-001', ['Sugar brown-50lbs-100']),
                    line(7, 70, '26-11-12-STAR-001', 1, day(1)))
    assert outcome['status'] == 'matched' and outcome['reasons'] == []
    link = outcome['links'][0]
    assert (link['transaction_id'], link['lot_id'], link['product_id']) == (7, 70, 1)
    assert link['basis'] == {'lot': 'same lot code', 'date_difference_days': 0, 'product': 'same_name'}


def test_the_form_may_be_filled_one_day_off_but_not_two():
    one = match(form_row(day(2), 'L-1', ['Sugar-50lbs-100']), line(7, 70, 'L-1', 1, day(1)))
    assert one['status'] == 'matched' and one['links'][0]['basis']['date_difference_days'] == -1
    two = match(form_row(day(3), 'L-1', ['Sugar-50lbs-100']), line(7, 70, 'L-1', 1, day(1)))
    assert two['status'] == 'unmatched' and two['links'] == []
    assert 'RCV-7' in two['reasons'][0] and '2 days from the Form date' in two['reasons'][0]


def test_no_receipt_no_lot_no_product_no_timestamp_are_listed_with_the_reason():
    assert match(form_row(day(1), 'L-9', ['Sugar-50lbs-1']), line(7, 70, 'L-1', 1, day(1)))['reasons'] == [
        'No FL receipt uses lot L-9.']
    assert match(form_row(day(1), '', ['Sugar-50lbs-1']))['reasons'] == ['The Form row has no internal lot number.']
    assert match(form_row(day(1), 'L-1', []))['reasons'] == ['The Form row names no product.']
    assert 'could not be read' in match(form_row(None, 'L-1', ['Sugar-50lbs-1']))['reasons'][0]


def test_wording_that_does_not_confirm_the_product_links_generously_and_says_so():
    # Lot and date leave one FL receipt line and one Form product: linked, marked not_confirmed.
    outcome = match(form_row(day(1), 'L-1', ['Honey-50lbs-10']), line(7, 70, 'L-1', 1, day(1), 'Sugar Brown'))
    assert outcome['status'] == 'matched' and outcome['reasons'] == []
    assert outcome['links'][0]['transaction_id'] == 7 and outcome['links'][0]['basis']['product'] == 'not_confirmed'
    # Two receipt lines and two Form products that the wording cannot tell apart: never guessed.
    outcome = match(form_row(day(1), 'L-1, L-2', ['Miel-50lbs-10', 'Avena-50lbs-4']),
                    line(7, 70, 'L-1', 1, day(1), 'Sugar Brown'), line(8, 71, 'L-2', 2, day(1), 'Honey Clover'))
    assert outcome['status'] == 'unmatched' and outcome['links'] == []
    assert sum('cannot be told which product on the Form it is' in r for r in outcome['reasons']) == 2
    assert sum(r.startswith('No FL receipt was found for') for r in outcome['reasons']) == 2
    # One confident match plus exactly one left over on each side: the left-over pair is linked, marked.
    outcome = match(form_row(day(1), 'L-1, L-2', ['Sugar-50lbs-10', 'Miel-50lbs-4']),
                    line(7, 70, 'L-1', 1, day(1), 'Sugar 6X'), line(8, 71, 'L-2', 2, day(1), 'Honey Clover'))
    assert outcome['status'] == 'matched'
    assert sorted((l['transaction_id'], l['basis']['product']) for l in outcome['links']) == [
        (7, 'same_name'), (8, 'not_confirmed')]


def test_a_form_lot_without_its_sequence_number_is_the_same_lot_when_exactly_one_receipt_fits():
    assert receiving_form.short_lot_key('26-11-13-DUTC-001') == '261113DUTC' == receiving_form.short_lot_key('261113dutc7')
    assert receiving_form.short_lot_key('261113DUTC') is None and receiving_form.short_lot_key('26-11-13') is None
    assert receiving_form.short_lot_key('122023') is None                # a date alone has no sequence to drop
    assert receiving_form.short_lot_key('26-11-13-A1B2-001') == '261113A1B2'   # the last segment is the sequence
    assert receiving_form.short_lot_key('26-11-13') is None and receiving_form.short_lot_key('2026-11-13') is None
    # The Form says 261113DUTC; FL holds 26-11-13-DUTC-001.
    outcome = match(form_row(day(1), '261113DUTC', ['Sugar-3000lbs-1']), line(7, 70, '26-11-13-DUTC-001', 1, day(1)))
    assert outcome['status'] == 'matched' and outcome['links'][0]['transaction_id'] == 7
    assert outcome['links'][0]['basis'] == {'lot': 'lot code without its sequence number',
                                            'date_difference_days': 0, 'product': 'same_name'}
    # -001 and -002 of the same product that day: more than one fits, so nothing is linked.
    outcome = match(form_row(day(1), '261113DUTC', ['Sugar-3000lbs-1']),
                    line(7, 70, '26-11-13-DUTC-001', 1, day(1)), line(8, 71, '26-11-13-DUTC-002', 1, day(1)))
    assert outcome['links'] == [] and 'More than one FL receipt fits' in outcome['reasons'][0]
    # -001 sugar and -002 honey: the product tells them apart, each Form product gets its one receipt.
    outcome = match(form_row(day(1), '261113DUTC', ['Sugar-50lbs-10', 'Honey-60lbs-4']),
                    line(7, 70, '26-11-13-DUTC-001', 1, day(1), 'Sugar 6X'), line(8, 71, '26-11-13-DUTC-002', 2, day(1), 'Honey Clover'))
    assert outcome['status'] == 'matched' and sorted(l['transaction_id'] for l in outcome['links']) == [7, 8]
    # The exact lot code always wins over a sequence-less reading.
    outcome = match(form_row(day(1), '261113DUTC', ['Sugar-50lbs-10']),
                    line(7, 70, '261113DUTC', 1, day(1)), line(8, 71, '26-11-13-DUTC-001', 1, day(1)))
    assert [l['transaction_id'] for l in outcome['links']] == [7] and outcome['links'][0]['basis']['lot'] == 'same lot code'
    # A short lot AND wording that does not confirm the product: two leniencies, not linked.
    outcome = match(form_row(day(1), '261113DUTC', ['Miel-60lbs-4']), line(7, 70, '26-11-13-DUTC-001', 1, day(1), 'Sugar Brown'))
    assert outcome['status'] == 'unmatched' and outcome['links'] == []
    assert "is the Form's lot with a sequence number added" in outcome['reasons'][0]
    # The date still has to be within a day.
    outcome = match(form_row(day(4), '261113DUTC', ['Sugar-50lbs-10']), line(7, 70, '26-11-13-DUTC-001', 1, day(1)))
    assert outcome['links'] == [] and '3 days from the Form date' in outcome['reasons'][0]
    # Only a missing SEQUENCE is forgiven: a different supplier code is a different lot.
    outcome = match(form_row(day(1), '261113DUT', ['Sugar-50lbs-10']), line(7, 70, '26-11-13-DUTC-001', 1, day(1)))
    assert outcome['links'] == [] and outcome['reasons'] == ['No FL receipt uses lot 261113DUT.']


def test_a_lot_on_the_row_with_no_receipt_is_listed_even_when_another_lot_matches():
    outcome = match(form_row(day(1), 'L-1, L-9', ['Sugar-50lbs-10']), line(7, 70, 'L-1', 1, day(1)))
    assert outcome['status'] == 'partly_matched' and [l['transaction_id'] for l in outcome['links']] == [7]
    assert outcome['reasons'] == ['No FL receipt uses lot L-9.']
    # ...and when the other lot's receipt is dated too far from the Form.
    outcome = match(form_row(day(1), 'L-1, L-2', ['Sugar-50lbs-10']),
                    line(7, 70, 'L-1', 1, day(1)), line(8, 71, 'L-2', 1, day(5)))
    assert outcome['status'] == 'partly_matched' and len(outcome['links']) == 1
    assert 'Lot L-2 is on FL receipt RCV-8' in outcome['reasons'][0] and '4 days from the Form date' in outcome['reasons'][0]
    # Nothing matches at all: one line, as before.
    assert match(form_row(day(1), 'L-8, L-9', ['Sugar-50lbs-10']))['reasons'] == ['No FL receipt uses lot L-8, L-9.']


def test_two_receipts_that_both_fit_are_not_chosen_between():
    outcome = match(form_row(day(1), 'L-1', ['Sugar-50lbs-10']),
                    line(7, 70, 'L-1', 1, day(1)), line(8, 71, 'L-1', 2, day(1)))
    assert outcome['status'] == 'unmatched' and outcome['links'] == []
    assert 'More than one FL receipt fits' in outcome['reasons'][0] and 'RCV-7, RCV-8' in outcome['reasons'][0]


def test_a_truck_with_two_products_links_each_and_lists_the_one_with_no_receipt():
    outcome = match(form_row(day(1), 'L-1, L-2', ['Sugar-50lbs-10', 'Honey-50lbs-4', 'Oats-50lbs-2']),
                    line(7, 70, 'L-1', 1, day(1), 'Sugar 6X'), line(8, 71, 'L-2', 2, day(1), 'Honey Clover'))
    assert outcome['status'] == 'partly_matched'
    assert sorted((l['transaction_id'], l['form_product']) for l in outcome['links']) == [
        (7, 'Sugar-50lbs-10'), (8, 'Honey-50lbs-4')]
    assert outcome['reasons'] == ['No FL receipt was found for "Oats-50lbs-2".']


# ───────────────────────── Google: read-only by construction ─────────────────────────

class Http:
    """Records every request the Google client makes."""

    def __init__(self, scope=' '.join(sorted(google_evidence.SCOPES)), status=200):
        self.sent, self.scope, self.status = [], scope, status

    class Answer:
        def __init__(self, status, body):
            self.status_code, self._body = status, body

        def json(self):
            return self._body

    def post(self, url, data=None):
        self.sent.append(('POST', url))
        return self.Answer(self.status, {'access_token': 'at-' + str(len(self.sent)), 'expires_in': 3600,
                                         'scope': self.scope})

    def get(self, url, params=None, headers=None):
        self.sent.append(('GET', url))
        if '/values/' in url:
            return self.Answer(200, {'values': [HEADER, ['', '', '11/12/2026 9:00:00']]})
        if 'sheets.googleapis.com' in url:
            return self.Answer(200, {'properties': {'timeZone': 'America/New_York'},
                                     'sheets': [{'properties': {'sheetId': 77, 'title': receiving_form.TAB}}]})
        return self.Answer(200, {'id': PHOTO, 'name': 'bol.jpg', 'mimeType': 'image/jpeg', 'size': '1234',
                                 'parents': ['folder'], 'trashed': False})


ENV = {'GOOGLE_OAUTH_CLIENT_ID': 'id', 'GOOGLE_OAUTH_CLIENT_SECRET': 'secret', 'GOOGLE_OAUTH_REFRESH_TOKEN': 'refresh'}


def test_the_google_client_only_ever_reads():
    google_evidence._token_cache.clear()
    http = Http()
    client = google_evidence.Client(env=dict(ENV, GOOGLE_OAUTH_REFRESH_TOKEN='refresh-reads'), http=http)
    read = client.sheet(receiving_form.SHEET_ID, receiving_form.TAB)
    assert read['gid'] == 77 and read['time_zone'] == 'America/New_York' and read['values'][0] == HEADER
    assert client.file_metadata(PHOTO)['size'] == 1234
    # One POST, to the token endpoint only; everything sent to Sheets and Drive is a GET.
    assert [m for m, _ in http.sent].count('POST') == 1 and http.sent[0] == ('POST', google_evidence.TOKEN_URL)
    assert all(method == 'GET' for method, url in http.sent if url != google_evidence.TOKEN_URL)
    # Static guard: the module names no call that could change a sheet or a file.
    source = (ROOT / 'google_evidence.py').read_text()
    calls = {node.func.attr for node in ast.walk(ast.parse(source))
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not calls & {'put', 'patch', 'delete', 'request'}
    assert not re.search(r'batchUpdate|:append|:clear|/upload/|/permissions|/trash', source)
    assert google_evidence.SCOPES == {'https://www.googleapis.com/auth/drive.readonly',
                                      'https://www.googleapis.com/auth/spreadsheets.readonly'}


@pytest.mark.parametrize('scope', ['https://www.googleapis.com/auth/drive', '',
                                   'https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/spreadsheets'])
def test_a_sign_in_wider_than_read_only_is_refused(scope):
    google_evidence._token_cache.clear()
    client = google_evidence.Client(env=dict(ENV, GOOGLE_OAUTH_REFRESH_TOKEN='refresh-' + scope[-12:]), http=Http(scope))
    with pytest.raises(google_evidence.GoogleError) as refused:
        client.file_metadata(PHOTO)
    assert refused.value.code == 'SCOPE_TOO_WIDE'


def test_google_errors_never_carry_a_credential():
    google_evidence._token_cache.clear()
    with pytest.raises(google_evidence.GoogleError) as missing:
        google_evidence.Client(env={}, http=Http()).file_metadata(PHOTO)
    assert missing.value.code == 'NOT_CONFIGURED'
    client = google_evidence.Client(env=dict(ENV, GOOGLE_OAUTH_REFRESH_TOKEN='refresh-denied'), http=Http(status=400))
    with pytest.raises(google_evidence.GoogleError) as denied:
        client.file_metadata(PHOTO)
    assert denied.value.code == 'AUTH_FAILED' and 'secret' not in str(denied.value) and 'refresh' not in str(denied.value)
    with pytest.raises(google_evidence.GoogleError) as bad:
        client.file_metadata('../../etc/passwd')
    assert bad.value.code == 'BAD_FILE_ID'


def test_a_download_that_drops_part_way_is_a_google_error_not_a_crash():
    import httpx
    google_evidence._token_cache.clear()
    closed = []

    class Dropping:
        status_code = 200

        def iter_bytes(self):
            yield b'first-piece'
            raise httpx.ReadTimeout('timed out')

        def close(self):
            closed.append(True)

    class StreamHttp(Http):
        def build_request(self, method, url, params=None, headers=None):
            self.sent.append((method, url))
            return (method, url)

        def send(self, request, stream=False):
            return Dropping()

    client = google_evidence.Client(env=dict(ENV, GOOGLE_OAUTH_REFRESH_TOKEN='refresh-drop'), http=StreamHttp())
    chunks = client.open_media(PHOTO)
    assert next(chunks) == b'first-piece'
    with pytest.raises(google_evidence.GoogleError) as dropped:
        next(chunks)
    assert dropped.value.code == 'UNAVAILABLE' and closed == [True]
    with pytest.raises(google_evidence.GoogleError):                     # and so the whole read fails the same way
        client.download(PHOTO)


# ───────────────────────── the nightly sync (database) ─────────────────────────

@pytest.fixture
def plant(db_cursor):
    return Plant(db_cursor)


@pytest.mark.db
def test_sync_links_the_clear_match_and_lists_the_rest(plant):
    sugar = plant.product('Sugar Brown')
    honey = plant.product('Honey Clover')
    sugar_lot, sugar_tx = plant.receive(sugar, 5000, day(1))
    honey_lot, _ = plant.receive(honey, 600, day(1))
    read = sheet(
        plant.form_row(day(-3), sugar_lot, ['Sugar brown-50lbs-100']),                  # before the start date: ignored
        plant.form_row(day(1), sugar_lot, ['Sugar brown-50lbs-100'], photo=[PHOTO, PHOTO_2]),
        plant.form_row(day(1), honey_lot, ['Sugar brown-50lbs-12'], hour=12),           # right lot, other product wording
        plant.form_row(day(2), 'NO-SUCH-LOT', ['Sugar brown-50lbs-1'], hour=13),
    )
    outcome = receiving_form.sync(plant.cur, read)
    assert (outcome['rows_read'], outcome['rows_new'], outcome['rows_linked'], outcome['links_new']) == (3, 3, 2, 3)
    plant.cur.execute("""SELECT l.transaction_id, l.lot_id, l.product_id, l.drive_file_id, l.match_basis, r.row_number
                         FROM receipt_bol_links l JOIN receiving_form_rows r ON r.id = l.form_row_id
                         WHERE l.transaction_id = ANY(%s) ORDER BY l.id""", (list(plant.txs),))
    links = plant.cur.fetchall()
    assert [(l['lot_id'], l['product_id'], l['drive_file_id'], l['row_number'], l['match_basis']['product']) for l in links] == [
        (sugar_lot, sugar, PHOTO, 3, 'same_name'), (sugar_lot, sugar, PHOTO_2, 3, 'same_name'),
        (honey_lot, honey, PHOTO, 4, 'not_confirmed')]                  # linked by lot and date; the wording did not confirm it
    assert links[0]['transaction_id'] == sugar_tx and links[0]['match_basis']['date_difference_days'] == 0
    listed = {item['row_number']: item for item in outcome['unmatched']}
    assert sorted(listed) == [5] and listed[5]['reasons'] == ['No FL receipt uses lot NO-SUCH-LOT.']
    assert listed[5]['sheet_row_url'].endswith('#gid=1234&range=A5')
    unconfirmed, = outcome['product_wording_unconfirmed']
    assert unconfirmed['row_number'] == 4 and unconfirmed['linked'][0]['fl_product'] == plant.name('Honey Clover')
    plant.cur.execute('SELECT * FROM receiving_form_sync_runs WHERE id = %s', (outcome['run_id'],))
    run = plant.cur.fetchone()
    assert (run['job_status'], run['rows_read'], run['unmatched_count']) == ('completed', 3, 1)
    assert [item['row_number'] for item in run['summary']['unmatched']] == [5]
    assert [item['row_number'] for item in run['summary']['product_wording_unconfirmed']] == [4]


@pytest.mark.db
def test_sync_links_the_live_forms_short_lot_numbers(plant):
    """The lots as the real Form writes them (261113DUTC) against FL's full codes (26-11-13-DUTC-001)."""
    honey, sugar = plant.product('Honey Clover'), plant.product('Sugar Brown')
    tag = ''.join(chr(ord('G') + int(c, 16)) for c in plant.tag[2:6])    # four letters, like a real supplier code; this test's own
    honey_lot, honey_tx = plant.receive(honey, 3000, day(2), code=f'26-11-13-{tag}-001')
    plant.receive(sugar, 5000, day(5), code=f'26-11-16-{tag}-001')
    plant.receive(sugar, 2500, day(5), code=f'26-11-16-{tag}-002')       # a second sugar lot that day
    outcome = receiving_form.sync(plant.cur, sheet(
        plant.form_row(day(2), f'261113{tag}', ['Honey clover-3000lbs-1']),
        plant.form_row(day(5), f'261116{tag}', ['Sugar brown-50lbs-100'], hour=12)))
    assert outcome['rows_linked'] == 1 and outcome['links_new'] == 1
    plant.cur.execute('SELECT transaction_id, lot_id, match_basis FROM receipt_bol_links WHERE transaction_id = %s', (honey_tx,))
    link = plant.cur.fetchone()
    assert link['lot_id'] == honey_lot and link['match_basis']['lot'] == 'lot code without its sequence number'
    listed, = outcome['unmatched']                                       # two sugar receipts fit: not chosen between
    assert listed['row_number'] == 3 and 'More than one FL receipt fits' in listed['reasons'][0]


@pytest.mark.db
def test_a_second_night_adds_nothing_and_a_changed_row_is_saved_as_a_new_version(plant):
    sugar = plant.product('Sugar Brown')
    lot, _ = plant.receive(sugar, 5000, day(1))
    first = sheet(plant.form_row(day(1), lot, ['Sugar brown-50lbs-100']))
    assert receiving_form.sync(plant.cur, first)['links_new'] == 1
    again = receiving_form.sync(plant.cur, first)
    assert (again['rows_new'], again['links_new'], again['rows_linked']) == (0, 0, 1)
    edited = sheet(plant.form_row(day(1), lot, ['Sugar brown-50lbs-100'], bol='999'))    # someone edited the sheet
    third = receiving_form.sync(plant.cur, edited)
    assert (third['rows_new'], third['links_new']) == (1, 1)
    plant.cur.execute('SELECT bol_number FROM receiving_form_rows WHERE row_key = %s ORDER BY id',
                      (stamp(day(1)) + '|1',))
    assert [r['bol_number'] for r in plant.cur.fetchall()][-2:] == ['632054', '999']     # both versions are kept


@pytest.mark.db
def test_a_voided_receipt_is_not_a_match(plant):
    sugar = plant.product('Sugar Brown')
    lot, tx = plant.receive(sugar, 5000, day(1))
    plant.void(tx)
    outcome = receiving_form.sync(plant.cur, sheet(plant.form_row(day(1), lot, ['Sugar brown-50lbs-100'])))
    assert outcome['links_new'] == 0 and outcome['unmatched'][0]['reasons'] == [f'No FL receipt uses lot {plant.lots[lot]}.']


@pytest.mark.db
def test_the_product_check_uses_the_fl_names_and_never_picks_among_candidates(plant):
    brown, white = plant.product('Sugar Brown'), plant.product('Sugar 6X')
    honey = plant.product('Honey Clover')
    matcher = receiving_form.ProductMatcher(plant.cur)
    assert matcher.basis(plant.name('sugar BROWN'), brown, plant.name('Sugar Brown')) == 'same_name'
    # "<tag> Sugar" could be either sugar: it fits whichever one the receipt (found by lot and date) holds...
    assert matcher.basis(plant.name('Sugar'), brown, plant.name('Sugar Brown')) in ('resolver_candidate', 'resolver_match')
    assert matcher.basis(plant.name('Sugar'), white, plant.name('Sugar 6X')) in ('resolver_candidate', 'resolver_match')
    # ...and never a product it does not name.
    assert matcher.basis(plant.name('Sugar Brown'), honey, plant.name('Honey Clover')) is None
    assert matcher.basis('', brown, plant.name('Sugar Brown')) is None
    plant.cur.execute('SELECT 1 AS ok')                                 # the transaction is still usable
    assert plant.cur.fetchone()['ok'] == 1


@pytest.mark.db
def test_a_form_name_whose_words_are_all_in_the_fl_name_fits_and_a_thousands_comma_is_not_a_word_break(plant):
    assert receiving_form._name_key('Chocolate Chips – 4,000 CT') == {'chocolate', 'chips', '4000', 'ct'}
    assert receiving_form._name_key('1,5 kg, 4,00') == {'1', '5', 'kg', '4', '00'}      # only a real thousands comma
    chips = plant.product('Chocolate Chips – 4,000 CT')
    other = plant.product('Chocolate Chips – 1,000 CT')
    matcher = receiving_form.ProductMatcher(plant.cur)
    full = plant.name('Chocolate Chips – 4,000 CT')
    assert matcher.basis(plant.name('Chocolate Chips 4000'), chips, full) in (
        'words_in_name', 'resolver_match', 'resolver_candidate')
    assert matcher.basis(plant.name('Chocolate Chips 4000'), other, plant.name('Chocolate Chips – 1,000 CT')) is None
    assert matcher.basis('4000', chips, full) is None                    # a number alone names nothing
    # The live Form's row: short lot and the short product wording, one receipt that day.
    tag = ''.join(chr(ord('G') + int(c, 16)) for c in plant.tag[2:6])
    lot, tx = plant.receive(chips, 2000, day(5), code=f'26-11-16-{tag}-001')
    outcome = receiving_form.sync(plant.cur, sheet(plant.form_row(day(5), f'261116{tag}', ['Chocolate Chips 4000-25lbs-80'])),
                                  dry_run=True)
    assert outcome['rows_linked'] == 1 and outcome['unmatched'] == [] and outcome['product_wording_unconfirmed'] == []


@pytest.mark.db
def test_sync_changes_no_existing_table_and_dry_run_writes_nothing(plant):
    sugar = plant.product('Sugar Brown')
    lot, _ = plant.receive(sugar, 5000, day(1))
    read = sheet(plant.form_row(day(1), lot, ['Sugar brown-50lbs-100']), plant.form_row(day(2), 'X-1', ['Oats-50lbs-1']))
    existing = existing_tables(plant.cur)
    assert {'transactions', 'transaction_lines', 'ledger_corrections', 'lots', 'products', 'exceptions'} <= set(existing)
    before, mine = fingerprint(plant.cur, existing), fingerprint(plant.cur, NEW_TABLES)
    dry = receiving_form.sync(plant.cur, read, dry_run=True)
    assert dry['saved'] is False and dry['run_id'] is None and dry['rows_linked'] == 1 and len(dry['unmatched']) == 1
    assert fingerprint(plant.cur, NEW_TABLES) == mine                    # a dry run writes nowhere
    receiving_form.sync(plant.cur, read)
    assert fingerprint(plant.cur, NEW_TABLES) != mine
    assert fingerprint(plant.cur, existing) == before                    # no existing table changed by one byte
    # Static guard: the module's only write statements are INSERTs into the three receiving tables.
    source = (ROOT / 'receiving_form.py').read_text()
    sql = ' '.join(node.value for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.Constant) and isinstance(node.value, str))
    assert set(re.findall(r'INSERT\s+INTO\s+(\w+)', sql, re.I)) == {
        'receiving_form_sync_runs', 'receiving_form_rows', 'receipt_bol_links'}
    assert not re.search(r'\b(UPDATE\s+\w+\s+SET|DELETE\s+FROM|TRUNCATE|DROP\s+TABLE|ALTER\s+TABLE|FOR\s+UPDATE)\b', sql, re.I)


# ───────────────────────── migration 097 ─────────────────────────

@pytest.mark.db
def test_migration_097_is_additive_append_only_and_reruns(plant):
    cur = plant.cur
    cur.execute(MIGRATION)                                               # a second apply changes nothing
    cur.execute("SELECT count(*) AS n FROM migration_markers WHERE name = '097_trial_recall'")
    assert cur.fetchone()['n'] == 1
    cur.execute("""SELECT c.relname, c.relrowsecurity,
                          (SELECT count(*) FROM pg_policy p WHERE p.polrelid = c.oid) AS policies,
                          (SELECT count(*) FROM pg_trigger t WHERE t.tgrelid = c.oid
                             AND t.tgname = c.relname || '_append_only') AS guards
                   FROM pg_class c WHERE c.relname = ANY(%s) AND c.relkind = 'r'""", (list(NEW_TABLES),))
    rows = cur.fetchall()
    assert len(rows) == 4 and all(r['relrowsecurity'] and r['policies'] == 0 and r['guards'] == 1 for r in rows)
    # No foreign key reaches an existing table: the jobs never lock the ledger.
    cur.execute("""SELECT conrelid::regclass::text AS source, confrelid::regclass::text AS target FROM pg_constraint
                   WHERE contype = 'f' AND conrelid::regclass::text = ANY(%s)""", (list(NEW_TABLES),))
    assert {(r['source'], r['target']) for r in cur.fetchall()} == {('receipt_bol_links', 'receiving_form_rows')}
    sugar = plant.product('Sugar Brown')
    lot, _ = plant.receive(sugar, 5000, day(1))
    receiving_form.sync(cur, sheet(plant.form_row(day(1), lot, ['Sugar brown-50lbs-100'])))
    for statement in ("UPDATE receipt_bol_links SET drive_file_id = 'x'", 'DELETE FROM receiving_form_rows',
                      "UPDATE receiving_form_sync_runs SET job_status = 'failed'", 'DELETE FROM trial_recall_runs'):
        cur.execute('SAVEPOINT guard')
        if statement.startswith('DELETE FROM trial_recall_runs'):
            cur.execute("""INSERT INTO trial_recall_runs (started_at, plant_date, cutoff_date, job_status, job_error,
                                                          email_status)
                           VALUES (now(), current_date, current_date, 'failed', 'x', 'not_attempted')""")
        with pytest.raises(psycopg2.Error):
            cur.execute(statement)
        cur.execute('ROLLBACK TO SAVEPOINT guard')
    cur.execute('SAVEPOINT down')
    with pytest.raises(psycopg2.Error, match='holds rows'):               # the down file never discards saved rows
        cur.execute(DOWN)
    cur.execute('ROLLBACK TO SAVEPOINT down')
    for file in ('migrations/097_trial_recall.sql', 'migrations/down/097_trial_recall_down.sql'):
        text = (ROOT / file).read_text().lower()
        code = '\n'.join(l for l in text.splitlines() if not l.strip().startswith('--'))
        for table in ('transactions', 'transaction_lines', 'lots ', 'ledger_corrections', 'products'):
            assert not re.search(rf'\b(alter table|drop table|update|delete from|references)\s+{table}', code), (file, table)


# ───────────────────────── the routes ─────────────────────────

@pytest.fixture
def linked(client, plant, monkeypatch):
    # `client` first: the app's startup must finish before this test's rows hold their locks.
    sugar = plant.product('Sugar Brown')
    lot, tx = plant.receive(sugar, 5000, day(1))
    receiving_form.sync(plant.cur, sheet(plant.form_row(day(1), lot, ['Sugar brown-50lbs-100'], photo=[PHOTO, PHOTO_2])))
    plant.cur.execute('SELECT receipt_number FROM transactions WHERE id = %s', (tx,))
    google = FakeGoogle(files={PHOTO: {'mime_type': 'image/jpeg', 'size': 1000},
                               PHOTO_2: {'mime_type': 'text/html', 'size': 10}})
    monkeypatch.setattr(receiving_form, 'client_factory', lambda: google)
    return {'receipt': plant.cur.fetchone()['receipt_number'], 'google': google, 'plant': plant}


@pytest.mark.db
def test_bol_photo_is_listed_and_streamed_only_for_a_named_person(client, linked, actors):
    key = lambda who: {'X-API-Key': actors[who]['key']}
    path = f'/receipts/{linked["receipt"]}/bol-photos'
    listed = client.get(path, headers=key('floor'))
    assert listed.status_code == 200, listed.text
    photos = listed.json()['photos']
    assert len(photos) == 2 and photos[0]['form_row_number'] == 2 and photos[0]['bol_number'] == '632054'
    assert 'drive_file_id' not in photos[0]                              # the Drive id never leaves the API
    for who in ('floor', 'office', 'owner'):
        photo = client.get(f'{path}/{photos[0]["photo_id"]}/file', headers=key(who))
        assert photo.status_code == 200 and photo.content == b'photo-bytes'
        assert photo.headers['content-type'] == 'image/jpeg' and photo.headers['cache-control'] == 'private, no-store'
        assert photo.headers['x-content-type-options'] == 'nosniff'
    # No person behind the key, or no key: refused before Drive is ever asked.
    asked = len(linked['google'].calls)
    for suffix in ('', f'/{photos[0]["photo_id"]}/file'):
        assert client.get(path + suffix).status_code == 401
        assert client.get(path + suffix, headers={'X-API-Key': main.API_KEY}).status_code == 403
        if main.DASHBOARD_API_KEY:
            assert client.get(path + suffix, headers={'X-API-Key': main.DASHBOARD_API_KEY}).status_code == 403
        assert client.get(path + suffix, headers=key('retired')).status_code in (401, 403)
    assert len(linked['google'].calls) == asked
    for method in ('post', 'put', 'patch', 'delete'):
        assert getattr(client, method)(path, headers=key('owner')).status_code == 405


@pytest.mark.db
def test_a_person_tapped_on_the_name_screen_can_open_the_photo_and_only_michael_the_lists(client, people, linked):
    path = f'/receipts/{linked["receipt"]}/bol-photos'
    for name in ('Arturo', 'Luz', 'Miriam', 'Michael'):
        listed = client.get(path, headers={'X-FL-Person': name})
        assert listed.status_code == 200, listed.text
        photo = client.get(f'{path}/{listed.json()["photos"][0]["photo_id"]}/file', headers={'X-FL-Person': name})
        assert photo.status_code == 200 and photo.content == b'photo-bytes'
    for route in ('/verification/receiving-form/unmatched', '/verification/trial-recall/runs'):
        assert client.get(route, headers={'X-FL-Person': 'Arturo'}).status_code == 403
        assert client.get(route, headers={'X-FL-Person': 'Michael'}).status_code == 200


@pytest.mark.db
def test_only_a_linked_photo_of_that_receipt_is_ever_served(client, linked, actors):
    key = {'X-API-Key': actors['office']['key']}
    path = f'/receipts/{linked["receipt"]}/bol-photos'
    first, second = (p['photo_id'] for p in client.get(path, headers=key).json()['photos'])
    assert client.get(f'{path}/999999999/file', headers=key).status_code == 404
    assert client.get(f'/receipts/RCV-NOT-THIS-ONE/bol-photos/{first}/file', headers=key).status_code == 404
    assert client.get('/receipts/RCV-NOT-THIS-ONE/bol-photos', headers=key).json()['photos'] == []
    refused = client.get(f'{path}/{second}/file', headers=key)            # the linked file is not a photo
    assert refused.status_code == 404 and refused.json()['detail']['error_code'] == 'BOL_PHOTO_NOT_FOUND'
    assert ('media', PHOTO_2) not in linked['google'].calls
    linked['google'].files[PHOTO]['size'] = google_evidence.MAX_PHOTO_BYTES + 1
    assert client.get(f'{path}/{first}/file', headers=key).status_code == 413
    linked['google'].files[PHOTO] = {'mime_type': 'image/jpeg', 'size': 5, 'trashed': True}
    assert client.get(f'{path}/{first}/file', headers=key).status_code == 404
    del linked['google'].files[PHOTO]                                      # deleted from Drive
    assert client.get(f'{path}/{first}/file', headers=key).status_code == 404


@pytest.mark.db
def test_a_receipt_shows_only_what_the_latest_read_stands_behind(client, plant, actors, monkeypatch):
    key = {'X-API-Key': actors['office']['key']}
    google = FakeGoogle(files={PHOTO: {'mime_type': 'image/jpeg', 'size': 9}, PHOTO_2: {'mime_type': 'image/jpeg', 'size': 9}})
    monkeypatch.setattr(receiving_form, 'client_factory', lambda: google)
    sugar = plant.product('Sugar Brown')
    lot, tx = plant.receive(sugar, 5000, day(1))
    plant.cur.execute('SELECT receipt_number FROM transactions WHERE id = %s', (tx,))
    path = f'/receipts/{plant.cur.fetchone()["receipt_number"]}/bol-photos'
    first = sheet(plant.form_row(day(1), lot, ['Sugar brown-50lbs-100'], photo=PHOTO))
    edited = sheet(plant.form_row(day(1), lot, ['Sugar brown-50lbs-100'], photo=PHOTO_2))

    def served():
        """The Drive files the receipt's buttons would open."""
        ids = [p['photo_id'] for p in client.get(path, headers=key).json()['photos']]
        before = len(google.calls)
        assert all(client.get(f'{path}/{i}/file', headers=key).status_code == 200 for i in ids)
        return ids, [call[1] for call in google.calls[before:] if call[0] == 'media']

    receiving_form.sync(plant.cur, first)
    original_ids, files = served()
    assert files == [PHOTO]
    receiving_form.sync(plant.cur, edited)                               # the photo link was changed in the sheet
    edited_ids, files = served()
    assert files == [PHOTO_2] and edited_ids != original_ids
    assert client.get(f'{path}/{original_ids[0]}/file', headers=key).status_code == 404   # the old one is no longer served
    reverted = receiving_form.sync(plant.cur, first)                     # ...and changed back
    assert reverted['rows_new'] == 0 and reverted['links_new'] == 0      # nothing new is saved for wording seen before
    again_ids, files = served()
    assert (again_ids, files) == (original_ids, [PHOTO])                 # the receipt follows the sheet as it is now
    assert client.get(f'{path}/{edited_ids[0]}/file', headers=key).status_code == 404
    plant.void(tx)                                                       # the receipt is voided later
    receiving_form.sync(plant.cur, first)
    assert served() == ([], [])
    receiving_form.record_failed_sync(plant.cur, 'UNAVAILABLE')          # a failed night changes nothing shown
    assert served() == ([], [])


@pytest.mark.db
def test_photo_folder_lock_and_google_not_set_up(client, linked, actors, monkeypatch):
    key = {'X-API-Key': actors['floor']['key']}
    path = f'/receipts/{linked["receipt"]}/bol-photos'
    first = client.get(path, headers=key).json()['photos'][0]['photo_id']
    monkeypatch.setenv('RECEIVING_FORM_PHOTO_FOLDER_ID', 'the-form-folder')
    assert client.get(f'{path}/{first}/file', headers=key).status_code == 404     # not in the Form's folder
    linked['google'].files[PHOTO]['parents'] = ['the-form-folder']
    assert client.get(f'{path}/{first}/file', headers=key).status_code == 200
    monkeypatch.setattr(receiving_form, 'client_factory', lambda: google_evidence.Client(env={}))
    missing = client.get(f'{path}/{first}/file', headers=key)
    assert missing.status_code == 503 and missing.json()['detail']['error_code'] == 'BOL_PHOTO_NOT_CONFIGURED'


@pytest.mark.db
def test_before_migration_097_the_receipt_simply_has_no_bol_photo(client, linked, actors, monkeypatch):
    monkeypatch.setattr(receiving_form, 'migration_applied', lambda cur: False)
    key = {'X-API-Key': actors['floor']['key']}
    assert client.get(f'/receipts/{linked["receipt"]}/bol-photos', headers=key).json()['photos'] == []
    assert client.get(f'/receipts/{linked["receipt"]}/bol-photos/1/file', headers=key).status_code == 404
    assert client.get('/verification/receiving-form/unmatched',
                      headers={'X-API-Key': actors['owner']['key']}).status_code == 503


@pytest.mark.db
def test_the_owner_list_of_form_rows_with_no_matching_receipt(client, plant, actors):
    key = lambda who: {'X-API-Key': actors[who]['key']}
    empty = client.get('/verification/receiving-form/unmatched', headers=key('owner'))
    assert empty.status_code == 200
    sugar = plant.product('Sugar Brown')
    lot, _ = plant.receive(sugar, 5000, day(1))
    receiving_form.sync(plant.cur, sheet(plant.form_row(day(1), lot, ['Sugar brown-50lbs-100']),
                                         plant.form_row(day(2), 'GHOST-1', ['Sugar brown-50lbs-7'], hour=14)))
    receiving_form.record_failed_sync(plant.cur, 'AUTH_FAILED')              # a later night that could not read the sheet
    for who in ('owner', 'office'):
        answer = client.get('/verification/receiving-form/unmatched', headers=key(who))
        assert answer.status_code == 200, answer.text
        body = answer.json()
        assert body['title'] == 'Receiving form row with no matching FL receipt'
        assert [u['row_number'] for u in body['unmatched']] == [3] and body['unmatched'][0]['lot_text'] == 'GHOST-1'
        assert body['sync']['unmatched_count'] == 1 and body['later_failure']['job_error'] == 'AUTH_FAILED'
        assert body['product_wording_unconfirmed'] == []
    assert client.get('/verification/receiving-form/unmatched', headers=key('floor')).status_code == 403
    assert client.get('/verification/receiving-form/unmatched').status_code == 401
    assert client.get('/verification/receiving-form/unmatched', headers={'X-API-Key': main.API_KEY}).status_code == 200


def test_gpt_schemas_are_untouched_and_the_routes_stay_out_of_openapi():
    methods = {'get', 'post', 'put', 'patch', 'delete'}
    for path, operations in (('openapi-gpt-v3.yaml', 30), ('gpt-configs/schemas/openapi-floor.yaml', 22)):
        text = (ROOT / path).read_text(encoding='utf-8')
        spec = yaml.safe_load(text)
        assert sum(1 for item in spec['paths'].values() for m in item if m in methods) == operations, path
        assert 'bol-photos' not in text and 'trial-recall' not in text and 'receiving-form' not in text, path
    documented = main.app.openapi()['paths']
    assert not [p for p in documented if 'bol-photos' in p or 'receiving-form' in p or 'trial-recall' in p]
    served = {(m, r.path) for r in main.app.routes for m in getattr(r, 'methods', ())
              if 'bol-photos' in r.path or 'receiving-form' in r.path}
    assert served == set(receiving_form.ROUTES) and all(m == 'GET' for m, _ in served)
    # Named people reach the photo routes (also when tapped on an approved browser); the public dashboard key never does.
    assert receiving_form.PHOTO_ROUTES <= main.ACTOR_WRITE_ALLOWLIST and receiving_form.PHOTO_ROUTES <= main.SELECTION_ALLOWLIST
    assert not receiving_form.ROUTES & main.DASHBOARD_KEY_ALLOWLIST and not receiving_form.ROUTES & main.READONLY_KEY_ALLOWLIST
    assert not receiving_form.OWNER_ROUTES & main.SELECTION_ALLOWLIST


# ───────────────────────── the command and the nightly hook ─────────────────────────

def test_the_nightly_hooks_are_off_unless_switched_on(monkeypatch):
    launched = []

    class Done:
        returncode = 0

    monkeypatch.setattr(exceptions_sweep.subprocess, 'run', lambda cmd, check: launched.append(cmd) or Done())
    for name in ('RECEIVING_FORM_SYNC_ENABLED', 'TRIAL_RECALL_ENABLED'):
        monkeypatch.delenv(name, raising=False)
    assert exceptions_sweep.run_receiving_form_sync() == 0 and exceptions_sweep.run_trial_recall() == 0 and launched == []
    monkeypatch.setenv('RECEIVING_FORM_SYNC_ENABLED', '1')
    monkeypatch.setenv('TRIAL_RECALL_ENABLED', 'true')
    assert exceptions_sweep.run_receiving_form_sync() == 0 and exceptions_sweep.run_trial_recall() == 0
    assert [cmd[1].rsplit('/', 1)[-1] for cmd in launched] == ['receiving_form_sync.py', 'trial_recall_drill.py']
    assert launched[1][2:] == ['--scheduled'] and launched[0][2:] == []


@pytest.mark.db
def test_the_command_reads_the_sheet_before_the_database_and_notes_a_failed_read(plant, monkeypatch):
    google = FakeGoogle([HEADER, plant.form_row(day(1), 'NO-LOT', ['Sugar-50lbs-1'])])
    read = receiving_form_sync.read_sheet(google)
    assert google.calls == [('sheet', receiving_form.SHEET_ID, receiving_form.TAB)]
    assert read['sheet_id'] == receiving_form.SHEET_ID and read['gid'] == 1234
    receiving_form.record_failed_sync(plant.cur, 'FormLayoutError')
    plant.cur.execute("SELECT job_status, job_error FROM receiving_form_sync_runs ORDER BY id DESC LIMIT 1")
    assert tuple(plant.cur.fetchone().values()) == ('failed', 'FormLayoutError')
