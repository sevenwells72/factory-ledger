"""Trial recall (trial_recall.py, scripts/trial_recall_drill.py, migration 097).

Pins the rules of the drill: strictly read-only on the ledger; posted
current-ledger data only; an unreadable BOL is UNVERIFIED, never PASS; all three
values shown on a disagreement; nothing before the cutoff is tested; nothing
eligible is "No eligible lot", not a failure.
"""
import ast
from datetime import datetime, timedelta, timezone
import os
import random
import re

import pytest

import extraction
import google_evidence
import main
import receiving_form
import reconciliation
import trial_recall
from scripts import trial_recall_drill
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.trial_recall_kit import (CUTOFF, NEW_TABLES, PHOTO, PHOTO_2, ROOT, FakeGoogle, Plant, day,
                                    existing_tables, fingerprint, sheet)

RUN_AT = datetime(2026, 11, 18, 8, 0, tzinfo=timezone.utc)        # Wed Nov 18, 03:00 Eastern


@pytest.fixture
def plant(db_cursor):
    return Plant(db_cursor)


def good_bol(plant, **changes):
    """What the extraction model reads from a BOL that agrees with the Form row and FL."""
    bol = {'readable': True, 'supplier_name': plant.supplier_name, 'manufacturer_name': 'Domino Foods',
           'carrier_name': 'Jack’s Eggs Trucking', 'bol_number': '632054', 'reference_numbers': ['PO 4412'],
           'document_date': None,
           'lines': [{'description': plant.name('Sugar Brown'), 'quantity': 225, 'unit': 'BAGS', 'lot_numbers': ['SUP-1']}]}
    bol.update(changes)
    return bol


def drill(plant, read, *, bol=None, google=None, extractor=None, seed=0, cutoff=CUTOFF):
    google = google or FakeGoogle(read['values'])
    data = trial_recall.load(plant.cur, read, cutoff=cutoff, rng=random.Random(seed))
    read_bol = bol if bol is not None else good_bol(plant)
    trial_recall.gather_evidence(data, google, extractor or (
        lambda content, mime: {'extraction': read_bol, 'extraction_model': 'test-model'}))
    trial_recall.confirm_names(plant.cur, data)
    analysis = trial_recall.analyse(data)
    return analysis, trial_recall.render_email(analysis, RUN_AT)


def clean(plant, **options):
    chain = plant.chain(**options)
    read = sheet(plant.form_row(chain['on'], chain['sugar_lot'], ['Sugar brown-50lbs-225']))
    return chain, read


def kinds(test):
    return sorted({flag['kind'] for flag in test['flags']})


# ───────────────────────── a clean chain ─────────────────────────

@pytest.mark.db
def test_a_clean_chain_passes_both_tests_and_full_recall_is_yes(plant):
    chain, read = clean(plant)
    analysis, email = drill(plant, read)
    forward, backward = analysis['forward'], analysis['backward']
    assert (forward['result'], forward['receiving_result'], forward['trace_result']) == ('pass', 'pass', 'pass'), forward
    assert backward['result'] == 'pass', (backward['flags'], backward['notes'])
    assert analysis['full_recall_possible'] == 'yes' and analysis['reasons'] == [] and analysis['flag_count'] == 0
    # Sheet vs BOL vs FL, every value shown.
    fields = {f['field']: f for f in forward['comparisons'][0]['fields'] + forward['document_fields']}
    assert fields['Quantity']['sheet'] == '225 x 50 lb = 11,250 lb' and fields['Quantity']['bol'] == '225 BAGS'
    assert fields['Quantity']['fl'] == '11,250 lb' and fields['Quantity']['status'] == 'agree'
    assert {name: f['status'] for name, f in fields.items()} == {
        'Product': 'agree', 'Quantity': 'agree', 'Internal lot': 'agree', 'Supplier lot': 'agree',
        'BOL / reference number': 'agree', 'Supplier': 'agree', 'Other products on the BOL': 'agree',
        'Manufacturer': 'cannot compare', 'Carrier': 'agree'}
    assert fields['Manufacturer']['bol'] == 'Domino Foods'
    # Forward: receipt -> batch -> cases -> shipment -> customer.
    trace = forward['traces'][0]
    assert [(s['type'], s['source_lot_id'], s['output_lot_id']) for s in trace['steps']] == [
        ('make', chain['sugar_lot'], chain['batch_lot']), ('pack', chain['batch_lot'], chain['case_lot'])]
    assert trace['customers'] == [plant.name('Market')] and trace['shipments'][0]['quantity_lb'] == 300
    assert forward['customers'] == [plant.name('Market')]
    # Backward: the finished lot back to the sugar lot, its Form row and its photo.
    assert backward['lot']['lot_id'] == chain['case_lot']
    ingredient, = backward['ingredients']
    assert ingredient['lot_id'] == chain['sugar_lot'] and ingredient['before_cutoff'] is False
    assert ingredient['evidence'] == 'Form row and BOL photo on file'
    receipt, = ingredient['receipts']
    assert receipt['transaction_id'] == chain['receipt'] and receipt['form_rows'][0]['row_number'] == 2
    assert receipt['photos'][0]['accessible'] is True and receipt['before_cutoff'] is False
    assert backward['other_shipments'] == [] and backward['own']['customers'] == [plant.name('Market')]
    # The inventory section adds up from posted lines.
    lots = {lot['lot_id']: lot for lot in forward['chain']['lots']}
    assert (lots[chain['sugar_lot']]['received'], lots[chain['sugar_lot']]['used'], lots[chain['sugar_lot']]['on_hand']) == (
        11250, 400, 10850)
    assert (lots[chain['case_lot']]['made'], lots[chain['case_lot']]['shipped'], lots[chain['case_lot']]['on_hand']) == (
        400, 300, 100)


@pytest.mark.db
def test_the_email_has_the_five_sections_in_order_and_links_to_the_sheet_row_and_the_bol(plant):
    _, read = clean(plant)
    _, email = drill(plant, read)
    text = email['text']
    assert email['subject'] == 'Factory Ledger trial recall, Wed Nov 18: Full recall possible: YES'
    order = [text.index(heading) for heading in (
        '1. RECEIVING VERIFICATION (SHEET / BOL VS FL)', '2. INVENTORY AND CONSUMPTION RECONCILIATION',
        '3. FORWARD TRACEABILITY', '4. MISSING OR CONFLICTING RECORDS', '5. FULL RECALL POSSIBLE: YES')]
    assert order == sorted(order)
    sheet_row = f'https://docs.google.com/spreadsheets/d/{receiving_form.SHEET_ID}/edit#gid=1234&range=A2'
    photo = f'https://drive.google.com/file/d/{PHOTO}/view'
    for link in (sheet_row, photo):
        assert link in text and link in email['html'].replace('&amp;', '&')
    assert 'Nothing was changed.' in text and 'nothing is assumed about which stock was used first' in text
    assert 'None of them is assumed to be the right one.' in text
    assert 'Sheet: 225 x 50 lb = 11,250 lb | BOL: 225 BAGS | FL: 11,250 lb' in text
    assert '<script' not in email['html']


# ───────────────────────── unreadable is UNVERIFIED, never PASS ─────────────────────────

def _unreadable(plant):
    return good_bol(plant, readable=False, lines=[])


def _raises(content, mime):
    raise extraction.ExtractionError('Vision API call failed: timeout')


@pytest.mark.db
@pytest.mark.parametrize('case,expected', [
    ('model says not readable', 'could not read product lines'),
    ('model returns no lines', 'could not read product lines'),
    ('model call fails', 'could not read this photo'),
    ('photo gone from drive', 'no such file'),
    ('photo in the trash', 'Drive trash'),
    ('phone photo that cannot be converted', 'could not be converted'),
    ('format the model cannot read', 'cannot be read by the extraction model'),
    ('not a photo', 'not a photo or PDF'),
])
def test_a_bol_that_cannot_be_opened_or_read_is_unverified_never_pass(plant, case, expected):
    _, read = clean(plant)
    google, bol, extractor = FakeGoogle(read['values']), None, None
    if case == 'model says not readable':
        bol = _unreadable(plant)
    elif case == 'model returns no lines':
        bol = good_bol(plant, lines=[])
    elif case == 'model call fails':
        extractor = _raises
    elif case == 'photo gone from drive':
        google.files = {}
    elif case == 'photo in the trash':
        google.files[PHOTO]['trashed'] = True
    elif case == 'phone photo that cannot be converted':
        google.files[PHOTO]['mime_type'] = 'image/heic'      # the fake file's bytes are not a real HEIC photo
    elif case == 'format the model cannot read':
        google.files[PHOTO]['mime_type'] = 'image/gif'
    else:
        google.files[PHOTO]['mime_type'] = 'text/html'
    analysis, email = drill(plant, read, bol=bol, google=google, extractor=extractor)
    forward = analysis['forward']
    assert forward['receiving_result'] == 'unverified' and forward['result'] == 'unverified'
    assert expected.lower() in ' '.join(forward['notes']).lower()
    # Unreadable evidence leaves the receiving check unverified; it does not by itself make the answer NO.
    assert analysis['full_recall_possible'] == 'yes'
    assert 'Forward test receiving verification: UNVERIFIED' in analysis['receiving_gaps']
    assert 'By themselves these do not change the answer above.' in email['text']
    assert 'Result: UNVERIFIED' in email['text'] and 'Result: PASS' not in email['text']
    # The trace itself still ran: the books are not what was unreadable.
    assert forward['trace_result'] == 'pass' and forward['customers'] == [plant.name('Market')]


@pytest.mark.db
def test_a_readable_bol_that_cannot_confirm_product_or_quantity_is_not_a_pass(plant):
    _, read = clean(plant)
    vague = good_bol(plant, lines=[{'description': 'ITEM 4421 CONF 10X', 'quantity': 3, 'unit': 'PALLETS',
                                    'lot_numbers': []}])
    analysis, _ = drill(plant, read, bol=vague)
    fields = {f['field']: f for f in analysis['forward']['comparisons'][0]['fields']}
    assert fields['Product']['status'] == 'cannot compare' and fields['Product']['bol'] == 'ITEM 4421 CONF 10X'
    assert fields['Quantity']['bol'] == '3 PALLETS' and fields['Quantity']['bol_compared'] is False
    assert analysis['forward']['receiving_result'] == 'unverified'


# ───────────────────────── disagreements: all three values, FL never assumed right ─────────────────────────

@pytest.mark.db
def test_a_quantity_difference_shows_all_three_values_and_fails(plant):
    _, read = clean(plant)
    short = good_bol(plant, lines=[{'description': plant.name('Sugar Brown'), 'quantity': 200, 'unit': 'BAGS',
                                    'lot_numbers': ['SUP-1']}])
    analysis, email = drill(plant, read, bol=short)
    quantity = next(f for f in analysis['forward']['comparisons'][0]['fields'] if f['field'] == 'Quantity')
    assert quantity['status'] == 'disagree'
    assert (quantity['sheet'], quantity['bol'], quantity['fl']) == ('225 x 50 lb = 11,250 lb', '200 BAGS', '11,250 lb')
    assert analysis['forward']['receiving_result'] == 'fail'
    assert analysis['full_recall_possible'] == 'yes'                    # a receiving mismatch alone is not a NO
    assert 'Quantity: the records disagree (Sheet: 225 x 50 lb = 11,250 lb | BOL: 200 BAGS | FL: 11,250 lb).' in email['text']
    assert 'Receiving checks that did not pass (section 1): Forward test receiving verification: FAIL' in email['text']
    assert not re.search(r'FL is (right|correct)|correct value', email['text'], re.I)


@pytest.mark.db
def test_fl_differing_from_the_sheet_and_the_bol_is_a_disagreement_too(plant):
    chain = plant.chain(received=10000)                                 # FL says 10,000 lb; the Form and the BOL say 11,250
    read = sheet(plant.form_row(chain['on'], chain['sugar_lot'], ['Sugar brown-50lbs-225']))
    by_weight = good_bol(plant, lines=[{'description': plant.name('Sugar Brown'), 'quantity': 11250, 'unit': 'LB',
                                        'lot_numbers': ['SUP-1']}])
    analysis, _ = drill(plant, read, bol=by_weight)
    quantity = next(f for f in analysis['forward']['comparisons'][0]['fields'] if f['field'] == 'Quantity')
    assert quantity['status'] == 'disagree' and quantity['fl'] == '10,000 lb' and quantity['bol'] == '11250 LB'
    assert analysis['forward']['receiving_result'] == 'fail'


@pytest.mark.db
@pytest.mark.parametrize('changes,field', [
    ({'bol_number': '777001', 'reference_numbers': []}, 'BOL / reference number'),
    ({'lines': 'other lot'}, 'Supplier lot'),
])
def test_other_disagreements_fail_with_the_values_shown(plant, changes, field):
    _, read = clean(plant)
    if changes.get('lines') == 'other lot':
        changes = {'lines': [{'description': plant.name('Sugar Brown'), 'quantity': 225, 'unit': 'BAGS',
                              'lot_numbers': ['ZZ-999']}]}
    analysis, email = drill(plant, read, bol=good_bol(plant, **changes))
    every = analysis['forward']['comparisons'][0]['fields'] + analysis['forward']['document_fields']
    assert next(f for f in every if f['field'] == field)['status'] == 'disagree'
    assert analysis['forward']['receiving_result'] == 'fail' and f'{field}: the records disagree' in email['text']


@pytest.mark.db
def test_a_form_row_with_no_fl_receipt_fails_and_is_listed(plant):
    plant.chain()
    read = sheet(plant.form_row(day(1), 'GHOST-LOT', ['Sugar brown-50lbs-225']))
    analysis, email = drill(plant, read)
    forward = analysis['forward']
    assert forward['result'] == 'fail' and forward['receipts'] == [] and forward['traces'] == []
    assert 'No FL receipt could be matched to this Form row' in forward['problems'][0]
    assert [row['row_number'] for row in analysis['unmatched_rows']] == [2]
    assert 'Receiving form row with no matching FL receipt: 1 of 1 row(s)' in email['text']
    assert forward['trace_result'] is None                              # nothing to trace from: not run
    # The backward test still traced its lot, and nothing there stops a customer being identified.
    assert analysis['backward']['result'] == 'fail' and 'RECEIVING_EVIDENCE_MISSING' in kinds(analysis['backward'])
    assert analysis['full_recall_possible'] == 'yes'


# ───────────────────────── the cutoff ─────────────────────────

@pytest.mark.db
def test_nothing_eligible_is_no_eligible_lot_not_a_failure(plant):
    plant.chain(on=day(-10))                                            # everything happened before the cutoff
    read = sheet(plant.form_row(day(-10), 'OLD-LOT', ['Sugar brown-50lbs-225']))
    analysis, email = drill(plant, read)
    assert analysis['forward']['result'] == analysis['backward']['result'] == 'no_eligible_lot'
    assert analysis['full_recall_possible'] == 'not_tested' and analysis['reasons'] == []
    assert 'Forward test: No eligible lot. Backward test: No eligible lot.' in email['text']
    assert 'Result: FAIL' not in email['text'] and 'Forward test: FAIL' not in email['text']
    assert email['subject'].endswith('Full recall possible: NOT TESTED (no eligible lot)')


@pytest.mark.db
def test_an_older_ingredient_is_shown_before_cutoff_not_tested(plant):
    old_sugar, new_honey = plant.product('Sugar Brown'), plant.product('Honey Clover')
    batch, cases = plant.product('Batch Granola', 'batch'), plant.product('Granola 25 LB', 'finished')
    sugar_lot, _ = plant.receive(old_sugar, 5000, day(-6), supplier_lot=None)    # before the cutoff, and no supplier lot
    honey_lot, _ = plant.receive(new_honey, 600, day(1))
    batch_lot, case_lot = plant.lot(batch, supplier_lot=None), plant.lot(cases, supplier_lot=None)
    plant.post('make', [(batch_lot, 400), (sugar_lot, -300), (honey_lot, -100)], day(2))
    plant.post('pack', [(case_lot, 400), (batch_lot, -400)], day(2), hour=13)
    plant.ship(case_lot, 100, day(3), plant.name('Market'))
    read = sheet(plant.form_row(day(1), honey_lot, ['Honey clover-60lbs-10']))
    honey_bol = good_bol(plant, lines=[{'description': plant.name('Honey Clover'), 'quantity': 600, 'unit': 'LB',
                                        'lot_numbers': ['SUP-1']}])
    analysis, email = drill(plant, read, bol=honey_bol)
    backward = analysis['backward']
    by_lot = {item['lot_id']: item for item in backward['ingredients']}
    assert by_lot[sugar_lot]['before_cutoff'] is True and 'evidence' not in by_lot[sugar_lot]
    assert by_lot[honey_lot]['before_cutoff'] is False and by_lot[honey_lot]['evidence'] == 'Form row and BOL photo on file'
    assert backward['not_tested_count'] == 1
    # The old lot is not tested, so its missing supplier lot is not counted against the drill.
    assert backward['result'] == 'pass' and 'MISSING_SUPPLIER_LOT' not in kinds(backward)
    assert sugar_lot not in {lot['lot_id'] for lot in backward['chain']['lots']}
    assert 'before cutoff - not tested' in email['text']
    assert '1 ingredient lot(s) in this chain are before the cutoff and were not tested' in email['text']


@pytest.mark.db
def test_only_shipped_finished_lots_made_on_or_after_the_cutoff_are_picked(plant):
    chain = plant.chain()
    plant.void(chain['ship'])                                           # a voided shipment is no shipment
    analysis, _ = drill(plant, sheet(plant.form_row(chain['on'], chain['sugar_lot'], ['Sugar brown-50lbs-225'])))
    assert analysis['backward']['result'] == 'no_eligible_lot'
    assert analysis['forward']['traces'][0]['shipments'] == []          # and the forward trace no longer shows it
    assert analysis['full_recall_possible'] == 'yes'                    # the forward test passed; nothing else to test


# ───────────────────────── backward: every other customer ─────────────────────────

@pytest.mark.db
def test_backward_goes_forward_again_to_every_other_customer(plant):
    chain, read = clean(plant, customer=None)
    other_cases = plant.lot(chain['cases'], supplier_lot=None)
    other_batch = plant.lot(chain['batch'], supplier_lot=None)
    plant.post('make', [(other_batch, 200), (chain['sugar_lot'], -200)], day(4))
    plant.post('pack', [(other_cases, 200), (other_batch, -200)], day(4), hour=13)
    plant.ship(other_cases, 150, day(5), plant.name('Other Grocer'))
    seen = set()
    for seed in range(8):                                               # either finished lot may be picked
        analysis, email = drill(plant, read, seed=seed)
        backward = analysis['backward']
        assert backward['pool_size'] == 2 and backward['result'] == 'pass'
        picked = backward['lot']['lot_id']
        seen.add(picked)
        mine, theirs = ((plant.name('Market'), plant.name('Other Grocer')) if picked == chain['case_lot']
                        else (plant.name('Other Grocer'), plant.name('Market')))
        assert backward['own']['customers'] == [mine] and backward['other_customers'] == [theirs]
        assert backward['other_shipments'][0]['via_ingredient_lots'] == [plant.lots[chain['sugar_lot']]]
        assert f'Other customers reached by the same ingredient lots: {theirs} (1 other shipment(s)).' in email['text']
        assert sorted(analysis['forward']['customers']) == sorted([mine, theirs])
    assert seen == {chain['case_lot'], other_cases}


# ───────────────────────── the flags ─────────────────────────

def _removed_by_adjustment(plant, chain):
    lost = plant.lot(chain['batch'], supplier_lot=None)
    plant.post('make', [(lost, 100), (chain['sugar_lot'], -100)], day(3))
    plant.post('adjust', [(lost, -100)], day(3), hour=14, reason_code='damage_disposal')


def _shipped_not_recorded(plant, chain):
    plant.cur.execute("INSERT INTO customers (name) VALUES (%s) RETURNING id", (plant.name('Unrecorded Buyer'),))
    customer = plant.cur.fetchone()['id']
    plant.cur.execute("""INSERT INTO sales_orders (order_number, customer_id, state, state_reason, state_note,
                                                   state_changed_at, state_changed_by, status)
                         VALUES (%s, %s, 'closed', 'shipped_not_recorded', 'left on a truck', %s, 'Michael', 'shipped')
                         RETURNING id""", (f'SO-{plant.tag}', customer, datetime(2026, 11, 15, 15, tzinfo=timezone.utc)))
    plant.cur.execute('INSERT INTO sales_order_lines (sales_order_id, product_id, quantity_lb) VALUES (%s, %s, 100)',
                      (plant.cur.fetchone()['id'], chain['cases']))


def _shipment_without_bol(plant, chain):
    plant.ship(chain['case_lot'], 50, day(4), plant.name('Market'), bol=None)


def _unknown_adjustment(plant, chain):
    plant.post('adjust', [(chain['batch_lot'], 25)], day(2), hour=11, reason_code='unknown')


def _shared_key(plant, chain):
    extra = plant.lot(chain['batch'], supplier_lot=None)
    plant.post('make', [(extra, 50), (chain['sugar_lot'], -50)], day(3), named=False)


def _no_supplier_lot(plant, chain):
    plant.cur.execute('ALTER TABLE lots DISABLE TRIGGER USER')
    plant.cur.execute("UPDATE lots SET supplier_lot_code = NULL, identity_status = 'unidentified' WHERE id = %s",
                      (chain['sugar_lot'],))
    plant.cur.execute('ALTER TABLE lots ENABLE TRIGGER USER')


def _below_zero(plant, chain):
    plant.ship(chain['case_lot'], 500, day(4), plant.name('Market'))      # 400 packed, 800 shipped


def _open_exception(plant, chain):
    plant.cur.execute("""INSERT INTO exceptions (kind, status, severity, lot_id, product_id)
                         VALUES ('SHORTAGE', 'open', 'warn', %s, %s)""", (chain['sugar_lot'], chain['sugar']))


def _pooled(plant, chain):
    plant.cur.execute('ALTER TABLE lots DISABLE TRIGGER USER')
    plant.cur.execute("UPDATE lots SET entry_source_notes = 'Opening stock. POOLED from 3 supplier lots' WHERE id = %s",
                      (chain['sugar_lot'],))
    plant.cur.execute('ALTER TABLE lots ENABLE TRIGGER USER')


@pytest.mark.db
@pytest.mark.parametrize('spoil,kind,level', [
    # An affected customer or shipment cannot be identified: Full recall possible is NO.
    (_removed_by_adjustment, 'MADE_REMOVED_BY_ADJUSTMENT', 'blocks'),
    (_shipped_not_recorded, 'ORDER_SHIPPED_NOT_RECORDED', 'blocks'),
    (_unknown_adjustment, 'UNKNOWN_ADJUSTMENT_INTO_PRODUCT', 'blocks'),
    (_below_zero, 'NEGATIVE_BALANCE', 'blocks'),
    (_open_exception, 'LOT_ATTRIBUTION_UNRESOLVED', 'blocks'),
    (_pooled, 'LOT_ATTRIBUTION_UNRESOLVED', 'blocks'),
    # A receiving-record gap: FAIL in the receiving section, not a NO by itself.
    (_no_supplier_lot, 'MISSING_SUPPLIER_LOT', 'receiving'),
    # Warnings only.
    (_shipment_without_bol, 'SHIPMENT_NO_BOL', 'warning'),
    (_shared_key, 'SHARED_KEY_POSTING', 'warning'),
])
def test_each_record_problem_is_flagged_at_its_level(plant, spoil, kind, level):
    chain, read = clean(plant)
    spoil(plant, chain)
    analysis, email = drill(plant, read)
    forward = analysis['forward']
    assert kind in kinds(forward), kinds(forward)
    flag = next(f for f in forward['flags'] if f['kind'] == kind)
    assert flag['level'] == level and flag['blocks'] is (level == 'blocks')
    assert flag['title'] in email['text'] and flag['message'] in email['text']
    if level == 'blocks':
        assert forward['trace_result'] == 'fail' and analysis['full_recall_possible'] == 'no'
        assert any(flag['message'] in reason for reason in analysis['reasons'])
    else:
        assert forward['trace_result'] == 'pass' and analysis['full_recall_possible'] == 'yes'
        assert analysis['reasons'] == []
        if level == 'receiving':
            assert forward['receiving_result'] == 'fail' and flag['message'] in forward['problems']
            assert 'by itself it does not make the answer NO' in email['text']
        else:
            assert forward['receiving_result'] == 'pass' and forward['result'] == 'pass'
            assert '(warning only)' in email['text']


@pytest.mark.db
def test_an_ingredient_lot_with_no_form_row_or_no_supplier_receipt_blocks_the_backward_test(plant):
    chain = plant.chain()
    honey = plant.product('Honey Clover')
    found = plant.lot(honey, supplier_lot='H-1')
    plant.post('adjust', [(found, 80)], day(1), reason_code='physical_count')      # stock with no supplier receipt
    extra_batch, extra_cases = plant.lot(chain['batch'], supplier_lot=None), plant.lot(chain['cases'], supplier_lot=None)
    plant.post('make', [(extra_batch, 60), (found, -60)], day(3))
    plant.post('pack', [(extra_cases, 60), (extra_batch, -60)], day(3), hour=13)
    plant.ship(extra_cases, 60, day(4), plant.name('Market'))
    plant.void(chain['ship'])                                           # so only the new finished lot is eligible
    analysis, _ = drill(plant, sheet())                                 # and the Form has no row at all
    backward = analysis['backward']
    assert backward['lot']['lot_id'] == extra_cases and backward['result'] == 'fail'
    assert 'LOT_ATTRIBUTION_UNRESOLVED' in kinds(backward)
    assert any('did not come in on a supplier receipt' in f['message'] for f in backward['flags'])
    # The sugar chain's own finished lot: its ingredient has a receipt but no Form row.
    plant.ship(chain['case_lot'], 10, day(5), plant.name('Market'))
    seen = {}
    for seed in range(8):
        result, _ = drill(plant, sheet(), seed=seed)
        seen[result['backward']['lot']['lot_id']] = result['backward']
    sugar_chain = seen[chain['case_lot']]
    assert 'RECEIVING_EVIDENCE_MISSING' in kinds(sugar_chain) and sugar_chain['result'] == 'fail'
    assert sugar_chain['ingredients'][0]['evidence'] == 'no Form row'


@pytest.mark.db
def test_stock_received_or_found_straight_into_an_intermediate_batch_blocks_the_backward_test(plant):
    chain, read = clean(plant)
    # 40 lb "found" was put straight into the batch lot that the finished lot was packed from.
    plant.post('receive', [(chain['batch_lot'], 40)], day(2), hour=12, shipper_name='Found inventory')
    analysis, email = drill(plant, read)
    backward = analysis['backward']
    assert backward['lot']['lot_id'] == chain['case_lot'] and backward['result'] == 'fail'
    flag = next(f for f in backward['flags'] if 'a batch inside this finished lot' in f['message'])
    assert flag['kind'] == 'LOT_ATTRIBUTION_UNRESOLVED' and flag['blocks'] is True
    assert plant.lots[chain['batch_lot']] in flag['message'] and flag['message'] in email['text']
    assert analysis['full_recall_possible'] == 'no'


@pytest.mark.db
def test_a_batch_first_made_before_the_cutoff_is_shown_and_not_tested(plant):
    sugar, batch = plant.product('Sugar Brown'), plant.product('Batch Granola', 'batch')
    cases = plant.product('Granola 25 LB', 'finished')
    sugar_lot, _ = plant.receive(sugar, 5000, day(-8), supplier_lot=None)
    batch_lot, case_lot = plant.lot(batch, supplier_lot=None), plant.lot(cases, supplier_lot=None)
    plant.post('make', [(batch_lot, 400), (sugar_lot, -400)], day(-3), named=False)       # before the cutoff
    plant.post('receive', [(batch_lot, 15)], day(-3), hour=12, shipper_name='Found inventory')
    plant.post('pack', [(case_lot, 400), (batch_lot, -400)], day(1))                        # packed after it
    plant.ship(case_lot, 100, day(2), plant.name('Market'))
    analysis, email = drill(plant, sheet())
    backward = analysis['backward']
    assert backward['lot']['lot_id'] == case_lot and backward['result'] == 'pass', backward['flags']
    assert backward['made_lots'][0]['before_cutoff'] is True and backward['made_not_tested_count'] == 1
    assert {lot['lot_id'] for lot in backward['chain']['lots']} == {case_lot}             # only the tested lot is flagged on
    assert '1 batch lot(s) in this chain were first made before the cutoff and were not tested.' in email['text']


@pytest.mark.db
def test_a_backward_photo_whose_contents_drive_refuses_is_unverified(plant):
    _, read = clean(plant)
    google = FakeGoogle(read['values'])
    data = trial_recall.load(plant.cur, read, rng=random.Random(0))
    google.refused_media.add(PHOTO)                                     # details visible, contents refused
    trial_recall.gather_evidence(data, google, lambda content, mime: {'extraction': good_bol(plant)})
    trial_recall.confirm_names(plant.cur, data)
    analysis = trial_recall.analyse(data)
    backward = analysis['backward']
    assert backward['ingredients'][0]['receipts'][0]['photos'][0]['accessible'] is False
    assert backward['ingredients'][0]['evidence'] == 'Form row, BOL photo could not be opened'
    assert backward['result'] == 'unverified' and backward['evidence_result'] == 'unverified'
    assert analysis['full_recall_possible'] == 'yes'                     # evidence that cannot be opened is not a NO
    assert analysis['forward']['receiving_result'] == 'unverified'        # the forward read could not open it either
    opened = [call for call in google.calls if call[0] == 'media']
    assert opened                                                        # the file itself was asked for, not only its details


@pytest.mark.db
def test_a_lot_with_no_dated_entry_is_unresolved_not_before_cutoff(plant):
    chain, read = clean(plant)
    mystery = plant.lot(plant.product('Honey Clover'), supplier_lot=None)     # used in a make, never received
    extra_batch, extra_cases = plant.lot(chain['batch'], supplier_lot=None), plant.lot(chain['cases'], supplier_lot=None)
    plant.post('make', [(extra_batch, 60), (mystery, -60)], day(3))
    plant.post('pack', [(extra_cases, 60), (extra_batch, -60)], day(3), hour=13)
    plant.ship(extra_cases, 60, day(4), plant.name('Market'))
    plant.void(chain['ship'])
    analysis, email = drill(plant, read)
    backward = analysis['backward']
    item, = backward['ingredients']
    assert item['lot_id'] == mystery and item['first_in'] is None and item['before_cutoff'] is False
    assert backward['result'] == 'fail' and backward['not_tested_count'] == 0
    assert {'LOT_ATTRIBUTION_UNRESOLVED', 'NEGATIVE_BALANCE'} <= set(kinds(backward))
    assert any('did not come in on a supplier receipt' in f['message'] for f in backward['flags'])
    assert 'before cutoff - not tested' not in email['text'] and analysis['full_recall_possible'] == 'no'


@pytest.mark.db
def test_each_receipt_into_an_ingredient_lot_needs_its_own_evidence(plant):
    chain, read = clean(plant)                                          # the first receipt has a Form row and a photo
    second = plant.post('receive', [(chain['sugar_lot'], 500)], day(5), hour=8, bol_reference='777',
                        supplier_id=plant.supplier, shipper_name=plant.supplier_name)     # a later truck, no Form row
    analysis, _ = drill(plant, read)
    backward = analysis['backward']
    receipts = {r['transaction_id']: r for r in backward['ingredients'][0]['receipts']}
    assert receipts[chain['receipt']]['evidence'] == 'Form row and BOL photo on file'
    assert receipts[second]['evidence'] == 'no Form row'
    assert backward['result'] == 'fail' and 'RECEIVING_EVIDENCE_MISSING' in kinds(backward)
    # Found stock added to the same lot is not covered by the supplier receipt's paperwork either.
    plant.void(second)
    found = plant.post('receive', [(chain['sugar_lot'], 30)], day(5), hour=9, shipper_name='Found inventory')
    analysis, email = drill(plant, read)
    backward = analysis['backward']
    receipts = {r['transaction_id']: r for r in backward['ingredients'][0]['receipts']}
    assert receipts[found]['evidence'] == 'not a supplier receipt' and backward['result'] == 'fail'
    assert any('came in on' in f['message'] and 'as found stock' in f['message'] for f in backward['flags'])
    assert 'not a supplier receipt' in email['text']


@pytest.mark.db
def test_an_older_receipt_into_a_lot_topped_up_after_the_cutoff_is_shown_not_tested(plant):
    sugar, batch = plant.product('Sugar Brown'), plant.product('Batch Granola', 'batch')
    cases = plant.product('Granola 25 LB', 'finished')
    sugar_lot, old = plant.receive(sugar, 5000, day(-6))
    new = plant.post('receive', [(sugar_lot, 2000)], day(1), bol_reference='632054', supplier_id=plant.supplier,
                     shipper_name=plant.supplier_name)
    batch_lot, case_lot = plant.lot(batch, supplier_lot=None), plant.lot(cases, supplier_lot=None)
    plant.post('make', [(batch_lot, 400), (sugar_lot, -400)], day(2))
    plant.post('pack', [(case_lot, 400), (batch_lot, -400)], day(2), hour=13)
    plant.ship(case_lot, 100, day(3), plant.name('Market'))
    analysis, _ = drill(plant, sheet(plant.form_row(day(1), sugar_lot, ['Sugar brown-50lbs-40'])))
    item, = analysis['backward']['ingredients']
    receipts = {r['transaction_id']: r for r in item['receipts']}
    assert item['before_cutoff'] is False                               # the lot took stock after the cutoff: it is tested
    assert receipts[old]['evidence'] == 'before cutoff - not tested' and receipts[old]['form_rows'] == []
    assert receipts[new]['evidence'] == 'Form row and BOL photo on file'
    assert analysis['backward']['result'] == 'pass'


@pytest.mark.db
def test_every_photo_on_the_form_row_must_be_readable(plant):
    chain = plant.chain()
    read = sheet(plant.form_row(chain['on'], chain['sugar_lot'], ['Sugar brown-50lbs-225'], photo=[PHOTO, PHOTO_2]))
    google = FakeGoogle(read['values'])                                 # the second photo is not in Drive
    analysis, email = drill(plant, read, google=google)
    forward = analysis['forward']
    assert [d['readable'] for d in forward['documents']] == [True, False]
    assert forward['receiving_result'] == 'unverified'                  # one good page does not cover a missing one
    assert any('1 of the 2 BOL photos on this Form row could not be opened or read' in n for n in forward['notes'])
    fields = {f['field']: f['status'] for f in forward['comparisons'][0]['fields']}
    assert fields['Product'] == 'agree' and fields['Quantity'] == 'agree'   # what was readable still agreed
    assert 'Result: UNVERIFIED' in email['text']
    # More photos than the model reads: the unread ones are not assumed fine.
    many = [PHOTO] * 1 + [f'{PHOTO[:-2]}{n:02d}' for n in range(trial_recall.MAX_PHOTOS_READ + 1)]
    read = sheet(plant.form_row(chain['on'], chain['sugar_lot'], ['Sugar brown-50lbs-225'], photo=many))
    google = FakeGoogle(read['values'], files={file_id: {'mime_type': 'image/jpeg', 'size': 9} for file_id in many})
    analysis, _ = drill(plant, read, google=google)
    assert analysis['forward']['receiving_result'] == 'unverified'
    assert any('2 more BOL photo(s) on this Form row were not read' in n for n in analysis['forward']['notes'])
    quantity = next(f for f in analysis['forward']['comparisons'][0]['fields'] if f['field'] == 'Quantity')
    assert quantity['status'] == 'agree' and quantity['bol'] == '225 BAGS'   # the same page six times is counted once


@pytest.mark.db
def test_quantity_is_not_confirmed_unless_fl_is_compared_too(plant):
    chain = plant.chain(received=1)                                      # FL says 1 lb
    # The Form gives no unit for the case size, so it has no weight; its 225 cases match the BOL's 225 bags.
    read = sheet(plant.form_row(chain['on'], chain['sugar_lot'], ['Sugar brown-50-225']))
    analysis, _ = drill(plant, read)
    quantity = next(f for f in analysis['forward']['comparisons'][0]['fields'] if f['field'] == 'Quantity')
    assert quantity['status'] == 'agree' and quantity['bol_compared'] is True and quantity['fl_compared'] is False
    assert quantity['fl'] == '1 lb' and quantity['sheet'] == '225 x 50 (no unit written)'
    assert analysis['forward']['receiving_result'] == 'unverified'       # Form and BOL agreeing proves nothing about FL
    assert any("FL's quantity could not be compared" in note for note in analysis['forward']['notes'])


@pytest.mark.db
def test_a_supplier_lot_printed_on_the_bol_and_missing_from_fl_is_a_disagreement(plant):
    _, read = clean(plant)
    two_lots = good_bol(plant, lines=[{'description': plant.name('Sugar Brown'), 'quantity': 225, 'unit': 'BAGS',
                                       'lot_numbers': ['SUP-1', 'SUP-2']}])
    analysis, email = drill(plant, read, bol=two_lots)
    lots = next(f for f in analysis['forward']['comparisons'][0]['fields'] if f['field'] == 'Supplier lot')
    assert (lots['bol'], lots['fl'], lots['status']) == ('SUP-1, SUP-2', 'SUP-1', 'disagree')
    assert analysis['forward']['receiving_result'] == 'fail'
    assert 'Supplier lot: the records disagree (Sheet: not asked on the Form | BOL: SUP-1, SUP-2 | FL: SUP-1).' in email['text']


@pytest.mark.db
def test_the_backward_test_flags_problems_on_every_branch_the_shared_ingredient_reached(plant):
    chain, read = clean(plant)
    other_batch = plant.lot(chain['batch'], supplier_lot=None)
    other_cases = plant.lot(chain['cases'], supplier_lot=None)
    plant.post('make', [(other_batch, 200), (chain['sugar_lot'], -200)], day(4))
    plant.post('pack', [(other_cases, 200), (other_batch, -200)], day(4), hour=13)
    plant.void(chain['ship'])
    plant.ship(chain['case_lot'], 100, day(5), plant.name('Market'))     # the only shipped finished lot: it is picked
    plant.post('adjust', [(other_batch, 30)], day(4), hour=11, reason_code='unknown')   # the problem is on the OTHER branch
    analysis, _ = drill(plant, read)
    backward = analysis['backward']
    assert backward['lot']['lot_id'] == chain['case_lot']
    assert {other_batch, other_cases} <= {lot['lot_id'] for lot in backward['chain']['lots']}
    flag = next(f for f in backward['flags'] if f['kind'] == 'UNKNOWN_ADJUSTMENT_INTO_PRODUCT')
    assert plant.lots[other_batch] in flag['message'] and backward['result'] == 'fail'
    assert analysis['full_recall_possible'] == 'no'


@pytest.mark.db
@pytest.mark.parametrize('where', ['the batch inside it', 'the finished lot itself'])
def test_a_make_or_pack_with_no_ingredient_lots_recorded_blocks_the_backward_test(plant, where):
    batch, cases = plant.product('Batch Granola', 'batch'), plant.product('Granola 25 LB', 'finished')
    batch_lot, case_lot = plant.lot(batch, supplier_lot=None), plant.lot(cases, supplier_lot=None)
    if where == 'the batch inside it':
        made = plant.post('make', [(batch_lot, 400)], day(2))            # output only: no ingredient lines at all
        plant.post('pack', [(case_lot, 400), (batch_lot, -400)], day(2), hour=13)
    else:
        made = plant.post('pack', [(case_lot, 400)], day(2))             # packed from nothing
    plant.ship(case_lot, 100, day(3), plant.name('Market'))
    analysis, email = drill(plant, sheet())
    backward = analysis['backward']
    assert backward['lot']['lot_id'] == case_lot and backward['ingredients'] == []
    assert backward['result'] == 'fail' and analysis['full_recall_possible'] == 'no'   # nothing to check is not a pass
    flag = next(f for f in backward['flags'] if f['kind'] == 'PRODUCTION_INPUTS_MISSING')
    assert flag['blocks'] is True and f'#{made}' in flag['message'] and 'records no ingredient lots at all' in flag['message']
    assert flag['message'] in email['text']


@pytest.mark.db
def test_a_product_on_the_bol_that_fl_never_received_is_unverified_never_a_fail_on_wording(plant):
    _, read = clean(plant)
    honey = plant.product('Honey Clover')
    sugar_line = {'description': plant.name('Sugar Brown'), 'quantity': 225, 'unit': 'BAGS', 'lot_numbers': ['SUP-1']}
    honey_line = {'description': plant.name('Honey Clover'), 'quantity': 10, 'unit': 'PAILS', 'lot_numbers': []}
    analysis, email = drill(plant, read, bol=good_bol(plant, lines=[sugar_line, honey_line]))
    other = next(f for f in analysis['forward']['document_fields'] if f['field'] == 'Other products on the BOL')
    assert other['status'] == 'cannot compare' and plant.name('Honey Clover') in other['bol']
    assert 'no FL receipt for this product within a day of the Form date' in other['fl']
    assert analysis['forward']['receiving_result'] == 'unverified'       # shown and not passed; wording alone never fails
    assert 'no FL receipt for this product within a day of the Form date' in email['text']
    # The same BOL when FL did receive the honey that day, on its own receipt: accounted for.
    plant.receive(honey, 600, day(1))
    analysis, _ = drill(plant, read, bol=good_bol(plant, lines=[sugar_line, honey_line]))
    other = next(f for f in analysis['forward']['document_fields'] if f['field'] == 'Other products on the BOL')
    assert other['status'] == 'agree' and 'on FL receipt REC-' in other['fl']
    assert analysis['forward']['receiving_result'] == 'pass'
    # A line the product list does not recognise is not judged, and is not a pass either.
    odd = {'description': 'PALLET EXCHANGE 48X40 GMA', 'quantity': 4, 'unit': 'EA', 'lot_numbers': []}
    analysis, _ = drill(plant, read, bol=good_bol(plant, lines=[sugar_line, odd]))
    other = next(f for f in analysis['forward']['document_fields'] if f['field'] == 'Other products on the BOL')
    assert other['status'] == 'cannot compare' and analysis['forward']['receiving_result'] == 'unverified'


def test_reference_numbers_are_compared_whole_never_as_part_of_a_longer_number():
    agree = trial_recall._numbers_agree
    assert agree('4412', ['PO 4412']) and agree('4412', ['PO-4412']) and agree('4412', ['PO#4412']) and agree('4412', ['PO4412'])
    assert agree('0632054', ['632054']) and agree('TO306', ['to-306']) and agree('632054', ['BOL: 632054', 'PO 1'])
    assert not agree('1234', ['BOL 912345']) and not agree('1234', ['12345']) and not agree('4412', ['944120'])
    assert not agree('', ['632054']) and not agree('632054', [])
    same = trial_recall._same_reference                                  # the label may be on either side
    assert same('4412', 'PO 4412') and same('PO 4412', '4412') and same('PO-4412', 'po4412') and same('0632054', '632054')
    assert not same('PO 1', 'PO 2') and not same('1234', '912345') and not same('', '') and not same('PO', 'PO')
    assert not same('BOL-2026-001', 'BOL-2026-002') and same('BOL-2026-001', '2026-001') and same('TO306', 'to-306')
    assert not same('2026', 'BOL-2026-001')                              # a part of a number is not the number


@pytest.mark.db
def test_a_label_on_the_reference_in_fl_is_not_a_disagreement(plant):
    sugar = plant.product('Sugar Brown')
    lot, _ = plant.receive(sugar, 11250, day(1), bol='BOL 632054')       # FL wrote the label, the Form did not
    analysis, _ = drill(plant, sheet(plant.form_row(day(1), lot, ['Sugar brown-50lbs-225'])))
    reference = next(f for f in analysis['forward']['document_fields'] if f['field'] == 'BOL / reference number')
    assert (reference['sheet'], reference['fl'], reference['status']) == ('632054', 'BOL 632054', 'agree')
    assert analysis['forward']['receiving_result'] == 'pass'


@pytest.mark.db
def test_a_bol_quantity_that_cannot_all_be_added_up_is_not_confirmed(plant):
    _, read = clean(plant)
    lines = [{'description': plant.name('Sugar Brown'), 'quantity': 11250, 'unit': 'LB', 'lot_numbers': ['SUP-1']},
             {'description': plant.name('Sugar Brown'), 'quantity': 1, 'unit': 'PALLETS', 'lot_numbers': ['SUP-1']}]
    analysis, _ = drill(plant, read, bol=good_bol(plant, lines=lines))
    quantity = next(f for f in analysis['forward']['comparisons'][0]['fields'] if f['field'] == 'Quantity')
    assert quantity['bol'] == '11250 LB; 1 PALLETS' and quantity['bol_compared'] is False   # shown, not added up
    assert quantity['status'] == 'agree' and quantity['fl_compared'] is True                # Form and FL agree with each other
    assert analysis['forward']['receiving_result'] == 'unverified'       # but the BOL did not confirm the quantity


@pytest.mark.db
def test_the_supplier_is_checked_on_every_matched_receipt(plant):
    chain = plant.chain()
    honey = plant.product('Honey Clover')
    plant.cur.execute('INSERT INTO suppliers (name) VALUES (%s) RETURNING id', (plant.name('Bee Farm'),))
    bee = plant.cur.fetchone()['id']
    honey_lot, _ = plant.receive(honey, 600, chain['on'], supplier_id=bee, shipper_name=plant.name('Bee Farm'))
    read = sheet(plant.form_row(chain['on'], f'{plant.lots[chain["sugar_lot"]]}, {plant.lots[honey_lot]}',
                                ['Sugar brown-50lbs-225', 'Honey clover-60lbs-10']))
    lines = [{'description': plant.name('Sugar Brown'), 'quantity': 225, 'unit': 'BAGS', 'lot_numbers': ['SUP-1']},
             {'description': plant.name('Honey Clover'), 'quantity': 10, 'unit': 'PAILS', 'lot_numbers': ['SUP-1']}]
    analysis, _ = drill(plant, read, bol=good_bol(plant, lines=lines))     # the BOL names the sugar supplier only
    assert len(analysis['forward']['receipts']) == 2
    supplier = next(f for f in analysis['forward']['document_fields'] if f['field'] == 'Supplier')
    assert supplier['status'] == 'disagree' and plant.name('Bee Farm') in supplier['fl']
    assert analysis['forward']['receiving_result'] == 'fail'


@pytest.mark.db
def test_the_same_line_on_two_different_bol_pages_is_never_silently_dropped_or_passed(plant):
    chain = plant.chain()
    read = sheet(plant.form_row(chain['on'], chain['sugar_lot'], ['Sugar brown-50lbs-225'], photo=[PHOTO, PHOTO_2]))
    google = FakeGoogle(read['values'], files={PHOTO: {'mime_type': 'image/jpeg', 'size': 9},
                                               PHOTO_2: {'mime_type': 'image/jpeg', 'size': 9}})
    sugar = {'description': plant.name('Sugar Brown'), 'quantity': 225, 'unit': 'BAGS', 'lot_numbers': ['SUP-1']}
    pages = iter([good_bol(plant, lines=[sugar]),                                        # page 1
                  good_bol(plant, lines=[sugar, {'description': 'PALLET EXCHANGE', 'quantity': 4, 'unit': 'EA',
                                                 'lot_numbers': []}])])                    # a different page, same sugar line
    analysis, _ = drill(plant, read, google=google,
                        extractor=lambda content, mime: {'extraction': next(pages), 'extraction_model': 'test'})
    forward = analysis['forward']
    quantity = next(f for f in forward['comparisons'][0]['fields'] if f['field'] == 'Quantity')
    assert quantity['bol'] == '225 BAGS; 225 BAGS' and quantity['status'] == 'disagree'   # both are kept and shown
    assert any('Two different BOL photos on this Form row show the same line' in n for n in forward['notes'])
    assert forward['receiving_result'] != 'pass'


@pytest.mark.db
def test_supplier_lots_of_another_delivery_into_the_same_lot_are_not_a_disagreement(plant):
    chain, read = clean(plant)
    # A later truck went into the same internal lot with its own supplier lot (recorded on the lot, not the delivery).
    plant.post('receive', [(chain['sugar_lot'], 500)], day(6), bol_reference='777', supplier_id=plant.supplier,
               shipper_name=plant.supplier_name)
    plant.cur.execute("INSERT INTO lot_supplier_codes (lot_id, supplier_lot_code) VALUES (%s, 'SUP-9')", (chain['sugar_lot'],))
    analysis, _ = drill(plant, read)                                    # the first truck's BOL prints only SUP-1
    receipt = analysis['forward']['receipts'][0]
    lots = next(f for f in analysis['forward']['comparisons'][0]['fields'] if f['field'] == 'Supplier lot')
    assert receipt['receipts_into_lot'] == 2 and lots['fl'] == 'SUP-1, SUP-9' and lots['bol'] == 'SUP-1'
    assert lots['status'] == 'agree' and analysis['forward']['receiving_result'] == 'pass'
    # With one delivery only, a code in FL that the BOL does not show IS a disagreement.
    solo = Plant(plant.cur)
    chain2, read2 = clean(solo)
    solo.cur.execute("INSERT INTO lot_supplier_codes (lot_id, supplier_lot_code) VALUES (%s, 'SUP-9')", (chain2['sugar_lot'],))
    seen = {}
    for seed in range(12):
        result, _ = drill(solo, read2, seed=seed)
        row_lot = result['forward']['row']['lot_text']
        seen[row_lot] = next(f for f in result['forward']['comparisons'][0]['fields'] if f['field'] == 'Supplier lot')
    assert seen[solo.lots[chain2['sugar_lot']]]['status'] == 'disagree'


@pytest.mark.db
def test_one_shipment_carrying_two_affected_lots_lists_the_other_lot(plant):
    chain, read = clean(plant)
    other_batch, other_cases = plant.lot(chain['batch'], supplier_lot=None), plant.lot(chain['cases'], supplier_lot=None)
    plant.post('make', [(other_batch, 200), (chain['sugar_lot'], -200)], day(4))
    plant.post('pack', [(other_cases, 200), (other_batch, -200)], day(4), hour=13)
    plant.void(chain['ship'])
    # One truck, both finished lots.
    both = plant.post('ship', [(chain['case_lot'], -100), (other_cases, -150)], day(5),
                      customer_name=plant.name('Market'), bol_reference='OUT-2')
    plant.cur.execute('INSERT INTO shipments (transaction_id, shipped_at, bol_reference) VALUES (%s, now(), %s)', (both, 'OUT-2'))
    for seed in range(8):
        analysis, _ = drill(plant, read, seed=seed)
        backward = analysis['backward']
        picked = backward['lot']['lot_id']
        other = other_cases if picked == chain['case_lot'] else chain['case_lot']
        assert [(s['transaction_id'], s['lot_id']) for s in backward['own']['shipments']] == [(both, picked)]
        assert [(s['transaction_id'], s['lot_id']) for s in backward['other_shipments']] == [(both, other)]


@pytest.mark.db
def test_form_wording_that_does_not_confirm_the_product_is_unverified_never_fail(plant):
    sugar = plant.product('Sugar Brown')
    lot, tx = plant.receive(sugar, 11250, day(1))
    # Lot and date pin the one FL receipt; the Form calls the product something FL does not recognise as it.
    read = sheet(plant.form_row(day(1), lot, ['Azucar morena-50lbs-225']))
    analysis, email = drill(plant, read)
    forward = analysis['forward']
    receipt, = forward['receipts']
    assert receipt['transaction_id'] == tx and receipt['basis']['product'] == 'not_confirmed'
    product = next(f for f in forward['comparisons'][0]['fields'] if f['field'] == 'Product')
    assert product['status'] == 'cannot compare' and product['form_unconfirmed'] is True
    assert product['sheet'] == plant.name('Azucar morena') and product['fl'] == plant.name('Sugar Brown')
    assert forward['receiving_result'] == 'unverified' and forward['problems'] == []     # not a FAIL
    assert any('the product wording on the Form does not confirm it' in note for note in forward['notes'])
    assert forward['trace_result'] == 'pass'                            # the lot is still traced
    assert 'FAIL' not in email['text'].split('2. INVENTORY')[0]


@pytest.mark.db
def test_a_heic_phone_photo_is_converted_and_read_or_left_unverified(plant, monkeypatch):
    _, read = clean(plant)
    google = FakeGoogle(read['values'], files={PHOTO: {'mime_type': 'image/heic', 'size': 9}})
    seen = []

    def extractor(content, mime):
        seen.append((content, mime))
        return {'extraction': good_bol(plant), 'extraction_model': 'test-model'}

    monkeypatch.setattr(extraction, 'heic_to_jpeg', lambda content: b'jpeg-made-from-' + content)
    analysis, _ = drill(plant, read, google=google, extractor=extractor)
    document, = analysis['forward']['documents']
    assert seen == [(b'jpeg-made-from-photo-bytes', 'image/jpeg')]        # the model is given the converted JPEG
    assert document['readable'] is True and document['converted_from'] == 'image/heic'
    assert analysis['forward']['receiving_result'] == 'pass'

    def broken(content):
        raise extraction.ExtractionError('HEIC photo could not be converted (ValueError)')

    monkeypatch.setattr(extraction, 'heic_to_jpeg', broken)
    seen.clear()
    analysis, _ = drill(plant, read, google=google, extractor=extractor)
    assert seen == [] and analysis['forward']['receiving_result'] == 'unverified'
    assert 'could not be converted' in analysis['forward']['documents'][0]['problem']


def test_heic_conversion_makes_a_real_jpeg_and_fails_safely():
    with pytest.raises(extraction.ExtractionError, match='could not be converted'):
        extraction.heic_to_jpeg(b'this is not a photo')                 # also what happens when the library is absent
    pillow_heif = pytest.importorskip('pillow_heif')                    # in requirements.txt; skip where not installed
    import io
    from PIL import Image
    pillow_heif.register_heif_opener()
    original = io.BytesIO()
    Image.new('RGB', (4000, 3000), (200, 120, 40)).save(original, format='HEIF')
    converted = extraction.heic_to_jpeg(original.getvalue())
    with Image.open(io.BytesIO(converted)) as image:
        assert image.format == 'JPEG' and max(image.size) == extraction.HEIC_MAX_SIDE
        red, green, blue = image.getpixel((10, 10))
        assert abs(red - 200) < 12 and abs(green - 120) < 12 and abs(blue - 40) < 12


@pytest.mark.db
def test_receiving_problems_and_warnings_never_make_the_answer_no_but_a_broken_trail_does(plant):
    chain, read = clean(plant)
    _shipment_without_bol(plant, chain)                                  # warning
    _shared_key(plant, chain)                                            # warning
    short = good_bol(plant, lines=[{'description': plant.name('Sugar Brown'), 'quantity': 200, 'unit': 'BAGS',
                                    'lot_numbers': ['ZZ-1']}])           # quantity and supplier lot both differ
    analysis, email = drill(plant, read, bol=short)
    assert analysis['forward']['receiving_result'] == 'fail' and analysis['forward']['trace_result'] == 'pass'
    assert analysis['full_recall_possible'] == 'yes' and analysis['reasons'] == []
    assert email['subject'].endswith('Full recall possible: YES')
    assert 'Every affected customer and shipment in the chains tested can be identified from the records.' in email['text']
    _removed_by_adjustment(plant, chain)                                 # made product with no destination
    analysis, email = drill(plant, read, bol=short)
    assert analysis['full_recall_possible'] == 'no' and len(analysis['reasons']) >= 1
    assert all('removed by adjustment' in reason for reason in analysis['reasons'])
    assert 'Why not (each of these stops an affected customer or shipment being identified):' in email['text']


@pytest.mark.db
def test_a_shipment_with_no_identifiable_customer_makes_the_answer_no(plant):
    chain, read = clean(plant)
    nameless = plant.post('ship', [(chain['case_lot'], -40)], day(5), bol_reference='OUT-9')    # no customer, no order
    plant.cur.execute("INSERT INTO shipments (transaction_id, shipped_at, bol_reference) VALUES (%s, now(), 'OUT-9')", (nameless,))
    analysis, email = drill(plant, read)
    for test in (analysis['forward'], analysis['backward']):
        flag = next(f for f in test['flags'] if f['kind'] == 'SHIPMENT_CUSTOMER_UNKNOWN')
        assert flag['blocks'] is True and f'#{nameless}' in flag['message'] and 'name no customer and no order' in flag['message']
    assert analysis['full_recall_possible'] == 'no'
    assert any('name no customer and no order' in reason for reason in analysis['reasons'])
    assert 'Shipment with no identifiable customer' in email['text']


def test_bol_package_counts_are_compared_by_kind_and_never_mixed():
    quantity = trial_recall._bol_quantity
    assert quantity({'quantity': 225, 'unit': 'BAGS'}) == ('count:bag', 225)
    assert quantity({'quantity': 10, 'unit': 'Drums'}) == ('count:drum', 10)
    assert quantity({'quantity': 4, 'unit': 'ctns.'}) == ('count:case', 4) == quantity({'quantity': 4, 'unit': 'CASE'})
    assert quantity({'quantity': 500, 'unit': 'LB'}) == ('lb', 500) and quantity({'quantity': 10, 'unit': 'KG'})[0] == 'lb'
    for unit in ('PCS', 'EA', 'PALLETS', None, ''):                       # not a package count: never compared
        assert quantity({'quantity': 12, 'unit': unit}) == (None, None)
    assert quantity({'quantity': None, 'unit': 'BAGS'}) == (None, None)


@pytest.mark.db
def test_bags_on_one_line_and_drums_on_another_are_not_added_up(plant):
    _, read = clean(plant)
    lines = [{'description': plant.name('Sugar Brown'), 'quantity': 200, 'unit': 'BAGS', 'lot_numbers': ['SUP-1']},
             {'description': plant.name('Sugar Brown'), 'quantity': 25, 'unit': 'DRUMS', 'lot_numbers': ['SUP-1']}]
    analysis, _ = drill(plant, read, bol=good_bol(plant, lines=lines))    # 200 + 25 would "match" the Form's 225
    quantity = next(f for f in analysis['forward']['comparisons'][0]['fields'] if f['field'] == 'Quantity')
    assert quantity['bol'] == '200 BAGS; 25 DRUMS' and quantity['bol_compared'] is False
    assert analysis['forward']['receiving_result'] == 'unverified'


# ───────────────────────── strictly read-only on the ledger ─────────────────────────

@pytest.mark.db
def test_the_whole_drill_runs_inside_a_read_only_transaction_and_changes_nothing(plant):
    chain, read = clean(plant)
    _unknown_adjustment(plant, chain)
    _shipped_not_recorded(plant, chain)
    every = existing_tables(plant.cur) + list(NEW_TABLES)
    before = fingerprint(plant.cur, every)
    plant.cur.execute('SAVEPOINT read_only_drill')
    plant.cur.execute('SET TRANSACTION READ ONLY')                       # nothing below may write, or Postgres refuses it
    analysis, email = drill(plant, read)
    with pytest.raises(Exception, match='read-only transaction'):
        plant.cur.execute("INSERT INTO migration_markers (name) VALUES ('should-not-be-possible')")
    plant.cur.execute('ROLLBACK TO SAVEPOINT read_only_drill')
    assert analysis['forward']['result'] == 'fail' and analysis['flag_count'] >= 2
    assert fingerprint(plant.cur, every) == before
    # Saving the run is the one write, into the one table.
    run_id = trial_recall.save_run(plant.cur, started_at=RUN_AT, analysis=analysis, email=email,
                                   delivery={'status': 'disabled'})
    after = fingerprint(plant.cur, every)
    assert {table for table in every if after[table] != before[table]} == {'trial_recall_runs'}
    plant.cur.execute('SELECT * FROM trial_recall_runs WHERE id = %s', (run_id,))
    row = plant.cur.fetchone()
    assert (row['job_status'], row['forward_result'], row['backward_result'], row['full_recall_possible']) == (
        'completed', analysis['forward']['result'], analysis['backward']['result'], 'no')
    assert row['plant_date'].isoformat() == '2026-11-18' and row['cutoff_date'] == CUTOFF and row['email_status'] == 'disabled'
    assert row['result']['email_text'] == email['text'] and row['result']['forward']['row']['row_number'] == 2


def test_the_module_only_reads_posted_current_ledger_data_and_writes_one_table():
    source = (ROOT / 'trial_recall.py').read_text()
    sql = ' '.join(node.value for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.Constant) and isinstance(node.value, str))
    assert set(re.findall(r'INSERT\s+INTO\s+(\w+)', sql, re.I)) == {'trial_recall_runs'}
    assert not re.search(r'\b(UPDATE\s+\w+\s+SET|DELETE\s+FROM|TRUNCATE|DROP\s+TABLE|ALTER\s+TABLE|FOR\s+UPDATE)\b', sql, re.I)
    for name in ('trial_recall.py', 'receiving_form.py'):
        text = (ROOT / name).read_text()
        code = ' '.join(node.value for node in ast.walk(ast.parse(text))
                        if isinstance(node, ast.Constant) and isinstance(node.value, str) and 'SELECT' in node.value)
        # Lines are only ever read through the current-ledger view...
        assert not re.search(r'(?<!ledger_current_)\btransaction_lines\b', code), name
        # ...and `transactions` only beside the current-ledger view (for its newer columns), never on its own.
        assert set(re.findall(r'(?:FROM|JOIN)\s+transactions\s+(\w+)\s+ON\s+(\S+\s*=\s*\S+)', code)) <= {
            ('tx', 'tx.id = t.id'), ('t', 't.id = l.transaction_id')}, name
        assert not re.search(r'FROM\s+transactions\b', code), name
        assert code.count("effective_status = 'posted'") >= 1
    drill_source = (ROOT / 'scripts/trial_recall_drill.py').read_text()
    assert 'REPEATABLE READ READ ONLY' in drill_source and 'set_session' not in drill_source
    for forbidden in ('readonly=True', 'SESSION CHARACTERISTICS', 'default_transaction_read_only', 'PGOPTIONS'):
        assert forbidden not in drill_source + source + (ROOT / 'scripts/receiving_form_sync.py').read_text()


def test_trace_helpers_match_main():
    api = trial_recall.TraceApi()
    for moment in (datetime(2026, 11, 12, 3, 30), datetime(2026, 7, 4, 23, 59, tzinfo=timezone.utc), None):
        assert api.format_timestamp(moment) == main.format_timestamp(moment)
        assert api._trace_stamp('received', moment) == main._trace_stamp('received', moment)
    for uom in (None, '', 'lb', 'LBS', '25 lb case', 'case', 'each', 'pounds', 'roll'):
        assert api.ledger_quantity_unit(uom) == main.ledger_quantity_unit(uom)
    for name in (None, '', 'Found', 'physical count', 'Blender Warehouse', 'Unknown', 'Star Snacks', 'n/a'):
        assert api._trace_shipper_kind(name) == main._trace_shipper_kind(name)
    assert trial_recall.PLANT_TIMEZONE == main.PLANT_TIMEZONE and trial_recall.CUTOFF.isoformat() == '2026-11-11'
    assert set(trial_recall.FLAGS) >= {'MADE_REMOVED_BY_ADJUSTMENT', 'ORDER_SHIPPED_NOT_RECORDED', 'SHIPMENT_NO_BOL',
                                      'UNKNOWN_ADJUSTMENT_INTO_PRODUCT', 'SHARED_KEY_POSTING', 'MISSING_SUPPLIER_LOT',
                                      'NEGATIVE_BALANCE', 'LOT_ATTRIBUTION_UNRESOLVED'}


# ───────────────────────── schedule, email, saved runs ─────────────────────────

def test_the_drill_runs_monday_wednesday_friday_plant_time():
    at = lambda day_of_month, hour_utc: datetime(2026, 11, day_of_month, hour_utc, 30, tzinfo=timezone.utc)
    # The cron fires 07:30 UTC = 02:30 Eastern in winter.
    assert [trial_recall.is_recall_day(at(d, 7)) for d in range(16, 23)] == [True, False, True, False, True, False, False]
    assert trial_recall.is_recall_day(at(17, 3)) is True                 # Tue 03:30 UTC is still Monday evening Eastern
    assert trial_recall.is_recall_day(datetime(2026, 7, 6, 7, 30, tzinfo=timezone.utc)) is True   # summer: 03:30 Eastern Monday


@pytest.mark.db
def test_email_is_off_by_default_and_uses_the_reconciliation_email_setup(plant):
    _, read = clean(plant)
    analysis, email = drill(plant, read)
    sent = []

    def sender(message, env):
        sent.append((message, env))
        return {'status': 'sent', 'to': env['ALERT_EMAIL_TO'], 'provider_id': 'email-9', 'error': None}

    assert trial_recall.deliver(email, env={'ALERT_EMAIL_TO': 'o@example.test', 'RESEND_API_KEY': 'k'}, sender=sender) == {
        'status': 'disabled'} and sent == []
    on = {'TRIAL_RECALL_EMAIL_ENABLED': '1', 'ALERT_EMAIL_TO': 'o@example.test', 'RESEND_API_KEY': 'k'}
    delivery = trial_recall.deliver(email, env=on, sender=sender)
    assert delivery['status'] == 'sent' and sent[0][0]['subject'] == email['subject']
    assert trial_recall.deliver(email, env=on, sender=lambda *a: 1 / 0)['status'] == 'failed'
    missing = trial_recall.deliver(email, env={'TRIAL_RECALL_EMAIL_ENABLED': '1'})    # the real sender, nothing configured
    assert missing['status'] == 'failed' and missing['error'] == 'ALERT_EMAIL_TO or RESEND_API_KEY is not set'
    assert trial_recall.deliver.__defaults__ == (None, None) and reconciliation.send_email is not None
    run_id = trial_recall.save_run(plant.cur, started_at=RUN_AT, analysis=analysis, email=email, delivery=delivery)
    plant.cur.execute('SELECT email_status, email_to, email_sent_at, email_provider_id FROM trial_recall_runs WHERE id = %s',
                      (run_id,))
    row = plant.cur.fetchone()
    assert (row['email_status'], row['email_to'], row['email_provider_id']) == ('sent', 'o@example.test', 'email-9')
    assert row['email_sent_at'] is not None
    assert trial_recall.ran_today(plant.cur, RUN_AT) is True
    assert trial_recall.ran_today(plant.cur, RUN_AT + timedelta(days=2)) is False
    trial_recall.record_failed_run(plant.cur, 'AUTH_FAILED', RUN_AT + timedelta(days=2))
    assert trial_recall.ran_today(plant.cur, RUN_AT + timedelta(days=2)) is False   # a failed drill may run again that day


@pytest.mark.db
def test_saved_runs_are_readable_by_owner_and_office_only(client, plant, actors, monkeypatch):
    key = lambda who: {'X-API-Key': actors[who]['key']}
    _, read = clean(plant)
    analysis, email = drill(plant, read)
    run_id = trial_recall.save_run(plant.cur, started_at=RUN_AT, analysis=analysis, email=email,
                                   delivery={'status': 'disabled'})
    for who in ('owner', 'office'):
        listed = client.get('/verification/trial-recall/runs?limit=1', headers=key(who))
        assert listed.status_code == 200, listed.text
        assert listed.json()['runs'][0]['id'] == run_id and 'result' not in listed.json()['runs'][0]
        detail = client.get(f'/verification/trial-recall/runs/{run_id}', headers=key(who)).json()['run']
        assert detail['full_recall_possible'] == 'yes' and detail['result']['email_subject'] == email['subject']
    assert client.get('/verification/trial-recall/runs/999999999', headers=key('owner')).status_code == 404
    assert client.get('/verification/trial-recall/runs', headers={'X-API-Key': main.API_KEY}).status_code == 200
    for path in ('runs', f'runs/{run_id}'):
        assert client.get(f'/verification/trial-recall/{path}', headers=key('floor')).status_code == 403
        assert client.get(f'/verification/trial-recall/{path}').status_code == 401
        if main.DASHBOARD_API_KEY:
            assert client.get(f'/verification/trial-recall/{path}',
                              headers={'X-API-Key': main.DASHBOARD_API_KEY}).status_code == 403
    for method in ('post', 'put', 'patch', 'delete'):
        assert getattr(client, method)('/verification/trial-recall/runs', headers=key('owner')).status_code == 405
    monkeypatch.setattr(receiving_form, 'migration_applied', lambda cur: False)
    assert client.get('/verification/trial-recall/runs', headers=key('owner')).status_code == 503
    served = {(m, r.path) for r in main.app.routes for m in getattr(r, 'methods', ()) if 'trial-recall' in r.path}
    assert served == set(trial_recall.ROUTES) and trial_recall.ROUTES <= main.ACTOR_WRITE_ALLOWLIST
    assert not trial_recall.ROUTES & (main.DASHBOARD_KEY_ALLOWLIST | main.READONLY_KEY_ALLOWLIST | main.SELECTION_ALLOWLIST)


# ───────────────────────── the BOL reader and the command ─────────────────────────

def test_the_bol_reader_uses_the_existing_extraction_model_and_never_guesses(monkeypatch):
    asked = {}

    class Block:
        type = 'tool_use'

        def __init__(self, value):
            self.input = value

    class Messages:
        def __init__(self, value):
            self.value = value

        def create(self, **kwargs):
            asked.update(kwargs)
            return type('Message', (), {'content': [Block(self.value)]})()

    def answer(value):
        monkeypatch.setattr(extraction, '_client', lambda: type('Client', (), {'messages': Messages(value)})())

    read = {'readable': True, 'supplier_name': 'Jack’s Eggs', 'manufacturer_name': None, 'carrier_name': None,
            'bol_number': '632054', 'reference_numbers': ['PO 1'], 'document_date': '2026-11-12',
            'lines': [{'description': 'SUGAR 6X 50#', 'quantity': 225, 'unit': 'BAGS', 'lot_numbers': ['L1']}]}
    answer(read)
    monkeypatch.delenv('EXTRACTION_MODEL', raising=False)
    out = extraction.extract_bol(b'jpeg', 'image/jpeg')
    assert out['extraction'] == read and out['extraction_model'] == extraction.DEFAULT_EXTRACTION_MODEL
    assert asked['model'] == extraction.DEFAULT_EXTRACTION_MODEL                    # the same model as document intake
    assert asked['tool_choice'] == {'type': 'tool', 'name': 'record_bill_of_lading'}
    assert 'NEVER guess' in extraction.BOL_EXTRACTION_PROMPT
    answer(dict(read, readable=False, lines=[]))
    assert extraction.extract_bol(b'jpeg', 'image/png')['extraction']['readable'] is False   # the caller's UNVERIFIED
    for bad in (dict(read, document_date='Nov 12'), dict(read, lines=[{'description': 'x', 'quantity': -1,
                                                                        'unit': None, 'lot_numbers': []}])):
        answer(bad)
        with pytest.raises(extraction.ExtractionError):
            extraction.extract_bol(b'jpeg', 'image/jpeg')
    with pytest.raises(extraction.ExtractionError):
        extraction.extract_bol(b'heic', 'image/heic')
    # The document-intake function cannot be asked for a BOL.
    with pytest.raises(extraction.ExtractionError, match='Unknown document kind'):
        extraction.extract_purchase_document(b'x', 'image/png', kind='bol')
    assert set(extraction._KIND_CONFIG) == {'purchase', 'sales'}


@pytest.mark.db
def test_the_command_reads_the_books_in_real_read_only_transactions(_db_connection):
    """The command's own connections, against the committed (empty) test database: every statement of the
    drill is accepted by Postgres inside REPEATABLE READ READ ONLY transactions."""
    url = os.environ['DATABASE_URL']
    read = FakeGoogle(sheet()['values'])
    started, analysis, email = trial_recall_drill.drill(url, client=read, now=RUN_AT, rng=random.Random(1))
    assert started == RUN_AT and read.calls[0][0] == 'sheet'
    assert analysis['forward']['result'] == 'no_eligible_lot'
    assert 'Trial recall, Wed Nov 18, 2026.' in email['text']
    with pytest.raises(Exception, match='read-only transaction'):
        trial_recall_drill.read_only(url, lambda cur: cur.execute("INSERT INTO migration_markers (name) VALUES ('x')"))
