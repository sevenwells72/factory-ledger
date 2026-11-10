"""Daily inventory reconciliation (reconciliation.py, scripts/daily_reconciliation.py, migration 094).

The 14 tests of docs/verification/daily-reconciliation-scoping.md §5 are numbered
test_01 … test_14; the rest pin Michael's build rules (tolerance on the day's
movement, always-listed items, coconut yield, the email switch) and the wiring.

Everything runs against the LOCAL test database inside the usual rolled-back
transaction. Migration 094 is applied inside that transaction, so the shared
test database never keeps it. Entry times are set explicitly (the created_at
triggers are switched off inside the transaction, as other suites do) so a
"night" is a fixed moment, not the wall clock.
"""
import ast
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import re
from uuid import uuid4

import psycopg2
import pytest
import yaml

import main
import reconciliation
from scripts import daily_reconciliation, exceptions_sweep
from tests.test_actor_attribution import actors, client  # noqa: F401

pytestmark = pytest.mark.db
ROOT = Path(__file__).resolve().parents[1]
MIGRATION = (ROOT / 'migrations/094_daily_reconciliation.sql').read_text()
DOWN = (ROOT / 'migrations/down/094_daily_reconciliation_down.sql').read_text()
FIXTURES = ROOT / 'tests/fixtures/reconciliation'
NEW_TABLES = ('inventory_snapshots', 'reconciliation_runs', 'reconciliation_flags')
ET = reconciliation.PLANT_TIMEZONE
EMAIL_ON = {'RECONCILIATION_EMAIL_ENABLED': '1', 'ALERT_EMAIL_TO': 'owner@example.test', 'RESEND_API_KEY': 're_test'}


def at(day, hour, minute=0):
    """A moment on plant day `day` (0 = Tue 2026-03-10) in Eastern time."""
    return datetime(2026, 3, 10, hour, minute, tzinfo=ET) + timedelta(days=day)


def night(day):
    """The 02:30 run that checks plant day `day`."""
    return at(day + 1, 2, 30)


class Kit:
    """A tiny plant: stock, one granola recipe, and helpers that post like the app does."""

    def __init__(self, cur):
        self.cur, self.sent, self.lots, self.txs = cur, [], set(), set()
        self.tag = 'RCN' + uuid4().hex[:6].upper()
        cur.execute(MIGRATION)
        for table, trigger in (('transactions', 'trg_transactions_created_at'),
                               ('transaction_lines', 'trg_transaction_lines_created_at'),
                               ('ledger_corrections', 'trg_ledger_corrections_created_at')):
            cur.execute(f'ALTER TABLE {table} DISABLE TRIGGER {trigger}')
        cur.execute("INSERT INTO actors (name, role, key_hash) VALUES (%s, 'floor', %s) RETURNING id",
                    (f'{self.tag} Worker', uuid4().hex * 2))
        self.actor = cur.fetchone()['id']
        self.oats, self.honey, self.nuts, self.water = (self.product(n) for n in ('Oats', 'Honey', 'Nuts', 'Water'))
        self.granola = self.product('Batch Granola', 'batch', batch_lb=100)
        self.bagged = self.product('Granola 25 LB', 'finished')
        self.recipe(self.granola, {self.oats: 60, self.honey: 30, self.nuts: 10})
        self.recipe(self.granola, {self.water: 5}, excluded=True)
        self.oats_lot, self.honey_lot, self.nuts_lot = (self.lot(p) for p in (self.oats, self.honey, self.nuts))
        self.post('receive', [(self.oats_lot, 5000), (self.honey_lot, 2000), (self.nuts_lot, 1000)], at(-1, 10))

    def product(self, name, type_='ingredient', batch_lb=None):
        self.cur.execute("INSERT INTO products (name, type, uom, default_batch_lb) VALUES (%s, %s, 'lb', %s) RETURNING id",
                         (f'{self.tag} {name}', type_, batch_lb))
        return self.cur.fetchone()['id']

    def recipe(self, batch, ingredients, excluded=False):
        for ingredient, quantity in ingredients.items():
            self.cur.execute('INSERT INTO batch_formulas (product_id, ingredient_product_id, quantity_lb, '
                             'exclude_from_inventory) VALUES (%s, %s, %s, %s)', (batch, ingredient, quantity, excluded))

    def lot(self, product):
        self.cur.execute('INSERT INTO lots (product_id, lot_code) VALUES (%s, %s) RETURNING id',
                         (product, f'{self.tag}-{len(self.lots) + 1:03d}'))
        lot = self.cur.fetchone()['id']
        self.lots.add(lot)
        return lot

    def post(self, type_, lines, when, *, business_date=None, named=True, reason_code=None):
        self.cur.execute("""INSERT INTO transactions (type, timestamp, status, occurred_at, business_date, created_at,
                                                      operator_id, entered_by_actor_id, reason_code)
                            VALUES (%s, %s, 'posted', %s, %s, %s, %s, %s, %s) RETURNING id""",
                         (type_, when.replace(tzinfo=None), when, business_date or when.astimezone(ET).date(), when,
                          f'{self.tag} Worker' if named else 'legacy-shared-key', self.actor if named else None,
                          reason_code))
        tx = self.cur.fetchone()['id']
        self.txs.add(tx)
        for lot, quantity in lines:
            self.cur.execute('SELECT product_id FROM lots WHERE id = %s', (lot,))
            self.cur.execute('INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb, created_at) '
                             'VALUES (%s, %s, %s, %s, %s)', (tx, self.cur.fetchone()['product_id'], lot, quantity, when))
        return tx

    def make(self, batches, when, *, into=None, **used):
        """A make of `batches` granola batches; keyword overrides the pounds of an ingredient."""
        into = into or self.lot(self.granola)
        lines = [(into, 100 * batches)]
        for name, per_batch in (('oats', 60), ('honey', 30), ('nuts', 10)):
            quantity = used.get(name, per_batch * batches)
            if quantity:
                lines.append((getattr(self, f'{name}_lot'), -quantity))
        return self.post('make', lines, when), into

    def void(self, tx, when, reason='Entered by mistake'):
        self.cur.execute("""INSERT INTO ledger_corrections (target_table, target_id, event_type, previous_values,
                                                            replacement_values, reason, operator_id, created_at)
                            VALUES ('transactions', %s, 'void', '{"status": "posted"}', '{"status": "voided"}',
                                    %s, 'legacy-shared-key', %s)""", (tx, reason, when))

    def sender(self, email, env):
        self.sent.append(email)
        return {'status': 'sent', 'to': env['ALERT_EMAIL_TO'], 'provider_id': 'email-1', 'error': None}

    def run(self, when, *, env=EMAIL_ON, sender=None, dry_run=False):
        outcome = reconciliation.run_job(self.cur, now=when, env=env, sender=sender or self.sender, dry_run=dry_run)
        self.cur.execute('SET CONSTRAINTS ALL IMMEDIATE')       # the deferred run references must hold, as at COMMIT
        self.cur.execute('SET CONSTRAINTS ALL DEFERRED')
        return outcome

    def flags(self, outcome, kind=None):
        """This test's flags only (the shared test database may hold other rows)."""
        return [f for f in outcome['analysis']['flags']
                if (f['lot_id'] in self.lots or f['transaction_id'] in self.txs) and kind in (None, f['kind'])]

    def kinds(self, outcome):
        return sorted({f['kind'] for f in self.flags(outcome)})

    def row(self, run_id):
        self.cur.execute('SELECT * FROM reconciliation_runs WHERE id = %s', (run_id,))
        return self.cur.fetchone()


@pytest.fixture
def kit(db_cursor):
    return Kit(db_cursor)


def _fingerprint(cur, tables):
    """Row count and content hash of each table."""
    out = {}
    for table in tables:
        cur.execute(f'SELECT count(*) AS n, md5(COALESCE(string_agg(t::text, \'|\' ORDER BY t::text), \'\')) AS h '
                    f'FROM {table} t')
        out[table] = tuple(cur.fetchone().values())
    return out


# ───────────────────────── the 14 planned tests ─────────────────────────

def test_01_snapshot_equals_the_view_and_excludes_voided(kit):
    mistake = kit.post('receive', [(kit.oats_lot, 777)], at(-1, 11))
    kit.void(mistake, at(-1, 11, 5))
    kit.cur.execute("INSERT INTO transactions (type, timestamp, status, occurred_at, business_date, created_at) "
                    "VALUES ('receive', %s, 'voided', %s, %s, %s) RETURNING id",
                    (at(-1, 12).replace(tzinfo=None), at(-1, 12), at(-1, 12).date(), at(-1, 12)))
    kit.cur.execute('INSERT INTO transaction_lines (transaction_id, product_id, lot_id, quantity_lb) VALUES (%s, %s, %s, 999)',
                    (kit.cur.fetchone()['id'], kit.honey, kit.honey_lot))
    run_id = kit.run(night(-1))['run_id']
    kit.cur.execute('SELECT lot_id, product_id, quantity_lb FROM inventory_snapshots WHERE job_run_id = %s', (run_id,))
    snapshot = {r['lot_id']: (r['product_id'], float(r['quantity_lb'])) for r in kit.cur.fetchall()}
    kit.cur.execute('SELECT lot_id, product_id, quantity_on_hand FROM v_lot_quantities WHERE quantity_on_hand <> 0')
    view = {r['lot_id']: (r['product_id'], float(r['quantity_on_hand'])) for r in kit.cur.fetchall()}
    assert snapshot == view
    assert snapshot[kit.oats_lot] == (kit.oats, 5000.0) and snapshot[kit.honey_lot] == (kit.honey, 2000.0)
    # and the timestamp rebuild used for replays agrees with the view
    # (as of now, so rows other tests left in a re-used database are in both)
    as_of = datetime.now(ET) + timedelta(days=1)
    assert dict(map(tuple, reconciliation.balances_at(kit.cur, as_of))) == {k: v[1] for k, v in view.items()}


def test_02_clean_day_has_zero_flags_and_still_sends_all_clear(kit):
    kit.run(night(-1))
    kit.post('receive', [(kit.oats_lot, 1500)], at(0, 9))
    _, batch = kit.make(3, at(0, 10))
    finished = kit.lot(kit.bagged)
    kit.post('pack', [(batch, -300), (finished, 300)], at(0, 13))
    kit.post('ship', [(finished, -250)], at(0, 15))
    outcome = kit.run(night(0))
    assert kit.flags(outcome) == []
    analysis = outcome['analysis']
    assert analysis['result'] == 'all_clear' and analysis['balance_checked']
    assert (analysis['entries_checked'], analysis['makes_checked'], analysis['lots_checked']) == (4, 1, 5)
    assert len(kit.sent) == 2 and 'all clear' in kit.sent[1]['subject'] and 'ALL CLEAR' in kit.sent[1]['text']
    row = kit.row(outcome['run_id'])
    assert (row['job_status'], row['result'], row['email_status']) == ('completed', 'all_clear', 'sent')
    assert row['plant_date'] == date(2026, 3, 10) and row['email_sent_at'] is not None and row['flag_count'] == 0


def test_03_make_tolerance_is_two_percent_with_a_five_pound_floor(kit):
    kit.run(night(-1))
    over, _ = kit.make(10, at(0, 9), oats=618)          # 600 expected: +18 lb = 3%
    kit.make(10, at(0, 10), oats=606)                   # +6 lb = 1%
    kit.make(10, at(0, 11), nuts=104)                   # 100 expected: +4 lb = 4%, under the 5 lb floor
    outcome = kit.run(night(0))
    flags = kit.flags(outcome)
    assert [(f['kind'], f['transaction_id'], f['product_id']) for f in flags] == [('MAKE_OFF_RECIPE', over, kit.oats)]
    assert (flags[0]['expected_lb'], flags[0]['actual_lb'], flags[0]['difference_lb'], flags[0]['tolerance_lb']) \
        == (600, 618, 18, 12)
    # batch_formulas keeps no history: the quantities the checker used are stored with the run
    stored = next(m for m in kit.row(outcome['run_id'])['summary']['makes'] if m['transaction_id'] == over)
    assert stored['batches'] == 10 and {i['product_id']: (i['expected_lb'], i['actual_lb']) for i in stored['ingredients']} \
        == {kit.oats: (600, 618), kit.honey: (300, 300), kit.nuts: (100, 100)}


def test_04_recorded_substitution_is_correct_and_an_unrecorded_one_is_flagged(kit):
    bran = kit.product('Oat Bran')
    ss = kit.product('Batch SS Original', 'batch', batch_lb=100)
    kit.recipe(ss, {kit.oats: 72, bran: 28})
    kit.run(night(-1))
    lines = lambda: [(kit.lot(ss), 400), (kit.oats_lot, -400)]          # no bran: 112 lb extra oats
    recorded = kit.post('make', lines(), at(0, 9))
    kit.cur.execute("""INSERT INTO transaction_substitutions (transaction_id, ingredient_product_id,
                           substitute_product_id, lot_id, reason_code) VALUES (%s, %s, %s, %s, 'oat_bran_auto_substitute')""",
                    (recorded, bran, kit.oats, kit.oats_lot))
    unrecorded = kit.post('make', lines(), at(0, 10))
    outcome = kit.run(night(0))
    assert [f for f in kit.flags(outcome) if f['transaction_id'] == recorded] == []
    by_kind = {f['kind']: f for f in kit.flags(outcome) if f['transaction_id'] == unrecorded}
    assert sorted(by_kind) == ['MAKE_INGREDIENT_MISSING', 'MAKE_OFF_RECIPE']
    missing = by_kind['MAKE_INGREDIENT_MISSING']
    assert (missing['product_id'], missing['expected_lb'], missing['actual_lb']) == (bran, 112, 0)
    assert 'no substitution was recorded' in missing['detail']['message']
    # a recorded "not used" (no substitute) is also correct
    batches, expected = reconciliation.expected_usage(
        400, 100, [{'ingredient_product_id': 1, 'quantity_lb': 72, 'exclude_from_inventory': False},
                   {'ingredient_product_id': 2, 'quantity_lb': 28, 'exclude_from_inventory': False}],
        {1: 288.0}, [{'ingredient_product_id': 2, 'substitute_product_id': None}])
    assert (batches, expected) == (4, {1: 288, 2: 0})


def test_05_water_is_skipped_and_a_batch_can_be_an_ingredient(kit):
    raisins = kit.product('Raisins')
    raisin_lot = kit.lot(raisins)
    kit.post('receive', [(raisin_lot, 500)], at(-1, 10))
    fruit_nut = kit.product('Batch Granola Fruit Nut', 'batch', batch_lb=326.3)
    kit.recipe(fruit_nut, {kit.granola: 276.3, raisins: 50})            # Batch Classic #9 inside Fruit Nut
    kit.run(night(-1))
    _, classic = kit.make(6, at(0, 9))                                  # recipe has a water line; none is deducted
    kit.post('make', [(kit.lot(fruit_nut), 652.6), (classic, -552.6), (raisin_lot, -100)], at(0, 11))
    outcome = kit.run(night(0))
    assert kit.flags(outcome) == [] and outcome['analysis']['makes_checked'] == 2
    fruit = next(m for m in outcome['analysis']['makes'] if m['product_id'] == fruit_nut)
    assert fruit['batches'] == 2 and {i['product_id'] for i in fruit['ingredients']} == {kit.granola, raisins}
    classic_make = next(m for m in outcome['analysis']['makes'] if m['product_id'] == kit.granola)
    assert kit.water not in {i['product_id'] for i in classic_make['ingredients']}


def test_06_voids_are_listed_and_the_balance_still_reconciles(kit):
    earlier = kit.post('receive', [(kit.honey_lot, 400)], at(-1, 11))
    kit.run(night(-1))
    kit.void(earlier, at(0, 9), reason='Wrong supplier lot')
    same_day = kit.post('adjust', [(kit.nuts_lot, 375)], at(0, 9, 23), named=False, reason_code='unknown')
    kit.void(same_day, at(0, 9, 25))
    outcome = kit.run(night(0))
    assert kit.kinds(outcome) == ['VOID_OF_EARLIER_ENTRY', 'VOID_SAME_DAY']     # no BALANCE_DIFFERENCE
    old = kit.flags(outcome, 'VOID_OF_EARLIER_ENTRY')[0]
    assert old['transaction_id'] == earlier and old['severity'] == 'info' and 'Wrong supplier lot' in old['detail']['message']
    new = kit.flags(outcome, 'VOID_SAME_DAY')[0]
    assert new['transaction_id'] == same_day and 'No effect on balances' in new['detail']['message']
    assert outcome['analysis']['result'] == 'all_clear'         # information only; a voided adjustment is not listed twice
    kit.cur.execute('SELECT quantity_lb FROM inventory_snapshots WHERE job_run_id = %s AND lot_id = %s',
                    (outcome['run_id'], kit.honey_lot))
    assert float(kit.cur.fetchone()['quantity_lb']) == 2000


def test_07_backdated_entry_is_listed_as_late(kit):
    kit.run(night(-1))
    late = kit.post('receive', [(kit.oats_lot, 100)], at(0, 10), business_date=date(2026, 3, 6))
    kit.post('receive', [(kit.oats_lot, 100)], at(0, 11))
    outcome = kit.run(night(0))
    assert [(f['kind'], f['transaction_id'], f['severity']) for f in kit.flags(outcome)] == [('LATE_ENTRY', late, 'info')]
    assert 'business date Mar 6' in kit.flags(outcome)[0]['detail']['message']


def test_08_pack_provenance_and_lots_below_zero(kit):
    kit.run(night(-1))
    finished = kit.lot(kit.bagged)
    orphan = kit.lot(kit.granola)                                       # never made
    kit.post('adjust', [(orphan, 200)], at(0, 8), reason_code='physical_count')
    no_make = kit.post('pack', [(orphan, -200), (finished, 200)], at(0, 9))
    _, short = kit.make(3.5, at(0, 10))                                 # 350 lb made
    kit.post('adjust', [(short, 25)], at(0, 10, 30), named=False, reason_code='unknown')
    exceeds = kit.post('pack', [(short, -375), (finished, 375)], at(0, 11))     # +25 lb = 7.1%
    _, close = kit.make(10, at(0, 12))                                  # 1,000 lb made
    kit.post('pack', [(close, -1010), (finished, 1010)], at(0, 13))             # +10 lb = 1%: no alarm, but below zero
    outcome = kit.run(night(0))
    found = {(f['kind'], f['lot_id']): f for f in kit.flags(outcome)}
    assert sorted(found) == sorted([('PACK_NO_MAKE', orphan), ('PACK_EXCEEDS_MAKE', short), ('NEGATIVE_LOT', close),
                                    ('UNEXPLAINED_ADJUSTMENT', short)])
    assert found[('PACK_NO_MAKE', orphan)]['transaction_id'] == no_make
    over = found[('PACK_EXCEEDS_MAKE', short)]
    assert (over['transaction_id'], over['expected_lb'], over['actual_lb'], over['difference_lb']) == (exceeds, 350, 375, 25)
    assert 'reason "unknown"' in over['detail']['message'] and 'shared key' in over['detail']['message']
    negative = found[('NEGATIVE_LOT', close)]
    assert (negative['severity'], negative['actual_lb']) == ('alarm', -10)
    assert outcome['analysis']['result'] == 'attention'


def test_09_missed_night_reconciles_across_the_gap(kit):
    baseline = kit.run(night(-1))['run_id']
    kit.make(3, at(0, 10))
    kit.post('receive', [(kit.honey_lot, 800)], at(1, 9))                # night(0) never ran
    kit.make(2, at(1, 11))
    outcome = kit.run(night(1))
    assert kit.flags(outcome) == []
    row = kit.row(outcome['run_id'])
    assert row['previous_run_id'] == baseline and row['window_start'] == night(-1) and row['window_end'] == night(1)
    assert (row['plant_date'], row['makes_checked'], row['balance_checked']) == (date(2026, 3, 11), 2, True)


def test_10_rerun_for_the_same_night_does_not_send_a_second_email(kit):
    baseline = kit.run(night(-1))['run_id']
    kit.make(3, at(0, 10))
    first = kit.run(night(0))
    second = kit.run(night(0) + timedelta(minutes=5))
    assert len(kit.sent) == 2                                           # the baseline night and ONE for this night
    assert (first['email_status'], second['email_status']) == ('sent', 'already_sent')
    rows = [kit.row(first['run_id']), kit.row(second['run_id'])]
    assert [r['previous_run_id'] for r in rows] == [baseline, baseline]  # the baseline never moves within a night
    assert rows[1]['email_sent_at'] is None and rows[1]['result'] == rows[0]['result'] == 'all_clear'


def test_11_email_failure_keeps_the_flags_and_records_the_error(kit):
    kit.run(night(-1))
    kit.post('adjust', [(kit.oats_lot, -40)], at(0, 9), named=False, reason_code='unknown')

    def broken(email, env):
        return {'status': 'failed', 'to': env['ALERT_EMAIL_TO'], 'provider_id': None,
                'error': 'Email service answered 503: unavailable'}
    outcome = kit.run(night(0), sender=broken)
    row = kit.row(outcome['run_id'])
    assert (row['job_status'], row['result'], row['email_status']) == ('completed', 'attention', 'failed')
    assert row['email_error'] == 'Email service answered 503: unavailable' and row['email_sent_at'] is None
    kit.cur.execute('SELECT kind FROM reconciliation_flags WHERE run_id = %s', (outcome['run_id'],))
    assert 'UNEXPLAINED_ADJUSTMENT' in [r['kind'] for r in kit.cur.fetchall()]
    retry = kit.run(night(0) + timedelta(minutes=10))                   # a failed send does not block a retry
    assert retry['email_status'] == 'sent' and len(kit.sent) == 2


def _broken(email, env):
    return {'status': 'failed', 'to': env['ALERT_EMAIL_TO'], 'provider_id': None,
            'error': 'Email service answered 503: unavailable'}


def test_a_failed_send_is_covered_and_named_in_the_next_email(kit):
    baseline = kit.run(night(-1))['run_id']
    adjust = kit.post('adjust', [(kit.oats_lot, -40)], at(0, 9), named=False, reason_code='unknown')
    failed = kit.run(night(0), sender=_broken)
    assert failed['email_status'] == 'failed' and len(kit.sent) == 1
    kit.make(2, at(1, 10))
    outcome = kit.run(night(1))                                         # nobody re-ran the failed night
    assert outcome['email_status'] == 'sent' and len(kit.sent) == 2
    assert [f['transaction_id'] for f in kit.flags(outcome, 'UNEXPLAINED_ADJUSTMENT')] == [adjust]
    assert kit.flags(outcome, 'BALANCE_DIFFERENCE') == []               # rolled forward from the delivered snapshot
    text = kit.sent[-1]['text']
    assert 'NOTE: the email for Tue Mar 10 was not delivered. This email covers that day as well.' in text
    row = kit.row(outcome['run_id'])
    assert row['previous_run_id'] == baseline and row['window_start'] == night(-1)
    assert [u['run_id'] for u in row['summary']['undelivered']] == [failed['run_id']]
    after = kit.run(night(2))                                           # delivered: the baseline moves on, no note
    assert kit.row(after['run_id'])['previous_run_id'] == outcome['run_id']
    assert 'NOTE:' not in kit.sent[-1]['text'] and kit.flags(after, 'UNEXPLAINED_ADJUSTMENT') == []


def test_an_extra_daytime_run_cannot_swallow_its_flags(kit):
    kit.run(night(-1))
    nightly = kit.run(night(0))
    adjust = kit.post('adjust', [(kit.oats_lot, -40)], at(1, 9), named=False, reason_code='unknown')
    manual = kit.run(at(1, 14))                                         # same plant day as the nightly: no email
    assert manual['email_status'] == 'already_sent' and len(kit.sent) == 2
    assert [f['transaction_id'] for f in kit.flags(manual, 'UNEXPLAINED_ADJUSTMENT')] == [adjust]
    kit.make(2, at(1, 15))
    outcome = kit.run(night(1))
    assert outcome['email_status'] == 'sent' and len(kit.sent) == 3
    assert [f['transaction_id'] for f in kit.flags(outcome, 'UNEXPLAINED_ADJUSTMENT')] == [adjust]
    assert kit.flags(outcome, 'BALANCE_DIFFERENCE') == []
    note = [line for line in kit.sent[-1]['text'].splitlines() if line.startswith('NOTE:')]
    assert len(note) == 1 and note[0].startswith('NOTE: an extra check was saved ')
    assert note[0].endswith(' without an email. This email covers everything it found.')
    row = kit.row(outcome['run_id'])
    assert row['previous_run_id'] == nightly['run_id'] and row['window_start'] == night(0)


def test_a_repeat_run_that_found_nothing_new_gets_no_note(kit):
    kit.run(night(-1))
    kit.post('adjust', [(kit.oats_lot, -40)], at(0, 9), named=False, reason_code='unknown')
    sent = kit.run(night(0))
    repeat = kit.run(night(0) + timedelta(minutes=5))                   # same findings, no second email
    assert (sent['email_status'], repeat['email_status']) == ('sent', 'already_sent')
    assert len(kit.flags(repeat, 'UNEXPLAINED_ADJUSTMENT')) == 1
    quiet = kit.run(night(1))
    assert 'NOTE:' not in kit.sent[-1]['text'] and kit.row(quiet['run_id'])['summary']['undelivered'] == []
    assert kit.row(quiet['run_id'])['previous_run_id'] == sent['run_id']

    late = kit.post('adjust', [(kit.oats_lot, -15)], night(1) + timedelta(minutes=2), named=False, reason_code='unknown')
    again = kit.run(night(1) + timedelta(minutes=5))                    # this repeat DID find something new
    assert again['email_status'] == 'already_sent'
    outcome = kit.run(night(2))
    assert [f['transaction_id'] for f in kit.flags(outcome, 'UNEXPLAINED_ADJUSTMENT')] == [late]
    assert kit.sent[-1]['text'].count('NOTE: an extra check was saved ') == 1
    assert [u['run_id'] for u in kit.row(outcome['run_id'])['summary']['undelivered']] == [again['run_id']]


def _utc(year, month, day, hour, minute=0):
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


@pytest.mark.parametrize('day,hours,start,end,entries', [
    # Sun 2026-11-01, clocks go back: 25 hours, and 1:30 AM happens twice.
    (date(2026, 11, 1), 25, _utc(2026, 11, 1, 4), _utc(2026, 11, 2, 5), [
        (_utc(2026, 11, 1, 3, 30), False),      # Oct 31 11:30 PM EDT
        (_utc(2026, 11, 1, 4, 30), True),       # Nov 1 12:30 AM EDT
        (_utc(2026, 11, 1, 5, 30), True),       # Nov 1 1:30 AM EDT (first time)
        (_utc(2026, 11, 1, 6, 30), True),       # Nov 1 1:30 AM EST (second time)
        (_utc(2026, 11, 2, 4, 30), True),       # Nov 1 11:30 PM EST
        (_utc(2026, 11, 2, 5, 30), False)]),    # Nov 2 12:30 AM EST
    # Sun 2026-03-08, clocks go forward: 23 hours, 2:00 to 3:00 AM does not exist.
    (date(2026, 3, 8), 23, _utc(2026, 3, 8, 5), _utc(2026, 3, 9, 4), [
        (_utc(2026, 3, 8, 4, 30), False),       # Mar 7 11:30 PM EST
        (_utc(2026, 3, 8, 6, 30), True),        # Mar 8 1:30 AM EST
        (_utc(2026, 3, 8, 7, 30), True),        # Mar 8 3:30 AM EDT
        (_utc(2026, 3, 9, 3, 30), True),        # Mar 8 11:30 PM EDT
        (_utc(2026, 3, 9, 4, 30), False)]),     # Mar 9 12:30 AM EDT
])
def test_daylight_saving_day_is_one_whole_eastern_day(kit, day, hours, start, end, entries):
    posted = [(kit.post('receive', [(kit.oats_lot, 1)], when), inside) for when, inside in entries]
    analysis = daily_reconciliation.run(kit.cur, dry_run=True, day=day)['analysis']
    window = [datetime.fromisoformat(analysis[key]) for key in ('window_start', 'window_end')]
    assert window == [start, end] and window[1] - window[0] == timedelta(hours=hours)
    assert analysis['plant_date'] == day.isoformat()
    data = reconciliation.load_window(kit.cur, start, end, reconstruct=True)
    in_window = {t['id'] for t in data['transactions'] if t['in_window']}
    assert [tx in in_window for tx, _ in posted] == [inside for _, inside in posted]
    assert all(t['entered_plant_date'] == day.isoformat() for t in data['transactions'] if t['id'] in kit.txs)
    assert kit.flags({'analysis': analysis}, 'LATE_ENTRY') == []        # each entry is on its own Eastern day
    # the next day picks up exactly where this one ended: nothing counted twice, nothing dropped
    following = daily_reconciliation.run(kit.cur, dry_run=True, day=day + timedelta(days=1))['analysis']
    assert datetime.fromisoformat(following['window_start']) == end


@pytest.mark.parametrize('run_at,plant_day,start,hours', [
    (_utc(2026, 11, 2, 7, 30), date(2026, 11, 1), _utc(2026, 11, 1, 7, 30), 24),   # 2:30 AM EST after the 25-hour day
    (_utc(2026, 3, 9, 7, 30), date(2026, 3, 8), _utc(2026, 3, 8, 7, 30), 24),      # 3:30 AM EDT after the 23-hour day
])
def test_nightly_runs_across_a_clock_change_check_the_right_day_once(kit, run_at, plant_day, start, hours):
    before = kit.run(start)                                             # the run the night before (07:30 UTC cron)
    inside = kit.post('receive', [(kit.oats_lot, 1)], start + timedelta(hours=1))
    outcome = kit.run(run_at)
    row = kit.row(outcome['run_id'])
    assert row['plant_date'] == plant_day and kit.row(before['run_id'])['plant_date'] == plant_day - timedelta(days=1)
    assert (row['previous_run_id'], row['window_start'], row['window_end']) == (before['run_id'], start, run_at)
    assert row['window_end'] - row['window_start'] == timedelta(hours=hours)
    # opening + the 1 lb receipt = closing: counted once, and this window starts where the last one ended
    assert kit.flags(outcome, 'BALANCE_DIFFERENCE') == [] and kit.flags(outcome, 'LATE_ENTRY') == []
    assert inside in {t['id'] for t in reconciliation.load_window(kit.cur, start, run_at, reconstruct=True)
                      ['transactions'] if t['in_window']}


def test_a_first_ever_send_that_fails_is_still_covered(kit):
    adjust = kit.post('adjust', [(kit.oats_lot, -40)], at(0, 9), named=False, reason_code='unknown')
    failed = kit.run(night(0), sender=_broken)
    outcome = kit.run(night(1))
    assert [f['transaction_id'] for f in kit.flags(outcome, 'UNEXPLAINED_ADJUSTMENT')] == [adjust]
    row = kit.row(outcome['run_id'])
    assert row['previous_run_id'] is None and row['window_start'] == kit.row(failed['run_id'])['window_start']
    assert 'NOTE: the email for Tue Mar 10 was not delivered.' in kit.sent[-1]['text']
    kit.cur.execute("INSERT INTO reconciliation_runs (started_at, plant_date, job_status, job_error, email_status) "
                    "VALUES (%s, %s, 'failed', 'OperationalError', 'not_attempted')", (night(2), date(2026, 3, 12)))
    kit.run(night(3))                                                   # a night whose job failed is named too
    assert 'NOTE: the check for Thu Mar 12 did not finish' in kit.sent[-1]['text']


def test_a_sender_that_raises_still_saves_the_run(kit):
    kit.run(night(-1))
    kit.post('adjust', [(kit.oats_lot, -40)], at(0, 9), named=False, reason_code='unknown')

    def explodes(email, env):
        raise KeyError('re_test')
    outcome = kit.run(night(0), sender=explodes)
    row = kit.row(outcome['run_id'])
    assert (row['job_status'], row['result'], row['email_status']) == ('completed', 'attention', 'failed')
    assert row['email_error'] == 'Email step failed (KeyError)' and row['flag_count'] >= 1


def _carried(outcome, kind):
    return [c for c in outcome['analysis']['carried'] if c['kind'] == kind]


def _short_lot(kit):
    """A lot holding 100 lb, received before the first run."""
    lot = kit.lot(kit.nuts)
    kit.post('receive', [(lot, 100)], at(-1, 11))
    return lot


def test_a_saved_alarm_corrected_before_the_next_email_is_still_shown(kit):
    lot = _short_lot(kit)
    kit.run(night(-1))
    kit.post('ship', [(lot, -101)], at(0, 9))                           # the lot is at -1 lb
    failed = kit.run(night(0), sender=_broken)
    assert [f['actual_lb'] for f in kit.flags(failed, 'NEGATIVE_LOT')] == [-1]
    kit.post('adjust', [(lot, 1)], at(1, 9), reason_code='physical_count')      # named, reasoned +1 lb correction
    outcome = kit.run(night(1))
    assert kit.flags(outcome, 'NEGATIVE_LOT') == []                     # tonight's books are clean...
    (alarm,) = _carried(outcome, 'NEGATIVE_LOT')                        # ...but the saved alarm is not lost
    assert (alarm['lot_id'], alarm['actual_lb'], alarm['run_id'], alarm['status']) == (lot, -1, failed['run_id'], 'corrected')
    assert outcome['analysis']['result'] == 'attention' and outcome['analysis']['carried_attention_count'] == 1
    email = kit.sent[-1]
    assert 'all clear' not in email['subject'] and 'ALL CLEAR' not in email['text']
    assert '1 earlier finding not emailed before' in email['subject']
    text = email['text']
    section = text[text.index('FROM UNDELIVERED RUNS'):text.index("TONIGHT'S FINDINGS")]
    assert 'is at −1 lb.' in section and 'Since then the lot was corrected. It now stands at 0 lb.' in section
    assert 'No lot is below zero.' in text[text.index("TONIGHT'S FINDINGS"):]
    row = kit.row(outcome['run_id'])
    assert (row['result'], row['email_status']) == ('attention', 'sent')
    assert [c['status'] for c in row['summary']['carried'] if c['kind'] == 'NEGATIVE_LOT'] == ['corrected']
    kit.cur.execute("SELECT count(*) AS n FROM reconciliation_flags WHERE run_id = %s AND kind = 'NEGATIVE_LOT'",
                    (outcome['run_id'],))
    assert kit.cur.fetchone()['n'] == 0                                 # carried findings are not saved a second time
    after = kit.run(night(2))                                           # delivered: not repeated
    assert after['analysis']['carried'] == [] and 'FROM UNDELIVERED RUNS' not in kit.sent[-1]['text']
    assert after['analysis']['result'] == 'all_clear'


def test_saved_flags_stay_owed_until_a_send_succeeds(kit):
    lot = _short_lot(kit)
    kit.run(night(-1))
    kit.post('ship', [(lot, -101)], at(0, 9))
    kit.run(night(0), sender=_broken)
    kit.post('adjust', [(lot, 1)], at(1, 9), reason_code='physical_count')
    second = kit.run(night(1), sender=_broken)                          # the email that carried it failed too
    assert [c['status'] for c in _carried(second, 'NEGATIVE_LOT')] == ['corrected']
    third = kit.run(night(2))
    assert [c['actual_lb'] for c in _carried(third, 'NEGATIVE_LOT')] == [-1]    # still owed, shown once
    assert kit.sent[-1]['text'].count('is at −1 lb.') == 1
    assert kit.run(night(3))['analysis']['carried'] == []

    # the same night: a failed send, a correction, then a retry that goes out
    kit.post('ship', [(lot, -2)], at(4, 9))
    kit.run(night(4), sender=_broken)
    kit.post('adjust', [(lot, 2)], night(4) + timedelta(minutes=3), reason_code='physical_count')
    retry = kit.run(night(4) + timedelta(minutes=10))
    assert retry['email_status'] == 'sent' and [c['actual_lb'] for c in _carried(retry, 'NEGATIVE_LOT')] == [-2]
    assert 'NOTE: an earlier run of this check was saved ' in kit.sent[-1]['text']
    assert kit.run(night(5))['analysis']['carried'] == []


def test_an_alarm_saved_by_an_extra_same_day_run_is_still_shown(kit):
    lot = _short_lot(kit)
    kit.run(night(-1))
    kit.run(night(0))
    kit.post('ship', [(lot, -101)], at(1, 9))
    manual = kit.run(at(1, 14))                                         # saved, no email (already sent that day)
    assert manual['email_status'] == 'already_sent' and len(kit.flags(manual, 'NEGATIVE_LOT')) == 1
    kit.post('adjust', [(lot, 1)], at(1, 15), reason_code='physical_count')
    outcome = kit.run(night(1))
    assert kit.flags(outcome, 'NEGATIVE_LOT') == []
    (alarm,) = _carried(outcome, 'NEGATIVE_LOT')
    assert (alarm['run_id'], alarm['actual_lb'], alarm['status']) == (manual['run_id'], -1, 'corrected')
    assert 'is at −1 lb.' in kit.sent[-1]['text'] and outcome['analysis']['result'] == 'attention'


def test_a_saved_finding_voided_before_the_next_email_is_still_shown(kit):
    kit.run(night(-1))
    adjust = kit.post('adjust', [(kit.oats_lot, -40)], at(0, 9), named=False, reason_code='unknown')
    failed = kit.run(night(0), sender=_broken)
    kit.void(adjust, at(1, 9))
    outcome = kit.run(night(1))
    assert kit.flags(outcome, 'UNEXPLAINED_ADJUSTMENT') == []           # voided: tonight's check does not list it
    (old,) = _carried(outcome, 'UNEXPLAINED_ADJUSTMENT')
    assert (old['transaction_id'], old['run_id'], old['status']) == (adjust, failed['run_id'], 'voided')
    assert f'Since then TX{adjust} was voided.' in kit.sent[-1]['text']
    assert outcome['analysis']['result'] == 'attention'


def test_a_saved_finding_removed_by_a_recipe_edit_is_still_shown(kit):
    kit.run(night(-1))
    make, _ = kit.make(10, at(0, 9), oats=700)                          # recipe says 600 lb
    failed = kit.run(night(0), sender=_broken)
    assert kit.kinds(failed) == ['BALANCE_DIFFERENCE', 'MAKE_OFF_RECIPE']
    kit.cur.execute('UPDATE batch_formulas SET quantity_lb = 70 WHERE product_id = %s AND ingredient_product_id = %s',
                    (kit.granola, kit.oats))                            # the recipe is edited to match
    outcome = kit.run(night(1))
    assert kit.flags(outcome) == []                                     # nothing is off against the NEW recipe
    (old,) = _carried(outcome, 'MAKE_OFF_RECIPE')
    assert (old['transaction_id'], old['expected_lb'], old['actual_lb'], old['status']) == (make, 600, 700, 'recipe_changed')
    assert [c['status'] for c in _carried(outcome, 'BALANCE_DIFFERENCE')] == ['no_longer_found']
    text = kit.sent[-1]['text']
    assert 'The recipe calls for 600 lb' in text and 'Since then the recipe on file changed. It now calls for 700 lb.' in text
    assert outcome['analysis']['result'] == 'attention' and outcome['analysis']['carried_attention_count'] == 2


def test_a_carried_finding_that_is_still_open_is_marked_and_counted_once(kit):
    lot = _short_lot(kit)
    kit.run(night(-1))
    kit.post('ship', [(lot, -101)], at(0, 9))
    kit.run(night(0), sender=_broken)
    outcome = kit.run(night(1))                                         # nobody corrected it
    assert [c['status'] for c in _carried(outcome, 'NEGATIVE_LOT')] == ['still_open']
    assert len(kit.flags(outcome, 'NEGATIVE_LOT')) == 1 and outcome['analysis']['carried_attention_count'] == 0
    assert 'Still open: it is also in tonight' in kit.sent[-1]['text']
    assert '1 lot below zero' in kit.sent[-1]['subject'] and 'earlier finding' not in kit.sent[-1]['subject']


def _et(month, day, hour=0, minute=0):
    return datetime(2026, month, day, hour, minute, tzinfo=ET)


@pytest.mark.parametrize('legacy_row', [False, True])
def test_a_first_ever_failed_job_does_not_lose_its_day(kit, legacy_row):
    first = kit.post('adjust', [(kit.oats_lot, -40)], _et(11, 9, 9), named=False, reason_code='unknown')
    if legacy_row:                                                      # a failure saved before bounds were recorded
        kit.cur.execute("INSERT INTO reconciliation_runs (started_at, plant_date, job_status, job_error, email_status) "
                        "VALUES (%s, %s, 'failed', 'OperationalError', 'not_attempted')", (_et(11, 10, 2, 30), date(2026, 11, 9)))
    else:
        reconciliation.record_failed_run(kit.cur, 'OperationalError', now=_et(11, 10, 2, 30))
        kit.cur.execute("SELECT plant_date, window_start, window_end, job_status FROM reconciliation_runs")
        (saved,) = kit.cur.fetchall()
        assert (saved['plant_date'], saved['window_start'], saved['window_end'], saved['job_status']) \
            == (date(2026, 11, 9), _et(11, 9), _et(11, 10, 2, 30), 'failed')
    second = kit.post('adjust', [(kit.oats_lot, -15)], _et(11, 10, 9), named=False, reason_code='unknown')
    outcome = kit.run(_et(11, 11, 2, 30))                               # nothing ran on the night of Nov 10
    row = kit.row(outcome['run_id'])
    assert (row['plant_date'], row['window_start'], row['previous_run_id']) == (date(2026, 11, 10), _et(11, 9), None)
    assert [f['transaction_id'] for f in kit.flags(outcome, 'UNEXPLAINED_ADJUSTMENT')] == [first, second]
    assert 'NOTE: the check for Mon Nov 9 did not finish' in kit.sent[-1]['text']


def test_a_failed_job_after_a_delivered_run_saves_where_it_started(kit):
    delivered = kit.run(night(-1))
    reconciliation.record_failed_run(kit.cur, 'OperationalError', now=night(0))
    kit.cur.execute("SELECT window_start, window_end FROM reconciliation_runs WHERE job_status = 'failed'")
    assert tuple(kit.cur.fetchone().values()) == (night(-1), night(0))
    adjust = kit.post('adjust', [(kit.oats_lot, -40)], at(0, 9), named=False, reason_code='unknown')
    outcome = kit.run(night(1))
    assert kit.row(outcome['run_id'])['previous_run_id'] == delivered['run_id']
    assert [f['transaction_id'] for f in kit.flags(outcome, 'UNEXPLAINED_ADJUSTMENT')] == [adjust]
    assert kit.row(outcome['run_id'])['window_start'] == night(-1)
    assert 'NOTE: the check for Tue Mar 10 did not finish' in kit.sent[-1]['text']


@pytest.mark.parametrize('day,start,end', [
    (date(2026, 11, 1), _utc(2026, 11, 1, 4), _utc(2026, 11, 2, 5)),    # 25-hour day
    (date(2026, 3, 8), _utc(2026, 3, 8, 5), _utc(2026, 3, 9, 4)),       # 23-hour day
    (date(2026, 6, 15), _utc(2026, 6, 15, 4), _utc(2026, 6, 16, 4)),    # an ordinary day
])
def test_an_entry_at_exactly_midnight_belongs_to_the_day_that_starts(kit, day, start, end):
    tick = timedelta(microseconds=1)
    lot = kit.lot(kit.nuts)
    kit.post('receive', [(lot, 1000)], start - timedelta(days=30))
    just_before = kit.post('ship', [(lot, -1)], start - tick)           # the last instant of the day before
    opens = kit.post('ship', [(lot, -10)], start)                       # exactly midnight: this day
    closes = kit.post('ship', [(lot, -100)], end)                       # exactly the next midnight: the next day
    voided = kit.post('ship', [(lot, -7)], start - timedelta(hours=1))
    kit.void(voided, end)                                               # a void at exactly the next midnight: next day too

    def replay(which):
        data = reconciliation.load_window(kit.cur, *which, reconstruct=True)
        entered = {t['id'] for t in data['transactions'] if t['in_window']} & {just_before, opens, closes, voided}
        balance = lambda key: dict(map(tuple, data[key])).get(lot, 0.0)
        analysis = reconciliation.analyse(data)
        assert [f for f in analysis['flags'] if f['lot_id'] == lot and f['kind'] == 'BALANCE_DIFFERENCE'] == []
        return entered, balance('opening'), balance('ending'), analysis

    previous_start = datetime.combine(day - timedelta(days=1), datetime.min.time(), tzinfo=ET)
    next_end = datetime.combine(day + timedelta(days=2), datetime.min.time(), tzinfo=ET)
    entered, opening, ending, analysis = replay((start, end))
    assert entered == {opens} and (opening, ending) == (992, 982)       # 1000 - 7 - 1, then - 10
    assert (analysis['window_start'], analysis['window_end']) == (start.isoformat(), end.isoformat())
    assert analysis['plant_date'] == day.isoformat()
    entered, opening, ending, _ = replay((previous_start, start))
    assert entered == {voided, just_before} and ending == 992           # the day before ends without the midnight entry
    entered, opening, ending, analysis = replay((end, next_end))
    assert entered == {closes} and (opening, ending) == (982, 889)      # - 100, and the 7 lb void comes back
    assert [f['transaction_id'] for f in analysis['flags'] if f['kind'] == 'VOID_OF_EARLIER_ENTRY'
            and f['transaction_id'] in kit.txs] == [voided]
    # the command gives the same day
    assert daily_reconciliation.run(kit.cur, dry_run=True, day=day)['analysis']['entries_checked'] >= 1


def test_nightly_windows_still_meet_at_the_snapshot_moment(kit):
    first = kit.run(night(-1))
    on_the_dot = kit.post('receive', [(kit.oats_lot, 5)], night(-1))    # entered at the very moment of the snapshot
    data = reconciliation.load_window(kit.cur, night(-1), night(0), opening_run_id=first['run_id'])
    assert on_the_dot not in {t['id'] for t in data['transactions'] if t['in_window']}     # (start, end]: the run before has it
    closing = kit.post('receive', [(kit.oats_lot, 7)], night(0))        # at the very moment of the next snapshot
    data = reconciliation.load_window(kit.cur, night(-1), night(0), opening_run_id=first['run_id'])
    assert closing in {t['id'] for t in data['transactions'] if t['in_window']}
    second = kit.run(night(0))
    assert kit.flags(second, 'BALANCE_DIFFERENCE') == []
    assert kit.flags(kit.run(night(1)), 'BALANCE_DIFFERENCE') == []     # and it is not counted again the night after


def test_12_a_run_changes_no_existing_table(kit):
    kit.make(10, at(0, 9), oats=700)
    kit.post('adjust', [(kit.nuts_lot, -1200)], at(0, 10), named=False, reason_code='unknown')
    kit.cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename <> ALL(%s) ORDER BY 1",
                    (list(NEW_TABLES),))
    existing = [r['tablename'] for r in kit.cur.fetchall()]
    assert {'transactions', 'transaction_lines', 'ledger_corrections', 'exceptions', 'lots', 'products'} <= set(existing)
    before = _fingerprint(kit.cur, existing)
    kit.run(night(-1))
    outcome = kit.run(night(0))
    assert kit.flags(outcome)                                           # the run found things and saved them...
    assert _fingerprint(kit.cur, existing) == before                    # ...and no existing table changed by one byte
    # Static guard: the module's only write statements are INSERTs into the three 094 tables.
    source = (ROOT / 'reconciliation.py').read_text()
    sql = ' '.join(node.value for node in ast.walk(ast.parse(source))
                   if isinstance(node, ast.Constant) and isinstance(node.value, str))
    assert set(re.findall(r'INSERT\s+INTO\s+(\w+)', sql, re.I)) == set(NEW_TABLES)
    assert not re.search(r'\b(UPDATE\s+\w+\s+SET|DELETE\s+FROM|TRUNCATE|DROP\s+TABLE|ALTER\s+TABLE|FOR\s+UPDATE)\b', sql, re.I)


@pytest.mark.parametrize('day,entries,makes,lots', [('2026-11-05', 10, 3, 22), ('2026-11-06', 8, 3, 23),
                                                    ('2026-11-09', 17, 7, 35)])
def test_13_replay_of_the_hand_run(day, entries, makes, lots):
    """The Nov 5 / 6 / 9 hand run (scoping §3), replayed from a saved read-only export. No database."""
    analysis = reconciliation.analyse(json.loads((FIXTURES / f'replay_{day}.json').read_text()))
    assert (analysis['entries_checked'], analysis['makes_checked'], analysis['lots_checked']) == (entries, makes, lots)
    assert all(m['checked'] and m['batches'] == round(m['batches']) for m in analysis['makes'])
    found = {(f['kind'], f['transaction_id']) for f in analysis['flags'] if f['kind'] != 'NEGATIVE_LOT'}
    assert not {k for k, _ in found} & {'BALANCE_DIFFERENCE', 'MAKE_OFF_RECIPE', 'MAKE_INGREDIENT_MISSING',
                                        'MAKE_UNLISTED_INGREDIENT', 'MAKE_NO_RECIPE', 'PACK_NO_MAKE', 'LATE_ENTRY'}
    negative = sorted(f['actual_lb'] for f in analysis['flags'] if f['kind'] == 'NEGATIVE_LOT')
    text = reconciliation.render_email(analysis)['text']
    if day == '2026-11-05':
        assert found == {('COCONUT_YIELD', 2961)} and negative == [-70, -68.49, -60, -55, -14]
        assert 'NOV 05 2026: 5,040 lb made, 5,095 lb packed (+55 lb, 1.1%)' in text
    elif day == '2026-11-06':
        assert found == {('COCONUT_YIELD', 2969)} and negative == [-70, -68.49, -60, -55, -14, -10]
        assert 'NOV 06 2026: 1,080 lb made, 1,090 lb packed (+10 lb, 0.9%)' in text
        assert [s['ingredient'] for m in analysis['makes'] for s in m['substitutions']] == ['Oat Bran']
    else:
        assert found == {('PACK_EXCEEDS_MAKE', 2973), ('UNEXPLAINED_ADJUSTMENT', 2972), ('VOID_SAME_DAY', 2971)}
        tx2973 = next(f for f in analysis['flags'] if f['kind'] == 'PACK_EXCEEDS_MAKE')
        assert (tx2973['severity'], tx2973['expected_lb'], tx2973['actual_lb'], tx2973['difference_lb']) \
            == ('alarm', 350, 375, 25)
        assert tx2973['tolerance_lb'] == 7 and 'adjustment TX2972 (+25 lb, reason "unknown"' in tx2973['detail']['message']
        assert 'TX2973 packed 375 lb from Batch SS Original Granola #1 lot OCT 28 2026' in text
        assert text.index('OVER TOLERANCE (1)') < text.index('LOTS BELOW ZERO (6)') < text.index('FOR INFORMATION')
        assert len(negative) == 6 and analysis['result'] == 'attention'


def test_14_gpt_schemas_are_untouched_and_the_routes_stay_out_of_openapi():
    methods = {'get', 'post', 'put', 'patch', 'delete'}
    for path, operations in (('openapi-gpt-v3.yaml', 30), ('gpt-configs/schemas/openapi-floor.yaml', 22)):
        text = (ROOT / path).read_text(encoding='utf-8')
        spec = yaml.safe_load(text)
        assert sum(1 for item in spec['paths'].values() for m in item if m in methods) == operations, path
        assert 'reconciliation' not in text.lower(), path
    for path in ROOT.glob('gpt-configs/**/*.yaml'):
        assert 'reconciliation' not in path.read_text(encoding='utf-8').lower(), path
    assert not [p for p in main.app.openapi()['paths'] if p.startswith('/verification')]
    served = {(m, r.path) for r in main.app.routes if getattr(r, 'path', '').startswith('/verification/reconciliation')
              for m in r.methods}
    assert served == set(reconciliation.ROUTES) and all(m == 'GET' for m, _ in served)


# ───────────────────────── Michael's build rules ─────────────────────────

def test_tolerance_is_measured_on_the_days_movement_not_the_balance(kit):
    assert [reconciliation.tolerance(x) for x in (0, 100, 250, 251, 5000)] == pytest.approx([5, 5, 5, 5.02, 100])
    big = kit.lot(kit.oats)
    kit.post('receive', [(big, 150000)], at(-1, 9))                     # 2% of the balance would be 3,000 lb
    kit.run(night(-1))
    kit.post('adjust', [(big, -100)], at(0, 9), reason_code='physical_count')
    # Something changes the books outside the normal entries: 50 lb vanish from an old line.
    kit.cur.execute('ALTER TABLE transaction_lines DISABLE TRIGGER trg_transaction_lines_original_append_only')
    kit.cur.execute('UPDATE transaction_lines SET quantity_lb = quantity_lb - 50 WHERE lot_id = %s AND quantity_lb = 150000', (big,))
    outcome = kit.run(night(0))
    (flag,) = kit.flags(outcome)
    assert (flag['kind'], flag['lot_id'], flag['severity']) == ('BALANCE_DIFFERENCE', big, 'alarm')
    assert (flag['expected_lb'], flag['actual_lb'], flag['difference_lb'], flag['tolerance_lb']) == (149900, 149850, -50, 5)
    assert flag['detail']['unexplained_lb'] == -50 and 'outside the normal entries' in flag['detail']['message']


def test_off_recipe_make_also_shows_in_the_lot_roll_forward(kit):
    kit.run(night(-1))
    make, _ = kit.make(10, at(0, 9), oats=700)                          # 100 lb over on one lot
    outcome = kit.run(night(0))
    assert kit.kinds(outcome) == ['BALANCE_DIFFERENCE', 'MAKE_OFF_RECIPE']
    lot_flag = kit.flags(outcome, 'BALANCE_DIFFERENCE')[0]
    assert (lot_flag['lot_id'], lot_flag['expected_lb'], lot_flag['actual_lb'], lot_flag['tolerance_lb']) \
        == (kit.oats_lot, 4400, 4300, 12)
    assert lot_flag['detail']['unexplained_lb'] == 0 and lot_flag['detail']['recipe_gap_lb'] == -100


def test_unknown_or_unnamed_adjustments_are_always_listed(kit):
    kit.run(night(-1))
    unknown = kit.post('adjust', [(kit.oats_lot, 2)], at(0, 9), reason_code='unknown')                  # named, no reason
    shared = kit.post('adjust', [(kit.honey_lot, -3)], at(0, 10), named=False, reason_code='physical_count')
    kit.post('adjust', [(kit.nuts_lot, 1400)], at(0, 11), reason_code='physical_count')                # named + reason: fine
    outcome = kit.run(night(0))
    listed = {f['transaction_id']: f for f in kit.flags(outcome)}
    assert sorted(listed) == [unknown, shared] and {f['kind'] for f in listed.values()} == {'UNEXPLAINED_ADJUSTMENT'}
    assert listed[unknown]['detail']['no_reason'] and not listed[unknown]['detail']['no_named_person']
    assert listed[shared]['detail']['no_named_person'] and not listed[shared]['detail']['no_reason']
    assert all(f['severity'] == 'review' for f in listed.values()) and outcome['analysis']['result'] == 'attention'
    assert 'ADJUSTMENTS WITH NO REASON OR NO NAMED PERSON (2)' in outcome['email']['text']


def test_coconut_overpack_within_two_percent_is_yield_not_an_alarm(kit, monkeypatch):
    assert reconciliation.COCONUT_YIELD_PRODUCT_IDS == {125, 126, 127, 128}     # the four coconut bulk batches
    coconut = kit.product('Batch Sweetened Flake', 'batch', batch_lb=360)       # on the list; the name is irrelevant
    lookalike = kit.product('Batch Coconut Granola', 'batch', batch_lb=360)     # "Coconut" in the name, not on the list
    monkeypatch.setattr(reconciliation, 'COCONUT_YIELD_PRODUCT_IDS', frozenset({coconut}))
    kit.recipe(coconut, {kit.honey: 100})
    kit.recipe(lookalike, {kit.honey: 100})
    finished = kit.lot(kit.bagged)
    kit.run(night(-1))
    within, beyond = kit.lot(coconut), kit.lot(coconut)
    kit.post('make', [(within, 5040), (kit.honey_lot, -1400)], at(0, 9))
    kit.post('pack', [(within, -5095), (finished, 5095)], at(0, 11))            # +55 lb = 1.1%
    kit.post('receive', [(kit.honey_lot, 1000)], at(0, 12))
    kit.post('make', [(beyond, 360), (kit.honey_lot, -100)], at(0, 13))
    kit.post('pack', [(beyond, -372), (finished, 372)], at(0, 14))              # +12 lb = 3.3%
    other = kit.lot(lookalike)
    kit.post('receive', [(kit.honey_lot, 1400)], at(0, 15))
    kit.post('make', [(other, 5040), (kit.honey_lot, -1400)], at(0, 16))
    kit.post('pack', [(other, -5095), (finished, 5095)], at(0, 17))             # +55 lb = 1.1%, inside tolerance
    outcome = kit.run(night(0))
    found = {(f['kind'], f['lot_id']): f for f in kit.flags(outcome)}
    assert sorted(found) == sorted([('COCONUT_YIELD', within), ('NEGATIVE_LOT', within),
                                    ('PACK_EXCEEDS_MAKE', beyond), ('NEGATIVE_LOT', beyond),
                                    ('NEGATIVE_LOT', other)])               # no coconut line for the lookalike
    assert found[('COCONUT_YIELD', within)]['severity'] == 'yield'
    text = outcome['email']['text']
    section = text[text.index('COCONUT YIELD'):text.index('FOR INFORMATION')]
    assert '5,040 lb made, 5,095 lb packed (+55 lb, 1.1%)' in section and '(1)' in section.splitlines()[0]
    assert '5,095' not in text[text.index('OVER TOLERANCE'):text.index('LOTS BELOW ZERO')]


def test_first_run_saves_a_baseline_and_says_so(kit):
    outcome = kit.run(night(-1))
    row = kit.row(outcome['run_id'])
    assert (row['previous_run_id'], row['balance_checked'], row['job_status']) == (None, False, 'completed')
    assert 'First run' in outcome['email']['text'] and not kit.flags(outcome, 'BALANCE_DIFFERENCE')


def test_email_is_off_by_default_and_the_run_is_still_saved(kit):
    assert not reconciliation.email_enabled({}) and reconciliation.email_enabled({'RECONCILIATION_EMAIL_ENABLED': 'true'})
    outcome = kit.run(night(-1), env={'ALERT_EMAIL_TO': 'owner@example.test', 'RESEND_API_KEY': 're_test'})
    assert kit.sent == [] and outcome['email_status'] == 'disabled'
    row = kit.row(outcome['run_id'])
    assert (row['job_status'], row['email_status'], row['email_sent_at']) == ('completed', 'disabled', None)
    assert row['summary']['email_text'] == outcome['email']['text']


def test_dry_run_writes_nothing_and_sends_nothing(kit):
    kit.post('adjust', [(kit.oats_lot, -40)], at(0, 9), named=False, reason_code='unknown')
    outcome = kit.run(night(0), dry_run=True)
    assert outcome['run_id'] is None and kit.sent == [] and kit.flags(outcome, 'UNEXPLAINED_ADJUSTMENT')
    for table in NEW_TABLES:
        kit.cur.execute(f'SELECT count(*) AS n FROM {table}')
        assert kit.cur.fetchone()['n'] == 0, table
    replayed = daily_reconciliation.run(kit.cur, dry_run=True, day=date(2026, 3, 10))
    assert replayed['run_id'] is None and replayed['analysis']['mode'] == 'replay'
    assert [f['kind'] for f in kit.flags(replayed)] == ['UNEXPLAINED_ADJUSTMENT']


def test_resend_delivery(monkeypatch):
    email = {'subject': 'S', 'text': 'T', 'html': '<p>T</p>'}
    calls = []

    class Reply:
        def __init__(self, status, body):
            self.status_code, self._body = status, body

        def json(self):
            return self._body

    def post(url, **kwargs):
        calls.append((url, kwargs))
        return Reply(200, {'id': 'abc-123'})
    monkeypatch.setattr(reconciliation.httpx, 'post', post)
    env = dict(EMAIL_ON, ALERT_EMAIL_TO='a@example.test, b@example.test')
    assert reconciliation.send_email(email, env) == {'status': 'sent', 'to': 'a@example.test, b@example.test',
                                                     'provider_id': 'abc-123', 'error': None}
    url, sent = calls[0]
    assert url == 'https://api.resend.com/emails' and sent['headers'] == {'Authorization': 'Bearer re_test'}
    assert sent['json'] == {'from': reconciliation.DEFAULT_EMAIL_FROM, 'to': ['a@example.test', 'b@example.test'],
                            'subject': 'S', 'text': 'T', 'html': '<p>T</p>'}
    monkeypatch.setattr(reconciliation.httpx, 'post', lambda url, **kw: Reply(422, {'message': 'domain is not verified'}))
    failed = reconciliation.send_email(email, env)
    assert failed['status'] == 'failed' and failed['error'] == 'Email service answered 422: domain is not verified'

    def down(url, **kwargs):
        raise reconciliation.httpx.ConnectError('re_test must never be echoed')
    monkeypatch.setattr(reconciliation.httpx, 'post', down)
    failed = reconciliation.send_email(email, env)
    assert failed['status'] == 'failed' and 're_test' not in failed['error']

    def crash(url, **kwargs):
        raise RuntimeError('re_test must never be echoed')              # not an httpx error
    monkeypatch.setattr(reconciliation.httpx, 'post', crash)
    failed = reconciliation.send_email(email, env)
    assert failed == {'status': 'failed', 'to': 'a@example.test, b@example.test', 'provider_id': None,
                      'error': 'Could not reach the email service (RuntimeError)'}
    monkeypatch.setattr(reconciliation.httpx, 'post', lambda url, **kw: Reply(200, ['not', 'an', 'object']))
    assert reconciliation.send_email(email, env)['status'] == 'sent'

    class BadReply(Reply):
        def json(self):
            raise RuntimeError('re_test must never be echoed')          # reading the answer fails, not the POST
    monkeypatch.setattr(reconciliation.httpx, 'post', lambda url, **kw: BadReply(200, None))
    failed = reconciliation.send_email(email, env)
    assert failed == {'status': 'failed', 'to': 'a@example.test, b@example.test', 'provider_id': None,
                      'error': 'Could not read the email service answer (RuntimeError)'}
    assert reconciliation.send_email(email, {'RECONCILIATION_EMAIL_ENABLED': '1'})['error'] \
        == 'ALERT_EMAIL_TO or RESEND_API_KEY is not set'


def test_migration_094_is_append_only_rerunnable_and_touches_nothing_else(kit):
    run_id = kit.run(night(-1))['run_id']
    kit.cur.execute(MIGRATION)                                          # safe to rerun
    for statement in ("UPDATE reconciliation_runs SET result = 'all_clear'", 'DELETE FROM inventory_snapshots',
                      "UPDATE reconciliation_flags SET kind = 'X'", 'DELETE FROM reconciliation_runs'):
        kit.post('adjust', [(kit.oats_lot, -1)], at(0, 9), named=False, reason_code='unknown')
        kit.run(night(0), dry_run=True)
        kit.cur.execute('SAVEPOINT guarded')
        if 'flags' in statement:
            kit.cur.execute("INSERT INTO reconciliation_flags (run_id, kind, severity) VALUES (%s, 'T', 'info')", (run_id,))
        with pytest.raises(psycopg2.Error, match='append-only'):
            kit.cur.execute(statement)
        kit.cur.execute('ROLLBACK TO SAVEPOINT guarded')
    kit.cur.execute("SELECT relname, relrowsecurity FROM pg_class WHERE relname = ANY(%s)", (list(NEW_TABLES),))
    assert {r['relname']: r['relrowsecurity'] for r in kit.cur.fetchall()} == dict.fromkeys(NEW_TABLES, True)
    kit.cur.execute("""SELECT count(*) AS n FROM pg_constraint c JOIN pg_class r ON r.oid = c.conrelid
                       WHERE c.contype = 'f' AND r.relname = ANY(%s) AND c.confrelid::regclass::text <> 'reconciliation_runs'""",
                    (list(NEW_TABLES),))
    assert kit.cur.fetchone()['n'] == 0                                 # no foreign key into any existing table
    kit.cur.execute('SAVEPOINT down094')
    with pytest.raises(psycopg2.Error, match='export them before reversing 094'):
        kit.cur.execute(DOWN)
    kit.cur.execute('ROLLBACK TO SAVEPOINT down094')
    statements = re.sub(r'--.*', '', MIGRATION)
    assert not re.search(r'\b(ALTER TABLE (?!%I)|DROP |UPDATE \w+ SET|DELETE FROM|REFERENCES (?!reconciliation_runs))', statements)


def test_script_guards_its_target_and_the_sweep_hook_is_off_by_default(monkeypatch):
    for key in ('ENVIRONMENT', 'PRODUCTION_DATABASE_HOST', 'PGHOSTADDR', 'PGSERVICE', 'PGSERVICEFILE',
                'RECONCILIATION_ENABLED'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(daily_reconciliation.psycopg2, 'connect',
                        lambda *a, **k: pytest.fail('must refuse before connecting'))
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:pw@db.example.com:5432/x')
    with pytest.raises(RuntimeError):
        daily_reconciliation.main([])
    monkeypatch.setenv('ENVIRONMENT', 'production')
    monkeypatch.setenv('PRODUCTION_DATABASE_HOST', 'somewhere.else')
    with pytest.raises(RuntimeError):
        daily_reconciliation.main(['--dry-run'])
    with pytest.raises(SystemExit):                                     # --date never saves, so it needs --dry-run
        daily_reconciliation.main(['--date', '2026-11-09'])
    launched = []

    class Done:
        returncode = 2
    monkeypatch.setattr(exceptions_sweep.subprocess, 'run', lambda cmd, check: launched.append(cmd) or Done())
    assert exceptions_sweep.run_daily_reconciliation() == 0 and launched == []
    monkeypatch.setenv('RECONCILIATION_ENABLED', '1')
    assert exceptions_sweep.run_daily_reconciliation() == 2
    assert [Path(part).name for part in launched[0][1:]] == ['daily_reconciliation.py']     # the plain command, no flags


def test_endpoints_are_read_only_and_for_owner_and_office(client, actors, db_cursor):
    key = lambda who: {'X-API-Key': actors[who]['key']}
    assert client.get('/verification/reconciliation/runs', headers=key('owner')).status_code == 503   # before 094
    kit = Kit(db_cursor)
    kit.run(night(-1))
    kit.post('adjust', [(kit.oats_lot, -40)], at(0, 9), named=False, reason_code='unknown')
    run_id = kit.run(night(0))['run_id']
    counts = _fingerprint(db_cursor, NEW_TABLES)
    # Every EXISTING table too. Signing in with a named key does its normal bookkeeping
    # (actors.last_used_at, key_actor_sessions); the owner accepts those two and nothing else.
    db_cursor.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename <> ALL(%s) ORDER BY 1",
                      (list(NEW_TABLES) + ['key_actor_sessions', 'actors'],))
    existing = [r['tablename'] for r in db_cursor.fetchall()]
    assert {'transactions', 'transaction_lines', 'ledger_corrections', 'lots', 'products', 'batch_formulas'} <= set(existing)

    def actors_without_sign_in_stamp():
        db_cursor.execute("SELECT count(*) AS n, md5(COALESCE(string_agg((to_jsonb(a) - 'last_used_at')::text, '|' "
                          "ORDER BY a.id), '')) AS h FROM actors a")
        return tuple(db_cursor.fetchone().values())
    before = _fingerprint(db_cursor, existing), actors_without_sign_in_stamp()
    for who in ('owner', 'office'):
        listed = client.get('/verification/reconciliation/runs?limit=1', headers=key(who))
        assert listed.status_code == 200, listed.text
        assert [r['id'] for r in listed.json()['runs']] == [run_id] and listed.json()['runs'][0]['result'] == 'attention'
        detail = client.get(f'/verification/reconciliation/runs/{run_id}', headers=key(who))
        assert detail.status_code == 200 and 'UNEXPLAINED_ADJUSTMENT' in [f['kind'] for f in detail.json()['flags']]
        assert detail.json()['run']['summary']['email_subject'].startswith('Factory Ledger daily check')
    assert client.get('/verification/reconciliation/runs/999999999', headers=key('owner')).status_code == 404
    assert client.get('/verification/reconciliation/runs', headers={'X-API-Key': main.API_KEY}).status_code == 200
    for path in ('runs', f'runs/{run_id}', 'preview'):
        assert client.get(f'/verification/reconciliation/{path}', headers=key('floor')).status_code == 403
        assert client.get(f'/verification/reconciliation/{path}').status_code == 401
        if main.DASHBOARD_API_KEY:
            assert client.get(f'/verification/reconciliation/{path}',
                              headers={'X-API-Key': main.DASHBOARD_API_KEY}).status_code == 403
    assert client.get('/verification/reconciliation/preview', headers=key('office')).status_code == 403
    for method in ('post', 'put', 'patch', 'delete'):
        assert getattr(client, method)('/verification/reconciliation/runs', headers=key('owner')).status_code == 405
    preview = client.get('/verification/reconciliation/preview', headers=key('owner'))
    assert preview.status_code == 200, preview.text
    assert preview.json()['saved'] is False and 'analysis' in preview.json() and 'text' in preview.json()['email']
    assert _fingerprint(db_cursor, NEW_TABLES) == counts                # the three GETs saved nothing
    # ...and no existing table changed, apart from the two sign-in bookkeeping writes allowed above
    assert (_fingerprint(db_cursor, existing), actors_without_sign_in_stamp()) == before
