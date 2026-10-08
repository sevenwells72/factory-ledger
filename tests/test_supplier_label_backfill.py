"""066 stays separate/deferred; dry-run matches SQL without writing to the DB."""
from pathlib import Path
import getpass
import os
from urllib.parse import urlsplit, urlunsplit, quote

import psycopg2
import pytest

from scripts.a5_readonly import readonly_cursor
from scripts.dry_run_supplier_labels import apply_assumptions, plan_labels, read_label_plan

pytestmark = pytest.mark.db
ROOT = Path(__file__).parents[1]
MIGRATION = ROOT / 'migrations/066_supplier_labels.sql'
MIGRATION_064 = ROOT / 'migrations/064_unidentified_lots.sql'
CLEANUP = ROOT / 'scripts/supplier_cleanup_dutch_2026_10_08.sql'


def acknowledge(cur):
    cur.execute("SET LOCAL factory_ledger.supplier_cleanup_confirmed='michael_approved'")


def apply_064(cur):
    """The schema fixture is a data-free dump: re-run idempotent 064 for its marker."""
    cur.execute(MIGRATION_064.read_text())


def test_backfill_refuses_before_cleanup_acknowledgment(db_cursor):
    apply_064(db_cursor)
    before = read_label_plan(db_cursor)
    db_cursor.execute('SAVEPOINT cleanup_gate')
    with pytest.raises(psycopg2.errors.RaiseException, match='066 deferred until Michael approves'):
        db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute('ROLLBACK TO SAVEPOINT cleanup_gate')
    assert read_label_plan(db_cursor) == before
    db_cursor.execute("SELECT count(*) AS n FROM migration_markers WHERE name='066_supplier_labels'")
    assert db_cursor.fetchone()['n'] == 0


def test_backfill_refuses_without_064(db_cursor):
    acknowledge(db_cursor)
    with pytest.raises(psycopg2.errors.RaiseException, match='Apply A5 schema migration 064'):
        db_cursor.execute(MIGRATION.read_text())


def test_preview_matches_backfill_collisions_retained_labels_and_pseudos(db_cursor):
    apply_064(db_cursor)
    db_cursor.execute("INSERT INTO suppliers(name,short_code) VALUES ('Reserved Zulu label','DUTA')")
    for i in range(32):
        db_cursor.execute('INSERT INTO suppliers(name,active) VALUES (%s,%s)', (f'Dutch Test Vendor {i}', i != 31))
    db_cursor.execute("INSERT INTO suppliers(name) VALUES ('  FOUND  inventory  '), ('123'), ('éx'), ('DUTC Valley')")
    preview = read_label_plan(db_cursor)
    assert any(row['decision'] == 'excluded pseudo-supplier' for row in preview)
    assert any(row['proposed_label'] == 'AABA' for row in preview)  # >26 collisions
    inactive = next(row for row in preview if row['name'] == 'Dutch Test Vendor 31')
    assert inactive['decision'] == 'skipped inactive' and inactive['proposed_label'] is None
    acknowledge(db_cursor)
    db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute('SELECT id,short_code FROM suppliers ORDER BY id')
    labels = {row['id']: row['short_code'] for row in db_cursor.fetchall()}
    assert labels == {row['id']: row['proposed_label'] for row in preview}
    # An idempotent rerun retains every existing label and marker timestamp.
    db_cursor.execute("SELECT applied_at FROM migration_markers WHERE name='066_supplier_labels'")
    timestamp = db_cursor.fetchone()['applied_at']
    db_cursor.execute(MIGRATION.read_text())
    assert {row['id']: row['proposed_label'] for row in read_label_plan(db_cursor)} == labels
    db_cursor.execute("SELECT applied_at FROM migration_markers WHERE name='066_supplier_labels'")
    assert db_cursor.fetchone()['applied_at'] == timestamp
    db_cursor.execute("INSERT INTO suppliers(name) VALUES ('Dutch Next Vendor') RETURNING short_code")
    assert db_cursor.fetchone()['short_code'] not in set(labels.values())
    db_cursor.execute("INSERT INTO suppliers(name) VALUES ('PHYSICAL COUNT') RETURNING short_code")
    assert db_cursor.fetchone()['short_code'] is None


def test_explicit_labels_kept_inactive_skipped_and_labelled_on_activation(db_cursor):
    apply_064(db_cursor)
    # An earlier INSERT-only trigger (staging, 2026-10-08) must be replaced, not kept.
    db_cursor.execute("""CREATE OR REPLACE FUNCTION a5_supplier_default_display_code() RETURNS trigger
        LANGUAGE plpgsql AS $$ BEGIN RETURN NEW; END $$""")
    db_cursor.execute("""CREATE TRIGGER a5_supplier_default_display_code BEFORE INSERT ON suppliers
        FOR EACH ROW EXECUTE FUNCTION a5_supplier_default_display_code()""")
    db_cursor.execute("""INSERT INTO suppliers(name,active,short_code) VALUES
        ('Dutch Gold Honey', true, 'DUTG'), ('Dutch Valley Foods', true, 'DUTV'),
        ('Dutch Gold', false, NULL), ('Dutch Valley', false, NULL), ('Dutchess Farms', true, NULL)""")
    preview = {row['name']: row for row in read_label_plan(db_cursor)}
    assert (preview['Dutch Gold Honey']['decision'], preview['Dutch Gold Honey']['proposed_label']) == ('retained', 'DUTG')
    assert (preview['Dutch Valley Foods']['decision'], preview['Dutch Valley Foods']['proposed_label']) == ('retained', 'DUTV')
    assert preview['Dutch Gold']['decision'] == preview['Dutch Valley']['decision'] == 'skipped inactive'
    assert preview['Dutchess Farms']['proposed_label'] == 'DUTC'  # natural token is free
    acknowledge(db_cursor)
    db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute("""SELECT name, active, short_code FROM suppliers
        WHERE name IN ('Dutch Gold Honey','Dutch Valley Foods','Dutch Gold','Dutch Valley','Dutchess Farms')""")
    rows = {row['name']: row for row in db_cursor.fetchall()}
    assert rows['Dutch Gold Honey']['short_code'] == 'DUTG'
    assert rows['Dutch Valley Foods']['short_code'] == 'DUTV'
    assert rows['Dutch Gold']['short_code'] is None and rows['Dutch Valley']['short_code'] is None
    assert rows['Dutchess Farms']['short_code'] == 'DUTC'
    db_cursor.execute("""SELECT pg_get_triggerdef(oid) AS def FROM pg_trigger
        WHERE tgrelid='suppliers'::regclass AND tgname='a5_supplier_default_display_code'""")
    assert 'BEFORE INSERT OR UPDATE OF active' in db_cursor.fetchone()['def']
    # Reactivating a label-less supplier assigns the next free code; explicit codes untouched.
    db_cursor.execute("UPDATE suppliers SET active=true WHERE name='Dutch Gold' RETURNING short_code")
    assert db_cursor.fetchone()['short_code'] == 'DUTA'
    db_cursor.execute("UPDATE suppliers SET active=true WHERE name='Dutch Gold Honey' RETURNING short_code")
    assert db_cursor.fetchone()['short_code'] == 'DUTG'
    # Inserting an inactive supplier leaves the label for activation time.
    db_cursor.execute("INSERT INTO suppliers(name,active) VALUES ('Dutch Dormant',false) RETURNING short_code")
    assert db_cursor.fetchone()['short_code'] is None


def test_dutch_cleanup_script_merges_deactivates_and_labels(db_cursor):
    apply_064(db_cursor)
    db_cursor.execute("""INSERT INTO suppliers(id,name,active) VALUES
        (11,'DUTC Valley',true), (12,'Dutch Gold',true), (13,'Dutch Gold Honey',true),
        (14,'Dutch Valley',true), (15,'Dutch Valley Food Dist.',true), (16,'Dutch Valley Foods',true)""")
    db_cursor.execute("UPDATE suppliers SET short_code='DUTA' WHERE id=12")  # staging-style auto label
    db_cursor.execute("""INSERT INTO products(id,name,type,active) VALUES (900001,'Honey test','ingredient',true)
        ON CONFLICT DO NOTHING""")
    db_cursor.execute("""INSERT INTO expected_receipts(product_id,supplier_id,expected_qty,status)
        VALUES (900001, 12, 100, 'open'), (900001, 15, 100, 'open') RETURNING id""")
    er_ids = [row['id'] for row in db_cursor.fetchall()]
    db_cursor.execute("""INSERT INTO supplier_product_aliases(supplier_id,vendor_description,product_id,lb_per_unit,unit)
        VALUES (11,'typo-era alias',900001,50,'BAG')""")
    db_cursor.execute(CLEANUP.read_text().replace('BEGIN;', '').replace('COMMIT;', ''))
    db_cursor.execute('SELECT id, supplier_id FROM expected_receipts WHERE id = ANY(%s) ORDER BY id', (er_ids,))
    assert [row['supplier_id'] for row in db_cursor.fetchall()] == [13, 16]
    db_cursor.execute("SELECT supplier_id FROM supplier_product_aliases WHERE vendor_description='typo-era alias'")
    assert db_cursor.fetchone()['supplier_id'] == 16
    db_cursor.execute('SELECT id, active, short_code FROM suppliers WHERE id BETWEEN 11 AND 16 ORDER BY id')
    rows = {row['id']: (row['active'], row['short_code']) for row in db_cursor.fetchall()}
    assert rows == {11: (False, None), 12: (False, None), 13: (True, 'DUTG'),
                    14: (False, None), 15: (False, None), 16: (True, 'DUTV')}
    db_cursor.execute("""SELECT alias, supplier_id FROM search_aliases WHERE kind='supplier'
        AND alias IN ('Dutch Gold','Dutch Valley','Dutch Valley Food Dist.','DUTC Valley') ORDER BY alias""")
    assert [(r['alias'], r['supplier_id']) for r in db_cursor.fetchall()] == [
        ('DUTC Valley', 16), ('Dutch Gold', 13), ('Dutch Valley', 16), ('Dutch Valley Food Dist.', 16)]
    # Re-running the script is a no-op (idempotent aliases, guards still pass).
    db_cursor.execute(CLEANUP.read_text().replace('BEGIN;', '').replace('COMMIT;', ''))
    db_cursor.execute("SELECT count(*) AS n FROM search_aliases WHERE kind='supplier' AND supplier_id IN (13,16)")
    assert db_cursor.fetchone()['n'] == 4
    # 066 afterwards keeps the explicit labels and skips the deactivated rows.
    acknowledge(db_cursor)
    db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute('SELECT id, short_code FROM suppliers WHERE id BETWEEN 11 AND 16 ORDER BY id')
    assert {row['id']: row['short_code'] for row in db_cursor.fetchall()} == {
        11: None, 12: None, 13: 'DUTG', 14: None, 15: None, 16: 'DUTV'}


def test_dutch_cleanup_script_refuses_a_different_catalog(db_cursor):
    db_cursor.execute("INSERT INTO suppliers(id,name) VALUES (11,'Someone Else')")
    with pytest.raises(psycopg2.errors.RaiseException, match='differ from the 2026-10-08 plan'):
        db_cursor.execute(CLEANUP.read_text().replace('BEGIN;', '').replace('COMMIT;', ''))


def _catalog_rows():
    names = {11: 'DUTC Valley', 12: 'Dutch Gold', 13: 'Dutch Gold Honey', 14: 'Dutch Valley',
             15: 'Dutch Valley Food Dist.', 16: 'Dutch Valley Foods', 17: 'Essex Food'}
    return [{'id': sid, 'name': name, 'active': True, 'current_label': None, 'pseudo': False,
             'base_code': 'DUTC' if sid < 17 else 'ESSE'} for sid, name in names.items()]


def test_assumptions_preview_the_post_cleanup_labels_without_a_database():
    planned = plan_labels(apply_assumptions(_catalog_rows(), inactive_ids=[11, 12, 14, 15],
                                            labels={13: 'DUTG', 16: 'DUTV'}))
    by_id = {row['id']: (row['decision'], row['proposed_label']) for row in planned}
    assert by_id[13] == ('retained', 'DUTG') and by_id[16] == ('retained', 'DUTV')
    assert all(by_id[sid] == ('skipped inactive', None) for sid in (11, 12, 14, 15))
    assert by_id[17] == ('assign', 'ESSE')
    # Without assumptions the pre-cleanup catalog collides its way through DUTC, DUTA, ...
    before = {row['id']: row['proposed_label'] for row in plan_labels(_catalog_rows())}
    assert [before[i] for i in range(11, 17)] == ['DUTC', 'DUTA', 'DUTB', 'DUTD', 'DUTE', 'DUTF']


@pytest.mark.parametrize('kwargs, message', [
    ({'inactive_ids': [99]}, 'unknown supplier id 99'),
    ({'labels': {13: 'dutg'}}, 'not four uppercase letters'),
    ({'labels': {13: 'DUTG', 16: 'DUTG'}}, 'already held by supplier'),
])
def test_assumptions_reject_bad_input(kwargs, message):
    with pytest.raises(ValueError, match=message):
        apply_assumptions(_catalog_rows(), **kwargs)


def test_preview_works_before_a5_short_code_schema_exists(db_cursor):
    db_cursor.execute("INSERT INTO suppliers(name) VALUES ('Dutch Gold preview'), ('Dutch Valley preview')")
    with_schema = read_label_plan(db_cursor)
    db_cursor.execute('ALTER TABLE suppliers DROP COLUMN short_code')
    assert read_label_plan(db_cursor) == with_schema


def test_readonly_connection_rejects_writes_and_rolls_back(tmp_path):
    uri = urlsplit(os.environ['TEST_DATABASE_URL'])
    if not uri.username:
        uri = uri._replace(netloc=quote(getpass.getuser()) + '@' + uri.netloc)
    path = tmp_path / 'database-uri'
    path.write_text(urlunsplit(uri))
    path.chmod(0o600)
    with readonly_cursor(path) as cur:
        cur.execute('SHOW transaction_read_only')
        assert cur.fetchone()['transaction_read_only'] == 'on'
        with pytest.raises(psycopg2.errors.ReadOnlySqlTransaction):
            cur.execute("INSERT INTO suppliers(name) VALUES ('Read-only script must reject this')")


@pytest.mark.parametrize('uri', [
    'postgresql://user:do-not-print@remote.example:6543/postgres',
    'postgresql://user:do-not-print@remote.example:5432/postgres?host=other.example',
])
def test_readonly_rejects_pooler_and_routing_override_before_connection(tmp_path, uri, monkeypatch):
    path = tmp_path / 'database-uri'
    path.write_text(uri)
    path.chmod(0o600)
    def unexpected_connection(*args, **kwargs):
        pytest.fail('Refused URI must never connect')
    monkeypatch.setattr(psycopg2, 'connect', unexpected_connection)
    with pytest.raises(RuntimeError) as exc:
        with readonly_cursor(path):
            pytest.fail('Refused URI must never yield a cursor')
    assert 'do-not-print' not in str(exc.value)
