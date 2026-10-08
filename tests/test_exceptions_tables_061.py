"""A3a: migration 061 — exceptions, shortage_flags, the fixed correction-reason
list (design §5.1 / §7.1 / R3) and the history-only transactions.reason_code
backfill. Tables and seeds only: nothing here goes through a route."""
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import psycopg2
from psycopg2 import errors
from psycopg2.extras import RealDictCursor
import pytest

import main

ROOT = Path(__file__).resolve().parents[1]
UP = (ROOT / 'migrations/061_exceptions_tables.sql').read_text()
DOWN = (ROOT / 'migrations/down/061_exceptions_tables_down.sql').read_text()
DRY_RUN = (ROOT / 'migrations/dry-runs/061_reason_code_backfill_dry_run.sql').read_text()
pytestmark = pytest.mark.db

# Design §5.1 in sort order: code, label_en, label_es, note_required, applies_to, adjust_sign.
SEED = [
    ('physical_count', 'Physical count', 'Conteo físico', False,
     ['adjust', 'found', 'resolve_exception'], 'any'),
    ('missing_receipt', 'Missing receipt', 'Recepción no registrada', False,
     ['found', 'adjust', 'resolve_exception'], 'positive'),
    ('missing_production', 'Missing production', 'Producción no registrada', False,
     ['adjust', 'found', 'resolve_exception'], 'any'),
    ('wrong_lot', 'Wrong lot used', 'Lote equivocado', False,
     ['adjust', 'void', 'rename_lot', 'update_supplier_lot'], 'any'),
    ('damage_disposal', 'Damage/disposal', 'Daño o desecho', False, ['adjust'], 'negative'),
    ('unrecorded_usage', 'Unrecorded usage', 'Uso no registrado', False, ['adjust'], 'negative'),
    ('data_entry_error', 'Data-entry error', 'Error de captura', False,
     ['void', 'adjust', 'rename_lot', 'update_supplier_lot'], 'any'),
    ('unknown', 'Unknown', 'Desconocido', True, ['adjust', 'found'], 'any'),
]
# Design §5.1 "mapping of today's codes".
LEGACY = {
    ('adjust', 'count_correction'): 'physical_count',
    ('adjust', 'damage'): 'damage_disposal',
    ('adjust', 'spoilage'): 'damage_disposal',
    ('adjust', 'sample'): 'unrecorded_usage',
    ('adjust', 'hydration_yield'): 'unrecorded_usage',
    ('adjust', 'other'): 'unknown',
    ('found', 'found_during_count'): 'physical_count',
    ('found', 'found_back_stock'): 'missing_receipt',
    ('found', 'predates_system'): 'missing_receipt',
    ('found', 'unreceived_delivery'): 'missing_receipt',
}
TABLES = ('correction_reasons', 'correction_reason_legacy_codes', 'exceptions', 'shortage_flags')
# What production history looks like: (type, adjust_reason, notes, expected reason_code).
LEGACY_ROWS = [
    ('adjust', 'damage', 'Adjustment: -35 lb', 'damage_disposal'),
    ('adjust', '  Spoilage ', 'x', 'damage_disposal'),
    ('adjust', 'count_correction', 'x', 'physical_count'),
    ('adjust', 'Correction from physical count', 'x', 'physical_count'),
    ('adjust', 'sample', 'x', 'unrecorded_usage'),
    ('adjust', 'Hydration/processing   yield correction', 'x', 'unrecorded_usage'),
    ('adjust', 'other', 'x', 'unknown'),
    ('adjust', 'swept the floor', 'x', 'unknown'),
    ('adjust', None, 'Found inventory: found_during_count', 'physical_count'),
    ('adjust', None, 'Found inventory with new product: predates_system', 'missing_receipt'),
    ('adjust', None, 'Found inventory: unreceived_delivery', 'missing_receipt'),
    ('adjust', None, 'Found inventory: mystery', 'unknown'),
    ('adjust', 'physical_count', 'already a new code', 'physical_count'),
    ('adjust', 'Damage Disposal', 'new code typed as words', 'damage_disposal'),
    ('adjust', None, 'Found inventory: physical_count', 'physical_count'),
    ('adjust', None, 'Adjustment: 5 lb', 'unknown'),
    ('adjust', '', None, 'unknown'),
    # design §11 item 26: count-like free text on adjust_reason → physical_count
    ('adjust', 'Physical count 2026-09-17 (Arturo) (entered via shared key by Michael)', 'x', 'physical_count'),
    ('adjust', 'PHYSICAL INVENTORY COUNT CORRECTION - lot not present', 'x', 'physical_count'),
    ('adjust', 'Count correction — lot not present per 2026-05-14 physical inventory', 'x', 'physical_count'),
    ('adjust', 'Physical cycle count 2026-07-21, floor count by Arturo', 'x', 'physical_count'),
    ('adjust', 'COCONUT-WIP-RECON-2026-09-15: Reconcile coconut batch WIP to stated physical zero', 'x', 'physical_count'),
    ('adjust', 'Physical inventory review performed with Arturo on June 8, 2026. Batch granola', 'x', 'physical_count'),
    ('adjust', 'physical inventory zero - full lot closeout', 'x', 'physical_count'),
    ('adjust', 'Inventory correction to support production', 'x', 'unknown'),
    ('adjust', 'Product merge – SKU deprecated', 'x', 'unknown'),
    # the tier never applies to found-inventory note codes
    ('adjust', None, 'Found inventory: physical count recon', 'unknown'),
    ('receive', None, 'Receive 2 cases', None),
    ('make', 'damage', 'make with stray reason text', None),
]


def seed_rows(cur):
    suffix = uuid4().hex[:10].upper()
    cur.execute("INSERT INTO products(name,odoo_code,type,uom) VALUES (%s,%s,'ingredient','lb') RETURNING id",
                ('A3a test ' + suffix, 'A3A-' + suffix))
    product = cur.fetchone()['id']
    cur.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id', (product, 'A3A-' + suffix))
    lot = cur.fetchone()['id']
    cur.execute("INSERT INTO transactions(type,timestamp,notes,occurred_at) VALUES ('make',now(),%s,now()) RETURNING id",
                ('A3a test ' + suffix,))
    return product, lot, cur.fetchone()['id']


def violates(cur, sql, params=()):
    cur.execute('SAVEPOINT a3a')
    with pytest.raises(psycopg2.IntegrityError):
        cur.execute(sql, params)
    cur.execute('ROLLBACK TO SAVEPOINT a3a')


def test_seed_matches_design_5_1_exactly(db_cursor):
    db_cursor.execute('SELECT code,label_en,label_es,note_required,applies_to,adjust_sign,active '
                      'FROM correction_reasons ORDER BY sort_order')
    rows = db_cursor.fetchall()
    assert [(r['code'], r['label_en'], r['label_es'], r['note_required'], r['applies_to'], r['adjust_sign'])
            for r in rows] == SEED
    assert all(r['active'] for r in rows)
    assert [r['code'] for r in rows if r['note_required']] == ['unknown']
    violates(db_cursor, "INSERT INTO correction_reasons(code,label_en,label_es,applies_to,sort_order) "
                        "VALUES ('Bad Code','x','y',ARRAY['adjust'],99)")
    violates(db_cursor, "INSERT INTO correction_reasons(code,label_en,label_es,applies_to,sort_order) "
                        "VALUES ('newcode','x','y',ARRAY['ship'],99)")
    violates(db_cursor, "INSERT INTO correction_reasons(code,label_en,label_es,applies_to,sort_order) "
                        "VALUES ('newcode','x','y',ARRAY[]::text[],99)")


def test_legacy_map_covers_every_code_the_live_route_still_offers(db_cursor):
    live = main.get_reason_codes(True)
    db_cursor.execute('SELECT source,legacy_code,reason_code,note_prefill FROM correction_reason_legacy_codes')
    rows = {(r['source'], r['legacy_code']): r for r in db_cursor.fetchall()}
    assert len(rows) == 16
    for entry in live['adjustment_reasons']:
        target = LEGACY[('adjust', entry['code'])]
        assert rows[('adjust', entry['code'])]['reason_code'] == target
        assert rows[('adjust', entry['description'].lower())]['reason_code'] == target
    for entry in live['found_inventory_reasons']:
        assert rows[('found', entry['code'])]['reason_code'] == LEGACY[('found', entry['code'])]
    for key, target in LEGACY.items():
        assert rows[key]['reason_code'] == target
    assert rows[('adjust', 'hydration_yield')]['note_prefill'] == 'hydration/yield'
    assert rows[('found', 'predates_system')]['note_prefill'] == 'predates system'
    assert not {k for k in rows if k[1] in ('incorrect_receive', 'product_merge', 'supplier_relabel')}
    violates(db_cursor, "INSERT INTO correction_reason_legacy_codes(source,legacy_code,reason_code) "
                        "VALUES ('adjust','Not Normalised','unknown')")
    violates(db_cursor, "INSERT INTO correction_reason_legacy_codes(source,legacy_code,reason_code) "
                        "VALUES ('reassign','x','unknown')")


def test_exceptions_accepts_design_shape_and_rejects_bad_values(db_cursor):
    product, lot, txn = seed_rows(db_cursor)
    db_cursor.execute("""INSERT INTO exceptions(kind,severity,product_id,lot_id,transaction_id,detail,due_at)
                         VALUES ('SHORTAGE','warn',%s,%s,%s,%s,now()+interval '2 days') RETURNING *""",
                      (product, lot, txn, '{"short_lb": 12}'))
    row = db_cursor.fetchone()
    assert row['status'] == 'open' and row['opened_at'] is not None and row['detail'] == {'short_lb': 12}
    violates(db_cursor, "INSERT INTO exceptions(kind,severity) VALUES ('NOT_A_KIND','warn')")
    violates(db_cursor, "INSERT INTO exceptions(kind,severity) VALUES ('SHORTAGE','urgent')")
    violates(db_cursor, "INSERT INTO exceptions(kind,severity,detail) VALUES ('SHORTAGE','warn','[]')")
    violates(db_cursor, "INSERT INTO exceptions(kind,severity,status) VALUES ('SHORTAGE','warn','closed')")
    violates(db_cursor, "INSERT INTO exceptions(kind,severity,status) VALUES ('SHORTAGE','warn','resolved')")
    violates(db_cursor, "INSERT INTO exceptions(kind,severity,status) VALUES ('SHORTAGE','warn','escalated')")
    violates(db_cursor, "INSERT INTO exceptions(kind,severity,ticket_id) VALUES ('SHORTAGE','warn',999999999)")
    violates(db_cursor, "INSERT INTO exceptions(kind,severity,lot_id) VALUES ('SHORTAGE','warn',-1)")
    violates(db_cursor, "INSERT INTO exceptions(kind,severity,owner_actor_id) VALUES ('SHORTAGE','warn',-1)")
    db_cursor.execute("UPDATE exceptions SET status='resolved', resolved_at=now(), resolution_kind='counted' "
                      "WHERE id=%s RETURNING status", (row['id'],))
    assert db_cursor.fetchone()['status'] == 'resolved'


def test_shortage_flags_shape_and_constraints(db_cursor):
    product, lot, txn = seed_rows(db_cursor)
    db_cursor.execute("INSERT INTO exceptions(kind,severity) VALUES ('SHORTAGE','warn') RETURNING id")
    exc = db_cursor.fetchone()['id']
    db_cursor.execute("""INSERT INTO shortage_flags(transaction_id,product_id,lot_id,short_lb,exception_id,due_at)
                         VALUES (%s,%s,%s,12.5,%s,now()+interval '2 days') RETURNING *""", (txn, product, lot, exc))
    row = db_cursor.fetchone()
    assert row['status'] == 'open' and row['resolution_kind'] is None and row['exception_id'] == exc
    base = (txn, product, lot)
    violates(db_cursor, "INSERT INTO shortage_flags(transaction_id,product_id,lot_id,short_lb,due_at) "
                        "VALUES (%s,%s,%s,0,now())", base)
    violates(db_cursor, "INSERT INTO shortage_flags(transaction_id,product_id,lot_id,short_lb,due_at,status) "
                        "VALUES (%s,%s,%s,1,now(),'resolved')", base)
    violates(db_cursor, "INSERT INTO shortage_flags(transaction_id,product_id,lot_id,short_lb,due_at,resolution_kind) "
                        "VALUES (%s,%s,%s,1,now(),'counted')", base)
    violates(db_cursor, "INSERT INTO shortage_flags(transaction_id,product_id,lot_id,short_lb,due_at,status,"
                        "resolved_at,resolution_kind) VALUES (%s,%s,%s,1,now(),'resolved',now(),'guessed')", base)
    violates(db_cursor, "INSERT INTO shortage_flags(transaction_id,product_id,lot_id,short_lb,due_at) "
                        "VALUES (%s,%s,NULL,1,now())", (txn, product))
    db_cursor.execute("UPDATE shortage_flags SET status='resolved', resolved_at=now(), resolution_kind='counted' "
                      "WHERE id=%s RETURNING status", (row['id'],))
    assert db_cursor.fetchone()['status'] == 'resolved'


def test_transactions_reason_code_is_nullable_and_fk_checked(db_cursor):
    db_cursor.execute("INSERT INTO transactions(type,timestamp,notes,occurred_at,adjust_reason,reason_code) "
                      "VALUES ('adjust',now(),'A3a',now(),'damage','damage_disposal') RETURNING reason_code")
    assert db_cursor.fetchone()['reason_code'] == 'damage_disposal'
    db_cursor.execute("INSERT INTO transactions(type,timestamp,notes,occurred_at) "
                      "VALUES ('receive',now(),'A3a',now()) RETURNING reason_code")
    assert db_cursor.fetchone()['reason_code'] is None
    violates(db_cursor, "INSERT INTO transactions(type,timestamp,notes,occurred_at,reason_code) "
                        "VALUES ('adjust',now(),'A3a',now(),'damage')")


def test_owner_only_posture_and_marker(db_cursor):
    db_cursor.execute("SELECT relname, relrowsecurity FROM pg_class WHERE relname = ANY(%s)", (list(TABLES),))
    rows = {r['relname']: r for r in db_cursor.fetchall()}
    assert set(rows) == set(TABLES) and all(rows[t]['relrowsecurity'] for t in TABLES)
    db_cursor.execute("SELECT c.relname FROM pg_class c, aclexplode(c.relacl) a "
                      "WHERE c.relname = ANY(%s) AND a.grantee = 0", (list(TABLES),))
    assert db_cursor.fetchall() == []
    db_cursor.execute("SELECT 1 FROM migration_markers WHERE name='061_exceptions_tables'")
    assert db_cursor.fetchone()


# ---------------------------------------------------------------------------
# Reversible DDL and the history backfill need a database the migration has
# NOT been applied to yet: build one from the schema dump minus psql
# meta-commands (so the pending \ir include is skipped), as the 058 tests do.
# ---------------------------------------------------------------------------
@contextmanager
def connect(url):
    conn = psycopg2.connect(url)
    try:
        with conn, conn.cursor(cursor_factory=RealDictCursor) as cur:
            yield cur
    finally:
        conn.close()


def apply(url, sql, **settings):
    with connect(url) as cur:
        cur.execute("SET LOCAL lock_timeout = '5s'")
        cur.execute('SET LOCAL search_path TO public')
        for name, value in settings.items():
            cur.execute('SET LOCAL ' + name + ' = %s', (value,))
        cur.execute(sql)


def guard_state(cur):
    cur.execute("SELECT tgenabled FROM pg_trigger WHERE tgrelid='public.transactions'::regclass "
                "AND tgname='trg_transactions_original_append_only'")
    row = cur.fetchone()
    return row and row['tgenabled']


@pytest.fixture
def isolated_database():
    parts = urlsplit(main.DATABASE_URL)
    assert parts.hostname in ('localhost', '127.0.0.1', '::1')
    name = 'a3a_mig_' + uuid4().hex
    url = urlunsplit(parts._replace(path='/' + name))
    admin = psycopg2.connect(main.DATABASE_URL)
    admin.autocommit = True
    with admin.cursor() as cur:
        cur.execute('CREATE DATABASE ' + name)
    try:
        schema = '\n'.join(line for line in (ROOT / 'tests/schema/schema.sql').read_text().splitlines()
                           if not line.startswith('\\'))
        with connect(url) as cur:
            cur.execute('CREATE EXTENSION pg_trgm')
            cur.execute(schema)
        yield url
    finally:
        with admin.cursor() as cur:
            cur.execute('DROP DATABASE ' + name + ' WITH (FORCE)')
        admin.close()


def test_backfill_maps_history_restores_guard_and_is_rerunnable(isolated_database):
    with connect(isolated_database) as cur:
        assert guard_state(cur) == 'O', 'schema dump must carry the 039 append-only guard'
        cur.execute("SELECT to_regclass('public.correction_reasons') AS t")
        assert cur.fetchone()['t'] is None, 'fixture must start before 061'
        expected = {}
        for type_, reason, notes, target in LEGACY_ROWS:
            cur.execute('INSERT INTO transactions(type,timestamp,adjust_reason,notes,occurred_at) '
                        'VALUES (%s,now(),%s,%s,now()) RETURNING id', (type_, reason, notes))
            expected[cur.fetchone()['id']] = target
        cur.execute(DRY_RUN)
        preview = {(r['source'], r['raw_value']): r['will_become'] for r in cur.fetchall()}
        assert preview[('adjust', 'damage')] == 'damage_disposal'
        assert preview[('adjust', 'swept the floor')] == 'unknown'
        assert preview[('found', 'predates_system')] == 'missing_receipt'
        assert preview[('found', 'physical_count')] == 'physical_count'
        assert preview[('adjust', 'Damage Disposal')] == 'damage_disposal'
        assert preview[('<no reason at all>', '<null>')] == 'unknown'   # NULL and blank adjust_reason
        assert preview[('adjust', 'Physical cycle count 2026-07-21, floor count by Arturo')] == 'physical_count'
        assert preview[('adjust', 'Inventory correction to support production')] == 'unknown'
        assert preview[('adjust', 'physical inventory zero - full lot closeout')] == 'physical_count'
        assert preview[('found', 'physical count recon')] == 'unknown'
        assert all(k[0] != 'make' for k in preview)
    apply(isolated_database, UP)
    with connect(isolated_database) as cur:
        cur.execute('SELECT id, type, adjust_reason, notes, reason_code FROM transactions WHERE id = ANY(%s)',
                    (list(expected),))
        rows = cur.fetchall()
        assert len(rows) == len(LEGACY_ROWS)
        for row in rows:
            assert row['reason_code'] == expected[row['id']], row
        cur.execute("SELECT count(*) AS n FROM transactions WHERE type='adjust' AND reason_code IS NULL")
        assert cur.fetchone()['n'] == 0
        cur.execute("SELECT count(*) AS n FROM transactions WHERE type<>'adjust' AND reason_code IS NOT NULL")
        assert cur.fetchone()['n'] == 0
        cur.execute("SELECT adjust_reason, notes FROM transactions WHERE adjust_reason='swept the floor'")
        assert cur.fetchone() == {'adjust_reason': 'swept the floor', 'notes': 'x'}
        assert guard_state(cur) == 'O'
        cur.execute('SAVEPOINT g')
        with pytest.raises(psycopg2.IntegrityError, match='append-only'):
            cur.execute("UPDATE transactions SET notes='tamper' WHERE id=%s", (next(iter(expected)),))
        cur.execute('ROLLBACK TO SAVEPOINT g')
        cur.execute("UPDATE correction_reasons SET label_es='Conteo físico (editado)' WHERE code='physical_count'")
    apply(isolated_database, UP)   # rerun: no error, no duplicate seeds, no overwrite of a maintained row
    with connect(isolated_database) as cur:
        cur.execute('SELECT count(*) AS n FROM correction_reasons')
        assert cur.fetchone()['n'] == 8
        cur.execute('SELECT count(*) AS n FROM correction_reason_legacy_codes')
        assert cur.fetchone()['n'] == 16
        cur.execute("SELECT label_es FROM correction_reasons WHERE code='physical_count'")
        assert cur.fetchone()['label_es'] == 'Conteo físico (editado)'
        cur.execute("SELECT count(*) AS n FROM migration_markers WHERE name='061_exceptions_tables'")
        assert cur.fetchone()['n'] == 1
        assert guard_state(cur) == 'O'


def test_backfill_skips_guard_dance_when_no_guard_exists(isolated_database):
    with connect(isolated_database) as cur:
        cur.execute('DROP TRIGGER trg_transactions_original_append_only ON transactions')
        cur.execute("INSERT INTO transactions(type,timestamp,adjust_reason,notes,occurred_at) "
                    "VALUES ('adjust',now(),'damage','x',now())")
    apply(isolated_database, UP)
    with connect(isolated_database) as cur:
        assert guard_state(cur) is None
        cur.execute("SELECT reason_code FROM transactions WHERE adjust_reason='damage'")
        assert cur.fetchone()['reason_code'] == 'damage_disposal'


def test_down_refuses_populated_queue_then_drops_cleanly_and_up_reseeds(isolated_database):
    apply(isolated_database, UP)
    with connect(isolated_database) as cur:
        cur.execute("INSERT INTO exceptions(kind,severity) VALUES ('LATE_ENTRY','info')")
    with pytest.raises(errors.RaiseException, match='061 rollback refused'):
        apply(isolated_database, DOWN)
    with connect(isolated_database) as cur:
        cur.execute("SELECT to_regclass('public.exceptions') AS t")
        assert cur.fetchone()['t'] == 'exceptions'
    apply(isolated_database, DOWN, **{'factory_ledger.confirm_exceptions_export': 'yes'})
    with connect(isolated_database) as cur:
        cur.execute("SELECT to_regclass('public.exceptions') AS a, to_regclass('public.shortage_flags') AS b, "
                    "to_regclass('public.correction_reasons') AS c, "
                    "to_regclass('public.correction_reason_legacy_codes') AS d")
        assert list(cur.fetchone().values()) == [None, None, None, None]
        cur.execute("SELECT count(*) AS n FROM information_schema.columns "
                    "WHERE table_name='transactions' AND column_name='reason_code'")
        assert cur.fetchone()['n'] == 0
        cur.execute("SELECT count(*) AS n FROM migration_markers WHERE name='061_exceptions_tables'")
        assert cur.fetchone()['n'] == 0
        assert guard_state(cur) == 'O'
    apply(isolated_database, UP)
    with connect(isolated_database) as cur:
        cur.execute('SELECT count(*) AS n FROM correction_reasons')
        assert cur.fetchone()['n'] == 8
