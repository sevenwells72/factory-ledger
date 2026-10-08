"""066 stays separate/deferred; dry-run matches SQL without writing to the DB."""
from pathlib import Path
import getpass
import os
from urllib.parse import urlsplit, urlunsplit, quote

import psycopg2
import pytest

from scripts.a5_readonly import readonly_cursor
from scripts.dry_run_supplier_labels import read_label_plan

pytestmark = pytest.mark.db
MIGRATION = Path(__file__).parents[1] / 'migrations/066_supplier_labels.sql'


def acknowledge(cur):
    cur.execute("SET LOCAL factory_ledger.supplier_cleanup_confirmed='michael_approved'")


def test_backfill_refuses_before_cleanup_acknowledgment(db_cursor):
    before = read_label_plan(db_cursor)
    db_cursor.execute('SAVEPOINT cleanup_gate')
    with pytest.raises(psycopg2.errors.RaiseException, match='066 deferred until Michael approves'):
        db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute('ROLLBACK TO SAVEPOINT cleanup_gate')
    assert read_label_plan(db_cursor) == before
    db_cursor.execute("SELECT count(*) AS n FROM migration_markers WHERE name='066_supplier_labels'")
    assert db_cursor.fetchone()['n'] == 0


def test_preview_matches_backfill_collisions_retained_labels_and_pseudos(db_cursor):
    db_cursor.execute("INSERT INTO suppliers(name,short_code) VALUES ('Reserved Zulu label','DUTA')")
    for i in range(32):
        db_cursor.execute('INSERT INTO suppliers(name,active) VALUES (%s,%s)', (f'Dutch Test Vendor {i}', i != 31))
    db_cursor.execute("INSERT INTO suppliers(name) VALUES ('  FOUND  inventory  '), ('123'), ('éx'), ('DUTC Valley')")
    preview = read_label_plan(db_cursor)
    assert any(row['decision'] == 'excluded pseudo-supplier' for row in preview)
    assert any(row['proposed_label'] == 'AABA' for row in preview)  # >26 collisions
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
