"""The reviewed rollback preserves the old constraint and never deletes alert history."""
from pathlib import Path

import psycopg2
import pytest


ROOT = Path(__file__).resolve().parents[1]
UP = (ROOT / 'migrations/084_assistant_service_alert.sql').read_text()
DOWN = (ROOT / 'migrations/down/084_assistant_service_alert_down.sql').read_text()
pytestmark = pytest.mark.db


def state(cur):
    cur.execute("""SELECT pg_get_constraintdef(oid) AS definition
        FROM pg_constraint
        WHERE conrelid='exceptions'::regclass AND conname='exceptions_kind_check'""")
    definition = cur.fetchone()['definition']
    cur.execute("""SELECT to_regclass('exceptions_one_assistant_outage_idx') IS NOT NULL AS has_index,
        EXISTS (SELECT 1 FROM migration_markers WHERE name='084_assistant_service_alert') AS has_marker""")
    return definition, dict(cur.fetchone())


def test_apply_rollback_apply_restores_exact_constraint_and_is_idempotent(db_cursor):
    db_cursor.execute('SET LOCAL search_path TO public')
    db_cursor.execute(DOWN)
    original = state(db_cursor)
    assert 'ASSISTANT_OFFLINE' not in original[0]
    assert original[1] == {'has_index': False, 'has_marker': False}
    db_cursor.execute(UP)
    applied = state(db_cursor)
    assert 'ASSISTANT_OFFLINE' in applied[0]
    assert applied[1] == {'has_index': True, 'has_marker': True}
    db_cursor.execute(UP)
    assert state(db_cursor) == applied
    db_cursor.execute(DOWN)
    assert state(db_cursor) == original
    db_cursor.execute(DOWN)
    assert state(db_cursor) == original
    db_cursor.execute(UP)
    assert state(db_cursor) == applied


@pytest.mark.parametrize('status', ['open', 'escalated', 'resolved', 'waived'])
def test_rollback_refuses_any_assistant_offline_history(db_cursor, status):
    db_cursor.execute('SET LOCAL search_path TO public')
    db_cursor.execute(UP)
    db_cursor.execute("""INSERT INTO exceptions(kind,severity,status,detail,due_at,resolved_at,escalated_at)
        VALUES ('ASSISTANT_OFFLINE','warn',%s,'{}',now()-interval '1 day',
                CASE WHEN %s IN ('resolved','waived') THEN now() END,
                CASE WHEN %s='escalated' THEN now() END) RETURNING id""", (status, status, status))
    row_id = db_cursor.fetchone()['id']
    before = state(db_cursor)
    db_cursor.execute('SAVEPOINT attempted_rollback')
    with pytest.raises(psycopg2.errors.RaiseException, match='ASSISTANT_OFFLINE exceptions exist'):
        db_cursor.execute(DOWN)
    db_cursor.execute('ROLLBACK TO SAVEPOINT attempted_rollback')
    assert state(db_cursor) == before
    db_cursor.execute('SELECT status FROM exceptions WHERE id=%s', (row_id,))
    assert db_cursor.fetchone()['status'] == status
