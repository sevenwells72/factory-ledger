"""Overdue work is never hidden by newer work or the historical opening-stock group."""
import pytest
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import headers, isolated_database  # noqa: F401

pytestmark = pytest.mark.db


def test_exception_limit_keeps_overdue_first(client, actors, db_cursor):
    ids = []
    for status, opened, due in [('open', '-2 days', '-1 day'),
                                ('open', '-1 minute', '1 day'),
                                ('resolved', '1 day', '2 days')]:
        db_cursor.execute("""INSERT INTO exceptions(kind,severity,status,detail,opened_at,due_at,resolved_at)
            VALUES ('PRE_MAKE_ADJUST','warn',%s,'{}',now()+%s::interval,now()+%s::interval,
                    CASE WHEN %s='resolved' THEN now() END)
            RETURNING id""", (status, opened, due, status))
        ids.append(db_cursor.fetchone()['id'])
    response = client.get('/exceptions?kind=PRE_MAKE_ADJUST&status=all&limit=1',
                          headers=headers(actors['owner']['key']))
    assert response.status_code == 200, response.text
    assert [row['id'] for row in response.json()['exceptions']] == [ids[0]]


def test_exception_queue_orders_deadlines_then_new_work_then_opening_stock(client, actors, db_cursor):
    db_cursor.execute("INSERT INTO products(name,type,uom,is_service) VALUES ('Exception queue oats','ingredient','lb',false) RETURNING id")
    product_id = db_cursor.fetchone()['id']
    db_cursor.execute("""INSERT INTO lots(product_id,lot_code,entry_source_notes)
        VALUES (%s,'OPEN-QUEUE','STAGING OPENING STOCK copied from production'),
               (%s,'OPEN-OVERDUE','STAGING OPENING STOCK copied from production') RETURNING id""",
        (product_id, product_id))
    lot_ids = [row['id'] for row in db_cursor.fetchall()]
    ids = []
    for kind, status, opened, due, opening in [
        ('PRE_MAKE_ADJUST', 'open', '-4 days', '-1 day', False),
        ('PRE_MAKE_ADJUST', 'open', '-2 minutes', '1 day', False),
        ('UNIDENTIFIED_LOT', 'open', '-1 minute', '1 day', True),
        ('UNIDENTIFIED_LOT', 'escalated', '-5 days', '-2 days', True),
        ('PRE_MAKE_ADJUST', 'open', '-3 minutes', None, False),
        ('PRE_MAKE_ADJUST', 'resolved', '1 day', '-3 days', False),
    ]:
        db_cursor.execute("""INSERT INTO exceptions(kind,severity,status,product_id,lot_id,detail,opened_at,due_at,resolved_at,escalated_at)
            VALUES (%s,'warn',%s,%s,%s,'{}',now()+%s::interval,now()+%s::interval,
                    CASE WHEN %s='resolved' THEN now() END,
                    CASE WHEN %s='escalated' THEN now() END) RETURNING id""",
            (kind, status, product_id, lot_ids.pop() if opening else None, opened, due, status, status))
        ids.append(db_cursor.fetchone()['id'])
    for limit in [1, 3, 100]:
        response = client.get(f'/exceptions?product_id={product_id}&status=all&limit={limit}',
                              headers=headers(actors['owner']['key']))
        assert response.status_code == 200, response.text
        rows = response.json()['exceptions']
        assert [row['id'] for row in rows] == [ids[i] for i in [3, 0, 1, 4, 2, 5]][:limit]
        assert rows[0]['overdue'] is True
        assert rows[0]['is_opening_stock'] is True
