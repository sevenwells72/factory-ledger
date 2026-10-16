"""The server's limited queue must return new unresolved work before old overdue rows."""
import pytest
from tests.test_actor_attribution import client, actors  # noqa: F401
from tests.test_write_tickets import headers, isolated_database  # noqa: F401

pytestmark = pytest.mark.db


def test_exception_limit_keeps_newest_unresolved_first(client, actors, db_cursor):
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
    assert [row['id'] for row in response.json()['exceptions']] == [ids[1]]
