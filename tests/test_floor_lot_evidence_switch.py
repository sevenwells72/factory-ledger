"""The default-off Floor rollout switch closes omitted/null evidence fallback."""
import pytest

from tests.test_actor_attribution import client  # noqa: F401
from tests.test_write_tickets import headers
from tests.test_write_tickets_part2 import items  # noqa: F401

pytestmark = pytest.mark.db


def request_body(action, items):
    if action == 'make':
        return {'product_name': items['batch']['name'], 'batches': 1}
    return {'source_product': items['batch']['name'], 'target_product': items['finished']['name'], 'cases': 2}


@pytest.mark.parametrize('action', ['make', 'pack'])
@pytest.mark.parametrize('flag', [None, '0', 'false'])
def test_default_off_retains_legacy_contract(client, items, monkeypatch, action, flag):
    if flag is None:
        monkeypatch.delenv('REQUIRE_MAKE_PACK_LOT_EVIDENCE', raising=False)
    else:
        monkeypatch.setenv('REQUIRE_MAKE_PACK_LOT_EVIDENCE', flag)
    result = client.post('/'+action, json=request_body(action, items) | {'mode': 'commit'}, headers=headers())
    assert result.status_code == 200, result.text


@pytest.mark.parametrize('action', ['make', 'pack'])
@pytest.mark.parametrize('fields', [{}, {'ingredient_lot_allocations': []}, {'lot_confirmations': []},
                                    {'ingredient_lot_allocations': None, 'lot_confirmations': None}])
def test_enabled_refuses_missing_or_null_arrays_without_writes(client, items, db_cursor, monkeypatch, action, fields):
    monkeypatch.setenv('REQUIRE_MAKE_PACK_LOT_EVIDENCE', '1')
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    before = db_cursor.fetchone()['n']
    result = client.post('/'+action, json=request_body(action, items) | {'mode': 'commit'} | fields, headers=headers())
    assert result.status_code == 422, result.text
    detail = result.json()['detail']
    assert detail['error_code'] == 'MAKE_PACK_LOT_EVIDENCE_REQUIRED'
    assert detail['message'] and detail['message_es'] and detail['required_fields']
    db_cursor.execute('SELECT count(*) AS n FROM transactions')
    assert db_cursor.fetchone()['n'] == before


@pytest.mark.parametrize('action', ['make', 'pack'])
def test_enabled_preview_never_confirms_fifo_and_empty_arrays_do_not_post(client, items, monkeypatch, action):
    monkeypatch.setenv('REQUIRE_MAKE_PACK_LOT_EVIDENCE', 'true')
    payload = request_body(action, items)
    preview = client.post('/'+action, json=payload, headers=headers())
    assert preview.status_code == 200, preview.text
    assert not preview.json()['can_commit']
    assert preview.json()['lot_confirmation_group']['ingredients']
    result = client.post('/'+action, json=payload | {'mode': 'commit',
        'ingredient_lot_allocations': [], 'lot_confirmations': []}, headers=headers())
    assert result.status_code == 422, result.text
    assert result.json()['detail']['error_code'] == 'LOT_NOT_CONFIRMED'


@pytest.mark.parametrize('action', ['make', 'pack'])
def test_enabled_accepts_explicit_verified_lots(client, items, monkeypatch, action):
    monkeypatch.setenv('REQUIRE_MAKE_PACK_LOT_EVIDENCE', 'on')
    ingredient = items['ingredient' if action == 'make' else 'batch']
    payload = request_body(action, items) | {'mode': 'commit',
        'ingredient_lot_allocations': [{'ingredient_product_id': ingredient['id'],
            'lot_id': ingredient['lot_id'], 'quantity_lb': 10}],
        'lot_confirmations': [{'lot_id': ingredient['lot_id'], 'method': 'full_code', 'value': ingredient['lot_code']}]}
    result = client.post('/'+action, json=payload, headers=headers())
    assert result.status_code == 200, result.text
    assert all(row['confirmed'] for row in result.json()['input_plan'])
