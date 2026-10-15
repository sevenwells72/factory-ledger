"""Round-C unknown spoken pack lots keep real FIFO choices and tag gates."""
import pytest

from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import enabled, reason_catalog, chat, model, turn, record  # noqa: F401
from tests.test_assistant_voice_e2e_round_c import voice_catalog, card  # noqa: F401

pytestmark = pytest.mark.db


@pytest.fixture
def pack_lots(voice_catalog, db_cursor):
    source = voice_catalog['fruit_batch']
    # The original fixture's made-up lot 1234 concealed the review failure.
    db_cursor.execute("UPDATE lots SET lot_code='B26-1015-004',received_at='2026-10-15T12:00:00Z' WHERE id=%s", (source['lot_id'],))
    source['lot_code'] = 'B26-1015-004'
    db_cursor.execute("INSERT INTO lots(product_id,lot_code,received_at) VALUES (%s,'26-10-14-GRAI-003','2026-10-14T12:00:00Z') RETURNING id", (source['id'],))
    first_id = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('receive','Round C real-format earlier pack lot') RETURNING id")
    tid = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,1000)', (tid, source['id'], first_id))
    db_cursor.execute("INSERT INTO lots(product_id,lot_code) VALUES (%s,'DEPLETED-PACK-LOT')", (source['id'],))
    return {'first_id': first_id, 'later_id': source['lot_id']}


@pytest.mark.parametrize('text,lot,lang', [
    ('Packed 40 cases of fruit and nut granola from batch lot one two three four.', 'one two three four', 'en'),
    ('Empacamos cuarenta cajas de fruit and nut granola del lote uno dos tres cuatro.', 'uno dos tres cuatro', 'es'),
])
def test_unknown_spoken_pack_lot_offers_only_parent_lots_fifo_then_keeps_selected_tag(client, chat, voice_catalog,
        pack_lots, monkeypatch, text, lot, lang):
    source, target = voice_catalog['fruit_batch'], voice_catalog['fruit']
    model(monkeypatch,
        ('resolve', {'kind': 'product', 'query': 'fruit and nut granola', 'context': {'action': 'pack'}}),
        ('packing_context', {'target_product_id': target['id'], 'lot_code': lot}))
    choices = card(turn(client, chat, text), 'choices')
    assert choices['language'] == lang and choices['resolution_kind'] == 'lot'
    assert choices['packing_lots'] is True
    assert ('No encontré el lote 1234' if lang == 'es' else "I couldn't find lot 1234") in choices['message']
    candidates = choices['result']['candidates']
    assert [row['id'] for row in candidates] == [pack_lots['first_id'], pack_lots['later_id']]
    assert [row['lot_code'] for row in candidates] == ['26-10-14-GRAI-003', 'B26-1015-004']
    assert all(row['product_id'] == source['id'] and '1,000 lb' in row['label'] for row in candidates)
    # A worker may choose the newer physical lot. The context must not silently
    # fall back to the oldest lot when the model omits lot_allocations.
    model(monkeypatch, ('prepare_pack', {'source_product_id': source['id'], 'target_product_id': target['id'], 'cases': 40}))
    draft = card(turn(client, chat, choice_id=choices['id'], selected_id=pack_lots['later_id']), 'draft')
    assert draft['prepared']['draft']['cases'] == 40
    assert draft['prepared']['draft']['input_plan'][0]['lot_id'] == pack_lots['later_id']
    assert all(not row.get('confirmed') for row in draft['prepared']['draft']['input_plan'])
    assert draft['prepared']['can_commit'] is False
    refused = record(client, chat, draft)
    assert refused.status_code == 422 and 'LOT_NOT_CONFIRMED' in refused.text


@pytest.mark.parametrize('spanish', [False, True])
def test_none_of_unknown_pack_lot_choices_keeps_lot_wording(client, chat, voice_catalog, pack_lots,
        monkeypatch, spanish):
    target = voice_catalog['fruit']
    model(monkeypatch,
        ('resolve', {'kind': 'product', 'query': 'fruit and nut granola', 'context': {'action': 'pack'}}),
        ('packing_context', {'target_product_id': target['id'], 'lot_code': '1234'}))
    choices = card(turn(client, chat, 'Empacamos 40 cajas del lote 1234' if spanish else 'Packed 40 cases from lot 1234'), 'choices')
    result = card(turn(client, chat, choice_id=choices['id'], selected_id=None), 'clarify')
    assert result['message'] == ('No hay un lote llamado 1234.' if spanish else 'No lot called 1234.')
    assert 'SKU' not in result['message']
