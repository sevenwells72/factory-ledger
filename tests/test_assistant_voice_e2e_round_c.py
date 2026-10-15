"""Round C phone transcripts through real routes, resolution, tickets and ledger.

Only OpenAI tool selection is deterministic here. These checks prove normalized
model input and real FL behavior; they do not claim real speech/model accuracy.
"""
import json
import re

import pytest

from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import enabled, reason_catalog, chat, model, turn, record
from tests.test_write_tickets import headers
from tests.test_assistant_today_entries import seed_day

pytestmark = pytest.mark.db


@pytest.fixture
def voice_catalog(db_cursor):
    cur = db_cursor
    catalog = {}
    for key, name, kind in [
        ('seeds', 'Sunflower Seeds', 'ingredient'),
        ('classic', 'Batch Classic Granola #9', 'batch'),
        ('ss1', 'Batch SS #1', 'batch'), ('ss2', 'Batch SS #2', 'batch'),
        ('ss3', 'Batch SS #3', 'batch'),
        ('fruit_batch', 'Fruit and Nut Granola Batch', 'batch'),
        ('fruit', 'Fruit and Nut Granola', 'finished'),
        ('sprinkles', 'Rainbow Sprinkles', 'finished'),
    ]:
        cur.execute("""INSERT INTO products(name,type,uom,default_batch_lb,case_size_lb,active)
            VALUES (%s,%s,'lb',10,5,true) RETURNING id,name""", (name, kind))
        catalog[key] = dict(cur.fetchone())
    cur.execute('UPDATE products SET parent_batch_product_id=%s WHERE id=%s',
                (catalog['fruit_batch']['id'], catalog['fruit']['id']))
    for key in ('classic', 'ss1', 'ss2', 'ss3'):
        cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES (%s,%s,10)',
                    (catalog[key]['id'], catalog['seeds']['id']))
    for key, code in [('seeds', 'SUNFLOWER-ABCD'), ('classic', 'CLASSIC-9012'),
                      ('fruit_batch', '1234'), ('ss1', 'P1585')]:
        item = catalog[key]
        cur.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id', (item['id'], code))
        item.update(lot_id=cur.fetchone()['id'], lot_code=code)
        cur.execute("INSERT INTO transactions(type,notes) VALUES ('receive','Round C fixture') RETURNING id")
        tid = cur.fetchone()['id']
        cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,1000)',
                    (tid, item['id'], item['lot_id']))
    catalog['suppliers'] = []
    for name in ('Grain Supply', 'A-1 Bakery Supply', 'Bakers Depot'):
        cur.execute('INSERT INTO suppliers(name) VALUES (%s) RETURNING id,name', (name,))
        catalog['suppliers'].append(dict(cur.fetchone()))
    catalog['customers'] = []
    for name in ('Lake Bakery', 'Rainbow Bakery'):
        cur.execute('INSERT INTO customers(name,active) VALUES (%s,true) RETURNING id,name', (name,))
        catalog['customers'].append(dict(cur.fetchone()))
    return catalog


def card(response, kind=None):
    assert response.status_code == 200, response.text
    result = response.json()['cards'][0]
    if kind:
        assert result['kind'] == kind, result
    assert not re.search(r"Couldn't look that up|Please clarify:|Internal server error|Which one\?", json.dumps(result))
    return result


def supplied_text(requests):
    return '\n'.join(str(row.get('content', '')) for request in requests for row in request['input']
                     if row.get('role') == 'user')


def record_with_physical_tags(client, chat, draft):
    prepared = draft['prepared']
    assert prepared['can_commit'] is False
    assert {b['code'] for b in prepared['blockers']} == {'LOT_NOT_CONFIRMED'}
    assert all(not row.get('confirmed') for row in prepared['draft']['input_plan'])
    rejected = record(client, chat, draft)
    assert rejected.status_code == 422 and 'LOT_NOT_CONFIRMED' in rejected.text
    evidence = [{'lot_id': row['lot_id'], 'method': 'full_code', 'value': row['lot_code']}
                for row in prepared['draft']['input_plan']]
    committed = record(client, chat, draft, lot_confirmations=evidence)
    assert committed.status_code == 200, committed.text
    assert committed.json()['kind'] == 'receipt'
    return committed.json()['result']


@pytest.mark.parametrize('text,query,lang', [
    ('How much classic number nine do we have?', 'classic number nine', 'en'),
    ('How much classic no. 9 do we have?', 'classic no. 9', 'en'),
    ('How much classic nine do we have?', 'classic nine', 'en'),
    ('¿Cuánto tenemos de classic número nueve?', 'classic número nueve', 'es'),
])
def test_spoken_classic_inventory_exact_phone_transcript(client, chat, voice_catalog, monkeypatch, text, query, lang):
    requests = model(monkeypatch, ('inventory_lookup', {'q': query}))
    result = card(turn(client, chat, text), 'read')
    assert result['language'] == lang
    assert [row['product'] for row in result['rows']] == [voice_catalog['classic']['name']]
    assert result['rows'][0]['total_on_hand'] == 1000
    assert '#9' in supplied_text(requests)


@pytest.mark.parametrize('text,query,key,batches,lang', [
    ('Made two batches of classic number nine.', 'classic number nine', 'classic', 2, 'en'),
    ('Hicimos dos tandas de classic número nueve.', 'classic número nueve', 'classic', 2, 'es'),
    ('Made one batch of SS number one.', 'SS number one', 'ss1', 1, 'en'),
    ('Hicimos una tanda de SS número uno.', 'SS número uno', 'ss1', 1, 'es'),
    ('Made one batch of Sunshine number one.', 'Sunshine number one', 'ss1', 1, 'en'),
    ('Hicimos una tanda de Sunshine número uno.', 'Sunshine número uno', 'ss1', 1, 'es'),
    ('Made two batches of SS number two.', 'SS number two', 'ss2', 2, 'en'),
    ('Hicimos dos tandas de SS número dos.', 'SS número dos', 'ss2', 2, 'es'),
    ('Made three batches of SS number three.', 'SS number three', 'ss3', 3, 'en'),
    ('Hicimos tres tandas de SS número tres.', 'SS número tres', 'ss3', 3, 'es'),
])
def test_spoken_make_records_only_after_physical_tags(client, chat, voice_catalog, db_cursor, monkeypatch,
                                                     text, query, key, batches, lang):
    target = voice_catalog[key]
    requests = model(monkeypatch, ('resolve', {'kind': 'product', 'query': query, 'context': {'action': 'make'}}),
                     ('prepare_make', {'product_id': target['id'], 'batches': batches}))
    draft = card(turn(client, chat, text))
    if draft['kind'] == 'choices':
        candidates = draft['result']['candidates']
        assert candidates and any(row['id'] == target['id'] for row in candidates)
        model(monkeypatch, ('prepare_make', {'product_id': target['id'], 'batches': batches}))
        draft = card(turn(client, chat, choice_id=draft['id'], selected_id=target['id']), 'draft')
    assert draft['language'] == lang
    assert draft['prepared']['draft']['batches'] == batches
    assert draft['prepared']['draft']['total_output_lb'] == batches * 10
    assert ('#9' if key == 'classic' else 'SS #' + key[-1]) in supplied_text(requests)
    result = record_with_physical_tags(client, chat, draft)
    assert result['receipt_number'].startswith('MK-')
    db_cursor.execute('SELECT count(*) AS n FROM transaction_lot_confirmations WHERE transaction_id=%s', (result['transaction_id'],))
    assert db_cursor.fetchone()['n'] == len(draft['prepared']['draft']['input_plan'])


@pytest.mark.parametrize('text,product_query,bol_word,lang', [
    ('We received four bags of sunflower seeds, 50 pounds each, from Green Supply, POL 5521.', 'sunflower seeds', 'POL', 'en'),
    ('Recibimos cuatro bolsas de semillas de girasol, cincuenta libras cada una, de Green Supply, POL 5521.', 'semillas de girasol', 'POL', 'es'),
    ('We received four bags of sunflower seeds, fifty lbs each, from Green Supply, B.O.L. 5521.', 'sunflower seeds', 'B.O.L.', 'en'),
    ('Recibimos cuatro bolsas de semillas de girasol, cincuenta libras cada una, de Green Supply, B O L 5521.', 'semillas de girasol', 'B O L', 'es'),
    ('We received four bags of sunflower seeds, fifty pounds each, from Green Supply, bowl 5521.', 'sunflower seeds', 'bowl', 'en'),
])
def test_misheard_receive_supplier_keeps_spoken_quantity_and_bol(client, chat, voice_catalog, db_cursor,
                                                               monkeypatch, text, product_query, bol_word, lang):
    product = voice_catalog['seeds']
    requests = model(monkeypatch, ('resolve', {'kind': 'product', 'query': product_query, 'context': {'action': 'receive'}}),
                     ('resolve', {'kind': 'supplier', 'query': 'Green Supply', 'context': {'action': 'receive'}}))
    choices = card(turn(client, chat, text), 'choices')
    assert choices['resolution_kind'] == 'supplier' and choices['language'] == lang
    grain = next(row for row in choices['result']['candidates'] if row['name'] == 'Grain Supply')
    assert 'Grain Supply' not in choices['message'], 'Names appear on buttons only'
    normalized = supplied_text(requests)
    assert '4' in normalized and '50 lb' in normalized and 'BOL 5521' in normalized
    body = {'product_id': product['id'], 'supplier_id': grain['id'], 'cases': 4, 'case_size_lb': 50,
            'bol_reference': '5521'}
    model(monkeypatch, ('prepare_receive', body))
    draft = card(turn(client, chat, choice_id=choices['id'], selected_id=grain['id']), 'draft')
    assert draft['prepared']['draft']['total_lb'] == 200
    committed = record(client, chat, draft)
    assert committed.status_code == 200, committed.text
    result = committed.json()['result']
    assert result['receipt_number'].startswith('RCV-')
    db_cursor.execute('SELECT supplier_id,bol_reference FROM transactions WHERE id=%s', (result['transaction_id'],))
    assert dict(db_cursor.fetchone()) == {'supplier_id': grain['id'], 'bol_reference': '5521'}


@pytest.mark.parametrize('text,lot,lang', [
    ('Packed 40 cases of fruit and nut granola from batch lot one two three four.', 'one two three four', 'en'),
    ('Empacamos cuarenta cajas de fruit and nut granola del lote uno dos tres cuatro.', 'uno dos tres cuatro', 'es'),
])
def test_spoken_pack_transcript_records_exact_source_after_tag(client, chat, voice_catalog, monkeypatch, text, lot, lang):
    source, target = voice_catalog['fruit_batch'], voice_catalog['fruit']
    requests = model(monkeypatch,
        ('resolve', {'kind': 'product', 'query': 'fruit and nut granola', 'context': {'action': 'pack'}}),
        ('packing_context', {'target_product_id': target['id'], 'lot_code': lot}),
        ('prepare_pack', {'source_product_id': source['id'], 'target_product_id': target['id'], 'cases': 40}))
    draft = card(turn(client, chat, text), 'draft')
    assert draft['language'] == lang and draft['prepared']['draft']['cases'] == 40
    assert draft['prepared']['draft']['total_lb'] == 200
    assert draft['prepared']['draft']['input_plan'][0]['lot_id'] == source['lot_id']
    assert '1234' in supplied_text(requests)
    assert record_with_physical_tags(client, chat, draft)['receipt_number'].startswith('PK-')


@pytest.mark.parametrize('text,code', [
    ('How much is in lot P fifteen eighty-five?', 'P fifteen eighty-five'),
    ('How much is in lot P one five eight five?', 'P one five eight five'),
    ('¿Cuánto hay en el lote P quince ochenta y cinco?', 'P quince ochenta y cinco'),
    ('¿Cuánto hay en el lote P uno cinco ocho cinco?', 'P uno cinco ocho cinco'),
])
def test_spoken_letter_digit_lot_uses_existing_exact_lot(client, chat, voice_catalog, monkeypatch, text, code):
    requests = model(monkeypatch, ('lot_lookup', {'lot_code': code}))
    result = card(turn(client, chat, text), 'read')
    assert result['rows'][0]['lot_code'] == 'P1585'
    assert result['rows'][0]['quantity_on_hand'] == 1000
    assert 'P1585' in supplied_text(requests)


@pytest.mark.parametrize('text,lang', [
    ('New order from fake bakery. Five cases of rainbow sprinkles.', 'en'),
    ('Nuevo pedido de fake bakery. Cinco cajas de rainbow sprinkles.', 'es'),
])
def test_unknown_order_customer_buttons_never_offer_supplier_ids_and_can_record(client, actors, voice_catalog,
                                                                               db_cursor, monkeypatch, text, lang):
    h = headers(actors['office']['key'])
    session = client.post('/assistant/session', headers=h)
    office_chat = (session.json()['session_id'], h)
    requests = model(monkeypatch, ('resolve', {'kind': 'customer', 'query': 'fake bakery', 'context': {'action': 'order'}}))
    choices = card(turn(client, office_chat, text), 'choices')
    assert choices['language'] == lang and choices['resolution_kind'] == 'customer'
    candidates = choices['result']['candidates']
    assert candidates
    assert {(row['id'], row['name']) for row in candidates} <= {(row['id'], row['name']) for row in voice_catalog['customers']}
    assert all(row['name'] not in ('A-1 Bakery Supply', 'Bakers Depot') for row in candidates)
    assert all(row['name'] not in choices['message'] for row in candidates)
    assert '5' in supplied_text(requests)
    selected = next(row for row in candidates if row['name'] == 'Lake Bakery')
    target = voice_catalog['sprinkles']
    model(monkeypatch, ('resolve', {'kind': 'product', 'query': 'rainbow sprinkles', 'context': {'action': 'order'}}),
          ('prepare_create_order', {'customer_id': selected['id'], 'lines': [{'product_id': target['id'], 'quantity': 5, 'unit': 'cases'}]}))
    draft = card(turn(client, office_chat, choice_id=choices['id'], selected_id=selected['id']), 'draft')
    assert draft['prepared']['action'] == 'create_order'
    committed = record(client, office_chat, draft)
    assert committed.status_code == 200, committed.text
    result = committed.json()['result']
    assert result['receipt_number'].startswith('SO-') or result['receipt_number'].startswith('ORD-')
    db_cursor.execute('SELECT customer_id FROM sales_orders WHERE ticket_id=%s', (draft['prepared']['ticket_id'],))
    assert db_cursor.fetchone()['customer_id'] == selected['id']


@pytest.mark.parametrize('text,lang', [
    ("Open today's entries. Is everything there correct?", 'en'),
    ("Open today's entries", 'en'), ('Is everything there correct?', 'en'),
    ('What did I record today?', 'en'), ("Show today's entries", 'en'),
    ('¿Qué registré hoy?', 'es'), ('Abre las entradas de hoy. ¿Está todo correcto?', 'es'),
])
def test_today_voice_queries_return_plain_worker_team_summary_and_newest_rows(client, chat, actors, db_cursor,
                                                                            monkeypatch, text, lang):
    receipts = seed_day(db_cursor, actors, 6)
    requests = model(monkeypatch, ('today_entries', {}))
    response = turn(client, chat, text)
    result = card(response, 'read')
    assert result['tool'] == 'today_entries' and result['language'] == lang
    assert '3' in result['summary'] and '6' in result['summary']
    assert re.search('equipo' if lang == 'es' else 'team', result['summary'], re.I)
    actual = [row['receipt_number'] for row in result['rows']]
    assert actual in (receipts[::-1], receipts[::2][::-1])
    assert len(actual) >= 3
    assert 'correct' not in result['summary'].lower(), 'The assistant reads receipts without certifying correctness'
    assert requests or result['rows'], 'Deterministic direct Today routing is also permitted'


@pytest.mark.parametrize('text,lang', [
    ('How much sunflour seeds do we have?', 'en'),
    ('¿Cuánto tenemos de sunflour seeds?', 'es'),
])
def test_misheard_product_offers_real_button_then_inventory(client, chat, voice_catalog, monkeypatch, text, lang):
    model(monkeypatch, ('inventory_lookup', {'q': 'sunflour seeds'}))
    choices = card(turn(client, chat, text), 'choices')
    assert choices['language'] == lang and choices['resolution_kind'] == 'product'
    selected = next(row for row in choices['result']['candidates'] if row['id'] == voice_catalog['seeds']['id'])
    assert selected['name'] == 'Sunflower Seeds'
    assert 'Sunflower Seeds' not in choices['message']
    model(monkeypatch, ('inventory_lookup', {'q': selected['name']}))
    result = card(turn(client, chat, choice_id=choices['id'], selected_id=selected['id']), 'read')
    assert result['rows'][0]['product'] == 'Sunflower Seeds'
    assert result['rows'][0]['total_on_hand'] == 1000
