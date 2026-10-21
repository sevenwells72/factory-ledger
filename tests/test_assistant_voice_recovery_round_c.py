"""Round C deterministic Today routing and human-only fuzzy identity recovery."""
import pytest
import assistant_replies as replies
from tests.test_actor_attribution import actors, client
from tests.test_fl_assistant import enabled, reason_catalog, chat, model, turn
from tests.test_assistant_today_entries import seed_day


@pytest.mark.parametrize('text', ["Open today's entries", "Open today's entries. Is everything there correct?",
    'Is everything there correct?', 'Qué registré hoy', 'What did I record today?', "Show today's entries", '¿Qué registré hoy?',
    'Abre las entradas de hoy. ¿Está todo correcto?'])
def test_day_review_intents(text):
    assert replies.is_today_query(text)


@pytest.mark.parametrize('text', ["Adjust today's entries by 5 lb", "Void today's entries", 'Made four batches today',
    'Recibimos cuatro bolsas hoy', 'Corrige las entradas de hoy', 'Ajustar los registros de hoy',
    'Anular las entradas de hoy', 'Record today’s entries', 'Record 10 lb today', 'Change today records', 'What inventory do we have today?'])
def test_day_review_never_swallows_writes_or_stock_questions(text):
    assert not replies.is_today_query(text)


@pytest.mark.db
@pytest.mark.parametrize('spanish', [False, True])
def test_direct_day_review_works_without_model_and_keeps_counts_with_bounded_preview(client, chat, actors, db_cursor, monkeypatch, spanish):
    expected = seed_day(db_cursor, actors, 601)
    import fl_assistant
    async def unavailable(*args, **kwargs):
        raise AssertionError('Today must not need the model')
    monkeypatch.setattr(fl_assistant, 'openai', unavailable)
    result = turn(client, chat, '¿Qué registré hoy?' if spanish else 'What did I record today?')
    assert result.status_code == 200, result.text
    card = result.json()['cards'][0]
    assert card['result']['own_count'] == 301 and card['result']['team_count'] == 601
    assert [row['receipt_number'] for row in card['rows']] == expected[::-1][:20]
    assert card['result']['view_all'] is True
    assert len(result.content) < 50000


@pytest.mark.db
@pytest.mark.parametrize('kind,name,query', [
    ('supplier', 'Grain Supply', 'Green Supply'), ('product', 'Sunflower Seeds', 'Sunflour Seeds'),
    ('customer', 'Lake Bakery', 'Fake Bakery'),
])
def test_similar_name_is_a_real_button_never_approved_before_click(client, chat, db_cursor, monkeypatch, kind, name, query):
    table = {'supplier': 'suppliers', 'customer': 'customers', 'product': 'products'}[kind]
    extra = ",type,is_service" if kind == 'product' else ''
    value = ",'ingredient',false" if kind == 'product' else ''
    db_cursor.execute(f'INSERT INTO {table}(name{extra}) VALUES (%s{value}) RETURNING id', (name,))
    identity = db_cursor.fetchone()['id']
    model(monkeypatch, ('resolve', {'kind': kind, 'query': query}))
    response = turn(client, chat, 'Look up ' + query)
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['kind'] == 'choices' and card['resolution_kind'] == kind
    assert any(c['id'] == identity for c in card['result']['candidates'])
    assert name not in card['message'] and '1)' not in card['message']
    db_cursor.execute('SELECT state FROM assistant_sessions WHERE id=%s', (chat[0],))
    assert identity not in db_cursor.fetchone()['state'].get('resolved', {}).get(kind, [])


@pytest.mark.db
@pytest.mark.parametrize('spanish', [False, True])
def test_misspelled_inventory_offers_catalog_buttons(client, chat, db_cursor, monkeypatch, spanish):
    db_cursor.execute("INSERT INTO products(name,type,is_service) VALUES ('Sunflower Seeds','ingredient',false) RETURNING id")
    identity = db_cursor.fetchone()['id']
    model(monkeypatch, ('inventory_lookup', {'q': 'Sunflour Seeds'}))
    response = turn(client, chat, '¿Cuánto tenemos de Sunflour Seeds?' if spanish else 'How much Sunflour Seeds do we have?')
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['kind'] == 'choices'
    assert any(c['id'] == identity for c in card['result']['candidates'])
    assert card['language'] == ('es' if spanish else 'en')


@pytest.mark.parametrize("text", ["Qué registré hoy", "Que registre hoy", "Muestra mis entradas hoy"])
def test_spanish_dictation_without_punctuation(text):
    assert replies.language(text) == "es"


@pytest.mark.parametrize('given,expected', [
    ('one two three four', '1234'), ('lot one two three four', '1234'),
    ('lote uno dos tres cuatro', '1234'), ('lot number one two three four', '1234'),
    ('P quince ochenta y cinco', 'P1585'), ('lote P quince ochenta y cinco', 'P1585'),
    ('LOT 1234', 'LOT 1234'), ('LOT-1234', 'LOT-1234'),
])
def test_model_lot_argument_with_or_without_spoken_label(given, expected):
    from fl_assistant import voice_lot_code
    assert voice_lot_code(given) == expected


@pytest.mark.parametrize('text', ['¿Está todo ahí correcto?', '¿Está todo correcto?'])
def test_spanish_review_followup_is_direct_read(text):
    assert replies.is_today_query(text)


@pytest.mark.db
@pytest.mark.parametrize('spanish', [False, True])
def test_literal_existing_word_tag_is_never_normalized(client, chat, db_cursor, monkeypatch, spanish):
    db_cursor.execute("INSERT INTO products(name,type,is_service) VALUES ('Literal tag oats','ingredient',false) RETURNING id")
    product = db_cursor.fetchone()['id']
    code = 'UNO DOS TRES CUATRO' if spanish else 'ONE TWO THREE FOUR'
    db_cursor.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s)', (product, code))
    requests = model(monkeypatch, ('lot_lookup', {'lot_code': code}))
    text = ('¿Cuánto hay en el lote ' if spanish else 'How much is in lot ') + code + '?'
    result = turn(client, chat, text)
    assert result.status_code == 200, result.text
    assert result.json()['cards'][0]['rows'][0]['lot_code'] == code
    assert code in str(requests[0]['input'])
