"""Round-C stock fixes use real staging names and parent relationships."""
import pytest
import resolution
import assistant_replies as replies
from tests.test_actor_attribution import actors, client  # noqa: F401
from tests.test_fl_assistant import enabled, reason_catalog, chat, model, turn  # noqa: F401

pytestmark = pytest.mark.db


@pytest.fixture
def staging_stock_catalog(db_cursor):
    cur = db_cursor
    result = {}
    # The staging catalog has several other recipes containing Classic #9.
    # It also has differently named finished children of the plain batch.
    for key, name, kind, parent, pounds in [
        ('bulk', 'Batch Classic Granola #9', 'batch', None, 18323.7),
        ('classic', 'Granola Classic 25 LB', 'finished', 'bulk', 9250),
        ('crunchy', 'Granola Crunchy CNS 10 LB Case', 'finished', 'bulk', 1700),
        ('wheat', 'Granola Wheat Free 25 LB', 'finished', 'bulk', 2000),
        ('cq', 'CQ Granola 10 LB', 'finished', 'bulk', 9800),
        ('empty', 'Granola Setton Good Ol 25 LB', 'finished', 'bulk', 0),
        ('fruit_bulk', 'Granola Fruit Nut Batch', 'batch', None, 280),
        ('fruit', 'Granola Fruit Nut 25 LB', 'finished', 'fruit_bulk', 50),
        ('ss', 'Batch Classic Granola #9 — EXTRA-KOSHER (SS)', 'batch', None, 100),
    ]:
        cur.execute("""INSERT INTO products(name,type,uom,active,parent_batch_product_id,case_size_lb,is_service)
                       VALUES (%s,%s,'lb',true,%s,%s,false) RETURNING id,name""",
                    (name, kind, result[parent]['id'] if parent else None, 25 if kind == 'finished' else None))
        result[key] = dict(cur.fetchone())
        if pounds:
            cur.execute('INSERT INTO lots(product_id,lot_code) VALUES (%s,%s) RETURNING id',
                        (result[key]['id'], 'C-STOCK-' + key.upper().replace('_', '-')))
            lot = cur.fetchone()['id']
            cur.execute("INSERT INTO transactions(type,notes) VALUES ('receive','staging catalog stock regression') RETURNING id")
            tid = cur.fetchone()['id']
            cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,%s)',
                        (tid, result[key]['id'], lot, pounds))
    for flavor in ('Fruit Nut', 'Chocolate Chip', 'Cinnamon Raisin', 'Vanilla Almond'):
        for suffix in ('', ' — EXTRA-KOSHER (SS)'):
            cur.execute("INSERT INTO products(name,type,uom,active,is_service) VALUES (%s,'batch','lb',true,false)",
                        ('Batch Classic ' + flavor + ' Granola #9' + suffix,))
    return result


@pytest.mark.parametrize('text,query,lang', [
    ('How much classic number nine do we have?', 'classic number nine', 'en'),
    ('¿Cuánto classic número nueve tenemos?', 'classic número nueve', 'es'),
])
def test_real_catalog_classic_core_then_direct_finished_only(client, chat, staging_stock_catalog, monkeypatch, text, query, lang):
    model(monkeypatch, ('inventory_lookup', {'q': query}))
    response = turn(client, chat, text)
    assert response.status_code == 200, response.text
    card = response.json()['cards'][0]
    assert card['kind'] == 'read' and card['language'] == lang
    assert [row['product'] for row in card['rows']] == [staging_stock_catalog[k]['name'] for k in ('bulk', 'cq', 'classic', 'crunchy', 'wheat')]
    assert card['result']['stock_summary_only'] is True
    assert '18,323.7 lb' in card['summary'] and '22,750 lb' in card['summary']
    assert ('productos terminados' if lang == 'es' else 'finished products') in card['summary']
    assert 'Fruit' not in card['summary'] and 'Wheat' not in card['summary']
    assert card['summary'].count('.') == 2  # Decimal point plus one sentence end.


@pytest.mark.parametrize('text', ['Show Classic #9 inventory including zero stock.', 'Muestra Classic #9 incluso los productos con cero existencias.'])
def test_zero_stock_rows_only_when_requested(client, chat, staging_stock_catalog, monkeypatch, text):
    model(monkeypatch, ('inventory_lookup', {'q': 'Classic #9'}))
    card = turn(client, chat, text).json()['cards'][0]
    assert card['result']['include_zero'] is True
    assert staging_stock_catalog['empty']['name'] in [r['product'] for r in card['rows']]


@pytest.mark.parametrize('query', ['Batch Classic Granola #9', '90002'])
def test_exact_catalog_identity_never_expands_its_children(db_cursor, staging_stock_catalog, query):
    bulk = staging_stock_catalog['bulk']
    db_cursor.execute("UPDATE products SET odoo_code='90002' WHERE id=%s", (bulk['id'],))
    resolved = resolution.resolve(db_cursor, resolution.ResolveRequest(kind='product', query=query))
    matches = resolution.inventory_products(db_cursor, query, resolved)
    assert [p['id'] for p in matches] == [bulk['id']]


@pytest.mark.parametrize('spanish', [False, True])
def test_genuinely_ambiguous_stock_uses_four_short_choices(client, chat, db_cursor, monkeypatch, spanish):
    for i in range(7):
        db_cursor.execute("INSERT INTO products(name,type,uom,active,is_service) VALUES (%s,'ingredient','lb',true,false)", ('RoundC Ambiguous Grain ' + str(i),))
    model(monkeypatch, ('inventory_lookup', {'q': 'RoundC Ambiguous Grain'}))
    response = turn(client, chat, '¿Cuánto hay de RoundC Ambiguous Grain?' if spanish else 'How much RoundC Ambiguous Grain?')
    card = response.json()['cards'][0]
    assert card['kind'] == 'choices'
    assert len(card['result']['candidates']) == 4 and card['result']['has_more'] is False


@pytest.mark.parametrize('text', ['¿Cuánta granola de frutas y nueces hay?', '¿Cuánto tenemos de granola de frutas y nueces?'])
def test_spanish_fruit_nut_uses_real_catalog_recipe(client, chat, staging_stock_catalog, monkeypatch, text):
    model(monkeypatch, ('inventory_lookup', {'q': 'granola de frutas y nueces'}))
    card = turn(client, chat, text).json()['cards'][0]
    assert card['kind'] == 'read' and card['language'] == 'es'
    assert {r['product'] for r in card['rows']} == {staging_stock_catalog[k]['name'] for k in ('fruit_bulk', 'fruit')}


def test_spanish_fruit_nut_resolves_for_write_actions(db_cursor, staging_stock_catalog):
    result = resolution.resolve(db_cursor, resolution.ResolveRequest(kind='product', query='granola de frutas y nueces', context={'action': 'make'}))
    assert result['outcome'] == 'match'
    assert result['match']['id'] == staging_stock_catalog['fruit_bulk']['id']


@pytest.mark.parametrize('text', ['include zero', 'even empty products', 'incluye cero', 'sin existencias'])
def test_explicit_empty_stock_vocabulary(text):
    assert replies.include_zero_stock(text)


@pytest.mark.parametrize('lang,expected', [('en', '(80 cases)'), ('es', '(80 cajas)')])
def test_plain_stock_summary_retains_supplied_finished_case_count(lang, expected):
    result = {'results': [{'product': 'Granola Classic 25 LB', 'total_on_hand': 2000, 'unit_count': 80, 'lots': [{'lot': 'CLASSIC-1', 'qty_on_hand': 2000}]}]}
    summary = replies.read_card('inventory_lookup', result, lang)['summary']
    assert '2,000 lb' in summary and expected in summary


@pytest.mark.parametrize('spanish', [False, True])
def test_hiding_zero_physical_stock_keeps_unverified_tag_evidence(client, chat, db_cursor, monkeypatch, spanish):
    db_cursor.execute("INSERT INTO products(name,type,uom,active,is_service) VALUES ('C Unverified Only Oats','ingredient','lb',true,false) RETURNING id")
    pid = db_cursor.fetchone()['id']
    code = 'UNVERIFIED-999-P%s-1' % pid
    db_cursor.execute("INSERT INTO lots(product_id,lot_code,entry_source) VALUES (%s,%s,'unverified_make_input') RETURNING id", (pid,code))
    lid = db_cursor.fetchone()['id']
    db_cursor.execute("INSERT INTO transactions(type,notes) VALUES ('make','unverified physical input') RETURNING id")
    tid = db_cursor.fetchone()['id']
    db_cursor.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES (%s,%s,%s,-10)', (tid,pid,lid))
    model(monkeypatch, ('inventory_lookup', {'q': 'C Unverified Only Oats'}))
    response = turn(client, chat, '¿Cuánto hay de C Unverified Only Oats?' if spanish else 'How much C Unverified Only Oats?')
    card = response.json()['cards'][0]
    assert card['kind'] == 'read' and card['rows'] == []
    assert ('No hay existencias' if spanish else 'There is no stock') in card['summary']
    assert ('pendientes de seguimiento' if spanish else 'unverified, awaiting follow-up') in card['summary']
    assert '-10' not in card['summary']
    assert [lot['lot'] for lot in card['result']['unverified_evidence']] == [code]
    db_cursor.execute('SELECT quantity_lb FROM transaction_lines WHERE transaction_id=%s', (tid,))
    assert db_cursor.fetchone()['quantity_lb'] == -10
