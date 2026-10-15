"""Round-B regressions for catalog identity, numeric replies, and floor Spanish."""
from pathlib import Path

import pytest

import resolution as r


@pytest.fixture
def round_b_catalog(db_cursor):
    cur = db_cursor
    cur.execute((Path(__file__).parents[1] / 'migrations/060_search_aliases.sql').read_text())
    cur.execute('DELETE FROM search_aliases')
    cur.execute("INSERT INTO search_aliases(kind,alias,expansion) VALUES ('token','SS','Sunshine')")
    def product(name, kind='ingredient', **fields):
        active = fields.pop('active', True)
        columns = ['name', 'type', 'active', *fields]
        cur.execute(f"INSERT INTO products ({','.join(columns)}) VALUES ({','.join(['%s'] * len(columns))}) RETURNING id",
                    (name, kind, active, *fields.values()))
        return cur.fetchone()['id']
    return cur, product


def resolved(catalog, kind, query, **context):
    return r.resolve(catalog[0], r.ResolveRequest(kind=kind, query=query, context=context))


def test_sunshine_customer_is_not_the_product_ss_alias(round_b_catalog):
    cur, _ = round_b_catalog
    cur.execute("INSERT INTO customers(name) VALUES ('Sunshine Granola'), ('SS') RETURNING id")
    sunshine, unrelated_ss = [row['id'] for row in cur.fetchall()]
    result = resolved(round_b_catalog, 'customer', 'Sunshine')
    assert result['match']['id'] == sunshine
    assert result['match']['id'] != unrelated_ss


@pytest.mark.parametrize('query,name', [('Sunshine', 'Sunshine Granola'), ('Setton', 'Setton International Foods'), ('UNIPRO', 'U.N.I.P.R.O. Foodservice')])
def test_real_customer_short_names(round_b_catalog, query, name):
    cur, _ = round_b_catalog
    cur.execute('INSERT INTO customers(name) VALUES (%s) RETURNING id', (name,))
    expected = cur.fetchone()['id']
    result = resolved(round_b_catalog, 'customer', query)
    assert result['outcome'] == 'match'
    assert result['match']['id'] == expected


def test_customer_short_name_keeps_real_ambiguity(round_b_catalog):
    cur, _ = round_b_catalog
    cur.execute("INSERT INTO customers(name) VALUES ('Setton North'), ('Setton South')")
    result = resolved(round_b_catalog, 'customer', 'Setton')
    assert result['outcome'] == 'ambiguous'
    assert {row['name'] for row in result['candidates']} == {'Setton North', 'Setton South'}


@pytest.mark.parametrize('by_id', [True, False])
def test_numeric_product_number_and_sku_are_exact(round_b_catalog, by_id):
    _, product = round_b_catalog
    expected = product('Round B Sunflower Seeds', odoo_code='70013')
    result = resolved(round_b_catalog, 'product', str(expected) if by_id else '70013')
    assert result['outcome'] == 'match'
    assert result['match']['id'] == expected
    assert result['confidence'] == 'high'
    assert result['match']['tier'] == 'exact'


def test_make_classic_nine_offers_only_unflavored_batches(round_b_catalog):
    _, product = round_b_catalog
    plain = product('Batch Classic Granola #9', 'batch')
    ss = product('Batch SS Classic Granola #9 (Kosher Ignition)', 'batch', kosher_tier='extra_kosher')
    product('Batch Classic Fruit Nut Granola #9', 'batch')
    product('Batch Classic Chocolate Chip Granola #9', 'batch')
    result = resolved(round_b_catalog, 'product', 'Classic #9', action='make')
    assert {row['id'] for row in result['candidates']} == {plain, ss}
    assert result['match'] is None


@pytest.mark.parametrize('spanish,english', [('semillas de girasol', 'Sunflower Seeds'), ('avena', 'Oats'), ('miel', 'Honey'), ('harina de arroz', 'Rice Flour'), ('canela', 'Cinnamon'), ('granola clásica #9', 'Classic Granola #9')])
def test_catalog_spanish_names(round_b_catalog, spanish, english):
    _, product = round_b_catalog
    expected = product(english)
    result = resolved(round_b_catalog, 'product', spanish)
    assert result['outcome'] == 'match'
    assert result['match']['id'] == expected
    assert 'exact' not in result['match']['why'].lower()


def test_spanish_alias_is_bound_to_live_catalog(round_b_catalog):
    _, product = round_b_catalog
    product('Sunflower Seeds', active=False)
    result = resolved(round_b_catalog, 'product', 'semillas de girasol')
    assert result['match'] is None
    assert result['candidates'] == []
    assert 'semillas de girasol' in result['ask']
    assert 'No reconocí' in result['ask']


def test_spanish_ambiguous_product_question_and_real_candidates(round_b_catalog):
    _, product = round_b_catalog
    product('Sunflower Seeds Raw')
    product('Sunflower Seeds Roasted')
    result = resolved(round_b_catalog, 'product', 'semillas de girasol')
    assert result['outcome'] == 'ambiguous'
    assert len(result['candidates']) == 2
    assert result['ask'].startswith('¿Cuál')
    assert all('exact' not in row['why'].lower() for row in result['candidates'])


def test_unrecognized_question_names_input_without_empty_which_one():
    result = r.decide('something unknown', [])
    assert 'something unknown' in result['ask']
    assert not result['ask'].startswith('Which one')
    assert result['candidates'] == []


def test_catalog_query_is_shared_with_plain_language_read_tools():
    assert r.catalog_query('semillas de girasol') == 'sunflower seeds'
    assert r.catalog_query('Granola clásica #9') == 'classic granola #9'
    assert r.catalog_query('UNIPRO') == 'unipro'


def test_product_number_collision_with_sku_never_auto_selects(round_b_catalog):
    _, product = round_b_catalog
    by_id = product('Round B Numbered Batch', 'batch')
    by_sku = product('Round B SKU Batch', 'batch', odoo_code=str(by_id))
    result = resolved(round_b_catalog, 'product', str(by_id), action='make')
    assert result['outcome'] == 'ambiguous'
    assert result['match'] is None
    assert {row['id'] for row in result['candidates']} == {by_id, by_sku}


def test_inactive_product_number_never_resolves(round_b_catalog):
    _, product = round_b_catalog
    inactive = product('Inactive Round B Product', active=False)
    assert resolved(round_b_catalog, 'product', str(inactive))['outcome'] == 'none'


def test_customer_punctuation_keeps_short_word_boundaries(round_b_catalog):
    cur, _ = round_b_catalog
    cur.execute("INSERT INTO customers(name) VALUES ('Setton-Foods'), ('Upsetton Foods') RETURNING id")
    setton, _ = [row['id'] for row in cur.fetchall()]
    result = resolved(round_b_catalog, 'customer', 'Setton')
    assert result['match']['id'] == setton


@pytest.mark.parametrize('identity', ['id', 'sku', 'full_name'])
def test_literal_nine_identity_precedes_classic_nine_family(round_b_catalog, identity):
    cur, product = round_b_catalog
    if identity == 'id':
        cur.execute("INSERT INTO products(id,name,type,active) VALUES (9,'Unrelated literal ninth batch','batch',true)")
        expected, query = 9, '9'
    elif identity == 'sku':
        expected = product('Unrelated literal SKU nine batch', 'batch', odoo_code='9')
        query = '9'
    else:
        expected = product('Batch Classic Granola #9', 'batch')
        query = 'Batch Classic Granola #9'
    product('Batch SS Classic Granola #9', 'batch', kosher_tier='extra_kosher')
    result = resolved(round_b_catalog, 'product', query, action='pack')
    assert result['outcome'] == 'match'
    assert result['match']['id'] == expected and result['match']['tier'] == 'exact'
    assert result['ask'] is None
