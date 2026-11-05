"""DCC-006: Office customer/product typeahead from the first character.

Prefix candidates are shown for a person to choose; they are never automatic
matches and never "known customer" keyword identities (PR #148 / row 233 intent).
Local disposable PostgreSQL only.
"""
from pathlib import Path

import pytest

import resolution as r


MIGRATION = Path(__file__).resolve().parents[1] / 'migrations/060_search_aliases.sql'
pytestmark = pytest.mark.db


@pytest.fixture
def cur(db_cursor):
    db_cursor.execute(MIGRATION.read_text())
    db_cursor.execute('DELETE FROM search_aliases')
    # Only this test's rows are searchable (all changes roll back).
    db_cursor.execute('UPDATE customers SET active=false')
    db_cursor.execute('UPDATE products SET active=false')
    db_cursor.execute('UPDATE suppliers SET active=false')
    return db_cursor


def add(cur, table, name, **fields):
    columns = ['name', 'active', *fields]
    cur.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join(['%s'] * len(columns))}) RETURNING id",
                (name, True, *fields.values()))
    return cur.fetchone()['id']


def resolve(cur, kind, query, **kwargs):
    return r.resolve(cur, r.ResolveRequest(kind=kind, query=query, **kwargs))


def ids(result):
    return [row['id'] for row in result['candidates']]


@pytest.fixture
def customers(cur):
    return {name: add(cur, 'customers', name) for name in
            ('Sunshine Granola', 'Sunshine Bakery', 'Setton Farms', 'Bassett Foods', 'Zephyr Foods')}


@pytest.mark.parametrize('query', ['S', 's', 'Su', 'Sun', 'Suns', 'sunsh'])
def test_customer_suggestions_start_at_first_character_and_narrow(cur, customers, query):
    result = resolve(cur, 'customer', query)
    found = set(ids(result))
    assert {customers['Sunshine Granola'], customers['Sunshine Bakery']} <= found
    assert customers['Zephyr Foods'] not in found
    if len(query) >= 2:
        assert customers['Setton Farms'] not in found
    assert result['outcome'] == 'ambiguous' and result['match'] is None


def test_customer_prefix_hits_are_choices_not_identities(cur, customers):
    result = resolve(cur, 'customer', 'zep')
    assert ids(result) == [customers['Zephyr Foods']]
    choice = result['candidates'][0]
    # Never auto-selected and never treated as a known (keyword) customer by the Assistant.
    assert result['outcome'] == 'ambiguous' and result['match'] is None and result['needs_clarification']
    assert choice['tier'] == 'prefix' and choice['score'] == r.PREFIX_SCORE < r.PLAUSIBLE
    assert resolve(cur, 'customer', 'zep', context={'action': 'order', 'group': 'office'})['match'] is None


def test_customer_multiword_prefix_and_word_prefix(cur, customers):
    assert ids(resolve(cur, 'customer', 'sunshine b')) == [customers['Sunshine Bakery']]
    assert ids(resolve(cur, 'customer', 'sun gra')) == [customers['Sunshine Granola']]
    # A later word may start with the query; names starting with it rank first,
    # and an existing whole-word keyword match still outranks both.
    farmhouse = add(cur, 'customers', 'Farmhouse Co')
    whole = add(cur, 'customers', 'Big Farm Supply')
    result = resolve(cur, 'customer', 'farm')
    assert ids(result) == [whole, farmhouse, customers['Setton Farms']]
    assert [row['tier'] for row in result['candidates']] == ['keyword', 'prefix', 'prefix']
    # Customers never match mid-word: Bassett does not start any word with "set".
    assert customers['Bassett Foods'] not in ids(resolve(cur, 'customer', 'set'))


def test_customer_exact_and_whole_word_behaviour_unchanged(cur, customers):
    exact = resolve(cur, 'customer', 'Sunshine Granola')
    assert exact['outcome'] == 'match' and exact['match']['id'] == customers['Sunshine Granola']
    assert exact['match']['tier'] == 'exact'
    keyword = resolve(cur, 'customer', 'zephyr')
    assert keyword['outcome'] == 'match' and keyword['match']['tier'] == 'keyword'
    assert resolve(cur, 'customer', 'qq')['outcome'] == 'none'


def test_product_prefix_ranks_before_mid_word_and_starts_at_one_character(cur):
    seeds = add(cur, 'products', 'Sunflower Seeds', type='ingredient')
    mix = add(cur, 'products', 'Classic Sunflower Mix', type='ingredient')
    middle = add(cur, 'products', 'Unsunk Oats', type='ingredient')
    other = add(cur, 'products', 'Raisins', type='ingredient')
    result = resolve(cur, 'product', 'sun')
    # Unchanged keyword scores; ties broken name-start, then word-start, then the rest.
    assert ids(result) == [seeds, mix, middle]
    assert {row['tier'] for row in result['candidates']} == {'keyword'}
    one = resolve(cur, 'product', 's')
    assert ids(one)[:2] == [seeds, mix] and other not in ids(one)
    assert all(row['tier'] == 'prefix' for row in one['candidates']) and one['match'] is None


def test_product_unique_keyword_match_still_auto_matches(cur):
    oats = add(cur, 'products', 'Rolled Oats', type='ingredient')
    add(cur, 'products', 'Oat Bran', type='ingredient')
    result = resolve(cur, 'product', 'rolled')
    assert result['outcome'] == 'match' and result['match']['id'] == oats


def test_supplier_prefix_typeahead(cur):
    supplier = add(cur, 'suppliers', 'Grain Millers Inc.')
    result = resolve(cur, 'supplier', 'gr')
    assert ids(result) == [supplier] and result['match'] is None


def test_prefix_text_is_literal_not_a_regex(cur, customers):
    for query in ('s.*', '(s', 's[', '\\'):
        result = resolve(cur, 'customer', query)
        assert result['outcome'] in ('none', 'ambiguous') and result['match'] is None


def test_paging_counts_prefix_candidates(cur):
    made = [add(cur, 'customers', f'Sun Test {n:02d}') for n in range(10)]
    first = resolve(cur, 'customer', 'sun', limit=8)
    assert first['candidate_count'] == 10 and first['has_more'] and len(first['candidates']) == 8
    second = resolve(cur, 'customer', 'sun', limit=8, offset=8)
    assert set(ids(first)) | set(ids(second)) == set(made)
