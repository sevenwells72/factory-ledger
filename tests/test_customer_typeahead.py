"""Customer picker regressions against disposable local PostgreSQL only."""
import os

import psycopg2
import pytest
from psycopg2.extras import RealDictCursor

import resolution as r
from tests.test_resolution import alias, catalog, client


@pytest.fixture
def customer(catalog):
    def insert(name, *, active=True, id=None):
        cur = catalog['cur']
        if id is None:
            cur.execute('INSERT INTO customers(name,active) VALUES (%s,%s) RETURNING id', (name, active))
        else:
            cur.execute('INSERT INTO customers(id,name,active) VALUES (%s,%s,%s) RETURNING id', (id, name, active))
        return cur.fetchone()['id']
    return insert


def search(client, endpoint, query):
    headers = {'X-API-Key': 'a4-dashboard'}
    if endpoint == '/resolve':
        response = client.post(endpoint, headers=headers, json={
            'kind': 'customer', 'query': query, 'context': {'group': 'office', 'action': 'order'}})
        assert response.status_code == 200, response.text
        return response.json()['candidates']
    response = client.get(endpoint, headers=headers, params={'q': query})
    assert response.status_code == 200, response.text
    return response.json()['customers' if endpoint == '/dashboard/api/search' else 'results']


ENDPOINTS = ['/resolve', '/customers/search', '/dashboard/api/search']


@pytest.mark.parametrize('endpoint', ENDPOINTS)
@pytest.mark.parametrize('query', ['se', 'set', 'SeT', 'farm', '  FARM  '])
def test_setton_farms_appears_while_typing(client, customer, endpoint, query):
    customer('Setton Farms')
    assert 'Setton Farms' in [row['name'] for row in search(client, endpoint, query)]


@pytest.mark.parametrize('query', ['', 's', '  s  ', '1'])
def test_short_customer_queries_do_not_search(query):
    class NoQueries:
        def execute(self, *args):
            pytest.fail('Short customer query reached the database')
    cur = NoQueries()
    result = r.resolve(cur, r.ResolveRequest(kind='customer', query=query))
    assert result['outcome'] == 'none' and result['candidates'] == []
    assert r.customer_search(cur, query) == []


@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_one_character_returns_no_customer_choices(client, customer, endpoint):
    customer('Setton Farms')
    assert search(client, endpoint, 's') == []


@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_name_prefix_then_word_prefix_then_existing_fuzzy(client, customer, endpoint):
    customer('Setton Farms')
    customer('Alpha Setton Valley')
    customer('Zetton Farmz')
    names = [row['name'] for row in search(client, endpoint, 'setton')]
    assert names.index('Setton Farms') < names.index('Alpha Setton Valley') < names.index('Zetton Farmz')
    customer('Setton')
    # Preserve the resolver's exact-identity precedence, including aliases.
    assert search(client, endpoint, 'setton')[0]['name'] == 'Setton'


@pytest.mark.parametrize('endpoint', ENDPOINTS)
@pytest.mark.parametrize('query', ['8675', '867530'])
def test_internal_customer_ids_never_match(client, customer, endpoint, query):
    customer('Unrelated Customer', id=867530)
    assert search(client, endpoint, query) == []
    if endpoint == '/resolve':
        response = client.post('/resolve', headers={'X-API-Key': 'a4-dashboard'},
                               json={'kind': 'customer', 'query': query}).json()
        assert response['outcome'] == 'none' and response['match'] is None
        assert response['confidence'] == 'none'


@pytest.mark.parametrize('endpoint', ENDPOINTS)
@pytest.mark.parametrize('source', ['legacy', 'search'])
def test_real_numeric_customer_codes_match_their_alias_not_an_internal_id(
        client, catalog, customer, endpoint, source):
    customer('Unrelated Internal ID', id=867530)
    cid = customer('Real Code Customer')
    if source == 'legacy':
        catalog['cur'].execute('INSERT INTO customer_aliases(customer_id,alias) VALUES (%s,%s)',
                               (cid, '867530'))
    else:
        alias(catalog, kind='customer', text='867530', expansion=None, customer_id=cid)
    assert [row['name'] for row in search(client, endpoint, '867530')] == ['Real Code Customer']
    result = r.resolve(catalog['cur'], r.ResolveRequest(kind='customer', query='867530'))
    assert result['match']['id'] == cid and result['match']['tier'] == 'alias'
    assert result['confidence'] == 'high'


@pytest.mark.parametrize('query', ['Sunshine Granolaa', 'Sunshine Granola, Inc.', 'Setton Farmss'])
def test_customer_near_misses_remain_low_confidence_trigrams(catalog, customer, query):
    cid = customer('Setton Farms' if query.startswith('Setton') else 'Sunshine Granola')
    result = r.resolve(catalog['cur'], r.ResolveRequest(kind='customer', query=query))
    choice = next(row for row in result['candidates'] if row['id'] == cid)
    assert choice['tier'] == 'trigram' and choice['score'] <= 0.79
    assert result['confidence'] == 'low'


def test_customer_reordered_literal_words_remain_keywords(catalog, customer):
    cid = customer('Setton Farms')
    result = r.resolve(catalog['cur'], r.ResolveRequest(kind='customer', query='farms setton'))
    assert result['match']['id'] == cid
    assert result['match']['tier'] == 'keyword' and result['match']['score'] == 0.8
    assert result['confidence'] == 'medium'


@pytest.mark.parametrize('endpoint', ENDPOINTS)
@pytest.mark.parametrize('source', ['legacy', 'search'])
def test_existing_customer_aliases_survive(client, catalog, customer, endpoint, source):
    cid = customer('Unrelated Canonical Customer')
    if source == 'legacy':
        catalog['cur'].execute('INSERT INTO customer_aliases(customer_id,alias) VALUES (%s,%s)', (cid, 'SF-CODE'))
    else:
        alias(catalog, kind='customer', text='SF-CODE', expansion=None, customer_id=cid)
    assert search(client, endpoint, 'sf-code')[0]['name'] == 'Unrelated Canonical Customer'


@pytest.mark.parametrize('endpoint', ENDPOINTS)
def test_word_boundaries_inactive_and_literal_search_text(client, customer, endpoint):
    customer('Acme-Farmfoods')
    customer('Setton Farms', active=False)
    customer('Asset Holdings')
    assert 'Acme-Farmfoods' in [row['name'] for row in search(client, endpoint, 'farm')]
    assert 'Setton Farms' not in [row['name'] for row in search(client, endpoint, 'set')]
    assert ('Asset Holdings' in [row['name'] for row in search(client, endpoint, 'set')]) == (endpoint != '/resolve')
    for query in ('%_', '[s', r'\s'):
        assert search(client, endpoint, query) == []


@pytest.mark.parametrize('endpoint,limit', [('/resolve', 8), ('/customers/search', 10), ('/dashboard/api/search', 10)])
def test_customer_choices_are_bounded_and_ranked_before_limit(client, customer, endpoint, limit):
    for i in range(15):
        customer(f'Alpha Setchoice {i:02}')
    customer('Setchoice First')
    rows = search(client, endpoint, 'setchoice')
    assert len(rows) == limit
    assert rows[0]['name'] == 'Setchoice First'


def test_customer_prefix_paging_keeps_global_ambiguity(catalog, customer):
    for i in range(12):
        customer(f'Setchoice {i:02}')
    result = r.resolve(catalog['cur'], r.ResolveRequest(kind='customer', query='setchoice', limit=4, offset=4))
    assert result['outcome'] == 'ambiguous' and result['match'] is None
    assert result['candidate_count'] == 12 and result['has_more']
    assert [row['name'] for row in result['candidates']] == [f'Setchoice {i:02}' for i in range(4, 8)]


def test_customer_search_runs_in_database_enforced_read_only_transaction():
    with psycopg2.connect(os.environ['TEST_DATABASE_URL']) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute('SET TRANSACTION READ ONLY')
            r.customer_search(cur, 'set')
            r.customer_search(cur, 'farm')
            r.resolve(cur, r.ResolveRequest(kind='customer', query='set'))
            cur.execute('SHOW transaction_read_only')
            assert cur.fetchone()['transaction_read_only'] == 'on'


@pytest.mark.parametrize('query', ['se', 'set', 'farm', 'Setton', 'Farms'])
def test_customer_prefixes_keep_known_customer_keyword_tier(catalog, customer, query):
    cid = customer('Setton Farms')
    result = r.resolve(catalog['cur'], r.ResolveRequest(kind='customer', query=query))
    choice = next(row for row in result['candidates'] if row['id'] == cid)
    assert choice['tier'] == 'keyword'
    assert result['confidence'] == 'medium'

@pytest.mark.parametrize('endpoint', ['/customers/search', '/dashboard/api/search'])
@pytest.mark.parametrize('query', ['sf-co', 'f-code'])
def test_legacy_partial_alias_search_survives(client, catalog, customer, endpoint, query):
    cid = customer('Unrelated Canonical Customer')
    catalog['cur'].execute('INSERT INTO customer_aliases(customer_id,alias) VALUES (%s,%s)', (cid, 'SF-CODE'))
    assert search(client, endpoint, query)[0]['name'] == 'Unrelated Canonical Customer'


def test_legacy_alias_fallback_is_ranked_below_name_prefixes(client, catalog, customer):
    cid = customer('Alpha Canonical Customer')
    customer('Set Name Prefix')
    customer('Zebra Set Word Prefix')
    catalog['cur'].execute('INSERT INTO customer_aliases(customer_id,alias) VALUES (%s,%s)', (cid, 'set-alias'))
    names = [row['name'] for row in search(client, '/customers/search', 'set')]
    assert names.index('Set Name Prefix') < names.index('Zebra Set Word Prefix') < names.index('Alpha Canonical Customer')


@pytest.mark.parametrize('endpoint', ['/customers/search', '/dashboard/api/search'])
@pytest.mark.parametrize('query,name', [('wells', 'SevenWells Granola Co'), ('ton', 'Setton Farms')])
def test_legacy_middle_of_name_search_survives(client, customer, endpoint, query, name):
    customer(name)
    assert name in [row['name'] for row in search(client, endpoint, query)]

@pytest.mark.parametrize('endpoint', ['/customers/search', '/dashboard/api/search'])
def test_legacy_middle_of_name_hits_rank_after_exact_name_and_word_prefixes(
        client, customer, endpoint):
    customer('Ton Foods')
    customer('Zebra Ton Foods')
    customer('Setton Farms')
    names = [row['name'] for row in search(client, endpoint, 'ton')]
    assert names.index('Ton Foods') < names.index('Zebra Ton Foods') < names.index('Setton Farms')
    customer('Ton')
    # Exact identities intentionally take precedence over partial suggestions.
    assert [row['name'] for row in search(client, endpoint, 'ton')] == ['Ton']


def test_legacy_middle_of_name_fallback_stays_below_keyword_confidence(catalog, customer):
    cid = customer('SevenWells Granola Co')
    req = r.ResolveRequest(kind='customer', query='wells')
    rows = r._parties(catalog['cur'], req, [], [], legacy_alias_substrings=True)
    choice = next(row for row in rows if row['id'] == cid)
    assert choice['score'] == 0.79 and choice['tier'] == 'trigram'
