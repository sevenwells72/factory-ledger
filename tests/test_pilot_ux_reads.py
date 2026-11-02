"""Pilot display lookups never select or confirm physical stock for a ticket."""
from uuid import uuid4
import pytest
import main
from tests.test_actor_attribution import client, actors
from tests.test_write_tickets_part2 import items, body
from tests.test_write_tickets import headers, commit

pytestmark = pytest.mark.db


def add_lot(cur, product_id, code, balance, days):
    cur.execute("INSERT INTO lots(product_id,lot_code,created_at,received_at) VALUES(%s,%s,clock_timestamp()-(%s||' days')::interval,clock_timestamp()-(%s||' days')::interval) RETURNING id", (product_id,code,days,days))
    lid=cur.fetchone()['id']
    cur.execute("INSERT INTO transactions(type) VALUES('receive') RETURNING id")
    tid=cur.fetchone()['id']
    cur.execute('INSERT INTO transaction_lines(transaction_id,product_id,lot_id,quantity_lb) VALUES(%s,%s,%s,%s)',(tid,product_id,lid,balance))
    return lid


def test_product_lot_search_includes_zero_only_on_request_and_newest_first(client, db_cursor, items, actors):
    pid=items['ingredient']['id'];name='Pilot UX Oats '+uuid4().hex[:10]
    db_cursor.execute('UPDATE products SET name=%s WHERE id=%s',(name,pid))
    old=add_lot(db_cursor,pid,'26-10-16-'+uuid4().hex[:4].upper()+'-001',20,4)
    new=add_lot(db_cursor,pid,'26-10-18-'+uuid4().hex[:4].upper()+'-001',10,2)
    empty=add_lot(db_cursor,pid,'26-10-19-'+uuid4().hex[:4].upper()+'-001',0,1)
    db_cursor.execute('SELECT count(*) n FROM write_tickets');tickets=db_cursor.fetchone()['n']
    for role in ('floor','office','owner'):
        r=client.get('/entry/lots',params={'query':name,'product_id':pid},headers=headers(actors[role]['key']))
        assert r.status_code==200,r.text
        lots=r.json()['lots'];ids=[row['id'] for row in lots]
        assert old in ids and new in ids and empty not in ids
        assert ids.index(new)<ids.index(old)
        assert all(row['product_id']==pid and row['product_name']==name for row in lots)
        assert all('confirmed' not in row for row in lots)
    r=client.get('/entry/lots',params={'query':'oats','include_empty':'true'},headers=headers(actors['floor']['key']))
    assert r.status_code==200,r.text
    assert empty in [row['id'] for row in r.json()['lots']]
    db_cursor.execute('SELECT count(*) n FROM write_tickets');assert db_cursor.fetchone()['n']==tickets


def test_lot_choices_exclude_merged_and_inactive_and_escape_wildcards(client, db_cursor, items, actors):
    pid=items['ingredient']['id'];token='26-10-20-'+uuid4().hex[:4].upper()
    visible=add_lot(db_cursor,pid,token+'-001',20,1)
    merged=add_lot(db_cursor,pid,token+'-002',20,2)
    db_cursor.execute("UPDATE lots SET status='merged',merged_into_lot_id=%s WHERE id=%s",(visible,merged))
    r=client.get('/entry/lots',params={'query':token},headers=headers(actors['floor']['key']))
    assert r.status_code==200,r.text
    assert [row['id'] for row in r.json()['lots']]==[visible]
    r=client.get('/entry/lots',params={'query':'%'},headers=headers(actors['floor']['key']))
    assert r.status_code==200 and r.json()['lots']==[]
    db_cursor.execute('UPDATE products SET active=false WHERE id=%s',(pid,))
    r=client.get('/entry/lots',params={'product_id':pid,'include_empty':'true'},headers=headers(actors['floor']['key']))
    assert r.status_code==200 and r.json()['lots']==[]


def test_pack_batch_context_filters_classic_source_without_confirming_any_lot(client, db_cursor, items, actors):
    pid=items['batch']['id'];token=uuid4().hex[:8]
    db_cursor.execute('UPDATE products SET name=%s WHERE id=%s',('Batch Classic Granola #9 '+token,pid))
    query='Classic #9 '+token
    r=client.post('/resolve',json={'kind':'product','query':query,'context':{'action':'pack','group':'floor','product_type':'batch'}},headers=headers(actors['floor']['key']))
    assert r.status_code==200,r.text
    candidates=r.json()['candidates']
    assert candidates and all(row['type']=='batch' for row in candidates)
    assert pid in [row['id'] for row in candidates]
    prepared=client.post('/pack/prepare',json=body('pack',items),headers=headers(actors['floor']['key'])).json()
    assert not prepared['can_commit'] and all(not row['confirmed'] for row in prepared['draft']['input_plan'])
    assert commit(client,prepared,actors['floor']['key']).status_code==422


@pytest.mark.parametrize('params',[{'product_id':0},{'product_id':-1},{'query':'x'*501}])
def test_lot_choice_request_validation(client,actors,params):
    assert client.get('/entry/lots',params=params,headers=headers(actors['floor']['key'])).status_code==422


def make_product(cur, name, code, *, kind='batch', active=True, recipe=True, tier=None):
    cur.execute("INSERT INTO products(name,odoo_code,type,uom,default_batch_lb,active,kosher_tier) VALUES(%s,%s,%s,'lb',323,%s,%s) RETURNING id",
                (name, code, kind, active, tier or 'regular'))
    pid = cur.fetchone()['id']
    if recipe:
        cur.execute("INSERT INTO products(name,odoo_code,type,uom) VALUES(%s,%s,'ingredient','lb') RETURNING id", (name+' oats', code+'-I'))
        cur.execute('INSERT INTO batch_formulas(product_id,ingredient_product_id,quantity_lb) VALUES(%s,%s,323)',
                    (pid, cur.fetchone()['id']))
    return pid


def test_make_product_search_lists_recipe_batches_by_partial_name_or_code(client, db_cursor, actors):
    """PR #173 staging: Make search must find manufacturing products by part of the name or code."""
    token = uuid4().hex[:6]; code = '9' + str(int(token, 16) % 10**6).zfill(6)
    regular = make_product(db_cursor, f'Batch Classic Granola #9 {token}', code)
    kosher = make_product(db_cursor, f'Batch Classic Granola #9 {token} EXTRA-KOSHER (SS)', code + '5', tier='extra_kosher')
    no_recipe = make_product(db_cursor, f'Batch Classic Granola #9 {token} no recipe', code + '6', recipe=False)
    inactive = make_product(db_cursor, f'Batch Classic Granola #9 {token} old', code + '7', active=False)
    ingredient = make_product(db_cursor, f'Classic {token} Honey', code + '8', kind='ingredient', recipe=False)
    db_cursor.execute('SELECT count(*) n FROM write_tickets'); tickets = db_cursor.fetchone()['n']
    for key in (actors['floor']['key'], main.DASHBOARD_API_KEY):
        for query in (code, code[:3] + ' ' + token, f'classic {token}', f'CLASSIC #9 {token}', token, f'granola 9 {token}'):
            r = client.get('/entry/make-products', params={'query': query}, headers=headers(key))
            assert r.status_code == 200, r.text
            ids = [row['id'] for row in r.json()['products']]
            assert regular in ids, query
            assert not {no_recipe, inactive, ingredient} & set(ids), query
            if kosher in ids:
                assert ids.index(regular) < ids.index(kosher), query
    row = next(row for row in client.get('/entry/make-products', params={'query': code}, headers=headers(actors['floor']['key'])).json()['products'])
    assert row['id'] == regular and row['label'] == f'Batch Classic Granola #9 {token} ({code})'
    assert float(row['default_batch_lb']) == 323 and float(row['recipe_lb']) == 323 and row['type'] == 'batch'
    listed = client.get('/entry/make-products', headers=headers(actors['floor']['key'])).json()
    assert listed['products'] and all(r['type'] == 'batch' and r['recipe_lb'] for r in listed['products'])
    assert client.get('/entry/make-products', params={'query': 'zz' + token}, headers=headers(actors['floor']['key'])).json()['products'] == []
    assert client.get('/entry/make-products', params={'query': '%'}, headers=headers(actors['floor']['key'])).json()['products'] == []
    db_cursor.execute('SELECT count(*) n FROM write_tickets'); assert db_cursor.fetchone()['n'] == tickets


def test_make_product_search_needs_a_key_and_bounds_the_query(client, actors):
    assert client.get('/entry/make-products', params={'query': 'classic'}).status_code in (401, 403)
    assert client.get('/entry/make-products', params={'query': 'x' * 201}, headers=headers(actors['floor']['key'])).status_code == 422
    assert ('GET', '/entry/make-products') in main.DASHBOARD_KEY_ALLOWLIST
