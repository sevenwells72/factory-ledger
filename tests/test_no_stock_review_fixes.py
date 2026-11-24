"""Owner-directed regressions from Claude Code review 7026524246."""
from pathlib import Path
import pytest
import main
from tests.test_actor_attribution import client, actors
from tests.test_write_tickets_part2 import items, prepare
from tests.test_write_tickets import headers, commit, error
from tests.test_exceptions_enforcement import adjust, found, resolve
from tests.test_adjust_destinations import order
from tests.test_no_stock_by_hand import rules, received, short

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('enabled',[False,True])
def test_sales_order_note_is_exactly_once(client,db_cursor,items,actors,order,enabled):
    db_cursor.execute((ROOT/'migrations/096_adjust_destinations.sql').read_text())
    db_cursor.execute("DELETE FROM migration_markers WHERE name='098_no_stock_by_hand'")
    if enabled:db_cursor.execute((ROOT/'migrations/098_no_stock_by_hand.sql').read_text())
    key=actors['floor']['key']
    posted=commit(client,prepare(client,'adjust',adjust(items,-20,'shipped_unrecorded',sales_order_id=order['id'],note='Friday truck'),key),key)
    assert posted.status_code==200,posted.text
    db_cursor.execute('SELECT notes FROM transactions WHERE id=%s',(posted.json()['transaction_id'],))
    assert db_cursor.fetchone()['notes']==f"Adjustment: -20 lb — {order['order_number']} ({order['customer_name']}) — Friday truck"


def test_held_found_retired_reason_can_be_approved_with_marker(client,db_cursor,items,actors):
    db_cursor.execute("DELETE FROM migration_markers WHERE name IN ('096_adjust_destinations','098_no_stock_by_hand')")
    key=actors['floor']['key']
    held=commit(client,prepare(client,'found',found(items,600,'missing_receipt'),key),key)
    assert held.status_code==202,held.text
    db_cursor.execute((ROOT/'migrations/096_adjust_destinations.sql').read_text())
    db_cursor.execute((ROOT/'migrations/098_no_stock_by_hand.sql').read_text())
    db_cursor.execute("UPDATE correction_reasons SET active=false WHERE code='missing_receipt'")
    error(client.post('/inventory/found/prepare',json=found(items,5,'missing_receipt'),headers=headers(key)),422,'REASON_CODE_INVALID')
    posted=client.post(f"/exceptions/{held.json()['exception_id']}/approve",json={},headers=headers(actors['owner']['key']))
    assert posted.status_code==200,posted.text
    assert posted.json()['reason_code']=='missing_receipt'


def test_pair_field_is_refused_when_marker_off(client,db_cursor,items,actors):
    db_cursor.execute("DELETE FROM migration_markers WHERE name='098_no_stock_by_hand'")
    response=client.post('/adjust/prepare',json=adjust(items,1,'physical_count',paired_lot_id=items['batch']['lot_id']),headers=headers(actors['floor']['key']))
    error(response,422,'PAIRED_MOVE_UNAVAILABLE')
    assert 'not enabled' in response.json()['detail']['message']

@pytest.mark.parametrize('reason,note,code',[('unknown','unknown origin','REASON_CODE_INVALID'),('count_unknown','count','REASON_CODE_INVALID'),('missing_production',None,'NO_STOCK_BY_HAND'),('data_entry_error',None,'NOTE_REQUIRED'),('wrong_lot',None,'NOTE_REQUIRED')])
@pytest.mark.parametrize('route',['ticket','direct'])
def test_ingredient_addition_source_refusals(client,db_cursor,items,actors,rules,reason,note,code,route):
    key=actors['owner']['key']
    if route=='ticket':path='/adjust/prepare';data=adjust(items,1,reason,note=note)
    else:path='/adjust';data=dict(mode='commit',product_name=items['ingredient']['name'],lot_code=items['ingredient']['lot_code'],adjustment_lb=1,reason=reason,note=note)
    before=main.lot_on_hand(db_cursor,items['ingredient']['lot_id'])
    error(client.post(path,json=data,headers=headers(key)),422,code)
    assert main.lot_on_hand(db_cursor,items['ingredient']['lot_id'])==before


def test_voided_under_pack_count_releases_allowance(client,db_cursor,items,actors,rules):
    received(client,db_cursor,items,actors,55)
    exc=short(client,db_cursor,items,actors,30)
    db_cursor.execute('SELECT transaction_id FROM exceptions WHERE id=%s',(exc,));made=db_cursor.fetchone()['transaction_id']
    counted=resolve(client,exc,actors['floor']['key'],resolution_kind='counted',counted_lb=0,note='empty')
    assert counted.status_code==200,counted.text
    # The resolution receipt names the count transaction separately from its shortage.
    db_cursor.execute('SELECT transaction_id FROM under_pack_counts WHERE lot_id=%s',(items['ingredient']['lot_id'],));count_id=db_cursor.fetchone()['transaction_id']
    for tid in [count_id,made]:
        response=client.post('/void/'+str(tid),json={'reason':'data_entry_error'},headers=headers(actors['owner']['key']))
        assert response.status_code==200,response.text
    assert main.lot_on_hand(db_cursor,items['ingredient']['lot_id'])==55
    exc=short(client,db_cursor,items,actors,30)
    posted=resolve(client,exc,actors['floor']['key'],resolution_kind='counted',counted_lb=0,note='empty after corrected make')
    assert posted.status_code==200,posted.text
    db_cursor.execute('SELECT count(*) AS n FROM under_pack_counts WHERE lot_id=%s',(items['ingredient']['lot_id'],));assert db_cursor.fetchone()['n']==2


from tests.test_adjust_destinations import (
    migrated,
    test_a_removal_must_say_where_the_stock_went as test_096_removal_reasons_with_098,
    test_a_destination_is_one_pick_and_lands_on_the_receipt as test_096_destinations_with_098,
    test_shipped_but_not_recorded_must_name_the_sales_order as test_096_order_with_098,
    test_over_500_lb_posts_at_once_without_a_photo_and_queues_the_owner_email as test_096_large_removal_with_098,
    test_a_held_found_keeps_its_reason_if_that_reason_is_retired_later as test_096_held_found_with_098,
)

@pytest.fixture(autouse=True)
def enable_098_for_existing_096_contracts(request,db_cursor):
    if request.node.name.startswith('test_096_'):
        db_cursor.execute((ROOT/'migrations/098_no_stock_by_hand.sql').read_text())


def test_held_reason_does_not_bypass_source_applicability(db_cursor,rules):
    import exceptions_enforcement as policy
    with pytest.raises(main.HTTPException) as refused:
        policy.resolve_reason(db_cursor,'found','wrong_lot',note='wrong tag',delta_lb=1,held_reason='wrong_lot')
    assert refused.value.detail['error_code']=='REASON_CODE_INVALID'
