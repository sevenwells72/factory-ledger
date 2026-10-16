"""Opening-stock grouping uses recorded origin markers, never ordinary found stock."""
import pytest
import exceptions_enforcement as ex

@pytest.mark.parametrize('note,expected', [
    ('STAGING OPENING STOCK copied from production 2026-10-14 (prod lot 1)', True),
    ('OPENING BALANCE 2026-10-14 | v3 approved L1', True),
    ('Found on the floor during a count', False), (None, False),
])
def test_opening_stock_marker(note, expected):
    row = dict(id=1, kind='UNIDENTIFIED_LOT', status='open', due_at=None,
               lot_entry_source='found_inventory', lot_entry_source_notes=note)
    assert ex._view(None, row)['is_opening_stock'] is expected

@pytest.mark.parametrize('role,expected', [('floor', []), ('office', []), ('owner', ['acknowledged'])])
def test_only_owner_can_acknowledge_assistant_offline(role, expected):
    assert ex.allowed_actions({'kind': 'ASSISTANT_OFFLINE', 'status': 'open'}, {'id': 1, 'role': role, 'key_kind': 'actor'}) == expected

@pytest.mark.parametrize('kind', ['PRE_MAKE_ADJUST', 'LARGE_CORRECTION', 'LATE_ENTRY'])
def test_new_exception_on_opening_lot_stays_visible(kind):
    row = dict(id=2, kind=kind, status='open', due_at=None,
               lot_entry_source_notes='STAGING OPENING STOCK copied from production 2026-10-14')
    assert ex._view(None, row)['is_opening_stock'] is False

def test_shortage_warning_rounds_display_but_preserves_stock_evidence():
    shortage = dict(product_name='Oats', lot_code='LOT-88.80000000000001', lot_id=9,
                    on_hand_lb=88.80000000000001, needed_lb=89.12345, short_lb=0.32345)
    warning = ex.shortage_warnings({'shortages': [shortage]})[0]
    assert 'has 88.8 lb; posting 89.12 lb' in warning['message']
    assert 'short 0.32 lb' in warning['message']
    assert 'tiene 88.8 lb' in warning['message_es']
    assert 'faltante de 0.32 lb' in warning['message_es']
    assert shortage['lot_code'] in warning['message_es']
    assert warning['refs']['short_lb'] == 0.32345
    assert warning['refs']['on_hand_lb'] == 88.80000000000001
