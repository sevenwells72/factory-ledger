"""The receiving form uses the server's purchased-ingredient policy."""
import pytest
import resolution
from tests.test_write_tickets import payload  # noqa: F401

pytestmark = pytest.mark.db


@pytest.mark.parametrize('kind,expected', [('ingredient', True), ('finished', False),
                                         ('packaging', False), ('has_bom', False), ('service', False)])
def test_receive_product_resolution_returns_authoritative_lot_requirement(db_cursor, payload, kind, expected):
    if kind in ('has_bom', 'service'):
        field = 'has_bom' if kind == 'has_bom' else 'is_service'
        db_cursor.execute('UPDATE products SET ' + field + '=true WHERE id=%s', (payload['product_id'],))
    else:
        db_cursor.execute('UPDATE products SET type=%s WHERE id=%s', (kind, payload['product_id']))
    result = resolution.resolve(db_cursor, resolution.ResolveRequest(
        kind='product', query=str(payload['product_id']), context={'action': 'receive'}))
    assert result['outcome'] == 'match'
    assert result['match']['supplier_lot_required'] is expected
    assert all(candidate['supplier_lot_required'] is expected for candidate in result['candidates'])


def test_other_product_resolution_keeps_its_existing_shape(db_cursor, payload):
    result = resolution.resolve(db_cursor, resolution.ResolveRequest(
        kind='product', query=str(payload['product_id']), context={'action': 'make'}))
    assert result['outcome'] == 'match'
    assert 'supplier_lot_required' not in result['match']
    assert all('supplier_lot_required' not in candidate for candidate in result['candidates'])
