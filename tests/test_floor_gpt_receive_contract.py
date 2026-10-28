"""The deployable Floor GPT can preview and commit an explicit no-label receipt."""
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest
import yaml

import lot_confirmation
from tests.test_actor_attribution import client  # noqa: F401
from tests.test_write_tickets import headers, payload  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
FLOOR_SCHEMA = ROOT / 'gpt-configs/schemas/openapi-floor.yaml'
RECEIVE_RULE = ('Ingredient label lot → supplier_lot_code. Operator says no lot on label '
                'or lot unreadable ("Sin lote en la etiqueta"/"lote ilegible") → '
                'no_supplier_lot: true, supplier_lot_code "". Never UNKNOWN/N/A; never assume.')


@pytest.fixture
def floor_schema():
    return yaml.safe_load(FLOOR_SCHEMA.read_text())


def test_floor_receive_schema_exposes_the_explicit_no_label_path(floor_schema):
    schema = floor_schema['components']['schemas']['ReceiveRequest']
    assert floor_schema['info']['version'] == '4.3.0'
    assert 'supplier_lot_code' not in schema['required']
    assert 'shipper_name' in schema['required']
    assert schema['properties']['no_supplier_lot']['type'] == 'boolean'
    assert schema['properties']['no_supplier_lot']['default'] is False
    assert schema['properties']['supplier_lot_code']['description'] == (
        'Lot printed on the supplier label. Required for purchased ingredients unless '
        'no_supplier_lot is true. Never send UNKNOWN or N/A.')
    assert schema['properties']['no_supplier_lot']['description'] == (
        'True only when the operator says the label has no lot or the lot is unreadable '
        '("No lot on label" / "Sin lote en la etiqueta"). The lot is recorded unidentified '
        'with the 7-business-day follow-up.')
    operation = floor_schema['paths']['/receive']['post']
    assert operation['requestBody']['content']['application/json']['schema']['$ref'] == (
        '#/components/schemas/ReceiveRequest')


def test_floor_schema_keeps_22_operations_and_receive_description_limits(floor_schema):
    methods = {'get', 'post', 'put', 'patch', 'delete', 'options', 'head', 'trace'}
    operations = [operation for path in floor_schema['paths'].values()
                  for method, operation in path.items() if method.lower() in methods]
    assert len(operations) == 22
    assert len({operation['operationId'] for operation in operations}) == 22

    def descriptions(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'description':
                    yield child
                else:
                    yield from descriptions(child)
        elif isinstance(value, list):
            for child in value:
                yield from descriptions(child)

    receive = floor_schema['components']['schemas']['ReceiveRequest']
    assert all(len(value) <= 300 for value in descriptions(receive))
    assert len(floor_schema['paths']['/receive']['post'].get('description', '')) <= 300


def test_compiled_floor_instructions_match_sources_and_fit_editor():
    shared = (ROOT / 'gpt-configs/sources/shared-rules.md').read_text().rstrip()
    floor = (ROOT / 'gpt-configs/sources/floor-specific.md').read_text().rstrip()
    compiled = (ROOT / 'gpt-configs/dist/GPT_FLOOR_INSTRUCTIONS.md').read_text()
    body = compiled.split('\n\n', 1)[1]
    assert body == shared + '\n\n' + floor + '\n'
    assert len(body) < 8000
    assert len(compiled) < 8000
    receive_section = floor.split('## RECEIVE\n', 1)[1].split('\n## ', 1)[0]
    assert RECEIVE_RULE in receive_section
    assert 'CNS lot as first large heading' in receive_section
    assert 'Write this on the pallet tag' in receive_section
    assert 'Escribe esto en la etiqueta del pallet' in receive_section
    assert 'Unreadable → "UNKNOWN"' not in receive_section


@pytest.mark.db
@pytest.mark.parametrize('include_blank_code', [False, True])
def test_floor_gpt_no_label_preview_and_commit_with_deployable_schema(
        client, db_cursor, payload, floor_schema, include_blank_code):
    db_cursor.execute('SELECT name FROM products WHERE id=%s', (payload['product_id'],))
    product_name = db_cursor.fetchone()['name']
    db_cursor.execute('SELECT name FROM suppliers WHERE id=%s', (payload['supplier_id'],))
    supplier_name = db_cursor.fetchone()['name']
    body = {key: payload[key] for key in (
        'cases', 'case_size_lb', 'bol_reference', 'lot_code', 'occurred_at')}
    body.update(product_name=product_name, shipper_name=supplier_name, no_supplier_lot=True)
    if include_blank_code:
        body['supplier_lot_code'] = ''
    validator = Draft202012Validator(floor_schema['components']['schemas']['ReceiveRequest'])

    preview_body = body | {'mode': 'preview'}
    validator.validate(preview_body)
    preview = client.post('/receive', json=preview_body, headers=headers())
    assert preview.status_code == 200, preview.text
    assert preview.json()['identity_status'] == 'unidentified'
    db_cursor.execute('SELECT count(*) AS n FROM lots WHERE product_id=%s', (payload['product_id'],))
    assert db_cursor.fetchone()['n'] == 0

    commit_body = body | {'mode': 'commit'}
    validator.validate(commit_body)
    committed = client.post('/receive', json=commit_body, headers=headers())
    assert committed.status_code == 200, committed.text
    result = committed.json()
    assert result['identity_status'] == 'unidentified'
    assert result['supplier_id'] == payload['supplier_id']
    assert result['exception_id']
    db_cursor.execute('SELECT supplier_id,identity_status,identify_by FROM lots WHERE id=%s',
                      (result['lot_id'],))
    lot = db_cursor.fetchone()
    assert lot['supplier_id'] == payload['supplier_id']
    assert lot['identity_status'] == 'unidentified'
    db_cursor.execute('SELECT created_at FROM transactions WHERE id=%s', (result['transaction_id'],))
    deadline = lot_confirmation.identification_deadline(db_cursor.fetchone()['created_at'])
    assert lot['identify_by'] == deadline.date()
    db_cursor.execute('SELECT kind,due_at FROM exceptions WHERE id=%s', (result['exception_id'],))
    exception = db_cursor.fetchone()
    assert exception['kind'] == 'UNIDENTIFIED_LOT'
    assert exception['due_at'] == deadline


def _all_descriptions(value):
    if isinstance(value, dict):
        for key, child in value.items():
            if key in ('description', 'summary') and isinstance(child, str):
                yield child
            else:
                yield from _all_descriptions(child)
    elif isinstance(value, list):
        for child in value:
            yield from _all_descriptions(child)


def test_every_floor_schema_description_fits_the_gpt_editor(floor_schema):
    # The ChatGPT Actions editor rejects any description over 300 chars.
    assert max(len(text) for text in _all_descriptions(floor_schema)) <= 300


def test_every_floor_operation_is_served_by_the_api(floor_schema):
    import main
    served = {(method, route.path) for route in main.app.routes
              for method in getattr(route, 'methods', ()) or ()}
    ops = [(method.upper(), path) for path, item in floor_schema['paths'].items()
           for method in item if method in ('get', 'post', 'put', 'patch', 'delete')]
    assert len(ops) == 22
    assert [op for op in ops if op not in served] == []


def test_floor_packing_slip_rule_relays_the_signed_link_and_never_a_key(floor_schema):
    floor = (ROOT / 'gpt-configs/sources/floor-specific.md').read_text()
    section = floor.split('## PACKING SLIP\n', 1)[1].split('\n## ', 1)[0]
    assert 'packing_slip_url' in section and '10 min' in section
    assert 'Ask the office for the packing slip' in section
    assert 'Pide el packing slip a la oficina' in section
    assert 'key=' not in floor and 'railway.app' not in section
    commit = floor_schema['paths']['/sales/orders/{order_id}/ship/commit']['post']
    assert 'packing_slip_url' in commit['responses']['200']['description']
