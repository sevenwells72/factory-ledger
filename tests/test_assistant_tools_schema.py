"""Strict model-call schema regressions; no ledger access."""
import pytest
from jsonschema import Draft202012Validator

import assistant_tools


def resolve_schema():
    return next(tool['parameters'] for tool in assistant_tools.tools([]) if tool['name'] == 'resolve')


@pytest.mark.parametrize('overrides', [{}, {'kind': 'product'}, {'query': 'Chia Seeds'},
                                      {'kind': 'product', 'query': ''}])
def test_resolve_schema_rejects_null_or_empty_identity(overrides):
    schema = resolve_schema()
    args = dict.fromkeys(schema['properties']) | overrides
    assert not Draft202012Validator(schema).is_valid(args)


def test_resolve_schema_requires_identity_and_keeps_other_fields_nullable():
    schema = resolve_schema()
    validator = Draft202012Validator(schema)
    args = dict.fromkeys(schema['properties']) | {'kind': 'product', 'query': 'Chia Seeds'}
    assert validator.is_valid(args)
    for field in ('kind', 'query'):
        assert not validator.is_valid({key: value for key, value in args.items() if key != field})


@pytest.mark.parametrize('name', ['prepare_create_customer', 'prepare_create_order'])
def test_customer_and_order_tools_are_strict_prepare_only(name):
    tool = next(tool for tool in assistant_tools.tools([]) if tool['name'] == name)
    assert tool['strict'] is True
    assert tool['parameters']['additionalProperties'] is False
    assert 'client_source' not in tool['parameters']['properties']
    assert not any('commit' in item['name'] or 'record' in item['name'] for item in assistant_tools.tools([]))
