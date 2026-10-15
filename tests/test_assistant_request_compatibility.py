"""The deployed dependency lock must produce usable OpenAI function schemas."""
from pathlib import Path
import re

from jsonschema import Draft202012Validator
import pydantic
import pytest

import assistant_tools


def nodes(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from nodes(child)
    elif isinstance(value, list):
        for child in value:
            yield from nodes(child)


def test_requirements_lock_the_reviewed_openai_compatible_pydantic():
    requirements = (Path(__file__).resolve().parents[1] / 'requirements.txt').read_text()
    assert re.search(r'^pydantic==2\.11\.9$', requirements, re.M)
    assert pydantic.__version__ == '2.11.9', 'Install requirements.txt before validating the deployed request'


@pytest.mark.parametrize('shift_summary', [False, True])
def test_actual_assistant_tool_request_has_strict_valid_schemas_without_lookaround(shift_summary):
    tools = assistant_tools.tools([{'code': 'physical_count', 'label_en': 'Physical count',
                                    'label_es': 'Conteo físico'}], shift_summary=shift_summary)
    for tool in tools:
        assert tool['type'] == 'function' and tool['strict'] is True
        Draft202012Validator.check_schema(tool['parameters'])
        for node in nodes(tool['parameters']):
            pattern = node.get('pattern', '')
            assert not any(token in pattern for token in ('(?=', '(?!', '(?<=', '(?<!')), (tool['name'], pattern)
            if node.get('type') == 'object':
                assert node.get('additionalProperties') is False
                assert set(node['required']) == set(node['properties'])
        assert '$ref' not in str(tool['parameters'])
    assert not any('commit' in tool['name'] for tool in tools)
