"""General persona coherence remains independent of product schemas."""
import json
from pathlib import Path

from adaptgym.persona import validate_persona


def profile():
    fixture = Path(__file__).resolve().parents[1] / 'examples/debug_shopping/user_persona.json'
    return json.loads(fixture.read_text())['profile']


def test_general_persona_excludes_product_fields():
    person = profile()
    assert validate_persona(person) == []
    for field, value in [('brand', 'Nike'), ('preferences', [])]:
        changed = {**person, field: value}
        assert validate_persona(changed)


def test_persona_name_and_family_must_be_coherent():
    person = profile()
    person['full_name'] = 'Someone Else'
    assert any('full_name' in error for error in validate_persona(person))
    person = profile()
    person['family']['relationship_status'] = 'married'
    assert any('requires a partner' in error for error in validate_persona(person))


def test_resident_child_must_appear_in_household():
    person = profile()
    person['family']['children'] = [
        {'first_name': 'Sam', 'age': 5, 'lives_in_household': True}
    ]
    assert any('missing from household.members' in error for error in validate_persona(person))
    person['household']['members'].append({'relationship': 'child', 'first_name': 'Sam', 'age': 5})
    assert validate_persona(person) == []
