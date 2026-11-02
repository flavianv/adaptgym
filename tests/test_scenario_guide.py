"""The in-app guide must cover the actual registry and both data-only verticals."""
import json
from pathlib import Path

import pytest

from adaptgym.scenario_guide import load_scenario_guide
from adaptgym.session_plans import CASE_MODES, PREFERENCE_CASES

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('vertical', ['debug_shopping', 'job_search'])
def test_every_registered_scenario_has_an_explanation_and_example(vertical):
    guide = load_scenario_guide(ROOT / 'examples' / vertical)
    assert {entry['id'] for entry in guide} == set(CASE_MODES)
    assert {entry['id'] for entry in guide if entry['preference_evidence']} == set(PREFERENCE_CASES)
    for entry in guide:
        assert entry['description'] and entry['common_mistake']
        assert len(entry['expected_behavior']) >= 2
        assert entry['example']['expected_result']
        assert any(turn['role'] == 'user' for turn in entry['example']['conversation'])
        if entry['preference_evidence']:
            assert entry['example']['implicit_variant']['user_reply']


def test_ambiguous_example_does_not_fabricate_a_finishing_user_message():
    guide = {entry['id']: entry for entry in load_scenario_guide(ROOT / 'examples/job_search')}
    example = guide['finish_task_ambiguous']['example']
    assert example['conversation'][-1]['role'] == 'environment'
    assert len([turn for turn in example['conversation'] if turn['role'] == 'user']) == 1
    assert not guide['completion_continue']['preference_evidence']
    assert guide['clarification_question']['group'] == 'context_dependent'
    assert not guide['clarification_question']['preference_evidence']


def test_another_vertical_can_omit_examples_without_breaking_the_guide(tmp_path):
    guide = load_scenario_guide(tmp_path)
    assert len(guide) == len(CASE_MODES)
    assert all(entry['example'] is None for entry in guide)
    (tmp_path / 'scenario_examples.json').write_text(json.dumps({'unknown_scenario': {}}))
    with pytest.raises(ValueError, match='unknown scenarios'):
        load_scenario_guide(tmp_path)
