"""Live extraction uses observed dialogue, not procedural targets or hidden labels."""
from pathlib import Path
import copy

import pytest

from adaptgym.bundles import VerticalBundle
from adaptgym.catalog import build_catalog
from adaptgym.live_chat import chat_turn
from adaptgym.openai_session_test import reference_state
from adaptgym.session_plans import generate_plan

ROOT = Path(__file__).resolve().parents[1]


class FixedClient:
    def __init__(self, value):
        self.value, self.payloads = value, []

    def json(self, instructions, payload, schema, *, purpose):
        self.payloads.append(copy.deepcopy(payload))
        return copy.deepcopy(self.value)


@pytest.mark.parametrize('domain', ['job_search', 'debug_shopping'])
def test_live_preference_update_uses_actual_context_and_preserves_previous_profile(domain):
    bundle = VerticalBundle.load(ROOT / 'examples' / domain)
    catalog = build_catalog(bundle.template, bundle.vertical)
    plan = generate_plan(bundle.vertical, catalog, 'compare_products', turn_count=4)
    first = plan['exchanges'][0]
    profile = reference_state(plan, first, bundle.vertical)
    ids = first['assistant']['product_ids'][:2]
    extractor = FixedClient(profile)
    assistant = FixedClient({'text': 'Here are two options.', 'product_ids': ids})
    result = chat_turn(bundle, catalog, [], None, first['shopper_instruction'], extractor, assistant)
    assert 'planned_state' not in extractor.payloads[0]
    assert extractor.payloads[0]['previous_prediction'] is None
    assert len(result['messages']) == 2
    assert result['profile']['preferences'] == []
    assert result['metrics']['matching_items'] >= 2
    updated = copy.deepcopy(profile)
    updated['preferences'] = [{'preferred': ids[0], 'over': ids[1]}]
    updated['target_products'] = [ids[0]]
    result2 = chat_turn(bundle, catalog, result['messages'], profile, 'I prefer the first one.',
                        FixedClient(updated), assistant)
    assert result2['metrics']['changed_attributes'] == []
    assert result2['profile']['preferences'] == updated['preferences']
    assert result2['profile']['selected_products'] == []
    assert profile['preferences'] == []  # New predictions never mutate the old input.
    broken = copy.deepcopy(updated)
    broken['preferences'].append({'preferred': ids[1], 'over': ids[0]})
    bad_result = chat_turn(bundle, catalog, result['messages'], profile, 'I prefer the first one.',
                           FixedClient(broken), assistant)
    assert not bad_result['metrics']['output_valid']
    assert bad_result['reply'] and bad_result['metrics']['validation_issue']
    assert 'user_profile' not in assistant.payloads[-1]
    with pytest.raises(ValueError, match='invalid catalog'):
        chat_turn(bundle, catalog, [], None, first['shopper_instruction'], extractor,
                  FixedClient({'text': 'Invented recommendation.', 'product_ids': ['not-in-catalog']}))


def test_replay_only_exposes_prefix_and_scores_preferences_against_hidden_targets():
    from adaptgym.live_chat import replay_turn
    bundle = VerticalBundle.load(ROOT / 'examples/job_search')
    catalog = build_catalog(bundle.template, bundle.vertical)
    plan = generate_plan(bundle.vertical, catalog, 'compare_products', turn_count=4)
    messages = [m for s in plan['exchanges'] for m in (
        {'id':s['shopper_message_id'],'role':'user','text':s['shopper_instruction']},
        {'id':s['assistant']['message_id'],'role':'assistant','text':s['assistant']['text']})]
    sample = {'plan':plan,'result':{'conversation':messages,
              'checkpoints':[{'verification':{'accepted':False}} for _ in plan['exchanges']]}}
    previous = None
    for index in range(3):
        expected = reference_state(plan,plan['exchanges'][index],bundle.vertical)
        client = FixedClient(expected)
        result = replay_turn(sample,index,previous,client)
        payload = client.payloads[0]
        assert len(payload['visible_conversation']) == 2*index+1
        assert set(payload) == {'visible_conversation','previous_prediction','vertical_schema','catalog'}
        assert payload['previous_prediction'] == previous
        assert result['score']['valid']
        assert result['metrics']['value_precision'] == result['metrics']['value_recall'] == 1
        assert result['metrics']['preferences_correct']
        assert not result['done']
        previous = result['profile']
    wrong = copy.deepcopy(previous)
    wrong['preferences'] = []
    result = replay_turn(sample,2,previous,FixedClient(wrong))
    assert not result['metrics']['preferences_correct']
    assert result['reference_profile']['preferences']
