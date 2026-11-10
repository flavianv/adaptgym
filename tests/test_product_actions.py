from pathlib import Path

import pytest

from adaptgym.bundles import VerticalBundle
from adaptgym.catalog import build_catalog
from adaptgym.product_interest import ProductInterest
from adaptgym.search_requests import build_search_request
from adaptgym.session_plans import generate_plan, validate_plan


ROOT = Path(__file__).resolve().parents[1]


def test_explicit_interest_withdrawal_selection_and_search_state():
    state = ProductInterest()
    shown = ['A', 'B']
    assert state.targets == []
    state.apply('express_interest', shown, shown=shown, message_id='u1')
    assert state.targets == shown and state.selected == []
    assert build_search_request({'active_task_id': 'task-1', 'claims': [], **state.snapshot()}, 'any')
    state.apply('withdraw_interest', ['B'], shown=shown, message_id='u2')
    assert state.targets == ['A'] and state.selected == []
    state.apply('select', ['A'], shown=shown, message_id='u3')
    assert state.targets == ['A'] and state.selected == ['A']
    assert build_search_request({'active_task_id': 'task-1', 'claims': [], **state.snapshot()}, 'any') is None
    with pytest.raises(ValueError):
        state.apply('express_interest', ['unseen'], shown=shown, message_id='u4')


@pytest.mark.parametrize('vertical', ['job_search', 'debug_shopping'])
def test_new_lab_scenarios_have_user_evidenced_shortlists(vertical):
    bundle = VerticalBundle.load(ROOT / 'examples' / vertical)
    catalog = build_catalog(bundle.template, bundle.vertical)
    for case, turns in [('express_interest', 3), ('withdraw_interest', 4), ('select', 3)]:
        plan = generate_plan(bundle.vertical, catalog, case, turn_count=turns, finisher='positive')
        validate_plan(plan)
        assert not plan['exchanges'][0]['target_products']
        if case == 'express_interest':
            assert len(plan['exchanges'][1]['target_products']) == 2
            assert plan['exchanges'][1]['selected_products'] == []
        if case == 'withdraw_interest':
            assert len(plan['exchanges'][1]['target_products']) == 2
            assert len(plan['exchanges'][2]['target_products']) == 1
        if case == 'select':
            assert len(plan['exchanges'][1]['selected_products']) == 1
            assert plan['exchanges'][1]['planned_state']['active_task_id'] == 'task-1'


@pytest.mark.parametrize('vertical', ['job_search', 'debug_shopping'])
def test_mixed_website_matrix_budget_questions_and_endings(vertical):
    bundle = VerticalBundle.load(ROOT / 'examples' / vertical)
    catalog = build_catalog(bundle.template, bundle.vertical)
    for seed in range(5):
        for turns in range(5, 13):
            for ending in ('positive', 'negative'):
                plan = generate_plan(bundle.vertical, catalog, 'mixed', seed=seed,
                                     turn_count=turns, finisher=ending)
                assert len(plan['exchanges']) == turns
                assert 6 <= plan['attribute_budget']['distinct_attributes'] <= 12
                assert sum(s['assistant']['deliberate_error'] for s in plan['exchanges']) <= 1
                for step in plan['exchanges']:
                    known = {(e['assignment']['attribute'], e['scope'], e['strength'])
                             for e in step['planned_state']['entries']}
                    asked = {(q['attribute'], q['scope'], q['strength'])
                             for q in step['assistant'].get('requested_claims', [])}
                    assert not known & asked


def test_item_preferences_are_relative_and_revisable_without_selection():
    state = ProductInterest()
    state.apply('express_interest', ['A', 'B', 'C'], shown=['A', 'B', 'C'], message_id='u1')
    event = state.prefer('A', ['B', 'C'], shown=['A', 'B', 'C'], message_id='u2')
    assert event['over'] == ['B', 'C']
    assert state.snapshot()['preferences'] == [{'preferred': 'A', 'over': 'B'}, {'preferred': 'A', 'over': 'C'}]
    assert state.targets == ['A', 'B', 'C'] and state.selected == []
    state.prefer('B', ['A'], shown=['A', 'B', 'C'], message_id='u3')
    assert {'preferred': 'A', 'over': 'B'} not in state.preferences
    assert {'preferred': 'B', 'over': 'A'} in state.preferences
    before = state.snapshot()
    with pytest.raises(ValueError):
        state.prefer('C', ['B'], shown=['A', 'B', 'C'], message_id='u4')
    assert state.snapshot() == before  # B > A > C > B is rejected atomically.
    for preferred, over in [('A', ['A']), ('unseen', ['A']), ('A', [])]:
        with pytest.raises(ValueError):
            state.prefer(preferred, over, shown=['A', 'B', 'C'], message_id='u5')
    assert state.snapshot() == before


@pytest.mark.parametrize('vertical', ['job_search', 'debug_shopping'])
@pytest.mark.parametrize('expression', ['explicit', 'implicit', 'mixed'])
def test_comparison_preference_targets_and_extraction_scores(vertical, expression):
    from adaptgym.openai_session_test import score_extraction, reference_state
    from adaptgym.search_requests import expected_outcome
    bundle = VerticalBundle.load(ROOT / 'examples' / vertical)
    catalog = build_catalog(bundle.template, bundle.vertical)
    for turns in range(4, 9):
        plan = generate_plan(bundle.vertical, catalog, 'compare_products', seed=turns,
                             expression=expression, turn_count=turns, finisher='positive')
        validate_plan(plan)
        assert plan['exchanges'][1]['preferences'] == []  # Asking is not preferring.
        step = plan['exchanges'][2]
        pair = step['preferences'][0]
        assert step['target_products'] == [pair['preferred']]
        assert step['selected_products'] == []
        assert step['planned_state']['active_task_id']
        assert expected_outcome(plan, plan['exchanges'][-1])['preferences'] == [pair]
        prediction = reference_state(plan, step, bundle.vertical)
        messages = [m for s in plan['exchanges'][:2] for m in (
            {'id': s['shopper_message_id'], 'role': 'user', 'text': s['shopper_instruction']},
            {'id': s['assistant']['message_id'], 'role': 'assistant', 'text': s['assistant']['text']})]
        messages.append({'id': step['shopper_message_id'], 'role': 'user', 'text': 'I prefer the first one.'})
        score = score_extraction(prediction, step, bundle.vertical, messages, catalog)
        assert score['valid'] and score['preferences_correct'] and score['preferences_tp'] == 1
        prediction['preferences'] = []
        score = score_extraction(prediction, step, bundle.vertical, messages, catalog)
        assert score['valid'] and not score['preferences_correct'] and score['preferences_fn'] == 1
        assert not score['strict_exact_state']
