"""Fixture, scenario coverage, programmed assistant, and API adapter regressions."""
import copy
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from adaptgym.catalog import build_catalog, classify_products, matching_ids, validate_catalog
from adaptgym.persona import GENERAL_PERSONA_SCHEMA, validate_persona
from adaptgym.openai_session_test import ResponsesClient, render_and_test, score_extraction
from adaptgym.session_plans import CASE_MIN_TURNS, CASE_MODES, generate_plan, validate_plan
from adaptgym.verticals import VerticalSchema
from adaptgym.search_requests import build_search_request, describe_intent, expected_outcome

ROOT = Path(__file__).resolve().parents[1]
DEBUG = ROOT / 'examples/debug_shopping'


def fixtures():
    vertical = VerticalSchema.load(DEBUG / 'domain_schema.json')
    template = json.loads((DEBUG / 'catalog_template.json').read_text())
    return vertical, template, build_catalog(template, vertical)


def test_debug_persona_and_exactly_30_template_products():
    vertical, template, catalog = fixtures()
    assert len(catalog['products']) == 30
    assert len({p['id'] for p in catalog['products']}) == 30
    assert len({p['name'] for p in catalog['products']}) == 30
    assert all(g['count'] == 1 for g in template['groups'])
    assert catalog == json.loads((DEBUG / 'catalog_30.json').read_text())
    profile = json.loads((DEBUG / 'user_persona.json').read_text())['profile']
    assert validate_persona(profile, schema=GENERAL_PERSONA_SCHEMA) == []
    assert 'preferences' not in profile


def test_template_overrides_keep_fixed_text_and_change_requested_values_only():
    vertical, template, before = fixtures()
    after = build_catalog(template, vertical, {'nike_pegasus_42': {'price': 120, 'size': 41, 'colour': 'blue', 'gender': 'women'}})
    assert after['products'][0]['attributes']['price'] == 120
    assert [p['name'] for p in before['products']] == [p['name'] for p in after['products']]
    assert before['products'][1:] == after['products'][1:]
    assert 'size 41 EU; women; blue; 120 EUR' in after['products'][0]['description']
    assert template['groups'][0]['attributes']['price'] == 140
    for key in set(vertical.attributes) - set(template['variable_attributes']):
        assert before['products'][0]['attributes'][key] == after['products'][0]['attributes'][key]
    assert before['products'][0]['provenance'] == after['products'][0]['provenance']



@pytest.mark.parametrize('overrides', [{'missing': {'price': 10}}, {'nike_pegasus_42': {'price': 2500}}, {'nike_pegasus_42': {'invented': 'x'}}, {'nike_pegasus_42': {'brand': 'Adidas'}}])
def test_bad_catalog_controls_are_rejected(overrides):
    vertical, template, _ = fixtures()
    with pytest.raises(ValueError):
        build_catalog(template, vertical, overrides)


@pytest.mark.parametrize('case', CASE_MODES)
def test_all_scenarios_are_seeded_valid_and_programmed(case):
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, case, seed=7)
    assert plan == generate_plan(vertical, catalog, case, seed=7)
    assert plan['mode'] == CASE_MODES[case]
    assert 1 <= len(plan['exchanges']) <= 5
    validate_plan(plan)
    for step in plan['exchanges']:
        assert step['assistant']['text']
        if step['assistant']['kind'] == 'recommend' and not step['assistant']['deliberate_error']:
            assert len(step['assistant']['product_ids']) == min(5, len(step['assistant']['matching_product_ids']))


@pytest.mark.parametrize('seed', [0, 13, 41])
def test_catalog_witness_conditions_across_randomizations(seed):
    vertical, _, catalog = fixtures()
    for case in CASE_MODES:
        validate_plan(generate_plan(vertical, catalog, case, seed=seed))


def test_no_match_relaxation_changes_from_zero_to_five_valid_recommendations():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'no_match_relax')
    first, second = plan['exchanges']
    assert first['assistant']['kind'] == 'no_match'
    assert first['assistant']['matching_product_ids'] == []
    assert len(second['assistant']['product_ids']) == 5
    assert second['reaction_to'] == 'a1'


def test_deliberate_assistant_error_is_marked_and_corrected_not_accepted():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'hard_constraint_violated')
    first, second = plan['exchanges']
    assert first['assistant']['violating_product_ids']
    assert first['assistant']['deliberate_error']
    assert not second['assistant']['violating_product_ids']
    assert first['planned_state']['entries'][-1]['assignment'] == second['planned_state']['entries'][-1]['assignment']


def test_second_product_reference_is_bound_to_actual_shortlist():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'select_product')
    assert plan['exchanges'][1]['target_product'] == plan['exchanges'][0]['assistant']['product_ids'][1]
    assert len(plan['exchanges'][1]['planned_state']['entries']) == len(plan['exchanges'][0]['planned_state']['entries'])


def test_unconfirmed_suggestion_creates_no_preference():
    vertical, _, catalog = fixtures()
    for case in ('suggestion_confirm', 'suggestion_reject'):
        plan = generate_plan(vertical, catalog, case)
        assert not any(e['scope'] == 'durable' for e in plan['exchanges'][0]['planned_state']['entries'])
        durable = [e for e in plan['exchanges'][1]['planned_state']['entries'] if e['scope'] == 'durable']
        assert bool(durable) == (case == 'suggestion_confirm')


def prediction_for(step, plan=None):
    result = {'active_task_id': step['planned_state']['active_task_id'],
            'target_product': step['target_product'],
            'claims': [{'attribute': e['assignment']['attribute'], 'operator': e['assignment']['operator'],
                        'value': e['assignment']['value'], 'scope': e['scope'], 'task_id': e['task_id'],
                        'strength': e['strength'], 'evidence': e['evidence']} for e in step['planned_state']['entries']]}

    result['search_request'] = build_search_request(result, 'footwear')
    result.update(expected_outcome(plan or {'exchanges': [step]}, step))
    result['intent'] = describe_intent(result, 'footwear')
    return result

class FakeShopper:
    def __init__(self):
        self.payloads = []

    def json(self, instructions, payload, schema, *, purpose):
        self.payloads.append(payload)
        assert 'planned_state' not in payload and 'expected_events' not in payload
        return {'text': 'A simulated shopper message.'}


class FakeExtractor:
    def __init__(self, plan):
        self.plan, self.index, self.payloads = plan, 0, []

    def json(self, instructions, payload, schema, *, purpose):
        self.payloads.append(copy.deepcopy(payload))
        assert set(payload) == {'vertical_schema', 'catalog', 'visible_conversation', 'previous_prediction'}
        result = prediction_for(self.plan['exchanges'][self.index], self.plan)
        self.index += 1
        return result


def test_runner_uses_fixed_assistant_and_public_prefixes_only():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'clarification_question')
    persona = json.loads((DEBUG / 'user_persona.json').read_text())
    shopper, extractor = FakeShopper(), FakeExtractor(plan)
    result = render_and_test(plan, persona, shopper, extractor)
    assert result['exact_checkpoints'] == 2
    assert result['conversation'][1]['text'] == plan['exchanges'][0]['assistant']['text']
    assert len(extractor.payloads[0]['visible_conversation']) == 1
    assert len(extractor.payloads[1]['visible_conversation']) == 3
    assert extractor.payloads[1]['previous_prediction'] == prediction_for(plan['exchanges'][0])
    assert result['label_status'] == 'planned_targets_not_independently_verified'


def test_invalid_prediction_never_gets_no_preference_credit():
    vertical, _, catalog = fixtures()
    step = generate_plan(vertical, catalog, 'finish_task')['exchanges'][-1]
    score = score_extraction({}, step, vertical, [], catalog)
    assert not score['valid'] and not score['exact_state']


def test_api_adapter_request_format_budget_and_no_secret_logging():
    import urllib.request
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self):
            return json.dumps({'status': 'completed', 'id': 'test-response', 'usage': {'input_tokens': 5},
                'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': '{"text":"Hello"}'}]}]}).encode()
    captured = []
    def send(request, **kwargs):
        captured.append(json.loads(request.data))
        return Response()
    from adaptgym.openai_session_test import SHOPPER_SCHEMA
    client = ResponsesClient('test-model', api_key='test-private-key', max_calls=1)
    with patch.object(urllib.request, 'urlopen', send):
        assert client.json('instructions', {}, SHOPPER_SCHEMA, purpose='shopper_turn') == {'text': 'Hello'}
        with pytest.raises(RuntimeError, match='budget'):
            client.json('instructions', {}, SHOPPER_SCHEMA, purpose='shopper_turn')
    assert captured[0]['store'] is False
    assert captured[0]['text']['format']['strict'] is True
    assert 'test-private-key' not in json.dumps(client.traces)


def test_api_adapter_requires_key(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(ValueError, match='OPENAI_API_KEY'):
        ResponsesClient('test-model')


def test_dissatisfaction_does_not_invent_a_preference():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'poor_results_positive_framing')
    first, second = plan['exchanges']
    assert first['assistant']['deliberate_error']
    assert len(first['assistant']['violating_product_ids']) == 5
    assert 'perfect' in first['assistant']['text']
    assert first['planned_state'] == second['planned_state']
    assert second['expected_events'] == []
    assert second['reaction_to'] == 'a1'
    assert not second['assistant']['violating_product_ids']


def test_distractor_roles_depend_on_current_requirements():
    vertical, _, catalog = fixtures()
    pool = json.loads((DEBUG / 'distractor_pool.json').read_text())
    validate_catalog(pool, vertical)
    assert len(pool['products']) == 30
    hard = [vertical.assignment('sport', 'road_running')]
    soft = [vertical.assignment('brand', 'Nike')]
    roles = classify_products(catalog, hard, soft)
    assert roles['DBG-SHOE-001']['role'] == 'suitable'
    assert roles['DBG-SHOE-002']['role'] == 'soft_miss'
    assert roles['DBG-SHOE-016']['role'] == 'hard_mismatch'
    assert classify_products(pool, [], [])['DIST-SHOE-001']['role'] == 'suitable'
    assert classify_products(pool, hard, [])['DIST-SHOE-001']['role'] == 'hard_mismatch'
    assert classify_products(pool, hard, [])['DIST-SHOE-003']['role'] == 'suitable'


def test_optional_pool_is_used_in_all_plans_without_duplicate_ids():
    path = ROOT / 'scripts/debug_session_scenarios.py'
    spec = importlib.util.spec_from_file_location('debug_session_scenarios', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for include in (False, True):
        for seed in (0, 7, 41):
            for case in CASE_MODES:
                _, plan = module.make_plan({'case': case, 'seed': seed, 'include_distractors': include})
                assert len(plan['catalog']['products']) == (60 if include else 30)
                validate_plan(plan)


@pytest.mark.parametrize('count', range(1, 8))
def test_requested_turn_count_preserves_core_scenario(count):
    vertical, _, catalog = fixtures()
    for case in CASE_MODES:
        if count < CASE_MIN_TURNS[case]:
            with pytest.raises(ValueError, match='requires at least'):
                generate_plan(vertical, catalog, case, turn_count=count)
        else:
            plan = generate_plan(vertical, catalog, case, turn_count=count)
            assert len(plan['exchanges']) == plan['turn_count'] == count
            assert plan['core_turn_count'] == CASE_MIN_TURNS[case] - int(case in ('finish_task_positive', 'finish_task_negative'))
            validate_plan(plan)


@pytest.mark.parametrize('count', [0, 8, 2.5, True, '3'])
def test_bad_turn_counts_are_rejected(count):
    vertical, _, catalog = fixtures()
    with pytest.raises(ValueError, match='integer from 1 to 7'):
        generate_plan(vertical, catalog, 'initial_need', turn_count=count)


@pytest.mark.parametrize('kind', ['positive', 'negative', 'ambiguous'])
def test_every_finisher_closes_lifecycle_without_retracting_durable_preferences(kind):
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'reinforce_preference', turn_count=4, finisher=kind)
    assert len(plan['exchanges']) == 4
    ending = plan['finisher']
    assert ending['type'] == kind
    assert ending['outcome'] == {'positive': 'success', 'negative': 'failure', 'ambiguous': 'unknown'}[kind]
    assert ending['terminal_profile']['active_task_id'] is None
    assert all(e['scope'] != 'task' for e in ending['terminal_profile']['entries'])
    assert any(e['scope'] == 'durable' for e in ending['terminal_profile']['entries'])
    if kind == 'ambiguous':
        assert ending['message_id'] is None and ending['trigger'] == 'user_left'
        assert not any(step.get('finisher') for step in plan['exchanges'])
        assert plan['exchanges'][-1]['planned_state']['active_task_id'] == 'task-1'
        assert ending['lifecycle_expirations']
    else:
        assert ending['message_id'] == plan['exchanges'][-1]['shopper_message_id']
        assert plan['exchanges'][-1]['finisher'] == kind
        assert plan['exchanges'][-1]['planned_state']['active_task_id'] is None
    validate_plan(plan)


def test_finishers_do_not_silently_exceed_turn_budget():
    vertical, _, catalog = fixtures()
    for kind in ('positive', 'negative'):
        with pytest.raises(ValueError, match='requires at least 3 turns'):
            generate_plan(vertical, catalog, 'revise_value', turn_count=2, finisher=kind)
    assert len(generate_plan(vertical, catalog, 'initial_need', turn_count=1, finisher='ambiguous')['exchanges']) == 1


def test_finisher_cannot_be_missing_or_have_tampered_lifecycle():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'initial_need')
    missing = copy.deepcopy(plan)
    del missing['finisher']
    with pytest.raises(ValueError, match='every chain needs'):
        validate_plan(missing)
    plan['finisher']['terminal_profile']['entries'] = plan['exchanges'][-1]['planned_state']['entries']
    with pytest.raises(ValueError, match='lifecycle does not match'):
        validate_plan(plan)


def test_departure_does_not_add_llm_calls_or_fabricated_chat_evidence():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'initial_need', turn_count=1, finisher='ambiguous')
    persona = json.loads((DEBUG / 'user_persona.json').read_text())
    shopper, extractor = FakeShopper(), FakeExtractor(plan)
    result = render_and_test(plan, persona, shopper, extractor)
    assert len(result['conversation']) == 2
    assert len(shopper.payloads) == len(extractor.payloads) == 1
    assert result['finisher']['outcome'] == 'unknown'
    assert extractor.payloads[0]['visible_conversation'][0]['role'] == 'user'
    assert len(extractor.payloads[0]['visible_conversation']) == 1
    assert 'finisher' not in extractor.payloads[0]


def test_seven_turn_shopper_only_sees_last_assistant_extractor_sees_full_prefix():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'initial_need', turn_count=7, finisher='positive')
    persona = json.loads((DEBUG / 'user_persona.json').read_text())
    shopper, extractor = FakeShopper(), FakeExtractor(plan)
    result = render_and_test(plan, persona, shopper, extractor)
    assert result['status'] == 'completed'
    assert len(shopper.payloads) == len(extractor.payloads) == 7
    for index, payload in enumerate(shopper.payloads):
        assert set(payload) == {'persona', 'last_assistant_message', 'current_instruction'}
        assert payload['current_instruction'] == plan['exchanges'][index]['shopper_instruction']
        assert payload['last_assistant_message'] == (None if index == 0 else {
            'id': plan['exchanges'][index - 1]['assistant']['message_id'],
            'role': 'assistant', 'text': plan['exchanges'][index - 1]['assistant']['text'],
        })
        history = extractor.payloads[index]['visible_conversation']
        assert len(history) == index * 2 + 1
        assert history[0]['id'] == 'u1'
        assert history[-1]['id'] == 'u' + str(index + 1)


@pytest.mark.parametrize('expression', ['explicit', 'implicit', 'mixed'])
def test_context_dependent_subtype_is_independent_of_evidence_mode(expression):
    from adaptgym.session_plans import CASE_SUBTYPES, CONTEXT_DEPENDENT_CASES
    vertical, _, catalog = fixtures()
    for case in CONTEXT_DEPENDENT_CASES:
        plan = generate_plan(vertical, catalog, case, expression=expression)
        assert plan['mode'] == 'assistant_reactive'
        assert plan['scenario_subtype'] == CASE_SUBTYPES[case] == 'context_dependent'
        assert plan['expression'] == expression
        assert plan['exchanges'][0]['turn_subtype'] is None
        reaction = plan['exchanges'][1]
        assert reaction['turn_subtype'] == 'context_dependent'
        assert reaction['reaction_to'] == 'a1'
        for event in reaction['expected_events']:
            assert 'a1' in event['evidence'][0]['message_ids']
        validate_plan(plan)
    self_contained = generate_plan(vertical, catalog, 'hard_constraint_violated', expression=expression)
    assert self_contained['mode'] == 'assistant_reactive'
    assert self_contained['scenario_subtype'] is None


def test_contextual_yes_no_resolve_against_actual_second_product():
    vertical, _, catalog = fixtures()
    for mode in ('explicit', 'implicit'):
        yes = generate_plan(vertical, catalog, 'suggestion_confirm', expression=mode)
        first, second = yes['exchanges']
        assert len(first['assistant']['product_ids']) == min(5, len(first['assistant']['matching_product_ids']))
        assert len(first['assistant']['product_ids']) >= 2
        assert 'second pair' in first['assistant']['text']
        anchor = next(p for p in catalog['products'] if p['id'] == first['assistant']['product_ids'][1])
        durable = [e for e in second['planned_state']['entries'] if e['scope'] == 'durable']
        assert len(durable) == 1
        assert durable[0]['assignment']['value'] == anchor['attributes']['brand']
        assert durable[0]['evidence_mode'] == mode
        assert durable[0]['evidence'][0]['message_ids'] == ['a1', 'u2']
    no = generate_plan(vertical, catalog, 'suggestion_reject')
    assert no['exchanges'][0]['planned_state'] == no['exchanges'][1]['planned_state']
    assert 'end the shopping task' in no['exchanges'][1]['shopper_instruction']


def test_context_subtype_rejects_inconsistent_plan_or_turn_tags():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'select_product')
    plan['mode'] = 'assistant_free'
    with pytest.raises(ValueError, match='mode/subtype'):
        validate_plan(plan)
    plan = generate_plan(vertical, catalog, 'select_product')
    plan['exchanges'][0]['turn_subtype'] = 'context_dependent'
    with pytest.raises(ValueError, match='actual assistant-context dependency'):
        validate_plan(plan)



def test_current_need_varies_type_feature_count_and_budget_and_is_reproducible():
    from adaptgym.current_need import generate_current_need
    vertical, _, catalog = fixtures()
    needs = [generate_current_need(vertical, catalog, seed=seed) for seed in range(60)]
    assert len({n['shoe_type']['value'] for n in needs}) >= 6
    assert {n['feature_count'] for n in needs} == {1, 2, 3, 4}
    budgets = {a['value'] for n in needs for a in n['requirements'] if a['attribute'] == 'price'}
    assert len(budgets) >= 3
    assert any(all(a['attribute'] != 'price' for a in n['requirements']) for n in needs)
    for seed, need in enumerate(needs):
        assert need == generate_current_need(vertical, catalog, seed=seed)
        assert need['matching_product_ids'] == matching_ids(catalog, need['requirements'])
        assert need['matching_product_ids']
        assert len(need['requirements']) == need['feature_count'] + 1


def test_any_middle_scenario_obeys_filters_and_exact_three_turn_budget():
    from adaptgym.session_plans import CASE_SUBTYPES, MIDDLE_CASES
    spec = importlib.util.spec_from_file_location('debug_middle_choices', ROOT / 'scripts/debug_session_scenarios.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cases = set()
    for mode, subtype in [('all', 'all'), ('assistant_free', 'all'), ('assistant_reactive', 'all'), ('assistant_reactive', 'context_dependent')]:
        for seed in range(8):
            payload = dict(case='any', seed=seed, turn_count=3, finisher='positive', mode_filter=mode, subtype_filter=subtype)
            _, plan = module.make_plan(payload)
            assert len(plan['exchanges']) == 3
            assert plan['core_turn_count'] == 2
            assert plan['scenario_type'] in MIDDLE_CASES
            assert plan['finisher']['type'] == 'positive'
            if mode != 'all':assert plan['mode'] == mode
            if subtype != 'all':assert CASE_SUBTYPES[plan['scenario_type']] == subtype
            assert module.make_plan(payload)[1] == plan
            cases.add(plan['scenario_type'])
    assert len(cases) >= 5


class FakeVerifier:
    def __init__(self, issue=None):
        self.payloads, self.issue = [], issue

    def json(self, instructions, payload, schema, *, purpose):
        assert purpose == 'intent_label_verification'
        assert set(payload) == {'vertical_schema', 'catalog', 'visible_conversation', 'candidate_intent'}
        self.payloads.append(copy.deepcopy(payload))
        mid = payload['visible_conversation'][-1]['id']
        check = {'verdict': 'supported', 'message_ids': [mid], 'reason': 'Supported by shopper.'}
        report = {'claim_checks': [{**check, 'claim_index': i}
                                  for i in range(len(payload['candidate_intent']['claims']))],
                  'active_task': {'observed_value': payload['candidate_intent']['active_task_id'],
                                  'certainty': 'resolved', 'message_ids': [mid], 'reason': 'Shopping task state.'},
                  'satisfied': {'observed_value': payload['candidate_intent']['satisfied'],
                                'certainty': 'resolved', 'message_ids': [mid], 'reason': 'Outcome evidence.'},
                  'target_product': {'observed_value': payload['candidate_intent']['target_product'],
                                     'certainty': 'resolved', 'message_ids': [mid], 'reason': 'Product evidence.'},
                  'additional_intent': [], 'intent': copy.deepcopy(check), 'search_request': copy.deepcopy(check), 'summary': 'All candidate labels are supported.'}
        if self.issue == 'contradiction':
            report['active_task']['observed_value'] = 'task-99'
        elif self.issue == 'uncertain':
            report['target_product']['certainty'] = 'uncertain'
        elif self.issue == 'omission':
            report['additional_intent'] = [{'description': 'Missing requirement', 'message_ids': [mid],
                'claim': {'attribute': 'price', 'operator': 'lte', 'value': 100,
                          'scope': 'task', 'task_id': 'task-1', 'strength': 'hard',
                          'evidence': [{'source_type': 'conversation', 'mode': 'explicit',
                                       'message_ids': [mid], 'justification': 'Budget.'}]}}]
        elif self.issue == 'future':
            report['active_task']['message_ids'] = ['u999']
        elif self.issue == 'missing_check':
            report['claim_checks'] = []
        elif self.issue == 'failure':
            raise RuntimeError('simulated API failure')
        return report


def test_verification_uses_visible_prefix_and_no_generator_or_extractor_secrets():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'clarification_question')
    persona = json.loads((DEBUG / 'user_persona.json').read_text())
    verifier = FakeVerifier()
    result = render_and_test(plan, persona, FakeShopper(), FakeExtractor(plan), verifier)
    assert result['label_status'] == 'model_verified'
    assert result['verified_checkpoints'] == len(plan['exchanges'])
    for index, payload in enumerate(verifier.payloads[::2]):
        assert len(payload['visible_conversation']) == 2 * index + 1
        assert payload['visible_conversation'][-1]['role'] == 'user'
        assert payload['candidate_intent']['claims'] == prediction_for(plan['exchanges'][index])['claims']
        assert result['checkpoints'][index]['verification']['accepted']


@pytest.mark.parametrize('issue,status', [('contradiction', 'completed'), ('uncertain', 'completed'),
    ('omission', 'completed'), ('future', 'failed'), ('missing_check', 'failed'), ('failure', 'failed')])
def test_verification_never_accepts_uncertain_incomplete_or_failed_audits(issue, status):
    from adaptgym.openai_session_test import verify_intent_labels
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'initial_need')
    step = plan['exchanges'][0]
    report = verify_intent_labels(FakeVerifier(issue), plan, step,
                                 [{'id': step['shopper_message_id'], 'role': 'user', 'text': 'My need.'}])
    assert report['status'] == status
    assert report['accepted'] is False


def test_verifier_failure_does_not_abort_extractor_run():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'clarification_question')
    persona = json.loads((DEBUG / 'user_persona.json').read_text())
    result = render_and_test(plan, persona, FakeShopper(), FakeExtractor(plan), FakeVerifier('failure'))
    assert result['status'] == 'completed'
    assert result['exact_checkpoints'] == len(plan['exchanges'])
    assert result['verified_checkpoints'] == 0
    assert all(c['verification']['status'] == 'failed' for c in result['checkpoints'])


def test_absence_of_selection_allows_empty_refs_but_claims_still_need_evidence():
    from adaptgym.openai_session_test import verify_intent_labels
    class AbsenceVerifier(FakeVerifier):
        def json(self, *args, **kwargs):
            report = super().json(*args, **kwargs)
            report['target_product']['message_ids'] = []
            return report
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'initial_need')
    step = plan['exchanges'][0]
    report = verify_intent_labels(AbsenceVerifier(), plan, step,
                                 [{'id': step['shopper_message_id'], 'role': 'user', 'text': 'My need.'}])
    assert report['accepted']


def test_verifier_compares_observed_null_selection_and_completed_task_in_code():
    from adaptgym.openai_session_test import verify_intent_labels
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'initial_need', turn_count=2, finisher='positive')
    step = plan['exchanges'][-1]
    result = verify_intent_labels(FakeVerifier(), plan, step,
        [{'id': step['shopper_message_id'], 'role': 'user', 'text': "I've found what I wanted; I'll proceed with the purchase."}])
    assert result['accepted']
    for field in ('active_task', 'target_product'):
        assert result[field]['observed_value'] is None
        assert result[field]['expected_value'] is None
        assert result[field]['verdict'] == 'supported'


def test_duplicate_omissions_do_not_reject_already_present_claims():
    from adaptgym.openai_session_test import verify_intent_labels
    class DuplicateVerifier(FakeVerifier):
        def json(self, instructions, payload, schema, *, purpose):
            report = super().json(instructions, payload, schema, purpose=purpose)
            report['additional_intent'] = [{'description': 'Already present',
                'message_ids': ['u1'], 'claim': copy.deepcopy(payload['candidate_intent']['claims'][0])}]
            return report
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'initial_need')
    report = verify_intent_labels(DuplicateVerifier(), plan, plan['exchanges'][0],
                                 [{'id': 'u1', 'role': 'user', 'text': 'My requirements.'}])
    assert report['accepted']
    assert report['additional_intent'] == []


def test_invalid_verifier_report_retains_diagnostic_instead_of_type_only_failure():
    from adaptgym.openai_session_test import verify_intent_labels
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'initial_need')
    report = verify_intent_labels(FakeVerifier('missing_check'), plan, plan['exchanges'][0],
                                 [{'id': 'u1', 'role': 'user', 'text': 'My requirements.'}])
    assert not report['accepted']
    assert report['validation_issue'] == 'verifier must check each claim exactly once'
    assert report['report']['claim_checks'] == []


def test_verifier_rejects_active_instead_of_a_task_identifier():
    from adaptgym.openai_session_test import verify_intent_labels
    class BadTaskIDVerifier(FakeVerifier):
        def json(self, *args, **kwargs):
            report = super().json(*args, **kwargs)
            report['active_task']['observed_value'] = 'active'
            return report
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'initial_need')
    report = verify_intent_labels(BadTaskIDVerifier(), plan, plan['exchanges'][0],
                                 [{'id': 'u1', 'role': 'user', 'text': 'My requirements.'}])
    assert report['status'] == 'failed'
    assert report['error_type'] == 'ValidationError'
    assert not report['accepted']


def test_search_request_separates_soft_preferences_and_task_overrides():
    intent = {'active_task_id': 'task-1', 'target_product': None, 'claims': [
        {'attribute': 'brand', 'operator': 'eq', 'value': 'Nike', 'scope': 'durable', 'task_id': None, 'strength': 'soft'},
        {'attribute': 'brand', 'operator': 'eq', 'value': 'Adidas', 'scope': 'task', 'task_id': 'task-1', 'strength': 'hard'},
        {'attribute': 'price', 'operator': 'lte', 'value': 150, 'scope': 'task', 'task_id': 'task-1', 'strength': 'hard'},
        {'attribute': 'colour', 'operator': 'eq', 'value': 'black', 'scope': 'durable', 'task_id': None, 'strength': 'soft'},
        {'attribute': 'size', 'operator': 'eq', 'value': 42, 'scope': 'unknown', 'task_id': None, 'strength': 'hard'},
    ]}
    request = build_search_request(intent, 'footwear')
    assert request['filters'] == [{'attribute': 'brand', 'operator': 'eq', 'value': 'Adidas'},
                                  {'attribute': 'price', 'operator': 'lte', 'value': 150}]
    assert request['preferences'] == [{'attribute': 'colour', 'operator': 'eq', 'value': 'black'}]
    assert build_search_request({**intent, 'active_task_id': None}, 'footwear') is None
    assert build_search_request({**intent, 'target_product': 'product-1'}, 'footwear') is None


def test_query_wording_is_not_scored_by_exact_string_match():
    vertical, _, catalog = fixtures()
    step = generate_plan(vertical, catalog, 'initial_need')['exchanges'][0]
    prediction = prediction_for(step)
    prediction['search_request']['query'] = 'Equivalent natural-language wording is judged separately'
    messages = [{'id': 'u1', 'role': 'user', 'text': step['shopper_instruction']}]
    score = score_extraction(prediction, step, vertical, messages, catalog)
    assert score['exact_state']
    assert score['query_text_evaluation'] == 'requires_semantic_judge'
    prediction['search_request']['filters'].append({'attribute': 'price', 'operator': 'lte', 'value': 10})
    score = score_extraction(prediction, step, vertical, messages, catalog)
    assert not score['search_constraints_correct']
    assert not score['exact_state']


def test_judge_rejects_unsupported_search_even_with_supported_claims():
    from adaptgym.openai_session_test import verify_intent_labels
    class BadSearchVerifier(FakeVerifier):
        def json(self, *args, **kwargs):
            report = super().json(*args, **kwargs)
            report['search_request']['verdict'] = 'contradicted'
            report['search_request']['reason'] = 'Query adds an unsupported constraint.'
            return report
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'initial_need')
    result = verify_intent_labels(BadSearchVerifier(), plan, plan['exchanges'][0],
                                 [{'id': 'u1', 'role': 'user', 'text': 'My requirements.'}])
    assert result['status'] == 'completed'
    assert not result['accepted']


def test_placeholder_query_is_invalid_even_when_intent_filters_are_correct():
    vertical, _, catalog = fixtures()
    step = generate_plan(vertical, catalog, 'initial_need')['exchanges'][0]
    prediction = prediction_for(step)
    prediction['search_request']['query'] = '/'
    score = score_extraction(prediction, step, vertical,
                             [{'id': 'u1', 'role': 'user', 'text': step['shopper_instruction']}], catalog)
    assert not score['valid']
    assert not score['exact_state']


def test_judge_rejects_wrong_intent_summary_despite_correct_attributes():
    from adaptgym.openai_session_test import verify_intent_labels
    class WrongSummaryVerifier(FakeVerifier):
        def json(self, *args, **kwargs):
            report = super().json(*args, **kwargs)
            report['intent']['verdict'] = 'contradicted'
            report['intent']['reason'] = 'Intent summary invents a different product need.'
            return report
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'initial_need')
    report = verify_intent_labels(WrongSummaryVerifier(), plan, plan['exchanges'][0],
                                 [{'id': 'u1', 'role': 'user', 'text': 'My requirements.'}])
    assert not report['accepted']
    assert report['status'] == 'completed'


def test_checkpoint_context_keeps_model_error_and_skips_invalid_update_without_gold():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'clarification_question', turn_count=3, finisher='positive')
    persona = json.loads((DEBUG / 'user_persona.json').read_text())
    class ImperfectExtractor(FakeExtractor):
        def json(self, *args, **kwargs):
            output = super().json(*args, **kwargs)
            if self.index == 1:
                claim = output['claims'][0]
                allowed = vertical.schema['properties'][claim['attribute']]['enum']
                claim['value'] = next(value for value in allowed if value != claim['value'])
            elif self.index == 2:
                output['claims'][0]['scope'] = 'unknown'  # Retains task_id: invalid.
            return output
    extractor = ImperfectExtractor(plan)
    result = render_and_test(plan, persona, FakeShopper(), extractor)
    first, second, third = result['checkpoints']
    assert first['previous_prediction'] is None
    assert first['score']['valid'] and not first['score']['exact_state']
    assert not second['score']['valid']
    assert second['previous_prediction'] == first['prediction']
    assert third['previous_prediction'] == first['prediction']
    assert third['previous_prediction'] != prediction_for(plan['exchanges'][0])
    assert extractor.payloads[2]['previous_prediction'] == third['previous_prediction']


@pytest.mark.parametrize('ending,expected', [('positive', True), ('negative', False), ('ambiguous', 'unk')])
def test_outcome_comes_from_expressed_ending_not_assistant_framing(ending, expected):
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'initial_need', turn_count=3, finisher=ending)
    outcome = expected_outcome(plan, plan['exchanges'][-1])
    assert outcome['satisfied'] == expected
    assert outcome['target_product'] is None


def test_outcome_keeps_selected_product_when_task_closes():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'select_product', turn_count=3, finisher='positive')
    persona = json.loads((DEBUG / 'user_persona.json').read_text())
    result = render_and_test(plan, persona, FakeShopper(), FakeExtractor(plan))
    selected = plan['exchanges'][1]['target_product']
    assert selected
    last = result['checkpoints'][-1]
    assert last['prediction']['target_product'] == selected
    assert last['prediction']['satisfied'] is True
    assert last['expected_outcome'] == {'satisfied': True, 'target_product': selected}
    assert last['score']['outcome_correct']


def test_target_product_is_the_only_product_identity_in_extractor_and_judge():
    from adaptgym.openai_session_test import EXTRACT_SCHEMA, VERIFICATION_SCHEMA
    assert 'target_product' in EXTRACT_SCHEMA['properties']
    assert 'selected_product_id' not in EXTRACT_SCHEMA['properties']
    assert 'target_product' in VERIFICATION_SCHEMA['properties']
    assert 'selected_product' not in VERIFICATION_SCHEMA['properties']


def test_target_product_persists_across_followups_and_completion():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'select_product', turn_count=7, finisher='positive')
    product = plan['exchanges'][1]['target_product']
    assert product
    assert plan['exchanges'][0]['target_product'] is None
    assert all(step['target_product'] == product for step in plan['exchanges'][1:])
    assert plan['finisher']['target_product'] == product
    assert expected_outcome(plan, plan['exchanges'][-1]) == {'satisfied': True, 'target_product': product}


def test_new_task_resets_product_and_satisfaction_instead_of_carrying_old_choice():
    vertical, _, catalog = fixtures()
    plan = generate_plan(vertical, catalog, 'select_product', turn_count=3, finisher='positive')
    new_task = {'exchange': 4, 'expected_events': [{'operation': 'begin_task', 'task_id': 'task-2'}],
                'target_product': None}
    plan['exchanges'].append(new_task)
    assert expected_outcome(plan, new_task) == {'satisfied': 'unk', 'target_product': None}
    intent = {'active_task_id': 'task-2', 'target_product': None, 'claims': []}
    assert build_search_request(intent, 'footwear') is not None


def test_negative_completion_summary_describes_stopping_not_an_active_search():
    state = {'active_task_id': None, 'target_product': None, 'claims': [], 'satisfied': False}
    summary = describe_intent(state, 'footwear')
    assert summary == 'The shopper stopped shopping without finding what they wanted.'
    assert not summary.startswith('Find')
    assert build_search_request(state, 'footwear') is None
    assert describe_intent({**state, 'satisfied': True}, 'footwear') == 'The shopper found what they wanted and finished shopping.'
