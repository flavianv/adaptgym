import copy

from adaptgym.player_diagnostics import annotate_evaluation

SCHEMA = {'type': 'object', 'additionalProperties': False,
          'x-adaptgym': {'vertical_id': 'test', 'version': '1'},
          'properties': {'sport': {'type': 'string', 'enum': ['road', 'trail'],
                                  'x-adaptgym': {'operators': ['eq', 'in']}}}}


def claim(attribute, value, scope='task', strength='hard'):
    return {'attribute': attribute, 'operator': 'eq', 'value': value,
            'scope': scope, 'strength': strength, 'evidence': [{'message_ids': ['u1']}]}


def test_open_wording_is_not_matched_but_scope_and_hardness_are():
    prediction = {'claims': [claim('unrelated_open_name', 'different words', 'durable', 'soft')]}
    reference = {'claims': [claim('new_feature', 'original words')]}
    raw = copy.deepcopy(prediction)
    trace = {'checkpoints': [{'profile': prediction, 'reference_profile': reference,
                              'messages': [{'id': 'u1', 'role': 'user'}]}]}
    annotate_evaluation(trace, SCHEMA)
    d = trace['checkpoints'][0]['error_diagnostics']
    assert d['open_counts']['penalty'] == 0
    assert d['metadata']['scope_errors'] == d['metadata']['strength_errors'] == 1
    assert trace['error_summary']['match_cov'] == 1
    assert prediction == raw


def test_missing_output_is_unassessed_and_counted_not_silently_correct():
    trace = {'checkpoints': [{'profile': None, 'reference_profile': {'claims': [claim('novel', 'yes')]},
                              'messages': [{'id': 'u1', 'role': 'user'}]}]}
    annotate_evaluation(trace, SCHEMA)
    s = trace['error_summary']
    assert s['scope_err'] is None and s['strength_err'] is None and s['match_cov'] == 0
    assert s['new_feat_err'] == 1
    assert trace['checkpoints'][0]['claim_kinds']['prediction'] == []


def test_carried_prior_citation_is_assessed_outside_recent_input_window():
    trace = {'checkpoints': [{'profile': {'claims': [claim('novel', 'yes', 'durable', 'soft')]},
                              'reference_profile': {'claims': [claim('novel', 'yes')]},
                              'messages': [{'id': 'u3', 'role': 'user'}],
                              'reference_user_message_ids': ['u1', 'u2', 'u3']}]}
    annotate_evaluation(trace, SCHEMA)
    assert trace['error_summary']['match_cov'] == 1
    assert trace['error_summary']['scope_err'] == trace['error_summary']['strength_err'] == 1


def test_malformed_claims_list_does_not_break_saved_player():
    trace = {'checkpoints': [{'profile': {'claims': None}, 'reference_profile': {'claims': [claim('novel', 'yes')]},
                              'messages': [{'id': 'u1', 'role': 'user'}]}]}
    annotate_evaluation(trace, SCHEMA)
    assert trace['checkpoints'][0]['claim_kinds']['prediction'] == []
    assert trace['error_summary']['metadata']['unusable_output_turns'] == 1
