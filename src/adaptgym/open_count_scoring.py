"""Exact typed claims plus count-only penalties for open claims.

Open names and values are never matched to reference names or values. Identities
are used only within each individual output to deduplicate keys and values.
This measures cardinality, not semantic correctness of open claims.
"""
from __future__ import annotations

import copy

from adaptgym.feature_scoring import atom, score_features

POLICY = 'typed-exact-open-counts-v1'
KINDS = ('new_feature', 'new_value')


def _typed_value(vertical, attribute, value):
    try:
        vertical._check_value(attribute, value)
        return True
    except (ValueError, TypeError):
        # Preserve the existing case-insensitive enum convention.
        return isinstance(value, str) and any(
            isinstance(item, str) and value.lower() == item.lower()
            for item in vertical._schema['properties'][attribute].get('enum', ())
        )


def partition(profile, vertical):
    """Return typed claims and distinct open-feature/open-value counts.

    A new feature is one distinct key absent from the schema, regardless of its
    values. A new value is one distinct (known field, out-of-schema scalar) pair;
    membership lists contribute their individual novel values. Duplicate claims
    and operator/scope variants do not create extra keys or values.
    """
    typed, new_features, new_values = [], set(), set()
    claims = profile.get('claims', []) if isinstance(profile, dict) else []
    if not isinstance(claims, list):
        claims = []
    properties = vertical._schema['properties']
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get('attribute'), str):
            continue
        attribute = claim['attribute'].lower()
        if not attribute.strip():
            continue
        if attribute not in properties:
            new_features.add(attribute)
            continue
        values = claim.get('value')
        values = values if isinstance(values, list) else [values]
        known = []
        for value in values:
            if _typed_value(vertical, attribute, value):
                known.append(value)
            elif isinstance(value, (str, int, float, bool)):
                new_values.add((attribute, atom(value)))
        if known:
            item = copy.deepcopy(claim)
            item['value'] = known if isinstance(claim.get('value'), list) else known[0]
            typed.append(item)
    result = copy.deepcopy(profile) if isinstance(profile, dict) else {}
    result['claims'] = typed
    return result, {'new_feature': len(new_features), 'new_value': len(new_values)}


def score_open_counts(prediction, reference, vertical):
    """L1 count penalty, separately by kind; no cross-output identity matching."""
    _, expected = partition(reference, vertical)
    _, predicted = partition(prediction, vertical)
    delta = {kind: predicted[kind] - expected[kind] for kind in KINDS}
    under = {kind: max(0, -delta[kind]) for kind in KINDS}
    over = {kind: max(0, delta[kind]) for kind in KINDS}
    penalties = {kind: abs(delta[kind]) for kind in KINDS}
    penalty = sum(penalties.values())
    return {'policy': POLICY, 'expected': expected, 'predicted': predicted,
            'delta': delta, 'under': under, 'over': over,
            'penalty_by_kind': penalties, 'penalty': penalty,
            'accepted': penalty == 0}


def score_typed_and_open_counts(prediction, reference, vertical):
    """Keep legacy typed scoring and penalize open count errors only.

    TP/FP/FN combine typed atoms with count overlap/excess/deficit. Open count
    overlap is explicitly not a semantic true-positive measurement.
    """
    typed_prediction, _ = partition(prediction, vertical)
    typed_reference, _ = partition(reference, vertical)
    present = isinstance(prediction, dict) and isinstance(prediction.get('claims'), list)
    typed = score_features(typed_prediction if present else None, typed_reference)
    counts = score_open_counts(prediction, reference, vertical)
    errors = list(typed['errors']) + [
        {'type': 'open_count_mismatch', 'attribute': kind + '_count',
         'expected': counts['expected'][kind], 'extracted': counts['predicted'][kind],
         'penalty': counts['penalty_by_kind'][kind]}
        for kind in KINDS if counts['penalty_by_kind'][kind]
    ]
    tp = typed['tp'] + sum(min(counts['expected'][k], counts['predicted'][k]) for k in KINDS)
    fp = typed['fp'] + sum(counts['over'].values())
    fn = typed['fn'] + sum(counts['under'].values())
    return {**typed, 'policy': POLICY, 'accepted': typed['accepted'] and counts['accepted'],
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp / (tp + fp) if tp + fp else None,
            'recall': tp / (tp + fn) if tp + fn else None,
            'errors': errors, 'error_count': len(errors),
            'typed': typed, 'open_counts': counts}


def summarize_open_counts(scores):
    scores = list(scores)
    return {'policy': POLICY, 'turns': len(scores),
            'count_matched_turns': sum(s['accepted'] for s in scores),
            'count_match_rate': sum(s['accepted'] for s in scores) / len(scores) if scores else None,
            'total_penalty': sum(s['penalty'] for s in scores),
            'mean_penalty': sum(s['penalty'] for s in scores) / len(scores) if scores else None,
            **{name: {k: sum(s[name][k] for s in scores) for k in KINDS}
               for name in ('expected', 'predicted', 'under', 'over', 'penalty_by_kind')}}


def score_open_row(prediction, row, record=None, error=None):
    """State/typed checks plus count-only open scoring for a saved extraction."""
    import json
    from jsonschema import Draft202012Validator
    from adaptgym.intent_state import validate_evidence
    from adaptgym.openai_session_test import EXTRACT_SCHEMA
    from adaptgym.search_requests import same_search_constraints
    from adaptgym.verticals import VerticalSchema

    target = json.loads(row['messages'][2]['content'])
    payload = json.loads(row['messages'][1]['content'])
    vertical = VerticalSchema(payload['vertical_schema'])
    fields = {'active_task_correct': 'active_task_id', 'selected_products_correct': 'selected_products',
              'target_products_correct': 'target_products', 'preferences_correct': 'preferences',
              'outcome_correct': 'satisfied'}
    score = {'valid': False, **{k: False for k in fields}, 'search_constraints_correct': False}

    def typed_request(request):
        if request is None:
            return None
        return {**request, **{key: partition({'claims': request[key]}, vertical)[0]['claims']
                             for key in ('filters', 'preferences')}}

    try:
        if prediction is None:
            raise ValueError(error or 'NoPrediction')
        Draft202012Validator(EXTRACT_SCHEMA).validate(prediction)
        for claim in prediction['claims']:
            validate_evidence(claim['evidence'], payload['visible_conversation'])
            if claim['scope'] == 'task' and (not prediction['active_task_id'] or
                                            claim['task_id'] != prediction['active_task_id']):
                raise ValueError('inactive task')
            if claim['scope'] != 'task' and claim['task_id'] is not None:
                raise ValueError('non-task ID')
        typed_prediction, _ = partition(prediction, vertical)
        for claim in typed_prediction['claims']:
            vertical.assignment(claim['attribute'], claim['value'], claim['operator'])
        shown = {p['id'] for p in payload['catalog']['products']}
        ids = set(prediction['target_products'] + prediction['selected_products'])
        ids.update(item for preference in prediction['preferences'] for item in preference.values())
        if not ids <= shown:
            raise ValueError('item not shown in visible catalog')
        score['valid'] = True
        for key, field in fields.items():
            def canonical(value):
                return sorted(json.dumps(item, sort_keys=True) for item in value) if isinstance(value, list) else value
            score[key] = canonical(prediction[field]) == canonical(target[field])
        # Open strings cannot silently re-enter exact matching through search state.
        score['search_constraints_correct'] = same_search_constraints(
            typed_request(prediction['search_request']), typed_request(target['search_request']))
    except (ValueError, KeyError, TypeError) as exc:
        score['error_type'] = type(exc).__name__
    except Exception as exc:
        score['error_type'] = type(exc).__name__
    feature = score_typed_and_open_counts(prediction, target, vertical)
    typed_prediction, _ = partition(prediction, vertical)
    typed_target, _ = partition(target, vertical)

    def signatures(profile):
        return {json.dumps({k: c.get(k) for k in ('attribute', 'operator', 'value', 'scope', 'strength', 'task_id')},
                           sort_keys=True) for c in profile['claims']}

    state_pass = (score['valid'] and all(score[k] for k in (*fields, 'search_constraints_correct'))
                  and signatures(typed_prediction) == signatures(typed_target)
                  and feature['open_counts']['accepted'])
    score['strict_exact_state'] = state_pass  # Legacy wire key; semantics are versioned below.
    return {'policy': POLICY, 'score': score, 'open_counts': feature['open_counts'],
            'value_tp': feature['tp'], 'value_fp': feature['fp'], 'value_fn': feature['fn'],
            'complete_value_coverage': feature['accepted'],
            'exact_values': not feature['fp'] and not feature['fn'] and feature['output_present'],
            'strict_exact_state': state_pass, 'typed_state_and_open_counts': state_pass,
            'missing_values': feature['errors'], 'extra_values': feature['warnings']}
