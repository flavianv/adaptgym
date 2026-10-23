"""Reference search-tool requests derived from current, scoped intent.

Query wording is illustrative; semantic correctness is assessed by the model judge.
"""
from __future__ import annotations

import json


def build_search_request(intent: dict, vertical_id: str) -> dict | None:
    if not intent['active_task_id'] or intent.get('target_product'):
        return None
    applicable = {}
    for claim in intent['claims']:
        if claim['scope'] == 'unknown' or claim['strength'] == 'unknown':
            continue
        if claim['scope'] == 'task' and claim['task_id'] != intent['active_task_id']:
            continue
        previous = applicable.get(claim['attribute'])
        if previous is None or claim['scope'] == 'task':
            applicable[claim['attribute']] = claim
    filters, preferences, terms = [], [], [vertical_id.replace('_', ' ')]
    for name, claim in sorted(applicable.items()):
        assignment = {key: claim[key] for key in ('attribute', 'operator', 'value')}
        (filters if claim['strength'] == 'hard' else preferences).append(assignment)
        if (claim['strength'] == 'hard' and claim['operator'] == 'eq' and
                isinstance(claim['value'], str) and name in
                {'category', 'shoe_type', 'sport', 'style', 'surface', 'primary_use'}):
            terms.append(claim['value'].replace('_', ' '))
    return {'query': ' '.join(terms), 'filters': filters, 'preferences': preferences}


def same_search_constraints(actual: dict | None, expected: dict | None) -> bool:
    if actual is None or expected is None:
        return actual is expected
    def signatures(items):
        return sorted(json.dumps(item, sort_keys=True) for item in items)
    return all(signatures(actual[key]) == signatures(expected[key]) for key in ('filters', 'preferences'))


def describe_intent(intent: dict, vertical_id: str) -> str:
    """Reference wording, judged semantically rather than by exact match."""
    def text(claim):
        value = claim['value']
        rendered = ', '.join(map(str, value)) if isinstance(value, list) else str(value)
        operator = {'eq': 'is', 'lte': 'at most', 'lt': 'less than', 'gte': 'at least',
                    'gt': 'greater than', 'in': 'one of'}[claim['operator']]
        return f"{claim['attribute']} {operator} {rendered}".replace('_', ' ')
    if not intent['active_task_id']:
        durable = [text(c) for c in intent['claims'] if c['scope'] == 'durable']
        completion = ('The shopper stopped shopping without finding what they wanted.'
                      if intent.get('satisfied') is False else
                      'The shopper found what they wanted and finished shopping.'
                      if intent.get('satisfied') is True else 'No active shopping task.')
        return completion + (' Retain preferences: ' + '; '.join(durable) + '.' if durable else '')
    request = build_search_request(intent, vertical_id)
    if request is None:
        return f"Product {intent['target_product']} selected for the current purchase."
    hard = '; '.join(text(c) for c in request['filters'])
    soft = '; '.join(text(c) for c in request['preferences'])
    return (f"Find {vertical_id.replace('_', ' ')} for this purchase" +
            (': ' + hard if hard else '') + '.' + (' Prefer ' + soft + '.' if soft else ''))


def expected_outcome(plan: dict, step: dict) -> dict:
    """Only consider exchanges through this shopper checkpoint."""
    product = None
    satisfied = 'unk'
    for exchange in plan['exchanges']:
        if any(e['operation'] == 'begin_task' for e in exchange['expected_events']):
            product, satisfied = None, 'unk'
        if exchange['target_product']:
            product = exchange['target_product']
        if exchange.get('finisher') in ('positive', 'negative'):
            satisfied = exchange['finisher'] == 'positive'
        if exchange['exchange'] == step['exchange']:
            break
    return {'satisfied': satisfied, 'target_product': product}
