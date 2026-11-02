"""Validated, data-only configuration shared by all recommendation verticals."""
from __future__ import annotations

import copy
from typing import Any


def simulation_config(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve explicit configuration or domain-neutral defaults for old schemas."""
    fields = schema['properties']
    categorical = [n for n, f in fields.items() if f['type'] == 'string'
                   and len(f.get('enum', [])) >= 2 and 'eq' in f['x-adaptgym']['operators']]
    numeric = [n for n, f in fields.items() if f['type'] in ('integer', 'number')
               and 'enum' not in f and set(f['x-adaptgym']['operators']) & {'lte', 'gte'}]
    defaults = {
        'schema_version': 'adaptgym.simulation.v1',
        'label': schema.get('title', schema['x-adaptgym']['vertical_id']),
        'context_attributes': [n for n, f in fields.items() if 'eq' in f['x-adaptgym']['operators']],
        'preference_attribute': categorical[0] if categorical else None,
        'numeric_attribute': numeric[0] if numeric else None,
        'numeric_operator': ('lte' if not numeric or 'lte' in fields[numeric[0]]['x-adaptgym']['operators'] else 'gte'),
        'feature_operators': {n: 'eq' for n, f in fields.items() if 'eq' in f['x-adaptgym']['operators']},
        'refinement_attributes': [n for n, f in fields.items() if 'eq' in f['x-adaptgym']['operators']],
        'query_attributes': [],
        'display_attributes': list(fields)[:4],
        'missing_attributes': 'reject',
        'vocabulary': {'user': 'user', 'item': 'item', 'task': 'search', 'activity': 'searching'},
        'domain_instructions': '',
        'service_questions': [['additional details', 'Additional details are not available in this synthetic catalog.']],
    }
    raw = schema.get('x-adaptgym', {}).get('simulation', {})
    if not isinstance(raw, dict) or set(raw) - set(defaults):
        raise ValueError('simulation: unknown configuration keys')
    config = {**defaults, **copy.deepcopy(raw)}
    if config['schema_version'] != defaults['schema_version']:
        raise ValueError('simulation: unsupported schema_version')
    if not isinstance(config['label'], str) or not config['label'].strip():
        raise ValueError('simulation: label must be nonempty text')
    for key in ('context_attributes', 'refinement_attributes', 'query_attributes', 'display_attributes'):
        names = config[key]
        if (not isinstance(names, list) or any(not isinstance(n, str) or n not in fields for n in names)
                or len(names) != len(set(names))):
            raise ValueError(f'simulation: {key} must contain distinct schema attributes')
    for key in ('preference_attribute', 'numeric_attribute'):
        name = config[key]
        if name is not None and (not isinstance(name, str) or name not in fields):
            raise ValueError(f'simulation: {key} must name a schema attribute')
    focal, numeric = config['preference_attribute'], config['numeric_attribute']
    if focal is not None and focal not in categorical:
        raise ValueError('simulation: preference_attribute needs at least two string enum values and eq')
    if config['numeric_operator'] not in ('lte', 'gte'):
        raise ValueError('simulation: numeric_operator must be lte or gte')
    if numeric is not None and (numeric not in numeric_fields(fields)
            or config['numeric_operator'] not in fields[numeric]['x-adaptgym']['operators']):
        raise ValueError('simulation: numeric_attribute must support the configured ordered operator')
    ops = config['feature_operators']
    if not isinstance(ops, dict) or any(n not in fields or op not in ('eq', 'lte', 'gte')
            or op not in fields[n]['x-adaptgym']['operators'] for n, op in ops.items()):
        raise ValueError('simulation: feature_operators must map schema attributes to supported eq/lte/gte operators')
    if any('eq' not in fields[n]['x-adaptgym']['operators'] for n in config['context_attributes'] + config['refinement_attributes']):
        raise ValueError('simulation: context and refinement attributes must support eq')
    vocab = config['vocabulary']
    if (not isinstance(vocab, dict) or set(vocab) != {'user', 'item', 'task', 'activity'}
            or any(not isinstance(v, str) or not v.strip() for v in vocab.values())):
        raise ValueError('simulation: vocabulary requires nonempty user/item/task/activity strings')
    if config['missing_attributes'] not in ('reject', 'omit'):
        raise ValueError('simulation: missing_attributes must be reject or omit')
    if not isinstance(config['domain_instructions'], str):
        raise ValueError('simulation: domain_instructions must be text')
    questions = config['service_questions']
    if not isinstance(questions, list) or not questions or any(not isinstance(q, list) or len(q) != 2
            or any(not isinstance(v, str) or not v.strip() for v in q) for q in questions):
        raise ValueError('simulation: service_questions needs nonempty [topic, answer] pairs')
    return config


def numeric_fields(fields: dict) -> set[str]:
    return {n for n, f in fields.items() if f['type'] in ('integer', 'number') and 'enum' not in f}


def render_domain(text: str, config: dict) -> str:
    for key, value in config['vocabulary'].items():
        text = text.replace('{' + key + '}', value)
    return text


def domain_prompt(base: str, config: dict) -> str:
    context = config['domain_instructions'].strip()
    return render_domain(base, config) + ('\nDomain context:\n' + context if context else '')
