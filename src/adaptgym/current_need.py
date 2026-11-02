"""Seeded catalog-grounded sampling of the user's current search needs."""
from __future__ import annotations

import random
from typing import Any

from adaptgym.catalog import matching_ids
from adaptgym.verticals import VerticalSchema


def generate_current_need(vertical: VerticalSchema, catalog: dict, *, seed: int,
                          minimum_matches: int = 1, min_features: int = 1,
                          max_features: int = 4, type_candidates: list[dict] | None = None,
                          diversity_attribute: str | None = None,
                          witness_attributes: tuple[str, ...] = ()) -> dict[str, Any]:
    """Sample a type and a variable number of feasible hard requirements.

    Numeric constraints are optional and use the configured comparison direction.
    A planner can require several matches and two distinct preference values.
    The seed controls both the choice of type and its requested features.
    """
    if not 1 <= minimum_matches or not 1 <= min_features <= max_features:
        raise ValueError('invalid need-generator bounds')
    rng = random.Random(seed)
    config = vertical.simulation
    products = catalog['products']

    def feasible(requirements):
        ids = matching_ids(catalog, requirements)
        found = [p for p in products if p['id'] in ids and
                 all(n in p['attributes'] and p['attributes'][n] != 'unspecified' for n in witness_attributes)]
        diverse = not diversity_attribute or len({p['attributes'][diversity_attribute] for p in found
                     if diversity_attribute in p['attributes'] and p['attributes'][diversity_attribute] != 'unspecified'}) >= 2
        return found if len(found) >= minimum_matches and diverse else []

    if type_candidates is None:
        type_candidates = [vertical.assignment(name, value)
                           for name in config['context_attributes']
                           for value in dict.fromkeys(p['attributes'][name] for p in products if name in p['attributes'])
                           if value != 'unspecified']
    viable_types = [a for a in type_candidates if feasible([a])]
    if not viable_types:
        raise ValueError('no context meets the requested match/diversity conditions')
    context = rng.choice(viable_types)
    requirements = [context]
    target = rng.randint(min_features, max_features)
    available = [name for name in config['feature_operators']
                 if name not in (context['attribute'], diversity_attribute)]
    for _ in range(target):
        options = []
        remaining = feasible(requirements)
        for name in available:
            candidates = []
            operator = config['feature_operators'][name]
            # Catalog witnesses keep thresholds feasible and avoid enumerating unbounded ranges.
            for value in dict.fromkeys(p['attributes'][name] for p in remaining if name in p['attributes']):
                if value != 'unspecified':
                    candidates.append(vertical.assignment(name, value, operator))
            supported = [a for a in candidates if feasible(requirements + [a])]
            if supported:
                options.append((name, supported))
        if not options:
            break
        name, candidates = rng.choice(options)
        requirements.append(rng.choice(candidates))
        available.remove(name)
    return {'context': context, 'feature_count': len(requirements) - 1,
            'requested_feature_count': target, 'requirements': requirements,
            'matching_product_ids': matching_ids(catalog, requirements), 'seed': seed}
