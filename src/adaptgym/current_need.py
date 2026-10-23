"""Seeded catalog-grounded sampling of what the shopper currently needs."""
from __future__ import annotations

import math
import random
from typing import Any

from adaptgym.catalog import matching_ids
from adaptgym.verticals import VerticalSchema


def generate_current_need(vertical: VerticalSchema, catalog: dict, *, seed: int,
                          minimum_matches: int = 1, min_features: int = 1,
                          max_features: int = 4, type_candidates: list[dict] | None = None,
                          diversity_attribute: str | None = None) -> dict[str, Any]:
    """Sample a type and a variable number of feasible hard requirements.

    Numeric budgets are optional, not always the catalog's maximum price.
    A planner can require several matches and two distinct preference values.
    The seed controls both the choice of type and its requested features.
    """
    if not 1 <= minimum_matches or not 1 <= min_features <= max_features:
        raise ValueError('invalid need-generator bounds')
    rng = random.Random(seed)
    fields = vertical.schema['properties']
    products = catalog['products']

    def feasible(requirements):
        ids = matching_ids(catalog, requirements)
        found = [p for p in products if p['id'] in ids]
        diverse = not diversity_attribute or len({p['attributes'][diversity_attribute] for p in found}) >= 2
        return found if len(found) >= minimum_matches and diverse else []

    if type_candidates is None:
        name = 'sport' if 'sport' in fields else 'category' if 'category' in fields else next(iter(fields))
        type_candidates = [vertical.assignment(name, value) for value in dict.fromkeys(p['attributes'][name] for p in products)]
    viable_types = [a for a in type_candidates if feasible([a])]
    if not viable_types:
        raise ValueError('no shoe type meets the requested match/diversity conditions')
    shoe_type = rng.choice(viable_types)
    requirements = [shoe_type]
    target = rng.randint(min_features, max_features)
    available = [name for name in ('price', 'size', 'colour', 'gender', 'brand', 'cushioning',
                                   'upper_material', 'surface', 'style', 'closure')
                 if name in fields and name not in (shoe_type['attribute'], diversity_attribute)]
    for _ in range(target):
        options = []
        remaining = feasible(requirements)
        for name in available:
            field = fields[name]
            candidates = []
            if field['type'] in ('number', 'integer') and 'lte' in field['x-adaptgym']['operators'] and 'enum' not in field:
                low = max(field.get('minimum', 0), min(p['attributes'][name] for p in remaining))
                high = min(field.get('maximum', math.inf), max(p['attributes'][name] for p in remaining) * 1.25)
                step = max(5, field.get('multipleOf', 1))
                values = [i * step for i in range(math.ceil(low / step), math.floor(high / step) + 1)]
                for value in values:
                    try:
                        candidates.append(vertical.assignment(name, value, 'lte'))
                    except ValueError:
                        pass
            elif 'eq' in field['x-adaptgym']['operators']:
                candidates = [vertical.assignment(name, value) for value in dict.fromkeys(p['attributes'][name] for p in remaining)
                              if value != 'unspecified']
            supported = [a for a in candidates if feasible(requirements + [a])]
            if supported:
                options.append((name, supported))
        if not options:
            break
        name, candidates = rng.choice(options)
        requirements.append(rng.choice(candidates))
        available.remove(name)
    return {'shoe_type': shoe_type, 'feature_count': len(requirements) - 1,
            'requested_feature_count': target, 'requirements': requirements,
            'matching_product_ids': matching_ids(catalog, requirements), 'seed': seed}
