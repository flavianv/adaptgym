"""A simulator vertical is a directory of JSON data, never executable plugins."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from adaptgym.catalog import build_catalog, validate_catalog
from adaptgym.persona import validate_persona
from adaptgym.verticals import VerticalSchema


@dataclass
class VerticalBundle:
    directory: Path
    vertical: VerticalSchema
    persona: dict
    template: dict
    distractors: dict

    @classmethod
    def load(cls, directory: str | Path) -> 'VerticalBundle':
        directory = Path(directory)
        for name in ('domain_schema.json', 'simulation.json', 'user_persona.json', 'catalog_template.json'):
            if not (directory / name).is_file():
                raise ValueError(f'vertical bundle is missing {name}: {directory}')
        def read(name):
            return json.loads((directory / name).read_text(encoding='utf-8'))
        vertical = VerticalSchema.load(directory / 'domain_schema.json')
        config = vertical.simulation
        if not config['context_attributes'] or not config['numeric_attribute'] or not config['preference_attribute']:
            raise ValueError('simulator bundle must configure context, numeric and preference attributes')
        persona, template = read('user_persona.json'), read('catalog_template.json')
        if validate_persona(persona['profile']):
            raise ValueError('bundle persona failed schema validation')
        build_catalog(template, vertical)
        pool_path = directory / 'distractor_pool.json'
        pool = read(pool_path.name) if pool_path.exists() else {
            'vertical_id': vertical.vertical_id, 'schema_version': vertical.version, 'products': []}
        if pool_path.exists():
            validate_catalog(pool, vertical)
        return cls(directory, vertical, persona, template, pool)
