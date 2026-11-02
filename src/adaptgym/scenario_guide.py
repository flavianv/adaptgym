"""User-facing scenario explanations with optional data-only vertical examples."""
import json
from pathlib import Path

from adaptgym.session_plans import CASE_MODES, CASE_SUBTYPES, CASE_MIN_TURNS, FINISHER_CASES, PREFERENCE_CASES


def load_scenario_guide(directory):
    descriptions = json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))
    if set(descriptions) != set(CASE_MODES):
        raise ValueError('scenario guide must explain every registered scenario')
    examples_path = Path(directory) / 'scenario_examples.json'
    examples = json.loads(examples_path.read_text(encoding='utf-8')) if examples_path.exists() else {}
    if set(examples) - set(CASE_MODES):
        raise ValueError('scenario examples reference unknown scenarios')
    result = []
    for name, mode in CASE_MODES.items():
        entry = descriptions[name]
        if any(not isinstance(entry.get(key), str) or not entry[key].strip()
               for key in ('title', 'description', 'common_mistake')):
            raise ValueError(f'{name}: scenario explanation is incomplete')
        if not entry.get('expected_behavior') or not all(isinstance(value, str) and value for value in entry['expected_behavior']):
            raise ValueError(f'{name}: expected behavior is missing')
        example = examples.get(name)
        if example is not None:
            if not example.get('conversation') or not example.get('expected_result'):
                raise ValueError(f'{name}: scenario example is incomplete')
            for turn in example['conversation']:
                if turn.get('role') not in ('user', 'assistant', 'environment') or not isinstance(turn.get('text'), str) or not turn['text'].strip():
                    raise ValueError(f'{name}: invalid example turn')
        phase = ('initial' if name == 'initial_need' else 'finisher'
                 if name in FINISHER_CASES or name in ('finish_task', 'completion_confirm') else 'middle')
        group = 'context_dependent' if CASE_SUBTYPES[name] == 'context_dependent' else mode
        result.append({'id': name, **entry, 'phase': phase, 'group': group,
                       'min_turns': CASE_MIN_TURNS[name], 'preference_evidence': name in PREFERENCE_CASES,
                       'example': example})
    return result
