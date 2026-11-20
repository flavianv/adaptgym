#!/usr/bin/env python3
"""Export frozen synthetic pilot traces for playback; never generates model calls."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from adaptgym.feature_scoring import summarize_features
from adaptgym.open_count_scoring import score_typed_and_open_counts, score_open_row
from adaptgym.verticals import VerticalSchema

MODES = ('full_reextract', 'accumulated_full', 'recent_with_state')
LABELS = {'luna': 'Luna', 'zero_shot': 'Qwen 4B zero-shot', 'v1': 'Qwen 4B v1',
          'v2': 'Qwen 4B v2', 'v3': 'Qwen 4B v3', 'v4': 'Qwen 4B v4-294'}
NOTICE = ('Synthetic 32-turn, six-conversation training-plan pilot; not held-out test evidence. '
          'Luna used provider sampling defaults; Qwen used temperature 0 / seed 7. '
          'Open content is scored by counts, not meaning. No model calls during playback.')


def read_rows(path):
    with gzip.open(path, 'rt') as file:
        return [json.loads(line) for line in file]


def fingerprints(value):
    if isinstance(value, dict):
        return {k: ('sha256:' + v if k.endswith('_sha256') and isinstance(v, str) and len(v) == 64
                    else fingerprints(v)) for k, v in value.items()}
    if isinstance(value, list):
        return [fingerprints(v) for v in value]
    return value


def export(data, records, predictions, output):
    rows = read_rows(data)
    with gzip.open(records, 'rt') as file:
        records = json.load(file)
    frozen = {}
    for model, path in predictions.items():
        items = read_rows(path)
        keyed = {(p['id'], p['mode']): p for p in items}
        assert len(keyed) == len(items) == len(rows) * len(MODES)
        assert set(keyed) == {(r['id'], mode) for r in rows for mode in MODES}
        frozen[model] = keyed
    index = json.loads((output / 'index.json').read_text())
    cases = []
    counts = {'footwear': 0, 'job_search': 0}
    for record in records:
        sid, domain, plan = record['id'], record['vertical'], record['plan']
        selected = sorted((r for r in rows if r['sample_id'] == sid), key=lambda r: int(r['id'].rsplit(':', 1)[1]))
        assert selected
        counts[domain] += 1
        alias = 'open-pilot-' + domain + '-' + str(counts[domain])
        conversation = json.loads(selected[-1]['messages'][1]['content'])['visible_conversation'][:]
        last = plan['exchanges'][-1]['assistant']
        conversation.append({'id': last['message_id'], 'role': 'assistant', 'text': last['text']})
        verification = {'accepted': True, 'note': 'Synthetic reference admitted after a separate preservation check. Extraction scoring uses deterministic checks, without an LLM judge.'}
        sample = {'sample_id': alias, 'plan': plan, 'result': {'conversation': conversation}}
        models = {}
        for model, items in frozen.items():
            models[model] = {}
            for mode in MODES:
                checkpoints, scores = [], []
                for turn, row in enumerate(selected, 1):
                    item = items[row['id'], mode]
                    profile = item['prediction']
                    payload = json.loads(row['messages'][1]['content'])
                    reference = json.loads(row['messages'][2]['content'])
                    step = plan['exchanges'][turn - 1]
                    # Keep the exact visible message window used by the saved request.
                    visible = json.loads(row['messages'][1]['content'])
                    visible['previous_prediction'] = item.get('previous_prediction')
                    if mode == 'full_reextract':
                        visible.pop('previous_prediction', None)
                    elif mode == 'recent_with_state':
                        visible['visible_conversation'] = visible['visible_conversation'][-3:]
                    request = {'prompt': row['messages'][0]['content'], 'payload': visible}
                    request_hash = hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
                    assert request_hash == item['input_sha256'], 'saved request identity mismatch'
                    reply = {'id': step['assistant']['message_id'], 'role': 'assistant', 'text': step['assistant']['text']}
                    score = score_typed_and_open_counts(profile, reference, VerticalSchema(payload['vertical_schema']))
                    scores.append(score)
                    checkpoints.append({'turn': turn, 'profile': profile, 'reference_profile': reference,
                        'score': (item.get('metrics') or score_open_row(profile, row, record))['score'], 'verification': verification,
                        'reference_user_message_ids': [m['id'] for m in payload['visible_conversation'] if m['role'] == 'user'],
                        'previous_prediction': item.get('previous_prediction'), 'messages': visible['visible_conversation'] + [reply],
                        'feature_score': score, 'input_sha256': item['input_sha256'],
                        'metrics': {'turns': turn, 'total_turns': len(selected), 'value_precision': score['precision'],
                                    'value_recall': score['recall'], 'feature_pass': score['accepted'],
                                    'attributes': len({c.get('attribute') for c in (profile or {}).get('claims', []) if isinstance(c, dict)}),
                                    'targets': len((profile or {}).get('target_products', [])),
                                    'selected': len((profile or {}).get('selected_products', [])),
                                    'item_preferences': len((profile or {}).get('preferences', []))}})
                models[model][mode] = {'completed': True, 'vertical': domain, 'label': LABELS[model],
                    'state_mode': mode, 'state_policy': 'latest_prediction',
                    'extractor_prompt': selected[0]['messages'][0]['content'], 'checkpoints': checkpoints,
                    'feature_summary': summarize_features(scores)}
        bundle = {'id': alias, 'vertical': domain, 'notice': NOTICE, 'sample': sample, 'models': models}
        (output / (alias + '.json')).write_text(json.dumps(fingerprints(bundle), ensure_ascii=False, separators=(',', ':')) + '\n')
        cases.append({'id': alias, 'vertical': domain, 'file': alias + '.json', 'label':
            'Open pilot · ' + ('Shoes' if domain == 'footwear' else 'Jobs') + ' ' + str(counts[domain]) + ' · ' + str(len(selected)) + ' turns',
            'turns': len(selected), 'cohort': 'synthetic_training_plan_pilot'})
    index['cases'] = cases + [c for c in index['cases'] if not c['id'].startswith('open-pilot-')]
    for model in predictions:
        if model not in {m['id'] for m in index['models']}:
            index['models'].append({'id': model, 'label': LABELS[model]})
    index['open_pilot'] = {'notice': NOTICE, 'conversations': len(records), 'turns': len(rows), 'models': list(predictions)}
    index['legacy_selection'] = index.get('legacy_selection', index['selection'])
    index['selection'] = {'method': 'Legacy test examples plus a separately labeled synthetic training-plan open pilot; see legacy_selection and open_pilot.'}
    (output / 'index.json').write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    return cases


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--prediction', action='append', required=True, help='MODEL=predictions.jsonl.gz')
    parser.add_argument('--output', type=Path, default=ROOT / 'examples/model_comparison')
    args = parser.parse_args()
    predictions = {name: Path(path) for name, path in (p.split('=', 1) for p in args.prediction)}
    if set(predictions) - LABELS.keys():
        parser.error('unsupported model label')
    print(json.dumps(export(args.data, args.records, predictions, args.output), indent=2))
