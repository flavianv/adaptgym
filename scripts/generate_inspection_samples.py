#!/usr/bin/env python3
"""Generate resumable, local-only conversation samples with reference targets.

This is generation plus label auditing, not an extraction evaluation. No model
prediction or accuracy score is fabricated from reference labels.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from debug_session_scenarios import load_env_file
from adaptgym.bundles import VerticalBundle
from adaptgym.catalog import build_catalog
from adaptgym.openai_session_test import (ResponsesClient, SHOPPER_SCHEMA, prompts_for,
                                         reference_state, verify_intent_labels)
from adaptgym.session_plans import generate_plan, validate_plan


def sample_specs():
    for domain, lengths in [('job_search', [3, 4, 5, 6, 8]), ('debug_shopping', [3, 4, 5, 7, 8])]:
        for i, turns in enumerate(lengths):
            yield {'id': ('jobs' if domain == 'job_search' else 'shoes') + f'-{i+1}',
                   'directory': domain, 'turns': turns, 'seed': 101 + i,
                   'case': ['express_interest', 'compare_products', 'compare_products', 'mixed', 'mixed'][i],
                   'expression': 'implicit' if i == 2 else 'explicit',
                   'finisher': 'negative' if i == 4 else 'positive'}


def generate_sample(spec, output, model, verify=True, language='English'):
    bundle = VerticalBundle.load(ROOT / 'examples' / spec['directory'])
    catalog = build_catalog(bundle.template, bundle.vertical)
    plan = generate_plan(bundle.vertical, catalog, spec['case'], seed=spec['seed'],
                         turn_count=spec['turns'], expression=spec['expression'], finisher=spec['finisher'])
    validate_plan(plan)
    client = ResponsesClient(model, max_calls=4 * spec['turns'], max_output_tokens=4000)
    prompts = prompts_for(bundle.vertical)
    persona = json.loads((ROOT / 'examples' / spec['directory'] / 'user_persona.json').read_text())
    messages, checkpoints = [], []
    for step in plan['exchanges']:
        target = reference_state(plan, step, bundle.vertical)
        attempts = []
        for attempt in range(2 if verify else 1):
            reply = client.json(prompts['conversation_generator'] + '\nWrite every user message in ' + language + ', regardless of persona language.', {
                'persona': persona['profile'], 'visible_conversation': copy.deepcopy(messages),
                'last_assistant_message': copy.deepcopy(messages[-1]) if messages else None,
                'current_instruction': step['shopper_instruction'],
                'retry_feedback': attempts[-1]['verification'] if attempts else None,
            }, SHOPPER_SCHEMA, purpose='shopper_turn')
            if not reply['text'].strip():
                raise ValueError('empty user message')
            prefix = messages + [{'id': step['shopper_message_id'], 'role': 'user', 'text': reply['text']}]
            audit = (verify_intent_labels(client, plan, step, prefix) if verify else
                     {'status': 'not_run', 'accepted': False, 'verdict': 'unverified'})
            attempts.append({'text': reply['text'], 'verification': audit})
            if audit['accepted'] or not verify:
                break
        messages = prefix
        checkpoints.append({'turn': step['exchange'], 'prediction': None,
                            'score': {'not_run': True, 'valid': False, 'exact_state': False},
                            'expected_outcome': {k: target[k] for k in ('satisfied', 'target_products', 'selected_products', 'preferences')},
                            'reference_target': target, 'expected_intent': target['intent'],
                            'expected_search_request': target['search_request'], 'verification': audit,
                            'intent_judgment': {'status': 'not_run', 'accepted': False}, 'generation_attempts': attempts})
        messages.append({'id': step['assistant']['message_id'], 'role': 'assistant', 'text': step['assistant']['text']})
    accepted = sum(c['verification']['accepted'] for c in checkpoints)
    result = {'status': 'completed', 'plan_id': plan['id'], 'conversation': messages,
              'checkpoints': checkpoints, 'finisher': plan['finisher'], 'total_checkpoints': len(checkpoints),
              'verified_checkpoints': accepted, 'label_status': 'model_verified' if accepted == len(checkpoints) else 'needs_review',
              'interpretation': 'Generated inspection sample. Extraction was not run; reference targets are not predictions.'}
    value = {'plan': plan, 'result': result, 'model': model, 'generation_only': True,
             'sample_id': spec['id'], 'language': language, 'api_traces': client.traces}
    (output / (spec['id'] + '.json')).write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    print(f"{spec['id']}: {len(checkpoints)} turns; {accepted}/{len(checkpoints)} labels supported", flush=True)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/inspection-samples')
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--model', default='gpt-6-luna')
    parser.add_argument('--language', default='English')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--no-verify', action='store_true')
    args = parser.parse_args()
    load_env_file(args.env_file)
    args.output.mkdir(parents=True, exist_ok=True)
    specs = list(sample_specs())
    # Publish the stable inventory first. Only completed sample files are served.
    inventory = [{'id': s['id'], 'file': s['id']+'.json', 'vertical': 'job_search' if s['directory']=='job_search' else 'footwear',
                  'label': f"{s['id']} · {s['turns']} turns · {s['case']} · {s['expression']} · {s['finisher']}"} for s in specs]
    (args.output / 'index.json').write_text(json.dumps(inventory, indent=2)+'\n')
    pending = [s for s in specs if not (args.output / (s['id']+'.json')).exists()]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(lambda s: generate_sample(s, args.output, args.model, not args.no_verify, args.language), pending))


if __name__ == '__main__':
    main()
