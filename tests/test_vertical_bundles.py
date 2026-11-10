"""One engine must run data-only verticals with opposing numeric objectives."""
import importlib.util
import json
from pathlib import Path
import threading
from urllib.request import urlopen, Request

import pytest

from adaptgym.bundles import VerticalBundle
from adaptgym.catalog import build_catalog, matches, validate_catalog
from adaptgym.openai_session_test import prompts_for, render_and_test
from adaptgym.session_plans import CASE_MODES, generate_plan, validate_plan
from adaptgym.verticals import VerticalSchema

ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / 'examples/job_search'


@pytest.mark.parametrize('directory', ['debug_shopping', 'job_search'])
@pytest.mark.parametrize('seed', [0, 7, 41])
def test_every_scenario_runs_from_bundle_data(directory, seed):
    bundle = VerticalBundle.load(ROOT / 'examples' / directory)
    catalog = build_catalog(bundle.template, bundle.vertical)
    for case in CASE_MODES:
        plan = generate_plan(bundle.vertical, catalog, case, seed=seed)
        validate_plan(plan)
        assert plan == generate_plan(bundle.vertical, catalog, case, seed=seed)
        for step in plan['exchanges']:
            for key in ('shopper_instruction',):
                assert '{task}' not in step[key] and '{activity}' not in step[key]
            if directory == 'job_search':
                assert not any(word in step['shopper_instruction'].lower()
                               for word in ('purchase', 'shopping', 'shoes', 'pair'))


def test_salary_relaxation_and_tradeoff_preserve_numeric_direction():
    bundle = VerticalBundle.load(JOBS)
    vertical, catalog = bundle.vertical, build_catalog(bundle.template, bundle.vertical)
    plan = generate_plan(vertical, catalog, 'no_match_relax')
    first, second = plan['exchanges']
    assert first['assistant']['matching_product_ids'] == []
    assert len(second['assistant']['product_ids']) == 5
    thresholds = [next(e['assignment'] for e in s['planned_state']['entries']
                      if e['assignment']['attribute'] == 'salary_hourly_jpy') for s in (first, second)]
    assert all(a['operator'] == 'gte' for a in thresholds)
    assert thresholds[1]['value'] < thresholds[0]['value']
    plan = generate_plan(vertical, catalog, 'tradeoff_accept')
    by_id = {p['id']: p for p in catalog['products']}
    a, b = [by_id[i]['attributes'] for i in plan['exchanges'][0]['assistant']['product_ids'][:2]]
    assert a['salary_hourly_jpy'] > b['salary_hourly_jpy']
    assert plan['exchanges'][1]['target_product'] == plan['exchanges'][0]['assistant']['product_ids'][0]


def test_omitted_attributes_are_unknown_and_do_not_match_constraints():
    bundle = VerticalBundle.load(JOBS)
    catalog = build_catalog(bundle.template, bundle.vertical)
    product = catalog['products'][0]
    product['attributes'].pop('salary_hourly_jpy')
    product['attributes'].pop('work_arrangement')
    validate_catalog(catalog, bundle.vertical)
    assert not matches(product, [bundle.vertical.assignment('salary_hourly_jpy', 0, 'gte')])
    assert not matches(product, [bundle.vertical.assignment('work_arrangement', 'on_site')])
    # Put missing-role items at the front of every possible context's shortlist.
    for product in catalog['products'][1:]:
        if product['attributes']['job_title'] == 'Accounting clerk':
            product['attributes'].pop('work_arrangement')
    for seed in (0, 7, 41):
        for case in CASE_MODES:
            validate_plan(generate_plan(bundle.vertical, catalog, case, seed=seed))


@pytest.mark.parametrize('updates', [
    {'numeric_operator': 'lt'},
    {'numeric_attribute': 'location_region'},
    {'preference_attribute': 'salary_hourly_jpy'},
    {'feature_operators': {'salary_hourly_jpy': 'eq', 'invented': 'eq'}},
    {'context_attributes': ['invented']},
    {'missing_attributes': 'fill_with_zero'},
    {'unknown_option': True},
])
def test_invalid_configuration_fails_early(updates):
    schema = VerticalBundle.load(JOBS).vertical.schema
    schema['x-adaptgym']['simulation'].update(updates)
    with pytest.raises(ValueError, match='simulation:'):
        _ = VerticalSchema(schema).simulation


def test_saved_schema_contains_everything_needed_to_replay_without_sidecar(tmp_path):
    bundle = VerticalBundle.load(JOBS)
    path = tmp_path / 'standalone.json'
    path.write_text(json.dumps(bundle.vertical.schema))
    replay = VerticalSchema.load(path)
    catalog = build_catalog(bundle.template, replay)
    assert replay.simulation == bundle.vertical.simulation
    assert generate_plan(replay, catalog, 'tradeoff_accept') == generate_plan(bundle.vertical, catalog, 'tradeoff_accept')


def test_third_vertical_can_rename_every_field_without_engine_changes():
    bundle = VerticalBundle.load(JOBS)
    schema = bundle.vertical.schema
    renamed = {name: 'custom_' + name for name in schema['properties']}
    schema['properties'] = {renamed[n]: f for n, f in schema['properties'].items()}
    for key in ('benchmark_attributes', 'preference_attributes'):
        if key in schema['x-adaptgym']:
            schema['x-adaptgym'][key] = [renamed[n] for n in schema['x-adaptgym'][key]]
    if 'disclosure_groups' in schema['x-adaptgym']:
        schema['x-adaptgym']['disclosure_groups'] = [[renamed[n] for n in group] for group in schema['x-adaptgym']['disclosure_groups']]
    schema['x-adaptgym']['vertical_id'] = 'custom_opportunities'
    config = schema['x-adaptgym']['simulation']
    for key in ('context_attributes', 'refinement_attributes', 'query_attributes', 'display_attributes'):
        config[key] = [renamed[n] for n in config[key]]
    for key in ('numeric_attribute', 'preference_attribute'):
        config[key] = renamed[config[key]]
    config['feature_operators'] = {renamed[n]: op for n, op in config['feature_operators'].items()}
    vertical = VerticalSchema(schema)
    catalog = build_catalog(bundle.template, bundle.vertical)
    catalog['vertical_id'] = vertical.vertical_id
    for product in catalog['products']:
        product['attributes'] = {renamed[n]: v for n, v in product['attributes'].items()}
    for case in CASE_MODES:
        validate_plan(generate_plan(vertical, catalog, case))


def test_all_llm_roles_receive_job_context_without_exposing_private_targets():
    bundle = VerticalBundle.load(JOBS)
    prompts = prompts_for(bundle.vertical)
    assert all('job seeker' in prompt and 'buying shoes' not in prompt for prompt in prompts.values())
    plan = generate_plan(bundle.vertical, build_catalog(bundle.template, bundle.vertical), 'select_product')
    class Capture:
        def __init__(self):
            self.calls = []
        def json(self, prompt, payload, schema, *, purpose):
            self.calls.append((purpose, prompt, payload))
            if purpose == 'shopper_turn':
                return {'text': 'I am looking for a suitable job.'}
            raise ValueError('offline capture; no model calls')
    client = Capture()
    result = render_and_test(plan, bundle.persona, client, client, verifier=client)
    assert result['status'] == 'completed'
    assert {'shopper_turn', 'intent_extraction', 'intent_label_verification'} <= {c[0] for c in client.calls}
    for purpose, prompt, payload in client.calls:
        assert 'job seeker' in prompt and 'buying shoes' not in prompt
        if purpose != 'shopper_turn':
            assert 'current_instruction' not in payload and 'planned_state' not in payload


def test_job_bundle_server_and_optional_distractors():
    spec = importlib.util.spec_from_file_location('job_demo', ROOT / 'scripts/debug_session_scenarios.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    server = demo.create_server(port=0, directory=JOBS)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f'http://127.0.0.1:{server.server_port}'
    try:
        with urlopen(base + '/api/fixtures') as response:
            fixtures = json.load(response)
        assert len(fixtures['catalog']['products']) == 36
        assert fixtures['simulation']['numeric_operator'] == 'gte'
        assert fixtures['distractor_pool']['products'] == []
        assert fixtures['prompts'] == prompts_for(VerticalSchema(fixtures['domain_schema']))
        payload = json.dumps({'case': 'no_match_relax', 'turn_count': 3, 'finisher': 'positive'}).encode()
        with urlopen(Request(base + '/api/plan', data=payload, headers={'Content-Type': 'application/json'})) as response:
            plan = json.load(response)
        validate_plan(plan)
        assert plan['vertical_id'] == 'job_search'
        with urlopen(base + '/lab') as response:
            assert b'attributeUnit' in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_footwear_v2_has_one_slot_for_generic_activities():
    bundle=VerticalBundle.load(ROOT/'examples/debug_shopping')
    vertical=bundle.vertical
    assert vertical.version=='2'
    sport_names={x.replace('_',' ').lower() for x in vertical.schema['properties']['sport']['enum']}
    uses={x.lower() for x in vertical.schema['properties']['primary_use']['enum']}
    assert not sport_names & uses
    vertical.assignment('sport','basketball')
    vertical.assignment('primary_use','Marathon racing')
    with pytest.raises(ValueError):vertical.assignment('primary_use','Basketball')
    catalog=build_catalog(bundle.template,vertical)
    for product in catalog['products']+bundle.distractors['products']:
        use=product['attributes'].get('primary_use')
        assert use is None or use.lower() not in sport_names
    assert all('primary_use' not in p['attributes'] for p in catalog['products'] if p['attributes']['sport']=='basketball')
