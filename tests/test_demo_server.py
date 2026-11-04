"""Hosted-demo health, public access and proxy-origin regressions."""
import importlib.util
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('hosted_demo', ROOT / 'scripts/debug_session_scenarios.py')
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


@pytest.fixture
def server_factory(monkeypatch):
    for key in ('ADAPTGYM_DEMO_PASSWORD', 'ADAPTGYM_DEMO_USER', 'ADAPTGYM_PUBLIC_URL',
                'RENDER_EXTERNAL_HOSTNAME', 'OPENAI_API_KEY'):
        monkeypatch.delenv(key, raising=False)
    servers = []

    def start(**environment):
        for key, value in environment.items():
            monkeypatch.setenv(key, value)
        server = demo.create_server(port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return 'http://127.0.0.1:' + str(server.server_port)

    yield start
    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(base, path='/', *, headers=None, payload=None):
    req = urllib.request.Request(base + path, headers=headers or {},
        data=None if payload is None else json.dumps(payload).encode())
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as response:
        return response.code, response.headers, response.read()


def test_local_demo_remains_usable_without_password(server_factory):
    base = server_factory()
    assert request(base)[0] == 200
    status, _, body = request(base, '/healthz')
    assert status == 200 and json.loads(body) == {'status': 'ok'}
    status, _, body = request(base, '/api/plan', headers={'Content-Type': 'application/json'}, payload={})
    assert status == 200 and len(json.loads(body)['exchanges']) == 3


def test_hosted_demo_has_no_login_even_with_legacy_password(server_factory, monkeypatch):
    base = server_factory(ADAPTGYM_DEMO_PASSWORD='legacy-test-password',
                          ADAPTGYM_DEMO_USER='legacy-user', OPENAI_API_KEY='test-server-key')
    for path in ('/', '/api/fixtures', '/api/last-test', '/healthz'):
        status, headers, body = request(base, path)
        assert status == 200
        assert 'WWW-Authenticate' not in headers
        assert b'legacy-test-password' not in body and b'test-server-key' not in body
    monkeypatch.setattr(demo, 'run_live', lambda payload, directory: {'status': 'mock-completed'})
    status, headers, body = request(base, '/api/test', headers={'Content-Type': 'application/json'}, payload={})
    assert status == 200 and json.loads(body)['status'] == 'mock-completed'
    assert 'WWW-Authenticate' not in headers


def test_render_and_custom_origins_work_without_allowing_cross_origin_posts(server_factory):
    base = server_factory(RENDER_EXTERNAL_HOSTNAME='adaptgym.onrender.com',
                          ADAPTGYM_PUBLIC_URL='https://demo.example.com',
                          ADAPTGYM_DEMO_PASSWORD='demo-test-password')
    for host in ('adaptgym.onrender.com', 'demo.example.com'):
        headers = {'Host': host, 'Origin': 'https://' + host, 'Content-Type': 'application/json'}
        status, _, body = request(base, '/api/plan', headers=headers, payload={})
        assert status == 200 and len(json.loads(body)['exchanges']) == 3
    for host, origin in [('attacker.example', 'https://attacker.example'),
                         ('adaptgym.onrender.com', 'https://attacker.example'),
                         ('adaptgym.onrender.com', 'https://demo.example.com')]:
        headers = {'Host': host, 'Origin': origin, 'Content-Type': 'application/json'}
        assert request(base, '/api/plan', headers=headers, payload={})[0] == 403


def test_public_bind_with_api_key_does_not_require_password(monkeypatch, server_factory):
    monkeypatch.setenv('OPENAI_API_KEY', 'test-api-credential')
    server = demo.create_server(port=0, host='0.0.0.0')
    server.server_close()


@pytest.mark.parametrize('value', ['https://user:password@example.com', 'https://example.com/path',
                                  'https://example.com?x=1', 'https://example.com#fragment', 'ftp://example.com'])
def test_invalid_public_origins_fail_before_serving(monkeypatch, server_factory, value):
    monkeypatch.setenv('ADAPTGYM_PUBLIC_URL', value)
    with pytest.raises(ValueError, match='public URL'):
        demo.create_server(port=0)


def test_judge_tab_receives_the_actual_backend_prompts(server_factory):
    base = server_factory()
    status, _, body = request(base, '/api/fixtures')
    assert status == 200
    prompts = json.loads(body)['prompts']
    vertical, _, _ = demo.load_fixtures()
    assert prompts['conversation_generator'] == demo.prompts_for(vertical)['conversation_generator']
    assert prompts['verifier'] == demo.prompts_for(vertical)['verifier']
    assert 'search_request' in prompts['verifier']
    assert b'LLM-Judge' in request(base, '/lab')[2]


def test_vertical_selection_routes_data_plans_and_saved_runs_independently(server_factory, monkeypatch):
    base = server_factory()
    registry = json.loads(request(base, '/api/verticals')[2])
    assert registry['default_vertical'] == 'footwear'
    assert {v['label'] for v in registry['verticals']} == {'Fashion/shoes', 'Job search'}
    before = json.loads(request(base, '/api/last-test?vertical=footwear')[2])
    for vertical, count in [('footwear', 30), ('job_search', 36)]:
        status, _, body = request(base, '/api/fixtures?vertical=' + vertical)
        data = json.loads(body)
        assert status == 200 and data['domain_schema']['x-adaptgym']['vertical_id'] == vertical
        assert len(data['catalog']['products']) == count
        assert len(data['scenario_guide']) == len(demo.CASE_MODES)
        assert all(entry['example'] for entry in data['scenario_guide'])
        status, _, body = request(base, '/api/plan', headers={'Content-Type': 'application/json'},
                                 payload={'vertical': vertical, 'case': 'no_match_relax'})
        assert status == 200 and json.loads(body)['vertical_id'] == vertical
    monkeypatch.setattr(demo, 'run_live', lambda payload, directory: {'directory': directory.name})
    status, _, body = request(base, '/api/test', headers={'Content-Type': 'application/json'},
                             payload={'vertical': 'job_search'})
    assert status == 200 and json.loads(body) == {'directory': 'job_search'}
    assert json.loads(request(base, '/api/last-test?vertical=job_search')[2]) == {'directory': 'job_search'}
    assert json.loads(request(base, '/api/last-test?vertical=footwear')[2]) == before
    assert b'id="vertical"' in request(base, '/lab')[2]


def test_vertical_selection_rejects_unknown_ids_and_paths(server_factory):
    base = server_factory()
    for value in ('missing', '../../etc', ''):
        assert request(base, '/api/fixtures?vertical=' + value)[0] == (200 if value == '' else 400)
        assert request(base, '/api/plan', headers={'Content-Type': 'application/json'},
                       payload={'vertical': value})[0] == 400


def test_busy_run_reports_status_across_verticals_and_releases_after_completion(server_factory, monkeypatch):
    base = server_factory()
    entered, finish = threading.Event(), threading.Event()
    outcome = []
    calls = []
    def live(payload, directory):
        calls.append(directory.name)
        entered.set()
        assert finish.wait(4)
        return {'status': 'completed'}
    monkeypatch.setattr(demo, 'run_live', live)
    headers = {'Content-Type': 'application/json'}
    thread = threading.Thread(target=lambda: outcome.append(request(base, '/api/test', headers=headers,
                                                                    payload={'vertical': 'footwear'})))
    thread.start()
    try:
        assert entered.wait(2)
        status = json.loads(request(base, '/api/run-status')[2])
        assert status['running'] and status['vertical'] == 'footwear'
        assert status['id'] and status['elapsed_seconds'] >= 0
        code, _, body = request(base, '/api/test', headers=headers, payload={'vertical': 'job_search'})
        assert code == 409 and json.loads(body)['run']['id'] == status['id']
        assert calls == ['debug_shopping']
    finally:
        finish.set()
        thread.join(timeout=3)
    assert outcome[0][0] == 200
    status = json.loads(request(base, '/api/run-status')[2])
    assert not status['running'] and status['outcome'] == 'completed'
    assert request(base, '/api/test', headers=headers, payload={'vertical': 'job_search'})[0] == 200


def test_failed_run_releases_lock_and_does_not_expose_exception_details(server_factory, monkeypatch):
    base = server_factory()
    def fail(payload, directory):
        raise RuntimeError('private upstream details')
    monkeypatch.setattr(demo, 'run_live', fail)
    headers = {'Content-Type': 'application/json'}
    for _ in range(2):
        code, _, body = request(base, '/api/test', headers=headers, payload={})
        assert code == 500 and b'private upstream details' not in body
        status = json.loads(request(base, '/api/run-status')[2])
        assert not status['running'] and status['outcome'] == 'failed'


@pytest.mark.parametrize('mode,subtype', [
    ('assistant_free', 'all'),
    ('assistant_reactive', 'non_context_dependent'),
    ('assistant_reactive', 'context_dependent'),
])
def test_group_sampling_respects_the_three_scenario_groups(mode, subtype):
    for seed in range(6):
        _, plan = demo.make_plan({'case': 'any', 'mode_filter': mode,
            'subtype_filter': subtype, 'seed': seed, 'turn_count': 5, 'finisher': 'ambiguous'})
        assert plan['mode'] == mode
        assert (plan['scenario_subtype'] == 'context_dependent') == (subtype == 'context_dependent')


def test_featured_run_preloads_without_api_calls_and_survives_server_restart(server_factory, monkeypatch):
    def no_api(*args, **kwargs):
        pytest.fail('Preloading must not call OpenAI')
    monkeypatch.setattr(demo, 'ResponsesClient', no_api)
    first = server_factory()
    status, _, body = request(first, '/api/last-test')
    assert status == 200
    saved = json.loads(body)
    assert saved['featured_example']['title']
    assert saved['result']['status'] == 'completed'
    checkpoints = saved['result']['checkpoints']
    assert len(checkpoints) == 3
    previous = None
    for checkpoint in checkpoints:
        assert checkpoint['verification']['accepted']
        assert checkpoint['intent_judgment']['accepted']
        assert checkpoint['score']['valid']
        assert checkpoint['previous_prediction'] == previous
        previous = checkpoint['prediction']
    assert checkpoints[-1]['prediction']['satisfied'] is True
    product = checkpoints[1]['prediction']['target_product']
    assert product
    assert saved['plan']['scenario_type'] == 'select_product'
    assert checkpoints[-1]['prediction']['target_product'] == product
    assert checkpoints[-1]['expected_outcome']['target_product'] == product
    monkeypatch.setattr(demo, 'run_live', lambda *args: {'result': {'status': 'user-run'}})
    assert request(first, '/api/test', headers={'Content-Type': 'application/json'}, payload={})[0] == 200
    assert json.loads(request(first, '/api/last-test')[2])['result']['status'] == 'user-run'
    second = server_factory()
    assert json.loads(request(second, '/api/last-test')[2]) == saved


def test_intro_lab_flows_and_allowlisted_figures_are_available(server_factory):
    import re
    base = server_factory()
    assert b'id="stateMode"' in request(base, '/')[2]
    assert b'Controlled conversations.' in request(base, '/about')[2]
    assert b'id="stateMode"' in request(base, '/lab')[2]
    assert b'id="vertical"' in request(base, '/?vertical=job_search')[2]
    status, _, body = request(base, '/flows')
    assert status == 200
    figures = re.findall(rb'<img[^>]+src="([^"]+)"', body)
    assert len(figures) == 4
    for figure in figures:
        status, headers, svg = request(base, figure.decode())
        assert status == 200 and headers['Content-Type'] == 'image/svg+xml'
        assert b'<svg' in svg
    for path in ('/assets/flows/../../README.md', '/assets/flows/unknown.svg'):
        assert request(base, path)[0] == 404


def test_unknown_live_state_mode_rejected_without_api_calls(server_factory, monkeypatch):
    base = server_factory()
    monkeypatch.setattr(demo, 'ResponsesClient', lambda *a, **k: pytest.fail('API client must not be created'))
    status, _, body = request(base, '/api/test', headers={'Content-Type': 'application/json'},
                              payload={'state_mode': 'unknown'})
    assert status == 400 and b'state mode' in body


def test_every_page_has_shared_top_navigation_and_current_page(server_factory):
    import re
    base = server_factory()
    for path in ('/', '/lab', '/about', '/flows'):
        status, _, body = request(base, path)
        assert status == 200
        navigation = re.search(rb'<nav[^>]+aria-label="Main navigation"[^>]*>(.*?)</nav>', body).group(1)
        for href in (b'/', b'/about', b'/flows', b'https://github.com/flavianv/adaptgym'):
            assert b'href="' + href + b'"' in navigation
        active_path = '/' if path == '/lab' else path
        assert b'href="' + active_path.encode() + b'" aria-current="page"' in navigation
