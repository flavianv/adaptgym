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
    assert prompts['conversation_generator'] == demo.SHOPPER_INSTRUCTIONS
    assert prompts['verifier'] == demo.VERIFIER_INSTRUCTIONS
    assert 'search_request' in prompts['verifier']
    assert b'LLM-Judge' in request(base)[2]


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
