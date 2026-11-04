#!/usr/bin/env python3
"""Generate procedural plans, test OpenAI shopper/extractor calls, or open a local UI."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
import shlex
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from adaptgym.catalog import build_catalog, validate_catalog
from adaptgym.bundles import VerticalBundle
from adaptgym.scenario_guide import load_scenario_guide
from adaptgym.openai_session_test import ResponsesClient, render_and_test, prompts_for, STATE_MODES
from adaptgym.session_plans import CASE_MIN_TURNS, CASE_MODES, CASE_SUBTYPES, MIDDLE_CASES, generate_plan, validate_plan

FIXTURES = ROOT / 'examples/debug_shopping'


def load_env_file(path):
    """Explicit opt-in dotenv subset; never execute shell expansion or print values."""
    if path is None:
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.startswith('export '):
            line = line[7:]
        key, separator, value = line.partition('=')
        if separator and key.strip() in ('OPENAI_API_KEY', 'OPENAI_MODEL'):
            parts = shlex.split(value, comments=True)
            if len(parts) == 1:
                os.environ.setdefault(key.strip(), parts[0])


def load_fixtures(directory=FIXTURES):
    bundle = VerticalBundle.load(directory)
    return bundle.vertical, bundle.persona, bundle.template


def make_plan(payload, directory=FIXTURES):
    vertical, persona, template = load_fixtures(directory)
    overrides = payload.get('group_overrides', {})
    if not isinstance(overrides, dict):
        raise ValueError('group_overrides must be an object')
    catalog = build_catalog(template, vertical, overrides)
    if payload.get('include_distractors', False):
        pool = VerticalBundle.load(directory).distractors
        if not pool['products']:
            raise ValueError('this vertical has no distractor pool')
        catalog['products'].extend(pool['products'])
        catalog['catalog_id'] += '-with-distractor-pool'
        validate_catalog(catalog, vertical)
    case = payload.get('case', 'any')
    turn_count = payload.get('turn_count', 3 if case == 'any' else None)
    finisher = payload.get('finisher', 'positive' if case == 'any' else 'auto')
    if case == 'any':
        mode = payload.get('mode_filter', 'all')
        subtype = payload.get('subtype_filter', 'all')
        if mode not in ('all', 'assistant_free', 'assistant_reactive') or subtype not in ('all', 'context_dependent', 'non_context_dependent'):
            raise ValueError('unknown scenario filter')
        candidates = [name for name in MIDDLE_CASES
                      if (mode == 'all' or CASE_MODES[name] == mode)
                      and (subtype == 'all' or CASE_SUBTYPES[name] == subtype
                           or (subtype == 'non_context_dependent' and CASE_SUBTYPES[name] is None))
                      and (turn_count is None or CASE_MIN_TURNS[name] + int(finisher in ('positive', 'negative')) <= turn_count)]
        if not candidates:
            raise ValueError('no middle scenario fits these filters and turn budget')
        case = random.Random(int(payload.get('seed', 7))).choice(candidates)
    plan = generate_plan(vertical, catalog, case,
                         seed=int(payload.get('seed', 7)), expression=payload.get('expression', 'explicit'),
                         turn_count=turn_count, finisher=finisher)
    return persona, plan


def run_live(payload, directory=FIXTURES):
    state_mode = payload.get('state_mode', 'accumulated')
    if state_mode not in STATE_MODES:
        raise ValueError('unknown evaluation state mode')
    persona, plan = make_plan(payload, directory)
    model = str(payload.get('model') or os.environ.get('OPENAI_MODEL', '')).strip()
    if not model:
        raise ValueError('choose an OpenAI model before running a live test')
    # Explicit per-run request cap; automatic HTTP retries are not performed.
    client = ResponsesClient(model, max_calls=4 * len(plan['exchanges']), max_output_tokens=2400)
    result = render_and_test(plan, persona, client, client, verifier=client, state_mode=state_mode)
    return {'plan': plan, 'result': result, 'api_traces': client.traces,
            'model': model, 'prompt_version': 'modular-vertical-v1',
            'same_model_roles': True}


def create_server(port=8768, directory=FIXTURES, *, host='127.0.0.1'):
    """Configure the local or hosted demo without starting its event loop."""
    directory = Path(directory).resolve()
    default_bundle = VerticalBundle.load(directory)
    default_vertical = default_bundle.vertical.vertical_id
    directories = {default_vertical: directory}
    labels = {default_vertical: default_bundle.vertical.simulation['label']}
    for candidate in sorted(directory.parent.iterdir()):
        if candidate.resolve() == directory or not (candidate / 'simulation.json').is_file():
            continue
        bundle = VerticalBundle.load(candidate)
        key = bundle.vertical.vertical_id
        if key in directories:
            raise ValueError(f'duplicate vertical ID: {key}')
        directories[key], labels[key] = candidate, bundle.vertical.simulation['label']
    def selected_directory(key):
        if not isinstance(key, str) or key not in directories:
            raise ValueError('unknown vertical')
        return directories[key]
    public_origins = set()
    render_hostname = os.environ.get('RENDER_EXTERNAL_HOSTNAME', '')
    configured_urls = [os.environ.get('ADAPTGYM_PUBLIC_URL', '')]
    if render_hostname:
        configured_urls.append('https://' + render_hostname)
    for value in filter(None, configured_urls):
        parsed = urlparse(value)
        if (parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username is not None
                or parsed.password is not None or parsed.path not in ('', '/') or parsed.query or parsed.fragment):
            raise ValueError('public URL must be an HTTP(S) origin without credentials, path, query, or fragment')
        public_origins.add(parsed.scheme + '://' + parsed.netloc)
    run_lock = threading.Lock()
    status_lock = threading.Lock()
    live_status = {'running': False, 'id': None, 'vertical': None, 'outcome': None}
    run_clock = {'started': None, 'finished': None}

    def status_snapshot():
        with status_lock:
            snapshot = dict(live_status)
            end = time.monotonic() if snapshot['running'] else run_clock['finished']
            snapshot['elapsed_seconds'] = round(end - run_clock['started']) if end is not None else 0
            return snapshot
    latest = {}
    for key, bundle_directory in directories.items():
        featured_path = bundle_directory / 'featured_run.json'
        featured = json.loads(featured_path.read_text()) if featured_path.exists() else None
        if featured is not None:
            validate_plan(featured['plan'])
        latest[key] = featured

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, status, value):
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            request_url = urlparse(self.path)
            path = request_url.path
            key = parse_qs(request_url.query).get('vertical', [default_vertical])[0]
            if path == '/healthz':
                return self.send_json(200, {'status': 'ok'})
            if path == '/api/run-status':
                return self.send_json(200, status_snapshot())
            pages = {'/': 'viewer.html', '/about': 'introduction.html',
                     '/about/': 'introduction.html', '/lab': 'viewer.html',
                     '/lab/': 'viewer.html', '/flows': 'flows.html', '/flows/': 'flows.html'}
            figures = {'adaptgym-evaluation-flow.svg', 'adaptgym-sft-training-flow.svg',
                       'adaptgym-rlvr-extraction-training-flow.svg',
                       'adaptgym-rlvr-end-to-end-training-flow.svg'}
            if path in pages or path == '/assets/site-header.css' or path.startswith('/assets/flows/'):
                if path in pages:
                    page = 'viewer.html' if path == '/' and 'vertical' in parse_qs(request_url.query) else pages[path]
                    body = (ROOT / 'src/adaptgym' / page).read_bytes()
                    content_type = 'text/html; charset=utf-8'
                elif path == '/assets/site-header.css':
                    body = (ROOT / 'src/adaptgym/site-header.css').read_bytes()
                    content_type = 'text/css; charset=utf-8'
                else:
                    name = path.removeprefix('/assets/flows/')
                    if name not in figures:
                        return self.send_json(404, {'error': 'not found'})
                    body = (ROOT / 'documentation/figures' / name).read_bytes()
                    content_type = 'image/svg+xml'
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(body)))
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
            elif path == '/api/verticals':
                self.send_json(200, {'default_vertical': default_vertical,
                    'verticals': [{'id': k, 'label': labels[k]} for k in sorted(directories, key=lambda k: labels[k])]})
            elif path in ('/api/last-test', '/api/fixtures'):
                try:
                    bundle_directory = selected_directory(key)
                except ValueError as exc:
                    return self.send_json(400, {'error': str(exc)})
                if path == '/api/last-test':
                    return self.send_json(200, latest[key])
                vertical, persona, template = load_fixtures(bundle_directory)
                self.send_json(200, {'persona': persona, 'domain_schema': vertical.schema,
                    'catalog_template': template, 'catalog': build_catalog(template, vertical),
                    'distractor_pool': VerticalBundle.load(bundle_directory).distractors,
                    'simulation': vertical.simulation,
                    'cases': CASE_MODES, 'case_min_turns': CASE_MIN_TURNS,
                    'case_subtypes': CASE_SUBTYPES,
                    'middle_cases': list(MIDDLE_CASES),
                    'scenario_guide': load_scenario_guide(bundle_directory),
                    'prompts': prompts_for(vertical),
                    'api_configured': bool(os.environ.get('OPENAI_API_KEY')),
                    'default_model': os.environ.get('OPENAI_MODEL', '')})
            else:
                self.send_json(404, {'error': 'not found'})

        def do_POST(self):
            origins = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'} | public_origins
            allowed = {urlparse(value).netloc for value in origins}
            request_host = self.headers.get('Host', '')
            origin = self.headers.get('Origin')
            if request_host not in allowed or (origin and (origin not in origins or urlparse(origin).netloc != request_host)):
                return self.send_json(403, {'error': 'same-origin requests only'})
            if self.headers.get('Content-Type', '').split(';')[0] != 'application/json':
                return self.send_json(415, {'error': 'JSON required'})
            try:
                length = int(self.headers.get('Content-Length', 0))
                if length < 1 or length > 100000:
                    raise ValueError('invalid request length')
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError('request must be an object')
                key = payload.get('vertical', default_vertical)
                bundle_directory = selected_directory(key)
                if self.path == '/api/plan':
                    _, plan = make_plan(payload, bundle_directory)
                    return self.send_json(200, plan)
                if self.path == '/api/test':
                    with status_lock:
                        acquired = run_lock.acquire(blocking=False)
                        if acquired:
                            live_status.update(running=True, id=uuid.uuid4().hex, vertical=key, outcome=None)
                            run_clock.update(started=time.monotonic(), finished=None)
                    if not acquired:
                        return self.send_json(409, {'error': 'A live run is already in progress.',
                                                   'run': status_snapshot()})
                    try:
                        result = run_live(payload, bundle_directory)
                        latest[key] = result
                        with status_lock:
                            live_status['outcome'] = 'completed'
                    except Exception:
                        with status_lock:
                            live_status['outcome'] = 'failed'
                        raise
                    finally:
                        with status_lock:
                            live_status['running'] = False
                            run_clock['finished'] = time.monotonic()
                            run_lock.release()
                    return self.send_json(200, result)
                self.send_json(404, {'error': 'not found'})
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json(400, {'error': str(exc)[:250]})
            except Exception:
                self.send_json(500, {'error': 'test failed; inspect local configuration'})

    return ThreadingHTTPServer((host, port), Handler)


def serve(port=8768, directory=FIXTURES, *, open_browser=True, host='127.0.0.1'):
    server = create_server(port, directory, host=host)
    url = f'http://{host}:{server.server_port}'
    print(f'Debug scenario webview: {url}', flush=True)
    if open_browser:
        threading.Timer(0.3, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vertical', '--fixtures', dest='fixtures', type=Path, default=FIXTURES,
                        help='directory containing a vertical bundle (schema, simulation, persona, catalog)')
    parser.add_argument('--case', choices=['all', *CASE_MODES], default='all')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--turns', type=int, choices=range(1, 8), help='shopper/assistant exchanges, preserving required core events')
    parser.add_argument('--finisher', choices=['auto', 'positive', 'negative', 'ambiguous'], default='auto')
    parser.add_argument('--expression', choices=['explicit', 'implicit', 'mixed'], default='explicit')
    parser.add_argument('--group-overrides', type=Path, help='JSON group name to attribute overrides')
    parser.add_argument('--include-distractors', action='store_true', help="add this vertical's optional distractor pool")
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/debug_session_plans.json')
    parser.add_argument('--env-file', type=Path, help='explicit dotenv file containing OPENAI_API_KEY / OPENAI_MODEL')
    parser.add_argument('--model', default='')
    parser.add_argument('--live', action='store_true', help='run paid OpenAI shopper and extraction calls')
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '8768')))
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    load_env_file(args.env_file)
    if args.serve:
        serve(args.port, args.fixtures, open_browser=not args.no_browser, host=args.host)
        return
    if args.live and args.case == 'all':
        parser.error('live tests require one explicit --case to bound API usage')
    payload = {'seed': args.seed, 'expression': args.expression, 'model': args.model,
               'turn_count': args.turns,
               'finisher': args.finisher,
               'include_distractors': args.include_distractors,
               'group_overrides': json.loads(args.group_overrides.read_text()) if args.group_overrides else {}}
    cases = CASE_MODES if args.case == 'all' else [args.case]
    outputs = []
    for case in cases:
        current = {**payload, 'case': case}
        outputs.append(run_live(current, args.fixtures) if args.live else make_plan(current, args.fixtures)[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(outputs, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'output': str(args.output), 'plans': len(outputs), 'live': args.live}))


if __name__ == '__main__':
    main()
