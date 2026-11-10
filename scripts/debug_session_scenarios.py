#!/usr/bin/env python3
"""Generate procedural plans, test OpenAI shopper/extractor calls, or open a local UI."""
from __future__ import annotations

import argparse
import hashlib
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
                      if (name != 'mixed' or mode == 'all' and subtype == 'all')
                      and (mode == 'all' or CASE_MODES[name] == mode)
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


def comparison_index():
    path=ROOT/'examples/model_comparison/index.json'
    return json.loads(path.read_text()) if path.is_file() else {'cases':[],'models':[]}


def comparison_bundle(sample_id):
    entry=next((e for e in comparison_index()['cases'] if e['id']==sample_id),None)
    if not entry or Path(entry['file']).name!=entry['file']:return None
    path=ROOT/'examples/model_comparison'/entry['file']
    return json.loads(path.read_text()) if path.is_file() else None


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
    from adaptgym.live_chat import chat_turn
    sessions = {}
    chat_lock = threading.Lock()
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
            if path == '/api/comparison':
                sample_id=parse_qs(request_url.query).get('id',[''])[0]
                if not sample_id:return self.send_json(200,comparison_index())
                value=comparison_bundle(sample_id)
                return self.send_json(200,value) if value else self.send_json(404,{'error':'unknown comparison sample'})
            if path == '/reports/qwen4b-lora-report.md':
                report=ROOT/'examples/model_comparison/report.md'
                if not report.is_file():return self.send_json(404,{'error':'report unavailable'})
                body=report.read_bytes();self.send_response(200);self.send_header('Content-Type','text/markdown; charset=utf-8');self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body);return
            if path == '/api/run-status':
                return self.send_json(200, status_snapshot())
            pages = {'/': 'viewer.html', '/about': 'introduction.html',
                     '/about/': 'introduction.html', '/lab': 'viewer.html',
                     '/lab/': 'viewer.html', '/live': 'live_chat.html', '/live/': 'live_chat.html', '/flows': 'flows.html', '/flows/': 'flows.html'}
            figures = {'adaptgym-evaluation-flow.svg', 'adaptgym-sft-training-flow.svg',
                       'adaptgym-rlvr-extraction-training-flow.svg',
                       'adaptgym-rlvr-end-to-end-training-flow.svg'}
            if path in pages or path in ('/favicon.svg','/assets/site-header.css','/assets/precomputed-player.js') or path.startswith('/assets/flows/'):
                if path in pages:
                    page = 'viewer.html' if path == '/' and 'vertical' in parse_qs(request_url.query) else pages[path]
                    body = (ROOT / 'src/adaptgym' / page).read_bytes()
                    content_type = 'text/html; charset=utf-8'
                elif path == '/favicon.svg':
                    body=(ROOT/'src/adaptgym/favicon.svg').read_bytes()
                    content_type='image/svg+xml'
                elif path == '/assets/precomputed-player.js':
                    body=(ROOT/'src/adaptgym/precomputed_player.js').read_bytes()
                    content_type='text/javascript; charset=utf-8'
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
            elif path in ('/api/samples', '/api/sample', '/api/replay-results'):
                sample_root = ROOT / 'runs/inspection-samples'
                manifest = sample_root / 'index.json'
                entries = json.loads(manifest.read_text()) if manifest.is_file() else []
                entries = [e for e in entries if Path(e['file']).name == e['file']
                           and (sample_root / e['file']).is_file()]
                if path == '/api/samples':
                    selected=[dict(e,evaluation_available=False) for e in entries if e['vertical']==key]
                    selected += [dict(e,evaluation_available=True) for e in comparison_index()['cases'] if e['vertical']==key]
                    base=(ROOT/'runs/inspection-evaluation').resolve();pointer=base/'latest.json'
                    if pointer.is_file():
                        run=(base/json.loads(pointer.read_text())['directory']).resolve()
                        if run.is_relative_to(base):
                            for entry in selected:
                                result_file=run/(entry['id']+'.json')
                                if result_file.is_file():
                                    result=json.loads(result_file.read_text())
                                    entry['evaluation_available']=bool(result.get('completed') and result.get('sample_sha256')==hashlib.sha256((sample_root/entry['file']).read_bytes()).hexdigest())
                    return self.send_json(200,selected)
                sample_id = parse_qs(request_url.query).get('id', [''])[0]
                comparison=comparison_bundle(sample_id)
                if comparison and comparison['vertical']==key:
                    if path=='/api/sample':return self.send_json(200,comparison['sample'])
                    model=parse_qs(request_url.query).get('model',['luna'])[0]
                    mode=parse_qs(request_url.query).get('state_mode',['rollout'])[0]
                    result=comparison['models'].get(model,{}).get(mode)
                    return self.send_json(200,result) if result else self.send_json(404,{'error':'unknown saved model or state mode'})
                entry = next((e for e in entries if e['id'] == sample_id and e['vertical'] == key), None)
                if entry is None:
                    return self.send_json(404, {'error': 'unknown sample'})
                if path == '/api/replay-results':
                    base = (ROOT/'runs/inspection-evaluation').resolve()
                    pointer = base/'latest.json'
                    if not pointer.is_file():
                        return self.send_json(404, {'error':'No saved evaluation yet'})
                    run_root = (base/json.loads(pointer.read_text())['directory']).resolve()
                    if not run_root.is_relative_to(base):
                        return self.send_json(400, {'error':'Invalid evaluation directory'})
                    result_file = run_root/(entry['id']+'.json')
                    if not result_file.is_file():
                        return self.send_json(404, {'error':'This sample has not been evaluated yet'})
                    result = json.loads(result_file.read_text())
                    if not result.get('completed'):
                        return self.send_json(409, {'error':'This sample is still being evaluated'})
                    if result['sample_sha256'] != hashlib.sha256((sample_root/entry['file']).read_bytes()).hexdigest():
                        return self.send_json(409, {'error':'Saved results belong to a different version of this sample'})
                    return self.send_json(200,result)
                return self.send_json(200, json.loads((sample_root / entry['file']).read_text()))
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
                if self.path == '/api/replay':
                    from adaptgym.live_chat import replay_turn, BASELINE_EXTRACTOR_PROMPT
                    sample_id = payload.get('sample_id')
                    sample_root = ROOT / 'runs/inspection-samples'
                    manifest = sample_root / 'index.json'
                    entries = json.loads(manifest.read_text()) if manifest.is_file() else []
                    entry = next((e for e in entries if e['id']==sample_id and e['vertical']==key), None)
                    comparison=comparison_bundle(sample_id)
                    if comparison and comparison['vertical']!=key:comparison=None
                    if not comparison and (not entry or Path(entry['file']).name != entry['file'] or not (sample_root/entry['file']).is_file()):
                        raise ValueError('unknown saved sample')
                    if not chat_lock.acquire(blocking=False):
                        return self.send_json(409, {'error': 'Another extractor call is running. Try again shortly.'})
                    try:
                        now = time.monotonic()
                        for expired in [sid for sid,v in sessions.items() if now-v['updated']>3600]:
                            del sessions[expired]
                        sid = payload.get('session_id')
                        if sid is not None and (not isinstance(sid,str) or len(sid)!=32):
                            raise ValueError('invalid replay session')
                        if sid:
                            session = sessions.get(sid)
                            if not session or session.get('sample_id')!=sample_id:
                                raise ValueError('replay expired or changed; reload the sample')
                        else:
                            if len(sessions)>=100:
                                return self.send_json(429, {'error':'Chat capacity reached'})
                            sid = uuid.uuid4().hex
                            session = {'sample_id':sample_id, 'profile':None, 'index':0,
                                       'attempts':0, 'valid':0, 'updated':now, 'vertical':key, 'mode':'replay'}
                            sessions[sid] = session
                        sample = comparison['sample'] if comparison else json.loads((sample_root/entry['file']).read_text())
                        if session['index'] >= len(sample['plan']['exchanges']):
                            raise ValueError('replay finished; reload to start again')
                        if session['attempts']>=20:
                            return self.send_json(429, {'error':'Replay attempt limit reached; reload to start again'})
                        session['attempts'] += 1
                        session['updated'] = now
                        client = ResponsesClient('gpt-6-luna',max_calls=1,max_output_tokens=4000)
                        value = replay_turn(sample,session['index'],session['profile'],client)
                        session['index'] += 1
                        if value['score']['valid']:
                            session['profile'] = value['profile']
                            session['valid'] += 1
                        value.update(session_id=sid,model='gpt-6-luna',extractor_prompt=BASELINE_EXTRACTOR_PROMPT)
                        value['metrics'].update(attempts=session['attempts'],valid_outputs=session['valid'])
                        return self.send_json(200,value)
                    finally:
                        chat_lock.release()
                if self.path == '/api/chat':
                    text = payload.get('message', '')
                    if not isinstance(text, str) or not text.strip() or len(text) > 2000:
                        raise ValueError('send a message of 1–2000 characters')
                    if not chat_lock.acquire(blocking=False):
                        return self.send_json(409, {'error': 'Another chat turn is running. Try again shortly.'})
                    sid = None
                    session = None
                    try:
                        now = time.monotonic()
                        for expired in [sid for sid, v in sessions.items() if now-v['updated']>3600]:
                            del sessions[expired]
                        sid = payload.get('session_id')
                        if sid is not None and (not isinstance(sid, str) or len(sid) != 32):
                            raise ValueError('invalid chat session; start a new chat')
                        if sid:
                            session = sessions.get(sid)
                            if not session or session['vertical'] != key or session.get('mode')=='replay':
                                raise ValueError('session expired or catalog changed; start a new chat')
                        else:
                            if len(sessions) >= 100:
                                return self.send_json(429, {'error': 'Chat capacity reached. Try again later.'})
                            sid = uuid.uuid4().hex
                            session = {'vertical': key, 'messages': [], 'profile': None,
                                       'attempts': 0, 'valid': 0, 'updated': now}
                            sessions[sid] = session
                        if session['attempts'] >= 20:
                            return self.send_json(429, {'error': '20-turn limit reached. Start a new chat.'})
                        session['attempts'] += 1
                        session['updated'] = now
                        bundle = VerticalBundle.load(bundle_directory)
                        model = 'gpt-6-luna'
                        client = ResponsesClient(model, max_calls=2, max_output_tokens=4000)
                        value = chat_turn(bundle, build_catalog(bundle.template, bundle.vertical),
                                          session['messages'], session['profile'], text.strip(), client, client)
                        session['messages'] = value['messages']
                        if value['metrics']['output_valid']:
                            session['profile'] = value['profile']
                            session['valid'] += 1
                        from adaptgym.live_chat import BASELINE_EXTRACTOR_PROMPT
                        value.update(session_id=sid, model=model, prompt_version='simple-baseline-v1',
                                     extractor_prompt=BASELINE_EXTRACTOR_PROMPT)
                        value['metrics'].update(valid_outputs=session['valid'], attempts=session['attempts'])
                        return self.send_json(200, value)
                    except Exception:
                        return self.send_json(502, {'error': 'The model turn failed validation or could not complete. Your previous profile is preserved.',
                                                   'session_id': sid if session else None, 'attempts': session['attempts'] if session else 0, 'valid_outputs': session['valid'] if session else 0})
                    finally:
                        chat_lock.release()
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
