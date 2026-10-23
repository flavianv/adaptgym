"""OpenAI Responses API shopper generation and fixed-transcript extractor tests.

No credentials or HTTP headers are serialized. Calls are explicitly bounded.
"""
from __future__ import annotations

import copy
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from adaptgym.intent_state import evidence_mode, validate_evidence
from adaptgym.session_plans import validate_plan
from adaptgym.verticals import VerticalSchema
from adaptgym.search_requests import build_search_request, same_search_constraints, describe_intent, expected_outcome


def object_schema(properties: dict) -> dict:
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


SHOPPER_SCHEMA = object_schema({'text': {'type': 'string'}})
EVIDENCE_SCHEMA = object_schema({
    'source_type': {'type': 'string', 'enum': ['conversation']},
    'mode': {'type': 'string', 'enum': ['explicit', 'implicit']},
    'message_ids': {'type': 'array', 'items': {'type': 'string'}},
    'justification': {'type': 'string'},
})
CLAIM_SCHEMA = object_schema({
    'attribute': {'type': 'string'}, 'operator': {'type': 'string', 'enum': ['eq', 'in', 'lt', 'lte', 'gt', 'gte']},
    'value': {'anyOf': [{'type': 'string'}, {'type': 'number'}, {'type': 'boolean'},
                         {'type': 'array', 'items': {'anyOf': [{'type': 'string'}, {'type': 'number'}, {'type': 'boolean'}]}}]},
    'scope': {'type': 'string', 'enum': ['durable', 'task', 'unknown']},
    'task_id': {'type': ['string', 'null']}, 'strength': {'type': 'string', 'enum': ['soft', 'hard', 'unknown']},
    'evidence': {'type': 'array', 'items': EVIDENCE_SCHEMA},
})
SEARCH_ASSIGNMENT_SCHEMA = object_schema({key: CLAIM_SCHEMA['properties'][key]
                                          for key in ('attribute', 'operator', 'value')})
SEARCH_REQUEST_SCHEMA = {'anyOf': [{'type': 'null'}, object_schema({
    'query': {'type': 'string', 'minLength': 3, 'pattern': '[A-Za-z0-9]'},
    'filters': {'type': 'array', 'items': SEARCH_ASSIGNMENT_SCHEMA},
    'preferences': {'type': 'array', 'items': SEARCH_ASSIGNMENT_SCHEMA},
})]}
EXTRACT_SCHEMA = object_schema({
    'intent': {'type': 'string', 'minLength': 3},
    'satisfied': {'enum': [True, False, 'unk']},
    'target_product': {'type': ['string', 'null']},
    'active_task_id': {'type': ['string', 'null']},
    'claims': {'type': 'array', 'items': CLAIM_SCHEMA},
    'search_request': SEARCH_REQUEST_SCHEMA,
})


class ResponsesClient:
    def __init__(self, model: str, *, api_key: str | None = None, max_calls: int = 20,
                 max_output_tokens: int = 2400, timeout: float = 120):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError('OPENAI_API_KEY is required for live testing; dry-run planning needs no key')
        if not model or max_calls < 1 or max_output_tokens < 1:
            raise ValueError('model, positive call limit, and positive output limit are required')
        self.model, self.max_calls = model, max_calls
        self.max_output_tokens, self.timeout = max_output_tokens, timeout
        self.calls = 0
        self.traces = []

    def json(self, instructions: str, payload: dict, schema: dict, *, purpose: str) -> dict:
        if self.calls >= self.max_calls:
            raise RuntimeError('API call budget exhausted')
        body = {'model': self.model, 'store': False, 'instructions': instructions,
                'input': json.dumps(payload, ensure_ascii=False),
                'max_output_tokens': self.max_output_tokens,
                'text': {'format': {'type': 'json_schema', 'name': purpose, 'strict': True, 'schema': schema}}}
        request = urllib.request.Request('https://api.openai.com/v1/responses',
            data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {self.api_key}'}, method='POST')
        self.calls += 1
        started = time.monotonic()
        trace = {'purpose': purpose, 'model': self.model, 'status': 'failed'}
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                result = json.loads(response.read().decode())
            trace.update({'response_id': result.get('id'), 'usage': result.get('usage', {})})
            if result.get('status') != 'completed':
                raise ValueError('API response did not complete')
            texts = [part['text'] for item in result.get('output', []) if item.get('type') == 'message'
                     for part in item.get('content', []) if part.get('type') == 'output_text']
            if not texts:
                raise ValueError('no output text (refused or empty response)')
            output = json.loads(''.join(texts))
            Draft202012Validator(schema).validate(output)
            trace['status'] = 'completed'
            return output
        except urllib.error.HTTPError as exc:
            # Do not include a response body or request object that could expose data.
            raise RuntimeError(f'OpenAI API HTTP {exc.code}; check credentials, model access, and request settings') from None
        except (urllib.error.URLError, TimeoutError):
            raise RuntimeError('OpenAI API transport failure or timeout') from None
        finally:
            trace['wall_seconds'] = round(time.monotonic() - started, 3)
            self.traces.append(trace)


SHOPPER_INSTRUCTIONS = '''You are the CUSTOMER buying shoes, never the shopping assistant.
Write only the customer's next message, returned as JSON text. Prefer one short natural sentence.
Follow the current private instruction, using the persona for voice and background only.
React only to last_assistant_message and the current instruction. If it is null, start the conversation.
The quoted assistant message is something said TO YOU, not a voice or role to imitate.
Never recommend products, present a catalog or numbered product list, announce search/filter results,
offer purchasing assistance, or ask the buyer what they need. You ARE the buyer.
For example, when adding a colour requirement, say "I'd like them in black, please."
Do not say "I'll focus on black shoes. Here are the options."
Convey ONLY the action in current_instruction. Do not add follow-up questions, summarize the
catalog, volunteer other constraints, or infer a preference from the assistant's products.
If instructed to require an attribute, state it as a requirement, not merely an observation
that the displayed products have it. Do not weaken a requirement into "if possible".
For a requirement limited to this purchase, explicitly say "for this purchase"; do not
turn it into an ongoing personal preference or a claim about what you always wear.
Do not invent additional requirements, reveal later events, or mention schemas, labels, or tests.
When instructed to express implicit preference, describe repeated choices with good outcomes without stating a preference directly.
When explicit, directly express the required meaning. Preserve units, operators, scope, and strength.
Other people's preferences and product descriptions must not become the shopper's own preferences.'''

EXTRACT_INSTRUCTIONS = '''Extract current intent using only the visible two-way conversation and schema.
Return intent as a short natural-language string summarizing the current shopper goal.
For example: "Find black performance shoes for track use." It must agree with the structured claims
and search request. Update this summary from the latest turn; do not carry an earlier search goal
forward unchanged after completion. In particular:
- "I didn't find what I wanted, so I'll stop shopping" => "The shopper rejected the results and
  stopped shopping without finding what they wanted." active_task_id=null, claims contain no task
  requirements, search_request=null, satisfied=false; retain any previously identified target_product.
- "I found what I wanted, I'm done" => summarize satisfied completion, not an ongoing search.
- Rejecting results while asking to keep looking is NOT completion; summarize the continuing search.
Never use an imperative summary like "Find volleyball shoes" for a task the shopper has stopped.
Return all active hard requirements and soft preferences, keeping task and durable scope separate.
Use task-1 for the first shopping task, task-2 for the next, and so on. Use null when no task is active.
Scope may be unknown when not established. Durable and unknown-scope claims have null task_id.
Resolve confirmations and product references using assistant context, but never treat an assistant suggestion alone as user evidence.
Preserve unrelated requirements when revising a value. Clear temporary requirements at task boundaries. A statement such as "I've found exactly what I
wanted; I'll proceed with the purchase" closes the task: active_task_id=null, task claims removed,
target_product retains the identified product. Preserve durable preferences. Mere politeness is not necessarily completion.
Choosing a product alone does NOT close the shopping task. For "I'll take the second one", keep
active_task_id=task-1 and existing task claims with task_id=task-1, set target_product to the chosen
ID, search_request=null, and satisfied="unk". Only a later expressed completion clears active_task_id
and task-scoped claims. Never output active_task_id=null while retaining task-scoped claims.
Represent a product choice only as target_product; do not infer all its attributes as durable preferences.
Cite real message IDs in evidence; include the assistant question/list when necessary, and the user's support.
Return value as the actual typed scalar or membership array, never as JSON encoded inside a string. Use schema operators and units.
Also extract satisfied as true for expressed success, false for expressed failure, or "unk" when
not established (including silence). Do not infer satisfaction just from recommendations or selection.
target_product is the catalog ID the shopper explicitly selected or identified; null if unspecified.
Retain that product through task completion. Reset both
outcome fields when a new task begins. Never invent a product because the user says "found it".
The previous prediction is fallible: correct it using the conversation if necessary.
Also emit search_request as part of the intent: {query, filters, preferences}.
query must be a useful, nonempty natural-language product search phrase naming the product
category and known needs. Example: "performance indoor court shoes", then "volleyball indoor court
shoes" after the sport is known. Never output placeholders such as "/", "*", or schema field names alone. filters contain ALL active hard requirements;
preferences contain soft preferences, never hard filters. Each uses attribute/operator/typed value.
Use the current task's value when it overrides a durable preference on the same attribute. Unknown
scope or strength stays in claims for clarification, not in automatic search constraints.
Return null if there is no active task or a product has already been selected. Selection lookup is
not a new catalog search. Never put unsupported constraints into query text. Emit active claims only, not superseded values. No private plan or expected states are available.'''


def expected_signature(entries: list[dict]) -> set[str]:
    return {json.dumps({'attribute': e['assignment']['attribute'], 'operator': e['assignment']['operator'],
                        'value': e['assignment']['value'], 'scope': e['scope'], 'strength': e['strength'],
                        'task_id': e['task_id']}, sort_keys=True) for e in entries}


def score_extraction(prediction: dict, step: dict, vertical: VerticalSchema, messages: list[dict], catalog: dict) -> dict:
    """Deterministic values/scope scoring, plus evidence reference visibility.

    Does not claim that a reference's existence proves semantic support.
    """
    expected = expected_signature(step['planned_state']['entries'])
    try:
        Draft202012Validator(EXTRACT_SCHEMA).validate(prediction)
        entries = []
        for claim in prediction['claims']:
            assignment = vertical.assignment(claim['attribute'], claim['value'], claim['operator'])
            validate_evidence(claim['evidence'], messages)
            if claim['scope'] == 'task':
                if not claim['task_id'] or claim['task_id'] != prediction['active_task_id']:
                    raise ValueError('task scope does not match the active task')
            elif claim['task_id'] is not None:
                raise ValueError('non-task scope cannot claim a task ID')
            entries.append({'assignment': assignment, 'scope': claim['scope'], 'task_id': claim['task_id'],
                            'strength': claim['strength'], 'evidence': claim['evidence'],
                            'evidence_mode': evidence_mode(claim['evidence'])})
        keys = [(e['assignment']['attribute'], e['scope'], e['task_id']) for e in entries]
        if len(keys) != len(set(keys)):
            raise ValueError('duplicate claims')
        selected = prediction['target_product']
        shown = {p['id'] for p in catalog['products'] if any(p['id'] in m['text'] for m in messages if m['role'] == 'assistant')}
        if selected is not None and selected not in shown:
            raise ValueError('selection must reference a previously displayed product')
        request = prediction['search_request']
        if request is not None:
            for item in request['filters'] + request['preferences']:
                vertical.assignment(item['attribute'], item['value'], item['operator'])
        expected_intent = {
            'active_task_id': step['planned_state']['active_task_id'],
            'target_product': step['target_product'],
            'claims': [{**e['assignment'], 'scope': e['scope'], 'task_id': e['task_id'],
                        'strength': e['strength']} for e in step['planned_state']['entries']],
        }
        request_correct = same_search_constraints(request, build_search_request(expected_intent, vertical.vertical_id))
        request_consistent = same_search_constraints(request, build_search_request(prediction, vertical.vertical_id))
        actual = expected_signature(entries)
        correct = len(actual & expected)
        expected_modes = {(e['assignment']['attribute'], e['scope'], e['task_id']): e['evidence_mode'] for e in step['planned_state']['entries']}
        modes_correct = sum(e['evidence_mode'] == expected_modes.get((e['assignment']['attribute'], e['scope'], e['task_id'])) for e in entries)
        outcome_correct = all(prediction[key] == step.get(key, default) for key, default in (
            ('satisfied', True if step.get('finisher') == 'positive' else False if step.get('finisher') == 'negative' else 'unk'),
            ('target_product', step['target_product'])))
        task_correct = prediction['active_task_id'] == step['planned_state']['active_task_id']
        selection_correct = selected == step['target_product']
        return {'valid': True, 'exact_state': actual == expected and task_correct and selection_correct and request_correct and outcome_correct,
                'outcome_correct': outcome_correct, 'correct_claims': correct, 'predicted_claims': len(actual), 'expected_claims': len(expected),
                'evidence_mode_correct': modes_correct, 'active_task_correct': task_correct,
                'target_product_correct': selection_correct, 'search_constraints_correct': request_correct,
                'search_consistent_with_intent': request_consistent, 'query_text_evaluation': 'requires_semantic_judge',
                'semantic_evidence_support': 'not_independently_audited'}
    except (ValueError, TypeError, KeyError, StopIteration, ValidationError) as exc:
        # Return a bounded type-only diagnostic. API errors are recorded by the caller.
        return {'valid': False, 'exact_state': False, 'correct_claims': 0,
                'predicted_claims': 0, 'expected_claims': len(expected), 'error_type': type(exc).__name__,
                'validation_issue': str(exc)[:240] if isinstance(exc, ValueError) else None}



VERDICT_SCHEMA = {'type': 'string', 'enum': ['supported', 'contradicted', 'insufficient_evidence']}
CHECK_SCHEMA = object_schema({
    'verdict': VERDICT_SCHEMA,
    'message_ids': {'type': 'array', 'items': {'type': 'string'}},
    'reason': {'type': 'string'},
})
STATE_CHECK_SCHEMA = object_schema({
    'observed_value': {'type': ['string', 'null']},
    'certainty': {'type': 'string', 'enum': ['resolved', 'uncertain']},
    'message_ids': {'type': 'array', 'items': {'type': 'string'}},
    'reason': {'type': 'string'},
})
VERIFICATION_SCHEMA = object_schema({
    'claim_checks': {'type': 'array', 'items': object_schema({
        'claim_index': {'type': 'integer'}, **CHECK_SCHEMA['properties'],
    })},
    'active_task': object_schema({**STATE_CHECK_SCHEMA['properties'],
        'observed_value': {'type': ['string', 'null'], 'pattern': '^task-[1-9][0-9]*$'},
    }),
    'additional_intent': {'type': 'array', 'items': object_schema({
        'claim': CLAIM_SCHEMA,
        'description': {'type': 'string'},
        'message_ids': {'type': 'array', 'items': {'type': 'string'}},
    })},
    'satisfied': object_schema({**STATE_CHECK_SCHEMA['properties'],
        'observed_value': {'enum': [True, False, 'unk']}}),
    'target_product': STATE_CHECK_SCHEMA,
    'intent': CHECK_SCHEMA,
    'search_request': CHECK_SCHEMA,
    'summary': {'type': 'string'},
})

VERIFIER_INSTRUCTIONS = '''You independently audit candidate intent labels against a conversation prefix.
The candidate is a hypothesis, not ground truth. You have no private generation instructions.
Check EVERY indexed claim for its value, operator, scope, strength, and evidence mode.
Interpret normal shopper language using the supplied schema. Attribute names need not be spoken:
"black shoes" explicitly supports colour=black; "performance shoes" supports style=performance
when performance is a style enum; "for indoor courts" supports surface=indoor_court.
Do not reject a value merely because the user omitted the schema field name. Still reject weaker
or ambiguous wording when it does not establish the proposed value, operator, scope, or strength.
Use supported, contradicted, or insufficient_evidence; uncertainty must not pass.
Cite only visible message IDs. Supported claims require shopper evidence; assistant-only statements
never establish a shopper preference. Include the relevant assistant question or product list for
short confirmations, comparisons, and references such as "the second one".
Inspect the candidate evidence references: they must actually support the claim, including its
explicit/implicit classification. Implicit behavior must justify the inference; do not invent needs.
For active_task and target_product, infer observed_value from the dialogue independently of
candidate values, and report certainty resolved or uncertain. Code compares observed and candidate
values; do not emit a match verdict yourself. Null is a REAL value meaning no active task or no
selection, not uncertainty. No product chosen => observed selected product null, certainty resolved.
For active_task observed_value use exactly task-1, task-2, etc., or null. Never use "active".
Task IDs are internal chronological labels; users need not say these IDs aloud.
A shopping request starts a task. "I've found exactly what I wanted, I'll proceed with the purchase"
is completion: active task null, task-scoped requirements retired, target product retained if identified.
"Keep looking" or "Can you show more?" keeps the task active. Mere "thanks" may be ambiguous.
Completion retains durable preferences. Product choice alone does not create durable preferences.
For example: candidate selection null + no choice made = matching nulls, NOT a contradiction.
For additional_intent, return ONLY missing CURRENT attribute claims as structured claims with evidence.
Do not restate claims already in the candidate, selected products, or active tasks (those have their
own checks). A completed task's old requirements are not omissions. Explicitly stopping after
unsatisfactory results closes the task just as success does: active_task null, task claims retired,
search_request null. If the summary still says "Find ...", flag the SUMMARY as stale; do not flag
correctly cleared claims or the null search request merely because the summary is stale. Report operator/scope/strength
mismatches in the relevant claim check rather than duplicating the same fact as an omission.
Infer satisfied independently as true (expressed success), false (expressed failure), or "unk"
(no expressed outcome yet, ambiguous departure, or silence). Return certainty resolved for "unk"
when no outcome is expressed. Do not infer satisfaction from an assistant's upbeat framing.
Infer target_product from explicit shopper choices or references, retained through completion;
null means no identified product, not uncertainty. Reset on a new task. The shopper can be satisfied
without naming any product: satisfied=true, target_product=null is valid. Cite shopper evidence for
known outcomes and target products. Report these fields as observed_value and certainty.
Judge the intent STRING too: does it faithfully summarize the shopper goal and agree with the
structured attributes and search request? Return its check in intent.
Judge the WHOLE intent, not string equality. Semantically equivalent query wording is allowed.
Audit search_request too: its query must name an appropriate product category and known needs.
A placeholder such as "/", "*", or meaningless text is WRONG even if filters are correct. Check: are hard constraints in filters and soft ones
in preferences, are there missing or invented constraints? Unknown-scope/strength claims must not
silently become search constraints. Task-specific values override durable values of the same attribute.
No active task or an already-selected product => search_request null. Null is supported when no
new search is needed. Include a search_request verdict with evidence and reasons.
Do not infer success from a silent departure. Assess this prefix only, without future information.
Return one claim_check per candidate claim, using its zero-based claim_index. Return JSON only.'''


def verify_intent_labels(client: Any, plan: dict, step: dict, messages: list[dict], *, candidate: dict | None = None) -> dict:
    # Separate audit without private scenario or extractor output.
    target = {
        'active_task_id': step['planned_state']['active_task_id'],
        'target_product': step['target_product'],
        'claims': [{'attribute': e['assignment']['attribute'], 'operator': e['assignment']['operator'],
                    'value': e['assignment']['value'], 'scope': e['scope'], 'task_id': e['task_id'],
                    'strength': e['strength'], 'evidence': e['evidence']}
                   for e in step['planned_state']['entries']],
    }
    target.update(expected_outcome(plan, step))
    target['search_request'] = build_search_request(target, plan['vertical_id'])
    target['intent'] = describe_intent(target, plan['vertical_id'])
    if candidate is not None:
        target = copy.deepcopy(candidate)
    report = None
    try:
        report = client.json(VERIFIER_INSTRUCTIONS, copy.deepcopy({
            'vertical_schema': plan['vertical_schema'], 'catalog': plan['catalog'],
            'visible_conversation': messages, 'candidate_intent': target,
        }), VERIFICATION_SCHEMA, purpose='intent_label_verification')
        Draft202012Validator(VERIFICATION_SCHEMA).validate(report)
        indices = [c['claim_index'] for c in report['claim_checks']]
        if sorted(indices) != list(range(len(target['claims']))):
            raise ValueError('verifier must check each claim exactly once')
        visible = {m['id']: m for m in messages}
        for field, expected in [('active_task', target['active_task_id']),
                                ('satisfied', target['satisfied']), ('target_product', target['target_product'])]:
            check = report[field]
            check['expected_value'] = expected
            check['verdict'] = ('insufficient_evidence' if check['certainty'] == 'uncertain'
                                else 'supported' if check['observed_value'] == expected else 'contradicted')
        # Structured omissions permit exact duplicate detection without trusting free-form prose.
        candidate_signatures = {json.dumps({k: c[k] for k in
            ('attribute', 'operator', 'value', 'scope', 'strength', 'task_id')}, sort_keys=True)
            for c in target['claims']}
        vertical = VerticalSchema(plan['vertical_schema'])
        omissions = []
        for item in report['additional_intent']:
            claim = item['claim']
            vertical.assignment(claim['attribute'], claim['value'], claim['operator'])
            validate_evidence(claim['evidence'], messages)
            signature = json.dumps({k: claim[k] for k in
                ('attribute', 'operator', 'value', 'scope', 'strength', 'task_id')}, sort_keys=True)
            if signature not in candidate_signatures:
                omissions.append(item)
        report['additional_intent'] = omissions
        checks = [*report['claim_checks'], report['active_task'], report['satisfied'], report['target_product'], report['intent'], report['search_request']]
        for item in [*checks, *report['additional_intent']]:
            if any(mid not in visible for mid in item['message_ids']):
                raise ValueError('verification must cite visible evidence')
        required_refs = [c for c in report['claim_checks'] if c['verdict'] == 'supported'] + report['additional_intent']
        if report['active_task']['observed_value'] is not None:
            required_refs.append(report['active_task'])
        if report['satisfied']['observed_value'] != 'unk':
            required_refs.append(report['satisfied'])
        if report['target_product']['observed_value'] is not None:
            required_refs.append(report['target_product'])
        if target['search_request'] is not None and report['search_request']['verdict'] == 'supported':
            required_refs.append(report['search_request'])
        if any(not item['message_ids'] for item in required_refs):
            raise ValueError('positive intent labels require evidence citations')
        for claim in report['claim_checks']:
            if claim['verdict'] == 'supported' and not any(
                    visible[mid]['role'] == 'user' for mid in claim['message_ids']):
                raise ValueError('supported claim needs shopper evidence')
        accepted = all(c['verdict'] == 'supported' for c in checks) and not report['additional_intent']
        return {'status': 'completed', 'accepted': accepted,
                'verdict': 'supported' if accepted else 'needs_review',
                'method': 'independent_model_pass', **report}
    except Exception as exc:
        return {'status': 'failed', 'accepted': False, 'verdict': 'unverified',
                'error_type': type(exc).__name__,
                'validation_issue': str(exc)[:240] if isinstance(exc, (ValueError, ValidationError)) else None,
                'report': report}


def render_and_test(plan: dict, persona: dict, shopper: Any, extractor: Any, verifier: Any | None = None) -> dict:
    """Generate shopper turns, append fixed assistant text, and test prefix extraction.

    This is a diagnostic smoke test, not independent certification of labels.
    Planned targets are visible only to the generator, scorer, and optional label auditor.
    """
    validate_plan(plan)
    vertical = VerticalSchema(plan['vertical_schema'])
    messages, checkpoints = [], []
    previous = None
    for step in plan['exchanges']:
        try:
            reply = shopper.json(SHOPPER_INSTRUCTIONS, {
                'persona': persona['profile'],
                'last_assistant_message': copy.deepcopy(messages[-1]) if messages else None,
                'current_instruction': step['shopper_instruction'],
            }, SHOPPER_SCHEMA, purpose='shopper_turn')
            if not reply['text'].strip():
                raise ValueError('empty shopper message')
        except Exception as exc:
            return {'plan_id': plan['id'], 'status': 'generation_failed', 'error_type': type(exc).__name__,
                    'conversation': messages, 'checkpoints': checkpoints}
        messages.append({'id': step['shopper_message_id'], 'role': 'user', 'text': reply['text']})
        target = {**step, **expected_outcome(plan, step)}
        previous_input = copy.deepcopy(previous)
        try:
            prediction = extractor.json(EXTRACT_INSTRUCTIONS, {
                'vertical_schema': plan['vertical_schema'], 'catalog': plan['catalog'],
                'visible_conversation': messages, 'previous_prediction': previous_input,
            }, EXTRACT_SCHEMA, purpose='intent_extraction')
            score = score_extraction(prediction, target, vertical, messages, plan['catalog'])
            if score['valid']:
                previous = prediction
        except Exception as exc:
            prediction = None
            score = {'valid': False, 'exact_state': False, 'correct_claims': 0,
                     'expected_claims': len(step['planned_state']['entries']), 'error_type': type(exc).__name__}
        verification = (verify_intent_labels(verifier, plan, target, messages) if verifier is not None
                        else {'status': 'not_run', 'accepted': False, 'verdict': 'unverified'})
        if verifier is not None:
            score['label_evidence_support'] = ('model_audit_supported' if verification['accepted']
                                                  else 'model_audit_flagged' if verification['status'] == 'completed'
                                                  else 'unverified')
        judgment = (verify_intent_labels(verifier, plan, target, messages, candidate=prediction)
                    if verifier is not None and prediction is not None and score['valid']
                    else {'status': 'not_run', 'accepted': False, 'verdict': 'unverified'})
        if prediction is not None and not score['valid']:
            judgment = {'status': 'invalid', 'accepted': False, 'verdict': 'incorrect',
                        'summary': 'Invalid intent: ' + (score.get('validation_issue') or score.get('error_type', 'validation failed'))}
        score['semantic_evidence_support'] = ('model_audit_supported' if judgment['accepted'] else
                                               'model_audit_flagged' if judgment['status'] == 'completed'
                                               else 'unverified')
        score['query_text_evaluation'] = judgment.get('search_request', {}).get('verdict', 'unverified')
        score['intent_judgment'] = ('right_intent' if judgment['accepted'] else
                                    'incorrect_intent' if judgment['status'] == 'invalid' else
                                    'needs_review' if judgment['status'] == 'completed' else 'unverified')
        reference_intent = {
            **expected_outcome(plan, step),
            'active_task_id': target['planned_state']['active_task_id'],
            'target_product': target['target_product'],
            'claims': [{**e['assignment'], 'scope': e['scope'], 'task_id': e['task_id'],
                        'strength': e['strength']} for e in target['planned_state']['entries']],
        }
        checkpoints.append({'turn': step['exchange'], 'prediction': prediction, 'score': score,
                            'previous_prediction': previous_input,
                            'expected_outcome': expected_outcome(plan, step),
                            'expected_intent': describe_intent(reference_intent, plan['vertical_id']),
                            'expected_search_request': build_search_request(reference_intent, plan['vertical_id']),
                            'verification': verification, 'intent_judgment': judgment})
        messages.append({'id': step['assistant']['message_id'], 'role': 'assistant', 'text': step['assistant']['text']})
    return {'plan_id': plan['id'], 'status': 'completed', 'conversation': messages,
            'finisher': copy.deepcopy(plan['finisher']),
            'checkpoints': checkpoints, 'exact_checkpoints': sum(c['score']['exact_state'] for c in checkpoints),
            'total_checkpoints': len(checkpoints),
            'label_status': ('model_verified' if verifier is not None and
                             all(c['verification']['accepted'] for c in checkpoints)
                             else 'planned_targets_not_independently_verified'),
            'verified_checkpoints': sum(c['verification']['accepted'] for c in checkpoints),
            'interpretation': 'Disagreement may reflect either shopper-generation drift or extractor error; inspect traces.'}
