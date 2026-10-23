"""Seeded concrete session plans with programmed assistant messages.

Plans are private generation instructions, not accepted dialogue answer keys.
"""
from __future__ import annotations

import copy
import random
from collections import Counter
from typing import Any

from adaptgym.catalog import classify_products, matching_ids, matches, validate_catalog
from adaptgym.intent_state import IntentState, evidence
from adaptgym.current_need import generate_current_need
from adaptgym.verticals import VerticalSchema

FREE_CASES = (
    'initial_need', 'add_requirement', 'clarify_requirement', 'reinforce_preference',
    'revise_value', 'retract_requirement', 'restore_value', 'finish_task',
    'start_another_task', 'irrelevant_information',
)
REACTIVE_CASES = (
    'clarification_question', 'select_product', 'compare_products',
    'soft_preference_missed', 'hard_constraint_violated',
    'no_match_relax', 'no_match_refuse', 'tradeoff_accept', 'tradeoff_reject',
    'misunderstanding', 'suggestion_confirm', 'suggestion_reject',
    'product_reference_refinement', 'completion_confirm', 'completion_continue',
    'poor_results_positive_framing',
)
CASE_MODES = {**{name: 'assistant_free' for name in FREE_CASES},
              **{name: 'assistant_reactive' for name in REACTIVE_CASES}}
FINISHER_CASES = {'finish_task_positive': 'positive', 'finish_task_negative': 'negative',
                  'finish_task_ambiguous': 'ambiguous'}
CASE_MODES.update({name: 'assistant_free' for name in FINISHER_CASES})
CASE_MIN_TURNS = {name: (1 if name == 'initial_need' else 3 if name == 'restore_value' else 2)
                  for name in CASE_MODES}
CASE_MIN_TURNS['finish_task_ambiguous'] = 1
# This describes interpretation difficulty, independently of evidence mode.
# A reactive reply that restates its values is not automatically in this set.
CONTEXT_DEPENDENT_CASES = frozenset({
    'clarification_question', 'select_product', 'compare_products',
    'product_reference_refinement', 'tradeoff_accept', 'tradeoff_reject',
    'suggestion_confirm', 'suggestion_reject',
})
CASE_SUBTYPES = {name: 'context_dependent' if name in CONTEXT_DEPENDENT_CASES else None
                 for name in CASE_MODES}
MIDDLE_CASES = tuple(name for name in CASE_MODES
                     if name not in FINISHER_CASES and name not in ('initial_need', 'finish_task', 'completion_confirm'))


def terminal_record(steps: list[dict], kind: str, *, native: bool = False) -> dict:
    """Session lifecycle cleanup is separate from conversation-supported intent.

    Departure is observable to the environment, never fabricated user evidence.
    Its terminal profile retires task entries without calling them retractions.
    """
    last = steps[-1]
    before = last['planned_state']
    return {'type': kind, 'outcome': {'positive': 'success', 'negative': 'failure', 'ambiguous': 'unknown'}[kind],
            'trigger': 'user_left' if kind == 'ambiguous' else 'user_confirmation',
            'message_id': None if kind == 'ambiguous' else last['shopper_message_id'],
            'after_message_id': last['assistant']['message_id'],
            'native_completion': native,
            'target_product': last['target_product'],
            'terminal_profile': {'active_task_id': None,
                                 'entries': copy.deepcopy([e for e in before['entries'] if e['scope'] != 'task'])},
            'lifecycle_expirations': copy.deepcopy([e for e in before['entries'] if e['scope'] == 'task'])}


def _describe(assignment: dict) -> str:
    return f"{assignment['attribute']} {assignment['operator']} {assignment['value']} {assignment.get('unit', '')}".strip()


def generate_plan(vertical: VerticalSchema, catalog: dict, case: str, *, seed: int = 7,
                  expression: str = 'explicit', recommendation_count: int = 5,
                  turn_count: int | None = None, finisher: str = 'auto') -> dict:
    """Plan every supported case against an actual catalog, or reject infeasibility.

    For the full debug suite a vertical needs one enumerated string attribute
    with two alternatives, one numeric attribute supporting lte, and another
    attribute shared by at least five products. No-match relaxation needs a
    valid threshold below the lowest catalog value. Unsupported catalogs fail
    explicitly, not by inventing products or relaxing requirements.
    """
    if case not in CASE_MODES or expression not in ('explicit', 'implicit', 'mixed'):
        raise ValueError('unknown scenario case or expression mode')
    if finisher not in ('auto', 'positive', 'negative', 'ambiguous'):
        raise ValueError('unknown finisher type')
    if case in FINISHER_CASES:
        if finisher not in ('auto', FINISHER_CASES[case]):
            raise ValueError('finisher selection conflicts with the chosen finisher scenario')
        finisher = FINISHER_CASES[case]
    if turn_count is not None:
        if type(turn_count) is not int or not 1 <= turn_count <= 7:
            raise ValueError('turn_count must be an integer from 1 to 7')
        if turn_count < CASE_MIN_TURNS[case]:
            raise ValueError(f'{case} requires at least {CASE_MIN_TURNS[case]} turns; choose another scenario or increase turns')
    if type(recommendation_count) is not int or recommendation_count < 2:
        raise ValueError('recommendation_count must be at least two')
    validate_catalog(catalog, vertical)
    rng = random.Random(seed)
    fields = vertical.schema['properties']
    numeric = next((name for name, field in fields.items()
                    if field['type'] in ('number', 'integer') and 'lte' in field['x-adaptgym']['operators']
                    and 'enum' not in field), None)
    focal = next((name for name, field in fields.items()
                  if field['type'] == 'string' and len(field.get('enum', [])) >= 2
                  and 'eq' in field['x-adaptgym']['operators']), None)
    if numeric is None or focal is None:
        raise ValueError('suite requires a numeric lte attribute and an enumerated string attribute')
    minimum_matches = recommendation_count + 1 if case == 'soft_preference_missed' else recommendation_count if case in ('no_match_relax', 'no_match_refuse') else 2
    context_candidates = []
    for name in vertical.attributes:
        if name in (focal, numeric) or 'eq' not in fields[name]['x-adaptgym']['operators']:
            continue
        for value, count in Counter(p['attributes'][name] for p in catalog['products']).most_common():
            subset = [p for p in catalog['products'] if p['attributes'][name] == value]
            groups = Counter(p['attributes'][focal] for p in subset)
            eligible = sorted(value for value, n in groups.items() if n >= 1)
            if len(subset) >= minimum_matches and len(eligible) >= 2:
                context_candidates.append((name, value, eligible, subset))
    if not context_candidates:
        raise ValueError('catalog needs a shared context with five products and two categorical alternatives')
    # Prefer an actual shopping use over incidental shared size/colour values.
    sport_candidates = [candidate for candidate in context_candidates if candidate[0] == 'sport']
    if sport_candidates:
        context_candidates = sport_candidates
    if case == 'soft_preference_missed':
        context_candidates = [c for c in context_candidates
                              if any(sum(p['attributes'][focal] != value for p in c[3]) >= recommendation_count
                                     for value in c[2])]
        if not context_candidates:
            raise ValueError('soft-preference case needs five alternatives missing the preference')
    need = generate_current_need(vertical, catalog, seed=rng.getrandbits(32),
        minimum_matches=minimum_matches, diversity_attribute=focal,
        type_candidates=[vertical.assignment(c[0], c[1]) for c in context_candidates])
    base = need['requirements']
    context, context_value = base[0]['attribute'], base[0]['value']
    subset = [p for p in catalog['products'] if p['id'] in need['matching_product_ids']]
    choices = sorted({p['attributes'][focal] for p in subset})
    preferred, alternative = rng.sample(choices, 2)
    if case == 'soft_preference_missed':
        acceptable = [value for value in choices if sum(p['attributes'][focal] != value for p in subset) >= recommendation_count]
        if not acceptable:
            raise ValueError('current need leaves too few alternatives to miss a soft preference')
        preferred = rng.choice(acceptable)
    numeric_requirement = next((a for a in base if a['attribute'] == numeric), None)
    if numeric_requirement is None:
        peak = max(p['attributes'][numeric] for p in subset)
        step = fields[numeric].get('multipleOf', 1)
        budget_values = [peak + step * n for n in range(1, 51)]
        budget_values = [v for v in budget_values if v <= fields[numeric].get('maximum', float('inf'))]
        numeric_requirement = vertical.assignment(numeric, rng.choice(budget_values) if budget_values else peak, 'lte')
    maximum = numeric_requirement['value']
    minimum = min(p['attributes'][numeric] for p in subset)
    below_minimum = minimum - fields[numeric].get('multipleOf', 1)
    preferred_assignment = vertical.assignment(focal, preferred)
    alternate_assignment = vertical.assignment(focal, alternative)
    state = IntentState([vertical])
    visible = []
    steps = []

    def change(assignment, operation='set', scope='task', strength='hard', mode='explicit'):
        return {'operation': operation, 'assignment': assignment, 'scope': scope,
                'strength': strength, 'evidence_mode': mode}

    def preference_mode(index):
        return ('explicit' if index % 2 == 0 else 'implicit') if expression == 'mixed' else expression

    def assistant(kind='recommend', *, attribute=None, ids=None, text=None,
                  deliberate=False, next_instruction=None):
        hard = [entry['assignment'] for entry in state.state['entries'] if entry['strength'] == 'hard']
        valid = matching_ids(catalog, hard)
        soft = [entry['assignment'] for entry in state.state['entries'] if entry['strength'] == 'soft']
        ranked = sorted(valid, key=lambda pid: -sum(matches(next(p for p in catalog['products'] if p['id'] == pid), [a]) for a in soft))
        selected = list(ids) if ids is not None else (ranked[:recommendation_count] if kind in ('recommend', 'tradeoff') else [])
        if len(selected) != len(set(selected)) or any(pid not in {p['id'] for p in catalog['products']} for pid in selected):
            raise ValueError('assistant IDs must reference distinct catalog members')
        invalid = [pid for pid in selected if pid not in valid]
        if invalid and not deliberate:
            raise ValueError('recommendation violates a hard requirement')
        if kind == 'no_match' and valid:
            raise ValueError('no-match script requires zero actual matches')
        if text is None:
            if kind == 'ask_attribute':
                text = f'What is your requirement for {attribute}?'
            elif kind == 'no_match':
                text = f'No products meet these requirements. {next_instruction or "Would you change a requirement?"}'
            elif kind == 'offer_completion':
                text = 'Would you like to finish this shopping task, or keep looking?'
            elif kind == 'suggest_preference':
                text = f'Do you generally prefer {focal} {preferred}?'
            elif kind == 'confirm_selection':
                text = f'You selected {selected[0]}.'
            else:
                text = 'Here are the catalog options:' if selected else 'Understood.'
        if selected:
            rows = []
            for index, pid in enumerate(selected, 1):
                product = next(p for p in catalog['products'] if p['id'] == pid)
                rows.append(f"{index}. {product['name']} [{pid}]: {product.get('description') or product['attributes']}")
            text += '\n' + '\n'.join(rows)
        return {'kind': kind, 'attribute': attribute, 'product_ids': selected,
                'text': text, 'matching_product_ids': valid,
                'deliberate_error': deliberate, 'violating_product_ids': invalid}

    current_target = None

    def add(instruction, changes=(), action='recommend', *, begin=None, end=False,
            selected=None, reaction_to=None, depends_on=(), **assistant_kwargs):
        nonlocal current_target
        if begin:
            current_target = None
        if selected is not None:
            current_target = selected
        index = len(steps) + 1
        mid = f'u{index}'
        visible.append({'id': mid, 'role': 'user', 'text': instruction})
        refs = list(dict.fromkeys([*depends_on, mid]))
        # State snapshots are planned targets. Text validation happens after rendering.
        records = [evidence('explicit', refs, 'Planned change; verify generated language before accepting labels.')]
        executed = []
        if begin:
            boundary = {'operation': 'begin_task', 'task_id': begin, 'evidence': records}
            state.apply(boundary, visible)
            executed.append(boundary)
        for item in changes:
            applied = {key: copy.deepcopy(value) for key, value in item.items() if key != 'evidence_mode'}
            mode = item.get('evidence_mode', 'explicit')
            applied['evidence'] = [evidence(mode, refs, 'Planned evidence; semantic verification is required.')]
            state.apply(applied, visible)
            executed.append(applied)
        if end:
            boundary = {'operation': 'end_task', 'task_id': state.state['active_task_id'], 'evidence': records}
            state.apply(boundary, visible)
            executed.append(boundary)
        reply = assistant(action, **assistant_kwargs)
        step = {'exchange': index, 'shopper_message_id': mid, 'shopper_instruction': instruction,
                'reaction_to': reaction_to, 'expected_events': executed,
                'turn_subtype': 'context_dependent' if case in CONTEXT_DEPENDENT_CASES and reaction_to else None,
                'planned_state': state.state, 'target_product': current_target,
                'assistant': {'message_id': f'a{index}', **reply}}
        step['product_roles'] = classify_products(catalog,
            [entry['assignment'] for entry in state.state['entries'] if entry['strength'] == 'hard'],
            [entry['assignment'] for entry in state.state['entries'] if entry['strength'] == 'soft'])
        steps.append(step)
        visible.append({'id': f'a{index}', 'role': 'assistant', 'text': reply['text']})
        return reply

    full_instruction = 'Start this purchase and ask naturally for ' + ', '.join(map(_describe, base)) + '. Combine these requirements in one shopper-led request.'
    base_changes = [change(a) for a in base]
    if case == 'initial_need' or case in FINISHER_CASES:
        add(full_instruction, base_changes, begin='task-1')
    elif case in ('add_requirement', 'clarify_requirement'):
        later = base[-1]
        initial = base[:-1]
        add('Start a purchase; require ' + ', '.join(map(_describe, initial)) + '.', [change(a) for a in initial], begin='task-1')
        add(('Volunteer an additional requirement: ' if case == 'add_requirement' else 'Clarify that this purchase requires ') + _describe(later),
            [change(later)], depends_on=('u1',))
    elif case == 'reinforce_preference':
        add(full_instruction + f' Also express a general soft preference for {focal} {preferred}; evidence mode {preference_mode(0)}.',
            base_changes + [change(preferred_assignment, scope='durable', strength='soft', mode=preference_mode(0))], begin='task-1')
        add(f'Reinforce the same general preference, without making it mandatory: {focal} {preferred}. Evidence mode {preference_mode(1)}.',
            [change(preferred_assignment, operation='reinforce', scope='durable', strength='soft', mode=preference_mode(1))])
    elif case in ('revise_value', 'retract_requirement', 'restore_value'):
        add(full_instruction + f' For this order, require {focal} {preferred}.', base_changes + [change(preferred_assignment)], begin='task-1')
        if case == 'retract_requirement':
            add(f'Remove only the {focal} requirement for this order.', [change(preferred_assignment, operation='remove')], depends_on=('u1',))
        else:
            add(f'Change only {focal} to {alternative} for this order.', [change(alternate_assignment)], depends_on=('u1',))
            if case == 'restore_value':
                add(f'Restore the original {focal} requirement ({preferred}), referring back naturally.', [change(preferred_assignment)], depends_on=('u1', 'u2'))
    elif case in ('finish_task', 'start_another_task'):
        add(full_instruction + f' Also state a general soft preference for {focal} {preferred}.', base_changes + [change(preferred_assignment, scope='durable', strength='soft')], begin='task-1')
        if case == 'finish_task':
            add('Explicitly finish this purchase.', end=True, action='acknowledge')
        else:
            add(f'Explicitly start a separate purchase; require {focal} {alternative} for this new order.', [change(alternate_assignment)], begin='task-2')
    elif case == 'irrelevant_information':
        add(full_instruction, base_changes, begin='task-1')
        add(f'Mention that a friend chooses {focal} {alternative}, without endorsing it or changing your own needs. Return to the shopping task.', action='recommend')
    elif case == 'clarification_question':
        add('Start a purchase and require ' + ', '.join(map(_describe, base[1:])) + f'. Do not reveal {context} yet.', [change(a) for a in base[1:]], begin='task-1', action='ask_attribute', attribute=context)
        add(f'Answer the assistant question with the value {context_value}; omit the attribute name so the question is necessary context.', [change(base[0])], reaction_to='a1', depends_on=('a1',))
    elif case in ('select_product', 'compare_products', 'product_reference_refinement'):
        reply = add(full_instruction, base_changes, begin='task-1')
        ids = reply['product_ids']
        if len(ids) < 2:
            raise ValueError('scenario requires at least two referenceable products')
        product = next(p for p in catalog['products'] if p['id'] == ids[1])
        if case == 'select_product':
            add('Select the second displayed product by its list position. Do not assert durable preferences for its attributes.', selected=ids[1], reaction_to='a1', depends_on=('a1',), action='confirm_selection', ids=[ids[1]])
        elif case == 'compare_products':
            add('Ask to compare the first and second displayed products, without adding requirements.', reaction_to='a1', depends_on=('a1',), action='compare', ids=ids[:2], text='Comparison of the two requested catalog products:')
        else:
            value = product['attributes'][focal]
            add(f'Refer to the second displayed product and require the same {focal} for this order. Do not name the value directly.', [change(vertical.assignment(focal, value))], reaction_to='a1', depends_on=('a1',))
    elif case == 'soft_preference_missed':
        ids = [p['id'] for p in subset if p['attributes'][focal] != preferred][:recommendation_count]
        add(full_instruction + f' Also express a general soft preference for {focal} {preferred}, evidence mode {preference_mode(0)}.', base_changes + [change(preferred_assignment, scope='durable', strength='soft', mode=preference_mode(0))], begin='task-1', ids=ids)
        add(f'React to the shortlist missing your preference; reinforce {focal} {preferred} as soft, not mandatory. Evidence mode {preference_mode(1)}.', [change(preferred_assignment, operation='reinforce', scope='durable', strength='soft', mode=preference_mode(1))], reaction_to='a1', depends_on=('u1', 'a1'))
    elif case == 'poor_results_positive_framing':
        # Tone does not establish relevance: deliberately recommend actual mismatches.
        wrong = [p['id'] for p in catalog['products'] if not matches(p, base)][:recommendation_count]
        if not wrong:
            raise ValueError('dissatisfaction case needs actual hard-constraint mismatches')
        add(full_instruction, base_changes, begin='task-1', ids=wrong, deliberate=True,
            text='Great news! These are perfect matches for exactly what you asked for:')
        add('React with dissatisfaction: these recommendations are not what you asked for. Refer to the existing request without introducing a new attribute, value, or preference.',
            reaction_to='a1', depends_on=('u1', 'a1'))
    elif case in ('hard_constraint_violated', 'misunderstanding'):
        ids = [p['id'] for p in subset if p['attributes'][focal] == alternative][:recommendation_count]
        add(full_instruction + f' Require {focal} {preferred} for this purchase.', base_changes + [change(preferred_assignment)], begin='task-1', ids=ids, deliberate=True,
            text=(f'I understood that {focal} {alternative} is required. Here are options:' if case == 'misunderstanding' else 'Here are options:'))
        add(f'Correct the assistant: your requirement is still {focal} {preferred}. Do not change the requirement.', [change(preferred_assignment, operation='reinforce')], reaction_to='a1', depends_on=('u1', 'a1'))
    elif case in ('no_match_relax', 'no_match_refuse'):
        low = vertical.assignment(numeric, below_minimum, 'lte')
        add('Start a purchase and require ' + ', '.join(map(_describe, [a for a in base if a['attribute'] != numeric] + [low])) + '.', [change(a) for a in base if a['attribute'] != numeric] + [change(low)], begin='task-1', action='no_match',
            next_instruction=f'Would you increase {numeric} maximum to {maximum} {numeric_requirement.get("unit", "")}?')
        if case == 'no_match_relax':
            add(f'Accept increasing the maximum to {maximum}, keeping other requirements unchanged.', [change(numeric_requirement)], reaction_to='a1', depends_on=('u1', 'a1'))
        else:
            add('Decline the proposed increase. Keep the original maximum; do not invent alternatives.', [change(low, operation='reinforce')], reaction_to='a1', depends_on=('u1', 'a1'), action='no_match', next_instruction='There are still no matches at your unchanged limit.')
    elif case in ('tradeoff_accept', 'tradeoff_reject'):
        # Trade-off is between soft brand preference and objectively cheaper price.
        groups = [(value, [p for p in subset if p['attributes'][focal] == value]) for value in choices]
        pairs = [(a, b) for a in groups for b in groups if a[0] != b[0]
                 and min(p['attributes'][numeric] for p in a[1]) > min(p['attributes'][numeric] for p in b[1])]
        if not pairs:
            raise ValueError('trade-off requires differently priced categorical alternatives')
        expensive, cheap = rng.choice(pairs)
        pref = vertical.assignment(focal, expensive[0])
        cheap_product = min(cheap[1], key=lambda p: p['attributes'][numeric])
        dear_product = min(expensive[1], key=lambda p: p['attributes'][numeric])
        ids = [cheap_product['id'], dear_product['id']] + [p['id'] for p in subset if p['id'] not in (cheap_product['id'], dear_product['id'])][:recommendation_count-2]
        add(full_instruction + f' State a general soft preference for {focal} {expensive[0]}.', base_changes + [change(pref, scope='durable', strength='soft')], begin='task-1', action='tradeoff', ids=ids,
            text=f"The first item has {focal} {cheap[0]} and {numeric} {cheap_product['attributes'][numeric]}; the second has your preferred {focal} {expensive[0]} and {numeric} {dear_product['attributes'][numeric]}.")
        chosen = cheap_product if case == 'tradeoff_accept' else dear_product
        position = 'first' if case == 'tradeoff_accept' else 'second'
        add(f'Choose the {position} displayed item after considering the trade-off. This is a one-off selection, not a change of durable preference.', selected=chosen['id'], reaction_to='a1', depends_on=('a1',), action='confirm_selection', ids=[chosen['id']])
    elif case in ('suggestion_confirm', 'suggestion_reject'):
        ids = matching_ids(catalog, base)[:recommendation_count]
        anchor = next(product for product in catalog['products'] if product['id'] == ids[1])
        proposal = vertical.assignment(focal, anchor['attributes'][focal])
        add(full_instruction + f' Do not express any {focal} preference yet.', base_changes, begin='task-1', action='suggest_preference', ids=ids,
            text=f'Here are the options. Do you generally prefer the {focal} of the second pair?')
        if case == 'suggestion_confirm':
            mode = preference_mode(0)
            instruction = ('Answer yes to the assistant question as a general soft preference. Do not repeat the attribute or value.'
                           if mode == 'explicit' else
                           f'Describe repeatedly choosing the {focal} of the second displayed pair with good experiences. Do not name its value or directly state a preference.')
            add(instruction, [change(proposal, scope='durable', strength='soft', mode=mode)], reaction_to='a1', depends_on=('a1',))
        else:
            add('Answer no to the assistant suggestion. Do not name the attribute or value, invent an alternative, or end the shopping task.', reaction_to='a1', depends_on=('a1',))
    elif case in ('completion_confirm', 'completion_continue'):
        add(full_instruction, base_changes, begin='task-1', action='offer_completion')
        if case == 'completion_confirm':
            add('Confirm that this shopping task is finished.', end=True, reaction_to='a1', depends_on=('a1',), action='acknowledge')
        else:
            add('Decline completion and ask to keep looking with the same requirements.', reaction_to='a1', depends_on=('a1',))
    # Preserve the complete core event before composing additional exchanges.
    service_questions = (
        ('delivery times', 'Delivery times are not provided in this debug catalog.'),
        ('returns policy', 'Returns policies are not provided in this debug catalog.'),
        ('warranty details', 'Warranty terms are not provided in this debug catalog.'),
        ('care instructions', 'Care instructions are not provided in this debug catalog.'),
        ('stock availability', 'These are synthetic offers; live stock is not provided.'),
        ('delivery costs', 'Delivery costs are not provided in this debug catalog.'),
    )
    core_turns = len(steps)
    native_completion = case in ('finish_task', 'completion_confirm')
    if native_completion and finisher == 'ambiguous':
        raise ValueError('this scenario explicitly completes the task; choose finish_task_ambiguous for user departure')
    # Default-length chains use departure when no dialogue slot remains.
    # Longer chains sample an explicit ending when feasible; native completion
    # already provides a final exchange and can be reused.
    if finisher == 'auto':
        if native_completion:
            finisher = 'positive'
        elif turn_count is not None and turn_count > core_turns:
            finisher = rng.choice(('positive', 'negative', 'ambiguous'))
        else:
            finisher = 'ambiguous'
    reuse_completion = native_completion and (turn_count is None or turn_count == core_turns)
    explicit_extra = finisher != 'ambiguous' and not reuse_completion
    target_turns = turn_count if turn_count is not None else core_turns + int(explicit_extra)
    if target_turns < core_turns + int(explicit_extra):
        raise ValueError(f'{case} with a {finisher} finisher requires at least {core_turns + 1} turns')
    content_turns = target_turns - int(explicit_extra)
    while len(steps) < content_turns:
        previous = steps[-1]
        selected = previous['target_product']
        if state.state['active_task_id'] is None or selected is not None:
            topic, answer = service_questions[len(steps) - core_turns]
            add(f'Ask about {topic}. Do not reopen a finished task or change the existing selection or requirements.',
                action='acknowledge', text=answer, selected=selected)
            continue
        hard = [entry['assignment'] for entry in state.state['entries'] if entry['strength'] == 'hard']
        valid = matching_ids(catalog, hard)
        if not valid:
            add('Ask whether any item meets the unchanged request. Keep all requirements and refuse to relax them.',
                action='no_match', text='There are still no matching products under the unchanged requirements.')
            continue
        established = {entry['assignment']['attribute'] for entry in state.state['entries']}
        candidates = [name for name in ('size', 'colour', 'gender', 'surface', 'upper_material', 'cushioning')
                      if name in fields and name not in established and 'eq' in fields[name]['x-adaptgym']['operators']]
        if candidates:
            name = candidates[0]
            witness_id = rng.choice(valid)
            witness = next(product for product in catalog['products'] if product['id'] == witness_id)
            assignment = vertical.assignment(name, witness['attributes'][name])
            add(f'Add this requirement for the current purchase: {_describe(assignment)}. Keep earlier requirements unchanged.',
                [change(assignment)])
        else:
            topic, answer = service_questions[len(steps) - core_turns]
            add(f'Ask about {topic}, without changing your shopping requirements.', action='acknowledge', text=answer)
    if finisher != 'ambiguous':
        instruction = ('Say you found what you wanted and explicitly finish shopping successfully. '
                       'Do not name a product unless a product was already selected.' if finisher == 'positive' else
                       'Say you did not find what you wanted and explicitly stop shopping unsuccessfully. '
                       'Do not invent a reason, retract preferences, or start another task.')
        if finisher == 'positive' and steps[-1]['assistant']['kind'] == 'no_match':
            instruction = 'Explicitly say you found what you needed elsewhere and are ending this shopping task successfully. Do not claim any displayed product matches or name a selected catalog item.'
        response = 'Glad you found what you needed. This shopping task is complete.' if finisher == 'positive' else 'Understood. We will stop here without a purchase.'
        if reuse_completion:
            steps[-1]['shopper_instruction'] = instruction
            steps[-1]['assistant']['text'] = response
        else:
            selected = steps[-1]['target_product'] if finisher == 'positive' else None
            add(instruction, end=state.state['active_task_id'] is not None,
                action='acknowledge', text=response, selected=selected)
        steps[-1]['finisher'] = finisher
    ending = terminal_record(steps, finisher, native=reuse_completion)
    plan = {'schema_version': 'adaptgym.session-plan.v2', 'id': f'{case}-{seed}-{expression}' + (f'-t{turn_count}' if turn_count is not None else '') + f'-{finisher}',
            'turn_count': len(steps), 'core_turn_count': core_turns,
            'finisher': ending, 'current_need': need,
            'mode': CASE_MODES[case], 'scenario_type': case, 'seed': seed,
            'scenario_subtype': CASE_SUBTYPES[case],
            'expression': expression, 'recommendation_count': recommendation_count,
            'vertical_id': vertical.vertical_id, 'vertical_schema': vertical.schema,
            'catalog': copy.deepcopy(catalog), 'exchanges': steps,
            'status': 'private_plan_requires_language_verification',
            'instructions': [
                'Generate only shopper messages. Assistant messages are fixed by code.',
                'React to actual displayed products and preserve list order.',
                'Do not reveal later instructions or invent requirements.',
                'A product choice does not imply durable preferences for all its attributes.',
                'Implicit preferences use repeated-choice evidence, without directly stating a preference.',
            ]}
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    """Replay private events and check catalog conditions for every exchange."""
    vertical = VerticalSchema(plan['vertical_schema'])
    validate_catalog(plan['catalog'], vertical)
    state = IntentState([vertical])
    messages = []
    catalog_ids = {p['id'] for p in plan['catalog']['products']}
    case = plan.get('scenario_type')
    if case not in CASE_MODES or plan.get('mode') != CASE_MODES[case] or plan.get('scenario_subtype') != CASE_SUBTYPES[case]:
        raise ValueError('scenario mode/subtype does not match its registered case')
    if not plan['exchanges'] or plan.get('turn_count') != len(plan['exchanges']):
        raise ValueError('chain needs a valid turn count')
    for step in plan['exchanges']:
        expected_subtype = 'context_dependent' if case in CONTEXT_DEPENDENT_CASES and step['reaction_to'] else None
        if step.get('turn_subtype') != expected_subtype:
            raise ValueError('turn subtype must reflect an actual assistant-context dependency')
        messages.append({'id': step['shopper_message_id'], 'role': 'user', 'text': step['shopper_instruction']})
        for event in step['expected_events']:
            state.apply(event, messages)
        if state.state != step['planned_state']:
            raise ValueError('planned state does not match replayed events')
        action = step['assistant']
        ids = action['product_ids']
        hard = [e['assignment'] for e in state.state['entries'] if e['strength'] == 'hard']
        actual = matching_ids(plan['catalog'], hard)
        if action['matching_product_ids'] != actual or len(ids) != len(set(ids)) or set(ids) - catalog_ids:
            raise ValueError('invalid catalog evidence in assistant action')
        invalid = [pid for pid in ids if pid not in actual]
        if invalid != action['violating_product_ids'] or (invalid and not action['deliberate_error']):
            raise ValueError('unmarked hard-constraint violation')
        if action['kind'] == 'no_match' and actual:
            raise ValueError('false no-match claim')
        if action['kind'] in ('recommend', 'tradeoff') and not action['deliberate_error'] and len(ids) != min(len(actual), plan['recommendation_count']):
            raise ValueError('incorrect shortlist count')
        if step['target_product'] is not None and step['target_product'] not in catalog_ids:
            raise ValueError('invalid selected product')
        if step['reaction_to'] and step['reaction_to'] not in {m['id'] for m in messages if m['role'] == 'assistant'}:
            raise ValueError('reaction must reference a previous assistant reply')
        messages.append({'id': action['message_id'], 'role': 'assistant', 'text': action['text']})
    ending = plan.get('finisher')
    if not isinstance(ending, dict) or ending.get('type') not in ('positive', 'negative', 'ambiguous'):
        raise ValueError('every chain needs a positive, negative, or ambiguous finisher')
    if ending != terminal_record(plan['exchanges'], ending['type'], native=ending.get('native_completion', False)):
        raise ValueError('finisher lifecycle does not match the final checkpoint')
    if any(step.get('finisher') for step in plan['exchanges'][:-1]):
        raise ValueError('no dialogue may follow a finisher')
    if ending['type'] != 'ambiguous':
        if plan['exchanges'][-1].get('finisher') != ending['type'] or state.state['active_task_id'] is not None:
            raise ValueError('explicit finisher must close the task in the final exchange')
    elif any(step.get('finisher') for step in plan['exchanges']):
        raise ValueError('departure must not fabricate a finishing shopper message')
