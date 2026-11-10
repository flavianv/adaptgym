"""Domain-independent dialogue planning over schema-defined requirements and preferences.

Properties are AdaptGym typed scalar assignments; item matching is supplied by the catalog.
Dependent property groups and readable question labels are declared in the domain schema.
"""
from __future__ import annotations

import copy
import json
import random

from adaptgym.catalog import matches, matching_ids
from adaptgym.intent_state import IntentState, evidence
from adaptgym.product_interest import PRODUCT_ACTIONS, ProductInterest
from adaptgym.session_plans import CASE_MODES, CASE_SUBTYPES, terminal_record, validate_plan

MIDDLE_TYPES = ('assistant_free', 'reactive_non_context', 'reactive_context')
CASES = dict(zip(MIDDLE_TYPES, ('add_requirement', 'hard_constraint_violated', 'clarification_question')))


def property_groups(assignments, linked_groups=(), known_properties=None):
    """Treat linked facts atomically; reject partial, unknown or overlapping groups."""
    names = [a['attribute'] for a in assignments]
    if len(names) != len(set(names)):
        raise ValueError('requirements must have distinct properties')
    present, grouped, used = set(names), [], set()
    for group in linked_groups:
        if (not isinstance(group, (list, tuple)) or len(group) < 2
                or len(set(group)) != len(group) or used.intersection(group)):
            raise ValueError('disclosure groups must contain distinct, non-overlapping properties')
        if known_properties is not None and not set(group) <= set(known_properties):
            raise ValueError('disclosure group contains an unknown property')
        used.update(group)
        included = present.intersection(group)
        if included and included != set(group):
            raise ValueError('linked properties must be supplied together')
        if included:
            grouped.append([a for name in group for a in assignments if a['attribute'] == name])
    return [[a] for a in assignments if a['attribute'] not in used] + grouped


def describe(assignment: dict) -> str:
    return f"{assignment['attribute']} {assignment['operator']} {assignment['value']} {assignment.get('unit', '')}".strip()


def hydrate(record: dict, catalog: dict, vertical) -> dict:
    return {**copy.deepcopy(record), 'catalog': catalog, 'vertical_schema': vertical.schema}


def middle_schedule(turns, product_actions, middle_type, seed, supplied=None):
    slots = turns - 2 - len(product_actions)
    if slots < 1:
        raise ValueError('reserve an initial request, a disclosure turn and a finisher')
    if supplied is not None:
        kinds = list(supplied)
        if len(kinds) != slots or any(kind not in MIDDLE_TYPES for kind in kinds):
            raise ValueError('middle schedule must name a valid type for every disclosure slot')
        return kinds
    if middle_type == 'mixed':
        rng = random.Random(seed)
        kinds = list(MIDDLE_TYPES[:slots]) + rng.choices(MIDDLE_TYPES, k=max(0, slots - len(MIDDLE_TYPES)))
        rng.shuffle(kinds)
        return kinds
    if middle_type not in MIDDLE_TYPES:
        raise ValueError('unknown middle type')
    return [middle_type] * slots


def attribute_budget(kinds, minimum=6, maximum=12):
    """Additional disclosure turns carry more attributes; correction/context turns carry fewer."""
    return min(maximum, minimum + sum(2 if kind == 'assistant_free' else 1 for kind in kinds[1:]))


def plan_dialogue(requirements, preferences, catalog, vertical, *, target_id, positive_ending,
                  persona_id, turns: int, middle_type: str, seed: int, product_actions=(), middle_types=None) -> dict:
    if type(turns) is not int or turns < 3 or middle_type not in (*MIDDLE_TYPES, 'mixed'):
        raise ValueError('scenarios require at least 3 turns and a registered middle type')
    kinds = middle_schedule(turns, product_actions, middle_type, seed, middle_types)
    vocabulary = vertical.simulation['vocabulary']
    task, item = vocabulary['task'], vocabulary['item']
    rng = random.Random(seed)
    hard = copy.deepcopy(requirements)
    soft = copy.deepcopy(preferences)
    for assignment in hard + soft:
        vertical.validate_assignment(assignment)
    linked = vertical.schema.get('x-adaptgym', {}).get('disclosure_groups', ())
    def tagged_groups(assignments, scope, strength):
        return [[dict(a, scope=scope, strength=strength) for a in group]
                for group in property_groups(assignments, linked, vertical.attributes)]

    def untag(assignment):
        return {k: v for k, v in assignment.items() if k not in ('scope', 'strength')}

    def partition(assignments):
        return ([untag(a) for a in assignments if a['scope'] == 'task'],
                [untag(a) for a in assignments if a['scope'] == 'durable'])

    groups = tagged_groups(hard, 'task', 'hard') + tagged_groups(soft, 'durable', 'soft')
    rng.shuffle(groups)
    if any(action not in PRODUCT_ACTIONS for action in product_actions):
        raise ValueError('unknown product action')
    if product_actions and not positive_ending:
        raise ValueError('post-search product actions require an eligible-item scenario')
    middle_count = turns - 2 - len(product_actions)
    if middle_count < 1:
        raise ValueError('reserve an initial request, a disclosure turn and a finisher')
    if len(groups) < middle_count + 1:
        raise ValueError('not enough undisclosed requirement/preference groups for this many turns')
    initial_n = rng.randint(1, len(groups) - middle_count)
    initial_choices = [i for i, group in enumerate(groups) if group[0]['scope'] == 'task']
    if 'reactive_non_context' in kinds:
        initial_choices = [i for i in initial_choices
                           if any(not matches(product, [untag(a) for a in groups[i]])
                                  for product in catalog['products'])]
    if not initial_choices:
        raise ValueError('scenario needs an initial requirement; correction tests need a violating item')
    first = rng.choice(initial_choices)
    groups[0], groups[first] = groups[first], groups[0]
    # Keep a requirement for a later turn as well, when there is more than one.
    later_hard_groups = [i for i in range(1, len(groups)) if groups[i][0]['scope'] == 'task']
    if later_hard_groups:
        reserve = rng.choice(later_hard_groups)
        groups[-1], groups[reserve] = groups[reserve], groups[-1]
    initial_hard, initial_soft = partition([a for group in groups[:initial_n] for a in group])
    remaining = groups[initial_n:]
    batches = [[a for group in remaining[i::middle_count] for a in group]
               for i in range(middle_count)]
    later_hard, later_soft = partition([a for batch in batches for a in batch])
    state, messages, steps = IntentState([vertical]), [], []
    case = CASES[kinds[0]]
    good = positive_ending
    anchor = target_id
    products = {p['id']: p for p in catalog['products']}
    interest = ProductInterest()
    if good and anchor not in products:
        raise ValueError('positive ending needs a target item in the catalog')

    def assignments():
        entries = state.state['entries']
        hard_now = [e['assignment'] for e in entries if e['strength'] == 'hard']
        overridden = {a['attribute'] for a in hard_now}
        soft_now = [e['assignment'] for e in entries if e['strength'] == 'soft'
                    and e['assignment']['attribute'] not in overridden]
        return hard_now, soft_now

    def action(*, prepare_middle=False, question=None, finish=False, interest_update=False, next_kind=None):
        hard_now, soft_now = assignments()
        valid = matching_ids(catalog, hard_now)
        ranked = sorted(valid, key=lambda pid: (
            -sum(matches(products[pid], [a]) for a in soft_now), pid != anchor, pid))
        ids = ranked[:3]
        kind, attribute, deliberate = ('recommend' if valid else 'no_match'), None, False
        requested = []
        requested_claims = []
        text = 'Here are options meeting the current hard requirements.' if valid else 'No options meet the current hard requirements.'
        if finish:
            kind, ids, text = 'acknowledge', [], (f'Your selected {item} is confirmed; the {task} is complete.' if good else f'The {task} is closed without a suitable {item}.')
        elif interest.selected:
            kind, ids = 'confirm_selection', list(interest.selected)
            text = f'I have noted your selection. Is this {task} complete?'
        elif interest_update:
            kind, ids = 'shortlist_update', list(interest.targets)
            text = ('I have updated your shortlist. No final choice has been made.' if ids
                    else 'Your shortlist is now empty. We can continue looking.')
        elif prepare_middle and next_kind == 'reactive_context':
            kind, ids = 'ask_attribute', []
            known = {(e['assignment']['attribute'], e['scope'], e['strength']) for e in state.state['entries']}
            if not question or any((a['attribute'], a['scope'], a['strength']) in known for a in question):
                raise ValueError('clarification questions must target undisclosed requirements')
            requested = [a['attribute'] for a in question]
            requested_claims = [{k: a[k] for k in ('attribute', 'scope', 'strength')} for a in question]
            attribute = question[0]['attribute']
            labels = [('your general preference for ' if a['scope'] == 'durable' else 'your mandatory requirement for ')
                      + vertical.schema['properties'][a['attribute']].get('title', a['attribute'].replace('_', ' '))
                      for a in question]
            text = f'For this {task}, what do you need regarding ' + '; then '.join(labels) + '?'
            if len(question) > 1:
                text += ' Please answer in that order.'
        elif (prepare_middle and next_kind == 'reactive_non_context'
              and not any(step['assistant']['deliberate_error'] for step in steps)):
            violating = [p['id'] for p in catalog['products'] if p['id'] not in valid]
            if not violating:
                raise ValueError('reactive correction needs a real violating catalog item')
            violations = {pid: sum(not matches(products[pid], [a]) for a in hard_now) for pid in violating}
            minimum = min(violations.values())
            chosen = rng.choice([pid for pid in violating if violations[pid] == minimum])
            failed = next(a for a in hard_now if not matches(products[chosen], [a]))
            kind, ids, deliberate, attribute = 'recommend', [chosen], True, failed['attribute']
            text = f'Would this {item} work for you?'
        for pid in ids:
            product = products[pid]
            # Show real attributes, not hidden planted roles or labels.
            values = {a['attribute']: product['attributes'].get(a['attribute']) for a in hard + soft}
            text += '\n' + f"{product['name']} [{pid}]: " + json.dumps(values, ensure_ascii=False)
        return {'kind': kind, 'attribute': attribute, 'product_ids': ids, 'text': text,
                'requested_attributes': requested, 'requested_claims': requested_claims,
                'matching_product_ids': valid, 'deliberate_error': deliberate,
                'violating_product_ids': [pid for pid in ids if pid not in valid]}

    def add(instruction, new_hard=(), new_soft=(), *, begin=False, reaction=False,
            reinforce=None, prepare_middle=False, question=None, finish=False, product_event=None, middle_kind=None, next_kind=None):
        index = len(steps) + 1
        uid, aid = f'u{index}', f'a{index}'
        refs = ([f'a{index - 1}'] if reaction else []) + [uid]
        messages.append({'id': uid, 'role': 'user', 'text': instruction})
        records = [evidence('explicit', refs, 'Planned evidence; generated language must pass the separate model audit.')]
        events = []
        if begin:
            events.append({'operation': 'begin_task', 'task_id': 'task-1', 'evidence': records})
        for strength, scope, values in (('hard', 'task', new_hard), ('soft', 'durable', new_soft)):
            events.extend({'operation': 'set', 'assignment': a, 'strength': strength, 'scope': scope,
                           'evidence': records} for a in values)
        if reinforce is not None:
            events.append({'operation': 'reinforce', 'assignment': reinforce,
                           'strength': 'hard', 'scope': 'task', 'evidence': records})
        shown = {pid for step in steps for pid in step['assistant']['product_ids']}
        product_events = []
        if product_event:
            product_events.append(interest.apply(*product_event, shown=shown, message_id=uid))
        selected = interest.selected[0] if len(interest.selected) == 1 else None
        if finish:
            hard_now, soft_now = assignments()
            if good:
                if not matches(products[anchor], hard_now + soft_now):
                    raise ValueError('positive ending requires an actual matching item')
                if anchor not in shown:
                    raise ValueError('selected item must have been displayed')
                selected = anchor
                if interest.selected != [anchor]:
                    product_events.append(interest.apply('select', [anchor], shown=shown, message_id=uid))
            elif matching_ids(catalog, hard_now):
                raise ValueError('negative scenario unexpectedly has eligible items')
            events.append({'operation': 'end_task', 'task_id': 'task-1', 'evidence': records})
        for event in events:
            state.apply(event, messages)
        reply = action(prepare_middle=prepare_middle, question=question, finish=finish,
                       interest_update=product_event is not None, next_kind=next_kind)
        step = {'exchange': index, 'shopper_message_id': uid, 'shopper_instruction': instruction,
                'reaction_to': f'a{index - 1}' if reaction else None,
                'turn_subtype': 'context_dependent' if reaction and (
                    middle_kind == 'reactive_context' or product_event and product_event[0] == 'express_interest') else None,
                'expected_events': events, 'planned_state': state.state, 'target_product': selected,
                **interest.snapshot(), 'expected_product_events': product_events,
                'middle_type': middle_kind,
                'user_action': product_event[0] if product_event else 'finish_task' if finish else None,
                'assistant': {'message_id': aid, **reply}}
        if finish:
            step['finisher'] = 'positive' if good else 'negative'
        steps.append(step)
        messages.append({'id': aid, 'role': 'assistant', 'text': reply['text']})

    def state_words(required, preferred):
        text = f' For this {task}, require ' + '; '.join(map(describe, required)) + '.' if required else ''
        if preferred:
            text += ' As general soft preferences, not mandatory conditions, prefer ' + '; '.join(map(describe, preferred)) + '.'
        return text

    add(f'Start this {task}.' + state_words(initial_hard, initial_soft),
        initial_hard, initial_soft, begin=True, prepare_middle=True, question=batches[0], next_kind=kinds[0])
    for i, batch in enumerate(batches):
        last_middle = i == middle_count - 1
        batch_hard, batch_soft = partition(batch)
        kind = kinds[i]
        if kind == 'reactive_context':
            instruction = 'Answer the assistant question with these values in its stated order, without repeating attribute names: ' + '; '.join(str(a['value']) for a in batch) + '. Preserve the required comparison operators and units: ' + '; '.join(a['operator'] + ' ' + a.get('unit', '') for a in batch) + '. Keep earlier needs unchanged. Preserve each answer as a mandatory task requirement or a general soft preference respectively: ' + '; '.join(a['scope'] + ' ' + a['strength'] for a in batch) + '.'
        elif kind == 'reactive_non_context':
            if steps[-1]['assistant']['deliberate_error']:
                wrong = steps[-1]['assistant']['attribute']
                prior = next(a for a in hard if a['attribute'] == wrong)
                instruction = f'Reject the assistant {item} because it violates this unchanged task requirement: ' + describe(prior) + '.' + state_words(batch_hard, batch_soft)
            else:
                instruction = 'React to the assistant results by adding the following needs. Do not claim the results violate an already known requirement. If there were no matches, acknowledge that without relaxing earlier needs.' + state_words(batch_hard, batch_soft)
        else:
            instruction = 'Add these new needs, keeping earlier requirements.' + state_words(batch_hard, batch_soft)
        add(instruction, batch_hard, batch_soft, reaction=kind != 'assistant_free',
            prepare_middle=not last_middle, question=None if last_middle else batches[i + 1],
            middle_kind=kind, next_kind=None if last_middle else kinds[i + 1])
    for operation in product_actions:
        displayed = steps[-1]['assistant']['product_ids']
        if operation == 'express_interest':
            ids = displayed[:2]
            if not ids:
                raise ValueError('interest requires displayed candidates')
            positions = ', '.join(str(displayed.index(pid) + 1) for pid in ids)
            instruction = f'Explicitly say the displayed options at positions {positions} interest you. Do not select one or end the task.'
        elif operation == 'withdraw_interest':
            if not interest.targets:
                raise ValueError('withdrawal requires an earlier expression of interest')
            ids = [interest.targets[-1]]
            instruction = f'Explicitly withdraw your interest in item [{ids[0]}]. Do not reject other items or end the task.'
        else:
            ids = [anchor]
            instruction = f'Explicitly select item [{anchor}]. If you previously withdrew interest, say you reconsidered. Do not finish the task yet.'
        add(instruction, reaction=True, product_event=(operation, ids))
    ending = (f'Explicitly choose the displayed {item} [{anchor}], say it meets your needs and you are satisfied, and close this {task} successfully.' if good else f'Explicitly say no {item} meets your unchanged requirements, you are dissatisfied, and close this {task} unsuccessfully without selecting an item. Do not relax any requirement.')
    if good and any(event['action'] == 'withdraw_interest' and anchor in event['product_ids'] for event in interest.events):
        ending = f'Explicitly say you reconsidered the previously displayed {item} [{anchor}], choose it after all, and close this {task} successfully.'
    if good and interest.selected == [anchor]:
        ending = f'Confirm that the already selected {item} [{anchor}] meets your needs, say you are satisfied, and explicitly close this {task}. Do not make a new selection.'
    add(ending, finish=True)
    plan = {'schema_version': 'adaptgym.session-plan.v2', 'id': f"{persona_id}-t{turns}",
            'generation_policy': 'budgeted_mixed_dialogue_v4',
            'user_id': persona_id, 'turn_count': turns, 'core_turn_count': turns,
            'scenario_type': case, 'mode': CASE_MODES[case], 'scenario_subtype': CASE_SUBTYPES[case],
            'middle_type': 'mixed' if len(set(kinds)) > 1 else kinds[0], 'middle_type_schedule': kinds, 'seed': seed, 'expression': 'explicit', 'recommendation_count': 3,
            'middle_type_scope': 'requirement_disclosure_turns',
            'vertical_id': vertical.vertical_id, 'exchanges': steps,
            'finisher': terminal_record(steps, 'positive' if good else 'negative'),
            'disclosure': {'initial_hard': [a['attribute'] for a in initial_hard],
                           'later_hard': [a['attribute'] for a in later_hard],
                           'initial_soft': [a['attribute'] for a in initial_soft],
                           'later_soft': [a['attribute'] for a in later_soft]},
            'product_actions': list(product_actions),
            'status': 'private_plan_requires_language_verification'}
    validate_plan(hydrate(plan, catalog, vertical))
    return plan



def sample_groups(groups, count, rng):
    """Choose an exact-size subset without enumerating the powerset."""
    candidates = list(groups)
    rng.shuffle(candidates)
    choices = {0: []}
    for group in candidates:
        for size, selected in list(choices.items()):
            total = size + len(group)
            if total <= count and total not in choices:
                choices[total] = selected + [group]
    if count not in choices:
        raise ValueError('linked properties cannot meet this attribute count')
    return [a for group in choices[count] for a in group]


def catalog_dialogue(vertical, catalog, *, seed, turns, finisher='positive'):
    """Build a mixed dialogue from a catalog witness, for the interactive lab."""
    if finisher not in ('auto', 'positive', 'negative'):
        raise ValueError('mixed dialogues require an explicit positive or negative ending')
    rng = random.Random(seed)
    positive = finisher != 'negative'
    actions = ('express_interest', 'withdraw_interest', 'select')[:max(0, min(3, turns - 5))] if positive else ()
    kinds = middle_schedule(turns, actions, 'mixed', seed)
    schema = vertical.schema
    meta = schema['x-adaptgym']
    pool = meta.get('benchmark_attributes', list(schema['properties']))
    preferred_fields = meta.get('preference_attributes', [vertical.simulation['preference_attribute']])
    candidates = []
    for product in catalog['products']:
        available = {name: product['attributes'][name] for name in pool
                     if name in product['attributes'] and product['attributes'][name] != 'unspecified'}
        for group in meta.get('disclosure_groups', ()):
            if not set(group) <= set(available):
                for name in group:
                    available.pop(name, None)
        if len(available) >= 6:
            candidates.append((product, available))
    if not candidates:
        raise ValueError('mixed dialogues need a catalog item with at least six supported properties')
    anchor, values = rng.choice(candidates)
    preferences = [vertical.assignment(name, values[name]) for name in preferred_fields if name in values]
    hard = [vertical.assignment(name, value) for name, value in values.items()
            if name not in {a['attribute'] for a in preferences}]
    desired = min(attribute_budget(kinds), len(values))
    groups = property_groups(hard, meta.get('disclosure_groups', ()), vertical.attributes)
    if positive:
        hard = sample_groups(groups, desired - len(preferences), rng)
    else:
        forced = None
        for group in groups:
            for assignment in group:
                name = assignment['attribute']
                if schema['properties'][name]['type'] not in ('integer', 'number'):
                    continue
                observed = [p['attributes'][name] for p in catalog['products'] if name in p['attributes']]
                for operator, value in [('gte', max(observed) + 1), ('lte', min(observed) - 1)]:
                    try:
                        impossible = vertical.assignment(name, value, operator)
                    except ValueError:
                        continue
                    if not matching_ids(catalog, [impossible]):
                        forced = (group, impossible)
                        break
                if forced:
                    break
            if forced:
                break
        if not forced:
            raise ValueError('catalog has no representable no-match numeric condition')
        group, impossible = forced
        hard = sample_groups([g for g in groups if g is not group],
                             desired - len(preferences) - len(group), rng)
        hard += [impossible if a['attribute'] == impossible['attribute'] else a for a in group]
    plan = plan_dialogue(hard, preferences, catalog, vertical, target_id=anchor['id'],
                         positive_ending=positive, persona_id=f'lab-mixed-{seed}', turns=turns,
                         middle_type='mixed', seed=seed, product_actions=actions, middle_types=kinds)
    plan.update(scenario_type='mixed', mode=CASE_MODES['mixed'], scenario_subtype=CASE_SUBTYPES['mixed'])
    plan['attribute_budget'] = {'distinct_attributes': desired, 'disclosure_turns': len(kinds),
                                'product_action_turns': len(actions)}
    validate_plan(hydrate(plan, catalog, vertical))
    return hydrate(plan, catalog, vertical)
