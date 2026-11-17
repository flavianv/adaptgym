"""A finite-state transducer for the shape of a dialogue.

The planner (``dialogue_planning``) is procedural. This module states the same turn structure as data, so it can be
checked: every plan the planner builds must be accepted by the machine (conformance), every transition must be reached
by some plan or carry a written reason it cannot be (reachability), and the machine itself can be enumerated for dead
ends (deadlock freedom).

Input symbols are user turn types, output symbols are assistant replies. The reply to a turn is planned with a look
ahead at the next user turn, so the machine expresses that look ahead as a constraint on the next input ("forced").
What it does not describe is the catalog: whether a partial item exists is decided by the planner, and the language
the shopper model produces is checked by the live runs.
"""
from __future__ import annotations

from collections import namedtuple
from itertools import product as cartesian

DISCLOSURE = ('ADD', 'REACT', 'ANSWER', 'REJECT', 'REVEAL')
ACTIONS = ('PREFER', 'INTEREST', 'WITHDRAW', 'SELECT')
CLOSES = ('CLOSE_NO', 'CHOOSE', 'CHOOSE_BEST', 'CONFIRM', 'RECONSIDER')
SYMBOLS = ('OPENER', *DISCLOSURE, *ACTIONS, *CLOSES)

# a reply is (kind, deliberate_error, hidden_violation, listed) with listed = number of items shown, capped at 2
Reply = namedtuple('Reply', 'kind deliberate hidden listed')
# spent: some earlier reply already carried a deliberate error (the planner shows at most one planned wrong item)
State = namedtuple('State', 'phase forced deliberate hidden listed endorsed selected withdrawn spent')
START = State('start', None, False, False, 0, 0, False, False, False)

LISTING = (
    [Reply(kind, False, False, n) for kind in ('recommend', 'scripted_results') for n in (1, 2)]
    + [Reply(kind, deliberate, hidden, n) for kind in ('recommend', 'scripted_results')
       for deliberate, hidden in ((True, False), (False, True)) for n in (1, 2)]
    + [Reply('scripted_results', True, True, n) for n in (1, 2)]          # a scripted wrong item, then a reveal
    + [Reply('ask_attribute', False, False, 0), Reply('no_match', False, False, 0), Reply('compare', False, False, 2)])
OUTPUTS = {
    **{symbol: LISTING for symbol in ('OPENER', *DISCLOSURE)},
    'PREFER': [Reply('preference_update', False, False, 2)],
    'INTEREST': [Reply('shortlist_update', False, False, n) for n in (1, 2)],
    'WITHDRAW': [Reply('shortlist_update', False, False, n) for n in (0, 1, 2)],
    'SELECT': [Reply('confirm_selection', False, False, 1)],
    **{symbol: [Reply('acknowledge', False, False, 0)] for symbol in CLOSES},
}


def outputs(symbol: str, env: dict, state: State | None = None) -> list[Reply]:
    """The replies a turn can draw.

    A comparison only leads into a product action and never opens a dialogue; "no match" exists only when the user is
    going to end unsuccessfully, and not on the opening or right after a reveal; the planner shows at most one planned
    wrong item."""
    positive = env['finisher'] == 'positive'
    replies = []
    for r in OUTPUTS[symbol]:
        if r.kind == 'compare' and (not positive or symbol == 'OPENER'):
            continue
        if r.kind == 'no_match' and (positive or symbol in ('OPENER', 'REVEAL')):
            continue          # a reveal needs listed items and leaves the witness acceptable, so something still matches
        if r.kind == 'recommend' and r.deliberate and state is not None and state.spent:
            continue
        replies.append(r)
    return replies


def allowed(state: State, symbol: str, env: dict) -> bool:
    """Whether the user may say this at this point."""
    if symbol == 'OPENER':
        return state.phase == 'start'
    if state.phase in ('start', 'done'):
        return False
    if state.forced is not None and symbol != state.forced:
        return False                              # the previous reply demanded a particular kind of turn
    if symbol in CLOSES:
        if symbol == 'CLOSE_NO':
            return env['finisher'] == 'negative' and state.phase == 'disclosing' and not state.withdrawn
        if env['finisher'] != 'positive':
            return False
        if symbol == 'CONFIRM':
            return state.selected
        if symbol == 'RECONSIDER':
            # The planner withdraws an alternative, never the item it keeps, so "I reconsidered it" cannot be said.
            # Its wording exists in the planner as a dead branch; turning it on means changing this rule.
            return False
        if state.selected:
            return False                          # an already selected item is confirmed, not chosen again
        return env['match'] == ('full' if symbol == 'CHOOSE' else 'partial')
    if symbol in DISCLOSURE:
        if state.phase != 'disclosing':
            return False
        if symbol == 'ANSWER':
            return state.forced == 'ANSWER'
        if symbol == 'REJECT':
            return state.deliberate
        if symbol == 'REACT':
            return state.spent and not state.deliberate     # otherwise the reply before it would have held the wrong item
        if symbol == 'REVEAL':
            return state.hidden and state.listed >= 1
        return True                               # ADD
    if env['finisher'] != 'positive':
        return False                              # product actions need an item the user can end up choosing
    if symbol == 'PREFER':
        return state.listed >= 2
    if symbol == 'INTEREST':
        return state.listed >= 1
    if symbol == 'WITHDRAW':
        return state.endorsed >= 2
    return not state.selected                     # SELECT


def forced_symbol(reply: Reply) -> str | None:
    """What the reply obliges the next user turn to be."""
    if reply.kind == 'ask_attribute':
        return 'ANSWER'
    if reply.kind == 'recommend' and reply.deliberate:
        return 'REJECT'
    if reply.kind == 'compare':
        return 'PREFER'
    return None


def advance(state: State, symbol: str, reply: Reply, *, endorsed: int | None = None, selected: bool | None = None) -> State:
    if symbol in CLOSES:
        return state._replace(phase='done', forced=None)
    phase = 'disclosing' if symbol in ('OPENER', *DISCLOSURE) else 'acting'
    return State(phase, forced_symbol(reply), reply.deliberate, reply.hidden, reply.listed,
                 min(2, max(0, state.endorsed + (1 if symbol == 'INTEREST' else 0) - (1 if symbol == 'WITHDRAW' else 0)))
                 if endorsed is None else endorsed,
                 (state.selected or symbol == 'SELECT') if selected is None else selected,
                 state.withdrawn or symbol == 'WITHDRAW', state.spent or reply.deliberate)


def cell(symbol: str, reply: Reply) -> tuple:
    """A transition for coverage: the user turn together with the kind of reply it drew."""
    return (symbol, reply.kind, '+'.join(f for f, on in (('deliberate', reply.deliberate), ('hidden', reply.hidden)) if on))


# ------------------------------------------------------------------------- reading a plan

def symbol_of(plan: dict, index: int) -> str:
    step = plan['exchanges'][index]
    if index == 0:
        return 'OPENER'
    if step.get('finisher'):
        if step['finisher'] == 'negative':
            return 'CLOSE_NO'
        text = step['shopper_instruction']
        if text.startswith('Confirm that'):
            return 'CONFIRM'
        if 'reconsidered' in text:
            return 'RECONSIDER'
        return 'CHOOSE_BEST' if (plan.get('choice') or {}).get('match') == 'partial' else 'CHOOSE'
    action = step.get('user_action')
    if action:
        return {'prefer': 'PREFER', 'express_interest': 'INTEREST', 'withdraw_interest': 'WITHDRAW',
                'select': 'SELECT'}[action]
    kind = step.get('middle_type')
    if kind == 'assistant_free':
        return 'ADD'
    if kind == 'reactive_context':
        return 'ANSWER'
    if kind == 'reveal_dislike':
        return 'REVEAL'
    if kind == 'reactive_non_context':
        return 'REJECT' if plan['exchanges'][index - 1]['assistant']['deliberate_error'] else 'REACT'
    raise ValueError(f'exchange {index + 1} has no recognisable user turn type')


def reply_of(step: dict) -> Reply:
    a = step['assistant']
    return Reply(a['kind'], bool(a['deliberate_error']), bool(a.get('hidden_violations')), min(2, len(a['product_ids'])))


def run(plan: dict) -> list[tuple]:
    """Feed a plan through the machine; return the transitions it used or raise ValueError at the first violation."""
    env = {'finisher': plan['finisher']['type'], 'match': (plan.get('choice') or {}).get('match')}
    state, used = START, []
    for index, step in enumerate(plan['exchanges']):
        symbol, reply = symbol_of(plan, index), reply_of(step)
        if not allowed(state, symbol, env):
            raise ValueError(f'exchange {index + 1}: {symbol} is not allowed in state {state}')
        if reply not in outputs(symbol, env, state):
            raise ValueError(f'exchange {index + 1}: {symbol} cannot draw the reply {reply}')
        used.append(cell(symbol, reply))
        state = advance(state, symbol, reply, endorsed=min(2, len(step.get('target_products') or ())),
                        selected=bool(step.get('selected_products')))
    if state.phase != 'done':
        raise ValueError('a dialogue ends with a closing turn')
    return used


# ------------------------------------------------------------------------- the machine on its own

def successors(state: State, env: dict):
    """Every (symbol, reply, next state) the machine allows from a state."""
    for symbol in SYMBOLS:
        if allowed(state, symbol, env):
            for reply in outputs(symbol, env, state):
                yield symbol, reply, advance(state, symbol, reply)


ENVIRONMENTS = [{'finisher': 'negative', 'match': None}, {'finisher': 'positive', 'match': 'full'},
                {'finisher': 'positive', 'match': 'partial'}]


def reachable(env: dict) -> set[State]:
    seen, queue = {START}, [START]
    while queue:
        state = queue.pop()
        for _, _, nxt in successors(state, env):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def dead_ends(env: dict) -> list[State]:
    """Reachable states other than the final one with no way to reach it: the machine could get stuck there."""
    states = reachable(env)
    can_finish = {s for s in states if s.phase == 'done'}
    changed = True
    while changed:
        changed = False
        for state in states - can_finish:
            if any(nxt in can_finish for _, _, nxt in successors(state, env)):
                can_finish.add(state)
                changed = True
    return sorted(states - can_finish, key=repr)


def all_cells() -> list[tuple]:
    """Every (user turn, reply) transition the machine can take from a reachable state, in any environment."""
    cells = set()
    for env in ENVIRONMENTS:
        for state in reachable(env):
            for symbol, reply, _ in successors(state, env):
                cells.add(cell(symbol, reply))
    return sorted(cells)


def paths(env: dict, max_turns: int):
    """Every symbol sequence of at most max_turns turns that the machine accepts and that ends in a closing turn.

    The machine is non-deterministic in its replies, so each prefix carries the set of states it can be in."""
    frontier = {(): frozenset({START})}
    for _ in range(max_turns):
        grown = {}
        for trail, states in frontier.items():
            for symbol in SYMBOLS:
                reached = frozenset(advance(s, symbol, r) for s in states if allowed(s, symbol, env)
                                    for r in outputs(symbol, env, s))
                if not reached:
                    continue
                if symbol in CLOSES:
                    yield trail + (symbol,)
                else:
                    grown[trail + (symbol,)] = reached
        frontier = grown
