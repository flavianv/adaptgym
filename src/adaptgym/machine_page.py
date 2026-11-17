"""Data for the simulator state machine page, computed from ``turn_machine`` so the page cannot drift from the code."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from adaptgym import turn_machine as machine

LABELS = {('negative', None): 'Ends unsuccessfully', ('positive', 'full'): 'Ends with a full match',
          ('positive', 'partial'): 'Ends with a partial match'}


@lru_cache(maxsize=1)
def machine_data() -> dict[str, Any]:
    """Every reachable state and transition, per kind of ending, in a compact form the page walks."""
    replies: list[list] = []
    reply_ids: dict = {}

    def reply_id(reply) -> int:
        if reply not in reply_ids:
            reply_ids[reply] = len(replies)
            replies.append([reply.kind, reply.deliberate, reply.hidden, reply.listed])
        return reply_ids[reply]

    envs = {}
    for env in machine.ENVIRONMENTS:
        ids, states, queue, trans = {machine.START: 0}, [machine.START], [machine.START], {}
        while queue:
            state = queue.pop(0)
            out = []
            for symbol, reply, nxt in machine.successors(state, env):
                if nxt not in ids:
                    ids[nxt] = len(states)
                    states.append(nxt)
                    queue.append(nxt)
                out.append([machine.SYMBOLS.index(symbol), reply_id(reply), ids[nxt]])
            trans[ids[state]] = out
        envs[f"{env['finisher']}-{env['match']}"] = {
            'label': LABELS[(env['finisher'], env['match'])], 'states': [list(s) for s in states],
            'trans': [trans[i] for i in range(len(states))], 'deadEnds': len(machine.dead_ends(env)),
            'paths6': sum(1 for _ in machine.paths(env, 6))}
    return {'symbols': list(machine.SYMBOLS), 'disclosure': list(machine.DISCLOSURE), 'actions': list(machine.ACTIONS),
            'closes': list(machine.CLOSES), 'stateFields': list(machine.State._fields), 'replies': replies, 'envs': envs,
            'cells': [list(c) for c in machine.all_cells()]}
