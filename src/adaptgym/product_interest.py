"""User-evidenced shortlist and selection, independent of catalog eligibility."""
from __future__ import annotations

import copy

PRODUCT_ACTIONS = ('express_interest', 'withdraw_interest', 'select')


def selected_products(record):
    """Read current lists or a legacy single-selection record without rewriting it."""
    if 'selected_products' in record:
        return list(record['selected_products'])
    value = record.get('target_product')
    return [value] if value else []


def target_products(record):
    return list(record['target_products']) if 'target_products' in record else selected_products(record)


class ProductInterest:
    def __init__(self):
        self.targets = []
        self.selected = []
        self.events = []
        self.preferences = []

    def apply(self, action, ids, *, shown, message_id):
        if action not in PRODUCT_ACTIONS:
            raise ValueError('unknown product action')
        if (not isinstance(ids, list) or not ids or any(not isinstance(x, str) or not x for x in ids)
                or len(ids) != len(set(ids))):
            raise ValueError('product actions need distinct product IDs')
        if not message_id or not set(ids) <= set(shown):
            raise ValueError('product actions require user evidence and previously displayed items')
        if action == 'withdraw_interest':
            if not set(ids) <= set(self.targets):
                raise ValueError('cannot withdraw interest that was never expressed')
            self.targets = [x for x in self.targets if x not in ids]
            self.selected = [x for x in self.selected if x not in ids]
        else:
            self.targets.extend(x for x in ids if x not in self.targets)
            if action == 'select':
                self.selected = list(ids)
        event = {'action': action, 'product_ids': list(ids), 'message_ids': [message_id]}
        self.events.append(event)
        return copy.deepcopy(event)

    def prefer(self, preferred, over, *, shown, message_id):
        """Record only the comparison set supported by this user message.

        A reversed pair revises the previous preference. Do not infer transitive
        pairs, selection, rejection, or durable attribute preferences.
        """
        if (not isinstance(over, list) or not over or len(over) != len(set(over))
                or not isinstance(preferred, str) or not preferred
                or any(not isinstance(x, str) or not x for x in over)
                or preferred in over or not message_id
                or not {preferred, *over} <= set(shown)):
            raise ValueError('preferences need distinct, displayed alternatives and user evidence')
        pairs = [p for p in self.preferences
                 if not (p['over'] == preferred and p['preferred'] in over)]
        for other in over:
            pair = {'preferred': preferred, 'over': other}
            if pair not in pairs:
                pairs.append(pair)
        # Check before mutation: inconsistent comparison cycles are not valid labels.
        graph = {}
        for pair in pairs:
            graph.setdefault(pair['preferred'], set()).add(pair['over'])
        def visit(node, path):
            if node in path:
                raise ValueError('cyclic item preferences')
            for nxt in graph.get(node, ()):
                visit(nxt, path | {node})
        for node in graph:
            visit(node, set())
        self.preferences = pairs
        self.targets.extend(x for x in [preferred] if x not in self.targets)
        event = {'action': 'prefer', 'preferred': preferred, 'over': list(over),
                 'message_ids': [message_id]}
        self.events.append(event)
        return copy.deepcopy(event)

    def snapshot(self):
        return {'target_products': list(self.targets), 'selected_products': list(self.selected),
                'preferences': copy.deepcopy(self.preferences)}
