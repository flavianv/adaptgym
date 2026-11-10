"""Shared intent semantics for planning, generation, audit and training export."""
CONTRACT_VERSION = 'intent-contract-v1'
INTENT_CONTRACT = '''Intent contract:
- Attribute requirements and durable attribute preferences belong in claims. An attribute preference
  (e.g. generally preferring a brand or work arrangement) NEVER creates an item-to-item ranking.
- target_products is an explicitly endorsed shortlist of catalog ENTRY IDs, not all eligible items.
  Asking for a comparison, mentioning a model, or requiring its attributes does not endorse an entry,
  even if only one catalog entry currently matches. Catalog descriptions alone establish no user facts.
- selected_products requires an explicit choice of a catalog entry, not mere liking or preference.
  Selection alone neither completes a task nor establishes satisfaction.
- preferences contains only directed {preferred: catalog-entry-ID, over: catalog-entry-ID} pairs.
  "I prefer the first one" after a clear comparison ranks it above the other compared entries.
  It endorses that entry without selecting it or rejecting the others. Attribute values are never IDs.
- Withdrawing interest removes only the identified entry from the shortlist/selection. It does not
  retract a model, brand or other attribute requirement unless the user separately says so. Wording
  must distinguish a particular listing from the model or class of items. New tasks reset item state.
- satisfied describes the overall task outcome: true for explicit overall success, false for explicit
  overall failure, and "unk" if not established. Rejecting one recommendation while continuing the
  search does not by itself establish an overall failure. Never infer an outcome from assistant tone.
- Do not require redundant re-encodings of the same fact in additional_intent. For example,
  a captured specific sport need does not also require an equivalent category label. A new sport
  mentioned in the conversation is still missing if the candidate records only a surface.
  An explicitly stated intended use that already names the activity (e.g. primary_use=Sprinting)
  captures that activity; do not demand an additional sport=sprinting claim for the same fact.
  A road surface alone, however, does not establish road running rather than walking.
- Labels must cover ALL current schema-defined intent established by the visible USER conversation,
  including unambiguous implications. Do not invent constraints from persona background, a matching
  catalog entry, or an assistant's suggestion. Retire task claims on completion; retain durable claims.
'''


def content_hash(value):
    import hashlib
    import json
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
