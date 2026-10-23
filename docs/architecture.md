# Current pipeline

AdaptGym contains one shopping-session implementation. The dashboard and command-line runner use the same planning and evaluation functions.

## Definitions

- **General persona:** synthetic background context, validated by `persona.py`. The demo loads a fixed persona; it does not generate personas through an LLM.
- **Vertical schema:** product attributes with types, allowed values, units, and comparison operators, validated by `verticals.py`.
- **Catalog template:** fixed model properties and description text, plus configurable offer values. `catalog.py` builds and validates catalog items.
- **Current need:** a seeded, catalog-backed choice of shoe type and requirements, sampled by `current_need.py`.
- **Session scenario:** the procedural sequence of shopper events and programmed assistant actions in `session_plans.py`.
- **Checkpoint:** the reference intent after a shopper message, before the following assistant answer. `intent_state.py` applies the planned transitions; `search_requests.py` derives search and outcome targets.

## Generation

```text
LOAD persona, vertical schema, and catalog template
BUILD catalog and validate inputs
SAMPLE a current need with a seeded random generator
PLAN shopper events, assistant actions, and an ending
APPLY intended state changes to build private checkpoint labels
VALIDATE scenario, product references, and evidence references

FOR EACH planned exchange
    SHOPPER LLM renders the current instruction
        using the persona and latest assistant reply
    APPEND the shopper message to the visible conversation
    EVALUATE this checkpoint
    APPEND the programmed assistant answer
```

Scenario families include independent changes, assistant reactions, and context-dependent reactions. Explicit/implicit/mixed expression is an independent setting. Chains end positively, negatively, or ambiguously. A selected product is retained through completion and reset when another task begins.

## Evaluation

```text
FOR EACH shopper checkpoint
    EXTRACTOR LLM receives the full visible prefix, schema, catalog,
        and its own previous valid prediction
    CODE validates the output and compares structured fields to the reference
    VERIFIER LLM checks whether the prefix supports the reference labels
    JUDGE LLM checks whether the prefix supports the model prediction
    SAVE predictions, targets, evidence, verdicts, and scores separately
```

The extractor never receives the private scenario or reference state. Its previous prediction may be wrong; valid means structurally valid, not necessarily correct. The verifier and judge receive candidate labels and the visible prefix, not future turns or private instructions. They can use the same underlying model; separate calls do not guarantee correctness.

Intent includes a summary, evidence-backed claims, a search request, satisfaction, and `target_product`. Search requests distinguish hard filters from soft preferences. Completed tasks and product selections do not require a new search request. Satisfaction is `true`, `false`, or `"unk"`; departure does not imply success.

## Artifacts and training

`render_and_test` returns the conversation and per-turn checkpoints. The CLI can save these along with the plan and API traces. The Fine-tuning tab displays context, reference targets, model outputs, and independent label checks. Unverified targets remain visibly distinct from accepted audits.

The current implementation generates candidate training examples and evaluates intent extraction. It does not train model weights, run an RL optimizer, call a live product-search service, or optimize the recommendation policy. Those integrations will need their own data splits, training adapter, and downstream reward design.
