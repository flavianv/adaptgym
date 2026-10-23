# AdaptGym: Benchmarking Useful Adaptation

AdaptGym is an open project for generating, evaluating, and ultimately training agents that track changing user intent. It creates conversations in which requirements accumulate, preferences evolve, and earlier information becomes invalid. Its goal is to measure whether an agent preserves the right information, updates its understanding, and uses that understanding to make better decisions.

The current application is conversational shopping. The broader direction includes personalized assistance, planning, and other tasks that require continuity across interactions.

## Two training objectives

**Supervised extraction:** a model translates the conversation so far into structured intent. It serves as a feature provider for another agent, which uses those features to retrieve products, rank recommendations, or ask follow-up questions.

**Adaptation within the agent:** a recommendation agent maintains and uses intent as part of its own capabilities. It updates its understanding as the shopper clarifies, revises, or rejects a suggestion, then chooses what to recommend or ask next.

Both objectives can use per-turn intent targets. Training the integrated agent will also require rewards for recommendation quality and task success. Correct extraction alone does not establish useful adaptation.

**Current status:** the repository implements procedural shopping scenarios, programmed assistant actions, LLM shopper generation, per-turn extraction tests, and a separate model pass for label verification. An RL optimizer, trained policy, and end-to-end recommendation training loop are not implemented.

## What the shopping environment covers

Three independent dimensions shape an episode:

- **Expression:** explicit, implicit, or mixed evidence. All modes map to predefined attributes; implicit evidence does not create new schema fields.
- **Interaction:** assistant-free changes or reactions to a programmed assistant action. A context-dependent subtype includes replies such as “yes,” “lower,” and “the second one.”
- **Scope:** durable preferences, temporary task requirements, or unknown scope.

The current shopping workflow focuses on multi-turn sessions. Multi-session shopping adaptation is a longer-term extension.

### Scenario families

**Assistant-free:** establish or add a requirement, reinforce a preference, revise a value, retract a requirement, restore an earlier value, clarify scope, change task, or introduce irrelevant information.

**Assistant-reactive:** answer a clarification, select or compare products, correct a missed preference or violated constraint, respond to no matches, accept or reject a trade-off, correct a misunderstanding, confirm or reject a suggestion, refine through a product reference, or respond to confidently presented poor recommendations.

Every chain has a positive, negative, or ambiguous ending. An ambiguous ending represents departure without a known outcome; silence is not evidence of success or rejection.

## Illustrative example

Consider a synthetic footwear catalog. Product names and offer values below are illustrative, rather than claims about real inventory.

```text
Turn 1
Shopper:   “For this purchase, I need road-running shoes in EU 42,
            no more than €150.”
Assistant: “Should I narrow the maximum price to €120?”

Turn 2
Shopper:   “Yes.”
Assistant: “Here are five matching options:
            1. Road A — €115
            2. Road B — €110
            3. Road C — €120
            4. Road D — €105
            5. Road E — €119”

Turn 3
Shopper:   “The second one.”
Assistant: “Road B, at €110. Does that finish your search?”

Turn 4
Shopper:   “Yes, that's what I wanted. I'm done shopping.”
```

Expected intent is recorded **after each shopper message, before the next assistant reply**:

1. Road-running shoes, size EU 42, price ≤ €150; all are requirements for the current task.
2. Preserve shoe type and size; replace the price maximum with €120. The evidence is the assistant's question plus the shopper's “Yes.”
3. Preserve the requirements and record Road B as the selected product. Resolve “the second one” against the preceding list; do not turn every attribute of Road B into a durable preference.
4. End the task and clear its temporary requirements. Record `satisfied: true` and retain Road B as `target_product`; unrelated durable preferences, if present, remain.

For supervised extraction, each conversation prefix is paired with its expected current intent. For adaptation supervision, the same checkpoints form transitions: **previous intent + new evidence → updated intent**.

A separate verifier checks whether the generated wording supports those labels. If Turn 1 instead says “Around €150, but I can spend more,” a hard €150 maximum should be flagged. The intended scenario is not sufficient evidence that its rendered dialogue is correct.

## How generation and verification work

The environment separates general user context from product attributes:

- **Persona schema and persona:** general background and non-product context. The demo uses a fixed debug persona.
- **Vertical schema:** product attributes, types, allowed values, units, and operators.
- **Catalog:** schema-conforming products with fixed descriptions and configurable offer attributes.
- **Session scenario:** a procedural plan of shopper events, assistant actions, scope, and intended state changes.
- **Answer keys:** expected intent states with evidence provenance at each shopper checkpoint.

```text
CODE: sample a need and build the session scenario
CODE: compute intended state transitions and checkpoint answer keys

FOR EACH exchange:
    SHOPPER LLM: render the current instruction using the persona
                 and latest assistant reply
    EXTRACTOR LLM: predict intent from the full visible conversation
                   and its own previous valid prediction
    CODE: score extraction against the planned target
    VERIFIER LLM: audit the candidate target against the visible prefix
    JUDGE LLM: audit the model prediction against the same prefix
    CODE: validate audits and record their verdicts
    CODE: append the programmed assistant reply
```

The assistant is programmed so scenarios can deliberately supply suitable products, violate a constraint, ask a question, or present a trade-off. The shopper LLM generates wording, not the scenario logic or assistant behavior.

### Search requests as part of intent

The extractor also reports `satisfied` (`true`, `false`, or `"unk"`) and `target_product` (an explicitly identified catalog ID or null). Satisfaction comes from shopper evidence, not assistant framing. The single `target_product` field is populated as soon as the shopper identifies a product, retained through completion, and reset to null when a new task starts. There is no separate selection field. A new task also resets satisfaction to `"unk"`.

The extractor returns a plain-language `intent` summary and `search_request` alongside its claims: a natural-language `query`, typed hard `filters`, and soft `preferences`. Task-specific values override durable values of the same attribute; unknown scope or strength does not silently become a search constraint. No active task or an already-selected product produces a null search request.

The dashboard shows a reference request and the model's request at each turn. Structured constraints are checked deterministically; a separate model judge assesses the whole extracted intent and query semantically, allowing equivalent wording. These are provider-neutral request previews; no external search tool is invoked.

### Independent label verification

The verifier receives the schema, catalog, visible conversation prefix, and candidate intent. It does **not** receive private scenario instructions, future messages, or extractor predictions.

It checks:

- Each claim's value, operator, scope, strength, and evidence classification.
- Supporting message references, including assistant context when needed.
- Active task and product selection.
- Current shopper intent missing from the candidate labels.

Claim verdicts are **supported**, **contradicted**, or **insufficient evidence**. A checkpoint passes the model audit only when all checks are supported and no omitted intent is reported. Failed audits remain unverified. The implementation flags examples for review; it does not automatically repair them.

This is a separate model pass, which can use the same underlying model as the generator. It is not human certification or a guarantee of correctness. Planned targets, audit verdicts, and extractor scores remain separately inspectable.

## Quick start

Install the project and development dependencies:

```bash
python3 -m pip install -e ".[dev]"
```

### Open the session laboratory

```bash
python3 scripts/debug_session_scenarios.py --serve --port 8768
```

Open <http://127.0.0.1:8768/>. A saved, model-verified three-turn product-selection example (request → choice → satisfied completion with a retained `target_product`) is preloaded from [featured_run.json](examples/debug_shopping/featured_run.json), including intent, search requests, judgments, and outcome. Exploring it and planning new scenarios require no API key. New live runs replace it for the current server session; restarting restores the saved example.

To enable live generation, extraction, and verification, supply `OPENAI_API_KEY` through the environment or an ignored local `.env` file:

```bash
python3 scripts/debug_session_scenarios.py --serve --port 8768 --env-file .env
```

Enter a model available to your account in the dashboard, build a scenario, then select **Generate & test**. A completed run makes up to four API calls per shopper turn: shopper generation, intent extraction, label verification, and a separate semantic judgment of the extracted intent.

The dashboard provides:

- **Conversation:** scenario controls and rendered dialogue side by side.
- **Fine-tuning:** per-turn supervised examples, adaptation transitions, model predictions, and label-verification reports.
- **Catalog:** products and their relationship to the planned requirements.
- **Inputs:** the persona and vertical schema.
- **LLM-Judge:** the actual backend conversation-generator and verifier prompts.

The debug catalog contains 30 sports-footwear models and an optional distractor pool. Offer size, price, colour, and gender are synthetic. Model names and descriptions are fixtures, not a live retail inventory.

### Generate plans without API calls

```bash
python3 scripts/debug_session_scenarios.py \
  --case clarification_question --turns 3 --finisher positive \
  --seed 7 --output runs/example_plan.json
```

See [the debug laboratory guide](examples/debug_shopping/README.md) for more options. For hosting, use [the Render blueprint](render.yaml) and [deployment guide](docs/render_deployment.md).

## Repository structure

- `src/adaptgym/`: catalog generation, schema validation, current needs, scenario planning, intent state, search requests, and LLM evaluation.
- `scripts/debug_session_scenarios.py`: command-line runner and dashboard server.
- `examples/debug_shopping/`: synthetic persona, footwear schema, catalog templates, dashboard, and saved example.
- `tests/`: regression tests for the current pipeline and server.
- `docs/`: [pipeline architecture](docs/architecture.md) and [deployment](docs/render_deployment.md).

This repository contains one shopping pipeline. Planning is procedural; shopper wording, extraction, and semantic verification use LLM calls. It does not include an RL trainer or earlier experimental implementations.

## Development and next steps

```bash
python3 -m pytest -q
```

Next steps include stronger label validation, reproducible held-out episode splits, a sequential training adapter, and integration with a model-training harness. Downstream evaluation will need to measure constraint satisfaction, recommendation relevance, and task outcomes alongside extraction quality.

The guiding distinction is simple: **data quality, intent quality, and decision quality are separate measurements.** AdaptGym's goal is to connect them without mistaking an accurate state representation for successful user assistance.
