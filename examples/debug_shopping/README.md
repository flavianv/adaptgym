# Shopping laboratory fixtures

This directory contains the inputs and browser UI for the current shopping pipeline:

- `user_persona.json`: a fixed synthetic general persona, with no product attributes.
- `domain_schema.json`: footwear attributes, types, units, and operators.
- `catalog_template.json`: 30 distinct sports-footwear models with fixed properties and description templates; offer size, price, colour, and gender can vary.
- `catalog_30.json`: the default materialized catalog, checked against the template in tests.
- `distractor_pool.json`: 30 additional footwear items available with `--include-distractors`.
- `featured_run.json`: saved product-selection conversation, model predictions, and verifier results. The final turn shows `satisfied: true` and a retained `target_product`.
- `viewer.html`: the dashboard served by the Python runner.

Names and product descriptions are test fixtures, not a live inventory. Offer attributes are synthetic. The saved run contains synthetic conversation data and API response metadata, never API credentials.

From the repository root:

```bash
python scripts/debug_session_scenarios.py --serve --port 8768
```

Open http://127.0.0.1:8768/. Inspecting the saved example or building scenarios requires no API key. To run the shopper, extractor, label verifier, and prediction judge, provide `OPENAI_API_KEY`, select a model, and click **Generate & test**. Each shopper turn uses up to four API calls. A run replaces the saved example in server memory; restarting restores it.

Generate plans without API calls:

```bash
python scripts/debug_session_scenarios.py --case clarification_question --turns 3 --finisher positive --seed 7 --output runs/plan.json
```

Run a live case and save its results:

```bash
python scripts/debug_session_scenarios.py --case select_product --turns 3 --finisher positive --live --model gpt-4.1-mini --env-file .env --output runs/live.json
```

Use `--expression explicit`, `implicit`, or `mixed` independently of the scenario family. `--turns` accepts 1–7 exchanges, subject to the selected scenario's minimum. `--finisher` accepts `positive`, `negative`, `ambiguous`, or `auto`. With `--case all`, plan generation enumerates every case; live runs require one case to bound API usage.

For controlled catalog experiments, `--group-overrides` accepts a JSON file mapping template group names to offer attribute overrides. Those controls remain in code and the CLI rather than the main dashboard.

See [the pipeline architecture](../../docs/architecture.md) for the boundary between procedural answer keys and LLM verification.
