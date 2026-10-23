# AdaptGym

Synthetic conversations for evaluating how agents update user intent across turns.
Tracks task requirements, durable preferences, search requests, and expressed outcomes.
Includes Fashion/shoes and Job search with synthetic catalog data.

## Run

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/debug_session_scenarios.py --serve --port 8769
```

Open http://127.0.0.1:8769/ and choose a vertical above the tabs.
Live model calls require `OPENAI_API_KEY` and a model available to your account.
Planning and the saved example work without an API key.

```bash
python3 scripts/debug_session_scenarios.py --vertical examples/job_search --case all
pytest -q
```

A vertical supplies `domain_schema.json`, `simulation.json`, `user_persona.json`,
and `catalog_template.json`. Optional distractor catalogs and saved runs are supported.
The simulator uses scalar attributes and programmed assistant actions; model training
and live application submission are not implemented.

## Before publishing

Enable local privacy and credential checks once:

```bash
git config core.hooksPath .githooks
```

Hooks require Python and `uvx`. Internal notes, real user data, credentials, and
private infrastructure references do not belong in this public repository.
