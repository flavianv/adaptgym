# AdaptGym

**Control what changes. Measure what the model remembers.**

A recommendation agent can understand the first request and still lose track when a
budget changes, another requirement arrives, or a reply refers to an earlier result.
AdaptGym tests whether a model retains, revises and discards the right user information
throughout a conversation.

[Project overview and evaluation/training flows](documentation/project-overview.md)

Planning and the preloaded example make no model calls. Live generation uses the
API key configured on your server.

## How it works

1. **Generate intent traces procedurally.** Control conversation length, disclosure
   timing, revisions and scripted endings, with a reference state after every user turn.
2. **Render natural conversations.** An LLM expresses the planned user events around
   programmed assistant questions and recommendations.
3. **Verify information preservation.** A separate LLM pass checks whether the visible
   conversation supports the intended labels. This is a quality check, not a guarantee.
4. **Extract user intent at each turn.** Recover current requirements and longer-term
   preferences without seeing the current target.
5. **Score extraction.** Compare predictions with the reference, keeping matching rules
   explicit. Per-turn tests reveal errors that a correct final reply can conceal.

The tests cover requirement retention, revisions, reference resolution, task boundaries,
and the distinction between task-specific needs and general preferences. The same
conversation–target pairs can provide supervision for fine-tuning.

**Available today:** footwear shopping and job-search environments, synthetic sessions,
label checks, and extraction tests. Assistant actions are programmed. SFT and RLVR
training loops, and evaluation of live recommendation decisions, are proposed extensions.
Scripted success and failure endings are scenario controls, not agent performance scores.

The lab offers two previous-state modes:

- **Accumulated:** use the model’s last valid prediction, exposing error accumulation
  and recovery.
- **Previous correct:** use the preceding reference state to isolate the current update.

Both receive the conversation prefix; neither receives the current or future target.
The default example includes its saved conversation and all three validation checkpoints.

Read the [project overview](documentation/project-overview.md) for the evaluation design,
matching policies and proposed training flows. Related work includes
[Spotify’s synthetic conversations and self-improvement loops](https://research.atspotify.com/2026/9/bootstrapping-conversational-recommendation-agents-at-spotify-synthetic-data-generation-and-self-improvement-loops).

## Run

```bash
python3 -m pip install -e ".[dev]"
python3 scripts/debug_session_scenarios.py --serve --port 8769
```

Open http://127.0.0.1:8769/ and choose a vertical above the tabs.
The session lab is the landing page (also `/lab`); `/about` explains the project,
and `/flows` displays the evaluation and proposed training diagrams.
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
