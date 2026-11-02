# Publication checks

Before every commit or push, apply `public-repo-privacy` from
`~/.codex/skills/public-repo-privacy/SKILL.md` when available.
This repo is public. Keep detailed docs and internal notes local.
Before committing, run `python3 scripts/check_public_content.py --staged` and
`python3 scripts/check_repository_secrets.py`. Before pushing, run both with
`--history --ref COMMIT` for each outgoing commit. Never bypass failed checks.
