#!/usr/bin/env python3
"""Reject private infrastructure references and obvious credentials without printing values.

Run before publishing; --history also examines reachable blobs and commit metadata.
This complements a dedicated secret scanner rather than replacing one.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess


RULES = {
    'internal infrastructure': re.compile(
        r'(?i)\b[\w.-]+\.(?:atlassian\.net|crto\.in|ctolabperformance\.com)\b'),
    'personal filesystem path': re.compile(r'/(?:Users|home)/[\w.-]+/'),
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'API credential': re.compile(r'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}'),
    'cloud credential': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'GitHub credential': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})'),
    'GitLab credential': re.compile(r'\bglpat-[A-Za-z0-9_-]{20,}'),
}
PRIVATE_PATH = re.compile(
    r'(^|/)(?:\.env(?:\..*)?|\.netrc|secrets|private|id_rsa|id_ed25519)(/|$)|\.(?:pem|key|p12|pfx)$')


def content_findings(name: str, content: bytes) -> set[str]:
    issues = set()
    if PRIVATE_PATH.search(name) and Path(name).name != '.env.example':
        issues.add('private file path')
    text = content.decode('utf-8', errors='replace')
    issues.update(label for label, pattern in RULES.items() if pattern.search(text))
    return issues


def audit(repo: Path, history: bool = False, staged: bool = False, ref: str = 'HEAD') -> tuple[int, set[tuple[str, str]]]:
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repo), *args])
    findings = set()
    count = 0
    if history:
        revision = git('rev-parse', '--verify', ref + '^{commit}').decode().strip()
        for line in git('rev-list', '--objects', revision).decode().splitlines():
            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue
            oid, name = parts
            if git('cat-file', '-t', oid).strip() != b'blob':
                continue
            count += 1
            findings.update((name, issue) for issue in content_findings(name, git('cat-file', 'blob', oid)))
        for line in git('log', revision, '--format=%H%x09%ae%x09%ce').decode().splitlines():
            oid, metadata = line.split('\t', 1)
            if RULES['internal infrastructure'].search(metadata):
                findings.add((oid[:12], 'private commit email'))
    elif staged:
        for entry in git('ls-files', '--stage', '-z').decode().split('\0'):
            if not entry:
                continue
            metadata, name = entry.split('\t', 1)
            mode, oid, stage = metadata.split()
            if stage != '0':
                findings.add((name, 'unresolved merge'))
                continue
            count += 1
            findings.update((name, issue) for issue in content_findings(name, git('cat-file', 'blob', oid)))
        for identity in ('GIT_AUTHOR_IDENT', 'GIT_COMMITTER_IDENT'):
            if RULES['internal infrastructure'].search(git('var', identity).decode()):
                findings.add((identity, 'private commit email'))
    else:
        for name in git('ls-files', '-z').decode().split('\0'):
            if not name or not (repo / name).is_file():
                continue
            count += 1
            findings.update((name, issue) for issue in content_findings(name, (repo / name).read_bytes()))
    return count, findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--staged', action='store_true', help='scan index contents and next-commit identities')
    parser.add_argument('--ref', default='HEAD', help='published commit whose history should be scanned')
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    args = parser.parse_args()
    count, findings = audit(args.repo.resolve(), args.history, args.staged, args.ref)
    for name, issue in sorted(findings):
        print(f'{name}: {issue}')
    print(f'Checked {count} files/blobs; {len(findings)} privacy findings. Values are never printed.')
    raise SystemExit(bool(findings))


if __name__ == '__main__':
    main()
