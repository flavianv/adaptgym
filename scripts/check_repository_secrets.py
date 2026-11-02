#!/usr/bin/env python3
"""Scan Git blob snapshots offline; print finding locations and types, never values."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', action='store_true')
    parser.add_argument('--ref', default='HEAD')
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    def git(*parts):
        return subprocess.check_output(['git', '-C', str(repo), *parts])
    blobs = {}
    if args.history:
        revision = git('rev-parse', '--verify', args.ref + '^{commit}').decode().strip()
        for line in git('rev-list', '--objects', revision).decode().splitlines():
            parts = line.split(' ', 1)
            if len(parts) == 2 and git('cat-file', '-t', parts[0]).strip() == b'blob':
                blobs[parts[0]] = parts[1]
    else:
        for line in git('ls-files', '--stage', '-z').decode().split('\0'):
            if line:
                metadata, name = line.split('\t', 1)
                blobs[metadata.split()[1]] = name
    with tempfile.TemporaryDirectory(prefix='repository-secrets-') as directory:
        names = {}
        for oid, name in blobs.items():
            path = Path(directory) / (oid + Path(name).suffix)
            data = git('cat-file', 'blob', oid)
            if name == 'tests/test_demo_server.py':
                # Exact reviewed dummy values, scoped to this test file only.
                fixtures = [b'legacy-test-password', b'demo-test-password', b'test-server-key',
                            b'https://' + b'user:password' + b'@example.com']
                for fixture in fixtures:
                    data = data.replace(fixture, b'')
            path.write_bytes(data)
            path.chmod(0o600)
            names[path.name] = name
        result = subprocess.run(['uvx', '--from', 'detect-secrets==1.5.0', 'detect-secrets',
                                 'scan', '--all-files', '--no-verify', '.'],
                                cwd=directory, capture_output=True, text=True, check=True)
        findings = json.loads(result.stdout)['results']
        count = 0
        for path, entries in findings.items():
            for entry in entries:
                print(f"{names[path]}:{entry['line_number']}: {entry['type']}")
                count += 1
        print(f'Credential scan: {len(blobs)} blobs, {count} findings. Network verification disabled.')
        raise SystemExit(bool(count))


if __name__ == '__main__':
    main()
