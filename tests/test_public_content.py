"""Keep privacy checks useful without putting real sensitive data in fixtures."""
import importlib.util
from pathlib import Path
import subprocess


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/check_public_content.py'
spec = importlib.util.spec_from_file_location('public_content', SCRIPT)
privacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(privacy)


def test_private_files_and_local_paths_are_rejected():
    assert privacy.content_findings('.env', b'') == {'private file path'}
    assert privacy.content_findings('secrets/config.json', b'') == {'private file path'}
    local = '/'.join(['', 'Users', 'example', 'workspace'])
    assert 'personal filesystem path' in privacy.content_findings('readme.md', local.encode())
    assert privacy.content_findings('.env.example', b'OPENAI_API_KEY=') == set()


def test_internal_links_are_rejected_and_public_schema_links_are_allowed():
    private_host = '.'.join(['example', 'atlassian', 'net'])
    assert 'internal infrastructure' in privacy.content_findings('schema.json', private_host.encode())
    assert privacy.content_findings('schema.json', b'https://json-schema.org/draft/2020-12/schema') == set()


def test_obvious_credential_is_flagged_without_using_a_real_token():
    synthetic = 'sk-' + 'x' * 24
    assert privacy.content_findings('config.txt', synthetic.encode()) == {'API credential'}


def test_staged_check_reads_index_even_when_worktree_has_been_cleaned(tmp_path):
    def git(*args):
        subprocess.run(['git', '-C', str(tmp_path), *args], check=True, capture_output=True)
    git('init')
    git('config', 'user.name', 'Example Developer')
    git('config', 'user.email', 'developer@example.com')
    path = tmp_path / 'notes.txt'
    path.write_text('.'.join(['example', 'atlassian', 'net']))
    git('add', 'notes.txt')
    path.write_text('public example')
    assert privacy.audit(tmp_path)[1] == set()
    assert privacy.audit(tmp_path, staged=True)[1] == {('notes.txt', 'internal infrastructure')}
