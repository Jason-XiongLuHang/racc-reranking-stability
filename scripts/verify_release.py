"""Verify the public allowlist, common secret/path patterns and release hashes."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {'.md', '.json', '.py', '.csv', '.cff', '.txt'}
SPECIAL = {'LICENSE', '.gitignore', '.gitattributes'}
# Numeric hashes and the public author name are intentional public metadata.
PATTERNS = {
    'email': r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',
    'absolute_windows_path': r'(?i)(?<![a-z])[a-z]:[/\\]',
    'absolute_runtime_path': r'(?i)/(?:content|mnt/workspace|home|Users|root|tmp|workspace)/',
    'private_service_path': r'(?i)(?:MyDrive|paperpass\.com/report/)',
    'private_key': r'-----BEGIN [A-Z ]*PRIVATE KEY-----',
    'service_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AIza[A-Za-z0-9_-]{25,}|sk-[A-Za-z0-9_-]{20,}|AKIA[A-Z0-9]{16})\b',
    'credential_assignment': r'(?i)(?:api_key|access_token|password|client_secret)\s*[:=]\s*[\"\'][^\"\']{8,}[\"\']',
}


def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not {'.git', '__pycache__'} & set(p.relative_to(ROOT).parts))


def scan():
    issues = []
    for path in files():
        relative = path.relative_to(ROOT).as_posix()
        if path.name not in SPECIAL and path.suffix not in ALLOWED:
            issues.append((relative, 'non-allowlisted file type'))
            continue
        if path.stat().st_size > 2_000_000:
            issues.append((relative, 'unexpected large file'))
        text = path.read_text(encoding='utf-8')
        # Pattern definitions themselves contain examples, not credentials.
        if relative != 'scripts/verify_release.py':
            for kind, pattern in PATTERNS.items():
                if re.search(pattern, text):
                    issues.append((relative, kind))
        if path.suffix == '.json':
            json.loads(text)
        if path.suffix == '.py':
            compile(text, relative, 'exec')
    return issues


def main():
    issues = scan()
    manifest = json.loads((ROOT / 'manifests/public_file_hashes.json').read_text())
    actual = {p.relative_to(ROOT).as_posix() for p in files()}
    assert actual == set(manifest['files']) | {'manifests/public_file_hashes.json'}, 'File allowlist changed'
    for name, expected in manifest['files'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts, 'Unsafe identity'
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, name
    assert not issues, issues
    print(f'PUBLIC_RELEASE_AUDIT_PASS: {len(actual)} public files; hashes, formats and privacy scan pass')


if __name__ == '__main__':
    main()
