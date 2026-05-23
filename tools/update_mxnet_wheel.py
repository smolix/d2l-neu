#!/usr/bin/env python3
"""Update pyproject.toml's mxnet wheel pin to the latest release.

Queries `https://api.github.com/repos/smolix/mxnet/releases/latest` for
the most recent published release, finds the cp312-cp312-linux_x86_64
wheel asset, and rewrites the `mxnet @ <url>` line in pyproject.toml in
place. Idempotent: skips the write (and exits 0) when already on the
latest wheel. The line being replaced may currently be either an
`https://` URL (the usual case) or a `file://` URL (when a local build
was pinned manually); both forms are recognised.

Usage:
    python3 tools/update_mxnet_wheel.py            # update + report
    python3 tools/update_mxnet_wheel.py --dry-run  # report only

After updating, run `make venv-mxnet` to apply the bump.

Why not auto-pull on every sync: makes builds non-reproducible, hits
the unauthenticated GitHub rate limit (60/hour), and breaks offline
builds. Make this an explicit step.
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = 'smolix/mxnet'
ASSET_SUFFIX = '-cp312-cp312-linux_x86_64.whl'
PYPROJECT = Path(__file__).resolve().parent.parent / 'pyproject.toml'
# Matches the entire mxnet wheel line including the python_version marker:
#   "mxnet @ https://… ; python_version == '3.12'"
# Also accepts `file://` so a manually-pinned local wheel can be replaced.
LINE_RE = re.compile(
    r'"mxnet @ (?:https|file)://[^"]+'
    r'(\s*;\s*python_version\s*==\s*\'3\.\d+\')?"',
    re.MULTILINE)


def find_latest_wheel():
    """Return (tag, wheel_name, wheel_url) for the latest cp312 wheel."""
    api = f'https://api.github.com/repos/{REPO}/releases/latest'
    req = urllib.request.Request(api, headers={'Accept': 'application/vnd.github+json'})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 403:
            sys.exit('GitHub API rate-limited (403). Wait or set GITHUB_TOKEN.')
        raise
    tag = data.get('tag_name', '?')
    for asset in data.get('assets', []):
        name = asset.get('name', '')
        if name.endswith(ASSET_SUFFIX):
            return tag, name, asset['browser_download_url']
    sys.exit(f'No {ASSET_SUFFIX} asset in release {tag!r}; '
             f'assets present: {[a["name"] for a in data.get("assets", [])]}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would change without writing.')
    args = parser.parse_args()

    tag, name, url = find_latest_wheel()
    print(f'Latest release: {tag}')
    print(f'Wheel: {name}')
    print(f'URL:   {url}')

    text = PYPROJECT.read_text()
    m = LINE_RE.search(text)
    if not m:
        sys.exit(f'No `"mxnet @ https://…"` line found in {PYPROJECT}')
    current = m.group(0)
    new_line = f'"mxnet @ {url} ; python_version == \'3.12\'"'
    if current == new_line:
        print('pyproject.toml already pinned to latest — no change.')
        return
    print('\n--- pyproject.toml diff ---')
    print(f'- {current}')
    print(f'+ {new_line}')
    if args.dry_run:
        print('\n(dry-run; not written)')
        return
    new_text = text[:m.start()] + new_line + text[m.end():]
    PYPROJECT.write_text(new_text)
    print(f'\nUpdated {PYPROJECT}. Run `make venv-mxnet` to apply.')


if __name__ == '__main__':
    main()
