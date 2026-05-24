#!/usr/bin/env python3
"""Update pyproject.toml's mxnet wheel pin to the latest available wheel.

Two sources:

  --source local   (default) — scan `../mxnet/dist/` for the newest
                   `*-cp312-cp312-linux_x86_64.whl` and pin it via
                   `file://`. Use this while the custom MXNet build is
                   still iterating locally and outpacing GitHub releases.

  --source github  — query `https://api.github.com/repos/smolix/mxnet/
                   releases/latest` and pin the `https://` download URL.
                   Use this once the build stabilises and we want
                   reproducible pins from published releases.

The line being replaced may currently be either `https://` or `file://`;
both forms are recognised. Idempotent: skips the write (and exits 0)
when already on the latest wheel.

Usage:
    python3 tools/update_mxnet_wheel.py                    # local, update + report
    python3 tools/update_mxnet_wheel.py --dry-run          # local, report only
    python3 tools/update_mxnet_wheel.py --source github    # github source

After updating, run `uv lock && make venv-mxnet` to apply the bump.
"""
import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO = 'smolix/mxnet'
ASSET_SUFFIX = '-cp312-cp312-linux_x86_64.whl'
PYPROJECT = Path(__file__).resolve().parent.parent / 'pyproject.toml'
DEFAULT_LOCAL_DIST = Path(__file__).resolve().parent.parent.parent / 'mxnet' / 'dist'

# Matches the entire mxnet wheel line including the python_version marker:
#   "mxnet @ https://… ; python_version == '3.12'"
# Also accepts `file://` so a manually-pinned local wheel can be replaced.
LINE_RE = re.compile(
    r'"mxnet @ (?:https|file)://[^"]+'
    r'(\s*;\s*python_version\s*==\s*\'3\.\d+\')?"',
    re.MULTILINE)

# Wheel name → version key. Filename looks like
#   mxnet-2.0.0+cu13.bw.20260523.10-cp312-cp312-linux_x86_64.whl
# The local version segment after `+` is `cu13.bw.<date>.<build>`; we sort by
# (date, build) so `.10` beats `.9`. Falls back to the full version string
# for wheels that don't match the +cu13.bw.<date>.<n> shape.
WHEEL_RE = re.compile(
    r'^mxnet-(?P<ver>[^-]+)-cp312-cp312-linux_x86_64\.whl$')
LOCAL_VER_RE = re.compile(
    r'\+cu13\.bw\.(?P<date>\d{8})\.(?P<build>\d+)$')


def wheel_sort_key(name):
    m = WHEEL_RE.match(name)
    if not m:
        return (0, 0, name)
    ver = m.group('ver')
    lm = LOCAL_VER_RE.search(ver)
    if lm:
        return (int(lm.group('date')), int(lm.group('build')), name)
    return (0, 0, ver)


def find_local_wheel(dist_dir):
    if not dist_dir.is_dir():
        sys.exit(f'Local dist dir not found: {dist_dir}')
    wheels = [p for p in dist_dir.iterdir()
              if p.name.endswith(ASSET_SUFFIX) and p.name.startswith('mxnet-')]
    if not wheels:
        sys.exit(f'No mxnet cp312 wheels in {dist_dir}')
    wheels.sort(key=lambda p: wheel_sort_key(p.name))
    newest = wheels[-1]
    # `file://` URL with `+` percent-encoded as `%2B` so uv parses the version
    # correctly (matches the existing pin's encoding).
    quoted = urllib.parse.quote(str(newest.resolve()), safe='/')
    url = f'file://{quoted}'
    return newest.name.replace(ASSET_SUFFIX, '').split('-', 1)[1], newest.name, url


def find_github_wheel():
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
    parser.add_argument('--source', choices=['local', 'github'], default='local',
                        help='Where to look for the wheel (default: local).')
    parser.add_argument('--dist-dir', type=Path, default=DEFAULT_LOCAL_DIST,
                        help=f'Local wheel dir for --source local '
                             f'(default: {DEFAULT_LOCAL_DIST}).')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would change without writing.')
    args = parser.parse_args()

    if args.source == 'local':
        tag, name, url = find_local_wheel(args.dist_dir)
        print(f'Source:  local ({args.dist_dir})')
    else:
        tag, name, url = find_github_wheel()
        print(f'Source:  github ({REPO})')
    print(f'Version: {tag}')
    print(f'Wheel:   {name}')
    print(f'URL:     {url}')

    text = PYPROJECT.read_text()
    m = LINE_RE.search(text)
    if not m:
        sys.exit(f'No `"mxnet @ …"` line found in {PYPROJECT}')
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
    print(f'\nUpdated {PYPROJECT}. Run `uv lock && make venv-mxnet` to apply.')


if __name__ == '__main__':
    main()
