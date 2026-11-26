#!/usr/bin/env python3
"""English releases: `make release TAG=en-YYYY.MM.DD` and `make release-check`.

A release tag is the snapshot every translation builds against
(d2l-localization DESIGN §1): sources, the outputs/ store, images and the
DATA_HUB pins of one commit on origin/main. `check` refuses unless

  1. TAG (when given) is en-YYYY.MM.DD and does not exist yet, locally or on
     origin;
  2. HEAD is a commit on origin/main (fetched first);
  3. the working tree is clean: no modified or staged tracked file, and no
     untracked file under a directory the build reads;
  4. the outputs/ store is fresh and complete (`tools/audit_outputs.py`:
     no stale, orphaned, missing or unmanifested notebook, no locked notebook
     with code drift);
  5. no Git-LFS pointer files under img/, static/ or outputs/;
  6. `make all-quick` succeeded on this commit, in English, on a clean tree
     (the stamp `stamp-build` writes at the end of all-quick).

`tag` runs the checks, then creates the annotated tag and pushes it over
HTTPS through gh's git credential helper (never SSH). `--dry-run` prints the
git commands instead of running them.

    python3 tools/release.py check [--tag TAG]
    python3 tools/release.py tag --tag TAG [--dry-run]
    python3 tools/release.py stamp-build        # end of `make all-quick`
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STAMP = REPO / '.all-quick.ok'
TAG_RE = re.compile(r'en-\d{4}\.\d{2}\.\d{2}')
REMOTE_URL = 'https://github.com/smolix/d2l-neu.git'
# Untracked files here could change what the build renders.
BUILD_INPUTS = ('chapter_', 'outputs/', 'img/', 'static/', 'tools/', 'i18n/',
                'd2l/', 'diagrams/', 'make/', 'index.md', '_quarto.yml',
                'Makefile', 'pyproject.toml', 'uv.lock', 'd2l.bib', '_d2l-')
LFS_DIRS = ('img', 'static', 'outputs')
LFS_HEAD = b'version https://git-lfs.github.com/spec/v1'


def git(*args, check=True, strip=True) -> str:
    r = subprocess.run(['git', *args], cwd=REPO, capture_output=True,
                       text=True)
    if check and r.returncode:
        raise SystemExit(f'git {" ".join(args)} failed:\n{r.stderr.strip()}')
    return r.stdout.strip() if strip else r.stdout


def head() -> str:
    return git('rev-parse', 'HEAD')


def dirty_paths():
    """(tracked changes, untracked build inputs, other untracked) of the tree."""
    tracked, inputs, other = [], [], []
    for line in git('status', '--porcelain=v1', '--untracked-files=all',
                    strip=False).splitlines():
        status, path = line[:2], line[3:]
        if status == '??':
            (inputs if path.startswith(BUILD_INPUTS) else other).append(path)
        else:
            tracked.append(path)
    return tracked, inputs, other


def lfs_pointers():
    out = []
    for d in LFS_DIRS:
        for p in (REPO / d).rglob('*'):
            try:
                if p.is_file() and not p.is_symlink() and p.stat().st_size < 1024:
                    with open(p, 'rb') as f:
                        if f.read(len(LFS_HEAD)) == LFS_HEAD:
                            out.append(p.relative_to(REPO).as_posix())
            except OSError:
                continue
    return out


def audit():
    """Problems of the outputs/ store, from tools/audit_outputs.py --json."""
    r = subprocess.run([sys.executable, 'tools/audit_outputs.py', '--json'],
                       cwd=REPO, capture_output=True, text=True)
    try:
        data = json.loads(r.stdout)
    except ValueError:
        return [f'audit_outputs.py failed (rc={r.returncode}):\n'
                f'{r.stderr.strip() or r.stdout.strip()}']
    problems = []
    for key, what in (('stale', 'stale notebook(s)'),
                      ('orphaned', 'manifest(s) with orphaned cell ids'),
                      ('missing', 'manifest(s) missing cells'),
                      ('no_manifest', 'notebook(s) never captured')):
        items = data.get(key) or []
        if items:
            names = list(items)[:8]
            problems.append(f'{len(items)} {what}: ' + ', '.join(names) +
                            (' ...' if len(items) > 8 else ''))
    bad_locks = {k: v for k, v in (data.get('locked') or {}).items()
                 if v in ('code-drift', 'no-manifest')}
    if bad_locks:
        problems.append(f'{len(bad_locks)} locked notebook(s) with '
                        f'code drift: ' + ', '.join(sorted(bad_locks)[:8]))
    return problems


def read_stamp():
    try:
        return dict(line.split('=', 1) for line in
                    STAMP.read_text().splitlines() if '=' in line)
    except OSError:
        return {}


def stamp_build():
    """Record a finished all-quick for HEAD (called by `make all-quick`)."""
    tracked, inputs, _ = dirty_paths()
    lang = os.environ.get('D2L_LANG') or 'en'
    locale = os.environ.get('D2L_LOCALE_FILE') or ''
    STAMP.write_text(
        f'commit={head()}\n'
        f'clean={"no" if tracked or inputs else "yes"}\n'
        f'lang={lang}\nlocale_file={locale}\n'
        f'date={datetime.now(timezone.utc).isoformat(timespec="seconds")}\n')
    print(f'release: recorded all-quick for {head()[:12]} in {STAMP.name}')


def remote_tags() -> set:
    out = git('ls-remote', '--tags', REMOTE_URL, check=False)
    return {line.split('refs/tags/')[-1].removesuffix('^{}')
            for line in out.splitlines() if 'refs/tags/' in line}


def check(tag=None) -> list:
    problems = []
    if tag is not None:
        if not TAG_RE.fullmatch(tag):
            problems.append(f'TAG={tag!r} is not en-YYYY.MM.DD')
        if git('tag', '--list', tag):
            problems.append(f'tag {tag} exists locally')
        if tag in remote_tags():
            problems.append(f'tag {tag} exists on origin')

    git('fetch', '--quiet', REMOTE_URL, 'main:refs/remotes/origin/main',
        check=False)
    if subprocess.run(['git', 'merge-base', '--is-ancestor', 'HEAD',
                       'origin/main'], cwd=REPO).returncode:
        problems.append(f'HEAD {head()[:12]} is not on origin/main')

    tracked, inputs, other = dirty_paths()
    if tracked:
        problems.append(f'{len(tracked)} modified tracked file(s): '
                        + ', '.join(tracked[:8]))
    if inputs:
        problems.append(f'{len(inputs)} untracked file(s) the build reads: '
                        + ', '.join(inputs[:8]))
    if other:
        print(f'note: {len(other)} untracked file(s) outside the build '
              f'inputs (not in the tag): {", ".join(other[:5])}'
              + (' ...' if len(other) > 5 else ''))

    problems += audit()

    ptrs = lfs_pointers()
    if ptrs:
        problems.append(f'{len(ptrs)} Git-LFS pointer file(s) (run `git lfs '
                        f'pull`): ' + ', '.join(ptrs[:8]))

    st = read_stamp()
    if not st:
        problems.append('no record of a successful `make all-quick` '
                        f'({STAMP.name} missing)')
    elif st.get('commit') != head():
        problems.append(f'`make all-quick` last succeeded on '
                        f'{st.get("commit", "?")[:12]}, not on HEAD '
                        f'{head()[:12]}')
    elif st.get('clean') != 'yes':
        problems.append('`make all-quick` ran on a dirty tree; rerun it on '
                        'the clean commit')
    elif st.get('lang') != 'en' or st.get('locale_file'):
        problems.append('`make all-quick` last ran localized '
                        f'(D2L_LANG={st.get("lang")}); rerun it in English')
    return problems


def report(problems, what) -> int:
    if problems:
        print(f'{what}: REFUSED')
        for p in problems:
            print(f'  ✗ {p}')
        return 1
    print(f'{what}: OK — {head()[:12]} is releasable')
    return 0


def tag_release(tag, dry_run=False) -> int:
    rc = report(check(tag), f'release {tag}')
    if rc:
        return rc
    cmds = [['git', 'tag', '-a', tag, '-m',
             f'English release {tag} ({head()[:12]})'],
            ['gh', 'auth', 'setup-git'],
            ['git', 'push', REMOTE_URL, f'refs/tags/{tag}']]
    for cmd in cmds:
        if dry_run:
            print('dry run: would run ' + ' '.join(
                f'"{c}"' if ' ' in c else c for c in cmd))
            continue
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode:
            print(f'release: {" ".join(cmd)} failed (rc={r.returncode})')
            return r.returncode
    if not dry_run:
        print(f'release: tagged {head()[:12]} as {tag} and pushed it')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check', help='run the release checks, no tag')
    c.add_argument('--tag')
    t = sub.add_parser('tag', help='check, then tag and push')
    t.add_argument('--tag', required=True)
    t.add_argument('--dry-run', action='store_true')
    sub.add_parser('stamp-build', help='record a green all-quick for HEAD')
    args = ap.parse_args(argv)
    if args.cmd == 'stamp-build':
        stamp_build()
        return 0
    if args.cmd == 'check':
        return report(check(args.tag), 'release-check')
    return tag_release(args.tag, args.dry_run)


if __name__ == '__main__':
    sys.exit(main())
