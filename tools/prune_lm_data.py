#!/usr/bin/env python3
"""List, and with --delete remove, d2l-data objects under lm/ that no git ref names.

Every Language Models artifact is a versioned object in the R2 bucket
`d2l-data` (`tools/upload_data_r2.py`, docs/build-guide.md §4.13). A new
version never overwrites the old one, so superseded objects accumulate. This
tool finds them (ruling D4 of the chapter 21 pass):

    make prune-data                 # dry run: list, change nothing
    make prune-data DELETE=1        # delete the listed objects, after a prompt
    python3 tools/prune_lm_data.py [--min-age-days 14] [--no-fetch] [--verbose] [--delete]

Reference rule. An object key is reduced to its stem: the name without its
extension and its `.<part>` suffixes, ending at the version marker
(`lm-cpt-runs-v1.cpt-qwen-A.json.gz` -> `lm-cpt-runs-v1`,
`lm-tuned-lens-qwen3-0.6b-v1.pt` -> `lm-tuned-lens-qwen3-0.6b-v1`); a name
without `-v<N>` loses everything from its first dot. A key inside a folder
(`lm/fwedu-r2-v1/train-00001.bin`) also has the stem of each folder. The key
is live if one of its stems occurs inside a quoted string (a `'` or `"` before
it and the same quote after it on one line) in any tracked file of any of
these refs: every live branch of the remote (`origin/main` among them) and
every tag (the `en-*` release tags pin the translated books). The match is
textual, so a key registered through a loop over a quoted key list,
`d2l.DATA_HUB[key] = (d2l.LM_DATA_URL + key + '.zip', sha1)`, stays live, and a
stem that also occurs in an unrelated string keeps its object: over-keeping
is the safe direction. A mention in prose outside quotes (`` `lm-x-v1` `` in
Markdown, too) does not keep an object.

Refs are read with `git grep` on each ref's tree (no checkout). By default the
tool first runs `git fetch --prune --tags <remote>
'+refs/heads/*:refs/remotes/<remote>/*'`, so a branch deleted on the remote no
longer counts; --no-fetch uses the remote-tracking refs as they are (a stale
ref of a deleted branch then still keeps its keys). Push a branch before
pruning: unpushed commits are not read.

Output: the referenced count, then the unreferenced objects with size and
last-modified time, objects under a `_test/` folder in their own list (they
are never matched), and unreferenced objects younger than --min-age-days
(default 14) held back. Parts of one stem are held back together while any of
them is young. A dry run changes nothing and exits 0. --delete deletes the
listed unreferenced and `_test/` objects past the grace period, only after a
confirmation line `delete <count>` typed on stdin, and checks each key is
gone.

Credentials: as tools/upload_data_r2.py (~/.config/d2l/r2-data.env, never
printed); needs the AWS CLI and git.
"""

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import upload_data_r2 as r2  # noqa: E402

PREFIX = 'lm/'
TEST_FOLDER = '_test'
VERSION_STEM = re.compile(r'(.+?-v\d+)(?=\.|$)')
QUOTES = ("'", '"')


class PruneError(Exception):
    pass


# ── stems ───────────────────────────────────────────────────────────────────

def name_stem(name):
    """Stem of one path component: up to its version marker, else up to its
    first dot."""
    m = VERSION_STEM.match(name)
    if m:
        return m.group(1)
    return name.split('.', 1)[0] or name


def key_parts(key, prefix=PREFIX):
    rel = key[len(prefix):] if key.startswith(prefix) else key
    return [p for p in rel.split('/') if p]


def key_stems(key, prefix=PREFIX):
    """Every stem that keeps *key*: its file name's and each folder's."""
    return {name_stem(p) for p in key_parts(key, prefix)}


def primary_stem(key, prefix=PREFIX):
    """The artifact a key belongs to: its first folder under the prefix, or
    its own name. Parts of one artifact share it."""
    return name_stem(key_parts(key, prefix)[0])


def is_test_key(key, prefix=PREFIX):
    return TEST_FOLDER in key_parts(key, prefix)[:-1]


# ── refs and matching ───────────────────────────────────────────────────────

def _git(repo, *args, ok=(0,)):
    r = subprocess.run(['git', '-C', str(repo), *args], capture_output=True,
                       encoding='utf-8', errors='replace')
    if r.returncode not in ok:
        raise PruneError(f"git {' '.join(args[:3])}: "
                         f'{r.stderr.strip()[-300:]}')
    return r


def collect_refs(repo, remote='origin', fetch=True):
    """[(ref name, commit)]: the remote's live branches (main first), then
    every tag."""
    if fetch:
        _git(repo, 'fetch', '--quiet', '--prune', '--tags', remote,
             f'+refs/heads/*:refs/remotes/{remote}/*')
    fmt = '--format=%(refname)%00%(objectname)%00%(*objectname)'
    out = []
    for space in (f'refs/remotes/{remote}/', 'refs/tags/'):
        for line in _git(repo, 'for-each-ref', fmt, space).stdout.splitlines():
            name, obj, peeled = line.split('\0')
            if name == f'refs/remotes/{remote}/HEAD':
                continue
            out.append((name, peeled or obj))
    main = f'refs/remotes/{remote}/main'
    if main not in {n for n, _ in out}:
        raise PruneError(f'{main} not found: fetch the remote first')
    out.sort(key=lambda r: (r[0] != main, not r[0].startswith('refs/remotes/'),
                            r[0]))
    return out


def short_ref(name):
    for p in ('refs/remotes/', 'refs/tags/', 'refs/heads/'):
        if name.startswith(p):
            return name[len(p):]
    return name


def quoted(line, i, j):
    """True if line[i:j] has the same quote character before and after it."""
    return any(q in line[:i] and q in line[j:] for q in QUOTES)


def find_stems(repo, refs, stems):
    """{stem: 'ref:path'} for every stem that occurs inside a quoted string of
    a tracked file of some ref; refs are read in order and a stem already
    found is not searched again."""
    remaining, found = set(stems), {}
    for name, _ in refs:
        if not remaining:
            break
        with tempfile.NamedTemporaryFile('w', suffix='.stems',
                                         delete=False) as f:
            f.write(''.join(s + '\n' for s in sorted(remaining)))
            patterns = f.name
        try:
            r = _git(repo, 'grep', '-I', '-z', '-F', '-f', patterns, name,
                     '--', ok=(0, 1))
        finally:
            os.unlink(patterns)
        for record in r.stdout.split('\n'):
            where, sep, line = record.partition('\0')
            if not sep:
                continue
            for s in [s for s in remaining if s in line]:
                start = line.find(s)
                while start != -1:
                    if quoted(line, start, start + len(s)):
                        found[s] = short_ref(where)
                        remaining.discard(s)
                        break
                    start = line.find(s, start + 1)
    return found


# ── classification ──────────────────────────────────────────────────────────

def parse_time(value):
    t = dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def classify(objects, found, now, min_age_days, prefix=PREFIX):
    """Split listed objects into referenced, unreferenced, held back (young)
    and test objects (each a list of the listing's dicts, sorted by key)."""
    grace = dt.timedelta(days=min_age_days)
    young = lambda o: now - parse_time(o['LastModified']) < grace  # noqa: E731
    out = {k: [] for k in ('referenced', 'unreferenced', 'held', 'test',
                           'test_held')}
    loose = []
    for o in objects:
        key = o['Key']
        if is_test_key(key, prefix):
            out['test_held' if young(o) else 'test'].append(o)
        elif key_stems(key, prefix) & found.keys():
            out['referenced'].append(o)
        else:
            loose.append(o)
    young_stems = {primary_stem(o['Key'], prefix) for o in loose if young(o)}
    for o in loose:
        held = primary_stem(o['Key'], prefix) in young_stems
        out['held' if held else 'unreferenced'].append(o)
    for v in out.values():
        v.sort(key=lambda o: o['Key'])
    return out


# ── report ──────────────────────────────────────────────────────────────────

def human(n):
    for unit, size in (('GB', 1e9), ('MB', 1e6), ('kB', 1e3)):
        if n >= size:
            return f'{n / size:.2f} {unit}'
    return f'{n} B'


def total(objs):
    return sum(int(o.get('Size', 0)) for o in objs)


def _table(title, objs, now):
    print(f'\n{title}: {len(objs)} objects, {human(total(objs))}')
    for o in objs:
        t = parse_time(o['LastModified'])
        age = (now - t).days
        print(f"  {o['Key']:<64} {human(int(o.get('Size', 0))):>10}  "
              f"{t:%Y-%m-%d %H:%M} UTC ({age} d)")


def report(groups, refs, found, objects, now, min_age_days, verbose=False,
           prefix=PREFIX):
    print(f'd2l-data {prefix}: {len(objects)} objects, '
          f'{human(total(objects))}')
    print(f'refs read ({len(refs)}): '
          + ', '.join(f'{short_ref(n)} {c[:9]}' for n, c in refs))
    ref = groups['referenced']
    print(f'\nReferenced: {len(ref)} objects, {human(total(ref))}')
    if verbose:
        for o in ref:
            hit = sorted(key_stems(o['Key'], prefix) & found.keys())[0]
            print(f"  {o['Key']:<64} {found[hit]}")
    _table('Unreferenced (deleted by DELETE=1)', groups['unreferenced'], now)
    _table(f'Unreferenced but younger than {min_age_days} days (held back)',
           groups['held'], now)
    _table(f'{TEST_FOLDER}/ objects (deleted by DELETE=1)', groups['test'],
           now)
    _table(f'{TEST_FOLDER}/ objects younger than {min_age_days} days '
           '(held back)', groups['test_held'], now)


def delete_objects(bucket, victims, input_fn):
    n, size = len(victims), human(total(victims))
    if not n:
        print('\nNothing to delete.')
        return 0
    print(f'\nDelete {n} objects ({size}) from d2l-data? '
          f'Type "delete {n}" to confirm: ', end='', flush=True)
    try:
        answer = input_fn()
    except EOFError:
        answer = ''
    if answer.strip() != f'delete {n}':
        print('Not confirmed; nothing deleted.')
        return 1
    for o in victims:
        bucket.delete(o['Key'])
        if bucket.head(o['Key']) is not None:
            raise PruneError(f"{o['Key']}: still present after delete")
        print(f"deleted: {o['Key']}", flush=True)
    print(f'Deleted {n} objects ({size}).')
    return 0


def survey(bucket, repo='.', remote='origin', fetch=True, min_age_days=14,
           now=None, prefix=PREFIX):
    """(refs, objects, found stems, groups of classify) for one bucket
    listing; changes nothing in the bucket."""
    now = now or dt.datetime.now(dt.timezone.utc)
    refs = collect_refs(repo, remote, fetch)
    objects = bucket.list(prefix)
    stems = set().union(*(key_stems(o['Key'], prefix) for o in objects
                          if not is_test_key(o['Key'], prefix)))
    found = find_stems(repo, refs, stems)
    return refs, objects, found, classify(objects, found, now, min_age_days,
                                          prefix)


def run(repo='.', remote='origin', fetch=True, min_age_days=14,
        delete=False, verbose=False, bucket=None, now=None, input_fn=input,
        prefix=PREFIX):
    now = now or dt.datetime.now(dt.timezone.utc)
    bucket = bucket or r2.connect()
    refs, objects, found, groups = survey(bucket, repo, remote, fetch,
                                          min_age_days, now, prefix)
    report(groups, refs, found, objects, now, min_age_days, verbose, prefix)
    victims = groups['unreferenced'] + groups['test']
    if not delete:
        print(f'\nDry run: nothing deleted. make prune-data DELETE=1 deletes '
              f'the {len(victims)} objects listed for deletion '
              f'({human(total(victims))}).', flush=True)
        return 0
    return delete_objects(bucket, victims, input_fn)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='List (dry run) or delete d2l-data lm/ objects that no '
                    'branch or tag of the remote names.')
    ap.add_argument('--delete', action='store_true',
                    help='delete the listed objects after a typed '
                         'confirmation')
    ap.add_argument('--min-age-days', type=float, default=14,
                    help='hold back unreferenced objects younger than this '
                         '(default 14)')
    ap.add_argument('--no-fetch', action='store_true',
                    help='do not fetch the remote first; use its '
                         'remote-tracking refs as they are')
    ap.add_argument('--remote', default='origin')
    ap.add_argument('--repo', default='.', help='git repository (default .)')
    ap.add_argument('--verbose', action='store_true',
                    help='print, per referenced object, a ref:path naming it')
    args = ap.parse_args(argv)
    try:
        return run(args.repo, args.remote, not args.no_fetch,
                   args.min_age_days, args.delete, args.verbose)
    except (PruneError, r2.UploadError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
