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
is live if one of its stems occurs in a tracked file of any of these refs:
every live branch of the remote (`origin/main` among them) and every tag (the
`en-*` release tags pin the translated books). An occurrence counts when the
same quote character (`'` or `"`) appears somewhere before it and somewhere
after it on its line, or, in a config file (`*.yaml`, `*.yml`, `*.toml`,
`*.ini`, `*.cfg`, `*.mk`, `Makefile`), when it stands as a whole token
(`- data/lm/x/lm-x-v1.json -> lm-x-v1` in tools/lm_campaigns.yaml). The match
is textual, so a key registered through a loop over a quoted key list,
`d2l.DATA_HUB[key] = (d2l.LM_DATA_URL + key + '.zip', sha1)`, stays live, and a
stem that also occurs in an unrelated string keeps its object: over-keeping
is the safe direction. A mention in Markdown prose (`` `lm-x-v1` `` too)
usually keeps nothing; it keeps its object only when quote characters happen
to surround it on its line (`it's `lm-x-v1`, don't`).

Refs are read with `git grep` on each ref's whole tree, wherever the tool is
started (no checkout). By default the tool fetches the remote's branches and
tags (`--depth 1`) into a temporary bare repository that borrows the local
object store (`objects/info/alternates`) and reads them there, so a branch
deleted on the remote no longer counts and the local repository's refs and
objects are left untouched. --no-fetch reads the local remote-tracking refs
and tags as they are (a stale ref of a deleted branch then still keeps its
keys). Push a branch before pruning: unpushed commits are not read.

Output: the referenced count, then the unreferenced objects with size and
last-modified time, objects under a `_test/` folder in their own list (they
are never matched), and unreferenced objects younger than --min-age-days
(default 14) held back. Parts of one stem are held back together while any of
them is young. A dry run changes nothing and exits 0. --delete deletes the
listed unreferenced and `_test/` objects past the grace period, only after a
confirmation line `delete <count>` typed on stdin, and checks each key is
gone. It refuses when they hold more than 25% of the listed bytes (a sign of a
wrong listing) unless --allow-large is given.

Credentials: as tools/upload_data_r2.py (~/.config/d2l/r2-data.env, never
printed); needs the AWS CLI and git.
"""

import argparse
import contextlib
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
CONFIG_SUFFIXES = ('.yaml', '.yml', '.toml', '.ini', '.cfg', '.mk')
CONFIG_NAMES = ('Makefile', 'GNUmakefile')
MAX_DELETE_SHARE = 0.25
# Keep `git grep`'s output format fixed whatever the user's config says.
GREP_CONFIG = ('-c', 'color.grep=never', '-c', 'grep.lineNumber=false',
               '-c', 'grep.column=false', '-c', 'grep.fullName=true',
               '-c', 'submodule.recurse=false')


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


def _abs_git_path(repo, what):
    out = _git(repo, 'rev-parse', '--path-format=absolute', what).stdout
    return Path(out.strip())


@contextlib.contextmanager
def ref_store(repo, remote='origin', fetch=True):
    """The git directory to read the refs from: *repo* itself without fetch;
    with fetch, a temporary bare repository holding the remote's branches
    (as refs/remotes/<remote>/*) and tags, fetched at depth 1 with the local
    object store as alternate. Nothing in *repo* changes."""
    if not fetch:
        yield repo
        return
    url = _git(repo, 'remote', 'get-url', remote).stdout.strip()
    if Path(repo, url).exists():                    # a local path remote
        url = str(Path(repo, url).resolve())
    common = _abs_git_path(repo, '--git-common-dir')
    with tempfile.TemporaryDirectory(prefix='prune-lm-data-') as d:
        _git(d, 'init', '--quiet', '--bare')
        Path(d, 'objects', 'info', 'alternates').write_text(
            f"{common / 'objects'}\n")
        # A shallow local repository must say so, or the remote assumes the
        # history under its shallow commits is present and omits it.
        if (common / 'shallow').exists():
            Path(d, 'shallow').write_bytes((common / 'shallow').read_bytes())
        _git(d, 'fetch', '--quiet', '--depth', '1', '--tags', url,
             f'+refs/heads/*:refs/remotes/{remote}/*')
        yield d


def collect_refs(repo, remote='origin'):
    """[(ref name, commit)] in *repo* (a ref_store): the remote-tracking
    branches (main first), then every tag."""
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
    """True if the same quote character occurs somewhere before line[i:j]
    and somewhere after it (not necessarily enclosing it)."""
    return any(q in line[:i] and q in line[j:] for q in QUOTES)


def _token_char(c):
    return c.isalnum() or c in '-_'


def whole_token(line, i, j):
    """True if line[i:j] is not part of a longer name (`lm-x-v1` in
    `lm-x-v1.json` or `-> lm-x-v1`, not in `lm-x-v12`)."""
    return ((i == 0 or not _token_char(line[i - 1]))
            and (j == len(line) or not _token_char(line[j])))


def is_config(path):
    name = path.rsplit('/', 1)[-1]
    return name in CONFIG_NAMES or name.endswith(CONFIG_SUFFIXES)


def counts(path, line, i, j):
    """Does the occurrence line[i:j] in file *path* keep its object?"""
    return quoted(line, i, j) or (is_config(path) and whole_token(line, i, j))


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
            # ':(top)': the whole tree, not the subtree under the cwd.
            r = _git(repo, *GREP_CONFIG, 'grep', '--no-color', '--full-name',
                     '-I', '-z', '-F', '-f', patterns, name, '--', ':(top)',
                     ok=(0, 1))
        finally:
            os.unlink(patterns)
        for record in r.stdout.split('\n'):
            where, sep, line = record.partition('\0')
            if not sep:
                continue
            path = where.split(':', 1)[-1]
            for s in [s for s in remaining if s in line]:
                start = line.find(s)
                while start != -1:
                    if counts(path, line, start, start + len(s)):
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
    with ref_store(repo, remote, fetch) as store:
        refs = collect_refs(store, remote)
        objects = bucket.list(prefix)
        stems = set().union(*(key_stems(o['Key'], prefix) for o in objects
                              if not is_test_key(o['Key'], prefix)))
        found = find_stems(store, refs, stems)
    return refs, objects, found, classify(objects, found, now, min_age_days,
                                          prefix)


def large_share(victims, objects):
    """The share of the listed bytes the victims hold, if it is over
    MAX_DELETE_SHARE, else None."""
    whole = total(objects)
    share = total(victims) / whole if whole else 0
    return share if share > MAX_DELETE_SHARE else None


def run(repo='.', remote='origin', fetch=True, min_age_days=14,
        delete=False, verbose=False, bucket=None, now=None, input_fn=None,
        prefix=PREFIX, allow_large=False):
    now = now or dt.datetime.now(dt.timezone.utc)
    bucket = bucket or r2.connect()
    refs, objects, found, groups = survey(bucket, repo, remote, fetch,
                                          min_age_days, now, prefix)
    report(groups, refs, found, objects, now, min_age_days, verbose, prefix)
    victims = groups['unreferenced'] + groups['test']
    share = large_share(victims, objects)
    if share is not None:
        msg = (f'the objects listed for deletion hold {share:.0%} of the '
               f'listed bytes (over {MAX_DELETE_SHARE:.0%}): check the '
               'listing first; --allow-large (ALLOW_LARGE=1) overrides')
        if delete and not allow_large:
            raise PruneError(f'refusing to delete: {msg}')
        print(f'\nWARNING: {msg}.', flush=True)
    if not delete:
        print(f'\nDry run: nothing deleted. make prune-data DELETE=1 deletes '
              f'the {len(victims)} objects listed for deletion '
              f'({human(total(victims))}).', flush=True)
        return 0
    return delete_objects(bucket, victims, input_fn or input)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description='List (dry run) or delete d2l-data lm/ objects that no '
                    'branch or tag of the remote names.')
    ap.add_argument('--delete', action='store_true',
                    help='delete the listed objects after a typed '
                         'confirmation')
    ap.add_argument('--allow-large', action='store_true',
                    help='delete even when the listed objects hold more '
                         f'than {MAX_DELETE_SHARE:.0%} of the bytes')
    ap.add_argument('--min-age-days', type=float, default=14,
                    help='hold back unreferenced objects younger than this '
                         '(default 14)')
    ap.add_argument('--no-fetch', action='store_true',
                    help='do not fetch the remote; read the local '
                         'remote-tracking refs and tags as they are')
    ap.add_argument('--remote', default='origin')
    ap.add_argument('--repo', default='.', help='git repository (default .)')
    ap.add_argument('--verbose', action='store_true',
                    help='print, per referenced object, a ref:path naming it')
    args = ap.parse_args(argv)
    try:
        return run(args.repo, args.remote, not args.no_fetch,
                   args.min_age_days, args.delete, args.verbose,
                   allow_large=args.allow_large)
    except (PruneError, r2.UploadError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
