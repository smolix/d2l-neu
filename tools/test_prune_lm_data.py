"""CPU tests of tools/prune_lm_data.py (ruling D4: prune unreferenced lm/ data).

A fake bucket listing stands in for R2 (no credentials, no network) and a
temporary git repository with a bare "remote" holds the refs: main, a live
branch, a branch deleted on the remote, and an `en-2026.11.01` tag whose
commit no branch carries. The clone's own config sets a hostile `git grep`
output format (line numbers, columns, colour). Run by `make test-lm`.
"""

import contextlib
import datetime as dt
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prune_lm_data as P  # noqa: E402

NOW = dt.datetime(2026, 11, 26, 12, 0, tzinfo=dt.timezone.utc)
ENV = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@example.com',
           GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@example.com')


def git(cwd, *args):
    subprocess.run(['git', '-c', 'commit.gpgsign=false', '-c',
                    'tag.gpgsign=false', '-c', 'init.defaultBranch=main',
                    *args], cwd=cwd, env=ENV, check=True,
                   capture_output=True)


def commit(repo, files, msg):
    for name, text in files.items():
        p = Path(repo, name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        git(repo, 'add', name)
    git(repo, 'commit', '-q', '-m', msg)


MAIN_FILES = {
    'registry-lib.md': (
        "d2l.DATA_HUB['lm-live-v1'] = (\n"
        "    d2l.LM_DATA_URL + 'lm-live-v1.json',\n"
        "    '0123456789abcdef0123456789abcdef01234567')\n"),
    # chapter_lm-pretrained/api.md registers its keys through a loop.
    'api.md': (
        "API_KEYS = {'lm-api-a-v1': 'aaaa', 'lm-api-b-v1': 'bbbb'}\n"
        "for key, sha1 in API_KEYS.items():\n"
        "    d2l.DATA_HUB[key] = (d2l.LM_DATA_URL + key + '.zip', sha1)\n"),
    'cpt.py': "RUNS = 'lm-cpt-runs-v1'  # one object per arm\n",
    'shards.py': 'SHARDS = "fwedu-x-v1"\n',
    'notes.md': ("The old lm-prose-v1 and `lm-tick-v1` were superseded; "
                 "don't use 'lm-apos-v1' either.\n"),
    # A registration in a subdirectory: found from any cwd.
    'chapter_x/reg-lib.md': "d2l.DATA_HUB['lm-deep-v1'] = (url, sha1)\n",
    # Config files count whole unquoted tokens; a longer name does not.
    'tools/campaigns.yaml': ("outputs:\n"
                             "  - data/lm/x/lm-yaml-v1.json -> lm-yaml-v1\n"
                             "  - data/lm/x/lm-tok-v12.json -> lm-tok-v12\n"),
    'tools/readme.md': "Campaign output lm-plain-v1 -> lm-plain-v1\n",
}


def obj(key, days, size=1000):
    return {'Key': key, 'Size': size,
            'LastModified': (NOW - dt.timedelta(days=days)).isoformat()}


OBJECTS = [
    obj('lm/lm-live-v1.json', 40),
    obj('lm/lm-live-v12.json', 40),              # a different version
    obj('lm/lm-api-a-v1.zip', 40),
    obj('lm/lm-api-b-v1.zip', 40),
    obj('lm/lm-cpt-runs-v1.cpt-qwen-A.json.gz', 40),
    obj('lm/lm-cpt-runs-v1.cpt-qwen-B.json.gz', 40),
    obj('lm/fwedu-x-v1/train-00000.bin', 40, 200_000_000),
    obj('lm/lm-branch-v1.json', 40),
    obj('lm/lm-gone-v1.json', 40),
    obj('lm/lm-tag-only-v1.json', 40),
    obj('lm/lm-prose-v1.json', 40),
    obj('lm/lm-tick-v1.json', 40),
    obj('lm/lm-apos-v1.json', 40),
    obj('lm/lm-deep-v1.json', 40),
    obj('lm/lm-yaml-v1.json', 40),
    obj('lm/lm-tok-v1.json', 40),
    obj('lm/lm-plain-v1.json', 40),
    obj('lm/lm-old-v1.json', 40, 5_000_000),
    obj('lm/lm-young-v1.json', 3),
    obj('lm/lm-multi-v1.part0.bin', 30),
    obj('lm/lm-multi-v1.part1.bin', 2),
    obj('lm/_test/lm-live-v1.json', 40),
    obj('lm/_test/new.txt', 1),
]


class FakeBucket:
    def __init__(self, objects):
        self.objects = {o['Key']: dict(o) for o in objects}
        self.deleted, self.prefixes = [], []

    def list(self, prefix):
        self.prefixes.append(prefix)
        return [dict(o) for k, o in sorted(self.objects.items())
                if k.startswith(prefix)]

    def head(self, key):
        return self.objects.get(key)

    def delete(self, key):
        self.deleted.append(key)
        self.objects.pop(key, None)


class StemTest(unittest.TestCase):
    def test_name_stem(self):
        cases = {
            'lm-cpt-runs-v1.cpt-qwen-A.json.gz': 'lm-cpt-runs-v1',
            'lm-r2-s500-v1.safetensors': 'lm-r2-s500-v1',
            'lm-tuned-lens-qwen3-0.6b-v1.pt': 'lm-tuned-lens-qwen3-0.6b-v1',
            'lm-rag-judge-v1cal-v1.zip': 'lm-rag-judge-v1cal-v1',
            'lm-cc-warc-256m-v1.pin.json': 'lm-cc-warc-256m-v1',
            'lm-fwedu-val-v1.tar': 'lm-fwedu-val-v1',
            'train-00001.bin': 'train-00001',
            'fwedu-r2-v1': 'fwedu-r2-v1',
        }
        for name, stem in cases.items():
            self.assertEqual(P.name_stem(name), stem, name)

    def test_key_stems_and_test_keys(self):
        self.assertEqual(P.key_stems('lm/fwedu-r2-v1/train-00001.bin'),
                         {'fwedu-r2-v1', 'train-00001'})
        self.assertEqual(P.primary_stem('lm/fwedu-r2-v1/train-00001.bin'),
                         'fwedu-r2-v1')
        self.assertTrue(P.is_test_key('lm/_test/x.txt'))
        self.assertFalse(P.is_test_key('lm/lm-_test-v1.json'))
        self.assertFalse(P.is_test_key('lm/_test'))

    def test_quoted(self):
        line = "don't use 'lm-x-v1' here"
        i = line.index('lm-x')
        self.assertTrue(P.quoted(line, i, i + 7))
        line = 'plain lm-x-v1 prose and `lm-x-v1`'
        i = line.index('lm-x')
        self.assertFalse(P.quoted(line, i, i + 7))

    def test_whole_token_in_config(self):
        line = '- data/lm/x/lm-x-v1.json -> lm-x-v1'
        for i in (line.index('lm-x'), line.rindex('lm-x')):
            self.assertTrue(P.counts('tools/c.yaml', line, i, i + 7))
            self.assertFalse(P.counts('notes.md', line, i, i + 7))
        line = 'x: lm-x-v12'
        i = line.index('lm-x')
        self.assertFalse(P.counts('c.yaml', line, i, i + 7))
        self.assertTrue(P.is_config('make/store.mk'))
        self.assertTrue(P.is_config('Makefile'))
        self.assertFalse(P.is_config('tools/x.py'))


class PruneRepoTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        remote, seed = root / 'remote.git', root / 'seed'
        git(root, 'init', '-q', '--bare', str(remote))
        git(root, 'init', '-q', str(seed))
        commit(seed, MAIN_FILES, 'main')
        git(seed, 'remote', 'add', 'origin', str(remote))
        git(seed, 'push', '-q', 'origin', 'main')
        for branch, key in (('feat-a', 'lm-branch-v1'),
                            ('feat-b', 'lm-gone-v1'),
                            ('release', 'lm-tag-only-v1')):
            git(seed, 'checkout', '-q', '-b', branch, 'main')
            commit(seed, {f'{branch}.py': f"KEY = '{key}'\n"}, branch)
            git(seed, 'push', '-q', 'origin', branch)
        git(seed, 'tag', '-a', 'en-2026.11.01', '-m', 'release', 'release')
        git(seed, 'push', '-q', 'origin', 'en-2026.11.01')
        cls.work = root / 'work'
        git(root, 'clone', '-q', str(remote), str(cls.work))
        for k, v in (('grep.lineNumber', 'true'), ('grep.column', 'true'),
                     ('color.grep', 'always'), ('color.ui', 'always')):
            git(cls.work, 'config', k, v)
        # Deleted on the remote after the clone fetched them: feat-b (its
        # remote-tracking ref in work is now stale) and release (only the tag
        # still carries its commit).
        git(remote, 'branch', '-D', 'feat-b', 'release')
        cls.feat_b = subprocess.run(
            ['git', 'rev-parse', 'refs/remotes/origin/feat-b'], cwd=cls.work,
            check=True, capture_output=True, text=True).stdout.strip()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def survey(self, repo=None, **kw):
        kw.setdefault('fetch', True)
        refs, _, found, groups = P.survey(FakeBucket(OBJECTS),
                                          repo or self.work, now=NOW, **kw)
        keys = {g: {o['Key'] for o in v} for g, v in groups.items()}
        return refs, found, keys

    def test_refs_read(self):
        refs, _, _ = self.survey()
        names = [n for n, _ in refs]
        self.assertEqual(names[0], 'refs/remotes/origin/main')
        self.assertIn('refs/remotes/origin/feat-a', names)
        self.assertIn('refs/tags/en-2026.11.01', names)
        self.assertNotIn('refs/remotes/origin/feat-b', names)
        self.assertNotIn('refs/remotes/origin/release', names)

    def test_classification(self):
        _, found, k = self.survey()
        self.assertEqual(k['referenced'], {
            'lm/lm-live-v1.json', 'lm/lm-api-a-v1.zip', 'lm/lm-api-b-v1.zip',
            'lm/lm-cpt-runs-v1.cpt-qwen-A.json.gz',
            'lm/lm-cpt-runs-v1.cpt-qwen-B.json.gz',
            'lm/fwedu-x-v1/train-00000.bin', 'lm/lm-branch-v1.json',
            'lm/lm-tag-only-v1.json', 'lm/lm-apos-v1.json',
            'lm/lm-deep-v1.json', 'lm/lm-yaml-v1.json'})
        self.assertEqual(k['unreferenced'], {
            'lm/lm-live-v12.json', 'lm/lm-gone-v1.json',
            'lm/lm-prose-v1.json', 'lm/lm-tick-v1.json',
            'lm/lm-old-v1.json', 'lm/lm-tok-v1.json', 'lm/lm-plain-v1.json'})
        # Grace period; the old part of a young multi-part object waits too.
        self.assertEqual(k['held'], {'lm/lm-young-v1.json',
                                     'lm/lm-multi-v1.part0.bin',
                                     'lm/lm-multi-v1.part1.bin'})
        # _test/ objects are listed apart even when their stem is live.
        self.assertEqual(k['test'], {'lm/_test/lm-live-v1.json'})
        self.assertEqual(k['test_held'], {'lm/_test/new.txt'})
        self.assertEqual(found['lm-tag-only-v1'],
                         'en-2026.11.01:release.py')
        self.assertTrue(found['lm-api-a-v1'].startswith('origin/main:'))
        self.assertEqual(found['lm-deep-v1'], 'origin/main:chapter_x/reg-lib.md')
        self.assertEqual(found['lm-yaml-v1'],
                         'origin/main:tools/campaigns.yaml')

    def test_any_cwd_gives_the_same_listing(self):
        for fetch in (True, False):
            _, top_found, top = self.survey(fetch=fetch)
            for sub in ('chapter_x', 'tools'):
                _, found, k = self.survey(repo=self.work / sub, fetch=fetch)
                self.assertEqual(k, top, (sub, fetch))
                self.assertEqual(found, top_found, (sub, fetch))

    def test_fetch_leaves_the_repo_alone(self):
        def state():
            return subprocess.run(
                ['git', 'for-each-ref'], cwd=self.work, check=True,
                capture_output=True, text=True).stdout
        git(self.work, 'update-ref', 'refs/remotes/origin/feat-b', self.feat_b)
        before = state()
        refs, _, _ = self.survey()
        self.assertEqual(state(), before)
        self.assertNotIn('refs/remotes/origin/feat-b', [n for n, _ in refs])

    def test_stale_ref_without_fetch_keeps_key(self):
        # An earlier fetch in this class may have pruned the stale ref.
        git(self.work, 'update-ref', 'refs/remotes/origin/feat-b', self.feat_b)
        _, _, k = self.survey(fetch=False)
        self.assertIn('lm/lm-gone-v1.json', k['referenced'])

    def test_grace_zero(self):
        _, _, k = self.survey(min_age_days=0)
        self.assertFalse(k['held'] | k['test_held'])
        self.assertIn('lm/lm-multi-v1.part1.bin', k['unreferenced'])

    def run_tool(self, delete, answer=None, objects=OBJECTS, **kw):
        bucket = FakeBucket(objects)

        def input_fn():
            if answer is None:
                raise EOFError
            return answer
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = P.run(self.work, delete=delete, bucket=bucket, now=NOW,
                       input_fn=input_fn, **kw)
        return rc, bucket, out.getvalue()

    def test_dry_run_deletes_nothing(self):
        rc, bucket, out = self.run_tool(delete=False)
        self.assertEqual(rc, 0)
        self.assertEqual(bucket.deleted, [])
        self.assertEqual(bucket.prefixes, ['lm/'])
        self.assertIn('Dry run: nothing deleted', out)
        self.assertIn('lm/lm-old-v1.json', out)
        self.assertIn('2026-10-17', out)              # last-modified date
        self.assertIn('5.00 MB', out)                 # size

    def test_delete_needs_confirmation(self):
        for answer in (None, 'yes', 'delete 7'):
            rc, bucket, out = self.run_tool(delete=True, answer=answer)
            self.assertEqual(rc, 1, answer)
            self.assertEqual(bucket.deleted, [], answer)
            self.assertIn('Type "delete 8" to confirm', out)

    VICTIMS = ['lm/lm-live-v12.json', 'lm/lm-gone-v1.json',
               'lm/lm-prose-v1.json', 'lm/lm-tick-v1.json',
               'lm/lm-old-v1.json', 'lm/lm-tok-v1.json',
               'lm/lm-plain-v1.json', 'lm/_test/lm-live-v1.json']

    def test_delete_confirmed(self):
        rc, bucket, out = self.run_tool(delete=True, answer='delete 8')
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(bucket.deleted), sorted(self.VICTIMS))
        self.assertIn('Deleted 8 objects', out)

    def test_large_share_needs_allow_large(self):
        big = [obj(o['Key'], 40, 500_000_000) if o['Key'] == 'lm/lm-old-v1.json'
               else o for o in OBJECTS]
        rc, bucket, out = self.run_tool(delete=False, objects=big)
        self.assertEqual(rc, 0)
        self.assertIn('WARNING: the objects listed for deletion hold', out)
        with self.assertRaises(P.PruneError):
            self.run_tool(delete=True, answer='delete 8', objects=big)
        rc, bucket, _ = self.run_tool(delete=True, answer='delete 8',
                                      objects=big, allow_large=True)
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(bucket.deleted), sorted(self.VICTIMS))

    def main_tool(self, *argv, answer='no'):
        bucket = FakeBucket(OBJECTS)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(P.r2, 'connect', return_value=bucket), \
                mock.patch('builtins.input', return_value=answer), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = P.main(['--repo', str(self.work), *argv])
        return rc, bucket, out.getvalue(), err.getvalue()

    def test_main_wiring(self):
        rc, bucket, out, _ = self.main_tool('--min-age-days', '0',
                                            '--verbose')
        self.assertEqual((rc, bucket.deleted), (0, []))
        self.assertIn('younger than 0.0 days (held back): 0 objects', out)
        self.assertIn('origin/main:chapter_x/reg-lib.md', out)   # --verbose
        # Without fetch the clone's stale origin/feat-b would keep a key.
        git(self.work, 'update-ref', '-d', 'refs/remotes/origin/feat-b')
        rc, bucket, out, _ = self.main_tool('--no-fetch', '--delete',
                                            answer='delete 8')
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(bucket.deleted), sorted(self.VICTIMS))
        rc, _, _, err = self.main_tool('--remote', 'nosuch')
        self.assertEqual(rc, 1)
        self.assertIn('ERROR: git remote get-url', err)

    def test_make_target_flags(self):
        root = Path(__file__).resolve().parent.parent
        cmd = ['make', '-s', '-n', '-f', 'make/store.mk', 'SHELL=/bin/bash',
               'LOGDIR=L', 'TS=T', 'prune-data']
        try:
            dry = subprocess.run(cmd, cwd=root, capture_output=True,
                                 text=True, check=True).stdout
            full = subprocess.run(
                cmd + ['DELETE=1', 'MIN_AGE_DAYS=3', 'NO_FETCH=1',
                       'ALLOW_LARGE=1'], cwd=root, capture_output=True,
                text=True, check=True).stdout
        except FileNotFoundError:
            self.skipTest('make not installed')
        flags = ('--delete', '--min-age-days', '--no-fetch', '--allow-large')
        for f in flags:
            self.assertNotIn(f, dry)
        self.assertIn('--delete', full)
        self.assertIn('--min-age-days 3', full)
        self.assertIn('--no-fetch', full)
        self.assertIn('--allow-large', full)
        self.assertIn('exit ${PIPESTATUS[0]}', full)

    def test_missing_main_raises(self):
        with tempfile.TemporaryDirectory() as d:
            git(d, 'init', '-q')
            with self.assertRaises(P.PruneError):
                P.collect_refs(d)


if __name__ == '__main__':
    unittest.main()
