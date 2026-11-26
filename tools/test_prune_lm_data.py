"""CPU tests of tools/prune_lm_data.py (ruling D4: prune unreferenced lm/ data).

A fake bucket listing stands in for R2 (no credentials, no network) and a
temporary git repository with a bare "remote" holds the refs: main, a live
branch, a branch deleted on the remote, and an `en-2026.11.01` tag whose
commit no branch carries. Run by `make test-lm`.
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

    def survey(self, **kw):
        kw.setdefault('fetch', True)
        refs, _, found, groups = P.survey(FakeBucket(OBJECTS), self.work,
                                          now=NOW, **kw)
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
            'lm/lm-tag-only-v1.json', 'lm/lm-apos-v1.json'})
        self.assertEqual(k['unreferenced'], {
            'lm/lm-live-v12.json', 'lm/lm-gone-v1.json',
            'lm/lm-prose-v1.json', 'lm/lm-tick-v1.json',
            'lm/lm-old-v1.json'})
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

    def test_stale_ref_without_fetch_keeps_key(self):
        # An earlier fetch in this class may have pruned the stale ref.
        git(self.work, 'update-ref', 'refs/remotes/origin/feat-b', self.feat_b)
        _, _, k = self.survey(fetch=False)
        self.assertIn('lm/lm-gone-v1.json', k['referenced'])

    def test_grace_zero(self):
        _, _, k = self.survey(min_age_days=0)
        self.assertFalse(k['held'] | k['test_held'])
        self.assertIn('lm/lm-multi-v1.part1.bin', k['unreferenced'])

    def run_tool(self, delete, answer=None):
        bucket = FakeBucket(OBJECTS)

        def input_fn():
            if answer is None:
                raise EOFError
            return answer
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = P.run(self.work, delete=delete, bucket=bucket, now=NOW,
                       input_fn=input_fn)
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
        for answer in (None, 'yes', 'delete 5'):
            rc, bucket, out = self.run_tool(delete=True, answer=answer)
            self.assertEqual(rc, 1, answer)
            self.assertEqual(bucket.deleted, [], answer)
            self.assertIn('Type "delete 6" to confirm', out)

    def test_delete_confirmed(self):
        rc, bucket, out = self.run_tool(delete=True, answer='delete 6')
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(bucket.deleted), sorted([
            'lm/lm-live-v12.json', 'lm/lm-gone-v1.json',
            'lm/lm-prose-v1.json', 'lm/lm-tick-v1.json', 'lm/lm-old-v1.json',
            'lm/_test/lm-live-v1.json']))
        self.assertIn('Deleted 6 objects', out)

    def test_missing_main_raises(self):
        with tempfile.TemporaryDirectory() as d:
            git(d, 'init', '-q')
            with self.assertRaises(P.PruneError):
                P.collect_refs(d, fetch=False)


if __name__ == '__main__':
    unittest.main()
