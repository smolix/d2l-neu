#!/usr/bin/env python3
"""CPU tests for chapter 22's subprocess runner (accelerator-design-lib.md):
`hw_run` returns the output of a command, kills a sleeping child and its
own child (the whole process group) at the timeout and returns a refusal
record, reports a missing binary and a non-zero exit as refusals, and
raises instead in a regeneration run; `hw_bin` finds a tool in D2L_HW_BIN;
`tools/hw_regen.py` end to end on a toy regeneration (three processes, the
median of the per-process values, the packed tar, refusal to overwrite).
Later steps add `hier_allreduce` on a 4-rank gloo group (chapter spec part
1.9).
"""
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import time
import unittest
from unittest import mock

from d2l import torch as d2l


class TestRun(unittest.TestCase):
    def test_output(self):
        run = d2l.hw_run(['sh', '-c', 'echo hi; echo err >&2'], 10)
        self.assertIsNone(run.refused)
        self.assertEqual((run.stdout, run.stderr, run.returncode),
                         ('hi\n', 'err\n', 0))

    def test_timeout_kills_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile = os.path.join(tmp, 'pid')
            t = time.time()
            run = d2l.hw_run(['sh', '-c', f'sleep 30 & echo $! > {pidfile};'
                              ' wait'], 1)
            self.assertLess(time.time() - t, 10)
            self.assertEqual(run.refused, 'timeout after 1 s')
            with open(pidfile) as f:
                pid = int(f.read())
            for _ in range(50):  # the grandchild is gone too
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                self.fail('the sleeping grandchild survived the timeout')

    def test_refusals(self):
        run = d2l.hw_run(['/nonexistent/tool'], 5)
        self.assertTrue(run.refused.startswith('cannot start'))
        run = d2l.hw_run(['sh', '-c', 'exit 3'], 5)
        self.assertEqual(run.refused, 'exit status 3')
        with mock.patch.dict(os.environ, {'D2L_RERUN': 'box-p2p'}):
            with self.assertRaisesRegex(RuntimeError, 'exit status 3'):
                d2l.hw_run(['sh', '-c', 'exit 3'], 5)

    def test_env(self):
        run = d2l.hw_run(['sh', '-c', 'echo $D2L_TEST_X'], 5,
                         env={'D2L_TEST_X': 'y'})
        self.assertEqual(run.stdout, 'y\n')

    def test_hw_bin(self):
        # not under /tmp, which may be mounted noexec
        cache = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), '.cache')
        os.makedirs(cache, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as tmp:
            tool = os.path.join(tmp, 'd2l_fake_perf')
            with open(tool, 'w') as f:
                f.write('#!/bin/sh\n')
            os.chmod(tool, os.stat(tool).st_mode | stat.S_IXUSR)
            with mock.patch.dict(os.environ, {'D2L_HW_BIN': tmp}):
                self.assertEqual(d2l.hw_bin('d2l_fake_perf'), tool)
            self.assertIsNone(d2l.hw_bin('d2l_fake_perf'))


TOY = textwrap.dedent("""
    import os
    import pandas as pd
    from d2l import torch as d2l
    d2l.HW_ARTIFACTS['t-toy'] = ('lm-hw-test-toy-v1', 'tar')
    def regen_toy(config):
        proc = int(os.environ.get('D2L_HW_PROC', '0'))
        d2l._hw_result('t-toy', lambda: pd.DataFrame(
            {'n': [1, 2], 'ms': [10.0 + proc, 20.0 * (1 + proc)]}),
            dict(config=config), entry='a')
    d2l.regen_toy = regen_toy
    d2l.HW_JOBS['t-toy'] = ('regen_toy', lambda: dict(config='box'))
    """)


class TestRegenCLI(unittest.TestCase):
    def test_toy(self):
        repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, 'sitecustomize.py'), 'w') as f:
                f.write(TOY)
            env = dict(os.environ, PYTHONPATH=os.pathsep.join([tmp, repo]))
            env.pop('D2L_RERUN', None)
            out = os.path.join(tmp, 'out')
            cmd = [sys.executable, os.path.join(repo, 'tools', 'hw_regen.py'),
                   't-toy', '--out', out]
            r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                               timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            self.assertIn('sha1', r.stdout)
            tar = os.path.join(out, 'lm-hw-test-toy-v1.tar')
            with tarfile.open(tar) as t:
                names = set(t.getnames())
            self.assertTrue({'a.json.gz', 'proc0/a.json.gz',
                             'proc2/a.json.gz'} <= names, names)
            res, meta = d2l._hw_read(os.path.join(
                out, 'lm-hw-test-toy-v1', 'a.json.gz'))
            self.assertEqual(list(res.ms), [11.0, 40.0])  # medians
            self.assertEqual(list(res.n), [1, 2])
            self.assertEqual(meta['env']['n_proc'], 3)
            self.assertEqual(meta['config'], {'config': 'box'})
            r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                               timeout=300)
            self.assertEqual(r.returncode, 2)  # never overwrites
            self.assertIn('--replace', r.stderr)


if __name__ == '__main__':
    unittest.main()
