#!/usr/bin/env python3
"""CPU tests for chapter 22's subprocess runner (accelerator-design-lib.md):
`hw_run` returns the output of a command, kills a sleeping child and its
own child (the whole process group) at the timeout and returns a refusal
record, reports a missing binary and a non-zero exit as refusals, and
raises instead in a regeneration run; `hw_bin` finds a tool in D2L_HW_BIN;
`tools/hw_regen.py` end to end on a toy regeneration (three processes, the
median of the per-process values, the packed tar, refusal to overwrite).
the driver's finish hook runs after the combine; the torch.distributed
fallback sweep of sec_lm_server_paths on a 2-rank gloo group under its own
timeout; the gloo loopback table of sec_lm_networks. Later steps add
`hier_allreduce` on a 4-rank gloo group (chapter spec part 1.9).
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
    def regen_toy_finish(folder):
        with open(os.path.join(folder, 'finish.txt'), 'w') as f:
            f.write(str(sorted(os.listdir(folder))))
    d2l.regen_toy_finish = regen_toy_finish
    d2l.HW_JOBS['t-toy'] = ('regen_toy', lambda: dict(config='box'))
    """)


class TestTorchrunSweep(unittest.TestCase):
    def test_gloo(self):
        rows, run = d2l._torchrun_sweep('all_reduce', 2, [8, 4096], 2, 1,
                                        {}, 300, backend='gloo')
        self.assertIsNone(run.refused, run.stderr[-2000:])
        self.assertEqual([r['size'] for r in rows], [8, 4096])
        self.assertTrue(all(r['time_us'] > 0 for r in rows))
        df = d2l._rows_from_times(rows, 'all_reduce', 2)
        self.assertEqual(list(df['size']), [8, 4096])

    def test_timeout(self):
        rows, run = d2l._torchrun_sweep('all_reduce', 2, [8] * 2000, 10**6,
                                        1, {}, 5, backend='gloo')
        self.assertEqual(run.refused, 'timeout after 5 s')
        left = []  # torchrun's workers run in sessions of their own
        for pid in filter(str.isdigit, os.listdir('/proc')):
            try:
                with open(f'/proc/{pid}/cmdline', 'rb') as f:
                    args = f.read().split(b'\0')
            except OSError:
                continue
            if any(a.endswith(b'/sweep.py') for a in args) and \
                    b'1000000' in args:
                left.append(pid)
        self.assertEqual(left, [], 'a sweep worker survived the timeout')

    def test_loopback_gloo(self):
        """sec_lm_networks' gloo floor: the long table fit_collectives
        reads, on 2 ranks over the loopback interface."""
        t = d2l._loopback_compute('gloo', (2,), '', 64, 2**16, 3, 1)
        self.assertEqual(list(t['size']), [64 * 2**i for i in range(11)])
        self.assertEqual(set(t.path), {'loopback'})
        self.assertEqual(t.attrs['backend'], 'gloo')
        ab = d2l.fit_collectives(t, alpha_below=2**12, plateau_from=2**15)
        self.assertGreater(ab.alpha.iloc[0], 0)


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
                             'proc2/a.json.gz', 'finish.txt'} <= names, names)
            with open(os.path.join(out, 'lm-hw-test-toy-v1',
                                   'finish.txt')) as f:
                self.assertIn('a.json.gz', f.read())  # after the combine
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
