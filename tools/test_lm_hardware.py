#!/usr/bin/env python3
"""CPU tests for chapter 22 (chapter_lm-hardware/), the parts the infra step
wrote: the chapter's registries and load-or-measure plumbing
(accelerator-design-lib.md): `_hw_result` with a temporary registry
(regenerate stores config and env, load, configuration mismatch raises,
absent raises in a normal run, the labelled pending result when the name is
in `HW_PENDING` and in any regeneration run, two regeneration names with
other calls in the same process, `D2L_RERUN=1` refused, a live check that
fails prints one line), multi-process regeneration folders, `hw_box_log`,
`hw_campaign_log` and `hw_status_line` (pending and registered),
`hw_registered`; the registry names (`HW_ARTIFACTS` equals the eight
regenerations of the spec, each with an `HW_JOBS` entry; one statement per
entry); an AST scan of every chapter-22 block: no function body holds an
`lm-hw-` key or a chapter short name it does not read itself; the storage
round trip (`_hw_encode`/`_hw_decode` of frames with a MultiIndex,
namedtuples, arrays, tuples; `_hw_pack`/`_hw_unpack` for tar and jsonl);
`hw_config` (raises for an unpinned id; a pinned config offline);
`HW_SHAPES['dense-70b']` (70.6B parameters); `fit_alpha_beta` (recovers
synthetic alpha and beta with 1% noise to 2%, seed 0); `show_table`
(rounding, headers, row cap, width check); the Triton stand-in.
Later steps add the section tests of the chapter spec (part 1.9).
"""
import ast
import contextlib
import glob
import io
import json
import os
import re
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch

from d2l import torch as d2l

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER = os.path.join(REPO, 'chapter_lm-hardware')

# The regeneration column of the chapter spec, part 2.
REGEN_NAMES = ('box-microbench', 'box-workloads', 'box-moe', 'box-storage',
               'box-p2p', 'box-net', 'box-coll', 'box-power')

Point = d2l.collections.namedtuple('Point', ['x', 'y'])


def read(path, mode='r'):
    with open(path, mode) as f:
        return f.read()


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


class Registry(unittest.TestCase):
    """A temporary registry of two names, an output folder, D2L_RERUN."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, 'rerun')
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.dict(d2l.HW_ARTIFACTS, {
            't-multi': ('lm-hw-test-multi-v1', 'tar'),
            't-table': ('lm-hw-test-table-v1', 'jsonl')}, clear=True))
        stack.enter_context(mock.patch.dict(d2l.DATA_HUB, {}))
        stack.enter_context(mock.patch.dict(os.environ, {
            'D2L_HW_OUT': self.out, 'D2L_RERUN': ''}))
        os.environ.pop('D2L_HW_PROC', None)
        self.calls = 0

    def tearDown(self):
        self.tmp.cleanup()

    def compute(self):
        self.calls += 1
        return pd.DataFrame({'n': [1, 2, 3], 'gbs': [1.5, 2.5, 3.5]})

    def regenerate(self, name, entry=None, config=None, env='t-multi'):
        os.environ['D2L_RERUN'] = env
        with quiet():
            r = d2l._hw_result(name, self.compute, config or dict(n=3),
                               entry=entry)
        os.environ['D2L_RERUN'] = ''
        return r

    def register(self, name):
        """Pack what a regeneration wrote and register it under its key."""
        key, kind = d2l.HW_ARTIFACTS[name]
        folder = os.path.join(self.out, key)
        packed = os.path.join(self.tmp.name, 'hub', key, key + d2l._HW_EXT[
            kind])
        os.makedirs(os.path.dirname(packed), exist_ok=True)
        d2l._hw_pack(folder, kind, packed)
        d2l.DATA_HUB[key] = ('file://' + packed, None)
        return packed


class TestResult(Registry):
    def test_regenerate_stores_config_and_env(self):
        r = self.regenerate('t-multi', entry='ladder')
        self.assertEqual(self.calls, 1)
        self.assertEqual(list(r.gbs), [1.5, 2.5, 3.5])
        res, meta = d2l._hw_read(os.path.join(
            self.out, 'lm-hw-test-multi-v1', 'ladder.json.gz'))
        self.assertEqual(meta['config'], {'n': 3})
        self.assertIn('contended', meta['env'])
        self.assertIn('neighbours_busy', meta['env'])
        self.assertEqual(len(meta['env']['host']), 8)
        pd.testing.assert_frame_equal(res, r)

    def test_load_after_registration(self):
        self.regenerate('t-multi', entry='ladder')
        packed = self.register('t-multi')
        with mock.patch.object(d2l, 'download_keyed', return_value=packed):
            with quiet():
                r = d2l._hw_result('t-multi', self.compute, dict(n=3),
                                   entry='ladder')
            self.assertEqual(self.calls, 1)  # loaded, not measured
            self.assertEqual(list(r.n), [1, 2, 3])
            with self.assertRaisesRegex(ValueError, 'n 4 != 3'):
                d2l._hw_result('t-multi', self.compute, dict(n=4),
                               entry='ladder')
            # a regeneration writes the next version
            self.assertEqual(os.path.basename(d2l._hw_regen_dir('t-multi')),
                             'lm-hw-test-multi-v2')

    def test_jsonl_round_trip(self):
        self.regenerate('t-table', env='t-table')
        packed = self.register('t-table')
        with mock.patch.object(d2l, 'download_keyed', return_value=packed):
            r = d2l.hw_box_log('t-table')
        self.assertEqual(list(r.gbs), [1.5, 2.5, 3.5])

    def test_absent(self):
        with self.assertRaisesRegex(KeyError, 'not committed yet'):
            d2l._hw_result('t-multi', self.compute, {})
        with mock.patch.object(d2l, 'HW_PENDING', frozenset({'t-multi'})):
            r = d2l._hw_result('t-multi', self.compute, {})
        self.assertEqual(r.attrs['label'], 't-multi pending')
        os.environ['D2L_RERUN'] = 't-table'  # another name regenerates
        r = d2l._hw_result('t-multi', self.compute, {},
                           empty=lambda: pd.DataFrame({'gbs': []}))
        self.assertEqual(list(r.columns), ['gbs'])
        self.assertEqual(r.attrs['label'], 't-multi pending')
        self.assertEqual(self.calls, 0)

    def test_two_names_and_others(self):
        os.environ['D2L_RERUN'] = 't-multi,t-table'
        with quiet():
            d2l._hw_result('t-multi', self.compute, {}, entry='a')
            d2l._hw_result('t-table', self.compute, {})
        self.assertEqual(self.calls, 2)

    def test_rerun_all_refused(self):
        os.environ['D2L_RERUN'] = '1'
        with self.assertRaisesRegex(ValueError, 'hw_regen'):
            d2l._hw_result('t-multi', self.compute, {})

    def test_regeneration_reads_what_it_wrote(self):
        self.regenerate('t-multi', entry='ladder')
        os.environ['D2L_RERUN'] = 't-table'
        r = d2l.hw_box_log('t-multi', 'ladder')
        self.assertEqual(list(r.n), [1, 2, 3])

    def test_process_folders(self):
        os.environ['D2L_HW_PROC'] = '1'
        try:
            self.regenerate('t-multi', entry='ladder')
        finally:
            del os.environ['D2L_HW_PROC']
        self.assertTrue(os.path.exists(os.path.join(
            self.out, 'lm-hw-test-multi-v1', 'proc1', 'ladder.json.gz')))

    def test_live_check_line(self):
        self.regenerate('t-multi', entry='ladder')
        packed = self.register('t-multi')

        def bad(result, same):
            raise RuntimeError('boom')
        buf = io.StringIO()
        with mock.patch.object(d2l, 'download_keyed', return_value=packed), \
                mock.patch.object(torch.cuda, 'is_available',
                                  return_value=True), \
                mock.patch.object(torch.cuda, 'get_device_name',
                                  return_value='NVIDIA GeForce RTX 9999'), \
                contextlib.redirect_stdout(buf):
            d2l._hw_result('t-multi', self.compute, dict(n=3),
                           entry='ladder', live_check=bad,
                           show=lambda r: print('shown'))
        self.assertEqual(buf.getvalue().splitlines(),
                         ['shown', 'live values on RTX 9999: not run '
                          '(RuntimeError)'])

    def test_box_log_entries(self):
        self.regenerate('t-multi', entry='a')
        self.regenerate('t-multi', entry='b')
        packed = self.register('t-multi')
        with mock.patch.object(d2l, 'download_keyed', return_value=packed):
            logs = d2l.hw_box_log('t-multi')
        self.assertEqual(sorted(logs), ['a', 'b'])
        self.assertTrue(d2l.hw_registered('t-multi'))
        self.assertFalse(d2l.hw_registered('t-table'))


class TestCampaigns(unittest.TestCase):
    def test_pending(self):
        log = d2l.hw_campaign_log('a100-microbench')
        self.assertFalse(log)
        self.assertEqual(d2l.hw_status_line(log),
                         'A100 PCIe 40 GB: campaign C-T2 not run yet')
        self.assertEqual(
            d2l.hw_status_line(d2l.hw_campaign_log('mfu-table')),
            'mfu-table: chapter 23\'s measurement, not committed yet')
        with self.assertRaises(KeyError):
            d2l.hw_campaign_log('no-such-log')
        self.assertFalse(d2l.hw_registered('node-p2p'))

    def test_registered_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'm.json')
            with open(path, 'w') as f:
                json.dump({'rows': [1, 2]}, f)
            key = d2l.HW_CAMPAIGNS['mfu-table'][0]
            with mock.patch.dict(d2l.DATA_HUB, {key: ('file://x', None)}), \
                    mock.patch.object(d2l, 'download_keyed',
                                      return_value=path):
                self.assertEqual(d2l.hw_campaign_log('mfu-table'),
                                 {'rows': [1, 2]})
                self.assertEqual(d2l.hw_status_line({'rows': []}), '')


class TestRegistries(unittest.TestCase):
    def test_names(self):
        self.assertEqual(set(d2l.HW_ARTIFACTS), set(REGEN_NAMES))
        self.assertEqual(set(d2l.HW_JOBS), set(REGEN_NAMES))
        for name, (driver, kwargs) in d2l.HW_JOBS.items():
            self.assertTrue(driver.startswith('regen_'), name)
            self.assertIsInstance(kwargs(), dict)
        for name, (key, kind) in d2l.HW_ARTIFACTS.items():
            self.assertTrue(key.startswith('lm-hw-buildbox-'), key)
            self.assertIn(kind, ('tar', 'jsonl'))
        for name, (key, kind, campaign, hw) in d2l.HW_CAMPAIGNS.items():
            self.assertRegex(key, r'^lm-[a-z0-9-]+-v\d+$')
            self.assertRegex(campaign, r'^(C-|chapter )')
        self.assertEqual(set(d2l.HW_STATIC), {
            'chip-specs', 'links', 'tiers', 'systems', 'prices',
            'inferencex-snapshot'})
        self.assertEqual(d2l.HW_PENDING, frozenset())

    def test_chapter_blocks(self):
        """One statement per registry entry, no definitions in a registry
        block, and no function body naming a key or a chapter short name
        it does not read itself (the freshness scan follows literals)."""
        names = (set(d2l.HW_ARTIFACTS) | set(d2l.HW_CAMPAIGNS)
                 | set(d2l.HW_STATIC))
        registries = {'HW_ARTIFACTS', 'HW_CAMPAIGNS', 'HW_STATIC'}
        for path in sorted(glob.glob(os.path.join(CHAPTER, '*.md'))):
            text = open(path, encoding='utf-8').read()
            for code in re.findall(r'```\{\.python[^\n]*\n(.*?)```', text,
                                   re.S):
                code = '\n'.join(l for l in code.splitlines()
                                 if not l.startswith('%%'))
                tree = ast.parse(code)
                targets = [s.targets[0].value for s in tree.body
                           if isinstance(s, ast.Assign)
                           and isinstance(s.targets[0], ast.Subscript)]
                keyed = [t for t in targets if isinstance(t, ast.Attribute)
                         and t.attr in registries]
                if keyed:  # a keyed registry block: only its entries
                    self.assertEqual(len(keyed), len(tree.body), path)
                for node in ast.walk(tree):
                    if not isinstance(node, (ast.FunctionDef,
                                             ast.ClassDef)):
                        continue
                    for c in ast.walk(node):
                        if isinstance(c, ast.Constant) and isinstance(
                                c.value, str):
                            self.assertNotIn('lm-hw-', c.value,
                                             f'{path}: {node.name}')
                            if node.name.startswith(('_hw_', 'hw_box',
                                                     'hw_campaign',
                                                     'hw_registered')):
                                self.assertNotIn(c.value, names,
                                                 f'{path}: {node.name}')


class TestStorage(unittest.TestCase):
    def test_round_trip(self):
        df = pd.DataFrame({'op': ['ar', 'ar', 'a2a'], 'g': [2, 4, 4],
                           'beta': [1.0, 2.0, np.nan]}).set_index(
                               ['op', 'g'])
        df.attrs['label'] = 'x'
        value = dict(frame=df, point=Point(1, (2.0, 3.0)),
                     arr=np.arange(4, dtype=np.float32), n=np.int64(3),
                     flat=pd.DataFrame({'a': [1]}))
        back = d2l._hw_decode(json.loads(json.dumps(d2l._hw_encode(value))))
        pd.testing.assert_frame_equal(back['frame'], df)
        self.assertEqual(back['frame'].attrs['label'], 'x')
        self.assertEqual(back['frame'].loc[('ar', 4), 'beta'], 2.0)
        self.assertEqual(back['point']._fields, ('x', 'y'))  # same shape
        self.assertEqual(tuple(back['point']), (1, (2.0, 3.0)))
        run = d2l.HWRun(['x'], 0, '', '', 1.0, None)  # a d2l type comes back
        self.assertIs(type(d2l._hw_decode(d2l._hw_encode(run))), d2l.HWRun)
        np.testing.assert_array_equal(back['arr'], value['arr'])
        self.assertEqual(back['arr'].dtype, np.float32)
        self.assertEqual(back['n'], 3)
        pd.testing.assert_frame_equal(back['flat'], value['flat'])

    def test_pack(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = os.path.join(tmp, 'k')
            d2l._hw_write(os.path.join(folder, 'a.json.gz'), {'v': 1},
                          dict(config={}))
            with open(os.path.join(folder, 'topo.txt'), 'w') as f:
                f.write('GPU0 X\n')
            out = d2l._hw_pack(folder, 'tar', os.path.join(tmp, 'k.tar'))
            again = d2l._hw_pack(folder, 'tar', os.path.join(tmp, 'k2.tar'))
            self.assertEqual(read(out, 'rb'), read(again, 'rb'))
            x = d2l._hw_unpack(out, 'tar')
            self.assertEqual(read(os.path.join(x, 'topo.txt')),
                             'GPU0 X\n')
            self.assertEqual(d2l._hw_read(os.path.join(x, 'a.json.gz'))[0],
                             {'v': 1})


class TestCombine(unittest.TestCase):
    def test_medians(self):
        frames = [pd.DataFrame({'n': [1, 2], 'ms': [1.0 + i, 2.0 * i],
                                'kind': ['a', 'b']}) for i in range(3)]
        out = d2l._hw_combine(frames)
        self.assertEqual(list(out.ms), [2.0, 2.0])
        self.assertEqual(list(out.n), [1, 2])
        self.assertEqual(out.attrs['n_proc'], 3)
        nested = d2l._hw_combine([dict(t=Point(i, 'x'), v=[i, 5],
                                       arr=np.array([i, 2 * i]))
                                  for i in (1, 5, 3)])
        self.assertEqual(nested['t'], Point(3.0, 'x'))
        self.assertEqual(nested['v'], [3.0, 5])
        np.testing.assert_array_equal(nested['arr'], [3.0, 6.0])
        with self.assertRaisesRegex(ValueError, 'kind'):
            d2l._hw_combine([frames[0], frames[1].assign(kind=['a', 'c'])])


class TestConfigs(unittest.TestCase):
    def test_unpinned(self):
        with self.assertRaisesRegex(KeyError, 'not pinned'):
            d2l.hw_config('Qwen/Qwen3-31B-Nonexistent')

    def test_dense_70b(self):
        cfg = d2l.HW_SHAPES['dense-70b']
        self.assertEqual(cfg.hidden_size, 8192)
        w = d2l.config_memory(cfg, 1, 1)['weights']
        self.assertAlmostEqual(w / 2 / 1e9, 70.55, places=1)

    def test_pinned_config(self):
        if not os.environ.get('HF_HOME'):
            self.skipTest('HF_HOME not set')
        with mock.patch.dict(os.environ, {'HF_HUB_OFFLINE': '1'}):
            try:
                cfg = d2l.hw_config('Qwen/Qwen3-30B-A3B')
            except OSError:
                self.skipTest('config not cached')
        self.assertEqual((cfg.num_experts, cfg.num_experts_per_tok), (128, 8))


class TestFit(unittest.TestCase):
    def test_fit_alpha_beta(self):
        rng = np.random.default_rng(0)
        s = np.repeat(2.0 ** np.arange(10, 31, 2), 5)
        alpha, beta = 8e-6, 22e9
        t = (alpha + s / beta) * (1 + 0.01 * rng.standard_normal(len(s)))
        a, b, cov = d2l.fit_alpha_beta(s, t)
        self.assertLess(abs(a / alpha - 1), 0.02)
        self.assertLess(abs(b / beta - 1), 0.02)
        self.assertEqual(cov.shape, (2, 2))
        self.assertTrue(np.all(np.diag(cov) > 0))
        a2, b2, _ = d2l.fit_alpha_beta(s, alpha + s / beta)
        self.assertAlmostEqual(a2 / alpha, 1, places=9)
        self.assertAlmostEqual(b2 / beta, 1, places=9)


class TestTable(unittest.TestCase):
    def test_show_table(self):
        df = pd.DataFrame({'gpu': ['A', 'B'], 'gbs': [912.345, 0.01234],
                           'n': [12000, 3]})
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            d2l.show_table(df, headers={'gbs': 'GB/s'})
        lines = buf.getvalue().splitlines()
        self.assertIn('GB/s', lines[0])
        self.assertIn('912', lines[1])
        self.assertIn('12,000', lines[1])
        self.assertIn('0.0123', lines[2])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            d2l.show_table(pd.DataFrame({'x': range(20)}), max_rows=15)
        self.assertEqual(buf.getvalue().splitlines()[-1], '... 5 more rows')
        with self.assertRaisesRegex(ValueError, 'wide'):
            d2l.show_table(pd.DataFrame({'x' * 120: [1]}))


class TestTriton(unittest.TestCase):
    def test_stand_in(self):
        tl = d2l._NoTriton()

        def kernel(x, BLOCK: tl.constexpr):
            pass
        k = tl.jit(kernel)
        with self.assertRaisesRegex(RuntimeError, 'not installed'):
            k[(1,)](1, BLOCK=2)
        auto = tl.autotune(configs=[tl.Config({})], key=['n'])(kernel)
        with self.assertRaisesRegex(RuntimeError, 'not installed'):
            auto[(1,)](1)

    def test_binding(self):
        try:
            import triton
        except ImportError:
            self.assertIsInstance(d2l.triton, d2l._NoTriton)
        else:
            self.assertIs(d2l.triton, triton)


if __name__ == '__main__':
    unittest.main()
