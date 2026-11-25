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
import math
import os
import re
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

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

    def test_regeneration_refuses_unregistered_dependency(self):
        """A tools/hw_regen.py process of one name never reads another
        name's unregistered folder (a notebook rerun still may)."""
        self.regenerate('t-multi', entry='ladder')
        os.environ['D2L_RERUN'] = 't-table'
        with mock.patch.dict(os.environ, {'D2L_HW_PROC': '0'}):
            with self.assertRaisesRegex(RuntimeError, 'not registered'):
                d2l.hw_box_log('t-multi', 'ladder')
            with mock.patch.object(d2l, 'HW_PENDING',
                                   frozenset({'t-multi'})):
                self.assertIsNotNone(d2l._hw_folder('t-multi'))

    def test_config_writes_campaign_log(self):
        with mock.patch.dict(d2l.HW_CAMPAIGNS, {
                't-camp': ('lm-hw-t-camp-v1', 'tar', 'C-T9', 'A100 PCIe 40 GB')}), \
                mock.patch.dict(d2l.HW_CONFIG_LOGS,
                                {('t-multi', 't9'): 't-camp'}, clear=True):
            self.assertEqual(d2l._hw_regen_key('t-multi'),
                             ('lm-hw-test-multi-v1', 'tar'))
            with mock.patch.dict(os.environ, {'D2L_HW_CONFIG': 't9'}):
                self.assertEqual(os.path.basename(
                    d2l._hw_regen_dir('t-multi')), 'lm-hw-t-camp-v1')
                r = self.regenerate('t-multi', entry='ladder')
                _, meta = d2l._hw_read(os.path.join(
                    self.out, 'lm-hw-t-camp-v1', 'ladder.json.gz'))
                self.assertEqual(meta['run'], 't9')
            with mock.patch.dict(os.environ, {'D2L_HW_CONFIG': 'nope'}):
                with self.assertRaisesRegex(KeyError, 'nope'):
                    d2l._hw_regen_dir('t-multi')

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
        lines = buf.getvalue().splitlines()
        self.assertEqual(lines[0], 'shown')
        self.assertRegex(lines[1], r'^committed run on .+; live values on '
                         r'RTX 9999$')  # the committed numbers' GPU named
        self.assertEqual(lines[2:], ['live values on RTX 9999: not run '
                                     '(RuntimeError)'])
        buf = io.StringIO()
        with mock.patch.object(d2l, 'download_keyed', return_value=packed), \
                mock.patch.object(torch.cuda, 'is_available',
                                  return_value=True), \
                mock.patch.object(torch.cuda, 'get_device_name',
                                  return_value='NVIDIA GeForce RTX 9999'), \
                contextlib.redirect_stdout(buf):
            d2l._hw_result('t-multi', self.compute, dict(n=3),
                           entry='ladder', live_check=lambda r, s: 'ok',
                           live_label='L1')
        self.assertEqual(buf.getvalue().splitlines()[-1],
                         'live values (L1) on RTX 9999: ok')

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
        for (name, config), log in d2l.HW_CONFIG_LOGS.items():
            self.assertIn(name, d2l.HW_JOBS)
            self.assertIn(config, d2l._HW_TIERS)
            self.assertIn(log, d2l.HW_CAMPAIGNS)

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
        # spread (max - min) / median per row of each column that differs
        np.testing.assert_allclose(out.attrs['spread']['ms'], [1.0, 2.0])
        self.assertNotIn('n', out.attrs['spread'])


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

    def test_fmt_significant_digits(self):
        self.assertEqual(d2l._hw_fmt(533907.3), '534,000')
        self.assertEqual(d2l._hw_fmt(1234.5), '1,230')
        self.assertEqual(d2l._hw_fmt(912.345), '912')
        self.assertEqual(d2l._hw_fmt(128256), '128,256')  # an exact count
        self.assertEqual(d2l._hw_fmt(1800.0), '1,800')  # a whole figure
        self.assertEqual(d2l._hw_fmt(np.nan), '')


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


# ---- 22.1 (accelerator-design): the chip table, ratios, floors, benches ----

SPECS_CSV = os.path.join(REPO, 'tools', 'lm_data', 'hw', 'chip_specs.csv')


def committed_specs():
    return pd.read_csv(SPECS_CSV).set_index('chip')


class TestChipSpecs(unittest.TestCase):
    def test_contract(self):
        df = committed_specs()
        d2l._check_specs(df)  # sources, dates, kinds, the book's four rows
        for col in ('p16_tflops', 'p8_tflops', 'p4_tflops'):
            self.assertTrue((df[col].dropna() > 0).all())
        self.assertEqual(df.loc['A100 PCIe 40 GB', 'mem_type'], 'HBM2')
        self.assertTrue(np.isnan(df.loc['A100 PCIe 40 GB', 'p8_tflops']))

    def test_book_table_regression(self):
        df = committed_specs()
        bad = df.copy()
        bad.loc['B200', 'p16_tflops'] = 4500.0  # a sparse figure
        with self.assertRaisesRegex(AssertionError, 'B200 p16'):
            d2l._check_specs(bad)
        bad = df.copy()
        bad.loc['RTX 4090', 'source_url'] = np.nan
        with self.assertRaisesRegex(AssertionError, 'source_url'):
            d2l._check_specs(bad)
        self.assertTrue(d2l._book_round(1.008, '1.0'))
        self.assertTrue(d2l._book_round(989.4, '989'))
        self.assertFalse(d2l._book_round(3.4, '3.35'))

    def test_registered_sha1(self):
        """The registered table is the committed CSV."""
        key, kind = d2l.HW_STATIC['chip-specs']
        self.assertEqual(kind, 'csv')
        self.assertIn(key, d2l.DATA_HUB)
        self.assertEqual(d2l.DATA_HUB[key][1], d2l.hashlib.sha1(
            read(SPECS_CSV, 'rb')).hexdigest())

    def test_derive_ratios(self):
        r = d2l.derive_ratios(committed_specs())
        self.assertAlmostEqual(r.loc['A100 PCIe 40 GB', 'ridge16'], 200.6, 1)
        self.assertAlmostEqual(r.loc['A100 PCIe 40 GB', 'sweep_ms'], 25.7, 1)
        self.assertLessEqual(abs(r.loc['RTX 4090', 'ridge16'] - 164), 1)
        self.assertAlmostEqual(r.loc['RTX 4090', 'sweep_ms'], 23.8, 1)
        self.assertAlmostEqual(r.loc['A100 SXM 80 GB', 'sweep_ms'], 39.2, 1)
        self.assertTrue(np.isnan(r.loc['Groq LPU', 'ridge16']))


def small_config(L=4, d=256, H=4, H_kv=2, V=1000):
    return d2l.HWShape(model_type='qwen3', hidden_size=d, num_hidden_layers=L,
                       num_attention_heads=H, num_key_value_heads=H_kv,
                       head_dim=d // H, intermediate_size=4 * d,
                       vocab_size=V, tie_word_embeddings=False,
                       max_position_embeddings=65536)


class TestFloors(unittest.TestCase):
    def specs(self, c=1):
        return pd.DataFrame({'mem_gb': [0.4 * c, 0.001],
                             'mem_bw_tbs': [1.0 * c, 1.0]},
                            index=['chip', 'tiny'])

    def test_sweep_floor(self):
        cfg, L = small_config(), 512
        mem = d2l.config_memory(cfg, L, 1)
        W, kv = mem['weights'], mem['kv_cache']
        f = d2l.sweep_floor(self.specs(), cfg, L)
        self.assertTrue(np.isnan(f.loc['tiny', 'b_max']))  # weights too big
        b = f.loc['chip', 'b_max']
        self.assertEqual(b, np.floor((0.9 * 0.4e9 - W) / kv))
        self.assertLessEqual(W + b * kv, 0.9 * 0.4e9)
        self.assertAlmostEqual(f.loc['chip', 'sweep_ms'], 0.4, 9)
        tau = f.loc['chip', 'floor_ms']
        self.assertLessEqual(tau, 0.9 * 0.4 + 1e-12)
        self.assertGreater(tau, 0.9 * 0.4 - kv / 1e12 * 1e3)
        self.assertAlmostEqual(f.loc['chip', 'user_tps_max'], 1e3 / tau)
        for c in (2, 4, 8):  # tensor parallelism: the floor is the chip's
            g = d2l.sweep_floor(self.specs(c), cfg, L)
            self.assertAlmostEqual(g.loc['chip', 'sweep_ms'], 0.4, 9)
            self.assertLess(math_fabs(g.loc['chip', 'floor_ms'] - 0.36),
                            kv / (c * 1e12) * 1e3 + 1e-12)

    def test_sram_chips(self):
        self.assertEqual(d2l.sram_chips_needed(16.4e9, 0.23e9, 0.8), 90)
        self.assertEqual(d2l.sram_chips_needed(10, 10, 1.0), 1)
        self.assertEqual(d2l.sram_chips_needed(11, 10, 1.0), 2)
        W, S = 141.1e9, 44e9
        prev = None
        for eta in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            n = d2l.sram_chips_needed(W, S, eta)
            self.assertGreaterEqual(n * S * eta, W)
            if prev is not None:
                self.assertLessEqual(n, prev)
            prev = n


def math_fabs(x):
    return float(np.abs(x))


class TestBenchHelpers(unittest.TestCase):
    def test_matmul_flops(self):
        from torch.utils.flop_counter import FlopCounterMode
        m, n, k = 24, 40, 56
        for fmt in ('fp32', 'bf16'):
            fn = d2l.hw_matmul_op(m, n, k, fmt, device='cpu')
            with FlopCounterMode(display=False) as counter:
                out = fn()
            self.assertEqual(counter.get_total_flops(), 2 * m * n * k)
            self.assertEqual(tuple(out.shape), (m, n))

    def test_fp8_scale(self):
        x = torch.randn(64, 64) * 1000
        x8, s = d2l.fp8_per_tensor(x)
        self.assertEqual(x8.dtype, torch.float8_e4m3fn)
        self.assertAlmostEqual(float(x8.float().abs().max()), 448.0, 0)
        err = (x8.float() * s - x).abs().max() / x.abs().max()
        self.assertLess(float(err), 0.07)  # E4M3: 3 mantissa bits

    def test_stats_and_formatting(self):
        med, mad = d2l._hw_stats([1.0, 2.0, 3.0, 10.0])
        self.assertEqual((med, mad), (2.5, 1.0))
        self.assertEqual(d2l._hw_sig(927.4), '930')
        self.assertEqual(d2l._hw_sig(4812.0), '4,800')
        self.assertEqual(d2l._hw_sig(0.01234), '0.012')
        self.assertEqual(d2l._hw_bytes(2**30), '1 GiB')
        self.assertEqual(d2l._hw_bytes(4096), '4 KiB')

    def test_agreement(self):
        pts = [('64 MiB', 100.0, 108.0, 0.001), ('1 GiB', 100.0, 80.0, 0.01)]
        line = d2l._hw_agreement(pts, True, 'GB/s')
        self.assertTrue(line.startswith('2 points, largest difference 20%'))
        self.assertIn('differs: 1 GiB 80 vs 100 GB/s', line)  # > 10%
        self.assertNotIn('64 MiB', line)
        line = d2l._hw_agreement(pts[:1], False, 'GB/s')
        self.assertEqual(line, '64 MiB 110 GB/s')  # live value, 2 digits

    def test_raw_round_trip(self):
        text = 'line one\nline two\n' * 50
        packed = d2l._hw_pack_raw(text)
        self.assertNotIn('line', packed)
        self.assertEqual(d2l._hw_unpack_raw(packed), text)
        self.assertEqual(packed, d2l._hw_pack_raw(text))  # mtime fixed


def synthetic_ladder():
    sizes = [2**k for k in range(18, 33)]
    rows = [dict(bytes=s, kind='read', gbs=4000.0 if s <= 2**25 else 900.0,
                 ms=1.0, mad=0.01) for s in sizes]
    rows += [dict(bytes=s, kind='copy', gbs=3000.0 if s <= 2**25 else 850.0,
                  ms=1.0, mad=0.01) for s in sizes]
    df = pd.DataFrame(rows)
    df.attrs.update(gpu='RTX 4090', l2_bytes=72 * 2**20,
                    mem_bytes=25386352640)
    return df


def specs_patch():
    """d2l.chip_specs() as the committed CSV (no download)."""
    return mock.patch.object(d2l, 'chip_specs', committed_specs)


class TestDisplays(unittest.TestCase):
    def test_plateaus(self):
        l2, dram, copy = d2l._ladder_plateaus(synthetic_ladder())
        self.assertEqual((l2, dram, copy), (4000.0, 900.0, 850.0))

    def test_datasheet_vs_measured(self):
        specs = committed_specs()
        dtypes = pd.DataFrame({'format': ['fp32', 'tf32', 'bf16', 'fp16',
                                          'fp8'],
                               'tflops': [60.0, 80.0, 150.0, 150.0, 300.0]})
        dtypes.attrs.update(gpu='RTX 4090', mem_bytes=25386352640)
        smem = dict(tbs=30.0, peak_tbs=45.0, n_sm=128, sm_mhz=2750.0,
                    gpu='RTX 4090')
        entries = dict(ladder=synthetic_ladder(), dtypes=dtypes, smem=smem)
        t = d2l._hw_sheet_table(specs.loc['RTX 4090'], entries)
        self.assertEqual(len(t), 8)
        self.assertAlmostEqual(t.loc['DRAM read (GB/s)', 'fraction'],
                               900 / 1008)
        self.assertAlmostEqual(t.loc['BF16 (TFLOP/s)', 'fraction'],
                               150 / 165.2)
        self.assertAlmostEqual(t.loc['L1 loads (TB/s)*', 'fraction'], 30 / 45)
        self.assertTrue(np.isnan(t.loc['L2 read (GB/s)', 'datasheet']))
        with specs_patch(), quiet() as buf:
            out = d2l.datasheet_vs_measured(specs.loc['RTX 4090'],
                                            synthetic_ladder(), dtypes, smem)
        self.assertIsNone(out)  # prints only
        lines = buf.getvalue().splitlines()
        self.assertLessEqual(len(lines), 15)
        self.assertLessEqual(max(map(len, lines)), 79)
        self.assertEqual(lines[-1], '* bound, not a datasheet figure: 128 '
                         'SMs x 128 B/clk x 2.75 GHz')
        # a log measured on another GPU is refused, not compared
        other = synthetic_ladder()
        other.attrs.update(gpu='A100-SXM4-80GB', mem_bytes=80 * 2**30)
        with specs_patch(), quiet() as buf:
            d2l.datasheet_vs_measured(specs.loc['A100 PCIe 40 GB'],
                                      dict(ladder=other))
        self.assertEqual(buf.getvalue(), 'measured on A100-SXM4-80GB (80 GiB)'
                         ', not the A100 PCIe 40 GB: no datasheet comparison\n')
        pending = d2l.hw_campaign_log('a100-microbench')
        with specs_patch(), quiet() as buf:
            d2l.datasheet_vs_measured(specs.loc['A100 PCIe 40 GB'], pending)
        self.assertTrue(buf.getvalue().startswith(
            'A100 PCIe 40 GB: campaign C-T2 not run yet'))

    def test_chip_row(self):
        specs = committed_specs()
        row = lambda gpu, mem: d2l._hw_chip_row(gpu, mem, specs)
        self.assertEqual(row('RTX 4090', 25386352640), 'RTX 4090')
        self.assertEqual(row('A100-PCIE-40GB', 42505273344),
                         'A100 PCIe 40 GB')
        self.assertIsNone(row('A100-SXM4-80GB', 85899345920))
        self.assertIsNone(row('A100-PCIE-40GB', 85899345920))  # memory
        self.assertEqual(row('A100-PCIE-40GB', None), 'A100 PCIe 40 GB')
        self.assertIsNone(row(None, None))

    def test_show_dtypes(self):
        r = pd.DataFrame({'format': ['bf16', 'fp8'], 'tflops': [175.0, 350.0],
                          'sm_mhz': [2790.0, 2805.0],
                          'power_w': [431.0, 445.0]})
        r.attrs.update(gpu='RTX 4090', mem_bytes=25386352640,
                       spread={'tflops': [0.012, 0.004]})
        with specs_patch(), quiet() as buf:
            d2l._show_dtypes(r)
        lines = buf.getvalue().splitlines()
        self.assertEqual(lines[0].split(), ['format', 'TFLOP/s', 'spread',
                                            'datasheet', 'fraction', 'SM',
                                            'GHz', 'W'])
        self.assertEqual(lines[1].split(), ['bf16', '180', '1.2%', '165',
                                            '106%', '2.79', '431'])
        self.assertEqual(lines[-1],
                         'datasheet peaks assume the 2.52 GHz boost clock')

    def test_show_env(self):
        env = dict(gpu='NVIDIA GeForce RTX 4090', n_gpu=1, n_host=4, n_sm=128,
                   clocks_max_sm=3120, clocks_max_mem=10501, power_limit=450,
                   power_min_limit=150, power_max_limit=600,
                   pcie_link_gen_max=4, pcie_link_width_max=16,
                   pcie_link_gen_current=1, pcie_link_width_current=16,
                   driver='595.71.05', cuda='12.8', torch='2.11.0',
                   nccl='2.28.9', triton='3.6.0')
        with quiet() as buf:
            d2l.show_env(env)
        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), 6)
        self.assertEqual(lines[0], 'GPU: RTX 4090 (1 visible, 4 on the host)')
        self.assertIn('Gen 4 x16 (max)', lines[4])
        self.assertEqual(lines[5], 'driver 595.71.05, NCCL 2.28.9, CUDA 12.8, '
                         'torch 2.11.0, Triton 3.6.0')
        partial = dict(env, clocks_max_sm=None, power_limit=None)  # no smi
        with quiet() as buf:
            d2l.show_env(partial)
        self.assertIn('max SM clock: n/a MHz', buf.getvalue())
        with quiet() as buf:  # a reader without a GPU
            d2l.show_env(dict(gpu=None, n_gpu=0, cuda=None, torch='2.11.0',
                              triton=None))
        self.assertEqual(buf.getvalue(), 'no CUDA GPU visible; CUDA None, '
                         'torch 2.11.0, Triton absent\n')

    def test_show_table_no_index_name(self):
        with quiet() as buf:
            d2l.show_table(pd.DataFrame({'x': [1.5]}, index=pd.Index(
                ['a'], name='chip')))
        self.assertEqual(len(buf.getvalue().splitlines()), 2)

    def test_plots(self):
        import matplotlib
        matplotlib.use('Agg')
        d2l.plot_chip_plane(d2l.derive_ratios(committed_specs()))
        with quiet() as buf:
            d2l.plot_ladder(synthetic_ladder(), committed_specs(),
                            reference=d2l.hw_campaign_log('a100-microbench'))
        self.assertEqual(buf.getvalue(),
                         'A100 PCIe 40 GB: campaign C-T2 pending\n')
        labels = [t.get_text() for t in
                  d2l.plt.gca().get_legend().get_texts()]
        self.assertEqual(labels, ['read', 'copy (read + write)',
                                  'datasheet 1,008 GB/s',
                                  'L2 edge (36-72 MiB)'])
        other = synthetic_ladder()
        other.attrs.update(gpu='A100-SXM4-80GB', mem_bytes=80 * 2**30)
        with quiet() as buf:
            d2l.plot_ladder(synthetic_ladder(), committed_specs(),
                            reference=dict(ladder=other))
        self.assertIn('not plotted', buf.getvalue())
        with specs_patch(), quiet() as buf:
            d2l._show_ladder(synthetic_ladder())
        lines = buf.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn('4.4x the DRAM read', lines[0])
        self.assertIn('89% of the datasheet 1,008 GB/s', lines[1])
        lat = pd.DataFrame({'bytes': [2**k for k in range(12, 31)],
                            'ns_per_load': [17.0] * 4 + [60.0] * 4
                            + [97.0] * 5 + [150.0] * 3 + [220.0] * 3})
        lat.attrs.update(l2_bytes=72 * 2**20, l1_bytes=128 * 2**10)
        self.assertEqual(d2l._latency_plateaus(lat), (17.0, 97.0, 220.0))
        with quiet() as buf:
            d2l._show_latency(lat)
        self.assertEqual(buf.getvalue(), 'latency per dependent load: L1 17 '
                         'ns, L2 97 ns, DRAM 220 ns\n')
        d2l.plot_latency(lat)
        d2l.plt.close('all')


class TestBenchLog(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(d2l, 'env_fingerprint',
                                  return_value=dict(gpu='x')), \
                mock.patch.object(d2l, '_hw_contention',
                                  side_effect=[(0, 0), (1, 0)]):
            path = os.path.join(tmp, 'a', 'log.jsonl')
            with d2l.BenchLog(path, 'torch', 'T1') as log:
                log.record('cmd one', dict(n=1), dict(gbs=1.5), raw='out\n')
            recs = d2l.BenchLog.read(path)
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual((r['tool'], r['tier'], r['cmd']),
                         ('torch', 'T1', 'cmd one'))
        self.assertEqual(r['results'], dict(gbs=1.5))
        self.assertEqual(r['raw'], 'out\n')
        self.assertTrue(r['contended'])  # foreign process at the record
        self.assertEqual(r['env'], dict(gpu='x'))

    def test_power_trace_without_gpu(self):
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            with d2l._hw_power_trace(10) as samples:
                pass
        self.assertEqual(samples, [])

    def test_driver_needs_gpu(self):
        with mock.patch.object(torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA'):
                d2l.regen_microbench()
        self.assertEqual(d2l.HW_JOBS['box-microbench'][0], 'regen_microbench')
        with mock.patch.object(torch.cuda, 'is_available', return_value=True), \
                mock.patch.dict(os.environ, {'D2L_HW_CONFIG': ''}):
            with self.assertRaisesRegex(ValueError, 'D2L_HW_CONFIG=t2'):
                d2l.regen_microbench('t2')


# ---- 22.2 (kernels): tiles, GEMM shapes, chunked attention, displays ----

def kernels_frame(rows, **attrs):
    df = pd.DataFrame(rows)
    df.attrs.update(gpu='RTX 4090', mem_bytes=24 * 2**30, n_sm=128, **attrs)
    return df


class TestTiles(unittest.TestCase):
    def test_tile_intensity_counts_bytes(self):
        """The formula equals a count of the bytes a tile loads slab by
        slab and the FLOPs it does (200 random shapes, seed 0)."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            Tm, Tn, Tk = (int(x) for x in rng.integers(1, 9, 3) * 16)
            K, b = Tk * int(rng.integers(1, 20)), int(rng.choice([1, 2, 4]))
            loaded = flops = 0
            for _ in range(0, K, Tk):  # one K-slab at a time
                loaded += b * Tk * (Tm + Tn)
                flops += 2 * Tm * Tn * Tk
            self.assertAlmostEqual(d2l.tile_intensity(Tm, Tn, b),
                                   flops / loaded, 9)
        self.assertEqual(d2l.tile_intensity(128, 128), 64)

    def test_tile_roof(self):
        t = d2l.tile_roof(committed_specs().loc[['RTX 4090',
                                                 'A100 PCIe 40 GB']])
        self.assertEqual(t.loc['RTX 4090', 'T'], 328)
        self.assertEqual(t.loc['A100 PCIe 40 GB', 'T'], 402)
        self.assertAlmostEqual(t.loc['RTX 4090', 'acc_kib'],
                               4 * 328**2 / 1024, 9)
        self.assertAlmostEqual(t.loc['A100 PCIe 40 GB', 'ridge'], 200.6, 1)
        self.assertTrue((t.acc_kib > t.rf_kib).all())


class TestGemmShapes(unittest.TestCase):
    def test_flop_counter(self):
        """Exact against FlopCounterMode's matmul FLOPs of a 2-layer Qwen3
        forward pass on the CPU (the attention scores counted apart)."""
        from torch.utils.flop_counter import FlopCounterMode
        from transformers import Qwen3Config, Qwen3ForCausalLM
        cfg = Qwen3Config(hidden_size=64, intermediate_size=96,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=16, vocab_size=500,
                          tie_word_embeddings=True, attn_implementation='sdpa')
        torch.manual_seed(0)
        model = Qwen3ForCausalLM(cfg).eval()
        T = 24
        with torch.no_grad(), FlopCounterMode(display=False) as c:
            model(torch.randint(0, 500, (1, T)))
        mm = sum(n for op, n in c.get_flop_counts()['Global'].items()
                 if str(op) in ('aten.mm', 'aten.addmm', 'aten.bmm'))
        g = d2l.gemm_shapes(cfg, T)
        self.assertEqual(round(g.gflop.sum() * 1e9), mm)
        self.assertEqual(list(g.name), ['QKV', 'out', 'gate-up', 'down',
                                        'head'])
        self.assertTrue((g.M == T).all())

    def test_flops_per_token(self):
        """Tied: sum(GFLOP) / tokens = flops_per_token / 3. Untied: chapter
        21's convention also counts the embedding table (V d) as a matrix,
        which no product of the forward pass multiplies."""
        R = d2l.r2_config('r2').arch
        g = d2l.gemm_shapes(R, 4096)
        self.assertAlmostEqual(g.gflop.sum() * 1e9 / 4096,
                               d2l.flops_per_token(R) / 3, 3)
        cfg = d2l.HW_SHAPES['dense-70b']
        g = d2l.gemm_shapes(cfg, 1)
        V, d = cfg.vocab_size, cfg.hidden_size
        self.assertAlmostEqual(g.gflop.sum() * 1e9,
                               d2l.flops_per_token(cfg) / 3 - 2 * V * d, 0)


class TestChunkedAttention(unittest.TestCase):
    def test_against_math_attention(self):
        """GQA, a sequence that is not a multiple of the chunk, FP32 on the
        CPU: equal to the math attention to 1e-5."""
        from torch.nn.attention import SDPBackend, sdpa_kernel
        g = torch.Generator().manual_seed(0)
        B, H, H_kv, N, d = 2, 6, 2, 300, 16
        q = torch.randn(B, H, N, d, generator=g)
        k, v = (torch.randn(B, H_kv, N, d, generator=g) for _ in range(2))
        out = d2l.chunked_causal_attention(q, k, v, chunk=128)
        with sdpa_kernel(SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(
                q, k.repeat_interleave(3, 1), v.repeat_interleave(3, 1),
                is_causal=True)
        torch.testing.assert_close(out, ref, atol=1e-5, rtol=0)
        K, V = k[1].repeat_interleave(3, 0), v[1].repeat_interleave(3, 0)
        torch.testing.assert_close(d2l._attention_fp32(q[1], K, V, 100, 64),
                                   ref[1, :, 100:164], atol=1e-5, rtol=0)

    def test_flops(self):
        q = torch.zeros(8, 12, 4096, 64)
        self.assertEqual(d2l._attn_flops(q), 4 * 8 * 12 * 4096**2 * 64 // 2)


class TestKernelDisplays(unittest.TestCase):
    def out(self, fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(*args, **kwargs)
        lines = buf.getvalue().splitlines()
        for line in lines:
            self.assertLessEqual(len(line), 79, line)
        return lines

    def test_check_close(self):
        a = torch.zeros(3)
        self.assertIn('max abs error 1.0e-06',
                      self.out(d2l.check_close, a, a + 1e-6, atol=1e-5)[0])
        with quiet(), self.assertRaises(AssertionError):
            d2l.check_close(a, a + 1e-3, atol=1e-5)

    def test_attention_shapes(self):
        R = d2l.r2_config('r2').arch
        lines = self.out(d2l.attention_shapes, {
            'reference': (R, 8), 'dense 70B': (d2l.HW_SHAPES['dense-70b'], 1)},
            4096, l2_bytes=72 * 2**20)
        self.assertEqual(lines[0], 'L2 of this GPU: 72 MiB')
        self.assertRegex(lines[2], r'reference\s+8\s+12\s+4\s+64\s+1\s+1\.4%')
        self.assertRegex(lines[3], r'2\s+2\.8%$')

    def test_sawtooth_period(self):
        N = np.arange(1024, 8193, 64)
        rng = np.random.default_rng(0)
        for p in (256, 512, 1024):
            y = 150 + 0.002 * N + 10 * ((N - 1024) % p) / p
            self.assertEqual(d2l._sawtooth_period(
                N, y + rng.normal(0, 2, len(N))), p)
        self.assertIsNone(d2l._sawtooth_period(N, rng.normal(0, 1, len(N))))

    def test_short_kernel(self):
        self.assertEqual(d2l._short_kernel(
            'void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<64,'
            ' 128>, true>(Flash_fwd_params)'), 'flash_fwd_kernel')
        self.assertEqual(d2l._short_kernel(
            'void cutlass::Kernel2<cutlass_80_tensorop_bf16_s16816gemm_relu_'
            'bf16_64x64_32x6_nn_align8>(Params)'),
            'cutlass_80_tensorop_bf16_s16816gemm_relu_bf16_64x64_32x6_...')
        self.assertEqual(d2l._short_kernel('ampere_bf16_s16816gemm_bf16_'
                                           '128x128_ldg8_f2f_tn'),
                         'ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_tn')

    def flash(self):
        rows = []
        for shape, sdpa in (('reference', 140.0), ('Qwen3-8B', 130.0)):
            for i, (bq, bk) in enumerate(((64, 32), (64, 64), (128, 64),
                                          (128, 128))):
                err = 'shared memory 128 KiB > 99 KiB' if bk == 128 else ''
                tf = math.nan if err else 100.0 + 10 * i
                rows.append(dict(shape=shape, arm=f'{bq} x {bk}, 4 warps',
                                 block_q=bq, block_k=bk, num_warps=4,
                                 tflops=tf, ms=1.0, error=err))
            rows.append(dict(shape=shape, arm='SDPA', block_q=0, block_k=0,
                             num_warps=0, tflops=sdpa, ms=1.0, error=''))
            rows.append(dict(shape=shape, arm='compiled', block_q=0,
                             block_k=0, num_warps=0, tflops=11.0, ms=1.0,
                             error=''))
        return kernels_frame(rows)

    def test_show_flash(self):
        lines = self.out(d2l._show_flash, self.flash())
        self.assertRegex(lines[1], r'reference\s+128 x 64, 4 warps\s+120\s+86%')
        self.assertTrue(any('SDPA' in l and '100%' in l for l in lines))
        self.assertEqual(' '.join(l.strip() for l in lines[-2:]),
                         'did not compile: reference 128 x 128, 4 warps, '
                         'Qwen3-8B 128 x 128, 4 warps (shared memory 128 KiB '
                         '> 99 KiB)')
        s = d2l._flash_summary(self.flash())
        self.assertEqual(len(s), 10)  # 3 best (one worst among them), 2 more

    def test_campaign_status(self):
        pending = d2l.CampaignPending('a100-microbench', 'C-T2',
                                      'A100 PCIe 40 GB')
        self.assertEqual(self.out(d2l.flash_sweep_table, pending),
                         ['A100 PCIe 40 GB: campaign C-T2 not run yet'])
        lines = self.out(d2l.flash_sweep_table, {'flash-sweep': self.flash()})
        self.assertIn('of SDPA', lines[0])

    def test_show_attention_and_fns(self):
        r = kernels_frame([
            dict(arm='chunked', ms=105.0, mad=0.1, tflops=2.0),
            dict(arm='SDPA', ms=1.5, mad=0.01, tflops=136.0)])
        lines = self.out(d2l._show_attention, r)
        self.assertEqual(len(lines), 4)  # no launch counts without a GPU
        lines = self.out(d2l._show_attention, r, {
            'chunked': (75, 'triton_poi'), 'SDPA': (1, 'flash_fwd_kernel')})
        self.assertRegex(lines[1], r'chunked\s+105\s+2\.0+\s+75')
        self.assertEqual(lines[-1], 'SDPA ran flash_fwd_kernel (the '
                         'FLASH_ATTENTION backend)')
        fns = kernels_frame([dict(arm='Triton', ms=1.2, mad=0.0, gbs=894.0)])
        with mock.patch.object(d2l, '_dram_read_gbs', lambda: 950.0):
            lines = self.out(d2l._show_fns, fns, {'Triton': (1, 'k')})
        self.assertRegex(lines[1], r'Triton\s+1,200\s+894\s+94%\s+1')

    def test_show_wave(self):
        N = np.arange(1024, 8193, 64)
        saw = 160 * (N / 512) / np.ceil(N / 512)  # one wave per 512 columns
        wave = kernels_frame(pd.concat([
            pd.DataFrame(dict(arm='cuBLAS', N=N, tflops=158 + (N % 128 > 0))),
            pd.DataFrame(dict(arm='Triton 128 x 128', N=N, tflops=saw))]),
            M=4096, K=4096)
        lines = self.out(d2l._show_wave, wave)
        self.assertEqual(lines[0], 'cuBLAS: 158-159 TFLOP/s')
        self.assertEqual(lines[1], 'Triton 128 x 128: 113-160 TFLOP/s, '
                         'saw-tooth period 512 columns (measured)')
        self.assertEqual(lines[2], 'one wave: n_SM T_m T_n / M = 128 x 128 '
                         'x 128 / 4,096 = 512 columns')
        self.assertEqual(len(lines), 3)
        lines = self.out(d2l._show_wave, wave,
                         {1024: '128 x 128', 2048: '128 x 64'})
        self.assertEqual(lines[3], 'cuBLAS tiles by kernel name: 128 x 128, '
                         '128 x 64')

    def test_empty_profiler_trace(self):
        """A profiler that records no kernel raises in _kernel_launches,
        drops the launch column (None) and fails inductor_kernels loudly,
        never a display of '0 kernels'."""
        class Empty:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def events(self):
                return []
        with mock.patch.object(torch.profiler, 'profile',
                               lambda **kw: Empty()), \
                mock.patch.object(torch.cuda, 'synchronize', lambda: None), \
                mock.patch.object(torch.cuda, '_sleep', lambda n: None), \
                mock.patch.object(torch.cuda, 'is_available', lambda: True), \
                mock.patch.dict(os.environ, {'D2L_RERUN': ''}):
            with self.assertRaises(d2l._NoKernelsRecorded):
                d2l._kernel_launches(lambda: None)
            self.assertIsNone(d2l._arm_launches({'f': lambda: None}, ()))
            with quiet(), self.assertRaises(d2l._NoKernelsRecorded):
                d2l.inductor_kernels(lambda: None)

    def test_kernel_kind(self):
        self.assertEqual(d2l._kernel_kind(
            'ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_32x3_tn'),
            'library GEMM, tile 256 x 128')
        self.assertEqual(d2l._kernel_kind(
            'triton_red_fused__to_copy_add_amax_arange_div_exp_full_gt_'
            'masked_fill_maximum_mul_repeat_sub_sum_unsqueeze_view_zeros_'
            'like_0'), 'Inductor reduction: masked_fill, amax, maximum, sub, '
            'exp, sum, div, mul')
        self.assertEqual(d2l._kernel_kind(
            'triton_poi_fused__to_copy_add_div_exp_maximum_mul_sub_view_2'),
            'Inductor pointwise: maximum, sub, exp, div, mul')

    def test_show_slowest(self):
        rows = [dict(name=n, M=m, N=151936 if n == 'head' else 4096,
                     K=4096, tflops=109.0 if (n, m) == ('head', 4096)
                     else 166.0) for n in ('QKV', 'head')
                for m in (2048, 4096, 8192)]
        with mock.patch.object(torch.cuda, 'is_available', lambda: False):
            lines = self.out(d2l._show_slowest, kernels_frame(rows))
        self.assertEqual(lines, ['slowest point: head at M = 4,096, 109 '
                                 'TFLOP/s (median of the sweep 166)'])

    def test_launches_skipped_in_regeneration(self):
        with mock.patch.dict(os.environ, {'D2L_RERUN': 'box-microbench'}):
            self.assertIsNone(d2l._arm_launches({'f': print}, ()))
            self.assertIsNone(d2l._wave_tiles(64, 64, [64]))


class TestCommittedClaims(unittest.TestCase):
    """Claims C1 and C2 on the committed box result: the DRAM read within
    15% of the datasheet, the L2 read at least twice the DRAM read."""

    def test_c1_c2(self):
        if not d2l.hw_registered('box-microbench'):
            self.skipTest('box-microbench is not registered yet')
        logs = d2l.hw_box_log('box-microbench')
        t = d2l._hw_sheet_table(committed_specs().loc['RTX 4090'], logs)
        self.assertGreaterEqual(t.loc['DRAM read (GB/s)', 'fraction'], 0.85)
        self.assertGreaterEqual(t.loc['L2 read (GB/s)', 'measured'],
                                2 * t.loc['DRAM read (GB/s)', 'measured'])


    def test_k1_k2(self):
        """K1: the best Triton configuration within 2x of SDPA's flash
        kernel at both shapes; K2: at least 10x the eager chunked
        attention (reference shape)."""
        if not d2l.hw_registered('box-microbench'):
            self.skipTest('box-microbench is not registered yet')
        flash = d2l.hw_box_log('box-microbench', 'flash-sweep')
        ways = d2l.hw_box_log('box-microbench', 'three-ways').set_index('arm')
        for shape, part in flash.groupby('shape'):
            best = part[part.block_q > 0].tflops.max()
            sdpa = float(part[part.arm == 'SDPA'].tflops.iloc[0])
            self.assertGreaterEqual(best / sdpa, 0.5, shape)
            if shape == 'reference':
                self.assertGreaterEqual(best / ways.loc['chunked', 'tflops'],
                                        10)


    def test_k5_sawtooth(self):
        """K5: the fixed-tile arm's saw-tooth period is one wave of tiles,
        n_SM T_m T_n / M columns, within one step of N."""
        if not d2l.hw_registered('box-microbench'):
            self.skipTest('box-microbench is not registered yet')
        wave = d2l.hw_box_log('box-microbench', 'wave')
        r = wave[wave.arm != 'cuBLAS']
        tm, tn = d2l._WAVE_TILE
        wave_cols = wave.attrs['n_sm'] * tm * tn / wave.attrs['M']
        period = d2l._sawtooth_period(r.N, r.tflops)
        self.assertIsNotNone(period)
        self.assertLessEqual(abs(period - wave_cols), r.N.diff().iloc[1])

if __name__ == '__main__':
    unittest.main()
