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
Section 22.4: the tier table's contract and display, the fio, nvbandwidth
and gdsio parsers on raw outputs (tools/lm_data/hw/samples), the topology
parser and the four-card groups, the fio command, fit_h2d, the offload
thresholds and kv_load_vs_recompute, the measurements, displays and the
shard rule on synthetic frames of known parameters, the offload crossover,
_CPUAdam against torch's AdamW, the scratch-file disk check.
Section 22.5: the link table (PCIe rows computed), the topology parser on
the box's matrices and the two published samples (ANSI escapes, a
header whose names run together), topo_graph and the displays, the
p2pBandwidthLatencyTest and nccl-tests parsers on raw outputs, the busbw
factors, coll_time, tree_link_loads exact for k = 4, 8, 16 and the D2
ratio, the bridge groups and placements, fit_collectives on synthetic
sweeps of known parameters, the pending campaign path, alpha_beta.json's
records (links, transports, no NaN) and box_alpha_beta (a tie needs a
path), the two-regime fits and their intervals over processes, the D2
pattern flows and times, the medians per relation class, the scrubber.
Section 22.7: the closed forms (D1 to D3) and the simulator (ring exact, the
tree an upper bound within 2% far above the crossover), NCCL's double
binary tree, the INFO digest on a sample, the forced-pair fits and the
measured and predicted crossover, the tuner check in noise units, the
opening's symptoms, alpha_beta.json's records, the hierarchical display.
Section 22.8: energy_per_token (trapezoid), the clock-event fractions,
dvfs_fit (the plan's synthetic numbers exactly, a noisy sweep recovered),
plot_dvfs's lines, hw_clock_sweep (PermissionError without root, -rgc
restored after a failing workload), cap_effect pending and its table, the
decode knee slopes, the cost functions (6 N D / (MFU P 3600) exactly, the
serving cost), owned_vs_rented's identity, owned_inputs from the price table,
u_star_interval (deterministic, brackets u*), the tornado order (rent
first, E6), the units check (and a unit mismatch raising), the price
table's contract (columns, kinds, dates, chips in chip_specs, a rent for
every chip the calculators read), the chapter-21 MFU conversion, the box's
Qwen3-8B depth extrapolation, the training-MFU and goodput fallbacks and
their campaign paths, the frontier and prefill readers, and the widths of
every display.
Fix pass F2: the claim lines computed from the committed data: K3's spill
summary (synthetic and committed), M1's plateau label and M2's per-bridge
line, S1's staged-copy fraction, S2's NODE/PHB sentence, C1's simulator
sentence (from the committed table), C2's fit residuals near the crossover.
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
import sys
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
            'hw-chip-specs', 'hw-links', 'hw-tiers', 'hw-systems',
            'hw-prices', 'hw-inferencex-snapshot'})
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

    def test_static_names(self):
        """Every static table is named hw-...; the DATA_HUB keys are
        unchanged by the rename (fix pass F1, library-review nit 5)."""
        self.assertEqual({n: k for n, (k, _) in d2l.HW_STATIC.items()}, {
            'hw-chip-specs': 'lm-hw-chip-specs-v1',
            'hw-links': 'lm-hw-links-v2', 'hw-tiers': 'lm-hw-tiers-v1',
            'hw-systems': 'lm-hw-systems-v1',
            'hw-prices': 'lm-hw-prices-v1',
            'hw-inferencex-snapshot': 'lm-hw-inferencex-snapshot-v1'})
        for name in d2l.HW_STATIC:
            self.assertTrue(name.startswith('hw-'), name)

    def test_freshness_scan(self):
        """The freshness scan (make audit-outputs) attributes a call of
        d2l.hw_prices to the price table's key and a bare page literal
        'prices', which another chapter may use, to no shard at all."""
        shards = os.path.join(REPO, 'd2l', '_blocks', 'pytorch')
        if not os.path.isfile(os.path.join(shards, 'hw_prices.py')):
            self.skipTest('no library shards (run make lib)')
        sys.path.insert(0, os.path.join(REPO, 'tools'))
        import scan_d2l_usage as scan
        graph = scan.shard_graph(shards)

        def deps(code):
            with tempfile.TemporaryDirectory() as tmp:
                page = os.path.join(tmp, 'page.md')
                with open(page, 'w', encoding='utf-8') as f:
                    f.write('```{.python .input #p-1}\n%%tab pytorch\n'
                            + code + '\n```\n')
                refs = scan.d2l_refs_per_framework(scan.Path(page))
            return scan.closure(refs['pytorch'], graph)
        prices = deps('d2l.hw_prices()')
        self.assertIn('HW_STATIC.hw-prices', prices)
        self.assertIn('DATA_HUB.lm-hw-prices-v1', prices)
        for word in ('prices', 'tiers', 'links', 'systems', 'chip-specs',
                     'inferencex-snapshot'):
            self.assertEqual(deps(f"x = {{'{word}': 1}}['{word}']"), set(),
                             word)

    def test_pages_call_no_private_helper(self):
        """No top-level statement of a chapter page outside a #@save block
        calls a private d2l._ helper (fix pass F1, library-review nit 6);
        the lib files and #@save blocks may. Saved blocks are found with
        build_lib's own parser, so a statement placed after a #@save
        definition in the same cell is still checked."""
        sys.path.insert(0, os.path.join(REPO, 'tools'))
        from build_lib import save_blocks
        for path in sorted(glob.glob(os.path.join(CHAPTER, '*.md'))):
            if path.endswith('-lib.md'):
                continue
            text = open(path, encoding='utf-8').read()
            for cid, code in re.findall(
                    r'```\{\.python[^\n]*#([\w-]+)[^\n]*\n(.*?)```',
                    text, re.S):
                # Cell magics and shell lines are not Python; blank them so
                # line numbers still match.
                lines = ['' if ln.lstrip().startswith(('%', '!')) else ln
                         for ln in code.split('\n')]
                saved = save_blocks('\n'.join(lines))
                where = f'{os.path.basename(path)}: {cid}'
                tree = ast.parse('\n'.join(lines), filename=where)
                for node in tree.body:  # each top-level statement
                    first = min([node.lineno] + [
                        dec.lineno for dec in
                        getattr(node, 'decorator_list', [])])
                    seg = re.sub(r'[ \t]*#\s*@save', '', '\n'.join(
                        lines[first - 1:node.end_lineno])).rstrip()
                    if not any(seg in block for block in saved):
                        self.assertNotRegex(seg, r'\bd2l\._',
                                            f'{where}, line {first}')


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
        key, kind = d2l.HW_STATIC['hw-chip-specs']
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
    def test_arch_dims(self):
        """The public reader equals chapter 21's _arch_fields for both
        Transformers configurations and the reference Arch."""
        arch = d2l.r2_config('r2').arch
        self.assertEqual(d2l.arch_dims(arch), d2l._arch_fields(arch))
        self.assertEqual(d2l.arch_dims(arch)[-1], arch.tie)
        for model in ('Qwen/Qwen3-0.6B', 'Qwen/Qwen3-8B'):
            cfg = cached_config(self, model)
            self.assertEqual(d2l.arch_dims(cfg), d2l._arch_fields(cfg))
        # Pinned values: a change in what chapter 21 reads fails here.
        self.assertEqual(
            d2l.arch_dims(cached_config(self, 'Qwen/Qwen3-0.6B')),
            (28, 1024, 16, 8, 128, 3072, 151936, True))

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

    def test_spill_summary(self):
        """K3's line: the fastest spill-free block_q 128 configuration by
        throughput (not the largest tile), light and heavy spills as % of
        the shape's best, 'none' where a shape has no such configuration."""
        t = pd.DataFrame({
            'TF/s ref': [100.0, 90.0, 99.4, 70.0, 80.0],
            'spills ref': [0, 0, 0, 5, 150],
            'TF/s 8B': [50.0, 60.0, 30.0, 45.0, 20.0],
            'spills 8B': [0, 40, 12, 0, 0]},
            index=['64x64 w4', '128x32 w4', '128x64 w8', '128x128 w8',
                   '128x128 w4'])
        line = d2l._spill_summary(t)
        self.assertEqual(line, 'spill-free block_q 128: ref 128x64 w8 99%, '
                         '8B 128x128 w8 75%; light 70%, none; heavy 80%, none')
        self.assertLessEqual(len(line), 100)

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


# ---- 22.3 (workload-hardware): serving and training models, fits, displays


def cached_config(test, model_id):
    """The pinned configuration from the local Hub cache, or skip."""
    with mock.patch.dict(os.environ, {'HF_HUB_OFFLINE': '1'}):
        try:
            return d2l._hw_model_config(model_id)
        except (OSError, KeyError):
            test.skipTest(f'{model_id} configuration not cached')


def lines_of(test, fn, *args, **kwargs):
    """What fn prints, each line at most 79 characters wide."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args, **kwargs)
    lines = buf.getvalue().splitlines()
    for line in lines:
        test.assertLessEqual(len(line), 79, line)
    return lines, out


class TestServingModel(unittest.TestCase):
    def setUp(self):
        self.chip = committed_specs().loc['RTX 4090']
        self.a100 = committed_specs().loc['A100 PCIe 40 GB']
        self.c8 = cached_config(self, 'Qwen/Qwen3-8B')
        self.c06 = cached_config(self, 'Qwen/Qwen3-0.6B')
        self.m = d2l.ServingModel(self.c8, self.chip)

    def test_n_params_against_meta_models(self):
        import transformers
        for model_id in ('Qwen/Qwen3-0.6B', 'Qwen/Qwen3-30B-A3B'):
            cfg = cached_config(self, model_id)
            with quiet(), torch.device('meta'):
                model = transformers.AutoModelForCausalLM.from_config(cfg)
            n = sum(p.numel() for p in model.parameters())
            self.assertEqual(d2l.ServingModel(cfg, self.chip).n_params()[0],
                             n, model_id)
        moe = d2l.ServingModel(cached_config(self, 'Qwen/Qwen3-30B-A3B'),
                               self.chip)
        self.assertEqual(moe.experts(), (128, 8, 768, 48))
        self.assertAlmostEqual(moe.n_params()[1] / 1e9, 3.35, places=2)

    def test_bytes_against_config_memory(self):
        mem = d2l.config_memory(self.c8, 2048, 1)
        self.assertEqual(self.m.weight_bytes(), mem['weights'])
        self.assertEqual(self.m.kv_bytes_per_token() * 2048, mem['kv_cache'])
        self.assertEqual(self.m.kv_bytes_per_token(), 147456)
        self.assertEqual(d2l.ServingModel(self.c06, self.chip)
                         .kv_bytes_per_token(), 114688)
        self.assertEqual(d2l.ServingModel(self.c8, self.chip,
                                          kv_bytes_override=1000)
                         .kv_bytes_per_token(), 1000)

    def test_max_batch(self):
        self.assertEqual([self.m.max_batch(L) for L in (512, 2048, 8192)],
                         [69, 17, 4])
        m06 = d2l.ServingModel(self.c06, self.chip)
        self.assertEqual([m06.max_batch(L) for L in (512, 2048)], [347, 86])
        self.assertEqual(self.m.n_chips, 1)

    def test_decode_limits(self):
        m = self.m
        t_c, t_m, t_l = m.decode_terms(1, 0)
        self.assertEqual(t_l, 0.0)
        self.assertAlmostEqual(t_m, m.read_bytes(1) / m.beta)
        self.assertAlmostEqual(m.decode_step(10**6, 0), 2 * m.n_matmul()
                               * 10**6 / m.P)  # compute-bound
        for L in (0, 512):  # at b*, compute time equals memory time
            b = m.b_star(L)
            t_c, t_m, _ = m.decode_terms(b, L)
            self.assertAlmostEqual(t_c / t_m, 1, places=9)
        self.assertAlmostEqual(m.b_star(0), m.P / m.beta, delta=0.1)
        self.assertTrue(math.isinf(m.b_star(m.L_crit() + 1)))
        self.assertTrue(math.isfinite(m.b_star(m.L_crit() - 1)))
        self.assertAlmostEqual(m.L_crit(), 626.4, places=1)
        a100 = d2l.ServingModel(self.c8, self.a100)
        self.assertAlmostEqual(a100.L_crit(), 511.6, places=1)
        for L in (512, 2048, 8192):  # W3
            self.assertLess(m.max_batch(L), m.b_star(L))

    def test_prefill(self):
        m = self.m
        head = 151936 * 4096
        self.assertEqual(m.n_body(), m.n_matmul() - head)
        n = m.prefill_crossover()
        self.assertAlmostEqual(n, m.n_body() / (36 * 32 * 128), places=6)
        self.assertAlmostEqual(n, 47106, delta=1)
        attn = 2 * 36 * 32 * 128 * n**2
        self.assertAlmostEqual(attn / (2 * m.n_body() * n), 1, places=9)
        self.assertAlmostEqual(m.prefill_time(4096) / m.prefill_time(2048), 2,
                               delta=0.1)
        flops = 2 * m.n_body() * 1024 + 2 * head + 2 * 36 * 32 * 128 * 1024**2
        self.assertAlmostEqual(m.prefill_time(1024), flops / m.P)

    def test_pending_coverage_is_nan(self):
        """A pending MoE coverage gives NaN bytes and step time, never the
        compute term alone."""
        moe = d2l.ServingModel(cached_config(self, 'Qwen/Qwen3-30B-A3B'),
                               self.chip)
        nan = lambda b: math.nan
        self.assertTrue(math.isnan(moe.read_bytes(8, nan)))
        self.assertTrue(math.isnan(moe.decode_step(1, 2048, nan)))
        self.assertTrue(math.isnan(moe.decode_step(64, 2048, nan)))
        self.assertTrue(math.isfinite(moe.decode_step(64, 2048)))

    def test_frontier_line(self):
        m = self.m
        f = m.frontier(L=(512, 2048), batches=[1, 2, 4, 8, 16, 32, 64, 128])
        bw = f[f.regime == 'bw']
        W, k = m.read_bytes(1), m.kv_bytes_per_token()
        line = (m.beta - W * bw.user_tps) / (bw.L * k)
        np.testing.assert_allclose(bw.gpu_tps, line, rtol=1e-9)
        self.assertEqual(set(f[f.b > 69].regime), {'infeasible'})
        self.assertAlmostEqual(m.gpu_s_per_mtok(4, 2048),
                               1e6 * m.decode_step(4, 2048) / 4)

    def test_moe_coverage(self):
        self.assertAlmostEqual(d2l.moe_expert_coverage(128, 8, 1), 8)
        self.assertAlmostEqual(d2l.moe_expert_coverage(128, 8, 10**4), 128)
        self.assertAlmostEqual(d2l.moe_expert_coverage(128, 8, 11) / 128, 0.5,
                               delta=0.02)
        rng = np.random.default_rng(0)
        for b in (1, 8, 64):
            sim = np.mean([len(set().union(*(rng.choice(128, 8, replace=False)
                                              for _ in range(b))))
                           for _ in range(2000)])
            self.assertLess(abs(sim / d2l.moe_expert_coverage(128, 8, b) - 1),
                            0.01, b)
        moe = d2l.ServingModel(cached_config(self, 'Qwen/Qwen3-30B-A3B'),
                               self.chip)
        self.assertLess(moe.read_bytes(1), moe.read_bytes(64))
        self.assertAlmostEqual(moe.read_bytes(8, 1.0), moe.read_bytes(10**6))
        self.assertAlmostEqual(moe.read_bytes(8, lambda b: 1.0),
                               moe.read_bytes(8, 1.0))
        self.assertEqual(self.m.read_bytes(1), self.m.read_bytes(64, 0.1))

    def test_rates(self):
        specs = committed_specs()
        P, beta = d2l.hw_rates(specs.loc['Groq LPU'])
        self.assertTrue(math.isinf(P))
        P, beta = d2l.hw_rates(self.chip, dtype_bytes=1)
        self.assertAlmostEqual(P, self.chip.p8_tflops * 1e12)
        P, beta = d2l.hw_rates(self.chip, dict(P_eff_bf16=1e14, beta_eff=9e11,
                                               eta_compute=0.5))
        self.assertEqual((P, beta), (5e13, 9e11))
        m = d2l.ServingModel(self.c8, self.chip, n_chips=2)
        self.assertEqual(m.P, 2 * self.m.P)
        self.assertEqual(m.max_batch(2048) > self.m.max_batch(2048), True)

    def test_launch_and_eta(self):
        calib = dict(P_eff_bf16=1.6e14, beta_eff=9.2e11, alpha_k=1.5e-6,
                     kernels_per_step=2000)
        m = d2l.ServingModel(self.c8, self.chip, calib)
        self.assertAlmostEqual(m.launch(), 3e-3)
        self.assertAlmostEqual(m.decode_step(1, 0) - m.launch(),
                               m.read_bytes(1) / 9.2e11)
        slow = d2l.ServingModel(self.c8, self.chip, dict(calib, eta_kv=0.5))
        self.assertAlmostEqual(slow.L_crit() * 2, m.L_crit())


class TestArchConfig(unittest.TestCase):
    def test_reference_arch_accepted(self):
        """Chapter 21's Arch reads like its Transformers configuration
        (lib review finding 4: it raised AttributeError)."""
        chip = committed_specs().loc['RTX 4090']
        arch = d2l.r2_config('r2').arch
        m = d2l.ServingModel(arch, chip)
        L, d, H, H_kv, d_h, f, V, _ = d2l._arch_fields(arch)
        norms = 2 * d_h + 2 * d
        attn = 2 * d * d_h * (H + H_kv)
        self.assertEqual(m.n_params()[0],
                         V * d + d + L * (attn + norms + 3 * d * f))
        self.assertEqual(m.kv_bytes_per_token(), 2 * L * H_kv * d_h * 2)
        t = d2l.TrainStepModel(arch, chip)
        self.assertEqual(t.N, m.n_params()[0])
        self.assertEqual(t.flops_per_token(4096), d2l.flops_per_token(
            arch, T=4096, causal_half=True) + 2 * V * d)
        self.assertIs(d2l._hw_cfg(m.cfg), m.cfg)


class TestTrainStepModel(unittest.TestCase):
    def test_state_and_mfu(self):
        chip = committed_specs().loc['RTX 4090']
        cfg = cached_config(self, 'Qwen/Qwen3-8B')
        t = d2l.TrainStepModel(cfg, chip)
        self.assertEqual(t.state, {'bf16_pure': 8, 'mixed16': 16})
        self.assertEqual(t.traffic['bf16_pure'], 14)
        self.assertAlmostEqual(t.N * 8 / 1e9, 65.5, places=1)
        Vd = cfg.vocab_size * cfg.hidden_size  # untied: lookup out, +2 V d
        self.assertEqual(t.flops_per_token(2048), d2l.flops_per_token(
            cfg, T=2048, causal_half=True) - 4 * Vd)
        tied = cached_config(self, 'Qwen/Qwen3-0.6B')
        Vd = tied.vocab_size * tied.hidden_size
        self.assertEqual(d2l.TrainStepModel(tied, chip).flops_per_token(512),
                         d2l.flops_per_token(tied, T=512, causal_half=True)
                         + 2 * Vd)
        self.assertAlmostEqual(t.mfu(1000, 2048), d2l.mfu(
            1000, cfg, chip.p16_tflops * 1e12, T=2048, causal_half=True))
        m, logits = t.serving, cfg.vocab_size * 2048
        self.assertAlmostEqual(t.step_time(1, 2048), t.flops_per_token(2048)
                               * 2048 / m.P + (14 * t.N + 40 * logits)
                               / m.beta)
        mixed = d2l.TrainStepModel(cfg, chip, recipe='mixed16')
        self.assertGreater(mixed.step_time(1, 2048), t.step_time(1, 2048))
        self.assertGreater(t.step_time(1, 2048, ckpt=True),
                           t.step_time(1, 2048))


def synthetic_grid(chip, calib, kernels=2000):
    """A workload grid whose times follow M3 with known parameters."""
    rows = []
    for b, L in ((1, 512), (4, 512), (16, 512), (1, 2048), (8, 2048),
                 (1, 8192), (2, 8192), (4, 8192), (2, 512)):
        rows.append(dict(model='Qwen/Qwen3-8B', workload='decode',
                         series='grid', b=b, L=L, n_layers=36))
    for n in (256, 1024, 4096, 16384, 512, 2048):
        rows.append(dict(model='Qwen/Qwen3-8B', workload='prefill',
                         series='grid', b=1, L=n, n_layers=36))
    for layers, b in ((1, 1), (2, 1), (3, 1), (1, 2), (2, 2), (3, 2)):
        rows.append(dict(model='Qwen/Qwen3-8B', workload='train',
                         series='grid', b=b, L=2048, n_layers=layers))
    rows.append(dict(model='Qwen/Qwen3-8B', workload='eager', series='eager',
                     b=1, L=2048, n_layers=36))
    grid = pd.DataFrame(rows)
    grid.attrs.update(kernels_per_step={'Qwen/Qwen3-8B': kernels},
                      calib=calib, chip='RTX 4090', cut=[['Qwen/Qwen3-8B',
                                                          32, 2048]])
    truth = dict(eta_compute=0.8, eta_kv=0.7, kappa=1.5)
    ms = []
    for row in grid.itertuples():
        c = d2l._ladder_calibs(calib, grid, row.model, row.n_layers,
                               truth)['M3']
        ms.append(d2l._predict(row, chip, c) * 1e3)
    grid['ms'], grid['mad'] = ms, 0.01
    grid['floor_ms'] = np.where(grid.workload.isin(['prefill', 'train']),
                                1.0, np.nan)
    grid.loc[grid.workload == 'prefill', 'floor_ms'] = 50.0
    grid.loc[(grid.workload == 'prefill') & (grid.L == 256), 'ms'] = 50.0
    graph = grid[(grid.workload == 'decode') & (grid.b == 1)
                 & (grid.L == 2048)].ms.iloc[0]
    grid.loc[grid.workload == 'eager', 'ms'] = 10 * graph
    return grid, truth


class TestWorkloadFits(unittest.TestCase):
    def setUp(self):
        cached_config(self, 'Qwen/Qwen3-8B')
        self.chip = committed_specs().loc['RTX 4090']
        self.calib = dict(P_eff_bf16=1.6e14, beta_eff=9.2e11, alpha_k=1.4e-6,
                          gpu='RTX 4090')

    def test_fit_recovers_parameters(self):
        grid, truth = synthetic_grid(self.chip, self.calib)
        lines, fit = lines_of(self, d2l.fit_workload_models, grid, self.calib,
                              self.chip)
        for k, v in truth.items():
            self.assertAlmostEqual(fit['params'][k], v, places=3, msg=k)
        held = fit['errors'][fit['errors'].held_out]
        self.assertLess(held.M3.abs().max(), 1e-6)
        self.assertGreater(held.M0.abs().max(), 0.15)
        self.assertTrue(any('M3 fitted on 7 of 21' in l for l in lines))
        self.assertIn('held-out points within 15%, M0 / M3: '
                      'Qwen3-8B 0 / 14 of 14', lines)
        self.assertEqual(lines[-3], 'Qwen3-8B training, 1-3 layers, a '
                         'straight line to within 0.0 ms:')
        self.assertRegex(lines[-2], r'^  1 x 2,048 tokens: [\d.]+ ms per '
                         r'layer \+ \d+ ms; 36 layers [\d.]+ s \(M3 [\d.]+ s\)$')
        self.assertLessEqual(len(lines), 15)
        self.assertFalse(fit['errors'].floor_set[fit['errors'].held_out].any())
        self.assertTrue(fit['errors'].floor_set.any())  # prefill at 256

    def test_floor_set_counted_apart(self):
        grid, _ = synthetic_grid(self.chip, self.calib)
        grid.loc[(grid.workload == 'prefill') & (grid.L == 512),
                 ['ms', 'floor_ms']] = 400.0  # a held-out floor-set point
        lines, fit = lines_of(self, d2l.fit_workload_models, grid,
                              self.calib, self.chip)
        self.assertIn('Qwen3-8B 0 / 13 of 13 (and 1 set by the launch '
                      'floor, 1 within)', ' '.join(' '.join(lines).split()))

    def test_pending_and_campaign(self):
        lines, out = lines_of(self, d2l.fit_workload_models,
                              d2l.hw_campaign_log('a100-workloads'),
                              chip=committed_specs().loc['A100 PCIe 40 GB'])
        self.assertIsNone(out)
        self.assertEqual(len(lines), 1)
        grid, _ = synthetic_grid(self.chip, self.calib)
        _, fit = lines_of(self, d2l.fit_workload_models, {'grid-t2': grid})
        self.assertIn('M3', fit['errors'])

    def test_show_grid(self):
        grid, _ = synthetic_grid(self.chip, self.calib)
        lines, _ = lines_of(self, d2l._show_grid, grid)
        self.assertIn('Qwen3-8B 9 decode, 6 prefill, 6 training', lines[0])
        self.assertIn('(32, 2,048)', lines[1])
        self.assertIn('10x the graph step', lines[2])
        self.assertIn('Qwen3-8B prefill 50-50 ms, Qwen3-8B train 1-1 ms',
                      ' '.join(' '.join(lines[3:]).split()))
        lines, _ = lines_of(self, d2l._show_grid, d2l._hw_pending(
            'box-workloads'))
        self.assertEqual(lines, ['box-workloads pending'])

    def test_live_rows(self):
        grid, _ = synthetic_grid(self.chip, self.calib)
        picks = grid.loc[d2l._live_rows(grid)]
        self.assertEqual(list(picks.workload),
                         ['decode', 'prefill', 'train', 'decode'])
        self.assertEqual(int(picks.iloc[1].L), 2048)
        small = grid.copy()  # a second model, and peaks on a 16 GB card
        small['model'] = 'Qwen/Qwen3-0.6B'
        both = pd.concat([grid.assign(peak_gib=19.0),
                          small.assign(peak_gib=9.0)], ignore_index=True)
        picks = both.loc[d2l._live_rows(both, card_gib=14.6)]
        self.assertEqual(set(picks.model), {'Qwen/Qwen3-0.6B'})
        self.assertEqual(list(picks.workload),
                         ['decode', 'prefill', 'train', 'decode'])
        self.assertEqual(len(d2l._live_rows(both, card_gib=23.5)), 4)
        self.assertEqual(d2l._live_rows(both, card_gib=8.0), [])

    def test_show_control(self):
        r = pd.DataFrame(dict(arm=['random', 'real', 'random again'],
                              ms=[19.10, 19.14, 19.12], mad=[0.002, 0.003,
                                                             0.002],
                              kernels=[2189] * 3, peak_gib=[16.0] * 3))
        lines, _ = lines_of(self, d2l._show_control, r, 'Qwen/Qwen3-8B', 1,
                            2048)
        self.assertEqual(lines[0], 'Qwen3-8B, b = 1, L = 2,048: random '
                         '19.10 ms, real 19.14, random again 19.12')
        self.assertEqual(lines[1], 'real minus random +0.030 ms (+0.2%), '
                         'noise 0.020 ms, the same kernels (2,189)')

    def test_show_calib(self):
        lines, _ = lines_of(self, d2l.show_calib, self.calib, self.chip)
        self.assertIn('160 TFLOP/s (97% of', lines[0])
        self.assertIn('920 GB/s (91% of', lines[1])
        self.assertIn('1.4 us', lines[2])
        lines, _ = lines_of(self, d2l.show_calib, None, self.chip)
        self.assertEqual(len(lines), 1)

    def test_hw_calibrate_from_log(self):
        alpha, beta = 1.1e-6, 9.2e11
        s = 2.0 ** np.arange(18, 31, 2)
        t = alpha + 2 * s / beta
        ladder = pd.DataFrame(dict(bytes=s, kind='copy', ms=t * 1e3,
                                   gbs=2 * s / t / 1e9))
        ladder.attrs['gpu'] = 'RTX 4090'
        log = {'ladder': ladder,
               'shape-sweep': pd.DataFrame(dict(tflops=[150.0, 171.0])),
               'dtypes': pd.DataFrame(dict(format=['bf16', 'fp8'],
                                           tflops=[160.0, 320.0]))}
        c = d2l.hw_calibrate(log=log)
        self.assertAlmostEqual(c['P_eff_bf16'], 171e12)
        self.assertAlmostEqual(c['alpha_k'] / alpha, 1, places=6)
        self.assertLess(abs(c['beta_eff'] / beta - 1), 0.01)
        self.assertEqual(c['gpu'], 'RTX 4090')
        if d2l.hw_registered('box-p2p'):  # the all-reduce fit of 22.5
            self.assertGreater(c['alpha_ar'], 0)
        else:
            self.assertIsNone(c['alpha_ar'])
        self.assertIsNone(d2l.hw_calibrate(log=pd.DataFrame()))

    def test_plot_frontier(self):
        import matplotlib
        matplotlib.use('Agg')
        cfg = cached_config(self, 'Qwen/Qwen3-8B')
        m = d2l.ServingModel(cfg, self.chip, self.calib)
        grid, _ = synthetic_grid(self.chip, self.calib)
        front = m.frontier(L=(512, 2048, 8192), batches=2**np.arange(8))
        lines, pts = lines_of(self, d2l.plot_frontier, front, m, grid)
        self.assertIn('compute roof', lines[0])
        self.assertTrue(lines[1].startswith('9 measured decode points: '),
                        lines[1])
        self.assertTrue(lines[2].startswith('L = 512: b = 1 to 2 adds '),
                        lines[2])
        fits = dict(errors=pd.DataFrame(dict(held_out=np.arange(len(grid))
                                             % 3 > 0), index=grid.index))
        lines, _ = lines_of(self, d2l.plot_frontier, front, m, grid, fits)
        self.assertTrue(lines[1].startswith('6 held-out of 9 measured'),
                        lines[1])
        self.assertIn(69, set(front[front.L == 512].b))  # the capacity cut
        for L, p in pts.groupby('L'):
            self.assertTrue(p.gpu_tps.is_monotonic_increasing)
            self.assertTrue(p.user_tps.is_monotonic_decreasing)
        lines, pts = lines_of(self, d2l.plot_frontier, front, m,
                              d2l._hw_pending('box-workloads'))
        self.assertEqual((len(lines), len(pts)), (1, 0))
        d2l.plt.close('all')


class TestMoECoverage(unittest.TestCase):
    def test_coverage_table(self):
        rng = np.random.default_rng(1)
        E, K = 32, 4
        uniform = np.sort(np.stack([np.stack([rng.choice(E, K, replace=False)
                                             for _ in range(6)])
                                   for _ in range(400)]), -1).astype(np.uint8)
        skewed = np.sort(np.stack([np.stack([rng.choice(8, K, replace=False)
                                            for _ in range(6)])
                                  for _ in range(400)]), -1).astype(np.uint8)
        t = d2l._coverage_table({'uniform': uniform, 'skewed': skewed}, E, K,
                                (1, 4, 16, 64), 100, 0)
        u = t[t.source == 'uniform'].set_index('b')
        self.assertTrue(np.allclose(u['mean'] / u.uniform, 1, atol=0.03))
        s = t[t.source == 'skewed'].set_index('b')
        self.assertLessEqual(s['mean'].max(), 8)
        self.assertTrue((s.lo <= s['mean']).all())
        self.assertTrue((s['mean'] <= s.hi).all())
        cov = d2l.MoECoverage(t, E, K, 6, {})
        self.assertAlmostEqual(cov.coverage(4, 'uniform'),
                               u['mean'][4] / E)
        self.assertTrue(math.isnan(d2l._moe_pending().coverage(4)))

    def test_plot_coverage(self):
        import matplotlib
        matplotlib.use('Agg')
        t = pd.DataFrame(dict(source=['prose'] * 3 + ['code'] * 3,
                              b=[1, 8, 64] * 2,
                              mean=[8, 40, 100, 8, 30, 80],
                              lo=[8, 36, 96, 8, 27, 76],
                              hi=[8, 44, 104, 8, 33, 84]))
        t['uniform'] = [d2l.moe_expert_coverage(128, 8, b) for b in t.b]
        cov = d2l.MoECoverage(t, 128, 8, 48, {})
        lines, _ = lines_of(self, d2l.plot_coverage, cov,
                            d2l.moe_expert_coverage)
        self.assertEqual(len(lines), 2)
        self.assertRegex(lines[1], r'b = 64: prose 0\.\d\d, code 0\.\d\d')
        lines, _ = lines_of(self, d2l.plot_coverage, d2l._moe_pending(),
                            d2l.moe_expert_coverage)
        self.assertEqual(lines, ['box-moe pending: the uniform formula only'])
        d2l.plt.close('all')

    def test_round_trip(self):
        t = pd.DataFrame(dict(source=['prose'], b=[1], mean=[8.0], lo=[8.0],
                              hi=[8.0], uniform=[8.0]))
        cov = d2l.MoECoverage(t, 128, 8, 48, {'prose': np.zeros(
            (2, 48, 8), np.uint8)})
        back = d2l._hw_decode(json.loads(json.dumps(d2l._hw_encode(cov))))
        self.assertIsInstance(back, d2l.MoECoverage)
        self.assertEqual(back.routes['prose'].shape, (2, 48, 8))


class TestLiveLine(unittest.TestCase):
    def test_long_live_line_wraps(self):
        """A live line with a long "differs:" list wraps at 80 columns
        (make audit-outputs flagged 111 and 113 columns)."""
        gpu = 'NVIDIA GeForce RTX 4090'
        meta = {'env': {'gpu': gpu}}
        text = ('2 points, largest difference 20%; differs: pinned h2d '
                '256 MiB 26 vs 21 GB/s, pinned d2h 256 MiB 24 vs 20 GB/s')
        out = io.StringIO()
        with mock.patch.object(torch.cuda, 'is_available', lambda: True), \
                mock.patch.object(torch.cuda, 'get_device_name',
                                  lambda i=0: gpu), \
                contextlib.redirect_stdout(out):
            d2l._hw_live(lambda r, same: text, None, meta, 'host link')
        lines = out.getvalue().splitlines()
        self.assertGreater(len(lines), 1)
        self.assertTrue(all(len(l) <= 80 for l in lines))
        self.assertTrue(lines[1].startswith('  '))


class TestInferenceX(unittest.TestCase):
    FIXTURE = ('model,hw,framework,precision,isl,osl,conc,tp,tps_per_gpu,'
               'tps_per_user,commit,retrieved\n'
               'llama70b,H100,vllm,fp8,1024,1024,64,8,1500,50,abc,2026-11-20\n'
               'llama70b,H200,vllm,fp8,1024,1024,64,8,2200,50,abc,2026-11-20\n'
               'llama70b,B200,vllm,fp8,1024,1024,64,8,4100,50,abc,2026-11-20\n'
               'llama70b,MI355X,vllm,fp8,1024,1024,64,8,3600,50,abc,'
               '2026-11-20\n')

    def test_pending(self):
        snap = d2l.inferencex_snapshot()
        if d2l.hw_registered('hw-inferencex-snapshot'):
            self.skipTest('snapshot registered')
        self.assertFalse(snap)
        lines, t = lines_of(self, d2l.rank_against, snap, committed_specs())
        self.assertTrue(any('not committed yet' in l for l in lines))
        self.assertIn('hops between', lines[-2])
        self.assertNotIn('InferenceX', lines[0])  # no empty columns pending
        self.assertTrue(math.isnan(t['rank']['Groq LPU']))
        self.assertTrue(math.isnan(t['rank']['RTX 4090']))
        self.assertTrue(t.inferencex.isna().all())
        self.assertGreater(t.predicted['B200'], t.predicted['H100 SXM'])
        self.assertEqual(int(t.tp['Groq LPU']), d2l.sram_chips_needed(
            70.55e9, 0.23e9))

    def test_fixture(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'snapshot.csv')
            with open(path, 'w') as f:
                f.write(self.FIXTURE)
            snap = d2l._read_snapshot(path, d2l.HW_SHAPES['dense-70b'])
        self.assertTrue(snap)
        self.assertEqual((snap.dtype_bytes, snap.context), (1, 1536))
        lines, t = lines_of(self, d2l.rank_against, snap, committed_specs())
        self.assertTrue(any(l.startswith('Kendall tau over 4 chips') for l in
                            lines))
        self.assertTrue(any(l.startswith('predicted, not ranked') and
                            'RTX 4090' in l for l in lines))
        self.assertEqual(d2l._kendall_tau([1, 2, 3], [1, 2, 3]), 1)
        self.assertEqual(d2l._kendall_tau([1, 2, 3], [3, 2, 1]), -1)

    def test_duplicate_rows(self):
        """Several rows per chip (TP degrees, interactivity points): the best
        tokens/s per GPU near the median interactivity point, its TP."""
        extra = ('llama70b,H100,vllm,fp8,1024,1024,64,4,1700,48,abc,'
                 '2026-11-20\n'
                 'llama70b,H100,vllm,fp8,1024,1024,256,8,3000,20,abc,'
                 '2026-11-20\n')
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'snapshot.csv')
            with open(path, 'w') as f:
                f.write(self.FIXTURE + extra)
            snap = d2l._read_snapshot(path, d2l.HW_SHAPES['dense-70b'])
        t = snap.table.set_index('chip')
        self.assertTrue(t.index.is_unique)
        self.assertEqual(t.loc['H100 SXM', 'tp'], 4)
        self.assertEqual(t.loc['H100 SXM', 'tps_per_gpu'], 1700)
        lines, r = lines_of(self, d2l.rank_against, snap, committed_specs())
        self.assertEqual(int(r.tp['H100 SXM']), 4)

    def test_tps_at(self):
        m = d2l.ServingModel(d2l.HW_SHAPES['dense-70b'],
                             committed_specs().loc['H100 SXM'], dtype_bytes=1,
                             kv_bytes=1, n_chips=8)
        relaxed, strict = d2l._tps_at(m, 20, 2048), d2l._tps_at(m, 60, 2048)
        self.assertGreater(relaxed, strict)  # less interactivity, more
        b = m.max_batch(2048)  # at 20 tokens/s per user capacity binds
        self.assertAlmostEqual(relaxed, b / m.decode_step(b, 2048) / 8)
        self.assertTrue(math.isnan(d2l._tps_at(m, 10**4, 2048)))


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

    # The spill counts of the committed kernels-spill output (register
    # metadata, deterministic for the pinned Triton; read on a GPU only).
    SPILLS = {
        'reference': (0, 0, 0, 0, 8, 0, 0, 0, 10, 0, 140, 8),
        'Qwen3-8B': (0, 0, 0, 0, 4, 0, 44, 0, 108, 20, 248, 22)}

    def test_k3_spill_summary(self):
        """K3 on the committed flash sweep: the fastest spill-free block_q
        128 configuration is 128 x 32 w4 at the reference shape (146 TF/s)
        and 128 x 32 w8 for Qwen3-8B (129); heavy spills cost about 16%
        (reference) and up to 43% (Qwen3-8B), from the unrounded medians."""
        if not d2l.hw_registered('box-microbench'):
            self.skipTest('box-microbench is not registered yet')
        flash = d2l.hw_box_log('box-microbench', 'flash-sweep')
        rows = {}
        for shape, spills in self.SPILLS.items():
            tri = flash[(flash['shape'] == shape) & (flash.block_q > 0)]
            self.assertEqual(len(tri), len(spills))
            tag = shape.split('-')[-1] if '-' in shape else shape[:3]
            for (_, r), n in zip(tri.iterrows(), spills):
                label = (f'{int(r.block_q)}x{int(r.block_k)} '
                         f'w{int(r.num_warps)}')
                rows.setdefault(label, {}).update(
                    {f'TF/s {tag}': r.tflops, f'spills {tag}': n})
        t = pd.DataFrame.from_dict(rows, orient='index')
        self.assertAlmostEqual(t.loc['128x32 w4', 'TF/s ref'], 146, delta=0.5)
        self.assertAlmostEqual(t.loc['128x32 w8', 'TF/s 8B'], 129, delta=0.5)
        heavy = 1 - t.loc['128x128 w4', 'TF/s ref'] / t['TF/s ref'].max()
        self.assertAlmostEqual(heavy, 0.16, delta=0.01)
        heavy = 1 - t.loc['128x128 w4', 'TF/s 8B'] / t['TF/s 8B'].max()
        self.assertAlmostEqual(heavy, 0.43, delta=0.01)
        self.assertEqual(d2l._spill_summary(t), (
            'spill-free block_q 128: ref 128x32 w4 98%, 8B 128x32 w8 97%; '
            'light 98-100%, 76%; heavy 84%, 57-80%'))

    def test_m1_m2_lines(self):
        """M1's plateau label names a range over copies (not an interval):
        pinned h2d 23 GB/s, range 21-24, 72% of 31.5; M2: about 36 GB/s per
        host bridge, four cards 98% of 2 x 36."""
        if not d2l.hw_registered('box-storage'):
            self.skipTest('box-storage is not registered yet')
        with quiet():
            h2d = d2l.hw_box_log('box-storage', 'h2d')
            agg = d2l.hw_box_log('box-storage', 'concurrent')
        links = d2l.fit_h2d(h2d, alpha_below=2**20, plateau_from=2**28)
        lines, _ = lines_of(self, d2l.plot_h2d, h2d, links)
        d2l.plt.close('all')
        self.assertIn('the median of the copies of 256 MiB and more, and '
                      'their range', lines[0])
        self.assertIn('beta 23 GB/s (range 21-24), 72% of 31.5', lines[1])
        lines, _ = lines_of(self, d2l._show_concurrent, agg)
        self.assertIn('per host bridge: about 36 GB/s h2d (two cards behind '
                      'it); all 4: 98% of 2 x 36', lines)

    def test_s1_fraction(self):
        """S1: P2P refused, GPU-to-GPU copies staged through host memory at
        0.85-0.96 of the pinned host link (PHB pairs to NODE pairs, against
        the pinned h2d plateau; unrounded medians)."""
        if not (d2l.hw_registered('box-p2p')
                and d2l.hw_registered('box-storage')):
            self.skipTest('box-p2p or box-storage is not registered yet')
        with quiet():
            mat = d2l.hw_box_log('box-p2p', 'gpu-matrix')
            h2d = d2l.hw_box_log('box-storage', 'h2d')
        self.assertFalse(np.asarray(mat['peer'])[
            ~np.eye(len(mat['peer']), dtype=bool)].any())
        link = d2l.fit_h2d(h2d, alpha_below=2**20, plateau_from=2**28)
        pinned = link.loc[('pinned', 'h2d'), 'beta'] / 1e9
        by = d2l._by_relation(mat)
        frac = {c: by[c] / pinned for c in ('PHB', 'NODE')}
        self.assertEqual((round(frac['PHB'], 2), round(frac['NODE'], 2)),
                         (0.85, 0.96))

    def test_s2_line(self):
        """S2: NODE pair 3.52 GB/s, PHB pair 2.42, intervals disjoint."""
        if not d2l.hw_registered('box-p2p'):
            self.skipTest('box-p2p is not registered yet')
        with quiet():
            nccl = d2l.hw_box_log('box-p2p', 'nccl')
        ab = d2l.fit_collectives(nccl, alpha_below=2**16, plateau_from=2**22)
        for path, beta in (('NODE', 3.52), ('PHB', 2.42)):
            self.assertAlmostEqual(ab.loc[('all_reduce', 2, path)].beta / 1e9,
                                   beta, delta=0.005)
        self.assertEqual(d2l._pair_paths_line(ab), (
            'two-card all-reduce: across the host bridges (NODE) 3.5 GB/s, '
            'behind one bridge (PHB) 2.4 GB/s (1.5x; intervals disjoint)'))
        lines, _ = lines_of(self, d2l.show_fits, ab)
        self.assertEqual(len(lines), 11)

    def test_c2_residuals(self):
        """C2: measured crossover 83 KiB, predicted 280 KiB; between 16 KiB
        and 1 MiB the fits miss the measured times by far more than the two
        fitted lines differ (Ring/LL +2% to +72%, Ring/Simple -1% to +48%;
        the lines +19% to -3% apart)."""
        if not d2l.hw_registered('box-coll'):
            self.skipTest('box-coll is not registered yet')
        with quiet():
            pairs = d2l.nccl_algo_proto_sweeps('box')
        x = d2l.fit_pairs(pairs).attrs['crossover']
        self.assertEqual(d2l._approx_bytes(x['measured']), '83 KiB')
        self.assertEqual(d2l._approx_bytes(x['predicted']), '280 KiB')
        r = {n: tuple(round(100 * e) for e in v)
             for n, v in x['resid'].items()}
        self.assertEqual(r, {'Ring/LL': (2, 72), 'Ring/Simple': (-1, 48)})
        self.assertEqual(tuple(round(100 * g) for g in x['gap']), (19, -3))
        self.assertGreater(min(max(map(abs, v)) for v in x['resid'].values()),
                           max(map(abs, x['gap'])))

    def test_w1_w4(self):
        """W1: for Qwen3-8B (the plan's question) M3 within 15% on every
        held-out point of each workload, M0 worse; W4: real weights within
        the control's noise, the same kernels in every arm."""
        if not d2l.hw_registered('box-workloads'):
            self.skipTest('box-workloads is not registered yet')
        log = d2l.hw_box_log('box-workloads')
        grid = log['grid-box']
        with quiet():
            fit = d2l.fit_workload_models(grid, None, committed_specs().loc[
                'RTX 4090'])
        held = fit['errors'][fit['errors'].held_out]
        held = held[held.model == 'Qwen/Qwen3-8B']
        for w, e in held.groupby('workload'):
            self.assertLessEqual(e.M3.abs().max(), 0.15, w)
            self.assertGreater(e.M0.abs().max(), e.M3.abs().max(), w)
        t = log['control-Qwen3-8B'].set_index('arm')
        random = (t.ms['random'] + t.ms['random again']) / 2
        self.assertLessEqual(abs(t.ms['real'] - random),
                             d2l._control_noise(t))
        self.assertEqual(t.kernels.nunique(), 1)

    def test_w5(self):
        """W5: measured expert coverage below the uniform formula at b = 64
        for both sources."""
        if not d2l.hw_registered('box-moe'):
            self.skipTest('box-moe is not registered yet')
        cov = d2l.hw_box_log('box-moe', 'coverage-box')
        at64 = cov.table[cov.table.b == 64]
        self.assertEqual(len(at64), 2)
        self.assertTrue((at64['mean'] < at64.uniform).all())

    def test_cross_check_21_6(self):
        """The Qwen3-0.6B decode points at L = 1,024 within 1.5x of 21.6's
        decode table at b = 1, 8, 32, 128 (tokens/s)."""
        if not (d2l.hw_registered('box-workloads')
                and 'lm-decode-rates-v1' in d2l.DATA_HUB):
            self.skipTest('box-workloads or lm-decode-rates-v1 not registered')
        grid = d2l.hw_box_log('box-workloads', 'grid-box')
        ours = grid[(grid.series == 'cross')].set_index('b')
        with quiet():
            rates = d2l.load_artifact('decode-rates')
        rates = pd.DataFrame(rates)
        cols = set(rates.columns)
        b_col = 'batch' if 'batch' in cols else 'b'
        tps = next(c for c in ('tokens_per_s', 'tok_s', 'tps') if c in cols)
        rows = rates[rates.model.astype(str).str.contains('0.6B')]
        for b in (1, 8, 32, 128):
            theirs = float(rows[rows[b_col] == b][tps].iloc[0])
            mine = b * 1e3 / ours.ms[b]
            self.assertLessEqual(max(mine / theirs, theirs / mine), 1.5, b)


# Section 22.4 (memory-tiers.md and memory-tiers-lib.md).
HW_DATA = os.path.join(REPO, 'tools', 'lm_data', 'hw')
SAMPLES = os.path.join(HW_DATA, 'samples')


def committed_tiers():
    return pd.read_csv(os.path.join(HW_DATA, 'tiers.csv'))


def synthetic_storage(seed=0):
    """Frames shaped like the box-storage entries, from known parameters:
    host link alpha 10 us and beta 25 GB/s (pinned), 10 GB/s (pageable);
    drives 6 GB/s at depth, 400K IOPS; object store alpha 70 ms, beta
    60 MB/s."""
    rng = np.random.default_rng(seed)
    rows = []
    for mode, beta in (('pinned', 25e9), ('pinned-2M', 26e9),
                       ('pageable', 10e9)):
        for d in ('h2d', 'd2h'):
            for s in 2**np.arange(12, 31):
                ms = (1e-5 + s / beta) * 1e3 * (1 + 0.01 * rng.standard_normal())
                rows.append(dict(mode=mode, direction=d, bytes=int(s), ms=ms,
                                 mad=0.01 * ms, gbs=s / ms / 1e6))
    h2d = pd.DataFrame(rows)
    h2d.attrs.update(pcie_gen=4, pcie_width=16, pcie_gen_idle=1,
                     pcie_width_idle=16)
    agg = pd.DataFrame([
        dict(group=g, n_cards=k, direction=d, gbs=k * 24.0 * f, mad_gbs=0.1)
        for g, k, f in (('one card', 1, 1.0),
                        ('two cards, one host bridge', 2, 0.95),
                        ('two cards, two host bridges', 2, 1.0),
                        ('all 4 cards', 4, 0.9))
        for d in ('h2d', 'd2h')])
    agg.attrs['nvbandwidth'] = {'host_to_device': [25.4, 25.7, 25.7, 26.0]}
    dram = dict(read=110.0, copy=80.0, triad=90.0, read_mad=0.01,
                copy_mad=0.01, triad_mad=0.01, read_threads=16,
                copy_threads=16, triad_threads=16, nbytes=2**31)
    frows = []
    for rw in ('read', 'randread'):
        for qd, jobs in ((1, 1), (64, 4)):
            for bs in 2**np.arange(12, 23, 2):
                t = 8e-5 / (qd * jobs) + bs / 6e9  # seconds per request
                bw = min(bs / t, 6e9)
                frows.append(dict(rw=rw, bs=int(bs), qd=qd, jobs=jobs,
                                  bw_gbs=bw / 1e9, iops=min(bw / bs, 4e5),
                                  lat_mean_us=80.0 * qd, lat_p99_us=200.0,
                                  mirror_share=0.49, n_drives=2))
    fio = pd.DataFrame(frows)
    orows = []
    for rep in range(3):
        for s in 2**np.arange(16, 27, 2):
            orows.append(dict(stream='single', bytes=int(s), rep=rep,
                              first_s=0.07, seconds=0.07 + s / 60e6))
        orows.append(dict(stream='8 streams', bytes=2**27, rep=rep,
                          first_s=math.nan, seconds=2**27 / 95e6))
    obj = pd.DataFrame(orows)
    obj.attrs['streams'] = 8
    return h2d, agg, dram, fio, obj


def synthetic_microbench():
    """box-microbench entries shaped like 22.1's: L2 of 64 MiB read at
    5,000 GB/s, GDDR at 950 GB/s; latencies 16, 98 and 220 ns; shared
    memory 36 TB/s."""
    l2 = 2**26
    sizes = 2**np.arange(18, 33)
    ladder = pd.DataFrame([dict(bytes=int(b), kind='read',
                                gbs=5000.0 if b <= l2 else 950.0)
                           for b in sizes])
    ladder.attrs.update(l2_bytes=l2, l1_bytes=2**17)
    lat = pd.DataFrame([dict(bytes=int(b), ns_per_load=16.0 if b <= 2**16
                             else 98.0 if b <= l2 else 220.0)
                        for b in 2**np.arange(12, 31)])
    lat.attrs.update(l2_bytes=l2, l1_bytes=2**17)
    return {'ladder': ladder, 'latency': lat, 'smem': dict(tbs=36.0)}


class TestMemoryTiers(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(d2l, '_hw_static',
                                    side_effect=lambda n: committed_tiers())
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(d2l.plt.close, 'all')

    def test_tier_table_contract(self):
        t = committed_tiers()
        self.assertEqual(len(t), t.tier.nunique())
        self.assertTrue(t.source_url.str.startswith('https://').all())
        self.assertTrue(t.retrieved.str.match(r'\d{4}-\d\d-\d\d$').all())
        ranks = t.cost_rank.dropna()
        self.assertTrue(((ranks >= 1) & (ranks <= 9)).all())
        self.assertTrue((t.bw_gbs.dropna() > 0).all())
        for tier in ('pcie_host', 'host_dram', 'nvme', 'object_store'):
            self.assertIn(tier, set(t.tier))  # the measured tiers
        self.assertAlmostEqual(t.set_index('tier').bw_gbs['pcie_host'],
                               d2l.pcie_ceiling(4, 16) / 1e9, places=1)

    def test_tier_table_display(self):
        lines, tiers = lines_of(self, d2l.tier_table)
        self.assertLessEqual(len(lines), 15)
        self.assertEqual(len(tiers), len(committed_tiers()))
        pending = d2l.CampaignPending('node-storage', 'C-T3', 'one node')
        lines, _ = lines_of(self, d2l.tier_table, pending)
        self.assertEqual(lines, ['one node: campaign C-T3 not run yet'])
        with mock.patch.object(d2l, 'hw_box_log',
                               side_effect=KeyError('not committed')):
            measured = d2l.tier_measurements(
                d2l.fit_h2d(synthetic_storage()[0]), *synthetic_storage()[1:])
        lines, t = lines_of(self, d2l.tier_table, {'tiers': measured})
        self.assertIn('measured GB/s', lines[0])
        self.assertAlmostEqual(
            t.set_index('tier').measured_bw_gbs['host_dram'], 110.0)

    def test_parse_fio(self):
        r = d2l.parse_fio(read(os.path.join(SAMPLES, 'box_fio_seqread.json')))
        self.assertAlmostEqual(r['bw_gbs'], 6.766, places=2)
        self.assertAlmostEqual(r['iops'], 6452.8, delta=1)
        self.assertEqual((r['rw'], r['bs'], r['qd'], r['jobs']),
                         ('read', 2**20, 32, 1))
        self.assertGreater(r['lat_p99_us'], r['lat_mean_us'] / 2)
        drives = r['drive_bytes']
        self.assertEqual(sorted(drives), ['nvme0n1', 'nvme1n1'])
        share = min(drives.values()) / sum(drives.values())
        self.assertGreater(share, 0.45)  # RAID 1 reads from both mirrors

    def test_parse_nvbandwidth(self):
        r = d2l.parse_nvbandwidth(read(os.path.join(SAMPLES,
                                                    'box_nvbandwidth.txt')))
        h = r['host_to_device_memcpy_ce']
        self.assertEqual(h.shape, (1, 4))
        self.assertAlmostEqual(h.sum(), r['host_to_device_memcpy_ce_sum'],
                               delta=0.05)
        self.assertAlmostEqual(r['host_to_device_memcpy_ce_sum'], 102.76)
        self.assertAlmostEqual(r['device_to_host_memcpy_ce'][0, 3], 24.83)
        self.assertNotIn('device_to_device_memcpy_read_ce', r)  # waived

    def test_parse_gdsio(self):
        g = d2l.parse_gdsio(read(os.path.join(SAMPLES, 'doc_gdsio.txt')))
        self.assertEqual(len(g), 9)
        self.assertEqual(set(g.xfer), {'CPUONLY', 'CPU_GPU', 'GPUD'})
        r = g[(g.io == 'READ') & (g.xfer == 'GPUD')].iloc[-1]
        self.assertEqual((r.threads, r.io_size), (4, 2**20))
        self.assertAlmostEqual(r.gib_s, 3.229036)
        self.assertAlmostEqual(r.lat_us, 1208.627088)

    def test_topology(self):
        rel = d2l._topo_relations(read(os.path.join(SAMPLES,
                                                    'box_topo_m.txt')))
        self.assertEqual(rel[(0, 1)], 'PHB')
        self.assertEqual(rel[(2, 3)], 'PHB')
        self.assertEqual(rel[(0, 2)], 'NODE')
        groups = d2l._bridge_groups(4, rel)
        self.assertEqual([g[1] for g in groups],
                         [(0,), (0, 1), (0, 2), (0, 1, 2, 3)])
        self.assertEqual(len(d2l._bridge_groups(2, {(0, 1): 'PHB'})), 2)

    def test_fio_command(self):
        with mock.patch.object(d2l, 'hw_bin', return_value='fio'):
            cmd = d2l._fio_cmd('f.dat', 2**34, 'randread', 4096, 64, 4)
        self.assertIn('--direct=1', cmd)
        self.assertIn('--ioengine=io_uring', cmd)
        self.assertIn(f'--offset_increment={2**32}', cmd)
        self.assertIn('--time_based', cmd)

    def test_fit_h2d(self):
        links = d2l.fit_h2d(synthetic_storage()[0])
        for mode, beta in (('pinned', 25e9), ('pinned-2M', 26e9),
                           ('pageable', 10e9)):
            r = links.loc[(mode, 'h2d')]
            self.assertAlmostEqual(r.beta / beta, 1, delta=0.03)
            self.assertAlmostEqual(r.beta_fit / beta, 1, delta=0.03)
            self.assertLessEqual(r.beta_lo, r.beta)
            self.assertGreaterEqual(r.beta_hi, r.beta)
            self.assertAlmostEqual(r.alpha * 1e6, 10, delta=1)
        # A plateau that is not alpha-beta shaped: the median of the large
        # copies, not the fit, is beta.
        h2d = synthetic_storage()[0]
        big = (h2d['mode'] == 'pinned') & (h2d.bytes == 2**30)
        h2d.loc[big, 'ms'] *= 25 / 20  # 20 GB/s at 1 GiB only
        r = d2l.fit_h2d(h2d).loc[('pinned', 'h2d')]
        self.assertAlmostEqual(r.beta_lo / 20e9, 1, delta=0.03)
        self.assertAlmostEqual(r.beta / 25e9, 1, delta=0.03)  # of 3 points

    def test_copies_per_timing(self):
        self.assertEqual(d2l._copies_per_timing(2**12), 50)
        self.assertEqual(d2l._copies_per_timing(2**30), 1)
        self.assertEqual(d2l._copies_per_timing(2**26), 2)

    def test_offload_breakeven(self):
        # The plan's A100 example: 250 TFLOP/s, 25 GB/s, 2 bytes each way.
        self.assertAlmostEqual(d2l.offload_breakeven(250e12, 25e9), 3333.33,
                               places=1)
        self.assertAlmostEqual(d2l.offload_breakeven(1e12, 1e9, 4),
                               4e12 / 6e9)

    def test_kv_load_vs_recompute(self):
        cfg = cached_config(self, 'Qwen/Qwen3-8B')
        n = 2**np.arange(8, 14)
        fast = d2l.kv_load_vs_recompute(cfg, n, 1e15, 1e14)
        self.assertEqual(fast['crossover'], 1)
        slow = d2l.kv_load_vs_recompute(cfg, n, 1e3, 1e14)
        self.assertEqual(slow['crossover'], np.inf)
        k = fast['k_kv']  # 36 layers x 8 heads x 128 x 2 (K, V) x 2 bytes
        self.assertEqual(k, 36 * 8 * 128 * 2 * 2)
        self.assertEqual(slow['n_max'], cfg.max_position_embeddings)
        r = d2l.kv_load_vs_recompute(cfg, n, 25e9, 1.5e14, alpha=1e-3)
        np.testing.assert_allclose(r['load_s'], 1e-3 + n * k / 25e9)
        self.assertTrue(np.all(np.diff(r['recompute_s']) > 0))
        # Above the crossover loading wins, below it recomputing does.
        c = r['crossover']
        self.assertTrue(np.isfinite(c) and c > 1)
        at = d2l.kv_load_vs_recompute(cfg, np.array([c, c // 2]), 25e9,
                                      1.5e14, alpha=1e-3)
        self.assertLessEqual(at['load_s'][0], at['recompute_s'][0])
        self.assertGreater(at['load_s'][1], at['recompute_s'][1])
        # The search stops at the context: a crossover beyond it is never.
        short = d2l.kv_load_vs_recompute(cfg, n, 25e9, 1.5e14, alpha=1e-3,
                                         n_max=int(c) // 4)
        self.assertEqual(short['crossover'], np.inf)
        # beta_eff replaces the chip's datasheet memory rate.
        chip = pd.Series(dict(p16_tflops=150.0, p8_tflops=np.nan,
                              mem_bw_tbs=1.0, mem_gb=24.0))
        a = d2l.kv_load_vs_recompute(cfg, [1], 25e9, 1.5e14, chip)
        b = d2l.kv_load_vs_recompute(cfg, [1], 25e9, 1.5e14, chip,
                                     beta_eff=0.5e12)
        self.assertGreater(b['recompute_s'][0], a['recompute_s'][0])
        self.assertEqual([d2l.prefix_words(x) for x in (1, 51234, np.inf)],
                         ['any', '51,000 tokens', 'never'])
        self.assertEqual(d2l.prefix_words(np.inf, 40960),
                         'never within 40,960')

    def test_measurements_and_displays(self):
        h2d, agg, dram, fio, obj = synthetic_storage()
        links = d2l.fit_h2d(h2d)
        lines, _ = lines_of(self, d2l.plot_h2d, h2d, links)
        self.assertEqual(len(lines), 6)
        self.assertEqual(lines[0], 'plateau beta: the median of the copies '
                         'of 256 MiB and more, and their range')
        self.assertRegex(lines[1], r'^pinned h2d: alpha 10 us, beta 25 GB/s '
                         r'\(range \d+-\d+\), \d+% of 31\.5$')
        self.assertIn('pinned, 2 MiB pages: h2d 26 GB/s', lines[3])
        self.assertIn('Gen 4 x16 during the sweep, Gen 1 x16 at rest',
                      lines[-1])
        lines, _ = lines_of(self, d2l._show_concurrent, agg)
        self.assertEqual(len(lines), 6)
        self.assertIn('(90% of 4 x one card)', lines[3])
        # M2: one bridge 2 x 24 x 0.95 = 45.6; four cards 86.4 = 95% of 2 x it
        self.assertEqual(lines[4], 'per host bridge: about 46 GB/s h2d (two '
                         'cards behind it); all 4: 95% of 2 x 46')
        self.assertIn('h2d 26 GB/s', lines[5])
        two = agg[agg.group.isin(['one card', 'two cards, two host bridges'])]
        self.assertIsNone(d2l._per_bridge(two))  # no pair behind one bridge
        lines, _ = lines_of(self, d2l._show_dram, dram)
        self.assertEqual(len(lines), 2)
        self.assertIn('triad 90 GB/s (16 threads)', lines[0])
        self.assertIn('reaches 44%', lines[1])
        lines, _ = lines_of(self, d2l._show_fio, fio)
        self.assertEqual(len(lines), 3)
        self.assertIn('73% of 2 x 4.12', lines[0])
        self.assertIn('49%', lines[1])
        beta, iops, knee = d2l._fio_knee(fio)
        self.assertAlmostEqual(beta / 6e9, 1, delta=0.01)
        self.assertAlmostEqual(knee, beta / iops)
        self.assertEqual(d2l._fio_latency(fio).bs, 4096)
        self.assertEqual(d2l._fio_latency(fio).qd, 1)
        o = d2l.object_fit(obj)
        self.assertAlmostEqual(o['alpha'], 0.07, delta=0.005)
        self.assertAlmostEqual(o['beta'] / 60e6, 1, delta=0.02)
        self.assertAlmostEqual(o['beta_streams'], 95e6)
        lines, _ = lines_of(self, d2l._show_object, obj)
        self.assertEqual(len(lines), 2)
        with mock.patch.object(d2l, 'hw_box_log',
                               return_value=synthetic_microbench()):
            m = d2l.tier_measurements(links, agg, dram, fio, obj)
        self.assertEqual((m.attrs['n_cards'], m.attrs['n_drives']), (4, 2))
        self.assertAlmostEqual(m.measured_bw_gbs['pcie_host_all'], 86.4)
        self.assertAlmostEqual(m.measured_bw_gbs['nvme'], 6.0, delta=0.05)
        self.assertAlmostEqual(m.measured_latency_ns['nvme'], 80e3)  # QD 1
        self.assertEqual(m.measured_bw_gbs['host_dram'], 110.0)
        self.assertEqual(m.measured_bw_gbs['gddr'], 950.0)
        self.assertEqual(m.measured_bw_gbs['l2'], 5000.0)
        self.assertEqual(m.measured_bw_gbs['smem'], 36000.0)
        self.assertEqual(m.measured_latency_ns['gddr'], 220.0)
        self.assertEqual(m.measured_latency_ns['l2'], 98.0)
        self.assertEqual(m.measured_latency_ns['smem'], 16.0)
        compare = d2l.plot_tiers(committed_tiers(), m)
        self.assertEqual(len(compare), 3)  # host link, GDDR, the drives
        self.assertNotIn('host DRAM', compare.index)
        self.assertAlmostEqual(compare.fraction['host link (PCIe)'], 25 / 31.5,
                               delta=0.02)
        nvme = [t for t in compare.index if 'RAID 1' in t]
        self.assertEqual(len(nvme), 1)
        self.assertAlmostEqual(compare.datasheet[nvme[0]], 2 * 4.125)
        lines, _ = lines_of(self, d2l.show_table, compare)
        lines, rule = lines_of(self, d2l.shard_rule, links, fio, obj)
        self.assertEqual(len(lines), 4)
        self.assertAlmostEqual(rule.s_90.iloc[-1], 9 * 0.07 * 60e6,
                               delta=0.05 * 9 * 0.07 * 60e6)
        self.assertAlmostEqual(rule.s_90.iloc[0], 9 * 1e-5 * 25e9,
                               delta=0.1 * 9 * 1e-5 * 25e9)
        recs = d2l._tiers_json(links, fio, obj)
        self.assertEqual({r['tier'] for r in recs},
                         {f'{d}_{m}' for d in ('h2d', 'd2h')
                          for m in ('pinned', 'pinned-2M', 'pageable')}
                         | {'nvme_seq_read', 'object_store'})
        nvme = [r for r in recs if r['tier'] == 'nvme_seq_read'][0]
        self.assertEqual(nvme['latency_us'], 80.0)  # one request, QD 1
        self.assertAlmostEqual(nvme['bw_gbs'], 6.0, delta=0.05)
        json.dumps(recs)  # the blueprint's file is JSON
        lines, _ = lines_of(self, d2l._show_shard, dict(
            bytes=2**27, drive_bps=3e9, loader_tps=4e8, itemsize=2))
        self.assertEqual(len(lines), 2)

    def test_offload_crossover_and_plot(self):
        # Stalled at 1 s up to 10,000 tokens, then the compute-only arm's
        # 0.1 ms per token; the CPU update alone 0.9 s.
        T = 2**np.arange(10, 18)
        rows = [dict(tokens=t, arm='offload', seconds=max(1.0, 1e-4 * t),
                     gpu_s=1e-4 * t, update_s=0.9) for t in T]
        rows += [dict(tokens=t, arm='offload-32', seconds=max(1.5, 1e-4 * t),
                      gpu_s=2e-4 * t, update_s=0.8) for t in T]
        rows += [dict(tokens=t, arm='compute', seconds=1e-4 * t,
                      gpu_s=1e-4 * t, update_s=np.nan) for t in T]
        rows += [dict(tokens=0, arm='transfer', seconds=0.1),
                 dict(tokens=0, arm='cpu', seconds=0.6)]
        run = pd.DataFrame(rows)
        run.attrs.update(N=1e9, peak_gib=14.0, cpu_threads=8,
                         contention_threads=32)
        self.assertAlmostEqual(d2l.offload_crossover(run), 10000, delta=100)
        pred = d2l.offload_prediction(run)
        self.assertAlmostEqual(pred['p_train'], 6e13)
        self.assertAlmostEqual(pred['predicted'], 9000)  # in-run update
        self.assertAlmostEqual(pred['t_cpu_alone'], 0.6)
        self.assertAlmostEqual(pred['T_link'], 1000)
        lines, _ = lines_of(self, d2l.plot_offload, run)
        self.assertEqual(len(lines), 5)
        self.assertIn('60 TFLOP/s', lines[0])
        self.assertIn('0.60 s alone, 0.90 s during', lines[1])
        self.assertIn('predicted 9,000, measured 10,000 tokens (1.1x)',
                      lines[2])
        self.assertIn('by 0.00 s', lines[3])
        self.assertIn('16,384 tokens: 1.64 s alone, 1.64 s (8 CPU threads), '
                      '3.28 s (32)', lines[4])

    @unittest.skipUnless(torch.cuda.is_available(), 'pinned memory')
    def test_cpu_adam_matches_adamw(self):
        torch.manual_seed(0)
        w = [torch.randn(1000), torch.randn(10, 30)]
        adam = d2l._CPUAdam([x.shape for x in w], w, lr=1e-2)
        ref = [x.clone().requires_grad_() for x in w]
        opt = torch.optim.AdamW(ref, lr=1e-2)
        for _ in range(3):
            g = [torch.randn_like(x).bfloat16() for x in w]
            for i in range(2):
                adam.g16[0][i].copy_(g[i])
                adam.update(i)
                ref[i].grad = g[i].float()
            opt.step()
        for i in range(2):
            torch.testing.assert_close(adam.master[i], ref[i].detach())
            torch.testing.assert_close(adam.w16[0][i],
                                       ref[i].detach().bfloat16())

    def test_scratch_refuses_full_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(d2l, '_HW_SCRATCH', tmp):
                self.assertIsNone(d2l._scratch(2**30, free_gb=1e9))
                path = d2l._scratch(2**30, free_gb=0)
                self.assertTrue(path.startswith(tmp))


def committed_links():
    return pd.read_csv(os.path.join(HW_DATA, 'links.csv'))


def sample(name):
    return read(os.path.join(HW_DATA, 'samples', name))


def synthetic_sweep(op, g, path, alpha, B, seed=0, noise=0.01):
    """A sweep table of t = alpha + S / B (B: algorithm bandwidth) with
    1% multiplicative noise, 8 B to 256 MiB."""
    rng = np.random.default_rng(seed)
    s = 8 * 2.0 ** np.arange(26)
    t = (alpha + s / B) * (1 + noise * rng.standard_normal(len(s)))
    algbw = s / t / 1e9
    return pd.DataFrame(dict(
        size=s.astype(int), count=(s // 4).astype(int), dtype='float',
        redop='sum', root=-1, oop_us=t * 1e6, oop_algbw=algbw,
        oop_busbw=algbw * d2l.nccl_busbw_factor(op, g), op=op, g=g,
        path=path, algo='default', env='', gpus='0,1'))


class TestServerPaths(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.object(d2l, '_hw_static',
                                    side_effect=lambda n: committed_links())
        patcher.start()
        self.addCleanup(patcher.stop)

    def box(self):
        return (d2l.parse_topo_matrix(sample('box_topo_m.txt')),
                d2l.parse_topo_matrix(sample('box_topo_p2p_r.txt')))

    def test_links_table(self):
        raw = committed_links()
        self.assertTrue(raw.source_url.str.startswith('https://').all())
        self.assertTrue(raw.retrieved.str.match(r'\d{4}-\d\d-\d\d$').all())
        lines, links = lines_of(self, d2l.links_table)
        self.assertLessEqual(len(lines), 13)
        self.assertAlmostEqual(links.loc['PCIe Gen 4 x16', 'gbs_per_dir'],
                               16 * 16 * 128 / 130 / 8)
        self.assertAlmostEqual(links.loc['PCIe Gen 3 x16', 'gbs_bidir'],
                               2 * 16 * 8 * 128 / 130 / 8)
        self.assertEqual(links.loc['PCIe Gen 6 x16', 'gbs_per_dir'], 128)
        self.assertEqual(links.loc['NVLink 3 (A100 SXM)', 'gbs_per_dir'], 300)
        self.assertTrue((links.gbs_bidir == 2 * links.gbs_per_dir).all())
        self.assertEqual(len(links), len(raw))

    def test_topo_box(self):
        topo, p2p = self.box()
        self.assertEqual(topo.labels, ['GPU0', 'GPU1', 'GPU2', 'GPU3'])
        self.assertEqual(topo.link[0], ['X', 'PHB', 'NODE', 'NODE'])
        self.assertEqual(topo.link[3][2], 'PHB')
        self.assertEqual(set(topo.cpu_affinity), {'0-63'})
        self.assertEqual(set(topo.numa_affinity), {'0'})
        self.assertIn('PHB', topo.legend)
        self.assertEqual(p2p.labels, topo.labels)
        off = {p2p.link[i][j] for i in range(4) for j in range(4) if i != j}
        self.assertEqual(off, {'CNS'})

    def test_topo_samples(self):
        nvs = d2l.parse_topo_matrix(sample('doc_nvswitch_node_topo_m.txt'))
        self.assertEqual(nvs.labels[-2:], ['NIC0', 'NIC1'])
        self.assertEqual({nvs.link[i][j] for i in range(8) for j in range(8)
                          if i != j}, {'NV12'})
        self.assertIsNone(nvs.cpu_affinity[8])
        self.assertEqual(nvs.numa_affinity[2], '1')
        br = d2l.parse_topo_matrix(sample('doc_bridged_pair_topo_m.txt'))
        self.assertEqual(len(br.labels), 5)
        self.assertEqual((br.link[0][1], br.link[2][3], br.link[0][2],
                          br.link[0][4]), ('NV12', 'NV12', 'PXB', 'SYS'))
        self.assertEqual(br.cpu_affinity[4], '32-63')

    def test_topo_broken_header(self):
        text = ('\x1b[4m\tGPU0GPU1\tCPU Affinity\x1b[0m\r\n'
                'GPU0\t X \tNV4\t0-7\t0\r\nGPU1\tNV4\t X \t0-7\t0\r\n')
        topo = d2l.parse_topo_matrix(text)
        self.assertEqual(topo.labels, ['GPU0', 'GPU1'])
        self.assertEqual(topo.link[0][1], 'NV4')
        with self.assertRaises(ValueError):
            d2l.parse_topo_matrix('no matrix here\n')
        with self.assertRaisesRegex(ValueError, 'link cells'):
            d2l.parse_topo_matrix('GPU0\t X \tPHB\nGPU1\tPHB\n')

    def test_topo_graph(self):
        links = d2l.links_table(show=False)
        g = d2l.topo_graph(self.box()[0], links)
        self.assertEqual(len(g.edges), 6)
        self.assertEqual(g.edges['GPU0', 'host bridge 0']['gbs'],
                         links.loc['PCIe Gen 4 x16', 'gbs_per_dir'])
        self.assertEqual(g.edges['GPU3', 'host bridge 1']['link'],
                         'PCIe Gen 4 x16')
        self.assertTrue(np.isnan(g.edges['host bridge 0', 'CPU 0']['gbs']))
        nvs = d2l.topo_graph(d2l.parse_topo_matrix(
            sample('doc_nvswitch_node_topo_m.txt')), links)
        self.assertEqual(nvs.degree['NVSwitch'], 8)
        self.assertEqual(nvs.edges['GPU5', 'NVSwitch']['gbs'], 300)
        self.assertIn('PCIe switch 0', nvs['NIC1'])
        br = d2l.topo_graph(d2l.parse_topo_matrix(
            sample('doc_bridged_pair_topo_m.txt')), links)
        self.assertNotIn('NVSwitch', br)
        self.assertEqual(br.edges['GPU2', 'GPU3']['link'],
                         'NVLink bridge 3 (A100 PCIe)')
        self.assertIn(('CPU 0', 'CPU 1'), br.edges)
        lines, _ = lines_of(self, d2l.show_edges, g)
        self.assertEqual(len(lines), 7)

    def test_show_topo(self):
        topo, p2p = self.box()
        samples = {'nvswitch-node': 'doc_nvswitch_node_topo_m.txt',
                   'bridged-workstation': 'doc_bridged_pair_topo_m.txt'}
        with mock.patch.object(d2l, 'hw_capture',
                               side_effect=lambda n: sample(samples[n])):
            lines, _ = lines_of(self, d2l.show_topo, topo, p2p,
                                samples=tuple(samples))
        self.assertEqual(len(lines), 10)
        self.assertIn('CNS', lines[2])
        self.assertIn('published samples (GitHub issues), not measured '
                      'here', lines[-3])
        self.assertIn('bridged-workstation: 5 GPUs', lines[-1])
        self.assertIn('28 NV12', lines[-2])
        self.assertIn('2 NV12', lines[-1])
        with self.assertRaises(KeyError):
            d2l.hw_capture('nvidia-smi -q')

    def test_parse_p2p(self):
        m = d2l.parse_p2p(sample('box_p2pBandwidthLatencyTest.txt'))
        self.assertEqual(len(m), 8)
        for v in m.values():
            self.assertEqual(v.shape, (4, 4))
        bw = m['Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)']
        off = bw[~np.eye(4, dtype=bool)]
        self.assertTrue(((off > 5) & (off < 40)).all())
        self.assertIn('P2P=Disabled Latency Matrix (us) CPU', m)

    def test_parse_nccl_tests(self):
        df = d2l.parse_nccl_tests(sample('box_all_reduce_perf_g4.txt'))
        self.assertEqual(list(df.columns), list(d2l._NCCL_COLUMNS))
        self.assertEqual((len(df), df['size'].iloc[0],
                          df['size'].iloc[-1]), (26, 8, 2**28))
        self.assertEqual(df.attrs['version'], '2.21.1')
        self.assertGreater(df.attrs['avg_busbw'], 0)
        big = df.iloc[-1]  # busbw = algbw x 2 (k - 1) / k, as printed
        self.assertAlmostEqual(big.oop_busbw / big.oop_algbw,
                               d2l.nccl_busbw_factor('all_reduce', 4),
                               places=2)
        a2a = d2l.parse_nccl_tests(sample('box_alltoall_perf_g4.txt'))
        self.assertEqual(len(a2a), 26)
        with self.assertRaisesRegex(ValueError, 'column layout'):
            d2l.parse_nccl_tests('#  size  time  algbw\n 8  1.0  0.0\n')

    def test_busbw_and_coll_time(self):
        f = d2l.nccl_busbw_factor
        self.assertEqual((f('all_reduce', 4), f('allgather', 4),
                          f('alltoall', 8), f('broadcast', 8)),
                         (1.5, 0.75, 7 / 8, 1.0))
        with self.assertRaises(ValueError):
            f('scan', 4)
        S, beta = 2**28, 2e9
        for k in (2, 4, 8):
            self.assertAlmostEqual(d2l.coll_time('allreduce', S, k, 0, beta),
                                   2 * (k - 1) / k * S / beta)
            self.assertAlmostEqual(d2l.coll_time('all_to_all', S, k, 0,
                                                 beta), (k - 1) / k * S / beta)
            self.assertAlmostEqual(d2l.coll_time('allreduce', 8, k, 1e-5,
                                                 beta) * 1e5,
                                   2 * (k - 1), delta=0.01)
        self.assertAlmostEqual(d2l.coll_time('allreduce', S, 8, 0, beta,
                                             algo='tree'), 6 * S / beta)
        for op in ('broadcast', 'reduce'):  # not modelled: refused
            with self.assertRaises(ValueError):
                d2l.coll_time(op, S, 4, 0, beta)

    def test_tree_link_loads(self):
        S = 1.0
        for k in (4, 8, 16):
            halves = [tuple(range(k // 2)), tuple(range(k // 2, k))]
            a2a = d2l.tree_link_loads(k, halves, 'alltoall', S)
            ring = d2l.tree_link_loads(k, halves, 'allreduce', S)
            self.assertAlmostEqual(a2a['uplink 0'], k * S / 4)
            self.assertAlmostEqual(a2a['GPU 0'], (k - 1) / k * S)
            self.assertEqual(set(ring.values()), {2 * (k - 1) / k * S})
            # D2: the tree's all-to-all is k^2 / (4 (k - 1)) times slower
            self.assertAlmostEqual(
                1 / d2l.a2a_busbw_ratio(k, halves, 1.0, 1.0),
                k**2 / (4 * (k - 1)))
            self.assertAlmostEqual(d2l.a2a_busbw_ratio(
                k, halves, 3.0, 0.5, staged=True), 1.0)
            staged = d2l.tree_link_loads(k, halves, 'alltoall', staged=True)
            self.assertAlmostEqual(staged['uplink 1'], (k - 1) * S / 2)
        self.assertAlmostEqual(1 / d2l.a2a_busbw_ratio(
            8, [tuple(range(4)), tuple(range(4, 8))], 1, 1), 64 / 28)
        one = [tuple(range(8))]  # one switch: the crossbar
        self.assertAlmostEqual(d2l.a2a_busbw_ratio(8, one, 1, 0.01), 1.0)
        # the box's numbers: the tree's all-to-all is GPU-link bound, the
        # star's uplink bound, 1.5 times apart
        box, beta, beta_u = [(0, 1), (2, 3)], 22e9, 28e9
        tree = d2l.pattern_time(4, box, 'alltoall', beta, beta_u, 2**28)
        star = d2l.pattern_time(4, box, 'alltoall', beta, beta_u, 2**28,
                                staged=True)
        self.assertAlmostEqual(tree, 2**28 / 28e9)
        self.assertAlmostEqual(star, 1.5 * 2**28 / 28e9)
        flows = d2l._pattern_flows(box, 'alltoall', 2**28)
        self.assertEqual((len(flows), flows[0][2]), (12, 2**26))
        ring = d2l._pattern_flows(box, 'allreduce', 2**28)
        self.assertEqual([f[:2] for f in ring],
                         [(0, 1), (1, 2), (2, 3), (3, 0)])
        self.assertEqual(ring[0][2], 3 * 2**28 // 2)
        with self.assertRaises(AssertionError):
            d2l.tree_link_loads(4, [(0, 1)], 'alltoall')
        with self.assertRaises(ValueError):
            d2l.tree_link_loads(4, [(0, 1), (2, 3)], 'scan')

    def test_bridge_groups(self):
        topo = self.box()[0]
        rel = lambda i, j: topo.link[i][j]
        groups, pats = d2l._bridge_pairs(rel, 4)
        self.assertEqual(groups, [(0, 1), (2, 3)])
        self.assertEqual(pats['two pairs across the bridges'],
                         [(0, 2), (1, 3)])
        self.assertEqual(d2l._placements(rel, 4),
                         {'PHB': (0, 1), 'NODE': (0, 2), 'all': (0, 1, 2, 3)})
        self.assertEqual(d2l._placements(rel, 2), {'PHB': (0, 1)})
        self.assertEqual(d2l._bridge_pairs(rel, 2)[1], {})
        gbs = np.array([[np.nan, 19.4, 22.0, 21.9], [19.5, np.nan, 22.0, 22.0],
                        [22.0, 22.0, np.nan, 19.6], [22.0, 22.0, 19.6, np.nan]])
        mat = dict(gbs=gbs, relation=[[topo.link[i][j] for j in range(4)]
                                      for i in range(4)])
        self.assertEqual(d2l._by_relation(mat), {'NODE': 22.0, 'PHB': 19.55})
        mat['peer'] = np.zeros((4, 4))
        self.assertEqual(d2l._record_link((0, 1), mat), 'pcie_host_staged')
        mat['peer'] = 1 - np.eye(4)
        self.assertEqual(d2l._record_link((0, 1, 2, 3), mat), 'pcie_p2p')
        self.assertEqual(d2l._record_link((0, 1), mat, 'NCCL_P2P_DISABLE=1'),
                         'pcie_host_staged')
        mat['relation'] = [['X' if i == j else 'NV12' for j in range(4)]
                           for i in range(4)]
        self.assertEqual(d2l._record_link((0, 2), mat), 'nvlink')

    def test_fit_collectives(self):
        nccl = pd.concat([
            synthetic_sweep('all_reduce', 2, 'PHB', 20e-6, 2e9, seed=0),
            synthetic_sweep('alltoall', 4, 'all', 40e-6, 1e9, seed=1)],
            ignore_index=True)
        ab = d2l.fit_collectives(nccl)
        r = ab.loc[('all_reduce', 2, 'PHB')]
        self.assertAlmostEqual(r.alpha / 20e-6, 1, delta=0.05)
        self.assertAlmostEqual(r.beta / (2e9 * 1.0), 1, delta=0.03)
        self.assertTrue(np.isnan(r.beta_lo))  # one process: no interval
        self.assertAlmostEqual(r.n_half, r.alpha * 2e9, delta=0.1 * r.n_half)
        self.assertAlmostEqual(r.alpha_step, r.alpha / 2)
        self.assertLess(r.resid_rel, 0.02)
        a = ab.loc[('alltoall', 4, 'all')]
        self.assertAlmostEqual(a.beta / (0.75 * 1e9), 1, delta=0.03)
        self.assertEqual(int(a.n_points), 26)
        self.assertTrue(d2l.fit_collectives(nccl, algo='Ring').empty)
        lines, _ = lines_of(self, d2l.show_fits, ab)
        self.assertEqual(len(lines), 3)  # no NODE pair: no S2 sentence
        self.assertIn('all-reduce', lines[1])
        self.assertIn(' - ', lines[1] + ' ')
        both = pd.concat([nccl, synthetic_sweep(
            'all_reduce', 2, 'NODE', 20e-6, 3e9, seed=3)], ignore_index=True)
        text = d2l._pair_paths_line(d2l.fit_collectives(both))
        self.assertEqual(text, 'two-card all-reduce: across the host bridges '
                         '(NODE) 3.0 GB/s, behind one bridge (PHB) 2.0 GB/s '
                         '(1.5x; no intervals (one process))')
        # three processes: intervals around the combined fit
        rng = np.random.default_rng(2)
        nccl.attrs['proc_us'] = [
            (nccl.oop_us * (1 + 0.02 * rng.standard_normal())).tolist()
            for _ in range(3)]
        r = d2l.fit_collectives(nccl).loc[('all_reduce', 2, 'PHB')]
        self.assertLess(r.beta_lo, r.beta)
        self.assertLess(r.beta, r.beta_hi)
        self.assertLess(r.alpha_lo, r.alpha_hi)
        # a plateau that is not alpha-beta shaped: the two regimes read it,
        # one line through every size would not
        bent = synthetic_sweep('all_reduce', 4, 'all', 30e-6, 1.6e9, noise=0)
        mid = (bent["size"] > 2**16) & (bent["size"] < 2**22)
        bent.loc[mid, 'oop_us'] *= 0.7  # a bump of bandwidth near 1 MiB
        r = d2l.fit_collectives(bent).loc[('all_reduce', 4, 'all')]
        self.assertAlmostEqual(r.beta / (1.5 * 1.6e9), 1, delta=0.02)
        self.assertGreater(r.beta_fit, 1.05 * r.beta)
        self.assertAlmostEqual(r.alpha / 30e-6, 1, delta=0.05)
        with self.assertRaisesRegex(ValueError, 'sizes'):
            d2l.fit_collectives(bent, plateau_from=2**30)

    def test_fit_pending(self):
        log = d2l.CampaignPending('node-p2p', 'C-T3', 'one 8 x A100 node')
        ab = d2l.fit_collectives(log)
        self.assertTrue(ab.empty)
        lines, _ = lines_of(self, d2l.show_fits, ab)
        self.assertEqual(lines, ['one 8 x A100 node: campaign C-T3 not run '
                                 'yet'])

    def test_alpha_beta_records(self):
        nccl = pd.concat([
            synthetic_sweep('all_reduce', 4, 'all', 25e-6, 1.5e9),
            synthetic_sweep('alltoall', 4, 'all', 27e-6, 2.7e9)],
            ignore_index=True)
        info = {'all': dict(transports=['SHM/direct/direct'])}
        recs = d2l.alpha_beta_records(d2l.fit_collectives(nccl),
                                      {'all': 'pcie_host_staged'}, info=info)
        fields = {'scope', 'link', 'op', 'g', 'algo', 'alpha_s', 'beta_Bps',
                  'alpha_ci95', 'beta_ci95', 'n_half_bytes', 'resid_rel',
                  'n_points', 'proto', 'transport'}
        for r in recs:
            self.assertTrue(fields <= set(r), r)
            self.assertEqual((r['scope'], r['link'], r['transport'],
                              r['algo']),
                             ('intra_node', 'pcie_host_staged', 'shm', None))
            self.assertEqual(r['beta_ci95'], [None, None])  # one process
            json.dumps(r, allow_nan=False)
        off = nccl.assign(env='NCCL_P2P_DISABLE=1')
        nv = d2l.alpha_beta_records(
            d2l.fit_collectives(off, env='NCCL_P2P_DISABLE=1'),
            {'all': 'nvlink'}, info={'all': dict(transports=['P2P/CUMEM'])})
        self.assertEqual({(r['link'], r['transport']) for r in nv},
                         {('nvlink', 'shm')})
        self.assertEqual({r['op'] for r in recs}, {'all_reduce',
                                                   'all_to_all'})
        self.assertEqual(d2l._transport({'transports': ['P2P/CUMEM']}),
                         'p2p')
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'alpha_beta.json')
            two = dict(recs[0], g=2, path='PHB')
            with open(path, 'w') as f:
                json.dump([two] + recs, f)
            with mock.patch.object(d2l, '_hw_box_file', return_value=path):
                r = d2l.box_alpha_beta()
                self.assertEqual((r['g'], r['op']), (4, 'all_reduce'))
                self.assertEqual(r['alpha'], r['alpha_s'])
                self.assertEqual(d2l.box_alpha_beta(g=2)['path'], 'PHB')
                self.assertEqual(d2l.box_alpha_beta('alltoall')['op'],
                                 'all_to_all')
            with open(path, 'w') as f:
                json.dump([two, dict(two, path='NODE')] + recs, f)
            with mock.patch.object(d2l, '_hw_box_file', return_value=path):
                with self.assertRaisesRegex(ValueError, 'pass path'):
                    d2l.box_alpha_beta(g=2)
                self.assertEqual(d2l.box_alpha_beta(g=2, path='NODE')['path'],
                                 'NODE')
                with self.assertRaises(KeyError):
                    d2l.box_alpha_beta(algo='Ring')
            with mock.patch.object(d2l, '_hw_box_file', return_value=None):
                with self.assertRaises(KeyError):
                    d2l.box_alpha_beta()

    def test_rows_from_times(self):
        df = d2l._rows_from_times([dict(size=2**20, time_us=1000.0)],
                                  'all_reduce', 4)
        self.assertEqual(list(df.columns), list(d2l._NCCL_COLUMNS))
        self.assertAlmostEqual(df.oop_algbw.iloc[0], 2**20 / 1e6)
        self.assertAlmostEqual(df.oop_busbw.iloc[0],
                               1.5 * df.oop_algbw.iloc[0])

    def test_scrub(self):
        with mock.patch('platform.node', return_value='boxname.lab.org'):
            out = d2l._hw_scrub('boxname:123:9 NCCL INFO via 10.0.0.12 '
                                '/home/someone/x boxname.lab.org boxnamex '
                                'sum -1boxname:7:7')
        self.assertEqual(out, '<host>:123:9 NCCL INFO via <ip> <path>/x '
                         '<host> boxnamex sum -1<host>:7:7')


def committed_systems():
    with open(os.path.join(HW_DATA, 'systems.json')) as f:
        return json.load(f)


BOX_AB = dict(g=4, alpha=30e-6, beta=2.4e9, alpha_s=30e-6, beta_Bps=2.4e9)

PERFTEST_BW = """\
---------------------------------------------------------------------------------------
                    RDMA_Write BW Test
 Dual-port       : OFF		Device         : mlx5_0
 Number of qps   : 1		Transport type : IB
---------------------------------------------------------------------------------------
 #bytes     #iterations    BW peak[Gb/sec]    BW average[Gb/sec]   MsgRate[Mpps]
 2          5000           0.12               0.11   		   6.824509
 1024       5000           62.10              61.85  		   7.549821
 1048576    5000           196.40             196.32 		   0.023403
 2097152    5000           196.50             196.44 		   0.011709
---------------------------------------------------------------------------------------
"""

PERFTEST_LAT = """\
---------------------------------------------------------------------------------------
                    RDMA_Write Latency Test
---------------------------------------------------------------------------------------
 #bytes #iterations    t_min[usec]    t_max[usec]  t_typical[usec]    t_avg[usec]    t_stdev[usec]   99% percentile[usec]   99.9% percentile[usec] 
 2       1000          1.02           3.63         1.06     	       1.08        	0.05   		1.27    		3.63   
 64      1000          1.10           2.98         1.14     	       1.15        	0.04   		1.30    		2.98   
 4096    1000          2.01           4.10         2.05     	       2.07        	0.06   		2.30    		4.10   
---------------------------------------------------------------------------------------
"""


class TestNetworks(unittest.TestCase):
    """sec_lm_networks: the systems table, fat tree, D1 ratios, layout
    calculator, loopback displays and the perftest parser (a hand-written
    sample of perftest's format: the box has no RDMA device)."""

    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(
            d2l, '_hw_static', side_effect=lambda n: committed_systems()))
        stack.enter_context(mock.patch.object(
            d2l, 'box_alpha_beta', return_value=dict(BOX_AB)))
        stack.enter_context(specs_patch())

    def test_systems_contract(self):
        rows = committed_systems()['rows']
        self.assertLessEqual(len(rows), 12)
        for r in rows:
            self.assertTrue(r['source_url'], r['system'])
            self.assertRegex(r['retrieved'], r'\d{4}-\d\d-\d\d$')
            self.assertIn(r['scaleup_dir'], ('bidir', 'per_dir'))
            self.assertIn(r['kind'], ('quoted', 'measured', 'spec'))
        specs = d2l.chip_specs()
        for r in rows:
            if r['chip']:
                self.assertIn(r['chip'], specs.index)

    def test_systems_table(self):
        sy = d2l.systems_table()
        self.assertEqual(sy.loc['8 x A100 SXM 40 GB', 'up_gbs_dir'], 300)
        self.assertEqual(sy.loc['TPU v7 pod', 'up_gbs_dir'], 200)  # 2 links
        self.assertEqual(sy.loc['this box (4 x RTX 4090)', 'up_gbs_dir'], 2.4)
        self.assertAlmostEqual(sy.loc['GB200 NVL72', 'domain_mem_tb'],
                               72 * 186.1 / 1e3)
        # 'no scale-out network' (the box) is not 'rate not stated' (Trn3)
        self.assertFalse(sy.loc['this box (4 x RTX 4090)', 'has_scaleout'])
        self.assertTrue(sy.loc['Trn3 UltraServer', 'has_scaleout'])
        lines, _ = lines_of(self, d2l.show_table,
                            sy[list(d2l.SYSTEMS_SHORT)],
                            headers=d2l.SYSTEMS_SHORT)
        self.assertLessEqual(len(lines), 13)

    def test_fat_tree_counts(self):
        for r in (4, 8, 16):
            two = d2l.fat_tree(r, 2)
            self.assertEqual((two['endpoints'], two['switches']),
                             (r**2 // 2, 3 * r // 2))
            self.assertEqual(two['links'], r * r)
            three = d2l.fat_tree(r, 3)
            self.assertEqual((three['endpoints'], three['switches']),
                             (r**3 // 4, 5 * r**2 // 4))
            self.assertEqual(three['links'], 3 * r**3 // 4)
            # every switch uses at most radix ports
            G = three['graph']
            self.assertLessEqual(max(G.degree(n) for n in G
                                     if not n.startswith('host')), r)

    def test_fat_tree_flow(self):
        beta = 50e9
        for r, o, tiers, rails in ((8, 1.0, 2, 1), (8, 3.0, 2, 2),
                                   (16, 1.0, 3, 1), (16, 3.0, 3, 4),
                                   (64, 2.0, 2, 8)):
            ft = d2l.fat_tree(r, tiers, oversub=o, rails=rails)
            self.assertAlmostEqual(ft['down'] / ft['up'], o)
            flow = d2l.cross_leaf_flow(ft, beta)
            self.assertAlmostEqual(flow, ft['up'] * beta)  # beta u, exact
            self.assertEqual(ft['hosts'] * rails, ft['endpoints'])

    def test_rail_wiring(self):
        ft = d2l.fat_tree(8, 2, rails=4)
        nics = {}
        for ep, leaf in ft['attach'].items():
            nics.setdefault(leaf, set()).add(ep.split('/')[1])
        self.assertTrue(all(len(v) == 1 for v in nics.values()))

    def test_layer_params_and_ratios(self):
        cfg = small_config(L=4, d=256, H=4, H_kv=2)
        N_layer = 2 * 256 * 64 * 6 + 3 * 256 * 1024
        self.assertEqual(d2l.layer_params(cfg), N_layer)
        par = dict(b=2, b_g=4, t=4, e=8, n_stage=2, T=1000, k=8)
        R = 500.0
        self.assertAlmostEqual(d2l.axis_ratio('tp', cfg, R, **par),
                               4 / 3 * 3 * 256 * 2 / N_layer * R)
        self.assertAlmostEqual(d2l.axis_ratio('pp', cfg, R, **par),
                               256 * 2 / (3 * 2 * N_layer) * R)
        self.assertAlmostEqual(d2l.axis_ratio('dp', cfg, R, **par),
                               4 / 3000 * R * 7 / 8)
        self.assertTrue(math.isnan(d2l.axis_ratio('ep', cfg, R, **par)))
        moe = d2l.HWShape(cfg, num_experts=16, num_experts_per_tok=2,
                          moe_intermediate_size=128, decoder_sparse_step=1,
                          mlp_only_layers=[])
        self.assertEqual(d2l.layer_params(moe),
                         2 * 256 * 64 * 6 + 2 * 3 * 256 * 128)
        self.assertAlmostEqual(d2l.axis_ratio('ep', moe, R, **par),
                               2 * 2 * (7 / 8) / (9 * 128) * R)
        # ratios are linear in the ridge
        self.assertAlmostEqual(d2l.axis_ratio('tp', cfg, 2 * R, **par),
                               2 * d2l.axis_ratio('tp', cfg, R, **par))

    def test_qwen3_8b_layer(self):
        cfg = cached_config(self, 'Qwen/Qwen3-8B')
        self.assertAlmostEqual(d2l.layer_params(cfg) / 1e6, 192.9, places=1)

    def toy_system(self, domain=8, up=600, out=50, chips=8):
        return pd.Series(dict(
            short='toy', chip='H100 SXM', chips=chips, domain_chips=domain,
            up_gbs_dir=up / 2, scaleout_gbs_per_chip=out, p16_tflops=1000.0,
            hbm_gb_per_chip=80.0, has_scaleout=True), name='toy system')

    def test_layout_fit(self):
        cfg = small_config(L=8, d=1024, H=8, H_kv=2)
        cfg['name_or_path'] = 'Qwen/Qwen3-toy'
        sys_ = self.toy_system()
        fit = d2l.layout_fit(cfg, sys_, dict(t=8, e=1, p=2, k=4), 2**20)
        self.assertEqual(list(fit.index), ['tp', 'pp', 'dp'])
        self.assertEqual(list(fit.link), ['scale-up', 'scale-out',
                                          'scale-out'])
        R_up, R_out = 1e15 / 300e9, 1e15 / 50e9
        par = dict(b=2, b_g=2, t=8, e=1, n_stage=2, T=2**20 / 4, k=4)
        self.assertAlmostEqual(fit.ratio['tp'],
                               d2l.axis_ratio('tp', cfg, R_up, **par))
        N = d2l.ServingModel(cfg, d2l.chip_specs().loc['H100 SXM']
                             ).n_params()[0]
        held = N / (8 * d2l.layer_params(cfg))  # embeddings' gradients too
        self.assertAlmostEqual(fit.ratio['dp'], 0.1 * d2l.axis_ratio(
            'dp', cfg, R_out, **par) * held)
        self.assertEqual(list(fit.fit), list(fit.ratio <= 0.15))
        self.assertEqual(fit.attrs['model'], 'toy')
        self.assertAlmostEqual(fit.attrs['mem_gb'], 16 * N / 16 / 1e9)
        # bytes per step: tp moves 8 (t-1)/t T d b per layer and replica
        T_r = 2**20 / 4
        tp_bytes = 8 * 7 / 8 * T_r * 1024 * 2 * 8 / 2  # L / p layers
        self.assertAlmostEqual(fit.gb_per_step['tp'] * 1e9 / tp_bytes, 1.0,
                               places=6)
        # a bigger domain moves pp onto the scale-up link
        big = d2l.layout_fit(cfg, self.toy_system(domain=72), dict(
            t=8, e=1, p=2, k=4), 2**20)
        self.assertEqual(big.link['pp'], 'scale-up')
        # mfu scales P_eff: compute takes 1 / mfu longer, every ratio
        # shrinks by mfu
        half = d2l.layout_fit(cfg, sys_, dict(t=8, e=1, p=2, k=4), 2**20,
                              mfu=0.5)
        for axis in fit.index:
            self.assertAlmostEqual(half.ratio[axis], fit.ratio[axis] / 2)

    def test_layout_fit_moe(self):
        """dp counts the gradients a chip holds; ep's bytes come from the
        expert FLOPs its ratio is measured against."""
        cfg = small_config(L=8, d=1024, H=8, H_kv=2)
        moe = d2l.HWShape(cfg, num_experts=16, num_experts_per_tok=2,
                          moe_intermediate_size=256, decoder_sparse_step=1,
                          mlp_only_layers=[], name_or_path='Qwen/Qwen3-moe')
        sys_ = self.toy_system(domain=8)
        layout, T = dict(t=1, e=8, p=2, k=16), 2**22
        fit = d2l.layout_fit(moe, sys_, layout, T)
        m = d2l.ServingModel(moe, d2l.chip_specs().loc['H100 SXM'])
        N = m.n_params()[0]
        N_exp = 8 * 16 * 3 * 1024 * 256
        held = (N - N_exp) / 2 + N_exp / 16  # per chip, t p = 2, e = 8
        par = dict(b=2, b_g=2, t=1, e=8, n_stage=2, T=T / 16, k=16)
        R_out = 1e15 / 50e9
        self.assertAlmostEqual(fit.ratio['dp'], 0.1 * d2l.axis_ratio(
            'dp', moe, R_out, **par) * held * 2 / (8 * d2l.layer_params(moe)))
        # dp bytes per chip: 2 (k-1)/k of the held gradients in bf16
        self.assertAlmostEqual(fit.gb_per_step['dp'] * 1e9,
                               2 * 15 / 16 * held * 2, delta=1e-6 * held)
        # ep bytes per chip: 4 all-to-alls of K d b (1 - 1/e) per token and
        # MoE layer, T / k tokens, L / p layers
        ep_bytes = 4 * 2 * 1024 * 2 * (7 / 8) * (T / 16) * 8 / 2
        self.assertAlmostEqual(fit.gb_per_step['ep'] * 1e9 / ep_bytes, 1.0,
                               places=6)
        self.assertAlmostEqual(d2l.expert_flop_share(moe),
                               2 * 3 * 1024 * 256 / d2l.layer_params(moe))

    def test_layout_for_and_display(self):
        sy = d2l.systems_table()
        cfg = small_config(L=8, d=1024, H=8, H_kv=2)
        cfg['name_or_path'] = 'Qwen/Qwen3-toy'
        box = sy.loc['this box (4 x RTX 4090)']
        T = 2**22
        # the box's link fits no tensor parallelism: t = 1, its 4 chips
        self.assertEqual(d2l.layout_for(cfg, box, dict(t=8, e=1, p=4), 1024,
                                        T), dict(t=1, e=1, p=4, k=1))
        node = sy.loc['8 x A100 SXM 40 GB']
        moe = d2l.HWShape(cfg, num_experts=16, num_experts_per_tok=2,
                          moe_intermediate_size=256, decoder_sparse_step=1,
                          mlp_only_layers=[])
        # with eps = inf every degree fits: the caps decide
        self.assertEqual(d2l.layout_for(moe, node, dict(t=1, e=8, p=4), 1024,
                                        T, eps=math.inf),
                         dict(t=1, e=8, p=4, k=256))
        self.assertEqual(d2l.layout_for(cfg, node, dict(t=8, e=1, p=4), 1024,
                                        T, eps=math.inf)['t'], 8)
        # the chosen t is the largest power of two whose tp ratio fits
        lay = d2l.layout_for(cfg, node, dict(t=8, e=1, p=4), 1024, T)
        fit = d2l.layout_fit(cfg, node, lay, T)
        if lay['t'] > 1:
            self.assertTrue(fit.fit['tp'])
        if lay['t'] < 8:
            up = dict(lay, t=2 * lay['t'], k=1024 // (2 * lay['t'] * 4))
            self.assertFalse(d2l.layout_fit(cfg, node, up, T).fit['tp'])
        # Trn3 states no rates: stated_rates drops it and says why
        lines, kept = lines_of(self, d2l.stated_rates, sy.loc[[
            '8 x A100 SXM 40 GB', 'Trn3 UltraServer',
            'this box (4 x RTX 4090)']])
        self.assertEqual(list(kept.short), ['A100 node', 'box'])
        self.assertEqual(lines, ['no stated rate, left out: Trn3 (bf16, '
                                 'scale-up, scale-out)'])
        fits = [d2l.layout_fit(cfg, s, d2l.layout_for(
            cfg, s, dict(t=8, e=1, p=4), 1024, T), T)
            for _, s in kept.iterrows()]
        lines, _ = lines_of(self, d2l.show_layout_fit, fits)
        self.assertLessEqual(len(lines), 15)
        self.assertRegex(lines[1], r'A100 node .*dp\d+')  # degrees shown
        lines, _ = lines_of(self, d2l.show_epsilon, fits, (0.0, math.inf))
        n = sum(len(f.dropna(subset=['ratio'])) for f in fits)
        n_fit = sum(int(f.fit.sum()) for f in fits)
        # eps = 0 flips every fit, eps = inf every miss: all are named
        text = ' '.join(lines)
        self.assertIn(f'eps = 0: {n_fit} changed', text)
        self.assertIn(f'eps = inf: {n - n_fit} changed', text)
        self.assertEqual(text.count(';'),  # every flip named, none hidden
                         max(n_fit - 1, 0) + max(n - n_fit - 1, 0))

    def test_diloco(self):
        self.assertEqual(d2l.diloco_wan_bandwidth(8e9, 2, 100, 2.0), 8e7)

    def test_perftest(self):
        bw = d2l.parse_perftest(PERFTEST_BW, 'bw')
        self.assertEqual(list(bw.bytes), [2, 1024, 1048576, 2097152])
        self.assertEqual(bw.bw_avg_gbits.iloc[-1], 196.44)
        lat = d2l.parse_perftest(PERFTEST_LAT, 'lat')
        self.assertEqual(list(lat.t_typical_us), [1.06, 1.14, 2.05])
        with self.assertRaises(ValueError):
            d2l.parse_perftest(PERFTEST_LAT, 'bw')
        fit = d2l.fit_perftest(dict(perftest={'mlx5_0': dict(
            bw=PERFTEST_BW, lat=PERFTEST_LAT)}))
        r = fit.loc[('ib_write', 1, 'mlx5_0')]
        self.assertAlmostEqual(r.alpha, 1.10e-6)
        self.assertAlmostEqual(r.beta, 196.38e9 / 8)
        lines, _ = lines_of(self, d2l.show_fits, fit)
        self.assertEqual(len(lines), 2)
        pending = d2l.fit_perftest(d2l.CampaignPending(
            'cluster-nccl', 'C-T4', 'two 8 x A100 nodes with RDMA'))
        lines, _ = lines_of(self, d2l.show_fits, pending)
        self.assertEqual(lines, ['two 8 x A100 nodes with RDMA: campaign '
                                 'C-T4 not run yet'])

    def loopback(self, backend, ranks, alpha, B):
        frames = [synthetic_sweep('all_reduce', k, 'loopback', alpha * k, B,
                                  seed=k) for k in ranks]
        t = pd.concat(frames, ignore_index=True)
        t.attrs.update(backend=backend, env='', alpha_below=2**16,
                       plateau_from=2**22,
                       info={2: dict(version='2.28.9',
                                     transports=['NET/Socket/0'])})
        return t

    def test_loopback_displays(self):
        floor = self.loopback('gloo', (2, 4), 300e-6, 1.4e9)
        sock = self.loopback('nccl', (2,), 40e-6, 1.0e9)
        lines, _ = lines_of(self, d2l._show_loopback, floor)
        self.assertEqual(len(lines), 2)
        self.assertIn('gloo over TCP, 2 ranks: alpha 600 us', lines[0])
        lines, _ = lines_of(self, d2l._show_loopback, sock)
        self.assertIn('NCCL over sockets, 2 GPUs', lines[0])
        # no transport recorded is not evidence of sockets
        bare = sock.copy()
        bare.attrs = dict(sock.attrs, info={2: dict(version='2.28.9',
                                                    transports=[])})
        self.assertEqual(d2l._loopback_name(bare),
                         'NCCL (transport not recorded)')
        tcp = self.pingpong(40e-6, 2e9)
        lines, _ = lines_of(self, d2l._show_loopback, tcp)
        self.assertEqual(len(lines), 1)
        self.assertIn('TCP ping-pong, 2 processes: round trip 40 us', lines[0])
        quoted = d2l.quoted_latencies()
        self.assertTrue(quoted.source_url.str.contains('https://').all())
        lines, ladder = lines_of(self, d2l.transport_ladder, tcp, floor, sock,
                                 dict(BOX_AB), quoted)
        self.assertEqual(len(ladder), 7)
        self.assertLessEqual(len(lines), 8)
        self.assertAlmostEqual(ladder.alpha_us['NCCL shm, 4 GPUs'], 30)
        self.assertAlmostEqual(ladder.msg_us['NCCL shm, 4 GPUs'], 5)
        self.assertAlmostEqual(ladder.msg_us['RDMA, one NDR port'], 2.5)
        self.assertAlmostEqual(ladder.alpha_us['TCP, 2 processes'], 40,
                               delta=2)
        self.assertAlmostEqual(ladder.beta_gbs['TCP, 2 processes'], 2.0,
                               delta=0.1)
        # gloo on 4 ranks: alpha 1,200 us over 6 ring steps
        self.assertAlmostEqual(ladder.msg_us['gloo, 4 ranks'],
                               ladder.alpha_us['gloo, 4 ranks'] / 6)
        self.assertTrue(all(len(c) <= 4 for c in ' '.join(
            lines[1:]).replace(',', '').split() if c.replace('.', '')
            .isdigit()))  # two significant digits
        betas = d2l.link_betas(d2l.systems_table(), dict(BOX_AB))
        self.assertEqual(betas, {'NVLink (A100)': 300e9,
                                 'NIC (A100, HDR)': 25e9,
                                 'box link': 2.4e9})

    def pingpong(self, alpha, beta, procs=()):
        """A TCP ping-pong table: round trip alpha + 2 S / beta."""
        s = 64 * 2.0 ** np.arange(19)
        t = pd.DataFrame(dict(size=s.astype(int),
                              rt_us=(alpha + 2 * s / beta) * 1e6))
        t.attrs.update(backend='tcp', env='', alpha_below=2**16,
                       plateau_from=2**22, cpu='toy CPU',
                       proc_us=[list(t.rt_us * f) for f in procs])
        return t

    def test_host_live(self):
        """The CPU-only live line needs no GPU and is keyed to the CPU."""
        tcp = self.pingpong(40e-6, 2e9)
        live = lambda b, e, size: float(tcp.rt_us[tcp['size'] == size].iloc[0])
        with mock.patch.object(d2l, '_loopback_live_us', side_effect=live), \
                mock.patch.object(torch.cuda, 'is_available',
                                  return_value=False):
            with mock.patch.object(d2l, '_cpu_model',
                                   return_value='toy CPU'):
                lines, _ = lines_of(self, d2l._host_live, tcp)
            self.assertEqual(lines, ['TCP live check (same CPU): 3 sizes on '
                                     '2 ranks, largest difference 0%'])
            with mock.patch.object(d2l, '_cpu_model',
                                   return_value='other CPU'):
                lines, _ = lines_of(self, d2l._host_live, tcp)
            self.assertTrue(lines[0].startswith(
                'TCP live values (other CPU): 1 KiB'))

    def test_loopback_live_line(self):
        floor = self.loopback('gloo', (2, 4), 300e-6, 1.4e9)
        sizes = d2l._HW_NET_LIVE['sizes']
        floor = floor[floor['size'].isin(sizes) | (floor.g == 4)]
        if not set(sizes) <= set(floor['size']):
            self.skipTest('synthetic sweep lacks the live sizes')
        with mock.patch.object(d2l, '_loopback_live_us',
                               side_effect=lambda b, e, size: float(
                                   floor.oop_us[(floor.g == 2) & (
                                       floor['size'] == size)].iloc[0])):
            text = d2l._loopback_live(floor, True)
        self.assertEqual(text, '3 sizes on 2 ranks, largest difference 0%')
        with mock.patch.object(d2l, '_loopback_live_us', return_value=1e9):
            self.assertTrue(d2l._loopback_live(floor, True).endswith(
                '(differs)'))

    def test_regen_finish(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i, scale in enumerate((1.0, 1.1)):
                os.makedirs(os.path.join(tmp, f'proc{i}'), exist_ok=True)
                for entry in ('gloo', 'nccl'):
                    t = self.loopback(entry, (2,), 100e-6 * scale, 1e9)
                    d2l._hw_write(os.path.join(tmp, f'proc{i}',
                                               f'{entry}.json.gz'), t, {})
                d2l._hw_write(os.path.join(tmp, f'proc{i}', 'tcp.json.gz'),
                              self.pingpong(40e-6 * scale, 2e9), {})
            for entry in ('gloo', 'nccl'):
                d2l._hw_write(os.path.join(tmp, f'{entry}.json.gz'),
                              self.loopback(entry, (2,), 100e-6, 1e9), {})
            d2l._hw_write(os.path.join(tmp, 'tcp.json.gz'),
                          self.pingpong(40e-6, 2e9), {})
            d2l.regen_net_finish(tmp)
            t, _ = d2l._hw_read(os.path.join(tmp, 'gloo.json.gz'))
            self.assertEqual(len(t.attrs['proc_us']), 2)
            ab = d2l.fit_collectives(t)
            self.assertFalse(math.isnan(ab.alpha_lo.iloc[0]))
            t, _ = d2l._hw_read(os.path.join(tmp, 'tcp.json.gz'))
            x = d2l._fit_tcp(t)
            self.assertLess(x['alpha_lo'], x['alpha_hi'])


NCCL_INFO_SAMPLE = """\
box:1:1 [0] NCCL INFO NCCL version 2.28.9+cuda12.9
box:1:1 [0] NCCL INFO comm 0x5 rank 0 nRanks 4 nNodes 1 localRanks 4 localRank 0 MNNVL 0
box:1:1 [0] NCCL INFO Channel 00/02 :    0   1   2   3
box:1:1 [0] NCCL INFO Channel 01/02 :    0   1   2   3
box:1:1 [0] NCCL INFO Trees [0] 1/-1/-1->0->-1 [1] 1/-1/-1->0->-1
box:1:1 [1] NCCL INFO Trees [0] 2/-1/-1->1->0 [1] 2/-1/-1->1->0
box:1:1 [2] NCCL INFO Trees [0] 3/-1/-1->2->1 [1] 3/-1/-1->2->1
box:1:1 [3] NCCL INFO Trees [0] -1/-1/-1->3->2 [1] -1/-1/-1->3->2
box:1:1 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[1] via SHM/direct/direct
box:1:1 [3] NCCL INFO Channel 00/0 : 3[3] -> 0[0] via SHM/direct/direct
box:1:1 [0] NCCL INFO 2 coll channels, 2 collnet channels, 0 nvls channels
box:1:1 [0] NCCL INFO Algorithm   |   Tree      |   Ring
box:1:1 [0] NCCL INFO Protocol    | LL | LL128 | Simple | LL | LL128 | Simple
box:1:1 [0] NCCL INFO AllReduce   | 20.1/ 4.0 | 0.0/ 0.0 | 60.0/ 9.5 | 15.2/ 3.1 | 0.0/ 0.0 | 40.3/ 8.7
box:1:1 [0] NCCL INFO AllReduce: 8 Bytes -> Algo 1 proto 0 time 15.2
box:1:1 [0] NCCL INFO AllReduce: 65536 Bytes -> Algo RING proto LL time 20.0
box:1:1 [0] NCCL INFO AllReduce: 268435456 Bytes -> Algo 1 proto 2 time 1.2
"""


def synthetic_pairs(params, dropped=None, g=4, seed=0, procs=3):
    """A forced-pair table (nccl_algo_proto_sweeps' layout) of
    t = alpha + phi S / beta per pair ({name: (alpha, beta busbw)}), with
    per-process times in attrs['proc_us']; dropped: {name: reason}."""
    dropped, phi = dropped or {}, d2l.nccl_busbw_factor('all_reduce', g)
    frames = []
    for i, (name, (alpha, beta)) in enumerate(params.items()):
        x = synthetic_sweep('all_reduce', g, 'all', alpha, beta / phi,
                            seed=seed + i, noise=0.002)
        x = x.assign(algo=name, proto='default' if name == 'default'
                     else name.split('/')[1], status='', used='')
        frames.append(x)
    for name, reason in dropped.items():
        x = frames[0].copy().assign(algo=name, proto=name.split('/')[1],
                                    status=reason)
        x[['oop_us', 'oop_algbw', 'oop_busbw']] = np.nan
        frames.append(x)
    out = pd.concat(frames, ignore_index=True)
    rng = np.random.default_rng(seed)
    out.attrs.update(dropped=dict(dropped), alpha_below=2**16,
                     plateau_from=2**22, proc_us=[
                         (out.oop_us * (1 + 0.003 * rng.standard_normal(
                             len(out)))).tolist() for _ in range(procs)])
    return out


def tuner_case(procs=3):
    """A forced-pair table in which NCCL's default is its own choice
    (Ring/LL up to 1 MiB, then Ring/Simple, as its `used` column says) run
    unforced, so forcing costs nothing; Ring/Simple is faster from 75 KB;
    Ring/LL128 is the fastest pair but NCCL's tuner disables it."""
    params = {'default': (15e-6, 3e9), 'Ring/LL': (15e-6, 3e9),
              'Ring/LL128': (5e-6, 20e9), 'Ring/Simple': (40e-6, 9e9),
              'Tree/LL': (18e-6, 2.5e9)}
    pairs = synthetic_pairs(params, procs=procs)
    own = np.where(pairs['size'] <= 2**20, 'Ring/LL', 'Ring/Simple')
    pairs['used'] = np.where(pairs.algo == 'default', own, '')
    procs_us = np.asarray(pairs.attrs['proc_us'])
    for choice in ('Ring/LL', 'Ring/Simple'):  # the default runs its choice
        to = ((pairs.algo == 'default') & (own == choice)).to_numpy()
        src = (pairs.algo == choice).to_numpy() & np.isin(
            pairs['size'], pairs['size'][to])
        pairs.loc[to, 'oop_us'] = pairs.oop_us[src].to_numpy()
        procs_us[:, to] = procs_us[:, src]
    pairs.attrs['proc_us'] = procs_us.tolist()
    pairs.attrs['tuner'] = {'Ring/LL': (12.6, 8.0), 'Ring/LL128': (29.0, 0.0),
                            'Ring/Simple': (42.6, 16.0),
                            'Tree/LL': (12.8, 2.9)}
    return pairs


class TestCollectives(unittest.TestCase):
    """Section 22.7: the closed forms against the simulator, NCCL's double
    binary tree, the INFO digest, the forced-pair fits, the crossover and
    the tuner check, the symptoms, the alpha_beta.json records."""

    def test_closed_forms(self):
        G, alpha, beta, d = 16, 10e-6, 25e9, 4
        S = G * alpha * beta * (G - 1 - d)  # D1
        self.assertAlmostEqual(S / 1e6, 44, places=0)
        self.assertAlmostEqual(
            d2l.ring_allreduce_time(S, G, alpha, beta),
            d2l.tree_allreduce_time(S, G, 1, alpha, alpha, beta))
        # D2: the hierarchical speedup tends to g
        flat = d2l.ring_allreduce_time(1e9, 8 * 10**4, alpha, beta)
        hier = d2l.hier_allreduce_time(1e9, 10**4, 8, alpha, 1e4 * beta,
                                       alpha, beta, rails=8)
        self.assertAlmostEqual(flat / hier, 8, delta=0.1)
        # D3: in-network halves the ring's bandwidth term asymptotically
        self.assertAlmostEqual(
            d2l.ring_allreduce_time(1e12, 1000, 0, beta)
            / d2l.sharp_allreduce_time(1e12, 0, beta), 2, delta=0.01)
        # the all-to-all on one node is the intra term alone
        self.assertEqual(d2l.a2a_inter_node_time(1e6, 1, 8, 1, 1, 0, 1e9),
                         7 * 1e6 / 8 / 1e9)

    def test_double_binary_tree(self):
        for G in (2, 3, 4, 7, 8, 16, 33):
            t0, t1 = d2l.double_binary_tree(G)
            for t in (t0, t1):
                self.assertEqual(sorted(t), list(range(G)))
                self.assertEqual(sum(p is None for p in t.values()), 1)
                kids = collections_counter(t)
                self.assertLessEqual(max(kids.values(), default=0), 2)
                for r in t:  # every rank reaches the root
                    seen, x = set(), r
                    while t[x] is not None:
                        self.assertNotIn(x, seen)
                        seen.add(x)
                        x = t[x]
            leaf0 = {r for r in t0 if r not in collections_counter(t0)}
            leaf1 = {r for r in t1 if r not in collections_counter(t1)}
            self.assertEqual({r for r in range(G) if r % 2}, leaf0 - {0})
            if G % 2 == 0:  # a leaf of one tree is interior in the other
                self.assertFalse(leaf0 & leaf1)
            # each rank sends at most S / 2 up each tree: 2 S in all
            sends = [(t0[r] is not None) + (t1[r] is not None)
                     + collections_counter(t0)[r] + collections_counter(t1)[r]
                     for r in range(G)]
            self.assertLessEqual(max(sends) / 2, 2)

    def test_simulator(self):
        alpha, beta = 10e-6, 25e9
        for G in (4, 8):
            topo = d2l.sim_topology(G, alpha, beta)
            d = math.ceil(math.log2(G))
            S_star = G * alpha * beta * (G - 1 - d)
            for f in (0.01, 1, 100):
                S = f * S_star
                ring = d2l.sim_collective(topo, 'ring', S, 1)
                self.assertAlmostEqual(
                    ring / d2l.ring_allreduce_time(S, G, alpha, beta), 1,
                    places=9)
                tree = d2l.sim_collective(topo, 'tree', S, 200)
                model = d2l.tree_allreduce_time(S, G, 1, alpha, alpha, beta)
                self.assertLess(tree / model, 1.02)  # an upper bound
                if f == 100:
                    self.assertAlmostEqual(tree / model, 1, delta=0.02)
        with self.assertRaises(ValueError):
            d2l.sim_topology(1, alpha, beta)

    def test_sim_summary(self):
        """C1's sentence from the committed simulator table (the cell's
        output, deterministic): ring exact, the tree at most 0.3% above the
        closed form, within 0.5% far from S_b, 24.9% below it at G = 4."""
        cols = ['0.01 S_b (%)', '1 S_b (%)', '100 S_b (%)']
        tree = {4: (-0.5, -24.9, 0.0), 8: (-0.5, -8.2, 0.0),
                16: (-0.5, -12.3, 0.3), 32: (-0.5, -4.7, 0.3)}
        err = pd.DataFrame(
            [[G, a] + list(v if a == 'tree' else (0.0,) * 3)
             for G, v in tree.items() for a in ('ring', 'tree')],
            columns=['G', 'algo'] + cols)
        lines, _ = lines_of(self, d2l.sim_summary, err)
        self.assertEqual(' '.join(l.strip() for l in lines),
                         'ring: exact; tree: exceeds the closed form by at '
                         'most 0.3%, within 0.5% of it far from S_b, up to '
                         '24.9% below it at S_b (G = 4)')
        err.loc[err.algo == 'tree', cols[-1]] = -0.1
        err.loc[err.algo == 'ring', cols[1]] = 0.2
        lines, _ = lines_of(self, d2l.sim_summary, err)
        text = ' '.join(l.strip() for l in lines)
        self.assertIn('ring: within 0.2%', text)
        self.assertIn('never above the closed form', text)

    def test_parse_info(self):
        d = d2l._parse_info(NCCL_INFO_SAMPLE)
        self.assertEqual((d['version'], d['n_ranks'], d['n_nodes']),
                         ('2.28.9', 4, 1))
        self.assertEqual(d['transports'], ['SHM/direct/direct'])
        self.assertEqual(d['channels'], 2)
        self.assertEqual(d['rings'][1], [0, 1, 2, 3])
        self.assertEqual(d['trees'][0], {0: -1, 1: 0, 2: 1, 3: 2})
        self.assertEqual(d['tuner'][('Ring', 'Simple')], (40.3, 8.7))
        self.assertEqual(d['tuner'][('Tree', 'LL128')], (0.0, 0.0))
        self.assertEqual(d['choices'], {8: 'Ring/LL', 65536: 'Ring/LL',
                                        268435456: 'Ring/Simple'})
        self.assertEqual(d2l._ranges(d['choices']),
                         'Ring/LL up to 64 KiB, Ring/Simple from 256 MiB')
        self.assertEqual(d2l._chain(d['trees'][0]),
                         '3 -> 2 -> 1 -> 0 (a chain to the root)')
        lines, _ = lines_of(self, d2l._show_info, d)
        self.assertLessEqual(len(lines), 8)
        text = '\n'.join(lines)
        self.assertIn('disabled: Ring/LL128, Tree/LL128', text)
        self.assertIn('SHM/direct/direct', text)
        # the digest survives storage (string keys) and still displays
        stored = d2l._hw_decode(json.loads(json.dumps(d2l._hw_encode(dict(
            d, tuner={f'{a}/{p}': v for (a, p), v in d['tuner'].items()})))))
        self.assertEqual(lines_of(self, d2l._show_info, stored)[0], lines)
        lines, _ = lines_of(self, d2l._show_info, dict(version=None,
                                                       label='x pending'))
        self.assertEqual(lines, ['x pending'])

    def test_parse_info_box(self):
        """The box's own INFO lines (NCCL 2.28.9, 4 x RTX 4090, SHM)."""
        d = d2l._parse_info(sample('box_nccl_info_g4.txt'))
        self.assertEqual((d['version'], d['n_ranks'], d['n_nodes'],
                          d['channels']), ('2.28.9', 4, 1, 4))
        self.assertEqual(d['transports'], ['SHM/direct/direct'])
        self.assertEqual(d['rings'], {c: [0, 1, 2, 3] for c in range(4)})
        self.assertEqual(d['trees'][3], {0: -1, 1: 0, 2: 1, 3: 2})
        self.assertEqual(d['tuner'][('Ring', 'Simple')], (42.6, 16.0))
        self.assertEqual(d['tuner'][('Tree', 'LL')], (12.8, 2.9))
        self.assertEqual(d['tuner'][('Ring', 'LL128')][1], 0.0)  # disabled
        self.assertEqual(d['tuner'][('NVLS', 'Simple')], (0.0, 0.0))
        self.assertEqual(d['tuner'][('PAT', 'LL')], (0.0, 0.0))
        self.assertEqual(len(d['tuner']), 21)  # 7 algorithms x 3
        self.assertEqual(d['choices'][8], 'Ring/LL')
        self.assertEqual(d['choices'][2**27], 'Ring/Simple')
        lines, _ = lines_of(self, d2l._show_info, d)
        self.assertLessEqual(len(lines), 8)
        self.assertIn('disabled: Ring/LL128, Tree/LL128', '\n'.join(lines))

    def params(self):
        return {'default': (20e-6, 9e9), 'Ring/LL': (15e-6, 3e9),
                'Ring/Simple': (40e-6, 9e9), 'Tree/LL': (18e-6, 2.5e9)}

    def test_fit_pairs(self):
        pairs = synthetic_pairs(self.params(), {'Ring/LL128': 'refused'})
        fits = d2l.fit_pairs(pairs)
        self.assertEqual(len(fits), 4)
        self.assertNotIn(('Ring', 'LL128'), fits.index)
        for name, (alpha, beta) in self.params().items():
            key = ('default',) * 2 if name == 'default' else tuple(
                name.split('/'))
            self.assertAlmostEqual(fits.loc[key].alpha / alpha, 1, delta=0.05)
            self.assertAlmostEqual(fits.loc[key].beta / beta, 1, delta=0.05)
            self.assertTrue(fits.loc[key].beta_lo < fits.loc[key].beta_hi)
        x = fits.attrs['crossover']
        phi = d2l.nccl_busbw_factor('all_reduce', 4)
        exact = 25e-6 / (phi * (1 / 3e9 - 1 / 9e9))
        self.assertAlmostEqual(x['predicted'] / exact, 1, delta=0.1)
        self.assertAlmostEqual(x['measured'] / exact, 1, delta=0.15)
        self.assertEqual(x['why'], '')
        self.assertEqual(x['window'], (2**14, 2**20))
        for lo, hi in x['resid'].values():  # 0.2% noise: the fits are exact
            self.assertLess(max(abs(lo), abs(hi)), 0.02)
        S = np.array([2**14, 2**20])  # the two exact lines against each other
        exact_gap = (40e-6 + phi * S / 9e9) / (15e-6 + phi * S / 3e9) - 1
        np.testing.assert_allclose(x['gap'], exact_gap, atol=0.03)
        lines, _ = lines_of(self, d2l.show_pair_fits, fits)
        self.assertLessEqual(len(lines), 10)
        text = ' '.join(l.strip() for l in lines[5:])
        self.assertIn('Ring/Simple overtakes Ring/LL', text)
        self.assertIn('fit against measured, 16 KiB to 1 MiB: Ring/LL', text)
        self.assertIn("Ring/Simple's fit against Ring/LL's:", text)
        self.assertIn('beta 95%', lines[0])
        self.assertAlmostEqual(d2l.pair_time(pairs, 'Ring/LL', 8),
                               15e-6, delta=1e-6)
        lines, _ = lines_of(self, d2l._show_pairs, pairs)
        self.assertEqual(lines, ['dropped (refused): Ring/LL128'])
        d2l.plot_algo_crossover(pairs)  # relative to the default
        ys = [l.get_ydata() for l in d2l.plt.gca().get_lines()[:4]]
        self.assertTrue(np.allclose(ys[3], 1))  # the default's own line
        self.assertLess(ys[0][0], 1)  # Ring/LL beats the default at 8 B
        d2l.plt.close('all')
        pending = d2l.fit_pairs(d2l.CampaignPending(
            'cluster-nccl', 'C-T4', 'two 8 x A100 nodes with RDMA'))
        self.assertTrue(pending.empty)
        self.assertIn('C-T4 not run yet', pending.attrs['status'])

    def test_fit_pairs_no_crossover(self):
        """LL's fitted bandwidth at or above Simple's: no crossover is
        predicted (never a negative size), and the line says why."""
        params = dict(self.params(), **{'Ring/LL': (15e-6, 9.5e9)})
        fits = d2l.fit_pairs(synthetic_pairs(params))
        x = fits.attrs['crossover']
        self.assertTrue(np.isnan(x['predicted']))
        self.assertIn('bandwidth is not below', x['why'])
        lines, _ = lines_of(self, d2l.show_pair_fits, fits)
        self.assertIn('no crossover predicted', '\n'.join(lines))
        text = ' '.join(l.strip() for l in lines)
        self.assertNotIn('-', text.split('predicted')[0][-8:])
        self.assertIn('fit against measured', text)  # residuals still shown

    def test_off_pairs(self):
        """A pair NCCL's tuner disables is marked in the fits and dashed in
        the plot."""
        pairs = tuner_case()
        self.assertEqual(d2l._tuner_off(pairs), {'Ring/LL128'})
        fits = d2l.fit_pairs(pairs)
        lines, _ = lines_of(self, d2l.show_pair_fits, fits)
        self.assertIn('LL128 (off)', '\n'.join(lines))
        d2l.plot_algo_crossover(pairs)
        ax = d2l.plt.gca()
        names = [t.get_text() for t in ax.get_legend().get_texts()]
        styles = dict(zip(names, [l.get_linestyle() for l in ax.get_lines()]))
        d2l.plt.close('all')
        self.assertEqual([n for n in names if 'LL128' in n],
                         ['Ring/LL128 (off)'])
        self.assertEqual(styles['Ring/LL128 (off)'], '--')
        self.assertEqual(styles['Ring/LL'], '-')

    def test_protocol_order(self):
        pairs = synthetic_pairs(self.params())
        lines, units = lines_of(self, d2l.protocol_order, pairs)
        self.assertLessEqual(len(lines), 4)
        self.assertGreater(units[0], 3)  # 15 us against 40 us at 8 B
        lines = ' '.join(lines).split('256 MiB:')
        self.assertIn('8 B: Ring/LL', lines[0])
        self.assertIn('Ring/LL faster', lines[0])
        self.assertIn('Ring/Simple faster', lines[1])
        # a 1% gap with one process (noise floor 2%) is within noise
        close = dict(self.params(), **{'Ring/LL': (15e-6, 8.9e9)})
        lines, units = lines_of(self, d2l.protocol_order,
                                synthetic_pairs(close, procs=1))
        self.assertIn('within noise', ' '.join(lines).split('256 MiB:')[1])
        self.assertLess(abs(units[1]), 3)

    def test_pair_check(self):
        """A forced pair whose INFO run reports no choice is unverified; one
        that reports another pair is ignored; the default's check returns
        the tuner's table."""
        run = d2l.HWRun(['all_reduce_perf'], 0, '', '', 0.1, None)
        spec = d2l._HW_COLL['box']
        cases = [('', 'unverified'),
                 ('NCCL INFO AllReduce: 8 Bytes -> Algo RING proto SIMPLE',
                  'ignored: ran Ring/Simple'),
                 ('NCCL INFO AllReduce: 8 Bytes -> Algo RING proto LL', '')]
        for text, reason in cases:
            with mock.patch.object(d2l, '_info_run',
                                   return_value=(run, text)), \
                    mock.patch.object(d2l, 'parse_nccl_tests',
                                      return_value=pd.DataFrame(
                                          dict(oop_wrong=[0], ip_wrong=[0]))):
                got, used, _ = d2l._pair_check((0, 1), 'Ring', 'LL', spec)
            self.assertEqual(got, reason)
        with mock.patch.object(d2l, '_info_run',
                               return_value=(run, NCCL_INFO_SAMPLE)), \
                mock.patch.object(d2l, 'parse_nccl_tests',
                                  return_value=pd.DataFrame(
                                      dict(oop_wrong=[0], ip_wrong=[0]))):
            got, used, tuner = d2l._pair_check((0, 1), 'default', 'default',
                                               spec)
        self.assertEqual(got, '')
        self.assertEqual(tuner[('Ring', 'Simple')], (40.3, 8.7))

    def test_regen_prepare(self):
        """A pair dropped in one process is dropped in all before the
        processes are combined, and the combination then succeeds."""
        a = synthetic_pairs(self.params(), procs=1)
        b = synthetic_pairs(self.params(), {'Tree/LL': 'timed out'},
                            seed=1, procs=1)
        b = b[b.algo != 'Tree/LL'].copy()  # the dropped copy, as computed
        drop = a[a.algo == 'Tree/LL'].copy()
        drop[[c for c in drop.columns if c not in (
            'size', 'op', 'g', 'path', 'algo', 'proto', 'env')]] = np.nan
        drop['status'], drop['used'] = 'timed out', ''
        b = pd.concat([b, drop]).loc[a.index]
        b.attrs.update(a.attrs, dropped={'Tree/LL': 'timed out'})
        a.attrs['dropped'] = {}
        with tempfile.TemporaryDirectory() as tmp:
            for i, t in enumerate((a, b)):
                os.makedirs(os.path.join(tmp, f'proc{i}'))
                d2l._hw_write(os.path.join(tmp, f'proc{i}', 'pairs.json.gz'),
                              t, dict(config={}))
            with self.assertRaises(ValueError):  # without the hook
                d2l._hw_combine([d2l._hw_read(os.path.join(
                    tmp, f'proc{i}', 'pairs.json.gz'))[0] for i in (0, 1)])
            d2l.regen_coll_prepare(tmp, 2)
            parts = [d2l._hw_read(os.path.join(
                tmp, f'proc{i}', 'pairs.json.gz'))[0] for i in (0, 1)]
        out = d2l._hw_combine(parts)
        tree = out[out.algo == 'Tree/LL']
        self.assertTrue(tree.oop_us.isna().all())
        self.assertEqual(set(tree.status), {'timed out in 1 of 2 runs'})
        self.assertEqual(out.attrs['dropped'],
                         {'Tree/LL': 'timed out in 1 of 2 runs'})
        self.assertNotIn('Tree/LL', d2l._kept(out))

    def test_pairs_live(self):
        """The forced-pair live line: compared on four cards, each pair's
        bus bandwidth at 256 MiB on two (uncompared)."""
        pairs = synthetic_pairs(self.params())
        def sweep(gpus, algo, proto, spec, timeout_s=None, sizes=None):
            name = d2l._pair_name(algo, proto)
            x = pairs[pairs.algo == name].reset_index(drop=True)
            if len(gpus) == 2:  # twice as fast
                x = x.assign(oop_us=x.oop_us / 2)
            x.attrs['refused'] = None
            return x
        with mock.patch.object(d2l, '_pair_sweep', side_effect=sweep):
            with mock.patch.object(torch.cuda, 'device_count',
                                   return_value=4):
                line = d2l._pairs_live(pairs, True)
            self.assertIn('9 points, largest difference 0%', line)
            with mock.patch.object(torch.cuda, 'device_count',
                                   return_value=2):
                line = d2l._pairs_live(pairs, True)
        head, values = line.split('\n')
        self.assertEqual(head, '2 GPUs, busbw at 256 MiB (GB/s):')
        self.assertIn('Ring/LL 4.0', values)  # 2 S / (t_4 / 2), phi_4 = 1.5
        self.assertLessEqual(len(values), 79)
        with mock.patch.object(torch.cuda, 'device_count', return_value=1):
            self.assertIn('needs two GPUs', d2l._pairs_live(pairs, True))

    def test_crossing(self):
        s = 2.0 ** np.arange(10)
        x = d2l._crossing(s, 1 + s, 10 + s / 2)  # equal at s = 18
        self.assertAlmostEqual(x, 18, delta=3)
        self.assertTrue(np.isnan(d2l._crossing(s, s, 2 * s)))

    def test_tuner_regret(self):
        """Forced against forced: NCCL keeps Ring/LL up to 1 MiB, but
        Ring/Simple is faster from 75 KB; Ring/LL128, faster still, is off
        in the tuner and never counts."""
        lines, out = lines_of(self, d2l.tuner_regret, tuner_case())
        self.assertLessEqual(len(lines), 9)
        self.assertLess(out.cost.abs().max(), 0.01)  # default = own choice
        self.assertNotIn('Ring/LL128', set(out.best))
        mid = out[out['size'] == 2**19].iloc[0]
        self.assertEqual((mid.own, mid.best), ('Ring/LL', 'Ring/Simple'))
        self.assertGreater(mid.units, 3)
        small = out[out['size'] == 8].iloc[0]
        self.assertLess(small.units, 0)  # Ring/LL is right at 8 B
        text = ' '.join(line.strip() for line in lines)
        self.assertIn('cost of forcing', lines[0])
        self.assertIn('Ring/Simple', lines[1])
        self.assertIn('faster than Ring/LL at 1 MiB', text)
        self.assertIn("Ring/LL128 left out: off in NCCL's", text)
        self.assertIn('switches Ring/LL -> Ring/Simple at 470 KiB', text)
        # the switch of the table itself: 13 + S / 8000 = 43 + S / 16000 us
        self.assertAlmostEqual(d2l._tuner_switch(
            tuner_case(), 'Ring/LL', 'Ring/Simple'), 480e3, delta=1e3)
        # without NCCL's choices there is nothing to test
        lines, out = lines_of(self, d2l.tuner_regret,
                              synthetic_pairs(self.params()))
        self.assertTrue(out.empty)
        self.assertEqual(len(lines), 1)

    def test_symptoms(self):
        ab = dict(alpha=20e-6, beta=2.36e9)
        lines, _ = lines_of(self, d2l.coll_symptoms, {'pairs': tuner_case()},
                            ab)
        self.assertEqual(len(lines), 2)
        self.assertIn("on 4 GPUs: NCCL's choice Ring/LL", lines[0])
        self.assertIn('forced Ring/Simple', lines[0])
        self.assertIn("NCCL's model of Ring/Simple: 24 GB/s busbw", lines[1])
        self.assertIn('measured line: 2.4 GB/s', lines[1])
        lines, _ = lines_of(self, d2l.coll_symptoms,
                            d2l._hw_pending('box-coll'), ab)
        self.assertEqual(lines, ['box-coll pending'])

    def test_records(self):
        fits = d2l.fit_pairs(synthetic_pairs(self.params()))
        info = d2l._parse_info(NCCL_INFO_SAMPLE)
        recs = d2l._coll_records(fits, info)
        self.assertEqual(len(recs), 4)
        default = [r for r in recs if r['algo'] is None]
        self.assertEqual(len(default), 1)
        self.assertIsNone(default[0]['proto'])
        self.assertEqual({r['link'] for r in recs}, {'pcie_host_staged'})
        self.assertEqual({r['transport'] for r in recs}, {'shm'})
        self.assertEqual({(r['algo'], r['proto']) for r in recs if r['algo']},
                         {('Ring', 'LL'), ('Ring', 'Simple'), ('Tree', 'LL')})
        json.dumps(recs, allow_nan=False)

    def test_hier_display(self):
        rows = [dict(variant=v, size=s, us=u * s / 2**24, mad_us=1.0)
                for s in (2**24, 2**28)
                for v, u in zip(d2l._HIER_ROWS, (9e4, 2e4, 3e4, 1e4, 2e4))]
        r = pd.DataFrame(rows)
        with mock.patch.object(d2l, '_hier_predict', return_value=None):
            lines, _ = lines_of(self, d2l._show_hier, r)
        self.assertLessEqual(len(lines), 10)  # two header rows, five rows
        text = '\n'.join(lines)
        self.assertIn('hierarchical 3.0x faster than flat gloo', text)
        self.assertIn('no model', text)
        self.assertIn('intra/inter bandwidth 4.0x measured', lines[-1])

    def test_c4_gate(self):
        """The 22.6 gate rule: the whole interval below half the
        shared-memory beta, or with one process the point below a third."""
        shm = dict(beta=2.36e9, alpha=30e-6)
        gloo = lambda b, lo, hi: pd.Series(dict(beta=b, beta_lo=lo,
                                                beta_hi=hi, alpha=2.4e-3))
        stands, line = d2l._c4_gate(gloo(1.21e9, 0.86e9, 1.6e9), shm)
        self.assertFalse(stands)  # the 22.6 smoke: interval straddles 1.18
        self.assertIn('fallback', line)
        self.assertIn('1/80 the latency', line)
        self.assertTrue(d2l._c4_gate(gloo(0.6e9, 0.5e9, 0.7e9), shm)[0])
        nan = float('nan')
        self.assertFalse(d2l._c4_gate(gloo(1.0e9, nan, nan), shm)[0])
        stands, line = d2l._c4_gate(gloo(0.7e9, nan, nan), shm)
        self.assertTrue(stands)
        self.assertIn('one run', line)

    def test_campaign_config_refused(self):
        with self.assertRaisesRegex(ValueError, 'campaign launcher'):
            d2l.nccl_algo_proto_sweeps('t4')


PRICES = os.path.join(REPO, 'tools', 'lm_data', 'hw', 'prices.csv')


def sweep_log(P_s=80.0, k=80.0, clocks=(0.6, 0.76, 0.92, 1.08, 1.24, 1.4),
              noise=0.0, seed=0, caps=False):
    """A clock (and power-cap) sweep of known parameters, as the root
    script writes it."""
    rng = np.random.default_rng(seed)
    f = np.array(clocks)
    P = (P_s + k * f**3) * (1 + noise * rng.standard_normal(len(f)))
    df = pd.DataFrame(dict(kind='clock', setting=f * 1e3,
                           workload='matmul-8192', sm_mhz=f * 1e3,
                           power_w=P, rate=f * 1e14, unit='FLOP'))
    if caps:
        rows = []
        for cap in (150, 225, 300, 450):
            rows.append(dict(kind='cap', setting=cap, workload='train-8192',
                             step_ms=100 * 450 / cap, step_mad_ms=1.0,
                             power_w=min(cap, 355.0)))
            rows.append(dict(kind='cap', setting=cap, workload='decode-b1',
                             step_ms=3.0 * max(1.0, 190 / cap),
                             step_mad_ms=0.03, power_w=min(cap, 190.0)))
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    df.attrs.update(source='measured on a test card', synthetic=False)
    return df


def decode_frame():
    """A decode series with J/token ~ 1/b below a knee of 20 and flat
    above it (L = 512), and a knee of 5 at L = 2048."""
    rows = []
    for L, knee in ((512, 20.0), (2048, 5.0)):
        for b in (1, 2, 4, 8, 16, 32, 64, 128):
            if L == 2048 and b == 128:
                continue
            j = 1.0 / min(b, knee)
            rows.append(dict(b=b, L=L, tokens_per_s=100.0 * b, j_per_token=j,
                             power_w=100.0 * b * j))
    e = pd.DataFrame(rows)
    e.attrs.update(knee={'512': 20.0, '2048': 5.0}, cut=[[128, 2048]],
                   model='Qwen/Qwen3-0.6B')
    return e


def max_width(fn, *args):
    import matplotlib
    matplotlib.use('Agg')
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = fn(*args)
    d2l.plt.close('all')
    lines = buf.getvalue().splitlines()
    return max(map(len, lines)) if lines else 0, lines, out


class TestEnergyCost(unittest.TestCase):
    """Section 22.8: power windows, the clock model, energy per token, the
    cost calculators, the price table and the MFU and goodput readers."""

    def setUp(self):
        self.prices = pd.read_csv(PRICES)
        self.prices['age'] = 0
        self.specs = pd.read_csv(os.path.join(
            REPO, 'tools', 'lm_data', 'hw', 'chip_specs.csv')).set_index(
                'chip')

    def test_energy_per_token(self):
        trace = [dict(t=0.0, power_w=100.0), dict(t=1.0, power_w=300.0),
                 dict(t=2.0, power_w=300.0)]
        self.assertAlmostEqual(d2l.energy_per_token(trace, 50), 500 / 50)

    def test_throttle(self):
        samples = [dict(throttle=0x4), dict(throttle=0x4 | 0x1),
                   dict(throttle=0x0), dict(throttle=0x4)]
        f = d2l._throttle_fractions(samples)
        self.assertEqual(f['thr_SW_power_cap'], 0.75)
        self.assertEqual(f['thr_idle'], 0.25)
        self.assertEqual(d2l._throttle_names(f), 'SW power cap')
        self.assertEqual(d2l._throttle_names({}), 'none')
        self.assertTrue(math.isnan(d2l._throttle_fractions([])['thr_idle']))

    def test_dvfs_plan_numbers(self):
        with mock.patch.dict(d2l.DATA_HUB, {}, clear=True):
            log = d2l.hw_clock_sweep_log()
        self.assertTrue(log.attrs['synthetic'])
        fit = d2l.dvfs_fit(log)
        self.assertAlmostEqual(fit.P_s, 80, places=6)
        self.assertAlmostEqual(fit.k, 80, places=6)
        self.assertAlmostEqual(fit.f_star, 0.5 ** (1 / 3), places=6)
        self.assertAlmostEqual(fit.f_star, 0.79, places=2)
        width, lines, _ = max_width(d2l.plot_dvfs, fit)
        self.assertLessEqual(width, 79)
        self.assertIn('P(f*) = 120 W', lines[0])
        self.assertIn('1.4x the work per joule', lines[1])
        self.assertIn('0.57x the throughput of 1.40 GHz', lines[1])
        self.assertIn('synthetic', lines[2])

    def test_dvfs_noisy_recovery(self):
        fit = d2l.dvfs_fit(sweep_log(P_s=60, k=90, noise=0.01))
        self.assertAlmostEqual(fit.P_s / 60, 1, delta=0.1)
        self.assertAlmostEqual(fit.k / 90, 1, delta=0.05)
        self.assertFalse(fit.synthetic)
        self.assertAlmostEqual(fit.step, 0.16)
        # measured: the swept minimum of power / rate is the clock nearest f*
        f = np.array([0.6, 0.76, 0.92, 1.08, 1.24, 1.4])
        self.assertEqual(fit.f_min, f[np.argmin((60 + 90 * f**3) / f)])
        width, lines, _ = max_width(d2l.plot_dvfs, fit)
        self.assertLessEqual(width, 79)
        self.assertIn('measured energy minimum', lines[-3])
        self.assertIn('inside the swept clocks', lines[-2])
        self.assertEqual(lines[-1], 'measured on a test card')
        # a sweep that stops above f*: the model extrapolates
        fit = d2l.dvfs_fit(sweep_log(P_s=60, k=90, clocks=(1.0, 1.2, 1.4)))
        self.assertGreater(min(fit.clocks), fit.f_star)
        width, lines, _ = max_width(d2l.plot_dvfs, fit)
        self.assertIn('the model extrapolates', lines[-2])
        self.assertLessEqual(width, 79)

    def test_clock_sweep_needs_root(self):
        refused = d2l.HWRun([], 4, '', 'denied', 0.1, 'exit status 4')
        calls = []

        def run(cmd, timeout_s, **kw):
            calls.append(cmd)
            return refused
        with mock.patch.object(d2l, 'hw_run', run), mock.patch.object(
                d2l, '_hw_visible_uuids', lambda: ['GPU-x']):
            with self.assertRaisesRegex(PermissionError, 'root'):
                d2l.hw_clock_sweep([1000, 2000], ['matmul-8192'])
            with self.assertRaisesRegex(ValueError, 'visible GPU 0'):
                d2l.hw_clock_sweep([1000], ['matmul-8192'], device=1)
        self.assertEqual(len(calls), 1)  # nothing after the refusal

    def test_clock_sweep_restores(self):
        ok = d2l.HWRun([], 0, '', '', 0.1, None)
        calls = []

        def run(cmd, timeout_s, **kw):
            calls.append(cmd[3:])
            return ok

        def rows(kind, setting, workloads, seconds=10):
            if setting == 2000:
                raise RuntimeError('workload failed')
            return [dict(kind=kind, setting=setting, workload=w)
                    for w in workloads]
        with mock.patch.object(d2l, 'hw_run', run), mock.patch.object(
                d2l, '_hw_visible_uuids', lambda: ['GPU-x']), \
                mock.patch.object(d2l, '_sweep_rows', rows):
            df = d2l.hw_clock_sweep([1000, 1500], ['a', 'b'])
            self.assertEqual(len(df), 4)
            self.assertEqual(calls[-1], ['-rgc'])
            calls.clear()
            with self.assertRaises(RuntimeError):
                d2l.hw_clock_sweep([1000, 2000], ['a'])
            self.assertEqual(calls[-1], ['-rgc'])
            self.assertIn(['-lgc', '2000,2000'], calls)

    def test_cap_effect(self):
        width, lines, out = max_width(d2l.cap_effect, sweep_log())
        self.assertEqual(len(out), 0)
        self.assertEqual(len(lines), 1)
        self.assertIn('not measured yet', lines[0])
        self.assertNotIn('C-box', lines[0])
        # the page's E5 check holds on the empty table
        above = out[out['cap W'] > 1.1 * out.attrs['decode_w']]
        self.assertTrue((above['decode ratio'] - 1 < above['noise']).all())
        width, lines, t = max_width(d2l.cap_effect, sweep_log(caps=True))
        self.assertLessEqual(width, 79)
        self.assertEqual(list(t['cap W']), [150, 225, 300])
        np.testing.assert_allclose(t['train ratio'], [3, 2, 1.5])
        np.testing.assert_allclose(t['decode ratio'], [190 / 150, 1, 1])
        np.testing.assert_allclose(t['noise'], 0.03)  # 3 x 0.01 < floor
        self.assertEqual(t.attrs['decode_w'], 190.0)
        self.assertIn('measured on a test card', lines[-1])
        self.assertIn('decode draws 190 W', lines[-1])
        # E5 holds above decode's draw, and the 150 W cap below it is out
        above = t[t['cap W'] > 1.1 * t.attrs['decode_w']]
        self.assertEqual(list(above['cap W']), [225, 300])
        self.assertTrue((above['decode ratio'] - 1 < above['noise']).all())
        log = sweep_log(caps=True)
        log = log[~log.workload.str.startswith('train')]
        log.attrs.update(source='x', synthetic=False)
        with self.assertRaisesRegex(ValueError, 'no train workload'):
            d2l.cap_effect(log)

    def test_quiet_window_discards_contended(self):
        flags = iter([1.0, 1.0, 0.0])
        window = lambda step, work, **kw: dict(contended=next(flags), n=1)
        with mock.patch.object(d2l, '_power_window', window):
            self.assertEqual(d2l._quiet_window(None, 1)['contended'], 0.0)
        window = lambda step, work, **kw: dict(contended=1.0)
        with mock.patch.object(d2l, '_power_window', window):
            with self.assertRaisesRegex(RuntimeError, 'foreign process'):
                d2l._quiet_window(None, 1)

    def test_power_window_counter(self):
        samples = [dict(t=0.1 * i, power_w=200.0, sm_mhz=2500,
                        throttle=0x4) for i in range(11)]

        @contextlib.contextmanager
        def trace(hz):
            yield samples
        energy = iter([1000.0, 1100.0])  # J at the window's two ends
        contention = iter([(0, 0), (1, 0)])  # a foreign process appears
        with mock.patch.object(d2l, '_hw_power_trace', trace), \
                mock.patch.object(d2l, '_hw_energy_j', lambda: next(energy)), \
                mock.patch.object(d2l, '_hw_contention',
                                  lambda: next(contention)), \
                mock.patch.object(d2l.torch.cuda, 'synchronize', lambda: 0):
            w = d2l._power_window(lambda: None, 5, steps=4, warm_s=0)
        self.assertEqual(w['steps'], 4)
        self.assertAlmostEqual(w['j_per_unit_counter'], 100 / 20)
        self.assertAlmostEqual(w['power_w'], 200.0)
        self.assertEqual(w['contended'], 1.0)
        self.assertEqual(w['thr_SW_power_cap'], 1.0)
        self.assertGreaterEqual(w['step_mad_ms'], 0)

    def test_cap_sweep_safety(self):
        with mock.patch.object(d2l, '_hw_visible_uuids', lambda: ['GPU-x']), \
                mock.patch.object(d2l, '_hw_gpu_limits', lambda: {}):
            with self.assertRaisesRegex(RuntimeError, 'default power limit'):
                d2l._cap_sweep([300], ['decode-b1'])
        refused = d2l.HWRun([], 4, '', 'denied', 0.1, 'exit status 4')
        calls = []

        def run(cmd, timeout_s, **kw):
            calls.append(cmd[3:])
            return refused
        with mock.patch.object(d2l, 'hw_run', run), mock.patch.object(
                d2l, '_hw_visible_uuids', lambda: ['GPU-x']), \
                mock.patch.object(d2l, '_hw_gpu_limits',
                                  lambda: {'power_default_limit': 450.0}):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf), \
                    self.assertRaisesRegex(PermissionError, 'root'):
                d2l._cap_sweep([300], ['decode-b1'])
        self.assertEqual(calls, [['-pl', '300'], ['-pl', '450']])
        self.assertIn('-pl 450 failed', buf.getvalue())  # one line
        self.assertEqual(len(buf.getvalue().splitlines()), 1)
        with mock.patch.object(d2l.os, 'geteuid', lambda: 0), \
                mock.patch.dict(os.environ, {'HF_HOME': ''}):
            with self.assertRaisesRegex(RuntimeError, 'HF_HOME'):
                d2l.hw_root_sweep()

    def test_decode_slopes_and_display(self):
        e = decode_frame()
        s = d2l.decode_slopes(e[e.L == 512], 20)
        self.assertAlmostEqual(s.below, -1, places=6)
        self.assertAlmostEqual(s.above, 0, places=6)
        width, lines, _ = max_width(d2l._show_decode_energy, e)
        self.assertLessEqual(width, 79)
        self.assertIn('J/token at b = 1, 0.050 at b = 64', lines[0])
        self.assertIn('(128, 2,048)', lines[-1])
        e = e.iloc[:0]
        e.attrs['label'] = 'box-power: pending'
        self.assertEqual(max_width(d2l._show_decode_energy, e)[1],
                         ['box-power: pending'])

    def test_profile_and_train_displays(self):
        prof = pd.DataFrame(dict(
            workload=['idle', 'decode-b1', 'matmul-8192'],
            power_w=[22.0, 150.0, 447.0], sm_mhz=[210.0, 2520.0, 2385.0],
            thr_idle=[1.0, 0, 0], thr_SW_power_cap=[0, 0, 1.0]))
        prof.attrs.update(field='power.draw (NVML, 1 s average)', hz=10,
                          power_limit_w=450, max_sm_mhz=3120)
        width, lines, _ = max_width(d2l._show_profile, prof)
        self.assertLessEqual(width, 79)
        self.assertIn('SW power cap', lines[3])
        self.assertIn('matmul 8192^3', lines[3])
        self.assertIn('limit 450 W', lines[-1])
        self.assertIn('idle (context open)', lines[1])
        self.assertNotIn('top SM', lines[-1])
        tr = pd.DataFrame(dict(tokens_per_step=[8192, 32768], accum=[1, 4],
                               tokens_per_s=[9e4, 1e5],
                               power_w=[420.0, 430.0],
                               j_per_token=[0.0047, 0.0043],
                               j_per_token_counter=[0.0048, 0.0043],
                               mfu=[0.45, 0.5]))
        tr.attrs['peak_flops'] = 165.2e12
        width, lines, _ = max_width(d2l._show_train_energy, tr)
        self.assertLessEqual(width, 79)
        self.assertIn('165 TFLOP/s', lines[-2])
        self.assertIn('within 2.1%', lines[-1])

    def test_cost_functions(self):
        N, D, P = 1e9, 1e12, 989.4e12
        gpu_h = d2l.cost_per_mtok_train(1.0, 6 * N, P, 0.4) * D / 1e6
        self.assertAlmostEqual(gpu_h, 6 * N * D / (0.4 * P * 3600))
        self.assertAlmostEqual(gpu_h, 4212, delta=1)
        self.assertAlmostEqual(
            d2l.cost_per_mtok_train(2.0, 6 * N, P, 0.4, goodput=0.5),
            4 * d2l.cost_per_mtok_train(1.0, 6 * N, P, 0.4))
        self.assertAlmostEqual(d2l.cost_per_mtok_serve(3.6, 1000, 0.5),
                               2.0)

    def test_owned_vs_rented(self):
        r = d2l.owned_vs_rented(30000, 30000, 1.0, 1.5, 0.1, 0.35, 3.0)
        self.assertAlmostEqual(r['owned_per_used_h'], 1 + 0.15 + 0.35)
        self.assertAlmostEqual(r['u_star'], 1.5 / 3.0)

    def test_owned_inputs_interval_tornado(self):
        inputs = d2l.owned_inputs(self.prices, self.specs)
        self.assertEqual(list(inputs), ['RTX 4090 (box)', 'H100 SXM'])
        box = inputs['RTX 4090 (box)']
        self.assertEqual(box['capex'], 12000 / 4)
        self.assertEqual(box['power_kw'], 0.45)
        self.assertEqual(box.kinds['capex'], 'assumption')
        self.assertEqual(box.kinds['rent_per_h'], 'quote')
        self.assertEqual(inputs['H100 SXM']['rent_per_h'],
                         d2l.rent_per_hour(self.prices, 'H100 SXM'))
        for x in inputs.values():
            u = d2l.owned_vs_rented(**x)['u_star']
            lo, hi = d2l.u_star_interval(x, draws=2000)
            self.assertLess(lo, u)
            self.assertLess(u, hi)
            self.assertEqual(d2l.u_star_interval(x, draws=2000), (lo, hi))
            order = d2l.tornado_order(x)
            self.assertEqual(order[0][0], 'rent_per_h')  # E6
            self.assertEqual(len(order), 7)
        import matplotlib
        matplotlib.use('Agg')
        d2l.plot_tornado(inputs)
        d2l.plt.close('all')

    def test_units(self):
        width, lines, _ = max_width(d2l.check_cost_units)
        self.assertEqual(len(lines), 1)
        self.assertLessEqual(width, 79)
        usd, s = d2l._Dim(dict(usd=1)), d2l._Dim(dict(s=1))
        with self.assertRaises(TypeError):
            usd + s
        self.assertEqual((usd / s * s).u, {'usd': 1})

    def test_price_table(self):
        d2l._check_prices(self.prices.drop(columns='age'))
        self.assertEqual(set(self.prices.kind), {'quote', 'assumption'})
        rents = self.prices[self.prices.item.str.startswith('rent')]
        for chip in ('A100 SXM 80 GB', 'H100 SXM', 'B200'):
            self.assertGreaterEqual(
                (rents.chip == chip).sum(), 2, chip)  # two providers
        for chip in ('RTX 4090', 'H100 SXM', 'B200', 'A100 SXM 80 GB'):
            self.assertGreater(d2l.rent_per_hour(self.prices, chip), 0)
        for item in ('electricity', 'PUE', 'life'):
            self.assertEqual((self.prices.item == item).sum(), 1, item)
        self.assertEqual(
            d2l.rent_per_hour(self.prices, 'H100 SXM'), 4.89)  # median
        with self.assertRaises(KeyError):
            d2l.rent_per_hour(self.prices, 'TPU v5e')
        import datetime
        ages = d2l._price_age(self.prices.drop(columns='age'),
                              datetime.date(2027, 3, 1))
        self.assertEqual(set(ages), {96})
        bad = self.prices.drop(columns='age').copy()
        bad.loc[0, 'kind'] = 'guess'
        with self.assertRaises(AssertionError):
            d2l._check_prices(bad)

    def test_price_display(self):
        prices = self.prices.copy()
        prices.loc[0, 'age'] = 120
        static = d2l._hw_static
        table = lambda n: (prices.drop(columns='age') if n == 'hw-prices'
                           else static(n))
        with mock.patch.object(d2l, '_hw_static', table), mock.patch.object(
                    d2l, '_price_age', lambda df: list(prices.age)):
            width, lines, out = max_width(d2l.hw_prices)
        self.assertLessEqual(width, 79)
        self.assertLessEqual(len(lines), 13)
        self.assertIn('120 (old)', lines[1])  # the A100 PCIe rents
        self.assertEqual(len(out), len(prices))

    def test_reference_mfu_conversion(self):
        arch = d2l.r2_config('r2').arch
        key = d2l.PRETRAIN_ARTIFACTS['reference-run-log'][0]
        log = pd.DataFrame(dict(step=np.arange(100),
                                mfu=[0.1] * 10 + [0.4] * 90))
        with mock.patch.dict(d2l.DATA_HUB, {key: ('x', None)}), \
                mock.patch.object(d2l, 'load_artifact', lambda n: log), \
                mock.patch.object(d2l, 'chip_specs', lambda: self.specs):
            m = d2l._reference_run_mfu(arch)
        full = d2l.flops_per_token(arch, T=arch.T)
        half = d2l.flops_per_token(arch, T=arch.T, causal_half=True)
        self.assertAlmostEqual(m, 0.4 * half / full * 165 / 165.2)
        with mock.patch.dict(d2l.DATA_HUB, {}, clear=True):
            self.assertIsNone(d2l._reference_run_mfu(arch))

    def test_box_qwen8b_mfu(self):
        cfg = d2l.hw_config('Qwen/Qwen3-8B')
        t1 = 50.0  # ms per layer, 20 ms outside the layers
        grid = pd.DataFrame(dict(
            workload='train', model='Qwen/Qwen3-8B', b=2, L=2048,
            n_layers=[1, 2, 3], ms=[20 + t1 * n for n in (1, 2, 3)]))
        with mock.patch.object(d2l, 'hw_box_log', lambda n, e: grid), \
                mock.patch.object(d2l, 'chip_specs', lambda: self.specs):
            m, s = d2l._box_qwen8b_mfu(cfg)
        t = (20 + t1 * cfg.num_hidden_layers) / 1e3
        self.assertEqual(s, 2048)
        self.assertAlmostEqual(m, d2l.mfu(2 * 2048 / t, cfg, 165.2e12,
                                          T=2048, causal_half=True))

    def test_training_mfu_paths(self):
        tr = pd.DataFrame(dict(mfu=[0.4, 0.45]))
        table = [dict(model='Qwen3-8B', chip='H100 SXM', mfu=0.38, T=4096)]
        with mock.patch.object(d2l, '_reference_run_mfu', lambda a: None), \
                mock.patch.object(d2l, 'hw_box_log', lambda n, e: tr), \
                mock.patch.object(d2l, '_box_qwen8b_mfu',
                                  lambda cfg: (0.7, 2048)), \
                mock.patch.object(d2l, 'hw_campaign_log', lambda n: table):
            df = d2l.training_mfu()
        self.assertEqual(list(df.model), ['reference'] + ['Qwen3-8B'] * 4)
        self.assertAlmostEqual(df.mfu[0], 0.425)  # the median
        box = df[(df.model == 'Qwen3-8B') & (df.chip == 'RTX 4090')].iloc[0]
        self.assertEqual(box['mfu'], 0.7)
        self.assertIn('upper bound', box['source'])
        h100 = df[df.chip == 'H100 SXM'].iloc[0]
        self.assertEqual((h100['mfu'], h100['T']), (0.38, 4096))
        for chip in ('A100 SXM 80 GB', 'B200'):
            r = df[df.chip == chip].iloc[0]
            self.assertEqual((r['mfu'], r['T']), (0.4, 2048))
            self.assertIn('stated', r['source'])
        # a 4090 MFU never appears on another chip
        self.assertEqual(set(df[df.mfu == 0.7].chip), {'RTX 4090'})
        pending = d2l.CampaignPending('mfu-table', 'chapter 23', 'x')
        with mock.patch.object(d2l, '_reference_run_mfu', lambda a: None), \
                mock.patch.object(d2l, 'hw_box_log', lambda n, e: tr), \
                mock.patch.object(d2l, '_box_qwen8b_mfu',
                                  lambda cfg: (0.7, 2048)), \
                mock.patch.object(d2l, 'hw_campaign_log',
                                  lambda n: pending):
            df = d2l.training_mfu()
        dc = df[df.chip != 'RTX 4090']
        self.assertEqual(list(dc.mfu), [0.4] * 3)
        self.assertLessEqual(max(map(len, df.source)), 30)
        with mock.patch.dict(d2l.DATA_HUB, {}, clear=True):
            self.assertEqual(d2l.training_goodput(0.85)[0], 0.85)
        with mock.patch.object(d2l, 'hw_campaign_log',
                               lambda n: {'goodput': 0.93}):
            self.assertEqual(d2l.training_goodput(),
                             (0.93, 'chapter 23, measured'))

    def test_serving_readers(self):
        grid = pd.DataFrame(dict(
            workload=['decode'] * 4 + ['prefill'], series='grid',
            model='Qwen/Qwen3-8B', L=[2048] * 4 + [2048],
            b=[1, 4, 16, 32, 1], ms=[20.0, 21.0, 25.0, 30.0, 250.0]))
        p = d2l.frontier_points(grid, 'Qwen/Qwen3-8B', 2048, (1, 4, 16))
        self.assertEqual(list(p.b), [1, 4, 16])
        np.testing.assert_allclose(p.user_tps, [50, 1e3 / 21, 40])
        np.testing.assert_allclose(p.gpu_tps, [50, 4e3 / 21, 640])
        self.assertEqual(d2l.prefill_seconds(grid, 'Qwen/Qwen3-8B', 2048),
                         0.25)

    def test_registry(self):
        self.assertIn('box-power', d2l.HW_JOBS)
        self.assertEqual(d2l.HW_STATIC['hw-prices'][0], 'lm-hw-prices-v1')
        self.assertIn('lm-hw-prices-v1', d2l.DATA_HUB)
        for name in ('box-clocksweep', 'mfu-table', 'par-goodput'):
            self.assertIn(name, d2l.HW_CAMPAIGNS)


def collections_counter(tree):
    """Children per rank of a {rank: parent} tree."""
    return d2l.collections.Counter(p for p in tree.values() if p is not None)


if __name__ == '__main__':
    unittest.main()
