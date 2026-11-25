#!/usr/bin/env python3
"""CPU tests for sec_lm_budgets (chapter_lm-pretraining/budgets.md and
budgets-lib.md): the page's optimum cells (the root at no demand equals the
closed form of eq_chinchilla_optimum; N* falls with demand; the plan's
smallest case 15.3B / 11.5B / 6.5B and 79 / 143 / 600 tokens per
parameter; the lifetime-optimal ratio grows with demand under both fits);
`lifetime_cost` at no demand is 6ND; the roofline at batch 1 equals
W / (N w + c k); `effective_tokens` at one epoch and its derivative;
`fit_law` recovers noiseless exponents, and a warm-started refit reaches
the minimum of the full search (the L-BFGS-B tolerance regression);
`huber_objective`'s gradient against finite differences; the optimum of a
fit with a tiny exponent raises no overflow, and the ladder-optimum cell's
feasibility test; the ladder label (`show_ladder_label`, the "no verdict"
and plateau lines of a simulated ladder; a committed ladder without plateau
points is refused); `fmt_power` and `fmt_sig`; the decode harness stops
when it is not faster than Hugging Face eager decode; the
static decode step equals `forward_cached` (PackedGPT and Qwen3GPT with
QK-norm); `decode_shapes` (r2's 18,432 cache bytes per token and
138,440,448 parameters); the pending `decode_rates` table; the simulated
ladder (the twelve planned points) and a committed ladder table;
`training_rate` from a log and while pending; the regeneration job
against the page's literals; the whole page on the pending path (output
at most 15 lines per cell and 100 columns, cells 12-17 under 90 s); and
the fit cells on an ill-identified ladder (like the smoke run's) under
90 s at the spec's fallback of 500 resamples.
"""
import contextlib
import io
import math
import os
import re
import time
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch

from d2l import torch as d2l


def uncommitted():
    """d2l.DATA_HUB without chapter 21's results: the state before a
    campaign's result is registered (the tests of pending and regeneration
    paths must not depend on which results run A has committed)."""
    keys = {key for key, _ in d2l.PRETRAIN_ARTIFACTS.values()}
    return mock.patch.object(d2l, 'DATA_HUB', {
        k: v for k, v in d2l.DATA_HUB.items() if k not in keys})

torch.set_num_threads(min(8, torch.get_num_threads()))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER = os.path.join(ROOT, 'chapter_lm-pretraining')
PAGE = os.path.join(CHAPTER, 'budgets.md')
C1 = {'reference-run', 'ladder', 'ladder-evals', 'reference-evals', 'inst-t2'}


def page_cells():
    with open(PAGE) as f:
        src = f.read()
    return dict(re.findall(r'```\{\.python \.input #([^}]*)\}\n'
                           r'%%tab pytorch\n(.*?)```', src, re.S))


def run_cells(names, ns):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        for name in names:
            exec(page_cells()[name], ns)
    return out.getvalue()


def math_ns():
    ns = {}
    run_cells(['budgets-imports', 'budgets-imports-extra', 'budgets-opening',
               'budgets-solve-optimum'], ns)
    return ns


FIT_FIRST = 'budgets-fitting-the-chinchilla-form-1'  # law, huber_objective
FIT_CELLS = [FIT_FIRST, 'budgets-fit-law',
             'budgets-fitting-the-chinchilla-form-2', 'budgets-recovery',
             'budgets-intervals', 'budgets-withheld', 'budgets-plateau',
             'budgets-ladder-optimum']  # spec cells 12-17


def fit_ns():
    ns = math_ns()
    run_cells([FIT_FIRST, 'budgets-fit-law'], ns)  # law, fit_law
    return ns


@contextlib.contextmanager
def in_chapter():
    old = os.getcwd()
    os.chdir(CHAPTER)
    try:
        yield
    finally:
        os.chdir(old)


def qwen_configs_cached():
    try:
        with in_chapter():
            d2l.decode_shapes(['Qwen/Qwen3-0.6B'])
        return True
    except Exception:  # noqa: BLE001 (offline and not cached)
        return False


class OptimumTest(unittest.TestCase):
    def test_closed_form_and_monotone(self):
        ns = math_ns()
        solve, train_opt = ns['solve_optimum'], ns['train_optimum']
        for fit in ns['FITS'].values():
            N0, D0 = solve(2.0, 0, *fit)
            N_c, D_c = train_opt(6 * N0 * D0, *fit)
            self.assertLess(abs(N0 / N_c - 1), 1e-8)
            self.assertLess(abs(D0 / D_c - 1), 1e-8)
            Ns = [solve(2.0, x, *fit)[0] for x in np.logspace(9, 16, 15)]
            self.assertTrue(np.all(np.diff(Ns) < 0))
            E, A, B, alpha, beta = fit
            N_min = (A / (2.0 - E)) ** (1 / alpha)
            near = [solve(2.0, x, *fit)[0] / N_min for x in (1e16, 1e18,
                                                              1e20)]
            self.assertTrue(near[0] > near[1] > near[2] > 1)  # toward N_min
            self.assertLess(near[2], 1.05)

    def test_smallest_case(self):  # the plan's numbers, Hoffmann, loss 2.0
        ns = math_ns()
        got = [ns['solve_optimum'](2.0, x, *ns['FITS']['Hoffmann'])
               for x in (0, 1e12, 1e13)]
        self.assertEqual([round(N / 1e9, 1) for N, _ in got],
                         [15.3, 11.5, 6.5])
        self.assertEqual([round(D / N) for N, D in got], [79, 143, 600])

    def test_foc_holds_at_root(self):
        ns = math_ns()
        E, A, B, alpha, beta = ns['FITS']['Besiroglu']
        N, D = ns['solve_optimum'](2.0, 1e12, E, A, B, alpha, beta)
        a, b = A * N ** -alpha, B * D ** -beta
        self.assertAlmostEqual(E + a + b, 2.0, places=10)
        self.assertAlmostEqual(alpha * a / (beta * b * (1 + 1e12 / (3 * D))),
                               1.0, places=8)
        # the root minimizes the lifetime cost along the contour
        cost = ns['lifetime_cost']
        for f in (0.98, 1.02):
            n = N * f
            d = (B / (2.0 - E - A * n ** -alpha)) ** (1 / beta)
            self.assertGreater(cost(n, d, 1e12), cost(N, D, 1e12))

    def test_lifetime_cost_no_demand(self):
        ns = math_ns()
        self.assertEqual(ns['lifetime_cost'](3e9, 7e10, 0), 6 * 3e9 * 7e10)

    def test_opening_numbers(self):
        out = run_cells(['budgets-imports', 'budgets-imports-extra',
                         'budgets-opening'], {})
        self.assertIn('reference model: 138M parameters, 3.15B tokens, '
                      '22.7 tokens per parameter', out)
        self.assertIn('Llama 3 8B: ≈ 15T tokens / 8.0B ≈ 1,875 per '
                      'parameter', out)
        self.assertEqual(len(out.splitlines()), 3)

    def test_ratio_band_cell(self):
        ns = math_ns()
        out = run_cells(['budgets-optimum-checks', 'budgets-ratio-band'], ns)
        self.assertIn('N* decreasing in demand: True', out)
        self.assertRegex(out, r'every D_inf up to \d×10\^1\d$')
        d2l.plt.close('all')


class RooflineAndRepeatTest(unittest.TestCase):
    def test_roofline_batch_one(self):
        ns = math_ns()
        code = page_cells()['budgets-decode-roofline']
        exec(code[:code.index('arch = cfg.arch')], ns)
        N, w, k, c = 138e6, 2, 18_432, 1024
        self.assertAlmostEqual(ns['decode_roofline'](N, w, k, c, 1),
                               ns['BW'] / (N * w + c * k), places=6)
        big = ns['decode_roofline'](N, w, k, c, 4096)  # compute-bound
        self.assertLessEqual(big, ns['PEAK'] / (2 * N) * (1 + 1e-12))

    def test_effective_tokens(self):
        self.assertEqual(d2l.effective_tokens(5.0, 1.0), 5.0)
        E, h = np.array([2.0, 7.0, 30.0]), 1e-6
        fd = (d2l.effective_tokens(1.0, E + h) - d2l.effective_tokens(
            1.0, E - h)) / (2 * h)
        np.testing.assert_allclose(fd, np.exp(-(E - 1) / 15.4), atol=1e-6)
        out = run_cells(['budgets-imports', 'budgets-imports-extra',
                         'budgets-repeated-data'], {})
        self.assertIn("marginal value dD'/dE at epoch 4 / 16 / 40: 0.82 / "
                      '0.38 / 0.08 of a fresh epoch', out)
        d2l.plt.close('all')


class FitLawTest(unittest.TestCase):
    def setUp(self):
        plan = d2l._ladder_plan()
        pts = plan[plan['finished']]
        self.N = pts['N'].to_numpy(float)
        self.D = pts['tokens'].to_numpy(float)
        self.truth = d2l.HOFFMANN_FIT

    def test_noiseless_recovery(self):
        ns = fit_ns()
        L = ns['law'](self.N, self.D, *self.truth)
        got = ns['fit_law'](self.N, self.D, L)
        np.testing.assert_allclose(got, self.truth, rtol=2e-3)

    def huber(self, ns, p, L):
        r = np.log(L) - np.log(ns['law'](self.N, self.D, *p))
        return float(np.sum(ns['special'].huber(1e-3, r)))

    def test_warm_start_converges(self):
        """Regression: with scipy's default tolerances a warm start stopped
        after a few iterations near its start, so refits from the truth
        'recovered' it on every noise draw."""
        ns = fit_ns()
        for seed in (0, 2, 9):
            rng = np.random.default_rng(seed)
            L = ns['law'](self.N, self.D, *self.truth) + rng.normal(
                0, 0.01, len(self.N))
            full = ns['fit_law'](self.N, self.D, L)
            warm = ns['fit_law'](self.N, self.D, L, init=self.truth)
            self.assertLess(self.huber(ns, warm, L),
                            self.huber(ns, self.truth, L) * 0.95)
            self.assertLess(abs(self.huber(ns, warm, L)
                                / self.huber(ns, full, L) - 1), 0.01)

    def test_gradient(self):
        ns = fit_ns()
        L = ns['law'](self.N, self.D, *self.truth) + np.random.default_rng(
            1).normal(0, 0.01, len(self.N))
        data = np.log(self.N), np.log(self.D), np.log(L)
        p, h = np.array([0.4, 6.0, 6.2, 0.3, 0.25]), 1e-6
        f, g = ns['huber_objective'](p, *data)
        fd = [(ns['huber_objective'](p + h * e, *data)[0]
               - ns['huber_objective'](p - h * e, *data)[0]) / (2 * h)
              for e in np.eye(5)]
        np.testing.assert_allclose(g, fd, rtol=1e-4, atol=1e-10)


class FeasibilityTest(unittest.TestCase):
    def test_tiny_exponent_no_overflow(self):
        """foc reads 1/D = (b/B)^(1/β), which underflows to 0 near N_min
        where D overflows; a fit with β = 0.02 solves without warnings."""
        ns = math_ns()
        fit = tuple(np.float64(x) for x in (1.07, 54.7, 8.94, 0.22, 0.02))
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            N, D = ns['solve_optimum'](np.float64(2.55), 1e11, *fit)
        self.assertTrue(np.isfinite(N) and np.isfinite(D))

    def test_reaches(self):
        code = page_cells()['budgets-ladder-optimum']
        ns = math_ns()
        exec(code[code.index('MAX_LOG ='):code.index('usable =')], ns)
        reaches = ns['reaches']
        self.assertTrue(reaches(3.27, *d2l.HOFFMANN_FIT))
        self.assertFalse(reaches(1.6, *d2l.HOFFMANN_FIT))  # E above it
        self.assertFalse(reaches(3.27, 1.69, 406.4, 410.7, 0.0, 0.28))
        self.assertFalse(reaches(3.27, 1.69, 406.4, 410.7, 0.05, 0.28))
        self.assertFalse(reaches(3.27, 1.69, 406.4, 0.5, 0.34, 0.28))  # D<1


class FormatTest(unittest.TestCase):
    def test_fmt_power(self):
        self.assertEqual(d2l.fmt_power(1e9), '10^9')
        self.assertEqual(d2l.fmt_power(7.2e11), '7×10^11')
        self.assertEqual(d2l.fmt_power(0), '0')
        self.assertEqual(d2l.fmt_power(2.34e12, 2, tex=True),
                         r'2.3\times10^{12}')
        self.assertEqual(d2l.fmt_power(1e11, tex=True), '10^{11}')

    def test_fmt_sig(self):
        self.assertEqual([d2l.fmt_sig(x) for x in (4367.6, 28.2, 0.374)],
                         ['4,368', '28', '0.37'])


def toy_packed():
    arch = d2l.Arch(L=2, d=32, H=4, H_kv=2, d_h=8, f=64, vocab=97, T=64,
                    rope_base=1e4, qk_norm=False, tie=True)
    torch.manual_seed(0)
    return d2l.PackedGPT.from_arch(arch).eval()


def toy_qwen():
    torch.manual_seed(0)
    return d2l.Qwen3GPT(97, 32, 4, 2, 2, head_dim=8, qk_norm=True,
                        ffn_hidden=64, max_len=64).eval()


class DecodeStepTest(unittest.TestCase):
    def check(self, model):
        ctx, steps, B, max_len = 9, 5, 3, 16
        X = torch.randint(0, 97, (B, ctx))
        ks, vs = model.init_cache(B, max_len)
        logits, ks, vs = model.forward_cached(X, ks, vs, 0)
        ks2, vs2 = ks.clone(), vs.clone()
        token = logits.argmax(-1, keepdim=True)
        tok2, pos = token.clone(), torch.tensor(ctx)
        step = d2l._decode_step(model, max_len)
        with torch.inference_mode():
            for t in range(ctx, ctx + steps):
                ref, ks, vs = model.forward_cached(token, ks, vs, t)
                got = step(tok2, ks2, vs2, pos)
                torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)
                token = ref.argmax(-1, keepdim=True)
                self.assertTrue(torch.equal(tok2, token))
            self.assertEqual(int(pos), ctx + steps)
        torch.testing.assert_close(ks2[:, :, :, :ctx + steps],
                                   ks[:, :, :, :ctx + steps])

    def test_packed(self):
        self.check(toy_packed())

    def test_qwen_qk_norm(self):
        self.check(toy_qwen())

    def test_estimate(self):
        self.assertEqual(d2l._decode_estimate(10, 4, 2, 8, 5),
                         (20 + 64 + 60) / 2**30)


class ShapesAndPendingTest(unittest.TestCase):
    def test_r2_shapes(self):
        s = d2l.decode_shapes(['r2'])
        self.assertEqual(int(s['kv_bytes'][0]), 18_432)
        self.assertEqual(int(s['params'][0]), 138_440_448)

    def test_qwen_shapes(self):
        if not qwen_configs_cached():
            self.skipTest('Qwen3 configs not cached')
        with in_chapter():
            s = d2l.decode_shapes(['Qwen/Qwen3-0.6B'])
        self.assertEqual(int(s['kv_bytes'][0]), 2 * 28 * 8 * 128 * 2)
        self.assertEqual(int(s['params'][0]), 596_049_920)

    def test_pending_table(self):
        with in_chapter(), mock.patch.dict(os.environ, {'D2L_RERUN': ''}), \
                uncommitted(), mock.patch.object(
                    d2l, 'PRETRAIN_PENDING',
                    d2l.PRETRAIN_PENDING | {'decode-rates'}), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            r = d2l.decode_rates(['r2'], (1, 8), 1024, 128)
        self.assertEqual(out.getvalue().strip(), '[decode-rates pending]')
        self.assertEqual(list(r['batch']), [1, 8])
        self.assertTrue(r['tok_s'].isna().all())
        self.assertEqual(list(r.columns), d2l._DECODE_COLUMNS)
        r['roofline'] = [1.0, 2.0]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            d2l.show_decode_table(r)
        self.assertEqual(len(out.getvalue().splitlines()), 3)

    def test_committed_table_checks_config(self):
        table = pd.DataFrame([dict(model='r2', params=1, kv_bytes=2, batch=1,
                                   tok_s=10.0, ms=100.0, ms_q1=99.0,
                                   ms_q3=101.0, peak_gib=1.0,
                                   stack='eager + CUDA graph', feasible=True,
                                   hf_ms=math.nan)])
        with in_chapter(), mock.patch.object(d2l, '_rerun_dir',
                                             return_value=self.tmp):
            path = os.path.join(self.tmp, 'lm-decode-rates-v1.json.gz')
            d2l._write_result(path, 'json-table', table, dict(
                config=dict(models=['r2'], batches=[1], ctx=1024,
                            new_tokens=128), label=''))
            hub = {'lm-decode-rates-v1': ('x', 'y')}
            with mock.patch.dict(d2l.DATA_HUB, hub), mock.patch.object(
                    d2l, 'download_keyed', return_value=path):
                r = d2l.decode_rates(['r2'], (1,), 1024, 128)
                self.assertEqual(float(r['tok_s'][0]), 10.0)
                with self.assertRaises(ValueError):
                    d2l.decode_rates(['r2'], (1,), 2048, 128)

    def setUp(self):
        import tempfile
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()


class HarnessCheckTest(unittest.TestCase):
    def test_slower_than_hf_stops(self):
        shapes = pd.DataFrame([dict(model='Qwen/Qwen3-0.6B', params=10,
                                    kv_bytes=4)])
        model = mock.Mock()
        model.token_emb.weight.shape = (97, 8)
        with mock.patch.object(d2l, 'decode_shapes', return_value=shapes), \
                mock.patch.object(d2l, '_decode_model', return_value=model), \
                mock.patch.object(d2l, '_time_decode',
                                  return_value=(30.0, 29.0, 31.0)), \
                mock.patch.object(d2l, '_hf_check_ms', return_value=25.0), \
                mock.patch.object(torch.cuda, 'reset_peak_memory_stats'), \
                mock.patch.object(torch.cuda, 'max_memory_reserved',
                                  return_value=2**30):
            with self.assertRaises(RuntimeError):
                d2l._decode_rates(['Qwen/Qwen3-0.6B'], (1,), 16, 4)
            d2l._hf_check_ms.return_value = 60.0
            table = d2l._decode_rates(['Qwen/Qwen3-0.6B'], (1,), 16, 4)
        self.assertEqual(float(table['hf_ms'][0]), 60.0)


class LadderTest(unittest.TestCase):
    def test_plan(self):
        plan = d2l._ladder_plan()
        fin = plan[plan['finished']]
        self.assertEqual(len(fin), 12)
        self.assertEqual(len(plan) - len(fin), 8)
        r2 = sorted(fin[fin['size'] == 'r2']['tokens'])
        self.assertEqual(r2, [786_432_000, 1_572_864_000, 3_145_728_000])
        n = dict(zip(fin['size'], fin['N']))
        self.assertEqual(n, {'ladder-25m': 25_172_352,
                             'ladder-54m': 54_275_904,
                             'ladder-88m': 88_096_512, 'r2': 138_440_448})

    def test_simulated(self):
        with mock.patch.object(d2l, 'PRETRAIN_PENDING',
                               d2l.PRETRAIN_PENDING | C1), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            a, b = d2l.ladder_points(), d2l.ladder_points()
        self.assertIn('simulated at the Hoffmann fit', out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertTrue(d2l.show_ladder_label(a))
        self.assertEqual(out.getvalue().strip(),
                         f'[{d2l.SIMULATED_LADDER}]')
        self.assertEqual(len(a), 12)
        self.assertTrue(a['finished'].all())
        pd.testing.assert_frame_equal(a, b)
        self.assertEqual(list(a.columns), d2l._LADDER_COLUMNS)
        self.assertEqual(list(dict.fromkeys(a['size'])),
                         list(d2l.LADDER_SIZES))

    def test_absent_raises(self):
        pending = d2l.PRETRAIN_PENDING - C1
        with mock.patch.object(d2l, 'PRETRAIN_PENDING', pending), \
                mock.patch.dict(os.environ, {'D2L_RERUN': ''}):
            with self.assertRaises(KeyError):
                d2l.ladder_points()

    def test_committed(self):
        rows = [dict(size=s, P=n, run=s, step=1, tokens=t, tokens_per_param=0,
                     finished=f, val_loss=3.0 + i, bpb=1.0)
                for i, (s, n, t, f) in enumerate(
                    d2l._ladder_plan().itertuples(index=False))]
        Evals = d2l.LadderEvals
        fake = Evals(pd.DataFrame(rows), pd.DataFrame(), 'smoke')
        hub = {d2l.PRETRAIN_ARTIFACTS['ladder-evals'][0]: ('x', 'y')}
        with mock.patch.dict(d2l.DATA_HUB, hub), mock.patch.object(
                d2l, 'load_artifact', return_value=fake), \
                contextlib.redirect_stdout(io.StringIO()):
            pts = d2l.ladder_points()
        self.assertEqual(len(pts), 20)
        self.assertTrue(pts['finished'][:12].all())
        self.assertFalse(pts['finished'][12:].any())
        np.testing.assert_allclose(pts['tokens_per_param'],
                                   pts['tokens'] / pts['N'])
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertFalse(d2l.show_ladder_label(pts))
        self.assertEqual(out.getvalue().strip(), '[smoke]')
        finished = Evals(pd.DataFrame([r for r in rows if r['finished']]),
                         pd.DataFrame(), '')
        with mock.patch.dict(d2l.DATA_HUB, hub), mock.patch.object(
                d2l, 'load_artifact', return_value=finished), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(ValueError):  # no plateau points
                d2l.ladder_points()


class TrainingRateTest(unittest.TestCase):
    def test_from_log(self):
        arch = d2l.r2_config('r2').arch
        log = pd.DataFrame(dict(tok_s=[1e6, 2e6, 2e6], world=[4, 4, 4],
                                mfu=[0.3, 0.35, 0.35]))
        log.attrs['config'] = dict(runs=[dict(arch=arch._asdict())])
        r = d2l.training_rate(log)
        self.assertAlmostEqual(r.seconds, 4 / 2e6)
        self.assertEqual(r.N, d2l.flops_per_token(arch) / 6)
        self.assertAlmostEqual(r.mfu, 0.35)

    def test_pending(self):
        with contextlib.redirect_stdout(io.StringIO()) as out:
            r = d2l.training_rate(pd.DataFrame())
        self.assertIn('MFU 0.40 assumed', out.getvalue())
        peak = d2l.GPU_PEAK_BF16['NVIDIA GeForce RTX 4090']
        self.assertAlmostEqual(r.seconds, 6 * r.N / (0.4 * peak))


class JobsTest(unittest.TestCase):
    def test_call(self):
        cells = page_cells()
        code = cells['budgets-decode-roofline']
        m = re.search(r'^MODELS = (\(.*?\))$', code, re.S | re.M)
        call = d2l._decode_call()
        self.assertEqual(eval(m.group(1)), call['models'])
        self.assertIn('CTX = 1024', code)
        self.assertEqual(call['ctx'], 1024)
        self.assertIn('NEW_TOKENS = 128', cells['budgets-decode-rates'])
        self.assertIn("d2l.decode_rates(MODELS, batches=(1, 8, 32, 128), "
                      "ctx=CTX,\n                         "
                      "new_tokens=NEW_TOKENS)", cells['budgets-decode-rates'])
        self.assertEqual(call['batches'], (1, 8, 32, 128))
        self.assertEqual(call['new_tokens'], 128)
        self.assertEqual(d2l.PRETRAIN_JOBS['decode-rates'][0], 'decode_rates')
        self.assertEqual(d2l.PRETRAIN_JOBS['decode-rates'][1](), call)
        self.assertIn('decode-rates', d2l.PRETRAIN_RERUN_NAMES)


class PageTest(unittest.TestCase):
    """The whole page on the pending path (C1 and the decode rates
    pending), on the CPU: every cell runs, every output is at most 15
    lines of at most 100 columns, and cells 12-17 take under 90 s."""
    def test_pending_page(self):
        if not qwen_configs_cached():
            self.skipTest('Qwen3 configs not cached')
        ns, fit_cells, t_fit, outs = {}, False, 0.0, {}
        with in_chapter(), mock.patch.object(  # decode rates pending too
                d2l, 'PRETRAIN_PENDING',
                d2l.PRETRAIN_PENDING | C1 | {'decode-rates'}), \
                uncommitted(), mock.patch.dict(os.environ, {'D2L_RERUN': ''}):
            for name, code in page_cells().items():
                fit_cells |= name == FIT_FIRST
                t = time.time()
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    exec(code, ns)
                t_fit += (time.time() - t) if fit_cells else 0.0
                outs[name] = out.getvalue()
                lines = out.getvalue().splitlines()
                self.assertLessEqual(len(lines), 15, name)
                for line in lines:
                    self.assertLessEqual(len(line), 100, (name, line))
                d2l.plt.close('all')
        self.assertLess(t_fit, 90)
        self.assertEqual(ns['dof'], 7)
        label = f'[{d2l.SIMULATED_LADDER}]'
        for name in FIT_CELLS[2:]:  # every fit cell's output carries it
            self.assertTrue(outs[name].startswith(label), name)
        self.assertIn('no verdict', outs['budgets-intervals'])
        self.assertNotIn('inside the 95% interval', outs['budgets-intervals'])
        self.assertIn('simulated ladder: no plateau points to add',
                      outs['budgets-plateau'])
        self.assertIn('roofline × 1 (rates pending)',
                      outs['budgets-measured-optimum'])

    def test_ill_identified_ladder_time(self):
        """A ladder like the smoke run's (exponents poorly pinned, so the
        refits run long): fit, recovery and bootstrap at the spec's fallback
        of 500 resamples take under 90 s."""
        plan = d2l._ladder_plan()
        fin = plan[plan['finished']].reset_index(drop=True)
        N, D = fin['N'].to_numpy(float), fin['tokens'].to_numpy(float)
        smoke_fit = (1.07, 54.7, 8.94, 0.22, 0.12)  # the smoke ladder's fit
        E, A, B, alpha, beta = smoke_fit
        fin['val_loss'] = (E + A / N ** alpha + B / D ** beta
                           + np.random.default_rng(0).normal(0, 0.06, len(N)))
        ns = math_ns()
        ns['fin'] = d2l._with_label(fin, 'smoke')
        cells = page_cells()
        t = time.time()
        with contextlib.redirect_stdout(io.StringIO()) as out:
            for name in FIT_CELLS[:5]:
                code = cells[name].replace('RESAMPLES = 1000',
                                           'RESAMPLES = 500')
                exec(code, ns)
        t = time.time() - t
        self.assertIn('500 resamples', out.getvalue())
        self.assertIn('cannot pin', out.getvalue())
        self.assertLess(t, 90)


if __name__ == '__main__':
    unittest.main()
