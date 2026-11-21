#!/usr/bin/env python3
"""CPU tests for sec_lm_data_mixtures (chapter_lm-data/data-mixtures.md),
the shared part written before the section: the FineWeb-Edu bucket rule
(golden ids, the split ranges partition 0-255), `bpb_from_nll` on a
uniform byte model, `token_byte_lengths` for byte-level, SentencePiece-style
and Transformers-wrapped tokenizers (the byte lengths of a text's ids sum to
its UTF-8 length), `load_pool` on fixture pools (one pool, several pools,
a length mismatch), and the chapter's source-file order. The section's own code: temperature sampling, effective tokens
and the mixture sampler; the proxy stream (determinism, stratified domain
counts, epochs, the pool guard, stages), train_proxy_lm (determinism,
stop_tokens, checkpoint and resume across an epoch boundary), bits per byte
of a uniform model through _proxy_eval; the D4 noise floor and the
mixing-law recovery of fit_mixture_regression (and its warm start),
choose_mixture (also bounded by the hull of the training mixtures), the run
design with its fixed corner mixtures; DoReMi's
closed-form step, the saddle of a toy game and the per-token clipping; and
tools/run_mix_campaign.py on the test configuration 'tiny' (CPU): its smoke
(kill and resume) and the first runs of a campaign."""
import hashlib
import json
import math
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
import unicodedata

import numpy as np
import torch
from scipy import optimize

from d2l import torch as d2l

ROOT = pathlib.Path(__file__).resolve().parents[1]
CPU = torch.device('cpu')
TINY = d2l.ProxyCfg(32, 2, 2, seq_len=16, batch=8, micro_batch=4,
                    vocab_size=64, dtype='float32')

TEXTS = ['The quick brown fox jumps over 13 lazy dogs.',
         'Grüße aus Köln: 3.14159 ist π, und 中文 hat keine Leerzeichen.',
         'def f(x):\n    return x ** 2  # squares\n',
         'Café naïve résumé — façade.'] * 50


class BucketTest(unittest.TestCase):
    def test_golden(self):
        golden = {'<urn:uuid:00000000-0000-0000-0000-000000000000>': 87,
                  '<urn:uuid:4ad6d48e-21c8-4a6f-bd16-3f43ce4f5e6b>': 158,
                  '<urn:uuid:ffffffff-ffff-ffff-ffff-ffffffffffff>': 56}
        for doc_id, b in golden.items():
            self.assertEqual(d2l.fwedu_bucket(doc_id), b)

    def test_partition(self):
        seen = [b for r in d2l.FWEDU_BUCKETS.values() for b in r]
        self.assertEqual(sorted(seen), list(range(256)))
        self.assertEqual(list(d2l.FWEDU_BUCKETS['val']), [0])
        self.assertEqual(list(d2l.FWEDU_BUCKETS['side']), list(range(1, 8)))
        counts = np.bincount([d2l.fwedu_bucket(f'<urn:uuid:{i}>')
                              for i in range(25600)], minlength=256)
        self.assertLess(abs(counts.mean() - 100), 1e-9)
        self.assertLess(counts.std(), 20)  # about sqrt(100)

    def test_files(self):
        names = [f.split('/')[-1][:3] for f in d2l._FWEDU_FILES]
        self.assertEqual(names, ['000', '001', '002', '003', '004', '006',
                                 '007', '008', '009', '010', '011'])


class BpbTest(unittest.TestCase):
    def test_uniform(self):
        n = 1000  # 1,000 byte tokens under a uniform model over 256 bytes
        self.assertAlmostEqual(d2l.bpb_from_nll(n * math.log(256), n), 8.0,
                               delta=1e-9)
        self.assertAlmostEqual(d2l.bpb_from_nll(math.log(2) * 3, 3), 1.0,
                               delta=1e-12)


class ByteLengthTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok = d2l.train_hf_bpe(TEXTS, 400, d2l.R2_SPECIALS)

    def check(self, lengths, encode):
        for t in TEXTS[:4]:
            t = unicodedata.normalize('NFC', t)
            self.assertEqual(int(lengths[encode(t)].sum()),
                             len(t.encode('utf-8')), t)

    def test_byte_level(self):
        lengths = d2l.token_byte_lengths(self.tok)
        self.assertEqual(len(lengths), 400)
        self.assertTrue((lengths[:32] == 0).all())
        self.assertTrue((lengths[32:] > 0).all())
        self.check(lengths, lambda t: self.tok.encode(t).ids)

    def test_transformers_wrapper(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, 'tokenizer.json')
            with open(p, 'w', encoding='utf-8') as f:
                f.write(self.tok.to_str(pretty=False))
            wrapped = d2l._r2_from_file(p)
        lengths = d2l.token_byte_lengths(wrapped)
        self.assertTrue((lengths == d2l.token_byte_lengths(self.tok)).all())
        self.check(lengths, lambda t: wrapped.encode(
            t, add_special_tokens=False))

    def test_sentencepiece_style(self):
        from tokenizers import Tokenizer, models
        vocab = {'▁hello': 0, '<0x41>': 1, 'é': 2, '<unk>': 3, '<s>': 4}
        tok = Tokenizer(models.WordLevel(vocab, unk_token='<unk>'))
        tok.add_special_tokens(['<s>'])
        self.assertEqual(list(d2l.token_byte_lengths(tok)), [6, 1, 2, 5, 0])


class PoolTest(unittest.TestCase):
    def test_load_pool(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids = np.arange(100, dtype=np.uint16)
            ids.tofile(os.path.join(tmp, 'web-1k.bin'))
            meta = dict(n_tokens=100, n_docs=3, dtype='uint16', source='x')
            with open(os.path.join(tmp, 'meta.json'), 'w') as f:
                json.dump(meta, f)
            for arg in (tmp, os.path.join(tmp, 'web-1k.bin')):
                pool = d2l.load_pool(arg)
                self.assertEqual(pool.meta, meta)
                self.assertTrue((pool.ids == ids).all())
                with self.assertRaises(ValueError):
                    pool.ids[0] = 1  # read-only
            np.arange(50, dtype=np.uint16).tofile(os.path.join(tmp,
                                                               'math-1k.bin'))
            with self.assertRaisesRegex(ValueError, '2 pools'):
                d2l.load_pool(tmp)
            with open(os.path.join(tmp, 'meta.json'), 'w') as f:
                json.dump({'web-1k.bin': meta,
                           'math-1k.bin': dict(meta, n_tokens=50)}, f)
            self.assertEqual(len(d2l.load_pool(tmp, 'math-1k').ids), 50)
            with open(os.path.join(tmp, 'meta.json'), 'w') as f:
                json.dump(dict(meta, n_tokens=99), f)
            with self.assertRaisesRegex(ValueError, 'meta says'):
                d2l.load_pool(tmp, 'web-1k')


def toy_pools(sizes=(3000, 2000, 1000), seed=0, vocab=64):
    """Random token pools (ids 1..vocab-1, an EOS 0 every 50 tokens)."""
    rng = np.random.default_rng(seed)
    out = {}
    for d, n in zip(d2l.MIX_DOMAINS, sizes):
        ids = rng.integers(1, vocab, n).astype(np.uint16)
        ids[49::50] = 0
        out[d] = d2l.TokenPool(ids, {})
    return out


class MixDefsTest(unittest.TestCase):
    def test_temperature(self):
        n = np.array([1000, 100, 10])
        for alpha in (0.0, 0.3, 0.7, 1.0):
            p = d2l.temperature_weights(n, alpha)
            self.assertAlmostEqual(p.sum(), 1, delta=1e-12)
            e = p / n
            self.assertTrue(alpha == 1 or e[2] > e[1] > e[0])
        self.assertTrue(np.allclose(d2l.temperature_weights(n, 1), n / 1110))
        self.assertTrue(np.allclose(d2l.temperature_weights(n, 0), 1 / 3))
        d = d2l.temperature_weights({'a': 4, 'b': 1}, 0.5)
        self.assertEqual(list(d), ['a', 'b'])
        self.assertAlmostEqual(d['a'], 2 / 3, delta=1e-12)

    def test_effective_tokens(self):
        U = 1e9
        self.assertAlmostEqual(d2l.effective_tokens(U, 4) / (4 * U), 0.932,
                               delta=0.005)
        self.assertAlmostEqual(d2l.effective_tokens(U, 1001) / (16.4 * U), 1,
                               delta=1e-3)
        self.assertEqual(d2l.effective_tokens(U, 1), U)
        self.assertEqual(d2l.effective_tokens(U, 0.5), U / 2)
        E = np.linspace(1, 10, 10)
        self.assertTrue((np.diff(d2l.effective_tokens(U, E)) > 0).all())
        self.assertTrue((d2l.effective_tokens(U, E) <= U * E + 1e-6).all())

    def test_sample_mixtures(self):
        for conc in (0.3, 1.0):
            W = d2l.sample_mixtures(200, 'abc', conc, 0, max_weight=0.8,
                                    min_weight=0.01)
            self.assertEqual(W.shape, (200, 3))
            self.assertTrue(np.allclose(W.sum(1), 1, atol=1e-12))
            self.assertTrue((W.max(1) <= 0.8).all() and (W >= 0.01).all())
            self.assertTrue((W == d2l.sample_mixtures(
                200, 'abc', conc, 0, 0.8, 0.01)).all())


class StreamTest(unittest.TestCase):
    def test_determinism(self):
        pools, w = toy_pools(), {'web': 0.5, 'code': 0.3, 'math': 0.2}
        h = lambda s: [hashlib.sha256(s.batch(k)[0].tobytes()).hexdigest()
                       for k in range(10)]
        a, b, c = (d2l._ProxyStream(pools, w, 2**12, 16, 8, seed)
                   for seed in (0, 0, 1))
        self.assertEqual(h(a), h(b))
        self.assertNotEqual(h(a), h(c))
        X, dom = a.batch(3)  # each row is a window of its domain's pool
        L = 16
        for x, i in zip(X, dom):
            ids = np.asarray(pools[a.names[i]].ids)
            starts = [j * L for j in range((len(ids) - 1) // L)
                      if (ids[j * L:j * L + L + 1] == x).all()]
            self.assertTrue(starts)

    def test_stratified(self):
        rng = np.random.default_rng(1)
        pools = toy_pools((20000, 20000, 20000))
        for _ in range(5):
            w = dict(zip(d2l.MIX_DOMAINS, rng.dirichlet([0.5] * 3)))
            s = d2l._ProxyStream(pools, w, 8 * 16 * 100, 16, 8, 0)
            for k in range(1, s.steps + 1):
                counts = np.bincount(s.dom[:8 * k], minlength=3)
                expect = 8 * k * np.array(list(w.values()))
                self.assertTrue((np.abs(counts - expect) <= 1 + 1e-9).all())

    def test_epochs(self):
        pools = {'web': toy_pools()['web']}
        L, W = 16, (3000 - 1) // 16
        E = 3
        s = d2l._ProxyStream(pools, {'web': 1.0}, E * W * L, L, 1, 0,
                             max_epochs=E)
        seen = [s.window(0, o) for o in s.ordinal[np.argsort(s.ordinal)]]
        for e in range(E):
            self.assertEqual(sorted(seen[e * W:(e + 1) * W]), list(range(W)))
        self.assertEqual(s.epoch_starts['web'], [0, W, 2 * W])
        self.assertAlmostEqual(s.epochs['web'], E)
        with self.assertRaisesRegex(ValueError, 'smaller than the run'):
            d2l._ProxyStream(pools, {'web': 1.0}, E * W * L, L, 1, 0)

    def test_stages(self):
        pools = toy_pools((20000, 20000, 20000))
        u = dict(zip(d2l.MIX_DOMAINS, [1 / 3] * 3))
        s = d2l._ProxyStream(pools, u, 8 * 16 * 100, 16, 8, 0,
                             stages=[(0.8, {'web': 1.0})])
        self.assertTrue((s.dom[8 * 80:] == 0).all())
        self.assertEqual(len(set(s.dom[:8 * 80].tolist())), 3)
        with self.assertRaisesRegex(ValueError, 'not a mixture'):
            d2l._ProxyStream(pools, {'web': 0.5}, 2**12, 16, 8, 0)


class TrainProxyTest(unittest.TestCase):
    def run_proxy(self, **kw):
        pools, w = toy_pools(), {'web': 0.5, 'code': 0.3, 'math': 0.2}
        return d2l.train_proxy_lm(pools, w, 8 * 16 * 30, TINY, seed=0,
                                  log_every=1, device=CPU, **kw)

    def test_determinism_and_prefix(self):
        a, b = self.run_proxy(), self.run_proxy()
        self.assertLess(abs(a['log'][19][2] - b['log'][19][2]), 1e-6)
        c = self.run_proxy(stop_tokens=8 * 16 * 12)
        self.assertEqual(c['stopped_at'], 12)
        for x, y in zip(a['log'][:12], c['log']):
            self.assertLess(abs(x[2] - y[2]), 1e-6)

    def test_resume_across_an_epoch(self):
        pools = {'web': toy_pools()['web']}  # 187 windows: an epoch is 23 steps
        run = lambda **kw: d2l.train_proxy_lm(
            pools, {'web': 1.0}, 8 * 16 * 40, TINY, seed=0, log_every=1,
            max_epochs=2, device=CPU, **kw)
        full = run()
        self.assertEqual(full['epoch_starts']['web'][:2], [0, 23])
        with tempfile.TemporaryDirectory() as tmp:
            ck = os.path.join(tmp, 'ck.pt')
            run(stop_tokens=8 * 16 * 20, ckpt=ck)
            rest = run(resume=ck)
        for x, y in zip(full['log'], rest['log']):
            self.assertEqual(x[0], y[0])
            self.assertLess(abs(x[2] - y[2]), 1e-6)

    def test_uniform_bpb(self):
        cfg = TINY._replace(dtype='float64', vocab_size=32768)
        model = d2l._proxy_model(cfg).double()
        torch.nn.init.zeros_(model.token_emb.weight)
        ids = toy_pools()['web'].ids
        lengths = np.random.default_rng(0).integers(1, 5, 64)
        r = d2l._proxy_eval(model, ids, cfg, lengths, 10, CPU)
        self.assertEqual(r['n_tokens'], 160)
        self.assertAlmostEqual(r['bpb'] * r['n_bytes'] / r['n_tokens'], 15,
                               delta=1e-9)


class RegressionTest(unittest.TestCase):
    def test_noise_floor(self):
        rng = np.random.default_rng(0)
        Wd = d2l.sample_mixtures(16, 'abc', 1.0, 0)
        Wn = d2l.sample_mixtures(200, 'abc', 1.0, 1)
        beta, sigma = np.array([1.2, -0.3, 0.2]), 0.02
        X, Xn = (np.hstack([np.ones((len(w), 1)), w[:, :2]]) for w in (Wd, Wn))
        err = np.zeros(len(Wn))
        for _ in range(5000):
            y = X @ beta + sigma * rng.normal(size=16)
            f = d2l.fit_mixture_regression(Wd, y[:, None], 'ridge', alpha=0.0)
            err += (f(Wn)[:, 0] - Xn @ beta
                    - sigma * rng.normal(size=len(Wn)))**2
        theory = sigma**2 * (1 + np.einsum('ij,jk,ik->i', Xn,
                                           np.linalg.inv(X.T @ X), Xn))
        self.assertLess(abs((err / 5000).mean() / theory.mean() - 1), 0.03)
        H = X @ np.linalg.inv(X.T @ X) @ X.T  # p / n at the design points
        self.assertAlmostEqual(np.trace(H) / 16, 3 / 16, delta=1e-12)

    def test_mixlaw_recovery_and_choice(self):
        W = d2l.sample_mixtures(16, 'abc', 1.0, 0)
        c, k = np.array([1.0, 0.8, 0.9]), np.array([0.5, 0.4, 0.6])
        T = np.array([[-3, 0, 0], [0, -4, 0], [0, 0, -2.5]])
        law = lambda w: c + k * np.exp(np.atleast_2d(w) @ T.T)
        f = d2l.fit_mixture_regression(W, law(W), 'mixlaw')
        Wn = d2l.sample_mixtures(50, 'abc', 1.0, 1)
        self.assertLess(np.abs(f(Wn) - law(Wn)).max(), 1e-6)
        self.assertEqual(f(Wn[0]).shape, (3,))
        t = np.array([0.7, 0.15, 0.15])
        w, v = d2l.choose_mixture(f, t)
        best = optimize.minimize(
            lambda z: float(law(np.exp(z) / np.exp(z).sum())[0] @ t),
            np.zeros(3), method='Nelder-Mead', options=dict(xatol=1e-10,
                                                            fatol=1e-12))
        self.assertLess(abs(v - best.fun), 1e-6)
        self.assertLessEqual(w.max(), 0.9)
        lin = d2l.fit_mixture_regression(W, law(W), 'ridge')
        self.assertLessEqual(d2l.choose_mixture(lin, t)[0].max(), 0.9)
        gbm = d2l.fit_mixture_regression(W, law(W), 'gbm')
        self.assertEqual(gbm(Wn).shape, (50, 3))
        with self.assertRaises(ValueError):
            d2l.fit_mixture_regression(W, law(W), 'lasso')
        # a warm start alone reproduces the fit
        g = d2l.fit_mixture_regression(W, law(W), 'mixlaw', init=f.params,
                                       starts=1)
        self.assertLess(np.abs(g(Wn) - f(Wn)).max(), 1e-9)

    def test_choice_within_the_hull(self):
        from scipy.spatial import Delaunay
        W = d2l.sample_mixtures(12, 'abc', 1.0, 0, 0.6)  # no weight above 0.6
        lin = lambda w: np.atleast_2d(w) @ np.diag([1.0, 2.0, 3.0])
        f = lambda w: lin(w) if np.ndim(w) == 2 else lin(w)[0]
        t = np.array([1.0, 1.0, 1.0])
        free, _ = d2l.choose_mixture(f, t)  # linear: the cap corner
        self.assertAlmostEqual(free[0], 0.9, delta=1e-6)
        w, v = d2l.choose_mixture(f, t, within=W)
        self.assertGreaterEqual(Delaunay(W[:, :-1]).find_simplex(w[:-1]), 0)
        best = min(float(f(x) @ t) for x in W)  # a linear optimum: a vertex
        self.assertLessEqual(v, best + 1e-9)

    def test_run_design(self):
        for k, n in (('t1', 41), ('t2', 49)):
            c = d2l.MIX_CONFIGS[k]
            specs = d2l.mix_run_specs(c)
            self.assertEqual(len(specs), n)
            train = [s['weights'] for s in specs if s['arm'] == 'train']
            self.assertEqual(train[-3:], [list(e) for e in d2l.MIX_EDGES])
            self.assertEqual(len({s['run_id'] for s in specs}), n)


class DoremiTest(unittest.TestCase):
    def test_closed_form_step(self):
        a_t, g, eta = np.array([0.5, 0.3, 0.2]), np.array([0.4, 1.0, -0.2]), 0.7
        sm = lambda z: np.exp(z - z.max()) / np.exp(z - z.max()).sum()
        obj = lambda z: -(sm(z) @ g - np.sum(sm(z) * np.log(sm(z) / a_t))
                          / eta)
        num = sm(optimize.minimize(obj, np.zeros(3), method='BFGS',
                                   options=dict(gtol=1e-12)).x)
        self.assertLess(np.abs(num - d2l._doremi_step(a_t, g, eta, 0)).max(),
                        1e-8)
        a = d2l._doremi_step(a_t, g, eta, 0.1)
        self.assertAlmostEqual(a.sum(), 1, delta=1e-12)
        self.assertGreaterEqual(a.min(), 0.1 / 3)

    def test_saddle(self):
        c, theta, alpha, path = np.array([0.0, 1.0, 3.0]), 0.0, np.ones(3) / 3, []
        for _ in range(20000):
            alpha = d2l._doremi_step(alpha, (theta - c)**2, 0.05, 1e-3)
            theta -= 0.01 * 2 * alpha @ (theta - c)
            path.append(alpha)
        self.assertLess(np.abs(np.mean(path, 0) - [0.5, 0, 0.5]).max(), 0.02)

    def test_per_token_clipping(self):
        nll_p, nll_r = torch.tensor([[2.0, 0.0]]), torch.tensor([[1.0, 1.0]])
        self.assertEqual(d2l._doremi_excess(nll_p, nll_r), 0.5)
        self.assertEqual(max((nll_p - nll_r).mean().item(), 0), 0.0)
        ref, prox = d2l._proxy_model(TINY, 0), d2l._proxy_model(TINY, 1)
        X = torch.randint(0, 64, (3, 17), generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            want = d2l._doremi_excess(*(d2l._per_token_nll(m, X, TINY)
                                        for m in (prox, ref)))
        w, trace = d2l.doremi_weights(ref, prox, lambda k: {'a': X, 'b': X},
                                      2, TINY, eta=1.0, smooth=0.0)
        self.assertAlmostEqual(trace[0]['excess'][0], want, delta=1e-6)
        self.assertTrue(np.allclose(w, 0.5))  # equal domains, equal weights


def fixture_campaign_pools(root, tok_path):
    """Pools of the 'tiny' launcher configuration (t1 file names) in
    root/tiny, built from random ids and stamped with the fixture
    tokenizer's hash."""
    sha = hashlib.sha256(pathlib.Path(tok_path).read_bytes()).hexdigest()
    folder = pathlib.Path(root) / 'tiny'
    folder.mkdir(parents=True)
    rng, meta = np.random.default_rng(0), {}
    for d in d2l.MIX_DOMAINS:
        for name, n in ((f'{d}-20m', 70_000), (f'{d}-val-1m', 4_000)):
            ids = rng.integers(32, 400, n).astype(np.uint16)
            ids[99::100] = 0
            ids.tofile(folder / f'{name}.bin')
            meta[f'{name}.bin'] = dict(
                domain=d, split='val' if 'val' in name else 'pool',
                n_tokens=n, n_docs=int((ids == 0).sum()), eos_id=0,
                n_bytes_nfc=0, n_bytes_scored=0, dtype='uint16',
                tokenizer_sha256=sha, sha256='', source='fixture',
                revision='', files={})
    (folder / 'meta.json').write_text(json.dumps(meta))


class CampaignTest(unittest.TestCase):
    def test_tiny_smoke_and_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tok = d2l.train_hf_bpe(TEXTS, 400, d2l.R2_SPECIALS)
            tok_path = os.path.join(tmp, 'tokenizer.json')
            with open(tok_path, 'w', encoding='utf-8') as f:
                f.write(tok.to_str(pretty=False))
            fixture_campaign_pools(os.path.join(tmp, 'mix'), tok_path)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                       PYTHONPATH=str(ROOT))
            out = os.path.join(tmp, 'out')
            base = [sys.executable, str(ROOT / 'tools' / 'run_mix_campaign.py'),
                    '--config', 'tiny', '--gpus', 'cpu', '--out', out,
                    '--pools', os.path.join(tmp, 'mix'), '--tokenizer',
                    tok_path]
            r = subprocess.run(base + ['--smoke'], env=env,
                               capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            smoke = json.loads(pathlib.Path(out, 'smoke-tiny.json').read_text())
            self.assertLessEqual(smoke['resume_fp32_max_diff'], 1e-6)
            self.assertTrue(smoke['resume_bf16_ok'])
            self.assertEqual(smoke['resume_bf16_max_diff'], 0.0)  # CPU
            self.assertEqual(smoke['campaign_runs'], 49)
            self.assertFalse(smoke['campaign_compile'])  # no compile on CPU
            self.assertGreater(smoke['predicted_hours_here'], 0)
            runs = 'lr-5e-04,lr-1e-03,lr-2e-03,lr-4e-03,pure-web'
            r = subprocess.run(base + ['--runs', runs], env=env,
                               capture_output=True, text=True, timeout=900)
            self.assertEqual(r.returncode, 0, r.stderr[-3000:])
            rec = json.loads(pathlib.Path(out, 'runs',
                                          'pure-web.json').read_text())
            choice = json.loads(pathlib.Path(out, 'lr_choice.json').read_text())
            self.assertEqual(rec['lr'], choice['chosen'])
            self.assertEqual(rec['weights'], [1.0, 0.0, 0.0])
            self.assertEqual(sorted(rec['final']), sorted(d2l.MIX_DOMAINS))
            self.assertEqual(len(rec['evals']), 8)
            self.assertEqual(rec['final']['web']['n_tokens'],  # all 62 windows
                             (4000 - 1) // 64 * 64)
            r = subprocess.run(base + ['--runs', runs], env=env,
                               capture_output=True, text=True, timeout=900)
            self.assertNotEqual(r.returncode, 0)  # the plan exists
            self.assertIn('--resume', r.stderr)


if __name__ == '__main__':
    unittest.main()
