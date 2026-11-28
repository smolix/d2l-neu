#!/usr/bin/env python3
"""CPU tests for sec_lm_long_context (chapter_lm-pretraining/long-context.md
and long-context-lib.md): `rope_inv_freq` checks (a)-(g) of the blueprint
(scale 1 gives the base frequencies; interpolation is division by s;
NTK-aware: first pair unchanged, last pair divided by s, the overhang of
0.43 at m = 23 on 8 of the 9 slow pairs; YaRN pair counts 9/12/11 at 4K and
14/12/6 at 1K; `yarn_mscale(4)`; `rope_mscale` scales the attention logits
by m^2 with the softmax scale untouched; gated: Transformers' YaRN
initialization with truncate=False to 1e-6). The needle probe: the word
pools (disjoint, single tokens in the form the window uses, a word whose
bare or leading-space form is several tokens left out), the windows (gaps
in (32, L - 64], needles at least 16 text tokens apart after line breaks,
keys and values absent from their window's text, deterministic), the
pilot's diagnostics on a fixture word tokenizer (and each check failing on
a window built to fail it), one prefill plus the queries in the same two
cache slots equal to full forwards on a tiny model (fp32 1e-5, also under
YaRN scaling), `reach` on synthetic Wilson tables (first bin failing: 0;
all passing: the top edge; a failing middle bin; strict 0.25), the octave
bins partition the queries (edges inclusive above), the format items
(oracle 100%, the cached candidate scorer against full forwards); the
lc-probe computation (checkpoints from their registered keys only, never a
job folder or a local next version; the keys and sha1 in the stored
configuration; an oracle miss stops it; the stored layout on toys) and the
reach-train plot (one recoloured line per checkpoint, legend beside). The
loss measurements (`nll_by_position` buckets average to the total;
`long_context_gain` on a synthetic repeat; `sliding_nlls` against an
explicit short context; `context_nlls` against `sliding_nlls` on the whole
window; `repeat_gaps` by hand); the staged stream; the arm configurations
and their identity (`evaluate` outside it, the seed-0 recipe check); the
scheme difference's two intervals (over seeds; over windows with the seeds
held fixed, the wider printed); the
periodic needle evaluation leaves the torch, NumPy and Python generators,
the stream state and the next batch untouched, and an arm trains to the
same losses and weights with and without it; a toy T1 campaign through
`lc_arms` (job folders, the merge, the stored result, the page's displays
of a T2 result on it as the labelled stand-in); a toy T2 campaign through
the part machinery (`--only t2-base` keeps one model's evaluation in its
job folder, the arms as part runs, the merge reading the kept evaluation);
the old timeline format is not read; the page's CPU cells; the page's later
cells on pending results; the regeneration jobs against the page's
literals; the local test books (100 whole books, prefix-stable windows).
"""
import contextlib
import io
import json
import math
import os
import random
import re
import tempfile
import unittest
from unittest import mock

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from d2l import torch as d2l

torch.set_num_threads(min(8, torch.get_num_threads()))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAPTER = os.path.join(ROOT, 'chapter_lm-pretraining')
PAGE = os.path.join(CHAPTER, 'long-context.md')
V = 32768  # the reference tokenizer's vocabulary
TOY = d2l.Arch(L=2, d=32, H=4, H_kv=2, d_h=8, f=64, vocab=V, T=64,
               rope_base=1e4, qk_norm=False, tie=True)


def page_cells():
    with open(PAGE) as f:
        src = f.read()
    return dict(re.findall(r'```\{\.python \.input #([^}]*)\}\n'
                           r'%%tab pytorch\n(.*?)```', src, re.S))


def run_cell(name, ns):
    with contextlib.redirect_stdout(io.StringIO()) as out:
        exec(page_cells()[name], ns)
    return out.getvalue()


def page_ns(**kw):
    return dict(math=math, np=np, pd=pd, torch=torch, d2l=d2l, **kw)


def toy_model(seed=0, max_len=64):
    torch.manual_seed(seed)
    return d2l.PackedGPT.from_arch(TOY._replace(T=max_len)).eval()


_TOK = []


def r2_tok():
    if not _TOK:
        try:
            with d2l._quiet():
                _TOK.append(d2l.r2_tokenizer())
        except Exception as e:  # noqa: BLE001
            raise unittest.SkipTest(f'reference tokenizer unavailable: {e}')
    return _TOK[0]


def r2_words(tok):
    """Lowercase words that are one reference token with a leading space
    (ids 1,000-6,000), the stand-in for _needle_words (no shard read)."""
    texts = tok.batch_decode([[i] for i in range(1000, 6000)])
    return [t[1:] for t in texts if re.fullmatch(r' [a-z]{4,9}', t)]


def r2_books(tok, n, T, seed=3):
    """n random windows of T + 1 reference ids with a line break every 12
    tokens: the stand-in for test_books."""
    nl, = d2l._encode(tok, '\n')
    g = torch.Generator().manual_seed(seed)
    B = torch.randint(300, 20_000, (n, T + 1), generator=g)
    B[:, 11::12] = nl
    return B


def books_patch(B):
    return mock.patch.object(
        d2l, 'test_books', lambda n, T, seed=0, tokenizer=None:
        B[:n, :T + 1])


@contextlib.contextmanager
def in_chapter():
    old = os.getcwd()
    os.chdir(CHAPTER)
    try:
        yield
    finally:
        os.chdir(old)


class RopeTest(unittest.TestCase):
    d, L = 64, 4096

    def test_scale_one(self):  # (a)
        theta = d2l.rope_inv_freq(self.d)
        base = 1.0 / 1e4 ** (torch.arange(0, self.d, 2, dtype=torch.float32)
                             / self.d)
        self.assertTrue(torch.equal(theta, base))
        for k in ('pi', 'ntk', 'yarn'):
            self.assertLessEqual(float((d2l.rope_inv_freq(
                self.d, scheme=k, s=1) - theta).abs().max()), 1e-7)
        with self.assertRaises(ValueError):
            d2l.rope_inv_freq(self.d, scheme='linear')

    def test_pi_and_ntk(self):  # (b), (c), (f)
        theta = d2l.rope_inv_freq(self.d)
        self.assertTrue(torch.equal(d2l.rope_inv_freq(self.d, scheme='pi',
                                                      s=4), theta / 4))
        ntk = d2l.rope_inv_freq(self.d, scheme='ntk', s=4)
        self.assertAlmostEqual(float(ntk[0]), 1.0, places=7)
        self.assertAlmostEqual(float(ntk[-1] / (theta[-1] / 4)), 1.0,
                               places=5)
        r = self.L * theta / (2 * math.pi)  # rotations within 4K
        self.assertEqual(int((r >= 1).sum()), 23)
        over = (4 * ntk / theta - 1)[r < 1]  # phase overhang, slow pairs
        self.assertEqual(int((over > 1e-5).sum()), 8)
        self.assertAlmostEqual(float(over[0]), 0.43, places=2)
        self.assertEqual(int(torch.argmax(over)), 0)

    def test_yarn_counts(self):  # (d)
        theta = d2l.rope_inv_freq(self.d)
        for L, counts in ((4096, (9, 12, 11)), (1024, (14, 12, 6))):
            for s in (2, 4, 8):
                ratio = d2l.rope_inv_freq(self.d, scheme='yarn', s=s,
                                          L=L) / theta
                got = (int((ratio == 1 / s).sum()),
                       int(((ratio > 1 / s) & (ratio < 1)).sum()),
                       int((ratio == 1).sum()))
                self.assertEqual(got, counts, (L, s))

    def test_mscale_value(self):  # (e)
        self.assertAlmostEqual(d2l.yarn_mscale(4), 1.1386, places=4)
        self.assertEqual(d2l.yarn_mscale(1), 1.0)
        self.assertEqual(d2l.yarn_mscale(0.5), 1.0)

    def test_mscale_once(self):  # (e): logits by m^2, softmax scale kept
        m = d2l.yarn_mscale(4)
        a, b = toy_model(), toy_model()
        a.set_rope(None, m)
        with torch.no_grad():
            for blk in b.blks:
                blk.attention.W_q.weight.mul_(m)
                blk.attention.W_k.weight.mul_(m)
        X = torch.randint(1, V, (2, 48))
        with torch.no_grad():
            torch.testing.assert_close(a(X), b(X), rtol=1e-5, atol=1e-5)
        pos = torch.arange(10)
        q, k = torch.randn(10, 8), torch.randn(10, 8)
        rot = lambda x, ms: d2l.rope(x, tables=d2l._rope_tables(
            pos, 8, 1e4, mscale=ms))
        ratio = (rot(q, m) @ rot(k, m).T) / (rot(q, 1.0) @ rot(k, 1.0).T)
        torch.testing.assert_close(ratio, torch.full_like(ratio, m * m),
                                   rtol=1e-4, atol=1e-4)

    def test_transformers(self):  # (g), gated
        try:
            hf, factor = d2l.transformers_yarn(self.d, s=4, L=self.L)
            rounded, _ = d2l.transformers_yarn(self.d, s=4, L=self.L,
                                               truncate=True)
        except Exception as e:  # noqa: BLE001
            self.skipTest(f'transformers YaRN unavailable: {e}')
        yarn = d2l.rope_inv_freq(self.d, scheme='yarn', s=4, L=self.L)
        self.assertLessEqual(float((yarn - hf).abs().max()), 1e-6)
        self.assertGreater(float((yarn - rounded).abs().max()), 1e-4)
        self.assertAlmostEqual(factor, d2l.yarn_mscale(4), places=6)


class WordTokenizer:
    """A fixture word tokenizer: '\\n', ' ', the 26 letters, ' <w>' for
    the words in spaced and '<w>' for those in bare; a word without its
    token falls back to letters (so its form is several tokens)."""
    def __init__(self, spaced, bare):
        self.vocab = (['<eos>', '\n', ' '] + list('abcdefghijklmnopqrstuvwxyz')
                      + [' ' + w for w in spaced] + list(bare))
        self.ids = {t: i for i, t in enumerate(self.vocab)}

    def __len__(self):
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.ids)

    def encode(self, text):
        out = []
        for piece in re.findall(r'\n| ?[a-z]+| ', text):
            if piece in self.ids:
                out.append(self.ids[piece])
                continue
            if piece.startswith(' '):
                out.append(self.ids[' '])
                piece = piece[1:]
            out += ([self.ids[piece]] if piece in self.ids else
                    [self.ids[c] for c in piece])
        return out

    def __call__(self, texts, add_special_tokens=False):
        if isinstance(texts, str):
            return {'input_ids': self.encode(texts)}
        return {'input_ids': [self.encode(t) for t in texts]}

    def decode(self, ids):
        return ''.join(self.vocab[int(i)] for i in ids)

    def batch_decode(self, seqs):
        return [self.decode(s) for s in seqs]


POOL = [f'{a}{b}word' for a in 'bcdfg' for b in 'aeiou'][:40]  # 40 words
FILLER = ['the', 'and', 'river', 'house', 'stone', 'night', 'garden',
          'window', 'letter', 'quiet']


def fixture_tokenizer():
    """POOL[1] and POOL[3] have no spaced token (not values); POOL[2] has
    no bare token (not a key); POOL[4] and POOL[5] occur in the text."""
    spaced = [w for i, w in enumerate(POOL) if i not in (1, 3)] + FILLER
    bare = [w for i, w in enumerate(POOL) if i % 2 == 0 and i != 2]
    return WordTokenizer(spaced, bare)


def fixture_books(tok, n, T, seed=0):
    """n windows of T + 1 tokens of fixture text: filler words with POOL[4]
    in the even windows and POOL[5] in the odd ones, a line break every
    eight words."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        words = list(rng.choice(FILLER, size=T)) + [POOL[4 + i % 2]] * 4
        rng.shuffle(words)
        text = ''.join(f' {w}' + ('\n' if k % 8 == 7 else '')
                       for k, w in enumerate(words))
        ids = tok.encode(text)
        assert len(ids) > T, len(ids)
        out.append(ids[:T + 1])
    return torch.tensor(out)


class NeedlePoolTest(unittest.TestCase):
    def test_fixture_pools(self):
        tok = fixture_tokenizer()
        keys, values = d2l.needle_pools(tok, POOL)
        kw, vw = [w for w, _ in keys], [w for w, _ in values]
        self.assertEqual(kw, [w for i, w in enumerate(POOL)
                              if i % 2 == 0 and i != 2])
        self.assertEqual(vw, [w for i, w in enumerate(POOL)
                              if i % 2 == 1 and i not in (1, 3)])
        self.assertFalse(set(kw) & set(vw))
        self.assertFalse({i for _, i in keys} & {i for _, i in values})
        for w, i in keys:  # a key on its own: one token
            self.assertEqual(tok.encode(w), [i])
        for w, i in values:  # a value with its leading space: one token
            self.assertEqual(tok.encode(' ' + w), [i])
        self.assertEqual(len(tok.encode(' ' + POOL[1])), 1 + len(POOL[1]))

    def test_r2_pools(self):
        tok = r2_tok()
        keys, values = d2l.needle_pools(tok, r2_words(tok))
        self.assertGreater(len(keys), 50)
        self.assertGreater(len(values), 200)
        self.assertFalse({w for w, _ in keys} & {w for w, _ in values})
        self.assertTrue(all(d2l._encode(tok, w) == [i] for w, i in keys))
        self.assertTrue(all(d2l._encode(tok, ' ' + w) == [i]
                            for w, i in values))


class NeedleWindowTest(unittest.TestCase):
    def windows(self, tok, L, n, B, seed=0):
        with books_patch(B):
            return d2l.needle_windows(tok, L, n, seed=seed)

    def test_layout(self):
        tok = r2_tok()
        B = r2_books(tok, 6, 4096)
        with mock.patch.object(d2l, '_needle_words', lambda: r2_words(tok)):
            for L in (1024, 2048, 4096):
                nd = self.windows(tok, L, 6, B)
                self.assertEqual(tuple(nd.prefix.shape), (6, L - 2))
                self.assertEqual(tuple(nd.gaps.shape), (6, 8))
                self.assertTrue(((nd.gaps > 32) & (nd.gaps <= L - 64)).all())
                v = (L - 1) - nd.gaps  # value positions, farthest first
                self.assertTrue(((v[:, 1:] - v[:, :-1]) >= 16 + 3).all())
                ends = d2l._line_ends(tok)
                P = nd.prefix
                rows = torch.arange(6)[:, None]
                self.assertTrue(ends[P[rows, v - 2]].all())  # after a break
                self.assertEqual(d2l.needle_diagnostics(tok, nd), dict(
                    oracle=True, single=True, absent=True, unique=True))
                again = self.windows(tok, L, 6, B)
                self.assertTrue(all(torch.equal(a, b) for a, b in
                                    zip(nd[:4], again[:4])))
                other = self.windows(tok, L, 6, B, seed=1)
                self.assertFalse(torch.equal(nd.gaps, other.gaps))

    def test_fixture_diagnostics(self):
        """The pilot's checks on a fixture tokenizer, and each failing on a
        window built to fail it."""
        tok = fixture_tokenizer()
        B = fixture_books(tok, 4, 1024)
        with mock.patch.object(d2l, '_needle_words', lambda: POOL):
            nd = self.windows(tok, 1024, 4, B)
            self.assertEqual(d2l.needle_diagnostics(tok, nd), dict(
                oracle=True, single=True, absent=True, unique=True))
            for i in range(4):  # the window's own pool word is never used
                used = set(nd.keys[i].tolist()) | set(nd.values[i].tolist())
                present = POOL[4 + i % 2]
                self.assertFalse({tok.ids.get(present),
                                  tok.ids.get(' ' + present)} & used)
            bad = nd._replace(prefix=nd.prefix.clone())
            bad.prefix[0, 5] = nd.keys[0, 0]  # a key in the text
            d = d2l.needle_diagnostics(tok, bad)
            self.assertFalse(d['absent'])
            self.assertFalse(d['unique'])
            pools = d2l.needle_pools(tok)
            split = ([(POOL[2], tok.ids[' ' + POOL[2]])], pools[1])
            with mock.patch.object(d2l, 'needle_pools', lambda t: split):
                self.assertFalse(d2l.needle_diagnostics(tok, nd)['single'])
            moved = nd._replace(gaps=nd.gaps + 1)
            d = d2l.needle_diagnostics(tok, moved)
            self.assertFalse(d['unique'])
            (w0, i0), = pools[0][:1]

            class Merged(WordTokenizer):  # '\n<first key>' is one token
                def encode(self, text):
                    if text.startswith('\n' + w0):
                        return [i0] + super().encode(text[1 + len(w0):])
                    return super().encode(text)

            merged = Merged.__new__(Merged)
            merged.__dict__.update(tok.__dict__)
            self.assertEqual(d2l._encode(merged, w0), [i0])  # bare form fine
            self.assertFalse(d2l.needle_diagnostics(merged, nd)['single'])

    def test_bins(self):
        """The octave bins partition the queries (a gap on an edge falls in
        the bin below it); seeds pooled per query average correctness."""
        rng = np.random.default_rng(0)
        gaps = np.concatenate([rng.integers(33, 1025, 500),
                               [33, 64, 65, 512, 1024]])
        rec = pd.DataFrame(dict(gap=gaps, correct=rng.random(505) < 0.6))
        b = d2l.recall_by_gap(rec, 1024)
        self.assertEqual(list(b['gap_lo']), [32, 64, 128, 256, 512])
        self.assertEqual(int(b['n'].sum()), len(rec))
        self.assertEqual(int(b['n'][0]), int(((gaps > 32) & (gaps <= 64))
                                             .sum()))
        self.assertEqual(int(b['k'].sum()), int(rec['correct'].sum()))
        self.assertTrue((b['lo'] <= b['rate']).all())
        self.assertEqual(d2l.gap_bins(16_384)[-1], (8192, 16_384))
        self.assertAlmostEqual(d2l.recall_between(rec, 32, 1024),
                               rec['correct'].mean())


class NeedleScorerTest(unittest.TestCase):
    def test_cached_equals_full(self):
        """One prefill plus each query in the same two cache slots equals
        a full forward of prefix + query (fp32, 1e-5), unscaled and with
        YaRN at scale 2; needle_recall picks the argmax over the values."""
        for scaled in (False, True):
            model = toy_model(max_len=80 if scaled else 160).float()
            if scaled:
                d2l._lc_set_scheme(model, 'yarn', 2, 80)
            g = torch.Generator().manual_seed(int(scaled))
            P = torch.randint(32, V, (2, 158), generator=g)
            keys = torch.randint(32, V, (2, 3), generator=g)
            values = torch.randint(32, V, (2, 3), generator=g)
            nd = d2l.Needles(P, keys, values, torch.full((2, 3), 40), 198)
            rec = d2l.needle_recall(model, nd, chunk=50)
            self.assertEqual(list(rec.columns), ['window', 'needle', 'gap',
                                                 'correct'])
            for i in range(2):
                with torch.inference_mode():
                    got = d2l._needle_logits(model, nd, i, chunk=50)
                    for j in range(3):
                        X = torch.cat([P[i], torch.tensor(
                            [198, int(keys[i, j])])])[None]
                        torch.testing.assert_close(
                            got[j], model(X)[0, -1], rtol=1e-5, atol=1e-5)
                pick = got[:, values[i]].argmax(-1)
                self.assertEqual(rec[rec['window'] == i]['correct'].tolist(),
                                 (pick == torch.arange(3)).tolist())

    def test_oracle(self):
        P = torch.tensor([[5, 6, 9, 7, 8, 9, 11, 12, 9]])
        nd = d2l.Needles(P, torch.tensor([[7, 11]]), torch.tensor([[8, 12]]),
                         torch.tensor([[6, 3]]), 9)
        self.assertEqual(d2l.needle_oracle(nd)['correct'].tolist(),
                         [True, True])
        nd2 = nd._replace(values=torch.tensor([[8, 13]]))
        self.assertEqual(d2l.needle_oracle(nd2)['correct'].tolist(),
                         [True, False])


class ReachTest(unittest.TestCase):
    @staticmethod
    def bins(los):
        k = np.arange(len(los))
        return pd.DataFrame(dict(gap_lo=32 * 2 ** k, gap_hi=64 * 2 ** k,
                                 lo=los))

    def test_edges(self):
        r = d2l.reach
        self.assertEqual(r(self.bins([0.4, 0.9, 0.9])), 0)  # first fails
        self.assertEqual(r(self.bins([0.6, 0.7, 0.8])), 256)  # all pass
        self.assertEqual(r(self.bins([0.6, 0.4, 0.8])), 64)
        self.assertEqual(r(self.bins([0.5, 0.5])), 128)  # at least lo
        self.assertEqual(r(self.bins([0.6, math.nan, 0.8])), 64)
        self.assertEqual(r(self.bins([0.3, 0.25]), 0.25, strict=True), 64)
        self.assertEqual(r(self.bins([0.6, 0.7, 0.8]).iloc[::-1]), 256)

    def test_wilson_tables(self):
        """From counts: 70 of 100 has a Wilson lower bound above 0.5, 55 of
        100 below; 40 of 100 is detectable (above 0.25), 30 of 100 not."""
        rows = []
        for k, (lo, hi) in zip((95, 70, 55, 40, 30),
                               d2l.gap_bins(1024)):
            rows += [(lo + 1, i < k) for i in range(100)]
        b = d2l.recall_by_gap(pd.DataFrame(rows, columns=['gap', 'correct']),
                              1024)
        self.assertEqual(d2l.reach(b), 128)
        self.assertEqual(d2l.reach(b, 0.25, strict=True), 512)


class FormatTest(unittest.TestCase):
    def test_items(self):
        """Both formats: the oracle recalls every item; the gold candidate
        is the queried line's value; the cached candidate scorer equals the
        summed log-probabilities of full forwards."""
        tok = r2_tok()
        B = [b.numpy() for b in r2_books(tok, 4, 2047)]
        with mock.patch.object(d2l, '_test_book_tokens', lambda t=None: B), \
                mock.patch.object(d2l, '_needle_words',
                                  lambda: r2_words(tok)):
            model = toy_model(max_len=1100).float()
            for fmt in ('template', 'one-hop'):
                items = d2l._format_items(tok, fmt, 1024, 4)
                again = d2l._format_items(tok, fmt, 1024, 4)
                self.assertEqual(items, again)
                for it in items:
                    self.assertEqual(d2l._format_oracle(it), it['gold'])
                    self.assertEqual(len(it['conts']), 8)
                    self.assertLessEqual(len(it['ctx']) + max(map(
                        len, it['conts'])), 1024)
                it = items[0]
                ll = d2l._choice_ll(model, it['ctx'], it['conts'], chunk=300)
                with torch.no_grad():
                    for j, c in enumerate(it['conts']):
                        X = torch.tensor([it['ctx'] + c])
                        lp = model(X[:, :-1])[0].log_softmax(-1)
                        n = len(it['ctx'])
                        ref = float(lp[n - 1 + torch.arange(len(c)),
                                       torch.tensor(c)].sum())
                        self.assertAlmostEqual(ll[j], ref, delta=1e-4)
        with self.assertRaises(ValueError):
            d2l._format_items(tok, 'two-hop', 1024, 1)

    def test_paired(self):
        """Item i of the two formats is the same book window: most of the
        one-hop context's trigrams occur in the template context (they
        share 1-3% when the offsets differ), and the candidate order is
        the same."""
        tok = r2_tok()
        B = [b.numpy() for b in r2_books(tok, 4, 2047)]
        with mock.patch.object(d2l, '_test_book_tokens', lambda t=None: B), \
                mock.patch.object(d2l, '_needle_words',
                                  lambda: r2_words(tok)):
            a = d2l._format_items(tok, 'template', 1024, 6)
            b = d2l._format_items(tok, 'one-hop', 1024, 6)
        tri = lambda x: {tuple(x[i:i + 3]) for i in range(len(x) - 2)}
        for x, y in zip(a, b):
            self.assertEqual(x['gold'], y['gold'])
            share = len(tri(y['ctx']) & tri(x['ctx'])) / len(tri(y['ctx']))
            self.assertGreater(share, 0.6)
        self.assertLess(len(tri(b[0]['ctx']) & tri(a[1]['ctx'])), 30)


def probe_result(steps=(500, 1000, 2000, 6000), n_win=40, seed=0):
    """A synthetic LcProbe: the format items of two models, and r2's
    queries at 4K whose recall at gaps 256-1K grows with the step."""
    rng = np.random.default_rng(seed)
    rows = []
    for m, p in (('oracle', (1.0, 1.0)), ('r2', (0.1, 0.9))):
        for f, q in zip(('template', 'one-hop'), p):
            rows += [(m, f, i, bool(rng.random() < q)) for i in range(60)]
    items = pd.DataFrame(rows, columns=d2l._LC_PROBE_COLUMNS['format_items'])
    table = [(m, f, len(x), int(x.sum()), x.mean(),
              *d2l.wilson_ci(int(x.sum()), len(x)))
             for (m, f), x in items.groupby(['model', 'format'],
                                            sort=False)['correct']]
    gaps = np.exp(rng.uniform(math.log(33), math.log(4032), (n_win, 8)))
    q, bins = [], []
    for k, step in enumerate(steps):
        p = np.where(gaps <= 64, [0.12, 0.9, 0.95, 0.95][k],
                     np.where(gaps <= 1024, 0.2 + 0.2 * k, 0.1))
        ok = rng.random(gaps.shape) < p
        x = pd.DataFrame([(step, i, j, int(gaps[i, j]), bool(ok[i, j]))
                          for i in range(n_win) for j in range(8)],
                         columns=['step', 'window', 'needle', 'gap',
                                  'correct']).assign(model='r2', tokens=step)
        q.append(x)
        bins.append(d2l.recall_by_gap(x, 4096).assign(model='r2', step=step,
                                                       tokens=step))
    P = d2l._LC_PROBE_COLUMNS
    return d2l.LcProbe(pd.DataFrame(table, columns=P['format']),
                       pd.concat(bins, ignore_index=True)[P['reach_train']],
                       items, pd.concat(q, ignore_index=True)[
                           P['reach_queries']], '')


class ProbeDisplayTest(unittest.TestCase):
    def test_copying_steps(self):
        """Copying appears between a and b only when a's interval holds
        1/8; otherwise 'by step b' alone."""
        def x(los, his):
            return pd.DataFrame(dict(gap_lo=32, step=[500, 1000, 2000],
                                     lo=los, hi=his))
        cs = d2l._copying_steps
        self.assertEqual(cs(x([0.05, 0.3, 0.6], [0.2, 0.5, 0.8])),
                         (500, 1000))
        self.assertEqual(cs(x([0.15, 0.3, 0.6], [0.24, 0.5, 0.8])),
                         (None, 1000))  # above chance, not detectable
        self.assertEqual(cs(x([0.3, 0.4, 0.6], [0.5, 0.6, 0.8])),
                         (None, 500))
        self.assertEqual(cs(x([0.05, 0.1, 0.2], [0.2, 0.3, 0.4])),
                         (None, None))

    def test_format(self):
        probe = probe_result()
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            d2l.show_lc_format(probe)
        lines = check_display(self, buf.getvalue())
        self.assertEqual(lines[0].split()[:3], ['model', 'template',
                                                'one-hop'])
        self.assertIn('template − one-hop', lines[0])
        it = probe.format_items
        r2 = it[it['model'] == 'r2'].pivot(index='item', columns='format',
                                           values='correct')
        d = (r2['template'].astype(float) - r2['one-hop']).mean()
        self.assertIn(f'{d:+.2f} [', lines[2])
        self.assertIn('+0.00 [+0.00, +0.00]', lines[1])  # the oracle

    def test_reach_train(self):
        probe = probe_result()
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            d2l.show_reach_train(probe)
        ax, = d2l.plt.gcf().axes  # one run, one panel
        curves = [l for l in ax.get_lines() if len(l.get_xdata()) > 2]
        self.assertEqual(len(curves), 4)  # one line per checkpoint
        self.assertEqual(len({l.get_color() for l in curves}), 4)
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()],
                         ['step 500', 'step 1,000', 'step 2,000',
                          'step 6,000'])
        self.assertEqual(ax.get_title(), 'r2 at 4K')
        d2l.plt.close('all')
        lines = check_display(self, buf.getvalue())
        self.assertEqual(lines[0], 'copying appears between steps 500 and '
                                   '1,000 (r2)')
        self.assertRegex(lines[1], r'^r2 reach at 4K \(steps 1,000 → 6,000\)'
                                   r': [\d,]+(, [\d,]+){2}; non-decreasing')
        q = probe.reach_queries
        q = q[(q['gap'] > 256) & (q['gap'] <= 1024)]
        d = (q[q['step'] == 6000]['correct'].mean()
             - q[q['step'] == 1000]['correct'].mean())
        self.assertTrue(lines[2].startswith(
            f'r2 recall 256-1K back, step 6,000 − step 1,000: {d:+.2f} ['),
            lines[2])
        self.assertTrue(lines[2].endswith('(paired by query)'))
        d2l.plt.close('all')

    def test_growth_fixed_steps(self):
        """Claim 11 compares steps 1000 and 6000 whatever the copying
        checkpoint: copying detectable at step 500 leaves both lines at
        1,000 → 6,000; a run without step 1000 is refused."""
        probe = probe_result()
        rt = probe.reach_train
        early = (rt['step'] == 500) & (rt['gap_lo'] == 32)
        rt.loc[early, ['lo', 'hi']] = [0.3, 0.6]  # detectable at step 500
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            d2l.show_reach_train(probe)
        d2l.plt.close('all')
        lines = check_display(self, buf.getvalue())
        self.assertEqual(lines[0], 'copying appears by step 500 (r2)')
        self.assertIn('(steps 1,000 → 6,000)', lines[1])
        self.assertIn('step 6,000 − step 1,000:', lines[2])
        drop = probe_result(steps=(500, 2000, 3000, 6000))
        with self.assertRaisesRegex(ValueError, 'step 1000'), \
                contextlib.redirect_stdout(io.StringIO()):
            d2l.show_reach_train(drop)
        d2l.plt.close('all')
        self.assertEqual(d2l._reach_verdict([256, 128, 512]),
                         'non-decreasing within one bin')
        self.assertEqual(d2l._reach_verdict([512, 128]), 'falling')

    def test_differences(self):
        """Paired: the mean of per-query differences; unpaired: the
        difference of the rates; both with intervals around them."""
        rng = np.random.default_rng(1)
        a = pd.DataFrame(dict(window=np.repeat(np.arange(30), 8),
                              needle=np.tile(np.arange(8), 30),
                              correct=rng.random(240) < 0.7))
        b = a.assign(correct=rng.random(240) < 0.4)
        d, lo, hi = d2l.recall_paired(a, b)
        self.assertAlmostEqual(d, a['correct'].mean() - b['correct'].mean())
        self.assertLess(lo, d)
        self.assertGreater(hi, d)
        c = b.iloc[:80]
        d, lo, hi = d2l.recall_unpaired(a, c)
        self.assertAlmostEqual(d, a['correct'].mean() - c['correct'].mean())
        self.assertLess(lo, d)
        self.assertTrue(all(math.isnan(v) for v in
                            d2l.recall_unpaired(a, c.iloc[:0])))


class ProbeComputeTest(unittest.TestCase):
    """lc-probe reads its checkpoints from their registered keys only (plan
    review, finding 12), names them with their sha1 in its configuration,
    refuses to start on an oracle miss, and returns the format table and
    reach during pretraining in the stored layout."""
    T1, R2 = (509, 1017, 1526), (500, 1000, 2000, 3000, 4000, 6000)

    def test_keys(self):
        keys = d2l._lc_probe_keys(self.T1, self.R2)
        self.assertEqual(list(keys), [
            'lm-r2t1-s1526-v1', 'lm-r2t1-s509-v1', 'lm-r2t1-s1017-v1',
            'lm-r2-s6000-v1', 'lm-r2-s500-v1', 'lm-r2-s1000-v1',
            'lm-r2-s2000-v1', 'lm-r2-s3000-v1', 'lm-r2-s4000-v1'])
        for k, sha1 in keys.items():
            self.assertEqual(sha1, d2l.DATA_HUB[k][1])
            self.assertRegex(sha1, r'^[0-9a-f]{40}$')
        hub = dict(d2l.DATA_HUB)
        del hub['lm-r2t1-s1017-v1']
        with mock.patch.dict(d2l.DATA_HUB, hub, clear=True), \
                self.assertRaisesRegex(KeyError, 'lm-r2t1-s1017-v1'):
            d2l._lc_probe_keys(self.T1, self.R2)
        with self.assertRaises(ValueError):  # a step the run never saved
            d2l._lc_probe_keys((500,), self.R2)

    def test_config_names_keys(self):
        seen = {}

        def result(name, compute, config, **kw):
            seen.update(name=name, config=config)
            return d2l._pending_result(name)
        with mock.patch.object(d2l, '_pretrain_result', result):
            d2l.lc_probe_runs(self.T1, self.R2)
        self.assertEqual(seen['name'], 'lc-probe')
        self.assertEqual(seen['config']['checkpoints'],
                         d2l._lc_probe_keys(self.T1, self.R2))

    def test_registered_not_local(self):
        """With D2L_RERUN naming lc-probe and t1-evals, a local next
        version and a t1 job folder both exist; load_reference_model reads
        the job folder, _lc_probe_model the registered key."""
        reg, local = toy_model(seed=1), toy_model(seed=2)
        with tempfile.TemporaryDirectory() as tmp:
            key = 'lm-r2t1-s509-v1'
            files = {}
            for name, m in (('reg', reg), ('next', local), ('job', local)):
                files[name] = os.path.join(tmp, f'{name}.safetensors')
                d2l._save_bf16(files[name], m, 509)
            calls = []

            def keyed(k, *a, **kw):
                calls.append(k)
                return files['reg']
            job = os.path.join(tmp, 'ch21', 't1-s0', 'ckpt')
            os.makedirs(job)
            os.link(files['job'], os.path.join(job, 's509.safetensors'))
            os.link(files['next'], os.path.join(
                tmp, d2l._next_key(key) + '.safetensors'))
            with mock.patch.object(d2l, 'download_keyed', keyed), \
                    mock.patch.object(d2l, '_rerun_dir', lambda: tmp), \
                    mock.patch.object(d2l, '_job_root',
                                      lambda: os.path.join(tmp, 'ch21')), \
                    mock.patch.dict(os.environ,
                                    {'D2L_RERUN': 'lc-probe,t1-evals'}):
                got = d2l._lc_probe_model('t1', 509, 'cpu')
                other = d2l.load_reference_model(509, preset='t1')
            self.assertEqual(calls, [key])
            w = lambda m: next(m.parameters())
            self.assertEqual(w(got).dtype, torch.bfloat16)
            self.assertTrue(torch.equal(w(got), w(reg).bfloat16()))
            self.assertFalse(torch.equal(w(other), w(reg).bfloat16()))

    def compute(self, oracle=None):
        """_lc_probe_compute on toys: 3 format items of 1,024 tokens from
        fixture books, 2 needle windows of 160 tokens per run; returns
        (result, the (preset, step) loads in order)."""
        tok = r2_tok()
        B = [b.numpy() for b in r2_books(tok, 4, 2047)]
        g = torch.Generator().manual_seed(0)
        nd = d2l.Needles(torch.randint(32, V, (2, 158), generator=g),
                         torch.randint(32, V, (2, 8), generator=g),
                         torch.randint(32, V, (2, 8), generator=g),
                         torch.tensor([[40, 50, 60, 70, 80, 90, 100, 120]] * 2),
                         198)
        loads = []

        def model(preset, step, device):
            loads.append((preset, step))
            return toy_model(seed=step or 0, max_len=1100).float()
        patches = [
            mock.patch.object(d2l, '_test_book_tokens', lambda t=None: B),
            mock.patch.object(d2l, '_needle_words', lambda: r2_words(tok)),
            mock.patch.object(d2l, 'lc_tokenizer', lambda *a: tok),
            mock.patch.object(d2l, '_lc_device', lambda: torch.device('cpu')),
            mock.patch.object(d2l, '_lc_probe_model', model),
            mock.patch.object(d2l, '_lc_load_reference', lambda m: (
                toy_model(seed=7, max_len=1100).float(), tok)),
            mock.patch.object(d2l, 'needle_windows', lambda t, L, n: nd),
            mock.patch.dict(d2l._LC_PROBE, items=3, windows=2)]
        if oracle:
            patches.append(mock.patch.object(d2l, '_format_oracle', oracle))
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            return d2l._lc_probe_compute(self.T1, self.R2), loads

    def test_compute(self):
        res, loads = self.compute()
        self.assertEqual(loads, [('t1', None), ('r2', None)]  # finals
                         + [('t1', s) for s in self.T1]
                         + [('r2', s) for s in self.R2])
        P = d2l._LC_PROBE_COLUMNS
        for field in ('format', 'reach_train', 'format_items',
                      'reach_queries'):
            self.assertEqual(list(getattr(res, field).columns), P[field])
        f = res.format.set_index(['model', 'format'])
        self.assertEqual(list(dict.fromkeys(res.format['model'])),
                         ['oracle', 't1', 'r2', 'Qwen3-0.6B-Base'])
        self.assertTrue((f['n'] == 3).all())
        self.assertEqual(list(f.loc['oracle']['rate']), [1.0, 1.0])
        rt = res.reach_train
        self.assertEqual(sorted(rt[rt['model'] == 't1']['step'].unique()),
                         list(self.T1))
        c = d2l.r2_config('r2')
        x = rt[(rt['model'] == 'r2') & (rt['step'] == 1000)]
        self.assertTrue((x['tokens'] == 1000 * c.global_batch * c.arch.T)
                        .all())
        self.assertEqual(int(x['n'].sum()), 16)  # 2 windows x 8 queries
        self.assertEqual(len(res.reach_queries), 16 * 9)

    def test_oracle_miss(self):
        with self.assertRaisesRegex(AssertionError, 'oracle copier misses'):
            self.compute(oracle=lambda it: -1)

    def test_oracle_checks_every_model(self):
        """The control runs on each model's own items: a miss on the
        reference's items (its own tokenizer) stops the job there."""
        real, calls = d2l._format_oracle, []

        def oracle(it):  # right on the first three models' items
            calls.append(1)
            return real(it) if len(calls) <= 24 else -1
        with self.assertRaisesRegex(AssertionError,
                                    r'misses 3 template items '
                                    r'\(Qwen3-0.6B-Base\)'):
            self.compute(oracle=oracle)


class MeasureTest(unittest.TestCase):
    def test_buckets_average(self):
        model = toy_model(max_len=256)
        torch.manual_seed(1)
        W = torch.randint(1, V, (4, 257))
        parts = d2l.nll_by_position(model, W, range(0, 257, 64))
        whole = d2l.nll_by_position(model, W, [0, 256])
        self.assertLess(abs(parts['nll'].mean() - whole['nll'][0]), 1e-6)
        self.assertEqual(list(parts['bucket']), ['0-64', '64-128',
                                                 '128-192', '192-256'])
        self.assertTrue((parts['lo'] <= parts['nll']).all()
                        and (parts['nll'] <= parts['hi']).all())

    def test_gain_oracle(self):
        Vs, T, short = 2048, 1024, 128
        torch.manual_seed(0)
        x = torch.stack([torch.randperm(Vs)[:T // 2] for _ in range(4)])
        rep = torch.cat([x, x, x[:, :1]], 1)
        g = {}
        for name, m in (('oracle', d2l.CopyOracle(Vs)),
                        ('unigram', d2l.UnigramModel(Vs))):
            g[name] = d2l.long_context_gain(m, rep, short_len=short,
                                            long_len=T, min_gap=short,
                                            rare_ids=torch.arange(Vs),
                                            edges=[short, T])
        # one target per row (the second copy's first) has no cue: gain 0
        self.assertAlmostEqual(g['oracle']['gain'][0], math.log(Vs) * (
            T // 2) / (T // 2 + 1), places=3)
        self.assertEqual(g['unigram']['gain'][0], 0.0)
        self.assertEqual(int(g['oracle']['n'][0]), 4 * (T // 2 + 1))

    def test_sliding(self):
        model = toy_model(max_len=128)
        torch.manual_seed(2)
        W = torch.randint(1, V, (2, 129))
        short = d2l.sliding_nlls(model, W, 32)
        full = d2l.window_nlls(model, W)
        torch.testing.assert_close(short[:, :32], full[:, :32], rtol=1e-5,
                                   atol=1e-5)
        for t in (40, 47, 100, 127):  # context: from (t // 16 - 1) * 16
            a = (t // 16 - 1) * 16
            ref = d2l.window_nlls(model, W[:, a:t + 2])[:, -1]
            torch.testing.assert_close(short[:, t], ref, rtol=1e-5,
                                       atol=1e-5)

    def test_context(self):
        """context_nlls on the second half equals sliding_nlls of the whole
        window there (c / 2 to c tokens back either way); c = T is the
        whole window."""
        model = toy_model(max_len=256)
        torch.manual_seed(4)
        W = torch.randint(1, V, (2, 257))
        for c in (32, 64):
            torch.testing.assert_close(
                d2l.context_nlls(model, W, c),
                d2l.sliding_nlls(model, W, c)[:, 128:], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(d2l.context_nlls(model, W, 256),
                                   d2l.window_nlls(model, W)[:, 128:])

    def test_repeat_gaps(self):
        W = torch.tensor([[5, 7, 5, 9, 7, 5, 3]])
        gaps = d2l.repeat_gaps(W, [5, 7, 9])
        self.assertEqual(gaps.tolist(), [[0, 2, 0, 3, 3, 0]])
        self.assertEqual(d2l.repeat_gaps(W, [7]).tolist(),
                         [[0, 0, 0, 3, 0, 0]])

    def test_window_ci(self):
        mean, lo, hi = d2l.window_ci(np.arange(10.0), np.ones(10))
        self.assertEqual(mean, 4.5)
        self.assertLess(lo, mean)
        self.assertGreater(hi, mean)
        self.assertTrue(all(math.isnan(v) for v in d2l.window_ci(
            [0.0], [0.0])))
        self.assertEqual(d2l.bucket_name(0, 4096), '0-4K')
        self.assertEqual(d2l.bucket_name(1536, 3072), '1.5-3K')
        self.assertEqual(d2l.bucket_name(256, 512), '256-512')


class StagedStreamTest(unittest.TestCase):
    def test_pieces(self):
        class Mix:
            rank, world, step = 0, 1, 0

            def batch(self, step):
                g = torch.Generator().manual_seed(step)
                X = torch.randint(1, 50, (3, 32), generator=g)
                X[0, 5] = X[1, 20] = X[2, 16] = 0  # EOS
                Y = torch.cat([X[:, 1:], X[:, :1]], 1)
                doc, pos = d2l.doc_ids_from_eos(X.int(), 0)
                return dict(input_ids=X, labels=Y, doc_ids=doc,
                            pos_ids=pos)

        cfg = dict(stages=[dict(T=16, s=2, steps=2), dict(T=32, s=4,
                                                          steps=2)])
        st = d2l._StagedStream(Mix(), cfg, 0)
        for step in range(4):
            b, ref = st.batch(step), Mix().batch(step)
            torch.testing.assert_close(b['labels'], ref['labels'])
            if step >= 2:
                torch.testing.assert_close(b['doc_ids'], ref['doc_ids'])
                continue
            doc, pos = b['doc_ids'], b['pos_ids']
            self.assertTrue((doc[:, 16:].amin(1) > doc[:, :16].amax(1))
                            .all())
            self.assertTrue((pos[:, 16] == 0).all())
            model = toy_model(max_len=32)
            with torch.no_grad():
                whole = model(b['input_ids'], d2l.PackCtx(doc, pos))
                for k in (0, 1):
                    Xk = b['input_ids'][:, 16 * k:16 * (k + 1)]
                    alone = model(Xk, d2l.PackCtx(*d2l.doc_ids_from_eos(
                        Xk.int(), 0)))
                    torch.testing.assert_close(
                        whole[:, 16 * k:16 * (k + 1)], alone, rtol=1e-4,
                        atol=1e-4)
        self.assertEqual(d2l._lc_stage_of(cfg, 1), 0)
        self.assertEqual(d2l._lc_stage_of(cfg, 2), 1)
        self.assertEqual(d2l._lc_stage_of(cfg, 9), 1)


class ConfigTest(unittest.TestCase):
    def test_t2(self):
        cfgs = d2l._lc_configs(**d2l._lc_t2_call())
        self.assertEqual(sorted(cfgs), sorted(d2l._LC_T2_NAMES))
        self.assertEqual(cfgs['PI', 0]['steps'], 381)
        self.assertEqual([st['steps'] for st in cfgs['YaRN-prog', 0][
            'stages']], [153, 229])
        self.assertEqual(cfgs['YaRN', 2]['decay'], round(0.15 * 381))
        self.assertEqual(cfgs['PI', 1]['global_batch'], 16)
        self.assertEqual(cfgs['PI', 0]['trained_len'], 4096)
        self.assertTrue(all(c['evaluate'] for c in cfgs.values()))
        self.assertTrue(d2l._lc_job(cfgs['YaRN-prog', 0]).endswith(
            os.path.join('lc', 't2-yarn-prog-s0')))
        with self.assertRaises(ValueError):
            d2l._lc_configs(**dict(d2l._lc_t2_call(), arms={
                'PI': dict(scheme='pi', stages=((4096, 4, 1.0),))}))
        s = d2l._lc_eval_settings('t2', cfgs['YaRN-prog', 0])
        self.assertEqual(s['lengths'], [4096, 8192, 16_384])
        self.assertEqual(d2l._lc_eval_settings('t2', None)['lengths'], [4096])

    def test_t1(self):
        cfgs = d2l._lc_configs(**d2l._lc_t1_call())
        self.assertEqual(len(cfgs), 11)
        self.assertEqual(cfgs['PI', 4]['steps'], 244)
        self.assertEqual([st['steps'] for st in cfgs['YaRN-prog', 0][
            'stages']], [98, 146])
        self.assertEqual(cfgs['PI', 0]['micro_batch'], d2l._LC_MICRO['t1'])
        self.assertFalse(any(c['evaluate'] for c in cfgs.values()))

    def test_identity(self):
        """evaluate is not part of an arm's identity; a seed-1 T2 arm whose
        recipe differs from seed 0's in anything but the seed is refused."""
        cfgs = d2l._lc_configs(**d2l._lc_t2_call())
        with mock.patch.object(d2l, '_lc_base_path', lambda b: __file__), \
                tempfile.TemporaryDirectory() as root, \
                mock.patch.object(d2l, '_job_root', lambda: root):
            a = d2l._lc_identity(cfgs['PI', 1])
            b = d2l._lc_identity(dict(cfgs['PI', 1], evaluate=False))
            self.assertEqual(a, b)
            self.assertNotIn('evaluate', a['config'])
            d2l._lc_check_recipe(cfgs['PI', 1])
            job = d2l._lc_job(cfgs['PI', 0])
            os.makedirs(job)
            with open(os.path.join(job, 'arm.json'), 'w') as f:
                json.dump(d2l._lc_identity(dict(cfgs['PI', 0], warmup=10)), f)
            with self.assertRaisesRegex(ValueError, 'warmup'):
                d2l._lc_check_recipe(cfgs['PI', 1])

    def test_t1_recipe(self):
        """The new T1 seeds 3-4 must have the recipe of the first build's
        seed 0 (its arm.json): a seed-3 arm against a seed 0 recorded with
        another warmup is refused, and _lc_arm_run makes the check for a
        T1 seed > 0 before it trains (it does not for seed 0)."""
        cfgs = d2l._lc_configs(**d2l._lc_t1_call())
        with mock.patch.object(d2l, '_lc_base_path', lambda b: __file__), \
                tempfile.TemporaryDirectory() as root, \
                mock.patch.object(d2l, '_job_root', lambda: root):
            job = d2l._lc_job(cfgs['YaRN', 0])
            os.makedirs(job)
            with open(os.path.join(job, 'arm.json'), 'w') as f:
                json.dump(d2l._lc_identity(cfgs['YaRN', 0]), f)
            for seed in (3, 4):
                d2l._lc_check_recipe(cfgs['YaRN', seed])
            with open(os.path.join(job, 'arm.json'), 'w') as f:
                json.dump(d2l._lc_identity(dict(cfgs['YaRN', 0],
                                                warmup=10)), f)
            with self.assertRaisesRegex(ValueError, 'seed 3 differs .*warmup'):
                d2l._lc_check_recipe(cfgs['YaRN', 3])

            class Checked(Exception):
                pass

            def check(cfg):
                raise Checked(cfg['seed'])
            with mock.patch.object(d2l, '_lc_check_recipe', check), \
                    mock.patch.object(d2l, '_lc_check_disk',
                                      lambda cfg: (_ for _ in ()).throw(
                                          OSError('no training here'))):
                with self.assertRaises(Checked):
                    d2l._lc_arm_run(cfgs['PI', 3])
                with self.assertRaisesRegex(OSError, 'no training here'):
                    d2l._lc_arm_run(cfgs['PI', 0])
            self.assertFalse(os.path.exists(d2l._lc_job(cfgs['PI', 3])))

    def test_set_scheme(self):
        m = toy_model()
        d2l._lc_set_scheme(m, 'yarn', 4, 64)
        self.assertEqual(m.max_len, 256)
        self.assertAlmostEqual(m.rope_mscale, d2l.yarn_mscale(4))
        self.assertEqual(m.config()['rope_scheme'], 'yarn')
        d2l._lc_set_scheme(m, 'yarn', 2, 64, mscale=1.0)
        self.assertEqual((m.max_len, m.rope_mscale), (128, 1.0))
        d2l._lc_set_scheme(m, 'none', 4, 64)
        self.assertIsNone(m.rope_inv_freq)

    def test_old_timeline(self):
        """An arm trained before the periodic needle evaluation: its
        timeline rows (no 'probe' field) are not read."""
        with tempfile.TemporaryDirectory() as job:
            with open(os.path.join(job, 'timeline.jsonl'), 'w') as f:
                f.write(json.dumps(dict(arm='PI', seed=0, step=64,
                                        recall_lo=0.1)) + '\n')
            self.assertEqual(len(d2l._lc_read_timeline(job)), 0)
            self.assertEqual(list(d2l._lc_read_timeline(job).columns),
                             d2l._LC_COLUMNS['timeline'])


class IdTokenizer:
    """A tiktoken-like encoding whose texts are space-separated ids."""
    n_vocab = V
    special_tokens_set = {'<|endoftext|>'}

    def encode_single_token(self, s):
        return 0

    def encode_ordinary(self, t):
        return [int(x) for x in t.split()]

    def encode_ordinary_batch(self, ts):
        return [self.encode_ordinary(t) for t in ts]


def fixture_shards(out, n_docs, lo, hi, seed, ids=None):
    """Documents of lo to hi random tokens (from ids; default 32 to V)."""
    rng = np.random.default_rng(seed)
    ids = np.arange(32, V) if ids is None else np.asarray(ids)
    docs = [(f'd{i}', ' '.join(map(str, rng.choice(ids, rng.integers(
        lo, hi + 1)))), {'part': 'pg19'}) for i in range(n_docs)]
    d2l.write_shards(docs, IdTokenizer(), out, 'train',
                     shard_tokens=10 ** 9, eos_id=0)
    return d2l.TokenShards(out)


TIERS = {t: dict(trained=512, length=1024, needles=3, zs_needles=2,
                 loss_windows=4, contexts=(256, 512), eval_needles=2,
                 eval_windows=2) for t in ('t1', 't2')}
ARMS = {'PI': dict(scheme='pi', stages=((1024, 2, 1.0),), seeds=(0, 1)),
        'YaRN': dict(scheme='yarn', stages=((1024, 2, 1.0),), seeds=(0, 1)),
        'YaRN-prog': dict(scheme='yarn', stages=((512, 1, 0.5),
                                                 (1024, 2, 0.5)),
                          seeds=(0,))}
CALL = dict(tokens=4 * 2048, seq_len=1024, batch_tokens=2048)
REF = 'Qwen3-0.6B-Base'


VS = 128  # the toy campaigns' vocabulary (the fixture word tokenizer's)


def small_model(seed=0, max_len=512):
    torch.manual_seed(seed)
    return d2l.PackedGPT.from_arch(TOY._replace(T=max_len, vocab=VS)).eval()


class ArmsBase(unittest.TestCase):
    """Fixture data in the fixture word tokenizer (vocabulary 128, so the
    toy forwards are cheap), a toy base at 512 and the patches of a toy
    campaign."""
    @classmethod
    def setUpClass(cls):
        cls.tok = fixture_tokenizer()
        assert len(cls.tok) <= VS
        filler = [cls.tok.ids[' ' + w] for w in FILLER] + [1]
        cls.tmp = tempfile.TemporaryDirectory()
        t = cls.tmp.name
        cls.shards = dict(
            long=fixture_shards(os.path.join(t, 'long'), 60, 300, 2400, 0,
                                filler),
            short=fixture_shards(os.path.join(t, 'short'), 200, 5, 300, 1,
                                 filler))
        cls.books = fixture_books(cls.tok, 6, 1024)
        cls.val = torch.from_numpy(np.random.default_rng(2).choice(
            filler[:-1], (4, 513)))
        cls.base = os.path.join(t, 'base.safetensors')
        d2l._save_bf16(cls.base, small_model(), 0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def patches(self, root, rerun='lc-t1', only=None):
        items = pd.DataFrame(dict(task=['t1', 't1', 't2'],
                                  item_id=['a', 'b', 'c'],
                                  score=[1.0, 0.0, 1.0]))
        val = self.val
        zs = dict(base='r2', schemes=d2l._LC_SCHEMES, lengths=(512, 1024))
        ref = dict(model_id='Qwen/' + REF, lengths=(512, 1024))
        env = {'D2L_RERUN': rerun, 'D2L_PRETRAIN_ONLY': only or ''}
        return [
            mock.patch.object(torch.cuda, 'is_available', lambda: False),
            mock.patch.object(d2l, '_job_root', lambda: root),
            mock.patch.object(d2l, '_rerun_dir',
                              lambda: os.path.join(root, 'rerun')),
            mock.patch.object(d2l, '_lc_base_path', lambda base: self.base),
            mock.patch.object(d2l, 'long_data', lambda tier: d2l.LcData(
                None, self.shards, '')),
            books_patch(self.books),
            mock.patch.object(d2l, '_needle_words', lambda: POOL),
            mock.patch.object(d2l, '_val_windows',
                              lambda T, tokens=None: (val, 0)),
            mock.patch.object(d2l, '_rare_ids', lambda: torch.arange(VS)),
            mock.patch.object(d2l, '_cpt_suite', lambda *a, **k: 'suite'),
            mock.patch.object(d2l, '_suite_items',
                              lambda *a, **k: items.copy()),
            mock.patch.object(d2l, '_lc_zeroshot_call', lambda: zs),
            mock.patch.object(d2l, '_lc_reference_call', lambda: ref),
            mock.patch.object(d2l, '_lc_load_reference', lambda m: (
                small_model(seed=5, max_len=1024), self.tok)),
            mock.patch.object(d2l, 'r2_tokenizer', lambda: self.tok),
            mock.patch.dict(d2l._LC_TIERS, TIERS),
            mock.patch.dict(d2l._LC_MICRO, {'t1': 2, 't2': 2}),
            mock.patch.object(d2l, '_LC_EVAL_EVERY', 2),
            mock.patch.object(d2l, '_LC_STATE_EVERY', 3),
            mock.patch.dict(os.environ, env)]

    @contextlib.contextmanager
    def campaign(self, root, rerun='lc-t1', only=None):
        with contextlib.ExitStack() as stack, \
                contextlib.redirect_stdout(io.StringIO()):
            for p in self.patches(root, rerun, only):
                stack.enter_context(p)
            yield


def check_display(test, out, first=None):
    lines = out.splitlines()
    if first is not None:
        test.assertEqual(lines[0], first, out)
    test.assertLessEqual(max(map(len, lines)), 100, out)
    test.assertNotRegex(out, r'Empty|-0\.00\b|dtype|nan|NaN', out)
    return lines


class EvalIsolationTest(ArmsBase):
    """The periodic needle evaluation leaves training untouched."""
    def cfg(self, evaluate):
        cfg = d2l._lc_configs({'PI': ARMS['PI']}, 'r2', **CALL)['PI', 1]
        return dict(cfg, evaluate=evaluate)

    def test_state_untouched(self):
        with tempfile.TemporaryDirectory() as root, self.campaign(root):
            cfg = self.cfg(True)
            stream = d2l._lc_stream(self.shards, cfg)
            stream.batch(0)
            model = d2l._load_bf16(self.base, torch.float32).train()
            d2l._lc_set_stage(model, cfg, 0)
            torch.manual_seed(7)
            np.random.seed(7)
            random.seed(7)

            def state():
                return (torch.get_rng_state(), np.random.get_state()[1].copy(),
                        random.getstate(), repr(stream.state_dict()))

            before, weights = state(), {k: v.clone() for k, v in
                                        model.state_dict().items()}
            row = d2l._lc_timeline_row(model, cfg, 0, d2l.lc_tokenizer())
            after = state()
            self.assertTrue(torch.equal(before[0], after[0]))
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2:], after[2:])
            self.assertTrue(model.training)
            for k, v in model.state_dict().items():
                self.assertTrue(torch.equal(v, weights[k]), k)
            self.assertEqual(set(d2l._LC_COLUMNS['timeline']) - set(row), set())
            self.assertLessEqual(row['far_n'] + row['near_n'], 16)

    def test_same_training(self):
        logs, weights = [], []
        with tempfile.TemporaryDirectory() as root, self.campaign(root):
            for evaluate in (False, True):
                cfg = self.cfg(evaluate)
                job = os.path.join(root, f'eval-{evaluate}')
                model = d2l._load_bf16(self.base, torch.float32).train()
                log = d2l._lc_train(model, d2l._lc_stream(self.shards, cfg),
                                    cfg, job)
                logs.append(log.set_index('step')['loss'].to_numpy())
                weights.append(d2l._load_bf16(os.path.join(
                    job, 'ckpt', f's{cfg["steps"]}.safetensors')))
                tl = d2l._lc_read_timeline(job)
                self.assertEqual(sorted(tl['step']), [0, 2, 4] if evaluate
                                 else [])
        np.testing.assert_array_equal(logs[0], logs[1])
        for (k, a), b in zip(weights[0].state_dict().items(),
                             weights[1].state_dict().values()):
            self.assertTrue(torch.equal(a, b), k)


class PairingTest(ArmsBase):
    """Claim 5 pairs YaRN with PI by seed: the seed fixes the data order,
    which the scheme does not touch."""
    def test_same_data_order(self):
        with tempfile.TemporaryDirectory() as root, self.campaign(root):
            cfgs = d2l._lc_configs(ARMS, 't1', **CALL)

            def batches(cfg):
                stream = d2l._lc_stream(self.shards, cfg)
                return [stream.batch(k) for k in range(cfg['steps'])]
            pi, yarn, prog = (batches(cfgs[k]) for k in (
                ('PI', 0), ('YaRN', 0), ('YaRN-prog', 0)))
            other = batches(cfgs['YaRN', 1])
        for a, b, c, d in zip(pi, yarn, prog, other):
            for k in ('input_ids', 'doc_ids', 'pos_ids'):
                self.assertTrue(torch.equal(a[k], b[k]), k)
            self.assertTrue(torch.equal(a['input_ids'], c['input_ids']))
            self.assertFalse(torch.equal(a['input_ids'], d['input_ids']))


class SchemeDifferenceTest(unittest.TestCase):
    """Claim 5's two intervals: seeds that agree on fixed evaluation windows
    give a narrow t interval; resampling the windows gives the wider one."""
    def runs(self, n_win=64, n_q=200, seeds=(0, 1, 2, 3, 4)):
        rng = np.random.default_rng(0)
        base = rng.normal(4.0, 0.5, n_win)  # per-window loss of PI
        effect = rng.normal(-0.04, 0.05, n_win)  # YaRN - PI varies by window
        w, n = [], []
        for s in seeds:
            for m, shift in (('PI', 0.0), ('YaRN', effect)):
                loss = base + shift + rng.normal(0, 1e-4, n_win)
                w.append(pd.DataFrame(dict(
                    model=m, seed=s, measure='nll', bucket='3-4K',
                    window=np.arange(n_win), sum=loss * 1024.0, n=1024.0)))
                for L, gaps in ((1024, (100, 800)), (4096, (100, 2000))):
                    k = np.arange(n_q)
                    n.append(pd.DataFrame(dict(
                        model=m, seed=s, length=L, window=k // 4,
                        needle=k % 4, gap=np.where(k % 2, *gaps),
                        correct=rng.random(n_q) < (0.6 if m == 'YaRN'
                                                   else 0.5))))
        return d2l.LcRuns(**{f: pd.DataFrame() for f in d2l.LcRuns._fields}
                          )._replace(windows=pd.concat(w, ignore_index=True),
                                     needles=pd.concat(n, ignore_index=True))

    def test_window_interval_wider(self):
        d = d2l.scheme_difference(self.runs(), 'YaRN', 'PI')
        self.assertEqual(d['seeds'], [0, 1, 2, 3, 4])
        t, w = d['loss'], d['windows']['loss']
        self.assertAlmostEqual(t[0], w[0], places=9)
        self.assertLess(t[2] - t[1], 0.01 * (w[2] - w[1]))
        self.assertTrue(w[1] < w[0] < w[2])
        self.assertEqual(d2l._lc_wider(d, 'loss'), (*w, 'windows'))
        for k in ('near', 'far'):  # recall: paired by query, same point
            self.assertAlmostEqual(d[k][0], d['windows'][k][0], places=9)
            self.assertTrue(d['windows'][k][1] < d['windows'][k][2])

    def test_one_seed(self):
        d = d2l.scheme_difference(self.runs(seeds=(0,)), 'YaRN', 'PI')
        self.assertTrue(math.isnan(d['loss'][1]))
        self.assertEqual(d2l._lc_wider(d, 'loss')[3], 'windows')

    def test_zero_rule(self):
        d = dict(x=(0.02, -0.01, 0.05), windows=dict(x=(0.02, 0.001, 0.2)))
        self.assertEqual(d2l._lc_wider(d, 'x')[3], 'seeds')  # contains 0
        d = dict(x=(0.1, 0.01, 0.19), windows=dict(x=(0.1, 0.09, 0.11)))
        self.assertEqual(d2l._lc_wider(d, 'x')[3], 'seeds')  # wider
        d = dict(x=(0.1, 0.09, 0.11), windows=dict(x=(0.1, 0.05, 0.15)))
        self.assertEqual(d2l._lc_wider(d, 'x')[3], 'windows')


class DisplayFormatTest(unittest.TestCase):
    """F35 review findings 4 and 5: a criterion met before the first step
    prints 'at start'; a bound that would round to 0 shows its sign."""
    def test_bounds(self):
        self.assertEqual(d2l._interval(0.4386, 0.0033, 0.874, 2, True),
                         '+0.44 [+0.00, +0.87]')
        self.assertEqual(d2l._interval(0.4386, 0.0033, 0.874, 2, True, True),
                         '+0.44 [+0.003, +0.87]')
        self.assertEqual(d2l._interval(-1.3e-4, -4.2e-4, 1.6e-4, 3, True,
                                       True), '+0.000 [-0.0004, +0.0002]')
        self.assertEqual(d2l._interval(0.17, 0.0, 0.333, 2, True, True),
                         '+0.17 [+0.00, +0.33]')  # an exact 0 stays 0.00
        self.assertEqual(d2l._interval(0.1, 1e-9, 0.2, 2, True, True),
                         '+0.10 [+0.00000, +0.20]')  # at most 3 more
        self.assertEqual(d2l._interval(0.1, -0.004, 0.2, 2, True, True),
                         '+0.10 [-0.004, +0.20]')
        self.assertEqual(d2l._bound(math.nan, 2), '-')

    def test_first_steps(self):
        spec = d2l._LC_TIERS['t1']
        T, L0 = spec['length'], spec['trained']
        zs = pd.DataFrame(dict(
            model='none', measure='nll', bucket=d2l.bucket_name(0, L0),
            window=[0, 1], sum=[3.0, 3.0], n=[1.0, 1.0]))
        steps = [0, 64, 128]
        tl = pd.DataFrame(dict(
            arm='PI', seed=1, length=T, step=steps,
            tokens=[s * 1_000_000 for s in steps],
            nll_last=[4.0, 3.2, 3.01], near_lo=[0.1, 0.2, 0.6],
            far_lo=[0.3, 0.3, 0.3], gain_lo=[-0.1, -0.1, -0.1]))
        logs = pd.DataFrame(dict(arm=['PI', 'PI', 'YaRN'], seed=[0, 1, 0]))
        runs = d2l.LcRuns(**{f: pd.DataFrame()
                             for f in d2l.LcRuns._fields})._replace(
            zs_windows=zs, timeline=tl, logs=logs, tier='t1',
            needles=pd.DataFrame(dict(model=['PI'])))
        table, missing = d2l.lc_first_steps(runs, 0.05)
        self.assertEqual(table.iloc[0].tolist(),
                         ['PI', 1, '128M', '128M', 'at start',
                          'not reached'])
        self.assertEqual(missing, ['PI s0', 'YaRN s0'])


class ArmsEndToEndTest(ArmsBase):
    def test_t1_campaign(self):
        """A toy T1 campaign through lc_arms with D2L_RERUN=lc-t1, and the
        page's displays of a T2 result on it as the labelled stand-in."""
        with tempfile.TemporaryDirectory() as root:
            with self.campaign(root):
                runs = d2l.lc_arms(ARMS, base='t1', **CALL)
                self.assertTrue(os.path.exists(os.path.join(
                    root, 'lc', 't1-yarn-prog-s0', 'DONE')))
                self.assertFalse(os.path.exists(os.path.join(
                    root, 'lc', 't1-pi-s1', 'state')))
                self.assertTrue(os.path.exists(os.path.join(
                    root, 'lc', 't1-base', 'evals.json.gz')))
                stored = d2l._read_result(os.path.join(
                    root, 'rerun', d2l._next_key('lm-lc-t1-v2') + '.json.gz'),
                    'json-table')
                shown = self.displays(runs)
        self.assertEqual(runs.tier, 't1')
        t = runs.table
        self.assertEqual(list(t['arm']), ['PI', 'YaRN', 'YaRN-prog', 'base'])
        self.assertEqual(list(t.columns), d2l._LC_COLUMNS['table'])
        self.assertEqual(t['Δ val'].iloc[-1], 0.0)
        self.assertEqual(len(runs.timeline), 0)  # T1: no periodic evaluation
        prog = runs.logs[runs.logs['arm'] == 'YaRN-prog']
        self.assertEqual(list(prog['step']), list(range(1, 5)))
        self.assertEqual(list(prog['stage']), [0] * 2 + [1] * 2)
        r = runs.recall
        self.assertEqual(set(r['seed']), {-1, 0, 1})
        self.assertEqual(set(r[r['model'] == 'base']['length']), {512})
        self.assertEqual(set(r[r['model'] == 'YaRN-prog']['length']),
                         {512, 1024})
        self.assertIn('PI, zero-shot', set(r['model']))
        self.assertEqual(list(runs.zs_table['scheme']),
                         list(d2l._LC_SCHEMES) + ['none'])
        self.assertEqual(stored.zs_config['lengths'], [512, 1024])
        self.assertEqual(list(runs.short['arm']), ['PI', 'YaRN', 'YaRN-prog'])
        self.assertEqual(list(stored.table['arm']), list(t['arm']))
        self.assertEqual(stored.table.attrs['config']['tokens'], 8192)
        self.assertEqual(stored.tier, 't1')
        self.assertEqual(set(runs.context['context']), {'256', '512',
                                                         'full'})
        d = d2l.scheme_difference(runs, 'YaRN', 'PI')
        self.assertEqual(d['seeds'], [0, 1])
        n = runs.needles
        unpaired = runs._replace(needles=n[~((n['model'] == 'YaRN')
                                             & (n['seed'] == 1))])
        with self.assertRaisesRegex(ValueError, 'pairs the arms by seed'):
            d2l.scheme_difference(unpaired, 'YaRN', 'PI')
        label = '[T2 run pending; showing the T1 configuration]'
        for cell, out in shown.items():
            if cell != 'seeds':
                check_display(self, out, label)
        self.assertEqual(shown['first-steps'].splitlines(), [label])
        self.assertIn('±', shown['zeroshot'])
        zl = shown['zeroshot'].splitlines()
        pairs = len(d2l._LC_ZS_PAIRS)
        self.assertEqual(len(zl), 2 + len(d2l._LC_SCHEMES) + 1 + pairs)
        self.assertIn('1K − 0-512', zl[1])
        self.assertIn('recall 0.5-1K', zl[1])
        control = zl[-1 - pairs]
        # claim 1b reads the control's Wilson bounds: [lo, hi], not ±
        self.assertRegex(control, r'^none at 512 \(control\): recall '
                                  r'128-256 \d\.\d\d \[\d\.\d\d, \d\.\d\d\], '
                                  r'256-512 \d\.\d\d \[\d\.\d\d, \d\.\d\d\]; '
                                  r'reach [\d,]+$')
        for line, (a, b) in zip(zl[-pairs:], d2l._LC_ZS_PAIRS):
            self.assertRegex(line, rf'^{a} − {b}, same windows: loss 0-512 '
                                   r'[-+]\d\.\d\d \[[-+]\d\.\d\d, '
                                   r'[-+]\d\.\d\d\], recall 32-512 '
                                   r'[-+]\d\.\d\d \[')
        z = runs.zs_table
        pi = z[(z['scheme'] == 'PI') & (z['length'] == 1024)].iloc[0]
        self.assertAlmostEqual(pi['Δ'], pi['last'] - pi['first'], places=6)
        self.assertTrue(pi['Δ lo'] <= pi['Δ'] <= pi['Δ hi'])
        self.zs_diff_checks(d2l._lc_zs_as_tuple(runs))
        self.assertIn('YaRN-prog − YaRN, seed 0 of each (single seed): '
                      'loss 0.5-1K ', shown['arms'])
        self.assertIn('PI (2 seeds)', shown['arms'])
        self.assertIn('YaRN-prog (1 seed)', shown['arms'])
        lines = check_display(self, shown['seeds'])
        self.assertEqual(lines[0].split()[0], 'arm')
        self.assertTrue(lines[0].endswith('32-64 at 512'), lines[0])
        self.assertEqual([x.split(' (')[0].split()[0] for x in lines[1:5]],
                         ['PI', 'YaRN', 'YaRN-prog', 'base'])
        self.assertIn('PI (2 seeds)', lines[1])
        self.assertIn('YaRN-prog (1 seed)', lines[3])
        r = runs.recall
        c = r[(r['model'] == 'base') & (r['length'] == 512)
              & (r['gap_lo'] == 32) & (r['seed'] == -1)].iloc[0]
        self.assertTrue(lines[4].endswith(d2l._pm(c['rate'], c['lo'],
                                                  c['hi'], 2)), lines[4])
        self.assertIn('near at 1K', lines[0])
        self.assertTrue(lines[5].startswith('YaRN − PI, T1 (2 seeds): loss '))
        self.assertRegex(lines[5], r'\((seeds|windows)\)$')
        self.assertTrue(lines[6].startswith('  near '))
        self.assertRegex(lines[6], r'^  near .* \((seeds|windows)\), far .* '
                                   r'\((seeds|windows)\)$')
        self.assertEqual(lines[7:], ['T2: pending'])

    def zs_diff_checks(self, zs):
        """Claim 4's paired scheme differences: the loss point estimate
        equals the table's difference and the near one the difference of the
        pooled near recalls over _LC_NEAR (the schemes share windows and
        queries), the intervals contain them, and schemes on other windows,
        token counts or queries are refused; a pair with an absent scheme
        prints 'not scored'."""
        z = zs.table.set_index('scheme')
        T = int(z['length'].max())
        z = z[z['length'] == T]
        near = lambda m: d2l._lc_range(zs.needles, m, T, *d2l._LC_NEAR)[0]
        for a, b in d2l._LC_ZS_PAIRS:
            d = d2l._lc_zs_diff(zs, a, b)
            self.assertAlmostEqual(d['loss'][0], z.loc[a, 'first']
                                   - z.loc[b, 'first'], places=6)
            self.assertAlmostEqual(d['near'][0], near(a) - near(b), places=6)
            for k in ('loss', 'near'):
                self.assertTrue(d[k][1] <= d[k][0] <= d[k][2], (a, b, k, d))
        w, n = zs.windows, zs.needles
        pi = (w['model'] == 'PI') & (w['measure'] == 'nll')
        other = zs._replace(windows=w[~(pi & (w['window'] == 0))])
        with self.assertRaisesRegex(ValueError, 'same windows'):
            d2l._lc_zs_diff(other, 'YaRN', 'PI')
        fewer = w.copy()
        fewer.loc[pi[pi].index[0], 'n'] -= 1
        with self.assertRaisesRegex(ValueError, 'same windows'):
            d2l._lc_zs_diff(zs._replace(windows=fewer), 'YaRN', 'PI')
        q = n[(n['model'] == 'PI') & (n['length'] == T)
              & (n['gap'] > d2l._LC_NEAR[0]) & (n['gap'] <= d2l._LC_NEAR[1])]
        self.assertGreater(len(q), 1)
        with self.assertRaisesRegex(ValueError, r'same windows \(queries\)'):
            d2l._lc_zs_diff(zs._replace(needles=n.drop(q.index[0])), 'YaRN',
                            'PI')
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            d2l.show_lc_zeroshot(zs, pairs=(('YaRN', 'XYZ'),))
        self.assertEqual(buf.getvalue().splitlines()[-1],
                         'YaRN − XYZ: not scored')

    def test_part_sizes(self):
        """The zero-shot and reference parts store their evaluation sizes
        and weights: read back at the same sizes, refused by the merge's
        reader under other sizes (a smoke's patched tier). The calls are
        looked up inside the campaign, which patches them."""
        parts = (('lc-zeroshot', lambda: d2l._lc_zeroshot_call(),
                  lambda r: d2l.LcZeroShot(r.table, r.windows, r.needles,
                                           '')),
                 ('lc-reference', lambda: d2l._lc_reference_call(),
                  lambda r: d2l.LcReference(r.windows, r.needles, '')))
        with tempfile.TemporaryDirectory() as root:
            with self.campaign(root, 'lc-zeroshot,lc-reference'):
                zs = d2l.lc_zeroshot(**d2l._lc_zeroshot_call())
                d2l.lc_reference(**d2l._lc_reference_call())
                for part, call, as_tuple in parts:
                    self.assertTrue(os.path.exists(d2l._lc_part_path(part)))
                    _, cfg = d2l._lc_part_for_merge(part, call(), None,
                                                    as_tuple)
                    self.assertEqual((cfg['needles'], cfg['windows'],
                                      cfg['contexts']), (2, 4, [256, 512,
                                                             1024]))
                    self.assertIn('weights' if part == 'lc-zeroshot'
                                  else 'revision', cfg)
            for change in (dict(zs_needles=1), dict(loss_windows=2)):
                small = {t: dict(v, **change) for t, v in TIERS.items()}
                with self.campaign(root, ''), \
                        mock.patch.dict(d2l._LC_TIERS, small):
                    for part, call, as_tuple in parts:
                        with self.assertRaisesRegex(
                                ValueError, 'needles|windows'):
                            d2l._lc_part_for_merge(part, call(), None,
                                                   as_tuple)
        self.assertEqual(len(zs.table), len(d2l._LC_SCHEMES) + 1)

    def displays(self, runs):
        """The page's displays of T2 results on runs as the T1 stand-in."""
        with mock.patch.object(d2l, 'load_artifact', lambda name: runs):
            lc = d2l._as_tuple(d2l.LcRuns, d2l._pending_result(
                'lc-t2-evals', 'lc-t1'), d2l._LC_COLUMNS)
            zs = d2l._lc_zs_as_tuple(d2l._pending_result('lc-zeroshot',
                                                         'lc-t1'))
        ns = page_ns(lc=lc, t1=runs, zs=zs)
        out = {}
        for name, code in (('zeroshot', 'd2l.show_lc_zeroshot(zs)'),
                           ('arms', 'd2l.show_lc_table(lc)'),
                           ('seeds', 'd2l.show_lc_seeds(t1, lc)')):
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                exec(code, ns)
            out[name] = buf.getvalue()
        for cell in ('position', 'recall-grid', 'gain', 'first-steps',
                     'short-cost'):
            out[cell] = run_cell(f'long-context-{cell}', ns)
        d2l.plt.close('all')
        return out

    def test_t2_parts(self):
        """A toy T2 campaign through the parts: --only t2-base keeps the
        base's evaluation in its folder; the arms run as part jobs (pending
        until the merge); the merge reads the kept evaluation and folds in
        the zero-shot and reference parts."""
        parts = ','.join(d2l._LC_T2_NAMES[k] for k in
                         d2l._lc_configs(ARMS, 'r2', **CALL))
        calls = []
        real = d2l._lc_eval

        def counted(*a, **k):
            calls.append(a[3])
            return real(*a, **k)

        with tempfile.TemporaryDirectory() as root:
            with self.campaign(root, 'lc-t2-evals', only='t2-base'):
                res = d2l.lc_arms(ARMS, base='r2', **CALL)
            self.assertIn('pending', res.label)
            kept = os.path.join(root, 'lc', 't2-base', 'evals.json.gz')
            self.assertTrue(os.path.exists(kept))
            with self.campaign(root, 'lc-t2-evals', only='t2-pi-s7'), \
                    self.assertRaisesRegex(ValueError, 'only'):
                d2l.lc_arms(ARMS, base='r2', **CALL)
            with self.campaign(root, parts):
                res = d2l.lc_arms(ARMS, base='r2', **CALL)
            self.assertIn('pending', res.label)
            mtime = os.path.getmtime(kept)
            with self.campaign(root, 'lc-t2-evals'), \
                    mock.patch.object(d2l, '_lc_eval', counted):
                runs = d2l.lc_arms(ARMS, base='r2', **CALL)
            self.assertEqual(os.path.getmtime(kept), mtime)
            self.assertTrue(os.path.exists(os.path.join(
                root, 'rerun', d2l._next_key('lm-lc-evals-v2') + '.json.gz')))
        # 5 arms, 5 zero-shot schemes, the reference; not the base again
        self.assertEqual(len(calls), 11)
        self.assertEqual(runs.label, '')
        self.assertEqual(runs.tier, 't2')
        t = runs.table
        self.assertEqual(list(t['arm']), ['PI', 'YaRN', 'YaRN-prog', 'base',
                                          REF])
        self.assertTrue(math.isnan(t['val'].iloc[-1]))
        tl = runs.timeline
        self.assertEqual(sorted(tl[(tl['arm'] == 'PI') & (tl['seed'] == 1)]
                                ['step']), [0, 2, 4])
        prog = tl[tl['arm'] == 'YaRN-prog'].sort_values('step')
        self.assertEqual(list(prog['length']), [512, 512, 1024])  # stages
        self.assertEqual(set(tl[tl['arm'] == 'PI']['length']), {1024})
        self.assertEqual(set(runs.ref_needles['model']), {REF})
        self.assertIn(REF, set(runs.context['model']))
        with tempfile.TemporaryDirectory() as root, \
                self.campaign(root, ''):  # the toy tiers
            self.t2_displays(runs)

    def t2_displays(self, runs):
        for show, n in ((d2l.show_lc_table, 8), (d2l.show_lc_short, 4)):
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                show(runs)
            self.assertEqual(len(check_display(self, buf.getvalue())), n)
        ns = page_ns(lc=runs, t1=runs)
        for cell in ('position', 'recall-grid', 'gain', 'first-steps',
                     'short-cost'):
            out = run_cell(f'long-context-{cell}', ns)
            if cell == 'recall-grid':
                self.assertIn(f'{REF}: reach ', out)
                self.assertRegex(out, r'(?m)^PI: reach [\d,]+, detectable '
                                      r'to [\d,]+; far − base top octave '
                                      r'[-+]\d\.\d\d \[')
                ref = out[out.index(f'{REF} at 1K by gap:'):].splitlines()
                self.assertEqual(len(ref), 2)
                self.assertTrue(ref[0].endswith('],'))
                self.assertTrue(ref[1].startswith('  ') and
                                ref[1][2].isdigit(), ref)
                self.assertEqual(len(re.findall(r'[\dK] [\d.]+ \[', ' '.join(
                    ref))), len(d2l.gap_bins(1024)))
                self.assertIn(f'\n{REF} at 1K by gap: 32-64 ', out)
                self.assertNotIn(f'{REF}: reach 0, detectable to 0; far', out)
            if cell != 'position':
                check_display(self, out)
        steps = run_cell('long-context-first-steps', ns).splitlines()
        self.assertEqual(steps[0].split(), ['arm', 'seed', 'loss', 'near',
                                            'recall', 'far', 'recall',
                                            'gain'])
        self.assertEqual(len(steps), 6)
        d2l.plt.close('all')

    def test_t2_first_build(self):
        """F35's campaign: seed 0 of PI and YaRN and the progressive arm
        are the first build's finished folders (trained without the
        periodic needle evaluation, an old-format timeline beside them);
        the part jobs read them and never retrain them or touch their
        weights; only seed 1 trains, after the recipe check against seed
        0's arm.json; the merge evaluates every model with the new probe;
        the timeline holds seed 1 only and the page says which arms have
        none; the T2 scheme difference pairs the two seeds."""
        trained, checked = [], []
        real_train, real_check = d2l._lc_train, d2l._lc_check_recipe

        def train(model, stream, cfg, job, stop=None):
            trained.append((cfg['arm'], cfg['seed'], cfg['evaluate']))
            return real_train(model, stream, cfg, job, stop)

        def check(cfg):
            checked.append((cfg['arm'], cfg['seed']))
            return real_check(cfg)

        with tempfile.TemporaryDirectory() as root:
            with self.campaign(root, ''):
                cfgs = d2l._lc_configs(ARMS, 'r2', **CALL)  # toy cadence
                first = [k for k in cfgs if k[1] == 0]
                for k in first:  # the first build: no periodic evaluation
                    d2l._lc_arm_run(dict(cfgs[k], evaluate=False))
                    with open(os.path.join(d2l._lc_job(cfgs[k]),
                                           'timeline.jsonl'), 'w') as f:
                        f.write(json.dumps(dict(arm=k[0], seed=0, step=2,
                                                recall_lo=0.1)) + '\n')
                ckpts = {k: os.path.join(d2l._lc_job(c), 'ckpt',
                                         f's{c["steps"]}.safetensors')
                         for k, c in cfgs.items() if k in first}
                before = {k: (d2l._file_id(p), os.path.getmtime(p))
                          for k, p in ckpts.items()}
            parts = ','.join(d2l._LC_T2_NAMES[k] for k in cfgs)
            with self.campaign(root, parts), \
                    mock.patch.object(d2l, '_lc_train', train), \
                    mock.patch.object(d2l, '_lc_check_recipe', check):
                res = d2l.lc_arms(ARMS, base='r2', **CALL)
            self.assertIn('pending', res.label)
            self.assertEqual(sorted(trained), [('PI', 1, True),
                                               ('YaRN', 1, True)])
            self.assertEqual(sorted(checked), [('PI', 1), ('YaRN', 1)])
            with self.campaign(root, 'lc-t2-evals'):
                runs = d2l.lc_arms(ARMS, base='r2', **CALL)
            self.assertEqual(sorted(trained), [('PI', 1, True),
                                               ('YaRN', 1, True)])
            for k, p in ckpts.items():  # read, never written
                self.assertEqual((d2l._file_id(p), os.path.getmtime(p)),
                                 before[k], k)
                self.assertTrue(os.path.exists(os.path.join(
                    os.path.dirname(os.path.dirname(p)), 'evals.json.gz')))
            with self.campaign(root, ''):
                ns = page_ns(lc=runs, t1=runs)
                steps = run_cell('long-context-first-steps', ns)
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    d2l.show_lc_seeds(runs, runs)  # T1 slot: not t1
                seeds = buf.getvalue()
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    d2l.show_lc_table(runs)
                table = buf.getvalue()
        self.assertEqual(runs.label, '')
        self.assertEqual(
            sorted(set(zip(runs.timeline['arm'], runs.timeline['seed']))),
            [('PI', 1), ('YaRN', 1)])
        self.assertEqual(set(runs.needles['seed']), {0, 1})
        self.assertEqual(set(runs.needles[runs.needles['model'] ==
                                          'YaRN-prog']['length']),
                         {512, 1024})
        lines = check_display(self, steps)
        self.assertEqual(len(lines), 4)  # header, PI s1, YaRN s1, missing
        self.assertEqual([x.split()[:2] for x in lines[1:3]],
                         [['PI', '1'], ['YaRN', '1']])
        self.assertEqual(lines[3], 'no timeline: PI s0, YaRN s0, '
                                   'YaRN-prog s0')
        lines = check_display(self, seeds)
        self.assertIn('T1: pending', lines)
        self.assertRegex(seeds, r'(?m)^YaRN − PI, T2 \(2 seeds\): loss '
                                r'[-+]\d\.\d{3} \[.*\] \((seeds|windows)\)$')
        self.assertEqual(d2l.scheme_difference(runs, 'YaRN', 'PI')['seeds'],
                         [0, 1])
        lines = check_display(self, table)
        self.assertEqual([x.split(' (')[0].strip() for x in lines[1:4]],
                         ['PI', 'YaRN', 'YaRN-prog'])
        self.assertIn('PI (2 seeds)', table)
        self.assertIn('YaRN-prog (1 seed)', table)
        self.assertRegex(table, r'YaRN-prog − YaRN, seed 0 of each \(single '
                                r'seed\)')
        d2l.plt.close('all')

    def test_resume_equal(self):
        cfg = dict(d2l._lc_configs(ARMS, 't1', 12 * 2048, 1024, 2048, 1e-3,
                                   2, 0.25, 0.5)['YaRN-prog', 0],
                   micro_batch=2, evaluate=False, trained_len=512)
        logs, ckpts = [], []
        with tempfile.TemporaryDirectory() as root, \
                contextlib.redirect_stdout(io.StringIO()), \
                mock.patch.object(d2l, '_LC_STATE_EVERY', 3):
            for name, stops in (('straight', (None,)),
                                ('resumed', (4, None))):
                job = os.path.join(root, name)
                for stop in stops:
                    model = d2l._load_bf16(self.base, torch.float32).train()
                    log = d2l._lc_train(model, d2l._lc_stream(self.shards,
                                                              cfg), cfg, job,
                                        stop=stop)
                logs.append(log.set_index('step')['loss'])
                ckpts.append(d2l._load_bf16(os.path.join(
                    job, 'ckpt', 's12.safetensors')))
        self.assertEqual(len(logs[0]), 12)
        np.testing.assert_allclose(logs[0].loc[4:].to_numpy(),
                                   logs[1].loc[4:].to_numpy(), rtol=1e-6)
        self.assertEqual(ckpts[1].rope_scheme, 'yarn')
        self.assertEqual(ckpts[1].max_len, 1024)
        self.assertAlmostEqual(ckpts[1].rope_mscale, d2l.yarn_mscale(2))


class PageCellsTest(unittest.TestCase):
    """The page's CPU cells, run from the page."""
    def test_scheme_checks(self):
        ns = page_ns()
        exec(page_cells()['long-context-rope-inv-freq'], ns)
        with mock.patch.object(d2l, 'rope_inv_freq', ns['rope_inv_freq']):
            out = run_cell('long-context-scheme-checks', ns)
        lines = out.splitlines()
        self.assertLessEqual(float(lines[0].rsplit(' ', 1)[1]), 1e-7)
        self.assertEqual(lines[1], 'PI equals θ/s: True')
        self.assertEqual(lines[2], 'YaRN pairs: 9 interpolated, 12 blended, '
                                   '11 untouched')
        self.assertEqual(lines[3], 'yarn_mscale(4) = 1.1386')
        self.assertTrue(lines[4].startswith('vs Transformers'))
        self.assertLessEqual(max(map(len, lines)), 80)

    def test_attention_share(self):
        out = run_cell('long-context-attention-share', page_ns())
        lines = out.splitlines()
        self.assertEqual(lines[0], 'F, dense attention: 1.51 / 2.19 / 3.55 '
                         'GFLOP per token at 4K / 8K / 16K')
        self.assertEqual(lines[1], 'attention share: 0.45 / 0.62 / 0.77 '
                         '(causal half: 0.29 / 0.45 / 0.62)')
        self.assertIn('-15%', lines[2])
        self.assertEqual(lines[3], 'FlopCounterMode at 16K (dense scores as '
                         'well): ratio 1.000')
        self.assertIn('no document mask: ×2.35', lines[4])
        self.assertLessEqual(max(map(len, lines)), 80)

    def test_measure_checks(self):
        torch.manual_seed(0)
        ns = page_ns(model=toy_model(max_len=256), s=4, L=64,
                     books=torch.randint(32, V, (4, 257)))
        out = run_cell('long-context-measure-checks', ns).splitlines()
        self.assertLess(float(out[0].rsplit(' ', 1)[1]), 1e-6)
        self.assertTrue(out[1].startswith('long-context gain: oracle +7.6'))
        self.assertTrue(out[1].endswith('unigram 0.000'))

    def test_task_checks(self):
        tok = r2_tok()
        ns = page_ns(tok=tok)
        with books_patch(r2_books(tok, 20, 16_384)), \
                mock.patch.object(d2l, '_needle_words',
                                  lambda: r2_words(tok)):
            out = run_cell('long-context-task-checks', ns).splitlines()
        self.assertRegex(out[0], r'^window: 16,384 tokens, 8 needles, gaps '
                                 r'\d+ to [\d,]+; \d+ keys, [\d,]+ '
                                 r'values$')
        self.assertEqual(out[1], 'oracle copier: 100%, 100% at 4K, 16K; the '
                                 'pilot checks hold at both')
        self.assertLessEqual(max(map(len, out)), 80)


class PendingPageTest(unittest.TestCase):
    """The page's later cells run on pending results (empty tables)."""
    def test_cells(self):
        names = {'lc-t1', 'lc-zeroshot', 'lc-reference', 'lc-probe',
                 'lc-t2-evals'}
        hub = {k: v for k, v in d2l.DATA_HUB.items()
               if not k.startswith(('lm-lc-t1-', 'lm-lc-evals-',
                                    'lm-lc-probe-'))}
        ns = page_ns()
        with mock.patch.object(d2l, 'PRETRAIN_PENDING', frozenset(names)), \
                mock.patch.dict(os.environ, {'D2L_RERUN': ''}), \
                mock.patch.dict(d2l.DATA_HUB, hub, clear=True):
            for cell in ('format-table', 'reach-train'):
                out = run_cell(f'long-context-{cell}', ns)
                self.assertEqual(out.splitlines(), ['[lc-probe pending]'])
            out = run_cell('long-context-zeroshot', ns)
            self.assertEqual(out.splitlines(), ['[lc-zeroshot pending]'])
            run_cell('long-context-schedule', ns)
            for cell in ('arms', 'position', 'recall-grid', 'gain',
                         'first-steps', 'short-cost'):
                out = run_cell(f'long-context-{cell}', ns)
                self.assertEqual(out.splitlines(), ['[lc-t2-evals pending]'],
                                 cell)
            out = run_cell('long-context-t1', ns)
            self.assertEqual(out.splitlines(), ['[lc-t1 pending]'])
            ref = d2l.lc_reference(**d2l._lc_reference_call())
            self.assertEqual(ref.label, 'lc-reference pending')
        self.assertIn('pending', ns['lc'].label)

    def test_fallback(self):
        """While the T2 arms are pending, the T1 result under its label."""
        t1 = d2l.LcRuns(**{f: pd.DataFrame() for f in d2l.LcRuns._fields},
                        )._replace(zs_config=None, ref_config=None, tier='t1',
                                   label='smoke')
        with mock.patch.object(d2l, 'load_artifact', lambda name: t1):
            res = d2l._pending_result('lc-t2-evals', 'lc-t1')
            zs = d2l._lc_zs_as_tuple(d2l._pending_result('lc-zeroshot',
                                                         'lc-t1'))
        self.assertEqual(res.label, 'T2 run pending; showing the T1 '
                                    'configuration (smoke)')
        self.assertEqual(zs.label, res.label)


class JobsTest(unittest.TestCase):
    """The regeneration jobs call what the page calls."""
    def literal(self, name):
        code = '\n'.join(page_cells().values())
        m = re.search(rf'^{name} = (\{{.*?\}})$', code, re.S | re.M)
        return eval(re.sub(r'#.*', '', m.group(1)), {'dict': dict})

    def test_calls(self):
        self.assertEqual(self.literal('ARMS'), d2l._lc_t2_call()['arms'])
        self.assertEqual(self.literal('ARMS_T1'), d2l._lc_t1_call()['arms'])
        self.assertEqual(self.literal('SCHEMES'),
                         d2l._lc_zeroshot_call()['schemes'])
        cells = page_cells()
        self.assertIn("d2l.lc_zeroshot('r2', schemes=SCHEMES, "
                      "lengths=(4096, 16_384))",
                      cells['long-context-zeroshot'])
        self.assertEqual(d2l._lc_zeroshot_call()['lengths'], (4096, 16384))
        self.assertIn("d2l.lc_arms(ARMS, base='r2', tokens=TOKENS, "
                      "seq_len=16_384,\n                 batch_tokens=262_144,"
                      " peak_lr=1e-4, warmup=20,\n                 "
                      "decay_frac=0.15, long_share=0.5)",
                      cells['long-context-arms'])
        self.assertIn('TOKENS = 100_000_000', cells['long-context-schedule'])
        self.assertIn("d2l.lc_arms(ARMS_T1, base='t1', tokens=16_000_000, "
                      "seq_len=4096,\n                 batch_tokens=65_536)",
                      cells['long-context-t1'])
        self.assertIn("d2l.lc_probe_runs(t1_steps=(509, 1017, 1526),\n"
                      "                          r2_steps=(500, 1000, 2000, "
                      "3000, 4000, 6000))", cells['long-context-format-table'])
        self.assertEqual(d2l._lc_probe_call(), dict(
            t1_steps=(509, 1017, 1526),
            r2_steps=(500, 1000, 2000, 3000, 4000, 6000)))
        t2 = set(d2l._LC_T2_NAMES.values()) | {'lc-t2-evals'}
        for k in t2:
            self.assertEqual(d2l.PRETRAIN_JOBS[k][0], 'lc_arms')
            self.assertEqual(d2l.PRETRAIN_JOBS[k][1](), d2l._lc_t2_call())
            self.assertEqual(d2l._RERUN_ARTIFACT[k], 'lc-evals')
        for k, f, drv in (('lc-t1', d2l._lc_t1_call, 'lc_arms'),
                          ('lc-zeroshot', d2l._lc_zeroshot_call,
                           'lc_zeroshot'),
                          ('lc-reference', d2l._lc_reference_call,
                           'lc_reference'),
                          ('lc-probe', d2l._lc_probe_call, 'lc_probe_runs')):
            self.assertEqual(d2l.PRETRAIN_JOBS[k], (drv, mock.ANY))
            self.assertEqual(d2l.PRETRAIN_JOBS[k][1](), f())
        self.assertEqual(t2 | {'lc-t1', 'lc-zeroshot', 'lc-reference',
                               'lc-probe'},
                         {n for n in d2l.PRETRAIN_RERUN_NAMES
                          if n.startswith('lc-')})
        self.assertIn(d2l._LC_REFERENCE, d2l.HF_REVISIONS)
        import importlib.util  # lm_regen's expandable-segments jobs exist
        spec = importlib.util.spec_from_file_location('lm_regen', os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'lm_regen.py'))
        regen = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(regen)
        self.assertEqual(regen.EXPANDABLE, {'lc-reference', 'lc-t2-evals'})
        self.assertLessEqual(regen.EXPANDABLE, set(d2l.PRETRAIN_JOBS))
        self.assertEqual(d2l.PRETRAIN_ARTIFACTS['pg19-test'][0],
                         'lm-pg19-test-v2')
        self.assertIn('lm-pg19-test-v2', d2l.DATA_HUB)


class DataTest(unittest.TestCase):
    """The builder's local shards, when present (no download)."""
    def test_local(self):
        with in_chapter():
            if not all(os.path.exists(os.path.join(p, 'manifest.json'))
                       for p in d2l._LC_LOCAL.values()):
                self.skipTest('local lc shards not built')
            test = d2l.TokenShards(d2l._LC_LOCAL['pg19-test'], 'train')
            self.assertEqual(test.n_docs, 100)
            self.assertGreater(test.n_tokens, 10_000_000)
            with mock.patch.object(d2l, '_lc_shards',
                                   lambda name, m=None: test), \
                    mock.patch.dict(d2l._LC_BOOKS, clear=True):
                torch.manual_seed(1)
                state = torch.get_rng_state()
                books = d2l.test_books(64, 16_384)
                self.assertTrue(torch.equal(state, torch.get_rng_state()))
                self.assertEqual(tuple(books.shape), (64, 16_385))
                self.assertFalse((books == 0).any())
                self.assertTrue(torch.equal(d2l.test_books(16, 16_384),
                                            books[:16]))
                self.assertFalse(torch.equal(d2l.test_books(16, 16_384,
                                                            seed=1),
                                             books[:16]))
                whole = [b for b in (test.doc(i)[:-1] for i in range(8))
                         if len(b) > 16_385]
                for i in range(4):  # window i from the i-th long book
                    b, w = whole[i], books[i].numpy()
                    hits = np.flatnonzero(b[:-16_384] == w[0])
                    self.assertTrue(any(np.array_equal(b[a:a + 16_385], w)
                                        for a in hits), i)
            long = d2l.TokenShards(d2l._LC_LOCAL['lc-long'], 'train',
                                   max_tokens=8_000_000)
            self.assertEqual(long.n_shards, 1)
            self.assertEqual(set(long.meta(0)['part']), {'pg19'})


if __name__ == '__main__':
    unittest.main()
