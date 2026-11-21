#!/usr/bin/env python3
"""CPU tests for sec_lm_dedup (chapter_lm-data/dedup.md and the dedup cells of
crawl-lib.md): the MinHash signature bit for bit (the fixture hash assignment
A3 grades with, and a Python-integer reference), the split multiplication, empty
signatures, the parallel signature driver, banding against brute force,
union-find against scipy, suffix and LCP arrays against the definition,
maximal repeats, the 13-gram index, near_dedup on planted clusters, best_br,
the labelled-pair generator's round trip, the Poisson-binomial bin test of the
S-curve, and the planted decontamination documents."""
import hashlib
import itertools
import unittest

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from d2l import torch as d2l

# The A3 pin: SHA-256 of the concatenated little-endian bytes of the k = 128,
# seed 0xD2 signatures of d2l.MINHASH_FIXTURE's five strings. Written here as
# a literal so a change to the library constant alone cannot pass.
MINHASH_FIXTURE_SHA256 = (
    'b9b07bf416ce76f2b5cf1ae3d12fc617bdfa5bc7e464230c6bca1ada3bda0195')
P = d2l.MINHASH_PRIME


def ref_signature(shingle_list, k=128, seed=0xD2):
    """The specification with Python integers (the reference)."""
    if not shingle_list:
        return None
    a, c = np.random.default_rng(seed).integers(1, P, size=(2, k),
                                                dtype=np.uint64)
    xs = {int.from_bytes(hashlib.blake2b(s.encode('utf-8'), digest_size=8)
                         .digest(), 'little') % P for s in shingle_list}
    return np.array([min((int(ai) * x + int(ci)) % P for x in xs)
                     for ai, ci in zip(a, c)], dtype=np.uint64)


def random_doc(rng, vocab, n=200):
    return ' '.join(vocab[i] for i in rng.integers(0, len(vocab), n))


class MinHashTest(unittest.TestCase):
    def test_fixture_bit_exact(self):
        sigs = [d2l.minhash_signature(d2l.shingles(t), 128)
                for t in d2l.MINHASH_FIXTURE]
        for t, s in zip(d2l.MINHASH_FIXTURE, sigs):
            np.testing.assert_array_equal(s, ref_signature(d2l.shingles(t)))
        h = hashlib.sha256(b''.join(s.astype('<u8').tobytes()
                                    for s in sigs)).hexdigest()
        self.assertEqual(h, MINHASH_FIXTURE_SHA256)
        self.assertEqual(d2l.MINHASH_FIXTURE_SHA256, MINHASH_FIXTURE_SHA256)

    def test_mulmod(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, P, 10**4, dtype=np.uint64)
        x = rng.integers(0, P, 10**4, dtype=np.uint64)
        ref = np.array([(int(u) * int(v)) % P for u, v in zip(a, x)],
                       dtype=np.uint64)
        np.testing.assert_array_equal(d2l._mulmod61(a, x), ref)
        e = np.array([0, 1, 2, P - 2, P - 1, 2**32 - 1, 2**32, 2**60],
                     dtype=np.uint64)
        A, X = np.meshgrid(e, e)
        ref = np.array([[(int(u) * int(v)) % P for u, v in zip(r, s)]
                        for r, s in zip(A, X)], dtype=np.uint64)
        np.testing.assert_array_equal(d2l._mulmod61(A, X), ref)

    def test_shingles_and_empty(self):
        self.assertEqual(d2l.shingles('Four words only here'), [])
        self.assertIsNone(d2l.minhash_signature(
            d2l.shingles('Four words only here')))
        # NFC and lowercase: composed and decomposed forms agree
        a = d2l.shingles('Café au lait, served HOT in the morning')
        b = d2l.shingles('café au lait served hot in the morning')
        self.assertEqual(a, b)
        self.assertEqual(a[0], 'café au lait served hot')
        self.assertEqual(len(d2l.shingles('a b c d e f g', 5)), 3)
        self.assertEqual(d2l.jaccard_exact([], []), 0.0)
        self.assertEqual(d2l.containment({1, 2}, {1, 2, 3, 4}), 1.0)
        self.assertEqual(d2l.jaccard_exact({1, 2}, {1, 2, 3, 4}), 0.5)

    def test_minhash_texts(self):
        rng = np.random.default_rng(1)
        vocab = [f'w{i}' for i in range(500)]
        texts = [random_doc(rng, vocab, int(rng.integers(0, 60)))
                 for _ in range(1100)] + ['too short']
        for workers in (1, 2):
            got = d2l._minhash_texts(texts, 64, (0xD2, 0xD3), 5, workers)
            for seed in (0xD2, 0xD3):
                for t, s in zip(texts, got[seed]):
                    want = d2l.minhash_signature(d2l.shingles(t), 64, seed)
                    if want is None:
                        self.assertIsNone(s)
                    else:
                        np.testing.assert_array_equal(s, want)

    def test_seed_rule(self):
        rng = np.random.default_rng(2)
        vocab = [f'w{i}' for i in range(300)]
        base = [random_doc(rng, vocab, 80) for _ in range(40)]
        texts = base + [t + ' extra words at the end' for t in base[:10]]
        for s in (0, 1):
            sigs = [d2l.minhash_signature(d2l.shingles(t), 128, 0xD2 + s)
                    for t in texts]
            pairs, _ = d2l.lsh_candidates(sigs, *d2l.best_br(128, 0.8))
            lab = d2l.union_find_clusters(len(texts), pairs)
            want = [i for i in range(len(texts)) if lab[i] == i]
            self.assertEqual(d2l.near_dedup(texts, seed=s), want)


class BandingTest(unittest.TestCase):
    def test_candidates_brute_force(self):
        rng = np.random.default_rng(0)
        sigs = [rng.integers(0, 4, 12).astype(np.uint64) for _ in range(300)]
        sigs[7] = None
        for b, r in ((3, 4), (4, 3), (6, 2)):
            got, rep = d2l.lsh_candidates(sigs, b, r)
            def band(i, k):
                return sigs[i][k * r:(k + 1) * r]
            want = {(i, j) for i, j in itertools.combinations(range(300), 2)
                    if sigs[i] is not None and sigs[j] is not None and any(
                        (band(i, k) == band(j, k)).all() for k in range(b))}
            self.assertEqual(got, want)
            self.assertEqual(rep['cut'], [])
        S, ok = d2l._sig_matrix(sigs, 12)
        ia, ib = np.triu_indices(300, 1)
        match = d2l._band_match(S[ia], S[ib], 4, 3) & ok[ia] & ok[ib]
        got, _ = d2l.lsh_candidates(sigs, 4, 3)
        self.assertEqual({(i, j) for i, j, m in zip(ia, ib, match) if m}, got)

    def test_max_bucket(self):
        sigs = [np.zeros(4, dtype=np.uint64) for _ in range(50)]
        pairs, rep = d2l.lsh_candidates(sigs, 2, 2, max_bucket=10)
        self.assertEqual(rep['cut'], [(0, 50), (1, 50)])
        self.assertLessEqual(len(pairs), 2 * 45)
        again, _ = d2l.lsh_candidates(sigs, 2, 2, max_bucket=10)
        self.assertEqual(pairs, again)  # seeded

    def test_union_find(self):
        rng = np.random.default_rng(0)
        for n in (1, 10, 500):
            pairs = {tuple(sorted(rng.choice(n, 2, replace=n < 2).tolist()))
                     for _ in range(n // 2)} if n > 1 else set()
            lab = d2l.union_find_clusters(n, pairs)
            if pairs:
                i, j = np.array(list(pairs)).T
                g = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
            else:
                g = coo_matrix((n, n))
            _, ref = connected_components(g, directed=False)
            for c in np.unique(ref):  # same partition, smallest index labels
                m = np.flatnonzero(ref == c)
                self.assertTrue((lab[m] == m.min()).all())

    def test_best_br_and_scurve(self):
        self.assertEqual(d2l.best_br(128, 0.8), (9, 13))
        self.assertAlmostEqual(float(d2l.scurve(0.5, 2, 2)), 0.4375)
        self.assertAlmostEqual(float(d2l.scurve(0.9, 2, 2)), 0.9639)
        self.assertEqual(round(float(d2l.scurve(0.8, 14, 8)), 4), 0.9235)
        self.assertEqual(d2l.best_br(128, 1.0), (1, 128))
        self.assertLess(d2l.best_br(128, 0.8, w_fn=5.0)[1], 13)

    def test_near_dedup_planted(self):
        rng = np.random.default_rng(0)
        vocab = [f'w{i}' for i in range(5000)]
        docs, cluster = [], []
        for c in range(100):
            words = random_doc(rng, vocab).split()
            docs.append(' '.join(words))
            for _ in range(2):  # one substituted word: J about 0.95
                w = list(words)
                w[int(rng.integers(20, 180))] = 'changed'
                docs.append(' '.join(w))
            cluster += [c] * 3
        order = rng.permutation(len(docs))
        docs = [docs[i] for i in order]
        cluster = np.array(cluster)[order]
        want = sorted(np.flatnonzero(cluster == c).min() for c in range(100))
        self.assertEqual(d2l.near_dedup(docs, 0.8), want)
        distinct = [random_doc(rng, vocab) for _ in range(300)]
        self.assertEqual(d2l.near_dedup(distinct, 1.0), list(range(300)))


class SCurveBinTest(unittest.TestCase):
    def test_bin_mean_not_bin_centre(self):
        rng = np.random.default_rng(0)
        # a steep bin of (9, 13), its pairs bunched at the lower edge
        J = np.concatenate([rng.uniform(0.80, 0.81, 320),
                            rng.uniform(0.84, 0.85, 80)])
        p = d2l.scurve(J, 9, 13)
        lo, hi = d2l._pb_interval(p)
        c_lo, c_hi = d2l._pb_interval(np.full(len(J), float(
            d2l.scurve(0.825, 9, 13))))
        obs = (rng.random((2000, len(J))) < p).sum(axis=1)
        inside = np.mean((obs >= lo) & (obs <= hi))
        centre = np.mean((obs >= c_lo) & (obs <= c_hi))
        self.assertGreaterEqual(inside, 0.93)
        self.assertLess(centre, inside - 0.05)

    def test_pb_interval_binomial(self):
        for n, q in ((50, 0.3), (200, 0.02), (10, 0.5)):
            lo, hi = d2l._pb_interval(np.full(n, q))
            self.assertEqual(lo, int(stats.binom.ppf(0.025, n, q)))
            self.assertEqual(hi, int(stats.binom.ppf(0.975, n, q)))
        self.assertEqual(d2l._pb_interval([1.0, 1.0, 0.0]), (2, 2))


class SuffixArrayTest(unittest.TestCase):
    def test_against_definition(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            x = rng.integers(0, 3, int(rng.integers(1, 2001))).tolist()
            sa = d2l.suffix_array(x)
            self.assertEqual(sa.tolist(),
                             sorted(range(len(x)), key=lambda i: x[i:]))
            lcp = d2l.lcp_array(x, sa)
            for i in range(1, len(x)):
                a, b = x[sa[i - 1]:], x[sa[i]:]
                h = next((k for k in range(min(len(a), len(b)))
                          if a[k] != b[k]), min(len(a), len(b)))
                self.assertEqual(lcp[i], h)
        self.assertEqual(d2l.suffix_array([]).tolist(), [])
        self.assertEqual(d2l.suffix_array([5]).tolist(), [0])

    def test_repeated_spans(self):
        rng = np.random.default_rng(1)
        span = rng.integers(100, 200, 60)
        seq = np.concatenate([rng.integers(0, 100, 300), span,
                              rng.integers(0, 100, 200), span,
                              rng.integers(0, 100, 100)])
        sa = d2l.suffix_array(seq)
        spans = d2l.repeated_spans(seq, sa, d2l.lcp_array(seq, sa), 50)
        self.assertEqual(spans[0][1:], (60, 2))
        self.assertEqual(seq[spans[0][0]:spans[0][0] + 60].tolist(),
                         span.tolist())
        self.assertTrue(all(length >= 50 for _, length, _ in spans))
        self.assertEqual(len(spans), 1)  # shifted copies are not maximal


class NgramIndexTest(unittest.TestCase):
    def test_planted_13gram(self):
        rng = np.random.default_rng(0)
        vocab = [f'w{i}' for i in range(5000)]
        gram = ' '.join(f'z{i}' for i in range(13))
        index = d2l.NgramIndex(13).add('Intro. ' + gram + '!', 'item-1')
        short = d2l.NgramIndex(13).add(' '.join(gram.split()[:12]), 'x')
        for _ in range(10):
            words = random_doc(rng, vocab, 300).split()
            at = int(rng.integers(0, 300))
            text = ' '.join(words[:at] + [gram.upper()] + words[at:])
            self.assertEqual(index.query(text), [('item-1', 1)])
            self.assertEqual(short.query(text), [])
            self.assertEqual(index.query(' '.join(words)), [])


class PairsTest(unittest.TestCase):
    def test_round_trip(self):
        rng = np.random.default_rng(0)
        vocab = [f'w{i}' for i in range(2000)]
        rows = []
        for i in range(60):
            paras = [random_doc(rng, vocab, int(rng.integers(20, 80)))
                     for _ in range(int(rng.integers(1, 5)))]
            rows.append(dict(id=f'd{i}', url=f'https://s{i}.example.org/a',
                             text='\n'.join(paras), role='stream'))
        for h in range(10):
            for k in range(3):
                rows.append(dict(id=f'h{h}-{k}',
                                 url=f'https://host{h}.example.org/{k}',
                                 text=random_doc(rng, vocab, 100),
                                 role='host_pair'))
        docs = pd.DataFrame(rows)
        spec = dict(d2l.DUP_PAIRS_SPEC, n_base=20, n_neg_host=15,
                    n_neg_random=15)
        pairs = d2l.make_dup_pairs(docs, spec, seed=0)
        self.assertEqual(len(pairs), 20 * 3 + 30)
        self.assertEqual(pairs, d2l.make_dup_pairs(docs, spec, seed=0))
        texts = dict(zip(docs.id, docs.text))
        vocab_s = d2l._dup_vocab(docs.text[docs.role == 'stream'])
        for row in pairs:
            a, b = d2l.pair_texts(row, texts, vocab_s)
            J = d2l.jaccard_exact(d2l.shingles(a), d2l.shingles(b))
            self.assertLessEqual(abs(J - row['J']), 1e-12)
            self.assertEqual(row['label'], J >= 0.8)
            self.assertIn(row['split'], ('train', 'test'))
        ops = {r['op'] for r in pairs}
        self.assertTrue({'neg_host', 'neg_random'} <= ops)
        for r in pairs:
            if r['op'] == 'neg_host':
                self.assertEqual(r['a'].split('-')[0], r['b'].split('-')[0])

    def test_perturbations(self):
        text = 'one two three four five six seven eight nine ten\nsecond para'
        vocab = ['zz']
        self.assertEqual(d2l._dup_perturb(text, 'trunc', dict(seed=0,
                                                              frac=0.5)),
                         'one two three four five six')
        sub = d2l._dup_perturb(text, 'sub', dict(seed=0, rate=0.1), vocab)
        self.assertEqual(sub.count('zz'), 1)
        dele = d2l._dup_perturb(text, 'del', dict(seed=0, rate=0.1))
        self.assertEqual(len(dele.split()), len(text.split()) - 1)
        boiler = d2l._dup_perturb(text, 'boiler', dict(seed=3, words=25))
        self.assertEqual(len(boiler.split()), len(text.split()) + 25)
        shuf = d2l._dup_perturb(text, 'shuffle', dict(seed=1))
        self.assertEqual(sorted(shuf.split('\n')), sorted(text.split('\n')))


class PlantsTest(unittest.TestCase):
    def test_plants_and_documents(self):
        items = []
        for bench in ('gsm8k', 'arc', 'mmlu'):
            for i in range(5):
                q = ' '.join(f'{bench}{i}q{j}' for j in range(15)) + '?'
                ch = None if bench == 'gsm8k' else [
                    (lab, f'option {lab} of {i}') for lab in 'ABCD']
                items.append(dict(item_id=f'{bench}-{i}', benchmark=bench,
                                  question=q, choices=ch,
                                  gold='7' if ch is None else 'B',
                                  text=q + '\nanswer'))
        stream = pd.DataFrame(dict(
            id=[f's{i}' for i in range(30)],
            text=[f'Sentence {i}. Another one here! Last line\nmore {i}.'
                  for i in range(30)]))
        rows = d2l.make_plants(items, stream, lambda t: [x[::-1] for x in t],
                               n_per=2, n_controls=5, seed=0)
        self.assertEqual(len(rows), 2 * 3 * 3 + 5)
        self.assertEqual(len({r['doc_id'] for r in rows}), len(rows))
        docs = d2l.contam_documents(rows, stream)
        index = d2l.NgramIndex(13)
        for it in items:
            index.add(it['text'], it['item_id'])
        for r in rows:
            hits = dict(index.query(docs[r['doc_id']]))
            if r['form'] in ('verbatim', 'reformat'):
                self.assertIn(r['item_id'], hits)
            if r['form'] == 'control':
                self.assertEqual(hits, {})
        arc = next(it for it in items if it['benchmark'] == 'arc')
        ref = d2l._plant_text(arc, 'reformat')
        self.assertIn('(a) option d of 0', ref)  # reversed, relabelled
        self.assertTrue(ref.endswith('answer: c'))  # B's text is now C

    def test_word_windows(self):
        text = ' '.join(str(i) for i in range(300))
        w = d2l.word_windows(text, 128, 64)
        self.assertEqual([s for s, _ in w], [0, 64, 128, 172])
        self.assertTrue(w[-1][1].endswith('299'))
        self.assertEqual(d2l.word_windows('a b', 128, 64), [(0, 'a b')])


if __name__ == '__main__':
    unittest.main()
