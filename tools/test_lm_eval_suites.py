#!/usr/bin/env python3
"""CPU tests for sec_lm_eval_suites (chapter_lm-pretrained/eval-suites.md
and its build-only companion eval-suites-lib.md).

Library code (spec 18.5 part 5), numbered as the spec's unit tests:
1-5. `bootstrap_ci`: Monte Carlo error across bootstrap seeds, agreement
     with the t-interval, the cluster/iid width ratio at DEFF = 4, the ratio
     estimator with unequal clusters, the errors and the warning, and
     chunking of the count matrices.
6-8. `paired_bootstrap`: identical vectors, the paired-t interval, and the
     bootstrap p-value against the sign-flip test.
9.   `EvalSuite.hash` changes with every manifest field the spec lists and
     not with `batch_size`; `item_seed` across processes with different
     `PYTHONHASHSEED`; `subset_ids` is prefix-stable.
10.  The run cache: an interrupted run resumes to the same per-item scores;
     another seed or batch size misses the cache.
11.  `EvalResult.compare` raises on different suites and item sets and
     aligns shuffled frames.
12.  `d2l.generate` with per-row seeds and a stub model of fixed logits.
13.  `judge_pairwise` with a stub judge (log-odds beta + delta * sign):
     recovery, swap, identical answers, packing, mass, truncation warning,
     `logits_to_keep=1`; and with a tiny random Transformers Qwen3 and the
     real Qwen3 and Qwen3.5 tokenizers.
14.  Debiasing under noise and the total-least-squares slope test.
15.  `bradley_terry`: recovery, a disconnected graph, batched weights.
16.  `min_detectable_effect`.
17.  Registry smoke test over the ten `EVAL_TASKS` (network; skipped unless
     `D2L_NET_TESTS=1`).
Also: the adapters on hand-written rows, the frozen manifests, manifest and
result round trips, dispatch by kind (mc, mc_partial, gen, code), the
summary and macro intervals, and `import d2l.torch` without Transformers.

Datasets are small parquet files written to a temporary folder, which
`EvalTask.dataset` may name instead of a Hub repository. The tokenizer
tests read the Qwen3-0.6B and Qwen3.5-0.8B tokenizers from the HF cache
and skip when they are missing.

Run: `.venv-lm/bin/python -m unittest tools/test_lm_eval_suites.py`, or
`make test-lm`.
"""

import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET = os.environ.get('D2L_NET_TESTS') == '1'
T_499 = 1.9647  # t quantile 0.975 at 499 degrees of freedom


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def width(r):
    return r.ci[1] - r.ci[0]


# ------------------------------------------------------------- fixtures
def arc_rows(n, seed=0):
    """ARC-shaped rows: id, question, choices {label, text}, answerKey; a
    few items use the numeric labels '1'..'4' as ARC does."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        labels = ['1', '2', '3', '4'] if i % 7 == 3 else list('ABCD')
        rows.append({'id': f'Q_{i:04d}', 'question': f'Question {i}?',
                     'choices': {'label': labels,
                                 'text': [f'opt{i}-{k}' for k in range(4)]},
                     'answerKey': labels[int(rng.integers(4))]})
    return rows


def gsm_rows(n):
    return [{'question': f'What is {i} + {i}?',
             'answer': f'{i} + {i} = <<{i}+{i}={2 * i}>>{2 * i}\n#### '
                       f'{2 * i:,}'} for i in range(n)]


def write_dataset(folder, files):
    """files: {relative path: list of row dicts} as parquet."""
    for rel, rows in files.items():
        path = os.path.join(folder, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
    return folder


class Fixture(unittest.TestCase):
    """A temporary folder with an ARC-like and a GSM8K-like dataset."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = self.tmp.name
        self.arc_dir = write_dataset(os.path.join(root, 'arc'), {
            'test-0.parquet': arc_rows(64), 'train-0.parquet': arc_rows(8)})
        self.gsm_dir = write_dataset(os.path.join(root, 'gsm'), {
            'main/test-0.parquet': gsm_rows(30),
            'main/train-0.parquet': gsm_rows(6)})
        self.arc = d2l.EvalTask('arc_fx', self.arc_dir, 'rev0',
                                ('test-0.parquet', 'train-0.parquet'), 'mc',
                                'arc', norm='char')
        self.gsm = d2l.EvalTask(
            'gsm_fx', self.gsm_dir, 'rev0',
            ('main/test-0.parquet', 'main/train-0.parquet'), 'gen', 'gsm8k',
            template='qa', shots=2,
            shots_from={'split': 'train', 'ids': [0, 1]},
            max_new_tokens=16, stop=('\n\nQ:',), extract='gsm8k')


class TinyLM(nn.Module):
    """Stands in for a model: run() reads only parameters and attributes."""
    def __init__(self, name='tiny/model', revision='r1'):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.name_or_path, self.revision = name, revision


def fake_scores(prompts, choices):
    """Deterministic scores from the texts (a stub for score_choices)."""
    out = torch.full((len(prompts), max(map(len, choices))), -float('inf'))
    for i, (p, cs) in enumerate(zip(prompts, choices)):
        for j, c in enumerate(cs):
            out[i, j] = (sum(map(ord, p + c)) * 7919 % 1000) / 1000
    return out


# ------------------------------------------------------- bootstrap (1-5)
class TestBootstrapCI(unittest.TestCase):
    def test_1_monte_carlo_error(self):
        x = np.random.default_rng(0).normal(size=500)
        r0, r1 = (d2l.bootstrap_ci(x, 10000, seed=s) for s in (0, 1))
        for e0, e1 in zip(r0.ci, r1.ci):
            self.assertLess(abs(e0 - e1), 0.03 * width(r0))

    def test_2_t_interval(self):
        x = np.random.default_rng(0).normal(size=500)
        r = d2l.bootstrap_ci(x)
        half = T_499 * x.std(ddof=1) / np.sqrt(len(x))
        for e, t in zip(r.ci, (x.mean() - half, x.mean() + half)):
            self.assertLess(abs(e - t), 0.05 * 2 * half)
        self.assertEqual((r.n, r.n_clusters, r.n_boot), (500, 500, 10000))
        self.assertIsNone(r.p_two_sided)
        self.assertAlmostEqual(r.mean, x.mean(), places=12)

    def test_3_cluster_width_ratio(self):
        # G = 500 clusters of m = 10, accuracies Beta(1, 1): rho = 1/3,
        # DEFF = 1 + 9/3 = 4, so the cluster interval is twice as wide.
        rng = np.random.default_rng(0)
        acc = rng.beta(1, 1, size=500)
        y = (rng.random((500, 10)) < acc[:, None]).astype(float).ravel()
        cluster = np.repeat(np.arange(500), 10)
        ratio = (width(d2l.bootstrap_ci(y, 4000, cluster)) /
                 width(d2l.bootstrap_ci(y, 4000)))
        self.assertLess(abs(ratio - 2.0), 0.15 * 2.0)

    def test_4_ratio_estimator_unequal_clusters(self):
        sizes = np.tile(np.arange(1, 10), 5)  # 45 clusters of 1 to 9
        cluster = np.repeat(np.arange(len(sizes)), sizes)
        y = np.random.default_rng(1).random(len(cluster)).round(2)
        r = d2l.bootstrap_ci(y, 2000, cluster)
        self.assertEqual(r.mean, y.mean())
        self.assertEqual((r.n, r.n_clusters), (len(y), 45))
        # cluster labels need not be sorted or numeric
        labels = np.array([f'c{c}' for c in cluster])[::-1]
        r2 = d2l.bootstrap_ci(y[::-1], 2000, labels)
        self.assertEqual(r2.mean, y[::-1].mean())

    def test_5_errors_warning_chunks(self):
        y = np.random.default_rng(2).random(300)
        with self.assertRaisesRegex(ValueError, '29 clusters'):
            d2l.bootstrap_ci(y, 100, np.arange(300) % 29)
        with self.assertRaisesRegex(ValueError, 'NaN'):
            d2l.bootstrap_ci(np.r_[y, np.nan], 100)
        with self.assertRaisesRegex(ValueError, 'cluster ids'):
            d2l.bootstrap_ci(y, 100, np.arange(299))
        with self.assertRaisesRegex(ValueError, 'at least 2'):
            d2l.bootstrap_ci([1.0], 100)
        with self.assertRaisesRegex(ValueError, 'pair them'):
            d2l.paired_bootstrap(y, y[:-1], 100)
        with self.assertWarnsRegex(UserWarning, '50 items'):
            d2l.bootstrap_ci(y[:50], 100)
        with self.assertWarnsRegex(UserWarning, '50 items'):
            d2l.paired_bootstrap(y[:50], y[50:100], 100)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            d2l.bootstrap_ci(y, 100)  # 300 items: no warning

    def test_5_chunking(self):
        # The count matrices are drawn in chunks; the replicates are those
        # of one unchunked matrix with the same seed (1,000 items).
        y = np.random.default_rng(3).random(1000)
        _, full, _ = d2l._bootstrap_replicates(y, 10000, None, 0,
                                               chunk=10**12)
        _, parts, _ = d2l._bootstrap_replicates(y, 10000, None, 0,
                                                chunk=10**5)  # 100 rows
        # (equal counts; the matrix products may round differently in
        # the last bit when the chunk shapes differ)
        np.testing.assert_allclose(full, parts, rtol=1e-13)
        # 10^4 items: the default chunks (2,000 + 200 rows) against chunks
        # of 1,000 rows
        y = np.random.default_rng(4).random(10**4)
        a = d2l.bootstrap_ci(y, 2200)
        _, m, _ = d2l._bootstrap_replicates(y, 2200, None, 0, chunk=10**7)
        np.testing.assert_allclose(a.ci, np.quantile(m, [0.025, 0.975]),
                                   rtol=1e-13)


# ------------------------------------------------- paired bootstrap (6-8)
def paired_binary_set(seed=0, n=500):
    """Two correlated binary score vectors on the same n items: a shared
    item difficulty, model a 0.25 logits stronger."""
    rng = np.random.default_rng(seed)
    difficulty = rng.normal(size=n)
    a = rng.random(n) < 1 / (1 + np.exp(-(difficulty + 0.25)))
    b = rng.random(n) < 1 / (1 + np.exp(-difficulty))
    return a.astype(float), b.astype(float)


class TestPairedBootstrap(unittest.TestCase):
    def test_6_identical(self):
        x = np.random.default_rng(0).random(200)
        r = d2l.paired_bootstrap(x, x)
        self.assertEqual((r.mean, r.ci, r.se, r.p_two_sided),
                         (0.0, (0.0, 0.0), 0.0, 1.0))
        self.assertEqual((r.n, r.n_clusters), (200, 200))

    def test_7_paired_t(self):
        a, b = paired_binary_set()
        d = a - b
        r = d2l.paired_bootstrap(a, b)
        half = T_499 * d.std(ddof=1) / np.sqrt(len(d))
        for e, t in zip(r.ci, (d.mean() - half, d.mean() + half)):
            self.assertLess(abs(e - t), 0.05 * 2 * half)
        self.assertAlmostEqual(r.mean, d.mean(), places=12)
        # pairing narrows the interval relative to two separate ones
        sep = np.hypot(*(d2l.bootstrap_ci(s).se for s in (a, b)))
        self.assertLess(r.se, sep)

    def test_8_sign_flip(self):
        # The sign-flip test of the statistics appendix, 10^4 flips. The
        # tolerance is the Monte Carlo error of the two p-values; it holds
        # on 16 of 16 data seeds with p < 0.5 (checked on 20 data seeds).
        a, b = paired_binary_set()
        d = a - b
        r = d2l.paired_bootstrap(a, b)
        signs = np.random.default_rng(0).choice([-1, 1], size=(10000, len(d)))
        p = np.mean(np.abs((signs * d).mean(1)) >= abs(d.mean()))
        self.assertTrue(0.01 < p < 0.5)
        self.assertLess(abs(r.p_two_sided - p), 3 * np.sqrt(2 * p * (1 - p)
                                                            / 1e4))

    def test_cluster_paired(self):
        # paired differences clustered: G = 100 problems by 8 samples
        rng = np.random.default_rng(5)
        acc = rng.beta(2, 2, size=(100, 1))
        a = (rng.random((100, 8)) < acc).astype(float).ravel()
        b = (rng.random((100, 8)) < acc * 0.9).astype(float).ravel()
        cl = np.repeat(np.arange(100), 8)
        iid, clu = (d2l.paired_bootstrap(a, b, 4000, c) for c in (None, cl))
        self.assertEqual((clu.n, clu.n_clusters), (800, 100))
        self.assertEqual(iid.mean, clu.mean)


# --------------------------------------------------------------- MDE (16)
class TestMDE(unittest.TestCase):
    def test_16_values(self):
        m = d2l.min_detectable_effect(np.sqrt(0.2), 500)
        self.assertAlmostEqual(m, 0.0560, delta=1e-4)
        self.assertAlmostEqual(
            d2l.min_detectable_effect(np.sqrt(0.2), 500, deff=4), 2 * m,
            places=12)
        self.assertAlmostEqual(d2l.min_detectable_effect(se=0.01), 0.0280,
                               delta=1e-4)
        self.assertAlmostEqual(d2l.min_detectable_effect(np.sqrt(0.5), 500),
                               0.0886, delta=1e-4)  # two independent samples
        self.assertIsInstance(m, float)

    def test_vectorized_and_errors(self):
        n = np.array([100, 400, 10000])
        m = d2l.min_detectable_effect(0.5, n)
        self.assertEqual(m.shape, (3,))
        np.testing.assert_allclose(m[0] / m[1], 2.0)
        with self.assertRaises(ValueError):
            d2l.min_detectable_effect(0.5)
        with self.assertRaises(ValueError):
            d2l.min_detectable_effect(0.5, 100, se=0.01)


# ------------------------------------------- tasks, ids, seeds, hash (9)
class TestTaskIdsSeeds(Fixture):
    def test_9_subset_ids(self):
        np.testing.assert_array_equal(d2l.subset_ids(1172, 500)[:100],
                                      d2l.subset_ids(1172, 100))
        np.testing.assert_array_equal(
            d2l.subset_ids(1172, 500),
            np.random.default_rng(0).permutation(1172)[:500])
        np.testing.assert_array_equal(d2l.subset_ids(7, None), np.arange(7))
        self.assertFalse(np.array_equal(d2l.subset_ids(100, 10, 1),
                                        d2l.subset_ids(100, 10, 0)))
        with self.assertRaises(ValueError):
            d2l.subset_ids(10, 11)

    def test_9_item_seed_across_processes(self):
        # the function's own source, without importing torch
        code = ('import hashlib\n' + inspect.getsource(d2l.item_seed) +
                'print(item_seed(0, "gsm8k", "17"), hash("gsm8k"))')
        outs = []
        for salt in ('1', '2'):
            env = {**os.environ, 'PYTHONHASHSEED': salt}
            outs.append(subprocess.run(
                [sys.executable, '-I', '-c', code], env=env,
                capture_output=True, text=True, check=True).stdout.split())
        self.assertEqual(outs[0][0], outs[1][0])
        self.assertNotEqual(outs[0][1], outs[1][1])  # hash() is salted
        s = d2l.item_seed(0, 'gsm8k', '17')
        self.assertEqual(s, int(outs[0][0]))
        self.assertTrue(0 <= s < 2**63)
        self.assertNotEqual(s, d2l.item_seed(1, 'gsm8k', '17'))
        self.assertNotEqual(s, d2l.item_seed(0, 'gsm8k', '18'))
        torch.Generator().manual_seed(s)  # a valid seed

    def test_eval_task_defaults(self):
        t = d2l.EvalTask('x', 'ds', 'rev', ('f',), 'mc', 'arc')
        self.assertEqual(
            (t.split, t.n, t.subset_seed, t.fmt, t.norm, t.chat, t.shots,
             t.stop, t.since), ('test', None, 0, 'cloze', 'token', False, 0,
                                (), None))
        self.assertEqual(len(t._asdict()), 23)
        json.dumps(t._asdict())  # JSON-ready

    def test_9_hash(self):
        base = d2l.EvalSuite([self.arc], n=20)
        h = base.hash
        self.assertEqual(h, d2l.EvalSuite([self.arc], n=20).hash)
        self.assertEqual(h, d2l.EvalSuite([self.arc], batch_size=3,
                                          n=20).hash)
        variants = {
            'n': dict(n=21), 'subset_seed': dict(subset_seed=1),
            'revision': dict(revision='rev1'),
            'files': dict(files=self.arc.files + ('dev-0.parquet',)),
            'norm': dict(norm='token'), 'template': dict(template='qa'),
            'chat': dict(chat=True), 'enable_thinking':
                dict(enable_thinking=False),
            'shots_from': dict(shots_from={'split': 'train', 'ids': []}),
            'stop': dict(stop=('\n',))}
        for field, change in variants.items():
            n = change.pop('n', 20)
            seed = change.get('subset_seed')
            t = self.arc._replace(**change)
            s = d2l.EvalSuite([t], n=n, seed=seed)
            self.assertNotEqual(s.hash, h, field)
        # the item ids enter the hash, not only the manifest
        s = d2l.EvalSuite([self.arc], n=20)
        s._items['arc_fx'] = s._items['arc_fx'][:-1]
        self.assertNotEqual(s.hash, h)

    def test_suite_tasks_and_items(self):
        s = d2l.EvalSuite([self.arc._asdict(), self.gsm], n=10, seed=3)
        self.assertEqual([t.id for t in s.tasks], ['arc_fx', 'gsm_fx'])
        self.assertTrue(all(t.n == 10 and t.subset_seed == 3
                            for t in s.tasks))
        ids = d2l.subset_ids(64, 10, 3)
        self.assertEqual([it['id'] for it in s.items('arc_fx')],
                         [f'Q_{i:04d}' for i in ids])
        self.assertEqual(s.tasks[1].stop, ('\n\nQ:',))
        # seed=None and n=None keep the tasks' own settings
        t = self.arc._replace(n=5, subset_seed=2)
        s2 = d2l.EvalSuite([t], n=None, seed=None)
        self.assertEqual((s2.tasks[0].n, s2.tasks[0].subset_seed), (5, 2))
        with self.assertRaisesRegex(ValueError, 'twice'):
            d2l.EvalSuite([self.arc, self.arc])

    def test_duck_typed_tasks(self):
        # a notebook redefines EvalTask: any record with _asdict() works
        Other = __import__('collections').namedtuple(
            'EvalTask', d2l.EvalTask._fields,
            defaults=d2l.EvalTask._field_defaults.values())
        other = Other(**self.arc._asdict())
        self.assertEqual(d2l.EvalSuite([other], n=10).hash,
                         d2l.EvalSuite([self.arc], n=10).hash)

    def test_manifest_round_trip(self):
        s = d2l.EvalSuite([self.arc, self.gsm], 'fx', n=12)
        path = os.path.join(self.tmp.name, 'fx.json')
        m = s.to_manifest(path)
        self.assertEqual(m['name'], 'fx')
        with open(path) as f:
            self.assertEqual(json.load(f), m)
        s2 = d2l.EvalSuite.from_manifest(path)
        self.assertEqual(s2.hash, s.hash)
        self.assertEqual(s2.tasks, s.tasks)
        self.assertEqual(s2.name, 'fx')

    def test_item_rendering(self):
        s = d2l.EvalSuite([self.arc, self.gsm], n=None, seed=None)
        it = s.items('arc_fx')[3]  # numeric labels
        row = arc_rows(64)[3]
        self.assertEqual(it['gold'], ['1', '2', '3', '4'].index(
            row['answerKey']))
        self.assertEqual(it['prompt'], 'Question: Question 3?\nAnswer:')
        self.assertEqual(it['choices'], [f' opt3-{k}' for k in range(4)])
        g = s.items('gsm_fx')[5]
        self.assertEqual(g['reference'], '10')
        self.assertEqual(g['prompt'], (
            'Q: What is 0 + 0?\nA: 0 + 0 = 0\n#### 0\n\n'
            'Q: What is 1 + 1?\nA: 1 + 1 = 2\n#### 2\n\n'
            'Q: What is 5 + 5?\nA:'))
        # a chat template without chat=True is refused
        with self.assertRaisesRegex(ValueError, 'chat=True'):
            d2l.EvalSuite([self.gsm._replace(template='chat_multi')])
        with self.assertRaisesRegex(ValueError, 'shots=3'):
            d2l.EvalSuite([self.gsm._replace(shots=3)])
        chat = self.gsm._replace(template='chat_multi', chat=True,
                                 enable_thinking=False)
        msgs = d2l.EvalSuite([chat]).items('gsm_fx')[0]['prompt']
        self.assertEqual([m['role'] for m in msgs],
                         ['user', 'assistant', 'user', 'assistant', 'user'])
        self.assertEqual(msgs[1]['content'], '0 + 0 = 0\n#### 0')


# ---------------------------------------------------------------- adapters
class TestAdapters(unittest.TestCase):
    def test_mmlu_same_subject_letter(self):
        with tempfile.TemporaryDirectory() as tmp:
            def rows(n, tag):
                return [{'question': f'{tag}{i} ', 'subject': subj,
                         'choices': ['w', 'x', 'y', 'z'], 'answer': i % 4}
                        for subj in ('astronomy', 'college_physics')
                        for i in range(n)]
            write_dataset(tmp, {'all/test-0.parquet': rows(3, 'q'),
                                'all/dev-0.parquet': rows(6, 'd')})
            t = d2l.EVAL_TASKS['mmlu']._replace(
                id='mmlu_fx', dataset=tmp, revision='r',
                files=('all/test-0.parquet', 'all/dev-0.parquet'))
            s = d2l.EvalSuite([t], n=None, seed=None)
            it = s.items('mmlu_fx')[4]  # college_physics, q1
            self.assertEqual(it['subject'], 'college_physics')
            self.assertTrue(it['prompt'].startswith(
                'The following are multiple choice questions (with answers) '
                'about college physics.\n\nQuestion: d0\nA. w\nB. x\nC. y\n'
                'D. z\nAnswer: A\n\nQuestion: d1\n'))
            self.assertEqual(it['prompt'].count('Question:'), 6)
            self.assertTrue(it['prompt'].endswith(
                'Question: q1\nA. w\nB. x\nC. y\nD. z\nAnswer:'))
            self.assertEqual(it['choices'], [' A', ' B', ' C', ' D'])
            self.assertEqual(it['gold'], 1)
            short = t._replace(shots=7)
            with self.assertRaisesRegex(ValueError, '6 of 7 shots'):
                d2l.EvalSuite([short])

    def test_row_adapters(self):
        A = d2l.EVAL_ADAPTERS
        w = A['winogrande']({'sentence': 'Ann beat Bo because _ trained.',
                             'option1': 'Ann', 'option2': 'Bo',
                             'answer': '2'}, 7)
        self.assertEqual(w, {'id': '7', 'prompts': ['Ann beat Bo because Ann',
                                                     'Ann beat Bo because Bo'],
                             'continuation': ' trained.', 'gold': 1})
        h = A['hellaswag']({'activity_label': 'Cooking', 'ctx_a': 'A man',
                            'ctx_b': 'stirs the pot [title] then',
                            'endings': ['tastes it.', 'leaves  [x]it.'],
                            'label': '0'}, 2)
        self.assertEqual(h['question'],
                         'Cooking: A man Stirs the pot. then')
        self.assertEqual(h['options'], ['tastes it.', 'leaves it.'])
        sc = A['sciq']({'question': 'Q?', 'distractor1': 'a',
                        'distractor2': 'b', 'distractor3': 'c',
                        'correct_answer': 'd', 'support': ' Text. '}, 0)
        self.assertEqual((sc['options'][sc['gold']], sc['header']),
                         ('d', 'Text.\n'))
        p = A['piqa']({'goal': 'g', 'sol1': 's1', 'sol2': 's2', 'label': 1},
                      4)
        self.assertEqual((p['options'], p['gold'], p['id']),
                         (['s1', 's2'], 1, '4'))
        he = A['humaneval']({'task_id': 'HumanEval/3', 'prompt': 'def f():\n',
                             'test': 'def check(c):\n    assert c()\n',
                             'entry_point': 'f'}, 3)
        self.assertEqual(he['id'], 'HumanEval/3')
        self.assertTrue(he['test'].endswith('\n\ncheck(f)\n'))
        g = A['gsm8k']({'question': 'q', 'answer':
                        'a <<1+2=3>>3 and 1,000\n#### 1,003'}, 11)
        self.assertEqual((g['id'], g['target'], g['reference']),
                         ('11', 'a 3 and 1,000\n#### 1,003', '1003'))
        mp = A['mmlu_pro']({'question_id': 70, 'question': 'Q ',
                            'options': ['a'] * 10, 'answer_index': 8,
                            'category': 'law'}, 0)
        self.assertEqual((mp['id'], mp['gold'], mp['subject']),
                         ('70', 8, 'law'))

    def test_registry(self):
        self.assertEqual(set(d2l.EVAL_TASKS), {
            'arc_challenge', 'hellaswag', 'mmlu', 'mmlu_pro', 'gsm8k',
            'humaneval', 'arc_easy', 'piqa', 'sciq', 'winogrande'})
        for tid, t in d2l.EVAL_TASKS.items():
            self.assertEqual(t.id, tid)
            self.assertRegex(t.revision, '^[0-9a-f]{40}$')
            self.assertIn(t.adapter, d2l.EVAL_ADAPTERS)
            self.assertIsInstance(t.files, tuple)
            for f in t.files:
                self.assertNotIn('*', f)
        self.assertEqual(d2l.EVAL_TASKS['winogrande'].kind, 'mc_partial')
        self.assertEqual(d2l.EVAL_TASKS['humaneval'].kind, 'code')

    def test_frozen_manifests(self):
        dev = {t.id: t for t in d2l.EVAL_SUITES['dev_v1']['tasks']}
        rep = {t.id: t for t in d2l.EVAL_SUITES['report_v1']['tasks']}
        self.assertEqual(list(dev), ['arc_challenge', 'gsm8k'])
        self.assertEqual(list(rep), ['mmlu_pro', 'mmlu', 'gsm8k'])
        R = d2l.EVAL_TASKS

        def diff(a, b):
            return {k for k, v in a._asdict().items() if b._asdict()[k] != v}
        self.assertEqual(diff(dev['arc_challenge'], R['arc_challenge']),
                         {'n'})
        self.assertEqual((dev['arc_challenge'].n, dev['arc_challenge'].fmt,
                          dev['arc_challenge'].norm), (500, 'cloze', 'char'))
        g = dev['gsm8k']
        self.assertEqual(diff(g, R['gsm8k']), {
            'n', 'template', 'chat', 'enable_thinking', 'stop'})
        self.assertEqual((g.n, g.template, g.chat, g.enable_thinking, g.shots,
                          g.shots_from, g.max_new_tokens, g.sampling),
                         (100, 'chat_multi', True, False, 4,
                          {'split': 'train', 'ids': [0, 1, 2, 3]}, 384, None))
        self.assertEqual(diff(rep['gsm8k'], R['gsm8k']), set())
        self.assertEqual((rep['mmlu_pro'].n, rep['mmlu'].n), (1000, 2000))
        self.assertEqual(diff(rep['mmlu'], R['mmlu']), {'n'})
        self.assertEqual(diff(rep['mmlu_pro'], R['mmlu_pro']), {'n'})

    def test_dev_v1_arc_ids_follow_the_subset_rule(self):
        # 1,172 ARC-shaped rows: the dev suite's 500 ids are the rows of
        # subset_ids(1172, 500, 0), the subset of sec_lm_scoring.
        with tempfile.TemporaryDirectory() as tmp:
            write_dataset(tmp, {'ARC-Challenge/test-00000-of-00001.parquet':
                                arc_rows(1172)})
            t = d2l.EVAL_SUITES['dev_v1']['tasks'][0]._replace(dataset=tmp)
            s = d2l.EvalSuite([t], n=None, seed=None)
            self.assertEqual([it['id'] for it in s.items('arc_challenge')],
                             [f'Q_{i:04d}' for i in d2l.subset_ids(1172, 500)])


# ------------------------------------------------ run and cache (10), kinds
class TestRun(Fixture):
    def setUp(self):
        super().setUp()
        self.calls = []

        def stub(model, tok, prompts, choices, norm='token', *,
                 batch_size=8, return_components=False):
            self.calls.append(len(prompts))
            if self.fail_at is not None and len(self.calls) == self.fail_at:
                raise KeyboardInterrupt('interrupted')
            return fake_scores(prompts, choices)
        self.fail_at = None
        patch = mock.patch.object(d2l, 'score_choices', stub)
        patch.start()
        self.addCleanup(patch.stop)

    def test_10_resume_and_misses(self):
        suite = d2l.EvalSuite([self.arc], batch_size=1, n=None)  # 64 items
        ref = suite.run(TinyLM(), None).per_item
        self.assertEqual(self.calls, [32, 32])  # a cache write per 32
        cache = os.path.join(self.tmp.name, 'cache')
        self.calls, self.fail_at = [], 2
        with self.assertRaises(KeyboardInterrupt):
            suite.run(TinyLM(), None, cache_dir=cache)
        self.calls, self.fail_at = [], None
        res = suite.run(TinyLM(), None, cache_dir=cache)
        self.assertEqual(self.calls, [32])  # only the missing half
        pd.testing.assert_frame_equal(res.per_item, ref)
        self.calls = []
        suite.run(TinyLM(), None, cache_dir=cache)
        self.assertEqual(self.calls, [])  # all cached
        suite.run(TinyLM(), None, cache_dir=cache, seed=1)
        self.assertEqual(sum(self.calls), 64)  # another seed: a miss
        self.calls = []
        d2l.EvalSuite([self.arc], batch_size=4, n=None).run(
            TinyLM(), None, cache_dir=cache)
        self.assertEqual(sum(self.calls), 64)  # another batch size: a miss
        self.calls = []
        suite.run(TinyLM(), None, cache_dir=cache, rerun=True)
        self.assertEqual(sum(self.calls), 64)
        suite.run(TinyLM(revision='r2'), None, cache_dir=cache)
        self.assertEqual(sum(self.calls), 128)  # another revision
        layout = os.path.join(cache, 'tiny--model@r1', suite.hash)
        self.assertEqual(len(os.listdir(layout)), 3)  # three run keys

    def test_mc_scores_and_meta(self):
        suite = d2l.EvalSuite([self.arc], 'fx', n=40)
        res = suite.run(TinyLM(), None)
        items = {it['id']: it for it in suite.items('arc_fx')}
        S = fake_scores([it['prompt'] for it in items.values()],
                        [it['choices'] for it in items.values()])
        expect = {i: float(int(s.argmax()) == it['gold'])
                  for (i, it), s in zip(items.items(), S)}
        df = res.per_item
        self.assertEqual(list(df.columns), ['task', 'item_id', 'cluster',
                                            'score', 'pred', 'finish_reason'])
        self.assertEqual(dict(zip(df.item_id, df.score)), expect)
        self.assertEqual(list(df.item_id), sorted(items))
        self.assertTrue((df.cluster == df.item_id).all())
        self.assertTrue(df.finish_reason.isna().all())
        m = res.meta
        self.assertEqual((m['model'], m['revision'], m['suite'],
                          m['suite_hash'], m['batch_size'], m['dtype']),
                         ('tiny/model', 'r1', 'fx', suite.hash, 8,
                          'torch.float32'))
        self.assertEqual(len(m['run_key']), 16)
        self.assertEqual(m['tasks'][0]['n'], 40)
        np.testing.assert_array_equal(res.scores('arc_fx'),
                                      df.score.to_numpy())
        self.assertEqual(list(res.scores()), ['arc_fx'])

    def test_gen_dispatch(self):
        seen = {}

        def generate(model, tok, prompts, max_new_tokens=None, **kw):
            seen.update(kw, prompts=prompts, max_new_tokens=max_new_tokens)
            last = [int(re.findall(r'\d+', p)[-1]) for p in prompts]
            return [d2l.Generation(f' so #### {2 * x}' if k % 3 else
                                   'no idea', [], None, 'stop')
                    for k, x in enumerate(last)]
        sampled = self.gsm._replace(sampling={'temperature': 0.7, 'p': 0.8,
                                              'k': 20})
        suite = d2l.EvalSuite([sampled], n=9)
        with mock.patch.object(d2l, 'generate', generate):
            res = suite.run(TinyLM(), None, seed=5)
        items = suite.items('gsm_fx')
        self.assertEqual(seen['seed'], [d2l.item_seed(5, 'gsm_fx', it['id'])
                                        for it in items])
        self.assertEqual((seen['strategy'], seen['temperature'], seen['p'],
                          seen['k'], seen['stop'], seen['chat'],
                          seen['max_new_tokens']),
                         ('sample', 0.7, 0.8, 20, ['\n\nQ:'], False, 16))
        df = res.per_item.set_index('item_id').loc[[it['id']
                                                     for it in items]]
        self.assertEqual(list(df.score), [0.0, 1.0, 1.0] * 3)
        self.assertTrue(pd.isna(df.pred.iloc[0]))  # nothing extracted
        self.assertEqual(list(df.pred[1:3]), [it['reference'] for it
                                               in items[1:3]])
        self.assertTrue((df.finish_reason == 'stop').all())

    def test_mc_partial_dispatch(self):
        rows = [{'sentence': f'Ann {i} met Bo so _ left.', 'option1': 'Ann',
                 'option2': 'Bo', 'answer': '1' if i % 2 else '2'}
                for i in range(6)]
        d = write_dataset(os.path.join(self.tmp.name, 'w'),
                          {'validation-0.parquet': rows})
        t = d2l.EVAL_TASKS['winogrande']._replace(
            dataset=d, files=('validation-0.parquet',))

        def stub(model, tok, prompts, choices, norm='token', **kw):
            self.assertEqual(norm, 'sum')
            self.assertEqual(choices, [[' left.']] * len(prompts))
            return torch.tensor([[1.0] if p.endswith('Ann') else [0.0]
                                 for p in prompts])
        with mock.patch.object(d2l, 'score_choices', stub):
            res = d2l.EvalSuite([t], n=None, seed=None).run(TinyLM(), None)
        self.assertEqual(list(res.per_item.pred), ['0'] * 6)
        self.assertEqual(list(res.per_item.score), [0.0, 1.0] * 3)

    def test_code_needs_sandbox(self):
        rows = [{'task_id': 'HumanEval/0', 'prompt': 'def f():\n',
                 'canonical_solution': '', 'entry_point': 'f',
                 'test': 'def check(c):\n    assert c() == 1\n'}]
        d = write_dataset(os.path.join(self.tmp.name, 'he'),
                          {'test-0.parquet': rows})
        t = d2l.EVAL_TASKS['humaneval']._replace(
            dataset=d, files=('test-0.parquet',))
        suite = d2l.EvalSuite([t], n=None, seed=None)
        with self.assertRaisesRegex(RuntimeError, 'sandbox'):
            suite.run(TinyLM(), None)
        programs = []

        def run_tests(program, test_code, timeout=5.0):
            programs.append((program, test_code))
            return True
        gen = [d2l.Generation('    return 1\n', [], None, 'stop')]
        with mock.patch.object(d2l, 'generate', lambda *a, **k: gen), \
                mock.patch.object(d2l, 'run_python_tests', run_tests):
            res = suite.run(TinyLM(), None, sandbox=True)
        self.assertEqual(programs, [('def f():\n    return 1\n',
                                     rows[0]['test'] + '\n\ncheck(f)\n')])
        self.assertEqual(list(res.per_item.score), [1.0])

    def test_save_load(self):
        res = d2l.EvalSuite([self.arc], n=40).run(TinyLM(), None)
        path = os.path.join(self.tmp.name, 'out', 'r.parquet')
        res.save(path)
        with open(path, 'rb') as f:
            first = f.read()
        res.save(path)
        with open(path, 'rb') as f:
            self.assertEqual(f.read(), first)  # deterministic
        with open(os.path.join(self.tmp.name, 'out', 'r.json')) as f:
            meta = json.load(f)
        self.assertNotIn('wall_clock', meta)
        back = d2l.EvalResult.load(path)
        pd.testing.assert_frame_equal(back.per_item, res.per_item)
        self.assertEqual(back.meta, meta)


# --------------------------------------------- compare, summary, macro (11)
def result(task_scores, suite_hash='h', clusters=None, seed=0):
    rows = []
    for task, scores in task_scores.items():
        for i, s in enumerate(scores):
            c = str(clusters[i]) if clusters is not None else f'{i:04d}'
            rows.append({'task': task, 'item_id': f'{i:04d}', 'cluster': c,
                         'score': float(s), 'pred': None,
                         'finish_reason': None})
    return d2l.EvalResult(pd.DataFrame(rows), {'suite_hash': suite_hash})


class TestResult(unittest.TestCase):
    def setUp(self):
        a, b = paired_binary_set(0, 300)
        self.a, self.b = a, b
        self.ra = result({'t1': a, 't2': a[::-1]})
        self.rb = result({'t1': b, 't2': b[::-1]})

    def test_11_compare(self):
        c = self.ra.compare(self.rb)
        self.assertEqual(set(c), {'t1', 't2'})
        ref = d2l.paired_bootstrap(self.a, self.b)
        self.assertEqual(c['t1'], ref)
        # shuffled frames align on (task, item_id)
        shuffled = d2l.EvalResult(self.rb.per_item.sample(frac=1,
                                                          random_state=0),
                                  self.rb.meta)
        shuffled.per_item = shuffled.per_item.sample(frac=1, random_state=1)
        self.assertEqual(self.ra.compare(shuffled)['t1'], ref)
        with self.assertRaisesRegex(ValueError, 'different suites'):
            self.ra.compare(result({'t1': self.b, 't2': self.b}, 'other'))
        fewer = result({'t1': self.b[:-1], 't2': self.b[::-1]})
        with self.assertRaisesRegex(ValueError, r"1 items.*'t1', '0299', "
                                    "'self'"):
            self.ra.compare(fewer)

    def test_compare_clusters(self):
        cl = np.arange(300) // 10  # 30 clusters
        ra = result({'t': self.a}, clusters=cl)
        rb = result({'t': self.b}, clusters=cl)
        r = ra.compare(rb)['t']
        self.assertEqual(r.n_clusters, 30)
        self.assertEqual(r, d2l.paired_bootstrap(self.a, self.b, 10000,
                                                 cl.astype(str)))
        # fewer than 30 shared clusters raise, as bootstrap_ci does
        # (lib-18.5-fix F6: they used to fall back to items with a warning)
        few = result({'t': self.a}, clusters=np.arange(300) // 20)
        few_b = result({'t': self.b}, clusters=np.arange(300) // 20)
        for call in (few.summary, few.macro, lambda: few.compare(few_b)):
            with self.assertRaisesRegex(ValueError, 'task t: 15 clusters'):
                call()

    def test_summary_macro_scores(self):
        s = self.ra.summary()
        self.assertEqual(list(s.task), ['t1', 't2'])
        self.assertAlmostEqual(s['mean'][0], self.a.mean())
        self.assertTrue((s.lo < s['mean']).all() and (s['mean'] < s.hi).all())
        m = self.ra.macro()
        self.assertAlmostEqual(m.mean, self.a.mean())
        self.assertEqual((m.n, m.n_clusters, m.n_boot), (600, 2, 10000))
        # resampling within tasks: the two identical tasks average to a
        # narrower interval than one task when their replicates differ
        self.assertLess(m.se, d2l.bootstrap_ci(self.a).se)
        scores = self.ra.scores('t1')
        np.testing.assert_array_equal(scores, self.a)


# -------------------------------------------- generate per-row seeds (12)
class FixedLogits(nn.Module):
    """Logits that depend on neither the input nor the batch."""
    def __init__(self, V=50):
        super().__init__()
        self.z = nn.Parameter(torch.randn(V, generator=torch.Generator(
        ).manual_seed(0)))

    def forward(self, X):
        return self.z.expand(*X.shape, -1)


class CharTok:
    """Characters as token ids (chr(48 + id) for ids < 50)."""
    eos_token_id = None

    def __call__(self, texts, add_special_tokens=False):
        one = isinstance(texts, str)
        ids = [[ord(c) - 48 for c in t] for t in ([texts] if one else texts)]
        return {'input_ids': ids[0] if one else ids}

    def decode(self, ids):
        return ''.join(chr(48 + i) for i in ids)


class TestGenerateSeeds(unittest.TestCase):
    def test_12_per_row_seeds(self):
        model, tok = FixedLogits(), CharTok()
        prompts = ['abc', 'abd', 'abe']
        seeds = [d2l.item_seed(0, 't', i) for i in range(3)]

        def gen(ps, ss, **kw):
            return [g.token_ids for g in d2l.generate(
                model, tok, ps, 6, strategy='sample', seed=ss, **kw)]
        together = gen(prompts, seeds)
        self.assertEqual(gen(prompts[1:2], seeds[1:2]), together[1:2])
        perm = [2, 0, 1]
        self.assertEqual(gen([prompts[i] for i in perm],
                             [seeds[i] for i in perm]),
                         [together[i] for i in perm])
        self.assertEqual(gen(prompts, seeds, batch_size=1), together)
        self.assertNotEqual(together[0], together[1])  # distinct streams


# ------------------------------------------------------ judge (13, 14)
class StubJudgeTok:
    """Characters as ids; a chat template that wraps the user message."""
    pad_token_id, eos_token_id = 0, 1

    def __init__(self):
        self.template_kwargs = []

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return ''.join(map(chr, ids))

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': self.encode(text)}

    def apply_chat_template(self, messages, tokenize=True, **kwargs):
        self.template_kwargs.append(kwargs)
        return '<u>' + messages[0]['content'] + '<a>'


QUESTION = re.compile(r'\[Question\]\n(.*?)\n\n\[Answer A\]\n(.*?)\n\n'
                      r'\[Answer B\]', re.S)


class StubJudge(nn.Module):
    """Reads each row's text back and answers with log-odds l of 'A' over
    'B': beta + delta if the question's answer a is in the first slot,
    beta - delta if it is second; P(A) + P(B) = mass."""
    V = 256

    def __init__(self, table, mass=0.9):
        super().__init__()
        self.table, self.mass, self.calls = table, mass, []
        self.p = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask, position_ids, use_cache,
                logits_to_keep):
        self.calls.append({'logits_to_keep': logits_to_keep,
                           'use_cache': use_cache,
                           'shape': tuple(input_ids.shape),
                           'pos': position_ids.clone(),
                           'mask': attention_mask.clone()})
        rows = []
        chars = input_ids.numpy().astype(np.uint8)
        for c, m in zip(chars, attention_mask.numpy().astype(bool)):
            text = c[m].tobytes().decode('latin-1')
            q, first = QUESTION.search(text).groups()
            a, beta, delta = self.table[q]
            l = beta + delta if first == a else beta - delta
            pa = self.mass / (1 + np.exp(-l))
            z = np.full(self.V, np.log((1 - self.mass) / (self.V - 2)))
            z[ord('A')], z[ord('B')] = np.log(pa), np.log(self.mass - pa)
            rows.append(z)
        out = torch.tensor(np.array(rows), dtype=torch.float32)[:, None]
        return type('Out', (), {'logits': out})


class TestJudgePairwise(unittest.TestCase):
    def setUp(self):
        self.tok = StubJudgeTok()
        rng = np.random.default_rng(0)
        self.qs = [f'q{i}' for i in range(12)]
        self.table = {q: (f'ans-a{i}' + ' x' * i, float(b), float(d))
                      for i, (q, b, d) in enumerate(zip(
                          self.qs, rng.normal(1, 0.5, 12),
                          rng.normal(0, 1, 12)))}
        self.A = [self.table[q][0] for q in self.qs]
        self.B = [f'ans-b{i}' for i in range(12)]
        self.judge = StubJudge(self.table)

    def test_13_recovery_swap_identical(self):
        res = d2l.judge_pairwise(self.judge, self.tok, self.qs, self.A,
                                 self.B)
        for q, r in zip(self.qs, res):
            _, beta, delta = self.table[q]
            self.assertAlmostEqual(r.delta, delta, delta=1e-6)
            self.assertAlmostEqual(r.position_bias, beta, delta=1e-6)
            self.assertAlmostEqual(r.pref_a, 1 / (1 + np.exp(-r.delta)),
                                   places=12)
            l1, l2 = r.logits
            self.assertAlmostEqual(l1, beta + delta, delta=1e-6)
            self.assertAlmostEqual(l2, beta - delta, delta=1e-6)
            self.assertEqual(r.verdict, 'a' if l1 > 0 > l2 else
                             'b' if l1 < 0 < l2 else 'tie')
            self.assertEqual(r.consistent, r.verdict != 'tie')
        self.assertEqual({r.verdict for r in res}, {'a', 'b', 'tie'})
        # swapping the answers negates delta exactly; the bias stays
        swapped = d2l.judge_pairwise(self.judge, self.tok, self.qs, self.B,
                                     self.A)
        for r, s in zip(res, swapped):
            self.assertEqual(s.delta, -r.delta)
            self.assertEqual(s.position_bias, r.position_bias)
        # identical answers: one prompt, delta exactly 0
        tie = d2l.judge_pairwise(self.judge, self.tok, 'q3', self.A[3],
                                 self.A[3])
        self.assertIsInstance(tie, d2l.JudgeResult)
        self.assertEqual(tie.delta, 0.0)
        self.assertEqual(tie.pref_a, 0.5)
        self.assertEqual(self.tok.template_kwargs[-1], {
            'add_generation_prompt': True, 'enable_thinking': False})

    def test_13_packing_mass_and_logits_to_keep(self):
        one = d2l.judge_pairwise(self.judge, self.tok, self.qs, self.A,
                                 self.B, batch_size=1)
        self.judge.calls = []
        packed, mass = d2l.judge_pairwise(
            self.judge, self.tok, self.qs, self.A, self.B, batch_size=8,
            max_batch_tokens=1500, return_mass=True)
        self.assertEqual(packed, one)
        np.testing.assert_allclose(mass, 0.9, atol=1e-6)
        self.assertEqual(mass.shape, (12, 2))
        for c in self.judge.calls:
            self.assertEqual(c['logits_to_keep'], 1)
            self.assertFalse(c['use_cache'])
            B, T = c['shape']
            self.assertTrue(B * T <= 1500 or B == 1)
            self.assertLessEqual(B, 8)
            # left padding: every row ends with a real token, and the
            # positions count from each row's first token
            self.assertTrue((c['mask'][:, -1] == 1).all())
            for p, m in zip(c['pos'], c['mask']):
                n = int(m.sum())
                self.assertEqual(p[-n:].tolist(), list(range(n)))
        self.assertGreater(len(self.judge.calls), 3)

    def test_13_truncation_warning(self):
        long = 'z' * 1600
        with self.assertWarnsRegex(UserWarning,
                                   'item 1: answer_b cut from 1600 to 1500'):
            d2l.judge_pairwise(StubJudge({**self.table, 'q1': (
                self.A[1], 0.0, 0.0)}), self.tok, self.qs[:2], self.A[:2],
                ['short', long])

    def test_errors(self):
        class BadTok(StubJudgeTok):
            def encode(self, text, add_special_tokens=False):
                return [1, 2] if text == 'A' else [3]
        with self.assertRaisesRegex(ValueError, 'single tokens'):
            d2l.judge_pairwise(self.judge, BadTok(), 'q0', 'a', 'b')
        with self.assertRaises(ValueError):
            d2l.judge_pairwise(self.judge, self.tok, ['q0'], ['a'], [])
        self.assertEqual(d2l.judge_pairwise(self.judge, self.tok, [], [], []),
                         [])

    def test_14_debiasing_under_noise(self):
        # 5,000 prompts, delta ~ N(0.3, 1), beta = 1, noise N(0, 0.3^2) in
        # each order, run through judge_pairwise with a stub judge.
        for gamma in (0.0, 0.3):
            rng = np.random.default_rng(1)
            n = 5000
            delta = rng.normal(0.3, 1, n)
            e1, e2 = rng.normal(0, 0.3, (2, n))
            l1 = 1 + (1 + gamma) * delta + e1
            l2 = 1 - (1 - gamma) * delta + e2
            qs = [f'p{i}' for i in range(n)]
            table = {q: (f'a{i}', (x + y) / 2, (x - y) / 2)
                     for i, (q, x, y) in enumerate(zip(qs, l1, l2))}
            res = d2l.judge_pairwise(
                StubJudge(table), self.tok, qs, [f'a{i}' for i in range(n)],
                [f'b{i}' for i in range(n)], batch_size=256)
            d_hat = np.array([r.delta for r in res])
            b_hat = np.array([r.position_bias for r in res])
            o = np.array([r.logits for r in res])
            if gamma == 0:
                self.assertLess(abs(b_hat.mean() - 1.0), 0.05)
                self.assertLess(abs((d_hat - delta).mean()), 0.05)
                self.assertGreater(np.corrcoef(d_hat, delta)[0, 1], 0.95)
            # total least squares: slope of -l2 on l1 from the leading
            # eigenvector of the 2 by 2 covariance
            vals, vecs = np.linalg.eigh(np.cov(o[:, 0], -o[:, 1]))
            v = vecs[:, -1]
            self.assertLess(abs(v[1] / v[0] - (1 - gamma) / (1 + gamma)),
                            0.05)


class TestJudgeTokenizers(unittest.TestCase):
    """The real Qwen3 and Qwen3.5 chat templates and a tiny random Qwen3
    behind them: single-token letters, the empty think block, and packed
    batches equal to one prompt at a time."""
    @staticmethod
    def tokenizer(model_id):
        try:
            from transformers import AutoTokenizer
        except ImportError as e:
            raise unittest.SkipTest(f'transformers not installed: {e}')
        try:
            return AutoTokenizer.from_pretrained(
                model_id, revision=d2l.HF_REVISIONS[model_id],
                local_files_only=True)
        except OSError:
            raise unittest.SkipTest(f'{model_id} tokenizer not in the HF '
                                    'cache')

    def test_letters_and_template(self):
        for model_id in ('Qwen/Qwen3-0.6B', 'Qwen/Qwen3.5-0.8B'):
            tok = self.tokenizer(model_id)
            ids = [tok.encode(c, add_special_tokens=False) for c in 'AB']
            self.assertTrue(all(len(i) == 1 for i in ids), model_id)
            text = tok.apply_chat_template(
                [{'role': 'user', 'content': 'x'}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            self.assertTrue(text.endswith('<think>\n\n</think>\n\n'),
                            (model_id, text[-40:]))

    def test_tiny_qwen3_packing(self):
        import transformers
        tok = self.tokenizer('Qwen/Qwen3-0.6B')
        torch.manual_seed(0)
        model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
            vocab_size=151936, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            num_hidden_layers=2, tie_word_embeddings=True,
            attn_implementation='sdpa')).eval()
        with torch.no_grad():
            for name, p in model.named_parameters():
                p.copy_(1 + 0.3 * torch.randn_like(p) if 'norm' in name
                        else 0.08 * torch.randn_like(p))
        qs = ['What is 2 + 2?', 'Name a colour.', 'Why is the sky blue?']
        A = ['4', 'Blue, as in the sky on a clear day.', 'Rayleigh scattering'
             ' of sunlight by air molecules.']
        B = ['Five.', 'Red.', 'Because of the sea.']
        one, m1 = d2l.judge_pairwise(model, tok, qs, A, B, batch_size=1,
                                     return_mass=True)
        packed, m2 = d2l.judge_pairwise(model, tok, qs, A, B, batch_size=6,
                                        max_batch_tokens=10**4,
                                        return_mass=True)
        for r, s in zip(one, packed):
            np.testing.assert_allclose(r.logits, s.logits, atol=1e-5)
        np.testing.assert_allclose(m1, m2, atol=1e-6)
        same = d2l.judge_pairwise(model, tok, qs[0], A[0], A[0])
        self.assertEqual(same.delta, 0.0)


# ------------------------------------------------------ Bradley-Terry (15)
class TestBradleyTerry(unittest.TestCase):
    def test_15_recovery(self):
        # 20,000 comparisons of 4 items; over 20 data seeds the largest
        # error was 0.034.
        true = np.array([0.6, 0.2, -0.1, -0.7])
        rng = np.random.default_rng(0)
        i = rng.integers(0, 4, 20000)
        j = (i + rng.integers(1, 4, 20000)) % 4
        win = rng.random(20000) < 1 / (1 + np.exp(-(true[i] - true[j])))
        s = d2l.bradley_terry(np.where(win, i, j), np.where(win, j, i), 4)
        self.assertEqual(s.shape, (4,))
        self.assertAlmostEqual(s.sum(), 0.0, places=10)
        np.testing.assert_allclose(s, true - true.mean(), atol=0.05)

    def test_15_disconnected_and_separated(self):
        s = d2l.bradley_terry([0, 0, 1, 2], [1, 1, 0, 3], 4)
        self.assertTrue(np.isfinite(s).all())
        self.assertGreater(s[0], s[1])
        self.assertGreater(s[2], s[3])
        s = d2l.bradley_terry([0] * 50, [1] * 50, 3)  # 0 always wins
        self.assertTrue(np.isfinite(s).all())
        self.assertGreater(s[0], s[2])
        self.assertGreater(s[2], s[1])

    def test_15_batched_weights(self):
        rng = np.random.default_rng(1)
        i = rng.integers(0, 4, 3000)
        j = (i + rng.integers(1, 4, 3000)) % 4
        W = rng.multinomial(3000, np.full(3000, 1 / 3000), size=6).astype(
            float)
        S = d2l.bradley_terry(i, j, 4, weights=W)
        self.assertEqual(S.shape, (6, 4))
        for b in range(6):
            np.testing.assert_allclose(
                S[b], d2l.bradley_terry(i, j, 4, weights=W[b]), atol=1e-8)
        # half-weighted records (ties) count half
        half = d2l.bradley_terry([0, 1], [1, 0], 2, weights=[0.5, 0.5])
        np.testing.assert_allclose(half, 0.0, atol=1e-12)


# ------------------------------------------------------- library imports
class TestImportWithoutTransformers(unittest.TestCase):
    def test_blocked(self):
        code = '''
import sys
for m in ('transformers', 'openai', 'accelerate', 'huggingface_hub',
          'tokenizers', 'safetensors'):
    sys.modules[m] = None
import numpy as np
from d2l import torch as d2l
r = d2l.paired_bootstrap(np.ones(200), np.zeros(200))
assert r.mean == 1.0 and r.p_two_sided == 0.0
d2l.bradley_terry([0], [1], 2)
d2l.min_detectable_effect(se=0.01)
print('ok', len(d2l.EVAL_TASKS))
'''
        out = subprocess.run([sys.executable, '-c', code], cwd=REPO,
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split(), ['ok', '10'])


# --------------------------------------------------- registry smoke (17)
@unittest.skipUnless(NET, 'network test: set D2L_NET_TESTS=1')
class TestRegistrySmoke(unittest.TestCase):
    def test_17_every_task(self):
        def scores(model, tok, prompts, choices, norm='token', **kw):
            return torch.zeros(len(prompts), max(map(len, choices)))
        gens = lambda model, tok, prompts, *a, **k: [  # noqa: E731
            d2l.Generation(' #### 1', [], None, 'eos') for _ in prompts]
        for tid, task in d2l.EVAL_TASKS.items():
            with self.subTest(task=tid):
                suite = quiet(d2l.EvalSuite, [tid], n=5)
                items = suite.items(tid)
                self.assertEqual(len(items), 5)
                with mock.patch.object(d2l, 'score_choices', scores), \
                        mock.patch.object(d2l, 'generate', gens), \
                        mock.patch.object(d2l, 'run_python_tests',
                                          lambda *a, **k: False):
                    if task.kind == 'code':
                        with self.assertRaises(RuntimeError):
                            suite.run(TinyLM(), None)
                        res = suite.run(TinyLM(), None, sandbox=True)
                    else:
                        res = suite.run(TinyLM(), None)
                self.assertEqual(len(res.per_item), 5)
                if task.kind == 'mc':
                    self.assertTrue(all(it['prompt'] and it['choices']
                                        for it in items))
                if tid == 'winogrande':
                    self.assertTrue(all(len(it['prompts']) == 2 and
                                        it['continuation'] for it in items))
                if tid == 'mmlu':
                    self.assertEqual(items[0]['prompt'].count('Answer:'), 6)
        # the dev suite's ARC ids are those of sec_lm_scoring
        dev = quiet(d2l.EvalSuite.from_manifest, 'dev_v1')
        rows = d2l._eval_load_rows(dev.tasks[0])
        self.assertEqual(len(rows), 1172)
        self.assertEqual([it['id'] for it in dev.items('arc_challenge')],
                         [rows[i]['id'] for i in d2l.subset_ids(1172, 500)])
        self.assertEqual(len(dev.items('gsm8k')), 100)
        print(f"\ndev_v1 hash {dev.hash}")



# ------------------------------------- review fixes (stage 2a, lib-18.5-fix)
class TestReviewFixes(Fixture):
    """Regression tests for the review of the 18.5 library (F1-F9)."""
    def stub_scores(self):
        return mock.patch.object(
            d2l, 'score_choices',
            lambda m, t, p, c, *a, **k: fake_scores(p, c))

    def test_f1_stop_and_files_as_strings(self):
        gsm = self.gsm._replace(stop='\n\nQ:')
        suite = d2l.EvalSuite([gsm], n=3)
        self.assertEqual(suite.tasks[0].stop, ('\n\nQ:',))
        self.assertEqual(suite.hash, d2l.EvalSuite([self.gsm], n=3).hash)
        seen = {}

        def generate(model, tok, prompts, *a, **kw):
            seen.update(kw)
            return [d2l.Generation('#### 2', [], None, 'eos')] * len(prompts)
        with mock.patch.object(d2l, 'generate', generate):
            suite.run(TinyLM(), None)
        self.assertEqual(seen['stop'], ['\n\nQ:'])
        arc = self.arc._replace(files='test-0.parquet')
        s = d2l.EvalSuite([arc], n=5)
        self.assertEqual(s.tasks[0].files, ('test-0.parquet',))
        self.assertEqual(len(s.items('arc_fx')), 5)
        self.assertEqual(d2l.EvalSuite([self.arc._replace(stop=None)],
                                       n=5).tasks[0].stop, ())

    def test_f2_hash_and_cache_cover_rendered_prompts(self):
        cache = os.path.join(self.tmp.name, 'cache')
        orig = d2l.format_mc

        def changed(q, opts, fmt='cloze', *, shots=()):
            p, c = orig(q, opts, fmt, shots=shots)
            return 'Q: ' + p, c
        with self.stub_scores():
            s1 = d2l.EvalSuite([self.arc], n=40)
            r1 = s1.run(TinyLM(), None, cache_dir=cache)
            with mock.patch.object(d2l, 'format_mc', changed):
                s2 = d2l.EvalSuite([self.arc], n=40)
                r2 = s2.run(TinyLM(), None, cache_dir=cache)
                fresh = s2.run(TinyLM(), None)
        self.assertEqual(s1.tasks, s2.tasks)  # same manifest
        self.assertNotEqual(s1.hash, s2.hash)  # other prompts
        self.assertNotEqual(r1.meta['suite_hash'], r2.meta['suite_hash'])
        np.testing.assert_array_equal(r2.scores('arc_fx'),
                                      fresh.scores('arc_fx'))
        self.assertFalse(np.array_equal(r1.scores('arc_fx'),
                                        r2.scores('arc_fx')))
        # an item's reference (not only its prompt) enters the hash
        s3 = d2l.EvalSuite([self.gsm], n=4)
        h = s3.hash
        s3._items['gsm_fx'][0]['reference'] = '-1'
        self.assertNotEqual(s3.hash, h)

    def test_f3_notebook_classes_have_library_methods(self):
        with open(os.path.join(REPO, 'chapter_lm-pretrained',
                               'eval-suites.md'), encoding='utf-8') as f:
            src = f.read()
        cells = re.findall(r'```\{\.python \.input #([\w-]+)\}\n'
                           r'%%tab pytorch\n(.*?)```', src, re.S)
        self.assertIn('eval-suites-attach', [c for c, _ in cells])
        ns = {}
        with mock.patch('sys.stdout'):
            for _, code in cells:
                exec(code, ns)
        EvalSuite, EvalResult = ns['EvalSuite'], ns['EvalResult']
        self.assertIsNot(EvalSuite, d2l.EvalSuite)
        path = os.path.join(self.tmp.name, 'fx.json')
        EvalSuite([self.arc], 'fx', n=12).to_manifest(path)
        suite = EvalSuite.from_manifest(path)
        self.assertIs(type(suite), EvalSuite)
        with self.stub_scores():
            res = suite.run(TinyLM(), None)
        self.assertIs(type(res), EvalResult)
        out = os.path.join(self.tmp.name, 'r.parquet')
        res.save(out)
        back = EvalResult.load(out)
        self.assertIs(type(back), EvalResult)
        pd.testing.assert_frame_equal(back.per_item, res.per_item)
        self.assertIs(type(d2l.EvalResult.load(out)), d2l.EvalResult)

    def test_f5_hellaswag_continuation_format(self):
        def rows(n):
            return [{'activity_label': 'Cooking', 'ctx_a': f'A man {i}',
                     'ctx_b': 'stirs the pot', 'endings': [
                         f'tastes it {k}.' for k in range(4)],
                     'label': str(i % 4)} for i in range(n)]
        d = write_dataset(os.path.join(self.tmp.name, 'hs'), {
            'validation-0.parquet': rows(5), 'train-0.parquet': rows(2)})
        t = d2l.EVAL_TASKS['hellaswag']._replace(
            dataset=d, files=('validation-0.parquet', 'train-0.parquet'))
        self.assertEqual(t.fmt, 'continuation')
        it = d2l.EvalSuite([t], n=None, seed=None).items('hellaswag')[3]
        # lm-evaluation-harness: the context, then ' ' + each ending
        self.assertEqual(it['prompt'], 'Cooking: A man 3 Stirs the pot')
        self.assertEqual(it['choices'], [f' tastes it {k}.'
                                         for k in range(4)])
        shot = t._replace(shots=1, shots_from={'split': 'train', 'ids': [1]})
        it = d2l.EvalSuite([shot], n=None, seed=None).items('hellaswag')[3]
        self.assertEqual(it['prompt'], 'Cooking: A man 1 Stirs the pot '
                         'tastes it 1.\n\nCooking: A man 3 Stirs the pot')

    def test_f7_numpy_values(self):
        s = d2l.EvalSuite([self.arc], n=np.int64(20), seed=np.int64(0))
        self.assertEqual(s.hash, d2l.EvalSuite([self.arc], n=20).hash)
        self.assertIs(type(s.tasks[0].n), int)
        g = self.gsm._replace(
            sampling={'temperature': np.float32(0.5)},
            shots_from={'split': 'train', 'ids': np.array([0, 1])})
        s = d2l.EvalSuite([g], n=3)
        self.assertEqual(s.tasks[0].shots_from['ids'], [0, 1])
        self.assertIs(type(s.tasks[0].sampling['temperature']), float)
        json.dumps(s.to_manifest())
        with self.assertRaisesRegex(TypeError, 'is not JSON'):
            d2l.EvalSuite([self.arc._replace(since=object())])

    def test_f8_sampling_keys_that_run_sets(self):
        for key in ('seed', 'stop', 'max_new_tokens', 'batch_size'):
            g = self.gsm._replace(sampling={'temperature': 0.7, key: 1})
            with self.assertRaisesRegex(ValueError, f"sampling sets.*{key}"):
                d2l.EvalSuite([g], n=3)

    def test_f9_mde_se_with_deff_array(self):
        m = d2l.min_detectable_effect(se=0.01, deff=np.array([1., 4.]))
        np.testing.assert_allclose(m, [0.0280, 0.0560], atol=1e-4)
        np.testing.assert_allclose(
            d2l.min_detectable_effect(0.5, 500, deff=np.array([1., 4.])),
            d2l.min_detectable_effect(0.5, np.array([500, 125])))

    def test_f9_mc_partial_uses_task_norm(self):
        rows = [{'sentence': 'Ann met Bo so _ left.', 'option1': 'Ann',
                 'option2': 'Bo', 'answer': '1'}] * 2
        d = write_dataset(os.path.join(self.tmp.name, 'w'),
                          {'validation-0.parquet': rows})
        t = d2l.EVAL_TASKS['winogrande']._replace(
            dataset=d, files=('validation-0.parquet',), norm='token')
        norms = []

        def stub(model, tok, prompts, choices, norm='sum', **kw):
            norms.append(norm)
            return torch.zeros(len(prompts), 1)
        with mock.patch.object(d2l, 'score_choices', stub):
            d2l.EvalSuite([t], n=None, seed=None).run(TinyLM(), None)
        self.assertEqual(norms, ['token'])


class BF16Judge(nn.Module):
    """A bf16 judge with an output head: the last hidden state is (1, 1, 0)
    when the question's answer a sits in the first slot, else (1, 1, 1).
    The head gives A 24.3125 and B 24.25 for the first, A 24.3125 and B
    24.375 for the second: log-odds +0.0625 and -0.0625, which bf16 logits
    round to 0 and -0.125 (steps of 0.125 between 16 and 32). With
    softcap, the logits are 30 tanh(z / 30), which the head does not
    give."""
    def __init__(self, answer_a, softcap=False):
        super().__init__()
        self.answer_a, self.softcap = answer_a, softcap
        self.lm_head = nn.Linear(3, 256, bias=False, dtype=torch.bfloat16)
        with torch.no_grad():
            self.lm_head.weight.zero_()
            self.lm_head.weight[ord('A')] = torch.tensor([16, 8.3125, 0])
            self.lm_head.weight[ord('B')] = torch.tensor([16, 8.25, 0.125])

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, attention_mask, position_ids, use_cache,
                logits_to_keep):
        h = []
        for c, m in zip(input_ids.numpy().astype(np.uint8),
                        attention_mask.numpy().astype(bool)):
            first = QUESTION.search(c[m].tobytes().decode('latin-1'))[2]
            h.append([1., 1., 0. if first == self.answer_a else 1.])
        h = torch.tensor(h, dtype=torch.bfloat16)[:, None]
        z = self.lm_head(h[:, -logits_to_keep:])
        if self.softcap:
            z = 30 * torch.tanh(z / 30)
        return type('Out', (), {'logits': z})


class TestJudgeFP32LogOdds(unittest.TestCase):
    """F4: the restricted log-odds come from fp32, not from bf16 logits."""
    def test_bf16_rounding(self):
        tok = StubJudgeTok()
        judge = BF16Judge('aaa')
        z = judge.lm_head(torch.tensor([[1., 1., 0.], [1., 1., 1.]],
                                       dtype=torch.bfloat16))
        self.assertEqual(z[:, [65, 66]].float().tolist(),
                         [[24.25, 24.25], [24.25, 24.375]])  # rounded
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            r, mass = d2l.judge_pairwise(judge, tok, 'q', 'aaa', 'bb',
                                         return_mass=True)
        self.assertEqual(r.logits, (0.0625, -0.0625))
        self.assertEqual((r.delta, r.position_bias, r.verdict),
                         (0.0625, 0.0, 'a'))
        self.assertGreater(mass.min(), 0.99)
        self.assertEqual(judge.lm_head._forward_hooks, {})  # removed
        # swapping the answers negates the log-odds
        s = d2l.judge_pairwise(judge, tok, 'q', 'bb', 'aaa')
        self.assertEqual((s.delta, s.verdict), (-0.0625, 'b'))

    def test_head_that_does_not_give_the_logits(self):
        tok = StubJudgeTok()
        with self.assertWarnsRegex(UserWarning, 'output head does not'):
            r = d2l.judge_pairwise(BF16Judge('aaa', softcap=True), tok, 'q',
                                   'aaa', 'bb')
        z = 30 * torch.tanh(torch.tensor([24.25, 24.25, 24.25, 24.375],
                                         dtype=torch.bfloat16) / 30)
        self.assertEqual(r.logits, (float(z[0]) - float(z[1]),
                                    float(z[2]) - float(z[3])))

    def test_tiny_qwen3_bf16(self):
        import transformers
        tok = TestJudgeTokenizers.tokenizer('Qwen/Qwen3-0.6B')
        torch.manual_seed(0)
        model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(
            vocab_size=151936, hidden_size=64, intermediate_size=128,
            num_attention_heads=4, num_key_value_heads=2, head_dim=16,
            num_hidden_layers=2, tie_word_embeddings=True,
            attn_implementation='sdpa')).eval()
        with torch.no_grad():
            for p in model.parameters():
                p.normal_(0, 0.5)
            model.model.norm.weight.fill_(5.0)  # logits of order 10 to 30
        model = model.to(torch.bfloat16)
        q, a, b = 'What is 2 + 2?', '4', 'Five.'
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            r = d2l.judge_pairwise(model, tok, q, a, b)
        AB = [tok.encode(c, add_special_tokens=False)[0] for c in 'AB']
        expect, rounded = [], []
        for first, second in ((a, b), (b, a)):
            text = tok.apply_chat_template(
                [{'role': 'user', 'content': d2l.JUDGE_RUBRIC_V1.format(
                    question=q, first=first, second=second)}],
                tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            X = torch.tensor([tok(text, add_special_tokens=False)
                              ['input_ids']])
            with torch.no_grad():
                h = model.model(input_ids=X).last_hidden_state[0, -1]
                zab = h.float() @ model.lm_head.weight[AB].float().T
                bf = model(input_ids=X).logits[0, -1, AB].float()
            expect.append(float(zab[0] - zab[1]))
            rounded.append(float(bf[0] - bf[1]))
            self.assertGreater(float(bf.abs().min()), 8)
        np.testing.assert_allclose(r.logits, expect, rtol=1e-5, atol=1e-5)
        # the bf16 logits would have given other log-odds
        self.assertGreater(np.abs(np.subtract(expect, rounded)).max(), 1e-3)


if __name__ == '__main__':
    unittest.main()
